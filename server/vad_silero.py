# -*- coding: utf-8 -*-
"""Silero VAD wrapper for R1 Voice Server.

Uses the silero-vad Python package directly (PyTorch JIT model).
VADIterator returns {'start': sample} / {'end': sample} events.

We wrap it to match the existing process_frame(pcm_bytes) → str interface.
"""

import torch
import numpy as np
import logging
import config
from silero_vad import VADIterator, load_silero_vad

logger = logging.getLogger("r1voice.vad")

SILERO_FRAME_SIZE = 512  # 32ms at 16kHz — Silero's required chunk size


class SileroVAD:
    """Wraps silero-vad's VADIterator to match our process_frame interface.

    Feed it 16kHz 16-bit mono PCM frames (any size — we buffer internally).
    Returns 'speech_start', 'speech', 'silence', or 'speech_end'.
    """

    def __init__(self, threshold=None, silence_frames=None, min_speech_frames=None):
        self.threshold = threshold or 0.3
        self.silence_frames = silence_frames or config.VAD_SILENCE_FRAMES
        self.min_speech_frames = min_speech_frames or config.VAD_MIN_SPEECH_FRAMES

        # Load model (PyTorch JIT, not ONNX — simpler, no extra deps)
        model = load_silero_vad(onnx=False)
        self.vad = VADIterator(
            model,
            threshold=self.threshold,
            sampling_rate=16000,
            min_silence_duration_ms=1500,  # 1.5s silence to end speech (was 500ms — too aggressive)
            speech_pad_ms=30,
        )
        logger.info("Silero VAD initialized (PyTorch JIT)")

        # Internal buffer for accumulating samples to 512-sample chunks
        self._buffer = np.array([], dtype=np.float32)
        self._in_speech = False
        self._grace_frames = 0  # Frames to skip VAD after wake (avoid wake word echo)

    def process_frame(self, pcm_bytes: bytes) -> str:
        """Process a PCM frame. Returns state string."""
        # Convert 16-bit PCM bytes → float32 [-1, 1]
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self._buffer = np.concatenate([self._buffer, samples])

        # Grace period: skip VAD for first N frames after wake
        # to avoid detecting the wake word echo or ding sound as speech
        if self._grace_frames > 0:
            # Consume buffer during grace period
            while len(self._buffer) >= SILERO_FRAME_SIZE:
                self._buffer = self._buffer[SILERO_FRAME_SIZE:]
            self._grace_frames -= 1
            return "silence"

        results = []

        # Process all complete 512-sample chunks
        while len(self._buffer) >= SILERO_FRAME_SIZE:
            chunk = self._buffer[:SILERO_FRAME_SIZE]
            self._buffer = self._buffer[SILERO_FRAME_SIZE:]

            event = self.vad(torch.from_numpy(chunk), return_seconds=False)
            if event is not None:
                results.append(event)

        # Map events to state strings
        had_start = any("start" in e for e in results)
        had_end = any("end" in e for e in results)

        if had_start and had_end:
            # Both in one frame — very short utterance
            self._in_speech = False
            return "speech_end"
        elif had_start:
            self._in_speech = True
            return "speech_start"
        elif had_end:
            self._in_speech = False
            return "speech_end"
        elif self._in_speech:
            return "speech"
        else:
            return "silence"

    def reset(self):
        """Reset VAD state."""
        self.vad.reset_states()
        self._buffer = np.array([], dtype=np.float32)
        self._in_speech = False
        self._grace_frames = 25  # Skip first ~500ms after wake (25 × 20ms frames)
