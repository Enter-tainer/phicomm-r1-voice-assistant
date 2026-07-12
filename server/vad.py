# -*- coding: utf-8 -*-
"""Energy-based VAD for R1 Voice Server.

Phase 1: simple RMS energy detection.
Phase 2 (TODO): upgrade to Silero VAD ONNX.
"""

import numpy as np
import config


class EnergyVAD:
    """Simple energy-based voice activity detector."""

    def __init__(self, threshold=None, silence_frames=None, min_speech_frames=None):
        self.threshold = threshold or config.VAD_ENERGY_THRESHOLD
        self.silence_frames = silence_frames or config.VAD_SILENCE_FRAMES
        self.min_speech_frames = min_speech_frames or config.VAD_MIN_SPEECH_FRAMES

        self._speech_frame_count = 0
        self._silence_frame_count = 0
        self._in_speech = False

    def process_frame(self, pcm_bytes: bytes) -> str:
        """Process a PCM frame. Returns 'speech', 'silence', 'speech_end', or 'speech_start'.

        Args:
            pcm_bytes: raw 16-bit PCM samples (little-endian)

        Returns:
            'speech_start' - transition from silence to speech
            'speech' - continuing speech
            'silence' - continuing silence
            'speech_end' - transition from speech to silence (after silence_frames consecutive silent frames)
        """
        # Convert bytes to int16 array
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)

        # Calculate RMS energy
        rms = np.sqrt(np.mean(samples ** 2)) if len(samples) > 0 else 0.0

        is_loud = rms > self.threshold

        if is_loud:
            self._silence_frame_count = 0
            self._speech_frame_count += 1

            if not self._in_speech and self._speech_frame_count >= self.min_speech_frames:
                self._in_speech = True
                return "speech_start"
            elif self._in_speech:
                return "speech"
            else:
                return "silence"  # not enough frames yet
        else:
            self._speech_frame_count = 0
            self._silence_frame_count += 1

            if self._in_speech and self._silence_frame_count >= self.silence_frames:
                self._in_speech = False
                return "speech_end"
            elif self._in_speech:
                return "speech"  # still in speech, just a brief pause
            else:
                return "silence"

    def reset(self):
        """Reset VAD state."""
        self._speech_frame_count = 0
        self._silence_frame_count = 0
        self._in_speech = False
