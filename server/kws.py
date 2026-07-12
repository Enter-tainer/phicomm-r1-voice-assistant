# -*- coding: utf-8 -*-
"""Server-side Keyword Spotting using Sherpa-onnx.

Detects wake word "你好小讯" from continuous audio stream.
Runs on the server (x86_64 Linux), not on the R1 device.
"""

import numpy as np
import logging
import os
import sherpa_onnx

logger = logging.getLogger("r1voice.kws")

# Model paths
KWS_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "kws",
                              "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20")
KEYWORDS_FILE = os.path.join(KWS_MODEL_DIR, "keywords.txt")


class ServerKWS:
    """Server-side keyword spotter using Sherpa-onnx.

    Feed it 16kHz 16-bit mono PCM frames.
    Returns True when wake word is detected.
    """

    def __init__(self):
        encoder = os.path.join(KWS_MODEL_DIR, "encoder-epoch-13-avg-2-chunk-16-left-64.int8.onnx")
        decoder = os.path.join(KWS_MODEL_DIR, "decoder-epoch-13-avg-2-chunk-16-left-64.onnx")
        joiner = os.path.join(KWS_MODEL_DIR, "joiner-epoch-13-avg-2-chunk-16-left-64.int8.onnx")
        tokens = os.path.join(KWS_MODEL_DIR, "tokens.txt")

        logger.info(f"Initializing Sherpa KWS:")
        logger.info(f"  encoder={encoder}")
        logger.info(f"  keywords={KEYWORDS_FILE}")

        self.kws = sherpa_onnx.KeywordSpotter(
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            tokens=tokens,
            keywords_file=KEYWORDS_FILE,
            num_threads=1,
            keywords_score=1.0,
            keywords_threshold=0.1,
            max_active_paths=4,
            num_trailing_blanks=3,
        )

        self.stream = self.kws.create_stream()
        self._samples_buffer = np.array([], dtype=np.float32)
        logger.info("Sherpa KWS initialized (你好小讯)")

    def process_frame(self, pcm_bytes: bytes) -> bool:
        """Process a PCM frame. Returns True if wake word detected.

        Args:
            pcm_bytes: 16kHz 16-bit mono PCM bytes

        Returns:
            True if wake word detected, False otherwise
        """
        # Convert to float32 [-1, 1]
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self._samples_buffer = np.concatenate([self._samples_buffer, samples])

        # Feed to KWS stream (Python API: sample_rate first, then samples)
        self.stream.accept_waveform(16000, samples)

        # Decode
        while self.kws.is_ready(self.stream):
            self.kws.decode(self.stream)

        # Check result
        result = self.kws.get_result(self.stream)
        if result and result.keyword and result.keyword.strip():
            logger.info(f"KWS detected: {result.keyword}")
            # Reset stream after detection
            self.kws.reset(self.stream)
            return True

        return False

    def reset(self):
        """Reset KWS state."""
        self.kws.reset(self.stream)
        self._samples_buffer = np.array([], dtype=np.float32)
