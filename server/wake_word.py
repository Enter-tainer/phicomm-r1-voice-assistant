# -*- coding: utf-8 -*-
"""Server-side wake word detection using openWakeWord."""

import logging
import numpy as np
import openwakeword
from openwakeword.model import Model

logger = logging.getLogger("r1voice.wake_word")


class WakeWordDetector:
    """Wraps openWakeWord for server-side wake word detection."""

    def __init__(self, model_name="hey_jarvis", inference_framework="onnx"):
        logger.info(f"Loading openWakeWord model: {model_name} ({inference_framework})")
        self.model = Model(
            wakeword_models=[model_name],
            inference_framework=inference_framework,
        )
        self.prediction_count = 0
        logger.info("Wake word model loaded")

    def predict(self, audio_samples: np.ndarray) -> float:
        """Predict wake word score for a chunk of audio.

        Args:
            audio_samples: float32 numpy array, ideally 1280 samples (80ms)

        Returns:
            Score between 0 and 1
        """
        if len(audio_samples) < 1280:
            # Pad to 1280
            audio_samples = np.pad(audio_samples, (0, 1280 - len(audio_samples)))

        prediction = self.model.predict(audio_samples)
        self.prediction_count += 1

        # Model returns dict like {"hey_jarvis": 0.5}
        for name, score in prediction.items():
            return float(score)

        return 0.0

    def reset(self):
        """Reset the model's prediction buffer."""
        self.model.reset()
        self.prediction_count = 0
        logger.info("Wake word model reset")
