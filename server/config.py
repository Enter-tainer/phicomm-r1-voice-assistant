# -*- coding: utf-8 -*-
"""Configuration for R1 Voice Server."""

import os

# WebSocket server
WS_HOST = "0.0.0.0"
WS_PORT = 8090

# Server IP (for logging)
SERVER_IP = "192.168.1.120"

# Hermes API (OpenAI-compatible)
HERMES_BASE = "http://localhost:8642/v1"
HERMES_API_KEY = os.environ.get("R1_HERMES_API_KEY", "")
HERMES_MODEL = "glm-5.2-fast"

# Auto-read API key from Hermes .env if not set via env
if not HERMES_API_KEY:
    try:
        env_path = os.path.expanduser("~/.hermes/.env")
        with open(env_path) as f:
            for line in f:
                if line.startswith("API_SERVER_KEY="):
                    HERMES_API_KEY = line.strip().split("=", 1)[1]
                    break
    except Exception:
        pass

# ASR (OpenAI-compatible, Qwen3-ASR)
ASR_BASE = "http://localhost:8083/v1"

# TTS (Edge TTS, no GPU)
TTS_VOICE = "zh-CN-XiaoxiaoNeural"
TTS_RATE = "+0%"
TTS_VOLUME = "+0%"

# Audio format
# Input (from R1): 16kHz 16bit mono PCM
INPUT_SAMPLE_RATE = 16000
INPUT_SAMPLE_SIZE = 2  # 16-bit
INPUT_CHANNELS = 1
INPUT_FRAME_MS = 20  # 20ms frames
INPUT_FRAME_BYTES = INPUT_SAMPLE_RATE * INPUT_SAMPLE_SIZE * INPUT_CHANNELS * INPUT_FRAME_MS // 1000  # 640 bytes

# Output (to R1): 48kHz 16bit mono PCM (from edge-tts, converted)
OUTPUT_SAMPLE_RATE = 48000
OUTPUT_SAMPLE_SIZE = 2
OUTPUT_CHANNELS = 1
OUTPUT_CHUNK_MS = 20  # 20ms chunks for streaming
OUTPUT_CHUNK_BYTES = OUTPUT_SAMPLE_RATE * OUTPUT_SAMPLE_SIZE * OUTPUT_CHANNELS * OUTPUT_CHUNK_MS // 1000  # 1920 bytes

# Wake word (openWakeWord, server-side)
WAKE_WORD_MODEL = "hey_jarvis"
WAKE_WORD_THRESHOLD = 0.5  # openWakeWord official recommendation (was 0.3 — too sensitive)

# Mic gain — R1 microphone sensitivity is very low (raw audio ~100-200 when speaking)
# Apply gain to bring it to normal levels (~3000-10000)
MIC_GAIN = 30.0

# VAD (Silero)
VAD_ENERGY_THRESHOLD = 0.3  # Silero speech probability threshold
VAD_SILENCE_FRAMES = 25  # frames of silence to end speech
VAD_MIN_SPEECH_FRAMES = 3  # minimum frames to count as speech

# State machine
STATE_IDLE = "idle"
STATE_LISTENING = "listening"
STATE_THINKING = "thinking"
STATE_SPEAKING = "speaking"
