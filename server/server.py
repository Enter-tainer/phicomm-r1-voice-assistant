# -*- coding: utf-8 -*-
"""R1 Voice Server — WebSocket server with server-side wake word detection.

Protocol:
  Client → Server (binary): 16kHz 16bit mono PCM frames (ALWAYS streaming)
  Client → Server (text):   JSON control {"type": "stop"|"bye"}
  Server → Client (text):   JSON state {"type":"state","state":"..."}
  Server → Client (binary): 48kHz 16bit mono PCM chunks (TTS audio + beeps)

State machine:
  IDLE:      Receiving audio, running openWakeWord. On detection → play beep → LISTENING
  LISTENING: Receiving audio, running VAD. On speech_end → THINKING
  THINKING:  ASR + Hermes. On response ready → SPEAKING
  SPEAKING:  Streaming TTS. On done → IDLE
"""

import asyncio
import json
import logging
import struct
import sys
import os
import time
import numpy as np
from pathlib import Path

import websockets
from websockets.server import serve

import config
from vad_silero import SileroVAD
from pipeline import run_pipeline
from wake_word import WakeWordDetector

# Load status sound PCM files at startup
_status_sounds = {}

def load_status_sounds():
    """Pre-load status sound WAV files as raw PCM bytes."""
    sounds_dir = Path(__file__).parent / "sounds"
    for name in ("thinking", "done", "error", "wake"):
        wav_path = sounds_dir / f"{name}.wav"
        if wav_path.exists():
            import wave
            with wave.open(str(wav_path), "rb") as wf:
                pcm = wf.readframes(wf.getnframes())
            _status_sounds[name] = pcm
            logger.info(f"Loaded status sound: {name} ({len(pcm)} bytes)")
        else:
            logger.warning(f"Status sound not found: {wav_path}")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("r1voice.server")


class ClientSession:
    """Per-client state."""

    def __init__(self, ws):
        self.ws = ws
        self.state = config.STATE_IDLE
        self.vad = SileroVAD()
        self.wake_word = WakeWordDetector()
        self.audio_buffer = bytearray()
        self.hermes_session_id = None
        self.is_streaming_tts = False
        self.last_wake_score = 0.0
        self.wake_score_log = []

    async def send_state(self, state: str, quiet: bool = False):
        """Send state change to client and update internal state."""
        self.state = state
        msg = json.dumps({"type": "state", "state": state})
        try:
            await self.ws.send(msg)
            if not quiet:
                logger.info(f"→ state: {state}")
        except Exception as e:
            logger.error(f"Failed to send state: {e}")

    async def send_tts_chunk(self, pcm: bytes):
        """Send a PCM chunk to client."""
        try:
            await self.ws.send(pcm)
        except Exception as e:
            logger.error(f"Failed to send TTS chunk: {e}")

    async def send_json(self, data: dict):
        """Send JSON message to client."""
        try:
            await self.ws.send(json.dumps(data))
        except Exception as e:
            logger.error(f"Failed to send JSON: {e}")


async def handle_binary(client: ClientSession, data: bytes):
    """Handle incoming PCM audio from R1.

    R1 streams audio continuously. In IDLE state we feed it to openWakeWord.
    In LISTENING state we feed it to VAD.
    In SPEAKING state we ignore it (no barge-in for now).
    """
    if client.state == config.STATE_IDLE:
        # Feed audio to openWakeWord
        # R1 mic sensitivity is very low — apply gain before prediction
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        samples = np.clip(samples * config.MIC_GAIN, -32768, 32767)

        score = client.wake_word.predict(samples)
        client.last_wake_score = score

        # Save continuous audio for debugging (first 10 seconds = 500 frames)
        if not hasattr(client, '_debug_audio'):
            client._debug_audio = bytearray()
        if len(client._debug_audio) < 640 * 500:  # 10 seconds
            client._debug_audio.extend(data)
            if len(client._debug_audio) >= 640 * 500:
                import wave
                try:
                    with wave.open("/tmp/r1_mic_continuous.wav", "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(16000)
                        wf.writeframes(bytes(client._debug_audio))
                    logger.info(f"Saved 10s debug audio: /tmp/r1_mic_continuous.wav ({len(client._debug_audio)} bytes)")
                except Exception as e:
                    logger.warning(f"Failed to save debug audio: {e}")

        # Log every frame with non-zero score or high audio
        audio_max = int(np.max(np.abs(samples))) if len(samples) > 0 else 0
        if score > 0.001 or (audio_max > 5000 and client.wake_word.prediction_count % 50 == 0):
            logger.info(f"Wake word: score={score:.6f} audio_max={audio_max} frame={client.wake_word.prediction_count}")

        # Log summary every ~5s
        client.wake_score_log.append(score)
        if len(client.wake_score_log) >= 250:
            max_score = max(client.wake_score_log)
            avg_score = sum(client.wake_score_log) / len(client.wake_score_log)
            logger.info(f"Wake word summary (last 5s): max={max_score:.6f} avg={avg_score:.6f} | audio_max={audio_max}")
            client.wake_score_log.clear()

        if score > config.WAKE_WORD_THRESHOLD:
            logger.info(f"🔥 Wake word detected! score={score:.4f}")
            client.wake_word.reset()

            # Play wake beep
            pcm = _status_sounds.get("wake")
            if pcm:
                from pipeline import chunk_pcm
                for chunk in chunk_pcm(pcm, config.OUTPUT_CHUNK_BYTES):
                    await client.send_tts_chunk(chunk)
                logger.info("Played wake beep")

                # Wait for beep to finish playing on device before starting VAD.
                # Without this, the mic picks up the beep echo and VAD false-triggers.
                beep_duration = len(pcm) / (
                    config.OUTPUT_SAMPLE_RATE
                    * config.OUTPUT_SAMPLE_SIZE
                    * config.OUTPUT_CHANNELS
                )
                await asyncio.sleep(beep_duration + 0.15)  # beep duration + 150ms buffer
                logger.info(f"Waited {beep_duration:.3f}s + 150ms for beep, starting VAD")

            # Switch to listening
            await client.send_state(config.STATE_LISTENING)
            client.vad.reset()
            client.audio_buffer.clear()

    elif client.state == config.STATE_LISTENING:
        # Apply mic gain before VAD
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        samples = np.clip(samples * config.MIC_GAIN, -32768, 32767)
        data = samples.astype(np.int16).tobytes()

        # Feed to VAD
        result = client.vad.process_frame(data)

        if result == "speech_start":
            logger.info("VAD: speech started")

        if result in ("speech", "speech_start"):
            client.audio_buffer.extend(data)

        elif result == "speech_end":
            logger.info(f"VAD: speech ended, buffer={len(client.audio_buffer)} bytes")
            client.vad.reset()

            min_bytes = config.INPUT_FRAME_BYTES * config.VAD_MIN_SPEECH_FRAMES
            if len(client.audio_buffer) >= min_bytes:
                pcm_data = bytes(client.audio_buffer)
                client.audio_buffer.clear()

                # Debug: save the audio
                import wave
                debug_path = f"/tmp/r1_debug_{int(time.time())}.wav"
                try:
                    with wave.open(debug_path, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(16000)
                        wf.writeframes(pcm_data)
                    logger.info(f"Debug audio saved: {debug_path} ({len(pcm_data)} bytes)")
                except Exception as e:
                    logger.warning(f"Failed to save debug audio: {e}")

                async def on_tts_done():
                    await client.send_json({"type": "tts_done"})

                async def on_status_sound(name):
                    pcm = _status_sounds.get(name)
                    if pcm:
                        from pipeline import chunk_pcm
                        for chunk in chunk_pcm(pcm, config.OUTPUT_CHUNK_BYTES):
                            await client.send_tts_chunk(chunk)
                        logger.info(f"Played status sound: {name}")

                try:
                    await run_pipeline(
                        pcm_data=pcm_data,
                        session_id=client.hermes_session_id,
                        on_state=client.send_state,
                        on_tts_chunk=client.send_tts_chunk,
                        on_asr_result=lambda text: client.send_json(
                            {"type": "asr_result", "text": text}
                        ),
                        on_tts_done=on_tts_done,
                        on_status_sound=on_status_sound,
                    )
                except Exception as e:
                    logger.error(f"Pipeline crashed: {e}", exc_info=True)
                    try:
                        await client.send_json({"type": "tts_done"})
                        await client.send_state(config.STATE_IDLE)
                    except Exception:
                        pass
            else:
                logger.info("Audio too short, discarding")
                client.audio_buffer.clear()
                pcm = _status_sounds.get("error")
                if pcm:
                    from pipeline import chunk_pcm
                    for chunk in chunk_pcm(pcm, config.OUTPUT_CHUNK_BYTES):
                        await client.send_tts_chunk(chunk)
                await client.send_json({"type": "tts_done"})
                await client.send_state(config.STATE_IDLE)

    elif client.state == config.STATE_SPEAKING:
        # Ignore audio while speaking (no barge-in)
        pass

    elif client.state == config.STATE_THINKING:
        # Ignore audio while thinking
        pass


async def handle_text(client: ClientSession, text: str):
    """Handle JSON control messages from R1."""
    try:
        msg = json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"Invalid JSON: {text}")
        return

    msg_type = msg.get("type")

    if msg_type == "stop":
        logger.info("← stop event")
        client.audio_buffer.clear()
        await client.send_state(config.STATE_IDLE)

    elif msg_type == "bye":
        logger.info("← bye event")

    else:
        logger.warning(f"Unknown message type: {msg_type}")


async def handle_client(websocket):
    """Handle a single WebSocket client connection."""
    remote = websocket.remote_address
    logger.info(f"Client connected: {remote}")

    client = ClientSession(websocket)
    await client.send_state(config.STATE_IDLE)

    try:
        async for message in websocket:
            if isinstance(message, bytes):
                await handle_binary(client, message)
            elif isinstance(message, str):
                await handle_text(client, message)
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Client disconnected: {remote}")
    except Exception as e:
        logger.error(f"Client handler error: {e}", exc_info=True)


async def main():
    """Start the WebSocket server."""
    load_status_sounds()
    logger.info(f"R1 Voice Server starting on {config.WS_HOST}:{config.WS_PORT}")
    logger.info(f"  Wake word: openWakeWord (hey_jarvis, ONNX, threshold={config.WAKE_WORD_THRESHOLD})")
    logger.info(f"  ASR: {config.ASR_BASE}")
    logger.info(f"  Hermes: {config.HERMES_BASE}")
    logger.info(f"  TTS: Edge TTS ({config.TTS_VOICE})")
    logger.info(f"  Input: {config.INPUT_SAMPLE_RATE}Hz {config.INPUT_FRAME_BYTES}B/frame")
    logger.info(f"  Output: {config.OUTPUT_SAMPLE_RATE}Hz {config.OUTPUT_CHUNK_BYTES}B/chunk")

    async with serve(handle_client, config.WS_HOST, config.WS_PORT,
                     ping_interval=None, ping_timeout=None):
        logger.info(f"✅ Listening on ws://{config.WS_HOST}:{config.WS_PORT}")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped")
