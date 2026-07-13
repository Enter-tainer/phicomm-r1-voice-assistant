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


import uuid

class ClientSession:
    """Per-client state."""

    def __init__(self, ws):
        self.ws = ws
        self.state = config.STATE_IDLE
        self.vad = SileroVAD()
        self.wake_word = WakeWordDetector()
        self.audio_buffer = bytearray()
        self.hermes_session_id = f"r1-voice-{uuid.uuid4().hex[:12]}"
        self.is_streaming_tts = False
        self.last_wake_score = 0.0
        self.wake_score_log = []
        self.ws_alive = True
        self.pipeline_task = None

    async def send_state(self, state: str, quiet: bool = False):
        """Send state change to client and update internal state."""
        self.state = state
        if not self.ws_alive:
            return
        msg = json.dumps({"type": "state", "state": state})
        try:
            await self.ws.send(msg)
            if not quiet:
                logger.info(f"→ state: {state}")
        except Exception as e:
            logger.error(f"Failed to send state: {e}")
            self.ws_alive = False

    async def send_tts_chunk(self, pcm: bytes):
        """Send a PCM chunk to client."""
        if not self.ws_alive:
            return
        try:
            await self.ws.send(pcm)
        except Exception as e:
            logger.error(f"Failed to send TTS chunk: {e}")
            self.ws_alive = False

    async def send_json(self, data: dict):
        """Send JSON message to client."""
        if not self.ws_alive:
            return
        try:
            await self.ws.send(json.dumps(data))
        except Exception as e:
            logger.error(f"Failed to send JSON: {e}")
            self.ws_alive = False


async def handle_binary(client: ClientSession, data: bytes):
    """Handle incoming PCM audio from R1."""
    if not hasattr(client, '_frame_count'):
        client._frame_count = 0
    client._frame_count += 1
    if client._frame_count <= 3 or client._frame_count % 250 == 0:
        logger.info(f"handle_binary: frame={client._frame_count} state={client.state} bytes={len(data)}")
    if client.state == config.STATE_IDLE:
        # Feed audio to openWakeWord
        # R1 mic sensitivity is very low — apply gain before prediction
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        samples = np.clip(samples * config.MIC_GAIN, -32768, 32767)

        score = client.wake_word.predict(samples)
        client.last_wake_score = score

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
                beep_duration = len(pcm) / (
                    config.OUTPUT_SAMPLE_RATE
                    * config.OUTPUT_SAMPLE_SIZE
                    * config.OUTPUT_CHANNELS
                )
                await asyncio.sleep(beep_duration + 0.15)
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

                # Launch pipeline as a background task so the message loop
                # keeps consuming audio frames (discarded during THINKING/SPEAKING).
                # This prevents the WebSocket internal queue from filling up and
                # killing the connection during long ASR/LLM/TTS operations.
                async def run_pipeline_safe():
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
                            is_alive=lambda: client.ws_alive,
                        )
                    except Exception as e:
                        logger.error(f"Pipeline crashed: {e}", exc_info=True)
                        try:
                            await client.send_json({"type": "tts_done"})
                            await client.send_state(config.STATE_IDLE)
                        except Exception:
                            pass
                    finally:
                        client.pipeline_task = None

                client.pipeline_task = asyncio.create_task(run_pipeline_safe())
                logger.info("Pipeline launched as background task")

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
        # Consume and discard audio while speaking (keeps WS message queue drained)
        pass

    elif client.state == config.STATE_THINKING:
        # Consume and discard audio while thinking (keeps WS message queue drained)
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
    logger.info(f"Session ID: {client.hermes_session_id}")
    await client.send_state(config.STATE_IDLE)

    try:
        logger.info(f"DEBUG: entering message loop for {remote}")
        msg_count = 0
        async for message in websocket:
            msg_count += 1
            if msg_count <= 3:
                logger.info(f"DEBUG: received message #{msg_count}, type={type(message).__name__}, len={len(message) if isinstance(message, (bytes, str)) else 'N/A'}")
            if isinstance(message, bytes):
                await handle_binary(client, message)
            elif isinstance(message, str):
                logger.info(f"Received text message: {message[:100]}")
                await handle_text(client, message)
            else:
                logger.warning(f"Unknown message type: {type(message)}")
        logger.info(f"DEBUG: message loop ended after {msg_count} messages")
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Client disconnected: {remote}")
    except Exception as e:
        logger.error(f"Client handler error: {e}", exc_info=True)
    finally:
        # Cancel any running pipeline when client disconnects
        if client.pipeline_task and not client.pipeline_task.done():
            client.pipeline_task.cancel()
            try:
                await client.pipeline_task
            except asyncio.CancelledError:
                pass


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
