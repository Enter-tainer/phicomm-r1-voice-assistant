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

# Global health state — written to /tmp/r1_server_state.json for the watchdog.
# The watchdog (r1_watchdog.py) reads this to detect audio stalls / disconnects
# and auto-recover the R1 (kill Phicomm services, restart app, reboot as last resort).
_server_state = {
    "connected": False,
    "last_frame_time": 0.0,
    "state": config.STATE_IDLE,
    "pid": os.getpid(),
    "server_time": time.time(),
}

STATE_FILE = "/tmp/r1_server_state.json"


def _write_state_file():
    _server_state["server_time"] = time.time()
    _server_state["state"] = _server_state.get("state", config.STATE_IDLE)
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(_server_state, f)
    except Exception:
        pass


async def state_writer_loop():
    """Periodically dump server health state for the watchdog."""
    while True:
        _write_state_file()
        await asyncio.sleep(5)

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

# --- Session grouping state (persisted across restarts) -------------------
# "Interaction" = a completed voice turn. We reuse the last session ID when a
# new connection arrives within SESSION_REUSE_WINDOW_MINUTES of the last
# interaction, so short reconnects keep conversation memory; longer gaps
# start a fresh session.
_session_state = {
    "session_id": None,
    "last_interaction_time": 0.0,
}


def _load_session_state():
    try:
        with open(config.SESSION_STATE_FILE) as f:
            data = json.load(f)
            _session_state["session_id"] = data.get("session_id")
            _session_state["last_interaction_time"] = data.get(
                "last_interaction_time", 0.0
            )
    except Exception:
        pass


def _save_session_state():
    try:
        os.makedirs(os.path.dirname(config.SESSION_STATE_FILE), exist_ok=True)
        with open(config.SESSION_STATE_FILE, "w") as f:
            json.dump(_session_state, f)
    except Exception as e:
        logger.warning(f"Failed to save session state: {e}")


def _pick_session_id() -> str:
    """Reuse the last session ID if within the interaction window, else new.

    The session ID groups turns that belong together (short reconnects keep
    memory via the gateway's session history) vs fresh starts (idle gap
    longer than SESSION_REUSE_WINDOW_MINUTES).
    """
    now = time.time()
    window = config.SESSION_REUSE_WINDOW_MINUTES * 60
    last = _session_state.get("last_interaction_time", 0.0)
    old_id = _session_state.get("session_id")
    if old_id and (now - last) <= window:
        logger.info(
            f"Reusing session {old_id} (last interaction "
            f"{(now - last) / 60:.1f} min ago, window {config.SESSION_REUSE_WINDOW_MINUTES} min)"
        )
        return old_id
    new_id = f"r1-voice-{uuid.uuid4().hex[:12]}"
    logger.info(
        f"New session {new_id} (last interaction {(now - last) / 60:.1f} min ago)"
    )
    _session_state["session_id"] = new_id
    _session_state["last_interaction_time"] = now
    _save_session_state()
    return new_id


def _mark_interaction():
    """Update last interaction time (called after a completed voice turn)."""
    _session_state["last_interaction_time"] = time.time()
    _save_session_state()


class ClientSession:
    """Per-client state."""

    def __init__(self, ws):
        self.ws = ws
        self.state = config.STATE_IDLE
        self.vad = SileroVAD()
        self.wake_word = WakeWordDetector()
        self.audio_buffer = bytearray()
        self.hermes_session_id = _pick_session_id()
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
        """Send a PCM chunk to client, applying dynamic backpressure.

        Uses the asyncio transport's write-buffer watermark as a closed-loop
        flow control: if the OS/TCP send buffer is already holding more than
        ~0.5s of audio (HIGH_WATER), we pause until it drains below the low
        watermark. This adapts to the R1's actual consumption rate instead of
        a fixed sleep:
          - R1 plays fast / network is fast → buffer stays low → no waiting
          - R1 stalls / network slows → buffer rises → we naturally slow down
        This prevents the WS deadlock that occurred when the server pushed a
        whole multi-second sentence back-to-back (AudioTrack buffer filled →
        writePcm blocked the read thread → TCP backed up → WS died).
        """
        if not self.ws_alive:
            return
        try:
            await self.ws.send(pcm)
        except Exception as e:
            logger.error(f"Failed to send TTS chunk: {e}")
            self.ws_alive = False
            return
        # Dynamic backpressure: pause while the write buffer is above the
        # high watermark, resume when it drains below the low watermark.
        # 96000 bytes = 1s of 48kHz 16-bit mono audio.
        HIGH_WATER = 48000   # 0.5s of audio
        LOW_WATER = 9600     # 0.1s of audio
        try:
            transport = self.ws.transport
            while transport.get_write_buffer_size() > HIGH_WATER:
                await asyncio.sleep(0.01)
                if not self.ws_alive:
                    return
                if transport.get_write_buffer_size() < LOW_WATER:
                    break
        except Exception:
            pass  # transport may be gone on disconnect; send errors already handled

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
    _server_state["last_frame_time"] = time.time()
    _server_state["connected"] = True
    if not hasattr(client, '_frame_count'):
        client._frame_count = 0
    client._frame_count += 1
    if client._frame_count <= 3 or client._frame_count % 250 == 0:
        logger.info(f"handle_binary: frame={client._frame_count} state={client.state} bytes={len(data)}")
    if client.state == config.STATE_IDLE:
        # Feed audio to openWakeWord
        # R1 mic sensitivity is very low — apply gain before prediction
        # openWakeWord internally normalizes int16 to [-1,1]; passing float32
        # in int16 range so the gain actually takes effect without clipping.
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
                        # A voice turn completed — update session grouping state
                        _mark_interaction()
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
    _server_state["connected"] = True

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
        _server_state["connected"] = False
        # Cancel any running pipeline when client disconnects
        if client.pipeline_task and not client.pipeline_task.done():
            client.pipeline_task.cancel()
            try:
                await client.pipeline_task
            except asyncio.CancelledError:
                pass


async def main():
    """Start the WebSocket server."""
    _load_session_state()
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
        asyncio.create_task(state_writer_loop())
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped")
