# -*- coding: utf-8 -*-
"""R1 Voice Server — WebSocket server with state machine.

Protocol:
  Client → Server (binary): 16kHz 16bit mono PCM frames
  Client → Server (text):   JSON control {"type": "wake"|"stop"|"bye"}
  Server → Client (text):   JSON state {"type":"state","state":"..."}
  Server → Client (binary): 48kHz 16bit mono PCM chunks (TTS audio)
"""

import asyncio
import json
import logging
import struct
import sys
import os
from pathlib import Path

import websockets
from websockets.server import serve

import config
from vad_silero import SileroVAD
from pipeline import run_pipeline

# Load status sound PCM files at startup
_status_sounds = {}

def load_status_sounds():
    """Pre-load status sound WAV files as raw PCM bytes."""
    sounds_dir = Path(__file__).parent / "sounds"
    for name in ("thinking", "done", "error"):
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
        self.audio_buffer = bytearray()
        self.hermes_session_id = None
        self.is_streaming_tts = False

    async def send_state(self, state: str):
        """Send state change to client and update internal state."""
        self.state = state
        msg = json.dumps({"type": "state", "state": state})
        try:
            await self.ws.send(msg)
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
    """Handle incoming PCM audio from R1."""
    if client.state == config.STATE_IDLE:
        # Local wake word detection is done on-device (openWakeWord/TFLite).
        # In IDLE state, client should not be sending audio.
        # If we receive audio in IDLE, just ignore it.
        pass

    elif client.state == config.STATE_LISTENING:
        # Feed to VAD
        result = client.vad.process_frame(data)

        if result == "speech_start":
            logger.info("VAD: speech started")

        if result in ("speech", "speech_start"):
            # Accumulate audio
            client.audio_buffer.extend(data)

        elif result == "speech_end":
            logger.info(f"VAD: speech ended, buffer={len(client.audio_buffer)} bytes")
            client.vad.reset()

            # Run pipeline if we have enough audio
            min_bytes = config.INPUT_FRAME_BYTES * config.VAD_MIN_SPEECH_FRAMES
            if len(client.audio_buffer) >= min_bytes:
                pcm_data = bytes(client.audio_buffer)
                client.audio_buffer.clear()

                # Debug: save the audio for analysis
                import wave, time as _time
                debug_path = f"/tmp/r1_debug_{int(_time.time())}.wav"
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
                    """Send tts_done so client resumes wake word detection."""
                    await client.send_json({"type": "tts_done"})

                async def on_status_sound(name):
                    """Send a pre-recorded status sound as PCM chunks."""
                    pcm = _status_sounds.get(name)
                    if pcm:
                        from pipeline import chunk_pcm
                        for chunk in chunk_pcm(pcm, config.OUTPUT_CHUNK_BYTES):
                            await client.send_tts_chunk(chunk)
                        logger.info(f"Played status sound: {name}")

                # Run ASR → Hermes → TTS
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
            else:
                logger.info("Audio too short, discarding")
                client.audio_buffer.clear()
                # Play error sound
                pcm = _status_sounds.get("error")
                if pcm:
                    from pipeline import chunk_pcm
                    for chunk in chunk_pcm(pcm, config.OUTPUT_CHUNK_BYTES):
                        await client.send_tts_chunk(chunk)
                    logger.info("Played status sound: error")
                await client.send_json({"type": "tts_done"})
                await client.send_state(config.STATE_IDLE)

    elif client.state == config.STATE_SPEAKING:
        # Client sending audio while we're speaking — could be barge-in
        # For now, just ignore
        pass


async def handle_text(client: ClientSession, text: str):
    """Handle JSON control messages from R1."""
    try:
        msg = json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"Invalid JSON: {text}")
        return

    msg_type = msg.get("type")

    if msg_type == "wake":
        logger.info("← wake event")
        if client.state == config.STATE_SPEAKING:
            # Barge-in: stop TTS, start listening
            logger.info("Barge-in: interrupting TTS")
            client.is_streaming_tts = False
        await client.send_state(config.STATE_LISTENING)
        client.vad.reset()
        client.audio_buffer.clear()

    elif msg_type == "stop":
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
