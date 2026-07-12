# -*- coding: utf-8 -*-
"""Voice pipeline: ASR → Hermes → Edge TTS.

Handles the full conversation turn:
1. Receive audio bytes (16kHz PCM)
2. ASR: transcribe speech to text
3. Hermes: generate response (with streaming)
4. Edge TTS: synthesize response to audio
5. Stream audio chunks back to client
"""

import io
import json
import struct
import wave
import asyncio
import logging

import aiohttp
import edge_tts

import config

logger = logging.getLogger("r1voice.pipeline")


async def pcm_to_wav_bytes(pcm_data: bytes, sample_rate: int = 16000) -> bytes:
    """Convert raw PCM bytes to WAV format bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


async def transcribe_audio(pcm_data: bytes) -> str:
    """Send audio to ASR service and get transcription.

    Args:
        pcm_data: raw 16kHz 16-bit mono PCM bytes

    Returns:
        Transcribed text string
    """
    wav_bytes = await pcm_to_wav_bytes(pcm_data, config.INPUT_SAMPLE_RATE)

    form = aiohttp.FormData()
    form.add_field("file", wav_bytes, filename="audio.wav", content_type="audio/wav")
    form.add_field("model", "qwen3-asr")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{config.ASR_BASE}/audio/transcriptions",
                data=form,
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"ASR error {resp.status}: {error_text}")
                    return ""
                result = await resp.json()
                text = result.get("text", "").strip()
                logger.info(f"ASR result: {text}")
                return text
    except Exception as e:
        logger.error(f"ASR exception: {e}")
        return ""


async def ask_hermes(text: str, session_id: str = None) -> str:
    """Send text to Hermes API and get response.

    Args:
        text: user's transcribed speech
        session_id: optional Hermes session ID for multi-turn context
        conversation_history: list of {role, content} dicts for multi-turn context

    Returns:
        Assistant's response text
    """
    headers = {
        "Authorization": f"Bearer {config.HERMES_API_KEY}",
        "Content-Type": "application/json",
    }

    messages = [
        {"role": "system", "content": "你是一个语音助手。回答要自然，适合语音播放。不要用markdown格式，不要用表格。"},
        {"role": "user", "content": text},
    ]

    body = {
        "model": config.HERMES_MODEL,
        "messages": messages,
        "stream": False,  # We'll collect full response, then TTS
    }

    # If we have a session ID, add it as a header for Hermes context
    extra_headers = {}
    if session_id:
        extra_headers["X-Hermes-Session-Id"] = session_id

    headers.update(extra_headers)

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{config.HERMES_BASE}/chat/completions",
                headers=headers,
                json=body,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"Hermes error {resp.status}: {error_text}")
                    return "抱歉，我遇到了一些问题。"
                result = await resp.json()
                response_text = result["choices"][0]["message"]["content"].strip()
                logger.info(f"Hermes response: {response_text[:100]}...")
                return response_text
    except Exception as e:
        logger.error(f"Hermes exception: {e}")
        return "抱歉，连接出了点问题。"


async def synthesize_tts_stream(text: str, on_chunk=None) -> int:
    """Stream TTS: split text into sentences, generate and send each sentence's audio immediately.

    This avoids the problem of generating a huge monolithic audio file for long responses.
    Each sentence is synthesized independently and its PCM chunks are sent as they're ready.

    Args:
        text: text to synthesize
        on_chunk: async callback(pcm_bytes) called for each PCM chunk as it's generated

    Returns:
        Total bytes of PCM generated
    """
    import re
    import tempfile
    import os
    import subprocess

    # Split text into sentences at Chinese/English sentence-ending punctuation
    # Don't split on decimal points (e.g., "29.5度") — only on 。！？\n followed by space or end
    sentences = re.split(r'(?<=[。！？\n])\s*', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        logger.warning("No sentences to synthesize")
        return 0

    total_pcm_bytes = 0

    for i, sentence in enumerate(sentences):
        logger.info(f"TTS sentence {i+1}/{len(sentences)}: {sentence[:50]}...")

        communicate = edge_tts.Communicate(
            text=sentence,
            voice=config.TTS_VOICE,
            rate=config.TTS_RATE,
            volume=config.TTS_VOLUME,
        )

        # Retry on connection timeout
        mp3_path = None
        wav_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3_file:
                mp3_path = mp3_file.name
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
                wav_path = wav_file.name

            for attempt in range(3):
                try:
                    await communicate.save(mp3_path)
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    logger.warning(f"TTS attempt {attempt+1} failed: {e}, retrying...")
                    await asyncio.sleep(1)
                    communicate = edge_tts.Communicate(
                        text=sentence,
                        voice=config.TTS_VOICE,
                        rate=config.TTS_RATE,
                        volume=config.TTS_VOLUME,
                    )

            # Convert to 48kHz 16-bit mono WAV
            proc = subprocess.run(
                [
                    "ffmpeg", "-i", mp3_path,
                    "-f", "wav",
                    "-ar", str(config.OUTPUT_SAMPLE_RATE),
                    "-ac", "1",
                    "-acodec", "pcm_s16le",
                    wav_path, "-y"
                ],
                capture_output=True,
                timeout=30,
            )

            if proc.returncode != 0:
                logger.error(f"ffmpeg error for sentence {i+1}: {proc.stderr.decode()}")
                continue

            # Read WAV, extract PCM
            with wave.open(wav_path, "rb") as wf:
                pcm_data = wf.readframes(wf.getnframes())

            total_pcm_bytes += len(pcm_data)
            logger.info(f"TTS sentence {i+1} done: {len(pcm_data)} bytes ({len(pcm_data) / config.OUTPUT_SAMPLE_RATE / 2:.2f}s)")

            # Stream chunks immediately
            if on_chunk and pcm_data:
                for chunk in chunk_pcm(pcm_data):
                    await on_chunk(chunk)

        finally:
            for path in [mp3_path, wav_path]:
                if path:
                    try:
                        os.unlink(path)
                    except:
                        pass

    logger.info(f"TTS streaming complete: {len(sentences)} sentences, {total_pcm_bytes} bytes total ({total_pcm_bytes / config.OUTPUT_SAMPLE_RATE / 2:.2f}s)")
    return total_pcm_bytes


def chunk_pcm(pcm_data: bytes, chunk_size: int = None) -> list[bytes]:
    """Split PCM data into chunks for streaming.

    Args:
        pcm_data: raw PCM bytes
        chunk_size: bytes per chunk (default: OUTPUT_CHUNK_BYTES)

    Returns:
        List of PCM byte chunks
    """
    if chunk_size is None:
        chunk_size = config.OUTPUT_CHUNK_BYTES

    chunks = []
    for i in range(0, len(pcm_data), chunk_size):
        chunk = pcm_data[i:i + chunk_size]
        chunks.append(chunk)
    return chunks


async def run_pipeline(
    pcm_data: bytes,
    session_id: str = None,
    on_state=None,
    on_tts_chunk=None,
    on_asr_result=None,
    on_tts_done=None,
    on_status_sound=None,
):
    """Run the full pipeline: ASR → Hermes → TTS.

    Args:
        pcm_data: raw 16kHz PCM bytes from user
        session_id: Hermes session ID for context
        on_state: async callback(state_str) for state changes
        on_tts_chunk: async callback(pcm_bytes) for TTS audio chunks
        on_asr_result: async callback(text_str) for ASR result

    Returns:
        (asr_text, hermes_text) tuple
    """
    import asyncio

    # Phase 1: ASR
    if on_state:
        await on_state("thinking")

    # Play "thinking" status sound to the user
    if on_status_sound:
        await on_status_sound("thinking")

    # Start a heartbeat task to keep WS alive during long operations
    async def heartbeat():
        """Send periodic heartbeat to prevent WS timeout. Only active during thinking phase."""
        while True:
            await asyncio.sleep(3)
            if on_state:
                try:
                    # Only send heartbeat if we're still in thinking phase
                    # (don't send during speaking — TTS chunks themselves keep WS alive)
                    await on_state("thinking")  # re-send state as heartbeat
                except Exception:
                    logger.warning("Heartbeat: WS send failed, stopping heartbeat")
                    return  # Stop heartbeat if WS is broken

    hb_task = asyncio.create_task(heartbeat())

    try:
        asr_text = await transcribe_audio(pcm_data)

        if not asr_text:
            logger.info("ASR returned empty text, going back to idle")
            if on_status_sound:
                await on_status_sound("error")
            if on_tts_done:
                await on_tts_done()
            if on_state:
                await on_state("idle")
            return ("", "")

        if on_asr_result:
            await on_asr_result(asr_text)

        # Phase 2: Hermes
        hermes_text = await ask_hermes(asr_text, session_id)

        if not hermes_text:
            if on_status_sound:
                await on_status_sound("error")
            if on_tts_done:
                await on_tts_done()
            if on_state:
                await on_state("idle")
            return (asr_text, "")

        # Phase 3: TTS (streaming — sentence by sentence)
        # Cancel heartbeat before speaking — TTS chunks themselves keep WS alive
        hb_task.cancel()
        try:
            await hb_task
        except asyncio.CancelledError:
            pass

        if on_state:
            await on_state("speaking")
        total_bytes = await synthesize_tts_stream(hermes_text, on_chunk=on_tts_chunk)

        if total_bytes == 0:
            logger.warning("TTS produced no audio")
            if on_status_sound:
                await on_status_sound("error")
            if on_tts_done:
                await on_tts_done()
            if on_state:
                await on_state("idle")
            return (asr_text, hermes_text)

        # Play "done" status sound after TTS finishes
        if on_status_sound:
            await on_status_sound("done")

        # Send tts_done so client stops playback and resumes wake word detection
        if on_tts_done:
            await on_tts_done()

        if on_state:
            await on_state("idle")

        return (asr_text, hermes_text)
    finally:
        hb_task.cancel()
