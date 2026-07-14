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
        logger.info(f"Hermes session: {session_id}")

    headers.update(extra_headers)

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{config.HERMES_BASE}/chat/completions",
                headers=headers,
                json=body,
                timeout=aiohttp.ClientTimeout(total=30),
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


async def _tts_sentence_to_pcm(sentence: str) -> bytes:
    """Synthesize one sentence to PCM via edge-tts stream() + ffmpeg.

    Uses edge-tts's streaming API to collect MP3 chunks in memory,
    then converts to 48kHz PCM with a single ffmpeg pipe.
    No temp files — faster and fewer failure points.
    """
    import subprocess

    communicate = edge_tts.Communicate(
        text=sentence,
        voice=config.TTS_VOICE,
        rate=config.TTS_RATE,
        volume=config.TTS_VOLUME,
    )

    # Collect MP3 bytes from edge-tts stream
    mp3_chunks = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_chunks.append(chunk["data"])

    if not mp3_chunks:
        return b""

    mp3_data = b"".join(mp3_chunks)

    # Convert MP3 → 48kHz 16-bit mono PCM via ffmpeg stdin pipe
    proc = subprocess.run(
        ["ffmpeg", "-i", "pipe:0",
         "-f", "wav",
         "-ar", str(config.OUTPUT_SAMPLE_RATE),
         "-ac", "1",
         "-acodec", "pcm_s16le",
         "pipe:1", "-y"],
        input=mp3_data,
        capture_output=True,
        timeout=15,
    )

    if proc.returncode != 0:
        logger.error(f"ffmpeg error: {proc.stderr.decode()[:200]}")
        return b""

    # Extract PCM from WAV
    import io as _io
    with wave.open(_io.BytesIO(proc.stdout), "rb") as wf:
        return wf.readframes(wf.getnframes())


async def synthesize_tts_stream(text: str, on_chunk=None) -> int:
    """Stream TTS: split text into sentences, generate and send each sentence's audio immediately.

    Each sentence is synthesized independently via edge-tts streaming API.
    PCM chunks are sent to the client as soon as each sentence is ready.
    If a sentence fails, it is skipped — the pipeline continues with the next sentence.

    Args:
        text: text to synthesize
        on_chunk: async callback(pcm_bytes) called for each PCM chunk as it is generated

    Returns:
        Total bytes of PCM generated
    """
    import re

    # Split text into sentences at Chinese sentence-ending punctuation
    # Don't split on decimal points (e.g., "29.5度") — only on 。！？\n
    sentences = re.split(r'(?<=[。！？\n])\s*', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        logger.warning("No sentences to synthesize")
        return 0

    total_pcm_bytes = 0
    failed_count = 0

    for i, sentence in enumerate(sentences):
        # If on_chunk's WS is dead, stop generating
        # (checked via on_chunk returning False or raising)
        logger.info(f"TTS sentence {i+1}/{len(sentences)}: {sentence[:50]}...")

        # Retry each sentence up to 2 times, but never let failure crash the pipeline
        pcm_data = b""
        for attempt in range(2):
            try:
                pcm_data = await _tts_sentence_to_pcm(sentence)
                if pcm_data:
                    break
                logger.warning(f"TTS sentence {i+1} returned empty audio (attempt {attempt+1})")
            except Exception as e:
                logger.warning(f"TTS sentence {i+1} attempt {attempt+1} failed: {e}")
                if attempt < 1:
                    await asyncio.sleep(1)

        if not pcm_data:
            logger.error(f"TTS sentence {i+1} failed after 2 attempts, skipping: {sentence[:60]}")
            failed_count += 1
            continue

        total_pcm_bytes += len(pcm_data)
        duration = len(pcm_data) / config.OUTPUT_SAMPLE_RATE / 2
        logger.info(f"TTS sentence {i+1} done: {len(pcm_data)} bytes ({duration:.2f}s)")

        # Stream chunks to client immediately
        if on_chunk:
            for chunk in chunk_pcm(pcm_data):
                await on_chunk(chunk)

    logger.info(f"TTS streaming complete: {len(sentences)} sentences, {total_pcm_bytes} bytes total "
                f"({total_pcm_bytes / config.OUTPUT_SAMPLE_RATE / 2:.2f}s), {failed_count} failed")
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
    is_alive=None,
):
    """Run the full pipeline: ASR → Hermes → TTS.

    Args:
        pcm_data: raw 16kHz PCM bytes from user
        session_id: Hermes session ID for context
        on_state: async callback(state_str) for state changes
        on_tts_chunk: async callback(pcm_bytes) for TTS audio chunks
        on_asr_result: async callback(text_str) for ASR result
        is_alive: callable() returning bool — if False, WS is dead, abort early

    Returns:
        (asr_text, hermes_text) tuple
    """
    def alive():
        return is_alive() if is_alive else True
    import asyncio

    # Phase 1: ASR
    if on_state:
        await on_state("thinking")

    # Play "thinking" status sound to the user
    if on_status_sound:
        await on_status_sound("thinking")

    _current_phase = "thinking"

    async def heartbeat():
        """Send periodic heartbeat to prevent WS timeout."""
        while True:
            await asyncio.sleep(3)
            if not alive():
                logger.warning("Heartbeat: WS dead, stopping")
                return
            if on_state:
                try:
                    await on_state(_current_phase, quiet=True)
                except Exception:
                    logger.warning("Heartbeat: WS send failed, stopping")
                    return

    hb_task = asyncio.create_task(heartbeat())

    try:
        # Check WS alive before each expensive phase
        if not alive():
            logger.warning("WS dead before ASR, aborting pipeline")
            return ("", "")

        asr_text = await transcribe_audio(pcm_data)

        if not alive():
            logger.warning("WS dead after ASR, aborting pipeline")
            return (asr_text, "")

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
        if not alive():
            logger.warning("WS dead before Hermes, aborting pipeline")
            return (asr_text, "")

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
        # NOTE: Do NOT cancel heartbeat here. Even during speaking, TTS generation
        # can take 10-20s per sentence (especially with edge-tts retries), and if
        # the R1 client doesn't receive any data for ~30s it disconnects.
        # The heartbeat keeps sending "speaking" state to keep the WS alive.
        # TTS audio chunks are additional data that also keeps it alive.

        if on_state:
            _current_phase = "speaking"
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
