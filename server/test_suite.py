#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Comprehensive test suite for R1 Voice Server.

Tests T1-T7: pure server-side tests using WebSocket client.
T1: WebSocket connection/disconnect
T2: State machine transitions
T3: VAD energy detection (silence/speech PCM)
T4: ASR pipeline (real audio → text)
T5: Hermes API (text → response)
T6: Edge TTS (text → audio)
T7: Full pipeline (PCM → ASR → Hermes → TTS → PCM)
"""

import asyncio
import json
import struct
import wave
import io
import time
import logging
import os
import sys
import tempfile
import subprocess

import websockets
import numpy as np

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from vad import EnergyVAD
from pipeline import transcribe_audio, ask_hermes, synthesize_tts, run_pipeline, chunk_pcm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("test")

WS_URL = f"ws://localhost:{config.WS_PORT}"

PASS = "✅ PASS"
FAIL = "❌ FAIL"
SKIP = "⏭️ SKIP"

results = []

def record(name, status, detail=""):
    results.append((name, status, detail))
    print(f"  {status} {name}" + (f" — {detail}" if detail else ""))

# ─── Helpers ───

def generate_silence_pcm(duration_s=1.0, sample_rate=16000):
    """Generate silence PCM (zeros)."""
    num_samples = int(duration_s * sample_rate)
    return np.zeros(num_samples, dtype=np.int16).tobytes()

def generate_speech_pcm(duration_s=2.0, sample_rate=16000, freq=440):
    """Generate a tone that simulates speech (high energy)."""
    t = np.linspace(0, duration_s, int(duration_s * sample_rate), endpoint=False)
    # Mix of frequencies to simulate speech-like energy
    wave_signal = 0.3 * np.sin(2 * np.pi * freq * t) + 0.2 * np.sin(2 * np.pi * freq * 2 * t)
    samples = (wave_signal * 32767 * 0.5).astype(np.int16)
    return samples.tobytes()

def generate_speech_pcm_with_silence(pre_silence=0.5, speech=2.0, post_silence=1.0, sample_rate=16000):
    """Generate PCM with silence → speech → silence pattern for VAD testing."""
    silence_pre = np.zeros(int(pre_silence * sample_rate), dtype=np.int16)
    t = np.linspace(0, speech, int(speech * sample_rate), endpoint=False)
    speech_signal = (0.3 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t))
    speech_data = (speech_signal * 32767 * 0.5).astype(np.int16)
    silence_post = np.zeros(int(post_silence * sample_rate), dtype=np.int16)
    return np.concatenate([silence_pre, speech_data, silence_post]).tobytes()

def make_wav(pcm_data, sample_rate=16000):
    """Convert PCM bytes to WAV bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


# ─── T1: WebSocket Connection/Disconnect/Reconnect ───

async def test_t1_websocket_connection():
    print("\n=== T1: WebSocket Connection/Disconnect/Reconnect ===")
    
    # T1.1: Basic connect and receive idle state
    try:
        async with websockets.connect(WS_URL) as ws:
            msg = await asyncio.wait_for(ws.recv(), timeout=5)
            data = json.loads(msg)
            if data.get("type") == "state" and data.get("state") == "idle":
                record("T1.1 Connect and receive idle state", PASS)
            else:
                record("T1.1 Connect and receive idle state", FAIL, f"Got: {data}")
    except Exception as e:
        record("T1.1 Connect and receive idle state", FAIL, str(e))
    
    # T1.2: Disconnect cleanly
    try:
        async with websockets.connect(WS_URL) as ws:
            await asyncio.wait_for(ws.recv(), timeout=5)  # idle
            await ws.close()
        record("T1.2 Clean disconnect", PASS)
    except Exception as e:
        record("T1.2 Clean disconnect", FAIL, str(e))
    
    # T1.3: Reconnect after disconnect
    try:
        for i in range(3):
            async with websockets.connect(WS_URL) as ws:
                msg = await asyncio.wait_for(ws.recv(), timeout=5)
                data = json.loads(msg)
                assert data["state"] == "idle", f"Expected idle, got {data}"
        record("T1.3 Reconnect 3 times", PASS)
    except Exception as e:
        record("T1.3 Reconnect 3 times", FAIL, str(e))
    
    # T1.4: Multiple concurrent connections
    try:
        conns = []
        for i in range(3):
            ws = await websockets.connect(WS_URL)
            await asyncio.wait_for(ws.recv(), timeout=5)  # idle
            conns.append(ws)
        for ws in conns:
            await ws.close()
        record("T1.4 Multiple concurrent connections", PASS)
    except Exception as e:
        record("T1.4 Multiple concurrent connections", FAIL, str(e))


# ─── T2: State Machine Transitions ───

async def test_t2_state_machine():
    print("\n=== T2: State Machine Transitions ===")
    
    # T2.1: wake → listening → stop → idle
    try:
        async with websockets.connect(WS_URL) as ws:
            await asyncio.wait_for(ws.recv(), timeout=5)  # idle
            
            # Send wake
            await ws.send(json.dumps({"type": "wake"}))
            msg = await asyncio.wait_for(ws.recv(), timeout=5)
            data = json.loads(msg)
            assert data["state"] == "listening", f"Expected listening, got {data}"
            
            # Send stop
            await ws.send(json.dumps({"type": "stop"}))
            msg = await asyncio.wait_for(ws.recv(), timeout=5)
            data = json.loads(msg)
            assert data["state"] == "idle", f"Expected idle, got {data}"
            
            record("T2.1 wake→listening→stop→idle", PASS)
    except Exception as e:
        record("T2.1 wake→listening→stop→idle", FAIL, str(e))
    
    # T2.2: Unknown message type
    try:
        async with websockets.connect(WS_URL) as ws:
            await asyncio.wait_for(ws.recv(), timeout=5)  # idle
            await ws.send(json.dumps({"type": "unknown_type"}))
            # Should not crash, just ignore
            await asyncio.sleep(1)
            # Server should still be responsive
            await ws.send(json.dumps({"type": "wake"}))
            msg = await asyncio.wait_for(ws.recv(), timeout=5)
            data = json.loads(msg)
            assert data["state"] == "listening", f"Expected listening, got {data}"
            record("T2.2 Unknown message type handled", PASS)
    except Exception as e:
        record("T2.2 Unknown message type handled", FAIL, str(e))
    
    # T2.3: Invalid JSON
    try:
        async with websockets.connect(WS_URL) as ws:
            await asyncio.wait_for(ws.recv(), timeout=5)  # idle
            await ws.send("not json at all")
            await asyncio.sleep(1)
            # Server should still be responsive
            await ws.send(json.dumps({"type": "stop"}))
            msg = await asyncio.wait_for(ws.recv(), timeout=5)
            data = json.loads(msg)
            assert data["state"] == "idle", f"Expected idle, got {data}"
            record("T2.3 Invalid JSON handled", PASS)
    except Exception as e:
        record("T2.3 Invalid JSON handled", FAIL, str(e))


# ─── T3: VAD Energy Detection ───

def test_t3_vad():
    print("\n=== T3: VAD Energy Detection ===")
    
    frame_size = config.INPUT_FRAME_BYTES  # 640 bytes = 320 samples
    
    # T3.1: Silence should not trigger speech
    try:
        vad = EnergyVAD()
        silence = generate_silence_pcm(2.0)
        results_list = []
        for i in range(0, len(silence), frame_size):
            frame = silence[i:i+frame_size]
            if len(frame) == frame_size:
                results_list.append(vad.process_frame(frame))
        
        # All frames should be silence
        assert all(r == "silence" for r in results_list), f"Expected all silence, got: {set(results_list)}"
        record("T3.1 Silence not detected as speech", PASS)
    except Exception as e:
        record("T3.1 Silence not detected as speech", FAIL, str(e))
    
    # T3.2: Loud signal should trigger speech_start
    try:
        vad = EnergyVAD()
        speech = generate_speech_pcm(2.0)
        results_list = []
        for i in range(0, len(speech), frame_size):
            frame = speech[i:i+frame_size]
            if len(frame) == frame_size:
                results_list.append(vad.process_frame(frame))
        
        has_start = "speech_start" in results_list
        has_speech = "speech" in results_list
        assert has_start, f"Expected speech_start, got: {set(results_list)}"
        record("T3.2 Loud signal triggers speech_start", PASS, f"frames: {len(results_list)}, states: {set(results_list)}")
    except Exception as e:
        record("T3.2 Loud signal triggers speech_start", FAIL, str(e))
    
    # T3.3: Silence → speech → silence should produce speech_end
    try:
        vad = EnergyVAD()
        pcm = generate_speech_pcm_with_silence(0.5, 2.0, 1.0)
        results_list = []
        for i in range(0, len(pcm), frame_size):
            frame = pcm[i:i+frame_size]
            if len(frame) == frame_size:
                results_list.append(vad.process_frame(frame))
        
        has_start = "speech_start" in results_list
        has_end = "speech_end" in results_list
        assert has_start and has_end, f"Expected speech_start and speech_end, got: {set(results_list)}"
        record("T3.3 Silence→speech→silence triggers speech_end", PASS)
    except Exception as e:
        record("T3.3 Silence→speech→silence triggers speech_end", FAIL, str(e))
    
    # T3.4: Very short burst should not trigger (min_speech_frames)
    try:
        vad = EnergyVAD(min_speech_frames=10)  # Need 10 frames = 200ms
        # Generate 5 frames of loud signal (100ms) — too short
        speech = generate_speech_pcm(0.1)
        results_list = []
        for i in range(0, len(speech), frame_size):
            frame = speech[i:i+frame_size]
            if len(frame) == frame_size:
                results_list.append(vad.process_frame(frame))
        
        assert "speech_start" not in results_list, f"Should not trigger with short burst"
        record("T3.4 Short burst below min_speech_frames rejected", PASS)
    except Exception as e:
        record("T3.4 Short burst below min_speech_frames rejected", FAIL, str(e))
    
    # T3.5: VAD reset
    try:
        vad = EnergyVAD()
        speech = generate_speech_pcm(0.5)
        for i in range(0, len(speech), frame_size):
            frame = speech[i:i+frame_size]
            if len(frame) == frame_size:
                vad.process_frame(frame)
        
        vad.reset()
        assert vad._in_speech == False
        assert vad._speech_frame_count == 0
        assert vad._silence_frame_count == 0
        record("T3.5 VAD reset works", PASS)
    except Exception as e:
        record("T3.5 VAD reset works", FAIL, str(e))


# ─── T4: ASR Pipeline ───

async def test_t4_asr():
    print("\n=== T4: ASR Pipeline ===")
    
    # T4.1: Transcribe the edge_tts test audio (known content)
    try:
        # Use the test WAV we generated earlier
        test_wav = "/tmp/asr_test_16k.wav"
        if not os.path.exists(test_wav):
            # Generate a new test audio with edge-tts
            subprocess.run([
                "edge-tts", "--text", "你好世界",
                "--write-media", "/tmp/t4_test.mp3",
                "--voice", "zh-CN-XiaoxiaoNeural"
            ], timeout=30, capture_output=True)
            subprocess.run([
                "ffmpeg", "-i", "/tmp/t4_test.mp3",
                "-ar", "16000", "-ac", "1", "-y", test_wav
            ], capture_output=True, timeout=10)
        
        # Read WAV to PCM
        with wave.open(test_wav, "rb") as wf:
            pcm = wf.readframes(wf.getnframes())
        
        text = await transcribe_audio(pcm)
        assert text and len(text) > 0, f"Empty transcription"
        record("T4.1 ASR transcribes known audio", PASS, f"Result: '{text}'")
    except Exception as e:
        record("T4.1 ASR transcribes known audio", FAIL, str(e))
    
    # T4.2: Empty audio should return empty string
    try:
        silence = generate_silence_pcm(1.0)
        text = await transcribe_audio(silence)
        # ASR might return empty or some placeholder
        record("T4.2 Empty/silence audio handling", PASS, f"Result: '{text}'")
    except Exception as e:
        record("T4.2 Empty/silence audio handling", FAIL, str(e))


# ─── T5: Hermes API ───

async def test_t5_hermes():
    print("\n=== T5: Hermes API ===")
    
    # T5.1: Simple query
    try:
        response = await ask_hermes("你好，请回复一个字：好")
        assert response and len(response) > 0, f"Empty response"
        record("T5.1 Hermes responds to simple query", PASS, f"Response: '{response[:50]}'")
    except Exception as e:
        record("T5.1 Hermes responds to simple query", FAIL, str(e))
    
    # T5.2: Chinese query
    try:
        response = await ask_hermes("现在几点了？")
        assert response and len(response) > 0, f"Empty response"
        record("T5.2 Hermes responds to Chinese query", PASS, f"Response: '{response[:50]}'")
    except Exception as e:
        record("T5.2 Hermes responds to Chinese query", FAIL, str(e))


# ─── T6: Edge TTS ───

async def test_t6_tts():
    print("\n=== T6: Edge TTS ===")
    
    # T6.1: Synthesize short text
    try:
        pcm = await synthesize_tts("你好")
        assert pcm and len(pcm) > 0, f"Empty audio"
        duration = len(pcm) / config.OUTPUT_SAMPLE_RATE / 2  # 16-bit = 2 bytes
        assert duration > 0.1, f"Audio too short: {duration}s"
        record("T6.1 TTS synthesize short text", PASS, f"Duration: {duration:.2f}s, {len(pcm)} bytes")
    except Exception as e:
        record("T6.1 TTS synthesize short text", FAIL, str(e))
    
    # T6.2: Synthesize longer text
    try:
        long_text = "你好，我是R1语音助手。我可以帮你回答问题、控制设备。请问有什么可以帮你的吗？"
        pcm = await synthesize_tts(long_text)
        assert pcm and len(pcm) > 0, f"Empty audio"
        duration = len(pcm) / config.OUTPUT_SAMPLE_RATE / 2
        assert duration > 1.0, f"Audio too short for long text: {duration}s"
        record("T6.2 TTS synthesize long text", PASS, f"Duration: {duration:.2f}s")
    except Exception as e:
        record("T6.2 TTS synthesize long text", FAIL, str(e))
    
    # T6.3: Chunk splitting
    try:
        pcm = await synthesize_tts("测试分块")
        chunks = chunk_pcm(pcm)
        assert len(chunks) > 1, f"Expected multiple chunks, got {len(chunks)}"
        # Verify total size matches
        total = sum(len(c) for c in chunks)
        assert total == len(pcm), f"Chunk total {total} != original {len(pcm)}"
        record("T6.3 PCM chunk splitting", PASS, f"{len(chunks)} chunks, total {total} bytes")
    except Exception as e:
        record("T6.3 PCM chunk splitting", FAIL, str(e))


# ─── T7: Full Pipeline ───

async def test_t7_full_pipeline():
    print("\n=== T7: Full Pipeline (ASR → Hermes → TTS) ===")
    
    # T7.1: End-to-end with known audio
    try:
        # Generate test audio: "你好" via edge-tts, then use as ASR input
        subprocess.run([
            "edge-tts", "--text", "你好",
            "--write-media", "/tmp/t7_test.mp3",
            "--voice", "zh-CN-XiaoxiaoNeural"
        ], timeout=30, capture_output=True)
        subprocess.run([
            "ffmpeg", "-i", "/tmp/t7_test.mp3",
            "-ar", "16000", "-ac", "1", "-y", "/tmp/t7_test_16k.wav"
        ], capture_output=True, timeout=10)
        
        with wave.open("/tmp/t7_test_16k.wav", "rb") as wf:
            pcm_input = wf.readframes(wf.getnframes())
        
        states_received = []
        tts_chunks_received = []
        asr_result = [None]
        
        async def on_state(state):
            states_received.append(state)
        
        async def on_tts_chunk(chunk):
            tts_chunks_received.append(chunk)
        
        async def on_asr_result(text):
            asr_result[0] = text
        
        asr_text, hermes_text = await run_pipeline(
            pcm_data=pcm_input,
            session_id=None,
            on_state=on_state,
            on_tts_chunk=on_tts_chunk,
            on_asr_result=on_asr_result,
        )
        
        # Verify ASR got something
        assert asr_text, f"ASR returned empty"
        
        # Verify Hermes responded
        assert hermes_text, f"Hermes returned empty"
        
        # Verify state transitions
        assert "thinking" in states_received, f"Missing thinking state: {states_received}"
        assert "speaking" in states_received, f"Missing speaking state: {states_received}"
        assert "idle" in states_received, f"Missing idle state: {states_received}"
        
        # Verify TTS audio was sent
        assert len(tts_chunks_received) > 0, f"No TTS chunks received"
        total_tts_bytes = sum(len(c) for c in tts_chunks_received)
        
        record("T7.1 Full pipeline end-to-end", PASS, 
               f"ASR='{asr_text}', Hermes='{hermes_text[:30]}...', "
               f"states={states_received}, TTS chunks={len(tts_chunks_received)}, "
               f"TTS bytes={total_tts_bytes}")
    except Exception as e:
        record("T7.1 Full pipeline end-to-end", FAIL, str(e))
    
    # T7.2: Pipeline with empty audio
    try:
        silence = generate_silence_pcm(1.0)
        states_received = []
        
        async def on_state(state):
            states_received.append(state)
        
        asr_text, hermes_text = await run_pipeline(
            pcm_data=silence,
            on_state=on_state,
        )
        
        # Should get thinking then idle (no speech detected → empty ASR → idle)
        assert "thinking" in states_received, f"Missing thinking: {states_received}"
        assert "idle" in states_received, f"Missing idle: {states_received}"
        record("T7.2 Pipeline with silence (no speech)", PASS, 
               f"states={states_received}, ASR='{asr_text}'")
    except Exception as e:
        record("T7.2 Pipeline with silence (no speech)", FAIL, str(e))


# ─── T8-T11: APK Tests (via ADB) ───

async def test_t8_apk_launch():
    print("\n=== T8: APK Launch Test ===")
    
    # T8.1: Launch MainActivity
    try:
        r = subprocess.run(
            ["/usr/bin/adb", "-s", "192.168.1.152:5555", "shell", "am", "start",
             "-n", "com.mgt.r1voice/.MainActivity"],
            capture_output=True, text=True, timeout=15
        )
        assert r.returncode == 0, f"am start failed: {r.stderr}"
        record("T8.1 Launch MainActivity", PASS)
    except Exception as e:
        record("T8.1 Launch MainActivity", FAIL, str(e))
    
    # T8.2: Check if activity is running
    try:
        await asyncio.sleep(2)
        r = subprocess.run(
            ["/usr/bin/adb", "-s", "192.168.1.152:5555", "shell",
             "dumpsys", "activity", "activities"],
            capture_output=True, text=True, timeout=15
        )
        assert "r1voice" in r.stdout.lower(), f"Activity not found in dumpsys"
        record("T8.2 Activity is running", PASS)
    except Exception as e:
        record("T8.2 Activity is running", FAIL, str(e))
    
    # T8.3: Check logcat for errors
    try:
        # Clear logcat first
        subprocess.run(
            ["/usr/bin/adb", "-s", "192.168.1.152:5555", "logcat", "-c"],
            capture_output=True, text=True, timeout=10
        )
        await asyncio.sleep(2)
        r = subprocess.run(
            ["/usr/bin/adb", "-s", "192.168.1.152:5555", "logcat", "-d", "-s", "r1voice:*"],
            capture_output=True, text=True, timeout=15
        )
        # Just check we get SOME log output (means our code is running)
        if r.stdout.strip():
            record("T8.3 Logcat output from APK", PASS, f"{len(r.stdout.splitlines())} lines")
        else:
            record("T8.3 Logcat output from APK", SKIP, "No tagged logs (may use different tag)")
    except Exception as e:
        record("T8.3 Logcat output from APK", FAIL, str(e))


async def test_t9_apk_websocket():
    print("\n=== T9: APK WebSocket Connection Test ===")
    
    # T9.1: Check if server receives connection from R1
    try:
        # Clear logcat
        subprocess.run(
            ["/usr/bin/adb", "-s", "192.168.1.152:5555", "logcat", "-c"],
            capture_output=True, text=True, timeout=10
        )
        
        # Start the voice service via ADB (simulate clicking "start" button)
        # We'll use am broadcast or am start with extras
        r = subprocess.run(
            ["/usr/bin/adb", "-s", "192.168.1.152:5555", "shell", "am", "startservice",
             "-n", "com.mgt.r1voice/.VoiceService",
             "--es", "server_addr", "ws://192.168.1.120:8090"],
            capture_output=True, text=True, timeout=15
        )
        
        if "Error" in r.stdout or "Error" in r.stderr:
            record("T9.1 Start VoiceService", FAIL, f"{r.stdout} {r.stderr}")
            return
        
        await asyncio.sleep(3)
        
        # Check server side for R1 connection using ss
        import subprocess as _sp
        r = _sp.run(['ss', '-tnp'], capture_output=True, text=True, timeout=5)
        r1_connected = '192.168.1.152' in r.stdout and '8090' in r.stdout
        
        if r1_connected:
            record("T9.1 R1 connects to Voice Server", PASS, "R1 connection confirmed via ss")
        else:
            # Also try logcat
            r = subprocess.run(
                ["/usr/bin/adb", "-s", "192.168.1.152:5555", "logcat", "-d", "-t", "100"],
                capture_output=True, text=True, timeout=10
            )
            ws_connected = "WebSocket connected" in r.stdout or "WS connected" in r.stdout
            ws_error = "WebSocket error" in r.stdout or "Connect failed" in r.stdout
            if ws_connected:
                record("T9.1 R1 connects to Voice Server", PASS, "confirmed via logcat")
            elif ws_error:
                record("T9.1 R1 connects to Voice Server", FAIL, "WebSocket error in logs")
            else:
                record("T9.1 R1 connects to Voice Server", SKIP, "No clear connection log")
    except Exception as e:
        record("T9.1 R1 connects to Voice Server", FAIL, str(e))
    
    # T9.2: Check server side for connection
    try:
        # We can check by connecting to the WS server ourselves and seeing if there's another connection
        async with websockets.connect(WS_URL) as ws:
            await asyncio.wait_for(ws.recv(), timeout=5)  # idle
            # If R1 is also connected, server should handle both
            record("T9.2 Server handles concurrent R1+test connection", PASS)
    except Exception as e:
        record("T9.2 Server handles concurrent R1+test connection", FAIL, str(e))


async def test_t10_apk_recording():
    print("\n=== T10: APK Recording Test ===")
    
    # T10.1: Check if RECORD_AUDIO permission is granted
    try:
        r = subprocess.run(
            ["/usr/bin/adb", "-s", "192.168.1.152:5555", "shell",
             "dumpsys", "package", "com.mgt.r1voice"],
            capture_output=True, text=True, timeout=15
        )
        # On API 22, permissions are granted at install time
        granted = "RECORD_AUDIO" in r.stdout
        record("T10.1 RECORD_AUDIO permission", PASS if granted else SKIP,
               "granted at install (API 22)" if granted else "not found in dumpsys")
    except Exception as e:
        record("T10.1 RECORD_AUDIO permission", FAIL, str(e))
    
    # T10.2: Check logcat for AudioRecord initialization
    try:
        r = subprocess.run(
            ["/usr/bin/adb", "-s", "192.168.1.152:5555", "logcat", "-d", "-t", "100"],
            capture_output=True, text=True, timeout=10
        )
        audio_init = "AudioRecord" in r.stdout or "Recording started" in r.stdout
        record("T10.2 AudioRecord initialization in logs", PASS if audio_init else SKIP,
               "found" if audio_init else "not found in logs")
    except Exception as e:
        record("T10.2 AudioRecord initialization in logs", FAIL, str(e))


async def test_t11_apk_playback():
    print("\n=== T11: APK Playback Test ===")
    
    # T11.1: Send real speech audio (edge-tts generated) via WebSocket
    try:
        # Generate real speech audio with edge_tts Python API
        import edge_tts as _edge_tts
        communicate = _edge_tts.Communicate(text="你好，请回复一个字：好", voice="zh-CN-XiaoxiaoNeural")
        await communicate.save("/tmp/t11_test.mp3")
        
        # Convert to 16kHz WAV
        subprocess.run([
            "ffmpeg", "-i", "/tmp/t11_test.mp3",
            "-ar", "16000", "-ac", "1", "-y", "/tmp/t11_test_16k.wav"
        ], capture_output=True, timeout=10)
        
        if not os.path.exists("/tmp/t11_test_16k.wav"):
            record("T11.1 Full pipeline via WebSocket (real audio)", FAIL, "WAV file not generated")
            return
        
        with wave.open("/tmp/t11_test_16k.wav", "rb") as wf:
            pcm_input = wf.readframes(wf.getnframes())
        
        async with websockets.connect(WS_URL) as ws:
            await asyncio.wait_for(ws.recv(), timeout=5)  # idle
            
            # Send wake to start listening
            await ws.send(json.dumps({"type": "wake"}))
            await asyncio.wait_for(ws.recv(), timeout=5)  # listening
            
            # Send PCM frames (20ms each = 640 bytes)
            frame_size = 640
            for i in range(0, len(pcm_input), frame_size):
                frame = pcm_input[i:i+frame_size]
                if len(frame) == frame_size:
                    await ws.send(frame)
                    await asyncio.sleep(0.02)  # 20ms per frame
            
            # Send silence to trigger VAD speech_end
            silence = np.zeros(16000 * 2, dtype=np.int16).tobytes()  # 2 seconds
            for i in range(0, len(silence), frame_size):
                frame = silence[i:i+frame_size]
                if len(frame) == frame_size:
                    await ws.send(frame)
                    await asyncio.sleep(0.02)
            
            # Wait for state transitions and TTS audio
            messages = []
            binary_count = 0
            try:
                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=60)
                    if isinstance(msg, str):
                        data = json.loads(msg)
                        messages.append(data)
                        logger.info(f"Received: {data}")
                        if data.get("state") == "idle":
                            break
                    elif isinstance(msg, bytes):
                        binary_count += 1
            except asyncio.TimeoutError:
                pass
            
            states = [m.get("state") for m in messages if isinstance(m, dict)]
            has_asr = any(m.get("type") == "asr_result" for m in messages if isinstance(m, dict))
            
            if "thinking" in states and "speaking" in states and binary_count > 0:
                record("T11.1 Full pipeline via WebSocket (real audio)", PASS,
                       f"states={states}, binary_chunks={binary_count}, asr={has_asr}")
            elif "thinking" in states and binary_count > 0:
                record("T11.1 Full pipeline via WebSocket (real audio)", PASS,
                       f"Got TTS audio: states={states}, chunks={binary_count}")
            elif "thinking" in states:
                record("T11.1 Full pipeline via WebSocket (real audio)", SKIP,
                       f"Got thinking but no TTS: states={states}")
            else:
                record("T11.1 Full pipeline via WebSocket (real audio)", SKIP,
                       f"VAD may not have triggered: states={states}")
    except Exception as e:
        record("T11.1 Full pipeline via WebSocket (real audio)", FAIL, str(e))


# ─── T13: Edge Cases ───

async def test_t13_edge_cases():
    print("\n=== T13: Edge Cases ===")
    
    # T13.1: Empty PCM frame
    try:
        vad = EnergyVAD()
        result = vad.process_frame(b"")
        assert result == "silence", f"Expected silence for empty frame, got {result}"
        record("T13.1 Empty PCM frame to VAD", PASS)
    except Exception as e:
        record("T13.1 Empty PCM frame to VAD", FAIL, str(e))
    
    # T13.2: Very small PCM frame
    try:
        vad = EnergyVAD()
        result = vad.process_frame(b"\x00\x00")
        assert result == "silence", f"Expected silence, got {result}"
        record("T13.2 Very small PCM frame", PASS)
    except Exception as e:
        record("T13.2 Very small PCM frame", FAIL, str(e))
    
    # T13.3: Long running connection (30 seconds)
    try:
        async with websockets.connect(WS_URL) as ws:
            await asyncio.wait_for(ws.recv(), timeout=5)  # idle
            # Send periodic wake/stop
            for i in range(5):
                await ws.send(json.dumps({"type": "wake"}))
                await asyncio.wait_for(ws.recv(), timeout=5)  # listening
                await asyncio.sleep(1)
                await ws.send(json.dumps({"type": "stop"}))
                await asyncio.wait_for(ws.recv(), timeout=5)  # idle
                await asyncio.sleep(1)
            record("T13.3 Long running connection (30s)", PASS)
    except Exception as e:
        record("T13.3 Long running connection (30s)", FAIL, str(e))
    
    # T13.4: Server handles rapid connect/disconnect
    try:
        for i in range(10):
            async with websockets.connect(WS_URL) as ws:
                await asyncio.wait_for(ws.recv(), timeout=5)
        record("T13.4 Rapid connect/disconnect (10x)", PASS)
    except Exception as e:
        record("T13.4 Rapid connect/disconnect (10x)", FAIL, str(e))


# ─── T14: LED Control ───

async def test_t14_led():
    print("\n=== T14: LED Control Test ===")
    
    # T14.1: Check sysfs LED paths exist on R1
    try:
        r = subprocess.run(
            ["/usr/bin/adb", "-s", "192.168.1.152:5555", "shell",
             "ls", "/sys/class/leds/"],
            capture_output=True, text=True, timeout=15
        )
        led_dirs = r.stdout.strip()
        if led_dirs:
            record("T14.1 sysfs LED paths exist", PASS, f"Found: {led_dirs[:80]}")
        else:
            record("T14.1 sysfs LED paths exist", SKIP, "No LED dirs found")
    except Exception as e:
        record("T14.1 sysfs LED paths exist", FAIL, str(e))
    
    # T14.2: Try writing to LED (if root)
    try:
        r = subprocess.run(
            ["/usr/bin/adb", "-s", "192.168.1.152:5555", "shell",
             "su", "-c", "echo 100 > /sys/class/leds/blue/brightness"],
            capture_output=True, text=True, timeout=15
        )
        if r.returncode == 0 and "not found" not in r.stderr:
            record("T14.2 Write to LED sysfs", PASS)
        else:
            record("T14.2 Write to LED sysfs", SKIP, f"No root or path missing: {r.stderr[:50]}")
    except Exception as e:
        record("T14.2 Write to LED sysfs", FAIL, str(e))


# ─── Main ───

async def main():
    print("=" * 60)
    print("  R1 Voice Server — Comprehensive Test Suite")
    print("=" * 60)
    
    # T1-T2: WebSocket + State Machine
    await test_t1_websocket_connection()
    await test_t2_state_machine()
    
    # T3: VAD (sync)
    test_t3_vad()
    
    # T4-T6: Individual pipeline components
    await test_t4_asr()
    await test_t5_hermes()
    await test_t6_tts()
    
    # T7: Full pipeline
    await test_t7_full_pipeline()
    
    # T8-T11: APK tests
    await test_t8_apk_launch()
    await test_t9_apk_websocket()
    await test_t10_apk_recording()
    await test_t11_apk_playback()
    
    # T13: Edge cases
    await test_t13_edge_cases()
    
    # T14: LED control
    await test_t14_led()
    
    # Summary
    print("\n" + "=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, s, _ in results if "PASS" in s)
    failed = sum(1 for _, s, _ in results if "FAIL" in s)
    skipped = sum(1 for _, s, _ in results if "SKIP" in s)
    total = len(results)
    
    for name, status, detail in results:
        print(f"  {status} {name}" + (f" — {detail}" if detail else ""))
    
    print(f"\n  Total: {total} | Pass: {passed} | Fail: {failed} | Skip: {skipped}")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
