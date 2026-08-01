package com.mgt.r1voice;

import android.media.AudioFormat;
import android.media.AudioTrack;
import android.util.Log;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

/**
 * AudioPlayer — streams 48kHz 16-bit mono PCM to speaker.
 *
 * Uses AudioTrack in STREAM mode for low-latency playback.
 *
 * THREAD MODEL (critical, fixed 2026-08-01):
 * The Java-WebSocket library delivers onMessage() callbacks on its single
 * read thread. AudioTrack.write() in STREAM mode BLOCKS when the internal
 * buffer is full. All AudioTrack calls therefore run on a DEDICATED playback
 * thread; writePcm() only enqueues and never blocks the caller.
 *
 * DESIGN (rev 2): the AudioTrack and the playback thread are created ONCE
 * and reused for the app's lifetime. start()/stop() only play()/pause() +
 * clear the queue — they never create/release AudioTrack and never join the
 * playback thread. This avoids:
 *   1. The WS read thread stalling in start() while waiting for a previous
 *      stop() to release AudioTrack (RK3229 stop()/release() can be slow) —
 *      which delayed the wake beep and cut it off mid-play.
 *   2. Rapid AudioTrack create/release cycles, a known trigger for the R1's
 *      AudioFlinger deadlock.
 */
public class AudioPlayer {

    private static final String TAG = "AudioPlayer";

    // Output format — must match server TTS output
    private static final int SAMPLE_RATE = 48000;
    private static final int CHANNEL_CONFIG = AudioFormat.CHANNEL_OUT_MONO;
    private static final int AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT;

    // Max PCM we queue before dropping (≈1s). Server paces TTS at ≤1.33x
    // realtime (15ms sleep per 20ms chunk), so a ~1s queue absorbs the
    // 0.33s/s headroom safely without ever dropping audio.
    private static final int MAX_QUEUE_BYTES = 96000;

    private final Object stateLock = new Object();
    private AudioTrack audioTrack;
    private volatile boolean isPlaying = false;
    private volatile boolean audioTrackFailed = false;
    private final BlockingQueue<byte[]> queue = new LinkedBlockingQueue<>();
    private int queuedBytes = 0;
    private Thread playbackThread;

    /** Create the AudioTrack + playback thread (once). Returns true on success. */
    public boolean init() {
        synchronized (stateLock) {
            if (audioTrack != null) return true;

            int minBuffer = AudioTrack.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT);
            int bufferSize = Math.max(minBuffer, 4096);
            try {
                audioTrack = new AudioTrack(
                    android.media.AudioManager.STREAM_SYSTEM,
                    SAMPLE_RATE,
                    CHANNEL_CONFIG,
                    AUDIO_FORMAT,
                    bufferSize,
                    AudioTrack.MODE_STREAM
                );
            } catch (Exception e) {
                Log.e(TAG, "Failed to create AudioTrack", e);
                audioTrackFailed = true;
                return false;
            }
            if (audioTrack.getState() != AudioTrack.STATE_INITIALIZED) {
                Log.e(TAG, "AudioTrack not initialized");
                try { audioTrack.release(); } catch (Exception ignored) {}
                audioTrack = null;
                audioTrackFailed = true;
                return false;
            }

            audioTrackFailed = false;
            playbackThread = new Thread(this::playbackLoop, "AudioPlayer-Playback");
            playbackThread.start();
            Log.i(TAG, "AudioTrack created, bufferSize=" + bufferSize);
            return true;
        }
    }

    /**
     * Start/resume playback. Non-blocking: clears stale queued audio and
     * calls AudioTrack.play(). Safe to call from the WS read thread.
     */
    public boolean start() {
        if (audioTrackFailed) {
            Log.w(TAG, "start() called but AudioTrack previously failed");
            return false;
        }
        synchronized (stateLock) {
            if (audioTrack == null && !init()) return false;
            clearQueue();
            isPlaying = true;
            if (audioTrack != null && audioTrack.getPlayState() != AudioTrack.PLAYSTATE_PLAYING) {
                try {
                    audioTrack.play();
                } catch (Exception e) {
                    Log.w(TAG, "AudioTrack.play error", e);
                }
            }
            return true;
        }
    }

    /**
     * Queue PCM data for playback. Non-blocking — safe to call from the WS
     * read thread. Returns false if not playing or the queue is over its cap.
     */
    public boolean writePcm(byte[] data) {
        if (!isPlaying || data == null || data.length == 0) {
            return false;
        }
        synchronized (queue) {
            if (queuedBytes + data.length > MAX_QUEUE_BYTES) {
                // Speaker is behind; drop rather than let the queue grow
                // unbounded (and never block the WS read thread).
                Log.w(TAG, "playback queue full, dropping " + data.length + " bytes");
                return false;
            }
            queuedBytes += data.length;
        }
        queue.offer(data);
        return true;
    }

    /**
     * Stop playback and return immediately. Non-blocking: pauses AudioTrack,
     * clears the queue; the AudioTrack + thread are kept for reuse.
     */
    public void stop() {
        isPlaying = false;
        synchronized (stateLock) {
            clearQueue();
            if (audioTrack != null) {
                try {
                    audioTrack.pause();
                } catch (Exception e) {
                    Log.w(TAG, "AudioTrack.pause error", e);
                }
            }
        }
        Log.i(TAG, "Playback stop requested");
    }

    /** Drop any stale queued audio (e.g. at the start of a new utterance). */
    public void reset() {
        clearQueue();
    }

    public boolean isPlaying() {
        return isPlaying;
    }

    /** Release everything (only called from VoiceService.onDestroy). */
    public void release() {
        stop();
        synchronized (stateLock) {
            Thread t = playbackThread;
            playbackThread = null;
            if (t != null && t != Thread.currentThread()) {
                t.interrupt();
                try {
                    t.join(1000);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
            if (audioTrack != null) {
                try { audioTrack.stop(); } catch (Exception ignored) {}
                try { audioTrack.release(); } catch (Exception ignored) {}
                audioTrack = null;
            }
        }
        Log.i(TAG, "AudioPlayer released");
    }

    private void clearQueue() {
        synchronized (queue) {
            queue.clear();
            queuedBytes = 0;
        }
    }

    // === Playback thread ====================================================

    private void playbackLoop() {
        while (true) {
            if (Thread.currentThread().isInterrupted()) {
                break;
            }
            byte[] data;
            try {
                data = queue.poll(200, TimeUnit.MILLISECONDS);
            } catch (InterruptedException e) {
                break; // release() requested
            }
            if (data == null) {
                continue; // idle tick (thread stays alive for reuse)
            }
            synchronized (queue) {
                queuedBytes -= data.length;
                if (queuedBytes < 0) queuedBytes = 0;
            }
            if (!isPlaying) {
                continue; // stop() happened while queued
            }
            if (audioTrack != null) {
                try {
                    audioTrack.write(data, 0, data.length);
                } catch (Exception e) {
                    Log.w(TAG, "AudioTrack.write error", e);
                    break;
                }
            }
        }
        Log.i(TAG, "Playback thread exited");
    }
}
