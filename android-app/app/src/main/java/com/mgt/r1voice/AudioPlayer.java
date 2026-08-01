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
 * buffer is full, and AudioTrack.stop()/release() can also block on this
 * device (RK3229, Android 5.1). Previously writePcm()/stop() ran directly on
 * the WS read thread — so when TTS finished, the read thread could stall in
 * write()/stop() and never process the server's "tts_done"/"state idle"
 * messages. The app therefore never set shouldStreamMic = true again, the mic
 * stream stayed muted, the server saw no frames, and the watchdog rebooted
 * the R1 right after every long answer.
 *
 * Fix: all AudioTrack calls run on a dedicated playback thread. writePcm()
 * only enqueues (never blocks the caller); stop() signals the thread and
 * returns immediately.
 */
public class AudioPlayer {

    private static final String TAG = "AudioPlayer";

    // Output format — must match server TTS output
    private static final int SAMPLE_RATE = 48000;
    private static final int CHANNEL_CONFIG = AudioFormat.CHANNEL_OUT_MONO;
    private static final int AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT;

    // Max PCM we queue before dropping (≈0.5s). Protects against unbounded
    // queue growth if the network outruns the speaker for a moment.
    private static final int MAX_QUEUE_BYTES = 24000;

    private final Object startLock = new Object();
    private AudioTrack audioTrack;
    private volatile boolean isPlaying = false;
    private volatile boolean stopRequested = false;
    private final BlockingQueue<byte[]> queue = new LinkedBlockingQueue<>();
    private int queuedBytes = 0;
    private Thread playbackThread;

    public boolean start() {
        synchronized (startLock) {
            if (isPlaying) return true;

            // If a previous playback thread is still winding down (stop() is
            // async), wait briefly for it to release the old AudioTrack before
            // creating a new one. Prevents two AudioTracks on the HAL at once
            // (a known trigger for the RK3229 AudioFlinger deadlock) and a
            // stale-thread race on `audioTrack`.
            Thread old = playbackThread;
            if (old != null && old.isAlive() && old != Thread.currentThread()) {
                try {
                    old.join(500);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }

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
                return false;
            }

            if (audioTrack.getState() != AudioTrack.STATE_INITIALIZED) {
                Log.e(TAG, "AudioTrack not initialized");
                audioTrack.release();
                audioTrack = null;
                return false;
            }

            audioTrack.play();
            isPlaying = true;
            stopRequested = false;

            playbackThread = new Thread(this::playbackLoop, "AudioPlayer-Playback");
            playbackThread.start();
            Log.i(TAG, "Playback started, bufferSize=" + bufferSize);
            return true;
        }
    }

    /**
     * Queue PCM data for playback. Non-blocking — safe to call from the WS
     * read thread. Returns false if the queue is over its cap (data dropped).
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
     * Stop playback and return immediately. Non-blocking — safe to call from
     * the WS read thread. The playback thread drains and releases AudioTrack.
     */
    public void stop() {
        stopRequested = true;
        isPlaying = false;
        synchronized (queue) {
            queue.clear();
            queuedBytes = 0;
        }
        Thread t = playbackThread;
        if (t != null && t != Thread.currentThread()) {
            t.interrupt();
        }
        Log.i(TAG, "Playback stop requested");
    }

    /** Called when a new TTS utterance starts; drops any stale queued audio. */
    public void reset() {
        synchronized (queue) {
            queue.clear();
            queuedBytes = 0;
        }
    }

    public boolean isPlaying() {
        return isPlaying;
    }

    // === Playback thread ====================================================

    private void playbackLoop() {
        try {
            while (!stopRequested && isPlaying) {
                byte[] data;
                try {
                    data = queue.poll(200, TimeUnit.MILLISECONDS);
                } catch (InterruptedException e) {
                    break; // stop() requested
                }
                if (data == null) {
                    continue; // idle tick
                }
                synchronized (queue) {
                    queuedBytes -= data.length;
                    if (queuedBytes < 0) queuedBytes = 0;
                }
                if (audioTrack != null && !stopRequested) {
                    try {
                        audioTrack.write(data, 0, data.length);
                    } catch (Exception e) {
                        Log.w(TAG, "AudioTrack.write error", e);
                        break;
                    }
                }
            }
        } finally {
            releaseAudioTrack();
        }
    }

    private void releaseAudioTrack() {
        AudioTrack track = audioTrack;
        audioTrack = null;
        isPlaying = false;
        if (track != null) {
            try {
                track.stop();
            } catch (Exception e) {
                Log.w(TAG, "Error stopping AudioTrack", e);
            }
            try {
                track.release();
            } catch (Exception e) {
                Log.w(TAG, "Error releasing AudioTrack", e);
            }
        }
        Log.i(TAG, "Playback stopped (thread exited)");
    }
}
