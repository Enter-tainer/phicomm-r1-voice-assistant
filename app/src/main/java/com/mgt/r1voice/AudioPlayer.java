package com.mgt.r1voice;

import android.media.AudioFormat;
import android.media.AudioTrack;
import android.util.Log;

/**
 * AudioPlayer — streams 48kHz 16-bit mono PCM to speaker.
 *
 * Uses AudioTrack in STREAM mode for low-latency playback.
 */
public class AudioPlayer {

    private static final String TAG = "AudioPlayer";

    // Output format — must match server TTS output
    private static final int SAMPLE_RATE = 48000;
    private static final int CHANNEL_CONFIG = AudioFormat.CHANNEL_OUT_MONO;
    private static final int AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT;

    private AudioTrack audioTrack;
    private boolean isPlaying = false;

    public boolean start() {
        if (isPlaying) return true;

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
        Log.i(TAG, "Playback started, bufferSize=" + bufferSize);
        return true;
    }

    /**
     * Write PCM data to the speaker. Called for each chunk received from server.
     */
    public void writePcm(byte[] data) {
        if (isPlaying && audioTrack != null) {
            audioTrack.write(data, 0, data.length);
        }
    }

    public void stop() {
        isPlaying = false;
        if (audioTrack != null) {
            try {
                audioTrack.stop();
            } catch (Exception e) {
                Log.w(TAG, "Error stopping AudioTrack", e);
            }
            audioTrack.release();
            audioTrack = null;
        }
        Log.i(TAG, "Playback stopped");
    }

    public boolean isPlaying() {
        return isPlaying;
    }
}
