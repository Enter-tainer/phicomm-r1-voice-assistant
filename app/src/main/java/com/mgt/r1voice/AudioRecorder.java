package com.mgt.r1voice;

import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.util.Log;

/**
 * AudioRecorder — captures 16kHz 16-bit mono PCM from microphone.
 *
 * Streams audio frames to a callback for WebSocket transmission.
 * Can reuse an existing AudioRecord (from WakeWordDetector) to avoid
 * AudioFlinger deadlock on R1 when release+create happens too fast.
 */
public class AudioRecorder {

    private static final String TAG = "AudioRecorder";

    // Audio config — must match server expectations
    private static final int SAMPLE_RATE = 16000;
    private static final int CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO;
    private static final int AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT;

    // Frame size: 80ms = 1280 samples * 2 bytes = 2560 bytes
    // Matches openWakeWord's expected input size for optimal detection
    private static final int FRAME_SIZE_MS = 80;
    private static final int FRAME_SIZE = SAMPLE_RATE * FRAME_SIZE_MS / 1000 * 2; // 2560 bytes

    private AudioRecord audioRecord;
    private boolean isRecording = false;
    private boolean ownsAudioRecord = false; // true if we created it, false if reused
    private Thread recordThread;

    public interface AudioFrameCallback {
        void onAudioFrame(byte[] data);
    }

    private AudioFrameCallback callback;

    public AudioRecorder(AudioFrameCallback callback) {
        this.callback = callback;
    }

    /**
     * Start recording using an existing AudioRecord (from WakeWordDetector).
     * This avoids creating a new AudioRecord which can deadlock AudioFlinger on R1.
     */
    public boolean startWithExistingRecord(AudioRecord existingRecord) {
        if (isRecording) return true;
        if (existingRecord == null) {
            Log.e(TAG, "Existing AudioRecord is null");
            return false;
        }

        audioRecord = existingRecord;
        ownsAudioRecord = false;
        isRecording = true;

        // AudioRecord should already be recording from WakeWordDetector
        // Just start reading from it in a new thread
        recordThread = new Thread(new Runnable() {
            @Override
            public void run() {
                byte[] buffer = new byte[FRAME_SIZE];
                while (isRecording) {
                    int read = audioRecord.read(buffer, 0, FRAME_SIZE);
                    if (read > 0 && callback != null) {
                        byte[] frame = new byte[read];
                        System.arraycopy(buffer, 0, frame, 0, read);
                        callback.onAudioFrame(frame);
                    }
                }
            }
        }, "AudioRecorder-Thread");
        recordThread.start();

        Log.i(TAG, "Recording started (reusing AudioRecord), frameSize=" + FRAME_SIZE);
        return true;
    }

    public boolean start() {
        if (isRecording) return true;

        int minBuffer = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT);
        int bufferSize = Math.max(minBuffer, FRAME_SIZE * 4);

        try {
            audioRecord = new AudioRecord(
                MediaRecorder.AudioSource.VOICE_RECOGNITION,
                SAMPLE_RATE,
                CHANNEL_CONFIG,
                AUDIO_FORMAT,
                bufferSize
            );
        } catch (Exception e) {
            Log.e(TAG, "Failed to create AudioRecord", e);
            return false;
        }

        if (audioRecord.getState() != AudioRecord.STATE_INITIALIZED) {
            Log.e(TAG, "AudioRecord not initialized");
            audioRecord.release();
            audioRecord = null;
            return false;
        }

        ownsAudioRecord = true;
        isRecording = true;
        audioRecord.startRecording();

        recordThread = new Thread(new Runnable() {
            @Override
            public void run() {
                byte[] buffer = new byte[FRAME_SIZE];
                while (isRecording) {
                    int read = audioRecord.read(buffer, 0, FRAME_SIZE);
                    if (read > 0 && callback != null) {
                        byte[] frame = new byte[read];
                        System.arraycopy(buffer, 0, frame, 0, read);
                        callback.onAudioFrame(frame);
                    }
                }
            }
        }, "AudioRecorder-Thread");
        recordThread.start();

        Log.i(TAG, "Recording started, frameSize=" + FRAME_SIZE);
        return true;
    }

    public void stop() {
        isRecording = false;
        if (recordThread != null) {
            try {
                recordThread.join(1000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            recordThread = null;
        }
        // Only release if we own the AudioRecord
        // If reused from WakeWordDetector, leave it alive
        if (ownsAudioRecord && audioRecord != null) {
            try {
                audioRecord.stop();
            } catch (Exception e) {
                Log.w(TAG, "Error stopping AudioRecord", e);
            }
            audioRecord.release();
            audioRecord = null;
        }
        // Don't null out audioRecord if we don't own it — WakeWordDetector still needs it
        if (!ownsAudioRecord) {
            audioRecord = null; // just clear our reference, don't release
        }
        Log.i(TAG, "Recording stopped");
    }

    public boolean isRecording() {
        return isRecording;
    }
}
