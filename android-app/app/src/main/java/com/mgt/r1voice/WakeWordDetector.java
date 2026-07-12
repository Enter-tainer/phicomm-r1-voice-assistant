package com.mgt.r1voice;

import android.content.Context;
import android.content.res.AssetManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.AudioTrack;
import android.media.MediaRecorder;
import android.media.ToneGenerator;
import android.media.AudioManager;
import android.util.Log;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtException;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;

/**
 * OpenWakeWord detector — runs entirely on-device using ONNX Runtime.
 *
 * Pipeline (ported from openWakeWord Python):
 *   1. Audio: 16kHz 16-bit mono PCM, 1280 samples (80ms) per frame
 *   2. melspectrogram.onnx: [1, 1280] → [N, 1, N, 32] melspec
 *   3. Transform: spec = spec / 10 + 2
 *   4. Sliding window: 76 frames, step 8 → embedding model input
 *   5. embedding_model.onnx: [batch, 76, 32, 1] → [batch, 1, 1, 96]
 *   6. Accumulate embeddings into feature_buffer (max 120 frames)
 *   7. hey_jarvis.onnx: [1, 16, 96] → [1, 1] score
 *   8. score > 0.15 → wake word detected
 *
 * CRITICAL: Buffers use flat float[] arrays, NOT float[][].
 * System.arraycopy on float[][] copies REFERENCES, not values. After repeated
 * shifts, all rows alias to the same float[] → all 76 melspec frames become
 * identical → score=0 forever. Flat arraycopy copies values correctly.
 */
public class WakeWordDetector {

    private static final String TAG = "OpenWakeWord";

    // Audio constants
    private static final int SAMPLE_RATE = 16000;
    private static final int FRAME_SAMPLES = 1280; // 80ms @ 16kHz
    private static final int MELSPEC_WINDOW = 76;
    private static final int MELSPEC_STEP = 8;
    private static final int MEL_BINS = 32;
    private static final int EMBEDDING_DIM = 96;
    private static final int FEATURE_BUFFER_MAX = 120;
    private static final int WAKEWORD_INPUT_FRAMES = 16;

    // Detection threshold (lowered for loopback/mic testing)
    private static final float DETECTION_THRESHOLD = 0.15f;

    private final Context context;
    private WakeWordListener listener;

    // ONNX Runtime
    private OrtEnvironment env;
    private OrtSession melspecSession;
    private OrtSession embeddingSession;
    private OrtSession wakewordSession;
    private String melspecInputName, melspecOutputName;
    private String embeddingInputName, embeddingOutputName;
    private String wakewordInputName, wakewordOutputName;

    // Buffers — FLAT float[] arrays to avoid reference aliasing bug
    // (System.arraycopy on float[][] copies references, not values)
    private float[] melspecFlat;        // [MELSPEC_MAX_FRAMES * MEL_BINS]
    private int melspecBufferSize = 0;  // number of frames
    private static final int MELSPEC_MAX_FRAMES = 970;
    private static final int MELSPEC_STRIDE = MEL_BINS; // 32

    private float[] featureFlat;        // [FEATURE_BUFFER_MAX * EMBEDDING_DIM]
    private int featureBufferSize = 0;
    private static final int FEATURE_STRIDE = EMBEDDING_DIM; // 96

    // Audio recording
    private AudioRecord audioRecord;
    private Thread detectThread;
    private volatile boolean isRunning = false;
    private int frameCount = 0;

    // Raw audio rolling buffer — the Python library passes the last
    // (n_samples + 160*3) = 1760 samples to the melspectrogram model
    // for STFT overlap context. Without this, melspec values are wrong.
    private static final int MELSPEC_CONTEXT = 160 * 3; // 480 samples = 30ms
    private float[] rawAudioBuffer = new float[MELSPEC_MAX_FRAMES * FRAME_SAMPLES + MELSPEC_CONTEXT];
    private int rawAudioSize = 0;

    // Fixed gain — R1 mic sensitivity is very low, need ~30x boost
    private static final float FIXED_GAIN = 30.0f;

    public interface WakeWordListener {
        void onWakeWordDetected();
    }

    public WakeWordDetector(Context context, WakeWordListener listener) {
        this.context = context;
        this.listener = listener;
    }

    public void start() {
        if (isRunning) return;

        try {
            loadModels();
            // Start AudioRecord FIRST, before initBuffers.
            // initBuffers takes ~4s (ONNX noise prefill), during which Phicomm
            // can restart and grab the audio HAL. By starting AudioRecord first,
            // we claim the audio input before Phicomm can interfere.
            isRunning = true;  // set before startRecording so detection loop doesn't exit
            startRecording();
            initBuffers();
            Log.i(TAG, "OpenWakeWord started (on-device ONNX Runtime, model: hey_jarvis)");
            // Play ding to signal user: wake word detection is ready
            playDing();
        } catch (Exception e) {
            isRunning = false;
            Log.e(TAG, "Failed to start: " + e.getMessage(), e);
        }
    }

    public void stop() {
        isRunning = false;
        if (detectThread != null) {
            detectThread.interrupt();
            detectThread = null;
        }
        if (audioRecord != null) {
            try {
                audioRecord.stop();
                audioRecord.release();
            } catch (Exception e) { }
            audioRecord = null;
        }
        Log.i(TAG, "OpenWakeWord stopped");
    }

    /**
     * Pause detection loop WITHOUT releasing AudioRecord.
     * The AudioRecord stays running so it can be reused by AudioRecorder
     * (creating a new AudioRecord after releasing one causes AudioFlinger
     * deadlock on R1's Android 5.1).
     */
    public void pauseDetection() {
        isRunning = false;
        if (detectThread != null) {
            detectThread.interrupt();
            try { detectThread.join(1000); } catch (InterruptedException e) {}
            detectThread = null;
        }
        Log.i(TAG, "Detection paused (AudioRecord kept alive)");
    }

    /**
     * Resume detection loop using existing AudioRecord.
     */
    public void resumeDetection() {
        if (isRunning) return;
        if (audioRecord == null) {
            Log.e(TAG, "Cannot resume: AudioRecord is null");
            return;
        }
        // Reset buffers for fresh detection
        melspecBufferSize = 0;
        featureBufferSize = 0;
        rawAudioSize = 0;
        // Re-init with ones prefill (quick, no ONNX prefill needed)
        for (int i = 0; i < MELSPEC_WINDOW; i++) {
            int off = i * MELSPEC_STRIDE;
            for (int b = 0; b < MEL_BINS; b++) {
                melspecFlat[off + b] = 1.0f;
            }
        }
        melspecBufferSize = MELSPEC_WINDOW;
        isRunning = true;
        frameCount = 0;
        detectThread = new Thread(this::detectionLoop, "OWW-Detect");
        detectThread.start();
        Log.i(TAG, "Detection resumed");
    }

    /**
     * Get the AudioRecord instance (for reuse by AudioRecorder).
     */
    public AudioRecord getAudioRecord() {
        return audioRecord;
    }

    public boolean isRunning() {
        return isRunning;
    }

    private void loadModels() throws IOException, OrtException {
        AssetManager am = context.getAssets();

        env = OrtEnvironment.getEnvironment();

        // Melspectrogram: input "input" [batch, samples] → output "output" [time, 1, ?, 32]
        byte[] melspecBytes = loadAsset(am, "models/melspectrogram.onnx");
        melspecSession = env.createSession(melspecBytes);
        melspecInputName = melspecSession.getInputNames().iterator().next();
        melspecOutputName = melspecSession.getOutputNames().iterator().next();
        Log.i(TAG, "Melspec: input=" + melspecInputName + " output=" + melspecOutputName);

        // Embedding: input "input_1" [batch, 76, 32, 1] → output "conv2d_19" [batch, 1, 1, 96]
        byte[] embBytes = loadAsset(am, "models/embedding_model.onnx");
        embeddingSession = env.createSession(embBytes);
        embeddingInputName = embeddingSession.getInputNames().iterator().next();
        embeddingOutputName = embeddingSession.getOutputNames().iterator().next();
        Log.i(TAG, "Embedding: input=" + embeddingInputName + " output=" + embeddingOutputName);

        // Wakeword: input "x.1" [1, 16, 96] → output "53" [1, 1]
        byte[] wkwBytes = loadAsset(am, "models/hey_jarvis.onnx");
        wakewordSession = env.createSession(wkwBytes);
        wakewordInputName = wakewordSession.getInputNames().iterator().next();
        wakewordOutputName = wakewordSession.getOutputNames().iterator().next();
        Log.i(TAG, "Wakeword: input=" + wakewordInputName + " output=" + wakewordOutputName);

        Log.i(TAG, "All models loaded successfully");
    }

    private byte[] loadAsset(AssetManager am, String path) throws IOException {
        InputStream is = am.open(path);
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        byte[] buf = new byte[8192];
        int n;
        while ((n = is.read(buf)) > 0) {
            bos.write(buf, 0, n);
        }
        is.close();
        return bos.toByteArray();
    }

    private void initBuffers() {
        melspecFlat = new float[MELSPEC_MAX_FRAMES * MEL_BINS];
        melspecBufferSize = 0;
        featureFlat = new float[FEATURE_BUFFER_MAX * EMBEDDING_DIM];
        featureBufferSize = 0;

        // Pre-fill melspec buffer with 76 frames of ones (matches Python library init)
        for (int i = 0; i < MELSPEC_WINDOW; i++) {
            int off = i * MELSPEC_STRIDE;
            for (int b = 0; b < MEL_BINS; b++) {
                melspecFlat[off + b] = 1.0f;
            }
        }
        melspecBufferSize = MELSPEC_WINDOW;

        // Pre-fill feature buffer by running 4 seconds of random noise through the pipeline
        // (matches Python library: self.feature_buffer = self._get_embeddings(random_noise))
        java.util.Random rng = new java.util.Random(42);
        float[] noise = new float[16000 * 4];
        for (int i = 0; i < noise.length; i++) {
            noise[i] = (float) rng.nextInt(2000) - 1000; // -1000 to 1000
        }

        // Run noise through melspec + embedding in 1280-sample chunks
        for (int start = 0; start + MELSPEC_CONTEXT + FRAME_SAMPLES <= noise.length; start += FRAME_SAMPLES) {
            int inputLen = FRAME_SAMPLES + MELSPEC_CONTEXT;
            float[] chunk = new float[inputLen];
            System.arraycopy(noise, start, chunk, 0, inputLen);

            try {
                long[] inputShape = {1, inputLen};
                OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(chunk), inputShape);
                Map<String, OnnxTensor> inputs = new HashMap<>();
                inputs.put(melspecInputName, inputTensor);
                OrtSession.Result result = melspecSession.run(inputs);
                float[][][][] melspecOut = (float[][][][]) result.get(0).getValue();
                result.close();
                inputTensor.close();

                int nFrames = melspecOut[0][0].length;
                for (int f = 0; f < nFrames; f++) {
                    if (melspecBufferSize >= MELSPEC_MAX_FRAMES) {
                        // FLAT arraycopy — copies values, not references!
                        System.arraycopy(melspecFlat, MELSPEC_STRIDE, melspecFlat, 0, (MELSPEC_MAX_FRAMES - 1) * MELSPEC_STRIDE);
                        melspecBufferSize = MELSPEC_MAX_FRAMES - 1;
                    }
                    int off = melspecBufferSize * MELSPEC_STRIDE;
                    for (int b = 0; b < MEL_BINS; b++) {
                        melspecFlat[off + b] = melspecOut[0][0][f][b] / 10f + 2f;
                    }
                    melspecBufferSize++;
                }

                // Generate embeddings from sliding windows (1 per frame)
                for (int i = 0; i < 1; i++) {
                    int windowEnd = melspecBufferSize - MELSPEC_STEP * i;
                    if (windowEnd <= 0) break;
                    int windowStart = windowEnd - MELSPEC_WINDOW;
                    if (windowStart < 0) break;

                    float[] embFlat = new float[MELSPEC_WINDOW * MEL_BINS];
                    for (int w = 0; w < MELSPEC_WINDOW; w++) {
                        int srcOff = (windowStart + w) * MELSPEC_STRIDE;
                        for (int b = 0; b < MEL_BINS; b++) {
                            embFlat[w * MEL_BINS + b] = melspecFlat[srcOff + b];
                        }
                    }
                    long[] embShape = {1, MELSPEC_WINDOW, MEL_BINS, 1};
                    OnnxTensor embTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(embFlat), embShape);
                    Map<String, OnnxTensor> embInputs = new HashMap<>();
                    embInputs.put(embeddingInputName, embTensor);
                    OrtSession.Result embResult = embeddingSession.run(embInputs);
                    float[][][][] embOut = (float[][][][]) embResult.get(0).getValue();
                    embResult.close();
                    embTensor.close();

                    if (featureBufferSize >= FEATURE_BUFFER_MAX) {
                        // FLAT arraycopy — copies values!
                        System.arraycopy(featureFlat, FEATURE_STRIDE, featureFlat, 0, (FEATURE_BUFFER_MAX - 1) * FEATURE_STRIDE);
                        featureBufferSize = FEATURE_BUFFER_MAX - 1;
                    }
                    int foff = featureBufferSize * FEATURE_STRIDE;
                    for (int d = 0; d < EMBEDDING_DIM; d++) {
                        featureFlat[foff + d] = embOut[0][0][0][d];
                    }
                    featureBufferSize++;
                }
            } catch (Exception e) {
                Log.w(TAG, "Pre-fill error: " + e.getMessage());
                break;
            }
        }

        // Reset melspec buffer to 76 frames of ones (matches Python library:
        // _get_embeddings does NOT modify self.melspectrogram_buffer, so after
        // init it stays at np.ones((76, 32)). The noise prefill above only
        // populates feature_buffer, not melspectrogram_buffer.)
        melspecBufferSize = MELSPEC_WINDOW;
        for (int i = 0; i < MELSPEC_WINDOW; i++) {
            int off = i * MELSPEC_STRIDE;
            for (int b = 0; b < MEL_BINS; b++) {
                melspecFlat[off + b] = 1.0f;
            }
        }

        Log.i(TAG, "Buffers initialized: melspec=" + melspecBufferSize + " features=" + featureBufferSize);
        rawAudioSize = 0;
    }

    private void startRecording() {
        Log.i(TAG, "startRecording: initializing AudioRecord...");

        // Hardcode buffer size — AudioRecord.getMinBufferSize() can block
        // indefinitely on R1's Android 5.1 audio HAL in certain states.
        // For 16kHz mono 16-bit: minBuf is typically 1280-2560 bytes.
        int minBuf = 2560; // safe lower bound
        int bufferSize = Math.max(minBuf, FRAME_SAMPLES * 2 * 4); // 10240 bytes
        Log.i(TAG, "startRecording: bufferSize=" + bufferSize);

        audioRecord = new AudioRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION,
                SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT, bufferSize);
        Log.i(TAG, "startRecording: AudioRecord created, state=" + audioRecord.getState());

        if (audioRecord.getState() != AudioRecord.STATE_INITIALIZED) {
            throw new RuntimeException("AudioRecord init failed, state=" + audioRecord.getState());
        }

        Log.i(TAG, "startRecording: starting recording...");
        audioRecord.startRecording();
        Log.i(TAG, "startRecording: recording started");

        detectThread = new Thread(this::detectionLoop, "OWW-Detect");
        detectThread.start();
    }

    private void detectionLoop() {
        android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_URGENT_AUDIO);

        short[] audioBuffer = new short[FRAME_SAMPLES];

        Log.i(TAG, "Detection loop started");

        while (isRunning && !Thread.interrupted()) {
            int read = audioRecord.read(audioBuffer, 0, FRAME_SAMPLES);
            if (read <= 0) {
                if (frameCount == 0) {
                    Log.w(TAG, "audioRecord.read returned " + read);
                }
                continue;
            }

            // Debug: log raw audio values periodically
            if (frameCount % 5 == 0) {
                int maxVal = 0;
                for (int i = 0; i < read; i++) {
                    int v = Math.abs(audioBuffer[i]);
                    if (v > maxVal) maxVal = v;
                }
                Log.i(TAG, String.format("audio: maxAbs=%d", maxVal));
            }

            // Add raw samples to rolling buffer with fixed gain
            for (int i = 0; i < read; i++) {
                if (rawAudioSize >= rawAudioBuffer.length) {
                    System.arraycopy(rawAudioBuffer, FRAME_SAMPLES, rawAudioBuffer, 0, rawAudioBuffer.length - FRAME_SAMPLES);
                    rawAudioSize -= FRAME_SAMPLES;
                }
                float val = audioBuffer[i] * FIXED_GAIN;
                if (val > 32767) val = 32767;
                if (val < -32768) val = -32768;
                rawAudioBuffer[rawAudioSize++] = val;
            }

            try {
                processFrame(read);
            } catch (Exception e) {
                Log.e(TAG, "processFrame error at frame " + frameCount, e);
                break;
            }

            frameCount++;
            if (frameCount == 1) {
                Log.i(TAG, "First frame processed! read=" + read);
            }
            if (frameCount % 75 == 0) {
                Log.i(TAG, "Running: " + frameCount + " frames (" + (frameCount * 80 / 1000) + "s), melspec=" + melspecBufferSize + " features=" + featureBufferSize);
            }
        }
        Log.w(TAG, "Detection loop ended. frameCount=" + frameCount + " isRunning=" + isRunning);
    }

    private void processFrame(int nNewSamples) {
        try {
            // Step 1: Melspectrogram
            // The Python library passes the last (n_samples + 160*3) = 1760 samples
            // to the melspectrogram model for STFT overlap context.
            int n_samples = nNewSamples; // typically 1280
            int melspecInputLen = n_samples + MELSPEC_CONTEXT; // 1760
            int copyStart = rawAudioSize - melspecInputLen;
            if (copyStart < 0) copyStart = 0;
            int actualLen = rawAudioSize - copyStart;

            float[] flatInput = new float[actualLen];
            System.arraycopy(rawAudioBuffer, copyStart, flatInput, 0, actualLen);
            long[] inputShape = {1, actualLen};
            OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(flatInput), inputShape);

            Map<String, OnnxTensor> melspecInputs = new HashMap<>();
            melspecInputs.put(melspecInputName, inputTensor);
            OrtSession.Result melspecResult = melspecSession.run(melspecInputs);
            float[][][][] melspecOut = (float[][][][]) melspecResult.get(0).getValue();
            melspecResult.close();
            inputTensor.close();

            // melspecOut shape: [time, 1, N, 32] — squeeze dims 0,1
            int nMelspecFrames = melspecOut[0][0].length;
            if (frameCount % 5 == 0) {
                Log.i(TAG, String.format("DEBUG melspec: inputLen=%d frames=%d val[0][0][0]=%.4f audio[0:3]=%.0f,%.0f,%.0f",
                    actualLen, nMelspecFrames, melspecOut[0][0][0][0],
                    flatInput[0], flatInput[Math.min(1,actualLen-1)], flatInput[Math.min(2,actualLen-1)]));
            }
            for (int f = 0; f < nMelspecFrames; f++) {
                if (melspecBufferSize >= MELSPEC_MAX_FRAMES) {
                    // FLAT arraycopy — copies VALUES, not references!
                    System.arraycopy(melspecFlat, MELSPEC_STRIDE, melspecFlat, 0, (MELSPEC_MAX_FRAMES - 1) * MELSPEC_STRIDE);
                    melspecBufferSize = MELSPEC_MAX_FRAMES - 1;
                }
                int off = melspecBufferSize * MELSPEC_STRIDE;
                for (int b = 0; b < MEL_BINS; b++) {
                    melspecFlat[off + b] = melspecOut[0][0][f][b] / 10f + 2f;
                }
                melspecBufferSize++;
            }

            // Step 2: Embeddings from sliding windows
            // The Python library loops: for i in np.arange(accumulated_samples//1280 - 1, -1, -1)
            // When accumulated_samples = 1280, this is np.arange(0, -1, -1) = [0], so only 1 iteration
            int nEmbedIterations = nNewSamples / FRAME_SAMPLES; // typically 1
            for (int i = nEmbedIterations - 1; i >= 0; i--) {
                int windowEnd = melspecBufferSize - MELSPEC_STEP * i;
                if (windowEnd <= 0) break;
                int windowStart = windowEnd - MELSPEC_WINDOW;
                if (windowStart < 0) break;

                // Input: [1, 76, 32, 1] — read from flat melspec buffer
                float[] embFlat = new float[MELSPEC_WINDOW * MEL_BINS];
                for (int w = 0; w < MELSPEC_WINDOW; w++) {
                    int srcOff = (windowStart + w) * MELSPEC_STRIDE;
                    for (int b = 0; b < MEL_BINS; b++) {
                        embFlat[w * MEL_BINS + b] = melspecFlat[srcOff + b];
                    }
                }
                long[] embShape = {1, MELSPEC_WINDOW, MEL_BINS, 1};
                OnnxTensor embTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(embFlat), embShape);

                Map<String, OnnxTensor> embInputs = new HashMap<>();
                embInputs.put(embeddingInputName, embTensor);
                OrtSession.Result embResult = embeddingSession.run(embInputs);
                float[][][][] embOut = (float[][][][]) embResult.get(0).getValue();
                embResult.close();
                embTensor.close();

                // embOut shape: [1, 1, 1, 96] → squeeze to [96]
                if (featureBufferSize >= FEATURE_BUFFER_MAX) {
                    // FLAT arraycopy — copies VALUES!
                    System.arraycopy(featureFlat, FEATURE_STRIDE, featureFlat, 0, (FEATURE_BUFFER_MAX - 1) * FEATURE_STRIDE);
                    featureBufferSize = FEATURE_BUFFER_MAX - 1;
                }
                int foff = featureBufferSize * FEATURE_STRIDE;
                for (int d = 0; d < EMBEDDING_DIM; d++) {
                    featureFlat[foff + d] = embOut[0][0][0][d];
                }
                featureBufferSize++;

                if (frameCount % 5 == 0) {
                    int w0Off = windowStart * MELSPEC_STRIDE;
                    int w75Off = (windowStart + MELSPEC_WINDOW - 1) * MELSPEC_STRIDE;
                    Log.i(TAG, String.format("DEBUG emb: feat[0][0]=%.4f feat[1][0]=%.4f melspecWin[0][0]=%.4f melspecWin[75][0]=%.4f",
                        embOut[0][0][0][0], embOut[0][0][0][1],
                        melspecFlat[w0Off], melspecFlat[w75Off]));
                }
            }

            // Step 3: Wakeword
            if (featureBufferSize >= WAKEWORD_INPUT_FRAMES) {
                float[] wkwFlat = new float[WAKEWORD_INPUT_FRAMES * EMBEDDING_DIM];
                int startFeature = featureBufferSize - WAKEWORD_INPUT_FRAMES;
                for (int f = 0; f < WAKEWORD_INPUT_FRAMES; f++) {
                    int srcOff = (startFeature + f) * FEATURE_STRIDE;
                    for (int d = 0; d < EMBEDDING_DIM; d++) {
                        wkwFlat[f * EMBEDDING_DIM + d] = featureFlat[srcOff + d];
                    }
                }
                long[] wkwShape = {1, WAKEWORD_INPUT_FRAMES, EMBEDDING_DIM};
                OnnxTensor wkwTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(wkwFlat), wkwShape);

                Map<String, OnnxTensor> wkwInputs = new HashMap<>();
                wkwInputs.put(wakewordInputName, wkwTensor);
                OrtSession.Result wkwResult = wakewordSession.run(wkwInputs);
                float[][] wkwOut = (float[][]) wkwResult.get(0).getValue();
                wkwResult.close();
                wkwTensor.close();

                float score = wkwOut[0][0];
                // Log score every 5 frames (~400ms) for real-time monitoring
                if (frameCount % 5 == 0) {
                    int sOff = startFeature * FEATURE_STRIDE;
                    Log.i(TAG, String.format("DEBUG wkw: feat[0][0]=%.4f wkwOut=%.6f score=%.4f features=%d melspec=%d",
                        featureFlat[sOff], wkwOut[0][0], score, featureBufferSize, melspecBufferSize));
                }
                if (score > DETECTION_THRESHOLD) {
                    Log.i(TAG, String.format("Wake word detected! score=%.3f", score));
                    // Play wake sound (two high beeps) so user knows it was heard
                    playWakeSound();
                    if (listener != null) {
                        listener.onWakeWordDetected();
                    }
                    featureBufferSize = 0;
                }
            }
        } catch (OrtException e) {
            Log.e(TAG, "ONNX inference error at frame " + frameCount, e);
            throw new RuntimeException(e);
        } catch (Exception e) {
            Log.e(TAG, "Process frame error at frame " + frameCount, e);
            throw new RuntimeException(e);
        }
    }

    /** Startup ding — low single beep, means "listening started" */
    private void playDing() {
        try {
            final ToneGenerator tone = new ToneGenerator(AudioManager.STREAM_SYSTEM, 100);
            tone.startTone(ToneGenerator.TONE_PROP_BEEP, 150);
            new Thread(() -> {
                try { Thread.sleep(200); } catch (InterruptedException e) {}
                tone.release();
            }).start();
        } catch (Exception e) {
            Log.w(TAG, "ToneGenerator failed: " + e.getMessage());
        }
    }

    /** Wake word detected — two high beeps, means "I heard you!" */
    private void playWakeSound() {
        new Thread(() -> {
            try {
                // First high beep
                ToneGenerator tone1 = new ToneGenerator(AudioManager.STREAM_SYSTEM, 100);
                tone1.startTone(ToneGenerator.TONE_PROP_BEEP2, 120);
                Thread.sleep(150);
                tone1.release();

                Thread.sleep(80);

                // Second high beep
                ToneGenerator tone2 = new ToneGenerator(AudioManager.STREAM_SYSTEM, 100);
                tone2.startTone(ToneGenerator.TONE_PROP_BEEP2, 120);
                Thread.sleep(150);
                tone2.release();
            } catch (Exception e) {
                Log.w(TAG, "Wake sound failed: " + e.getMessage());
            }
        }).start();
    }

    /**
     * Play a WAV file through STREAM_SYSTEM speaker (for testing mic pickup).
     * Loads from assets, plays via AudioTrack.
     */
    public void playTestAudio(String assetPath) {
        try {
            byte[] wavBytes = loadAsset(context.getAssets(), assetPath);

            // Find data chunk
            int dataStart = 44;
            for (int i = 12; i < wavBytes.length - 8; i++) {
                if (wavBytes[i] == 'd' && wavBytes[i+1] == 'a' &&
                    wavBytes[i+2] == 't' && wavBytes[i+3] == 'a') {
                    dataStart = i + 8;
                    break;
                }
            }

            int nSamples = (wavBytes.length - dataStart) / 2;
            short[] samples = new short[nSamples];
            for (int i = 0; i < nSamples; i++) {
                samples[i] = (short) ((wavBytes[dataStart + i*2] & 0xFF) |
                                      (wavBytes[dataStart + i*2 + 1] << 8));
            }

            // Play via STREAM_SYSTEM (routed to speaker on R1)
            int minBuf = AudioTrack.getMinBufferSize(16000,
                    AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT);
            int bufSize = Math.max(minBuf, nSamples * 2);

            AudioTrack track = new AudioTrack(AudioManager.STREAM_SYSTEM,
                    16000, AudioFormat.CHANNEL_OUT_MONO,
                    AudioFormat.ENCODING_PCM_16BIT, bufSize,
                    AudioTrack.MODE_STATIC);

            track.write(samples, 0, nSamples);
            track.play();

            Log.i(TAG, "Playing test audio: " + nSamples + " samples (" + (nSamples/16) + "ms)");

            // Wait for playback
            try { Thread.sleep(nSamples / 16 + 100); } catch (InterruptedException e) {}
            track.stop();
            track.release();

            Log.i(TAG, "Test audio playback done");
        } catch (Exception e) {
            Log.e(TAG, "playTestAudio error", e);
        }
    }

    public void reset() {
        melspecBufferSize = 0;
        featureBufferSize = 0;
        rawAudioSize = 0;
    }

    /**
     * Auto loopback test: play hey_jarvis_test.wav through speaker while
     * recording from mic, then run the recording through the pipeline.
     * This is fully automated — no human needed.
     */
    public void autoLoopbackTest() {
        new Thread(() -> {
            Log.i(TAG, "=== AUTO LOOPBACK TEST START ===");

            try {
                // 1. Load models
                if (env == null) {
                    loadModels();
                }

                // 2. Load test WAV
                byte[] wavBytes = loadAsset(context.getAssets(), "models/hey_jarvis_test.wav");
                int dataStart = 44;
                for (int i = 12; i < wavBytes.length - 8; i++) {
                    if (wavBytes[i] == 'd' && wavBytes[i+1] == 'a' &&
                        wavBytes[i+2] == 't' && wavBytes[i+3] == 'a') {
                        dataStart = i + 8;
                        break;
                    }
                }
                int nWavSamples = (wavBytes.length - dataStart) / 2;
                short[] wavSamples = new short[nWavSamples];
                for (int i = 0; i < nWavSamples; i++) {
                    wavSamples[i] = (short) ((wavBytes[dataStart + i*2] & 0xFF) |
                                             (wavBytes[dataStart + i*2 + 1] << 8));
                }
                Log.i(TAG, String.format("Test WAV: %d samples (%.1fs)", nWavSamples, nWavSamples / 16000.0));

                // 3. Record while playing the WAV through speaker
                //    Record 5 seconds total (1s silence + 3s wav + 1s silence)
                int recordSamples = 16000 * 5;
                int minBuf = AudioRecord.getMinBufferSize(16000,
                        AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
                int bufSize = Math.max(minBuf, recordSamples * 2);

                AudioRecord recorder = new AudioRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION,
                        16000, AudioFormat.CHANNEL_IN_MONO,
                        AudioFormat.ENCODING_PCM_16BIT, bufSize);

                if (recorder.getState() != AudioRecord.STATE_INITIALIZED) {
                    Log.e(TAG, "AudioRecord init failed in loopback test");
                    recorder.release();
                    return;
                }

                // Start recording
                recorder.startRecording();

                // Play WAV through speaker (STREAM_SYSTEM)
                int playMinBuf = AudioTrack.getMinBufferSize(16000,
                        AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT);
                int playBufSize = Math.max(playMinBuf, nWavSamples * 2);
                AudioTrack track = new AudioTrack(AudioManager.STREAM_SYSTEM,
                        16000, AudioFormat.CHANNEL_OUT_MONO,
                        AudioFormat.ENCODING_PCM_16BIT, playBufSize,
                        AudioTrack.MODE_STATIC);
                track.write(wavSamples, 0, nWavSamples);

                // Wait 1 second (silence), then play
                Thread.sleep(1000);
                Log.i(TAG, "Playing test WAV through speaker...");
                track.play();

                // Record while playing + 1s after
                short[] recordedData = new short[recordSamples];
                int totalRead = 0;
                while (totalRead < recordSamples) {
                    int read = recorder.read(recordedData, totalRead, recordSamples - totalRead);
                    if (read <= 0) break;
                    totalRead += read;
                }

                track.stop();
                track.release();
                recorder.stop();
                recorder.release();

                // Log recording stats
                int maxVal = 0;
                long sumSq = 0;
                for (int i = 0; i < totalRead; i++) {
                    int v = Math.abs(recordedData[i]);
                    if (v > maxVal) maxVal = v;
                    sumSq += (long) recordedData[i] * recordedData[i];
                }
                double rms = Math.sqrt((double) sumSq / totalRead);
                Log.i(TAG, String.format("Loopback recorded: %d samples, maxAbs=%d RMS=%.1f", totalRead, maxVal, rms));

                // 4. Run the recording through the pipeline
                initBuffers();
                rawAudioSize = 0;
                frameCount = 0;

                // Feed 1s silence first (like predict_clip)
                for (int i = 0; i < 16000; i++) {
                    if (rawAudioSize >= rawAudioBuffer.length) {
                        System.arraycopy(rawAudioBuffer, FRAME_SAMPLES, rawAudioBuffer, 0, rawAudioBuffer.length - FRAME_SAMPLES);
                        rawAudioSize -= FRAME_SAMPLES;
                    }
                    rawAudioBuffer[rawAudioSize++] = 0.0f;
                }

                // Feed recorded audio
                float maxScore = 0;
                int frame = 0;
                for (int start = 0; start + FRAME_SAMPLES <= totalRead; start += FRAME_SAMPLES) {
                    for (int i = 0; i < FRAME_SAMPLES; i++) {
                        if (rawAudioSize >= rawAudioBuffer.length) {
                            System.arraycopy(rawAudioBuffer, FRAME_SAMPLES, rawAudioBuffer, 0, rawAudioBuffer.length - FRAME_SAMPLES);
                            rawAudioSize -= FRAME_SAMPLES;
                        }
                        // Use raw audio values (no AGC) — same as file test
                        rawAudioBuffer[rawAudioSize++] = (float) recordedData[start + i];
                    }

                    try {
                        processFrame(FRAME_SAMPLES);
                    } catch (Exception e) {
                        Log.e(TAG, "Loopback pipeline error at frame " + frame, e);
                        break;
                    }

                    if (featureBufferSize >= WAKEWORD_INPUT_FRAMES) {
                        int startFeat = featureBufferSize - WAKEWORD_INPUT_FRAMES;
                        float[] wkwFlat = new float[WAKEWORD_INPUT_FRAMES * EMBEDDING_DIM];
                        for (int f = 0; f < WAKEWORD_INPUT_FRAMES; f++) {
                            int srcOff = (startFeat + f) * FEATURE_STRIDE;
                            for (int d = 0; d < EMBEDDING_DIM; d++) {
                                wkwFlat[f * EMBEDDING_DIM + d] = featureFlat[srcOff + d];
                            }
                        }
                        long[] wkwShape = {1, WAKEWORD_INPUT_FRAMES, EMBEDDING_DIM};
                        OnnxTensor wkwTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(wkwFlat), wkwShape);
                        Map<String, OnnxTensor> wkwInputs = new HashMap<>();
                        wkwInputs.put(wakewordInputName, wkwTensor);
                        OrtSession.Result wkwResult = wakewordSession.run(wkwInputs);
                        float[][] wkwOut = (float[][]) wkwResult.get(0).getValue();
                        wkwResult.close();
                        wkwTensor.close();

                        float score = wkwOut[0][0];
                        if (score > maxScore) maxScore = score;
                        if (score > 0.01 || frame % 10 == 0) {
                            Log.i(TAG, String.format("LOOPBACK frame %d: score=%.6f (max=%.6f)", frame, score, maxScore));
                        }
                    }

                    frame++;
                }

                Log.i(TAG, String.format("=== LOOPBACK TEST DONE: %d frames, maxScore=%.6f ===", frame, maxScore));
                if (maxScore > 0.5) {
                    Log.i(TAG, ">>> LOOPBACK TEST PASSED! Mic pipeline works!");
                } else if (maxScore > 0.1) {
                    Log.i(TAG, ">>> LOOPBACK: Low score but not zero. Audio quality issue.");
                } else {
                    Log.i(TAG, ">>> LOOPBACK: Score near zero. Pipeline or audio issue.");
                }

            } catch (Exception e) {
                Log.e(TAG, "Auto loopback test error", e);
            }
        }, "LoopbackTest").start();
    }

    /**
     * Test mode: process a WAV file from assets through the pipeline.
     * Bypasses the microphone to verify the ONNX pipeline works on-device.
     */
    public void testWithFile(String assetPath) {
        Log.i(TAG, "=== FILE TEST: " + assetPath + " ===");

        try {
            // Load models if not already loaded
            if (env == null) {
                loadModels();
            }
            initBuffers();
            rawAudioSize = 0;
            frameCount = 0;

            // Load WAV file from assets
            byte[] wavBytes = loadAsset(context.getAssets(), assetPath);

            // Parse WAV header (skip 44-byte header, get PCM data)
            int dataStart = 44;
            // Find "data" chunk
            for (int i = 12; i < wavBytes.length - 8; i++) {
                if (wavBytes[i] == 'd' && wavBytes[i+1] == 'a' &&
                    wavBytes[i+2] == 't' && wavBytes[i+3] == 'a') {
                    dataStart = i + 8;
                    break;
                }
            }

            int nSamples = (wavBytes.length - dataStart) / 2;
            Log.i(TAG, "WAV: " + nSamples + " samples (" + (nSamples / 16) + "ms)");

            // Convert to short array
            short[] samples = new short[nSamples];
            for (int i = 0; i < nSamples; i++) {
                samples[i] = (short) ((wavBytes[dataStart + i*2] & 0xFF) |
                                      (wavBytes[dataStart + i*2 + 1] << 8));
            }

            // Pad with 1 second of silence (like predict_clip)
            int silenceSamples = 16000;

            // Feed silence first
            for (int i = 0; i < silenceSamples; i++) {
                if (rawAudioSize >= rawAudioBuffer.length) {
                    System.arraycopy(rawAudioBuffer, FRAME_SAMPLES, rawAudioBuffer, 0, rawAudioBuffer.length - FRAME_SAMPLES);
                    rawAudioSize -= FRAME_SAMPLES;
                }
                rawAudioBuffer[rawAudioSize++] = 0.0f;
            }

            // Feed actual audio in 1280-sample chunks
            float maxScore = 0;
            int frame = 0;
            for (int start = 0; start + FRAME_SAMPLES <= nSamples; start += FRAME_SAMPLES) {
                // Add chunk to raw buffer
                for (int i = 0; i < FRAME_SAMPLES; i++) {
                    if (rawAudioSize >= rawAudioBuffer.length) {
                        System.arraycopy(rawAudioBuffer, FRAME_SAMPLES, rawAudioBuffer, 0, rawAudioBuffer.length - FRAME_SAMPLES);
                        rawAudioSize -= FRAME_SAMPLES;
                    }
                    rawAudioBuffer[rawAudioSize++] = (float) samples[start + i];
                }

                // Process
                try {
                    processFrame(FRAME_SAMPLES);
                } catch (Exception e) {
                    Log.e(TAG, "File test error at frame " + frame, e);
                    break;
                }

                // Check score
                if (featureBufferSize >= WAKEWORD_INPUT_FRAMES) {
                    int startFeat = featureBufferSize - WAKEWORD_INPUT_FRAMES;
                    float[] wkwFlat = new float[WAKEWORD_INPUT_FRAMES * EMBEDDING_DIM];
                    for (int f = 0; f < WAKEWORD_INPUT_FRAMES; f++) {
                        int srcOff = (startFeat + f) * FEATURE_STRIDE;
                        for (int d = 0; d < EMBEDDING_DIM; d++) {
                            wkwFlat[f * EMBEDDING_DIM + d] = featureFlat[srcOff + d];
                        }
                    }
                    long[] wkwShape = {1, WAKEWORD_INPUT_FRAMES, EMBEDDING_DIM};
                    OnnxTensor wkwTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(wkwFlat), wkwShape);
                    Map<String, OnnxTensor> wkwInputs = new HashMap<>();
                    wkwInputs.put(wakewordInputName, wkwTensor);
                    OrtSession.Result wkwResult = wakewordSession.run(wkwInputs);
                    float[][] wkwOut = (float[][]) wkwResult.get(0).getValue();
                    wkwResult.close();
                    wkwTensor.close();

                    float score = wkwOut[0][0];
                    if (score > maxScore) maxScore = score;
                    if (score > 0.01) {
                        Log.i(TAG, String.format("FILE TEST frame %d: score=%.6f", frame, score));
                    }
                }

                frame++;
            }

            Log.i(TAG, String.format("=== FILE TEST DONE: %d frames, maxScore=%.6f ===", frame, maxScore));
            if (maxScore > 0.5) {
                Log.i(TAG, ">>> FILE TEST PASSED! Pipeline works on R1!");
            } else {
                Log.i(TAG, ">>> FILE TEST FAILED. Pipeline issue on R1.");
            }

        } catch (Exception e) {
            Log.e(TAG, "File test error", e);
        }
    }
}
