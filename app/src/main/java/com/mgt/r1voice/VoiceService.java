package com.mgt.r1voice;

import android.app.Notification;
import android.app.Service;
import android.content.Intent;
import android.os.IBinder;
import android.util.Log;

import org.json.JSONException;
import org.json.JSONObject;

/**
 * VoiceService — foreground service managing the voice assistant lifecycle.
 *
 * Server-side wake word detection. R1 is a dumb audio device:
 * - Continuously streams 16kHz PCM to server via WebSocket
 * - Receives TTS PCM + state updates from server
 *
 * State machine is managed entirely by the server.
 * R1 just needs to know when to mute its mic (during TTS playback).
 */
public class VoiceService extends Service {

    private static final String TAG = "VoiceService";
    private static final int NOTIF_ID = 1001;

    public static String currentState = "";

    private WsClient wsClient;
    private AudioRecorder audioRecorder;
    private AudioPlayer audioPlayer;

    private String serverAddr;
    private boolean isRunning = false;

    // Whether we should be streaming mic audio to server
    // Stop during TTS playback to avoid echo
    private boolean shouldStreamMic = true;

    @Override
    public void onCreate() {
        super.onCreate();
        Log.i(TAG, "VoiceService created");
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent != null && intent.hasExtra("server_addr")) {
            serverAddr = intent.getStringExtra("server_addr");
        } else {
            serverAddr = getSharedPreferences("r1voice", MODE_PRIVATE)
                .getString("server_addr", "ws://192.168.1.120:8090");
        }

        if (!isRunning) {
            isRunning = true;
            startForeground();
            startVoiceAssistant();
        }

        return START_STICKY;
    }

    private void startForeground() {
        Notification notification = new Notification.Builder(this)
                .setContentTitle("R1 语音助手")
                .setContentText("服务端唤醒词运行中...")
                .setSmallIcon(android.R.drawable.ic_btn_speak_now)
                .setOngoing(true)
                .build();
        startForeground(NOTIF_ID, notification);
    }

    private void startVoiceAssistant() {
        Log.i(TAG, "Starting voice assistant, server=" + serverAddr);

        audioPlayer = new AudioPlayer();

        // Connect WebSocket
        wsClient = new WsClient(serverAddr, new WsClient.WsListener() {
            @Override
            public void onConnected() {
                Log.i(TAG, "WS connected");
                updateState("idle");
                // Start streaming mic audio immediately
                startRecording();
            }

            @Override
            public void onDisconnected() {
                Log.i(TAG, "WS disconnected");
                updateState("");
                stopRecording();
                stopPlayback();
            }

            @Override
            public void onTextMessage(String message) {
                handleTextMessage(message);
            }

            @Override
            public void onBinaryMessage(byte[] data) {
                // Binary data = TTS PCM chunk
                if (!audioPlayer.isPlaying()) {
                    audioPlayer.start();
                }
                audioPlayer.writePcm(data);
            }
        });

        wsClient.connect();
    }

    private void handleTextMessage(String message) {
        try {
            JSONObject json = new JSONObject(message);
            String type = json.optString("type");

            switch (type) {
                case "state":
                    String state = json.optString("state");
                    updateState(state);
                    // Stop mic streaming during speaking to avoid echo
                    if ("speaking".equals(state)) {
                        shouldStreamMic = false;
                    } else if ("idle".equals(state)) {
                        shouldStreamMic = true;
                    }
                    break;

                case "tts_done":
                    Log.i(TAG, "TTS done, stopping playback");
                    audioPlayer.stop();
                    shouldStreamMic = true;
                    updateState("idle");
                    break;

                case "asr_result":
                    String text = json.optString("text");
                    Log.i(TAG, "ASR: " + text);
                    break;

                default:
                    Log.w(TAG, "Unknown message type: " + type);
            }
        } catch (JSONException e) {
            Log.e(TAG, "Parse error", e);
        }
    }

    private void updateState(String state) {
        currentState = state;
        Log.i(TAG, "State → " + state);

        switch (state) {
            case "idle":
                LedController.setColor(LedController.COLOR_IDLE);
                break;
            case "listening":
                LedController.setColor(LedController.COLOR_LISTENING);
                break;
            case "thinking":
                LedController.setColor(LedController.COLOR_THINKING);
                break;
            case "speaking":
                LedController.setColor(LedController.COLOR_SPEAKING);
                break;
            default:
                LedController.setColor(0, 0, 0);
        }
    }

    // === Audio Recording (continuous streaming) ===

    private void startRecording() {
        if (audioRecorder != null && audioRecorder.isRecording()) return;

        audioRecorder = new AudioRecorder(new AudioRecorder.AudioFrameCallback() {
            @Override
            public void onAudioFrame(byte[] data) {
                // Only stream mic when we're not playing TTS
                if (shouldStreamMic && wsClient != null && wsClient.isConnected()) {
                    wsClient.sendBinary(data);
                }
            }
        });

        audioRecorder.start();
        Log.i(TAG, "Mic streaming started");
    }

    private void stopRecording() {
        if (audioRecorder != null) {
            audioRecorder.stop();
            audioRecorder = null;
        }
    }

    // === Audio Playback ===

    private void stopPlayback() {
        if (audioPlayer != null) {
            audioPlayer.stop();
        }
    }

    @Override
    public void onDestroy() {
        Log.i(TAG, "VoiceService destroyed");
        isRunning = false;
        stopRecording();
        stopPlayback();
        if (wsClient != null) {
            wsClient.disconnect();
            wsClient = null;
        }
        LedController.setColor(0, 0, 0);
        super.onDestroy();
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
}
