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
 * State machine: IDLE → LISTENING → THINKING → SPEAKING → IDLE
 *
 * IDLE:       Local openWakeWord listening for wake word (on-device TFLite)
 * LISTENING:  Recording audio, streaming to server for ASR
 * THINKING:   Server processing ASR + Hermes
 * SPEAKING:   Receiving TTS audio from server, playing via AudioPlayer
 */
public class VoiceService extends Service {

    private static final String TAG = "VoiceService";
    private static final int NOTIF_ID = 1001;

    public static String currentState = "";

    private WsClient wsClient;
    private AudioRecorder audioRecorder;
    private AudioPlayer audioPlayer;
    private WakeWordDetector wakeWordDetector;

    private String serverAddr;
    private boolean isRunning = false;

    private static final String STATE_IDLE = "idle";
    private static final String STATE_LISTENING = "listening";
    private static final String STATE_THINKING = "thinking";
    private static final String STATE_SPEAKING = "speaking";

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
                .setContentText("本地唤醒词运行中...")
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
                updateState(STATE_IDLE);
                startWakeWordDetection();
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
                    break;

                case "tts_done":
                    Log.i(TAG, "TTS done, stopping playback");
                    audioPlayer.stop();
                    // Back to idle — stop recording and resume wake word detection
                    stopRecording();
                    updateState(STATE_IDLE);
                    // Resume existing wake word detector (AudioRecord is still alive)
                    if (wakeWordDetector != null) {
                        wakeWordDetector.resumeDetection();
                    } else {
                        startWakeWordDetection();
                    }
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
            case STATE_IDLE:
                LedController.setColor(LedController.COLOR_IDLE);
                break;
            case STATE_LISTENING:
                LedController.setColor(LedController.COLOR_LISTENING);
                break;
            case STATE_THINKING:
                LedController.setColor(LedController.COLOR_THINKING);
                break;
            case STATE_SPEAKING:
                LedController.setColor(LedController.COLOR_SPEAKING);
                break;
            default:
                LedController.setColor(0, 0, 0);
        }
    }

    // === Local Wake Word Detection (openWakeWord) ===

    private void startWakeWordDetection() {
        if (wakeWordDetector != null && wakeWordDetector.isRunning()) return;

        Log.i(TAG, "Starting local wake word detection (openWakeWord)");
        wakeWordDetector = new WakeWordDetector(this, new WakeWordDetector.WakeWordListener() {
            @Override
            public void onWakeWordDetected() {
                Log.i(TAG, "Wake word detected locally!");
                
                // Pause detection loop but keep AudioRecord alive
                // (releasing + creating new AudioRecord causes AudioFlinger deadlock on R1)
                if (wakeWordDetector != null) {
                    wakeWordDetector.pauseDetection();
                }

                // Delay recording start so the wake beep finishes first
                // (beep is ~350ms, wait 500ms to be safe)
                new Thread(() -> {
                    try { Thread.sleep(500); } catch (InterruptedException e) { return; }
                    
                    // Notify server to enter listening state
                    if (wsClient != null && wsClient.isConnected()) {
                        try {
                            JSONObject msg = new JSONObject();
                            msg.put("type", "wake");
                            wsClient.sendText(msg.toString());
                        } catch (JSONException e) {
                            Log.e(TAG, "Error sending wake", e);
                        }
                    }

                    // Switch to listening — start recording using the SAME AudioRecord
                    updateState(STATE_LISTENING);
                    startRecording();
                }).start();
            }
        });

        wakeWordDetector.start();
    }

    private void stopWakeWordDetection() {
        if (wakeWordDetector != null) {
            wakeWordDetector.stop();
            wakeWordDetector = null;
        }
    }

    // === Audio Recording (after wake word) ===

    private void startRecording() {
        if (audioRecorder != null && audioRecorder.isRecording()) return;

        audioRecorder = new AudioRecorder(new AudioRecorder.AudioFrameCallback() {
            @Override
            public void onAudioFrame(byte[] data) {
                if (wsClient != null && wsClient.isConnected()) {
                    wsClient.sendBinary(data);
                }
            }
        });

        // Reuse AudioRecord from WakeWordDetector to avoid AudioFlinger deadlock
        if (wakeWordDetector != null && wakeWordDetector.getAudioRecord() != null) {
            audioRecorder.startWithExistingRecord(wakeWordDetector.getAudioRecord());
        } else {
            audioRecorder.start();
        }
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
        stopWakeWordDetection();
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
