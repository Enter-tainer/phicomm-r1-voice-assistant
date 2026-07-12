package com.mgt.r1voice;

import android.app.Activity;
import android.content.Intent;
import android.content.SharedPreferences;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioRecord;
import android.media.AudioTrack;
import android.media.MediaRecorder;
import android.media.ToneGenerator;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

public class MainActivity extends Activity {

    private static final String TAG = "AudioTest";
    private static final String PREFS = "r1voice";
    private static final String KEY_SERVER = "server_addr";

    private EditText etServerAddr;
    private Button btnStart, btnStop, btnTestAudio;
    private TextView tvStatus, tvState;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        etServerAddr = (EditText) findViewById(R.id.etServerAddr);
        btnStart = (Button) findViewById(R.id.btnStart);
        btnStop = (Button) findViewById(R.id.btnStop);
        btnTestAudio = (Button) findViewById(R.id.btnTestAudio);
        tvStatus = (TextView) findViewById(R.id.tvStatus);
        tvState = (TextView) findViewById(R.id.tvState);

        // Load saved server address
        SharedPreferences prefs = getSharedPreferences(PREFS, MODE_PRIVATE);
        String savedAddr = prefs.getString(KEY_SERVER, "ws://192.168.1.120:8090");
        etServerAddr.setText(savedAddr);

        btnStart.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String addr = etServerAddr.getText().toString().trim();
                if (addr.isEmpty()) return;

                getSharedPreferences(PREFS, MODE_PRIVATE)
                    .edit()
                    .putString(KEY_SERVER, addr)
                    .apply();

                Intent intent = new Intent(MainActivity.this, VoiceService.class);
                intent.putExtra(KEY_SERVER, addr);
                startService(intent);

                btnStart.setEnabled(false);
                btnStop.setEnabled(true);
                tvStatus.setText("状态: 服务已启动");
            }
        });

        btnStop.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                stopService(new Intent(MainActivity.this, VoiceService.class));
                btnStart.setEnabled(true);
                btnStop.setEnabled(false);
                tvStatus.setText("状态: 未连接");
                tvState.setText("");
            }
        });

        btnTestAudio.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                runAudioTest();
            }
        });
    }

    /**
     * Audio test: record 2 seconds, then play it back.
     * Also plays a "ding" before recording so user knows when to speak.
     */
    private void runAudioTest() {
        btnTestAudio.setEnabled(false);
        btnTestAudio.setText("测试中...");
        tvStatus.setText("状态: 录音测试中...");

        new Thread(new Runnable() {
            @Override
            public void run() {
                final int sampleRate = 16000;
                final int recordSeconds = 2;
                final int numSamples = sampleRate * recordSeconds;

                try {
                    // Step 1: Play "ding" to signal start
                    Log.i(TAG, "=== Audio Test Start ===");
                    playBeep(800, 200);
                    Thread.sleep(300);

                    // Step 2: Record 2 seconds
                    Log.i(TAG, "Recording " + recordSeconds + "s...");
                    int minBuf = AudioRecord.getMinBufferSize(sampleRate,
                            AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
                    int bufSize = Math.max(minBuf, numSamples * 2);

                    AudioRecord recorder = new AudioRecord(
                            MediaRecorder.AudioSource.VOICE_RECOGNITION,
                            sampleRate, AudioFormat.CHANNEL_IN_MONO,
                            AudioFormat.ENCODING_PCM_16BIT, bufSize);

                    if (recorder.getState() != AudioRecord.STATE_INITIALIZED) {
                        Log.e(TAG, "AudioRecord init failed!");
                        recorder.release();
                        return;
                    }

                    short[] recordedData = new short[numSamples];
                    recorder.startRecording();

                    int totalRead = 0;
                    while (totalRead < numSamples) {
                        int read = recorder.read(recordedData, totalRead, numSamples - totalRead);
                        if (read <= 0) {
                            Log.e(TAG, "read failed at " + totalRead + ": " + read);
                            break;
                        }
                        totalRead += read;
                    }
                    recorder.stop();
                    recorder.release();

                    // Log volume stats
                    int maxVal = 0, minVal = 0;
                    long sumSq = 0;
                    for (int i = 0; i < totalRead; i++) {
                        if (recordedData[i] > maxVal) maxVal = recordedData[i];
                        if (recordedData[i] < minVal) minVal = recordedData[i];
                        sumSq += (long) recordedData[i] * recordedData[i];
                    }
                    double rms = Math.sqrt((double) sumSq / totalRead);
                    Log.i(TAG, String.format(
                        "Recorded %d samples. max=%d min=%d RMS=%.1f", totalRead, maxVal, minVal, rms));

                    // Step 3: Play "ding" then play back recording
                    Thread.sleep(200);
                    playBeep(600, 200);
                    Thread.sleep(300);

                    Log.i(TAG, "Playing back...");
                    int playMinBuf = AudioTrack.getMinBufferSize(sampleRate,
                            AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT);
                    int playBufSize = Math.max(playMinBuf, numSamples * 2);

                    AudioTrack track = new AudioTrack(AudioManager.STREAM_SYSTEM,
                            sampleRate, AudioFormat.CHANNEL_OUT_MONO,
                            AudioFormat.ENCODING_PCM_16BIT, playBufSize,
                            AudioTrack.MODE_STREAM);

                    // Boost volume: multiply samples by 4x for playback
                    short[] playData = new short[totalRead];
                    for (int i = 0; i < totalRead; i++) {
                        int boosted = recordedData[i] * 4;
                        if (boosted > 32767) boosted = 32767;
                        if (boosted < -32768) boosted = -32768;
                        playData[i] = (short) boosted;
                    }

                    track.play();
                    track.write(playData, 0, totalRead);

                    // Wait for playback to finish
                    Thread.sleep((totalRead / sampleRate * 1000) + 200);
                    track.stop();
                    track.release();

                    // Final ding
                    playBeep(1000, 150);

                    Log.i(TAG, "=== Audio Test Done ===");

                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            btnTestAudio.setEnabled(true);
                            btnTestAudio.setText("录音测试 (录2秒→播放)");
                            tvStatus.setText("状态: 测试完成，看logcat");
                        }
                    });

                } catch (Exception e) {
                    Log.e(TAG, "Audio test error", e);
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            btnTestAudio.setEnabled(true);
                            btnTestAudio.setText("录音测试 (录2秒→播放)");
                            tvStatus.setText("状态: 测试出错");
                        }
                    });
                }
            }
        }).start();
    }

    private void playBeep(int freq, int durationMs) {
        try {
            // Use STREAM_SYSTEM instead of STREAM_MUSIC because on R1,
            // STREAM_MUSIC is not routed to the speaker
            ToneGenerator tone = new ToneGenerator(AudioManager.STREAM_SYSTEM, 100);
            tone.startTone(ToneGenerator.TONE_PROP_BEEP, durationMs);
            try { Thread.sleep(durationMs + 50); } catch (InterruptedException e) {}
            tone.release();
        } catch (Exception e) {
            Log.w(TAG, "ToneGenerator failed", e);
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        String state = VoiceService.currentState;
        if (state != null && !state.isEmpty()) {
            tvState.setText(state);
        }
    }
}
