package com.mgt.r1voice;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.os.Handler;
import android.util.Log;

/**
 * BootReceiver — starts VoiceService on device boot.
 *
 * IMPORTANT: Uses goAsync() to keep the receiver alive during the 30s delay.
 * R1's audio HAL (4-mic array) is initialized by Phicomm's mediaserver during
 * boot. If our app creates an AudioRecord before Phicomm finishes, AudioFlinger
 * deadlocks permanently (no root access to restart mediaserver).
 */
public class BootReceiver extends BroadcastReceiver {

    private static final String TAG = "BootReceiver";
    private static final long BOOT_DELAY_MS = 30000; // 30 seconds

    @Override
    public void onReceive(final Context context, Intent intent) {
        if (!Intent.ACTION_BOOT_COMPLETED.equals(intent.getAction())) return;

        Log.i(TAG, "Boot completed, starting VoiceService in " + (BOOT_DELAY_MS/1000) + "s...");

        final String serverAddr = context.getSharedPreferences("r1voice", Context.MODE_PRIVATE)
            .getString("server_addr", "ws://192.168.1.120:8090");

        // Use goAsync to keep the receiver alive during the delay
        final PendingResult pendingResult = goAsync();

        new Handler().postDelayed(new Runnable() {
            @Override
            public void run() {
                try {
                    Log.i(TAG, "Starting VoiceService after boot delay");
                    Intent serviceIntent = new Intent(context, VoiceService.class);
                    serviceIntent.putExtra("server_addr", serverAddr);
                    context.startService(serviceIntent);
                } finally {
                    pendingResult.finish();
                }
            }
        }, BOOT_DELAY_MS);
    }
}
