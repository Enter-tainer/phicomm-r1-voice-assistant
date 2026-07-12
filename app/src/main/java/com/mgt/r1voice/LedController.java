package com.mgt.r1voice;

import android.util.Log;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

/**
 * LedController — controls R1 bottom RGB LED via sysfs.
 *
 * Requires root access.
 * LED paths: /sys/class/leds/{red,green,blue}/brightness
 */
public class LedController {

    private static final String TAG = "LedController";

    private static final String RED_PATH = "/sys/class/leds/red/brightness";
    private static final String GREEN_PATH = "/sys/class/leds/green/brightness";
    private static final String BLUE_PATH = "/sys/class/leds/blue/brightness";

    // State colors (0-255)
    public static final int[] COLOR_IDLE = {0, 0, 30};       // dim blue
    public static final int[] COLOR_LISTENING = {0, 80, 0};   // green
    public static final int[] COLOR_THINKING = {80, 80, 0};   // yellow
    public static final int[] COLOR_SPEAKING = {0, 80, 80};   // cyan

    public static void setColor(int r, int g, int b) {
        writeSysfs(RED_PATH, r);
        writeSysfs(GREEN_PATH, g);
        writeSysfs(BLUE_PATH, b);
    }

    public static void setColor(int[] rgb) {
        if (rgb != null && rgb.length >= 3) {
            setColor(rgb[0], rgb[1], rgb[2]);
        }
    }

    private static void writeSysfs(String path, int value) {
        try {
            File f = new File(path);
            if (!f.exists()) {
                Log.w(TAG, "sysfs path not found: " + path);
                return;
            }
            // Use su for root access
            Process p = Runtime.getRuntime().exec(new String[]{
                "su", "-c", "echo " + value + " > " + path
            });
            p.waitFor();
        } catch (Exception e) {
            Log.w(TAG, "Failed to write " + path + ": " + e.getMessage());
        }
    }
}
