package com.mgt.r1voice;

import android.util.Log;

import org.java_websocket.client.WebSocketClient;
import org.java_websocket.handshake.ServerHandshake;

import java.net.URI;
import java.nio.ByteBuffer;

/**
 * WsClient — WebSocket client for connecting to the Voice Server.
 *
 * Handles:
 * - Connection and reconnection
 * - Sending binary (PCM) and text (JSON) messages
 * - Receiving binary (TTS PCM) and text (state/control) messages
 */
public class WsClient {

    private static final String TAG = "WsClient";

    public interface WsListener {
        void onConnected();
        void onDisconnected();
        void onTextMessage(String message);
        void onBinaryMessage(byte[] data);
    }

    private WsListener listener;
    private WebSocketClient ws;
    private String serverUrl;
    private boolean shouldReconnect = false;
    private Thread reconnectThread;

    public WsClient(String url, WsListener listener) {
        this.serverUrl = url;
        this.listener = listener;
    }

    public void connect() {
        shouldReconnect = true;
        doConnect();
    }

    private void doConnect() {
        try {
            URI uri = URI.create(serverUrl);
            ws = new WebSocketClient(uri) {
                @Override
                public void onOpen(ServerHandshake handshake) {
                    Log.i(TAG, "WebSocket connected");
                    if (listener != null) listener.onConnected();
                }

                @Override
                public void onMessage(String message) {
                    if (listener != null) listener.onTextMessage(message);
                }

                @Override
                public void onMessage(ByteBuffer bytes) {
                    byte[] data = new byte[bytes.remaining()];
                    bytes.get(data);
                    if (listener != null) listener.onBinaryMessage(data);
                }

                @Override
                public void onClose(int code, String reason, boolean remote) {
                    Log.i(TAG, "WebSocket closed: " + code + " " + reason);
                    if (listener != null) listener.onDisconnected();
                    if (shouldReconnect) {
                        scheduleReconnect();
                    }
                }

                @Override
                public void onError(Exception ex) {
                    Log.e(TAG, "WebSocket error", ex);
                }
            };
            ws.connect();
        } catch (Exception e) {
            Log.e(TAG, "Connect failed", e);
            if (shouldReconnect) {
                scheduleReconnect();
            }
        }
    }

    private void scheduleReconnect() {
        if (reconnectThread != null && reconnectThread.isAlive()) return;

        reconnectThread = new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    Thread.sleep(3000);
                } catch (InterruptedException e) {
                    return;
                }
                if (shouldReconnect) {
                    Log.i(TAG, "Reconnecting...");
                    doConnect();
                }
            }
        }, "WsClient-Reconnect");
        reconnectThread.start();
    }

    public void sendText(String text) {
        if (ws != null && ws.isOpen()) {
            ws.send(text);
        }
    }

    public void sendBinary(byte[] data) {
        if (ws != null && ws.isOpen()) {
            ws.send(data);
        } else {
            Log.w(TAG, "sendBinary: ws not open! ws=" + (ws != null) + " open=" + (ws != null && ws.isOpen()));
        }
    }

    public boolean isConnected() {
        return ws != null && ws.isOpen();
    }

    public void disconnect() {
        shouldReconnect = false;
        if (reconnectThread != null) {
            reconnectThread.interrupt();
            reconnectThread = null;
        }
        if (ws != null) {
            ws.close();
            ws = null;
        }
    }
}
