#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R1 Voice Assistant Watchdog — auto-recovery for R1 AudioFlinger deadlocks.

Monitors the R1 voice server health state (written by server.py to
/tmp/r1_server_state.json) and the R1 device itself. When it detects:

  - R1 connected but audio frames stalled for > STALL_TIMEOUT (AudioFlinger deadlock)
  - R1 not connected at all (app crashed, WS dead, adbd hung)

it executes a recovery ladder:

  1. SOFT: ADB force-stop Phicomm speaker services + restart our app
  2. HARD: adb reboot R1 (clears AudioFlinger deadlock that soft recovery can't fix)
  3. POST-REBOOT: wait for device to come back, then force-stop Phicomm and start app

Designed to run as a systemd user service (Restart=always) so it survives
server.py crashes. Read-only watcher + subprocess ADB control.
"""

import json
import logging
import os
import subprocess
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("r1.watchdog")

# --- Config ---------------------------------------------------------------
R1_ADB = "192.168.1.152:5555"
R1_IP = "192.168.1.152"
SERVER_IP = "192.168.1.120"
WS_PORT = 8090
STATE_FILE = "/tmp/r1_server_state.json"

# If no audio frame arrives for this long while connected → deadlock
STALL_TIMEOUT = 60          # seconds
# If R1 is not connected for this long → app/WS dead
NOT_CONNECTED_TIMEOUT = 90  # seconds
CHECK_INTERVAL = 10         # seconds
SOFT_RECOVERY_COOLDOWN = 120  # min seconds between soft recoveries
SOFT_OBSERVE_WINDOW = 90      # seconds to wait after soft recovery before escalating
HARD_RECOVERY_COOLDOWN = 600   # min seconds between reboots
MAX_REBOOTS_IN_WINDOW = 2      # max reboots per 1h (avoid reboot loop)
REBOOT_WINDOW = 3600

APP_PACKAGE = "com.mgt.r1voice"
APP_ACTIVITY = "com.mgt.r1voice/.MainActivity"
APP_SERVICE = "com.mgt.r1voice/.VoiceService"
PHICOMM_PACKAGES = [
    "com.phicomm.speaker",
    "com.phicomm.speaker.device",
    "com.phicomm.speaker.player",
]
SERVER_ADDR = f"ws://{SERVER_IP}:{WS_PORT}"

ADB_BIN = "adb"
ADB_TIMEOUT = 20

# --- ADB helpers ----------------------------------------------------------

def adb(*args, timeout=ADB_TIMEOUT):
    """Run adb -s R1 ... command. Returns (rc, stdout, stderr)."""
    cmd = [ADB_BIN, "-s", R1_ADB, *args]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except subprocess.TimeoutExpired:
        logger.warning(f"adb {' '.join(args)} timed out after {timeout}s")
        return -1, "", "timeout"
    except Exception as e:
        logger.error(f"adb error: {e}")
        return -1, "", str(e)


def adb_ensure_connected() -> bool:
    """Ensure ADB transport to R1 is alive. Returns True if device online."""
    # Fast path: check current state
    rc, out, _ = adb("get-state", timeout=8)
    if rc == 0 and "device" in out:
        return True
    # Reconnect: kill-server can help clear stale transport
    subprocess.run([ADB_BIN, "kill-server"], capture_output=True, timeout=15)
    time.sleep(1)
    subprocess.run([ADB_BIN, "start-server"], capture_output=True, timeout=15)
    time.sleep(1)
    rc, out, _ = adb("connect", R1_ADB, timeout=15)
    logger.info(f"adb connect: {out}")
    time.sleep(2)
    rc, out, _ = adb("get-state", timeout=8)
    return rc == 0 and "device" in out


def adb_force_stop(pkg: str):
    rc, out, err = adb("shell", "am", "force-stop", pkg, timeout=15)
    if rc != 0:
        logger.warning(f"force-stop {pkg} rc={rc} err={err}")


def start_app():
    adb_force_stop(APP_PACKAGE)
    time.sleep(1)
    adb("shell", "am", "start", "-n", APP_ACTIVITY, timeout=15)
    time.sleep(3)
    adb("shell", "am", "startservice", "-n", APP_SERVICE,
        "--es", "server_addr", SERVER_ADDR, timeout=15)


def kill_phicomm():
    for pkg in PHICOMM_PACKAGES:
        adb_force_stop(pkg)


def reboot_r1():
    logger.warning("⚡ HARD recovery: adb reboot R1")
    adb("reboot", timeout=15)


def r1_online() -> bool:
    """Check R1 reachable via ICMP."""
    try:
        r = subprocess.run(["ping", "-c", "1", "-W", "2", R1_IP],
                           capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


# --- State file -----------------------------------------------------------

def read_state() -> dict:
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


# --- Recovery ladder ------------------------------------------------------

class Watchdog:
    def __init__(self):
        self.last_soft = 0.0
        self.last_hard = 0.0
        self.reboot_times = []   # timestamps of reboots within window
        self.start_ts = time.time()
        self.state = "initial"
        self.last_state_log = 0.0

    def _log_state(self, new_state: str):
        now = time.time()
        if new_state != self.state or now - self.last_state_log > 60:
            logger.info(f"watchdog state: {new_state}")
            self.state = new_state
            self.last_state_log = now

    def _reboots_recent(self) -> int:
        now = time.time()
        self.reboot_times = [t for t in self.reboot_times if now - t < REBOOT_WINDOW]
        return len(self.reboot_times)

    def soft_recover(self, reason: str):
        now = time.time()
        # ADB unavailable → R1 may be rebooting; retry next check without
        # consuming cooldown (otherwise recovery would be delayed 120s)
        if not adb_ensure_connected():
            logger.info(f"soft recovery: ADB unavailable (R1 booting?), "
                        f"will retry next check, reason={reason}")
            return
        if now - self.last_soft < SOFT_RECOVERY_COOLDOWN:
            logger.info(f"soft recovery skipped (cooldown), reason={reason}")
            return
        self.last_soft = now
        logger.warning(f"🔧 SOFT recovery triggered: {reason}")
        kill_phicomm()
        start_app()

    def hard_recover(self, reason: str):
        now = time.time()
        if self._reboots_recent() >= MAX_REBOOTS_IN_WINDOW:
            logger.error("hard recovery: too many reboots in window, skipping "
                         "— manual intervention needed")
            return
        if now - self.last_hard < HARD_RECOVERY_COOLDOWN:
            logger.info(f"hard recovery skipped (cooldown), reason={reason}")
            return
        if not adb_ensure_connected():
            # R1 unreachable — don't consume cooldown, retry next check
            logger.info(f"hard recovery: ADB unavailable (R1 unreachable), "
                        f"will retry next check, reason={reason}")
            return
        self.last_hard = now
        self.reboot_times.append(now)
        reboot_r1()
        self._wait_for_boot()

    def _wait_for_boot(self, timeout=180):
        """Wait for R1 to come back online after reboot, then restore app."""
        logger.info(f"waiting for R1 boot (max {timeout}s)...")
        deadline = time.time() + timeout
        while time.time() < deadline:
            if adb_ensure_connected():
                logger.info("R1 ADB back online after reboot")
                # Give Android a bit more time to settle
                time.sleep(15)
                # BootReceiver is disabled; restore app manually
                kill_phicomm()
                start_app()
                return True
            time.sleep(5)
        logger.error("R1 did not come back after reboot")
        return False

    def check_once(self):
        state = read_state()
        now = time.time()
        connected = state.get("connected", False)
        last_frame = state.get("last_frame_time", 0.0)
        server_time = state.get("server_time", 0.0)

        # Case 0: state file missing/stale → server.py not running (or just started)
        if not state or now - server_time > 30:
            self._log_state("server-down")
            logger.warning("state file stale/missing — is server.py running? "
                           "systemd Restart=always should have restarted it")
            return

        if connected:
            # CRITICAL: during TTS playback the R1 mutes its mic (to avoid
            # echo), so no audio frames arrive while state == "speaking".
            # That is EXPECTED, not a stall — never reboot mid-speech.
            # (2026-08-01: this was misdiagnosed as a deadlock and the R1 was
            # rebooted while happily narrating a long answer.)
            cur_state = state.get("state", "idle")
            if cur_state == "speaking":
                self._log_state("speaking")
                return
            if now - last_frame > STALL_TIMEOUT:
                self._log_state("stalled")
                # Ladder: soft first, then hard if soft didn't help.
                if self.last_soft == 0:
                    # Never soft-recovered yet → try soft
                    self.soft_recover(f"audio stalled for {now - last_frame:.0f}s")
                elif now - self.last_soft < SOFT_OBSERVE_WINDOW:
                    # Soft recovery running — give it time to take effect
                    self._log_state("stalled-soft-recovering")
                elif now - self.last_hard < HARD_RECOVERY_COOLDOWN:
                    # Hard recovery cooling down — wait
                    self._log_state("stalled-hard-cooldown")
                else:
                    # Soft already tried and observed → escalate to hard
                    self.hard_recover(f"audio still stalled {now - last_frame:.0f}s "
                                      f"after soft recovery")
            else:
                self._log_state("healthy")
        else:
            # Not connected: either app crashed, WS died, or R1 rebooted itself
            if now - last_frame > NOT_CONNECTED_TIMEOUT and last_frame > 0:
                self._log_state("disconnected")
                self.soft_recover(f"R1 disconnected for {now - last_frame:.0f}s")
            elif last_frame == 0 and now - self.start_ts > 120:
                # Server has never seen a frame since watchdog start
                self._log_state("never-connected")
                self.soft_recover("no frames since server start")
            else:
                self._log_state("waiting-first-connect")

    def run(self):
        logger.info(f"R1 watchdog started (R1={R1_ADB}, stall={STALL_TIMEOUT}s, "
                    f"not_connected={NOT_CONNECTED_TIMEOUT}s)")
        while True:
            try:
                self.check_once()
            except Exception as e:
                logger.error(f"watchdog check error: {e}", exc_info=True)
            time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    Watchdog().run()
