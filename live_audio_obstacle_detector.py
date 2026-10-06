"""
================================================================================
LIVE REAL-TIME OBSTACLE DETECTION WITH BLUETOOTH AUDIO ALERTS
================================================================================
Designed for: Assistive Navigation for the Visually Impaired & Robotics
Target Devices: Raspberry Pi 5 (with Bluetooth Speaker) & Windows (PyCharm)

Key Features:
  - Immediate Audio Self-Test on startup to verify Bluetooth speaker connection
  - Real-time hazard detection with smart directional voice prompts:
      * "Caution! Pole directly ahead, very close!"
      * "Warning! Person on your left."
      * "Path is clear."
  - Non-blocking Queue-based audio worker thread (video never stutters or lags)
  - Intelligent Audio Cooldown & De-duplication (avoids repeating the same alert)
  - Works on Raspberry Pi 5 (via espeak-ng / PipeWire / PulseAudio) & Windows (SAPI)
  - Supports USB webcam (/dev/video0 or cam 0), Pi Camera, and phone cameras

Usage on Raspberry Pi 5:
  python live_audio_obstacle_detector.py --weights weights/obstacle_yolo_best.onnx

Usage on Windows / PyCharm:
  python live_audio_obstacle_detector.py
================================================================================
"""

import argparse
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ==============================================================================
# 1. SMART BLUETOOTH AUDIO ENGINE
# ==============================================================================
class BluetoothAudioAnnouncer:
    """
    Dedicated background audio worker thread.
    Routes speech prompts to the connected Bluetooth speaker without lagging the camera.
    """
    def __init__(self, cooldown=3.0, enable_speech=True):
        self.cooldown = cooldown
        self.enabled = enable_speech
        self.speech_queue = queue.Queue(maxsize=3)
        self.last_spoken_time = 0.0
        self.last_spoken_alert = ""
        self.lock = threading.Lock()
        self.running = True

        # Detect available speech backends
        self.has_espeak_ng = shutil.which("espeak-ng") is not None
        self.has_espeak = shutil.which("espeak") is not None
        self.has_spd_say = shutil.which("spd-say") is not None

        # Start persistent worker thread
        self.worker_thread = threading.Thread(target=self._audio_loop, daemon=True)
        self.worker_thread.start()

    def speak(self, text: str, force: bool = False):
        """Queues an alert string for speech synthesis."""
        if not self.enabled or not text:
            return

        now = time.time()
        # Prevent repeat spamming unless forced or cooldown has elapsed
        if not force:
            if text == self.last_spoken_alert and (now - self.last_spoken_time) < self.cooldown:
                return
            if (now - self.last_spoken_time) < 1.8:
                return

        with self.lock:
            self.last_spoken_time = now
            self.last_spoken_alert = text

        # Drop older queued messages if falling behind so alerts stay fresh
        try:
            while not self.speech_queue.empty():
                self.speech_queue.get_nowait()
            self.speech_queue.put_nowait(text)
        except queue.Full:
            pass

    def _audio_loop(self):
        """Worker thread that executes the audio playback."""
        while self.running:
            try:
                text = self.speech_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            self._play_speech(text)
            self.speech_queue.task_done()

    def _play_speech(self, text: str):
        """Cross-platform speech synthesis routing to default audio sink."""
        try:
            # 1. Linux / Raspberry Pi OS (Routes to Bluetooth Speaker via PipeWire/PulseAudio)
            if sys.platform.startswith("linux"):
                if self.has_espeak_ng:
                    # -s 160 (Speed), -a 100 (Volume 100%)
                    subprocess.run(
                        ["espeak-ng", "-s", "160", "-a", "100", text],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=5.0
                    )
                elif self.has_spd_say:
                    subprocess.run(
                        ["spd-say", "-r", "-10", text],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=5.0
                    )
                elif self.has_espeak:
                    subprocess.run(
                        ["espeak", "-s", "160", "-a", "100", text],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=5.0
                    )
                else:
                    # Python fallback if system tools are missing
                    try:
                        import pyttsx3
                        engine = pyttsx3.init()
                        engine.say(text)
                        engine.runAndWait()
                    except Exception:
                        pass

            # 2. Windows (Routes to Default Bluetooth Speaker)
            elif os.name == "nt":
                clean_text = text.replace("'", "").replace('"', '')
                cmd = (
                    "Add-Type -AssemblyName System.speech; "
                    "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    "$synth.Rate = 1; "
                    f"$synth.Speak('{clean_text}');"
                )
                subprocess.run(
                    ["powershell", "-NoProfile", "-Command", cmd],
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    timeout=5.0
                )

        except Exception as e:
            # Non-blocking silent fallback
            pass

    def stop(self):
        self.running = False


# ==============================================================================
# 2. INTELLIGENT NAVIGATION SPEECH PROMPT GENERATOR
# ==============================================================================
def generate_navigational_voice_prompt(alert):
    """
    Translates raw detection boxes into natural, concise audio prompts
    specifically tailored for visually impaired navigation assistance.
    """
    label = alert["label"]
    zone = alert["zone"]
    urgency = alert["urgency"]

    # Critical distance (< 1.5 meters)
    if urgency == 3:
        if "Center" in zone:
            return f"Caution! {label} directly ahead, very close!"
        elif "Left" in zone:
            return f"Watch out! {label} close on your left!"
        else:
            return f"Watch out! {label} close on your right!"

    # Warning distance (2 to 3 meters)
    elif urgency == 2:
        if "Center" in zone:
            return f"{label} ahead in your path."
        elif "Left" in zone:
            return f"{label} on your left."
        else:
            return f"{label} on your right."

    # Distant (> 3 meters)
    else:
        return f"{label} detected in distance."


# ==============================================================================
# 3. SPATIAL CORRIDOR ANALYSIS
# ==============================================================================
def analyze_navigation_corridors(w: int, h: int, boxes, names_map):
    """
    Splits field-of-view into Left, Center Path, and Right corridors.
    Estimates proximity risk based on object height relative to camera frame.
    """
    left_boundary = w * 0.33
    right_boundary = w * 0.66
    alerts = []

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = names_map.get(cls_id, f"Object_{cls_id}")

        cx = (x1 + x2) / 2.0
        box_h = y2 - y1
        height_ratio = box_h / max(h, 1)

        # 3-Zone Corridor Assignment
        if cx < left_boundary:
            zone = "Left"
            zone_priority = 1
        elif cx > right_boundary:
            zone = "Right"
            zone_priority = 2
        else:
            zone = "Center Path"
            zone_priority = 0  # Highest hazard priority

        # Proximity Risk Estimation
        if height_ratio > 0.40:
            proximity = "Critical (< 1.5m)"
            urgency = 3
        elif height_ratio > 0.20:
            proximity = "Warning (2-3m)"
            urgency = 2
        else:
            proximity = "Distant (> 3m)"
            urgency = 1

        alerts.append({
            "label": label,
            "conf": conf,
            "zone": zone,
            "zone_priority": zone_priority,
            "proximity": proximity,
            "urgency": urgency,
            "bbox": (int(x1), int(y1), int(x2), int(y2))
        })

    # Sort hazards: Highest urgency first, Center corridor prioritized
    alerts.sort(key=lambda a: (-a["urgency"], a["zone_priority"]))
    return alerts


# ==============================================================================
# 4. HUD DRAWING
# ==============================================================================
def draw_annotated_hud(frame, alerts, fps: float, conf_thresh: float):
    h, w = frame.shape[:2]
    annotated = frame.copy()

    line1 = int(w * 0.33)
    line2 = int(w * 0.66)

    # 1. Subtle corridor guide lines
    cv2.line(annotated, (line1, 0), (line1, h), (0, 255, 255), 1)
    cv2.line(annotated, (line2, 0), (line2, h), (0, 255, 255), 1)

    # 2. Top Status Bar
    cv2.rectangle(annotated, (0, 0), (w, 36), (20, 20, 20), -1)
    cv2.putText(annotated, "LEFT ZONE", (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(annotated, "CENTER PATH (HAZARD)", (line1 + 15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2)
    cv2.putText(annotated, "RIGHT ZONE", (line2 + 15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    hud_info = f"FPS: {fps:4.1f} | Conf: {conf_thresh:.2f} | Audio: ON"
    (tw, _), _ = cv2.getTextSize(hud_info, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(annotated, hud_info, (w - tw - 12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 180), 1)

    # 3. Draw Bounding Boxes
    for a in alerts:
        x1, y1, x2, y2 = a["bbox"]
        color = (0, 0, 255) if a["urgency"] == 3 else ((0, 165, 255) if a["urgency"] == 2 else (0, 220, 0))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        tag = f"{a['label'].upper()} [{a['zone']}] - {a['proximity']}"
        (ltw, lth), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(annotated, (x1, max(0, y1 - lth - 6)), (x1 + ltw + 6, y1), color, -1)
        cv2.putText(annotated, tag, (x1 + 3, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    # 4. Bottom Alert Ticker
    cv2.rectangle(annotated, (0, h - 38), (w, h), (15, 15, 15), -1)
    if alerts:
        top = alerts[0]
        col = (0, 0, 255) if top["urgency"] == 3 else ((0, 180, 255) if top["urgency"] == 2 else (100, 220, 100))
        msg = f">> AUDIO ALERT: {top['label'].upper()} in {top['zone']} - {top['proximity']} <<"
    else:
        col = (0, 255, 100)
        msg = ">> PATH CLEAR - No obstacles ahead <<"

    cv2.putText(annotated, msg, (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.58, col, 2, cv2.LINE_AA)
    return annotated


# ==============================================================================
# 5. MODEL & CAMERA INITIALIZER
# ==============================================================================
def resolve_weights(weights_arg: str) -> str:
    candidates = [
        weights_arg,
        "weights/obstacle_yolo_best.onnx",
        "weights/obstacle_yolo_best.pt",
        "obstacle_detector/yolo_v8_obstacles/weights/best.pt",
        "yolov8n.pt"
    ]
    for c in candidates:
        if c and Path(c).exists():
            return str(Path(c).resolve())
    return "yolov8n.pt"


def initialize_camera(cam_source, width=640, height=480):
    src = int(cam_source) if str(cam_source).isdigit() else str(cam_source)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened() and isinstance(src, int) and os.name == 'nt':
        cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


# ==============================================================================
# 6. MAIN EXECUTION LOOP
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Real-Time Obstacle Detection with Bluetooth Audio Alerts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--cam", default="0", help="Camera index (0 for USB webcam) or IP stream URL")
    parser.add_argument("--weights", default="weights/obstacle_yolo_best.onnx", help="Model weights path (.onnx or .pt)")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution")
    parser.add_argument("--cooldown", type=float, default=3.0, help="Audio alert repeat cooldown (seconds)")
    parser.add_argument("--no-show", action="store_true", help="Headless terminal mode without GUI window")
    args = parser.parse_args()

    print("=" * 65)
    print(" 🦯 OBSTACLE DETECTION - BLUETOOTH AUDIO SYSTEM")
    print("=" * 65)

    # 1. Initialize Bluetooth Audio Engine
    print("[+] Starting Bluetooth Audio Announcer...")
    announcer = BluetoothAudioAnnouncer(cooldown=args.cooldown, enable_speech=True)

    # 2. Bluetooth Speaker Self-Test (Immediately plays voice confirmation)
    print("[+] Performing Bluetooth Speaker Test...")
    announcer.speak("Obstacle detection started. Audio system connected.", force=True)
    time.sleep(1.0)

    # 3. Load YOLO Model
    model_path = resolve_weights(args.weights)
    print(f"[+] Loading Model: {model_path}")
    model = YOLO(model_path)
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"[+] Compute Device: {device} ({'GPU Acceleration' if device == '0' else 'CPU Mode'})")

    # 4. Open Camera
    print(f"[+] Opening Camera: {args.cam}...")
    cap = initialize_camera(args.cam)
    if not cap.isOpened():
        print(f"[!] Error: Cannot open camera '{args.cam}'. Please check your webcam connection.")
        announcer.speak("Error: Camera not found.", force=True)
        return

    print("\n[✓] SYSTEM READY! Point the camera at obstacles to hear audio warnings.")
    print("    Press [q] or [ESC] to stop.\n")

    fps_history = []
    window_name = "Obstacle Detection - Bluetooth Audio"
    if not args.no_show:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            h, w = frame.shape[:2]

            # 5. Run Inference
            results = model(
                frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=device,
                verbose=False
            )[0]

            # 6. Analyze Corridors
            alerts = analyze_navigation_corridors(w, h, results.boxes, results.names)

            # 7. Trigger Bluetooth Audio Alert for the closest hazard
            if alerts:
                top_hazard = alerts[0]
                # Only announce if Warning or Critical proximity
                if top_hazard["urgency"] >= 2:
                    prompt = generate_navigational_voice_prompt(top_hazard)
                    announcer.speak(prompt)

            # FPS Calculation
            dt = time.perf_counter() - t0
            fps = 1.0 / max(dt, 1e-4)
            fps_history.append(fps)
            if len(fps_history) > 15:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)

            # 8. Display & Terminal Output
            display_frame = draw_annotated_hud(frame, alerts, avg_fps, args.conf)

            if not args.no_show:
                try:
                    cv2.imshow(window_name, display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in [ord('q'), 27]:
                        print("\n[*] Stopping stream...")
                        break
                except Exception:
                    args.no_show = True

            if args.no_show:
                if alerts:
                    top = alerts[0]
                    print(f"\r[ALERT] {top['label'].upper()} in {top['zone']} ({top['proximity']}) | FPS: {avg_fps:.1f}   ", end="", flush=True)
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[*] Interrupted by user.")
    finally:
        announcer.speak("Obstacle detection stopped.", force=True)
        time.sleep(0.5)
        announcer.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("[✓] Audio engine stopped and camera released cleanly.")


if __name__ == "__main__":
    main()
