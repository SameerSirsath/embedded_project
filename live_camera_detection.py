"""
================================================================================
LIVE REAL-TIME OBSTACLE DETECTION VIA CAMERA / iVCam MOBILE
================================================================================
Designed for: Assistive Navigation (Blind / Visually Impaired) & Edge Robotics
Features:
  - Real-time mobile camera (iVCam), webcam, or ESP32-CAM stream detection
  - Live Camera Switcher (Press [n] to toggle between Laptop Camera and iVCam Mobile!)
  - CUDA GPU acceleration (RTX 3050 FP16 inference for 60+ FPS)
  - 3-Zone Corridor HUD: Left | Center (Direct Path Hazard) | Right
  - Proximity Risk Analysis: Critical (<1.5m) | Warning (2-3m) | Info (>3m)
  - Non-blocking Asynchronous Voice Alerts (Windows Speech Engine)
  - Interactive Key Controls:
      [n]         : Switch Camera (Toggle between Laptop Webcam and iVCam Mobile)
      [q] / [ESC] : Quit
      [s]         : Save snapshot to runs/live_snapshots/
      [c]         : Toggle Corridor HUD
      [v]         : Toggle Voice Alerts
      [+] / [-]   : Adjust Confidence Threshold live

Run in PyCharm:
  Right-click 'live_camera_detection.py' -> Run 'live_camera_detection'
Or via Terminal:
  python live_camera_detection.py --cam 1         # Run directly on iVCam mobile camera
  python live_camera_detection.py --cam 0         # Run on laptop integrated camera
================================================================================
"""

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ---------------------------------------------------------
# Voice Alert Engine (Asynchronous Non-Blocking Worker)
# ---------------------------------------------------------
class VoiceAlertWorker:
    """Speaks alerts in a background thread so the video stream never stutters."""
    def __init__(self, enabled=True, cooldown=2.5):
        self.enabled = enabled
        self.cooldown = cooldown
        self.last_spoken_time = 0.0
        self.last_spoken_msg = ""
        self.lock = threading.Lock()
        self._tts_engine = None
        self._init_engine()

    def _init_engine(self):
        try:
            import pyttsx3
            self._tts_engine = pyttsx3.init()
            self._tts_engine.setProperty('rate', 170)
            self._tts_engine.setProperty('volume', 0.9)
        except Exception:
            self._tts_engine = None

    def speak(self, text: str):
        if not self.enabled:
            return

        now = time.time()
        # Cooldown prevents repetitive speech spamming
        if now - self.last_spoken_time < self.cooldown:
            return

        with self.lock:
            self.last_spoken_time = now
            self.last_spoken_msg = text

        threading.Thread(target=self._speak_thread, args=(text,), daemon=True).start()

    def _speak_thread(self, text: str):
        try:
            if self._tts_engine is not None:
                self._tts_engine.say(text)
                self._tts_engine.runAndWait()
            else:
                # Fallback to Windows native PowerShell speech synthesizer
                import subprocess
                cmd = f"Add-Type -AssemblyName System.speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}');"
                subprocess.run(
                    ["powershell", "-NoProfile", "-Command", cmd],
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                    timeout=3.0
                )
        except Exception:
            pass


# ---------------------------------------------------------
# Model Resolution Helper
# ---------------------------------------------------------
def find_model_weights(weights_arg: str) -> str:
    candidates = [
        weights_arg,
        "weights/obstacle_yolo_best.pt",
        "weights/obstacle_yolo_best.onnx",
        "obstacle_detector/yolo_v8_obstacles/weights/best.pt",
        "yolov8n.pt"
    ]
    for c in candidates:
        if c and Path(c).exists():
            return str(Path(c).resolve())

    print("[!] Trained obstacle weights not found. Using pretrained 'yolov8n.pt'...")
    return "yolov8n.pt"


# ---------------------------------------------------------
# Camera Probe & Helper
# ---------------------------------------------------------
def probe_available_cameras():
    """Detects available camera indices and labels them for iVCam."""
    available = []
    for idx in range(4):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                label = "e2eSoft iVCam (Mobile Phone)" if idx == 1 else ("Integrated Laptop Camera" if idx == 0 else f"Camera {idx}")
                available.append((idx, label))
            cap.release()
    return available


def open_camera(cam_src, width=1280, height=720):
    """Opens a camera capture handle with optimal buffer and dimensions."""
    src = int(cam_src) if str(cam_src).isdigit() else str(cam_src)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened() and isinstance(src, int) and os.name == 'nt':
        cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


# ---------------------------------------------------------
# Spatial Navigation Corridor Analysis
# ---------------------------------------------------------
def analyze_frame_corridors(w: int, h: int, boxes, names_map):
    """
    Categorizes detections into Left, Center (Direct Hazard), and Right corridors.
    Estimates proximity risk based on vertical bounding box scale.
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

        # Spatial Corridor Zone
        if cx < left_boundary:
            zone = "Left"
            zone_code = 1
        elif cx > right_boundary:
            zone = "Right"
            zone_code = 2
        else:
            zone = "Center Path"
            zone_code = 0  # Highest navigational priority

        # Proximity Risk Estimation
        if height_ratio > 0.42:
            proximity = "Critical (< 1.5m)"
            urgency = 3
        elif height_ratio > 0.22:
            proximity = "Warning (2-3m)"
            urgency = 2
        else:
            proximity = "Distant (> 3m)"
            urgency = 1

        alerts.append({
            "label": label,
            "conf": conf,
            "zone": zone,
            "zone_code": zone_code,
            "proximity": proximity,
            "urgency": urgency,
            "bbox": (int(x1), int(y1), int(x2), int(y2))
        })

    # Sort so closest hazards in the center path come first
    alerts.sort(key=lambda a: (-a["urgency"], a["zone_code"]))
    return alerts


# ---------------------------------------------------------
# HUD Drawing
# ---------------------------------------------------------
def draw_live_hud(frame, alerts, fps: float, conf_thresh: float, show_corridors: bool, voice_on: bool, cam_label: str):
    h, w = frame.shape[:2]
    annotated = frame.copy()

    # 1. Corridor Guidelines (if enabled)
    line1 = int(w * 0.33)
    line2 = int(w * 0.66)
    if show_corridors:
        overlay = annotated.copy()
        cv2.line(overlay, (line1, 0), (line1, h), (0, 255, 255), 2)
        cv2.line(overlay, (line2, 0), (line2, h), (0, 255, 255), 2)
        # Highlight center danger path with subtle red tint
        cv2.rectangle(overlay, (line1, 0), (line2, h), (0, 0, 120), -1)
        cv2.addWeighted(overlay, 0.22, annotated, 0.78, 0, annotated)

    # 2. Top Header Bar
    cv2.rectangle(annotated, (0, 0), (w, 42), (20, 20, 20), -1)
    if show_corridors:
        cv2.putText(annotated, "LEFT ZONE", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(annotated, "CENTER (DIRECT HAZARD)", (line1 + 15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
        cv2.putText(annotated, "RIGHT ZONE", (line2 + 20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    # Stats info on top right
    stats_text = f"{cam_label} | FPS: {fps:4.1f} | Conf: {conf_thresh:.2f} | Voice: {'ON' if voice_on else 'OFF'}"
    (tw, _), _ = cv2.getTextSize(stats_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    cv2.putText(annotated, stats_text, (w - tw - 15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 180), 2)

    # 3. Draw Detected Obstacle Bounding Boxes
    for a in alerts:
        x1, y1, x2, y2 = a["bbox"]
        urgency = a["urgency"]

        # Color-coded by hazard level
        if urgency == 3:
            box_color = (0, 0, 255)       # Red for Critical
        elif urgency == 2:
            box_color = (0, 165, 255)     # Orange for Warning
        else:
            box_color = (0, 230, 0)       # Green for Info

        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

        # Label tag with zone and proximity
        tag = f"{a['label'].upper()} {a['conf']*100:.0f}% [{a['zone']}] - {a['proximity']}"
        (ltw, lth), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(annotated, (x1, max(0, y1 - lth - 8)), (x1 + ltw + 6, y1), box_color, -1)
        cv2.putText(annotated, tag, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    # 4. Bottom Navigational Alert Ticker
    cv2.rectangle(annotated, (0, h - 45), (w, h), (15, 15, 15), -1)
    if alerts:
        top = alerts[0]
        if top["urgency"] == 3:
            banner_col = (0, 0, 255)
            prefix = "CRITICAL HAZARD"
        elif top["urgency"] == 2:
            banner_col = (0, 180, 255)
            prefix = "WARNING"
        else:
            banner_col = (100, 220, 100)
            prefix = "INFO"

        msg = f">> [{prefix}] {top['label'].upper()} in {top['zone']} - {top['proximity']} <<"
    else:
        banner_col = (0, 255, 100)
        msg = ">> PATH CLEAR - No obstacles detected in direct path <<"

    cv2.putText(annotated, msg, (20, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.65, banner_col, 2, cv2.LINE_AA)

    # Controls help in bottom right
    help_str = "[n] Switch Cam  [c] Grid  [v] Voice  [s] Snap  [+/-] Conf  [q] Quit"
    (hw, _), _ = cv2.getTextSize(help_str, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.putText(annotated, help_str, (w - hw - 15, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

    return annotated


# ---------------------------------------------------------
# Main Live Streaming Loop
# ---------------------------------------------------------
def run_live_camera(args):
    # 1. Resolve Model
    model_path = find_model_weights(args.weights)
    print("==================================================================")
    print(f" [✓] Model File : {model_path}")
    print(f" [✓] Device     : {args.device}")
    print(f" [✓] Voice Ann  : {'ENABLED' if not args.no_voice else 'DISABLED'}")
    print("==================================================================")

    # 2. Probe Available Cameras
    cams = probe_available_cameras()
    print("[+] Detected Camera Devices:")
    for c_idx, c_lbl in cams:
        print(f"    [{c_idx}] {c_lbl}")

    # Determine default camera index:
    # If user passed an argument use it, otherwise prefer iVCam (index 1) if available!
    if args.cam is not None:
        current_cam_src = args.cam
    elif any(c[0] == 1 for c in cams):
        current_cam_src = 1
        print("    -> Automatically selected [1] e2eSoft iVCam (Mobile Camera)")
    else:
        current_cam_src = 0

    # Load YOLO Model
    model = YOLO(model_path)
    if args.device == "0" and torch.cuda.is_available():
        model.to("cuda")

    # Open Camera
    cap = open_camera(current_cam_src, args.width, args.height)
    if not cap.isOpened():
        print(f"\n[!] ERROR: Could not open camera source '{current_cam_src}'.")
        print("    Trying fallback camera index 0...")
        current_cam_src = 0
        cap = open_camera(0, args.width, args.height)
        if not cap.isOpened():
            print("[!] Failed to open any camera.")
            return

    # Snapshot directory
    snapshot_dir = Path("runs/live_snapshots")
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Voice worker
    voice_worker = VoiceAlertWorker(enabled=(not args.no_voice))

    print("\n[+] LIVE CAMERA STREAM RUNNING!")
    print("    Press [n] to toggle between Laptop Camera and iVCam Mobile Camera.")
    print("    Press [q] or [ESC] to exit.")
    print("    Press [s] to save a snapshot.")
    print("    Press [c] to toggle Corridor HUD.")
    print("    Press [v] to toggle Voice alerts.")
    print("    Press [+] or [-] to adjust confidence.\n")

    conf_thresh = args.conf
    show_corridors = True
    fps_history = []
    window_name = "Live Obstacle Detection (iVCam Mobile & Assistive Mobility)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[!] Waiting for frame from camera...")
                time.sleep(0.1)
                continue

            h, w = frame.shape[:2]

            # 3. Model Inference (FP16 half-precision on GPU for 60+ FPS)
            results = model(
                frame,
                conf=conf_thresh,
                imgsz=args.imgsz,
                device=args.device,
                half=(args.device == "0" and torch.cuda.is_available()),
                verbose=False
            )[0]

            # 4. Spatial Corridor & Proximity Analysis
            alerts = analyze_frame_corridors(w, h, results.boxes, results.names)

            # 5. Voice Alert Trigger (announces closest hazard)
            if alerts and voice_worker.enabled:
                top_hazard = alerts[0]
                if top_hazard["urgency"] >= 2:  # Warning or Critical
                    spoken_msg = f"{top_hazard['label']} in {top_hazard['zone']}"
                    voice_worker.speak(spoken_msg)

            # 6. Calculate smoothed FPS
            dt = time.perf_counter() - t0
            current_fps = 1.0 / max(dt, 1e-4)
            fps_history.append(current_fps)
            if len(fps_history) > 20:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)

            # Determine label for HUD
            cam_str = str(current_cam_src)
            label_for_hud = "iVCam Mobile" if cam_str == "1" else ("Laptop Cam" if cam_str == "0" else f"Cam {cam_str}")

            # 7. Render HUD
            display_frame = draw_live_hud(
                frame,
                alerts,
                fps=avg_fps,
                conf_thresh=conf_thresh,
                show_corridors=show_corridors,
                voice_on=voice_worker.enabled,
                cam_label=label_for_hud
            )

            cv2.imshow(window_name, display_frame)

            # 8. Keyboard Controls
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), 27]:  # 'q' or ESC
                print("[*] Stream stopped by user.")
                break
            elif key == ord('n'):
                # Switch camera (Toggle between 0 and 1)
                new_cam = 0 if str(current_cam_src) == "1" else 1
                print(f"\n[*] Switching camera from [{current_cam_src}] to [{new_cam}]...")
                cap.release()
                new_cap = open_camera(new_cam, args.width, args.height)
                if new_cap.isOpened():
                    cap = new_cap
                    current_cam_src = new_cam
                    print(f"[✓] Successfully switched to Camera [{new_cam}]!")
                else:
                    print(f"[!] Could not open Camera [{new_cam}], reverting back...")
                    cap = open_camera(current_cam_src, args.width, args.height)
            elif key == ord('s'):
                snap_path = snapshot_dir / f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(str(snap_path), display_frame)
                print(f"[✓] Saved snapshot: {snap_path}")
            elif key == ord('c'):
                show_corridors = not show_corridors
                print(f"[*] Corridor HUD: {'ENABLED' if show_corridors else 'DISABLED'}")
            elif key == ord('v'):
                voice_worker.enabled = not voice_worker.enabled
                print(f"[*] Voice Alerts: {'ENABLED' if voice_worker.enabled else 'DISABLED'}")
            elif key in [ord('+'), ord('=')]:
                conf_thresh = min(0.95, round(conf_thresh + 0.05, 2))
                print(f"[*] Confidence threshold: {conf_thresh}")
            elif key in [ord('-'), ord('_')]:
                conf_thresh = max(0.10, round(conf_thresh - 0.05, 2))
                print(f"[*] Confidence threshold: {conf_thresh}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[✓] Camera released and windows closed cleanly.")


# ---------------------------------------------------------
# Entry Point
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Live Real-Time Obstacle Detection via Camera / iVCam Mobile",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--cam", default=None, help="Camera index (0 for Laptop, 1 for iVCam Mobile)")
    parser.add_argument("--weights", default="weights/obstacle_yolo_best.pt", help="Model weights path")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution")
    parser.add_argument("--width", type=int, default=1280, help="Camera capture width")
    parser.add_argument("--height", type=int, default=720, help="Camera capture height")
    parser.add_argument(
        "--device",
        default="0" if torch.cuda.is_available() else "cpu",
        help="Device: '0' for CUDA GPU, 'cpu' for CPU"
    )
    parser.add_argument("--no-voice", action="store_true", help="Disable spoken voice alerts")

    args = parser.parse_args()
    run_live_camera(args)


if __name__ == "__main__":
    main()
