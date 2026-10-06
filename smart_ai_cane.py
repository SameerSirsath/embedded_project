"""
================================================================================
SMART AI NAVIGATION CANE WITH YOLO OBSTACLE DETECTION & HARDWARE INTEGRATION
================================================================================
Integrated Features:
  - Computer Vision: YOLOv8 / ONNX real-time obstacle detection
  - 3-Zone Spatial Corridors: Left | Center Path (Direct Hazard) | Right
  - Non-blocking Bluetooth Audio alerts (espeak-ng / espeak background queue)
  - Ultrasonic sensor distance measurement (HC-SR04) + Active Buzzer
  - Emergency SOS Button with GPS live location coordinates (NEO-6M / Serial)
  - Instant Telegram emergency messaging with Google Maps link
  - Multi-threaded VideoStream for smooth real-time performance
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
import requests
from ultralytics import YOLO

# GPS imports with safe fallback
try:
    import serial
    import pynmea2
    HAS_GPS_LIBS = True
except ImportError:
    HAS_GPS_LIBS = False
    print("[!] 'pyserial' or 'pynmea2' not installed. GPS will run in simulated mode.")

# GPIO import with safe fallback for testing
try:
    import RPi.GPIO as GPIO
    HAS_RPI_GPIO = True
except (ImportError, RuntimeError):
    HAS_RPI_GPIO = False
    print("[!] RPi.GPIO not detected. Running in simulation mode for GPIO pins.")


# =========================================================
# ---------------- TELEGRAM SETTINGS ----------------------
# =========================================================

BOT_TOKEN = "8781158016:AAEw_jaSW7kqGpDOIJPd9A9d98CcEzT254I"
CHAT_ID = "6019364300"


# =========================================================
# ---------------- GPIO SETTINGS --------------------------
# =========================================================

# Ultrasonic Pins (HC-SR04)
TRIG = 23
ECHO = 24

# SOS Emergency Button
BUTTON = 17

# Buzzer Alert
BUZZER = 18

if HAS_RPI_GPIO:
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    GPIO.setup(TRIG, GPIO.OUT)
    GPIO.setup(ECHO, GPIO.IN)

    GPIO.setup(BUTTON, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    GPIO.setup(BUZZER, GPIO.OUT)
    GPIO.output(BUZZER, GPIO.LOW)


# =========================================================
# ---------------- GPS SETTINGS ---------------------------
# =========================================================

gps_serial = None
if HAS_GPS_LIBS:
    try:
        gps_serial = serial.Serial("/dev/ttyS0", 9600, timeout=1)
    except Exception:
        gps_serial = None
        print("[!] GPS serial port /dev/ttyS0 not available. Location will be simulated if button is pressed.")


# =========================================================
# ---------------- ULTRASONIC FUNCTION --------------------
# =========================================================

def get_distance():
    """Measures distance in cm using the HC-SR04 ultrasonic sensor."""
    if not HAS_RPI_GPIO:
        return -1

    try:
        GPIO.output(TRIG, False)
        time.sleep(0.01)

        GPIO.output(TRIG, True)
        time.sleep(0.00001)
        GPIO.output(TRIG, False)

        pulse_start = None
        pulse_end = None
        timeout = time.time()

        while GPIO.input(ECHO) == 0:
            pulse_start = time.time()
            if pulse_start - timeout > 0.04:
                return -1

        while GPIO.input(ECHO) == 1:
            pulse_end = time.time()
            if pulse_end - timeout > 0.04:
                return -1

        if pulse_start is None or pulse_end is None:
            return -1

        duration = pulse_end - pulse_start
        distance = duration * 17150
        return round(distance, 1)
    except Exception:
        return -1


# =========================================================
# ---------------- GPS FUNCTION ---------------------------
# =========================================================

def get_gps_location():
    """Reads NMEA sentences from GPS module to extract latitude & longitude."""
    if gps_serial is None:
        return None, None

    start_time = time.time()
    while time.time() - start_time < 12:
        try:
            data = gps_serial.readline().decode('ascii', errors='replace')
            if data.startswith('$GPGGA'):
                msg = pynmea2.parse(data)
                latitude = msg.latitude
                longitude = msg.longitude
                if latitude != 0 and longitude != 0:
                    return latitude, longitude
        except Exception:
            pass

    return None, None


# =========================================================
# ---------------- TELEGRAM FUNCTION ----------------------
# =========================================================

def send_telegram_message(message):
    """Sends emergency alert message to Telegram bot in a background thread."""
    def _send():
        try:
            url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
            data = {
                "chat_id": CHAT_ID,
                "text": message
            }
            requests.post(url, data=data, timeout=5)
            print("[✓] Telegram SOS message sent successfully!")
        except Exception as e:
            print(f"[!] Failed to send Telegram message: {e}")

    threading.Thread(target=_send, daemon=True).start()


# =========================================================
# ---------------- NON-BLOCKING AUDIO ENGINE --------------
# =========================================================

class AsyncAudioAnnouncer:
    """
    Dedicated non-blocking audio worker thread.
    Speaks alerts through the Bluetooth speaker / 3.5mm jack
    without freezing the camera video stream or sensor loops!
    """
    def __init__(self, cooldown=3.0):
        self.cooldown = cooldown
        self.speech_queue = queue.Queue(maxsize=3)
        self.last_spoken_time = 0.0
        self.last_spoken_text = ""
        self.lock = threading.Lock()
        self.running = True

        self.has_espeak_ng = shutil.which("espeak-ng") is not None
        self.has_espeak = shutil.which("espeak") is not None
        self.has_spd_say = shutil.which("spd-say") is not None

        # Start background worker
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def speak(self, text: str, force: bool = False):
        if not text:
            return

        now = time.time()
        if not force:
            if text == self.last_spoken_text and (now - self.last_spoken_time) < self.cooldown:
                return
            if (now - self.last_spoken_time) < 1.8:
                return

        with self.lock:
            self.last_spoken_time = now
            self.last_spoken_text = text

        try:
            while not self.speech_queue.empty():
                self.speech_queue.get_nowait()
            self.speech_queue.put_nowait(text)
        except queue.Full:
            pass

    def _worker_loop(self):
        while self.running:
            try:
                text = self.speech_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            self._play(text)
            self.speech_queue.task_done()

    def _play(self, text: str):
        try:
            if sys.platform.startswith("linux"):
                if self.has_espeak_ng:
                    subprocess.run(["espeak-ng", "-s", "155", "-a", "100", text],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=4.0)
                elif self.has_spd_say:
                    subprocess.run(["spd-say", "-r", "-10", text],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=4.0)
                elif self.has_espeak:
                    subprocess.run(["espeak", "-s", "155", "-a", "100", text],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=4.0)
            elif os.name == "nt":
                clean = text.replace("'", "").replace('"', '')
                cmd = f"Add-Type -AssemblyName System.speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{clean}');"
                subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                               creationflags=subprocess.CREATE_NO_WINDOW, timeout=4.0)
        except Exception:
            pass

    def stop(self):
        self.running = False


# =========================================================
# ---------------- SPATIAL CORRIDOR ANALYSIS --------------
# =========================================================

def analyze_navigation_corridors(w: int, h: int, boxes, names_map):
    """
    Partitions camera view into Left, Center Path, and Right corridors.
    Computes spatial urgency: Center Path hazards have highest priority.
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

        # 3-Zone Corridor
        if cx < left_boundary:
            zone = "Left"
            zone_code = 1
        elif cx > right_boundary:
            zone = "Right"
            zone_code = 2
        else:
            zone = "Center Path"
            zone_code = 0  # Highest hazard priority

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
            "zone_code": zone_code,
            "proximity": proximity,
            "urgency": urgency,
            "bbox": (int(x1), int(y1), int(x2), int(y2))
        })

    # Sort so closest Center Path hazards come first
    alerts.sort(key=lambda a: (-a["urgency"], a["zone_code"]))
    return alerts


def generate_voice_alert(alert):
    """Formats natural directional speech prompt for the visually impaired."""
    label = alert["label"]
    zone = alert["zone"]
    urgency = alert["urgency"]

    if urgency == 3:
        if "Center" in zone:
            return f"Caution! {label} directly ahead, very close!"
        elif "Left" in zone:
            return f"Watch out! {label} close on your left!"
        else:
            return f"Watch out! {label} close on your right!"
    elif urgency == 2:
        if "Center" in zone:
            return f"{label} ahead in your path."
        elif "Left" in zone:
            return f"{label} on your left."
        else:
            return f"{label} on your right."
    else:
        return f"{label} detected in distance."


# =========================================================
# ---------------- VIDEO STREAM ---------------------------
# =========================================================

class VideoStream:
    """Threaded VideoStream for smooth, low-latency webcam capture."""
    def __init__(self, src=0, resolution=(640, 480)):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()
            if not self.grabbed:
                time.sleep(0.01)

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()


# =========================================================
# ---------------- MODEL RESOLVER -------------------------
# =========================================================

def resolve_model_path(user_path: str) -> str:
    candidates = [
        user_path,
        "weights/obstacle_yolo_best.onnx",
        "weights/obstacle_yolo_best.pt",
        "obstacle_detector/yolo_v8_obstacles/weights/best.pt",
        "yolov8n.pt"
    ]
    for c in candidates:
        if c and Path(c).exists():
            return str(Path(c).resolve())
    return "yolov8n.pt"


# =========================================================
# ---------------- ARGUMENTS ------------------------------
# =========================================================

parser = argparse.ArgumentParser(description="Smart AI Navigation Cane - YOLO & IoT Fusion")
parser.add_argument('--weights', default='weights/obstacle_yolo_best.onnx', help='Path to model weights (.onnx or .pt)')
parser.add_argument('--conf', type=float, default=0.35, help='Confidence threshold')
parser.add_argument('--cam', default='0', help='Camera index or stream URL')
parser.add_argument('--resolution', default='640x480', help='Camera resolution WxH')
parser.add_argument('--no-show', action='store_true', help='Run headless without GUI window')
args = parser.parse_args()

resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)


# =========================================================
# ---------------- LOAD MODEL & AUDIO ---------------------
# =========================================================

print("\n" + "=" * 65)
print(" 🦯 SMART AI NAVIGATION CANE - SYSTEM STARTING")
print("=" * 65)

# 1. Initialize Audio Engine
audio_announcer = AsyncAudioAnnouncer(cooldown=3.0)
audio_announcer.speak("Smart AI Cane active. Audio connected.", force=True)

# 2. Load YOLO Model
model_file = resolve_model_path(args.weights)
print(f"[+] Loading Detection Model: {model_file}")
model = YOLO(model_file)
device = "0" if torch.cuda.is_available() else "cpu"
print(f"[+] Hardware Accelerator   : {device} ({'GPU Acceleration' if device == '0' else 'Raspberry Pi CPU Mode'})")

# 3. Start Video Stream
cam_index = int(args.cam) if str(args.cam).isdigit() else str(args.cam)
print(f"[+] Initializing Camera     : {cam_index}")
videostream = VideoStream(src=cam_index, resolution=(imW, imH)).start()
time.sleep(1.0)

frame_count = 0
distance = -1
last_sos_time = 0.0
last_ultrasonic_speech = 0.0
fps_history = []

print("\n[✓] SYSTEM ONLINE & MONITORING!")
print("    Press [q] or [ESC] on screen to quit.\n")


# =========================================================
# ---------------- MAIN REAL-TIME LOOP --------------------
# =========================================================

try:
    while True:
        t0 = time.perf_counter()
        frame1 = videostream.read()
        if frame1 is None:
            time.sleep(0.01)
            continue

        frame = frame1.copy()
        frame_count += 1

        # -----------------------------------------------------
        # 1. Ultrasonic Sensor Reading (Every 4 frames)
        # -----------------------------------------------------
        if frame_count % 4 == 0:
            distance = get_distance()

        # -----------------------------------------------------
        # 2. SOS Emergency Button Check
        # -----------------------------------------------------
        if HAS_RPI_GPIO and GPIO.input(BUTTON) == 0:
            current_time = time.time()
            if current_time - last_sos_time > 15:
                print("\n🚨 SOS BUTTON PRESSED! Sending emergency alerts...")
                audio_announcer.speak("Emergency alert activated! Sending live location.", force=True)

                lat, lon = get_gps_location()
                if lat and lon:
                    maps_link = f"https://maps.google.com/?q={lat},{lon}"
                    message = f"🚨 EMERGENCY ALERT!\n\nUser pressed the SOS button.\n\n📍 Live Location:\n{maps_link}"
                else:
                    message = "🚨 EMERGENCY ALERT!\n\nUser pressed the SOS button.\n(GPS fix pending, location unavailable)"

                send_telegram_message(message)
                last_sos_time = current_time

        # -----------------------------------------------------
        # 3. Ultrasonic Close Obstacle Priority Check (< 30 cm)
        # -----------------------------------------------------
        if distance != -1 and distance < 30:
            if HAS_RPI_GPIO:
                GPIO.output(BUZZER, GPIO.HIGH)

            current_time = time.time()
            if (current_time - last_ultrasonic_speech) > 4.0:
                audio_announcer.speak(f"Stop! Obstacle very close at {int(distance)} centimeters!", force=True)
                last_ultrasonic_speech = current_time
        else:
            if HAS_RPI_GPIO:
                GPIO.output(BUZZER, GPIO.LOW)

        # -----------------------------------------------------
        # 4. YOLO Object & Obstacle Detection
        # -----------------------------------------------------
        results = model(
            frame,
            conf=args.conf,
            imgsz=640,
            device=device,
            verbose=False
        )[0]

        # -----------------------------------------------------
        # 5. Spatial Navigation Corridors & Priority Alerts
        # -----------------------------------------------------
        alerts = analyze_navigation_corridors(imW, imH, results.boxes, results.names)

        # Voice announcements for camera obstacles (if ultrasonic is not already alarming)
        if alerts and (distance == -1 or distance >= 30):
            top_hazard = alerts[0]
            if top_hazard["urgency"] >= 2:  # Warning or Critical
                voice_msg = generate_voice_alert(top_hazard)
                audio_announcer.speak(voice_msg)

        # -----------------------------------------------------
        # 6. FPS Calculation
        # -----------------------------------------------------
        dt = time.perf_counter() - t0
        fps = 1.0 / max(dt, 1e-4)
        fps_history.append(fps)
        if len(fps_history) > 15:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history)

        # -----------------------------------------------------
        # 7. Draw Visual Corridor HUD & Annotations
        # -----------------------------------------------------
        line1 = int(imW * 0.33)
        line2 = int(imW * 0.66)

        # Corridor guide lines
        cv2.line(frame, (line1, 0), (line1, imH), (0, 255, 255), 1)
        cv2.line(frame, (line2, 0), (line2, imH), (0, 255, 255), 1)

        # Top Status Banner
        cv2.rectangle(frame, (0, 0), (imW, 34), (20, 20, 20), -1)
        cv2.putText(frame, "LEFT", (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, "CENTER (DIRECT PATH)", (line1 + 10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)
        cv2.putText(frame, "RIGHT", (line2 + 15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        dist_str = f"Dist: {int(distance)}cm" if distance != -1 else "Dist: --"
        stat_tag = f"FPS: {avg_fps:.1f} | {dist_str}"
        cv2.putText(frame, stat_tag, (imW - 170, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 180), 1)

        # Draw detected bounding boxes
        for a in alerts:
            x1, y1, x2, y2 = a["bbox"]
            color = (0, 0, 255) if a["urgency"] == 3 else ((0, 165, 255) if a["urgency"] == 2 else (0, 220, 0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label_txt = f"{a['label'].upper()} [{a['zone']}] {a['conf']*100:.0f}%"
            (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label_txt, (x1 + 3, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # Bottom Alert Bar
        cv2.rectangle(frame, (0, imH - 35), (imW, imH), (15, 15, 15), -1)
        if distance != -1 and distance < 30:
            msg = f">> DANGER: OBJECT {int(distance)}cm CLOSE (BUZZER ACTIVE) <<"
            bcol = (0, 0, 255)
        elif alerts:
            top = alerts[0]
            msg = f">> {top['label'].upper()} in {top['zone']} - {top['proximity']} <<"
            bcol = (0, 0, 255) if top["urgency"] == 3 else ((0, 180, 255) if top["urgency"] == 2 else (100, 220, 100))
        else:
            msg = ">> PATH CLEAR - Safe to proceed <<"
            bcol = (0, 255, 100)

        cv2.putText(frame, msg, (15, imH - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, bcol, 2, cv2.LINE_AA)

        # -----------------------------------------------------
        # 8. Render Frame / Handle Headless
        # -----------------------------------------------------
        if not args.no_show:
            try:
                cv2.imshow('🦯 Smart AI Navigation Cane', frame)
                if cv2.waitKey(1) == ord('q'):
                    break
            except Exception:
                args.no_show = True

        if args.no_show:
            if alerts:
                top = alerts[0]
                print(f"\r[ALERT] {top['label'].upper()} in {top['zone']} ({top['proximity']}) | {dist_str} | FPS: {avg_fps:.1f}   ", end="", flush=True)
            time.sleep(0.01)

except KeyboardInterrupt:
    print("\n[*] Stopping Smart Cane System...")

# =========================================================
# ---------------- CLEANUP --------------------------------
# =========================================================

print("[+] Cleaning up resources...")
audio_announcer.speak("Smart cane system shutting down.", force=True)
time.sleep(0.8)
audio_announcer.stop()

videostream.stop()
cv2.destroyAllWindows()

if HAS_RPI_GPIO:
    GPIO.output(BUZZER, GPIO.LOW)
    GPIO.cleanup()

print("[✓] System shut down cleanly.")
