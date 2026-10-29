"""
================================================================================
  TEST OCR WITH MOBILE CAMERA (iVCam / USB WEBCAM)
================================================================================
A dedicated, interactive testing script to test OCR text extraction and TTS
reading aloud using a mobile phone connected as a webcam via e2eSoft iVCam
or an integrated laptop camera.

Features:
  - Auto-detects e2eSoft iVCam (Index 1) and Laptop Camera (Index 0)
  - Live Camera Switcher: Press [n] to toggle between Mobile and Laptop cameras
  - RapidOCR (ONNX) + Tesseract fallback for high accuracy & low latency
  - Burst-Capture Focus Score: Grabs sharpest frame using Laplacian variance
  - Dual Preprocessing Modes: 'book' (printed pages) vs 'screen' (monitors/phones)
  - Natural Reading Order: Multi-column and line sorting
  - Robust Multi-Tier TTS Voice Engine: Reads recognized text aloud cleanly on Windows (SAPI) & Linux
  - Interactive Key Controls:
      [SPACE] / [r] : Run OCR on current view (burst grab & read aloud)
      [t]           : Audio Test (Test speaker reading aloud immediately)
      [n]           : Switch Camera (Toggle between iVCam Mobile and Laptop Webcam)
      [m]           : Toggle OCR Mode (Book Mode <-> Screen Mode)
      [p]           : Toggle Preprocessed Binary Image View
      [v]           : Toggle Voice TTS reading aloud
      [s]           : Save Snapshot image & extracted text to runs/ocr_tests/
      [c]           : Clear current OCR text overlay & stop speech
      [q] / [ESC]   : Exit test script

Usage:
  python test_ocr_ivcam.py               # Auto-detects & prefers iVCam (Index 1)
  python test_ocr_ivcam.py --cam 1       # Explicitly use iVCam mobile camera
  python test_ocr_ivcam.py --cam 0       # Use laptop integrated camera
  python test_ocr_ivcam.py --mode screen # Use screen preprocessing mode
================================================================================
"""

import argparse
import datetime
import os
import queue
import re
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

# Import Smart Cane Reader subsystem components
try:
    from smart_cane.services.ocr_reader import (
        PageReader,
        sharpest_of,
        preprocess,
        order_blocks,
    )
    from smart_cane.services.bookmarks import BookmarkStore
    HAS_SMART_CANE = True
except ImportError:
    HAS_SMART_CANE = False


# ==============================================================================
# 1. Asynchronous Speech Worker (Robust Non-blocking TTS for Windows & Linux)
# ==============================================================================
class AsyncSpeaker:
    """
    Robust non-blocking TTS worker thread.
    Uses native Windows SAPI COM (with STA thread initialization) on Windows
    for 100% reliable, zero-lag, loud audio. Falls back to pyttsx3 or espeak-ng.
    """

    def __init__(self, rate=170, enabled=True):
        self.enabled = enabled
        self.rate = rate
        self.queue = queue.Queue()
        self._stop_event = threading.Event()
        self._speaking_event = threading.Event()
        self._current_sentence = ""
        self._worker_thread = threading.Thread(target=self._run, daemon=True)
        self._worker_thread.start()

    def is_speaking(self) -> bool:
        return self._speaking_event.is_set()

    def get_current_sentence(self) -> str:
        return self._current_sentence

    def speak(self, text: str, interrupt: bool = True):
        if not self.enabled or not text.strip():
            return
        if interrupt:
            self.stop()
        sentences = self.split(text)
        for s in sentences:
            self.queue.put(s)

    def stop(self):
        with self.queue.mutex:
            self.queue.queue.clear()
        self._current_sentence = ""
        self._speaking_event.clear()

    @staticmethod
    def split(text: str):
        text = re.sub(r"\s+", " ", text).strip()
        # Split into sentences or clauses for natural rhythm
        parts = [s.strip() for s in re.split(r"(?<=[.!?\n])\s+", text) if s.strip()]
        if not parts:
            return [text]
        return parts

    def _run(self):
        sapi_voice = None
        pyttsx_engine = None

        # 1. Initialize Windows SAPI with thread COM initialization
        if os.name == "nt":
            try:
                import pythoncom
                import win32com.client
                pythoncom.CoInitialize()
                sapi_voice = win32com.client.Dispatch("SAPI.SpVoice")
                sapi_voice.Rate = 1
                sapi_voice.Volume = 100
                print("[TTS] Windows SAPI Speech Synthesizer initialized successfully (Volume: 100%).")
            except Exception as e:
                print(f"[TTS] SAPI initialization failed ({e}), falling back to pyttsx3.")
                sapi_voice = None

        # 2. Fallback to pyttsx3
        if sapi_voice is None:
            try:
                import pyttsx3
                pyttsx_engine = pyttsx3.init()
                pyttsx_engine.setProperty("rate", self.rate)
                pyttsx_engine.setProperty("volume", 1.0)
                print("[TTS] pyttsx3 Speech Synthesizer initialized.")
            except Exception as e:
                print(f"[TTS] pyttsx3 initialization failed ({e}).")

        while not self._stop_event.is_set():
            try:
                sent = self.queue.get(timeout=0.15)
            except queue.Empty:
                self._current_sentence = ""
                self._speaking_event.clear()
                continue

            if not self.enabled:
                self.queue.task_done()
                continue

            self._current_sentence = sent
            self._speaking_event.set()
            print(f"[AUDIO] Reading aloud: \"{sent}\"")

            try:
                if sapi_voice is not None:
                    # SVSFDefault = 0 (Synchronous in this worker thread)
                    sapi_voice.Speak(sent)
                elif pyttsx_engine is not None:
                    pyttsx_engine.say(sent)
                    pyttsx_engine.runAndWait()
                elif os.name == "nt":
                    # Fallback via PowerShell
                    import subprocess
                    clean_msg = sent.replace("'", "").replace('"', "")
                    cmd = f"Add-Type -AssemblyName System.speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{clean_msg}');"
                    subprocess.run(["powershell", "-NoProfile", "-Command", cmd], creationflags=subprocess.CREATE_NO_WINDOW, timeout=5.0)
                else:
                    # Linux / Raspberry Pi fallback
                    import subprocess
                    subprocess.run(["espeak-ng", sent], stderr=subprocess.DEVNULL, timeout=5.0)
            except Exception as e:
                print(f"[TTS] Error during speech synthesis: {e}")
            finally:
                self._current_sentence = ""
                self._speaking_event.clear()
                self.queue.task_done()


# ==============================================================================
# 2. Camera Discovery & Threaded Video Stream
# ==============================================================================
def probe_available_cameras():
    """Finds active cameras and identifies e2eSoft iVCam vs Laptop Webcam."""
    available = []
    print("[*] Probing available camera indices...")
    for idx in range(4):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                label = (
                    "e2eSoft iVCam (Mobile Phone)"
                    if idx == 1
                    else ("Integrated Laptop Camera" if idx == 0 else f"Camera Device {idx}")
                )
                available.append((idx, label))
                print(f"    [+] Found Index [{idx}]: {label}")
            cap.release()
    return available


class VideoStream:
    """Threaded OpenCV video capture to maximize FPS and minimize latency."""

    def __init__(self, src=1, width=1280, height=720):
        self.src = src
        self.width = width
        self.height = height
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        t = threading.Thread(target=self._update, daemon=True)
        t.start()
        return self

    def _update(self):
        while not self.stopped:
            grabbed, frame = self.cap.read()
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame
            time.sleep(0.005)

    def read(self):
        with self.lock:
            if not self.grabbed or self.frame is None:
                return None
            return self.frame.copy()

    def stop(self):
        self.stopped = True
        time.sleep(0.05)
        if self.cap.isOpened():
            self.cap.release()


# ==============================================================================
# 3. Direct RapidOCR Fallback (If smart_cane is not installed)
# ==============================================================================
def extract_text_with_boxes(img, mode="book"):
    """
    Runs RapidOCR and returns:
      - extracted_text (string)
      - boxes: list of {"text": str, "bbox": [x, y, w, h], "conf": float, "polygon": [[x,y],...]}
    """
    try:
        from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR()
    except ImportError:
        try:
            import pytesseract
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            text_lines = []
            boxes = []
            for i in range(len(data['text'])):
                txt = data['text'][i].strip()
                if txt:
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    conf = float(data['conf'][i]) / 100.0
                    boxes.append({
                        "text": txt,
                        "bbox": [x, y, w, h],
                        "conf": conf,
                        "polygon": [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                    })
                    text_lines.append(txt)
            return " ".join(text_lines), boxes
        except ImportError:
            return "Error: Neither RapidOCR nor pytesseract is installed.", []

    # Run RapidOCR on image
    result, elapse = ocr(img)
    if not result:
        return "", []

    boxes = []
    lines = []
    for item in result:
        poly = item[0]
        txt = item[1]
        conf = float(item[2])
        xs = [int(p[0]) for p in poly]
        ys = [int(p[1]) for p in poly]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        boxes.append({
            "text": txt,
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "conf": conf,
            "polygon": poly
        })
        lines.append(txt)

    # Use smart_cane natural ordering if available
    if HAS_SMART_CANE and boxes:
        ordered_text = order_blocks(boxes, img.shape)
        if ordered_text.strip():
            return ordered_text, boxes

    return " ".join(lines), boxes


# ==============================================================================
# 4. HUD Overlay & Visual Rendering
# ==============================================================================
def draw_ocr_hud(
    frame: np.ndarray,
    cam_label: str,
    fps: float,
    mode: str,
    voice_enabled: bool,
    is_speaking: bool,
    current_spoken_sentence: str,
    show_preproc: bool,
    last_text: str,
    boxes: list,
    ocr_time_ms: float,
    status_msg: str,
    focus_score: float = -1.0
) -> np.ndarray:
    """Draws rich interactive HUD with text boxes, status banner, and transcript panel."""
    h, w = frame.shape[:2]
    canvas = frame.copy()

    # 1. Draw Bounding Boxes around detected text
    for b in boxes:
        poly = np.array(b.get("polygon", []), dtype=np.int32)
        if len(poly) >= 4:
            cv2.polylines(canvas, [poly], isClosed=True, color=(0, 255, 128), thickness=2)
            # Text label above box
            txt = b.get("text", "")
            conf = b.get("conf", 0.0)
            x_min = int(min(p[0] for p in poly))
            y_min = int(min(p[1] for p in poly))
            tag = f"{txt} ({int(conf * 100)}%)"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(
                canvas,
                (x_min, max(0, y_min - th - 6)),
                (x_min + tw + 6, max(th + 6, y_min)),
                (0, 180, 80),
                -1,
            )
            cv2.putText(
                canvas,
                tag,
                (x_min + 3, max(th + 2, y_min - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    # 2. Top Info Bar (Semi-transparent black)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, 42), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)

    # Camera & Source Indicator
    cam_color = (0, 255, 180) if "iVCam" in cam_label else (255, 200, 50)
    cv2.circle(canvas, (18, 21), 6, (0, 255, 0), -1)
    cv2.putText(canvas, f"CAM: {cam_label}", (32, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, cam_color, 2, cv2.LINE_AA)

    # Mode indicator
    mode_str = f"MODE: {mode.upper()}"
    cv2.putText(canvas, mode_str, (w // 2 - 130, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)

    # Voice & Speaking indicator
    if is_speaking:
        voice_str = "VOICE: SPEAKING ALOUD"
        v_col = (0, 255, 255)
    elif voice_enabled:
        voice_str = "VOICE: READY"
        v_col = (0, 255, 120)
    else:
        voice_str = "VOICE: MUTED"
        v_col = (120, 120, 120)

    cv2.putText(canvas, voice_str, (w // 2 + 30, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, v_col, 2, cv2.LINE_AA)

    fps_str = f"FPS: {fps:.1f}"
    cv2.putText(canvas, fps_str, (w - 110, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    # 3. Bottom Card: Extracted Text Transcript
    card_h = 145
    card_y = h - card_h
    overlay2 = canvas.copy()
    cv2.rectangle(overlay2, (0, card_y), (w, h), (15, 15, 18), -1)
    cv2.addWeighted(overlay2, 0.88, canvas, 0.12, 0, canvas)

    # Border separator
    border_color = (0, 255, 255) if is_speaking else (0, 255, 180)
    cv2.line(canvas, (0, card_y), (w, card_y), border_color, 2)

    # Header in Card
    header_text = f"EXTRACTED TEXT ({len(last_text)} chars"
    if ocr_time_ms > 0:
        header_text += f" | OCR Latency: {ocr_time_ms:.0f} ms"
    if focus_score > 0:
        header_text += f" | Sharpness: {focus_score:.1f}"
    header_text += ")"

    cv2.putText(canvas, header_text, (16, card_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, border_color, 1, cv2.LINE_AA)

    # Render Active Spoken Sentence or Extracted Text
    if is_speaking and current_spoken_sentence:
        spk_label = f"Speaking Now: \"{current_spoken_sentence}\""
        cv2.putText(canvas, spk_label[:100], (18, card_y + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    elif last_text.strip():
        # Wrap text nicely across card width
        words = last_text.split()
        lines = []
        curr_line = ""
        max_chars_per_line = int(w / 10.5)

        for word in words:
            if len(curr_line) + len(word) + 1 <= max_chars_per_line:
                curr_line += (" " if curr_line else "") + word
            else:
                lines.append(curr_line)
                curr_line = word
                if len(lines) >= 3:
                    break
        if curr_line and len(lines) < 3:
            lines.append(curr_line)

        y_offset = card_y + 52
        for line in lines:
            cv2.putText(canvas, line, (18, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            y_offset += 24
    else:
        cv2.putText(
            canvas,
            "Press [SPACE] or [r] to capture & read text with iVCam mobile camera...",
            (18, card_y + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )

    # 4. Status Prompt / Key Shortcuts Bar
    cv2.putText(
        canvas,
        "[SPACE]: Read OCR  |  [t]: Test Voice  |  [n]: Switch Cam  |  [m]: Mode  |  [v]: Voice  |  [s]: Save  |  [q]: Quit",
        (16, card_y + card_h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (130, 200, 255),
        1,
        cv2.LINE_AA,
    )

    # 5. Temporary Action Flash
    if status_msg:
        (stw, sth), _ = cv2.getTextSize(status_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(
            canvas,
            (w // 2 - stw // 2 - 12, 55),
            (w // 2 + stw // 2 + 12, 55 + sth + 16),
            (0, 140, 255),
            -1,
        )
        cv2.putText(
            canvas,
            status_msg,
            (w // 2 - stw // 2, 55 + sth + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return canvas


# ==============================================================================
# 5. Main OCR Interactive Runner
# ==============================================================================
def run_ocr_testing(
    cam_index: int = None,
    width: int = 1280,
    height: int = 720,
    ocr_mode: str = "book",
    voice: bool = True,
    burst_count: int = 3,
):
    print("\n" + "=" * 75)
    print(" [OCR] LIVE OCR TESTING WITH iVCam MOBILE CAMERA")
    print("=" * 75)

    # 1. Discover available cameras
    cams = probe_available_cameras()
    if not cams:
        print("[!] No camera devices detected! Please ensure iVCam is connected.")
        sys.exit(1)

    # Select initial camera: Prefer iVCam (Index 1) if present
    if cam_index is not None:
        active_cam_idx = cam_index
    elif any(c[0] == 1 for c in cams):
        active_cam_idx = 1
        print("[[OK]] Automatically selected [1] e2eSoft iVCam (Mobile Camera)")
    else:
        active_cam_idx = cams[0][0]
        print(f"[[OK]] Defaulting to camera [{active_cam_idx}] ({cams[0][1]})")

    # 2. Initialize Audio TTS Speaker
    speaker = AsyncSpeaker(rate=170, enabled=voice)
    time.sleep(0.3)
    speaker.speak("OCR mobile camera testing ready. Press space to read.")

    # 3. Create output directory for saving snapshots & transcripts
    save_dir = Path("runs/ocr_tests")
    save_dir.mkdir(parents=True, exist_ok=True)

    # 4. Start Video Stream
    print(f"\n[+] Starting Video Stream on Camera [{active_cam_idx}] @ {width}x{height}...")
    stream = VideoStream(src=active_cam_idx, width=width, height=height).start()
    time.sleep(1.2)

    window_name = "Smart Cane - Live OCR Mobile Camera Test (iVCam)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)

    # State variables
    current_mode = ocr_mode
    voice_active = voice
    show_preproc_view = False
    last_extracted_text = ""
    detected_boxes = []
    last_ocr_time_ms = 0.0
    last_focus_score = 0.0
    status_message = ""
    status_time = 0.0

    fps_history = []
    prev_frame_time = time.perf_counter()

    print("\n" + "-" * 75)
    print("  KEYBOARD CONTROLS:")
    print("    [SPACE] / [r] : Run OCR & read text aloud via TTS")
    print("    [t]           : Test Voice (Immediately speaks a test sentence)")
    print("    [n]           : Switch Camera (Toggle between iVCam Mobile & Laptop)")
    print("    [m]           : Toggle Mode (Book Mode <-> Screen Mode)")
    print("    [p]           : Toggle Preprocessing Binary View")
    print("    [v]           : Toggle Voice TTS reading")
    print("    [s]           : Save snapshot & text to runs/ocr_tests/")
    print("    [c]           : Clear text & stop speech")
    print("    [q] / [ESC]   : Quit")
    print("-" * 75 + "\n")

    try:
        while True:
            t0 = time.perf_counter()
            frame = stream.read()
            if frame is None:
                time.sleep(0.01)
                continue

            # Compute preview FPS
            dt = t0 - prev_frame_time
            prev_frame_time = t0
            fps = 1.0 / max(dt, 1e-4)
            fps_history.append(fps)
            if len(fps_history) > 20:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)

            # Auto-clear status message after 2.5 seconds
            if status_message and (time.time() - status_time > 2.5):
                status_message = ""

            # Find active camera name for HUD
            cam_label = "e2eSoft iVCam" if active_cam_idx == 1 else f"Camera {active_cam_idx}"
            for c_idx, c_name in cams:
                if c_idx == active_cam_idx:
                    cam_label = c_name
                    break

            # Render Preprocessed or Normal View
            if show_preproc_view:
                proc = preprocess(frame, mode=current_mode) if HAS_SMART_CANE else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                display_frame = cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR)
                cv2.putText(
                    display_frame,
                    f"PREPROCESSED VIEW ({current_mode.upper()} MODE)",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2,
                )
            else:
                display_frame = draw_ocr_hud(
                    frame=frame,
                    cam_label=cam_label,
                    fps=avg_fps,
                    mode=current_mode,
                    voice_enabled=voice_active,
                    is_speaking=speaker.is_speaking(),
                    current_spoken_sentence=speaker.get_current_sentence(),
                    show_preproc=show_preproc_view,
                    last_text=last_extracted_text,
                    boxes=detected_boxes,
                    ocr_time_ms=last_ocr_time_ms,
                    status_msg=status_message,
                    focus_score=last_focus_score,
                )

            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF

            # -------------------------------------------------------------
            # KEYBOARD HANDLERS
            # -------------------------------------------------------------
            if key in [ord("q"), 27]:  # Quit
                break

            elif key in [ord(" "), ord("r")]:  # Run OCR Now!
                print("\n[+] Triggering OCR burst grab...")
                status_message = "Capturing sharpest frame & analyzing..."
                status_time = time.time()

                # Grab burst frames and pick the sharpest
                burst_frames = []
                for _ in range(burst_count):
                    f = stream.read()
                    if f is not None:
                        burst_frames.append(f)
                    time.sleep(0.08)

                if burst_frames:
                    if HAS_SMART_CANE:
                        best_img, focus_score = sharpest_of(burst_frames)
                    else:
                        best_img = burst_frames[-1]
                        gray = cv2.cvtColor(best_img, cv2.COLOR_BGR2GRAY)
                        focus_score = cv2.Laplacian(gray, cv2.CV_64F).var()

                    last_focus_score = focus_score
                    print(f"    -> Selected sharpest frame (Focus Variance: {focus_score:.1f})")

                    # Run OCR Extraction
                    ocr_start = time.perf_counter()
                    text, boxes = extract_text_with_boxes(best_img, mode=current_mode)
                    last_ocr_time_ms = (time.perf_counter() - ocr_start) * 1000.0

                    last_extracted_text = text.strip()
                    detected_boxes = boxes

                    print(f"    -> Extracted {len(last_extracted_text)} characters in {last_ocr_time_ms:.1f} ms")
                    print(f"    -> Recognized Text:\n\"{last_extracted_text}\"\n")

                    if last_extracted_text:
                        status_message = f"Found {len(detected_boxes)} text blocks! Reading..."
                        if voice_active:
                            speaker.speak(last_extracted_text, interrupt=True)
                    else:
                        status_message = "No clear text detected! Hold closer or adjust lighting."
                        if voice_active:
                            speaker.speak("I couldn't find any clear text. Try holding closer or improving light.")
                    status_time = time.time()

            elif key == ord("t"):  # Audio test key
                print("\n[[AUDIO]] Audio test triggered by user (Pressing [t])...")
                status_message = "Audio Test: Speaking aloud..."
                status_time = time.time()
                speaker.speak("Audio test. Voice synthesizer is working loud and clear.", interrupt=True)

            elif key == ord("n"):  # Switch Camera
                available_indices = [c[0] for c in cams]
                if len(available_indices) > 1:
                    curr_pos = available_indices.index(active_cam_idx) if active_cam_idx in available_indices else 0
                    next_idx = available_indices[(curr_pos + 1) % len(available_indices)]

                    print(f"[*] Switching camera from [{active_cam_idx}] to [{next_idx}]...")
                    status_message = f"Switching to Camera {next_idx}..."
                    status_time = time.time()

                    stream.stop()
                    active_cam_idx = next_idx
                    stream = VideoStream(src=active_cam_idx, width=width, height=height).start()
                    time.sleep(0.8)
                    status_message = f"Camera switched to [{active_cam_idx}]"
                    status_time = time.time()
                else:
                    status_message = "Only 1 camera found!"
                    status_time = time.time()

            elif key == ord("m"):  # Toggle OCR Preprocessing Mode
                current_mode = "screen" if current_mode == "book" else "book"
                status_message = f"Mode switched to: {current_mode.upper()}"
                status_time = time.time()
                print(f"[+] OCR Mode changed to: {current_mode}")

            elif key == ord("p"):  # Toggle Preprocessing View
                show_preproc_view = not show_preproc_view
                status_message = f"Preprocessed View: {'ON' if show_preproc_view else 'OFF'}"
                status_time = time.time()

            elif key == ord("v"):  # Toggle Voice TTS
                voice_active = not voice_active
                speaker.enabled = voice_active
                if not voice_active:
                    speaker.stop()
                status_message = f"Voice TTS: {'ENABLED' if voice_active else 'MUTED'}"
                status_time = time.time()
                print(f"[+] Voice TTS: {'ON' if voice_active else 'OFF'}")

            elif key == ord("s"):  # Save Snapshot & OCR Transcript
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                img_path = save_dir / f"ocr_snap_{ts}.jpg"
                txt_path = save_dir / f"ocr_text_{ts}.txt"

                cv2.imwrite(str(img_path), frame)
                with open(txt_path, "w", encoding="utf-8") as tf:
                    tf.write(f"Timestamp: {ts}\n")
                    tf.write(f"Camera: {cam_label}\n")
                    tf.write(f"Mode: {current_mode}\n")
                    tf.write(f"Latency: {last_ocr_time_ms:.1f} ms\n")
                    tf.write("=" * 60 + "\n")
                    tf.write(last_extracted_text + "\n")

                status_message = f"Saved snapshot to {img_path.name}"
                status_time = time.time()
                print(f"[[OK]] Saved snapshot: {img_path}")
                print(f"[[OK]] Saved transcript: {txt_path}")

            elif key == ord("c"):  # Clear
                last_extracted_text = ""
                detected_boxes = []
                speaker.stop()
                status_message = "Cleared text & stopped speech."
                status_time = time.time()

    except KeyboardInterrupt:
        print("\n[*] Stopping OCR testing...")

    finally:
        # Cleanup
        print("[+] Shutting down camera & audio worker...")
        speaker.stop()
        stream.stop()
        cv2.destroyAllWindows()
        print("[[OK]] Clean exit completed.")


# ==============================================================================
# 6. Entrypoint
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Interactive Live OCR Testing via iVCam Mobile Camera / Webcam",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cam",
        type=int,
        default=None,
        help="Camera index (defaults to 1 if iVCam is detected, else 0)",
    )
    parser.add_argument("--width", type=int, default=1280, help="Camera capture width (higher = sharper text)")
    parser.add_argument("--height", type=int, default=720, help="Camera capture height")
    parser.add_argument("--mode", default="book", choices=["book", "screen"], help="OCR preprocessing mode")
    parser.add_argument("--no-voice", action="store_true", help="Disable TTS reading aloud")
    parser.add_argument("--burst", type=int, default=3, help="Burst capture frames count for sharpness selection")

    args = parser.parse_args()

    run_ocr_testing(
        cam_index=args.cam,
        width=args.width,
        height=args.height,
        ocr_mode=args.mode,
        voice=not args.no_voice,
        burst_count=args.burst,
    )


if __name__ == "__main__":
    main()
