"""
================================================================================
DEDICATED OBSTACLE DETECTOR TEST SCRIPT
================================================================================
Targeted for: Visually Impaired Navigation Assistance & Robotics
Supports:
  - PyTorch (.pt) and ONNX (.onnx) models
  - Single Image, Folder of Images, Video File, or Live Webcam (source '0')
  - Spatial Corridor Partitioning: Left / Center / Right
  - Proximity Risk Warning (Critical Close / Warning Approaching / Info Distant)
  - PyCharm Run/Debug Configuration Friendly

Usage Examples:
  # 1. Test sample directory (Default)
  python test_obstacle_detector.py

  # 2. Test a specific image
  python test_obstacle_detector.py --source data/test_samples/synthetic_street_sample.jpg

  # 3. Test on Live Webcam
  python test_obstacle_detector.py --source 0 --assistive

  # 4. Test on a Video
  python test_obstacle_detector.py --source path/to/video.mp4 --conf 0.4
================================================================================
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO


# Standard Assistive Obstacle Classes
DEFAULT_CLASSES = [
    'car', 'bike', 'bus', 'chair', 'table',
    'pole', 'person', 'stairs', 'wall'
]

# Distinct BGR Colors for obstacle visualization
CLASS_COLORS = [
    (255, 100, 50),   # car - Blue-orange
    (50, 205, 50),    # bike - Lime
    (0, 140, 255),    # bus - Orange
    (255, 191, 0),    # chair - Deep sky blue
    (147, 20, 255),   # table - Pink/purple
    (0, 0, 255),      # pole - Red (High risk)
    (0, 255, 255),    # person - Yellow (Pedestrian)
    (0, 69, 255),     # stairs - Red-orange (Fall risk)
    (128, 128, 128)   # wall - Gray
]


def load_class_metadata(weights_dir: Path):
    """Loads class names from classes.json if available."""
    meta_path = weights_dir / "classes.json"
    if meta_path.exists():
        try:
            with open(meta_path, 'r') as f:
                data = json.load(f)
                return data.get("classes", DEFAULT_CLASSES)
        except Exception as e:
            print(f"[!] Warning reading classes.json: {e}")
    return DEFAULT_CLASSES


def resolve_model_path(weights_arg: str) -> str:
    """Finds best available weights, falling back gracefully to baseline."""
    user_path = Path(weights_arg)
    if user_path.exists():
        return str(user_path)

    default_trained = Path("weights/obstacle_yolo_best.pt")
    if default_trained.exists():
        return str(default_trained)

    default_onnx = Path("weights/obstacle_yolo_best.onnx")
    if default_onnx.exists():
        return str(default_onnx)

    run_checkpoint = Path("obstacle_detector/yolo_v8_obstacles/weights/best.pt")
    if run_checkpoint.exists():
        return str(run_checkpoint)

    # Fallback to standard pretrained YOLOv8 nano
    print(f"[!] Target weights '{weights_arg}' not found.")
    print("    Downloading/using 'yolov8n.pt' as baseline model for testing...")
    return "yolov8n.pt"


def evaluate_assistive_navigation(img_w: int, img_h: int, boxes, class_names_map):
    """
    Computes spatial navigation corridor (Left / Center / Right) and
    proximity estimation (Critical / Warning / Info).
    """
    left_boundary = img_w * 0.33
    right_boundary = img_w * 0.66
    alerts = []

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = class_names_map.get(cls_id, f"Class_{cls_id}")

        cx = (x1 + x2) / 2.0
        box_h = y2 - y1
        height_ratio = box_h / max(img_h, 1)

        # Corridor Zone
        if cx < left_boundary:
            zone = "LEFT"
            zone_color = (255, 200, 0)
        elif cx > right_boundary:
            zone = "RIGHT"
            zone_color = (255, 200, 0)
        else:
            zone = "CENTER (AHEAD)"
            zone_color = (0, 0, 255)

        # Proximity Risk
        if height_ratio > 0.45:
            proximity = "CRITICAL (<1.5m)"
            urgency = 3
        elif height_ratio > 0.25:
            proximity = "WARNING (2-3m)"
            urgency = 2
        else:
            proximity = "INFO (>3m)"
            urgency = 1

        alerts.append({
            "label": label,
            "conf": conf,
            "zone": zone,
            "zone_color": zone_color,
            "proximity": proximity,
            "urgency": urgency,
            "bbox": (int(x1), int(y1), int(x2), int(y2))
        })

    # Sort alerts so most critical / center hazards come first
    alerts.sort(key=lambda a: (-a["urgency"], 0 if "CENTER" in a["zone"] else 1))
    return alerts


def draw_assistive_hud(frame, alerts, fps_text: str = ""):
    """Draws navigation corridors, HUD banners, and warning messages."""
    h, w = frame.shape[:2]
    annotated = frame.copy()

    # Draw corridor boundary lines
    line1_x = int(w * 0.33)
    line2_x = int(w * 0.66)
    cv2.line(annotated, (line1_x, 0), (line1_x, h), (255, 255, 0), 1, cv2.LINE_AA)
    cv2.line(annotated, (line2_x, 0), (line2_x, h), (255, 255, 0), 1, cv2.LINE_AA)

    # Top HUD Bar
    cv2.rectangle(annotated, (0, 0), (w, 42), (20, 20, 20), -1)
    cv2.putText(annotated, "LEFT ZONE", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)
    cv2.putText(annotated, "CENTER PATH", (line1_x + 30, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
    cv2.putText(annotated, "RIGHT ZONE", (line2_x + 30, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)

    if fps_text:
        cv2.putText(annotated, fps_text, (w - 180, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 120), 2)

    # Render bounding boxes with custom labels
    for a in alerts:
        x1, y1, x2, y2 = a["bbox"]
        color = (0, 0, 255) if a["urgency"] == 3 else ((0, 165, 255) if a["urgency"] == 2 else (0, 220, 0))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        tag = f"{a['label']} {a['conf']*100:.0f}% [{a['zone']}]"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, max(th + 6, y1)), color, -1)
        cv2.putText(annotated, tag, (x1 + 3, max(y1 - 3, th + 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Bottom Alert Banner
    cv2.rectangle(annotated, (0, h - 45), (w, h), (15, 15, 15), -1)
    if alerts:
        top_alert = alerts[0]
        banner_color = (0, 0, 255) if top_alert["urgency"] == 3 else ((0, 180, 255) if top_alert["urgency"] == 2 else (200, 200, 200))
        banner_msg = f">> ALERT: {top_alert['label'].upper()} in {top_alert['zone']} - {top_alert['proximity']} <<"
    else:
        banner_color = (0, 255, 0)
        banner_msg = ">> PATH CLEAR - No immediate obstacle detected <<"

    cv2.putText(annotated, banner_msg, (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, banner_color, 2, cv2.LINE_AA)
    return annotated


def process_image(model, img_path: Path, args, save_dir: Path):
    """Performs inference and saves/displays result for a single image."""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[!] Error: Could not read image {img_path}")
        return

    h, w = img.shape[:2]
    t0 = time.perf_counter()
    results = model(img, conf=args.conf, imgsz=args.imgsz, device=args.device)[0]
    latency_ms = (time.perf_counter() - t0) * 1000

    alerts = evaluate_assistive_navigation(w, h, results.boxes, results.names)

    if args.assistive:
        annotated = draw_assistive_hud(img, alerts, f"Latency: {latency_ms:.1f}ms")
    else:
        annotated = results.plot()

    # Console logging
    print(f"\n[Image: {img_path.name}] Latency: {latency_ms:.1f} ms | Found {len(alerts)} obstacles:")
    for a in alerts:
        print(f"  • {a['label'].upper():<10} | Zone: {a['zone']:<15} | Risk: {a['proximity']:<22} (Conf: {a['conf']*100:.1f}%)")

    # Save output
    if args.save:
        save_file = save_dir / f"result_{img_path.name}"
        cv2.imwrite(str(save_file), annotated)
        print(f"  Saved result to: {save_file}")

    if not args.no_show:
        cv2.imshow(f"Obstacle Detection - {img_path.name}", annotated)
        print("  Press any key on image window to continue (or 'q' to quit)...")
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        if key == ord('q'):
            sys.exit(0)


def process_video_or_webcam(model, source_val, args, save_dir: Path):
    """Performs real-time streaming inference on video or webcam feed."""
    is_webcam = str(source_val).isdigit()
    src = int(source_val) if is_webcam else str(source_val)

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[!] Error: Cannot open video/camera source '{source_val}'")
        return

    fps_tracker = []
    writer = None
    save_file = save_dir / "result_video.mp4"

    print(f"\n[+] Streaming started from source: {source_val}")
    print("    Press 'q' in the display window to exit stream.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            t0 = time.perf_counter()
            results = model(frame, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)[0]
            dt = time.perf_counter() - t0
            fps = 1.0 / max(dt, 1e-4)

            fps_tracker.append(fps)
            if len(fps_tracker) > 30:
                fps_tracker.pop(0)
            avg_fps = sum(fps_tracker) / len(fps_tracker)

            alerts = evaluate_assistive_navigation(w, h, results.boxes, results.names)

            if args.assistive:
                annotated = draw_assistive_hud(frame, alerts, f"FPS: {avg_fps:.1f}")
            else:
                annotated = results.plot()

            if args.save:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(str(save_file), fourcc, 20.0, (w, h))
                writer.write(annotated)

            if not args.no_show:
                cv2.imshow("YOLO Obstacle Detector - Real-Time", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[*] Stream stopped by user.")
                    break
    finally:
        cap.release()
        if writer:
            writer.release()
            print(f"[✓] Saved output video: {save_file}")
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Run Obstacle Detection Inference with YOLOv8 (Assistive Navigation & Robotics)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/obstacle_yolo_best.pt",
        help="Path to trained model weights (.pt or .onnx)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/test_samples",
        help="Input path: image file, directory of images, video file, or '0' for webcam"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Detection confidence threshold"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image resolution"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0" if torch.cuda.is_available() else "cpu",
        help="Compute device: '0', 'cpu', etc."
    )
    parser.add_argument(
        "--assistive",
        action="store_true",
        default=True,
        help="Enable Assistive Corridor Navigation HUD (Left, Center, Right) and risk alarms"
    )
    parser.add_argument(
        "--no-assistive",
        dest="assistive",
        action="store_false",
        help="Disable assistive HUD and use standard YOLO bounding box overlay"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save annotated results to runs/test_results"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="runs/test_results",
        help="Directory to store test outputs"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display OpenCV GUI window (useful for headless / automated runs)"
    )

    args = parser.parse_args()

    # Prepare directories
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = resolve_model_path(args.weights)
    print("=================================================================")
    print(f" [✓] Loading Model : {model_path}")
    print(f" [✓] Device        : {args.device}")
    print(f" [✓] Confidence    : {args.conf}")
    print(f" [✓] Source        : {args.source}")
    print(f" [✓] Assistive HUD : {args.assistive}")
    print("=================================================================")

    model = YOLO(model_path)

    # Check input source type
    src_str = str(args.source)
    if src_str.isdigit() or src_str.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        process_video_or_webcam(model, src_str, args, save_dir)
    else:
        src_path = Path(src_str)
        if src_path.is_file():
            process_image(model, src_path, args, save_dir)
        elif src_path.is_dir():
            valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
            files = [p for p in src_path.iterdir() if p.suffix.lower() in valid_exts]
            if not files:
                print(f"[!] No image files found in directory: {src_path}")
                return
            print(f"[+] Found {len(files)} test image(s) in {src_path}")
            for f in files:
                process_image(model, f, args, save_dir)
        else:
            print(f"[!] Source path does not exist: {src_path}")


if __name__ == "__main__":
    main()
