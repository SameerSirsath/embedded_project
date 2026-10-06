#!/bin/bash
# ==============================================================================
# Raspberry Pi 5 Automated Setup Script for YOLO Obstacle Detection
# ==============================================================================
set -e

echo "=== [1/4] Updating package lists and installing system dependencies ==="
sudo apt update
sudo apt install -y python3-pip python3-venv git espeak-ng libcamera-tools \
    libgl1-mesa-glx libglib2.0-0 v4l-utils

echo "=== [2/4] Setting up Python virtual environment ==="
if [ ! -d "venv" ]; then
    python3 -m venv venv --system-site-packages
    echo "[✓] Virtual environment created in ./venv"
fi

source venv/bin/activate

echo "=== [3/4] Installing Python packages ==="
pip install --upgrade pip
# Ultralytics and ONNX Runtime optimized for ARM64 CPU
pip install ultralytics onnxruntime opencv-python matplotlib pyyaml pyttsx3

echo "=== [4/4] Verification check ==="
python -c "
import cv2, onnxruntime, ultralytics
print('[✓] OpenCV version:', cv2.__version__)
print('[✓] ONNX Runtime version:', onnxruntime.__version__)
print('[✓] Ultralytics version:', ultralytics.__version__)
print('[✓] Environment is ready for Raspberry Pi 5 real-time detection!')
"

echo "=============================================================================="
echo "Setup complete! To run live detection on Raspberry Pi 5:"
echo "  1. Activate virtual environment:"
echo "       source venv/bin/activate"
echo ""
echo "  2. For USB Webcam:"
echo "       python live_camera_detection.py --cam 0 --weights weights/obstacle_yolo_best.onnx"
echo ""
echo "  3. For Official Raspberry Pi Camera Module (CSI ribbon):"
echo "       libcamerify python live_camera_detection.py --cam 0 --weights weights/obstacle_yolo_best.onnx"
echo "=============================================================================="
