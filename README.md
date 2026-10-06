# 🦯 YOLOv8 Obstacle Detection for Assistive Navigation & Robotics

A complete, production-ready computer vision project built with **Ultralytics YOLOv8**, designed for detecting obstacles (pedestrians, vehicles, poles, stairs, chairs, tables, and walls) to assist visually impaired individuals and edge robotics.

---

## 📁 Project Structure

```text
embedded_project/
├── .idea/                           # PyCharm project configuration (created automatically by PyCharm)
├── config/
│   └── data_custom.yaml             # Custom YOLO dataset configuration template
├── data/
│   ├── dataset/                     # Local dataset directory (train/val splits)
│   └── test_samples/                # Images or videos for quick inference testing
│       └── synthetic_street_sample.jpg
├── runs/
│   └── test_results/                # Annotated inference outputs and saved videos
├── weights/
│   ├── obstacle_yolo_best.pt        # Trained PyTorch model checkpoint
│   ├── obstacle_yolo_best.onnx      # Exported ONNX model for embedded/edge deployment
│   └── classes.json                 # Target classes and metadata
├── requirements.txt                 # Project dependencies
├── train_obstacle_detector.ipynb    # Complete training & model export notebook
├── test_obstacle_detector.ipynb     # Interactive testing notebook in PyCharm
├── test_obstacle_detector.py        # Standalone Python inference script (webcam, video, images)
├── live_camera_detection.py         # Dedicated live camera detection script (interactive HUD & voice)
└── README.md                        # Documentation & setup guide
```

---

## 🎯 Target Obstacle Classes

1. **`car`** – Moving or parked automobiles
2. **`bike`** – Bicycles and motorcycles
3. **`bus`** – Large transit vehicles
4. **`chair`** – Indoor and outdoor seating hazards
5. **`table`** – Tables and desks
6. **`pole`** – Utility poles, street signs, light posts (critical head/torso collision hazards)
7. **`person`** – Pedestrians in navigation path
8. **`stairs`** – Staircases (fall hazards)
9. **`wall`** – Boundary walls and barriers

---

## ⚙️ Setup & PyCharm Configuration

### 1. Open the Project in PyCharm
1. Launch **PyCharm**.
2. Click **Open** and select the directory: `f:\embedded_project`.

### 2. Configure Python Interpreter & Virtual Environment
1. In PyCharm, go to **Settings / Preferences** (`Ctrl + Alt + S`).
2. Navigate to **Project: embedded_project** -> **Python Interpreter**.
3. Select your installed **Python 3.11** interpreter (CUDA-enabled PyTorch is pre-verified).
4. Open the PyCharm Terminal and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 1. Training the Model (`train_obstacle_detector.ipynb`)

Open `train_obstacle_detector.ipynb` inside PyCharm's built-in notebook editor.

### Key Sections:
- **Environment & GPU Check**: Verifies CUDA availability.
- **Data Preparation**: 
  - **Option A**: Automatic download via Roboflow API using your free API key.
  - **Option B**: Local dataset using `config/data_custom.yaml`.
- **Data Augmentation**: Robust Albumentations pipeline + native YOLO Mosaic/MixUp.
- **Model Training**: Trains `yolov8m.pt` (or `yolov8n.pt` for ultra-low latency).
- **Evaluation**: Computes mAP50, mAP50-95, precision, recall, and renders the Confusion Matrix.
- **Model Export (Section 10)**:
  - Automatically copies `runs/.../weights/best.pt` into `weights/obstacle_yolo_best.pt`.
  - Exports to **ONNX** (`weights/obstacle_yolo_best.onnx`) with `opset=12` for edge hardware.
  - Generates `weights/classes.json` with class mappings for testing and microcontrollers.

---

## 🧪 2. Dedicated Testing Suite

You can test your model using either the **interactive notebook** or the **command-line Python script**.

### Option A: Interactive Testing Notebook (`test_obstacle_detector.ipynb`)
- Open `test_obstacle_detector.ipynb` in PyCharm.
- Run all cells to test sample images with the **Assistive Navigation Corridor Overlay**.
- Inspect latency metrics (ms and FPS) across 50 warmup and benchmark passes.

### Option B: Standalone Testing Script (`test_obstacle_detector.py`)
Run the testing script directly from the PyCharm Terminal or by right-clicking `test_obstacle_detector.py` -> **Run**:

#### 1. Test Single Image
```bash
python test_obstacle_detector.py --source data/test_samples/synthetic_street_sample.jpg
```

#### 2. Test Folder of Images
```bash
python test_obstacle_detector.py --source data/test_samples/
```

#### 3. Test on Live Webcam
```bash
python test_obstacle_detector.py --source 0 --assistive
```

#### 4. Test on a Video File
```bash
python test_obstacle_detector.py --source path/to/navigation_video.mp4 --conf 0.4
```

### Option C: Live Real-Time Camera Detection (`live_camera_detection.py`)
Run real-time obstacle detection directly from your webcam, USB camera, or ESP32-CAM stream with audio alerts and interactive HUD:

```bash
# Default Laptop/Built-in Webcam
python live_camera_detection.py

# External USB Camera (Index 1)
python live_camera_detection.py --cam 1

# ESP32-CAM or IP Camera Stream
python live_camera_detection.py --cam http://192.168.1.100:81/stream
```

**Interactive Controls During Live Camera Feed**:
- **`[q]` / `[ESC]`**: Exit stream
- **`[s]`**: Save snapshot to `runs/live_snapshots/`
- **`[c]`**: Toggle Navigation Corridor HUD (Left / Center / Right)
- **`[v]`**: Toggle Spoken Voice Alerts
- **`[+]` / `[-]`**: Adjust detection confidence threshold live on screen


#### 5. Headless Mode (Automated / No GUI Popup)
```bash
python test_obstacle_detector.py --source data/test_samples/ --no-show --save
```

---

## 🔊 Assistive Navigation Corridor Features

The testing suite includes a spatial safety analysis designed for assistive mobility:
- **Spatial Partitioning**: Divides the camera frame into 3 vertical corridors:
  - **LEFT ZONE**: Objects on the user's left.
  - **CENTER PATH**: Direct hazard in front of user (triggers red warning box & priority alert).
  - **RIGHT ZONE**: Objects on the user's right.
- **Proximity Estimation**:
  - `CRITICAL (< 1.5m)`: Large bounding box occupying over 45% of vertical frame height.
  - `WARNING (2-3m)`: Obstacle occupying 25–45% of vertical frame height.
  - `INFO (> 3m)`: Distant obstacle.
- **Simulated Speech/Audio Output**: Prints audio alert text (e.g. `ALERT: POLE in CENTER PATH - CRITICAL (<1.5m)`).

---

## ⚡ Edge & Microcontroller Deployment (ESP32 / Raspberry Pi)

1. **ONNX Export**: The exported `weights/obstacle_yolo_best.onnx` can be run using `onnxruntime` on Raspberry Pi 4/5 or Jetson Nano.
2. **ESP32 Companion Architecture**:
   - For an ESP32-CAM or ESP32-S3, stream camera frames over HTTP/WebSocket or serial to the host computer / companion board running `test_obstacle_detector.py`.
   - The host computes obstacle positions and sends concise command packets back to the ESP32 (e.g., `ALERT_CENTER_POLE`, `HAPTIC_LEFT_MOTOR_PULSE`).
