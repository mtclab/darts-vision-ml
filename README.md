# DartsVision ML Training

Training pipelines for darts detection models. Uses the DeepDarts/bhabha-kapil dataset (16k images, MIT) to train multiple YOLO11 variants for the DartsVision Android app.

## Approaches

| # | Model | What It Detects | Precision | Size | Use Case |
|---|-------|---------------|-----------|------|----------|
| 1 | **YOLO11n-pose** | Board + 7 keypoints (4 cal + 3 dart tips) | **±2°** | ~6 MB | Primary: dart + board detection with keypoint accuracy |
| 2 | **YOLO11n-detect** | Board bbox + dart tip bbox | ±5° | ~4 MB | Simpler: bbox-only, less precise dart tip |
| 3 | **Board calibration** | Board + 4 calibration keypoints | ±1° | ~5 MB | Replace CV BoardDetector with ML |
| 4 | **Frame-diff + CV** | No ML needed | ±5° | 0 MB | Current: diff empty board vs board with darts |

### How they combine in DartsVision

```
Camera Frame
    │
    ├─[Board Detection]────────────────────────────┐
    │  Option A: CV BoardDetector (current)         │
    │  Option B: ML board calibration model (#3)    │
    │                                               │
    ├─[Dart Detection]─────────────────────────────┤
    │  Priority 1: Frame-diff (reference captured)  │
    │  Priority 2: YOLO11n-pose (#1) if model avail │
    │  Priority 3: YOLO11n-detect (#2) fallback     │
    │  Priority 4: CV contour detection (last resort)│
    │                                               │
    └─► ScoreValidator ► UI overlay                  │
```

## Quick Start (GPU Server with Docker)

```bash
# 1. Clone
git clone <repo-url>
cd darts-vision-ml

# 2. Build Docker image (includes ultralytics + CUDA)
docker compose build

# 3. Download dataset + convert formats (runs on CPU, takes ~30 min)
docker compose run train python scripts/download_and_convert.py

# 4. Train models (needs GPU)

# Primary: YOLO11n-pose (dart tips + calibration keypoints)
docker compose run train python scripts/train_pose.py --epochs 100 --batch 16

# Alternative: YOLO11n-detect (bounding boxes only)
docker compose run train python scripts/train_detect.py --epochs 100 --batch 16

# Board calibration keypoints (replaces CV BoardDetector)
docker compose run train python scripts/train_calibration.py --epochs 100 --batch 16

# 5. Export to TFLite for Android
docker compose run train python scripts/export_tflite.py --all

# 6. Copy TFLite models to Android app
cp models/*.tflite ../darts_vision/app/src/main/assets/models/
```

## Without Docker (Local Python)

```bash
# Install dependencies
pip install -r requirements.txt

# Download + convert
python scripts/download_and_convert.py

# Train (requires GPU for reasonable speed)
python scripts/train_pose.py --epochs 100 --batch 16 --gpu 0

# Export
python scripts/export_tflite.py --all
```

## Dataset

**Source:** [bhabha-kapil/Dartboard-Detection-Dataset](https://huggingface.co/datasets/bhabha-kapil/Dartboard-Detection-Dataset)

| Property | Value |
|----------|-------|
| Images | 16,050 (800×800px) |
| Size | 7.21 GB |
| License | MIT |
| Annotations | Class 0: dart tip, Class 1-4: calibration corners |
| Split | 80/10/10 train/val/test (auto-generated) |

### Format Conversions

The `download_and_convert.py` script converts the source format into:

1. **yolo11_pose/** — One `dartboard` class per image, 7 keypoints (4 cal + 3 dart tips)
2. **yolo11_detect/** — `dartboard` + `dart_tip` bounding box classes
3. **board_calibration/** — One `dartboard` class, 4 calibration keypoints only

## Training Details

### YOLO11n-pose (Recommended)

- **Architecture:** YOLO11 nano with pose head
- **Input:** 640×640
- **Keypoints:** 7 per board (4 calibration + 3 dart tips)
- **Augmentation:** HSV jitter, rotation (±15°), flip, mosaic, mixup, copy-paste, erasing
- **Expected:** ~50 mAP on dart tips, 30-80ms inference on mobile
- **Output:** `runs/pose/yolo11n_darts_pose/weights/best.pt`

### YOLO11n-detect (Simpler)

- **Architecture:** YOLO11 nano detection
- **Classes:** dartboard (bbox), dart_tip (small bbox)
- **No keypoints** — dart position estimated from bbox center (±5px error)
- **Good for:** Board detection when you don't need precise dart tip

### Board Calibration (Replace CV)

- **Architecture:** YOLO11 nano pose (4 keypoints only)
- **Keypoints:** double-20, double-6, double-3, double-11 corners
- **Purpose:** Replace OpenCV BoardDetector with ML-based detection
- **Benefit:** More robust to weird angles, lighting, board styles

## Export for Android

```bash
# Export all models
python scripts/export_tflite.py --all

# Export specific model with INT8 quantization (smaller)
python scripts/export_tflite.py --model runs/pose/yolo11n_darts_pose/weights/best.pt --int8

# Export + copy directly to Android assets
python scripts/export_tflite.py --all --copy-android
```

### TFLite Model Sizes (Expected)

| Model | FP32 | INT8 |
|-------|------|------|
| YOLO11n-pose | ~6 MB | ~2 MB |
| YOLO11n-detect | ~4 MB | ~1.5 MB |
| Board calibration | ~5 MB | ~1.5 MB |

## Directory Structure

```
darts-vision-ml/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── configs/
│   ├── dataset_pose.yaml
│   ├── dataset_detect.yaml
│   └── dataset_calibration.yaml
├── scripts/
│   ├── download_and_convert.py   # Dataset download + format conversion
│   ├── train_pose.py             # YOLO11n-pose training
│   ├── train_detect.py           # YOLO11n-detect training
│   ├── train_calibration.py      # Board calibration training
│   └── export_tflite.py          # Export to TFLite for Android
├── data/
│   ├── raw/                      # Downloaded dataset (not in git)
│   └── processed/                # Converted datasets (not in git)
├── runs/                         # Training runs (not in git)
├── models/                       # Exported TFLite models
└── docs/
    └── approaches.md
```

## Integration with DartsVision Android App

Trained TFLite models integrate via `DartDetector.kt`:

1. Place `.tflite` model in `app/src/main/assets/models/`
2. `DartDetector.tryLoadYoloModel()` auto-detects the model
3. Detection priority: frame-diff → YOLO → CV fallback

See `../darts_vision/docs/yolo-dart-detection.md` for full integration details.