# DartsVision ML Training

Training pipelines for darts detection models. Uses the [DeepDarts dataset](https://github.com/wmcnally/deep-darts) (16k images, McNally et al.) to train YOLO11 variants for the DartsVision Android app.

## Quick Start (Docker + GPU)

### 1. Download the DeepDarts dataset

Get `cropped_images.zip` (3.35 GB) and `labels_pkl.zip` (340 KB) from [IEEE Dataport](https://ieee-dataport.org/open-access/deepdarts-dataset).

Extract into the repo on the host:

```bash
mkdir -p data/raw/deep-darts/dataset
unzip cropped_images.zip -d data/raw/deep-darts/dataset/
# The zip has nested structure: cropped_images/cropped_images/800/ — flatten it:
mv data/raw/deep-darts/dataset/cropped_images/cropped_images/800 data/raw/deep-darts/dataset/cropped_images_tmp
rm -rf data/raw/deep-darts/dataset/cropped_images
mv data/raw/deep-darts/dataset/cropped_images_tmp data/raw/deep-darts/dataset/cropped_images
unzip labels_pkl.zip -d data/raw/deep-darts/dataset/
```

Expected structure:
```
data/raw/deep-darts/dataset/
  labels.pkl
  cropped_images/
    d1_02_04_2020/
      IMG_1081.JPG
      ...
    d2_03_08_2020/
      ...
```

### 2. Build Docker image

```bash
docker compose build
```

### 3. Convert dataset (inside Docker)

**Important:** Conversion must run inside the Docker container so symlinks and config paths use `/workspace/` (the container path), not the host path.

```bash
docker compose run train python scripts/download_and_convert.py
```

This creates symlinks + YOLO11 label files in `data/processed/` and regenerates `configs/dataset_*.yaml` with correct paths.

### 4. Train models

```bash
# Single GPU
docker compose run train python scripts/train_pose.py --epochs 100 --batch 16 --gpu 0

# Multi-GPU DDP (batch auto-scales)
docker compose run train python scripts/train_pose.py --epochs 100 --gpu 0,1,2,3

# All three models:
docker compose run train python scripts/train_pose.py --epochs 100 --gpu 0
docker compose run train python scripts/train_detect.py --epochs 100 --gpu 0
docker compose run train python scripts/train_calibration.py --epochs 100 --gpu 0
```

### 5. Export to TFLite

```bash
docker compose run train python scripts/export_tflite.py --all
```

### 6. Copy to Android app

```bash
cp models/*.tflite ../darts_vision/app/src/main/assets/models/
```

## Without Docker (Local Python)

If you have CUDA + Python on the host:

```bash
pip install -r requirements.txt
python scripts/download_and_convert.py   # creates symlinks + configs for THIS machine
python scripts/train_pose.py --epochs 100 --gpu 0
python scripts/export_tflite.py --all
```

**Note:** If you switch between Docker and local Python, re-run `download_and_convert.py` — it regenerates symlinks and configs with the correct paths for the current environment.

## Approaches

| # | Model | What It Detects | Precision | Size | Use Case |
|---|-------|---------------|-----------|------|----------|
| 1 | **YOLO11n-pose** | Board + 7 keypoints (4 cal + 3 dart tips) | ~2 deg | ~6 MB | Primary: dart + board detection with keypoint accuracy |
| 2 | **YOLO11n-detect** | Board bbox + dart tip bbox | ~5 deg | ~4 MB | Simpler: bbox-only, less precise dart tip |
| 3 | **Board calibration** | Board + 4 calibration keypoints | ~1 deg | ~5 MB | Replace CV BoardDetector with ML |

## Dataset

**Source:** [DeepDarts](https://github.com/wmcnally/deep-darts) by McNally et al. ([IEEE Dataport](https://ieee-dataport.org/open-access/deepdarts-dataset))

| Property | Value |
|----------|-------|
| Images | 16,050 (800x800 px, cropped dartboards) |
| Annotations | keypoints: 4 calibration corners + up to 3 dart tips per image |
| Split | d1: 15,000 face-on + d2: 1,050 multi-angle |
| Train/Val/Test | ~80/8/13% (DeepDarts official splits) |

### DeepDarts Annotation Format

`labels.pkl` is a pandas DataFrame with columns:
- `img_folder` — e.g. `d1_02_04_2020`
- `img_name` — e.g. `IMG_1081.JPG`
- `bbox` — original crop bbox `[x, y, w, h]` in pixels
- `xy` — list of `[x, y]` normalized keypoints (0-1). Points 0-3 = calibration corners, 4+ = dart tips

### YOLO11 Format Conversions

`download_and_convert.py` produces three datasets:

1. **yolo11_pose/** — One `dartboard` class, 7 keypoints (class cx cy w h x1 y1 v1 ... x7 y7 v7)
2. **yolo11_detect/** — Two classes: `dartboard` + `dart_tip` bounding boxes
3. **board_calibration/** — One `dartboard` class, 4 calibration keypoints

Images are symlinked (not copied) to save disk space.

## Training Details

### YOLO11n-pose (Recommended)

- **Architecture:** YOLO11 nano with pose head
- **Input:** 640x640
- **Keypoints:** 7 per board (4 calibration + 3 dart tips)
- **Augmentation:** HSV jitter, rotation, flip, mosaic, mixup, erasing
- **Output:** `runs/pose/yolo11n_darts_pose/weights/best.pt`

### YOLO11n-detect (Simpler)

- **Architecture:** YOLO11 nano detection
- **Classes:** dartboard (bbox), dart_tip (small bbox ~20x20 px)
- **Output:** `runs/detect/yolo11n_darts_detect/weights/best.pt`

### Board Calibration

- **Architecture:** YOLO11 nano pose (4 keypoints only)
- **Keypoints:** double-20, double-6, double-3, double-11 corners
- **More rotation augmentation (+-30 deg)** since calibration needs angle invariance
- **Output:** `runs/calibration/yolo11n_board_calibration/weights/best.pt`

### Multi-GPU DDP

```bash
# 4 GPUs — batch auto-scales to 64 (16 per GPU)
docker compose run train python scripts/train_pose.py --gpu 0,1,2,3

# Override batch size
docker compose run train python scripts/train_pose.py --gpu 0,1,2,3 --batch 128
```

`docker-compose.yml` sets `shm_size: 16g` for DDP shared memory.

## Export for Android

| Model | FP32 | INT8 |
|-------|------|------|
| YOLO11n-pose | ~6 MB | ~2 MB |
| YOLO11n-detect | ~4 MB | ~1.5 MB |
| Board calibration | ~5 MB | ~1.5 MB |

```bash
# Export all
docker compose run train python scripts/export_tflite.py --all

# INT8 quantized
docker compose run train python scripts/export_tflite.py --all --int8
```

## Directory Structure

```
darts-vision-ml/
  Dockerfile
  docker-compose.yml
  requirements.txt
  configs/                       # Auto-generated by download_and_convert.py
    dataset_pose.yaml
    dataset_detect.yaml
    dataset_calibration.yaml
  scripts/
    download_and_convert.py
    train_pose.py
    train_detect.py
    train_calibration.py
    export_tflite.py
  data/
    raw/deep-darts/dataset/      # Source dataset (extract manually)
      labels.pkl
      cropped_images/
    processed/                    # Symlinks + labels (generated)
      yolo11_pose/
      yolo11_detect/
      board_calibration/
  runs/                           # Training runs (generated)
  models/                         # Exported TFLite (generated)
```

## Troubleshooting

**Symlinks broken?** Re-run `python scripts/download_and_convert.py` inside the same environment (Docker or local) where you'll train. Symlinks use absolute paths from the conversion machine.

**CUDA OOM?** Reduce batch: `--batch 8` or `--batch 4`.

**Docker can't see GPU?** Ensure [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) is installed.