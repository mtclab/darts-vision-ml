# Darts-Vision-ML Training Pipeline

On-device automatic darts scoring — model training & evaluation.
Trains a YOLOv8n object detector to locate dart tips and board calibration corners, then exports to INT8 TFLite for the Flutter app.

## Project Structure

```
darts-vision-ml/
├── data/
│   ├── raw/
│   │   └── deep-darts/
│   │       └── dataset/
│   │           ├── labels.pkl        # DeepDarts annotations (DataFrame)
│   │           └── cropped_images/ # 16k cropped board images
│   ├── processed/
│   │   └── yolo_detect_deepdarts/  # Converted YOLO format (.txt labels)
│   └── darts.yaml                    # YOLO dataset config
├── src/
    │   ├── convert_dataset.py    # labels.pkl → YOLO detection .txt
    │   ├── convert_pose.py       # labels.pkl → YOLO pose .txt (recommended)
    │   ├── train.py              # Detection training
    │   ├── train_pose.py         # Pose estimation training
    │   ├── export_tflite.py      # INT8 quantization export
    │   └── evaluate.py           # End-to-end scoring accuracy (PCS / MASE)
├── models/               # Final .tflite outputs
├── runs/                 # Training checkpoints
└── requirements.txt
```

## Prerequisites

- Python 3.10+
- CUDA 12.x
- 3× NVIDIA GPU (tested on RTX 4000 Ada SFF ×3)

## Setup

```bash
cd darts-vision-ml
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

## 1. Prepare Dataset

Place DeepDarts data into `data/raw/deep-darts/dataset/`:

```
data/raw/deep-darts/dataset/
├── labels.pkl          # DataFrame with columns img_folder, img_name, bbox, xy
└── cropped_images/     # 16k cropped board images organized by folder
```

Convert to YOLO format:

```bash
python src/convert_dataset.py \
    --labels data/raw/deep-darts/dataset/labels.pkl \
    --output data/processed/yolo_detect_deepdarts
```

This writes `train/`, `val/`, `test/` folders with images + `.txt` labels.

## 2. Train (DDP on 3 GPUs)

Use the `darts.yaml` generated inside the output directory (it has the correct dataset root path):

```bash
torchrun --nproc_per_node=3 src/train.py \
    --data data/processed/yolo_detect_deepdarts/darts.yaml \
    --model yolov8n.pt \
    --epochs 50 \
    --imgsz 800 \
    --batch 24 \
    --device 0,1,2 \
    --project runs/darts \
    --name yolov8n_800
```

Expected: ~2–3 hours for 50 epochs on 3× RTX 4000 Ada.

Best checkpoint saved to:
```
runs/darts/yolov8n_800/weights/best.pt
```

## 3. Export to TFLite INT8

```bash
python src/export_tflite.py \
    --weights runs/darts/yolov8n_800/weights/best.pt \
    --data data/processed/yolo_detect_deepdarts/darts.yaml \
    --imgsz 800 \
    --output models/
```

Output:
```
models/darts_detector.tflite   # ~3–6 MB, INT8 quantized
```

Copy to Flutter app:
```bash
cp models/darts_detector.tflite ../darts_vision_flutter/assets/models/
```

## 4. Evaluate End-to-End Accuracy

```bash
python src/evaluate.py \
    --model runs/darts/yolov8n_800/weights/best.pt \
    --labels data/raw/deep-darts/dataset/labels.pkl \
    --conf 0.25 \
    --device 0
```

Output metrics:
- **PCS** (Percent Correct Score): exact total score match
- **MASE** (Mean Absolute Score Error)
- Per-dart detection & localization error

## 5. Training for Best Scoring Accuracy

The default config gives ~98.7% mAP50 but end-to-end scoring accuracy (PCS/MASE) is limited by **bbox center approximation** and **low resolution**. For production, train with these improvements:

### 5.1 Tighter Bounding Boxes

Default `bbox-size=0.025` creates 50px boxes on 2000px crops. Corner localization errors blow up through homography warp.

```bash
python src/convert_dataset.py \
    --labels data/raw/deep-darts/dataset/labels.pkl \
    --output data/processed/yolo_detect_tight \
    --bbox-size 0.01
```

### 5.2 Higher Resolution + Better Model

`imgsz=800` downsamples 2000px crops too aggressively. Use `yolov8s` at `1280`:

```bash
torchrun --nproc_per_node=3 src/train.py \
    --data data/processed/yolo_detect_tight/darts.yaml \
    --model yolov8s.pt \
    --imgsz 1280 \
    --batch 10 \
    --epochs 100 \
    --patience 20 \
    --lr0 0.005 \
    --degrees 5 \
    --translate 0.05 \
    --scale 0.1 \
    --hsv_h 0.01 \
    --hsv_s 0.2 \
    --hsv_v 0.2 \
    --mosaic 0 \
    --mixup 0 \
    --fliplr 0 \
    --close_mosaic 0 \
    --project runs/darts \
    --name yolov8s_1280_tight
```

**Why these changes:**
- `mosaic=0`: Mixing 4 images destroys board geometry (corners from img A + darts from img B = invalid homography)
- `fliplr=0`: Horizontal flip breaks left/right scoring on asymmetric boards
- `degrees=5`: Boards are always upright; large rotation moves corners off-image
- `scale=0.1`: Camera distance is fixed in your setup
- `batch=10`: Memory tradeoff for `imgsz=1280` on 20GB GPUs

### 5.3 Pose Estimation (Maximum Accuracy)

Train **YOLOv8-pose** to directly regress 7 keypoints, eliminating the bbox-center approximation entirely.

Convert to pose format:

```bash
python src/convert_pose.py \
    --labels data/raw/deep-darts/dataset/labels.pkl \
    --output data/processed/yolo_pose_darts
```

Train:

```bash
torchrun --nproc_per_node=3 src/train_pose.py \
    --data data/processed/yolo_pose_darts/pose.yaml \
    --model yolov8n-pose.pt \
    --epochs 100 \
    --imgsz 1280 \
    --batch 10 \
    --project runs/darts_pose \
    --name yolov8n_pose_1280
```

Export to TFLite:

```bash
python src/export_tflite.py \
    --weights runs/darts_pose/yolov8n_pose_1280/weights/best.pt \
    --data data/processed/yolo_pose_darts/pose.yaml \
    --imgsz 1280 \
    --output models/
```

| Approach | mAP50 | mAP50-95 | PCS | Best For |
|----------|-------|----------|-----|----------|
| `yolov8n` default | 98.7% | 75.4% | ~ | Quick baseline |
| `yolov8s` 1280 tight | ~99% | ~80% | ↑↑ | Production scoring |
| `yolov8n-pose` | - | - | ↑↑↑ | Maximum accuracy |

## Customization

| Parameter | Default | Effect |
|-----------|---------|--------|
| `imgsz` | 800 | Input resolution (match app model input) |
| `epochs` | 50 | More epochs for higher accuracy |
| `batch` | 24 | Per-GPU batch (×3 for DDP) |
| `conf` | 0.25 | Inference confidence threshold |

## Data Format Notes

DeepDarts `labels.pkl` is a DataFrame with columns:
- `img_folder`: folder name under `cropped_images/`
- `img_name`: image filename
- `bbox`: bounding box `[x, y, w, h]`
- `xy`: list of `[x, y]` points (4 calibration corners + up to 3 darts, normalized 0-1)

Script converts `xy` list to `(7, 3)` array:
- indices 0-3: calibration corners
- indices 4-6: darts
- 3rd column: visibility flag (0 or 1)

YOLO format per image `.txt`:
```
1 0.5123 0.4821 0.0250 0.0250   # class cal_corner, center, bbox_size
0 0.3456 0.6789 0.0250 0.0250   # class dart
```

## License

See [LICENSE](../LICENSE).
