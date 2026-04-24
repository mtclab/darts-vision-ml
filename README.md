# Darts-Vision-ML Training Pipeline

On-device automatic darts scoring — model training & evaluation.
Trains a YOLOv8n object detector to locate dart tips and board calibration corners, then exports to INT8 TFLite for the Flutter app.

## Project Structure

```
darts-vision-ml/
├── data/
│   ├── raw/              # Place DeepDarts images + labels.pkl here
│   ├── yolo/             # Converted YOLO format (.txt labels)
│   └── darts.yaml        # YOLO dataset config
├── src/
│   ├── convert_dataset.py  # labels.pkl → YOLO .txt
│   ├── train.py          # DDP training script
│   ├── export_tflite.py  # INT8 quantization export
│   └── evaluate.py       # End-to-end scoring accuracy (PCS / MASE)
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

Place DeepDarts data into `data/raw/`:

```
data/raw/
  ├── images/           # 16k cropped board images
  └── labels.pkl        # DeepDarts annotations
```

Convert to YOLO format:

```bash
python src/convert_dataset.py \
    --labels data/raw/labels.pkl \
    --output data/yolo
```

This writes `train/`, `val/`, `test/` folders with images + `.txt` labels.

## 2. Train (DDP on 3 GPUs)

```bash
torchrun --nproc_per_node=3 src/train.py \
    --data data/darts.yaml \
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
    --data data/darts.yaml \
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
    --labels data/raw/labels.pkl \
    --conf 0.25 \
    --device 0
```

Output metrics:
- **PCS** (Percent Correct Score): exact total score match
- **MASE** (Mean Absolute Score Error)
- Per-dart detection & localization error

## Customization

| Parameter | Default | Effect |
|-----------|---------|--------|
| `imgsz` | 800 | Input resolution (match app model input) |
| `epochs` | 50 | More epochs for higher accuracy |
| `batch` | 24 | Per-GPU batch (×3 for DDP) |
| `conf` | 0.25 | Inference confidence threshold |

## Data Format Notes

DeepDarts `labels.pkl` stores ground truth as:
- `img_paths`: list of image file paths
- `gt`: ndarray `(N, 7, 3)` → 4 calibration corners + up to 3 darts, visibility flag in 3rd channel

YOLO format per image `.txt`:
```
1 0.5123 0.4821 0.0250 0.0250   # class cal_corner, center, bbox_size
0 0.3456 0.6789 0.0250 0.0250   # class dart
```

## License

See [LICENSE](../LICENSE).
