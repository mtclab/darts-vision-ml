# DartsVision ML Training

Training and evaluation pipelines for darts detection and scoring models. Uses the [DeepDarts dataset](https://github.com/wmcnally/deep-darts) (16k images, McNally et al.) to train YOLO11 variants and evaluate Qwen3.5-VL models for the DartsVision Android app.

## App Architecture (3 Modes)

DartsVision is a **hand-held** darts scoring app — the phone is held in hand, not statically mounted. This rules out frame-differencing (which requires a stable reference frame).

| Mode | How | Offline? | Speed | Accuracy |
|------|-----|----------|-------|----------|
| **Quick** | YOLO board calibration (4 kp) + YOLO darts pose (7 kp) | Yes | Real-time (~30ms) | Good (~2 deg) |
| **Precise** | Camera frame → VLM → "T20, S5, D16" | Depends on VLM | 5-30s | Best |
| **Manual** | User taps score on board UI | Yes | Instant | Perfect |

Precise Mode VLM options (in priority order):
1. **On-device**: Qwen3.5-0.8B via LiteRT-LM (~1.2GB) — if accuracy ≥80%
2. **Cloud (free)**: Gemini 2.5 Flash API — 1,500 req/day free tier
3. **Self-hosted**: Ollama on user's own GPU — full privacy, zero cost

## Models

| Model | Purpose | Size | Status |
|-------|---------|------|--------|
| YOLO11n-pose (board_calibration) | Board + 4 calibration keypoints | ~11 MB TFLite | Trained (mAP50-95=0.995) |
| YOLO11n-pose (darts_pose) | Board + 7 keypoints (4 cal + 3 dart tips) | ~11 MB TFLite | Trained (mAP50-95=0.912) |
| Qwen3.5-0.8B | VLM dart score recognition (baseline) | ~1.2 GB LiteRT | Under evaluation |
| Qwen3.5-0.8B LoRA | VLM dart score recognition (fine-tuned) | ~1.2 GB + LoRA | Pending baseline results |
| Qwen3.5-2B | VLM dart score recognition (comparison) | ~4 GB LiteRT | Pending evaluation |
| ~~SmolVLM-500M~~ | ~~VLM via ONNX Runtime~~ | ~~485 MB~~ | Removed — wrong tool for measurement task |

See [docs/approaches.md](docs/approaches.md) for full comparison.

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

### 3. Convert dataset (inside Docker — required)

**Critical:** Conversion **must** run inside the Docker container. Symlinks and `configs/dataset_*.yaml` use absolute paths from the conversion environment. If you convert on the host, the paths won't exist inside Docker and training will fail.

```bash
docker compose run train python scripts/download_and_convert.py
```

This creates symlinks + YOLO11 label files in `data/processed/` and generates `configs/dataset_*.yaml` with `/workspace/...` paths. Re-running cleans old data first.

### 4. Train YOLO models

```bash
# Board calibration (4 keypoints, primary detection)
docker compose run train python scripts/train_board_calibration.py --epochs 100 --gpu 0

# Darts pose (7 keypoints, Quick Mode detection)
docker compose run train python scripts/train_darts_pose.py --epochs 100 --gpu 0
```

Alternative (bbox-only, less precise — skip unless pose unavailable):
```bash
docker compose run train python scripts/train_darts_detect.py --epochs 100 --gpu 0
```

Each script validates that `configs/dataset_*.yaml` exists and the dataset path is accessible before training. If validation fails, re-run step 3.

Multi-GPU DDP (batch auto-scales):
```bash
docker compose run train python scripts/train_darts_pose.py --epochs 100 --gpu 0,1,2,3
```

Resume interrupted training:
```bash
docker compose run train python scripts/train_darts_pose.py --resume runs/darts_pose/train/weights/last.pt
```

### 5. Export to TFLite

```bash
# Export recommended models (calibration + pose)
docker compose run train python scripts/export_tflite.py

# Single model
docker compose run train python scripts/export_tflite.py --model runs/board_calibration/train/weights/best.pt
```

### 6. Copy to Android app

```bash
cp models/*.tflite ../darts_vision/app/src/main/assets/models/
```

### 7. Evaluate Qwen3.5-VL for Precise Mode

```bash
# Build (same image serves both YOLO training and VLM work)
docker compose build

# Interactive test on your own images
docker compose run qwen python scripts/test_qwen_vision.py \
  --image /path/to/dartboard.jpg --prompt all

# Full evaluation on all 1,070 val images (multi-GPU)
docker compose run qwen python scripts/evaluate_qwen_dataset.py \
  --model Qwen/Qwen3.5-0.8B --all --gpus 0,1,2,3

# Compare with 2B
docker compose run qwen python scripts/evaluate_qwen_dataset.py \
  --model Qwen/Qwen3.5-2B --all --output results/qwen_2b_baseline.csv
```

See **Qwen3.5-VL section** below for full fine-tuning pipeline.

## Without Docker (Local Python)

If you have CUDA + Python on the host:

```bash
pip install -r requirements.txt
python scripts/download_and_convert.py
python scripts/train_board_calibration.py --epochs 100 --gpu 0
python scripts/train_darts_pose.py --epochs 100 --gpu 0
python scripts/export_tflite.py
```

**Note:** If you switch between Docker and local Python, re-run `download_and_convert.py` — it cleans old data and regenerates symlinks + configs with the correct paths for the current environment.

## YOLO Model Details

| Model | What It Detects | Precision | TFLite (FP32 / INT8) | Use Case |
|-------|---------------|-----------|----------------------|----------|
| **Board calibration** | Board + 4 calibration keypoints | ~1 deg | `board_calibration_float32.tflite` / `_int8` | Quick Mode: board geometry |
| **Darts pose** | Board + 7 keypoints (4 cal + 3 dart tips) | ~2 deg | `darts_pose_float32.tflite` / `_int8` | Quick Mode: dart detection |
| **Darts detect** | Board bbox + dart tip bbox | ~5 deg | `darts_detect_float32.tflite` / `_int8` | Alternative: bbox-only, less precise |

## Dataset

**Source:** [DeepDarts](https://github.com/wmcnally/deep-darts) by McNally et al. ([IEEE Dataport](https://ieee-dataport.org/open-access/deepdarts-dataset))

| Property | Value |
|----------|-------|
| Images | 16,050 (800x800 px, cropped dartboards) |
| Annotations | keypoints: 4 calibration corners + up to 3 dart tips per image |
| Split | d1: 15,000 face-on + d2: 1,050 multi-angle |
| Train/Val/Test | ~80/8/13% (DeepDarts official splits) |

**Weakness for hand-held use:** 94% of images are face-on (d1 split). Only ~1,050 are multi-angle. Augmenting with phone photos at various angles will improve robustness.

### DeepDarts Annotation Format

`labels.pkl` is a pandas DataFrame with columns:
- `img_folder` — e.g. `d1_02_04_2020`
- `img_name` — e.g. `IMG_1081.JPG`
- `bbox` — original crop bbox `[x, y, w, h]` in pixels
- `xy` — list of `[x, y]` normalized keypoints (0-1). Points 0-3 = calibration corners, 4+ = dart tips

### YOLO11 Format Conversions

`download_and_convert.py` produces three datasets:

1. **board_calibration/** — One `dartboard` class, 4 calibration keypoints
2. **yolo11_pose/** — One `dartboard` class, 7 keypoints (class cx cy w h x1 y1 v1 ... x7 y7 v7)
3. **yolo11_detect/** — Two classes: `dartboard` + `dart_tip` bounding boxes

Images are symlinked (not copied) to save disk space.

## Training Details

### Board Calibration

- **Architecture:** YOLO11 nano pose (4 keypoints only)
- **Keypoints:** double-20, double-6, double-3, double-11 corners
- **Rotation augmentation (+-30 deg)** for angle invariance
- **Output:** `runs/board_calibration/train/weights/best.pt`

### Darts Pose

- **Architecture:** YOLO11 nano pose (7 keypoints)
- **Input:** 640x640
- **Keypoints:** 7 per board (4 calibration + 3 dart tips)
- **Augmentation:** HSV jitter, rotation, flip, mosaic, mixup, erasing
- **Output:** `runs/darts_pose/train/weights/best.pt`

### Darts Detect (Alternative)

- **Architecture:** YOLO11 nano detection
- **Classes:** dartboard (bbox), dart_tip (small bbox ~20x20 px)
- **Output:** `runs/darts_detect/train/weights/best.pt`

### Multi-GPU DDP

```bash
# 4 GPUs — batch auto-scales to 64 (16 per GPU)
docker compose run train python scripts/train_darts_pose.py --gpu 0,1,2,3

# Override batch size
docker compose run train python scripts/train_darts_pose.py --gpu 0,1,2,3 --batch 128
```

`docker-compose.yml` sets `shm_size: 16g` for DDP shared memory.

## Export for Android

| Model | TFLite Name (FP32) | FP32 Size |
|-------|--------------------|-----------|
| Board calibration | `board_calibration_float32.tflite` | ~11 MB |
| Darts pose | `darts_pose_float32.tflite` | ~11 MB |

## Directory Structure

```
darts-vision-ml/
  Dockerfile
  docker-compose.yml
  requirements.txt
  configs/                          # Auto-generated by download_and_convert.py
    dataset_calibration.yaml       #   (gitignored — environment-specific paths)
    dataset_pose.yaml
    dataset_detect.yaml
    qwen_lora_config.yaml          # LoRA hyperparameters for Qwen3.5
  scripts/
    download_and_convert.py         # Dataset conversion + config generation
    train_utils.py                  # Shared: ensure_pretrained, parse_device, validate_dataset
    train_board_calibration.py      # Board calibration (4 keypoints) — Quick Mode
    train_darts_pose.py             # Darts pose (7 keypoints) — Quick Mode
    train_darts_detect.py           # Darts detect (bbox) — alternative
    export_tflite.py                # Export .pt → .tflite
    dart_board.py                   # Shared: dart board geometry, keypoint-to-score conversion
    test_qwen_vision.py             # Interactive VLM testing (multiple prompts)
    evaluate_qwen_dataset.py        # Systematic VLM evaluation against DeepDarts (multi-GPU)
    prepare_qwen_training.py        # Convert YOLO labels → VLM instruction format
    train_qwen_lora.py              # LoRA fine-tuning for Qwen3.5-VL
  data/
    raw/deep-darts/dataset/         # Source dataset (extract manually)
      labels.pkl
      cropped_images/
    processed/                      # Symlinks + labels (generated, cleaned on re-run)
      board_calibration/
      yolo11_pose/
      yolo11_detect/
      vlm_train/                    # Qwen3.5 training data (JSONL, generated)
  weights/                          # Pretrained YOLO weights (auto-downloaded, gitignored)
  runs/                             # Training runs (generated)
    board_calibration/train/weights/
    darts_pose/train/weights/
    darts_detect/train/weights/
    qwen_lora/                      # Qwen3.5 LoRA training output
  models/                           # Exported TFLite (generated)
  results/                          # VLM evaluation results (CSV)
```

## Troubleshooting

**`[ERROR] Dataset path not found`** — The `configs/dataset_*.yaml` has paths from a different environment. Re-run conversion inside Docker:
```bash
docker compose run train python scripts/download_and_convert.py
```

**`[ERROR] Config not found`** — You haven't run dataset conversion yet. See step 3 above.

**Pretrained `.pt` files in repo root** — Pretrained weights now download to `weights/` (gitignored). If you see `.pt` files in root, they can be deleted.

**Symlinks broken?** — Re-run `python scripts/download_and_convert.py` inside the same environment (Docker or local) where you'll train.

**CUDA OOM?** — Reduce batch: `--batch 8` or `--batch 4`.

**Docker can't see GPU?** — Ensure [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) is installed.

**Resume interrupted training** — All scripts support `--resume`:
```bash
docker compose run train python scripts/train_darts_pose.py --resume runs/darts_pose/train/weights/last.pt
```

---

## Qwen3.5-VL: VLM Dart Score Recognition (Precise Mode)

### Overview

Evaluate and fine-tune Qwen3.5-VL models (0.8B, 2B) for reading dart scores from camera frames. This is the **Precise Mode** — send full frame to VLM, get back scores like "T20, S5, D16". Complements YOLO Quick Mode.

**Why Qwen3.5-VL?**
- Qwen3.5 (Feb 2026) uses hybrid GatedDeltaNet+attention — efficient for on-device
- 0.8B has existing LiteRT-LM conversion (~1.2GB) — could run on-device
- Benchmarks: OCRBench 79.1%, RefCOCO 77.8%, RealWorldQA 61.6%
- Apache 2.0 license
- Requires `transformers>=4.57.0`, model class `Qwen3_5ForConditionalGeneration`

### Docker Setup

```bash
# Build (same image serves both YOLO training and VLM work)
docker compose build

# VLM evaluation and fine-tuning
docker compose run qwen bash
```

The `qwen` service has GPU access, HuggingFace cache volume, and all VLM dependencies.

### Step 1: Baseline Test on Custom Images

```bash
# Test on your own dart board screenshots
docker compose run qwen python scripts/test_qwen_vision.py \
  --image /path/to/dartboard.jpg \
  --model Qwen/Qwen3.5-0.8B

# Test all 4 prompt strategies
docker compose run qwen python scripts/test_qwen_vision.py \
  --image /path/to/dartboard.jpg \
  --prompt all

# Compare 0.8B vs 2B
docker compose run qwen python scripts/test_qwen_vision.py \
  --image /path/to/dartboard.jpg \
  --model Qwen/Qwen3.5-2B
```

### Step 2: Systematic Evaluation on DeepDarts

```bash
# Evaluate on 200 val images (quick, ~30 min)
docker compose run qwen python scripts/evaluate_qwen_dataset.py \
  --model Qwen/Qwen3.5-0.8B \
  --num-images 200

# Full evaluation on all 1,070 val images (multi-GPU)
docker compose run qwen python scripts/evaluate_qwen_dataset.py \
  --model Qwen/Qwen3.5-0.8B \
  --all --gpus 0,1,2,3

# Compare with 2B
docker compose run qwen python scripts/evaluate_qwen_dataset.py \
  --model Qwen/Qwen3.5-2B \
  --all --gpus 0,1,2,3 \
  --output results/qwen_2b_baseline.csv
```

Outputs: `results/<model>_baseline_<timestamp>.csv` with per-image accuracy.

### Step 3: Prepare Fine-Tuning Data

```bash
# Convert DeepDarts labels → VLM instruction format (JSONL)
docker compose run qwen python scripts/prepare_qwen_training.py

# Output: data/processed/vlm_train/train.jsonl, val.jsonl
```

### Step 4: LoRA Fine-Tuning

```bash
# Fine-tune with default config (configs/qwen_lora_config.yaml)
docker compose run qwen python scripts/train_qwen_lora.py

# Override parameters
docker compose run qwen python scripts/train_qwen_lora.py \
  --epochs 5 --lr 1e-4 --batch-size 2

# Resume interrupted training
docker compose run qwen python scripts/train_qwen_lora.py \
  --resume runs/qwen_lora/checkpoint-500
```

### Step 5: Evaluate Fine-Tuned Model

```bash
# Re-run evaluation with the fine-tuned model
docker compose run qwen python scripts/evaluate_qwen_dataset.py \
  --model runs/qwen_lora/merged_model/qwen3.5-0.8b-darts-lora \
  --all \
  --output results/qwen_0.8b_lora.csv
```

### Next Steps After Evaluation

| Baseline Accuracy | Action |
|-------------------|--------|
| ≥ 80% | Skip fine-tuning, export Qwen3.5-0.8B to LiteRT for on-device Precise Mode |
| 50-80% | LoRA fine-tune on DeepDarts, likely reaches ≥ 85% |
| < 50% | Try Qwen3.5-2B; if still bad, use cloud VLM (Gemini) as Precise Mode |

If Qwen3.5 on-device doesn't work: **Gemini 2.5 Flash** is the cloud Precise Mode. Free tier = 1,500 req/day (enough for darts: ~3 req/turn). Backend proxy supports user's own API key + self-hosted Ollama.

### Qwen Scripts

| Script | Purpose |
|--------|---------|
| `scripts/dart_board.py` | Shared dart board geometry and keypoint-to-score conversion |
| `scripts/test_qwen_vision.py` | Interactive test with 4 prompt strategies + latency measurement |
| `scripts/evaluate_qwen_dataset.py` | Systematic eval against DeepDarts ground truth (multi-GPU) |
| `scripts/prepare_qwen_training.py` | Convert YOLO labels to VLM training format (JSONL) |
| `scripts/train_qwen_lora.py` | LoRA fine-tuning with PEFT + TRL |