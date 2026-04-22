# DartsVision ML Training

Training and evaluation pipelines for darts detection and scoring models. Uses the [DeepDarts dataset](https://github.com/wmcnally/deep-darts) (16k images, McNally et al.) to train YOLO11 variants and benchmark VLM models for the DartsVision Android app.

## App Architecture (3 Modes)

DartsVision is a **hand-held** darts scoring app — the phone is held in hand, not statically mounted. This rules out frame-differencing (requires a stable reference frame).

| Mode | How | Offline? | Speed | Accuracy |
|------|-----|----------|-------|----------|
| **Quick** | YOLO board calibration (4 kp) + YOLO darts pose (7 kp) | Yes | Real-time (~30ms) | TBD (evaluate with `evaluate_yolo_scoring.py`) |
| **Precise** | Cloud VLM (Gemini 2.5 Flash or self-hosted Ollama) | No (or LAN) | 5–30s | Best available |
| **Manual** | User taps score on board UI | Yes | Instant | Perfect |

**Accuracy reality check (2026-04-22):**

| Approach | Dart Count | Segment | Ring | Full Score | Offline? |
|----------|-----------|---------|------|------------|----------|
| **Qwen3.5-0.8B** (on-device) | 23.9% | 0.8% | 2.0% | **0.0%** | Yes (unusable) |
| **Qwen3.5-2B** (on-device) | 7.3% | 8.2% | 2.6% | **0.0%** | Yes (unusable) |
| **YOLO Quick** (bundled) | TBD | TBD | TBD | TBD | Yes (needs evaluation) |

On-device VLMs tested so far are **incompetent** at reading dart boards (0% full accuracy). LoRA fine-tuning would only help a model that already has some baseline competence. Current plan: use YOLO for offline Quick Mode, cloud VLM for Precise Mode, and skip on-device VLM until it actually works.

## Models

| Model | Purpose | Size | Status |
|-------|---------|------|--------|
| YOLO11n-pose (board_calibration) | 4 board calibration keypoints | ~11 MB TFLite | Trained (mAP50-95=0.995);¡end-to-end accuracy TBD |
| YOLO11n-pose (darts_pose) | 7 keypoints (4 cal + 3 dart tips) | ~11 MB TFLite | Trained (mAP50-95=0.912);¡end-to-end accuracy TBD |
| ~~Qwen3.5-0.8B~~ | ~~VLM on-device~~ | ~~~1.2 GB LiteRT~~ | **0% full accuracy — abandoned** |
| ~~Qwen3.5-2B~~ | ~~VLM on-device~~ | ~~~4 GB~~ | **0% full accuracy — abandoned** |
| ~~Qwen3-VL-2B~~ | ~~VLM on-device~~ | ~~~4 GB~~ | **0% full accuracy — abandoned** |
| ~~Granite-Vision-2B~~ | ~~VLM on-device~~ | ~~~4 GB~~ | **0% full accuracy — abandoned** |
| ~~Moondream2~~ | ~~VLM on-device~~ | ~~~3 GB~~ | **0% full accuracy — abandoned** |

¡mAP is detection quality, not final score accuracy. Run `evaluate_yolo_scoring.py` to get DartCnt / Segment / Ring / Full metrics directly comparable to VLM results.

## Quick Start (Docker + GPU)

### 1. Download DeepDarts Dataset

Get `cropped_images.zip` (3.35 GB) and `labels_pkl.zip` (340 KB) from [IEEE Dataport](https://ieee-dataport.org/open-access/deepdarts-dataset).

Extract:
```bash
mkdir -p data/raw/deep-darts/dataset
unzip cropped_images.zip -d data/raw/deep-darts/dataset/
# Flatten nested structure if needed:
mv data/raw/deep-darts/dataset/cropped_images/cropped_images/800 data/raw/deep-darts/dataset/cropped_images_tmp
rm -rf data/raw/deep-darts/dataset/cropped_images
mv data/raw/deep-darts/dataset/cropped_images_tmp data/raw/deep-darts/dataset/cropped_images
unzip labels_pkl.zip -d data/raw/deep-darts/dataset/
```

### 2. Build Docker Image

```bash
docker compose build
```

### 3. Convert Dataset (inside Docker)

**Must run inside Docker.** Symlinks and `configs/dataset_*.yaml` use absolute paths from the conversion environment.

```bash
docker compose run train python scripts/download_and_convert.py
```

Creates `data/processed/` with symlinks + YOLO labels, and generates `configs/dataset_*.yaml` with `/workspace/...` paths.

### 4. Evaluate Existing Models

Before training anything new, know where you stand:

```bash
# End-to-end YOLO scoring accuracy (TFLite models — same as Android)
docker compose run train python scripts/evaluate_yolo_scoring.py \
  --cal-model /workspace/../darts_vision/app/src/main/assets/models/board_calibration_float32.tflite \
  --pose-model /workspace/../darts_vision/app/src/main/assets/models/darts_pose_float32.tflite \
  --backend tflite --split val --save-csv results/tflite_scoring.csv

# Or with PyTorch models (if you have trained weights)
docker compose run train python scripts/evaluate_yolo_scoring.py \
  --cal-model runs/board_calibration/train/weights/best.pt \
  --pose-model runs/darts_pose/train/weights/best.pt \
  --backend pytorch --split val --save-csv results/pytorch_scoring.csv
```

Metrics match VLM benchmark format: DartCnt, Segment, Ring, Full Score.

### 5. Train Better YOLO Models (if needed)

If YOLO scoring accuracy is **<70%**, retrain or augment:

```bash
# Board calibration (4 keypoints)
docker compose run train python scripts/train_board_calibration.py --epochs 100 --gpu 0

# Darts pose (7 keypoints)
docker compose run train python scripts/train_darts_pose.py --epochs 100 --gpu 0
```

Multi-GPU DDP:
```bash
docker compose run train python scripts/train_darts_pose.py --epochs 100 --gpu 0,1,2,3
```

Resume:
```bash
docker compose run train python scripts/train_darts_pose.py --resume runs/darts_pose/train/weights/last.pt
```

### 6. Export to TFLite

```bash
# Export recommended models (calibration + pose)
docker compose run train python scripts/export_tflite.py

# Copy to Android app
cp models/*.tflite ../darts_vision/app/src/main/assets/models/
```

### 7. Benchmark Cloud VLM (for Precise Mode comparison)

```bash
# Gemini 2.5 Flash (via REST — no model download, fast)
docker compose run train python scripts/evaluate_vlm_benchmark.py \
  --model gemini-2.5-flash --all --gpus 0

# Self-hosted Ollama (llava:13b)
docker compose run train python scripts/evaluate_vlm_benchmark.py \
  --model ollama/llava:13b --all --gpus 0

# Compare all candidates
docker compose run qwen python scripts/evaluate_vlm_benchmark.py \
  --models Qwen/Qwen3.5-0.8B,Qwen/Qwen3.5-2B,Qwen/Qwen3-VL-2B-Instruct,ibm-granite/granite-vision-3.2-2b,vikhyatk/moondream2 \
  --all --gpus 0,1,2
```

## Without Docker (Local Python)

```bash
pip install -r requirements.txt
python scripts/download_and_convert.py
python scripts/train_board_calibration.py --epochs 100 --gpu 0
python scripts/train_darts_pose.py --epochs 100 --gpu 0
python scripts/export_tflite.py
```

**Note:** If you switch between Docker and local Python, re-run `download_and_convert.py` — paths in `configs/` are environment-specific.

## YOLO Model Details

| Model | What It Detects | Precision | TFLite (FP32) | End-to-end Accuracy |
|-------|---------------|-----------|----------------|---------------------|
| **Board calibration** | 4 calibration keypoints | ~1 deg | `board_calibration_float32.tflite` | Run `evaluate_yolo_scoring.py` |
| **Darts pose** | 7 keypoints (4 cal + 3 dart tips) | ~2 deg | `darts_pose_float32.tflite` | Run `evaluate_yolo_scoring.py` |
| **Darts detect** | Board bbox + dart tip bbox | ~5 deg | `darts_detect_float32.tflite` | Not yet evaluated |

## Dataset

**Source:** [DeepDarts](https://github.com/wmcnally/deep-darts) by McNally et al. ([IEEE Dataport](https://ieee-dataport.org/open-access/deepdarts-dataset))

| Property | Value |
|----------|-------|
| Images | 16,050 (800×800 px, cropped dartboards) |
| Annotations | keypoints: 4 calibration corners + up to 3 dart tips per image |
| Split | d1: 15,000 face-on + d2: 1,050 multi-angle |
| Train/Val/Test | ~80/8/13% (DeepDarts official splits) |

**Weakness for hand-held use:** 94% face-on. Multi-angle augmentation (phone photos) will improve robustness.

### DeepDarts Annotation Format

`labels.pkl` is a pandas DataFrame with columns:
- `img_folder` — e.g. `d1_02_04_2020`
- `img_name` — e.g. `IMG_1081.JPG`
- `bbox` — original crop bbox `[x, y, w, h]`
- `xy` — list of `[x, y]` normalized keypoints (0–1). Points 0–3 = calibration, 4+ = dart tips

## Evaluation Scripts

| Script | What it evaluates | Backend | Use case |
|--------|-----------------|---------|----------|
| `evaluate_yolo_scoring.py` | YOLO → homography → scores | PyTorch or TFLite | End-to-end Quick Mode accuracy |
| `evaluate_tflite_scoring.py` | TFLite → homography → scores | TFLite only | Same as Android inference |
| `evaluate_vlm_benchmark.py` | VLM text → score parsing | transformers | Cloud VLM accuracy comparison |
| `test_qwen_vision.py` | Single-image VLM test | transformers | Quick prompt experiments |

## Training Details

### Board Calibration

- Architecture: YOLO11 nano pose (4 keypoints)
- Keypoints: double-20, double-6, double-3, double-11 corners
- Rotation augmentation (±30°)
- Output: `runs/board_calibration/train/weights/best.pt`

### Darts Pose

- Architecture: YOLO11 nano pose (7 keypoints)
- Input: 640×640
- Augmentation: HSV jitter, rotation, flip, mosaic, mixup, erasing
- Output: `runs/darts_pose/train/weights/best.pt`

### Multi-GPU DDP

```bash
# Batch auto-scales to 16 per GPU
docker compose run train python scripts/train_darts_pose.py --gpu 0,1,2,3
```

`docker-compose.yml` sets `shm_size: 16g` for DDP shared memory.

## Directory Structure

```
darts-vision-ml/
  Dockerfile
  docker-compose.yml
  requirements.txt
  configs/
    dataset_calibration.yaml        # Auto-generated (gitignored)
    dataset_pose.yaml
    dataset_detect.yaml
    qwen_lora_config.yaml          # LoRA hyperparameters (kept for reference)
  scripts/
    download_and_convert.py         # Dataset conversion
    train_utils.py                  # Shared helpers
    train_board_calibration.py      # Train board calibration model
    train_darts_pose.py             # Train darts pose model
    train_darts_detect.py           # Train bbox-only model
    export_tflite.py                # Export .pt → .tflite
    evaluate_yolo_scoring.py        # NEW: end-to-end YOLO accuracy
    evaluate_tflite_scoring.py      # NEW: TFLite end-to-end accuracy
    evaluate_vlm_benchmark.py       # VLM comparison benchmark
    test_qwen_vision.py             # Interactive VLM testing
    dart_board.py                   # Board geometry + score classification
    prepare_qwen_training.py        # Convert labels → VLM training format
    train_qwen_lora.py              # LoRA fine-tuning (dormant — baseline 0%)
  data/
    raw/deep-darts/dataset/         # Source dataset
    processed/                      # Generated symlinks + labels
  weights/                          # Pretrained YOLO weights (gitignored)
  runs/                             # Training outputs (gitignored)
  models/                           # Exported TFLite (gitignored)
  results/                          # Evaluation CSVs (gitignored)
```

## Troubleshooting

**`[ERROR] Dataset path not found`** — Re-run conversion inside Docker:
```bash
docker compose run train python scripts/download_and_convert.py
```

**Broken symlinks** — Re-run `download_and_convert.py` in the same environment.

**CUDA OOM** — Reduce batch: `--batch 8` or `--batch 4`.

**Docker can't see GPU** — Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

**Resume training** — All scripts support `--resume`:
```bash
docker compose run train python scripts/train_darts_pose.py --resume runs/darts_pose/train/weights/last.pt
```

## What's Next

1. **Run YOLO scoring benchmark** — know actual offline accuracy.
2. **Test cloud Gemini** on DeepDarts val images — know cloud accuracy.
3. **Test Ollama** (llava:13b on your GPU server) — know LAN accuracy.
4. **Fix whatever is broken** — train better YOLO models if end-to-end <70%.
5. **Skip LoRA** — on-device VLM baseline is 0%. LoRA can't fix incompetence.

## License

See `../darts_vision/LICENSE` for details.
