#!/usr/bin/env python3
"""
End-to-end YOLO scoring benchmark (for GPU server Docker).

Measures the full scoring pipeline accuracy: YOLO keypoint detection
→ homography → dart score classification. Not just detection mAP.

Uses the same inference pipeline as the Kotlin app:
  1. board_calibration model → 4 keypoints → homography
  2. darts_pose model → 7 keypoints → dart tip positions
  3. homography → mm coords → score classification

This script can use either:
  - PyTorch .pt models (for GPU server with ultralytics)
  - TFLite models (same as Android)

Metrics (same as VLM benchmark for fair comparison):
    DartCnt:   correct number of darts detected
    Segment:   correct segment (1-20)
    Ring:      correct ring (single/triple/double/bull/bullseye)
    Full:      exact score match (segment + ring)

Usage:
    # PyTorch models (GPU)
    python scripts/evaluate_yolo_scoring.py --cal-model runs/board_calibration/train/weights/best.pt --pose-model runs/darts_pose/train/weights/best.pt

    # TFLite models (CPU or GPU delegate)
    python scripts/evaluate_yolo_scoring.py --cal-model models/board_calibration_float32.tflite --pose-model models/darts_pose_float32.tflite --backend tflite

    # Specific split
    python scripts/evaluate_yolo_scoring.py --split test --save-csv results/yolo_scoring.csv
"""

import argparse
import csv
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from dart_board import keypoints_to_scores, DartScore

PROCESSED_DIR = Path("data/processed/yolo11_pose")
INPUT_SIZE = 640
CONF_THRESH = 0.5
KPT_CONF_THRESH = 0.3
NUM_GRID_CELLS = 8400


def load_ground_truth(label_path: Path, img_width: int = 800, img_height: int = 800) -> List[DartScore]:
    text = label_path.read_text().strip()
    if not text:
        return []
    parts = text.split()
    if len(parts) < 5 + 7 * 3:
        return []
    cal_keypoints = []
    for i in range(4):
        x = float(parts[5 + i * 3])
        y = float(parts[5 + i * 3 + 1])
        v = float(parts[5 + i * 3 + 2])
        cal_keypoints.append((x, y) if v > 0.5 else None)
    dart_keypoints = []
    for i in range(3):
        idx = 5 + (4 + i) * 3
        if idx + 2 < len(parts):
            x = float(parts[idx])
            y = float(parts[idx + 1])
            v = float(parts[idx + 2])
            dart_keypoints.append((x, y) if v > 0.5 else None)
    return keypoints_to_scores(cal_keypoints, dart_keypoints, img_width, img_height)


class UltralyticsDetector:
    """YOLO detector using PyTorch/Ultralytics (GPU server)."""

    def __init__(self, model_path: str):
        from ultralytics import YOLO
        self.model = YOLO(model_path)

    def detect(self, image: Image.Image) -> Optional[dict]:
        results = self.model(image, verbose=False)
        if not results or len(results) == 0:
            return None
        r = results[0]
        if r.keypoints is None or len(r.keypoints.data) == 0:
            return None
        kpts = r.keypoints.data[0].cpu().numpy()
        if len(kpts) < 7:
            return None
        keypoint_list = []
        for i in range(len(kpts)):
            x, y, conf = float(kpts[i][0]), float(kpts[i][1]), float(kpts[i][2])
            keypoint_list.append((x / INPUT_SIZE, y / INPUT_SIZE, conf))
        return {"keypoints": keypoint_list}

    def release(self):
        del self.model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class TfliteDetector:
    """YOLO detector using TFLite (same as Android app)."""

    def __init__(self, model_path: str):
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            try:
                import tensorflow.lite as tflite
            except ImportError:
                raise ImportError("TensorFlow Lite not found. Install: pip install tensorflow tflite-runtime")
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        self.input_idx = input_details[0]["index"]
        self.output_idx = output_details[0]["index"]
        ndim = output_details[0]["shape"][1]
        self.num_keypoints = (ndim - 5) // 3

    def detect(self, image: Image.Image) -> Optional[dict]:
        inp = self._preprocess(image)
        self.interpreter.set_tensor(self.input_idx, inp)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_idx)
        return self._parse_output(out[0])

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        img = image.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE))
        arr = np.array(img).astype(np.float32) / 255.0
        return np.expand_dims(arr, axis=0)

    def _parse_output(self, channels: np.ndarray) -> Optional[dict]:
        nk = self.num_keypoints
        best_conf = 0.0
        best_kpts = None
        for i in range(NUM_GRID_CELLS):
            obj_conf = channels[4, i]
            if obj_conf < CONF_THRESH:
                continue
            kpts = []
            for k in range(nk):
                kx = channels[5 + k * 3, i]
                ky = channels[5 + k * 3 + 1, i]
                kv = channels[5 + k * 3 + 2, i]
                kpts.append((float(kx), float(ky), float(kv)))
            if obj_conf > best_conf:
                best_conf = obj_conf
                best_kpts = kpts
        if best_kpts is None:
            return None
        # Normalize to [0,1]
        norm_kpts = [(x / INPUT_SIZE, y / INPUT_SIZE, v) for x, y, v in best_kpts]
        return {"keypoints": norm_kpts}

    def release(self):
        del self.interpreter


def evaluate_image(image: Image.Image, label_path: Path,
                   cal_detector, pose_detector) -> dict:
    gt_scores = load_ground_truth(label_path, image.width, image.height)

    start = time.perf_counter()

    cal_result = cal_detector.detect(image)
    if cal_result is None or len(cal_result.get("keypoints", [])) < 4:
        inf_time = (time.perf_counter() - start) * 1000
        return {
            "image": label_path.stem,
            "gt_count": len(gt_scores),
            "pred_count": 0,
            "dart_count_correct": 0,
            "segment_correct": 0,
            "ring_correct": 0,
            "full_correct": 0,
            "total_gt": len(gt_scores),
            "inference_ms": inf_time,
            "detected": False,
        }

    cal_keypoints = []
    for i in range(4):
        kx, ky, kv = cal_result["keypoints"][i]
        cal_keypoints.append((kx, ky) if kv >= KPT_CONF_THRESH else None)

    pose_result = pose_detector.detect(image)
    if pose_result is None:
        inf_time = (time.perf_counter() - start) * 1000
        return {
            "image": label_path.stem,
            "gt_count": len(gt_scores),
            "pred_count": 0,
            "dart_count_correct": 0,
            "segment_correct": 0,
            "ring_correct": 0,
            "full_correct": 0,
            "total_gt": len(gt_scores),
            "inference_ms": inf_time,
            "detected": True,
        }

    dart_tips = []
    n_kpts = len(pose_result.get("keypoints", []))
    for i in range(min(3, n_kpts - 4)):
        kx, ky, kv = pose_result["keypoints"][4 + i]
        dart_tips.append((kx, ky) if kv >= KPT_CONF_THRESH else None)

    inf_time = (time.perf_counter() - start) * 1000

    pred_scores = keypoints_to_scores(cal_keypoints, dart_tips, image.width, image.height)

    gt_count = len(gt_scores)
    pred_count = len(pred_scores)
    dart_count_correct = int(gt_count == pred_count)

    segment_correct = 0
    ring_correct = 0
    full_correct = 0

    if gt_count > 0 and pred_count > 0:
        costs = np.zeros((gt_count, pred_count))
        for i, gt in enumerate(gt_scores):
            for j, pred in enumerate(pred_scores):
                ad = abs(((gt.angle_deg - pred.angle_deg) + 180) % 360 - 180)
                rd = abs(gt.radius_mm - pred.radius_mm)
                costs[i, j] = ad * 2.0 + rd

        matched_gt = set()
        matched_pred = set()
        flat = [(costs[i, j], i, j) for i in range(gt_count) for j in range(pred_count)]
        flat.sort()

        for cost, i, j in flat:
            if i in matched_gt or j in matched_pred:
                continue
            if cost > 55.0:
                break
            matched_gt.add(i)
            matched_pred.add(j)
            g = gt_scores[i]
            p = pred_scores[j]
            if g.segment == p.segment:
                segment_correct += 1
            if g.ring == p.ring:
                ring_correct += 1
            if g.segment == p.segment and g.ring == p.ring:
                full_correct += 1

    return {
        "image": label_path.stem,
        "gt_count": gt_count,
        "pred_count": pred_count,
        "dart_count_correct": dart_count_correct,
        "segment_correct": segment_correct,
        "ring_correct": ring_correct,
        "full_correct": full_correct,
        "total_gt": gt_count,
        "inference_ms": inf_time,
        "detected": True,
    }


def run_benchmark(cal_model: str, pose_model: str, split: str, save_csv: Optional[str], backend: str):
    if backend == "tflite":
        cal_detector = TfliteDetector(cal_model)
        pose_detector = TfliteDetector(pose_model)
    else:
        cal_detector = UltralyticsDetector(cal_model)
        pose_detector = UltralyticsDetector(pose_model)

    image_dir = PROCESSED_DIR / "images" / split
    label_dir = PROCESSED_DIR / "labels" / split

    if not image_dir.exists() or not label_dir.exists():
        print(f"[ERROR] Dataset not found. Run: python scripts/download_and_convert.py")
        return

    label_files = sorted(label_dir.glob("*.txt"))
    if not label_files:
        print(f"[ERROR] No labels in {label_dir}")
        return

    print(f"[EVAL] Benchmarking {len(label_files)} {split} images ({backend})")
    print(f"[CAL]  {cal_model}")
    print(f"[POSE] {pose_model}")

    results = []
    total_time = 0.0

    for label_path in tqdm(label_files, desc=f"Evaluating {split}"):
        exts = [".JPG", ".jpg", ".png"]
        image_path = None
        for ext in exts:
            p = image_dir / (label_path.stem + ext)
            if p.exists():
                image_path = p
                break
        if image_path is None:
            continue

        image = Image.open(image_path)
        res = evaluate_image(image, label_path, cal_detector, pose_detector)
        results.append(res)
        total_time += res["inference_ms"]

    cal_detector.release()
    pose_detector.release()

    total_images = len(results)
    total_gt = sum(r["total_gt"] for r in results)

    dca = sum(r["dart_count_correct"] for r in results) / max(total_images, 1) * 100
    sa = sum(r["segment_correct"] for r in results) / max(total_gt, 1) * 100
    ra = sum(r["ring_correct"] for r in results) / max(total_gt, 1) * 100
    fa = sum(r["full_correct"] for r in results) / max(total_gt, 1) * 100
    avg_ms = total_time / max(total_images, 1)
    tput = 1000.0 / avg_ms if avg_ms > 0 else 0

    model_name = Path(cal_model).stem
    pose_name = Path(pose_model).stem

    print("\n" + "=" * 80)
    print(f"YOLO END-TO-END SCORING BENCHMARK ({backend}, {split})")
    print("=" * 80)
    print(f"{'Model':<30} {'DartCnt':>8} {'Segment':>8} {'Ring':>8} {'Full':>8} {'AvgMs':>8} {'Img/s':>8}")
    print("-" * 80)
    combined = f"{model_name}+{pose_name}"
    print(f"{combined:<30} {dca:>7.1f}% {sa:>7.1f}% {ra:>7.1f}% {fa:>7.1f}% {avg_ms:>7.1f}ms {tput:>7.1f}/s")
    print("=" * 80)

    if save_csv:
        p = Path(save_csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n[SAVE] Per-image results -> {p}")

    return {
        "model": combined,
        "backend": backend,
        "split": split,
        "dart_count_accuracy": dca,
        "segment_accuracy": sa,
        "ring_accuracy": ra,
        "full_score_accuracy": fa,
        "avg_inference_ms": avg_ms,
        "throughput": tput,
        "total_images": total_images,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO end-to-end scoring accuracy")
    parser.add_argument("--cal-model", type=str, required=True, help="Calibration model (.pt or .tflite)")
    parser.add_argument("--pose-model", type=str, required=True, help="Pose model (.pt or .tflite)")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "pytorch", "tflite"],
                        help="Inference backend")
    parser.add_argument("--save-csv", type=str, default=None, help="Path to save per-image CSV")
    args = parser.parse_args()

    for p in (args.cal_model, args.pose_model):
        if not Path(p).exists():
            print(f"[ERROR] Model not found: {p}")
            return

    backend = args.backend
    if backend == "auto":
        backend = "tflite" if args.cal_model.endswith(".tflite") else "pytorch"

    run_benchmark(args.cal_model, args.pose_model, args.split, args.save_csv, backend)


if __name__ == "__main__":
    main()
