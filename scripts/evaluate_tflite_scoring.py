#!/usr/bin/env python3
"""
End-to-end YOLO TFLite scoring benchmark.

Runs the actual TFLite models used in the Android app against DeepDarts
ground truth to measure real-world scoring accuracy.

Uses the same inference pipeline as the Kotlin app:
  1. board_calibration_float32.tflite → 4 keypoints → homography
  2. darts_pose_float32.tflite → 7 keypoints → dart tip positions
  3. homography → mm coords → score classification

Metrics (same as VLM benchmark for fair comparison):
    DartCnt:   correct number of darts detected
    Segment:   correct segment (1-20)
    Ring:      correct ring (single/triple/double/bull/bullseye)
    Full:      exact score match (segment + ring)

Usage:
    python scripts/evaluate_tflite_scoring.py
    python scripts/evaluate_tflite_scoring.py --split test --save-csv results/tflite_scoring.csv
    python scripts/evaluate_tflite_scoring.py --cal-model models/board_calibration_float32.tflite --pose-model models/darts_pose_float32.tflite
"""

import argparse
import csv
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        raise ImportError("TensorFlow Lite not found. Install: pip install tensorflow tflite-runtime")

from dart_board import classify_dart, keypoints_to_scores, DartScore

CAL_MODEL_DEFAULT = "../darts_vision/app/src/main/assets/models/board_calibration_float32.tflite"
POSE_MODEL_DEFAULT = "../darts_vision/app/src/main/assets/models/darts_pose_float32.tflite"

INPUT_SIZE = 640
CONF_THRESH = 0.5
NMS_IOU_THRESH = 0.45
KPT_CONF_THRESH = 0.3
NUM_GRID_CELLS = 8400

DEEPDARTS_CAL_MM = {
    "D20": (0.0, -170.0),
    "D6": (170.0, 0.0),
    "D3": (0.0, 170.0),
    "D11": (-170.0, 0.0),
}


class TflitePoseDetector:
    """YOLO11n-pose TFLite inference engine. Same logic as YoloPoseDetector.kt."""

    def __init__(self, model_path: str):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        self.input_idx = input_details[0]["index"]
        self.output_idx = output_details[0]["index"]
        self.in_shape = input_details[0]["shape"]
        self.num_keypoints = (output_details[0]["shape"][1] - 5) // 3

    def detect(self, image: Image.Image) -> Optional[dict]:
        input_tensor = self._preprocess(image)
        self.interpreter.set_tensor(self.input_idx, input_tensor)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_idx)
        return self._parse_output(output[0])

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        img = image.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE))
        arr = np.array(img).astype(np.float32) / 255.0
        return np.expand_dims(arr, axis=0)

    def _parse_output(self, channels: np.ndarray) -> Optional[dict]:
        num_kpts = self.num_keypoints
        output_channels = 5 + num_kpts * 3
        candidates = []

        for i in range(NUM_GRID_CELLS):
            obj_conf = channels[4, i]
            if obj_conf < CONF_THRESH:
                continue
            cx, cy, w, h = channels[0:4, i]
            keypoints = []
            for k in range(num_kpts):
                kx = channels[5 + k * 3, i]
                ky = channels[5 + k * 3 + 1, i]
                kv = channels[5 + k * 3 + 2, i]
                keypoints.append((kx, ky, kv))
            candidates.append((obj_conf, cx, cy, w, h, keypoints))

        if not candidates:
            return None

        # NMS (simplified — for single board, just pick best)
        candidates.sort(key=lambda x: x[0], reverse=True)
        best = candidates[0]

        obj_conf, cx, cy, w, h, keypoints = best
        return {
            "confidence": float(obj_conf),
            "bbox": (float(cx), float(cy), float(w), float(h)),
            "keypoints": [(float(kx), float(ky), float(kv)) for kx, ky, kv in keypoints],
        }

    def release(self):
        del self.interpreter


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


def evaluate_image(image: Image.Image, label_path: Path,
                   cal_detector: TflitePoseDetector, pose_detector: TflitePoseDetector) -> dict:
    gt_scores = load_ground_truth(label_path)

    start = time.perf_counter()

    # Step 1: Board calibration
    cal_result = cal_detector.detect(image)
    if cal_result is None:
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

    # Step 2: Darts pose
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
    for i in range(3):
        kx, ky, kv = pose_result["keypoints"][4 + i]
        dart_tips.append((kx, ky) if kv >= KPT_CONF_THRESH else None)

    inf_time = (time.perf_counter() - start) * 1000

    pred_scores = keypoints_to_scores(cal_keypoints, dart_tips, image.width, image.height)

    # Match predictions to ground truth
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
                angle_diff = abs(((gt.angle_deg - pred.angle_deg) + 180) % 360 - 180)
                radius_diff = abs(gt.radius_mm - pred.radius_mm)
                costs[i, j] = angle_diff * 2.0 + radius_diff

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

            gt = gt_scores[i]
            pred = pred_scores[j]
            if gt.segment == pred.segment:
                segment_correct += 1
            if gt.ring == pred.ring:
                ring_correct += 1
            if gt.segment == pred.segment and gt.ring == pred.ring:
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


def run_benchmark(cal_model: str, pose_model: str, split: str = "val", save_csv: Optional[str] = None):
    print(f"[LOAD] Calibration model: {cal_model}")
    print(f"[LOAD] Pose model: {pose_model}")

    cal_detector = TflitePoseDetector(cal_model)
    pose_detector = TflitePoseDetector(pose_model)

    data_dir = Path("data/processed/yolo11_pose")
    image_dir = data_dir / "images" / split
    label_dir = data_dir / "labels" / split

    if not image_dir.exists() or not label_dir.exists():
        print(f"[ERROR] Dataset not found: {image_dir} or {label_dir}")
        print("Run: python scripts/download_and_convert.py")
        return

    label_files = sorted(label_dir.glob("*.txt"))
    if not label_files:
        print(f"[ERROR] No label files in {label_dir}")
        return

    print(f"[EVAL] Benchmarking {len(label_files)} {split} images")

    results = []
    total_time = 0.0

    for label_path in tqdm(label_files, desc=f"Evaluating {split}"):
        ext = ".JPG"
        image_path = image_dir / (label_path.stem + ext)
        if not image_path.exists():
            image_path = image_dir / (label_path.stem + ".jpg")
        if not image_path.exists():
            image_path = image_dir / (label_path.stem + ".png")
        if not image_path.exists():
            continue

        image = Image.open(image_path)
        res = evaluate_image(image, label_path, cal_detector, pose_detector)
        results.append(res)
        total_time += res["inference_ms"]

    cal_detector.release()
    pose_detector.release()

    total_images = len(results)
    total_gt_darts = sum(r["total_gt"] for r in results)

    dart_count_acc = sum(r["dart_count_correct"] for r in results) / max(total_images, 1) * 100
    segment_acc = sum(r["segment_correct"] for r in results) / max(total_gt_darts, 1) * 100
    ring_acc = sum(r["ring_correct"] for r in results) / max(total_gt_darts, 1) * 100
    full_acc = sum(r["full_correct"] for r in results) / max(total_gt_darts, 1) * 100
    avg_time = total_time / max(total_images, 1)
    throughput = 1000.0 / avg_time if avg_time > 0 else 0

    print("\n" + "=" * 80)
    print(f"TFLITE END-TO-END SCORING BENCHMARK ({split})")
    print("=" * 80)
    print(f"{'Model':<30} {'DartCnt':>8} {'Segment':>8} {'Ring':>8} {'Full':>8} {'AvgMs':>8} {'Img/s':>8}")
    print("-" * 80)
    print(f"{'YOLO_TFLite':<30} {dart_count_acc:>7.1f}% {segment_acc:>7.1f}% {ring_acc:>7.1f}% {full_acc:>7.1f}% {avg_time:>7.1f}ms {throughput:>7.1f}/s")
    print("=" * 80)

    if save_csv:
        csv_path = Path(save_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n[SAVE] Per-image results -> {csv_path}")

    return {
        "model": "YOLO_TFLite",
        "split": split,
        "dart_count_accuracy": dart_count_acc,
        "segment_accuracy": segment_acc,
        "ring_accuracy": ring_acc,
        "full_score_accuracy": full_acc,
        "avg_inference_ms": avg_time,
        "throughput": throughput,
        "total_images": total_images,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO TFLite end-to-end scoring accuracy")
    parser.add_argument("--cal-model", type=str, default=CAL_MODEL_DEFAULT, help="Board calibration TFLite model")
    parser.add_argument("--pose-model", type=str, default=POSE_MODEL_DEFAULT, help="Darts pose TFLite model")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--save-csv", type=str, default=None, help="Path to save per-image CSV")
    args = parser.parse_args()

    run_benchmark(args.cal_model, args.pose_model, args.split, args.save_csv)


if __name__ == "__main__":
    main()
