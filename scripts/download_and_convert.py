#!/usr/bin/env python3
"""
Download and convert dart detection datasets for YOLO11 training.

Uses HuggingFace datasets library to download bhabha-kapil/Dartboard-Detection-Dataset,
then converts to YOLO11-compatible formats.

Usage:
    python scripts/download_and_convert.py [--skip-download]
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

PROCESSED_DIR = Path("data/processed")

CLASS_MAP = {
    0: "dart_tip",
    1: "cal_double_20",
    2: "cal_double_6",
    3: "cal_double_3",
    4: "cal_double_11",
}


def download_dataset():
    from datasets import load_dataset

    print("[DOWNLOAD] Downloading bhabha-kapil/Dartboard-Detection-Dataset...")
    print("[INFO] This is ~7.2 GB. First run will take 10-30 minutes.")

    ds = load_dataset("bhabha-kapil/Dartboard-Detection-Dataset")

    print(f"[INFO] Dataset loaded. Splits: {list(ds.keys())}")
    for split, data in ds.items():
        print(f"  {split}: {len(data)} images")

    return ds


def convert_to_yolo11_pose(ds):
    print("[CONVERT] Converting to YOLO11-pose format...")

    output_dir = PROCESSED_DIR / "yolo11_pose"
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    all_data = ds.get("train", ds.get("test", ds.values().__iter__().__next__()))
    total = len(all_data)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)

    for i in range(total):
        if i < train_end:
            split = "train"
        elif i < val_end:
            split = "val"
        else:
            split = "test"

        row = all_data[i]
        img = row["image"]
        labels = row.get("label", [])

        img_filename = f"dart_{i:05d}.jpg"
        img_path = output_dir / "images" / split / img_filename
        img.save(str(img_path))

        img_w, img_h = img.size

        dart_tips = []
        cal_points = []

        if isinstance(labels, dict):
            for class_id, bbox in zip(labels.get("class", []), labels.get("bbox", [])):
                if class_id == 0:
                    dart_tips.append(bbox)
                elif class_id in (1, 2, 3, 4):
                    cal_points.append((class_id, bbox))
        elif isinstance(labels, list):
            for item in labels:
                if isinstance(item, dict):
                    cls = item.get("class", item.get("class_id", 0))
                    bbox = item.get("bbox", item.get("box", []))
                    if cls == 0:
                        dart_tips.append(bbox)
                    elif cls in (1, 2, 3, 4):
                        cal_points.append((cls, bbox))

        cal_points.sort(key=lambda x: x[0])

        keypoints = []
        for _, bbox in cal_points[:4]:
            if len(bbox) >= 2:
                x1, y1 = bbox[0], bbox[1]
                x2 = bbox[2] if len(bbox) > 2 else bbox[0]
                y2 = bbox[3] if len(bbox) > 3 else bbox[1]
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                if cx > 1.5 or cy > 1.5:
                    cx /= img_w
                    cy /= img_h
                keypoints.extend([cx, cy, 2])

        for bbox in dart_tips[:3]:
            if len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            elif len(bbox) >= 2:
                x1, y1 = bbox[0], bbox[1]
                x2, y2 = bbox[0], bbox[1]
            else:
                continue
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            if cx > 1.5 or cy > 1.5:
                cx /= img_w
                cy /= img_h
            keypoints.extend([cx, cy, 2])

        num_kpts = 7
        while len(keypoints) // 3 < num_kpts:
            keypoints.extend([0.0, 0.0, 0.0])
        keypoints = keypoints[:num_kpts * 3]

        bbox_x = 0.5
        bbox_y = 0.5
        bbox_w = 0.9
        bbox_h = 0.9

        kpt_str = " ".join(f"{v:.6f}" for v in keypoints)
        yolo_line = f"0 {bbox_x:.6f} {bbox_y:.6f} {bbox_w:.6f} {bbox_h:.6f} {kpt_str}"

        label_path = output_dir / "labels" / split / f"dart_{i:05d}.txt"
        with open(label_path, "w") as f:
            f.write(yolo_line + "\n")

    print(f"[DONE] YOLO11-pose dataset at {output_dir}")


def convert_to_yolo11_detect(ds):
    print("[CONVERT] Converting to YOLO11-detect format...")

    output_dir = PROCESSED_DIR / "yolo11_detect"
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    all_data = ds.get("train", ds.get("test", ds.values().__iter__().__next__()))
    total = len(all_data)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)

    for i in range(total):
        if i < train_end:
            split = "train"
        elif i < val_end:
            split = "val"
        else:
            split = "test"

        row = all_data[i]
        img = row["image"]
        labels = row.get("label", [])

        img_filename = f"dart_{i:05d}.jpg"
        img_path = output_dir / "images" / split / img_filename
        img.save(str(img_path))

        img_w, img_h = img.size

        yolo_lines = []
        all_points = []

        if isinstance(labels, dict):
            for class_id, bbox in zip(labels.get("class", []), labels.get("bbox", [])):
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                else:
                    continue
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                bw, bh = abs(x2 - x1), abs(y2 - y1)

                if cx > 1.5 or cy > 1.5:
                    cx /= img_w
                    cy /= img_h
                    bw /= img_w
                    bh /= img_h

                if class_id == 0:
                    yolo_lines.append(f"1 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                all_points.append((cx, cy))

        elif isinstance(labels, list):
            for item in labels:
                if isinstance(item, dict):
                    cls = item.get("class", item.get("class_id", 0))
                    bbox = item.get("bbox", item.get("box", []))
                    if len(bbox) >= 4:
                        x1, y1, x2, y2 = bbox
                        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                        bw, bh = abs(x2 - x1), abs(y2 - y1)
                        if cx > 1.5:
                            cx /= img_w
                            cy /= img_h
                            bw /= img_w
                            bh /= img_h
                        if cls == 0:
                            yolo_lines.append(f"1 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                        all_points.append((cx, cy))

        if len(all_points) >= 2:
            xs = [p[0] for p in all_points]
            ys = [p[1] for p in all_points]
            min_x, max_x = min(xs) - 0.05, max(xs) + 0.05
            min_y, max_y = min(ys) - 0.05, max(ys) + 0.05
            bcx = (min_x + max_x) / 2
            bcy = (min_y + max_y) / 2
            bw = max_x - min_x
            bh = max_y - min_y
            yolo_lines.insert(0, f"0 {bcx:.6f} {bcy:.6f} {bw:.6f} {bh:.6f}")

        label_path = output_dir / "labels" / split / f"dart_{i:05d}.txt"
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines) + "\n")

    print(f"[DONE] YOLO11-detect dataset at {output_dir}")


def convert_to_board_calibration(ds):
    print("[CONVERT] Converting to board calibration format...")

    output_dir = PROCESSED_DIR / "board_calibration"
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    all_data = ds.get("train", ds.get("test", ds.values().__iter__().__next__()))
    total = len(all_data)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)

    skip_count = 0
    for i in range(total):
        if i < train_end:
            split = "train"
        elif i < val_end:
            split = "val"
        else:
            split = "test"

        row = all_data[i]
        img = row["image"]
        labels = row.get("label", [])

        img_filename = f"dart_{i:05d}.jpg"
        img_path = output_dir / "images" / split / img_filename
        img.save(str(img_path))

        img_w, img_h = img.size

        cal_points = []

        if isinstance(labels, dict):
            for class_id, bbox in zip(labels.get("class", []), labels.get("bbox", [])):
                if class_id in (1, 2, 3, 4):
                    if len(bbox) >= 4:
                        x1, y1, x2, y2 = bbox
                        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                        if cx > 1.5:
                            cx /= img_w
                            cy /= img_h
                        cal_points.append((class_id, cx, cy))
        elif isinstance(labels, list):
            for item in labels:
                if isinstance(item, dict):
                    cls = item.get("class", item.get("class_id", 0))
                    bbox = item.get("bbox", item.get("box", []))
                    if cls in (1, 2, 3, 4) and len(bbox) >= 4:
                        x1, y1, x2, y2 = bbox
                        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                        if cx > 1.5:
                            cx /= img_w
                            cy /= img_h
                        cal_points.append((cls, cx, cy))

        cal_points.sort(key=lambda x: x[0])

        if len(cal_points) < 4:
            skip_count += 1
            continue

        keypoints = []
        for _, cx, cy in cal_points[:4]:
            keypoints.extend([cx, cy, 2])

        kpt_str = " ".join(f"{v:.6f}" for v in keypoints)
        yolo_line = f"0 0.5 0.5 0.9 0.9 {kpt_str}"

        label_path = output_dir / "labels" / split / f"dart_{i:05d}.txt"
        with open(label_path, "w") as f:
            f.write(yolo_line + "\n")

    print(f"[DONE] Board calibration dataset at {output_dir}")
    if skip_count:
        print(f"[INFO] Skipped {skip_count} images with <4 calibration points")


def write_dataset_yaml():
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)
    root = os.getcwd()

    configs = {
        "dataset_pose.yaml": f"""# YOLO11-pose: dart tips + calibration keypoints
path: {root}/data/processed/yolo11_pose
train: images/train
val: images/val
test: images/test

# 0: double-20 corner (top) — flips to itself
# 1: double-6 corner (right) — flips to double-11 (left)
# 2: double-3 corner (bottom) — flips to itself
# 3: double-11 corner (left) — flips to double-6 (right)
# 4-6: dart tips — no symmetric mapping

kpt_shape: [7, 3]
flip_idx: [0, 3, 2, 1, 4, 5, 6]
names:
  0: dartboard
""",
        "dataset_detect.yaml": f"""# YOLO11-detect: dartboard + dart tip bounding boxes
path: {root}/data/processed/yolo11_detect
train: images/train
val: images/val
test: images/test

names:
  0: dartboard
  1: dart_tip
""",
        "dataset_calibration.yaml": f"""# Board calibration: 4 calibration keypoints only
path: {root}/data/processed/board_calibration
train: images/train
val: images/val
test: images/test

# 0: double-20 corner (top) — flips to itself
# 1: double-6 corner (right) — flips to double-11 (left)
# 2: double-3 corner (bottom) — flips to itself
# 3: double-11 corner (left) — flips to double-6 (right)

kpt_shape: [4, 3]
flip_idx: [0, 3, 2, 1]
names:
  0: dartboard
""",
    }

    for filename, content in configs.items():
        with open(configs_dir / filename, "w") as f:
            f.write(content)

    print(f"[DONE] Dataset configs written to {configs_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Download and convert dart detection datasets")
    parser.add_argument("--skip-download", action="store_true", help="Skip dataset download (use cached)")
    args = parser.parse_args()

    if args.skip_download:
        print("[SKIP] Loading from cache...")
        from datasets import load_dataset
        ds = load_dataset("bhabha-kapil/Dartboard-Detection-Dataset")
    else:
        ds = download_dataset()

    convert_to_yolo11_pose(ds)
    convert_to_yolo11_detect(ds)
    convert_to_board_calibration(ds)
    write_dataset_yaml()

    print("\n[DONE] All datasets converted.")
    print("Next steps:")
    print("  1. docker compose build && docker compose run train bash")
    print("  2. python scripts/train_pose.py          # YOLO11n-pose")
    print("  3. python scripts/train_detect.py        # YOLO11n-detect")
    print("  4. python scripts/train_calibration.py   # Board calibration")


if __name__ == "__main__":
    main()