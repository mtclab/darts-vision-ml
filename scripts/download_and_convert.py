#!/usr/bin/env python3
"""
Convert DeepDarts dataset (labels.pkl + cropped_images) to YOLO11 formats.

Uses the real annotated DeepDarts dataset from McNally et al.
Images: data/raw/deep-darts/dataset/cropped_images/<folder>/<name>
Labels: data/raw/deep-darts/dataset/labels.pkl

Produces three YOLO11 datasets:
  1. yolo11_pose     — dartboard + 7 keypoints (4 cal + 3 dart tips)
  2. yolo11_detect   — dartboard bbox + dart tip bboxes
  3. board_calibration — dartboard + 4 calibration keypoints

Usage:
    python scripts/download_and_convert.py
    python scripts/download_and_convert.py --split d1       # face-on only
    python scripts/download_and_convert.py --split d2       # multi-angle only
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

RAW_DIR = Path("data/raw/deep-darts/dataset")
PROCESSED_DIR = Path("data/processed")

D1_VAL = ["d1_02_06_2020", "d1_02_16_2020", "d1_02_22_2020"]
D1_TEST = [
    "d1_03_03_2020", "d1_03_19_2020", "d1_03_23_2020",
    "d1_03_27_2020", "d1_03_28_2020", "d1_03_30_2020", "d1_03_31_2020",
]
D2_VAL = ["d2_02_03_2021", "d2_02_05_2021"]
D2_TEST = ["d2_03_03_2020", "d2_02_10_2021", "d2_02_03_2021_2"]

BBOX_SIZE = 0.025


def load_and_split(split_filter):
    df = pd.read_pickle(RAW_DIR / "labels.pkl")

    if split_filter:
        df = df[df.img_folder.str.startswith(split_filter)]

    val_folders = D1_VAL if split_filter == "d1" else D2_VAL if split_filter == "d2" else D1_VAL + D2_VAL
    test_folders = D1_TEST if split_filter == "d1" else D2_TEST if split_filter == "d2" else D1_TEST + D2_TEST

    splits = {}
    splits["val"] = df[np.isin(df.img_folder, val_folders)]
    splits["test"] = df[np.isin(df.img_folder, test_folders)]
    splits["train"] = df[np.logical_not(np.isin(df.img_folder, val_folders + test_folders))]
    return splits


def get_split(row_folder, val_folders, test_folders):
    if row_folder in test_folders:
        return "test"
    if row_folder in val_folders:
        return "val"
    return "train"


def link_image(row, output_dir, split, idx):
    src = RAW_DIR / "cropped_images" / row["img_folder"] / row["img_name"]
    ext = Path(row["img_name"]).suffix
    dst_name = f"dart_{idx:05d}{ext}"
    dst = output_dir / "images" / split / dst_name
    if not dst.exists():
        os.symlink(str(src.resolve()), str(dst))
    return dst_name


def xy_to_keypoints(xy_list):
    cal_points = []
    for i in range(4):
        if i < len(xy_list):
            cal_points.append((xy_list[i][0], xy_list[i][1]))
        else:
            cal_points.append(None)

    dart_tips = []
    for i in range(4, len(xy_list)):
        dart_tips.append((xy_list[i][0], xy_list[i][1]))
    dart_tips = dart_tips[:3]

    return cal_points, dart_tips


def compute_board_bbox(cal_points, dart_tips, margin=0.05):
    all_pts = [p for p in cal_points if p is not None] + [p for p in dart_tips if p is not None]
    if not all_pts:
        return 0.5, 0.5, 0.9, 0.9
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    w = max(xs) - min(xs) + margin
    h = max(ys) - min(ys) + margin
    return cx, cy, w, h


def convert_to_yolo11_pose(splits, val_folders, test_folders):
    print("[CONVERT] YOLO11-pose format...")

    output_dir = PROCESSED_DIR / "yolo11_pose"
    for s in ["train", "val", "test"]:
        (output_dir / "images" / s).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / s).mkdir(parents=True, exist_ok=True)

    idx = 0
    skipped = 0
    for split_name in ["train", "val", "test"]:
        data = splits[split_name]
        for _, row in tqdm(data.iterrows(), total=len(data), desc=f"pose/{split_name}"):
            actual_split = get_split(row["img_folder"], val_folders, test_folders)
            cal_points, dart_tips = xy_to_keypoints(row["xy"])

            if all(p is None for p in cal_points):
                skipped += 1
                continue

            dst_name = link_image(row, output_dir, actual_split, idx)

            keypoints = []
            for p in cal_points:
                if p is not None:
                    keypoints.extend([p[0], p[1], 2])
                else:
                    keypoints.extend([0.0, 0.0, 0.0])
            for i in range(3):
                if i < len(dart_tips) and dart_tips[i] is not None:
                    keypoints.extend([dart_tips[i][0], dart_tips[i][1], 2])
                else:
                    keypoints.extend([0.0, 0.0, 0.0])

            cx, cy, w, h = compute_board_bbox(cal_points, dart_tips)
            kpts = " ".join(f"{v:.6f}" for v in keypoints)
            label = f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {kpts}"

            label_path = output_dir / "labels" / actual_split / f"dart_{idx:05d}.txt"
            label_path.write_text(label + "\n")
            idx += 1

    print(f"[DONE] YOLO11-pose: {idx} images, {skipped} skipped -> {output_dir}")


def convert_to_yolo11_detect(splits, val_folders, test_folders):
    print("[CONVERT] YOLO11-detect format...")

    output_dir = PROCESSED_DIR / "yolo11_detect"
    for s in ["train", "val", "test"]:
        (output_dir / "images" / s).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / s).mkdir(parents=True, exist_ok=True)

    idx = 0
    for split_name in ["train", "val", "test"]:
        data = splits[split_name]
        for _, row in tqdm(data.iterrows(), total=len(data), desc=f"detect/{split_name}"):
            actual_split = get_split(row["img_folder"], val_folders, test_folders)
            cal_points, dart_tips = xy_to_keypoints(row["xy"])

            dst_name = link_image(row, output_dir, actual_split, idx)

            lines = []

            cx, cy, w, h = compute_board_bbox(cal_points, dart_tips)
            lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            for tip in dart_tips:
                if tip is not None:
                    tx, ty = tip
                    tw = BBOX_SIZE
                    th = BBOX_SIZE
                    lines.append(f"1 {tx:.6f} {ty:.6f} {tw:.6f} {th:.6f}")

            label_path = output_dir / "labels" / actual_split / f"dart_{idx:05d}.txt"
            label_path.write_text("\n".join(lines) + "\n")
            idx += 1

    print(f"[DONE] YOLO11-detect: {idx} images -> {output_dir}")


def convert_to_board_calibration(splits, val_folders, test_folders):
    print("[CONVERT] Board calibration format...")

    output_dir = PROCESSED_DIR / "board_calibration"
    for s in ["train", "val", "test"]:
        (output_dir / "images" / s).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / s).mkdir(parents=True, exist_ok=True)

    idx = 0
    skipped = 0
    for split_name in ["train", "val", "test"]:
        data = splits[split_name]
        for _, row in tqdm(data.iterrows(), total=len(data), desc=f"cal/{split_name}"):
            actual_split = get_split(row["img_folder"], val_folders, test_folders)
            cal_points, _ = xy_to_keypoints(row["xy"])

            if any(p is None for p in cal_points):
                skipped += 1
                continue

            dst_name = link_image(row, output_dir, actual_split, idx)

            keypoints = []
            for p in cal_points:
                keypoints.extend([p[0], p[1], 2])

            kpts = " ".join(f"{v:.6f}" for v in keypoints)
            cx, cy, w, h = compute_board_bbox(cal_points, [])
            label = f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {kpts}"

            label_path = output_dir / "labels" / actual_split / f"dart_{idx:05d}.txt"
            label_path.write_text(label + "\n")
            idx += 1

    print(f"[DONE] Board calibration: {idx} images, {skipped} skipped -> {output_dir}")


def write_dataset_yaml():
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)
    root = Path.cwd()

    configs = {
        "dataset_pose.yaml": f"""path: {root}/data/processed/yolo11_pose
train: images/train
val: images/val
test: images/test

kpt_shape: [7, 3]
flip_idx: [0, 3, 2, 1, 4, 5, 6]

names:
  0: dartboard
""",
        "dataset_detect.yaml": f"""path: {root}/data/processed/yolo11_detect
train: images/train
val: images/val
test: images/test

names:
  0: dartboard
  1: dart_tip
""",
        "dataset_calibration.yaml": f"""path: {root}/data/processed/board_calibration
train: images/train
val: images/val
test: images/test

kpt_shape: [4, 3]
flip_idx: [0, 3, 2, 1]

names:
  0: dartboard
""",
    }

    for filename, content in configs.items():
        (configs_dir / filename).write_text(content)

    print(f"[DONE] Dataset configs -> {configs_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Convert DeepDarts to YOLO11 formats")
    parser.add_argument("--split", choices=["d1", "d2", None], default=None, help="Dataset split (default: both)")
    args = parser.parse_args()

    pkl_path = RAW_DIR / "labels.pkl"
    if not pkl_path.exists():
        print(f"[ERROR] {pkl_path} not found.")
        print("Download the DeepDarts dataset first:")
        print("  1. Get cropped_images.zip and labels_pkl.zip from https://ieee-dataport.org/open-access/deepdarts-dataset")
        print("  2. Extract to data/raw/deep-darts/dataset/")
        print("     unzip cropped_images.zip -> data/raw/deep-darts/dataset/cropped_images/")
        print("     unzip labels_pkl.zip     -> data/raw/deep-darts/dataset/labels.pkl")
        return

    img_dir = RAW_DIR / "cropped_images"
    if not img_dir.exists() or not any(img_dir.iterdir()):
        print(f"[ERROR] {img_dir} not found or empty.")
        print("Extract cropped_images.zip to data/raw/deep-darts/dataset/cropped_images/")
        return

    print(f"[INFO] Loading labels.pkl (split={args.split or 'all'})...")
    splits = load_and_split(args.split)
    total = sum(len(s) for s in splits.values())
    print(f"[INFO] {total} images: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    if args.split == "d1":
        val_folders, test_folders = D1_VAL, D1_TEST
    elif args.split == "d2":
        val_folders, test_folders = D2_VAL, D2_TEST
    else:
        val_folders, test_folders = D1_VAL + D2_VAL, D1_TEST + D2_TEST

    convert_to_yolo11_pose(splits, val_folders, test_folders)
    convert_to_yolo11_detect(splits, val_folders, test_folders)
    convert_to_board_calibration(splits, val_folders, test_folders)
    write_dataset_yaml()

    print("\n[DONE] All datasets converted.")
    print("Next: python scripts/train_pose.py")


if __name__ == "__main__":
    main()