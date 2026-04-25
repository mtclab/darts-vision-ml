"""
Convert DeepDarts labels.pkl to YOLO pose format.

YOLO pose format per image:
    <class> <x> <y> <w> <h> <px1> <py1> <pv1> <px2> <py2> <pv2> ...

We represent the board + darts as a single pose instance with 7 keypoints:
    0: top-left calibration corner
    1: top-right calibration corner
    2: bottom-left calibration corner
    3: bottom-right calibration corner
    4: dart 1
    5: dart 2
    6: dart 3

Classes: 0=board (only one class for the pose object)

Usage:
    python src/convert_pose.py \
        --labels data/raw/deep-darts/dataset/labels.pkl \
        --output data/processed/yolo_pose_darts
"""

import argparse
import os
import pickle
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split


NUM_KPTS = 7


def load_labels(pkl_path: str):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


def xy_list_to_array(xy_list: list) -> np.ndarray:
    """Convert list of [x, y] points to (NUM_KPTS, 3) array with visibility."""
    arr = np.zeros((NUM_KPTS, 3), dtype=np.float32)
    n = min(len(xy_list), NUM_KPTS)
    arr[:n, :2] = np.array(xy_list[:n])
    arr[:n, 2] = 2
    return arr


def get_bbox_from_keypoints(kpts: np.ndarray) -> Tuple[float, float, float, float]:
    """Derive tight bbox from visible keypoints (normalized)."""
    vis = kpts[kpts[:, 2] > 0][:, :2]
    if len(vis) == 0:
        return 0.5, 0.5, 1.0, 1.0
    x1, y1 = vis.min(axis=0)
    x2, y2 = vis.max(axis=0)
    pad = 0.05
    x1 = max(0.0, x1 - pad)
    y1 = max(0.0, y1 - pad)
    x2 = min(1.0, x2 + pad)
    y2 = min(1.0, y2 + pad)
    xc = (x1 + x2) / 2.0
    yc = (y1 + y2) / 2.0
    return xc, yc, x2 - x1, y2 - y1


def save_pose_label(img_name: str, keypoints: np.ndarray, label_dir: Path):
    """Save one image's pose as YOLO .txt file."""
    xc, yc, w, h = get_bbox_from_keypoints(keypoints)
    parts = [0, xc, yc, w, h]
    for i in range(NUM_KPTS):
        x, y, v = keypoints[i]
        parts.extend([x, y, int(v)])
    line = (
        " ".join(
            f"{p:.6f}" if isinstance(p, float) else str(p) for p in parts
        )
        + "\n"
    )
    out_path = label_dir / f"{Path(img_name).stem}.txt"
    with open(out_path, "w") as f:
        f.write(line)


def write_yaml(output_root: Path):
    """Write pose dataset YAML for Ultralytics."""
    yaml_path = output_root / "pose.yaml"
    content = f"""# YOLOv8 Pose Dataset Configuration for Darts-Vision-ML
# Task: pose
# kpt_shape: [7, 3]  # 7 keypoints [x, y, visibility]
# Classes: 0=board

path: {output_root.resolve()}  # absolute dataset root dir
train: images/train
val: images/val
test: images/test

# Classes
names:
  0: board

# Keypoint shape & names
kpt_shape: [7, 3]
flip_idx: [1, 0, 3, 2, 4, 5, 6]  # left-right swaps for corner pairs
"""
    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"Wrote {yaml_path}")


def split_and_write(
    img_paths,
    keypoints_list,
    output_root: Path,
    test_size=0.15,
    val_size=0.15,
    random_state=42,
):
    """Split dataset and write YOLO pose labels."""
    output_root.mkdir(parents=True, exist_ok=True)

    for s in ["train", "val", "test"]:
        (output_root / "images" / s).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / s).mkdir(parents=True, exist_ok=True)

    write_yaml(output_root)

    train_val_idx, test_idx = train_test_split(
        range(len(img_paths)), test_size=test_size, random_state=random_state
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size / (1 - test_size),
        random_state=random_state,
    )

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}

    missing_images = 0
    for split_name, indices in splits.items():
        img_dir = output_root / "images" / split_name
        label_dir = output_root / "labels" / split_name

        for idx in indices:
            img_path = img_paths[idx]
            img_name = Path(img_path).name
            src_img = Path(img_path)
            if src_img.exists():
                shutil.copy2(src_img, img_dir / img_name)
            else:
                missing_images += 1
                if missing_images <= 5:
                    print(f"[WARN] Missing image: {src_img}")

            save_pose_label(img_name, keypoints_list[idx], label_dir)

    if missing_images:
        print(f"[WARN] {missing_images}/{len(img_paths)} images missing")

    print(f"Splits written to {output_root}")
    for name, idxs in splits.items():
        img_dir = output_root / "images" / name
        n_imgs = len(list(img_dir.iterdir())) if img_dir.exists() else 0
        n_labels = len(list((output_root / "labels" / name).glob("*.txt")))
        print(f"  {name}: {len(idxs)} labels, {n_imgs} images")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, help="Path to labels.pkl")
    parser.add_argument(
        "--output",
        default="data/processed/yolo_pose_darts",
        help="Output directory",
    )
    parser.add_argument(
        "--images",
        default=None,
        help="Base image directory (if needed to resolve relative paths)",
    )
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    args = parser.parse_args()

    pkl_path = Path(args.labels).resolve()

    # Auto-extract labels_pkl.zip if present
    pkl_zip = pkl_path.parent / "labels_pkl.zip"
    if pkl_zip.exists() and not pkl_path.exists():
        print(f"[EXTRACT] {pkl_zip}")
        shutil.unpack_archive(str(pkl_zip), str(pkl_path.parent))
        nested = pkl_path.parent / "labels_pkl" / "labels.pkl"
        if nested.exists():
            shutil.move(str(nested), str(pkl_path))
            (pkl_path.parent / "labels_pkl").rmdir()

    if not pkl_path.exists():
        print(f"[ERROR] {pkl_path} not found.")
        print(
            "Download the DeepDarts dataset first:\n"
            "  https://ieee-dataport.org/open-access/deepdarts-dataset"
        )
        sys.exit(1)

    data = load_labels(str(pkl_path))

    if hasattr(data, "columns"):
        print(f"Loaded {len(data)} images (DataFrame)")
        img_paths = [
            os.path.join(str(f), str(n))
            for f, n in zip(data["img_folder"], data["img_name"])
        ]
        kpts_list = [xy_list_to_array(xy) for xy in data["xy"]]
    else:
        print(f"Loaded {len(data['img_paths'])} images (dict)")
        img_paths = data["img_paths"]
        gts = data["gt"]

        def xy_array_to_pose(arr):
            out = np.zeros((NUM_KPTS, 3), dtype=np.float32)
            n = min(arr.shape[0], NUM_KPTS)
            out[:n, :2] = arr[:n, :2]
            out[:n, 2] = 2
            return out

        kpts_list = [xy_array_to_pose(g) for g in gts]

    if args.images:
        base = Path(args.images)
    else:
        base = pkl_path.parent / "cropped_images"

    # Auto-extract cropped_images.zip if present
    img_zip = pkl_path.parent / "cropped_images.zip"
    if img_zip.exists() and (not base.exists() or not any(base.iterdir())):
        print(f"[EXTRACT] {img_zip}")
        shutil.unpack_archive(str(img_zip), str(pkl_path.parent))

    img_paths = [str(base / p) for p in img_paths]

    split_and_write(
        img_paths,
        kpts_list,
        Path(args.output),
        test_size=args.test_size,
        val_size=args.val_size,
    )


if __name__ == "__main__":
    main()
