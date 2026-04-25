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


def build_image_lookup(search_root: Path) -> dict:
    """Build a mapping of lowercase relative paths to actual paths."""
    print(f"[INDEX] Scanning {search_root} for images...")
    lookup = {}
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif", ".heic", ".heif", ".tiff", ".tif", ".jp2", ".dng", ".mpo", ".bmp"}
    count = 0
    for p in search_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            try:
                rel = str(p.relative_to(search_root)).lower()
            except ValueError:
                continue
            lookup[rel] = p
            count += 1
            if count & 0x3FFF == 0:
                print(f"  ...indexed {count} images")
    print(f"[INDEX] Found {count} images")
    return lookup


def build_parent_lookup(lookup: dict) -> dict:
    """Build a lookup by (parent_folder_name, basename) for disambiguation."""
    parent = {}
    for rel, p in lookup.items():
        parts = Path(rel).parts
        if len(parts) < 2:
            continue
        folder_name = parts[-2].lower()
        basename = parts[-1].lower()
        parent[(folder_name, basename)] = p
    return parent


def find_image_path(
    folder: str, name: str, search_root: Path, lookup: dict, parent_lookup: dict
) -> Path:
    """Resolve actual image path."""
    exact = search_root / folder / name
    if exact.exists():
        return exact

    key = (folder.lower(), name.lower())
    if key in parent_lookup:
        return parent_lookup[key]

    rel_candidates = [
        f"{folder}/{name}".lower(),
        f"cropped_images/{folder}/{name}".lower(),
        f"images/{folder}/{name}".lower(),
        name.lower(),
    ]
    for rel in rel_candidates:
        if rel in lookup:
            return lookup[rel]

    return exact


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
    img_entries,
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

    indices = list(range(len(img_entries)))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size / (1 - test_size),
        random_state=random_state,
    )

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}

    missing_images = 0
    for split_name, idxs in splits.items():
        img_dir = output_root / "images" / split_name
        label_dir = output_root / "labels" / split_name

        for idx in idxs:
            src_img, img_name, gt = img_entries[idx]
            if src_img.exists():
                shutil.copy2(src_img, img_dir / img_name)
            else:
                missing_images += 1
                if missing_images <= 5:
                    print(f"[WARN] Missing image: {src_img}")

            save_pose_label(img_name, gt, label_dir)

    if missing_images:
        print(f"[WARN] {missing_images}/{len(img_entries)} images missing")

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
        print("Download the DeepDarts dataset first:")
        print("  https://ieee-dataport.org/open-access/deepdarts-dataset")
        sys.exit(1)

    data = load_labels(str(pkl_path))

    if hasattr(data, "columns"):
        print(f"Loaded {len(data)} images (DataFrame)")
        folders = data["img_folder"].astype(str).tolist()
        names = data["img_name"].astype(str).tolist()
        kpts_list = [xy_list_to_array(xy) for xy in data["xy"]]
    else:
        print(f"Loaded {len(data['img_paths'])} images (dict)")
        folders = [str(Path(p).parent) for p in data["img_paths"]]
        names = [str(Path(p).name) for p in data["img_paths"]]
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
        # Handle double-wrapped zip
        nested = base / "cropped_images"
        if nested.exists() and nested.is_dir():
            for item in nested.iterdir():
                shutil.move(str(item), str(base / item.name))
            nested.rmdir()

    # Build image lookup
    search_root = pkl_path.parent
    img_lookup = build_image_lookup(search_root)
    parent_lookup = build_parent_lookup(img_lookup)

    img_entries = []
    for folder, name, gt in zip(folders, names, kpts_list):
        src = find_image_path(folder, name, search_root, img_lookup, parent_lookup)
        img_entries.append((src, name, gt))

    found = sum(1 for src, _, _ in img_entries if src.exists())
    print(f"[CHECK] {found}/{len(img_entries)} images resolved")
    if found == 0:
        print(f"[ERROR] Zero images found under {search_root}")
        sys.exit(1)

    split_and_write(
        img_entries,
        Path(args.output),
        test_size=args.test_size,
        val_size=args.val_size,
    )


if __name__ == "__main__":
    main()
