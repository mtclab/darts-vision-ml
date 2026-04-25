"""
Convert DeepDarts labels.pkl to YOLO detection format.

DeepDarts annotations: 7 keypoints [cal1, cal2, cal3, cal4, dart1, dart2, dart3]
YOLO format per image: one .txt with rows of <class> <x_center> <y_center> <width> <height>
Classes: 0=dart, 1=cal_corner

Usage:
    python src/convert_dataset.py \
        --labels data/raw/deep-darts/dataset/labels.pkl \
        --images data/raw/deep-darts/dataset/cropped_images \
        --output data/processed/yolo_detect_deepdarts \
        --bbox-size 0.025
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


CLASSES = {"dart": 0, "cal_corner": 1}
BBOX_SIZE = 0.025  # fraction of image side


def load_labels(pkl_path: str):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


def keypoint_to_yolo_bbox(
    x: float, y: float, bbox_size: float
) -> Tuple[float, float, float, float]:
    """Convert normalized keypoint (0-1) to YOLO bbox."""
    return (x, y, bbox_size, bbox_size)


def save_yolo_labels(
    img_name: str,
    keypoints: np.ndarray,
    label_dir: Path,
    bbox_size: float = BBOX_SIZE,
):
    """Save one image's keypoints as YOLO .txt file."""
    lines = []
    for i in range(7):
        x, y, vis = keypoints[i]
        if vis == 0:
            continue
        cls_id = CLASSES["cal_corner"] if i < 4 else CLASSES["dart"]
        xc, yc, w, h = keypoint_to_yolo_bbox(x, y, bbox_size)
        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    out_path = label_dir / f"{Path(img_name).stem}.txt"
    with open(out_path, "w") as f:
        f.writelines(lines)


def xy_list_to_array(xy_list: list) -> np.ndarray:
    """Convert list of [x, y] points to (7, 3) array with visibility."""
    arr = np.zeros((7, 3), dtype=np.float32)
    n = min(len(xy_list), 7)
    arr[:n, :2] = np.array(xy_list[:n])
    arr[:n, 2] = 1
    return arr


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
            if count & 0x3FFF == 0:  # log every 16k, rarely triggers for small sets
                print(f"  ...indexed {count} images")
    print(f"[INDEX] Found {count} images")
    return lookup


def find_image_path(
    folder: str, name: str, search_root: Path, lookup: dict
) -> Path:
    """Resolve actual image path, trying exact then lookup."""
    # Exact path under search_root
    exact = search_root / folder / name
    if exact.exists():
        return exact

    # Lookup via lowercase relative path
    rel_candidates = [
        f"{folder}/{name}".lower(),
        f"cropped_images/{folder}/{name}".lower(),
        f"images/{folder}/{name}".lower(),
        name.lower(),
    ]
    for rel in rel_candidates:
        if rel in lookup:
            return lookup[rel]

    return exact  # fallback so caller can report missing


def write_yaml(output_root: Path):
    """Write dataset YAML with absolute path for Ultralytics."""
    yaml_path = output_root / "darts.yaml"
    content = f"""# YOLOv8 Dataset Configuration for Darts-Vision-ML
# Classes: 0=dart, 1=cal_corner

path: {output_root.resolve()}  # absolute dataset root dir
train: images/train
val: images/val
test: images/test

# Classes
names:
  0: dart
  1: cal_corner
"""
    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"Wrote {yaml_path}")


def split_and_write(
    img_entries,
    output_root: Path,
    bbox_size: float,
    test_size=0.15,
    val_size=0.15,
    random_state=42,
):
    """Split dataset into train/val/test and write YOLO labels."""
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

            save_yolo_labels(img_name, gt, label_dir, bbox_size)

    if missing_images:
        print(f"[WARN] {missing_images}/{len(img_entries)} images missing")

    print(f"Splits written to {output_root}")
    for name, idxs in splits.items():
        img_dir = output_root / "images" / name
        n_imgs = len(list(img_dir.iterdir())) if img_dir.exists() else 0
        n_labels = len(list((output_root / "labels" / name).glob("*.txt")))
        print(f"  {name}: {len(idxs)} labels, {n_imgs} images")


def try_extract_zip(zip_path: Path, dest: Path) -> bool:
    """Extract zip if it exists and destination is empty."""
    if not zip_path.exists():
        return False
    if dest.exists() and any(dest.iterdir()):
        return False
    print(f"[EXTRACT] {zip_path}")
    shutil.unpack_archive(str(zip_path), str(dest.parent if zip_path.name.lower() == "cropped_images.zip" else dest))
    # Handle nested zip: zip contains a single folder with same name
    nested = dest.parent / zip_path.stem / "cropped_images"
    if nested.exists() and nested.is_dir() and not (dest.exists() and any(dest.iterdir())):
        # Move contents up one level
        for item in nested.iterdir():
            shutil.move(str(item), str(dest / item.name))
        (dest.parent / zip_path.stem).rmdir()
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, help="Path to labels.pkl")
    parser.add_argument(
        "--output",
        default="data/processed/yolo_detect_deepdarts",
        help="Output directory",
    )
    parser.add_argument(
        "--images",
        default=None,
        help="Base image directory (if needed to resolve relative paths)",
    )
    parser.add_argument("--bbox-size", type=float, default=BBOX_SIZE)
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
        print(f"Loaded {len(data)} images")
        folders = data["img_folder"].astype(str).tolist()
        names = data["img_name"].astype(str).tolist()
        gts = [xy_list_to_array(xy) for xy in data["xy"]]
    else:
        print(f"Loaded {len(data['img_paths'])} images")
        folders = [str(Path(p).parent) for p in data["img_paths"]]
        names = [str(Path(p).name) for p in data["img_paths"]]
        gts = data["gt"]

    # Determine base directory for images
    if args.images:
        base = Path(args.images)
    else:
        base = pkl_path.parent / "cropped_images"

    # Auto-extract cropped_images.zip if present
    img_zip = pkl_path.parent / "cropped_images.zip"
    if img_zip.exists() and (not base.exists() or not any(base.iterdir())):
        print(f"[EXTRACT] {img_zip}")
        shutil.unpack_archive(str(img_zip), str(pkl_path.parent))
        # Handle double-wrapped zip: cropped_images/cropped_images/...
        nested = base / "cropped_images"
        if nested.exists() and nested.is_dir():
            for item in nested.iterdir():
                shutil.move(str(item), str(base / item.name))
            nested.rmdir()

    # Build image lookup from dataset root (handles any nesting depth)
    search_root = pkl_path.parent
    img_lookup = build_image_lookup(search_root)

    img_entries = []
    for folder, name, gt in zip(folders, names, gts):
        src = find_image_path(folder, name, search_root, img_lookup)
        img_entries.append((src, name, gt))

    # Sanity check
    found = sum(1 for src, _, _ in img_entries if src.exists())
    print(f"[CHECK] {found}/{len(img_entries)} images resolved")
    if found == 0:
        print(f"[ERROR] Zero images found under {search_root}")
        print("Verify your dataset is extracted correctly:")
        print(f"  ls {search_root}")
        sys.exit(1)

    split_and_write(
        img_entries,
        Path(args.output),
        bbox_size=args.bbox_size,
        test_size=args.test_size,
        val_size=args.val_size,
    )


if __name__ == "__main__":
    main()
