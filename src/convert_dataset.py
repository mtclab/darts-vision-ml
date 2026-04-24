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
import pickle
import os
import shutil
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


def keypoint_to_yolo_bbox(x: float, y: float, bbox_size: float = BBOX_SIZE) -> Tuple[int, float, float, float, float]:
    """Convert normalized keypoint (0-1) to YOLO bbox."""
    return (x, y, bbox_size, bbox_size)


def save_yolo_labels(img_name: str, keypoints: np.ndarray, output_dir: Path, bbox_size: float = BBOX_SIZE):
    """Save one image's keypoints as YOLO .txt file.
    keypoints shape: (7, 3) -> [cal1..cal4, dart1..dart3], last col is visibility
    """
    lines = []
    for i in range(7):
        x, y, vis = keypoints[i]
        if vis == 0:
            continue
        cls = CLASSES["cal_corner"] if i < 4 else CLASSES["dart"]
        xc, yc, w, h = keypoint_to_yolo_bbox(x, y, bbox_size)
        lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
    
    out_path = output_dir / f"{Path(img_name).stem}.txt"
    with open(out_path, "w") as f:
        f.writelines(lines)


def xy_list_to_array(xy_list: list) -> np.ndarray:
    """Convert list of [x, y] points to (7, 3) array with visibility."""
    arr = np.zeros((7, 3), dtype=np.float32)
    n = min(len(xy_list), 7)
    arr[:n, :2] = np.array(xy_list[:n])
    arr[:n, 2] = 1
    return arr


def write_yaml(output_root: Path):
    """Write dataset YAML with absolute path for Ultralytics.
    Ultralytics resolves relative paths from its own datasets/ dir,
    so absolute path is required for portability.
    """
    yaml_path = output_root / "darts.yaml"
    content = f"""# YOLOv8 Dataset Configuration for Darts-Vision-ML
# Classes: 0=dart, 1=cal_corner

path: {output_root.resolve()}  # absolute dataset root dir
train: train
val: val
test: test

# Classes
names:
  0: dart
  1: cal_corner
"""
    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"Wrote {yaml_path}")


def split_and_write(img_paths, gts, output_root: Path, test_size=0.15, val_size=0.15, random_state=42):
    """Split dataset into train/val/test and write YOLO labels."""
    output_root.mkdir(parents=True, exist_ok=True)
    write_yaml(output_root)

    # First split: separate test
    train_val_idx, test_idx = train_test_split(
        range(len(img_paths)), test_size=test_size, random_state=random_state
    )
    # Second split: separate val from train
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_size/(1-test_size), random_state=random_state
    )

    splits = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }

    for split_name, indices in splits.items():
        split_dir = output_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        for idx in indices:
            img_path = img_paths[idx]
            img_name = Path(img_path).name

            # Copy image
            src_img = Path(img_path)
            if src_img.exists():
                shutil.copy2(src_img, split_dir / img_name)

            # Write labels
            save_yolo_labels(img_name, gts[idx], split_dir, BBOX_SIZE)

    print(f"Splits written to {output_root}")
    for name, idxs in splits.items():
        print(f"  {name}: {len(idxs)} images")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, help="Path to labels.pkl")
    parser.add_argument("--output", default="data/processed/yolo_detect_deepdarts", help="Output directory")
    parser.add_argument("--images", default=None, help="Base image directory (if needed to resolve relative paths)")
    parser.add_argument("--bbox-size", type=float, default=BBOX_SIZE)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    args = parser.parse_args()

    pkl_path = Path(args.labels).resolve()
    data = load_labels(str(pkl_path))

    if hasattr(data, 'columns'):
        # DataFrame with columns img_folder, img_name, bbox, xy
        print(f"Loaded {len(data)} images")
        img_paths = [os.path.join(str(f), str(n)) for f, n in zip(data['img_folder'], data['img_name'])]
        gts = [xy_list_to_array(xy) for xy in data['xy']]
    else:
        print(f"Loaded {len(data['img_paths'])} images")
        img_paths = data['img_paths']
        gts = data['gt']

    # Auto-resolve image base directory next to the .pkl if not provided
    if args.images:
        base = Path(args.images)
    else:
        base = pkl_path.parent / "cropped_images"
    img_paths = [str(base / p) for p in img_paths]

    split_and_write(
        img_paths,
        gts,
        Path(args.output),
        test_size=args.test_size,
        val_size=args.val_size,
    )


if __name__ == "__main__":
    main()
