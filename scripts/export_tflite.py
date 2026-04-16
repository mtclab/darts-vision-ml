#!/usr/bin/env python3
"""
Export trained YOLO11 models to TFLite format for Android deployment.

Usage:
    python scripts/export_tflite.py --model runs/pose/yolo11n_darts_pose/weights/best.pt
    python scripts/export_tflite.py --all
    python scripts/export_tflite.py --model best.pt --int8  # Quantized for smaller size
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

MODELS_DIR = Path("models")
ANDROID_ASSETS = Path("../darts_vision/app/src/main/assets/models")


def export_model(model_path: str, int8: bool = False, copy_to_android: bool = False):
    model = YOLO(model_path)

    export_path = model.export(
        format="tflite",
        imgsz=640,
        int8=int8,
    )

    dest = MODELS_DIR / Path(export_path).name
    MODELS_DIR.mkdir(exist_ok=True)
    shutil.copy2(str(export_path), str(dest))

    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"\n[EXPORT] Model: {model_path}")
    print(f"  TFLite: {dest}")
    print(f"  Size:   {size_mb:.1f} MB")
    print(f"  Int8:   {int8}")

    if copy_to_android:
        if ANDROID_ASSETS.exists():
            shutil.copy2(str(dest), str(ANDROID_ASSETS / dest.name))
            print(f"  Copied to Android assets: {ANDROID_ASSETS / dest.name}")
        else:
            print(f"  [WARN] Android assets dir not found: {ANDROID_ASSETS}")

    return dest


def export_all(int8: bool = False, copy_to_android: bool = False):
    model_dirs = {
        "pose": Path("runs/pose/yolo11n_darts_pose/weights/best.pt"),
        "detect": Path("runs/detect/yolo11n_darts_detect/weights/best.pt"),
        "calibration": Path("runs/calibration/yolo11n_board_calibration/weights/best.pt"),
    }

    for name, path in model_dirs.items():
        if path.exists():
            print(f"\n[EXPORT] {name}: {path}")
            export_model(str(path), int8=int8, copy_to_android=copy_to_android)
        else:
            print(f"[SKIP] {name}: {path} not found")


def main():
    parser = argparse.ArgumentParser(description="Export YOLO11 models to TFLite")
    parser.add_argument("--model", type=str, help="Path to specific .pt model")
    parser.add_argument("--all", action="store_true", help="Export all trained models")
    parser.add_argument("--int8", action="store_true", help="Quantize to INT8 (smaller, faster)")
    parser.add_argument("--copy-android", action="store_true", help="Copy to Android assets")
    args = parser.parse_args()

    if args.all:
        export_all(int8=args.int8, copy_to_android=args.copy_android)
    elif args.model:
        export_model(args.model, int8=args.int8, copy_to_android=args.copy_android)
    else:
        print("Specify --model <path> or --all")
        print("\nAvailable models after training:")
        for name, path in [
            ("Pose (dart tips + calibration)", "runs/pose/yolo11n_darts_pose/weights/best.pt"),
            ("Detect (bounding boxes)", "runs/detect/yolo11n_darts_detect/weights/best.pt"),
            ("Calibration (board keypoints)", "runs/calibration/yolo11n_board_calibration/weights/best.pt"),
        ]:
            exists = "✓" if Path(path).exists() else " "
            print(f"  [{exists}] {name}: {path}")


if __name__ == "__main__":
    main()