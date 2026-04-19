#!/usr/bin/env python3
"""
Export trained YOLO11 models to TFLite for Android deployment.

Usage:
    python scripts/export_tflite.py                  # export recommended models (calibration + pose)
    python scripts/export_tflite.py --all            # export all trained models
    python scripts/export_tflite.py --model runs/board_calibration/train/weights/best.pt
    python scripts/export_tflite.py --all --int8     # INT8 quantized
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

MODELS_DIR = Path("models")
ANDROID_ASSETS = Path("../darts_vision/app/src/main/assets/models")

MODELS = {
    "board_calibration": Path("runs/board_calibration/train/weights/best.pt"),
    "darts_pose": Path("runs/darts_pose/train/weights/best.pt"),
    "darts_detect": Path("runs/darts_detect/train/weights/best.pt"),
}

RECOMMENDED = ["board_calibration", "darts_pose"]


def detect_model_key(model_path: str) -> str:
    for key, _ in MODELS.items():
        if key in model_path:
            return key
    return "unknown"


def export_model(model_path: str, int8: bool = False, copy_to_android: bool = False, model_key: str = None):
    model = YOLO(model_path)

    export_path = model.export(
        format="tflite",
        imgsz=640,
        int8=int8,
    )

    export_path = Path(export_path)
    if not export_path.exists():
        print(f"[ERROR] Export file not found: {export_path}")
        return None

    key = model_key or detect_model_key(model_path)
    suffix = "_int8" if int8 else "_float32"
    dest_name = f"{key}{suffix}.tflite"
    dest = MODELS_DIR / dest_name
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


def export_recommended(int8: bool = False, copy_to_android: bool = False):
    print("[INFO] Exporting recommended models (calibration + pose)")
    for name in RECOMMENDED:
        path = MODELS[name]
        if path.exists():
            print(f"\n[EXPORT] {name}: {path}")
            export_model(str(path), int8=int8, copy_to_android=copy_to_android, model_key=name)
        else:
            print(f"[SKIP] {name}: {path} not found")


def export_all(int8: bool = False, copy_to_android: bool = False):
    print("[INFO] Exporting all trained models")
    for name, path in MODELS.items():
        if path.exists():
            print(f"\n[EXPORT] {name}: {path}")
            export_model(str(path), int8=int8, copy_to_android=copy_to_android, model_key=name)
        else:
            print(f"[SKIP] {name}: {path} not found")


def main():
    parser = argparse.ArgumentParser(description="Export YOLO11 models to TFLite")
    parser.add_argument("--model", type=str, help="Path to specific .pt model")
    parser.add_argument("--all", action="store_true", help="Export all trained models (not just recommended)")
    parser.add_argument("--int8", action="store_true", help="Quantize to INT8")
    parser.add_argument("--copy-android", action="store_true", help="Copy to Android assets")
    args = parser.parse_args()

    if args.all:
        export_all(int8=args.int8, copy_to_android=args.copy_android)
    elif args.model:
        export_model(args.model, int8=args.int8, copy_to_android=args.copy_android)
    else:
        export_recommended(int8=args.int8, copy_to_android=args.copy_android)


if __name__ == "__main__":
    main()