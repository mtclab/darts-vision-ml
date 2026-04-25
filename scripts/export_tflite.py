#!/usr/bin/env python3
"""
Export trained YOLO models to TFLite for Android deployment.

Usage:
    python scripts/export_tflite.py                   # export recommended models
    python scripts/export_tflite.py --all             # export all trained models
    python scripts/export_tflite.py --model runs/darts/yolov8s_1280_tight/weights/best.pt --imgsz 1280
    python scripts/export_tflite.py --all --int8      # INT8 quantized
    python scripts/export_tflite.py --model /path/to/best.pt --name my_model --imgsz 1280
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

MODELS_DIR = Path("models")
ANDROID_ASSETS = Path("../darts_vision/app/src/main/assets/models")

# Default expected paths; script will try to auto-resolve if these don't exist
DEFAULT_MODELS = {
    "board_calibration": Path("runs/board_calibration/train/weights/best.pt"),
    "darts_pose": Path("runs/darts_pose/train/weights/best.pt"),
    "darts_detect": Path("runs/darts_detect/train/weights/best.pt"),
}

RECOMMENDED = ["board_calibration", "darts_pose"]


def resolve_model_path(path: Path) -> Path:
    """If exact path doesn't exist, try to find a close match (handles auto-increment suffixes)."""
    if path.exists():
        return path
    # Try finding paths with auto-increment suffix like best-2.pt or directories like yolov8s_1280_tight-3
    parent = path.parent
    if not parent.parent.exists():
        return path
    # Check for alternative run directories with same prefix
    prefix = parent.name
    alt_dirs = sorted(parent.parent.glob(f"{prefix}*"))
    for alt_dir in alt_dirs:
        candidate = alt_dir / "weights" / "best.pt"
        if candidate.exists():
            print(f"[INFO] Resolved {path} -> {candidate}")
            return candidate
    return path


def detect_model_key(model_path: str) -> str:
    path = Path(model_path)
    for key, default_path in DEFAULT_MODELS.items():
        if key in model_path or (path.resolve() == default_path.resolve()):
            return key
    return path.stem if path.suffix == ".pt" else "unknown"


def export_model(
    model_path: str,
    imgsz: int = 640,
    int8: bool = False,
    copy_to_android: bool = False,
    model_key: str = None,
    output_name: str = None,
):
    model_path = str(resolve_model_path(Path(model_path)))
    model = YOLO(model_path)

    export_path = model.export(
        format="tflite",
        imgsz=imgsz,
        int8=int8,
        simplify=True,
    )

    export_path = Path(export_path)
    if not export_path.exists():
        # Try alternative naming that YOLO sometimes produces
        alt = export_path.with_name(f"{Path(model_path).stem}_int8.tflite")
        if alt.exists():
            export_path = alt
        else:
            print(f"[ERROR] Export file not found: {export_path}")
            return None

    key = model_key or detect_model_key(model_path)
    suffix = "_int8" if int8 else "_float32"
    dest_name = output_name or f"{key}{suffix}.tflite"
    dest = MODELS_DIR / dest_name
    MODELS_DIR.mkdir(exist_ok=True)
    shutil.copy2(str(export_path), str(dest))

    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"\n[EXPORT] Model: {model_path}")
    print(f"  TFLite: {dest}")
    print(f"  Size:   {size_mb:.1f} MB")
    print(f"  Int8:   {int8}")
    print(f"  Input:  {imgsz}x{imgsz}")

    if copy_to_android:
        if ANDROID_ASSETS.exists():
            shutil.copy2(str(dest), str(ANDROID_ASSETS / dest.name))
            print(f"  Copied to Android assets: {ANDROID_ASSETS / dest.name}")
        else:
            print(f"  [WARN] Android assets dir not found: {ANDROID_ASSETS}")

    return dest


def export_recommended(imgsz: int = 640, int8: bool = False, copy_to_android: bool = False):
    print(f"[INFO] Exporting recommended models (calibration + pose) @ {imgsz}px")
    for name in RECOMMENDED:
        path = resolve_model_path(DEFAULT_MODELS[name])
        if path.exists():
            print(f"\n[EXPORT] {name}: {path}")
            export_model(str(path), imgsz=imgsz, int8=int8, copy_to_android=copy_to_android, model_key=name)
        else:
            print(f"[SKIP] {name}: {path} not found")


def export_all(imgsz: int = 640, int8: bool = False, copy_to_android: bool = False):
    print(f"[INFO] Exporting all trained models @ {imgsz}px")
    for name, default_path in DEFAULT_MODELS.items():
        path = resolve_model_path(default_path)
        if path.exists():
            print(f"\n[EXPORT] {name}: {path}")
            export_model(str(path), imgsz=imgsz, int8=int8, copy_to_android=copy_to_android, model_key=name)
        else:
            print(f"[SKIP] {name}: {path} not found")


def main():
    parser = argparse.ArgumentParser(description="Export YOLO models to TFLite")
    parser.add_argument("--model", type=str, help="Path to specific .pt model")
    parser.add_argument("--all", action="store_true", help="Export all trained models (not just recommended)")
    parser.add_argument("--int8", action="store_true", help="Quantize to INT8")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size for export")
    parser.add_argument("--name", type=str, default=None, help="Output filename (default: auto from model key)")
    parser.add_argument("--copy-android", action="store_true", help="Copy to Android assets")
    args = parser.parse_args()

    if args.all:
        export_all(imgsz=args.imgsz, int8=args.int8, copy_to_android=args.copy_android)
    elif args.model:
        export_model(args.model, imgsz=args.imgsz, int8=args.int8, copy_to_android=args.copy_android, output_name=args.name)
    else:
        export_recommended(imgsz=args.imgsz, int8=args.int8, copy_to_android=args.copy_android)


if __name__ == "__main__":
    main()
