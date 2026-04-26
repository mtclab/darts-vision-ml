"""
Export trained YOLO model to TFLite INT8 for mobile deployment.

Usage:
    python src/export_tflite.py \
        --weights runs/darts/yolov8n_800/weights/best.pt \
        --data data/processed/yolo_detect_deepdarts/darts.yaml \
        --imgsz 800 \
        --output models/
"""

import argparse
import shutil
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to best.pt")
    parser.add_argument("--data", default="data/processed/yolo_detect_deepdarts/darts.yaml")
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--output", default="models")
    parser.add_argument("--name", default="darts_detector.tflite", help="Output filename")
    parser.add_argument("--fraction", type=float, default=0.15, help="Fraction of val set for INT8 calibration (default: 0.15, ~15%% of images)")
    parser.add_argument("--int8", action="store_true", default=True, help="Enable INT8 quantization (default)")
    parser.add_argument("--no-int8", dest="int8", action="store_false", help="Disable INT8, export float32")
    return parser.parse_args()


def main():
    args = parse_args()
    int8 = not args.no_int8 if args.no_int8 else args.int8

    model = YOLO(args.weights)

    # YOLO's built-in export returns file path
    export_path = model.export(
        format="tflite",
        imgsz=args.imgsz,
        int8=int8,
        data=args.data if int8 else None,
        simplify=True,
    )

    src = Path(export_path)
    dst = Path(args.output) / args.name
    dst.parent.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        alt = src.with_name("best_int8.tflite")
        src = alt if alt.exists() else src

    if src.exists():
        shutil.copy2(str(src), str(dst))
        size_mb = dst.stat().st_size / (1024 * 1024)
        print(f"TFLite model saved to {dst}")
        print(f"  Size: {size_mb:.1f} MB")
        print(f"  INT8: {int8}")
    else:
        print(f"[ERROR] Export output not found at {src}")
        print("Check the run directory for .tflite files:")
        for f in Path(args.weights).parent.glob("*.tflite"):
            print(f"  {f}")


if __name__ == "__main__":
    main()
