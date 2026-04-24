"""
Export trained YOLO model to TFLite INT8 for mobile deployment.

Usage:
    python src/export_tflite.py \
        --weights runs/darts/yolov8n_800/weights/best.pt \
        --data data/darts.yaml \
        --imgsz 800 \
        --output models/
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to best.pt")
    parser.add_argument("--data", default="data/darts.yaml")
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--output", default="models")
    return parser.parse_args()


def main():
    args = parse_args()
    
    model = YOLO(args.weights)
    
    # Export to TFLite with INT8 quantization
    model.export(
        format="tflite",
        imgsz=args.imgsz,
        int8=True,
        data=args.data,
        simplify=True,
    )
    
    # Move exported model to models/
    weights_path = Path(args.weights)
    src = weights_path.parent / "best_int8.tflite"
    dst = Path(args.output) / "darts_detector.tflite"
    dst.parent.mkdir(parents=True, exist_ok=True)
    
    if src.exists():
        dst.write_bytes(src.read_bytes())
        print(f"TFLite model saved to {dst}")
    else:
        print(f"Export output not found at {src}")
        print("Check runs/darts/*/weights/ for .tflite files")


if __name__ == "__main__":
    main()
