#!/usr/bin/env python3
"""
Train YOLO11n-detection model for dart + board detection.

Simpler than pose model — just bounding boxes for dartboard and dart tips.
Less precise (bbox center ≈ dart position) but easier to train and deploy.

Usage:
    python scripts/train_detect.py [--epochs 100] [--batch 16] [--gpu 0]
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def train_detect(args):
    model = YOLO("yolo11n.pt")

    results = model.train(
        data=str(Path("configs/dataset_detect.yaml").resolve()),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=640,
        device=args.gpu,
        project="runs/detect",
        name="yolo11n_darts_detect",
        exist_ok=True,
        patience=20,
        save_period=10,
        val=True,
        plots=True,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,
        translate=0.1,
        scale=0.5,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
    )

    print(f"\n[TRAIN] Best model: runs/detect/yolo11n_darts_detect/weights/best.pt")
    return results


def validate(model_path: str):
    model = YOLO(model_path)
    metrics = model.val(data=str(Path("configs/dataset_detect.yaml").resolve()))
    print(f"\n[VAL] Detection metrics:")
    print(f"  Precision:  {metrics.box.mp:.4f}")
    print(f"  Recall:     {metrics.box.mr:.4f}")
    print(f"  mAP50:      {metrics.box.map50:.4f}")
    print(f"  mAP50-95:   {metrics.box.map:.4f}")
    return metrics


def export_tflite(model_path: str):
    model = YOLO(model_path)
    export_path = model.export(format="tflite", imgsz=640)
    print(f"\n[EXPORT] TFLite model: {export_path}")
    return export_path


def main():
    parser = argparse.ArgumentParser(description="Train YOLO11n-detection dart model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    if args.validate_only:
        model_path = args.model or "runs/detect/yolo11n_darts_detect/weights/best.pt"
        validate(model_path)
        return

    if args.export:
        model_path = args.model or "runs/detect/yolo11n_darts_detect/weights/best.pt"
        export_tflite(model_path)
        return

    train_detect(args)
    print("\n[DONE] Training complete.")


if __name__ == "__main__":
    main()