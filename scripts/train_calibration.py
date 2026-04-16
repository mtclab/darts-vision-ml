#!/usr/bin/env python3
"""
Train YOLO11n-pose model for board calibration keypoint detection.

Learns to detect 4 calibration keypoints on a dartboard:
- Double-20 corner
- Double-6 corner
- Double-3 corner
- Double-11 corner

This model can REPLACE the CV-based BoardDetector, providing more robust
board detection and homography estimation from the 4 calibration points.

Usage:
    python scripts/train_calibration.py [--epochs 100] [--batch 16] [--gpu 0]
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def train_calibration(args):
    model = YOLO("yolo11n-pose.pt")

    results = model.train(
        data=str(Path("configs/dataset_calibration.yaml").resolve()),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=640,
        device=args.gpu,
        project="runs/calibration",
        name="yolo11n_board_calibration",
        exist_ok=True,
        patience=20,
        save_period=10,
        val=True,
        plots=True,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=30.0,
        translate=0.1,
        scale=0.7,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
    )

    print(f"\n[TRAIN] Best model: runs/calibration/yolo11n_board_calibration/weights/best.pt")
    return results


def validate(model_path: str):
    model = YOLO(model_path)
    metrics = model.val(data=str(Path("configs/dataset_calibration.yaml").resolve()))
    print(f"\n[VAL] Calibration metrics:")
    print(f"  Box mAP50:     {metrics.box.map50:.4f}")
    print(f"  Box mAP50-95:  {metrics.box.map:.4f}")
    if hasattr(metrics, 'kpt'):
        print(f"  KP mAP50:      {metrics.kpt.map50:.4f}")
    return metrics


def export_tflite(model_path: str):
    model = YOLO(model_path)
    export_path = model.export(format="tflite", imgsz=640)
    print(f"\n[EXPORT] TFLite model: {export_path}")
    print(f"[INFO] Copy to: darts_vision/app/src/main/assets/models/board_calibration.tflite")
    return export_path


def main():
    parser = argparse.ArgumentParser(description="Train board calibration keypoint model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    if args.validate_only:
        model_path = args.model or "runs/calibration/yolo11n_board_calibration/weights/best.pt"
        validate(model_path)
        return

    if args.export:
        model_path = args.model or "runs/calibration/yolo11n_board_calibration/weights/best.pt"
        export_tflite(model_path)
        return

    train_calibration(args)
    print("\n[DONE] Training complete.")


if __name__ == "__main__":
    main()