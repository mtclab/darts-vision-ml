#!/usr/bin/env python3
"""
Train YOLO11n-pose model for dart detection.

Detects dartboard with up to 7 keypoints:
- 4 calibration points (double-20, double-6, double-3, double-11 corners)
- Up to 3 dart tip positions

Usage:
    python scripts/train_pose.py [--epochs 100] [--batch 16] [--gpu 0]
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def train_pose(args):
    model = YOLO("yolo11n-pose.pt")

    results = model.train(
        data=str(Path("configs/dataset_pose.yaml").resolve()),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=640,
        device=args.gpu,
        project="runs/pose",
        name="yolo11n_darts_pose",
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
        copy_paste=0.3,
        erasing=0.4,
    )

    print(f"\n[TRAIN] Best model: {runs/pose/yolo11n_darts_pose/weights/best.pt}")
    print(f"[VAL]   Results: {results}")

    return results


def validate(model_path: str):
    model = YOLO(model_path)
    metrics = model.val(data=str(Path("configs/dataset_pose.yaml").resolve()))
    print(f"\n[VAL] Pose metrics:")
    print(f"  Box Precision: {metrics.box.mp:.4f}")
    print(f"  Box Recall:    {metrics.box.mr:.4f}")
    print(f"  Box mAP50:     {metrics.box.map50:.4f}")
    print(f"  Box mAP50-95:  {metrics.box.map:.4f}")
    print(f"  KP mAP50:      {metrics.kpt.map50:.4f}" if hasattr(metrics, 'kpt') else "")
    return metrics


def export_tflite(model_path: str):
    model = YOLO(model_path)
    export_path = model.export(format="tflite", imgsz=640)
    print(f"\n[EXPORT] TFLite model: {export_path}")
    print(f"[INFO] Copy this to: darts_vision/app/src/main/assets/models/yolov8n_darts.tflite")
    return export_path


def main():
    parser = argparse.ArgumentParser(description="Train YOLO11n-pose dart detection model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--gpu", type=str, default="0", help="GPU device (0, 1, or cpu)")
    parser.add_argument("--resume", type=str, default=None, help="Path to resume training from")
    parser.add_argument("--validate-only", action="store_true", help="Run validation only")
    parser.add_argument("--export", action="store_true", help="Export to TFLite after training")
    parser.add_argument("--model", type=str, default=None, help="Path to model for validate/export")
    args = parser.parse_args()

    if args.validate_only:
        model_path = args.model or "runs/pose/yolo11n_darts_pose/weights/best.pt"
        validate(model_path)
        return

    if args.export:
        model_path = args.model or "runs/pose/yolo11n_darts_pose/weights/best.pt"
        export_tflite(model_path)
        return

    if args.resume:
        model = YOLO(args.resume)
        model.train(resume=True)
        return

    results = train_pose(args)
    print("\n[DONE] Training complete.")
    print("Next: python scripts/train_pose.py --export")


if __name__ == "__main__":
    main()