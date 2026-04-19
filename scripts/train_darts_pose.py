#!/usr/bin/env python3
"""
Train darts-pose model (board + 7 keypoints: 4 calibration + 3 dart tips).

Recommended fallback model for single-frame dart detection when no
reference frame is available for frame-differencing.

Usage:
    python scripts/train_darts_pose.py --gpu 0
    python scripts/train_darts_pose.py --gpu 0,1,2,3
    python scripts/train_darts_pose.py --resume runs/darts_pose/weights/last.pt
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

from train_utils import ensure_pretrained, parse_device, validate_dataset

RUN_DIR = "runs/darts_pose"
CONFIG = "configs/dataset_pose.yaml"


def train_darts_pose(args):
    if not validate_dataset(CONFIG):
        return None

    weights = ensure_pretrained("yolo11n-pose.pt")
    model = YOLO(weights)

    results = model.train(
        data=str(Path(CONFIG).resolve()),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=640,
        device=args.device,
        project=RUN_DIR,
        name="weights",
        exist_ok=True,
        patience=25,
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
        erasing=0.4,
    )

    print(f"\n[TRAIN] Best model: {RUN_DIR}/weights/best.pt")
    print(f"[VAL]   Results: {results}")
    return results


def validate(model_path: str):
    model = YOLO(model_path)
    metrics = model.val(data=str(Path(CONFIG).resolve()))
    print(f"\n[VAL] Darts Pose metrics:")
    print(f"  Box Precision: {metrics.box.mp:.4f}")
    print(f"  Box Recall:    {metrics.box.mr:.4f}")
    print(f"  Box mAP50:     {metrics.box.map50:.4f}")
    print(f"  Box mAP50-95:  {metrics.box.map:.4f}")
    if hasattr(metrics, 'kpt'):
        print(f"  KP mAP50:      {metrics.kpt.map50:.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train darts-pose model (7 keypoints)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()
    args.device = parse_device(args.gpu)
    if isinstance(args.device, list) and args.batch == 16:
        args.batch = len(args.device) * 16
        print(f"[DDP] {len(args.device)} GPUs detected, auto-scaling batch to {args.batch}")

    if args.validate_only:
        model_path = args.model or f"{RUN_DIR}/weights/best.pt"
        validate(model_path)
        return

    if args.resume:
        model = YOLO(args.resume)
        results = model.train(resume=True)
        print(f"\n[RESUME] Continued from: {args.resume}")
        print(f"[VAL]   Results: {results}")
        return

    train_darts_pose(args)
    print("\n[DONE] Training complete.")


if __name__ == "__main__":
    main()