#!/usr/bin/env python3
"""
Train board-calibration model (board + 4 calibration keypoints).

Primary model — replaces CV BoardDetector with ML homography estimation.
Detects double-20, double-6, double-3, double-11 corners for perspective correction.

Usage:
    python scripts/train_board_calibration.py --gpu 0
    python scripts/train_board_calibration.py --gpu 0,1,2,3
    python scripts/train_board_calibration.py --resume runs/board_calibration/train/weights/last.pt
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

from train_utils import ensure_pretrained, parse_device, validate_dataset

RUN_DIR = Path.cwd() / "runs" / "board_calibration"
CONFIG = "configs/dataset_calibration.yaml"


def train_board_calibration(args):
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
        project=str(RUN_DIR),
        name="train",
        exist_ok=True,
        patience=25,
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

    print(f"\n[TRAIN] Best model: {RUN_DIR / 'train' / 'weights' / 'best.pt'}")
    return results


def validate(model_path: str):
    model = YOLO(model_path)
    metrics = model.val(data=str(Path(CONFIG).resolve()))
    print(f"\n[VAL] Board Calibration metrics:")
    print(f"  Box mAP50:     {metrics.box.map50:.4f}")
    print(f"  Box mAP50-95:  {metrics.box.map:.4f}")
    if hasattr(metrics, 'kpt'):
        print(f"  KP mAP50:      {metrics.kpt.map50:.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train board-calibration model (4 keypoints)")
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
        model_path = args.model or str(RUN_DIR / "train" / "weights" / "best.pt")
        validate(model_path)
        return

    if args.resume:
        model = YOLO(args.resume)
        results = model.train(resume=True)
        print(f"\n[RESUME] Continued from: {args.resume}")
        print(f"[VAL]   Results: {results}")
        return

    train_board_calibration(args)
    print("\n[DONE] Training complete.")


if __name__ == "__main__":
    main()