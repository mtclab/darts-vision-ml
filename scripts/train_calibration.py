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
    Single GPU:   python scripts/train_calibration.py --gpu 0
    Multi-GPU:   python scripts/train_calibration.py --gpu 0,1,2,3
    CPU:         python scripts/train_calibration.py --gpu cpu
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

WEIGHTS_DIR = Path("weights")


def ensure_pretrained(name: str) -> str:
    WEIGHTS_DIR.mkdir(exist_ok=True)
    dest = WEIGHTS_DIR / name
    if not dest.exists():
        model = YOLO(name)
        src = Path(model.ckpt_path)
        if src.exists() and src != dest:
            import shutil
            shutil.move(str(src), str(dest))
    return str(dest)


def parse_device(device_str: str):
    if device_str.lower() == "cpu":
        return "cpu"
    parts = [int(x.strip()) for x in device_str.split(",")]
    return parts if len(parts) > 1 else parts[0]


def train_calibration(args):
    config_path = Path("configs/dataset_calibration.yaml").resolve()
    if not config_path.exists():
        print(f"[ERROR] Config not found: {config_path}")
        print("Run: python scripts/download_and_convert.py")
        return None

    import yaml
    with open(config_path) as f:
        ds_cfg = yaml.safe_load(f)
    ds_path = Path(ds_cfg["path"])
    if not ds_path.exists():
        print(f"[ERROR] Dataset path not found: {ds_path}")
        print(f"Config points to: {ds_cfg['path']}")
        print("Re-run inside Docker: python scripts/download_and_convert.py")
        return None

    weights = ensure_pretrained("yolo11n-pose.pt")
    model = YOLO(weights)

    results = model.train(
        data=str(Path("configs/dataset_calibration.yaml").resolve()),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=640,
        device=args.device,
        project="runs/calibration",
        name="yolo11n_board_calibration",
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

    print(f"\n[TRAIN] Best model: 'runs/calibration/yolo11n_board_calibration/weights/best.pt'")
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
    parser.add_argument("--gpu", type=str, default="0", help="GPU device(s): '0', '0,1,2,3', or 'cpu'")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Path to resume training from")
    args = parser.parse_args()
    args.device = parse_device(args.gpu)
    if isinstance(args.device, list) and args.batch == 16:
        args.batch = len(args.device) * 16
        print(f"[DDP] {len(args.device)} GPUs detected, auto-scaling batch to {args.batch}")

    if args.validate_only:
        model_path = args.model or "runs/calibration/yolo11n_board_calibration/weights/best.pt"
        validate(model_path)
        return

    if args.export:
        model_path = args.model or "runs/calibration/yolo11n_board_calibration/weights/best.pt"
        export_tflite(model_path)
        return

    if args.resume:
        model = YOLO(args.resume)
        results = model.train(resume=True)
        print(f"\n[RESUME] Continued from: {args.resume}")
        print(f"[VAL]   Results: {results}")
        return

    train_calibration(args)
    print("\n[DONE] Training complete.")


if __name__ == "__main__":
    main()
