"""
Train YOLOv8-pose on DeepDarts keypoints using DDP across 3 GPUs.

Usage:
    torchrun --nproc_per_node=3 src/train_pose.py \
        --data data/processed/yolo_pose_darts/pose.yaml \
        --model yolov8n-pose.pt \
        --epochs 100 \
        --imgsz 1280 \
        --batch 10 \
        --project runs/darts_pose
"""

import argparse
import sys
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/yolo_pose_darts/pose.yaml")
    parser.add_argument("--model", default="yolov8n-pose.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--project", default="runs/darts_pose")
    parser.add_argument("--name", default="yolov8n_pose_1280")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr0", type=float, default=0.005)
    parser.add_argument("--lrf", type=float, default=0.005)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0,1,2")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve data path relative to project root
    project_root = Path(__file__).resolve().parent.parent
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = project_root / data_path
    if not data_path.exists():
        print(f"[ERROR] Data config not found: {data_path}")
        print(f"  CWD: {Path.cwd()}")
        print(f"  Project root: {project_root}")
        sys.exit(1)

    model = YOLO(args.model)

    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        exist_ok=True,
        patience=args.patience,
        lr0=args.lr0,
        lrf=args.lrf,
        workers=args.workers,
        device=args.device,
        resume=args.resume,
        seed=args.seed,
        # No heavy aug for keypoints
        degrees=5,
        translate=0.05,
        scale=0.1,
        shear=0.0,
        perspective=0.0,
        hsv_h=0.01,
        hsv_s=0.2,
        hsv_v=0.2,
        mosaic=0.0,
        mixup=0.0,
        flipud=0.0,
        fliplr=0.0,
        close_mosaic=0,
        weight_decay=0.0005,
        dropout=0.0,
        plots=True,
        save=True,
        save_period=5,
    )

    save_dir = getattr(results, "save_dir", Path(args.project) / args.name)
    best = Path(save_dir) / "weights" / "best.pt"
    print(f"Training complete. Best model: {best}")


if __name__ == "__main__":
    main()
