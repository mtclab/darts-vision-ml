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

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
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

    print(f"Training complete. Best model: {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
