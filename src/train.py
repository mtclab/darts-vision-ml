"""
Train YOLOv8n on DeepDarts dataset using DDP across 3 GPUs.

Usage:
    torchrun --nproc_per_node=3 src/train.py \
        --data data/darts.yaml \
        --model yolov8n.pt \
        --epochs 50 \
        --imgsz 800 \
        --batch 24 \
        --project runs/darts
"""

import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/darts.yaml")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--project", default="runs/darts")
    parser.add_argument("--name", default="yolov8n_800")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0,1,2")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load model
    model = YOLO(args.model)
    
    # Train with heavy augmentation for generalization
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
        # Augmentation
        degrees=15,
        translate=0.1,
        scale=0.3,
        shear=0.0,
        perspective=0.0,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        mosaic=1.0,
        mixup=0.0,
        flipud=0.0,
        fliplr=0.5,
        close_mosaic=5,
        # Regularization
        weight_decay=0.0005,
        dropout=0.0,
        # Logging
        plots=True,
        save=True,
        save_period=5,
    )
    
    print(f"Training complete. Best model: {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
