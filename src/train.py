"""
Train YOLOv8 on DeepDarts dataset using DDP across 3 GPUs.

Quick baseline:
    torchrun --nproc_per_node=3 src/train.py \
        --data data/processed/yolo_detect_deepdarts/darts.yaml \
        --model yolov8n.pt \
        --epochs 50 \
        --imgsz 800 \
        --batch 24 \
        --project runs/darts

Production (tight bbox, no mosaic):
    torchrun --nproc_per_node=3 src/train.py \
        --data data/processed/yolo_detect_tight/darts.yaml \
        --model yolov8s.pt \
        --imgsz 1280 \
        --batch 10 \
        --epochs 100 \
        --mosaic 0 \
        --fliplr 0
"""

import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser()
    # Core training
    parser.add_argument("--data", default="data/processed/yolo_detect_deepdarts/darts.yaml")
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

    # Augmentation (default to older baseline values; override from CLI)
    parser.add_argument("--degrees", type=float, default=15)
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=0.3)
    parser.add_argument("--shear", type=float, default=0.0)
    parser.add_argument("--perspective", type=float, default=0.0)
    parser.add_argument("--hsv_h", type=float, default=0.015)
    parser.add_argument("--hsv_s", type=float, default=0.5)
    parser.add_argument("--hsv_v", type=float, default=0.3)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--close_mosaic", type=int, default=5)

    # Regularization
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--dropout", type=float, default=0.0)

    # Logging
    parser.add_argument("--plots", action="store_true", default=True)
    parser.add_argument("--save", action="store_true", default=True)
    parser.add_argument("--save_period", type=int, default=5)

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
        # Augmentation
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        mosaic=args.mosaic,
        mixup=args.mixup,
        flipud=args.flipud,
        fliplr=args.fliplr,
        close_mosaic=args.close_mosaic,
        # Regularization
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        # Logging
        plots=args.plots,
        save=args.save,
        save_period=args.save_period,
    )

    print(f"Training complete. Best model: {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
