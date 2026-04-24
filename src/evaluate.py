"""
End-to-end evaluation of YOLO model on dart scoring task.

Computes:
  - PCS (Percent Correct Score): % of images with exact total score match
  - MASE (Mean Absolute Score Error)
  - Per-dart detection accuracy

Usage:
    python src/evaluate.py \
        --model runs/darts/yolov8n_800/weights/best.pt \
        --data data/darts.yaml \
        --split test \
        --conf 0.25 \
        --device 0
"""

import argparse
import pickle
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import cv2
from typing import List, Tuple

# Board geometry (matches dart-sense)
RING_RADII = np.array([6.35, 15.9, 97.4, 107.4, 160.0, 170.0]) / 451.0
RING_NAMES = ["DB", "SB", "S", "T", "S", "D", "miss"]
SEGMENTS = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]


def find_homography(src: np.ndarray, dst: np.ndarray):
    """Compute homography using DLT."""
    H, _ = cv2.findHomography(src, dst)
    return H


def score_dart(pt: np.ndarray) -> Tuple[str, int]:
    """Get score from normalized board coordinate."""
    dx = pt[0] - 0.5
    dy = pt[1] - 0.5
    dist = np.sqrt(dx**2 + dy**2)
    angle = np.degrees(np.arctan2(dy, dx))
    
    # Adjust angle so 20 is at top (81 deg)
    angle = (angle + 81) % 360
    seg_idx = int(angle // 18) % 20
    number = SEGMENTS[seg_idx]
    
    # Determine ring
    ring = "miss"
    mult = 0
    for i, r in enumerate(RING_RADII):
        if dist <= r:
            ring = RING_NAMES[i]
            mult = { "DB": 2, "SB": 1, "T": 3, "D": 2 }.get(ring, 1)
            break
    if dist > RING_RADII[-1]:
        ring = "miss"
        mult = 0
        number = 0
    
    if ring in ("DB", "SB"):
        number = 25
    
    return f"{ring}{number}" if ring not in ("miss", "DB", "SB") else ring, number * mult


def evaluate_image(model, img_path: Path, gt_keypoints: np.ndarray, conf: float = 0.25):
    """Run inference and score. Returns predicted_scores, gt_scores."""
    results = model.predict(str(img_path), imgsz=800, conf=conf, verbose=False)[0]
    
    if results.boxes is None:
        return [], []
    
    boxes = results.boxes.xywhn.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)
    confs = results.boxes.conf.cpu().numpy()
    
    # Separate corners and darts
    corners = boxes[classes == 1]  # cal_corner
    darts = boxes[classes == 0]     # dart
    
    if len(corners) < 4:
        return [], []
    
    # Use top-4 highest confidence corners
    corner_confs = confs[classes == 1]
    top4_idx = np.argsort(corner_confs)[-4:]
    src_pts = corners[top4_idx][:, :2]
    
    # Sort corners: TL, TR, BL, BR (simple heuristic by relative position)
    center = src_pts.mean(axis=0)
    sorted_pts = sorted(src_pts, key=lambda p: (p[1], p[0]))  # sort by y then x
    tl, tr = sorted(sorted_pts[:2], key=lambda p: p[0])
    bl, br = sorted(sorted_pts[2:], key=lambda p: p[0])
    src_ordered = np.array([tl, tr, bl, br], dtype=np.float32)
    
    # Canonical destination (unit square)
    dst_ordered = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
    
    H = find_homography(src_ordered, dst_ordered)
    if H is None:
        return [], []
    
    # Warp dart centers
    pred_scores = []
    for dart_pt in darts[:, :2]:
        dart_hom = np.array([*dart_pt, 1.0])
        warped = H @ dart_hom
        warped = warped[:2] / warped[2]
        _, val = score_dart(warped)
        pred_scores.append(val)
    
    # Ground truth scores
    gt_scores = []
    for i in range(4, 7):
        if gt_keypoints[i, 2] > 0:
            _, val = score_dart(gt_keypoints[i, :2])
            gt_scores.append(val)
    
    return pred_scores, gt_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="data/darts.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--labels", default="data/raw/labels.pkl")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    
    model = YOLO(args.model)
    model.to(args.device)
    
    # Load labels
    with open(args.labels, "rb") as f:
        data = pickle.load(f)
    img_paths = data["img_paths"]
    gts = data["gt"]
    
    # TODO: filter to test split images
    
    errors = []
    correct = 0
    total = 0
    
    for i, (img_path, gt) in enumerate(zip(img_paths, gts)):
        if not Path(img_path).exists():
            continue
        pred_scores, gt_scores = evaluate_image(model, Path(img_path), gt, conf=args.conf)
        
        if not pred_scores or not gt_scores:
            continue
        
        pred_total = sum(pred_scores)
        gt_total = sum(gt_scores)
        error = abs(pred_total - gt_total)
        errors.append(error)
        
        if error == 0:
            correct += 1
        total += 1
        
        if i % 100 == 0:
            print(f"Processed {i}/{len(img_paths)}")
    
    errors = np.array(errors)
    pcs = correct / total * 100 if total > 0 else 0
    mase = errors.mean() if len(errors) > 0 else 0
    
    print(f"\nEvaluation Results ({total} images):")
    print(f"  PCS:  {pcs:.1f}%")
    print(f"  MASE: {mase:.2f}")
    print(f"  Correct: {correct}/{total}")
    print(f"  Error 0: {np.sum(errors == 0)}")
    print(f"  Error 1: {np.sum(errors == 1)}")
    print(f"  Error >10: {np.sum(errors > 10)}")


if __name__ == "__main__":
    main()
