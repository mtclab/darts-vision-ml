#!/usr/bin/env python3
"""
Prepare DeepDarts training data for Qwen3.5-VL fine-tuning.

Converts YOLO pose labels to VLM instruction format (JSONL) suitable for
LoRA fine-tuning with TRL's SFTTrainer. Each sample contains an image
with a structured prompt asking the model to identify dart scores,
and the ground truth answer derived from keypoint annotations.

Usage:
    python scripts/prepare_qwen_training.py
    python scripts/prepare_qwen_training.py --num-prompts 3
    python scripts/prepare_qwen_training.py --output-dir data/processed/vlm_train
"""

import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from dart_board import (
    SEGMENT_ORDER,
    classify_dart,
    format_score,
    score_from_polar,
    ring_from_radius,
    segment_from_angle,
    DEEPDARTS_CAL_MM,
    BULLSEYE_OUTER,
    BULL_OUTER,
    DOUBLE_OUTER,
)

RAW_DIR = Path("data/raw/deep-darts/dataset")
PROCESSED_DIR = Path("data/processed")

D1_VAL = ["d1_02_06_2020", "d1_02_16_2020", "d1_02_22_2020"]
D1_TEST = [
    "d1_03_03_2020", "d1_03_19_2020", "d1_03_23_2020",
    "d1_03_27_2020", "d1_03_28_2020", "d1_03_30_2020", "d1_03_31_2020",
]
D2_VAL = ["d2_02_03_2021", "d2_02_05_2021"]
D2_TEST = ["d2_03_03_2020", "d2_02_10_2021", "d2_02_03_2021_2"]

PROMPT_TEMPLATES = [
    "Look at this dartboard image. Identify where each dart landed. For each dart, respond with ONLY the score label (T20, D16, 5, Bull, Bullseye, or Miss). List each dart on a new line.",
    "What dart scores are visible on this board? Use format: T<segment> for triple, D<segment> for double, <segment> for single, Bull, Bullseye, or Miss. One per line.",
    "This is a standard dartboard. List all dart scores you can see. Use T for triple, D for double, just the number for single, Bull for 25, Bullseye for 50.",
]


def compute_ground_truth_scores(xy_list):
    if len(xy_list) < 4:
        return []

    img_w, img_h = 800, 800

    cal_kps = [(xy_list[i][0], xy_list[i][1]) for i in range(4)]
    dart_kps = [(xy_list[i][0], xy_list[i][1]) for i in range(4, min(len(xy_list), 7))]

    src_pts = []
    dst_pts_mm = []
    cal_names = ["D20", "D6", "D3", "D11"]
    for i, name in enumerate(cal_names):
        px, py = cal_kps[i]
        src_pts.append([px * img_w, py * img_h])
        dst_pts_mm.append(list(DEEPDARTS_CAL_MM[name]))

    H, _ = cv2.findHomography(
        np.array(src_pts, dtype=np.float32),
        np.array(dst_pts_mm, dtype=np.float32),
    )
    if H is None:
        return []

    scores = []
    for kp in dart_kps:
        px, py = kp[0], kp[1]
        if px == 0 and py == 0:
            continue
        pt_px = np.array([px * img_w, py * img_h, 1.0])
        pt_mm = H @ pt_px
        pt_mm = pt_mm[:2] / pt_mm[2]

        radius_mm = math.sqrt(pt_mm[0] ** 2 + pt_mm[1] ** 2)
        angle_deg = math.degrees(math.atan2(pt_mm[0], -pt_mm[1])) % 360.0

        ds = classify_dart(angle_deg, radius_mm)
        scores.append(ds)

    return scores


def create_entry(img_path: Path, scores: list, prompt_idx: int) -> dict:
    if not scores:
        prompt = PROMPT_TEMPLATES[prompt_idx % len(PROMPT_TEMPLATES)]
        answer = "No darts visible on this board."
    else:
        prompt = PROMPT_TEMPLATES[prompt_idx % len(PROMPT_TEMPLATES)]
        answer = "\n".join(s.label for s in scores)

    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(img_path)},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": answer,
            },
        ],
        "ground_truth": [s.label for s in scores],
        "n_darts": len(scores),
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare VLM training data from DeepDarts")
    parser.add_argument("--output-dir", default="data/processed/vlm_train", help="Output directory")
    parser.add_argument("--num-prompts", type=int, default=3, help="Number of prompt variations per image")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction of train set for validation")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pkl_path = RAW_DIR / "labels.pkl"
    if not pkl_path.exists():
        print(f"[ERROR] {pkl_path} not found. Run download_and_convert.py first.")
        return

    df = pd.read_pickle(pkl_path)
    val_folders = D1_VAL + D2_VAL
    test_folders = D1_TEST + D2_TEST

    train_entries = []
    val_entries = []

    random.seed(args.seed)

    print(f"[INFO] Processing {len(df)} images...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_path = RAW_DIR / "cropped_images" / row.img_folder / row.img_name
        if not img_path.exists():
            continue

        xy_list = row.xy
        scores = compute_ground_truth_scores(xy_list)

        is_test = row.img_folder in test_folders
        is_val = row.img_folder in val_folders

        for p in range(args.num_prompts):
            entry = create_entry(img_path, scores, p)
            if is_test:
                continue
            elif is_val:
                val_entries.append(entry)
            else:
                train_entries.append(entry)

    random.shuffle(train_entries)

    n_val_from_train = int(len(train_entries) * args.val_split / args.num_prompts) * args.num_prompts
    val_from_train = train_entries[:n_val_from_train]
    train_entries = train_entries[n_val_from_train:]
    val_entries.extend(val_from_train)

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"

    with open(train_path, "w") as f:
        for entry in train_entries:
            json.dump(entry, f)
            f.write("\n")

    with open(val_path, "w") as f:
        for entry in val_entries:
            json.dump(entry, f)
            f.write("\n")

    train_darts = sum(e["n_darts"] for e in train_entries)
    val_darts = sum(e["n_darts"] for e in val_entries)
    train_with_darts = sum(1 for e in train_entries if e["n_darts"] > 0)
    val_with_darts = sum(1 for e in val_entries if e["n_darts"] > 0)

    print(f"\n{'='*60}")
    print(f"TRAINING DATA SUMMARY")
    print(f"{'='*60}")
    print(f"Train entries:         {len(train_entries)}")
    print(f"  With darts:          {train_with_darts}")
    print(f"  Total dart labels:   {train_darts}")
    print(f"Val entries:           {len(val_entries)}")
    print(f"  With darts:          {val_with_darts}")
    print(f"  Total dart labels:   {val_darts}")
    print(f"Prompt variations:     {args.num_prompts}")
    print(f"Output:                {output_dir}")
    print(f"  train.jsonl:         {train_path}")
    print(f"  val.jsonl:           {val_path}")


if __name__ == "__main__":
    main()