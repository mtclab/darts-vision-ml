#!/usr/bin/env python3
"""
Systematic evaluation of Qwen3.5-VL models against DeepDarts ground truth.

Reads YOLO pose labels from the val split, computes ground truth scores,
sends images to Qwen VL, parses responses, and reports accuracy metrics.

Usage:
    python scripts/evaluate_qwen_dataset.py
    python scripts/evaluate_qwen_dataset.py --model Qwen/Qwen3.5-2B
    python scripts/evaluate_qwen_dataset.py --num-images 200
    python scripts/evaluate_qwen_dataset.py --all
    python scripts/evaluate_qwen_dataset.py --output results/qwen_0.8b_baseline.csv
"""

import argparse
import csv
import json
import math
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from dart_board import (
    DartScore,
    SEGMENT_ORDER,
    SEGMENT_ANGLE,
    classify_dart,
    keypoints_to_scores,
    labels_to_ground_truth,
    format_score,
    score_from_polar,
    ring_from_radius,
    segment_from_angle,
)

RAW_DIR = Path("data/raw/deep-darts/dataset")
PROCESSED_DIR = Path("data/processed")

D1_VAL = ["d1_02_06_2020", "d1_02_16_2020", "d1_02_22_2020"]
D2_VAL = ["d2_02_03_2021", "d2_02_05_2021"]
VAL_FOLDERS = D1_VAL + D2_VAL

PROMPT_STRUCTURED = """Look at this dartboard image. The board has standard segment numbering (clockwise from top): 20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5.

Identify where each dart landed. For each dart, respond with ONLY:
- T<segment> for triple (e.g., T20)
- D<segment> for double (e.g., D16)
- <segment> for single (e.g., 5)
- Bull for outer bull (25)
- Bullseye for inner bull (50)
- Miss for off the board

List each dart on a new line. Example:
T20
S5
D16"""


def parse_dart_response(text: str) -> list:
    patterns = [
        r"[Tt](\d{1,2})",
        r"[Dd](\d{1,2})",
        r"[Ss](\d{1,2})",
        r"\bBullseye\b",
        r"\bBull\b",
        r"\bMiss\b",
        r"\b(\d{1,2})\b",
    ]
    results = []
    lines = text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
        line_clean = re.sub(r"[\*\-\.\,\:\)]", " ", line).strip()
        if not line_clean:
            continue

        m = re.search(r"[Tt](\d{1,2})", line_clean)
        if m:
            seg = int(m.group(1))
            if 1 <= seg <= 20:
                results.append(("triple", seg))
                continue

        m = re.search(r"[Dd](\d{1,2})", line_clean)
        if m:
            seg = int(m.group(1))
            if 1 <= seg <= 20:
                results.append(("double", seg))
                continue

        m = re.search(r"[Ss](\d{1,2})", line_clean)
        if m:
            seg = int(m.group(1))
            if 1 <= seg <= 20:
                results.append(("single", seg))
                continue

        m = re.search(r"\bBullseye\b", line_clean, re.IGNORECASE)
        if m:
            results.append(("bullseye", 0))
            continue

        m = re.search(r"\bBull\b", line_clean, re.IGNORECASE)
        if m and not re.search(r"Bullseye", line_clean, re.IGNORECASE):
            results.append(("bull", 0))
            continue

        m = re.search(r"\b(\d{1,2})\b", line_clean)
        if m:
            seg = int(m.group(1))
            if 1 <= seg <= 20:
                results.append(("single", seg))
                continue

        if re.search(r"miss", line_clean, re.IGNORECASE):
            results.append(("outside", 0))

    return results


def load_model(model_name: str, device: str = "auto"):
    from transformers import AutoProcessor, AutoModelForImageTextToText

    print(f"[LOAD] Loading {model_name}...")
    t0 = time.time()

    processor = AutoProcessor.from_pretrained(model_name)

    device_map = "auto" if device == "auto" else ({"": 0} if device == "cuda" else device)

    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
    )

    load_time = time.time() - t0
    vram_mb = 0
    if torch.cuda.is_available():
        vram_mb = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"[LOAD] Done in {load_time:.1f}s, VRAM: {vram_mb:.0f} MB")

    return model, processor


def run_inference(model, processor, image: Image.Image, prompt: str, max_new_tokens: int = 256):
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    t0 = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.8,
            top_k=20,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    inference_time = time.time() - t0

    return output_text, inference_time


def get_val_images():
    pkl_path = RAW_DIR / "labels.pkl"
    if not pkl_path.exists():
        print(f"[ERROR] {pkl_path} not found. Run download_and_convert.py first.")
        return []

    df = pd.read_pickle(pkl_path)
    val_df = df[df.img_folder.isin(VAL_FOLDERS)].reset_index(drop=True)

    val_images = []
    for _, row in val_df.iterrows():
        img_path = RAW_DIR / "cropped_images" / row.img_folder / row.img_name
        if img_path.exists():
            val_images.append({
                "path": img_path,
                "folder": row.img_folder,
                "name": row.img_name,
                "xy": row.xy,
                "bbox": row.bbox,
            })

    return val_images


def compute_ground_truth(xy_list) -> list:
    if len(xy_list) < 4:
        return []

    cal_keypoints = []
    for i in range(4):
        if i < len(xy_list):
            cal_keypoints.append(tuple(xy_list[i]))
        else:
            cal_keypoints.append(None)

    dart_keypoints = []
    for i in range(4, min(len(xy_list), 7)):
        dart_keypoints.append(tuple(xy_list[i]))

    import cv2
    img_w, img_h = 800, 800

    cal_names = ["D20", "D6", "D3", "D11"]
    from dart_board import DEEPDARTS_CAL_MM

    src_pts = []
    dst_pts_mm = []
    for i, name in enumerate(cal_names):
        if cal_keypoints[i] is not None:
            px, py = cal_keypoints[i]
            src_pts.append([px * img_w, py * img_h])
            dst_pts_mm.append(list(DEEPDARTS_CAL_MM[name]))

    if len(src_pts) < 4:
        return []

    H, _ = cv2.findHomography(
        np.array(src_pts, dtype=np.float32),
        np.array(dst_pts_mm, dtype=np.float32),
    )
    if H is None:
        return []

    scores = []
    for kp in dart_keypoints:
        if kp is None or (kp[0] == 0 and kp[1] == 0):
            continue
        px, py = kp
        pt_px = np.array([px * img_w, py * img_h, 1.0])
        pt_mm = H @ pt_px
        pt_mm = pt_mm[:2] / pt_mm[2]

        radius_mm = math.sqrt(pt_mm[0] ** 2 + pt_mm[1] ** 2)
        angle_deg = math.degrees(math.atan2(pt_mm[0], -pt_mm[1])) % 360.0

        ds = classify_dart(angle_deg, radius_mm)
        scores.append(ds)

    return scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen VL on DeepDarts dataset")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B", help="Model name")
    parser.add_argument("--num-images", type=int, default=200, help="Number of images to evaluate")
    parser.add_argument("--all", action="store_true", help="Evaluate ALL val images (1070)")
    parser.add_argument("--output", default=None, help="Output CSV path (default: results/<model>_<timestamp>.csv)")
    parser.add_argument("--device", default="auto", help="Device: auto, cuda, cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for image selection")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max new tokens per inference")
    args = parser.parse_args()

    if args.all:
        args.num_images = 0  # 0 means all

    model_short = args.model.split("/")[-1]
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"{model_short}_baseline_{timestamp}.csv"

    model, processor = load_model(args.model, args.device)

    val_images = get_val_images()
    if not val_images:
        print("[ERROR] No val images found")
        return

    if args.num_images > 0 and args.num_images < len(val_images):
        random.seed(args.seed)
        val_images = random.sample(val_images, args.num_images)

    print(f"\n[EVAL] Evaluating {len(val_images)} images with {args.model}")
    print(f"[EVAL] Output: {output_path}")

    results = []
    total_gt_darts = 0
    total_correct_segment = 0
    total_correct_ring = 0
    total_correct_full = 0
    total_dart_count_correct = 0
    total_images = 0
    total_inference_time = 0

    for entry in tqdm(val_images, desc="Evaluating"):
        img_path = entry["path"]
        xy_list = entry["xy"]

        gt_scores = compute_ground_truth(xy_list)
        if not gt_scores:
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Cannot open {img_path}: {e}")
            continue

        gt_labels = [s.label for s in gt_scores]
        gt_rings = [s.ring for s in gt_scores]
        gt_segments = [s.segment for s in gt_scores]

        try:
            response, inf_time = run_inference(model, processor, image, PROMPT_STRUCTURED, args.max_tokens)
        except Exception as e:
            print(f"[WARN] Inference failed for {img_path.name}: {e}")
            continue

        total_inference_time += inf_time

        parsed = parse_dart_response(response)
        pred_rings = [r for r, s in parsed if r != "outside"]
        pred_segments = [s for r, s in parsed if r != "outside"]

        n_gt = len(gt_scores)
        n_pred = len(pred_rings)

        dart_count_correct = (n_pred == n_gt)
        if n_pred == n_gt:
            total_dart_count_correct += 1

        seg_matches = 0
        ring_matches = 0
        full_matches = 0
        n_compare = min(n_gt, n_pred)

        for i in range(n_compare):
            if gt_segments[i] == pred_segments[i]:
                seg_matches += 1
            gt_ring_mapped = gt_rings[i]
            pred_ring = pred_rings[i]
            if gt_ring_mapped in ("single", "single_outer"):
                gt_ring_mapped = "single"
            if pred_ring == gt_ring_mapped:
                ring_matches += 1
            if gt_segments[i] == pred_segments[i] and pred_ring == gt_ring_mapped:
                full_matches += 1

        total_gt_darts += n_gt
        total_correct_segment += seg_matches
        total_correct_ring += ring_matches
        total_correct_full += full_matches
        total_images += 1

        results.append({
            "image": img_path.name,
            "n_gt_darts": n_gt,
            "n_pred_darts": n_pred,
            "dart_count_correct": dart_count_correct,
            "gt_labels": ", ".join(gt_labels),
            "pred_response": response.strip(),
            "pred_parsed": ", ".join(
                [f"{'T' if r=='triple' else 'D' if r=='double' else 'S' if r=='single' else r}{s}" for r, s in parsed]
            ),
            "inference_time_s": round(inf_time, 3),
        })

    with open(output_path, "w", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    print(f"\n{'='*60}")
    print(f"RESULTS: {args.model} on {total_images} images")
    print(f"{'='*60}")
    print(f"Total darts (ground truth): {total_gt_darts}")
    print(f"Dart count accuracy:         {total_dart_count_correct}/{total_images} = {total_dart_count_correct/max(total_images,1)*100:.1f}%")
    print(f"Segment accuracy:            {total_correct_segment}/{total_gt_darts} = {total_correct_segment/max(total_gt_darts,1)*100:.1f}%")
    print(f"Ring accuracy:               {total_correct_ring}/{total_gt_darts} = {total_correct_ring/max(total_gt_darts,1)*100:.1f}%")
    print(f"Full score accuracy:         {total_correct_full}/{total_gt_darts} = {total_correct_full/max(total_gt_darts,1)*100:.1f}%")
    print(f"Avg inference time:          {total_inference_time/max(total_images,1):.2f}s")
    print(f"Output saved to:             {output_path}")

    import cv2
    del model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()