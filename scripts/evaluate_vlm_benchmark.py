#!/usr/bin/env python3
"""
Benchmark multiple VLM models against DeepDarts ground truth.
Supports multi-GPU parallel inference with models running sequentially.

Model adapters handle different APIs:
  - Qwen3.5 (0.8B, 2B): AutoModelForImageTextToText + qwen_vl_utils
  - Qwen3-VL (2B, 4B): Qwen3VLForConditionalGeneration + qwen_vl_utils
  - Granite-Vision-3.2-2B: AutoModelForVision2Seq + apply_chat_template
  - Moondream2: Moondream class + encode_image + answer_question

Usage:
    # Single model
    python scripts/evaluate_vlm_benchmark.py --model Qwen/Qwen3.5-0.8B --all --gpus 0,1,2

    # Multiple models sequentially
    python scripts/evaluate_vlm_benchmark.py --models Qwen/Qwen3.5-0.8B,Qwen/Qwen3.5-2B,Qwen/Qwen3-VL-2B-Instruct --all --gpus 0,1,2

    # Quick test with subset
    python scripts/evaluate_vlm_benchmark.py --models Qwen/Qwen3.5-0.8B,ibm-granite/granite-vision-3.2-2b --num-images 50 --gpus 0
"""

import argparse
import csv
import math
import multiprocessing as mp
from multiprocessing import Queue
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

MODEL_ADAPTERS = {
    "qwen35": "qwen35",
    "qwen3_vl": "qwen3_vl",
    "granite_vision": "granite_vision",
    "moondream": "moondream",
}


def detect_adapter(model_name: str) -> str:
    name_lower = model_name.lower()
    if "qwen3.5" in name_lower or "qwen3_5" in name_lower or name_lower.startswith("qwen/qwen3.5"):
        return "qwen35"
    if "qwen3-vl" in name_lower or "qwen3_vl" in name_lower or "qwen/qwen3-vl" in name_lower:
        return "qwen3_vl"
    if "granite" in name_lower or "ibm-granite" in name_lower:
        return "granite_vision"
    if "moondream" in name_lower:
        return "moondream"
    return "qwen35"


def parse_dart_response(text: str) -> list:
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
                "path": str(img_path),
                "folder": row.img_folder,
                "name": row.img_name,
                "xy": row.xy,
                "bbox": row.bbox,
            })

    return val_images


def compute_ground_truth(xy_list) -> list:
    import cv2
    from dart_board import DEEPDARTS_CAL_MM

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

    img_w, img_h = 800, 800
    cal_names = ["D20", "D6", "D3", "D11"]

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


def worker_fn(gpu_id, model_name, adapter_type, image_chunk, result_queue, max_tokens):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch
    torch.cuda.set_device(0)

    print(f"[GPU {gpu_id}] Loading {model_name} (adapter: {adapter_type})...")
    t0 = time.time()

    tokenizer = None
    pvi = None

    if adapter_type == "qwen35":
        from transformers import AutoProcessor, AutoModelForImageTextToText
        from qwen_vl_utils import process_vision_info

        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        pvi = process_vision_info

    elif adapter_type == "qwen3_vl":
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info

        processor = AutoProcessor.from_pretrained(model_name)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        pvi = process_vision_info

    elif adapter_type == "granite_vision":
        from transformers import AutoProcessor, AutoModelForVision2Seq

        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    elif adapter_type == "moondream":
        from transformers import AutoTokenizer
        from moondream.hf import Moondream

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = Moondream.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        ).to("cuda:0")
        model.eval()
        processor = None

    else:
        result_queue.put({"error": f"Unknown adapter type: {adapter_type}"})
        return

    load_time = time.time() - t0
    vram_mb = 0
    if torch.cuda.is_available():
        vram_mb = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"[GPU {gpu_id}] Loaded {model_name} in {load_time:.1f}s, VRAM: {vram_mb:.0f} MB, processing {len(image_chunk)} images")

    for entry in image_chunk:
        img_path = Path(entry["path"])
        xy_list = entry["xy"]

        gt_scores = compute_ground_truth(xy_list)
        if not gt_scores:
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        try:
            response, inf_time = _run_inference(
                model, processor, image, adapter_type, max_tokens,
                tokenizer=tokenizer, process_vision_info=pvi,
            )
        except Exception as e:
            print(f"[GPU {gpu_id}] Error on {img_path.name}: {e}")
            continue

        gt_labels = [s.label for s in gt_scores]
        gt_rings = [s.ring for s in gt_scores]
        gt_segments = [s.segment for s in gt_scores]

        parsed = parse_dart_response(response)
        pred_rings = [r for r, s in parsed if r != "outside"]
        pred_segments = [s for r, s in parsed if r != "outside"]

        n_gt = len(gt_scores)
        n_pred = len(pred_rings)

        dart_count_correct = (n_pred == n_gt)

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

        result_queue.put({
            "image": img_path.name,
            "n_gt_darts": n_gt,
            "n_pred_darts": n_pred,
            "dart_count_correct": dart_count_correct,
            "n_gt_darts_val": n_gt,
            "seg_matches": seg_matches,
            "ring_matches": ring_matches,
            "full_matches": full_matches,
            "gt_labels": ", ".join(gt_labels),
            "pred_response": response.strip(),
            "pred_parsed": ", ".join(
                [f"{'T' if r=='triple' else 'D' if r=='double' else 'S' if r=='single' else r}{s}" for r, s in parsed]
            ),
            "inference_time_s": round(inf_time, 3),
            "gpu": gpu_id,
        })

    del model
    if processor is not None:
        del processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[GPU {gpu_id}] Done with {model_name}")


def _run_inference(model, processor, image: Image.Image, adapter_type: str, max_tokens: int, tokenizer=None, process_vision_info=None):
    import torch

    t0 = time.time()

    if adapter_type == "qwen35":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPT_STRUCTURED},
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

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    elif adapter_type == "qwen3_vl":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPT_STRUCTURED},
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

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    elif adapter_type == "granite_vision":
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPT_STRUCTURED},
                ],
            },
        ]
        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)

        response = processor.decode(output[0], skip_special_tokens=True)

    elif adapter_type == "moondream":
        image_embeds = model.encode_image(image)
        result = model.query(image_embeds, PROMPT_STRUCTURED)
        response = result["answer"] if isinstance(result, dict) else str(result)

    else:
        raise ValueError(f"Unknown adapter: {adapter_type}")

    inf_time = time.time() - t0
    return response, inf_time


def evaluate_model(model_name: str, adapter_type: str, val_images: list, gpu_ids: list, max_tokens: int, output_path: Path):
    n_gpus = len(gpu_ids)
    model_short = model_name.split("/")[-1]

    print(f"\n{'='*60}")
    print(f"EVALUATING: {model_name} ({adapter_type})")
    print(f"{'='*60}")
    print(f"Images: {len(val_images)}, GPUs: {gpu_ids}")

    chunks = [[] for _ in range(n_gpus)]
    for i, entry in enumerate(val_images):
        chunks[i % n_gpus].append(entry)

    for i, chunk in enumerate(chunks):
        print(f"  GPU {gpu_ids[i]}: {len(chunk)} images")

    result_queue = Queue()
    processes = []

    t0_total = time.time()
    for i, gpu_id in enumerate(gpu_ids):
        p = mp.Process(
            target=worker_fn,
            args=(gpu_id, model_name, adapter_type, chunks[i], result_queue, max_tokens),
        )
        p.start()
        processes.append(p)

    total_expected = sum(len(c) for c in chunks)
    results = []
    pbar = tqdm(total=total_expected, desc=f"  {model_short}")

    while any(p.is_alive() for p in processes):
        while not result_queue.empty():
            try:
                result = result_queue.get_nowait()
                if "error" in result:
                    print(f"[ERROR] {result['error']}")
                    continue
                results.append(result)
                pbar.update(1)
            except Exception:
                break
        time.sleep(0.1)

    while not result_queue.empty():
        try:
            result = result_queue.get_nowait()
            if "error" not in result:
                results.append(result)
                pbar.update(1)
        except Exception:
            break

    for p in processes:
        p.join(timeout=30)

    pbar.close()
    total_time = time.time() - t0_total

    total_gt_darts = 0
    total_correct_segment = 0
    total_correct_ring = 0
    total_correct_full = 0
    total_dart_count_correct = 0
    total_images = 0
    total_inference_time = 0

    csv_results = []
    for r in results:
        n_gt = r["n_gt_darts_val"]
        total_gt_darts += n_gt
        total_correct_segment += r["seg_matches"]
        total_correct_ring += r["ring_matches"]
        total_correct_full += r["full_matches"]
        if r["dart_count_correct"]:
            total_dart_count_correct += 1
        total_images += 1
        total_inference_time += r["inference_time_s"]

        csv_results.append({
            "image": r["image"],
            "n_gt_darts": r["n_gt_darts"],
            "n_pred_darts": r["n_pred_darts"],
            "dart_count_correct": r["dart_count_correct"],
            "gt_labels": r["gt_labels"],
            "pred_response": r["pred_response"],
            "pred_parsed": r["pred_parsed"],
            "inference_time_s": r["inference_time_s"],
            "gpu": r["gpu"],
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        if csv_results:
            writer = csv.DictWriter(f, fieldnames=csv_results[0].keys())
            writer.writeheader()
            writer.writerows(csv_results)

    avg_inf = total_inference_time / max(total_images, 1)
    print(f"\n{'='*60}")
    print(f"RESULTS: {model_name} on {total_images} images ({n_gpus} GPU(s))")
    print(f"{'='*60}")
    print(f"Total darts (ground truth): {total_gt_darts}")
    print(f"Dart count accuracy:         {total_dart_count_correct}/{total_images} = {total_dart_count_correct/max(total_images,1)*100:.1f}%")
    print(f"Segment accuracy:            {total_correct_segment}/{total_gt_darts} = {total_correct_segment/max(total_gt_darts,1)*100:.1f}%")
    print(f"Ring accuracy:               {total_correct_ring}/{total_gt_darts} = {total_correct_ring/max(total_gt_darts,1)*100:.1f}%")
    print(f"Full score accuracy:         {total_correct_full}/{total_gt_darts} = {total_correct_full/max(total_gt_darts,1)*100:.1f}%")
    print(f"Avg inference time:          {avg_inf:.2f}s")
    print(f"Wall clock time:            {total_time:.1f}s")
    print(f"Throughput:                 {total_images/max(total_time,1):.1f} images/s")
    print(f"Output saved to:             {output_path}")

    return {
        "model": model_name,
        "model_short": model_short,
        "adapter": adapter_type,
        "total_images": total_images,
        "total_gt_darts": total_gt_darts,
        "dart_count_accuracy": total_dart_count_correct / max(total_images, 1) * 100,
        "segment_accuracy": total_correct_segment / max(total_gt_darts, 1) * 100,
        "ring_accuracy": total_correct_ring / max(total_gt_darts, 1) * 100,
        "full_score_accuracy": total_correct_full / max(total_gt_darts, 1) * 100,
        "avg_inference_time": avg_inf,
        "wall_clock_time": total_time,
        "throughput": total_images / max(total_time, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark VLM models on DeepDarts (multi-GPU)")
    parser.add_argument("--model", default=None, help="Single model name (e.g., Qwen/Qwen3.5-0.8B)")
    parser.add_argument("--models", default=None, help="Comma-separated model names for sequential benchmark")
    parser.add_argument("--num-images", type=int, default=200, help="Number of images to evaluate")
    parser.add_argument("--all", action="store_true", help="Evaluate ALL val images (~1070)")
    parser.add_argument("--output-dir", default="results", help="Output directory for CSV results")
    parser.add_argument("--gpus", default=None, help="GPU IDs, comma-separated (e.g., 0,1,2). Default: auto-detect")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for image selection")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max new tokens per inference")
    args = parser.parse_args()

    if args.all:
        args.num_images = 0

    if args.models:
        model_list = [m.strip() for m in args.models.split(",")]
    elif args.model:
        model_list = [args.model]
    else:
        model_list = ["Qwen/Qwen3.5-0.8B"]

    if args.gpus:
        gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
    else:
        gpu_ids = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        if not gpu_ids:
            print("[ERROR] No GPUs found. Use --gpus to specify.")
            return

    print(f"[SETUP] Using {len(gpu_ids)} GPU(s): {gpu_ids}")
    print(f"[SETUP] Models to evaluate: {model_list}")

    results_dir = Path(args.output_dir)
    results_dir.mkdir(exist_ok=True)

    val_images = get_val_images()
    if not val_images:
        print("[ERROR] No val images found")
        return

    if args.num_images > 0 and args.num_images < len(val_images):
        random.seed(args.seed)
        val_images = random.sample(val_images, args.num_images)

    all_results = []
    for model_name in model_list:
        adapter_type = detect_adapter(model_name)
        model_short = model_name.split("/")[-1]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"{model_short}_baseline_{timestamp}.csv"

        result = evaluate_model(
            model_name=model_name,
            adapter_type=adapter_type,
            val_images=val_images,
            gpu_ids=gpu_ids,
            max_tokens=args.max_tokens,
            output_path=output_path,
        )
        all_results.append(result)

    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print(f"BENCHMARK COMPARISON ({len(all_results)} models, {len(val_images)} images)")
        print(f"{'='*80}")
        header = f"{'Model':<30} {'DartCnt':>8} {'Segment':>8} {'Ring':>8} {'Full':>8} {'AvgInf':>8} {'Img/s':>8}"
        print(header)
        print("-" * len(header))
        for r in all_results:
            print(f"{r['model_short']:<30} {r['dart_count_accuracy']:>7.1f}% {r['segment_accuracy']:>7.1f}% {r['ring_accuracy']:>7.1f}% {r['full_score_accuracy']:>7.1f}% {r['avg_inference_time']:>7.2f}s {r['throughput']:>7.1f}/s")
        print(f"{'='*80}")

    print(f"\nDone. Results in {results_dir}/")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()