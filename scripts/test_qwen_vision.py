#!/usr/bin/env python3
"""
Interactive test of Qwen3.5-VL models on dart board images.

Tests multiple prompt strategies and measures accuracy + latency.
Supports Qwen3.5-0.8B and Qwen3.5-2B (or any Qwen VL model).

Usage:
    python scripts/test_qwen_vision.py --image path/to/image.jpg
    python scripts/test_qwen_vision.py --image path/to/image.jpg --model Qwen/Qwen3.5-2B
    python scripts/test_qwen_vision.py --image screenshot1.jpg screenshot2.jpg
    python scripts/test_qwen_vision.py --image image.jpg --prompt structured
    python scripts/test_qwen_vision.py --image image.jpg --prompt all
"""

import argparse
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


PROMPTS = {
    "simple": "What dart scores are visible on this dartboard?",

    "structured": """Look at this dartboard image. The board has the standard segment numbering (clockwise from top): 20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5.

Identify where each dart landed. For each dart, respond with:
- The segment number (1-20)
- The ring: S (single), D (double), or T (triple)
- Or: Bull (25) or Bullseye (50)
- Or: Miss (off the board)

Format your response as a simple list, one per line. Example:
T20
S5
D16""",

    "fewshot": """You are a darts scoring assistant. I will show you a dartboard image and you must identify the score of each dart.

The board has 20 segments numbered clockwise from top: 20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5.

Score format:
- T20 = triple 20 (60 points)
- D20 = double 20 (40 points)
- 20 = single 20 (just the number)
- Bull = outer bull (25 points)
- Bullseye = inner bull (50 points)
- Miss = off the board

Now look at this dartboard and list each dart's score, one per line:""",

    "coordinate": """Look at this dartboard image carefully.

The board segments are numbered clockwise from top: 20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5.

1. How many darts are visible on the board?
2. For each dart, what segment and ring did it land in?

Respond in this exact format:
count: <number>
dart1: <score> (e.g., T20, S5, D16, Bull, Bullseye, Miss)
dart2: <score>
...""",
}


def load_model(model_name: str, device: str = "auto"):
    print(f"[LOAD] Loading {model_name}...")
    t0 = time.time()

    processor = AutoProcessor.from_pretrained(model_name)

    if device == "auto":
        device_map = "auto"
    elif device == "cuda":
        device_map = {"": 0}
    else:
        device_map = device

    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
    )

    load_time = time.time() - t0
    vram_mb = 0
    if torch.cuda.is_available():
        vram_mb = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"[LOAD] Model loaded in {load_time:.1f}s, VRAM: {vram_mb:.0f} MB")

    return model, processor


def run_inference(model, processor, image: Image.Image, prompt_text: str, max_new_tokens: int = 512):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
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
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            do_sample=True,
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
    num_tokens = len(generated_ids_trimmed[0])
    tokens_per_sec = num_tokens / inference_time if inference_time > 0 else 0

    return output_text, inference_time, num_tokens, tokens_per_sec


def process_vision_info(messages):
    try:
        from qwen_vl_utils import process_vision_info as _pvi
        return _pvi(messages)
    except ImportError:
        images = []
        videos = []
        for msg in messages:
            for content in msg.get("content", []):
                if content.get("type") == "image":
                    img = content.get("image")
                    if isinstance(img, str):
                        img = Image.open(img).convert("RGB")
                    images.append(img)
        return images, videos


def main():
    parser = argparse.ArgumentParser(description="Test Qwen VL models on dart board images")
    parser.add_argument("--image", nargs="+", required=True, help="Path(s) to dart board image(s)")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B", help="Model name (default: Qwen/Qwen3.5-0.8B)")
    parser.add_argument("--prompt", default="structured", choices=list(PROMPTS.keys()) + ["all"], help="Prompt strategy")
    parser.add_argument("--device", default="auto", help="Device: auto, cuda, cpu")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max new tokens to generate")
    parser.add_argument("--resize", type=int, default=None, help="Resize image to NxN before sending (default: no resize)")
    args = parser.parse_args()

    model, processor = load_model(args.model, args.device)

    prompt_names = list(PROMPTS.keys()) if args.prompt == "all" else [args.prompt]

    for img_path in args.image:
        img_path = Path(img_path)
        if not img_path.exists():
            print(f"[ERROR] Image not found: {img_path}")
            continue

        image = Image.open(img_path).convert("RGB")
        if args.resize:
            image = image.resize((args.resize, args.resize))

        print(f"\n{'='*80}")
        print(f"IMAGE: {img_path.name} ({image.size[0]}x{image.size[1]})")
        print(f"{'='*80}")

        for prompt_name in prompt_names:
            prompt_text = PROMPTS[prompt_name]
            print(f"\n--- Prompt: {prompt_name} ---")

            try:
                response, inf_time, n_tokens, tps = run_inference(
                    model, processor, image, prompt_text, args.max_tokens
                )
                print(f"Response ({inf_time:.2f}s, {n_tokens} tokens, {tps:.1f} tok/s):")
                print(response)
            except Exception as e:
                print(f"[ERROR] Inference failed: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()