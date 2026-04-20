#!/usr/bin/env python3
"""
LoRA fine-tuning of Qwen3.5-VL for dart board score recognition.

Uses the training data prepared by prepare_qwen_training.py and
LoRA configuration from configs/qwen_lora_config.yaml.

Usage:
    python scripts/train_qwen_lora.py
    python scripts/train_qwen_lora.py --config configs/qwen_lora_config.yaml
    python scripts/train_qwen_lora.py --epochs 5 --lr 1e-4
    python scripts/train_qjen_lora.py --resume runs/qwen_lora/checkpoint-500
"""

import argparse
import json
from pathlib import Path

import yaml
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class DartBoardDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_path: str, processor, max_length: int = 1024):
        self.samples = []
        self.processor = processor
        self.max_length = max_length

        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

        print(f"[DATA] Loaded {len(self.samples)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        messages = sample["messages"]

        user_content = messages[0]["content"]
        assistant_content = messages[1]["content"]

        image_path = None
        text_prompt = ""
        for item in user_content:
            if item["type"] == "image":
                image_path = item["image"]
            elif item["type"] == "text":
                text_prompt = item["text"]

        if image_path and Path(image_path).exists():
            image = Image.open(image_path).convert("RGB")
        else:
            image = Image.new("RGB", (384, 384), (128, 128, 128))

        full_messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text_prompt},
            ]},
            {"role": "assistant", "content": assistant_content},
        ]

        text = self.processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )

        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(full_messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        labels = inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        inputs["labels"] = labels

        return inputs


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune Qwen3.5-VL for dart boards")
    parser.add_argument("--config", default="configs/qwen_lora_config.yaml", help="Config file")
    parser.add_argument("--model", default=None, help="Override model name from config")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--output-dir", default="runs/qwen_lora", help="Output directory")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    config = load_config(args.config)

    model_name = args.model or config["model"]["name"]
    max_length = config["model"].get("max_length", 1024)

    lora_cfg = config["lora"]
    train_cfg = config["training"]
    data_cfg = config["data"]

    epochs = args.epochs or train_cfg["epochs"]
    lr = args.lr or train_cfg["learning_rate"]
    batch_size = args.batch_size or train_cfg["batch_size"]

    print(f"[CONFIG] Model: {model_name}")
    print(f"[CONFIG] LoRA rank: {lora_cfg['rank']}, alpha: {lora_cfg['alpha']}")
    print(f"[CONFIG] Epochs: {epochs}, LR: {lr}, Batch: {batch_size}")
    print(f"[CONFIG] Output: {args.output_dir}")

    processor = AutoProcessor.from_pretrained(model_name)

    print("[LOAD] Loading base model...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    print("[PEFT] Applying LoRA adapters...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("[DATA] Loading datasets...")
    train_dataset = DartBoardDataset(
        data_cfg["train_file"], processor, max_length
    )
    val_dataset = DartBoardDataset(
        data_cfg["val_file"], processor, max_length
    ) if Path(data_cfg["val_file"]).exists() else None

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=lr,
        lr_scheduler_type=train_cfg.get("lr_scheduler", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        fp16=train_cfg.get("fp16", True) and torch.cuda.is_available(),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        logging_steps=10,
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=100 if val_dataset else None,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=val_dataset is not None,
        metric_for_best_model="eval_loss" if val_dataset else None,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print(f"[TRAIN] Starting LoRA fine-tuning for {epochs} epochs...")
    trainer.train(resume_from_checkpoint=args.resume)

    output_dir = Path(args.output_dir)
    merged_dir = output_dir / "merged_model"
    print(f"\n[SAVE] Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(str(merged_dir / "qwen3.5-0.8b-darts-lora"))
    processor.save_pretrained(str(merged_dir / "qwen3.5-0.8b-darts-lora"))
    print(f"[SAVE] Merged model saved to: {merged_dir / 'qwen3.5-0.8b-darts-lora'}")

    lora_dir = output_dir / "lora_weights_only"
    model.save_pretrained(str(lora_dir))
    print(f"[SAVE] LoRA weights saved to: {lora_dir}")

    print(f"\n[DONE] Training complete.")
    print(f"  LoRA weights:  {lora_dir}")
    print(f"  Merged model:  {merged_dir / 'qwen3.5-0.8b-darts-lora'}")
    print(f"\nTo export for Android (LiteRT-LM):")
    print(f"  python scripts/export_qwen_litert.py --model {merged_dir / 'qwen3.5-0.8b-darts-lora'}")


if __name__ == "__main__":
    main()