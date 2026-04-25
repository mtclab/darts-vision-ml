#!/usr/bin/env python3
"""Diagnose why Ultralytics can't find training images."""
from pathlib import Path
import yaml
import sys

def diagnose(data_yaml: str):
    p = Path(data_yaml).resolve()
    print(f"YAML path: {p}")
    if not p.exists():
        print("[FAIL] YAML file does not exist.")
        return

    with open(p) as f:
        cfg = yaml.safe_load(f)

    base = Path(cfg.get("path", p.parent))
    if not base.is_absolute():
        base = p.parent / base
    print(f"Dataset root: {base.resolve()}")

    for split in ["train", "val", "test"]:
        img_dir = base / cfg.get(split, split)
        print(f"\n[{split}] Image dir: {img_dir.resolve()}")
        if not img_dir.exists():
            print("  [FAIL] Directory does not exist.")
            continue
        files = list(img_dir.iterdir())
        print(f"  Total entries: {len(files)}")
        imgs = [f for f in files if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
        print(f"  Image files: {len(imgs)}")
        if imgs:
            print(f"  Sample: {imgs[0].name}")
        if not imgs:
            print("  [FAIL] No recognized image files.")
            # Check for broken symlinks
            symlinks = [f for f in files if f.is_symlink() and not f.exists()]
            if symlinks:
                print(f"  Broken symlinks: {len(symlinks)}")
                print(f"    Example -> {symlinks[0].readlink()}")

    # Check labels
    for split in ["train", "val"]:
        label_dir = base / "labels" / split
        if label_dir.exists():
            n = len(list(label_dir.glob("*.txt")))
            print(f"\n[{split}] Labels: {n} .txt files in {label_dir}")
        else:
            print(f"\n[{split}] Labels dir missing: {label_dir}")

if __name__ == "__main__":
    diagnose(sys.argv[1] if len(sys.argv) > 1 else "data/processed/yolo_detect_tight/darts.yaml")
