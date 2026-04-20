#!/usr/bin/env python3
"""
Legacy entry point — delegates to evaluate_vlm_benchmark.py.

All evaluation logic has moved to evaluate_vlm_benchmark.py which supports
multiple model families (Qwen3.5, Qwen3-VL, Granite, Moondream).
"""

import sys
import subprocess

args = sys.argv[1:]

if "--model" not in args and "--models" not in args:
    args = ["--model", "Qwen/Qwen3.5-0.8B"] + args

subprocess.run([sys.executable, __file__.replace("evaluate_qwen_dataset.py", "evaluate_vlm_benchmark.py")] + args)