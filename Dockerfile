FROM ultralytics/ultralytics:latest

WORKDIR /workspace

# Layer 1: Core NLP deps (transformers pulls tokenizers, huggingface-hub, accelerate)
RUN pip install --no-cache-dir transformers==5.5.4

# Layer 2: PEFT + datasets (depend on transformers)
RUN pip install --no-cache-dir peft==0.19.1 datasets==4.8.4

# Layer 3: Qwen VL utils (avoid decord extra — source builds hang on some platforms)
RUN pip install --no-cache-dir qwen-vl-utils==0.0.14

# Layer 4: moondream 0.2.1 (provides moondream.hf.Moondream for evaluate_vlm_benchmark.py)
# --no-deps because its deps (torch, transformers, etc.) are already in the image
RUN pip install --no-cache-dir --no-deps moondream==0.2.1

# Layer 5: multiprocess (datasets internal dependency, safe standalone)
RUN pip install --no-cache-dir multiprocess==0.70.19

# Install decord with binary wheel fallback (skip if unavailable — video feat optional)
RUN pip install --no-cache-dir decord || echo "decord install skipped"

ENV PYTHONPATH=/workspace
ENV TRANSFORMERS_CACHE=/workspace/.cache/hf
ENV HF_HOME=/workspace/.cache/hf

CMD ["bash"]