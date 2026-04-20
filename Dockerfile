FROM ultralytics/ultralytics:latest

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/workspace
ENV TRANSFORMERS_CACHE=/workspace/.cache/hf
ENV HF_HOME=/workspace/.cache/hf

CMD ["bash"]