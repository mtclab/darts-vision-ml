FROM ultralytics/ultralytics:latest

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision ultralytics opencv-python numpy Pillow PyYAML matplotlib tqdm pandas scikit-learn
RUN pip install --no-cache-dir multiprocess==0.70.11
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/workspace
ENV TRANSFORMERS_CACHE=/workspace/.cache/hf
ENV HF_HOME=/workspace/.cache/hf

CMD ["bash"]