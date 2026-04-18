FROM ultralytics/ultralytics:latest

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/workspace

CMD ["bash"]