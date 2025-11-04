FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# ========== CÀI THƯ VIỆN ==========
RUN apt-get update && apt-get install -y git wget && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir \
    transformers==4.44.0 \
    accelerate==0.33.0 \
    sentencepiece==0.1.99 \
    safetensors==0.4.5 \
    torch==2.2.0 \
    tqdm==4.66.5

# ========== COPY SCRIPT ==========
WORKDIR /workspace
COPY inference_phi3.py /workspace/inference_phi3.py

# ========== CHẠY ==========
CMD ["python", "inference_phi3.py"]
