# Base image with CUDA, cuDNN, and PyTorch preinstalled
FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

# System deps for building native extensions (e.g., DeepSpeed) and common tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        wget \
        libaio-dev \
        openmpi-bin \
        libopenmpi-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/O-LoRA

# Install Python deps first to leverage Docker layer caching
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu118 -r requirements.txt

# Copy project code
COPY . .

# Make local packages importable
ENV PYTHONPATH=/workspace/O-LoRA/src \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
    HF_HOME=/workspace/.cache/huggingface \
    WANDB_DISABLED=false

CMD ["bash"]
