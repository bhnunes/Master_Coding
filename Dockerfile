FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 1. Install System Dependencies
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev git libopenjp2-7-dev \
    libopenjp2-tools openslide-tools texlive-latex-base \
    texlive-latex-recommended texlive-latex-extra \
    texlive-fonts-recommended texlive-fonts-extra \
    && rm -rf /var/lib/apt/lists/*

# 2. Install PyTorch first (Crucial for CUDA compatibility)
RUN pip3 install --no-cache-dir -U --index-url https://download.pytorch.org/whl/cu126 \
    torch==2.8.0 torchvision torchaudio

# 3. Copy and install your requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# 4. Install specific Git repos that aren't on PyPI
RUN pip3 install --no-cache-dir git+https://github.com/qubvel/segmentation_models.pytorch

WORKDIR /workspace