# CUDA 12.6 base image (matches cu126 wheels)
FROM nvidia/cuda:12.6.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    git \
    wget \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.12
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y python3.12 python3.12-venv python3.12-distutils

# Make python3 point to 3.12
RUN ln -sf /usr/bin/python3.12 /usr/bin/python3
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3

# Upgrade pip
RUN pip install --upgrade pip

# Copy only requirements first (for caching)
COPY requirements.txt .

# Install PyTorch GPU (CUDA 12.6)
RUN pip install torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu126

# Install torch-scatter (must match torch + cu126)
RUN pip install torch-scatter \
    -f https://pytorch-geometric.com/whl/torch-2.6.0+cu126.html

# Install torch-geometric
RUN pip install torch-geometric

# Install remaining dependencies
RUN pip install -r requirements.txt

# Copy full project
COPY . .

CMD ["python3", "train.py"]
