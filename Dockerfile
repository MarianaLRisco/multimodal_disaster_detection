FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

COPY requirements.txt .

RUN pip3 install --no-cache-dir \
    -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu118

RUN pip3 install --no-cache-dir -r requirements.txt