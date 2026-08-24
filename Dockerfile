# Torch is preinstalled in this base image so the RunPod GitHub builder stays
# well inside its build-time limits (no compiling torch from scratch).
#
# Base image requirements — do not downgrade without checking both:
#   1. diffusers 0.40.0 requires torch >= 2.6.
#   2. CUDA 12.8 provides Blackwell (sm_120) kernels, needed for RTX PRO 6000
#      class GPUs. A cu12.4 image imports fine and then fails at the first
#      CUDA op on Blackwell.
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

# ffmpeg for export_to_video / imageio muxing.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY handler.py .

# RunPod invokes the container and the handler starts the serverless worker loop.
CMD ["python", "-u", "handler.py"]
