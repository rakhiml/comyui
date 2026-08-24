# Base image requirements — do not downgrade without checking both:
#   1. diffusers 0.40.0 requires torch >= 2.6.
#   2. CUDA 12.8 has kernels for every GPU pool this runs on, including
#      Blackwell (sm_120) for RTX PRO 6000. A cu12.4 image imports fine and
#      then fails at the first CUDA op on Blackwell.
#
# This is the -runtime image, not -devel: nothing here compiles CUDA
# extensions, and runtime is ~4.3GB compressed vs ~11.7GB for the RunPod
# devel images. Image size is cold-start latency on serverless — a worker
# cannot start until the whole image is pulled.
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

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

# The serverless worker loop starts here.
CMD ["python", "-u", "handler.py"]
