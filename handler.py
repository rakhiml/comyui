"""
RunPod Serverless handler — LTX-2.3 image-to-video (i2v).

Design notes:
- The pipeline is loaded ONCE at module import (worker cold start) into a global
  singleton, so subsequent jobs on the same worker reuse it.
- Model weights are cached to the attached RunPod network volume (mounted at
  /runpod-volume) via HF_HOME, so they download only once per volume instead of
  once per cold start. Pre-seed the volume before opening traffic (see README).
- Defaults target the DISTILLED checkpoint (8 steps, CFG=1) for cheapest
  serverless generation. Override with the MODEL_ID env var.
"""

import base64
import io
import os
import tempfile
import time

import torch
from PIL import Image

# --- Cache location: point HF at the network volume if present -----------------
# RunPod network volumes mount at /runpod-volume on serverless workers.
_VOLUME = "/runpod-volume"
if os.path.isdir(_VOLUME):
    os.environ.setdefault("HF_HOME", os.path.join(_VOLUME, "hf"))
os.makedirs(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), exist_ok=True)

import runpod  # noqa: E402
from diffusers import LTX2ImageToVideoPipeline  # noqa: E402
from diffusers.utils import export_to_video, load_image  # noqa: E402

# Distilled diffusers-format checkpoint = fewest steps = cheapest per clip.
# Full model: "Lightricks/LTX-2.3".  fp8 (if/when available) further cuts VRAM.
MODEL_ID = os.environ.get("MODEL_ID", "diffusers/LTX-2.3-Distilled-Diffusers")
# Distilled runs at 8 steps / guidance 1.0. Bump both for the full dev model.
DEFAULT_STEPS = int(os.environ.get("DEFAULT_STEPS", "8"))
DEFAULT_GUIDANCE = float(os.environ.get("DEFAULT_GUIDANCE", "1.0"))
HF_TOKEN = os.environ.get("HF_TOKEN")  # only needed if the repo is gated

# --- Load the pipeline once ----------------------------------------------------
print(f"[boot] loading {MODEL_ID} (HF_HOME={os.environ.get('HF_HOME')}) ...", flush=True)
_t0 = time.time()
# Load LTX2ImageToVideoPipeline EXPLICITLY. The repo's model_index.json declares
# _class_name "LTX2Pipeline", so DiffusionPipeline.from_pretrained would resolve
# to the text-to-video class — whose __call__ takes no `image` argument and would
# fail on the first request. Only LTX2ImageToVideoPipeline accepts `image`.
PIPE = LTX2ImageToVideoPipeline.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="cuda",
    token=HF_TOKEN,
)
# If you move to a 48GB card and hit OOM, drop device_map="cuda" above and call
# PIPE.enable_model_cpu_offload() here instead — slower, but far less VRAM.
print(f"[boot] pipeline ready in {time.time() - _t0:.1f}s", flush=True)


def _round_to(value: int, multiple: int) -> int:
    """Round to nearest positive multiple (LTX needs w/h divisible by 32)."""
    value = max(multiple, int(value))
    return int(round(value / multiple)) * multiple


def _round_frames(n: int) -> int:
    """LTX requires num_frames % 8 == 1 (i.e. 8*k + 1)."""
    n = max(9, int(n))
    return ((n - 1) // 8) * 8 + 1


def _load_input_image(spec: str) -> Image.Image:
    """Accept an http(s) URL or a base64 string (optionally a data: URI)."""
    if spec.startswith("http://") or spec.startswith("https://"):
        return load_image(spec).convert("RGB")
    if spec.startswith("data:"):
        spec = spec.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(spec))).convert("RGB")


def handler(job):
    job_input = job.get("input", {}) or {}

    image_spec = job_input.get("image")
    prompt = job_input.get("prompt")
    if not image_spec:
        return {"error": "Missing 'image' (http(s) URL or base64 string)."}
    if not prompt:
        return {"error": "Missing 'prompt'."}

    # Clamp to LTX's dimension constraints instead of erroring.
    width = _round_to(job_input.get("width", 768), 32)
    height = _round_to(job_input.get("height", 512), 32)
    num_frames = _round_frames(job_input.get("num_frames", 97))  # ~4s @ 24fps
    steps = int(job_input.get("steps", DEFAULT_STEPS))
    guidance = float(job_input.get("guidance_scale", DEFAULT_GUIDANCE))
    fps = int(job_input.get("fps", 24))
    seed = job_input.get("seed")

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(int(seed))

    try:
        image = _load_input_image(image_spec)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Could not load input image: {exc}"}

    t0 = time.time()
    result = PIPE(
        image=image,
        prompt=prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        frame_rate=fps,
        num_inference_steps=steps,
        guidance_scale=guidance,
        # Explicitly off. It already defaults to False and this repo ships no
        # prompt_enhancer component, but pin it so an upstream default change
        # cannot silently start rewriting prompts.
        enable_prompt_enhancement=False,
        generator=generator,
    )
    frames = result.frames[0]

    # NOTE: LTX-2.3 generates video+audio jointly; export_to_video writes the
    # video track only. See README for muxing audio if you need it.
    out_path = os.path.join(tempfile.mkdtemp(), "output.mp4")
    export_to_video(frames, out_path, fps=fps)
    gen_s = time.time() - t0

    meta = {
        "seconds": round(gen_s, 1),
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "steps": steps,
        "model": MODEL_ID,
    }

    # If S3-compatible creds are set on the endpoint, upload and return a URL
    # (base64 mp4 can exceed RunPod's sync response payload limit for longer/HD clips).
    if os.environ.get("BUCKET_ENDPOINT_URL"):
        from runpod.serverless.utils import rp_upload

        url = rp_upload.upload_file_to_bucket(job["id"], out_path)
        return {"video_url": url, **meta}

    with open(out_path, "rb") as f:
        return {"video_base64": base64.b64encode(f.read()).decode("utf-8"), **meta}


runpod.serverless.start({"handler": handler})
