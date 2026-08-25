"""
LTX-2.3 image-to-video as a RunPod Flash endpoint.

Why this exists alongside handler.py:
RunPod's GitHub-integration builder stalls in "waiting for build" and never
produces an image, so the Dockerfile path in this repo cannot be deployed.
Flash packages local code into an artifact and runs it on a prebuilt RunPod
runtime image, so there is no Docker build step to stall.

`handler.py` is intentionally left in place — it remains the entry point for
the Dockerfile/GHCR path, and both share the same verified inference logic.

Deploy:
    pip install runpod-flash
    flash login
    flash deploy
"""

from runpod_flash import DataCenter, Endpoint, GpuGroup, NetworkVolume, PodTemplate

# The existing 150GB volume in US-KS-2.
#
# datacenter and size MUST be passed explicitly. NetworkVolume(id=...) does not
# look up the real volume — it silently defaults to EU-RO-1 / 100GB, which then
# propagates into the endpoint's `locations` as a list of 11 datacenters. A
# worker scheduled outside US-KS-2 would find no volume mounted and try to pull
# ~95GB of weights onto the container disk.
WEIGHTS_VOLUME = NetworkVolume(
    id="us16ywe3fc",
    datacenter=DataCenter.US_KS_2,
    size=150,
)

MODEL_ID = "diffusers/LTX-2.3-Distilled-Diffusers"


@Endpoint(
    name="ltx23-i2v",
    # H100 80GB. A100 (AMPERE_80) has no capacity in the volume's datacenter.
    gpu=GpuGroup.ADA_80_PRO,
    # One worker until the volume is seeded; several cold workers would
    # otherwise race to download the same ~95GB of weights.
    workers=(0, 1),
    idle_timeout=300,
    # The seeding run downloads ~95GB, loads a 22B model, then generates.
    # The 600s default would kill it partway through.
    execution_timeout_ms=1_800_000,
    # This list MUST stay an inline literal. Flash collects build-time
    # dependencies by static AST analysis and only accepts an ast.List here
    # (see extract_remote_dependencies in its build.py) — passing a module-level
    # constant is silently skipped, the artifact ships with zero packages, and
    # the worker then pip-installs reactively one import failure at a time.
    #
    # torch is deliberately absent: the Flash GPU runtime image already provides
    # it, and `flash deploy` auto-excludes torch packages anyway.
    dependencies=[
        "diffusers==0.40.0",
        "transformers>=4.44.0",
        "accelerate>=0.34.0",
        "imageio[ffmpeg]>=2.34.0",
        "imageio-ffmpeg>=0.5.1",
        "Pillow>=10.0.0",
        "sentencepiece>=0.2.0",
        "hf_transfer>=0.1.6",
    ],
    system_dependencies=["ffmpeg"],
    volume=WEIGHTS_VOLUME,
    # Pin scheduling to the volume's datacenter. Without this the endpoint is
    # offered every datacenter Flash knows about, and only US-KS-2 has the volume.
    datacenter=DataCenter.US_KS_2,
    template=PodTemplate(containerDiskInGb=30),
    # 12.8 covers H100 and Blackwell; the A100 pool reports 12.4 unavailable.
    min_cuda_version="12.8",
)
class LTXImageToVideo:
    """Instantiated once per worker, so the pipeline loads on cold start only."""

    def __init__(self):
        import os
        import time

        # Point the HF cache at the network volume BEFORE importing diffusers,
        # which reads HF_HOME at import time. The mount path is not documented,
        # so probe the known candidates rather than assuming one; falling back
        # to container disk would try to fit ~95GB into 30GB and fail.
        for candidate in ("/runpod-volume", "/workspace", "/runpod_volume"):
            if os.path.isdir(candidate):
                os.environ["HF_HOME"] = os.path.join(candidate, "hf")
                break
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        os.makedirs(
            os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
            exist_ok=True,
        )
        print(f"[boot] HF_HOME={os.environ.get('HF_HOME')}", flush=True)

        import torch
        from diffusers import LTX2ImageToVideoPipeline

        # Load LTX2ImageToVideoPipeline explicitly. The repo's model_index.json
        # declares _class_name "LTX2Pipeline" — the text-to-video class, whose
        # __call__ takes no `image` argument — so DiffusionPipeline.from_pretrained
        # would resolve to the wrong class and fail on the first request.
        t0 = time.time()
        self.pipe = LTX2ImageToVideoPipeline.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            device_map="cuda",
        )
        print(f"[boot] pipeline ready in {time.time() - t0:.1f}s", flush=True)

    def generate(
        self,
        image: str,
        prompt: str,
        width: int = 768,
        height: int = 512,
        num_frames: int = 97,
        steps: int = 8,
        guidance_scale: float = 1.0,
        fps: int = 24,
        seed: int = None,
    ) -> dict:
        import base64
        import io
        import os
        import tempfile
        import time

        import torch
        from diffusers.utils import export_to_video, load_image
        from PIL import Image

        # LTX requires width/height divisible by 32 and num_frames == 8k+1.
        # Clamp rather than reject, so odd client values still produce video.
        width = max(32, int(round(int(width) / 32)) * 32)
        height = max(32, int(round(int(height) / 32)) * 32)
        num_frames = ((max(9, int(num_frames)) - 1) // 8) * 8 + 1

        if image.startswith("http://") or image.startswith("https://"):
            init_image = load_image(image).convert("RGB")
        else:
            raw = image.split(",", 1)[1] if image.startswith("data:") else image
            init_image = Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(int(seed))

        t0 = time.time()
        result = self.pipe(
            image=init_image,
            prompt=prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            frame_rate=fps,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            # Explicitly off. It already defaults to False and this checkpoint
            # ships no prompt_enhancer component, but pin it so an upstream
            # default change cannot silently start rewriting prompts.
            enable_prompt_enhancement=False,
            generator=generator,
        )

        # LTX-2.3 generates video and audio jointly; export_to_video writes the
        # video track only. Mux result.audio separately if you need sound.
        out_path = os.path.join(tempfile.mkdtemp(), "output.mp4")
        export_to_video(result.frames[0], out_path, fps=fps)

        with open(out_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")

        return {
            "video_base64": video_b64,
            "seconds": round(time.time() - t0, 1),
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "steps": int(steps),
            "model": MODEL_ID,
        }
