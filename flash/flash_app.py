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

import os

# Set before torch initialises its CUDA allocator. Without this the allocator
# uses fixed-size segments and fragments across requests: a worker was seen
# holding 15.98GB "reserved but unallocated" while a 15.35GB decode failed.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from runpod_flash import DataCenter, Endpoint, GpuGroup, NetworkVolume, PodTemplate  # noqa: E402

# HF_TOKEN is deliberately NOT set here.
#
# Set it as an environment variable on the endpoint in the Runpod console
# (Serverless -> ltx23-i2v -> Manage -> Environment Variables). huggingface_hub
# reads HF_TOKEN from the worker environment on its own, so no code needs to
# read or forward it, and the token never touches this public repository or a
# local shell.
#
# It is optional for the default checkpoint, which is public and ungated — it
# raises Hugging Face rate limits on the initial ~95GB pull and silences the
# "sending unauthenticated requests" warning. It becomes mandatory only if
# MODEL_ID is pointed at a gated checkpoint.
#
# NOTE: verify the variable survives a `flash deploy`. Flash manages the
# endpoint's env (it writes FLASH_APP, FLASH_ENV and a source fingerprint), so
# a redeploy may reset console-set values.

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

# The full dev checkpoint. The standalone distilled checkpoint was noticeably
# weaker — anatomy errors, motion that jitters rather than follows the prompt —
# and distillation is baked into its weights, so there is no dial to back it off.
#
# Dev restores quality and can be sped back up by applying the distilled LoRA at
# partial strength, which is what the ComfyUI templates do. Override with the
# MODEL_ID env var to A/B against the distilled checkpoint without redeploying
# code.
MODEL_ID = os.environ.get("MODEL_ID", "diffusers/LTX-2.3-Diffusers")
IS_DISTILLED = "distilled" in MODEL_ID.lower()

# The distilled checkpoint is fixed at 8 steps with CFG 1.0. The dev checkpoint
# wants many more steps and real guidance — and only above CFG 1.0 does a
# negative prompt do anything, since the video CFG delta is
# (guidance_scale - 1) * (cond - uncond).
DEFAULT_STEPS = 8 if IS_DISTILLED else 30
DEFAULT_GUIDANCE = 1.0 if IS_DISTILLED else 3.0

# Applied on top of dev to recover most of the distilled speed while keeping
# dev quality. diffusers auto-converts this Lightricks-format file.
DISTILL_LORA_REPO = "Lightricks/LTX-2.3"
DISTILL_LORA_FILE = "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"

# Applied when a request supplies no negative prompt. Classifier-free guidance
# is active on this setup even at guidance_scale=1.0, because
# do_classifier_free_guidance also fires on audio_guidance_scale (default 7.0),
# so this actually steers the output rather than being inert.
#
# Pass any non-empty negative_prompt to override it.
DEFAULT_NEGATIVE_PROMPT = (
    "low quality, worst quality, blurry, out of focus, soft focus, low resolution, "
    "pixelated, noisy, grainy, oversaturated, undersaturated, overexposed, "
    "underexposed, bad lighting, compression artifacts, jpeg artifacts, flickering, "
    "frame flicker, temporal inconsistency, unstable image, jitter, camera shake, "
    "unwanted camera movement, warping, morphing, melting, deformation, distorted "
    "anatomy, bad anatomy, malformed body, extra limbs, extra arms, extra legs, "
    "extra fingers, missing fingers, fused fingers, malformed hands, distorted face, "
    "asymmetrical face, facial warping, inconsistent facial features, identity "
    "change, face morphing, eye deformation, crossed eyes, unnatural eyes, mouth "
    "distortion, lip-sync artifacts, unnatural motion, jerky motion, erratic "
    "movement, teleportation, sudden position changes, object duplication, "
    "disappearing objects, appearing objects, inconsistent clothing, changing "
    "clothes, changing hairstyle, changing colors, background morphing, background "
    "flicker, geometry distortion, texture crawling, unnatural physics, floating "
    "objects, duplicated subjects, ghosting, trails, motion artifacts, static image, "
    "frozen motion, excessive motion blur, text, subtitles, captions, watermark, "
    "logo, UI, border, cropped subject"
)


@Endpoint(
    name="ltx23-i2v",
    # Several 80GB+ pools, because the endpoint is pinned to one datacenter and
    # per-DC stock shifts hour to hour — a single pool leaves workers THROTTLED
    # when that pool drains. AMPERE_80 (A100, $2.72/hr) is the cheapest of these
    # and ADA_80_PRO (H100) is $4.79/hr, so listing A100 first also favours the
    # cheaper option when both have capacity.
    gpu=[GpuGroup.AMPERE_80, GpuGroup.ADA_80_PRO, GpuGroup.BLACKWELL_96],
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
        # PyAV. LTX re-compresses the conditioning image through H.264 at a
        # given CRF before encoding it, to match how the model saw conditioning
        # frames in training. Without `av` the pipeline raises at call time.
        # The alternative is passing image_crf=0 to skip re-compression, but
        # that changes the conditioning the model receives.
        "av>=12.0.0",
    ],
    system_dependencies=["ffmpeg"],
    # Set on the endpoint, not just in Python. torch reads this when its CUDA
    # allocator initialises, and the module-level os.environ call in this file
    # only wins if nothing has touched CUDA before our import. Putting it in the
    # process environment removes that ordering question.
    #
    # NOTE: Flash owns the endpoint's env, so deploying this may clear an
    # HF_TOKEN set by hand in the console. Re-add it there afterwards if needed;
    # it is optional now that the weights are cached on the volume.
    env={
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        # Reclaim the distilled checkpoint's cache. The volume cannot hold it
        # alongside the dev weights, and we no longer generate with it. Clear
        # this once the space has been reclaimed; leaving it set is harmless
        # but re-checks the path on every cold start.
        "PURGE_HF_REPOS": "diffusers/LTX-2.3-Distilled-Diffusers",
    },
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
        self._volume_root = None
        for candidate in ("/runpod-volume", "/workspace", "/runpod_volume"):
            if os.path.isdir(candidate):
                self._volume_root = candidate
                os.environ["HF_HOME"] = os.path.join(candidate, "hf")
                break

        # LoRA files live on the same volume, under loras/.
        self._lora_dir = (
            os.path.join(self._volume_root, "loras") if self._volume_root else None
        )
        if self._lora_dir:
            os.makedirs(self._lora_dir, exist_ok=True)

        # adapter_name -> True, for adapters already loaded into this worker's
        # pipeline. Loading is expensive; switching between loaded adapters is not.
        self._loaded_adapters = {}
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        os.makedirs(
            os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
            exist_ok=True,
        )
        print(f"[boot] HF_HOME={os.environ.get('HF_HOME')}", flush=True)

        # Drop checkpoints named in PURGE_HF_REPOS from the volume's HF cache.
        #
        # The volume holds one model comfortably and not two: adding the dev
        # checkpoint alongside the distilled one exhausted it and every worker
        # failed with "Disk quota exceeded (os error 122)". Growing the volume
        # is a one-way door on RunPod — volumes can only grow — so reclaiming
        # the checkpoint we no longer use is the cheaper fix.
        #
        # Only ever removes a Hugging Face cache directory, which is
        # re-downloadable, and only for repos named explicitly in the env var.
        purge = [r.strip() for r in os.environ.get("PURGE_HF_REPOS", "").split(",") if r.strip()]
        if purge and self._volume_root:
            import shutil

            hub = os.path.join(os.environ["HF_HOME"], "hub")
            for repo in purge:
                if repo == MODEL_ID:
                    print(f"[purge] refusing to remove the active model {repo}", flush=True)
                    continue
                target = os.path.join(hub, "models--" + repo.replace("/", "--"))
                if os.path.isdir(target):
                    shutil.rmtree(target, ignore_errors=True)
                    print(f"[purge] removed {target}", flush=True)

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
        )

        # Do NOT pin the whole pipeline to the GPU with device_map="cuda".
        #
        # These weights are ~69GB resident. On a 93GB card that leaves ~24GB,
        # and the VAE decode alone asks for a single ~15GB block. The first
        # request fits on clean memory; the second fails once any fragmentation
        # appears. That budget is marginal by construction, so freeing memory
        # between jobs cannot rescue it — the resident footprint has to come
        # down.
        #
        # Model CPU offload keeps only the module currently executing on the
        # GPU and parks the rest in host RAM, so the transformer and the VAE no
        # longer have to coexist in VRAM. It costs some speed per clip and buys
        # a worker that serves many requests instead of exactly one.
        self.pipe.enable_model_cpu_offload()

        # VAE tiling is OFF by default because it costs image quality: the VAE
        # decodes each tile separately and blends the overlaps, so tiles that
        # are small relative to the frame leave visible seams and texture
        # artifacts. An earlier 256px/192px-stride setting here (64px of
        # overlap on a 768x512 frame) did exactly that.
        #
        # Model CPU offload above is what buys the VRAM headroom; tiling was
        # belt-and-braces. Re-enable it only if decodes OOM again — and if so
        # prefer large tiles with generous overlap over small ones.
        if os.environ.get("ENABLE_VAE_TILING", "").lower() in ("1", "true", "yes"):
            try:
                self.pipe.vae.enable_tiling(
                    tile_sample_min_height=512,
                    tile_sample_min_width=512,
                    tile_sample_stride_height=384,
                    tile_sample_stride_width=384,
                )
                print("[boot] VAE tiling enabled (512px tiles)", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[boot] VAE tiling unavailable: {exc}", flush=True)
        else:
            print("[boot] VAE tiling off (full-frame decode)", flush=True)

        print(f"[boot] pipeline ready in {time.time() - t0:.1f}s", flush=True)
        self._free_gpu("after load")

    def _free_gpu(self, label):
        """Return cached blocks to the driver between requests.

        The pipeline is a per-worker singleton, so anything still referenced
        after a generation stays resident for the life of the worker. Without
        this the second request inherits the first one's peak.
        """
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(
            f"[mem] {label}: allocated={alloc:.1f}GB reserved={reserved:.1f}GB",
            flush=True,
        )

    def _apply_lora(self, lora, lora_scale):
        """Select the LoRA for THIS request, or clear any left over from the last.

        The pipeline is a per-worker singleton, so an adapter enabled for one
        request stays enabled for every later request on that worker unless it
        is explicitly turned off. That failure is silent — no error, just
        wrong-looking video — so the no-LoRA branch below is the important one.
        """
        import os

        if not lora:
            if self._loaded_adapters:
                self.pipe.disable_lora()
            return None

        # "distill" is a shorthand for the official distilled LoRA. Applied to
        # the dev checkpoint at partial strength it recovers most of the
        # distilled speed while keeping dev quality — the trade the standalone
        # distilled checkpoint makes for you at full strength with no dial.
        # diffusers auto-converts this Lightricks-format file.
        if lora == "distill":
            adapter = "distill"
            if adapter not in self._loaded_adapters:
                print(f"[lora] loading {DISTILL_LORA_FILE}", flush=True)
                self.pipe.load_lora_weights(
                    DISTILL_LORA_REPO,
                    weight_name=DISTILL_LORA_FILE,
                    adapter_name=adapter,
                )
                self._loaded_adapters[adapter] = True
            self.pipe.set_adapters([adapter], [float(lora_scale)])
            self.pipe.enable_lora()
            return adapter

        # A bare name means a file on the volume; anything with a slash or a
        # .safetensors suffix is treated as an explicit path or an HF repo id.
        source = lora
        if "/" not in lora and self._lora_dir:
            candidate = os.path.join(self._lora_dir, lora)
            if not os.path.exists(candidate) and not lora.endswith(".safetensors"):
                candidate += ".safetensors"
            if os.path.exists(candidate):
                source = candidate
            else:
                available = sorted(os.listdir(self._lora_dir)) if self._lora_dir else []
                raise FileNotFoundError(
                    f"LoRA '{lora}' not found in {self._lora_dir}. "
                    f"Available: {available or 'none uploaded yet'}"
                )

        adapter = "".join(c if c.isalnum() else "_" for c in lora)
        if adapter not in self._loaded_adapters:
            print(f"[lora] loading {source} as '{adapter}'", flush=True)
            self.pipe.load_lora_weights(source, adapter_name=adapter)
            self._loaded_adapters[adapter] = True

        self.pipe.set_adapters([adapter], [float(lora_scale)])
        self.pipe.enable_lora()
        return adapter

    def generate(
        self,
        image: str,
        prompt: str,
        negative_prompt: str = None,
        width: int = 768,
        height: int = 512,
        num_frames: int = 121,
        steps: int = None,
        guidance_scale: float = None,
        fps: int = 24,
        seed: int = None,
        lora: str = None,
        lora_scale: float = 1.0,
        include_audio: bool = True,
    ) -> dict:
        import base64
        import io
        import os
        import tempfile
        import time

        import torch
        from diffusers.utils import load_image
        from PIL import Image

        # Prefer the library's own constants so they track the checkpoint, but
        # fall back to the published values if this internal path ever moves.
        try:
            from diffusers.pipelines.ltx2.utils import (
                DEFAULT_IMAGE_CRF,
                DISTILLED_SIGMA_VALUES,
            )
        except ImportError:
            DISTILLED_SIGMA_VALUES = [
                1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875,
            ]
            DEFAULT_IMAGE_CRF = 33

        # Defaults follow the loaded checkpoint rather than being hardcoded, so
        # switching MODEL_ID does not silently leave distilled settings applied
        # to the dev model.
        steps = DEFAULT_STEPS if steps is None else int(steps)
        guidance_scale = (
            DEFAULT_GUIDANCE if guidance_scale is None else float(guidance_scale)
        )

        # Warn on the combination that silently produces bad video: the dev
        # checkpoint denoised with the distilled step count and no distillation
        # applied. It does not error, it just comes out under-denoised.
        if not IS_DISTILLED and steps < 20 and lora != "distill":
            print(
                f"[warn] {steps} steps on the dev checkpoint without lora='distill'. "
                f"Dev expects ~{DEFAULT_STEPS} steps and guidance ~{DEFAULT_GUIDANCE}; "
                f"low step counts here look under-denoised rather than fast.",
                flush=True,
            )

        # LTX requires width/height divisible by 32 and num_frames == 8k+1.
        # Clamp rather than reject, so odd client values still produce video.
        width = max(32, int(round(int(width) / 32)) * 32)
        height = max(32, int(round(int(height) / 32)) * 32)
        num_frames = ((max(9, int(num_frames)) - 1) // 8) * 8 + 1

        active_adapter = self._apply_lora(lora, lora_scale)

        # Missing or blank falls back to the shared default; any non-empty value
        # from the caller wins.
        if not (negative_prompt or "").strip():
            negative_prompt = DEFAULT_NEGATIVE_PROMPT

        if image.startswith("http://") or image.startswith("https://"):
            init_image = load_image(image).convert("RGB")
        else:
            raw = image.split(",", 1)[1] if image.startswith("data:") else image
            init_image = Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(int(seed))

        t0 = time.time()
        result = None
        try:
            result = self.pipe(
                image=init_image,
                prompt=prompt,
                # None becomes "" inside the pipeline. Classifier-free guidance
                # is active here even though guidance_scale is 1.0, because
                # do_classifier_free_guidance also fires on audio_guidance_scale
                # (default 7.0), so a negative prompt does affect the output.
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                frame_rate=fps,
                num_inference_steps=int(steps),
                # The distilled checkpoint is trained against a fixed, very
                # non-uniform sigma schedule — five of its eight steps sit
                # between 0.975 and 1.0, then it drops sharply. Running it on
                # the scheduler's default sigmas is off-distribution and shows
                # up as malformed anatomy, incoherent motion and background
                # garbage rather than as an error. The model card is explicit
                # that these must always be passed.
                #
                # Only supplied at the schedule's own step count; a caller
                # asking for a different number of steps has left the schedule
                # behind anyway.
                sigmas=(
                    DISTILLED_SIGMA_VALUES
                    if (IS_DISTILLED or active_adapter == "distill")
                    and steps == len(DISTILLED_SIGMA_VALUES)
                    else None
                ),
                guidance_scale=float(guidance_scale),
                # Match the H.264 compression the conditioning images were
                # trained against (33 for LTX-2.3).
                image_crf=DEFAULT_IMAGE_CRF,
                # Explicitly off. It already defaults to False and this checkpoint
                # ships no prompt_enhancer component, but pin it so an upstream
                # default change cannot silently start rewriting prompts.
                enable_prompt_enhancement=False,
                # Float frames in [0, 1] rather than PIL. encode_video wants
                # these, and it avoids a needless round trip through uint8.
                output_type="np",
                generator=generator,
            )

            # Encode with LTX's own PyAV encoder, not export_to_video.
            #
            # export_to_video defaults to quality=5.0 on a 0-10 scale, i.e.
            # middling variable bitrate — a 2s 768x512 clip came out at 76KB,
            # about 1.5KB per frame, which is where the compression artifacts
            # came from. encode_video is the path the official LTX-2 example
            # uses, and it muxes the audio the model generates instead of
            # discarding it.
            from diffusers.utils import encode_video

            out_path = os.path.join(tempfile.mkdtemp(), "output.mp4")
            audio_track = None
            audio_rate = None
            if include_audio and getattr(result, "audio", None) is not None:
                audio_track = result.audio[0].float().cpu()
                audio_rate = self.pipe.vocoder.config.output_sampling_rate

            encode_video(
                result.frames[0],
                fps=int(fps),
                output_path=out_path,
                audio=audio_track,
                audio_sample_rate=audio_rate,
            )

            with open(out_path, "rb") as f:
                video_b64 = base64.b64encode(f.read()).decode("utf-8")
        finally:
            # Runs on the failure path too: an OOM mid-generation otherwise
            # leaves the partial allocation resident and every subsequent
            # request on this worker inherits it.
            del result, generator, init_image
            self._free_gpu("after generate")

        return {
            "video_base64": video_b64,
            "seconds": round(time.time() - t0, 1),
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "steps": int(steps),
            "model": MODEL_ID,
            "lora": lora,
            "lora_scale": float(lora_scale) if active_adapter else None,
            "negative_prompt": negative_prompt,
            "has_audio": bool(include_audio),
        }
