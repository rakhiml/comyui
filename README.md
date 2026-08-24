# LTX-2.3 Image-to-Video — RunPod Serverless (GitHub deploy)

A serverless worker that runs **LTX-2.3** image-to-video (i2v). Deploys to RunPod
directly from this GitHub repo — RunPod reads the `Dockerfile`, builds the image
in its own registry, and redeploys on every push to the watched branch. No local
Docker, no manual image push.

## Files

| File | Purpose |
|------|---------|
| `handler.py` | RunPod serverless handler; loads the pipeline once, runs i2v, returns mp4 |
| `Dockerfile` | Build recipe (torch-preinstalled CUDA base + ffmpeg) |
| `requirements.txt` | Python deps, `diffusers` pinned to 0.40.0 |
| `test_input.json` | Sample job for local testing |

## Notes before you deploy

- **No Hugging Face token needed.** `diffusers/LTX-2.3-Distilled-Diffusers` is
  public and ungated (verified via the HF API), so `HF_TOKEN` is optional.
- **Dependency pin:** `LTX2Pipeline` ships in stable **diffusers 0.40.0**
  (2026-08-20), so this repo pins `diffusers==0.40.0` — no git install, and
  rebuilds are reproducible. Model cards still saying "install diffusers from
  git" predate that release.
- **Base image matters.** diffusers 0.40.0 needs `torch>=2.6`, and the A100
  serverless pool reports CUDA **12.8** (not 12.4), so the Dockerfile uses a
  torch-2.8 / cu12.8 base. Don't downgrade it.
- **Pipeline class is loaded explicitly.** The repo's `model_index.json` declares
  `_class_name: "LTX2Pipeline"` — the *text-to-video* class, whose `__call__`
  takes no `image` argument. The handler therefore loads
  `LTX2ImageToVideoPipeline` directly. Using `DiffusionPipeline.from_pretrained`
  here resolves to the wrong class and fails on the first request.
- **Prompt enhancement is off.** Verified against the diffusers 0.40.0 source:
  `enable_prompt_enhancement` defaults to `False`, `prompt_enhancer` is an
  optional component that this repo does not ship, and the handler passes the
  flag explicitly anyway. Your prompts reach the model verbatim.

## Deployed configuration

The endpoint must be created in the **console** — the REST v2 API accepts only
`image`/`templateId`, so GitHub-sourced endpoints cannot be created via API/MCP.

| Setting | Value |
|---|---|
| Repo / branch | `rakhiml/comyui` @ `main`, Dockerfile path `Dockerfile` |
| Network volume | `ltx23-i2v-weights` (`us16ywe3fc`), 150 GB |
| Data center | **US-KS-2** (the volume's DC — the endpoint must match) |
| GPU pool | `AMPERE_80` — A100 80 GB, $2.72/hr serverless |
| Container disk | 30 GB (image is large: torch 2.8 + CUDA) |
| Workers | min 0, max 1 for the first run; raise max after weights are cached |
| Execution timeout | **1800 s for the first (seeding) job** — ~95 GB of downloads plus model load plus one generation will blow through 900 s. Lower it to ~600 s once weights are cached. A timeout isn't fatal (HF downloads resume), just billed waste. |

**Console steps:** Serverless → New Endpoint → Import Git Repository → pick the
repo/branch → set Dockerfile path → attach the volume → pick the GPU pool →
Deploy.

## Weights on the network volume (important)

- **Network volumes are datacenter-locked**, and that constrains GPU choice. Not
  every DC even supports volumes (US-MO-1, for example, does not), and not every
  DC stocks large GPUs — EU-RO-1 had no ≥48 GB serverless stock at all. Verify
  the GPU has stock in the volume's DC *before* creating either.
- **The model repo is ~95 GB**, which is why the volume is 150 GB rather than 100.
- **Seed the volume with one warm-up request before scaling out.** Keep
  `workersMax=1` for the first job so a single worker downloads the weights into
  `/runpod-volume/hf`; several cold workers otherwise race to download the same
  ~95 GB concurrently. Raise `workersMax` once that first job succeeds.

## Request format

```json
{
  "input": {
    "image": "https://.../cat.png",     // http(s) URL OR base64 (data: URI ok)
    "prompt": "cinematic slow push-in, soft light",
    "width": 768,                          // clamped to a multiple of 32
    "height": 512,                         // clamped to a multiple of 32
    "num_frames": 97,                      // clamped to 8*k + 1
    "steps": 8,                            // distilled default; raise for full model
    "guidance_scale": 1.0,                 // distilled = 1.0
    "fps": 24,
    "seed": 42
  }
}
```

**Response:** `{"video_base64": "...", ...}` by default. Set `BUCKET_ENDPOINT_URL`
(+ standard S3 creds) on the endpoint to switch to `{"video_url": "..."}` — do this
for longer/HD clips, since base64 mp4 can exceed RunPod's sync response payload limit.

## Env vars

| Var | Default | Notes |
|-----|---------|-------|
| `MODEL_ID` | `diffusers/LTX-2.3-Distilled-Diffusers` | Set to `Lightricks/LTX-2.3` for the full dev model |
| `DEFAULT_STEPS` | `8` | Distilled is 8 steps; full model wants ~30+ |
| `DEFAULT_GUIDANCE` | `1.0` | Distilled uses CFG=1 |
| `HF_TOKEN` | — | Not needed for the default model (public/ungated); set only if you switch to a gated checkpoint |
| `BUCKET_ENDPOINT_URL` | — | Set (with S3 creds) to return a URL instead of base64 |

## Local test

```bash
pip install -r requirements.txt
python handler.py          # runpod runs test_input.json automatically in local mode
```

## Notes / limitations

- **Audio:** LTX-2.3 generates video+audio jointly, but `export_to_video()` writes
  the **video track only**. If you need the audio, capture it from the pipeline
  output and mux it in with ffmpeg — not wired up in this scaffold.
- Defaults target cheapest generation (distilled, ~4s clip). Tune `num_frames`,
  `width/height`, and `steps` for quality vs cost.
