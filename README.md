# LTX-2.3 Image-to-Video — RunPod Serverless (GitHub deploy)

A serverless worker that runs **LTX-2.3** image-to-video (i2v). Deploys to RunPod
directly from this GitHub repo — RunPod reads the `Dockerfile`, builds the image
in its own registry, and redeploys on every push to the watched branch. No local
Docker, no manual image push.

## Files

| File | Purpose |
|------|---------|
| `handler.py` | RunPod serverless handler; loads the pipeline once, runs i2v, returns mp4 |
| `Dockerfile` | Build recipe (torch-preinstalled CUDA base + diffusers from git) |
| `requirements.txt` | Python deps — **read the diffusers pinning note inside** |
| `test_input.json` | Sample job for local testing |

## ⚠️ Before you deploy — two things to fix

1. **Pin diffusers.** LTX-2.3 support isn't in a stable diffusers release yet, so
   `requirements.txt` installs it from git `@main`. That makes rebuilds
   non-reproducible. Replace `main` with a verified commit SHA before production.
2. **The model may be gated on Hugging Face.** If so, accept the license on the
   model page and set `HF_TOKEN` as an endpoint environment variable / secret.

## Deploy steps

1. Push this folder to a GitHub repo.
2. RunPod console → **Settings → Connections → GitHub → Connect**, authorize this repo.
3. **Serverless → New Endpoint → Import Git Repository** → pick the repo + branch;
   Dockerfile path is `Dockerfile` (root).
4. **Attach a network volume** (see below) — mounts at `/runpod-volume`, where the
   handler caches weights.
5. **GPU:** default **80GB (A100/H100)**. Floor is **48GB (L40S)** — on 48GB you may
   need `enable_model_cpu_offload()` (commented in `handler.py`). The 22B model in
   bf16 does **not** fit 24GB; the "24GB" figure only applies to quantized variants.
6. **Deploy Endpoint.** Verify with `/mcp` (RunPod MCP) or the console.

## Weights on the network volume (important)

- **Network volumes are datacenter-specific.** The endpoint must live in the **same
  DC** as the volume, which constrains which GPUs are available there. Pick the DC
  by GPU availability first, create the volume there, then the endpoint.
- **Pre-seed the volume** before opening traffic. Otherwise the first cold start
  downloads ~40GB on billed GPU time, and multiple concurrent cold workers all
  download to the same volume at once. One-time warm-up: spin up a cheap Pod with
  the volume attached and run the model once (or send a single request with
  `min_workers=1`) so weights land in `/runpod-volume/hf` before scaling out.

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
| `HF_TOKEN` | — | Required only if the HF repo is gated |
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
