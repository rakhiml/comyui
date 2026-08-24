# LTX-2.3 Image-to-Video — RunPod Serverless

A serverless worker that runs **LTX-2.3** image-to-video (i2v). Every push to
`main` builds the image on GitHub's runners and pushes it to GHCR; the RunPod
endpoint runs that image. No local Docker required.

## Why not RunPod's GitHub integration

RunPod can build from a repo directly, but its build queue stalls: a build here
sat in *"waiting for build"* indefinitely, and the same failure is reported by
other users against RunPod's own official repos. Building on GitHub Actions
sidesteps that queue entirely and has a second benefit — the endpoint then runs
a plain image reference, which the REST API can set, so the endpoint can be
created and updated programmatically instead of only through the console.

If you see a *"Could not find `runpod.serverless.start()`"* warning in the
console, it is a false negative. This handler matches RunPod's canonical
documented form exactly, and the widely-deployed `wlsdml1114/generate_video`
Hub worker passes the same check with a `CMD` that doesn't even invoke its
handler directly.

## Files

| File | Purpose |
|------|---------|
| `handler.py` | RunPod serverless handler; loads the pipeline once, runs i2v, returns mp4 |
| `Dockerfile` | Build recipe (slim torch/CUDA runtime base + ffmpeg) |
| `requirements.txt` | Python deps, `diffusers` pinned to 0.40.0 |
| `.github/workflows/build.yml` | Builds and pushes the image to GHCR on every push to `main` |
| `test_input.json` | Sample job for local testing |

## Notes before you deploy

- **No Hugging Face token needed.** `diffusers/LTX-2.3-Distilled-Diffusers` is
  public and ungated (verified via the HF API), so `HF_TOKEN` is optional.
- **Dependency pin:** `LTX2Pipeline` ships in stable **diffusers 0.40.0**
  (2026-08-20), so this repo pins `diffusers==0.40.0` — no git install, and
  rebuilds are reproducible. Model cards still saying "install diffusers from
  git" predate that release.
- **Base image matters.** diffusers 0.40.0 needs `torch>=2.6`, and the GPU pools
  report CUDA **12.8** (not 12.4), so the Dockerfile uses a cu12.8 base. It is
  the `-runtime` image, not `-devel`: nothing here compiles CUDA extensions, and
  runtime is ~4.3 GB compressed against ~11.7 GB for RunPod's devel images. On
  serverless, image size *is* cold-start latency — a worker cannot start until
  the whole image is pulled, and an 11.7 GB image left workers stuck in
  `INITIALIZING` for 25+ minutes here.
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

Because the endpoint runs a plain image, it can be created via the REST API or
the RunPod MCP server rather than the console.

**One-time setup:** after the first workflow run, the GHCR package is private by
default. Either make it public (GitHub → your profile → Packages →
`comyui` → Package settings → Change visibility → Public), or keep it private
and register a container registry credential with RunPod and pass its id as
`containerRegistryAuthId`. Public is simpler and leaks nothing the public repo
doesn't already expose.

| Setting | Value |
|---|---|
| Image | `ghcr.io/rakhiml/comyui:latest` (built by `.github/workflows/build.yml`) |
| Network volume | `ltx23-i2v-weights` (`us16ywe3fc`), 150 GB |
| Data center | **US-KS-2** (the volume's DC — the endpoint must match) |
| GPU pool | `BLACKWELL_96` — RTX PRO 6000, 96 GB, $3.49/hr. Add `ADA_80_PRO` (H100 80 GB, $4.79/hr) as a second pool for resilience. **Not `AMPERE_80`** — A100 has no US-KS-2 capacity. |
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
- **Serverless GPU stock moves hour to hour.** US-KS-2 listed A100 capacity when
  this volume was created and had dropped it within the hour, while still
  offering RTX PRO 6000 and H100. If the console's endpoint wizard does not show
  your network volume, that usually means the GPU pool you selected has no
  capacity in the volume's DC — change the pool rather than the volume. Check
  current stock per DC with the RunPod MCP (`get-gpu-type`, `product: SERVERLESS`)
  and select several pools so one pool draining doesn't starve the endpoint.
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
