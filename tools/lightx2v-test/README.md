# LightX2V Wan2.2-NVFP4-Sparse — test rig (set up 2026-06-27, GPU-free prep)

Goal: a quick "how fast / how good" run of the **lightx2v Wan2.2-NVFP4-Sparse** (A14B)
model on the new **RTX 5060 Ti (16GB, Blackwell)**, vs our LTX-2.3 prod, for the shot:
*camera facing a footpath, a man walks from the distance toward the camera then past — 5s, 1280×704.*

## What this is
- **Model**: `lightx2v/Wan2.2-NVFP4-Sparse` — finetune of Wan2.2-**T2V**-A14B (we chose T2V so the
  test is self-contained: no start image / flux2 step needed). 4-step distill, NVFP4 (4-bit, Blackwell
  tensor cores), **SLA** sparse self-attention (80% sparsity, Triton) + SageAttention2 cross-attn.
- **Framework**: LightX2V (PyTorch), run via the official docker image. Our ggml fork can NOT run these
  NVFP4 safetensors — this is a separate PyTorch stack, on purpose (reference impl = real quality/speed).
- Headline claim (on a 5090): ~59× vs 50-step baseline @720p. The 5060 Ti is slower than a 5090 but
  should still crush our ggml Q8 baseline (~7 min for 5s@1024×574 in ComfyUI).

## How to run (once a GPU is free)
```bash
cd tools/lightx2v-test
./run_footpath_test.sh          # auto-picks the 5060 Ti by name, writes out/footpath_t2v_1280x704_<stamp>.mp4
```
Knobs: `PROMPT="..." ./run_footpath_test.sh`, `GPU_NAME=3060 ./run_footpath_test.sh` (force card).
Edit `config_t2v_1280x704_footpath.json` for res / length / steps / guidance / shift.

## Status of prerequisites (check before running)
- [ ] **docker image** `lightx2v/lightx2v:26052801-cu130-5090` — `docker images | grep lightx2v`
- [ ] **models** present — `du -sh ../../models/lightx2v/*` (expect ~17G nvfp4-sparse + ~13G base)
  - `models/lightx2v/nvfp4-sparse/Wan2.2-T2V-A14B_NVFP4_Sparse_{high,low}.safetensors`
  - `models/lightx2v/wan22-base-t2v/{models_t5_umt5-xxl-enc-bf16.pth, Wan2.1_VAE.pth, configuration.json, google/umt5-xxl/}`

## Validated at setup (no GPU)
- Image is a **runtime base** (cuda13.0 + `/app/miniconda` python3.11): provides torch **2.11.0+cu130**,
  `sageattention`, `spas_sage_attn` (the sparse-sage/SLA kernel), `flash_attn` 2.8.4, triton 3.6.0 — but
  **NOT the `lightx2v` package**. So the harness mounts the cloned repo (`tools/LightX2V`) as the source.
- `lightx2v` imports cleanly from the mounted clone (today's `main` against the May-28 image) — **no
  dependency / version-skew import errors**; it only stops at its import-time CUDA device check (expected
  without a GPU). So the package + deps are good; what's left genuinely needs the card.

## ⚠️ Still unverified (needs the GPU) — possible first-run friction
1. **sm_120 kernels**: image is tagged `-5090`. The 5060 Ti is also sm_120 (GB206) so CUTLASS-NVFP4 + SLA
   should run, but if kernels were built `sm_120a`-only there could be an arch mismatch. If so: rebuild the
   NVFP4 kernel in-container (`uv build --wheel` with `-DCUTLASS_NVCC_ARCHS=120`), or run the non-NVFP4
   `_comfy`/fp8 path.
2. **Blackwell offload**: research flagged async-offload instability on sm_120. The config uses block CPU
   offload (not async). If you hit offload hangs, try `cpu_offload:false` (A14B NVFP4 ~8.4GB/expert should
   fit 16GB without it) or look for a `--disable-async-offload` flag in `python -m lightx2v.infer --help`.
3. **Repo↔image version skew**: import is clean, but a runtime API mismatch is still possible. Fallback:
   `cd tools/LightX2V && git log --until=2026-05-29 -1` then `git checkout <that commit>` to date-match the image.
4. **1280×704**: divisible by 16 (valid). If VAE/patch complains, fall back to native 1280×720.

## i2v follow-up (not downloaded yet)
For the production-shaped path (flux2 still → i2v), pull the I2V NVFP4 weights and use `--task i2v`:
```bash
hf download lightx2v/Wan2.2-NVFP4-Sparse Wan2.2-I2V-A14B_NVFP4_Sparse_{high,low}.safetensors \
  --local-dir ../../models/lightx2v/nvfp4-sparse-i2v
```
(then a config with task i2v + `use_image_encoder:false` + an `--image_path`).

## DELETABILITY — everything this session added to disk
| Path | Size | Delete? | Re-create |
|---|---|---|---|
| `models/lightx2v/` | ~30 GB | ✅ safe | re-`hf download` (see `scratchpad/dl_lightx2v.sh`) |
| docker image `lightx2v/lightx2v:26052801-cu130-5090` | ~15–25 GB | ✅ safe | `docker pull` again |
| `tools/LightX2V/` (repo clone, reference only) | <1 GB | ✅ safe | `git clone --depth 1 https://github.com/ModelTC/LightX2V` |
| `tools/lightx2v-test/out/` (renders + logs) | grows | ✅ safe | re-run |
| `tools/lightx2v-test/` (this harness) | tiny | ❌ keep | — the deliverable |
| `models/*.gguf` (wan22 VACE a14b distill etc.) | ~24 GB | ✅ safe (separate — for the *ggml* Wan path, not this) | re-rsync from 10.0.0.151 |

Nuke the whole PyTorch experiment: `docker rmi lightx2v/lightx2v:26052801-cu130-5090; rm -rf ../../models/lightx2v tools/LightX2V tools/lightx2v-test/out`.

## Reality check on the goal
If it's fast *and* good, the catch you already called: **Wan has no native audio** (unlike LTX-2.3's joint
A/V). A Wan win on silent video means re-adding a lip-sync/audio sidecar (LatentSync, or our S2V port which
is audio-*driven*). So this answers the *video-quality* question; audio stays a separate bolt-on.
