# Shared context — porting the "Denoise-AI" LTX-2.3 workflow to longcat-avatar.cpp

You are one of several subagents working **in parallel** inside the git worktree
`/home/dbrain/dev/longcat-avatar-ltxdenoise` (branch `ltx-denoise-workflow`, forked off
`longcat-avatar.cpp` master). The main checkout with all the models is
`/home/dbrain/dev/longcat-avatar.cpp` (shared repo; large model files live there under `models/`).

## Hard constraints (read first)
- **NO GPU work. NO renders. NO heavy builds** (no `make`, no docker CUDA build, no sd-cli compile).
  This is a CPU-only "work ahead" pass. Writing code + docs + light python is fine.
- **Never kill foreign processes/containers.** A 7.61 GB LoRA download is running in the background.
- Only write to YOUR assigned output file(s). Other agents own other files — do not touch theirs.
- Read the C++/py source freely (read-only) except where your task says to add a new file.

## The goal
Our LTX-2.3 output has mushy faces on high-motion / distant-character footage ("LTX poison"),
though we love its speed + native audio. A Korean creator's ("Denoise-AI") ComfyUI workflow gets
clean high-motion faces. We reverse-engineered it and want to reproduce it on our ggml/nvfp4 stack.

## What the Denoise-AI workflow actually does (reverse-engineered from the JSONs in ./workflows/)
Base model + **partial-strength distillation LoRA** + **low-res base → spatial x2 upscale → hires refine**
+ **NAG** negative guidance + a **detailer LoRA** on the hires pass. Per-stage wiring (ver1; verify in JSON):

| Stage | res | sigmas (ManualSigmas) | model = dev-fp8 + LoRAs | guidance |
|---|---|---|---|---|
| S1 preview | 544×960 | `1.0,0.99375,0.9875,0.98125,0.975,0.909375` | distill-lora@**0.5** | **NAG(scale14, α0.35, τ2.5)** |
| S2 main (base) | 544×960 | `0.725, 0.421875, 0.0` | distill@**0.65** | cfg1 |
| — spatial ×2 upscale (LTXVLatentUpsampler + `ltx-2.3-spatial-upscaler-x2-1.1`) → 1088×1920 — |
| S3 upsc-preview | 1088×1920 | `0.421875, 0.0` | distill@**0.8** + detailer@**0.7** | cfg1 |
| S4 upscale/refine | 1088×1920 | `0.421875, 0.0` | distill@**0.8** + detailer@**0.8** | cfg1 |

- **fps = 24**, frames = 121, sampler = **euler**, cfg = **1** everywhere (guidance is via NAG, not CFG).
- Base checkpoint: `ltx-2.3-22b-dev-fp8.safetensors` (the **dev / non-distilled** model).
- distill LoRA: `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` (rank-384, 7.61 GB, downloading now).
- detailer LoRA: `ltx-2-19b-ic-lora-detailer.safetensors` — **NOTE: it's LTX-2 (19B), our base is LTX-2.3 (22B)**
  → dim-compat is UNKNOWN and must be checked; the workflow leaves it OFF on the base stages.
- Text encoder: `gemma_3_12B_it_fp8_scaled` (we keep our Q4_K_XL Gemma; TE ≠ face-quality bottleneck).
- Final decode: `VAEDecodeTiled` (tile 512, overlap 64).
- "preview" stages (S1, S3) are inspection taps you can drop; S2 (base) + upscale + S4 (refine) are load-bearing.

## Our stack — what already exists vs. what's missing
On the box (`longcat-avatar.cpp`):
- **nvfp4 LTX in prod**: `models/ltx2/nvfp4-CLEAN.gguf` **is distilled-1.1 in nvfp4**; `nvfp4mixed-CLEAN.gguf`.
  ggml has full `GGML_TYPE_NVFP4=40` (block_nvfp4: fp4 E2M1 + per-16 fp8 scales + per-tensor global).
- **LoRA fold precedent**: `tools/fold_lipdub_lora.py` folds an LTX LoRA into `nvfp4-CLEAN.gguf` in place
  (dequant block → add `B@A` delta → requantize), CPU/numpy only, strength hardcoded to 1.0.
- **BF16 source on disk**: `/mnt/hdd/ltx2-official/ltx-2.3-22b-distilled-1.1.safetensors` (46 GB);
  official fp4 at `/mnt/ssd/models/ltx-fp4/ltx-2.3-22b-distilled-1.1-nvfp4{,mixed}.safetensors`.
- **Runtime LoRA w/ strength**: `<lora:name:strength>` (regex `examples/common/common.cpp:1996`);
  `apply_loras` runs for **video** gen (`src/stable-diffusion.cpp:8390`); `--lora-apply-mode auto`
  → `at_runtime` for quantized bases. LoRA loads from `.safetensors/.gguf/.pt`.
- **hires / upscaler**: `--hires --hires-upscaler ltx-2.3-spatial-upscaler-x2-1.1 --hires-steps --hires-sigmas`
  (see `run_ltx_upscale_ab.sh`, `ltx_run.sh`). Spatial+temporal x2 upscalers are on the box.
- **nvfp4 quantizer**: `avatar_nvfp4_1_imatrix.sh` + `avatar_nvfp4_2_quantize.sh` (`-M convert --type nvfp4 --imatrix`).
- **SVDQuant spike**: `spike_cutlass_fp4/comfy-kitchen/comfy_kitchen/.../svdquant.py` (for LoRA rank-reduction).

MISSING (this is the work):
- **NAG** (normalized attention guidance) — not implemented; fork has CFG negative-prompt only.
- **Per-phase LoRA strength** — LoRA is global via prompt; `--hires` reuses same model+LoRA.
- **Base-pass custom sigma schedule** — we have `--hires-sigmas` but need base `--sigmas` equivalent.
- **The distill-lora fold** as a generalized, strength-parameterized script.

## The key insight for our port (the cheap win)
`nvfp4-CLEAN.gguf` == `dev + distill-lora@1.0` (that's what "distilled" is). So:
`dev + distill@s  =  distilled-1.1 + distill-lora@(s−1.0)`.
→ To get "dev + distill@0.65" we **fold the distill LoRA at strength −0.35 onto nvfp4-CLEAN** — no 46 GB
dev download, no re-quant. Same architecture / param count / nvfp4 format / kernels ⇒ **same VRAM & per-step
speed as current prod**, only less distillation baked in. The low-res-base+upscale structure is what lets the
weaker few-step distillation still converge fast. (Assumption to verify: distilled-1.1 = dev + this-lora@1.0;
a fold at −1.0 should recover ~pure dev.)

## GPU phase (later — for aligning your docs/scripts, NOT to run now)
A/B a "busy clip" (distant characters + motion = LTX poison): 1280×704, **48 fps**, **193 frames** (4 s),
current workflow+model vs. this ported workflow at best settings → both onto the eye-test page for the owner.
Big wins sought: (1) if this cleans motion at **24 fps** we do 50% less work; (2) if it makes footage usable
faster than the very-slow wan2.2+lipdub fallback. Usage modes to support: **i2v+supplied-audio, t2v+supplied-audio,
generated-audio, segment-continuation (chained clips, same characters)**.
