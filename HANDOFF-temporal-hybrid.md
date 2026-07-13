# LTX temporal/I2V handoff — 2026-07-13

## Constraint

Repository: `/home/dbrain/dev/longcat-avatar-temporal`, branch `ltx-temporal-hybrid`.
Runnable CUDA build worktree: `/home/dbrain/dev/longcat-avatar-ltxdenoise`.
Models: `/home/dbrain/dev/longcat-avatar.cpp/models/ltx2`.

**Do not use GPU 1 or start a CUDA build/render until the user explicitly releases it.**
GPU 1 is temporarily in use. CPU-only source/history/community-workflow investigation is safe.
The source worktree is clean. The many untracked `ltx-denoise-repro/_*` directories are render
artifacts: do not remove, stage, or modify them.

Recent pushed commits:

- `91ab32e ltx: retain adaptive refine appearance anchor`
- `f128441 ltx: release stage-one DiT before upscale`
- `f80d8cc repro: support banking LTX chain latents`
- `9a3985b repro: support init image smoke renders`

## Goals

1. Restore correct I2V for long temporally windowed clips. A source/keyframe must condition stage
   1 at base-grid resolution and correctly appear again in high-resolution refine. It must
   generalise to arbitrary externally supplied keyframes, not only frame zero.
2. Make windowing clean: I2V needs immutable source-keyframe identity separately from short
   recent-motion context. For T2V, test a clean low-motion/sharp generated-frame handoff rather
   than periodically replacing a global identity anchor.
3. Follow proven community LTX/Comfy workflows and IC/identity LoRAs rather than inventing guide
   token semantics.

An “original reference image” exists only for I2V/keyframe jobs. Pure T2V has no locked image:
a generated anchor is only a fallback continuity aid, whereas an input image or character LoRA
can genuinely constrain identity.

The 20-second adaptive-anchor experiment is not a production answer: it has colour flashes and
hair/identity changes, and refreshes a generated appearance anchor every three refine tiles.

## Build and repro (only after GPU is available)

Harness: `ltx-denoise-repro/run_singing_clip.sh`. It documents all knobs at its head.

Build changed source files into the CUDA worktree:

```bash
cd /home/dbrain/dev/longcat-avatar-temporal
docker rm -f ltx-hybrid-build >/dev/null 2>&1 || true
docker run --name ltx-hybrid-build --gpus '"device=1"' --entrypoint bash \
  -v /home/dbrain/dev/longcat-avatar-ltxdenoise:/src \
  -v "$PWD/src/model/vae/ltx_vae.hpp:/src/src/model/vae/ltx_vae.hpp:ro" \
  -v "$PWD/src/stable-diffusion.cpp:/src/src/stable-diffusion.cpp:ro" \
  -v "$PWD/src/core/ggml_extend.hpp:/src/src/core/ggml_extend.hpp:ro" \
  -v "$PWD/include/stable-diffusion.h:/src/include/stable-diffusion.h:ro" \
  -w /src longcat-avatar-dev:builder-cudnn-ff \
  -lc 'cmake --build build-sa3 --target sd-cli -j8'
```

97-frame I2V repro:

```bash
cd /home/dbrain/dev/longcat-avatar-temporal
OUTDIR="$PWD/ltx-denoise-repro/_candidate_i2v"
mkdir -p "$OUTDIR"
cp ltx-denoise-repro/_i2v_anchor_smoke_97f/init.png "$OUTDIR/init.png"
env WT=/home/dbrain/dev/longcat-avatar-ltxdenoise GPU=1 SEED=123 SEGMENTS=1 FRAMES=97 \
  HIRES=1 MAXV=9 VAE_SPATIAL_TILES=1x2 VAE_SPATIOTEMPORAL_BLEND=1 \
  VAE_TEMPORAL_STREAM=1 VAE_TBLEND_FRAMES=8 VAE_TBLEND_OVERLAP=2 \
  BASE_TBLEND_FRAMES=14 BASE_TBLEND_OVERLAP=4 REFINE_TEMPORAL_BLEND=1 \
  REFINE_TBLEND_FRAMES=8 REFINE_TBLEND_OVERLAP=2 INIT_IMG=/work/init.png \
  OUTDIR="$OUTDIR" bash ltx-denoise-repro/run_singing_clip.sh
```

The input path is container-relative; it must be at `$OUTDIR/init.png`. For the base-stage
control, change only `HIRES=0`. Do not use
`LTX_REFINE_TEMPORAL_ANCHOR_REFRESH` in a quality run until it has a convincing A/B.

## Confirmed I2V regression boundary

User report: the first I2V output frame is pink/yellow/purple tiled blocks, then later frames
become plausible but soft. The input PNG is valid.

| Run | Result |
| --- | --- |
| `ltx-denoise-repro/_i2v_stage1_control_97f/` (`HIRES=0`) | Clean base-resolution I2V opening frame. |
| `ltx-denoise-repro/_i2v_anchor_smoke_97f/` (`HIRES=1`) | Corrupt tiled opening frame. |
| `_i2v_skip_stage2_97f/` (temporary diagnostic: stage-two image conditioning off) | Still corrupt. |
| `_i2v_spatialonly_97f/` (temporary diagnostic: no spatiotemporal VAE blend/stream) | Still corrupt. |

So this is not the VAE streaming/tiled decode change and not stage-two source reinjection alone.
I2V has no usable continuation reference in this repro, so `temporal_refine_enabled` is false
and it performs a full refine sample: `LTX_REFINE_TEMPORAL_BLEND` is not responsible either.

Narrowest hypothesis: stage-one I2V’s frozen/in-place image latent reaches the latent spatial
upscaler in an unsupported representation and corrupts the first causal high-res block. This is a
regression in the high-res I2V/upscaler path; I2V previously worked with upscale/refine.

Rejected/reverted experiments; do not revive without matching Comfy semantics:

- raw latent concatenation as a separate `video_reference`/appended keyframe guide;
- post-decode replacement of the opening frame;
- diagnostic high-res conditioning skip.

Appending still produced tiled stage-one output: this C++ sampler does not replicate Comfy
`clean_latent` behavior by simply concatenating a guide tensor.

Relevant blocks in `src/stable-diffusion.cpp` (line numbers drift):

- stage-one I2V encode/mask and `apply_ltxav_condition_image_by_latent_index` (~7060);
- stage-two `apply_ltxv_refine_image_conditioning` (~8754);
- base sample/window selection (~9470);
- latent spatial upscaler (~9700);
- refine/window branch (~10200);
- `decode_video_outputs` (~8263).

First CPU-only task: use `git log -p` and `git blame` around those blocks to identify the
regressing contract/change. Do not add speculative probes to the clean baseline.

## Existing quality/VRAM evidence

- Good short candidates: `_final_anchor_seed123_241f` and
  `_final_anchor_seed777_241f` (241 frames; seed 777 is visually more stable).
- Bad 20-second quality candidate: `_final_single20s_anchor/singing_clip.webm`, 473 frames,
  ~1000.7 s engine time, 11,396 MiB peak. Periodic generated-anchor refresh is the main suspect.
- Chain output: `_final_chain2x360_upscalefree/singing_clip.webm`, 682 frames; nominal
  15-second segments completed at 11,258 MiB peak.
- The old 13.4GB continuation spike was stage-one DiT still resident during latent-upscaler
  Conv3D, not a progressive leak. `f128441` frees it before upscale.
- 2- and 3-segment 97-frame reset smokes (`_chain2_reset_97f`, `_chain3_reset_97f`) peak at
  10,934 MiB. An earlier claim that continuation necessarily still hits 13.3GB is stale.

## Community evidence and intended architecture

1. [RuneXX long-video I2V/T2V workflow](https://huggingface.co/RuneXX/LTX-2.3-Workflows/blob/main/Long-Video-Experimental/LTX-2.3_-_I2V_T2V_Long_Video_Custom_Audio_singlepass_loop.json):
   ~497-frame windows, 25-frame overlap; prior tail/latest generated I2V for local continuation,
   but the original reference remains separately in `LTXVAddLatentGuide` at `latent_idx=-1`,
   strength 1. This is the target separation: immutable identity plus recent motion.
2. [Lightricks negative-index guide implementation](https://github.com/Lightricks/ComfyUI-LTXVideo/blob/master/latents.py#L401-L470):
   replicate its guide semantics, not its tensor shape superficially.
3. [WhatDreamsCost Director guide](https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI/blob/main/ltx_director_guide.py#L477-L600):
   timeline keyframes, `LTXVAddGuide`, optional IC-LoRA; nonzero-time image guides ramp their
   first two latent strengths 0.25 → 0.65. Local checkout:
   `/tmp/wdc-comfy/ltx_director_guide.py`.
4. [Official Lightricks looping sampler](https://github.com/Lightricks/ComfyUI-LTXVideo/blob/master/looping_sampler.md):
   overlap plus `optional_negative_index_latents` for global coherence (typically 25–30%
   overlap), modest AdaIN 0.1–0.3.
5. Optional experimental multi-reference direction:
   [RuneXX Licon-MSR workflow](https://huggingface.co/RuneXX/LTX-2.3-Workflows/blob/main/Multi-ref-character-sheet/LTX-2.3_-_I2V_multi-subject-reference_Licon-MSR-lora.json).

## Work order

1. CPU-only: identify the specific high-res I2V regression and compare current construction to
   Lightricks/Comfy guide construction. Write a concrete proposal before editing.
2. Once GPU is released: rebuild and reproduce 97f `HIRES=0`/ `HIRES=1`; validate a minimal
   root-cause repair before long tests.
3. Implement keyframes as a timeline. Each keyframe must be encoded at its stage resolution; at
   high-res refine guide/reinject it at its intended latent index, not by replacing pixels after
   decode. Keep an immutable source guide across every temporal window and a short tail only for
   motion context.
4. T2V: avoid global anchor refresh. Test larger overlap and a deterministic best-handoff-frame
   selector (low motion, sharpness, no occlusion) for local continuity versus literal-tail handoff.
   Character/IC LoRA is the stronger answer where identity must persist.
5. Validation order: 97f high-res I2V → 241f high-res windowed I2V → keyframe after frame zero →
   multi-window/continuation I2V → T2V A/B. Separately commit/push only evidence-backed changes.
