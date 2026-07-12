# LTX temporal windowing handoff

## Repository and build

Worktree: `/home/dbrain/dev/longcat-avatar-temporal`.

The runnable binary is built in `/home/dbrain/dev/longcat-avatar-ltxdenoise` because it has the
CUDA build tree. Build the changed sources with:

```bash
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

Use `ltx-denoise-repro/run_singing_clip.sh`. It defaults to the production CUDA/FP4 stack and
has the relevant controls:

```bash
env WT=/home/dbrain/dev/longcat-avatar-ltxdenoise GPU=1 SEED=42 \
  SEGMENTS=1 FRAMES=241 HIRES=1 MAXV=9 \
  VAE_SPATIAL_TILES=1x2 VAE_SPATIOTEMPORAL_BLEND=1 VAE_TEMPORAL_STREAM=1 \
  VAE_TBLEND_FRAMES=8 VAE_TBLEND_OVERLAP=2 \
  BASE_TBLEND_FRAMES=14 BASE_TBLEND_OVERLAP=4 \
  REFINE_TEMPORAL_BLEND=1 REFINE_TBLEND_FRAMES=8 REFINE_TBLEND_OVERLAP=2 \
  OUTDIR=$PWD/ltx-denoise-repro/_candidate \
  bash ltx-denoise-repro/run_singing_clip.sh
```

For live DiT/VAE attribution, add `LONGCAT_LIVE_VRAM=1`. The harness samples aggregate GPU
memory to `OUTDIR/vram.log`.

## Current validated runtime result

Single 10-second high-res clip, seed 42:

- 241 frames / 10.041s / 1920x1088
- base window `14/4`, refine window `8/2`, VAE stream `8/2`, spatial VAE `1x2`
- total `461.83s`
- peak `11120 MiB`
- output: `ltx-denoise-repro/_window14o4_refine8_full_241f/singing_clip.webm`

The 12/2 refine alternative had the same peak but was slower (`483.75s`), so 8/2 is current
runtime winner. `MAXV=9` and normal DiT shared residency are retained.

Single 20-second high-res clip also completed:

- 480 frames / 20.0s / 1920x1088
- total `966.91s`
- peak `11394 MiB`
- output: `ltx-denoise-repro/_window14o4_refine8_full_480f/singing_clip.webm`

The duration-dependent active sets are correctly window-bounded: no OOM and no growth beyond
the VAE's 10 temporal tiles.

## What was changed

- Audio-aligned base DiT temporal windows with absolute video/audio RoPE positions.
- Frozen-overlap base windows, upscaler windows, and refine windows; final tails are re-anchored
  to a regular full-size tile to avoid short-tail graph-cut VRAM spikes.
- High-res refine windows use absolute positions, preventing post-seam lip-sync loss.
- Causal VAE temporal streaming and spatial `1x2` feathered decode support long clips.
- Live VAE/DiT memory attribution (`LONGCAT_LIVE_VRAM=1`).
- The singing harness now exposes frame count, seed, all temporal-window controls, and frees the
  text encoder's unused runtime copy after CPU embeddings are materialized.
- Continuation high-res references can now be supplied to each refine tile with combined
  target+reference absolute positions. Previously their presence disabled refine windowing and
  forced a full 15-second stage-2 allocation/OOM.

## Continuation status

The repaired `SEGMENTS=2 FRAMES=360` test completed and stitched `682` output frames
(`28.42s`, after causal VAE and 24-frame seam trimming). The former stage-2 refine OOM is fixed.
However, segment-2 decode currently peaks at `13326 MiB`, above the 11.5 GiB target. Do not call
continuations memory-complete until that decode co-residency is attributed and removed.

Output: `ltx-denoise-repro/_window14o4_refine8_chain2x360f_refwindow/singing_clip.webm`.

### Continuation decode investigation, avoid full rerenders

The seg-2 decode log shows `DiT_gpu=0`, `VAE_gpu=0`, and `audioVAE_gpu=0` at decode entry, yet
the process rises to 13,326 MiB while the VAE tiled decode runs. The active VAE set itself is
roughly 1,385 MiB weights + 7,243 MiB compute + 520 MiB cache, so roughly 2 GiB is unattributed
process-global CUDA/cuDNN state. The continuation's high-resolution reference encode creates
larger conv3d plans immediately before refine, making cuDNN plan/async-pool retention a leading
hypothesis. There is an existing `ggml_backend_cuda_release_cudnn_plans()` and an
`LTXAV_CHAIN_CUDNN_RESET` boundary hook; test a *pre-decode* reset carefully, but do not commit it
without an A/B peak measurement.

Do not repeatedly run a 29s chain to test this. The engine can bank per-segment latents with the
existing `LTXAV_SAVE_VIDEO_LATENT` path; inspect `generate_video_chain` around the per-segment
save/continuation logic and use a banked seg-0 tail/refined guide or `LTX_LOAD_LATENTS` to isolate
the seg-2 refine/decode. A useful test harness should separate: (1) stage-2 continuation reference
encode, (2) refine, and (3) VAE decode, recording board memory after each. Reuse the exact 45-latent
frame / 1x2 / VAE-8/2 geometry from the failed seg-2 decode rather than re-running the full chain.

## Next goal: preserve identity / avoid motion mush

The user observes crisp frames early in a clip but progressive softness and a hair-scale/identity
change near 6-7s, especially while the subject bops/dances. The 8/2 refine boundary is exactly
at 6.67s, strongly implicating refine-window context rather than a random point in the model.

Important findings:

- The only visual state currently carried between DiT windows is frozen contiguous latent overlap.
  There is no LTXAV transformer K/V cache.
- Every new window recomputes Q/K/V at every diffusion sigma. A useful 4-frame K/V history would
  cost about 1.5 GiB at base resolution and about 6 GiB at high resolution: not viable.
- The model uses one global noise tensor sliced by windows, so boundary drift is primarily loss of
  old visual context, not independent reseeding.
- Do not solve this by a fixed external portrait: the solution must work for T2V and I2V, where
  the subject may emerge after frame zero. A promising research direction is an internal,
  model-generated sharp anchor selected after the subject is established, carried as a separate
  reference-token block while recent contiguous overlap remains motion context. It must be allowed
  to update/re-anchor rather than locking the initial pose.

Before implementing a reference scheme, diagnose whether softness begins in base latent, refine,
or VAE output. Compare output around each boundary and use the 12/2 render (boundary shifted to
7.33s) as an A/B against 8/2 (boundary 6.67s). Research common T2V/I2V temporal-anchor methods;
do not assume a fixed first-frame identity image.

Also exercise I2V/reference-image paths when changing temporal/reference logic. The continuation
guide code intentionally only enables same-grid continuation guides; relip/reference-image grids
can have different geometry and remain on their pre-existing full path.

## Overnight-agent objective

Keep the single-window duration win intact (20s under 11.5 GiB), then:

1. bring continuation segment-2 decode under 11.5 GiB without changing MAXV=9 or reducing DiT
   residency as a blanket budget lever;
2. diagnose and reduce temporal-window mush/identity drift for T2V and I2V, especially motion at
   a refine boundary;
3. prefer short latent/reuse or phase-isolated experiments over full 15-30 second rerenders;
4. commit and push each verified improvement on `ltx-temporal-hybrid`.
