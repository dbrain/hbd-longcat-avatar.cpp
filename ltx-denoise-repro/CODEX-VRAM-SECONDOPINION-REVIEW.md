# Codex VRAM continuation second-opinion review

Date: 2026-07-07

Scope: fresh code read of the continuation/refine VRAM issue starting from
`VRAM-CONTINUATION-SECONDOPINION.md`. I intentionally did not read the existing
repo markdown trail beyond that brief.

## Bottom line

The frame-count diagnosis is basically right, with one implementation nuance:
the deployed warm chain appears to use the LTXAV in-memory continuation
`keyframe-append` path, not the file-based head-overwrite path. That path appends
the K prior latent frames as extra guide tokens to the base target sequence. For
97 pixel frames, base T is 13 latent frames; with K=3, refine T becomes 16. The
1080p refine sees all 16 latent frames after spatial latent upscale, so the
single-121f diagnostic should land near the continuation seg-2 peak if allocator
state is comparable.

There is probably no "free" memory lever that keeps K=3, 1080p, full dense
attention, and full 5.47 GB shared-resident DiT all unchanged. The most promising
non-quality lever is not full `free-DiT-in-refine`; it is a capped/partial
shared-resident set for the refine, saving only the 1.2-1.6 GB needed instead of
streaming the whole 5.47 GB payload.

## Code-read findings

1. Continuation frame growth is real on the in-memory chain path.

   In `generate_video_chain`, seg>0 passes `vp.cont_latent = cont_buf.data()` and
   `vp.cont_latent_frames = K`. The LTXAV preparation path then goes through the
   in-memory continuation branch and, by default, calls
   `apply_ltxav_video_guide_by_keyframe_index`. That helper concatenates the prior
   latent frames after the target latent sequence and builds positions for
   `target_latent_frames + keyframe_frames`.

   That is exactly the 13 -> 16 latent-frame growth in the brief.

2. The file-based `--cont-latent` branch is different.

   The `cont_latent_path` branch loads the prior tail, writes it into the head of
   `init_latent`, and holds those slots fixed. It does not grow T unless an
   optional appearance anchor consumes another head slot. That path can reduce
   VRAM, but it changes semantics: the segment reuses head slots for overlap, so
   if you later drop overlap frames you get fewer new frames per segment unless
   you request more frames.

3. Hires/refine preserves the grown temporal length.

   Before latent spatial upscale, `generate_video_ex` saves the base latent for
   the next chain segment. Then the full current `final_latent` is spatially
   upscaled and passed into the hires refine. Spatial upscale changes W/H, not T.
   Therefore the appended K guide frames are paid at 1080p during refine.

4. The VRAM breakdown line is a true reserve-time high-water, but `overhead` is a
   bucket, not a single allocation class.

   The code logs `driver_used` immediately after `ggml_gallocr_reserve`, plus the
   current compute buffer and known param buffers. The residual includes CUDA
   context, CUDA/cuDNN workspaces, backend/VMM pool high-water, graph/cache
   buffers, allocator fragmentation, and anything not represented by the named
   param buffers. It is very plausible that this residual scales with sequence
   length, but I would avoid calling it all "flash-attn overhead" without a tensor
   dump and CUDA allocator trace.

5. `LTX_MAX_VRAM` failing to move the 5.47 GB resident set is expected from the
   current code.

   The shared-resident set is selected by `compute_shared_resident_set`: any
   tensor read by at least `LONGCAT_SHARED_RESIDENT_MIN_SEGS` graph-cut segments
   is pinned. There is no byte budget in that selection. `max_vram` affects graph
   segmentation and streaming budget, not the final size of this derived resident
   set.

6. There is already a crude existing resident-set knob.

   `LONGCAT_SHARED_RESIDENT_MIN_SEGS` defaults to 2. Raising it should shrink the
   set if the reuse-count distribution is not all-or-nothing. It may be too coarse,
   but it is worth one measurement before coding a new policy:

   - Try `LONGCAT_SHARED_RESIDENT_MIN_SEGS=3`, then maybe 4.
   - Record resident MB, refine s/it, and total peak.
   - If resident stays ~5471 or collapses too far, implement a byte cap.

## Options

### Option A: K=1 or K=2

This is the cleanest production lever. It attacks the true multiplier: sequence
length.

Expected from the provided numbers:

- K=2: T=15, about -817 MiB from current, likely still around 12.1-12.3 GB.
- K=1: T=14, about -1.6 GB, likely fits under 11.5 GB.

K=2 is only viable if combined with a smaller allocator/resident win. K=1 is the
lowest engineering risk, but it spends continuity margin.

Recommendation: run seam-stress A/B for K=1 and K=2 regardless of other work.
If K=1 is visually acceptable, take it.

### Option B: partial shared-resident cap for refine

This is the best "keep quality/continuity/resolution" engineering option.

Current behavior pins the whole derived shared set, ~5471 MB. The brief's target
needs roughly 1.2-1.6 GB less, not a full 5.47 GB eviction. Add an env such as:

`LTX_SHARED_RESIDENT_MAX_MB=3800`

Selection policy:

- Compute the usual shared set and per-param segment reuse count.
- Sort by reuse count descending, then bytes ascending or original graph order.
- Keep tensors until the byte cap is reached.
- Stream the rest through the existing partial/prefetch path.
- Apply this only to LTXAV/large offloaded DiT, or gate by env.

Why this is promising:

- It preserves most of the speed benefit.
- It directly targets the fixed resident term that is keeping refine above the
  cap.
- It is much cheaper than full `release_all_gpu_param_residency()` before refine.
- The existing `filter_out_resident` and partial offload path already support
  streaming tensors not in `resident_param_set`.

Risks:

- Need to ensure the carried-forward healthy set respects the cap too.
- Need stable deterministic ordering so runs are reproducible.
- If the hot set has poor granularity, speed may degrade more than expected.

Fast precursor test: try `LONGCAT_SHARED_RESIDENT_MIN_SEGS=3/4`. If that saves
~1.5 GB with acceptable refine time, a byte cap may not be necessary.

### Option C: refine-only resident policy

A refinement of Option B: cap or disable shared-resident only for the 3-step hires
refine, while leaving base sampling fully pinned.

The code currently does not label `compute_with_graph_cuts` as base vs refine.
But `generate_video_ex` knows the phase immediately before calling
`sample(... hires_sigma_sched ...)`. A scoped env/runner flag could temporarily
set a smaller resident cap for that sample only.

This is attractive because the refine has only 3 steps; streaming some weights
there is less painful than slowing the 8-step base pass.

### Option D: full free-DiT before refine

This is the guaranteed fit lever. It frees the whole resident DiT payload before
the refine and lets every refine step cold-stream or rederive/offload.

Use this if partial residency takes too long to stabilize. It should preserve
quality and continuity, but the time penalty is real and lands in every segment.

### Option E: legacy/head-overwrite continuation for memory

The file/head-overwrite semantics keep T at 13 for a 97-frame request, so refine
should look closer to the single 97f case.

The catch is output accounting. If those first K latent slots represent overlap
and the stitcher drops overlap frames, each segment yields fewer new frames. To
keep the same final duration, you either need more segments or larger requested
segment lengths, which gives back some of the memory benefit.

I would treat this as a workflow design option, not a drop-in fix.

### Option F: reduce output resolution or refine tokens

The 1.5x upscale option is valid but is a quality lever. Another variant would be
refine only the new frames and carry/upscale/decode the fixed overlap separately,
but dense temporal attention means the generated frames would no longer attend to
the refined overlap tokens in the same graph. That is a model-behavior change, not
just an allocator optimization.

## Diagnostics I would run next

1. Single 121f / 16-latent render with the exact same env.

   This confirms whether continuation-specific state matters after controlling
   for T. Expected: close to seg-2 refine peak.

2. `LONGCAT_TENSOR_DUMP=1` on single 97f and continuation seg-2 refine.

   The code already dumps top graph tensors and gallocr peak. This will separate
   "compute buffer grows linearly with T" from "residual allocator/workspace grows
   outside gallocr".

3. `LONGCAT_SHARED_RESIDENT_MIN_SEGS=3` and `=4`.

   This is the cheapest probe for partial residency. Capture:

   - `shared-resident set: N params (M MB)`
   - `[VRAM] ltxav reserve`
   - refine seconds/iteration
   - final peak

4. K=1/K=2 seam stress.

   Test continuous camera motion and persistent subject motion. K=2 plus a small
   resident reduction is probably the nicest compromise if seam quality holds.

## Smallest likely path to <= 11776 MiB

Most practical:

1. Try K=1. If acceptable, ship it.
2. If K=1 is not acceptable, try K=2 plus resident-set shrink.
3. First resident shrink probe is `LONGCAT_SHARED_RESIDENT_MIN_SEGS=3`.
4. If that is too coarse, implement `LTX_SHARED_RESIDENT_MAX_MB` and target a
   refine resident set around 3.8-4.2 GB.
5. Keep full `free-DiT-in-refine` as the fallback fit switch.

My preferred engineering bet is Option C: a refine-only resident byte cap. It
keeps the current visual behavior and pays only the amount of speed needed to get
under the cap.
