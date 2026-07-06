# LTX-2.3 1080p continuation-chain crash — root-cause analysis

Repo: `longcat-avatar-ltxdenoise` @ `ltx-denoise-workflow`
Engine: `src/stable-diffusion.cpp`, offload core `src/core/ggml_extend.hpp`, planner `src/core/ggml_graph_cut.cpp`.
Code analysis only (no build / no GPU).

## Symptom (as captured)
Multi-segment chain (`generate_video_chain` → `generate_video_ex`/seg) with x2 hires upscale.
- Seg 1 @1080p (960×544 base → refine) completes.
- Seg 2 @1080p (continuation keyframe-append prepended) dies **at the BASE sample, step 1** —
  before the upscale/refine — with the base step ~**4× slower** than seg 1 (34.6 vs ~9 s/it).
  Forked CUDA child closes cleanly (EOF), **no ggml/CUDA assert**, GPU ~11 GB, host ~17 GB free.
- Seg ≥2 @720p, upscale=FALSE: all 8 segments fine.
- Not audio (repro'd with audio=none), not the raw memory ceiling.

Log right before death:
```
generate_video 960x544x97 -> LTX latent spatial upscale
LTXAV_VAE_LAZY: released offloaded video+audio VAE GPU params ... before DiT sample+refine
ltxav graph cut max_vram=7168 merged 50 segments -> 3 segments
|======>  1/8 - 34.60s/it   <- 4x slow, then child dies
```

## Root cause (ranked)

### #1 (primary) — seg-2 base cold-re-offloads the DiT against a STALE, inflated streaming budget → mis-planned graph-cut overruns the board mid-step

Two independent things drop the DiT's GPU residency between seg 1 and seg 2:
1. `LTXAV_DIT_FREE_DURING_DECODE` inside seg-1's `generate_video_ex`
   (`stable-diffusion.cpp:9285-9297`): `diffusion_model->release_all_gpu_param_residency()`
   **and** `ggml_backend_cuda_trim_pools(DIFFUSION)` — frees the 5471 MB shared-resident DiT
   payload **and returns the DIFFUSION VMM pool high-water to the OS**, before seg-1's decode.
2. `generate_video_chain` between segments (`stable-diffusion.cpp:10217-10220` →
   `release_chain_segment_gpu_residency()` `:3338`, which at `:3347` does
   `release_streaming_residency()` + `free_compute_buffer()` on `diffusion_model`).

So seg 2 starts with the DiT fully host-offloaded (`params_offloaded_to_host()==true`,
`params_backend != runtime_backend`) and a pool that was just trimmed to the OS. Seg-2's base
`sample()` (`:8970`) must therefore **cold re-offload all 5471 MB from host** and re-grow the pool.

The killer is the streaming budget in `GGMLRunner::…` (ggml_extend.hpp `:3700-3749`):
- `:3706-3708` — while the DiT is resident (seg 1), its own `resident_runtime_params_buffer`
  size is **added back to free_vram** ("this buffer will be reused"), so seg-1's `effective_budget`
  is computed as ≈ whole board.
- `:3722-3729` — `observed_max_effective_budget_` is a **monotonic max**: once seg-1 latches the
  inflated value, seg-2 does `effective_budget = observed_max_effective_budget_` even though at
  seg 2 the resident buffer is gone (null → no self-credit) and real free VRAM is much lower.
- **Neither `release_streaming_residency()` (`:4393` → `restore_resident_params`) nor
  `release_all_gpu_param_residency()` (`:4761`) resets `observed_max_effective_budget_`.** It is
  reset ONLY on the `free_params_buffer` path (`:4695`), which a warm resident chain never calls.

Net: seg-2's base graph is planned by `resolve_plan` (`:3751`) against a budget larger than the
real free VRAM. The planner keeps the merge coarse — `"merged 50 segments -> 3 segments"`
(ggml_graph_cut.cpp `:830`) — and the DiT streams per-segment from host (→ 34.6 s/it ≈ 4×). The
cold re-offload upload buffer stacked on the 3 fat streamed compute segments overruns the board
mid-step; the child dies on an async illegal-access / failed `cudaMalloc` (no ggml assert; the
coarse 11 GB sample misses the sub-second transient because the pool was trimmed to OS and the
re-grow allocations are large + fragmented).

**Why seg 1 survives:** DiT freshly resident, pool warm, budget correctly self-credited — no cold
re-offload, single resident graph, ~9 s/it.

**Why 720p/upscale=FALSE survives:** it is the *hires* path that forces the render into the tight
`max_vram=7168` regime (baked MAXV=7, commit `c7f019e`) where the base graph must be cut into
multiple streamed segments. The smaller 720p base graph fits real free VRAM in a single/loose cut
even under the stale inflated budget, so the same cold re-offload completes cheaply and all 8
segments run. (Continuation adds only +3 latent frames — 13→16 — which alone is harmless: proven by
720p seg-2 running fine. It is the *interaction* cold-DiT-reoffload × tight-hires-cut that kills.)

This exactly matches the coordinator's capture: crash at seg-2 BASE, 4× slow, eviction flags on for
both runs, and "bad re-offload / huge transient allocation when hires is pending."

### #2 (secondary / contributing) — `LTXAV_DIT_FREE_DURING_DECODE` + chain reclaim are redundant and both trim the DiT before a segment that immediately needs it
On a chain segment that has a successor, freeing the DiT (and pool) at decode is pure waste: the
next segment re-materializes it 34.6 s later. The free is only a win on the *final* segment (frees
VRAM for its own VAE decode peak with nothing following).

### Ruled out
- **The 2nd VAE eviction (`LTXAV_VAE_LAZY`, `:9211`)**: fires identically on seg 1 and seg 2, and the
  plain-continuation refine path takes the early return in `apply_ltxv_refine_image_conditioning`
  (`:8433-8436`, no init/end image, not relip) so it never touches the VAE during refine. Not the
  differentiator. Also the crash is at the BASE, before the refine/2nd-eviction runs at all.
- **Dimensional/OOB in upscale or refine**: the crash precedes them; and the DiT forward handles the
  16-frame (target+K) grid fine (720p seg-2 proves it).

## Proposed fixes (in order)

**A. Don't free/trim the DiT during decode on a non-final chain segment (smallest, matches steer).**
`generate_video_ex` knows it is mid-chain when `final_latent_out != nullptr` (the chain passes
`want_latent = seg+1 < n_chain`, `:10036-10040`). Gate the `LTXAV_DIT_FREE_DURING_DECODE` block
(`:9285`) on `final_latent_out == nullptr`, i.e. only free on the last segment / non-chain render.
Optionally also skip the `diffusion_model` reclaim inside `release_chain_segment_gpu_residency()`
(`:3347`) when a successor follows, so the DiT stays warm-resident across the seam. Keeps seg-2's
DiT resident → no cold re-offload, no 4× streaming, no overrun.

**B. Fix the real bug: invalidate the latched streaming budget when residency is released.**
Add `observed_max_effective_budget_ = 0;` to `release_streaming_residency()` (`:4393`) and
`release_all_gpu_param_residency()` (`:4761`). The monotonic max is only valid while the
self-credited resident buffer exists; once it is freed, reusing the inflated max mis-plans every
subsequent graph. (Equivalently: at `:3722-3729`, don't raise `effective_budget` back to
`observed_max_` when `resident_runtime_params_buffer == nullptr`.) This hardens *all* release→re-
offload sequences, not just this chain.

Ship **A** as the immediate stopper and **B** as the correctness fix; they are independent and
compose.

## How to confirm (env A/B, no rebuild — do this first)
1. Re-run the crashing 1080p chain with **`LTXAV_DIT_FREE_DURING_DECODE=0`**.
   - Survives + seg-2 base back to ~9 s/it ⇒ confirms the DiT decode-free/re-offload trigger (fix A).
2. If it still dies, add **`LTXAV_NO_CHAIN_GPU_RECLAIM=1`** (disables
   `release_chain_segment_gpu_residency`, `:10217`) ⇒ isolates the reclaim path as the residency
   dropper.
One of these two flips localizes it definitively. To see the stale budget directly, log
`effective_budget` / `observed_max_effective_budget_` / real `free_clamp` at each base plan
(ggml_extend.hpp `:3711`/`:3727`): expect seg-2 `effective_budget` ≫ real free VRAM.
