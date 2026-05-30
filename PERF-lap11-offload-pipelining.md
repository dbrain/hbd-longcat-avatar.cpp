# lap-11: offload-path pipelining — 2026-05-30

Two levers in the weight-OFFLOAD path (default `--offload-to-cpu --mmap`), the recoverable slice of
lap-10's keep-resident finding (offload costs ~13% wall but is VRAM-budget-forced; qwen 4302 + UNet 5636
= 9938 MB > 7.5 GB so they alternate per gen). Budget ≤7.5 GB (7680 MiB) and bit-exactness held.

Baseline (default, warm, n=4 median): wall **15.84s** / cond **1.72s** / dit 13.61s (1.704/step) / vae 0.41s /
peak **6557 MiB** / md5 `6c0a783425ea`.

## LEVER A — kill the redundant 2nd qwen offload — **LANDED, bit-exact**
**Root cause.** CFG does two consecutive text encodes (cond `stable-diffusion.cpp:4219`, uncond `:4235`),
`free_params_buffer()` only after both (`:4258`). Each encode is `GGMLRunner::compute(..., free_compute_buffer_immediately=true)`
(`llm.hpp:1691`). `execute_graph`'s success tail calls `free_compute_buffer()`, which (`ggml_extend.hpp:3264`)
called BOTH `restore_partial_params()` AND `restore_all_params()` — and `restore_all_params` sets
`params_on_runtime_backend=false` + frees `runtime_params_buffer`, pushing the whole 4302 MB qwen back to CPU.
So the uncond encode's `offload_all_params()` re-uploaded all 4302 MB (the 2nd "qwen3 offload params … 0.6s"
H2D in the log). Confirms lap-10's "qwen offloads TWICE".

**Fix (narrowly gated).**
- `ggml_extend.hpp`: added `bool keep_params_resident_` (default false). `free_compute_buffer()` now skips
  `restore_all_params()` when set (still frees the compute allocr = activations, and still
  `restore_partial_params()`). Added `set_keep_params_resident(bool)`; clearing it (false) immediately
  `restore_all_params()` (matches default end-of-compute behaviour → frees runtime VRAM).
- `conditioner.hpp`: virtual `set_keep_params_resident(bool)` on `Conditioner` (default no-op);
  `LLMEmbedder` override forwards to `llm->set_keep_params_resident`.
- `stable-diffusion.cpp` `prepare_image_generation_embeds`: set resident=true before the cond encode
  (only when `use_uncond || use_high_noise_uncond`), clear=false after both encodes (restores qwen to CPU
  before the UNet loads — VRAM model unchanged), then the existing `free_params_buffer()` runs.

**Why VRAM-safe & bit-exact.** Same 4302 MB qwen is resident either way; we only skip the round-trip
CPU→GPU between the two encodes. Activations (compute allocr) are still freed between encodes, so the
co-residency peak is identical. The math is untouched — only WHEN weights move.

**Measured (n=4 median, default config):**
| metric | baseline | lever A | Δ |
|--|--|--|--|
| wall | 15.84s | **15.21s** | −0.63s (−4.0%) |
| cond | 1.72s | **1.03s** | **−0.69s (−40%)** |
| dit | 13.61s | 13.60s | flat |
| vae | 0.41s | 0.41s | flat |
| VRAM peak | 6557 MiB | **6549 MiB** | flat (within 7.5 GB) |
| md5 | 6c0a783425ea | **6c0a783425ea** | **MATCH, PSNR 999** |

Log proof: each gen now shows exactly ONE `qwen3 offload params (4302 MB) … 0.56s` per gen (was two).

## LEVER B — pipeline the UNet step-0 H2D with step-0 compute — **NOT achievable w/ reasonable effort; reverted**
**Target.** The flux DiT runs with `free_compute_buffer_immediately=false` (`flux.hpp:1490`), so its weights
load once via a blocking bulk `offload_all_params()` (5636 MB, "flux offload params … 0.77s") at sampling
start and stay resident for all 8 steps. The GPU is idle during that 0.77s (~5% wall, one-time). Goal: stream
UNet weight-groups on a copy stream while step-0 consumes each group as it lands.

**Finding: the existing lap-32.2 pipelining machinery does NOT engage for the flux DiT, and cannot by config.**
The pipelining (`compute_with_graph_cuts` + `kick_off_prefetch`/`commit_prefetched_state`/`partial_prefetch_event_`)
is SEGMENT-granularity and only fires when `should_use_graph_cut_segmented_compute()` is true:
`has_cuts && segments.size()>1 && max_graph_vram_bytes>0 && offload-mode-CUDA`. The default serve passes no
`--max-vram` → `max_graph_vram_bytes=0` → `can_attempt_graph_cut_segmented_compute()` false → the DiT always
takes the plain `execute_graph` path with one bulk `offload_all_params`. The whole machinery is dormant for
flux2's default config.

**Probed whether `--max-vram` can force it (empirical, n=2-4):**
- `--max-vram 4`: qwen plan "merged 29 segments → 1" (whole graph fits budget); **NO flux cut plan emitted**;
  flux still does the single bulk offload (5636 MB, 0.84s); VRAM peak unchanged (6549). No win.
- `--max-vram 1`: qwen now segments (29→5) and DOES use the pipelined path — but `segment_overhead=437ms`
  dominates (per-segment build/plan), cond rises to ~1.54s, and crucially **Lever A's keep-resident is lost
  on the segmented path** (params stream per-segment, never held by `offload_all_params`). The **flux DiT
  STILL does the single bulk `offload_all_params` (5636 MB, 0.72-0.77s)** — it does not segment even at a
  1 GiB budget.

The flux DiT has cut markers (`flux.hpp:931-957`, per double/single block) but its base plan resolves to a
single segment for the budgeter (`apply_max_vram_budget` early-returns when `segments.size()<=1`) — unlike
the qwen layer-stack which splits cleanly. The graph-cut budgeter keys on per-segment *activation+input-param*
VRAM (`graph_cut_segment_vram_bytes`), not on overlapping weight-load with compute; even when it does split
(qwen), it RE-STREAMS weights every call and adds large segment overhead — the opposite of what an 8-step
gen wants (weights resident once, reused 8×). Engaging it for the DiT would turn one 0.77s H2D into 8×
per-step streaming + overhead = a net LOSS.

**What a real Lever B would require (NOT attempted):** a from-scratch restructure of the flux DiT forward —
manually split into N weight-groups, allocate per-group runtime buffers, `cudaMemcpyAsync` each on the copy
stream, and `ggml_backend_event_wait`-order each group's consumption WITHIN step-0 only (then keep resident
for steps 1-7). This is intra-graph layer-by-layer pipelining the segment machinery was never built for.

**Decision: STOP (per task guidance).** Ceiling is ~0.77s (~5% wall, one-time). Against that: (1) silent
corruption failure mode (a weight consumed before its H2D completes → NaN image, only caught by md5/PSNR),
(2) the offload pointer-swap area (`offload_all_params`/`restore_all_params`/`restore_partial_params`
buffer/data/extra swaps) is exactly where a prior segfault lived (dangling `free_params_buffer` ptrs), (3) no
reusable plumbing — the segment path doesn't even apply to the DiT. A from-scratch restructure for a ~5%
one-time win with high corruption + lifetime risk is not "reasonable effort." Reverted cleanly (no Lever B
code in tree). Re-open only if the offload model itself is replaced (e.g. a true async-prefetch layer hook in
GGMLRunner that keeps weights resident after step-0).

## Net
- **Lever A LANDED**: −0.63s wall (−4.0%), cond −40%, bit-exact (md5 6c0a783425ea, PSNR 999), VRAM 6549 MiB
  (within 7.5 GB). Uncommitted.
- **Lever B NOT achievable** with the existing machinery and not worth a from-scratch restructure; documented
  with evidence, no code left in tree.
- New default warm baseline: **wall 15.21s** / cond 1.03s / dit 13.60s / vae 0.41s / peak 6549 MiB /
  md5 6c0a783425ea.
- Files touched (Lever A, uncommitted): `src/ggml_extend.hpp`, `src/conditioner.hpp`, `src/stable-diffusion.cpp`.

## Lever C — keep the UNet resident across batch_count seeds — LANDED, bit-exact
Multi-seed batches (`batch_count > 1`, seeds = seed, seed+1, …) reuse the qwen text-encode (computed
once before the batch loop) but were RE-UPLOADING the 5636 MB UNet for every seed: each `sample()` ends
by restoring params to CPU, so the next seed pays another ~0.8s H2D. Nothing else touches VRAM between
seeds (text encoder already freed, VAE runs after the loop), so we keep the UNet resident across the
loop via Lever A's `set_keep_params_resident()` flag, cleared before the post-loop free / VAE decode.
- `src/stable-diffusion.cpp` `generate_image`: `set_keep_params_resident(true)` before the batch loop
  (gated `batch_count > 1`), `(false)` after the loop and on the failure path. `set_keep_params_resident`
  is inherited from GGMLRunner (diffusion_model is a `DiffusionModelRunner`) — no passthrough needed.
- **Measured: batch=3 → ONE flux UNet offload (was 3); wall 41.52s; VRAM peak 6549 MiB (unchanged);
  seed-42 image md5 6c0a783425ea bit-exact.** Single-gen (batch=1) path untouched (gate is no-op):
  wall 15.33s, md5 6c0a783425ea.
- For the standard 3-seed workflow: **3 separate gens 3×15.33 = 46.0s → batch=3 41.5s (−4.5s, −10%)**
  (qwen shared once + UNet shared once + activations freed between seeds → peak flat). Saves ~0.8s per
  extra seed beyond the qwen sharing. Files: `src/stable-diffusion.cpp` (uncommitted).
