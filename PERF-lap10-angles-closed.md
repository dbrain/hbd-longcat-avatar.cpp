# lap-10: the 3 remaining angles closed + 1024² numbers — 2026-05-30

Follow-up dig after lap-9 ("is there anything else?"). Three angles I'd asserted without testing,
now closed with evidence, plus a 1024² characterization.

## Angle 1 — matmul stall reason (ncu per-reason breakdown) → COMPUTE-BOUND, not latency
Earlier I called mul_mat_q "latency-bound at 16.67% occupancy." ncu per-reason warp-stall (mul_mat_q
<12,128,0>, grid 384) corrects this:
| stall reason | avg warps stalled |
|--|--|
| **math_pipe_throttle** | **0.61** ← dominant |
| not_selected | 0.51 (eligible work queued behind busy pipe) |
| wait (arith dependency) | 0.42 |
| long_scoreboard (global/L2 loads) | 0.29 |
| mio_throttle | 0.19 |
| short_scoreboard (smem/dequant) | 0.05 (negligible) |

**Verdict: the int8 dequant+MAC arithmetic pipe is SATURATED.** Memory is NOT the limiter
(long_scoreboard 0.29, short_scoreboard 0.05). This is why raising occupancy (smaller tile) made it
slower — there is no latency to hide; the math pipe is the ceiling. Confirms lap-5's "38% of int8 peak,
lost to Q4_K dequant" — the dequant integer ops compete with the MACs on the arithmetic pipe. The ONLY
matmul lever is fewer dequant instructions: a cheaper-unpack quant (Q4_0 = lower quality, off-table) or
Q8_0 (trivial dequant but 2× VRAM → ~11GB UNet, budget-gated). **No quality-neutral path.**

## Angle 2 — offload H2D overlap → qwen already hidden; only flux UNet (0.85s) serial
Clean timeline (8-step gen): wall 15.84 = cond 1.72 + dit 13.61 + vae 0.41 + overhead 0.10.
- `offload_all_params()` (ggml_extend.hpp) is a synchronous bulk H2D. Logs show qwen offloaded twice
  (0.63+0.65s, the 2 CFG encode passes) + flux UNet once (0.85s).
- **The qwen offloads (1.28s) are NOT on the critical path** — cond wall is only 1.72s (≈ the 2 compute
  graphs alone), proving the lap-32.2 pipelining backend already overlaps them. overhead is 0.10s.
- The **flux UNet load (0.85s, ~5.4% wall) IS serial**, inside the dit timer at sampling start. It can't
  prefetch during encode: qwen(4302) + UNet(5636) = 9938 MB > 7.5 GB budget (they alternate in the same
  VRAM — the offload is VRAM-budget-forced, same root as the dead keep-resident lever).
- **This is the single largest remaining LOSSLESS lever**: pipeline the UNet H2D with step-0 compute
  (stream layer-group weights, consume as they land) via the existing partial_offload machinery
  (kick_off_prefetch/events). Ceiling ~0.85s = ~5% wall. Deep (the partial path is wired for video
  segments, not intra-graph step-0 layer pipelining) + crash risk. Not attempted — flagged for decision.

## Angle 3 — layout-copies (6.7%) → architectural, need a custom attention kernel
- `cpy_perm_coalesced` (2.6%) = the `ggml_ext_cont(ggml_permute(q/k/v,0,2,1,3))` in ggml_ext_attention_ext:
  the qkv Linear emits head-INTERLEAVED tokens, but ggml `flash_attn_ext` requires CONTIGUOUS
  head-batched [d_head, L, n_head·N]. The cont is mandatory for FA's input layout; can't be removed by
  weight reorder (L is runtime). 
- `concat_T_cont` (2.7%) = `ggml_concat(txt, img, dim=2)` for q/k/v before joint attention — FA can't take
  two separate sequences, so the merge must materialize.
- Both are inherent to (ggml FA layout) + (joint attention). Removing them needs a custom fused attention
  kernel reading strided/split Q/K/V — deep, and FA is already at its 25% occupancy ceiling (lap-8), so a
  custom kernel wouldn't obviously win. Not viable.

## 1024×1024 characterization (klein-base, 8-step/cfg5/seed42, n=3 median)
| metric | 512² | 1024² |
|--|--|--|
| wall | 15.84s | **47.97s** (3.0×) |
| dit s/step | 1.70 | **5.55** (3.3×) |
| cond | 1.72s | 1.72s (size-independent) |
| vae | 0.41s | 1.47s |
| **VRAM peak** | 6557 MiB | **7819 MiB (7.64 GB)** |
- 1024² = ~48s/image. DiT tokens go 1536 → 4608 (4096 img + 512 txt); dit scales 3.3×.
- **VRAM peak 7819 MiB is ~140 MiB OVER the 7.5 GB (7680) budget** (cold run-0 was 7531, under; warm 7819).
  Fits physically on the 12 GB card, but breaches a 7.5 GB co-residency cap. Deterministic (md5 7363fd8fe492).
- If 1024² is wanted under budget: VAE tiling (`vae_tiling`) would cut the VAE-decode peak; the DiT-phase
  peak would need profiling to confirm which phase owns the 7819.

## Net
All listed angles closed. Matmul (69%) is arithmetic-pipe-bound (proven) — no quality-neutral path.
FA (8%) occupancy-walled. Layout (6.7%) architectural. Necessary kernels ~12%. The one real lossless
lever left is **UNet-load pipelining (~5%)** — deep, bounded, flagged. 1024² works at ~48s but peaks
~0.14 GB over the 7.5 GB budget.
