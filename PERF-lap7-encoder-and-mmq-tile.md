# lap-7: encoder early-stop (lossless, ~0 win) + MMQ tile-shrink (DEAD) — 2026-05-30

Baseline (klein-base 512²/8-step/cfg5/seed42, warm, n≥3 median, after rebuild):
**wall 15.83s · cond 1.62s · dit 13.63s (1.704 s/step) · vae 0.41s · peak 6557 MiB · md5 6c0a783425ea**

## Lever 1 — encoder early-stop at max(out_layers)  [SHIPPED, lossless, wall-neutral]
- `src/llm.hpp` `forward_embeds`: loop ran all 36 Qwen layers; flux2-klein consumes only
  `out_layers={9,18,27}`. Added `stop_layers = max(out_layers)` when `!return_all_hidden_states`
  and `max_out <= num_layers`. Runs 27/36 layers. **Bit-exact** (md5 unchanged) — layers 28-36
  feed only the unused final norm.
- **Result: wall-neutral.** cond stays 1.61s. Confirmed active via log ("running 27 of 36 layers")
  but the per-pass condition-graph compute stayed ~800ms (was ~800ms at 36 layers too).
- WHY no win: the Qwen encode at M=512 tokens is **latency/overhead-bound, not layer-throughput-
  bound**. FLOP math: full 36-layer Q4 compute should be ~220ms if compute-bound; measured ~800ms
  → 3.6× off → dominated by kernel-launch latency + attention overhead at small M. Cutting 25% of
  the non-dominant fraction → ~0 wall change. Kept anyway (strictly less work, frees nothing on the
  VRAM peak since peak is DiT-bound).
- The bulk weight offload ("qwen3 offload params 4302 MB, 398 tensors, 0.51s") is NOT graph-aware —
  it still H2Ds all 36 layers. A graph-aware offload could shave ~0.13s but it's deep (single bulk
  op in ggml_extend) for ~0.8% wall → not worth it.

## Lever 2 — MMQ mmq_x 128→64 (force smaller tile for occupancy)  [DEAD — measured regression]
Added env `GGML_MMQ_X_CAP` (mmq.cuh:~4063, caps the selector's mmq_x). Sweep:

| mmq_x | s/step | vs 128 |
|--|--|--|
| 128 (default) | 1.704 | — |
| 96  | 1.835 | +7.7% |
| 64  | 2.00  | +17% |

Monotonic: **larger tile = faster**. The selector's greedy fewest-tiles pick (128) is optimal.

**ncu proof the occupancy hypothesis is false** (DiT linear1 mul_mat_q, grid 384 vs 768):
| metric | cap128 | cap64 |
|--|--|--|
| Duration | 1.90 ms | 2.41 ms |
| Achieved occupancy | 16.66% | **16.66%** (unchanged!) |
| Registers/thread | 221 | **248** (rose) |
| Dyn smem/block | 57.86 KB | 48.38 KB |
| Block limit (regs / smem) | 1 / 1 | 1 / 1 |
| SM throughput | 52.4% | 46.0% |
| DRAM throughput | 13.0% | 15.7% |
| Eligible warps/sched | 0.81 | 0.72 |
| % cycles no eligible warp | 46% | 53% |

Shrinking the tile cut smem but the compiler **raised registers (221→248)**, so still 1 block/SM,
occupancy pinned at 16.66%. The smaller tile also dequantizes each Q4_K weight/activation tile and
reuses it across fewer output columns → worse dequant amortization (consistent with lap-5: Q4_K MMQ
is dequant-amortization-bound). NET: slower + no occupancy gain. **GGML_MMQ_X_CAP is a regression;
left in as 0/unset = upstream, for future experiments.**

## Lever 2b — force 2 blocks/SM via __launch_bounds__(...,2)  [DEAD]
`__launch_bounds__(256,1)` on Volta+ (MMQ_EXPERIMENT_MIN_BLOCKS define, default 1) lets the compiler
use up to 256 regs → 221 → 1 block/SM. Forcing minBlocks=2 caps regs at 128 (spills rest to local).
Needs cap64 too (smem 58KB×2=116KB > sm_86's 100KB/SM hard limit; cap64=48KB×2=96KB fits).
- Result: minBlocks=2 + cap64 = **1.856 s/step**. Better than cap64 alone (2.00) — so 2 blocks DOES
  hide some latency (confirms latency-bound) — but still **worse than the 128-tile baseline (1.704)**.
- You cannot have both: the optimal 128 tile needs 58KB smem → 2 blocks = 116KB > 100KB hard cap.
  Any tile small enough for 2 blocks loses more to dequant-amortization than occupancy buys.

## Lever 2c — shrink mmq_y (row tile) instead of mmq_x  [DEAD — architectural]
Idea: keep mmq_x=128 (good weight-dequant amortization across columns) but halve mmq_y 128→64 to cut
x-tile smem + accumulator regs → fit 2 blocks. **Won't compile:** `static_assert(nwarps*tile_C::I ==
mmq_y)` (mmq.cuh:3248). The mma fragment layout locks mmq_y = nwarps(8) × tile_C::I(16) = 128. To get
mmq_y=64 you must also set nwarps=4 → 4 warps/block × 2 blocks = **8 warps/SM = same 16.67%**. No gain.

## MATMUL VERDICT — occupancy path conclusively dead (3 angles)
The dominant DiT matmul (48% of GPU time) is latency-bound at a **hard 16.67% occupancy ceiling = 1
block/SM**, jointly capped by 58KB smem (>50KB needed for 2 blocks) AND 221 regs (>128 needed). Every
route to 2 blocks/SM requires a smaller output tile, and a smaller tile costs more in Q4_K dequant
re-amortization than the extra occupancy hides (measured, not inferred). nwarps/mmq_y are locked
together by the mma fragment shape. cuBLAS-F16 already ruled out (lap-5, +4.5%). The only remaining
matmul lever would be a faster Q4_K in-kernel dequant (math/ISA-level kernel rewrite) or a lower-bit
weight quant (quality cost, off the table). **No tuning path forward on the matmul.**

## The real wall (matmul)
DiT mul_mat_q is **latency-bound at a hard 16.67% occupancy ceiling** = 1 block/SM, co-limited by
221 registers AND 58 KB smem. For 2 blocks/SM on sm_86 (65536 regs/SM, ~100KB smem) you need BOTH
≤128 regs/thread AND ≤50KB smem. cap64 gets smem under (48KB) but registers are structurally ~220-248
regardless of tile → **registers are the wall**. Next test: force 2 blocks via `__launch_bounds__`
(spills regs to local mem) — the longcat DiT lap-29 trick — to see if latency-hiding beats the spill.
