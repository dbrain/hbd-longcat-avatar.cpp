# NAVA DiT modulation 2-timestep collapse — RESULTS (2026-06-05)

Branch `nava-port`. Implements `HANDOFF-nava-VRAM-MODULATION.md`. Code: `src/nava.hpp`.

## What shipped
The AdaLN modulation table is computed on a COMPACT 2-timestep input `e=[dim,6,2]`
(col0 = clean anchor t=0, col1 = `tval`) instead of the full per-token `[dim,6,L_total]`.
Each block expands the 2 columns back per-token with PURE COPIES (`ggml_repeat`+`ggml_concat`),
so the downstream op graph is unchanged.
- Helpers `expand_mod_chunk` (2-segment, video/audio double + heads) and
  `expand_mod_chunk_single` (4-segment over concat[video++audio], independent anchors).
- `build_graph` scans the host timestep for `n_clean_i` / `n_anchor_a` / `tval`, builds
  `t2={0,tval}`, runs `time_embed(t2)`, passes anchor counts to the blocks. Expansion gated
  on `e->ne[2]==2` (==1 → text-mode broadcast, unchanged; >2 → full per-token, debug only).

## Bit-exactness — IMPORTANT nuance (it's a quality WIN, not a bug)
The compact path is **NOT bit-identical** to the old prod path, but the divergence is the
compact path being **more accurate**, not wrong:
- Proven by elimination. A `NAVA_MOD_FROM_FULL=1` mode runs the matmul at FULL width then
  slices out the 2 unique columns and feeds the SAME expansion → **bit-exact to baseline
  (maxabs 0)**. So the expansion logic is perfect; the ONLY divergence is the matmul width.
- `time_projection.1.weight` is Q5_0. ggml-cuda picks the matmul kernel by N: width-2 →
  **MMVQ** (activations stay F32) vs width-L → **MMQ** (activations quantized to int8). So
  the compact table is computed with F32 activations = closer to the PyTorch reference.
- Owner heard the OLD 6-step audio as the broken one ("bewore" vs "before"); compact moves
  toward correct. Divergence shrinks with steps: audio latent cos compact-vs-fromfull
  0.928 @6 steps → 0.981 @25 steps.
- `NAVA_MOD_FROM_FULL=1` = provably-bit-exact-to-old-prod escape hatch (materializes the full
  table transiently, +146 MiB peak, ~same wall). Default is compact.

## VRAM — the collapse is what makes native res FIT
| config @ 1280×704 q5_k 13f | DiT compute buffer | peak VRAM | fits 7.5 GB? |
|---|---|---|---|
| OLD baseline (no collapse) | ~4.4 GB | **9453 MiB** | ❌ |
| from-full (bit-exact) | 1787 MB | 7367 MiB | ✅ |
| **compact (default)** | 1641 MB | **7233 MiB** | ✅ |

@896×448/13f the DiT buffer drops 1771 → 627 MB. Steps don't affect VRAM.

## Native-res quant ladder (compact, 1280×704 × 25 steps, seed 42, peter)
| quant | peak VRAM | fits 7.5 GB? | wall | s/step |
|---|---|---|---|---|
| q8_0 | 9241 MiB | ❌ | 716s | 24.3 |
| q6_k | 7957 MiB | ❌ (just over) | 767s | 26.0 |
| q5_k | 7233 MiB | ✅ | 734s | 25.0 |
| q4_k | 6551 MiB | ✅ | 726s | 24.7 |
Speed flat across quants (quant = VRAM only). Only q5_k / q4_k fit the coexistence budget.

Latent cos vs q8 (MISLEADING — owner judges by ear; non-monotonic): audio q6 .989 / q5 .890 /
q4 .618; video q6 .996 / q5 .936 / **q4 .962 (> q5!)**. The q5-vs-q4 choice is an EAR/EYE call —
clips on :8097: `q8_native q6k_native q5k_native_compact q4k_native q5k_native_fromfull`.

## OPEN / NEXT
- Owner ear verdict: compact vs from-full (q5), and q5 vs q4 at native spec. Lock the prod quant.
- Step-cache t15 (`NAVA_CACHE_THRESH=0.15`) clip at native res for the speed-vs-audio A/B.
- **umT5 context is PER-SEGMENT** for the real talking-avatar product (speech changes each
  continuation segment) — 6.0 GB, can't co-reside with the DiT, lifecycle must be
  encode→free→DiT→free→VAE; per-segment cost UNMEASURED. Wire inline-encode into `nava render`
  (already links `T5Embedder`) + time the multi-segment chain. See project_nava_continuation_m1.
- Hardest remaining bit-exact SPEED lever: video-VAE direct-conv3d (~−10-15s/clip).
