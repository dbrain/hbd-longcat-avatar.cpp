# lap-9: text-padding trim (BIG lever, but quality-gated) + ALL-ANGLES CONCLUSION — 2026-05-30

## The one large lever found: trim the 512-token text padding
The DiT runs joint attention over **L=1536 = ~1024 image + 512 text tokens**, but flux2-klein pads
text to `min_length=512` (conditioner.hpp) while a real prompt is ~15 tokens. ~32% of every step's
sequence is text padding. Added env `FLUX2_TEXT_MINLEN` (default 512) to trim it. Sweep vs golden:

| minlen | L_k | dit s/step | wall | vs base | PSNR vs golden |
|--|--|--|--|--|--|
| 512 (default) | 1536 | 1.704 | 15.83s | — | 999 (identical) |
| 384 | 1408 | 1.619 | 15.21s | −4% | **26.6 dB** |
| 256 | 1280 | 1.430 | 13.58s | −14% | **25.0 dB** |
| 128 | 1152 | **1.315** | **12.57s** | **−21%** | **22.1 dB** |

**Verdict: QUALITY TRADEOFF, off the table under the no-quality-loss constraint.** PSNR never reaches
the ~31 dB noise floor — even a mild 25%-trim (384) sits at 26.6 dB (a visibly different image). The
padded positions materially influence the output: FLUX.2 was trained at the full 512 length with **no
text-padding mask in the joint blocks**, so the padding embeddings are baked into the learned behavior.
Trimming (or masking) both deviate — there is NO lossless version of this lever. Left in as an opt-in
knob (default off) in case the product wants to accept the tradeoff: ~−21% wall for ~22 dB. **User's call.**

## ALL-ANGLES CONCLUSION — no lossless perf path remains on the RTX 3060
DiT GPU time (12.83s, 81% of a 15.8s gen) accounted for, every kernel >2% with a reducible verdict:

| kernel | % DiT | reducible? | evidence |
|--|--|--|--|
| mul_mat_q (matmul) | 69 | **NO** | latency-bound, 16.67% occ = 1 block/SM, hard-capped by 58KB smem (>50KB) AND 221 regs (>128). mmq_x↓ slower (occ unchanged, ncu); force-2-blocks needs smem>100KB cap; mmq_y locked by mma frag; cuBLAS +4.5% (lap5). |
| flash_attn_ext_f16 | 8 | **NO** | occ-ceiling 25% = 3 blocks/SM, reg(168)+smem(27.8KB) co-limited, latency-bound (80% no-eligible). launch_bounds↑ would spill. |
| k_bin_bcast (broadcast) | 3.5 | partial | modulate +shift folded into gate_add (lap-8, −0.7%, bit-exact). Rest = gateless residual/time-embed adds, no fusable partner. |
| concat_T_cont + cpy_perm + cpy_scalar (layout) | 6.7 | architectural | joint-attn concat, attn head permute, rope-prep contiguity. Lossless removal needs FA to read strided Q/K/V (deep, uncertain, ≤6.7% ceiling). |
| rms_norm + norm + rope_pe | 6 | **NO** | necessary math (LayerNorm/RMSNorm + RoPE). |
| quantize_mmq_q8_1 | 2.1 | **NO** | MMQ activation-quant companion; required by the int8 matmul path that already wins. |
| mul_add_bcast (gate_add) | 2.1 | done | already fused (longcat lap-28.3). |
| unary (silu/gelu) | 1.9 | **NO** | activation, necessary. |

**77% of the DiT (matmul + FA) is latency-bound at fixed register+shared-memory occupancy walls** that
hold at each kernel's optimal tile — proven by ncu, not inferred. Shrinking tiles to raise occupancy is
measured-slower (matmul) or spills (FA). The only theoretical matmul lever is a faster Q4_K in-kernel
dequant (ISA-level rewrite, large effort, uncertain) or lower-bit weight quant (quality, off-table).
The remaining ~23% is necessary kernels (~12%) + architectural layout-copies (~6.7%) + a thin fusable
broadcast slice (taken). **There is no lossless tuning path that moves the needle materially.**

## Shipped this session (uncommitted, both bit-exact md5 6c0a783425ea, default build):
1. Encoder early-stop (27/36 Qwen layers) — lossless, wall-neutral (encoder is overhead-bound at M=512).
2. Modulate AdaLN shift-fusion — lossless, dit −0.7% (1.704→1.692 s/step).
Net default: 15.83s → ~15.78s, peak 6553 MiB, bit-identical. Experiment scaffolding (env/defines,
all default-off): FLUX2_TEXT_MINLEN, GGML_MMQ_X_CAP, MMQ_EXPERIMENT_MIN_BLOCKS, MMQ_Y_OVERRIDE.
Tools: tools/lap_bench.py (md5+PSNR+timings), tools/serve_cap.sh (cap+minlen sweeps).
