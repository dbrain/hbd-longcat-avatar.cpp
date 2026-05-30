# lap-8: modulate AdaLN shift-fusion (lossless) — 2026-05-30

## DiT-phase kernel breakdown (nsys, 8-step gen, DiT window 1.80→15.11s, 12.83s GPU)
| kernel | calls | ms | % DiT |
|--|--|--|--|
| mul_mat_q | 1792 | 8891 | 69.3 |  ← DEAD (lap-7 verdict)
| flash_attn_ext_f16 | 512 | 1015 | 7.9 |
| k_bin_bcast | 1552 | 451 | 3.5 |  ← unfused broadcasts (residuals + modulate shift)
| rms_norm_f32 | 1280 | 362 | 2.8 |  ← necessary
| concat_T_cont | 656 | 350 | 2.7 |  ← layout (joint-attn concat)
| cpy_perm_coalesced | 1185 | 339 | 2.6 |  ← layout (head permute / rope prep)
| rope_pe_f32 | 1024 | 306 | 2.4 |  ← necessary
| quantize_mmq_q8_1 | 1792 | 266 | 2.1 |  ← MMQ companion, necessary
| mul_add_bcast_dim1 | 1808 | 263 | 2.1 |  ← already-fused gate_add (longcat lap-28.3)
| unary_op (silu/gelu) | 720 | 239 | 1.9 |  ← necessary
| cpy_scalar_contiguous | 1024 | 149 | 1.2 |  ← layout
Split: matmul 69% / FA 8% / layout-copy 6.7% / broadcast 5.6% / norm+rope 6% / quantize 2.1%.

## The fusion
`modulate()` (flux.hpp:248) = `x = x + x*scale; x = x + shift`. The `x + x*scale` already hits the
LongCat gate_add fusion (gate=scale, contiguous [d0,1,d2,Nb] view). The trailing `+ shift` (same
broadcast layout) was a separate `k_bin_bcast`. Extended `ggml_cuda_op_mul_add_bcast` + its detection
(ggml-cuda.cu) to fold it: `dst = x + x*scale + shift` in one kernel when add_n is consumed only by
the shift-ADD. Bit-exact (extra `__fadd_rn` = same single rounding as the bcast-ADD it replaces).
Gated residuals (no trailing shift) fall back to the plain gate_add (shift=nullptr).

## Result
**dit 1.704 → 1.692 s/step (−0.7%), wall 15.83s flat, md5 6c0a783425ea (BIT-EXACT).** Kept (lossless).
Small because modulate-shifts are only ~1/5 of k_bin_bcast (the rest are gateless residual adds +
time-embed adds, which have no fusable partner). Eliminates one full [C,L,N] read+write + intermediate
buffer per modulate call.

## Why the glue ceiling is low
Every individual glue kernel is 1-3% of DiT and most are *necessary* (rms_norm, rope, quantize-for-MMQ,
silu). The fusable broadcasts are a fraction of 3.5%. Layout-copies (6.7%) are architectural (joint-attn
concat, attention head permute, rope-prep contiguity) — reducing them losslessly needs FA to accept
strided Q/K/V (deep, uncertain). Realistic total remaining lossless headroom ≈ a few % of wall, spread
across fiddly fusions — the 69% matmul mass is immovable (lap-7).
