# NAVA cpp — PROFILING BIBLE (kernel-level, every phase) — 2026-06-05

Branch `nava-port`. RTX 3060, serial GPU. Locked config: q6_K + FA + tile16, 896x448, 10 steps,
13 frames (49 pixel frames), seed 42. **Baseline wall 175.94s** (reproduced exactly this session).

Tooling: nsys 2024.1.1 (bundled in toolchain nsight-compute host dir) for kernel-time
breakdown; ncu for per-kernel counters. Trace = 1 DiT step + full video VAE + full audio VAE
(`bench/results/nava_full_1step.nsys-rep`, 236 MB).

## Wall-time phase split (MEASURED, baseline)
| phase | wall | % | VRAM | GPU-busy? |
|---|---|---|---|---|
| load DiT + I2V encode | ~2.4s | 1% | ramp | — |
| **DiT sampling (10 steps)** | **95.4s** | **54%** | 7579 MiB (flat peak) | ~100% GPU-bound |
| **video VAE decode** | **46.1s** | **26%** | 4719 MiB | mixed (10s overhead) |
| **audio VAE decode** | **29.7s** | **17%** | 695 MiB | 1 naive kernel |
| mux | ~1.5s | 1% | — | — |

## KERNEL-LEVEL ATTRIBUTION (nsys, per phase)

### DiT (1 step = 3 forwards: cond+uncond+align-mmask; ~9.6s GPU-busy)
| kernel | s/step | inst | what |
|---|---|---|---|
| `mul_mat_q` (ggml_type 14 = Q6_K) | **4.17** | 1207 | FFN + attn projection weight matmuls |
| `flash_attn_ext_f16<128,128,64>` | **1.94** | 240 | joint self-attn (~5k tok, head_dim 128) |
| `k_bin_bcast op_add` | 1.64 | 4128 | bias adds / residuals |
| cpy_perm_coalesced | 0.37 | 1448 | layout |
| quantize_mmq_q8_1 | 0.27 | 1207 | activation→q8_1 for MMQ |
| rms_norm/norm/silu/pad/concat/cpy | ~1.2 | — | glue |
- **DiT is genuinely GPU-compute-bound.** mul_mat_q + flash_attn = 6.1s = 64% of the step.
  Over 10 steps: mul_mat_q ≈ 42s, flash_attn ≈ 19s, k_bin_bcast ≈ 16s.
- The 3rd forward (align-mmask) is 1/3 of this — `NAVA_NO_ALIGN_CFG=1` removes it (−33% DiT,
  quality-load-bearing on hard prompts).

### video VAE decode (Wan2.2 48ch causal conv3d, tile16, 10 tiles, 49 frames; ~46s)
| kernel | s | inst | what |
|---|---|---|---|
| **`concat_T_cont<float,2>`** | **9.90** | **6,105,281** | ⚠️ 6.1 MILLION tiny concats — causal-conv temporal cache |
| `im2col_3d_tiled` | 6.45 | 4410 | conv3d lowering |
| `ampere_h16816gemm_256x128` | 5.61 | 1630 | conv3d gemm (tensor core) |
| `ampere_h1688gemm_128x128_stages` | 4.16 | 1280 | conv3d gemm |
| `im2col_kernel` | 2.88 | 650 | conv lowering (non-3d) |
| `ampere_h1688gemm_128x128_tn` | 2.83 | 650 | conv gemm |
| `pad_f32` | 2.11 | 3931 | padding glue |
| `k_bin_bcast` | 1.98 | 11312 | adds |
| cpy_perm_transpose/coalesced | 2.24 | 13094 | layout |
- conv compute (im2col+gemm) ≈ 21.9s is the real work.
- **`concat_T_cont` 9.9s / 20% of the phase is PURE LAUNCH OVERHEAD** — 6.1M kernels @ 1.6µs.
  This is the #1 video-VAE target: eliminate/batch the causal temporal-cache concat.
- pad_f32 (2.1s) + cpy (2.2s) + k_bin_bcast (2.0s) are also glue, choppable.

### audio VAE decode (LTX audio VAE; ~29.7s)
| kernel | s | inst | avg | what |
|---|---|---|---|---|
| **`conv_transpose_1d_kernel`** | **27.13** | **11** | **2.47s** | ⚠️ naive 1D transposed conv = the ENTIRE phase |
| im2col_kernel | 1.52 | 633 | | |
| mul_mat_vec_f | 0.72 | 200 | | |
- **The audio VAE IS one naive kernel.** 11 launches, avg 2.47s, max 5.64s. 27s = 15% of the
  whole render wall in a single un-optimized ggml conv_transpose_1d.
- `NAVA_AUDIO_DISABLE_BWE=1` already cuts audio VAE 29.7→~9s (−21s wall) → BWE upsampling is
  where most of these conv_transpose_1d live. But BWE is a quality feature — the real win is to
  make conv_transpose_1d FAST (im2col+gemm or proper kernel) so we keep BWE at full speed.

## OPTIMIZATION PRIORITY (heads to chop, by measured size)
1. **audio VAE `conv_transpose_1d` — 27s.** Biggest single kernel in the pipeline. Replace naive
   impl with im2col+gemm / better kernel. NO quality loss (decode is exact). Target: −15..20s.
2. **video VAE `concat_T_cont` — 10s of 6.1M launches.** Batch/eliminate causal-conv temporal
   concat. NO quality risk. Target: −8..10s.
3. **DiT mul_mat_q (42s/render) + flash_attn (19s).** Genuinely compute-bound — needs ncu to
   confirm roofline (occupancy/DRAM%) before claiming floor. Hardest; quant is quality-gated.
4. video VAE glue (pad/cpy/k_bin_bcast ~6s) — secondary.

NEXT: ncu the DiT mul_mat_q + flash_attn (roofline proof); read the conv_transpose_1d and
causal-conv3d-concat source to find the structural fix.
