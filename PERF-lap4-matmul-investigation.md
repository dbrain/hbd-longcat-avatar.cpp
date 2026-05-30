# lap-4: matmul "floor" re-examined (2026-05-29, code+profile only, NO new GPU run)

**Prior "matmul at the floor / turboquant says exhausted" was wrong.** Turboquant = M=1 decode
(memory-bound). flux DiT = M=1536 (compute regime). Mined existing profiles/lap1.sqlite + read code.

## what the 66% actually is (phase-separated from lap1.sqlite by start-time)
- "66% matmul" was whole-run (cond+DiT+vae). **~26% of it is the Qwen ENCODER** (grid-384
  bucket, mostly in 0-2000ms cond phase), not the DiT.
- **DiT-phase-only** (start>2100ms, GPU 99% busy — no idle gaps, not launch-bound):
  - mul_mat_q **69.2%**, flash_attn 7.9%, layout copies (concat_T+cpy_perm+cpy_scalar) **6.7%**,
    modulation bcast (k_bin_bcast+mul_add_bcast) 5.5%, norms+rope 6.0%, quantize_mmq_q8_1 2.1%,
    silu/gelu 1.9%.

## the one matmul that matters
- **single-block `linear1`**: N = 3*4096 + mlp_hidden*2 = 12288 + 24576 = **36864**, K=4096, M=1536
  (fused qkv + SwiGLU gate+up; flux2 mlp_ratio=3, silu, 24 single blocks). architecture already
  maximally fused (1 in-matmul + 1 out-matmul/block; no redundant quantize).
- profile: grid=3456, x192, **2258 ms = 48% of ALL GPU time**, avg 11.76 ms/launch.
- FLOPs = 2*1536*4096*36864 = 464 GFLOP → 11.76ms = **39.4 TFLOP/s**.
- RTX 3060 int8 tensor peak ~102 TOPS → **~38% utilization**. NOT a hardware wall — lost to Q4_K
  in-kernel dequant + scheduling.

## why the kernel is even MMQ
- `ggml_cuda_should_use_mmq` (mmq.cu:307): `if (turing_mma_available(cc)) return true;` —
  **unconditional on sm_86, M never consulted.** The dequant→cuBLAS-F16 path (FORCE_CUBLAS) is
  never compared. The M-dependent `ne11 < MMQ_DP4A_MAX_BATCH_SIZE` (line 320) is dp4a-only (non-mma hw).

## experiments to run when GPU is free (priority)
1. **FREE, no rebuild — env A/B through resident server, seed 42, fixed cfg:**
   - `GGML_CUDA_FORCE_CUBLAS=1` (+ `GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F=1`) → forces dequant+cuBLAS hgemm.
   - cuBLAS ~85% of 51 TFLOP/s(F16) = ~43 vs MMQ 39 → maybe wins, minus per-call Q4_K→F16 dequant
     (302MB write+read per linear1 ≈ 1.7ms). likely wash; MEASURE, don't assume.
2. **CODE — CFG batching N=2 (M 1536→3072).** today cond+uncond = 2 sequential `run_condition`
   → `compute()` calls (stable-diffusion.cpp ~2214/2235), each a full 5.6GB weight read. Batching:
   same img latent x, stack the 2 text contexts on batch dim (pad to equal n_txt). reads weights
   once for both AND larger M better amortizes Q4_K dequant per weight-tile → should lift the 38%
   util + halve launches. diffusers does this by default; sd.cpp left it sequential (VRAM). we now
   have headroom (6.4/12 GB). risk: activation peak ~2×, padding logic, verify PSNR.
3. lower-value: layout copies 6.7% (concat in single blocks), modulation-fusion 5.5% — risky, small.

## quant angle (quality tradeoff, ask first)
- the 38% util is Q4_K dequant cost. Q8_0 DiT = trivial dequant → higher MMQ util but 2× weight
  (11GB UNet, breaks budget). Q5_0/Q4_0 simpler dequant than Q4_K, same-ish size — could test util.
  DiT quant is separate from the encoder Q4 floor.
