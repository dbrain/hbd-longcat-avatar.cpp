# lap-5: cuBLAS A/B (measured) + the 38%-util root cause (2026-05-30)

**MMQ is the correct kernel — cuBLAS-F16 measured SLOWER. The ~38% TC util is Q5_K dequant cost
(the hot matmuls are Q5_K, not Q4_K).** GPU was free; ran real A/B through resident server, seed 42.

## cuBLAS A/B (FORCE_CUBLAS is COMPILE-time, not env — rebuilt build-cublas/ with -DGGML_CUDA_FORCE_CUBLAS=ON)
| variant | DiT s/step | VRAM peak | md5 | vs MMQ |
|--|--|--|--|--|
| MMQ (shipped) | **1.695** | **6455** | 6c0a783425ea | — |
| cuBLAS 32F accum | 1.77 | 6767 | 1e488d6fd469 | **+4.5% slower, +312 MiB** |
| cuBLAS 16F accum | 1.772 | 6767 | 1e488d6fd469 | +4.5% slower |
- env `GGML_CUDA_FORCE_CUBLAS=1` did NOTHING (bit-identical to MMQ) → it's a `#ifdef`, not getenv.
- cuBLAS engaged only with the build flag (md5 changed → confirms dequant→F16→hgemm path live).
- COMPUTE_16F made no difference here (same md5/time).
- **verdict: int8 MMQ's 2× peak beats cuBLAS-F16's higher %-efficiency once per-call Q5_K→F16 dequant
  (+312 MiB scratch) is paid. The unconditional `should_use_mmq → true` on sm_86 is CORRECT for M=1536.**
  build-cublas/ removed.

## why MMQ only hits ~38% of int8 peak — it's just Q4_K MMQ (CORRECTED — not Q5_K)
- DiT GGUF (klein-base-9b-Q4_K_M) is MIXED: 6375M Q4_K + 2349M Q5_K + 354M type-30 + F32.
- the Q5_K (~26%) is the standard llama.cpp **Q4_K_M `use_more_bits`** recipe, NOT a special case:
  ALL attention qkv (img+txt, every block) + first/last ~2 blocks → Q5_K; everything else Q4_K.
  per-block single_blocks.N.linear1: blk0,1=Q5_K, blk2-21=Q4_K, blk22,23=Q5_K.
- so the 48%-of-GPU matmul (single linear1, 24 of them) is **20/24 Q4_K**. The ~38% util is
  **Q4_K MMQ's achieved efficiency** at M=1536 on the 3060 — NOT a Q5_K dequant penalty (earlier
  draft mis-attributed this from sampling only blocks 0-1).
- => the matmul IS at the practical floor for off-the-shelf kernels: cuBLAS loses (above), and
  beating 38% means out-engineering ggml's tuned int8 k-quant MMQ (huge, not worth it).

## DiT quant-format lever — NOT worth it (downgraded)
- the hot matmuls are already Q4_K. only 4/24 single linear1 + the attn-qkv are Q5_K, and those
  are the layers the recipe deliberately protects. requant Q5_K→Q4_K would give ~2-3% DiT at best
  and cost quality on exactly the sensitive layers. skip.
- (going BELOW Q4_K e.g. Q4_0/Q3 = same dequant cost class, real quality loss → no.)

## ruled out
- **CFG batching (M 1536→3072): NOT worth it.** tile math: gridX=3456 = 12 M-tiles × 288 N-tiles,
  ~123 waves over 28 SMs at M=1536 — no wave-quantization, full within-tile dequant amortization
  already. GPU 99% busy (62ms idle/6250) → launch savings negligible. weights already resident
  (offloaded once, not re-H2D per forward). glue ops memory-bound → 2× data = 2× time, no saving.
  → expected ~neutral; big code change (cond/uncond pad). skipped.
- cuBLAS (above).
