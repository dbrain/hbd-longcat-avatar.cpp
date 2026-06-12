# PERF-NOTES — pixal3d (Phase C kickoff)

Correctness-first port is DONE + feature-complete (image → textured GLB, pure C++/ggml from GGUF,
matching Python). This doc banks the perf intel for the Phase C run. **Precision may loosen here
(tf32/fp16/quant); judge by E2E mesh IoU vs the fp32 result, NOT a tight elementwise tol.**

## Measured timing (RTX 3060, fp32, NVIDIA_TF32_OVERRIDE=0)
Textured CLI run (`pixal3d --tex`, 1024 path): **509.7s total** (untextured geometry: ~340s).

| stage | time | where it runs | notes |
|---|---|---|---|
| DINOv3@512 | 2.8s | ggml CUDA | shared by stage1 + stage2 |
| SS DiT (12 step ×2 fwd) | 62.9s | ggml CUDA | dense 16³=4096-token DiT, 30 blocks, fp32 |
| SS VAE decode | ~3s | ggml CUDA | 3D conv decode → coords |
| NAF@512 | 3.7s | ggml CUDA | |
| M2 DiT (12×2) | 14.8s | ggml CUDA | sparse, N1≈1120 tokens |
| **M3a upsample** | **~30s** | **host + spike-conv GPU** | coord-growth to ~363k; CPU-bound |
| DINOv3@1024 | 3.5s | ggml CUDA | 64×64 patches |
| NAF@1024 | 4.3s | ggml CUDA | |
| M3b DiT (12×2) | 73.2s | ggml CUDA | sparse, M≈4633 tokens (slowest DiT) |
| **M4 shape decode + mesh** | **112.9s** | **host + spike-conv GPU** | grows to ~1.47M voxels; CPU-bound |
| tex DiT (12×1, CFG-off) | 41.8s | ggml CUDA | in_ch 64; CFG-off so 1 fwd/step |
| **M6 tex decode** | **111.7s** | **host + spike-conv GPU** | ~1.47M voxels; CPU-bound |

## Bottleneck #1 — the host sparse-VAE decodes (M3a + M4 + tex = ~255s, **50% of wall**)
The submanifold sparse conv runs on the GPU (the spike kernel, `sparse_subm_conv.cu`, dispatched
per layer), but EVERYTHING ELSE in `sparse_vae_pipeline.hpp` runs on **CPU (OpenMP)**:
- dense per-voxel ops: `svae::linear` / `layernorm` / `silu` over up to **1.47M voxels × {64..1024}ch**
- `build_nmap` — an `unordered_map<int64,int>` hashmap rebuilt **per level** over up to 1.47M coords
- `c2s_grow` coord growth + `gather_children` / `repeat_interleave` (host index arithmetic)
So the GPU lights up for the conv but idles between launches → wall-clock is CPU-bound. This is the
correctness-first design (data-dependent coord growth → no static GPU graph). **Levers (highest value):**
- **Move the dense ops to GPU**: `linear`(it's just a [N,Cin]×[Cin,Cout] GEMM → cuBLAS/ggml), `layernorm`,
  `silu` — all trivially parallel per-voxel. Keep coords/nmap on host, push feats to GPU, do GEMM+norm+act
  on GPU, pull back only what the next coord-growth needs. Biggest single win.
- **Faster nmap**: replace the per-level `unordered_map` with a sorted-coord + binary-search or a
  radix/Morton hash; or build it ON the GPU. It's rebuilt 8× (4 levels × shape+tex) over ~1.5M coords.
- **Fuse the spike conv with the surrounding dense ops** (the conv already gathers; fold the bias/act).
- The conv itself: the spike kernel is fp32 no-tensor-core; tf32 or fp16 accum + MMQ would speed it
  (precision loosens — judge by mesh IoU).
- Validate cheaply: `m4_mesh` / `m6_tex_decode_test` run the decode on golden input + check bit/IoU —
  optimize against those (no full 510s E2E needed per iteration).

## Bottleneck #2 — the 4 DiTs (SS 63 + M2 15 + M3b 73 + tex 42 = **193s, 38%**)
ggml CUDA graphs, already on GPU, fp32 (tf32 off). The weights are the 1.3B slat DiTs (5.5GB fp32 each).
- **Quantize weights via GGUF Q-types** (Q8_0 near-lossless, Q4_K aggressive) — leverages the A2 GGUF
  infra. ggml `mul_mat(W_quant, x_f32)` uses MMQ kernels → faster matmul + ~4× less VRAM. `pack_gguf`
  needs a `--type` arg (quantize the 2D matmul weights; keep 1D norms/biases + conv f32). The harness
  `gguf_fetch` already creates the tensor with its stored type, so `lin()` just works. Judge by E2E IoU.
- **tf32 / fp16 accum** for the DiT matmuls (drop the NVIDIA_TF32_OVERRIDE=0) — free ~2× on the matmuls;
  was disabled for correctness-validation, legitimate to re-enable in the perf phase (judge by IoU).
- M3b is the slowest DiT (73s, M≈4633 sparse tokens, 30 blocks ×12×2). Sparse attention is full (varlen)
  over M tokens — the attention is O(M²); fine at M≈4.6k but watch if M grows.

## Bottleneck #3 — VRAM / co-residency
Currently low_vram sequential (each stage's harness opens→computes→frees; peak = max stage ≈ DINOv3@1024
+ NAF ≈ 6.3GB alloc / 7.6GB reserved). Quantizing the DiTs (Q4_K) makes 7.5GB co-residency feasible
(keep models resident → skip reload). Not needed for correctness; a throughput lever for a server.

## Cheap-to-validate first lever (recommended start)
Move `svae::linear`/`layernorm`/`silu` to the GPU in the decode path, validated by `m4_mesh` (golden
input, checks bit-exact verts/faces) + `m6_tex_decode_test` (PBR maxabs vs oracle). That's ~50% of wall
and validates WITHOUT a full E2E run. Then quantize the DiTs (one E2E run to confirm mesh IoU ≈ 1.0).
