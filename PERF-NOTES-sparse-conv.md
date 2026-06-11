# Perf notes — submanifold sparse conv (for the aggressive perf run)

Living doc. Goal: **beat flex_gemm on speed AND stay light on VRAM** (full-pipeline
budget ~7.5 GB). Get it working first, then chase the number — once tf32 tensor
cores are in, `maxrel` moves from ~1e-7 (fp32) to ~1e-3..1e-5; that's ROUNDING,
not breakage (flex_gemm itself runs fp16/tf32). Validate by E2E output, not by
clinging to 1e-7.

## Measured rungs (RTX 3060, real Pixal3D layers, 30-iter warm; `tools/sparse_spike/`)
| layer (K=3³) | Rung-1 naive fp32 | Rung-1.5 cuBLAS tf32 | flex_gemm | im2col VRAM |
|---|---|---|---|---|
| Ci1024→Co1024 N1126 | 87.5 ms | 6.7 ms | 2.9 ms | 0.12 GB |
| Ci1024→Co4096 N1126 | 402.7 ms | 20.9 ms | 20.1 ms | 0.12 GB |
| Ci128→Co128 N89377 | 79.5 ms | 14.2 ms | 2.6 ms | 1.24 GB |
| Ci128→Co512 N89377 | 313.3 ms | 32.1 ms | 138.6 ms* | 1.24 GB |
| Ci256→Co1024 N21063 | 340.6 ms | 26.1 ms | 10.9 ms | 0.58 GB |
| Ci256→Co256 N21063 | 84.6 ms | 9.3 ms | 2.6 ms | 0.58 GB |
| Ci512→Co2048 N4872 | 333.7 ms | 22.8 ms | 11.2 ms | 0.27 GB |
| Ci512→Co512 N4872 | 81.6 ms | 7.2 ms | 2.9 ms | 0.27 GB |
| Ci64→Co64 N382533 | 62.9 ms | 26.6 ms | 165.9 ms* | **2.64 GB** |

Correctness: Rung-1 maxrel ~2-6e-7 (fp32), Rung-1.5 ~1-5e-5 (tf32). Both PASS.

## ⚠️ Baseline caveats (READ before comparing)
- `*` = flex_gemm baseline is **autotune-COLD** for single-call shapes (its first
  call includes Triton JIT+autotune). `flexgemm_timing.json` min_ms == the one cold
  call when count==1. **Discard the favorable comparisons on Ci128Co512 N89377 and
  Ci64Co64 N382533** — they flatter us. Trust only multi-call (warm) shapes.
- Warm-baseline truth: cuBLAS tf32 is **~2-2.5× SLOWER than flex_gemm** on most
  layers, MATCHES on the big GEMM (Ci1024→4096, 0.96×). So tensor cores close the
  *compute* gap; flex_gemm's remaining edge is **avoiding the im2col round-trip**.
- For a clean baseline in the perf run: re-bench flex_gemm with a warmup loop per
  shape (autotune cache is already warm at `golden_model/flexgemm_autotune_cache_3060.json`)
  so every number is warm. Current min_ms is "good enough to steer", not publication.

## What this proves
1. **Tensor cores are the lever** — Rung-1.5 is 10-50× faster than naive Rung-1.
2. **Explicit im2col is a dead end for prod** — up to **2.64 GB for ONE layer**
   (N·V·Cin·4). With weights + other tensors resident under 7.5 GB, that's
   out. Also bandwidth-bound: writing+reading im2col dominates at small Cout
   (why cuBLAS loses on Ci*→Co128/256/512).
3. **Rung-2 = implicit + tensor cores** (fuse the neighbour-gather INTO the MMA,
   exactly like flex_gemm). Only extra buffer = neighbor_map (N·V·4 = 41 MB for
   the big layer, **64× less** than im2col). This is the path to fast AND light.

## Rung-2 plan + guesses/levers for the perf session
Target: match-or-beat flex_gemm warm, im2col-free, ≤ a few hundred MB scratch.
1. **Implicit-GEMM tensor-core kernel** — tile B1(N)×B2(Co), K-loop over (v, Ci-block).
   Inside the loop: gather `feats[neighbor_map[n,v]]` for the B1 rows into shared
   memory (mask sentinel→0), then `mma.sync` tf32 against the weight block. This is
   a direct CUDA transcription of `flex_gemm`'s
   `sparse_submanifold_conv_fwd_implicit_gemm` (already read; see PORT-SPEC).
   - **Reuse the repo's MMA helpers** — `ggml/src/ggml-cuda/mma.cuh` + the mmq/DiT
     tensor-core matmul already in longcat-avatar.cpp. Don't hand-roll wmma.
   - Start from tile configs the 3060 autotuner picked:
     `golden_model/flexgemm_autotune_cache_3060.json` (and `config.py`:
     B1∈{32..128} B2∈{32..256} BK∈{32,64}, warps 2-8). Good seeds, skip blind sweeps.
2. **Masked K-loop** (flex_gemm `masked_implicit_gemm`) — skip kernel taps v that are
   empty for a voxel block. Real data is ~40% sentinel, but per-BLOCK a tap is often
   fully empty → real savings. Needs the gray-code `sorted_idx` reorder
   (`neighbor_map_post_process_1/2`) so a block shares valid taps. Add AFTER the
   plain implicit kernel works.
3. **SplitK** — 3060 has **28 SMs**. Small-N/Co layers (e.g. Ci1024Co1024 N1126 =
   tiny grid) under-occupy → split the K reduction across SMs (flex_gemm does this
   via `get_num_sm`). Likely the win on the small-N channel-heavy layers where we're
   2× behind.
4. **Rulebook caching** — currently built on HOST per call. In prod, build ONCE per
   resolution (GPU hashmap kernel) and reuse across all convs at that resolution
   (the `indice_key`/`spatial_cache` pattern). The SS+shape+texture VAEs run MANY
   convs per resolution → big amortization. Until then, host rulebook is fine for
   correctness/perf-of-the-conv-itself.
5. **fp16 storage** — prod feats are fp16; we validate f32. Storing feats/weight
   fp16 halves bandwidth + VRAM and matches flex_gemm. Do this once the f32 implicit
   kernel is correct (tolerance becomes ~1e-3, expected).
6. **GPU rulebook (hashmap)** — port `hashmap_build_submanifold_conv_neighbour_map`
   to CUDA (open-addressing, size 2.0·N). Net-new but simple. Lower priority than the
   GEMM (rulebook is amortized; GEMM is per-call).

### Suspected ceiling / things to know
- The big GEMM (Ci1024→4096) already matches flex_gemm via cuBLAS → an implicit
  tensor-core kernel should match-or-beat it AND drop the im2col. The HARD layers are
  the **small-Cout, large-N** ones (Ci128→128 N89377): bandwidth-bound, low arithmetic
  intensity → masked + good gather coalescing + fp16 matter most there.
- Watch occupancy on the channel-heavy small-N layers (Ci1024Co4096 N1126): few
  output tiles → splitk or smaller B1×B2 to fill 28 SMs.
- The `Ci64→Co64` layer (N=382k) is the cheapest per-element but biggest N — gather
  coalescing dominates; neighbor_map layout (row-major [N,V]) gather pattern matters.

## Harness (all in `tools/sparse_spike/`, toolchain nvcc, sm_86)
- `sparse_conv.cpp` — CPU f64 oracle (bit-exact, synthetic + real). `npy.hpp` I/O.
- `sparse_subm_conv.cu` + `test_subm.cpp` — Rung-1 implicit naive + CPU mirror.
- `bench_cublas.cu` — Rung-1.5 explicit cuBLAS tf32 + im2col VRAM report.
- `run_bench.sh` — one-command CUDA correctness+timing vs baseline.
- Goldens: `golden_model/` (real, gitignored 2.1 GB), `golden_ref/` (synthetic).
- Baseline: `flexgemm_timing.json` (warm caveat above) + autotune cache.
