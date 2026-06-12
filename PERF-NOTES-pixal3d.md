# PERF-NOTES — pixal3d (Phase C)

Correctness-first port is DONE + feature-complete (image → textured GLB, pure C++/ggml from GGUF,
matching Python). This doc banks the perf intel for the Phase C run. **Precision may loosen here
(tf32/fp16/quant); judge by E2E mesh IoU vs the fp32 result, NOT a tight elementwise tol.**

---
## ✅ LAP 1 (2026-06-12) — GPU-resident sparse-VAE decode. **decodes 254s → 50s (−80%), bit-exact.**

**The kickoff's #1-target hypothesis was WRONG.** Profiled the M4 host decode op-by-op
(`m4_profile.cpp`, golden input, `SVAE_PROF=1`): the cost is the **dense `linear` (MLP GEMMs)
on CPU OpenMP = 82%**, NOT the conv's per-call malloc/H2D/upload (that overhead is **0.86s total**,
negligible). Conv kernel itself = 15.5%. Breakdown (164s profiled; the npy reloads inflate it ~50s
over the real 113s, but the proportions hold):
| op | time | % |  | conv sub-breakdown (40 calls) |
|---|---|---|---|---|
| **linear (CPU GEMM)** | **134.7s** | **82%** | | malloc 0.13s · h2d 0.59s |
| conv (GPU kernel) | 25.5s | 15.5% | | **kernel 17.8s** · d2h 1.86s · free 0.14s |
| silu+layernorm+gather+rest | ~3.4s | 2% | | |

**Fix = GPU-resident decode** (`svp_gpu.hpp` + `svae_cuda.cu`): keep `feats` RESIDENT on the GPU
across all 4 levels — `linear`→cuBLAS sgemm, layernorm/silu/add/gather/repeat→CUDA kernels, conv→spike
launcher (resident ptrs). Weights upload to GPU **once** (cached in `svpg::GpuW`), not per-call. Only
subdiv decisions (`>0`) + the final head/coords pull back to host (coord-growth + mesh extraction stay
host, unchanged). Drop-in: `svpg::{m3a_upsample,m4_decode_mesh,m6_tex_decode}` swapped into
`pixal3d_chain.hpp` (CUDA path), validated in isolation:
| stage | host | **GPU-resident** | correctness vs fp32 oracle |
|---|---|---|---|
| M3a upsample | ~30s | **6.97s** | hr_coords IoU **1.000000** (set-equal, N=382584) |
| M4 decode+mesh | 112.9s | **21.8s** | verts maxabs **5.96e-08**, faces count identical, 12/9.75M idx flips (head boundary noise) |
| M6 tex decode | 111.7s | **21.4s** | coord IoU **1.000000**, PBR maxabs **3.5e-6** |

Numerics matched: layernorm reduces in **double** (host uses double); cuBLAS fp32 under
`NVIDIA_TF32_OVERRIDE=0`. Tests: `m3a_gpu_test` / `m4_gpu_test` / `m6_gpu_test` (`./build.sh <t> cuda`).

**E2E CONFIRMED (textured `pixal3d --tex`, full 1024 path):** **509.7s → 298.8s (−41%)**, mesh
**byte-identical** to the f32 baseline (N1=1120, M=4633, verts 1471142, faces 3049548). Per-stage:
DINOv3 2.6+3.4 · SS DiT 62.9 · NAF 3.7+4.3 · M2 DiT 14.9 · **M3a 5.5** (was ~30) · M3b DiT 73.2 ·
**M4 19.8** (was 112.9) · tex DiT 42.0 · **M6 23.2** (was 111.7). Decodes 255s → **49s**.

**Wall is now DiT-bound: SS 62.9 + M2 14.9 + M3b 73.2 + tex 42.0 = 193s = 65% of the 299s.**
→ LAP 2 = quantize the DiTs (Q8_0 MMQ + tf32). Then the conv kernel (~17.8s, the decode floor:
naive fp32 implicit-GEMM, no tensor cores).

---
## ✅ LAP 2 (2026-06-12) — DiT weight quantization. **F16 = near-lossless, Q8-speed. ~−39% per DiT.**

Quantize the 2D matmul weights of the DiT GGUFs (`pack_gguf.cpp` `--type {f16,q8_0,...}`: name ends
`.weight`, ndim==2, ne0%block==0; keeps 1D norms/biases/`.gamma`/conv f32; ss_flow input in=8 →
auto-F16). ggml reads quant/f16 GGUF natively (`gguf_fetch` + `ggml_mul_mat`). The fp16 tensor-core
path needs `GGML_PREC_DEFAULT` → added env toggle **`PIXAL3D_FAST`** to `m1_ggml.hpp` lin()/attention()
(drops the correctness-first `GGML_PREC_F32` forcing). Accumulation precision via
`GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F` (fp32 accum on the fp16 path).

**M3b DiT A/B (slowest DiT, N=4734, golden cond; cosine vs the fp32 torch oracle):**
| config | wall | cosine vs fp32 | verdict |
|---|---|---|---|
| f32 (baseline) | 85.7s | 0.999996 | — |
| **F16 + FAST + fp32-accum** | **59.6s** | **0.999853** | ★ max quality + fast |
| F16 + FAST (fp16-accum) | 52.4s | 0.999696 | fastest, still near-lossless |
| F16 (no FAST, fp32 cublas) | 76.3s | 0.999937 | no tensor cores |
| Q8_0 (all matmuls) | 51.7s | **0.9677** | too lossy |
| Q8_0 + FAST | 50.3s | 0.9696 | too lossy |

**⟹ F16 is strictly better than Q8 here: SAME speed (~52s), ~100× lower error.** Q8 MMQ (int8) and
F16 (fp16 tensor-core) run at the same rate on the 3060 for these shapes, but F16's 10-bit mantissa
preserves the iterative flow far better.

**Mixed-precision study (owner asked: "is Q8 fine for most of the model but lossy on one part?").
ANSWER = NO — the Q8 loss is DISTRIBUTED + CUMULATIVE, not localized.** Q8-everything + keep-f16 on
the suspected-sensitive components recovered nothing:
| keep at f16 (rest Q8) | cosine vs fp32 |
|---|---|
| out_layer | 0.9697 |
| out_layer + input_layer | 0.9695 |
| out_layer + input_layer + adaLN_modulation | 0.9701 |

(all ≈ the all-Q8 0.9677 — no single layer is the culprit). The error is ~0.4%/weight × 245 weights
across 30 blocks, **integrated over 12 sampler steps** → it compounds. You can't claw it back by
sparing a few layers; you'd have to keep the bulk high-precision, i.e. F16 everywhere (which is free
here). **imatrix is the right lever ONLY to push *below* F16** (Q4_K/Q5_K for the VRAM stretch): it
weights quant error by per-input-channel activation energy, reducing the distributed error globally.
Capture activations from the 12-step forward → per-column mean(act²) → pass to `ggml_quantize_chunk`
(currently nullptr). Not needed for speed/quality (F16 wins); it's the sub-4GB-VRAM play.

**Decision: F16 all 4 DiTs + `PIXAL3D_FAST` + `GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F`.** GGUF set in
`weights_gguf_f16/` (DiTs f16, decoders/dinov3/naf symlinked f32). VRAM: each DiT 5.5GB→2.8GB.
SS DiT is occupancy-sensitive (its z_s→coords via >0) — validate N1==1120 in the E2E before trusting.

**E2E VALIDATED (`PIXAL3D_GGUF_DIR=weights_gguf_f16 PIXAL3D_FAST=1 GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F=1
./pixal3d --tex`):** **266.1s**, mesh near-identical to the f32 baseline — **N1=1120, M=4633 (both EXACT)**,
verts 1471258 / faces 3050326 (vs f32 1471142/3049548 = +0.008%/+0.025%, a few boundary voxels from
the 0.9999-cosine shape_slat). DiT compute −30%: SS 62.9→44.5, M2 14.9→9.8, M3b 73.2→51.6, tex 42.0→29.6.
⚠️ **The 266s INCLUDES a ~24s one-time HDD weight-load penalty** (weights live on `/mnt/hdd/pixal3d` per
owner request, symlinked back; first-mmap from HDD: dino +4.7, shape_dec/M3a +12, tex_dec/M6 +7). On SSD
or for a resident-weight server this E2E ≈ **242s**.

### ▶ COMBINED LAP1+LAP2 SCORECARD
| | original f32 | **now (F16 DiT + GPU decode)** |
|---|---|---|
| textured E2E | 509.7s | **266s** (HDD) / **~242s** (SSD-equiv) = **−48% to −53%** |
| decodes (M3a+M4+M6) | 254.6s | **~50s** |
| DiTs (SS+M2+M3b+tex) | 192.7s | **135s** |
| mesh | — | byte-near-identical (N1/M exact, verts/faces within 0.03%) |

**Remaining levers (priority order):** (1) **spike conv kernel** ~17.8s decode floor (naive fp32
implicit-GEMM, no tensor cores → tf32/fp16/tiled-MMA; validate via `m4_gpu_test` mesh IoU); (2) the SS
DiT is the single biggest stage (44.5s, 4096-token O(N²) attention → flash-attn could help); (3) VRAM
co-residency + imatrix-Q4 for the sub-4GB stretch.

---
## ✅ LAP 3 (2026-06-12) — f16 tensor-core attention. **−12% per DiT, near-lossless.**

After F16 weights, the DiT linears run on fp16 tensor cores but the **self/cross-attention matmuls
stayed fp32** (q/k/v are fp32 activations → no tensor cores) — so attention became the dominant DiT
cost. Fix: in `m1_ggml.hpp attention()`, under `PIXAL3D_FAST`, `ggml_cast` the matmul src0 (kp for QK,
vp for AV) to **F16** → ggml takes the fp16 tensor-core path; softmax stays fp32; with
`GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F` the matmul accumulates in fp32. Cross-attn (5 kv tokens) casts
harmlessly. q/k are qk-rms-normed (unit scale) so f16 range is safe.
- **M3b A/B: 59.6s → 52.5s (−12%), cosine vs fp32 0.999853 → 0.999877** (unchanged/marginally better).
  Combines fp16-accum speed with fp32-accum quality. Off by default (needs the env), on in the perf config.
- **E2E (warm cache): 216.0s**, N1=1120 exact, M=4631, verts 1479318 / faces 3096252 (vs f32
  1471142/3049548 = +0.55%/+1.5% — boundary realization within the bf16 noise floor; render = identical
  Miku). The f16-attn adds a hair more boundary noise than f16-weights-only (which was +0.008%), but
  both are near-lossless by cosine; f16-weights-only is the absolute-max-quality variant if that 0.5%
  ever matters.

### ▶ MEASURED peak VRAM (--fast run) = **8199 MiB (~8.0 GB), at the M3b DiT** (t=94s)
Sampled `nvidia-smi` at 5 Hz over a full `--fast --tex` E2E. Peak is the M3b sparse DiT (M=4633 tokens
→ the largest dense-attention working set: ~1 GB scores + 2.7 GB f16 weights + the f16-attn cast
buffers + activations). 12 GB card → ample headroom; ~0.5 GB over the 7.5 GB target — acceptable
(owner: "not a wall"). The f16-attn casts slightly inflate the DiT peak vs f16-weights-only. VRAM
reduction is parked behind feature-complete (the UV-atlas raster may shift the profile → re-measure
after). (Supersedes the Python-baseline guess of 6.3 GB alloc / 7.6 GB reserved in vram.json.)

### ▶ `pixal3d --fast` (NEW) = the validated perf config in one flag
`pixal3d --model weights_gguf_f16 --image in.png --out out.glb --tex --fast` sets `PIXAL3D_FAST=1` +
`GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F=1` (keeps `NVIDIA_TF32_OVERRIDE=0` so the f32 stages — incl. the
occupancy-sensitive SS-DiT→coords — stay bit-exact). Requires the **`weights_gguf_f16/`** GGUF set
(4 DiTs f16, decoders/dinov3/naf f32). Weights live on `/mnt/hdd/pixal3d/` (symlinked back).

### ▶ FINAL SCORECARD (Phase C, all 3 laps)
| stage | original f32 | **now (F16 DiT + f16-attn + GPU decode)** |
|---|---|---|
| **textured E2E** | **509.7s** | **216s warm** / ~250s cold-HDD = **−51% to −58%** |
| decodes (M3a+M4+M6) | 254.6s | ~45s (M3a 5.5 + M4 19.7 + M6 19.3) |
| DiTs (SS+M2+M3b+tex) | 192.7s | ~131s (SS 42 + M2 9.7 + M3b 50.7 + tex 29) |
| mesh fidelity | — | N1/M exact-ish, verts within 0.55%, render-identical Miku |

**NEXT LEVERS (documented, not yet done — diminishing returns vs the 3 laps above):**
1. **spike conv kernel** (~18s, the decode floor): naive fp32 implicit-GEMM, no tensor cores. A
   **bit-exact-preserving** win = better fp32 tiling / shared-mem weight staging / larger TN reuse
   (keeps the decode's current bit-exactness). A tensor-core (tf32/f16) rewrite is faster but would
   perturb subdiv decisions (judge by mesh IoU). Validate via `m4_gpu_test`.
2. **SS DiT** (42s, biggest single stage): 4096-token dense DiT, 30 blocks × 24 forwards. Per-forward
   fixed overhead (graph launch + uploads) × 24 may now matter; consider a persistent step-cgraph /
   cross-step KV-style reuse, or true flash-attn (`ggml_flash_attn_ext`).
3. **VRAM co-residency + imatrix-Q4** for the sub-4GB stretch (imatrix is the right tool for the
   distributed DiT quant error; capture per-channel act² from the 12-step forward → `ggml_quantize_chunk`).

---
## (ORIGINAL kickoff notes below — #1 hypothesis superseded by LAP 1 above)

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

---
## ✅ LAP 4 (2026-06-12) — cleanup build (A1/A2/A3) + VRAM re-profile. **peak 8199 → 5895 MiB (−28%, ≤7.5GB).**

Post-feature-complete punch-list (FINDINGS-14). Net VRAM/robustness, quality preserved (N1=1120, M=4631).
- **VRAM peak was MISATTRIBUTED to the M3b DiT.** Per-stage isolated `nvidia-smi`: M3b single-shot =
  4569 MiB; **NAF@1024 = 8185 MiB = the true peak** (its ImageEncoder runs 128-ch k3 convs at the 1024²
  guide res → `ggml_conv_2d` im2col ~4.8GB f32, the chain's biggest single allocation).
- **F16 im2col under `--fast`** (`naf_graph.hpp conv()`: cast the conv kernel→F16 so `ggml_conv_2d`'s
  im2col is F16; matmul still F32-accumulates). **NAF@1024 8185 → 5883 MiB**, near-lossless
  (`naf_1024_test` meanabs 8.06e-6 vs fp32 oracle; E2E N1=1120 unchanged). Default path stays f32.
- **Query-tiled attention** (`m1_ggml.hpp`, `PIXAL3D_ATTN_CAP_MB`, default 3072): bounds the
  `[tk,tq,head]` scores; **fixes the complex-asset OOM** (turtle M=15313 cleared the prior 11GB-alloc
  failure); Miku stays single-shot (unchanged). Replaces the flash-attn attempt (NaN on non-256-multiple
  n_kv with a null mask). Tiled M3b is bit-better (cosine 0.999945 @CAP256 vs 0.999877 single-shot).
- **A1 hole-fill** (`svae::fill_holes`, advancing-front ear-fill): chain boundary edges −75% (52425→
  13043), faces only. **Configurable `--decimate`** now on the plain GLB too; game-asset `--decimate
  40000` → bake/unwrap 9.5s (was 140s @150k). Full-watertight chart-collapse still needs a remesh.
- **A3 auto-camera** (`estimate_camera.py`, host MoGe-2): fov/distance EXACT vs the Python ref.
- **New E2E peak = NAF@1024 5895 MiB** (5.76 GB ≤ 7.5GB budget). Next-biggest = M3b DiT (4569). Remaining
  quality-preserving levers unchanged (spike conv ~18s; SS-DiT overhead). imatrix-Q4 = the sub-4GB play.

---
## ✅ LAP 5 (2026-06-12) — flash-attn LANDED on the sparse DiTs (M2/M3b). **M3b −19%, cosine 0.9991.**

The lap-1..4 flash attempts NaN'd; the root cause (cracked via capture-replay, FINDINGS-15) was NOT the
kernel/mask/q-pad/allocation — it's **f16 PV-accumulator overflow on un-normalized V**. V (unlike q/k) is
not QK-RMS-normed, so at low-t sampler steps the residual stream grows and V hits absmax ~1300; the cc8.6
`mma_f16` kernel's exp-weighted PV accumulate overflows f16 → +inf → softmax 0/0 → NaN. (`set_prec(F32)`
covers QK/softmax, NOT the PV accum — a real ggml-cuda gap.) **Fix** (`m1_ggml.hpp attention()` flash
branch): pre-scale V by 1/64 (power-of-2 = bit-exact), flash, scale output back up. `PIXAL3D_FA_VSCALE`.
- **m3b A/B (N=4734): dense-tiled 52.4s → flash 43.4s (−17%); cosine vs fp32 oracle 0.999144 (no NaN).**
- **E2E (`--remesh --tex --fast` + `PIXAL3D_FLASH=1`): N1=1120 exact, M3b DiT 50.7→41.0s (−19%), M2 9.7→9.1s.**
  No NaN through the chain; mesh clean. SS DiT stays DENSE (occupancy→coords, N1 must stay exact; flash's
  0.9991 would risk it). Gate `PIXAL3D_FLASH` (sparse M2/M3b only); tex DiT could be wired next.
- Cleaner follow-up: patch the `mma_f16` kernel PV accumulator to honor PREC_F32 (recovers 0.9991→0.9998,
  drops the V-scale workaround). Repro tooling left in-tree: `fa_repro.cpp` (FA_LOAD replay, FA_REALPATH,
  FA_QKDOWN/VDOWN, FA_GALLOC, FA_BLOCKS), m3b `PIXAL3D_FA_CAPTURE`+`CAPFWD`, harness `PIXAL3D_{SCHED,NO_GALLOC}`.
