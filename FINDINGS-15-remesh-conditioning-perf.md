# FINDINGS-15 — proper remesh + conditioning flags + in-process front-end + perf/VRAM

**Date:** 2026-06-12 (continues FINDINGS-14). Scope: the "do it properly" pass —
(1) real manifold-watertight REMESH replacing the mirror-cap flap hack; (2) expose the Trellis.2
conditioning flags; (3) in-process front-end (MoGe cam + rmbg service); (4) full perf/VRAM pull.
Validate vs the true-fp32 oracle; judge E2E by mesh IoU/render, loosen tol only for perf.

---
## 1. A1 — proper watertight REMESH = Marching **Tetrahedra** (not Marching Cubes) ✅

**Chose Marching Tetrahedra over classic Marching Cubes.** MC's 256-entry triangle table has 6
ambiguous cases that crack (re-introduce boundary) or go non-manifold without a topology-correct
table (Lewiner/MC33) — error-prone to port. A **tetrahedron has NO ambiguous cases**: with the
translation-invariant **Kuhn 6-tet decomposition** (all 6 tets share the cube main diagonal 0–7),
adjacent cubes split every shared face along the SAME world diagonal, so the surface is GUARANTEED
watertight (0 boundary) AND 2-manifold (every edge exactly 2 faces) — provable from first principles,
no magic table. (`remesh.hpp::marching_tetrahedra`, gated behind `--remesh` / `in.remesh`.)

- **Vertex dedup gotcha:** tet edges are NOT only axis-aligned cube edges — they include cube
  face-diagonals and the main diagonal. Keying a vertex by `(min,axis)` (cube-edge assumption) gave
  4.6M non-manifold edges. Fix = canonical global pair `(lo-corner, direction-mask∈1..7)` → dedups
  every shared grid edge across all cubes. **Result on Miku occupancy (1,547,112 voxels @ grid1024):
  verts 8,135,588 / faces 16,275,344, boundary=0, nonmanifold=0 — CLEAN MANIFOLD WATERTIGHT** (2.3s,
  pure host). Render = clean Miku (twin-tails/skirt/boots all correct; `miku_remesh_render.png`).

- **Manifold ≠ tight atlas — the real chart driver is NORMAL SMOOTHNESS, not topology.** Surprise
  measurement: the raw marching-tet mesh decimated→150k still fell to meshopt `simplifySloppy` AND
  unwrapped to **48,305 charts** (5196² atlas) — WORSE than the dual-grid's 30,091. Cause: MT places
  verts at edge midpoints → a diagonally-faceted "staircase" whose per-triangle normals are noisy;
  xatlas seeds a chart per few coherent-normal faces. Topology was never the chart blocker.
  **Fix = Taubin (λ/μ) smoothing** of the manifold mesh (`remesh.hpp::taubin_smooth`, uniform
  umbrella, alternating +λ/−μ passes — low-pass without Laplacian shrinkage; moves vertices only so
  watertight+manifold preserved). At grid1024 the sub-mm smoothing is invisible (actually closer to
  the organic original than the blocky voxels). _[smoothing sweep + chosen iters: TODO fill]_

- Textures bake **volumetrically** (grid_sample at the rasterized 3D position), so the remeshed
  topology is independent of the PBR volume — the UV-atlas bake works unchanged on the new mesh.

- **In-pipeline VALIDATED** (`pixal3d --remesh --tex --fast`, Miku): `[7b] REMESH (marching-tet on
  1,479,407 voxels, smooth 3): verts=7,668,622 faces=15,340,220 boundary=0 nonmanifold=0 = CLEAN
  MANIFOLD ✓ (9.5s)`. N1=1120 (conditioning defaults reproduce the validated stage-1).

- **Quality-decimation / tight-atlas goal NOT achievable from this remesh (honest finding).** meshopt
  quality collapse STALLS at ~13.8M faces on the (manifold, smoothed) MT mesh regardless of
  target_error (1.0) or options (Prune flag at err 1.0 nukes the whole mesh; at err 0.05 leaves the
  floor; Regularize doesn't help) → it falls to `simplifySloppy` (→139k, works). Euler χ→genus≈1043
  is modest, so genus alone doesn't explain it; meshopt barely engages on the voxel-lattice surface.
  Sloppy's irregular triangulation → noisy normals → xatlas seeds ~48k charts (WORSE than the
  dual-grid's 30k; 5196² atlas) — because the chart driver is NORMAL coherence, not topology, and
  Taubin-smoothing the source doesn't fix sloppy-decimated normals. **Conclusion: `--remesh` delivers
  the clean manifold watertight GEOMETRY (the owner's stated concern — no degenerate flaps; ideal for
  the plain/`--vcolor` GLB and downstream rigging), but is NOT better than the dual-grid path for the
  textured atlas (and its bake is slower). A tight atlas would need a genus/lattice-clean remesh
  (dual-contouring on a smoothed SDF) or attribute-aware decimation — scoped follow-up.**

Wiring: `m4_decode_mesh` (svp + svpg) gained `out_coords1024` to expose the grid-1024 occupancy;
`run_geometry` step 7b `--remesh` branch builds the marching-tet mesh (+ Taubin smooth,
`PIXAL3D_REMESH_SMOOTH` iters, default 3) and discards the dual-grid extractor output. Default path
stays BIT-EXACT vs o_voxel (remesh is gated behind `--remesh`).

- **Full E2E confirmed** (`pixal3d --remesh --tex --fast`): chain 243.9s, N1=1120, M=4631; remesh
  7.67M v/15.34M f manifold; textured `miku_remesh_tex.glb` (41MB) renders as a clean correctly-
  textured Miku (4 views). The atlas numbers CONFIRM the limitation in hard data: ComputeCharts
  **840s**, **47,888 charts**, 5160² atlas at **16.6% util** — emphatically loose/slow vs the
  dual-grid path. (So: ship `--remesh` for geometry; the textured-atlas path is slow + not improved.)

---
## 1b. A1 — TIGHT-ATLAS REMESH SOLVED (lap-16): solidify + coarse-MC + consistent winding ✅

The marching-tet conclusion above ("not better than dual-grid for the atlas") was **superseded** this
lap. The proper recipe lands a clean watertight low-poly mesh that beats Python on the atlas. Three
chained discoveries (each measured offline on `refs/stage5/head_coords.npy` via `remesh_test.cpp`,
seconds, no GPU):

1. **The M4 occupancy is a thin SHELL, not a solid.** Box-averaging a shell + iso=0.5 SHATTERS it
   into ~1243 disconnected fragments (genus≈−814) → xatlas needs a chart per fragment. Fix =
   **solidify**: coarse-rasterise the shell (seals the M4 extractor's small gaps), flood-fill the
   exterior (6-conn), interior=solid → ONE connected volume. (`marching_cubes_solid` in `remesh.hpp`.)
2. **March a COARSE grid directly** (downsample stride, default 6 → grid 170) on the box-blurred solid
   → a SMOOTH, watertight, single-component low-poly mesh (~205k faces, 0.7s) with NO decimation
   stall. (meshopt quality decimation genuinely can't collapse a full-res lattice MC surface — that
   was a real meshopt limit, not just normals.)
3. **THE lever: consistent winding.** The per-tet `od` orient test is noisy on skewed interpolated
   tris → ~10% of faces wound backwards (`is_winding_consistent=False`, adjacent-normal p90=178°).
   xatlas treats every normal-flip edge as a hard seam → **18,437 charts**. `orient_consistent()`
   (BFS over the manifold + global volume-sign flip) → **1,238 charts** (15×) AND it **unblocks
   meshopt quality decimation** (winding flips were blocking that too) → decimate to 80k QUALITY (not
   sloppy) → **356 charts**.

**Texture bake:** the solidified surface sits a few voxels off the sparse PBR shell, so the trilinear
`grid_sample` misses → black. Fix = **nearest-occupied-voxel fallback** (`VolIndex::find_nearest`,
expanding-shell search, radius = `stride*3+4`; the golden trilinear path stays bit-exact since the
fallback only fires when `tw==0`, which never happens on the on-shell golden mesh).

**Measured (Miku, stride6 → decimate 80k, `tex_bake_test`):** atlas 2444², **charts ~1045**,
**ComputeCharts 13s** (was 840s for marching-tet, 77s for dual-grid), **util 55.2%** (Python pyref =
**25%**, our old dual-grid = 20%). Render = clean smooth correctly-textured Miku, **visually ≈ Python
pyref** and clearly better than the old marching-tet (which had hair striping/faceting — *that* was
the "janky" complaint). `compare_remesh_3up.png`. **Reference check that reframed the lap:** the
Python pyref atlas is only ~25% util with 3.52M verts/3.25M faces (heavily seamed) — it is NOT the
"hundreds of charts / >50% util" the handoff assumed; cumesh gets few charts only via normal-cone
pre-clustering. We now MATCH/BEAT it without that. Wired into `pixal3d --remesh` (chain step 7b uses
`marching_cubes_solid`+`taubin_smooth`; bake passes the fallback radius). Env knobs:
`PIXAL3D_REMESH_{STRIDE,BLUR,SMOOTH}`. The plain marching-tet stays available as the absolute-max-detail
geometry base.

---
## 2. A1 — Trellis.2 conditioning flags exposed ✅

The chain hardcoded the sampler knobs Python's `run_inference` exposes. Now in `ChainInput` as a
per-stage `StageSampler {guidance, rescale, rescale_t, lo, hi, steps}` + `seed`, defaults == the
`inference.py` defaults (SS gs7.5/gr0.7/rt5/[0.6,1]/12; shape gs7.5/gr0.5/rt3/[0.6,1]/12; tex
gs1/gr0/rt3/[0.6,0.9]/12). CLI flags: `--seed`, `--guidance` (ss+shape shorthand = "how close to the
image"), `--steps` (all three), and per-stage `--{ss,shape,tex}-{guidance,rescale,rescale-t,steps}`,
plus `--mesh-scale`. The shape_slat_* config drives BOTH M2 (lr) and M3b (hr), matching Python.
`image_resolution` (512) is structurally baked into the cond graphs (CFG512/CFG1024); `max_num_tokens`
is superseded by query-tiled attention (A2) — both noted as API params, not wired to behaviour.

---
## 3. A3 — in-process front-end — SCOPED (MoGe is bigger than the handoff assumed)

**Correction:** MoGe-2 has **no intrinsics head**. `infer()` recovers intrinsics via
`recover_focal_shift()` over the FULL predicted **point map** + mask (`moge/model/v2.py:254-266`).
So a faithful in-process port needs the entire **ViT-L DINOv2 backbone** (SwiGLU + LayerScale +
register tokens) + **DPT points_head** + **mask_head** + the **focal-shift solve** — a multi-day,
separately-golden-validated port, NOT the "small intrinsics head" the handoff assumed. _[matte via
existing host rmbg SERVICE: TODO; MoGe backbone port: scoped follow-up, validate incrementally.]_

---
## 4. D — perf / VRAM pull (ncu/nsys evidence)

### ncu/nsys now WORK under docker (the toolset, sorted)
The host toolchain `ncu` is a stub (no nsight-compute install) and pixal3d builds host-side (needs
glibc 2.43, which the Ubuntu builder lacks) — that's why this project never profiled before. Fix:
**build container-native** inside `longcat-avatar-dev:builder` (glibc 2.39, g++13.3, CUDA 12.9):
- `ggml/build-cuda-docker` — cmake `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86` (in-container).
- `m3b_sampler_test_docker` — the M3b DiT in isolation; reproduces cosine 0.999878 in-container.
- **ncu**: real (Nsight Compute 2025.2.1 at `/usr/local/cuda/bin/ncu`); needs `--cap-add SYS_ADMIN`.
- **nsys** ("named weird"): bundled inside nsight-compute at `/opt/nvidia/nsight-compute/2025.2.1/
  host/target-linux-x64/nsys`. It COLLECTS but its in-place importer is broken (no libdw); fix =
  `QdstrmImporter` (same dir, `linux-desktop-glibc_2_11_3-x64/`) after `apt-get install libdw1`,
  then `nsys stats --report cuda_gpu_kern_sum <rep>`.

### M3b DiT GPU-time breakdown (nsys, --fast f16 path)
| % | kernel | meaning |
|---|---|---|
| **28.5%** | `cpy_perm_coalesced` (6930×) | permute/cont layout shuffles |
| 20.1%+6.8%+3.3% | cutlass f16 gemm | the matmuls (~30%) |
| **12.0%** | `soft_max_f32` | attention softmax (materialized scores) |
| **9.2%** | `convert_unary f→half` (6972×) | the f16 casts |
| 5.2% | `k_bin_bcast add` | residual/bias |
ncu SpeedOfLight on these: perm-copy SM 48%/DRAM 38%, softmax 40%/40%, cast 53%/66% — all
**under-saturated** = memory/overhead-bound, not compute. So **~38% shuffle/cast + 12% softmax**
(almost all in/around `attention()`) is pure overhead the matmuls don't need. This is exactly why
flash-attn is the #1 lever (it fuses QK+softmax+AV, no `[tk,tq,head]` scores, no separate perm/cast).

### flash-attn — ATTEMPTED, still NaN (deeper than the handoff's hypothesis)
Implemented "done right" per the handoff: permute q/k/v→[d,n,head], pad K/V `n_kv`→×256 (zeros),
F16 K/V, **F32 q** (the `mma_f16` D=128 kernel asserts `Q->type==F32`), F16 mask (built once,
shared across 30 blocks; `m1_ggml.hpp attention()` flash branch + `slat_dit_graph.hpp`, gated
`PIXAL3D_FLASH`). Result: the `ggml_cuda_flash_attn_ext_mma_f16_case<128,128,64,1>` kernel runs with
**no CUDA error but produces all-NaN output** — and it NaNs **even with an all-zero mask**. So the
NaN is NOT the null-mask OOB the handoff hypothesized (pad+mask does not fix it); the cc86/D=128
`mma_f16` path itself fails on these shapes (n_head=12, gqa_ratio=1, N≈4734). Left gated OFF;
**query-tiled dense attention stays the default** (validated, bit-exact, OOM-safe). A real fix needs
a ggml-cuda kernel patch or forcing a different FA kernel (vec/tile/wmma) — scoped follow-up. The
profiling above quantifies the prize (~50% of DiT is the addressable attention overhead).

### flash-attn — LANDED (lap-16): root cause = f16 PV-accumulator overflow on un-normalized V ✅
**FIXED + validated.** The NaN was NEVER the kernel, mask, q-pad, or allocation — it was **V magnitude**.
Diagnosis chain (the capture-replay method the owner suggested cracked it):
1. Standalone `fa_repro` (kernel + cont/permute/pad/cast + gallocr + recompute + 30 chained blocks +
   real captured block-0 q/k/v) **NEVER NaN'd**. compute-sanitizer on the real pipeline: **zero memory
   errors**. So not the kernel, not the ops, not OOB.
2. Per-forward scan: forwards 0-19 finite, **forward 20 (the LAST sampler step, t=0.214) goes all-NaN**.
   Captured fwd-20's q/k/v → **replayed through `fa_repro` → REPRODUCED the NaN** (offline, seconds).
3. Bisected with `FA_QKDOWN`/`FA_VDOWN`: scaling q/k does nothing; **scaling V down kills the NaN**.
   Cause: V is NOT QK-RMS-normed (only q/k are), so at low-t steps the residual stream grows and V
   hits absmax ~1300; the mma_f16 kernel's exp-weighted PV accumulator overflows f16 → +inf → softmax
   0/0 → NaN. `ggml_flash_attn_ext_set_prec(GGML_PREC_F32)` covers QK/softmax but NOT the PV accumulate.
**FIX** (`m1_ggml.hpp attention()`): pre-scale V by a power of 2 (1/64, EXACT in f16 — pure exponent
shift, zero precision loss), flash, then scale the output back up. `PIXAL3D_FA_VSCALE` tunable.
**Result: m3b flash cosine 0.999144 vs the fp32 oracle (≥0.999 ✓), NO NaN, m3b 52.4s→43.4s (−17%).**
Gated `PIXAL3D_FLASH` (applies to the sparse M2/M3b DiTs; SS DiT stays dense — occupancy-sensitive,
N1 must stay exact). Cleaner follow-up: patch the mma_f16 kernel's PV accumulator to honor PREC_F32
(would also recover flash 0.9991 → dense's 0.9998). Also added (defensive, kept): q-dim pad to ×256.

### (superseded) earlier lap-16 note: setup VERIFIED correct, NaN is the cc8.6 D=128 KERNEL — WRONG, see above.
Re-audited the FA setup against ggml asserts (`ggml.c:4187-4192`) and `slat_dit_graph.hpp` mask build:
all correct — mask F16 [nkvpad,nqpad,1,1] (256-padded), −65504 on pad keys, `mask->ne[0]==K n_kv`,
`mask->ne[1]>=Q n_q`, gqa_ratio=1, q F32 / K,V F16, q/k are rms-normed+roped (unit scale, f16-safe).
The dispatcher (`fattn.cu ggml_cuda_get_best_fattn_kernel`) forces **MMA_F16** for cc8.6 + D=128 +
n_q>1 (VEC only fires at n_q==1; line 533). Added env overrides to the fork
(`PIXAL3D_FA_{WMMA,TILE,VEC}` — legal since our mask is `ne[2]==1`, not per-head) and rebuilt
`ggml/build-cuda`. **Result: MMA_F16 → all-NaN (even zero mask); forcing WMMA_F16 → HANG (10 min,
0% GPU util, 397% CPU, never launches the DiT on GPU).** So it is NOT a kernel-selection fix — the
D=128/nh=12/gqa=1/N≈4734 shape breaks both Ampere FA kernels in this fork. **Next lap = a dedicated
`compute-sanitizer`/cuda-gdb session on the docker `ggml/build-cuda-docker`** (the env hooks are in
place to bisect): minimal standalone `ggml_flash_attn_ext` repro at exactly these dims + CPU ref,
sweep D=64/nh=8/gqa≥2 to pin the failing axis, then patch the kernel. The query-tiled dense path
stays the validated default (cosine 0.999877). Prize unchanged: ~50% of DiT GPU = attention overhead.

### Remaining levers (unchanged, quality-preserving): spike conv tensor-core, NAF spatial tiling.
### Quant LAST: DiTs are F16 now (Q8 = step down; plain-Q8 cosine 0.968, distributed error →
imatrix-Q8 before Q4); peak is NAF activation/im2col, NOT DiT weights, so weight-quant won't move it.
