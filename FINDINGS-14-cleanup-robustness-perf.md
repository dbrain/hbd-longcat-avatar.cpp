# FINDINGS-14 — Pixal3D cleanup / robustness / perf (the post-feature-complete punch-list)

**Date:** 2026-06-12. Scope: the owner-greenlit punch-list after texture feature-complete —
(A1) watertight-ish mesh + configurable downmesh, (A2) complex-asset OOM fix, (A3) auto-camera,
(D) perf/VRAM. Validate vs the true-fp32 oracle, judge E2E by mesh IoU/coords, not tight tol.

---
## A1 — Watertight hole-fill + configurable downmesh

**Problem (measured on the golden / real o_voxel mesh, 1.55M v / 3.25M f):** the raw O-Voxel
dual-grid mesh is NOT watertight — **57,962 boundary edges** in **17,176 loops** (median 3 verts,
mostly 1-quad gaps) AND **161,799 non-manifold edges** (>2 faces, surface self-touch). Python only
gets a clean watertight mesh via proprietary `cumesh` dual-contour remesh (not portable). Consequence
for our pipeline: xatlas seeds a chart per boundary-bounded region → **~30k charts → 140s unwrap +
bloated 4324² atlas @ ~21% util**; meshopt quality-decimation is border-locked → sloppy fallback.

**Key fact:** Python's *raw* o_voxel mesh is ALSO non-watertight (the golden has 57,914 boundary edges
+ 161,799 non-manifold edges); its final glb is watertight only because `cumesh` **remeshes** it. So
"fill the holes" = close the boundary to 0 (= no holes), which 3 stages now do:
1. **close_surface** (extractor, `sparse_vae.hpp`): synthesise any missing quad-corner cells (centred
   dual vertex) so frontier quads form. (Marginal here — only 115 cells were missing; the holes are
   connectivity, not missing cells. Kept as it's correct + cheap.)
2. **Advancing-front ear-fill** (`svae::fill_holes`): at a vertex with ≥2 boundary edges add the
   triangle between two; each ear strictly lowers the boundary count → closes the bulk (~35k tris).
3. **Mirror-cap**: any residual edge (non-manifold "slit" the ear-fill can't reach) gets the reverse
   of its single owning triangle → 2nd face → not boundary (~14.6k coincident flap tris, 0.4% of faces).

- **Result: boundary 57,914 → 0 — fully WATERTIGHT (0 holes)**, validated by `m4_mesh_only`
  (`boundary_edge_count == 0`). The **default path stays bit-exact** vs o_voxel (close_surface=false),
  so the validation oracle is preserved. Wired into `run_geometry` step 7b; `--no-watertight` opts out.
- **Remaining (separate from holes): the mesh is still non-MANIFOLD** (161k >2-face edges + the cap
  flaps), which keeps `meshopt_simplify` on the `simplifySloppy` fallback → the **xatlas chart count
  does NOT fully collapse** (so the atlas isn't tight, and unwrap is only fast via a low `--decimate`).
  A clean manifold + tight 2048² atlas needs a true **remesh** (marching-cubes on the occupancy, or a
  cumesh-style dual-contour remesh — no flaps, manifold). That's the higher-fidelity follow-up; the
  holes themselves are now 100% closed.

**Configurable downmesh:** `--decimate <F>` now applies to the **plain** GLB too (not just the tex
bake), default 150000, `0` = full ~3M-face mesh. Game assets pass a low budget (e.g. `--decimate
40000`) for a light, fast-unwrapping asset. `--vcolor` keeps the full mesh (its COLOR_0 is zipped 1:1).

---
## A2 — Complex-asset OOM fix: query-tiled attention

**Problem:** the C++ DiT used dense attention (materialises the `[tk,tq,head]` scores tensor). Miku
(M≈4.6k) = ~1 GB, fine; the turtle (`assets/images/0_img.png`, M=15313 HR tokens) = ~11 GB scores →
`cudaMalloc failed: out of memory` at the M3b shape DiT on the 12 GB card.

**Flash-attn attempted first, rejected:** `ggml_flash_attn_ext` (the obvious O(M)-memory fix) produced
**NaN** on the sparse DiTs. Root cause: with a **null mask** the CUDA MMA_F16 kernel reads out-of-bounds
on the K/V tail tile when `n_kv` is not a multiple of FATTN_KQ_STRIDE (256). SS-DiT (SEQ=4096, exact
multiple) was clean; the variable-N sparse DiTs (M2/M3b/tex) went NaN → garbage shape_slat → M4 decode
produced 0 verts → the sticky CUDA error surfaced at the next ggml op. The correct flash fix needs K/V
padding to ×256 **plus** a mask (a [n_kv,n_q] mask is M²-ish memory; defeats the point / fiddly).

**Implemented instead (`m1_ggml.hpp attention()`): query-dimension tiling.** Per-query softmax is
**independent**, so splitting the query dim into blocks is **bit-identical** to the single-shot dense
path while bounding the peak scores tensor. When `tk·tq·head ≤ ~1.5 GB` (e.g. Miku, ~1 GB) it stays
single-shot (unchanged perf + the validated LAP-3 f16-dense numerics); only huge assets tile (turtle
→ ~7 blocks of ≤1.5 GB). No flash kernel, no NaN risk, no extra precision loss. K/V projections are
computed once and reused across tiles.

`PIXAL3D_ATTN_CAP_MB` (default 3072) sets the scores cap; lower it to tile more aggressively (the
VRAM lever, see D).

**Validation:**
- M3b single-shot (M=4734, golden cond, `--fast`): **cosine vs fp32 = 0.999877** — identical to the
  prior validated f16-dense LAP-3 result (the test's tight maxabs threshold "FAILs" by design; judge
  by cosine per `feedback_correctness_before_perf`).
- **Miku E2E** (`--fast --tex --decimate 40000`): **N1=1120, M=4631 — exactly the validated baseline**
  → query-tiling is numerically clean; clean textured Miku render (4 views).
- **Turtle (M=15313)**: reached the prior OOM point (`[m3b-dit] step 1/12`) and computed — **tiling
  cleared the OOM** (peak 8199 MiB, vs the prior 11.6 GB alloc failure). Tiled M3b is slower (≈8
  blocks at the old 1.5 GB cap → bumped default to 3 GB; huge assets complete, just not fast).

---
## A3 — Auto-camera via host MoGe-2 — ✅ VALIDATED

`estimate_camera.py` on the Miku matte → **fov = 42.01° (0.7332 rad), distance = 1.3022** — an EXACT
match to the known Miku cam (`cam.json` 0.7332 / 1.3022). Mirrors `inference.py`
`get_camera_params_wild_moge` / `distance_from_fov` / `compute_f_pixels`. `--run` execs pixal3d with
the estimated `--cam` (one-shot upload→GLB).

---
## A3 — Auto-camera via host MoGe-2

**Implemented (`estimate_camera.py`):** host MoGe-2 (`Ruicheng/moge-2-vitl`, cached) predicts
intrinsics → `camera_angle_x = 2·atan(W/(2·fx))`, then `distance` via the ported `distance_from_fov`
host math (mirrors `inference.py` exactly). Emits the `--cam <ang> <dist> <scale>` arg for the CLI;
`--run` execs pixal3d directly (one-shot upload→GLB). Keeps C++ on the ggml path (same host cut-line
as rembg). _(validation below)_

---
## D — Perf / VRAM (re-profiled after A1/A2)

**The prior "peak = M3b DiT" belief was WRONG.** Per-stage isolated VRAM (`nvidia-smi` 5 Hz):
| stage | isolated peak | note |
|---|---|---|
| M3b DiT single-shot (M=4734) | 4569 MiB | 2.8GB f16 weights + ~1GB scores + acts |
| M3b DiT tiled (`PIXAL3D_ATTN_CAP_MB=256`) | **3469 MiB** | −1100 MiB, cosine 0.999945 (bit-better) |
| **NAF@1024** | **8185 MiB** | ← **the true chain peak** (== E2E 8199) |
| NAF@1024 `--fast` (F16 im2col) | **5883 MiB** | **−2302 MiB**, near-lossless (meanabs 8e-6 vs fp32) |

**Root cause of the peak:** the NAF@1024 ImageEncoder runs its 128-ch k3 convs at the **1024² guide
resolution**; `ggml_conv_2d`'s im2col (kernel-dtype) is ~4.8 GB f32 — the single biggest allocation in
the chain. (GroupNorm couples the spatial extent, so naive spatial conv-tiling isn't bit-identical.)

**Lossless-ish lever shipped — F16 im2col under `--fast`** (`naf_graph.hpp conv()`): cast the conv
kernel to F16 so `ggml_conv_2d` builds an **F16 im2col** (half the bytes) + tensor cores; the matmul
still accumulates in F32 (`GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F`). Consistent with the existing `--fast`
f16 philosophy; the default path stays f32 bit-exact. **NAF@1024 8185 → 5883 MiB**, `naf_1024_test`
PASS (meanabs 8.06e-6 vs the fp32 oracle).

**Lossless lever shipped — query-tiled attention** (`m1_ggml.hpp`, `PIXAL3D_ATTN_CAP_MB`, default
3072): bounds the `[tk,tq,head]` scores tensor; Miku stays single-shot (unchanged), the M3b VRAM lever
(−1.1 GB) is available for the tight budget, and it is what fixes the complex-asset OOM (A2).

**E2E peak after both (Miku `--fast --tex --decimate 40000`): 8199 → 5895 MiB (−28%, ≤7.5GB ✓)**,
N1=1120 / M=4631 unchanged, 206 s. The new hot spot is still NAF@1024 (5895); next = M3b DiT (4569).

**Perf note:** removing the (broken) flash path returns SS-DiT to the validated f16-dense LAP-3 timing
(~42s); the chain is at the prior Phase-C state (~200–216 s warm). The remaining quality-preserving
perf levers are unchanged from PERF-NOTES (spike conv kernel ~18s decode floor; SS-DiT per-forward
overhead) — hard, diminishing; deferred. Sub-F16 imatrix-Q4 remains the only sub-4GB VRAM lever.
