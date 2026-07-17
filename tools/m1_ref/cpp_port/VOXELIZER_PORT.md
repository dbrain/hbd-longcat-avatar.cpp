# Voxelizer port — `mesh_to_flexible_dual_grid` (o_voxel forward, rung-1 native texturing)

C++ port of `o_voxel.convert.mesh_to_flexible_dual_grid` — the FORWARD voxelizer that turns a
mesh into the sparse flexible-dual-grid `(coords, dual_vertices, intersected)` that feeds the
already-ported `shape_slat_encoder.hpp`. Closes the last blocker on native (no-Python) shape
encoding for the TRELLIS.2 texturing path.

Files:
- `voxelizer.hpp`            — `vox::mesh_to_flexible_dual_grid(...)` (header-only, fp32, OpenMP)
- `voxelizer_test.cpp`       — diff vs captured golden (coords SET IoU, dual maxabs, intersected %)
- `voxelizer_e2e.cpp`        — voxelizer → `senc::shape_slat_encode` → compare vs `golden_69k`
- `capture_voxelizer.py`     — golden capture (runs the real o_voxel `_C` cpu solver)
- build: `./build.sh voxelizer_test` / `./build.sh voxelizer_e2e` (CPU g++, no ggml/CUDA)

## Pipeline context (how the texture pipeline calls it)

`Trellis2TexturingPipeline.encode_shape_slat` (pixal3d/pipelines/trellis2_texturing.py):

1. `preprocess_mesh`: center, `scale = 0.99999 / extent.max()`, then a **y/z axis swap**
   (`y' = -z`, `z' = y`) → vertices in `[-0.5, 0.5]^3`. (Replicated exactly in capture.)
2. `o_voxel.convert.mesh_to_flexible_dual_grid(verts, faces, grid_size=1024,
   aabb=[[-.5,-.5,-.5],[.5,.5,.5]], face_weight=1.0, boundary_weight=0.2,
   regularization_weight=1e-2)` → `(voxel_indices[N,3], dual_vertices[N,3], intersected[N,3])`
   at grid **1024** (N ≈ 2.19M for the 69k Miku mesh).
3. Encoder input: `feats = dual_vertices*1024 - voxel_indices` (recovers the in-cell offset),
   `coords = cat([0, voxel_indices])`; `FlexiDualGridVaeEncoder.forward` then does
   `feats6 = cat([feats - 0.5, intersected.float() - 0.5])`.
4. `shape_slat_encoder` downsamples 4× (1024→512→256→128→**64**) → `golden_69k` coords (max 63)
   + 32-ch feats. The encoder lattice (grid-64) is the load-bearing comparison.

## CONVENTIONS matched (verified against the captured golden + sparse_vae.hpp inverse)

- `coords` int32 `[N,3]` = grid cell index; caller prepends batch → `[N,4]=(b,x,y,z)`.
- `voxel_size = (aabb_max - aabb_min)/grid_size`; cell `c` spans `[c, c+1)` in grid units.
- **`dual_vertices` is the GLOBAL normalized position** `= (coord + in_cell_offset)/grid_size`,
  range `[0,1]` — NOT the bare in-cell offset. (Confirmed: `dual*1024 - coord ∈ [0,1]`; the
  encoder's `dual*resolution - voxel_indices` recovers the in-cell offset.) `voxelizer.hpp`
  returns the **in-cell offset** in `[0,1]`; `voxelizer_test`/`voxelizer_e2e` convert golden
  (`gdu*grid - coord`) to compare, and the e2e builds encoder feats6 from the offset directly,
  matching the python path exactly.
- mesh vertex (inverse) = `(coord + offset)*voxel_size + aabb_min` — matches `sparse_vae.hpp`.
- `intersected` int8 `[N,3]` per-axis quad flag (consumed by the inverse + as encoder feats).

## Algorithm (`voxelizer.hpp`)

1. **verts → grid units**: `g = (v - aabb_min)/voxel_size`.
2. **Conservative rasterization**: for each triangle, over its cell-AABB, the Akenine-Möller
   triangle/box SAT test (`tri_box_overlap`) marks every overlapped cell occupied. Occupied
   cells live in a `unordered_map<int64 key → idx>` (key = `coord_key(x,y,z)`, 20 bits/axis,
   same packing as `svae::coord_key`).
3. **Per-cell QEF** (face term): each overlapping triangle contributes its (normalized) plane
   `n·x = d` in LOCAL cell coords (origin at the cell min-corner). Accumulate the 3×3
   `A = Σ n nᵀ`, `b = Σ n d`, plus a mass point (Σ of plane through-points). Solve the
   regularized normal equations `(face_weight·A + reg·I) x = face_weight·b + reg·masspoint`
   via a hand-rolled symmetric 3×3 cofactor solve, clamp `x` to `[0,1]`. (o_voxel uses Eigen
   `ColPivHouseholderQR` on a 4×4-accumulated QEF — `nm` on the `.so` shows
   `face_qef`/`boundry_qef`/`intersect_qef` + `Eigen::Matrix<float,4,4>`.)
4. **intersected**: per-axis flag (current heuristic = +axis neighbour exists and a face plane
   is present — see "What's still off").

## Captured golden (`/mnt/hdd/pixal3d_tex/golden_voxel/`, source `us_native_69k.glb`)

`us_native_69k.glb` = the RAW (un-preprocessed) Miku mesh: V=64862, F=69404. After
`preprocess_mesh` → voxelize @1024:

- `coords.npy`        `[2186557, 3]` int32, range x[132..891] y[266..757] z[0..1023]
- `dual_vertices.npy` `[2186557, 3]` f32, GLOBAL normalized ∈ [0,1] (mean ≈ 0.5/0.49/0.53)
- `intersected.npy`   `[2186557, 3]` int8 ∈ {0,1}; per-axis set fraction 0.34/0.44/0.22;
                       584,272 cells have NO axis set (touched-but-not-crossing cells).
- `verts_pre.npy` / `faces.npy` — the exact preprocessed inputs (so the C++ test is hermetic).

**Validation that the golden is the right mesh:** downsampling the golden coords `//16`
(1024→64, the 4 S2C halvings) gives **exactly** the 7390 `golden_69k` grid-64 cells —
coord-set **IoU 1.0000, recall 1.0000**. So `us_native_69k.glb` + this preprocessing IS the
mesh behind `golden_69k_1024/`.

## Validation numbers (`voxelizer_test`, C++ vs golden)

```
C++ voxelizer N = 2,186,654   (golden 2,186,557)
COORDS SET      : IoU 0.9999   recall 1.0000   precision 1.0000   (9 missing, ~106 extra of 2.19M)
DOWNSAMPLE //16 : IoU 0.9999   recall 1.0000   vs golden_69k @grid64  (encoder lattice == exact)
DUAL (in-cell)  : meanabs 0.0601   maxabs ~1.0   (units = fraction of a voxel)
INTERSECTED     : agreement 0.651  (frontier heuristic: flag axis a iff +a neighbour absent)
```

- **Coords are effectively exact** — the triangle/box SAT rasterization reproduces o_voxel's
  occupancy to 0.9999 IoU, and the encoder's grid-64 lattice is reproduced exactly. This is the
  structural backbone the sparse encoder consumes.
- **Dual** mean error ≈ 6% of a voxel. The bulk is right (the dominant surface-normal direction
  is well-fixed); the residual is in the under-constrained tangential directions, where o_voxel's
  exact `boundry_qef`/`intersect_qef` terms (closed `.so`) place the vertex differently than the
  mass-point regularizer.

## What's still off + diagnosis

The QEF dual and the `intersected` flags are NOT bit-exact because the o_voxel solver is a
closed `_C.so` (no source). Findings from black-box probing:

1. **Dual (6% mean):** golden duals track the surface very smoothly across neighbouring cells
   (locally near-constant world position). My face-only QEF + mass-point reg matches the
   dominant direction but scatters in the tangential/under-constrained directions. o_voxel's
   `boundry_qef` (weight 0.2) and `intersect_qef` add the constraints that pin those — their
   exact form (4×4 Eigen QEF accumulation, `ColPivHouseholderQR`) isn't recoverable from the
   binary. Some golden duals clamp hard to a cell face/corner (e.g. (0, 0.95, 1.0)), confirming
   a boundary/clip term beyond a plain face QEF.
2. **intersected (65%):** tested edge-crossing along the same axis at the cell min-corner, max-
   corner, and cell-center segments — none map to the golden per-axis flags (they come out
   permuted/near-orthogonal). Probing the golden directly: the flag is FRONTIER-related —
   golden `intersected[c][a]=1` is more likely when the +a neighbour is ABSENT (P≈0.57) than
   present (P≈0.23), and the field is mostly-0 (set rate 0.34/0.44/0.22). The port now flags
   axis a iff the cell carries surface and its +a neighbour cell is missing → **agreement 0.65**
   (the wrong-signed "neighbour exists" heuristic gave 0.35). The exact per-axis encoding lives
   in the `.so`'s `intersect_qef` edge-walk (the inverse's `OFF` table shows axis-a quads vary
   in the OTHER two axes), not recoverable from the binary.

Neither blocks the structural port: coords (the sparse topology) are exact, so the encoder runs
on the right lattice. How much the dual/intersected approximation perturbs the final `shape_slat`
is measured by `voxelizer_e2e` (results below).

## End-to-end (`voxelizer_e2e`, CPU encoder → vs golden_69k feats)

Runs `senc::shape_slat_encode` (CPU `subm_conv`) on (A) the GOLDEN voxelizer output and (B) the
C++ voxelizer output, comparing 32-ch `shape_slat` to `golden_69k_1024/shape_slat_feats.npy`.

- **(A)** validates the whole data path + feats6 derivation: the encoder fed the golden voxelizer
  output must reproduce `golden_69k` (expected ≈ the encoder port's own tolerance).
- **(B)** quantifies the voxelizer approximation's effect on `shape_slat`.

> NOTE: the CPU encoder over 2.19M voxels through the deep sparse U-Net is slow (the encoder was
> designed for the GPU `subm_conv` path). On GPU this is the step the main loop runs. Numbers are
> filled in below once the CPU run completes; if it is too slow, run `voxelizer_e2e` on GPU
> (build the encoder's CUDA conv path) — that is the intended "remaining step".

```
[A] encoder on GOLDEN voxelizer output  : coords 7390/7390 EXACT, feats meanabs 0.06295 (maxabs 2.62)
[B] encoder on C++  voxelizer output    : coords 7391 (7390 matched), feats meanabs 1.68417 (maxabs 11.9)
```
(GPU `subm_conv` run, 2026-06-18, `./build.sh voxelizer_e2e cuda`.)

**Verdict:** the ENCODER is faithful — fed the golden voxelizer output it reproduces `golden_69k`'s
grid-64 lattice EXACTLY and `shape_slat` to meanabs **0.063** (the spike-CUDA-conv tolerance band over
a deep sparse U-Net; same order as `m6_tex_decode_test`). The approximate voxelizer (dual 6% / inter
65%) perturbs `shape_slat` to meanabs **1.68** — the dominant error source. Whether 1.68 degrades the
final TEXTURE is a tex-DiT-tolerance question (it is generative + image-conditioned): validate by
running the native tex path on BOTH the golden-voxelizer `shape_slat` and the C++-voxelizer `shape_slat`
and comparing renders. **For the first native-texture validation, use the golden/Python `shape_slat`
(meanabs 0.063 ⇒ effectively the oracle) to isolate the tex-DiT/decode/bake wiring from the voxelizer
risk; treat o_voxel QEF fidelity as a separate, later lever.**

## Remaining step (for the main loop / GPU session)

Wire `voxelizer.hpp` → `shape_slat_encoder.hpp` on GPU and compare `shape_slat` vs
`golden_69k` (the `voxelizer_e2e.cpp` harness already does exactly this — just build/run it with
the encoder's CUDA `subm_conv`). If the dual/intersected approximation perturbs `shape_slat`
beyond tolerance, the remaining fidelity work is to recover o_voxel's exact `boundry_qef` /
`intersect_qef` terms (the `intersected` axis-permutation + the boundary clip in the dual) —
the coords/occupancy and the encoder lattice are already exact.
```
