# FINDINGS-13 — UV-atlas PBR textures (feature-complete: real `to_glb` textures, not COLOR_0)

**Date:** 2026-06-12. **Goal:** replace the interim per-vertex `COLOR_0` bake with full
**UV-atlas PBR textures** (baseColor + metallicRoughness + TEXCOORD_0), matching the Python
`o_voxel.postprocess.to_glb` / `Trellis2TexturingPipeline.postprocess_mesh`. **Status: DONE +
validated** (correct textured Miku, parity vs Python confirmed). Files: `tools/m1_ref/cpp_port/`.

## What the Python reference actually does (two paths)
- `inference.py:265` → `o_voxel.postprocess.to_glb(..., remesh=True, texture_size=4096)` — the FULL
  path: `cumesh` fill_holes + **dual-contour remesh** + decimate(1M) + BVH-reproject + xatlas
  `uv_unwrap` + nvdiffrast raster + `flex_gemm.grid_sample_3d` + `cv2.inpaint`. The remesh/BVH/
  decimate are **proprietary CUDA (`cumesh` `_C.so`)** — not portable.
- `trellis2_texturing.py:287` `postprocess_mesh` — the **no-remesh/no-decimate** path: xatlas
  `uv_unwrap` → nvdiffrast raster → `grid_sample_3d` at `(pos+0.5)*resolution` → `cv2.inpaint` →
  PBR material. This is exactly what `to_glb` reduces to without remeshing, and the **portable
  recipe**: the only net-new MATH op is `grid_sample_3d`; the rest is geometry (UV + raster + embed).

We port the `postprocess_mesh` path. Texture CONTENT is identical (same volume, same trilinear
sample); only the mesh topology differs (we keep/decimate the raw extracted mesh; Python remeshes).

## The pieces (all in `tools/m1_ref/cpp_port/`)
1. **`tex_grid_sample.hpp`** — host port of `flex_gemm` `GridSample3dTorch._trilinear` (the one net-new
   math op). Voxel at int coord `c` = cube centred at `c+0.5`; 8 neighbours `int(q±0.5)`; weight
   `∏(1-|c+0.5-q|)`; **renormalise by the sum of present-neighbour weights** (missing → 0). Hashmap
   over the sparse coords (`unordered_map`), OpenMP over queries.
   - **VALIDATED bit-exact**: vs `flex_gemm` GPU on the real 1.5M-voxel PBR volume + 1.5M mesh-vertex
     queries → **maxabs 4.77e-7** (`grid_sample_test`, `tex_grid_sample_capture.py`). numpy-ref vs
     flex_gemm = 3.6e-7 (confirms the algorithm before the C++).
2. **`thirdparty/xatlas.{h,cpp}`** (jpcy/xatlas, MIT) — UV unwrap (== what `cumesh.uv_unwrap` wraps).
3. **`thirdparty/meshoptimizer/`** (zeux, MIT) — quadric/sloppy decimation (== `cumesh.simplify` role).
4. **`tex_atlas.hpp`** — `texatlas::bake(verts, faces, pbr_feats, pbr_coords, grid_res, tex_size,
   decimate_target_faces)`: decimate → xatlas unwrap → **CPU triangle rasterizer** (barycentric 3D
   surface position per texel) → `grid_sample_3d` the per-voxel 6-ch PBR → inpaint gutter → baseColor
   (RGBA) + metallicRoughness (R0/G=rough/B=metal) uint8 atlases. No BVH needed (no remesh → the
   rasterized position IS on the original surface).
5. **`glb_textured.hpp`** — textured glTF 2.0 writer using **nlohmann json** (`thirdparty/json.hpp`)
   + embedded PNG textures (`stb_image_write`): POSITION/NORMAL/TEXCOORD_0 + 2 PNG images + samplers +
   `pbrMetallicRoughness{baseColorTexture, metallicRoughnessTexture}`. (Also satisfies the task #3
   "richer glTF → nlohmann" cleanup for the textured path.)
6. **`pixal3d --tex`** wired: `run_geometry` now surfaces the full PBR volume (`out_pbr_feats` +
   `out_pbr_coords`); the CLI bakes + writes the textured GLB. `--texsize N` (default 2048),
   `--decimate F` (default 150000), `--vcolor` (the old interim COLOR_0 path).

## Validation / parity
- **grid_sample**: bit-exact 4.77e-7 (above).
- **Visual**: `tex_bake_test` on the golden mesh + golden PBR volume → `render_textured.py` =
  **correctly-textured Miku** (teal twin-tails, gray top, black skirt + thigh-highs, skin tones — all
  in the right places). Side-by-side vs the Python reference (`tex_bake_python_ref.py`, the SAME
  `postprocess_mesh` core on the SAME golden mesh+volume) = same character, same colours.
- **Atlas content**: baseColor valid-region mean RGB — mine `[51,108,100]` vs Python `[51,101,94]`
  (≈ the documented COLOR_0 mean `[0.20,0.41,0.38]·255`). Matches; small delta = different UV layout +
  decimation + inpaint (TELEA vs dilation).

## Parity statement (what differs from core Python)
- **Texture content & sampling: matched** (trilinear `grid_sample_3d` bit-exact; same baseColor/
  metallicRoughness packing, same PBR layout `base_color3/metallic/roughness/alpha`, alphaMode OPAQUE,
  doubleSided, metallic/roughnessFactor 1.0).
- **Mesh topology: differs by design** — we do NOT replicate `cumesh`'s dual-contour **remesh** +
  BVH-reproject + hole-fill (proprietary CUDA). We decimate (meshoptimizer) + xatlas-unwrap the raw
  O-Voxel mesh directly. Result is the same surface; UV layout is xatlas-default (Python pre-clusters
  charts by normal-cone), so the atlas pixel arrangement differs (content is the same).
- **inpaint**: dilation rings vs `cv2.INPAINT_TELEA` (gutter-seal only; visually equivalent).

## Generalization data point (non-Miku)
A non-Miku asset (the turtle-castle `assets/images/0_img.png`) ran through the SAME flow (raw photo →
`preprocess_photo.py` → `pixal3d --tex`). Stages 1-5 ran (N1=3605, M3a Nh=1.14M → M=15313 HR tokens),
then it **OOM'd at the M3b shape DiT** (`gallocr_alloc_graph failed`). Cause: the C++ DiT uses DENSE
attention — the M² scores tensor at M=15313 (≈ 15313²·12·4 B ≈ 11 GB) blows the 12 GB card. Miku-class
complexity (M≈4.6k → ~1 GB scores) is fine. **This is the already-parked "DiT flash-attn" perf lever**
(Python uses `ATTN_BACKEND=flash_attn`, no materialized scores) — orthogonal to the texture work, but
it bounds how complex an uploaded asset the chain handles today. Relevant to the upload-PNG API: high-
token objects need the flash-attn DiT (and/or the Python token-budget downstep loop) before they fit.

## KNOWN PERF ITEM (parked — feature works, this is speed only)
xatlas `ComputeCharts` is the bake bottleneck: the O-Voxel dual-grid mesh has **~24k tiny holes/
boundaries** (intrinsic to the extraction — a quad only forms where all 4 neighbour voxels exist), so
xatlas makes **a chart per boundary-bounded region** (~24k charts @110k faces → 77s). Confirmed
TOPOLOGY-driven (maxCost=1e9 + normalDeviationWeight=0 still gives 24,635 charts). Python avoids this
by **remeshing to a watertight mesh first** (cumesh). Mitigations:
- **Current**: decimate to ~150k faces (sloppy) → ~100s unwrap → clean textured Miku. Tunable via
  `--decimate`.
- **The real fix (parked, the Phase-C-style perf lever)**: port `cumesh`'s normal-cone face
  CLUSTERING (region-grow faces within a cone half-angle into ~hundreds of charts, planar-project
  each, xatlas `PackCharts` only — skips the expensive per-boundary `ComputeCharts`). OR a
  hole-fill/weld pre-pass to cut the boundary count. Either makes full-res unwrap fast.
