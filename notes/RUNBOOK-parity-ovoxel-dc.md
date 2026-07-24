# image→rig Python-parity delivery — how to run

Produces a Python-Pixal3D-quality textured mesh from one image: **smooth O-Voxel dual-contour
geometry + direct volume bake** (no MC-solid staircase, no reproject colour-slide).
Proven on Miku + soldier (2026-07-25).

## Run it

```
cd tools/m1_ref/cpp_port
bash shootout/native_ovoxel_dc_parity.sh <input_image_or_matte.png> <outdir> [texsize=2048] [band=1] [taubin=2]
```

- **Output:** `<outdir>/parity.glb` (+ `.atlas.png`). That's the deliverable.
- **Time:** ~5–6 min (decode on 3060 ~4.5 min, DC + CPU bake ~1 min).
- **Input:** a matte PNG (RGBA or black-matte). Raw photos → matte first with BiRefNet
  (see `native_image_to_rig_from_image.sh`).

## What it does (4 steps, all automated)

1. `image_to_rig --no-clean --no-refine --material-cache-only --stage-dir --dump-geo`
   → keeps the **smooth O-Voxel decoder mesh** (`coarse.glb`) + caches the PBR volume; skips bake/rig.
2. `narrow_band_dc_probe coarse.glb … 1024 1` → dual-contour remesh (= Python `remesh_narrow_band_dc`).
3. `mesh_taubin … 2` → light feature-preserving smoothing.
4. `texture_rebake_native RP_VOLUME_DIRECT=1 ATL_BASECOLOR_SRGB=0 --bake volume-trilinear`
   → **direct** bake (no reproject slide), **raw** baseColor (Python-dark darks).

## Must-knows / gotchas

- **3060 ONLY.** CUDA enumerates fastest-first so index 0 = 5060 Ti. The wrapper pins the 3060 by
  UUID (`GPU-3b9ac5cf-…`). Override with `GPU_3060_UUID=` if the card changes.
- **Model dir:** `/home/dbrain/models/3d/geo_f16_native` (override `GEO_MODEL_DIR=`). Seed 42, res 1024.
- **Don't texture the refined mesh** — refine is a generative densify, its bake reprojects and slides
  colour (murky face, tie-on-collar). Refined mesh is a *separate* hi-detail asset, not the parity one.
- `--dump-geo` writes the PBR `.bin`s but NOT `coarse.glb`; `--stage-dir` writes `coarse.glb`. Need both.
- Orientation for A/B renders varies per character (Miku front = yaw 180, soldier front = yaw 0) — check.

## Eye-test / A/B

- Two-up page (rotatable): `puppy-eyetest/inline-3d/parity.html` (Python | native).
- Options board: `puppy-eyetest/inline-3d/tex-options.html` (★ tile = this).

## Full background

- Root cause + proof: memory `project_image_to_rig_coarse_parity_ovoxel_dc`.
- Manifest of every quality fix: `~/handoffs/tools/m1_ref/cpp_port/QUALITY-FIXES-MANIFEST-2026-07-24.md`.
- Not yet done: fold into `image_to_rig` as a `--dc-remesh` default (replace `build_mc_remesh` at
  `pixal3d_chain.hpp:488`); minor "bigger than Python" scale-normalization diff.
