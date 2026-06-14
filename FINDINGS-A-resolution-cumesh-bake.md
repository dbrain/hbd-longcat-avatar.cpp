# RESOLUTION — texture quality gap (closes HANDOFF-A-quality-gap-after-fillbg)

User confirmed in model-viewer: the cumesh GPU bake matches the reference (B) bang-on. The native
snap-to-dense detour is abandoned.

## What the gap actually was

Codex spent the prior session tuning the **native snap-to-dense reprojection** layer (RP_FRONTDOT,
RP_MAXDIST_VOX, TEX_FILL_BACKGROUND, Telea radius/blur). That was the wrong layer. Two facts settle it:

1. **The production design is a GPU-Python post-step**, not a pure-native bake. `bake_uv.py`'s own
   header documents it: the C++ chain dumps `dump_{mesh,dense,pbr}_*` (PIXAL3D_DUMP_BAKE=1) and the
   texture is baked by `bake_uv.py` (cumesh GPU xatlas unwrap + nvdiffrast raster + direct volume
   grid_sample + cv2 inpaint). The reference **B (`miku_500k.glb`) is itself a `bake_uv.py` output.**

2. **The quality driver is the UV unwrap, not the sampling.** Python uses cumesh's GPU xatlas →
   clean, non-folding charts (~16 s). Native's only unwrap options (precluster→xatlas, or full
   xatlas ComputeCharts) **fold and seam** → teal-interior splatter, cracks, "flower mouth". The
   native bake's `RP_OFF` branch already does the *identical* direct-volume grid_sample as Python
   (`tex_atlas.hpp:985`); it was being starved by a bad unwrap, then patched with reproject hacks.

## The production recipe (reproduces B from the current dump)

    PY=/mnt/hdd/3d/avatar-shootout/Pixal3D/.venv/bin/python
    $PY bake_uv.py --dump . --mesh dense --decimate 500000 --texsize 2048 --inpaint 3 \
        --out pyref_500k_repro.glb
    # cumesh simplify 3.12M -> ~490k tris, unwrap 816k v, 2048^2 atlas. ~16 s on the 3060.

B = `--mesh dense --decimate 500000 --texsize 2048` (490 358 tris vs B's 489 823 — same pipeline).

## Quality variant that ships as "A" on compare.html

    $PY bake_uv.py --dump . --mesh dense --decimate 500000 --texsize 2048 --inpaint 4 --ssaa 2 \
        --out pyref_500k_ssaa2.glb

`--ssaa 2` bakes at 4096^2 then area-downsamples to 2048^2 → cleaner chart edges (AA across texels).

## Do NOT reopen

- native snap-to-dense reproject (RP_*), TEX_FILL_BACKGROUND, precluster atlas tuning — all dead.
- The native C++ unwrap cannot beat cumesh without a real conformal GPU unwrap in C++ (the thing
  cumesh wraps), which is out of scope. Keep the Python cumesh bake as the production bake step.

## Production recipe (the keeper, confirmed by user)

    $PY bake_uv.py --dump . --mesh dense --decimate <N> --texsize <T> --inpaint <r> --ssaa 2 --out <glb>
    # 500k/2048 matches B; 4096 is sharper (seams sub-pixel); 200k/1024 + 100k/512 downscale clean.
    # TELEA gutter + ssaa2, NO --normal_offset, NO --dilate.

## DEAD — seam "fix" flags made it WORSE (do not reuse)

`--normal_offset` and `--dilate` (nearest-valid/Voronoi gutter) were added to chase the teal-on-skirt
seam slivers. User verdict in model-viewer: cracked the tie + other thin parts and looked cludgy;
plain TELEA ssaa2 stayed cleanest and matched B. Root causes:
- normal_offset pushes thin-part (tie) sample points PAST the shell into empty voxels -> teal cracks.
- nearest-valid gutter makes HARD Voronoi edges; TELEA's smooth diffusion reads better under mip/bilinear.
The flags remain in bake_uv.py but default OFF and are NOT recommended (recorded dead-end, not a lever).

## IMMACULATE recipe (push past "ok" — confirmed best in offline zoom)

    $PY bake_uv.py --dump . --mesh dense --decimate 500000 --texsize 8192 --inpaint 6 --ssaa 1 \
        --bilateral 7,30,7 --out imm_500k_8192.glb     # ~3 min, 144 MB, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

Two levers, both real wins (verified by tight UNLIT face crops vs B and vs the 4096 keeper):
- **8192 texel density** — the residual hairline noise is SEAMS between the many thin hair charts.
  At 8192 the seam ring is sub-pixel, so the teal hair reads smooth. (4096 already good; 8192 better.)
- **`--bilateral d,sigmaColor,sigmaSpace`** — edge-preserving smoothing on baseColor at supersample
  res. Evens the per-texel volume-sample speckle on flat skin/cloth WITHOUT blurring crisp boundaries
  (eyes, hair/skin edge) — the colour-similarity guard also stops it pulling teal-hair gutter into skin.
  7,30,7 is gentle; 11,45,11 stronger. This is the legit lever (unlike normal_offset/dilate).

### Game LODs — bake the texture HIGH, downsample it, MATCH the mesh to the texel budget

A natively-baked small texture (e.g. --texsize 512 directly) CRACKS: at 100k tris / 512^2 each UV
chart is smaller than its gutter, so seams dominate ("cracked mess"). The fix is the game-pipeline
standard, NOT KTX2 (KTX2 only shrinks file size, it does not touch cracking):
- bake the texture content at 4096 internal then area-downsample to the target (proper AA) via --ssaa,
- AND decimate the mesh to fit the texel budget (fewer, bigger charts).

    # 512 LOD: 4096 internal (ssaa 8) -> 512, mesh 25k tris      = 3.1 MB, clean
    $PY bake_uv.py --dump . --mesh dense --decimate 25000 --texsize 512  --ssaa 8 --inpaint 4 --bilateral 5,25,5 --out lod512.glb
    # 1024 LOD: 4096 internal (ssaa 4) -> 1024, mesh 68k tris    = 8.9 MB, ~= 4096 (slightly softer)
    $PY bake_uv.py --dump . --mesh dense --decimate 70000 --texsize 1024 --ssaa 4 --inpaint 5 --bilateral 7,30,7 --out lod1024.glb

Tiers: 25k/512 (3.1 MB) · 68k/1024 (8.9 MB) · 500k/4096+bilat (65 MB) · 500k/8192+bilat (144 MB ceiling).

## NOT native C++ — the bake is Python GPU on the C++ dump

The model/geometry/dense-shell/PBR-volume come from the native C++ Pixal3D chain (PIXAL3D_DUMP_BAKE).
The UV unwrap (cumesh GPU xatlas) + texture bake (nvdiffrast raster + cv2/scipy) is PYTHON. This is by
design (see top of this doc + bake_uv.py header): native C++ xatlas unwrap folds/seams and is too slow;
cumesh is the production unwrap. A pure-native bake is the abandoned path. Making it 100% native C++
would require a conformal GPU unwrap in C++ — large, and not worth it while cumesh works.

### What did NOT help
- **Denser mesh (full 3.12M `--mesh dense`)** — texture STARVES at 2048 (3M tris can't fit) → severe
  white/teal speckle, far worse. More geometry needs proportional texel budget, which hits the 12 GB
  wall. The 500k simplification face is already smooth; the win is texture res + bilateral, not polys.
- bake_uv now chunks the grid_sample (CH=6e6) + frees CuMesh VRAM (`del cm`) so 8192^2 fits in 12 GB.

## Open (not started)

- quantized/KTX2 GLB export (KHR_mesh_quantization + KTX2/Basis) — ~3-5x smaller for game use. trimesh
  writes float32 geo + PNG tex today, so the GLB sizes are pre-compression.
