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

## Open stretch goals (next)

- micro seam cracks on zoom (teal slivers on the black skirt, hairline face cracks) — seam/mip +
  thin-shell interior-sampling artifacts, present in B too. See bake_uv.py normal-offset + dilation.
- downscale to 1024/512 + quantized GLB for a game-usable asset.
