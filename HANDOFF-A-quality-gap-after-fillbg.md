# HANDOFF A2 - Native Pixal3D/Trellis texture quality gap after fill-background pass

## Mission

Make the native C++ Pixal3D/Trellis.2 textured GLB match or beat the Python/cumesh reference ("B") while
also supporting smaller game-asset outputs. The user is judging quality visually in `model-viewer` via
`tools/m1_ref/cpp_port/compare.html`, not by offline pyrender-style crops.

Current user verdict: A is still bad. The latest A may be less broken than some earlier attempts, but it
does not reach B. Do not declare victory without the user confirming in model-viewer.

## Current compare page state

Serve from:

```bash
cd /home/dbrain/dev/longcat-sparse-spike/tools/m1_ref/cpp_port
python3 -m http.server 8011
```

Current `compare.html` is gitignored and points textured mode at:

- A: `codex_cpucluster_500_fillbg_strictsnap_telea_blur04.glb?v=fillbg-20260614-1021`
- B: `miku_500k.glb?v=ref-20260613`
- C: `codex_cpucluster_500_strictsnap_telea_blur04.glb?v=cpucluster-20260614-1004`

The server was running as:

```text
python3 -m http.server 8011
```

If the user says the page did not change, first inspect `compare.html` and the network-loaded filenames.
They already hard-refreshed once and saw files load correctly, so visual sameness is likely real, not cache.

## Source commits from this session

Relevant recent commits:

```text
3c437c8 Fill native atlas background before resizing
ec2ee5d Reject suspect texture reprojection samples
1a56247 Add native atlas inpaint path
aea142e Keep atlas raster supersampling opt-in
78153a4 Improve native atlas raster coverage
f8ac208 Add native CuMesh texture atlas path
```

`3c437c8` is the latest source change. It adds:

- `dilate_background(...)` in `tools/m1_ref/cpp_port/tex_atlas.hpp`
- optional `TEX_FILL_BACKGROUND=1`
- optional `ATL_FIT_RES=1`, `ATL_NO_BLOCK_ALIGN=1`, `ATL_NO_BILINEAR=1` diagnostics

Generated GLBs/PNGs and `compare.html` are ignored and were not committed.

## What was tried today

### 1. Strict reprojection filtering

Goal: reject bad texels caused by snap-to-dense choosing an implausible dense triangle.

Added controls:

- `RP_NO_BACKFALLBACK=1`
- `RP_MAXDIST_VOX=3`
- `RP_FRONTDOT=0.2`

Representative output:

```bash
ATL_NATIVE_CUMESH=1 CUMESH_TARGET=500000 ATL_PAD=4 \
RP_NO_BACKFALLBACK=1 RP_MAXDIST_VOX=3 RP_FRONTDOT=0.2 \
TEX_INPAINT_ITERS=8 TEX_TELEA_INPAINT=1 TEX_TELEA_RADIUS=3 \
TEX_FINAL_SIZE=2048 TEX_BASE_BLUR=0.4 \
./tex_reproject 2048 codex_nativecumesh_500_strictsnap2048_telea_blur04.glb
```

Result: rejected only about 0.16-0.19% of covered texels. User still saw the same visible cracks/triangles.
Conclusion: suspect-sample rejection alone is not the quality gap.

### 2. Flat material / roughness diagnostic

Goal: test whether model-viewer lighting or metallic/roughness maps were amplifying triangle artifacts.

Temporary `GLB_FLAT_MR=1` was added and tested, then removed before commit.

Result: flat material did not remove the visible damage. The issue is mostly baseColor/UV/bake, not PBR
roughness/lighting.

### 3. Native bridge clusterer vs independent CPU clusterer

Goal: determine whether `ATL_NATIVE_CUMESH=1` cluster generation was causing the native outputs to converge
to the same broken look.

Generated:

```bash
ATL_PAD=4 CUMESH_TARGET=500000 \
RP_NO_BACKFALLBACK=1 RP_MAXDIST_VOX=3 RP_FRONTDOT=0.2 \
TEX_INPAINT_ITERS=8 TEX_TELEA_INPAINT=1 TEX_TELEA_RADIUS=3 \
TEX_FINAL_SIZE=2048 TEX_BASE_BLUR=0.4 \
./tex_reproject 2048 codex_cpucluster_500_strictsnap_telea_blur04.glb
```

Log shape:

```text
cumesh-cpu precluster: ~54.5k xatlas meshes
atlas ~5440x5440, charts ~78k, out ~575k verts / ~490k tris
resize final textures: ~5440x5440 -> 2048x2048
```

Result: not enough. It looked materially different in some offline crops but the user still sees A/C quality
as bad compared to B. Conclusion: native bridge clusterer is not the sole blocker.

### 4. Pack-to-target experiments

Hypothesis: huge native atlas then final resize causes cross-island bleed and triangle artifacts.

Added `ATL_FIT_RES=1` to repeatedly repack xatlas charts at lower texel density until target resolution.

Findings:

- With `ATL_PAD=4`, xatlas needed 4 sub-atlases at 2048 even at very low density. Our GLB writer emits only
  one baseColor texture, so multi-atlas output is unusable unless the writer learns texture arrays/multiple
  materials or the packer is constrained differently.
- With `ATL_PAD=0 ATL_NO_BLOCK_ALIGN=1 ATL_FIT_RES=1`, xatlas produced one 2048 atlas, but coverage collapsed
  to about 7%. Output was smaller (`28 MB`) but likely too soft/low-detail and still not a real match.

Generated:

```bash
ATL_FIT_RES=1 ATL_PAD=0 ATL_NO_BLOCK_ALIGN=1 CUMESH_TARGET=500000 \
RP_NO_BACKFALLBACK=1 RP_MAXDIST_VOX=3 RP_FRONTDOT=0.2 \
TEX_INPAINT_ITERS=8 TEX_TELEA_INPAINT=1 TEX_TELEA_RADIUS=3 \
TEX_BASE_BLUR=0.4 \
./tex_reproject 2048 codex_cpucluster_500_pad0_fitres_strictsnap_telea_blur04.glb
```

Result: useful diagnostic, not the quality target.

### 5. Texture extraction and alpha/background inspection

Used Python only as diagnostic tooling:

```python
import trimesh
scene = trimesh.load("file.glb", force="scene")
mat = list(scene.geometry.values())[0].visual.material
mat.baseColorTexture.save("file.baseColor.png")
```

Important measurement:

- B (`miku_500k.glb`) baseColor alpha0: about `6489`
- native CPU-cluster no-fill baseColor alpha0: about `101363`
- pad0 fit-res baseColor alpha0: about `3277140`
- fill-background candidate alpha0: `0`

This led to the `TEX_FILL_BACKGROUND=1` source change.

### 6. Fill remaining atlas background before resize

Hypothesis: even after local gutter inpaint, unused atlas texels remain black/transparent. Model-viewer
bilinear/mipmap sampling pulls those bad background texels into tiny UV islands, making the "micro cracks",
"burnt face", and small triangles more visible. Python/cumesh's output appears much more fully filled.

Generated:

```bash
TEX_FILL_BACKGROUND=1 ATL_PAD=4 CUMESH_TARGET=500000 \
RP_NO_BACKFALLBACK=1 RP_MAXDIST_VOX=3 RP_FRONTDOT=0.2 \
TEX_INPAINT_ITERS=8 TEX_TELEA_INPAINT=1 TEX_TELEA_RADIUS=3 \
TEX_BASE_BLUR=0.4 \
./tex_reproject 2048 codex_cpucluster_500_fillbg_strictsnap_telea_blur04.glb
```

Log shape:

```text
atlas 5440x5440, charts=78601, sub-atlases=1
rasterized: 4216319 / 29593600 texels covered (14.2%)
interior holes: 7583 (0.18% of covered)
filling remaining atlas background by nearest valid dilation
resize final textures: 5440x5440 -> 2048x2048
wrote codex_cpucluster_500_fillbg_strictsnap_telea_blur04.glb
```

Texture alpha improved:

```text
miku_500k.glb                                     alpha0 6489
codex_cpucluster_500_fillbg_strictsnap_telea...   alpha0 0
```

Result: offline crop looked cleaner, especially unlit/baseColor, but user says A is still bad in model-viewer.
Conclusion: background fill is a real fix but not the core parity gap.

## What not to repeat

- Do not keep flipping blur, Telea radius, front-dot, max distance, or material factors as the main plan.
  Those were tested and do not get to B.
- Do not trust `_mv_render.py` crops as final quality. They can miss or understate the model-viewer defects.
- Do not assume B has only a few UV islands. A quick rendered-asset UV diagnostic estimated B has even more
  UV islands than native:

```text
miku_500k.glb                              approx UV islands 182025
codex_cpucluster_500_fill/pad/no-fill       approx UV islands 78000
```

That means "too many islands" is not by itself the explanation. The Python path survives many islands because
its parameterization, packing, dilation/inpaint, and texture filtering behavior collectively avoid visible
damage.

## Likely remaining quality gap

The native path is still skirting around Python behavior. To reach B, focus on direct parity with Python's
actual bake, not more knobs.

Most likely deltas:

1. Python/cumesh UV unwrap is not just clustering. It uses its own `uv_unwrap` pipeline:
   `remove_degenerate_faces()` -> `compute_charts()` -> add each chart to `cumesh.xatlas.Atlas` ->
   xatlas compute/pack defaults -> output vertices/faces/uvs.

2. Python texture bake uses nvdiffrast coverage and cv2 inpaint. Native uses a custom CPU UV rasterizer,
   bounded Telea approximation, and then optional nearest-background fill. The raster coverage/inpaint
   semantics are still not equivalent.

3. Python's successful visual result may be hiding micro seams through color consistency/smoothing more than
   through a fundamentally different mesh. User explicitly observed that Python is not perfect when zoomed,
   but its color consistency hides defects much better.

4. Native final resize from a ~5440 atlas to 2048 may still be blending too much across chart boundaries,
   even after background fill. Python may pack directly to 2048 or use different filtering/inpaint order.

5. Native simplification may still differ enough from Python cumesh simplification around mouth/hair. User's
   repeated "flower mouth", "spikey", "little triangles" feedback suggests geometry and texture both matter,
   but the latest strong signal was color/texture making similar mesh defects look much worse.

## Best next steps

### Step 1 - Build an actual Python parity harness

Stop comparing only final screenshots. Export from B and native:

- baseColor PNG
- metallicRoughness PNG
- vertices/faces/uvs
- per-face UV area and tiny-triangle stats
- atlas background/mask/valid coverage
- mip-level previews of baseColor, especially around face/mouth/hair islands

Then compare native bake stages against Python stages. The goal is to identify where the bad colors enter:

- UV coordinates/packing
- raster coverage
- dense snap/reproject
- volume sampling
- inpaint/dilation
- resize/filtering
- GLB material/output

### Step 2 - Reproduce Python bake order exactly in C++

Use Python only as oracle. Implement native equivalents:

- xatlas pack defaults matching Python where relevant: padding 0, blockAlign false, compute chart defaults
- target-resolution behavior without post-resize if Python does not post-resize
- OpenCV Telea-like inpaint semantics on the same invalid mask, not a bounded gutter-only approximation
- nvdiffrast-style conservative raster/coverage rules if native raster differs

### Step 3 - Stop relying on `TEX_FINAL_SIZE` shrink after huge atlas

The native 500k CPU-cluster path naturally wants ~5440x5440 with padding 4. Resizing that to 2048 is probably
not equivalent to Python and can smear islands. Either:

- pack natively at 2048 in one atlas with acceptable coverage, or
- write true multi-atlas/multi-material GLB support, or
- reduce chart overhead while preserving Python-like color quality.

### Step 4 - Investigate Python/cumesh simplification parity

The target B was made from a 500k cumesh simplification. Native `CUMESH_TARGET=500000` is a C++ bridge/native
implementation, not proven bit/parity equivalent. Compare geometry directly:

- face count and vertex count
- mouth/hair connected components
- skinny triangles and triangle aspect histograms
- silhouette and local curvature around mouth/hair

If the mouth is geometrically different before texturing, no bake fix will fully solve "flower mouth".

### Step 5 - Only after B parity, shrink

Once native visually matches B at ~500k/2048:

- add quantized GLB output (`KHR_mesh_quantization`)
- consider `EXT_meshopt_compression`
- try 1024 textures and lower face budgets
- possibly drop/constant metallicRoughness if visually safe

Do not optimize size before quality parity; it hides the real deltas.

## Useful files

- `tools/m1_ref/cpp_port/tex_atlas.hpp` - unwrap, raster, inpaint, background fill, texture resize
- `tools/m1_ref/cpp_port/tex_reproject.hpp` - dense snap/reproject controls
- `tools/m1_ref/cpp_port/tex_reproject.cpp` - offline bake harness
- `tools/m1_ref/cpp_port/glb_textured.hpp` - textured GLB writer
- `tools/m1_ref/cpp_port/bake_uv.py` - Python/cumesh reference path; oracle only
- `tools/m1_ref/cpp_port/compare.html` - model-viewer comparison page; gitignored

## Build / generate commands

Build:

```bash
cd /home/dbrain/dev/longcat-sparse-spike/tools/m1_ref/cpp_port
./build.sh tex_reproject cuda
```

Latest A generation:

```bash
TEX_FILL_BACKGROUND=1 ATL_PAD=4 CUMESH_TARGET=500000 \
RP_NO_BACKFALLBACK=1 RP_MAXDIST_VOX=3 RP_FRONTDOT=0.2 \
TEX_INPAINT_ITERS=8 TEX_TELEA_INPAINT=1 TEX_TELEA_RADIUS=3 \
TEX_BASE_BLUR=0.4 \
./tex_reproject 2048 codex_cpucluster_500_fillbg_strictsnap_telea_blur04.glb
```

Offline render helper, useful only as a coarse check:

```bash
PY=/mnt/hdd/3d/avatar-shootout/Pixal3D/.venv/bin/python
$PY _mv_render.py in.glb out.png
AMB=1.0 LINT=2.5 $PY _mv_render.py in.glb out.png
UNLIT=1 $PY _mv_render.py in.glb out.png
RES=1024x1440 YAWS=0,45,90 $PY _mv_render.py in.glb out.png
```

## Current honest status

Native C++ is closer than before in implementation coverage, and `TEX_FILL_BACKGROUND=1` is a legitimate
texture-filtering fix, but quality is still not B. The next pass should be a parity/debugging pass against
Python's actual bake stages, not another sweep of blur/reprojection/inpaint parameters.
