# HANDOFF B — texture bake quality (next steps) + remaining Python→C++ ports

Continues `FINDINGS-A-resolution-cumesh-bake.md` (READ IT — full bake recipe, every diagnosed fix,
and the dead-ends). This handoff = what's left.

## Where we are (2026-06-14)

**The native C++ texture bake reached parity with Python.** `./tex_reproject` (pure C++/CUDA — the
cumesh dep is `thirdparty/cumesh_native`, linked into the binary, NOT Python) matches the Python
`bake_uv.py` oracle at 4096². The full recipe + the 5 diagnosed fixes (topo normals, RP_OFF direct
volume sample, ATL_PYREF_XATLAS opts, TEX_FBR=0, TEX_RASTER_SS=2, no-resize+fill) are in FINDINGS-A.

User verdict: full-res native (v6, 4096²) == Python ("realistically what we'd hit; very slightly worse,
close enough"). This is the ceiling for THIS pipeline.

### compare.html state (gitignored — record for next session)
Served via `cd tools/m1_ref/cpp_port && python3 -m http.server 8011 --bind 0.0.0.0` → :8011/compare.html
- **Tab "Native C++ vs Python"**: A=`native_v6.glb` (native 4096, the parity bake) · B=`miku_500k.glb`
  (Python ref) · C=`pyref_500k_4096.glb` (Python 4096). A≈C.
- **Tab "Game-asset"**: A=`native_2048.glb` (full mesh, tex downscaled to 2048 — has resize cracks,
  see below) · B=`native_v6.glb` (4096, clean, 86 MB) · C=`native_lod2_1024.glb` (decimated 70k mesh,
  the LOD wall).

## The game-asset wall (don't re-discover this)

- **Texture downscale of the FULL mesh** is *mostly* fine, BUT the in-loop resize of the 182k-chart
  atlas down to 2048 (2.3×) half-merges adjacent tiny charts → dark cracks (that's why `native_2048`
  looks cracky while `native_v6` @4096 is clean). To downscale cleanly, bake from a BIGGER atlas so the
  downsample ratio is ≥4× (strong AA) — e.g. raise TEX_RASTER_SS and/or bake the atlas larger, then
  TEX_FINAL_SIZE=2048. Untested (ran out of GPU time). This is the cheapest "clean 2048" path.
- **MESH decimation breaks the UV**: cumesh `simplify_to_faces` (== Python `cm.simplify`) emits
  NON-MANIFOLD geometry (F≈4V; e.g. 5939 V / 23935 F). xatlas must cut a chart at every non-manifold
  edge → 12k–27k charts even with ATL_MAXCOST=100000 → low-res = seam soup. **A clean low-poly LOD is
  not achievable from this pipeline** without a retopo step (below). Python hits the identical wall.

## Next quality steps

### 1. KTX2 + mesh quantization — DONE, FULLY IN-PROCESS (2026-06-14)
Shipped as a NATIVE C++ in-process packer (user wanted zero off-process node/python; gltfpack-the-binary
is C++ but still a subprocess, so we vendored the libs instead):
- `pixal3d --tex --pack [hero|small]` writes a compressed GLB DIRECTLY from the in-memory atlas — no
  second process, no GLB re-parse. hero=UASTC+Zstd (near-lossless), small=ETC1S.
- Geometry: `meshoptimizer` (already vendored) — KHR_mesh_quantization (int16 pos / int8-OCTAHEDRAL
  normal / uint16 uv) + EXT_meshopt_compression (encodeVertexBuffer/IndexBuffer, fallback-buffer layout).
- Textures: vendored `thirdparty/basis_universal` → `ktx2_encode.hpp` (basis_compressor, UASTC+Zstd or
  ETC1S) → KHR_texture_basisu (KTX2). Built once into `thirdparty/basis_universal/build/libbasisu_enc.a`
  via `build_basisu.sh` (toolchain g++ 12.4, ABI-matches pixal3d's CUDA link).
- Files: `glb_packed.hpp` (writer), `ktx2_encode.hpp` (KTX2), `build_basisu.sh`. Tests: `glb_pack_test`
  (round-trip: idx/pos/nrm-oct decode + KTX2 magic — ALL VALID) and `glb_repack REPACK_INPROC=hero|small`
  (real-scale CPU validation: repacks native_v6.glb without GPU).
- Full-scale result (native_v6: 816k v / 4776² tex, 86 MB uncompressed): hero 45.9 MB, small 17.9 MB,
  both structurally valid. (gltfpack ref on the same file: 42 / 12 MB — small-tier delta is our
  quality-first ETC1S q192, tunable via the etc1s_quality param.)
- STILL TODO (GPU): run `pixal3d --tex --pack` on a real generation to confirm the live path + eyeball
  in model-viewer (CPU repack already proves the writer; this just exercises it end-to-end on GPU).
- The standalone `gltfpack` binary (~/.local/bin) + `pack_glb.sh` remain only as the A/B reference target.

### 2. Clean 2048 tier — DONE (2026-06-14)
`TEX_FINAL_SIZE=2048 TEX_RASTER_SS=1 ./tex_reproject 8192 native_clean2048.glb` → bakes atlas 8806²
then area-downsamples 4.3× to 2048 (≥4× = strong AA, no chart-merge cracks; 182k charts, 0.51% holes,
46s, no OOM at SS=1 since the big downsample IS the AA). Packed in-process: `native_clean2048_inproc.glb`
(hero/UASTC 15.1 MB) / `native_clean2048_inproc_small.glb` (small/ETC1S 9.6 MB). A/B vs the OLD cracky
2.33× `native_2048*` in model-viewer to confirm the cracks are gone.

### 3. Bake v6 flags in as binary defaults — DONE (2026-06-14)
The v6 flag-set is now the DEFAULT in `tex_reproject.cpp` main() via `setenv(k,v,0)` (env still overrides
any single flag), so the finalized command is just `./tex_reproject 4096 out.glb`. Deliberately did NOT
touch the shared `tex_atlas.hpp::bake()` defaults — that path is also used by `pixal3d --tex` (a different
unwrap), and changing it could silently regress the production binary. `TEX_KEEP_ATLAS_SIZE` is auto-
skipped when `TEX_FINAL_SIZE` is set, so the game-LOD recipe still downsamples. Compiles clean.
STILL TODO (GPU): one run to confirm the default-flag bake is byte-identical to the old env-soup native_v6.

### 4. True low-poly LODs (BIGGER — needs a new tool, only if mobile/traditional-engine target)
The bake pipeline can't produce clean low-poly LODs (non-manifold decimator, see wall). To get them:
- Add a **manifold retopo + clean re-unwrap** pass after geometry: Instant Meshes (quad retopo),
  or a DCC remesher, or `pymeshlab` isotropic remesh → then xatlas unwraps into FEW big charts → bake
  the texture onto THAT. This is the standard game path and the only way to ~5–20 MB low-poly assets.
- Alternative for high-end targets: **UE5 Nanite** eats the dense 490k mesh as-is (auto-LOD), so no
  retopo needed there — full mesh @ 2048 + KTX2 is shippable for Nanite/high-end. Retopo is only for
  mobile / traditional LOD chains.

## Remaining Python → C++ ports (production path)

The CORE chain is native C++/ggml: `pixal3d --image matte.png --cam ... --tex --out x.glb` (geometry +
PBR volume), and now the texture BAKE (`tex_reproject`). bake_uv.py and the `_bake_*`/`render_*`/
`*_zoom`/compare_render py are now ORACLE/eval-only — they do NOT need porting.

Two INPUT front-ends are still Python (both are deliberate "host cut-lines" for arbitrary-photo upload;
both have a documented bypass so the C++ path runs without them on controlled inputs):

1. **`estimate_camera.py` — the "camera angle thing" (MoGe-2 FOV/distance).** Runs MoGe-2
   (`Ruicheng/moge-2-vitl`, a ViT-L monocular-geometry PyTorch model) to predict per-image intrinsics →
   `camera_angle_x` + distance, which proj-mode conditioning needs to land image features on the right
   voxels. Wrong FOV → distorted geometry. **Bypass:** pixal3d `--fov`/`--cam` (a fixed camera works for
   training-like framing; only "upload any photo" needs the estimate). **Port = MoGe-2 ViT-L → ggml**
   (medium-large; the host math `get_camera_params_wild_moge`/`distance_from_fov` is already pure and
   mirrored in the script — only the ViT inference is unported).

2. **`preprocess_photo.py` — background removal + framing.** rembg / BiRefNet RMBG-2.0 (PyTorch) to
   matte arbitrary photos → crop to subject + square + premultiply on black (the model expects a
   centered subject on black, per its training renders). **Bypass:** feed an image that already has a
   clean alpha/matte (no rembg needed); the crop/premultiply math is trivial host CV. **Port = BiRefNet
   → ggml** (medium), or swap a lighter segmenter, or keep as a host step.

Neither blocks "model looks good" — they're the "upload literally any photo" conveniences. If the
product is "drop in a pre-matted, sensibly-framed image", the pipeline is already 100% native C++.
(Worth confirming separately that pixal3d `--tex` generates the PBR volume natively — the bake consumes
`dump_pbr_{f,c}.bin`; if those still come from a Python texture-DiT step, that's a third port. Memory
flagged Phase-B textures as the last gen stage; verify with `pixal3d --tex` provenance.)

## Key files
- `tools/m1_ref/cpp_port/tex_reproject.{cpp,hpp}` — native bake harness + reproject (RP_OFF = the path we use)
- `tools/m1_ref/cpp_port/tex_atlas.hpp` — unwrap/pack/raster/inpaint/normals/resize (all the flags)
- `tools/m1_ref/cpp_port/tex_grid_sample.hpp` — trilinear sparse-volume sampler (+ the TEX_FBR fallback)
- `tools/m1_ref/cpp_port/native_cumesh_bridge.{cpp,hpp}` — links cumesh_native (simplify + compute_charts)
- `tools/m1_ref/cpp_port/bake_uv.py` — Python ORACLE only (cumesh uv_unwrap + nvdiffrast + cv2)
- `tools/m1_ref/cpp_port/estimate_camera.py`, `preprocess_photo.py` — the two unported host front-ends
- `tools/m1_ref/cpp_port/build.sh tex_reproject cuda` — build (C++/CUDA builds fine on this host)
- `FINDINGS-A-resolution-cumesh-bake.md` — the full bake recipe + dead-ends (READ FIRST)
