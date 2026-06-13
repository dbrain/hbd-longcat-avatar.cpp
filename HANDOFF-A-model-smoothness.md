# HANDOFF A — Pixal3D texture: smooth + game-cheap model (for a fresh agent, no prior context)

You are continuing a C++/ggml port of **Pixal3D** (TRELLIS.2): image → textured 3D GLB. Geometry + the
texture *look* are DONE. **Your job: make the textured model both SMOOTH and CHEAP — a usable game asset
(small file, cheap to render) that looks as close to the SOURCE IMAGE as possible.** Right now the
smoothest version is too heavy (44 MB). Shrink it without losing the look, and do it in **NATIVE C++**
(see "Hard constraint" below).

---
## 0. Hard constraint — NATIVE C++ ONLY
The whole point of this port is to replace the proprietary Python stack (cumesh / flex_gemm / torch) with
self-contained **C++/ggml**. Python (`bake_uv.py`, `*_render.py`, pyref) is a **reference/oracle only** —
NOT the production path. Do not solve anything with a Python sidecar or `pip install`. If a step is hard
in C++, say so and ask — don't quietly fall back to Python.

---
## 1. Where things live + how to build
- Repo: `/home/dbrain/dev/longcat-sparse-spike`, branch `spike/sparse-conv-3d`. Work dir:
  `tools/m1_ref/cpp_port/`.
- Build a target: `cd tools/m1_ref/cpp_port && ./build.sh <name> cuda` (CUDA) or `./build.sh <name>` (CPU).
  - `pixal3d` = the full image→GLB chain (CUDA). `tex_reproject` = the offline texture-bake harness (CPU,
    links bundled `thirdparty/xatlas.cpp` + `thirdparty/meshoptimizer/`).
- Weights: `/mnt/hdd/pixal3d/weights_gguf_f16` (pass as `--model`). Env: always `NVIDIA_TF32_OVERRIDE=0`;
  `--fast` + `PIXAL3D_FLASH=1` for the f16 perf path.
- `.gitignore` excludes `*.glb *.ply *.png *.bin`; `compare*.html` are gitignored (dev viewers). Commit
  source as you go; **no `Co-Authored-By`/AI mentions in commit messages** (global rule).
- Two test assets (both use the DEFAULT camera, just a different image):
  - **Miku**: `--image ../../sparse_spike/golden_stages/pre/preprocessed.png`
  - **Turtle** (heavy: N1=3605, ~520 s E2E, dense mesh ~9 M faces): `--image prep_test_matte.png`

## 2. How to run the chain + get a texture bake
The chain dumps the aligned mesh+PBR-volume; the texture is baked from that dump.
```
# 1) chain → dump (writes dump_*.bin: QEM mesh + dense mesh + PBR volume; ~184s Miku / 520s turtle)
NVIDIA_TF32_OVERRIDE=0 PIXAL3D_FLASH=1 PIXAL3D_FORCE_UVATLAS=1 PIXAL3D_DUMP_BAKE=1 \
  ./pixal3d --model /mnt/hdd/pixal3d/weights_gguf_f16 \
  --image ../../sparse_spike/golden_stages/pre/preprocessed.png \
  --out miku.glb --remesh --tex --fast
# (dump files: dump_mesh_{v,f}.bin = 200k QEM mesh; dump_dense_{v,f}.bin = full dense mesh;
#  dump_pbr_{f,c}.bin = the per-voxel 6-ch PBR volume; dump_bake.txt / dump_dense.txt = dims.
#  Miku dump is backed up as miku_*.bin — `cp miku_*.bin dump_*.bin` to restore without re-running.)
```
Run the chain harness-tracked in the background from your MAIN loop (`run_in_background:true`); do NOT
detach with a trailing `&` inside the command (it un-tracks the job). Sweep `pgrep -x pixal3d` before any
handoff; never leave strays. `nvidia-smi` shows ~247 MiB idle baseline.

## 3. How to JUDGE (use the compare pages — model-viewer, the real renderer)
`compare.html` + `compare_turtle.html` load the GLBs in `<model-viewer>` (three.js, "neutral" IBL,
exposure 1.0) — this is the real game-engine-like view. Serve + open:
```
cd tools/m1_ref/cpp_port && python3 -m http.server 8011   # Miku  -> http://<host>:8011/compare.html
                            python3 -m http.server 8012   # turtle-> http://<host>:8012/compare_turtle.html
```
Each page has 3 synced cells (A/B/C); edit the `SRC` map near the bottom of the html to point cells at
your GLBs. **Judge SMOOTHNESS + COLOUR against the SOURCE IMAGE** (`preprocessed.png` / `prep_test_matte.png`),
NOT against pyref. Quick offline sanity renders exist (`_mv_render.py <glb> <out>` = bright even IBL-ish
pyrender) but they're only a crutch — **the owner judges in model-viewer**; pyrender at low ambient is
misleadingly dark/harsh and over-reads artifacts.

## 4. CURRENT STATE — texture look is SOLVED; size is the open problem
The texture bake produces a smooth, correctly-coloured UV PBR texture with NO teal-interior holes. The
pipeline: **conformal UV unwrap of the crisp 200k mesh** → rasterize per-texel 3D position → sample the
PBR volume → sRGB baseColor + metallicRoughness textures. Validated on Miku + turtle vs their source
images (compare pages live).

**Two ways the unwrap has been done — same quality, very different speed:**
- **C++ xatlas** (bundled `thirdparty/xatlas`, real `ComputeCharts`): SMOOTH + correct (verified render).
  But SLOW: ~173 s on the 200k QEM, >20 min on the dense mesh. This is the NATIVE path and its quality is
  fine — speed is the only issue (and it's offline asset-gen, so slow may be acceptable). Run it via
  `tex_reproject` with `NO_PRECL=1` (real unwrap) — see §6.
- **cumesh** (Python/GPU xatlas, `bake_uv.py`): same look, ~5 s. **This is the REFERENCE only — it is
  Python and must NOT be the production path.** It exists so you can see the target quality fast.

**The size problem — the smoothest version (B) is too heavy:**
- Owner picked **B** = a finer **500k-face** mesh (smoother hair-tail silhouette than 200k) → **44 MB**.
- Breakdown of B's 44 MB: **~32 MB mesh** (815k verts × float32 position+normal+uv + uint32 indices —
  vert count inflated by UV-seam duplication) + **~12 MB textures** (two 2048² PNGs: baseColor + metalRough).
- At body scale 200k ≈ 500k (both smooth); the ONLY visible win from 500k is the thin hair-tail
  **silhouette** (mesh edges). So the cost (extra verts) buys silhouette smoothness only.

## 5. YOUR GOAL — small, cheap-to-render, still smooth, close to the source
Make B-quality (or close) into a **cheap game asset**: small file, low-ish poly, smooth, source-accurate.
Native-C++ levers (all doable with the bundled libs — NO Python, NO new binaries):
1. **Mesh quantization → KHR_mesh_quantization.** `thirdparty/meshoptimizer` has `meshopt_quantize*`
   (positions→int16 with a node scale/translation, UVs→unorm16, normals→int8/oct). model-viewer decodes
   it natively. Cuts the 32 MB mesh ~3-4×. Implement in `glb_textured.hpp`/`glb_writer.hpp` (they already
   hand-write glTF JSON + binary).
2. **Mesh compression → EXT_meshopt_compression.** `meshopt_encodeVertexBuffer` + `meshopt_encodeIndexBuffer`
   (bundled). model-viewer supports it. Stacks with quantization → mesh can drop to a few MB.
3. **Fewer verts for the same silhouette.** 815k verts for 500k tris = heavy UV-seam duplication. Fewer
   charts → fewer seams → fewer verts. Or adaptive decimation (more faces only where the silhouette needs
   them — the tails — fewer on flat areas). The feature-preserving QEM is in `qem.hpp`
   (`PIXAL3D_REMESH_FACES` tunes the chain's budget).
4. **Smaller textures.** 2048² is overkill for a game model — try 1024² (`--texsize 1024` in the harness;
   the C++ bake uses `texsize`). Consider whether metalRough can be a scalar factor instead of a texture
   (check if metallic/roughness are near-uniform in the volume). Optional later: KTX2/Basis (heavier).
5. **Right-size the mesh budget.** Find the smallest face count whose silhouette still reads smooth in
   model-viewer (B=500k was "best"; maybe 300-350k + quantization is the sweet spot).

**Definition of done:** a GLB that loads in model-viewer looking ~as smooth + colour-accurate as B
(judge vs the source image), at a game-reasonable size (target ~5-10 MB, cheap to render), produced by a
**native C++** path. Validate on BOTH Miku and turtle. Don't declare "smooth/small" without looking in
model-viewer (the owner caught false "clean"s before — judge by render).

## 6. The offline bake harness (`tex_reproject`) — your fast iteration loop, NO GPU chain re-run
`./build.sh tex_reproject` then run in `tools/m1_ref/cpp_port/` (reads `dump_*.bin`):
```
NO_PRECL=1 ./tex_reproject 2048 out.glb     # real C++ xatlas unwrap (SLOW ~173s) + snap+volume bake
./tex_reproject 2048 out.glb                # FAST C++ precluster atlas — DEAD END (folds → streaks); see below
QEM_TARGET=500000 NO_PRECL=1 ./tex_reproject 2048 out.glb   # feature-preserving QEM to 500k, then unwrap
```
Env knobs: `NO_PRECL` (real xatlas vs precluster), `QEM_TARGET=<faces>` (QEM-decimate the dense mesh,
feature-preserving), `TEX_MESH_FACES=<faces>` (sloppy decimate), `RP_FRONTDOT` (front-face reject),
`ATL_CONE` (precluster cone), `RP_ATTR=1` (barycentric dense attr instead of snap+volume). The bake logic
is `tex_atlas.hpp::bake(...)` (snap-to-dense reproject in `tex_reproject.hpp`); add the
quantization/compression output here first, then wire into `pixal3d` proper.

## 7. DEAD ENDS — do not repeat (all measured this session)
- **C++ precluster planar atlas** (`tex_atlas.hpp` precluster path, `ATL_CONE`): FOLDS where the surface
  curves → teal-interior bleed + sliver STREAKS. This was the whole "streaky" problem. Tighter cones
  (25-55°) don't fix it. Use a real conformal unwrap (xatlas `ComputeCharts`), not the precluster.
- **Per-vertex colour (COLOR_0)** for a web mesh: needs ~1.5 M verts to not "crack" (facet
  discontinuities); 400-800k still shows a facet "net". Dead for small meshes. Use a UV texture.
- **COLOR_0 in general**: it's LINEAR in glTF (skips the sRGB decode) → renders washed/too-bright. A
  baseColor TEXTURE is sRGB → correct. (This was the lap-17 "washed out" complaint.)
- **Sampling the PBR volume at the raw QEM texel position**: the 200k QEM surface is slightly off the
  voxel shell → hits the teal interior. Fix = snap the texel onto the DENSE mesh (closest-point-on-tri,
  front-face reject) then sample the volume there (`tex_reproject.hpp` + `tex_atlas.hpp` reproject mode).
- **pyref is a defect-bearing reference**: it (Python cumesh bake) HAS little teal-interior holes; our
  snap+volume avoids them. Target the SOURCE IMAGE, not pyref.

## 8. Pointers
- `HANDOFF-B-perf-infra.md` — the separate VRAM/perf/de-Python-the-bake task.
- `FINDINGS-15-remesh-conditioning-perf.md` §1c/§5/§6 — the geometry remesh + texture saga history.
- `PERF-NOTES-pixal3d.md` — timing/VRAM laps.
- Source of truth for the look: the SOURCE images + the compare pages in model-viewer.
