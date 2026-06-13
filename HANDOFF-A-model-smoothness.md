# HANDOFF A — Pixal3D: NATIVE-C++ crack-free texture + game-cheap model

**For a fresh agent (e.g. Codex) with zero prior context. Self-contained: build, run, iterate, judge,
the exact blocker, the leads, and how to drive the compare page.**

---
## 0. MISSION (one sentence)
Produce the textured Miku/turtle model at **cumesh quality** (smooth, crack-free, colour-accurate to the
SOURCE IMAGE) but via a **100% native C++/ggml** pipeline (no Python in production), and **compress it to a
cheap game asset (~5-10 MB)**. The look is already achieved by a Python reference; your job is to make it
native and small.

### HARD CONSTRAINT — native C++ only
The whole point of this port is to replace the proprietary Python stack (cumesh / flex_gemm / torch) with
self-contained C++/ggml. Python (`bake_uv.py`, `*_render.py`, pyref) is a **REFERENCE/ORACLE only** — you
may RUN it to see the target or reverse-engineer it, but it must NOT be in the production path. No
`pip install`, no Python sidecar as the answer. If a step is genuinely blocked in C++, say so — don't
quietly keep the Python.

---
## 1. TL;DR — what's done, what's the ONE blocker
- **DONE (native C++):** image→geometry chain (`pixal3d`), the crisp 200k QEM mesh, the PBR-volume decode,
  and the texture-bake MATH (rasterize per-texel 3D position → snap onto the dense shell → sample the PBR
  volume → sRGB baseColor + metallicRoughness). All in `tex_atlas.hpp` / `tex_reproject.*` /
  `glb_textured.hpp`.
- **THE BLOCKER (the only thing keeping it on Python):** a **crack-free conformal UV unwrap** of the 200k
  mesh, fast. Two native attempts FAIL (judge in model-viewer, NOT pyrender — pyrender hides seam cracks):
  - C++ xatlas `ComputeCharts` from scratch (`NO_PRECL=1`): ~37k fragmented charts → **UV-seam CRACKS all
    over** + slow (173 s) + bloated 4276² atlas.
  - C++ "precluster" (normal-cone region-grow + **planar** projection, `tex_atlas.hpp precluster_charts`):
    folds where the surface curves → teal-interior + sliver **STREAKS**.
- **HOW cumesh (the Python reference) wins — REVERSE-ENGINEERED THIS SESSION:** its `uv_unwrap` does its
  own **GPU "fast clustering"** (`Get 61933 clusters after fast clustering`) into ~62k clusters, feeds
  those to xatlas which **parameterizes + packs** them → **6 final atlas charts, crack-free, ~5 s**. So
  the secret is **good clustering + a proper (conformal, NOT planar) per-cluster UV**, then xatlas pack.
  Same xatlas you have in C++ (`thirdparty/xatlas`) — the gap is the clustering + parameterization.

## 2. Environment, build, run
- Repo `/home/dbrain/dev/longcat-sparse-spike`, branch `spike/sparse-conv-3d`. Work dir
  `tools/m1_ref/cpp_port/`. C++ builds fine on this host.
- Build: `cd tools/m1_ref/cpp_port && ./build.sh <name> cuda` (CUDA) / `./build.sh <name>` (CPU).
  `pixal3d` = full chain; `tex_reproject` = offline bake harness (links bundled `thirdparty/xatlas.cpp` +
  `thirdparty/meshoptimizer/`).
- Weights `/mnt/hdd/pixal3d/weights_gguf_f16` (`--model`). Always env `NVIDIA_TF32_OVERRIDE=0`; `--fast` +
  `PIXAL3D_FLASH=1` = the validated f16 path.
- Test assets (both DEFAULT camera): Miku `--image ../../sparse_spike/golden_stages/pre/preprocessed.png`;
  turtle (heavy, ~520 s, dense mesh ~9 M f) `--image prep_test_matte.png`.
- Python venv (REFERENCE ONLY): `/mnt/hdd/3d/avatar-shootout/Pixal3D/.venv/bin/python`.
- Commits: source as you go; `.gitignore` excludes `*.glb *.ply *.png *.bin`; `compare*.html` gitignored.
  **No `Co-Authored-By`/AI mentions in commit messages.**
- Long GPU runs: harness-tracked background from your MAIN loop, NOT a trailing `&` inside the command (it
  un-tracks the job). Sweep `pgrep -x pixal3d` before handoff; leave no strays (idle GPU baseline ~247 MiB).

## 3. The iterate-fast loop (NO GPU chain re-run needed)
The chain dumps the aligned mesh + PBR volume once; iterate the bake on the dump.
```
# (one time) chain -> dump_*.bin  (Miku ~184s).  PIXAL3D_FORCE_UVATLAS makes it emit the PBR volume +
# the dense mesh; PIXAL3D_DUMP_BAKE writes the dump.
NVIDIA_TF32_OVERRIDE=0 PIXAL3D_FLASH=1 PIXAL3D_FORCE_UVATLAS=1 PIXAL3D_DUMP_BAKE=1 \
  ./pixal3d --model /mnt/hdd/pixal3d/weights_gguf_f16 \
  --image ../../sparse_spike/golden_stages/pre/preprocessed.png --out miku.glb --remesh --tex --fast
# dump files: dump_mesh_{v,f}.bin = 200k QEM mesh (int64 faces);  dump_dense_{v,f}.bin = full dense mesh;
#   dump_pbr_{f,c}.bin = per-voxel 6-ch PBR volume + int32 [b,x,y,z] coords;  dump_bake.txt/dump_dense.txt = dims.
# The Miku dump is BACKED UP as miku_*.bin:  cp miku_*.bin dump_*.bin  to restore without re-running.
# (Currently the dump on disk is Miku.)

# native C++ bake harness (reads dump_*.bin):
./build.sh tex_reproject
NO_PRECL=1 ./tex_reproject 2048 out.glb     # real xatlas unwrap (SLOW, CRACKS) -- the failing native path
./tex_reproject 2048 out.glb                # precluster planar (FOLDS/STREAKS) -- the other failing native path
# env: QEM_TARGET=<faces> (feature-preserving QEM-decimate the dense mesh, e.g. 500000),
#      ATL_CONE=<deg> (precluster cone), RP_FRONTDOT=<dot> (front-face reject), RP_ATTR=1 (mesh-attr vs snap+volume).
```
The bake entry point is `texatlas::bake(...)` in `tex_atlas.hpp` (rasterize + snap-to-dense reproject +
sample + inpaint + emit). Snap logic in `tex_reproject.hpp`. Add your new unwrap + the compression here,
then wire into `pixal3d --tex` (drop the Python).

## 4. The REFERENCE you must match (Python, look only)
`bake_uv.py` is the cumesh sidecar that makes the crack-free result. Run it to SEE the target / inspect
cumesh:
```
/mnt/hdd/3d/avatar-shootout/Pixal3D/.venv/bin/python bake_uv.py --dump . --mesh qem --texsize 2048 --out ref.glb
#   --ssaa 2 / --blur 0.8 = texture smoothing;  --decimate 500000 = cumesh-simplify the dense mesh first
#   (this is how "B" = the 500k smoother-silhouette model was made — the owner's preferred look).
```
Its pipeline (== Trellis2 postprocess): `cumesh.uv_unwrap` → nvdiffrast rasterize → flex_gemm
`grid_sample_3d` → cv2 inpaint → PBR glTF. **The crack-free magic = cumesh's GPU fast-clustering before
xatlas (§1).** cumesh API of note: `simplify(target_num_faces)`, `repair_non_manifold_edges`,
`unify_face_orientations`, `remove_degenerate_faces`, `uv_unwrap(compute_charts_kwargs=...,
xatlas_compute_charts_kwargs=..., xatlas_pack_charts_kwargs=...)`. You may inspect these to reverse-engineer
the clustering, then implement equivalently in C++.

## 5. TASK 1 — native crack-free unwrap (the blocker). Leads, best first
1. **Replicate cumesh's clustering + give each cluster a conformal UV, then xatlas-pack.** The C++
   precluster already region-grows clusters + feeds xatlas via `AddUvMesh` — but it uses a **planar**
   projection that FOLDS. Replace the planar UV with a per-cluster **conformal/LSCM** parameterization
   (xatlas can compute charts if you `AddMesh` the clustered mesh and let `ComputeCharts` parameterize,
   seeded so it's fast; or implement a small LSCM per cluster). Tighter, non-folding clusters → few clean
   charts → crack-free, like cumesh.
2. **Manifold-repair the mesh, THEN xatlas ComputeCharts.** The 37k-chart fragmentation comes from the
   mesh's non-manifold edges. Weld coincident verts + `repair_non_manifold_edges`-equivalent +
   `unify_face_orientations`-equivalent (bundled `meshoptimizer` has some primitives; may need your own
   weld) so xatlas sees a clean mesh → few charts. (Note: the QEM mesh has 0 boundary edges + 32 connected
   components but is non-manifold; a naive `make_manifold` tried earlier added +253k boundary → worse —
   needs a smarter repair.)
3. **Seam-aware texture dilation (cheap stopgap).** Even with many charts, aggressively pad/dilate each
   chart's gutter in the atlas (bump `padding` + inpaint iters in `tex_atlas.hpp`) so model-viewer's
   bilinear/mipmap can't sample across seams → may hide the cracks without a better unwrap.
4. **A GPU/better unwrapper in C++ (owner wants GPU for speed).** Honest landscape: standard UV unwrappers
   are CPU (xatlas, Microsoft UVAtlas, Boundary-First-Flattening, OptCuts). True GPU UV-unwrap is rare/
   research. cumesh's speed comes from GPU *clustering* + CPU xatlas pack, not a GPU xatlas — so the
   fastest realistic native route is **GPU clustering (lead 1) + the bundled CPU xatlas pack**, not finding
   a drop-in GPU library. Evaluate before committing to porting anything heavy.
- Downstream of the unwrap is DONE: `tex_atlas.hpp bake(reproject=true)` (snap+volume) + `glb_textured.hpp`
  (sRGB baseColor + metalRough). Only the unwrap is blocking.

## 6. TASK 2 — compress to a cheap game asset (native C++, straightforward)
"B" (the smooth 500k model) is 44 MB = ~32 MB mesh (815k verts × f32 pos+normal+uv + u32 indices) + ~12 MB
textures (two 2048² PNGs). Native levers (bundled `thirdparty/meshoptimizer` — model-viewer & game engines
decode all of these natively):
- **KHR_mesh_quantization**: `meshopt_quantize*` — positions→int16 (with a node scale/translation),
  UVs→unorm16, normals→int8/oct. ~3-4× smaller mesh. Emit from `glb_textured.hpp`/`glb_writer.hpp` (they
  already hand-write glTF JSON + binary).
- **EXT_meshopt_compression**: `meshopt_encodeVertexBuffer` + `meshopt_encodeIndexBuffer`. Stacks with
  quantization → mesh to a few MB.
- **Smaller textures**: 1024² instead of 2048² (the bake `texsize`); check if metallic/roughness are near-
  uniform → drop the metalRough texture for scalar factors.
- **Right-size the mesh**: find the smallest face budget whose silhouette still reads smooth in model-viewer
  (200k cracks-aside is nearly as smooth as 500k at body scale; 500k only wins on the thin hair-tail
  silhouette). `PIXAL3D_REMESH_FACES` (chain) or `QEM_TARGET` (bake harness) tune it.
- Target: ~5-10 MB, loads + renders cheap in model-viewer, look ≈ B.

## 7. The compare page — how to drive it (this is how the owner judges)
`compare.html` (Miku) + `compare_turtle.html` (turtle) load GLBs in `<model-viewer>` (three.js, "neutral"
IBL, exposure 1.0) — the real game-engine-like view. **Always judge texture quality HERE, not in pyrender.**
- Serve: `cd tools/m1_ref/cpp_port && python3 -m http.server 8011` (Miku) and `... 8012` (turtle). Open
  `http://<host>:8011/compare.html` and `:8012/compare_turtle.html`. Hard-reload (Ctrl+Shift+R) after
  changing a GLB (browser caches by filename — use new filenames or hard-reload).
- Point the 3 cells at your GLBs: edit the `const SRC = { tex: { a:'fileA.glb', b:'fileB.glb',
  c:'fileC.glb', capA:'…', capB:'…', capC:'…' }, geo:{…} }` object near the bottom `<script>` of the html.
  `a`/`b`/`c` are the three side-by-side model-viewer cells (cameras auto-sync); `capA/B/C` are the HTML
  captions above each. There's a "Textured" tab and an "Untextured geometry" tab (`setMode('tex'|'geo')`).
  GLBs must sit in the served dir (`tools/m1_ref/cpp_port/`).
- Compare against the SOURCE images: `preprocessed.png` (Miku) / `prep_test_matte.png` (turtle).
- Quick offline sanity render (crutch only): `python3 _mv_render.py <glb> <out.png>` (bright even IBL-ish).
  It does NOT show UV-seam cracks — model-viewer does. Trust model-viewer.

## 8. Files map
- `pixal3d_chain.hpp` — the chain; `PIXAL3D_DUMP_BAKE` block dumps the QEM + dense mesh + PBR volume.
- `tex_atlas.hpp` — `bake(...)` (raster + reproject + sample + inpaint + atlas), `precluster_charts(...)`
  (the folding planar clustering — replace its UV), xatlas calls. **Your unwrap + compression land here.**
- `tex_reproject.hpp/.cpp` — snap-to-dense closest-point (DenseHash) + the offline bake harness.
- `tex_grid_sample.hpp` — PBR-volume trilinear sample. `glb_textured.hpp`/`glb_writer.hpp` — glTF writers
  (extend for quantization/compression). `qem.hpp` — feature-preserving decimation.
- `thirdparty/xatlas.{h,cpp}`, `thirdparty/meshoptimizer/` — bundled, native.
- `bake_uv.py` — the Python cumesh REFERENCE (to delete once native lands).

## 9. DEAD ENDS — measured, do not repeat
- **Judge texture in pyrender** → hides UV-seam cracks (the whole reason "C++ xatlas looked fine" was wrong).
  Use model-viewer.
- **C++ precluster planar projection** → folds → teal-interior + sliver streaks. Tighter cones (25-55°)
  don't fix it. Needs conformal per-cluster UV.
- **C++ xatlas ComputeCharts from scratch on the non-manifold mesh** → 37k charts → cracks + 173 s + bloated atlas.
- **Per-vertex colour (COLOR_0)** for a web mesh → "cracks"/facet-net unless ~1.5 M verts; and COLOR_0 is
  LINEAR in glTF → washed/too-bright. Use a baseColor TEXTURE (sRGB).
- **Sampling the PBR volume at the raw QEM texel position** (no snap) → hits the teal interior. Keep the
  snap-to-dense + volume sample (already implemented).
- **pyref is a defective reference** — it has teal-interior holes; target the SOURCE IMAGE.

## 10. DEFINITION OF DONE
A GLB that, loaded in model-viewer, looks ~as smooth + colour-accurate as the cumesh "B" (judge vs the
source image — smooth, NO cracks, correct colours), at a game-cheap size (~5-10 MB), produced by a fully
**native C++** pipeline (no Python in the path). Validate on BOTH Miku and turtle; update both compare
pages. Don't declare "smooth/crack-free/small" without looking in model-viewer.

See also `HANDOFF-B-perf-infra.md` (separate track: Miku sub-4 GB VRAM, host-CPU poles, de-python the
front-end). `FINDINGS-15` / `PERF-NOTES-pixal3d.md` = history.
