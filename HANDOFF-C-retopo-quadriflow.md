# HANDOFF C — auto quad-retopo (QuadriFlow) → clean low-res textures + animation-ready topology

Continues HANDOFF-B (the in-process meshopt+KTX2 packer + bake quality, both DONE). This is the **next
real quality lever** and the prerequisite for the project's end goal (**rig + animate**, mute 3D game
assets). Pure scope — no code written yet.

## Why this exists (read first — it reframes "texture quality" as "topology")

The bake reached Python parity at 4096² (HANDOFF-B). But the mesh unwraps into **~182k tiny charts**
(hair strands / thin parts) because the geometry is **non-manifold** (cumesh `simplify` emits F≈4V; the
marching-tet `--remesh` path also goes non-manifold after QEM). Consequences, all one root cause:

1. **2048 texture seams** — owner verdict 2026-06-14: 4096/UASTC is "the cleanest model I've seen", but
   2048 (downsampled) has "teal seeping through" hair-chart seams; ETC1S 2048 is worse. With 182k charts
   there's no texel budget for gutters wide enough to survive a downsample. Bake-param tuning CANNOT fix
   it (more downsample = more cross-chart averaging = worse). See HANDOFF-B "Packed-tier visual verdict".
2. **No clean low-poly LODs** — mesh decimation explodes charts to 12-27k → seam soup at 512/1024.
3. **Bad deformation under animation** — triangle-soup / non-manifold meshes pinch and collapse at joints
   no matter how good the skin weights are. **The rigger (SkinTokens, [[project_avatar_rig_path]]) is
   geometry-AGNOSTIC** — it will RUN on any topology — but **deformation QUALITY is gated by topology**.

**All three are fixed by one upstream change: replace the 182k-chart non-manifold mesh with a manifold,
quad-dominant, field-aligned mesh that unwraps into a few hundred BIG charts.** That is auto quad-retopo.
It sits **upstream of BOTH** the texture bake (UVs bake onto the retopo'd mesh) **and** SkinTokens (which
skins the retopo'd mesh). Net-new stage — the TRELLIS.2/Pixal3D Python pipeline has NO retopo (only
`fill_holes` + cumesh `simplify` + `to_glb`), so there is **no oracle to match** here; the bar is the
visual quality gate (model-viewer) + clean charts, not bit-parity.

## Tool decision: QuadriFlow (recommended) vs Instant Meshes

Both implement the same "Instant Field-Aligned Meshes" family (comparable quality). For an automated,
headless, C++-first pipeline:

| | QuadriFlow (hjwdzh/QuadriFlow) | Instant Meshes (wjakob/instant-meshes) |
|---|---|---|
| License | **MIT** (commercial OK) | BSD-style (commercial OK — Modo ships it) |
| Headless | **YES — CLI/lib, no GUI** | GUI app (NanoGUI + OpenGL); core is separable but coupled |
| Deps | Boost+Lemon (maxflow) OR `BUILD_FREE_LICENSE`→SparseLU (MPL2, slower, **no Boost**); Eigen; pcg32 | NanoGUI, TBB, OpenGL, Eigen |
| Determinism | seeded pcg32 (`-seed`); min-cost-flow `-mcf` = robust | seeded |
| Fit | **automation pick** | needs core extraction to go headless |

**⇒ Use QuadriFlow.** MIT, headless, minimal deps (build with `BUILD_FREE_LICENSE` to drop the Boost
graph dep if Boost is a hassle on the toolchain). Instant Meshes only if QuadriFlow quality disappoints.
CLI shape: `quadriflow -i in.obj -o out.obj -f <target_quad_faces> [-mcf] [-sharp] [-seed N]`.

## Where it slots (the pipeline change)

    pixal3d geometry  →  manifold mesh  →  QuadriFlow retopo  →  triangulate  →  ┌─ xatlas ComputeCharts → bake (tex_atlas) → meshopt+KTX2 pack
    (dense / --remesh)   (the QF input)    (quad-dominant)      (for glTF)      └─ SkinTokens rig → animate
                                                                                  (both consume the SAME retopo'd mesh)

- **QuadriFlow input** = a reasonably clean MANIFOLD mesh. Feed the **marching-tet `--remesh`** output
  (already exists, provably manifold/watertight) or the dense shell post-`fill_holes` — NOT the raw
  182k-chart soup (QF wants manifold-ish input). The marching-tet manifold mesh is the natural feed.
- **Output** = quad-dominant mesh at a target face budget (e.g. 20-80k). glTF is triangles, so
  **triangulate** (each quad→2 tris) — the triangulation FOLLOWS the quad edges, so the field-aligned
  edge flow (the thing that makes it deform + unwrap well) is preserved.
- **Then the existing downstream just works on a better input**: real xatlas `ComputeCharts` (NOT the
  182k-chart precluster) → hundreds of big charts → `tex_atlas` grid_sample bake onto the new UVs →
  the HANDOFF-B `glb_packed` meshopt+KTX2 pack. Clean 2048 AND clean 512/1024 LODs fall out.

## Implementation ladder (R0–R6)

- **R0 — vendor + build QuadriFlow headless.** Clone hjwdzh/QuadriFlow into `thirdparty/` (gitignore +
  bootstrap-on-demand like `build_basisu.sh`, pinned ref). Build the lib/CLI with the toolchain g++
  (C++ builds fine on this host). Try `BUILD_FREE_LICENSE=ON` first to avoid Boost. Smoke-test: feed a
  test OBJ, get a quad OBJ at target faces. DELIVERABLE: `quadriflow` runs headless here.
- **R1 — wire geometry → QuadriFlow.** Feed pixal3d's manifold mesh (marching-tet `--remesh`, or dense
  post-fill_holes) into QuadriFlow at a target budget; get the quad mesh back; triangulate for glTF.
  Either shell out to the CLI (fast to wire) or link the lib (cleaner; do this once it works). Add a
  `--retopo <faces>` flag to pixal3d. DELIVERABLE: pixal3d emits a field-aligned-topology GLB.
- **R2 — re-unwrap + bake on the retopo'd mesh = THE PROOF.** Run real xatlas `ComputeCharts` on the
  quad-derived mesh (drop the precluster path for this mesh). Confirm **chart count collapses to
  hundreds** (not 182k). Bake the PBR volume onto the new UVs (reuse `tex_atlas` grid_sample/raster/
  inpaint). **Bake at 2048 and confirm the teal hair-chart seams are GONE** (the HANDOFF-B failure).
  This is the success gate — validate in model-viewer + compare.html.
- **R3 — LOD tiers.** Generate 512/1024 cleanly (QuadriFlow at lower target, or manifold-preserving
  meshopt decimate of the quad mesh). The whole point: low-res is now clean because charts are big.
- **R4 — integrate + pack.** Wire `--retopo` into the chain + the in-process `--pack` (HANDOFF-B). One
  command: image → retopo'd, textured, compressed, multi-LOD GLB.
- **R5 — rig handoff.** Feed the retopo'd mesh to SkinTokens (geometry-agnostic, [[project_avatar_rig_path]])
  → verify it skins quad-derived topology cleanly. Pose-test deformation at a joint vs the old soup mesh
  (the actual animation-quality win). SkinTokens port is a separate effort (HANDOFF-RIGGING-skintokens.md);
  this rung just confirms the topology feeds it well.
- **R6 — determinism + perf + edges.** Seed pcg32; handle QF failure/degenerate inputs; perf (QF on a
  decimated input is fast; don't feed 3M faces raw). Validate the ladder end-to-end.

## Reuse inventory (most of the downstream already exists)

- `tex_atlas.hpp` — xatlas unwrap + CPU raster + `tex_grid_sample.hpp` volume bake + TELEA inpaint. Feed
  it the retopo'd mesh; the `ATL_PYREF_XATLAS` real-ComputeCharts path already exists (it's the precluster
  that we stop using for the clean manifold input).
- `glb_packed.hpp` + `ktx2_encode.hpp` + `build_basisu.sh` — the HANDOFF-B in-process meshopt+KTX2 packer.
- `thirdparty/{xatlas,meshoptimizer}` vendored; `thirdparty/cumesh_native` (marching-tet remesh source).
- pixal3d `--remesh` (marching-tet manifold) = the QuadriFlow input source.
- **Validation harness**: `gltf_validator -o file.glb` (0 errors), `meshopt_verify file.glb` (decodes
  meshopt streams — catches what the validator can't), `compare.html` :8011 (model-viewer A/B/C tabs).
  See HANDOFF-B for the four model-viewer/glTF gotchas already found + fixed.

## Risks / gotchas

- **QuadriFlow input must be manifold-ish** — feed the marching-tet `--remesh` mesh, NOT the 182k-chart
  soup. If QF chokes, clean/weld first.
- **Quad→tri for glTF** — triangulate along quad diagonals; keep the edge flow. Don't re-decimate after
  (that re-breaks it — same lesson as the bake's `bake_deci=0` for the remesh path).
- **Boost dep** — prefer `BUILD_FREE_LICENSE=ON` (SparseLU, MPL2) to skip the Boost maxflow graph; only
  fall back to Boost if SparseLU is too slow on these meshes.
- **Determinism** — seed pcg32 so the same image → same topology (matters for a reproducible pipeline).
- **It's net-new, no oracle** — judge by the quality gate (clean 2048, charts in the hundreds, clean
  joint deformation), not bit-parity. Get owner eyeball in model-viewer at R2 (the seam-kill proof).
- **License is clean** (MIT) — resolves the [[project_avatar_rig_path]] commercial-use flag for this stage.
- Don't re-litigate dead levers from FINDINGS-A (normal_offset/dilate, bigger-atlas-more-downsample).

## Success criteria (Definition of Done)
1. `pixal3d --retopo <N> --tex --pack` → a GLB with **hundreds** of charts (not 182k), field-aligned
   quad-derived topology.
2. **Clean 2048** (and 512/1024 LODs) — teal hair-chart seams GONE, owner-confirmed in model-viewer.
3. `gltf_validator` 0 errors + `meshopt_verify` all streams decode.
4. Retopo'd mesh skins + deforms cleanly under SkinTokens (R5 spot-check).
5. Deterministic (seeded), headless, C++ (no python/node), MIT-clean.

## Key files
- This doc + `MINI-PROMPT-retopo.md` (agent kickoff) + HANDOFF-B (packer/bake context) +
  FINDINGS-A (bake recipe + dead levers) + HANDOFF-RIGGING-skintokens.md (the rig stage downstream).
- `tools/m1_ref/cpp_port/` — pixal3d.cpp, tex_atlas.hpp, glb_packed.hpp, ktx2_encode.hpp, build.sh,
  build_basisu.sh (the bootstrap pattern to copy for QuadriFlow), meshopt_verify.cpp, compare.html.
