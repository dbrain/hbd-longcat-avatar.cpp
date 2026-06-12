# HANDOFF — Pixal3D lap-18: get TEXTURE QUALITY perfect (owner's #1), then sub-4GB VRAM

**Owner's words (end of lap-17):** shape is great now; texture "a lot better but colours are WASHED
OUT and it's still CRACKY like she was smashed porcelain and glued back together. let's get quality
perfect before going deep elsewhere (or concurrent with perf as both need reruns)." So: **TEXTURE
QUALITY is the job.** Perf/VRAM (sub-4GB) can run concurrently since both need E2E reruns.

Run FULLY AUTONOMOUS (owner intermittently around, gives sharp feedback — judge by RENDER vs pyref, do
NOT declare "perfect" early; the owner caught two false "clean"s last lap). GPU+CPU free. C++ builds
fine on this host; long GPU runs `run_in_background:true` from the MAIN loop (sub-agents deadlock on
bg completion); no `pkill -f`/rm-globs; sweep `pgrep -x pixal3d` before handoff.

## READ FIRST (these ARE the resume state — lap-17 banked everything as it went)
- `FINDINGS-15-remesh-conditioning-perf.md` §1c (QEM crispness — DONE), §5 (precluster atlas), **§6 +
  §6b (the texture saga + every dead-end)**.
- `PERF-NOTES-pixal3d.md` LAP 17 (VRAM re-profile: peak 6021 MiB at the 1024² stage).
- memory `project_3dgen_cpp_port` (lap-17 + "LAP 17 TEXTURE" entries), `feedback_correctness_before_perf`,
  `reference_subagent_background_stall`, `feedback_no_build_on_server` (C++ OK), `reference_ncu_docker_syadmin`.

## WHERE WE ARE (end of lap-17) — shape DONE, texture "clean but not perfect"
`pixal3d --remesh --tex --fast` (+`PIXAL3D_FLASH=1`) → crisp ~200k-f QEM mesh (heels/fingers/chin kept)
+ **per-vertex COLOR_0** (`miku_qem_vcolor.glb`, ~5MB). E2E 184s (N1=1120, M=4633). compare.html on
`:8011` (tex tab A = miku_qem_vcolor.glb, geo tab A = miku_qem_geo.glb). Render front-on vs pyref =
`ours_vs_pyref.png`: recognisable clean Miku, BUT washed + cracky (see below).

**What's solid (don't redo):**
- **QEM remesh** (`qem.hpp`, non-manifold-tolerant Garland-Heckbert): 3.25M→200k in ~3s, sharp features
  kept. Wired `pixal3d_chain.hpp` step 7b. `PIXAL3D_REMESH_FACES` (0=raw mesh), `PIXAL3D_REMESH_AGGR`.
- **colour-CARRY through QEM** (`qem.hpp Vert.col`, `qem_simplify(in_col,out_col)`): colour the DENSE
  on-shell mesh, average through collapses. Chain defers QEM to the texture branch (`remesh_deferred`).
- **Fast atlas precluster** (`tex_atlas.hpp precluster_charts` + AddUvMesh): ~2s vs >180s xatlas hang.
  Currently UNUSED for remesh (per-vertex won) but kept for the UV path; `PIXAL3D_FORCE_UVATLAS`.

## THE JOB — fix WASHED-OUT + CRACKY. Both are inherent to per-vertex-colour-on-a-faceted-mesh.
**Diagnosis (lap-17):**
- **WASHED OUT** = (a) the colour-carry AVERAGES colours through collapses → desaturates teal↔dark
  blends; (b) suspected COLOR_0 linear-vs-sRGB mismatch (glTF COLOR_0 is linear; pyref bakes an sRGB
  baseColor texture — verify the colour space; a missing sRGB encode washes everything).
- **CRACKY / smashed-porcelain** = per-vertex colour is DISCONTINUOUS at the QEM facet edges (flat
  facets with per-vertex colour read as cracks), + dark patches on the thin twin-tails (QEM merges a
  tail vert with a spatially-near dark back/interior vert → averaged dark).

**THE FIX (scoped, the proper SOTA path — Python's approach): BVH-REPROJECT onto the DENSE mesh →
bake a real UV PBR TEXTURE.** Per-vertex colour is the wrong representation; a 2K UV texture is smooth
(no facet cracks), full-saturation (no averaging), and carries metallic/roughness. Plan:
1. Rebuild the DENSE dual-grid mesh at step 8 (fast, `flexible_dual_grid_to_mesh`); colour each dense
   vertex EXACTLY (it sits on the shell → trilinear hits its voxel; or 1:1 zip). This is the clean,
   full-saturation, hole-free source-of-truth (it IS pyref's colouring).
2. Build a BVH over the dense mesh triangles (or a uniform spatial-hash grid at grid-1024 — the mesh
   is dense so a hash grid + point-triangle distance over the 3×3×3 neighbourhood works; AVOID
   nearest-VERTEX, it jumps across thin gaps — use closest-POINT-on-triangle).
3. For the QEM mesh's UV atlas (use the fast precluster, it's fine for layout): for each TEXEL's
   rasterised 3D position, BVH-closest-point on the dense mesh → barycentric-interp the dense
   per-vertex PBR (6-ch). This stays on the correct surface (kills the interior-teal AND the
   back-of-tail dark) and is smooth across facets (kills the cracks). Output baseColor + metalRough.
   → This is the per-texel analogue of what made per-vertex clean, but on a continuous UV texture.
   - front-face awareness for the thin tails: if closest-point picks a triangle whose normal opposes
     the QEM surface normal, reject it (pick the same-facing candidate) — stops back-of-hair bleed.
4. Verify colour space: encode baseColor to sRGB if the glTF expects it (compare a flat patch's RGB to
   pyref's texture at the same point). Fixes "washed out".

**ITERATE OFFLINE, NO GPU (seconds), with ALIGNED data** — this is critical, lap-17 burned time on
MISALIGNED golden data:
- `pixal3d ... PIXAL3D_DUMP_BAKE=1` dumps the same-run mesh+PBR → `dump_*.bin` + `dump_bake.txt`.
- `tex_bake_dump` (`./build.sh tex_bake_dump`) loads them; iterate the BVH-reproject bake there +
  render. `front_render.py <glb> <out>` (off-axis lit) + the COLOR_0 manual-GLB-parse render snippet
  (pyrender mis-reads `glb::write_glb` COLOR_0 as TextureVisuals — parse the GLB chunks by hand, see
  the lap-17 transcript / `glb_vcolor_render.py`). Compare vs `pyref_front.png` / `miku_uvatlas_pyref.glb`.
- A dump already exists (`dump_*.bin`, Miku, N1=1120) — start there, no GPU run needed to prototype the
  reproject. Only run the full E2E once the offline bake looks perfect.

**DEAD ENDS (do NOT repeat — all measured in §6/§6b):** UV-atlas-on-QEM (xatlas hangs on non-manifold;
precluster charts FOLD → interior teal); make_manifold→xatlas (nm 53769→0 but +253k boundary, still
slow); aggressive ChartOptions (timed out); per-vertex nearest-voxel (zombie noise); per-vertex
trilinear without carry (noise); naive planar precluster wide cone (sliver streaks). The golden
stage4-PBR-on-stage5-mesh offline tests are MISALIGNED → false teal; ALWAYS use same-run dump data.

## RUN BOTH Miku AND turtle, with TWO compare pages (owner wants two things to eyeball)
Every texture/perf change must be validated on BOTH assets and surfaced on its OWN compare page:
- Miku: input `tools/sparse_spike/golden_stages/pre/preprocessed.png` (default cam). compare.html
  (`:8011`) tex/geo tab A → `miku_qem_*.glb`.
- Turtle (the HEAVY asset, N1=3605 / 9M f / M3b ~169s — slow + high-VRAM is INHERENT, not a bug):
  input `tools/m1_ref/cpp_port/prep_test_matte.png`. **Spin up a SECOND page `compare_turtle.html`**
  (copy compare.html, point at `turtle_qem_*.glb`, serve on a 2nd port e.g. `:8012`). Owner: "running
  for Miku and spinning up an extra compare page for turtle so I have two things to look at."
- Render both front-on vs their refs each iteration; the texture fix + perf must hold on the turtle too.

## CONCURRENT: VRAM — sub-4GB Miku AND sub-10GB turtle (owner #2)
Owner: "vram wise it should be sub 4 GB for Miku but [sub] 10 for turtle as well ideally." Miku peak
**6021 MiB at the stage-5 1024² conv im2col** (baseline ~3.5GB; PERF-NOTES LAP 17). **Turtle NOT yet
peak-measured** — an 8s poll caught 5611 MiB at M3b but the true peak (denser im2col/attention) is
higher; MEASURE it (tight poll or per-phase cudaMemGetInfo) and drive it <10GB. FIRST instrument which
op is the 1024² spike (DINOv3@1024 attention vs NAF@1024 im2col — add `cudaMemGetInfo` per-phase; the
poll label lagged). Then: tile the @1024 conv spatially (overlapping tiles + 1px reflect halo for k3,
im2col+matmul per tile, concat) — this helps BOTH assets — OR wire flash to DINOv3@1024. Validate
N1==1120 + mesh IoU + render. Next tier after the spike: DINOv3@512 4185 + SS DiT 4007.

## CONCURRENT: kill the remaining HOST-CPU poles (owner: "atlas still seems CPU bound")
The fast precluster made xatlas ~2s, but several poles are STILL host CPU and dominate the non-DiT
wall (and the turtle's atlas was minutes at 296k f). Move to GPU:
- **xatlas PackCharts** is CPU (~2s Miku, worse on turtle) — precluster already skips the slow
  ComputeCharts; if the UV-PBR path needs it faster, GPU the pack or cap chart count.
- **Tex BAKE raster + grid_sample** (`tex_atlas.hpp` CPU triangle rasterizer + `tex_grid_sample.hpp`):
  the per-texel 3D-pos raster + volume sample is CPU — nvdiffrast-style GPU raster + GPU grid_sample.
  The lap-18 BVH-reproject (closest-point on the dense mesh per texel) should be done ON GPU too
  (it's the new heavy per-texel op).
- **M4 mesh-extract ~19.8s HOST** (`build_nmap` hashmap-per-level + host coord arithmetic) — GPU/Morton
  neighbour-map. Flagged since lap-15, still open; it's a big chunk of the turtle's CPU tail.
Measure the per-stage wall on BOTH assets (the chain prints per-stage `(%.1fs)`); the host-CPU poles
are M4-extract + bake-raster + pack. Owner cares because the turtle made them visible.

## ALSO QUEUED (lower priority, after texture)
- MoGe camera as a warm Python host SERVICE (owner #3, kill the last per-call Python; mirror
  `kobbler docker/matting` rmbg pattern; `estimate_camera.py` math is validated). NOT the PNG→GLB API
  (owner deferred that to post-rigging).
- SS DiT flash (still dense, 42s, N1-sensitive — validate N1==1120 before trusting).

## DoD (lap-18): textured E2E front-on vs pyref shows SATURATED, CONTINUOUS (no cracks/patches) colour
matching Python — heels/fingers/chin still crisp. Then sub-4GB. Judge by render; don't false-"perfect".
Build: `cd tools/m1_ref/cpp_port && ./build.sh <t> cuda`. Weights `/mnt/hdd/pixal3d/weights_gguf_f16`.
Miku input = `tools/sparse_spike/golden_stages/pre/preprocessed.png` (default cam = Miku). Turtle =
`tools/m1_ref/cpp_port/prep_test_matte.png` (heavy). Branch `spike/sparse-conv-3d`, commit as you go
(source only; cpp_port/.gitignore excludes glb/ply/png/bin; compare.html is gitignored — dev viewer).
EOF
echo "wrote handoff"