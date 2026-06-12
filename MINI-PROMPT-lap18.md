Continue the C++/ggml Pixal3D (TRELLIS.2) port — FULL AUTONOMY, owner intermittently around and gives
sharp feedback (judge by RENDER vs pyref, NEVER false-"perfect" — he caught two false "clean"s last
lap). GPU+CPU free. C++ builds fine on this host. Long GPU runs: `run_in_background:true` from the MAIN
loop (sub-agents DEADLOCK on bg completion). No pkill -f / rm-globs; sweep `pgrep -x pixal3d` before any
handoff; leave no strays.

## THE JOB (owner's #1): make the TEXTURE QUALITY perfect. Shape is already DONE (crisp QEM remesh).
End of lap-17 the textured Miku (per-vertex COLOR_0, `miku_qem_vcolor.glb`, compare.html tex tab A) is
"a lot better but colours WASHED OUT and still CRACKY like she was smashed porcelain and glued back
together." Both are inherent to per-vertex-colour-on-a-faceted-mesh: averaging through the QEM collapses
DESATURATES, per-vertex colour is DISCONTINUOUS at facet edges → cracks, and the thin twin-tails get
dark back-of-hair patches. **THE FIX = BVH-reproject the QEM mesh's UV texels onto the DENSE dual-grid
mesh** (which has exact, smooth, full-saturation per-vertex PBR — it IS pyref's colouring) **→ bake a
real UV PBR TEXTURE** (baseColor + metallic/roughness), NOT per-vertex colour. Use the fast precluster
atlas for layout; closest-POINT-on-triangle (not nearest-vertex — it jumps thin gaps); front-face reject
for the thin tails; verify sRGB colour space (the "wash"). Prototype OFFLINE with ZERO GPU runs: a
same-run aligned dump already exists (`dump_*.bin` via `PIXAL3D_DUMP_BAKE`) → iterate in `tex_bake_dump`
+ render vs `pyref_front.png`. (Golden stage4-PBR-on-stage5-mesh offline tests are MISALIGNED → false
teal; only use same-run dump data.)

## RUN BOTH ASSETS + TWO COMPARE PAGES (owner: "two things to look at")
Validate every change on BOTH and surface each on its own page:
- Miku: input `tools/sparse_spike/golden_stages/pre/preprocessed.png` (default cam) → compare.html on
  `:8011`.
- Turtle (HEAVY: N1=3605 / 9M f / M3b ~169s — slow+high-VRAM is INHERENT): input
  `tools/m1_ref/cpp_port/prep_test_matte.png` → make a SECOND page `compare_turtle.html` on `:8012`.

## CONCURRENT (both need E2E reruns anyway):
- **VRAM: sub-4GB Miku AND sub-10GB turtle.** Miku peak 6021 MiB at the stage-5 1024² conv im2col;
  turtle peak NOT yet measured (>5.6GB) — measure it. Instrument which op is the spike (DINOv3@1024
  attn vs NAF@1024 im2col, `cudaMemGetInfo` per-phase — the poll label lagged), then TILE the @1024
  conv spatially (overlap + 1px reflect halo, im2col+matmul per tile, concat) — helps both — or flash
  DINOv3@1024. Validate N1==1120 + mesh IoU + render.
- **Atlas/CPU poles (owner: "atlas still seems CPU bound").** Precluster made xatlas ~2s but PackCharts,
  the tex BAKE raster + grid_sample, the new BVH-reproject, and **M4 mesh-extract (~19.8s HOST)** are
  all CPU. Move the bake raster + grid_sample + BVH + M4 extract to GPU. Owner saw it on the turtle.

READ IN FULL FIRST: `HANDOFF-NEXT-lap18-texture-quality.md` (plan + every dead-end + the offline-aligned
iter harness), then FINDINGS-15 §1c/§5/§6/§6b, PERF-NOTES LAP 17, memory `project_3dgen_cpp_port`,
`feedback_correctness_before_perf`, `reference_subagent_background_stall`, `reference_ncu_docker_syadmin`.
Build `cd tools/m1_ref/cpp_port && ./build.sh <t> cuda`; weights `/mnt/hdd/pixal3d/weights_gguf_f16`;
`--fast` + `PIXAL3D_FLASH=1`; `NVIDIA_TF32_OVERRIDE=0`. Branch `spike/sparse-conv-3d`, commit source as
you go, NO Co-Authored-By (cpp_port/.gitignore excludes glb/ply/png/bin; compare.html is gitignored).

DoD: textured E2E front-on vs pyref = SATURATED, CONTINUOUS (no cracks/patches) for BOTH Miku and
turtle, heels/fingers/chin still crisp; two compare pages live; Miku <4GB, turtle <10GB. Judge by
render; don't declare "perfect" early.
