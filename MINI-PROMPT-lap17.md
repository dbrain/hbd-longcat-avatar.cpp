Continue the C++/ggml Pixal3D (TRELLIS.2) port — FULL AUTOMATION, get it DONE (owner away, GPU+CPU
free). Decide, do, golden-validate, document, continue. Only halt if hard-stuck.

## THE JOB THIS LAP: make the remeshed/decimated model CRISP — "Python quality, ideally better".
The owner's words across the last lap: the current remesh "rounds ALL the corners — heel spikes become
nubs, fingers fuse into mittens, chin/neck merge — it's gone PS4 → PS2." The ideal: **"all shapes kept,
just fewer 3D-pixels so it's cheap to render."** That is *feature-preserving simplification*. The last
agent (me) chased atlas-speed and flash-attn and declared those "done" while THIS — the actual point —
came out softer. Do NOT repeat that. **Nothing else matters until the geometry render visibly keeps the
sharp features (heel spikes, fingers, chin gap) at a web-sized poly count.**

WHY it's soft: the current pipeline (`remesh.hpp marching_cubes_solid`) SMOOTHS then decimates —
solidify + box-blur field + Taubin + coarse marching-cubes. MC interpolates the iso-crossing → rounds
hard edges by construction; the blur/Taubin round more. We NEVER actually tried QEM. The base mesh is
genuinely massive (~3.25M tris) so simplification IS needed — but with the RIGHT tool:
- **PRIMARY: QEM (quadric error metric) decimation on a SHARP MANIFOLD mesh.** QEM provably preserves
  the silhouette + sharp edges (collapses only flat regions). The crisp source is the FULL dual-grid
  O-Voxel mesh, but it's non-manifold + ~24k holes → meshopt falls to `simplifySloppy` (rounds). So:
  weld + `orient_consistent` (we have it) the full mesh JUST enough that meshopt QUALITY (`meshopt_simplify`,
  NOT Sloppy) engages → decimate to ~150-300k → crisp + low-poly. `meshopt_simplifyWithAttributes`
  (pass normals) helps lock features. THIS is the owner's "all shapes kept, fewer polys".
- **IF QEM-on-dual-grid can't get manifold cheaply: DUAL CONTOURING** (what Python/cumesh does) — places
  verts via the QEF to RECONSTRUCT sharp edges MC can't → then QEM. Bigger build; do it if (A) stalls.
- Do NOT pre-smooth the source before QEM (that's the bug). Validate by RENDER vs the crisp target.

FAST LOOP (no E2E, no GPU, seconds): `tools/m1_ref/cpp_port/iter_remesh.sh <stride> <blur> <smooth>`
renders GEOMETRY → geo_<tag>.png. Compare vs `miku_golden_geo.png` (the full 3.25M-tri crisp target).
Add a QEM path to `remesh_test.cpp` and iterate THERE until the heel spikes / fingers survive at ~150k
faces. Only run the full E2E (textured) once the geometry is crisp.

DEFINITION OF DONE (be honest — do not declare victory early): the textured E2E render, side-by-side
vs `miku_uvatlas_pyref_render.png`, shows the heel spikes pointed (not nubs), fingers separated (not a
mitten), chin/neck distinct — at a web-sized poly budget. If it's not there, it's not done.

## ONLY AFTER crispness genuinely matches pyref — then the rest (see HANDOFF for detail):
- Fast atlas: xatlas ComputeCharts 30-42s CPU pole → normal-cone chart pre-cluster (→ PackCharts only)
  + move bake raster/grid_sample + M4 mesh-extract (19.8s host) to GPU. Validate the heavy TURTLE asset
  E2E (owner never saw its output).
- Perf/VRAM re-profile (ncu/nsys/compute-sanitizer work under docker). SS DiT (42s) + tex DiT (23.5s)
  still DENSE → wire flash (V-scale fix already in attention()) but VALIDATE N1==1120 + mesh IoU.
- Kill last Python: MoGe-2 camera as a warm host service (mirror the rmbg matting service).
- PNG→GLB API: wrap pixal3d_chain.hpp as a native server (worker-isolation, idle-unload true-0, REST).

READ IN FULL FIRST: HANDOFF-NEXT-lap17-atlas-perf-api.md (P0 = this crispness work, with both routes),
FINDINGS-15-remesh-conditioning-perf.md (§1b remesh state + why it rounds, §4 flash-attn), PERF-NOTES-
pixal3d.md. Memory: project_3dgen_cpp_port, reference_ncu_docker_syadmin, feedback_correctness_before_perf,
reference_subagent_background_stall, feedback_no_build_on_server, feedback_clean_up_background_procs.

Build: cd tools/m1_ref/cpp_port && ./build.sh <t> cuda. Weights /mnt/hdd/pixal3d (symlinked); --fast =
perf config (+weights_gguf_f16/); flash = PIXAL3D_FLASH=1; NVIDIA_TF32_OVERRIDE=0. Render venv =
/mnt/hdd/3d/avatar-shootout/Pixal3D/.venv/bin/python. Long jobs run_in_background:true from the MAIN loop
(sub-agents deadlock on bg completion); chain build&&run, track+kill PIDs, no pkill -f / rm-globs.
compare.html on :8011. Keep docs/memory current. Worktree UNCOMMITTED — leave it.
