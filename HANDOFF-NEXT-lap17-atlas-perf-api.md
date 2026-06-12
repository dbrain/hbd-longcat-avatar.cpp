# HANDOFF — Pixal3D lap-17: fast atlas · perf/VRAM re-profile · MoGe service · PNG→GLB API

**Run FULLY AUTONOMOUSLY (owner away). "Full automation — get it done": drive each item to a WIN, not
"ready for the next guy".** Golden-validate vs the true-fp32 oracle (judge E2E by mesh IoU / render;
loosen tol only for perf). Worktree `/home/dbrain/dev/longcat-sparse-spike` is UNCOMMITTED — leave it
so. Build: `cd tools/m1_ref/cpp_port && ./build.sh <target> cuda`; weights `/mnt/hdd/pixal3d`
(symlinked); `--fast` = f16 perf config + needs `weights_gguf_f16/`; `NVIDIA_TF32_OVERRIDE=0` default;
**long runs `run_in_background:true` from the MAIN loop** (sub-agents deadlock on bg completion); no
`pkill -f` / rm-globs (kill by PID). compare.html on `:8011` (start `python3 -m http.server 8011` in
`tools/m1_ref/cpp_port`). Render venv = `/mnt/hdd/3d/avatar-shootout/Pixal3D/.venv/bin/python`
(trimesh+pyrender; base python has neither). **READ IN FULL:** `FINDINGS-15-remesh-conditioning-perf.md`
(§1b remesh, §4 flash-attn), `PERF-NOTES-pixal3d.md` (LAP 1-5), memory `project_3dgen_cpp_port`,
`reference_ncu_docker_syadmin`, `feedback_correctness_before_perf`, `reference_subagent_background_stall`,
`feedback_no_build_on_server` (C++ cpp builds FINE here; Rust no-build).

## WHERE WE ARE (end of lap-16) — image→textured-GLB works, clean, ≈Python
`pixal3d --remesh --tex --fast` (+ `PIXAL3D_FLASH=1`) → clean watertight textured Miku, render ≈ pyref.
Per-stage (RTX 3060, Miku, flash on): DINOv3 1.3+1.7 · **SS DiT 42** · NAF 3.7+4.2 · M2 9.1 · M3a 5.6 ·
**M3b DiT 41** (flash −19%) · M4 mesh-extract **19.8 (HOST CPU)** · remesh 2.5 · tex DiT 23.5 · tex
decode 19.6 · **xatlas ComputeCharts ~30-42 (HOST CPU)**. Mesh: solidify→coarse-MC(stride4,grid256)→
consistent-winding→quality-decimate(150k)→nearest-voxel-tex-fallback(r=stride·12)+hole-inpaint. Atlas
**~48% util** (pyref=25%), ~3.3k charts. **Three remaining poles are HOST CPU: M4 extract (19.8s),
xatlas ComputeCharts (30-42s), tex bake raster.** The owner's north star: **a REST API that takes a PNG
and returns a GLB.**

---
## PRIORITY 0 — CRISPNESS: keep ALL shapes, just fewer polys (owner's #1 complaint)
**Owner: the remesh "rounds all the corners — heel spikes become nubs, going PS4→PS2". The goal is
"all shapes kept but lower poly so it's cheap to render" = FEATURE-PRESERVING decimation.** Our current
pipeline SMOOTHS then decimates (solidify + box-blur field + Taubin + coarse marching-cubes), which
rounds sharp features by construction (MC interpolates the iso-crossing; the blur/Taubin round more).
The base mesh IS massive (~3.25M tris) so remesh/decimation IS needed — but the RIGHT tool is **QEM
(quadric error metric) decimation**, which collapses flat regions while PRESERVING the silhouette +
sharp edges. Two routes (pick by render vs the golden full mesh):
- **(A) QEM on a manifold full-detail mesh.** The crisp source is the full dual-grid O-Voxel mesh, but
  it's non-manifold + ~24k holes → meshopt falls to `simplifySloppy` (which does NOT preserve features
  → the rounding). Get it manifold MINIMALLY (weld + `orient_consistent`, which we now have) so meshopt
  QUALITY (QEM) engages → low-poly, feature-preserved. OR march the SOLID at full grid 1024 (stride 1 —
  the flood-fill needs speeding up, it timed out at 120s; do it on a coarse grid then upsample, or
  bound the bbox) with blur 0 (sharp) → QEM-decimate. The winding-fix already unblocked QEM on the
  remesh; the lever is feeding QEM a SHARP manifold mesh, not a pre-smoothed one.
- **(B) Dual contouring** (what Python/cumesh does): places vertices via the QEF to RECONSTRUCT sharp
  edges (heel spikes, fingers) — MC can't. Manifold dual-contouring (or libigl/CGAL) → QEM. This is the
  principled "keep the spikes" fix; bigger implementation.
**FAST ITERATION (no E2E, no GPU): `./iter_remesh.sh <stride> <blur> <smooth>`** runs the remesh offline
on `refs/stage5/head_coords.npy` (seconds) + renders the GEOMETRY → `geo_<tag>.png`. Compare vs
`miku_golden_geo.png` (the crisp 3.25M-tri target). Sweep here BEFORE any E2E. Texture needs the chain
PBR (full E2E), but SHAPE crispness is fully visible in the geometry render. Levers: finer stride (2 vs
3 vs 4 — grid 512/341/256), blur (0=sharp staircase, 1=slight round), POST_SMOOTH (0=sharpest). Current
default is stride 3 / blur 1 / smooth 1 (best balance found this lap; still rounds hard edges — hence A/B).

## PRIORITY 1 — FAST ATLAS (the 30-42s CPU pole) + finish the bake on GPU
**The lever is NOT "xatlas on GPU" (it's CPU-only). It's normal-cone CHART PRE-CLUSTERING (what Python/
cumesh does), which turns the slow `ComputeCharts` into a fast `PackCharts`.** Region-grow the clean
(now consistent-winding!) remesh faces into ~hundreds of charts within a normal-cone half-angle, planar-
project each, hand xatlas the charts via `MeshDecl` UVs + `ChartOptions{useInputMeshUvs=true}` so it
only PACKS (PackCharts was always ~1s). Target ComputeCharts 30s → ~few s. (FINDINGS-13 "parked perf
#1" scoped this; the winding-fix makes region-grow trivial now — normals are coherent.) Validate util
stays ≥ Python's 25% and render is clean.
- **Also move the tex BAKE raster to GPU** (`tex_atlas.hpp` CPU triangle rasterizer + `tex_grid_sample.hpp`
  grid_sample). nvdiffrast-style GPU raster → per-texel 3D pos on GPU → GPU grid_sample (port the
  `texgs::sample_one` trilinear + `find_nearest` fallback to a CUDA kernel). The hole-inpaint can stay
  CPU (cheap) or move too.
- **M4 mesh-extract (19.8s HOST)** is the other CPU pole: `build_nmap` hashmap-per-level + host coord
  arithmetic. GPU/Morton neighbor-map + extraction. (Was flagged lap-15, still open.)
- **VALIDATE THE HEAVY (turtle) ASSET END-TO-END** — owner: "we tried the turtle, haven't seen output."
  `assets/images/0_img.png` → `preprocess_photo.py` → `pixal3d --remesh --tex --fast --flash`. The turtle
  is high-token (M3a Nh~1.14M → M=15313) — confirm flash (now landed) clears the old M3b OOM and the
  remesh/atlas produce a clean GLB. Render it; put it in compare.html.

## PRIORITY 2 — PERF/VRAM RE-PROFILE (ncu/nsys/vram now WORK under docker)
Post-flash, re-profile to find the new heavy hitters (`reference_ncu_docker_syadmin`: build container-
native in `longcat-avatar-dev:builder`, `ggml/build-cuda-docker`, `m3b_sampler_test_docker`; ncu needs
`--cap-add SYS_ADMIN`; nsys bundled in nsight-compute; `compute-sanitizer` @
`/mnt/hdd/3d/avatar-shootout/toolchain/bin`). Likely targets now:
- **SS DiT (42s) is the single biggest stage and is STILL DENSE** (no flash — it's occupancy→coords,
  N1=1120 must stay EXACT). Wire flash to it (the V-scale fix is in `attention()` already) BUT validate
  N1==1120 + mesh IoU before trusting; flash's 0.9991 cosine may shift a boundary voxel. If N1 holds,
  that's another big win. The fa_mask build is in `slat_dit_graph.hpp` — replicate in `ss_dit_graph.hpp`.
- **tex DiT (23.5s)** also dense → wire flash (it's CFG-off, 1 fwd/step).
- spike conv kernel (~18s decode floor, naive fp32 implicit-GEMM, no tensor cores → tf32/f16 tiled MMA;
  judge by mesh IoU). NAF@1024 im2col VRAM peak (5.9GB). Re-measure peak VRAM after the atlas changes.
- **Cleaner flash fix (drops the V-scale workaround + recovers 0.9991→0.9998):** patch the ggml-cuda
  `mma_f16` flash kernel so the PV (V-weighted) accumulator honors `GGML_PREC_F32` (currently only QK/
  softmax do — that gap is the whole reason V overflows f16; see FINDINGS-15 §4). We own the fork
  (`ggml/build-cuda`). `fa_repro.cpp` (FA_LOAD replay of captured fwd-20 q/k/v) is the fast test harness.

## PRIORITY 3 — KILL THE LAST PYTHON: MoGe camera as a warm host SERVICE
The inference path's only remaining Python is the **camera estimate** (`estimate_camera.py`, MoGe-2).
Matte is already a warm C++ service (`kobbler docker/matting`, RMBG-2.0, :8898). **Decision: ship MoGe
as a warm Python host service** (the ViT-L+DPT+focal-solve port is multi-day; MoGe has NO intrinsics
head — see FINDINGS-15 §3). Mirror the matting-service pattern: a tiny persistent server that loads
`Ruicheng/moge-2-vitl` ONCE, `POST /camera {png} -> {camera_angle_x, distance, mesh_scale}` using
`estimate_camera.py`'s already-validated math (fov 42.01°/dist 1.3022 on Miku); idle-unload to return
VRAM. (Do NOT run it concurrently with a GPU E2E — owner GPU contention rule.) `estimate_camera.py`'s
`estimate()` + `distance_from_fov()` are the reusable core.

## PRIORITY 4 — THE PNG→GLB API (the north star) — `pixal3d` is a CLI; wrap it as a server
End state: `POST /generate {png}` → (matting service: matte) → (MoGe service: camera) → in-process
pixal3d chain → GLB. Build a native server around `run_geometry`/`pixal3d_chain.hpp` (the chain is
already a library) with the koblem heavy-engine contract: **worker-isolation + idle-unload to true-0
VRAM** (mirror acestep/flux2/songgen `*-server` worker-isolation), async jobs + cancel + `/unload`,
REST + a panel. HTTP-call rmbg + MoGe over the network (no per-call Python subprocess piping — the
pattern owner endorsed). Then koblem engine wiring (off-server Rust build) is the final mile.

## QUALITY / KNOWN GAPS (judge by render)
- **Fingers still merge** at stride 4 (fingers ~4-8 voxels < a stride-4 cell). `PIXAL3D_REMESH_STRIDE=3`
  (grid 341) or 2 (grid 512) resolves them — finer march → quality-decimate keeps them; cost is more
  faces + slower remesh/unwrap (mitigated once P1 atlas is fast). Pick the default once atlas is fast.
- **Texture concavity holes** (skirt folds, underarm): the remeshed surface sits off the sparse PBR
  shell → grid_sample misses → black. Lap-16 mitigations: nearest-voxel fallback (r=stride·12) +
  interior-hole inpaint (`tex_atlas.hpp`, fills covered-but-zero texels from valid chart neighbours,
  `TEX_INPAINT_ITERS`). The PROPER fix (Python's) = **BVH-reproject**: sample the texture at the nearest
  point on the ORIGINAL dual-grid mesh, not the nearest voxel. Worth it for crisp concavities.
- flash is gated `PIXAL3D_FLASH` (opt-in); fold into `--fast` once SS/tex DiT validated + N1 confirmed.

## WHAT'S DONE THIS LAP (lap-16) — keep, build on
- **Tight-atlas remesh SOLVED** (FINDINGS-15 §1b): solidify+coarse-MC+`orient_consistent`+quality-
  decimate+nearest-voxel-fallback+hole-inpaint. 840s→13-42s, util 16.6%→48-55%, render ≈ pyref.
  `remesh.hpp` (`marching_cubes_solid`, `orient_consistent`), `tex_grid_sample.hpp` (`find_nearest`),
  `tex_atlas.hpp` (hole-inpaint), wired `pixal3d --remesh`. Env `PIXAL3D_REMESH_{STRIDE,BLUR,SMOOTH}`.
- **Flash-attn LANDED** (FINDINGS-15 §4, PERF-NOTES LAP 5): root cause = f16 PV-accum overflow on un-
  normalized V; fix = power-of-2 V pre-scale. m3b −17%, M3b E2E −19%, cosine 0.9991, N1 exact. Repro
  tooling in-tree: `fa_repro.cpp`, m3b `PIXAL3D_FA_CAPTURE`+`CAPFWD`, harness `PIXAL3D_{SCHED,NO_GALLOC}`.

## DEFINITION OF DONE (lap-17)
1. Atlas ComputeCharts 30-42s → ~few s (normal-cone pre-cluster), bake raster on GPU, M4 extract on
   GPU/seconds → **no long CPU tail**. Turtle asset validated E2E + in compare.html.
2. Perf/VRAM re-profiled post-flash; SS/tex DiT flash wired + N1-validated (or documented why not);
   next lever pulled; peak VRAM re-measured.
3. MoGe-as-service shipped; the PNG→GLB path wired end-to-end behind a native server (worker-isolation,
   idle-unload true-0, REST). Fingers/holes default chosen by render.
