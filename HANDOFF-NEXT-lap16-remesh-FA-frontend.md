# HANDOFF — Pixal3D next lap: tight-atlas remesh · flash-attn · in-process front-end

**Run FULLY AUTONOMOUSLY (owner away). "Full automation — get it done": each item below is to be
DRIVEN TO A WIN, not left half-pulled.** Golden-validate vs the true-fp32 oracle (judge E2E by mesh
IoU / cosine / render; loosen tol only for perf). Worktree `/home/dbrain/dev/longcat-sparse-spike`
is UNCOMMITTED — leave it so. Build: `cd tools/m1_ref/cpp_port && ./build.sh <t> cuda`; weights
`/mnt/hdd/pixal3d` (symlinked); `--fast` = f16 perf config; `NVIDIA_TF32_OVERRIDE=0` default; long
runs `run_in_background:true` from the MAIN loop (sub-agents deadlock on bg completion); no `pkill -f`
/ rm-globs (kill by PID). compare.html on `:8011`. READ IN FULL: `FINDINGS-15-remesh-conditioning-
perf.md`, `FINDINGS-13-uvatlas-textures.md` (the atlas bottleneck), `PERF-NOTES-pixal3d.md`. Memory:
`project_3dgen_cpp_port`, `reference_ncu_docker_syadmin`, `feedback_correctness_before_perf`,
`reference_subagent_background_stall`, `feedback_no_build_on_server`.

## THE ONE NUMBER THAT MATTERS: wall time image→textured-GLB, and WHY we're behind Python
Measured this lap (`pixal3d --remesh --tex --fast`, Miku, RTX 3060):
| stage | time | where |
|---|---|---|
| DINOv3@512/1024 | 1.1+1.7s | GPU |
| **SS DiT** | **41.9s** | GPU (4096-tok dense DiT) |
| NAF@512/1024 | 3.7+4.2s | GPU |
| M2 DiT | 9.7s | GPU |
| M3a upsample | 5.5s | GPU-resident + host coord-growth |
| **M3b DiT** | **50.7s** | GPU (sparse DiT, M≈4631) |
| M4 mesh extract | 19.8s | **HOST CPU** (coord-growth + extractor) |
| remesh (marching-tet+taubin) | 9.5s | HOST CPU |
| tex DiT | 29.0s | GPU |
| tex decode | 26.9s | GPU-resident + host coord-growth |
| **xatlas ComputeCharts** | **840.1s** | **HOST CPU — THE wall-time killer** |
→ chain 243.9s + atlas 852s = **~18 min**. Normal (non-remesh) path: chain ~216s + atlas ~77-100s
(dual-grid 24k charts) = **~5-6 min**. **Python is faster AND its atlas is cleaner because
`to_glb(remesh=True)` runs cumesh dual-contour REMESH (CUDA) to a smooth low-poly watertight mesh
FIRST → few charts, tight high-util atlas, fast unwrap.** Model quality we already MATCH (cosine
0.9999, same mesh). The whole gap is the missing clean remesh + the CPU atlas it causes.

---
## PRIORITY 1 — PROPER REMESH → TIGHT ATLAS (the linchpin: fixes atlas + CPU time + wall time)

**State:** this lap replaced the degenerate mirror-cap flap hack with **Marching Tetrahedra** (Kuhn
6-tet → provably MANIFOLD + WATERTIGHT, boundary=0/nonmanifold=0, renders clean; `remesh.hpp`,
gated `--remesh`). **But manifold ≠ tight-atlas:** the voxel-lattice MT surface is diagonally
faceted; meshopt QUALITY decimation STALLS (~13.8M faces, any options/target_error — `SimplifyPrune`
flag @err1.0 nukes the mesh, @0.05 leaves the floor; Regularize doesn't help; χ→genus≈1043 doesn't
explain it) → falls to `simplifySloppy` → noisy normals → **47,888 charts, 5160² atlas @ 16.6%
util, 840s ComputeCharts**. WORSE than the dual-grid path. So MT-of-occupancy is the wrong remesh
for the atlas.

**GOAL:** a SMOOTH, LOW-GENUS, LOW-POLY, MANIFOLD mesh (what cumesh produces) → quality-decimatable
→ **charts in the tens-to-low-hundreds, atlas util >50%, ComputeCharts <~10s, render clean.**
Validate against the Python pyref atlas (`miku_uvatlas_pyref.glb`; compare chart count + util).

**This is ALSO the "our model looks janky vs Python" complaint (owner, this lap).** compare.html
shows our textured GLB (139,633 f, sloppy-decimated, loose 16.6%-util atlas) next to the Python
pyref (3,251,686 f, FULL undecimated mesh) — unfair on poly count AND our sloppy decimation gives
faceted normals + texture stretch. Our model is CLEAN at full res (geometry tab, 1.5M f matches
Python). So the visual gap == the atlas gap == this Priority. **A clean remesh that quality-decimates
to a moderate, smooth, well-normaled mesh fixes the render too.** Deliverable: a web-sized
(~few-hundred-k-face) clean textured GLB that stands next to the pyref without looking low-poly;
update compare.html cell A to it.

**Approaches (pick by trying cheapest first; libraries exist — use them):**
1. **SDF + Marching Cubes (cheapest, extends `remesh.hpp`):** build a *smoothed* signed-distance /
   low-pass field from the grid-1024 occupancy (narrow-band distance transform, or a 3³/5³ box blur
   of the binary occupancy), run MC at iso=0.5 → a SMOOTH mesh (low curvature) whose meshopt quadric
   decimation WORKS (the stall is curvature/faceting, not genus) → tight atlas. Keep it manifold
   (use the same edge-dedup). This is the most likely quick win and stays in-tree.
2. **OpenVDB (best quality, heavy dep):** occupancy → level set (`tools::particlesToLevelSet` or
   `meshToVolume`) → `volumeToMesh` with **adaptivity** → smooth adaptive low-poly manifold mesh
   directly (OpenVDB's adaptive mesher fills cumesh's role). Buildable in the docker builder.
3. **Instant Meshes / isotropic remesh (libigl / CGAL `isotropic_remeshing`):** feed the marching-
   tet mesh → field-aligned / isotropic remesh to ~100k faces with coherent normals → ideal for the
   atlas.
4. **Attribute-aware decimation (try FIRST, 5 min):** `meshopt_simplifyWithAttributes` passing
   per-vertex normals (+ weights) on the MT mesh — may collapse where plain quadric stalled.

Also: **xatlas chart-clustering** — even with a decent mesh, pre-cluster faces by normal-cone before
unwrap (Python does this) and/or raise `ATL_MAXCOST` further; but the mesh quality is the lever.
Keep the manifold MT result available as the rigging geometry base regardless.

**HARD REQUIREMENT (owner): the post-DiT path must be GPU-bound OR genuinely seconds on CPU — no long
CPU stretches.** Right now the whole tail is CPU: xatlas ComputeCharts (840s/77s — bundled
`thirdparty/xatlas.cpp`, compiled INTO pixal3d, single-threaded in-process), our CPU triangle
rasterizer + grid_sample bake (`tex_atlas.hpp`), and M4 mesh extraction (19.8s host). Python's tail is
GPU (cumesh CUDA remesh + nvdiffrast raster). Acceptance: with a clean remesh, xatlas should drop to
**seconds** (then CPU is fine) — verify it actually does. If ANY tail stage stays a long pole (xatlas,
raster, M4 extract), move it to the GPU (GPU rasterizer à la nvdiffrast for the bake; GPU/Morton
neighbor-map + extraction for M4; a GPU unwrapper if xatlas can't get to seconds). Profile the tail
(it's CPU so use plain timing / perf, not ncu) and report each stage's time.

**Files:** `remesh.hpp` (marching_tetrahedra/taubin_smooth — add the SDF+MC path here), `tex_atlas.hpp`
(`decimate`/`bake`), `pixal3d_chain.hpp` step 7b, `pixal3d.cpp` (`--remesh`). Offline iterate on the
saved occupancy `refs/stage5/head_coords.npy` via `remesh_test.cpp` (seconds, no GPU) — that's how
this lap iterated; the chart/decimate/atlas harness is already in `remesh_test.cpp`.

---
## PRIORITY 2 — FLASH-ATTN (whatever it takes) + RE-PROFILE for the next lever

**Why it's worth it (ncu/nsys evidence, this lap):** the M3b DiT spends **~50% of GPU time on
attention overhead** — `cpy_perm_coalesced` 28.5% (6930 launches!) + `convert_unary f→half` 9.2% +
`soft_max_f32` 12% — all UNDER-saturated (memory/overhead-bound per ncu SpeedOfLight, SM 40-53%).
Only ~30% is real matmul. SS DiT (41.9s) + M3b (50.7s) are the GPU long-poles. `ggml_flash_attn_ext`
fuses QK+softmax+AV (no `[tk,tq,head]` scores, no separate permute/cast) → attacks that ~50%.

**State:** implemented "done right" (permute q/k/v→[d,n,head], pad K/V `n_kv`→×256 zeros, F16 K/V,
**q stays F32** — the `mma_f16` kernel asserts `Q->type==F32`, F16 mask built once shared across
blocks; `m1_ggml.hpp attention()` flash branch + `slat_dit_graph.hpp`, gated `PIXAL3D_FLASH`).
**Result: all-NaN, even with an all-zero mask** → it is NOT the null-mask OOB the prior handoff
hypothesized; the cc86 / D=128 `ggml_cuda_flash_attn_ext_mma_f16_case<128,128,64,1>` kernel itself
fails on these shapes (n_head=12, gqa_ratio=1, N≈4734). Gated OFF; query-tiled stays default.

**Crack it (whatever it takes):**
1. **Minimal standalone repro:** a tiny test calling `ggml_flash_attn_ext` on exactly D=128/nh=12/
   N=4734 with a CPU reference; bisect the NaN — try D=64, nh=8 (pow2), gqa_ratio≥2 (duplicate KV
   heads), padded vs unpadded n_q, F16-vs-F32 q on kernels that accept F16. Pin the exact failing
   condition.
2. **Force a different FA kernel:** the dispatcher (`fattn.cu ggml_cuda_get_best_fattn_kernel`) picks
   mma_f16 for D=128/cc86. Try routing to the **wmma** (`fattn-wmma-f16.cu`), **tile**, or **vec**
   kernel — one may be correct. May need a small ggml-cuda patch to the selector (WE OWN the ggml
   fork — patch it; the build is `ggml/build-cuda` host + `build-cuda-docker` for profiling).
3. **If a kernel bug:** fix it in our ggml fork (the mma_f16 D128 path), or upstream-cherry-pick.
4. **Parallel win even without flash:** kill the 28.5% `cpy_perm_coalesced` + 9.2% cast in the DENSE
   path — investigate why attention() emits so many `ggml_cont`/`ggml_permute`/`ggml_cast` (6930
   copies!). Caching the f16 K/V cast, fusing permutes, or a custom fused-attention kernel could
   reclaim a big chunk without flash.

**Then RE-PROFILE** (ncu/nsys now work — see `reference_ncu_docker_syadmin`: container-native build
in `longcat-avatar-dev:builder`, `ggml/build-cuda-docker`, `m3b_sampler_test_docker`; ncu needs
`--cap-add SYS_ADMIN`; nsys is bundled at `/opt/nvidia/nsight-compute/2025.2.1/host/target-linux-
x64/nsys`, salvage its qdstrm via `apt install libdw1` + `QdstrmImporter` then `nsys stats --report
cuda_gpu_kern_sum`). Find the next lever that screams out. **Quantify & attack the remaining HOST CPU
too:** M4 mesh extract 19.8s (coord-growth + extractor — `build_nmap` hashmap per level, host index
arithmetic; candidate for GPU/Morton) + tex-decode coord-growth. (The 840s xatlas CPU is Priority 1.)

**Then quant LAST:** DiTs are F16 now (Q8 = step down; plain-Q8 cosine 0.968 distributed error →
imatrix-Q8 before Q4); the VRAM peak is NAF activation/im2col (5.9GB), NOT DiT weights — so
weight-quant won't move the peak. Quantify what it actually buys at peak vs off-peak. NAF spatial
conv tiling (GroupNorm-coupled) is the lossless peak-VRAM lever; spike conv tensor-core (~18s decode
floor, judge by mesh IoU) is the other.

---
## PRIORITY 3 — IN-PROCESS FRONT-END (work out the right architecture)

**Correction (verified `moge/model/v2.py`):** MoGe-2 has NO intrinsics head — intrinsics come from
`recover_focal_shift()` over the FULL predicted POINT MAP + mask. A faithful in-process C++/ggml port
= ViT-L DINOv2 (SwiGLU+LayerScale+reg tokens) + DPT points_head + mask_head + the focal solve = a
multi-day, separately-golden-validated port (the prior handoff's "just the intrinsics head" was
wrong).

**Recommended pragmatic architecture (libraries/pattern — likely the right call):** run **MoGe-2 as a
warm HOST SERVICE**, exactly like the existing rmbg service the owner endorsed. A tiny persistent
Python service (load `Ruicheng/moge-2-vitl` once, HTTP endpoint → `{camera_angle_x, distance,
mesh_scale}` from `estimate_camera.py`'s already-validated math, fov=42.01°/dist=1.3022 on Miku).
The C++ GPU service calls it over HTTP like rmbg → **no per-call Python subprocess piping** (which
is what the owner objected to), no multi-day port. End state: upload-PNG → (rmbg service: matte) →
(MoGe service: camera) → in-process pixal3d chain → GLB, all behind the GPU-service API (koblem heavy
engine: worker-isolation, idle-unload true-0, REST + panel — see PART E of the older cleanup handoff).
**Decide explicitly:** MoGe-as-service (fast to ship, matches rmbg pattern) vs the full ViT-L ggml
port (own multi-day lap). If unsure, ship the service now; schedule the port as a stretch.

---
## WHAT'S ALREADY DONE (keep, build on) — this lap
- A1 **marching-tet remesh** = clean manifold watertight, no flaps (`remesh.hpp`; `--remesh`; in-
  pipeline + textured-render validated `miku_remesh_tex.glb`/`_render.png`). Geometry base for rigging.
- A1 **conditioning flags** fully exposed (CLI + ChainInput `StageSampler`; defaults==inference.py).
- D **ncu/nsys working under docker** (container-native build; the kernel breakdown above).
- compare.html updated (cell A = `miku_remesh_tex.glb`). FINDINGS-15 has the full detail + the
  flash-attn NaN repro + the meshopt-stall analysis.

## DEFINITION OF DONE (next lap)
1. `pixal3d --remesh --tex` → **tight atlas** (charts ≲ hundreds, util >50%, ComputeCharts <~few s),
   clean textured render, wall time in Python's ballpark; compared to the pyref atlas. **No long CPU
   tail** — every post-DiT stage is GPU-bound or genuinely seconds (xatlas/raster/M4); moved to GPU
   if not.
2. Flash-attn (or an equivalent attention-overhead kill) LANDED + validated (cosine ≥0.999) +
   measured DiT speedup; re-profiled; the next lever identified/pulled; host-CPU stages quantified.
3. In-process front-end architecture decided + the matte+camera path wired end-to-end behind the API.
