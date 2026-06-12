Continue the C++/ggml Pixal3D (TRELLIS.2) port. **FULL AUTOMATION — get it DONE** (owner away; GPU+CPU
free). Each item is to be DRIVEN TO A WIN this lap, not left half-pulled. Decide, do, golden-validate vs
the true-fp32 oracle (judge E2E by mesh IoU/cosine/render; loosen tol only for perf), document, continue.
Only halt if hard-stuck.

START by reading IN FULL: `/home/dbrain/dev/longcat-sparse-spike/HANDOFF-NEXT-lap16-remesh-FA-frontend.md`
(the plan, with the measured timing table + root-cause analysis), then `FINDINGS-15-remesh-conditioning-
perf.md` + `FINDINGS-13-uvatlas-textures.md` (the atlas bottleneck) + `PERF-NOTES-pixal3d.md`. Memory:
`project_3dgen_cpp_port`, `reference_ncu_docker_syadmin` (ncu/nsys-in-docker recipe), `feedback_
correctness_before_perf`, `reference_subagent_background_stall`, `feedback_no_build_on_server`.

THE GAP vs Python (all one root cause): our wall time image→textured-GLB is ~18min (`--remesh --tex`) /
~6min (normal) vs Python's ~1-2min, AND our textured model looks janky/low-poly vs Python's. Reason:
**xatlas ComputeCharts is the CPU wall-time killer (840s remesh / 77-100s normal) because we never do a
clean REMESH** — Python's `to_glb(remesh=True)` runs cumesh dual-contour remesh → smooth low-poly
watertight mesh → tight atlas → fast. Our model quality MATCHES Python at full res (cosine 0.9999); the
gap is purely the missing clean remesh + the loose/slow atlas + sloppy-decimated faceting it forces.

DO, IN PRIORITY ORDER (full detail + file map + validation + library options in the handoff):
1. **PROPER REMESH → TIGHT ATLAS** (the linchpin — fixes atlas + the 840s CPU + wall time + the janky
   render together). This lap landed marching-tetrahedra = provably MANIFOLD WATERTIGHT (no flaps) but
   it's NOT quality-decimatable (meshopt stalls ~13.8M → sloppy → 47,888 charts, 16.6% util). Make a
   SMOOTH/LOW-POLY/LOW-GENUS manifold mesh instead: try (cheap→heavy) attribute-aware decimation →
   SDF+marching-cubes on a smoothed occupancy field (extends remesh.hpp) → OpenVDB volumeToMesh
   (adaptive) → Instant-Meshes/CGAL isotropic remesh. Target: charts ≲ hundreds, util >50%,
   ComputeCharts <~few s, clean web-sized textured GLB; compare to the pyref; update compare.html. Iterate
   OFFLINE on `refs/stage5/head_coords.npy` via `remesh_test.cpp` (seconds, no GPU). **HARD REQ (owner):
   the whole post-DiT tail (remesh + decimate + xatlas unwrap + raster/grid_sample bake + M4 extract)
   must be GPU-bound OR genuinely seconds on CPU — NO long CPU stretches like today's 840s xatlas. If a
   clean remesh doesn't get a stage to seconds, move it to the GPU (nvdiffrast-style raster, GPU/Morton
   M4, GPU unwrapper).**
2. **FLASH-ATTN — whatever it takes — then RE-PROFILE.** ncu/nsys (working now, see the reference memory)
   showed ~50% of DiT GPU time is attention OVERHEAD (28.5% permute-copies + 9.2% f16 casts + 12%
   softmax, all under-saturated). This lap's FA attempt NaNs even with an all-zero mask → it's the
   cc86/D=128 `mma_f16` kernel itself on these shapes (nh=12, gqa=1), NOT the mask. Crack it: minimal
   standalone repro to bisect the NaN; force a different FA kernel (wmma/tile/vec) — patch our ggml fork's
   selector if needed; OR a parallel win = kill the 6930 perm-copies/casts in the dense path. Validate
   cosine ≥0.999 + measure DiT speedup; re-profile for the next lever; quantify+attack the HOST CPU
   stages (M4 extract 19.8s, tex-decode coord-growth). Quant LAST (DiTs are F16; peak is NAF, not DiT
   weights). Build/profile CONTAINER-NATIVE in `longcat-avatar-dev:builder` (`ggml/build-cuda-docker`).
3. **IN-PROCESS FRONT-END — work out the architecture.** MoGe-2 has NO intrinsics head (it's a full
   ViT-L + point-map + focal-solve — multi-day to port). RECOMMENDED: run MoGe as a warm HOST SERVICE
   like the existing rmbg service (HTTP, model loaded once) → C++ GPU service calls rmbg + MoGe services
   for matte+camera, runs the pixal3d chain in-process → GLB behind the API (koblem heavy engine). That
   satisfies "no Python piping" without the multi-day port. Decide service-vs-port explicitly; ship the
   service path now.

KEY: build `cd tools/m1_ref/cpp_port && ./build.sh <t> cuda`; weights `/mnt/hdd/pixal3d` (symlinked);
`--fast` = perf config; default stays bit-exact; NVIDIA_TF32_OVERRIDE=0; persistent-weights buffer;
float64 t_seq; DiT attention query-tiled (PIXAL3D_ATTN_CAP_MB); remesh gated `--remesh` (+ Taubin,
PIXAL3D_REMESH_SMOOTH); flash gated `PIXAL3D_FLASH` (currently NaN/off). C++/CUDA builds fine here; NO
Rust builds. Long jobs run_in_background:true from the MAIN loop (sub-agents deadlock on bg completion);
kill by PID, no pkill -f / rm-globs. compare.html on :8011. Keep FINDINGS/PERF-NOTES/memory current.
Worktree UNCOMMITTED — leave it so.

DONE = `pixal3d --remesh --tex` gives a TIGHT atlas (charts ≲ hundreds, util >50%, ComputeCharts <~10s)
+ a clean textured render that stands next to Python's, wall time in Python's ballpark; flash-attn (or an
equivalent attention-overhead kill) LANDED + validated + re-profiled with the next lever pulled; in-
process front-end architecture decided + matte+camera wired end-to-end behind the API. Each step golden-
validated + documented.
