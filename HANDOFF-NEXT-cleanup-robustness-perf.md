# HANDOFF — Pixal3D C++/ggml port: mesh cleanup + robustness + deferred performance

**Status (2026-06-12):** the C++/ggml port is **feature-complete on content** — `pixal3d --tex`
takes an image and emits a textured PBR GLB (UV-atlas baseColor + metallicRoughness + TEXCOORD_0)
matching the Python library's texture content, validated E2E. The geometry chain, the UV-atlas bake,
and a raw-photo front-end are all done. See `FINDINGS-13-uvatlas-textures.md` +
`HANDOFF-NEXT-feature-complete.md` (top status block) for what shipped.

**This handoff = the remaining work to make it a robust, polished, fast product** (the punch-list the
owner greenlit): (A) watertight mesh cleanup, (B) complex-asset robustness, (C) auto-camera, then
(D) all the deferred performance laps + VRAM targets. Rigging/motion are **separate models** wired at
the API layer — see PART E, out of scope for this binary.

Run **fully autonomously** (owner away; GPU + CPU free). Decide, do, golden-validate, document,
continue. Only halt if genuinely hard-stuck. Memory: `project_3dgen_cpp_port`, `project_avatar_rig_path`,
`feedback_correctness_before_perf`, `reference_subagent_background_stall`, `feedback_no_build_on_server`.

---
## Orientation (read first)
- **Build:** `cd tools/m1_ref/cpp_port && ./build.sh <target> cuda`. Full CLI = `./build.sh pixal3d cuda`
  (ggml-cuda graphs + spike conv + GPU-resident decode + xatlas + meshopt). Standalone tests:
  `grid_sample_test` (CPU), `tex_bake_test` (CPU, xatlas+meshopt), `m{3a,4,6}_gpu_test cuda`.
- **Run:** `LD_LIBRARY_PATH=/mnt/hdd/3d/avatar-shootout/toolchain/lib:/usr/lib ./pixal3d --model
  weights_gguf_f16 --image <matte.png> --out out.glb --tex --fast` (the `--fast` perf config; default
  path stays bit-exact). `--texsize N` (atlas), `--decimate F` (pre-unwrap face target), `--vcolor`
  (old COLOR_0), `--cpu`, `--ply`. Weights on `/mnt/hdd/pixal3d/` (symlinked in).
- **Validate philosophy (do not skip):** correctness/precision FIRST, vs the **TRUE-fp32 oracle**
  (NOT the bf16/tf32 golden); judge E2E by **mesh IoU / cosine**, not a fixed elementwise tol once
  perf loosens precision. `feedback_correctness_before_perf`.
- **Live viewer:** `compare.html` on `http://10.0.0.208:8011/` (UV-atlas tab + geometry tab).
- **Goldens:** `tools/sparse_spike/golden_stages/` (stage boundaries) + `cpp_port/refs/` (fp32 oracles).
  Python reference env: `/mnt/hdd/3d/avatar-shootout/Pixal3D/.venv/bin/python` (GPU for flex_gemm/cumesh).

---
## PART A — LOOSE ENDS (feature/quality)

### A1. Watertight mesh cleanup  ★ BIGGEST LEVER (also kills the chart explosion + slow unwrap)
**Problem:** our output is the raw O-Voxel dual-grid mesh. It has **~24k tiny holes** (a quad only
forms where all 4 neighbour voxels exist), so it is NOT watertight. Consequences: (1) lower mesh
quality than Python; (2) xatlas makes a chart per hole-bounded region → **~24k charts → ~100-140s
unwrap + a bloated 4324² atlas at ~20% utilization**. Python's `o_voxel.postprocess.to_glb` fixes this
with `cumesh` (proprietary CUDA): `fill_holes(3e-2)` + **narrow-band dual-contour remesh** + decimate
+ manifold repair + a BVH reproject (samples the volume at the *original* surface to correct remesh
drift). We can't port cumesh.

**Why this is one job:** make the mesh watertight → boundary count collapses → xatlas charts drop from
~24k to tens → unwrap goes from ~120s to seconds and the atlas packs tight at 2048². So **A1 == the
parked "chart-explosion unwrap" perf item.** Two birds.

**Best way (recommended ladder — try cheap first, escalate):**
1. **Targeted FDG hole-fill (cheapest, preserves learned geometry).** Most holes are 1-quad gaps
   (exactly 3 of the 4 neighbour voxels exist). In `svae::flexible_dual_grid_to_mesh` you already build
   the voxel hashmap + iterate edges; add a pass that, for a quad with 3/4 neighbours present, emits a
   triangle (fan-fill) or synthesises the 4th dual-vertex from its 3 neighbours. Validate: boundary-edge
   count ↓, chart count ↓ (rebuild `tex_bake_test`, watch `charts=`), render unchanged. This may close
   most holes for free.
2. **Bundle a robust watertight repair lib (the guaranteed path).** If 1 leaves too many boundaries:
   bundle **Manifold** (github.com/elalish/manifold, MIT — guaranteed-manifold output, modern, used by
   Blender/OpenSCAD) or **MeshFix** (Attene, the classic "make any mesh watertight"). Run it on the
   extracted mesh before decimate/unwrap. Manifold is the cleaner modern choice; bundle like xatlas/
   meshopt under `thirdparty/`. After this, decimate (meshopt) → xatlas will be fast + tight.
3. **Narrow-band dual-contour remesh (closest to Python, most work).** Reimplement `remesh_narrow_band_dc`
   from the occupancy + dual vertices (build a narrow-band SDF, dual-contour, project-back to the FDG
   surface). Highest fidelity to Python's output but a real project; only if 1+2 aren't good enough.
**Decimation note:** `meshopt_simplify` (quality) stalls ~2× target on the holey mesh (border-locked);
once watertight it should reach the target cleanly without the `meshopt_simplifySloppy` fallback in
`texatlas::decimate` (revisit that fallback after A1). Python decimates to **1M verts** — match that as
the default once unwrap is cheap.
**Validate A1:** boundary-edge count == 0 (or ~0), Euler characteristic sane, `charts=` from
`tex_bake_test` drops to tens, atlas back to 2048² high-utilization, render still a clean Miku. Files:
`sparse_vae.hpp` (mesh extractor), `tex_atlas.hpp` (`decimate`/`bake`), `tex_bake_test.cpp`.

### A2. Complex-asset robustness  ★ HARD BLOCKER for the upload-anything API
**Problem:** the C++ DiT attention is **dense (materialises the M² scores tensor)**. At Miku scale
(M≈4.6k) that's ~1 GB and fine; the turtle asset (`assets/images/0_img.png`) hit **M=15313 HR tokens →
~11 GB of scores → `gallocr_alloc_graph failed` (OOM)** at the M3b shape DiT on the 12 GB card. Python
runs fine because it uses flash-attention (no materialised scores). So today the chain only handles
~Miku-complexity assets; the API will get more complex ones.

**Fixes (do both):**
1. **Flash-attention in the DiT (the real fix — also a perf + VRAM win).** Wire `ggml_flash_attn_ext`
   into the attention in `m1_ggml.hpp` (`attention()`) for the sparse DiT (`slat_dit_graph.hpp`) and the
   dense SS DiT (`ss_dit_graph.hpp`). O(M) memory, no scores tensor → turtle fits, SS/M3b speed up, peak
   VRAM drops. Constraints: `ggml_flash_attn_ext` wants F16 K/V + head_dim padding (head_dim 128 here =
   fine); q/k are qk-rms-normed (unit scale) so F16 is safe (we already F16 the attention under `--fast`).
   Validate: cosine vs the current dense path on Miku (must match ~0.9999), then turtle completes +
   renders. This is the SAME work as perf lap B3.
2. **Token-budget downstep (cheap safety fallback, matches Python).** Python's `run()` loops: if
   `num_tokens > cap` (PIXAL3D_MAX_TOKENS=49152) it steps `actual_hr_resolution -= 128` (coarser HR grid
   → fewer tokens) down to a floor. Our C++ chain hardcodes grid64 (`pixal3d_chain.hpp`, the M3a→M3b
   gateway `geo::quantize_grid_unique(...,512,64)`). Implement the loop: quantize at grid_res =
   actual_hr//16, count unique tokens, step actual_hr down by 128 until under a VRAM-safe cap, and thread
   `actual_hr_resolution` into M3b cond (`proj_cond_shape` grid res) + decode `set_resolution`. Lower
   geometry detail for huge assets, but it completes. Use as a guard even with flash-attn.
**Validate A2:** the turtle (`prep_test_matte.png` already made) runs to a textured GLB without OOM;
Miku output unchanged. `nvidia-smi` peak stays < 12 GB (ideally toward the 7.5 GB target — see D).

### A3. Auto-camera (host MoGe)
**What it is / why it matters:** the proj-mode conditioning back-projects image features onto the voxel
grid using the **camera** (FOV `camera_angle_x` + `distance`). Wrong FOV → features land on the wrong
voxels → distorted/incorrect geometry. Today we hardcode `--fov` (default = the Miku cam, 42°). Python
estimates the camera **per image** with **MoGe-2** (`Ruicheng/moge-2-vitl`, a monocular-geometry model):
predict intrinsics → `camera_angle_x = 2·atan(W/(2·fx))`, then derive `distance` via `distance_from_fov`
(pure host math). For "upload anything → 3D" we need this; a fixed FOV only works for assets shot like
the training data.
**Best way (host cut-line, same as rembg — keep C++ on the ggml path):** add MoGe to the host front-end
(`preprocess_photo.py` or a sibling `estimate_camera.py`): run MoGe-2 on the matte, compute
`camera_angle_x` + `distance`, emit them so the CLI is called with `--cam <ang> <dist> <scale>`. The
exact reference is `inference.py`: `get_camera_params_wild_moge()` + `distance_from_fov()` + the
`compute_f_pixels` helper (all read + understood; `distance_from_fov` is portable host math, only MoGe
inference is the model dep). MoGe is a normal torch model in the venv path (or `pip`/HF download).
**Validate A3:** estimated FOV on a few assets is sane (compare to the Python `get_camera_params_wild_moge`
output on the same images) and the resulting geometry is undistorted vs a wrong-FOV run.

---
## PART D — DEFERRED PERFORMANCE LAPS + VRAM (after/with A; re-profile, features shifted the picture)
Full prior intel: `PERF-NOTES-pixal3d.md` (LAP 1-3 done: GPU-resident decode, F16 DiT weights, f16-attn
→ textured E2E 509.7s → ~216s warm). These are the *remaining* levers, priority order:

- **D1 = A1** (watertight) — also the unwrap-speed perf fix (~120s → seconds, atlas 4324²→2048²).
- **D2 = A2.1** (DiT flash-attn) — biggest remaining speed + VRAM lever: SS DiT (42s) + M3b DiT are the
  wall, and the M² scores tensor is the VRAM peak. `ggml_flash_attn_ext`. Validate by mesh cosine/IoU.
- **D3 spike conv kernel (~18s, the decode floor).** Naive fp32 implicit-GEMM, no tensor cores
  (`sparse_subm_conv.cu`). Bit-exact win = better fp32 tiling / shared-mem weight staging / larger TN
  reuse. A tf32/f16 rewrite is faster but perturbs subdiv decisions → judge by **mesh IoU** via
  `m4_gpu_test`. Keep the decode's current bit-exactness if going the fp32-tiling route.
- **D4 VRAM target ≤ 7.5 GB (measured peak 8.0 GB @ M3b DiT on Miku; owner: 8 GB acceptable but 7.5 is
  the budget; sub-4 GB is the stretch).** Levers, in order: (a) flash-attn (D2) drops the scores tensor —
  do this first, re-measure; (b) **imatrix-Q4 the DiTs** for the sub-F16/sub-4GB play — the F16→Q4
  quant error is *distributed* (proven: sparing single layers recovers nothing), so it needs imatrix:
  capture per-input-channel mean(act²) over the 12-step forward → pass to `ggml_quantize_chunk` (today
  nullptr) in `pack_gguf.cpp --type q4_k`; (c) stream/quantize the decode weights (shape_dec ~1.9 GB f32
  resident). Note the UV-atlas bake adds **host RAM** (mesh + atlas), not VRAM. Re-measure peak with
  `nvidia-smi` 5 Hz over a full `--fast --tex` run after D1/D2.

**Acceptance for D:** every perf change validated by **E2E mesh IoU/cosine ≈ unchanged** (precision may
loosen — that's rounding, not breakage). Bank each lap in `PERF-NOTES-pixal3d.md`.

---
## PART E — BEYOND (separate models, wire at the API layer — NOT this binary)
The end state is an upload-PNG → textured-rigged-asset **API** (koblem heavy GPU engine like
acestep/flux2/longcat: worker-isolation, idle-unload true-0, REST + panel). After A-D:
- **Auto-rigging: SkinTokens** (Qwen3-0.6B core → cheap ggml port) — a **separate model/binary**.
- **Motion** (HY-Motion / clip-retarget) — a **separate model**.
These are **separate pieces wired at the API layer**, not inside `pixal3d`. See `project_avatar_rig_path`
+ `project_3dgen_cpp_port`. The API flow: upload PNG → `preprocess_photo.py` (rembg + crop + A3 camera)
→ `pixal3d --tex` (A1/A2/D) → GLB → SkinTokens rig → motion. Don't pull rig/motion into the geom binary.

---
## NON-NEGOTIABLES / GOTCHAS (carried forward)
- Persistent-weights buffer (gallocr recompute → NaN); **float64** t_seq + guidance-interval decision
  (M2/M3b hit t=0.6 exactly); `NVIDIA_TF32_OVERRIDE=0` for the bit-exact default path; fp32 tanh-GELU;
  conv weight `[Co,Kd,Kh,Kw,Ci]`→spike `[27,Cin,Cout]`; tex-decode oracle needs GPU VISIBLE (flex_gemm
  Triton import). The `--fast` path is opt-in; keep the default bit-exact for validation.
- **C++/CUDA builds fine here** (toolchain nvcc, sm_86); **NO Rust builds** (koblem/koblibs are off-server).
  No `pkill -f` (kill by PID), no `rm`-globs. Long jobs **`run_in_background: true`** (harness-tracked);
  **sub-agents DEADLOCK on background completion** → drive long builds/runs from the MAIN loop; bounded
  read-only Explore sub-agents are fine. No multi-agent Workflow for research.
- One GPU → one CUDA run at a time; one heavy torch/python process at a time (don't fan out oracles).
- Keep `FINDINGS-*`, `PERF-NOTES-pixal3d.md`, and memory `project_3dgen_cpp_port` current. Worktree is
  UNCOMMITTED — leave it so unless there's a clear reason to checkpoint.

## FILE MAP (the texture/bake work added this run)
`tex_grid_sample.hpp` (sparse trilinear sample) · `tex_atlas.hpp` (decimate+unwrap+raster+bake) ·
`glb_textured.hpp` (nlohmann textured glTF + PNG) · `grid_sample_test.cpp` / `tex_bake_test.cpp`
(validation) · `tex_grid_sample_capture.py` / `tex_bake_python_ref.py` (oracles) · `preprocess_photo.py`
(rembg front-end) · `render_textured.py` (textured render) · `thirdparty/{xatlas.{h,cpp},meshoptimizer/}`.
Chain: `pixal3d.cpp` + `pixal3d_chain.hpp` (`run_geometry` surfaces the PBR volume) + `svp_gpu.hpp`
(GPU-resident decode, `m6_tex_decode` has `out_coords`). DiTs: `ss_dit_graph.hpp`, `slat_dit_graph.hpp`,
`m1_ggml.hpp` (`attention()` — the flash-attn target).

## DEFINITION OF DONE (this handoff)
Watertight mesh (A1) → fast/tight unwrap; the turtle (and complex assets) run without OOM (A2);
auto-camera from a raw photo (A3); the perf laps + VRAM ≤ 7.5 GB landed and E2E-IoU-validated (D).
Each step golden-validated and documented. Rig/motion (E) stay separate, for the API layer.
