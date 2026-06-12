# HANDOFF — Pixal3D: proper remesh + in-process front-end + full perf/VRAM pull

**Status (2026-06-12):** the cleanup punch-list (FINDINGS-14) landed an INTERIM build — image →
**watertight** (0 holes) textured GLB, complex assets no longer OOM, auto-camera works, VRAM 8.2→5.9 GB.
But three of those were done the quick way; the owner wants them done **properly**. Run **fully
autonomously** (owner away; GPU + CPU free). Decide, do, golden-validate vs the true-fp32 oracle (judge
E2E by mesh IoU/cosine, not tight tol once perf loosens precision), document, continue. Only halt if
hard-stuck. Memory: `project_3dgen_cpp_port`, `project_avatar_rig_path`, `feedback_correctness_before_perf`,
`reference_subagent_background_stall`, `feedback_no_build_on_server`. READ IN FULL: `FINDINGS-14-cleanup-
robustness-perf.md`, `PERF-NOTES-pixal3d.md` (LAP 1-4), this file.

Build/run unchanged: `cd tools/m1_ref/cpp_port && ./build.sh <target> cuda`; full CLI `./build.sh pixal3d
cuda`. Run: `LD_LIBRARY_PATH=/mnt/hdd/3d/avatar-shootout/toolchain/lib:/usr/lib ./pixal3d --model
weights_gguf_f16 --image <matte.png> --out out.glb --tex --fast`. Weights on `/mnt/hdd/pixal3d` (symlinked).
NVIDIA_TF32_OVERRIDE=0 default; `--fast` = the f16/perf config. C++/CUDA builds fine here; **NO Rust builds**.
Long jobs `run_in_background:true` from the MAIN loop (sub-agents deadlock on bg completion). No `pkill -f`,
no rm-globs. Worktree UNCOMMITTED — leave it so. Live viewer: compare.html on `http://10.0.0.208:8011/`
(serves `miku_watertight.glb` — the current build).

---
## WHAT'S DONE (this session — keep, build on)
- **A2 OOM = SOLVED, keep as-is.** `m1_ggml.hpp attention()` tiles the query dim (`PIXAL3D_ATTN_CAP_MB`,
  default 3072) — bit-identical to dense (Miku cosine 0.999877, N1=1120/M=4631 exact), the turtle
  (M=15313) clears the old 11GB-scores OOM. Flash-attn was tried + REJECTED (NaN: ggml's null-mask MMA
  kernel reads OOB on non-256-multiple n_kv). The CORRECT flash (pad n_kv→×256 + a mask) is a PERF task
  below, not a correctness one.
- **D VRAM = 8199→5895 MiB (≤7.5GB), keep.** The real peak was **NAF@1024** (8185 isolated), NOT the M3b
  DiT — its ImageEncoder runs 128ch-k3 convs at 1024² → `ggml_conv_2d` im2col ~4.8GB f32. Fixed by F16
  im2col under `--fast` (`naf_graph.hpp conv()`: cast kernel→F16; matmul F32-accumulates) → 5883, near-
  lossless (naf_1024_test meanabs 8e-6). **IMPORTANT for the quant discussion below: the peak is
  ACTIVATION/im2col-bound (NAF), not WEIGHT-bound — DiT weights are 2.8GB f16 and are NOT the peak.**

## REMAINING WORK (do in order)

### 1. A1 — proper watertight REMESH (replace the mirror-cap hack) ★
**Why:** the current "watertight" is `close_surface` + ear-fill + **mirror-cap** (reverse-of-owning-tri
on residual non-manifold "slit" edges → ~14.6k coincident zero-area flap tris). It reaches boundary==0
but it is a HACK that ADDS degenerate geometry — owner: *"the model-generated meshes already have their
issues, let's not add to it."* Also it leaves the mesh **non-manifold** (161k >2-face edges) → meshopt
stays on `simplifySloppy` → xatlas chart count never collapses (~9k charts, loose atlas; only fast via a
low `--decimate`). Python's `cumesh` does a real dual-contour **remesh** → clean manifold watertight.
**Task:** port a real remesh so the watertight mode is clean (manifold, no flaps) AND unblocks quality
decimation → few charts → tight 2048² atlas. Two candidate algorithms:
  - **Marching Cubes on the occupancy** (simplest robust): we already have the ~1.5M occupied voxel
    coords @grid1024 going into the extractor. Standard MC tables + sparse iteration over occupied
    cells → guaranteed watertight 2-manifold, decimates cleanly. Slightly blockier than the dual-grid
    (loses the learned sub-voxel dual vertices) but at 1/1024 voxel = sub-mm = invisible. Lowest risk.
  - **Manifold dual-contouring** (higher fidelity, more work): keep the learned dual vertices, use a
    manifold-guaranteeing connectivity. Closest to cumesh's output.
  Recommendation: do MC first (robust, fast to land + validate), keep dual-contour as a stretch. Gate it
  the same way (`close_surface`/`in.watertight`); default path stays bit-exact vs o_voxel for the M4
  validation. Validate: boundary==0 AND meshopt `simplify` (quality, not sloppy) reaches the target AND
  xatlas charts drop to tens AND a tight 2048² high-util atlas AND render = clean Miku. The interim
  mirror-cap + the `close_surface` synth-cell path can then be removed (or kept behind a flag).
  Files: `sparse_vae.hpp` (`flexible_dual_grid_to_mesh`, `fill_holes` — the mirror-cap/ear-fill to
  replace), `svp_gpu.hpp`/`sparse_vae_pipeline.hpp` (`m4_decode_mesh`, thread the mode), `tex_atlas.hpp`
  (`decimate` — drop the sloppy fallback once manifold), `pixal3d_chain.hpp` step 7b.

### 2. A1 — expose the Trellis.2/Pixal3D conditioning flags ("how close to the image") ★
The C++ chain HARDCODES the sampler params that Python's `inference.py run_inference` exposes. Surface
them as CLI flags (and as API params for the eventual service). Current hardcoded values in
`pixal3d_chain.hpp` `flow_sampler(...)` calls (args = init_t, **guidance_strength**, **guidance_rescale**,
**rescale_t**, interval_lo, interval_hi, **steps**):
  - SS DiT:  gs **7.5**, gr 0.7, rt 5.0, [0.6,1.0], 12 steps
  - M2 DiT:  gs 7.5, gr 0.5, rt 3.0, [0.6,1.0], 12
  - M3b DiT: gs 7.5, gr 0.5, rt 3.0, [0.6,1.0], 12
  - tex DiT: gs 1.0, gr 0.0, rt 3.0, [0.6,0.9], 12 (CFG off)
  - **seed** is hardcoded `trandn::Generator(42)` in run_geometry.
**guidance_strength = "how close to the image"** (higher = more faithful to the input, less invention);
**steps** = detail/speed; **seed** = variation. Map to Python's `ss_guidance_strength` /
`shape_slat_guidance_strength` / `tex_slat_guidance_strength` / `*_sampling_steps` / `seed` / `mesh_scale`
/ `image_resolution` / `max_num_tokens`. Add `--guidance`, `--steps`, `--seed`, etc. (sensible per-stage
defaults = the current values). Check `inference.py` for the full exposed set + the `max_num_tokens`
down-step loop (the Python complex-asset guard — our query-tiling already handles the OOM, but the
token-budget knob may still be wanted as an API param).

### 3. A3 — IN-PROCESS front-end (no Python piping) ★
Owner: *"we have rmbg running as a service... i'd like this all in-process... wired via an API on this
service. I don't want to pipe off Python commands."* End state = the GPU service exposes upload-PNG→GLB
in-process. So:
  - **MoGe-2 camera estimation → port to C++/ggml in-process** (replace `estimate_camera.py`). MoGe-2 =
    `Ruicheng/moge-2-vitl`, a ViT-L monocular-geometry model (cached at `~/.cache/huggingface/hub/
    models--Ruicheng--moge-2-vitl`). It's a normal ViT (the DINOv3 port is the template — CLIP-ViT-ish
    + a geometry/intrinsics head). We only need its predicted **intrinsics** (`fx`) → `camera_angle_x =
    2·atan(W/(2·fx))`; the `distance_from_fov` math is already pure host (see `estimate_camera.py` /
    `inference.py`). This is a real model port (like DINOv3) but bounded — we only need the intrinsics
    head, not the full point map. Validate vs `estimate_camera.py` (which gives fov=42.01°/dist=1.3022
    on the Miku matte — EXACT vs the Python ref).
  - **rembg preprocess → use the existing service** (not `preprocess_photo.py`). Wire the C++/API path to
    call the running rmbg service for the matte instead of the Python script.
  - The deliverable is the upload-PNG → (rmbg service) → (in-process MoGe cam) → `pixal3d` chain → GLB,
    all in-process behind an API (koblem-style heavy GPU engine: worker-isolation, idle-unload true-0,
    REST + panel — see PART E of `HANDOFF-NEXT-cleanup-robustness-perf.md`). Rig/motion stay separate.

### 4. D — full kernel-level perf + VRAM pull (ncu/nsys via docker) ★
The owner profiles via the **docker builder images** (which have ncu/nsys; host toolchain does NOT —
that's why this session only got per-stage VRAM, not kernel traces). Get proper **ncu/nsys** traces of
the hot stages, dig into the hot kernels, rerun, find the new hot spot. Per-stage breakdown now (Miku
`--fast`, ~206s, peak 5895 MiB):
  | stage | time | VRAM(iso) | lever |
  |---|---|---|---|
  | M3b DiT | 50.7s | 4569 | flash-attn (mask+pad) ≈ −20% |
  | SS DiT | 41.9s | ~4.5G | same |
  | tex DiT | 29.0s | ~4.5G | same |
  | M4+M6 decode | 40s | <2G | spike conv kernel (fp32, no tensor cores) |
  | NAF@1024 | 4.2s | **5883←peak** | spatial conv tiling (GroupNorm-coupled) |
Levers to PULL, in order (quality-preserving FIRST; owner: exhaust everything before quant):
  1. **flash-attn done right** — the −20% DiT win backed out this session. Pad n_kv→×256 + supply a
     mask (a [n_kv,n_q] f16 mask is ~50MB Miku / ~470MB turtle — fine vs the 11GB scores it replaces);
     don't pad n_q. Validate cosine vs fp32 (≈ the 0.9999 the tiled path gives). DiTs = 59% of wall.
  2. **spike conv kernel** (~18s decode floor): fp32 implicit-GEMM, no tensor cores (`sparse_subm_conv.cu`).
     tf32/f16/tiled-MMA → judge by mesh IoU via `m4_gpu_test` (it perturbs subdiv → not bit-exact).
  3. **NAF@1024 spatial tiling** to drop the activation peak further (GroupNorm couples the spatial
     extent → must compute GN stats globally, then tile the conv im2col). Lossless.
  4. **THEN quantization (the LAST step).** ⚠️ Two facts the owner flagged: (a) **the DiTs are F16 now**,
     so even **Q8 is a step DOWN** — try plain-Q8 and imatrix-Q8 BEFORE Q4. (b) **the VRAM PEAK is NAF
     activation/im2col, NOT DiT weights** — so weight-quant (Q8/Q4) lowers the *DiT-stage* footprint and
     speeds matmuls, but does NOT move the 5.9GB NAF peak. Quantify what Q8/Q4 actually buys at the peak
     vs off-peak. `pack_gguf.cpp` already has `--type {f16,q8_0,q4_k,...}`; imatrix = capture per-channel
     act² over the 12-step forward → `ggml_quantize_chunk` (currently nullptr). Earlier finding: plain-Q8
     on the DiTs was too lossy (cosine 0.968, distributed error) → imatrix is the tool if going sub-F16.

---
## NON-NEGOTIABLES / GOTCHAS (carried)
- Persistent-weights buffer (gallocr recompute→NaN); **float64** t_seq + guidance-interval (M2/M3b hit
  t=0.6 exactly); `NVIDIA_TF32_OVERRIDE=0` for the bit-exact default; fp32 tanh-GELU; the `--fast` path
  is opt-in (keep default bit-exact for validation). DiT attention = query-tiled (PIXAL3D_ATTN_CAP_MB).
- Validate vs the **true-fp32 oracle**, NOT the bf16/tf32 golden. The M4 extractor default path must stay
  BIT-EXACT vs o_voxel (the remesh is a separate gated mode).
- Keep `FINDINGS-*`, `PERF-NOTES-pixal3d.md`, memory `project_3dgen_cpp_port` current.

## FILE MAP (changed this session)
`m1_ggml.hpp` (query-tiled attention + PIXAL3D_ATTN_CAP_MB) · `naf_graph.hpp` (F16 im2col under fast) ·
`sparse_vae.hpp` (`fill_holes` ear+mirror-cap, `boundary_edge_count`, `flexible_dual_grid_to_mesh`
close_surface) · `svp_gpu.hpp`/`sparse_vae_pipeline.hpp` (`m4_decode_mesh` close_surface param) ·
`pixal3d_chain.hpp` (step 7b watertight + in.watertight) · `pixal3d.cpp` (--decimate on plain GLB,
--no-watertight) · `estimate_camera.py` (host MoGe — TO BE PORTED in-process) · `m4_mesh_only.cpp`
(watertight test) · `tex_bake_test.cpp` (FILL_HOLES env) · `compare.html` (→ miku_watertight.glb).

## DEFINITION OF DONE
Clean manifold watertight mesh via a real remesh (no flaps) → tight 2048² atlas; conditioning flags
exposed (CLI + API-ready); MoGe camera + rmbg fully in-process (no Python piping), behind the service
API; perf/VRAM levers pulled with ncu/nsys evidence, stopping just before (and quantifying) the
imatrix-Q8/Q4 step. Each step golden-validated + documented.
