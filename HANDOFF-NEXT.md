# HANDOFF — Pixal3D/TRELLIS.2 C++ port: next steps (autonomous)

You are continuing the C++/ggml port of the Pixal3D (TRELLIS.2) image→3D pipeline.
**Run fully autonomously — the owner is asleep, GPU + CPU are 100% free, don't wait on
them unless you are genuinely, hard-stuck** (a decision that reverses direction, or a
tool that's actually broken). Otherwise: decide, do, validate, document, continue.

## Where things stand (updated 2026-06-12 — EVERY GEOMETRY COMPONENT DONE; NEXT = M5 chain assembly)
- **NEWEST (2026-06-12) — M3a + M4 + M5-shape-cond all VALIDATED.** Every geometry component of the pipeline
  is now ported + validated:
  - **M3a** sparse-VAE upsample: `decoder.upsample(lr_slat,x4)→hr_coords` **bit-exact CPU+CUDA** (hr_coords
    SET-EQUAL fp32 oracle 382584; per-stage coords EXACT). `m3a_upsample.cpp`+`sparse_vae.hpp`.
  - **M4** O-Voxel mesh: full `decoder.forward` (bit-exact) + `flexible_dual_grid_to_mesh` **BIT-EXACT vs
    the real o_voxel `_C.so`** (verts maxabs 0.0, faces elem-diff 8/9.75M; 1547112v/3251950f). `m4_mesh.cpp`
    +`m4_mesh_only.cpp`.
  - **M5 shape cond** (true cond, both stages): DINOv3@512/1024 + NAF@512/1024 + proj_grid16/32/64; stage2
    cond (cosine 1.0, lr 2.2e-5) + stage3b cond (cosine 1.0, lr 2.2e-5, hr 1.29e-4) validated vs golden.
    `stage2_cond_test`/`stage3b_cond_test`/`dinov3_1024_test`/`naf_1024_test`; parameterized dinov3/naf graphs.
- **Earlier rungs (full detail in `E2E-PORT-KICKOFF.md` PROGRESS LOG, newest-first):** spike (submanifold
  sparse conv, bit-exact) · M0 goldens (`tools/sparse_spike/golden_stages/`) · M1 fp32 refs · RUNG-1 (Stage-1
  image→coords in ggml, == torch fp32: 1120 coords, IoU 0.9859) · M2 (shape-LR DiT+sampler+denorm+NAF) ·
  M3b (shape-HR DiT). All validated CPU+CUDA vs true-fp32 oracles. Durable gotchas: (a) t_seq/guidance-interval
  MUST be float64 (M2/M3b hit a step exactly at t=0.6); (b) validate vs the true-fp32 oracle, NOT the tf32/bf16
  golden; (c) persistent-weights buffer + `NVIDIA_TF32_OVERRIDE=0` for fp32 on CUDA.

## ★ THE PRODUCT GOAL — a usable CLI (read this first)
The north star is **a self-contained CLI: load a Pixal3D/TRELLIS.2 GGUF, give it an image, get a usable
3D model out — matching the Python library.** Then textures, then make it fast. Three phases:

**PHASE A — Geometry CLI (image → untextured GLB, from GGUF).** All the geometry *math* is DONE + validated
(every stage below). What's left is **assembly + packaging**:
  1. **M5 chain assembly** — wire the validated programs (stage1_e2e + shape cond + m2/m3a/m3b/m4) into ONE
     driver: image → mesh. Net-new plumbing only (cross-stage seed-42 noise, grid64 quantize+unique, mesh→GLB).
     Precise step-by-step: **`HANDOFF-M5-assembly.md`**.
  2. **GGUF (the deferred task #5)** — the ports currently load per-tensor `.npy` (dev format). For a real CLI,
     convert the safetensors (ss_flow, ss_dec, dinov3, slat_flow_512/1024, shape_dec, NAF) → GGUF and load in
     C++ (base repo has GGUF load+convert in `src/convert.cpp`). This is the bridge from dev-harness to product.
  3. **Front-end + CLI** — `pixal3d --model m.gguf --image in.png --out out.glb [--fov F | --cam ...]`. Camera
     scalars (camera_angle_x, distance, mesh_scale) are inputs; recommended cut-line = host-side rembg (BiRefNet)
     + `--fov` to bypass MoGe (both are standard nets off the ggml math path; port only for "raw photo → GLB").
  → **Phase-A done = `pixal3d` produces an untextured GLB from a photo, matching Python geometry (IoU ~0.99).**

**PHASE B — Textures (M6 = feature-complete).** tex SLat DiT (`slat_flow_imgshape2tex_dit_1_3B_1024`, in_ch 64
= 32 noise ‖ 32 shape_slat re-normed, CFG off, interval [0.6,0.9]; reuses build_slat_dit_forward) + tex decoder
(`tex_dec_next_dc_f16c32_fp16`, out 6 PBR, pred_subdiv=false → reuses shape's `subs` as guide_subs; reuses the
M3a/M4 sparse-VAE backbone) + NAF@1024 (reused) + **textured-GLB bake** (`o_voxel.postprocess.to_glb` — UV unwrap
+ atlas + sample attrs from the volume; the largest net-new M6 piece, `_C.so`-backed → golden-validate). Goldens
already captured: `golden_stages/stage4_{cond,out}`. → **image → TEXTURED PBR GLB.**

**PHASE C — Performance run ("fast and usable").** Only AFTER feature-complete. Loosen precision / quantize
(GGUF Q-types) / fuse / CUDA-optimize the per-stage kernels / fit ~7.5 GB co-resident (the low_vram budget;
per-stage peak is DINOv3@1024+NAF ≈ 6.3 GB alloc / 7.6 GB reserved, from M0 `vram.json`). Expect precision to
loosen (tf32/fp16, 1e-7→1e-3) — that's rounding, judged by E2E mesh IoU, NOT breakage. Bank perf intel as you
go (a PERF-NOTES doc). See memory `feedback_correctness_before_perf`.

### How to build/run the C++ port (cpp_port/)
`cd tools/m1_ref/cpp_port && ./build.sh <test> [cuda]` then `./​<test>` (CPU) or
`LD_LIBRARY_PATH=/mnt/hdd/3d/avatar-shootout/toolchain/lib:/usr/lib NVIDIA_TF32_OVERRIDE=0 ./<test> cuda`.
Validated tests by stage:
- **Stage1**: proj_grid_test, dinov3_test, ss_dit_test, ss_vae_test, sampler_test, **stage1_e2e** (image→coords).
- **M2/M3b**: slat_dit_test, **m2_sampler_test**, **m3b_sampler_test**, naf_test (NAF@512).
- **M3a**: **m3a_upsample** (bit-exact upsample, CPU+CUDA).
- **M4**: **m4_mesh** (decoder→head→mesh, CPU+CUDA), **m4_mesh_only** (mesh extractor isolation, emits
  `miku_geometry.ply`). `build.sh` cuda branch covers m3a/m4 (nvcc spike `.cu` + g++ host, no ggml).
- **M5 cond**: **stage2_cond_test**, **stage3b_cond_test**, **dinov3_1024_test**, **naf_1024_test**.
Weights: `weights_npy/` (per-tensor `.npy` — DEV format; **GGUF is Phase-A item #2**) from `unpack_weights.py`
(+ `stage3a_capture.py` exports shape_dec). Refs `refs/` from the `*_capture.py` / `torch_stage*_ref.py` oracles.
Headers: `m1_ggml.hpp` (harness), `dinov3_graph.hpp` (Cfg CFG512/1024), `naf_graph.hpp` (Cfg CFG512/1024),
`slat_dit_graph.hpp`, `ss_dit_graph.hpp`, `ss_vae_graph.hpp`, `sparse_vae.hpp` (M3a/M4 + write_ply).

## FULL SCOPE LADDER — the horizon (GOAL = FEATURE-COMPLETE with Pixal3D = textured GLB from a photo)
Don't lose sight of the end state. The detailed "## The ladder" below is geometry-only (M2–M5);
this is the whole arc:
```
[DONE✅] spike (sparse conv) · M0 goldens · M1 fp32 refs · RUNG-1 (Stage-1 image→coords, ==torch fp32)
[DONE✅] M2 (shape-LR DiT+sampler+denorm+NAF→lr_slat) · M3a (upsample→hr_coords, BIT-EXACT)
[DONE✅] M3b (shape-HR DiT→shape_slat) · M4 (decoder + O-Voxel mesh extractor, BIT-EXACT vs _C.so)
[DONE✅] M5 shape COND (DINOv3@512/1024 + NAF@512/1024 + proj_grid16/32/64; both shape conds validated)
         ── ^ EVERY GEOMETRY COMPONENT validated. Cascade validated stage-by-stage w/ golden hand-offs.
── PHASE A (geometry CLI): assembly + packaging of the validated components ──
A1  M5 CHAIN ASSEMBLY: wire stage1+cond+m2+m3a+m3b+m4 → one image→mesh driver (+seed-42 noise, GLB).
    ── ^ "functionally complete geometry". Plan: HANDOFF-M5-assembly.md.
A2  GGUF (deferred task #5): safetensors → GGUF + load in C++ (replace per-tensor .npy dev format).
A3  CLI + front-end: `pixal3d --model m.gguf --image in.png --out out.glb [--fov F]`; host rembg + --fov.
── PHASE B (feature-complete) ──
M6  TEXTURE branch: tex SLat DiT (in_ch 64, reuses M2/M3 DiT) + tex decoder (out 6 PBR, reuses M3/M4
    sparse-VAE) + NAF@1024 (reused) + textured-GLB bake (o_voxel.postprocess.to_glb, UV/atlas).
    → image → TEXTURED PBR GLB ✅   (goldens stage4_{cond,out} captured)
── PHASE C (product goals, separate phases) ──
PERF  loosen precision / quantize (GGUF Q-types) / fuse / fit ~7.5 GB co-resident / CUDA-optimize.
PROD  integrate as longcat-avatar.cpp subsystem / koblem heavy engine (worker-iso, idle-unload,
      REST + panel, docker) — like acestep/flux2/longcat.
RIG/MOTION  SkinTokens rig + body motion → textured RIGGED GLB — the north-star; separate model chain.
```

## Working style — parallelize PREP, serialize VALIDATION (obey the contention rules)
The port is already built for isolation: each op = a numpy/torch fp32 ref + a standalone ggml
test validated against a captured golden, on a shared harness (`m1_ggml.hpp`) + per-stage graph
headers. Every remaining stage (NAF, sparse up/C2S/subdiv, FDG mesh extractor, tex DiT/decoder)
is a self-contained unit you build + validate in isolation — same recipe as Stage-1. So:

**USE SUBAGENTS (Agent tool) for the parallelizable, non-contending prep:**
- Read-only research/spec, in parallel: bounded `Explore`/`general-purpose` reads of the Pixal3D
  source are explicitly fine (e.g. one maps NAF `na2d`, one the FDG mesh head, one the sparse
  up/down index maps). They don't contend.
- Authoring a stage's numpy ref + C++ graph header + standalone test SCAFFOLD (CPU-light file
  writing, no GPU) — a subagent can draft `naf_ref.py`+`naf_graph.hpp`+`naf_test.cpp` in isolation.

**DO NOT fan out the heavy/contending work — keep it SERIAL in the main loop:**
- GPU is SINGLE → only ONE ggml CUDA build+run at a time; no parallel GPU runs.
- Heavy torch/numpy oracles are ONE PROCESS AT A TIME (CPU/RAM/swap — running two thrashed the
  box; memory `feedback_no_heavy_parallel_subagents_during_gpu_test`).
- Agent subagents are NOT woken on `run_in_background` completion → they DEADLOCK on long
  GPU/torch jobs (memory `reference_subagent_background_stall`). Drive every long build/validate
  from the MAIN loop (`run_in_background:true`, you get the notification). No multi-agent Workflow
  for research (`feedback_no_workflow_for_basic_research`).

**Rhythm per stage:** (subagents, parallel) read source + draft ref/graph/test scaffold →
(main loop, serial) run the torch oracle once → build → run the ggml test on GPU → validate vs
golden → fix → document. Always re-validate a subagent-authored scaffold yourself before trusting it.

## Mentality / approach (obey)
1. **Correctness + precision FIRST; performance is a SEPARATE LATER phase.** Keep fp32.
   Slow-but-correct wins. Don't tune perf now.
2. **fp32 is the oracle, not the bf16 golden.** The shipped golden ran a bf16 torso; your
   fp32 port will differ from it by rounding (~98–99% voxel IoU, a few threshold flips).
   That is CORRECT. Validate against the **fp32 torch** path (run the real module on CPU,
   CUDA hidden) for the tight signal (~1e-5), and treat the bf16 golden as the E2E sanity
   (IoU/set comparison, not elementwise tol). This was the key M1 lesson.
3. **Golden-validate every op and every stage.** Per-op: numpy/C++ ref vs the real torch
   op on CPU (tight). Per-stage: vs the captured `golden_stages/` boundary. Never trust an
   unvalidated port. The proven recipe (used for sparse-conv, proj-cond, pixel_shuffle,
   DINOv3-RoPE, all of M1): port → validate bit/tol-exact → only then build on it.
4. **Reuse the base repo.** This worktree is a longcat-avatar.cpp (sd.cpp/ggml) fork; it
   already has conv3d, affine+non-affine LayerNorm, GroupNorm32, cross-attention, generic
   RoPE-apply, per-head RMSNorm, SiLU/GELU, Linear, timestep-embed, adaLN modulation,
   pixel-shuffle/`depth_to_space_3d`, GGUF load+convert. The inventory (file:line, symbols)
   is in `E2E-PORT-KICKOFF.md`. **Genuinely net-new in C++ for Stage-1: only `grid_sample`
   (bilinear/border/align_corners=False) and a `DINOv3` encoder.**
5. **Heavy CPU work is STRICTLY sequential — one multi-GB python/torch process at a time.**
   Do NOT fan out parallel torch subagents (it thrashed the box last session). Bounded
   read-only Explore subagents in parallel are fine. CUDA C++ builds fine here (docker
   `iter.sh` builders or the toolchain nvcc; sm_86); **NO Rust builds** on this host.
6. **No `pkill -f`** (kills own shell — kill by explicit PID). No `rm`-globs (delete named
   files; use fresh dirs). Run long jobs harness-tracked (`run_in_background: true`), not
   detached `&`.
7. **Keep docs + memory current as you go** (PROGRESS LOG in the kickoff, per-stage findings,
   memory `project_3dgen_cpp_port`). Worktree is UNCOMMITTED — leave it uncommitted unless
   there's a clear reason; `git add`/commit is fine to checkpoint but not required.

## The ladder (do in order; validate each before moving on)
### Rung-1 — Stage-1 in C++/ggml (image → coords)  ← START HERE
The math is fully de-risked by the `tools/m1_ref/` fp32 refs (your oracle). Port to
C++/ggml in the worktree:
- **Author the 2 net-new ops**, each validated GPU-free first then on GPU:
  - `grid_sample` (CUDA): bilinear, `align_corners=False`, `padding_mode=border`. Exact
    semantics + a CPU ref + a torch-validated golden are in `tools/proj_cond/` — validate
    the kernel against `tools/proj_cond/golden_ref/proj_case0/` then against the real
    DINOv3 map in `golden_stages/stage1_cond/`.
  - `DINOv3` ViT-L/16 encoder: adapt the base CLIP ViT; deltas (2D-axial RoPE on patch
    tokens only, LayerScale, 4 reg + cls, k_proj biasless, plain final `F.layer_norm`,
    erf-GELU) are spec'd in `DINOV3-ENCODER-SPEC.md`; the RoPE math is validated in
    `tools/proj_cond/test_dinov3_rope.py`.
- **Wire the rest** from base kernels: SS dense DiT (30× ModulatedTransformerCrossBlock,
  share_mod, qk-rms-norm, complex RoPE via saved `rope_phases`, ProjectAttention =
  per-block proj_linear + global cross-attn add), FlowEulerGuidanceIntervalSampler
  (params in CONFIGS-RESOLVED), SS VAE decoder (conv3d + `pixel_shuffle_3d` gather +
  ChannelLayerNorm32). All exact in `M1-STAGE1-PORT-SPEC.md`; mirror `tools/m1_ref/*.py`.
- **Weights → GGUF**: convert `ss_flow`, `ss_dec`, `dinov3` safetensors → gguf (base
  `src/convert.cpp`); `tools/m1_ref/weights/*_keys.json` lists exact tensor names/shapes.
  rope_phases is complex (export keeps it as [...,2] real/imag).
- **Validate**: C++ each op vs the `m1_ref` fp32 numpy outputs (tight), then C++
  image→coords vs the fp32 ref (should hit the same ~1120 coords) and vs the bf16 golden
  (~98–99% IoU). Build via the docker `iter.sh` builder (canonical for the real subsystem).

### M2 — Shape SLat LR @512 (sparse DiT)
Net-new vs M1: full **varlen sparse attention** (#8), **sparse 3D RoPE** (#10), sparse
ProjectAttention (#11), SparseLinear/SparseLayerNorm; plus the **NAF** conditioner
(#6, 9×9 neighborhood attn → proj C=2048). Sparse conv already done. Build the net-new
ops as validated refs first (numpy/C++ vs real torch op), then assemble the sparse DiT +
sampler. Validate vs `golden_stages/stage2_cond/` (cond) and `stage2_out/lr_slat_*` (the
denormalized [N,32] latent). Weights: export `slat_flow_img2shape_dit_1_3B_512`.
ElasticSLatFlowModel inference == `SLatFlowModel.forward` (structured_latent_flow.py).

### M3 — Upsample LR→HR + Shape HR @1024
3a: `shape_slat_decoder.upsample(×4)` → hr_coords (validate vs `stage3a_up/hr_coords`,
N≈382554 @ lr-res512) → quantize/unique to grid64 → M≈4734 tokens. Uses the sparse VAE
backbone: SparseConvNeXtBlock3d + SparseResBlockC2S3d (channel↔spatial, #14) + subdivision
prediction (#15). 3b: shape HR sparse DiT (= M2 arch, grid 64) → validate vs
`stage3b_out/shape_slat_*` [M,32]. Weights: `..._1024`, `shape_dec_next_dc_f16c32`.

### M4 — O-Voxel mesh extraction (biggest net-new; CUDA `_C.so`-only → golden-validate)
`flexible_dual_grid_to_mesh` (GPU hashmap voxel insert/lookup + edge-quad assembly +
quad_lerp diagonal). FDG head split: vertices=2·sigmoid(h[0:3])−0.5, intersected=h[3:6]>0,
quad_lerp=softplus(h[6:7]). Validate verts/faces vs `golden_stages/stage5_mesh/`
(vertices [1547076,3], faces [3251686,3], + 4 subs). Reconstruct semantics from the python
wrapper (`o_voxel/convert/flexible_dual_grid.py`) — no CUDA source in tree.

### M5 — Full geometry E2E → untextured GLB
Chain M1–M4 in C++; export verts/faces (OBJ/PLY ok; textured GLB bake = PHASE-2). Match
`stage5_final/`. Then: integrate as a longcat-avatar.cpp subsystem / koblem engine; THEN
the performance run (loosen precision, fuse, CUDA kernels — its own phase).

## Resources (all on disk)
- Goldens: `tools/sparse_spike/golden_stages/{pre,stage1_cond,stage1_ssdec,stage1_out,
  stage2_cond,stage2_out,stage3a_up,stage3b_cond,stage3b_out,stage4_*,stage5_mesh,
  stage5_final,configs,vram.json,stages_manifest.json}` + `cam.json`.
- fp32 refs + weights: `tools/m1_ref/{dinov3_proj,ss_dit,ss_vae_decode,stage1_e2e,
  torch_stage1_ref,export_weights}.py`, `tools/m1_ref/weights/*.npz` + `*_keys.json`.
- proj-cond + RoPE refs: `tools/proj_cond/`.
- Model + venv (for torch oracles, CPU): `/mnt/hdd/3d/avatar-shootout/Pixal3D` (+ `.venv`).
  Run torch CPU-only with `CUDA_VISIBLE_DEVICES=""`. Regenerate any golden via
  `tools/sparse_spike/golden_stage_runner.py` (one GPU decode).
- C++ build: docker `iter.sh` builder (canonical) or toolchain nvcc
  `/mnt/hdd/3d/avatar-shootout/toolchain` (sm_86, `-ccbin` its g++, its lib on LD_LIBRARY_PATH).

## Definition of done (this autonomous run)
Make real, validated progress down the ladder. Land at least Rung-1 (C++ Stage-1
image→coords matching the fp32 ref), further if it goes well. Leave: passing validations,
updated PROGRESS LOG + memory, and a crisp note of exactly where you stopped and why. Only
stop early if hard-stuck — and if so, document the blocker precisely.
