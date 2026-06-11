# HANDOFF — Pixal3D/TRELLIS.2 C++ port: next steps (autonomous)

You are continuing the C++/ggml port of the Pixal3D (TRELLIS.2) image→3D pipeline.
**Run fully autonomously — the owner is asleep, GPU + CPU are 100% free, don't wait on
them unless you are genuinely, hard-stuck** (a decision that reverses direction, or a
tool that's actually broken). Otherwise: decide, do, validate, document, continue.

## Where things stand (2026-06-12)
- **Spike**: submanifold sparse conv3d ported + validated (CPU bit-exact + CUDA Rung-1).
- **M0 DONE**: every stage boundary of one real decode (miku.png, res 1024) captured in
  `tools/sparse_spike/golden_stages/` (409 MB, gitignored) + per-stage VRAM (`vram.json`)
  + all model configs (`golden_stages/configs/`). This is the golden set for M1–M5.
- **M1 Stage-1 DONE at reference level**: fp32 numpy refs in `tools/m1_ref/`
  (`dinov3_proj.py`, `ss_dit.py`, `ss_vae_decode.py`, assembled in `stage1_e2e.py`)
  reproduce the real **fp32** torch pipeline EXACTLY (image→coords 1120≡1120, IoU 0.9859;
  cross-checked by `torch_stage1_ref.py` = real model+sampler). vs the **bf16** golden =
  98.6% IoU (11 coarse boundary voxels = bf16-torso rounding, expected, refined downstream).
- Read these first, in order: `E2E-PORT-KICKOFF.md` (philosophy + ladder + the PROGRESS
  LOG at top), `CONFIGS-RESOLVED.md`, `M1-STAGE1-PORT-SPEC.md`, `DINOV3-ENCODER-SPEC.md`,
  `E2E-PORT-MAP.md`, the 3 spike docs. Memory: `project_3dgen_cpp_port`,
  `feedback_correctness_before_perf`, `feedback_no_heavy_parallel_subagents_during_gpu_test`,
  `feedback_no_workflow_for_basic_research`, `reference_subagent_background_stall`.

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
