# E2E port kickoff — Pixal3D/TRELLIS.2 → C++/ggml (image → mesh, geometry-first)

Handoff for a fresh agent taking over the **end-to-end functional port**. The
sparse-conv spike (the hardest novel primitive) is DONE and proven; now port the
rest of the pipeline stage-by-stage until image→GLB runs in C++ matching Python.

## North star (don't lose this)
Productionise Pixal3D as a **mute 3D-game-asset generator** — image → textured-
rigged GLB — as a koblem heavy GPU engine (worker-isolation, idle-unload, REST +
panel), like acestep/flux2/longcat. C++/ggml port is the target (no heavy Python on
the shared 3060). Pipeline: **Pixal3D mesh → SkinTokens rig → motion**. AniGen
(one-shot gen+rig) shares the same structured-latent shapes, so this port largely
transfers to it later. Full strategy: memory `project_3dgen_cpp_port.md`,
`project_avatar_rig_path.md`.

## The operating philosophy (from the owner — obey these)
1. **Functional E2E first, performance LAST.** Get "image → correct mesh GLB"
   working under C++, slow is fine. Only AFTER it's feature-complete do a
   performance run. See memory `feedback_correctness_before_perf`.
2. **Correctness == numeric precision.** Keep high precision (fp32/f64) during the
   functional port; validate each stage tight against Python goldens. Perf tuning
   later legitimately loosens precision (tf32/fp16 → ~1e-3); that's rounding, judged
   by E2E output, not breakage. Don't drop precision for speed now.
3. **VRAM is the real risk.** Replicate Pixal3D `low_vram`: sequential stages, each
   model resident on GPU only during its stage, offloaded after → peak = max(stage),
   not sum (Python fits the 3060 ~7.9 GB this way). **Instrument peak VRAM per stage
   from the first capture.** 7.5 GB co-resident is a perf-phase target, not now.
4. **Golden-tensor methodology** (proven in the spike): monkeypatch the Python model
   to dump each stage's input/output tensors → port the stage in C++/ggml → validate
   bit/tol-exact vs the goldens. The model run is GPU-but-one-time; C++ dev is then
   offline.
5. No multi-agent **Workflow** for research (`feedback_no_workflow_for_basic_research`)
   — but Explore/general-purpose sub-agents for bounded reads are encouraged. No
   `rm`-globs, no `pkill -f` (kills own shell — use PIDs). C++/cpp builds fine on this
   host; NO Rust builds here. Coordinate GPU with the owner (it's shared).

## What's DONE (the spike) — read these
- `HANDOFF-sparse-conv-spike.md` — full spike status.
- `PORT-SPEC-flexgemm-submanifold.md` — the sparse-conv algorithm + ggml mapping.
- `PERF-NOTES-sparse-conv.md` — perf intel (PARKED until the perf run).
- **Submanifold sparse conv3d is ported + validated**: `tools/sparse_spike/` —
  CPU f64 oracle (`sparse_conv.cpp`), CUDA implicit kernel (`sparse_subm_conv.cu`,
  fp32, ~1e-7 on all 9 real layers + synthetic), goldens + flex_gemm baseline. This
  is ONE of the net-new primitives; the rest are below.

## The pipeline (read `E2E-PORT-MAP.md` for the full stage-by-stage detail)
Live pipeline file: **`pixal3d/pipelines/pixal3d_image_to_3d.py`** (NOT
`trellis2_image_to_3d.py` — that's an older near-identical sibling). Geometry path:
PRE-A rembg/crop → PRE-B MoGe camera → (1) Sparse-Structure dense DiT @16³ + 3D-conv
VAE → coords → (2) Shape-SLat LR @512 sparse DiT → (3a) decoder.upsample×4 → HR
coords → (3b) Shape-SLat HR sparse DiT → shape_slat → (5) sparse-VAE decode →
O-Voxel mesh → GLB. (Stage 4 texture = PHASE-2, skip for first E2E.)

### Net-new primitives still to port (geometry) — see map for file:line
- submanifold sparse conv3d — **DONE** (spike).
- 2D bilinear grid_sample + camera unprojection (the "proj" image conditioning).
- DINOv3 2D-axial-RoPE + LayerScale (image-cond ViT).
- pixel_shuffle_3d (SS VAE decode).
- varlen + windowed/double-windowed **sparse attention** + sparse 3D RoPE (shape DiT).
- sparse up/downsample + spatial↔channel reshuffle + subdivision prediction.
- NAF guide-upsampler w/ 9×9 **natten** neighborhood attention (only grid-32/64 tiers
  — avoidable in the first slice).
- **O-Voxel flexible-dual-grid mesh extractor** — biggest net-new piece; CUDA source
  NOT in tree (compiled `_C.so` only) → MUST validate against golden tensors.

### Portable (your existing wheelhouse)
DINOv3 ViT, dense flow DiT, dense 3D conv, layernorm/matmul/standard attention, and
the **FlowEulerGuidanceIntervalSampler** (Euler flow-matching + CFG + guidance-
interval + std-rescale + Möbius t-warp; portable; pseudocode in the map).

### ⚠️ Gotchas (from the pipeline mapping)
- **`pipeline.json` is fetched from HF at runtime, not on disk** — holds exact channel
  widths, sampler `guidance_interval`, normalization mean/std, per-block attn_mode/
  share_mod. **Capture it first** (it's in the HF cache after a run, or print it).
- `o_voxel` and `flex_gemm` ship as compiled `_C.so` only (no source) → golden-validate
  the mesh extractor and (already done) the conv.
- "proj" image cond is injected as a **plain additive residual** (proj_linear + add),
  not cross-attention.

## Recommended milestone ladder (functional, golden-validated each step)
- **M0 — capture goldens for every stage boundary.** Extend the spike's
  `golden_hook.py` pattern to dump: preprocessed image, image-cond features, SS noise+
  coords out, shape-slat in/out, mesh verts/faces. + per-stage peak VRAM. One GPU run.
  Also grab `pipeline.json`.
- **M1 — FIRST SLICE: Stage 1 (grid-16, no-NAF) E2E.** Image → SS dense DiT (sampler)
  → SS VAE decode (pixel_shuffle_3d) → coords. Exercises dense DiT + proj grid_sample +
  the sampler with NO sparse-attention/NAF/mesh dependency. Smallest real E2E loop.
- **M2 — Shape SLat sparse DiT** (sparse attention + sparse RoPE) → match shape latent.
- **M3 — Shape sparse-VAE decode** (conv DONE + up/down + subdivision) → match substruct.
- **M4 — O-Voxel mesh extraction** → match verts/faces → write untextured GLB.
- **M5 — wire full geometry E2E**: image → untextured mesh GLB, numerically matching
  Python. ← "functionally complete" geometry milestone.
- **M6+** — texture branch (NATTEN etc.); then integrate into longcat-avatar.cpp +
  docker build; THEN the performance run.

## Environment / how to run (all verified this session)
- **Model + venv**: `/mnt/hdd/3d/avatar-shootout/Pixal3D` (+ `.venv` py3.10, flex_gemm
  installed, only-backend). Run a decode: `run_pixal3d.sh <img> <res>` (low_vram).
  Test image e.g. `/mnt/hdd/3d/avatar-shootout/assets/miku.png`.
- **C++ port worktree**: `/home/dbrain/dev/longcat-sparse-spike` (branch
  `spike/sparse-conv-3d`, off longcat-avatar.cpp@5e26fc5; UNCOMMITTED). ggml submodule
  checked out. The sd.cpp/ggml base has DiT/ViT/VAE/attention kernels to reuse.
- **Host has NO system nvcc** → use the toolchain: `/mnt/hdd/3d/avatar-shootout/
  toolchain/bin/nvcc` (CUDA 12.4, `-arch=sm_86 -ccbin <toolchain>/bin/g++`), run with
  `LD_LIBRARY_PATH=<toolchain>/lib`. Compile is GPU-free; see
  `tools/sparse_spike/run_bench.sh` for the pattern. Canonical integration build is
  the longcat-avatar **docker** builder (owner prefers docker for the real subsystem).
- **GGUF**: NOT needed yet — we validate ops against captured `.npy` goldens, not by
  loading the model in C++. GGUF (safetensors→GGUF) only matters once running the
  whole pipeline in C++ (a later milestone).
- Goldens so far: `tools/sparse_spike/golden_model/` (real conv layers, gitignored
  2.1 GB) + `flexgemm_timing.json` + 3060 autotune cache.

## PROGRESS LOG

### 2026-06-11 (latest) — M1 Rung-0 reference modules VALIDATED (numpy fp32, CPU)
Per-module fp32 numpy references in `tools/m1_ref/` (weights exported safetensors→npz
via `export_weights.py`: ss_flow/ss_dec/dinov3 + `*_keys.json`; rope_phases kept complex
as [...,2]). Each validated against the real torch math:
- **SS VAE decode** (`ss_vae_decode.py`): z_s golden → ss_logits (median abs 1.75e-2 vs
  fp16-torso golden) → **coords SET-EQUAL to golden, N=1126** ✅ (self-verified).
- **SS DiT + FlowEuler sampler** (`ss_dit.py`): numpy `model_forward` vs the REAL
  `SparseStructureFlowModel` on CPU (single forward, shared cond) = **maxabs 1.5e-5** ✅
  — proves the block math (complex RoPE, qk-rms-norm, share_mod per-block modulation add,
  ProjectAttention, 30 blocks). E2E 12-step-vs-golden tail is correct-but-slow naive numpy;
  killed mid-run to spare swap (cross-check already proved correctness; noise repro =
  `torch.manual_seed(42);randn(1,8,16,16,16)`).
- **DINOv3+proj** (`dinov3_proj.py`): image→(z_global,z_proj) vs golden = maxabs 1.5e-5 ✅
  (reuses validated proj_cond_ref + dinov3 RoPE; gotchas: RoPE on patch tokens only,
  k_proj biasless, plain final F.layer_norm, erf-gelu). [re-confirm in main loop pending]
Gotchas captured in each file's report.

**M1 STAGE-1 E2E COMPLETE ✅ (reference level, 2026-06-12)** — `stage1_e2e.py` chains
image → dinov3_proj → ss_dit(sampler) → ss_vae_decode → coords with NO golden inputs
(only the preprocessed image + cam scalars). Result: cond vs golden 1.5e-5; z_s vs golden
median 3.6e-3 / maxabs 1.23; **coords 1120 vs golden 1126, IoU 0.9859**.
DEFINITIVE fp32 closure (`torch_stage1_ref.py`): the REAL torch model + REAL
FlowEulerGuidanceIntervalSampler in fp32 (same seed-42 noise + golden cond) produces the
**IDENTICAL** result — z_s maxabs 1.226/median 3.607e-3, **coords 1120, IoU 0.9859**. So
the fp32 path (mine == torch) lands on 1120; the golden's 1126 is purely its **bf16 torso**
(11 coarse res-32 boundary voxels flip at the occupancy threshold — bf16 rounding, not a
bug; these get refined away in Stage 2/3). The numpy reference is bit-faithful to the real
fp32 pipeline incl. the 12-step sampler. naive-numpy E2E = ~63 min (attention/python-loop
bound); torch fp32 = ~9.4 min. Perf is the C++/ggml port's job, not the ref's.

**REMAINING:** C++/ggml port of Stage-1 (Rung-1 — reuse base kernels per the inventory;
only grid_sample + DINOv3 encoder are genuinely new); then M2 (shape sparse-DiT, full
varlen sparse attn + sparse RoPE) → M3 (shape sparse-VAE: sparse conv DONE + up/C2S +
subdiv) → M4 (O-Voxel mesh) → M5 (full geometry GLB). All M2+ goldens already captured in
golden_stages/. NOTE: ran two heavy torch subagents in parallel earlier → contended w/
owner's GPU test + swap → [[feedback_no_heavy_parallel_subagents_during_gpu_test]]; heavy
CPU work strictly sequential/one-at-a-time.

### 2026-06-11 (later) — M0 CAPTURED ✅ (GPU run, one decode, miku.png, res 1024)
`golden_stage_runner.py` ran clean (exit 0, full decode + GLB export). All stage
goldens in `tools/sparse_spike/golden_stages/` (409 MB, gitignored) + `vram.json` +
`stages_manifest.json` + copied `configs/`. Every shape matched CONFIGS-RESOLVED
predictions (slat=32ch, SS proj dense[1,4096,1024], shape proj 2048, z_s=8). Goldens
verified loadable, value ranges sane (coords in-grid, verts in aabb[-0.5,0.5]).

**Per-stage peak VRAM (MiB alloc / reserved) — the low_vram budget the port must hold:**
| stage | alloc | reserved | what |
|---|---|---|---|
| stage1_cond | 1212 | 1244 | DINOv3 ss (grid16, no NAF) |
| stage1 | 2805 | 2848 | SS flow(30blk) + VAE decode |
| stage2_cond | 4384 | 4628 | DINOv3 shape_512 + NAF |
| stage2 | 2759 | 4630 | shape LR flow |
| stage3a | 1433 | 1562 | upsample decoder |
| **stage3b_cond** | **6334** | **7632** | DINOv3@1024 + NAF (HR cond) |
| **stage4_cond** | **6339** | **7634** | tex cond (PHASE-2) — **PEAK** |
| stage4 | 2993 | 7642 | tex flow |
| stage5_decode | 2371 | 2624 | shape decode + O-Voxel mesh |
→ **Peak ≈ 6.3 GB alloc / 7.6 GB reserved at the DINOv3@1024+NAF conditioner** (matches
the map's "Stage 3b heaviest"). Geometry-only (skip tex stage4) peak is the same 6.3/7.6
at stage3b_cond. Perf-phase target 7.5 GB co-resident is plausible. NAF is the spike.

**Real data-flow (miku.png) — golden token/voxel counts for sizing the port:**
SS coords 1126 @res32 → LR slat[1126,32] → upsample×4 → hr_coords[382554,4] @lr-res512
→ quantize/unique → **4734 tokens** @grid64 → HR shape_slat[4734,32] → mesh 1,547,076
verts / 3,251,686 faces (pre fill_holes); final 1,549,929 / 3,269,048 (post + GLB).
subs (4 levels): 4734 → 20533 → 87442 → 367673 voxels. (hr_coords N=382554 ≈ the
sparse-conv spike's heaviest layer N=382533 — consistent.)

### 2026-06-11 (earlier) — CPU prep (GPU was busy; no GPU touched)
- **M0 dumper BUILT (ready to fire, GPU-gated):** `tools/sparse_spike/golden_stage_hook.py`
  + `golden_stage_runner.py`. Monkeypatches the pipeline to dump EVERY stage boundary
  (pre image, SS cond global/proj, z_s, ss_logits, coords, lr_slat, hr_coords,
  shape_slat, tex_slat, mesh verts/faces+subs) + **per-stage peak VRAM** (`vram.json`)
  + copies all configs. Saves incrementally (survives a PHASE-2 GLB-export crash).
  Both byte-compile clean. **To run (coordinate GPU first):**
  `cd /mnt/hdd/3d/avatar-shootout/Pixal3D && source .venv/bin/activate &&
   ATTN_BACKEND=sdpa python <spike>/tools/sparse_spike/golden_stage_runner.py
   --image /mnt/hdd/3d/avatar-shootout/assets/miku.png --resolution 1024`
- **`pipeline.json` + ALL per-model configs FOUND ON DISK** (no GPU/network needed):
  `~/.cache/huggingface/hub/models--TencentARC--Pixal3D/snapshots/0b31.../{pipeline.json,
  ckpts/*.json}`. Open Questions #1/#2/#3 from the map are RESOLVED → **`CONFIGS-RESOLVED.md`**.
  Key map corrections: **shape/tex SLat latent = 32 ch (not 8)**; SS latent z_s = 8;
  ALL flow models **pe_mode=rope, share_mod=true, attn_mode='full'** (NO windowed attn
  on geometry path → primitive #9 NOT needed for M1–M5); flow class is
  `ElasticSLatFlowModel` (inference == `SLatFlowModel.forward`); decoder blocks =
  `SparseConvNeXtBlock3d` + `SparseResBlockC2S3d` (channel↔spatial, not pixel-shuffle);
  tex guidance_interval = [0.6,0.9] (vs [0.6,1.0]).
- **M1 spec WRITTEN:** `M1-STAGE1-PORT-SPEC.md` — exact block math (ModulatedTransformer
  CrossBlock + ProjectAttention), proj grid_sample/unproject recipe, SS VAE +
  pixel_shuffle_3d, sampler params, reuse inventory. Mechanical to port once M0 goldens
  exist. NOTE: norm2 in the DiT block is AFFINE (norm1/norm3 non-affine).
- **Base-repo kernel inventory (Explore):** almost ALL of M1 is reusable in the
  sd.cpp/ggml base — `ggml_ext_conv_3d`, affine+non-affine LayerNorm (`ggml_ext_layer_norm`),
  `GroupNorm32`, cross-attn (`ggml_ext_attention_ext`), generic RoPE-apply
  (`Rope::apply_rope` / `ggml_rope_pe`), per-head `RMSNorm`, SiLU/GELU, `Linear`,
  `ggml_ext_timestep_embedding`, adaLN modulation (`z_image.hpp`), pixel-shuffle
  (`PixelShuffleND`) + `depth_to_space_3d`, GGUF load+convert (`src/convert.cpp`).
  **GAPS / net-new to author:** (1) **grid_sample** (bilinear/border/align_corners=False —
  no dedicated op; only `ggml_upscale`); (2) **DINOv3 encoder** (only CLIP ViT exists —
  template to adapt: add 2D-axial RoPE + 4 reg tokens + LayerScale); (3) ProjectAttention
  (trivial compose = cross-attn + per-block `proj_linear` add). ggml submodule @ 19727d01.
- **PROJ-COND net-new RISK RETIRED (GPU-free), `tools/proj_cond/`:** the camera-unproject
  is host scalar arithmetic (grid + 3 camera scalars are fixed per stage → it produces the
  grid_sample coords on the host; only the bilinear gather needs a GPU kernel). Built
  numpy ref (`proj_cond_ref.py`) + **validated bit/tol-exact vs the REAL torch `ProjGrid`
  on CPU** (`test_proj_cond.py`, CUDA hidden) across SS grid16 / shape grid32 / grid64 —
  **maxabs ~1e-5** (fp32 ordering noise). C++ mirror (`proj_cond.cpp`, 4x4 inv + CPU
  grid_sample) vs the dumped torch golden = **maxabs 1.53e-5 PASS**. So the proj geometry +
  bilinear sampler are locked; the only remaining proj work is a CUDA grid_sample kernel
  (validate vs `golden_ref/proj_case0/` + the M0 `stage1_cond` real DINOv3 map).
- **NOT touched:** no DiT/VAE C++ yet (correctness-first needs the M0 goldens — author
  after the GPU capture). Nothing committed (worktree still UNCOMMITTED).

## First actions for the fresh agent
1. Read `E2E-PORT-MAP.md` (the detailed stage map) + this doc + the 3 spike docs.
2. M0: write the staged golden dumper (generalize `tools/sparse_spike/golden_hook.py`),
   run ONE Pixal3D decode to capture all stage boundaries + `pipeline.json` + per-stage
   VRAM. (Needs GPU once — coordinate with owner.)
3. M1: port Stage 1 (grid-16 SS) and validate vs golden. Then march the ladder.
Keep the handoff docs updated as you go. Save durable findings to memory
(`/home/dbrain/.claude/projects/-home-dbrain-dev-kobbler/memory/`).
