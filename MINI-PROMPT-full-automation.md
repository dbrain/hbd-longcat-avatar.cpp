# MINI-PROMPT — finish the C++/ggml port stack (image→rigged-avatar), full automation

You are continuing a 3D-asset pipeline on the RTX 3060 (12GB, shared — coordinate). **READ FIRST, in order:**
1. `~/dev/longcat-sparse-spike/HANDOFF-PORTS-STACK.md`  ← THE LADDER (per-port: state, files, goldens, gotchas)
2. The per-port handoff as you reach each rung (`HANDOFF-F-skintokens-port-...`, `HANDOFF-RIGGING-skintokens.md`,
   the texturing writeup in memory `project_pixal3d_native_mesh_texturing`).
3. memories: `project_3dgen_cpp_port`, `project_pixal3d_native_mesh_texturing`, `project_avatar_rig_path`,
   `reference_subagent_background_stall`, `feedback_no_build_on_server`, `feedback_no_declaring_winners`.

**Goal:** work DOWN the port ladder in `HANDOFF-PORTS-STACK.md`, turning the Python/docker stages into
validated C++/ggml ports. Default order:
**(0) GLB reader → (1) native texturing [goldens banked] → (6) finish SkinTokens R3/R4/R6/R7 →
(5) per-part split/recombine → (2)/(4) UltraShape/P3-SAM → (3) MoGe.**
Don't boil the ocean — land ONE rung at a time, validated + rendered, before the next.

## Orchestration (keep the main context light)
- **Subagents (Agent tool, general-purpose) = author + compile + CPU-only checks.** Hand a subagent a
  single well-scoped unit (e.g. "port `mesh_to_flexible_dual_grid` into `$CP/voxelize.hpp`, build it via
  `./build.sh ... `, run the CPU self-test, return the diff summary + build/test status"). They read the
  Python ref, write the `.cpp/.hpp`, compile (foreground `build.sh` ≈ 90s, completes synchronously), and
  return a concise result — NOT file dumps. Spawn them in parallel only for independent units.
- **Main loop (you) = goldens + GPU runs + confirm.** Capture fp32 oracle goldens (the `capture_*.py` /
  the dockerized Python), run GPU validations + RENDER to confirm, and drive any iterative GPU fix.
- **HARD constraint:** sub-agents are NOT woken on `run_in_background` completion → they DEADLOCK on
  background/GPU jobs ([[reference_subagent_background_stall]]). So **all GPU runs and any
  `run_in_background` work stay on the MAIN loop.** Subagents do bounded, synchronous work only
  (read / write / `build.sh` / quick CPU test). Don't fan out heavy parallel torch/python during a GPU
  test ([[feedback_no_heavy_parallel_subagents_during_gpu_test]]).

## Where to start (rung 1 is mostly done already)
Native texturing's C++ tex DiT + tex decoder + cumesh bake ALREADY exist (the `pixal3d --tex` path). New
pieces only: the **`shape_slat_encoder`** (mirror of the ported `shape_dec` VAE; gguf-convert
`/mnt/hdd/pixal3d_tex/trellis2_4b/ckpts/shape_enc_*`) + the **`mesh_to_flexible_dual_grid`** voxelizer, then
wire encode→tex-DiT→dec→bake. **Validate the encoder bit-wise vs the banked goldens**
`/mnt/hdd/pixal3d_tex/golden_{69k,ultrashape,usdense}_*` (shape_slat feats+coords). The Python oracle runs
via `$CP/shootout/pixal3d_tex_run.sh` (set `GOLDEN_DIR=`); 1024 fits the 3060 in C++ for free (no autograd).

## Hard rules
- Validate every rung vs the **true-fp32 oracle** (`NVIDIA_TF32_OVERRIDE=0`, eager attn) — NOT tf32/bf16.
- **RENDER and look** — never judge geometry/texture from logs or face counts.
- **Big stuff (weights, goldens, GLBs) on `/mnt/hdd`** — the SSD is for final models (owner pref).
- C++/docker builds are fine on this host; **NO Rust builds**. GPU is shared — ask if a bench is mid-run.
- Generative stages (texture flow, UltraShape DiT) are **seeded, not bit-reproducible** — validate the
  deterministic encoders/decoders bit-wise, eyeball the sampled output (≥2 seeds for an A/B).
- Launch long GPU renders/captures **harness-tracked** (`run_in_background:true`) from the MAIN loop, not
  detached `&` ([[feedback_long_run_instrumentation]]). Clean up stray procs; chain build→run with `&&`.
- Commit per landed rung **only when the owner asks** (no Co-Authored-By). 1536 `--resolution` WIP stays
  uncommitted (0-face bug). `compare.html` + GLBs are gitignored dev artifacts.

Validate by RENDERING / golden-diffing, not from logs. One rung at a time. Keep the main thread thin —
delegate the writing+compiling, keep the confirming.
