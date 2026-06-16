# MINI-PROMPT — geometry-model eval (clean agent kickoff)

You are evaluating replacements for our TRELLIS.2/pixal3d geometry generator. Goal: pick the model that
produces the cleanest **riggable CHARACTER mesh** — separable fingers, smooth watertight surface — to feed
our (geometry-agnostic) SkinTokens rigger. Work from the **main loop** (GPU runs deadlock sub-agents).

READ FIRST:
- `HANDOFF-G-geometry-model-eval.md` — the VETTED 2026 shortlist with verified weights + links. DO NOT
  re-research the landscape; start from this lineup.
- `HANDOFF-F-skintokens-port-...md` — the rigger port state (R1 VecSet encoder + R2 Qwen3 + R5 detok done).
- memories: `project_3dgen_cpp_port`, `project_avatar_rig_path`, `feedback_no_build_on_server`
  (C++/CUDA builds OK on-server; NO Rust builds), `reference_subagent_background_stall`.

CONSTRAINTS:
- Weights-available is the only filter (ignore licenses entirely). Don't filter by VRAM — make big models
  work (offload/quant); current box is RTX 3060 12GB. **GPU is shared — ASK before heavy generation runs.**
- **Eval LIGHT, don't build a stack.** Per-model throwaway venv under `/mnt/hdd/3d/avatar-shootout/<model>/`
  (mirror the existing SkinTokens/Pixal3D venv pattern); use the repo's own inference script or a ComfyUI
  node if it has one. Do NOT build serving/training infra. The C++/ggml port comes AFTER we pick a winner.

TASK — for each shortlist model (priority order: **StdGEN, Step1X-3D, TripoSG, PartPacker**; baseline
**Hunyuan3D-2.1**; long-shot: verify whether **AniGen** has open weights at all):
1. **Recipe to run it**: repo + exact HF weight files + deps (pin what matters, e.g. flash-attn/torch),
   and the ONE command that goes `image → mesh (GLB/OBJ)`. Note real VRAM + wall-time on the 3060
   (quantize/offload if needed) and whether a ComfyUI node exists.
2. **Head-to-head**: pick ONE finger-heavy test character (reuse the Miku input image we already use, +
   maybe one more), generate a mesh from each model, export GLB, and render hand close-ups + silhouette
   (reuse `tools/m1_ref/cpp_port/render_*` / the `imadapt_web` model-viewer page). Put them side-by-side
   vs the TRELLIS.2 baseline.
3. **Judge** on: do fingers SEPARATE? is the surface clean/manifold (not voxel-staircased)? is it watertight
   and reasonable poly count? A-pose / part-separable bonus (rig-friendliness). RENDER and look — never call
   it good from logs/face counts.
4. **Wire-in cost**: for the top 1-2, sketch what a C++/ggml port entails (the `cpp_port` M1Harness + GGUF +
   validate-vs-fp32-oracle pattern). Note that the shortlist is mostly the VecSet/flow/SDF family = the SAME
   family as the already-validated R1 VecSet encoder, so much of the encoder/DiT machinery is reusable.

DELIVERABLE: `HANDOFF-H-geometry-eval-results.md` (repo root) with — the verified run recipe per model, the
rendered head-to-head with an honest finger/surface verdict, the WINNER (or "TRELLIS.2 still best, here's
why"), and the port plan for the winner. Commit any eval scripts on `spike/sparse-conv-3d`
(`tools/m1_ref/cpp_port/.gitignore` keeps only `*.cpp/*.hpp/*.sh/*.py`; .md docs at repo root).
Start by running StdGEN end-to-end on the Miku test image — that's the highest-signal first move.
