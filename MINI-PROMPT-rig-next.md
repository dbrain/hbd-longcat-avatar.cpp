# MINI-PROMPT — rigger: real-cond e2e + quality ladder (continue the image→rigged-avatar C++ port)

You are continuing the image→rigged-avatar C++/ggml port on an RTX 3060 (shared — ask before a long GPU
run if a bench may be active). **C++/docker build fine on this host; NO Rust builds.**

**READ FIRST, in order:**
1. `~/dev/longcat-sparse-spike/HANDOFF-rig-next.md` — your task, state, build/run commands, the remaining
   bits, levers, and gotchas. (This is the spec — follow it.)
2. `~/dev/longcat-sparse-spike/FINDINGS-rig-perf-pass.md` — what the just-finished perf pass did (so you
   don't redo it) and the measured floor.
3. memories: `project_rig_audit_gilly`, `project_3dgen_cpp_port`, `project_pixal3d_dense_to_lowpoly`,
   `feedback_no_declaring_winners`, `feedback_automate_no_manual_stitching`, `reference_subagent_background_stall`.

**Where things stand:** the C++ auto-rigger is **perf-optimized + forward-faithful + distribution-parity**
with Python (maxnew=2048 in 4.5 GB, ~6×/~30 tok/s, 0 runaways, forward tfprobe KL 0.0006/step — all DONE,
committed), BUT it is **NOT at structural quality parity**: on golden cond it runs busier/wider (J 53–78 vs
Python's clean tight 42–56 / 61-joint) with some asymmetric seeds — visible in `:8013/rig_perf.html` (C++
skeletons overlaid in the textured gilly mesh). That's the asterisk and your job #1. Precision is ruled out
(fp32-KV tested = no gain; keep f16-KV). Everything was validated on **golden-cond**.

**Your job (in order):**
1. **Close the rig-structure gap to Python (golden-cond) — the asterisk.** Forward is faithful (don't chase
   it), precision is ruled out (don't retry KV-precision/quant). The gap is in the BEAM SEARCH reaching/
   keeping Python's coherent ~61-joint hypotheses — recheck the C++ beam bookkeeping vs transformers 5.9.0
   `_beam_search` on the busier/malformed seeds, and check detok for zero-length/coincident joints. Judge by
   RENDER + rig_score, NO sticky-tape. Goal: golden-cond rigs tracking Python's tight 42–56 / clean 61.
2. **Real-cond end-to-end.** Run the rigger with the REAL C++ VecSet R1 encoder (`r1=real cond=real
   r1w=$A/r1w`, e2e dir `$A/samp`, `$A=/mnt/hdd/3d/avatar-shootout/rig_audit`) on a real generated mesh —
   NOT the giraffe golden. Sweep seeds, compare distribution + RENDER to golden-cond
   (`:8013/rig_perf.html`). R1 `mesh_cond` is cosine 0.9998 vs golden and that diff is load-bearing — if
   real-cond rigs are worse, the follow-up is tightening R1 fidelity (vecset encoder / FPS query sampling).
   Validate on a FRESH mesh (e.g. a new pixal3d output), not just gilly.
3. **PORT the quality-ladder models to C++/ggml — the SkinTokens way (GGUF-pack → ggml forward → validate
   bit-exact vs Python fp32 oracle), NOT Python/docker wiring.** Targets: **UltraShape** (dense-mesh clean),
   **P3-SAM / Hunyuan3D-Part** (part segmentation → per-part adaptive retopo; fingers = geometry), and
   **MoGe** (monocular geometry est. — port for camera **FOV** from the input image; public weights exist,
   GGUF-pack them). Use the existing port infra (`pack_gguf.cpp`, `gguf_reader.hpp`, `PIXAL3D_GGUF_DIR`).
   Then chain into the e2e so raw image → clean riggable textured GLB, fully C++ (no manual stitching).
4. Then **productionize as a koblem engine** (worker-isolation + routes + frontend, like sa3/flux2/qwen3-tts;
   `docker/rigprof/` is the build template).

**Rules:** judge rigs by RENDER + `rig_score`, not token match (generation is stochastic; render 2 seeds).
Stay faithful to HF — NO sticky-tape. Keep GPU runs on the MAIN loop (sub-agents stall on background GPU
jobs). Don't fan out heavy parallel torch/python during a GPU test. Clean up stray procs. Commit only when
the owner asks. The perf levers are tuned + measured at floor — don't re-litigate them (read FINDINGS first).
