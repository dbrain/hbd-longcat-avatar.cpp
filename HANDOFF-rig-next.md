# HANDOFF — rigger: the remaining bits (after the perf pass)

**Date:** 2026-06-17 · **Worktree:** `~/dev/longcat-sparse-spike` · **$CP =** `tools/m1_ref/cpp_port` ·
**$A =** `/mnt/hdd/3d/avatar-shootout/rig_audit` · **GPU:** one RTX 3060 (shared — ask if a bench is mid-run).
**C++/docker builds fine here; NO Rust builds.** **READ FIRST:** memory `project_rig_audit_gilly`,
`FINDINGS-rig-perf-pass.md` (perf detail), and `$A/DIAGNOSIS-decode.md` (the journey).

## State you're inheriting
The C++ auto-rigger (`skintokens_e2e`, R1→R3→R5→R4→R6) is **perf-optimized (DONE)** and **forward-faithful**,
but **NOT yet at structural quality parity with Python** — that's your job #1 below.
- **Perf (DONE — don't redo):** see below.
- **Forward + distribution parity (DONE):** beam = faithful transformers 5.9.0 `_beam_search`
  (`rig_beam_generate.hpp`), `prec=fp32`, forward proven bit-faithful (tfprobe KL 0.0006/step). The C++
  reproduces Python's *distribution* (termination rate, joint-count range, ~2 malformed). Don't add
  sticky-tape (no joint-caps / eos-prefs — tried + reverted).
- **THE ASTERISK (NOT done):** the C++ does NOT reproduce Python's clean **61-joint structure**. On golden
  cond it runs **busier/wider (J 53–78 vs Python's tight 42–56)** with some asymmetric/degenerate seeds —
  visible in the render (`:8013/rig_perf.html`, C++ skeletons overlaid in the textured gilly mesh: denser
  bones, occasional zero-length joints). This is a **beam-search / structure** gap, NOT precision: KV
  precision is ruled out (fp32-KV @2048 tested = no gain, 3 malformed, costs 2.4 GB → keep f16-KV).
- **Perf (this pass):** maxnew=2048 budget now RUNS in **4.5 GB** (was ~14 GB OOM) at **~30 tok/s clean
  (≈6×)**, **0/10 runaways**. Levers (all in `qwen3_decode.hpp` / `qwen3_batched.hpp` /
  `rig_beam_generate_batched.hpp`): f16-KV (`RIG_KV_F16`, default on) + shared-prefix KV (bit-exact) +
  2-segment batched-beam decode (default; `RIG_BEAM_SEQ=1` = old sequential A/B) + single-buffer in-place
  reorder. ncu-proven the decode is compute/latency-bound at M=10 — CUDA graphs measured dead; weight-quant
  (`RIG_W_F16`, default OFF) saves 0.69 GB but regresses quality. All validated mathologically vs the
  sequential oracle + Python @2048 (judge rigs by RENDER, not token-match).
- **Visual check-off:** `http://10.0.0.208:8013/rig_perf.html` (4 clean perf rigs + Python ref, skeleton-in-
  mesh) and `rig_compare.html` / `skel_ab.html`. Server: `python3 -m http.server 8013 --directory $A/web`.
- **Prod build template:** `docker/rigprof/` (reusable image `rigprof:12.4`; `run.sh` mounts worktree+hdd
  with `--gpus all --cap-add SYS_ADMIN`; `build_in_container.sh` builds ggml+binary in-container — needed
  because the host GLIBC 2.43 won't run in stock containers, and it's how prod must build).

## Build + run
- Build: `cd $CP && ./build.sh skintokens_e2e cuda` (+ `./build.sh rig_score cuda` for the scorer).
- Golden-cond run (the validated path):
  ```
  PIXAL3D_GGUF_DIR=/mnt/hdd/3d/avatar-shootout/_weights/skin_vae_gguf \
    ./skintokens_e2e $A/golden_gilly $A/inputs/qwen3_w /tmp/o.glb cuda beam prec=fp32 seed=1 \
       r1=banked cond=banked beams=10 maxnew=2048
  ./rig_score /tmp/o.glb 55
  ```
- Docker (prod-shaped) build+run: `./docker/rigprof/run.sh bash /work/docker/rigprof/build_in_container.sh`
  then `./docker/rigprof/run.sh ./skintokens_e2e_docker <args...>`.
- Sweep harness: `$A/perf/{sweep_2048.sh, batched_validate.sh}` (10-seed J/eos/rig_score/VRAM tables).

## THE REMAINING BITS (your work, roughly in order)
1. **Close the rig-structure gap to Python (golden-cond) — THE asterisk, do this first.** On golden cond the
   C++ rig is busier/wider than Python's clean 61 (see render). Forward is proven faithful (don't chase it)
   and precision is ruled out (fp32-KV tested = no gain — DON'T retry KV-precision/quant levers). So the gap
   is in the BEAM SEARCH reaching/keeping Python's coherent ~61-joint hypotheses: compare the C++ beam
   bookkeeping (selection, length-norm, finished-pool, the 2*num_beams joint draw) against transformers
   5.9.0 `_beam_search` once more on the busier/malformed seeds; and check the detok for zero-length/
   coincident joints (e.g. perf_s9 had 4 joints at local origin). Judge by RENDER + `rig_score` (no
   sticky-tape). Goal: golden-cond rigs that track Python's tight 42–56 / clean 61-joint structure.
2. **Real-cond end-to-end.** EVERYTHING above used golden-cond (banked `learned_mesh_cond.npy`,
   R3 isolated from R1). The real path runs the C++ VecSet R1 encoder on actual mesh verts/normals:
   `r1=real cond=real r1w=$A/r1w` with e2e dir `$A/samp` (a real generated mesh, NOT the giraffe golden).
   R1 `mesh_cond` is **cosine 0.9998** vs golden — the DIAGNOSIS notes this tiny diff is load-bearing
   (flipped 49→109 joints in early tests). So: run the rigger on real cond, sweep seeds, compare the
   distribution/render to golden-cond. If real-cond is worse, the follow-up is **R1 mesh_cond fidelity**
   (tighten the vecset encoder / FPS query sampling toward 0.9999+; FPS `rng.choice` is currently a
   deterministic mt19937 replica of Python's nondeterministic draw — see `skintokens_e2e.cpp` R1=real path
   + `vecset_encoder.hpp`). Validate on a NEW mesh (e.g. a fresh pixal3d output), not just gilly.
3. **PORT the quality-ladder models to C++/ggml — the SkinTokens way, NOT Python wiring.** The end goal is
   pure C++/ggml (Python only ever wraps *unported* models — `feedback_automate_no_manual_stitching`). Port
   each the way SkinTokens/pixal3d were ported (see [[project_3dgen_cpp_port]] A2): `pack_gguf.cpp` packs the
   weights to GGUF (npy-vs-gguf bit-identical), a ggml forward in `$CP`, `gguf_reader.hpp` + env
   `PIXAL3D_GGUF_DIR`, validated bit-exact vs the Python oracle (fp32 oracle, NOT a tf32/bf16 golden — see
   the project_3dgen gotchas). Targets:
   - **UltraShape** (dense-mesh clean) and **P3-SAM / Hunyuan3D-Part** (part segmentation → per-part adaptive
     retopo; fingers are geometry not normals) — currently Python/docker only
     ([[project_pixal3d_dense_to_lowpoly]], [[project_pixal3d_retopo_manifoldplus]]). Port both to C++/ggml.
   - **MoGe** (monocular geometry estimation) — port for **camera FOV** estimation from the input image (MoGe
     predicts an affine-invariant point map + FOV; feeds a correct camera/FOV prior into the image→3D front
     end). Public MoGe weights exist and are presumably packable — source them, GGUF-pack, port the forward,
     validate vs Python.
   Then chain the ports into the e2e so raw image → clean riggable textured GLB, fully C++ (no manual stitching).
4. **Productionize as a koblem engine** (the broader [[project_3dgen_cpp_port]] goal): worker-isolation +
   routes + frontend, like the other cpp ports (sa3/flux2/qwen3-tts patterns). The `docker/rigprof` image
   is the build template. Off-server Rust build.

## Levers / gotchas
- Envs: `RIG_KV_F16` (default on; =0 for f32 KV A/B), `RIG_BEAM_SEQ=1` (sequential decode), `RIG_W_F16`
  (f16 weights — OFF; regresses quality, VRAM-only), `RIG_LOGIT_PROBE=1` (step-0 logit parity probe).
- Generation is stochastic + fp32≠bf16: judge by GEOMETRY (render + rig_score), not token match. Two seeds.
- Drive GPU runs from the MAIN loop; sub-agents stall on `run_in_background` GPU jobs ([[reference_subagent_background_stall]]).
- Paged/trie KV (the absolute ~2.5 GB VRAM floor) is documented in FINDINGS but NOT worth building (custom
  CUDA kernel, rigger-only, small/negative speed; we're already at 4.5 GB). Skip unless VRAM gets critical.
- Commit only when the owner asks (no Co-Authored-By). The perf pass + rig pipeline are committed as of this
  handoff; `ggml` submodule dirtiness is just the generated `build-cuda-docker/` dir — don't commit it.
