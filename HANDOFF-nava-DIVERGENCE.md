# ✅✅✅ RESOLVED 2026-06-04 — it was a DUMMY-CONTEXT path bug, NOT the forward/sampler.

**The render was silently denoising with a DUMMY (sin-based) cond context the whole time.**
`run_render` loaded the cond context with `load_or_dummy(".", o.context, ...)`, which builds the
path `"." + "/" + "/abs/path/dense_ctx.bin"` = `".//abs/path..."` — a *relative* path that never
exists — so it fell through to the sin-based `dummy()` context. (The existence check at the call
site tested `o.context` directly, which DID exist, so nothing erred. `ctx_neg` was loaded correctly
via `load_tensor_from_file_as_tensor`, so the render ran with **dummy-cond + real-neg**.) Result:
every clip was unconditioned garbage, and the dummy-cond−vs−real-neg CFG guidance was biased →
the latent std ran away (→1.88) → dark/incoherent.

**Everything the previous 4 sessions chased was a red herring.** The forward is bit-faithful to
PyTorch F32 (proved this session: blocks 96–122 dB, heads 80–85 dB, render-unpatchify 80 dB on the
REAL step-0 input). q8 and f16 forwards are both faithful. The schedule (extra_one_step=True), the
joint compute_va path, audio co-denoise — all fine.

**How it was found:** teacher-forced PyTorch's own bounded trajectory, ran the cpp forward on the
identical latents → forward faithful. Then per-block tap-diff of the *render's* forward vs the
*harness's* forward on byte-identical inputs showed `context_embedded` already diverged at the input
(render std 0.036 vs harness 0.21) → the render's text-embedding input was the dummy.

**FIX (working tree, UNCOMMITTED):**
- `examples/nava/main.cpp` `run_render`: load `ctx_pos` directly via
  `sd::load_tensor_from_file_as_tensor<float>(o.context)` (mirroring `ctx_neg`), instead of the
  broken `load_or_dummy(".", o.context)`.
- `src/nava.hpp` `build_graph`: benign hardening of the joint path (no debug taps when
  `return_joint`; `ggml_set_output` on `vel_v`/`vel_a`). Not load-bearing for the fix; safe to keep.

**Verified:** 256² frozen std now denoises bounded 0.976→0.685→0.868 (≈ PyTorch 0.976→0.66→0.825,
was →1.88). 832×480 q8 render = a coherent photorealistic man talking in a study (rgb mean 0.34,
full range; was a dark void mean 0.07). Clip on eye-test :8097 as `nava_FIXED_832`.

Render cmd that works:
```
LD_LIBRARY_PATH=/mnt/hdd/3d/avatar-shootout/toolchain/lib build-nava/bin/nava render --cuda \
  --gguf models/nava-dit-q8_0.gguf --vae models/wan2.2-vae-48ch-f16.gguf \
  --context /mnt/hdd/nava/cpp-runs/_ctx/dense_ctx.bin \
  --neg-context /mnt/hdd/nava/cpp-runs/_ctx/neg_ctx.bin \
  --audio-neg-context /mnt/hdd/nava/cpp-runs/_ctx/audio_neg_ctx.bin \
  --steps 10 --frames 13 --width 832 --height 480 --seed 42 --cfg 3.0 \
  --out-name X --runs-dir /mnt/hdd/nava/cpp-runs
```
NEXT (per owner): I2V clean-latent anchor (real product), Phase-3 audio (LTX VAE+vocoder), Q4_K, perf.

Tooling built this session (all in ~/dev/NAVA/): nava_teacher_drive.py (PyTorch teacher trajectory),
nava_cpp_forward_diff.py, nava_dump_f32_step0.py (TRUE-F32 ref via the .bfloat16()->.float() sed
patch). Diff: longcat-avatar.cpp/tools/nava_tensor_diff.py.

---

# NAVA cpp — sampler DIVERGENCE handoff (START HERE; supersedes the "head bug" hunt)
# (HISTORICAL — superseded by the RESOLVED section above; kept for context)

Branch `nava-port` in `/home/dbrain/dev/longcat-avatar.cpp`. Read `HANDOFF-nava.md` for the
full arch/forward spec (the "bible"). THIS doc is the current, corrected diagnosis of the one
remaining problem: **renders are incoherent (dark void / washed blur) because the iterated
flow-match trajectory diverges, even though the single forward is faithful.** The owner's bar:
**a coherent clip. Not partially working.**

## TL;DR (what is and isn't the bug — trust this, don't re-derive)
- The **forward is correct** per-block (65–91 dB vs PyTorch at every t, on clean AND drifted
  latents). The previous session's "head bug" was a **PHANTOM chasing STALE dumps** (traj_f16cpu
  is from a pre-21:01 binary; the current binary's cond forward is 50 dB faithful, head matches an
  independent f64 numpy head at 86 dB).
- The bug is a **small, DETERMINISTIC, ALGORITHMIC difference between the cpp forward and PyTorch's**
  (~11% systematic component per block, the rest token-dependent), invisible in the per-block diff
  because it sits **below the bf16 noise floor** (the PyTorch reference runs bf16 autocast). It
  **compounds** through the stiff 10-step 2-branch-CFG Euler sampler: the latent under-denoises
  (std stops dropping ~step 5, then the big low-σ steps blow it up; mean drifts negative → dark).
- **Per-step trajectory diff (cpp vs PyTorch, identical noise+ctx+recipe):** PSNR 47.8 dB (step 0)
  → 11.7 dB (step 9). cpp std 0.97→1.85 (explodes); PyTorch std 0.98→0.65→0.80 (denoises, coherent).
  Tool: `~/dev/NAVA/nava_sampler_compare.py` (now also dumps clean PyTorch per-step latents to
  `/mnt/hdd/nava/traj_cpp/py_step_NN.bin`).

## RULED OUT this session — do NOT re-investigate (each killed by a measurement)
- **Head** — math verified vs PyTorch `model_mm.py:1118`; numpy-f64 head reproduces cpp@86 dB; the
  50 dB head-vs-PyTorch gap is just bf16 rounding (PyTorch Head is bf16-hardcoded, line 1124-1127).
- **Audio coupling** — `NAVA_FREEZE_AUDIO=1` gives an IDENTICAL video divergence (std 0.974→1.880
  vs co-denoised 1.852). Audio is innocent. (Answers the owner's repeated "is audio messing with
  video" — no.)
- **RoPE** — an audit agent claimed a "2× dims" bug; FALSE. `rope_params(max,dim)` returns dim/2
  complex pairs (`model_mm.py:38-45`); cpp's 22-rotated/42-identity audio + [22,21,21] video match.
- **`compute_va` joint wrapper** — bit-identical to `compute()` even at L_audio=8 (PSNR 999).
- **text_embedding** — GELU is tanh both sides; `context_embedded` matches at 72 dB.
- **σ-schedule** — cpp `FlowMatchSched` matches PyTorch `flow_match.py` set_timesteps+step EXACTLY
  (`linspace(1,σmin,n+1)[:-1]`, shift `shift·s/(1+(shift-1)s)`, `x+v·(σ_next−σ)`, final σ=0).
- **Sampler stiffness / step count** — 30 steps is WORSE than 10 (more low-σ evals accumulate the
  error). UniPC (validated multistep solver, wired behind `NAVA_UNIPC=1`) gives the SAME explosion
  → the velocity is biased; no solver fixes a biased velocity.
- **Quant / weight precision** — q8_0 ≈ f16-CPU(f32 compute) ≈ identical divergence. **DECISIVE:
  PyTorch with `NAVA_fp8.safetensors` (fp8 — COARSER than our q8) is COHERENT** (it made the
  reference clip). Coarser weights cohere, finer-weight cpp doesn't ⇒ it is NOT weight precision;
  it is the cpp compute ALGORITHM differing from PyTorch's.

## The coherent REFERENCE (the target)
- PyTorch `euler_recipe.mp4` = a coherent man in a suit, study, talking. **Same recipe the cpp
  uses** (Euler, 2-branch CFG video=3/audio=2, `align_3d_cfg:false`, dense Chinese caption, 832×480,
  10 steps, seed 42). Made by `run_nava.sh` + `/mnt/hdd/nava/nava_run_euler.yaml` with the **fp8**
  ckpt. Copied to `/mnt/hdd/nava/cpp-runs/_REF_pytorch_euler/clip.webm` (visible on eye-test :8097).
- Uncond context = the **negative prompt** (pipeline default `negative_prompt_mode=True`,
  `pipeline_nava.py:401-408`), NOT zeros. cpp matches (uses neg_ctx.bin / audio_neg_ctx.bin).
- The fp8 ckpt `/mnt/hdd/nava/NAVA_fp8.safetensors` (6.9 GB) is the SAME DiT backbone (380 Linears),
  fp8-quantized — fits RAM easily and is the proven-coherent PyTorch path.

## NEXT STEPS (ranked — the owner wants it WORKING, doesn't care which)
1. **LOCALIZE the algo diff with a numpy-f64 block autopsy (RAM-cheap, DEFINITIVE).** The clean
   f32 PyTorch reference is RAM-blocked (f32 model = ~50 GB peak at load on this 31 GB box). Instead
   reimplement ONE DiT double block in **f64 numpy** (only that block's weights — tiny), feed it the
   cpp's ACTUAL block input (dump it), and diff **sub-op by sub-op** (qkv, qk-RMSNorm, joint-RoPE,
   softmax-scale, cross-attn, modulated FFN, gates) against the cpp's per-sub-op output. The cpp
   already taps block outputs via `LONGCAT_DUMP_DIR`; add finer taps as needed. The sub-op where cpp
   deviates from f64-numpy beyond f16 rounding (~60 dB) IS the bug.
   ALREADY CONFIRMED MATCHING this session (don't re-check): attention softmax scale = 1/√128 both
   sides (`ggml_extend.hpp:1336` vs PyTorch SDPA default, q_scale/softmax_scale=None); shared-FFN
   GELU = tanh both sides (`nava.hpp:394` `ggml_ext_gelu(.,true)` vs `model_mm.py:817`
   `GELU(approximate='tanh')`); qk-RMSNorm = full-dim 3072, applied pre-head-reshape, eps=block eps
   1e-6 both sides; AdaLN chunk order [shift,scale,gate]×2 matches; CFG combine + σ-schedule match.
   REMAINING suspects for the autopsy (subtler, in every block): **(a)** the JOINT-RoPE *application*
   inside attention — `gen_nava_joint_pe` interleaving/order + how it's applied to cat[q_vid,q_audio]
   (the pe CONSTRUCTION was reviewed, the APPLICATION path in ggml vs PyTorch `rope_apply_joint` less
   so); **(b)** `ggml_ext_timestep_embedding` numerics vs `sinusoidal_embedding_1d` (`model_mm.py:24`:
   half=128, `pow(10000,-arange(half)/half)`, **cos-first then sin** — verify ggml's arange divisor
   and cos/sin order EXACTLY; token-constant so it injects a per-channel-DC bias every block);
   **(c)** cross-attention to text context (q from norm3(x), k/v from context); **(d)** residual /
   gate accumulation order. NOTE: block-diff is ~89% token-dependent (bf16-floor) + ~11% SYSTEMATIC
   — analyze the systematic (per-channel-DC, token-averaged) component, not raw PSNR. (Earlier a
   "100% systematic" reading was a metric bug — averaging over the wrong axis; corrected to ~11%.)
2. **OR ship a working clip via I2V (the actual use case + robust).** Image+audio+prompt→video is
   what NAVA was built/evaluated for (all PyTorch evals are I2V). The first-frame **clean-latent
   anchor** (splice the input image's VAE latent at token 0; per-token timestep t=0 for the first
   frame's H'·W' tokens — `model_mm.py:1459-1463`) PINS the trajectory each step, so the small
   per-step diff can't run away. Currently STUBBED in `src/nava.hpp` (uniform-t, no anchor). This is
   the highest-probability path to a coherent clip even if the algo diff isn't fully closed.
3. Last resort: make the cpp compute norms/head/attention in **bf16** to mirror PyTorch (uncertain —
   f32-cpp also diverges, so weak evidence this is the lever; big invasive change).

## Tooling built this session (all working; paths are absolute)
- Build: `export PATH=/mnt/hdd/3d/avatar-shootout/toolchain/bin:$PATH;
  export LD_LIBRARY_PATH=/mnt/hdd/3d/avatar-shootout/toolchain/lib;
  cmake --build build-nava --target nava -j8`. Binary `build-nava/bin/nava`.
- **Render** (current; q8 fits 12 GB GPU, f16 OOMs): `NAVA_UNIPC=1? build-nava/bin/nava render --cuda
  --gguf models/nava-dit-q8_0.gguf --vae models/wan2.2-vae-48ch-f16.gguf
  --context /mnt/hdd/nava/cpp-runs/_ctx/dense_ctx.bin --neg-context .../neg_ctx.bin
  --audio-neg-context .../audio_neg_ctx.bin --steps 10 --frames 5 --width 832 --height 480 --seed 42
  --cfg 3.0 --out-name X --runs-dir /mnt/hdd/nava/cpp-runs`. Prints per-step latent std (watch it
  stay <1 vs run away). Frame: `ffmpeg -i clip.webm -vf select=eq(n\,8) -frames:v 1 f.png`.
- **Env flags added to `examples/nava/main.cpp` run_render:** `NAVA_UNIPC` (UniPC solver, per-stream,
  built+working but does NOT fix divergence), `NAVA_FREEZE_AUDIO` (freeze audio stream — proves audio
  innocent), `NAVA_DUMP_TRAJ=<dir>` (dump vid_noise/aud_noise/vel_vid_{cond,uncond,cfg}_NN/vid_step_NN),
  `NAVA_DUMP_LATENT`. Single-forward harness: `NAVA_VALIDATE_VA`, `NAVA_HEAD_F32`.
- **cpp single-forward + per-block taps:** `LONGCAT_DUMP_DIR=<out> build-nava/bin/nava
  models/nava-dit-f16.gguf <in_dir> <out>` where in_dir has video.bin[W,H,F,48]/audio.bin[128,L]/
  context.bin[4096,512]/timestep.bin[1] (sd.cpp int32 header). Reads L_audio from audio.bin (NOT
  hardcoded — feed an L=8 audio.bin to reproduce render conditions). Dumps every block + velocity.
- **PyTorch per-block ref:** `~/dev/NAVA/nava_bisect_ref.py <ctx.bin> <out_dir> [vid.bin] [t]`
  (parameterized: video latent + timestep; `NAVA_F32=1` runs f16-weights→f32 + no autocast, BUT the
  model hardcodes `.bfloat16()` inline in block forwards so it's not pure f32 without patching
  model_mm.py — see below). Standalone backbone-only (no umT5/VAE), ~13 GB / ~200 s load.
- **Diff:** `python tools/nava_tensor_diff.py <ref.npz> <cpp_dump_dir>` (per-tensor PSNR, normalizes
  double_blocks.N↔double_block_N; head_video↔velocity_video_patched NOT name-matched — compare by hand).
- **Trajectory diff:** `~/dev/NAVA/nava_sampler_compare.py` (256/F2, replicates cpp sampler in
  PyTorch from cpp's NAVA_DUMP_TRAJ dumps; per-step latent PSNR). **bf16 model.**
- **f32-exact patch trick (RAM-permitting):** `sed -i 's/\.bfloat16()/\.float()/g;
  s/dtype == torch.bfloat16/dtype in (torch.bfloat16, torch.float32)/g'
  ~/dev/NAVA/nava_src/models/nava/modules/model_mm.py` makes PyTorch f32 (the `amp.autocast('cuda')`
  contexts are CPU no-ops). Backup at `/tmp/model_mm.py.bak`. **CURRENTLY REVERTED (clean).** A full
  f32 load OOMs at 31 GB; would need mmap/layer-streaming or a >=64 GB box. The numpy-f64 block
  autopsy (step 1) avoids this entirely.
- Bisect inputs/dumps under `/mnt/hdd/nava/bisect/` (cond/uncond/lowt6 + out_*/ref_*). Contexts in
  `/mnt/hdd/nava/cpp-runs/_ctx/` (dense_ctx, neg_ctx, audio_neg_ctx, all raw umT5 [4096,512]).
- GGUFs in `models/`: nava-dit-{f16,q8_0,q4_0}.gguf, wan2.2-vae-48ch-f16.gguf. Eye-test page on :8097.

## Gotchas
- 256² is OOD (banded); always validate coherence at **832×480** (the bucket; PyTorch coheres there).
- GPU/CPU serial — ONE job at a time (2 = OOM). Never run a cpp F16 CPU harness alongside the
  PyTorch backbone (12.6 GB + 13 GB → OOM). cpp builds on this box are fine (no-build rule is Rust).
- The render's audio output is silent/stub (Phase 3, not blocking video coherence).
- Inject CURRENT DATE in any LLM prompt rewriting; the dense caption must be a dense zh caption.
