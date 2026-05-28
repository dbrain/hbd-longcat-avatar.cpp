# LongCat-Avatar.cpp — DiT PERF HANDOFF (lap-27, the levers that are left)

*Written end of lap-26 (2026-05-28). Lap-26 SHIPPED the cond-frame K/V cache: **bit-exact PSNR 99.00
at the 480/8-step benchmark, ~8% sampling win**. This doc is the next campaign. Read
`PERF-DiT-lap26.md` for the full lap-26 record (floor proofs, the cond-cache root-cause saga, the
offload-is-buggered finding).*

---

## ⏱️ FIRST FIVE MINUTES — stand up the eye-test server
```
cd ~/dev/longcat-avatar.cpp
python3 tools/serve_clips.py --dir build --port 8011   # host stdlib, no GPU, detached
```
→ **http://10.0.0.208:8011/** — renders land in `build/*.webm`. This is your **quality checkpoint /
final gate**, NOT a stop-and-wait: render an A/B pair after each win so if quality regresses you can
SEE which change did it. Don't pause the campaign waiting for eyeballs — keep working; the clips are
there when the owner looks.

---

## 🔥 MOOD / MANDATE — read twice
**NO QUITTING.** Nothing is "too hard." **Fork-class means we're getting our hands dirty — NOT "too
hard."** Forking a kernel, adding a ggml op, rewriting the allocator path — all fair game; the owner is
explicitly up for "rewrite ggml." The owner is an active, relentless steerer who WILL push back on any
"we're at the floor" / "too hard" / "good enough" — so **preempt it: just do the hard thing.** Every
"floor" claim in this codebase has been disprovable with a profiler (im2col "at roofline" @7% BW; VAE
conv "GEMM-bound" was 16.7%-occupancy; MUL_MAT "floored" was doing 7× redundant cond work). So "compute-
bound" is the START of the investigation. **Measure, never predict** — wall-clock guesses lied all
through this campaign; ncu + bit-exact A/B are ground truth. Quote numbers with step-count + PSNR.

**Gates (mandatory, every change):** bit-exact → `tools/clip_compare.py <base> <new>` reads **PSNR
99.00** (run on host; builder image has no numpy). Quality trade (needs owner OK) → ac16 0.83-0.84 flat
across ALL frames incl. the LAST (watch the watercolour melt) + 2 seeds. The benchmark is **480×832,
25f, --steps 8** — wins must show there (320p is a fast draft preview only; untrained res).

ONE GPU (RTX 3060, sm_86, 12 GB). Stop prod acestep/tts/llama before heavy runs; `docker rm -f
longcat-avatar-iter` strays; never two GPU jobs at once.

---

## What's DONE (lap-26)
- **Cond-frame K/V cache — SHIPPED, bit-exact, ~8% win.** The cond frame's 48-block forward is step-
  invariant (init_latent input + t=0); cache it, run steps 2-8 noise-only. Gated `LONGCAT_COND_CACHE=1`
  (default off — **flip to prod after the commit + owner eyeball**). RESIDENT-ONLY (see offload below).
  Forced FFN tile=2 when active (VRAM fit; bit-exact). Root cause of the hard 480 bug: final_layer's
  F16 output Linear is cuBLAS/M-dependent → must run final_layer on FULL tokens (done).

## The levers that are LEFT (ranked)
1. **Re-PROFILE FIRST (task #9).** Cond-cache removed ~8% → the bottleneck moved. Re-run
   `LONGCAT_OP_PROFILE=1` (steps 1) + ncu before grinding anything. It moved 4× during the VAE campaign;
   don't trust the stale ranking below until re-measured.
2. **Flash FA2/3-style kernel (task #7) — the biggest remaining kernel headroom, FORK-CLASS.** FLASH is
   33.7% of the step @ only **62.7% of its roofline**, **memory-latency-bound** (ncu: long_scoreboard
   62%, 2 blocks/SM, 16.6% occupancy). ggml's MMA kernel is already FA2-class (cp.async double-buffer),
   so beating it means: better K/V pipelining / warp-specialization / smem staging tuned to head_dim=128
   + huge L_k on sm_86. ncols1 tile sweep proved 64 optimal (bit-exact tuning exhausted). Realistic ~3-5%
   wall, uncertain — but this is exactly the "get hands dirty" target. Don't declare it floored without
   an ncu stall-reason teardown of YOUR kernel vs ggml's.
3. **Fused adaLN-modulate kernel (task #6) — modest.** Collapse modulate (LayerNorm+affine) + gate_add
   elementwise chains into fewer DRAM passes. Tail is mostly bandwidth-saturated (CONT 90% DRAM) so the
   ceiling is low (~few %). Bit-exact-able.
4. **⚠️ OFFLOAD IS BUGGERED — fix in a future lap (recurring).** `--offload-to-cpu` + cond-cache = PSNR
   12 dB garbage. The segmented graph-cut path (`bind_segment_cached_inputs`) mis-binds cross-`compute()`
   persistent leaves — same bug class as lap-20 view-output liveness ("why does this keep happening").
   Audit the graph-cut persistent-leaf binding + the cache_buffer realloc-per-compute. The resident path
   is clean; offload is the liability. Until fixed, the cond-cache prod gate MUST exclude offload.
5. **Prod-flip the cond-cache:** add the prod enable (config/env) + the `!offload` hard gate; commit; ship.

## What is NOT a lever (proven dead — don't re-burn)
- MUL_MAT precision (pf32) — MMQ ignores the prec flag. MUL_MAT itself = cuBLAS-corroborated floor
  (36 TF = 93% of cuBLAS-F16); the only waste was the cond recompute (now cached).
- Step count 4≈8 — OWNER-REJECTED (8 = distilled benchmark).
- Block-sparse attn — off-distribution (ref `proof_gen.py:61` disables BSA for the avatar).
- Cross-attn K/V cache — <1% (M=512 text / M=32 audio vs M=10920 FFN).
- Q3_K weights (+7% slower), CUDA graphs (~5ms, compute-bound), FP8 (no sm_86 compute), tail F16 norms.

## Method
Build: `~/dev/kobbler/docker/longcat-avatar-dev/iter.sh build` (~90s, CUDA builder image, sm_86).
Kernel changes → `ggml/` submodule (commit there, then `git add ggml` + parent — submodule-first).
Run sd-cli directly via `docker run … -e LONGCAT_… longcat-avatar-dev:builder /src/build/bin/sd-cli …`
(`iter.sh cli` doesn't pass env). Standard render flags in PERF-DiT-lap26.md. Profiling recipes (ncu,
op-profile, the cmp_tap.py bisect harness) in PERF-DiT-lap26.md "HOW TO PROFILE" + the lap-26 saga.
