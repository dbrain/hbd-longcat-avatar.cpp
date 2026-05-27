# LongCat-Avatar.cpp — DiT / SAMPLING PERF HANDOFF (Lever 3, the 77% — START HERE)

*Self-contained. The VAE phase (laps 13–25) is done — see `HANDOFF.md` for the VAE levers + the
conv-3d-direct post-mortem. This doc is the next campaign: the **DiT sampling loop, 77% of the clip
wall**. Written end of lap-25 (2026-05-28).*

---

## ⏱️ FIRST FIVE MINUTES — stand up the eye-test server (do this NOW, before anything)

The owner wants to eyeball clips without asking. This gets dropped from every handoff and they always
have to stop and request it. Do it first, tell them the URL in your first message:

```
cd ~/dev/longcat-avatar.cpp
python3 tools/serve_clips.py --dir build --port 8011   # host stdlib, no GPU, runs detached
```
→ **http://10.0.0.208:8011/** — renders land in `build/*.webm`. (For live re-rendering instead:
`ITER_PORT=8095 iter.sh serve`.) Generate an 8-step baseline clip early so there's something to A/B
against (`v8_base.webm` from lap-25 may still be in `build/`).

---

## 🔥 MOOD / MANDATE — read this twice, it is the whole job

**Go nuts. No quitting without 100% proof that the floor is the floor — in every sense.** This codebase
has had agents (me included) confidently declare a floor that wasn't there *over and over*: im2col was
"at the roofline" at **7%** of memory bandwidth; the VAE conv was "GEMM-bound / 3× off cuBLAS" when ncu
showed it was **occupancy-bound with the tensor cores 92% idle**. Every "we're at the floor" claim so far
has been disprovable with a profiler. So your bar for "done" is not "it's compute-bound." Your bar is:

> **If asked "what's left to tune?", the answer is a confident *nothing* — with proof for every word.**

"Compute-bound" is the *beginning* of the investigation, not the end. When you hit it, you must answer,
**with measured evidence, all of these:**

1. **Roofline of *which* kernel?** Name it. Is it even the right kernel for the shape (cuBLAS/MMQ are
   NOT optimal for every M/N/K — tall-skinny, tiny-batch, and odd head_dims all have regimes where a
   hand kernel wins)? What % of *its* roofline is it actually at (ncu `sm__throughput` /
   `*_tensor_op_hmma` / `dram__throughput`)? 93% of the wrong kernel's roofline is still a loss.
2. **Is any of that compute WASTED?** This is the lever a roofline *cannot see*. Is every FLOP needed?
   - **Recompute across steps:** the graph is rebuilt + recomputed all 8 DMD steps. Anything
     step-invariant being recomputed is pure waste (→ cross-attn KV, below).
   - **Work elimination:** dense self-attention computes all N² interactions; LongCat's own design keeps
     **~12.5%** of attention blocks (the paper's "<10% of dense"). The other 87.5% is wasted compute the
     roofline counts as "busy." (→ block-sparse, below.)
   - **Thrown-away results:** padding, masked positions, over-wide tiles, `ggml_cont` round-trips.
3. **Is the precision justified?** Every `Linear` here forces `GGML_PREC_F32` (`pf32`) for bf16 parity.
   That ~doubles matmul cost. Is it load-bearing on *every* matmul, or just the few with overflow risk?
   (The VAE conv shipped F16 accumulation at 40 dB — measure the same here, gated.) Don't assume; prove.
4. **Is it actually saturating, or stalled?** ncu occupancy + stall reasons. The VAE conv *looked*
   compute-bound and was 16.7%-occupancy latency-bound. A hot kernel at 40% occupancy has 2× on the table.
5. **Is it even dispatched the way you think?** Is flash-attn actually taken, or did a gate
   (`can_use_flash_attn`, `L_k % 256`) silently fall back to the materialized-scores path? Verify in ncu
   that the kernel you think is running is the one running.

Only when **all five** are answered "yes, proven optimal" for the top ops is it the floor floor. Measure,
never predict. Quote measured numbers with step-count and say profiler-vs-wall (profilers serialize →
absolute ms inflate, proportions exact). Forking a kernel / adding a ggml op / restructuring the graph
are all fair game when the measurement justifies it — the owner is explicitly up for "rewrite ggml."

**Bit-exact or owner-OK'd quality trade, every change.** Gate EVERY change (gates below). The "watercolour
melt" at end-of-clip is the failure the owner has been burned by — check the LAST frames specifically.

ONE GPU (RTX 3060, sm_86, 12 GB). Stop prod acestep/tts/llama before heavy runs; `docker rm -f
longcat-avatar-iter` for strays; never run two GPU jobs at once.

---

## What this is

13.6B audio-driven video DiT (DMD-distilled, runs CFG-free at **one eval/step**, 8 steps but 4≈8),
Q4_K weights on ggml **MMQ** (integer dot-product, no dequant→cuBLAS). Single clip (25f/8-step) ≈ **181 s
wall: DiT sampling ~140 s (77%)**, VAE ~15 s, encode ~16 s. The DiT is the fish. Source: the transformer
blocks (×~48) are in `src/longcat_avatar.hpp`; the model is `mmdit.hpp` / `common_dit.hpp`. Build graph =
`longcat_avatar.hpp:1148` (`build_graph`), rebuilt every step (sampler loop ~`:1271–1289`).

## Where the time goes — re-profile FRESH before trusting this

DiT **step** graph (`LONGCAT_OP_PROFILE`, the block with `FLASH_ATTN_EXT` present, ~10775 nodes;
proportions exact, abs inflated):

| op | % | notes |
|---|---|---|
| **MUL_MAT** | **45%** | Q4_K weight matmuls (MMQ) + F32 cross-attn projections (`pf32`). Audit: which dominate? precision justified? |
| **FLASH_ATTN_EXT** | **33%** | full 3D-global self-attn, ~10920 tokens @ 25f/480p (O(N²)). lap-21: ~**63% of roof** — under-saturated. |
| ADD | 5.7% | residuals |
| CONT | 4.3% | **734 ms / 1543 calls** — same shape as the shipped VAE lever-1 coalesced-cont win. Low-risk lever. |
| MUL | 3.6% | |
| SCALE | 2.6% | |
| NORM | 1.5% | RMSNorm (forced F32) |
| ROPE_PE | 1.2% | already a fused custom op (`ggml_rope_pe`, shipped −4.9%) |

Re-derive these with a fresh `LONGCAT_OP_PROFILE` run — don't trust a stale table.

---

## The levers (ranked by the "is the work needed" lens, not just the roofline)

### 1. Block-sparse / windowed self-attention — the algorithmic ceiling (biggest, fork-class)
The #1 work-elimination win. Self-attn is **33% of the wall and only 63% of its own roofline** — but the
deeper point is most of that compute is *wasted*: LongCat's own BSA keeps ~12.5% of blocks. A roofline
profile can NEVER show this — it's a type-(2) wrong-algorithm win.
- **Start approximate (cheapest entry):** a *static local-window block mask* — each spatiotemporal cube
  attends to its neighbours + the reference/first-frame cubes. Static indices (no dynamic per-step
  mean-pool→top-k gating, no varlen), keeps most of the FLOP cut. Full dynamic top-k (`sparsity=0.875`,
  cubes `[4,4,8]`=128 tokens) is "the last 20%, not the first 80%."
- **Don't port the Triton kernel — reimplement, forward-only.**
- **Constraints:** quality trade (NOT bit-exact) → owner OK + coherence gate (ac16 0.83–0.84 flat incl.
  LAST frames). T,H,W each divisible by `[4,4,8]` — pick 480p dims that divide cleanly (this is why the
  ref's `mask_frame_range` path disables BSA). The DMD distill is robust (4-step≈8-step) — encouraging.
- self-attn lives at `longcat_avatar.hpp:209` (`self_attn`), through `ggml_ext_attention_ext` →
  `ggml_flash_attn_ext` (`ggml_extend.hpp:1380`); note the `kv_scale=1/256` F16-overflow guard and the
  `can_use_flash_attn` / `L_k % 256` gate at `:1388–1409` (verify which path actually runs!).

### 2. Cross-attn K/V caching across the 8 DMD steps (modest, cheap-ish — measure first)
`build_graph` is rebuilt + recomputed every step. Text `cross_attn.kv_linear` (`:103`) and per-block
`audio_cross_attn.kv_linear` (`:110`) re-project **step-invariant** conditioning on every step × 48 blocks
— only Q changes per step. Project K/V once, reuse. **Honest sizing: a few %, not a headline** (self-attn
Q/K/V all change per step → NOT cacheable; only the small-M cross-attn GEMMs are). `audio_proj` windowing
is already hoisted. Needs cross-graph tensor persistence (graph rebuilt per step). **Microbench the
cross-attn GEMM share of the step first** (`LONGCAT_OP_PROFILE` / ncu) — if <2%, skip.

### 3. MUL_MAT 45% — audit precision + dispatch (the "is compute wasted" matmul angle)
Don't accept "45%, it's cuBLAS/MMQ, done." **Profile which matmuls dominate** (weight-MMQ vs F32
cross-attn). Then: (a) is `pf32` (forced F32) load-bearing on each, or can some accumulate F16 like the
VAE conv did (40 dB, gated)? (b) are the MMQ kernels at their roofline for *these* shapes (ncu
`*_tensor_op` / `dram__throughput` / occupancy)? (c) any matmul on a shape where MMQ/cuBLAS is known-weak
(tall-skinny / tiny-N) that a hand kernel beats? PERF.md lap-21 §7 has the matmul roofline microbench
(`tools/roofline_dit.cpp`) — rebuild and use it on the *actual* DiT shapes.

### 4. CONT 4.3% (734 ms / 1543 calls) — low-risk, proven playbook
Same strided-`cpy_scalar` coalescing gap that VAE lever-1 fixed (`LONGCAT_CONT_PROF` to find the slow
shapes, then route to the `cpy_perm_*` fast paths in `ggml/src/ggml-cuda/cpy.cu`). Bit-exact-able. The
DiT-step CONTs are attention-reshape / permute copies — profile them.

### What is NOT a lever (proven / triaged in `additional-levers.md` — do not burn cycles)
- **CFG / "cut 3→2→1 evals, ~3×"** — N/A. The port is already CFG-free at one eval/step
  (`--cfg-scale 1.0`; `resolve_guidance` never sets `use_uncond`). The advisor's headline was about the
  PyTorch ref. **Do not chase a 2–3× DiT win that doesn't exist here.**
- **INT8 tensor cores** — already on Q4_K MMQ (integer). Q3_K measured +7% slower.
- **CUDA graphs across fixed-shape steps** — measured dead (~5 ms overhead, compute-bound).
- **FP8 K/V** — sm_86 has no FP8 *compute*; bandwidth-only, and the 25f path isn't VRAM-bound.
- **FP32→F16 LayerNorm** — norms are 1.5%; deliberate F32 for parity. Risk ≫ reward (but see lever 3
  for the matmul precision angle, which is the real version of this).

---

## HOW TO PROFILE (this is the heart of "prove the floor")

**The lesson from the VAE campaign: inference from wall-clock deltas LIED; ncu was ground truth.** I spent
laps believing the conv was bandwidth- then GEMM-bound; ncu showed DRAM 1.3% / tensor 8% / 16.7%
occupancy — occupancy-bound. **Use the profiler before concluding anything.**

**1. Op-type breakdown (where to look):** `LONGCAT_OP_PROFILE=1` on a render → per-op-type ms/% for every
graph >1000 nodes (you'll see the VAE tiles AND the DiT step). Proportions exact, abs inflated (it
serializes). This tells you which op to ncu.

**2. ncu per-kernel ground truth (the decider).** ncu is in the builder image (`/usr/local/cuda/bin/ncu`).
Run inside docker with `--cap-add=SYS_ADMIN`, filter to the kernel, skip warmup launches, grab a few:
```
docker run --rm --gpus all --cap-add=SYS_ADMIN -e <flags> -v "$PWD:/src" -v "$PWD/models:/models" -w /src \
  longcat-avatar-dev:builder ncu --target-processes all -k "regex:<kernel_substr>" -c 3 --launch-skip 40 \
  --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__registers_per_thread,launch__occupancy_limit_registers,launch__waves_per_multiprocessor,\
gpu__time_duration.sum \
  /src/build/bin/sd-cli <standard render, --steps 1>
```
Read it like this: **DRAM% high → memory-bound** (coalescing/precision/reuse). **tensor_op% high →
genuinely compute-bound** (then it's algorithm/precision, not kernel). **Both low + warps_active low →
occupancy/latency-bound** (registers/smem/stalls — the VAE trap). For attn kernels, `flash` may show as
`fattn`/`flash_attn_ext` — match the regex to the real symbol (check the launch line ncu prints).

**3. Stall reasons (why a low-occupancy kernel stalls):** add `--metrics
group:smsp__pcsamp_warp_stall_reasons`. `long_scoreboard`=memory latency, `wait`=dependency,
`barrier`=`__syncthreads`, `math_pipe_throttle`=ALU saturated. This is how you tell latency-bound from
throughput-bound.

**4. Roofline microbench (isolated, right shape/precision):** `tools/roofline_dit.cpp` (build recipe in
PERF.md lap-21 §7) — runs the matmul/flash shapes in isolation so you compare against cuBLAS/MMQ/a hand
kernel without the full-graph noise. Use the **actual** DiT M/N/K and head_dim, not generic ones.

**5. The work-audit (no profiler — read the graph):** count ops, look for recompute across steps, padded
dims, masked-out work, `ggml_cont` chains. The biggest DiT win (block-sparse) is invisible to ncu — it's
in the algorithm. Cross-reference the PyTorch ref `longcat_video/` for what work is *structurally*
skippable.

---

## Method / environment (build · render · gate)

**Build:** `~/dev/kobbler/docker/longcat-avatar-dev/iter.sh build` (~90 s; CUDA builder image sm_86,
ccache; host has no CUDA — binaries run from the image). Kernel changes go in the `ggml/` **submodule**
(commit there, then `git add ggml` + commit in parent — submodule-first). `iter.sh cli` doesn't pass env;
run `docker run … -e LONGCAT_… longcat-avatar-dev:builder /src/build/bin/sd-cli …` directly.

**Standard render (hold constant):**
```
-M vid_gen -m models/longcat-avatar-1.5-dit-dmd-q4_k.gguf --t5xxl models/longcat-umt5-xxl-q8_0.gguf \
  --vae models/longcat-wan-vae-f16.gguf --audio-vae models/longcat-whisper-v3-encoder-f16.gguf \
  --init-img models/_testinputs/girl_480x832.png --audio models/_testinputs/speech_16k.wav \
  -p "a person talking" --cfg-scale 1.0 --video-frames 25 -W 480 -H 832 \
  --steps 8 --diffusion-fa --seed 42 --clip-on-cpu --max-vram 9 -o build/<name>.webm
```
**DiT is step-dependent** (unlike the VAE) — sampling scales with `--steps`. Use `--steps 8` for real
timing/quality; you can profile the *step graph* at `--steps 1` (one step = one DiT graph) but the
sampling-loop wins (KV cache, step skipping) need ≥2 steps to measure. Watch the
`sampling completed, taking Xs` log line.

**GATES (mandatory):** bit-exact change → `clip_compare.py <base> <new>` must read **PSNR 99.00**.
Quality trade (BSA, F16 matmul — needs owner OK) → `clip_compare.py <base> <new>` PSNR + **`clip_compare.py
<clip>` single-arg ac16 0.83–0.84 FLAT across ALL frames, watch the LAST frames for melt.** Host has
numpy+PIL+ffmpeg; the builder image does NOT — run clip_compare on the host. Always render at **2 seeds**
when judging a quality trade so the owner can A/B real regression vs sampling noise.

---

## Hard-won lessons from the VAE campaign (internalize before you start)
- **ncu is truth; wall-clock inference lies.** Every wrong "floor" was an un-profiled assumption.
- **A fused/custom kernel that holds accumulators across a long reduction is occupancy-capped** — that
  killed the VAE conv (see `HANDOFF.md` Lever 2 + tag `lap25-conv3d-direct-occupancy-floor`). Any extra
  smem/regs to hide latency cuts the binding occupancy. Watch for the same trap in any attn kernel rewrite.
- **F16 accumulation held quality at 40 dB** in the VAE conv — the matmul-precision lever (3) is real,
  but gate it hard (ac16, last frames, 2 seeds).
- **Don't grind a 2% knob while a 40% one waits.** Re-profile after every win — the bottleneck relocates
  (it moved 4× during the VAE phase). Block-sparse (33%, algorithmic) almost certainly beats every
  kernel-level matmul poke; sequence it first unless a cheap measurement says otherwise.
- **Prove the floor, don't assert it.** If you're about to type "we're at the floor / it's compute-bound,"
  you owe the five answers in the MOOD section, each with an ncu number or an A/B. The confident "nothing
  left" is only earned with receipts.
