# Additional perf levers — external advisor review (triage)

*Written 2026-05-27. An external advisor (reasoning largely from the **PyTorch reference**
`proof_gen.py` / `longcat_video/`, not this port) produced a batch of perf suggestions. This file
triages each against the **actual sd.cpp port state** (HANDOFF.md, PERF.md laps 00-25, the source).
Goal: tell the perf agent what's a real lever vs. what's already-done / dead / not-applicable, so
the advisor's headline claims don't burn cycles.*

**Bottom line up front:** the advisor's single biggest claim — *"cut DiT evals 3→2→1 per step, up to
~3×"* — **does not apply to this port.** It's reasoning about the reference pipeline's CFG defaults.
This port already runs CFG-free at one DiT eval per step. The two suggestions with genuine headroom
here are **(1) block-sparse / windowed attention** and **(2) cross-attn K/V caching across steps**;
everything else is already done, measured dead, or N/A. Detail below.

---

## IN — worth the perf agent's time

### 1. Block-sparse / windowed attention (the one genuinely new big lever)
**Status: not pursued — port is dense by deliberate decision ("start dense, skip BSA").**

The advisor's BSA deep-dive is correct and lands on this port's real ceiling. Self-attention is full
3D-global over the flattened `[T,H,W]` sequence (~10920 tokens at 25f, ~37k at 93f), O(N²). The
profiled DiT step is **FLASH_ATTN_EXT 33%** of wall and lap-21 measured flash at only **~63% of its
roofline** — it's the single biggest under-saturated hot op. HANDOFF.md lever-3 frames the fix as an
*FA2/FA3 kernel* (a roofline play). The advisor adds the orthogonal **work-elimination** angle:
LongCat's own BSA keeps ~12.5% of blocks → the paper's "<10% of dense." That's a type-(3) win a
roofline profile can't see.

What the advisor got right and is worth lifting:
- **Don't port the Triton kernel** — reimplement. Forward only (skip the training `_attn_bwd_*`).
- **Approximate, don't replicate (advisor option a — start here):** a *static local-window block mask*
  — each spatiotemporal cube attends to its neighbours + the reference/first-frame cubes — drops the
  expensive dynamic per-step gating (mean-pool → block-score → top-k), gives static indices (no
  varlen, no top-k), and keeps most of the FLOP cut. Cheapest entry to the win.
- Full dynamic top-k (`sparsity=0.875`, cubes `[4,4,8]`=128 tokens) is "the last 20%, not the first 80%."

Caveats this port must respect:
- **Quality trade, not bit-exact.** Violates the standing bit-exact mandate → needs owner OK + the
  coherence gate (`clip_compare.py`, ac16 0.83–0.84 flat across **all** frames incl. the last — watch
  for the "watercolour melt"). The DMD distill is robust (4-step ≈ 8-step), which is encouraging.
- **Divisibility constraint:** BSA needs T,H,W each divisible by `[4,4,8]`. Pick 480p dims that divide
  cleanly. (This is exactly why the reference's `mask_frame_range` path disables BSA.)
- Fork-class effort. But it attacks 33% of the clip wall and the algorithmic ceiling, so the ROI beats
  the parked VAE lever-2 (whole ceiling ~3.9 s/clip). Sequence it as **lever 3** behind the cheap
  measurement below.

### 2. Cross-attn K/V caching across the DMD steps (modest, cheap-ish — measure first)
**Status: not done. `build_graph` is rebuilt + recomputed every step (`longcat_avatar.hpp:1288`).**

Text cross-attn `kv_linear` (context → K/V) and per-block `audio_cross_attn.kv_linear` (audio_hidden →
K/V) re-project from step-invariant conditioning on **every** one of the 8 steps × 48 blocks. Only Q
changes per step. Projecting K/V once and reusing them kills those GEMMs + the conditioning re-reads.

Honest sizing (the advisor's own estimate, and it's right): **a few %, not a headline.** Self-attn (the
37k-token monster) *cannot* be cached — Q/K/V all come from the changing latent. Only the small-M
cross-attn GEMMs are cacheable (text ≤512 tokens, audio 32 tokens/frame). **`audio_proj` windowing is
already hoisted out of the loop** (done — see the sdcpp-port memo / `longcat_avatar.hpp:~1232`); this is
specifically the per-block `kv_linear` projections that still live inside the per-step graph.

Cost: sd.cpp rebuilds the graph per step, so caching needs cross-graph tensor plumbing (persist K/V
buffers across `compute()` calls). Cheap reward / real-ish plumbing — **microbench the cross-attn GEMM
share of the DiT step first** (it'll show in `LONGCAT_OP_PROFILE`) before building it. If it's <2% of
the step, skip.

---

## OUT — already done, measured dead, or not applicable

### ✗ All CFG / guidance levers — the advisor's headline (~3×). **NOT APPLICABLE.**
The advisor's #1/#2 ("test text CFG off", "batch the cond+uncond"), the dual-axis `audio_gs=1`
cancellation ("3→2 evals"), and "drop the audio cond/noise branch" are **all** reasoning about the
PyTorch reference `proof_gen.py`, which runs `do_classifier_free_guidance` with a 2- or 3-eval
dual-axis CFG path.

**This port never ported any of that.** Proof (per the project's own #1 rule — proven, not asserted):
- Standard render runs `--cfg-scale 1.0` (HANDOFF.md "Standard render").
- `resolve_guidance` (`stable-diffusion.cpp:3359`): `txt_cfg != 1.f` is the *only* thing that sets
  `use_uncond = true`. At `txt_cfg == 1.f` it stays false.
- Therefore the uncond condition is never built (`:4160 if (request->use_uncond ...)`) and the second
  DiT pass is skipped (`:2169 if (!uncond.empty())`).
- ⇒ **The port already runs exactly one DiT eval per step.** There is no batch-2, no third pass, no
  dual-axis CFG to collapse. The DMD distill runs CFG-free, which is what the advisor was steering
  toward — we're already there.

This is the most important callout: **do not let the perf agent chase a 2–3× DiT win that doesn't
exist in this codebase.**

### ✗ INT8 tensor cores / "are your GEMMs on the dequant→FP16 path?"
Already on the integer path the advisor recommends. DiT weights are **Q4_K on ggml MMQ** (integer
dot-product, no dequant→cuBLAS-FP16 round-trip). PERF.md: Q4_K is the floor on Ampere, Q3_K measured
**+7% slower**, vendor MMQ/flash kernels confirmed optimal by reading the dispatchers. Nothing to flip.

### ✗ CUDA graphs across the fixed-shape steps
**Measured dead.** PERF.md: CUDA-graph reuse = ~5 ms overhead, the DiT step is compute-bound (MUL_MAT
45% + flash 33%), so erasing launch overhead buys nothing. Same conclusion the turboquant laps reached
for llama.

### ✗ "Profile the VAE decode separately, it's a fat fixed slice"
Generic advice; the port is ~5 laps ahead. VAE has a full op profile (MUL_MAT 26% / IM2COL_3D 22% /
CONCAT 11% / …) and shipped wins: spatial tiling (570→54 s headline), tiled smem-halo im2col_3d
(−76%), coalesced CONT (−15%). One correction to the advisor's framing: VAE is **~10% of single-clip
wall**, not 20–40% — but it **compounds ×N for chained clips**, which is why it was finished first.

### ✗ Coalescing the rearrange / RoPE / attention-reshape hotspots
The advisor's instinct is right and the port already acted on it: custom **fused-RoPE CUDA op
`ggml_rope_pe`** (shipped, −4.9% all lengths, bit-exact) and **CONT pixel-shuffle/permute coalescing**
(lap-24, the 3.4× strided-`cpy_scalar` gap, shipped). This *is* the port's recent active lever, not a
gap.

### ✗ FP32 LayerNorm → FP16-with-FP32-accumulate
Deliberate choice, not an oversight, and blocked by the bit-exact mandate. The port forces
`GGML_PREC_F32` (`pf32 = true`) on **every** Linear for parity with the reference's bf16 (which has the
F32 exponent range) — there are explicit overflow/NaN notes (ffn.w2 1/256 pre-scale, flash-attn V
kv_scale 1/256). Norms are only ~1.5% of the DiT step anyway. Risk ≫ reward.

### ✗ Evict the run-once encoders (umT5 / Whisper / VAE-encoder) for VRAM
Essentially already done: umT5 + Whisper run on **CPU** (`--clip-on-cpu`), VAE encoder runs once. The
25f fast path already fits 12 GB with ~3 GB headroom. The *one* remaining sliver the advisor's logic
points at is already a known, logged minor lever: **umT5 GPU-encode-then-free** (load→encode→free
sequencing, ~−14 s / −6% one-time; today it's CPU 16 s because umT5 6 GB + DiT 8.5 GB can't coexist at
load). Minor, known, not new.

### ✗ FP8 K/V storage; drop the audio cond/noise branch
- FP8 K/V: Ampere (sm_86) has no FP8 *compute*; this is a bandwidth/VRAM-only trick, and the 25f path
  isn't VRAM-bound. KV already runs F16 with a 1/256 overflow scale. Marginal.
- "Drop the audio cond/noise branch at `audio_gs=1`": N/A — the port implements no audio CFG, and cond
  (ref-image) frames already receive **no audio** (`longcat_avatar.hpp` num_cond_latents split).

---

## One thing the advisor got dead right (worth internalizing)
*"'At the floor' almost always only establishes (1) this-kernel-is-at-roofline. The big wins live in
(2) wrong-algorithm and (3) redundant-work, and they don't show up in a roofline profile."* This is the
same lesson HANDOFF.md's #1 rule was burned into by the VAE phase (im2col was "at the floor" at 7% of
bandwidth). The BSA lever (IN #1) is precisely a type-(2)/(3) win behind a "flash is at 63% of roofline"
reading — the advisor's framing is the right lens for it.
