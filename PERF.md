# LongCat-Video-Avatar.cpp — PERF / VRAM tracking

Tracking doc for the perf/VRAM optimization phase. The port is feature-complete
(image + audio → talking video, pure C++/ggml on a 12GB RTX 3060); this file
records the optimization laps that make it fast + VRAM-friendly enough for quick
vtuber-clip turnaround.

## Standard render config (held constant for comparisons)

```
-M vid_gen -m models/longcat-avatar-1.5-dit-dmd-q4_k.gguf \
  --t5xxl models/longcat-umt5-xxl-q8_0.gguf \
  --vae models/longcat-wan-vae-f16.gguf \
  --audio-vae models/longcat-whisper-v3-encoder-f16.gguf \
  --init-img models/_testinputs/girl_480x832.png \
  --audio models/_testinputs/speech_16k.wav \
  -p "a person talking" --cfg-scale 1.0 --video-frames 25 -W 480 -H 832 \
  --steps 8 --diffusion-fa --seed 42 --clip-on-cpu --max-vram 9
```

Per-lever the offload / VAE flags are the variable under test. Peak VRAM is the
max `nvidia-smi memory.used` sampled at 0.5s during the run (includes ~120 MiB
of idle prod processes on the shared GPU; subtract for the model's own peak).

## Phase breakdown (where the wall time goes)

| Phase | What |
|-------|------|
| model load | mmap DiT q4_k (8.9 GB) + umT5 (CPU) + VAE + whisper |
| text encode (umT5) | one-time, CPU |
| audio (whisper+window) | one-time mel→whisper encoder→AudioProjModel inputs |
| DiT sampling | 8 DMD steps × 48 blocks (self-attn + text-cross + audio-cross + SwiGLU) |
| VAE decode | Wan VAE temporal decode, 7 latent frames → 25 video frames |

## Laps

| lap | lever | wall (s) | peak VRAM (MiB) | quality | commit |
|-----|-------|----------|-----------------|---------|--------|
| 00  | BASELINE (`--vae-on-cpu`) + lever-1 fixes (fps 25 default, audio auto-mux) | 768.7 | 10535 | coherent (ac16≈0.83 all frames) | 83e3957 |
| 01  | GPU VAE decode + spatial tiling (default-on for avatar) | 238.4 | 10779 | PSNR 40.1dB vs lap00 (min 36.2) | 736eb79 |
| 02  | configurable DMD steps (default 8; `--steps 6` opt-in) | 238.4 (8) / 197.4 (6) | 10779 | 8-step unchanged; 6-step coherent, ~35.6dB vs 8-step | bcda117 |
| 03  | DiT-step PROFILED + reusable phase/op profilers + audio `gate_mul` (drop pointless zeros-add) | 237.8 (8) | 10779 | bit-identical to BEST (99 dB all frames) | 877c258 |
| 05  | DiT glue cuts: `scale_bias` fuse + redundant qkv-cont removal + audio silu reuse | sampling 162.3 (was 163.7) | 10779 | bit-identical to BEST (99 dB all frames) | 24d2d9c |

(rows appended per lap below)

### lap 00 — baseline
- Config: standard + `--vae-on-cpu`.
- **Wall 768.66s**, peak VRAM 10535 MiB (during DiT sampling; VAE runs on CPU).
- Per-phase wall: model load ~31s | encode_first_stage (ref-image VAE encode) 16.3s |
  text encode (umT5, CPU) 16.5s | DiT sampling 164.2s | **VAE decode (CPU) 569.7s**.
- **VAE decode is 74% of wall** — the headline target for lever 2 (GPU VAE).
- Latent healthy: predecode std 0.89, per-frame std 0.53→0.99, nnan=0.
- Quality gate: `tools/clip_compare.py` ac16 ≈ 0.83-0.84 on all 25 frames
  (natural-image structure, not noise). The port produces a coherent talking
  avatar — the PORT-PROGRESS "generated frames still noise" status is STALE;
  current tree renders coherent frames.
- Lever 1 validated: output webm has a `pcm_s16le @ 16000 Hz` audio stream
  auto-muxed (verified via ffprobe); fps defaults to 25 when `--audio` is given.
- Checkpoint clip: `models/_perf/lap00_baseline.webm`.

### lap 01 — GPU VAE decode + spatial tiling (lever 2, THE big one)
- **First tried: drop `--vae-on-cpu`, full-clip GPU decode.** → **CUDA OOM** at
  peak ~11.9 GiB. The Wan-VAE temporal decode builds the whole clip (7 latent
  frames → 25 video frames, all per-frame decode ops + concats) into ONE graph;
  the activations for 480x832 don't fit 12 GB. `--max-vram` graph-cut does not
  rescue it (VAE intermediates, not just matmul offload). Dead end as-is.
  - The per-frame `decode_partial` path in `wan.hpp` is disabled ("chunk 1 result
    is weird") because the CausalConv3d feature cache holds graph-node pointers
    that don't survive across separate `compute()` calls (gallocr-aliasing trap,
    cf. siglip2 memo) — would need host round-tripping of the cache. Deferred.
- **What won: GPU VAE + `--vae-tiling` (spatial 2D tiling).** Latent 60x104 →
  ~15 tiles of 32, each a small-spatial full-temporal decode graph, stitched with
  overlap. **VAE decode 569.7s → 54.03s (10.5x), total wall 768.7s → 238.4s
  (3.2x), peak VRAM 10779 MiB (fits, no OOM).** DiT sampling (164s) is now the
  dominant phase.
  - Quality gate: `clip_compare.py` GPU-tiled vs CPU baseline = **mean PSNR
    40.06 dB, min 36.24 dB**, ac16 identical to 3 d.p. The divergence is GPU-vs-CPU
    float + tile-seam blending; 40 dB is above the ~37.6 dB VAE-vs-input ceiling,
    so visually equivalent. Audio still muxed (pcm_s16le 16kHz verified).
- **Shipped as the avatar default** (`stable-diffusion.cpp generate_video`): when
  the VAE is on GPU and tiling wasn't explicitly requested, enable spatial tiling
  for the avatar (logs "enabling VAE spatial tiling by default"). `--vae-on-cpu`
  and explicit `--vae-tiling`/`--vae-tile-size` both still take precedence. So the
  standard config (no `--vae-on-cpu`) now gets the fast path automatically.
- Checkpoint clip: `models/_perf/lap01_gpuvae_tiled.webm`.

### lap 02 — configurable DMD step count (lever 4, sampling)
- DiT sampling is now the dominant phase (164s of the 238s wall, 69%). The avatar
  hardcoded an 8-step DMD schedule regardless of `--steps`. `build_longcat_dmd_sigmas`
  re-derives a valid distilled schedule for ANY step count (distill indices scale
  with the count), so `--steps` can trade quality for speed.
- **Wired `--steps` to the DMD schedule**: default stays **8** (the count the DMD
  LoRA was distilled for — proven, unchanged). `--steps N` with N<8 uses an N-step
  distilled schedule. N>=8 / unset → 8.
- **Measured `--steps 6`**: sampling 164.3s → 123.1s, total **238.4s → 197.4s
  (-17%)**, same VRAM. Output stays coherent (ac16≈0.83 all frames). Differs from
  the 8-step trajectory (mean PSNR 35.6 dB, min 29 dB vs 8-step) — structurally
  sound but a visibly different render.
- **NOT made the default**: whether 6-step lip-sync holds vs 8-step is a human
  eyeball call (the structural metrics can't judge mouth fidelity), and the LoRA's
  distill target is 8. Shipped as an opt-in `--steps 6` speed lever; default 8.
- Checkpoint clip: `models/_perf/lap02_6steps.webm` (6-step, for A/B vs lap01 8-step).

### lap 03 — DiT SAMPLING PROFILED (the headline finding) + audio gate_mul micro-win

**This lap was a deep profile of the DiT step (the remaining 69% of wall, ~20.4s/step
× 8). Verdict: the step is COMPUTE-BOUND on MUL_MAT + FLASH_ATTN; there is NO large
safe pure-speed lever, and CUDA-graph reuse (the forecast marquee win) is DEAD.**

**Profiling aids added (both env-gated, zero-cost when off, reusable):**
- `LONGCAT_PROFILE=1` → per-`execute_graph` phase log in `ggml_extend.hpp`
  (`nodes / alloc / copy_in / compute` ms). Committed.
- `LONGCAT_OP_PROFILE=1` → per-op-TYPE wall aggregator in the ggml-cuda node loop
  (`cudaStreamSynchronize` around each node; inflates absolute time but proportions
  are exact). This lives in the **ggml submodule** (`ggml-cuda.cu` node loop); it was
  used to take the breakdown below and then REVERTED to keep the submodule pristine
  (detached HEAD — committing it would dangle the parent's submodule pointer). To
  re-take an op profile: re-apply the ~25-line `[OP_PROFILE]` block around
  `ggml_cuda_compute_forward` in `ggml/src/ggml-cuda/ggml-cuda.cu` and rebuild.

**Per-step phase split (`LONGCAT_PROFILE`, 25f/8-step/480p, prod config):**
- DiT step compute = **20.40s**; graph build+alloc+copy_in = **~5 ms** (0.02%).
  ⇒ **per-step launch/build overhead is negligible → CUDA-graph capture/reuse cannot
  help.** (ggml's CUDA-graph support is also compiled OFF — `GGML_CUDA_GRAPHS`
  defaults OFF, iter.sh doesn't set it — but even compiled-in it keys on
  `cgraph->nodes[0]` + a 2-call warmup, and every step rebuilds a fresh graph in a
  reset compute_ctx, so it would never warm up. Not worth chasing for 5 ms/step.)

**Per-op-TYPE breakdown of ONE DiT step (`LONGCAT_OP_PROFILE`, 13705 nodes, audio on):**

| op | ms | % | calls | notes |
|----|----|---|-------|-------|
| **MUL_MAT** | 7586 | **38%** | 730 | Q4_K weight matmuls — compute-bound, the floor |
| **FLASH_ATTN_EXT** | 6737 | **34%** | 144 | 96 self (2-pass cond+noise/block) + 48 text-cross |
| CONT | 1744 | 8.7% | 2023 | attention reshape/permute conts (diffuse) |
| ADD | 1194 | 6.0% | 1213 | residual + modulate-shift broadcasts |
| MUL | 918 | 4.6% | 626 | modulate-scale + gate broadcasts |
| SCALE | 455 | 2.3% | 1059 | incl. the kv_scale=1/256 F16-guard scales |
| CONCAT | 349 | 1.7% | 241 | cond-zero prepends + cond/noise concat |
| REPEAT | 260 | 1.3% | 673 | `ggml_ext_ones/zeros/full` materializations |
| NORM | 255 | 1.3% | 242 | LayerNorms |
| CPY | 217 | 1.1% | 384 | |
| PAD | 144 | 0.7% | 192 | flash L_k pad-to-256 |

- **MUL_MAT (38%) + FLASH_ATTN (34%) = 72% is irreducible compute** on uniform-Q4_K
  weights at 480p/10920 tokens. The remaining ~28% is diffuse glue (CONT/ADD/MUL/
  SCALE/CONCAT/REPEAT), each individually small and bandwidth-bound — no single
  removable hotspot. **Conclusion: the only material levers left are quality-sensitive
  (fewer steps, quant ladder) — there is no safe default-on pure-speed win.**

**Audio cross-attn cost measured (lever 3):** audio-on step 20.40s / 13705 nodes vs
audio-off step 18.44s / 9801 nodes ⇒ **+1.96s/step (+11%), +3904 nodes** (confirms the
session-5 estimate). The audio path adds, per block, a non-flash per-frame attention
(K=32) + a SECOND full-token modulate+gate over the ~9100 noise tokens. The
modulate/gate bandwidth is the cost, not the tiny K=32 attention — not a cheap win.

**Micro-win shipped (safe, default-on): `gate_mul`.** The audio path computed its
contribution as `gate_add(zeros_like(ao), ao, gate)` — materializing a full
[4096 × ~9100] zero tensor and adding it, per block. Replaced with a new `gate_mul`
(gated multiply, no zeros residual; the cond-frame zeros are prepended separately
anyway). Nodes 13705→13561 (−144 = 48× {scale+repeat+add}). Sampling 164.32s→163.66s,
wall 238.46→237.76s (−0.4%, within step-to-step noise but a clean removal of pointless
work). **Output bit-identical to BEST (PSNR 99 dB / ac16 0.83 all 25 frames).**
- Checkpoint clip: `models/_perf/lap03_gate_mul.webm`.

### lap 03b — step-count A/B clips for the owner (OPT-IN quality lever 4, no default change)

Since DiT sampling has no safe default-on speed lever (above), the only remaining
speed gains are the quality-sensitive `--steps` knob. Rendered a fresh 8/6/4-step set
on the CURRENT build (gate_mul) at the prod config (25f/480p/seed 42, audio on) so the
owner can eyeball lip-sync vs speed in the morning. All structurally coherent
(ac16 ≈ 0.83 on every frame); PSNR-vs-8step measures trajectory divergence, NOT a
quality verdict — mouth fidelity is the human call.

| steps | wall (s) | sampling (s) | vs 8-step wall | coherence | PSNR vs 8-step |
|-------|----------|--------------|----------------|-----------|----------------|
| 8 (default) | 237.8 | 163.7 | — | ac16 0.83 | (reference) |
| 6 | 196.7 | 122.6 | **−17%** | ac16 0.83 | mean 35.6 / min 29.0 dB |
| 4 | 155.8 | 81.7 | **−35%** | ac16 0.83 | mean 29.9 / min 23.0 dB |

- Clips: `models/_perf/lap03_gate_mul.webm` (8), `lap03_6steps.webm` (6),
  `lap03_4steps.webm` (4). **Default unchanged (8).** Recommendation pending the
  owner's lip-sync eyeball: 6-step looks like the safe speed pick (−17%, modest
  trajectory drift); 4-step (−35%) is more divergent and likely softens lip detail —
  judge before adopting. No code change in 03b (the `--steps` plumbing is lap 02).

### lap 04 — FULL-LENGTH (93f) HITS A HARD VRAM WALL; graph-cut + CPU-offload enables 81f (3.24s) — the practical full-length ceiling on 12 GB

**Goal: the owner's morning A/B needs the *native segment length* (93 frames @ 25fps ≈ 3.7s),
not the 25f clips (which "cut off mid-word").** Rendering 93f surfaced a hard VRAM wall and
required a real code change to get most of the way there.

**THE 93f WALL (measured):** at 93 frames the latent is 24 frames × 1560 spatial = ~37,440
tokens. The DiT forward's *activation* compute buffer is **13,310 MiB** — over the 11,909 MiB
the 3060 actually exposes **even with ZERO resident weights** (`--offload-to-cpu` puts all
params in RAM; the single compute graph still wants 13.3 GiB). The compute buffer scales
super-linearly with frames: 25f ≈ 2 GiB · 49f ≈ 4.6 GiB · 93f ≈ 13.3 GiB. With weights
resident on GPU (8539 MiB) the ceiling is only ~33f; the activation buffer is the wall.
`--max-vram` graph-cut alone does NOT rescue it — the monolithic forward marks no graph-cut
boundaries, so `ggml_gallocr_reserve` reserves the whole-graph buffer up front (same class as
the lap-01 full-clip VAE OOM).

**THE FIX (committed, `src/longcat_avatar.hpp`):** mark graph-cut boundaries on the avatar
DiT (mirrors `anima.hpp`): a `longcat.prelude` group for the cross-block-persistent inputs
(`t_emb`/`context`/`pe`/`audio`/initial `x`), then `x` after **every block** PLUS two
**intra-block** cuts (`post_self_attn`, `post_cross_attn`) so each block splits into
self-attn / cross-attn / FFN sub-segments. With `--offload-to-cpu` (params on the CPU backend
≠ the CUDA runtime backend — the condition that *enables* the segmented path) the runner now
reserves ONE sub-segment's activation buffer at a time and streams only that sub-segment's
weights to GPU. **The marks are inert when weights are resident** (`can_attempt_graph_cut_
segmented_compute()` is false when params_backend == runtime_backend), so the fast 25f path is
unchanged. Build green via iter.sh.

**RESULT — the real ceiling is 81 frames (3.24s), NOT 93.** Even per-sub-segment, block-0's
**self-attention** sub-segment is an atomic 12,675 MiB at 93f (flash-attn + RoPE conts over
37k tokens) — it cannot be cut at block/sub-block boundaries, so 93f stays over the card.
Empirically (steps=1 probes, offload + `--max-vram 9`):

| frames | seconds @25fps | fits 12 GB? | self-attn sub-segment |
|--------|----------------|-------------|------------------------|
| 73 | 2.92 | ✅ | < budget |
| 81 | 3.24 | ✅ (peak 11603 MiB) | ~11.3 GiB |
| 85 | 3.40 | ❌ OOM (accumulated frag) | — |
| 93 (native) | 3.72 | ❌ OOM | **12,675 MiB atomic** |

**81f is the shipped full-length.** It covers 81 of the audio's 100 frames (4.0s wav →
windowed to 81). Reaching the true native 93f would need **splitting the self-attention
internally** (temporal-chunked flash over the 24 latent frames) — that changes the proven
full-temporal attention math and is correctness-risky; deferred for the owner to scope. 81f is
3.24s, far past the 25f "cuts off mid-word" problem, and plenty for a lip-sync A/B.

**COST of offload (the tradeoff):** weights stream from RAM per sub-segment, so DiT sampling
is ~124 s/step (8.5 GB × 48 blocks × 3 sub-segments re-streamed each step) vs ~20 s/step for
resident-weight 25f. Full-length is minutes, as expected — it's a turnaround knob, not the hot
path. VAE tiled decode at 81f is ~175s (10 tiles × ~17.5s; bigger temporal extent per tile).

**Full-length A/B set (81f / 480p / seed 42 / audio on / offload + graph-cut + max-vram 9):**

| steps | wall (s) | sampling (s) | decode (s) | peak VRAM (MiB) | vs 8-step | clip |
|-------|----------|--------------|------------|-----------------|-----------|------|
| 8 (default) | 1217.99 | 992.23 | 174.95 | 11603 | — | `models/_perf/fulllen_81f_8step.webm` |
| 6 | 966.58 | 743.68 | ~175 | 11603 | **−21%** | `fulllen_81f_6step.webm` |
| 4 | 710.92 | 495.63 | 175.06 | 11603 | **−42%** | `fulllen_81f_4step.webm` |

All carry muxed audio (pcm_s16le 16 kHz, verified via ffprobe). **Default step count unchanged
(8).** The 6-vs-4 lip-sync fidelity at full length is the owner's eyeball call (same as lap 03b
at 25f). Note: at full length the step count trades the same way as 25f (sampling is linear in
steps), so 6-step ≈ −25% sampling, 4-step ≈ −50% sampling vs 8-step.

### lap 05 — re-opened the "compute-bound, exhausted" verdict: glue cuts (shipped) + Q4_0 quant (faster, opt-in) + 93f wall narrowed; BSA verdict

The lap-03 "no safe default-on pure-speed lever" conclusion was PARTIALLY premature.
This lap re-ran the four owner-listed levers MEASURED. Results, honest:

**Lever 1 — step-invariant recompute: ASSESSED, NOT WORTH (sub-noise).** The
denoise loop runs the DiT 8× (cfg-scale 1.0 → one `compute()` per step, no
cond/uncond doubling). Step-invariant recompute candidates: `y_embedder`
(context projection, 2× Linear(4096,4096) on ~512 text tokens), `audio_proj`
(AudioProjModel, proj1_vf 51200×512 on ~6 tokens), the RoPE `pe` table (already
a fixed input buffer, set once). Math: the y_embedder is ~2 of the 730 MUL_MATs
and runs on 512 tokens vs the per-block matmuls on 10920 — together y_embedder +
audio_proj are <0.5% of a step. Caching them across `compute()` calls means
persisting output tensors in a backend buffer (each step is a fresh graph in a
reset compute_ctx) — real gallocr-aliasing risk (cf. siglip2/lap-01 VAE memo) for
a <0.5% win. **Verdict: the saving is below the step-to-step noise floor; not
implemented.** (Recorded so a future session doesn't re-derive it.)

**Lever 2 — the ~28% glue ops: SWEPT, three clean cuts SHIPPED (lap 05).** The
glue is diffuse but not untouchable. Three mathematically-exact, quality-neutral
removals landed (`src/longcat_avatar.hpp`):
  1. **`scale_bias` fuse.** `modulate()` and `FinalLayer` computed the adaLN
     `(scale + 1)` as `ggml_add(scale, ggml_ext_ones(...))` — `ggml_ext_ones` is a
     SCALE + a REPEAT (materialized ones tensor), then an ADD. Replaced with one
     fused `ggml_scale_bias(scale, 1, 1)` (CUDA `dst = s*x + b`, single kernel).
     Removes ~145 REPEATs + ~145 SCALEs/step (maps onto the profiled REPEAT 673 /
     SCALE 1059 counts) and converts the ADD → scale_bias.
  2. **Redundant qkv-cont removal.** `self_attn` did `ggml_reshape_4d(ggml_cont(
     parts[i]))` on each of q/k/v. But `split_qkv` already conts the permuted qkv,
     so each q/k/v view is a CONTIGUOUS outer-dim sub-block (split is along ne[3]=3)
     that `reshape_4d` accepts directly (verified: `ggml_is_contiguous` true for an
     outer-dim slice; render runs, no assert). Removes 3 full-size copies/block
     (~179 MB @ 25f, ~613 MB @ 93f) × 48 blocks/step.
  3. **Audio adaLN silu reuse.** The audio path recomputed `silu(t_mod)` that the
     main adaLN already computed (`t_act`); now reuses it (−1 SiLU/block when audio
     on).
  - **Measured: sampling 163.66s → 162.26s @ 25f/8-step (−0.86%).** Output
    **BIT-IDENTICAL to BEST** (PSNR 99 dB / ac16 identical all 25 frames). Small but
    real and free — same class as lap-03 `gate_mul`. The remaining glue (the
    `ggml_ext_attention_ext` internal permute-conts to the flash `[N,n_head,L,d]`
    layout, the cond/noise split conts, the cross-attn kv-split conts) is either
    flash-kernel-layout-required or on small tensors (n_ctx 512 / audio 32) — no
    further large quality-neutral cut found. **The 28% is now genuinely diffuse +
    mostly structural, not "untried".**

**Lever 3 — Q4_0 quant: FASTER (−2.4%) but NOT quality-neutral → OPT-IN, default
stays Q4_K.** lap-04 only tested Q3_K (dead: +7% slower + worse). Q4_0 is the
untested faster-MMQ-path candidate. Requanted the DMD-folded q8_0 → Q4_0
(`models/longcat-avatar-1.5-dit-dmd-q4_0.gguf`, 8.94 GB, same size as Q4_K).
  - **Speed: 2-step sampling Q4_0 39.52s vs Q4_K 40.48s; 8-step 158.36s vs
    162.26s = −2.4%.** Real (Q4_0's simpler block format has a leaner MMQ dp4a path
    on Ampere than Q4_K's K-means + scale-min unpack). Refutes the implicit
    "Q4_K is the floor" — Q4_0 IS faster here.
  - **Quality: NOT neutral.** 8-step Q4_0 vs BEST = mean PSNR **30.70 dB, min
    25.38 dB** (clip stays coherent, ac16 ~0.83 tracking BEST, latent healthy std
    0.883 nnan=0) — but 30 dB ≠ the 99 dB bit-identical bar and below the ~37 dB
    VAE ceiling, so it's a VISIBLY different render. Q4_0's coarser quant shifts the
    denoise trajectory; mouth/lip fidelity (what matters for an avatar) is a
    human-eyeball call, same as `--steps 6`. **Default unchanged (Q4_K). Q4_0 gguf
    kept as an opt-in speed knob; A/B clip `models/_perf/lap05_q4_0_8step.webm`.**
    Better outcome than Q3_K (which was slower AND worse).

**Lever 4 — BSA / long-seq attention: SCOPED. Verdict: BSA is the WRONG tool for
this port (it's a quality cut here), and the 93f VRAM wall is now ~107 MiB away
via the lever-2 cont cut, but closing it needs more dense-activation surgery.**
  - **The reference avatar inference path runs DENSE, not BSA — by design.** The
    ckpt config ships `enable_bsa=False, bsa_params=None`; `proof_gen.py` (the
    single-audio→video path we port) explicitly sets `m.enable_bsa = False`;
    `modules/avatar/attention.py:69` only uses BSA when `batch>1` ("bsa will not be
    used in image training / sampling") and DISABLES it in the cond/noise split
    ("close bsa to prevent the temporal dimension from being divisible by bsa
    chunks"). BSA (`block_sparse_attention/`) is a DYNAMIC content-dependent sparse
    pattern (block-mean Q/K → top-k / CDF-threshold block selection) in a 946-LoC
    custom Triton kernel — used by the BASE long-video model for multi-batch/cp-split,
    NOT the avatar singletalk. The avatar noise-token self-attn is
    `_process_attn(q_noise, k, v)` = **full dense over all K/V**. So implementing BSA
    (or any windowed/temporal-chunked K/V) for the avatar would be a SPARSE
    APPROXIMATION of the reference dense attention = a QUALITY CUT, which the lever
    rules forbid. **Do not implement BSA for the avatar port.**
  - **The 93f wall is purely a dense-activation memory problem (quality-neutral to
    attack).** lap-04 measured the 93f self-attn sub-segment at 12,675 MiB atomic.
    The lever-2 qkv-cont removal (3× ~613 MB freed) cut it to **12,008 MiB** (probed:
    `ggml_gallocr_reserve` wants 12591433856 B). The card exposes ~11,901 MiB free
    → **93f now misses by only ~107 MiB** (was ~770). 85f still OOMs (frag-class, a
    580 MiB mid-run alloc fails). So the cont cut measurably narrowed the wall but
    didn't clear 93f. Closing the last ~107 MiB needs MORE quality-neutral
    dense-activation reduction in the self-attn sub-segment (candidates, none yet
    done: avoid the qkv_out↔split_qkv-cont 1.84 GB double-buffering by restructuring
    the qkv split; free q/k pre-RoPE earlier; F16 the q_rope/k_rope intermediates).
    These are 93f-only (the 25f hot path already fits) — a turnaround-knob win, not
    hot-path. **81f remains the shipped full-length; 93f is one more activation
    lever away, NOT a fundamental wall.** See PORT-PROGRESS for the exact next cut.

### lever 5 — Q3_K quant ladder: DEAD END on Ampere (measured, no lap)
- DiT is uniform Q4_K (MUL_MAT 38% of the step). Tested whether a smaller Q3_K DiT
  speeds up the matmuls. Requant the DMD-folded q8_0 intermediate → Q3_K (6.84 GB vs
  Q4_K 8.94 GB, −2.1 GB) via `sd-cli -M convert --tensor-type-rules
  "model.diffusion_model.=q3_K"`.
- **VERIFY-FIRST measurement (25f / 2-step / resident weights, the cheap fast path):**
  Q4_K **20.4 s/step** vs Q3_K **21.8 s/step (+7% SLOWER)**. Despite 2.1 GB less weight
  bandwidth, ggml's Q3_K MMQ kernel costs more per matmul on Ampere (3-bit unpack), and
  this DiT is COMPUTE-bound (lap 03), not bandwidth-bound at batch-2 — so smaller weights
  don't help. Same finding class as the memory note "Q5_K_M slower than Q4_K_M on Ampere;
  Q4_K is near the floor." **No speed gain + worse quality → not pursued.** Q3_K gguf
  deleted (kept nothing slower-and-worse around). Q4_K stays the prod DiT quant.
- umT5 text encode is 16.4s one-time on CPU (`--clip-on-cpu`). Tried dropping the
  flag to run it on GPU. → **immediate CUDA OOM at load**: new_sd_ctx allocates
  ALL param buffers upfront, so umT5 (6 GB) + DiT (8.5 GB) = 14.5 GB coexist at
  init and blow the 12 GB card before any compute runs.
- Putting umT5 on GPU would need lazy load order (load TE → encode → free → load
  DiT), which the init path doesn't do — deeper offload surgery for a ~14s
  one-time win. Not worth it. **Keep umT5 on CPU (`--clip-on-cpu`).**
- Free win already banked in lap 01: with the VAE on GPU, the ref-image
  `encode_first_stage` dropped 16.3s (CPU) → 1.9s (GPU). That was the bigger
  one-time cost and it's gone.

### VAE tile-size tuning — 32 (default) is optimal (no lap)
- ~1.5 GB VRAM headroom (peak 10779 of 12288) suggested larger tiles (fewer
  tiles → less per-tile gallocr/warmup overhead) might speed up the decode.
- Measured `--vae-tile-size 48x48`: VAE decode **54s → 73.9s (SLOWER)**, same VRAM.
  The bigger per-tile spatial graph (more activation work per op + larger overlap
  recompute) costs more than the per-tile overhead it removes. So the default
  32-latent tile is the sweet spot. No change shipped.

## SUMMARY (this session)

- **Baseline → best: 768.7s → 238.4s total wall (3.2x), VRAM 10535 → 10779 MiB
  (both fit 12 GB).** All renders stay coherent (ac16≈0.83) and the output webm
  carries muxed audio at 25 fps.
- Phase shape now (8-step default): model load ~13s | text encode (umT5, CPU)
  16.4s | DiT sampling 164s | VAE decode 54s.
- **Winning levers:** (1) fps-25 default + input-audio auto-mux (viewable clips,
  lever 1); (2) **GPU VAE decode via spatial tiling, default-on for the avatar —
  the headline 10.5x VAE / 3.2x total win (lever 2)**.
- **Opt-in lever:** `--steps 6` → 197.4s total (-17%), coherent, but a different
  render vs 8-step; left non-default (lip-sync fidelity is a human call).
- **Dead ends recorded:** full-clip GPU VAE (OOM); GPU text encode (load-time OOM,
  TE+DiT coexist); larger VAE tiles (slower); **CUDA-graph reuse across the 8 DiT
  steps (lap 03 — PROFILED dead: step is 72% MUL_MAT+FLASH compute, only ~5 ms/step
  build/launch overhead to recover)**. DiT step now fully profiled (lap 03); no safe
  default-on pure-speed lever remains, only quality-sensitive steps/quant.

## NEXT (if a future session continues)
- **lap 03 closed the safe-pure-speed search on DiT sampling.** The step is now
  PROFILED and proven compute-bound: MUL_MAT 38% + FLASH_ATTN 34% = 72% irreducible
  Q4_K compute, build/alloc/copy ~5 ms/step. **CUDA-graph reuse is DEAD** (no
  launch overhead to recover + every step rebuilds a fresh graph that would never warm
  up). The ~28% glue is diffuse (CONT/ADD/MUL/SCALE/CONCAT/REPEAT), no single hotspot;
  the one clean removal (audio gate_mul) banked ~0.4%. **Do not re-attempt CUDA-graph
  capture or speculative glue micro-opts — measured/argued dead.**
- The ONLY remaining levers are quality-sensitive, and both attack the 72%:
  1. **Quant ladder (lever 5)** — DiT is uniform Q4_K (38% MUL_MAT). A Q3_K (or
     mixed) DiT would shrink the matmul bandwidth, but ggml MMQ behaviour + the
     ffn.w2/flash F16-overflow guards make it bench-heavy, and coherence must be
     validated with the `check_qkv` oracle. Keep any new gguf OPT-IN. Untouched.
  2. **Fewer steps (lever 4)** — A/B clips rendered (lap 03b): 6-step −17%, 4-step
     −35%, both structurally coherent. **OWNER DECISION PENDING** (lip-sync eyeball;
     see `models/_perf/lap03_{gate_mul,6steps,4steps}.webm`). Default stays 8.
- FLASH_ATTN (34%) is the other half of the 72%. The self-attn runs a 2-pass cond/
  noise split (correctness-load-bearing) + a kv_scale=1/256 F16 guard (adds 3 SCALE
  ops/call, also load-bearing). No safe reduction found; a fused-scale or
  single-pass-when-cond-tiny experiment would be micro (<3%) and risk correctness.
- **The PORT-PROGRESS STATUS block "generated frames still noise" is STALE** — the
  current tree renders coherent talking avatars (the `-out = -out` flow-sign fix
  landed; see build_graph). Updated in this session.
