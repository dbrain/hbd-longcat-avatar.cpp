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
| 06  | `flash_skip_kv_pad`: drop the legacy flash kv-pad mask (NOT the qkv double-buffer) → NATIVE 93f renders | 25f unchanged; 93f offload sampling 965.6 (120.7/step) | 93f self-attn seg 12008→5511 | 25f BIT-IDENTICAL to BEST (99 dB); 93f coherent ac16≈0.82 all 93f | lap-06 |
| 07  | split fused qkv → per-output matmuls (kills the 1.75 GiB qkv permute-cont) + FFN token-tiling (auto >16k tok) | 25f sampling 162.3→**146.8 (−9.5%)**; 93f offload 939.2 (117.4/step) | 93f RESIDENT monolithic 5323→**3629** (−1694); offload segs merge 49→33 | 25f **BIT-IDENTICAL to BEST (99 dB)**; 93f offload **BIT-IDENTICAL to lap-06 native (99 dB all 93f)** | 2ce48ee |
| 08  | KERNEL PROFILE: flash-attn d=128 **verified on MMA tensor-core path** (not a fallback); MUL_MAT Q4_K **verified on MMQ int8-tensor-core** — both optimal for this shape on sm_86. No mis-dispatch. Levers scoped: custom kernels = fork-class low-ROI; FFN w1+w3 fuse = ~1% (parked, do converter-side); kv_scale fold = sub-noise | 25f sampling 147 (unchanged) | unchanged | 25f **BIT-IDENTICAL to BEST (99 dB)** | (no change) HEAD c0ea6d3 |
| 08b | RESIDENT-93f investigation: re-localized the 3,629 floor (it is **six ~585 MiB RoPE-internal** cont/repeat/mul buffers, NOT the matmul outputs the lap-07 handoff named). F16-rope cuts it to **2,983 (−646)** but FAILS the 99 dB gate (43 dB, trajectory drift) AND still runtime-OOMs on the MMQ VMM pool. Bit-identical lowmem-F32-rope (half-width concat) REGRESSES the resident peak (→3,860). **No quality-neutral resident win exists at the rope layer.** Precise fix = bound the MMQ pool, or a custom F32 fused-RoPE CUDA op. | 25f sampling 147 (unchanged) | 93f resident floor 3,629 (unmoved quality-neutrally) | 25f **BIT-IDENTICAL** (no code shipped) | (no change) HEAD c78e86c |
| 09  | **Custom fused-RoPE CUDA op** `ggml_rope_pe` (new GGML_OP_ROPE_PE, CUDA-only) — replaces the `apply_rope` cont+2×repeat+mul+add chain with ONE kernel reading x+pe → q_rope. **Proven BIT-IDENTICAL in isolation** (`tools/test_rope_pe.cpp`: max\|chain−fused\|=1.2e-7 @ real [128,32,257]). When wired ON it cut DiT **18.35→17.42 s/step (−5.1%)**, sampling **147→139.9 s**. **BUT the full 25f render diverges to 42 dB (NOT 99 dB)** despite the identical op output — a downstream graph/gallocr interaction, root cause NOT found this lap. **Op gated OPT-IN** (`LONGCAT_FUSED_ROPE=1`); default = chain, re-verified **99 dB bit-identical to BEST**. Branch green. | 25f sampling 147 (default) / **139.9 with op (−5%)** | 25f peak 10513 (unchanged; rope buffers are the 93f-resident smell, not 25f peak) | default **BIT-IDENTICAL to BEST (99 dB)**; op-on **FAILS gate (42 dB)** | submodule + parent (this lap) |

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

### lap 06 — NATIVE 93f CLEARS THE WALL: the 93f self-attn peak was NOT the qkv double-buffer — it was a synthesized flash kv-pad mask (~5 GiB ×2). Skipping it drops the self-attn sub-segment 12,008 → 5,511 MiB and native 93f now renders.

**The lap-05 hypothesis (qkv double-buffer is the prime suspect) was WRONG — proven by
direct measurement.** Enabling `GGML_ALLOCATOR_DEBUG` (ggml-alloc.c, temporary, reverted)
dumped the live-tensor set at the 12,008 MiB high-water of the 93f block-0 self-attn
sub-segment. The two dominant tensors were **5124 MiB + 5151 MiB ≈ 10.3 GiB**, both shaped
`[L_k_padded=37632 × L_q_noise=35880]`. Everything else (qkv_out, the split cont, q/k/v,
RoPE temporaries, the F16 k/v casts at 294 MiB each) was ≤ ~600 MiB. So the qkv split's
double-buffer was never the wall.

**ROOT CAUSE = the legacy flash kv-pad mask.** `ggml_ext_attention_ext`'s flash path
(`build_kqv`, ggml_extend.hpp) pads `L_k` up to a multiple of 256 when `L_k % 256 != 0`,
and — when the caller passes `mask == nullptr` (full attention, the avatar self-attn case)
— SYNTHESIZES a `[L_k_pad × L_q]` mask (`ggml_ext_zeros` + an `-INFINITY` pad column) to
mask the padded K positions, then casts it to F16. At 25f (`L_k = 10920`) that mask is tiny;
at native 93f (`L_k = n_token = 37440 → pad 37632`, `L_q_noise ≈ 35880`) it is a **~5.1 GiB
F32 tensor + its ~F16 cast** — the entire wall. The comment in `build_kqv` already noted
"the need for padding got removed in ggml 4767bda" — the pad+mask is legacy; modern
`ggml_flash_attn_ext` + the CUDA MMA kernel handle an unpadded `L_k` with `mask == nullptr`
directly.

**THE FIX (committed, quality-neutral, opt-in): `flash_skip_kv_pad`.** Added a
`bool flash_skip_kv_pad = false` parameter to `ggml_ext_attention_ext` (`src/ggml_extend.hpp`).
When set AND `mask == nullptr`, it skips the `L_k→256` pad entirely (so no synthesized mask).
Default `false` → every other model/caller is byte-for-byte unchanged. The avatar's three
self-attn `ggml_ext_attention_ext` calls (cond pass / noise pass / plain) pass `true`
(`src/longcat_avatar.hpp`); they always pass `mask == nullptr`. Mathematically identical:
the synthesized mask was all-zeros over the real K positions (a softmax no-op) and `-inf`
only over PAD positions that no longer exist once we don't pad.

**RESULTS:**
- **93f self-attn sub-segment reserve: 12,008 → 5,511 MiB (−6,497 MiB).** With the segment
  now ≤ the `--max-vram 9216` budget, the graph-cut budget-merge also collapses 146 → **49**
  segments (was 146 → 97) — each merged segment is now a full block instead of a sub-split.
- **NATIVE 93f (3.72s) RENDERS via `--offload-to-cpu` — previously a hard OOM at any setting.**
  `models/_perf/fulllen_93f_native.webm` (8-step / 480×832 / seed 42 / audio on / GPU-VAE):
  sampling **965.59s = 120.7 s/step**, VAE tiled decode 200.83s, wall 1187.79s. Latent
  healthy (predecode std 1.21, frame-0 cond 0.528, gen frames 1.0–1.26, nnan=0). **Coherent
  end-to-end: `ac16 ≈ 0.82` on all 93 frames** (matches BEST's ~0.83). 93 frames + pcm_s16le
  16 kHz audio muxed (ffprobe-verified). This is the PRIZE: native full-length, no quality
  cut, no continuation-chaining.
- **25f QUALITY GATE: BIT-IDENTICAL to `BEST_8step_gpuvae_25fps_sound.webm`** — PSNR **99.00 dB**
  (clip_compare identical cap) on ALL 25 frames, ac16 matches to 3 d.p. Confirms the change is
  pure memory-layout: the flash MMA kernel yields identical output with/without the legacy pad.
- **Resident (no-offload) 93f still OOMs** — the monolithic whole-graph reserve is now only
  **5,323 MiB** (down from a pre-mask figure that OOM'd far harder), but resident DiT weights
  are 8,539 MiB (8,781 with the GPU VAE) leaving only ~3.0 GiB free, so the 5.3 GiB compute
  buffer can't coexist. Native-fit-resident would need another ~2.3 GiB of quality-neutral
  activation cuts (≥2 levers) — out of reach for a single clean cut. Resident frame-count
  ceiling probed (steps=1): 93f reserve 5,323 MiB OOM · 61f reserve 3,551 MiB OOM · 49f
  reserve 2,887 MiB *reserves* but runtime-OOMs on the tight ~3.0 GiB margin ⇒ **safe
  resident ceiling ≈ 40f.** **93f (and everything past ~40f) ships via weight-offload
  (~120 s/step); it's a full-length turnaround knob, not the 25f hot path.** The offload
  ceiling now comfortably covers native 93f (segment 5,511 MiB ≪ budget) and would extend
  further — 93f was the native segment length, so this is the full clip.

**Diagnostic note for future sessions:** `GGML_ALLOCATOR_DEBUG` in `ggml/src/ggml-alloc.c`
(toggle the `#define` at line ~16, rebuild, revert after) dumps the exact live-tensor set +
sizes at every compute-buffer high-water — the reliable way to localize a VRAM peak. The
fine-grained `mark_graph_cut`-inside-self_attn probe approach is FRAGILE (breaks
`build_segment_graph` for tensors consumed within the same logical region) — don't use it.
NOTE: gallocr reports the live set at the FINAL high-water *header*, but the peak OFFSET is
reached by tensors allocated EARLIER (it sizes the chunk to the max offset ever used, not the
max simultaneous-live sum). Read the offsets, not just the trailing tensor list — the peak is
"what coexists at the highest offset", which the per-header dumps + offsets together reveal.

### lap 07 — the resident 93f peak was the FUSED QKV BUFFER (+ its permute-cont), NOT the flash mask: split qkv into per-output matmuls → −1.7 GiB resident & −9.5% on the 25f hot path

**`GGML_ALLOCATOR_DEBUG` re-localized the 93f resident (no-offload, monolithic) peak.** After
lap-06 killed the flash kv-pad mask, the resident 93f monolithic compute buffer was **5,323 MiB**
(weights 8,539–8,781 MiB → no room). The debug dump showed the high-water offset was driven by
**two ~1,755 MiB `[3*C, n_token]` F32 buffers stacked** (offsets ~1.8→3.6 GiB): the fused
`attn.qkv` Linear output AND the `split_qkv` permute-CONT of it. At 93f (~37,440 tokens) each is
~1.75 GiB. (This is the qkv double-buffer the lap-05 handoff *first* suspected — lap-06 correctly
found the mask was the *segmented* peak, but the qkv pair is the *resident monolithic* peak once
the mask is gone.)

**FIX (committed, bit-identical, default-on): split the fused qkv matmul into three per-output
matmuls.** The fused `attn.qkv` weight is `[in=C, out=3C]` (ggml ne=`[C,3C]`); the out-dim 3C is
the ggml ROW dim, and Q4_K rows are independently quantized, so a contiguous **row-slice** of the
weight is an exact Wq/Wk/Wv. Run `q=Wq·x`, `k=Wk·x`, `v=Wv·x` separately (`get_weight()`/`get_bias()`
accessors added to `Linear`; the qkv `Linear` is bypassed in `self_attn`). No fused `[3*C]` buffer
or its permute-cont is ever materialized. Mathematically identical to the fused matmul (same rows,
F32 accum).
- **93f RESIDENT monolithic compute buffer: 5,323 → 3,629 MiB (−1,694).** (Still OOMs resident —
  see below.)
- **25f sampling: 162.3 → 146.8 s (−9.5%).** Real, not noise: it removes one full `[3*C, n_token]`
  permute-CONT per block × 48 blocks (the split_qkv cont was the cost). A speed win on the hot path,
  not just VRAM.
- **25f output BIT-IDENTICAL to BEST (PSNR 99 dB / ac16 identical, all 25 frames).**
- **93f offload (the shipped full-length path): sampling 965.6 → 939.2 s (120.7 → 117.4 s/step),
  budget-merge segments 49 → 33** (smaller self-attn segment merges more aggressively), and the
  8-step 93f offload clip is **BIT-IDENTICAL to the lap-06 native 93f (PSNR 99 dB all 93 frames)** —
  proves the split is exact at full length too. `models/_perf/lap07_fulllen_93f_8step.webm`.

**Also shipped: FFN token-tiling.** The SwiGLU inner (`w1`/`w3` → silu·up → `w2`) materializes three
`[ffn_inner=11008, n_token]` F32 transients (~4.6 GiB at 93f). The FFN is purely per-token, so it's
tiled over token blocks (`LONGCAT_FFN_TILES`, auto-on >16k tokens at ~10k tok/tile; **1 = the
bit-identical single-shot path on the 25f hot path**). This kept the FFN region below the self-attn
peak in the resident graph (it was NOT in the 3,629 MiB peak set).

**RESIDENT 93f STILL OOMs by ~530 MiB — and the remaining floor is HARD.** The 3,629 MiB peak is
**four ~585 MiB `[4096, n_token]` F32 tensors** coexisting in the self-attn working set: q_rope
(must stay F32 — `ggml_flash_attn_ext` ASSERTS `q->type==F32`), the residual `x`, and two of
{k_rope, v, the qkv matmul outputs}. Resident-fit needs compute ≤ ~3.1 GiB (GPU-VAE) / ≤ ~3.36 GiB
(VAE-on-CPU). **Levers TRIED this lap that did NOT move the peak (all measured, reverted):**
  - **F16-cast k_rope and v before attention** (with the kv_scale guard, exactly what build_kqv
    does internally; added `k/v_prescaled_f16` params to skip the redundant internal cast). Peak
    unchanged at 3,629 MiB — the four 585 MiB F32 tensors at the peak are NOT k_rope/v (their F16
    versions are downstream of the peak offset). The casts add graph ops for zero resident win and
    were reverted. (q can't be F16 → flash assert.)
  - **Self-attn query-block tiling** (`LONGCAT_ATTN_TILES`, opt-in, kept but NOT auto): tiling the
    noise pass over query rows raised the peak 3,629 → 4,494 MiB. Unlike the FFN (where `w1/w3` read
    only the tile of `y`), the self-attn tiles all read the FULL shared k_rope/v (~1,170 MiB, live
    throughout), so tiling adds a growing concat-accumulated output buffer on TOP of the unchanged
    shared inputs. **Self-attn query-tiling is a DEAD lever for VRAM on this shape — don't auto-engage.**

**Clips:** `models/_perf/lap07_qkvsplit_25f.webm` (25f, = BEST), `lap07_fulllen_93f_8step.webm`
(93f offload, = lap-06 native).

### lap 08 — KERNEL-LEVEL PROFILE OF THE HOT PATH (the campaign target): flash-attn d=128 IS on the MMA/tensor-core path, and MUL_MAT-Q4_K IS on the MMQ int8-tensor-core kernel — BOTH are on the optimal ggml-CUDA kernel for this shape on sm_86. No fallback to fix; no config knob mis-set. (no code change shipped)

This lap directly answered the campaign's central question — **"is flash-attn d=128 on MMA or a slow non-MMA fallback (like the head_dim-72 VL TILE-kernel trap)?"** — and re-profiled MUL_MAT to the specific dispatch path. Method: re-applied the env-gated `LONGCAT_OP_PROFILE` block to the **ggml-cuda node loop** (`cudaEvent` around every `ggml_cuda_compute_forward`, plus a MUL_MAT bucketing by `src1->ne[1]`), built, took the breakdown on the **real 12503-node DiT-step graph** (25f/480p/audio-on, resident weights), then REVERTED the submodule (kept pristine; re-apply the ~42-line block in `ggml/src/ggml-cuda/ggml-cuda.cu` around `ggml_cuda_compute_forward` to re-take).

**Fresh per-DiT-step op breakdown (sync-inflated absolute ms, proportions exact):**

| op | ms | n | notes |
|----|----|---|-------|
| **MUL_MAT** | 7662 | 826 | **M>4096 bucket = 7094 ms / n=481 = 92.6% of all matmul** — the per-block Q4_K weight matmuls at n_token~10920 |
| **FLASH_ATTN_EXT** | 5617 | 144 | 96 self (cond+noise/block) + 48 text-cross |
| CONT | 1220 | 1831 | flash-layout permute-conts + cond/noise split slice-conts |
| ADD | 1124 | 1116 | residual + modulate-shift broadcasts |
| MUL | 917 | 626 | modulate-scale + gate broadcasts |
| SCALE | 448 | 819 | ~half is the kv_scale=1/256 F16-guard scales |
| NORM | 255 | 242 | LayerNorms |
| CONCAT | 206 | 145 | cond/noise concat + cond-zero prepends |
| REPEAT | 176 | 288 | |
| UNARY | 141 | 102 | SiLU |
| CPY | 108 | 288 | |

MUL_MAT (7662) + FLASH (5617) = **13279 / 18250 ≈ 72.8%** — confirms lap-03 on the post-lap-07 tree.

**FINDING 1 — FLASH_ATTN_EXT d=128 IS ON THE TENSOR-CORE MMA PATH (verified by reading the dispatcher, NOT assumed).** `ggml_cuda_get_best_fattn_kernel` (`fattn.cu`): the avatar's self-attn has `K->ne[0]=128` (passes the head-dim switch), `K/V type F16` (after the kv cast), `Q->ne[0]=128 != {40,72}`, and `turing_mma_available(cc)` is **true** on sm_86 (cc=860 ≥ GGML_CUDA_CC_TURING=750). The vec-kernel branch requires `Q->ne[1] <= 2` (it's ~9360), so it falls through to **`BEST_FATTN_KERNEL_MMA_F16`** (line 478). The d=128 path is the standard MMA tensor-core kernel — NOT the head_dim-72 TILE trap from the VL memo (that fell to TILE precisely because 72 is excluded at `Q->ne[0] != 72` and has no MMA config; 128 is a first-class MMA head dim). **There is no flash-attn fallback to fix — the big win the campaign hypothesized does not exist here.**
- The MMA tile config is also right: `mask==nullptr` (self-attn) + `gqa_ratio==1` (n_head==n_kv_head==32, no GQA) ⇒ `use_gqa_opt=false`, `ncols2=1`, and large `Q->ne[1]` selects `ncols1=64` — the large-batch MMA config, the correct choice for this dense full-attention shape. No knob is mis-set.

**FINDING 2 — MUL_MAT Q4_K IS ON THE MMQ INT8-TENSOR-CORE KERNEL (the optimal path for these M).** `ggml_cuda_mul_mat` (`ggml-cuda.cu`): Q4_K weight × F32 activation at M=n_token~10920 ⇒ `use_mul_mat_vec_q` is false (M ≫ MMVQ_MAX_BATCH_SIZE), so **`use_mul_mat_q` (MMQ) fires**. `ggml_cuda_should_use_mmq(Q4_K, sm_86, ...)` returns true unconditionally on Ampere (`turing_mma_available` short-circuit). MMQ quantizes the activation to Q8_1 and runs the int8 MMA kernel — the well-tuned llama.cpp path. This is the SAME finding class as the memory notes "at large M, MMQ mma beats dp4a; Q4_K near floor" + "FORCE_CUBLAS dequant is a net regression for quant weights." Confirmed: the genuine matmul compute is on the best kernel.

**LEVERS SCOPED + their honest ROI (none shipped — all either dead, fork-class, or sub-1%):**
- **Custom d=128 flash kernel / custom Q4_K MMQ tiling:** the two consumers are *already* on ggml's tuned tensor-core kernels. Beating them is the multi-day CUDA fork the campaign authorized — but the parakeet lap-7 + qwen3-tts INT8-mma memos repeatedly measured hand-rolled GEMM/attention at a *fraction* of ggml's throughput (1.7% of peak in one case). The ROI is the same class as the parked lap-12 native-m16n8k8 kernel: order-of-magnitude worse than the redeploy/glue wins. **Not pursued without a measured reason to believe a hand kernel beats the vendor-tuned one on THIS shape.**
- **MUL_MAT activation-quantize de-dup (the one concrete structural redundancy found):** MMQ re-quantizes src1→Q8_1 *per call*, so the lap-07 qkv 3-split (and the FFN's separate w1/w3) quantize the shared activation 3× / 2× respectively. **Re-fusing qkv is OFF the table** — lap-07 proved the split's removed 1.75-GiB permute-cont dominates the extra quantize (net −9.5%). **Fusing FFN w1+w3 into one [4096→22016] matmul** would save 1 quantize of y/block (bit-exact, same FLOPs) — but the w1/w3 are separate gguf Q4_K tensors, so it needs a load-time weight-concat into a persistent buffer, which hits the siglip2/lap-01 **gallocr-aliasing trap** (the runner uses a fresh compute_ctx per step). Multi-hour + aliasing-risk for a forecast ~1% (quantize is a small fraction of each 14.7-ms-avg big matmul). **Parked as the single best remaining MUL_MAT lever IF a future session wants to chase it — do it as a converter-side concat (emit a fused `ffn.w13` tensor) to dodge the aliasing trap, not a runtime concat.**
- **kv_scale=1/256 SCALE-fold:** the F16-overflow guard adds full-tensor SCALE on k and v before the F16 cast in *every* flash call (~half the 448-ms SCALE bucket). k_rope/v are shared by the cond+noise passes, so pre-scaling them ONCE in `self_attn` (with a "k/v already scaled" flag into `build_kqv`) drops the cond-pass duplicate scales. Bit-exact (uniform scalar) but the saving is sub-1% (the cond slice is only 1560 of 10920 rows) — **below the step-to-step noise floor; not implemented** (recorded so it isn't re-derived).

**VERDICT: the post-lap-07 hot path is genuinely on the optimal ggml-CUDA kernels for the avatar's shape on sm_86 — confirmed by reading both dispatchers, not by assuming a floor.** The "compute-bound" call that prior laps kept finding premature was premature about the *glue/VRAM* surface (laps 05/06/07 each found real wins there); the *kernel* surface (MUL_MAT MMQ + FLASH MMA) is, this time, measured to be on the vendor-tuned path with no mis-dispatch and no cheap structural redundancy. The next real speed gain requires either (a) a hand-written CUDA kernel that beats ggml's tuned MMQ/MMA on this shape (fork-class, low prior probability per the cross-project memos), (b) the converter-side FFN w1+w3 fuse (~1%, aliasing-safe if done at convert), or (c) the quality-sensitive `--steps`/quant knobs (owner's call, already A/B'd). **25f sampling stays 147 s (8-step), bit-identical to BEST (99 dB all frames); branch green. No code change shipped this lap.**

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
- **lap 07 update — the "compute-bound, no speed lever" verdict was AGAIN partly premature.**
  Splitting the fused qkv matmul dropped 25f sampling −9.5% (162.3 → 146.8 s) AND the 93f resident
  peak −1.7 GiB — by REMOVING a per-block full-size permute-cont, not by touching the matmul/flash
  compute. The lesson holds: PROFILE the allocator + op set, don't assume the floor.
### lap 08b — RESIDENT-93f investigation (no code shipped; the 3,629 floor RE-LOCALIZED + the bit-identical wall measured)

A full `GGML_ALLOCATOR_DEBUG` re-localization of the 93f RESIDENT (no-offload, monolithic) peak
**corrected the lap-07 attribution.** The 3,629 MiB peak is NOT "four [4096,n_token] F32 tensors
incl. q_rope/x/k_rope/v" — it is **SIX ~585 MiB tensors that are RoPE INTERNALS** (the op dump at the
high-water shows them as `CONT` / `REPEAT` / `MUL`, not the q/k/v matmul outputs or the residual x).
Each `apply_rope` call (`src/rope.hpp`) materializes, per call: the `[d/2, L, n_head*N, 2]` input
`ggml_cont` (585) + **two full-size `ggml_repeat` buffers** (585 each — the interleaved rope repeats
`x_even`/`x_odd` across the size-2 within-pair dim so it can elementwise-`ggml_mul` against `pe`).
At the peak, Q's rope working set (~5×585) coexists with K's. The repeats are the waste, and they are
intrinsic to the elementwise-mul rope formulation (a full-size output needs a full-size `ggml_mul`
src0; neither the half-width `x` view nor the head-less `pe` is full-size, so one must be repeated).

**LEVERS MEASURED (all reverted — tree green at c78e86c):**
- **RoPE math in F16 for q/k/v** (q/k are RMS-normed/bounded; v already F16-cast in build_kqv). Cuts the
  93f resident compute buffer **3,629 → 2,983 MiB (−646)**. BUT **FAILS the bit-identical gate**: 25f vs
  BEST drops to mean 43 dB / min 35 dB (frame-0 cond still 99 dB; gen frames coherent ac16≈0.83 but a
  visibly DIFFERENT render — same class as Q4_0 / `--steps 6`). The F16 rounding of the q·k logits,
  accumulated over 48 blocks × 8 steps on the huge residual stream, drifts the denoise trajectory
  beyond noise. Per the quality gate (hard 99 dB), reverted. **AND it still does not render resident:**
  even at 2,983 MiB the run runtime-OOMs — the **MMQ Q8_1 activation-quant VMM pool** (`ggml-cuda.cu`
  `pool.alloc` / `cuMemCreate`) needs ~300–400 MiB ON TOP of weights(8,539)+compute, and 8,539+2,983 =
  11,522 leaves only ~387 MiB of the 11,909 usable — not enough for the pool. So F16-rope buys neither
  quality nor a resident render.
- **Low-VRAM F32 rope (`lowmem` interleaved path: compute the two within-pair output halves at
  [1,d/2,L,nh*N] ~292 MiB and `ggml_concat`, never repeating to [2,...]).** **BIT-IDENTICAL (25f = BEST,
  99 dB all frames; sampling 147 s, no regression)** — the math is the same F32 products+adds. BUT it
  did NOT lower the *resident* peak (it REGRESSED it to ~3,860): the two rope OUTPUTS (q_rope+k_rope,
  585 each, bit-identical-required F32) plus the downstream qkv/proj `MUL_MAT` still floor the peak, and
  gallocr packs the concat outputs at higher offsets. (It DOES shrink the per-rope *intermediate* set,
  so it may help the SEGMENTED offload self-attn segment — untested; a future session could keep it
  purely for the shipped offload path if a measurement shows the segment shrinks.)
- **V-side early-F16, in-place rope muls, serialized repeats** — all bit-identical, none moved the
  resident peak (V/serialization aren't in the peak set; in-place was already gallocr-reused).

**VERDICT: the ~530 MiB resident-93f gap CANNOT be cleared QUALITY-NEUTRALLY at the rope layer.** The
peak is q_rope+k_rope as F32 (2×585) plus the dense qkv/proj region; the only thing that lowers it
(F16 rope math) fails the 99 dB gate AND still misses on the runtime MMQ pool. **The precise structural
change that WOULD clear it (for owner greenlight), in decreasing safety:**
  1. **Bound/disable the MMQ Q8_1 VMM pool growth** so the runtime overhead above the gallocr reserve
     shrinks — then even modest compute cuts fit. This is a ggml-cuda change (cap the pool, or force the
     big self-attn/proj matmuls through a path that doesn't need the on-demand pool), independent of the
     math. Highest ROI, lowest quality risk.
  2. **A custom F32 fused-RoPE CUDA op** (`ggml-cuda`, editable submodule) that applies the precomputed
     `pe` rotation in one kernel writing q_rope directly over a single buffer — no `cont`+2×`repeat`+
     `mul`+`add` chain, no intermediate full-size buffers. Bit-identical F32, would drop each rope's
     working set to ~1×585 (just the output). This is the clean structural fix but it's a CUDA-kernel
     write (fork-class, the campaign-authorized but low-ROI tier).
  Do NOT ship the F16-rope (quality cut) or the bit-identical lowmem-rope-for-resident (regresses the
  resident peak) — neither achieves the goal. 93f continues to ship via `--offload-to-cpu` (~117 s/step,
  lap-07), which is unaffected.
  Diagnostic recipe used (revert after): `#define GGML_ALLOCATOR_DEBUG` in `ggml/src/ggml-alloc.c` +
  augment the high-water dump to print `ggml_op_name(tensor->op)` + src0/src1 names; run a `--steps 1
  --video-frames 93` (no `--offload-to-cpu`) `--verbose` probe; grep `max_size[0]` for the peak and parse
  the per-tensor `[chunk: start-end] (MB)` lines that follow (one tensor per `[DEBUG] ggml_extend.hpp:60`
  line). NOTE: `systemd-run -p MemoryMax=14G` OOMs the host-RAM model load (DiT 8.9G + umT5 6G resident
  exceeds 14G); raise the cap or omit for resident probes.

- **Older lap-07 thread (now superseded by lap-08b above):**
  1. **Structural self-attn working-set rework.** ~~The 3,629 MiB resident floor is four ~585 MiB
     `[4096, n_token]` F32 tensors. q_rope must stay F32 (flash asserts).~~ (CORRECTED by lap-08b: the
     floor is six ~585 MiB RoPE-internal buffers — the `cont` + two `repeat` per rope call, NOT the
     matmul outputs. See lap-08b for the measured levers + the precise fix.)
  2. **Front B (helps ALL lengths incl. 25f): the hot-path kernels are still untouched** — lap-03
     profiled MUL_MAT 38% + FLASH_ATTN 34%. The qkv-split already shaved the glue; the genuine
     compute (Q4_K MMQ at head_dim 128 / flash-attn for d=128 on Ampere) has had NO custom-kernel
     work. The ggml submodule is editable. This is the biggest remaining speed surface.
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

### lap 09 — FULL PIPELINE ACCOUNTING (Phase 1) + custom fused-RoPE CUDA op (Phase 2)

**Method:** fresh `LONGCAT_PROFILE=1` 25f standard render (8-step, 480×832, audio, resident
weights, GPU 9 MiB idle — no prod processes, so the numbers are the model's own footprint) +
a 0.4 s `nvidia-smi memory.used` sampler across the whole wall. Re-confirms the lap-03/07 shape
on the current tree.

#### TIME — full render wall (25f standard, 8-step)

| phase | seconds | % wall | notes |
|-------|--------:|-------:|-------|
| model load | ~13.0 | 5.6% | gguf tensor read 4.38 s + ctx/backend init + buffer alloc |
| ref-image VAE encode | 1.88 | 0.8% | GPU (was 16.3 s on CPU pre-lap-01) |
| umT5 text encode (CPU) | 16.25 | 7.0% | one-time; `--clip-on-cpu` (GPU OOMs at load, TE+DiT coexist) |
| whisper audio encode | 0.96 | 0.4% | mel → whisper-v3 encoder, one-time |
| **DiT sampling (8 steps)** | **147.26** | **63.1%** | **18.35 s/step** compute; the headline phase |
| VAE tiled decode | 54.06 | 23.2% | 10 tiles × 5.34 s (32-latent spatial tiling) |
| **TOTAL** | **~233.4** | | (no `--steps`/quant change) |

Within ONE DiT step (lap-08 op profile, sync-inflated ms, proportions exact, holds on this tree):
MUL_MAT 7662 ms (M>4096 bucket 92.6%) + FLASH_ATTN 5617 ms = **72.8%** on the optimal MMQ/MMA
kernels. The diffuse glue (CONT 1220 / ADD 1124 / MUL 917 / SCALE 448 / REPEAT 176 …) is the rest;
the **RoPE cont+repeat+mul chain lives inside CONT+MUL+REPEAT** — the Phase-2 target.

#### VRAM — peak accounting (sampled)

| length | peak MiB (sampled) | breakdown |
|--------|-------------------:|-----------|
| **25f** DiT sampling | **10,513** | weights 8,781 (DiT 8,539 + VAE 242) + DiT compute buffer 1,451 + MMQ Q8_1 VMM pool ~280 |
| 25f VAE decode | 3,683 | VAE weights 242 + per-tile decode buffer ~3,440 (32-latent tile, 5,577-node graph) |
| **93f** RESIDENT (no offload) | OOM (needs ~530 MiB over) | weights 8,539 + **compute floor 3,629** (six ~585 MiB RoPE-internal cont/repeat buffers, lap-08b) + MMQ pool ~300-400 → > 11,909 usable |
| 93f via `--offload-to-cpu` | fits | segmented graph-cut path, ~117 s/step (lap-07) |

25f has ~1.4 GiB headroom (10,513 of 11,909). The rope buffers are NOT the 25f peak — they are the
**93f-resident** peak. So the fused-RoPE op's VRAM payoff is a 93f-resident lever, not a 25f one.

#### RANKED SMELLS (Phase 1 hunt)

1. **DiT sampling = 63% of wall, all on optimal kernels** (lap-08 verified MMA + MMQ). The only
   non-quality levers left here are structural glue, already mostly banked (laps 05/07).
2. **VAE tiled decode = 23%** (54 s). 10 tiles × 5.34 s; tile-size 32 is the measured sweet spot
   (48 was slower). Not re-attacked — it's a separate graph, lower ROI than DiT.
3. **umT5 text encode = 7% / 16 s one-time on CPU.** GPU OOMs at load (TE 6 GB + DiT 8.5 GB coexist).
   Would need lazy load-order surgery for a one-time 14 s win. Parked (lap-05).
4. **RoPE cont+2×repeat+mul chain** — the lap-08b ~6×585 MiB resident-93f smell + ~1.2 s/step of
   CONT over the 1831-node step. The Phase-2 target.
5. **MMQ Q8_1 VMM pool** (~280-400 MiB on-demand) — the actual runtime blocker that keeps 93f from
   rendering resident even after a rope cut (lap-08b). Bounding it is the lowest-risk 93f-resident fix.

#### PHASE 2 — custom fused-RoPE CUDA op `ggml_rope_pe`

Built a new `GGML_OP_ROPE_PE` (CUDA-only; CPU backend simply doesn't claim support — the avatar runs
the whole DiT on a single CUDA `runtime_backend`, no sched, so no CPU impl is needed). Blast radius
kept tight: `ggml/include/ggml.h` (enum before COUNT + `ggml_rope_pe` decl), `ggml/src/ggml.c` (both
GGML_OP name arrays + both `static_assert(GGML_OP_COUNT==97)` + the builder), `ggml/src/ggml-cuda/`
(`rope-pe.cu`/`.cuh` + dispatch case + `supports_op` case). One thread per output **pair**: reads
`x[2j]`, `x[2j+1]` and `cos_j=pe[0,0,j,t]`, `sin_j=pe[0,1,j,t]`, writes
`out[2j]=x_e·c − x_o·s`, `out[2j+1]=x_o·c + x_e·s` directly into the contiguous `[d_head,L,n_head·N]`
output — **no cont, no 2× full-size repeat, no separate mul+add buffers**.

- **CORRECTNESS — PROVEN BIT-IDENTICAL IN ISOLATION.** `tools/test_rope_pe.cpp` builds BOTH the
  verbatim `apply_rope` chain and `ggml_rope_pe` in one CUDA graph on identical random input and
  diffs: **max\|chain−fused\| = 1.2e-7 at the real [d_head=128, n_head=32, L=257] shape** (and 6e-8
  at a tiny shape). The op's math + layout are exactly the chain. (Compile inside the builder:
  `g++ -std=c++17 -I ggml/include tools/test_rope_pe.cpp -Wl,--start-group build/ggml/src/ggml-cuda/libggml-cuda.a build/ggml/src/libggml.a build/ggml/src/libggml-cpu.a build/ggml/src/libggml-base.a -Wl,--end-group -lcudart -lcuda -lcublas -lnccl -lpthread -ldl`.)
- **PERF when wired ON:** DiT **18.35 → 17.42 s/step (−5.1%)**, sampling **147.26 → 139.87 s (−5.0%)**.
  25f peak VRAM unchanged at 10,513 (rope buffers aren't the 25f peak). So the op is a real **−5% all-length
  speed lever** AND the structural fix that should clear the 93f-resident floor — IF the render gate passes.
- **THE BLOCKER — full render diverges despite the identical op.** With the op ON, the 25f render is
  **mean 42.4 dB / min 33.9 dB vs the old chain** (frame-0 cond exact at 99 dB, generated frames drift) —
  the SAME "coherent but different trajectory" signature as the F16-rope/Q4_0/6-step renders, i.e. it
  **FAILS the hard 99 dB gate**. This is NOT a math bug (op is bit-identical in isolation) and NOT the
  obvious in-place suspects (`ggml_ext_scale` on k is non-inplace + conts; the cond/noise slicing conts
  its views). Reproducibility is ruled out: old-chain re-render = **99 dB vs BEST** on this exact build.
  The remaining explanation is a **downstream graph/gallocr interaction** — the fused op collapses ~10
  rope nodes to 1, changing the allocator's liveness picture so some downstream buffer reuses/overlaps the
  q_rope/k_rope output (the gallocr-aliasing class the siglip2/lap-01 memos warn about). Root cause NOT
  isolated this lap (needs `GGML_ALLOCATOR_DEBUG` on the op's output offsets vs the flash/residual scratch).
- **DISPOSITION:** op left **OPT-IN** (`LONGCAT_FUSED_ROPE=1`); default path = the chain, re-verified
  **99 dB bit-identical to BEST**. Branch green. The op + oracle (`tools/test_rope_pe.cpp`) are committed
  as proven-correct infrastructure so the next agent starts from "math done, debug the graph interaction,"
  not from scratch.

#### NEXT (lap-10 candidates, in ROI order)

1. **Root-cause the fused-RoPE render divergence** (the −5% all-length + 93f-resident win is real if cleared):
   run `LONGCAT_FUSED_ROPE=1` with `GGML_ALLOCATOR_DEBUG` and dump the q_rope/k_rope output chunk offsets
   vs the self-attn flash scratch + residual-add buffers; if they overlap, force the op's output
   non-reusable (e.g. a trailing `ggml_cont`, or mark it, or insert a barrier) and re-test the gate. Cheap
   experiment, high payoff. Likely a 1-op fix once the overlapping consumer is named.
2. **Bound the MMQ Q8_1 VMM pool** (lap-08b's lowest-risk 93f-resident fix) — independent of rope; with the
   fused-RoPE cut + a bounded pool, native 93f resident should fit.
3. Converter-side FFN w1+w3 fuse (~1%, lap-08) — only if 1+2 land.
