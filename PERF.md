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
| 03  | DiT-step PROFILED + reusable phase/op profilers + audio `gate_mul` (drop pointless zeros-add) | 237.8 (8) | 10779 | bit-identical to BEST (99 dB all frames) | _this lap_ |

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

### lever 3 — GPU text encode: DEAD END on this VRAM budget (no lap)
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
  TE+DiT coexist); larger VAE tiles (slower). DiT-sampling deep levers (CUDA-graph
  reuse across the 8 steps) not attempted — the step is MUL_MAT-compute-bound at
  ~20s, so launch-overhead caching is low-ROI (cf. memory's prod-vs-profile note).

## NEXT (if a future session continues)
- DiT sampling (164s) is the remaining 69% of wall. The only large lever left is
  attacking the 48-block q4_k DiT compute itself: CUDA-graph capture/reuse across
  the 8 identical-shape steps (deep, multi-day; likely small since compute-bound),
  or a quant-ladder pass (lever 5) — Q3_K on robust DiT tensors to shrink/speed,
  measured against PPL/coherence. Both are quality-sensitive and bench-heavy.
- The PORT-PROGRESS STATUS block is STALE (says generated frames are noise); the
  current tree renders coherent talking avatars. Update it.
