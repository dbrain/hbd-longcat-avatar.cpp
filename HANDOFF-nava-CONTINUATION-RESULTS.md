# NAVA cpp — CLIP CONTINUATION: results + mechanism (2026-06-05)

Branch `nava-port`. Supersedes the "plan" docs (HANDOFF-nava-CONTINUATION-NEXT.md) with
MEASURED results. Build commit base `3e333a7`; this work committed as `1c39120`.

## TL;DR — M=1 is the continuation path. The re-encode N>1 idea is sound but NAVA is i2v.

`nava encode-video` (the N>1 re-encode primitive) is implemented and **bit-exact** vs the
i2v `--image` path at K=1. It works mechanically. BUT pinning M>1 clean anchor frames is
**out-of-distribution for NAVA** (i2v-trained on exactly 1 clean frame) → compounding
"doll-like" high-frequency drift. **Use M=1** (i2v from the prior segment's last decoded
frame): in-distribution, stays photoreal. Motion restarts softly at seams (no velocity
carry) — the warm-start trick (separate task) is the next lever to soften that.

## What shipped (commit 1c39120, examples/nava/main.cpp)

1. **`nava encode-video --vae <gguf> --width W --height H --out <out.bin> [--cuda] f0 f1 …`**
   Re-encodes K decoded PIXEL frames (temporal order) through the Wan2.2 causal VAE into a
   diffusion-latent anchor block `[W_lat,H_lat,M,48]`, M = 1+(K-1)/4. Feed to
   `render --video-anchor <out.bin>`. Input is 5D `[W,H,K,3,1]` internally (decode-symmetric).
   K=1 == `render --image` anchor **bit-for-bit** (verified with `cmp`) — the layout gate.
   Per-frame std diagnostic confirms the causal I/P structure (frame0 I-frame std~0.5,
   frames1+ P-frames ramping to ~0.9).

2. **`NAVA_DUMP_TAIL=<dir>` (+ `NAVA_TAIL_K`, default 9)** on `render`: dumps the last K
   decoded frames LOSSLESSLY from the `rgb` tensor (not the VP9 webm). Chain from these —
   the webm round-trip is otherwise an extra (small) error source. (Turned out NOT to be the
   dominant drift cause — see below — but it's correct hygiene.)

## The drift, measured (var-of-Laplacian of the last frame = HF/"plastic" energy)

3-hop chain, peter command-center clip, q6_K + FA + tile16, 896x448, 13 frames, 10 steps:

| anchor M | seg0 | seg1 | seg2 | seg3 | growth |
|---|---|---|---|---|---|
| **M=1** (i2v last frame) | 60 | 67 | 89 | 101 | **1.7×** |
| M=2 | 60 | 96 | — | — | |
| M=3 | 60 | 127 | 244 | 457 | **7.6×** |

One i2v hop's sharpness scales monotonically with M (67/96/127 for M=1/2/3). Lossless-vs-lossy
chains gave near-identical curves (60→457 vs 58→452 at M=3) → **webm round-trip is NOT the
cause**. Ref-locking anchor frame0 to the original portrait had **zero effect** (126.4 vs
126.8) → **not an appearance-reference-drift problem either**.

## ROOT CAUSE (measured, not guessed)

Same 9 pixels (seg0's frames 41-49), two latent representations:

| frame | as seg0's tail (full 49-frame history) | as a fresh 9-frame re-encode (fed to seg1) |
|---|---|---|
| first | std **0.874** (P-frame, deep in sequence) | std **0.52** (I-frame — RESET) |
| middle | 0.878 | 0.77 |
| last | 0.884 | 0.86 |

The Wan2.2 VAE is **causal-temporal**: the *first* frame of ANY encoded block is forced to a
history-less I-frame (std~0.5) because the encoder has nothing before it. Slicing pixels out
of a clip and re-encoding **wipes the motion/velocity** the full-sequence encode stored
(std~0.88). The DiT in seg1 then sees clean P-frame anchors (frames 1..M-1) with reset history
— input NAVA never trained on (it only ever saw ONE clean I-frame at frame 0). That OOD shows
up as HF junk, compounding each hop.

**Consequence:** NAVA + this causal VAE *structurally* cannot carry true motion velocity across
a hard cut — the re-encode always resets the leading frame. M=1 (one I-frame anchor = exactly
the i2v training regime) is the in-distribution operating point. longcat-avatar avoids this with
a dedicated `cont_latent`/reference conditioning pathway in its DiT (3-way attn split +
ref-RoPE); NAVA's DiT has no such input, so porting true motion-continuity = DiT surgery +
likely a fine-tune (out of scope).

## Recommended continuation recipe (M=1)

```
# seg N: render with NAVA_DUMP_TAIL to get the last frame losslessly
NAVA_DUMP_TAIL=$D/segN_tail NAVA_TAIL_K=1 NAVA_VAE_TILE=16 nava render --cuda \
  --gguf models/nava-dit-q6_k.gguf --context ctx.bin --neg-context vneg.bin \
  --image $D/seg{N-1}_tail/f001.png  (or --image portrait.bin for seg0) \
  --vae models/wan2.2-vae-48ch-f16.gguf --audio-vae models/nava-ltx-audio-vae-f16.gguf \
  --steps 10 --frames 13 --width 896 --height 448 --seed <fresh> --out-name segN
# concat: seg0[full] + segK[from pixel frame 2] (M=1 overlap = 1 frame)
```
Seeds: use a FRESH seed per segment (same-seed was marginally WORSE: 74.5 vs 67.3 — the
anchor dominates, noise realization is independent).

## Eye-server artifacts (http://10.0.0.208:8097)
- `chainL_CONCAT_m1` — 4-segment M=1 chain, photoreal (THE recommended path).
- `chain3_CONCAT_m3` / `chainL_seg{1,2,3}_m3` — M=3, shows the doll-drift (counter-example).
- `chainL_seg1_m1`/`_m2`/`_reflock` — the M-sweep + ref-lock null result.

## Open follow-ups
- **Warm-start trick** (separate task): init segN latent = √(1-σ²)·prev_tail_latent + σ·noise
  (img2img/SDEdit strength) instead of pinning a clean anchor — sidesteps both OOD slots, may
  soften the M=1 seam without doll-drift. ~1 render to evaluate.
- **Audio/speech continuity** across segments needs the shared-audio path (audio-VAE encoder,
  HANDOFF-nava-AUDIO-ENCODER-SPEC.md), NOT the rough per-seam `--audio-anchor` token pinning.
- Optional: continuity-match (exposure/color, avatar_render.cpp:30-71) as a pixel post — minor.
