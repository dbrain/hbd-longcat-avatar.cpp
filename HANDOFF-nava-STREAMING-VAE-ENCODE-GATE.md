# NAVA continuity — Streaming/Cached VAE Encode (lever #2): GATE RESULT — NO-GO (2026-06-06)

Branch `nava-port`. Closes lever #2 from `HANDOFF-nava-VIDEO-CONTINUITY-DEEP.md`. This is the
cheap, decisive gate (no full render needed) for the "prime the Wan2.2 VAE encoder's causal
cache with the prior tail, then encode the continuation anchor as a cached chunk so it carries
P-frame temporal state" hypothesis.

## TL;DR — NO-GO. Frozen-model single-frame continuity is information-theoretically capped. Go LoRA (lever #1).

The streaming cached-encode was implemented and **validated** (parity bit-exact). It *does* inject
a real, tail-dependent change into the anchor latent — but that change is **direction-blind**
(encodes motion *magnitude/presence*, not *direction*), keeps the latent **I-like** in std (not the
true P-frame state), and renders the latent **OOD as a standalone anchor** (degraded decode-back).
A single VAE latent frame cannot carry motion *direction* — forward and reversed motion alias to
nearly the same latent. Since NAVA's i2v slot is exactly one clean frame, **no** single-frame
conditioning (pixel or latent, fresh or cached) can carry velocity. That is the wall.

## What was built (committed)
- `src/wan.hpp`:
  - `WanVAE::encode_partial(ctx, x, i, b)` — per-chunk streaming encode primitive, mirror of
    `decode_partial`. (Plus the `enc_feat_idx:*` encoder-cache branch in
    `WanVAERunner::build_graph_partial`, parallel to the decoder's `feat_idx:*`.)
  - `WanVAE::encode_tail(ctx, x, b)` — **the working path**: in-graph streaming encode (ONE
    forward graph, cache carried as ordinary intra-graph deps, exactly as `encode()` does).
    The cross-graph `build_graph_partial` path is **disabled/buggy** (see the "chunk 1 result is
    weird" note in `_compute`) — driving the cache across `compute()` calls corrupts chunk>=1
    (it blew the latent up to std~3.5 before the switch to single-graph). `encode_tail` also
    front-pads the priming tail to size 1 + 4k so the encoder only ever sees chunk sizes 1 or 4
    (a 3-frame chunk crashes `Down_ResidualBlock`'s avg-shortcut add).
  - `WanVAERunner::encode_streaming(n_threads, frames)` — wraps `encode_tail`; mirrors the
    `encode()` wrapper's `[0,1]->[-1,1]` input scaling (the bypass of which was a parity bug).
- `examples/nava/main.cpp`: `nava encode-stream` subcommand (prime tail frames + 1 target,
  dumps the diffusion-space anchor + a decode-back preview). Positionals = temporal order, last
  is the target. One positional == fresh I-encode (== `render --image` anchor, bit-parity gate).

## Reproduce
```
build: cmake --build build-nava --target nava -j8   (toolchain env per VIDEO-CONTINUITY-DEEP §Env)
gate:  bash /mnt/hdd/nava/gate/gate_run.sh           (drives encode-stream/encode-video + analyze.py)
```
Bench source: a fresh `car3p_rev` seg0 (strong translational motion; the existing `MOTION_car3p_rev`
was a DUMMY-context smoke render) rendered with inline umT5 (`--umt5 longcat-umt5-xxl-q8_0.gguf
--prompt "$(cat warm_exp/motion/car3p_rev.txt)"`), `NAVA_DUMP_TAIL=.../car_tail NAVA_TAIL_K=13`.
Artifacts + previews in `/mnt/hdd/nava/gate/` (`montage_decodebacks.png` is the eye sheet).

## The numbers (all latents are ONE diffusion-space frame [56,28,1,48], baseline = (a))

| artifact | std | cos vs (a) | note |
|---|---|---|---|
| (a) fresh I-encode of f013 (M=1 anchor) | 0.674 | 1.000 | baseline |
| (b) cached, prime 1 tail frame | 0.694 | 0.751 | |
| (c) cached, prime 5 tail frames | 0.737 | 0.784 | |
| (d) cached extrapolated-next (linear)+prime | 0.736 | 0.785 | |
| (e) I-encode of linear, NO prime | 0.686 | 0.969 | extrapolation alone is small |
| ref: true in-seq P-frame (encode-video f009-13, f1) | **0.964** | 0.887 | what a P-frame looks like |
| ref: DiT latent f12 (deep, what NAVA makes) | **1.149** | 0.874 | |
| PARITY: encode-stream(f013) vs encode-video(f013) | — | **1.0000, L2 0.000** | impl validated |

Decode-back fidelity to the true f013 pixels: **(a) 0.0016** (perfect round-trip) vs **(c) 0.046 /
(d) 0.043 / static 0.049** (~28x worse → OOD/degraded as a standalone anchor).

### DEFINITIVE direction control — encode the SAME target f013, vary only the primed tail
| primed tail | cos vs (a) | |
|---|---|---|
| forward (real moving tail f008-12) | 0.784 | |
| static (f013 x5, zero motion) | 0.799 | |
| reversed (f012..f008, opposite motion) | 0.734 | |
| **forward vs reversed (opposite directions)** | **0.953** | should be MAXIMALLY different if velocity were encoded |
| forward vs static | 0.925 | |
| static vs reversed | 0.827 | |

**forward ≈ reversed (0.953) > forward vs static (0.925).** Opposite motion directions produce the
*most similar* latents; the only thing the cache discriminates is motion *presence/magnitude*
(static is the outlier), never *direction*. Direction is information-destroyed in a single causal
latent frame.

## Decision-rule mapping (HANDOFF-nava-VIDEO-CONTINUITY-DEEP step 5)
- **Not P-like** (std 0.69-0.74, vs true P 0.96 / deep DiT 1.15): priming barely lifts std; the
  cached 1-frame chunk is a blend of the I-reset (main residual path is history-conditioned, but
  `AvgDown3D` shortcut is history-blind), not a real P-frame.
- **I-like band → "does it carry direction?"** → **NO** (forward≈reversed control). The large
  cos-0.78 change from (a) is a generic history-perturbation + magnitude, not velocity.
- The literal heuristic ("(d) differs from (a) → one render justified") is technically met, but the
  direction control — a more direct probe than the prescribed 4 artifacts — refutes the premise. A
  render would test a provably direction-blind, OOD-degraded anchor. It is at best a *bridge*, and a
  *weaker* one than the already-tested extrapolated-anchor bridge (lever #3, `linear` scored 5.44 vs
  baseline 4.43 — worse). **Render withheld; it cannot clear the bar the data already sets.**

## Recommendation
**Stop chasing frozen-model continuity. Commit to the LoRA continuation adapter (lever #1).**
The gate generalizes the wall: NAVA's i2v conditioning is one clean frame, and one VAE latent frame
is information-theoretically direction-ambiguous (forward/reverse alias). So velocity cannot enter
the frozen model through the anchor — by any single-frame trick. The only inputs that *can* carry
direction are (a) a multi-frame sequence the weights are trained to read as motion state, or (b) a
trained cross-attn/KV adapter — i.e. the LoRA. The self-supervised data + dump tooling for it
already exist (`NAVA_DUMP_LATENT` / `NAVA_DUMP_TAIL`; longcat's `cont_latent` shape to copy).
Ship M=1 + audio-drive for avatars today; build the LoRA for general active-motion.
