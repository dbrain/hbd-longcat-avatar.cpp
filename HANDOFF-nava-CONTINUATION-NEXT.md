# NAVA cpp — CLIP CONTINUATION: authoritative pickup (next agent)

Self-contained. Read `HANDOFF-nava-SESSION-features-perf.md` for the broader session (voice
clone, perf) and build/run mechanics. This doc is ONLY about chaining clips so motion +
speech continue across segments. Branch `nava-port`. 1× RTX 3060, serial GPU. cpp builds
on-server. Eye :8097 / ear :8099.

## LOCKED CONFIG (use for ALL ongoing continuation work)
**q6_K DiT + flash attention (default ON) + `NAVA_VAE_TILE=16` + i2v encode tiling OFF
(already default).** Peak ~7.6 GB, ~176s per ~2s clip, audio cos 0.991 vs q8. If VRAM ever
gets tight, reduce `--frames` (weights are fixed at 5.5 GB; activation + decode buffer scale
with frame count) — don't drop quant. `--gguf models/nava-dit-q6_k.gguf`.

## TL;DR state
- **N=1 continuation WORKS today** (proven). Re-encode the prior clip's last decoded PIXEL
  frame as an i2v `--image` → render the next segment. Eye: `chain_seg1_reenc` /
  `chain_CONCAT_reenc` (clean) vs `chain_seg1_n1` / `chain_CONCAT_n1` (broken raw-latent).
- **The raw-latent `--video-anchor` path is WRONG and should be retired** for the frame-0
  slot (see root cause). `tools/nava_chain_tail.py` (video half) produces invalid anchors.
- **N>1 is the real goal (smooth MOTION continuity) and is NOT done.** Plan below.
- **Drift over time is a real risk** (the owner's worry) — the prod avatar's anti-drift
  mechanisms are the answer; port them. Refs below.

## ROOT CAUSE (proven 2026-06-05)
Wan2.2 TI2V 16x VAE is causal-temporal. Per `src/wan.hpp:1046-1059` the encoder processes
**frame 0 alone as a 1-frame "I-frame"**, then frames 1+ in **4-frame "P-frame" chunks**.
These live in different statistical regimes — measured on seg0's own final latent:
- frame 0 (the i2v image anchor): **std 0.495**, abs-mean 0.40, range ±2.1
- frames 1-12 (generated):        **std ~0.90**, abs-mean ~0.73, range ±3.8
- frame0-vs-frame12 cosine = 0.75 (genuinely different objects).
Splicing a prior clip's LAST latent (a P-frame, std 0.9) into the next clip's frame-0
(I-frame) slot → the causal decode smears the background to gibberish. Re-encoding the last
decoded PIXEL frame through the VAE instead yields a proper I-frame latent (measured std
0.50054) and a clean render. This matches longcat-avatar's PROD design exactly.

## REFERENCE IMPLEMENTATION (longcat-avatar prod — copy these patterns)
`examples/common/avatar_render.cpp` is the prod avatar that chains long videos. Key parts:
- **Drift sink (default chaining), lines ~198-223**: take the prior segment's last decoded
  PIXEL frames → `sd_ctx_encode_video_frames(...)` → keep the last `num_cond_latents` latent
  frames → feed as `cont_latent` to the next segment. Re-encoding snaps back onto the VAE
  manifold every segment (kills latent-space error accumulation). This is the N>1 primitive.
- **Reference anchor (anti-drift identity), lines ~157-259**: segment 0's frame-0 latent (the
  ORIGINAL clean portrait) is captured ONCE and re-prepended/held fixed on EVERY continuation
  segment, so identity doesn't drift even as motion continues from the (drifting) tail.
- **Continuity match, `continuity_match_segment` lines ~30-71**: per-segment exposure/color
  correction to hide seams.
- Legacy raw-latent passthrough is gated by `LONGCAT_CONT_RAW_LATENT` and is documented there
  as drifting "off the VAE manifold" — i.e. exactly the bug we hit. Don't use it.
- `sd_ctx_encode_video_frames` API: `src/stable-diffusion.cpp:~6262`. WanVAE encode (I/P
  chunking): `src/wan.hpp:930-1065`. The encoder has NO cross-call streaming state (feature
  caches are cleared per encode pass), so multi-frame continuation = re-encode pixel frames,
  NOT persist VAE state.

## NEXT TASKS

### 1. N>1 video continuation (the whole point — smooth motion)
The `--video-anchor` SPLICE machinery in `examples/nava/main.cpp` is CORRECT (pins frames
0..M-1, per-token timestep=0, re-splices each step). Only the anchor SOURCE was wrong (raw
latent). Fix = feed it a VAE-RE-ENCODED block instead:
- Decode seg0 → take the last K decoded PIXEL frames (K = 1 + (M-1)*4 for M anchor latent
  frames, e.g. M=3 ⇒ 9 pixel frames).
- Re-encode those K pixel frames through the WanVAE as ONE block → `[W_lat,H_lat,M,48]`
  diffusion latent (frame0 = I-frame std~0.5, frames1+ = P-frames — the VAE builds the valid
  sequence automatically). The existing `--image` path already does this for K=1; generalize
  it to load `[W,H,K,3]` and `encode_video=true` → `mu [W_lat,H_lat,48,M]` →
  `vae_to_diffusion_latents` → `[W_lat,H_lat,M,48]`, set `n_anchor_v = M`.
- Cleanest shape: add a `nava encode-video <frames...> <out.bin>` subcommand (or extend
  `--image` to accept a multi-frame stack), then chain via `--video-anchor <reencoded.bin>`.
  Update/replace `tools/nava_chain_tail.py` (video half) — it must NOT slice raw latents; it
  should hand off DECODED pixel frames to the cpp VAE encode.
- Validate: M=2/3 anchors → seg1's frames 0..M-1 should match the re-encoded tail AND motion
  should continue smoothly (no floating-mouth/gibberish). Eyeball on :8097.

### 2. Drift-over-time (owner's worry: "last frame is noisier, gets worse each segment")
Real: i2v pins frame 0 to the clean input, so later generated frames drift/blur; seeding the
next clip from the drifting tail compounds. Mitigations (port from avatar_render.cpp):
- **Re-encode (drift sink)** — already the plan above; snaps back to the manifold each seg.
- **Persistent reference anchor** — keep seg0's ORIGINAL clean frame-0 latent and inject it as
  a fixed identity reference every segment (NOT just the drifting tail). For NAVA this means
  combining a motion anchor (last M frames) with an identity reference (original frame 0) —
  decide how to present both to the joint DiT (e.g. reference as an extra clean token set, or
  blend). Design choice — see how avatar feeds ref_latent + cont_latent together.
- **Anchor from a slightly-earlier (cleaner) frame** and/or **overlap segments + crossfade**
  to avoid keying off the single worst frame.
- **Continuity match** (exposure/color) across seams.

### 3. Audio continuation across the seam
Audio latents have NO I/P split, so the audio tail-anchor (`--audio-anchor`, K tokens) is
structurally OK but was rough (env_CV 0.42) — likely needs a smaller K / different blend, or
the proper "shared external audio" path (the LTX audio-VAE ENCODER, NOT ported; spec in the
session handoff §4). Decide: per-seam token anchor vs one continuous audio track sliced per
segment (the owner's "shared audio spread across the whole clip" = encoder path).

## TEST RECIPE (what's on the servers now)
```
# seg0
NAVA_DUMP_LATENT=s0v.bin NAVA_DUMP_AUDIO_LATENT=s0a.bin \
NAVA_VAE_TILE=16 nava render --cuda --gguf models/nava-dit-q6_k.gguf \
  --context /mnt/hdd/nava/peter_ctx.bin --neg-context /mnt/hdd/nava/vneg_now.bin \
  --image /mnt/hdd/nava/peter_896x448.bin --vae models/wan2.2-vae-48ch-f16.gguf \
  --audio-vae models/nava-ltx-audio-vae-f16.gguf --steps 10 --frames 13 \
  --width 896 --height 448 --seed 42 --out-name chain_seg0
# N=1 continuation (CORRECT, proven): re-encode seg0's last decoded frame
ffmpeg -sseof -0.1 -i cpp-runs/chain_seg0/clip.webm -update 1 -frames:v 1 last.png
python tools/nava_prep_image.py last.png 896 448 last.bin
nava render ... --image last.bin --seed 123 --out-name chain_seg1_reenc   # clean
```
Eye :8097 — `chain_seg1_reenc`/`chain_CONCAT_reenc` (correct) vs `chain_seg1_n1`,
`chain_seg1_n3k13` (broken raw-latent). Ear :8099 — row 27 (chain audio).

## Files
- `examples/nava/main.cpp` — render; `--image` (i2v, K=1, the correct primitive),
  `--video-anchor`/`--audio-anchor` (raw-latent splice machinery — splice logic is fine, the
  raw-latent SOURCE is the bug), per-token clean-anchor timestep, `splice_anchor` lambda.
- `tools/nava_chain_tail.py` — RAW-latent tail extractor (video half DEPRECATED; audio half ok).
- `tools/nava_prep_image.py` — PNG → `[W,H,1,3]` bin for `--image`.
- Reference: `examples/common/avatar_render.cpp`, `src/wan.hpp`, `src/stable-diffusion.cpp`.
