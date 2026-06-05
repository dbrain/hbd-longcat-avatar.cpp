# NAVA continuation smoothness — visual seam + audio drive (2026-06-06)

Branch `nava-port`. Session goal: smooth, **endless** multi-segment continuation, ideally
driven by **one continuous (user-supplied) audio stream** — a general video-continuity
mission, not just talking-avatar. Owner's framing: *"if I cut mid-clip, what would I be fed
to continue? The new segment should start with that."*

> **Setup (repo path, GPU constraints, build/run commands, model & asset paths, review
> servers, CLI flags, drivers): see `HANDOFF-nava-VIDEO-CONTINUITY-DEEP.md` §"Environment & how
> to reproduce" — it's self-contained and applies to everything below.**

## TL;DR
- **AUDIO DRIVE = SHIPPED + VALIDATED.** External wav → `audio-encode` → per-segment latent
  slices → pinned via `--audio-anchor` → the avatar lip-syncs to it, **continuously across
  seams** (no speech restart). Demo: JFK's voice drives the peter avatar for 3 chained
  segments; rendered-audio-vs-JFK envelope corr **0.999 / 0.998 / 0.998**. This is the
  north-star, working. Eye/ear test pending owner. Clips: `cpp-runs/AUDIODRIVE_{2,3}seg_jfk`.
- **VISUAL seam = lever shipped, eye-tuning parked.** A graded warm-start lever family is in
  (`--warm-latent/-strength/-overlap/-audio`). It can restore liveness but true frame-level
  motion continuity is blocked by NAVA's architecture (see below). M=1 remains the best
  *clean* video path; owner reads its seam as a *natural pause*, not a hard cut.

---

## AUDIO: external-audio drive (the win)

Pipeline (all on the single 3060, serial):
```
wav --(audio-encode)--> [128,T] latent --(slice)--> [128,52]/seg --(--audio-anchor pin)--> render
```
1. **Encoder** (ported by a sub-agent, merged here): `nava audio-encode <ENC_gguf> <in.wav>
   <out.bin> [--cuda]`. CPU-validated vs python `wrapped_encode`: mel parity 3.7e-5, latent
   cosine 0.9997, round-trip SNR 21.2 dB. gguf: `models/nava-ltx-audio-vae-ENC-f16.gguf`
   (separate file; the decode-only `nava-ltx-audio-vae-f16.gguf` is untouched). Checkpoint is
   HEIGHT-causal (spec said width — trust the checkpoint), reuses the decode-side conv blocks.
2. **Slice**: `tools/nava_slice_audio_latent.py in.bin prefix 52 N` → contiguous `[128,52]`
   per segment (one encode, no chunk-boundary edge effects → speech flows).
3. **Drive**: `render ... --audio-anchor seg_k.bin` pins 51/52 audio tokens clean (timestep
   0) → the video conditions on the external audio (sound-to-video). The pinned audio passes
   through to the output (energy-envelope corr **0.995** input-vs-output → confirms the
   external audio is actually driving, not regenerated).
4. **Continuity**: each segment gets its contiguous slice of ONE encoded stream → the voice is
   the same continuous track across the whole chained clip. Validated 0.998+ per seam.

Driver: `/mnt/hdd/nava/warm_exp/audio_drive_demo.sh <wav>` (STEPS, NSEG env). Renders N M=1
segments each pinned to its slice, concats video (drops the 1-frame M=1 overlap), muxes the
continuous source audio.

**Gotchas / polish**: `--audio-anchor` clamps to `audio_len-1` (51/52) so 1 token denoises —
negligible (envelope 0.999); a true "pin all N" flag is a trivial future add. The latent must
be exactly `[128, audio_len]` (audio_len = `max(8, ceil(((frames-1)*4+1)/24*25))` = 52 for 13
frames). Encoder already applies the model's per-channel normalize — do NOT re-normalize.

---

## VISUAL: warm-start lever + why frame-velocity continuity is hard

The problem with M=1 (i2v from the prior segment's ONE last frame): a single frame has **no
velocity** — the model restarts motion from a dead stop at each seam (a falling ball wouldn't
keep falling; a head mid-turn can lock/reverse). Owner perceives it as a forgivable *natural
pause* on the avatar, but it's not true continuity.

**Lever shipped** (`examples/nava/main.cpp`, commit 4623460):
`--warm-latent <prior_final_latent.bin>` (from `NAVA_DUMP_LATENT`) `--warm-strength s`
`--warm-overlap O` `--warm-audio <prior_audio_latent.bin>`. **Graded soft-anchor** design:
warm only the first O *overlap* frames from the prior tail SEQUENCE (carries velocity),
soft-anchor them at `sigma_held` until the schedule reaches it, leave the BODY frames as pure
noise (free → lively), warm+sync the audio so the joint attention never desyncs. Metric:
`tools/nava_seam_metrics.py` (motion-continuity / liveness / sharpness-drift + filmstrip).

**Measured outcomes (q5_k, 896×448, 25 steps; eye-confirmed by owner):**
| variant | liveness (intra motion) | seam | eye verdict |
|---|---|---|---|
| M=1 baseline | ~4 (alive: blinks, cheeks) | natural settle | *"natural pause"* — OK |
| uniform warm s0.75 | — | — | *"a real mess"* (too few denoise steps on noised seed) |
| hybrid (clean anchor + uniform warm) | ~1.5 (frozen) | smooth | *"dead-eyed, jitters"* |
| graded soft-anchor s0.7 o4 | **5.1 (livelier than M=1)** | **seam JUMP (motion 27 vs ~5)** | body diverges from overlap |

**Why graded jumps**: the free body frames need to be *conditioned on a near-clean
multi-frame overlap* to continue its motion — but NAVA is a **1-clean-frame i2v model**;
conditioning on >1 clean frame is out-of-distribution (the prior agent measured M>1 going
"doll-like"). So you can have liveness (free body) XOR seam-faithful motion (clean overlap),
not both, without architecture changes. **True frame-level motion continuity needs DiT
surgery** (a dedicated continuation/reference-conditioning pathway + likely a fine-tune, à la
longcat-avatar's 3-way attn split + ref-RoPE) — out of scope. Honest call, flagged.

**Clips for the owner's eye** (`:8097`): `SEAM_m1_25` (natural baseline), `SEAM_hybrid_s0.7_o4_25`
(dead-eye), `SEAM_gradsync_s0.7_o4_25` (lively but seam jump). Metric is a pre-filter; the
"natural vs dead-eyed" distinction is fundamentally an eye call.

## Recommended product path (today, no DiT surgery)
M=1 video continuation **+** external-audio drive (continuous voice). The audio continuity is
the dominant UX win; M=1's visual seam reads as a natural beat. For motion-heavy general
continuity (ball drop), the warm-start lever is in place to revisit if/when DiT continuation
conditioning is on the table.

## Files
- `tools/nava_seam_metrics.py` — automated seam cleanliness (motion/liveness/sharpness + filmstrip).
- `tools/nava_slice_audio_latent.py` — slice one encoded stream per segment.
- `src/ltx_audio_vae.h` + `audio-encode` subcommand + `tools/convert_ltx_audio_vae.py --with-encoder` — the encoder.
- `/mnt/hdd/nava/warm_exp/{render,chain,audio_drive_demo}.sh` — drivers (working dir, not repo).
