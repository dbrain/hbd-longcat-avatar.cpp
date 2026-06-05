# NAVA video continuity — deep dive: what was tried, why it fails, what's left (2026-06-06)

Branch `nava-port`. Companion to `HANDOFF-nava-CONTINUATION-SMOOTHNESS.md` (audio drive) and
`HANDOFF-nava-CONTINUATION-RESULTS.md` (M=1 / encode-video). This doc is the honest, deep
record of the VIDEO seam-continuity problem: every lever tried, the measured/eye-confirmed
failure mode of each, the root architectural constraint, and the only paths left.

## The goal (owner's framing)
General **endless** video continuity (not just talking-avatar): *"if I cut mid-clip, what
would I be fed to continue? The new segment should start with that."* Must handle **active
motion** — a ball dropping keeps dropping, a head mid-turn keeps turning the same way. NAVA
renders ~2s base clips (13 latent frames), so long content = chaining segments, so the seam is
the whole ballgame.

## The hard constraint (now fully characterized)
NAVA is a **1-clean-frame i2v** model with a **causal-temporal VAE**, denoising video+audio in
ONE **joint attention that assumes every token is at the same noise level each step**. It has
exactly three trained conditioning inputs:
1. **1 clean i2v frame** (frame 0) — carries POSE but **zero velocity** (one still frame).
2. **audio cross-attention** — drives mouth/sound (now used, see audio-drive handoff).
3. **text context** (umT5) — semantic, not motion.

**None carries frame-to-frame velocity.** And two independent constraints block injecting it:
- The i2v slot needs an **I-frame** (causal-VAE first frame, std~0.5). A prior P-frame latent
  (std~0.88, which DOES hold velocity) spliced there → gibberish (measured, prior agent).
- The joint attention needs **matched noise levels** across all tokens each step. Any scheme
  that puts some frames at a different level than others corrupts the rest.

## Levers tried — and their measured/eye-confirmed failure mode
| # | lever | mechanism | result | why |
|---|---|---|---|---|
| 1 | **M=1** (baseline) | i2v from prior LAST frame | photoreal, alive (blink/cheeks); seam reads as a *natural pause* (owner) but motion RESETS — unreliable for active motion | 1 frame = no velocity |
| 2 | M>1 clean anchor | pin K prior frames clean | "doll-like" HF drift, compounds per hop | clean P-frames are OOD (model saw only 1 clean frame) |
| 3 | raw prior-latent splice | put prior P-frame latent in frame-0 slot | gibberish background | I-frame slot needs an I-frame |
| 4 | **uniform warm** (SDEdit, all frames) | warm ALL frames from prior tail, truncated schedule | coherent BUT **dead-eyed**, only ok for **neutral heads** (owner) | warm DAMPS motion → neutralizes; kills facial micro-life |
| 5 | **hybrid** (clean anchor + uniform warm) | frame0 clean + warm rest | **dead-eyed + head jitter** (owner) | over-constrained; still damps |
| 6 | **graded soft-anchor** (this session) | warm ONLY overlap frames, soft-hold at σ_held, body = free noise | livelier than M=1 (motion 5.1) BUT **breaks into CONFETTI after the seam** (owner) | overlap held at σ≈0.66 while body denoises from full noise → **mismatched noise levels in the joint attention** → body can't clean up → residual noise |
| 7 | **longer base clips** | render 25 frames (~4s) | renders coherently (no OOM at 896×448, wan_vae buf 3475MB; coherent to 4s, model goes to ~10s) | **VRAM-bound and doesn't solve it** — just moves the seam; owner still has to plan around it. NOT a fix. |

The **confetti (lever 6) is the key new insight**: it proves you cannot mix per-frame noise
levels in NAVA. That collapses the SDEdit option space to "all matched & free" (= M=1, resets)
or "all matched & warm" (= uniform, dead/neutral-only). Velocity-continuity + liveness is
**provably unreachable** by any noise-init trick on the frozen model.

## Automated test harness (built this session)
`tools/nava_seam_metrics.py` — decodes the chained clip, reports per-seam **motion-continuity**
(seam jump ratio + jerk), **liveness** (median intra-segment motion: frozen≈1.5, alive≈4-5),
**sharpness drift** (per-hop "plastic" growth), + a seam filmstrip. Caveat learned: the metric
ranks seams but **cannot distinguish "natural settle" from "dead-eyed" from "confetti"** —
those are eye calls (motion magnitude alone is ambiguous; confetti reads as high motion). Use
it as a pre-filter, not a judge.

## What's LEFT — honest assessment
1. **DiT continuation pathway via LoRA (the real fix).** Add an input the weights learn to read
   as "prior motion state" — prior-segment tail latents as extra attention KV, or a small
   cross-attn adapter (longcat-avatar's proven shape: 3-way attn split + ref-RoPE + `cont_latent`).
   Train **only a LoRA**, freeze the 6.3B base. **Data is self-supervised + free**: render long
   clips, chop into overlapping (segN-1 tail → segN) pairs, train the adapter to reconstruct
   segN from the tail. Effort: days + a training loop, NOT a full retrain. This is the only path
   to reliable ACTIVE-motion continuity. **Open scope decision for the owner.**
2. **Crafted velocity-encoding anchor (zero-shot long shot).** Feed the existing i2v slot a
   single frame that *encodes* direction — a motion-smear / optical-flow-extrapolated "next"
   frame instead of the sharp last frame. Low confidence (blur ≠ velocity vector; model may just
   sharpen it), but it's ONE render to falsify. Untried.
3. **Calm-head + audio-drive (shippable TODAY for avatars).** The audio drive already makes
   mouth+voice continuous across seams (the hardest part). Owner confirmed neutral heads work.
   So: damp head motion (uniform warm or just prompt/seed for stillness) + audio-drive → seams
   nearly invisible for a talking head. Does NOT generalize to active non-face motion.
4. **Optical-flow interpolation across the seam (post-process).** RIFE/FILM between seg0-last and
   seg1-first. Smooths POSITION but not the velocity reset → "neutral-only," same limit as warm.

## Recommendation
- **Ship now:** M=1 (or calm-head) + external audio-drive for the avatar. The continuous voice
  is the dominant UX win and it works.
- **For the general active-motion mission:** the LoRA continuation adapter (#1) is the honest
  answer — fine-tune is needed, but it's lightweight (LoRA) and self-supervised. Everything
  short of it (frozen-model noise tricks) is provably capped by the matched-noise-level +
  1-clean-frame constraints documented above.
