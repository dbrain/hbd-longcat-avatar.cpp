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

## Environment & how to reproduce (self-contained — no memory access assumed)
- **Repo:** `/home/dbrain/dev/longcat-avatar.cpp`, branch `nava-port` (sd.cpp/ggml C++ port of
  NAVA = ernie-research 6.3B joint audio+video MMDiT). Python reference model:
  `/home/dbrain/dev/NAVA` (the `nava_src` tree). Scratch/working dir: `/mnt/hdd/nava`.
- **Hardware:** ONE RTX 3060 (12 GB), **serial** — run only one GPU job at a time. **Drive GPU
  jobs from the main agent loop**: Agent-tool sub-agents are NOT woken on `run_in_background`
  completion and will deadlock; if you must use a sub-agent, mandate FOREGROUND commands only.
- **Build (on-server, CUDA):**
  ```
  export PATH=/mnt/hdd/3d/avatar-shootout/toolchain/bin:$PATH
  export LD_LIBRARY_PATH=/mnt/hdd/3d/avatar-shootout/toolchain/lib
  cmake --build build-nava --target nava -j8     # from repo root; warnings-as-errors may be on
  ```
  Binary: `build-nava/bin/nava`. (Same env vars must be set at runtime for the CUDA libs.)
- **Models** (in `models/`): DiT `nava-dit-q5_k.gguf` (owner's prod quant; q6_k also exists),
  video VAE `wan2.2-vae-48ch-f16.gguf`, audio VAE decode `nava-ltx-audio-vae-f16.gguf`, audio
  VAE **encode** `nava-ltx-audio-vae-ENC-f16.gguf` (separate file, has the encoder tensors).
- **Assets** (in `/mnt/hdd/nava/`): `peter_ctx.bin` (umT5 cond context for the "peter" avatar
  prompt, raw [4096,512]), `vneg_now.bin` (video neg context), `peter_896x448_crop.bin` /
  `peter.png` portrait. The render takes a precomputed `--context` bin OR encodes inline via
  `--umt5 <gguf> --prompt`.
- **Run a base render** (q5_k good-settings, 896×448 / 13 frames / ~2s):
  ```
  NAVA_VAE_TILE=16 build-nava/bin/nava render --cuda \
    --gguf models/nava-dit-q5_k.gguf --vae models/wan2.2-vae-48ch-f16.gguf \
    --audio-vae models/nava-ltx-audio-vae-f16.gguf \
    --context /mnt/hdd/nava/peter_ctx.bin --neg-context /mnt/hdd/nava/vneg_now.bin \
    --steps 25 --frames 13 --width 896 --height 448 --cfg 3.0 --shift 5.0 --seed 42 \
    --out-name myrun --runs-dir /mnt/hdd/nava/cpp-runs
  ```
  Writes `/mnt/hdd/nava/cpp-runs/myrun/{clip.webm,meta.json}`.
- **Continuation flags added this session** (`examples/nava/main.cpp`): `--image <png>` (M=1
  i2v anchor = prior segment's last decoded frame); `--warm-latent <bin> --warm-strength s
  --warm-overlap O --warm-audio <bin>` (graded warm-start, lever 6 below). Useful dump envs:
  `NAVA_DUMP_LATENT=<bin>` (final video latent [W/16,H/16,frames,48]), `NAVA_DUMP_AUDIO_LATENT=<bin>`
  ([128,audio_len]), `NAVA_DUMP_TAIL=<dir> NAVA_TAIL_K=1` (last decoded frame as lossless PNG —
  the M=1 chain seed), `NAVA_DUMP_WAV=<wav>` (decoded audio).
- **Review servers** (owner is headless): eye = `http://10.0.0.208:8097` (serves
  `/mnt/hdd/nava/cpp-runs/*/clip.webm` + meta.json, `tools/nava_eyetest_server.py`), ear =
  `http://10.0.0.208:8099`. Put any clip to review in its own `cpp-runs/<name>/` dir.
- **Drivers used this session** (`/mnt/hdd/nava/warm_exp/`, working dir not repo): `render.sh`
  (seg0/m1/warm/hybrid/sync cases), `chain.sh` (multi-seg), `audio_drive_demo.sh` (the
  external-audio showcase). **Metric tool** `tools/nava_seam_metrics.py` needs system `python3`
  (numpy+PIL) + `ffmpeg`.
- **Companion handoffs** (read for the other half of the picture): `HANDOFF-nava-CONTINUATION-
  SMOOTHNESS.md` (audio-drive pipeline + commands), `HANDOFF-nava-CONTINUATION-RESULTS.md`
  (M=1 / `encode-video` primitive + the original drift measurements).

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
   cross-attn adapter. **This is a PYTHON training job, not a C++ change** — do it against the
   reference model at `/home/dbrain/dev/NAVA` (`nava_src`), NOT this C++ port (the port is for
   fast inference; you'd export the trained LoRA to gguf afterward, mirroring how the existing
   DiT was converted via `tools/convert_nava_dit.py`).
   - **Proven architecture shape to copy:** longcat-avatar's continuation conditioning — a 3-way
     attention split + reference-RoPE + a `cont_latent` input. Reference impl lives in this same
     fork's longcat path (search the repo for `cont_latent` / `avatar_render.cpp` ~L198-223 and
     the longcat DiT) and in `~/dev/longcat-video-ref/`. Port that input shape into NAVA's DiT.
   - **Data is self-supervised + FREE, and the dump tooling already exists:** render long clips
     (`--frames 25..` works, coherent to ~4s; model goes to ~10s, VRAM-bound), then for each
     training pair dump (a) the prior segment's tail diffusion latent via `NAVA_DUMP_LATENT` and
     (b) its last decoded frame via `NAVA_DUMP_TAIL`; target = the next overlapping segment.
     Train the adapter to reconstruct segN's latents given segN-1's tail. No labels.
   - **Train only the LoRA, freeze the 6.3B base.** Validate by eye on the seam (the metric
     can't judge it — see harness caveat above). Effort: days + a training loop, NOT a full
     retrain. **This is the only path to reliable ACTIVE-motion continuity. Open scope decision
     for the owner — training infra does not exist yet and must be built in python.**
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
