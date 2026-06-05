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

## Motion test bench + continuity-test VALIDITY (critical — read before any render eval)
**A prompt-driven motion clip CANNOT validate velocity continuity.** If the motion is described
in the text, every continuation segment gets that same text and the model RE-GENERATES the
motion regardless of whether it read the prior frames' velocity. (This also means earlier seam
tests that reused the "man talking" prompt every segment were partly prompt-faked.) So:

- **Valid continuity protocol (owner-confirmed):** establish the motion — ideally an *unexpected*
  motion — in **seg0's prompt ONLY**, then give the continuation a **NEUTRAL prompt that does
  NOT mention the moving subject/direction**. The model's default (neutral prompt + appearance
  anchor) is the *natural* motion; so if the continuation keeps doing the *unexpected* thing,
  that signal can ONLY have come from the prior frames → true velocity continuity. If it reverts
  to the default, the motion reset. Example: seg0 = "a car reversing / rear-facing but moving
  toward camera", seg1 = "a car on a desert highway" (no direction) → does it keep reversing?

- **Verified motion benches** (NAVA renders these; q5_k 896×448, prompts in
  `/mnt/hdd/nava/warm_exp/motion/*.txt`, clips `cpp-runs/MOTION_*`):
  - `car3p_rev` — **best**: 3rd-person car on a desert highway with strong translational motion
    (approaches/grows across the clip). Position+size are trackable and NOT fully pinned by text.
  - `highway_fwd` — 1st-person ground-cam road push (the NAVA demo prompt, verbatim).
  - `headturn` — strong head rotation (profile→back→profile); the owner's named failure case.
  - `orbit` — an element does orbit the head (followed); but it's prompt-defined so weak as a
    *continuity* test (re-faked each segment). Avatar talking-head = WORST bench (near-static).
  Use `tools/nava_motion_anchor.py` for extrapolated anchors. Judge by EYE on :8097 — the metric
  can't tell continued-motion from reset (both read as motion).

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
2. **Streaming / cached VAE encode (best non-LoRA experiment, unimplemented).** This is the
   strongest code-discovered frozen-model path left. The idea is to stop treating every
   continuation segment's first encoded frame as a fresh causal-VAE I-frame. Instead, prime the
   Wan2.2 VAE **encoder** cache with the prior decoded tail, then encode the continuation anchor
   / first frames in the same causal history. If it works, the new segment's initial latent may
   carry some "P-frame" temporal state before the DiT sees it.

   Why this is plausible:
   - NAVA's DiT i2v path receives only one clean frame and no explicit velocity. Python literally
     substitutes frame 0 in `predict_eps()` (`/home/dbrain/dev/NAVA/nava_src/model_nava.py`,
     `xt_reshaped[:, :1] = first_frames[i]`) and then keeps denoising `latents_vision` normally
     in `pipeline_nava.py`. That explains why last-frame chaining preserves pose but loses
     direction.
   - The video VAE is causal in time, so its encoder cache is exactly where short-range temporal
     history exists before the DiT. Full-frame `WanVAE::encode()` currently clears that history at
     both ends: `src/wan.hpp` `WanVAE::encode()` calls `clear_cache()` before encoding and again
     before return.
   - C++ already has reusable **decoder** cache plumbing (`WanVAE::decode_partial()` and
     `WanVAERunner::build_graph_partial()` cache `feat_idx:*`). That is the pattern to copy for
     `_enc_feat_map` / `enc_feat_idx:*`.

   Critical caveat: do **not** assume the resulting cached P-latent can be pinned as a clean i2v
   frame. Previous raw prior P-latent splicing into frame 0 made gibberish, because the DiT's
   clean i2v slot expects an I-frame-like latent. Treat this as an experiment with two possible
   uses:
   - **A. Cached encode as initialization only:** use the cached-encoded tail/next-frame latent to
     warm-start matching frames at an appropriate global sigma, but keep the clean i2v anchor as
     the normal last-frame I-latent. This avoids lying to `first_frame_is_clean`.
   - **B. Cached encode as a weak/noisy overlap:** inject cached latents into an overlap region at
     the same noise level as the rest of the segment, then let the sampler denoise all tokens on a
     matched schedule. Do not mix clean / held / live sigmas inside joint attention; lever #6
     already showed that mismatched per-frame noise levels produce confetti.

   Minimal C++ implementation plan:
   1. Add `WanVAE::encode_partial(...)` next to `decode_partial(...)` in `src/wan.hpp`. It should
      accept one encoder chunk, use `_enc_feat_map`, reset `_enc_conv_idx = 0` per chunk, and
      return the encoded `mu` for that chunk. Mirror the chunking rules from full `encode()`:
      first chunk is 1 frame, later chunks are 4 frames.
   2. Add an encoder-cache branch to `WanVAERunner::build_graph_partial()`. Today it only reloads
      / stores `_feat_map` (`feat_idx:*`) for decoder cache; add parallel reload/store for
      `_enc_feat_map` (`enc_feat_idx:*`).
   3. Add a small runner API or env-gated tool path that can encode a sequence in streaming mode:
      call partial encode for prior tail frames first to fill `_enc_feat_map`, discard or save
      those outputs, then encode the next segment's anchor/first frames without clearing
      `_enc_feat_map`.
   4. **Go/no-go GATE — dump FOUR artifacts, no render needed** (this is the cheap, decisive
      step; do it BEFORE spending any GPU on a render):
      a. fresh I-encode of `last.png` (cache cleared — the M=1 baseline anchor);
      b. cached encode of `last.png` after priming with **1** tail frame;
      c. cached encode of `last.png` after priming with **2–4** tail frames;
      d. cached encode of an **extrapolated next-frame** anchor (`linear.png` from
         `tools/nava_motion_anchor.py`) after priming with the tail.
      For EACH: record mean/std/min/max, **latent L2 + cosine distance from (a)**, and a
      **VAE decode preview** (decode the latent back to pixels and eyeball it) — std alone is a
      first filter, not the whole decision.
   5. **Decision rule** (std is the first cut; distance + decode preview confirm):
      - **P-like, std ~0.8–0.9** (≈ prior tail P-frame): this is effectively the raw-P-latent
        splice — do **NOT** clean-pin it (expect gibberish). → frozen-model continuity is
        exhausted; **stop and go LoRA**.
      - **I-like, std ~0.5–0.7**: might be pinnable. Now ask *does it carry direction?* — check
        L2/cosine vs the fresh I-encode (a): if it's nearly identical to (a) it added nothing;
        if it differs meaningfully AND the extrapolated-anchor (d) differs from (a) too, the
        cache is injecting real temporal signal. → **one render justified.**
      - **Intermediate / ambiguous**: the only genuinely interesting surprise case — escalate to
        a render only if (d) meaningfully differs from (a).
      - **If the only viable use is same-sigma warm init** (anchor still must be the fresh
        I-latent): expect **bridge-at-best**, not the general solution. Still possibly worth it
        for hard seams, but set expectations.
   6. ONLY if the gate passes: conservative render test — normal `--image last.png` clean anchor
      plus the cached latent used as a same-sigma warm init for frame 1 / a small overlap, then
      compare against `MOTION_CHAIN_last` (score 4.43) with `tools/nava_seam_metrics.py` and by
      eye. Remember the metric can't judge dead-eye/confetti/natural — eye is the judge.

   Expected failure modes:
   - If cached P-like latents are clean-pinned, expect the same gibberish/background instability
     as raw P-latent splice.
   - If cached latents are held at a different sigma from neighboring frames, expect confetti due
     to mismatched noise levels in joint audio/video attention.
   - If it only improves frame 0/1 and then the segment still re-plans motion, it is a bridge, not
     a solution. That still may be useful for hard seams.

   Test artifacts already available for comparison:
   - Baseline last-frame chain: `MOTION_CHAIN_last`, score 4.43.
   - Exact Python-style i2v substitution was tested with `NAVA_MODEL_INPUT_ANCHOR=1` and was much
     worse (early noise/confetti; score 22.92 with drop 1), so keep the existing C++ default
     repinned-anchor path for product renders.
   - Align-guidance-off continuation was tested as `MOTION_CHAIN_last_noalign`, score 7.42, worse
     than baseline due to a sharpness/detail jump. Do not chase `NAVA_NO_ALIGN_CFG=1` as the main
     continuity fix.
3. **Crafted velocity-encoding anchor (zero-shot bridge, now tested).** Feed the existing i2v
   slot a single frame that *encodes* direction — an extrapolated "next" frame generated from
   the prior segment's last 2-3 decoded frames. This stays in-distribution for NAVA because the
   model still receives exactly one clean i2v frame. Tool: `tools/nava_motion_anchor.py`.
   Tested on 2026-06-06 with `MOTION_seg0` -> `MOTION_seg1_{last,linear,accel,smear}`.
   Important concat rule: plain `last` is an overlap anchor, so drop seg1 frame 0; extrapolated
   anchors are intended as the next frame, so **do not drop** seg1 frame 0.

   Result: mechanically works as a **one-frame bridge**. `linear_nodrop` / `accel_nodrop` make
   the first generated frame after the seam look like a plausible continuation of the prior
   motion. But the DiT still has no persistent velocity state, so frames 1-2 of the new segment
   settle into a new independent trajectory; metrics show larger luma/sharpness jumps than the
   plain M=1 baseline. Review clips:
   - `MOTION_CHAIN_last` — baseline, drop 1, metric score 4.43.
   - `MOTION_CHAIN_linear_nodrop` — extrapolated next-frame bridge, score 5.44.
   - `MOTION_CHAIN_accel_nodrop` — second-order bridge, score 6.56.
   - `MOTION_CHAIN_smear_nodrop` — damped bridge, score 5.45.

   Honest use: try it when a seam is obviously mid-motion and one bridged frame is more valuable
   than the slight exposure/detail state change. It is not the general active-motion solution.
4. **Calm-head + audio-drive (shippable TODAY for avatars).** The audio drive already makes
   mouth+voice continuous across seams (the hardest part). Owner confirmed neutral heads work.
   So: damp head motion (uniform warm or just prompt/seed for stillness) + audio-drive → seams
   nearly invisible for a talking head. Does NOT generalize to active non-face motion.
5. **Optical-flow interpolation across the seam (post-process).** RIFE/FILM between seg0-last and
   seg1-first. Smooths POSITION but not the velocity reset → "neutral-only," same limit as warm.

## Recommendation
- **Ship now:** M=1 (or calm-head) + external audio-drive for the avatar. The continuous voice
  is the dominant UX win and it works.
- **For the general active-motion mission:** the LoRA continuation adapter (#1) is the honest
  answer — fine-tune is needed, but it's lightweight (LoRA) and self-supervised. Everything
  short of it (frozen-model noise tricks) is provably capped by the matched-noise-level +
  1-clean-frame constraints documented above.
- **Best next frozen-model experiment:** implement streaming/cached VAE encode (#2). It is not
  guaranteed, but it is the only remaining non-training idea grounded in a real motion-carrying
  state in the current codebase.
