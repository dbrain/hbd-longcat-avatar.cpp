# NAVA cpp — MARATHON SESSION BRIEF (paste this to the next agent)

This is the copy-paste prompt for a single long autonomous session. Do everything; don't
stop until the Definition of Done is met. Use subagents/Claude workflows where they help.

---

You are running a LONG autonomous session on the NAVA cpp port (joint audio+video MMDiT),
branch `nava-port` in `/home/dbrain/dev/longcat-avatar.cpp`. Finish clip continuation, the
audio-VAE encoder, and a deep performance/VRAM pass. RUN UNTIL COMPLETE — don't hand back
early. Commit milestones, publish review clips, write handoffs as you go.

READ FIRST (authoritative, self-contained):
- `HANDOFF-nava-CONTINUATION-NEXT.md` — continuation root cause, code refs, plan, recipe.
- `HANDOFF-nava-SESSION-features-perf.md` — voice clone, perf results, audio-encoder spec (§4).
Build on commit `3b5710b` (i2v image preprocessing + VAE-encode Python-parity already fixed;
audio-latent-parity is RESOLVED — the audio divergence was input-side, not the decode path).

LOCKED CONFIG for all renders: **q6_K DiT + flash attention (default ON) + `NAVA_VAE_TILE=16`
+ i2v encode tiling off (default)**. `--gguf models/nava-dit-q6_k.gguf`. Peak ~7.6GB,
~176s/2s-clip. If VRAM tight, drop `--frames` (weights fixed 5.5GB; activation+decode scale
with frames). q4_K (6.0GB, ear-OK) is the headroom alt.

## SCOPE — three tracks, in this order (features change the pipeline; perf comes last)

### TRACK A — Clip continuation finished (smooth multi-segment motion + speech)
1. **N>1 video** (the whole point): the `--video-anchor` SPLICE machinery is correct; the
   raw-latent SOURCE is wrong (Wan2.2 VAE frame0 = causal I-frame std~0.5, frames1+ =
   P-frames std~0.9 — proven). Fix = re-encode the last K decoded PIXEL frames through the
   WanVAE as the anchor block. Add `nava encode-video <frames...> <out.bin>` (generalize the
   single-frame `--image` encode to N pixel frames → `[W,H,M,48]`); retire the raw-latent
   half of `tools/nava_chain_tail.py`. (N=1 already works via `--image last.png`.)
2. **Anti-drift** (so quality doesn't degrade over segments): port longcat-avatar's
   patterns — persistent reference anchor (re-inject seg0's ORIGINAL clean frame-0 every
   segment for identity) + continuity-match (exposure/color). Refs: `avatar_render.cpp:157-259,
   30-71`, `sd_ctx_encode_video_frames` in `stable-diffusion.cpp`.
3. **Audio seam**: refine the `--audio-anchor` token continuation (was rough, env_CV 0.42) or
   defer to the shared-audio encoder path (Track B). Decide + validate.
4. Validate: a 3+ segment chained clip with CONTINUOUS motion + non-restarting speech, no
   drift/gibberish. Numeric gate: re-encoded anchor std ≈0.5 (I-frame); eyeball :8097.

### TRACK B — Audio-VAE ENCODER (shared external audio drive)
Port the LTX AudioEncoder (4-level conv ch=128 ch_mult 1/2/4/8, 2 ResBlocks/level, attn,
mid, norm_out, conv_out→double_z take mean 8ch) + mel front-end (torchaudio MelSpectrogram
n_fft=1024 hop=160 win=1024 n_mels=64 f_max=8000 slaney power=1.0 log clamp 1e-5) → patchify
[T,128] → per-channel-stats normalize. Reuse decode-side blocks in `src/ltx_audio_vae.h`.
Encoder weights NOT in the gguf — un-skip `audio_vae.encoder.*` in
`tools/convert_ltx_audio_vae.py` and repack. Validate cpp `encode()` vs python
`wrapped_encode` (latent corr). Then wire: external wav → encode → audio condition / anchor,
so one shared audio track drives the whole multi-segment clip ("spread across the whole clip").

### TRACK C — Deep perf / VRAM ("prove the floor", run LAST on the final pipeline)
- **ncu kernel profile** of the DiT FFN/attention matmuls (`ncu` is in the toolchain;
  `--cap-add SYS_ADMIN` for hw counters) — confirm compute-bound vs occupancy/latency-bound
  (q6_K≈q8 speed ⇒ MMQ-bound; verify, like flux2 lap-6). Find any inefficiency.
- **audio-VAE 30s for 2s (RTF~15)** — now FAIR GAME (audio-parity done). Almost certainly the
  48k BWE generator (2nd vocoder); `NAVA_AUDIO_DISABLE_BWE=1` exists. Measure the speed win +
  confirm no quality regression (coordinate: this is the formerly-protected decode path).
- **q5_K** test (audio cos + VRAM) for more headroom; **weight-offload** eval (q8-exact under
  budget via the longcat prefetch-thread pattern) — only if wanted.
- **align_3d_cfg** is +33% DiT forwards; quantify its quality value, expose `NAVA_NO_ALIGN_CFG`.
- Deliver: a perf writeup (floor proven from measured angles) + the final prod config + numbers.

## AUTOMATION (the user explicitly wants subagents/workflows used where they fit)
- **Claude workflows / subagents for READ-ONLY + parallel + authoring**: code mapping,
  researching avatar ref-anchor/continuity-match + the encoder layer shapes, drafting the
  `encode-video` subcommand and the encoder port in isolation, adversarial review of findings
  before you commit them. Fan out broadly; verify claims.
- **MAIN LOOP for ALL GPU/build/render/profiling** — Agent-tool subagents STALL on background
  GPU jobs (never wake on completion), and the GPU is SERIAL (one job at a time, 12GB 3060).
  Never run two model loads at once. After killing a GPU job, sleep 2-3s before the next.
- This is a heavy run: lean toward thoroughness, adversarially verify, but serialize GPU.

## HARD CONSTRAINTS
- Build on-server is fine: `export PATH=/mnt/hdd/3d/avatar-shootout/toolchain/bin:$PATH;
  export LD_LIBRARY_PATH=/mnt/hdd/3d/avatar-shootout/toolchain/lib;
  cmake --build build-nava --target nava -j8` (also `--target sd-cli` for k-quant convert).
- Review is HEADLESS: render to `/mnt/hdd/nava/cpp-runs/<name>` (eye http://10.0.0.208:8097
  auto-lists); add audio rows to http://10.0.0.208:8099 (`tools/nava_audio_demo_add_row.py`).
  Validate NUMERICALLY too (dump latents, compare vs python/q8), not just by eye.
- Commit each milestone (no Claude/AI trailers — owner rule). Update the handoff docs + the
  memory note. Don't touch the audio-parity agent's leftover uncommitted tools.
- Python runs THRASH the disk (big model loads) — rare + serial, never concurrent with a GPU job.

## ALREADY PROVEN — DO NOT REDO
- Voice clone works (SpkToken + spk splice + timbre_cfg); ReDimNet runs offline per voice.
- N=1 continuation via `--image last.png` is clean; raw-latent splice at frame0 = gibberish.
- FA (default on) = −2GB / −38% DiT, quality held. q4_K kills audio latent-cos (0.75) but
  ear-OK (latent cosine OVERSTATES perceptual for the audio stream — judge by ear). q6_K holds
  (0.991). VAE decode tile16 = 3.4GB. i2v encode full-frame = isolated phase, doesn't move ceiling.

## DEFINITION OF DONE
3+ segment chained clip (continuous motion + speech, no drift) at the locked config • encoder
ported+validated+wired for shared-audio • perf writeup with the floor proven + final prod
config/numbers • all committed • handoffs + memory updated • clips on :8097/:8099.
