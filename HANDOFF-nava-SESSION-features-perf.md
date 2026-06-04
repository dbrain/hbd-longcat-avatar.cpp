# NAVA cpp — session handoff: voice clone + clip continuation + perf (≤7.5GB)

Branch `nava-port`. This session implemented three of the owner's priorities on top of the
decode-parity/quality baseline (commit `b0b639d`). All work is COMMITTED. The audio-latent
-parity micro-issue is owned by a SEPARATE (paused) agent — I did NOT touch the audio-VAE
**decode** path (BWE/clamp/resample); that's theirs.

Commits this session (newest first):
- `c7244ee` perf: configurable VAE tile (NAVA_VAE_TILE) + sd-cli opus build fix
- `6b710af` enable flash attention on the DiT (default ON)
- `7524f9a` clip continuation via latent chaining (N video + K audio anchors)
- `d5ae1f5` voice clone (ReDimNet spk + SpkToken splice + context flip + timbre_cfg)
- `b0b639d` checkpoint: decode-parity + audio-latent-parity baseline

Review artifacts (owner is headless):
- Ear A/B http://10.0.0.208:8099 — rows 24-28 (see below).
- Eye http://10.0.0.208:8097 — dirs `cpp_peter_CLONE_*`, `chain_*`, `q6k_tile16_seg0`, `perf_baseline`.

---
## 1. VOICE CLONE  [DONE, validated]
The speaker path stubbed in `src/nava.hpp` is implemented. spk_emb is PRECOMPUTED offline
(ReDimNet is NOT ported — it runs once per voice in python; the [1,192] vector is tiny).

- `tools/nava_spk_extract.py <ref.wav> <out.bin>` — ReDimNet 'M' ft_mix → [192] f32 .bin.
  Faithful to `local_audio_vae.py:238-245` (16k, mono mean, 30s clip). Run with the NAVA venv.
- `src/nava.hpp`: SpkToken (LayerNorm192→Linear192-3072→SiLU→Linear3072→out_norm), loaded
  from the gguf (weights were present, unused). `splice_spk()` projects the [192] and writes
  the [3072] token into the text-embedded context at the `<extra_id_2>` position.
- `examples/nava/main.cpp`: `--spk-emb <192.bin> [--spk-pos i,j] [--timbre-cfg S]`. spk_pos
  auto-loads from `<context>.spkpos` (encode-prompt now emits this sidecar). The COND + align
  (mmask) forwards use the spk-spliced context; uncond stays spk=None (= video-neg, both
  streams) — faithful to pipeline_nava.py. timbre_cfg adds the 4th audio-only forward:
  `eps += S*(cond − timbre_uncond)`. Default timbre scale (python) = 2.0.
- Validated: clone speech env_CV 0.57 (peter self), 0.54 (wolverine cross-voice); clone vs
  no-clone audio latent corr 0.44 (materially different — not a no-op). Architecture is a
  line-by-line port of model_mm.py SpkToken + pipeline_nava.py CFG (4 fwd: cond/uncond/mmask
  /timbre). Ear rows 24 (no-clone), 25 (peter clone), 26 (wolverine voice on peter video).
- OPEN: a python+clone(peter) numeric A/B is the only validation NOT run (python render is
  expensive/thrashes; structural port is faithful). Run once to confirm if desired.

### Recommended clone command
```
nava encode-prompt models/longcat-umt5-xxl-q8_0.gguf prompt.txt ctx.bin   # writes ctx.bin.spkpos
python tools/nava_spk_extract.py ref.wav voice_spk.bin                      # NAVA venv
nava render --cuda --gguf models/nava-dit-q6_k.gguf --context ctx.bin \
  --neg-context vneg_now.bin --image peter_896x448.bin \
  --vae models/wan2.2-vae-48ch-f16.gguf --audio-vae models/nava-ltx-audio-vae-f16.gguf \
  --spk-emb voice_spk.bin --timbre-cfg 2.0 --steps 10 --frames 13 --width 896 --height 448 ...
```

---
## 2. CLIP CONTINUATION (latent chaining)  [DONE, validated]
KEY INSIGHT: clip-to-clip chaining needs NO audio-VAE encoder — both video AND audio latents
chain DIRECTLY (a prior segment's denoised latents become the next segment's clean anchors).
The audio-VAE ENCODER is only needed to drive from an EXTERNAL waveform ("shared audio"); see §4.

- `examples/nava/main.cpp`: generalized the i2v single clean-anchor to N video anchor frames
  + K audio anchor tokens (per-token timestep=0, re-spliced after every step).
  `--video-anchor [W,H,N,48]` pins frames 0..N-1; `--audio-anchor [128,K]` pins the first K
  audio tokens. `--image` still gives the N=1 i2v anchor.
- `tools/nava_chain_tail.py --vid <final_latent.bin> --n N --vid-out a.bin --aud <aud.bin>
  --k K --aud-out b.bin` — extracts the trailing N frames / K tokens from a prior render's
  `NAVA_DUMP_LATENT` / `NAVA_DUMP_AUDIO_LATENT`.
- Validated 2-segment chain (chain_seg0 → chain_seg1_n3k13, eye server + ear row 27):
  anchor frames preserved EXACTLY through the trajectory (corr 1.0000), new frames coherent
  -but-new (corr 0.82 vs prior seg = continuing, not copying). VIDEO continuation is clean.
  AUDIO tail-anchor runs but is rough (concat env_CV 0.42, aud_std elevated) — crude
  tail-token pinning is not the right tool for speech continuity; the proper "shared audio
  across segments" wants the LTX encoder (§4).
- NOTE on N>1: python's `first_frame_is_clean` only ever marks ONE frame clean (model_mm.py
  `_first_images_seq_len` = a single frame's H'W'); N>1 anchors are genuinely UNTESTED in the
  trained model. N=3 ran fine numerically; eyeball chain_seg1 motion to judge if N>1 helps or
  hurts. N=1 (= i2v from the last generated frame) is the safe baseline.

### Chaining workflow
```
# seg0
NAVA_DUMP_LATENT=s0_vid.bin NAVA_DUMP_AUDIO_LATENT=s0_aud.bin nava render ... --image img.bin --out-name seg0
# tail
python tools/nava_chain_tail.py --vid s0_vid.bin --n 1 --vid-out a_vid.bin --aud s0_aud.bin --k 13 --aud-out a_aud.bin
# seg1
nava render ... --video-anchor a_vid.bin --audio-anchor a_aud.bin --seed 123 --out-name seg1
# concat seg0 + seg1[trim anchor overlap]  (see the ffmpeg in the session for trim offsets)
```

---
## 3. PERF / VRAM ≤7.5GB  [TARGETS MET]
All numbers @ 896x448 / 13 frames / 10 steps, RTX 3060, peter i2v. VRAM = sampled peak.

| config | peak VRAM | wall | DiT sample | video VAE | audio VAE | audio cos vs q8 |
|---|---|---|---|---|---|---|
| q8, no FA (baseline) | **11039 MiB** | 233s | 141s | 56s | 30s | 1.000 (ref) |
| q8 + FA | 9069 | 179s | 87s | 56s | 30s | 0.996 |
| q4_K + FA | 9069* | 179s | 89s | 56s | 30s | **0.754 (BAD)** |
| q4_K(audio-q8) + FA | 9071* | 180s | 89s | 56s | 30s | 0.783 (still bad) |
| **q4_K + FA + tile16** | **6173 MiB** | 176s | 89s | 46s | 30s | 0.754* (ear-OK) |
| **q6_K + FA + tile16** | **7579 MiB** | **176s** | 95s | 46s | 30s | **0.991 (held)** |

\* once weights shrink, the VAE decode (8.8GB) becomes the ceiling — see tile fix.

WINNING CONFIG = **q4_K or q6_K DiT + flash-attention (default) + `NAVA_VAE_TILE=16`**:
- q6_K: peak **7579 MiB (7.40 GiB)**, audio latent cos 0.991 — the conservative pick.
- q4_K: peak **6173 MiB (6.0 GiB)** — 1.4GB lighter; latent cos 0.754 BUT that metric
  OVERSTATES the perceptual gap (the audio-VAE decode is phase/detail-tolerant: q4_K-vs-q8
  decoded-waveform sample corr ≈ 0 from phase, yet spectral centroid 1354 vs 1424 Hz and
  RMS 0.31 vs 0.25 are close). Owner ear-tested q4_K = OK → q4_K is viable for the budget,
  freeing headroom for higher res / longer clips. Ear rows 28 (q6_K) vs 29 (q4_K) to A/B.
- Latent cosine is a POOR perceptual proxy for the audio stream — judge audio by ear.

CONTINUATION note: N>1 video anchors go out-of-distribution (model only ever saw 1 clean
frame in training) → artifacts (floating mouth). **Use N=1** (= i2v from the last generated
frame); see eye `chain_seg1_n1` / `chain_CONCAT_n1` (stable) vs `chain_seg1_n3k13` (whacky).

Levers, measured (don't re-derive):
- **Flash attention** (commit 6b710af, default ON): the joint self-attn (~5k tokens,
  head_dim 128) was unfused. FA cuts DiT activation ~2GB (11039→9069) and DiT time −38%
  (141→87s), quality held (latent corr 0.996). `NAVA_NO_FLASH=1` reverts. BIG WIN.
- **q6_K vs q4_K**: NAVA's AUDIO stream is very sensitive to 4-bit on the SHARED MMDiT blocks
  (q4_K audio cos 0.75; protecting just the audio-named tensors barely helped → 0.78). q6_K
  preserves it (0.991) at 5.5GB weights. VIDEO survives q4_K fine (0.97). Make k-quants with
  `sd-cli --mode convert -m models/nava-dit-f16.gguf -o OUT --tensor-type-rules ".*=q6_K"`
  (NOTE: ggml type name is `q6_K`/`q4_K` — capital K; lowercase silently no-ops). q4_K/q6_K
  MMQ matmuls are ~same speed as q8 (no DiT speedup from quant; the win is pure VRAM).
- **VAE decode tile** (commit c7244ee): the decode compute buffer scales ~quadratically with
  the latent tile edge. 24×24 = 7.7GB (peak 8.8GB); `NAVA_VAE_TILE=16` = 3.4GB (and 56→46s).
  Default left at 24; set 16 for the budget. Eyeball q6k_tile16_seg0 for tile seams (0.25
  overlap should hide them; bump to 20 if visible).

REMAINING perf levers (NOT done — the "prove the floor" deep dive):
- **DiT phase is the ceiling** (5581 q6_K weights + ~2000 FA activation = 7579). To go lower:
  q5_K (~4.8GB, audio risk — test it) OR weight OFFLOAD (keep q8 quality, stream weights with
  the longcat prefetch-thread pattern — the owner's "offload tuning"; not needed now that q6_K
  fits+holds quality, but it's the path to q8-exact under budget).
- **audio VAE decode 30s for 2s audio (RTF ~15)** — almost certainly the 48k BWE generator
  (a 2nd full vocoder). python's effective output is 16k. `NAVA_AUDIO_DISABLE_BWE=1` exists.
  BUT this is the PAUSED audio-parity agent's correctness territory — coordinate before
  touching. Likely a big speed win (could ~halve the 30s).
- **align_3d_cfg (mmask) forward** is 1 of 3 DiT forwards/step (+33% DiT time). It's quality
  -load-bearing on hard prompts (fixes audio→noise divergence) so it stays on by default, but
  on easy prompts `NAVA_NO_ALIGN_CFG=1` would cut DiT ~33%.
- **ncu kernel profile** of the DiT FFN/attention matmuls (ncu is in the toolchain;
  `--cap-add SYS_ADMIN` for hw counters) to confirm compute-bound vs occupancy-bound — the
  next dedicated perf lap (mirrors flux2 lap-6 etc.). q-vs-q8 same speed ⇒ MMQ/compute-bound.

---
## 4. Audio-VAE ENCODER (external/"shared" audio drive)  [NOT done — spec only]
Needed ONLY for driving from an external waveform (the owner's "feed a shared audio spread
across the whole clip"), NOT for clip chaining (§2). Full spec in the mapping notes; summary:
- Port the LTX `AudioEncoder` (4-level conv: ch=128, ch_mult 1/2/4/8, 2 ResBlocks/level, attn
  at one res, mid, norm_out, conv_out→double_z, take mean 8ch) + mel front-end (torchaudio
  MelSpectrogram n_fft=1024 hop=160 win=1024 n_mels=64 f_max=8000 slaney power=1.0, log clamp
  1e-5) → patchify [T,128] → per-channel-stats normalize (mean/std-of-means, already in gguf).
- The decode-side blocks in `src/ltx_audio_vae.h` (PixelNorm2D, AudioResnetBlock2D, convs) are
  reusable; need a strided AudioDownsample2D + encoder attention.
- ENCODER WEIGHTS ARE NOT IN THE GGUF: `tools/convert_ltx_audio_vae.py` skips
  `audio_vae.encoder.*` (SKIP_PREFIX). Remove that to repack (~44 tensors).
- Validate cpp `encode()` vs python `wrapped_encode` on a wav (latent corr).
- Then wire: external wav → encode → use as the audio clean-anchor / per-segment audio
  condition (the per-token-t=0 audio-anchor plumbing from §2 already exists).

---
## Build / run
```
export PATH=/mnt/hdd/3d/avatar-shootout/toolchain/bin:$PATH
export LD_LIBRARY_PATH=/mnt/hdd/3d/avatar-shootout/toolchain/lib
cmake --build build-nava --target nava -j8        # also: --target sd-cli (for k-quant convert)
```
Models in `models/`: nava-dit-{f16,q8_0,q4_0,q4_k,q4k-audioq8,q6_k}.gguf, wan2.2-vae-48ch-f16,
nava-ltx-audio-vae-f16, longcat-umt5-xxl-q8_0. Scratch/dumps in /mnt/hdd/nava/.
