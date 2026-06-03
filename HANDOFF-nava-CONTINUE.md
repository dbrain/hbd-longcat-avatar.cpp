# NAVA cpp — CONTINUATION PROMPT (paste this to the next agent)

Branch `nava-port` in `/home/dbrain/dev/longcat-avatar.cpp`. The Phase-2 incoherence is **FIXED**
(was a dummy-context path bug in `run_render`, NOT the forward — see HANDOFF-nava-DIVERGENCE.md top).
Text-mode T2V now renders coherent 832×480 clips (`nava_FIXED_832`, `nava_FIXED_seed7` on eye-test
:8097). The DiT forward is bit-faithful to PyTorch F32 (proven: blocks 96–122 dB, heads 80–85 dB).

## How to work (owner's rules — IMPORTANT)
- **Drive GPU/CPU from the MAIN loop**, not from Agent-tool subagents (they stall on
  run_in_background completion → deadlock; see memory `reference_subagent_background_stall`).
  Use subagents ONLY to AUTHOR code/scripts (they must not run heavy/background jobs).
- GPU/CPU is SERIAL — one video job at a time (2 = OOM on the 12GB 3060 / 31GB box).
- cpp builds on this box are FINE. Build:
  `export PATH=/mnt/hdd/3d/avatar-shootout/toolchain/bin:$PATH;
   export LD_LIBRARY_PATH=/mnt/hdd/3d/avatar-shootout/toolchain/lib;
   cmake --build build-nava --target nava -j8` → `build-nava/bin/nava`.
- Validate numerically against PyTorch the way the fix was found: standalone F32 backbone dumper
  (`~/dev/NAVA/nava_dump_f32_step0.py` — applies the `.bfloat16()->.float()` sed patch, loads F32,
  dumps per-block npz) + `tools/nava_tensor_diff.py`. Eye-test = the human's final judgment.
- Render cmd (text-mode, WORKS):
  `LD_LIBRARY_PATH=...toolchain/lib build-nava/bin/nava render --cuda
   --gguf models/nava-dit-q8_0.gguf --vae models/wan2.2-vae-48ch-f16.gguf
   --context .../_ctx/dense_ctx.bin --neg-context .../_ctx/neg_ctx.bin
   --audio-neg-context .../_ctx/audio_neg_ctx.bin --steps 10 --frames 13 --width 832 --height 480
   --seed 42 --cfg 3.0 --out-name X --runs-dir /mnt/hdd/nava/cpp-runs`
  (q8 fits 12GB; f16 OOMs on GPU). Watch the printed per-step latent std — must stay bounded (~<1),
  not run away to 1.8+ (that was the bug). Frame check: `ffmpeg -y -i clip.webm -vf select=eq(n\,24)
  -frames:v 1 f.png` then Read f.png.

## REMAINING WORK (in priority order — the owner wants the project COMPLETED)

### ✅ 1. I2V clean-latent anchor — DONE + WORKING + VALIDATED 2026-06-04
Image→talking-head works (`nava_I2V_face`). Per-token anchor validated vs PyTorch (100–120 dB),
text-mode regression bit-identical. `--image <RGB bin>` (prep via `tools/nava_prep_image.py`).
See HANDOFF-nava.md "✅ I2V" section. So the NEXT priority is Phase-3 audio (item 2).

### (done) 1. I2V clean-latent anchor  ← was IN PROGRESS
NAVA's real use case (image+audio+prompt→video). Plan in HANDOFF-nava.md "NEXT: I2V" section. The
per-token-timestep design (the crux): currently `time_embed` makes a SCALAR e0 [dim,6,1] broadcast
over all tokens; I2V needs PER-TOKEN e0 where the first `h_grid*w_grid` VIDEO tokens (frame 0) get
t=0 and the rest get t (model_mm.py:1459-1463). Refactor `nava.hpp`: time_embed → per-token,
chunk6 → permute chunks to [dim,L,1], NavaHead → per-token e_time, runner → build the per-token t
+ slice e0 per stream. Then `main.cpp`: load image (wire stb/media_io) → WanVAERunner ENCODE
(decode_graph=false, src/wan.hpp:1029/1210) → diffusion latent (inverse of diffusion_to_vae_latents:
`(z-mean)*scale/std`) → splice at frame 0. Validate: (a) regression — per-token t with ALL tokens=t
must bit-match the scalar path (proves text-mode unbroken); (b) numerical vs PyTorch I2V ref
(`~/dev/NAVA/nava_dump_i2v_ref.py`, first_frame_is_clean=True); (c) eye-test an image→video clip.

### ✅ 2. Phase-3 audio — DONE + VALIDATED 2026-06-04. THE RENDER NOW HAS SOUND.
**Audio is SORTED.** The render decodes the co-denoised audio latent through the LTX audio VAE and
muxes Opus into the webm. End-to-end live: `nava_AUDIO_i2v` (832×480, 13f) — webm carries a VP8 video
track + a 48 kHz Opus audio track; the waveform is speech-structured (formant bands <2 kHz, syllable
envelope), the video is a coherent talking head. Eye/ear-test it on :8097.
- **The "T=8-only decoder length bug" in the handoff was already fixed in the binary** — verified a
  full sweep T∈{4,8,12,13,16,24,25,32,48,51,64,100} ALL decode cleanly (waveform = (T·4−3)·480 samp @ 48k).
- **Numerical proof (cpp vs PyTorch `nava_audio_vae_decode_ref.py` on the SAME real co-denoised latent,
  T=18):** waveform corr **0.965** (phase-sensitive!), log-mag spectrogram **0.943**, spectral-envelope
  **0.992**, RMS 0.03797 vs 0.03770 (0.7%). (The handoff's old 0.10/0.64 were on random OUT-of-distribution
  latents → vocoder phase chaos; on a real latent the decode is decisively correct. Residual gap = f16 cpp
  weights vs f32 py + 48k→16k linear resample.)
- **Wiring (committed-ready, in working tree):** `examples/nava/main.cpp` — `--audio-vae <gguf>` (default
  `models/nava-ltx-audio-vae-f16.gguf`, `--no-audio` to silence); after the video VAE decode, frees the
  video VAE, loads `LTXV::LTXAudioVAERunner`, `decode(n_threads, audio_latent)` (RAW latent — unscale is a
  NAVA no-op, per_channel_statistics renorm is inside decode), planar `[n_samp,2]`→interleaved `sd_audio_t`
  @ 48 kHz, passed to `create_webm_from_sd_images`. `examples/nava/CMakeLists.txt` — enables `SD_USE_OPUS`
  (+ `-I${Opus_INCLUDEDIR}` compile-opt, because the conda cross-sysroot toolchain doesn't search host
  /usr/include and CMake strips it from include dirs). `NAVA_DUMP_AUDIO_LATENT=<path>` env dumps the latent.
- KNOWN FOLLOW-UP (perf, not correctness): the audio decode is slow at long T (~28.6s at T=51) — vocoder/bwe
  is unoptimised. Fold into item 3. ReDimNet speaker-enc (voice cloning) still stubbed — later.

#### (historical) the original Phase-3 plan that this resolved:
The audio LATENT [L_audio,128] is ALREADY correctly co-denoised by the DiT in the render loop
(validated). What's missing is a full **LTX-2 audio VAE decoder port** to cpp/ggml:
- Weights: `/mnt/hdd/nava/params/LTX2/ltx-2.3-22b-dev_audio_vae.safetensors` (365 MB).
- PyTorch impl to port: `~/dev/NAVA/nava_src/vae/local_audio_vae.py` (LocalAudioVAEAdapter) +
  `~/dev/NAVA/nava_src/vendor/ltx_core/` (the actual LTX VAE). Read these first.
- Decode recipe (pipeline_nava.py:625): `latents = latents/scaling_factor + shift_factor` (config
  audio_vae.config.{scaling_factor,shift_factor}, line 242-243), then `audio_vae.decode(...)` →
  16 kHz waveform (sample_rate=16000). LTX audio VAE likely emits waveform directly (verify whether a
  separate vocoder exists in ltx_core or it's an end-to-end waveform VAE).
- Steps: (a) converter `tools/convert_ltx_audio_vae.py` → gguf (mirror tools/convert_wan22_vae.py);
  (b) port the decoder in a new `src/ltx_audio_vae.hpp` (or reuse src/ltx_audio_vae.h if it already
  scaffolds it — check); (c) validate decoded waveform vs a PyTorch reference (numerical + listen);
  (d) mux Opus into the webm (heed silent-webm lesson: browsers need Opus/Vorbis not raw PCM — the
  longcat-avatar Opus path is the reference); (e) ReDimNet speaker enc is stubbed (voice cloning, later).
This is multi-hour — treat as its own focused session. Drive validation from main loop; subagents author.

**HEAD-START (found 2026-06-04):** `src/ltx_audio_vae.h` (LTXV namespace, 55 KB) ALREADY implements a
full LTX audio VAE **decoder + vocoder** (auto-detects config from weight names like
`audio_vae.decoder.conv_in.conv.weight`, `vocoder.vocoder.conv_pre.weight`). And the NAVA audio latent
[L,128] = **8 latent_channels × 16 freq_bins** (config.latent_channels=8, latent_frequency_bins=16,
8×16=128) — matches LTXAudioVAEConfig exactly. So the path is likely: (1) write a converter that maps
the NAVA `ltx-2.3-22b-dev_audio_vae.safetensors` tensor names → the LTXV-expected names + emits gguf;
(2) instantiate the LTXV audio-VAE runner in the render, reshape the co-denoised audio latent [L,128]
→ [L,8,16], unscale, decode → 16 kHz waveform; (3) mux Opus. FIRST verify the LTX-2.3 weight names/arch
actually match the in-tree LTXV decoder (they may differ from LTX-Video — diff the safetensors keys vs
the `require(...)` names in ltx_audio_vae.h). If they match, audio is a wiring job, not a port.

**CONFIRMED 2026-06-04: THE NAMES MATCH.** The LTX-2.3 safetensors (1329 tensors) has exactly the keys
the in-tree decoder requires: `audio_vae.decoder.{conv_in,conv_out,up.0.block.0.conv1}.conv.weight`,
`audio_vae.per_channel_statistics.{mean,std}-of-means`, `vocoder.vocoder.{conv_pre,conv_post}.weight`,
and `vocoder.bwe_generator.*` (so has_bwe=true). Prefixes: vocoder.vocoder.* (667), vocoder.bwe_generator.*
(557), audio_vae.decoder.* (56), audio_vae.encoder.* (44, unused for decode), per_channel_statistics (2).
⇒ **Audio = (1) converter ltx-2.3 safetensors→gguf (verbatim names, 5D-conv pack à la convert_wan22_vae.py;
decoder+vocoder+bwe+per_channel_statistics only, skip encoder), (2) wire LTXV audio-VAE runner in render:
reshape co-denoised audio latent [L,128]→[L,8,16] (latent_ch×freq), unscale (/scaling_factor+shift_factor),
decode→16kHz waveform, (3) mux Opus.** Check ltx_audio_vae.h for the runner's public decode signature +
expected input ne-order.

**PROGRESS 2026-06-04: audio VAE gguf BUILT.** `tools/convert_ltx_audio_vae.py` (authored + run) →
`models/nava-ltx-audio-vae-f16.gguf` (356 MB, 1285 tensors: decoder+vocoder+bwe+mel_stft+stats,
encoder skipped; verbatim names since the runner loads with prefix=""). Decode API:
`LTXV::LTXAudioVAERunner(backend, backend, tsm, "")` → alloc_params_buffer → get_param_tensors("") →
load_tensors → `decode(n_threads, latent)`. **Decode input contract** (from AudioDecoder::forward,
ltx_audio_vae.h:841 `reshape_4d(latent, freq_bins*latent_ch=128, ne[2], 1, ne[3])` + runner
`target_time = latent->ne[1]*downsample - (downsample-1)`): the input latent ne is **[16(freq), 8(ch),
time=L_audio, batch=1]** — so the render's co-denoised `audio_latent` [128, L] must be reshaped to
[16,8,L,1]. **VALIDATE the 128→(freq16,ch8) split order against `~/dev/NAVA/nava_src/vae/local_audio_vae.py`
decode** (freq-major vs ch-major — get this right or audio is noise). Then unscale is applied INSIDE
decode via per_channel_statistics (mean/std-of-means) — so likely feed the RAW co-denoised latent
(check whether pipeline_nava's `/scaling_factor + shift_factor` is separate from the VAE's
per-channel stats; line 625 unscale is the LATENT-space one, the per_channel_statistics is the VAE's
own — may both apply). REMAINING: wire decode in render (after video decode, before webm) → waveform
[time] at output_sample_rate (48kHz w/ bwe) → mux Opus (longcat-avatar Opus path is the reference;
the render currently links silent — add Opus). Quick converter sanity: add a `ltx-audio-test <gguf>
<latent.bin>` subcmd calling `LTXAudioVAERunner::load_from_file_and_test` to confirm weights load.

**DONE + VALIDATED 2026-06-04:** added `nava ltx-audio-test <gguf> <latent.bin>` (main.cpp) →
**"ltx audio vae model loaded"** with ALL tensors resolved (converter is correct; no missing/shape
errors). The CPU decode then hit `GGML_ASSERT(src0->type == GGML_TYPE_F16)` (ggml-cpu ops.cpp:6287) —
a CPU-backend op limitation (same class as the nava harness's CPU concessions), NOT a converter bug.
⇒ **Run the audio decode on CUDA** (the render's backend), like the video VAE. So Phase-3 audio
remaining = purely: (1) validate the [128]→[16,8] freq/ch split vs local_audio_vae.py, (2) wire the
CUDA decode of the co-denoised audio latent into the render, (3) Opus-mux. The gguf
(`models/nava-ltx-audio-vae-f16.gguf`) + decoder are ready.

**WIRING DETAILS nailed from `~/dev/NAVA/nava_src/vae/local_audio_vae.py` (2026-06-04):**
- `LocalVideoVAEAdapter.config.scaling_factor=1.0, shift_factor=0.0` (line 163) ⇒ the pipeline's
  latent unscale (`/scaling_factor + shift_factor`, pipeline_nava.py:625) is a **NO-OP for NAVA**.
  Feed the RAW co-denoised audio latent to the VAE; the per_channel_statistics (mean/std-of-means)
  renorm happens INSIDE the cpp decode (ltx_audio_vae.h, already wired). So: no separate unscale.
- `wrapped_decode` (line 103-114): input `latents` is `[batch, 128, T]` (feature=128, time=T), then
  `.transpose(1,2)` → `[batch, T, 128]`, `unpatchify` with shape(channels=z_channels=8, frames=T,
  mel_bins=128//8=16). So the **128 = 8 channels × 16 mel_bins**. The render's `audio_latent` is
  ggml ne `[128, L_audio]` (128=ne0, T=ne1) — that IS `[128, T]`; add batch → `[128, T, 1]` (or the
  ne-order the cpp `AudioDecoder` expects; the CPU run loaded fine, so just match what
  ltx_audio_vae.h:841's reshape consumes — verify channels-major vs mel-major split by trying both
  and listening / comparing to a PyTorch `wrapped_decode` reference on the same latent).
- Output: waveform at `output_sample_rate()` (48 kHz with bwe). Resample if needed, mux Opus.
This is now a mechanical wiring task — the hard unknowns (names, arch, unscale, decoder) are resolved.

**STATUS 2026-06-04 (honest):**
- ✅ gguf loads (converter correct), CUDA decode RUNS → stereo waveform (cpp `nava ltx-audio-test
  <gguf> <latent.bin> [out.bin]`, CUDA inline; CPU hits an F16 assert so MUST use CUDA).
- ✅ Reshape order CODE-CONFIRMED channels-major (channel=idx//16, mel=idx%16): the render's
  audio_latent [128,L] → cpp decode input ne **[16(mel), 8(ch), L, 1]** (plain reshape; ne0=mel fastest
  = idx%16, ne1=ch = idx//16 — matches PyTorch unpatchify `(c f)->...,c=8,f=16`).
- 🟢 **Decode LIKELY CORRECT (numerical, 2026-06-04).** Waveform cross-corr on a random latent was
  only 0.10, BUT that's the wrong metric (vocoder phase chaos on random input + cpp-48k-native-bwe vs
  PyTorch-48k→16k-resample paths). The phase-robust check — **log-magnitude SPECTROGRAM correlation of
  the two output waveforms (cpp 48k→16k vs py 16k)** — is **0.64 with MATCHING energy (cpp 0.0007 vs
  py 0.0006)**. A wrong reshape/decode → garbage energy + ~0 spectral corr; matching energy + broad
  spectral-envelope match ⇒ the channels-major reshape + decode are functionally right. Not 0.9+ (resample
  path + f16-vs-f32 + random-latent), so NOT a bit-exact proof — **final confirmation = wire into render
  with a REAL co-denoised latent and LISTEN** (intelligible speech synced to the head). For tighter
  numerical proof if wanted: tap the decoder MEL (pre-vocoder) both sides, or compare at 48k (tap
  PyTorch pre-resample). Tooling: `~/dev/NAVA/nava_audio_vae_decode_ref.py`, cpp `ltx-audio-test` subcmd.
- 🔴 **cpp DECODER LENGTH BUG (the real audio blocker, found 2026-06-04).** Swept T∈{4,8,12,16,24,25,
  32,48,51}: **ONLY T=8 decodes; ALL others assert** `ggml_nelements(a)==ne0*ne1*ne2*ne3` at
  ggml.c:3725 (a reshape size mismatch). Not a divisibility pattern (16/24/32/48 are mult-of-8 and
  still fail). **PyTorch decodes T=16 fine** (ref gave [1,2,9760]) ⇒ this is a **cpp-side bug in the
  in-tree LTX-Video audio decoder/vocoder** (it was apparently only ever run at one fixed length; some
  reshape in the vocoder STFT/iSTFT or bwe path hardcodes/miscomputes a size that only matches T=8).
  The render needs VARIABLE L_audio (8 for 2 frames, ~51 for 13). **FIX REQUIRED before audio works:**
  build with a debugger / add prints to find WHICH ggml_reshape op at ggml.c:3725 fires (Vocoder or
  BWE forward in ltx_audio_vae.h — grep `ggml_reshape` there), compare its computed dims vs actual at
  T=8 (pass) vs T=16 (fail), fix the length formula. The STFT (mel_hop_length=160, base_upsample_rates
  {5,2,2,2,2,2}=160×, mel_bins=64, latent_downsample_factor=4) length bookkeeping is the suspect.
  Until fixed, audio decode is T=8-only ⇒ NOT renderable at real durations. This moves Phase-3 from
  "wiring" to "**fix cpp decoder length handling, then wire + Opus mux**" — a real debugging task.
- REMAINING: confirm decode (mel-compare or listen) → wire render (reshape+decode after video, handle
  T-pad) → Opus mux (render links silent; add Opus, longcat-avatar Opus path is the reference).

### 3. Q4_K quant + perf
Q4_K is the owner's target end-state quant (q8 is the current expedient; both forwards are faithful).
Then VRAM/RAM-churn lowering, then speed (FA/fusion/occupancy/offload-pipelining à la longcat lap-34).

### 4. Worker isolation + koblem integration (Phase 5)
fork+IPC idle-VRAM-0, /unload, cooperative cancel; koblem heavy bucket; prompt-rewrite via prod 9B
(English→dense Chinese caption, keep `<S>` English). Deferred AV-chaining per HANDOFF-av-chaining.md.

## I2V status (update me as you go)
- PyTorch I2V reference dumper: being authored at `~/dev/NAVA/nava_dump_i2v_ref.py`.
- nava.hpp per-token-timestep refactor: <fill in>.
- main.cpp image-load + VAE-encode + splice: <fill in>.
