# NAVA cpp port — TASK: make the cpp audio-VAE DECODE match python (kill "clippy/warble")

You are picking up a C++ port of NAVA (ernie-research, 6.3B joint audio+video MMDiT).
You have NO shared memory with the previous agent — this doc is self-contained.

## Repos / layout
- C++ port:  `/home/dbrain/dev/longcat-avatar.cpp`  branch `nava-port` (sd.cpp/ggml stack).
- Python reference (ground truth): `/home/dbrain/dev/NAVA`.
- Scratch/data/models: `/mnt/hdd/nava/` (prompts, ggufs, dumps, render outputs, audio A/B).
- Build (CUDA, needs the pinned toolchain):
  ```
  export PATH=/mnt/hdd/3d/avatar-shootout/toolchain/bin:$PATH
  export LD_LIBRARY_PATH=/mnt/hdd/3d/avatar-shootout/toolchain/lib
  cd /home/dbrain/dev/longcat-avatar.cpp
  cmake --build build-nava --target nava -j8
  ```
  Binary: `./build-nava/bin/nava`. (C++ rebuilds on this box are fine.)
- Hardware: 1× RTX 3060 (12 GB VRAM), 31 GB RAM, GPU is SERIAL. NEVER run two model
  loads at once (a python bf16 model ~12 GB + a cpp f16 ~12 GB = RAM OOM). After
  killing a GPU process, sleep a couple seconds before the next (VRAM release race).

## Already FIXED — do NOT re-investigate (the speech is correct now)
The original bug was garbled SPEECH on the hard sample "peter". Root cause was the umT5
TEXT context, now fixed in `examples/nava/main.cpp` `encode-prompt` (committed):
1. The python data loader inserts the umT5 sentinel `<extra_id_2>` (id 256297) after
   each `<S>` speech marker (`NAVA/nava_src/data/t2v.py:391`,
   `text.replace("<S>","<S><extra_id_2>")`, use_speech_special_token=false). cpp now
   does the same, WITH a trailing space (so sentencepiece re-adds the metaspace to the
   next word: `<extra_id_2>We` → `[<extra_id_2>, ▁We]`). cpp tokens == HF (0 diffs).
2. NFKC full-width punctuation + EOS-at-511 truncation (older fixes in nava_normalize_text).
The DiT forward is also validated faithful (per-block audio-token cos 0.9999, audio
velocity cos 0.9999 at all timesteps, q8≈fp8). The align_3d_cfg 3-way CFG is implemented
and correct. So: text, tokenizer, DiT forward, scheduler — all faithful. Don't chase them.

## YOUR TASK — audio-VAE DECODE parity
With correct text + a given audio latent, the cpp render's audio is intelligible but the
TIMBRE differs from python: owner hears "warbles / clippy", worse on some seeds. Part of
the thinness is genuine seed/latent variance (reproduced in python too), BUT there is a
REAL, bounded cpp-DECODE coloration on top of it. Goal: make the cpp audio-VAE decode
match python's decode of the SAME latent, so any latent decodes to python-quality audio
(owner does NOT want to re-roll seeds to dodge it).

### Evidence (decode the SAME latent two ways)
- cpp decode vs python decode of the identical audio latent: waveform correlation only
  **~0.97** overall (windowed 0.92–0.99) — not 1.0.
- cpp **hard-clamps the waveform to ±1.0** (`src/ltx_audio_vae.h`, the BWE `ggml_clamp`
  ~line 978); python clamps to **±0.99** (`NAVA/nava_src/vae/local_audio_vae.py`
  `_audio_post_process`, `limit=0.99`). The hard ±1.0 clamp reads as "clippy" and hits
  more on loud seeds.
- cpp decode is ~**0.7× quieter** (sample RMS) than python decode of the same latent —
  a gain/scale discrepancy somewhere in the decode chain.
- SAMPLE-RATE / BWE mismatch (IMPORTANT): python's final audio is **16 kHz, NO bandwidth
  extension** (`LtxAudioVAE.wrapped_decode` = mel-decoder → vocoder → resample to 16k;
  `init_ltx_vae` builds only encoder/decoder/vocoder, **no bwe_generator**). The cpp
  decode runs an extra **16k→48k BWE generator** (the gguf has `vocoder.bwe_generator.*`
  weights; `config.has_bwe` auto-on) and outputs **48 kHz**. The BWE >8 kHz band is tiny
  (~0.27% energy) so it's not the main timbre issue, but it IS a structural divergence
  from python — decide whether to match python (16k, skip BWE) or keep BWE and ensure
  the ≤8 kHz band matches.

### The decode chain to audit (cpp vs python, side by side)
- cpp:    `src/ltx_audio_vae.h` — `LTXAudioVAE::decode()` (mel decoder → vocoder → BWE),
          `target_time = latent_len*4 - 3`, per-channel statistics (mean/std) applied in
          `AudioDecoder::forward`, vocoder, BWE generator + `crop_waveform_samples`.
  - cpp render audio path + the decode call: `examples/nava/main.cpp` ~line 947-985.
  - standalone cpp decode of a latent: `./build-nava/bin/nava ltx-audio-test
    models/nava-ltx-audio-vae-f16.gguf <latent.bin> <out_wave.bin>` (CUDA; CPU hits an
    F16 assert).
- python: `NAVA/nava_src/vae/local_audio_vae.py` — `LtxAudioVAE.wrapped_decode`
  (unpatchify → decoder → vocoder → resample 48k→16k), `_audio_post_process` (clamp 0.99).
  - Reference decode of a cpp latent (THE ORACLE):
    `/mnt/hdd/nava/.venv/bin/python /home/dbrain/dev/NAVA/nava_audio_vae_decode_ref.py
     <latent.bin> <out_wave.bin>`  → 16 kHz waveform .bin (ne [n_samples, 2]).
  - The audio-VAE checkpoint dir it loads: `/mnt/hdd/nava/params/` (appends
    `LTX2/ltx-2.3-22b-dev_audio_vae.safetensors`). The gguf was converted via
    `tools/convert_ltx_audio_vae.py`.

### Ready-made test inputs (no GPU needed to compare, once you have a latent)
- A real cpp audio latent (peter, seed 42, fixed context), ggml ne [128, 52]:
  `/mnt/hdd/nava/cpp_peter_audlat_FIXED.bin`
- Decode it both ways and compare per-window:
  ```
  # python (oracle, 16k, no BWE)
  /mnt/hdd/nava/.venv/bin/python /home/dbrain/dev/NAVA/nava_audio_vae_decode_ref.py \
     /mnt/hdd/nava/cpp_peter_audlat_FIXED.bin /tmp/py_dec.bin
  # cpp (48k + BWE)
  ./build-nava/bin/nava ltx-audio-test models/nava-ltx-audio-vae-f16.gguf \
     /mnt/hdd/nava/cpp_peter_audlat_FIXED.bin /tmp/cpp_dec.bin
  # convert .bin -> wav: python3 tools/nava_bin_to_wav.py <in.bin> <out.wav> [sr]
  ```
- Existing A/B already rendered (same seed-42 latent): clip 10 = cpp decode,
  clip 11 = python decode (see eye/ear server below).
- `.bin` format (all of these): little-endian int32 `n_dims, name_len, ggml_type(0=f32)`,
  then int32 `dims[n_dims]` in ggml ne-order (ne[0] = fastest), then `name[name_len]`
  bytes, then f32 payload ne[0]-fastest. (Readers: `tools/nava_npz_to_bin.py`,
  `tools/nava_bin_to_wav.py`, `NAVA/nava_audio_vae_decode_ref.py read_bin`.)

### Suggested approach
1. Decode `cpp_peter_audlat_FIXED.bin` with BOTH paths; resample cpp 48k→16k; align and
   compute per-window correlation + RMS ratio + peak. Confirm the ~0.97 / 0.7× / clamp.
2. Walk the chain to find where they diverge: dump the intermediate mel (after the mel
   decoder) and the vocoder output in both, compare. The `_ref` script and
   `tools/check_mel.py` / `tools/compare_vae_recon.py` may help; add taps as needed.
3. Likely suspects, in order: per-channel-statistics (mean/std) application & ordering
   (the latent splits 128 = 8 z-channels × 16 mel-bins, CHANNEL-major: ch=idx//16,
   mel=idx%16); vocoder weight/scale; the BWE path & final crop; the ±1.0 vs ±0.99 clamp;
   any global gain. Fix cpp to match python's 16k output bit-for-behavior on the ≤8 kHz
   band; decide BWE policy with the owner.

## HOW TO RUN a full render (produces clip.webm + meta.json)
```
export PATH=/mnt/hdd/3d/avatar-shootout/toolchain/bin:$PATH
export LD_LIBRARY_PATH=/mnt/hdd/3d/avatar-shootout/toolchain/lib
cd /home/dbrain/dev/longcat-avatar.cpp
./build-nava/bin/nava render --cuda \
  --gguf models/nava-dit-q8_0.gguf \
  --context <ctx.bin> \
  --neg-context /mnt/hdd/nava/vneg_now.bin --audio-neg-context /mnt/hdd/nava/aneg_now.bin \
  --image /mnt/hdd/nava/peter_896x448.bin \
  --vae models/wan2.2-vae-48ch-f16.gguf --audio-vae models/nava-ltx-audio-vae-f16.gguf \
  --steps 10 --frames 13 --width 896 --height 448 --fps 24 \
  --cfg 3.0 --cfg-align 3.0 --cfg-align-audio 2.0 --shift 5.0 --seed 42 \
  --runs-dir /mnt/hdd/nava/cpp-runs --out-name <NAME>
```
- `--cuda` is REQUIRED (defaults to CPU otherwise — hours). DiT must be q8 (f16 won't
  fit 12 GB). ~4 min/render.
- `<ctx.bin>` = a raw umT5 context [4096,512]. Make one from a prompt with:
  `./build-nava/bin/nava encode-prompt models/longcat-umt5-xxl-q8_0.gguf
   /mnt/hdd/nava/peter_prompt.txt /mnt/hdd/nava/myctx.bin` (this now does the
   <extra_id_2> fix). Or use the pipeline's context:
   `/mnt/hdd/nava/cpp-runs/_ref_peter_i2v/bin/context.bin`.
- Dump the audio latent from a render: prefix `NAVA_DUMP_AUDIO_LATENT=/path/lat.bin`.
- Per-step audio trajectory dumps: prefix `NAVA_DUMP_TRAJ=/some/dir` (writes
  aud_noise/vid_noise + va_cfg_NN/va_cond_NN/aud_step_NN per step).
- Render is deterministic per `--seed`.

## HOW TO PUBLISH to the eye/ear test server (owner is headless — this is how they review)
- VIDEO gallery — http://10.0.0.208:8097  (server `tools/nava_eyetest_server.py`,
  RUNS_DIR=`/mnt/hdd/nava/cpp-runs`). It auto-lists every `<NAME>/` that has
  `clip.webm`/`clip.mp4` + `meta.json`. So: render with `--runs-dir /mnt/hdd/nava/cpp-runs
  --out-name <NAME>` and it shows up on refresh. If the server isn't running:
  `RUNS_DIR=/mnt/hdd/nava/cpp-runs PORT=8097 python3 tools/nava_eyetest_server.py &`
- AUDIO A/B — http://10.0.0.208:8099  (a plain `python3 -m http.server 8099 --directory
  /mnt/hdd/nava/audio_demo`). To add a clip: extract wav into that dir and add a row to
  `/mnt/hdd/nava/audio_demo/index.html` (it's a static hand-edited page; the dir index is
  shadowed by index.html, so you MUST add the `<audio src=...>` row yourself):
  ```
  export PATH=/mnt/hdd/3d/avatar-shootout/toolchain/bin:$PATH
  ffmpeg -y -i /mnt/hdd/nava/cpp-runs/<NAME>/clip.webm -vn -c:a pcm_s16le \
     /mnt/hdd/nava/audio_demo/NN_label.wav
  # then edit /mnt/hdd/nava/audio_demo/index.html to add: <audio controls src="NN_label.wav">
  ```
  Direct file URLs work regardless of the index: http://10.0.0.208:8099/NN_label.wav
- If you start a server with `&`, it may get killed when the shell returns; prefer the
  owner-managed servers already running on :8097/:8099, or `nohup ... & disown`.

## METRICS used so far
- formant proxy: ffmpeg → 16k mono f32 → np.fft.rfft → energy[1.5–4 kHz]/total.
  python peter ≈ 0.107; cpp varies by seed 0.010–0.077. (Variance is partly real seed.)
- clipping: peak abs + fraction of |x|>0.99. cpp peaks exactly 1.000 (hard clamp);
  python ~0.99 (its clamp; mp4/aac decode can overshoot to ~1.3 as an artifact).
- decode parity: per-window waveform correlation cpp-vs-python on the SAME latent.

## Key files
- cpp decode: `src/ltx_audio_vae.h` ; render audio: `examples/nava/main.cpp` ~947-985
- python decode: `NAVA/nava_src/vae/local_audio_vae.py` ; oracle:
  `NAVA/nava_audio_vae_decode_ref.py`
- prompts: `/mnt/hdd/nava/peter_prompt.txt`, negatives `vneg_now.bin`/`aneg_now.bin`,
  image `peter_896x448.bin`, data jsonl `peter_noclone.jsonl`
- ggufs in `models/`: nava-dit-{q8_0,f16}.gguf, longcat-umt5-xxl-q8_0.gguf,
  wan2.2-vae-48ch-f16.gguf, nava-ltx-audio-vae-f16.gguf
- python full render harness: `/mnt/hdd/nava/run_nava.sh` (fp8, fits 12GB, ~14 min).
