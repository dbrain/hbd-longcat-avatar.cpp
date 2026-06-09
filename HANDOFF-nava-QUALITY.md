# NAVA cpp — QUALITY INVESTIGATION HANDOFF (paste to a fresh agent)

Branch `nava-port` in `/home/dbrain/dev/longcat-avatar.cpp`. The core port (DiT forward, joint
audio-video denoise, video VAE, audio VAE) is **done and numerically validated**. This doc is about
the **open quality issues** found while doing an exact-params side-by-side vs the PyTorch reference,
plus everything shipped this session. Owner wants: **feature-complete BEFORE perf/memory** (don't tune
perf against a codebase that's still changing). Order: **(1) quality → (2) ReDimNet → (3) perf/Q4_K.**

## Owner rules (IMPORTANT — these bit us)
- **Drive GPU/CPU from the MAIN loop**, never from Agent-tool subagents (they stall on
  run_in_background completion). Subagents only for authoring code. (Whole session was main-loop.)
- **CPU umT5 thrashes the box** (swap/IO) — the owner's interactive session locks up. NEVER load
  umT5/the fp8 model on CPU casually. The cpp `encode-prompt` (GPU, q8) replaces it (see below).
- cpp builds on this box are FINE (Rust-only no-build rule). Toolchain:
  `export PATH=/mnt/hdd/3d/avatar-shootout/toolchain/bin:$PATH;
   export LD_LIBRARY_PATH=/mnt/hdd/3d/avatar-shootout/toolchain/lib;
   cmake --build build-nava --target nava -j8` → `build-nava/bin/nava`.
- GPU is SERIAL — one job at a time (12 GB 3060). q8 DiT OOMs above ~832×480/13f.
- **Gather diagnostic data during every run** (owner ask) to minimise reruns — dump latents, log
  per-step std, analyse audio envelope. cpp renders already print per-step `std`/`aud_std`.

## ✅ Shipped this session (all in working tree, UNCOMMITTED — owner is holding the commit)
1. **Phase-3 audio END-TO-END**: co-denoised audio latent → in-tree LTX audio VAE (`src/ltx_audio_vae.h`,
   gguf `models/nava-ltx-audio-vae-f16.gguf`) → 48 kHz Opus muxed into the webm. `examples/nava/main.cpp`
   `--audio-vae` (default on, `--no-audio` to silence). Decode validated vs PyTorch on a real latent:
   waveform corr 0.965 / spectrogram 0.943 / envelope 0.992. (The handoff's "T=8-only decoder bug" was
   already fixed — full T-sweep 4..100 decodes.) CMake: `SD_USE_OPUS` for the nava target (needs
   `-I${Opus_INCLUDEDIR}` — conda cross-sysroot toolchain ignores host /usr/include & CMake strips it).
2. **`nava encode-prompt <umt5_gguf> <prompt.txt> <out.bin>`** — text→umT5(GPU,q8)→context bin [4096,512],
   zero-padded, **input truncated to 512 tokens** (NAVA `text_len=512` cap, `model_loading_utils.py`).
   Kills the python umT5 dependency (no more thrash). Validated vs python on the smoke prompt: identical
   344 tokens, cosine **0.98** (gap = q8 weights vs python bf16). umT5 gguf prefix = `text_encoders.t5xxl.transformer`.
3. **UniPC is now the DEFAULT sampler** (was Euler). Matches python prod (`nava_run.yaml scheduler_unipc=true`).
   Opt out with `NAVA_EULER=1`. THIS FIXED THE VIDEO FUZZ — but see the audio regression below.

## 🔴 OPEN QUALITY ISSUES (the reason for this handoff)

### ⚡ UPDATE — audio-divergence narrowed by elimination (session 2). READ THIS FIRST.
The peter audio diverges to **flat noise** (env_CV 0.18–0.22, vs action 0.99 = clean). Systematically ruled out:
- **NOT resolution/aspect.** Aspect WAS wrong (peter.png is 2.035:1; cpp 832×480=1.73:1 stretched the face AND
  the I2V anchor — fix: render the 2:1 bucket **896×448**, matching python's 640 bucket; `peter_896x448.bin`,
  run `cpp_peter_896`). But audio diverges at BOTH 832 and 896 → resolution is not the audio cause.
- **NOT the sampler.** Per-stream split (UniPC video + **Euler audio**, new `NAVA_AUDIO_EULER=1`) tamed the
  final-step `aud_std` spike (1.25→**1.056**, ≈ action's clean ~1.0) **but env_CV stayed 0.20 (still noise).**
  ⇒ the audio latent converges to the WRONG CONTENT, not a magnitude runaway. UniPC-on-audio is NOT the root cause.
- **NOT a corrupt context bin.** `dense_peter_talk512.bin`: 512 nonzero tok, std 0.098, range [-2.15,1.98],
  no NaN/Inf — in line with the validated smoke bin. The cpp pipeline is fully self-contained (cpp umT5, no python).
- **NOT truncated speech.** peter's `<S>…<E>` is at ~token 238 (cap 512) — well within; it survives.
- **action correction**: `action.jsonl` DOES carry `spk_wavs:[peter.wav]` (ran WITH a speaker embed), and was a
  made-up non-official prompt. So we do NOT have a clean python no-spk reference yet (that control got killed).

- **NOT the missing speaker embedding.** (Owner: spk_emb is an OPTIONAL cloning feature, not required.) Proof:
  cpp **action** produces clean speech (env_CV 0.99) with the SAME stubbed-zero spk. So cpp can voice fine without
  it — ReDimNet is NOT the fix here, it's just the optional clone feature (task #5, unrelated to this bug).

⇒ **It's a cpp AUDIO-PATH PORT BUG** that bites the harder peter conditioning while action sails through. Prime
suspects: the audio-stream **forward** (patch_embedding_audio / NavaHead head_audio / the joint self-attention's
audio half), the **audio CFG** (`cfg_audio=2.0`, the separate audio-neg path), or the audio **latent→VAE** convention.
**Next debug = validate the cpp AUDIO stream against PyTorch on peter**: the VIDEO forward is 96–122 dB faithful, but
the audio stream was only loosely checked — dump the cpp per-step audio velocity and diff vs a PyTorch joint denoise
on the same peter context, step 0 first (isolates forward from sampler). Also render the **wolverine** official
sample: if its audio diverges too → systematic audio-path bug; if only peter → narrow to peter's conditioning.


### Symptom (owner eye/ear test on peter_talk, the official NAVA sample)
- Face/eyes **blur with motion**; the "camera" **jumps**.
- **Voice drops out** / goes to noise, especially with UniPC.
- action (a made-up, simple prompt) looks & sounds much cleaner — so it's **prompt-dependent fragility**.

### Hard data gathered (all reproducible, see commands)
Per-step **video** latent std: peter and action are nearly identical (0.95→~0.66, then the big final
flow-match step to ~0.85). So global video denoising is NOT the differentiator.

Per-step **audio** latent std diverges for peter: ends at **1.389** (runaway) vs action **0.997**.

Audio envelope coefficient-of-variation (CV>~0.5 = speech with syllable structure; <0.5 = flat noise):
| render (peter_talk, official sample, 832×480-ish, 10 steps) | env_CV | verdict |
|---|---|---|
| cpp **Euler**, no spk | 0.34 | partial voice, degrades at end |
| cpp **UniPC**, no spk | **0.18** | voice → flat noise (UniPC made audio WORSE) |
| python **UniPC + clone** (ReDimNet) @896×448 | 0.66 | speech intact |
| cpp action (UniPC, no spk, easy prompt) | 0.99 | clean speech |

### Two independent causes, both supported by the data
1. **UniPC on the AUDIO stream destabilises the voice** (Euler 0.34 → UniPC 0.18). The owner remembered
   pre-UniPC peter "had voice, went weird at the end" — matches. Video wants UniPC (fixes fuzz); audio
   does not. **The cpp already has SEPARATE per-stream schedulers `usched_v` / `usched_a`
   (examples/nava/main.cpp ~line 661+).** → **TRY: UniPC for video, Euler (or fewer-order) for audio.**
   This is the single highest-leverage next experiment and is ~a few-line change + one render.
2. **Missing speaker embedding (ReDimNet stubbed → we feed zeros)** degrades audio stability (python
   WITH clone = 0.66, cpp WITHOUT = 0.18–0.34). NAVA is a JOINT AV MMDiT; the handoff warns a degenerate
   audio stream "poisons the joint attention and the video never coheres" — so the diverged audio is a
   prime suspect for the **video face-blur/camera-jump** too (not just bad sound). `SpkToken` is stubbed
   at `src/nava.hpp:584` (weights ARE in the gguf, unused).
3. (Lesser) General softness on peter = **resolution + prompt aesthetic**: peter's prompt explicitly asks
   for grey / low-saturation / cold-light / blurry-background — reads as grain at 832×480. python ran the
   960 bucket (1344×672). Closes when Q4_K unlocks native-res renders. NOT a port bug.

### EXACT-PARAMS CONTROL STILL OWED (owner's real ask, not yet done right)
The proper apples-to-apples is **python with NO clone (built-in voice)** at the SAME params as cpp, so the
only variable is the port — NOT the spk_emb. My python control accidentally used the clone (peter_talk.jsonl
has `spk_wavs`). **TODO: rerun python on peter with spk_wavs removed** (built-in speech), 640 bucket, 10
steps, UniPC, `--is_i2v`, and compare its audio env_CV + face stability to cpp UniPC. That isolates
"is the cpp audio divergence a port bug or inherent to no-spk?".

## Decisive next experiments (in order)
1. ~~Per-stream sampler split~~ **DONE (session 2): ruled out** — `NAVA_AUDIO_EULER=1` tamed `aud_std` but
   env_CV stayed 0.20 (noise). Audio lands on wrong content, not a magnitude runaway. (Env still useful for A/Bs.)
2. **Validate the AUDIO stream forward vs PyTorch on peter** — dump the cpp per-step audio velocity
   (`NAVA_DUMP_TRAJ` dumps video vel; add the audio vel alongside) and diff against a PyTorch joint-denoise on the
   SAME peter context. The video forward is 96–122 dB faithful; the audio stream was only loosely checked. This
   tells you port-bug vs inherent. Also: render the **wolverine** official sample (another hard voice) — does its
   audio diverge too? If yes, it's systematic; if only peter, it's prompt-specific.
3. **python no-clone control** (the true exact-params, no-spk baseline) — TARGETED single run only (owner: python
   runs must be rare). `peter_noclone.jsonl` already written (spk_wavs removed). spk is OPTIONAL, so python SHOULD
   voice fine without it → if python stays clean here and cpp diverges, that confirms the cpp audio-path bug (and
   gives a golden audio trajectory to diff against). Render at 640 bucket (896×448), 10 steps, `--is_i2v`.
4. **ReDimNet port** (task #5) — the OPTIONAL voice-cloning feature, NOT the divergence fix (spk_emb is optional;
   action voices fine without it). Do it for the feature after the audio-path bug is fixed. Port ReDimNet-M
   (`torch.hub IDRnD/ReDimNet model_name='M' train_type='ft_mix' dataset='vb2+vox2+cnc'`, → L2-normed
   192-dim emb) → gguf + cpp encoder, validate emb vs torch.hub on peter.wav (cosine ~1.0), un-stub
   `SpkToken` in nava.hpp (project 192→token, splice into context; weights already in gguf), wire
   `--spk-wav`. Then re-test peter audio+video stability.
4. **Q4_K + perf/mem** (task #6) LAST — unlocks native-res renders (closes the resolution softness) and
   is the owner's target quant. q8 + q4_0 exist; q4_k = 2-step (nava-dit-f16.gguf → `sd-cli --mode convert
   --tensor-type-rules "...=q4_k"`; sd-cli needs building). Validate forward fidelity vs PyTorch.

## Reproduction / tooling
- **Render** (cpp, UniPC default now): `build-nava/bin/nava render --cuda --gguf models/nava-dit-q8_0.gguf
  --vae models/wan2.2-vae-48ch-f16.gguf --context <dense.bin> --neg-context <video_neg.bin>
  --audio-neg-context <audio_neg.bin> --image <peter_832x480.bin> --steps 10 --frames 13 --width 832
  --height 480 --seed 42 --cfg 3.0 --out-name X --runs-dir /mnt/hdd/nava/cpp-runs`
  (add `NAVA_EULER=1` for all-Euler; `NAVA_AUDIO_EULER=1` for UniPC-video+Euler-audio;
  `NAVA_DUMP_AUDIO_LATENT=<path>` dumps the co-denoised audio latent.) **Use the aspect-matched image**:
  peter.png is 2:1 → render **896×448** with `peter_896x448.bin` (NOT 832×480, which stretches the face+anchor).
- **Encode a prompt** (cpp, GPU): `build-nava/bin/nava encode-prompt models/longcat-umt5-xxl-q8_0.gguf
  /tmp/PROMPT.txt /mnt/hdd/nava/cpp-runs/_ctx/dense_X.bin`. Prompt texts extracted from the jsonl with a
  tiny `json.loads(...)["prompt"]` (instant, no model). Fixed negatives already encoded:
  `_ctx/cpp_video_neg_ctx.bin`, `_ctx/cpp_audio_neg_ctx.bin`. peter dense: `_ctx/dense_peter_talk512.bin`.
- **Audio-vae decode test**: `build-nava/bin/nava ltx-audio-test <gguf> <latent.bin> [out.bin]`.
- **PyTorch python run** (THRASHES — owner-gated): `NAME=.. DATA_FILE=/mnt/hdd/nava/peter_talk.jsonl
  CFG=/mnt/hdd/nava/nava_640.yaml STEPS=10 FRAMES=13 GENFLAGS="--is_i2v" bash /mnt/hdd/nava/run_nava.sh`.
  image_size bucket = res (W/H args IGNORED for I2V): 640→~896×448, 960→1344×672. ~16 min wall incl load,
  peak ~11.8 GB VRAM / ~25 GB RAM. Audio-vae ref decode: `~/dev/NAVA/nava_audio_vae_decode_ref.py`.
- **Audio env_CV analysis** (the speech/noise discriminator): extract `ffmpeg -i clip.webm -ar 16000 -ac 1
  out.wav`, then env CV = std/mean of per-800-sample RMS; CV<0.5 ≈ noise, >0.5 ≈ speech. Script inline in
  the session log; also speechband% (200–3400 Hz energy fraction).
- **Eye-test**: `tools/nava_eyetest_server.py` serves `/mnt/hdd/nava/cpp-runs` on :8097. Side-by-side page
  at `/mnt/hdd/nava/sxs/index.html` (served :8098) — python vs cpp, both cases.

## Key facts / gotchas
- peter_talk **IS** the official NAVA sample (`infer_cases/timbre/prompts.jsonl` line 1) — byte-exact prompt
  match confirmed. `action` was a made-up prompt (NOT official) → its "vortex/squinty" weirdness is
  prompt-vs-image conflict (warm-room prompt on a command-center anchor image), a red herring for quality.
- The other official sample = "wolverine" (car interior), `infer_cases/timbre/{wolverine.png,wolverine.wav}`.
- peter prompt = 574 tokens → truncated to 512 (matches python). Fidelity fix already in encode-prompt.
- No NAVA per-sample golden output ships (only `assets/teaser_0520.mp4`, a heavily-cut 49 s montage — not
  usable as a clean reference). Treat python-full (960 bucket) as golden-equivalent.
- I2V images prepped at `/mnt/hdd/nava/peter_{832x480,896x448,1344x672}.bin` via `tools/nava_prep_image.py`.
- Models present: `nava-dit-{f16,q8_0,q4_0}.gguf`, `nava-ltx-audio-vae-f16.gguf`, `wan2.2-vae-48ch-f16.gguf`,
  `longcat-umt5-xxl-q8_0.gguf`. (No q4_k yet.)

## Uncommitted working-tree changes (owner holding the commit)
`examples/nava/main.cpp` (audio decode+mux, encode-prompt, 512-cap, UniPC default, per-stream sampler
`NAVA_AUDIO_EULER`, audio-latent dump),
`examples/nava/CMakeLists.txt` (SD_USE_OPUS + opus include). Plus prior untracked port (committed earlier
as `c8dac7a`/pushed). When ready: `git add examples/nava && git commit` (no Claude/AI trailers — owner rule).
