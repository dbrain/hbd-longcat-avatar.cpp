# NAVA cpp — peter garble: RESOLVED (mostly) + residual handoff

Branch `nava-port` in `/home/dbrain/dev/longcat-avatar.cpp`. Working tree UNCOMMITTED
(owner holding commit). The long "peter_talk garbled audio + warped face" hunt is
**essentially fixed**; one small residual remains (end-of-clip degradation). This doc:
what the bug actually was, what's fixed, and the precise residual for fresh eyes.

## THE BUG (root cause, after a long detour)
The cpp **tokenizer** mis-encoded the peter prompt's text → wrong umT5 context → garbled
audio AND video (joint model). Two cpp tokenizer bugs (both fixed in
`examples/nava/main.cpp` `encode-prompt` + `nava_normalize_text`):
1. **NFKC full-width punctuation not normalized.** python's HF tokenizer NFKC-maps
   full-width forms to ASCII (`，`U+FF0C→`,`, `：`U+FF1A→`:`); cpp didn't → 34 wrong token
   ids on peter. Fix: `nava_normalize_text()` maps U+FF01–FF5E → cp−0xFEE0, U+3000→space,
   + whitespace_clean. After this cpp token ids match python EXACTLY (0 diffs).
2. **EOS truncation.** A 574-tok prompt capped to 512: python keeps first 511 + EOS at
   511; cpp kept first 512 (dropped EOS). umT5 is bidirectional so the missing terminator
   shifts the whole encoding. Fix: `tokens.resize(511); push_back(eos); masks.assign(512,0.0f)`.

Result: peter went from total garble → **intelligible speech** (owner ear-confirmed clip 7).

## DEAD ENDS (do NOT re-investigate — proven)
- **umT5 is NOT the bug.** cpp umT5 == NAVA's own `nava_src/.../t5.py T5EncoderModel` at
  **cos 1.0 at every position** (and == HF `UMT5EncoderModel`), on identical tokens.
- **umT5 weights longcat == nava** (both standard umT5-xxl; cpp-nava-q8 vs cpp-longcat-f16
  = 0.9995). NAVA uses `models/longcat-umt5-xxl-q8_0.gguf` directly. (The nava-umt5 + f16
  ggufs I built were redundant and were deleted.)
- **q8 vs f16 umT5** identical (0.6318 both). **Relative-position-bias bucketing** formula
  matches python exactly. **f16 DiT** doesn't fit the 12GB GPU and isn't the issue.
- The "context cos 0.63 / position-decay" I chased for a long time was measured vs a
  **CONTAMINATED reference**: the dump `_ref_peter_i2v/ref_tensors.npz` `input_vid_context_raw`.
  That dump disagrees with a FRESH `t5.py` encode of the same prompt by the same 0.63 —
  i.e. the dump is the outlier, not cpp. (See residual.)

## ALSO FIXED ALONG THE WAY (real divergences, keep)
- **align_3d_cfg (3-way CFG + masking_modality pass)** in nava.hpp + run_render — cpp was
  doing 2-way CFG; python (nava_640.yaml align_3d_cfg=true) does 3-way with a separate-attn
  forward. (nava.hpp DoubleBlock/SingleBlock `joint_attn` flag, NavaRunner.mask_modality,
  compute_va(mask_modality); main.cpp `--cfg-align`/`--cfg-align-audio`, env NAVA_NO_ALIGN_CFG.)
- **audio-neg context**: no-spk → python merges to the VIDEO negative for both streams;
  cpp ran a separate `--audio-neg-context`. Now reuses video-neg (env NAVA_SEPARATE_AUDIO_NEG
  to restore). main.cpp run_render.
- **audio_len round→ceil** (t2v.py:417 parity).
- decode, DiT forward (incl i2v per-token), sigma schedule, UniPC — all proven faithful.

## THE RESIDUAL (for fresh eyes)
Clip 7 (cpp's own FIXED context) is intelligible but the LAST ~2 words degrade + clip:
"...before they know what hit-" → "...betruer they know what h-". Clips 5/6 (which used the
DUMP's context) were FULLY clean. So the dump's context renders a cleaner END than a fresh
encode — and cpp uses a fresh-encode-equivalent context.

So the mystery: **the NAVA pipeline produces a context whose LATE tokens differ from a fresh
`tm([prompt])` encode (~0.63 cosine there), and the pipeline's version is the correct/clean
one.** cpp matches the fresh encode (1.0), so cpp is "correct" by that measure but renders a
slightly-off END.

Decisive next steps:
1. **Get the pipeline's ACTUAL tokens for peter.** Re-run the dump capturing the umT5 INPUT
   token ids (the monkeypatch only saved post-umT5 `input_vid_context_raw`). Compare to cpp's
   512 ids. If they differ at late positions → the pipeline tokenizes/truncates peter
   differently than `encode-prompt` (likely the real late-token source). Check t2v.py data
   loader prompt handling + t5.py `__call__` `ctx_list=[u[:seq_lens]]` + how the pipeline
   pads the context back to 512 for the backbone.
2. **Audio length**: cpp clip = 2.05s, python A = 2.18s (same 52 audio tokens) → "hit" clips.
   Check the audio-VAE decode token→sample ratio vs python (`nava_audio_vae_decode_ref.py`)
   and the audio_len computation.

## Tooling / repro (all small-res, no bf16, no full-res)
Build: `export PATH=/mnt/hdd/3d/avatar-shootout/toolchain/bin:$PATH;
export LD_LIBRARY_PATH=/mnt/hdd/3d/avatar-shootout/toolchain/lib;
cmake --build build-nava --target nava -j8`
- encode-prompt: `./build-nava/bin/nava encode-prompt models/longcat-umt5-xxl-q8_0.gguf
  /mnt/hdd/nava/peter_prompt.txt <out.bin>`  (NAVA_DUMP_TOKENS=<file> dumps token ids)
- context cosine check + token-id diff scripts: inline python in the session; reference =
  fresh `t5.py` encode (NOT the dump). NAVA umT5: `/mnt/hdd/nava/Wan2.2-TI2V-5B/
  models_t5_umt5-xxl-enc-bf16.pth`; tokenizer `google/umt5-xxl`.
- render peter: see `cpp_peter_FIXEDctx` invocation (q8 DiT, 896x448/13f, peter_896x448.bin,
  vneg_FIXED.bin/aneg_FIXED.bin). Audio eye/ear: http://10.0.0.208:8099 (clips 1-7),
  env_CV is a BAD judge here — use the ear.
- python full ref render (fp8, ~14min, fits): `run_nava.sh`. Don't load umT5 bf16 on GPU
  (OOM); CPU encode is fine for one prompt.
