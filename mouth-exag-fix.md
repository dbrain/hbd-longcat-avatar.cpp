Read the whole path. Good news: the code is unusually clean about where the "mouth gain" lives, and it also **corrects one thing I told you earlier**. Here's the recipe, surgical → crude.

## How voice→mouth actually works here (confirmed from source)

`pipeline_longcat_video_avatar.py:get_audio_embedding_whisper`:
1. Audio → **loudness-normalized to −23 LUFS** (`_loudness_norm`, line 580) → mel → Whisper encoder, output at 50fps.
2. It keeps **all 33 hidden states** and collapses them into **5 feature slots** (line 606-611): `feat0..3` = mean of layer bands `[0:8],[8:16],[16:24],[24:32]`, `feat4` = final layer 32. So early slots = raw acoustic, later = linguistic. Result is `[T, 5, D]`.
3. In the DiT (`modules/avatar/longcat_video_dit_avatar.py`), `AudioProjModel` turns those 5 slots into audio tokens, and each block does an **audio cross-attention then a gated residual add** (lines 161-178):
   ```python
   audio_add_x = (audio_gate_mca * audio_output_noise...)   # learned per-timestep gate
   x = x + audio_add_x                                       # the face gets pushed here
   ```

That last line is the whole ballgame. **The audio's entire influence on the face is one additive term.** Mouth amplitude = how big `audio_add_x` is.

## The recipe, in order of effort vs. payoff

**1. Scale the audio residual — the single best knob.** Change line 178 to:
```python
x = x + audio_mouth_scale * audio_add_x   # try 0.6–0.8
```
This uniformly dials down how far the audio drags the face from its text/image baseline, across every block, without touching timing (timing lives in *which* tokens attend, not their magnitude). Too low → mumbly/desynced. This is your first experiment, and it's one line + plumbing a CLI arg. It's also why fiddling with audio CFG did nothing: even at cfg=1 the conditional audio path *always* adds here — it's not a CFG effect.

**2. Pre-filter the audio (free, no model change).** Worth knowing: the **v1.0 (wav2vec2) path applies `_smooth_transients` — a 3kHz Butterworth low-pass — and `_add_noise_floor`, but the v1.5 Whisper path drops both** (compare lines 580 vs 621-623). That low-pass softens the plosive/sibilant transients that spike visemes. Re-applying it to your wav before feeding v1.5 is a legit, reversible test for softer mouth shapes.

**3. Re-weight the layer bands** (lines 606-611). Down-weighting the early acoustic slots (`feat0/feat1`) shifts conditioning toward linguistic content over raw articulation energy. More experimental — `AudioProjModel` was trained on the equal 5-stack, so big changes degrade — but a mild scaling is cheap to try.

## Correction to my earlier advice ⚠️

**The loudness/volume trick is mostly dead.** Line 580 renormalizes every clip to −23 LUFS *before* encoding, so just lowering your wav's gain gets undone. Only *dynamic-range compression* (changing peak-to-average shape, not overall level) survives the LUFS normalization, and even that's partially washed out. So scratch "knock down RMS" — go with band-limiting (#2) instead if you want an audio-only lever.

**Bottom line for "less exaggerated mouth":** the clean recipe is `audio_mouth_scale ≈ 0.7` on line 178, optionally stacked with a 3kHz low-pass on the input audio. Both are reversible, neither needs retraining, and they attack the exaggeration at the two places it's actually created.

Sources: [github.com/meituan-longcat/LongCat-Video](https://github.com/meituan-longcat/LongCat-Video) (read directly: `longcat_video/pipeline_longcat_video_avatar.py`, `longcat_video/modules/avatar/longcat_video_dit_avatar.py`).
