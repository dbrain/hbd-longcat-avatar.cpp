# NAVA audio parity handoff

Current conclusion: the audible C++ dirt/hotter/darker issue is no longer primarily
audio-VAE decode. It is the audio latent that C++ produces before decode.

## 2026-06-05 ROOT CAUSE FOUND: I2V image preprocessing (stretch vs center-crop)

The "frame-0 anchor mismatch is real but forcing Python's anchor makes it worse"
paradox is resolved. The frame-0 video anchor differs (corr 0.93) because the C++
`--image` bin was built with a DISTORTING stretch-resize, not Python's
aspect-preserving resize + center crop. The earlier pyanchor LATENT-injection
experiments were the wrong fix vector (latent-space injection); the actual bug is
upstream in IMAGE preprocessing, so the clean fix is to feed C++ a correctly
preprocessed image and let its own VAE re-encode.

Concretely:
- `tools/nava_prep_image.py` (old) did `Image.resize((W,H), LANCZOS)` — a direct
  stretch to 896x448, which distorts when the source aspect != 2:1.
- peter.png is 1447x711 (aspect 2.035), target 896x448 (aspect 2.0). Stretch
  squishes ~1.7% horizontally and never crops.
- Python `nava_src/vae/local_video_vae.py:_resize_center_crop` does
  aspect-PRESERVING LANCZOS resize (scale=max -> 912x448) then center-crops 8px
  each side to 896x448.
- Both Python (`vae2_2.py:805 mu=(mu-mean)*1/std`, scale=[mean,1/std]) and C++
  (`wan.hpp vae_to_diffusion_latents`) apply the SAME per-channel WAN2.2 latent
  normalization (std constants verified identical). Normalization is NOT the bug;
  the earlier "C++ normalizes, Python doesn't" note was wrong.

`tools/nava_prep_image.py` now defaults to `_resize_center_crop` parity; the
legacy distorting path is behind `--stretch`. Regenerate the image with:

```bash
/mnt/hdd/nava/.venv/bin/python tools/nava_prep_image.py \
  /home/dbrain/dev/NAVA/infer_cases/timbre/peter.png 896 448 \
  /mnt/hdd/nava/peter_896x448_crop.bin
```

Step-0 cond audio velocity (NAVA_STOP_AFTER_STEP0_COND=1), vs Python va_cond_00:

```text
stretched image (old):    corr=0.999241 reldiff=0.041835
center-crop image (fix):  corr=0.999714 reldiff=0.024091   <- ~42% less step-0 error
oracle floor (py inputs): reldiff=0.011364
```

Residual above the oracle floor (0.024 vs 0.011) is the C++ WAN-VAE encode vs
Python encode (LANCZOS kernel / VAE numerics).

### Full-render result: latent metric DOWN, perceptual metric UP (metric was wrong)

Full crop render (`cpp_peter_fp8fold_crop_seed42`) vs Python:

```text
final audio latent: corr 0.976365 -> 0.959161 (WORSE), reldiff 0.219 -> 0.283
```

But the WAV metrics (what the owner hears: "hotter/louder/darker") IMPROVED, vs the
matched-noise Python target `16_python_from_cpp42noise.wav`:

```text
                     rms(loudness)  centroid(bright)
stretch (before):       1.111           0.937
center-crop (fix):      1.035           0.948     <- both toward 1.0 (Python)
```

So latent-L2-distance-to-Python is the WRONG objective across the q8-quantized model
(codex was optimizing it). The image-preprocessing fix demonstrably reduces the
audible loudness gap (+11% -> +3.5%). Owner ear check: "it's clean now." The eval
clip is row 30; a minimal target-vs-now-vs-before page is
`/mnt/hdd/nava/audio_demo/compare_gt_vs_now.html` (served on :8099). The target there
is `16_python_from_cpp42noise.wav` (SAME seed-42 init noise = true sample-for-sample
A/B), NOT `GT_python_full_render.wav` (different noise = quality bar only).

### In-port image handling (no more prep script)

`examples/nava/main.cpp` now decodes a real image (`--image foo.png/.jpg/...`) and does
the Python-parity preprocessing itself; the pre-baked RGB `.bin` is still accepted
(continuation / back-compat). Auto-detected by extension (`looks_like_image_file`).

- `load_image_resize_center_crop()` — decode (media_io stb) -> aspect-preserving resize
  (scale=max -> fills target) -> center crop -> /255 -> ggml ne [W,H,1,3].
- `nava_lanczos_resize_rgb8()` — PIL-compatible separable **Lanczos-3** (precompute_coeffs,
  support=3*max(scale,1), uint8 between passes). stb has NO Lanczos; its closest cubic
  (CATMULLROM) measurably diverged through the chaotic DiT and pushed audio DARKER
  (centroid 0.897 < even the stretch bug's 0.937). Lanczos was required, not optional.

Step-0 `va_cond` parity (NAVA_STOP_AFTER_STEP0_COND=1):

```text
                          vs PIL-prep crop   vs Python
stretch (old bin):              -             0.0418
native CATMULLROM:            0.0234          0.0312
native LANCZOS (shipped):     0.0064          0.0240   <- == prep-LANCZOS (clip 30)
PIL-prep crop (clip 30):        -             0.0241
```

Native LANCZOS reproduces the verified-good clip-30 input (reldiff 0.006 = float-vs-uint8
noise). `tools/nava_prep_image.py` also fixed (center-crop default, `--stretch` legacy)
but is now only a convenience — the port no longer needs it. Verified-good repro command:

```bash
./build-nava/bin/nava render --cuda --gguf models/nava-dit-fp8fold-q8_0.gguf \
  --image /home/dbrain/dev/NAVA/infer_cases/timbre/peter.png \
  --context .../context.bin --neg-context .../vneg_now.bin --audio-neg-context .../aneg_now.bin \
  --vae models/wan2.2-vae-48ch-f16.gguf --audio-vae models/nava-ltx-audio-vae-f16.gguf \
  --steps 10 --frames 13 --width 896 --height 448 --fps 24 \
  --cfg 3.0 --cfg-align 3.0 --cfg-align-audio 2.0 --shift 5.0 --seed 42 ...
```

### VAE-encode TILING was the last input-side gap — FIXED (input parity now bit-exact)

The remaining step-0 residual (0.024 vs oracle 0.011) was NOT precision — it was VAE
encode **tiling**. C++ encoded the i2v frame in overlapping 24x24 latent tiles
(`et.enabled = W_lat>24 || H_lat>24`); Python encodes the full frame in one pass. The
seam blending perturbed the frame-0 anchor. Same-pixels A/B (peter PIL-LANCZOS crop,
anchor vs Python `py_step0_frame0_anchor.bin`):

```text
                    anchor vs py            step-0 va_cond vs py
tiled (old):    corr 0.9906 reldiff 0.137     reldiff 0.0241
full-frame:     corr 0.99999 reldiff 0.0054   reldiff 0.0118   <- == oracle floor 0.0114
```

Per-column error confirmed seams: tiled error spikes at tile boundaries (w~28-30 and
right edge w=55, up to 0.24); full-frame is flat ~0.002. So F16-vs-bf16 VAE precision is
negligible — it was ALL tiling. **Input side is now bit-parity with Python.**

Fix: `examples/nava/main.cpp` encodes the i2v frame full-frame by default
(`NAVA_VAE_ENCODE_TILE=1` re-enables tiling for huge inputs). VRAM cost ~zero: the encode
runs in isolation BEFORE the DiT loads (VAE freed right after), peak ~2.6GB (1.34 params
+ 1.26 compute) vs the DiT sampling peak ~8.9GB which is unchanged. Decode tiling (the
expensive many-frame path) is untouched. `NAVA_DUMP_ANCHOR=<path>` dumps the frame-0
anchor latent [W_lat,H_lat,1,48] for parity checks.

INPUT-SIDE PARITY COMPLETE: preprocessing (Lanczos center-crop) + full-frame encode put
the C++ frame-0 anchor and step-0 audio velocity at the Python oracle floor. Any residual
audible difference is now downstream (q8 DiT weights / accumulated sampling), not input.

## GPU-slot results on 2026-06-04

Two full C++ GPU renders were run after the decode/mux fixes:

1. fp8-folded q8 GGUF with integer model timesteps:

```text
/mnt/hdd/nava/cpp_seed42_fp8fold_final_audio_latent.bin
/mnt/hdd/nava/audio_demo/22_cpp_fp8fold_render_seed42.wav
latent: corr=0.967669 reldiff=0.259565 std_ratio=1.041833
wav: rms_ratio=1.133106 centroid_ratio=0.933979
```

2. original q8 GGUF with integer model timesteps:

```text
/mnt/hdd/nava/cpp_seed42_q8_intt_final_audio_latent.bin
/mnt/hdd/nava/audio_demo/23_cpp_q8_intt_render_seed42.wav
latent: corr=0.962094 reldiff=0.283341 std_ratio=1.051434
wav: rms_ratio=1.135412 centroid_ratio=0.945919
```

Baseline current q8 float-timestep final latent remains better:

```text
/mnt/hdd/nava/cpp_seed42_current_final_audio_latent.bin
latent: corr=0.968739 reldiff=0.254497 std_ratio=1.038822
```

Conclusion: fp8 scale folding did not improve the final audio latent, and
strict integer UniPC model timesteps made q8 worse. Do not spend another GPU run
on those two hypotheses unless another change depends on them.

## Timestep status

Python's `FlowUniPCMultistepScheduler.set_timesteps()` keeps float sigmas
internally but exposes `self.timesteps` as `torch.int64`, so the model sees:

```text
999, 978, 952, 921, 882, 833, 769, 681, 555, 357
```

However, measured renders show C++ is closer with its original float
`sigma*1000` timesteps. Therefore `examples/nava/main.cpp` now defaults to
float model timesteps and keeps the strict Python-source path behind:

```bash
NAVA_UNIPC_INT_TIMESTEP=1
```

The per-step log prints both `t` and `model_t` so render logs make this visible.

Also fixed the optional `NAVA_AUDIO_EULER=1` diagnostic path: when video/model
timesteps are UniPC but audio is stepped with Euler, the Euler delta now uses
the UniPC audio sigma ladder instead of the auxiliary personalized Euler
schedule. Default audio UniPC output is unchanged, but audio-Euler A/Bs are no
longer biased by a mismatched timestep/dsigma pair.

Added a lightweight audio-only trajectory dump:

```bash
NAVA_DUMP_AUDIO_TRAJ=/path/to/dir ./build-nava/bin/nava render ...
```

This writes only small audio tensors:

```text
aud_noise.bin
va_cond_XX.bin
va_uncond_XX.bin
va_mmask_XX.bin
va_cfg_XX.bin
aud_step_XX.bin
```

It avoids the large video dumps from `NAVA_DUMP_TRAJ` and is enabled by default
in `tools/nava_run_fp8fold_latdump.sh` at:

```text
/mnt/hdd/nava/cpp_seed42_fp8fold_audio_traj
```

CPU verification after these changes:

```bash
/mnt/hdd/nava/.venv/bin/python examples/nava/unipc_ref.py /tmp/nava_unipc_ref_check
./build-nava/bin/nava unipc-test /tmp/nava_unipc_ref_check
```

```text
OVERALL MAX ABS ERROR = 2.145767e-06
GATE (<1e-4): PASS
```

```bash
DRY_RUN=1 bash tools/nava_run_fp8fold_latdump.sh
cmake --build build-nava --target nava -j8
```

Both pass with the pinned toolchain environment.

## Evidence

- Python injected run final latent:
  `/mnt/hdd/nava/py_inject42_final_audio_latent.bin`
- C++ current q8 final latent:
  `/mnt/hdd/nava/cpp_seed42_current_final_audio_latent.bin`
- C++ audio VAE decode of the Python final latent:
  `/mnt/hdd/nava/audio_demo/21_cpp_decode_python_final_latent_seed42.wav`

Owner ear check: clip 21 matches clip 16 (`16_python_from_cpp42noise.wav`) closely.
Metric check:

```bash
python3 tools/nava_compare_audio_wavs.py \
  /mnt/hdd/nava/audio_demo/16_python_from_cpp42noise.wav \
  /mnt/hdd/nava/audio_demo/21_cpp_decode_python_final_latent_seed42.wav
```

```text
py16 vs 21: corr=0.976539, RMS 0.281472/0.292902
```

Latent gap with current C++ q8 model:

```bash
python3 tools/nava_compare_audio_latents.py \
  /mnt/hdd/nava/py_inject42_final_audio_latent.bin \
  /mnt/hdd/nava/cpp_seed42_current_final_audio_latent.bin
```

Current result:

```text
overall corr=0.968739 rmsdiff=0.322337 reldiff=0.254497 std_ratio=1.038822
worst steps around 40-45
```

## Band-aid status

The render-time EQ/postprocess experiment was removed from `examples/nava/main.cpp`.
There should be no `NAVA_AUDIO_POST` hook in the tree.

## Decode-side fixes kept

- C++ audio VAE now follows Python's effective `VocoderWithBWE -> torchaudio resample
  to 16 kHz -> clamp ±0.99` behavior.
- WebM/Opus muxing preserves stereo instead of downmixing to mono.

## New model parity candidate

Python render uses `/mnt/hdd/nava/NAVA_fp8.safetensors`, which stores:

```text
1052 BF16 tensors
380 F8_E4M3 matrices
380 matching *.weight_scale tensors
```

`tools/convert_nava_dit.py` now supports `F8_E4M3` and folds
`weight * weight_scale[:, None]` before writing GGUF. This was CPU-validated
against PyTorch for `backbone.double_blocks.0.cross_attn.k.weight` with exact
match (`maxabs=0.0`).

Built artifact:

```text
models/nava-dit-fp8fold-q8_0.gguf
```

CPU inventory check:

```bash
uv run --with gguf python3 - <<'PY'
import gguf
r = gguf.GGUFReader('models/nava-dit-fp8fold-q8_0.gguf')
print(len(r.tensors))
print(sum(t.name.endswith('.weight_scale') for t in r.tensors))
PY
```

Current result:

```text
tensor_count=1052
weight_scale_tensors=0
```

## Current audit / next target

The obvious input-side mismatches have been checked and are not the current
smoking gun for the no-clone peter seed-42 run:

- no speaker embedding is present in `/mnt/hdd/nava/peter_noclone.jsonl`, so
  `timbre_cfg: true` in YAML is inert on both Python and C++.
- Python's no-speaker joint model chooses `context_vid` for both streams. C++'s
  default audio-uncond reuse of the video negative context is therefore correct
  for this sample; `NAVA_SEPARATE_AUDIO_NEG=1` is not the parity path here.
- C++ hardcoded audio CFG values match `/mnt/hdd/nava/nava_640.yaml`
  (`audio_guidance_scale=2.0`, `audio_align_guidance_scale=2.0`).
- audio length is 52 tokens, matching the Python data path for the 13-frame,
  24 fps, 25 audio-token/sec setup.
- audio RoPE shape/frequency matches Python: 22 complex pairs with the 0.24
  temporal scale, remaining head pairs identity.
- for image-only I2V, C++ does not zero audio timesteps; only video frame-0
  tokens are clean. That matches Python's separate video/audio prepare calls.

The final latent error is structured, not a flat gain:

```text
current q8 final latent vs Python injected final latent:
overall corr=0.968739 reldiff=0.254497 std_ratio=1.038822
worst audio-token steps: 40-45
worst mel-index groups: c % 16 around 5-11
```

That pattern fits a trajectory/model-input branch drift more than decode or
postprocessing. Branch-level Python-vs-C++ audio trajectory comparison has now
been run; see the 2026-06-05 section below.

## 2026-06-05 trajectory/oracle results

The remaining issue has now been localized away from decode, CFG algebra,
scheduler update, and gross backbone implementation bugs.

Active diagnostic started after this handoff update:

```text
Goal: localize the first block/layer divergence inside step-0 conditional DiT.
C++ change: examples/nava/main.cpp supports NAVA_STOP_AFTER_STEP0_COND=1, which
stops immediately after the first conditional compute_va() so LONGCAT_DUMP_DIR
contains only cond-branch block taps.
Planned outputs:
  /mnt/hdd/nava/py_step0_cond_ref/ref_tensors.npz
  /mnt/hdd/nava/cpp_step0_cond_fp8fold_q8/
  /mnt/hdd/nava/cpp_step0_cond_fp8fold_q8_oracle/
  /mnt/hdd/nava/py_vs_cpp_step0_cond_fp8fold_q8_audio_diff.txt
  /mnt/hdd/nava/py_vs_cpp_step0_cond_fp8fold_q8_audio_slice_diff.txt
Recovery: build target nava, run Python with NAVA_DUMP_REF=1 and C++ with
NAVA_STOP_AFTER_STEP0_COND=1 LONGCAT_DUMP_DIR=<cpp dir>, then compare with
tools/nava_audio_diff.py and tools/nava_tensor_diff.py.
Update: render-path LONGCAT dumps are disabled for compute_va(), so the useful
block-tap path is Python NAVA_DUMP_REF -> tools/nava_npz_to_bin.py -> C++
single-forward oracle (`./build-nava/bin/nava <gguf> <bin-dir> <out-dir>`).
The q8 oracle has run; F16 oracle is the next precision check.
```

Python reference trajectory with C++ seed-42 init noise:

```text
/mnt/hdd/nava/py_inject42_audio_traj/
/mnt/hdd/nava/py_inject42_audio_traj_final_audio_latent.bin
wall=832s, peak_vram=10111 MiB, peak_ram ~= 27.6 GB
```

C++ q8 float-timestep trajectory:

```text
/mnt/hdd/nava/cpp_seed42_q8_float_audio_traj/
/mnt/hdd/nava/cpp_seed42_q8_float_final_audio_latent.bin
latent vs Python: corr=0.968068 reldiff=0.252856 std_ratio=1.020164
wav vs py16: rms_ratio=1.081156 centroid_ratio=0.927282
```

Branch comparison shows exact initial noise and the first meaningful divergence
inside the first DiT velocity prediction. For the best latent-parity run
(`fp8fold q8 + float model timesteps`):

```text
aud_noise.bin      corr=1.000000 rmsdiff=0.000000 reldiff=0.000000 std_ratio=1.000000
aud_step_00.bin    corr=0.999988 rmsdiff=0.005000 reldiff=0.005048 std_ratio=1.000045
va_cfg_00.bin      corr=0.992883 rmsdiff=0.229976 reldiff=0.144215 std_ratio=1.070831
va_cond_00.bin     corr=0.999329 rmsdiff=0.042641 reldiff=0.039281 std_ratio=1.000702
va_mmask_00.bin    corr=0.999942 rmsdiff=0.012430 reldiff=0.010616 std_ratio=1.000666
va_uncond_00.bin   corr=0.998981 rmsdiff=0.050828 reldiff=0.045182 std_ratio=1.001318
```

The same pattern holds for q8 float, with exact initial noise and larger
model-velocity branch errors amplified by CFG:

```text
aud_noise: exact
va_cfg   max_reldiff=0.337244 mean=0.243393 min_corr=0.943076
va_cond  max_reldiff=0.246793 mean=0.128869 min_corr=0.969675
va_mmask max_reldiff=0.245261 mean=0.124713 min_corr=0.970066
va_uncond max_reldiff=0.243389 mean=0.145920 min_corr=0.970614
```

CFG recomposition was checked directly from dumps:

```text
va_cfg == cond + 2*(cond-uncond) + 2*(cond-mmask)
```

Python and C++ both satisfy that equation to float epsilon, so CFG sign/scale is
not the bug.

The next useful diagnostic is not another full decode A/B. It is to localize
within the first DiT velocity call, starting with `va_cond_00` and/or
`va_uncond_00`, by dumping comparable per-block audio hidden states for the
actual denoise step-0 inputs. The existing one-forward oracle proves that a
separate reference input matches well, but it does not yet say which block/layer
first diverges under the real seed-42 denoise trajectory. A good next probe:

```text
1. Add a narrow "dump first audio DiT call only" mode on both Python and C++.
2. For step 0, branch cond/uncond/mmask, dump audio-token hidden states after
   each double block, each single block, and the final audio head.
3. Compare block-by-block. If block 0 already has a few-percent audio error,
   drill into q/k/v/o, norm modulation, RoPE, and MLP for that block. If early
   blocks match and error appears later, the first bad block is the target.
4. Prefer this over more full 14-minute Python renders; reuse the existing
   seed-42 init noise and stop after the first denoise forward.
```

No-flash C++ q8 was worse, so flash attention is not the fix:

```text
/mnt/hdd/nava/cpp_seed42_q8_noflash_audio_traj/
latent vs Python: corr=0.968739 reldiff=0.254497 std_ratio=1.038822
wav vs py16: rms_ratio=1.118002 centroid_ratio=0.941279
```

fp8-fold q8 with float timesteps is the best latent trajectory tested:

```text
/mnt/hdd/nava/cpp_seed42_fp8fold_float_audio_traj/
/mnt/hdd/nava/cpp_seed42_fp8fold_float_final_audio_latent.bin
latent vs Python: corr=0.976365 reldiff=0.218818 std_ratio=1.025669
wav vs py16: rms_ratio=1.111571 centroid_ratio=0.936633
```

Mixed fp8-fold q8 with audio projection matrices forced F16 was tested:

```text
models/nava-dit-fp8fold-q8_audiof16.gguf
tensors: 309 q8_0 + 743 f16 = 1052
params buffer: 7539.80 MB(VRAM), compute buffer: 1771.36 MB(VRAM)

/mnt/hdd/nava/cpp_seed42_fp8fold_audiof16_audio_traj/
/mnt/hdd/nava/cpp_seed42_fp8fold_audiof16_final_audio_latent.bin
latent vs Python: corr=0.972642 reldiff=0.234747 std_ratio=1.023429
wav vs py16: rms_ratio=1.097733 centroid_ratio=0.940379
```

This is a small loudness/tone improvement over plain fp8-fold, but the latent
trajectory is worse. Treat it as a diagnostic, not a proven default.

Single-forward oracle on the existing Peter I2V reference dump still passes:

```text
ref: /mnt/hdd/nava/cpp-runs/_ref_peter_i2v/ref_tensors.npz
cpp f16: /mnt/hdd/nava/cpp-runs/_ref_peter_i2v/cpp_out
cpp q8 : /mnt/hdd/nava/cpp-runs/_ref_peter_i2v/cpp_out_q8

f16 velocity_audio: PSNR=53.63 dB, cos=0.99993
q8  velocity_audio: PSNR=53.14 dB, cos=0.99992
all block taps pass PSNR>=40 dB
```

Interpretation: C++ and Python agree very closely for one real-shape model
forward. The audible dirt appears to be accumulated denoising sensitivity to the
effective DiT weights/precision over repeated CFG-guided steps, not an obvious
wrong input, decode path, CFG algebra, scheduler equation, or attention-kernel
implementation bug.

## 2026-06-05 final state / recovery note

The last GPU run in this session completed and should not be resumed; there are
no live render processes from it.

Command that was running:

```bash
OUT_NAME=cpp_peter_fp8fold_modelinput_seed42 \
OUT_LATENT=/mnt/hdd/nava/cpp_seed42_fp8fold_modelinput_final_audio_latent.bin \
OUT_WAV=/mnt/hdd/nava/audio_demo/29_cpp_fp8fold_modelinput_seed42.wav \
DEMO_ROW_ID=29 \
DEMO_ROW_LABEL='cpp fp8fold model-input-anchor seed42' \
RENDER_LOG=/mnt/hdd/nava/cpp_seed42_fp8fold_modelinput_render.log \
AUDIO_TRAJ_DIR=/mnt/hdd/nava/cpp_seed42_fp8fold_modelinput_audio_traj \
LATENT_COMPARE_LOG=/mnt/hdd/nava/cpp_seed42_fp8fold_modelinput_latent_compare.txt \
WAV_COMPARE_LOG=/mnt/hdd/nava/cpp_seed42_fp8fold_modelinput_wav_compare.txt \
bash tools/nava_run_fp8fold_latdump.sh
```

Result:

```text
latent vs Python injected final:
  corr=0.973509 reldiff=0.231127 std_ratio=1.023843
wav vs 16_python_from_cpp42noise.wav:
  rms_ratio=1.106451 peak_ratio=1.059168 centroid_ratio=0.933656
  band ratios: 80-300=1.101 300-800=1.120 800-1500=1.060
               1500-3000=1.228 3000-5000=1.090 5000-7600=0.997
```

Interpretation: Python-style per-forward I2V anchor substitution did not fix the
audible issue and is slightly worse than the current best normal fp8-fold float
run:

```text
/mnt/hdd/nava/cpp_seed42_fp8fold_float_final_audio_latent.bin
latent: corr=0.976365 reldiff=0.218818 std_ratio=1.025669
wav:    rms_ratio=1.111571 centroid_ratio=0.936633
```

`examples/nava/main.cpp` was left in a recoverable state:

- default render behavior is back to the known-better historical C++ path:
  clean I2V anchor is part of sampler state and is re-pinned after each update.
- `NAVA_MODEL_INPUT_ANCHOR=1` enables the Python-style per-forward anchor
  diagnostic tested by row 29.
- `NAVA_REPIN_AFTER_STEP=1` can be combined with `NAVA_MODEL_INPUT_ANCHOR=1` for
  A/Bs.
- `NAVA_INJECT_VIDEO=/path/to/video_noise.bin` and
  `NAVA_INJECT_AUDIO=/path/to/audio_noise.bin` inject exact sampler init state.
- `NAVA_STOP_AFTER_STEP0_COND=1` stops after the first conditional DiT forward
  and writes `va_cond_00.bin` when `NAVA_DUMP_AUDIO_TRAJ` is set.

The corrected single-forward oracle used Python's per-token I2V timestep:

```text
/mnt/hdd/nava/py_step0_cond_ref/ref_tensors.npz
/mnt/hdd/nava/cpp_step0_cond_fp8fold_q8_i2v_oracle/
/mnt/hdd/nava/py_vs_cpp_step0_cond_fp8fold_q8_i2v_audio_slice_diff.txt
velocity_audio corr=0.999936 reldiff=0.011364
```

Input cross-checks:

```text
/mnt/hdd/nava/step0_video_input_crosscheck.txt
audio noise: exact
video frames 1-12: exact
video frame 0 anchor: corr=0.927721 reldiff=0.380470

/mnt/hdd/nava/step0_pyanchor_crosscheck.txt
forcing /mnt/hdd/nava/py_step0_frame0_anchor.bin fixes step-0 cond to oracle level:
va py vs pyanchor reldiff=0.011667
```

However, full py-anchor renders were worse:

```text
/mnt/hdd/nava/cpp_seed42_fp8fold_pyanchor_final_audio_latent.bin
latent: corr=0.863865 reldiff=0.507309 std_ratio=0.965840

/mnt/hdd/nava/cpp_seed42_fp8fold_pyanchor_modelinput_final_audio_latent.bin
same audio metrics as pyanchor pinned-state run

/mnt/hdd/nava/cpp_seed42_fp8fold_injectstate_pyanchor_final_audio_latent.bin
same audio metrics again
```

So the frame-0 image anchor mismatch is real and explains the step-0 cond oracle
gap, but it is not the audible parity fix. Repinning-vs-model-input state has
also been falsified for audio.

CFG algebra was recomputed from dumps and is exact on both sides:

```text
va_cfg = cond + 2*(cond-uncond) + 2*(cond-mmask)
```

The current best explanation is still branch/model trajectory drift amplified by
guidance. The next target should be a narrow first-forward block tap for the real
denoise inputs, not another full 14-minute Python render:

```text
Dump Python and C++ step-0 cond/uncond/mmask video+audio hidden states after each
double block, each single block, and the final audio head. Compare block-by-block.
If block 0 is already off, inspect q/k/v/o, norm modulation, RoPE, and MLP there.
If blocks stay close until later, the first bad block is the next concrete target.
```

Build verification after the final code gating:

```bash
export PATH=/mnt/hdd/3d/avatar-shootout/toolchain/bin:$PATH
export LD_LIBRARY_PATH=/mnt/hdd/3d/avatar-shootout/toolchain/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
cmake --build build-nava --target nava -j8
```

```text
[100%] Built target nava
```

## Repro commands

Python trajectory wrapper, preflight only:

```bash
bash -n tools/nava_run_python_audio_traj.sh
DRY_RUN=1 bash tools/nava_run_python_audio_traj.sh
```

Actual Python run when the GPU slot is available:

```bash
bash tools/nava_run_python_audio_traj.sh
```

Default outputs:

```text
/mnt/hdd/nava/py_inject42_audio_traj/
/mnt/hdd/nava/py_inject42_audio_traj_final_audio_latent.bin
/mnt/hdd/nava/py_vs_cpp_q8_intt_audio_traj_compare.txt
/mnt/hdd/nava/py_vs_cpp_q8_intt_final_latent_compare.txt
```

Important: the current default comparison target is the existing
`cpp_seed42_q8_intt_audio_traj`, because that is the only q8 trajectory dump
currently on disk. For best-current parity, first capture a C++ q8 **float**
timestep trajectory (no `NAVA_UNIPC_INT_TIMESTEP`) with:

```bash
GGUF=models/nava-dit-q8_0.gguf \
OUT_NAME=cpp_peter_q8_float_traj_seed42 \
OUT_LATENT=/mnt/hdd/nava/cpp_seed42_q8_float_final_audio_latent.bin \
OUT_WAV=/mnt/hdd/nava/audio_demo/24_cpp_q8_float_render_seed42.wav \
DEMO_ROW_ID=24 \
DEMO_ROW_LABEL="cpp q8 float-t render seed42" \
LATENT_COMPARE_LOG=/mnt/hdd/nava/cpp_seed42_q8_float_latent_compare.txt \
WAV_COMPARE_LOG=/mnt/hdd/nava/cpp_seed42_q8_float_wav_compare.txt \
RENDER_LOG=/mnt/hdd/nava/cpp_seed42_q8_float_render.log \
AUDIO_TRAJ_DIR=/mnt/hdd/nava/cpp_seed42_q8_float_audio_traj \
bash tools/nava_run_fp8fold_latdump.sh
```

CPU-only dry-run/preflight:

```bash
DRY_RUN=1 bash tools/nava_run_fp8fold_latdump.sh
```

If the render was already completed and only the compare/extract/index steps need
to be rerun:

```bash
SKIP_RENDER=1 bash tools/nava_run_fp8fold_latdump.sh
```

Then compare Python trajectory to the float C++ trajectory:

```bash
CPP_AUDIO_TRAJ=/mnt/hdd/nava/cpp_seed42_q8_float_audio_traj \
CPP_FINAL_LATENT=/mnt/hdd/nava/cpp_seed42_q8_float_final_audio_latent.bin \
TRAJ_COMPARE_LOG=/mnt/hdd/nava/py_vs_cpp_q8_float_audio_traj_compare.txt \
LATENT_COMPARE_LOG=/mnt/hdd/nava/py_vs_cpp_q8_float_final_latent_compare.txt \
bash tools/nava_run_python_audio_traj.sh
```

Branch interpretation:

```text
va_cond diverges first   -> base conditional forward/input/layout issue
va_uncond diverges first -> negative/context branch issue
va_mmask diverges first  -> masking_modality/self-attention split issue
va_cfg diverges first    -> CFG algebra/scales issue
aud_step diverges first  -> scheduler/update state issue
```

Old fp8-fold wrapper defaults, kept for reference:

```text
/mnt/hdd/nava/cpp_seed42_fp8fold_final_audio_latent.bin
/mnt/hdd/nava/cpp_seed42_fp8fold_audio_traj/
/mnt/hdd/nava/audio_demo/22_cpp_fp8fold_render_seed42.wav
/mnt/hdd/nava/cpp_seed42_fp8fold_render.log
/mnt/hdd/nava/cpp_seed42_fp8fold_latent_compare.txt
/mnt/hdd/nava/cpp_seed42_fp8fold_wav_compare.txt
```

Expanded command:

```bash
export PATH=/mnt/hdd/3d/avatar-shootout/toolchain/bin:$PATH
export LD_LIBRARY_PATH=/mnt/hdd/3d/avatar-shootout/toolchain/lib
export NAVA_DUMP_AUDIO_LATENT=/mnt/hdd/nava/cpp_seed42_fp8fold_final_audio_latent.bin

./build-nava/bin/nava render --cuda \
  --gguf models/nava-dit-fp8fold-q8_0.gguf \
  --context /mnt/hdd/nava/cpp-runs/_ref_peter_i2v/bin/context.bin \
  --neg-context /mnt/hdd/nava/vneg_now.bin --audio-neg-context /mnt/hdd/nava/aneg_now.bin \
  --image /mnt/hdd/nava/peter_896x448.bin \
  --vae models/wan2.2-vae-48ch-f16.gguf --audio-vae models/nava-ltx-audio-vae-f16.gguf \
  --steps 10 --frames 13 --width 896 --height 448 --fps 24 \
  --cfg 3.0 --cfg-align 3.0 --cfg-align-audio 2.0 --shift 5.0 --seed 42 \
  --runs-dir /mnt/hdd/nava/cpp-runs --out-name cpp_peter_fp8fold_latdump_seed42
```

Then compare:

```bash
python3 tools/nava_compare_audio_latents.py \
  /mnt/hdd/nava/py_inject42_final_audio_latent.bin \
  /mnt/hdd/nava/cpp_seed42_fp8fold_final_audio_latent.bin
```

Pass criterion: materially improve over current q8 (`corr=0.968739`,
`reldiff=0.254497`, `std_ratio=1.038822`). If it does not improve, the next
suspect is sampler/trajectory accumulation rather than model effective weights.

If the latent improves, extract the rendered audio and compare tone/loudness with:

```bash
python3 tools/nava_compare_audio_wavs.py \
  /mnt/hdd/nava/audio_demo/16_python_from_cpp42noise.wav \
  <new_cpp_extracted_audio.wav>
```
