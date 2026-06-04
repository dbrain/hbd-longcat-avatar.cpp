# NAVA audio parity handoff

Current conclusion: the audible C++ dirt/hotter/darker issue is no longer primarily
audio-VAE decode. It is the audio latent that C++ produces before decode.

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

## Next GPU-slot test

Do not rerun the fp8-fold/integer-timestep tests as-is. The next useful GPU test
should target a new sampler/model-input hypothesis, and should still capture the
audio-only trajectory dump. The wrapper can be reused with overrides:

```bash
GGUF=models/nava-dit-q8_0.gguf \
OUT_NAME=<new_name> \
OUT_LATENT=/mnt/hdd/nava/<new>_final_audio_latent.bin \
OUT_WAV=/mnt/hdd/nava/audio_demo/<NN>_<label>.wav \
DEMO_ROW_ID=<NN> \
DEMO_ROW_LABEL="<label>" \
LATENT_COMPARE_LOG=/mnt/hdd/nava/<new>_latent_compare.txt \
WAV_COMPARE_LOG=/mnt/hdd/nava/<new>_wav_compare.txt \
RENDER_LOG=/mnt/hdd/nava/<new>_render.log \
AUDIO_TRAJ_DIR=/mnt/hdd/nava/<new>_audio_traj \
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

Default wrapper outputs:

```text
/mnt/hdd/nava/cpp_seed42_fp8fold_final_audio_latent.bin
/mnt/hdd/nava/cpp_seed42_fp8fold_audio_traj/
/mnt/hdd/nava/audio_demo/22_cpp_fp8fold_render_seed42.wav
/mnt/hdd/nava/cpp_seed42_fp8fold_render.log
/mnt/hdd/nava/cpp_seed42_fp8fold_latent_compare.txt
/mnt/hdd/nava/cpp_seed42_fp8fold_wav_compare.txt
```

If row 22 is already taken by the time this runs, override:

```bash
DEMO_ROW_ID=23 \
OUT_WAV=/mnt/hdd/nava/audio_demo/23_cpp_fp8fold_render_seed42.wav \
bash tools/nava_run_fp8fold_latdump.sh
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
