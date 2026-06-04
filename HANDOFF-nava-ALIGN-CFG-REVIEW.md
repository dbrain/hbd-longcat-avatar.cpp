# NAVA cpp — align_3d_cfg fix: INDEPENDENT REVIEW BRIEF

You are auditing a change made to fix a known bug in the NAVA cpp port (branch `nava-port`
in `/home/dbrain/dev/longcat-avatar.cpp`). Be skeptical. Verify each claim from primary
sources (the python reference + the cpp code), don't trust this doc. The python reference
is `/home/dbrain/dev/NAVA` (the real NAVA repo). The working tree is UNCOMMITTED — do not
commit. cpp builds on-box are fine (toolchain below). Drive any GPU work from the MAIN loop
(subagents stall on background GPU jobs); GPU is a single 12GB RTX 3060, serial.

## The bug being fixed
On the hard official sample `peter_talk`, the cpp render produces audio that DIVERGES TO NOISE
(env_CV ~0.18; >0.5 = speech, <0.5 = noise), while easy prompts (`action`) are clean (0.99).
Prior elimination (see HANDOFF-nava-QUALITY.md) ruled out resolution, sampler, context bin,
truncation, and the (optional) speaker embed. It is prompt-dependent fragility.

## The DIAGNOSIS I reached (verify it)
1. **The cpp DiT forward is FAITHFUL, including the audio stream.** So the bug is NOT the
   forward — it's the sampler/CFG. Evidence: a step-0 forward diff of the cpp audio stream vs
   the PyTorch bf16 reference on the smoke case gave 45 dB on the audio velocity head and
   59–75 dB on every joint block's audio-token slice (cos 0.9999+). The 45 dB head matches the
   known F16-vs-bf16 head precision floor (same as video).
2. **The cpp render used a 2-way CFG; python uses a 3-way `align_3d_cfg` CFG.** `nava_640.yaml`
   sets `align_3d_cfg: true`. Python (verify `nava_src/pipeline_nava.py:540-555`) computes,
   for BOTH streams, a cond-based 3-term guidance with a THIRD forward pass that runs with
   `masking_modality=True`:
   ```
   eps_vision = eps_cond + g_v*(eps_cond - eps_uncond) + ga_v*(eps_cond - eps_mmask)   # g_v=3, ga_v=3
   eps_audio  = eps_cond + g_a*(eps_cond - eps_uncond) + ga_a*(eps_cond - eps_mmask)   # g_a=2, ga_a=2
   ```
   `masking_modality=True` (verify `nava_src/models/nava/modules/model_mm.py`, the
   `use_joint_attention=(not masking_modality)` flag in the double/single attention blocks)
   means the self-attention runs SEPARATE intra-modal (video attends only to video, audio only
   to audio) instead of joint cross-modal. The cpp render omitted this entire 3rd pass + align
   term and used the wrong base (uncond instead of cond). The (timbre 4th term) drops for the
   no-speaker case since `effective_timbre = timbre_cfg and spk_embs is not None`.

## The FIX I implemented (audit for correctness)
All in the working tree. Key functions:

### `src/nava.hpp`
- `DoubleBlock::forward(...)` — added trailing `bool joint_attn = true`. When false
  (masking_modality), instead of concatenating qv/qa and doing one joint `Rope::attention`,
  it slices the joint `pe` (token axis is `ne[3]`) into video `[0,L_vid)` and audio
  `[L_vid,L_total)` parts and runs TWO separate `Rope::attention` calls (video qkv with video
  pe, audio qkv with audio pe). Rest unchanged.
- `SingleBlock::forward(...)` — added `int64_t L_vid=-1, bool joint_attn=true`. When false,
  splits q/k/v at `L_vid` on the token axis `ne[2]`, slices pe on `ne[3]`, two attentions,
  re-concats on dim 1.
- `NavaRunner` — new member `bool mask_modality=false;`. `build_graph` passes `!mask_modality`
  as `joint_attn` to every block (and `L_vid` to SingleBlock). `compute_va(...)` got a new
  `bool mask_modality_arg=false` param that sets/clears the member around the compute.

### `examples/nava/main.cpp`
- `RenderOpts` — added `cfg_align=3.0` (video align), `cfg_align_audio=2.0` (audio align);
  CLI flags `--cfg-align`, `--cfg-align-audio`.
- `run_render` step loop — added a 3rd forward `compute_va(latent, audio_latent, ctx_pos, ts,
  /*mask_modality=*/true)` → (vv_mmask, va_mmask), and changed BOTH CFG combines to the
  cond-based 3-term formula above. Env `NAVA_NO_ALIGN_CFG=1` reverts to the old 2-way for A/B.
- `run_single_forward` — env `NAVA_MASK_MODALITY=1` sets `runner->mask_modality=true` so the
  oracle can run the separate-attention path for validation.

## THE PROBLEM (this is why you're reviewing)
With the fix ON, peter got WORSE, not better: **env_CV 0.115 (align on) vs 0.18 (align off)**.
A correct align term should move it toward speech (python's full 3-way gives ~0.66). So either:
- (A) my masking_modality cpp path is numerically WRONG (plausible-but-incorrect), or
- (B) it's correct and q8 quantization sabotages the align term: `(eps_cond - eps_mmask)` is a
  SMALL difference (CPU self-consistency test: cond-vs-mmask audio differ only ~10%), so at q8
  the independent quant noise on the two terms may dominate that 10% signal. (The 2-way term
  `(cond-uncond)` is a LARGE difference, robust to quant.)

Self-consistency check already done (cpp, f16, CPU): mask flag changes the computation
(audio reldiff 0.10, video 0.04), all finite, sane magnitude — so it's not a crash/NaN bug.

A python masking_modality reference dump is being produced to numerically gate the cpp mmask
path (cpp-mmask vs python-mmask, same method as the forward gate). Check
`/mnt/hdd/nava/cpp-runs/_ref_mmask/ref_tensors.npz` — if present, run the diff (below).

## WHAT TO INDEPENDENTLY VERIFY (priority order)
1. **Is the cpp masking_modality forward numerically correct vs PyTorch?** This is THE open
   question. If `_ref_mmask/ref_tensors.npz` exists:
   ```
   cd /home/dbrain/dev/longcat-avatar.cpp
   export PATH=/mnt/hdd/3d/avatar-shootout/toolchain/bin:$PATH
   export LD_LIBRARY_PATH=/mnt/hdd/3d/avatar-shootout/toolchain/lib
   python3 tools/nava_npz_to_bin.py /mnt/hdd/nava/cpp-runs/_ref_mmask/ref_tensors.npz /mnt/hdd/nava/cpp-runs/_ref_mmask/bin
   NAVA_MASK_MODALITY=1 LONGCAT_DUMP_DIR=/mnt/hdd/nava/cpp-runs/_ref_mmask/cpp_out \
     ./build-nava/bin/nava models/nava-dit-f16.gguf /mnt/hdd/nava/cpp-runs/_ref_mmask/bin /mnt/hdd/nava/cpp-runs/_ref_mmask/cpp_out
   python3 tools/nava_audio_diff.py /mnt/hdd/nava/cpp-runs/_ref_mmask/ref_tensors.npz /mnt/hdd/nava/cpp-runs/_ref_mmask/cpp_out
   ```
   High dB (like the joint gate: blocks 60+, head ~45) => my mmask code is correct => the
   regression is q8. Low dB => find the bug in the separate-attention path (nava.hpp).
   To regenerate the mmask reference if missing: `bash /mnt/hdd/nava/dump_ref_cpu_mmask.sh`
   (all-CPU bf16, ~minutes; do NOT use the group-offload or streaming variants — they OOM /
   hit a t5-on-CPU device bug respectively).

2. **Audit the pe-slicing.** I claim `gen_nava_joint_pe` (nava.hpp ~line 77) writes the audio
   tokens' rope using audio's OWN 1D positions (`pi` from 0, lines ~133-139), so slicing the
   joint pe's audio portion == python's `rope_apply(q_audio, freqs_audio)` in separate mode.
   Verify this against python's `rope_apply` vs `rope_apply_joint` (model_mm.py). If the joint
   pe offsets audio positions by the video length, my slice is wrong.

3. **Audit the CFG formula + scales + base** in `run_render` against `pipeline_nava.py:540-555`.
   Confirm: base is `eps_cond` (not uncond), scales 3/3 video and 2/2 audio (from nava_640.yaml,
   NOT the function defaults 5/4 in pipeline_nava.py:339-347), audio uncond uses the right neg
   context (cpp uses `--audio-neg-context`; python uses encoded negatives because
   `negative_prompt_mode` defaults True — confirm the cpp neg ctx bins are encoded negatives,
   not zeros).

4. **Re-derive the forward gate** to confirm claim #1 independently:
   ```
   cd /home/dbrain/dev/longcat-avatar.cpp
   LONGCAT_DUMP_DIR=/tmp/cpp_joint ./build-nava/bin/nava models/nava-dit-f16.gguf /mnt/hdd/nava/cpp-runs/_ref/bin /tmp/cpp_joint
   python3 tools/nava_audio_diff.py /mnt/hdd/nava/cpp-runs/_ref/ref_tensors.npz /tmp/cpp_joint
   ```

5. **Reproduce the regression** (q8, ~5min GPU each, MAIN loop only):
   - align off: `NAVA_NO_ALIGN_CFG=1 ./build-nava/bin/nava render --cuda --gguf models/nava-dit-q8_0.gguf --vae models/wan2.2-vae-48ch-f16.gguf --context /mnt/hdd/nava/cpp-runs/_ctx/dense_peter_talk512.bin --neg-context /mnt/hdd/nava/cpp-runs/_ctx/cpp_video_neg_ctx.bin --audio-neg-context /mnt/hdd/nava/cpp-runs/_ctx/cpp_audio_neg_ctx.bin --image /mnt/hdd/nava/peter_896x448.bin --steps 10 --frames 13 --width 896 --height 448 --seed 42 --cfg 3.0 --audio-vae models/nava-ltx-audio-vae-f16.gguf --out-name peter_alignoff --runs-dir /mnt/hdd/nava/cpp-runs`
   - align on: same without `NAVA_NO_ALIGN_CFG=1`, `--out-name peter_alignon`.
   - env_CV: `python3 tools/nava_env_cv.py /mnt/hdd/nava/cpp-runs/peter_align*/clip.webm`

## Suspicious things worth extra scrutiny
- SingleBlock token-axis split: q/k/v from `self_attn->qkv` — confirm token axis is `ne[2]`
  (I split dim 2) and pe token axis is `ne[3]` (I slice dim 3). A wrong axis would silently
  produce plausible-but-wrong values.
- `Rope::attention` with a SLICED pe: does `apply_rope` assume pe is contiguous/full-length?
  (`ggml_ext_slice` conts by default.) If it indexes pe by absolute token position internally,
  the audio slice (starting at pe token 0 after slicing) could misalign.
- Whether masking_modality in python changes ANYTHING besides self-attn topology (I assumed
  only `use_joint_attention`; confirm cross-attn / context / modulation are untouched).
- env_CV variance: one render isn't definitive — re-run / try a second seed before concluding.

## Build / tooling
```
export PATH=/mnt/hdd/3d/avatar-shootout/toolchain/bin:$PATH
export LD_LIBRARY_PATH=/mnt/hdd/3d/avatar-shootout/toolchain/lib
cmake --build build-nava --target nava -j8     # -> build-nava/bin/nava
```
Eye-test server: http://10.0.0.208:8097 (RUNS_DIR=/mnt/hdd/nava/cpp-runs). New diff tools:
`tools/nava_audio_diff.py` (audio-slice PSNR/cos), `tools/nava_env_cv.py` (speech/noise CV).
Python is load-bound and the box has 31GB RAM / 12GB VRAM — bf16 master is 24GB, so only the
all-CPU dump (`dump_ref_cpu_mmask.sh`) fits reliably; group-offload OOMs at pipe.to.
```
