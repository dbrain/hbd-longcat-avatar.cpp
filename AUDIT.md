# LongCat-Video-Avatar C++/ggml port — structural-divergence audit

Read-only comparison of `src/*.hpp/.cpp` against the PyTorch reference at
`~/dev/longcat-video-ref/longcat_video/`. Scope per request: the LESS-validated
paths (continuation/chaining, audio, 3D-RoPE, schedule, VAE-norm). The single-clip
per-block self-/cross-attn/SwiGLU/RoPE/qk-norm math is TRUSTED (cos-0.999 block-0
oracle) and was NOT re-litigated.

The "watercolour melt" ref-anchor bug (missing per-segment portrait re-anchor) is
already being fixed; this audit looks for OTHER divergences. The fixed ref-anchor
machinery (`gen_wan_pe_ref`, 3-way self-attn split, ref-positioned RoPE grid) was
checked and is **faithful** — see "Verified-correct" at the bottom.

## Ranked divergences

| # | Area | Impact | Divergence (one line) |
|---|------|--------|------------------------|
| 1 | Continuation timestep | **HIGH** — ✅ FIXED (2026-05-27) | `process_timesteps` now zeroes ALL fixed-cond frames per-frame off the denoise_mask (mask==0 ⇒ t=0), matching `timestep[:, :num_cond_latents]=0` (ref+cond_tail). Was: only frame 0. |
| 2 | Continuation audio align | **HIGH** — ✅ FIXED (2026-05-27) | When num_ref_latents>0 the per-render audio_hidden now duplicates audio frame 0 for the ref slot then trims to latent T (`longcat_avatar.hpp` build_graph), matching `longcat_video_dit_avatar.py` L438-441. |
| 3 | Continuation latent length | **MED–HIGH** — ✅ FIXED (2026-05-27) | The ref is PREPENDED as an EXTRA latent frame (init_latent T = base+num_ref), then stripped before decode via `latents.ref_image_num` — matches `cat([ref, latents])` + `latents[:, :, num_ref:]`. No generated frame lost; RoPE grid uses ref-positioned grid_t. Was: overwrote frame 0 (lost a frame + off-by-one). |
| 4 | Whisper feature truncation | **MED** | Reference trims encoder output to `video_length*2` BEFORE interpolation; C++ interpolates full `T_enc` → possible global lip-sync time-compression when audio is zero-padded |
| 5 | `enhance_hf` tail schedule | **LOW** (scope) | Non-distill HF tail schedule (`tail_uniform` 500→0) not implemented; distill path unaffected |
| 6 | 3-term CFG (text+audio guidance) | **LOW** (scope) | `audio_guidance_scale`/`text_guidance_scale` 3-term CFG not implemented; correct for distilled guidance=1.0 default, breaks non-distill CFG renders |

---

## 1. Continuation per-frame timestep only zeroes frame 0  — HIGH

**Reference** `pipeline_longcat_video_avatar.py` `generate_avc` (no-kv-cache branch,
the path the C++ emulates), L1405-1406:
```python
if not use_kv_cache:
    timestep[:, :num_cond_latents] = 0   # ALL cond frames (ref + cond_tail) -> t=0
```
and `generate_ai2v` L1068: `timestep[:, :1] = 0` (only 1 cond frame there).

**Port** `src/stable-diffusion.cpp` `process_timesteps` L1705-1711:
```cpp
auto new_timesteps = std::vector<float>(init_latent.shape()[2], timesteps[0]);
if (!denoise_mask.empty()) {
    float value = denoise_mask.index(0,0,0,0,0);
    if (value == 0.f) new_timesteps[0] = 0.f;   // <-- ONLY index 0
}
```
It only consults `denoise_mask[0]` and only zeroes `new_timesteps[0]`. For **ai2v**
(`num_cond_latents==1`) this is correct. For **continuation** the fixed-cond region
is `num_cond_latents = num_ref(1) + cond_tail(4) = 5` latent frames (defaults:
`--cont-cond-frames 13` → 4 latents, +1 ref). Frames 1..4 are held fixed in latent
space by the `denoise_mask` (L2066) but their **adaLN timestep stays at the noisy
`t`** instead of 0.

Consequence: the cond_tail frames' adaLN modulation (shift/scale/gate_msa,
shift/scale/gate_mlp) and the final-layer modulation are computed at the wrong
timestep. Those frames are discarded on output, but they are the K/V the noise
frames attend to in self-attn (cond pass + noise pass) — so a wrong-`t` normed cond
representation contaminates the generated frames. This is the same *class* of error
as the ref-anchor bug (cond conditioning fed wrong), and is the most likely
remaining quality regressor in chaining.

**Fix sketch:** zero `new_timesteps[0 .. num_cond_latents-1]` by reading the
denoise_mask per-frame (mask==0 ⇒ t=0), not just frame 0. The mask already encodes
exactly the fixed-cond frames (`fill_slice(...,0,num_cond_latents,0.0f)` at L4888).

---

## 2. Missing ref-frame audio prepend/trim in continuation  — HIGH

**Reference** `longcat_video_dit_avatar.py` L438-441:
```python
if num_ref_latents is not None and num_ref_latents > 0:
    audio_start_ref = audio_hidden_states[:, [0], :, :]      # duplicate first audio frame
    audio_hidden_states = torch.cat([audio_start_ref, audio_hidden_states], dim=1)
audio_hidden_states = audio_hidden_states[:, -N_t:]          # trim to latent T
```
So when a ref latent is prepended, the audio is shifted: the **ref** slot is given a
*duplicate* of audio[0], and cond/noise frames keep audio[0..]. The audio is then
trimmed to the last `N_t` (= latent T including ref).

**Port** `src/longcat_avatar.hpp`: `audio_proj` returns `[768,32,N_t]` with
`N_t = 1 + (T_video-1)/4` (= 24 for 93 frames) and **never prepends a ref frame**
(`build_proj_inputs` is ref-agnostic; the runner calls `audio_proj` then threads the
result straight in). The per-block audio path then strips `n_cond_frames` from the
FRONT (`src/longcat_avatar.hpp` L607-612):
```cpp
a_noise = ggml_view_3d(... audio, ... T_noise, ... offset = audio->nb[2]*n_cond_frames);
```
With `n_cond_frames=5` it takes audio frames [5 .. 5+T_noise). Because the C++ never
adds the ref-dup frame, every audio frame is off by one relative to the reference
(the ref slot consumes a real audio frame instead of a dup), and the noise frames
are driven by audio shifted ~1 latent frame (~4 video frames @ 25fps) early.

Note this is *separate from* and *compounds* the divergence #3 frame-count mismatch.

**Fix sketch:** in the continuation path, prepend a duplicate of audio frame 0 to
`audio_hidden` when `num_ref_latents>0`, then trim to the latent T, before threading
into the blocks — mirroring L438-441. Equivalently, build the proj inputs with the
ref frame's window duplicating frame 0.

---

## 3. Ref latent overwrites a frame instead of being prepended  — MED–HIGH

**Reference** `generate_avc` L1374-1379:
```python
latents = prepare_latents(num_frames=93, ...)   # T = 24 latent frames
...
if ref_latent is not None:
    num_cond_latents += 1
    num_ref_latents = 1
    latents = torch.cat([ref_latent, latents], dim=2)   # T -> 25 (ref is EXTRA)
```
The ref is an *additional* 25th latent frame. After denoising it is stripped
(L1494-1496: `latents = latents[:,:,num_ref_latents:]`), so the segment still
decodes **24 latent frames = 93 video frames** of genuine content.

**Port** `src/stable-diffusion.cpp` L4865-4885: `init_latent =
generate_init_latent(W,H,request->frames=93)` → **T = 24**, then the ref is written
*into* frame 0 (`slice_assign(...,2,0,1,ref_latent)`) and cond_tail into [1..5).
There is no prepend; T stays 24. The ref therefore *occupies* what would have been a
generated latent slot. On output the first `cond_decoded_v=13` frames (which include
the ref at index 0) are dropped.

Consequences vs reference:
- One fewer genuinely-generated latent frame per segment (the ref eats a slot).
- The noise frames' temporal RoPE positions differ: `gen_vid_ids_ref` gives
  cond_tail+noise the grid `0,1,…,22` (T-1=23 non-ref frames), whereas the reference
  noise frames span `0,1,…,23` (24 non-ref frames). Off-by-one temporal grid for the
  whole noise block.
- Combined with #2, the audio↔latent frame mapping and the RoPE temporal grid are
  both shifted, which can degrade lip-sync continuity at the seam.

**Fix sketch:** allocate the segment latent with T = (segment frames) + num_ref,
prepend the ref at index 0, denoise, then strip the ref before decode — matching the
reference's cat/strip. This also makes #1's `num_cond_latents` and #2's `N_t` line
up naturally.

---

## 4. Whisper hidden states interpolated over full T_enc  — MED

**Reference** `get_audio_embedding_whisper` L603-612:
```python
audio_prompts = torch.cat(enc_chunks, dim=1)        # [1, T_enc_total, 33, D]
audio_prompts = audio_prompts[:, :video_length * 2] # TRUNCATE to 2x video frames first
feat0 = linear_interpolation_fps(audio_prompts[:,:,0:8].mean(2), 50, fps, video_length)
...   # interpolate the truncated [:, :video_length*2] window to video_length
```
The encoder output is trimmed to `video_length*2` mel-frame-equivalents *before*
grouping/interpolation, so the 50→25 fps map is exactly 2:1 over the valid region.

**Port** `src/longcat_audio.hpp` `build_full_audio_emb` L715-737: interpolates the
**entire** `T_enc` to `video_length` with `align_corners=True`:
```cpp
src = o * (T_enc - 1) / (out_len - 1);   // maps [0,T_enc) -> [0,video_length)
```
There is no `video_length*2` truncation. When `T_enc ≈ video_length*2` (audio not
over-padded) the two are numerically ~identical. But continuation pads the wav to
`generate_duration` (reference repro L262-267); if the padded `T_enc` materially
exceeds `video_length*2`, the C++ compresses more encoder frames into the same
`video_length`, globally time-warping the audio features (lip-sync drift that grows
with padding). Lower confidence than #1–#3 because in the common (non-over-padded)
case it is a no-op; flag + measure with a padded clip.

**Fix sketch:** clamp the interpolation source span to `video_length*2` encoder
frames (i.e. interpolate `min(T_enc, video_length*2) -> video_length`).

---

## 5. `enhance_hf` HF-tail schedule not ported  — LOW (deliberate scope)

**Reference** `generate_avc` L1328-1337 builds a 10-step uniform tail (`500 → 0`)
appended to the filtered timesteps when `enhance_hf=True` (the non-distill default).
The port's schedule (`build_longcat_dmd_sigmas`, `src/stable-diffusion.cpp` L3089)
only implements the **distill** 8-step DMD schedule (`use_distill`), which is the
target config. `enhance_hf` and `use_distill` are mutually exclusive in the
reference (`assert not (use_distill and enhance_hf)`), so this only matters for
non-distill renders, which the port does not target. Note in docs if non-distill is
ever wanted.

## 6. 3-term (text + audio) CFG not ported  — LOW (deliberate scope)

**Reference** `generate_avc` L1454-1476 runs an extra uncond forward and combines
`uncond + text_scale*(cond-uncond_text) + audio_scale*(uncond_text-uncond)`. The
port runs a single forward (no CFG). Correct for the distilled default
(`text_guidance_scale=audio_guidance_scale=1.0` ⇒ `do_classifier_free_guidance=False`
⇒ single forward in the reference too). Only diverges if a caller sets a guidance
scale > 1 on a non-distilled model. Out of the distill scope.

---

## Verified-correct (looked divergent, are equivalent / faithful ports)

- **DMD v1.5 sigma schedule** (`build_longcat_dmd_sigmas`): bit-faithful to
  `get_timesteps_sigmas(avatar-v1.5, use_distill=True)` + `set_timesteps`'s linear
  shift=7.0. `distill_indices` math, the `flip(linspace(0,1,1000))[di]` →
  `(999-di)/999`, the shift formula `shift*s/(1+(shift-1)*s)`, and the terminal-0
  append all match (`scheduler_config.json`: shift 7.0, linear, no dynamic shift, no
  shift_terminal).
- **Ref-positioned 3D RoPE grid** (`gen_vid_ids_ref`): `grid_t = [ref_img_index,
  0,1,2,…]` matches `avatar/rope_3d.py::precompute_freqs_cis_3d`'s
  `concat([frame_index], arange(0, num_frames-num_ref_latents))`. Axis split
  {44,42,42} matches `dim_t=head_dim-4*(head_dim//6)=44`, `dim_h=dim_w=2*(head_dim//6)
  =42`. Interleaved (GPT-J) convention matches `repeat(...,r=2)+rotate_half`.
- **3-way self-attn split** (`self_attn`, `num_ref_latents>0` branch): the
  ref/cond/noise passes + the `mask_frame_range` carve-out (`start_noise`/`end_noise`
  formulas, the front/maskref/back q-slices against ref-excluded k/v) reproduce
  `avatar/attention.py` `Attention.forward`'s `num_cond_latents>1` branch exactly,
  including the noise-local token offsets `start_pos = start_noise*n_per_frame`.
- **VAE latent normalization**: `vae_to_diffusion_latents = (x-mean)/std` (Wan VAE
  `scale_factor=1.0`) matches `normalize_latents = (x-mean)*(1/std)`; denorm matches.
  Wan2.1 16-ch mean/std constants present.
- **Audio AudioProjModel + dual-window**: `proj1`(32000)/`proj1_vf`(51200) dims,
  the first-frame/latter-frame window construction (`build_proj_inputs`: mid+1 / 1 /
  W-mid sub-frame entries, concat order first/middle/last), the 33→5 hidden-state
  grouping (`[0:8],[8:16],[16:24],[24:32],hs[32]`), `vae_scale=4`, `audio_window=5`,
  ENC_FPS=50, −23 LUFS norm, and the per-block audio cross-attn (per-frame batched,
  `mod_norm_attn` reuse, `pre_video_crs_attn_norm` on q, audio prenorm = Identity for
  v1.5, gated additive residual) all match the reference. The continuation-only
  ref-prepend (#2) is the lone audio gap.
- **Audio per-segment offset**: `audio_frame_offset = seg*new_per_seg` with
  `new_per_seg = 93-13 = 80` equals the reference `audio_start_idx +=
  audio_stride*(num_frames-num_cond_frames) = 80/seg`. Global-timeline windowing
  (whisper run once on the full wav, windowed per segment) matches.
- **denoise_mask latent-hold**: `noised_input = noised_input*mask + init_latent*(1-mask)`
  correctly pins ALL cond frames in latent space (the bug is in the *timestep*, #1,
  not the latent hold).
