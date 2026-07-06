# Audio-drive lip-sync regression — root-cause + fix

Repo: /home/dbrain/dev/longcat-avatar-ltxdenoise  (branch ltx-denoise-workflow)
File: src/stable-diffusion.cpp

## TL;DR
The driving-audio latent IS encoded and packed correctly, and STAGE-1 pins it
(mask=0, timestep=0) so stage-1 gets lip-sync. The **STAGE-2 latent-upscale
"refine" pass did NOT pin the driving audio** — it re-denoised the audio slot
from noise while the video refine cross-attends to it, washing the stage-1
mouth motion back out. This bites only when `request.hires.enabled` (the
two-stage sample+refine) is on — which the recent @1280 upscale/parity recipe
turned on. The VAE eviction (`LTXAV_VAE_LAZY`) is NOT the cause.

## The mechanism (a2v, fixed-audio path)
Lip-sync relies on the audio slot being held FIXED:
1. encode drive wav -> audio latent (`encode_ltxav_drive_audio`, :5277; log
   "LTXAV audio-drive: encoded ... (lip-sync target)" :5316) -> stored in
   `latents.audio_latent`, `latents.audio_fixed=true` (:6353-6356).
2. packed into init_latent + denoise_mask, audio slot pinned mask=0 (:7789-7802,
   passing `latents.audio_fixed ? 0.0f : 1.0f`).
3. per step: audio timesteps forced to 0 and the audio slot restored from
   init_latent (:2732-2739, :2751-2763) -> DiT cross-attends a CLEAN driving
   audio -> mouth tracks it.

## The gap (stage-2 refine) — FIXED
`generate_video_ex`, `if (latent_upscale_enabled)` (:9021), builds
`hires_denoise_mask` via `apply_ltxv_refine_image_conditioning` (:9111):

- Plain t2v / **continuation chain segment (no init/end image, not relip)**:
  the function EARLY-RETURNED at :8433-8435 leaving `hires_denoise_mask` EMPTY.
  Stage-2 `sample()` then ran with an empty mask, so `!denoise_mask.empty()` was
  FALSE (:2732): audio_timesteps never zeroed and the audio-slot restore at
  :2762 skipped -> the clean audio slot in x_t was denoised as a video latent
  (`x * c_in`, :2750) -> stage-2 refine washed out the stage-1 lip-sync.

- init/end-image path: :8516-8518 called
  `pack_ltxav_audio_and_video_denoise_mask(video_mask, video_latent, audio_latent)`
  WITHOUT the 4th arg. Its default is `audio_mask_value = 1.f` (:4983) = audio
  slot GENERATED/UNPINNED — same bug even when a mask is built.

Correct call sites that DO pin: stage-1 :7797/:7800 and relip stage-2 :8417 both
pass `latents.audio_fixed ? 0.0f : 1.0f`. The two-stage NON-relip refine was the
only path that forgot.

Symptom fit: the OUTPUT audio decoded at :9308 comes from the (drifted) stage-2
final_latent, but for the music-video koblem muxes the real song separately, so
the audio "sounds fine" while the mouth (baked into the VIDEO latent) is wrong.

## Why the VAE eviction (original prime suspect) is NOT it
`LTXAV_VAE_LAZY` (:8739, :9211) and the two-stage frees (:9133-9148) free only
GPU *VAE param* buffers. The audio latent lives as a CPU `sd::Tensor<float>`
packed into x_t/init_latent BEFORE any eviction; sampling never uses VAE params.
Regime A (offload, prod) auto-re-offloads params from the host CPU/mmap home at
the next encode/decode, so the per-segment encode succeeds (log confirms
T_enc=102, 64667 samples = non-silent). Eviction is a VRAM lever only.

## IMPLEMENTED (2026-07-07)
Two surgical edits to `apply_ltxv_refine_image_conditioning` in
src/stable-diffusion.cpp, mirroring the stage-1 (:7800) and relip stage-2
(:8417) mask semantics exactly:

1. **Continuation/t2v early-return path (~:8433-8454).** Before the early
   `return true`, when `sd_version_is_ltxav(...) && latents.audio_fixed` and the
   upscaled latent carries a packed audio slot (`latent->shape()[3] > lat_ch`):
   slice the video latent, unpack the audio latent, build a full-generated video
   mask (1.0) and set `*denoise_mask = pack_ltxav_audio_and_video_denoise_mask(
   video_mask, video_latent, audio_latent, latents.audio_fixed ? 0.0f : 1.0f)`.
   Gated on `audio_fixed`, so no-drive/t2v/i2v renders keep the empty-mask path
   byte-identical.

2. **init/end-image refine path (~:8540).** Added the missing 4th arg
   `latents.audio_fixed ? 0.0f : 1.0f` to the existing
   `pack_ltxav_audio_and_video_denoise_mask(...)` call. For no-drive renders
   `audio_fixed` is false -> 1.0 = the previous default -> byte-identical; only
   the a2v case changes to 0.0 (pinned).

Result: the stage-2 hires refine now holds the driving audio fixed exactly like
stage-1 and the relip stage-2 path, so the two-stage upscale render preserves
lip-sync. No behavior change for pure t2v/i2v/video (no audio drive).

Not built here (GPU rebuild/validate owned by the coordinator).
