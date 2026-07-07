# Refine only the surviving frames — `LTX_REFINE_CONTEXT_FRAMES=N`

Env-gated knob to cut LTX continuation **hires/refine** VRAM by not spending the refine
pass on continuation frames that get discarded before decode. Default (unset) =
byte-identical to current prod.

## The flow (verified in `src/stable-diffusion.cpp`)

A continuation segment (`seg>0`, `--ltx-chain-segments`) uses the **keyframe-append**
convention (default; `LTXAV_CONT_LEGACY_HEAD=0`):

1. `latents.init_latent` = the segment's base window of **T = video_target_frame_count**
   latent frames (13 for 97 pixel frames: `(97-1)/8+1 = 13`).
2. The prior segment's motion tail (**K = cont_latent_frames**, e.g. 3) is
   **CONCATENATED at the TENSOR-TAIL** as extra guide tokens
   (`apply_ltxav_video_guide_by_keyframe_index`, sd:5138 / 6774), and given a RoPE
   position at the **segment start** (`keyframe_frame_idx=0`). `video_conditioning_frame_count = K`.
   Layout: `[ target(T) , guide(K) ]`, total **T+K = 16**.
3. Base pass samples all **16** frames (full temporal attention → coherence).
4. LTX latent spatial upscale ×2 → 1920×1088, **still 16 T** (upscale preserves frame count, sd:9062).
5. **3-step hires/refine sample on all 16** (sd:9282).
6. `final_latent = slice(final_latent, 2, 0, T)` **crops the K guide tokens** (sd:9439) — they
   NEVER reach the VAE decode.
7. Decode the surviving **T=13** target frames.
8. Harness (`examples/cli/main.cpp:1224`) additionally drops the first `cond_decoded_v`
   **decoded pixel** frames of that window (the warm-up re-render), replacing them with the
   prior segment's already-rendered frames.

**So the refine wastes compute+VRAM on the K=3 guide tokens** that are cropped at step 6
before decode. Those are the frames the knob removes from the refine.

> NB — the original brief's "K prepended at the head, dropped at stitch" describes the
> `LTXAV_CONT_LEGACY_HEAD` / WAN / avatar-ref paths (guide at head, sd:6936/7226). The
> **default LTX chain path is guide-at-tail**, cropped at the latent level (sd:9439). Same
> idea (discard-before-decode), opposite tensor location. The knob keys off
> `video_conditioning_frame_count`, so it is correct regardless of head/tail.

## The knob

`LTX_REFINE_CONTEXT_FRAMES=N` = how many of the K throwaway guide tokens to KEEP in the
refine as tail-context (the N guide tokens **closest to the surviving set**, sliced from the
tail-side; the earliest are dropped first):

- **N=0** — refine only the surviving target frames; no guide context (max VRAM save, max seam risk).
- **N=1/2** — keep N guide tokens so the refine has seam-adjacent context to refine against (likely sweet spot).
- **N ≥ K / unset** — refine all frames = current behavior = byte-identical.

Refine temporal length = `T + min(N, K)`.

### Slice point + why it's clean (no rope rebuild, no reassembly)

Slice inserted at **sd:~9163**, right after `apply_ltxv_refine_image_conditioning` returns
and before `noise = randn_like(x_t)`:

```
x_t = sd::ops::slice(x_t, 2, 0, T_tgt + keep);   // keep = min(N, K)
```

- **RoPE:** for the *plain* continuation refine, `apply_ltxv_refine_image_conditioning`
  returns early (no init/end image, no relip) and leaves `hires_video_positions` **empty**, so
  `sample()` (sd:2846: `video_positions.empty() ? nullptr`) builds default sequential positions.
  The surviving target frames sit at head indices `[0,T)` in both the sliced and unsliced
  tensors → **their positions are identical**; only the trailing context set they attend to
  shrinks. No position rebuild needed.
- **Downstream:** the retained context frames sit at `[T, T+keep)` and are discarded by the
  **same existing crop** at sd:9439 (`slice(final_latent, 2, 0, T)`), so `final_latent` ends up
  exactly the T target frames as today. **No reassembly / bookkeeping change.**
- **Chain seed:** `chain_base_latent` (next-segment seed) is captured **pre-upscale** at
  sd:9057 from the base latent and is **not** touched by the refine slice → the next segment
  chains identically. VERIFIED.
- `noise` is `randn_like(x_t)` *after* the slice → matches the reduced shape.

### Guards (any miss → no-op, byte-identical; the failing guard is logged)

LTXAV · `video_target_frame_count>0` · `video_conditioning_frame_count>0` ·
`!relip_twostage` · no init/end image · **empty `hires_video_positions`** · **empty
`hires_video_reference`** · `x_t.T == T+K`. Unset env or `keep>=K` → no slice.

**Audio IS handled** (this is the deployed lipdub/driven-audio continuation): audio is packed
as frame-count-keyed extra channels, so the packed frame dim can't be sliced directly. Instead
the knob unpacks the audio (`unpack_ltxav_audio_latent` returns `{16, audio_length, 8}`,
independent of video T), slices the **video** frames only, repacks the **same** audio latent,
and rebuilds the audio-pinned `hires_denoise_mask` for the sliced video. `audio_length` is
unchanged so the post-refine audio decode (:9384) and crop (:9439) are unaffected.

Only `hires_video_positions`/`hires_video_reference` non-empty forces a no-op — those arise
**exclusively** from the relip-two-stage (sd:8414) or FLF2V end-image (sd:8538) branches, which
build frame-coupled positions/reference token blocks (a separate follow-up).

## Frame counts / expected VRAM — K=3, T=13 (97f 1080p continuation)

| N (context) | refine T | approx refine peak¹ |
|---|---|---|
| 0 | 13 | ~10527 MiB (single-97f level; fits) |
| 1 | 14 | ~11.3 GB |
| 2 | 15 | ~12.2 GB |
| unset / ≥3 (default) | 16 | 12977 MiB (current) |

¹ Interpolated from the measured 16f=12977 / 13f≈10527 endpoints (~0.82 GB/frame). Attention
is O(tokens²) so the true curve is slightly convex; treat as a guide, measure on GPU.

## Risks

- **Detail seam (verdict: moderate, mitigated by N≥1).** The base pass saw all 16 frames
  (full temporal attention → structure/coherence is locked before the refine). The refine is a
  light 3-step polish at `denoising_strength`. Dropping guide tokens only shrinks the *refine's*
  attention context; the low-frequency join is already set by the base. Additional slack for the
  plain path: the refine's guide tokens currently get **default (sequential) positions**, not
  their true seam-start RoPE, so at N=large they aren't providing precise seam context in the
  first place — N=0 loses less than the head-placement mental model implies. N=1/2 restores
  local context cheaply. Sweep N=0/1/2 vs the byte-cap on seam-stress prompts.
- **Stitch/decode fixed frame count: OK.** The crop at 9439 uses `video_target_frame_count`
  (absolute T), not `total − K`, so feeding fewer trailing frames still yields exactly T. Decode
  consumes the post-crop T frames. No index assumes T+K.
- **Audio pin / save-for-next-chain: HANDLED (unpack → slice video → repack).** Audio is packed
  as **extra channels** whose count = `ceil(audio_values / (W·H·frames))`
  (`pack_ltxav_audio_and_video_latents`, sd:4964) — frame-count-keyed. The audio latent itself is
  a segment-global timeline (`{16, audio_length, 8}`) **independent of the video guide frames**, so
  the knob unpacks it, slices only the video frames, and repacks the *same* audio (with the
  audio-pinned mask rebuilt for the sliced video). `audio_length` is unchanged, so both the audio
  decode (:9384) and the crop (:9439) behave as before. `chain_base_latent` is pre-upscale → the
  refine subset can't affect the next-segment chain seed. **Verified by construction; needs a GPU
  A/B (audio identical, seam) to confirm.**

## Root cause of the first no-op (corrected)

The v1 knob required `audio_length==0` and empty `hires_denoise_mask`, so it no-op'd on the
deployed continuation — which is **audio-driven** (talking-head), giving `audio_length>0` and a
non-empty audio-pinned mask. (`hires_video_positions` is *empty* on that path: the plain-cont
refine's `audio_fixed` branch, sd:8458, sets only the mask, never positions — positions come
solely from relip/end-image.) v2 handles the audio+mask directly, so it now fires on the
audio-driven keyframe-append continuation. The remaining no-op paths are relip-two-stage and
FLF2V end-image (non-empty `hires_video_positions`/`hires_video_reference`), a separate follow-up.

## Status

**IMPLEMENTED (v2)** env-gated for the LTXAV keyframe-append continuation refine including the
audio-driven case (byte-identical when unset or on a guarded-out path; the specific failing guard
is logged). Relip-two-stage / FLF2V end-image refines still no-op (frame-coupled positions).
**Not yet compiled/GPU-validated** (no build per active-bench constraint) — needs `make` + a
seg-2 A/B at N=0/1/2 vs default to confirm it fires (refine T=13 → ~10.5 GB) and holds the seam +
audio.
