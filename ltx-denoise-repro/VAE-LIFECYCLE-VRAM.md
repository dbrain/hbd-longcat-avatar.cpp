# LTXAV VAE lifecycle — drop the ~1.7 GB VAE squat during DiT sample+refine

## Problem (measured)

During the DiT sampling + latent-upscale refine stage of an LTX-2.3 t2v render, the
**video VAE (~1385 MB)** and **audio VAE (~353 MB)** sit on the GPU but are **not used** again
until the much-later decode stage. They squat ~1.7 GB through the refine, whose reserve is the
**binding VRAM peak**:

```
[VRAM] ltxav reserve: driver_used=11913 MB
ltx_video_vae offload params (1385.02 MB, 170 tensors) to runtime backend (CUDA0)
```

That pushes the refine peak just over an 11.5 GB target. `LONGCAT_VAE_KEEP_RESIDENT=0` does **not**
help: that flag only governs per-tile re-offload *inside* VAE decode (`ltx_vae.hpp` tiled loop).

## Prod recipe: `--offload-to-cpu --mmap --max-vram 7`

This is the load-bearing detail (`run_parity_nvfp4.sh`). Under `--offload-to-cpu` the VAE params'
**home** lives on the CPU `params_backend`; each VAE compute UPLOADS them to a GPU
`runtime_params_buffer` via `offload_all_params()` (that's the "offload params 1385 MB to runtime
backend (CUDA0)" log). That GPU copy is what squats. So the params are **not** left file-backed on
the GPU under `--mmap` — there is a real 1.7 GB of GPU VRAM to reclaim.

## Change

`src/stable-diffusion.cpp`, in `generate_video_ex`, right after `prepare_video_generation_embeds`
finishes all VAE *encoding* (~:8554, just past the existing `WAN_VAE_FREE_DURING_DIT` block) and
before the sample loop. New env flag **`LTXAV_VAE_LAZY=1`** (opt-in, default off). It handles the
two residency regimes separately:

### Regime (A) — `--offload-to-cpu` (our prod, incl. `--mmap`)

```cpp
first_stage_model->set_keep_params_resident(false);   // -> restore_all_params()
if (audio_vae_model) audio_vae_model->set_keep_params_resident(false);
```

`restore_all_params()` frees the GPU `runtime_params_buffer` (the 1385 / 353 MB copies) and remaps
the tensors back to their CPU/mmap-backed home. The home `params_buffer` survives untouched. At
decode, `VAE::decode()` -> `_compute()` -> `execute_graph()` calls `offload_all_params()` again,
re-uploading the params from that home — **the same mechanism that loaded them at setup**. No reload
loader, no disk re-read, mmap-safe.

New public accessor `GGMLRunner::params_offloaded_to_host()` (`ggml_extend.hpp`, returns
`params_backend != runtime_backend`) selects this regime.

### Regime (B) — params directly on the GPU (no offload)

```cpp
first_stage_model->free_params_buffer();   // + audio VAE
// reload_first_stage_model() / reload_audio_vae_model() reload from disk before decode
```

Here `restore_all_params()` is a no-op (nothing offloaded), so the buffers are freed and reloaded
from disk before decode via `resident_reload_loader` (captured for every LTXAV no-mmap ctx, :1335).
The existing unconditional decode reloads (`reload_first_stage_model()` at :9128,
`reload_audio_vae_model()` at :9027) re-materialize them (~0.1 s each). Gated on the loader being
present, so the free never happens without a proven reload path. If neither regime applies (e.g.
`--mmap` *without* `--offload-to-cpu` → params file-backed on GPU, no captured loader) the flag
**skips** and logs — abort-safe, VAE stays present.

`ltx_vae.hpp` decode is unchanged: its own `set_keep_params_resident(true)` tile-loop guard (and the
`(false)` + `free_compute_buffer()` after) still runs, untouched.

## Why this is safe

- **Reload/re-offload proven-present before the free**:
  - Regime A: the CPU/mmap home `params_buffer` is still allocated and `execute_graph` *always*
    calls `offload_all_params()` before compute — the re-upload cannot be missing.
  - Regime B: gated on `resident_reload_loader`; skipped if absent.
- **No post-sample VAE encode is starved**: the only VAE *encode* after the base sample is the relip
  stage-2 reference re-encode inside `apply_ltxv_refine_image_conditioning`, which runs **only** on
  `latents.relip_twostage`. The flag is gated `!relip_twostage`, leaving relip to FIX-3
  (`LTXAV_TWOSTAGE_FREE_UNUSED`), which frees at the correct later point (:8907).
- **No per-tile re-offload OOM**: the in-decode `set_keep_params_resident(true)` guard that protects
  the spatial-tile loop is untouched. Regime A sets keep=false *before* sampling; the decode's own
  keep=true re-establishes tile residency.
- **Decode reloads stay no-ops in regime A**: params_buffer is not freed, so
  `get_params_buffer_size() != 0` → `reload_*` return true immediately.
- **Idempotent / harmless**: `set_keep_params_resident(false)` on a VAE that isn't currently
  offload-resident is a no-op; `free_params_buffer()` null-checks its buffer.
- **Silent video (audio_length == 0)**: audio VAE released and never re-materialized (its decode
  block doesn't run) — VRAM stays reclaimed, no correctness impact.

## Expected effect

- Refine/sample peak drops by ~**1.7 GB** (video ~1385 MB + audio ~353 MB) → `driver_used ≈ 11913`
  should fall to roughly **~10.2 GB**, under the 11.5 GB target. The pre-sample pool trim
  (`LTXAV_PRE_SAMPLE_POOL_TRIM`, before both samples) reclaims the freed GPU buffer to the driver.
- Cost (regime A): one extra `offload_all_params()` re-upload at decode (~0.1–0.2 s each for the
  two VAEs) — negligible vs the sample+refine wall.

## Risk to the decode path

Low. Regime A leaves the home params intact and the decode re-offloads exactly as an un-freed run
would. Regime B aborts before `decode_video_outputs` if a reload fails (existing guard at :9128)
rather than decoding garbage. The VAE is fully present whenever decode runs.

## How to verify (owner runs the GPU)

Intended path: `DIT=nvfp4-imatrix-dev050.gguf`, `--offload-to-cpu --mmap --max-vram 7`, t2v (no
init image), `LTXAV_VAE_LAZY=1`. Expect in the log:

```
LTXAV_VAE_LAZY: released offloaded video+audio VAE GPU params before DiT sample+refine; re-offload from host at decode
...
[VRAM] ltxav reserve: driver_used=<~10200> MB      # ~1.7 GB lower at the refine peak
...
ltx_video_vae offload params (1385.02 MB, 170 tensors) to runtime backend (CUDA0)   # ONLY at decode now
```

Default behavior (flag unset) is byte-for-byte unchanged.
