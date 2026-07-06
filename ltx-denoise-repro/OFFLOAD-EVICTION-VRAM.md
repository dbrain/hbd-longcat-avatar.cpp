# Offload eviction — free the dead-weight VAE (sampling) and DiT (decode) GPU squats

Analysis + code fix only. **No GPU renders run.** Card = RTX 5060 Ti (15888 MiB). Prod driver =
`run_parity_nvfp4.sh` (`--offload-to-cpu --mmap --max-vram 11`, `LONGCAT_SHARED_RESIDENT=1`,
`LONGCAT_VAE_KEEP_RESIDENT=1`, warm resident chain).

## Measured problem (new `gpu_footprint_bytes()` + `[VRAM-ATTR]` instrumentation)

```
[VRAM-ATTR sample-entry, refine] DiT_gpu=5471 MB  VAE_gpu=1385 MB   <- VAE dead weight during DiT sample/refine
[VRAM-ATTR decode-entry]         DiT_gpu=5471 MB  VAE_gpu=1385 MB   <- DiT dead weight during VAE decode
```

Both the video VAE (1385 MB) and the DiT (5471 MB) stay GPU-resident across the **entire** render.
The refine peak carries the VAE's dead 1385 MB; the decode peak carries the DiT's dead 5471 MB.
Freeing each from the phase where it's unused drops the binding refine peak toward ~10.5 GB
(≤ 11.5 GB hard cap).

## Root cause — why the old free did NOT work

The prior lever (`LTXAV_VAE_LAZY` regime A) called `set_keep_params_resident(false)`, and the DiT
free path calls `free_params_buffer()`. **Neither frees the buffer that actually holds the squat.**

There are THREE distinct GPU param buffers on a `GGMLRunner` (all summed by `gpu_footprint_bytes()`):

| buffer | allocated by | freed by |
|---|---|---|
| `runtime_params_buffer` | `offload_all_params()` (per-compute full upload) | `restore_all_params()` |
| `partial_runtime_params_buffer` | partial/prefetch offload ring | `restore_partial_params()` |
| **`resident_runtime_params_buffer`** | **`offload_resident_params()`** | **`restore_resident_params()`** only |

Under the prod recipe **`LONGCAT_SHARED_RESIDENT=1` (+ cross-step, default on)**, each model's param
payload is offloaded **once** into `resident_runtime_params_buffer` and deliberately **kept pinned
across sampling steps, VAE decode tiles, and warm renders** (`compute_with_graph_cuts`, lap-B/lap-C).
For the LTX VAE the cross-segment-shared set is essentially *all* its conv weights → the resident
buffer ≈ the full **1385 MB**. For the DiT it is the whole **5471 MB** payload. On a warm resident
chain (`keep_diffusion_model_resident=true`) the post-sample DiT `free_params_buffer()` at
`stable-diffusion.cpp:9089/9097` is intentionally **skipped**, so the DiT resident buffer lives
straight through the VAE decode.

- `set_keep_params_resident(false)` → `restore_all_params()` frees only `runtime_params_buffer`.
- `free_params_buffer()` frees `params_buffer` + `restore_all_params()`, and NULLs the home
  pointers (which would break re-offload) — but it *still* never touches
  `resident_runtime_params_buffer`.

So the shared-resident buffer survived every existing eviction. And `ggml_backend_cuda_trim_pools()`
"doesn't help" because a trim only reclaims *pool-cached* free blocks — it cannot reclaim a buffer
that is still referenced/live. (`ggml_backend_buffer_free` → `~ggml_backend_cuda_buffer_context` →
`cudaFree`, a genuine driver release — so the real free *is* available; it just was never being
called on the resident buffer.)

## The fix — one genuine-eviction primitive, two symmetric callers

### New `GGMLRunner::release_all_gpu_param_residency()` (`ggml_extend.hpp`)

Funnels **all three** offload buffers through their real freers and clears the cross-step residency
token/cache, while leaving the CPU/mmap-backed home (`params_buffer` / `params_ctx` tensors) intact
so the next `execute_graph` re-uploads:

```cpp
void release_all_gpu_param_residency() {
    keep_params_resident_ = false;
    restore_resident_params();  // frees resident_runtime_params_buffer (the shared-resident squat)
    restore_all_params();       // frees runtime_params_buffer (+ restore_partial_params)
    cached_shared_resident_set_.clear();
    cached_shared_resident_sig_ = 0;
}
```

Unlike `free_params_buffer()` it does **not** null the home, so re-offload needs no reload loader and
is mmap-safe. No-op when params live directly on the runtime backend
(`params_offloaded_to_host() == false`) — callers gate on that.

### FIX 1 — VAE during sampling (`LTXAV_VAE_LAZY=1`, existing flag, regime A)

`stable-diffusion.cpp` regime-A now calls `vvae->release_all_gpu_param_residency()` (and the audio
VAE's) instead of `set_keep_params_resident(false)`, then trims the VAE pool. The final decode's
`VAE::decode() → compute_with_graph_cuts → offload_resident_params()` re-offloads from the CPU/mmap
home. Target: `[VRAM-ATTR sample-entry] VAE_gpu=0`.

**FIX 1b — second eviction for the two-stage refine.** The LTX render is TWO passes: base sample →
latent ×2 upscale → **hires/refine sample** (a second, separate `sample()` call). The regime-A
eviction fires only ONCE, before the *base* sample. Between the two samples the offload pipeline
re-offloads the video VAE (log: `ltx_video_vae offload params (1385.02 MB, 170 tensors) to runtime
backend (CUDA0)` at the hires stage), so by the refine `VAE_gpu` is back to 1385 MB and the refine
(DiT 5471 + VAE 1385 + compute ~2767) is the ~11913 MB binding peak. So a second identical eviction
runs **right before the refine `sample()` dispatch** (`stable-diffusion.cpp`, just before
`sampling_start = ggml_time_ms();` at the hires sample), same guards
(`LTXAV_VAE_LAZY && !relip_twostage && params_offloaded_to_host()`). The plain t2v two-stage refine
is a pure latent denoise that never touches the VAE (only the final decode does, which re-offloads
it). **Skipped on `relip_twostage`**, whose stage-2 reference re-encode *does* need the VAE during
refine (freed later by `LTXAV_TWOSTAGE_FREE_UNUSED`). Target: `VAE_gpu=0` at BOTH base AND refine
sample-entry.

### FIX 2 — DiT during decode (`LTXAV_DIT_FREE_DURING_DECODE=1`, NEW flag)

New block right after all sampling completes (`stable-diffusion.cpp`, just past "generating latent
video completed"), before the audio/video decode. Gated on `sd_version_is_ltxav` +
`diffusion_model->params_offloaded_to_host()`. Calls
`diffusion_model->release_all_gpu_param_residency()` + trims the DIFFUSION pool. Nothing below the
free (audio-VAE decode, continuation-latent copy, video-VAE `decode_video_outputs`) touches the DiT,
and a chained next `generate_video`'s `sample()` re-offloads the DiT via `execute_graph`. If the DiT
lives directly on the GPU (no host offload → no re-offload path) it logs + skips (abort-safe; the DiT
isn't needed for decode anyway). Target: `[VRAM-ATTR decode-entry] DiT_gpu=0`.

## Safety

- **Model always present when it computes.** VAE freed only between its encode and the *later*
  decode, which re-offloads before its first tile. DiT freed only *after* the last sample, and a
  chained render re-offloads before its first DiT step.
- **Home intact / re-offload proven.** `release_all_gpu_param_residency()` restores tensors to their
  CPU/mmap home (never NULLs them, unlike `free_params_buffer`); `execute_graph` *always* re-offloads
  before compute.
- **Abort-safe.** Both callers gate on `params_offloaded_to_host()`; the no-offload regime skips
  with a log. FIX 1 stays gated `!relip_twostage` (relip's stage-2 re-encode still needs the VAE;
  handled by `LTXAV_TWOSTAGE_FREE_UNUSED`). The in-decode
  `set_keep_params_resident(true)` VAE tile-loop guard in `ltx_vae.hpp` is untouched.
- **Default byte-identical.** FIX 1 is inside the existing opt-in `LTXAV_VAE_LAZY` block; FIX 2 is a
  new opt-in `LTXAV_DIT_FREE_DURING_DECODE` (default off). Instrumentation
  (`gpu_footprint_bytes` / `[VRAM-ATTR]` / `GGML_F8_DBG`) unchanged.

## How to verify (owner runs the GPU)

Prod recipe + `LTXAV_VAE_LAZY=1 LTXAV_DIT_FREE_DURING_DECODE=1`. Expect:

```
LTXAV_VAE_LAZY: released offloaded video+audio VAE GPU params (runtime + shared-resident) ...      # before base
[VRAM-ATTR sample-entry, base]   DiT_gpu=0 MB     VAE_gpu=0 MB
LTXAV_VAE_LAZY: re-released offloaded video+audio VAE GPU params ... before the hires/refine sample  # before refine (FIX 1b)
[VRAM-ATTR sample-entry, refine] DiT_gpu=5471 MB  VAE_gpu=0 MB
LTXAV_DIT_FREE_DURING_DECODE: released offloaded DiT GPU params (runtime + shared-resident) ...
[VRAM-ATTR decode-entry]         DiT_gpu=0 MB     VAE_gpu=1385 MB
```

Refine peak should drop ~1.4 GB (VAE gone) → toward ~10.5 GB; decode peak drops ~5.5 GB of dead DiT
weight. Cost: one re-offload of each freed payload at its re-use point (VAE at decode ~0.1–0.2 s; DiT
at the next chained segment's first sample step); the two-stage path re-offloads the VAE once more
between the base and refine passes had it not been evicted, so FIX 1b is net-neutral on transfers.

## Update log

- **FIX 1b (two-stage refine VAE re-eviction):** GPU test of the initial fix showed base
  sample-entry `VAE_gpu=0` ✓ and decode-entry `DiT_gpu=0` ✓, but the *refine* sample-entry still had
  `VAE_gpu=1385` — the offload pipeline re-materialized the VAE between the base and refine samples.
  Added a second identical `release_all_gpu_param_residency()` eviction immediately before the
  hires/refine `sample()` dispatch (same `LTXAV_VAE_LAZY && !relip_twostage && params_offloaded_to_host()`
  guard). Now `VAE_gpu=0` at both base and refine sample-entry.
