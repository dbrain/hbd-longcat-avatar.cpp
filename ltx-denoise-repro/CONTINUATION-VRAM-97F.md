# LTX-2.3 1080p 97f multi-segment continuation — VRAM reclaim analysis

Goal: get the 960×544 → x2 → 1920×1088, 97-frame, multi-segment CONTINUATION under
the **11776 MiB** cap. Works today but peaks ~13 GB.

Measured (LONGCAT_VRAM_BREAKDOWN=1, 2-seg 97f, all fixes deployed):

| phase | driver_used | compute_buf | resident | runtime(VAE) | "overhead"* |
|-------|-------------|-------------|----------|--------------|-------------|
| SEG-2 REFINE (true max) | 12977 | 3369 | 5471 (DiT) | — (VAE evicted) | 3719 |
| SEG-2 DECODE | 12771 | 6532 (VAE) | 0 (DiT freed) | 1385 | 4854 |
| single 1080p REFINE | 11010 | 2767 | 5471 | — | ~2772 |
| single 1080p DECODE | 11164 | 6532 | 0 | 1385 | ~3247 |

*"overhead" = driver_used − (compute_buf + resident + runtime + partial + prefetched):
the CUDA ctx + cuDNN workspace + VMM pool high-water + **any GPU buffer not attributed to
the phase's own breakdown line** (e.g. the DiT's leftover streaming buffers, which land
in the VAE decode line's overhead).

Continuation delta: REFINE compute +602 (keyframe tokens), DECODE overhead +1607.

---

## Mechanism findings (code, cited)

### 1. `trim_pools` DOES reclaim — but the DiT partial/prefetch buffers are NOT in that pool
`ggml_backend_cuda_trim_pools` (`ggml/src/ggml-cuda/ggml-cuda.cu:5835-5857`) genuinely
returns the VMM compute pool to the driver: `pools[d][s].reset()` → `~ggml_cuda_pool_vmm`
→ `cuMemUnmap` + `cuMemAddressFree` (`:563-575`). So the DIFFUSION compute-pool high-water
(the ~3.3 GB sampling scratch) IS unmapped by the DIT_FREE trim. **That part works.**

**But** the DiT param-streaming buffers are separate `ggml_backend_buffer`s that live
OUTSIDE the ggml_cuda_pool, so `trim_pools` never touches them, and
`release_all_gpu_param_residency()` (`src/core/ggml_extend.hpp:4996-5007`) leaks them:

- it frees `runtime_params_buffer` + `resident_runtime_params_buffer` (real cudaFree), but
- `restore_partial_params()` (`:3571-3608`) routes the partial buffer through
  `pool_return_partial_buffer()` (`:3202-3220`) which **pushes it into `prefetch_buf_pool_`,
  not cudaFree**;
- it **never touches `prefetched_state_.buf`** (the unconsumed "+1 segment" prefetch) nor
  `cache_buffer`.

`kPrefetchPoolCap = 2` (`:3148`), and the REFINE breakdown shows `partial=209 prefetched=209`,
so ~**209 (prefetched, orphaned) + up to 2×209 (pool) ≈ 420–627 MB** of DiT streaming
buffers squat on the board through the seg-2 VAE decode. `gpu_footprint_bytes()`
(`:4943-4951`) explicitly counts `prefetched_state_.buf + prefetch_buf_pool_ + cache_buffer`
as real GPU residency — confirming these are live VRAM, not accounting noise. They are only
freed in the destructor's `free_pipelining_backend()` (`:4787` → `:2925-2936`), never at the
DIT_FREE seam. **This is exactly the prompt's "freed-to-POOL but not returned to the driver"
hypothesis, confirmed.**

### 2. REFINE peak 12977 — compute + attributed streaming, little reclaimable
At REFINE the breakdown line is the DiT's own, so `partial`/`prefetched` are *attributed*
(not hidden in overhead). resident 5471 is load-bearing (fix 4f01aaa, keep). The +602
compute is the keyframe tokens; the only lossless lever on it is FFN/attention tiling
(below). A `LTXAV_PRE_SAMPLE_POOL_TRIM` already trims the DIFFUSION pool before the refine
(`src/stable-diffusion.cpp:9210-9215`), so the +947 refine overhead vs single is cuDNN /
context growth, not a reclaimable buffer. REFINE is the harder ceiling.

### 3. Audio path forces run_ax even for no-audio
`make_ltxav_empty_audio_latent` (`src/stable-diffusion.cpp:5256`, called at `:6365/:6368`)
gives every LTXAV render `audio_length>0` → `run_ax=true` (`src/model/diffusion/ltxv.hpp:1452`)
→ audio self/cross-attn + a2v/v2a built every step + audio VAE reloaded at decode
(`:9344` → `reload_audio_vae_model` `:1779`, ~353 MB). The audio VAE is freed again *before*
the video-decode peak (`:3316-3322`), so disabling audio mainly relieves the SAMPLING peaks,
not the 6532 video-decode peak.

---

## Ranked reclaimable chunks + fixes

| # | chunk | MB | where | chain-only? |
|---|-------|----|-------|-------------|
| 1 | DiT `prefetch_buf_pool_` + orphaned `prefetched_state_.buf` (+`cache_buffer`) held through decode | **~420–627** | DECODE overhead | no (both paths leak it; freeing is byte-identical) |
| 2 | audio DiT compute + ~353 MB audio-VAE reload, no-audio only | ~353 + compute | REFINE + audio-decode | no (only fires when audio genuinely absent) |
| 3 | FFN intermediate cap 4096→2048 | ~128 | REFINE/base compute | no (lossless, all paths) |

### FIX 1 (top — do this): free the DiT streaming buffers before the seg-2 decode
The cleanest edit is to fold the prefetch-pool + prefetched-state teardown into
`release_all_gpu_param_residency()` (`src/core/ggml_extend.hpp:4996`), which is the
"drop ALL GPU residency, keep host home" method and is exactly what the DIT_FREE path calls.
Add, after the existing `restore_resident_params()` / `restore_all_params()`:

```cpp
// Streaming scratch the restore paths route back into the pool (partial ->
// pool_return_partial_buffer) or never touch (prefetched_state_.buf). trim_pools
// can't reclaim these (they are backend buffers, not the ggml_cuda_pool). The next
// segment's kick_off_prefetch / pool_alloc_ctx_tensors re-creates them lazily, so
// freeing here is safe and re-offload-clean.
if (prefetched_state_.buf != nullptr) {
    ggml_backend_buffer_free(prefetched_state_.buf);
    prefetched_state_.buf = nullptr;
}
if (prefetched_state_.ctx != nullptr) { ggml_free(prefetched_state_.ctx); prefetched_state_.ctx = nullptr; }
prefetched_state_.pairs.clear();
prefetched_state_.event_recorded = false;
free_prefetch_buffer_pool();   // :3222
```

The DIT_FREE block already trims the DIFFUSION pool right after
(`src/stable-diffusion.cpp:9330`), so the freed buffers leave the board immediately.
**Expected: ~420–627 MB off the DECODE peak → 12771 → ~12150–12350.** Byte-identical
(buffers rebuilt on the next segment's first offload). Also trims between segments (the
chain reclaim funnels through the same release paths), lowering the inter-segment valley.

### FIX 2 (belt-and-suspenders, chain-only if VAE≠DIFFUSION backend): VAE-pool trim before decode
In the DIT_FREE block (`src/stable-diffusion.cpp:9323-9335`), alongside the existing
`ggml_backend_cuda_trim_pools(backend_for(DIFFUSION))`, add
`ggml_backend_cuda_trim_pools(backend_for(SDBackendModule::VAE))`. On the single-GPU default
recipe DIFFUSION and VAE resolve to the same cached `cuda0` backend
(`src/core/ggml_extend_backend.cpp:601` caches by name), so this is a redundant no-op there;
if the recipe ever splits VAE onto its own backend it reclaims the seg-1-decode + seg-2
reference-encode VMM high-water the DIFFUSION-only trim misses. Harmless, byte-identical.

### FIX 3 (REFINE lever, lossless): tighten the FFN tile
`LONGCAT_FFN_TILE_TOKENS` is already 4096; the tiled FFN (`src/model/common/block.hpp:308-324`)
is lossless (same math, concatenated). At ~26k tokens the full [inner_dim, tokens]
intermediate is ~1.6 GB; tiled@4096 ≈ 256 MB, @2048 ≈ 128 MB → **~128 MB off the REFINE/base
compute**. Small, but it is the only lossless lever on the 12977 REFINE peak's compute.
(Attention-scratch tiling would give more but is not a drop-in env today.)

### FIX 4 (no-audio renders only): genuinely disable audio
For a render with no audio at all, skip `make_ltxav_empty_audio_latent` and set
`latents.audio_length = 0` (guard at `src/stable-diffusion.cpp:6356-6369`) so `run_ax=false`.
Drops the per-step audio DiT compute and the ~353 MB audio-VAE reload. **The production MV
uses audio, so this does NOT help the real target** — quantified only per the brief.

---

## Bottom line
- **DECODE** (12771): FIX 1 alone → ~12150–12350; still over. FIX 1 + FIX 3 + (no-audio) FIX 4
  → ~11800, at the line. To clear 11776 with margin on the decode, FIX 1 is necessary but a
  further ~400 MB is needed — the largest remaining decode lever is the VAE decode compute
  itself (6532), i.e. `LTX_VAE_TT` temporal tiling (already known ~+24% time for ~−1.4 GB),
  which is the reliable way under-cap if FIX 1+3 aren't enough.
- **REFINE** (12977): no single reclaimable buffer (streaming is attributed, VAE evicted,
  resident is load-bearing). FIX 3 (~128) + attention tiling of the +602 keyframe compute are
  the only lossless levers; otherwise `--max-vram` down one notch is the compute-working-set
  lever that flattens it.
- Highest-confidence, code-proven win = **FIX 1** (the `release_all_gpu_param_residency`
  prefetch-pool + prefetched-state leak). Recommend landing FIX 1 + FIX 2, re-measuring the
  decode breakdown, then deciding whether `LTX_VAE_TT` is needed to finish the decode and
  `--max-vram`/attention-tiling to finish the refine.
