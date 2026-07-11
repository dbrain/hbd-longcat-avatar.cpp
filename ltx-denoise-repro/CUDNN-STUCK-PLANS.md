# Stuck cuDNN plan/workspace memory — ~1.7GB/phase, cross-project (2026-07-07)

**Ask for a fresh reviewer (Codex):** find a way to **reclaim the ~1.7GB of persistent cuDNN
device memory** that accumulates once cuDNN attention/conv3d plans are built and is never returned
to the driver. This is not LTX-specific — **any warm CUDA worker running `GGML_CUDNN_ATTN` and/or
`GGML_CUDNN_CONV3D` across multiple models or pipeline phases inherits it** (flux2, wan, ltx,
avatar — all reuse a persistent CUDA child). Solving it is a cross-project VRAM win.

## The observation (GPU-measured, LTX-2.3 1080p continuation, `LONGCAT_VRAM_BREAKDOWN=1`)
Two identical-workload phases, ~1.7GB apart, and the delta tracks "were cuDNN plans built yet":
```
                       driver_used   compute_buf   named params      unnamed overhead
seg-1 refine (13f):      10531         2767         resident 5471          ~1875   ← clean
seg-2 refine (13f):      12207         2765         resident 5471          ~3553   ← +1678
seg-1 decode:            11009         6532 (VAE)   runtime 1385           ~3092   ← clean
seg-2 decode:            12747         6532 (VAE)   runtime 1385           ~4830   ← +1738
```
`pool=0` on every line → it is NOT the ggml gallocr/VMM pool (`ggml_backend_cuda_trim_pools`
already runs and can't see it). The VRAM breakdown logs at `ggml_gallocr_reserve` time (before
compute), so a phase's line reflects plans built in *earlier* phases. seg-1's refine/decode run
before the big SDPA/conv3d plans exist → clean; every seg-2 phase runs after → +1.7GB. The uniform
+1678 (SDPA, refine) / +1738 (conv3d, decode) = the two cuDNN plan families.

## Where it lives (code)
- SDPA plans: `ggml/src/ggml-cuda/fattn-cudnn.cu` — file-scope `static ... g_plan_cache` of
  `fe::graph::Graph` objects (cudnn-frontend). Built plans reserve cuDNN-**internal** device memory.
- conv3d plans: `ggml/src/ggml-cuda/conv3d-cudnn.cu` — file-scope `static ... g_conv3d_cache`, same pattern.
- Per-call *workspace* goes through `ctx.pool()` and IS reclaimed — that's not the leak. The leak is
  the persistent memory tied to the built plans / the cuDNN handle.

## What we TRIED (all failed — this is the key input for the reviewer)
1. **Clear the plan caches** (`ggml_backend_cuda_release_cudnn_plans()`, engine `85c61afa`): destroys
   the `fe::graph::Graph` objects in both caches at the segment boundary, keeps the thread_local
   cudnn handle. Wired + CONFIRMED FIRING (`release_chain_segment_gpu_residency`). **No effect** —
   seg-2 decode still 12747. ⇒ **destroying the graph objects does NOT free the device memory.**
   Strong signal the memory is held by the **cuDNN handle's reserved workspace / internal allocator**,
   not the frontend graph objects.
2. **Disable conv3d cuDNN** (`GGML_CUDNN_CONV3D=0`): **CRASHES** the render at ~149s (the non-cuDNN
   conv3d fallback is broken / OOMs differently). Not viable as-is.
3. **ggml pool trims** (LTXAV_CHAIN_POOL_TRIM, LTXAV_END_RENDER_RECLAIM, pre-sample trims): reclaim
   the ggml pool but not this (it's outside the pool, `pool=0`).
4. **MAXV / shared-resident knobs**: orthogonal (that's the DiT weight residency, 5471, separate).

## Directions for the reviewer to investigate
- **Is it the cudnn handle's persistent workspace?** cudnn-frontend/cuDNN may retain an internal
  workspace or plan-descriptor memory tied to `cudnnHandle_t` that outlives the `fe::graph::Graph`.
  Would `cudnnDestroy(handle)` + recreate at the segment boundary reclaim it? Cost = handle re-init
  (once per segment, ~ms) — acceptable if it returns 1.7GB. Check what the thread_local handle in
  fattn-cudnn.cu / conv3d-cudnn.cu holds and whether destroying it frees device memory.
- **cudnn-frontend plan deallocation:** does `fe::graph::Graph` destruction actually call the cuDNN
  APIs that free the execution plan's device memory, or just drop the C++ wrapper? May need explicit
  `cudnnBackendDestroyDescriptor` / plan teardown the frontend doesn't do on destruction.
- **Workspace limit:** can the plans be built with a bounded/zero persistent workspace
  (`fe::graph::Graph::set_workspace_limit` or the deterministic/`CUDNN_ATTN` workspace knobs) so they
  don't reserve 1.7GB in the first place? Trade a little speed for the memory.
- **Why does CONV3D=0 crash?** Fixing the non-cuDNN conv3d fallback is a separate but useful escape
  hatch (it removes the conv3d plans cleanly).
- **cudnnGetProperty / memory introspection:** confirm the 1.7GB is cuDNN-internal (not fragmentation)
  before/after handle destroy, to prove the mechanism.

## Reproduce
Engine at LTX_REF (see kobbler compose), `LONGCAT_VRAM_BREAKDOWN=1`, `GGML_CUDNN_ATTN=1
GGML_CUDNN_CONV3D=1`. Run a 2-seg 97f 1080p continuation (scratchpad/renders/cont2_1080.json). Compare
seg-1 vs seg-2 refine/decode `[VRAM] ltxav reserve` lines — the +1.7GB is the target. `LTXAV_CHAIN_CUDNN_RESET=1`
toggles the (ineffective) cache-clear. Cross-check: `CUDNN_OFF=1` A/B in docker/*/iter_seg2.sh.
Related engine commits: ggml `85c61afa` (the cache-clear), parent `63d1f773`.

## Why it matters beyond LTX
The persistent CUDA worker pattern (worker-isolation: one warm child serves many renders/models) means
this 1.7GB is paid by every co-resident service on the 16GB card. Reclaiming it at model/phase
boundaries frees headroom for flux2 ↔ wan ↔ ltx ↔ avatar co-residency and for larger single renders.
