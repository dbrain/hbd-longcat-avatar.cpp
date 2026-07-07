# Refine/decode +1.7 GB overhead reclaim — cuDNN plan-cache high-water

## Symptom (GPU-measured, `LONGCAT_VRAM_BREAKDOWN=1`, 2-seg 97f 1080p continuation)

Uniform ~+1.7 GB of **unnamed** reserve-time high-water in EVERY seg-2 phase vs seg-1,
with byte-identical named buffers:

| phase          | driver_used | compute_buf | resident | partial/prefetched | pool | unnamed |
|----------------|-------------|-------------|----------|--------------------|------|---------|
| seg-1 refine   | 10531       | 2767        | 5471     | 209/209            | 0    | ~1875   |
| seg-2 refine   | 12207       | 2765        | 5471     | 209/209            | 0    | ~3553 (**+1678**) |
| seg-1 decode   | 11009       | 6532 (VAE)  | 0 (rt 1385) | –               | 0    | ~3092   |
| seg-2 decode   | 12747       | 6532 (VAE)  | 0 (rt 1385) | –               | 0    | ~4830 (**+1738**) |

`pool=0` everywhere → NOT the ggml gallocr / prefetch pool. Run peak = seg-2 **decode** = 12747.

## Where the +1.7 GB physically lives

**Persistent cuDNN execution-plan caches**, built during seg-1 and never freed.

- The recipe runs with `GGML_CUDNN_ATTN=1` (+ `GGML_CUDNN_CONV3D=1` on the relip/dub
  continuation, see `run_ltx_relip.sh:40`, `run_ltx_dub.sh:40`; `iter_seg2.sh:20` even
  labels the `CUDNN_OFF` A/B "cuDNN attn scratch scales w/ tokens").
- cuDNN SDPA plans: `ggml/src/ggml-cuda/fattn-cudnn.cu:136` — `static std::unordered_map
  g_plan_cache`, keyed by `(B,H,Lq,Lkv,D,io_half)`. Each entry holds a
  `shared_ptr<fe::graph::Graph>` whose built execution plan reserves cuDNN-backend
  **device memory**. The map is file-scope `static`, guarded by `g_plan_mtx`, and is
  **never cleared** for the process lifetime.
- cuDNN conv3d plans: `ggml/src/ggml-cuda/conv3d-cudnn.cu:261` — `static
  g_conv3d_cache`, same story, for the VAE-decoder 3D convs.
- The per-call *workspace* (`ws.alloc(ctx.pool())`, fattn-cudnn.cu:330 / conv3d-cudnn.cu:556)
  DOES go through the ggml VMM pool and IS reclaimed by the trims — that is not the leak.
  The leak is the cuDNN-**internal** plan memory, which lives **outside** the ggml pool.

### Why it survives the trims and why it's uniform across refine + decode

- `ggml_backend_cuda_trim_pools` (ggml-cuda.cu:5835) unmaps only `cuda_ctx->pools[d][s]`
  (the ggml VMM pool). It cannot touch cuDNN-internal plan memory. All SD modules resolve
  to ONE shared backend object (`init_cached_backend` caches by name, ggml_extend_backend.cpp:601),
  so the trim already covers VAE+DiT+TE — yet the +1.7 GB survives it → confirms it's not pool.
- The breakdown is logged at gallocr **reserve** time (ggml_extend.hpp:2606), *before* the
  graph executes. So a phase's log reflects only plans built in EARLIER phases.
- seg-1 order: base(13f) → refine(13f, logs 1875) → decode (builds the big **VAE conv3d
  plans** + audio/x SDPA plans DURING compute, i.e. AFTER its own reserve log). So seg-1's
  refine and decode logs never include the conv3d plans.
- seg-2 runs entirely AFTER seg-1's decode, so its refine AND decode reserve logs both
  inherit seg-1's decode-built conv3d/SDPA plans (plus seg-2's own 16f base plan). That is
  the uniform ~+1.7 GB, present in seg-2's two phases and absent from seg-1's two phases.

This is the single mechanism that explains all four numbers.

## The fix (env-gated, default byte-identical)

Drop the cuDNN plan caches at the **between-segment boundary**, before seg-2's pipeline —
so seg-2 starts as clean as seg-1 did.

- ggml (submodule): `ggml_cuda_cudnn_sdpa_release_plans()` / `ggml_cuda_cudnn_conv3d_release_plans()`
  clear their respective plan caches under their mutexes (destroying the `fe::graph::Graph`
  shared_ptrs → cuDNN frees the plan device memory). The thread_local handles and the
  per-weight reorder caches (raw cudaMalloc, keyed by persistent weight ptr) are KEPT →
  no leak, no weight re-reorder. Public export `ggml_backend_cuda_release_cudnn_plans()`
  (ggml-cuda.h) calls both; no-op stubs on non-cuDNN builds.
- parent: `release_chain_segment_gpu_residency()` (stable-diffusion.cpp:3338) calls it under
  `LTXAV_CHAIN_CUDNN_RESET`. Nothing is in flight there (all segment compute completed +
  synced), and it fires ONCE per segment boundary — the next segment rebuilds only the plans
  it needs (one-time build, NOT per-step), so the refine/decode is not slowed.

Expected: seg-2 refine → ~10531, seg-2 decode → ~11009 (seg-1 levels) → run peak ~11009,
fits ≤11.5 GB.

**Enable:** add `-e LTXAV_CHAIN_CUDNN_RESET=1` to the continuation run (iter_seg2.sh /
run_ltx_relip.sh). Default off = byte-identical to today.

## Confidence + the one confirming diagnostic

Localization to cuDNN is code-proven (survives the ggml-pool trim, which is the only other
thing `pool=0`/named buffers leave; caches are process-persistent; timing explains the
seg-1-clean / seg-2-taxed asymmetry). The one thing NOT provable from code alone is the
**magnitude** (that these plans total ~1.7 GB rather than, say, VMM fragmentation of a
non-pool buffer). The A/B that pins it, on the next GPU run:

1. Re-run the 2-seg continuation with `LTXAV_CHAIN_CUDNN_RESET=1` + `LONGCAT_VRAM_BREAKDOWN=1`.
   - seg-2 refine/decode overhead drops to seg-1 levels → **confirmed, and this is the fix.**
   - unchanged → it is suspect #2 (below); revert-safe (env off = no-op).
2. Cross-check with the existing `CUDNN_OFF=1` toggle (iter_seg2.sh:20): if the +1.7 GB
   vanishes with cuDNN disabled entirely, cuDNN is the source.

### Top-2 suspects (ranked)
1. **cuDNN plan-cache device memory** (this fix). Strong.
2. CUDA VMM fragmentation / a non-pool backend buffer held across the boundary. Weaker:
   the shared-backend pool trim already reclaims the VMM pool, and `pool=0` shows the
   gallocr/prefetch pools are empty — so a residual would have to be a leaked standalone
   backend buffer, none of which scales with the base-pass frame count the way the plans do.
