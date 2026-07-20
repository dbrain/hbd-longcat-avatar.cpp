# REBUILD ON UPSTREAM — handoff for the *proper* merge (do later, fresh)

> This supersedes the "we are the base, pull their stuff in" approach that produced the current
> `upstream-merge-2026-07-19` branch. That branch WORKS and builds (a usable superset today), but it made
> the fork the center of gravity — the opposite of the intent. Read this before touching the next merge.

## North star (owner's stated intent — do not drift from it again)

sd.cpp and ggml are **upstream projects = the parent = source of truth.** The fork exists ONLY to add three
things. The fork should get **smaller over time** as upstream improves. If upstream did these three well, we
wouldn't touch it.

**The three value buckets (the ONLY things worth re-applying):**
1. **API / GPU-sharing layer** — loads/unloads cleanly so multiple engines can time-share a GPU (warm
   residency, `/v1/admin/load|unload`, the async worker-isolated server). This is the "works the way I want" layer.
2. **Models / workflows upstream doesn't have** — avatar, audio-driven, video-continuation/chain workflows.
3. **Performance / VRAM tuning for the specific devices** (3060 / 5060 Ti) — NVFP4/FP8 GEMM, VAE kernels,
   graph-cut VRAM/offload, SA3.

Anything in the current fork that does NOT serve one of those three is **incidental divergence to shed**, not
value to preserve.

**Method:** clean upstream branch → re-apply ONLY the three buckets on top → regression-test each →
keep the delta minimal. **NOT** "fork is base, merge upstream in."

## The unlock: most of the current fork does NOT get re-ported

On a clean upstream base, **everything upstream already ships comes free.** Do NOT re-port any of it:
- All models upstream has: krea2, boogu, hunyuan-video, lingbot, minit2i, **sefi**, **qwen-image-layered**, flux2, etc.
- Features: **AnimateDiff, ADetailer, PuLID, streaming-convert**, imatrix.

⚠️ This session's "pull-in" (commits `aa351a4`/`7fa96b5`/`3c5fe62`/`0fd28e7`/`369baec`) hand-ported all of the
above ONTO fork conventions. On the rebuild that work is **discarded** — it's upstream-native there. The pull-in
kept the *current* heavy fork alive and usable NOW; the rebuild makes ~most of it unnecessary. Both are
legitimate; they serve different timelines. Don't feel obligated to carry the pull-in forward.

## The fork's REAL additive value — concrete inventory (mine these from the current branch)

**11 fork-only new files** (`git diff --name-status ea4e566 HEAD` → `A`), tagged by bucket:
- Bucket 2 (models/workflows): `longcat_avatar.hpp`, `longcat_audio.hpp`, `wav2vec2.hpp`, `wan_s2v.hpp`,
  `wan_shotstream.hpp`, `wan_sla.hpp`, `vace.hpp`, `infinitetalk.hpp`.
  `nava.hpp` — **SHELVED (weights deleted 2026-07-05); likely DROP** unless resurrected.
- Bucket 3 (perf infra): `core/layer_registry.{cpp,h}` — the graph-cut layer partitioning; **tied to the
  ModelManager decision below** (it's the fork's replacement machinery).

**Bucket-3 perf/VRAM tuning woven INTO modified upstream files** (cherry-pick the hunks, not the whole file):
- `core/ggml_extend.hpp` (+4k lines): NVFP4/FP8 cuBLASLt GEMM, SA3 attention, masked-flash tweaks.
- `model/vae/wan_vae.hpp`: `WAN_VAE_RMS_KERNEL`/`_CF`, `feat_cache` causal streaming, `dec_dim_override`, F16 path.
- `model/vae/ltx_vae.hpp`, LTX spatial VAE decode + eviction recipes.
- `runtime/denoiser.hpp`, the sigma/scheduler tuning.
- graph-cut marks (`mark_graph_cut`) scattered across every DiT (`flux.hpp`, `qwen_image.hpp`, `ltxv.hpp`, …).

**Bucket-1 + Bucket-2 orchestration woven into `stable-diffusion.cpp` (+13k lines — the biggest file):**
generate_video / LTX continuation+chains / audio mux / warm-server residency / load-unload / the whole video
engine. This is the hardest to re-apply because it's entangled with the ModelManager decision AND with genuine
new features. Most of it is bucket-1/bucket-2 value, but it's not a clean patch — budget real time here.

## THE load-bearing decision — answer this FIRST (it sizes everything else)

The fork **DELETED 9 upstream files** — upstream's entire multi-GPU/backend architecture:
`model_manager.{cpp,h}`, `backend_fit.{cpp,h}`, `layer_split_partition.{cpp,h}`, `weight_manager.h`,
`runtime/imatrix.{cpp,h}` — and replaced them with `layer_registry` + the `(backend, params_backend)` runner
convention. **`ModelManager` is in 23 upstream files but only 1 fork file** — this single replacement is why
every upstream model arrives needing hand-porting. It is THE tax.

On the rebuild, decide before anything else:

- **(A) Replace again** — if graph-cut VRAM/offload is genuinely load-bearing on the 3060/5060Ti and can't sit
  on upstream's backend layer. Keep the deletion, but FIRST re-measure: upstream's backend/offload/split may have
  improved since ea4e566, shrinking what graph-cut must still do. This keeps the fork heavy.
- **(B) Shim** — keep upstream's `ModelManager` and express graph-cut VRAM/offload as an *additive* layer so
  upstream models load **unmodified**. This is the true thin overlay and the only path where the fork shrinks
  over time. More upfront architecture work; pays back on every future sync.

Sub-decision: **imatrix** — upstream's `runtime/imatrix.{cpp,h}` (param-threaded) vs the fork's global
`g_convert_imatrix`. Pick upstream's unless there's a concrete reason not to (fewer fork touch-points).

To decide A vs B, spike: *is `layer_registry`/graph-cut a genuine architectural replacement of ModelManager, or
did it just happen to be implemented as one because ModelManager was in the way?* That fact decides it.

## Rebuild steps

1. `git fetch upstream` — branch clean from the **current** upstream tip (NOT `ea4e566`; it will have moved,
   and may already have more of the fork's models/features).
2. Answer the ModelManager A/B question (spike layer_registry ↔ ModelManager coupling).
3. Re-apply **bucket 2** (the ~10 keeper new files) — adapt to whichever runner convention (A/B) you chose.
4. Re-apply **bucket 3** perf hunks — cherry-pick onto upstream's now-current files (these are where conflicts
   live; apply hunk-by-hunk, GPU-verify each).
5. Re-apply **bucket 1** API/GPU-sharing (mostly koblem-side + engine load/unload paths).
6. **Regression-test each step**: per-model smoke render, a full load→unload→load cycle (the GPU-sharing
   invariant), VRAM ceiling on 3060 AND 5060 Ti, and determinism where it matters. Eye-test page: `:8077`.

## ggml — identical treatment (it's upstream too)

Fork ggml value (cherry-pick these patches onto clean ggml tip, don't carry the merge):
- SA3: forbid atomic delta-gemm reductions; use active-device SM count for the scheduler.
- FP8/NVFP4 GEMM (+ determinism probes, env-gated).
- CUDA/LTX determinism fixes.
Current ggml merge `531a2c72` (fork `55a4f61b` ⨝ upstream v0.17 `9be31331`) is clean + in-sync + builds, but is
a MERGE not a patch stack — for the rebuild, distill the named patches, don't re-merge.

## Current branch = a MINE, not a push target

`upstream-merge-2026-07-19` (superproject) + ggml `531a2c72`: a working superset snapshot. Use it to (a) read the
full value inventory, (b) copy bucket-2/3 hunks FROM. **Nothing is pushed.** Do NOT push it as the base — it's
what you distill *from*, not the foundation to build *on*.

## The cadence lesson (so this never recurs)

The pain came from a months-old base + one big-bang catch-up. After the rebuild lands thin, **track `upstream`
continuously** — small frequent merges, port new fork-relevant deltas as they land. The `upstream` remote is
already configured on both repos.
