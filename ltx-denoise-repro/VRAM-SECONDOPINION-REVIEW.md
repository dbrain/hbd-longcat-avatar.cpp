# 1080p continuation VRAM — independent second opinion (cold review)

Code-only review, no builds/GPU. Repo `/home/dbrain/dev/longcat-avatar-ltxdenoise`,
branch `ltx-denoise-workflow` @ 734998e. Verdicts below are from reading the actual
paths, not the brief's summary.

---

## Q1 — Is the frame-count mechanism right, or is there continuation-specific overhead?

**Verdict: the mechanism is right. Continuation is a longer clip, NOT extra tokens.
The single-121f diagnostic is a VALID equivalence test. Confidence: HIGH.**

Evidence:
- The continuation prepends K anchor latent frames into the denoised tensor `vx`. It does
  **not** add reference tokens. The only path that *concatenates* extra attention tokens is
  the relip two-stage `video_reference_tensor`/`vx_ref` block —
  `src/model/diffusion/ltxv.hpp:2236-2243` (`ref_token_count = vx_ref->ne[0]*ne[1]*ne[2]`)
  and the combined count at `ltxv.hpp:2320` (`video_token_count = vx grid + ref_token_count`).
  For a plain keyframe continuation `video_reference_tensor.empty()` → `ref_token_count==0` →
  the sequence is exactly the 16-latent-frame target grid. Same graph *shape* as a plain
  121-pixel-frame render.
- Corroboration that this render is the plain (non-relip) path: `LTXAV_VAE_LAZY` frees the
  VAE **before** the refine and is gated `!relip_twostage`
  (`src/stable-diffusion.cpp:9262-9272`). The brief's resident=5471 excludes the 1385 MB
  VAE, so VAE_LAZY fired → this is *not* relip_twostage → no ref-token concat.

Caveat to check in the log (cheap): confirm the refine `sample()` at
`stable-diffusion.cpp:9275` is NOT going through `apply_ltxv_refine_image_conditioning`'s
`video_reference` branch (`stable-diffusion.cpp:8346+`). If it ever did, the 121f diagnostic
would *under*-count. Given VAE_LAZY fired, it isn't — but eyeball the
`ltxav modulation collapse: N video tokens -> U unique` DEBUG line: N should equal
60×34×16 ≈ 32.6k, not more.

---

## Q2 — Where does the refine `overhead` 3719 physically live? Is it irreducible?

**Verdict: the team is PARTLY WRONG to call it a "genuine keyframe working set."
It is THREE things, and one of them (~600–800 MB) is the offload machinery's speed
buffers, which are reducible at the refine — but NOT by the pre-refine free they shipped.
Confidence: MEDIUM-HIGH (exact split needs the log; the components are certain).**

First, a correctness note on the number itself: **"overhead 3719" is a DERIVED subtraction**
(`driver_used − compute_buf − resident`). The breakdown logger
(`ggml_extend.hpp:2595-2615`) never prints an "overhead" field — it prints
`driver_used`, `compute_buf`, and **separately** `partial / prefetched / pool / runtime /
resident`. So the team already has the attribution in the log and should read those fields
rather than treat 3719 as one opaque lump.

The 3719 decomposes into:

1. **Fixed CUDA context** (cuBLAS/cuDNN handles + kernel image): ~400–700 MB.
   Frame-independent. Irreducible. This is the bulk of the *single* 1873 baseline.

2. **DiT param-streaming double-buffer + pool** — `partial_runtime_params_buffer`
   (~209 MB, the current streamed block) + `prefetched_state_.buf` (~209 MB, the +1
   lookahead) + `prefetch_buf_pool_` (up to `kPrefetchPoolCap=2` × ~209 MB).
   `ggml_extend.hpp:2225, 2285, 3148`. **~600–800 MB, frame-INDEPENDENT** (block weights are
   the same size at any frame count). The refine *streams the non-resident blocks*
   (`offload_partial_params` skips `resident_param_set`, `ggml_extend.hpp:2989`), so this is
   live at the refine peak.
   → **This is the part the team mis-labeled "irreducible keyframe working set."** It is the
   offload machinery, not the model activations.
   → **Why FIX-1/734998e didn't move the refine peak:** `free_streaming_scratch_buffers()`
   frees these *once, before* the refine (`stable-diffusion.cpp:9223-9234`), but the refine's
   own `sample()` immediately **rebuilds** identical buffers as it streams. Freeing-once can't
   lower a peak that the consumer recreates. To cut it you must *prevent the rebuild* for the
   refine (cap the pool to 0 + disable the +1 prefetch for the refine call), not free it once.

3. **cuDNN SDPA pool high-water — the FRAME-SCALING term.** `fattn-cudnn.cu:330-342`: the
   SDPA `workspace` **and** the `O` scratch (`N*H*Lq*D` halfs) are allocated from
   `ctx.pool()` = the ggml_cuda VMM pool, **outside gallocr** (so NOT in `compute_buf`).
   Both scale with `Lq` (token/frame count). At 16 vs 13 frames the O-scratch alone is
   ~N·H·Lq·D·2 B (order ~250–300 MB, +~23% for the 3 extra frames). The VMM pool keeps the
   high-water for reuse across 48 blocks × 3 steps, so it sits live at the peak.
   → **This is where the +1846 "overhead scales with frames" actually comes from** — the
   team attributed it to flash-attn/cuDNN "working buffers" loosely; it's specifically the
   pool-resident SDPA O + workspace. It is genuinely irreducible **for a given token count**
   (in active use every attention call; trimming mid-refine just forces re-alloc), so the
   team's "irreducible" claim is correct *for component 3 only*. It shrinks only by shrinking
   tokens (K / resolution) — i.e. it tracks compute, exactly as observed.

Net: of the 3719, roughly context (~500, irreducible) + streaming buffers (~600–800,
**reducible at small speed cost**) + cuDNN pool (~rest, frame-scaling, irreducible). The
reducible ~600–800 MB is the lever the pre-refine free *aimed at but missed* (freed too
early). One deeper, riskier win in component 3: `fattn-cudnn.cu:337-342` writes O to a pool
scratch then copies out; writing straight to the gallocr dst would drop one O-sized buffer
(~250 MB) from the pool high-water — kernel-level change, not recommended first.

---

## Q3 — Partial resident pin (top-N hottest, stream the rest)? Does a knob exist?

**Verdict: NO size/bytes cap exists today, but it is a ~15–20 line add, and it is the
single best lever — strictly dominant over the binary free-DiT. The `min_segs` knob that
DOES exist is useless here. Confidence: HIGH.**

- `compute_shared_resident_set` (`ggml_extend.hpp:3065-3088`) pins **every** param with
  `seg_count >= min_segs` — there is **no bytes budget and no top-N cap**. It accumulates
  `shared` and `shared_bytes` but never truncates.
- The existing `LONGCAT_SHARED_RESIDENT_MIN_SEGS` knob (`ggml_extend.hpp:3008-3015`) does
  **not** shrink the 5471: the pinned 306 params are the *global* payload (modulation /
  embedders / connector, per the comment at `ggml_extend.hpp:2985`) read by **every** base
  cut-segment. Raising the threshold to any value ≤ #segments keeps them all. Confirmed the
  brief's "MAXV doesn't shrink resident" for the same structural reason: the set is derived
  from the **base** plan (`shared_resident_base_plan_enabled`, `ggml_extend.hpp:3017-3033`),
  independent of the MAXV-controlled merge.
- **The add:** in `compute_shared_resident_set`, when `LONGCAT_SHARED_RESIDENT_MAX_MB` is set,
  sort the qualifying tensors by hotness (`seg_count` desc, then nbytes) and truncate at the
  byte budget; the dropped params simply aren't in `resident_param_set`, so
  `offload_partial_params` streams them (`ggml_extend.hpp:2989, filter_out_resident:3092`).
  This is a **continuous** knob from full-pin (5471, fast) to zero-pin (free-DiT, slow).
  cap=0 *is* the free-DiT-in-refine lever — one mechanism covers both Q3 and lever #2.
- **Scoping:** the base sample (8 steps) wants full pin; only the refine (3 steps) needs the
  cap. The refine is a distinct `sample()` at `stable-diffusion.cpp:9275`, so set a
  member/env immediately before it and reset after (the set is re-derived + cached per plan
  signature, `ggml_extend.hpp:4328`). ~20 lines total including the scope gate.

---

## Q4 — Smallest change that clears 11776, and the missed lever

**Ranked:**

1. **[RECOMMENDED] Byte-capped resident pin for the refine only** (Q3 add). Pin ~4000 MB
   → −1471 → ~11.5 GB, clears with margin. Only the coldest ~1.5 GB of the global payload
   streams during 3 refine steps = a *fraction* of a full DiT re-stream per step → small
   speed hit, **far** cheaper than free-DiT's 2.5×. No quality/continuity/resolution loss,
   byte-identical output. Single localized change. **This is the answer to the brief's Q3
   and its Q4 "smallest total change" simultaneously.**

2. **Free-DiT-in-refine** = the same knob at cap=0. Fits trivially (~7–8 GB refine). Bigger
   speed hit (~+22 min/8-seg per the brief) but dead simple and the safest on continuity.
   Fallback if the partial-pin's coldest-stream still doesn't fit or you want max headroom.

3. **K=2 (15 latent) + refine streaming-buffer suppression.** K=2 ≈ −817 → ~12.16 GB; add
   pool-cap-0 + no-prefetch *during the refine* (component 2) for another ~500–700 MB →
   ~11.5 GB. More moving parts and K carries a real seam-continuity risk (the one lever with
   a *quality* cost). Only if you refuse any resident change.

**Is there a free lunch?** Nearly: **suppressing the streaming double-buffer + pool *for the
refine* (kPrefetchPoolCap→0, disable the +1 prefetch)** is quality/continuity/byte-identical
and reclaims ~500–700 MB — its *only* cost is a little allocator churn / lost H2D overlap
across 3 steps. It is NOT literally free and NOT enough alone (12977→~12.4 GB), but it's the
cheapest real byte and it's the thing the shipped pre-refine free was *reaching for and
missed by freeing too early*. Worth wiring correctly regardless of which primary lever wins.

**The lever the team MISSED / mis-framed:**
- The partial (byte-capped) pin — they asked about it (Q3) but concluded no knob exists and
  didn't note it's a trivial add that dominates their two "real trade" levers.
- The correct framing of the streaming buffers: freeing them once pre-refine can't lower a
  peak the refine rebuilds; the fix is a during-refine *policy* (cap + no-prefetch), not a
  boundary free.
- The "overhead scales with frames" is specifically `fattn-cudnn.cu` pool O-scratch +
  workspace (`fattn-cudnn.cu:330-342`), not a diffuse "keyframe working set" — so the
  frame-scaling part genuinely tracks tokens and only K/resolution move it, but a big
  frame-INDEPENDENT chunk (context + streaming buffers) was wrongly folded into "irreducible."

**Bottom line:** it is NOT a pure pick-your-poison. The resident/speed axis has a *continuous*
knob the code doesn't yet expose; adding a ~20-line byte-cap gives a partial pin that clears
11776 with a small, bounded speed cost and zero quality/continuity/resolution loss — strictly
better than K-reduction (quality risk) or full free-DiT (large speed cost).
