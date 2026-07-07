# Seg-2 coarse cut + 0 shared-resident — cut-planner / shared-resident heuristic analysis

Scope: the **cut planner** (`src/core/ggml_graph_cut.cpp`) and the **shared-resident
derivation** (`src/core/ggml_extend.hpp`). The *why the continuation+hires graph
topology changes* is the sibling agent's domain (`generate_video_ex`); this doc proves
what the planner/heuristic do with that graph and how to make the DiT stay pinned.

All line numbers are on branch `ltx-denoise-workflow`.

---

## 1. The exact path that emits the log lines

Both discriminator lines come from the **non-streaming** graph-cut path
(`compute()` → `resolve_graph_cut_plan` → `compute_with_graph_cuts`), i.e.
`stream_layers_enabled == false`, `LONGCAT_SHARED_RESIDENT=1`:

- `merged %zu segments -> %zu segments` — `ggml_graph_cut.cpp:830` inside
  `apply_max_vram_budget()`.
- `shared-resident set: %zu params (%.0f MB) read by >=%d of %zu segments` —
  `ggml_extend.hpp:3042` inside `compute_shared_resident_set()`, called once from
  `compute_with_graph_cuts` at `ggml_extend.hpp:4211`.

Critically, `compute_shared_resident_set(gf, plan)` at 4211 is passed **`plan` = the
RESOLVED / MERGED plan** returned by `resolve_graph_cut_plan` (4862) →
`resolve_plan` → `apply_max_vram_budget`. It is NOT the base (unmerged) plan. So the
"read by >=N segments" tally is computed over the **post-merge** segment list
(50 for seg-1, 3 for seg-2).

## 2. What the heuristic actually counts (ggml_extend.hpp:3023-3046)

`compute_shared_resident_set` walks each merged segment's `runtime_param_tensors`
(`ggml_graph_cut.cpp:404-419`), which returns exactly the segment's `INPUT_PARAM`
refs (params-ctx-membership leaves, `build_segment` line 199) minus any with a null
buffer (409). It increments `seg_count[t]` once per segment that reads `t`, then keeps
`t` iff `seg_count[t] >= shared_resident_min_segments()` (default 2, `3033/3037`).

So a param is "shared-resident" **only if its param leaf is an `INPUT_PARAM` of ≥2
of the MERGED segments.** Two consequences fall straight out of the code:

- **Merging monotonically destroys the signal.** A weight read by base-blocks 5,6,7
  is "shared" across 3 base segments, but once 5,6,7 are merged into one segment it is
  `INPUT_PARAM` to a *single* merged segment → `seg_count==1` → dropped. The finer the
  cut, the more sharing is visible; the coarser the cut, the less.
- **`>=2 of N` is brittle at small N.** At the degenerate end (a plan merged to N=1)
  every param lands in one segment → `seg_count==1` for everything → 0 shared by
  construction, even though the whole model is obviously "shared" across steps.

## 3. Why seg-1 = 306/50 but seg-2 = 0/3 — the unified root

Seg-1 (hires, **no** continuation): the budget merge is a no-op (50 base → 50 merged;
no `merged … -> …` line). The global payload (adaLN modulation + video-embeddings
connector + embedders, 306 tensors / 5471 MB) is applied per-block, so each of those
leaves is an `INPUT_PARAM` of many of the 50 fine segments → `seg_count` ≫ 2 → 306
kept → DiT pinned resident → base ~9 s/it.

Seg-2 (continuation keyframe-append, T 13→16 + rebuilt video_positions, hires pending):
`apply_max_vram_budget` merges **50 → 3**. That merge is greedy on
`graph_cut_segment_vram_bytes = compute_buffer + input_param_bytes +
input_previous_cut_bytes + output_bytes` (`ggml_graph_cut.cpp:78-83, 782, 802`). For 50
base blocks to collapse into 3 segments under the same 7168 MB budget that left seg-1
un-merged, **the per-base-segment `input_param_bytes` must have collapsed** — you cannot
pack ~17 blocks that each independently carry the 5471 MB payload into one 7168 MB
segment. The payload weight leaves are no longer presented as per-block `INPUT_PARAM`
leaves in seg-2's graph.

That single fact produces BOTH symptoms:

1. **Coarse merge (50→3):** with per-segment `input_param_bytes` collapsed, many blocks
   fit under budget → greedy packs them (`ggml_graph_cut.cpp:784-813`).
2. **0 shared-resident:** the payload is no longer an `INPUT_PARAM` shared across ≥2
   merged segments (it is either localized to one segment or no longer a graph leaf at
   all) → `compute_shared_resident_set` returns 0 → nothing pinned → the DiT streams
   cold every step (~4× → 36 s/it) → child SIGKILL.

They are two faces of one graph change. This is confirmed by the coordinator's A/B:
disabling every eviction/reclaim path (`LTXAV_DIT_FREE_DURING_DECODE=0`,
`LTXAV_NO_CHAIN_GPU_RECLAIM=1`) leaves seg-2 at "0 params" — because the DiT is never
*evicted*; it is simply never *derived as shared*, so it is never *pinned*, so the
segloop streams it. `MAXV=5` giving "0 of 4" is the same thing at a slightly finer
merge: the payload still isn't `INPUT_PARAM`-shared across the merged segments.

**Note on residency identity (rules out a red herring):** `offload_resident_params`
(`ggml_extend.hpp:3619-3627`) swaps `buffer/data/extra` onto the *original* params-ctx
tensor and `restore_resident_params` (3653-3661) swaps them back with a non-null
buffer. Pointer identity and `params_tensor_set_` membership are preserved across
segments. So seg-1's residency does NOT corrupt seg-2's classification — the collapse
is purely in seg-2's own graph topology feeding the planner.

**Trigger boundary:** *why* the seg-2 continuation graph stops presenting the payload as
per-block `INPUT_PARAM` leaves (rebuilt `video_positions` / keyframe-append localizing
or precomputing the modulation) is graph construction — the sibling agent's
`generate_video_ex` analysis. The continuation code itself (`stable-diffusion.cpp:6884`,
`apply_ltxav_video_guide_by_keyframe_index`) mutates only `init_latent`/`denoise_mask`,
not weights, so the topology change is in how the hires+continuation DiT forward is
assembled, not in the weights.

## 4. Proposed fix — decouple "keep the DiT resident" from the >=2-of-N reuse heuristic

The heuristic infers residency from **cross-merged-segment reuse**, which is exactly the
signal coarse merging erases — a self-defeating coupling. Three layered fixes; ship
(A)+(B), (C) is a cheap guard.

**(A) Derive the shared set from the BASE plan, not the merged plan.**
In `compute_with_graph_cuts` (4211), compute the shared set over
`graph_cut_plan_cache_.graph_cut_plan` (the un-merged base plan already cached by
`resolve_plan` at `ggml_graph_cut.cpp:863-864`, and already reused for
`annotate_residency` at `ggml_extend.hpp:4434-4436`) instead of the resolved `plan`.
The base plan has maximal granularity, so any weight applied in ≥2 base cut-segments is
detected as shared regardless of how coarsely the budget merges for execution. This
restores seg-1-style detection whenever the payload still spans base segments. Both
plans reference the same `gf` and the same param tensors, so the pinned set is bound
identically; only the *counting granularity* changes. Low-risk, principled.

**(B) Force-resident the DiT on chain/continuation segments (guaranteed floor).**
(A) still fails if seg-2's construction localizes the payload to a *single base
segment* (then even the base plan shows `seg_count==1`). So add a topology-independent
floor: carry the shared-resident set forward across chain segments. The tensors are
pointer-stable params-ctx objects (§3 note), so cache the *previous* segment's resident
set on the runner and, when the fresh derivation yields a set that is empty (or much
smaller in bytes) while `total_params_bytes_` is large, reuse/union the prior set rather
than un-pinning. Concretely: keep `cached_shared_resident_set_` sticky across a chain
(do not let a degenerate re-derivation zero it), gated so a genuine model/plan change
(param pointer set changes) still refreshes it. This makes seg-2 pin the same
306/5471 MB seg-1 proved fits (GPU stays ~11 GB, host fine), keeping base at ~9 s/it.

**(C) Loud diagnostic / fail-fast instead of silent cold-stream+SIGKILL.**
In `compute_shared_resident_set`, when `shared_resident_active()` and the model is large
(`total_params_bytes_ >= ring_min_model_bytes()`) but the returned set is empty, emit a
`LOG_WARN` ("shared-resident derivation degenerate: 0 params kept for an N-GB model;
DiT will cold-stream") and, ideally, trip (B)'s fallback. Also worth logging, in
`apply_max_vram_budget`, when per-segment `input_param_bytes` collapses vs the base plan
(a > … ratio) — that is the fingerprint of the topology change and turns a 20-minute
silent death into an obvious one line.

**(Optional D) Harden the threshold for small N.** Clamp so that at small merged N the
rule doesn't self-zero: e.g. treat any param whose bytes exceed a fraction of
`total_params_bytes_` as resident even at `seg_count==1`, or lower the effective
min-segments floor when the plan is coarse. Secondary to (A)/(B).

---

## 4b. IMPLEMENTED (branch ltx-denoise-workflow)

Confirmed by live VRAM-breakdown: SEG-1 `resident=5471MB compute_buf=850MB` (9s/it) vs
SEG-2 `resident=0MB compute_buf=5682MB` (36s/it, dies) — merged-plan derivation
self-zeroing. Fixes applied, all in `src/core/ggml_extend.hpp`:

- **Fix A** — `compute_with_graph_cuts` now derives `cached_shared_resident_set_` from
  the BASE cached plan `graph_cut_plan_cache_.graph_cut_plan` (guarded by `available &&
  has_cuts && segments.size()>1 && plan_matches_graph(gf,·)`) instead of the resolved
  MERGED `plan`, and keys the cross-step cache on the derivation plan's signature.
  Gated by `shared_resident_base_plan_enabled()` (`LTXAV_SHARED_RESIDENT_BASE_PLAN`,
  default ON; `=0` restores the legacy merged-plan derivation for A/B). Risk: for a
  single render that legitimately merges, the base plan could pin a slightly larger set;
  in a standard transformer the extra "adjacent-block-shared" params don't exist so it
  resolves to the same ~306 global payload — and the env gate lets it be turned off.

- **Fix B** — safety floor: new members `last_shared_resident_set_` /
  `last_shared_resident_bytes_` hold the last HEALTHY set for the loaded model (survive
  residency teardown, cleared in `free_params_ctx()` on model unload). When a fresh
  derivation collapses (`fresh_bytes*2 < last_bytes`) for a large offloaded model
  (`params_backend!=runtime_backend && total_params_bytes_ >= ring_min_model_bytes()`),
  the prior segment's pointer-stable set is carried forward (filtered to tensors still in
  `params_tensor_set_` — guards a model reload / address reuse). Risk: inert on healthy
  renders (only fires when the set would otherwise be < half the prior good one), so
  single renders are unaffected; worst case it pins the known-good ~5471 MB the machine
  already proved fits.

- **Fix C** — `LOG_WARN` when the derivation is degenerate (`fresh_bytes*8 <
  total_params_bytes_`) for a large offloaded model, and a second `LOG_WARN` when Fix B
  carries a set forward — a silent 4×-slow cold-stream + SIGKILL becomes one log line.

Non-chain / single-render path is unchanged: the whole block is gated on
`shared_resident_active() && plan.segments.size()>1`; with no merge the base plan equals
the merged plan (identical set); Fix B/C are inert unless the set is degenerate.

## 4c. FOLLOW-UP (commit 2) — Fix A insufficient, Fix B wasn't firing

Live validation of commit 1: SEG-1 `306 params (5471 MB)` resident, 22→10 s/it, GOOD.
SEG-2 with Fix A active derived over the **50-segment BASE plan** yet STILL got
`0 params read by >=2 of 50 segments` → Fix C warned, but the DiT stayed un-pinned
(`resident=0`, 36 s/it, died). So Fix A alone can't save seg-2 (the continuation
keyframe-append graph presents the DiT weights as read-by-exactly-one cut-segment even
in the fine base plan), and the carry-forward (Fix B) is the necessary path — but it
did not fire. Two robustness bugs fixed:

- **Carry filter used `params_tensor_set_`** (`compute_with_graph_cuts`), which
  `free_params_buffer()` transiently clears (`ggml_extend.hpp:4798`). Now the carry is
  validated against a live walk of the `params_ctx` tensor structs — those keep stable
  addresses across every between-segment teardown (`release_chain_segment_gpu_residency`
  → `release_streaming_residency`; `LTXAV_DIT_FREE_DURING_DECODE` →
  `release_all_gpu_param_residency`), only a real model reload frees them (which also
  clears `last_shared_resident_set_` via `free_params_ctx`). `generate_video_chain`
  (`stable-diffusion.cpp:10010`) is a single-process loop over the persistent
  `diffusion_model`, so `last_shared_resident_set_` genuinely survives seg-0→seg-1.

- **Defensive `params_tensor_set_` rebuild** in `resolve_graph_cut_plan` before the plan
  is built: if the set is ever empty at plan-build time, EVERY DiT weight is
  misclassified `INPUT_EXTERNAL` → 0 param bytes → both seg-2 symptoms (0 shared + coarse
  merge). Rebuilding from live `params_ctx` (idempotent; no-op when populated → single
  render byte-identical) fixes Fix A directly in that case.

- **Fix C** now reports the prior-healthy-set size + live-params count, so a future
  regression says whether the carry *source* or the *filter* was the gap.

Expected seg-2 log now: `shared-resident carry-forward: prior healthy set 306 params
(5471 MB), N live in params_ctx -> carrying 306 forward …` and the reserve line shows
`resident=5471` / ~10 s/it like seg-1.

## 5. File:line index

- `ggml_extend.hpp:4211` — shared set computed over the **merged** `plan` (root of the
  granularity coupling). Fix target (A).
- `ggml_extend.hpp:3023-3046` — `compute_shared_resident_set`; `>=min_segs` rule at
  3033/3037; log at 3042. Fix targets (C)/(D).
- `ggml_extend.hpp:2984-2991` — `shared_resident_min_segments()` (default 2).
- `ggml_extend.hpp:4434-4436` — base plan already available at this scope (for A/B).
- `ggml_extend.hpp:3619-3627, 3653-3661` — residency swap preserves tensor identity
  (rules out residency-drop as the cause).
- `ggml_graph_cut.cpp:764-827` — greedy budget merge; sizing at `78-83, 782, 802`.
- `ggml_graph_cut.cpp:404-419` — `runtime_param_tensors` (INPUT_PARAM + non-null-buffer
  filter feeding the tally).
- `ggml_graph_cut.cpp:199, 255-256` — INPUT_PARAM classification via
  `params_tensor_set_` membership; only INPUT_PARAM contributes `input_param_bytes`.
- `ggml_graph_cut.cpp:830` / `860-889` — merge log; base+budgeted plan caching.
