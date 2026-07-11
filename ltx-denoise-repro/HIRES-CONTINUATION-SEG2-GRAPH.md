# LTX-2.3 hires × continuation — seg-2 BASE graph construction analysis

Repo: `longcat-avatar-ltxdenoise` @ `ltx-denoise-workflow`
File: `src/stable-diffusion.cpp` (`generate_video_ex` ~:8385, `generate_video_chain` ~:9581,
continuation append ~:5104/:6813, hires block ~:9021).
Code analysis only — no build, no GPU.

**Scope split:** the offload/streaming-budget bug (`observed_max_effective_budget_` in
`ggml_extend.hpp`, the graph-cut planner in `ggml_graph_cut.cpp`, shared-resident heuristic) is
the *second agent's* lane and is already root-caused + fixed in `CONTINUATION-1080P-HIRES-CRASH.md`
(commit `c9cb8c9`, budget reset in `release_streaming_residency`/`release_all_gpu_param_residency`).
**This doc covers the GRAPH-CONSTRUCTION side that FEEDS that cut** — what `generate_video_ex`
builds for seg-2's base sample, and which piece of the construction/orchestration triggers the
cold re-offload that the mis-cut then punishes.

---

## What the seg-2 base graph is actually built from

The base sample is `sd->sample(diffusion_model, …)` at **:8970**, run over
`x_t = latents.init_latent` (**:8785**) with `T = init_latent.shape()[2]` (**:8783**) and spatial
`W,H = request.width/height / vae_scale` (**:8781-8782**). Key facts:

1. **Base geometry is `request`, NOT `hires_request`.** `hires_request` is a *copy* made at
   **:8649** and is only used AFTER the base sample, inside the upscale/refine block (**:9021-9250**).
   The base graph is 960×544 × T. The spatial 2× to 1920×1088 happens in
   `upscale_ltx_spatial_video_latent` (**:9023**) and the refine `sample()` (**:9224**), both *after*
   base step 1 — so **nothing in the construction reserves or sizes the base graph for the upscaled
   1920×1088 working set.** (Rules out coordinator Q2's "base plans for the upscaled set".)

2. **The ONLY structural difference between seg-1 and seg-2's base graph is the continuation
   keyframe-append.** For an in-memory chain, `prepare_video_generation_latents` takes the
   `cont_latent` branch (**:6813**) and calls `apply_ltxav_video_guide_by_keyframe_index`
   (**:6884 → :5104**), which:
   - `init_latent = concat(init_latent, guide, dim2)` (**:5122**) → **T grows target→target+K**
     (13→16 latent frames for K=3);
   - concats the frozen guide mask (**:5125**) and **rebuilds `video_positions` over T+K**
     (**:5128**, `build_ltxv_video_positions`, keyframe_frame_idx=0).
   `x_t` (**:8785**) therefore carries K extra latent frames on seg-2. The appended guide tokens are
   real tokens the DiT attends over (frozen via mask, cropped only *after* sampling via
   `video_conditioning_frame_count`). Seg-1 has no continuation → no append → T=13.

3. **The guide geometry matches the base grid (no shape mismatch).** When hires is on,
   `chain_base_latent = final_latent` captures the **pre-upscale** base latent for the next segment
   (**:9016-9019**, comment: "the continuation latent handed back … must be the BASE (pre-upscale)
   latent"). So seg-2's guide arrives at 960×544 and the shape checks at :5113-:5115 pass identically
   whether hires is on or off. (Rules out coordinator Q1's "shape differs when hires enabled".)

4. **Cached text conditioning is inert to graph geometry.** "avatar: reusing cached text
   conditioning" (**:7839**) is a prompt-keyed reuse of the umT5 *text embeds* only
   (**:7834-:7843**); it carries no latent geometry and is reused identically on seg-1 hires and
   seg-2. (Rules out coordinator Q3.)

### Construction verdict: the append is +K frames, ~1.2–1.5×, NOT 4×

13→16 latent frames is +23% tokens; LTX full-3D self-attention is O(N²), so the base compute buffer
grows ~1.5× at most. **That is proven harmless on its own** — the 720p flat chain runs the SAME
append (its seg-2 base is actually a *larger* 1280×704 grid than the hires 960×544 base) through all
8 segments. So **no construction-side working-set inflation explains the 4× (36 vs 9 s/it).** The 4×
is a *streaming* penalty, not a bigger graph.

---

## What in MY lane actually triggers the 4× (the orchestration, not the tensor shapes)

The 4× appears because seg-2's base DiT is **cold — fully host-offloaded — at plan time**, so the cut
streams all ~5471 MB from host per merged segment (34.6 s/it ≈ 4×), and the cold re-offload upload
buffer stacked on the fat streamed segments overruns the board. Seg-1's base is warm-resident and
self-credited, so it plans one resident graph (~9 s/it). **What drops the DiT residency between the
segments lives in `generate_video_ex`/`generate_video_chain` — my lane:**

- **`LTXAV_DIT_FREE_DURING_DECODE` (:9285-:9297).** After seg-1's samples, this does
  `diffusion_model->release_all_gpu_param_residency()` + `ggml_backend_cuda_trim_pools(DIFFUSION)` —
  frees the 5471 MB resident DiT and returns the pool high-water to the OS *before seg-1's decode*.
- **`release_chain_segment_gpu_residency()` between segments (:10217-:10220 → :3338).** For every
  non-final segment on the warm chain it calls, per runner, `release_streaming_residency()` +
  `free_compute_buffer()` + `free_cache_ctx_and_buffer()` (**:3343-:3345**) — dropping the DiT's
  shared-resident payload and restoring params to host.

Either one leaves seg-2's base sample to **cold re-offload the DiT from host and re-grow the trimmed
pool** — then hand a cold graph to the cut. That cold plan is where the second agent's stale-budget
bug bites (`observed_max_effective_budget_` latched high while seg-1 was resident, never reset on
these two release paths → seg-2 plans against a too-large budget → coarse "merged 50 → 3 segments"
streamed cut → overrun). The append's +K frames is the small extra nudge that tips the already
knife-edge tight-`max_vram` hires cut over the board.

### Why hires is the discriminator (and 720p is not)

Hires forces the render into the tight `max_vram=7168` regime (baked MAXV=7 recipe, commit
`c7f019e`) so the *later* 1920×1088 refine fits. That tight cap makes seg-2's cold-re-offloaded base
plan sit right at the board edge, so the cold upload + streamed segments + the +K append overrun.
720p-flat runs a looser budget where the same cold re-offload + append still fit in one/loose cut,
so all 8 segments survive. **It is the interaction (cold-DiT-re-offload × tight-hires-cut × +K
append), not any single construction term.**

---

## Proposed fix (construction/orchestration lane — composes with the committed budget-reset)

**Keep the DiT warm-resident across the chain seam so seg-2's base is planned exactly like seg-1's
(self-credited, single resident graph) — no cold re-offload for the mis-cut to punish.**

`generate_video_ex` already knows it is mid-chain: the chain passes `want_latent = (seg+1 < n_chain)`
(**:10036**) and forwards `final_latent_out != nullptr` only on non-final segments (**:10037-:10040**).

1. **Gate `LTXAV_DIT_FREE_DURING_DECODE` on the final segment only** — add
   `&& final_latent_out == nullptr` to the condition at **:9285**. Freeing/trimming the DiT at decode
   is pure waste on a segment whose successor re-materializes it 34.6 s later; the free is a genuine
   VRAM win only on the last segment (its own VAE-decode peak, nothing following).
2. **Skip the DiT reclaim inside `release_chain_segment_gpu_residency()` when a successor follows** —
   e.g. don't `reclaim(diffusion_model)`/`reclaim(high_noise_diffusion_model)` at **:3347-:3348** on
   the mid-chain call (or add an "keep DiT warm" arg driven by `seg+1 < n_chain` at the call site
   **:10217**). Still reclaim VAE/audio/compute/cache to hold the +1.4 GB chain-anchor line.

This is the construction-side counterpart to the already-committed budget reset (fix B). Ship both:
**A keeps the DiT warm so the seg-2 base graph is built + planned like seg-1's; B hardens every
release→re-offload path so a cold plan can never inherit a stale budget.** They are independent and
compose. Env A/B first (no rebuild): `LTXAV_DIT_FREE_DURING_DECODE=0` and
`LTXAV_NO_CHAIN_GPU_RECLAIM=1` — either flip should restore seg-2 base to ~9 s/it and let the 1080p
chain finish.
