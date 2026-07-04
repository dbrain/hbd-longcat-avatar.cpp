# C++ implementation plan — 3 missing features for the Denoise-AI LTX-2.3 port

Read-only code investigation of `longcat-avatar.cpp` (sd.cpp/ggml lineage). Scope = the
three capabilities `_CONTEXT.md` lists as MISSING: per-phase LoRA strength, base-pass custom
sigmas, and NAG. All `file:line` refs are against the worktree
`/home/dbrain/dev/longcat-avatar-ltxdenoise` as read on 2026-07-05. **No builds/renders were
run** (CPU-only design pass).

Headline findings up front:

- **FEATURE 2 (base `--sigmas`) is ALREADY IMPLEMENTED.** The flag exists, is parsed, is wired
  to `sample_params.custom_sigmas`, and is consumed by the base sampler for LTX. It is "done,
  needs a GPU smoke-test", not a code task.
- **FEATURE 1 (per-phase LoRA)** — the runtime machinery to change LoRA strength between the
  base and hires passes already exists (`apply_loras` is re-callable and diffs against
  `curr_lora_state`). Only a small plumb-through of a second LoRA set + one extra `apply_loras`
  call between passes is needed. **The two-pre-folded-gguf alternative does NOT work through the
  current `--hires` path** (hires reuses the same in-memory DiT; the only swap primitive reloads
  the whole ~11 GB DiT from disk). Verdict: do the flag, not the two-gguf swap, for the coupled
  base→upscale→refine pipeline.
- **FEATURE 3 (NAG)** is genuinely hard (L) and, per the reverse-engineered workflow, only ever
  runs on the **droppable S1/S3 preview taps** — no load-bearing stage uses it. Recommend
  **skip for v1**.

---

## FEATURE 1 — Per-phase / per-pass LoRA strength

### Current behavior

LoRA is a single global set applied once, before any sampling, and reused verbatim for the
hires refine pass.

- LoRAs are parsed from the prompt (`<lora:name:strength>`) in
  `SDGenerationParams::extract_and_remove_lora`, regex at `examples/common/common.cpp:1996`.
- The parsed map is flattened into `lora_vec` (`examples/common/common.cpp:2358-2364`) and handed
  to the API as `params.loras` / `params.lora_count` (`examples/common/common.cpp:2397-2398`).
  Struct is `sd_lora_t { bool is_high_noise; float multiplier; const char* path; }`
  (`include/stable-diffusion.h:317-321`).
- In `generate_video`, LoRAs are applied exactly once, up front:
  `sd_ctx->sd->apply_loras(sd_vid_gen_params->loras, sd_vid_gen_params->lora_count);`
  at `src/stable-diffusion.cpp:8390` — **before** the base sample and the hires refine, which
  are two calls to the *same* `sd_ctx->sd->diffusion_model`:
  - base sample: `src/stable-diffusion.cpp:8619` (`final_latent = sd_ctx->sd->sample(diffusion_model, …)`)
  - hires refine sample: `src/stable-diffusion.cpp:~8825` (`final_latent = sd_ctx->sd->sample(diffusion_model, …)` inside the `if (latent_upscale_enabled)` block).
- The hires refine reuses `sd_vid_gen_params->sample_params.guidance` and the same LoRA state;
  there is **no** second `apply_loras` between the passes and **no** per-phase LoRA field on
  `sd_hires_params_t` (`include/stable-diffusion.h:337-349` — it has sigmas/steps/denoise/scale
  but no `loras`).

### Can `apply_loras` be re-invoked mid-generation with a different state? — YES.

`apply_loras` (`src/stable-diffusion.cpp:2168`) dispatches on `apply_lora_immediately`:

- **`apply_loras_immediately`** (`:1970`): folds the LoRA delta into the weights, but it computes
  a *diff* against `curr_lora_state` (`:316`): `lora_state_diff = new - curr`, applies only the
  delta, then sets `curr_lora_state = new` (`:2011`). So calling it again with
  `{distill: 0.8}` after `{distill: 0.65}` applies a `+0.15` delta correctly. **Cost: re-folds
  changed tensors each call** — for an nvfp4 base this is a dequant→add→requant over the affected
  weights (seconds), paid once at the base→hires boundary. Acceptable but not free.
- **`apply_loras_at_runtime`** (`:2014`): attaches a `MultiLoraAdapter` weight-adapter; the
  forward routes linears through `forward_with_lora`. Re-calling with a new state reuses the
  already-loaded `diffusion_lora_models`, just overwriting `lora_model->multiplier`
  (`:2074-2081`) or attaching a fresh adapter. **Cost: ~free** (a pointer/scalar swap; no re-fold,
  no reload). This is the mode auto-selected for quantized bases (`:563-586`:
  `have_quantized_weight || streaming_constrained → apply_lora_immediately = false`), i.e. our
  nvfp4 prod path already lands here. `--lora-apply-mode auto` (the default) therefore gives us
  the cheap swap for free.

Net: the engine can already switch LoRA strength between the base and hires passes. What is
missing is (a) a place to *carry* a second LoRA set and (b) the extra `apply_loras` call.

### The two-pre-folded-gguf alternative — evaluate honestly

Can the current `--hires` path swap the *diffusion model* (base gguf folded @0.65 → refine gguf
folded @0.8) between passes? **No, not within one `generate_video` call.**

- Both passes call `sd_ctx->sd->diffusion_model` (`:8619` and `:~8825`); `request.hires.model_path`
  is the **latent upsampler** (`SD_HIRES_UPSCALER_MODEL`, consumed by
  `upscale_ltx_spatial_video_latent` at `:8672`), *not* a DiT override. There is no hook to point
  the refine `sample()` at a different DiT.
- A swap primitive *does* exist — `swap_diffusion_model` (`src/stable-diffusion.cpp:1837`,
  C-API `sd_ctx_swap_diffusion_model` `:10237`, used by the server variant-swap in
  `examples/server/async_jobs.cpp:575`) — but it **re-inits a loader and reloads every DiT param
  tensor from disk** (`:1861-1903`). For a 22B nvfp4 gguf that is an ~11 GB disk read *per render*
  at the base→refine boundary. That is far worse than the runtime-LoRA multiplier swap, and it
  fights the warm-resident worker model.

Conclusion: **two-folded-ggufs is not viable through `--hires` for the coupled pipeline.** It
would only make sense if we split the workflow into two *separate* CLI invocations (base gen →
`LTX_SAVE_LATENTS`, then a second process doing upscale+refine off `LTX_LOAD_LATENTS` with the
0.8-folded gguf). That split is possible today with **zero code** (the latent bank harness exists
at `:8604 LTX_LOAD_LATENTS` / `LTX_SAVE_LATENTS`), but it double-loads the DiT, breaks native
audio/continuation bookkeeping that lives inside the single `generate_video`, and can't share the
upscaler step. So for v1 **prefer Feature 1's flag** (one process, cheap runtime-LoRA reweight).

### Precise change (smallest viable)

Mirror `--hires-sigmas` with a `--hires-lora` set, and re-apply LoRAs before the refine sample.

1. **API**: add `const sd_lora_t* loras; uint32_t lora_count;` to `sd_hires_params_t`
   (`include/stable-diffusion.h:337-349`). (Alternatively a general `sample_params.loras`
   per-phase spec, but a hires-scoped field matches the existing `--hires-*` pattern and is the
   smaller diff.)
2. **CLI**: add a `--hires-lora "<name:strength>[,…]"` parser in `examples/common/common.cpp`
   next to `on_hires_sigmas_arg` (`:1309`), building a second `lora_vec_hires`, assigned into the
   new `params.hires.loras` fields near `:2429-2430` (where hires sigmas are assigned). Also add
   the JSON hook mirroring `hires_json["custom_sigmas"]` (`:1847`).
3. **Engine**: in `generate_video`, immediately before the hires refine `sample()`
   (`src/stable-diffusion.cpp:~8825`, after `apply_ltxv_refine_image_conditioning` and the
   pool-trim), insert:
   ```cpp
   if (request.hires.lora_count > 0)
       sd_ctx->sd->apply_loras(request.hires.loras, request.hires.lora_count);
   ```
   Because `apply_loras` diffs against `curr_lora_state`, passing the *full* hires set
   (`distill@0.8` [+ `detailer@0.8`]) transitions cleanly from the base set (`distill@0.65`).
   No teardown needed. (For symmetry/robustness, optionally re-assert the base set at the top of
   any subsequent chained segment — the existing up-front `apply_loras` at `:8390` already does
   this on the next `generate_video` call, so chaining is unaffected.)
4. **detailer LoRA caveat** (from `_CONTEXT.md`): it's an LTX-2 (19B) IC-LoRA against our LTX-2.3
   (22B) base — dim-compat unknown. `load_lora_model_from_file` will simply skip tensors whose
   names/shapes don't match (`lora->lora_tensors.empty()` guard at `:2000`), so a mismatch
   degrades to "distill-only on the refine pass" rather than crashing. Verify at GPU time whether
   any detailer tensors actually bind.

### Effort / risk

- **Effort: S–M.** ~1 struct field, ~1 CLI parser + JSON hook, ~1 `apply_loras` call. The hard
  part (re-callable, diffing LoRA state; cheap runtime swap for quantized bases) already exists.
- **Risk: Low** on the runtime path (nvfp4 → `apply_loras_at_runtime`, pure multiplier swap).
  Medium only if someone forces `--lora-apply-mode immediately` on a quantized base (re-fold cost
  + double the fold error); default `auto` avoids this. Detailer dim-mismatch is a data risk, not
  a code risk, and fails safe.

---

## FEATURE 2 — Base-pass custom sigma schedule — **ALREADY DONE**

### Finding

A `--sigmas` flag for the **base** sampler already exists and is fully wired end-to-end. This is
not a code task; it is a test task.

- **Parse**: `on_sigmas_arg` (`examples/common/common.cpp:1302-1307`) →
  `parse_sigmas_arg(argv, &custom_sigmas, "--sigmas")`. Registered in `manual_options` as
  `"--sigmas"` (`examples/common/common.cpp:1435-1437`), help text
  `custom sigma values for the sampler, comma-separated`. JSON hook at
  `examples/common/common.cpp:1875-1878`.
- **Carry**: assigned to `sample_params.custom_sigmas` / `custom_sigmas_count` for **both** the
  image path (`:2383-2384`) and the **video** path (`:2457-2458`). Field defined at
  `include/stable-diffusion.h:267-268`.
- **Consume (base sampler)**: `SamplePlan` (used by `generate_video`) builds its sigma vector at
  `src/stable-diffusion.cpp:4455-4462`:
  ```cpp
  } else if (sample_params->custom_sigmas_count > 0) {
      sigmas = std::vector<float>(custom_sigmas, custom_sigmas + custom_sigmas_count);
      total_steps = sigmas.size() - 1; …
  ```
  used RAW (no flow-shift), exactly the ManualSigmas semantics the workflow needs. Importantly
  the LTX-distilled and LongCat-DMD schedule overrides are gated on
  `sample_params->custom_sigmas_count <= 0` (`:8358`, `:4491`, `:4517`), so a user-supplied
  `--sigmas` list is **honored, not clobbered**, for LTX video.

The euler + explicit-sigma-vector path is already proven by the hires side (`--hires-sigmas` →
`make_hires_sigma_schedule` at `:5786`, custom branch `:5797-5803`; default LTX stage-2 sigmas
`{0.909375,0.725,0.421875,0.0}` at `:8358-8361`). So the base and hires now use the same
sigma-vector plumbing.

### Precise change

None required. Confirm the workflow's base schedule is fed as
`--sigmas 0.85,0.725,0.421875,0.0` (or the S2 `0.725,0.421875,0.0`) and drop
`--steps/--scheduler` for that pass. Validation already enforces ≥2 values on the hires side
(`:2310`); note the **base** `--sigmas` has no explicit `<2` guard, so pass a well-formed list.

### Effort / risk

- **Effort: XS (wiring already present).** At most: add a `<2`-value sanity check for base
  `--sigmas` to match the hires guard, and update the run scripts.
- **Risk: Negligible.** Only open item is GPU-time verification that no LTX-specific override
  slips in (code says it won't) and that `sample_steps` is coerced correctly (it is, `:4460-4462`).

---

## FEATURE 3 — NAG (Normalized Attention Guidance)

### Current behavior — CFG only, at the wrong altitude for NAG

Guidance today is classifier-free guidance combined at the **noise-prediction output** level, via
two *separate* full DiT forwards:

- In the sampler's `denoise` lambda, the positive forward `cond_out = run_condition(*positive_condition,…)`
  (`src/stable-diffusion.cpp:2814`) and, if `!uncond.empty()`, the negative forward
  `uncond_out = run_condition(uncond,…)` (`:2885`) each call `work_diffusion_model->compute(...)`
  end-to-end.
- They are blended at the output by `primary_guidance.forward({pred_cond, pred_uncond,…})`
  (`:2911`, guider in `src/runtime/guidance.{h,cpp}`).
- The DiT only ever sees **one** context per forward: `diffusion_params.context = condition.c_crossattn`
  (`:2751`), routed for LTX via `LTXAVDiffusionExtra` (`:2771`).

NAG is fundamentally different: it blends the **cross-attention output** (attention feature
space), *inside* the DiT, using **both** the positive and negative text context within a single
forward — it cannot be expressed at the sampler's output-combine altitude.

### Where NAG would have to hook — the LTX cross-attention

The cross-attention (`attn2`) is `CrossAttention::forward` at
`src/model/diffusion/ltxv.hpp:713`, producing `to_out.0(attn(q, k(context), v(context)))`.
Invocation sites:

- video cross-attn: `BasicTransformerBlock::forward` at `ltxv.hpp:896` (adaln path) / `:899`
  (plain path) — `attn2->forward(ctx, x/q, context, …)`.
- AV variant: `BasicAVTransformerBlock` cross-attn at `ltxv.hpp:1291` / `:1296`.

`context` (the projected/connectored text tokens, `v_context`) is produced once by
`preprocess_contexts` (`ltxv.hpp:1579-1673`, runs `caption_projection` + `connector`) and threaded
through the block loop `LTXAVModelBlock::forward` (`:1854-1863`) from `build_graph`
(`:2036`, context bound around `:2269`).

NAG per cross-attn block (workflow params scale=14, α=0.35, τ=2.5), operating on the attention
output `z` (before/after `to_out.0`):
```
z_pos  = attn(q, k(ctx_pos), v(ctx_pos))
z_neg  = attn(q, k(ctx_neg), v(ctx_neg))
z_ext  = z_pos + scale*(z_pos - z_neg)                    # extrapolate
r      = ||z_ext|| / ||z_pos||     (per token, over feature dim)
z_nag  = z_ext * (r > tau ? tau/r : 1)                    # norm clamp toward z_pos
z_out  = alpha*z_nag + (1-alpha)*z_pos                    # mix
```

### Precise change (why it's L)

1. **Plumb a second context.** Add `ctx_neg` alongside `context` through: `build_graph`
   signature (`ltxv.hpp:2036`) → `LTXAVModelBlock::forward` (`:1699`) → `preprocess_contexts`
   (must run caption_projection **and** the connector on the neg context too, `:1579-1673`) →
   the block loop (`:1854`) → `BasicTransformerBlock/BasicAVTransformerBlock::forward` (`:859`,
   `:1266`) → `CrossAttention::forward` (`:713`). That is 5 nested forward signatures + a second
   connector pass.
2. **New ggml ops in the hot path.** Inside `CrossAttention::forward`, run `attn` twice (second
   K/V from `ctx_neg`) and build the extrapolate → per-token L2-norm ratio → `min(τ/r,1)` clamp →
   α-mix subgraph. Per-token norm over the feature dim + a clamped scale + two blends, per block,
   for all `num_layers` blocks.
3. **Params + gating.** New `--nag-scale/--nag-alpha/--nag-tau` (+ enable) flags, carried on the
   LTX extra, only active on the stage(s) you choose.

Cost at runtime: **~doubles cross-attention compute** (extra K/V projection + second attention +
the norm/mix ops) for every block on every NAG step, plus a second full connector pass on the neg
context. On our nvfp4 22B at 1088×1920 that is a real per-step tax on the exact stage where the
workflow deliberately keeps it (preview only).

### Effort / risk

- **Effort: L.** Deepest, most invasive change of the three — new tensor plumbing through the
  whole DiT forward stack + new numeric subgraph in the attention hot loop + a duplicated
  connector pass. Also the highest correctness risk (norm/clamp/mix must match the reference math,
  and it touches the flash-attn/kv-scale path at `ltxv.hpp:748-772`).
- **Risk: High** (numerical + perf + broad blast radius on the shared `CrossAttention` used by
  self-attn, cross-attn, connectors, and the AV path).

### Recommendation — **skip for v1**

Per the reverse-engineered stage table in `_CONTEXT.md`, NAG runs **only on S1 (and S3)
preview** — the inspection taps that are explicitly droppable. The load-bearing stages (S2 base,
S4 refine) run **cfg=1 with no NAG**. So NAG buys nothing on the frames we actually keep. Ship v1
without it; revisit only if a later eye-test shows the base/refine faces still smear in a way that
a preview-only guidance would have caught (unlikely, since we don't decode the preview).

---

## Priority + sequencing

| # | Feature | Verdict | Effort | Do for v1? |
|---|---|---|---|---|
| 2 | Base `--sigmas` | **Already implemented** — parse+carry+consume all present (`common.cpp:1302/1435/2457`, `sd.cpp:4455`) | XS (test only) | **Yes — free** |
| 1 | Per-phase LoRA | New `--hires-lora` field + one extra `apply_loras` at `sd.cpp:~8825`; runtime reweight is already cheap for nvfp4. Two-gguf swap **rejected** (whole-DiT disk reload). | S–M | **Yes** |
| 3 | NAG | Deep DiT plumbing + attention-space blend; only on droppable preview stages | L | **No — skip** |

**Recommended order**

1. **Feature 2 first (zero code):** wire the workflow's base ManualSigmas into the run script via
   the existing `--sigmas`; confirm at GPU time that LTX honors it and denoises (gate at
   `sd.cpp:8358`/`4491` says it will).
2. **Feature 1 next:** land `--hires-lora` + the extra `apply_loras` before the refine sample.
   This is the actual face-quality lever the workflow relies on (distill@0.65 base → distill@0.8
   refine). Pairs naturally with the distill-LoRA fold work (fold −0.35 onto `nvfp4-CLEAN.gguf`
   for the base pass) that `_CONTEXT.md` calls the cheap win — note that if you go the
   **pre-folded base-gguf** route (dev+distill@0.65 baked in) you may not even need `--hires-lora`
   for the *base*, only for the refine's +0.8/detailer; still smaller than NAG.
3. **Feature 3 last / optional:** only if a post-Feature-1 eye-test demands it. Given it's
   preview-only in the source workflow, treat as out-of-scope for v1.

**Build/test plan** (builds deferred to GPU/toolchain time per the hard constraints):

- Feature 2: no build; just run the existing binary with `--sigmas 0.725,0.421875,0.0` and check
  the `custom_sigmas_count>0` branch logs (`sd.cpp:4459` "set total_steps") and that faces
  denoise. A/B against `--steps 3 --scheduler simple`.
- Feature 1: after the S/M edit, `make quick`-equivalent DiT build; smoke a short LTX clip with
  `--hires --hires-lora "distill:0.8"` and confirm the log shows a second `apply_loras completed`
  between "sampling completed" and "LTX latent spatial upscale refine". Verify VRAM/step-time are
  unchanged from prod (runtime-LoRA reweight, not a re-fold) — the `_CONTEXT.md` "same VRAM & per-
  step speed" expectation. Confirm whether any detailer-LoRA tensors actually bind (dim-compat).
- Feature 3: only if pursued — unit-test the norm/clamp/mix math on a tiny tensor before touching
  the graph, then A/B a preview-stage clip with/without NAG.
