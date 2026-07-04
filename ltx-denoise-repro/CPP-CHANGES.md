# CPP-CHANGES — `--hires-lora` (per-phase LoRA) + NAG (Normalized Attention Guidance)

Code written **ahead of the build** (no compile/GPU per the hard constraints). Two features landed
sequentially in one editor pass because they share `stable-diffusion.cpp`, `common.h`, `common.cpp`.
All line numbers below were current at edit time; they drift as the files change — grep the marker
strings (`FEATURE 1`, `NAG`) to relocate.

> Note on naming: the CPP-IMPLEMENTATION-PLAN.md calls per-phase LoRA "Feature 1" and NAG "Feature
> 3" (and recommended skipping NAG). This task explicitly required **both** `--hires-lora` and NAG,
> so NAG is implemented here regardless of the plan's skip recommendation.

---

## FEATURE 1 — `--hires-lora` (per-phase LoRA strength on the refine pass)

### What it does
Lets the hires/refine pass apply a **different** LoRA set/strength than the base pass. Denoise-AI v1:
distill@0.65 on the base, distill@0.8 + detailer@0.7 on the refine. The detailer simply joins as
another entry in the hires LoRA map — **no detailer special-casing in code** (it's just a `name:strength`).

### Mechanism (why it's cheap)
`generate_video` already applies the base LoRAs once up front (`stable-diffusion.cpp:8390`) and calls
`sample()` twice against the *same* resident DiT (base @ `:8619`, refine @ `~8838`). `apply_loras`
diffs the requested state against `curr_lora_state`, so re-calling it with the full refine set right
before the refine `sample()` transitions cleanly (base→refine) and, for an nvfp4 base, is the cheap
runtime-multiplier swap (`apply_loras_at_runtime`) — no re-fold, same VRAM/step-time as prod.

### Files / edits
| File:marker | Edit intent |
|---|---|
| `include/stable-diffusion.h` (`sd_hires_params_t`) | Appended `const sd_lora_t* loras; uint32_t lora_count;` (ABI-safe: at struct end). |
| `src/stable-diffusion.cpp` `sd_hires_params_init` | Zero the two new fields (`loras=nullptr; lora_count=0`) so the image path & any `{}`-init default to "reuse base LoRAs". |
| `src/stable-diffusion.cpp` (`FEATURE 1 (--hires-lora)`, ~`:8836`, just after the pre-sample pool trim, before the refine `sample()`) | `if (request.hires.lora_count > 0 && request.hires.loras) sd_ctx->sd->apply_loras(request.hires.loras, request.hires.lora_count);` |
| `examples/common/common.h` (`SDGenerationParams`) | Added `std::string hires_lora_spec;`, `std::map<std::string,float> hires_lora_map;`, `std::vector<sd_lora_t> hires_lora_vec;` (backing storage next to `lora_vec`). |
| `examples/common/common.cpp` `on_hires_lora_arg` (next to `on_hires_sigmas_arg`) | New CLI parser: stashes the raw spec string. |
| `examples/common/common.cpp` `manual_options` (next to `--hires-sigmas`) | Registered `--hires-lora`. |
| `examples/common/common.cpp` hires JSON block (next to `custom_sigmas`) | `hires.loras` (string) → `hires_lora_spec`. |
| `examples/common/common.cpp` `resolve()` (after `extract_and_remove_lora`) | Parse spec → `hires_lora_map`, resolving paths against `lora_model_dir` (mirrors `extract_and_remove_lora`). Unresolvable names warn+skip. |
| `examples/common/common.cpp` `to_sd_vid_gen_params_t()` (after hires custom_sigmas assign) | Flatten `hires_lora_map` → `hires_lora_vec` → `params.hires.loras/lora_count`. |

Video path only (the apply hook lives in `generate_video`). The image path leaves `hires.loras` at
the `sd_hires_params_init` default (nullptr/0).

### Spec syntax
`--hires-lora "distill:0.8,ltx-2-19b-ic-lora-detailer:0.7"` — comma/space/semicolon separated; each
entry is `name:strength` or the wrapper `<lora:name:strength>`. Names resolve against `--lora-model-dir`
with `.gguf/.safetensors/.pt` probing, exactly like prompt `<lora:…>`. **No `|high_noise|` support**
on hires (Wan-MoE concept; LTX refine is a single DiT).

### Detailer caveat (data, not code)
Per COMPAT-REPORT.md the detailer is dimensionally compatible (480/480 targets bind, 0 mismatches).
If a future detailer mismatches, `load_lora_model_from_file` skips non-matching tensors
(`lora->lora_tensors.empty()` guard) → degrades to "distill-only on the refine", never crashes.

### Usage / ablation
```
# hires-lora ON (refine gets a stronger distill + detailer):
sd-cli ... --hires --hires-upscaler ltx-2.3-spatial-upscaler-x2-1.1 \
    --hires-sigmas "0.7250,0.421875,0.0" \
    --hires-lora "ltx-2.3-22b-distilled-lora-384-1.1:0.8,ltx-2-19b-ic-lora-detailer:0.7" \
    -p "... <lora:ltx-2.3-22b-distilled-lora-384-1.1:0.65> ..."   # base pass distill@0.65

# hires-lora OFF (A/B baseline): drop --hires-lora → refine reuses the base pass LoRA state.
```
Look for the log line `hires: applying N per-phase refine LoRA(s) before the refine sample` between
`sampling completed` and `LTX latent spatial upscale refine`. Verify VRAM/step-time unchanged (runtime
reweight, not a re-fold) — confirms the nvfp4 `apply_loras_at_runtime` path.

---

## FEATURE 2 — NAG (Normalized Attention Guidance) — ablatable toggle, DEFAULT OFF

### The math implemented (per NAG paper / LTX2_NAG; canonical form)
Inside the LTX **video text cross-attention**, with the shared query `q`, positive text context and
negative text context, compute the two attention outputs (post-`to_out.0`) `z_pos`, `z_neg`, then:
```
z_ext  = z_pos + scale * (z_pos - z_neg)                       # extrapolate in feature space
n_pos  = ||z_pos||   (per-token L2 over the feature dim)
n_ext  = ||z_ext||   (per-token L2)
factor = min(1, tau * n_pos / n_ext)                           # clamp z_ext norm to <= tau*||z_pos||
z_nag  = z_ext * factor
z_out  = alpha * z_nag + (1 - alpha) * z_pos
```
**Two deliberate deviations from the task's written formulas — read these:**
1. **Norm clamp.** The task wrote `z_nag = z_tilde * min(tau, ||z_pos||/||z_tilde||)`. Taken
   literally that ignores `tau` in the common case (z_ext larger than z_pos ⇒ the `min` always
   picks `||z_pos||/||z_ext|| < 1`, forcing z_nag's norm to exactly `||z_pos||`, i.e. tau=1
   behavior). I implemented the **canonical** NAG clamp `factor = min(1, tau*||z_pos||/||z_ext||)`
   (== the plan's `r=||z_ext||/||z_pos||; z_ext*(r>tau ? tau/r : 1)`), which correctly allows the
   extrapolated norm to grow up to `tau×` before clamping. **Treated the task's literal form as a
   typo.** ← verify against the reference at GPU time.
2. **`scale` convention.** I implemented `z_ext = z_pos + scale*(z_pos-z_neg)` (both the task and the
   plan wrote this). The NAG reference repo / diffusers processor use
   `z_ext = z_pos*scale - z_neg*(scale-1) = z_pos + (scale-1)*(z_pos-z_neg)`. So **our `--nag-scale S`
   ≈ the ComfyUI/reference convention `S+1`.** To reproduce the workflow's ComfyUI "scale=14", pass
   **`--nag-scale 13`** (or change the one line at `ltxv.hpp` `z_ext = …` to the `(scale-1)` form).
   ← the single highest-value thing to pin at eval time.

### The hook approach chosen — option (a): thread the neg context into ONE forward
Rather than the sampler's two-forward output-level CFG (which can't express attention-space NAG), the
**negative text context is threaded into the same DiT forward as the positive** and blended inside the
cross-attention. Chain (each signature got one optional trailing param, default null → legacy byte-identical):

```
denoise lambda (stable-diffusion.cpp)
  └─ run_condition(..., nag_pass=true)                       # sets LTXAVDiffusionExtra.nag_context = &uncond.c_crossattn (+ scale/alpha/tau)
       └─ LTXAVRunner::compute(DiffusionParams)              # reads extra.nag_* into members
            └─ LTXAVRunner::compute(..., nag_context)        # new trailing arg
                 └─ build_graph(..., nag_context_tensor)     # make_optional_input; sets runner_ctx.ltx_nag_{scale,alpha,tau}
                      └─ LTXAVModelBlock::forward(..., nag_context)   # preprocess_contexts(neg) → v_context_neg (video branch only)
                           └─ BasicAVTransformerBlock::forward(..., v_context_neg)
                                └─ apply_text_cross_attention(..., context_neg)   # applies SAME prompt-mod to neg
                                     └─ CrossAttention::forward(..., nag_context)  # z_neg = attend(neg); NAG blend
```

### Where the negative context comes from (the load-bearing plumbing)
NAG needs the encoded negative text **even at CFG=1** (the workflow runs cfg=1 everywhere). At cfg=1
`resolve_guidance` leaves `use_uncond=false` ⇒ the negative prompt is never encoded. Fix:
- `GenerationRequest::resolve` **forces `use_uncond=true` when `LTXAV_NAG_SCALE != 0`** (LTX only) →
  `prepare_video_generation_embeds` materializes `embeds.uncond`, and `generate_video` passes it into
  `sample()` as the `uncond` carrier.
- In the sampler, `nag_owns_guidance = nag_enabled && cfg<=1` **suppresses the now-redundant CFG uncond
  forward** (`if (!uncond.empty() && !nag_owns_guidance)`). Its cfg-1 combine would be a no-op anyway,
  so we pay only NAG's in-attention cost (one extra K/V+attention+to_out per NAG'd cross-attn block),
  not a whole extra forward. When `cfg>1` the uncond forward still runs (user wants NAG **and** CFG).

### Sigma gate (faithful S1-only + ablation)
Our base pass is a single sampler (not the workflow's split S1/S2). `--nag-until-sigma` (default 0.9)
applies NAG only while `sigma >= until_sigma`, matching the workflow's high-noise-only application,
and is enforced in the denoise lambda where the current sigma is known. On the low-sigma refine pass
NAG auto-disables via this gate.

### Params / env bridge (mirrors the A2V knob pattern exactly)
CLI flags → per-render env (`apply_ltx_relip_env` / CLI inline setenv) → read in `sample()` and
`resolve()`. Default OFF (`nag_scale=0`).

| Flag | env | default | meaning |
|---|---|---|---|
| `--nag-scale` | `LTXAV_NAG_SCALE` | `0` (OFF) | extrapolation coeff (see scale-convention note; ComfyUI 14 ≈ our 13) |
| `--nag-alpha` | `LTXAV_NAG_ALPHA` | `0.35` | blend weight of z_nag vs z_pos |
| `--nag-tau` | `LTXAV_NAG_TAU` | `2.5` | per-token norm-clamp ceiling |
| `--nag-until-sigma` | `LTXAV_NAG_UNTIL_SIGMA` | `0.9` | apply NAG only while sigma ≥ this (0 = every step) |

### Files / edits (NAG)
| File | Edit |
|---|---|
| `src/model/diffusion/ltxv.hpp` `CrossAttention::forward` | New `nag_context=nullptr` param. Refactored into `attend(kv_context)` + `apply_gate_if_any(out)` lambdas (q computed once, shared); when `nag_context && ctx->ltx_nag_scale!=0` runs the neg attention and the NAG blend (post-`to_out.0`, per-token L2 via `ggml_sum_rows`+`ggml_sqrt`, `ggml_clamp` for the tau clamp). Legacy path byte-identical when `nag_context==null`. |
| `src/model/diffusion/ltxv.hpp` `apply_text_cross_attention` | New `context_neg=nullptr` param; applies the SAME `prompt_scale_shift` modulation to the neg context; forwards it as `nag_context`. |
| `src/model/diffusion/ltxv.hpp` `BasicAVTransformerBlock::forward` | New `v_context_neg=nullptr` param, passed to the **video** attn2 (`apply_text_cross_attention`) only. |
| `src/model/diffusion/ltxv.hpp` `LTXAVModelBlock::forward` | New `nag_context=nullptr` param; `preprocess_contexts(nag_context, …, process_audio=false)` → `v_context_neg`; threaded into the block loop. |
| `src/model/diffusion/ltxv.hpp` `build_graph` | New `nag_context_tensor={}` param; `make_optional_input` → `nag_context`; sets `runner_ctx.ltx_nag_{scale,alpha,tau}`; passes `nag_context` to `model.forward`. |
| `src/model/diffusion/ltxv.hpp` `LTXAVRunner` | Members `nag_scale_/nag_alpha_/nag_tau_`; both `compute()` overloads thread `nag_context`; `compute(DiffusionParams)` reads `extra->nag_*`. |
| `src/model/diffusion/model.hpp` `LTXAVDiffusionExtra` | Appended `nag_context/nag_scale/nag_alpha/nag_tau` (default-off). |
| `src/core/ggml_extend.hpp` `GGMLRunnerContext` | Appended `ltx_nag_scale/alpha/tau` (default off). |
| `src/stable-diffusion.cpp` `sample()` top | Read `LTXAV_NAG_*` env; compute `nag_enabled`, `nag_owns_guidance`. |
| `src/stable-diffusion.cpp` denoise lambda | `run_condition` gained `nag_pass`; on the cond forward sets `ltx_extra.nag_*` from `uncond.c_crossattn` when `nag_pass && nag_enabled && sigma>=until_sigma && !uncond.c_crossattn.empty()`; primary cond forward called with `nag_pass=true`; CFG uncond forward gated by `!nag_owns_guidance`. |
| `src/stable-diffusion.cpp` `GenerationRequest::resolve` | Force `use_uncond=true` when LTX + `LTXAV_NAG_SCALE!=0`. |
| `examples/common/common.h` | NAG fields on `SDGenerationParams`. |
| `examples/common/common.cpp` | `--nag-*` flags, JSON `load_if_exists`, `apply_ltx_relip_env` setenv. |
| `examples/cli/main.cpp` | Inline setenv bridge (gated on `nag_scale!=0`). |

### Usage / ablation (A/B, sweep)
```
# NAG OFF (default / baseline): pass nothing, or --nag-scale 0.
# NAG ON, faithful-ish to workflow (ComfyUI scale 14 ≈ our 13, S1-only via sigma gate):
sd-cli ... -p "PROMPT" -n "NEG PROMPT" \
    --nag-scale 13 --nag-alpha 0.35 --nag-tau 2.5 --nag-until-sigma 0.9
# scale sweep (confound-free): re-run with --nag-scale 7 / 13 / 20, everything else fixed.
# apply NAG on every step (not just high-noise): --nag-until-sigma 0
```
Env-only ablation (warm server / harness): set `LTXAV_NAG_SCALE`/`_ALPHA`/`_TAU`/`_UNTIL_SIGMA`.
Log line to confirm active: `LTXAV NAG (normalized attention guidance) scale=… until_sigma=…`.

---

## BUILD-VERIFICATION CHECKLIST (where I was uncertain — check these first at compile+eval)

**Correctness / numerics (NAG):**
1. **scale convention** (highest value) — `--nag-scale S` here == extrapolation coeff `S` in
   `z_pos + S*(z_pos−z_neg)`. ComfyUI LTX2_NAG "scale=14" is almost certainly `z_pos*14 − z_neg*13`
   (coeff 13). Confirm against the actual LTX2_NAG node source; if so, our default recipe should pass
   `--nag-scale 13` (and `v3` used scale 15 ⇒ our 14). One-line switch documented in the file.
2. **norm-clamp form** — I used canonical `min(1, tau*||z_pos||/||z_ext||)`, NOT the task's literal
   `min(tau, ||z_pos||/||z_ext||)` (believed a typo). Verify the clamp direction on a tiny tensor
   before a full render.
3. **per-token L2 axis** — `ggml_sum_rows` sums `ne[0]` (the feature/query_dim). Assumes `z_pos`/`z_ext`
   are `[query_dim, tokens, batch]` after `to_out.0`. If `ggml_ext_attention_ext`/`to_out.0` return a
   different layout (e.g. an extra head axis), the norm would be over the wrong axis — **check the
   shape of `to_out_0->forward(out)` at runtime.**
4. **blend space (pre vs post `to_out.0`)** — I apply `to_out.0` to both z_pos/z_neg then blend
   (norm computed in the projected space), matching the diffusers NAG processor. If the reference
   blends *pre*-projection, move the two `to_out_0->forward` calls after the blend.
5. **eps guard** — denominator clamped to `[1e-6, 3e38]`. Fine unless a real token legitimately has
   ~0 norm; revisit only if NAG output shows speckle.
6. **`ggml_mul` broadcast** — relies on `factor` `[1,tok,b]` broadcasting over `z_ext`'s feature dim.
   Confirm ggml's `ggml_mul` broadcasts `ne[0]==1` (it should via `ggml_can_repeat`).

**Context / shape (NAG):**
7. **neg context sequence length == pos length** — `LTXAVModelBlock::forward` runs the neg context
   through `preprocess_contexts` reusing `video_connector_pe`, which `build_graph` sized from the
   **positive** context's `ne[1]`. If the Gemma TE does not pad pos/neg prompts to the same length,
   the connector RoPE will mis-size. **Verify pos/neg `c_crossattn` shapes match**; if not, either pad
   both to a common length or build a separate connector pe for the neg (follow-up).
8. **connector actually runs?** — if the encoded context arrives "fully processed" (`ne[1] >= 1024`),
   `preprocess_contexts` just slices (no connector, no pe) and item 7 is moot. Check which branch
   fires for our Gemma-Q4_K_XL path.
9. **`use_uncond` force reaches embeds** — confirm forcing `use_uncond=true` in
   `GenerationRequest::resolve` actually makes `prepare_video_generation_embeds` populate
   `embeds.uncond.c_crossattn` at cfg=1 (it keys on `request.use_uncond`).
10. **negative prompt present** — NAG with an empty `-n` encodes the empty-string negative (weak
    effect). Pass a real negative prompt for a meaningful test.

**ABI / signatures:**
11. `sd_hires_params_t` grew by two fields (appended). Checked callers: `sd_hires_params_init`,
    `sd_img_gen_params_init`, field-by-field assigns, and `= {}` inits are all safe. Re-grep for any
    positional aggregate init of `sd_hires_params_t` after other edits.
12. `LTXAVDiffusionExtra` grew by 4 fields with default member initializers → still an aggregate; the
    one positional init in `stable-diffusion.cpp` provides 7 initializers and the rest default. Confirm
    the compiler accepts partial aggregate init (C++14 default-member-initializers).
13. All ltxv.hpp signature changes added **trailing default-null** params. Verified single callers for
    `BasicAVTransformerBlock::forward` (block loop) and `model.forward` (build_graph); audio
    `apply_text_cross_attention` and non-AV `BasicTransformerBlock` leave the new params defaulted
    (no NAG there — documented).
14. **non-AV LTX path (`BasicTransformerBlock`) is NOT NAG-wired** — only the AV path
    (`BasicAVTransformerBlock`, what `LTXAVRunner` uses) is. Fine for our prod native-audio model; a
    2-line change would extend it if a non-AV LTX variant ever needs NAG.

**Perf / honesty:**
15. NAG ≈ doubles the **video text cross-attn** cost (extra K/V proj + attention + `to_out.0` + the
    norm/blend ops) on every NAG'd step — but only while `sigma >= until_sigma`, so with the default
    gate it's a handful of early steps. It does NOT double the whole forward. Measure the real
    per-step delta; if the connector re-runs on the neg (item 8), that's an extra connector pass too.
16. **NAG risk is genuinely high** (numeric + the two convention ambiguities + the pos/neg-length
    assumption + broad blast radius on the shared `CrossAttention`). Smoke a tiny clip and eyeball
    that NAG-off is byte-identical to pre-change (the `nag_context==null` gate should guarantee it),
    THEN A/B NAG-on.
