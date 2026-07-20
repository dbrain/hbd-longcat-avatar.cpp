# PULID — apply-ready integration edits

Fork worktree: `/home/dbrain/dev/longcat-avatar-mergewt` (branch `upstream-merge-2026-07-19`)
Upstream tip `ea4e566` (leejet "feat: add hunyuan video 1.5"), merge base `19bdfe2`, fork HEAD.
Every anchor below was `git show ea4e566:` / `Read` / `grep`-verified against the CURRENT fork tree.
**No source was edited; nothing built.**

PuLID = Flux ID-customization adapter. An **ID embedding** (2048-dim, ~32 tokens) — produced OFFLINE
in Python by `scripts/pulid_extract_id.py` (ArcFace + EVA-CLIP → gguf `pulid_id` tensor) — is injected
inside the Flux DiT via **PerceiverAttention cross-attention (`pulid_ca.*`) blocks** interleaved between
the double/single stream blocks. The CA-block *weights* live in a separate PuLID weights file merged into
the diffusion-model tensor map; the *runtime* ID embedding is loaded per-request by a `GenerationExtension`
and fed into the DiT through `FluxDiffusionExtra.pulid_id` via a new `before_diffusion` hook.

---

## 0. TL;DR verdict (read first)

| # | Capability | Portability to fork | Where |
|---|---|---|---|
| **A. Public params** (`sd_pulid_params_t`, `pulid_weights_path`, `pulid_params`) | **Clean additive.** New C-struct + 2 fields; zero effect on existing paths. | §1 |
| **B. `FluxDiffusionExtra += pulid_id, pulid_id_weight`** | **Clean additive**, default `nullptr`/`1.0`. Inert until §4 reads it. | §2 |
| **C. Extension interface** (`before_diffusion` hook + `pulid_params` in condition ctx + `create_pulid_extension` decl) | **Additive, but the fork's `GenerationExtension` interface has DIVERGED** from upstream (no `before_diffusion`; different condition-ctx fields; `GenerationExtensionTensorContext`; `collect_param_tensors`/`alloc_params_buffer`). Add the hook + ctx field; do NOT adopt upstream's `get_param_tensors`/`collect_loras`/`runner_done`/`model_manager`. | §3 |
| **D. `pulid_extension.cpp`** (new file) | **Ports ~verbatim** from upstream — it uses NO GGMLRunner, NO backend, NO ModelManager (unlike photomaker/adetailer). Only adaptation: it reads `ctx.pulid_params` which §3 adds to the fork's condition ctx. | §5 |
| **E. Flux DiT injection (`flux.hpp`)** — THE CRUX | **Highly portable — mechanical re-insertion.** The fork's `flux.hpp` is upstream **minus the pulid lines** (verified: 414-line diff, all pulid hunks are pure deletions). Re-add them, translating ONLY 3 fork conventions: `increase_ref_index` bool (not `Rope::RefIndexMode`), `(backend, params_backend)` ctor, 3-arg `GGMLRunner::compute<float>(…, false)`. Upstream ea4e566 flux.hpp **already carries `mark_graph_cut` marks** (incl. `flux.pulid_ca.*`), so the injection is graph-cut-native out of the box. | §4 |
| **F. Weights merge** (`pulid_weights_path` → tensor map) | **Clean** — one `model_loader.init_from_file(path, "model.diffusion_model.")` after the t5xxl load. | §6 |
| **G. `before_diffusion` call site** (denoise loop) | **New control flow, no fork anchor** but trivial placement — one loop after the per-condition `extra` is built, before `compute()`. | §7 |
| **H. Registration + prepare plumbing** | Additive; clear fork anchors. | §8 |
| **I. CLI (`examples/common`)** | Small, additive; koblem/server surface is out of scope for this C++ doc (flag to owner). | §9 |

**Bottom line:** unlike qwen_image_layered, PuLID has **no missing primitive and no orphaned-runner wiring**
— `pulid.hpp` is already byte-identical to upstream and is a plain `GGMLBlock` consumed inside `flux.hpp`.
The port is: (1) additive plumbing (§1–§3, §6, §8, §9), (2) copy the extension (§5), (3) re-insert the
deleted flux pulid lines (§4) + one denoise-loop hook (§7). The ONLY real risks are **no checkpoint on-box**
(nothing is GPU-validated) and the **VRAM/graph-cut cost of ~20 extra CA attention blocks**.

---

## 1. `include/stable-diffusion.h` — public params (additive)

### 1a. `sd_ctx_params_t` — add `pulid_weights_path`

Anchor (unique, fork line 202):

```c
    const char* photo_maker_path;
    const char* tensor_type_rules;
```

Replace with:

```c
    const char* photo_maker_path;
    const char* pulid_weights_path;
    const char* tensor_type_rules;
```

### 1b. `sd_pulid_params_t` — new struct (mirror upstream, place after `sd_pm_params_t`)

Anchor (unique, fork lines 279-284):

```c
typedef struct {
    sd_image_t* id_images;
    int id_images_count;
    const char* id_embed_path;
    float style_strength;
} sd_pm_params_t;  // photo maker
```

Insert **after** it:

```c
typedef struct {
    const char* id_embedding_path;
    float id_weight;
} sd_pulid_params_t;
```

### 1c. `sd_img_gen_params_t` — add `pulid_params`

Anchor (unique, fork line 391):

```c
    sd_pm_params_t pm_params;
    sd_tiling_params_t vae_tiling_params;
```

Replace with:

```c
    sd_pm_params_t pm_params;
    sd_pulid_params_t pulid_params;
    sd_tiling_params_t vae_tiling_params;
```

> The fork zero-inits `sd_ctx_params` via `*sd_ctx_params = {}` (`stable-diffusion.cpp:4266`) so
> `pulid_weights_path` defaults to `nullptr` automatically — **no explicit default-init line needed**
> (unlike upstream's `sd_ctx_params->pulid_weights_path = nullptr;`). See §8c for the `pulid_params` default.

---

## 2. `src/model/diffusion/model.hpp` — `FluxDiffusionExtra` fields (additive)

Anchor (unique, fork lines 21-24):

```cpp
struct FluxDiffusionExtra {
    const sd::Tensor<float>* guidance   = nullptr;
    const std::vector<int>* skip_layers = nullptr;
};
```

Replace with (matches upstream `model.hpp:54-59`):

```cpp
struct FluxDiffusionExtra {
    const sd::Tensor<float>* guidance   = nullptr;
    const std::vector<int>* skip_layers = nullptr;
    const sd::Tensor<float>* pulid_id   = nullptr;
    float pulid_id_weight               = 1.0f;
};
```

> **Do NOT** pull upstream's `RefImageParams`/`REF_IMAGE_PRESETS`/`RefImageResizeMode` block that also lives
> in upstream `model.hpp` — that is Batch 2/Scope-B (deferred) and the fork uses `bool increase_ref_index`.
> Only the two `pulid_*` fields are in scope here.

---

## 3. `src/extensions/generation_extension.h` — hook + ctx field + decl

**The fork's extension interface has diverged from upstream.** Do NOT wholesale-adopt upstream's header
(it drags in `model_manager.h`, `get_param_tensors`, `collect_loras`, `runner_done`, `model_manager` in the
init ctx). Make three surgical additions.

### 3a. `before_diffusion` virtual hook on `GenerationExtension`

Anchor (unique, fork lines 65-68 — the last virtual before the struct closes):

```cpp
    virtual const SDCondition& before_condition(int step,
                                                const SDCondition& condition) const {
        return condition;
    }
};
```

Replace with (adds the hook + the include the signature needs):

```cpp
    virtual const SDCondition& before_condition(int step,
                                                const SDCondition& condition) const {
        return condition;
    }

    // Called in the denoise loop for each enabled extension, after the per-step
    // DiffusionParams (including its version-specific `extra`) has been built, but
    // before diffusion_model->compute(). Lets an extension feed data into the diffusion
    // forward that the conditioning-side hooks can't reach — typically by setting fields
    // on the architecture-specific `params.extra` variant (e.g. PuLID's id embedding on
    // FluxDiffusionExtra). Mutates `params` only, never the extension. Default no-op.
    virtual void before_diffusion(DiffusionParams& /*params*/, int /*step*/) const {}
};
```

And add the include for `DiffusionParams`. Anchor (unique, fork lines 10-13):

```cpp
#include "conditioning/conditioner.hpp"
#include "core/ggml_extend_backend.h"
#include "model_loader.h"
#include "stable-diffusion.h"
```

Replace with:

```cpp
#include "conditioning/conditioner.hpp"
#include "core/ggml_extend_backend.h"
#include "model/diffusion/model.hpp"
#include "model_loader.h"
#include "stable-diffusion.h"
```

> `model/diffusion/model.hpp` defines `DiffusionParams` + `FluxDiffusionExtra`. It does **not** pull
> `model_manager.h` in the fork (the fork's model.hpp includes only `ggml_extend.hpp`/`tensor_ggml.hpp`),
> so this is a clean include — unlike upstream, which also includes `model_manager.h` here.

### 3b. `pulid_params` on `GenerationExtensionConditionContext`

The fork's condition ctx carries `pm_params` but **not** `pulid_params`. Add it.

Anchor (unique, fork lines 32-41):

```cpp
struct GenerationExtensionConditionContext {
    Conditioner* conditioner;
    ConditionerParams& condition_params;
    const sd_pm_params_t& pm_params;
    std::map<std::string, ggml_tensor*>& tensors;
    SDVersion version;
    int n_threads;
    int total_steps;
    bool free_params_immediately;
};
```

Replace with (insert `pulid_params` right after `pm_params` — see §8b for the matching aggregate-init):

```cpp
struct GenerationExtensionConditionContext {
    Conditioner* conditioner;
    ConditionerParams& condition_params;
    const sd_pm_params_t& pm_params;
    const sd_pulid_params_t& pulid_params;
    std::map<std::string, ggml_tensor*>& tensors;
    SDVersion version;
    int n_threads;
    int total_steps;
    bool free_params_immediately;
};
```

> **Field ORDER matters** — the fork constructs this struct with **aggregate (brace) initialization**
> (`stable-diffusion.cpp:2443`), so inserting `pulid_params` after `pm_params` forces the matching insert in
> §8b. Keep them adjacent.

### 3c. `create_pulid_extension` forward declaration

Anchor (unique, fork line 71):

```cpp
std::shared_ptr<GenerationExtension> create_photomaker_extension();

#endif
```

Replace with:

```cpp
std::shared_ptr<GenerationExtension> create_photomaker_extension();
std::shared_ptr<GenerationExtension> create_pulid_extension();

#endif
```

---

## 4. `src/model/diffusion/flux.hpp` — the DiT injection (THE CRUX)

**Verified premise:** `diff upstream(ea4e566) fork` on `flux.hpp` is 414 lines and **every pulid hunk is a
pure deletion** — the fork is upstream minus pulid. So this section is a *re-insertion*, translating only:
(i) fork's `increase_ref_index` bool ↔ upstream `Rope::RefIndexMode ref_index_mode`;
(ii) fork's `(backend, params_backend)` `FluxRunner` ctor (no ModelManager / model_args);
(iii) fork's 3-arg `GGMLRunner::compute<float>(get_graph, n_threads, false)`.
The `pulid.hpp` block (`PuLIDPerceiverAttentionCA`) is **already present and byte-identical to upstream**
(`git diff` = identical) — no change to that file.

### 4a. Includes

Anchor (unique, fork lines 4-9):

```cpp
#include <memory>
#include <vector>

#include "model/common/rope.hpp"
#include "model/diffusion/dit.hpp"
#include "model/diffusion/model.hpp"
```

Replace with (re-add the two upstream includes):

```cpp
#include <memory>
#include <vector>

#include "core/util.h"
#include "model/adapter/pulid.hpp"
#include "model/common/rope.hpp"
#include "model/diffusion/dit.hpp"
#include "model/diffusion/model.hpp"
```

### 4b. `FluxConfig` — pulid config fields

Anchor (unique, fork lines 53-56):

```cpp
        float ref_index_scale     = 1.f;
        ChromaRadianceConfig chroma_radiance_params;

        static FluxConfig detect_from_weights(const String2TensorStorage& tensor_storage_map,
```

Replace with:

```cpp
        float ref_index_scale     = 1.f;
        ChromaRadianceConfig chroma_radiance_params;

        bool pulid_enabled        = false;
        int pulid_double_interval = 2;
        int pulid_single_interval = 4;

        static FluxConfig detect_from_weights(const String2TensorStorage& tensor_storage_map,
```

### 4c. `detect_from_weights` — light `pulid_enabled` on `pulid_ca.` tensors

Anchor (unique, fork region — the `key_norm.scale` head_dim detection tail):

```cpp
                if (ends_with(name, "double_blocks.0.txt_attn.norm.key_norm.scale")) {
                    head_dim = tensor_storage.ne[0];
                }
            }
            if (actual_radiance_patch_size > 0 && actual_radiance_patch_size != config.patch_size) {
```

Replace with:

```cpp
                if (ends_with(name, "double_blocks.0.txt_attn.norm.key_norm.scale")) {
                    head_dim = tensor_storage.ne[0];
                }
                if (name.find("pulid_ca.") != std::string::npos) {
                    config.pulid_enabled = true;
                }
            }
            if (actual_radiance_patch_size > 0 && actual_radiance_patch_size != config.patch_size) {
```

> This `for`-loop over `tensor_storage_map` is where the merged `pulid_weights_path` tensors (§6) are seen.
> The `name.find` scan must be inside the same loop that iterates tensor names — verify by grep that the
> surrounding block is the `for (const auto& [name, tensor_storage] : tensor_storage_map)` body (it is; the
> `key_norm.scale` checks above are in it).

### 4d. Block construction — build the `pulid_ca.*` CA blocks

Anchor (unique, fork — the `share_modulation` block-build tail, which closes the ctor body):

```cpp
            if (config.share_modulation) {
                blocks["double_stream_modulation_img"] = std::make_shared<Modulation>(config.hidden_size, true, !config.disable_bias);
                blocks["double_stream_modulation_txt"] = std::make_shared<Modulation>(config.hidden_size, true, !config.disable_bias);
                blocks["single_stream_modulation"]     = std::make_shared<Modulation>(config.hidden_size, false, !config.disable_bias);
            }
        }
```

Replace with (re-add the pulid_ca construction, restoring the upstream block verbatim):

```cpp
            if (config.share_modulation) {
                blocks["double_stream_modulation_img"] = std::make_shared<Modulation>(config.hidden_size, true, !config.disable_bias);
                blocks["double_stream_modulation_txt"] = std::make_shared<Modulation>(config.hidden_size, true, !config.disable_bias);
                blocks["single_stream_modulation"]     = std::make_shared<Modulation>(config.hidden_size, false, !config.disable_bias);
            }

            if (config.pulid_enabled) {
                int num_double_ca = (config.depth + config.pulid_double_interval - 1) / config.pulid_double_interval;
                int num_single_ca = (config.depth_single_blocks + config.pulid_single_interval - 1) / config.pulid_single_interval;
                int num_ca        = num_double_ca + num_single_ca;
                for (int i = 0; i < num_ca; i++) {
                    blocks["pulid_ca." + std::to_string(i)] =
                        std::shared_ptr<GGMLBlock>(new PuLIDPerceiverAttentionCA(
                            /*dim=*/config.hidden_size,
                            /*dim_head=*/PuLIDPerceiverAttentionCA::DEFAULT_DIM_HEAD,
                            /*heads=*/PuLIDPerceiverAttentionCA::DEFAULT_HEADS,
                            /*kv_dim=*/PuLIDPerceiverAttentionCA::DEFAULT_KV_DIM));
                }
            }
        }
```

### 4e. `forward_orig` — signature + the two injection sites

**Signature.** Anchor (unique, fork — `forward_orig` param tail):

```cpp
                                  ggml_tensor* pe,
                                  ggml_tensor* mod_index_arange = nullptr,
                                  std::vector<int> skip_layers  = {}) {
```

Replace with:

```cpp
                                  ggml_tensor* pe,
                                  ggml_tensor* mod_index_arange = nullptr,
                                  std::vector<int> skip_layers  = {},
                                  ggml_tensor* pulid_id         = nullptr,
                                  float pulid_id_weight         = 1.0f) {
```

**Guard vars** (re-add just after the prelude `mark_graph_cut` for `vec`). Anchor (unique):

```cpp
            sd::ggml_graph_cut::mark_graph_cut(txt, "flux.prelude", "txt");
            sd::ggml_graph_cut::mark_graph_cut(vec, "flux.prelude", "vec");

            for (int i = 0; i < config.depth; i++) {
```

Replace with:

```cpp
            sd::ggml_graph_cut::mark_graph_cut(txt, "flux.prelude", "txt");
            sd::ggml_graph_cut::mark_graph_cut(vec, "flux.prelude", "vec");

            const bool pulid_active = config.pulid_enabled && pulid_id != nullptr;
            if (pulid_active && !skip_layers.empty()) {
                LOG_WARN("PuLID + skip_layers is not supported; disabling PuLID for this generation.");
            }
            const bool pulid_run = pulid_active && skip_layers.empty();
            int ca_idx           = 0;

            for (int i = 0; i < config.depth; i++) {
```

**Double-stream injection.** Anchor (unique — the end of the double-block loop body):

```cpp
                sd::ggml_graph_cut::mark_graph_cut(img, "flux.double_blocks." + std::to_string(i), "img");
                sd::ggml_graph_cut::mark_graph_cut(txt, "flux.double_blocks." + std::to_string(i), "txt");
            }

            auto txt_img = ggml_concat(ctx->ggml_ctx, txt, img, 1);  // [N, n_txt_token + n_img_token, hidden_size]
```

Replace with (re-add the double-CA block AND restore the `n_txt_tok` binding the single-CA site needs):

```cpp
                sd::ggml_graph_cut::mark_graph_cut(img, "flux.double_blocks." + std::to_string(i), "img");
                sd::ggml_graph_cut::mark_graph_cut(txt, "flux.double_blocks." + std::to_string(i), "txt");

                if (pulid_run && (i % config.pulid_double_interval == 0)) {
                    auto pulid_ca = std::dynamic_pointer_cast<PuLIDPerceiverAttentionCA>(
                        blocks["pulid_ca." + std::to_string(ca_idx)]);
                    ggml_tensor* ca_out = pulid_ca->forward(ctx, pulid_id, img);  // [N, n_img_token, hidden_size]
                    img                 = ggml_add(ctx->ggml_ctx, img, ggml_scale(ctx->ggml_ctx, ca_out, pulid_id_weight));
                    sd::ggml_graph_cut::mark_graph_cut(img, "flux.pulid_ca." + std::to_string(ca_idx), "img");
                    ca_idx++;
                }
            }

            auto txt_img            = ggml_concat(ctx->ggml_ctx, txt, img, 1);  // [N, n_txt_token + n_img_token, hidden_size]
            const int64_t n_txt_tok = txt->ne[1];
```

**Single-stream injection.** Anchor (unique — the end of the single-block loop body):

```cpp
                txt_img = block->forward(ctx, txt_img, vec, pe, txt_img_mask, ss_mods);
                sd::ggml_graph_cut::mark_graph_cut(txt_img, "flux.single_blocks." + std::to_string(i), "txt_img");
            }

            img = ggml_view_3d(ctx->ggml_ctx,
```

Replace with (re-add the single-CA block verbatim from upstream):

```cpp
                txt_img = block->forward(ctx, txt_img, vec, pe, txt_img_mask, ss_mods);
                sd::ggml_graph_cut::mark_graph_cut(txt_img, "flux.single_blocks." + std::to_string(i), "txt_img");

                if (pulid_run && (i % config.pulid_single_interval == 0)) {
                    auto pulid_ca = std::dynamic_pointer_cast<PuLIDPerceiverAttentionCA>(
                        blocks["pulid_ca." + std::to_string(ca_idx)]);
                    ggml_tensor* txt_part = ggml_view_3d(ctx->ggml_ctx, txt_img,
                                                         txt_img->ne[0], n_txt_tok, txt_img->ne[2],
                                                         txt_img->nb[1], txt_img->nb[2],
                                                         0);
                    ggml_tensor* img_part = ggml_view_3d(ctx->ggml_ctx, txt_img,
                                                         txt_img->ne[0],
                                                         txt_img->ne[1] - n_txt_tok,
                                                         txt_img->ne[2],
                                                         txt_img->nb[1],
                                                         txt_img->nb[2],
                                                         n_txt_tok * txt_img->nb[1]);
                    txt_part              = ggml_cont(ctx->ggml_ctx, txt_part);
                    img_part              = ggml_cont(ctx->ggml_ctx, img_part);
                    ggml_tensor* ca_out   = pulid_ca->forward(ctx, pulid_id, img_part);
                    img_part              = ggml_add(ctx->ggml_ctx, img_part, ggml_scale(ctx->ggml_ctx, ca_out, pulid_id_weight));
                    txt_img               = ggml_concat(ctx->ggml_ctx, txt_part, img_part, 1);
                    sd::ggml_graph_cut::mark_graph_cut(txt_img, "flux.pulid_ca." + std::to_string(ca_idx), "txt_img");
                    ca_idx++;
                }
            }

            img = ggml_view_3d(ctx->ggml_ctx,
```

### 4f. Thread pulid through `forward_flux_chroma`, `forward_chroma_radiance`, `forward`

All three have the same pattern: add the two params to the signature and pass them to the `forward_orig` /
delegated call. Each param block currently ends `std::vector<int> skip_layers = {}) {`.

**`forward_chroma_radiance`** — signature anchor (unique, its param tail):

```cpp
                                             std::vector<ggml_tensor*> ref_latents = {},
                                             std::vector<int> skip_layers          = {}) {
            GGML_ASSERT(x->ne[3] == 1);
```
→ append `, ggml_tensor* pulid_id = nullptr, float pulid_id_weight = 1.0f` after `skip_layers = {}`.
And its `forward_orig` call anchor:
```cpp
            auto out = forward_orig(ctx, img, context, timestep, y, guidance, pe, mod_index_arange, skip_layers);  // [N, n_img_token, hidden_size]
```
→ replace `skip_layers);` with `skip_layers,\n                                    pulid_id, pulid_id_weight);`

**`forward_flux_chroma`** — signature anchor (unique, its param tail):
```cpp
                                         std::vector<ggml_tensor*> ref_latents = {},
                                         std::vector<int> skip_layers          = {}) {
            GGML_ASSERT(x->ne[3] == 1);
```
→ append the same two params. And its `forward_orig` call anchor:
```cpp
            auto out = forward_orig(ctx, img, context, timestep, y, guidance, pe, mod_index_arange, skip_layers);  // [N, num_tokens, C * patch_size * patch_size]
```
→ replace `skip_layers);` with `skip_layers,\n                                    pulid_id, pulid_id_weight);`

**`forward`** — signature anchor (unique, its param tail):
```cpp
                             std::vector<ggml_tensor*> ref_latents = {},
                             std::vector<int> skip_layers          = {}) {
            // Forward pass of DiT.
```
→ append the same two params. Then BOTH delegated calls (`forward_chroma_radiance(...)` and
`forward_flux_chroma(...)`) currently end `skip_layers);` — replace each with
`skip_layers,\n                                               pulid_id,\n                                               pulid_id_weight);`
(match the indentation of each call; grep both — there are exactly two `skip_layers);` closers inside `forward`).

### 4g. `FluxRunner::build_graph` — add pulid params, make the input tensor, pass to `flux.forward`

**Keep the fork's signature shape** (`bool increase_ref_index`, NOT upstream's `Rope::RefIndexMode`).
Signature anchor (unique, fork):

```cpp
                                 const std::vector<sd::Tensor<float>>& ref_latents_tensor = {},
                                 bool increase_ref_index                                  = false,
                                 std::vector<int> skip_layers                             = {}) {
```

Replace with:

```cpp
                                 const std::vector<sd::Tensor<float>>& ref_latents_tensor = {},
                                 bool increase_ref_index                                  = false,
                                 std::vector<int> skip_layers                             = {},
                                 const sd::Tensor<float>& pulid_id_tensor                 = {},
                                 float pulid_id_weight                                    = 1.0f) {
```

Make the input tensor just before `auto runner_ctx = get_context();`. Anchor (unique, fork):

```cpp
            auto runner_ctx = get_context();

            ggml_tensor* out = flux.forward(&runner_ctx,
```

Replace with:

```cpp
            ggml_tensor* pulid_id = pulid_id_tensor.empty()
                                        ? nullptr
                                        : make_input(pulid_id_tensor);

            auto runner_ctx = get_context();

            ggml_tensor* out = flux.forward(&runner_ctx,
```

And the `flux.forward(...)` argument list — anchor (unique, fork, its tail):

```cpp
                                            ref_latents,
                                            skip_layers);

            ggml_build_forward_expand(gf, out);
```

Replace with:

```cpp
                                            ref_latents,
                                            skip_layers,
                                            pulid_id,
                                            pulid_id_weight);

            ggml_build_forward_expand(gf, out);
```

### 4h. `FluxRunner::compute` (tensor form) — add pulid params, thread into build_graph

Signature anchor (unique, fork):

```cpp
                                  const std::vector<sd::Tensor<float>>& ref_latents = {},
                                  bool increase_ref_index                           = false,
                                  std::vector<int> skip_layers                      = std::vector<int>()) {
```

Replace with:

```cpp
                                  const std::vector<sd::Tensor<float>>& ref_latents = {},
                                  bool increase_ref_index                           = false,
                                  std::vector<int> skip_layers                      = std::vector<int>(),
                                  const sd::Tensor<float>& pulid_id                 = {},
                                  float pulid_id_weight                             = 1.0f) {
```

The `get_graph` lambda — anchor (unique, fork; note the fork's **3-arg** `GGMLRunner::compute`):

```cpp
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(x, timesteps, context, c_concat, y, guidance, ref_latents, increase_ref_index, skip_layers);
            };

            auto result = restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false), x.dim());
```

Replace with (thread pulid into the build_graph call ONLY — leave the 3-arg `GGMLRunner::compute` untouched):

```cpp
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(x, timesteps, context, c_concat, y, guidance, ref_latents, increase_ref_index, skip_layers, pulid_id, pulid_id_weight);
            };

            auto result = restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false), x.dim());
```

### 4i. `FluxRunner::compute(DiffusionParams)` — pull pulid off the extra

Anchor (unique, fork — the delegated `compute(...)` tail):

```cpp
                           tensor_or_empty(extra->guidance),
                           diffusion_params.ref_latents ? *diffusion_params.ref_latents : empty_ref_latents,
                           diffusion_params.increase_ref_index,
                           extra->skip_layers ? *extra->skip_layers : empty_skip_layers);
```

Replace with:

```cpp
                           tensor_or_empty(extra->guidance),
                           diffusion_params.ref_latents ? *diffusion_params.ref_latents : empty_ref_latents,
                           diffusion_params.increase_ref_index,
                           extra->skip_layers ? *extra->skip_layers : empty_skip_layers,
                           tensor_or_empty(extra->pulid_id),
                           extra->pulid_id_weight);
```

> `tensor_or_empty` and `diffusion_extra_as<FluxDiffusionExtra>` are already used at this call site
> (fork `model.hpp:118` defines `tensor_or_empty`). No new helper needed.
>
> **Do NOT touch `flux.hpp`'s `test()`** for pulid — the fork already rewrote it off ModelManager
> (verified in the diff). It passes no pulid args (defaults apply); leave it.

---

## 5. `src/extensions/pulid_extension.cpp` — NEW FILE (ports ~verbatim)

Create `src/extensions/pulid_extension.cpp` from upstream `git show ea4e566:src/extensions/pulid_extension.cpp`
**verbatim** — it already fits the fork:

- It includes only `extensions/generation_extension.h`, `<cstring>`, `<variant>`, `core/tensor_ggml.hpp`,
  `core/util.h`, `gguf.h` — all present in the fork.
- `PuLIDExtension` overrides only `name/is_enabled/init/reset_runtime_condition/prepare_condition/before_diffusion`
  — every one of those exists on the fork's `GenerationExtension` **after §3a**. It does NOT override
  `collect_param_tensors`/`alloc_params_buffer`/`get_params_buffer_size` (fork-specific, default no-ops are
  correct — PuLID owns no GGMLRunner and no param buffer).
- `init(ctx)` reads `ctx.params->pulid_weights_path` — present after §1a. **It does NOT call
  `ensure_backend_pair`** (no backend needed), so no `SDBackendModule::PULID` enum is required.
- `prepare_condition(ctx)` reads `ctx.pulid_params.id_embedding_path` / `.id_weight` — present after §3b.
  It returns `false` (does not modify conditioning).
- `before_diffusion(params, step)` does `std::get_if<FluxDiffusionExtra>(&params.extra)` and sets
  `pulid_id` / `pulid_id_weight` — present after §2. **This is the only place the id embedding reaches the DiT.**

**One verify-before-copy check:** the upstream file's `load_pulid_id_embedding` uses
`sd::Tensor<float> out({token_dim, num_tokens, 1});` and `out.data()`. Confirm the fork's
`sd::Tensor<float>` (in `core/tensor_ggml.hpp`) has the same 3-arg brace ctor + `.data()`/`.empty()` — it is
used identically by the fork's `photomaker_extension.cpp` (`sd::Tensor<float> id_image_tensor; … .empty()`),
so this is safe.

**Build wiring:** add `src/extensions/pulid_extension.cpp` to the fork's build list the same way
`photomaker_extension.cpp` is registered. Grep the fork build for `photomaker_extension.cpp`:

```
grep -rn "photomaker_extension" CMakeLists.txt Makefile* cmake/ .merge-port-specs/build.sh 2>/dev/null
```

Add a sibling entry for `pulid_extension.cpp` in whichever list contains `photomaker_extension.cpp`.
(If the fork globs `src/extensions/*.cpp`, no edit is needed — verify.)

---

## 6. `src/stable-diffusion.cpp` — merge PuLID weights into the tensor map

Anchor (unique, fork lines 493-497 — the t5xxl load block):

```cpp
        if (strlen(SAFE_STR(sd_ctx_params->t5xxl_path)) > 0) {
            LOG_INFO("loading t5xxl from '%s'", sd_ctx_params->t5xxl_path);
            if (!model_loader.init_from_file(sd_ctx_params->t5xxl_path, "text_encoders.t5xxl.transformer.")) {
                LOG_WARN("loading t5xxl from '%s' failed", sd_ctx_params->t5xxl_path);
            }
        }
```

Insert **after** it (mirrors upstream `stable-diffusion.cpp:766-772`):

```cpp
        if (strlen(SAFE_STR(sd_ctx_params->pulid_weights_path)) > 0) {
            LOG_INFO("loading PuLID weights from '%s'", sd_ctx_params->pulid_weights_path);
            if (!model_loader.init_from_file(sd_ctx_params->pulid_weights_path,
                                             "model.diffusion_model.")) {
                LOG_WARN("loading PuLID weights from '%s' failed", sd_ctx_params->pulid_weights_path);
            }
        }
```

> This merges the `pulid_ca.*` tensors into `model_loader`'s `tensor_storage_map` under the
> `model.diffusion_model.` prefix, so (a) §4c's `detect_from_weights` sees `pulid_ca.` and sets
> `pulid_enabled`, and (b) §4d's block map has weights to load. **Order:** it must run BEFORE the point where
> `tensor_storage_map` is finalized and `FluxRunner` is constructed (the fork builds the runner much later,
> ~line 893+). The t5xxl anchor (line ~493) is in the same early per-file loading pass, so ordering holds.
>
> **Optional logging parity:** add a `"pulid_weights_path: %s\n"` line to `sd_ctx_params_to_str`
> (fork ~line 4316, next to `"photo_maker_path: %s\n"`) + the matching `SAFE_STR(sd_ctx_params->pulid_weights_path)`
> arg (~line 4356). Cosmetic; skip if minimizing surface.

---

## 7. `src/stable-diffusion.cpp` — `before_diffusion` call site (new control flow)

The hook must fire per diffusion forward, after `diffusion_params.extra` is built and after the step-cache
early-return, before `compute()`. In the fork this is inside the `run_condition` lambda.

Anchor (unique, fork lines 3250-3255):

```cpp
                sd::Tensor<float> cached_output;
                if (step_cache.before_condition(&condition, noised_input, &cached_output)) {
                    return std::move(cached_output);
                }

                auto output_opt = work_diffusion_model->compute(n_threads, diffusion_params);
```

Replace with (mirrors upstream `stable-diffusion.cpp:2580-2586`):

```cpp
                sd::Tensor<float> cached_output;
                if (step_cache.before_condition(&condition, noised_input, &cached_output)) {
                    return std::move(cached_output);
                }

                for (const auto& extension : generation_extensions) {
                    extension->before_diffusion(diffusion_params, step);
                }

                auto output_opt = work_diffusion_model->compute(n_threads, diffusion_params);
```

> `generation_extensions` (fork member, line 304) and `step` (captured by the `run_condition` lambda — already
> used at line 3236 for `LongCatAvatarDiffusionExtra{step}`) are both in scope. The lambda runs for cond AND
> uncond forwards; PuLID injecting on both matches classic PuLID (ID applied to both). No-op for every
> non-PuLID run (`before_diffusion` default is empty; PuLID's early-returns if `id_embedding.empty()`).
>
> There is a **second** `work_diffusion_model->compute` at fork line ~3330 (a secondary sub-pass). Decide
> whether that pass also needs the hook. For flux/PuLID the primary `run_condition` path (line 3255) is the
> one that carries `FluxDiffusionExtra`; the 3330 pass is a fork video/ltx sub-branch. **Recommend: gate the
> hook to the 3255 site only** unless a flux forward can reach 3330 (grep the enclosing branch — it is not a
> flux path). Flag for review.

---

## 8. `src/stable-diffusion.cpp` — register the extension + thread `pulid_params`

### 8a. Register `create_pulid_extension` alongside photomaker

Anchor (unique, fork lines 1221-1240 — the photomaker registration block):

```cpp
            {
                generation_extensions.clear();
                auto photomaker_extension = create_photomaker_extension();
                GenerationExtensionInitContext extension_ctx{
                    sd_ctx_params,
                    version,
                    tensor_storage_map,
                    model_loader,
                    n_threads,
                    [this](SDBackendModule module) { return ensure_backend_pair(module); },
                    [this](SDBackendModule module) { return backend_for(module); },
                    [this](SDBackendModule module) { return params_backend_for(module); },
                };
                if (!photomaker_extension->init(extension_ctx)) {
                    return false;
                }
                if (photomaker_extension->is_enabled()) {
                    generation_extensions.push_back(photomaker_extension);
                }
            }
```

Insert **after** the closing `}` of that block (reuse the SAME `extension_ctx` shape; PuLID's `init` only
reads `params->pulid_weights_path`, ignoring the backend lambdas):

```cpp
            {
                auto pulid_extension = create_pulid_extension();
                GenerationExtensionInitContext extension_ctx{
                    sd_ctx_params,
                    version,
                    tensor_storage_map,
                    model_loader,
                    n_threads,
                    [this](SDBackendModule module) { return ensure_backend_pair(module); },
                    [this](SDBackendModule module) { return backend_for(module); },
                    [this](SDBackendModule module) { return params_backend_for(module); },
                };
                if (!pulid_extension->init(extension_ctx)) {
                    return false;
                }
                if (pulid_extension->is_enabled()) {
                    generation_extensions.push_back(pulid_extension);
                }
            }
```

> **Do NOT** `generation_extensions.clear()` in the PuLID block (photomaker already cleared). The subsequent
> `collect_param_tensors` loop (fork line 1247) iterates all registered extensions — PuLID's default
> `collect_param_tensors` is a no-op (no params buffer), which is correct.

### 8b. `prepare_generation_extensions` — accept + forward `pulid_params`

Anchor (unique, fork lines 2439-2452):

```cpp
    void prepare_generation_extensions(const sd_pm_params_t& pm_params,
                                       ConditionerParams& condition_params,
                                       int total_steps) {
        reset_generation_extensions();
        GenerationExtensionConditionContext ctx{
            cond_stage_model.get(),
            condition_params,
            pm_params,
            tensors,
            version,
            n_threads,
            total_steps,
            free_params_immediately,
        };
```

Replace with (add the param + the matching brace-init slot right after `pm_params` — must line up with §3b):

```cpp
    void prepare_generation_extensions(const sd_pm_params_t& pm_params,
                                       const sd_pulid_params_t& pulid_params,
                                       ConditionerParams& condition_params,
                                       int total_steps) {
        reset_generation_extensions();
        GenerationExtensionConditionContext ctx{
            cond_stage_model.get(),
            condition_params,
            pm_params,
            pulid_params,
            tensors,
            version,
            n_threads,
            total_steps,
            free_params_immediately,
        };
```

### 8c. Call site — pass `request->pulid_params`

Anchor (unique, fork lines 7084-7086):

```cpp
    sd_ctx->sd->prepare_generation_extensions(request->pm_params,
                                              condition_params,
                                              plan->total_steps);
```

Replace with:

```cpp
    sd_ctx->sd->prepare_generation_extensions(request->pm_params,
                                              request->pulid_params,
                                              condition_params,
                                              plan->total_steps);
```

### 8d. Default-init `pulid_params` in `sd_img_gen_params`

Anchor (unique, fork line 4456):

```cpp
    sd_img_gen_params->pm_params         = {nullptr, 0, nullptr, 20.f};
```

Insert **after** it (mirrors upstream `stable-diffusion.cpp:3497`):

```cpp
    sd_img_gen_params->pulid_params      = {nullptr, 1.0f};
```

> The img-gen unpack path already copies the whole request struct; `pm_params` is unpacked at fork line 4916
> (`pm_params = sd_img_gen_params->pm_params;`). If the fork similarly caches a local `pulid_params`, mirror
> it there — but since §8c reads `request->pulid_params` directly, no extra local is required. Verify the
> `request` at line 7084 is the `sd_img_gen_params`/derived struct that now carries `pulid_params` (§1c).

---

## 9. `examples/common/` — CLI surface (optional; koblem is the real driver)

The fork DOES ship `examples/common/common.{cpp,h}` (with photomaker CLI). Mirror upstream's three pulid
options if a CLI path is wanted; otherwise defer to Batch 7 and wire only the koblem/server request struct
(out of scope for this C++ doc — flag to owner).

Upstream additions (from `git show ea4e566:examples/common/common.cpp` / `.h`), landing next to the existing
`photo_maker_path` entries (fork `common.cpp:418`, `common.h:136`):

- `common.h`: `std::string pulid_weights_path;` (next to `photo_maker_path`), and
  `std::string pulid_id_embedding_path;` + `float pulid_id_weight = 1.0f;`.
- `common.cpp`: option `--pulid-weights` → `&pulid_weights_path`; `--pulid-id-embedding` →
  `&pulid_id_embedding_path`; `--pulid-id-weight` → `&pulid_id_weight`.
- Wire into params: `sd_ctx_params.pulid_weights_path = pulid_weights_path.c_str();` and
  `params.pulid_params = { pulid_id_embedding_path.empty() ? nullptr : pulid_id_embedding_path.c_str(), pulid_id_weight };`.
- `scripts/pulid_extract_id.py` + `docs/pulid.md`: take upstream wholesale (docs Batch 7). The python script
  is the ONLY producer of the `pulid_id` gguf the extension loads — it is required to actually run PuLID.

---

## 10. Blockers, missing primitives, risks + staged apply plan

**BLOCKER-1 (no checkpoint / no id gguf on-box):** there is no PuLID-enabled Flux checkpoint and no
`pulid_extract_id.py`-produced `pulid_id` gguf on the box, so **nothing here is GPU-validated**. Unverified
against real tensors: the `pulid_ca.*` tensor names/prefix after the §6 merge, the id-embedding shape
(`load_pulid_id_embedding` expects `[token_dim, num_tokens]` = `[2048, 32]`), the `to_kv` view split
(`inner_dim` offset) orientation, and whether the checkpoint's CA weights match
`PuLIDPerceiverAttentionCA`'s block names (`norm1/norm2/to_q/to_kv/to_out`). All are self-consistent with
upstream but must be checkpoint-validated. Requires: a PuLID Flux checkpoint + running the Python extractor.

**BLOCKER-2 (extension-interface divergence, §3):** the fork's `GenerationExtension` predates upstream's
`before_diffusion`/`pulid_params` and uses a *different* method set (`collect_param_tensors`,
`alloc_params_buffer`, `GenerationExtensionTensorContext`). §3 adds the two missing pieces WITHOUT adopting
upstream's `get_param_tensors`/`collect_loras`/`runner_done`/`model_manager`. This is the one place the port
touches a shared interface used by `photomaker_extension.cpp` — the added `pulid_params` field shifts the
brace-init at fork line 2443, so §3b and §8b MUST land together or the aggregate-init mis-binds.

**RISK — graph-cut / offload / VRAM (§4):** `pulid_enabled` adds ~`(depth/2 + depth_single/4)` CA blocks
(≈20 for Flux-dev's 19+38 blocks), each = `to_q(3072→2048)` + `to_kv(2048→4096)` + `to_out(2048→3072)` + 2
LayerNorms + one `ggml_ext_attention_ext`. These load into `params_ctx` like any flux block and their
`mark_graph_cut("flux.pulid_ca.N", …)` marks make them first-class graph-cut segments (upstream ea4e566 flux
already emits these marks, so the fork's segmenter handles them). **Verify:** (a) the merged `pulid_weights_path`
tensors are NOT swept into any `ignore_tensors` set and DO receive the fork's quant/nvfp4 `tensor_type_rules`;
(b) the extra CA VRAM fits under `--max-vram`/graph-cut on the 3060/5060Ti; (c) `ggml_ext_attention_ext(…,
ctx->backend, …)` (used inside `pulid.hpp`) is the fork's current signature (it is — `pulid.hpp` is
byte-identical to upstream and compiles against the same helper the fork exposes). Flag (b) as the main
GPU-validation item.

**RISK — `before_diffusion` firing on both cond+uncond and on the secondary compute (§7):** the hook lives in
the `run_condition` lambda (line 3255), correct for the primary flux forward. A second `compute` at line ~3330
is a non-flux (video/ltx) sub-pass; do NOT duplicate the hook there unless a flux path can reach it (it
cannot — verify the enclosing branch). Harmless either way (PuLID early-returns unless a flux `extra` +
non-empty id embedding is present), but keep the surface minimal.

**RISK — determinism:** `PuLIDPerceiverAttentionCA` uses `ggml_ext_attention_ext(..., diag_mask_inf=false)`
which may route through flash-attn; that inherits the fork's existing (documented) flash nondeterminism, no
new source. Not a blocker.

**ModelManager entanglement — NONE to port.** Upstream's `generation_extension.h` init ctx carries
`std::shared_ptr<ModelManager> model_manager` and its `flux.hpp::test()` uses `ModelManager`; the fork already
dropped both (verified in the flux.hpp diff — `test()` is rewritten off ModelManager, and §3 does not add
`model_manager`). PuLID's extension never touches ModelManager. Nothing to translate.

**No missing primitive.** Unlike qwen_image_layered (needed `Conv2dBut3d`, temporal RoPE) and unlike
adetailer (needs a yolov8 GGMLRunner), PuLID needs **zero new primitives**: `pulid.hpp` is present +
identical, `PuLIDPerceiverAttentionCA` is a plain `GGMLBlock`, and the ID encoder runs offline in Python.

### Staged apply plan (self-contained plumbing first, DiT-touch last)

- **Stage 1 — inert public + struct plumbing (safe, builds, zero behavior change):**
  §1 (params structs), §2 (`FluxDiffusionExtra` fields), §3 (`before_diffusion` hook + `pulid_params` ctx
  field + `create_pulid_extension` decl). Nothing constructs a PuLID extension yet; every path unchanged.
- **Stage 2 — weights + registration + prepare plumbing (still inert):**
  §6 (weights merge), §8 (register extension, prepare signature/call, default-init), §9 (optional CLI).
  `FluxDiffusionExtra.pulid_id` stays `nullptr` because §4 hasn't been applied, so flux ignores it.
- **Stage 3 — the extension + the hook call:**
  §5 (`pulid_extension.cpp` new file + build wiring), §7 (`before_diffusion` loop in denoise). Now
  `before_diffusion` sets `pulid_id` on the extra — but §4 flux still doesn't read it. Build + verify a
  normal (non-PuLID) flux render is byte-identical (id path off ⇒ no-op).
- **Stage 4 — DiT-touching final stage (highest risk, GPU-gated):**
  §4 (flux.hpp: includes, config, detection, block build, `forward_orig` double+single injection, threading
  through `forward_flux_chroma`/`forward_chroma_radiance`/`forward`/`build_graph`/`compute`×2). This lights up
  the whole path. GPU-validate: (1) non-PuLID flux render still bit-exact (pulid_enabled=false ⇒ no blocks
  built, no injection); (2) with a PuLID checkpoint + extracted id gguf, the CA blocks load, VRAM fits, and
  the ID takes effect. Do NOT declare done without (1)+(2).

**Verification status:** all fork anchors (`include/stable-diffusion.h`, `src/model/diffusion/model.hpp`,
`src/extensions/generation_extension.h`, `src/extensions/photomaker_extension.cpp`,
`src/model/diffusion/flux.hpp`, `src/model/adapter/pulid.hpp`, `src/stable-diffusion.cpp`) were
`Read`/`grep`/`git show`-confirmed against the current worktree. The `flux.hpp` re-insertion was validated by a
full `diff` of fork vs upstream (`ea4e566`) proving every pulid hunk is a clean deletion. `pulid.hpp` and
`pulid_extension.cpp` were confirmed byte-identical / copy-ready against upstream. **Cannot build or run**
(project rule + no checkpoint).
