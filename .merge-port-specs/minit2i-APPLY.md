# MiniT2I — apply-ready integration edits

Fork worktree: `/home/dbrain/dev/longcat-avatar-mergewt` (branch `upstream-merge-2026-07-19`)
Upstream tip: `ea4e566`. Every anchor below was `grep`-verified against the CURRENT fork tree.

## Status of already-done pieces (verify, do NOT re-add)

- `src/model/diffusion/minit2i.hpp` — adapted + committed (fork Runner ctor, 3-arg `compute`).
- `src/model.h` — `VERSION_MINIT2I` at line 55 (between `VERSION_BOOGU_IMAGE` and `VERSION_HUNYUAN_VIDEO`),
  `sd_version_is_minit2i()` at line 206, and it is in `sd_version_is_dit()`. Confirmed.
- `src/stable-diffusion.cpp` `model_version_to_str[]` — **ALREADY correct**. The krea2 commit (`7502d89`) appended
  `"Krea2","Boogu Image","MiniT2I","HunyuanVideo 1.5","LingBot Video"`. `"MiniT2I"` sits at array index 55,
  index-aligned with `VERSION_MINIT2I`. **NO EDIT NEEDED here** (the spec's "append MiniT2I" step is already done).
- `include/stable-diffusion.h` / `prediction_to_str[]` — `FLUX2_FLOW_PRED` (header line 83) and `"flux2_flow"`
  (cpp line 3975) anchors both exist; edits below still required for `MINIT2I_FLOW_PRED`.

---

# PART A — THREE NEW CLASSES

## A1. `MiniT2IConditioner` → `src/conditioning/conditioner.hpp`

Insertion point: **immediately before** `struct AnimaConditioner : public Conditioner {` (currently line 1487) —
same position as upstream. Anchor (unique):

```cpp
struct AnimaConditioner : public Conditioner {
```

Insert the following struct + a blank line **before** that anchor line.

Adaptation vs upstream `ea4e566:src/conditioning/conditioner.hpp:1527`:
- ctor gains `ggml_backend_t params_backend`; `T5Runner` ctor is the fork 5-arg
  `T5Runner(backend, params_backend, tensor_storage_map, prefix, is_umt5=false)` — verified against fork uses at
  lines 627/1011/1276. `weight_manager` / `RunnerWeightManager` dropped everywhere.
- `t5->compute(...)` is the fork **3-arg** form `(n_threads, input_ids, attention_mask)` — the upstream trailing
  `false, true, true` bools do not exist on the fork `T5Runner::compute` (fork t5.hpp:395). Verified vs fork call
  sites 925/1210/1424.
- Overrides pruned to match the fork `Conditioner` base (base at conditioner.hpp:110). **Dropped** upstream
  overrides that are NOT virtuals in the fork base: `set_runtime_backends`, `set_graph_cut_layer_split_enabled`,
  `set_graph_cut_layer_split_backend_vram_limits`, `get_layer_split_param_tensors`, `runner_done`.
  **Added** fork base pure-virtuals upstream omitted: `alloc_params_buffer`, `free_params_buffer`,
  `get_params_buffer_size` (mirrors fork `FluxCLIPEmbedder`, conditioner.hpp:980).

```cpp
struct MiniT2IConditioner : public Conditioner {
    T5UniGramTokenizer tokenizer;
    std::shared_ptr<T5Runner> t5;
    size_t prompt_length = 256;

    MiniT2IConditioner(ggml_backend_t backend,
                       ggml_backend_t params_backend,
                       const String2TensorStorage& tensor_storage_map = {}) {
        bool use_t5 = false;
        for (const auto& pair : tensor_storage_map) {
            if (pair.first.find("text_encoders.t5xxl") != std::string::npos) {
                use_t5 = true;
                break;
            }
        }
        if (!use_t5) {
            LOG_WARN("IMPORTANT NOTICE: No MiniT2I T5 text encoder provided, cannot process prompts!");
            return;
        }
        t5 = std::make_shared<T5Runner>(backend, params_backend, tensor_storage_map, "text_encoders.t5xxl.transformer", false);
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        if (t5) {
            t5->get_param_tensors(tensors, "text_encoders.t5xxl.transformer");
        }
    }

    bool alloc_params_buffer() override {
        if (t5) {
            return t5->alloc_params_buffer();
        }
        return true;
    }

    void free_params_buffer() override {
        if (t5) {
            t5->free_params_buffer();
        }
    }

    size_t get_params_buffer_size() override {
        size_t buffer_size = 0;
        if (t5) {
            buffer_size += t5->get_params_buffer_size();
        }
        return buffer_size;
    }

    void set_max_graph_vram_bytes(size_t max_vram_bytes) override {
        if (t5) {
            t5->set_max_graph_vram_bytes(max_vram_bytes);
        }
    }

    void set_stream_layers_enabled(bool enabled) override {
        if (t5) {
            t5->set_stream_layers_enabled(enabled);
        }
    }

    void set_flash_attention_enabled(bool enabled) override {
        if (t5) {
            t5->set_flash_attention_enabled(enabled);
        }
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        if (t5) {
            t5->set_weight_adapter(adapter);
        }
    }

    SDCondition get_learned_condition(int n_threads,
                                      const ConditionerParams& conditioner_params) override {
        SDCondition result;
        if (!t5) {
            result.c_crossattn = sd::Tensor<float>::zeros({1024, static_cast<int64_t>(prompt_length)});
            result.c_vector    = sd::Tensor<float>::zeros({static_cast<int64_t>(prompt_length)});
            return result;
        }

        std::vector<int> tokens = tokenizer.encode(conditioner_params.text);
        if (tokens.size() > prompt_length) {
            tokens.resize(prompt_length);
        }
        std::vector<float> mask(tokens.size(), 1.0f);
        while (tokens.size() < prompt_length) {
            tokens.push_back(tokenizer.PAD_TOKEN_ID);
            mask.push_back(0.0f);
        }

        sd::Tensor<int32_t> input_ids({static_cast<int64_t>(tokens.size())}, tokens);
        std::vector<float> t5_mask(mask.size(), 0.0f);
        for (size_t i = 0; i < mask.size(); ++i) {
            t5_mask[i] = mask[i] > 0.0f ? 0.0f : -HUGE_VALF;
        }
        sd::Tensor<float> hidden_states = t5->compute(n_threads,
                                                      input_ids,
                                                      sd::Tensor<float>::from_vector(t5_mask));
        GGML_ASSERT(!hidden_states.empty());
        result.c_crossattn = std::move(hidden_states);
        result.c_vector    = sd::Tensor<float>::from_vector(mask);
        return result;
    }
};
```

## A2. `MiniT2IFlowDenoiser` → `src/runtime/denoiser.hpp`

Insertion point: **immediately before** the `denoise_cb_t` typedef (currently line 917) — same position as upstream.
Anchor (unique):

```cpp
typedef std::function<sd::guidance::GuiderOutput(const sd::Tensor<float>&, float, int)> denoise_cb_t;
```

Insert the following (with the leading comment) **before** that anchor line. It ports **verbatim** from
`ea4e566:src/runtime/denoiser.hpp:1360` — no backend/weight threading needed (pure math). Fork `Denoiser` base
(denoiser.hpp:636) matches every override signature (`sigma_min/max`, `sigma_to_t`, `t_to_sigma`, `get_scalings`,
`noise_scaling`, `inverse_noise_scaling`, `get_sigmas`). `SD_UNUSED` is available in this header (used at line 776).
`noise_level_to_sigma` is a plain (non-override) helper — harmless.

```cpp
// MiniT2I predicts x0 directly and integrates a linear flow ODE:
//   x_{t+dt} = x_t + (x0 - x_t)/(1 - t) * dt,  t in [0, 1), x0 = start = noise * 2.
// Mapping sigma = 1 - t makes the generic Euler update
//   x += (x - denoised)/sigma * (sigma_next - sigma)
// exactly reproduce that step when denoised == x0. To make the generic
// `denoised = pred * c_out + x * c_skip` yield x0 from the model's raw x0
// prediction we use c_skip = 0, c_out = 1, c_in = 1. Sigmas run linearly 1 -> 0.
struct MiniT2IFlowDenoiser : public Denoiser {
    float sigma_min() override {
        return 0.0f;
    }

    float sigma_max() override {
        return 1.0f;
    }

    float sigma_to_t(float sigma) override {
        return 1.0f - sigma;
    }

    float t_to_sigma(float t) override {
        return 1.0f - t;
    }

    std::vector<float> get_scalings(float sigma) override {
        SD_UNUSED(sigma);
        float c_skip = 0.0f;
        float c_out  = 1.0f;
        float c_in   = 1.0f;
        return {c_skip, c_out, c_in};
    }

    sd::Tensor<float> noise_scaling(float sigma,
                                    const sd::Tensor<float>& noise,
                                    const sd::Tensor<float>& latent) override {
        SD_UNUSED(sigma);
        SD_UNUSED(latent);
        // Sampling starts from x0_init = noise * 2 (see MiniT2I reference).
        return noise * 2.0f;
    }

    sd::Tensor<float> inverse_noise_scaling(float sigma, const sd::Tensor<float>& latent) override {
        SD_UNUSED(sigma);
        return latent;
    }

    float noise_level_to_sigma(float noise_level) {
        SD_UNUSED(noise_level);
        return 1.0f;
    }

    std::vector<float> get_sigmas(uint32_t n, int image_seq_len, scheduler_t scheduler_type, SDVersion version, const char* extra_sample_args = nullptr) override {
        SD_UNUSED(image_seq_len);
        SD_UNUSED(scheduler_type);
        SD_UNUSED(version);
        SD_UNUSED(extra_sample_args);
        // Uniform t schedule 0 -> 1 => sigma 1 -> 0, matching the reference loop.
        std::vector<float> sigmas;
        sigmas.reserve(n + 1);
        for (uint32_t i = 0; i < n; ++i) {
            sigmas.push_back(1.0f - static_cast<float>(i) / static_cast<float>(n));
        }
        sigmas.push_back(0.0f);
        return sigmas;
    }
};
```

## A3. `MiniT2IDiffusionExtra` + variant → `src/model/diffusion/model.hpp`

### A3a. Add the struct

Anchor (unique, currently lines 73–75):

```cpp
struct LongCatAvatarDiffusionExtra {
    int step = -1;  // sampler step index for the cond-frame K/V cache (lap-26); -1 = unknown
};
```

Insert **after** that struct's closing `};`:

```cpp

struct MiniT2IDiffusionExtra {
    const sd::Tensor<float>* mask = nullptr;
};
```

### A3b. Add to the variant

Anchor (unique, currently line 85 — last variant member):

```cpp
                                          LongCatAvatarDiffusionExtra>;
```

Replace with:

```cpp
                                          LongCatAvatarDiffusionExtra,
                                          MiniT2IDiffusionExtra>;
```

---

# PART B — WIRING EDITS

## B1. `include/stable-diffusion.h` — prediction enum

Anchor (unique, lines 83–84):

```cpp
    FLUX2_FLOW_PRED,
    PREDICTION_COUNT
```

Replace with:

```cpp
    FLUX2_FLOW_PRED,
    MINIT2I_FLOW_PRED,
    PREDICTION_COUNT
```

## B2. `src/stable-diffusion.cpp` — `prediction_to_str[]`

Anchor (unique, lines 3969–3976):

```cpp
const char* prediction_to_str[] = {
    "eps",
    "v",
    "edm_v",
    "sd3_flow",
    "flux_flow",
    "flux2_flow",
};
```

Replace the `"flux2_flow",` line with:

```cpp
    "flux2_flow",
    "minit2i_flow",
```

(index-aligned with `MINIT2I_FLOW_PRED`.)

## B3. `src/stable-diffusion.cpp` — include

Anchor (unique, line 35):

```cpp
#include "model/diffusion/mmdit.hpp"
```

Insert **before** it (keeps `ltxv` < `minit2i` < `mmdit` ordering):

```cpp
#include "model/diffusion/minit2i.hpp"
```

## B4. `src/model_loader.cpp` — detection

Anchor (unique, lines 515–518 — the VERSION_LENS block):

```cpp
        if (tensor_storage.name.find("model.diffusion_model.transformer_blocks.0.attn.norm_added_q.weight") != std::string::npos &&
            tensor_storage_map.find("model.diffusion_model.transformer_blocks.0.img_mlp.w1.weight") != tensor_storage_map.end()) {
            return VERSION_LENS;
        }
```

Insert **immediately after** that block's closing `}`:

```cpp
        if (tensor_storage.name.find("net.img_embedder.proj1.weight") != std::string::npos) {
            return VERSION_MINIT2I;
        }
```

(Substring matches `model.diffusion_model.model.net.img_embedder.proj1.weight`; does NOT collide with HiDream's
`model.x_embedder.proj1.weight` — verified different string.)

## B5. `src/stable-diffusion.cpp` — diffusion-model dispatch

Anchor (unique, line 711 — start of the krea2 dispatch branch):

```cpp
            } else if (sd_version_is_krea2(version)) {
```

Insert **before** that line (fork convention: `backend_for`/`params_backend_for`, no `model_manager`; conditioner is
the global `MiniT2IConditioner`, runner is namespaced `MiniT2I::MiniT2IRunner`, DiT prefix
`model.diffusion_model.model.net`):

```cpp
            } else if (sd_version_is_minit2i(version)) {
                cond_stage_model = std::make_shared<MiniT2IConditioner>(backend_for(SDBackendModule::TE),
                                                                        params_backend_for(SDBackendModule::TE),
                                                                        tensor_storage_map);
                diffusion_model  = std::make_shared<MiniT2I::MiniT2IRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                           params_backend_for(SDBackendModule::DIFFUSION),
                                                                           tensor_storage_map,
                                                                           "model.diffusion_model.model.net");
```

## B6. `src/stable-diffusion.cpp` — FakeVAE construction

Anchor (unique, line 1076):

```cpp
            if (version == VERSION_CHROMA_RADIANCE || version == VERSION_HIDREAM_O1) {
```

Replace with:

```cpp
            if (version == VERSION_CHROMA_RADIANCE || version == VERSION_HIDREAM_O1 || sd_version_is_minit2i(version)) {
```

## B7. `src/stable-diffusion.cpp` — `get_latent_channel()`

Anchor (unique, lines 3547–3549 — the PiD branch inside the `sd_version_is_dit` block):

```cpp
            } else if (sd_version_is_pid(version)) {
                latent_channel = 3;
            } else if (sd_version_uses_flux2_vae(version)) {
```

Replace with:

```cpp
            } else if (sd_version_is_pid(version)) {
                latent_channel = 3;
            } else if (sd_version_is_minit2i(version)) {
                latent_channel = 3;
            } else if (sd_version_uses_flux2_vae(version)) {
```

## B8. `src/stable-diffusion.cpp` — `decode_first_stage()` clamp

Anchor (unique, lines 3711–3713):

```cpp
        if (sd_version_is_pid(version)) {
            return sd::ops::clamp((x + 1.f) * 0.5f, 0.0f, 1.0f);
        }
```

Replace the `if` condition line with:

```cpp
        if (sd_version_is_pid(version) || sd_version_is_minit2i(version)) {
            return sd::ops::clamp((x + 1.f) * 0.5f, 0.0f, 1.0f);
        }
```

## B9. `src/stable-diffusion.cpp` — diffusion-extra assignment

This lives inside the `run_condition = [&](const SDCondition& condition, ...)` lambda, so the in-scope condition
variable is **`condition`** (verified). Anchor (unique, lines 3133–3136):

```cpp
                } else if (sd_version_is_longcat_avatar(version)) {
                    diffusion_params.extra = LongCatAvatarDiffusionExtra{step};  // cond-frame K/V cache (lap-26)
                } else {
                    diffusion_params.extra = std::monostate{};
                }
```

Replace with:

```cpp
                } else if (sd_version_is_longcat_avatar(version)) {
                    diffusion_params.extra = LongCatAvatarDiffusionExtra{step};  // cond-frame K/V cache (lap-26)
                } else if (sd_version_is_minit2i(version)) {
                    diffusion_params.extra = MiniT2IDiffusionExtra{
                        condition.c_vector.empty() ? nullptr : &condition.c_vector};
                } else {
                    diffusion_params.extra = std::monostate{};
                }
```

(`condition.c_crossattn` already feeds `diffusion_params.context`; the extra carries `c_vector` as the DiT text mask.
The conditioner always returns a non-empty `c_vector` of length `prompt_length`, so the runner's
`GGML_ASSERT(extra->mask != nullptr)` holds.)

## B10. `src/stable-diffusion.cpp` — pred_type detection

Anchor (unique, lines 1656–1660):

```cpp
                } else if (sd_version_is_flux2(version)) {
                    pred_type = FLUX2_FLOW_PRED;
                } else {
                    pred_type = EPS_PRED;
                }
```

Replace with:

```cpp
                } else if (sd_version_is_flux2(version)) {
                    pred_type = FLUX2_FLOW_PRED;
                } else if (sd_version_is_minit2i(version)) {
                    pred_type = MINIT2I_FLOW_PRED;
                } else {
                    pred_type = EPS_PRED;
                }
```

## B11. `src/stable-diffusion.cpp` — pred_type switch case

Anchor (unique, lines 1685–1688):

```cpp
                case FLUX_FLOW_PRED: {
                    LOG_INFO("running in Flux FLOW mode");
                    denoiser = std::make_shared<FluxFlowDenoiser>();
                    break;
                }
```

Insert **after** that case block:

```cpp
                case MINIT2I_FLOW_PRED: {
                    LOG_INFO("running in MiniT2I FLOW mode");
                    denoiser = std::make_shared<MiniT2IFlowDenoiser>();
                    break;
                }
```

## B12. `src/stable-diffusion.cpp` — uncond derivation

`cond` is in scope here (`auto cond = ...->get_learned_condition(...)` a few lines above; verified). `zeros_like`
exists (`sd::Tensor<float>::zeros_like`, tensor.hpp:245). Anchor (unique, lines 6980–6983):

```cpp
    if (request->use_uncond || request->use_high_noise_uncond) {
        if (sd_version_is_ideogram4(sd_ctx->sd->version)) {
            uncond.c_vector = sd::Tensor<float>::from_vector({1.0f});
        } else {
```

Replace with:

```cpp
    if (request->use_uncond || request->use_high_noise_uncond) {
        if (sd_version_is_ideogram4(sd_ctx->sd->version)) {
            uncond.c_vector = sd::Tensor<float>::from_vector({1.0f});
        } else if (sd_version_is_minit2i(sd_ctx->sd->version)) {
            uncond.c_crossattn = cond.c_crossattn;
            uncond.c_vector    = sd::Tensor<float>::zeros_like(cond.c_vector);
        } else {
```

---

# VERIFICATION / RISK NOTES

- **All anchors verified** against the current fork tree at the line numbers cited.
- `model_version_to_str[]` already contains `"MiniT2I"` (krea2 commit) at the aligned index — do NOT append again.
- `T5Runner::compute` is 3-arg in the fork; upstream's trailing `false,true,true` were dropped (they map to the
  4/5-arg `GGMLRunner::compute` which the fork's `T5Runner` already hardcodes as `auto_free=true`).
- Fork `Conditioner` base lacks upstream's `set_runtime_backends` / `set_graph_cut_*` / `get_layer_split_param_tensors`
  / `runner_done` virtuals — those overrides were dropped; the fork's required pure-virtuals
  `alloc_params_buffer` / `free_params_buffer` / `get_params_buffer_size` were added (mirroring `FluxCLIPEmbedder`).
- **Cannot build/compile-check here** (per instructions) and **no real MiniT2I checkpoint** was available: the
  detection substring `net.img_embedder.proj1.weight` and the DiT prefix `model.diffusion_model.model.net` are
  self-consistent with the hpp's own `detect_from_weights`, but unverified against actual weights — if a checkpoint
  stores the DiT under a different top-level prefix, adjust B4's detection substring and B5's runner prefix together.
- `MiniT2IConditioner::get_learned_condition` emits `c_crossattn` of shape `[1024, 256]` and a `[256]` mask; the
  T5 hidden dim (1024) is assumed from upstream and could differ per checkpoint — only affects the empty-encoder
  fallback path.
- `MiniT2IFlowDenoiser::noise_level_to_sigma` is a non-virtual helper (no fork base equivalent); it is inert.
