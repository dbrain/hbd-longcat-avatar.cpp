# sefi_image — fork-adapted apply-ready port

Worktree: `/home/dbrain/dev/longcat-avatar-mergewt` (branch `upstream-merge-2026-07-19`)
Upstream tip: `ea4e566`  ·  merge base: `19bdfe2`  ·  fork HEAD: `c58a01d`

**Every anchor below was verified against the CURRENT fork tree** (not the draft's assumptions).
Line numbers are informational; the quoted anchor strings are what to match. All code is written
in **fork convention** (`increase_ref_index` bool, `backend_for/params_backend_for`, fork enum layout).

## What sefi_image actually is (verified)

- `src/model/diffusion/sefi_image.hpp` is **already in the fork tree and byte-identical to
  `ea4e566`** (`git diff ea4e566 -- src/model/diffusion/sefi_image.hpp` = empty). It contains only
  three `GGMLBlock` helper structs (`SefiImageConfig`, `SefiTimestepEmbedding`,
  `SefiDualTimestepEmbeddings`). **No edit needed, and it needs none** — but it is currently
  **orphaned** (nothing `#include`s it). Edit F1 wires it in.
- It rides `Flux::FluxRunner` (no runner of its own) + `LLMEmbedder` (arch `QWEN3_VL`).
- Detection tensor: `dual_time_embed.semantic_embedder.linear_1.weight`.

### sefi_image.hpp primitive audit (all present in fork — compiles once included)

| symbol used | provided by |
|---|---|
| `GGMLBlock`, `Linear`, `GGMLRunnerContext` | `src/model/common/block.hpp` (already `#include`d by sefi_image.hpp) |
| `ggml_silu_inplace` | `src/model/common/block.hpp` |
| `ggml_ext_timestep_embedding` | `src/core/ggml_extend.hpp:2088` |
| `ggml_concat` | ggml core |
| `starts_with` / `ends_with` | `src/core/util.{h,cpp}` |
| `String2TensorStorage` | `model_loader.h` |

**No fork-missing primitive blocks sefi_image.hpp.**

### Critical translation: RoPE ref-index (verified in fork tree)

Upstream `flux.hpp` uses `Rope::RefIndexMode ref_index_mode` and sets
`ref_index_mode = Rope::RefIndexMode::INCREASE` for flux2 **and** sefi. The fork has **no
`RefIndexMode` enum** (`grep RefIndexMode src/model/common/rope.hpp` = 0 hits); `gen_flux_pe`
and `gen_refs_ids` take **`bool increase_ref_index`** instead. The fork's flux `build_graph`
already threads a per-model ref-index: `flux.hpp:1426-1432` forces `increase_ref_index = true`
for flux2. **sefi uses the exact same INCREASE behavior as flux2**, so the only RoPE edit is to
extend that version condition (Edit F6). `txt_arange` for sefi = `{3}`, same as flux2 (confirmed
upstream `flux.hpp:1542-1544`). No new bool plumbing is required.

---

## EDITS

### model.h

**M1 — enum: add `VERSION_SEFI_IMAGE`.**
The fork enum has grown well past the draft's assumption (it now ends `…IDEOGRAM4, WAN_S2V,
KREA2, BOOGU_IMAGE, MINIT2I, HUNYUAN_VIDEO, LINGBOT_VIDEO, COUNT`). Append before `VERSION_COUNT`,
adjacent to the other upstream-adopted models.
Anchor (`src/model.h:57-58`):
```cpp
    VERSION_LINGBOT_VIDEO,
    VERSION_COUNT,
```
Replace with:
```cpp
    VERSION_LINGBOT_VIDEO,
    VERSION_SEFI_IMAGE,
    VERSION_COUNT,
```
> **Positional-array sync:** adding this enum value requires **Edit S1** (`model_version_to_str[]`)
> — that array is indexed by `SDVersion` and must gain a matching entry in the same slot.

**M2 — helper `sd_version_is_sefi_image`.** Match the fork's newer one-liner style.
Anchor (`src/model.h:213-215`):
```cpp
static inline bool sd_version_is_lingbot_video(SDVersion version) {
    return version == VERSION_LINGBOT_VIDEO;
}
```
Insert immediately after:
```cpp

static inline bool sd_version_is_sefi_image(SDVersion version) {
    return version == VERSION_SEFI_IMAGE;
}
```

**M3 — `sd_version_uses_flux2_vae`.** Anchor (`src/model.h:218`):
```cpp
    if (sd_version_is_flux2(version) || sd_version_is_ernie_image(version) || sd_version_is_lens(version) || sd_version_is_ideogram4(version)) {
```
Replace with:
```cpp
    if (sd_version_is_flux2(version) || sd_version_is_ernie_image(version) || sd_version_is_lens(version) || sd_version_is_ideogram4(version) || sd_version_is_sefi_image(version)) {
```

**M4 — `sd_version_is_dit`.** Anchor (`src/model.h:258-259`):
```cpp
        sd_version_is_hunyuan_video(version) ||
        sd_version_is_lingbot_video(version)) {
```
Replace with:
```cpp
        sd_version_is_hunyuan_video(version) ||
        sd_version_is_lingbot_video(version) ||
        sd_version_is_sefi_image(version)) {
```

---

### model_loader.cpp

**L1 — detection probe.** Placement matches upstream exactly: right after the flux2
`is_flux2 = true` probe and before the `single_blocks.47` probe (so the early `return` wins over
the later `is_flux2` resolution at `:681`). Anchor (`src/model_loader.cpp:554-557`):
```cpp
        if (tensor_storage.name.find("model.diffusion_model.double_stream_modulation_img.lin.weight") != std::string::npos) {
            is_flux2 = true;
        }
        if (tensor_storage.name.find("single_blocks.47.linear1.weight") != std::string::npos) {
```
Insert the sefi probe between the two `if` blocks:
```cpp
        if (tensor_storage.name.find("model.diffusion_model.double_stream_modulation_img.lin.weight") != std::string::npos) {
            is_flux2 = true;
        }
        if (tensor_storage.name.find("dual_time_embed.semantic_embedder.linear_1.weight") != std::string::npos) {
            return VERSION_SEFI_IMAGE;
        }
        if (tensor_storage.name.find("single_blocks.47.linear1.weight") != std::string::npos) {
```
(Substring `find` intentionally omits the `model.diffusion_model.` prefix — matches upstream.)

---

### include/stable-diffusion.h

**H1 — `prediction_t` enum.** The fork enum **diverges** from upstream: the fork has
`FLUX2_FLOW_PRED` in the slot upstream uses for `SEFI_FLOW_PRED`. Both are needed. Append the new
value at the end (keeps all existing numeric values stable). Anchor (`include/stable-diffusion.h:86-87`):
```cpp
    MINIT2I_FLOW_PRED,
    PREDICTION_COUNT
```
Replace with:
```cpp
    MINIT2I_FLOW_PRED,
    SEFI_FLOW_PRED,
    PREDICTION_COUNT
```
> Requires **Edit S8** (`prediction_to_str[]`) to append `"sefi_flow"` in the same slot.

---

### src/model/diffusion/flux.hpp

**F1 — include the (currently orphaned) sefi header.** Anchor (`flux.hpp:7-10`):
```cpp
#include "model/common/rope.hpp"
#include "model/diffusion/dit.hpp"
#include "model/diffusion/model.hpp"
#include "model_loader.h"
```
Replace with:
```cpp
#include "model/common/rope.hpp"
#include "model/diffusion/dit.hpp"
#include "model/diffusion/model.hpp"
#include "model/diffusion/sefi_image.hpp"
#include "model_loader.h"
```

**F2 — `FluxConfig` fields.** Anchor (`flux.hpp:26-28`):
```cpp
        SDVersion version         = VERSION_FLUX;
        bool is_chroma            = false;
        int patch_size            = 2;
```
Replace with:
```cpp
        SDVersion version         = VERSION_FLUX;
        bool is_chroma            = false;
        bool is_sefi              = false;
        int64_t semantic_channels = 0;
        float sefi_delta_t        = 0.1f;
        int patch_size            = 2;
```

**F3 — `detect_from_weights` sefi branch.** **CORRECTION vs draft:** upstream sets
`out_channels = 128 + semantic_channels` (**= 144**), NOT 128. Verbatim from `ea4e566:flux.hpp:96-110`.
Anchor — the end of the fork's flux2 block and the start of the longcat block (`flux.hpp:84-86`):
```cpp
                config.ref_index_scale  = 10.f;
                config.use_mlp_silu_act = true;
            } else if (sd_version_is_longcat(version)) {
```
Replace with:
```cpp
                config.ref_index_scale  = 10.f;
                config.use_mlp_silu_act = true;
            } else if (sd_version_is_sefi_image(version)) {
                config.is_sefi           = true;
                config.semantic_channels = 16;
                config.in_channels       = 128 + config.semantic_channels;  // 144
                config.patch_size        = 1;
                config.out_channels      = 128 + config.semantic_channels;  // 144
                config.mlp_ratio         = 3.f;
                config.theta             = 2000;
                config.axes_dim          = {32, 32, 32, 32};
                config.vec_in_dim        = 0;
                config.qkv_bias          = false;
                config.disable_bias      = true;
                config.share_modulation  = true;
                config.ref_index_scale   = 10.f;
                config.use_mlp_silu_act  = true;
            } else if (sd_version_is_longcat(version)) {
```

**F4 — block construction (`dual_time_embed`).** Anchor (`flux.hpp:897-899`):
```cpp
            if (config.is_chroma) {
                blocks["distilled_guidance_layer"] = std::make_shared<ChromaApproximator>(config.in_dim, config.hidden_size);
            } else {
```
Replace with:
```cpp
            if (config.is_chroma) {
                blocks["distilled_guidance_layer"] = std::make_shared<ChromaApproximator>(config.in_dim, config.hidden_size);
            } else if (config.is_sefi) {
                blocks["dual_time_embed"] = std::make_shared<SefiImage::SefiDualTimestepEmbeddings>(256, config.hidden_size);
            } else {
```

**F5 — forward `vec` branch.** Anchor — the end of the fork's `is_chroma` vec block and the start
of the generic `else` (`flux.hpp:1005-1008`):
```cpp
                if (y != nullptr) {
                    txt_img_mask = ggml_pad(ctx->ggml_ctx, y, static_cast<int>(img->ne[1]), 0, 0, 0);
                }
            } else {
                auto time_in = std::dynamic_pointer_cast<MLPEmbedder>(blocks["time_in"]);
```
Replace with:
```cpp
                if (y != nullptr) {
                    txt_img_mask = ggml_pad(ctx->ggml_ctx, y, static_cast<int>(img->ne[1]), 0, 0, 0);
                }
            } else if (config.is_sefi) {
                auto dual_time_embed = std::dynamic_pointer_cast<SefiImage::SefiDualTimestepEmbeddings>(blocks["dual_time_embed"]);
                auto timestep_sem    = ggml_view_1d(ctx->ggml_ctx, timesteps, 1, 0);
                auto timestep_tex    = ggml_view_1d(ctx->ggml_ctx, timesteps, 1, ggml_element_size(timesteps));
                vec                  = dual_time_embed->forward(ctx, timestep_sem, timestep_tex);
            } else {
                auto time_in = std::dynamic_pointer_cast<MLPEmbedder>(blocks["time_in"]);
```
> This is why the timesteps tensor fed to the sefi DiT must be a **2-element** `[t_sem, t_tex]`
> vector — see Edit S6 (`process_timesteps`).

**F6 — RoPE ref-index (the fork translation).** The fork's `build_graph` forces
`increase_ref_index = true` for flux2; sefi is identical. Anchor (`flux.hpp:1426-1429`):
```cpp
            std::set<int> txt_arange_dims;
            if (sd_version_is_flux2(version)) {
                txt_arange_dims    = {3};
                increase_ref_index = true;
```
Replace with:
```cpp
            std::set<int> txt_arange_dims;
            if (sd_version_is_flux2(version) || sd_version_is_sefi_image(version)) {
                txt_arange_dims    = {3};
                increase_ref_index = true;
```
> Upstream's `ref_index_mode = Rope::RefIndexMode::INCREASE` is fully represented by the fork's
> `increase_ref_index = true`. `version` and `config` are already in scope here.

---

### src/conditioning/conditioner.hpp

**C1 — LLMEmbedder arch map → QWEN3_VL.** Anchor (`conditioner.hpp:1789`):
```cpp
        } else if (sd_version_is_ideogram4(version) || sd_version_is_krea2(version) || sd_version_is_boogu_image(version) || sd_version_is_lingbot_video(version)) {
            arch = LLM::LLMArch::QWEN3_VL;
```
Replace the condition line with:
```cpp
        } else if (sd_version_is_ideogram4(version) || sd_version_is_krea2(version) || sd_version_is_boogu_image(version) || sd_version_is_lingbot_video(version) || sd_version_is_sefi_image(version)) {
            arch = LLM::LLMArch::QWEN3_VL;
```

**C2 — sefi prompt template.** Verbatim from `ea4e566:conditioner.hpp:2569-2580`, adapted to the
fork's builder (fork uses the same `conditioner_params.text` / `prompt_attn_range` names). Insert
right after the fork's `VERSION_FLUX2` branch. Anchor (`conditioner.hpp:2153-2154`):
```cpp
            prompt += "[/INST]";
        } else if (sd_version_is_ideogram4(version)) {
```
Replace with:
```cpp
            prompt += "[/INST]";
        } else if (sd_version_is_sefi_image(version)) {
            prompt_template_encode_start_idx = 0;
            min_length                       = 1024;
            out_layers                       = {9, 18, 27};

            prompt = "<|im_start|>user\n";

            prompt_attn_range.first = static_cast<int>(prompt.size());
            prompt += conditioner_params.text;
            prompt_attn_range.second = static_cast<int>(prompt.size());

            prompt += "<|im_end|>\n<|im_start|>assistant\n";
        } else if (sd_version_is_ideogram4(version)) {
```

---

### src/runtime/denoiser.hpp

**D1 — `SefiFlowDenoiser` struct.** Verbatim from `ea4e566:denoiser.hpp:1277-1349`. All dependencies
present in fork: `flux_time_shift` (`:1189`), `TIMESTEPS`, `scheduler_t`, `parse_key_value_args`
(`:452`), `parse_strict_float` (`:454`). Insert after the fork's `FluxFlowDenoiser` and before
`Flux2FlowDenoiser`. Anchor (`denoiser.hpp:1204-1206`):
```cpp
    float t_to_sigma(float t) override {
        t = t + 1;
        return flux_time_shift(shift, 1.0f, t / TIMESTEPS);
    }
};

struct Flux2FlowDenoiser : public FluxFlowDenoiser {
```
Insert the struct between the closing `};` of `FluxFlowDenoiser` and `struct Flux2FlowDenoiser`:
```cpp
    float t_to_sigma(float t) override {
        t = t + 1;
        return flux_time_shift(shift, 1.0f, t / TIMESTEPS);
    }
};

struct SefiFlowDenoiser : public FluxFlowDenoiser {
    static constexpr int kNumTrainTimesteps = 1000;
    static constexpr int kSemChannels       = 16;
    static constexpr int kTotalChannels     = 144;

    float delta_t              = 0.1f;
    float timestep_shift_alpha = 1.0f;

    std::vector<float> sem_sigmas;
    std::vector<float> tex_sigmas;
    std::vector<float> sem_timesteps;
    std::vector<float> tex_timesteps;

    SefiFlowDenoiser() = default;

    static float apply_alpha_shift(float u_unit, float alpha) {
        if (alpha == 1.0f) {
            return u_unit;
        }
        float denom = 1.0f + (alpha - 1.0f) * u_unit;
        return (alpha * u_unit) / denom;
    }

    std::vector<float> get_sigmas(uint32_t n,
                                  int image_seq_len,
                                  scheduler_t scheduler_type,
                                  SDVersion version,
                                  const char* extra_sample_args = nullptr) override {
        sem_sigmas.clear();
        tex_sigmas.clear();
        sem_timesteps.clear();
        tex_timesteps.clear();

        for (const auto& [key, value] : parse_key_value_args(extra_sample_args, "sefi scheduler arg")) {
            if (key == "sefi_alpha") {
                if (!parse_strict_float(value, timestep_shift_alpha)) {
                    LOG_WARN("ignoring invalid sefi scheduler arg '%s=%s'", key.c_str(), value.c_str());
                }
            } else if (key == "sefi_delta_t") {
                if (!parse_strict_float(value, delta_t)) {
                    LOG_WARN("ignoring invalid sefi scheduler arg '%s=%s'", key.c_str(), value.c_str());
                }
            }
        }

        for (uint32_t i = 0; i <= n; ++i) {
            float u_base    = static_cast<float>(i) / static_cast<float>(n);
            float u_shifted = apply_alpha_shift(u_base, timestep_shift_alpha);
            float u_sem_raw = u_shifted * (1.0f + delta_t);

            float u_sem = std::min(u_sem_raw, 1.0f);
            float u_tex = std::max(0.0f, std::min(u_sem_raw - delta_t, 1.0f));

            int idx_sem = std::min(kNumTrainTimesteps - 1,
                                   std::max(0, static_cast<int>(u_sem * (kNumTrainTimesteps - 1))));
            int idx_tex = std::min(kNumTrainTimesteps - 1,
                                   std::max(0, static_cast<int>(u_tex * (kNumTrainTimesteps - 1))));

            float t_sem     = static_cast<float>(kNumTrainTimesteps - idx_sem);
            float t_tex     = static_cast<float>(kNumTrainTimesteps - idx_tex);
            float sigma_sem = t_sem / static_cast<float>(kNumTrainTimesteps);
            float sigma_tex = t_tex / static_cast<float>(kNumTrainTimesteps);

            sem_timesteps.push_back(t_sem);
            tex_timesteps.push_back(t_tex);
            sem_sigmas.push_back(sigma_sem);
            tex_sigmas.push_back(sigma_tex);
        }
        LOG_DEBUG("SefiFlowDenoiser: built %u-step dual schedule (alpha=%.2f delta_t=%.2f)",
                  n, timestep_shift_alpha, delta_t);
        return tex_sigmas;
    }
};

struct Flux2FlowDenoiser : public FluxFlowDenoiser {
```

**D2 — `sample_sefi_euler` free function.** Verbatim from `ea4e566:denoiser.hpp:1523-1554`. Deps:
`denoise_cb_t` (fork `:1294`), `sd::ops::slice` (`tensor.hpp:1098`, 4-arg start/end form),
`sd::ops::slice_assign` (`tensor.hpp:1146`, `(dst,dim,start,end,src)`), `GuiderOutput.pred`. Insert
immediately before the fork's `sample_euler`. Anchor (`denoiser.hpp:1397`):
```cpp
static sd::Tensor<float> sample_euler(denoise_cb_t model,
                                      sd::Tensor<float> x,
                                      const std::vector<float>& sigmas) {
```
Insert before it:
```cpp
static sd::Tensor<float> sample_sefi_euler(SefiFlowDenoiser* sefi,
                                           denoise_cb_t model,
                                           sd::Tensor<float> x) {
    const std::vector<float>& sigma_tex_vec = sefi->tex_sigmas;
    const std::vector<float>& sigma_sem_vec = sefi->sem_sigmas;
    int steps                               = static_cast<int>(sigma_tex_vec.size()) - 1;
    for (int i = 0; i < steps; i++) {
        float sigma_tex_cur  = sigma_tex_vec[i];
        float sigma_tex_next = sigma_tex_vec[i + 1];
        float sigma_sem_cur  = sigma_sem_vec[i];
        float sigma_sem_next = sigma_sem_vec[i + 1];
        if (sigma_tex_cur <= 1e-9f) {
            continue;
        }
        auto denoised_opt = model(x, sigma_tex_cur, i + 1);
        if (denoised_opt.pred.empty()) {
            return {};
        }
        sd::Tensor<float> denoised = std::move(denoised_opt.pred);
        sd::Tensor<float> velocity = (x - denoised) / sigma_tex_cur;

        auto x_sem      = sd::ops::slice(x, 2, 0, SefiFlowDenoiser::kSemChannels);
        auto x_tex      = sd::ops::slice(x, 2, SefiFlowDenoiser::kSemChannels, SefiFlowDenoiser::kTotalChannels);
        auto vel_sem    = sd::ops::slice(velocity, 2, 0, SefiFlowDenoiser::kSemChannels);
        auto vel_tex    = sd::ops::slice(velocity, 2, SefiFlowDenoiser::kSemChannels, SefiFlowDenoiser::kTotalChannels);
        auto x_sem_next = x_sem + vel_sem * (sigma_sem_next - sigma_sem_cur);
        auto x_tex_next = x_tex + vel_tex * (sigma_tex_next - sigma_tex_cur);

        sd::ops::slice_assign(&x, 2, 0, SefiFlowDenoiser::kSemChannels, x_sem_next);
        sd::ops::slice_assign(&x, 2, SefiFlowDenoiser::kSemChannels, SefiFlowDenoiser::kTotalChannels, x_tex_next);
    }
    return x;
}

static sd::Tensor<float> sample_euler(denoise_cb_t model,
                                      sd::Tensor<float> x,
                                      const std::vector<float>& sigmas) {
```

**D3 — wire sefi into the fork's `sample_k_diffusion` dispatch.** The fork's `sample_k_diffusion`
has **no `denoiser_for_dispatch` parameter** (upstream added one). Add it (defaulted, so no other
caller breaks) and dispatch to `sample_sefi_euler` at the top. Anchor (`denoiser.hpp:2328-2336`):
```cpp
static sd::Tensor<float> sample_k_diffusion(sample_method_t method,
                                            denoise_cb_t model,
                                            sd::Tensor<float> x,
                                            std::vector<float> sigmas,
                                            std::shared_ptr<RNG> rng,
                                            float eta,
                                            bool is_flow_denoiser,
                                            const char* extra_sample_args) {
    SamplerExtraArgs extra_args = parse_key_value_args(extra_sample_args, "extra sample arg");
    switch (method) {
```
Replace with:
```cpp
static sd::Tensor<float> sample_k_diffusion(sample_method_t method,
                                            denoise_cb_t model,
                                            sd::Tensor<float> x,
                                            std::vector<float> sigmas,
                                            std::shared_ptr<RNG> rng,
                                            float eta,
                                            bool is_flow_denoiser,
                                            const char* extra_sample_args,
                                            std::shared_ptr<Denoiser> denoiser_for_dispatch = nullptr) {
    if (denoiser_for_dispatch) {
        if (auto sefi = std::dynamic_pointer_cast<SefiFlowDenoiser>(denoiser_for_dispatch)) {
            return sample_sefi_euler(sefi.get(), model, std::move(x));
        }
    }
    SamplerExtraArgs extra_args = parse_key_value_args(extra_sample_args, "extra sample arg");
    switch (method) {
```

---

### src/stable-diffusion.cpp

**S1 — `model_version_to_str[]` positional sync** (required by Edit M1). Anchor (`:114-115`):
```cpp
    "LingBot Video",
};
```
Replace with:
```cpp
    "LingBot Video",
    "SEFI Image",
};
```

**S2 — diffusion-model dispatch: merge sefi into the flux2 branch.** The fork's flux2 branch body
is exactly what sefi needs (`backend_for`/`params_backend_for`, `LLMEmbedder` + `Flux::FluxRunner`,
no `model_manager`). Anchor (`:818`):
```cpp
            } else if (sd_version_is_flux2(version)) {
                bool is_chroma   = false;
```
Replace the condition line with:
```cpp
            } else if (sd_version_is_flux2(version) || sd_version_is_sefi_image(version)) {
                bool is_chroma   = false;
```
> The `FluxRunner` last ctor arg stays `sd_ctx_params->chroma_use_dit_mask` (fork's flux2 value).

**S3 — pred_type selection.** Anchor (`:1714-1716`):
```cpp
                } else if (sd_version_is_flux2(version)) {
                    pred_type = FLUX2_FLOW_PRED;
                } else if (sd_version_is_minit2i(version)) {
```
Replace with:
```cpp
                } else if (sd_version_is_flux2(version)) {
                    pred_type = FLUX2_FLOW_PRED;
                } else if (sd_version_is_sefi_image(version)) {
                    pred_type = SEFI_FLOW_PRED;
                } else if (sd_version_is_minit2i(version)) {
```

**S4 — denoiser switch: add `SEFI_FLOW_PRED` case.** Anchor — the fork's flux2 case
(`:1750-1754`):
```cpp
                case FLUX2_FLOW_PRED: {
                    LOG_INFO("running in Flux2 FLOW mode");
                    denoiser = std::make_shared<Flux2FlowDenoiser>();
                    break;
                }
                case MINIT2I_FLOW_PRED: {
```
Replace with:
```cpp
                case FLUX2_FLOW_PRED: {
                    LOG_INFO("running in Flux2 FLOW mode");
                    denoiser = std::make_shared<Flux2FlowDenoiser>();
                    break;
                }
                case SEFI_FLOW_PRED: {
                    LOG_INFO("running in SEFI FLOW mode");
                    denoiser = std::make_shared<SefiFlowDenoiser>();
                    break;
                }
                case MINIT2I_FLOW_PRED: {
```

**S5 — `FluxDiffusionExtra` grouping.** Anchor (`:3166`):
```cpp
                } else if (sd_version_is_flux(version) || sd_version_is_flux2(version) || sd_version_is_longcat(version)) {
                    diffusion_params.extra = FluxDiffusionExtra{&guidance_tensor,
```
Replace the condition line with:
```cpp
                } else if (sd_version_is_flux(version) || sd_version_is_flux2(version) || sd_version_is_longcat(version) || sd_version_is_sefi_image(version)) {
                    diffusion_params.extra = FluxDiffusionExtra{&guidance_tensor,
```

**S6 — `process_timesteps`: dual sem/tex schedule.** The fork's `process_timesteps` has **no `step`
param** (upstream's does). Add one and a sefi branch that returns the 2-element `[t_sem, t_tex]`
vector for the current step. `denoiser` is a class member (`:334`), and `SefiFlowDenoiser` is
visible via `denoiser.hpp`.

S6a — signature. Anchor (`:2462-2464`):
```cpp
    std::vector<float> process_timesteps(const std::vector<float>& timesteps,
                                         const sd::Tensor<float>& init_latent,
                                         const sd::Tensor<float>& denoise_mask) {
        if (diffusion_model->get_desc() == "Wan2.2-TI2V-5B" || sd_version_is_longcat_avatar(version) ||
```
Replace with:
```cpp
    std::vector<float> process_timesteps(const std::vector<float>& timesteps,
                                         const sd::Tensor<float>& init_latent,
                                         const sd::Tensor<float>& denoise_mask,
                                         int step = 0) {
        if (auto sefi_denoiser = std::dynamic_pointer_cast<SefiFlowDenoiser>(denoiser)) {
            int sched_idx = step > 0 ? step - 1 : 0;
            if (sched_idx >= static_cast<int>(sefi_denoiser->tex_timesteps.size())) {
                sched_idx = static_cast<int>(sefi_denoiser->tex_timesteps.size()) - 1;
            }
            return {sefi_denoiser->sem_timesteps[sched_idx],
                    sefi_denoiser->tex_timesteps[sched_idx]};
        }
        if (diffusion_model->get_desc() == "Wan2.2-TI2V-5B" || sd_version_is_longcat_avatar(version) ||
```

S6b — call site. Anchor (`:3079`):
```cpp
                timesteps_vec = process_timesteps(timesteps_vec, init_latent, denoise_mask);
```
Replace with:
```cpp
                timesteps_vec = process_timesteps(timesteps_vec, init_latent, denoise_mask, step);
```
> `step` is in scope in the denoise callback (used at `:3089`, `:3129`, `:3136`). The sample loop
> passes step = `i + 1` (see `sample_sefi_euler`), so `sched_idx = i` — the correct schedule index.

**S7 — pass the denoiser into `sample_k_diffusion`** (pairs with Edit D3). Anchor (`:3558`):
```cpp
        auto x0_opt = sample_k_diffusion(method, denoise, x_t, sigmas, sampler_rng, eta, is_flow_denoiser, extra_sample_args);
```
Replace with:
```cpp
        auto x0_opt = sample_k_diffusion(method, denoise, x_t, sigmas, sampler_rng, eta, is_flow_denoiser, extra_sample_args, denoiser);
```
> `denoiser` is the class member; in scope here.

**S8 — `prediction_to_str[]` positional sync** (required by Edit H1). Anchor (`:4058-4060`):
```cpp
    "flux2_flow",
    "minit2i_flow",
};
```
Replace with:
```cpp
    "flux2_flow",
    "minit2i_flow",
    "sefi_flow",
};
```

**S9 — get_latent_channel = 144.** sefi is in `sd_version_uses_flux2_vae`, which would wrongly give
128; the sefi special-case must precede it. Anchor (`:3629-3632`):
```cpp
            } else if (sd_version_is_minit2i(version)) {
                latent_channel = 3;
            } else if (sd_version_uses_flux2_vae(version)) {
                latent_channel = 128;
```
Replace with:
```cpp
            } else if (sd_version_is_minit2i(version)) {
                latent_channel = 3;
            } else if (sd_version_is_sefi_image(version)) {
                latent_channel = 144;
            } else if (sd_version_uses_flux2_vae(version)) {
                latent_channel = 128;
```

---

## Notes / flags (non-blocking)

- **VAE:** none needed. sefi is a member of `sd_version_uses_flux2_vae` (Edit M3), so the existing
  flux2-VAE factory path handles it. (The DiT latent is 144ch; the VAE still operates on the 128
  texture channels — Edit S9 keeps the DiT/init-latent bookkeeping at 144.)
- **`get_default_ref_image_preset` (upstream `sd.cpp:2907`, returns `"flux2"` for sefi): NO fork
  equivalent.** The fork has no such function and no `"flux_kontext"`/`"longcat"` preset strings
  (`grep` = 0 hits); ref-image resize was unified differently in this merge (commit `c58a01d`
  "unify VLM ref-image resize"). **Not a compile blocker.** If sefi is exercised with reference
  images, verify the fork's VLM ref-image resize path handles it — but text-to-image sefi does not
  touch this.
- **`sd_version_supports_ref_latent_img_cfg` (`sd.cpp:140`):** upstream does **not** add sefi here,
  so it is intentionally omitted.
- Upstream's other flux2/sefi groupings all map to the edits above; no additional
  `|| sd_version_is_sefi_image` site was found unaccounted for (checked every `sefi` hit in
  `ea4e566:src/stable-diffusion.cpp`: 1046→S2, 1692/1728→S3/S4, 2045→S6, 2543→S5, 2761→S9,
  2907→flagged, 3149→S8).

## BLOCKER SUMMARY

**No compile blockers.** Every symbol the port needs is present in the fork tree:

- `sefi_image.hpp` is byte-identical to upstream and all its primitives exist (audit table above);
  it only needed wiring (`#include`, Edit F1).
- The RoPE concern is resolved cleanly: the fork's `increase_ref_index=true` path **is** upstream's
  `RefIndexMode::INCREASE`, and sefi behaves identically to flux2 — a one-line condition extension
  (Edit F6). No `RefIndexMode` enum is referenced anywhere in the ported code.
- `SefiDualTimestepEmbeddings` compiles against fork primitives (`Linear`, `ggml_silu_inplace`,
  `ggml_ext_timestep_embedding`, `ggml_concat`, `GGMLRunnerContext`).
- Denoiser deps (`flux_time_shift`, `parse_key_value_args`, `parse_strict_float`, `sd::ops::slice`,
  `sd::ops::slice_assign`, `denoise_cb_t`/`GuiderOutput.pred`) all present.

**Fork-divergence adaptations made (not upstream-verbatim):**
1. Enum: fork's `FLUX2_FLOW_PRED` occupies upstream's `SEFI_FLOW_PRED` slot → `SEFI_FLOW_PRED`
   appended at end of `prediction_t` + `prediction_to_str[]` (Edits H1/S8).
2. Dispatch/pred/denoiser: fork keeps a separate `Flux2FlowDenoiser`; sefi added as its own
   parallel path rather than replacing flux2 (Edits S2–S4).
3. `sample_k_diffusion`: fork lacks the `denoiser_for_dispatch` param → added defaulted + threaded
   from the one call site (Edits D3/S7).
4. `process_timesteps`: fork lacks the `step` param → added defaulted (Edits S6a/S6b).
5. `FluxConfig::detect_from_weights`: **draft error corrected** — `out_channels = 144`
   (`128 + semantic_channels`), not 128; `sefi_delta_t` field added (Edits F2/F3).

**One thing to verify at runtime (not a build blocker):** sefi + reference-image usage, since the
fork has no `get_default_ref_image_preset`. Text-to-image sefi is fully wired by the edits above.
