# QWEN_IMAGE_LAYERED — apply-ready integration edits

Fork worktree: `/home/dbrain/dev/longcat-avatar-mergewt` (branch `upstream-merge-2026-07-19`)
Upstream tip `ea4e566`, merge base `19bdfe2`, fork HEAD. No draft spec existed — this is built from upstream.
Every anchor below was `grep`/`Read`-verified against the CURRENT fork tree. **No source was edited; nothing built.**

---

## 0. TL;DR verdict (read first)

Qwen-Image-Layered = a Qwen-Image DiT variant that emits an **RGBA (4-channel)** image plus N transparent
**layers**, stacked on a temporal/frame axis, each layer decoded independently through the Wan-VAE running in a
new **2-D mode**. Three independent capabilities are needed:

| Capability | Portability to fork | Where |
|---|---|---|
| **A. Model plumbing** (enum, fold into `sd_version_is_qwen_image`, detection, `model_version_to_str`) | **Clean.** Membership fold means the VAE-factory + DiT dispatch + `is_dit`/`uses_wan_vae` all light up automatically. | §1–§4 |
| **B. DiT `addition_t_cond`** (per-layer time-conditioning embedding) | **Clean, gated additive** — `use_additional_t_cond` defaults false, zero effect on existing `VERSION_QWEN_IMAGE`. | §5 |
| **C. Wan-VAE 2-D mode** (`Conv2dBut3d`, `is_2D` plumbed through Residual/Encoder/Decoder/WanVAE/Runner) | **Portable as a fully `is_2D`-gated additive layer, default false → 3-D video path byte-untouched.** This is the core request and it is achievable without a rewrite. | §7 |
| **D. Rope temporal-ids for layers** | **Partial gap.** Fork's `gen_qwen_image_pe` has no `t` axis and fork's `gen_vid_ids` lacks the `scale_rope` param upstream relies on. Additive `t=1`-default path is possible but layer-axis fidelity is **unverifiable without a checkpoint.** | §6 |
| **E. End-to-end layered orchestration** (5-D init latent for layers, per-layer decode loop, `get_image_channels()==4`, `qwen_image_layers` request field/API) | **The real blocker.** The fork's generate/decode orchestration has diverged massively (video-first) and has **no upstream-parallel anchor** for the image layered path. Requires new, hand-placed code. | §8 + §10 |

**Bottom line:** the VAE 2-D mode (the explicitly-hard, explicitly-requested piece) ports cleanly as an additive
`is_2D`-gated layer with **no risk to the fork's Wan/video 3-D VAE path**. The genuine blocker is E — the
layered generation/decode *pipeline* in `stable-diffusion.cpp`, which has no fork anchor and no reference
checkpoint to validate against. Land A–C first; treat D/E as a separate, checkpoint-gated milestone.

---

## 1. `src/model.h` — enum + membership fold

### 1a. Add the version (append at end to preserve `model_version_to_str` index alignment)

Anchor (unique, fork enum tail):

```cpp
    VERSION_HUNYUAN_VIDEO,
    VERSION_LINGBOT_VIDEO,
    VERSION_COUNT,
```

Replace with:

```cpp
    VERSION_HUNYUAN_VIDEO,
    VERSION_LINGBOT_VIDEO,
    VERSION_QWEN_IMAGE_LAYERED,
    VERSION_COUNT,
```

> The fork appends new versions before `VERSION_COUNT` (that is how MiniT2I/Hunyuan/LingBot were added) so the
> `model_version_to_str[]` array stays index-aligned. Do **not** insert next to `VERSION_QWEN_IMAGE` — that would
> shift every later ordinal and desync the string table.

### 1b. Fold into `sd_version_is_qwen_image` (this is the elegant path — do NOT add a separate helper)

Upstream has **no** `sd_version_is_qwen_image_layered()`; it folds the layered version into
`sd_version_is_qwen_image()` so it inherits `sd_version_is_dit()` and `sd_version_uses_wan_vae()` and the whole
dispatch for free. Match that.

Anchor (unique, fork line 131–136):

```cpp
static inline bool sd_version_is_qwen_image(SDVersion version) {
    if (version == VERSION_QWEN_IMAGE) {
        return true;
    }
    return false;
}
```

Replace with:

```cpp
static inline bool sd_version_is_qwen_image(SDVersion version) {
    if (version == VERSION_QWEN_IMAGE || version == VERSION_QWEN_IMAGE_LAYERED) {
        return true;
    }
    return false;
}
```

**Consequences of the fold (verified against fork tree — no extra edits needed at these sites):**
- `sd_version_is_dit()` (fork `model.h:239`, line 245 already lists `sd_version_is_qwen_image`) → layered is DiT. ✔
- DiT dispatch `else if (sd_version_is_qwen_image(version))` (fork `stable-diffusion.cpp:893`) → picks
  `Qwen::QwenImageRunner`, passing `version` through. ✔
- VAE factory `sd_version_is_qwen_image(version)` branch (fork `stable-diffusion.cpp:1096`) → picks
  `WAN::WanVAERunner`, passing `version` through. ✔
- Preprocessing factor `sd_version_is_qwen_image(...) ? 32 : 16` (fork `stable-diffusion.cpp:6900`) → 32. ✔

> Optional helper for readability at the few `version == VERSION_QWEN_IMAGE_LAYERED` call-sites below: none is
> required — use the raw enum compare exactly as upstream does.

---

## 2. `src/stable-diffusion.cpp` — `model_version_to_str[]`

Anchor (unique, fork array tail, currently ends at line ~113):

```cpp
    "HunyuanVideo 1.5",
    "LingBot Video",
};
```

Replace with:

```cpp
    "HunyuanVideo 1.5",
    "LingBot Video",
    "Qwen Image Layered",
};
```

(index-aligned with `VERSION_QWEN_IMAGE_LAYERED` appended in §1a.)

---

## 3. `src/model_loader.cpp` — detection

Upstream discriminates layered from plain Qwen-Image by the presence of
`model.diffusion_model.time_text_embed.addition_t_embedding.weight` on the **same** trigger tensor
(`...transformer_blocks.0.img_mod.1.weight`). Find the fork's existing Qwen-Image detection block and split it.

Locate the fork's current Qwen-Image detection (grep `img_mod.1.weight` in `src/model_loader.cpp`). It currently
looks like:

```cpp
        if (tensor_storage.name.find("model.diffusion_model.transformer_blocks.0.img_mod.1.weight") != std::string::npos) {
            return VERSION_QWEN_IMAGE;
        }
```

Replace with (mirrors upstream `model_loader.cpp:500`):

```cpp
        if (tensor_storage.name.find("model.diffusion_model.transformer_blocks.0.img_mod.1.weight") != std::string::npos) {
            if (tensor_storage_map.find("model.diffusion_model.time_text_embed.addition_t_embedding.weight") != tensor_storage_map.end()) {
                return VERSION_QWEN_IMAGE_LAYERED;
            }
            return VERSION_QWEN_IMAGE;
        }
```

> Verify the exact current fork wording of the trigger `find(...)` string before editing (the fork may spell the
> prefix slightly differently); keep that spelling and only add the inner `addition_t_embedding` discriminator.
> `tensor_storage_map` is the in-scope map variable used by the surrounding detection lambda (same as upstream).

---

## 4. Backup / nav / misc

No table or backup changes — `SDVersion` is runtime-detected, never serialized. No web/nav impact (this is the C++
engine, not kobbler Rust/web). Nothing else in §1's fold requires companion edits.

---

## 5. `src/model/diffusion/qwen_image.hpp` — `addition_t_cond` (gated additive; default OFF)

All of this is guarded by `config.use_additional_t_cond` (default `false`) → **zero effect on the existing
`VERSION_QWEN_IMAGE` path**. Translate upstream's diff to fork convention: **keep the fork's
`(backend, params_backend, …)` runner ctor and the fork's 3-arg `GGMLRunner::compute<float>(get_graph, n_threads,
false)`; do NOT adopt upstream's `RunnerWeightManager`/`ModelManager`/`RefIndexMode`/5-D refactor here.**

### 5a. `QwenImageConfig` — new field

Anchor (unique, fork `qwen_image.hpp`, the config struct member list — grep `bool zero_cond_t`):

```cpp
        bool zero_cond_t            = false;
```

Insert **after** it:

```cpp
        bool use_additional_t_cond  = false;
```

### 5b. `QwenTimestepProjEmbeddings` — optional `addition_t_embedding` sub-block

Anchor (unique, fork struct — grep `struct QwenTimestepProjEmbeddings`). Replace the whole class body's ctor +
`forward` (currently the single-arg form) with the layered-aware version:

```cpp
    struct QwenTimestepProjEmbeddings : public GGMLBlock {
    protected:
        bool use_additional_t_cond = false;

    public:
        QwenTimestepProjEmbeddings(int64_t embedding_dim, bool use_additional_t_cond = false)
            : use_additional_t_cond(use_additional_t_cond) {
            blocks["timestep_embedder"] = std::shared_ptr<GGMLBlock>(new TimestepEmbedding(256, embedding_dim));
            if (use_additional_t_cond) {
                blocks["addition_t_embedding"] = std::shared_ptr<GGMLBlock>(new Embedding(2, embedding_dim));
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* timesteps,
                             ggml_tensor* addition_t_cond = nullptr) {
            auto timestep_embedder = std::dynamic_pointer_cast<TimestepEmbedding>(blocks["timestep_embedder"]);
            auto timesteps_proj    = ggml_ext_timestep_embedding(ctx->ggml_ctx, timesteps, 256, 10000, 1.f);
            auto timesteps_emb     = timestep_embedder->forward(ctx, timesteps_proj);
            if (use_additional_t_cond) {
                GGML_ASSERT(addition_t_cond != nullptr);
                auto addition_t_embedding = std::dynamic_pointer_cast<Embedding>(blocks["addition_t_embedding"]);
                auto addition_t_emb       = addition_t_embedding->forward(ctx, addition_t_cond);
                timesteps_emb             = ggml_add(ctx->ggml_ctx, timesteps_emb, addition_t_emb);
            }
            return timesteps_emb;
        }
    };
```

Verified fork primitives: `Embedding` (`ggml_extend.hpp:6095`), `ggml_ext_timestep_embedding`, `TimestepEmbedding`
(already used by the fork's existing single-arg ctor). The default-argument `addition_t_cond = nullptr` keeps every
existing `forward(ctx, timestep)` call-site valid.

### 5c. `QwenImageModel` ctor — pass the flag

Anchor (unique, fork `qwen_image.hpp:404`):

```cpp
            blocks["time_text_embed"] = std::shared_ptr<GGMLBlock>(new QwenTimestepProjEmbeddings(inner_dim));
```

Replace with:

```cpp
            blocks["time_text_embed"] = std::shared_ptr<GGMLBlock>(new QwenTimestepProjEmbeddings(inner_dim, config.use_additional_t_cond));
```

### 5d. `QwenImageModel::forward_orig` — thread `addition_t_cond`

Anchor (unique, fork `qwen_image.hpp:424`) — add the param:

```cpp
        ggml_tensor* forward_orig(GGMLRunnerContext* ctx,
                                  ggml_tensor* x,
                                  ggml_tensor* timestep,
                                  ggml_tensor* context,
                                  ggml_tensor* pe,
                                  ggml_tensor* modulate_index = nullptr) {
```

Replace the signature with (insert `addition_t_cond` after `timestep`):

```cpp
        ggml_tensor* forward_orig(GGMLRunnerContext* ctx,
                                  ggml_tensor* x,
                                  ggml_tensor* timestep,
                                  ggml_tensor* addition_t_cond,
                                  ggml_tensor* context,
                                  ggml_tensor* pe,
                                  ggml_tensor* modulate_index = nullptr) {
```

Then the two `time_text_embed->forward(...)` calls inside it (fork lines ~438 and ~440). Anchor:

```cpp
            auto t_emb = time_text_embed->forward(ctx, timestep);
            if (config.zero_cond_t) {
                auto t_emb_0 = time_text_embed->forward(ctx, ggml_ext_zeros_like(ctx->ggml_ctx, timestep));
```

Replace with:

```cpp
            auto t_emb = time_text_embed->forward(ctx, timestep, addition_t_cond);
            if (config.zero_cond_t) {
                auto t_emb_0 = time_text_embed->forward(ctx, ggml_ext_zeros_like(ctx->ggml_ctx, timestep), addition_t_cond);
```

### 5e. `QwenImageModel::forward` — thread `addition_t_cond`

Anchor (unique, fork `qwen_image.hpp:468`):

```cpp
        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* timestep,
                             ggml_tensor* context,
                             ggml_tensor* pe,
                             std::vector<ggml_tensor*> ref_latents = {},
                             ggml_tensor* modulate_index           = nullptr) {
```

Replace the signature (insert `addition_t_cond` after `timestep`):

```cpp
        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* timestep,
                             ggml_tensor* addition_t_cond,
                             ggml_tensor* context,
                             ggml_tensor* pe,
                             std::vector<ggml_tensor*> ref_latents = {},
                             ggml_tensor* modulate_index           = nullptr) {
```

And the single call to `forward_orig` inside it. Anchor (unique, fork line ~500):

```cpp
            auto out = forward_orig(ctx, img, timestep, context, pe, modulate_index);  // [N, h_len*w_len, ph*pw*C]
```

Replace with:

```cpp
            auto out = forward_orig(ctx, img, timestep, addition_t_cond, context, pe, modulate_index);  // [N, h_len*w_len, ph*pw*C]
```

> **Scope decision (important):** the fork's `QwenImageModel::forward` is the **2-D (4-D tensor)** form
> (`x: [N,C,H,W]`, `pad_and_patchify`, `unpatchify_and_crop`). Do **NOT** adopt upstream's 5-D/time-axis rewrite of
> this method here. The layered latent's per-layer decode is handled by slicing layers **before** the DiT step at
> the orchestration layer (§8), so each DiT forward still sees a single 4-D image. This keeps §5 a pure additive
> conditioning change and avoids the entangled 5-D patchify refactor. See §6/§10 for why the temporal-RoPE nuance
> is deferred.

### 5f. `QwenImageRunner` ctor — store `version`, enable layered config

Anchor (unique, fork `qwen_image.hpp:520`):

```cpp
        QwenImageRunner(ggml_backend_t backend,
                        ggml_backend_t params_backend,
                        const String2TensorStorage& tensor_storage_map = {},
                        const std::string prefix                       = "",
                        SDVersion version                              = VERSION_QWEN_IMAGE,
                        bool zero_cond_t                               = false)
            : DiffusionModelRunner(backend, params_backend, prefix),
              config(QwenImageConfig::detect_from_weights(tensor_storage_map, prefix)) {
            config.zero_cond_t = config.zero_cond_t || zero_cond_t;
            qwen_image         = QwenImageModel(config);
            qwen_image.init(params_ctx, tensor_storage_map, prefix);
        }
```

Replace with (adds `version(version)` to the init list — **the fork currently leaves the `version` member
uninitialised**, a latent bug; layered needs it — and sets `use_additional_t_cond`):

```cpp
        QwenImageRunner(ggml_backend_t backend,
                        ggml_backend_t params_backend,
                        const String2TensorStorage& tensor_storage_map = {},
                        const std::string prefix                       = "",
                        SDVersion version                              = VERSION_QWEN_IMAGE,
                        bool zero_cond_t                               = false)
            : DiffusionModelRunner(backend, params_backend, prefix),
              config(QwenImageConfig::detect_from_weights(tensor_storage_map, prefix)),
              version(version) {
            config.zero_cond_t = config.zero_cond_t || zero_cond_t;
            if (version == VERSION_QWEN_IMAGE_LAYERED) {
                config.use_additional_t_cond = true;
            }
            qwen_image = QwenImageModel(config);
            qwen_image.init(params_ctx, tensor_storage_map, prefix);
        }
```

> The `SDVersion version;` member already exists (fork `qwen_image.hpp:518`); only the initialiser was missing.

### 5g. `QwenImageRunner::build_graph` — create `addition_t_cond`, forward it

Anchor (unique, inside fork `build_graph`, just before the `pe_vec = Rope::gen_qwen_image_pe(...)` call, fork line
~558). Insert an `addition_t_cond` set-up block **before** the `pe_vec = ...` line:

```cpp
            ggml_tensor* addition_t_cond = nullptr;
            if (version == VERSION_QWEN_IMAGE_LAYERED) {
                int batch_size = static_cast<int>(x->ne[3]);
                additional_t_cond_vec.assign(static_cast<size_t>(batch_size), 0);
                addition_t_cond = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_I32, batch_size);
                set_backend_tensor_data(addition_t_cond, additional_t_cond_vec.data());
            }
```

Add the backing member. Anchor (unique, fork `qwen_image.hpp:516`):

```cpp
        std::vector<float> modulate_index_vec;
```

Insert **after** it:

```cpp
        std::vector<int32_t> additional_t_cond_vec;
```

And the single `qwen_image.forward(...)` call inside `build_graph` (fork line ~677). Anchor:

```cpp
            ggml_tensor* out = qwen_image.forward(&runner_ctx,
                                                  x,
                                                  timesteps,
                                                  context,
                                                  pe,
                                                  ref_latents,
                                                  modulate_index);
```

Replace with (insert `addition_t_cond` after `timesteps`):

```cpp
            ggml_tensor* out = qwen_image.forward(&runner_ctx,
                                                  x,
                                                  timesteps,
                                                  addition_t_cond,
                                                  context,
                                                  pe,
                                                  ref_latents,
                                                  modulate_index);
```

Verified: `set_backend_tensor_data`, `compute_ctx`, `ggml_new_tensor_1d` all already used in this exact fork method
(the `pe`/`modulate_index` set-up right above uses them). `additional_t_cond_vec` is all-zeros of length
`batch_size` — matches upstream (`addition_t_cond` selects `Embedding(2, …)` row 0 for the "generated" layer).

---

## 6. `src/model/common/rope.hpp` — temporal-RoPE for layers (PARTIAL GAP — read the risk)

**This is where the fork lacks a primitive.** Upstream's layered path routes `gen_qwen_image_ids` through
`gen_vid_ids(t, …, scale_rope=true)` so the layer/frame axis gets a temporal position id, and adds the
`RefIndexMode` enum. The fork's `gen_qwen_image_pe`/`gen_qwen_image_ids` (line 479/511) have **no `t` axis**, use
`bool increase_ref_index`, and route through `gen_flux_img_ids(...)`; the fork's `gen_vid_ids` (line 682)
**lacks the `scale_rope` parameter** upstream added in this same commit.

**Because §5f decodes layers one at a time (each DiT forward sees a single 4-D image, `t`-axis length 1), the
temporal RoPE id for the layer is 0 in both formulations** — i.e. for the single-image-per-forward strategy the
fork's existing `gen_qwen_image_pe` is already correct and **no rope change is required for §5's design.**

**Do NOT change `rope.hpp` for the §5 design.** Only if a future implementer chooses upstream's *batched* 5-D
single-forward strategy (all layers in one DiT call, layers on the time axis) does the rope temporal id become
load-bearing, and then the following is needed — recorded here so the gap is explicit:

- Add `int t` as the leading param of `gen_qwen_image_ids` + `gen_qwen_image_pe` (default `1`); update the one fork
  call-site (`qwen_image.hpp` build_graph) to pass `time_len`.
- For `t == 1`, keep the current `gen_flux_img_ids(h, w, patch_size, bs, 3, 0, 0, 0, /*scale_rope=*/true)` call
  **verbatim** (behaviour-preserving for `VERSION_QWEN_IMAGE`).
- For `t > 1`, use a temporal id generator equivalent to upstream's
  `gen_vid_ids(t, h, w, 1, patch, patch, bs, 0, 0, 0, /*scale_rope=*/true)` — which requires **adding a
  `bool scale_rope = false` param to the fork's `gen_vid_ids`** (it currently has none) that applies
  `h_offset -= h_len/2; w_offset -= w_len/2` before building ids. Gate it (`scale_rope` default false) so the
  fork's lingbot/hunyuan callers are unaffected.

**RISK:** with no layered checkpoint available, the numeric equivalence of the fork's `gen_flux_img_ids` axis-0 vs
upstream's `gen_vid_ids` axis-0 for `t==1`, and the `txt_id_start` tweak upstream also made
(`std::max(h_len,w_len)/2` vs the fork's `std::max(h_len,w_len)`, plus a different linspace endpoint), **cannot be
validated**. The `txt_id_start` change is a *shared*-code numeric change to the existing Qwen-Image path — **do not
adopt it** (it would perturb the working `VERSION_QWEN_IMAGE`); accept the layered text-token positions may differ
slightly from reference. Flag for checkpoint-time validation.

---

## 7. `src/model/vae/wan_vae.hpp` — Wan-VAE 2-D mode (CORE; fully `is_2D`-gated, default false)

**Design contract: every new behaviour is guarded by `is_2D`, default `false`. When `is_2D == false` the
constructed blocks and the forward paths are byte-identical to today's fork 3-D video VAE. The fork's Wan/video
path is not disturbed.** All the fork's decode optimizations (`WAN_VAE_RMS_KERNEL`, `WAN_VAE_RMS_CF`,
`WAN_VAE_F16`, `feat_cache` causal streaming, `dec_dim_override`) live entirely on the `is_2D == false` side and
are left untouched.

Fork-convention deltas vs the upstream diff:
- **Keep** the fork's `WanVAERunner(backend, params_backend, …)` ctor and `get_param_tensors(tensors, prefix)`
  override — do **NOT** adopt upstream's `RunnerWeightManager`/`VAE(version, backend, prefix, weight_manager)` or
  the prefix-less `get_param_tensors`.
- **Do NOT** adopt upstream's `ggml_ext_pad_ext(ctx->ggml_ctx, ctx->backend, …)` / `ggml_ext_conv_3d(…, ctx->backend, …)`
  backend-arg change — the fork's signatures (`ggml_extend.hpp:1201`, `:1307`) take **no** backend arg, and the new
  2-D convs go through `Conv2d`/`Conv2dBut3d` which never call those helpers anyway.

### 7a. `Conv2dBut3d` — new block (ports verbatim; fork has `Conv2d`)

`Conv2d` is defined in the fork at `ggml_extend.hpp:3443` with `forward(GGMLRunnerContext* ctx, ggml_tensor* x)` —
identical signature to upstream, so this ports 1:1. Anchor (unique, fork `wan_vae.hpp` — the class **after**
`CausalConv3d`, grep `class Resample : public GGMLBlock`, insert **before** it):

```cpp
    class Conv2dBut3d : public Conv2d {
    public:
        using Conv2d::Conv2d;

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            ggml_tensor* x_swapped = ggml_permute(ctx->ggml_ctx, x, 0, 1, 3, 2);
            x_swapped              = ggml_cont(ctx->ggml_ctx, x_swapped);

            ggml_tensor* out = Conv2d::forward(ctx, x_swapped);

            ggml_tensor* out_swapped = ggml_permute(ctx->ggml_ctx, out, 0, 1, 3, 2);
            out_swapped              = ggml_cont(ctx->ggml_ctx, out_swapped);

            return out_swapped;
        }
    };
```

> Semantics: a `[C, T, H, W]` tensor has its T-axis swapped out (`permute 0,1,3,2` → `[C, W? ]` — actually swaps
> ne[2]↔ne[3], moving the batch/frame axis so the ordinary 2-D conv sees `[W,H,·,·]` per plane), a plain 2-D conv
> is run, then the axes are swapped back. The 2-D VAE checkpoint stores `residual.*`, `conv1`, `head.2`, `conv2`,
> `shortcut` weights as 4-D `[out,in,kh,kw]` (Conv2d layout) rather than 5-D causal-conv, which is exactly what the
> `is_2D` detection in §7f keys on.

### 7b. `ResidualBlock` — `is_2D` ctor + forward branches

Ctor anchor (unique, fork `wan_vae.hpp:552`):

```cpp
        ResidualBlock(int64_t in_dim, int64_t out_dim)
            : in_dim(in_dim), out_dim(out_dim) {
            blocks["residual.0"] = std::shared_ptr<GGMLBlock>(new RMS_norm(in_dim));
            // residual.1 is nn.SiLU()
            blocks["residual.2"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(in_dim, out_dim, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}));
            blocks["residual.3"] = std::shared_ptr<GGMLBlock>(new RMS_norm(out_dim));
            // residual.4 is nn.SiLU()
            // residual.5 is nn.Dropout()
            blocks["residual.6"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(out_dim, out_dim, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}));
            if (in_dim != out_dim) {
                blocks["shortcut"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(in_dim, out_dim, {1, 1, 1}));
            }
        }
```

Replace with (add `bool is_2D = false` param + member, gate conv construction):

```cpp
        ResidualBlock(int64_t in_dim, int64_t out_dim, bool is_2D = false)
            : in_dim(in_dim), out_dim(out_dim), is_2D(is_2D) {
            blocks["residual.0"] = std::shared_ptr<GGMLBlock>(new RMS_norm(in_dim));
            // residual.1 is nn.SiLU()
            if (is_2D) {
                blocks["residual.2"] = std::shared_ptr<GGMLBlock>(new Conv2dBut3d(in_dim, out_dim, {3, 3}, {1, 1}, {1, 1}));
            } else {
                blocks["residual.2"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(in_dim, out_dim, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}));
            }
            blocks["residual.3"] = std::shared_ptr<GGMLBlock>(new RMS_norm(out_dim));
            // residual.4 is nn.SiLU()
            // residual.5 is nn.Dropout()
            if (is_2D) {
                blocks["residual.6"] = std::shared_ptr<GGMLBlock>(new Conv2dBut3d(out_dim, out_dim, {3, 3}, {1, 1}, {1, 1}));
            } else {
                blocks["residual.6"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(out_dim, out_dim, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}));
            }
            if (in_dim != out_dim) {
                if (is_2D) {
                    blocks["shortcut"] = std::shared_ptr<GGMLBlock>(new Conv2dBut3d(in_dim, out_dim, {1, 1}));
                } else {
                    blocks["shortcut"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(in_dim, out_dim, {1, 1, 1}));
                }
            }
        }
```

Add the member. Anchor (unique, fork `wan_vae.hpp:548-549`):

```cpp
        int64_t in_dim;
        int64_t out_dim;
```

Replace with:

```cpp
        int64_t in_dim;
        int64_t out_dim;
        bool is_2D = false;
```

Forward — two edits. The fork's `ResidualBlock::forward` (line 566) is heavily optimized (RMS kernel/CF, feat_cache
causal). Add `is_2D` short-circuits that bypass the cache machinery (the 2-D VAE is stateless per plane).

(i) shortcut. Anchor (unique, fork line 574-578):

```cpp
            if (in_dim != out_dim) {
                auto shortcut = std::dynamic_pointer_cast<CausalConv3d>(blocks["shortcut"]);

                h = shortcut->forward(ctx, x);
            }
```

Replace with:

```cpp
            if (in_dim != out_dim) {
                if (is_2D) {
                    auto shortcut = std::dynamic_pointer_cast<Conv2dBut3d>(blocks["shortcut"]);
                    h             = shortcut->forward(ctx, x);
                } else {
                    auto shortcut = std::dynamic_pointer_cast<CausalConv3d>(blocks["shortcut"]);
                    h             = shortcut->forward(ctx, x);
                }
            }
```

(ii) the conv layers (`i == 2 || i == 6`). Anchor (unique, fork lines 606-625):

```cpp
                } else if (i == 2 || i == 6) {  // CausalConv3d
                    auto layer = std::dynamic_pointer_cast<CausalConv3d>(blocks["residual." + std::to_string(i)]);

                    // x is already contiguous [W,H,T,C] here (SiLU de-transposed it), so the cache
                    // concat/pad below get contiguous inputs.
                    if (feat_cache.size() > 0) {
                        int idx      = feat_idx;
                        auto cache_x = ggml_ext_slice(ctx->ggml_ctx, x, 2, -CACHE_T, x->ne[2]);
                        if (cache_x->ne[2] < 2 && feat_cache[idx] != nullptr) {
                            // cache last frame of last two chunk
                            cache_x = ggml_concat(ctx->ggml_ctx,
                                                  ggml_ext_slice(ctx->ggml_ctx, feat_cache[idx], 2, -1, feat_cache[idx]->ne[2]),
                                                  cache_x,
                                                  2);
                        }

                        x               = layer->forward(ctx, x, feat_cache[idx]);
                        feat_cache[idx] = cache_x;
                        feat_idx += 1;
                    }
                } else if (i == 1 || i == 4) {
```

Replace the branch head with an `is_2D` short-circuit (leave the existing 3-D body untouched under `else`):

```cpp
                } else if (i == 2 || i == 6) {  // conv
                    if (is_2D) {
                        auto layer = std::dynamic_pointer_cast<Conv2dBut3d>(blocks["residual." + std::to_string(i)]);
                        x          = layer->forward(ctx, x);
                    } else {
                        auto layer = std::dynamic_pointer_cast<CausalConv3d>(blocks["residual." + std::to_string(i)]);

                        // x is already contiguous [W,H,T,C] here (SiLU de-transposed it), so the cache
                        // concat/pad below get contiguous inputs.
                        if (feat_cache.size() > 0) {
                            int idx      = feat_idx;
                            auto cache_x = ggml_ext_slice(ctx->ggml_ctx, x, 2, -CACHE_T, x->ne[2]);
                            if (cache_x->ne[2] < 2 && feat_cache[idx] != nullptr) {
                                cache_x = ggml_concat(ctx->ggml_ctx,
                                                      ggml_ext_slice(ctx->ggml_ctx, feat_cache[idx], 2, -1, feat_cache[idx]->ne[2]),
                                                      cache_x,
                                                      2);
                            }

                            x               = layer->forward(ctx, x, feat_cache[idx]);
                            feat_cache[idx] = cache_x;
                            feat_idx += 1;
                        }
                    }
                } else if (i == 1 || i == 4) {
```

> NOTE the RMS steps (`i == 0 || i == 3`) and SiLU (`i == 1 || i == 4`) are dtype/shape-agnostic and work in 2-D
> unchanged — leave them. In `is_2D` mode `feat_cache` is still passed non-empty by the callers but the 2-D conv
> branch never reads it, matching upstream.

### 7c. `Down_ResidualBlock` / `Up_ResidualBlock` — `is_2D` passthrough

The fork uses `Down_ResidualBlock`/`Up_ResidualBlock` (upstream calls them `Downsample`/`Upsample`). These are only
instantiated by `Encoder3d`/`Decoder3d` **on the `wan2_2` path**, which is mutually exclusive with `is_2D` (a 2-D
Qwen-layered VAE is never wan2_2). Therefore the minimal correct change is to add a `bool is_2D = false` trailing
param to both ctors and forward it to their inner `ResidualBlock(in_dim, out_dim, is_2D)` constructions
(fork lines 660 and 722), for completeness/symmetry. Since `is_2D && wan2_2` never co-occur, you may instead skip
these two and rely on the `Encoder3d`/`Decoder3d` `is_2D` path (§7d/e) building plain `ResidualBlock`s — **document
whichever you choose.** Recommended: add the passthrough param (2 ctors, 2 forwarded args) to stay faithful.

### 7d. `Encoder3d` — `in_channels` + `is_2D`

The encoder is only needed for **img2img / encode**; pure layered t2i uses decode only. Port it anyway (gated) so
`is_2D` VAEs load and encode correctly. Add members + ctor params + gate the four conv constructions and the
`conv1`/`head.2` forwards.

Ctor member anchor (unique, fork `wan_vae.hpp:841-846`):

```cpp
        bool wan2_2;
        int64_t dim;
        int64_t z_dim;
        std::vector<int> dim_mult;
        int num_res_blocks;
        std::vector<bool> temperal_downsample;
```

Replace with:

```cpp
        bool wan2_2;
        int64_t in_channels;
        int64_t dim;
        int64_t z_dim;
        std::vector<int> dim_mult;
        int num_res_blocks;
        std::vector<bool> temperal_downsample;
        bool is_2D = false;
```

Ctor signature anchor (unique, fork line 849-860). Add `int64_t in_channels = 3` (after `z_dim`) and
`bool is_2D = false` (after `wan2_2`), init both. Then:
- `conv1`: when `is_2D`, `new Conv2dBut3d(in_channels, dims[0], {3,3}, {1,1}, {1,1})`; else keep the existing
  wan2_2/non-wan2_2 `CausalConv3d(in_channels, …)` (replace the hard-coded `12`/`3` with `in_channels`).
- Every `new ResidualBlock(in_dim, out_dim)` in the loop + `middle.0`/`middle.2` → append `, is_2D`.
- `Down_ResidualBlock(...)` → append `, is_2D` (see §7c).
- `head.2`: when `is_2D`, `new Conv2dBut3d(out_dim, z_dim, {3,3}, {1,1}, {1,1})`; else the existing `CausalConv3d`.

Forward: guard `conv1` and `head.2` with an `is_2D` branch that calls the `Conv2dBut3d` cast + `forward(ctx, x)`
(no cache), leaving the existing `feat_cache.size() > 0` bodies under `else if`. Mirror exactly the upstream diff
hunks at wan_vae.hpp encoder-forward `conv1` (`+ else if (is_2D)`) and `head.2`.

### 7e. `Decoder3d` — `out_channels` + `is_2D`

Ctor member anchor (unique, fork `wan_vae.hpp:1007-1012`):

```cpp
        bool wan2_2;
        int64_t dim;
        int64_t z_dim;
        std::vector<int> dim_mult;
        int num_res_blocks;
        std::vector<bool> temperal_upsample;
```

Replace with:

```cpp
        bool wan2_2;
        int64_t out_channels;
        int64_t dim;
        int64_t z_dim;
        std::vector<int> dim_mult;
        int num_res_blocks;
        std::vector<bool> temperal_upsample;
        bool is_2D = false;
```

Ctor signature (fork line 1015-1026): add `int64_t out_channels = 3` (after `z_dim`) and `bool is_2D = false`
(after `wan2_2`); init both. Then:
- `conv1` (line 1034): when `is_2D`, `new Conv2dBut3d(z_dim, dims[0], {3,3}, {1,1}, {1,1})`; else existing.
- `middle.0`/`middle.2` `ResidualBlock` (1037,1039) → append `, is_2D`.
- `Up_ResidualBlock(...)` (1050) → append `, is_2D` (§7c).
- The plain `ResidualBlock(in_dim, out_dim)` in the non-wan2_2 loop (1062) → append `, is_2D`.
- `head.2` (1078-1085): introduce `int64_t final_dim = out_channels;` and use it. When `is_2D`,
  `new Conv2dBut3d(out_dim, final_dim, {3,3}, {1,1}, {1,1})`; else the existing `CausalConv3d(out_dim, final_dim, …)`
  (replace hard-coded `12`/`3` with `final_dim`; **keep the fork's `force_prec_f32`/cuDNN flag args on the 3-D
  branch** — upstream doesn't have them but the fork does; the 2-D `Conv2dBut3d` ctor does not take them).

> **Fork-specific caution:** the fork's `head.2` `CausalConv3d` ctor uses the 8-arg form
> `CausalConv3d(out_dim, N, {3,3,3}, {1,1,1}, {1,1,1}, {1,1,1}, true, true)` (the trailing `true, true` =
> force_f32 / cuDNN F32-IO). The `is_2D` branch must NOT copy those — `Conv2dBut3d`/`Conv2d` has its own ctor
> arity. Keep the 3-D branch exactly as the fork has it.

Forward (fork line 1088): guard `conv1` and `head.2` with `is_2D` short-circuits (cast to `Conv2dBut3d`,
`forward(ctx, x)`, no cache), leaving the existing bodies under `else if`. The RMS-kernel/CF head block
(1160-1172) is dtype/shape-agnostic — leave it; it precedes `head.2` and works in 2-D.

### 7f. `WanVAE` — `is_2D` + `input_channels` + `patch_size`

Members anchor (unique, fork `wan_vae.hpp:1197-1205`):

```cpp
        bool wan2_2                           = false;
        bool decode_only                      = true;
        int64_t dim                           = 96;
        int64_t dec_dim                       = 96;
        int64_t z_dim                         = 16;
        std::vector<int> dim_mult             = {1, 2, 4, 4};
        int num_res_blocks                    = 2;
        std::vector<bool> temperal_upsample   = {true, true, false};
        std::vector<bool> temperal_downsample = {false, true, true};
```

Add three members after `decode_only`:

```cpp
        bool wan2_2                           = false;
        bool decode_only                      = true;
        bool is_2D                            = false;
        int64_t input_channels                = 3;
        int patch_size                        = 1;
        int64_t dim                           = 96;
        int64_t dec_dim                       = 96;
        int64_t z_dim                         = 16;
        std::vector<int> dim_mult             = {1, 2, 4, 4};
        int num_res_blocks                    = 2;
        std::vector<bool> temperal_upsample   = {true, true, false};
        std::vector<bool> temperal_downsample = {false, true, true};
```

Ctor anchor (unique, fork `wan_vae.hpp:1228-1248`):

```cpp
        WanVAE(bool decode_only = true, bool wan2_2 = false, int64_t dec_dim_override = -1)
            : decode_only(decode_only), wan2_2(wan2_2) {
            // attn_scales is always []
            if (wan2_2) {
                dim     = 160;
                dec_dim = 256;
                z_dim   = 48;

                _conv_num     = 34;
                _enc_conv_num = 26;
            }
            if (dec_dim_override > 0) {
                dec_dim = dec_dim_override;
            }
            if (!decode_only) {
                blocks["encoder"] = std::shared_ptr<GGMLBlock>(new Encoder3d(dim, z_dim * 2, dim_mult, num_res_blocks, temperal_downsample, wan2_2));
                blocks["conv1"]   = std::shared_ptr<GGMLBlock>(new CausalConv3d(z_dim * 2, z_dim * 2, {1, 1, 1}));
            }
            blocks["decoder"] = std::shared_ptr<GGMLBlock>(new Decoder3d(dec_dim, z_dim, dim_mult, num_res_blocks, temperal_upsample, wan2_2));
            blocks["conv2"]   = std::shared_ptr<GGMLBlock>(new CausalConv3d(z_dim, z_dim, {1, 1, 1}));
        }
```

Replace with (keep the fork's `dec_dim_override` param; add trailing `bool is_2D`; wire `input_channels`,
`patch_size`, temporal-flag override, and the gated conv construction):

```cpp
        WanVAE(bool decode_only = true, bool wan2_2 = false, int64_t dec_dim_override = -1, bool is_2D = false, int64_t input_channels_override = -1)
            : decode_only(decode_only), wan2_2(wan2_2), is_2D(is_2D) {
            // attn_scales is always []
            if (wan2_2) {
                dim            = 160;
                dec_dim        = 256;
                z_dim          = 48;
                input_channels = 12;
                patch_size     = 2;

                _conv_num     = 34;
                _enc_conv_num = 26;
            }
            if (input_channels_override > 0) {
                input_channels = input_channels_override;  // e.g. 4 for VERSION_QWEN_IMAGE_LAYERED
            }
            if (dec_dim_override > 0) {
                dec_dim = dec_dim_override;
            }
            if (is_2D) {
                temperal_upsample   = {false, false, false};
                temperal_downsample = {false, false, false};
            }
            if (!decode_only) {
                blocks["encoder"] = std::shared_ptr<GGMLBlock>(new Encoder3d(dim, z_dim * 2, input_channels, dim_mult, num_res_blocks, temperal_downsample, wan2_2, is_2D));
                if (is_2D) {
                    blocks["conv1"] = std::shared_ptr<GGMLBlock>(new Conv2dBut3d(z_dim * 2, z_dim * 2, {1, 1}));
                } else {
                    blocks["conv1"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(z_dim * 2, z_dim * 2, {1, 1, 1}));
                }
            }
            blocks["decoder"] = std::shared_ptr<GGMLBlock>(new Decoder3d(dec_dim, z_dim, input_channels, dim_mult, num_res_blocks, temperal_upsample, wan2_2, is_2D));
            if (is_2D) {
                blocks["conv2"] = std::shared_ptr<GGMLBlock>(new Conv2dBut3d(z_dim, z_dim, {1, 1}));
            } else {
                blocks["conv2"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(z_dim, z_dim, {1, 1, 1}));
            }
        }
```

> `input_channels_override` (default −1) is passed by `WanVAERunner` as `4` for layered and `−1` otherwise, so the
> wan2.1 default (3) and the wan2.2 `input_channels=12` derivation are preserved. `patch_size` replaces the
> hard-coded `2` in encode/decode's `patchify`/`unpatchify` calls (see §7g) — for wan2.1 (`patch_size==1`) those
> are no-ops, so behaviour is unchanged.

### 7g. `WanVAE::encode` / `decode` — use `patch_size`, gate `conv1`/`conv2`

The fork's `encode`/`decode` currently do `if (wan2_2) { x = patchify(…, 2, b); }`. Generalize to `patch_size`
(no-op when 1) and gate the pointwise `conv1`/`conv2` through `Conv2dBut3d` when `is_2D`.

encode (fork lines 1312-1334): replace `if (wan2_2) { x = patchify(ctx->ggml_ctx, x, 2, b); }` with
`x = patchify(ctx->ggml_ctx, x, patch_size, b);` and the final `out = conv1->forward(ctx, out);` with an `is_2D`
branch casting `blocks["conv1"]` to `Conv2dBut3d`. **Optional 2-D encode input-frame de-duplication** (upstream
drops 3-of-4 frames on the `is_2D` non-first-chunk slice) is only exercised for encoding *video* through a 2-D VAE
(a warned, degenerate case) — safe to omit for the layered image use.

decode (fork lines 1341-1390): change `auto x = conv2->forward(ctx, z);` to an `is_2D` branch casting
`blocks["conv2"]` to `Conv2dBut3d`; change `if (wan2_2) { out = unpatchify(ctx->ggml_ctx, out, 2, b); }` to
`out = unpatchify(ctx->ggml_ctx, out, patch_size, b);`. **Keep the fork's `WAN_VAE_F16` casts** — they operate on
the accumulator regardless of 2-D/3-D and are harmless. The per-layer decode (§8) calls `decode` with a latent of
`ne[2] == 1` (a single layer), so the loop runs once; upstream's "repeat the frame 4×" 2-D branch is only for
multi-frame 2-D decode and is **not needed** in the per-layer flow — omit it, or add it guarded by
`is_2D && iter_ > 1` for completeness.

Add `LOG_WARN("Using 2D VAE to decode video, expect poor results");` behind `if (z->ne[2] > 1 && is_2D)` at
`decode` entry if you want the upstream diagnostic.

### 7h. `WanVAERunner` — detect `is_2D`, forward to `WanVAE`

Add a detection helper mirroring the fork's existing `detect_dec_dim` (same shape, keyed on
`decoder.conv1.weight` ne[2] > 3). Anchor (unique, insert after `detect_dec_dim`'s closing brace, fork
`wan_vae.hpp:1589`):

```cpp
        static bool detect_is_2D(const String2TensorStorage& tensor_storage_map) {
            for (const auto& [name, tensor_storage] : tensor_storage_map) {
                if (ends_with(name, "decoder.conv1.weight")) {
                    return tensor_storage.ne[2] > 3;
                }
            }
            return false;
        }
```

Ctor anchor (unique, fork `wan_vae.hpp:1591-1601`):

```cpp
        WanVAERunner(ggml_backend_t backend,
                     ggml_backend_t params_backend,
                     const String2TensorStorage& tensor_storage_map = {},
                     const std::string prefix                       = "",
                     bool decode_only                               = false,
                     SDVersion version                              = VERSION_WAN2)
            : decode_only(decode_only),
              ae(decode_only, version == VERSION_WAN2_2_TI2V, detect_dec_dim(tensor_storage_map)),
              VAE(version, backend, params_backend) {
            ae.init(params_ctx, tensor_storage_map, prefix);
        }
```

Replace the `ae(...)` member-initializer to pass `is_2D` and the layered `input_channels_override`:

```cpp
        WanVAERunner(ggml_backend_t backend,
                     ggml_backend_t params_backend,
                     const String2TensorStorage& tensor_storage_map = {},
                     const std::string prefix                       = "",
                     bool decode_only                               = false,
                     SDVersion version                              = VERSION_WAN2)
            : decode_only(decode_only),
              ae(decode_only,
                 version == VERSION_WAN2_2_TI2V,
                 detect_dec_dim(tensor_storage_map),
                 detect_is_2D(tensor_storage_map),
                 version == VERSION_QWEN_IMAGE_LAYERED ? 4 : -1),
              VAE(version, backend, params_backend) {
            ae.init(params_ctx, tensor_storage_map, prefix);
        }
```

Verified fork primitives: `ends_with` (used across the fork), `String2TensorStorage`, `tensor_storage.ne[]`
(same struct the fork's `detect_dec_dim` reads). `get_param_tensors(tensors, prefix)` override stays as-is.

**Non-disturbance guarantee:** for every existing model, `detect_is_2D` returns `false` (their
`decoder.conv1.weight` is a 5-D causal-conv weight with `ne[2] == 3`), and `input_channels_override == -1`, so the
`ae(...)` construction is behaviourally identical to today. The `_conv_num`/`_enc_conv_num` cache sizes are only
consulted on the `is_2D == false` path. **No risk to the fork's Wan/video VAE.**

---

## 8. `src/stable-diffusion.cpp` — orchestration (the real work; NO clean fork anchor)

The fork's `sd_version_is_qwen_image` fold (§1b) already routes DiT dispatch (line 893) and the VAE factory
(line 1096) for layered — **no edits there.** But the fork has **no** `get_image_channels()`, **no**
`qwen_image_layers` request field, and **no** layered init-latent / per-layer decode path. These must be added.
The fork's generate/decode orchestration has diverged heavily from upstream (video-first, many bespoke paths), so
there is no upstream hunk to translate 1:1 — the implementer must place these deliberately.

### 8a. `get_latent_channel()` — already correct (verify only)

Layered latent channel is 16 (Wan2.1 VAE `z_dim`). The fork's `get_latent_channel()` (line 3614) sends any
`sd_version_is_dit` Qwen-Image to the `else → 16` branch. Upstream does not special-case layered either. **No edit
needed.** (`get_vae_scale_factor` / `video_frames_to_latent_frames` also leave Qwen-Image unchanged → `T = frames`,
so a layers+1 "frame" count maps 1:1 to latent layers — matching upstream.)

### 8b. `get_image_channels()` — NEW (fork lacks it entirely)

Upstream (`stable-diffusion.cpp:2772`):

```cpp
    int get_image_channels() const {
        return version == VERSION_QWEN_IMAGE_LAYERED ? 4 : 3;
    }
```

**The fork has no such method** (grep returned nothing). Add it to the same `StableDiffusionGGML` struct (place it
near `get_latent_channel`, fork `stable-diffusion.cpp:3614`), then thread the return value into every place the
fork currently hard-codes `3` output channels when converting a decoded latent tensor to an image
(`sd_tensor_to_image` / the decode-to-image conversion). **This is a fork-wide audit**, not a single edit — the
fork's image writer assumes 3 channels. For RGBA layered output each layer must be written as 4-channel PNG
(RGBA). Flag: enumerate the decode-to-image sinks and pass `get_image_channels()` through; this is new surface with
no upstream parallel in the fork's code.

### 8c. Layered init latent (5-D over layers) — NEW branch

Upstream (`stable-diffusion.cpp:4713`) builds the init latent as a **5-D** tensor whose "frame" axis carries the
layers:

```cpp
        if (sd_ctx->sd->version == VERSION_QWEN_IMAGE_LAYERED) {
            init_latent = sd_ctx->sd->generate_init_latent(request->width, request->height, request->qwen_image_layers + 1, true);
        } else {
            init_latent = sd_ctx->sd->generate_init_latent(request->width, request->height);
        }
```

The fork's `generate_init_latent(w, h, frames, video=true)` already produces `sd::zeros<float>({W,H,T,C,1})` with
`T = video_frames_to_latent_frames(frames) = frames` for Qwen-Image (§8a) — so the primitive exists. **The missing
piece is the branch + the `qwen_image_layers` request field.** In the fork's image generate path (the non-video
`generate_init_latent(width, height)` call, fork line 6855), add the layered branch above. Because the fork's t2i
sampler loop assumes a 4-D latent, the layered 5-D latent implies either (a) sample all layers jointly in one 5-D
DiT forward (needs §6's temporal-RoPE and upstream's 5-D `QwenImageModel::forward` — the path §5f deliberately
did NOT take), or (b) sample layers as a batch/loop of 4-D forwards. **Choose (b) to match §5's design**: iterate
the layer axis, run the 4-D DiT per layer (each with `addition_t_cond` row 0), stack results on `ne[2]`. This is
new control flow with no fork anchor — the single biggest integration task.

### 8d. Per-layer decode loop — NEW

Upstream (`stable-diffusion.cpp:5024`) slices each layer off the 5-D final latent, squeezes the layer axis, and
decodes each through the 2-D VAE:

```cpp
        if (sd_ctx->sd->version == VERSION_QWEN_IMAGE_LAYERED) {
            int qwen_image_latent_layers = request.qwen_image_layers + 1;
            ... // validate dim>=5 and shape[2] >= layers
            for (int layer_index = 0; layer_index < qwen_image_latent_layers; layer_index++) {
                sd::Tensor<float> layer_latent = sd::ops::slice(final_latents[i], 2, layer_index, layer_index + 1);
                layer_latent.squeeze_(2);
                sd::Tensor<float> image = sd_ctx->sd->decode_first_stage(layer_latent);
                decoded_images.push_back(std::move(image));
            }
        }
```

Add the equivalent branch in the fork's decode-latents loop (fork `stable-diffusion.cpp` around the
`decode_first_stage` call in the final image-decode loop — grep `decode_first_stage` in the image path). Fork
primitives verified present: `sd::ops::slice`, `Tensor::squeeze_`, `decode_first_stage`. Each `layer_latent` is
`ne[2] == 1` after squeeze → `WanVAE::decode` runs its per-frame loop once (§7g), returning one RGBA image via the
2-D VAE. `decode_first_stage`'s existing clamp/scale is version-agnostic; confirm no `+1)*0.5` remap is applied for
Qwen (Wan VAE output is already image-space) — the fork applies the Wan latent mean/std in
`vae_output_to_latents`/`get_latents_mean_std`, unaffected here.

### 8e. `qwen_image_layers` — NEW request/API field

Upstream carries `request->qwen_image_layers` (a small int, N extra transparent layers; total layers = N+1). The
fork's request/`sd_ctx_params` structs have **no such field** (grep returned nothing). Add:
- an `int qwen_image_layers = 0;` field to the fork's image-generation request struct (mirror where
  `qwen_image_zero_cond_t` lives, fork `sd_ctx_params`), and
- plumb it from the CLI/API surface (koblem side, out of scope for this C++ doc — flag to the owner).

Default `0` = a single RGBA image (1 layer), which keeps every non-layered request unaffected.

---

## 9. Conditioner — reuses `LLMEmbedder` with vision OFF (confirmed)

Upstream (`stable-diffusion.cpp:1136`): `bool enable_vision = version != VERSION_QWEN_IMAGE_LAYERED;` — layered uses
the **same** Qwen `LLMEmbedder` text encoder as plain Qwen-Image, but with the vision tower disabled.

The fork's Qwen dispatch (line 893-903) sets `enable_vision` only from `!vae_decode_only`. To match upstream, gate
it also on layered. Anchor (unique, fork lines 894-896):

```cpp
                bool enable_vision = false;
                if (!vae_decode_only) {
                    enable_vision = true;
                }
```

Replace with:

```cpp
                bool enable_vision = false;
                if (!vae_decode_only && version != VERSION_QWEN_IMAGE_LAYERED) {
                    enable_vision = true;
                }
```

No new conditioner class — `LLMEmbedder` is reused verbatim (the fold in §1b already routes layered here).

---

## 10. Blockers, missing primitives, risks

**BLOCKER-1 (integration, largest):** §8 — the fork has no `get_image_channels()`, no `qwen_image_layers` field,
no 5-D layered init-latent branch, and no per-layer decode loop, and its generate/decode orchestration has diverged
too far to translate upstream 1:1. This is real new control flow + a fork-wide 3→4 channel (RGBA) output audit.
Land §1–§5, §7, §9 first (they are self-contained and low-risk); treat §8 as a separate milestone.

**BLOCKER-2 (no checkpoint):** No Qwen-Image-Layered weights are available on-box, so nothing here can be
GPU-validated. In particular: the `Conv2dBut3d` axis-swap orientation, the `decoder.conv1.weight` ne[2]>3
detection, the `input_channels=4` assumption, and the RoPE nuance (§6) are all self-consistent with upstream but
**unverified against real tensors.** If the checkpoint stores conv weights transposed differently, adjust the
`Conv2dBut3d` permute.

**GAP-1 (rope, §6):** the fork's `gen_vid_ids` lacks the `scale_rope` param and `gen_qwen_image_pe` lacks the `t`
axis. §5's single-image-per-forward design sidesteps this (layer temporal id = 0), so **no rope edit is required
for the recommended design.** Only upstream's batched 5-D strategy needs it; do not adopt upstream's `txt_id_start`
change (it perturbs the working `VERSION_QWEN_IMAGE`).

**RISK to the fork's Wan/video 3-D VAE path: NONE, by construction.** Every VAE change (§7) is gated on `is_2D`
(default false, only true when `decoder.conv1.weight` ne[2] > 3, which no existing model satisfies) or on the new
`input_channels_override`/`patch_size` (which reduce to today's values for wan2.1/wan2.2). The fork's decode
optimizations (`WAN_VAE_RMS_KERNEL`, `WAN_VAE_RMS_CF`, `WAN_VAE_F16`, causal `feat_cache`, `dec_dim_override`,
`WAN_ZERO_T_ROPE`) all live on the `is_2D == false` side and are untouched. The `Conv2dBut3d` class is new and
inert unless constructed. This satisfies the "add 2-D as a gated additive layer, do not rewrite the video VAE"
requirement.

**RISK (fork convention drift):** the upstream diffs for `qwen_image.hpp` and `wan_vae.hpp` also carry an unrelated
`RunnerWeightManager`/`ModelManager` refactor and a `ggml_ext_*(…, ctx->backend, …)` signature change. **None of
that is adopted here** — this doc translates only the layered/2-D-specific hunks into the fork's
`(backend, params_backend)` + no-backend-arg conventions. Do not pull the ModelManager pieces.

**Verification status:** all fork anchors (`model.h`, `qwen_image.hpp`, `wan_vae.hpp`, `stable-diffusion.cpp`,
`rope.hpp`, `dit.hpp`, `ggml_extend.hpp`) were `Read`/`grep`-confirmed at the cited lines against the current
worktree. Fork primitives confirmed present: `Conv2d` (`ggml_extend.hpp:3443`), `Embedding`
(`ggml_extend.hpp:6095`), `DiT::patchify_3d`/`unpatchify_3d`/`pad_to_patch_size` (`dit.hpp`),
`Rope::gen_vid_ids` (`rope.hpp:682`, **no scale_rope**), `parse_key_value_args`/`parse_strict_bool`
(`core/util.h`), `ends_with`, `set_backend_tensor_data`, `detect_dec_dim`. Cannot build/compile-check here
(per project rules) and no checkpoint to run.
