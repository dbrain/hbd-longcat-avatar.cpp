# lingbot_video — APPLY-ready shared-file edits

Worktree: `/home/dbrain/dev/longcat-avatar-mergewt` (branch `upstream-merge-2026-07-19`)
Fork HEAD: `ca0d922` (boogu) on top of `7502d89` (krea2) on top of merge `6c0fdf2`.
Upstream ref: `ea4e566`.

Scope: this doc is the EXACT remaining shared-file work to make `src/model/diffusion/lingbot_video.hpp`
(already adapted + committed) compile and run once it is `#include`d from `stable-diffusion.cpp`.

**Do NOT build. Every edit below was verified against the current fork tree with grep + Read.**

---

## STATUS OF PRE-DONE ITEMS (verified, no action)

- `src/model.h` — enum `VERSION_LINGBOT_VIDEO` present at line 57 (between `VERSION_HUNYUAN_VIDEO` and
  `VERSION_COUNT`); helper `sd_version_is_lingbot_video` at line 213; membership in `sd_version_is_dit()`
  at line 255. **All three already landed (via the krea2 commit 7502d89). No edit needed.**
- `src/stable-diffusion.cpp` `model_version_to_str[]` — the string `"LingBot Video",` is **already present**
  as the last element (added in 7502d89). Its array index equals the enum index. **No edit needed** — spec
  section B ("append the name") is already satisfied; do NOT append it again.

---

## PART A — THREE BUILD-BLOCKER PRIMITIVES (critical)

All three symbols the model .hpp calls are **absent** from the fork's shared files. They compile-error the
instant `lingbot_video.hpp` is `#include`d (the MoE + forward bodies are inline in non-template classes).

### A.1 — `DiT::patchify_3d` + `DiT::unpatchify_3d`  →  `src/model/diffusion/dit.hpp`

Verified: the fork has a 2D `patchify`/`unpatchify` (lines 7 / 36) **and** an un-suffixed 3D
`patchify`/`unpatchify` (lines 107 / 138) — but **NO `_3d`-suffixed variants**. The model .hpp calls
`DiT::patchify_3d(..., N, false)` and `DiT::unpatchify_3d(..., false)` (7-/9-arg with a trailing
`patch_last=false`). The fork's un-suffixed 3D helpers have a DIFFERENT signature (no `patch_last`; and the
un-suffixed `unpatchify` derives `N` from `x->ne[3]`, whereas `unpatchify_3d` uses `x->ne[2]`), so they
cannot satisfy the call. Add the upstream `_3d` variants as NEW, separately-named functions — they coexist
with the un-suffixed ones with no redefinition. Bodies use only `ggml_reshape_4d`, `ggml_ext_cont`,
`ggml_ext_torch_permute`, `GGML_ASSERT` — all present in the fork (`ggml_extend.hpp:697,706`).

**Anchor** (unique — the namespace close, line 164):
```cpp
}  // namespace DiT
```
**Replace with** (insert the two functions BEFORE the close):
```cpp
    inline ggml_tensor* patchify_3d(ggml_context* ctx,
                                    ggml_tensor* x,
                                    int pt,
                                    int ph,
                                    int pw,
                                    int64_t N       = 1,
                                    bool patch_last = true) {
        // x: [N*C, T, H, W]
        // return: [N, t_len*h_len*w_len, C*pt*ph*pw] if patch_last else [N, t_len*h_len*w_len, C*pt*ph*pw] or [N, t_len*h_len*w_len, pt*ph*pw*C]
        int64_t C     = x->ne[3] / N;
        int64_t T     = x->ne[2];
        int64_t H     = x->ne[1];
        int64_t W     = x->ne[0];
        int64_t t_len = T / pt;
        int64_t h_len = H / ph;
        int64_t w_len = W / pw;

        GGML_ASSERT(C * N == x->ne[3]);
        GGML_ASSERT(t_len * pt == T && h_len * ph == H && w_len * pw == W);

        x = ggml_reshape_4d(ctx, x, pw * w_len, ph * h_len, pt, t_len * C * N);  // [N*C*t_len, pt, h_len*ph, w_len*pw]
        x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));      // [N*C*t_len, h_len*ph, pt, w_len*pw]
        x = ggml_reshape_4d(ctx, x, pw * w_len, pt, ph, h_len * t_len * C * N);  // [N*C*t_len*h_len, ph, pt, w_len*pw]
        x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));      // [N*C*t_len*h_len, pt, ph, w_len*pw]
        x = ggml_reshape_4d(ctx, x, pw, w_len, ph * pt, h_len * t_len * C * N);  // [N*C*t_len*h_len, pt*ph, w_len, pw]
        x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));      // [N*C*t_len*h_len, w_len, pt*ph, pw]
        x = ggml_reshape_4d(ctx, x, pw * ph * pt, w_len * h_len * t_len, C, N);  // [N, C, t_len*h_len*w_len, pt*ph*pw]
        if (patch_last) {
            x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));  // [N, t_len*h_len*w_len, C, pt*ph*pw]
        } else {
            x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 2, 0, 1, 3));  // [N, t_len*h_len*w_len, pt*ph*pw, C]
        }
        // [N, t_len*h_len*w_len, C*pt*ph*pw] or [N, t_len*h_len*w_len, pt*ph*pw*C]
        x = ggml_reshape_4d(ctx, x, pw * ph * pt * C, w_len * h_len * t_len, N, 1);
        return x;
    }

    inline ggml_tensor* unpatchify_3d(ggml_context* ctx,
                                      ggml_tensor* x,
                                      int64_t t_len,
                                      int64_t h_len,
                                      int64_t w_len,
                                      int pt,
                                      int ph,
                                      int pw,
                                      bool patch_last = true) {
        // x: [N, t_len*h_len*w_len, C*pt*ph*pw] if patch_last else [N, t_len*h_len*w_len, pt*ph*pw*C]
        // return: [N*C, t_len*pt, h_len*ph, w_len*pw]
        int64_t N = x->ne[2];
        int64_t C = x->ne[0] / pt / ph / pw;

        GGML_ASSERT(C * pt * ph * pw == x->ne[0]);

        if (patch_last) {
            x = ggml_reshape_4d(ctx, x, pw * ph * pt, C, w_len * h_len * t_len, N);  // [N, t_len*h_len*w_len, C, pt*ph*pw]
            x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));      // [N, C, t_len*h_len*w_len, pt*ph*pw]
        } else {
            x = ggml_reshape_4d(ctx, x, C, pw * ph * pt, w_len * h_len * t_len, N);  // [N, t_len*h_len*w_len, pt*ph*pw, C]
            x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 1, 2, 0, 3));      // [N, C, t_len*h_len*w_len, pt*ph*pw]
        }

        x = ggml_reshape_4d(ctx, x, pw, ph * pt, w_len, h_len * t_len * C * N);  // [N*C*t_len*h_len, w_len, pt*ph, pw]
        x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));      // [N*C*t_len*h_len, pt*ph, w_len, pw]
        x = ggml_reshape_4d(ctx, x, pw * w_len, ph, pt, h_len * t_len * C * N);  // [N*C*t_len*h_len, pt, ph, w_len*pw]
        x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));      // [N*C*t_len*h_len, ph, pt, w_len*pw]
        x = ggml_reshape_4d(ctx, x, pw * w_len, pt, ph * h_len, t_len * C * N);  // [N*C*t_len, h_len*ph, pt, w_len*pw]
        x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));      // [N*C*t_len, pt, h_len*ph, w_len*pw]
        x = ggml_reshape_4d(ctx, x, pw * w_len, ph * h_len, pt * t_len, C * N);  // [N*C, t_len*pt, h_len*ph, w_len*pw]
        return x;
    }
}  // namespace DiT
```

### A.2 — `Rope::gen_lingbot_video_pe` (+ its dependency `gen_lingbot_video_ids`)  →  `src/model/common/rope.hpp`

**CORRECTION TO THE TASK PREMISE:** the task said "the fork already has `gen_lingbot_video_ids` (merge notes
rope.hpp H2)". **That is FALSE for the current tree** — `grep -n lingbot src/model/common/rope.hpp` returns
NOTHING. Neither `gen_lingbot_video_ids` nor `gen_lingbot_video_pe` exists. Both must be added.

Dependency check (verified present in fork rope.hpp):
- `gen_vid_ids(...)` — fork signature `(t,h,w,pt,ph,pw,bs, int t_offset=0, int h_offset=0, int w_offset=0)`
  (line 682). Upstream `gen_lingbot_video_ids` calls `gen_vid_ids(t,h,w,pt,ph,pw,bs, context_len + 1)`, i.e.
  the 8th positional arg `= t_offset = context_len + 1`. This is IDENTICAL semantics in fork and upstream —
  the fork's `gen_vid_ids` differs from upstream only by an extra trailing `bool scale_rope` param (unused
  here) and the `WAN_ZERO_T_ROPE` diagnostic (a no-op by default → byte-identical). Safe.
- `concat_ids` (line 164) and the 4-arg `embed_nd` (line 181) — both present. Safe.

**Anchor** (unique — the namespace close, line 1031):
```cpp
};  // namespace Rope
```
**Replace with** (insert BOTH functions, ids first, BEFORE the close):
```cpp
    __STATIC_INLINE__ std::vector<std::vector<float>> gen_lingbot_video_ids(int t,
                                                                            int h,
                                                                            int w,
                                                                            int pt,
                                                                            int ph,
                                                                            int pw,
                                                                            int bs,
                                                                            int context_len) {
        auto vid_ids_repeated = gen_vid_ids(t, h, w, pt, ph, pw, bs, context_len + 1);

        std::vector<std::vector<float>> txt_ids(bs * context_len, std::vector<float>(3, 0.0f));
        for (int i = 0; i < bs; ++i) {
            for (int j = 0; j < context_len; ++j) {
                txt_ids[i * context_len + j][0] = static_cast<float>(j + 1);
            }
        }

        return concat_ids(vid_ids_repeated, txt_ids, bs);
    }

    __STATIC_INLINE__ std::vector<float> gen_lingbot_video_pe(int t,
                                                              int h,
                                                              int w,
                                                              int pt,
                                                              int ph,
                                                              int pw,
                                                              int bs,
                                                              int context_len,
                                                              int theta,
                                                              const std::vector<int>& axes_dim) {
        std::vector<std::vector<float>> ids = gen_lingbot_video_ids(t, h, w, pt, ph, pw, bs, context_len);
        return embed_nd(ids, bs, static_cast<float>(theta), axes_dim);
    }
};  // namespace Rope
```

### A.3 — `GGMLRunnerContext::bind_backend_tensor_data` + backing member  →  `src/core/ggml_extend.hpp`

Verified: `struct GGMLRunnerContext` has NEITHER the method nor the `std::function` member. The backing
`GGMLRunner::set_backend_tensor_data(ggml_tensor*, const void*)` DOES exist (line 5737), so the lambda is
valid. Three edits in this file:

**A.3a — add the member.** Anchor (unique, line 2219):
```cpp
    std::function<void(const std::string&, ggml_tensor*)> cache_tensor;
```
Replace with:
```cpp
    std::function<void(const std::string&, ggml_tensor*)> cache_tensor;
    std::function<void(ggml_tensor*, const void*)> set_backend_tensor_data;
```

**A.3b — add the method.** Anchor (unique — the `persist_cache_tensor` method just before the struct's
closing brace, lines 2259-2265):
```cpp
    void persist_cache_tensor(const std::string& name, ggml_tensor* tensor) const {
        if (!cache_tensor || tensor == nullptr) {
            return;
        }
        cache_tensor(name, tensor);
    }
};
```
Replace with:
```cpp
    void persist_cache_tensor(const std::string& name, ggml_tensor* tensor) const {
        if (!cache_tensor || tensor == nullptr) {
            return;
        }
        cache_tensor(name, tensor);
    }

    void bind_backend_tensor_data(ggml_tensor* tensor, const void* data) const {
        if (!set_backend_tensor_data || tensor == nullptr || data == nullptr) {
            return;
        }
        set_backend_tensor_data(tensor, data);
    }
};
```

**A.3c — wire it in `GGMLRunner::get_context()`.** Anchor (unique, lines 5348-5350):
```cpp
        runner_ctx.cache_tensor = [this](const std::string& name, ggml_tensor* tensor) {
            this->cache(name, tensor);
        };
```
Replace with:
```cpp
        runner_ctx.cache_tensor = [this](const std::string& name, ggml_tensor* tensor) {
            this->cache(name, tensor);
        };
        runner_ctx.set_backend_tensor_data = [this](ggml_tensor* t, const void* d) {
            this->set_backend_tensor_data(t, d);
        };
```

---

## PART B — WIRING EDITS

### B.1 — detection  →  `src/model_loader.cpp`

Verified: `get_sd_version()` per-tensor loop has the LTXAV check at 541-543 and the wan heuristic
(`is_wan = true`) at 544-545. Placing the lingbot check between them makes it win (it `return`s
immediately, before `is_wan` can be set). Krea2/boogu detection precede this block at 501/504.

**Anchor** (unique, lines 541-545):
```cpp
        if (tensor_storage.name.find("model.diffusion_model.adaln_single.emb.timestep_embedder.linear_1.bias") != std::string::npos) {
            return VERSION_LTXAV;
        }
        if (tensor_storage.name.find("model.diffusion_model.blocks.0.cross_attn.norm_k.weight") != std::string::npos) {
            is_wan = true;
        }
```
**Replace with:**
```cpp
        if (tensor_storage.name.find("model.diffusion_model.adaln_single.emb.timestep_embedder.linear_1.bias") != std::string::npos) {
            return VERSION_LTXAV;
        }
        if (tensor_storage.name.find("model.diffusion_model.patch_embedder.weight") != std::string::npos) {
            return VERSION_LINGBOT_VIDEO;
        }
        if (tensor_storage.name.find("model.diffusion_model.blocks.0.cross_attn.norm_k.weight") != std::string::npos) {
            is_wan = true;
        }
```

### B.2 — include  →  `src/stable-diffusion.cpp`

**Anchor** (unique, line 31 — added by the boogu commit):
```cpp
#include "model/diffusion/boogu.hpp"
```
**Replace with:**
```cpp
#include "model/diffusion/boogu.hpp"
#include "model/diffusion/lingbot_video.hpp"
```

### B.3 — dispatch case  →  `src/stable-diffusion.cpp`

Fork convention: `backend_for` + `params_backend_for`, no `model_manager`. `LLMEmbedder` 6-arg ctor
(same as krea2/boogu), `LingBotVideoRunner` 4-arg ctor. Insert after the boogu else-if (ends line 732)
and before `} else if (sd_version_is_flux(version)) {` (line 733).

**Anchor** (unique — the boogu diffusion_model tail immediately followed by the flux branch, lines 729-733):
```cpp
                diffusion_model  = std::make_shared<Boogu::BooguImageRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                            params_backend_for(SDBackendModule::DIFFUSION),
                                                                            tensor_storage_map,
                                                                            "model.diffusion_model");
            } else if (sd_version_is_flux(version)) {
```
**Replace with:**
```cpp
                diffusion_model  = std::make_shared<Boogu::BooguImageRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                            params_backend_for(SDBackendModule::DIFFUSION),
                                                                            tensor_storage_map,
                                                                            "model.diffusion_model");
            } else if (sd_version_is_lingbot_video(version)) {
                bool enable_vision = false;
                for (const auto& [name, _] : tensor_storage_map) {
                    if (starts_with(name, "text_encoders.llm.visual.")) {
                        enable_vision = true;
                        break;
                    }
                }
                cond_stage_model = std::make_shared<LLMEmbedder>(backend_for(SDBackendModule::TE),
                                                                 params_backend_for(SDBackendModule::TE),
                                                                 tensor_storage_map,
                                                                 version,
                                                                 "",
                                                                 enable_vision);
                diffusion_model  = std::make_shared<LingBotVideo::LingBotVideoRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                                     params_backend_for(SDBackendModule::DIFFUSION),
                                                                                     tensor_storage_map,
                                                                                     "model.diffusion_model");
            } else if (sd_version_is_flux(version)) {
```

### B.4 — create_vae WAN branch  →  `src/stable-diffusion.cpp`

**Anchor** (unique, lines 1042-1047):
```cpp
                } else if (sd_version_is_wan(version) ||
                           sd_version_is_qwen_image(version) ||
                           sd_version_is_anima(version) ||
                           sd_version_is_longcat_avatar(version) ||
                           sd_version_is_krea2(version)) {
                    return std::make_shared<WAN::WanVAERunner>(backend_for(SDBackendModule::VAE),
```
**Replace with:**
```cpp
                } else if (sd_version_is_wan(version) ||
                           sd_version_is_qwen_image(version) ||
                           sd_version_is_anima(version) ||
                           sd_version_is_longcat_avatar(version) ||
                           sd_version_is_krea2(version) ||
                           sd_version_is_lingbot_video(version)) {
                    return std::make_shared<WAN::WanVAERunner>(backend_for(SDBackendModule::VAE),
```

### B.5 — create_tae WAN branch  →  `src/stable-diffusion.cpp`

**Anchor** (unique, lines 1002-1006):
```cpp
                if (sd_version_is_wan(version) ||
                    sd_version_is_qwen_image(version) ||
                    sd_version_is_anima(version) ||
                    sd_version_is_ltxav(version)) {
                    return std::make_shared<TinyVideoAutoEncoder>(backend_for(SDBackendModule::VAE),
```
**Replace with:**
```cpp
                if (sd_version_is_wan(version) ||
                    sd_version_is_qwen_image(version) ||
                    sd_version_is_anima(version) ||
                    sd_version_is_ltxav(version) ||
                    sd_version_is_lingbot_video(version)) {
                    return std::make_shared<TinyVideoAutoEncoder>(backend_for(SDBackendModule::VAE),
```
(Harmless unless a TAE/taew checkpoint is supplied.)

---

## PART C — VIDEO PIPELINE QUIRKS  (`src/stable-diffusion.cpp`)

Upstream adds `sd_version_is_lingbot_video` alongside `sd_version_is_wan` at 8 runtime sites. The fork has
diverged (different line numbers, `sd_version_is_longcat_avatar` where upstream added `hunyuan_video`).
Each fork equivalent below was located + verified. Add lingbot ONLY (do not add hunyuan — not wired here).

### C.1 — scheduler / pred_type (FLOW_PRED group).  Fork lines 1606-1616.

Upstream (ea4e566:1641-1644) puts lingbot in the sd3/wan **FLOW_PRED** group with NO lingbot-specific
`flow_shift` else-if → it falls through to the final `else { default_flow_shift = 3.f; }`. Mirror that.

**Anchor** (unique, lines 1614-1616):
```cpp
                           sd_version_is_ideogram4(version) ||
                           sd_version_is_boogu_image(version)) {
                    pred_type = FLOW_PRED;
```
**Replace with:**
```cpp
                           sd_version_is_ideogram4(version) ||
                           sd_version_is_boogu_image(version) ||
                           sd_version_is_lingbot_video(version)) {
                    pred_type = FLOW_PRED;
```
(No flow_shift else-if added → lingbot uses `default_flow_shift = 3.f`, matching upstream.)

### C.2 — denoise-mask blend.  Fork line 3024.

**Anchor** (unique):
```cpp
            if (!denoise_mask.empty() && (version == VERSION_WAN2_2_TI2V || version == VERSION_WAN2_2_I2V || sd_version_is_ltxav(version) || sd_version_is_longcat_avatar(version))) {
```
**Replace with:**
```cpp
            if (!denoise_mask.empty() && (version == VERSION_WAN2_2_TI2V || version == VERSION_WAN2_2_I2V || sd_version_is_ltxav(version) || sd_version_is_longcat_avatar(version) || sd_version_is_lingbot_video(version))) {
```

### C.3 — down_factor = 2.  Fork lines 3527-3528.

**Anchor** (unique):
```cpp
            if (sd_version_is_wan(version) || sd_version_is_longcat_avatar(version)) {
                down_factor = 2;
```
**Replace with:**
```cpp
            if (sd_version_is_wan(version) || sd_version_is_longcat_avatar(version) || sd_version_is_lingbot_video(version)) {
                down_factor = 2;
```

### C.4 — video_frames_to_latent_frames (÷4 +1).  Fork line 3582.

**Anchor** (unique — the `else if` inside `video_frames_to_latent_frames`, followed by the `/4` assignment):
```cpp
        } else if (sd_version_is_wan(version) || sd_version_is_longcat_avatar(version)) {
            latent_frames = ((frames - 1) / 4) + 1;
        }
```
**Replace with:**
```cpp
        } else if (sd_version_is_wan(version) || sd_version_is_longcat_avatar(version) || sd_version_is_lingbot_video(version)) {
            latent_frames = ((frames - 1) / 4) + 1;
        }
```

### C.5 — latent_frames_to_video_frames (×4 +1).  Fork line 3596.

**Anchor** (unique — the `if` inside `latent_frames_to_video_frames`, followed by the `* 4 + 1` return):
```cpp
        if (sd_version_is_wan(version) || sd_version_is_longcat_avatar(version)) {
            return (latent_frames - 1) * 4 + 1;
        }
```
**Replace with:**
```cpp
        if (sd_version_is_wan(version) || sd_version_is_longcat_avatar(version) || sd_version_is_lingbot_video(version)) {
            return (latent_frames - 1) * 4 + 1;
        }
```

### C.6 — is_video predicate.  Fork line 4497.

**Anchor** (unique):
```cpp
    return version == VERSION_SVD || sd_version_is_wan(version) || sd_version_is_ltxav(version) || sd_version_is_longcat_avatar(version);
```
**Replace with:**
```cpp
    return version == VERSION_SVD || sd_version_is_wan(version) || sd_version_is_ltxav(version) || sd_version_is_longcat_avatar(version) || sd_version_is_lingbot_video(version);
```

### C.7 — img2vid encode_first_stage path.  Fork line 9135 (insert before the longcat_avatar branch).

Upstream (ea4e566:5828) is a distinct `else if (sd_version_is_lingbot_video(...) && !start_image.empty())`
branch. The fork's video latent-prep chain has a byte-for-byte equivalent template at the longcat_avatar
branch (fork 9135). Insert the lingbot branch immediately before it. Verified fork API: `generate_init_latent(w,h,frames,true)`, `encode_first_stage`, `sd::ops::slice_assign`, `sd::full<float>`, `sd::ops::fill_slice`, and `latents.ref_images` (member exists, used at fork 6941 style) all present.

**Anchor** (unique, line 9135):
```cpp
    } else if (sd_version_is_longcat_avatar(sd_ctx->sd->version) && !start_image.empty()) {
```
**Replace with:**
```cpp
    } else if (sd_version_is_lingbot_video(sd_ctx->sd->version) && !start_image.empty()) {
        LOG_INFO("LingBot Video IMG2VID");

        int64_t t1             = ggml_time_ms();
        auto init_img          = start_image.reshape({start_image.shape()[0], start_image.shape()[1], 1, start_image.shape()[2], 1});
        auto init_image_latent = sd_ctx->sd->encode_first_stage(init_img);
        if (init_image_latent.empty()) {
            LOG_ERROR("failed to encode init video frame");
            return std::nullopt;
        }

        latents.init_latent = sd_ctx->sd->generate_init_latent(request->width, request->height, request->frames, true);
        sd::ops::slice_assign(&latents.init_latent, 2, 0, init_image_latent.shape()[2], init_image_latent);

        latents.denoise_mask = sd::full<float>({latents.init_latent.shape()[0], latents.init_latent.shape()[1], latents.init_latent.shape()[2], 1, 1}, 1.f);
        sd::ops::fill_slice(&latents.denoise_mask, 2, 0, init_image_latent.shape()[2], 0.0f);

        latents.ref_images.push_back(start_image);

        int64_t t2 = ggml_time_ms();
        LOG_INFO("encode_first_stage completed, taking %" PRId64 " ms", t2 - t1);
    } else if (sd_version_is_longcat_avatar(sd_ctx->sd->version) && !start_image.empty()) {
```
**RISK:** verify this branch's `latents` variable is referenced by value (`latents.`) in the fork function —
the adjacent longcat_avatar branch uses `latents.init_latent`, so the style matches. Also confirm the branch
is placed in the SAME if/else-if chain (not a sibling function). Both confirmed by Read at fork 9135.

### C.8 — ref-image AREA resize.  **NO FORK EQUIVALENT — cannot apply.**

Upstream (ea4e566:5953) sets `condition_params.ref_image_params.vlm_resize_mode = RefImageResizeMode::AREA;`.
**The fork's `ConditionerParams` (conditioner.hpp:101) has NO `ref_image_params` field, and the fork has
NO `RefImageParams` struct and NO `RefImageResizeMode` enum anywhere** (`grep -rn` across `src/` = 0 hits;
upstream has them in `conditioner.hpp`). The entire VLM ref-image resize-mode infrastructure
(`resize_image_dims(... RefImageResizeMode mode)`, `vlm_min_size`/`vlm_max_size`, AREA vs LONGEST_SIDE) that
upstream's LLM conditioner uses is **absent from the fork's conditioner**. This quirk therefore **cannot be
wired as a shared-file one-liner** — it would require porting the `RefImageParams` struct + the AREA/
LONGEST_SIDE resize logic into the fork's `LLMEmbedder`/conditioner, which is out of scope for this doc.
**Flagged as a functional gap (see risks).**

---

## BLOCKERS / RISKS (summary)

1. **[premise correction] `gen_lingbot_video_ids` is NOT in the fork.** The task stated it was already present
   (merge notes rope.hpp H2). It is not — BOTH `gen_lingbot_video_ids` and `gen_lingbot_video_pe` must be
   added (A.2). Both are provided verbatim; deps (`gen_vid_ids` 8-arg/`t_offset`, `concat_ids`, 4-arg
   `embed_nd`) verified present and semantically identical.
2. **[functional gap] ref-image AREA resize (C.8) has no fork equivalent** — the fork conditioner lacks
   `RefImageParams`/`RefImageResizeMode`/`vlm_resize_mode`. Left unported. For lingbot IMG2VID the reference
   frame will be resized by the fork LLMEmbedder's default vision path, not the upstream AREA mode; this may
   change VLM conditioning fidelity vs upstream. Not a build blocker.
3. **[verify at apply] img2vid branch (C.7)** relies on `latents.ref_images.push_back` and value-semantics of
   `latents`; the adjacent longcat_avatar branch confirms the style, but the push into `ref_images` combined
   with the missing AREA-resize (risk 2) is the one place upstream/fork behavior may diverge.
4. **[not build-tested]** Server policy + task forbid building. All edits verified by source inspection only.
   Anchors are current-tree exact (grep/Read confirmed). After applying, a compile is required to confirm the
   three primitives resolve every symbol the .hpp ODR-uses.
5. **[no scheduler special-case]** lingbot joins the FLOW_PRED group with `default_flow_shift = 3.f`
   (upstream parity); if lingbot needs a different flow shift, add an else-if in the shift ladder (C.1).
