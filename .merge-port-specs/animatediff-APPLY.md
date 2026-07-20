# AnimateDiff (SD1.5 motion modules) — fork-adapted APPLY doc

Target: single-GPU graph-cut fork of sd.cpp. Worktree
`/home/dbrain/dev/longcat-avatar-mergewt`, branch `upstream-merge-2026-07-19`.
Merge target = upstream `ea4e566` (already has AnimateDiff fully wired).
Merge base = `19bdfe2` = current fork state (no AnimateDiff wiring).

**Do not build. Do not edit source from this doc's authoring pass — this is the plan.**

---

## 0. TL;DR — what AnimateDiff actually is (the tricky part)

AnimateDiff is **NOT a model dispatch**. It gets:

- **No `VERSION_` enum.** It is a motion-module add-on that rides an existing
  `VERSION_SD1 / VERSION_SD1_INPAINT / VERSION_SD1_PIX2PIX` UNet.
- **No `*Runner`, no conditioner, no VAE.** It reuses SD1.5's CLIP + AutoencoderKL.
- **No `backend_for(DIFFUSION)` / `params_backend_for(...)` construction.** No new
  runner is built, so the fork's `params_backend` vs upstream `weight_manager`
  convention divergence is *irrelevant to AnimateDiff* — the motion blocks live
  inside the existing `UnetModelBlock` (a `GGMLBlock`), which is identical between
  fork and upstream.

Activation is a **weight-name + boolean-flag** mechanism, in two independent layers:

1. **Graph layer (unet.hpp):** `UNetConfig::detect_from_weights` sets
   `enable_animatediff` when the motion tensors are present in the UNet tensor map;
   `UnetModelBlock` then instantiates an `AnimateDiff::AnimateDiffModel` block and
   `forward()` calls it via three `apply_motion_*` lambdas — but **only when
   `num_video_frames > 1`**. This layer is a **verbatim paste** into the fork
   (verified: fork unet.hpp == upstream unet.hpp except these hunks; see §2).

2. **Glue layer (stable-diffusion.cpp + public API):** load the motion-module file
   into the SAME `ModelLoader` as the UNet (prefix `model.diffusion_model.motion_module.`),
   set a runtime `animatediff_loaded` flag, thread the frame count into
   `UNetDiffusionExtra::num_video_frames`, decode the N latent frames one-by-one
   through the SD VAE, and dispatch `generate_video()` on an SD1 model to a small
   AnimateDiff driver instead of the LTX/Wan machinery. This layer **cannot be
   pasted verbatim** — the fork's `stable-diffusion.cpp` is ~13.5k lines of
   LTX/Wan-centric code and diverges hard from upstream's ~6k. Fork-adapted
   anchors are given in §4; genuine blockers in §5.

### Verdict by file

| File | Change | Confidence |
|---|---|---|
| `src/model/diffusion/animatediff.hpp` | **none** — already in fork, byte-identical to upstream, compiles against fork primitives | verified |
| `src/model/diffusion/unet.hpp` | 7 additive hunks, **verbatim** | verified (diff-confirmed) |
| `src/model/diffusion/model.hpp` | **none** — `UNetDiffusionExtra` field order already correct | verified |
| `src/stable-diffusion.cpp` | 5 fork-adapted glue edits (§4) + generate driver (§5) | anchors verified; driver needs fork work |
| `include/stable-diffusion.h` | add `motion_module_path` field | verified missing |
| `examples/common/common.{h,cpp}` | CLI plumbing for `--motion-module` | verified missing |

---

## 1. Header — `src/model/diffusion/animatediff.hpp` (NO CHANGE)

The file is already present in the fork tree and is **byte-identical to
`git show ea4e566:src/model/diffusion/animatediff.hpp`**. It contains no `*Runner`,
no `model_manager` / `weight_manager` / `ModelManager` references — it is a pure
`GGMLBlock` motion-module library.

**Verified against fork primitives** (all present, signatures match):

| Symbol used in header | Fork definition | Status |
|---|---|---|
| `GGMLBlock`, `blocks[...]`, `params[...]`, `init_params` override | `src/model/common/block.hpp` | OK |
| `Linear(in, out, bias)` | `block.hpp` | OK |
| `LayerNorm`, `GroupNorm` | `block.hpp` | OK |
| `FeedForward(dim, dim_out, mult, Activation::GEGLU)` | `block.hpp` | OK |
| `Embedding` (unused here but present) | `block.hpp` | OK |
| `GGMLRunnerContext` w/ `->ggml_ctx`, `->backend` | `core/ggml_extend.hpp` | OK |
| `ggml_ext_attention_ext(ctx, backend, q, k, v, n_head, mask, bool)` (8-arg) | `core/ggml_extend.hpp` | OK — header uses the 8-arg form |
| `String2TensorStorage`, `TensorStorage` | `model.h` | OK |

**Action:** confirm at build that `#include "core/ggml_extend.hpp"` and
`#include "model/common/block.hpp"` resolve (they do in-tree). No edit.

---

## 2. `src/model/diffusion/unet.hpp` — 7 VERBATIM additive hunks

`diff <(git show ea4e566:...unet.hpp) src/model/diffusion/unet.hpp` shows the fork
differs from upstream in exactly two orthogonal categories: **(a) these 7 AnimateDiff
hunks (missing in fork)** and **(b) the runner-ctor convention** (`params_backend`
vs `weight_manager`, already handled in the fork). The AnimateDiff hunks **never
touch the ctor**, so they apply cleanly. Paste each hunk exactly.

### Hunk 1 — include (top of file, after `#include "model/diffusion/model.hpp"`, fork line 9)
```cpp
#include "model/diffusion/animatediff.hpp"
```

### Hunk 2 — `UNetConfig` fields (after `int adm_in_channels = 2816;`, fork line 31)
```cpp
    bool enable_animatediff                = false;
    bool animatediff_has_mid_block         = false;
```

### Hunk 3 — detection (in `detect_from_weights`, immediately AFTER the `find_weight` lambda definition, i.e. after fork line 85 `};` and before the `if (const TensorStorage* input = find_weight("input_blocks.0.0.weight"))` block at fork line 87)
```cpp
        if (find_weight("motion_module.down_blocks.0.motion_modules.0.temporal_transformer.proj_in.weight") != nullptr) {
            config.enable_animatediff = true;
            if (find_weight("motion_module.mid_block.motion_modules.0.temporal_transformer.proj_in.weight") != nullptr) {
                config.animatediff_has_mid_block = true;
            }
        }
```
> Note: `find_weight` prefixes with the UNet prefix (`model.diffusion_model`), so it
> resolves `model.diffusion_model.motion_module.down_blocks.0...` — matching the load
> prefix in §4.3. This is the ONLY detection AnimateDiff needs. **No `get_sd_version`
> edit, no `VERSION_` branch in `model_loader.cpp`.**

### Hunk 4 — block instantiation (in `UnetModelBlock` ctor, AFTER `blocks["out.2"] = ...Conv2d(...)`, fork line 475)
```cpp

        if (this->config.enable_animatediff) {
            AnimateDiff::MotionModuleConfig mm_cfg;
            mm_cfg.enable_mid_block = this->config.animatediff_has_mid_block;
            blocks["motion_module"] = std::make_shared<AnimateDiff::AnimateDiffModel>(mm_cfg);
        }
```

### Hunk 5 — the three motion lambdas (in `UnetModelBlock::forward`, AFTER `hs.push_back(h);` for input block 0, fork line 585, and BEFORE the `// input block 1-11` comment at fork line 586)
```cpp

        auto motion_root        = config.enable_animatediff && num_video_frames > 1
                                      ? std::dynamic_pointer_cast<AnimateDiff::AnimateDiffModel>(blocks["motion_module"])
                                      : nullptr;
        auto apply_motion_input = [&](int input_block_idx, ggml_tensor* h_in) -> ggml_tensor* {
            if (!motion_root)
                return h_in;
            int di = (input_block_idx - 1) / 3;
            int mj = (input_block_idx - 1) % 3;
            if (di < 0 || di >= (int)channel_mult.size() || mj < 0 || mj >= num_res_blocks)
                return h_in;
            auto mm = motion_root->motion("down_blocks." + std::to_string(di) + ".motion_modules." + std::to_string(mj));
            if (!mm)
                return h_in;
            return mm->forward(ctx, h_in, num_video_frames);
        };
        auto apply_motion_output = [&](int output_block_idx, ggml_tensor* h_in) -> ggml_tensor* {
            if (!motion_root)
                return h_in;
            int ui = output_block_idx / 3;
            int mj = output_block_idx % 3;
            if (ui < 0 || ui >= (int)channel_mult.size() || mj < 0 || mj > num_res_blocks)
                return h_in;
            auto mm = motion_root->motion("up_blocks." + std::to_string(ui) + ".motion_modules." + std::to_string(mj));
            if (!mm)
                return h_in;
            return mm->forward(ctx, h_in, num_video_frames);
        };
        auto apply_motion_mid = [&](ggml_tensor* h_in) -> ggml_tensor* {
            if (!motion_root)
                return h_in;
            auto mm = motion_root->motion("mid_block.motion_modules.0");
            if (!mm)
                return h_in;
            return mm->forward(ctx, h_in, num_video_frames);
        };
```

### Hunk 6a — input-block call site (inside the input loop, AFTER `attention_layer_forward(...)` closes and BEFORE `mark_graph_cut(...)`, fork line 599–600)
The fork currently reads:
```cpp
                }
                sd::ggml_graph_cut::mark_graph_cut(h, "unet.input_blocks." + std::to_string(input_block_idx), "h");
                hs.push_back(h);
```
Insert `apply_motion_input` between the closing `}` and the `mark_graph_cut`:
```cpp
                }
                h = apply_motion_input(input_block_idx, h);
                sd::ggml_graph_cut::mark_graph_cut(h, "unet.input_blocks." + std::to_string(input_block_idx), "h");
                hs.push_back(h);
```

### Hunk 6b — middle-block call site (inside `if (!tiny_unet) { ... }`, AFTER the `middle_block.2` resblock, as the LAST statement in that block, fork line 626)
The fork currently reads:
```cpp
                h = resblock_forward("middle_block.2", ctx, h, emb, num_video_frames);             // [N, 4*model_channels, h/8, w/8]
            }
        }
        sd::ggml_graph_cut::mark_graph_cut(h, "unet.middle_block", "h");
```
Add `apply_motion_mid` before the inner-`if` and outer-`if` both close:
```cpp
                h = resblock_forward("middle_block.2", ctx, h, emb, num_video_frames);             // [N, 4*model_channels, h/8, w/8]
            }
            h = apply_motion_mid(h);
        }
        sd::ggml_graph_cut::mark_graph_cut(h, "unet.middle_block", "h");
```
> Placement matches upstream: `apply_motion_mid` is INSIDE `if (!tiny_unet)` but
> OUTSIDE the `if (version != VERSION_SDXL_SSD1B && ...)` inner block.

### Hunk 6c — output-block call site (inside the output loop, AFTER the attention `if (...find(...))` block closes and BEFORE the upsample `if (i > 0 && j == num_res_blocks)` block, fork line 661–663)
The fork currently reads:
```cpp
                    up_sample_idx++;
                }

                if (i > 0 && j == num_res_blocks) {
```
Insert `apply_motion_output` between them:
```cpp
                    up_sample_idx++;
                }

                h = apply_motion_output(output_block_idx, h);

                if (i > 0 && j == num_res_blocks) {
```

> **Verification note:** `output_block_idx` at this call site is the pre-increment
> value (the loop does `output_block_idx += 1;` later, near fork line 678). This
> matches upstream's `apply_motion_output(output_block_idx, h)` placement exactly —
> confirmed by the upstream diff. Do not move the call after the increment.

**Nothing else in unet.hpp changes.** The runner ctor already uses
`(backend, params_backend, tensor_storage_map, prefix, version)` and
`GGMLRunner::compute<float>(get_graph, n_threads, false)` — leave as-is.

---

## 3. `src/model/diffusion/model.hpp` — NO CHANGE

`UNetDiffusionExtra` already exists with the exact field order the wiring needs:
```cpp
struct UNetDiffusionExtra {
    int num_video_frames                           = -1;   // field 0
    const std::vector<sd::Tensor<float>>* controls = nullptr;
    float control_strength                         = 0.f;
};
```
Aggregate-init `UNetDiffusionExtra{nvf, &controls, control_strength}` is valid.
`UNetModelRunner::compute(...)` already reads `extra->num_video_frames` and threads
it into `build_graph`. **No edit.**

---

## 4. `src/stable-diffusion.cpp` — glue (fork-adapted anchors, VERIFIED)

All upstream line numbers in the draft `.merge-port-specs/animatediff.json` are WRONG
for the fork (they reference upstream's ~6k-line file). Use these **fork anchors**
(verified in the current worktree).

### 4.1 Runtime state flags — in the `sd` struct
**Fork anchor:** line 306 `bool apply_lora_immediately = false;` (inside the `sd`
struct). Add immediately after it:
```cpp
    bool animatediff_loaded    = false;
    int  animatediff_num_frames = 0;
```

### 4.2 `sd_version_supports_animatediff` file-static
**Fork anchor:** the `sd_version_supports_*` file-static region, e.g. near line 140
`static bool sd_version_supports_ref_latent_img_cfg(...)` or next to line 4581
`static bool sd_version_supports_video_generation(...)`. Add (anywhere in that
region, before first use):
```cpp
static bool sd_version_supports_animatediff(SDVersion version) {
    return version == VERSION_SD1 || version == VERSION_SD1_INPAINT || version == VERSION_SD1_PIX2PIX;
}
```
> The fork's `SDVersion` enum has all three (used elsewhere). Compiles unchanged.
> Note the fork's `sd_version_supports_video_generation` (line 4581) does NOT list
> SD1 — so an SD1 model is an *image* model, and `generate_video()` would reject it
> unless the AnimateDiff dispatch (§5) intercepts first. This matches upstream.

### 4.3 Motion-module LOAD into the shared `ModelLoader`
**Fork anchor:** the model-load block that ends the diffusion-model group — after the
`uncond_diffusion_model_path` `init_from_file` block (fork line ~448–452) and before
`bool is_unet = sd_version_is_unet(model_loader.get_sd_version());` (fork line 455).
Add:
```cpp
        if (strlen(SAFE_STR(sd_ctx_params->motion_module_path)) > 0) {
            LOG_INFO("loading motion module (AnimateDiff) from '%s'", sd_ctx_params->motion_module_path);
            if (!model_loader.init_from_file(sd_ctx_params->motion_module_path,
                                             "model.diffusion_model.motion_module.")) {
                LOG_WARN("loading motion module from '%s' failed", sd_ctx_params->motion_module_path);
            } else {
                animatediff_loaded = true;
            }
        }
```
> The `.motion_module.` prefix nests under `model.diffusion_model`, so
> `UNetConfig::detect_from_weights` (Hunk 3, prefix `model.diffusion_model`) sees
> `motion_module.down_blocks.0...` and flips `enable_animatediff`. `animatediff_loaded`
> here is the *runtime* flag on the `sd` struct (set inside the load method — confirm
> `this->animatediff_loaded` is reachable; the load method is a member of `sd`).
> **Do NOT** anchor "before control_net_path" as the draft says — the fork loads
> control_net via `control_net->load_from_file` (line 1571), not `init_from_file`.

### 4.4 Thread frame count into `UNetDiffusionExtra`
**Fork anchor:** line 3162, the denoise lambda's version dispatch:
```cpp
                if (sd_version_is_unet(version)) {
                    diffusion_params.extra = UNetDiffusionExtra{-1, &controls, control_strength};
                } else if (sd_version_is_sd3(version)) {
```
Replace the `UNetDiffusionExtra{-1, ...}` line with a computed frame count.
`noised_input` is in scope here (declared line 3092, assigned into
`diffusion_params.x` at line 3132):
```cpp
                if (sd_version_is_unet(version)) {
                    int nvf = -1;
                    if (animatediff_loaded && noised_input.dim() >= 4 && noised_input.shape()[3] > 1) {
                        nvf = static_cast<int>(noised_input.shape()[3]);
                    }
                    diffusion_params.extra = UNetDiffusionExtra{nvf, &controls, control_strength};
                } else if (sd_version_is_sd3(version)) {
```
> `nvf > 1` propagates to `UnetModelBlock::forward`'s `num_video_frames`, which is
> the SOLE trigger for `motion_root` (Hunk 5). When AnimateDiff is not loaded, `nvf`
> stays `-1` and behavior is byte-identical to today.
>
> There is a SECOND `UNetDiffusionExtra{}` at line 2148 (a separate helper compute
> path). Leaving it default (`num_video_frames == -1`) is safe: the runner's
> `build_graph` sets `num_video_frames = x->ne[3]`, so if that path ever runs a
> frame-batched latent it still activates motion; no edit required.

### 4.5 Per-frame VAE decode
**Fork anchor:** `decode_image_outputs(sd_ctx, request, final_latents)` at line 7157,
specifically its decode loop (line ~7168 `sd::Tensor<float> image = sd_ctx->sd->decode_first_stage(final_latents[i]);`).
Replace the single-shot decode with an AnimateDiff branch. The fork already has
`sd::ops::slice` (tensor.hpp:834 `namespace ops`, used at sd.cpp:2595/3759):
```cpp
    for (size_t i = 0; i < final_latents.size(); i++) {
        int64_t t1 = ggml_time_ms();
        if (sd_ctx->sd->animatediff_num_frames > 1 &&
            final_latents[i].dim() >= 4 &&
            final_latents[i].shape()[3] == sd_ctx->sd->animatediff_num_frames) {
            int n_frames = sd_ctx->sd->animatediff_num_frames;
            for (int f = 0; f < n_frames; ++f) {
                sd::Tensor<float> frame_latent = sd::ops::slice(final_latents[i], 3, f, f + 1);
                sd::Tensor<float> image        = sd_ctx->sd->decode_first_stage(frame_latent);
                if (image.empty()) {
                    LOG_ERROR("decode_first_stage failed for AnimateDiff frame %d/%d", f + 1, n_frames);
                    return nullptr;
                }
                decoded_images.push_back(std::move(image));
            }
        } else {
            sd::Tensor<float> image = sd_ctx->sd->decode_first_stage(final_latents[i]);
            if (image.empty()) {
                LOG_ERROR("decode_first_stage failed for latent %" PRId64, i + 1);
                // (keep the fork's existing free_params_buffer cleanup here)
                return nullptr;
            }
            decoded_images.push_back(std::move(image));
        }
        int64_t t2 = ggml_time_ms();
        LOG_INFO("latent %zu decoded, taking %.2fs", i + 1, (t2 - t1) * 1.0f / 1000);
    }
```
> Preserve the fork's existing error-path `free_params_buffer()` cleanup in the
> `else` branch (present at fork lines 7172–7176). `sd::ops::slice(t, 3, f, f+1)`
> extracts frame `f` on the batch/frame axis (dim 3) of the SD1 latent `[W,H,4,N]`.

---

## 5. BLOCKERS — needs hands-on fork work (do NOT paste upstream verbatim)

### 5.1 Public API field `motion_module_path` (VERIFIED MISSING)
`include/stable-diffusion.h` `sd_ctx_params_t` has no `motion_module_path`. Upstream
adds it right after `control_net_path` (upstream include line 203). Add to the fork
struct after `const char* control_net_path;` (fork line 198):
```cpp
    const char* motion_module_path;
```
Then in **`sd_ctx_params_init`** (fork sd.cpp line 4229) add a default
`sd_ctx_params->motion_module_path = "";` next to the other path defaults, and in
the debug-print helper (`sd_ctx_params_to_str`-style, near line 4278) if you want it
logged.

> **Struct-layout caution:** `sd_ctx_params_t` is initialized in
> `examples/common/common.cpp` via a **positional aggregate initializer** (line
> ~779–800, e.g. `control_net_path.c_str(),` then `embedding_vec.data(),`). Inserting
> `motion_module_path` into the struct **shifts every subsequent positional field**.
> You MUST add `motion_module_path.c_str(),` at the matching position in that
> initializer (right after `control_net_path.c_str(),`) or the whole struct silently
> misaligns. This is the single highest-risk edit in the port.

### 5.2 CLI plumbing (VERIFIED MISSING)
`examples/common/common.h` `SDParams` needs `std::string motion_module_path;`
(next to `control_net_path`, line ~133). `examples/common/common.cpp` needs an arg
entry (pattern of line 394 `&control_net_path`) e.g. `--motion-module`, plus the
positional aggregate-init entry from §5.1, plus a `load_if_exists("motion_module_path", ...)`
if JSON config is supported (pattern near line 1832).

### 5.3 `generate_animatediff_video()` + dispatch — **CONTRACT MISMATCH, rewrite required**
Upstream's `generate_animatediff_video` calls
`generate_image(sd_ctx, &img_gen_params, frames_out, num_frames_out)` — but the
**fork's `generate_image` has a different signature** (verified line 7380):
```cpp
SD_API sd_image_t* generate_image(sd_ctx_t* sd_ctx, const sd_img_gen_params_t* sd_img_gen_params);
```
It returns `sd_image_t*` (an array of `batch_count` images via `decode_image_outputs`)
and takes **no** `frames_out` / `num_frames_out` out-params. Upstream returns `bool`.
So upstream's driver **cannot be pasted**. Fork-adapted driver:
```cpp
static bool generate_animatediff_video(sd_ctx_t* sd_ctx,
                                       const sd_vid_gen_params_t* sd_vid_gen_params,
                                       sd_image_t** frames_out,
                                       int* num_frames_out) {
    int n_frames = sd_vid_gen_params->video_frames;
    if (n_frames < 1) { LOG_ERROR("AnimateDiff: --video-frames must be >= 1"); return false; }
    if (n_frames > 32) {
        LOG_WARN("AnimateDiff motion modules have a 32-frame positional-encoding context; capping to 32");
        n_frames = 32;
    }
    sd_img_gen_params_t img_gen_params;
    sd_img_gen_params_init(&img_gen_params);
    // copy the fork's actual sd_vid_gen_params_t -> sd_img_gen_params_t fields
    // (prompt, negative_prompt, clip_skip, width, height, sample_params, strength,
    //  init_image, seed, vae_tiling_params, cache, hires, circular_x/y, loras...).
    // VERIFY each field name against the FORK's structs (include/stable-diffusion.h);
    // do not assume upstream field parity.
    img_gen_params.batch_count = 1;

    sd_ctx->sd->animatediff_num_frames = n_frames;
    sd_image_t* imgs = generate_image(sd_ctx, &img_gen_params);   // fork signature: returns array
    sd_ctx->sd->animatediff_num_frames = 0;

    if (imgs == nullptr) return false;
    *frames_out     = imgs;          // decode_image_outputs emitted n_frames images (§4.5)
    *num_frames_out = n_frames;
    return true;
}
```
Dispatch: at the TOP of the fork's public **`generate_video`** (SD_API, line 13495),
after the null/out-init guards, add:
```cpp
    if (sd_ctx->sd->animatediff_loaded && sd_version_supports_animatediff(sd_ctx->sd->version)) {
        LOG_INFO("AnimateDiff dispatch: %d frames, %dx%d",
                 sd_vid_gen_params->video_frames, sd_vid_gen_params->width, sd_vid_gen_params->height);
        return generate_animatediff_video(sd_ctx, sd_vid_gen_params, frames_out, num_frames_out);
    }
```
> **Verify the fork's `generate_video` param names** (`frames_out`, `num_frames_out`,
> `audio_out`) at line 13495 and pass through accordingly. The fork's real entry may
> be `generate_video_ex` (line 10607) with `generate_video` as a thin wrapper — put
> the dispatch in whichever is the outermost public entry, before any LTX/Wan setup.

### 5.4 Latent frame dimension — **the real activation question**
For the motion lambdas to fire, the sampled latent must have shape `[W,H,4,N]` with
`N = animatediff_num_frames` on dim 3, so that §4.4's `noised_input.shape()[3] > 1`
holds. In upstream, `generate_image`'s init-latent stage replicates/creates the
N-frame latent when `animatediff_num_frames > 1` (upstream sd.cpp ~4727). **The fork
builds initial latents via `generate_init_latent(width, height)` (image path, 4D
single-frame, line 6855) — it does NOT frame-expand for AnimateDiff.**

Required fork edit (location, not verbatim): in the fork's **image** latent-prep path
(the `generate_init_latent(request->width, request->height)` call at line 6855 and
the pure-noise `init_latent` construction it feeds), when
`sd_ctx->sd->animatediff_num_frames > 1`, expand the latent's dim-3 to `n_frames`:
- t2v (no init image): allocate `[W,H,4,n_frames]` noise (the fork's RNG/noise fill
  must cover all N frames — check `randn_like` / `generate_init_latent` semantics).
- i2v (init image present): replicate the single encoded frame across the N slots
  using `sd::ops::slice_assign` (present, tensor.hpp:1146).

This is the one piece with no clean fork anchor and the most likely to need
iteration. **Flag: without this, the UNet runs but `num_video_frames == 1` and the
motion modules are inert (silent no-op) — exactly the failure mode the memory notes
warn about for variant swaps.**

---

## 6. Build-time checklist

1. `unet.hpp` includes `animatediff.hpp`; all 7 hunks applied at the anchors in §2.
2. `AnimateDiff::MotionModuleConfig`, `AnimateDiffModel`, `->motion(key)` resolve
   (header §1).
3. `sd_ctx_params_t` positional aggregate-init in `examples/common/common.cpp`
   re-aligned after inserting `motion_module_path` (§5.1) — count the fields.
4. `sd_version_supports_animatediff` declared before its first use (§4.2 vs §5.3
   ordering).
5. `noised_input` really is in scope at the §4.4 anchor (fork line 3162) — confirm
   the enclosing lambda captures it (it assigns `diffusion_params.x = &noised_input`
   at line 3132).
6. `sd::ops::slice` / `slice_assign` compile against 4D SD1 latents (dim 3).
7. Fork `generate_video` / `generate_image` signatures matched exactly in §5.3.
8. Smoke test: load an SD1.5 checkpoint + AnimateDiff motion module (`mm_sd_v15_v2`),
   `--video-frames 16`, confirm (a) `enable_animatediff=true` in the unet debug log,
   (b) `num_video_frames=16` reaches the UNet, (c) 16 distinct decoded frames out.
   If frames are identical → §5.4 latent frame-expansion is missing.

## 7. Symbols verified present in fork (no missing primitives in the graph layer)
`GGMLBlock`, `Linear`, `LayerNorm`, `GroupNorm`, `FeedForward`(GEGLU), `Embedding`,
`GGMLRunnerContext`(`ggml_ctx`,`backend`), `ggml_ext_attention_ext`(8-arg),
`ggml_ext_cont`, `ggml_view_3d`, `ggml_repeat`, `String2TensorStorage`,
`TensorStorage`, `UNetDiffusionExtra`, `sd::ops::slice`, `sd::ops::slice_assign`,
`decode_first_stage`, `generate_init_latent`, `SAFE_STR`, `ModelLoader::init_from_file`.
**No missing primitive blocks the graph layer.** All open risk is in the
stable-diffusion.cpp/API glue (§5).
