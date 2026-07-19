# Track B — "Who did it better" pass: Text Encoders + VAEs

Scope: `src/model/te/{clip,llm,t5}.hpp`, `src/model/vae/{vae,auto_encoder_kl,tae,ltx_vae,ltx_audio_vae,wan_vae}.hpp`.
Method: `git diff 19bdfe2 ea4e566 -- <file>` (upstream base→tip) compared against fork HEAD (`785ba0d`).
Bias: strongly KEEP-FORK when uncertain; only ADOPT clearly safe, isolated, beneficial, non-ModelManager changes.

---

## ADOPT

### ADOPT-1 — t5.hpp: real `T5Config::detect_from_weights` (config auto-detection incl. `inner_dim`)
**File:** `src/model/te/t5.hpp`
**Why:** Fork's `detect_from_weights` is a stub that `(void)`s the weights and returns hardcoded defaults. Upstream makes it actually read shapes from `tensor_storage_map`: `vocab_size`/`model_dim` from `shared.weight`, `model_dim`/`inner_dim`/`num_heads` from `encoder.block.0.layer.0.SelfAttention.q.weight` (num_heads = inner_dim/64 when divisible), `ff_dim` from `...DenseReluDense.wi_0.weight`, and `num_layers` by scanning `encoder.block.N`. Also adds a distinct `inner_dim` field so `T5Stack` receives inner_dim instead of a second copy of model_dim (fork currently passes `config.model_dim` twice at the `T5Stack(...)` construction).
**Self-contained / safe:** Pure detection function, no ModelManager dependency; symbols used (`starts_with`, `atoi`, `split`/`substr`, `std::max`) already in the fork. For the fork's live T5 models (t5-xxl, umt5-xxl) every detected value equals the existing hardcoded default (q.weight = [4096,4096] ⇒ inner_dim 4096, num_heads 64; wi_0 ⇒ ff_dim 10240; 24 blocks; umt5 vocab 256384), so it is behavior-preserving for current models while unlocking non-default T5 sizes.

**Fork anchor (the stub to replace) — `struct T5Config`:**
```cpp
    static T5Config detect_from_weights(const String2TensorStorage& tensor_storage_map,
                                        const std::string& prefix,
                                        bool is_umt5 = false) {
        (void)tensor_storage_map;
        (void)prefix;
        T5Config config;
        if (is_umt5) {
            config.vocab_size         = 256384;
            config.relative_attention = false;
        }
        return config;
    }
```
Add field `int64_t inner_dim = 4096;` to `T5Config` (after `model_dim`) and replace the stub body with upstream's detection (see `git diff 19bdfe2 ea4e566 -- src/model/te/t5.hpp`, lines adding `find_tensor`/shape reads/`detected_layers`).

**Second anchor — `T5Model` ctor (make inner_dim actually flow through):**
```cpp
        blocks["encoder"] = std::shared_ptr<GGMLBlock>(new T5Stack(config.num_layers,
                                                                   config.model_dim,
                                                                   config.model_dim,   // <-- change 2nd arg to config.inner_dim
                                                                   config.ff_dim,
```
Change the second `config.model_dim` to `config.inner_dim`. (`T5Stack`/`T5Block`/`T5Attention` already take and use an `inner_dim` parameter in the fork — verified — so no block-level change is needed.)

**Do NOT** pull the surrounding ctor/`compute()`/`T5Embedder`/test-main hunks: those are weight_manager/ModelManager threading = KEEP-FORK.

---

### ADOPT-2 — vae.hpp: separate image-VAE encode tile factor (2.0) from video (1.30539) — *low value, VRAM caveat*
**File:** `src/model/vae/vae.hpp`, in `VAE::encode(...)`
**Why:** Upstream splits the encode tile factor so image VAEs default to 64-latent-pixel tiles (a 512px SD image encodes as a single tile → no encode-time tile seams), while video VAEs (wan/hunyuan/ltxav) keep the legacy 1.30539f. Fork uses 1.30539f for every VAE. Gated to non-video, so the LTX/Wan video engine is untouched.
**Caveat (why low-value, not "clearly safe"):** Larger image-encode tiles raise per-tile VRAM for the flux/SD/flux2 image VAEs on the 3060. It is a quality win for image encode but changes the memory footprint of a live path — owner should GPU-validate before keeping. Include only if that trade is wanted.

**Fork anchor — `VAE::encode`:**
```cpp
            get_tile_sizes(tile_size_x, tile_size_y, tile_overlap, tiling_params, W, H, 1.30539f);
```
**Upstream replacement:**
```cpp
            // Image VAE encode is more sensitive to tile boundary context than decode.
            // Keep the smaller legacy factor for video VAEs, but default image encode
            // tiles to 64 latent pixels so a 512px SD image is encoded as one tile.
            const float encode_tile_factor = (sd_version_is_wan(version) || sd_version_is_hunyuan_video(version) || sd_version_is_ltxav(version)) ? 1.30539f : 2.0f;
            get_tile_sizes(tile_size_x, tile_size_y, tile_overlap, tiling_params, W, H, encode_tile_factor);
```
Referenced helpers `sd_version_is_wan` / `sd_version_is_hunyuan_video` / `sd_version_is_ltxav` all exist in the fork (`src/model.h`). Do NOT pull the neighboring ctor/`weight_prefix`/`get_scale_factor` hunks (ModelManager/new-arch — see KEEP-FORK).

---

## KEEP-FORK (grouped)

**A. All ModelManager / weight_manager / residency threading (the whole point of the fork's architecture).**
Every file: ctor signature swap `ggml_backend_t params_backend` → `std::shared_ptr<RunnerWeightManager> weight_manager`; `GGMLRunner(backend, weight_manager)`; `weight_prefix` member + `get_param_tensors(tensors)` (dropped `prefix` arg); `get_params_buffer_size`→`get_params_mem_size`; removal of `alloc_params_buffer()`; the `_test`/`_main` harnesses rewritten to `ModelManager::register_runner_params(...) + validate_registered_tensors()`. Fork's runners deliberately take `(backend, params_backend, ...)` with graph-cut/offload — explicitly out of scope per brief.

**B. `compute<float>(..., auto_free, free_compute_buffer, free_compute_params)` free-flag plumbing.** clip/t5/llm/aekl/tae/ltx_vae/ltx_audio_vae/wan_vae all gain 2 extra compute-buffer free flags. This is the upstream compute-buffer-lifetime scheme, coupled to weight_manager residency; the fork manages compute buffers via its own graph-cut path. KEEP-FORK.

**C. Host-readback → in-graph scalar conversions (required only because weight_manager makes params non-host-resident).**
- `auto_encoder_kl.hpp` `get_alpha()`: `ggml_ext_backend_tensor_get_f32(mix_factor)` + host sigmoid → in-graph `ggml_ext_cast_f32` + `ggml_sigmoid` + `ggml_mul`.
- `ltx_vae.hpp` (×2): `timestep_scale_multiplier` host readback + `ggml_ext_scale` → in-graph `ggml_ext_cast_f32` + `ggml_mul`.
Fork's params are host-resident, so the existing scalar-readback path is correct and simpler. Not a bug in the fork. KEEP-FORK.

**D. `ggml_ext_pad_ext` / `ggml_ext_conv_3d` backend-arg signature change.** Upstream's wrappers gained a leading `ctx->backend` parameter (tae.hpp, ltx_audio_vae.hpp, wan_vae.hpp). The fork's `ggml_ext_pad_ext` has a *different* signature (no backend; `circular_x/y` tail) and the fork still uses raw `ggml_pad_ext` in TAEHV. Adopting would require the upstream ggml_ext API, not just the call sites. Architectural / not portable in isolation. KEEP-FORK.

**E. Qwen3-VL vision cluster in `llm.hpp` — RISKY, live path, unverifiable checkpoint layout.**
Upstream adds `VisionParams::split_patch_embed`, vision config auto-detection (patch_size/in_channels/hidden_size/num_position_embeddings/intermediate_size/out_hidden_size/vision num_layers from `visual.*` weights), the `(QWEN3||QWEN3_VL)&&num_layers==28 ⇒ num_heads=16` guard, and a **restructured `VisionPatchEmbed`**: renames `llama_cpp_style`→`split_patch_embed`, sets Conv2d `bias=false` on `proj.0`/`proj.1`, and instead applies a single `params["bias"]` tensor *after* summing the two split convs.
This is a genuine correctness rework **only if** the Qwen3-VL checkpoints store a single combined `visual.patch_embed.bias` (which upstream's detection implies) rather than per-conv `proj.{0,1}.bias` (what the fork's `bias=(arch==QWEN3_VL)` on each Conv2d assumes). The fork's Qwen3-VL vision is a **live path** (`hidream_o1.hpp`, `conditioner.hpp` ref-image encode via `encode_image`). Which bias layout the fork's actual checkpoints use cannot be determined statically. Per the "cannot afford a destabilized engine" mandate this is a HOLD, not an adopt. **Recommended follow-up (verify, don't blind-adopt):** if a real Qwen3-VL edit checkpoint shows a missing/zeroed patch-embed bias in the fork, port upstream's `split_patch_embed`+combined-bias rework as a unit; otherwise leave fork as-is.

**F. `wan_vae.hpp` 2D-VAE (`is_2D` / `Conv2dBut3d` / `VERSION_QWEN_IMAGE_LAYERED`) feature — rewrite, out of conservative scope.**
The 220-line insertion is a new-architecture feature (Qwen-Image layered 2D VAE): new `Conv2dBut3d` class, `is_2D` plumbed through `ResidualBlock`/`Encoder3d`/`Decoder3d`/`WanVAE`/`WanVAERunner`, `input_channels`/`patch_size` generalization, `is_2D` weight-shape detection (`decoder.conv1.weight` ne[2]>3). No isolated bugfix is separable from it, it targets a version the fork doesn't carry, and it rewrites the fork's video-engine Wan VAE. KEEP-FORK.

**G. `auto_encoder_kl.hpp` misc:**
- `sd_version_uses_flux_vae(version)` helper replacing the `is_flux||is_z_image||is_longcat` disjunction: cosmetic refactor; helper not defined in fork; fork's explicit form is correct. KEEP-FORK.
- `sd_version_is_sefi_image` latent slice in `diffusion_to_vae_latents` and the `minit2i` scale_factor path: **sefi_image and minit2i are handled by their own Track-A port specs** (`.merge-port-specs/sefi_image.json`, `minit2i-APPLY.md`) — defer to those, not this pass.

**H. `vae.hpp get_scale_factor` new-arch entries** (`hunyuan_video ⇒ 16`, `minit2i ⇒ 1`): new-model scale factors. hunyuan-video VAE is a separate fork port (`hunyuan.json`); minit2i via its own spec. If the minit2i port needs scale_factor==1 it should be added by that spec, not here. Not a general TE/VAE fix. KEEP-FORK (defer).

**I. `tae.hpp` hunyuan-video TAEHV entries + conditional encode permute.** `z_channels=32/patch=2` for hunyuan_video and gating the encode/decode permute on `is_wan||hunyuan||ltxav`: new-arch (hunyuan) + a guard that is behaviorally identical for the fork's TAEHV usage (only ever video). Belongs with the hunyuan-video port if wanted. KEEP-FORK.

**J. `clip.hpp`** — diff is purely the weight_manager ctor + compute free-flags (groups A/B). Nothing to adopt.

---

## Summary

Two ADOPT items, both self-contained and non-ModelManager. The clear winner is **ADOPT-1 (t5.hpp `detect_from_weights`)**: it replaces a genuine fork stub with real weight-shape config detection (and wires the already-present `inner_dim` through `T5Stack`), is behavior-preserving for the fork's current t5-xxl/umt5 models, and unlocks non-standard T5 checkpoints at zero risk. **ADOPT-2 (image-VAE encode tile factor 2.0)** is a lower-value image-encode quality tweak gated away from the video engine, but it shifts image-encode VRAM and should be GPU-validated before keeping. Everything else is KEEP-FORK: weight_manager/ModelManager threading, compute-buffer free flags, host-readback→in-graph scalar conversions (only needed under lazy residency), the `ggml_ext_*` backend-arg API change, and — held deliberately for risk — the Qwen3-VL vision patch-embed rework (live path, unverifiable bias layout) and the wan_vae 2D/Qwen-Image feature rewrite. sefi_image / minit2i / hunyuan-video touches are deferred to their existing Track-A port specs.
