# Diffusion / common / adapter merge audit — fork HEAD vs upstream leejet

Scope: `src/model/diffusion/{anima,control,dit,flux,hidream_o1,ideogram4,lens,ltxv,mmdit,model,pid,qwen_image}.hpp`,
`src/model/common/{block,rope}.hpp`, `src/model/adapter/{lora,pmid}.hpp`.

Refs: merge base `19bdfe2`, upstream tip `ea4e566`, fork HEAD `c263e53`.
Method: per file, `git diff 19bdfe2 ea4e566 -- <file>` (what upstream changed) compared against fork HEAD
(`git show HEAD:<file>`), plus `git diff ea4e566 HEAD` (how fork diverges from upstream).

## File-level relationship (computed by blob hash)

| File | fork==base? | fork==uptip? | verdict |
|---|---|---|---|
| anima.hpp | yes | no | took-fork (upstream changed, we kept base) |
| control.hpp | yes | no | took-fork |
| hidream_o1.hpp | yes | no | took-fork |
| ideogram4.hpp | yes | no | took-fork |
| lens.hpp | yes | no | took-fork |
| pid.hpp | yes | no | took-fork |
| qwen_image.hpp | yes | no | took-fork |
| lora.hpp | yes | no | took-fork |
| pmid.hpp | yes | no | took-fork |
| dit.hpp | no | no | partial (Track A adopt + kept old fns) |
| flux.hpp | no | no | partial (Track A adopt + kept-fork + drops) |
| mmdit.hpp | no | no | partial (Track A adopt + kept-fork) |
| model.hpp | no | no | partial (kept-fork + fork additions + drops) |
| block.hpp | no | no | partial (fork reverted upstream + own feature) |
| rope.hpp | no | no | partial (Track A adopt + kept-fork + drops) |
| ltxv.hpp | no | no | partial (massive fork engine; upstream barely touched) |

Reading of "took-fork": for all 9 took-fork files the *dominant* upstream change is the runner-convention
migration (`ggml_backend_t params_backend` → `std::shared_ptr<RunnerWeightManager> weight_manager` +
`ModelManager::register_runner_params` + `compute<float>(…, false,false,false)` signature). That IS the
declared keep-fork core (backend/params_backend vs ModelManager/weight_manager), so "we kept base" is
**correct and intentional** for the convention. The risk is the *other* upstream changes that ride along in
the same commits and got shed with it. Those are itemized below.

---

## Change ledger

Classification: **ADOPT** (self-contained, better/equal, take it) · **ALREADY-HAVE** (fork already has it,
usually via Track A) · **KEEP-FORK** (fork convention/core, correctly not shed) · **DROPPED-FEATURE**
(upstream capability with no fork equivalent).

| File | Change | Class | Rationale / anchor |
|---|---|---|---|
| **model.hpp** | `RefImageParams` + `REF_IMAGE_PRESETS` + `RefImageResizeMode` framework; `DiffusionParams.increase_ref_index` → `ref_image_params` | **DROPPED-FEATURE** | Keystone editing/reference framework. Fork kept the old `bool increase_ref_index`. See dedicated section below — core DiT-side ref still works via the bool; the *framework* (presets, VLM routing, DECREASE, resize modes, force_ref_timestep_zero) is gone. |
| model.hpp | `FluxDiffusionExtra.pulid_id` / `pulid_id_weight` | DROPPED-FEATURE | Part of flux-PuLID (see flux.hpp). |
| model.hpp | `MiniT2IDiffusionExtra`, `HunyuanVideoDiffusionExtra` | ALREADY-HAVE | Present in fork (Track A minit2i/hunyuan). Fork also adds its own `LongCatAvatarDiffusionExtra` + rich `LTXAVDiffusionExtra` (LipDub/NAG/video_reference) — keep-fork. |
| model.hpp | weight_manager runner ctor; virtual param/`set_*` methods | KEEP-FORK | Fork keeps `params_backend` ctor and adds avatar/LTX virtuals (`has_audio_stream`, `set_circular_axes`, `set_max_graph_vram_bytes` forwarding). Deliberate core. |
| **rope.hpp** | `enum RefIndexMode {FIXED,INCREASE,DECREASE}` replacing `bool increase_ref_index` across `gen_refs_ids/gen_flux_ids/gen_qwen_image_*/gen_z_image_*` | DROPPED-FEATURE (drift) | Fork reverted to the bool. `INCREASE`/`FIXED` are covered by the bool; **`DECREASE` is genuinely lost** (used only by qwen_layered). |
| rope.hpp | `embed_interleaved_mrope(..., axis_wrap_dims)` | DROPPED-FEATURE | Fork reverted the extra param → ideogram4 seamless/circular tiling can't be wired (see ideogram4). |
| rope.hpp | `gen_qwen_image_ids/pe` gain `t` (time) axis; `txt_id_start = max(h,w)/2` + changed linspace endpoint; img_ids via `gen_vid_ids(scale_rope)` | DROPPED-FEATURE + numeric | Fork reverted to `txt_id_start = max(h,w)`, `gen_flux_img_ids`, no `t`. The `t` support is for qwen video/layered (dropped). **The `/2` + endpoint change also alters *standard* Qwen-Image rope positions** — a numeric divergence from upstream, not just layered. Flagged for A/B (see ADOPT-with-caution). |
| rope.hpp | `gen_hunyuan_video_pe`, `gen_lingbot_video_pe`, `gen_vid_ids` | ALREADY-HAVE | Present in fork (Track A), just relocated to end of namespace. |
| rope.hpp | `attention_pe` uses `n_head=q->ne[1]` (was `v->ne[1]`) | KEEP-FORK (minor) | Fork keeps `v->ne[1]` and instead adds fused-RoPE (`ggml_rope_pe`/`_ni`, `allow_fused_rope`) + `flash_attn`/`flash_skip_kv_pad` params + `gen_vid_ids_ref`/`gen_wan_pe_ref` (avatar). For self-attn q/k/v n_head is equal so the upstream change is inert; fork's fused path is the real work. |
| **flux.hpp** | SEFI image model (`is_sefi`, `SefiDualTimestepEmbeddings`, `sefi_image.hpp`, `semantic_channels`, version branch) | DROPPED-FEATURE (tracked) | Fork removed the flux wiring. `sefi_image.hpp` **exists in fork** (scaffold) and `.merge-port-specs/sefi_image.json` exists → this is a *tracked pending port*, not a silent drop. No `VERSION_SEFI`/`sd_version_is_sefi` in fork. |
| flux.hpp | flux PuLID (`pulid_ca.*` blocks, `PuLIDPerceiverAttentionCA` double+single injection, `pulid.hpp`, `pulid_enabled/intervals`) | DROPPED-FEATURE (tracked) | Fork removed flux wiring + the `forward_orig` pulid args. `pulid.hpp` **exists in fork** (scaffold, referenced only by itself) → tracked, not silent. |
| flux.hpp | `chroma_use_dit_mask` model-arg parsing; `use_mask` default `true` | DROPPED-FEATURE (minor) | Fork keeps the old `use_mask` ctor param (default `false`). Chroma DiT-mask default + runtime toggle lost. `parse_key_value_args` exists in fork, so re-adopt is cheap. |
| flux.hpp | `LastLayer` gains `patch_volume` | ALREADY-HAVE | Track A. Fork HEAD == upstream on this. |
| flux.hpp | weight_manager ctor / RefIndexMode params / compute bools | KEEP-FORK | Fork keeps `params_backend` + `increase_ref_index`. Consistent with model.hpp/rope.hpp. |
| **qwen_image.hpp** | `VERSION_QWEN_IMAGE_LAYERED` variant: `use_additional_t_cond`, `addition_t_embedding`, 3D time-axis patchify/unpatchify path, `DECREASE` ref index | DROPPED-FEATURE | Not in fork (no `QWEN_IMAGE_LAYERED`, no `use_additional_t_cond`). Layered/video Qwen image editing. Depends on rope DECREASE + `patchify_3d`/`unpatchify_3d` (fork HAS those). |
| qwen_image.hpp | `qwen_image_zero_cond_t` model-arg parsing | DROPPED-FEATURE (minor) | Runtime toggle for `zero_cond_t`; fork keeps the ctor bool. Cheap. |
| qwen_image.hpp | ROCm `set_force_prec_f32` | KEEP-FORK/minor | See block.hpp ROCm note. |
| qwen_image.hpp | weight_manager ctor / RefIndexMode | KEEP-FORK | Convention. |
| **pid.hpp** | PixelDiT **1.5** (`pit_lq_inject`): `replicate_pad_2d`, `SigmaAwareGate` per-token, `PiDResBlock` replicate-pad, `LQProjection2D` unpatchify+`pit_head`, `pit_lq_gate`, 128-ch latent detection | DROPPED-FEATURE | Self-contained to pid.hpp. Niche leejet Pixel-DiT super-res model; low value for kobbler. Not tracked. |
| pid.hpp | weight_manager ctor | KEEP-FORK | Convention. |
| **anima.hpp** | Reference-latent conditioning: `ref_latents` concat in `forward`, `gen_pe(…, ref_latents, RefIndexMode::FIXED)`, `RefImageParams`/`cosmos_reference` preset in `compute` | DROPPED-FEATURE | Anima reference/editing (Cosmos-reference). Depends on RefImageParams + RefIndexMode (both dropped). Blast radius = anima.hpp + the model.hpp/rope.hpp framework. |
| anima.hpp | weight_manager ctor | KEEP-FORK | Convention. |
| **ideogram4.hpp** | circular/seamless tiling: `gen_ideogram4_pe(…, circular_x, circular_y)` → `embed_interleaved_mrope(axis_wrap_dims)` | DROPPED-FEATURE | Fork has `runner_ctx.circular_x/y_enabled` already, but rope `embed_interleaved_mrope` lacks `axis_wrap_dims` (reverted). So ideogram4 seamless tiling is unreachable. Self-contained once rope param restored. |
| ideogram4.hpp | weight_manager ctor; `get_context()` hoist | KEEP-FORK | Convention. |
| **control.hpp** | `CONTROL_NET_GRAPH_SIZE` `1536` → `MAX_GRAPH_SIZE` (SDXL ControlNet transformer_depth=10 fit) | **ADOPT** | Fork still has `1536` (line 7). `MAX_GRAPH_SIZE`(=327680) is defined in `ggml_extend.hpp` and in scope. One-line fix; only matters if SDXL ControlNet is exercised, but it's a real "graph too small" bug guard. Anchor: `#define CONTROL_NET_GRAPH_SIZE MAX_GRAPH_SIZE`. |
| control.hpp | guided-hint cache via `cache()`/`get_cache_tensor_by_name`, `ggml_set_output` instead of `alloc_control_ctx`; ModelManager load | KEEP-FORK | Tied to weight_manager + GGMLRunner cache infra (fork convention). Correctly not shed. |
| **lora.hpp** | LoKr bookkeeping bug: `if (lokr_w2_a) applied_lora_tensors.insert(lokr_w2_name)` → `…lokr_w2_a_name` | **ADOPT** | Fork HEAD still buggy at **line 615** (the `apply_runtime`/`stat` block; the build_lora_graph block at ~445 is already correct). Effect is cosmetic (spurious "unused tensor" report for LoKr `w2_a`), not a correctness bug, but a clean self-contained fix. |
| lora.hpp | ModelManager/`register_param_tensors`/`release_loaded_tensors`; `preprocess_lora_tensors(set<string>)`; `warn_unused` | KEEP-FORK | Convention. |
| **pmid.hpp** | weight_manager/ModelManager migration only | KEEP-FORK | No feature content. |
| **hidream_o1.hpp** | weight_manager, `compute(auto_free,…)`, `runner_done`, drop of `alloc/free_params_buffer` overrides | KEEP-FORK | Pure convention. No dropped feature. |
| **lens.hpp** | weight_manager + compute bools | KEEP-FORK | Pure convention. No dropped feature. |
| **dit.hpp** | `patchify`→`patchify_3d` / `unpatchify`→`unpatchify_3d` rename + `patch_last` flag | ALREADY-HAVE (superset) | Fork adopted `patchify_3d`/`unpatchify_3d` (Track A) **and kept** the old `patchify`/`unpatchify` for existing callers. Superset — nothing dropped. |
| **mmdit.hpp** | `PatchEmbed` 2D/3D (Conv3d, tuple ctor, 3D forward) | ALREADY-HAVE | Track A. fork HEAD == upstream on PatchEmbed. |
| mmdit.hpp | weight_manager ctor + `register_runner_params` test() | KEEP-FORK | Fork kept `params_backend`. Convention. |
| **block.hpp** | `AlphaBlender::get_alpha` CPU-readback → graph-side (`ggml_ext_cast_f32`+`ggml_sigmoid`) | ADOPT (low-pri) | Fork **reverted** to CPU readback. Graph-side is offload-friendlier (no device→host sync at build), which aligns with the keep-fork deferred-weight core — but it's SVD temporal mixing, low traffic. Anchor: upstream `get_alpha(GGMLRunnerContext*)`. |
| block.hpp | `sd_backend_is("ROCm")` added to `force_prec_f32` gate (FeedForward + AlphaBlender) | ADOPT (trivial) | Fork **reverted** to Vulkan-only. We run CUDA (3060/5060Ti), so functionally moot, but it's a silently-dropped upstream portability line; re-adopt is trivial. Also applies in qwen_image.hpp. |
| block.hpp | `LONGCAT_FFN_TILE_TOKENS` token-tiled FFN | KEEP-FORK | Fork's own long-video VRAM feature; not an upstream item. |
| **ltxv.hpp** | `modulate(…)` → `LTXV::modulate(…)` namespace qualifier; weight_manager ctor; compute bools | KEEP-FORK | Upstream barely touched ltxv (namespace disambiguation + convention). Fork's +1184-line divergence is the LTX/audio video engine (keep-fork core). **No upstream ltxv feature dropped.** |

---

## The RefImageParams framework — how much is actually lost

The owner's worry lands hardest here. Precise picture:

- Fork retained `bool increase_ref_index` end-to-end (model.hpp → rope `gen_refs_ids`/`gen_flux_ids`/
  `gen_qwen_image_*`/`gen_z_image_*`). So the **DiT-side reference-latent mechanism still works** for
  `FIXED` (bool=false) and `INCREASE` (bool=true). flux_kontext, flux2-edit, qwen-edit, z_image ref paths
  are NOT dead.
- What the framework adds on top, and is genuinely absent in fork:
  1. **`RefIndexMode::DECREASE`** — only consumer is qwen_layered. Lost with that model.
  2. **Preset table** (`REF_IMAGE_PRESETS`: flux_kontext/longcat/flux2/qwen/qwen_layered/z_image_omni/
     krea2_ostris_edit/krea2_edit/cosmos_reference) — centralizes per-model ref policy including
     `pass_to_vlm`, `pass_to_dit`, `force_ref_timestep_zero`, `resize_before_vae`,
     `vae_input_max_pixels`, `vlm_resize_mode/min/max`. Much of this is VLM-preprocessing/plumbing that
     lives OUTSIDE the assigned files (conditioner/pipeline). Within these files the only DiT-visible loss
     is DECREASE + the `pass_to_dit` gate (fork always passes ref latents when present).
  3. **krea2 edit presets** (`krea2_ostris_edit`, `krea2_edit` with LONGEST_SIDE 768 resize). krea2 is a
     Track A port that "built"; confirm whether its edit variant relies on these presets — if the krea2
     port only does base t2i, no loss; if it does img-edit, the resize/index policy is missing.

Net: the **catastrophic** reading ("we can't do reference/editing at all") is FALSE. The **accurate**
reading: the expanded, preset-driven, VLM-aware editing framework + DECREASE mode were shed alongside the
convention revert. Re-adopting it is a cross-cutting change (model.hpp + rope.hpp signatures + every
editing runner's `compute`), not self-contained, and it collides head-on with the kept `increase_ref_index`
convention — so it should be a deliberate decision, not a drive-by.

---

## Counts

- Files audited: **16**.
- KEEP-FORK (correct, do not shed): runner-convention on all 9 took-fork files + mmdit/model/ltxv;
  fused-RoPE + LTX engine + FFN tiling. No action.
- ALREADY-HAVE (Track A, verified present): dit (patchify_3d + kept old), mmdit PatchEmbed 2D/3D,
  flux LastLayer patch_volume, rope hunyuan/lingbot video PE, model MiniT2I/Hunyuan extras. **5** — no action.
- **ADOPT** (self-contained, take it): **4**
  - lora.hpp LoKr `lokr_w2_a_name` bugfix (line 615) — cosmetic but clean.
  - control.hpp `CONTROL_NET_GRAPH_SIZE = MAX_GRAPH_SIZE` — SDXL ControlNet fit guard.
  - block.hpp/qwen_image.hpp ROCm `force_prec_f32` gate — trivial portability (moot on our CUDA HW).
  - block.hpp graph-side `AlphaBlender::get_alpha` — offload-friendly, low-traffic (SVD).
  - (borderline) rope qwen_image `txt_id_start` numeric change — ADOPT-WITH-CAUTION, needs A/B; it alters
    standard Qwen-Image rope, entangled with the dropped layered path.
- **DROPPED-FEATURE**: **6 distinct capabilities**
  1. RefImageParams/REF_IMAGE_PRESETS/RefIndexMode-DECREASE editing framework (model+rope+anima+qwen+flux).
     Blast radius: large/cross-cutting; collides with kept `increase_ref_index`. Core ref still works;
     only DECREASE + preset/VLM policy lost. **Highest-scope, but partly redundant — decide deliberately.**
  2. qwen_layered (`VERSION_QWEN_IMAGE_LAYERED`) video/layered edit — qwen_image + rope DECREASE. Medium
     effort; needs #1. Not tracked.
  3. flux PuLID identity adapter — **scaffolded** (`pulid.hpp` present) → tracked pending port. Effort:
     re-wire flux `pulid_ca` + model extra. Medium.
  4. flux SEFI image model — **scaffolded** (`sefi_image.hpp` + `sefi_image.json` spec) → tracked pending
     port. Effort: re-wire flux is_sefi branch + version enum. Medium.
  5. PixelDiT 1.5 (`pit_lq_inject`) — self-contained to pid.hpp. Niche super-res model; low value. Not tracked.
  6. Ideogram4 seamless/circular tiling — ideogram4 + rope `axis_wrap_dims`. Self-contained-ish; niche.
  - Minor toggles also dropped: `chroma_use_dit_mask`, `qwen_image_zero_cond_t` model-args (cheap re-adds).

## Highest-value item

**ADOPT the lora.hpp LoKr `lokr_w2_a_name` fix (line 615)** — the only strictly-better, zero-risk,
self-contained correctness/bookkeeping fix in scope; fork demonstrably still carries the bug that upstream
fixed. `CONTROL_NET_GRAPH_SIZE = MAX_GRAPH_SIZE` is the next self-contained adopt if SDXL ControlNet is in use.

## Biggest DROPPED-FEATURE

The **RefImageParams editing framework + `RefIndexMode::DECREASE`** (and the qwen_layered model that needs
it). It is the one genuinely large capability shed alongside the convention revert — but note it is only
*partially* lost: fork's retained `increase_ref_index` still drives FIXED/INCREASE reference conditioning,
so flux-kontext/flux2/qwen/z_image reference paths are intact. What's missing is DECREASE, the preset/VLM
policy layer, and qwen_layered. **PuLID and SEFI are already scaffolded (headers + a json spec) and thus
tracked, not silent drops** — they need re-wiring into flux, not a from-scratch port.
