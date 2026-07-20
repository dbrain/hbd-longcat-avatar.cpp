# Audit — "Who did it better + what did we silently drop"
Fork HEAD (`c263e53`, branch `upstream-merge-2026-07-19`) vs upstream leejet `ea4e566`, merge base `19bdfe2`.

Scope of THIS ledger: the PART-1 files assigned (conditioner, te/clip+llm, extensions, examples CLI/server), the 12 upstream files ABSENT from our tree (PART 2), and docs (PART 3). TE/VAE "who did it better" was done in `trackB-te-vae.md`; this doc cross-references it and covers only what it missed (conditioner integration side, encode_image, extensions, the dropped features).

Method per file: `git diff 19bdfe2 ea4e566 -- <file>` (upstream base→tip) compared to fork HEAD content. Bias: KEEP-FORK core = graph-cut VRAM/offload runner convention (`backend`/`params_backend` vs upstream `ModelManager`/`weight_manager`), NVFP4/FP8 cuBLASLt GEMM, LTX/Wan/audio video engine, SA3, longcat-avatar. Otherwise ADOPT upstream where better-or-equal + self-contained.

---

## PART 1 — differing assigned files

### 1a. `src/conditioning/conditioner.hpp` — RefImageParams / AREA-resize gap  ★ owner priority
Upstream diff is 1789 lines; the load-bearing new capability is the **ref-image resize framework**. Full port plan in the dedicated section below. Summary of the gap:

- Upstream adds to `ConditionerParams` a single field `RefImageParams ref_image_params;` (defined in `src/model/diffusion/model.hpp`, which the fork's `ConditionerParams` **lacks entirely**).
- Upstream adds a member helper `void resize_image_dims(height,width, h_bar&,w_bar&, factor,min_size,max_size, RefImageResizeMode mode)` implementing **AREA**, **LONGEST_SIDE**, and a `min==max` special case, then replaces every inline ref-image resize block (qwen, longcat, lingbot_video, z_image, hunyuan, …) with a call to it, parameterised by `conditioner_params.ref_image_params.vlm_resize_mode / vlm_min_size / vlm_max_size`.
- The fork instead carries **three hand-inlined copies** of an AREA-only resize with hardcoded constants: `qwen_image` branch (line ~1982, min `384²` / max `560²`), `longcat` branch (~2050, same), `lingbot_video` branch (~2282, min `2*factor` / max `128*factor` — see BUG below). The fork even left a self-aware marker at line 2315: `// fork default (no RefImageParams/AREA-resize): min 2*factor, max 128*factor.`

**Silent-drop / latent bug found (lingbot_video):** the fork's lingbot branch compares an *area* (`h_bar*w_bar`) against *linear-scaled* bounds (`2*factor` / `128*factor`). Upstream's lingbot uses proper AREA-mode defaults (`min = 4*factor*factor`, `max = 16384*factor*factor`) when `vlm_min/max_size <= 0`. For `factor≈28` the fork caps ref-image area at ~3584 px² (~60×60) — almost certainly a mis-scale that starves the lingbot i2v vision encoder. This is exactly the kind of dropped-capability regression the owner wants surfaced; upstream's parameterised path fixes it as a side effect.

**Also missed by trackB — `encode_image` call arity:** upstream's conditioner calls `llm->encode_image(n_threads, resized, false, true, true)` (the 3 compute-buffer free flags). The fork's `encode_image(n_threads, image)` uses its own graph-cut buffer management (trackB group B = KEEP-FORK). So when porting the resize helper, **keep the fork's 2-arg call** — do not pull the extra bools.

### 1b. `src/model/te/clip.hpp`, `src/model/te/llm.hpp` — cross-ref trackB
`clip.hpp`: trackB item J — diff is purely weight_manager ctor + compute free-flags. **Nothing to adopt.** Confirmed.

`llm.hpp`: trackB covered the weight_manager/compute-flag churn and flagged (item E) the **Qwen3-VL vision `split_patch_embed` + combined-bias rework** as HOLD. Re-confirmed and characterized precisely:
- Upstream renames `VisionPatchEmbed::llama_cpp_style` → `split_patch_embed`, sets `bias=false` on the two split Conv2d (`proj.0`,`proj.1`), and instead adds a **single** `params["bias"]` tensor applied *after* summing the two conv outputs. The fork keeps per-conv bias (`bias = (arch==QWEN3_VL)` on each Conv2d).
- Also adds vision-config auto-detection (patch_size/in_channels/hidden_size/num_position_embeddings/intermediate_size/out_hidden_size/num_layers from `visual.*`) and the `(QWEN3||QWEN3_VL) && num_layers==28 ⇒ num_heads=16` guard.
- **Safety / checkpoints affected:** correctness-equivalent *only if* the fork's Qwen3-VL edit checkpoints store a single combined `visual.patch_embed.bias` (upstream's assumption) rather than per-conv `proj.{0,1}.bias` (fork's assumption). Cannot be decided statically. This is a **LIVE path** in the fork (`hidream_o1.hpp`, and every `conditioner.hpp` ref-image branch that calls `llm->encode_image`). **Verdict: HOLD.** Verify with a real Qwen3-VL checkpoint (dump `visual.patch_embed.*` tensor names); adopt the split_patch_embed+combined-bias rework *as a unit* only if a checkpoint shows a single combined bias. Do not blind-adopt — a wrong bias layout silently corrupts every i2v/edit ref-image encode.

### 1c. `src/extensions/generation_extension.h`, `photomaker_extension.cpp`
- **PhotoMaker: KEPT and working** in the fork (`create_photomaker_extension()` present; `pmid.hpp` adapter present). The upstream `photomaker_extension.cpp` diff (-76 lines) is the ModelManager/weight_manager refactor = KEEP-FORK.
- **PuLID: DROPPED.** The upstream `generation_extension.h` diff adds three things the fork's version lacks: (1) `std::shared_ptr<GenerationExtension> create_pulid_extension();`, (2) a `sd_pulid_params_t& pulid_params` member on `GenerationExtensionConditionContext`, (3) a new virtual hook `virtual void before_diffusion(DiffusionParams& params, int step) const {}` — the injection point that lets an extension write an identity embedding into `params.extra` before `diffusion_model->compute()`. The fork's `generation_extension.h` has none of these; its `FluxDiffusionExtra` is only `{guidance, skip_layers}` (upstream's is `{guidance, skip_layers, pulid_id, pulid_id_weight}`). See PuLID assessment below.

### 1d. examples/ (CLI, common, server) — fork owns these; isolated upstream wins only
Overwhelmingly KEEP-FORK (fork's graph-cut runner CLI, string `max_vram`/`split_mode`/`auto_fit`/`rpc_servers` plumbing, LTX/Wan/longcat routes, old `sd_image_t* generate_image(...)` core API). `media_io.{cpp,h}`, `routes.h`, `runtime.h`, `async_jobs.h`, `resource_owners.hpp` are **unchanged upstream** in this window (nothing to port; fork divergence is its own). Genuinely isolated upstream additions worth porting:

| Item | File | Effort | Note |
|---|---|---|---|
| `--prompt-file` / `--negative-prompt-file` | `examples/common/common.cpp` (`on_prompt_file_arg`, `on_negative_prompt_file_arg`) | S | pure CLI, zero core dep; long-prompt / no-shell-escape. Highest value. |
| A1111 `info` JSON population | `examples/server/routes_sdapi.cpp` (`prepare_info_field`) | M | fork returns `out["info"]=""`; only needs `sd_basename`/`sd_sample_method_name`/`sd_scheduler_name` (all present). Do NOT pull the `generate_image` bool-signature churn in the same hunk. |
| `.ckpt` embedding/LoRA extension | `examples/common/common.cpp` (`build_embedding_map`, `extract_and_remove_lora` `valid_ext`) | S | trivial, no core dep. |

Marginal/skip: batch metadata-seed `seed + i/images_per_batch` (no-op unless Qwen-layered ported, entangled with bool-signature); `"normal"` scheduler alias (fork parser rejects it → misleading). NOT portable (need absent core support): ADetailer mode, `--list-devices`/`sd_list_devices`, DPM++2M-SDE sampler aliases, upstream imatrix flags, the `generate_image` bool+out-param signature.

---

## PART 2 — the 12 upstream files ABSENT from our tree

All 12 confirmed ABSENT at fork HEAD. Classification:

| # | Upstream file(s) | Verdict | Fork equivalent / port |
|---|---|---|---|
| 1 | `src/model_manager.{cpp,h}` (1329 L) | **Replaced (architectural)** | Fork's `backend`/`params_backend` GGMLRunner convention + graph-cut offload. Upstream `ModelManager : RunnerWeightManager` (Disk/ParamBackend residency, LoraSpec, staged compute-backend). This IS the keep-fork core divergence — do NOT adopt. |
| 2 | `src/weight_manager.h` (19 L) | **Replaced (architectural)** | The pure-virtual `RunnerWeightManager` interface (`assign_compute_backend`/`prepare_params`/`release_*`) that `ModelManager` implements. Fork runners take `(backend, params_backend)` directly. Do NOT adopt. |
| 3 | `src/core/layer_split_partition.{cpp,h}` | **Replaced (architectural)** | Fork has its own `src/core/ggml_graph_cut.{cpp,h}` (the keep-fork graph-cut engine). Upstream's `partition_graph_cut_layer_split(...)` builds on `ModelManager` residency + `sd::ggml_graph_cut::Plan`. Same problem solved on the other runner convention. Do NOT adopt. |
| 4 | `src/core/backend_fit.{cpp,h}` | **Replaced (architectural), 1 marginal isolate** | `derive_backend_specs(ModelLoader, budgets, runtime_spec, params_spec)` auto-derives DiT/VAE/CONDITIONER placement — fork does this via its own `prepare_backend_assignments()` + `max_vram`/`auto_fit` config (examples/common) and env (`MAXV`/`TBF`/`VWT`). Tied to ModelManager. One possibly-isolate: `prepare_vae_decode_retry_tiling(sd_tiling_params_t&, bool)` (VAE-decode-OOM retry with tiling) — but it keys off `sd_tiling_params_t`; low value vs fork's existing VAE eviction/tiling. Effectively replaced. |
| 5 | `src/runtime/imatrix.{cpp,h}` | **Replaced (fork has tailored equivalent)** | Upstream `IMatrixCollector` (llama.cpp/Kawrakow: `collect_imatrix`/`save_imatrix`/`load_imatrix`). Fork has its OWN imatrix path in `src/convert.cpp` (`sd_imatrix_collect_begin`, `sd_imatrix_collect_write_gguf`, `convert_with_imatrix`) + CLI `--collect-imatrix`/`--imatrix`, purpose-built for nvfp4-twolevel AWQ importance. Different API, don't port. |
| 6 | `src/detailer.{cpp,h}` (+ new `src/model/detector/yolov8.h`) | **DROPPED FEATURE (ADetailer)** | See assessment below. Fork carries the building blocks (`yolov8.h`, `docs/adetailer.md`, `scripts/convert_yolov8_to_safetensors.py`) but NOT the orchestration; `yolov8.h` is **not #included anywhere** in the fork = orphaned. |
| 7 | `src/extensions/pulid_extension.cpp` | **DROPPED FEATURE (PuLID)** | See assessment below. Fork carries `src/model/adapter/pulid.hpp` + `scripts/pulid_extract_id.py` + `docs/pulid.md`, but `pulid` is referenced **nowhere else** in fork `src/` = orphaned adapter. |

Net: **7 of the omitted files are architectural/replaced** (model_manager, weight_manager, layer_split_partition ×2, backend_fit ×2, imatrix ×2 → items 1–5 spanning 9 files) and **2 are genuinely-dropped features** (detailer + yolov8, pulid_extension → items 6–7).

---

## PART 3 — docs/ (27 files)
**Take upstream `docs/` wholesale — safe.** The fork made **zero** independent edits to any doc (`git diff 19bdfe2 HEAD -- docs/` is 12 files, all pure additions byte-identical to upstream tip: adetailer, animatediff, boogu_image, edit, hunyuan_video, imatrix, krea2, lingbot_video, minit2i, pulid, rpc, sefi_image). The other 21 upstream-touched docs (backend.md +136, performance.md +54, wan/flux2/ltx2/qwen_image_edit reworks, plus ~15 trivial link/wording tweaks) would simply gain upstream's improvements with nothing to clobber. No cherry-pick / keep-fork carve-outs needed.

---

## RefImageParams i2v ref-image resize — PORT PLAN

**Owner goal:** give the fork the parameterised AREA / LONGEST_SIDE ref-image resize (and fix the lingbot mis-scale). Two scopes — recommend the **narrow** one; the full one is a separate large refactor.

### Scope A — NARROW: resize helper only  (recommended, ~S–M)
Delivers AREA + LONGEST_SIDE + per-model tuning of the VLM ref-image encode, fixes the lingbot bug, touches **only `conditioner.hpp` (+ a small enum)**. Does NOT pull RoPE ref-index / pass_to_vlm / pass_to_dit / vae-resize.

1. **Add the enum + a minimal params struct.** Put in `src/conditioning/conditioner.hpp` (or `model.hpp` if preferred), *without* upstream's `Rope::RefIndexMode` dependency:
   ```cpp
   enum class RefImageResizeMode { NONE, LONGEST_SIDE, AREA };
   struct RefImageResizeParams { RefImageResizeMode mode = RefImageResizeMode::AREA; int min_size = -1; int max_size = -1; };
   ```
2. **Add one field to the fork's `ConditionerParams`** (line ~101): `RefImageResizeParams ref_image_resize;` (default AREA, -1/-1 → preserves current behavior when callers don't set it).
3. **Copy upstream `resize_image_dims(...)`** verbatim (uses only `std::sqrt/round/floor/ceil/max` — no fork-missing symbols) as a member of the conditioner struct(s) that need it. It is self-contained.
4. **Replace the three inline blocks** (`qwen_image` ~1982, `longcat` ~2050, `lingbot_video` ~2282) with the `min_pixels/max_pixels` default-selection (mode-aware: AREA → `4*factor²`/`16384*factor²`, else `2*factor`/`128*factor`) + a `resize_image_dims(...)` call, mirroring upstream's lingbot block. **Keep the fork's `llm->encode_image(n_threads, resized)` 2-arg call** (do not add upstream's free-flag bools). **Preserve the fork's tuned qwen/longcat constants** (`384²`/`560²`) by passing them as `min_size/max_size` rather than taking upstream's `4*factor²`/`16384*factor²` defaults — i.e. behavior-preserving for qwen/longcat, bug-fixing for lingbot.
5. **Caller:** in the fork's `stable-diffusion.cpp` where `ConditionerParams` is populated, set `ref_image_resize` from a per-model preset (a tiny local map keyed on `SDVersion`, replacing the need for upstream's `REF_IMAGE_PRESETS`). Only the resize fields are needed; ignore pass_to_vlm/ref_index_mode/etc.

**Blast radius (Scope A):** `conditioner.hpp` (enum + struct field + helper + 3 edited branches), `stable-diffusion.cpp` (1 caller populates `ref_image_resize`). No DiT, no RoPE, no model.hpp preset table, no `Rope::RefIndexMode`. Risk: low; behavior-preserving for qwen/longcat, fixes lingbot.

### Scope B — FULL RefImageParams framework  (NOT recommended now, ~L)
Upstream's full `RefImageParams` also carries `pass_to_vlm`, `pass_to_dit`, `Rope::RefIndexMode ref_index_mode`, `force_ref_timestep_zero`, `resize_before_vae`, `vae_input_max_pixels`, plus the `REF_IMAGE_PRESETS` table, and threads them through `stable-diffusion.cpp` **and every ref-image DiT** (flux, qwen_image, z_image, krea2, boogu, anima). Critically it depends on **`Rope::RefIndexMode`** (upstream `rope.hpp` lines 15/355/406/431/572/602) — an enum the **fork's `rope.hpp` does not have**, controlling how ref-image RoPE positions are indexed (FIXED/INCREASE/DECREASE). Adopting the full framework means porting the RoPE ref-index reindexing into the fork's rope + every DiT forward — a large, engine-touching change with no isolated win over Scope A for the resize goal. Defer unless a specific model needs per-ref RoPE reindexing.

---

## ADetailer (detailer.cpp) & PuLID (pulid_extension.cpp) — port assessment

Both are **genuinely-dropped features**, not replaced-by-equivalent. In both, the fork already carries the model/NN building blocks + scripts + docs, but the **runtime orchestration glue was never ported**, leaving the NN definitions orphaned (unreferenced).

### ADetailer — `src/detailer.{cpp,h}` + `src/model/detector/yolov8.h`
- **What it does:** YOLOv8 detect (`decode_detections` + NMS `box_iou` + `filter_and_sort_detections` by area/L2R/center) → build a mask per detection (dilate/erode/blur/pad) → run an **inpaint img2img pass** over each region via the existing `sd_ctx`. Public C API: `new_adetailer_ctx(detector_path,…)`, `adetail_image(ctx, sd_ctx, params, …)`, `free_adetailer_ctx`. Params (`ADetailerParams`) cover confidence/nms/dilate/mask ratios/inpaint size/denoise strength/sampler/scheduler.
- **ModelManager dependency:** yes but shallow — `adetailer_ctx_t` news its **own** `ModelManager` (detailer.cpp:676) purely to host the YOLOv8 detector weights (Disk vs ParamBackend residency). It does **not** entangle with the main DiT runner; `adetail_image` reuses the caller's `sd_ctx` for the inpaint pass.
- **Port onto fork runner convention (~M–L):** the fork already has `yolov8.h` (the detector network) — wire it as a normal fork `GGMLRunner(backend, params_backend, …)` instead of `ModelManager`, drop the ModelManager residency lines, keep the pure-CPU detect/NMS/mask/sort logic verbatim (no core deps), and route the inpaint pass through the fork's existing img2img. Also needs CLI/server `ad_*` params (see examples audit — currently absent from fork core) and `resource_owners.hpp` `ADetailerCtxPtr`. Blast radius: 1 new `.cpp`/`.h` on fork convention + yolov8 runner instantiation + CLI/server param surface. Self-contained relative to the video engine.

### PuLID — `src/extensions/pulid_extension.cpp`
- **What it does:** `load_pulid_id_embedding(path)` reads a `pulid_id` tensor from a gguf (produced by the fork's own `scripts/pulid_extract_id.py`), and the extension injects it into the Flux DiT via `FluxDiffusionExtra.pulid_id` (+ `pulid_id_weight`) using the new `before_diffusion(params, step)` hook. Registered via `create_pulid_extension()`.
- **What the fork is missing (bigger than ADetailer because it touches the DiT forward):**
  1. `FluxDiffusionExtra` needs `pulid_id` + `pulid_id_weight` fields (fork has only `{guidance, skip_layers}`).
  2. `flux.hpp` must **consume** `pulid_id` — inject the ID tokens into the Flux blocks. Fork `flux.hpp` has **zero** pulid references, so the actual cross-attention/insertion path (the hard part, and where upstream's 17 `stable-diffusion.cpp` pulid refs live) is absent. The fork's `pulid.hpp` provides only the ID *encoder* adapter, not the block injection.
  3. `generation_extension.h` needs the `before_diffusion` hook + `create_pulid_extension` + `sd_pulid_params_t` on the condition context.
  4. `sd_pulid_params_t` + CLI/server param surface.
- **ModelManager dependency:** the extension glue itself is light (gguf read + tensor handoff); the heavier upstream coupling is that `GenerationExtension::init` now carries `std::shared_ptr<ModelManager>` and `collect_loras(std::vector<ModelManager::LoraSpec>&)`. On the fork these map to the fork's extension init (backend/params_backend) + existing LoRA apply path.
- **Port effort (~L):** larger than ADetailer because it requires modifying the Flux DiT forward (block-level ID injection), not just adding a standalone context. Recommend porting **ADetailer first** (isolated, no DiT change) and treating PuLID as a separate DiT-touching task gated on need.

### Bottom line for owner
- **Dropped features (no fork equivalent):** ADetailer (`detailer.*` + orphaned `yolov8.h`) and PuLID (`pulid_extension.cpp` + orphaned `pulid.hpp`). Both have building blocks in-tree; only the orchestration was dropped.
- **Everything else omitted** (model_manager, weight_manager, layer_split_partition, backend_fit, imatrix — 9 files) is **architectural, deliberately replaced** by the fork's graph-cut `backend`/`params_backend` runner + fork's own imatrix — correctly not carried, per keep-fork mandate.
- **Immediate ADOPT candidates (low risk):** conditioner Scope-A resize port (fixes lingbot bug), `--prompt-file`/`--negative-prompt-file`, A1111 `info` JSON, `.ckpt` ext, docs/ wholesale.
- **HOLD (verify first):** llm.hpp Qwen3-VL split_patch_embed+combined-bias rework — live path, checkpoint-dependent bias layout.
