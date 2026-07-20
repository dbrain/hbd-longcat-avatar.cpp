# ADETAILER — apply-ready integration edits

Fork worktree: `/home/dbrain/dev/longcat-avatar-mergewt` (branch `upstream-merge-2026-07-19`)
Upstream tip `ea4e566`, merge base `19bdfe2`, fork HEAD.
Every fork anchor below was `grep`/`Read`-verified against the CURRENT worktree. **No source was edited; nothing built.**

Cross-refs: `PULL-IN-ROADMAP.md` Batch 4, `audit-conditioner-dropped.md` PART 2 items 6 + ADetailer assessment.

---

## 0. TL;DR verdict (read first)

ADetailer = a post-process detail pass: **YOLOv8 detect → NMS → per-detection mask (offset/dilate/erode/blur)
→ filter/sort → crop+resize each region → img2img inpaint through the existing `sd_ctx` → composite back**.
Two runtime pieces: a standalone **detector context** (its own backend + YOLOv8 GGMLRunner) and the **orchestration**
(`adetail_image`) that reuses the caller's `sd_ctx`. Almost all of it is pure-CPU and ports verbatim; the only
fork-divergence work is (a) de-staling the orphaned `yolov8.h` runner, (b) rewriting the detector *load* path off
`ModelManager` onto the fork's `ModelLoader::load_tensors` + `alloc_params_buffer` convention (identical to the
fork's own `ESRGAN`/`UpscalerGGML`), and (c) adapting the inpaint call to the fork's `sd_image_t* generate_image(...)`
return-a-pointer API (upstream uses a bool out-param form).

| # | Capability | Portability to fork | Where |
|---|---|---|---|
| **A** | Pure-CPU DSP: letterbox, DFL decode, NMS, mask morphology/blur, crop/composite, prompt `[SEP]`/`[PROMPT]`/`[SKIP]`, `extra_ad_args` parse | **Verbatim.** No fork-missing symbols (`sd::Tensor`/`sd::full`/`std::*`/`nlohmann::json`). Copy `detailer.cpp`'s static helpers 1:1. | §1, §4 |
| **B** | `yolov8.h` detector network (blocks + `YOLOv8Config` detect + `build_graph`) | **Byte-identical to upstream but STALE** — its `YOLOv8Runner` ctor takes `RunnerWeightManager` and its `compute()` passes 3 trailing bools; the fork has neither. Two small ctor/compute edits, network body untouched. | §2 |
| **C** | Detector context load (`ADetailerGGML::load_from_file`) | **Rewrite off `ModelManager`** onto fork's standalone-runner pattern (`SDBackendManager` 7-arg `init` + `ModelLoader::load_tensors` + `alloc_params_buffer`). The fork's `ESRGAN::load_from_file`/`UpscalerGGML` is the exact template. | §3, §4 |
| **D** | Inpaint pass (`adetail_image`) | **Adapt one call**: fork `generate_image` returns `sd_image_t*` (not upstream's `bool ... , out**, count*`); fork has no `pulid_params` field and no `free_sd_images`. Everything else (crop/mask/composite loop) is verbatim. | §5 |
| **E** | Public C API (`sd_adetailer_params_t`, `new_adetailer_ctx`/`adetail_image`/`free_adetailer_ctx`) | **Clean additive** to `include/stable-diffusion.h` — copy upstream's decls unchanged (they already use the fork-compatible `sd_img_gen_params_t*`). | §6 |
| **F** | Detector backend module slot | **Reuse `SDBackendModule::UPSCALER`** (zero enum churn) — recommended; or add a `DETECTOR` enum (3 sites). | §3, §7 |
| **G** | CLI / server param surface (`ad_model`, `ad_prompt`, `ad_negative_prompt`, `extra_ad_args`, `ADETAILER` mode) + `ADetailerCtxPtr` | **Fork owns `examples/`** and its CLI diverged; wire the standalone-ctx idiom + `resource_owners.hpp` deleter. koblem/server surface is out-of-scope for this C++ doc — flag to owner. | §8 |

**Bottom line:** ADetailer is the *low-risk* dropped feature (unlike PuLID it never touches the DiT forward). The
detector is fully isolated from the video engine and reuses the caller's `sd_ctx` for inpaint. Land §1–§7 as one
self-contained commit (mirrors the AnimateDiff/sefi adopts); §8 CLI/server is optional polish. The only real
gotchas are the three fork-divergence adaptations (B ctor/compute, C load, D `generate_image`) — all mechanical.
**GPU/checkpoint-VERIFY:** no converted YOLOv8 detector weights confirmed on-box (see §10).

---

## 1. NEW FILES — `src/detailer.cpp` + `src/detailer.h`

Both files are **ABSENT** from the fork (`ls src/detailer.*` → not found; confirmed in audit PART 2 item 6). Create
them from `git show ea4e566:src/detailer.{cpp,h}`, with the edits in §2–§5 applied. **Build pickup is automatic** —
`CMakeLists.txt:272` globs `file(GLOB SD_LIB_SOURCES CONFIGURE_DEPENDS "src/*.cpp" …)`, so `src/detailer.cpp` is
compiled without any build-file edit (verified). `#include "json.hpp"` resolves to `thirdparty/json.hpp` (present;
already included the same way by `src/model/te/llm.hpp`, `src/tokenizers/*`).

### 1a. `detailer.h` — drop the `ModelManager` include + member, keep the rest

Upstream `detailer.h` includes `"model_manager.h"` and declares `std::shared_ptr<ModelManager> model_manager;`
inside `ADetailerGGML`. The fork has **no `model_manager.h`** (architectural non-port). Port the header as:

- **Remove** `#include "model_manager.h"`.
- **Keep** `#include "core/ggml_extend_backend.h"`, `#include "model/detector/yolov8.h"`, `#include "stable-diffusion.h"`.
- In `struct ADetailerGGML`: **remove** the `std::shared_ptr<ModelManager> model_manager;` member; **keep**
  `SDBackendManager backend_manager;`, `std::shared_ptr<YOLOv8Runner> detector;`, `std::vector<std::string> class_names;`,
  `int n_threads`, the two backend-spec strings, and the `ADetailerGGML(...)` ctor / `~ADetailerGGML()` /
  `load_from_file` / `predict` declarations.
- Keep `ADetailerDetection`, `ADetailerSort`, `ADetailerParams` structs **verbatim** (pure PODs, no fork deps).

The `SDBackendManager backend_manager;` member is valid — the fork's `SDBackendManager`
(`src/core/ggml_extend_backend.h:39`) is the same class the fork's own `UpscalerGGML` embeds.

### 1b. `detailer.cpp` — copy the DSP helpers verbatim

Copy verbatim (no edits): `trim_copy`, `lower_copy`, `parse_bool/int/float`, `image_channel`, `bilinear_channel`,
`make_letterbox_input`, `logistic`, `dfl_expectation`, `box_iou`, `decode_detections`, `detection_area`,
`filter_and_sort_detections`, `bbox_mask`, `offset_mask`, `morphology_mask`, `gaussian_blur_mask`, `make_masks`,
`mask_bbox`, `expand_crop`, `resize_crop_image`, `resize_crop_mask`, `copy_as_rgb`, `composite_crop`,
`split_prompts`, `resolve_prompt`, `prompt_is_skip`, `validate_params`, `parse_class_names`, `parse_adetailer_params`.
Verified fork primitives these rely on: `sd::full<float>` (`src/core/tensor.hpp:815`), `sd::Tensor<float>`
(`tensor.hpp`), `.index()/.data()/.shape()/.dim()/.empty()`, `ggml_time_ms`, `str_to_sample_method`/`str_to_scheduler`
(`include/stable-diffusion.h:778,780`), `SAMPLE_METHOD_COUNT`/`SCHEDULER_COUNT`, `nlohmann::json`. All present.

The functions that change are `ADetailerGGML::load_from_file` (§4a), `ADetailerGGML::predict` (§4b, verbatim but
note the compute call), and `adetail_image` (§5). The C-API glue (`new_adetailer_ctx`/`free_adetailer_ctx`) copies
verbatim except it constructs the fork-adapted `ADetailerGGML` (unchanged signature).

---

## 2. `src/model/detector/yolov8.h` — de-stale the orphaned runner (network body untouched)

**The fork's `yolov8.h` is byte-identical to upstream** (`diff <(git show ea4e566:src/model/detector/yolov8.h)
<(cat src/model/detector/yolov8.h)` → IDENTICAL) and is `#include`d **nowhere** = orphaned. It references two symbols
the fork doesn't have, so it must be de-staled before it will compile. **Only the `YOLOv8Runner` struct changes; the
`Conv`/`C2f`/`SPPF`/`Detect`/`YOLOv8Model` blocks + `YOLOv8Config::detect_from_weights` + `build_graph` are correct
as-is** (they use `GGMLRunnerContext`, `make_input`, `get_context`, `new_graph_custom` — all present in the fork:
`ggml_extend.hpp:5753`, `:5338`, `:2601`).

### 2a. Ctor — `RunnerWeightManager` → `(backend, params_backend)`

Anchor (fork `src/model/detector/yolov8.h:323-332`, verbatim current):

```cpp
    YOLOv8Runner(ggml_backend_t backend,
                 const String2TensorStorage& tensor_storage_map,
                 std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
        : GGMLRunner(backend, weight_manager),
          config(YOLOv8Config::detect_from_weights(tensor_storage_map)) {
        if (config.valid) {
            model = std::make_unique<YOLOv8Model>(config);
            model->init(params_ctx, tensor_storage_map, "");
        }
    }
```

Replace with (fork's two-backend ctor — matches `GGMLRunner(ggml_backend_t backend, ggml_backend_t params_backend)`
at `ggml_extend.hpp:5319`, and the fork's own `ESRGAN(backend, params_backend, …)` at `esrgan.hpp:163`):

```cpp
    YOLOv8Runner(ggml_backend_t backend,
                 ggml_backend_t params_backend,
                 const String2TensorStorage& tensor_storage_map)
        : GGMLRunner(backend, params_backend),
          config(YOLOv8Config::detect_from_weights(tensor_storage_map)) {
        if (config.valid) {
            model = std::make_unique<YOLOv8Model>(config);
            model->init(params_ctx, tensor_storage_map, "");
        }
    }
```

> `String2TensorStorage`, `params_ctx`, `config` all in scope exactly as before. `RunnerWeightManager` is a
> **deliberate non-port** (audit PART 2 item 2 — the fork replaces it with the `(backend, params_backend)` runner
> convention). `get_param_tensors(std::map<std::string, ggml_tensor*>&)` at line 338 stays as-is; the detailer
> load path (§4a) calls it exactly like `ESRGAN::load_from_file` does.

### 2b. `compute()` — 3 trailing bools → the fork's 2-bool signature

Anchor (fork `src/model/detector/yolov8.h:356-359`, verbatim current):

```cpp
    sd::Tensor<float> compute(int n_threads, const sd::Tensor<float>& input) {
        auto get_graph = [&]() { return build_graph(input); };
        return take_or_empty(GGMLRunner::compute<float>(get_graph, n_threads, false, false, false));
    }
```

Replace with (the fork's `compute` is `compute(get_graph, n_threads, bool free_compute_buffer_immediately,
bool no_return=false)` — `ggml_extend.hpp:5807`, only **2** bool params):

```cpp
    sd::Tensor<float> compute(int n_threads, const sd::Tensor<float>& input) {
        auto get_graph = [&]() { return build_graph(input); };
        return take_or_empty(GGMLRunner::compute<float>(get_graph, n_threads, false));
    }
```

> `take_or_empty` exists (`ggml_extend.hpp:2477`). The caller (`ADetailerGGML::predict`) then calls
> `detector->free_compute_buffer()` explicitly (verbatim from upstream) — so `free_compute_buffer_immediately=false`
> here is correct and matches upstream intent.

---

## 3. `ADetailerGGML` ctor + members — fork backend convention

The fork's `SDBackendManager` differs from upstream's in two load-bearing ways (verified
`src/core/ggml_extend_backend.h:52`):
- `init(backend_spec, params_backend_spec, offload_params_to_cpu, keep_clip_on_cpu, keep_vae_on_cpu,
  keep_control_net_on_cpu, error*)` — **7 args** (upstream `detailer.h` calls a 4-arg `init(spec, spec, nullptr, &error)`).
- No `params_backend_is_disk(...)` (that is `ModelManager` Disk residency = dropped).
- `SDBackendModule` enum has **no `DETECTOR`** — members: `DIFFUSION, TE, CLIP_VISION, VAE, CONTROL_NET,
  PHOTOMAKER, UPSCALER` (`ggml_extend_backend.h:13`).

**Recommended (zero enum churn): reuse `SDBackendModule::UPSCALER`** as the detector's module slot. The detector runs
in its own `SDBackendManager` instance, so the module identity only picks which backend-spec override applies;
`UPSCALER` behaves as a plain single-model slot. (Faithful alternative: add a `DETECTOR` enum — see §7.)

The `ADetailerGGML(int, std::string, std::string)` ctor body is **verbatim** from upstream (just member-inits the
n_threads/spec strings). The `~ADetailerGGML()` drops the `model_manager.reset();` line, keeping only
`detector.reset();`.

---

## 4. `ADetailerGGML::load_from_file` + `predict`

### 4a. `load_from_file` — REWRITE off `ModelManager` onto `ModelLoader::load_tensors`

Upstream's `load_from_file` (`detailer.cpp:~640`) uses `backend_manager.init(spec, spec, nullptr, &error)`, then
`model_manager = std::make_shared<ModelManager>(); … loader.init_from_file(); detector =
std::make_shared<YOLOv8Runner>(backend, tsm, model_manager); … model_manager->register_param_tensors("YOLOv8", …,
ResidencyMode::Disk|ParamBackend, …); model_manager->validate_registered_tensors();`. **None of the ModelManager
machinery exists in the fork.** Replace the whole method body with the fork's standalone-runner load sequence,
copied structurally from `ESRGAN::load_from_file` (`src/model/upscaler/esrgan.hpp:175`) +
`UpscalerGGML::load_from_file` (`src/upscaler.cpp:34`):

```cpp
bool ADetailerGGML::load_from_file(const std::string& detector_path) {
    ggml_log_set(ggml_log_callback_default, nullptr);

    std::string error;
    // Fork's 7-arg SDBackendManager::init (no ModelManager, no Disk residency).
    if (!backend_manager.init(backend_spec.c_str(),
                              params_backend_spec.c_str(),
                              /*offload_params_to_cpu=*/false,
                              /*keep_clip_on_cpu=*/false,
                              /*keep_vae_on_cpu=*/false,
                              /*keep_control_net_on_cpu=*/false,
                              &error)) {
        LOG_ERROR("ADetailer backend config failed: %s", error.c_str());
        return false;
    }
    const SDBackendModule module = SDBackendModule::UPSCALER;  // detector slot (see §3)
    ggml_backend_t backend        = backend_manager.runtime_backend(module);
    ggml_backend_t params_backend = backend_manager.params_backend(module);
    if (backend == nullptr || params_backend == nullptr) {
        LOG_ERROR("failed to initialize detector backend");
        return false;
    }

    ModelLoader loader;
    if (!loader.init_from_file(detector_path)) {
        LOG_ERROR("failed to load ADetailer detector: '%s'", detector_path.c_str());
        return false;
    }

    detector = std::make_shared<YOLOv8Runner>(backend, params_backend, loader.get_tensor_storage_map());
    if (!detector || !detector->model || !detector->config.valid) {
        LOG_ERROR("unsupported YOLOv8 detection weights: '%s'", detector_path.c_str());
        return false;
    }

    // Fork params load: alloc on params_backend, then load actual tensor data via ModelLoader.
    if (!detector->alloc_params_buffer()) {
        LOG_ERROR("failed to allocate YOLOv8 detector params buffer");
        return false;
    }
    std::map<std::string, ggml_tensor*> tensors;
    detector->get_param_tensors(tensors);
    if (!loader.load_tensors(tensors, {}, n_threads)) {
        LOG_ERROR("failed to load YOLOv8 detector tensors");
        return false;
    }

    class_names.assign(static_cast<size_t>(detector->config.num_classes), "");
    auto names_item = loader.get_metadata().find("yolov8.names");
    if (names_item != loader.get_metadata().end()) {
        class_names = parse_class_names(names_item->second, detector->config.num_classes);
    }
    return true;
}
```

> **VERIFY the exact ModelLoader API names before writing** — the fork's `ESRGAN::load_from_file` uses
> `ModelLoader model_loader; model_loader.init_from_file_and_convert_name(path); … model_loader.load_tensors(tensors,
> {}, n_threads);` (`esrgan.hpp:178,339`). Two open items to confirm against `src/model_loader.h`:
> 1. **Name conversion:** ESRGAN uses `init_from_file_and_convert_name` (applies fork name remap); upstream detailer
>    used plain `init_from_file`. Prefer `init_from_file` for YOLOv8 (its tensor names `model.N.conv.weight` are the
>    convert-script output and must not be remapped). Confirm `ModelLoader::init_from_file` exists (upstream had it;
>    fork keeps the loader). If only `..._and_convert_name` exists, verify it is a no-op for `model.*`/`yolov8.*` keys.
> 2. **`get_metadata()` / `get_tensor_storage_map()`** — `get_tensor_storage_map()` is used across the fork
>    (`upscaler.cpp:81` region); confirm `get_metadata()` returns the `yolov8.names` map entry (upstream had it; if
>    the fork's loader lacks `get_metadata`, class-name labels degrade gracefully to `class_N` — non-fatal, drop that block).
> `alloc_params_buffer()` is a `GGMLRunner` protected method the fork exposes internally (`esrgan.hpp:273` calls it) —
> callable from `YOLOv8Runner` since `ADetailerGGML` holds the runner; if it is `protected`, add a thin public
> `bool load_params(ModelLoader&, int)` method on `YOLOv8Runner` mirroring `ESRGAN::load_from_file`'s tail, and call
> that instead (cleaner — keeps the alloc+load inside the runner exactly like ESRGAN does). **Recommended: give
> `YOLOv8Runner` a `load_from_file`-style method** so §4a just does `detector->load_params(loader, n_threads)`.

### 4b. `predict` — verbatim

Copy `ADetailerGGML::predict` (`detailer.cpp:~700`) unchanged. It calls
`sd::Tensor<float> raw = detector->compute(n_threads, input.tensor); detector->free_compute_buffer();` — both valid
after §2b. `LetterboxInput`, `decode_detections`, the LOG_INFO loop — all pure-CPU, verbatim.

---

## 5. `adetail_image` — adapt the inpaint pass to the fork's `generate_image`

The detect/mask/crop/composite loop is verbatim. **Three fork adaptations** inside the per-mask loop:

### 5a. `generation.pulid_params = {}` — DELETE (fork has no such field)

The fork's `sd_img_gen_params_t` (`include/stable-diffusion.h:371-395`) has **no `pulid_params`** member (PuLID is a
separate dropped feature, Batch 5). Remove the `generation.pulid_params = {};` line. Keep `generation.pm_params = {};`
(present), `generation.control_image = {};` (present), `generation.ref_images = nullptr; generation.ref_images_count
= 0;` (present), `generation.hires.enabled = false;` (`sd_hires_params_t::enabled` present, line 344). All other
`generation.*` assignments upstream makes map 1:1 to fork fields: `init_image`, `mask_image`, `width`, `height`,
`strength`, `seed`, `batch_count`, `sample_params.{sample_steps,custom_sigmas,custom_sigmas_count,guidance.txt_cfg,
sample_method,scheduler}` — all verified present (`sd_sample_params_t` at lines 266-277).

### 5b. `generate_image` — bool out-param form → fork's pointer-return form

Upstream (`detailer.cpp:~988`):

```cpp
        sd_image_t* generated = nullptr;
        int generated_count   = 0;
        bool success          = generate_image(sd_ctx, &generation, &generated, &generated_count);
        free(local_image.data);
        free(local_mask.data);
        if (!success || generated_count <= 0 || generated == nullptr || generated[0].data == nullptr) {
            free_sd_images(generated, generated_count);
            free(current.data);
            return false;
        }

        Mask feather = gaussian_blur_mask(masks[i], params.mask_blur);
        composite_crop(&current, generated[0], feather, crop);
        free_sd_images(generated, generated_count);
        ++applied;
```

Replace with (fork API: `sd_image_t* generate_image(sd_ctx_t*, const sd_img_gen_params_t*)` —
`include/stable-diffusion.h:808` — returns a malloc'd array of `batch_count` images, `nullptr` on failure; the fork
has **no `free_sd_images`** helper, so free manually, matching the fork's `SDImageVec::clear`/`adopt` idiom):

```cpp
        sd_image_t* generated = generate_image(sd_ctx, &generation);  // batch_count == 1
        free(local_image.data);
        free(local_mask.data);
        if (generated == nullptr || generated[0].data == nullptr) {
            if (generated != nullptr) {
                free(generated[0].data);
                free(generated);
            }
            free(current.data);
            return false;
        }

        Mask feather = gaussian_blur_mask(masks[i], params.mask_blur);
        composite_crop(&current, generated[0], feather, crop);
        free(generated[0].data);
        free(generated);
        ++applied;
```

> `generation.batch_count = 1;` is already set above (verbatim upstream), so exactly one image comes back — index
> `[0]` is safe. The fork frees an image array by `free(img.data)` per element then `free(array)` (see
> `resource_owners.hpp:227` `SDImageVec::clear`). Do **not** call `free_sd_images` — it does not exist in the fork.

### 5c. `new_adetailer_ctx` / `free_adetailer_ctx` — verbatim

Copy both C-API functions unchanged (`SAFE_STR` is present at `src/core/util.h:14`). They construct the
fork-adapted `ADetailerGGML` (same 3-arg ctor signature) and call the §4a `load_from_file`.

---

## 6. `include/stable-diffusion.h` — public C API (clean additive)

The fork **has none of these** (grep for `adetailer`/`sd_adetailer_params_t` in `include/` → nothing). Copy the
upstream additions **verbatim** — they are already fork-compatible (they reference `sd_ctx_t`, `sd_image_t`,
`sd_img_gen_params_t`, all present). Anchor: place near the other ctx-based public decls (e.g. after the
`upscaler_ctx_t` / `new_upscaler_ctx` block, or before `convert(...)` as upstream does).

Add (from `git diff 19bdfe2 ea4e566 -- include/stable-diffusion.h`):

```cpp
typedef struct adetailer_ctx_t adetailer_ctx_t;

typedef struct {
    const char* prompt;
    const char* negative_prompt;
    const char* extra_ad_args;
} sd_adetailer_params_t;

SD_API adetailer_ctx_t* new_adetailer_ctx(const char* detector_path,
                                          int n_threads,
                                          const char* backend,
                                          const char* params_backend);
SD_API void free_adetailer_ctx(adetailer_ctx_t* adetailer_ctx);
SD_API bool adetail_image(adetailer_ctx_t* adetailer_ctx,
                          sd_ctx_t* sd_ctx,
                          sd_image_t input_image,
                          const sd_adetailer_params_t* adetailer_params,
                          const sd_img_gen_params_t* inpaint_params,
                          sd_image_t** images_out,
                          int* num_images_out);
```

> Do **NOT** pull the adjacent `convert_with_components(...)` decl from the same upstream diff hunk — that is an
> unrelated converter addition (Batch 6/7 territory), not part of ADetailer.

---

## 7. (Optional) `SDBackendModule::DETECTOR` — faithful alternative to reusing UPSCALER

Only if you prefer a dedicated slot over §3's `UPSCALER` reuse. Three edits, all mechanical:

1. `src/core/ggml_extend_backend.h:13-21` — add `DETECTOR,` to the `enum class SDBackendModule`.
2. `src/core/ggml_extend_backend.cpp:638` `sd_backend_module_name(...)` switch — add
   `case SDBackendModule::DETECTOR: return "detector";` (the switch has a trailing `return "unknown";`, so omission
   compiles but mislabels logs — add the case).
3. `src/core/ggml_extend_backend.cpp:54-78` (string→module parser, if you want `--backend detector:…` CLI overrides)
   — add a `"detector"` branch mapping to `SDBackendModule::DETECTOR`. Optional; without it the detector still gets
   the default backend via `SDBackendAssignment::get()` fallback.

**Recommendation:** skip this; reuse `UPSCALER` (§3). The detector is a peer standalone context and never coexists
with the ESRGAN upscaler in the same `SDBackendManager` instance, so the slot name is cosmetic.

---

## 8. CLI / server param surface (fork owns `examples/`; koblem out-of-scope)

Upstream adds an `ADETAILER` run-mode and `ad_*` params to `examples/cli/main.cpp` + `examples/common/common.{cpp,h}`.
The fork's `examples/` diverged heavily (graph-cut CLI, LTX/Wan routes) so there is **no 1:1 hunk** — wire
deliberately, mirroring the fork's existing `UpscalerCtxPtr` idiom.

### 8a. `resource_owners.hpp` — add the ctx deleter (the fork DOES have this file)

`resource_owners.hpp` **exists** at `examples/common/resource_owners.hpp` (the roadmap's "may not exist" caveat is
resolved: it exists, in `examples/common/`). It already defines `UpscalerCtxDeleter` + `UpscalerCtxPtr`
(lines 35-48). Add the parallel ADetailer owner:

Anchor (fork `examples/common/resource_owners.hpp:35-48`):

```cpp
struct UpscalerCtxDeleter {
    void operator()(upscaler_ctx_t* ctx) const {
        if (ctx != nullptr) {
            free_upscaler_ctx(ctx);
        }
    }
};

template <typename T>
using FreeUniquePtr = std::unique_ptr<T, FreeDeleter>;

using FilePtr        = std::unique_ptr<FILE, FileCloser>;
using SDCtxPtr       = std::unique_ptr<sd_ctx_t, SDCtxDeleter>;
using UpscalerCtxPtr = std::unique_ptr<upscaler_ctx_t, UpscalerCtxDeleter>;
```

Insert an `ADetailerCtxDeleter` after `UpscalerCtxDeleter` and an `ADetailerCtxPtr` alias after `UpscalerCtxPtr`:

```cpp
struct ADetailerCtxDeleter {
    void operator()(adetailer_ctx_t* ctx) const {
        if (ctx != nullptr) {
            free_adetailer_ctx(ctx);
        }
    }
};
```
```cpp
using ADetailerCtxPtr = std::unique_ptr<adetailer_ctx_t, ADetailerCtxDeleter>;
```

### 8b. CLI params (fork `examples/common/common.{h,cpp}`)

Upstream adds to its `SDGenerationParams`/arg table: `ad_model_path` (`--ad-model`), `ad_prompt` (`--ad-prompt`),
`ad_negative_prompt` (`--ad-negative-prompt`), `extra_ad_args` (`--extra-ad-args`), plus an `ADETAILER` value in the
run-mode enum and its `"adetailer"` string. Add the equivalent std::string fields to the fork's generation-params
struct in `examples/common/common.h` and register `--ad-model` / `--ad-prompt` / `--ad-negative-prompt` /
`--extra-ad-args` in the fork's arg-parse table in `common.cpp`, plus the mode enum + `load_if_exists("ad_model",…)`
/ `load_if_exists("extra_ad_args",…)` JSON restore (mirror upstream lines). **Verify the fork's generation-params
struct + mode-enum names** before editing (the fork renamed/extended these vs upstream).

### 8c. CLI orchestration (fork `examples/cli/main.cpp`)

Port upstream's `apply_adetailer(...)` helper, adapted to the fork's image-vector type: upstream iterates a
`SDImageVec results` (the fork **has** `SDImageVec` in `resource_owners.hpp:135`), calls `new_adetailer_ctx(...,
sd_ctx_params.backend, sd_ctx_params.params_backend)`, then per result builds an `sd_img_gen_params_t`
(seed offset, and for `IMG_GEN` mode `width=512,height=512,strength=0.4`), calls `adetail_image(...)`, and swaps the
result in. Replace upstream's `free_sd_images(detailed_images, detailed_count)` with the manual
free-each-`.data`-then-`free(array)` idiom (fork has no `free_sd_images`; see §5b). Gate: only run when
`ad_model_path` non-empty, and only in `IMG_GEN`/`ADETAILER` modes. This CLI path is **optional** — the public C API
(§6) is the load-bearing deliverable; kobbler drives via koblem/server, not this CLI.

### 8d. Server / koblem surface — FLAG TO OWNER (out-of-scope for this C++ doc)

The kobbler/koblem server request struct + JSON field for `ad_model`/`extra_ad_args` is Rust/TS and lives outside
this engine repo. Not portable from the sd.cpp diff. Note for the owner: to expose ADetailer through the server,
add an `ad_model`/`ad_prompt`/`ad_negative_prompt`/`extra_ad_args` field to koblem's image-gen request and call the
new `new_adetailer_ctx`/`adetail_image`/`free_adetailer_ctx` C API after the base image is generated.

---

## 9. Backup / nav / docs

- No table/backup/nav changes (this is the C++ engine, not kobbler Rust/web).
- `docs/adetailer.md` is already in-tree (audit PART 3 — the fork carries it byte-identical to upstream; it was
  added as a pure doc with no orchestration). No doc edit needed. `scripts/convert_yolov8_to_safetensors.py` likewise
  present (reference for producing the detector `.safetensors` — confirm on-box before GPU-verify, §10).

---

## 10. Blockers, missing primitives, risks

**BLOCKER — NONE that stop the port.** ADetailer is the low-risk dropped feature: it never touches the DiT forward
(contrast PuLID/Batch 5), the detector is a fully isolated standalone context, and the inpaint pass reuses the
existing `sd_ctx`/`generate_image`. All fork divergences are mechanical adaptations (§2 ctor/compute, §4a load, §5b
`generate_image`), each with a verified fork template (`ESRGAN`/`UpscalerGGML`).

**VERIFY-1 (ModelLoader API names, §4a):** confirm against `src/model_loader.h`: `ModelLoader::init_from_file`
(vs the fork's `init_from_file_and_convert_name` used by ESRGAN), `load_tensors(map, {}, n_threads)`,
`get_tensor_storage_map()`, `get_metadata()`. `load_tensors`/`get_tensor_storage_map` are confirmed in-use by
`esrgan.hpp`/`upscaler.cpp`; `init_from_file` (no-convert) and `get_metadata` need a one-line grep to confirm the
fork kept them. If `get_metadata` is absent, the `yolov8.names` class-label lookup degrades to `class_N` (non-fatal).
**Recommendation:** encapsulate the alloc+load inside a new `YOLOv8Runner::load_from_file(ModelLoader&, int)` method
(mirror `ESRGAN::load_from_file`'s tail) so `alloc_params_buffer()` visibility is a non-issue and §4a stays a
one-liner.

**VERIFY-2 (`alloc_params_buffer` visibility):** it is called from within `ESRGAN::load_from_file` (i.e. from a
`GGMLRunner` subclass, `esrgan.hpp:273`). Calling it from `ADetailerGGML` (which only *holds* a `YOLOv8Runner`)
may hit `protected`. Fold the alloc+load into the runner (VERIFY-1 recommendation) to keep it in-subclass.

**VERIFY-3 (detector module slot, §3):** reusing `SDBackendModule::UPSCALER` is behaviourally fine (the detailer owns
its own `SDBackendManager`), but confirm the fork's `SDBackendManager::init` populates the `UPSCALER` slot from the
default backend when no `upscaler:` override is given (it does for the fork's own upscaler path). If a user passes a
`--backend upscaler:cpu` override it would also steer the detector — acceptable, or add `DETECTOR` (§7) to decouple.

**GPU/CHECKPOINT-VERIFY (the real gate):** no converted YOLOv8 detector `.safetensors` confirmed on-box. The whole
pass is **unvalidated end-to-end** until one exists (produce via `scripts/convert_yolov8_to_safetensors.py`).
Specifically unverifiable statically: `YOLOv8Config::detect_from_weights` matching the convert-script tensor names,
the DFL/stride decode (`decode_detections`) numerics, and that `init_from_file` (no name-convert) leaves
`model.N.*`/`yolov8.*` keys intact. The detect→mask→inpaint→composite logic is pure-CPU and deterministic, so a
single smoke render (face-detector on a generated portrait) fully validates it.

**RISK to existing paths: NONE, by construction.** `detailer.{cpp,h}` are new files; `yolov8.h` is currently
orphaned (`#include`d nowhere) so de-staling it (§2) cannot regress anything until `detailer.cpp` includes it. The
public-API additions (§6) are pure additions. No existing model, VAE, sampler, or the video engine is touched.
Reusing `SDBackendModule::UPSCALER` adds no new enum value and no switch churn.

**Fork-convention drift NOT adopted (deliberate):** upstream's `ModelManager`/`RunnerWeightManager`/`ResidencyMode`
(Disk vs ParamBackend) detector-weight residency, upstream's 4-arg `SDBackendManager::init`, upstream's 3-bool
`GGMLRunner::compute`, upstream's `free_sd_images` helper, and upstream's `generate_image` bool-out-param signature
— all replaced by the fork's `(backend, params_backend)` + `ModelLoader::load_tensors` + `alloc_params_buffer` +
manual image-free + `sd_image_t* generate_image(...)` conventions, exactly as AnimateDiff/sefi were adopted.

**Verification status:** all fork anchors Read/grep-confirmed at cited lines against the current worktree —
`yolov8.h` (byte-identical to upstream, ctor 323, get_param_tensors 338, compute 356), `GGMLRunner`
(`ggml_extend.hpp:5319` ctor, `:5807` compute 2-bool, `:5753` make_input, `:5338` get_context, `:2601`
new_graph_custom, `:2477` take_or_empty), `ESRGAN` (`esrgan.hpp:163` ctor, `:175` load_from_file, `:339`
load_tensors), `UpscalerGGML` (`upscaler.cpp:34` load, `:151` ctx, `:155` new_ctx), `SDBackendManager`
(`ggml_extend_backend.h:13` enum, `:52` init 7-arg, `.cpp:638` module_name), `sd_img_gen_params_t`
(`stable-diffusion.h:371-395`, NO pulid_params), `sd_sample_params_t` (`:266-277`), `generate_image` (`:808`,
pointer return), `resource_owners.hpp` (`:35` UpscalerCtxDeleter, `:135` SDImageVec), `CMakeLists.txt:272` GLOB,
`thirdparty/json.hpp`, `util.h:14` SAFE_STR. Cannot build/compile-check here (project rule) and no detector
checkpoint to run.
