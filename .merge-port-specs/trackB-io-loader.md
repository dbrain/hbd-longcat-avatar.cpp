# Track B — I/O + Loading + Tokenizers "who did it better" review

Scope: upstream (leejet) changes `19bdfe2..ea4e566` vs fork `HEAD` (785ba0d).
Read-only analysis. Files: `model_io/{safetensors_io,gguf_io,tensor_storage,streaming_writer}`,
`name_conversion.cpp`, `model_loader.{cpp,h}`, `tokenizers/{bpe_tokenizer,clip_tokenizer,tokenizer.h}`.

Bias: strong KEEP-FORK. Only isolated, safe, output-neutral (or malformed-only) changes are ADOPT.

---

## ADOPT

### A1 — Sharded safetensors `*.index.json` support  (HIGHEST VALUE)

New-format support the fork lacks. HF ships large DiTs (Qwen-Image, Hunyuan-Video, Flux2, etc. —
exactly the models this fork is porting) as `model.safetensors.index.json` + numbered shards.
The fork's `init_from_file` has **no `.json` branch** and no `read_safetensors_index_file`
(verified: `git grep "index.json\|weight_map\|read_safetensors_index" HEAD` returns nothing).
Fully self-contained: no ModelManager / multi-GPU / weight_manager coupling. It only parses a JSON
`weight_map`, dedups shard paths, and re-enters `init_from_file` per shard. Verified collision-free:
no caller in the fork passes a `.json` path to `init_from_file` (all callers pass model/vae/embedding
files — `convert.cpp`, `ltxv.hpp`, `ltx_latent_upscaler.hpp`, `ltx_audio_vae.hpp`). `ends_with` exists
in `src/core/util.h`. Fork is C++17 (structured bindings in use), so `std::filesystem` is available.

**Note:** adopt the index reader + wiring ONLY. Do NOT pull upstream's `metadata_` out-param /
`__metadata__` extraction (dead capability — no consumer in the fork), nor upstream's `add_file_path`
refactor. Keep the fork's `file_paths_.push_back(...)` convention untouched — the index path re-enters
`init_from_file`, which handles file registration itself.

**Edit 1 — `src/model_io/safetensors_io.h`**
Anchor (fork):
```cpp
bool read_safetensors_file(const std::string& file_path,
                           std::vector<TensorStorage>& tensor_storages,
                           std::string* error = nullptr);
```
Add immediately after:
```cpp
bool read_safetensors_index_file(const std::string& file_path,
                                 std::vector<std::string>& shard_paths,
                                 std::string* error = nullptr);
```

**Edit 2 — `src/model_io/safetensors_io.cpp`**
Add includes near the top include block (after `#include <fstream>`):
```cpp
#include <filesystem>
#include <unordered_set>
```
Add below `static constexpr size_t ST_HEADER_SIZE_LEN = 8;` (or near the top of the file):
```cpp
namespace fs = std::filesystem;
```
Anchor (fork, unique) — the line `static bool ggml_type_to_safetensors_dtype(ggml_type type, std::string* dtype) {`.
Insert the following two functions **immediately before** that anchor:
```cpp
static std::string resolve_index_shard_path(const std::string& index_path, const std::string& shard_path) {
    fs::path shard_fs_path(shard_path);
    if (shard_fs_path.is_absolute()) {
        return shard_fs_path.lexically_normal().string();
    }
    return (fs::path(index_path).parent_path() / shard_fs_path).lexically_normal().string();
}

bool read_safetensors_index_file(const std::string& file_path,
                                 std::vector<std::string>& shard_paths,
                                 std::string* error) {
    shard_paths.clear();

    std::ifstream file(file_path);
    if (!file.is_open()) {
        set_error(error, "failed to open '" + file_path + "'");
        return false;
    }

    nlohmann::json index;
    try {
        index = nlohmann::json::parse(file);
    } catch (const std::exception&) {
        set_error(error, "parsing safetensors index failed: '" + file_path + "'");
        return false;
    }

    if (!index.is_object() || !index.contains("weight_map") || !index["weight_map"].is_object()) {
        set_error(error, "invalid safetensors index '" + file_path + "'");
        return false;
    }

    std::unordered_set<std::string> seen_shard_paths;
    for (const auto& item : index["weight_map"].items()) {
        if (!item.value().is_string()) {
            set_error(error, "invalid shard path for tensor '" + item.key() + "'");
            return false;
        }

        std::string shard_path = resolve_index_shard_path(file_path,
                                                          item.value().get<std::string>());
        if (seen_shard_paths.insert(shard_path).second) {
            shard_paths.push_back(std::move(shard_path));
        }
    }

    if (shard_paths.empty()) {
        set_error(error, "safetensors index has no tensors: '" + file_path + "'");
        return false;
    }

    return true;
}
```

**Edit 3 — `src/model_loader.cpp`, `init_from_file`**
Anchor (fork):
```cpp
    } else if (is_gguf_file(file_path)) {
        LOG_INFO("load %s using gguf format", file_path.c_str());
        return init_from_gguf_file(file_path, prefix);
    } else if (is_safetensors_file(file_path)) {
```
Insert the new branch between the gguf branch and the safetensors branch:
```cpp
    } else if (ends_with(file_path, ".json")) {
        LOG_INFO("load %s using safetensors index format", file_path.c_str());
        return init_from_safetensors_index_file(file_path, prefix);
```

**Edit 4 — `src/model_loader.cpp`, new method** (fork-style, re-enters `init_from_file`; no `add_file_path` needed).
Anchor (fork, unique) — the comment line
`/*================================================= TorchLegacyModelLoader ==================================================*/`.
Insert **immediately before** that comment (i.e. right after the end of `init_from_safetensors_file`):
```cpp
bool ModelLoader::init_from_safetensors_index_file(const std::string& file_path, const std::string& prefix) {
    LOG_DEBUG("init from safetensors index '%s', prefix = '%s'", file_path.c_str(), prefix.c_str());

    std::vector<std::string> shard_paths;
    std::string error;
    if (!read_safetensors_index_file(file_path, shard_paths, &error)) {
        LOG_ERROR("%s", error.c_str());
        return false;
    }

    for (const std::string& shard_path : shard_paths) {
        if (!init_from_file(shard_path, prefix)) {
            return false;
        }
    }

    return true;
}
```

**Edit 5 — `src/model_loader.h`**
Anchor (fork, private section):
```cpp
    bool init_from_diffusers_file(const std::string& file_path, const std::string& prefix = "");
```
Add immediately after:
```cpp
    bool init_from_safetensors_index_file(const std::string& file_path, const std::string& prefix = "");
```

Rationale: adds ubiquitous multi-shard HF safetensors loading with zero coupling to the fork's
custom loader/VRAM architecture; the models the fork ports increasingly ship sharded-only.

---

### A2 — Safetensors header/offset bounds hardening  (LOW value, defensive, output-neutral)

Three additive guards in `read_safetensors_file` / `is_safetensors_file`. They fire **only on
malformed/corrupt files** (valid files satisfy `header_size ≤ file_size-8` and `begin ≤ end ≤ data_size`),
so they cannot change loading of any valid model. Turns "later out-of-range read/crash" into a clean
error. Self-contained. The refined header check also makes the `file_size_ - data_start` subtraction
underflow-safe. All referenced symbols already exist in the fork.

**Edit 1 — `src/model_io/safetensors_io.cpp`, `is_safetensors_file`**
Replace:
```cpp
    if (header_size_ >= file_size_ || header_size_ <= 2) {
```
with:
```cpp
    if (header_size_ > file_size_ - ST_HEADER_SIZE_LEN || header_size_ <= 2) {
```

**Edit 2 — `src/model_io/safetensors_io.cpp`, `read_safetensors_file` header check**
Replace:
```cpp
    size_t header_size_ = model_io::read_u64(header_size_buf);
    if (header_size_ >= file_size_) {
        set_error(error, "invalid safetensor file '" + file_path + "'");
        return false;
    }
```
with:
```cpp
    size_t header_size_ = model_io::read_u64(header_size_buf);
    if (header_size_ > file_size_ - ST_HEADER_SIZE_LEN) {
        set_error(error, "invalid safetensor file '" + file_path + "'");
        return false;
    }
    const size_t data_start = ST_HEADER_SIZE_LEN + header_size_;
```

**Edit 3 — `src/model_io/safetensors_io.cpp`, per-tensor offset check**
Anchor:
```cpp
        size_t begin = tensor_info["data_offsets"][0].get<size_t>();
        size_t end   = tensor_info["data_offsets"][1].get<size_t>();
```
Insert immediately after:
```cpp
        if (begin > end || end > file_size_ - data_start) {
            set_error(error, "data offsets out of bounds for tensor '" + name + "'");
            return false;
        }
```
(Optional cosmetic: the existing `TensorStorage(... , ST_HEADER_SIZE_LEN + header_size_ + begin)` may
be simplified to `data_start + begin`; identical value, not required.)

Rationale: pure robustness guard against corrupt/truncated safetensors; no effect on valid loads.

---

## KEEP-FORK

**Tokenizers — `normalize_before_split` (upstream #1670, `7f0e728`).** Adds a flag that normalizes the
whole CLIP prompt (lowercase + whitespace_clean) **before** special-token splitting instead of
per-token after. This changes core CLIP tokenization ordering for **every** SD/CLIP image prompt to fix
a narrow case: mixed-case textual-inversion embedding names. It is not self-contained — the real fix
lives in `src/conditioning/conditioner.hpp` (`normalize_embedding_name` / `append_embedding_tokens`),
which the fork rewrote (RunnerWeightManager). Output-changing + cross-cutting + niche benefit → too
risky for a GPU-validated engine.

**Streaming writers (`StreamingModelWriter`, `GGUFStreamingWriter`, `SafetensorsStreamingWriter`,
`TensorWritePlan`, new `streaming_writer.h`).** A whole new model-*writing* subsystem for the
convert/quantize path. Zero consumers in the fork (`git grep` for all four symbols outside `model_io/`
= nothing); would need conversion-pipeline wiring to be anything but dead code. New unused surface, not
a fix.

**`read_safetensors_file` `metadata` out-param + `__metadata__` extraction.** Backward-compatible but
no consumer in the fork (its lone caller uses 3 args). Dead capability.

**`gguf_io` read path.** Unchanged by upstream (only the writer was added). Nothing to adopt.

**`model_loader.cpp` architectural changes** — all KEEP-FORK: `load_tensors` signature rewrite
(member `n_threads_`, `target_tensor_names`, `log_progress`), new `ModelLoader()` ctor / `add_file_path`
/ `set_n_threads` / `model_files_processed`, selective-load `load_tensor` / `load_float_tensor`, the
imatrix quantization integration (`runtime/imatrix.h`, `convert_tensor` imatrix arg,
`get_imatrix_collector`), and the RPC-backend serialization mutex (explicitly the multi-GPU/RPC
territory to avoid). These are coupled infra + a quantization feature, not isolated fixes. The fork's
loader is deliberately its own architecture.

**`model_loader.cpp` new `VERSION_*` detections** (SEFI_IMAGE, BOOGU_IMAGE, LINGBOT_VIDEO, KREA2,
MINIT2I, HUNYUAN_VIDEO, QWEN_IMAGE_LAYERED) and the removal of `first_stage_model.bn.` from
`unused_tensors`. These are new-model support that the fork is porting through its own merge tracks
(krea2/boogu built, hunyuan/lingbot/minit2i in progress per project notes). Owned by the model-port
work — do not double-adopt here.

**`name_conversion.cpp` (+323 lines).** All new-model name maps / conversions:
`convert_qwen3_vl_vision_name`, `convert_hunyuan_video_to_original_flux`,
`convert_diffusers_dit_to_original_krea2`, `convert_diffusers_controlnet_to_original_sdxl`,
`convert_diffusers_to_original_wan_vae` (the fork already owns a Wan VAE/VACE engine — overlap risk),
`convert_esrgan_tensor_name`, plus the `convert_sep_to_dot` tweak adding `cross_attn`/`output_proj` to
the no-split list (alters an existing shared conversion for all models carrying those substrings —
risky, benefit only for the new models). These belong to the model-port tracks, not the I/O layer.

**`tensor_storage.h` `TensorWritePlan` struct.** Only consumed by the streaming writers (kept-fork
above); adopt together with A-nothing. Comes for free if/when streaming writers are ever wired, but no
reason to add now.

---

## Summary

**2 ADOPT items.** A1 — sharded safetensors `*.index.json` loading (the highest-value item: genuine,
ubiquitous new-format support the fork completely lacks, fully self-contained, verified collision-free,
and directly relevant since the fork ports the large sharded HF DiTs). A2 — safetensors
header/offset bounds hardening (safe, malformed-file-only, output-neutral, low value). Everything else
is KEEP-FORK: the CLIP `normalize_before_split` fix is output-changing and coupled to fork-rewritten
`conditioner.hpp`; the streaming-writer subsystem and metadata out-param are unused new surface; and the
loader refactor / imatrix / RPC-mutex / new `VERSION_*` detections / `name_conversion.cpp` additions are
either architectural or owned by the deliberate model-port tracks.
