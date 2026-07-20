# STREAMING / BOUNDED-MEMORY CONVERT — apply-ready integration edits

Fork worktree: `/home/dbrain/dev/longcat-avatar-mergewt` (branch `upstream-merge-2026-07-19`)
Upstream tip `ea4e566`, merge base `19bdfe2`, fork HEAD. Roadmap Batch 6; audit `audit-core-io.md` #27/#28/#31 + `trackB-io-loader.md`.
Every anchor below was `git show ea4e566:` / `Read` / `grep`-verified against the CURRENT fork tree.
**No source was edited; nothing built. Nothing here is convert-run or GPU validated (no large model exercised).**

Upstream `convert.cpp` streams tensors to disk one at a time under a 1 GB memory budget across N worker
threads, so a whole model never lives in one `ggml_ctx` (this is what caps on-box DiT quant size today). The
fork is **half-scaffolded**: `src/model_io/streaming_writer.h` was copied in and references `TensorWritePlan`,
but that type is **defined nowhere in the fork** → the header is an orphan that would not compile if `#include`d
(verified: `grep -rn '#include.*streaming_writer' src/` returns nothing — it is included by no TU today).

The fork's `src/convert.cpp` also carries a **fork-unique nvfp4/imatrix hook** that upstream lacks. The port must
compose that hook with the streaming path, not drop it. The good news (see §7): the fork's imatrix is a **global**
consumed deep inside `convert_tensor()` — orthogonal to how tensors reach disk — so the re-weave is nearly free.

---

## 0. TL;DR verdict (read first)

| # | Capability | Portability to fork | Where |
|---|---|---|---|
| **A. `TensorWritePlan` type** (un-orphan `streaming_writer.h`) | **Clean, trivial.** Upstream's struct drops verbatim into the fork's `tensor_storage.h` between the byte-identical `TensorWriteInfo` struct and the `on_new_tensor_cb_t` typedef. Zero behavior change. | §1 |
| **B. `ModelLoader::load_tensor(storage, dst)` — singular selective load** | **THE missing primitive.** Fork has only bulk `load_tensors(cb)` — no per-tensor selective load. Upstream's `load_tensor` delegates to a 4-arg `load_tensors(cb, …, target_tensor_names, …)` overload the fork **does not have**. Re-implement on the fork's existing `load_tensors(cb)` (callback matches one name, returns `nullptr` for the rest → fork's loop `continue`s without reading). Force `n_threads=1` to avoid nested thread pools. | §2 |
| **C. `sd_type_to_ggml_type` helper** | **Trivial / optional.** Fork lacks it (audit #22); fork casts `(ggml_type)output_type` directly. Either add the one-liner or keep the cast. | §3 |
| **D. `GGUFStreamingWriter`** (metadata-then-seeked-tensor) | **Ports ~verbatim** into the fork's `gguf_io.{h,cpp}`. Every dep present in fork (`gguf_init_empty`, `gguf_write_to_file_ptr`, `gguf_get_meta_size`, `gguf_get_tensor_offset`, `set_error`). | §4 |
| **E. `SafetensorsStreamingWriter`** | **Ports ~verbatim** into `safetensors_io.{h,cpp}`. Deps present (`ggml_type_to_safetensors_dtype`, `nlohmann::ordered_json`, `set_error`, `ST_HEADER_SIZE_LEN`). | §5 |
| **F. `convert.cpp` streaming rewrite** (the risky restructure) | **New code, no 1:1 anchor.** Add upstream's `collect_tensors_for_export` / `load_tensor_for_export` / `stream_tensor_data` / `write_model_file_streaming` / `export_loaded_model` / `convert_with_components`. **RECOMMENDED: keep the fork's existing whole-`ctx` `convert()` UNCHANGED as the default; add streaming as an opt-in path** so non-streaming output is byte-identical by construction. Do NOT adopt upstream's `runtime/imatrix.h` param-threaded imatrix. | §6 |
| **G. nvfp4 imatrix re-weave** | **Nearly free.** Fork imatrix = global `g_convert_imatrix`, consumed in `convert_tensor()` inside `load_tensors()`. Streaming's per-tensor load routes through the SAME `convert_tensor`, so the global applies automatically. `convert_with_imatrix()` keeps setting the global around whichever convert runs. | §7 |
| **H. Public API + CLI** | **Additive.** Optionally export `convert_with_components` (mirror upstream) + `n_threads`/`--threads` on the CONVERT CLI. Not required if streaming stays internal. koblem/server surface out of scope (flag owner). | §8 |

**Bottom line:** the only genuinely missing primitive is `ModelLoader::load_tensor` (§2); everything else is
either a verbatim drop (`TensorWritePlan`, both streaming writers) or new orchestration in `convert.cpp` (§6).
The imatrix "re-weave" (the one non-mechanical step the roadmap flags) turns out to be **almost a no-op** because
the fork threads imatrix through a global that lives below the write path (§7). Biggest risks: the `load_tensor`
correctness/cost (O(N) metadata scan per tensor), and NOT regressing the working non-streaming convert — both
handled by keeping the old path as the untouched default and landing streaming opt-in first.

---

## 1. `src/model_io/tensor_storage.h` — add `TensorWritePlan` (un-orphan the writer)

The surrounding context (`TensorWriteInfo` struct + the `on_new_tensor_cb_t` typedef) is **byte-identical**
between fork and upstream, so this is a clean insert in the exact upstream position.

Anchor (unique, fork lines 124-130):

```cpp
struct TensorWriteInfo {
    int64_t ne[SD_MAX_DIMS] = {1, 1, 1, 1, 1};
    int n_dims              = 0;
    ggml_tensor* tensor     = nullptr;
};

typedef std::function<bool(const TensorStorage&, ggml_tensor**)> on_new_tensor_cb_t;
```

Replace with (insert `TensorWritePlan` between the struct and the typedef — matches upstream `tensor_storage.h:130`):

```cpp
struct TensorWriteInfo {
    int64_t ne[SD_MAX_DIMS] = {1, 1, 1, 1, 1};
    int n_dims              = 0;
    ggml_tensor* tensor     = nullptr;
};

struct TensorWritePlan {
    std::string name;
    ggml_type type          = GGML_TYPE_F32;
    int64_t ne[SD_MAX_DIMS] = {1, 1, 1, 1, 1};
    int n_dims              = 0;

    int64_t nelements() const {
        int64_t n = 1;
        for (int i = 0; i < SD_MAX_DIMS; i++) {
            n *= ne[i];
        }
        return n;
    }

    uint64_t nbytes() const {
        return nelements() * ggml_type_size(type) / ggml_blck_size(type);
    }
};

typedef std::function<bool(const TensorStorage&, ggml_tensor**)> on_new_tensor_cb_t;
```

> Verified present in the fork's `tensor_storage.h` scope: `SD_MAX_DIMS`, `ggml_type`, `ggml_type_size`,
> `ggml_blck_size`, `<string>`/`<functional>` (the file already uses `std::string` and `std::function`).
> After this edit `streaming_writer.h` (which references `TensorWritePlan`) compiles — it is currently an orphan
> (§0). **This edit alone is inert** (nothing constructs a `TensorWritePlan` until §4/§5/§6). Land it first.

---

## 2. `src/model_loader.{h,cpp}` — add `load_tensor()` (singular selective) — THE missing primitive

Upstream's streaming path calls `model_loader.load_tensor(job.info.storage, tensor)` once per tensor. Upstream's
`load_tensor` (upstream `model_loader.cpp:1285`) delegates to a **4-arg** `load_tensors(cb, false,
&target_tensor_names, false)` overload with a selective-name filter. **The fork has no such overload** — its
`load_tensors(on_new_tensor_cb_t, int n_threads = 0, bool use_mmap = false)` (fork `model_loader.h:68`) takes no
`target_tensor_names`. So re-implement `load_tensor` on the fork's existing bulk `load_tensors(cb)`: the callback
returns the destination tensor only for the matching name and `nullptr` for every other tensor. Verified in the
fork's loader loop, when the callback sets `*dst_tensor == nullptr` the worker **`continue`s without reading**
(fork `model_loader.cpp:1112-1116`):

```cpp
                    if (dst_tensor == nullptr) {
                        t1 = ggml_time_ms();
                        read_time_ms.fetch_add(t1 - t0);
                        continue;
                    }
```

so non-target tensors cost only a metadata visit, not a disk read. And `process_model_files()` is idempotent
(`if (model_files_processed) return;`, fork `model_loader.cpp:861-863`), so calling `load_tensors` per tensor
re-parses file metadata **once**, not N times.

### 2a. Declaration — `src/model_loader.h`

Anchor (unique, fork lines 68-72):

```cpp
    bool load_tensors(on_new_tensor_cb_t on_new_tensor_cb, int n_threads = 0, bool use_mmap = false);
    bool load_tensors(std::map<std::string, ggml_tensor*>& tensors,
                      std::set<std::string> ignore_tensors = {},
                      int n_threads                        = 0,
                      bool use_mmap                        = false);
```

Replace with (append the singular declaration):

```cpp
    bool load_tensors(on_new_tensor_cb_t on_new_tensor_cb, int n_threads = 0, bool use_mmap = false);
    bool load_tensors(std::map<std::string, ggml_tensor*>& tensors,
                      std::set<std::string> ignore_tensors = {},
                      int n_threads                        = 0,
                      bool use_mmap                        = false);

    // Selectively load a single tensor (by storage identity) into an already-
    // allocated dst tensor, converting to dst->type via the shared convert_tensor
    // path (so the active g_convert_imatrix applies). Used by streaming convert.
    bool load_tensor(const TensorStorage& tensor_storage, ggml_tensor* dst_tensor);
```

### 2b. Implementation — `src/model_loader.cpp`

Insert **after** the `load_tensors(std::map<…>&, …)` overload's closing brace. Anchor (unique, fork lines
1351-1354 — the tail of that overload, immediately before `tensor_should_be_converted`):

```cpp
    if (some_tensor_not_init) {
        return false;
    }
    return true;
}

bool ModelLoader::tensor_should_be_converted(const TensorStorage& tensor_storage, ggml_type type) {
```

Replace with (insert the new method between them; adapted from upstream `model_loader.cpp:1285`, but built on the
fork's `load_tensors(cb, n_threads, use_mmap)` — NOT the absent 4-arg selective overload):

```cpp
    if (some_tensor_not_init) {
        return false;
    }
    return true;
}

bool ModelLoader::load_tensor(const TensorStorage& tensor_storage, ggml_tensor* dst_tensor) {
    if (dst_tensor == nullptr || dst_tensor->data == nullptr) {
        LOG_ERROR("load tensor failed: null destination for '%s'", tensor_storage.name.c_str());
        return false;
    }

    bool loaded = false;
    auto on_new_tensor_cb = [&](const TensorStorage& current_tensor_storage, ggml_tensor** out_tensor) -> bool {
        *out_tensor = nullptr;
        if (current_tensor_storage.name != tensor_storage.name) {
            return true;  // skip: fork loop continues without reading
        }
        if (current_tensor_storage.file_index != tensor_storage.file_index ||
            current_tensor_storage.offset != tensor_storage.offset ||
            current_tensor_storage.index_in_zip != tensor_storage.index_in_zip) {
            LOG_ERROR("load tensor failed: storage mismatch for '%s'", tensor_storage.name.c_str());
            return false;
        }
        if (current_tensor_storage.n_dims != tensor_storage.n_dims ||
            current_tensor_storage.nelements() != tensor_storage.nelements()) {
            LOG_ERROR("load tensor failed: metadata changed for '%s'", tensor_storage.name.c_str());
            return false;
        }
        for (int i = 0; i < current_tensor_storage.n_dims; i++) {
            if (current_tensor_storage.ne[i] != dst_tensor->ne[i]) {
                LOG_ERROR("load tensor failed: shape mismatch for '%s'", tensor_storage.name.c_str());
                return false;
            }
        }
        *out_tensor = dst_tensor;
        loaded      = true;
        return true;
    };

    // n_threads=1: this runs *inside* the streaming worker pool (§6). Forcing a
    // single inner thread avoids nested thread-pool explosion (N workers each
    // spinning up N loader threads). use_mmap=false: streaming pre-disabled mmap.
    if (!load_tensors(on_new_tensor_cb, /*n_threads=*/1, /*use_mmap=*/false)) {
        LOG_ERROR("load tensor failed: '%s'", tensor_storage.name.c_str());
        return false;
    }
    if (!loaded) {
        LOG_ERROR("load tensor failed: tensor '%s' not found", tensor_storage.name.c_str());
        return false;
    }
    return true;
}

bool ModelLoader::tensor_should_be_converted(const TensorStorage& tensor_storage, ggml_type type) {
```

> Verified fork fields on `TensorStorage`: `name`, `file_index`, `offset`, `index_in_zip`, `n_dims`, `ne[]`,
> `nelements()` (all used by the fork loader loop + `chunk()`/`reverse_ne()` in `tensor_storage.h`).
> **Cost note (RISK, §9):** each call iterates the full `file_data[].tensors` metadata list (O(N)) → O(N²)
> metadata scanning for the whole model, but **no O(N²) disk reads** (non-matching tensors `continue`). For a
> DiT of a few thousand tensors this is a bounded CPU cost, not I/O. If it ever matters, a future optimization is
> to add the real selective `target_tensor_names` short-circuit to `process_model_files`/`load_tensors` — out of
> scope here.
>
> **Determinism / conversion:** because the match returns `dst_tensor` with `dst_tensor->type == job.info.type`,
> the fork's loader hits its `tensor_storage.type != dst_tensor->type` branch and calls
> `convert_tensor(target_buf, tensor_storage.type, convert_buf, dst_tensor->type, …, dst_tensor->name)`
> (fork `model_loader.cpp:1207-1214`) — the SAME quantizer the whole-model path uses, imatrix included (§7).

---

## 3. `sd_type_to_ggml_type` — trivial helper (OPTIONAL)

Upstream's streaming `export_loaded_model` uses `sd_type_to_ggml_type(output_type)` (audit #22). The fork lacks
it and casts `(ggml_type)output_type` directly (fork `convert.cpp:106`). **Recommendation: keep the fork's cast**
in §6 (`ggml_type type = (ggml_type)output_type;`) — it is exactly what the fork's existing `convert()` does, so
no new symbol is needed. If you prefer parity, add to `src/core/util.{h,cpp}`:

```cpp
// util.h
ggml_type sd_type_to_ggml_type(sd_type_t type);
// util.cpp
ggml_type sd_type_to_ggml_type(sd_type_t type) { return (ggml_type)type; }
```

Not required. The rest of this doc uses the cast to minimize surface.

---

## 4. `src/model_io/gguf_io.{h,cpp}` — `GGUFStreamingWriter` (ports ~verbatim)

All deps verified present in the fork's `gguf_io.cpp`: `set_error` (static, fork `gguf_io.cpp:12`),
`gguf_init_empty` (used fork `gguf_io.cpp:100`), `gguf_get_tensor_offset` (used fork `gguf_io.cpp:77`),
`gguf_write_to_file_ptr` + `gguf_get_meta_size` (declared `ggml/include/gguf.h:200/204`).

### 4a. Header — `src/model_io/gguf_io.h`

Anchor (unique, fork — the whole current header body):

```cpp
#include <string>
#include <vector>

#include "tensor_storage.h"

bool is_gguf_file(const std::string& file_path);
bool read_gguf_file(const std::string& file_path,
                    std::vector<TensorStorage>& tensor_storages,
                    std::string* error = nullptr);
bool write_gguf_file(const std::string& file_path,
                     const std::vector<TensorWriteInfo>& tensors,
                     std::string* error = nullptr);
```

Replace with (add the `streaming_writer.h` include + the class; keep the old `write_gguf_file` untouched):

```cpp
#include <string>
#include <vector>

#include "streaming_writer.h"
#include "tensor_storage.h"

struct ggml_context;
struct gguf_context;

bool is_gguf_file(const std::string& file_path);
bool read_gguf_file(const std::string& file_path,
                    std::vector<TensorStorage>& tensor_storages,
                    std::string* error = nullptr);
bool write_gguf_file(const std::string& file_path,
                     const std::vector<TensorWriteInfo>& tensors,
                     std::string* error = nullptr);

class GGUFStreamingWriter : public StreamingModelWriter {
public:
    GGUFStreamingWriter() = default;
    ~GGUFStreamingWriter();

    bool write_metadata(const std::string& file_path,
                        const std::vector<TensorWritePlan>& tensors,
                        std::string* error = nullptr) override;
    bool write_tensor(std::ostream& output,
                      size_t tensor_index,
                      const uint8_t* data,
                      size_t size,
                      std::string* error = nullptr) const override;
    uint64_t file_size() const override;
    void close();

private:
    std::vector<TensorWritePlan> tensors_;
    std::vector<uint64_t> tensor_offsets_;
    uint64_t file_size_     = 0;
    ggml_context* meta_ctx_ = nullptr;
    gguf_context* gguf_ctx_ = nullptr;
};
```

### 4b. Implementation — `src/model_io/gguf_io.cpp`

Copy upstream `git show ea4e566:src/model_io/gguf_io.cpp`'s `GGUFStreamingWriter::{~,write_metadata,write_tensor,
file_size,close}` methods **verbatim** and append them to the fork's `gguf_io.cpp` (after `write_gguf_file`). They
reference only fork-present symbols (`set_error`, `ggml_init`, `ggml_tensor_overhead`, `ggml_new_tensor`,
`ggml_set_name`, `gguf_init_empty`, `gguf_add_tensor`, `gguf_write_to_file_ptr`, `gguf_get_meta_size`,
`gguf_get_tensor_offset`, `gguf_free`, `ggml_free`, `LOG_INFO`). Ensure `<ostream>` is available (the base class
takes `std::ostream&`; `streaming_writer.h` includes `<iosfwd>`, so add `#include <ostream>` to `gguf_io.cpp` if
not already transitively present). No other change.

> The GGUF on-disk **tensor order** in the streaming writer follows the `TensorWritePlan` vector order, which in
> §6 comes from `collect_tensors_for_export` = iteration of `get_tensor_storage_map()` (a `std::map`, name-sorted,
> deterministic). The old `write_gguf_file` path orders by (multi-threaded, nondeterministic) load order. Both are
> valid GGUF; they are NOT guaranteed byte-identical to each other — see §9 (this is why §6 keeps the old path as
> the default and validates streaming by round-trip, not file-diff).

---

## 5. `src/model_io/safetensors_io.{h,cpp}` — `SafetensorsStreamingWriter` (ports ~verbatim)

Deps verified present in the fork's `safetensors_io.cpp`: `ggml_type_to_safetensors_dtype` (fork
`safetensors_io.cpp:289`), `nlohmann::ordered_json` (used fork `safetensors_io.cpp:311`), `set_error` (fork
`safetensors_io.cpp:18`), `ST_HEADER_SIZE_LEN` (fork `safetensors_io.cpp:15`). Note the fork's `read_safetensors_file`
signature is the **3-arg** form (no `metadata` out-param — trackB declined that) and the fork already carries the
`read_safetensors_index_file` decl + bounds hardening (trackB A1/A2 landed). Preserve those.

### 5a. Header — `src/model_io/safetensors_io.h`

Anchor (unique, fork lines 1-17 region — the include block + write decl):

```cpp
#include <map>
#include <string>
#include <vector>

#include "tensor_storage.h"

bool is_safetensors_file(const std::string& file_path);
bool read_safetensors_file(const std::string& file_path,
                           std::vector<TensorStorage>& tensor_storages,
                           std::string* error = nullptr);
```

> **Verify the exact current fork wording** of `read_safetensors_file` (3-arg above) and whether
> `read_safetensors_index_file` is already declared here — do not disturb either. Only ADD the `streaming_writer.h`
> include and the class.

Add `#include "streaming_writer.h"` next to `#include "tensor_storage.h"`, and insert the class **after**
`write_safetensors_file(...)`'s declaration:

```cpp
class SafetensorsStreamingWriter : public StreamingModelWriter {
public:
    SafetensorsStreamingWriter() = default;

    bool write_metadata(const std::string& file_path,
                        const std::vector<TensorWritePlan>& tensors,
                        std::string* error = nullptr) override;
    bool write_tensor(std::ostream& output,
                      size_t tensor_index,
                      const uint8_t* data,
                      size_t size,
                      std::string* error = nullptr) const override;
    uint64_t file_size() const override;

private:
    std::string file_path_;
    std::vector<TensorWritePlan> tensors_;
    std::vector<uint64_t> tensor_offsets_;
    uint64_t data_start_ = 0;
    uint64_t file_size_  = 0;
};
```

### 5b. Implementation — `src/model_io/safetensors_io.cpp`

Copy upstream's `SafetensorsStreamingWriter::{write_metadata,write_tensor,file_size}` **verbatim** and append to the
fork's `safetensors_io.cpp`. Add `#include <ostream>` if not transitively present. All referenced symbols exist in
the fork.

> **Header padding note (RISK, §9):** upstream's streaming `write_metadata` writes `header.dump()` with **no
> alignment padding**. If the fork's existing `write_safetensors_file` pads the JSON header to an alignment
> boundary, the streaming safetensors output will differ from the old path at the header (both remain valid
> safetensors). Another reason §6 keeps the old writer as the default for `.safetensors`.

---

## 6. `src/convert.cpp` — the streaming path (RISKY restructure; keep old path as default)

**Design decision (RECOMMENDED, lowest-risk):** DO NOT replace the fork's existing whole-`ctx` `convert()` /
`load_tensors_for_export` (fork `convert.cpp:40-154`). Leave that code **untouched** so the default,
non-streaming convert is **byte-identical by construction** (its code path does not change at all). ADD the
streaming functions alongside, exposed as an **opt-in** (`convert_with_components` with `n_threads`, mirroring
upstream) and selected explicitly (public API §8 and/or an env/CLI gate). This satisfies the roadmap's "finish the
streaming writer" without risking the working quant path. Flipping the default to streaming (upstream's structure,
where `convert()` delegates to `convert_with_components`) is a **follow-up** gated on round-trip validation (§9).

### 6a. Includes

Anchor (unique, fork lines 1-15):

```cpp
#include <cstring>
#include <map>
#include <mutex>
#include <regex>
#include <vector>

#include "core/ggml_extend.hpp"  // SDImatrixCollector (nvfp4-twolevel)
#include "gguf.h"                 // gguf_init_from_file (imatrix load)
#include "model_io/gguf_io.h"
#include "model_io/safetensors_io.h"
#include "model_loader.h"
#include "stable-diffusion.h"
#include "util.h"

#include "ggml_extend_backend.h"
```

Replace with (add the threading/streaming headers + `streaming_writer.h`; keep every existing include — the fork's
`core/ggml_extend.hpp`/`gguf.h`/`ggml_extend_backend.h` are needed by the imatrix code that stays):

```cpp
#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <thread>
#include <vector>

#include "core/ggml_extend.hpp"  // SDImatrixCollector (nvfp4-twolevel)
#include "gguf.h"                 // gguf_init_from_file (imatrix load)
#include "model_io/gguf_io.h"
#include "model_io/safetensors_io.h"
#include "model_io/streaming_writer.h"
#include "model_loader.h"
#include "stable-diffusion.h"
#include "util.h"

#include "ggml_extend_backend.h"
```

### 6b. Streaming helpers — insert after the existing `get_export_tensor_type` (fork `convert.cpp:38`)

The fork already defines `get_export_tensor_type` (fork `convert.cpp:17-38`), **identical** to upstream — reuse it,
do not duplicate. Insert the following block **after** it (before the existing `load_tensors_for_export`). This is
upstream's streaming machinery, verbatim except it calls the fork's new `load_tensor` (§2) and uses
`sd_get_num_physical_cores()` (fork `util.cpp:306`) + `pretty_bytes_progress()` (fork `util.cpp:547`):

```cpp
struct TensorExportInfo {
    TensorStorage storage;
    ggml_type type;
};

struct TensorExportJob {
    TensorExportInfo info;
    std::vector<uint8_t> data;
    std::string error;
    bool success = false;
};

static bool collect_tensors_for_export(ModelLoader& model_loader,
                                       ggml_type type,
                                       const TensorTypeRules& tensor_type_rules,
                                       std::vector<TensorExportInfo>& tensors) {
    tensors.clear();
    tensors.reserve(model_loader.get_tensor_storage_map().size());
    for (const auto& kv : model_loader.get_tensor_storage_map()) {
        const TensorStorage& tensor_storage = kv.second;
        TensorExportInfo info;
        info.storage = tensor_storage;
        info.type    = get_export_tensor_type(model_loader, tensor_storage, type, tensor_type_rules);
        tensors.push_back(std::move(info));
    }
    LOG_INFO("collected %zu tensors for export", tensors.size());
    return true;
}

static size_t export_tensor_nbytes(const TensorExportInfo& info) {
    TensorStorage output_storage = info.storage;
    output_storage.type          = info.type;
    return static_cast<size_t>(output_storage.nbytes());
}

static TensorWritePlan tensor_write_plan_from_export_info(const TensorExportInfo& info) {
    TensorWritePlan plan;
    plan.name   = info.storage.name;
    plan.type   = info.type;
    plan.n_dims = info.storage.n_dims;
    for (int i = 0; i < SD_MAX_DIMS; i++) {
        plan.ne[i] = info.storage.ne[i];
    }
    return plan;
}

static std::vector<TensorWritePlan> tensor_write_plans_from_export_infos(const std::vector<TensorExportInfo>& tensors) {
    std::vector<TensorWritePlan> plans;
    plans.reserve(tensors.size());
    for (const TensorExportInfo& info : tensors) {
        plans.push_back(tensor_write_plan_from_export_info(info));
    }
    return plans;
}

static bool preallocate_output_file(const std::string& output_path, uint64_t file_size, std::string* error) {
    if (file_size == 0) {
        return true;
    }
    std::fstream file(output_path, std::ios::binary | std::ios::in | std::ios::out);
    if (!file.is_open()) {
        if (error != nullptr) {
            *error = "failed to open output file '" + output_path + "' for preallocation";
        }
        return false;
    }
    file.seekp(static_cast<std::streamoff>(file_size - 1), std::ios::beg);
    file.put('\0');
    file.flush();
    if (!file) {
        if (error != nullptr) {
            *error = "failed to preallocate output file '" + output_path + "'";
        }
        return false;
    }
    return true;
}

static bool load_tensor_for_export(ModelLoader& model_loader, TensorExportJob& job) {
    size_t mem_size = 1 * 1024 * 1024;
    mem_size += ggml_tensor_overhead();
    TensorStorage output_storage = job.info.storage;
    output_storage.type          = job.info.type;
    mem_size += static_cast<size_t>(output_storage.nbytes());

    ggml_context* ggml_ctx = ggml_init({mem_size, nullptr, false});
    if (ggml_ctx == nullptr) {
        job.error = "ggml_init failed for tensor '" + job.info.storage.name + "'";
        return false;
    }
    ggml_tensor* tensor = ggml_new_tensor(ggml_ctx, job.info.type, job.info.storage.n_dims, job.info.storage.ne);
    if (tensor == nullptr) {
        ggml_free(ggml_ctx);
        job.error = "ggml_new_tensor failed for tensor '" + job.info.storage.name + "'";
        return false;
    }
    ggml_set_name(tensor, job.info.storage.name.c_str());

    const size_t tensor_nbytes = ggml_nbytes(tensor);
    if (tensor_nbytes > 0 && !model_loader.load_tensor(job.info.storage, tensor)) {
        ggml_free(ggml_ctx);
        job.error = "failed to load tensor '" + job.info.storage.name + "'";
        return false;
    }
    job.data.resize(tensor_nbytes);
    if (tensor_nbytes > 0) {
        memcpy(job.data.data(), tensor->data, tensor_nbytes);
    }
    ggml_free(ggml_ctx);
    return true;
}

static bool stream_tensor_data(ModelLoader& model_loader,
                               const std::string& output_path,
                               const std::vector<TensorExportInfo>& tensors,
                               const StreamingModelWriter& writer,
                               int n_threads,
                               std::string* error) {
    n_threads = n_threads > 0 ? n_threads : sd_get_num_physical_cores();
    n_threads = std::max(1, n_threads);
    LOG_INFO("streaming convert with %d threads", n_threads);

    int64_t start_time       = ggml_time_ms();
    uint64_t bytes_written   = 0;
    size_t tensors_written   = 0;
    size_t next_tensor_index = 0;
    bool failed              = false;
    std::string failure;

    const size_t memory_budget = 1024ull * 1024ull * 1024ull;
    size_t reserved_bytes      = 0;

    std::mutex work_mutex;
    std::mutex progress_mutex;
    std::condition_variable memory_cv;
    std::vector<std::thread> workers;
    workers.reserve(n_threads);

    auto reserve_memory = [&](size_t bytes) -> bool {
        std::unique_lock<std::mutex> lock(work_mutex);
        memory_cv.wait(lock, [&]() {
            return failed || reserved_bytes == 0 || reserved_bytes + bytes <= memory_budget;
        });
        if (failed) {
            return false;
        }
        reserved_bytes += bytes;
        return true;
    };
    auto release_memory = [&](size_t bytes) {
        {
            std::lock_guard<std::mutex> lock(work_mutex);
            reserved_bytes -= std::min(reserved_bytes, bytes);
        }
        memory_cv.notify_all();
    };
    auto fail = [&](const std::string& message) {
        {
            std::lock_guard<std::mutex> lock(work_mutex);
            if (!failed) {
                failed  = true;
                failure = message;
            }
        }
        memory_cv.notify_all();
    };

    for (int worker = 0; worker < n_threads; worker++) {
        workers.emplace_back([&]() {
            std::fstream output_file(output_path, std::ios::binary | std::ios::in | std::ios::out);
            if (!output_file.is_open()) {
                fail("failed to open output file '" + output_path + "' for tensor writing");
                return;
            }
            while (true) {
                size_t tensor_index = 0;
                {
                    std::lock_guard<std::mutex> lock(work_mutex);
                    if (failed || next_tensor_index >= tensors.size()) {
                        return;
                    }
                    tensor_index = next_tensor_index++;
                }
                const size_t tensor_bytes = export_tensor_nbytes(tensors[tensor_index]);
                if (!reserve_memory(tensor_bytes)) {
                    return;
                }
                TensorExportJob job;
                job.info = tensors[tensor_index];
                try {
                    job.success = load_tensor_for_export(model_loader, job);
                } catch (const std::exception& e) {
                    job.error   = e.what();
                    job.success = false;
                }
                if (!job.success) {
                    release_memory(tensor_bytes);
                    fail(job.error.empty() ? "streaming conversion failed" : job.error);
                    return;
                }
                std::string write_error;
                if (!writer.write_tensor(output_file,
                                         tensor_index,
                                         job.data.empty() ? nullptr : job.data.data(),
                                         job.data.size(),
                                         &write_error)) {
                    release_memory(tensor_bytes);
                    fail(write_error.empty() ? "streaming conversion write failed" : write_error);
                    return;
                }
                {
                    std::lock_guard<std::mutex> lock(progress_mutex);
                    bytes_written += job.data.size();
                    tensors_written++;
                    float elapsed_seconds = (ggml_time_ms() - start_time) / 1000.0f;
                    pretty_bytes_progress(static_cast<int>(tensors_written),
                                          static_cast<int>(tensors.size()),
                                          bytes_written,
                                          elapsed_seconds);
                }
                release_memory(tensor_bytes);
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    printf("\n");
    if (failed) {
        if (error != nullptr) {
            *error = failure;
        }
        return false;
    }
    LOG_INFO("streaming conversion completed, taking %.2fs", (ggml_time_ms() - start_time) / 1000.f);
    return true;
}

static bool write_model_file_streaming(ModelLoader& model_loader,
                                       const std::string& output_path,
                                       const std::vector<TensorExportInfo>& tensors,
                                       StreamingModelWriter& writer,
                                       int n_threads,
                                       std::string* error) {
    std::vector<TensorWritePlan> plans = tensor_write_plans_from_export_infos(tensors);
    if (!writer.write_metadata(output_path, plans, error)) {
        return false;
    }
    if (!preallocate_output_file(output_path, writer.file_size(), error)) {
        return false;
    }
    model_loader.process_model_files(false, false);  // pin mmap OFF before threaded per-tensor loads
    return stream_tensor_data(model_loader, output_path, tensors, writer, n_threads, error);
}

static bool export_loaded_model_streaming(ModelLoader& model_loader,
                                          const char* output_path,
                                          sd_type_t output_type,
                                          const char* tensor_type_rules,
                                          int n_threads) {
    ggml_type type             = (ggml_type)output_type;  // fork convention (no sd_type_to_ggml_type; §3)
    bool output_is_safetensors = ends_with(output_path, ".safetensors");
    TensorTypeRules type_rules = parse_tensor_type_rules(tensor_type_rules);

    std::vector<TensorExportInfo> tensors;
    bool success = collect_tensors_for_export(model_loader, type, type_rules, tensors);
    std::string error;
    if (success) {
        std::unique_ptr<StreamingModelWriter> writer;
        if (output_is_safetensors) {
            writer = std::make_unique<SafetensorsStreamingWriter>();
        } else {
            writer = std::make_unique<GGUFStreamingWriter>();
        }
        success = write_model_file_streaming(model_loader, output_path, tensors, *writer, n_threads, &error);
    }
    if (!success && !error.empty()) {
        LOG_ERROR("%s", error.c_str());
    }
    return success;
}
```

> Verified fork primitives used above: `TensorTypeRules`/`parse_tensor_type_rules` (used fork `convert.cpp:108`),
> `ends_with` (`core/util.h`), `TensorStorage::nbytes()`, `SD_MAX_DIMS`, `ggml_init`/`ggml_new_tensor`/
> `ggml_nbytes`/`ggml_set_name`/`ggml_free`/`ggml_tensor_overhead`/`ggml_time_ms`, `sd_get_num_physical_cores`,
> `pretty_bytes_progress`, `process_model_files(false,false)`, `get_tensor_storage_map()`,
> `tensor_should_be_converted` (inside `get_export_tensor_type`). **NO ModelManager / RunnerWeightManager /
> `runtime/imatrix.h` is referenced** — upstream's streaming convert.cpp is architecture-clean and translates 1:1.

### 6c. Public streaming entry — `convert_with_components` (opt-in)

Add a `convert_with_components` mirroring upstream (multi-component load + streaming export). Insert after
`convert_with_imatrix` (or after the existing `convert()` — placement is free). This is the opt-in streaming API:

```cpp
bool convert_with_components(const char* model_path,
                             const char* clip_l_path,
                             const char* clip_g_path,
                             const char* t5xxl_path,
                             const char* diffusion_model_path,
                             const char* vae_path,
                             const char* output_path,
                             sd_type_t output_type,
                             const char* tensor_type_rules,
                             bool convert_name,
                             int n_threads) {
    ModelLoader model_loader;
    bool loaded_any = false;
    auto init_path = [&](const char* path, const char* prefix) -> bool {
        if (path == nullptr || strlen(path) == 0) {
            return true;
        }
        if (!model_loader.init_from_file(path, prefix)) {
            LOG_ERROR("init model loader from file failed: '%s'", path);
            return false;
        }
        loaded_any = true;
        return true;
    };

    if (!init_path(model_path, "") ||
        !init_path(clip_l_path, "text_encoders.clip_l.transformer.") ||
        !init_path(clip_g_path, "text_encoders.clip_g.transformer.") ||
        !init_path(t5xxl_path, "text_encoders.t5xxl.transformer.") ||
        !init_path(diffusion_model_path, "model.diffusion_model.") ||
        !init_path(vae_path, "vae.")) {
        return false;
    }
    if (!loaded_any) {
        LOG_ERROR("no input model path provided for convert");
        return false;
    }
    if (convert_name) {
        model_loader.convert_tensors_name();
    }
    return export_loaded_model_streaming(model_loader, output_path, output_type, tensor_type_rules, n_threads);
}
```

> Verified: fork `ModelLoader::init_from_file(path, prefix)` (fork `model_loader.cpp:259`),
> `convert_tensors_name()` (fork `model_loader.cpp:288`). The `vae.`/`text_encoders.*`/`model.diffusion_model.`
> prefixes match the fork's existing `convert()` (`vae.`) and the loader's prefix convention.
>
> **DO NOT** rewrite the fork's `convert()` to delegate to this (that is upstream's structure and would change the
> default writer — §9). Keep `convert()` (fork `convert.cpp:83-154`) exactly as-is for the default path.

---

## 7. nvfp4 imatrix re-weave — the one "non-mechanical" step (turns out nearly free)

**The fork's imatrix is not threaded through `convert()`'s arguments — it is a global.** `convert_with_imatrix()`
(fork `convert.cpp:265-287`) loads the imatrix gguf, calls `set_convert_imatrix(&imatrix)` (fork
`model_loader.cpp:41`, sets `g_convert_imatrix`), runs `convert(...)`, then clears the global. The global is read
**inside `convert_tensor()`** via `resolve_imatrix(name, n_per_row, fallback)` (fork `model_loader.cpp:183-199`),
and `convert_tensor` is called from `load_tensors()` at fork `model_loader.cpp:1207-1214` with `dst_tensor->name`.

Because the streaming path's per-tensor `load_tensor_for_export` → `ModelLoader::load_tensor` (§2) → the fork's
`load_tensors(cb)` → the **same** `convert_tensor(..., dst_tensor->name)`, the active `g_convert_imatrix` applies
**automatically** to streaming quantization, with zero extra wiring. The re-weave is therefore:

1. **Keep `convert_with_imatrix()` as the imatrix entry point** — it sets/clears the global around whichever
   convert runs. If §6 keeps the default `convert()` (non-streaming), imatrix already works there today (unchanged).
2. **To use imatrix WITH streaming**, route the streaming entry through the same global set/clear. Two options:
   - **(a) Reuse the existing wrapper unchanged** — since the global is set for the whole `convert()` call, if you
     later flip `convert()`'s body to streaming (§9 follow-up) the imatrix keeps working with no edit.
   - **(b) Add a thin `convert_with_components_imatrix(...)`** (or an imatrix arg on `convert_with_components`) that
     does the same `set_convert_imatrix(&imatrix)` / run / `set_convert_imatrix(nullptr)` bracket around
     `convert_with_components`. Recommended if you expose streaming as a distinct API (§8).

**Do NOT** adopt upstream's imatrix model here: upstream includes `runtime/imatrix.h` and threads an `imatrix`
vector as a **parameter** of `convert_tensor` (upstream `model_loader.cpp:155-161`) fed by `get_imatrix_collector()`.
The fork's global-based `convert_tensor` is a deliberate fork divergence (audit #26 KEEP-FORK). Porting upstream's
streaming `convert.cpp` does **not** drag in upstream's imatrix (upstream's streaming functions never touch imatrix
directly — they rely on `load_tensor`→`load_tensors`→`convert_tensor`, exactly as the fork does). Keep the fork's
`convert_tensor` + `g_convert_imatrix` + `resolve_imatrix` untouched.

> **Thread-safety of the global under streaming:** `g_convert_imatrix` is set once before the streaming workers
> spawn and cleared after they join; the workers only **read** it (via `resolve_imatrix`). Concurrent reads of a
> `std::map` that is not being mutated are safe. No new synchronization needed. (Verified: `resolve_imatrix` does
> lookups only; nothing in the convert path writes `g_convert_imatrix` except the outer set/clear.)

---

## 8. Public API + CLI surface (additive; optional)

### 8a. `include/stable-diffusion.h` — export `convert_with_components` (optional)

Only needed if callers (CLI/koblem) select streaming explicitly. Anchor (unique, fork lines 978-983):

```c
SD_API bool convert(const char* input_path,
                    const char* vae_path,
                    const char* output_path,
                    enum sd_type_t output_type,
                    const char* tensor_type_rules,
                    bool convert_name);
```

Insert **after** it (mirror upstream `stable-diffusion.h:541`):

```c
SD_API bool convert_with_components(const char* model_path,
                                    const char* clip_l_path,
                                    const char* clip_g_path,
                                    const char* t5xxl_path,
                                    const char* diffusion_model_path,
                                    const char* vae_path,
                                    const char* output_path,
                                    enum sd_type_t output_type,
                                    const char* tensor_type_rules,
                                    bool convert_name,
                                    int n_threads);
```

> If you want imatrix + streaming from one entry, also declare a `convert_with_components_imatrix(...)` (the §7(b)
> wrapper) next to the existing `convert_with_imatrix` (fork `stable-diffusion.h:1009`). Keep `convert_with_imatrix`
> as-is.

### 8b. CLI — `examples/cli/main.cpp` (optional)

The fork's CONVERT branch calls `convert_with_imatrix(...)` (fork `main.cpp:692-698`) with no `n_threads`.
To offer streaming, add a `--streaming-convert` (or reuse an existing `--threads`/`n_threads`) flag and, when set,
call `convert_with_components(model, clip_l, clip_g, t5xxl, diffusion, vae, out, wtype, rules, convert_name,
n_threads)` (or the imatrix wrapper). Leave the default branch on `convert_with_imatrix` so existing behavior is
unchanged. **koblem/server request surface is out of scope for this C++ doc — flag to owner** if the video/convert
server should expose streaming convert.

### 8c. Build

No CMake edit: the fork globs `src/*.cpp`, `src/model_io/*.cpp` with `CONFIGURE_DEPENDS` (fork `CMakeLists.txt:272-292`).
All edits are to **existing** files (`convert.cpp`, `gguf_io.{h,cpp}`, `safetensors_io.{h,cpp}`, `tensor_storage.h`,
`model_loader.{h,cpp}`); `streaming_writer.h` already exists. Nothing new to register.

---

## 9. Blockers, missing primitives, risks + staged apply plan

**BLOCKER-1 (missing primitive — `ModelLoader::load_tensor`, §2):** the streaming path's fundamental dependency.
The fork has no per-tensor selective load and no `target_tensor_names` overload of `load_tensors`. §2 rebuilds
`load_tensor` on the fork's bulk `load_tensors(cb)`. This is the one place the port touches the fork's
KEEP-FORK loader. It is **additive** (a new method; no existing signature changes), but it must be correct: the
callback identity checks (`file_index`/`offset`/`index_in_zip`/shape) and the `nullptr`-skips-read behavior are
load-bearing. **Cost:** O(N) metadata scan per tensor → O(N²) metadata iteration for the whole model (no O(N²)
disk reads). Bounded for typical DiTs; revisit only if convert time regresses.

**BLOCKER-2 (no large model exercised):** nothing here is convert-run tested. Unverified against a real model:
that `load_tensor_for_export`'s per-tensor `ggml_init` sizing (`1MB + overhead + output nbytes`) suffices for
every tensor's quantized type; that the streaming GGUF/safetensors files load back correctly across all quant
types the fork uses (nvfp4/fp8/Q*/f16); that the 1 GB budget + N threads actually caps RSS on the box's largest
DiT. **Requires a real convert run** (CPU-only, no GPU) — a smoke convert of one large DiT with `--tensor-type-rules`
and an imatrix, then load-back + a render A/B.

**RISK — non-streaming convert corruption: NONE, by construction.** §6 leaves the fork's `convert()` /
`load_tensors_for_export` / whole-`ctx` path **untouched**; the default convert path's bytes are unchanged because
its code is unchanged. This is the explicit reason to add streaming opt-in rather than rewrite `convert()` to
delegate (upstream's structure). Do not flip the default until BLOCKER-2 validation passes.

**RISK — streaming output is NOT byte-identical to the old writer** (and need not be): (a) tensor **order** differs
(streaming = name-sorted `std::map` iteration; old = nondeterministic multi-threaded load order); (b) the
safetensors streaming `write_metadata` writes an **unpadded** JSON header, which may differ from the fork's old
`write_safetensors_file` if that pads to alignment. Both outputs are valid GGUF/safetensors. **Validate streaming
by round-trip** (load the streamed file, compare per-tensor data / run a render), NOT by `diff` against the old
file. Note also the old `convert()` path is itself order-nondeterministic today.

**RISK — nested threading (§2 mitigation):** the streaming worker pool (N threads) calls `load_tensor` →
`load_tensors`, which itself spawns a thread pool. §2 forces `n_threads=1` on the inner call to prevent N×N thread
explosion. If that guard is dropped, expect thread-count blowup and possible oversubscription. Keep it.

**RISK — imatrix global concurrency (§7):** `g_convert_imatrix` is read by all streaming workers via
`resolve_imatrix`. Safe **only** because it is set before workers spawn and cleared after they join, and never
mutated during. Do not introduce any mid-convert mutation of the global.

**ModelManager / RunnerWeightManager entanglement: NONE to port.** Upstream's streaming `convert.cpp` is
architecture-clean — it uses `ModelLoader` + `ggml_ctx` per tensor, no ModelManager, no `params_backend`, no
weight manager. The only upstream-vs-fork translation is the imatrix model (§7, keep fork's global) and the
`sd_type_to_ggml_type` helper (§3, keep the cast). This is why Batch 6 is a genuine, self-contained DROPPED-FEATURE
rather than an architectural pull-in.

**No fork anchor exists for:** the streaming functions in §6 (new code, no upstream-parallel line in the fork's
divergent `convert.cpp`), the `convert_with_components` entry (§6c/§8a), and the CLI flag (§8b). These are placed
deliberately, guided by upstream's structure.

### Staged apply plan (add type → wire writer → re-weave imatrix → verify)

- **Stage 1 — un-orphan the type (inert, compiles, zero behavior change):**
  §1 (`TensorWritePlan` in `tensor_storage.h`). Now `streaming_writer.h` would compile if included. Nothing
  includes it yet. Verify the tree still builds off-server.
- **Stage 2 — the missing primitive + the writers (still inert; nothing constructs them):**
  §2 (`ModelLoader::load_tensor`), §4 (`GGUFStreamingWriter`), §5 (`SafetensorsStreamingWriter`), §3 (skip or add
  helper). All additive; no existing path calls any of it. Build off-server.
- **Stage 3 — wire the streaming convert path (opt-in; default convert UNCHANGED):**
  §6 (streaming helpers + `convert_with_components` + `export_loaded_model_streaming`), §7(b) optional imatrix
  wrapper, §8 (public decl + optional CLI flag). The default `convert()`/`convert_with_imatrix` still runs the old
  whole-`ctx` path. **Verify: a normal non-streaming convert (no `--streaming` flag) is byte-identical to
  pre-change** (its code is untouched — this should be trivially true; confirm with one convert).
- **Stage 4 — validate streaming on a real model (CPU convert; no GPU):**
  Run a streaming convert of one large DiT (with `--tensor-type-rules` and an imatrix), confirm bounded RSS
  (≈1 GB + per-tensor working set, not whole-model), load the result back, and A/B a render vs the non-streaming
  output of the same model. Confirm imatrix took effect (compare against a no-imatrix streaming convert). Only
  after this passes should you consider **Stage 5 (optional, deferred): flip `convert()` to delegate to
  `convert_with_components`** (upstream's default), which makes streaming the default and requires re-confirming
  the imatrix bracket still wraps it (§7). Do not do Stage 5 until Stage 4 is clean.

**Verification status:** all fork anchors (`tensor_storage.h`, `model_loader.{h,cpp}`, `convert.cpp`,
`gguf_io.{h,cpp}`, `safetensors_io.{h,cpp}`, `include/stable-diffusion.h`, `examples/cli/main.cpp`, `CMakeLists.txt`)
were `Read`/`grep`/`git show`-confirmed against the current worktree. Confirmed present in the fork:
`process_model_files` (idempotent), `pretty_bytes_progress`, `sd_get_num_physical_cores`, `set_error`,
`ggml_type_to_safetensors_dtype`, `nlohmann::ordered_json`, `ST_HEADER_SIZE_LEN`, `gguf_init_empty`,
`gguf_write_to_file_ptr`, `gguf_get_meta_size`, `gguf_get_tensor_offset`, `g_convert_imatrix`/`set_convert_imatrix`/
`resolve_imatrix`/`convert_tensor`. Confirmed ABSENT (this port adds): `TensorWritePlan`, `ModelLoader::load_tensor`,
`GGUFStreamingWriter`, `SafetensorsStreamingWriter`, `convert_with_components`, `sd_type_to_ggml_type`.
`streaming_writer.h` exists but is an orphan (included nowhere). **Cannot build or run a convert here** (project
rule: no Rust/web builds; and no large-model convert exercised — all quant/RSS/round-trip claims are Stage-4 items).
