#include "model_manager.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <cstdint>
#include <iterator>
#include <mutex>
#include <thread>
#include <unordered_set>

#include <sys/mman.h>
#include <unistd.h>

#include "core/ggml_extend_backend.h"
#include "core/util.h"
#include "model/adapter/lora.hpp"
#include "model/adapter/lora_fold.hpp"
#ifdef SD_USE_CUDA
#include "ggml-cuda.h"
#endif

// ---------------------------------------------------------------------------
// WEIGHT-CONTENT EPOCH — what the CUDA backend's byte-derived caches key on.
//
// The backend caches things computed FROM a weight's bytes (the per-tensor e4m3 scale, an
// amax) under the weight's NAME. A name is the right identity across an address change --
// staging moves a weight every graph without changing what it is -- but it says nothing about
// the bytes, and this file rewrites those bytes in place under an unchanged name: the LoRA
// fold merges the delta into the params copy, and the unfold restores the pristine file bytes
// with MADV_DONTNEED. Neither is visible to ggml (the unfold is a madvise() on a mapping; it
// makes no ggml call at all), so ggml is told, once, how to ASK.
//
// Process-global rather than per-ModelManager on purpose: the caches it guards are themselves
// process-global, and the provider is registered from a static initialiser, so there is no
// lifetime to get wrong and nothing to unregister when a context goes away. Two managers
// folding at once merely invalidate each other's cached scalars, which costs a recompute and
// never correctness.
//
// Bumped at exactly the three places this file can change what a weight's bytes ARE. The GPU
// fold ALSO self-invalidates inside ggml (ggml_cuda_lora_fold_* calls
// ggml_cuda_weight_content_bump), so the fold bump here is what covers SD_LORA_FOLD_CPU=1.
namespace {
std::atomic<uint64_t> g_weight_content_epoch{0};

void bump_weight_content_epoch(const char* why) {
    const uint64_t e = g_weight_content_epoch.fetch_add(1, std::memory_order_acq_rel) + 1;
    LOG_DEBUG("weight-content epoch -> %llu (%s)", (unsigned long long)e, why);
}

#ifdef SD_USE_CUDA
uint64_t weight_content_epoch_provider(void*) {
    return g_weight_content_epoch.load(std::memory_order_acquire);
}

struct WeightContentEpochRegistrar {
    WeightContentEpochRegistrar() {
        ggml_cuda_set_weight_content_epoch_provider(weight_content_epoch_provider, nullptr);
    }
};
const WeightContentEpochRegistrar g_weight_content_epoch_registrar;
#endif
}  // namespace

static size_t aligned_offset(const void* buffer, size_t offset, size_t alignment) {
    GGML_ASSERT(alignment != 0 && (alignment & (alignment - 1)) == 0);
    size_t align = (alignment - ((reinterpret_cast<uintptr_t>(buffer) + offset) % alignment)) % alignment;
    return offset + align;
}

static bool lora_specs_equal(const std::vector<ModelManager::LoraSpec>& lhs,
                             const std::vector<ModelManager::LoraSpec>& rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i].path != rhs[i].path ||
            lhs[i].multiplier != rhs[i].multiplier ||
            lhs[i].is_high_noise != rhs[i].is_high_noise ||
            lhs[i].tensor_name_prefix_filter != rhs[i].tensor_name_prefix_filter ||
            lhs[i].required != rhs[i].required) {
            return false;
        }
    }
    return true;
}

static std::string lora_id(const ModelManager::LoraSpec& lora) {
    return lora.is_high_noise ? "|high_noise|" + lora.path : lora.path;
}

static bool backend_supports_host_buffer(ggml_backend_t backend) {
    if (backend == nullptr) {
        return false;
    }
    if (sd_backend_is_cpu(backend)) {
        return true;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev == nullptr) {
        return false;
    }
    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(dev, &props);
    return props.caps.buffer_from_host_ptr;
}

// Fold-at-load is OFF by default: it is a numerics change (see the header comment in
// model/adapter/lora_fold.hpp -- stochastic rounding trades ~13% more weight noise for
// getting the delta applied at all) and has NOT been quality-checked on a GPU render.
// SD_LORA_FOLD=1 opts in; SD_LORA_FOLD=0 or unset keeps the per-step adapter branch.
static bool lora_fold_at_load_enabled() {
    const char* value = std::getenv("SD_LORA_FOLD");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

// Break copy-on-write on a range we are about to have the GPU DMA into, ahead of time.
//
// WHY. The fold's write-back is a D2H into the mmap'd model, and those pages have never
// been written, so essentially every one of them takes a COW fault DURING the copy, one at
// a time, with the DMA engine waiting. MEASURED on this card: a D2H into already-faulted
// pages runs 11.49 GB/s, into fresh ones 2.12 GB/s -- a 5.4x penalty that accounts for the
// fold's 5.5 ms/tensor download exactly.
//
// This is NOT the pinned-staging idea, which was tried and failed. Pinning relocated the
// fault (DMA into pinned memory, then a memcpy that faults) rather than removing it, and
// measured as noise. Nor is it LONGCAT_DIT_NO_MMAP=1, which swaps COW faults for
// first-touch anonymous faults -- also still faults, also no change.
//
// MADV_POPULATE_WRITE does the whole range in one syscall, in the kernel, without a trap
// per page. The manual touch loop is the fallback for kernels/mappings that reject it.
static void prefault_for_write(void* addr, size_t bytes) {
    if (addr == nullptr || bytes == 0) {
        return;
    }
    const size_t page   = (size_t)sysconf(_SC_PAGESIZE);
    const uintptr_t beg = (uintptr_t)addr & ~(uintptr_t)(page - 1);
    const uintptr_t end = ((uintptr_t)addr + bytes + page - 1) & ~(uintptr_t)(page - 1);
#ifdef MADV_POPULATE_WRITE
    if (madvise((void*)beg, (size_t)(end - beg), MADV_POPULATE_WRITE) == 0) {
        return;
    }
#endif
    for (uintptr_t p = beg; p < end; p += page) {
        volatile char* c = (volatile char*)p;
        *c               = *c;  // read-modify-write: breaks COW without changing the byte
    }
}

static bool model_manager_profile_enabled() {
    const char* value = std::getenv("SD_MODEL_MANAGER_PROFILE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static size_t model_manager_resident_headroom_bytes() {
    const char* value = std::getenv("LONGCAT_SHARED_RESIDENT_HEADROOM_MB");
    const double mb = value != nullptr && value[0] != '\0' ? std::atof(value) : 512.0;
    return static_cast<size_t>(std::max(0.0, mb) * 1024.0 * 1024.0);
}

ModelManager::~ModelManager() {
    release_all();
}

void ModelManager::set_common_ignore_tensors(std::set<std::string> ignore_tensors) {
    common_ignore_tensors_ = std::move(ignore_tensors);
}

void ModelManager::set_loras(std::vector<LoraSpec> loras, SDVersion version) {
    if (loras.empty() && loras_.empty()) {
        lora_version_ = version;
        return;
    }
    if (lora_version_ == version && lora_specs_equal(loras_, loras)) {
        return;
    }

    loras_        = std::move(loras);
    lora_version_ = version;
    current_lora_epoch_++;
    // The adapter set changed, so every weight a fold/apply touches is about to mean something
    // different -- including the case where it means PRISTINE again (an empty set after a
    // folded render). Bumped here, before any fold runs, so nothing cached under the outgoing
    // set can be served under the incoming one.
    bump_weight_content_epoch("lora set changed");
    reset_lora_applied_params();
}

std::set<std::string> ModelManager::tensor_names() const {
    std::set<std::string> names;
    for (const auto& state : tensor_states_) {
        if (state != nullptr) {
            names.insert(state->name);
        }
    }
    return names;
}

ggml_tensor* ModelManager::find_tensor(const std::string& name) const {
    auto it = tensor_states_by_name_.find(name);
    if (it == tensor_states_by_name_.end() || it->second == nullptr) {
        return nullptr;
    }
    return it->second->tensor;
}

size_t estimate_tensors_size(const std::map<std::string, ggml_tensor*>& tensors) {
    size_t size = 0;
    std::unordered_set<ggml_tensor*> seen;
    for (const auto& pair : tensors) {
        ggml_tensor* tensor = pair.second;
        if (tensor == nullptr || seen.find(tensor) != seen.end()) {
            continue;
        }
        seen.insert(tensor);
        size += ggml_nbytes(tensor);
    }
    return size;
}

void ModelManager::set_split_buffer_type(ggml_backend_t compute_backend, ggml_backend_buffer_type_t split_buft) {
    if (compute_backend == nullptr) {
        return;
    }
    if (split_buft == nullptr) {
        split_buffer_types_.erase(compute_backend);
        return;
    }
    split_buffer_types_[compute_backend] = split_buft;
}

bool ModelManager::tensor_shape_supports_split_buffer(const ggml_tensor* tensor) {
    return tensor != nullptr &&
           tensor->view_src == nullptr &&
           ggml_is_contiguous(tensor) &&
           ggml_n_dims(tensor) == 2 &&
           tensor->ne[0] >= 256 &&
           tensor->ne[1] >= 256;
}

ggml_backend_buffer_type_t ModelManager::split_buffer_type_for(const TensorState& state) const {
    if (!state.allow_split_buffer || !tensor_shape_supports_split_buffer(state.tensor)) {
        return nullptr;
    }
    auto it = split_buffer_types_.find(state.compute_backend);
    return it != split_buffer_types_.end() ? it->second : nullptr;
}

bool ModelManager::register_param_tensors(const std::string& desc,
                                          std::map<std::string, ggml_tensor*> tensors,
                                          ResidencyMode residency_mode,
                                          ggml_backend_t compute_backend,
                                          ggml_backend_t params_backend,
                                          size_t* registered_tensor_size,
                                          bool allow_split_buffer,
                                          bool params_follow_compute_backend) {
    if (desc.empty()) {
        LOG_ERROR("model manager tensor desc is empty");
        return false;
    }
    if (registered_tensor_size != nullptr) {
        *registered_tensor_size += estimate_tensors_size(tensors);
    }

    std::vector<std::unique_ptr<TensorState>> new_states;
    new_states.reserve(tensors.size());

    for (const auto& pair : tensors) {
        const std::string& name = pair.first;
        ggml_tensor* tensor     = pair.second;
        if (tensor == nullptr) {
            continue;
        }
        if (tensor_states_by_name_.find(name) != tensor_states_by_name_.end()) {
            LOG_ERROR("model manager tensor name '%s' is already registered", name.c_str());
            return false;
        }
        ggml_set_name(tensor, name.c_str());

        auto state                           = std::make_unique<TensorState>();
        state->name                          = name;
        state->tensor                        = tensor;
        state->desc                          = desc;
        state->residency_mode                = residency_mode;
        state->compute_backend               = compute_backend;
        state->params_backend                = params_backend;
        state->allow_split_buffer            = allow_split_buffer;
        state->params_follow_compute_backend = params_follow_compute_backend;
        new_states.push_back(std::move(state));
    }

    for (auto& state : new_states) {
        TensorState* registered_state                  = state.get();
        tensor_states_by_name_[registered_state->name] = registered_state;
        tensor_states_.push_back(std::move(state));
    }
    return true;
}

bool ModelManager::unregister_param_tensors(const std::string& desc, size_t* registered_tensor_size) {
    if (desc.empty()) {
        return true;
    }

    std::unordered_set<TensorState*> target_states;
    size_t released_size = 0;
    for (auto& state : tensor_states_) {
        if (state == nullptr || state->desc != desc) {
            continue;
        }
        if (state->active_prepare_count > 0) {
            LOG_ERROR("model manager cannot unregister active %s tensor '%s'",
                      desc.c_str(),
                      state->name.c_str());
            return false;
        }
        target_states.insert(state.get());
        if (state->tensor != nullptr) {
            released_size += ggml_nbytes(state->tensor);
        }
    }

    if (target_states.empty()) {
        return true;
    }

    release_compute_staging_blocks(false);

    std::vector<ParamsStorageBlock*> storage_blocks_to_release;
    std::unordered_set<TensorState*> affected_storage_states;
    for (const auto& block : params_storage_blocks_) {
        if (block == nullptr) {
            continue;
        }
        bool has_target_state = false;
        for (TensorState* state : block->states) {
            if (state != nullptr && target_states.count(state) > 0) {
                has_target_state = true;
                break;
            }
        }
        if (!has_target_state) {
            continue;
        }
        storage_blocks_to_release.push_back(block.get());
        for (TensorState* state : block->states) {
            if (state != nullptr) {
                affected_storage_states.insert(state);
            }
        }
    }

    for (TensorState* state : affected_storage_states) {
        if (state == nullptr) {
            continue;
        }
        if (state->active_prepare_count > 0 || state->staged_to_compute_backend) {
            LOG_ERROR("model manager cannot unregister %s while tensor '%s' is active",
                      desc.c_str(),
                      state->name.c_str());
            return false;
        }
    }

    for (ParamsStorageBlock* block : storage_blocks_to_release) {
        if (block != nullptr) {
            free_params_storage_block(*block);
            erase_params_storage_block(block);
        }
    }

    for (auto it = tensor_states_by_name_.begin(); it != tensor_states_by_name_.end();) {
        if (target_states.count(it->second) > 0) {
            it = tensor_states_by_name_.erase(it);
        } else {
            ++it;
        }
    }
    tensor_states_.erase(std::remove_if(tensor_states_.begin(),
                                        tensor_states_.end(),
                                        [&](const std::unique_ptr<TensorState>& s) {
                                            return s == nullptr || target_states.count(s.get()) > 0;
                                        }),
                         tensor_states_.end());

    if (registered_tensor_size != nullptr) {
        if (released_size > *registered_tensor_size) {
            *registered_tensor_size = 0;
        } else {
            *registered_tensor_size -= released_size;
        }
    }
    return true;
}

bool ModelManager::load_all_params_eagerly() {
    std::vector<TensorState*> all_states;
    all_states.reserve(tensor_states_.size());
    for (const auto& s : tensor_states_) {
        if (s != nullptr) {
            all_states.push_back(s.get());
        }
    }
    return load_tensors_to_params_backend(all_states);
}

void ModelManager::reclaim_transient_compute_buffers() {
    // A caller uses this only at a serialized request/window boundary, after
    // every participating runner has completed.  Force-release avoids a stale
    // bookkeeping reference pinning a temporary GPU staging block across the
    // next video window; host/mmap parameter storage is intentionally kept.
    release_compute_staging_blocks(true);
}

bool ModelManager::validate_registered_tensors() {
    bool ok = true;
    for (const auto& state : tensor_states_) {
        if (state == nullptr) {
            ok = false;
            continue;
        }
        bool state_ok = validate_tensor(*state);
        if (state_ok) {
            state->metadata_validated = true;
        }
        ok = state_ok && ok;
    }
    return ok;
}

bool ModelManager::load_tensors_to_params_backend(const std::vector<TensorState*>& states) {
    std::vector<TensorState*> need_load;
    need_load.reserve(states.size());
    for (TensorState* state : states) {
        if (state == nullptr || should_ignore(*state) || is_optional_missing_tensor(state->name)) {
            continue;
        }
        if (!state->metadata_validated) {
            if (!validate_tensor(*state)) {
                return false;
            }
            state->metadata_validated = true;
        }
        if (!state->loaded_to_params_backend) {
            need_load.push_back(state);
        }
    }
    if (need_load.empty()) {
        return true;
    }

    std::vector<ParamsStorageBlock*> created_storage_blocks;
    if (!mmap_params(need_load, created_storage_blocks)) {
        for (ParamsStorageBlock* block : created_storage_blocks) {
            if (block != nullptr) {
                free_params_storage_block(*block);
                erase_params_storage_block(block);
            }
        }
        return false;
    }

    std::vector<TensorState*> need_alloc;
    need_alloc.reserve(need_load.size());
    for (TensorState* state : need_load) {
        if (state->tensor != nullptr && state->tensor->data == nullptr && state->tensor->view_src == nullptr) {
            need_alloc.push_back(state);
        }
    }

    if (!alloc_params_buffers(need_alloc, created_storage_blocks) ||
        !load_tensors(need_load)) {
        for (ParamsStorageBlock* block : created_storage_blocks) {
            if (block != nullptr) {
                free_params_storage_block(*block);
                erase_params_storage_block(block);
            }
        }
        return false;
    }
    for (ParamsStorageBlock* block : created_storage_blocks) {
        if (block != nullptr && block->buffer != nullptr) {
            LOG_DEBUG("model manager prepared params backend buffer (%6.2f MB, %zu tensors, %s)",
                      ggml_backend_buffer_get_size(block->buffer) / (1024.f * 1024.f),
                      block->states.size(),
                      ggml_backend_buffer_is_host(block->buffer) ? "RAM" : "VRAM");
        }
    }

    return true;
}

bool ModelManager::stage_tensors_to_compute_backend(const std::vector<TensorState*>& states) {
    std::map<std::pair<ggml_backend_t, ggml_backend_buffer_type_t>, std::vector<TensorState*>> states_by_staging_target;
    for (TensorState* state : states) {
        if (state == nullptr || should_ignore(*state) || is_optional_missing_tensor(state->name)) {
            continue;
        }
        if (state->compute_backend == nullptr) {
            LOG_ERROR("model manager compute backend is null for tensor '%s'", state->name.c_str());
            return false;
        }
        if (state->params_backend == nullptr) {
            LOG_ERROR("model manager params backend is null for tensor '%s'", state->name.c_str());
            return false;
        }
        if (state->compute_backend == state->params_backend || state->staged_to_compute_backend) {
            continue;
        }
        if (!state->loaded_to_params_backend || state->tensor == nullptr || state->tensor->data == nullptr) {
            LOG_ERROR("model manager tensor '%s' is not loaded to params backend", state->name.c_str());
            return false;
        }
        ggml_backend_buffer_type_t staging_buft = split_buffer_type_for(*state);
        if (staging_buft == nullptr) {
            staging_buft = ggml_backend_get_default_buffer_type(state->compute_backend);
        }
        states_by_staging_target[{state->compute_backend, staging_buft}].push_back(state);
    }

    for (const auto& pair : states_by_staging_target) {
        ggml_backend_t compute_backend          = pair.first.first;
        ggml_backend_buffer_type_t staging_buft = pair.first.second;
        const std::vector<TensorState*>& states = pair.second;
        if (states.empty()) {
            continue;
        }

        const bool profile = model_manager_profile_enabled();
        int64_t t0 = ggml_time_ms();

        ggml_init_params init_params;
        init_params.mem_size   = std::max<size_t>(1, states.size()) * ggml_tensor_overhead();
        init_params.mem_buffer = nullptr;
        init_params.no_alloc   = true;

        ggml_context* staging_ctx = ggml_init(init_params);
        GGML_ASSERT(staging_ctx != nullptr);

        std::vector<std::pair<TensorState*, ggml_tensor*>> staged_tensors;
        staged_tensors.reserve(states.size());
        for (TensorState* state : states) {
            ggml_tensor* staging_tensor = ggml_dup_tensor(staging_ctx, state->tensor);
            ggml_set_name(staging_tensor, state->tensor->name);
            staged_tensors.push_back({state, staging_tensor});
        }

        ggml_backend_buffer_t compute_buffer = ggml_backend_alloc_ctx_tensors_from_buft(staging_ctx, staging_buft);
        if (compute_buffer == nullptr) {
            LOG_ERROR("model manager alloc compute params backend buffer failed, num_tensors = %zu",
                      staged_tensors.size());
            ggml_free(staging_ctx);
            return false;
        }
        ggml_backend_buffer_set_usage(compute_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        int64_t t_alloc = ggml_time_ms();

        for (auto& staged_tensor : staged_tensors) {
            TensorState* state          = staged_tensor.first;
            ggml_tensor* managed_tensor = state->tensor;
            ggml_tensor* staging_tensor = staged_tensor.second;
            ggml_backend_tensor_copy(managed_tensor, staging_tensor);
            std::swap(managed_tensor->buffer, staging_tensor->buffer);
            std::swap(managed_tensor->data, staging_tensor->data);
            std::swap(managed_tensor->extra, staging_tensor->extra);
        }
        int64_t t_enqueue = ggml_time_ms();
        ggml_backend_synchronize(compute_backend);
        int64_t t_sync = ggml_time_ms();

        auto block             = std::make_unique<ComputeStagingBlock>();
        block->compute_backend = compute_backend;
        block->buffer          = compute_buffer;
        block->staging_ctx     = staging_ctx;
        block->staged_tensors  = std::move(staged_tensors);
        for (auto& staged_tensor : block->staged_tensors) {
            TensorState* state               = staged_tensor.first;
            state->staged_to_compute_backend = true;
        }
        compute_staging_blocks_.push_back(std::move(block));

        int64_t t1 = ggml_time_ms();
        LOG_DEBUG("model manager staged compute params (%6.2f MB, %zu tensors) to %s, taking %.2fs",
                  ggml_backend_buffer_get_size(compute_buffer) / (1024.f * 1024.f),
                  states.size(),
                  ggml_backend_name(compute_backend),
                  (t1 - t0) * 1.0f / 1000);
        if (profile) {
            size_t free_bytes = 0;
            size_t total_bytes = 0;
            if (ggml_backend_dev_t device = ggml_backend_get_device(compute_backend); device != nullptr) {
                ggml_backend_dev_memory(device, &free_bytes, &total_bytes);
            }
            LOG_INFO("[MM_PROFILE] stage %.2f MiB / %zu tensors: alloc=%lld ms enqueue=%lld ms sync=%lld ms total=%lld ms",
                     ggml_backend_buffer_get_size(compute_buffer) / (1024.f * 1024.f),
                     states.size(),
                     (long long) (t_alloc - t0),
                     (long long) (t_enqueue - t_alloc),
                     (long long) (t_sync - t_enqueue),
                     (long long) (t_sync - t0));
            LOG_INFO("[MM_PROFILE] post-stage VRAM: free=%.2f MiB total=%.2f MiB",
                     free_bytes / (1024.0 * 1024.0), total_bytes / (1024.0 * 1024.0));
        }
    }

    return true;
}

// Undo fold_loras_into_params(). Not by subtracting the delta back out -- that is not
// bit-exact in the target dtype and would drift with every adapter change -- but by
// throwing away the process-private copy-on-write pages the fold wrote, so the weights
// re-read from the model file on next touch. MAP_PRIVATE means the file was never
// written, so it is still the pristine checkpoint.
//
// Tensors whose params block OWNS its bytes (alloc'd buffer, not mmap) need nothing here:
// release_params_storage_blocks() frees the buffer and the next load_tensors() genuinely
// re-reads them. restore_mmapped_bytes() returns false for those and they are not counted.
void ModelManager::unfold_loras_from_params() {
    size_t restored = 0, failed = 0;
    const int64_t t0 = ggml_time_ms();
    for (auto& state : tensor_states_) {
        if (state == nullptr || state->folded_lora_epoch == UINT64_MAX) {
            continue;
        }
        ggml_tensor* tensor = state->tensor;
        if (tensor == nullptr || tensor->data == nullptr) {
            // Already released without being restored -- the mutated pages are still live
            // in the mapping and we can no longer locate them. Only reachable if some
            // other path force-released the block first; loud, because the next render
            // would silently inherit the merge.
            LOG_ERROR("lora unfold: '%s' was folded but its params copy is already released",
                      state->name.c_str());
            failed++;
            continue;
        }
        if (tensor->buffer != nullptr && !ggml_backend_buffer_is_host(tensor->buffer)) {
            continue;  // device memory: freeing the buffer really does discard it
        }
        if (model_loader_.restore_mmapped_bytes(tensor->data, ggml_nbytes(tensor))) {
            restored++;
        }
    }
    if (restored > 0 || failed > 0) {
        LOG_INFO("lora unfold: restored %zu tensor(s) from the model file in %.2fs (%zu failed)",
                 restored, (ggml_time_ms() - t0) / 1000.0, failed);
    }
    if (restored > 0) {
        // madvise() is invisible to ggml: the pages now hold the pristine checkpoint under the
        // same names and addresses the folded bytes had. Nothing downstream can notice unless
        // it is told.
        bump_weight_content_epoch("lora unfold");
    }
}

void ModelManager::release_fold_loras() {
    for (auto& lora : fold_loras_) {
        if (lora != nullptr) {
            lora->release_loaded_tensors();
        }
    }
    fold_loras_.clear();
    if (fold_cpu_backend_ != nullptr) {
        ggml_backend_free(fold_cpu_backend_);
        fold_cpu_backend_ = nullptr;
    }
    fold_loras_epoch_ = UINT64_MAX;
    fold_loras_failed_ = false;
}

// Dequantise a whole host-resident LoRA tensor to f32. LoRA ggufs here are Q8_0 (see
// tools/convert_lora_q8.py), so this is not optional.
static bool lora_tensor_to_f32(const ggml_tensor* t, std::vector<float>& out) {
    if (t == nullptr || t->data == nullptr) {
        return false;
    }
    const int64_t n = ggml_nelements(t);
    out.resize((size_t)n);
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(out.data(), t->data, sizeof(float) * (size_t)n);
        return true;
    }
    const ggml_type_traits* traits = ggml_get_type_traits(t->type);
    if (traits == nullptr || traits->to_float == nullptr) {
        return false;
    }
    // Rows must be whole blocks for to_float; LoRA tensors are 2-D and contiguous.
    const int64_t row = t->ne[0];
    if (row % ggml_blck_size(t->type) != 0) {
        return false;
    }
    const int64_t nrows   = n / row;
    const size_t row_size = ggml_row_size(t->type, row);
    for (int64_t r = 0; r < nrows; ++r) {
        traits->to_float((const char*)t->data + (size_t)r * row_size, out.data() + (size_t)r * row, row);
    }
    return true;
}

static float lora_scalar(const ggml_tensor* t) {
    if (t == nullptr || t->data == nullptr) {
        return 0.0f;
    }
    switch (t->type) {
        case GGML_TYPE_F32:
            return *(const float*)t->data;
        case GGML_TYPE_F16:
            return ggml_fp16_to_fp32(*(const ggml_fp16_t*)t->data);
        case GGML_TYPE_BF16:
            return ggml_bf16_to_fp32(*(const ggml_bf16_t*)t->data);
        default: {
            std::vector<float> v;
            return lora_tensor_to_f32(t, v) && !v.empty() ? v[0] : 0.0f;
        }
    }
}

// Merge the LoRA delta into the PARAMS-backend copy of every candidate weight, once per
// LoRA epoch. Runs BEFORE staging, which is the whole point: a compute staging block is
// rebuilt every graph, so anything written there is thrown away under weight offload.
bool ModelManager::fold_loras_into_params(const std::vector<TensorState*>& states) {
    if (loras_.empty() || !lora_fold_at_load_enabled() || fold_loras_failed_) {
        return true;
    }

    std::vector<TensorState*> candidates;
    for (TensorState* state : states) {
        if (state == nullptr || state->tensor == nullptr || state->lora_fold_declined ||
            should_ignore(*state) || is_optional_missing_tensor(state->name)) {
            continue;
        }
        if (state->folded_lora_epoch == current_lora_epoch_ || !state->loaded_to_params_backend) {
            continue;
        }
        if (state->desc == "LoRA") {
            continue;  // the adapters themselves are never fold targets
        }
        if (state->staged_to_compute_backend) {
            // Its params copy is not the live one right now; folding it would be silently
            // discarded. Leave the epoch unset so it is folded once it is unstaged.
            continue;
        }
        // Deliberately NOT pre-filtered on rank/type here. `lora_fold_declined` means "hand
        // this one to the graph apply path", and that path reloads the whole adapter onto the
        // COMPUTE backend for any group it is given -- so declining every bias and norm up
        // front would drag a 1.3 GB LoRA back onto the GPU on every prepare_params, which is
        // exactly the per-graph cost this fold exists to remove. A tensor is only declined
        // below, once a LoRA is known to actually target it.
        candidates.push_back(state);
    }
    if (candidates.empty()) {
        return true;
    }

    if (enable_mmap_ && !writable_mmap_) {
        LOG_ERROR("lora fold-at-load needs a writable mmap; refusing to fold read-only weights");
        fold_loras_failed_ = true;
        return false;
    }
    // Fail closed: an unfoldable fold is worse than no fold. Without page discard, the
    // merge survives set_loras({}) and every later render on this worker silently
    // inherits it -- and STACKS, adapter on adapter. That was live for two days.
    if (enable_mmap_ && !MmapWrapper::supports_discard_private_writes()) {
        LOG_ERROR("lora fold-at-load cannot be undone on this platform (no private-page discard); refusing to fold");
        fold_loras_failed_ = true;
        return false;
    }

    if (fold_loras_epoch_ != current_lora_epoch_) {
        release_fold_loras();
        fold_cpu_backend_ = sd_backend_cpu_init();
        if (fold_cpu_backend_ == nullptr) {
            LOG_ERROR("lora fold-at-load could not create a CPU backend");
            fold_loras_failed_ = true;
            return false;
        }
        if (n_threads_ > 0) {
            sd_backend_cpu_set_n_threads(fold_cpu_backend_, n_threads_);
        }
        const std::set<std::string> all_names = tensor_names();
        for (const LoraSpec& spec : loras_) {
            auto lora = std::make_shared<LoraModel>(lora_id(spec),
                                                    fold_cpu_backend_,
                                                    fold_cpu_backend_,
                                                    spec.path,
                                                    spec.is_high_noise ? "model.high_noise_" : "",
                                                    lora_version_);
            LoraModel::filter_t filter = nullptr;
            if (!spec.tensor_name_prefix_filter.empty()) {
                filter = [&spec](const std::string& tensor_name) {
                    return starts_with(tensor_name, spec.tensor_name_prefix_filter);
                };
            }
            if (!lora->load_from_file(n_threads_, filter) || lora->lora_tensors.empty()) {
                LOG_WARN("lora fold-at-load could not read %s", spec.path.c_str());
                if (spec.required) {
                    fold_loras_failed_ = true;
                    return false;
                }
                continue;
            }
            lora->preprocess_lora_tensors(all_names);
            lora->multiplier = spec.multiplier;
            fold_loras_.push_back(std::move(lora));
        }
        fold_loras_epoch_ = current_lora_epoch_;
        LOG_INFO("lora fold-at-load: %zu adapter(s) resident on CPU for epoch %llu",
                 fold_loras_.size(), (unsigned long long)current_lora_epoch_);
    }
    if (fold_loras_.empty()) {
        for (TensorState* state : candidates) {
            state->folded_lora_epoch = current_lora_epoch_;
        }
        return true;
    }

    const int64_t t0 = ggml_time_ms();
    size_t folded = 0, declined = 0, gpu_folded = 0;
    // Prefer the GPU fold; SD_LORA_FOLD_CPU=1 forces the CPU path (A/B and fallback test).
    const char* cpu_env = std::getenv("SD_LORA_FOLD_CPU");
    const bool use_gpu_fold = !(cpu_env != nullptr && cpu_env[0] != '\0' && cpu_env[0] != '0');

    // Fault in every page the fold is about to write, in parallel, before writing any of
    // them. See prefault_for_write: the download leg is 5.4x slower into never-written
    // pages, and it is ~10 GB of them. Doing it here costs one bulk populate per tensor
    // instead of ~2400 in-DMA faults per tensor, and it also pre-warms the SOURCE side of
    // the upload leg, since that reads the same pages.
    //
    // Only tensors a LoRA actually targets: `candidates` deliberately contains every weight
    // that has not been folded yet, most of which nothing targets, and COW-breaking those
    // would fault in the whole model for no reason.
    {
        std::vector<TensorState*> targets;
        targets.reserve(candidates.size());
        for (TensorState* state : candidates) {
            if (state->tensor == nullptr || state->tensor->data == nullptr) {
                continue;
            }
            if (state->tensor->buffer != nullptr && !ggml_backend_buffer_is_host(state->tensor->buffer)) {
                continue;  // device memory: no host pages to fault
            }
            for (auto& lora : fold_loras_) {
                if (lora->lora_tensors.count("lora." + state->name + ".lora_down") > 0) {
                    targets.push_back(state);
                    break;
                }
            }
        }
        if (!targets.empty()) {
            const int64_t t_pf   = ggml_time_ms();
            const int nthreads   = std::max(1, std::min(n_threads_ > 0 ? n_threads_ : 8, (int)targets.size()));
            std::atomic<size_t> next{0};
            auto worker = [&]() {
                for (size_t i = next++; i < targets.size(); i = next++) {
                    prefault_for_write(targets[i]->tensor->data, ggml_nbytes(targets[i]->tensor));
                }
            };
            std::vector<std::thread> pool;
            pool.reserve((size_t)nthreads - 1);
            for (int t = 1; t < nthreads; ++t) {
                pool.emplace_back(worker);
            }
            worker();
            for (std::thread& th : pool) {
                th.join();
            }
            LOG_DEBUG("lora fold-at-load: pre-faulted %zu target(s) on %d threads, taking %.2fs",
                      targets.size(), nthreads, (ggml_time_ms() - t_pf) * 1.0f / 1000);
        }
    }
    auto in_of  = [](const ggml_tensor* t) { return (int64_t)t->ne[0]; };
    auto out_of = [](const ggml_tensor* t) { return (int64_t)t->ne[1]; };
    (void)use_gpu_fold; (void)in_of; (void)out_of;
    std::vector<float> host_scratch;

    // Host-side phase timers. The CUDA profiler only ever saw its own phases, which summed
    // to half the pass -- the other half was invisible and was assumed to be transfers. It
    // was the CPU fold. Do not remove these: the split between "building the delta" and
    // "merging it" is the only thing that says which half to attack next.
    int64_t us_delta = 0, us_gpu = 0, us_cpu = 0;
    size_t cpu_folded = 0;

    // SD_LORA_FOLD_VERIFY=1: fold every DENSE tensor BOTH ways and report the difference.
    //
    // Moving the dense targets from the CPU fold to a CUDA kernel is the only arithmetic
    // change in this pass, and "the render still looked right" cannot distinguish a correct
    // kernel from one that is subtly wrong on a minority of tensors. These two paths SHOULD
    // agree to within float rounding -- unlike the NVFP4 pair, which cannot be compared this
    // way because stochastic rounding turns a 1 ULP GEMM difference into a flipped nibble.
    // SD_LORA_FOLD_DENSE=0 -> dense targets go back to the CPU fold (isolating control).
    const char* dense_env = std::getenv("SD_LORA_FOLD_DENSE");
    const bool dense_gpu_enabled = dense_env == nullptr || dense_env[0] == '\0' || dense_env[0] != '0';
    // Only read inside the SD_USE_CUDA block below; keep a CPU-only build warning-free.
    (void)dense_gpu_enabled;

    const char* verify_env = std::getenv("SD_LORA_FOLD_VERIFY");
    const bool verify_dense = verify_env != nullptr && verify_env[0] != '\0' && verify_env[0] != '0';
    std::vector<uint8_t> verify_pre;
    double verify_max_abs = 0.0, verify_max_rel = 0.0, verify_sum_abs = 0.0;
    size_t verify_n = 0, verify_elems = 0, verify_differing = 0;
    int64_t verify_max_ulps = 0;

    // ---- pre-pass: load every `.wglobal` sidecar the fold will need ------------------
    // load_tensors_to_params_backend() mutates shared loader and storage-block state, so it
    // cannot run inside the threaded build below. Hoisting it here is precisely what makes
    // that build parallelisable; leaving it in the per-tensor path would be a data race
    // that only shows up under load.
    {
        std::vector<TensorState*> wg_needed;
        for (TensorState* state : candidates) {
            if (state->tensor == nullptr || state->tensor->type != GGML_TYPE_NVFP4) {
                continue;
            }
            auto it = tensor_states_by_name_.find(state->name + ".wglobal");
            if (it != tensor_states_by_name_.end() && it->second != nullptr &&
                !it->second->loaded_to_params_backend) {
                wg_needed.push_back(it->second);
            }
        }
        if (!wg_needed.empty()) {
            load_tensors_to_params_backend(wg_needed);
        }
    }

    // One tensor's delta inputs. The f32 buffers live in `owned` and the ModuleDelta
    // pointers point into them, so a FoldWork must outlive the merge that consumes it.
    // Buffers are reused across chunks (a `used` cursor, not clear()) because handing
    // megabyte blocks back to the allocator per tensor means the next tensor takes a page
    // fault on every byte it writes into their replacements.
    struct FoldWork {
        TensorState* state = nullptr;
        std::vector<sd_lora_fold::ModuleDelta> deltas;
        std::deque<std::vector<float>> owned;  // deque: element addresses stay stable
        size_t used     = 0;
        float wglobal   = 1.0f;
        bool declined   = false;
        bool no_wglobal = false;
        std::vector<float>& next_buf() {
            if (used == owned.size()) {
                owned.emplace_back();
            }
            return owned[used++];
        }
        void reset(TensorState* s) {
            state      = s;
            deltas.clear();
            used       = 0;
            wglobal    = 1.0f;
            declined   = false;
            no_wglobal = false;
        }
    };

    // Build one tensor's deltas. MUST stay free of shared mutable state: it runs on
    // `build_threads` threads at once. Reading fold_loras_, tensor_states_by_name_ and
    // already-loaded tensor data is fine; anything that loads, logs or counts is not, so
    // those are deferred to the sequential merge below via the flags on FoldWork.
    auto build_one = [&](FoldWork& work) {
        TensorState* state = work.state;
        ggml_tensor* w     = state->tensor;
        for (auto& lora : fold_loras_) {
            int64_t row_begin = 0;
            for (int index = 0;; ++index) {
                const std::string key = index == 0 ? state->name : state->name + "." + std::to_string(index);
                auto down_it = lora->lora_tensors.find("lora." + key + ".lora_down");
                auto up_it   = lora->lora_tensors.find("lora." + key + ".lora_up");
                if (down_it == lora->lora_tensors.end() || up_it == lora->lora_tensors.end()) {
                    break;
                }
                ggml_tensor* down = down_it->second;
                ggml_tensor* up   = up_it->second;
                if (ggml_n_dims(down) != 2 || ggml_n_dims(up) != 2) {
                    work.declined = true;
                    break;
                }
                const int64_t rank = down->ne[1];

                float scale = 1.0f;
                auto scale_it = lora->lora_tensors.find("lora." + key + ".scale");
                if (scale_it != lora->lora_tensors.end()) {
                    scale = lora_scalar(scale_it->second);
                } else {
                    auto alpha_it = lora->lora_tensors.find("lora." + key + ".alpha");
                    if (alpha_it != lora->lora_tensors.end() && rank > 0) {
                        scale = lora_scalar(alpha_it->second) / (float)rank;
                    }
                }
                scale *= lora->multiplier;

                std::vector<float>& down_f32 = work.next_buf();
                std::vector<float>& up_f32   = work.next_buf();
                if (!lora_tensor_to_f32(down, down_f32) || !lora_tensor_to_f32(up, up_f32)) {
                    work.declined = true;
                    break;
                }
                sd_lora_fold::ModuleDelta d;
                d.down      = down_f32.data();
                d.up        = up_f32.data();
                d.in        = down->ne[0];
                d.rank      = rank;
                d.rows      = up->ne[1];
                d.row_begin = row_begin;
                d.scale     = scale;

                auto mid_it = lora->lora_tensors.find("lora." + key + ".lora_mid");
                if (mid_it != lora->lora_tensors.end()) {
                    std::vector<float>& mid_f32 = work.next_buf();
                    if (!lora_tensor_to_f32(mid_it->second, mid_f32)) {
                        work.declined = true;
                        break;
                    }
                    d.mid = mid_f32.data();
                }
                work.deltas.push_back(d);
                row_begin += d.rows;
            }
            if (work.declined) {
                break;
            }
        }
        if (work.declined || work.deltas.empty() || w->type != GGML_TYPE_NVFP4) {
            return;
        }
        // The runtime multiplies this Linear's output by `.wglobal` (or folds it into the
        // cuBLASLt alpha), so the stored nibbles live in a scaled domain and the TRUE-unit
        // delta has to be divided by it before it can be merged. The sidecar was loaded by
        // the pre-pass above; this only reads it.
        auto wg_it = tensor_states_by_name_.find(state->name + ".wglobal");
        if (wg_it == tensor_states_by_name_.end()) {
            // FLAT nvfp4 build (krea2 ships one): there is no per-tensor global scale
            // ANYWHERE in the model, so the stored nibbles are already in true units and
            // wglobal is exactly 1.0 -- there is nothing for the runtime to multiply by.
            // Declining here was not conservative, it was fatal: SD_LORA_FOLD=1 forces
            // apply_lora_immediately, and the graph apply path a decline routes to cannot
            // convert NVFP4 (GGML_ASSERT(convert_func != nullptr), ggml-cuda.cu). Every
            // krea2 nvfp4 tensor declined, and the first render killed the worker.
            // ABSENT is 1.0; only a wglobal that exists-but-is-unloaded is a real decline.
            work.wglobal = 1.0f;
            return;
        }
        TensorState* wg_state = wg_it->second;
        ggml_tensor* wg       = wg_state != nullptr ? wg_state->tensor : nullptr;
        if (wg == nullptr || wg->data == nullptr) {
            work.no_wglobal = true;
            work.declined   = true;
            return;
        }
        ggml_backend_tensor_get(wg, &work.wglobal, 0, sizeof(float));
        if (!(work.wglobal > 0.0f)) {
            work.wglobal = 1.0f;
        }
    };

    // Build a CHUNK of tensors' deltas on every core, then merge that chunk on the GPU.
    //
    // MEASURED before this change: delta-build was 14.27 s of an 18.92 s fold -- 75% of it
    // -- purely because it ran one tensor at a time while every other core and the GPU sat
    // idle. It is per-tensor CPU work (Q8_0 -> f32 for each module) with no cross-tensor
    // dependency, so it parallelises exactly.
    //
    // Chunked rather than all-at-once because the f32 expansion of a whole 1632-module
    // adapter is gigabytes; a chunk bounds it to build_threads*2 tensors' worth.
    const int build_threads = std::max(1, n_threads_ > 0 ? n_threads_ : 8);
    const size_t chunk      = (size_t)build_threads * 2;
    std::vector<FoldWork> works(chunk);
    size_t work_index = 0, work_n = 0;

    for (size_t ci = 0; ci < candidates.size(); ++ci) {
        if (work_index == work_n) {
            work_n = std::min(chunk, candidates.size() - ci);
            for (size_t i = 0; i < work_n; ++i) {
                works[i].reset(candidates[ci + i]);
            }
            const int64_t t_build0 = ggml_time_us();
            std::atomic<size_t> next{0};
            auto worker = [&]() {
                for (size_t i = next++; i < work_n; i = next++) {
                    build_one(works[i]);
                }
            };
            std::vector<std::thread> pool;
            pool.reserve((size_t)build_threads - 1);
            for (int t = 1; t < build_threads; ++t) {
                pool.emplace_back(worker);
            }
            worker();
            for (std::thread& th : pool) {
                th.join();
            }
            us_delta += ggml_time_us() - t_build0;
            work_index = 0;
        }

        FoldWork& work                                 = works[work_index++];
        TensorState* state                             = work.state;
        ggml_tensor* w                                 = state->tensor;
        std::vector<sd_lora_fold::ModuleDelta>& deltas = work.deltas;
        const float wglobal                            = work.wglobal;

        if (work.no_wglobal) {
            // NOT a soft fallback: SD_LORA_FOLD=1 already forced apply_lora_immediately, and
            // the graph apply path a declined NVFP4 tensor lands on has no convert kernel for
            // it -- the next render aborts the worker on GGML_ASSERT(convert_func != nullptr).
            // Say so, at ERROR, naming the way out.
            LOG_ERROR(
                "lora fold-at-load: nvfp4 tensor '%s' has a .wglobal sidecar that is not loaded; "
                "it cannot be folded AND cannot go through the graph apply path (no NVFP4 convert "
                "kernel). Unset SD_LORA_FOLD to use the per-step runtime adapter instead.",
                state->name.c_str());
        }
        if (work.declined) {
            state->lora_fold_declined = true;
            declined++;
            continue;
        }
        if (deltas.empty()) {
            state->folded_lora_epoch = current_lora_epoch_;  // nothing targets it; done
            continue;
        }

        const bool host = w->buffer != nullptr && ggml_backend_buffer_is_host(w->buffer);
        void* data      = w->data;
        if (!host) {
            host_scratch.resize(ggml_nbytes(w) / sizeof(float) + 1);
            ggml_backend_tensor_get(w, host_scratch.data(), 0, ggml_nbytes(w));
            data = host_scratch.data();
        }
        bool merged = false;
#ifdef SD_USE_CUDA
        // The GPU is idle during model load and this work is trivially parallel, so prefer
        // it for EVERY supported weight type, not just NVFP4.
        //
        // Sending only NVFP4 to the GPU was the earlier rule, on the reasoning that a plain
        // BF16/F32 add is not worth a round trip. That reasoning was wrong, and measurably:
        // the merge is cheap for those types but the DELTA is not, and building it on the
        // CPU means `accumulate_rows` doing a [rows, in] x rank GEMM by hand. MEASURED on
        // the full 1632-module adapter, the 288 tensors left on the CPU cost 14.8 s of a
        // 29 s fold -- more than the entire CUDA half. The round trip is ~2 ms; the CPU GEMM
        // it replaces is ~50.
        if (use_gpu_fold) {
            // The dense kernel indexes the weight linearly over in*out, so it needs a
            // contiguous tensor; the CPU fold walks rows and does not.
            //
            // SD_LORA_FOLD_DENSE=0 sends these back to the CPU fold. That is the ISOLATING
            // CONTROL for this whole change: the dense kernel is the only edit here that can
            // alter a weight value, so with it off the fold must come out BIT-IDENTICAL to
            // the pre-change engine. If it does not, the pre-faulting or the threaded
            // delta-build has a bug, and the render diff is not the rounding story it looks
            // like. Keep it -- "I believe the difference is benign" is not a measurement.
            const bool dense = (w->type == GGML_TYPE_F32 || w->type == GGML_TYPE_F16 ||
                                w->type == GGML_TYPE_BF16) &&
                               ggml_is_contiguous(w) && dense_gpu_enabled;
            if ((w->type == GGML_TYPE_NVFP4 || dense) && ggml_n_dims(w) == 2) {
                std::vector<ggml_cuda_lora_module> gmods;
                gmods.reserve(deltas.size());
                for (const sd_lora_fold::ModuleDelta& d : deltas) {
                    if (d.mid != nullptr) {  // lora_mid needs the chained CPU path
                        gmods.clear();
                        break;
                    }
                    gmods.push_back({d.down, d.up, d.rank, d.row_begin, d.rows, d.scale});
                }
                if (!gmods.empty()) {
                    if (verify_dense && dense) {
                        // Capture the pristine bytes BEFORE the GPU writes them, so the CPU
                        // fold below starts from the same input rather than a folded one.
                        const uint8_t* src = (const uint8_t*)data;
                        verify_pre.assign(src, src + ggml_nbytes(w));
                    }
                    const int64_t t_gpu0 = ggml_time_us();
                    // A dense weight carries no `.wglobal` scaling, so the delta goes in as
                    // it comes out of the GEMM; only NVFP4 needs the divide.
                    merged = dense ? ggml_cuda_lora_fold_dense(data, w->type, in_of(w), out_of(w),
                                                               gmods.data(), (int)gmods.size())
                                   : ggml_cuda_lora_fold_nvfp4(data, in_of(w), out_of(w),
                                                               wglobal != 0.0f ? 1.0f / wglobal : 1.0f,
                                                               gmods.data(), (int)gmods.size(),
                                                               sd_lora_fold::seed_of(state->name));
                    us_gpu += ggml_time_us() - t_gpu0;
                    if (!merged && !warned_gpu_fold_) {
                        LOG_WARN("lora fold-at-load: GPU fold unavailable, falling back to CPU");
                        warned_gpu_fold_ = true;
                    }
                    gpu_folded += merged ? 1 : 0;
                    if (merged && verify_dense && dense) {
                        sd_lora_fold::fold_into_tensor(w, verify_pre.data(), wglobal, deltas,
                                                       state->name, n_threads_);
                        // Distance in ULPs, not just in absolute value. A max|d| alone
                        // cannot separate "the two paths rounded the last bit differently"
                        // from "the kernel is subtly wrong", and a max RELATIVE error is
                        // misleading here because the worst case always lands on some
                        // near-cancelling element where one quantum is a large fraction.
                        // IEEE formats are monotonic in their bit pattern within a sign, so
                        // mapping to a signed ordinal makes |ord(a) - ord(b)| the exact
                        // number of representable steps between them. 1 means adjacent.
                        auto ord16 = [](uint16_t bits) -> int64_t {
                            return (bits & 0x8000u) ? (int64_t)0x8000 - (int64_t)(bits & 0x7FFFu)
                                                    : (int64_t)0x8000 + (int64_t)bits;
                        };
                        auto ord32 = [](uint32_t bits) -> int64_t {
                            return (bits & 0x80000000u) ? (int64_t)0x80000000 - (int64_t)(bits & 0x7FFFFFFFu)
                                                        : (int64_t)0x80000000 + (int64_t)bits;
                        };
                        const int64_t n = ggml_nelements(w);
                        for (int64_t i = 0; i < n; ++i) {
                            float a = 0.0f, b = 0.0f;  // a = GPU result, b = CPU result
                            int64_t ulps = 0;
                            if (w->type == GGML_TYPE_F32) {
                                const uint32_t ba = ((const uint32_t*)data)[i];
                                const uint32_t bb = ((const uint32_t*)verify_pre.data())[i];
                                std::memcpy(&a, &ba, 4);
                                std::memcpy(&b, &bb, 4);
                                ulps = std::llabs(ord32(ba) - ord32(bb));
                            } else if (w->type == GGML_TYPE_F16) {
                                a = ggml_fp16_to_fp32(((const ggml_fp16_t*)data)[i]);
                                b = ggml_fp16_to_fp32(((const ggml_fp16_t*)verify_pre.data())[i]);
                                ulps = std::llabs(ord16(((const uint16_t*)data)[i]) -
                                                  ord16(((const uint16_t*)verify_pre.data())[i]));
                            } else {
                                a = ggml_bf16_to_fp32(((const ggml_bf16_t*)data)[i]);
                                b = ggml_bf16_to_fp32(((const ggml_bf16_t*)verify_pre.data())[i]);
                                ulps = std::llabs(ord16(((const uint16_t*)data)[i]) -
                                                  ord16(((const uint16_t*)verify_pre.data())[i]));
                            }
                            const double d = std::fabs((double)a - (double)b);
                            verify_max_abs = std::max(verify_max_abs, d);
                            const double mag = std::max(std::fabs((double)a), std::fabs((double)b));
                            if (mag > 0.0) {
                                verify_max_rel = std::max(verify_max_rel, d / mag);
                            }
                            verify_sum_abs += d;
                            verify_elems++;
                            if (ulps != 0) {
                                verify_differing++;
                                verify_max_ulps = std::max(verify_max_ulps, ulps);
                            }
                        }
                        verify_n++;
                    }
                }
            }
        }
#endif
        if (!merged) {
            const int64_t t_cpu0 = ggml_time_us();
            const bool ok = sd_lora_fold::fold_into_tensor(w, data, wglobal, deltas, state->name, n_threads_);
            us_cpu += ggml_time_us() - t_cpu0;
            cpu_folded++;
            if (!ok) {
                state->lora_fold_declined = true;
                declined++;
                continue;
            }
        }
        if (!host) {
            ggml_backend_tensor_set(w, data, 0, ggml_nbytes(w));
        }
        state->folded_lora_epoch = current_lora_epoch_;
        folded++;
    }

#ifdef SD_USE_CUDA
    // Hand back the fold's device scratch (delta buffer + staging). It is cached across
    // tensors so the pass does not cudaMalloc 1344 times, but keeping it afterwards costs
    // ~364 MiB of VRAM for the life of the worker -- measured as peak 12513 MiB against a
    // 12149 MiB base, i.e. the GPU fold gave back less than the CPU fold did, which defeats
    // half the point. Re-allocating on the next prepare_params is a couple of mallocs.
    if (gpu_folded > 0) {
        ggml_cuda_lora_fold_release();
    }
#endif
    if (folded > 0) {
        // Covers the CPU fold, and the deferred case: a tensor that was staged when the epoch
        // opened is folded on a LATER prepare_params, still under the SAME lora epoch, so the
        // set_loras() bump alone would not have invalidated anything cached in between.
        bump_weight_content_epoch("lora fold");
    }
    if (folded > 0 || declined > 0) {
        LOG_INFO("lora fold-at-load: merged %zu tensor(s) (%zu on GPU), declined %zu, taking %.2fs",
                 folded, gpu_folded, declined, (ggml_time_ms() - t0) * 1.0f / 1000);
        LOG_INFO("lora fold-at-load: delta-build %.2fs | gpu-merge %.2fs | cpu-merge %.2fs (%zu tensors)",
                 us_delta / 1e6, us_gpu / 1e6, us_cpu / 1e6, cpu_folded);
        if (verify_n > 0) {
            LOG_INFO("lora fold-at-load: VERIFY dense gpu-vs-cpu over %zu tensor(s), %zu elements:",
                     verify_n, verify_elems);
            LOG_INFO("  max ULP distance %lld  (1 = adjacent representable values, i.e. a rounding tie)",
                     (long long)verify_max_ulps);
            LOG_INFO("  differing %zu / %zu (%.4f%%), mean|d| %.3e, max|d| %.3e, max rel %.3e",
                     verify_differing, verify_elems,
                     verify_elems ? 100.0 * (double)verify_differing / (double)verify_elems : 0.0,
                     verify_elems ? verify_sum_abs / (double)verify_elems : 0.0,
                     verify_max_abs, verify_max_rel);
        }
    }
    return true;
}

bool ModelManager::apply_loras_to_params(const std::vector<TensorState*>& states) {
    if (loras_.empty()) {
        return true;
    }

    struct LoraApplyGroup {
        std::map<std::string, ggml_tensor*> model_tensors;
        std::vector<TensorState*> states;
    };

    std::map<ggml_backend_t, LoraApplyGroup> groups;
    for (TensorState* state : states) {
        if (state == nullptr || state->tensor == nullptr ||
            should_ignore(*state) || is_optional_missing_tensor(state->name)) {
            continue;
        }
        if (state->applied_lora_epoch == current_lora_epoch_) {
            continue;
        }
        if (state->folded_lora_epoch == current_lora_epoch_) {
            // Already merged into the params copy; adding it again here would double it.
            state->applied_lora_epoch = current_lora_epoch_;
            continue;
        }
        if (state->compute_backend == nullptr) {
            LOG_ERROR("model manager compute backend is null for lora target tensor '%s'", state->name.c_str());
            return false;
        }
        if (state->tensor->buffer != nullptr &&
            ggml_backend_buffer_get_type(state->tensor->buffer) == split_buffer_type_for(*state)) {
            if (!warned_split_lora_skip_) {
                LOG_WARN(
                    "model manager skipping direct lora application to row-split tensors "
                    "(use --lora-apply-mode at_runtime with row split)");
                warned_split_lora_skip_ = true;
            }
            state->applied_lora_epoch = current_lora_epoch_;
            continue;
        }
        if (state->tensor->data == nullptr) {
            LOG_ERROR("model manager lora target tensor '%s' is not prepared", state->name.c_str());
            return false;
        }
        LoraApplyGroup& group            = groups[state->compute_backend];
        group.model_tensors[state->name] = state->tensor;
        group.states.push_back(state);
    }

    if (groups.empty()) {
        return true;
    }

    std::set<std::string> all_tensor_names = tensor_names();
    for (auto& group_pair : groups) {
        ggml_backend_t compute_backend = group_pair.first;
        LoraApplyGroup& group          = group_pair.second;
        for (const LoraSpec& lora_spec : loras_) {
            if (group.model_tensors.empty()) {
                continue;
            }

            std::string id = lora_id(lora_spec);
            auto lora      = std::make_shared<LoraModel>(id,
                                                    compute_backend,
                                                    compute_backend,
                                                    lora_spec.path,
                                                    lora_spec.is_high_noise ? "model.high_noise_" : "",
                                                    lora_version_);

            LoraModel::filter_t lora_tensor_filter = nullptr;
            if (!lora_spec.tensor_name_prefix_filter.empty()) {
                lora_tensor_filter = [&](const std::string& tensor_name) {
                    return starts_with(tensor_name, lora_spec.tensor_name_prefix_filter);
                };
            }
            if (!lora->load_from_file(n_threads_, lora_tensor_filter)) {
                LOG_WARN("load lora tensors from %s failed", lora_spec.path.c_str());
                if (lora_spec.required) {
                    return false;
                }
                continue;
            }
            if (lora->lora_tensors.empty()) {
                if (lora_spec.required) {
                    LOG_ERROR("required lora has no tensors: %s", lora_spec.path.c_str());
                    return false;
                }
                continue;
            }
            lora->multiplier = lora_spec.multiplier;
            lora->apply(group.model_tensors, all_tensor_names, lora_version_, n_threads_, false);
            lora->release_loaded_tensors();
        }

        for (TensorState* state : group.states) {
            if (state != nullptr) {
                state->applied_lora_epoch = current_lora_epoch_;
            }
        }
    }
    // NO weight-content bump here, deliberately. This path DOES rewrite tensors in place under
    // unchanged names, but it rewrites them to the SAME bytes every time: free_compute_staging_block()
    // resets applied_lora_epoch, so under weight offload the apply re-runs on every staging cycle,
    // always from freshly staged pristine bytes with the same adapter at the same multiplier. The
    // post-apply bytes are therefore invariant for the whole lora epoch, and set_loras() already
    // bumps at the only transition that changes them.
    //
    // Bumping here anyway would be the expensive kind of wrong: staging cycles run ~23 times for a
    // single 25-frame render, so it would invalidate every byte-derived cache ~23 times per render
    // for no correctness gain — perfectly correct and perfectly useless, which is the failure mode
    // this whole family of fixes has to avoid.
    return true;
}

void ModelManager::invalidate_cudnn_conv_weight_caches() {
#ifdef SD_USE_CUDA
    bool have_cuda = false;
    for (const auto& state : tensor_states_) {
        if (state != nullptr && state->compute_backend != nullptr &&
            ggml_backend_is_cuda(state->compute_backend)) {
            have_cuda = true;
            break;
        }
    }
    if (!have_cuda) {
        return;
    }
    // Both wrappers sync the device first; no conv may be in flight.
    ggml_backend_cuda_release_cudnn_conv3d_weights();
    ggml_backend_cuda_release_cudnn_conv2d_weights();
#endif
}

void ModelManager::reset_lora_applied_params() {
    // FIRST, while tensor->data still points into the mapping: undo any fold. Releasing
    // the params blocks below does NOT do this -- see TensorState::folded_lora_epoch --
    // and free_params_storage_block() nulls the data pointers we need to find the pages.
    unfold_loras_from_params();
    // The LoRA epoch moved: any conv weight this adapter touches now means something
    // different under the same tensor name. See invalidate_cudnn_conv_weight_caches().
    invalidate_cudnn_conv_weight_caches();
    release_compute_staging_blocks(true);
    release_params_storage_blocks(true);
    for (auto& state : tensor_states_) {
        state->applied_lora_epoch  = UINT64_MAX;
        state->folded_lora_epoch   = UINT64_MAX;
        state->lora_fold_declined  = false;
    }
    release_fold_loras();
}

bool ModelManager::should_ignore(const TensorState& state) const {
    for (const auto& ignore_prefix : common_ignore_tensors_) {
        if (starts_with(state.name, ignore_prefix)) {
            return true;
        }
    }
    return false;
}

bool ModelManager::is_optional_missing_tensor(const std::string& name) const {
    return name.find("cond_stage_model.transformer.text_model.encoder.layers.23") != std::string::npos ||
           name.find("alphas_cumprod") != std::string::npos;
}

bool ModelManager::validate_tensor(const TensorState& state) const {
    if (state.tensor == nullptr || should_ignore(state) || is_optional_missing_tensor(state.name)) {
        return true;
    }

    const auto& tensor_storage_map = model_loader_.get_tensor_storage_map();
    auto ts_it                     = tensor_storage_map.find(state.name);
    if (ts_it == tensor_storage_map.end()) {
        LOG_ERROR("%s tensor '%s' not in model metadata", state.desc.c_str(), state.name.c_str());
        return false;
    }

    const TensorStorage& tensor_storage = ts_it->second;
    if (state.tensor->ne[0] != tensor_storage.ne[0] ||
        state.tensor->ne[1] != tensor_storage.ne[1] ||
        state.tensor->ne[2] != tensor_storage.ne[2] ||
        state.tensor->ne[3] != tensor_storage.ne[3]) {
        LOG_ERROR(
            "%s tensor '%s' has wrong shape in model metadata: got [%d, %d, %d, %d], expected [%d, %d, %d, %d]",
            state.desc.c_str(),
            state.name.c_str(),
            (int)tensor_storage.ne[0], (int)tensor_storage.ne[1], (int)tensor_storage.ne[2], (int)tensor_storage.ne[3],
            (int)state.tensor->ne[0], (int)state.tensor->ne[1], (int)state.tensor->ne[2], (int)state.tensor->ne[3]);
        return false;
    }
    return true;
}

bool ModelManager::mmap_params(const std::vector<TensorState*>& states,
                               std::vector<ParamsStorageBlock*>& created_storage_blocks) {
    std::map<std::string, ggml_tensor*> mmap_candidates;
    std::map<std::string, TensorState*> mmap_states;
    for (TensorState* state : states) {
        if (state == nullptr || !can_mmap_storage(*state) || state->tensor == nullptr ||
            state->tensor->data != nullptr || state->tensor->view_src != nullptr) {
            continue;
        }
        mmap_candidates[state->name] = state->tensor;
        mmap_states[state->name]     = state;
    }
    if (mmap_candidates.empty()) {
        return true;
    }

    auto mmap_store = model_loader_.mmap_tensors(mmap_candidates, {}, writable_mmap_);
    if (mmap_store.empty()) {
        return true;
    }

    auto block                = std::make_unique<ParamsStorageBlock>();
    block->mmap_tensor_stores = std::move(mmap_store);
    ParamsStorageBlock* raw   = block.get();
    for (const auto& pair : mmap_states) {
        TensorState* state = pair.second;
        if (state != nullptr && state->tensor != nullptr && state->tensor->data != nullptr) {
            block->states.push_back(state);
        }
    }

    if (!block->states.empty()) {
        params_storage_blocks_.push_back(std::move(block));
        created_storage_blocks.push_back(raw);
    }
    return true;
}

bool ModelManager::can_mmap_storage(const TensorState& state) const {
    if (!enable_mmap_ || state.residency_mode != ResidencyMode::ParamBackend) {
        return false;
    }
    if (state.compute_backend == nullptr || state.params_backend == nullptr) {
        return false;
    }
    return sd_backend_is_cpu(state.compute_backend) ||
           sd_backend_is_cpu(state.params_backend) ||
           backend_supports_host_buffer(state.compute_backend);
}

bool ModelManager::alloc_params_buffers(const std::vector<TensorState*>& states,
                                        std::vector<ParamsStorageBlock*>& created_storage_blocks) {
    std::map<std::pair<ggml_backend_buffer_type_t, int>, std::vector<TensorState*>> states_by_buffer_type;
    for (TensorState* state : states) {
        if (state == nullptr || state->tensor == nullptr) {
            continue;
        }
        ggml_backend_buffer_type_t params_buft = params_buffer_type_for(*state);
        if (params_buft == nullptr) {
            return false;
        }
        states_by_buffer_type[{params_buft, static_cast<int>(state->residency_mode)}].push_back(state);
    }

    for (const auto& pair : states_by_buffer_type) {
        ggml_backend_buffer_type_t params_buft  = pair.first.first;
        const std::vector<TensorState*>& states = pair.second;
        size_t alignment                        = ggml_backend_buft_get_alignment(params_buft);
        size_t max_size                         = ggml_backend_buft_get_max_size(params_buft);

        auto alloc_chunk = [&](const std::vector<TensorState*>& chunk, size_t chunk_size) -> bool {
            if (chunk.empty() || chunk_size == 0) {
                return true;
            }

            ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(params_buft, chunk_size);
            if (buffer == nullptr) {
                LOG_ERROR("model manager alloc params backend buffer failed, size = %.2fMB",
                          chunk_size / (1024.0 * 1024.0));
                return false;
            }
            ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

            std::vector<ggml_tensor*> initialized_tensors;
            void* base    = ggml_backend_buffer_get_base(buffer);
            size_t offset = aligned_offset(base, 0, ggml_backend_buffer_get_alignment(buffer));
            for (TensorState* state : chunk) {
                ggml_tensor* tensor     = state->tensor;
                size_t tensor_size      = GGML_PAD(ggml_backend_buffer_get_alloc_size(buffer, tensor),
                                                   ggml_backend_buffer_get_alignment(buffer));
                enum ggml_status status = ggml_backend_tensor_alloc(buffer, tensor, static_cast<char*>(base) + offset);
                if (status != GGML_STATUS_SUCCESS) {
                    LOG_ERROR("model manager failed to initialize params tensor '%s'", ggml_get_name(tensor));
                    for (ggml_tensor* initialized : initialized_tensors) {
                        initialized->buffer = nullptr;
                        initialized->data   = nullptr;
                        initialized->extra  = nullptr;
                    }
                    LOG_DEBUG("model manager releasing params backend buffer (%6.2f MB, %zu tensors, %s)",
                              ggml_backend_buffer_get_size(buffer) / (1024.f * 1024.f),
                              initialized_tensors.size(),
                              ggml_backend_buffer_is_host(buffer) ? "RAM" : "VRAM");
                    ggml_backend_buffer_free(buffer);
                    return false;
                }
                initialized_tensors.push_back(tensor);
                offset += tensor_size;
            }

            auto block              = std::make_unique<ParamsStorageBlock>();
            block->buffer           = buffer;
            block->states           = chunk;
            ParamsStorageBlock* raw = block.get();
            params_storage_blocks_.push_back(std::move(block));
            created_storage_blocks.push_back(raw);

            return true;
        };

        std::vector<TensorState*> chunk;
        size_t chunk_size = 0;
        for (TensorState* state : states) {
            ggml_tensor* tensor = state->tensor;
            size_t tensor_size  = GGML_PAD(ggml_backend_buft_get_alloc_size(params_buft, tensor), alignment);
            // Some backends, e.g. Vulkan, report a preferred chunk size here rather than a
            // hard per-tensor allocation limit. Oversized tensors are allocated alone.
            if (!chunk.empty() && max_size > 0 && chunk_size + tensor_size > max_size) {
                if (!alloc_chunk(chunk, chunk_size)) {
                    return false;
                }
                chunk.clear();
                chunk_size = 0;
            }
            chunk.push_back(state);
            chunk_size += tensor_size;
        }

        if (!alloc_chunk(chunk, chunk_size)) {
            return false;
        }
    }

    return true;
}

bool ModelManager::load_tensors(const std::vector<TensorState*>& states) {
    std::map<std::string, TensorState*> states_by_name;
    std::set<std::string> target_tensor_names;
    for (TensorState* state : states) {
        if (state == nullptr) {
            continue;
        }
        states_by_name[state->name] = state;
        target_tensor_names.insert(state->name);
    }
    if (states_by_name.empty()) {
        return true;
    }

    std::set<std::string> loaded_names;
    std::mutex loaded_names_mutex;
    auto on_new_tensor_cb = [&](const TensorStorage& tensor_storage, ggml_tensor** dst_tensor) -> bool {
        const std::string& name = tensor_storage.name;
        *dst_tensor             = nullptr;

        auto state_it = states_by_name.find(name);
        if (state_it == states_by_name.end()) {
            return true;
        }

        TensorState* state = state_it->second;
        if (state == nullptr || state->tensor == nullptr) {
            LOG_ERROR("model manager tensor '%s' is null", name.c_str());
            return false;
        }

        if (state->tensor->ne[0] != tensor_storage.ne[0] ||
            state->tensor->ne[1] != tensor_storage.ne[1] ||
            state->tensor->ne[2] != tensor_storage.ne[2] ||
            state->tensor->ne[3] != tensor_storage.ne[3]) {
            LOG_ERROR(
                "model manager tensor '%s' has wrong shape in model file: got [%d, %d, %d, %d], expected [%d, %d, %d, %d]",
                name.c_str(),
                (int)tensor_storage.ne[0], (int)tensor_storage.ne[1], (int)tensor_storage.ne[2], (int)tensor_storage.ne[3],
                (int)state->tensor->ne[0], (int)state->tensor->ne[1], (int)state->tensor->ne[2], (int)state->tensor->ne[3]);
            return false;
        }

        {
            std::lock_guard<std::mutex> lock(loaded_names_mutex);
            loaded_names.insert(name);
        }
        *dst_tensor = state->tensor;
        return true;
    };

    if (!model_loader_.load_tensors(on_new_tensor_cb, enable_mmap_, &target_tensor_names)) {
        LOG_ERROR("model manager load tensors failed");
        return false;
    }

    bool missing = false;
    for (const auto& pair : states_by_name) {
        const std::string& name = pair.first;
        if (loaded_names.find(name) == loaded_names.end()) {
            LOG_ERROR("model manager tensor '%s' was not loaded", name.c_str());
            missing = true;
        }
    }
    if (missing) {
        return false;
    }

    for (const auto& pair : states_by_name) {
        pair.second->loaded_to_params_backend = true;
    }
    return true;
}

ggml_backend_buffer_type_t ModelManager::params_buffer_type_for(const TensorState& state) const {
    if (state.params_backend == nullptr) {
        LOG_ERROR("model manager params backend is null for tensor '%s'", state.name.c_str());
        return nullptr;
    }
    ggml_backend_buffer_type_t params_buft = nullptr;
    if (state.compute_backend != nullptr && state.params_backend != state.compute_backend) {
        ggml_backend_dev_t compute_dev = ggml_backend_get_device(state.compute_backend);
        if (compute_dev != nullptr) {
            params_buft = ggml_backend_dev_host_buffer_type(compute_dev);
        }
    } else if (state.params_backend == state.compute_backend) {
        params_buft = split_buffer_type_for(state);
    }
    if (params_buft == nullptr) {
        params_buft = ggml_backend_get_default_buffer_type(state.params_backend);
    }
    return params_buft;
}

void ModelManager::free_compute_staging_block(ComputeStagingBlock& block) {
    for (auto& staged_tensor : block.staged_tensors) {
        TensorState* state          = staged_tensor.first;
        ggml_tensor* staging_tensor = staged_tensor.second;
        if (state == nullptr || state->tensor == nullptr || staging_tensor == nullptr) {
            continue;
        }
        ggml_tensor* managed_tensor = state->tensor;
        managed_tensor->buffer      = staging_tensor->buffer;
        managed_tensor->data        = staging_tensor->data;
        managed_tensor->extra       = staging_tensor->extra;
        staging_tensor->buffer      = nullptr;
        staging_tensor->data        = nullptr;
        staging_tensor->extra       = nullptr;

        state->staged_to_compute_backend = false;
        state->applied_lora_epoch        = UINT64_MAX;
    }

    if (block.buffer != nullptr) {
        LOG_DEBUG("model manager releasing compute params (%6.2f MB, %zu tensors) from %s",
                  ggml_backend_buffer_get_size(block.buffer) / (1024.f * 1024.f),
                  block.staged_tensors.size(),
                  block.compute_backend != nullptr ? ggml_backend_name(block.compute_backend) : "unknown");
        // NOTE: nothing is invalidated here on purpose. Staging swaps managed_tensor->data
        // to point INTO the buffer we have just freed, so from here on that address range
        // is free for the pool to hand to a DIFFERENT tensor. That used to corrupt the
        // cuDNN conv weight-reorder caches, which were keyed by the weight's DEVICE
        // POINTER: every conv after the first render took a stale hit and convolved with
        // another tensor's reordered weights.
        //
        // Those caches are now keyed by STABLE IDENTITY instead (tensor name + buffer +
        // shape + type + device — ggml/src/ggml-cuda/cudnn-weight-key.cuh), so a recycled
        // address cannot alias and the reorder survives the unstage. Releasing here would
        // be correct but pointlessly slower, and — more to the point — it would leave
        // correctness depending on every future path that recycles a staged address
        // remembering to do the same thing. It does not.
        ggml_backend_buffer_free(block.buffer);
        block.buffer = nullptr;
    }
    if (block.staging_ctx != nullptr) {
        ggml_free(block.staging_ctx);
        block.staging_ctx = nullptr;
    }
    block.staged_tensors.clear();
}

void ModelManager::release_compute_staging_blocks(bool force,
                                                  const std::unordered_set<TensorState*>* target_states) {
    for (auto it = compute_staging_blocks_.begin(); it != compute_staging_blocks_.end();) {
        ComputeStagingBlock* block = it->get();
        bool can_release           = force;
        if (!can_release) {
            can_release = std::all_of(block->staged_tensors.begin(),
                                      block->staged_tensors.end(),
                                      [target_states](const std::pair<TensorState*, ggml_tensor*>& pair) {
                                          TensorState* state = pair.first;
                                          if (state == nullptr) {
                                              return true;
                                          }
                                          if (target_states != nullptr &&
                                              target_states->find(state) == target_states->end()) {
                                              return false;
                                          }
                                          return state->active_prepare_count == 0 &&
                                                 state->retained_compute_count == 0;
                                      });
        }

        if (can_release) {
            if (model_manager_profile_enabled() && block->buffer != nullptr) {
                LOG_INFO("[MM_PROFILE] release compute %.2f MiB / %zu tensors",
                         ggml_backend_buffer_get_size(block->buffer) / (1024.f * 1024.f),
                         block->staged_tensors.size());
            }
            free_compute_staging_block(*block);
            it = compute_staging_blocks_.erase(it);
        } else {
            if (model_manager_profile_enabled() && block->buffer != nullptr) {
                size_t active_tensors = 0;
                for (const auto& staged_tensor : block->staged_tensors) {
                    const TensorState* state = staged_tensor.first;
                    if (state != nullptr && state->active_prepare_count > 0) {
                        ++active_tensors;
                    }
                }
                LOG_INFO("[MM_PROFILE] retain compute %.2f MiB / %zu tensors (%zu active)",
                         ggml_backend_buffer_get_size(block->buffer) / (1024.f * 1024.f),
                         block->staged_tensors.size(),
                         active_tensors);
            }
            ++it;
        }
    }
}

void ModelManager::free_params_storage_block(ParamsStorageBlock& block) {
    if (block.buffer != nullptr) {
        // The weights behind these names are going away; the next load may bring different
        // content back under the same names. See invalidate_cudnn_conv_weight_caches().
        invalidate_cudnn_conv_weight_caches();
        LOG_DEBUG("model manager releasing params backend buffer (%6.2f MB, %zu tensors, %s)",
                  ggml_backend_buffer_get_size(block.buffer) / (1024.f * 1024.f),
                  block.states.size(),
                  ggml_backend_buffer_is_host(block.buffer) ? "RAM" : "VRAM");
        ggml_backend_buffer_free(block.buffer);
        block.buffer = nullptr;
    }
    block.mmap_tensor_stores.clear();

    for (TensorState* state : block.states) {
        if (state == nullptr || state->tensor == nullptr) {
            continue;
        }
        state->tensor->buffer = nullptr;
        state->tensor->data   = nullptr;
        state->tensor->extra  = nullptr;

        state->loaded_to_params_backend = false;
        state->applied_lora_epoch       = UINT64_MAX;
        // 🔴 Clearing this does NOT undo a merge. For an alloc'd buffer it is accurate --
        // the buffer is gone and the next load_tensors() re-reads the file. For an mmap
        // block it is not: the mapping outlives this block (ModelLoader owns it) and the
        // next mmap_tensors() hands back the same copy-on-write pages. Callers that force
        // a release across a LoRA epoch must run unfold_loras_from_params() FIRST;
        // reset_lora_applied_params() does.
        state->folded_lora_epoch  = UINT64_MAX;
        state->lora_fold_declined = false;
    }
    block.states.clear();
}

void ModelManager::release_params_storage_blocks(bool force,
                                                 const std::unordered_set<TensorState*>* target_states) {
    for (auto it = params_storage_blocks_.begin(); it != params_storage_blocks_.end();) {
        ParamsStorageBlock* block = it->get();
        bool can_release          = force;
        if (!can_release) {
            can_release = std::all_of(block->states.begin(),
                                      block->states.end(),
                                      [target_states](TensorState* state) {
                                          if (state == nullptr) {
                                              return true;
                                          }
                                          if (target_states != nullptr &&
                                              target_states->find(state) == target_states->end()) {
                                              return false;
                                          }
                                          return state->active_prepare_count == 0 &&
                                                 !state->staged_to_compute_backend &&
                                                 state->residency_mode == ResidencyMode::Disk;
                                      });
        }

        if (can_release) {
            free_params_storage_block(*block);
            it = params_storage_blocks_.erase(it);
        } else {
            ++it;
        }
    }
}

void ModelManager::erase_params_storage_block(ParamsStorageBlock* block) {
    auto it = std::find_if(params_storage_blocks_.begin(),
                           params_storage_blocks_.end(),
                           [block](const std::unique_ptr<ParamsStorageBlock>& item) {
                               return item.get() == block;
                           });
    if (it != params_storage_blocks_.end()) {
        params_storage_blocks_.erase(it);
    }
}

void ModelManager::release_all() {
    for (auto& state : tensor_states_) {
        state->active_prepare_count = 0;
        state->applied_lora_epoch   = UINT64_MAX;
        state->folded_lora_epoch    = UINT64_MAX;
    }
    release_compute_staging_blocks(true);
    release_params_storage_blocks(true);
    release_fold_loras();
}

bool ModelManager::resolve_required_tensor_states(const std::vector<ggml_tensor*>& tensors,
                                                  std::vector<TensorState*>& required_states) const {
    required_states.clear();
    std::unordered_set<TensorState*> seen;
    for (ggml_tensor* tensor : tensors) {
        if (tensor == nullptr) {
            continue;
        }
        const char* raw_name = ggml_get_name(tensor);
        if (raw_name == nullptr || raw_name[0] == '\0') {
            LOG_ERROR("model manager unnamed tensor is not registered");
            return false;
        }
        auto state_it = tensor_states_by_name_.find(raw_name);
        if (state_it == tensor_states_by_name_.end()) {
            LOG_ERROR("model manager tensor '%s' is not registered", raw_name);
            return false;
        }
        TensorState* state = state_it->second;
        if (state == nullptr) {
            LOG_ERROR("model manager tensor '%s' has no tensor state", raw_name);
            return false;
        }
        if (seen.insert(state).second) {
            required_states.push_back(state);
        }
    }
    return true;
}

bool ModelManager::assign_compute_backend(const std::vector<ggml_tensor*>& tensors,
                                          ggml_backend_t compute_backend) {
    if (tensors.empty()) {
        return true;
    }
    if (compute_backend == nullptr) {
        LOG_ERROR("model manager cannot assign tensors to a null compute backend");
        return false;
    }

    std::vector<TensorState*> required_states;
    if (!resolve_required_tensor_states(tensors, required_states)) {
        return false;
    }

    for (TensorState* state : required_states) {
        if (state == nullptr || state->tensor == nullptr) {
            continue;
        }

        const bool params_follow_compute = state->params_follow_compute_backend ||
                                           state->residency_mode == ResidencyMode::Disk;
        const bool compute_changes = state->compute_backend != compute_backend;
        const bool params_changes  = params_follow_compute && state->params_backend != compute_backend;
        if (!compute_changes && !params_changes) {
            continue;
        }

        if (state->active_prepare_count > 0 || state->staged_to_compute_backend) {
            LOG_ERROR("model manager cannot move active tensor '%s' to another compute backend",
                      state->name.c_str());
            return false;
        }
        if (params_changes && state->loaded_to_params_backend) {
            LOG_ERROR("model manager cannot move loaded tensor '%s' to another params backend",
                      state->name.c_str());
            return false;
        }

        state->compute_backend = compute_backend;
        if (params_follow_compute) {
            state->params_backend = compute_backend;
        }
    }

    return true;
}

bool ModelManager::prepare_params(const std::vector<ggml_tensor*>& tensors) {
    if (tensors.empty()) {
        return true;
    }

    std::vector<TensorState*> required_states;
    if (!resolve_required_tensor_states(tensors, required_states)) {
        return false;
    }

    if (!load_tensors_to_params_backend(required_states)) {
        return false;
    }

    // Merge before staging: a fold written to a compute staging block would be discarded
    // the next time that block is rebuilt, which under weight offload is every graph.
    if (!fold_loras_into_params(required_states)) {
        release_params_storage_blocks(false);
        return false;
    }

    if (!stage_tensors_to_compute_backend(required_states)) {
        release_compute_staging_blocks(false);
        release_params_storage_blocks(false);
        return false;
    }

    if (!apply_loras_to_params(required_states)) {
        release_compute_staging_blocks(false);
        release_params_storage_blocks(false);
        return false;
    }

    for (TensorState* state : required_states) {
        if (state == nullptr) {
            continue;
        }
        state->active_prepare_count++;
    }
    return true;
}

bool ModelManager::retain_compute_backend_params(const std::vector<ggml_tensor*>& tensors) {
    if (tensors.empty()) {
        return true;
    }

    std::vector<TensorState*> required_states;
    if (!resolve_required_tensor_states(tensors, required_states)) {
        return false;
    }

    size_t new_bytes = 0;
    ggml_backend_t compute_backend = nullptr;
    for (TensorState* state : required_states) {
        if (state == nullptr || should_ignore(*state) || is_optional_missing_tensor(state->name)) {
            continue;
        }
        if (state->retained_compute_count > 0) {
            continue;
        }
        if (state->compute_backend == nullptr || state->tensor == nullptr) {
            LOG_ERROR("model manager cannot retain tensor '%s' without a compute backend", state->name.c_str());
            return false;
        }
        if (compute_backend == nullptr) {
            compute_backend = state->compute_backend;
        } else if (compute_backend != state->compute_backend) {
            LOG_ERROR("model manager cannot retain tensors across different compute backends");
            return false;
        }
        new_bytes += ggml_nbytes(state->tensor);
    }

    if (compute_backend != nullptr) {
        ggml_backend_dev_t device = ggml_backend_get_device(compute_backend);
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        if (device != nullptr) {
            ggml_backend_dev_memory(device, &free_bytes, &total_bytes);
        }
        const size_t headroom = model_manager_resident_headroom_bytes();
        if (free_bytes != 0 && (new_bytes > free_bytes || free_bytes - new_bytes < headroom)) {
            LOG_WARN("model manager declined %.2f MiB retained compute weights: %.2f MiB free, %.2f MiB headroom",
                     new_bytes / (1024.0 * 1024.0), free_bytes / (1024.0 * 1024.0),
                     headroom / (1024.0 * 1024.0));
            return false;
        }
    }

    // Same ordering rule as prepare_params: merge into the params copy BEFORE it is staged,
    // or a retained resident tensor would be pinned on the GPU without the LoRA in it.
    if (!load_tensors_to_params_backend(required_states) ||
        !fold_loras_into_params(required_states) ||
        !stage_tensors_to_compute_backend(required_states)) {
        release_compute_staging_blocks(false);
        release_params_storage_blocks(false);
        return false;
    }
    for (TensorState* state : required_states) {
        if (state != nullptr && !should_ignore(*state) && !is_optional_missing_tensor(state->name)) {
            state->retained_compute_count++;
        }
    }
    LOG_INFO("model manager retained %zu compute tensors (%.2f MiB newly reserved)",
             required_states.size(), new_bytes / (1024.0 * 1024.0));
    return true;
}

void ModelManager::finish_compute_backend_usage(const std::vector<TensorState*>& states) {
    if (states.empty()) {
        return;
    }

    std::unordered_set<TensorState*> target_states;
    for (TensorState* state : states) {
        if (state == nullptr || !target_states.insert(state).second) {
            continue;
        }
        if (state->active_prepare_count > 0) {
            state->active_prepare_count--;
        }
    }
    release_compute_staging_blocks(false, &target_states);
}

void ModelManager::release_compute_backend_params(const std::vector<ggml_tensor*>& tensors) {
    if (tensors.empty()) {
        return;
    }
    std::vector<TensorState*> required_states;
    if (!resolve_required_tensor_states(tensors, required_states)) {
        return;
    }
    finish_compute_backend_usage(required_states);
}

void ModelManager::release_retained_compute_backend_params(const std::vector<ggml_tensor*>& tensors) {
    if (tensors.empty()) {
        return;
    }
    std::vector<TensorState*> required_states;
    if (!resolve_required_tensor_states(tensors, required_states)) {
        return;
    }
    for (TensorState* state : required_states) {
        if (state != nullptr && state->retained_compute_count > 0) {
            state->retained_compute_count--;
        }
    }
    release_compute_staging_blocks(false);
}

void ModelManager::release_params_backend_params(const std::vector<ggml_tensor*>& tensors) {
    if (tensors.empty()) {
        return;
    }
    std::vector<TensorState*> required_states;
    if (!resolve_required_tensor_states(tensors, required_states)) {
        return;
    }
    if (required_states.empty()) {
        return;
    }
    std::unordered_set<TensorState*> target_states(required_states.begin(), required_states.end());
    release_params_storage_blocks(false, &target_states);
}
