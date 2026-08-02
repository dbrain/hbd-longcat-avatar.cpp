#ifndef __MODEL_MANAGER_H__
#define __MODEL_MANAGER_H__

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "model_loader.h"
#include "weight_manager.h"

// Only ever held by shared_ptr here; lora.hpp includes this header, so the full type is
// deliberately not visible until model_manager.cpp, which does include it.
struct LoraModel;

class ModelManager : public RunnerWeightManager {
public:
    enum class ResidencyMode {
        Disk,
        ParamBackend,
    };

    struct LoraSpec {
        std::string path;
        float multiplier   = 1.0f;
        bool is_high_noise = false;
        std::string tensor_name_prefix_filter;
        bool required = false;
    };

private:
    struct TensorState {
        std::string name;
        ggml_tensor* tensor = nullptr;
        std::string desc;

        ResidencyMode residency_mode       = ResidencyMode::ParamBackend;
        ggml_backend_t compute_backend     = nullptr;
        ggml_backend_t params_backend      = nullptr;
        bool allow_split_buffer            = false;
        bool params_follow_compute_backend = false;
        bool metadata_validated            = false;
        enum ggml_op usage_op              = GGML_OP_NONE;

        int active_prepare_count = 0;
        int retained_compute_count = 0;

        bool loaded_to_params_backend  = false;
        bool staged_to_compute_backend = false;
        uint64_t applied_lora_epoch    = UINT64_MAX;
        // Set once the LoRA delta has been merged into the PARAMS-backend copy of this
        // tensor. Distinct from applied_lora_epoch, which tracks the compute-staged copy
        // and is therefore reset every time a staging block is freed -- a fold survives
        // re-staging.
        //
        // 🔴 CLEARING THIS DOES NOT UNDO THE MERGE, and for a long time the comment here
        // claimed it did ("the next load re-reads pristine weights from the model file").
        // That is true only for a params block that owns its bytes -- an alloc'd buffer,
        // freed and refilled by load_tensors(). It is FALSE for the mmap path, which is
        // the one prod runs and the only one the fold is even allowed on: those bytes are
        // copy-on-write pages of a mapping the ModelLoader owns for its whole life, so
        // releasing the block just re-points tensor->data at the same mutated pages.
        // ModelManager::unfold_loras_from_params() is what actually restores them, and it
        // needs this flag, so it must run BEFORE the flag is cleared.
        uint64_t folded_lora_epoch = UINT64_MAX;
        // A LoRA target this fold cannot handle (non-2D, or a row length that is not a
        // whole number of NVFP4 blocks). Left to the graph-based apply path.
        bool lora_fold_declined = false;
    };

    struct ParamsStorageBlock {
        ggml_backend_buffer_t buffer = nullptr;
        std::vector<MmapTensorStore> mmap_tensor_stores;
        std::vector<TensorState*> states;
    };

    struct ComputeStagingBlock {
        ggml_backend_t compute_backend = nullptr;
        ggml_backend_buffer_t buffer   = nullptr;
        ggml_context* staging_ctx      = nullptr;
        std::vector<std::pair<TensorState*, ggml_tensor*>> staged_tensors;
    };

    ModelLoader model_loader_;
    std::vector<std::unique_ptr<TensorState>> tensor_states_;
    std::map<std::string, TensorState*> tensor_states_by_name_;
    std::vector<std::unique_ptr<ParamsStorageBlock>> params_storage_blocks_;
    std::vector<std::unique_ptr<ComputeStagingBlock>> compute_staging_blocks_;
    std::map<ggml_backend_t, ggml_backend_buffer_type_t> split_buffer_types_;
    bool warned_split_lora_skip_ = false;
    bool warned_gpu_fold_        = false;
    std::set<std::string> common_ignore_tensors_;
    std::vector<LoraSpec> loras_;
    // LoRA files kept open only for the duration of one fold epoch. Held here rather than
    // reloaded per prepare_params() because under weight offload prepare_params() runs
    // once per graph, and re-reading a 1.3 GB adapter each time would dwarf the fold.
    std::vector<std::shared_ptr<LoraModel>> fold_loras_;
    ggml_backend_t fold_cpu_backend_ = nullptr;
    uint64_t fold_loras_epoch_       = UINT64_MAX;
    bool fold_loras_failed_          = false;
    SDVersion lora_version_      = VERSION_COUNT;
    uint64_t current_lora_epoch_ = 0;
    int n_threads_               = 0;
    bool enable_mmap_            = false;
    bool writable_mmap_          = false;

    void finish_compute_backend_usage(const std::vector<TensorState*>& states);
    void release_all();

    bool resolve_required_tensor_states(const std::vector<ggml_tensor*>& tensors,
                                        std::vector<TensorState*>& required_states) const;
    bool should_ignore(const TensorState& state) const;
    bool is_optional_missing_tensor(const std::string& name) const;
    bool validate_tensor(const TensorState& state) const;

    bool load_tensors_to_params_backend(const std::vector<TensorState*>& states);
    bool apply_loras_to_params(const std::vector<TensorState*>& states);
    bool fold_loras_into_params(const std::vector<TensorState*>& states);
    void unfold_loras_from_params();
    void release_fold_loras();
    bool mmap_params(const std::vector<TensorState*>& states,
                     std::vector<ParamsStorageBlock*>& created_storage_blocks);
    bool can_mmap_storage(const TensorState& state) const;
    bool alloc_params_buffers(const std::vector<TensorState*>& states,
                              std::vector<ParamsStorageBlock*>& created_storage_blocks);
    bool load_tensors(const std::vector<TensorState*>& states);
    bool stage_tensors_to_compute_backend(const std::vector<TensorState*>& states);

    ggml_backend_buffer_type_t params_buffer_type_for(const TensorState& state) const;
    ggml_backend_buffer_type_t split_buffer_type_for(const TensorState& state) const;
    void release_compute_staging_blocks(bool force                                            = false,
                                        const std::unordered_set<TensorState*>* target_states = nullptr);
    void release_params_storage_blocks(bool force                                            = false,
                                       const std::unordered_set<TensorState*>* target_states = nullptr);
    void free_compute_staging_block(ComputeStagingBlock& block);
    void free_params_storage_block(ParamsStorageBlock& block);
    void erase_params_storage_block(ParamsStorageBlock* block);
    void reset_lora_applied_params();

    // Belt-and-braces invalidation of the cuDNN conv2d/conv3d reordered-weight caches.
    //
    // Those caches are keyed by tensor NAME (+ backend buffer, shape, type, device), which
    // is stable across staging -- that is what makes weight offload correct without
    // touching them per render, and it is deliberately NOT called from
    // free_compute_staging_block(). The one thing a name cannot capture is the CONTENT
    // under that name changing, which happens at exactly two places, both rare and both
    // central:
    //   * a LoRA epoch change  -- the delta is merged into the params copy
    //     (fold_loras_into_params) or the staged copy (apply_loras_to_params);
    //   * params being unloaded -- the next load may bring a DIFFERENT checkpoint into the
    //     same registered tensor names.
    // No-op unless some tensor actually computes on a CUDA backend.
    void invalidate_cudnn_conv_weight_caches();

public:
    ~ModelManager() override;

    ModelLoader& loader() { return model_loader_; }
    const ModelLoader& loader() const { return model_loader_; }

    void set_n_threads(int n_threads) {
        n_threads_ = n_threads;
        model_loader_.set_n_threads(n_threads);
    }
    void set_enable_mmap(bool enable_mmap) { enable_mmap_ = enable_mmap; }
    void set_writable_mmap(bool writable_mmap) { writable_mmap_ = writable_mmap; }
    void set_common_ignore_tensors(std::set<std::string> ignore_tensors);
    void set_loras(std::vector<LoraSpec> loras, SDVersion version);
    void set_split_buffer_type(ggml_backend_t compute_backend, ggml_backend_buffer_type_t split_buft);

    static bool tensor_shape_supports_split_buffer(const ggml_tensor* tensor);

    std::set<std::string> tensor_names() const;

    // Registered graph tensor for a full (untruncated) loader tensor name, or nullptr.
    // Needed to recover the tensor's ggml ->name, which ggml_set_name truncates to
    // GGML_MAX_NAME and which is therefore the only key the CUDA backend ever sees.
    ggml_tensor* find_tensor(const std::string& name) const;

    bool register_param_tensors(const std::string& desc,
                                std::map<std::string, ggml_tensor*> tensors,
                                ResidencyMode residency_mode,
                                ggml_backend_t compute_backend,
                                ggml_backend_t params_backend,
                                size_t* registered_tensor_size                         = nullptr,
                                bool allow_split_buffer                                = false,
                                bool params_follow_compute_backend                     = false,
                                const std::map<ggml_tensor*, enum ggml_op>* tensor_ops = nullptr);

    bool unregister_param_tensors(const std::string& desc,
                                  size_t* registered_tensor_size = nullptr);

    template <typename Runner>
    bool register_runner_params(const std::string& desc,
                                Runner& runner,
                                ResidencyMode residency_mode,
                                ggml_backend_t compute_backend,
                                ggml_backend_t params_backend,
                                size_t* registered_tensor_size = nullptr) {
        std::map<std::string, ggml_tensor*> tensors;
        runner.get_param_tensors(tensors);
        return register_param_tensors(desc,
                                      std::move(tensors),
                                      residency_mode,
                                      compute_backend,
                                      params_backend,
                                      registered_tensor_size);
    }

    template <typename Runner>
    bool register_runner_params(const std::string& desc,
                                Runner& runner,
                                const std::string& prefix,
                                ResidencyMode residency_mode,
                                ggml_backend_t compute_backend,
                                ggml_backend_t params_backend,
                                size_t* registered_tensor_size = nullptr) {
        std::map<std::string, ggml_tensor*> tensors;
        runner.get_param_tensors(tensors, prefix);
        return register_param_tensors(desc,
                                      std::move(tensors),
                                      residency_mode,
                                      compute_backend,
                                      params_backend,
                                      registered_tensor_size);
    }

    bool validate_registered_tensors();
    bool load_all_params_eagerly();

    // A request/window boundary is stronger than a normal graph completion: no
    // runner may still own a staged compute buffer, while CPU/mmap parameter
    // storage must remain intact for the next window.  This deliberately does
    // not touch params_storage_blocks_.
    void reclaim_transient_compute_buffers();

    bool assign_compute_backend(const std::vector<ggml_tensor*>& tensors,
                                ggml_backend_t compute_backend) override;
    bool prepare_params(const std::vector<ggml_tensor*>& tensors) override;
    bool retain_compute_backend_params(const std::vector<ggml_tensor*>& tensors) override;
    void release_compute_backend_params(const std::vector<ggml_tensor*>& tensors) override;
    void release_retained_compute_backend_params(const std::vector<ggml_tensor*>& tensors) override;
    void release_params_backend_params(const std::vector<ggml_tensor*>& tensors) override;
};

#endif  // __MODEL_MANAGER_H__
