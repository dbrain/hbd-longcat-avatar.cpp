#ifndef __MODEL_LOADER_H__
#define __MODEL_LOADER_H__

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "model.h"

TensorTypeRules parse_tensor_type_rules(const std::string& tensor_type_rules);

class MmapWrapper;

struct ModelFileData {
    std::string path;
    std::vector<TensorStorage> tensors;
    std::shared_ptr<MmapWrapper> mmapped;
    std::shared_ptr<struct ggml_backend_buffer> mmbuffer;
    bool is_zip;
    // Mapped PROT_WRITE, i.e. anything may have written through it in place. Only such a
    // mapping needs (or can be) restored -- see ModelLoader::restore_mmapped_bytes.
    bool writable_mmap = false;
};

struct MmapTensorStore {
    std::shared_ptr<MmapWrapper> mmapped;
    std::shared_ptr<struct ggml_backend_buffer> mmbuffer;
};

bool is_unused_tensor(const std::string& name);

class ModelLoader {
protected:
    SDVersion version_ = VERSION_COUNT;
    std::vector<std::string> file_paths_;
    std::vector<ModelFileData> file_data;
    bool model_files_processed = false;
    String2TensorStorage tensor_storage_map;
    std::map<std::string, std::string> metadata_;
    int n_threads_;

    size_t add_file_path(const std::string& file_path);
    void add_tensor_storage(const TensorStorage& tensor_storage);

    bool init_from_gguf_file(const std::string& file_path, const std::string& prefix = "");
    bool init_from_safetensors_file(const std::string& file_path, const std::string& prefix = "");
    bool init_from_safetensors_index_file(const std::string& file_path, const std::string& prefix = "");
    bool init_from_torch_zip_file(const std::string& file_path, const std::string& prefix = "");
    bool init_from_torch_legacy_file(const std::string& file_path, const std::string& prefix = "");
    bool init_from_diffusers_file(const std::string& file_path, const std::string& prefix = "");

public:
    ModelLoader();

    bool init_from_file(const std::string& file_path, const std::string& prefix = "");
    void convert_tensors_name();
    bool init_from_file_and_convert_name(const std::string& file_path,
                                         const std::string& prefix = "",
                                         SDVersion version         = VERSION_COUNT);
    SDVersion get_sd_version();
    std::map<ggml_type, uint32_t> get_wtype_stat();
    std::map<ggml_type, uint32_t> get_conditioner_wtype_stat();
    std::map<ggml_type, uint32_t> get_diffusion_model_wtype_stat();
    std::map<ggml_type, uint32_t> get_vae_wtype_stat();
    String2TensorStorage& get_tensor_storage_map() { return tensor_storage_map; }
    const String2TensorStorage& get_tensor_storage_map() const { return tensor_storage_map; }
    const std::map<std::string, std::string>& get_metadata() const { return metadata_; }
    void set_n_threads(int n_threads);
    void set_wtype_override(ggml_type wtype, std::string tensor_type_rules = "");
    void process_model_files(bool enable_mmap = false, bool writable_mmap = true);
    std::vector<MmapTensorStore> mmap_tensors(std::map<std::string, ggml_tensor*>& tensors,
                                              std::set<std::string> ignore_tensors = {},
                                              bool writable                        = true);

    // Restore [ptr, ptr + bytes) to the on-disk bytes, when ptr points into one of the
    // writable MAP_PRIVATE model mappings this loader owns. Returns false if the range
    // is not in any mapping, or the platform cannot discard private pages.
    //
    // 🔴 This exists because RELEASING A PARAMS STORAGE BLOCK DOES NOT UNDO AN IN-PLACE
    // WRITE. The mapping is created once by process_model_files() (latched on
    // model_files_processed) and owned by `file_data` for the loader's whole life;
    // MmapTensorStore holds only shared_ptr COPIES of it, so
    // ModelManager::free_params_storage_block()'s `mmap_tensor_stores.clear()` drops the
    // refcount 2 -> 1 and never munmaps. The next mmap_tensors() then re-points
    // tensor->data at the SAME copy-on-write pages, mutations and all.
    bool restore_mmapped_bytes(const void* ptr, size_t bytes);

    // True if this loader has at least one writable mapping whose in-place writes could
    // survive a params-storage release. The LoRA fold checks this and fails closed.
    bool has_writable_mmap() const;
    bool load_tensors(on_new_tensor_cb_t on_new_tensor_cb,
                      bool use_mmap                                    = false,
                      const std::set<std::string>* target_tensor_names = nullptr,
                      bool log_progress                                = true);
    bool load_tensors(std::map<std::string, ggml_tensor*>& tensors,
                      std::set<std::string> ignore_tensors = {},
                      bool use_mmap                        = false);
    bool load_float_tensor(const std::string& name,
                           std::vector<float>& data,
                           int n_threads = 0,
                           bool use_mmap = false);
    bool load_tensor(const TensorStorage& tensor_storage, ggml_tensor* dst_tensor);

    std::vector<std::string> get_tensor_names() const {
        std::vector<std::string> names;
        for (const auto& [name, tensor_storage] : tensor_storage_map) {
            names.push_back(name);
        }
        return names;
    }

    bool tensor_should_be_converted(const TensorStorage& tensor_storage, ggml_type type);
    int64_t get_params_mem_size(ggml_backend_t backend, ggml_type type = GGML_TYPE_COUNT);
    ~ModelLoader() = default;
};

#endif  // __MODEL_LOADER_H__
