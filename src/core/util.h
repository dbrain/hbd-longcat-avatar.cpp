#ifndef __SD_CORE_UTIL_H__
#define __SD_CORE_UTIL_H__

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/tensor.hpp"
#include "ggml-backend.h"
#include "stable-diffusion.h"

#define SAFE_STR(s) ((s) ? (s) : "")
#define BOOL_STR(b) ((b) ? "true" : "false")

bool ends_with(const std::string& str, const std::string& ending);
bool starts_with(const std::string& str, const std::string& start);
bool contains(const std::string& str, const std::string& substr);

std::string sd_format(const char* fmt, ...);

void replace_all_chars(std::string& str, char target, char replacement);

int round_up_to(int value, int base);

bool file_exists(const std::string& filename);
bool is_directory(const std::string& path);

std::u32string utf8_to_utf32(const std::string& utf8_str);
std::string utf32_to_utf8(const std::u32string& utf32_str);
std::u32string unicode_value_to_utf32(int unicode_value);
// std::string sd_basename(const std::string& path);

sd_image_t tensor_to_sd_image(const sd::Tensor<float>& tensor, int frame_index = 0);

sd::Tensor<float> sd_image_to_tensor(sd_image_t image,
                                     int target_width  = -1,
                                     int target_height = -1,
                                     bool scale        = true);

sd::Tensor<float> clip_preprocess(const sd::Tensor<float>& image, int target_width, int target_height);

class MmapWrapper {
public:
    static std::unique_ptr<MmapWrapper> create(const std::string& filename, bool writable = false);

    virtual ~MmapWrapper() = default;

    MmapWrapper(const MmapWrapper&)            = delete;
    MmapWrapper& operator=(const MmapWrapper&) = delete;
    MmapWrapper(MmapWrapper&&)                 = delete;
    MmapWrapper& operator=(MmapWrapper&&)      = delete;

    const uint8_t* data() const { return static_cast<uint8_t*>(data_); }
    uint8_t* writable_data() { return static_cast<uint8_t*>(data_); }
    size_t size() const { return size_; }
    bool copy_data(void* buf, size_t n, size_t offset) const;

    // True if discard_private_writes() is implemented on this platform. The LoRA fold
    // refuses to run without it: a fold it cannot undo is silent cross-request corruption.
    static bool supports_discard_private_writes();

    // Throw away this process's private (copy-on-write) copies of the pages covering
    // [offset, offset + bytes), so the next access re-reads the ON-DISK bytes. This is
    // the ONLY way to undo an in-place write to a MAP_PRIVATE file mapping short of
    // unmapping it, and it is what makes the LoRA fold reversible.
    //
    // The range is rounded OUTWARD to whole pages, which is required for correctness (a
    // partial head/tail page rounded inward would keep its mutated bytes) and harmless:
    // every neighbouring byte in those pages either belongs to another folded tensor,
    // which is being restored too, or is unmodified and therefore already equal to the
    // file. MAP_PRIVATE means nothing was ever written back, so "the file" is pristine.
    bool discard_private_writes(size_t offset, size_t bytes);

protected:
    MmapWrapper(void* data, size_t size)
        : data_(data), size_(size) {}
    void* data_  = nullptr;
    size_t size_ = 0;
};

std::string path_join(const std::string& p1, const std::string& p2);
std::vector<std::string> split_string(const std::string& str, char delimiter);

using KeyValueArgs = std::vector<std::pair<std::string, std::string>>;

KeyValueArgs parse_key_value_args(const char* args, const char* context = "key=value arg");
KeyValueArgs parse_key_value_args(const std::string& args, const char* context = "key=value arg");
bool parse_strict_float(const std::string& text, float& value);
bool parse_strict_int(const std::string& text, int& value);
bool parse_strict_bool(const std::string& text, bool& value);

void pretty_progress(int step, int steps, float time);
void pretty_bytes_progress(int step, int steps, uint64_t bytes_processed, float elapsed_seconds);

void log_printf(sd_log_level_t level, const char* file, int line, const char* format, ...);

ggml_type sd_type_to_ggml_type(sd_type_t sdtype);

std::string trim(const std::string& s);

std::vector<std::pair<std::string, float>> parse_prompt_attention(const std::string& text);
std::vector<std::pair<std::string, float>> split_quotation_attention(
    const std::vector<std::pair<std::string, float>>& parsed_attention);

sd_progress_cb_t sd_get_progress_callback();
void* sd_get_progress_callback_data();

sd_preview_cb_t sd_get_preview_callback();
void* sd_get_preview_callback_data();
preview_t sd_get_preview_mode();
int sd_get_preview_interval();
bool sd_should_preview_denoised();
bool sd_should_preview_noisy();

sd_graph_eval_callback_t sd_get_backend_eval_callback();
void* sd_get_backend_eval_callback_data();

// Names the runner whose graph is currently executing, so SD_NODE_TRACE can be scoped to one
// stage (e.g. the DiT) instead of burning its node budget on the text/audio encoders.
void sd_set_trace_runner_desc(const char* desc);

// test if the backend is a specific one, e.g. "CUDA", "ROCm", "Vulkan" etc.
bool sd_backend_is(ggml_backend_t backend, const std::string& name);

#define LOG_DEBUG(format, ...) log_printf(SD_LOG_DEBUG, __FILE__, __LINE__, format, ##__VA_ARGS__)
#define LOG_INFO(format, ...) log_printf(SD_LOG_INFO, __FILE__, __LINE__, format, ##__VA_ARGS__)
#define LOG_WARN(format, ...) log_printf(SD_LOG_WARN, __FILE__, __LINE__, format, ##__VA_ARGS__)
#define LOG_ERROR(format, ...) log_printf(SD_LOG_ERROR, __FILE__, __LINE__, format, ##__VA_ARGS__)
#endif  // __SD_CORE_UTIL_H__
