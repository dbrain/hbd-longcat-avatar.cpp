#include "media_io.h"
#include "log.h"
#include "resource_owners.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "stb_image_write.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_STATIC
#include "stb_image_resize.h"

#ifdef SD_USE_WEBP
#include "webp/decode.h"
#include "webp/encode.h"
#include "webp/mux.h"
#endif

#ifdef SD_USE_WEBM
#include "mkvmuxer/mkvmuxer.h"
#include "mkvmuxer/mkvwriter.h"
#ifdef SD_USE_OPUS
#include <opus/opus.h>
#endif
#ifdef SD_USE_VPX
#include <vpx/vp8cx.h>
#include <vpx/vpx_encoder.h>
#endif
#endif

namespace fs = std::filesystem;

#ifdef SD_USE_WEBP
struct WebPFreeDeleter {
    void operator()(void* ptr) const {
        if (ptr != nullptr) {
            WebPFree(ptr);
        }
    }
};

struct WebPMuxDeleter {
    void operator()(WebPMux* mux) const {
        if (mux != nullptr) {
            WebPMuxDelete(mux);
        }
    }
};

struct WebPAnimEncoderDeleter {
    void operator()(WebPAnimEncoder* enc) const {
        if (enc != nullptr) {
            WebPAnimEncoderDelete(enc);
        }
    }
};

struct WebPDataGuard {
    WebPDataGuard() {
        WebPDataInit(&data);
    }

    ~WebPDataGuard() {
        WebPDataClear(&data);
    }

    WebPData data;
};

struct WebPPictureGuard {
    WebPPictureGuard()
        : initialized(WebPPictureInit(&picture) != 0) {
    }

    ~WebPPictureGuard() {
        if (initialized) {
            WebPPictureFree(&picture);
        }
    }

    WebPPicture picture;
    bool initialized;
};

using WebPBufferPtr      = std::unique_ptr<uint8_t, WebPFreeDeleter>;
using WebPMuxPtr         = std::unique_ptr<WebPMux, WebPMuxDeleter>;
using WebPAnimEncoderPtr = std::unique_ptr<WebPAnimEncoder, WebPAnimEncoderDeleter>;
#endif

#ifdef SD_USE_WEBM
class MemoryMkvWriter : public mkvmuxer::IMkvWriter {
public:
    mkvmuxer::int32 Write(const void* buf, mkvmuxer::uint32 len) override {
        if (buf == nullptr && len > 0) {
            return -1;
        }
        const size_t end_pos = position_ + static_cast<size_t>(len);
        if (end_pos > data_.size()) {
            data_.resize(end_pos);
        }
        if (len > 0) {
            memcpy(data_.data() + position_, buf, len);
        }
        position_ = end_pos;
        return 0;
    }

    mkvmuxer::int64 Position() const override {
        return static_cast<mkvmuxer::int64>(position_);
    }

    mkvmuxer::int32 Position(mkvmuxer::int64 position) override {
        if (position < 0) {
            return -1;
        }
        const size_t target = static_cast<size_t>(position);
        if (target > data_.size()) {
            data_.resize(target);
        }
        position_ = target;
        return 0;
    }

    bool Seekable() const override {
        return true;
    }

    void ElementStartNotify(mkvmuxer::uint64, mkvmuxer::int64) override {
    }

    const std::vector<uint8_t>& data() const {
        return data_;
    }

private:
    std::vector<uint8_t> data_;
    size_t position_ = 0;
};
#endif

bool read_binary_file_bytes(const char* path, std::vector<uint8_t>& data) {
    std::ifstream fin(fs::path(path), std::ios::binary);
    if (!fin) {
        return false;
    }

    fin.seekg(0, std::ios::end);
    std::streampos size = fin.tellg();
    if (size < 0) {
        return false;
    }
    fin.seekg(0, std::ios::beg);

    data.resize(static_cast<size_t>(size));
    if (!data.empty()) {
        fin.read(reinterpret_cast<char*>(data.data()), size);
        if (!fin) {
            return false;
        }
    }
    return true;
}

bool write_binary_file_bytes(const std::string& path, const std::vector<uint8_t>& data) {
    std::ofstream fout(fs::path(path), std::ios::binary);
    if (!fout) {
        return false;
    }

    if (!data.empty()) {
        fout.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
        if (!fout) {
            return false;
        }
    }
    return true;
}

uint32_t read_u32_le_bytes(const uint8_t* data) {
    return static_cast<uint32_t>(data[0]) |
           (static_cast<uint32_t>(data[1]) << 8) |
           (static_cast<uint32_t>(data[2]) << 16) |
           (static_cast<uint32_t>(data[3]) << 24);
}

int stbi_ext_write_png_to_func(stbi_write_func* func,
                               void* context,
                               int x,
                               int y,
                               int comp,
                               const void* data,
                               int stride_bytes,
                               const char* parameters) {
    int len            = 0;
    unsigned char* png = stbi_write_png_to_mem((const unsigned char*)data, stride_bytes, x, y, comp, &len, parameters);
    if (png == nullptr) {
        return 0;
    }
    func(context, png, len);
    STBIW_FREE(png);
    return 1;
}

bool is_webp_signature(const uint8_t* data, size_t size) {
    return size >= 12 &&
           memcmp(data, "RIFF", 4) == 0 &&
           memcmp(data + 8, "WEBP", 4) == 0;
}

std::string xml_escape(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());

    for (char ch : value) {
        switch (ch) {
            case '&':
                escaped += "&amp;";
                break;
            case '<':
                escaped += "&lt;";
                break;
            case '>':
                escaped += "&gt;";
                break;
            case '"':
                escaped += "&quot;";
                break;
            case '\'':
                escaped += "&apos;";
                break;
            default:
                escaped += ch;
                break;
        }
    }

    return escaped;
}

#ifdef SD_USE_WEBP
uint8_t* decode_webp_image_to_buffer(const uint8_t* data,
                                     size_t size,
                                     int& width,
                                     int& height,
                                     int expected_channel,
                                     int& source_channel_count) {
    WebPBitstreamFeatures features;
    if (WebPGetFeatures(data, size, &features) != VP8_STATUS_OK) {
        return nullptr;
    }

    width                = features.width;
    height               = features.height;
    source_channel_count = features.has_alpha ? 4 : 3;

    const size_t pixel_count = static_cast<size_t>(width) * static_cast<size_t>(height);

    if (expected_channel == 1) {
        int decoded_width  = width;
        int decoded_height = height;
        WebPBufferPtr decoded(features.has_alpha
                                  ? WebPDecodeRGBA(data, size, &decoded_width, &decoded_height)
                                  : WebPDecodeRGB(data, size, &decoded_width, &decoded_height));
        if (decoded == nullptr) {
            return nullptr;
        }

        FreeUniquePtr<uint8_t> grayscale((uint8_t*)malloc(pixel_count));
        if (grayscale == nullptr) {
            return nullptr;
        }

        const int decoded_channels = features.has_alpha ? 4 : 3;
        for (size_t i = 0; i < pixel_count; ++i) {
            const uint8_t* src = decoded.get() + i * decoded_channels;
            grayscale.get()[i] = static_cast<uint8_t>((77 * src[0] + 150 * src[1] + 29 * src[2] + 128) >> 8);
        }

        return grayscale.release();
    }

    if (expected_channel != 3 && expected_channel != 4) {
        return nullptr;
    }

    int decoded_width  = width;
    int decoded_height = height;
    WebPBufferPtr decoded((expected_channel == 4)
                              ? WebPDecodeRGBA(data, size, &decoded_width, &decoded_height)
                              : WebPDecodeRGB(data, size, &decoded_width, &decoded_height));
    if (decoded == nullptr) {
        return nullptr;
    }

    const size_t out_size = pixel_count * static_cast<size_t>(expected_channel);
    FreeUniquePtr<uint8_t> output((uint8_t*)malloc(out_size));
    if (output == nullptr) {
        return nullptr;
    }

    memcpy(output.get(), decoded.get(), out_size);
    return output.release();
}

std::string build_webp_xmp_packet(const std::string& parameters) {
    if (parameters.empty()) {
        return "";
    }

    const std::string escaped_parameters = xml_escape(parameters);
    return "<?xpacket begin=\"\" id=\"W5M0MpCehiHzreSzNTczkc9d\"?>\n"
           "<x:xmpmeta xmlns:x=\"adobe:ns:meta/\">\n"
           "  <rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n"
           "    <rdf:Description xmlns:sdcpp=\"https://github.com/leejet/stable-diffusion.cpp/ns/1.0/\">\n"
           "      <sdcpp:parameters>" +
           escaped_parameters +
           "</sdcpp:parameters>\n"
           "    </rdf:Description>\n"
           "  </rdf:RDF>\n"
           "</x:xmpmeta>\n"
           "<?xpacket end=\"w\"?>";
}

bool encode_webp_image_to_vector(const uint8_t* image,
                                 int width,
                                 int height,
                                 int channels,
                                 const std::string& parameters,
                                 int quality,
                                 std::vector<uint8_t>& out) {
    if (image == nullptr || width <= 0 || height <= 0) {
        return false;
    }

    std::vector<uint8_t> rgb_image;
    const uint8_t* input_image = image;
    int input_channels         = channels;

    if (channels == 1) {
        rgb_image.resize(static_cast<size_t>(width) * static_cast<size_t>(height) * 3);
        for (int i = 0; i < width * height; ++i) {
            rgb_image[i * 3 + 0] = image[i];
            rgb_image[i * 3 + 1] = image[i];
            rgb_image[i * 3 + 2] = image[i];
        }
        input_image    = rgb_image.data();
        input_channels = 3;
    }

    if (input_channels != 3 && input_channels != 4) {
        return false;
    }

    uint8_t* encoded_raw = nullptr;
    size_t encoded_size  = (input_channels == 4)
                               ? WebPEncodeRGBA(input_image, width, height, width * input_channels, static_cast<float>(quality), &encoded_raw)
                               : WebPEncodeRGB(input_image, width, height, width * input_channels, static_cast<float>(quality), &encoded_raw);
    WebPBufferPtr encoded(encoded_raw);
    if (encoded == nullptr || encoded_size == 0) {
        return false;
    }

    out.assign(encoded.get(), encoded.get() + encoded_size);

    if (parameters.empty()) {
        return true;
    }

    WebPData image_data;
    WebPDataInit(&image_data);
    WebPDataGuard assembled_data;

    image_data.bytes = out.data();
    image_data.size  = out.size();

    WebPMuxPtr mux(WebPMuxNew());
    if (mux == nullptr) {
        return false;
    }

    const std::string xmp_packet = build_webp_xmp_packet(parameters);
    WebPData xmp_data;
    WebPDataInit(&xmp_data);
    xmp_data.bytes = reinterpret_cast<const uint8_t*>(xmp_packet.data());
    xmp_data.size  = xmp_packet.size();

    const bool ok = WebPMuxSetImage(mux.get(), &image_data, 1) == WEBP_MUX_OK &&
                    WebPMuxSetChunk(mux.get(), "XMP ", &xmp_data, 1) == WEBP_MUX_OK &&
                    WebPMuxAssemble(mux.get(), &assembled_data.data) == WEBP_MUX_OK;

    if (ok) {
        out.assign(assembled_data.data.bytes, assembled_data.data.bytes + assembled_data.data.size);
    }

    return ok;
}

#ifdef SD_USE_WEBM
bool extract_vp8_frame_from_webp(const std::vector<uint8_t>& webp_data, std::vector<uint8_t>& vp8_frame) {
    if (!is_webp_signature(webp_data.data(), webp_data.size())) {
        return false;
    }

    size_t offset = 12;
    while (offset + 8 <= webp_data.size()) {
        const uint8_t* chunk     = webp_data.data() + offset;
        const uint32_t chunk_len = read_u32_le_bytes(chunk + 4);
        const size_t chunk_start = offset + 8;
        const size_t padded_len  = static_cast<size_t>(chunk_len) + (chunk_len & 1u);

        if (chunk_start + chunk_len > webp_data.size()) {
            return false;
        }

        if (memcmp(chunk, "VP8 ", 4) == 0) {
            vp8_frame.assign(webp_data.data() + chunk_start,
                             webp_data.data() + chunk_start + chunk_len);
            return !vp8_frame.empty();
        }

        offset = chunk_start + padded_len;
    }

    return false;
}

bool encode_sd_image_to_vp8_frame(const sd_image_t& image, int quality, std::vector<uint8_t>& vp8_frame) {
    if (image.data == nullptr || image.width == 0 || image.height == 0) {
        return false;
    }

    const int width         = static_cast<int>(image.width);
    const int height        = static_cast<int>(image.height);
    const int input_channel = static_cast<int>(image.channel);
    if (input_channel != 1 && input_channel != 3 && input_channel != 4) {
        return false;
    }

    std::vector<uint8_t> rgb_buffer;
    const uint8_t* rgb_data = image.data;
    if (input_channel == 1) {
        rgb_buffer.resize(static_cast<size_t>(width) * static_cast<size_t>(height) * 3);
        for (int i = 0; i < width * height; ++i) {
            rgb_buffer[i * 3 + 0] = image.data[i];
            rgb_buffer[i * 3 + 1] = image.data[i];
            rgb_buffer[i * 3 + 2] = image.data[i];
        }
        rgb_data = rgb_buffer.data();
    } else if (input_channel == 4) {
        rgb_buffer.resize(static_cast<size_t>(width) * static_cast<size_t>(height) * 3);
        for (int i = 0; i < width * height; ++i) {
            rgb_buffer[i * 3 + 0] = image.data[i * 4 + 0];
            rgb_buffer[i * 3 + 1] = image.data[i * 4 + 1];
            rgb_buffer[i * 3 + 2] = image.data[i * 4 + 2];
        }
        rgb_data = rgb_buffer.data();
    }

    std::vector<uint8_t> encoded_webp;
    if (!encode_webp_image_to_vector(rgb_data, width, height, 3, "", quality, encoded_webp)) {
        return false;
    }

    return extract_vp8_frame_from_webp(encoded_webp, vp8_frame);
}
#endif
#endif

uint8_t* load_image_common(bool from_memory,
                           const char* image_path_or_bytes,
                           int len,
                           int& width,
                           int& height,
                           int expected_width,
                           int expected_height,
                           int expected_channel) {
    const char* image_path;
    FreeUniquePtr<uint8_t> image_buffer;
    int source_channel_count = 0;

#ifdef SD_USE_WEBP
    if (from_memory) {
        image_path = "memory";
        if (len > 0 && is_webp_signature(reinterpret_cast<const uint8_t*>(image_path_or_bytes), static_cast<size_t>(len))) {
            image_buffer.reset(decode_webp_image_to_buffer(reinterpret_cast<const uint8_t*>(image_path_or_bytes),
                                                           static_cast<size_t>(len),
                                                           width,
                                                           height,
                                                           expected_channel,
                                                           source_channel_count));
        }
    } else {
        image_path = image_path_or_bytes;
        if (encoded_image_format_from_path(image_path_or_bytes) == EncodedImageFormat::WEBP) {
            std::vector<uint8_t> file_bytes;
            if (!read_binary_file_bytes(image_path_or_bytes, file_bytes)) {
                LOG_ERROR("load image from '%s' failed", image_path_or_bytes);
                return nullptr;
            }
            if (!is_webp_signature(file_bytes.data(), file_bytes.size())) {
                LOG_ERROR("load image from '%s' failed", image_path_or_bytes);
                return nullptr;
            }
            image_buffer.reset(decode_webp_image_to_buffer(file_bytes.data(),
                                                           file_bytes.size(),
                                                           width,
                                                           height,
                                                           expected_channel,
                                                           source_channel_count));
        }
    }
#endif

    if (from_memory) {
        image_path = "memory";
        if (image_buffer == nullptr) {
            int c = 0;
            image_buffer.reset((uint8_t*)stbi_load_from_memory((const stbi_uc*)image_path_or_bytes, len, &width, &height, &c, expected_channel));
            source_channel_count = c;
        }
    } else {
        image_path = image_path_or_bytes;
        if (image_buffer == nullptr) {
            int c = 0;
            image_buffer.reset((uint8_t*)stbi_load(image_path_or_bytes, &width, &height, &c, expected_channel));
            source_channel_count = c;
        }
    }
    if (image_buffer == nullptr) {
        LOG_ERROR("load image from '%s' failed", image_path);
        return nullptr;
    }
    if (source_channel_count < expected_channel) {
        fprintf(stderr,
                "the number of channels for the input image must be >= %d,"
                "but got %d channels, image_path = %s",
                expected_channel,
                source_channel_count,
                image_path);
        return nullptr;
    }
    if (width <= 0) {
        LOG_ERROR("error: the width of image must be greater than 0, image_path = %s", image_path);
        return nullptr;
    }
    if (height <= 0) {
        LOG_ERROR("error: the height of image must be greater than 0, image_path = %s", image_path);
        return nullptr;
    }

    if ((expected_width > 0 && expected_height > 0) && (height != expected_height || width != expected_width)) {
        float dst_aspect = (float)expected_width / (float)expected_height;
        float src_aspect = (float)width / (float)height;

        int crop_x = 0, crop_y = 0;
        int crop_w = width, crop_h = height;

        if (src_aspect > dst_aspect) {
            crop_w = (int)(height * dst_aspect);
            crop_x = (width - crop_w) / 2;
        } else if (src_aspect < dst_aspect) {
            crop_h = (int)(width / dst_aspect);
            crop_y = (height - crop_h) / 2;
        }

        if (crop_x != 0 || crop_y != 0) {
            LOG_INFO("crop input image from %dx%d to %dx%d, image_path = %s", width, height, crop_w, crop_h, image_path);
            FreeUniquePtr<uint8_t> cropped_image_buffer((uint8_t*)malloc(crop_w * crop_h * expected_channel));
            if (cropped_image_buffer == nullptr) {
                LOG_ERROR("error: allocate memory for crop\n");
                return nullptr;
            }
            for (int row = 0; row < crop_h; row++) {
                uint8_t* src = image_buffer.get() + ((crop_y + row) * width + crop_x) * expected_channel;
                uint8_t* dst = cropped_image_buffer.get() + (row * crop_w) * expected_channel;
                memcpy(dst, src, crop_w * expected_channel);
            }

            width        = crop_w;
            height       = crop_h;
            image_buffer = std::move(cropped_image_buffer);
        }

        LOG_INFO("resize input image from %dx%d to %dx%d", width, height, expected_width, expected_height);
        FreeUniquePtr<uint8_t> resized_image_buffer((uint8_t*)malloc(expected_height * expected_width * expected_channel));
        if (resized_image_buffer == nullptr) {
            LOG_ERROR("error: allocate memory for resize input image\n");
            return nullptr;
        }
        stbir_resize(image_buffer.get(), width, height, 0,
                     resized_image_buffer.get(), expected_width, expected_height, 0, STBIR_TYPE_UINT8,
                     expected_channel, STBIR_ALPHA_CHANNEL_NONE, 0,
                     STBIR_EDGE_CLAMP, STBIR_EDGE_CLAMP,
                     STBIR_FILTER_BOX, STBIR_FILTER_BOX,
                     STBIR_COLORSPACE_SRGB, nullptr);
        width        = expected_width;
        height       = expected_height;
        image_buffer = std::move(resized_image_buffer);
    }
    return image_buffer.release();
}

typedef struct {
    uint32_t offset;
    uint32_t size;
} avi_index_entry;

typedef struct {
    char fourcc[4];
    uint32_t flags;
    uint32_t offset;
    uint32_t size;
} avi_chunk_index_entry;

void write_u32_le(FILE* f, uint32_t val) {
    fwrite(&val, 4, 1, f);
}

void write_u16_le(FILE* f, uint16_t val) {
    fwrite(&val, 2, 1, f);
}

void write_u32_le(std::vector<uint8_t>& data, uint32_t val) {
    data.push_back(static_cast<uint8_t>(val & 0xFF));
    data.push_back(static_cast<uint8_t>((val >> 8) & 0xFF));
    data.push_back(static_cast<uint8_t>((val >> 16) & 0xFF));
    data.push_back(static_cast<uint8_t>((val >> 24) & 0xFF));
}

void write_u16_le(std::vector<uint8_t>& data, uint16_t val) {
    data.push_back(static_cast<uint8_t>(val & 0xFF));
    data.push_back(static_cast<uint8_t>((val >> 8) & 0xFF));
}

void patch_u32_le(std::vector<uint8_t>& data, size_t offset, uint32_t val) {
    if (offset + 4 > data.size()) {
        return;
    }
    data[offset + 0] = static_cast<uint8_t>(val & 0xFF);
    data[offset + 1] = static_cast<uint8_t>((val >> 8) & 0xFF);
    data[offset + 2] = static_cast<uint8_t>((val >> 16) & 0xFF);
    data[offset + 3] = static_cast<uint8_t>((val >> 24) & 0xFF);
}

void write_fourcc(std::vector<uint8_t>& data, const char* fourcc) {
    data.insert(data.end(), fourcc, fourcc + 4);
}

static std::vector<uint8_t> audio_to_pcm16_bytes(const sd_audio_t* audio) {
    if (audio == nullptr || audio->data == nullptr || audio->sample_count == 0 || audio->channels == 0 || audio->sample_rate == 0) {
        return {};
    }

    const size_t pcm_samples = static_cast<size_t>(audio->sample_count) * static_cast<size_t>(audio->channels);
    std::vector<uint8_t> bytes(pcm_samples * sizeof(int16_t));
    auto* pcm = reinterpret_cast<int16_t*>(bytes.data());
    for (size_t i = 0; i < pcm_samples; ++i) {
        const float sample = std::clamp(audio->data[i], -1.0f, 1.0f);
        pcm[i]             = static_cast<int16_t>(std::lrint(sample * 32767.0f));
    }
    return bytes;
}

static std::pair<uint64_t, uint64_t> audio_sample_range_for_video_frame(const sd_audio_t* audio, int frame_idx, int num_frames, int fps) {
    if (audio == nullptr || fps <= 0 || num_frames <= 0) {
        return {0, 0};
    }
    const uint64_t total = audio->sample_count;
    const uint64_t start = static_cast<uint64_t>((static_cast<long double>(frame_idx) * total) / num_frames);
    const uint64_t end   = frame_idx + 1 == num_frames
                               ? total
                               : static_cast<uint64_t>((static_cast<long double>(frame_idx + 1) * total) / num_frames);
    return {start, std::max(start, end)};
}

EncodedImageFormat encoded_image_format_from_path(const std::string& path) {
    std::string ext = fs::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext == ".jpg" || ext == ".jpeg" || ext == ".jpe") {
        return EncodedImageFormat::JPEG;
    }
    if (ext == ".png") {
        return EncodedImageFormat::PNG;
    }
    if (ext == ".webp") {
        return EncodedImageFormat::WEBP;
    }
    return EncodedImageFormat::UNKNOWN;
}

std::vector<uint8_t> encode_image_to_vector(EncodedImageFormat format,
                                            const uint8_t* image,
                                            int width,
                                            int height,
                                            int channels,
                                            const std::string& parameters,
                                            int quality) {
    std::vector<uint8_t> buffer;

    auto write_func = [&buffer](void* context, void* data, int size) {
        (void)context;
        uint8_t* src = reinterpret_cast<uint8_t*>(data);
        buffer.insert(buffer.end(), src, src + size);
    };

    struct ContextWrapper {
        decltype(write_func)& func;
    } ctx{write_func};

    auto c_func = [](void* context, void* data, int size) {
        auto* wrapper = reinterpret_cast<ContextWrapper*>(context);
        wrapper->func(context, data, size);
    };

    int result = 0;
    switch (format) {
        case EncodedImageFormat::JPEG:
            result = stbi_write_jpg_to_func(c_func, &ctx, width, height, channels, image, quality);
            break;
        case EncodedImageFormat::PNG:
            result = stbi_ext_write_png_to_func(c_func, &ctx, width, height, channels, image, width * channels, parameters.empty() ? nullptr : parameters.c_str());
            break;
        case EncodedImageFormat::WEBP:
#ifdef SD_USE_WEBP
            if (!encode_webp_image_to_vector(image, width, height, channels, parameters, quality, buffer)) {
                buffer.clear();
            }
            result = buffer.empty() ? 0 : 1;
            break;
#else
            result = 0;
            break;
#endif
        default:
            result = 0;
            break;
    }

    if (!result) {
        buffer.clear();
    }
    return buffer;
}

bool write_image_to_file(const std::string& path,
                         const uint8_t* image,
                         int width,
                         int height,
                         int channels,
                         const std::string& parameters,
                         int quality) {
    const EncodedImageFormat format = encoded_image_format_from_path(path);

    switch (format) {
        case EncodedImageFormat::JPEG:
            return stbi_write_jpg(path.c_str(), width, height, channels, image, quality, parameters.empty() ? nullptr : parameters.c_str()) != 0;
        case EncodedImageFormat::PNG:
            return stbi_write_png(path.c_str(), width, height, channels, image, 0, parameters.empty() ? nullptr : parameters.c_str()) != 0;
        case EncodedImageFormat::WEBP: {
            const std::vector<uint8_t> encoded = encode_image_to_vector(format, image, width, height, channels, parameters, quality);
            return !encoded.empty() && write_binary_file_bytes(path, encoded);
        }
        default:
            return false;
    }
}

uint8_t* load_image_from_file(const char* image_path,
                              int& width,
                              int& height,
                              int expected_width,
                              int expected_height,
                              int expected_channel) {
    return load_image_common(false, image_path, 0, width, height, expected_width, expected_height, expected_channel);
}

bool load_sd_image_from_file(sd_image_t* image,
                             const char* image_path,
                             int expected_width,
                             int expected_height,
                             int expected_channel) {
    int width;
    int height;
    image->data = load_image_common(false, image_path, 0, width, height, expected_width, expected_height, expected_channel);
    if (image->data == nullptr) {
        return false;
    }
    image->width   = width;
    image->height  = height;
    image->channel = expected_channel;
    return true;
}

uint8_t* load_image_from_memory(const char* image_bytes,
                                int len,
                                int& width,
                                int& height,
                                int expected_width,
                                int expected_height,
                                int expected_channel) {
    return load_image_common(true, image_bytes, len, width, height, expected_width, expected_height, expected_channel);
}

std::vector<uint8_t> create_mjpg_avi_from_sd_images_to_vector(sd_image_t* images, int num_images, int fps, int quality, const sd_audio_t* audio) {
    if (num_images == 0) {
        fprintf(stderr, "Error: Image array is empty.\n");
        return {};
    }

    uint32_t width    = images[0].width;
    uint32_t height   = images[0].height;
    uint32_t channels = images[0].channel;
    if (channels != 3 && channels != 4) {
        fprintf(stderr, "Error: Unsupported channel count: %u\n", channels);
        return {};
    }

    // stb_image_write changes JPEG sampling behavior above quality 90.
    // MJPG AVI playback is more compatible when we keep the encoder on the
    // <= 90 path.
    const int mjpg_quality               = std::clamp(quality, 1, 90);
    const bool has_audio                 = audio != nullptr && audio->data != nullptr && audio->sample_count > 0 && audio->channels > 0 && audio->sample_rate > 0;
    const std::vector<uint8_t> audio_pcm = audio_to_pcm16_bytes(audio);
    const uint16_t audio_bits_per_sample = 16;
    const uint16_t audio_block_align     = has_audio ? static_cast<uint16_t>(audio->channels * (audio_bits_per_sample / 8)) : 0;
    const uint32_t audio_byte_rate       = has_audio ? static_cast<uint32_t>(audio->sample_rate * audio_block_align) : 0;
    const uint32_t audio_data_size       = has_audio ? static_cast<uint32_t>(audio_pcm.size()) : 0;

    std::vector<uint8_t> avi_data;
    avi_data.reserve(static_cast<size_t>(num_images) * 1024);

    write_fourcc(avi_data, "RIFF");
    const size_t riff_size_pos = avi_data.size();
    write_u32_le(avi_data, 0);
    write_fourcc(avi_data, "AVI ");

    write_fourcc(avi_data, "LIST");
    uint32_t hdrl_size = 4 + 8 + 56 + 8 + 4 + 8 + 56 + 8 + 40;
    if (has_audio) {
        hdrl_size += 8 + (4 + 8 + 56 + 8 + 16);
    }
    write_u32_le(avi_data, hdrl_size);
    write_fourcc(avi_data, "hdrl");

    write_fourcc(avi_data, "avih");
    write_u32_le(avi_data, 56);
    write_u32_le(avi_data, 1000000 / fps);
    write_u32_le(avi_data, 0);
    write_u32_le(avi_data, 0);
    write_u32_le(avi_data, 0x110);
    write_u32_le(avi_data, num_images);
    write_u32_le(avi_data, 0);
    write_u32_le(avi_data, has_audio ? 2 : 1);
    write_u32_le(avi_data, width * height * 3);
    write_u32_le(avi_data, width);
    write_u32_le(avi_data, height);
    write_u32_le(avi_data, 0);
    write_u32_le(avi_data, 0);
    write_u32_le(avi_data, 0);
    write_u32_le(avi_data, 0);

    write_fourcc(avi_data, "LIST");
    write_u32_le(avi_data, 4 + 8 + 56 + 8 + 40);
    write_fourcc(avi_data, "strl");

    write_fourcc(avi_data, "strh");
    write_u32_le(avi_data, 56);
    write_fourcc(avi_data, "vids");
    write_fourcc(avi_data, "MJPG");
    write_u32_le(avi_data, 0);
    write_u16_le(avi_data, 0);
    write_u16_le(avi_data, 0);
    write_u32_le(avi_data, 0);
    write_u32_le(avi_data, 1);
    write_u32_le(avi_data, fps);
    write_u32_le(avi_data, 0);
    write_u32_le(avi_data, num_images);
    write_u32_le(avi_data, width * height * 3);
    write_u32_le(avi_data, static_cast<uint32_t>(-1));
    write_u32_le(avi_data, 0);
    write_u16_le(avi_data, 0);
    write_u16_le(avi_data, 0);
    write_u16_le(avi_data, 0);
    write_u16_le(avi_data, 0);

    write_fourcc(avi_data, "strf");
    write_u32_le(avi_data, 40);
    write_u32_le(avi_data, 40);
    write_u32_le(avi_data, width);
    write_u32_le(avi_data, height);
    write_u16_le(avi_data, 1);
    write_u16_le(avi_data, 24);
    write_fourcc(avi_data, "MJPG");
    write_u32_le(avi_data, width * height * 3);
    write_u32_le(avi_data, 0);
    write_u32_le(avi_data, 0);
    write_u32_le(avi_data, 0);
    write_u32_le(avi_data, 0);

    if (has_audio) {
        write_fourcc(avi_data, "LIST");
        write_u32_le(avi_data, 4 + 8 + 56 + 8 + 16);
        write_fourcc(avi_data, "strl");

        write_fourcc(avi_data, "strh");
        write_u32_le(avi_data, 56);
        write_fourcc(avi_data, "auds");
        write_u32_le(avi_data, 0);
        write_u32_le(avi_data, 0);
        write_u16_le(avi_data, 0);
        write_u16_le(avi_data, 0);
        write_u32_le(avi_data, 0);
        write_u32_le(avi_data, audio_block_align);
        write_u32_le(avi_data, audio_byte_rate);
        write_u32_le(avi_data, 0);
        write_u32_le(avi_data, static_cast<uint32_t>(audio->sample_count));
        write_u32_le(avi_data, audio_data_size);
        write_u32_le(avi_data, static_cast<uint32_t>(-1));
        write_u32_le(avi_data, audio_block_align);
        write_u16_le(avi_data, 0);
        write_u16_le(avi_data, 0);
        write_u16_le(avi_data, 0);
        write_u16_le(avi_data, 0);

        write_fourcc(avi_data, "strf");
        write_u32_le(avi_data, 16);
        write_u16_le(avi_data, 1);
        write_u16_le(avi_data, static_cast<uint16_t>(audio->channels));
        write_u32_le(avi_data, audio->sample_rate);
        write_u32_le(avi_data, audio_byte_rate);
        write_u16_le(avi_data, audio_block_align);
        write_u16_le(avi_data, audio_bits_per_sample);
    }

    write_fourcc(avi_data, "LIST");
    const size_t movi_size_pos = avi_data.size();
    write_u32_le(avi_data, 0);
    write_fourcc(avi_data, "movi");

    std::vector<avi_chunk_index_entry> index;
    index.reserve(static_cast<size_t>(num_images) + (has_audio ? 1 : 0));
    std::vector<uint8_t> jpeg_data;

    for (int i = 0; i < num_images; i++) {
        jpeg_data.clear();

        auto write_to_buf = [](void* context, void* data, int size) {
            auto* buffer       = reinterpret_cast<std::vector<uint8_t>*>(context);
            const uint8_t* src = reinterpret_cast<const uint8_t*>(data);
            buffer->insert(buffer->end(), src, src + size);
        };

        if (!stbi_write_jpg_to_func(write_to_buf, &jpeg_data, images[i].width, images[i].height, channels, images[i].data, mjpg_quality)) {
            fprintf(stderr, "Error: Failed to encode JPEG frame.\n");
            return {};
        }

        avi_chunk_index_entry video_entry = {};
        memcpy(video_entry.fourcc, "00dc", 4);
        video_entry.flags  = 0x10;
        video_entry.offset = static_cast<uint32_t>(avi_data.size());
        write_fourcc(avi_data, "00dc");
        write_u32_le(avi_data, static_cast<uint32_t>(jpeg_data.size()));
        video_entry.size = static_cast<uint32_t>(jpeg_data.size());
        avi_data.insert(avi_data.end(), jpeg_data.begin(), jpeg_data.end());
        index.push_back(video_entry);

        if (jpeg_data.size() % 2) {
            avi_data.push_back(0);
        }
    }

    if (has_audio && !audio_pcm.empty()) {
        avi_chunk_index_entry audio_entry = {};
        memcpy(audio_entry.fourcc, "01wb", 4);
        audio_entry.flags  = 0;
        audio_entry.offset = static_cast<uint32_t>(avi_data.size());
        audio_entry.size   = static_cast<uint32_t>(audio_pcm.size());
        write_fourcc(avi_data, "01wb");
        write_u32_le(avi_data, static_cast<uint32_t>(audio_pcm.size()));
        avi_data.insert(avi_data.end(), audio_pcm.begin(), audio_pcm.end());
        index.push_back(audio_entry);
        if (audio_pcm.size() % 2 != 0) {
            avi_data.push_back(0);
        }
    }

    const size_t movi_size = avi_data.size() - movi_size_pos - 4;
    patch_u32_le(avi_data, movi_size_pos, static_cast<uint32_t>(movi_size));

    write_fourcc(avi_data, "idx1");
    write_u32_le(avi_data, static_cast<uint32_t>(index.size() * 16));
    for (const auto& entry : index) {
        write_fourcc(avi_data, entry.fourcc);
        write_u32_le(avi_data, entry.flags);
        write_u32_le(avi_data, entry.offset);
        write_u32_le(avi_data, entry.size);
    }

    const size_t file_size = avi_data.size() - riff_size_pos - 4;
    patch_u32_le(avi_data, riff_size_pos, static_cast<uint32_t>(file_size));

    return avi_data;
}

int create_mjpg_avi_from_sd_images(const char* filename, sd_image_t* images, int num_images, int fps, int quality, const sd_audio_t* audio) {
    std::vector<uint8_t> avi_data = create_mjpg_avi_from_sd_images_to_vector(images, num_images, fps, quality, audio);
    if (avi_data.empty()) {
        return -1;
    }
    if (!write_binary_file_bytes(filename, avi_data)) {
        perror("Error opening file for writing");
        return -1;
    }
    return 0;
}

#ifdef SD_USE_WEBP
std::vector<uint8_t> create_animated_webp_from_sd_images_to_vector(sd_image_t* images, int num_images, int fps, int quality) {
    if (num_images == 0) {
        fprintf(stderr, "Error: Image array is empty.\n");
        return {};
    }
    if (fps <= 0) {
        fprintf(stderr, "Error: FPS must be positive.\n");
        return {};
    }

    const int width    = static_cast<int>(images[0].width);
    const int height   = static_cast<int>(images[0].height);
    const int channels = static_cast<int>(images[0].channel);
    if (channels != 1 && channels != 3 && channels != 4) {
        fprintf(stderr, "Error: Unsupported channel count: %d\n", channels);
        return {};
    }

    WebPAnimEncoderOptions anim_options;
    WebPConfig config;
    if (!WebPAnimEncoderOptionsInit(&anim_options) || !WebPConfigInit(&config)) {
        fprintf(stderr, "Error: Failed to initialize WebP animation encoder.\n");
        return {};
    }

    config.quality      = static_cast<float>(quality);
    config.method       = 4;
    config.thread_level = 1;
    if (channels == 4) {
        config.exact = 1;
    }
    if (!WebPValidateConfig(&config)) {
        fprintf(stderr, "Error: Invalid WebP encoder configuration.\n");
        return {};
    }

    WebPAnimEncoderPtr enc(WebPAnimEncoderNew(width, height, &anim_options));
    if (enc == nullptr) {
        fprintf(stderr, "Error: Could not create WebPAnimEncoder object.\n");
        return {};
    }

    const int frame_duration_ms = std::max(1, static_cast<int>(std::lround(1000.0 / static_cast<double>(fps))));
    int timestamp_ms            = 0;

    for (int i = 0; i < num_images; ++i) {
        const sd_image_t& image = images[i];
        if (static_cast<int>(image.width) != width || static_cast<int>(image.height) != height) {
            fprintf(stderr, "Error: Frame dimensions do not match.\n");
            return {};
        }

        WebPPictureGuard picture;
        if (!picture.initialized) {
            fprintf(stderr, "Error: Failed to initialize WebPPicture.\n");
            return {};
        }
        picture.picture.use_argb = 1;
        picture.picture.width    = width;
        picture.picture.height   = height;

        bool picture_ok = false;
        std::vector<uint8_t> rgb_buffer;
        if (image.channel == 1) {
            rgb_buffer.resize(static_cast<size_t>(width) * static_cast<size_t>(height) * 3);
            for (int p = 0; p < width * height; ++p) {
                rgb_buffer[p * 3 + 0] = image.data[p];
                rgb_buffer[p * 3 + 1] = image.data[p];
                rgb_buffer[p * 3 + 2] = image.data[p];
            }
            picture_ok = WebPPictureImportRGB(&picture.picture, rgb_buffer.data(), width * 3) != 0;
        } else if (image.channel == 4) {
            picture_ok = WebPPictureImportRGBA(&picture.picture, image.data, width * 4) != 0;
        } else {
            picture_ok = WebPPictureImportRGB(&picture.picture, image.data, width * 3) != 0;
        }

        if (!picture_ok) {
            fprintf(stderr, "Error: Failed to import frame into WebPPicture.\n");
            return {};
        }

        if (!WebPAnimEncoderAdd(enc.get(), &picture.picture, timestamp_ms, &config)) {
            fprintf(stderr, "Error: Failed to add frame to animated WebP: %s\n", WebPAnimEncoderGetError(enc.get()));
            return {};
        }

        timestamp_ms += frame_duration_ms;
    }

    if (!WebPAnimEncoderAdd(enc.get(), nullptr, timestamp_ms, nullptr)) {
        fprintf(stderr, "Error: Failed to finalize animated WebP frames: %s\n", WebPAnimEncoderGetError(enc.get()));
        return {};
    }

    WebPDataGuard webp_data;
    if (!WebPAnimEncoderAssemble(enc.get(), &webp_data.data)) {
        fprintf(stderr, "Error: Failed to assemble animated WebP: %s\n", WebPAnimEncoderGetError(enc.get()));
        return {};
    }

    return std::vector<uint8_t>(webp_data.data.bytes, webp_data.data.bytes + webp_data.data.size);
}

int create_animated_webp_from_sd_images(const char* filename, sd_image_t* images, int num_images, int fps, int quality) {
    std::vector<uint8_t> webp_data = create_animated_webp_from_sd_images_to_vector(images, num_images, fps, quality);
    if (webp_data.empty()) {
        return -1;
    }
    if (!write_binary_file_bytes(filename, webp_data)) {
        perror("Error opening file for writing");
        return -1;
    }
    return 0;
}
#endif

#ifdef SD_USE_WEBM
#ifdef SD_USE_OPUS
namespace {
struct OpusEncodedPacket {
    uint64_t timestamp_ns;
    std::vector<uint8_t> data;
};

static std::vector<float> resample_interleaved_hann_sinc(const float* src,
                                                         uint64_t src_n,
                                                         uint32_t src_ch,
                                                         uint32_t dst_ch,
                                                         uint32_t src_sr,
                                                         uint32_t dst_sr) {
    if (src == nullptr || src_n == 0 || src_ch == 0 || dst_ch == 0 || src_sr == 0 || dst_sr == 0) {
        return {};
    }

    const uint64_t dst_n = static_cast<uint64_t>(
        std::ceil(static_cast<double>(dst_sr) * static_cast<double>(src_n) / static_cast<double>(src_sr)));
    std::vector<float> dst(static_cast<size_t>(dst_n * dst_ch), 0.0f);

    int orig = static_cast<int>(src_sr);
    int neo  = static_cast<int>(dst_sr);
    int gcd  = std::gcd(orig, neo);
    orig /= gcd;
    neo /= gcd;

    constexpr int lowpass_filter_width = 6;
    constexpr double rolloff           = 0.99;
    constexpr double pi                = 3.14159265358979323846;
    const double base_freq             = static_cast<double>(std::min(orig, neo)) * rolloff;
    const int width                    = static_cast<int>(std::ceil(lowpass_filter_width * static_cast<double>(orig) / base_freq));
    const double scale                 = base_freq / static_cast<double>(orig);

    for (uint64_t i = 0; i < dst_n; ++i) {
        const double center = static_cast<double>(i) * static_cast<double>(src_sr) / static_cast<double>(dst_sr);
        const int64_t left  = static_cast<int64_t>(std::floor(center)) - width - 1;
        const int64_t right = static_cast<int64_t>(std::floor(center)) + width + 1;
        for (uint32_t c = 0; c < dst_ch; ++c) {
            double acc = 0.0;
            for (int64_t j = left; j <= right; ++j) {
                if (j < 0 || j >= static_cast<int64_t>(src_n)) {
                    continue;
                }
                double t = (static_cast<double>(j) / static_cast<double>(orig) -
                            static_cast<double>(i) / static_cast<double>(neo)) *
                           base_freq;
                t = std::clamp(t, -static_cast<double>(lowpass_filter_width), static_cast<double>(lowpass_filter_width));
                double window = std::cos(t * pi / static_cast<double>(lowpass_filter_width) / 2.0);
                window *= window;
                double t_pi = t * pi;
                double sinc = t_pi == 0.0 ? 1.0 : std::sin(t_pi) / t_pi;
                const uint32_t src_c = src_ch == 1 ? 0 : std::min<uint32_t>(c, src_ch - 1);
                acc += static_cast<double>(src[static_cast<size_t>(j * src_ch + src_c)]) * sinc * window * scale;
            }
            dst[static_cast<size_t>(i * dst_ch + c)] = static_cast<float>(acc);
        }
    }
    return dst;
}

// Encode audio as Opus for WebM muxing. WebM browsers (Chrome/Firefox) only
// decode Vorbis/Opus — a raw-PCM (A_PCM/INT/LIT) track is muxed fine but plays
// SILENT in a <video> element. Preserve mono/stereo, resample to 48 kHz
// (Opus' native rate), and emit 20 ms packets plus the OpusHead CodecPrivate.
static bool encode_audio_to_opus(const sd_audio_t* audio,
                                 std::vector<OpusEncodedPacket>& packets,
                                 std::vector<uint8_t>& opus_head,
                                 uint64_t& codec_delay_ns,
                                 uint32_t& opus_channels) {
    if (audio == nullptr || audio->data == nullptr || audio->sample_count == 0 ||
        audio->channels == 0 || audio->sample_rate == 0) {
        return false;
    }

    const uint32_t src_sr = audio->sample_rate;
    const uint32_t src_ch = audio->channels;
    const uint64_t src_n  = audio->sample_count;
    const uint32_t dst_sr = 48000;
    const uint32_t dst_ch = src_ch == 1 ? 1 : 2;
    opus_channels         = dst_ch;

    std::vector<float> res;
    if (src_sr == dst_sr) {
        res.resize(static_cast<size_t>(src_n * dst_ch));
        for (uint64_t i = 0; i < src_n; ++i) {
            for (uint32_t c = 0; c < dst_ch; ++c) {
                const uint32_t src_c = src_ch == 1 ? 0 : std::min<uint32_t>(c, src_ch - 1);
                res[static_cast<size_t>(i * dst_ch + c)] = audio->data[static_cast<size_t>(i * src_ch + src_c)];
            }
        }
    } else {
        res = resample_interleaved_hann_sinc(audio->data, src_n, src_ch, dst_ch, src_sr, dst_sr);
    }
    if (res.empty()) {
        return false;
    }

    int err          = 0;
    OpusEncoder* enc = opus_encoder_create(static_cast<opus_int32>(dst_sr), static_cast<int>(dst_ch), OPUS_APPLICATION_AUDIO, &err);
    if (enc == nullptr || err != OPUS_OK) {
        if (enc != nullptr) {
            opus_encoder_destroy(enc);
        }
        return false;
    }
    opus_encoder_ctl(enc, OPUS_SET_BITRATE(dst_ch == 2 ? 128000 : 96000));

    opus_int32 lookahead = 0;  // encoder pre-skip, in 48 kHz samples
    opus_encoder_ctl(enc, OPUS_GET_LOOKAHEAD(&lookahead));

    const int frame_size              = static_cast<int>(dst_sr) / 50;  // 20 ms = 960 samples @ 48 kHz
    const uint64_t frame_dur_ns       = 20000000ULL;
    std::vector<int16_t> pcm(static_cast<size_t>(frame_size * dst_ch));
    std::vector<uint8_t> out(4000);
    uint64_t ts_ns = 0;
    const uint64_t dst_n = static_cast<uint64_t>(res.size() / dst_ch);

    for (uint64_t off = 0; off < dst_n; off += static_cast<uint64_t>(frame_size)) {
        for (int k = 0; k < frame_size; ++k) {
            const uint64_t idx = off + static_cast<uint64_t>(k);
            for (uint32_t c = 0; c < dst_ch; ++c) {
                float s = idx < dst_n ? res[static_cast<size_t>(idx * dst_ch + c)] : 0.0f;
                s       = std::clamp(s, -1.0f, 1.0f);
                pcm[static_cast<size_t>(k * dst_ch + c)] = static_cast<int16_t>(std::lrint(s * 32767.0f));
            }
        }
        const opus_int32 n = opus_encode(enc, pcm.data(), frame_size, out.data(), static_cast<opus_int32>(out.size()));
        if (n < 0) {
            opus_encoder_destroy(enc);
            return false;
        }
        if (n > 0) {
            packets.push_back({ts_ns, std::vector<uint8_t>(out.begin(), out.begin() + n)});
        }
        ts_ns += frame_dur_ns;
    }
    opus_encoder_destroy(enc);

    // OpusHead identification header (19 bytes, channel mapping family 0).
    const uint16_t pre_skip   = static_cast<uint16_t>(lookahead);
    static const char magic[] = {'O', 'p', 'u', 's', 'H', 'e', 'a', 'd'};
    opus_head.assign(magic, magic + sizeof(magic));
    opus_head.push_back(1);                                            // version
    opus_head.push_back(static_cast<uint8_t>(dst_ch));                 // channel count
    opus_head.push_back(static_cast<uint8_t>(pre_skip & 0xff));        // pre-skip lo
    opus_head.push_back(static_cast<uint8_t>((pre_skip >> 8) & 0xff)); // pre-skip hi
    const uint32_t in_sr = dst_sr;                                     // input sample rate (informational)
    opus_head.push_back(static_cast<uint8_t>(in_sr & 0xff));
    opus_head.push_back(static_cast<uint8_t>((in_sr >> 8) & 0xff));
    opus_head.push_back(static_cast<uint8_t>((in_sr >> 16) & 0xff));
    opus_head.push_back(static_cast<uint8_t>((in_sr >> 24) & 0xff));
    opus_head.push_back(0);  // output gain lo
    opus_head.push_back(0);  // output gain hi
    opus_head.push_back(0);  // channel mapping family

    codec_delay_ns = static_cast<uint64_t>(pre_skip) * 1000000000ULL / dst_sr;
    return !packets.empty();
}
}  // namespace
#endif  // SD_USE_OPUS

#ifdef SD_USE_VPX
// ─── VP9 10-bit encoder ──────────────────────────────────────────────────────────────────────
// Replaces the libwebp→VP8-keyframe hack: a real, stateful libvpx VP9 encoder emitting 10-bit
// (profile 2) inter-frame video with BT.709/limited-range signalling and constant-quality rate
// control. Frames are fed strictly in order; the muxer replays the encoded packets (with each
// packet's real keyframe flag) so the streaming and whole-array paths stay identical.
//
// The RGB→YUV conversion folds in an optional black-point lift (LTX's VAE floors dark scenes at
// ~5% grey, so true black never reaches 0). Tunable at runtime without a rebuild:
//   LTXAV_VP9_CQ           constant-quality level 4..63 (default 20; lower = higher quality/bigger).
//                          Default leans toward quality; ~6x larger than the old size-first CQ32 but
//                          visually near-transparent. Bump toward 28-32 if size matters more.
//   LTXAV_VP9_BLACK_POINT  STATIC input black point 0..0.3 mapped to 0 (default 0 = off). A fixed
//                          linear lift clips shadow detail below the threshold, so it is off by
//                          default; leave it 0 and prefer a content-adaptive black level upstream.
//   LTXAV_VP9_DITHER       ordered (Bayer 4x4) luma dither on the 10-bit quantise (default 0 = off).
//                          10-bit has ample headroom vs 8-bit banding, so dither is unnecessary and
//                          net-harmful: on flat regions it turns sub-LSB model noise into visible
//                          ±1-LSB toggling that reads as temporal shimmer. Set 1 only if a black-
//                          point lift is contouring a genuinely 8-bit-ish source.
//   LTXAV_VP9_CPU_USED     libvpx good-quality speed 0..8 (default 4; higher = faster/lower quality)
//   LTXAV_VP9_KF_SECONDS   max keyframe interval in seconds for seeking (default 2)
static float vp9_env_float(const char* name, float defv, float lo, float hi) {
    const char* s = getenv(name);
    if (s == nullptr || *s == '\0') {
        return defv;
    }
    char* end = nullptr;
    const float v = strtof(s, &end);
    if (end == s) {
        return defv;
    }
    return std::max(lo, std::min(hi, v));
}

struct EncodedVideoPacket {
    std::vector<uint8_t> bytes;
    bool                 is_key = false;
};

// One ordered-dither Bayer 4x4 threshold matrix, values recentred to [-0.5, 0.5).
static float bayer4(int x, int y) {
    static const int m[16] = {0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5};
    return (static_cast<float>(m[(y & 3) * 4 + (x & 3)]) + 0.5f) / 16.0f - 0.5f;
}

// RGB8 → 10-bit BT.709 limited-range YUV 4:2:0, with optional black-point lift and optional luma
// dither. Width/height are assumed even (LTX dims are ×32-snapped). Planes are sized by the caller's
// vpx_image strides. `dither` gates the ordered Bayer term: off by default because at 10 bits it
// converts sub-LSB per-frame model noise on flat regions into visible ±1-LSB temporal shimmer.
static void rgb8_to_yuv420p10(const uint8_t* rgb, int channels, int w, int h, float black_point,
                              bool dither, uint16_t* yp, int ys, uint16_t* up, int us, uint16_t* vp, int vs) {
    const float inv = black_point < 1.0f ? 1.0f / (1.0f - black_point) : 1.0f;
    auto lift       = [&](float c01) { return std::max(0.0f, (c01 - black_point) * inv); };
    auto clamp10    = [](float v) { return static_cast<uint16_t>(std::max(0.0f, std::min(1023.0f, v))); };

    // Luma (per-pixel). Dither adds the spatial Bayer offset; without it we round-to-nearest (+0.5)
    // so the truncating cast does not floor-bias, keeping flat regions temporally stable.
    for (int y = 0; y < h; ++y) {
        uint16_t* yrow = reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(yp) + static_cast<size_t>(y) * ys);
        for (int x = 0; x < w; ++x) {
            const uint8_t* px = rgb + (static_cast<size_t>(y) * w + x) * channels;
            const float r = lift(px[0] / 255.0f), g = lift(px[1] / 255.0f), b = lift(px[2] / 255.0f);
            const float yl = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            const float d  = dither ? bayer4(x, y) : 0.5f;
            yrow[x] = clamp10(64.0f + 876.0f * yl + d);
        }
    }
    // Chroma 4:2:0 — average each 2×2 block of lifted RGB, then convert (co-sited enough for video).
    for (int y = 0; y < h; y += 2) {
        uint16_t* urow = reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(up) + static_cast<size_t>(y / 2) * us);
        uint16_t* vrow = reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(vp) + static_cast<size_t>(y / 2) * vs);
        for (int x = 0; x < w; x += 2) {
            float r = 0.0f, g = 0.0f, b = 0.0f;
            for (int dy = 0; dy < 2; ++dy) {
                for (int dx = 0; dx < 2; ++dx) {
                    const uint8_t* px = rgb + (static_cast<size_t>(y + dy) * w + (x + dx)) * channels;
                    r += lift(px[0] / 255.0f);
                    g += lift(px[1] / 255.0f);
                    b += lift(px[2] / 255.0f);
                }
            }
            r *= 0.25f;
            g *= 0.25f;
            b *= 0.25f;
            const float yl = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            urow[x / 2] = clamp10(512.0f + 896.0f * ((b - yl) / 1.8556f));
            vrow[x / 2] = clamp10(512.0f + 896.0f * ((r - yl) / 1.5748f));
        }
    }
}

// Extract an RGB view of an sd_image_t (handles gray/RGBA); returns a pointer + channel count into
// `scratch` when a conversion is needed, else a pointer straight into the frame.
static const uint8_t* sd_image_as_rgb(const sd_image_t& image, std::vector<uint8_t>& scratch, int& out_channels) {
    const int w = static_cast<int>(image.width), h = static_cast<int>(image.height);
    const int c = static_cast<int>(image.channel);
    if (c == 3 || c == 4) {
        out_channels = c;
        return image.data;
    }
    scratch.resize(static_cast<size_t>(w) * h * 3);
    for (int i = 0; i < w * h; ++i) {
        scratch[i * 3 + 0] = scratch[i * 3 + 1] = scratch[i * 3 + 2] = image.data[i];
    }
    out_channels = 3;
    return scratch.data();
}

class Vp9Encoder {
public:
    ~Vp9Encoder() {
        if (initialized_) {
            vpx_codec_destroy(&ctx_);
        }
        if (img_alloc_) {
            vpx_img_free(&img_);
        }
    }

    bool init(int width, int height, int fps) {
        width_  = width;
        height_ = height;
        fps_    = fps > 0 ? fps : 24;
        if ((width & 1) || (height & 1)) {
            fprintf(stderr, "Vp9Encoder: odd dimensions %dx%d unsupported.\n", width, height);
            return false;
        }
        black_point_    = vp9_env_float("LTXAV_VP9_BLACK_POINT", 0.0f, 0.0f, 0.3f);
        dither_         = vp9_env_float("LTXAV_VP9_DITHER", 0.0f, 0.0f, 1.0f) >= 0.5f;
        const int cq    = static_cast<int>(vp9_env_float("LTXAV_VP9_CQ", 20.0f, 4.0f, 63.0f));
        const int cpu   = static_cast<int>(vp9_env_float("LTXAV_VP9_CPU_USED", 4.0f, 0.0f, 8.0f));
        const float kfs = vp9_env_float("LTXAV_VP9_KF_SECONDS", 2.0f, 0.0f, 100.0f);

        vpx_codec_iface_t* iface = vpx_codec_vp9_cx();
        vpx_codec_enc_cfg_t cfg{};
        if (vpx_codec_enc_config_default(iface, &cfg, 0) != VPX_CODEC_OK) {
            fprintf(stderr, "Vp9Encoder: enc_config_default failed.\n");
            return false;
        }
        cfg.g_w               = static_cast<unsigned>(width);
        cfg.g_h               = static_cast<unsigned>(height);
        cfg.g_timebase.num    = 1;
        cfg.g_timebase.den    = fps_;
        cfg.g_profile         = 2;  // 10/12-bit 4:2:0
        cfg.g_bit_depth       = VPX_BITS_10;
        cfg.g_input_bit_depth = 10;
        cfg.g_pass            = VPX_RC_ONE_PASS;
        cfg.g_lag_in_frames   = 0;  // 1-in-1-out: keeps the streaming spool a clean frame↔packet map
        cfg.rc_end_usage      = VPX_Q;
        cfg.rc_target_bitrate = 0;
        cfg.kf_mode           = VPX_KF_AUTO;
        cfg.kf_max_dist       = kfs > 0.0f ? std::max(1u, static_cast<unsigned>(kfs * fps_)) : 9999u;
        unsigned hw_threads   = std::thread::hardware_concurrency();
        cfg.g_threads         = std::max(1u, std::min(hw_threads ? hw_threads : 4u, 8u));

        if (vpx_codec_enc_init(&ctx_, iface, &cfg, VPX_CODEC_USE_HIGHBITDEPTH) != VPX_CODEC_OK) {
            fprintf(stderr, "Vp9Encoder: enc_init failed: %s\n", vpx_codec_error(&ctx_));
            return false;
        }
        initialized_ = true;
        vpx_codec_control(&ctx_, VP8E_SET_CPUUSED, cpu);
        vpx_codec_control(&ctx_, VP8E_SET_CQ_LEVEL, cq);  // shared VP8/VP9 constant-quality control
        vpx_codec_control(&ctx_, VP9E_SET_ROW_MT, 1);
        vpx_codec_control(&ctx_, VP9E_SET_COLOR_SPACE, VPX_CS_BT_709);
        vpx_codec_control(&ctx_, VP9E_SET_COLOR_RANGE, VPX_CR_STUDIO_RANGE);

        if (vpx_img_alloc(&img_, VPX_IMG_FMT_I42016, width, height, 1) == nullptr) {
            fprintf(stderr, "Vp9Encoder: img_alloc failed.\n");
            return false;
        }
        img_.bit_depth = 10;  // 10-bit samples in the 16-bit I42016 planes
        img_alloc_     = true;
        return true;
    }

    // Encode one frame, appending every emitted packet (usually exactly one with lag=0) to `out`.
    bool encode(const sd_image_t& image, std::vector<EncodedVideoPacket>& out) {
        if (!initialized_ || !img_alloc_) {
            return false;
        }
        if (static_cast<int>(image.width) != width_ || static_cast<int>(image.height) != height_) {
            fprintf(stderr, "Vp9Encoder: frame dimensions do not match.\n");
            return false;
        }
        std::vector<uint8_t> scratch;
        int channels        = 3;
        const uint8_t* rgb  = sd_image_as_rgb(image, scratch, channels);
        rgb8_to_yuv420p10(rgb, channels, width_, height_, black_point_, dither_,
                          reinterpret_cast<uint16_t*>(img_.planes[VPX_PLANE_Y]), img_.stride[VPX_PLANE_Y],
                          reinterpret_cast<uint16_t*>(img_.planes[VPX_PLANE_U]), img_.stride[VPX_PLANE_U],
                          reinterpret_cast<uint16_t*>(img_.planes[VPX_PLANE_V]), img_.stride[VPX_PLANE_V]);
        if (vpx_codec_encode(&ctx_, &img_, pts_++, 1, 0, VPX_DL_GOOD_QUALITY) != VPX_CODEC_OK) {
            fprintf(stderr, "Vp9Encoder: encode failed: %s\n", vpx_codec_error(&ctx_));
            return false;
        }
        return drain(out);
    }

    // Flush the encoder (lag=0 leaves nothing pending, but this is correct regardless).
    bool finish(std::vector<EncodedVideoPacket>& out) {
        if (!initialized_) {
            return false;
        }
        if (vpx_codec_encode(&ctx_, nullptr, pts_, 1, 0, VPX_DL_GOOD_QUALITY) != VPX_CODEC_OK) {
            fprintf(stderr, "Vp9Encoder: flush failed: %s\n", vpx_codec_error(&ctx_));
            return false;
        }
        return drain(out);
    }

private:
    bool drain(std::vector<EncodedVideoPacket>& out) {
        vpx_codec_iter_t iter = nullptr;
        const vpx_codec_cx_pkt_t* pkt = nullptr;
        while ((pkt = vpx_codec_get_cx_data(&ctx_, &iter)) != nullptr) {
            if (pkt->kind != VPX_CODEC_CX_FRAME_PKT) {
                continue;
            }
            EncodedVideoPacket p;
            p.is_key = (pkt->data.frame.flags & VPX_FRAME_IS_KEY) != 0;
            p.bytes.assign(static_cast<const uint8_t*>(pkt->data.frame.buf),
                           static_cast<const uint8_t*>(pkt->data.frame.buf) + pkt->data.frame.sz);
            out.push_back(std::move(p));
        }
        return true;
    }

    vpx_codec_ctx_t ctx_{};
    vpx_image_t     img_{};
    bool            initialized_ = false;
    bool            img_alloc_   = false;
    int             width_       = 0;
    int             height_      = 0;
    int             fps_         = 24;
    float           black_point_ = 0.0f;
    bool            dither_      = false;
    vpx_codec_pts_t pts_         = 0;
};

// True when this build was compiled with VP9 support AND it is not force-disabled at runtime.
static bool webm_use_vp9() {
    const char* codec = getenv("LTXAV_WEBM_CODEC");
    if (codec != nullptr && (strcmp(codec, "vp8") == 0 || strcmp(codec, "VP8") == 0)) {
        return false;
    }
    return true;
}
#endif  // SD_USE_VPX

// Shared WebM mux core. `get_frame(i, out, is_key)` must fill `out` with frame i's encoded video
// bytes, set `is_key` to whether it is a keyframe, and return true (false = abort the container).
// This is the SINGLE mux sequence — track setup, per-frame audio interleave, timestamps, Opus/PCM
// handling and Finalize — that both the whole-array encoder and the incremental (streaming) writer
// funnel through, so their outputs match for the same frames + audio + num_images + fps. Muxing is
// codec-agnostic and order-preserving: pre-encoding frames elsewhere (a spool, a buffer) and
// feeding them here in order reproduces the exact bytes an inline encode would have produced, which
// holds for VP9 inter-frame packets too as long as each packet's real keyframe flag is preserved.
// use_vp9 selects the codec id (V_VP9 vs V_VP8) and, for VP9, writes BT.709/limited colour tags so
// browsers render the range correctly instead of guessing.
static std::vector<uint8_t> mux_webm(int width, int height, int num_images, int fps,
                                     const sd_audio_t* audio, bool use_vp9,
                                     const std::function<bool(int, std::vector<uint8_t>&, bool&)>& get_frame) {
    if (num_images == 0) {
        fprintf(stderr, "Error: Image array is empty.\n");
        return {};
    }
    if (fps <= 0) {
        fprintf(stderr, "Error: FPS must be positive.\n");
        return {};
    }
    if (width <= 0 || height <= 0) {
        fprintf(stderr, "Error: Invalid frame dimensions.\n");
        return {};
    }

    MemoryMkvWriter writer;

    const int ret = [&]() -> int {
        mkvmuxer::Segment segment;
        if (!segment.Init(&writer)) {
            fprintf(stderr, "Error: Failed to initialize WebM muxer.\n");
            return -1;
        }

        segment.set_mode(mkvmuxer::Segment::kFile);
        segment.OutputCues(true);

        const uint64_t track_number = segment.AddVideoTrack(width, height, 0);
        if (track_number == 0) {
            fprintf(stderr, "Error: Failed to add video track.\n");
            return -1;
        }
        if (!segment.CuesTrack(track_number)) {
            fprintf(stderr, "Error: Failed to set WebM cues track.\n");
            return -1;
        }

        mkvmuxer::VideoTrack* video_track = static_cast<mkvmuxer::VideoTrack*>(segment.GetTrackByNumber(track_number));
        if (video_track != nullptr) {
            video_track->set_display_width(static_cast<uint64_t>(width));
            video_track->set_display_height(static_cast<uint64_t>(height));
            video_track->set_frame_rate(static_cast<double>(fps));
            if (use_vp9) {
                video_track->set_codec_id("V_VP9");
                // Signal BT.709 / limited range so browsers stop guessing the range (the untagged
                // VP8 path rendered washed-out blacks). Matches the encoder's VP9E colour controls.
                mkvmuxer::Colour colour;
                colour.set_matrix_coefficients(1);       // BT.709
                colour.set_range(1);                     // broadcast / limited (studio)
                colour.set_primaries(1);                 // BT.709
                colour.set_transfer_characteristics(1);  // BT.709
                colour.set_bits_per_channel(10);
                colour.set_chroma_subsampling_horz(1);
                colour.set_chroma_subsampling_vert(1);
                if (!video_track->SetColour(colour)) {
                    fprintf(stderr, "Error: Failed to set WebM colour metadata.\n");
                    return -1;
                }
            }
        }

        uint64_t audio_track_number = 0;

        // Prefer Opus — raw PCM in WebM is silent in browsers (Vorbis/Opus only).
        bool use_opus = false;
#ifdef SD_USE_OPUS
        std::vector<OpusEncodedPacket> opus_packets;
        std::vector<uint8_t> opus_head;
        uint64_t opus_codec_delay_ns = 0;
        uint32_t opus_channels       = 0;
        use_opus = audio != nullptr && encode_audio_to_opus(audio, opus_packets, opus_head, opus_codec_delay_ns, opus_channels);
#endif

        std::vector<uint8_t> audio_pcm;
        if (!use_opus) {
            audio_pcm = audio_to_pcm16_bytes(audio);
        }

#ifdef SD_USE_OPUS
        if (use_opus) {
            audio_track_number = segment.AddAudioTrack(48000, static_cast<int32_t>(opus_channels), 0);
            if (audio_track_number == 0) {
                fprintf(stderr, "Error: Failed to add audio track.\n");
                return -1;
            }
            auto* audio_track = static_cast<mkvmuxer::AudioTrack*>(segment.GetTrackByNumber(audio_track_number));
            if (audio_track == nullptr) {
                fprintf(stderr, "Error: Failed to get audio track.\n");
                return -1;
            }
            audio_track->set_codec_id("A_OPUS");
            audio_track->SetCodecPrivate(opus_head.data(), opus_head.size());
            audio_track->set_sample_rate(48000.0);
            audio_track->set_channels(opus_channels);
            audio_track->set_codec_delay(opus_codec_delay_ns);
            audio_track->set_seek_pre_roll(80000000ULL);  // 80 ms, per the WebM Opus guidelines
        } else
#endif
            if (audio != nullptr && !audio_pcm.empty()) {
            audio_track_number = segment.AddAudioTrack(static_cast<int32_t>(audio->sample_rate), static_cast<int32_t>(audio->channels), 0);
            if (audio_track_number == 0) {
                fprintf(stderr, "Error: Failed to add audio track.\n");
                return -1;
            }
            auto* audio_track = static_cast<mkvmuxer::AudioTrack*>(segment.GetTrackByNumber(audio_track_number));
            if (audio_track == nullptr) {
                fprintf(stderr, "Error: Failed to get audio track.\n");
                return -1;
            }
            audio_track->set_codec_id("A_PCM/INT/LIT");
            audio_track->set_bit_depth(16);
            audio_track->set_sample_rate(static_cast<double>(audio->sample_rate));
            audio_track->set_channels(audio->channels);
        }

#ifdef SD_USE_OPUS
        size_t opus_idx = 0;  // next Opus packet awaiting mux
#endif
        segment.GetSegmentInfo()->set_writing_app("stable-diffusion.cpp");
        segment.GetSegmentInfo()->set_muxing_app("stable-diffusion.cpp");

        const uint64_t frame_duration_ns = std::max<uint64_t>(
            1, static_cast<uint64_t>(std::llround(1000000000.0 / static_cast<double>(fps))));
        uint64_t timestamp_ns = 0;

        for (int i = 0; i < num_images; ++i) {
            std::vector<uint8_t> video_frame;
            bool is_key = true;
            if (!get_frame(i, video_frame, is_key)) {
                return -1;
            }

            if (!segment.AddFrame(video_frame.data(),
                                  static_cast<uint64_t>(video_frame.size()),
                                  track_number,
                                  timestamp_ns,
                                  is_key)) {
                fprintf(stderr, "Error: Failed to mux frame %d into WebM.\n", i);
                return -1;
            }

            if (audio_track_number != 0) {
#ifdef SD_USE_OPUS
                if (use_opus) {
                    // Flush every Opus packet that begins before the next video frame.
                    const uint64_t next_video_ts = timestamp_ns + frame_duration_ns;
                    while (opus_idx < opus_packets.size() && opus_packets[opus_idx].timestamp_ns < next_video_ts) {
                        const auto& pkt = opus_packets[opus_idx];
                        if (!segment.AddFrame(pkt.data.data(), pkt.data.size(), audio_track_number, pkt.timestamp_ns, true)) {
                            fprintf(stderr, "Error: Failed to mux Opus packet %zu into WebM.\n", opus_idx);
                            return -1;
                        }
                        ++opus_idx;
                    }
                } else
#endif
                {
                    auto [audio_begin, audio_end] = audio_sample_range_for_video_frame(audio, i, num_images, fps);
                    const uint64_t frame_samples  = audio_end - audio_begin;
                    if (frame_samples > 0) {
                        const uint64_t frame_bytes = frame_samples * audio->channels * sizeof(int16_t);
                        const uint8_t* frame_ptr   = audio_pcm.data() + audio_begin * audio->channels * sizeof(int16_t);
                        if (!segment.AddFrame(frame_ptr,
                                              frame_bytes,
                                              audio_track_number,
                                              timestamp_ns,
                                              true)) {
                            fprintf(stderr, "Error: Failed to mux audio chunk %d into WebM.\n", i);
                            return -1;
                        }
                    }
                }
            }

            timestamp_ns += frame_duration_ns;
        }

#ifdef SD_USE_OPUS
        // Flush any Opus packets trailing past the final video frame.
        if (use_opus && audio_track_number != 0) {
            for (; opus_idx < opus_packets.size(); ++opus_idx) {
                const auto& pkt = opus_packets[opus_idx];
                if (!segment.AddFrame(pkt.data.data(), pkt.data.size(), audio_track_number, pkt.timestamp_ns, true)) {
                    fprintf(stderr, "Error: Failed to mux trailing Opus packet %zu into WebM.\n", opus_idx);
                    return -1;
                }
            }
        }
#endif

        if (!segment.Finalize()) {
            fprintf(stderr, "Error: Failed to finalize WebM output.\n");
            return -1;
        }
        return 0;
    }();
    if (ret != 0) {
        return {};
    }
    return writer.data();
}

std::vector<uint8_t> create_webm_from_sd_images_to_vector(sd_image_t* images, int num_images, int fps, int quality,
                                                           const sd_audio_t* audio, bool consume_image_data) {
    if (num_images == 0) {
        fprintf(stderr, "Error: Image array is empty.\n");
        return {};
    }
    const int width  = static_cast<int>(images[0].width);
    const int height = static_cast<int>(images[0].height);

#ifdef SD_USE_VPX
    if (webm_use_vp9()) {
        // Encode every frame in order through one stateful VP9 encoder (lag=0 → exactly one packet
        // per frame), buffering the small encoded packets, then mux. Free each pixel buffer as soon
        // as it is encoded on transfer-of-ownership.
        Vp9Encoder enc;
        if (!enc.init(width, height, fps)) {
            return {};
        }
        std::vector<EncodedVideoPacket> packets;
        packets.reserve(static_cast<size_t>(num_images));
        for (int i = 0; i < num_images; ++i) {
            sd_image_t& image = images[i];
            if (static_cast<int>(image.width) != width || static_cast<int>(image.height) != height) {
                fprintf(stderr, "Error: Frame dimensions do not match.\n");
                return {};
            }
            if (!enc.encode(image, packets)) {
                return {};
            }
            if (consume_image_data) {
                free(image.data);
                image.data = nullptr;
            }
        }
        enc.finish(packets);
        if (static_cast<int>(packets.size()) != num_images) {
            fprintf(stderr, "Error: VP9 emitted %zu packets for %d frames.\n", packets.size(), num_images);
            return {};
        }
        auto get_frame = [&](int i, std::vector<uint8_t>& out, bool& is_key) -> bool {
            out    = std::move(packets[i].bytes);
            is_key = packets[i].is_key;
            return true;
        };
        return mux_webm(width, height, num_images, fps, audio, true, get_frame);
    }
#endif

    // VP8 fallback (build without SD_USE_VPX, or LTXAV_WEBM_CODEC=vp8): per-frame independent
    // keyframe, released as soon as it is encoded (the encoded bytes do not alias the pixels).
    auto get_frame = [&](int i, std::vector<uint8_t>& out, bool& is_key) -> bool {
        is_key            = true;
        sd_image_t& image = images[i];
        if (static_cast<int>(image.width) != width || static_cast<int>(image.height) != height) {
            fprintf(stderr, "Error: Frame dimensions do not match.\n");
            return false;
        }
        if (!encode_sd_image_to_vp8_frame(image, quality, out)) {
            fprintf(stderr, "Error: Failed to encode frame %d as VP8.\n", i);
            return false;
        }
        if (consume_image_data) {
            free(image.data);
            image.data = nullptr;
        }
        return true;
    };

    return mux_webm(width, height, num_images, fps, audio, false, get_frame);
}

// ─── Incremental (streaming) WebM writer ─────────────────────────────────────────────────────
// Encodes each frame as it arrives (VP9 10-bit via a stateful encoder, or VP8 in the fallback) and
// SPOOLS the (small) encoded packet to a temp file so peak RAM stays bounded to the caller's live
// frame window regardless of clip length. The audio is buffered whole (tiny). finalize() replays
// the spool through the shared mux core with the now-known total frame count, muxes the
// length-matched audio, writes the container atomically (.tmp→rename), and returns the bytes.
// Matches create_webm_from_sd_images_to_vector over the same frames (same encoder, same order).
IncrementalWebmWriter::IncrementalWebmWriter() = default;

IncrementalWebmWriter::~IncrementalWebmWriter() {
    // Abort path: drop the spool + any half-written .tmp (a successful finalize() already removed
    // the spool and renamed the .tmp away, so these become harmless no-ops).
    if (spool_out_.is_open()) {
        spool_out_.close();
    }
    if (spool_in_.is_open()) {
        spool_in_.close();
    }
    std::error_code ec;
    if (!spool_path_.empty()) {
        fs::remove(fs::path(spool_path_), ec);
    }
    if (!final_path_.empty()) {
        fs::remove(fs::path(final_path_ + ".tmp"), ec);
    }
    if (audio_owned_.data != nullptr) {
        free(audio_owned_.data);
        audio_owned_.data = nullptr;
    }
}

// Spool one encoded packet: [len:u32-le][is_key:u8][bytes]. The keyframe flag matters for VP9
// inter-frame packets (the mux must tag it); VP8 keyframes always pass true.
static bool spool_write_record(std::ofstream& out, const uint8_t* bytes, size_t len, bool is_key) {
    unsigned char hdr[5];
    hdr[0] = static_cast<unsigned char>(len & 0xFF);
    hdr[1] = static_cast<unsigned char>((len >> 8) & 0xFF);
    hdr[2] = static_cast<unsigned char>((len >> 16) & 0xFF);
    hdr[3] = static_cast<unsigned char>((len >> 24) & 0xFF);
    hdr[4] = is_key ? 1 : 0;
    out.write(reinterpret_cast<const char*>(hdr), 5);
    if (len > 0) {
        out.write(reinterpret_cast<const char*>(bytes), static_cast<std::streamsize>(len));
    }
    return static_cast<bool>(out);
}

bool IncrementalWebmWriter::open(const std::string& final_path, int fps, int quality) {
    final_path_ = final_path;
    spool_path_ = final_path + ".vspool";
    fps_        = fps;
    quality_    = quality;
    width_      = 0;
    height_     = 0;
    num_frames_ = 0;
    failed_     = false;
#ifdef SD_USE_VPX
    use_vp9_ = webm_use_vp9();  // encoder is created lazily on the first frame (needs its dims)
    vp9_.reset();
#else
    use_vp9_ = false;
#endif
    std::error_code ec;
    fs::remove(fs::path(spool_path_), ec);
    spool_out_.open(fs::path(spool_path_), std::ios::binary | std::ios::trunc);
    if (!spool_out_) {
        fprintf(stderr, "IncrementalWebmWriter: cannot open spool %s\n", spool_path_.c_str());
        failed_ = true;
        return false;
    }
    return true;
}

bool IncrementalWebmWriter::append_video_frame(const sd_image_t& image) {
    if (failed_) {
        return false;
    }
    if (!spool_out_.is_open()) {
        failed_ = true;
        return false;
    }
    const int w = static_cast<int>(image.width);
    const int h = static_cast<int>(image.height);
    if (num_frames_ == 0) {
        width_  = w;  // dims fixed by the first frame, matching create_webm's images[0]
        height_ = h;
    } else if (w != width_ || h != height_) {
        fprintf(stderr, "IncrementalWebmWriter: frame dimensions do not match.\n");
        failed_ = true;
        return false;
    }

#ifdef SD_USE_VPX
    if (use_vp9_) {
        if (num_frames_ == 0) {
            vp9_ = std::make_unique<Vp9Encoder>();
            if (!vp9_->init(width_, height_, fps_)) {
                failed_ = true;
                return false;
            }
        }
        std::vector<EncodedVideoPacket> packets;
        if (!vp9_->encode(image, packets)) {
            failed_ = true;
            return false;
        }
        for (const auto& p : packets) {
            if (!spool_write_record(spool_out_, p.bytes.data(), p.bytes.size(), p.is_key)) {
                fprintf(stderr, "IncrementalWebmWriter: spool write failed at frame %d.\n", num_frames_);
                failed_ = true;
                return false;
            }
            ++num_frames_;
        }
        return true;
    }
#endif

    std::vector<uint8_t> vp8;
    if (!encode_sd_image_to_vp8_frame(image, quality_, vp8)) {
        fprintf(stderr, "IncrementalWebmWriter: failed to encode frame %d as VP8.\n", num_frames_);
        failed_ = true;
        return false;
    }
    if (!spool_write_record(spool_out_, vp8.data(), vp8.size(), true)) {
        fprintf(stderr, "IncrementalWebmWriter: spool write failed at frame %d.\n", num_frames_);
        failed_ = true;
        return false;
    }
    ++num_frames_;
    return true;
}

void IncrementalWebmWriter::set_audio(const sd_audio_t* audio) {
    if (audio_owned_.data != nullptr) {
        free(audio_owned_.data);
        audio_owned_.data = nullptr;
    }
    audio_owned_ = sd_audio_t{};
    if (audio == nullptr || audio->data == nullptr || audio->sample_count == 0 || audio->channels == 0) {
        return;
    }
    const size_t n = static_cast<size_t>(audio->sample_count) * static_cast<size_t>(audio->channels);
    audio_owned_.data = static_cast<float*>(malloc(n * sizeof(float)));
    if (audio_owned_.data == nullptr) {
        return;  // no audio track rather than crash; video still muxes
    }
    memcpy(audio_owned_.data, audio->data, n * sizeof(float));
    audio_owned_.sample_rate  = audio->sample_rate;
    audio_owned_.channels     = audio->channels;
    audio_owned_.sample_count = audio->sample_count;
}

std::vector<uint8_t> IncrementalWebmWriter::finalize() {
#ifdef SD_USE_VPX
    // Drain any packets the encoder is still holding (lag=0 leaves none, but this is correct
    // regardless) before the spool is closed, so the container has every frame.
    if (use_vp9_ && vp9_ && !failed_ && spool_out_.is_open()) {
        std::vector<EncodedVideoPacket> tail;
        if (vp9_->finish(tail)) {
            for (const auto& p : tail) {
                if (!spool_write_record(spool_out_, p.bytes.data(), p.bytes.size(), p.is_key)) {
                    failed_ = true;
                    break;
                }
                ++num_frames_;
            }
        }
    }
#endif
    if (spool_out_.is_open()) {
        spool_out_.flush();
        spool_out_.close();
    }
    if (failed_ || num_frames_ == 0) {
        return {};
    }
    spool_in_.open(fs::path(spool_path_), std::ios::binary);
    if (!spool_in_) {
        fprintf(stderr, "IncrementalWebmWriter: cannot reopen spool %s\n", spool_path_.c_str());
        return {};
    }
    int next_read = 0;
    auto get_frame = [&](int i, std::vector<uint8_t>& out, bool& is_key) -> bool {
        if (i != next_read) {  // the mux core reads strictly in order; guard the assumption
            fprintf(stderr, "IncrementalWebmWriter: non-sequential frame read (%d != %d).\n", i, next_read);
            return false;
        }
        unsigned char hdr[5];
        spool_in_.read(reinterpret_cast<char*>(hdr), 5);
        if (!spool_in_ || spool_in_.gcount() != 5) {
            fprintf(stderr, "IncrementalWebmWriter: spool header read failed at frame %d.\n", i);
            return false;
        }
        const uint32_t len = static_cast<uint32_t>(hdr[0]) | (static_cast<uint32_t>(hdr[1]) << 8) |
                             (static_cast<uint32_t>(hdr[2]) << 16) | (static_cast<uint32_t>(hdr[3]) << 24);
        is_key             = hdr[4] != 0;
        out.resize(len);
        if (len > 0) {
            spool_in_.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(len));
            if (!spool_in_ || static_cast<uint32_t>(spool_in_.gcount()) != len) {
                fprintf(stderr, "IncrementalWebmWriter: spool body read failed at frame %d.\n", i);
                return false;
            }
        }
        ++next_read;
        return true;
    };

    const sd_audio_t* audio = audio_owned_.data != nullptr ? &audio_owned_ : nullptr;
    std::vector<uint8_t> bytes = mux_webm(width_, height_, num_frames_, fps_, audio, use_vp9_, get_frame);

    spool_in_.close();
    std::error_code ec;
    fs::remove(fs::path(spool_path_), ec);
    spool_path_.clear();

    if (bytes.empty()) {
        return {};
    }

    // B4 atomic publish: write .tmp then rename onto final_path.
    const std::string tmp_path = final_path_ + ".tmp";
    fs::remove(fs::path(tmp_path), ec);
    {
        std::ofstream out(fs::path(tmp_path), std::ios::binary | std::ios::trunc);
        if (!out) {
            fprintf(stderr, "IncrementalWebmWriter: cannot open %s\n", tmp_path.c_str());
            return {};
        }
        out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!out) {
            fprintf(stderr, "IncrementalWebmWriter: write to %s failed\n", tmp_path.c_str());
            return {};
        }
    }
    fs::rename(fs::path(tmp_path), fs::path(final_path_), ec);
    if (ec) {
        fprintf(stderr, "IncrementalWebmWriter: rename %s -> %s failed: %s\n",
                tmp_path.c_str(), final_path_.c_str(), ec.message().c_str());
        return {};
    }
    return bytes;
}

int create_webm_from_sd_images(const char* filename, sd_image_t* images, int num_images, int fps, int quality, const sd_audio_t* audio) {
    std::vector<uint8_t> webm_data = create_webm_from_sd_images_to_vector(images, num_images, fps, quality, audio);
    if (webm_data.empty()) {
        return -1;
    }
    if (!write_binary_file_bytes(filename, webm_data)) {
        perror("Error opening file for writing");
        return -1;
    }
    return 0;
}
#endif

std::vector<uint8_t> create_video_from_sd_images_to_vector(const std::string& output_format,
                                                           sd_image_t* images,
                                                           int num_images,
                                                           int fps,
                                                           int quality,
                                                           const sd_audio_t* audio,
                                                           bool consume_image_data) {
    std::string format = output_format;
    std::transform(format.begin(), format.end(), format.begin(),
                   [](unsigned char c) { return static_cast<char>(tolower(c)); });
    if (!format.empty() && format[0] == '.') {
        format.erase(format.begin());
    }

#ifdef SD_USE_WEBM
    if (format == "webm") {
        return create_webm_from_sd_images_to_vector(images, num_images, fps, quality, audio, consume_image_data);
    }
#endif

#ifdef SD_USE_WEBP
    if (format == "webp") {
        return create_animated_webp_from_sd_images_to_vector(images, num_images, fps, quality);
    }
#endif

    return create_mjpg_avi_from_sd_images_to_vector(images, num_images, fps, quality, audio);
}

int create_video_from_sd_images(const char* filename, sd_image_t* images, int num_images, int fps, int quality, const sd_audio_t* audio) {
    std::string path                = filename ? filename : "";
    auto pos                        = path.find_last_of('.');
    std::string ext                 = pos == std::string::npos ? "" : path.substr(pos);
    std::vector<uint8_t> video_data = create_video_from_sd_images_to_vector(ext, images, num_images, fps, quality, audio);
    if (video_data.empty()) {
        return -1;
    }
    if (!write_binary_file_bytes(filename, video_data)) {
        perror("Error opening file for writing");
        return -1;
    }
    return 0;
}

bool write_wav_to_file(const std::string& path,
                       const float* interleaved_samples,
                       uint64_t sample_count,
                       uint32_t channels,
                       uint32_t sample_rate) {
    if (interleaved_samples == nullptr || sample_count == 0 || channels == 0 || sample_rate == 0) {
        return false;
    }

    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    uint32_t bits_per_sample  = 16;
    uint32_t bytes_per_sample = bits_per_sample / 8;
    uint32_t block_align      = channels * bytes_per_sample;
    uint32_t byte_rate        = sample_rate * block_align;
    uint32_t data_size        = static_cast<uint32_t>(sample_count * channels * bytes_per_sample);
    uint32_t riff_size        = 36 + data_size;

    file.write("RIFF", 4);
    file.write(reinterpret_cast<const char*>(&riff_size), sizeof(riff_size));
    file.write("WAVE", 4);
    file.write("fmt ", 4);

    uint32_t fmt_size            = 16;
    uint16_t audio_format        = 1;
    uint16_t wav_channels        = static_cast<uint16_t>(channels);
    uint16_t wav_block_align     = static_cast<uint16_t>(block_align);
    uint16_t wav_bits_per_sample = static_cast<uint16_t>(bits_per_sample);
    file.write(reinterpret_cast<const char*>(&fmt_size), sizeof(fmt_size));
    file.write(reinterpret_cast<const char*>(&audio_format), sizeof(audio_format));
    file.write(reinterpret_cast<const char*>(&wav_channels), sizeof(wav_channels));
    file.write(reinterpret_cast<const char*>(&sample_rate), sizeof(sample_rate));
    file.write(reinterpret_cast<const char*>(&byte_rate), sizeof(byte_rate));
    file.write(reinterpret_cast<const char*>(&wav_block_align), sizeof(wav_block_align));
    file.write(reinterpret_cast<const char*>(&wav_bits_per_sample), sizeof(wav_bits_per_sample));

    file.write("data", 4);
    file.write(reinterpret_cast<const char*>(&data_size), sizeof(data_size));

    std::vector<int16_t> pcm(sample_count * channels);
    for (size_t i = 0; i < pcm.size(); ++i) {
        float sample = std::max(-1.0f, std::min(1.0f, interleaved_samples[i]));
        pcm[i]       = static_cast<int16_t>(std::lrint(sample * 32767.0f));
    }
    file.write(reinterpret_cast<const char*>(pcm.data()), static_cast<std::streamsize>(pcm.size() * sizeof(int16_t)));
    return file.good();
}
