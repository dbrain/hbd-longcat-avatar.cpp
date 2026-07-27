#ifndef __MEDIA_IO_H__
#define __MEDIA_IO_H__

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "stable-diffusion.h"

enum class EncodedImageFormat {
    JPEG,
    PNG,
    WEBP,
    UNKNOWN,
};

EncodedImageFormat encoded_image_format_from_path(const std::string& path);

std::vector<uint8_t> encode_image_to_vector(EncodedImageFormat format,
                                            const uint8_t* image,
                                            int width,
                                            int height,
                                            int channels,
                                            const std::string& parameters = "",
                                            int quality                   = 90);

bool write_image_to_file(const std::string& path,
                         const uint8_t* image,
                         int width,
                         int height,
                         int channels,
                         const std::string& parameters = "",
                         int quality                   = 90);

uint8_t* load_image_from_file(const char* image_path,
                              int& width,
                              int& height,
                              int expected_width   = 0,
                              int expected_height  = 0,
                              int expected_channel = 3);

bool load_sd_image_from_file(sd_image_t* image,
                             const char* image_path,
                             int expected_width   = 0,
                             int expected_height  = 0,
                             int expected_channel = 3);

uint8_t* load_image_from_memory(const char* image_bytes,
                                int len,
                                int& width,
                                int& height,
                                int expected_width   = 0,
                                int expected_height  = 0,
                                int expected_channel = 3);

int create_mjpg_avi_from_sd_images(const char* filename,
                                   sd_image_t* images,
                                   int num_images,
                                   int fps,
                                   int quality             = 90,
                                   const sd_audio_t* audio = nullptr);
std::vector<uint8_t> create_mjpg_avi_from_sd_images_to_vector(sd_image_t* images,
                                                              int num_images,
                                                              int fps,
                                                              int quality             = 90,
                                                              const sd_audio_t* audio = nullptr);

#ifdef SD_USE_WEBP
int create_animated_webp_from_sd_images(const char* filename,
                                        sd_image_t* images,
                                        int num_images,
                                        int fps,
                                        int quality = 90);
std::vector<uint8_t> create_animated_webp_from_sd_images_to_vector(sd_image_t* images,
                                                                   int num_images,
                                                                   int fps,
                                                                   int quality = 90);
#endif

#ifdef SD_USE_WEBM
int create_webm_from_sd_images(const char* filename,
                               sd_image_t* images,
                               int num_images,
                               int fps,
                               int quality             = 90,
                               const sd_audio_t* audio = nullptr);
std::vector<uint8_t> create_webm_from_sd_images_to_vector(sd_image_t* images,
                                                          int num_images,
                                                          int fps,
                                                          int quality             = 90,
                                                          const sd_audio_t* audio = nullptr);
#endif

// One encoded video frame as it comes out of the codec, before muxing.
struct EncodedWebmPacket {
    std::vector<uint8_t> data;
    bool keyframe = false;
};

// Streaming WebM encode. Encodes each frame on arrival and retains only the compressed packet,
// so a long chain never holds its decoded timeline in RAM (a raw 1280x704 frame is 2.7 MB; its
// VP9 packet is tens of KB). begin() returns false when VP9 is unavailable, in which case the
// caller should fall back to accumulating frames and calling the one-shot encoder.
class Vp9Encoder;
class IncrementalWebmEncoder {
public:
    IncrementalWebmEncoder();
    ~IncrementalWebmEncoder();
    IncrementalWebmEncoder(const IncrementalWebmEncoder&)            = delete;
    IncrementalWebmEncoder& operator=(const IncrementalWebmEncoder&) = delete;

    bool begin(int width, int height, int fps);
    // The caller retains ownership of `image` and may free it as soon as this returns.
    bool append(const sd_image_t& image);
    std::vector<uint8_t> finalize(const sd_audio_t* audio, int quality);

    int  frames() const { return frames_; }
    bool failed() const { return failed_; }
    bool active() const { return encoder_ != nullptr; }

private:
    std::unique_ptr<Vp9Encoder> encoder_;
    std::vector<EncodedWebmPacket> packets_;
    int  width_  = 0;
    int  height_ = 0;
    int  fps_    = 0;
    int  frames_ = 0;
    bool failed_ = false;
};


int create_video_from_sd_images(const char* filename,
                                sd_image_t* images,
                                int num_images,
                                int fps,
                                int quality             = 90,
                                const sd_audio_t* audio = nullptr);
std::vector<uint8_t> create_video_from_sd_images_to_vector(const std::string& output_format,
                                                           sd_image_t* images,
                                                           int num_images,
                                                           int fps,
                                                           int quality             = 90,
                                                           const sd_audio_t* audio = nullptr);

bool write_wav_to_file(const std::string& path,
                       const float* interleaved_samples,
                       uint64_t sample_count,
                       uint32_t channels,
                       uint32_t sample_rate);

#endif  // __MEDIA_IO_H__
