#ifndef __MEDIA_IO_H__
#define __MEDIA_IO_H__

#include <cstdint>
#include <fstream>
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
                                                          const sd_audio_t* audio = nullptr,
                                                          bool consume_image_data = false);

// Streaming WebM encoder. Encodes+spools each frame's VP8 keyframe to disk as it arrives so the
// caller never has to hold the whole decoded timeline in RAM; finalize() replays the spool through
// the same mux core the whole-array encoder uses (byte-identical output) with the length-matched
// audio, and publishes the container atomically (.tmp→rename). Additive — the whole-array API is
// unchanged. Not copyable (owns a spool + a buffered audio track).
class IncrementalWebmWriter {
public:
    IncrementalWebmWriter();
    ~IncrementalWebmWriter();
    IncrementalWebmWriter(const IncrementalWebmWriter&)            = delete;
    IncrementalWebmWriter& operator=(const IncrementalWebmWriter&) = delete;

    // Begin an encode targeting final_path (VP8 spool goes to final_path + ".vp8spool").
    bool open(const std::string& final_path, int fps, int quality);
    // Encode one frame and append it to the spool. Output dims are captured from the first frame
    // (mirrors create_webm's images[0]). The caller retains ownership of `image` and frees it after.
    bool append_video_frame(const sd_image_t& image);
    // Buffer the whole audio track (deep-copied) to mux at finalize().
    void set_audio(const sd_audio_t* audio);
    // Mux spool + audio, publish final_path atomically, and return the encoded bytes ({} on failure).
    std::vector<uint8_t> finalize();

    int  frame_count() const { return num_frames_; }
    bool failed() const { return failed_; }

private:
    std::string   final_path_;
    std::string   spool_path_;
    std::ofstream spool_out_;
    std::ifstream spool_in_;
    int           fps_        = 0;
    int           quality_    = 90;
    int           width_      = 0;
    int           height_     = 0;
    int           num_frames_ = 0;
    bool          failed_     = false;
    sd_audio_t    audio_owned_{};  // owned deep copy (data via malloc/free)
};
#endif

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
                                                           const sd_audio_t* audio = nullptr,
                                                           bool consume_image_data = false);

bool write_wav_to_file(const std::string& path,
                       const float* interleaved_samples,
                       uint64_t sample_count,
                       uint32_t channels,
                       uint32_t sample_rate);

#endif  // __MEDIA_IO_H__
