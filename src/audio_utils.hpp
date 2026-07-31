#ifndef __SD_AUDIO_UTILS_HPP__
#define __SD_AUDIO_UTILS_HPP__

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "audio_resample.hpp"

namespace sd::audio {

// Decode the PCM/float WAV formats used by the model audio encoders, mix to mono,
// and resample to 16 kHz. Kept independent from a particular model so the
// LongCat, S2V, and InfiniteTalk front ends share identical drive-audio semantics.
//
// Mono is right here: every consumer of this loader is a whisper/wav2vec2 front
// end, which is a mono model. The rate conversion is the shared anti-aliased one
// — it used to be two-tap linear interpolation with no filter, identical to the
// bug fixed in LONGCAT_AUDIO::load_wav_16k, and these front ends were folding
// 8-24 kHz down into the encoder's band exactly the same way.
inline bool load_wav_16k_mono(const std::string& path, std::vector<float>& out) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }
    const std::vector<char> bytes((std::istreambuf_iterator<char>(file)), {});
    if (bytes.size() < 44 || std::memcmp(bytes.data(), "RIFF", 4) != 0 ||
        std::memcmp(bytes.data() + 8, "WAVE", 4) != 0) {
        return false;
    }

    const auto read_u16 = [&](size_t offset) {
        uint16_t value;
        std::memcpy(&value, bytes.data() + offset, sizeof(value));
        return value;
    };
    const auto read_u32 = [&](size_t offset) {
        uint32_t value;
        std::memcpy(&value, bytes.data() + offset, sizeof(value));
        return value;
    };

    uint16_t format = 1;
    uint16_t channels = 1;
    uint16_t bits_per_sample = 16;
    uint32_t sample_rate = 16000;
    size_t data_offset = 0;
    size_t data_size = 0;
    for (size_t offset = 12; offset + 8 <= bytes.size();) {
        const uint32_t chunk_size = read_u32(offset + 4);
        if (std::memcmp(bytes.data() + offset, "fmt ", 4) == 0 && chunk_size >= 16 &&
            offset + 8 + chunk_size <= bytes.size()) {
            format = read_u16(offset + 8);
            channels = read_u16(offset + 10);
            sample_rate = read_u32(offset + 12);
            bits_per_sample = read_u16(offset + 22);
        } else if (std::memcmp(bytes.data() + offset, "data", 4) == 0) {
            data_offset = offset + 8;
            data_size = std::min<size_t>(chunk_size, bytes.size() - data_offset);
            break;
        }
        const size_t next = offset + 8 + static_cast<size_t>(chunk_size) + (chunk_size & 1U);
        if (next <= offset || next > bytes.size()) {
            return false;
        }
        offset = next;
    }
    if (data_offset == 0 || channels == 0 || sample_rate == 0) {
        return false;
    }

    const size_t sample_bytes = bits_per_sample / 8;
    if (sample_bytes == 0 || data_size / sample_bytes < channels) {
        return false;
    }
    const size_t frames = data_size / (sample_bytes * channels);
    std::vector<float> mono(frames);
    const char* data = bytes.data() + data_offset;
    for (size_t frame = 0; frame < frames; ++frame) {
        float sum = 0.0f;
        for (uint16_t channel = 0; channel < channels; ++channel) {
            const char* sample = data + (frame * channels + channel) * sample_bytes;
            float value = 0.0f;
            if (format == 1 && bits_per_sample == 16) {
                int16_t pcm;
                std::memcpy(&pcm, sample, sizeof(pcm));
                value = pcm / 32768.0f;
            } else if (format == 1 && bits_per_sample == 32) {
                int32_t pcm;
                std::memcpy(&pcm, sample, sizeof(pcm));
                value = pcm / 2147483648.0f;
            } else if (format == 3 && bits_per_sample == 32) {
                std::memcpy(&value, sample, sizeof(value));
            } else {
                return false;
            }
            sum += value;
        }
        mono[frame] = sum / channels;
    }

    if (sample_rate == 16000 || mono.empty()) {
        out = std::move(mono);
        return true;
    }
    sd_audio_resample::resample(mono, sample_rate, 16000, out);
    return !out.empty();
}

}  // namespace sd::audio

#endif  // __SD_AUDIO_UTILS_HPP__
