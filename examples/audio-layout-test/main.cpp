// Standalone reproduction harness for the LTX 2-channel audio layout bug.
//
// The audio VAE always emits 2 channels. waveform_to_sd_audio() lays the buffer
// out PLANAR (channel-major: all ch0 samples, then all ch1 samples) via
// permute({1,0,2,3}); ChainAudioAcc rejoins planar too. But the output writers
// audio_to_pcm16_bytes() (AVI) and encode_audio_to_opus() (WEBM) read the buffer
// as INTERLEAVED (L,R,L,R...). For 2 channels that scrambles the audio.
//
// This harness hands the real create_*_from_sd_images() writers a hand-built
// PLANAR buffer (byte-identical layout to what the model emits) so the scramble
// reproduces deterministically with no GPU / no model. Left channel = 440 Hz,
// right channel = 880 Hz. A correct writer keeps them distinct; the buggy writer
// collapses them and doubles the pitch.
//
//   build:  cmake --build /src/build --target sd-audio-layout-test
//   run:    /src/build/bin/sd-audio-layout-test <out_dir>

#include "common/media_io.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    std::string outdir = argc > 1 ? argv[1] : ".";

    const uint32_t sr       = 48000;
    const double   secs     = 2.0;
    const uint64_t N        = static_cast<uint64_t>(sr * secs);  // per-channel samples
    const uint32_t channels = 2;

    // PLANAR (channel-major): [ ch0=440Hz over [0,N) ][ ch1=880Hz over [N,2N) ]
    // This is exactly what waveform_to_sd_audio() writes into sd_audio_t.data.
    std::vector<float> data(static_cast<size_t>(N) * channels);
    for (uint64_t i = 0; i < N; ++i) {
        double t                                = static_cast<double>(i) / sr;
        data[i]                                 = 0.6f * static_cast<float>(std::sin(2.0 * M_PI * 440.0 * t));
        data[static_cast<size_t>(N) + i]        = 0.6f * static_cast<float>(std::sin(2.0 * M_PI * 880.0 * t));
    }

    sd_audio_t audio;
    audio.sample_rate  = sr;
    audio.channels     = channels;
    audio.sample_count = N;
    audio.data         = data.data();

    // Minimal video: small gray frames spanning the same 2 s.
    const int      fps = 24;
    const int      num = static_cast<int>(fps * secs);
    const uint32_t W = 32, H = 32, C = 3;
    std::vector<std::vector<uint8_t>> frame_bufs(num);
    std::vector<sd_image_t>           frames(num);
    for (int f = 0; f < num; ++f) {
        frame_bufs[f].assign(static_cast<size_t>(W) * H * C, static_cast<uint8_t>(20 + f));
        frames[f].width   = W;
        frames[f].height  = H;
        frames[f].channel = C;
        frames[f].data    = frame_bufs[f].data();
    }

    std::string avi  = outdir + "/clip.avi";
    std::string webm = outdir + "/clip.webm";
    int         ra   = create_mjpg_avi_from_sd_images(avi.c_str(), frames.data(), num, fps, 90, &audio);
    int         rw   = create_webm_from_sd_images(webm.c_str(), frames.data(), num, fps, 90, &audio);

    printf("audio: %u Hz, %u ch, %llu samples/ch, PLANAR (ch0=440Hz [0,N), ch1=880Hz [N,2N))\n",
           sr, channels, static_cast<unsigned long long>(N));
    printf("wrote AVI  (rc=%d) %s\n", ra, avi.c_str());
    printf("wrote WEBM (rc=%d) %s\n", rw, webm.c_str());
    return (ra == 0 && rw == 0) ? 0 : 1;
}
