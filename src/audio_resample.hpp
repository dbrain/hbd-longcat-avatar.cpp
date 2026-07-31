#ifndef __SD_AUDIO_RESAMPLE_HPP__
#define __SD_AUDIO_RESAMPLE_HPP__

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Sample-rate conversion for the drive-audio loaders.
//
// Deliberately free of every other header in this tree (no ggml, no logging) so
// it can be compiled and measured on its own — see tests/audio_resample_test.cpp,
// which is what produced the stopband numbers quoted below.
//
// WHY THIS EXISTS. Both drive-audio loaders — LONGCAT_AUDIO::load_wav_16k_mono
// and sd::audio::load_wav_16k_mono — converted to 16 kHz with a two-tap linear
// interpolation and NO anti-alias filter. Linear interpolation is a very poor
// low-pass (a triangular kernel, first null at the input rate), so decimating
// 48 kHz to 16 kHz folded everything in 8-24 kHz straight back down into the
// 0-8 kHz band the model actually reads. Measured on real jobs: +5.7 dB of
// spurious energy at 6-7 kHz and +7.8 dB at 7-8 kHz, and in the encoder's own
// log-mel input the top 6 mel bins came out 6.97 dB wrong, with aliasing
// accounting for 54% of total mel-input error.
//
// The anti-alias filter has to be applied BEFORE the rate is reduced, which
// means it has to BE the interpolation kernel. This is why the existing
// LONGCAT_AUDIO_LOWPASS biquad could never have covered for it: that runs on
// the already-decimated signal, and no filter can unfold energy that has
// already folded.
namespace sd_audio_resample {

    // Zero-crossings of the sinc retained each side of centre. Sets the
    // TRANSITION width; the window below sets the stopband depth. 24 puts the
    // transition inside a few percent of the cutoff, which at 48k->16k means
    // the 7-8 kHz band the mel front end cares about is fully passband.
    constexpr int kSincZeros = 24;

    // Kernel samples per input sample in the lookup table. The kernel is
    // tabulated once and read with linear interpolation so the per-tap cost is
    // two loads and a lerp instead of a sin() — without it a three-minute drive
    // track would spend minutes in libm rather than ~2 s.
    constexpr int kKernelDensity = 512;

    // Linear-phase windowed-sinc rate conversion of one channel.
    //
    // The cutoff tracks the LOWER of the two Nyquist limits, so the same code is
    // correct upsampling (cutoff = the output's Nyquist, nothing to reject) and
    // downsampling (cutoff = the output's Nyquist expressed against the input
    // rate, which is what rejects the fold-down).
    //
    // Linear phase and centred on the output sample instant, so this introduces
    // NO group delay — important, because the drive track is aligned to the
    // video timeline by sample index and a delay here would read as lip-sync
    // drift rather than as a filter.
    inline void resample(const std::vector<float>& in,
                         uint32_t                  in_rate,
                         uint32_t                  out_rate,
                         std::vector<float>&       out) {
        if (in.empty() || in_rate == 0 || out_rate == 0) {
            out.clear();
            return;
        }
        if (in_rate == out_rate) {
            out = in;
            return;
        }

        const double ratio  = static_cast<double>(out_rate) / static_cast<double>(in_rate);
        const double cutoff = std::min(1.0, ratio) * 0.5;   // cycles per INPUT sample
        const double half   = kSincZeros / (2.0 * cutoff);  // kernel half-width, input samples

        // One side of the symmetric kernel. Un-normalised: the running weight
        // sum below divides it out per output sample, which fixes the unity DC
        // gain exactly and also absorbs the tiny phase-dependent ripple a
        // finite window leaves behind.
        const size_t       table_n = static_cast<size_t>(half * kKernelDensity) + 2;
        std::vector<float> table(table_n, 0.f);
        for (size_t t = 0; t < table_n; ++t) {
            const double dx = static_cast<double>(t) / kKernelDensity;
            if (dx > half) {
                continue;
            }
            const double x    = 2.0 * cutoff * dx;
            const double sinc = (x < 1e-9) ? 1.0 : std::sin(M_PI * x) / (M_PI * x);
            // Blackman window, evaluated on the |dx| half. Its ~-74 dB sidelobe
            // floor is what sets the stopband, comfortably under a 16-bit
            // source's own noise floor.
            const double phase = (dx / half + 1.0) * 0.5;
            const double win   = 0.42 - 0.5 * std::cos(2.0 * M_PI * phase) + 0.08 * std::cos(4.0 * M_PI * phase);
            table[t]           = static_cast<float>(sinc * win);
        }

        const long   n_in  = static_cast<long>(in.size());
        const size_t n_out = static_cast<size_t>(std::floor(static_cast<double>(in.size()) * ratio));
        const long   taps  = static_cast<long>(std::ceil(half));
        out.assign(n_out, 0.f);

        for (size_t i = 0; i < n_out; ++i) {
            const double centre = static_cast<double>(i) / ratio;  // fractional input index
            const long   base   = static_cast<long>(std::floor(centre));
            double       acc = 0.0, wsum = 0.0;
            for (long j = base - taps; j <= base + taps; ++j) {
                const double dx = std::fabs(centre - static_cast<double>(j));
                if (dx > half) {
                    continue;
                }
                const double pos = dx * kKernelDensity;
                const size_t p0  = static_cast<size_t>(pos);
                if (p0 + 1 >= table_n) {
                    continue;
                }
                const double frac = pos - static_cast<double>(p0);
                const double k    = table[p0] * (1.0 - frac) + table[p0 + 1] * frac;
                // Edge-EXTEND rather than zero-pad. A step to silence at the
                // clip boundary is a far bigger transient than repeating the
                // end sample, and the wsum normalisation keeps the gain right
                // either way.
                const long idx = std::min<long>(std::max<long>(j, 0), n_in - 1);
                acc += k * static_cast<double>(in[static_cast<size_t>(idx)]);
                wsum += k;
            }
            out[i] = static_cast<float>(wsum != 0.0 ? acc / wsum : 0.0);
        }
    }

    // Interleaved multi-channel convenience wrapper. Each channel is converted
    // independently, which is correct here: the filter is linear phase and
    // identical per channel, so the inter-channel phase relationship the stereo
    // image lives in is preserved exactly.
    inline void resample_interleaved(const std::vector<float>& in,
                                     uint32_t                  channels,
                                     uint32_t                  in_rate,
                                     uint32_t                  out_rate,
                                     std::vector<float>&       out) {
        if (channels <= 1) {
            resample(in, in_rate, out_rate, out);
            return;
        }
        if (in.empty() || in_rate == 0 || out_rate == 0) {
            out.clear();
            return;
        }
        if (in_rate == out_rate) {
            out = in;
            return;
        }
        const size_t       frames = in.size() / channels;
        std::vector<float> plane(frames), converted;
        out.clear();
        for (uint32_t c = 0; c < channels; ++c) {
            for (size_t f = 0; f < frames; ++f) {
                plane[f] = in[f * channels + c];
            }
            resample(plane, in_rate, out_rate, converted);
            if (out.empty()) {
                out.assign(converted.size() * channels, 0.f);
            }
            const size_t n = std::min<size_t>(converted.size(), out.size() / channels);
            for (size_t f = 0; f < n; ++f) {
                out[f * channels + c] = converted[f];
            }
        }
    }

}  // namespace sd_audio_resample

#endif  // __SD_AUDIO_RESAMPLE_HPP__
