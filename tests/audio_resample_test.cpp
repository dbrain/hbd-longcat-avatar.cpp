// Standalone measurement of the drive-audio resampler.
//
// Builds with nothing but the standard library:
//   g++ -O2 -std=c++17 -I../src audio_resample_test.cpp -o /tmp/rs && /tmp/rs
//
// It compares the NEW windowed-sinc converter against the two-tap linear
// interpolation it replaces, on the exact conversion the drive path does most
// (48 kHz -> 16 kHz). The point is that linear interpolation has no anti-alias
// filter, so everything above the output's 8 kHz Nyquist folds back down into
// the band the audio encoder actually reads.

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

#include "audio_resample.hpp"

namespace {

constexpr double kPi = 3.14159265358979323846;

// The old implementation, verbatim in behaviour: two-tap linear interpolation,
// no filtering of any kind.
std::vector<float> resample_linear_old(const std::vector<float>& in, uint32_t in_rate, uint32_t out_rate) {
    if (in_rate == out_rate) return in;
    const double ratio = static_cast<double>(out_rate) / static_cast<double>(in_rate);
    const size_t out_n = static_cast<size_t>(in.size() * ratio);
    std::vector<float> out(out_n);
    for (size_t i = 0; i < out_n; i++) {
        const double src = i / ratio;
        const size_t i0  = static_cast<size_t>(src);
        const size_t i1  = std::min(i0 + 1, in.size() - 1);
        const float  w   = static_cast<float>(src - i0);
        out[i] = in[i0] * (1 - w) + in[i1] * w;
    }
    return out;
}

std::vector<float> tone(double hz, uint32_t rate, double seconds, double amp = 1.0) {
    const size_t n = static_cast<size_t>(rate * seconds);
    std::vector<float> out(n);
    for (size_t i = 0; i < n; i++) {
        out[i] = static_cast<float>(amp * std::sin(2.0 * kPi * hz * i / rate));
    }
    return out;
}

// Linear sweep from lo to hi over the whole clip.
std::vector<float> sweep(double lo, double hi, uint32_t rate, double seconds) {
    const size_t n = static_cast<size_t>(rate * seconds);
    std::vector<float> out(n);
    double phase = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double t = static_cast<double>(i) / n;
        const double f = lo + (hi - lo) * t;
        phase += 2.0 * kPi * f / rate;
        out[i] = static_cast<float>(std::sin(phase));
    }
    return out;
}

// RMS amplitude at one frequency, via a direct (windowed) correlation. Hann
// window so a tone that is not bin-centred is still measured accurately.
double amplitude_at(const std::vector<float>& x, double hz, uint32_t rate) {
    double re = 0.0, im = 0.0, wsum = 0.0;
    for (size_t i = 0; i < x.size(); i++) {
        const double w = 0.5 - 0.5 * std::cos(2.0 * kPi * i / (x.size() - 1));
        const double a = 2.0 * kPi * hz * i / rate;
        re += w * x[i] * std::cos(a);
        im += w * x[i] * std::sin(a);
        wsum += w;
    }
    return 2.0 * std::sqrt(re * re + im * im) / wsum;
}

// Total energy in [lo, hi) Hz, by brute-force correlation on a 25 Hz grid.
double band_energy(const std::vector<float>& x, double lo, double hi, uint32_t rate) {
    double acc = 0.0;
    for (double f = lo; f < hi; f += 25.0) {
        const double a = amplitude_at(x, f, rate);
        acc += a * a;
    }
    return acc;
}

double db(double ratio) { return 20.0 * std::log10(std::max(ratio, 1e-12)); }
double db_power(double ratio) { return 10.0 * std::log10(std::max(ratio, 1e-12)); }

int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) failures++;
}

}  // namespace

int main() {
    using sd_audio_resample::resample;

    std::printf("\n=== 1. Out-of-band tone must NOT fold into the passband (48k -> 16k) ===\n");
    std::printf("    A tone above the output's 8 kHz Nyquist has nowhere legitimate to go.\n");
    std::printf("    Without an anti-alias filter it reappears at |f - 16000|.\n\n");
    struct { double src_hz; double alias_hz; } folds[] = {
        {20000.0,  4000.0},   // 20k -> 4k
        {14000.0,  2000.0},   // 14k -> 2k
        {9000.0,   7000.0},   // 9k  -> 7k  (just above Nyquist, the hardest case)
        {22000.0,  6000.0},   // 22k -> 6k
    };
    for (auto& f : folds) {
        const auto src = tone(f.src_hz, 48000, 1.0);
        const auto old_out = resample_linear_old(src, 48000, 16000);
        std::vector<float> new_out;
        resample(src, 48000, 16000, new_out);
        const double a_old = amplitude_at(old_out, f.alias_hz, 16000);
        const double a_new = amplitude_at(new_out, f.alias_hz, 16000);
        std::printf("  %6.0f Hz source -> alias at %5.0f Hz:  old %7.2f dBFS   new %7.2f dBFS   (%.1f dB better)\n",
                    f.src_hz, f.alias_hz, db(a_old), db(a_new), db(a_old) - db(a_new));
        check(db(a_new) < -60.0, "new converter rejects the fold-down below -60 dBFS");
    }

    std::printf("\n=== 2. Broadband sweep: energy that should not exist in 6-8 kHz ===\n");
    std::printf("    A 0-24 kHz sweep decimated to 16 kHz should keep only its 0-8 kHz part.\n\n");
    {
        const auto src = sweep(0.0, 24000.0, 48000, 2.0);
        const auto old_out = resample_linear_old(src, 48000, 16000);
        std::vector<float> new_out;
        resample(src, 48000, 16000, new_out);
        // Reference: the same sweep generated directly at 16 kHz only up to 8 kHz,
        // occupying the same fraction of the clip, is what "correct" looks like.
        for (auto band : {std::pair<double,double>{6000, 7000}, {7000, 8000}, {4000, 5000}}) {
            const double e_old = band_energy(old_out, band.first, band.second, 16000);
            const double e_new = band_energy(new_out, band.first, band.second, 16000);
            std::printf("  %4.0f-%4.0f Hz:  old %7.2f dB   new %7.2f dB   old is %+.2f dB hotter\n",
                        band.first, band.second, db_power(e_old), db_power(e_new),
                        db_power(e_old) - db_power(e_new));
        }
        std::printf("  (6-7k and 7-8k are the bands the field measurement called out at +5.7 / +7.8 dB.)\n");
        const double e6_old = band_energy(old_out, 6000, 7000, 16000);
        const double e6_new = band_energy(new_out, 6000, 7000, 16000);
        check(db_power(e6_old) - db_power(e6_new) > 3.0, "old converter is measurably hotter at 6-7 kHz");
    }

    std::printf("\n=== 3. Passband must be untouched ===\n\n");
    for (double hz : {100.0, 1000.0, 3000.0, 6000.0}) {
        const auto src = tone(hz, 48000, 1.0);
        std::vector<float> new_out;
        resample(src, 48000, 16000, new_out);
        const double a = amplitude_at(new_out, hz, 16000);
        std::printf("  %5.0f Hz: amplitude %.4f (%+.3f dB)\n", hz, a, db(a));
        check(std::fabs(db(a)) < 0.15, "passband gain within 0.15 dB of unity");
    }

    std::printf("\n=== 4. No group delay (the drive track is aligned by sample index) ===\n\n");
    {
        // An IMPULSE, not a tone. A 1 kHz tone is periodic every 16 samples at
        // 16 kHz, so lags of 0, +/-16, +/-32 all correlate identically and the
        // argmax is meaningless. An impulse has one unambiguous peak.
        std::vector<float> src(48000, 0.f);
        const size_t at_in = 24000;              // 0.5 s in
        src[at_in] = 1.f;
        std::vector<float> out;
        resample(src, 48000, 16000, out);
        const size_t expect = at_in / 3;         // 48k -> 16k
        size_t peak = 0; double best = -1e30;
        for (size_t i = 0; i < out.size(); i++) {
            if (std::fabs(out[i]) > best) { best = std::fabs(out[i]); peak = i; }
        }
        std::printf("  impulse at input sample %zu -> output peak at %zu (expected %zu)\n",
                    at_in, peak, expect);
        check(peak == expect, "converter is zero-delay (impulse peak lands where it should)");

        // And the kernel is symmetric about that peak — an asymmetric response
        // would be a fractional delay even with the peak in the right bin.
        double asym = 0.0;
        for (size_t k = 1; k <= 8; k++) {
            asym = std::max(asym, static_cast<double>(std::fabs(out[peak - k] - out[peak + k])));
        }
        std::printf("  max |h[peak-k] - h[peak+k]| over k=1..8: %.3e\n", asym);
        check(asym < 1e-6, "impulse response is symmetric (linear phase, no fractional delay)");
    }

    std::printf("\n=== 5. Upsampling and non-integer ratios still work ===\n\n");
    for (auto rate : {8000u, 22050u, 44100u, 32000u, 48000u, 96000u}) {
        const double hz = 1000.0;
        const auto src = tone(hz, rate, 0.5);
        std::vector<float> out;
        resample(src, rate, 16000, out);
        const double a = amplitude_at(out, hz, 16000);
        const size_t want = static_cast<size_t>(src.size() * 16000.0 / rate);
        std::printf("  %6u Hz -> 16000 Hz: %zu samples (expected ~%zu), 1 kHz amplitude %.4f (%+.3f dB)\n",
                    rate, out.size(), want, a, db(a));
        check(std::fabs(db(a)) < 0.15 && out.size() == want, "rate converts with correct length and gain");
    }

    std::printf("\n=== 6. Stereo: channels convert independently, image preserved ===\n\n");
    {
        // Two channels with a deliberate phase difference — the thing a mono
        // downmix destroys and the thing an independent per-channel convert
        // must keep.
        const uint32_t rate = 48000;
        const size_t frames = rate / 2;
        std::vector<float> inter(frames * 2);
        for (size_t i = 0; i < frames; i++) {
            inter[i * 2 + 0] = static_cast<float>(std::sin(2.0 * kPi * 1000.0 * i / rate));
            inter[i * 2 + 1] = static_cast<float>(std::sin(2.0 * kPi * 1000.0 * i / rate + 1.0));
        }
        std::vector<float> out;
        sd_audio_resample::resample_interleaved(inter, 2, rate, 16000, out);
        const size_t out_frames = out.size() / 2;
        std::vector<float> l(out_frames), r(out_frames);
        for (size_t i = 0; i < out_frames; i++) { l[i] = out[i * 2]; r[i] = out[i * 2 + 1]; }
        // Normalised cross-correlation at lag 0: for a 1.0 rad phase offset the
        // expected value is cos(1.0) = 0.5403.
        double num = 0, dl = 0, dr = 0;
        for (size_t i = 64; i + 64 < out_frames; i++) { num += l[i] * r[i]; dl += l[i] * l[i]; dr += r[i] * r[i]; }
        const double ncc = num / std::sqrt(dl * dr);
        std::printf("  L/R correlation after convert: %.4f (expected cos(1.0) = %.4f)\n", ncc, std::cos(1.0));
        check(std::fabs(ncc - std::cos(1.0)) < 0.01, "stereo phase relationship survives the convert");
        check(out_frames == frames / 3, "interleaved length is right");
    }

    std::printf("\n%s (%d failure%s)\n\n", failures == 0 ? "ALL CHECKS PASSED" : "CHECKS FAILED",
                failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
