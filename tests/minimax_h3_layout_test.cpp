// Dumps the MiniMax-H3 packed-sequence layout as JSON so it can be diffed row-for-row against the
// Python reference.  The layout header is deliberately backend-free, so this builds standalone:
//
//   g++ -std=c++17 -I src -O2 -o /tmp/h3_layout tests/minimax_h3_layout_test.cpp
//   ./h3_layout > cpp.json
//   python3 ~/handoffs/longcat-avatar.cpp/minimax-h3/tools/ref_layout.py > ref.json
//   python3 ~/handoffs/longcat-avatar.cpp/minimax-h3/tools/diff_layout.py ref.json cpp.json
//
// The case list must stay in lockstep with CASES in ref_layout.py.

#include <cstdio>
#include <string>
#include <vector>

#include "model/diffusion/minimax_h3_layout.hpp"

using namespace MiniMaxH3;

namespace {

    struct Case {
        std::string name;
        LayoutRequest req;
    };

    Case make(const std::string& name, int64_t text_len, int64_t latent_t, int64_t latent_h, int64_t latent_w, int64_t audio_t) {
        Case c;
        c.name         = name;
        c.req.text_len = text_len;
        c.req.latent_t = latent_t;
        c.req.latent_h = latent_h;
        c.req.latent_w = latent_w;
        c.req.audio_t  = audio_t;
        return c;
    }

    Reference ref_image(int64_t h, int64_t w) {
        Reference r;
        r.kind     = RefKind::Image;
        r.latent_h = h;
        r.latent_w = w;
        return r;
    }

    Reference ref_audio(int64_t t) {
        Reference r;
        r.kind        = RefKind::Audio;
        r.ref_audio_t = t;
        return r;
    }

    Reference ref_video(RefKind kind, int64_t audio_t, int64_t latent_t, int64_t h, int64_t w) {
        Reference r;
        r.kind        = kind;
        r.ref_audio_t = audio_t;
        r.latent_t    = latent_t;
        r.latent_h    = h;
        r.latent_w    = w;
        return r;
    }

    std::vector<Case> build_cases() {
        std::vector<Case> cases;

        cases.push_back(make("t2va_small", 8, 6, 24, 42, 13));
        cases.push_back(make("t2va_768x1344", 77, 31, 48, 84, 124));
        cases.push_back(make("t2va_square", 16, 6, 32, 32, 13));
        cases.push_back(make("t2va_wide", 16, 6, 16, 96, 13));
        cases.push_back(make("t2va_one_frame", 4, 1, 32, 32, 2));
        cases.push_back(make("t2va_long", 4, 23, 16, 16, 90));

        {
            Case c            = make("fl2va_first", 16, 6, 24, 42, 13);
            c.req.frame_count = 21;
            c.req.keyframes   = {Keyframe{0}};
            cases.push_back(c);
        }
        {
            Case c            = make("fl2va_first_last", 16, 6, 24, 42, 13);
            c.req.frame_count = 21;
            c.req.keyframes   = {Keyframe{0}, Keyframe{20}};
            cases.push_back(c);
        }
        {
            Case c     = make("ref2va_image", 16, 6, 24, 42, 13);
            c.req.refs = {ref_image(32, 32)};
            cases.push_back(c);
        }
        {
            Case c     = make("ref2va_two_images", 16, 6, 24, 42, 13);
            c.req.refs = {ref_image(32, 32), ref_image(16, 64)};
            cases.push_back(c);
        }
        {
            Case c     = make("ref2va_audio", 16, 6, 24, 42, 13);
            c.req.refs = {ref_audio(9)};
            cases.push_back(c);
        }
        {
            Case c     = make("ref2va_video_audio", 16, 6, 24, 42, 13);
            c.req.refs = {ref_video(RefKind::VideoAudio, 9, 3, 16, 28)};
            cases.push_back(c);
        }
        {
            Case c     = make("ref2va_video_rt_dominates", 16, 6, 24, 42, 13);
            c.req.refs = {ref_video(RefKind::Video, 40, 2, 16, 16)};
            cases.push_back(c);
        }
        {
            Case c     = make("ref2va_video_spans_dominate", 16, 6, 24, 42, 13);
            c.req.refs = {ref_video(RefKind::Video, 1, 8, 16, 16)};
            cases.push_back(c);
        }
        {
            Case c     = make("ref2va_mixed", 16, 6, 24, 42, 13);
            c.req.refs = {ref_image(32, 32),
                          ref_audio(5),
                          ref_video(RefKind::VideoAudio, 7, 3, 16, 28),
                          ref_image(24, 24)};
            cases.push_back(c);
        }
        {
            Case c     = make("ref2va_zero_audio", 16, 6, 24, 42, 13);
            c.req.refs = {ref_audio(0), ref_image(32, 32)};
            cases.push_back(c);
        }

        return cases;
    }

    void emit(const Case& c, bool last) {
        PackedLayout layout = PackedLayout::build(c.req);

        printf("  {\n");
        printf("   \"audio_pos\": [");
        for (size_t i = 0; i < layout.audio_pos.size(); i++) {
            printf("%s%lld", i ? ", " : "", static_cast<long long>(layout.audio_pos[i]));
        }
        printf("],\n");
        printf("   \"audio_update\": [");
        for (size_t i = 0; i < layout.audio_update.size(); i++) {
            printf("%s%s", i ? ", " : "", layout.audio_update[i] ? "true" : "false");
        }
        printf("],\n");
        printf("   \"img_pos\": [");
        for (size_t i = 0; i < layout.img_pos.size(); i++) {
            printf("%s%lld", i ? ", " : "", static_cast<long long>(layout.img_pos[i]));
        }
        printf("],\n");
        printf("   \"img_update\": [");
        for (size_t i = 0; i < layout.img_update.size(); i++) {
            printf("%s%s", i ? ", " : "", layout.img_update[i] ? "true" : "false");
        }
        printf("],\n");
        printf("   \"name\": \"%s\",\n", c.name.c_str());
        printf("   \"position_ids\": [");
        for (size_t i = 0; i < layout.position_ids.size(); i++) {
            // 17 significant digits round-trips an IEEE754 double exactly, so the diff compares the
            // real values rather than a printf rounding of them.
            printf("%s[%.17g, %.17g, %.17g]",
                   i ? ", " : "",
                   layout.position_ids[i][0],
                   layout.position_ids[i][1],
                   layout.position_ids[i][2]);
        }
        printf("],\n");
        printf("   \"segments\": [");
        for (size_t i = 0; i < layout.segments.size(); i++) {
            printf("%s[%lld, %lld, \"%s\"]",
                   i ? ", " : "",
                   static_cast<long long>(layout.segments[i].start),
                   static_cast<long long>(layout.segments[i].stop),
                   segment_kind_name(layout.segments[i].kind));
        }
        printf("],\n");
        printf("   \"seq_len\": %lld\n", static_cast<long long>(layout.seq_len));
        printf("  }%s\n", last ? "" : ",");
    }

}  // namespace

int main() {
    std::vector<Case> cases = build_cases();
    printf("[\n");
    for (size_t i = 0; i < cases.size(); i++) {
        emit(cases[i], i + 1 == cases.size());
    }
    printf("]\n");
    return 0;
}
