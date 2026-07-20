// Committed matte stage (C++; replaces the inline heredoc + the py prototype). pixal3d wants a SQUARE,
// BLACK-bg, RGB matte centered with a small margin. From an RGBA (or RGB) source:
//   crop to the alpha bbox -> premultiply onto black -> pad to a square canvas with a margin -> write RGB PNG.
// Deterministic, parameterized, reproducible. No deps beyond header-only stb (same stb the port already uses).
//   build: ./build.sh make_matte        run: ./make_matte <src.png> <out_matte.png> [margin=0.05] [alpha_thresh=8]
#define STB_IMAGE_IMPLEMENTATION
#include "../../../thirdparty/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../../thirdparty/stb_image_write.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>

int main(int argc, char** argv) {
    if (argc == 3 && !std::strcmp(argv[1], "--inspect-input")) {
        int W, H, C;
        unsigned char* d = stbi_load(argv[2], &W, &H, &C, 4);
        if (!d) { std::fprintf(stderr, "make_matte: load failed %s (%s)\n", argv[2], stbi_failure_reason()); return 1; }
        bool transparent = false, visible = false;
        for (int i=0; i<W*H; i++) {
            const unsigned char a=d[(size_t)i*4+3];
            transparent = transparent || a < 255;
            visible = visible || a > 8;
        }
        int border=0, black=0;
        for (int y=0; y<H; y++) for (int x=0; x<W; x++) if (x==0 || y==0 || x==W-1 || y==H-1) {
            border++;
            const unsigned char* p=&d[(size_t)(y*W+x)*4];
            if (std::max({p[0],p[1],p[2]}) <= 8) black++;
        }
        const char* kind = (transparent && visible) ? "rgba-cutout" : (border && black*100 >= border*95 ? "black-matte" : "opaque");
        std::printf("input_kind=%s\n", kind);
        std::printf("source_channels=%d\n", C);
        std::printf("black_border_percent=%d\n", border ? black*100/border : 0);
        stbi_image_free(d);
        return 0;
    }
    if (argc < 3) { printf("usage: make_matte <src.png> <out_matte.png> [margin=0.05] [alpha_thresh=8]\n"); return 1; }
    const char* src = argv[1];
    const char* out = argv[2];
    float margin = (argc > 3) ? (float)atof(argv[3]) : 0.05f;   // border each side = margin*subject (0.05 ~ 10% total)
    int   athr   = (argc > 4) ? atoi(argv[4]) : 8;              // alpha cutoff for "foreground"

    int W, H, C;
    unsigned char* d = stbi_load(src, &W, &H, &C, 4);           // force RGBA (alpha = matte; opaque src -> a=255)
    if (!d) { printf("make_matte: load failed %s (%s)\n", src, stbi_failure_reason()); return 1; }

    // crop to alpha bbox (fall back to full frame if there's no real alpha, e.g. opaque RGB source)
    int x0 = W, y0 = H, x1 = 0, y1 = 0; bool any = false;
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        if (d[(size_t)(y*W + x)*4 + 3] > athr) { any = true;
            if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
    }
    if (!any) { x0 = 0; y0 = 0; x1 = W - 1; y1 = H - 1; }
    int w = x1 - x0 + 1, h = y1 - y0 + 1;

    int side = (int)lroundf((float)(w > h ? w : h) * (1.0f + 2.0f * margin));
    std::vector<unsigned char> canvas((size_t)side * side * 3, 0);   // black RGB
    int ox = (side - w) / 2, oy = (side - h) / 2;
    for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
        const unsigned char* p = &d[(size_t)((y0 + y)*W + (x0 + x))*4];
        float a = p[3] / 255.0f;                                     // premultiply onto black
        unsigned char* q = &canvas[(size_t)((oy + y)*side + (ox + x))*3];
        for (int k = 0; k < 3; k++) q[k] = (unsigned char)lroundf(p[k] * a);
    }
    stbi_image_free(d);

    if (!stbi_write_png(out, side, side, 3, canvas.data(), side * 3)) {
        printf("make_matte: write failed %s\n", out); return 1; }
    printf("[matte] %s %dx%d -> alpha bbox %dx%d -> %s %dx%d (margin %.0f%% each side)\n",
           src, W, H, w, h, out, side, side, margin * 100.0f);
    return 0;
}
