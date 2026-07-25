// Committed matte stage (C++; replaces the inline heredoc + the py prototype). pixal3d wants a SQUARE,
// BLACK-bg, RGB matte centered with a small margin. From an RGBA (or RGB) source:
//   crop to the alpha bbox -> premultiply onto black -> pad to a square canvas with a margin -> write RGB PNG.
// Opaque black mattes use a near-black border flood for the crop.  That keeps black hair, pupils,
// boots and clothing enclosed by the silhouette while preventing a large black canvas from shrinking
// the subject presented to Pixal's image conditioner.
// Deterministic, parameterized, reproducible. No deps beyond header-only stb (same stb the port already uses).
//   build: ./build.sh make_matte        run: ./make_matte [--rgba] <src.png> <out_matte.png> [margin=0.05] [alpha_thresh=8]
#define STB_IMAGE_IMPLEMENTATION
#include "../../../thirdparty/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../../thirdparty/stb_image_write.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <queue>
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
        // TWO rings, not one. The outermost 1px ring alone is satisfied by a PICTURE FRAME: a
        // studio render with a WHITE background and a thin black border passes it and gets called
        // a black-matte, so the pipeline consumes the entire background as subject. That is exactly
        // what happened to toy1_matte.png (outer ring 0, but inner ring luma 231, 63% near-white)
        // and it produced a mesh with the framing baked in. Sample a ring set 6% in as well: a real
        // black-matte is black on BOTH, a frame is black only on the outer one.
        auto ring_black = [&](int inset, int& count) {
            int black = 0; count = 0;
            const int x0 = inset, x1 = W-1-inset, y0 = inset, y1 = H-1-inset;
            if (x0 >= x1 || y0 >= y1) return 0;
            for (int y=y0; y<=y1; y++) for (int x=x0; x<=x1; x++) {
                if (x!=x0 && x!=x1 && y!=y0 && y!=y1) continue;
                count++;
                const unsigned char* p=&d[(size_t)(y*W+x)*4];
                if (std::max({p[0],p[1],p[2]}) <= 8) black++;
            }
            return black;
        };
        int border=0, inner_count=0;
        const int black = ring_black(0, border);
        const int inset = std::max(2, (int)(std::min(W,H) * 0.06f));
        const int inner_black = ring_black(inset, inner_count);
        const int outer_pct = border ? black*100/border : 0;
        const int inner_pct = inner_count ? inner_black*100/inner_count : 0;

        // Inner-ring threshold is deliberately LOOSE (60, not 95): a standing subject's arms and
        // feet legitimately cross a ring 6% in, so a genuine matte measures 90-96 there, not 100.
        // Calibrated on real inputs — genuine mattes: char1 96, gilly 93, soldier 90, toy2 90;
        // the framed white-background render: toy1 0. The question is "is the interior background
        // dark at all", not "is it pristine".
        const int INNER_MIN = 60;
        const char* kind;
        if (transparent && visible)            kind = "rgba-cutout";
        else if (outer_pct >= 95 && inner_pct >= INNER_MIN) kind = "black-matte";
        // Black frame, non-black interior: NOT a matte. Must go through the neural matte lane, so
        // give it a distinct name rather than letting it masquerade as a prepared frame.
        else if (outer_pct >= 95)              kind = "framed-opaque";
        else                                   kind = "opaque";
        std::printf("input_kind=%s\n", kind);
        std::printf("source_channels=%d\n", C);
        std::printf("black_border_percent=%d\n", outer_pct);
        std::printf("black_inner_ring_percent=%d\n", inner_pct);
        stbi_image_free(d);
        return 0;
    }
    int arg = 1; const bool keep_rgba = arg < argc && !std::strcmp(argv[arg], "--rgba"); if (keep_rgba) arg++;
    if (argc - arg < 2) { printf("usage: make_matte [--rgba] <src.png> <out_matte.png> [margin=0.05] [alpha_thresh=8]\n"); return 1; }
    const char* src = argv[arg++];
    const char* out = argv[arg++];
    float margin = (argc > arg) ? (float)atof(argv[arg]) : 0.05f;   // border each side = margin*subject (0.05 ~ 10% total)
    int   athr   = (argc > arg+1) ? atoi(argv[arg+1]) : 8;          // alpha cutoff for "foreground"

    int W, H, C;
    unsigned char* d = stbi_load(src, &W, &H, &C, 4);           // force RGBA (alpha = matte; opaque src -> a=255)
    if (!d) { printf("make_matte: load failed %s (%s)\n", src, stbi_failure_reason()); return 1; }

    // Crop to alpha bbox.  An opaque black matte has no useful alpha, so find only the dark
    // component connected to the border.  We retain original RGB inside the crop; this is framing,
    // never a luminance-derived alpha that could erase dark details.
    int x0 = W, y0 = H, x1 = 0, y1 = 0; bool any = false;
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        if (d[(size_t)(y*W + x)*4 + 3] > athr) { any = true;
            if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
    }
    bool opaque=true;
    for (int i=0;i<W*H;i++) if (d[(size_t)i*4+3] != 255) { opaque=false; break; }
    int border=0, black=0;
    for (int y=0;y<H;y++) for (int x=0;x<W;x++) if (x==0||y==0||x==W-1||y==H-1) {
        border++;
        const unsigned char* p=&d[(size_t)(y*W+x)*4];
        if (std::max({p[0],p[1],p[2]}) <= 8) black++;
    }
    const bool black_matte=opaque && border && black*100 >= border*95;
    if (black_matte) {
        std::vector<unsigned char> outside((size_t)W*H,0); std::queue<int> q;
        auto is_bg=[&](int x,int y){ const unsigned char* p=&d[(size_t)(y*W+x)*4]; return std::max({p[0],p[1],p[2]}) <= 8; };
        auto push=[&](int x,int y){ const size_t i=(size_t)y*W+x; if (!outside[i] && is_bg(x,y)) { outside[i]=1; q.push(y*W+x); } };
        for(int x=0;x<W;x++){ push(x,0); push(x,H-1); }
        for(int y=1;y+1<H;y++){ push(0,y); push(W-1,y); }
        while(!q.empty()) { const int p=q.front(); q.pop(); const int x=p%W,y=p/W;
            if(x) push(x-1,y); if(x+1<W) push(x+1,y); if(y) push(x,y-1); if(y+1<H) push(x,y+1); }
        x0=W; y0=H; x1=-1; y1=-1;
        for(int y=0;y<H;y++) for(int x=0;x<W;x++) if(!outside[(size_t)y*W+x]) {
            x0=std::min(x0,x); y0=std::min(y0,y); x1=std::max(x1,x); y1=std::max(y1,y); }
        any=x1>=x0;
    }
    if (!any) { x0 = 0; y0 = 0; x1 = W - 1; y1 = H - 1; }
    int w = x1 - x0 + 1, h = y1 - y0 + 1;

    int side = (int)lroundf((float)(w > h ? w : h) * (1.0f + 2.0f * margin));
    const int out_c = keep_rgba ? 4 : 3;
    std::vector<unsigned char> canvas((size_t)side * side * out_c, 0); // transparent/black canvas
    int ox = (side - w) / 2, oy = (side - h) / 2;
    for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
        const unsigned char* p = &d[(size_t)((y0 + y)*W + (x0 + x))*4];
        float a = black_matte ? 1.0f : p[3] / 255.0f;                 // opaque matte already has black bg
        unsigned char* q = &canvas[(size_t)((oy + y)*side + (ox + x))*out_c];
        // RGBA preserves the unpremultiplied cutout colour; image_io premultiplies it for every
        // model/projection consumer.  RGB remains the legacy black-composited matte contract.
        for (int k = 0; k < 3; k++) q[k] = keep_rgba ? p[k] : (unsigned char)lroundf(p[k] * a);
        if (keep_rgba) q[3] = (unsigned char)lroundf(a * 255.0f);
    }
    stbi_image_free(d);

    if (!stbi_write_png(out, side, side, out_c, canvas.data(), side * out_c)) {
        printf("make_matte: write failed %s\n", out); return 1; }
    printf("[matte] %s %dx%d -> %s bbox %dx%d -> %s %dx%d (margin %.0f%% each side)\n",
           src, W, H, black_matte?"border-flood":"alpha", w, h, out, side, side, margin * 100.0f);
    return 0;
}
