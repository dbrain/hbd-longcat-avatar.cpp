// ktx2_test — validate the vendored basis_universal KTX2 encoder (ktx2_encode.hpp) builds + emits
// valid KTX2. Generates a 512x512 RGBA gradient, encodes UASTC (hero) + ETC1S (small), writes .ktx2.
//   ./build.sh ktx2_test && ./ktx2_test
#include "ktx2_encode.hpp"
#include <cstdio>
#include <vector>

int main() {
    const int W = 512, H = 512;
    std::vector<uint8_t> rgba(W * H * 4);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            uint8_t* px = &rgba[(y * W + x) * 4];
            px[0] = (uint8_t)(x * 255 / (W - 1));
            px[1] = (uint8_t)(y * 255 / (H - 1));
            px[2] = (uint8_t)(((x ^ y) & 0xFF));
            px[3] = 255;
        }

    auto write = [](const char* path, const std::vector<uint8_t>& b) {
        if (b.empty()) { printf("  FAILED (empty) %s\n", path); return false; }
        // KTX2 magic: \xABKTX 20\xBB\r\n\x1A\n
        static const uint8_t MAGIC[12] = {0xAB,0x4B,0x54,0x58,0x20,0x32,0x30,0xBB,0x0D,0x0A,0x1A,0x0A};
        bool ok = b.size() > 12;
        for (int i = 0; i < 12 && ok; i++) ok = (b[i] == MAGIC[i]);
        FILE* f = fopen(path, "wb"); if (f) { fwrite(b.data(), 1, b.size(), f); fclose(f); }
        printf("  %-22s %8zu bytes  magic=%s\n", path, b.size(), ok ? "OK" : "BAD");
        return ok;
    };

    printf("[ktx2_test] encoding 512x512 gradient...\n");
    bool ok = true;
    ok &= write("ktx2_test_uastc.ktx2", ktx2enc::encode(rgba.data(), W, H, /*uastc*/true,  0,   /*srgb*/true, /*comps*/4, 8));
    ok &= write("ktx2_test_etc1s.ktx2", ktx2enc::encode(rgba.data(), W, H, /*uastc*/false, 192, /*srgb*/true, /*comps*/4, 8));
    printf("[ktx2_test] %s\n", ok ? "ALL VALID" : "FAILED");
    return ok ? 0 : 1;
}
