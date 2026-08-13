// meshopt_verify — decode every EXT_meshopt_compression bufferView in a GLB and report OK/FAIL.
// Catches bad byteOffset/byteLength/version that glTF-Validator can't (it treats meshopt as unsupported).
//   ./build.sh meshopt_verify && ./meshopt_verify file.glb
#include "../../../thirdparty/json.hpp"
#include "../../../thirdparty/meshoptimizer/meshoptimizer.h"
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
using json = nlohmann::json;

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: meshopt_verify file.glb\n"); return 1; }
    FILE* f = fopen(argv[1], "rb"); if (!f) { printf("open failed\n"); return 1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> d(n); size_t rd = fread(d.data(), 1, n, f); (void)rd; fclose(f);
    uint32_t jlen; memcpy(&jlen, &d[12], 4);
    json j = json::parse(d.begin() + 20, d.begin() + 20 + jlen);
    size_t binstart = 20 + ((jlen + 3) & ~3u) + 8;
    printf("file=%s size=%ld binstart=%zu\n", argv[1], n, binstart);

    bool ok = true;
    auto& bvs = j["bufferViews"];
    for (size_t i = 0; i < bvs.size(); i++) {
        if (!bvs[i].contains("extensions")) continue;
        auto& e = bvs[i]["extensions"]["EXT_meshopt_compression"];
        std::string mode = e["mode"];
        uint32_t bo = e["byteOffset"], bl = e["byteLength"], cnt = e["count"], stride = e["byteStride"];
        if (binstart + bo + bl > (size_t)n) { printf("  bv[%zu] %s OFFSET OUT OF RANGE (%u+%u > bin)\n", i, mode.c_str(), bo, bl); ok=false; continue; }
        const uint8_t* src = d.data() + binstart + bo;
        int rc; std::vector<uint8_t> out;
        if (mode == "TRIANGLES") { out.resize((size_t)cnt * stride); rc = meshopt_decodeIndexBuffer(out.data(), cnt, stride, src, bl); }
        else { out.resize((size_t)cnt * stride); rc = meshopt_decodeVertexBuffer(out.data(), cnt, stride, src, bl); }
        printf("  bv[%zu] %-10s off=%u len=%u count=%u stride=%u hdr=0x%02x -> %s\n",
               i, mode.c_str(), bo, bl, cnt, stride, src[0], rc==0?"DECODE OK":"DECODE FAIL");
        if (rc != 0) ok = false;
    }
    printf("%s\n", ok ? "ALL MESHOPT STREAMS DECODE" : "FAILED");
    return ok ? 0 : 1;
}
