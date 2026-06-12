// Pack a per-tensor .npy weight dir into ONE GGUF (the dev->product bridge, A2).
//   pack_gguf <npy_model_dir> <out.gguf>
// Each tensor is stored named by its key (filename minus .npy), fp32, with ggml ne =
// reversed(npy C-shape) so the GGUF-backed weight() reads it identically to npy_load+rev_ne.
// The only >4D case is ss_dec's conv3d weights [OC,IC,KD,KH,KW]: collapse to 4D
// [OC*IC,KD,KH,KW] (ggml ne [KW,KH,KD,OC*IC]) == exactly what weight_conv3d() builds, so its
// GGUF path is a plain fetch. GGUF==.npy is bit-exact (validated by gguf_validate.cpp).
// Build: ./build.sh pack_gguf
#include "ggml.h"
#include "gguf.h"
#include "../../sparse_spike/npy.hpp"
#include <cstdio>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc < 3) { printf("usage: pack_gguf <npy_model_dir> <out.gguf>\n"); return 1; }
    std::string dir = argv[1], out = argv[2];

    // enumerate .npy files
    std::vector<std::string> keys;
    for (auto& e : fs::directory_iterator(dir))
        if (e.path().extension() == ".npy") keys.push_back(e.path().stem().string());
    std::sort(keys.begin(), keys.end());
    if (keys.empty()) { printf("no .npy in %s\n", dir.c_str()); return 1; }

    // size a ggml context: sum of tensor bytes (with alignment) + per-tensor overhead
    size_t total = 0;
    for (auto& k : keys) { NpyArray a = npy_load(dir + "/" + k + ".npy"); total += (size_t)a.numel()*4 + 256; }
    size_t mem = total + keys.size() * ggml_tensor_overhead() + (1u<<20);
    ggml_context* ctx = ggml_init({mem, nullptr, false});
    if (!ctx) { printf("ggml_init failed (%.1f GB)\n", mem/1e9); return 1; }

    gguf_context* gctx = gguf_init_empty();
    for (auto& k : keys) {
        NpyArray a = npy_load(dir + "/" + k + ".npy");
        if (a.descr != "<f4") { printf("  !! %s not f32 (%s) -- skipping\n", k.c_str(), a.descr.c_str()); continue; }
        // ggml ne = reversed npy shape; collapse 5D conv3d -> 4D (merge first two dims)
        std::vector<int64_t> sh(a.shape.begin(), a.shape.end());
        std::vector<int64_t> ne4;
        if (sh.size() == 5) {                  // [d0,d1,d2,d3,d4] -> [d4,d3,d2,d0*d1]
            ne4 = { sh[4], sh[3], sh[2], sh[0]*sh[1] };
        } else if (sh.size() <= 4) {
            for (auto it = sh.rbegin(); it != sh.rend(); ++it) ne4.push_back(*it);
            if (ne4.empty()) ne4.push_back(1);
        } else { printf("  !! %s ndim=%zu unsupported\n", k.c_str(), sh.size()); return 1; }
        int nd = (int)ne4.size();
        int64_t ne[4] = {1,1,1,1}; for (int i=0;i<nd;i++) ne[i]=ne4[i];
        ggml_tensor* t = ggml_new_tensor(ctx, GGML_TYPE_F32, nd, ne);
        ggml_set_name(t, k.c_str());
        if ((size_t)ggml_nbytes(t) != (size_t)a.numel()*4) {
            printf("  !! %s byte mismatch ggml=%zu npy=%zu\n", k.c_str(), (size_t)ggml_nbytes(t), (size_t)a.numel()*4); return 1;
        }
        memcpy(t->data, a.raw.data(), (size_t)a.numel()*4);
        gguf_add_tensor(gctx, t);
    }
    if (!gguf_write_to_file(gctx, out.c_str(), false)) { printf("write failed %s\n", out.c_str()); return 1; }
    printf("packed %zu tensors -> %s (%.1f MB)\n", keys.size(), out.c_str(), total/1e6);
    gguf_free(gctx);
    ggml_free(ctx);
    return 0;
}
