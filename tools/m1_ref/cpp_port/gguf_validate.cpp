// Validate a packed GGUF is BIT-EXACT vs the source per-tensor .npy dir.
//   gguf_validate <npy_dir> <gguf>
// For each tensor in the GGUF: load the matching .npy, assert byte-count + raw bytes identical.
// (5D conv3d weights are stored 4D-collapsed but the RAW BYTES are unchanged.) Build: ./build.sh gguf_validate
#include "ggml.h"
#include "gguf.h"
#include "../../sparse_spike/npy.hpp"
#include <cstdio>
#include <cstring>
#include <string>

int main(int argc, char** argv) {
    if (argc < 3) { printf("usage: gguf_validate <npy_dir> <gguf>\n"); return 1; }
    std::string dir = argv[1], path = argv[2];
    ggml_context* ctx_data = nullptr;
    gguf_init_params p{ /*no_alloc=*/false, /*ctx=*/&ctx_data };
    gguf_context* g = gguf_init_from_file(path.c_str(), p);
    if (!g) { printf("gguf_init_from_file failed: %s\n", path.c_str()); return 1; }
    int64_t n = gguf_get_n_tensors(g);
    int bad = 0; size_t tot = 0;
    for (int64_t i = 0; i < n; i++) {
        const char* name = gguf_get_tensor_name(g, i);
        ggml_tensor* t = ggml_get_tensor(ctx_data, name);
        NpyArray a = npy_load(dir + "/" + std::string(name) + ".npy");
        size_t gb = ggml_nbytes(t), nb = (size_t)a.numel() * 4;
        if (gb != nb) { printf("  [%s] BYTE COUNT gguf=%zu npy=%zu\n", name, gb, nb); bad++; continue; }
        if (memcmp(t->data, a.raw.data(), nb) != 0) {
            // find first diff
            const uint8_t* x = (const uint8_t*)t->data; const uint8_t* y = a.raw.data();
            size_t j = 0; while (j < nb && x[j] == y[j]) j++;
            printf("  [%s] BYTES DIFFER at offset %zu/%zu\n", name, j, nb); bad++; continue;
        }
        tot += nb;
    }
    printf("[gguf_validate] %lld tensors, %.1f MB  -> %s\n", (long long)n, tot/1e6,
           bad == 0 ? "BIT-EXACT vs .npy" : "MISMATCH");
    gguf_free(g); ggml_free(ctx_data);
    return bad == 0 ? 0 : 1;
}
