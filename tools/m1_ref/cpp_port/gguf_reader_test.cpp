// Validate the dependency-free GgufReader (used by svp::WLoad) reads tensors BIT-EXACT vs .npy.
//   gguf_reader_test <npy_dir> <gguf>
// Build: g++ -O2 -std=c++17 -I../../../ggml/include -I../../../ggml/src gguf_reader_test.cpp -o gguf_reader_test
#include "gguf_reader.hpp"
#include "../../sparse_spike/npy.hpp"
#include <cstdio>
#include <cstring>
#include <string>

int main(int argc, char** argv) {
    if (argc < 3) { printf("usage: gguf_reader_test <npy_dir> <gguf>\n"); return 1; }
    std::string dir = argv[1], path = argv[2];
    GgufReader g(path);
    int bad = 0; size_t tot = 0; int n = 0;
    for (auto& kv : g.tensors) {
        const std::string& name = kv.first;
        std::vector<float> gd = g.tensor_f32(name);
        NpyArray a = npy_load(dir + "/" + name + ".npy");
        if ((int64_t)gd.size() != a.numel()) { printf("  [%s] COUNT gguf=%zu npy=%lld\n", name.c_str(), gd.size(), (long long)a.numel()); bad++; continue; }
        if (memcmp(gd.data(), a.f32(), gd.size()*4) != 0) { printf("  [%s] BYTES DIFFER\n", name.c_str()); bad++; continue; }
        tot += gd.size()*4; n++;
    }
    printf("[gguf_reader_test] %d tensors %.1f MB -> %s\n", n, tot/1e6, bad==0 ? "BIT-EXACT vs .npy" : "MISMATCH");
    return bad==0 ? 0 : 1;
}
