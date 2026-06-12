// Pack a per-tensor .npy weight dir into ONE GGUF (the dev->product bridge, A2).
//   pack_gguf <npy_model_dir> <out.gguf> [--type q8_0|q4_k|q6_k]
// Each tensor is stored named by its key (filename minus .npy), with ggml ne =
// reversed(npy C-shape) so the GGUF-backed weight() reads it identically to npy_load+rev_ne.
// The only >4D case is ss_dec's conv3d weights [OC,IC,KD,KH,KW]: collapse to 4D
// [OC*IC,KD,KH,KW] (ggml ne [KW,KH,KD,OC*IC]) == exactly what weight_conv3d() builds, so its
// GGUF path is a plain fetch. GGUF==.npy is bit-exact for f32 (gguf_validate.cpp).
//
// --type (Phase C #2, DiT GGUFs only): quantize the 2D matmul weights (names ending in
// ".weight" with ndim==2 and contraction dim a multiple of the block size) to the given
// Q-type; keep everything else (1D norms/biases, .gamma scales, conv kernels) f32. ggml's
// gguf_fetch + ggml_mul_mat read quant weights natively (MMQ). Do NOT quantize the sparse-VAE
// decoder GGUFs (shape_dec/tex_dec) — the standalone svpg reader expects f32.
#include "ggml.h"
#include "gguf.h"
#include "../../sparse_spike/npy.hpp"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

static ggml_type parse_qtype(const std::string& s) {
    if (s == "f16")  return GGML_TYPE_F16;
    if (s == "q8_0") return GGML_TYPE_Q8_0;
    if (s == "q4_0") return GGML_TYPE_Q4_0;
    if (s == "q4_k") return GGML_TYPE_Q4_K;
    if (s == "q5_k") return GGML_TYPE_Q5_K;
    if (s == "q6_k") return GGML_TYPE_Q6_K;
    return GGML_TYPE_F32;
}

int main(int argc, char** argv) {
    if (argc < 3) { printf("usage: pack_gguf <npy_model_dir> <out.gguf> [--type q8_0|q4_k|q6_k]\n"); return 1; }
    std::string dir = argv[1], out = argv[2];
    ggml_type qtype = GGML_TYPE_F32;
    std::vector<std::string> keepf16;   // name substrings to keep at F16 (mixed precision study)
    for (int i = 3; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--type" && i + 1 < argc) qtype = parse_qtype(argv[++i]);
        else if (a == "--keep" && i + 1 < argc) keepf16.push_back(argv[++i]);  // -> F16 (near-lossless)
    }
    int qblk = (qtype == GGML_TYPE_F32) ? 1 : (int)ggml_blck_size(qtype);
    printf("pack_gguf: type=%s (block=%d) for 2D matmul .weight tensors",
           qtype==GGML_TYPE_F32 ? "f32" : ggml_type_name(qtype), qblk);
    for (auto& s : keepf16) printf(" [keep-f16:%s]", s.c_str());
    printf("\n");

    // enumerate .npy files
    std::vector<std::string> keys;
    for (auto& e : fs::directory_iterator(dir))
        if (e.path().extension() == ".npy") keys.push_back(e.path().stem().string());
    std::sort(keys.begin(), keys.end());
    if (keys.empty()) { printf("no .npy in %s\n", dir.c_str()); return 1; }

    // a matmul weight = name ends in ".weight", ndim==2, contraction dim (ne0) % block == 0.
    auto ends_with = [](const std::string& s, const std::string& suf){
        return s.size() >= suf.size() && s.compare(s.size()-suf.size(), suf.size(), suf) == 0; };

    // size a ggml context: f32 tensors keep full bytes; quant tensors are smaller, so f32 sum
    // is a safe upper bound. + per-tensor overhead.
    size_t total = 0;
    for (auto& k : keys) { NpyArray a = npy_load(dir + "/" + k + ".npy"); total += (size_t)a.numel()*4 + 256; }
    size_t mem = total + keys.size() * ggml_tensor_overhead() + (1u<<20);
    ggml_context* ctx = ggml_init({mem, nullptr, false});
    if (!ctx) { printf("ggml_init failed (%.1f GB)\n", mem/1e9); return 1; }

    gguf_context* gctx = gguf_init_empty();
    size_t out_bytes = 0; int n_quant = 0;
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

        // per-tensor target type: only 2D matmul weights are eligible. --keep substrings -> F16
        // (near-lossless); otherwise the main qtype if block-aligned, else F16 fallback.
        bool is_matmul = (qtype != GGML_TYPE_F32) && ends_with(k, ".weight") && nd == 2;
        ggml_type tt = GGML_TYPE_F32;
        if (is_matmul) {
            bool keep = false;
            for (auto& s : keepf16) if (k.find(s) != std::string::npos) keep = true;
            if (keep || qtype == GGML_TYPE_F16) tt = GGML_TYPE_F16;
            else if (ne[0] % qblk == 0) tt = qtype;
            else tt = GGML_TYPE_F16;   // block-misaligned (e.g. ss_flow input in=8) -> F16
        }
        ggml_tensor* t = ggml_new_tensor(ctx, tt, nd, ne);
        ggml_set_name(t, k.c_str());
        if (tt == GGML_TYPE_F16) {
            ggml_fp32_to_fp16_row(a.f32(), (ggml_fp16_t*)t->data, (int64_t)a.numel());
            n_quant++;
        } else if (tt != GGML_TYPE_F32) {
            ggml_quantize_chunk(tt, a.f32(), t->data, 0, ne[1], ne[0], nullptr);  // 2D [in,out]: nrows=ne1, n_per_row=ne0
            n_quant++;
        } else {
            if ((size_t)ggml_nbytes(t) != (size_t)a.numel()*4) {
                printf("  !! %s byte mismatch ggml=%zu npy=%zu\n", k.c_str(), (size_t)ggml_nbytes(t), (size_t)a.numel()*4); return 1;
            }
            memcpy(t->data, a.raw.data(), (size_t)a.numel()*4);
        }
        out_bytes += ggml_nbytes(t);
        gguf_add_tensor(gctx, t);
    }
    if (!gguf_write_to_file(gctx, out.c_str(), false)) { printf("write failed %s\n", out.c_str()); return 1; }
    printf("packed %zu tensors (%d quantized) -> %s (%.1f MB on disk)\n",
           keys.size(), n_quant, out.c_str(), out_bytes/1e6);
    gguf_free(gctx);
    ggml_free(ctx);
    return 0;
}
