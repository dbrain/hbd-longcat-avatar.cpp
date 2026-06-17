// rig_fsq_test.cpp — CPU self-test for rig_fsq.hpp (the SkinTokens skin-VAE
// FSQ.indices_to_codes port).  PURE CPU, header-only, no ggml (uses the
// dependency-free gguf_reader.hpp mmap loader for the project_out weights).
//
// Pipeline (mirrors src/model/tokenrig.py decode(), ~L536):
//   1. load the golden emitted sequence output_ids.npy ('<i8', [567])
//   2. take the tokens AFTER the tokenizer-eos (258) that fall in the VAE range
//      [tokenizer_vocab, tokenizer_vocab + vae_vocab):
//          vae_index = skin_token - tokenizer_vocab   (offset 267)
//      (tokenrig.py: `indices = skin_tokens - tokenizer.vocab_size`; vocab_size==267)
//   3. rig::Fsq::indices_to_codes(indices)  ->  [N, 512]
//   4. compare to the Python reference _r4_fsq_codes_full.npy ('<f4', [N,512])
//      produced by calling the REAL FSQ.indices_to_codes on the same indices.
//   assert max-abs diff < 1e-5  -> print OK, exit 0.
#include "rig_fsq.hpp"
#include "gguf_reader.hpp"

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>

static const int   TOKENIZER_VOCAB = 267;   // model.tokenizer.vocab_size
static const int   VAE_VOCAB       = 32768;  // model.vae.vocab_size (== 8^5)
static const int   TOK_EOS         = 258;    // tokenizer eos (the "switch" token)

// ---- Minimal .npy header parse (shared) -------------------------------------
static std::ifstream open_npy(const std::string& path, std::string& descr,
                              std::vector<size_t>& shape) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "FAIL: cannot open %s\n", path.c_str()); exit(2); }
    char magic[6]; f.read(magic, 6);
    if (memcmp(magic, "\x93NUMPY", 6) != 0) { fprintf(stderr, "FAIL: bad npy magic %s\n", path.c_str()); exit(2); }
    unsigned char ver[2]; f.read((char*)ver, 2);
    uint16_t hlen; f.read((char*)&hlen, 2);
    std::string header(hlen, '\0'); f.read(&header[0], hlen);
    // descr
    size_t dp = header.find("'descr'");
    size_t q1 = header.find('\'', dp + 7);
    size_t q2 = header.find('\'', q1 + 1);
    descr = header.substr(q1 + 1, q2 - q1 - 1);
    // shape
    size_t sp = header.find("'shape'");
    size_t lp = header.find('(', sp), rp = header.find(')', lp);
    std::string shp = header.substr(lp + 1, rp - lp - 1);
    shape.clear();
    size_t pos = 0;
    while (pos < shp.size()) {
        while (pos < shp.size() && (shp[pos] == ' ' || shp[pos] == ',')) pos++;
        if (pos >= shp.size()) break;
        size_t end = pos;
        while (end < shp.size() && shp[end] >= '0' && shp[end] <= '9') end++;
        if (end > pos) shape.push_back((size_t)std::stoul(shp.substr(pos, end - pos)));
        pos = end + 1;
    }
    return f;
}

static std::vector<int64_t> load_npy_i8(const std::string& path, std::vector<size_t>& shape) {
    std::string descr;
    std::ifstream f = open_npy(path, descr, shape);
    if (descr.find("i8") == std::string::npos) { fprintf(stderr, "FAIL: %s not <i8 (%s)\n", path.c_str(), descr.c_str()); exit(2); }
    size_t n = 1; for (size_t d : shape) n *= d;
    std::vector<int64_t> data(n);
    f.read((char*)data.data(), (std::streamsize)(n * sizeof(int64_t)));
    if (!f) { fprintf(stderr, "FAIL: short read %s\n", path.c_str()); exit(2); }
    return data;
}

static std::vector<float> load_npy_f4(const std::string& path, std::vector<size_t>& shape) {
    std::string descr;
    std::ifstream f = open_npy(path, descr, shape);
    if (descr.find("f4") == std::string::npos) { fprintf(stderr, "FAIL: %s not <f4 (%s)\n", path.c_str(), descr.c_str()); exit(2); }
    size_t n = 1; for (size_t d : shape) n *= d;
    std::vector<float> data(n);
    f.read((char*)data.data(), (std::streamsize)(n * sizeof(float)));
    if (!f) { fprintf(stderr, "FAIL: short read %s\n", path.c_str()); exit(2); }
    return data;
}

int main(int argc, char** argv) {
    const char* e2e  = (argc > 1) ? argv[1] : "/tmp/skintokens_e2e";
    const char* gguf = (argc > 2) ? argv[2]
        : "/mnt/hdd/3d/avatar-shootout/_weights/skin_vae_gguf/skin_vae.gguf";

    // 1. golden emitted tokens
    std::string ids_path = std::string(e2e) + "/output_ids.npy";
    std::vector<size_t> sh;
    std::vector<int64_t> out = load_npy_i8(ids_path, sh);
    printf("loaded %s: %zu tokens\n", ids_path.c_str(), out.size());

    // 2. skin tokens = post-(tokenizer-eos) tokens in the VAE range -> vae indices
    size_t eos_pos = out.size();
    for (size_t i = 0; i < out.size(); ++i) if (out[i] == TOK_EOS) { eos_pos = i; break; }
    if (eos_pos == out.size()) { fprintf(stderr, "FAIL: tokenizer-eos (%d) not found\n", TOK_EOS); return 2; }
    std::vector<int64_t> indices;
    for (size_t i = eos_pos + 1; i < out.size(); ++i) {
        int64_t tk = out[i];
        if (tk >= TOKENIZER_VOCAB && tk < (int64_t)TOKENIZER_VOCAB + VAE_VOCAB)
            indices.push_back(tk - TOKENIZER_VOCAB);   // model-vocab id -> vae index
    }
    if (indices.empty()) { fprintf(stderr, "FAIL: no skin tokens\n"); return 2; }
    int64_t imin = indices[0], imax = indices[0];
    for (int64_t v : indices) { if (v < imin) imin = v; if (v > imax) imax = v; }
    printf("skin indices: N=%zu  range=[%lld..%lld]  (offset=tokenizer_vocab=%d)\n",
           indices.size(), (long long)imin, (long long)imax, TOKENIZER_VOCAB);
    if (imin < 0 || imax >= VAE_VOCAB) { fprintf(stderr, "FAIL: index out of [0,%d)\n", VAE_VOCAB); return 2; }

    // 3. load FSQ project_out from the GGUF + run the port
    rig::Fsq fsq;
    fsq.init();
    printf("FSQ: levels=[8,8,8,8,8] codebook_dim=%d codebook_size=%lld basis=[1,8,64,512,4096]\n",
           fsq.codebook_dim, (long long)fsq.codebook_size);
    {
        GgufReader gr(gguf);
        const char* WN = "model.FSQ.project_out.weight";  // ne [5,512] -> torch [512,5]
        const char* BN = "model.FSQ.project_out.bias";    // [512]
        if (!gr.has(WN) || !gr.has(BN)) { fprintf(stderr, "FAIL: %s missing project_out\n", gguf); return 2; }
        fsq.proj_out_w = gr.tensor_f32(WN);   // row-major [out_dim*codebook_dim]
        fsq.proj_out_b = gr.tensor_f32(BN);
        if ((int)fsq.proj_out_b.size() != fsq.out_dim ||
            (int)fsq.proj_out_w.size() != fsq.out_dim * fsq.codebook_dim) {
            fprintf(stderr, "FAIL: project_out size mismatch w=%zu b=%zu\n",
                    fsq.proj_out_w.size(), fsq.proj_out_b.size()); return 2;
        }
    }
    std::vector<float> codes = fsq.indices_to_codes(indices, /*project=*/true);  // [N*512]

    // 4. compare to Python reference (full, post-project_out)
    std::string ref_path = std::string(e2e) + "/_r4_fsq_codes_full.npy";
    std::vector<size_t> rsh;
    std::vector<float> ref = load_npy_f4(ref_path, rsh);
    if (codes.size() != ref.size()) {
        fprintf(stderr, "FAIL: size mismatch ours=%zu ref=%zu (ref shape %zux%zu)\n",
                codes.size(), ref.size(), rsh.size()>0?rsh[0]:0, rsh.size()>1?rsh[1]:0);
        return 2;
    }
    double maxabs = 0.0; size_t at = 0;
    for (size_t i = 0; i < codes.size(); ++i) {
        double d = std::fabs((double)codes[i] - (double)ref[i]);
        if (d > maxabs) { maxabs = d; at = i; }
    }
    printf("compared %zu values vs %s\n", codes.size(), ref_path.c_str());
    printf("max-abs diff = %.3e  (at idx %zu: ours=%.6f ref=%.6f)\n",
           maxabs, at, codes[at], ref[at]);

    // also report the raw (pre-project) diff for the follow-up decoder port
    {
        std::string raw_path = std::string(e2e) + "/_r4_fsq_codes_raw.npy";
        std::ifstream tf(raw_path, std::ios::binary);
        if (tf) {
            std::vector<size_t> qsh;
            std::vector<float> rawref = load_npy_f4(raw_path, qsh);
            std::vector<float> rawours = fsq.indices_to_codes(indices, /*project=*/false);
            double m = 0.0;
            for (size_t i = 0; i < rawours.size() && i < rawref.size(); ++i)
                m = std::max(m, (double)std::fabs((double)rawours[i] - (double)rawref[i]));
            printf("raw mixed-radix dequant max-abs diff = %.3e (pre-project_out)\n", m);
        }
    }

    const double TOL = 1e-5;
    if (maxabs >= TOL) { fprintf(stderr, "FAIL: max-abs diff %.3e >= %.0e\n", maxabs, TOL); return 1; }
    printf("OK (max-abs diff %.3e < %.0e)\n", maxabs, TOL);
    return 0;
}
