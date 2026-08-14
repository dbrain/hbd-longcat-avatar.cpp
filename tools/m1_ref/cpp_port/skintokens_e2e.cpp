// skintokens_e2e.cpp — full SkinTokens auto-rigging end-to-end driver (R1 -> R3 -> R5 -> R4 -> R6).
// Mirrors qwen3_r2_test / rig_skin_decoder_test harness setup. CUDA build links ggml-cuda.
//
//   build: ./build.sh skintokens_e2e cuda      (or cpu)
//   run (banked mesh_cond, the R3-isolation default):
//     PIXAL3D_GGUF_DIR=/mnt/hdd/3d/avatar-shootout/_weights/skin_vae_gguf
//       ./skintokens_e2e [e2e] [qwen3_w] [out.glb] [cuda] [r1]
//
// STAGES:
//   R1 (mesh encoder)  : vecset encoder(verts,normals) -> mesh_cond [n_cond,hidden].   (opt: `r1` flag;
//                        DEFAULT loads the banked learned_mesh_cond.npy to isolate R3 from R1's
//                        nondeterministic FPS query sampling — see vecset_encode.cpp.)
//   R3 (AR generate)   : rig_generate(qwen3, mesh_cond, grammar) -> tokens (bos..model_eos).
//   R5 (detok)         : detok::detokenize(tokens) -> joints[J,3] + parents[J].
//   R4 (skin decode)   : tokens -> FSQ codes + cond_encoder(verts,normals) [banked cond_latents]
//                        -> per-joint decoder -> skin_pred[N,J].
//   R6 (GLB)           : glb::write_rigged_glb(verts, faces, joints, parents, skin_pred, normals).
//
// VALIDATION PRINTS (the real run happens on the GPU main loop): token-match vs output_ids.npy,
// joints max-abs vs detok_joints.npy, skin_pred max-abs vs r4_oracle_skin_pred_fp32.npy (+bf16 golden).
#include "qwen3_forward.hpp"
#include "rig_generate.hpp"
#include "rig_beam_generate.hpp"
#include "rig_beam_generate_batched.hpp"
#include "rig_sample.hpp"
#include "rig_grammar.hpp"
#include "rig_bone_names.hpp"
#include "rig_structural_select.hpp"
#include "detok_r5.hpp"
#include "rig_skin_decoder.hpp"
#include "rig_fsq.hpp"
#include "vecset_encoder.hpp"
#include "mesh_sample.hpp"   // rig::fps / rig::gather — cond-query selection for R4 cond=real
#include "glb_rigged.hpp"
#include "gguf_reader.hpp"
#include "../../sparse_spike/npy.hpp"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <fstream>
#include <chrono>
#include <filesystem>

#ifdef M1_USE_CUDA
extern "C" void qwen_cublaslt_hook_enable();
#endif

// Peak-VRAM sampling via the CUDA runtime — only available in the CUDA build (CPU build does NOT
// link cudart). Guarded by M1_USE_CUDA (set by build.sh's cuda branch). Falls back to a no-op note.
#ifdef M1_USE_CUDA
#include <cuda_runtime.h>
#endif

// Sample USED VRAM (total-free) in MiB, or -1.0 if unavailable.
// cudaMemGetInfo's `free` reflects ALL device allocations (incl. the ggml-cuda weight buffers),
// so total-free is the true device high-watermark at the instant of the call. The catch: it must
// be sampled WHILE the allocations are live — the per-stage M1Harness objects free their weight
// buffers when they go out of scope, so samples must happen INSIDE the harness scope, after the
// weights have been uploaded (i.e. after the first/last compute), not after the closing brace.
static double e2e_vram_used_mib() {
#ifdef M1_USE_CUDA
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess && total_b > 0)
        return (double)(total_b - free_b) / (1024.0 * 1024.0);
#endif
    return -1.0;
}

// Total device memory in MiB, or -1.0 if unavailable.
static double e2e_vram_total_mib() {
#ifdef M1_USE_CUDA
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess && total_b > 0)
        return (double)total_b / (1024.0 * 1024.0);
#endif
    return -1.0;
}

static const int TOKENIZER_VOCAB = 267;
static const int VAE_VOCAB       = 32768;
static const int TOK_EOS         = 258;

static bool file_exists(const std::string& p) { std::ifstream f(p, std::ios::binary); return (bool)f; }

// ---- host Fourier embed (FourierEmbedder, exact) — mirrors vecset_encode.cpp. logspace freqs
//      2^[0..F), include_pi=FALSE, include_input=TRUE. out per point = [xyz(3), sin(3F), cos(3F)]
//      then concat the 3 normal feats -> in_dim. pts/feats row-major [N,3]. Returns [in_dim, N]. ----
static std::vector<float> e2e_fourier_embed(const float* pts, const float* feats, int64_t N,
                                            const VecsetCfg& cfg) {
    const int F = cfg.num_freqs;
    std::vector<float> freq(F);
    for (int k = 0; k < F; ++k) {
        float f = std::pow(2.0f, (float)k);          // logspace=True
        if (cfg.include_pi) f *= (float)M_PI;        // include_pi=False here
        freq[k] = f;
    }
    const int in_dim = cfg.in_dim();                 // 3 + 6F + point_feats
    std::vector<float> out((size_t)N * in_dim);
    for (int64_t i = 0; i < N; ++i) {
        const float* p = pts + i * 3;
        float* o = out.data() + i * in_dim;
        int w = 0;
        if (cfg.include_input) for (int c = 0; c < 3; ++c) o[w++] = p[c];                         // xyz
        for (int c = 0; c < 3; ++c) for (int k = 0; k < F; ++k) o[w++] = std::sin(p[c] * freq[k]); // sin
        for (int c = 0; c < 3; ++c) for (int k = 0; k < F; ++k) o[w++] = std::cos(p[c] * freq[k]); // cos
        if (feats) for (int c = 0; c < cfg.point_feats; ++c) o[w++] = feats[i * 3 + c];           // normals
    }
    return out;
}

// cosine similarity + max-abs of a host buffer vs an npy (both flattened, same nelem expected).
static void e2e_cos_vs_npy(const std::vector<float>& got, const std::string& path, const char* tag) {
    if (!file_exists(path)) { printf("[STAGE R1] (no %s to compare against)\n", path.c_str()); return; }
    NpyArray a = npy_load(path);
    if (a.numel() != (int64_t)got.size()) {
        printf("[STAGE R1] %s nelem mismatch got=%zu ref=%lld\n", tag, got.size(), (long long)a.numel());
        return;
    }
    const float* r = a.f32();
    double dot = 0, na = 0, nb = 0, maxabs = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        double x = got[i], y = r[i];
        dot += x * y; na += x * x; nb += y * y;
        maxabs = std::max(maxabs, std::fabs(x - y));
    }
    double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
    printf("[STAGE R1] computed mesh_cond vs %s: cosine=%.6f max-abs=%.3e (n=%zu)\n",
           tag, cos, maxabs, got.size());
}

// ---- minimal .npy writer (little-endian, descr from caller). 1-D or 2-D shape. ----
// Dumps the generated tokens + detok joints/parents so the GPU main loop can diff/visualize.
static void npy_write(const std::string& path, const void* data, size_t elem_bytes,
                      const std::string& descr, const std::vector<int64_t>& shape) {
    std::string sh = "(";
    for (size_t i = 0; i < shape.size(); ++i) { sh += std::to_string(shape[i]); sh += ","; }
    sh += ")";
    std::string hdr = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " + sh + ", }";
    size_t base = 10 + hdr.size() + 1;             // magic(6)+ver(2)+len(2) + header + '\n'
    size_t pad = (64 - (base % 64)) % 64;
    hdr.append(pad, ' '); hdr += '\n';
    uint16_t hlen = (uint16_t)hdr.size();
    int64_t numel = 1; for (auto d : shape) numel *= d;
    std::ofstream f(path, std::ios::binary);
    f.write("\x93NUMPY", 6);
    char ver[2] = { 1, 0 }; f.write(ver, 2);
    f.write(reinterpret_cast<const char*>(&hlen), 2);
    f.write(hdr.data(), hdr.size());
    f.write(reinterpret_cast<const char*>(data), (size_t)numel * elem_bytes);
}

int main(int argc, char** argv) {
#ifdef M1_USE_CUDA
    qwen_cublaslt_hook_enable();
#endif
    // R3/R4/R5 dump arrays are the explicit hand-off to combine_rig_tex_main.
    // A fresh host may not have the historical test directory, so create it
    // before writing rather than reporting a successful dump of a dropped file.
    std::filesystem::create_directories("/tmp/skintokens_e2e");
    std::string e2e = "/tmp/skintokens_e2e";
    std::string qwen3_w = "/tmp/qwen3_w";
    std::string out_glb = "/tmp/skintokens_e2e_out.glb";
    bool use_cuda = false, do_r1 = false;
    // R1 mode: "banked" (default-fallback, loads learned_mesh_cond.npy) | "real" (run the vecset
    // encoder + output_proj on the input verts/normals). r1w = the R1 encoder npy weight dir.
    std::string r1_mode = "banked";
    std::string r1w = "/tmp/vecset_r0";
    // R4 cond_latents mode: "banked" (default, loads cond_latents.npy — only valid for the giraffe
    // golden) | "real" (COMPUTE cond_latents via the cond-encoder from THIS mesh's verts/normals).
    // "auto" (default) follows r1: real when r1=real (rigging a new mesh), banked otherwise.
    std::string cond_mode = "auto";
    // R3 decode mode: "sample" | "beam" (HF official) | "greedy".  Default = beam (Python = num_beams=10).
    std::string r3_mode = "beam";
    int num_beams = 10;            // the proven HF beam config (Python official).
    // HF OFFICIAL sampling params (SkinTokens demo.py / spec.py: do_sample=True, temp=1.0, top_k=5,
    // top_p=0.95, repetition_penalty=2.0, num_beams=10, max_length=2048). These are the DEFAULTS now —
    // the model is bf16-native (spec.py torch_dtype=bfloat16); fp32 is an upcast that under-articulates.
    uint64_t seed       = 0;
    float    temp       = 1.0f;
    float    reppen     = 2.0f;
    int      topk       = 5;
    float    topp       = 0.95f;
    int      maxnew     = -1;     // -1 => per-mode default (1534 = max_length 2048 - 514-token prefix
                                  // [512 mesh_cond + 2 start], matching Python exactly); `maxnew=` overrides.
    ggml_type prec      = GGML_TYPE_BF16;  // AR compute precision. bf16 = model's NATIVE regime (matches
                                           // Python's torch_dtype=bfloat16). prec=fp32 upcasts (shorter rigs).
    bool do_sample      = true;   // `nosample` => deterministic beam (no warpers/RNG; HF do_sample=False)
    // Opt-in profile gates. Humanoid selection requires the 22-bone Mixamo
    // contract; generic selection validates only a well-formed skeleton tree
    // and deliberately does not invent creature semantics from anonymous
    // joints. Both are kept off for parity/oracle tooling.
    bool structural_select = false;
    bool generic_structural_select = false;
    int generic_candidate_rank = -1;
    rig::PrefixPrefillMode prefix_prefill = rig::PrefixPrefillMode::SharedB1;
    bool diagnostic_prefix_parity = false;
    int pos = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "cuda") use_cuda = true;
        else if (a == "r1") { do_r1 = true; r1_mode = "real"; }   // legacy bare flag -> real path
        else if (a == "r1=real")   { do_r1 = true; r1_mode = "real"; }
        else if (a == "r1=banked") { do_r1 = true; r1_mode = "banked"; }
        else if (a.rfind("r1w=", 0) == 0) r1w = a.c_str() + 4;
        else if (a == "cond=real")   cond_mode = "real";
        else if (a == "cond=banked") cond_mode = "banked";
        else if (a == "cond=auto")   cond_mode = "auto";
        else if (a == "beam") r3_mode = "beam";
        else if (a == "greedy") r3_mode = "greedy";
        else if (a == "sample") r3_mode = "sample";
        else if (a == "tfprobe") r3_mode = "tfprobe";
        else if (a == "nosample") do_sample = false;   // deterministic beam (HF do_sample=False)
        else if (a == "structural-select") structural_select = true;
        else if (a == "generic-structural-select") generic_structural_select = true;
        else if (a.rfind("generic-candidate=", 0) == 0) generic_candidate_rank = std::atoi(a.c_str() + 18);
        else if (a == "prefix_prefill=shared-b1") prefix_prefill = rig::PrefixPrefillMode::SharedB1;
        else if (a == "prefix_prefill=hf-batched-diag") prefix_prefill = rig::PrefixPrefillMode::HfBatchedDiagnostic;
        else if (a == "diagnostic_prefix_parity=1") diagnostic_prefix_parity = true;
        else if (a.rfind("beams=", 0) == 0) num_beams = std::atoi(a.c_str() + 6);
        else if (a.rfind("seed=",  0) == 0) seed   = std::strtoull(a.c_str() + 5, nullptr, 10);
        else if (a.rfind("temp=",  0) == 0) temp   = std::atof(a.c_str() + 5);
        else if (a.rfind("reppen=",0) == 0) reppen = std::atof(a.c_str() + 7);
        else if (a.rfind("topk=",  0) == 0) topk   = std::atoi(a.c_str() + 5);
        else if (a.rfind("topp=",  0) == 0) topp   = std::atof(a.c_str() + 5);
        else if (a.rfind("maxnew=",0) == 0) maxnew = std::atoi(a.c_str() + 7);
        else if (a.rfind("prec=",  0) == 0) {
            std::string p = a.c_str() + 5;
            if (p == "f16") prec = GGML_TYPE_F16; else if (p == "bf16") prec = GGML_TYPE_BF16;
            else prec = GGML_TYPE_F32;
        }
        else if (pos == 0) { e2e = a; pos++; }
        else if (pos == 1) { qwen3_w = a; pos++; }
        else if (pos == 2) { out_glb = a; pos++; }
    }
    if (structural_select && generic_structural_select) {
        std::fprintf(stderr, "choose either structural-select (humanoid) or generic-structural-select\n");
        return 2;
    }
    if (generic_candidate_rank >= 0 && !generic_structural_select) {
        std::fprintf(stderr, "generic-candidate=N requires generic-structural-select\n");
        return 2;
    }
    if (prefix_prefill == rig::PrefixPrefillMode::HfBatchedDiagnostic && !diagnostic_prefix_parity) {
        std::fprintf(stderr, "prefix_prefill=hf-batched-diag requires diagnostic_prefix_parity=1; it is not a production mode\n");
        return 2;
    }
    if (prefix_prefill == rig::PrefixPrefillMode::HfBatchedDiagnostic && r3_mode != "beam") {
        std::fprintf(stderr, "prefix_prefill=hf-batched-diag requires beam decoding\n");
        return 2;
    }
    if (prefix_prefill == rig::PrefixPrefillMode::HfBatchedDiagnostic &&
        std::getenv("RIG_BEAM_SEQ") && std::getenv("RIG_BEAM_SEQ")[0] != '0') {
        std::fprintf(stderr, "prefix_prefill=hf-batched-diag requires the batched beam decoder (unset RIG_BEAM_SEQ)\n");
        return 2;
    }
    std::string r3_detail;
    if (r3_mode == "beam")        r3_detail = " (num_beams=" + std::to_string(num_beams) + ")";
    else if (r3_mode == "sample") r3_detail = " (seed=" + std::to_string(seed) +
                                              " temp=" + std::to_string(temp) +
                                              " reppen=" + std::to_string(reppen) +
                                              " topk=" + std::to_string(topk) +
                                              " topp=" + std::to_string(topp) + ")";
    const char* r1_desc = (do_r1 && r1_mode == "real") ? "R1=real (vecset+output_proj)"
                                                       : "R1=banked learned_mesh_cond.npy";
    printf("[e2e] backend=%s  e2e=%s  qwen3_w=%s  out=%s  mesh_cond=%s  R3=%s%s  cond=%s\n",
           use_cuda ? "cuda" : "cpu", e2e.c_str(), qwen3_w.c_str(), out_glb.c_str(),
           r1_desc, r3_mode.c_str(), r3_detail.c_str(), cond_mode.c_str());

    const char* gguf = "/mnt/hdd/3d/avatar-shootout/_weights/skin_vae_gguf/skin_vae.gguf";

    // ---- load the input mesh (validation uses the e2e golden verts/normals as sampled points) ----
    NpyArray verts_a = npy_load(e2e + "/vertices.npy");   // [N,3]
    NpyArray norms_a = npy_load(e2e + "/normals.npy");     // [N,3]
    const int N = (int)verts_a.shape[0];
    std::vector<float> verts(verts_a.f32(), verts_a.f32() + verts_a.numel());
    std::vector<float> norms(norms_a.f32(), norms_a.f32() + norms_a.numel());
    printf("[e2e] mesh: N=%d vertices\n", N);

    Qwen3Cfg qcfg;            // hidden 896, vocab 33036, 28 layers, prefix "transformer."
    qcfg.compute_type = prec; // AR low-precision path (f32 default = validated)
    printf("[e2e] AR precision = %s\n", prec==GGML_TYPE_F16?"f16":prec==GGML_TYPE_BF16?"bf16":"f32");
    rig::GrammarSpec gspec;   // num_discrete 256, model_vocab 33036, model_eos 33035, tps 4
    const int hidden = qcfg.hidden;

    // ---- PERF instrumentation (steady_clock; NOT wall/time-of-day). Per-stage wall time + VRAM. ----
    using clk = std::chrono::steady_clock;
    auto ms_since = [](clk::time_point a, clk::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    double t_r1 = 0, t_r3 = 0, t_r5 = 0, t_r4 = 0, t_r6 = 0;
    int    r3_generated = 0;                 // generated-token count for R3 tok/s.
    // Baseline = device-used BEFORE any of our CUDA allocations (other procs / CUDA ctx warmup).
    double vram_baseline = e2e_vram_used_mib();
    double vram_total    = e2e_vram_total_mib();
    double vram_peak     = vram_baseline;    // running device high-watermark (used = total-free).
    // NOTE: sample WHILE the relevant M1Harness is still in scope (weights uploaded, not yet freed).
    auto vram_sample = [&]() { double u = e2e_vram_used_mib(); if (u > vram_peak) vram_peak = u; };
    auto t_total0 = clk::now();
    auto t_r1_0 = clk::now();

    // ================= R1: mesh-condition encoder ====================================
    // mesh_cond: host row-major [n_cond, hidden]. Default = banked learned_mesh_cond.npy [1,n_cond,hidden].
    // The R1 path (vecset) requires the FPS-sampled cond query points (nondeterministic), so for the
    // isolation default we consume the golden. (vecset_encode.cpp is the standalone R1 validation.)
    std::vector<float> mesh_cond;
    int n_cond = 0;

    // Helper: load the banked learned_mesh_cond.npy [1,n_cond,hidden] -> mesh_cond + n_cond.
    auto load_banked_cond = [&]() -> bool {
        if (!file_exists(e2e + "/learned_mesh_cond.npy")) {
            printf("[e2e] FAIL: banked learned_mesh_cond.npy not present.\n"); return false;
        }
        NpyArray mc = npy_load(e2e + "/learned_mesh_cond.npy");   // [1, n_cond, hidden]
        n_cond = (int)mc.shape[mc.shape.size() - 2];
        int mc_h = (int)mc.shape[mc.shape.size() - 1];
        if (mc_h != hidden) { printf("[e2e] FAIL: mesh_cond hidden %d != %d\n", mc_h, hidden); return false; }
        mesh_cond.assign(mc.f32(), mc.f32() + mc.numel());
        printf("[STAGE R1] banked learned_mesh_cond: n_cond=%d hidden=%d\n", n_cond, mc_h);
        return true;
    };

    if (!do_r1 || r1_mode == "banked") {
        if (!load_banked_cond()) return 1;
    } else {
        // ---- R1 = REAL: run the VecSet mesh encoder on the input verts/normals, then output_proj.
        //   encode(): input_proj(cat[fourier(pc), normals]) for data(kv) + sampled queries,
        //   cross_attn -> 8x self_attn -> ln_post  => latents [width=512, Q=tokens_skin_cond=512].
        //   output_proj = Sequential(Linear(512->896), RMSNorm(896))  (tokenrig.py L111).
        // Query sampling is the model input, not a harmless preprocessing detail.  Reproduce the
        // Python `default_rng(0).choice(N, 2048, replace=False)` + deterministic FPS locally;
        // never borrow sample points from the reference/weight directory.
        VecsetCfg vcfg;   // dims hard-set from the proven checkpoint (vecset_encoder.hpp header).
        printf("[STAGE R1] R1=real: vecset encoder + output_proj. r1w=%s\n", r1w.c_str());

        // output_proj weights — EXTRACTED-FROM-CKPT requirement. They are part of the TokenRig
        // top-level state_dict, NOT the mesh_encoder.encoder.* dump and NOT the skin gguf:
        //   output_proj.0.weight [896,512]  output_proj.0.bias [896]   (the Linear)
        //   output_proj.1.weight [896]                                  (the RMSNorm gamma)
        // ckpt = experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt ["state_dict"].
        std::string op_w = r1w + "/output_proj.0.weight.npy";
        std::string op_b = r1w + "/output_proj.0.bias.npy";
        std::string op_n = r1w + "/output_proj.1.weight.npy";
        bool have_op = file_exists(op_w) && file_exists(op_b) && file_exists(op_n);

        if (!have_op) {
            printf("[STAGE R1] WARN: output_proj weights NOT on disk; cannot form real mesh_cond.\n");
            if (!load_banked_cond()) return 1;
        } else {
            const int K = std::min(N, 512 * 4);
            std::vector<int> sampled_idx = rig::numpy_choice_seed0_without_replacement(N, K);
            if ((int)sampled_idx.size() != K) { printf("[STAGE R1] FAIL: native NumPy choice failed\n"); return 1; }
            std::vector<float> pre_pc((size_t)K * 3), pre_feats((size_t)K * 3);
            for (int i = 0; i < K; ++i) for (int k = 0; k < 3; ++k) {
                pre_pc[(size_t)i * 3 + k] = verts[(size_t)sampled_idx[i] * 3 + k];
                pre_feats[(size_t)i * 3 + k] = norms[(size_t)sampled_idx[i] * 3 + k];
            }
            std::vector<int> fidx = rig::fps(pre_pc, 512, 0);
            std::vector<float> spc, sft;
            rig::gather(pre_pc, fidx, spc);
            rig::gather(pre_feats, fidx, sft);
            const int64_t Q = (int64_t)fidx.size();
            if (const char* dump_queries = std::getenv("RIG_DUMP_R1_QUERIES")) {
                if (*dump_queries) {
                    const std::string base(dump_queries);
                    npy_write(base + "_pc.npy", spc.data(), sizeof(float), "<f4", {Q, 3});
                    npy_write(base + "_feats.npy", sft.data(), sizeof(float), "<f4", {Q, 3});
                    printf("[STAGE R1] dumped native PCG64+FPS queries -> %s_{pc,feats}.npy\n", base.c_str());
                }
            }
            printf("[STAGE R1] inputs: N=%d data points, Q=%lld native PCG64 choice+FPS queries\n",
                   N, (long long)Q);

            // host Fourier embed (+ normals concat) for the kv data (all verts) + the queries.
            std::vector<float> data_embed = e2e_fourier_embed(verts.data(), norms.data(), N, vcfg);  // [in_dim,N]
            std::vector<float> samp_embed = e2e_fourier_embed(spc.data(),   sft.data(),  Q, vcfg);   // [in_dim,Q]

            // ggml graph for the encoder (npy weights from r1w; PIXAL3D_GGUF_DIR/<base>.gguf absent
            // for this base -> npy fallback). output_proj applied IN-GRAPH when its weights exist.
            M1Harness Hr(r1w, 2048, use_cuda);
            ggml_context* rctx = Hr.ctx;
            int64_t dne[4] = { vcfg.in_dim(), N, 1, 1 };
            int64_t sne[4] = { vcfg.in_dim(), Q, 1, 1 };
            ggml_tensor* data_in = Hr.input("data_embed", 2, dne);
            ggml_tensor* samp_in = Hr.input("sampled_embed", 2, sne);
            ggml_tensor* latents = build_vecset_encoder(Hr, rctx, vcfg, data_in, samp_in);  // [width=512, Q]

            ggml_tensor* cond_t = latents;   // [512, Q]
            if (have_op) {
                // output_proj.0: Linear(512->896, bias).  output_proj.1: RMSNorm(896) -> rms_norm * gamma.
                cond_t = lin(rctx, Hr.weight("output_proj.0.weight"),
                             Hr.weight("output_proj.0.bias"), cond_t);            // [896, Q]
                cond_t = ggml_rms_norm(rctx, cond_t, 1e-5f);                       // RMSNorm eps (nn.RMSNorm default 1e-5)
                cond_t = ggml_mul(rctx, cond_t, Hr.weight("output_proj.1.weight"));// * gamma
            }
            ggml_set_output(cond_t);

            ggml_cgraph* gf = new_graph(rctx, 16384);
            ggml_build_forward_expand(gf, cond_t);
            Hr.alloc_and_upload(gf);
            Hr.upload_input_raw(data_in, data_embed);
            Hr.upload_input_raw(samp_in, samp_embed);
            Hr.compute(gf);
            int out_h = (int)cond_t->ne[0];
            printf("[STAGE R1] ran encoder graph -> mesh_cond [%d, %lld]\n", out_h, (long long)cond_t->ne[1]);
            vram_sample();   // sample while Hr (vecset encoder weights) is still resident.

            if (out_h == hidden) {
                // read back row-major [Q, hidden] (ggml ne0=hidden contiguous == row-major last dim).
                std::vector<float> computed((size_t)Q * hidden);
                ggml_backend_tensor_get(cond_t, computed.data(), 0, computed.size() * sizeof(float));
                // compare vs the banked learned_mesh_cond (bf16 model output; cosine, not bit-exact).
                e2e_cos_vs_npy(computed, e2e + "/learned_mesh_cond.npy", "banked learned_mesh_cond");
                mesh_cond = std::move(computed);
                n_cond = (int)Q;
                printf("[STAGE R1] using REAL mesh_cond: n_cond=%d hidden=%d\n", n_cond, hidden);
            } else { printf("[STAGE R1] FAIL: output_proj returned hidden=%d (expected %d)\n", out_h, hidden); return 1; }
        }
    }

    t_r1 = ms_since(t_r1_0, clk::now());
    // (R1 peak is sampled in-scope above for the --r1 real path; banked mode holds no resident harness.)
    if (const char* dump_cond = std::getenv("RIG_DUMP_MESH_COND")) {
        if (*dump_cond && n_cond > 0 && mesh_cond.size() == (size_t)n_cond * (size_t)hidden) {
            npy_write(dump_cond, mesh_cond.data(), sizeof(float), "<f4", {n_cond, hidden});
            printf("[STAGE R1] dumped mesh_cond [%d,%d] -> %s\n", n_cond, hidden, dump_cond);
        }
    }

    // ================= R3: AR generate ===============================================
    auto t_r3_0 = clk::now();
    // Load embed_tokens.weight [vocab, hidden] (host gather) — npy from the qwen3 weight dir.
    NpyArray embed_a = npy_load(qwen3_w + "/transformer.model.embed_tokens.weight.npy");  // [vocab,hidden]
    if ((int)embed_a.shape[0] != qcfg.vocab || (int)embed_a.shape[1] != hidden) {
        printf("[e2e] FAIL: embed_tokens shape [%lld,%lld] != [%d,%d]\n",
               (long long)embed_a.shape[0], (long long)embed_a.shape[1], qcfg.vocab, hidden);
        return 1;
    }
    std::vector<float> embed_table(embed_a.f32(), embed_a.f32() + embed_a.numel());

    std::vector<int> start_tokens = { gspec.token_id_bos, gspec.token_id_cls_none };  // [257, 263]
    printf("[STAGE R3] start_tokens = {%d, %d} (bos, cls_none); n_cond=%d\n",
           start_tokens[0], start_tokens[1], n_cond);
    printf("[STAGE R3] prefix_prefill=%s effective_batch=%d cache=%s production_eligible=%s%s\n",
           prefix_prefill == rig::PrefixPrefillMode::SharedB1 ? "shared-b1" : "hf-batched-diag",
           prefix_prefill == rig::PrefixPrefillMode::SharedB1 ? 1 : num_beams,
           prefix_prefill == rig::PrefixPrefillMode::SharedB1 ? "shared" : "expanded-then-column0",
           prefix_prefill == rig::PrefixPrefillMode::SharedB1 ? "yes" : "no",
           diagnostic_prefix_parity ? " diagnostic_prefix_parity=acknowledged" : "");

    // Fresh harness with qwen3 weights (npy mode: basename "qwen3_w" won't resolve to a gguf even
    // when PIXAL3D_GGUF_DIR is set for the skin VAE -> falls back to per-tensor npy).
    std::vector<int> tokens;
    std::vector<rig::RigHyp> completed_beams;
    {
        M1Harness Hq(qwen3_w, 64, use_cuda);   // small metadata ctx; weights live in ctx_w
        // f16 weight storage (halves resident weight VRAM; matmul still F32-accumulate). Opt-in via
        // RIG_W_F16=1. Precision change (not bit-exact) — validate the distribution like f16-KV.
        Hq.w_f16 = std::getenv("RIG_W_F16") && std::getenv("RIG_W_F16")[0] != '0';
        if (Hq.w_f16) printf("[STAGE R3] weights = F16 storage (RIG_W_F16)\n");
        // ---- tfprobe: teacher-force a CLEAN golden output_ids_clean.npy + golden mesh_cond, and print
        //      the C++ rank/logp of the switch token (258) at the golden's switch context vs the joints
        //      before it. Isolates qwen3 forward fidelity from search. (compare vs probe_py.py.) ----
        if (r3_mode == "tfprobe") {
            NpyArray sa = npy_load(e2e + "/output_ids_clean.npy");
            std::vector<int> seq(sa.numel());
            for (size_t i = 0; i < (size_t)sa.numel(); ++i) seq[i] = (int)((const int64_t*)sa.raw.data())[i];
            int w = -1; for (int i = 0; i < (int)seq.size(); ++i) if (seq[i] == gspec.token_id_eos) { w = i; break; }
            printf("[tfprobe] golden len=%zu switch(258)@%d  n_cond=%d\n", seq.size(), w, n_cond);
            std::vector<int> seq_in(seq.begin(), seq.end() - 1);
            std::vector<float> out = rig::rig_teacher_force_logits(Hq, qcfg, embed_table.data(),
                                                                   mesh_cond.data(), n_cond, seq_in);
            const int V = qcfg.vocab;
            auto rank_logp = [&](int col, int tok, int& rank, float& lp, int& am) {
                const float* r = &out[(size_t)col * V];
                double mx = -1e300; am = 0; for (int v = 0; v < V; ++v) { if (r[v] > mx) { mx = r[v]; am = v; } }
                rank = 0; for (int v = 0; v < V; ++v) if (r[v] > r[tok]) rank++;
                double s = 0; for (int v = 0; v < V; ++v) s += std::exp((double)r[v] - mx);
                lp = (float)((double)r[tok] - mx - std::log(s));
            };
            for (int j = std::max(1, w - 3); j <= w + 1 && j < (int)seq.size(); ++j) {
                int col = n_cond + (j - 1);   // column predicting seq[j]
                int r258, ramx, gr, gam; float lp258, glp;
                rank_logp(col, gspec.token_id_eos, r258, lp258, ramx);
                rank_logp(col, seq[j], gr, glp, gam);
                printf("[tfprobe]   pred seq[%d]=%d : token258 rank=%d logp=%.3f | golden tok %d rank=%d logp=%.3f | argmax=%d%s\n",
                       j, seq[j], r258, lp258, seq[j], gr, glp, ramx, (j == w ? "  <== golden switch" : ""));
            }
            // FULL-POSITION DUMP: every generated-prediction column [G,V] (cols predicting seq[1..L-1])
            // -> .npy for an apples-to-apples forward-parity diff vs Python (dump_py_logits.py + tf_diff.py).
            {
                int G = (int)seq.size() - 1;
                const std::string dpath = e2e + "/cpp_tf_logits.npy";
                FILE* fp = fopen(dpath.c_str(), "wb");
                if (fp) {
                    char hdr[160];
                    int n = snprintf(hdr, sizeof(hdr),
                        "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }", G, V);
                    int hlen = n + 1;                          // +1 trailing newline
                    int pad = (64 - ((10 + hlen) % 64)) % 64;  // 10-byte prefix; pad header to 64B align
                    hlen += pad;
                    unsigned char pre[10] = {0x93,'N','U','M','P','Y',1,0,
                                             (unsigned char)(hlen & 0xff),(unsigned char)((hlen>>8)&0xff)};
                    fwrite(pre,1,10,fp); fwrite(hdr,1,n,fp);
                    for (int i=0;i<pad;i++) fputc(' ',fp);
                    fputc('\n',fp);
                    for (int j=1;j<(int)seq.size();++j){ int col=n_cond+(j-1);
                        fwrite(&out[(size_t)col*V], sizeof(float), V, fp); }
                    fclose(fp);
                    printf("[tfprobe] dumped C++ logits [%d,%d] -> %s\n", G, V, dpath.c_str());
                }
            }
            return 0;
        }
        if (r3_mode == "sample") {
            tokens = rig::rig_generate_sample(Hq, qcfg, embed_table.data(),
                                              mesh_cond.data(), n_cond, start_tokens, gspec,
                                              /*repetition_penalty=*/reppen, /*temperature=*/temp,
                                              /*top_k=*/topk, /*top_p=*/topp,
                                              /*max_new_tokens=*/(maxnew>0?maxnew:1534), /*seed=*/seed, /*verbose=*/true);
        } else if (r3_mode == "beam") {
            // BEAM-SAMPLE (do_sample=true) by default — matches the golden HF config
            // (.generate(num_beams=10, do_sample=True, temperature=1.5, top_k=10, top_p=0.95,
            //  repetition_penalty=2.0)). Warper params are the shared CLI overrides (temp/topk/topp/seed).
            // DEFAULT = BATCHED decode (all beams in one forward; bit-exact to sequential, ~Nx faster).
            // Force the original sequential path with env RIG_BEAM_SEQ=1 (A/B parity check).
            bool seq_beam = std::getenv("RIG_BEAM_SEQ") && std::getenv("RIG_BEAM_SEQ")[0] != '0';
            printf("[STAGE R3] beam decode = %s\n", seq_beam ? "SEQUENTIAL" : "BATCHED");
            if (seq_beam) {
                tokens = rig::rig_beam_generate(Hq, qcfg, embed_table.data(),
                    mesh_cond.data(), n_cond, start_tokens, gspec,
                    /*num_beams=*/num_beams, /*max_new_tokens=*/(maxnew>0?maxnew:1534),
                    /*length_penalty=*/1.0f, /*repetition_penalty=*/reppen,
                    /*do_sample=*/do_sample, /*temperature=*/temp,
                    /*top_k=*/topk, /*top_p=*/topp, /*seed=*/seed,
                    /*verbose=*/true, &completed_beams);
            } else {
                tokens = rig::rig_beam_generate_batched(Hq, qcfg, embed_table.data(),
                    mesh_cond.data(), n_cond, start_tokens, gspec,
                    /*num_beams=*/num_beams, /*max_new_tokens=*/(maxnew>0?maxnew:1534),
                    /*length_penalty=*/1.0f, /*repetition_penalty=*/reppen,
                    /*do_sample=*/do_sample, /*temperature=*/temp,
                    /*top_k=*/topk, /*top_p=*/topp, /*seed=*/seed,
                    /*verbose=*/true, &completed_beams, prefix_prefill);
            }
        } else {
            tokens = rig::rig_generate(Hq, qcfg, embed_table.data(),
                                       mesh_cond.data(), n_cond, start_tokens, gspec,
                                       /*max_new_tokens=*/(maxnew>0?maxnew:1534), /*verbose=*/true);
        }
        // Sample VRAM peak HERE — Hq (and its ~3 GB weight buffer + the AR/beam KV caches) is still
        // alive in this scope. Sampling after the closing brace would see the device post-free.
        vram_sample();
    }
    t_r3 = ms_since(t_r3_0, clk::now());
    // R3 tok/s basis = generated tokens (full sequence minus the start_tokens prefix), all R3 modes.
    r3_generated = std::max(0, (int)tokens.size() - (int)start_tokens.size());
    printf("[STAGE R3] generated %zu tokens (first eos@? last=%d)\n", tokens.size(), tokens.empty() ? -1 : tokens.back());

    // token-match vs golden output_ids.npy (validation only — absent when rigging a fresh mesh).
    if (file_exists(e2e + "/output_ids.npy")) {
        NpyArray oi = npy_load(e2e + "/output_ids.npy");
        const int64_t* g = oi.i64(); int64_t ng = oi.numel();
        int64_t cmp = std::min((int64_t)tokens.size(), ng), match = 0;
        for (int64_t i = 0; i < cmp; ++i) if (tokens[i] == (int)g[i]) match++;
        printf("[STAGE R3] token match vs golden: %lld/%lld over first %lld (cpp_len=%zu golden_len=%lld)\n",
               (long long)match, (long long)cmp, (long long)cmp, tokens.size(), (long long)ng);
    }

    // dump generated tokens (int64, [n]) for the main loop to diff/visualize.
    {
        const std::string dump_dir = "/tmp/skintokens_e2e";
        std::vector<int64_t> tk(tokens.begin(), tokens.end());
        std::string fn = "/gen_tokens_" + r3_mode + ".npy";   // gen_tokens_sample.npy / _beam / _greedy
        npy_write(dump_dir + fn, tk.data(), sizeof(int64_t), "<i8", { (int64_t)tk.size() });
        printf("[STAGE R3] dumped %zu tokens -> %s%s\n", tk.size(), dump_dir.c_str(), fn.c_str());
    }

    // ================= R5: detokenize -> joints + parents ============================
    auto t_r5_0 = clk::now();
    detok::Spec dspec = detok::load_spec(e2e + "/tok_spec.txt");
    if (structural_select || generic_structural_select) {
        if (r3_mode != "beam" || completed_beams.empty()) {
            printf("[STAGE R3] FAIL: structural selection requires completed beam hypotheses.\n");
            return 1;
        }
        // The gate itself now lives in rig_structural_select.hpp so the inline image_to_rig rig
        // (rig_pipeline.hpp) runs the SAME selection instead of taking the top-normscore beam.
        rig::StructuralPick pick = rig::structural_select(completed_beams, dspec,
                                                          generic_structural_select,
                                                          generic_candidate_rank, /*verbose=*/true);
        if (pick.index < 0) {
            printf("[STAGE R3] FAIL: no completed real-conditioned beam passed the %s gate.\n",
                   generic_structural_select ? "generic skeleton" : "22-bone anatomy");
            return 1;
        }
        tokens = pick.tokens;
    }
    std::vector<int64_t> tok64(tokens.begin(), tokens.end());
    detok::Skeleton sk = detok::detokenize(tok64.data(), (int64_t)tok64.size(), dspec);
    if (!sk.ok) { printf("[STAGE R5] FAIL: %s\n", sk.err.c_str()); return 1; }
    if (generic_structural_select) {
        std::string normalized;
        if (rig::normalize_generic_parent_fan(sk.joints, sk.parents, &normalized))
            printf("[STAGE R5] %s\n", normalized.c_str());
    }
    const int J = sk.J;
    printf("[STAGE R5] J=%d joints, parents derived\n", J);
    {
        std::string jp = e2e + "/detok_joints.npy";
        if (file_exists(jp)) {
            NpyArray gj = npy_load(jp); const float* g = gj.f32();
            int Jg = (int)gj.shape[0];
            float maxabs = 0;
            for (int k = 0; k < std::min(J, Jg); ++k)
                for (int c = 0; c < 3; ++c) maxabs = std::max(maxabs, std::fabs(sk.joints[(size_t)k * 3 + c] - g[k * 3 + c]));
            printf("[STAGE R5] joints max-abs vs golden = %.3e (J cpp=%d golden=%d)\n", maxabs, J, Jg);
        }
    }

    // dump detok joints [J,3] (float32) + parents [J] (int64) for the main loop to diff/visualize.
    {
        const std::string dump_dir = "/tmp/skintokens_e2e";
        npy_write(dump_dir + "/gen_joints.npy", sk.joints.data(), sizeof(float), "<f4",
                  { (int64_t)J, 3 });
        std::vector<int64_t> par(sk.parents.begin(), sk.parents.end());
        npy_write(dump_dir + "/gen_parents.npy", par.data(), sizeof(int64_t), "<i8",
                  { (int64_t)par.size() });
        printf("[STAGE R5] dumped joints[%d,3] -> %s/gen_joints.npy + parents[%zu] -> gen_parents.npy\n",
               J, dump_dir.c_str(), par.size());
    }

    t_r5 = ms_since(t_r5_0, clk::now());

    // ================= R4: skin-weight decode =======================================
    auto t_r4_0 = clk::now();
    rig::SkinVaeCfg scfg;
    const int Tz = scfg.tokens_per_skin;

    // skin indices: ALL group tokens after the first tokenizer-eos (incl. trailing model-eos ->
    // raw 33035-267=32768 -> wraps to FSQ index 0). need = J*tokens_per_skin.
    size_t eos_pos = tok64.size();
    for (size_t i = 0; i < tok64.size(); ++i) if (tok64[i] == TOK_EOS) { eos_pos = i; break; }
    const int need = J * Tz;
    std::vector<int64_t> indices; indices.reserve(need);
    bool have_skin = (eos_pos + 1 + need) <= tok64.size();
    if (!have_skin) {
        printf("[STAGE R4] WARN: not enough skin tokens after eos (have %lld, need %d) — generation likely truncated.\n",
               (long long)((int64_t)tok64.size() - (int64_t)eos_pos - 1), need);
    }
    for (int i = 0; i < need && have_skin; ++i) {
        int64_t tk = tok64[eos_pos + 1 + i] - TOKENIZER_VOCAB;
        if (tk < 0) tk = 0;
        if (tk >= VAE_VOCAB) tk -= VAE_VOCAB;  // model-eos wraps to 0
        indices.push_back(tk);
    }

    // FSQ indices -> codes [need, 512]
    rig::Fsq fsq; fsq.init();
    { GgufReader gr(gguf);
      fsq.proj_out_w = gr.tensor_f32("model.FSQ.project_out.weight");
      fsq.proj_out_b = gr.tensor_f32("model.FSQ.project_out.bias"); }
    std::vector<float> codes = have_skin ? fsq.indices_to_codes(indices, true) : std::vector<float>();

    // ---- cond_latents resolution ---------------------------------------------------------------
    // Two paths (R4 is the only stage that still consumed a giraffe-specific golden):
    //   banked: load cond_latents.npy  (ONLY valid for the banked giraffe; leaks the giraffe rig
    //           onto any other mesh — so it must NOT be used when rigging a new mesh).
    //   real:   COMPUTE cond_latents from THIS mesh's verts/normals via the validated cond-encoder:
    //             cond_kv_embed = skin_embed_with_feats(ALL N verts, normals)            [in_feat, N]
    //             cond_q  = default_rng(0).choice(N, 4*384) -> fps(->384, start 0), then
    //                       cond_q_embed = skin_embed_with_feats(those 384, their normals)
    //             cond_latents = build_cond_encoder(cond_q_embed, cond_kv_embed)          [512, 384]
    // Mode resolution: explicit cond=real|banked wins; "auto" => real when r1=real (we're rigging a
    // NEW mesh, so the banked giraffe cond_latents would be wrong), else banked.
    bool cond_real = (cond_mode == "real") ||
                     (cond_mode == "auto" && do_r1 && r1_mode == "real");
    // The skin cond-token count = tokens_skin_cond = 384 (decoder cross-attn KV is z(4)+cond(384)).
    const int SKIN_COND_TOKENS = 384;

    int n_cond_skin = 0;
    std::vector<float> cond_lat;
    // Always load the banked cond_latents if present — it's the banked path AND the giraffe sanity
    // reference for the cond=real cosine print.
    std::vector<float> cond_lat_banked;
    int n_cond_banked = 0;
    if (file_exists(e2e + "/cond_latents.npy")) {
        NpyArray cl_a = npy_load(e2e + "/cond_latents.npy");      // [384, 512]
        n_cond_banked = (int)cl_a.shape[0];
        cond_lat_banked.assign(cl_a.f32(), cl_a.f32() + cl_a.numel());
    }

    if (cond_real && have_skin) {
        // ---- COMPUTE cond_latents from this mesh (mesh-agnostic; no giraffe leakage) ----
        if (!std::getenv("PIXAL3D_GGUF_DIR"))
            setenv("PIXAL3D_GGUF_DIR", "/mnt/hdd/3d/avatar-shootout/_weights/skin_vae_gguf", 0);

        // cond-query selection: exact fixed-seed NumPy choice(N, 4*384) -> FPS -> 384 (start 0).
        // This is the same seeded condition sampling used by the parity probe; mt19937 here would
        // silently produce a different learned skin field even after R1 had become exact.
        const int Qc = std::min(SKIN_COND_TOKENS, N);
        const int Kc = std::min(Qc * 4, N);
        if (seed != 0) { printf("[STAGE R4] FAIL: exact native condition sampling requires seed=0\n"); return 1; }
        std::vector<int> pre_idx = rig::numpy_choice_seed0_without_replacement(N, Kc);
        if ((int)pre_idx.size() != Kc) { printf("[STAGE R4] FAIL: native NumPy choice failed\n"); return 1; }
        std::vector<float> pre_pts((size_t)Kc * 3), pre_nrm((size_t)Kc * 3);
        for (int i = 0; i < Kc; ++i)
            for (int k = 0; k < 3; ++k) {
                pre_pts[(size_t)i * 3 + k] = verts[(size_t)pre_idx[i] * 3 + k];
                pre_nrm[(size_t)i * 3 + k] = norms[(size_t)pre_idx[i] * 3 + k];
            }
        std::vector<int> fidx = rig::fps(pre_pts, Qc, 0);   // 4Q -> Q, start index 0 (random_start=False)
        const int Q = (int)fidx.size();
        std::vector<float> q_pts((size_t)Q * 3), q_nrm((size_t)Q * 3);
        for (int i = 0; i < Q; ++i)
            for (int k = 0; k < 3; ++k) {
                q_pts[(size_t)i * 3 + k] = pre_pts[(size_t)fidx[i] * 3 + k];
                q_nrm[(size_t)i * 3 + k] = pre_nrm[(size_t)fidx[i] * 3 + k];
            }

        // embed cond_q (the 384 FPS points) and cond_kv (ALL N points). cond = cat[verts,normals]
        // so the "features" fed to the PMPE embedder are the per-point normals (cond_channels=3).
        std::vector<float> cq_embed = rig::skin_embed_with_feats(q_pts.data(), q_nrm.data(), Q, scfg);  // [in_feat,Q]
        std::vector<float> ckv_embed = rig::skin_embed_with_feats(verts.data(), norms.data(), N, scfg); // [in_feat,N]

        M1Harness Hc("skin_vae", 4096, use_cuda);
        int64_t qcne[4] = { scfg.in_feat(), Q, 1, 1 };
        int64_t kcne[4] = { scfg.in_feat(), N, 1, 1 };
        ggml_tensor* qin = Hc.input("cond_q", 2, qcne);
        ggml_tensor* kin = Hc.input("cond_kv", 2, kcne);
        ggml_tensor* cl  = rig::build_cond_encoder(Hc, Hc.ctx, scfg, qin, kin);   // [512, Q]
        ggml_set_output(cl);
        ggml_cgraph* gfc = new_graph(Hc.ctx, 16384);
        ggml_build_forward_expand(gfc, cl);
        Hc.alloc_and_upload(gfc);
        Hc.upload_input_raw(qin, cq_embed);
        Hc.upload_input_raw(kin, ckv_embed);
        Hc.compute(gfc);
        vram_sample();   // sample while the cond-encoder weights are still resident.

        n_cond_skin = Q;
        cond_lat.resize((size_t)scfg.latent * Q);   // ggml [512, Q] read back row-major == [Q, 512]
        ggml_backend_tensor_get(cl, cond_lat.data(), 0, cond_lat.size() * sizeof(float));
        printf("[STAGE R4] cond_latents (REAL, computed): [%d,%d]; cond-query=NumPy-PCG64-choice(%d,%d)->fps->%d (start 0)\n",
               n_cond_skin, scfg.latent, N, Kc, Q);

        // sanity cosine vs banked giraffe cond_latents (high ~>0.99 ONLY for the giraffe; a NEW mesh
        // will legitimately differ — informational, NOT gated). Also bf16-golden-vs-fp32-compute, so
        // even the giraffe won't be bit-exact (cosine is the right stat).
        if (!cond_lat_banked.empty() && (int)cond_lat_banked.size() == (int)cond_lat.size()) {
            double dot = 0, na = 0, nb = 0, maxabs = 0;
            for (size_t i = 0; i < cond_lat.size(); ++i) {
                double a = cond_lat[i], b = cond_lat_banked[i];
                dot += a * b; na += a * a; nb += b * b;
                maxabs = std::max(maxabs, std::fabs(a - b));
            }
            double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
            printf("[STAGE R4] computed cond_latents vs banked cond_latents.npy: cosine=%.6f max-abs=%.3e "
                   "(EXPECT ~>0.99 for the giraffe golden; a new mesh differs legitimately — NOT gated)\n",
                   cos, maxabs);
        } else if (!cond_lat_banked.empty()) {
            printf("[STAGE R4] (cond_latents shape %d != banked %d -> skip cosine sanity; new mesh / token-count)\n",
                   n_cond_skin, n_cond_banked);
        }
    } else {
        // ---- BANKED cond_latents (fallback; giraffe validation path) ----
        if (cond_real && !have_skin)
            printf("[STAGE R4] cond=real requested but no skin tokens (truncated gen) -> banked.\n");
        if (cond_lat_banked.empty()) {
            printf("[STAGE R4] FAIL: banked cond_latents.npy not present and cond=real not selected.\n");
            return 1;
        }
        n_cond_skin = n_cond_banked;
        cond_lat = cond_lat_banked;
        printf("[STAGE R4] cond_latents (banked): [%d,%d]; skin group = %d tokens\n", n_cond_skin, scfg.latent, need);
    }

    std::vector<float> skin_pred((size_t)N * J, 0.f);
    if (have_skin) {
        // GGUF mode for the skin VAE: env PIXAL3D_GGUF_DIR + basename "skin_vae".
        if (!std::getenv("PIXAL3D_GGUF_DIR"))
            setenv("PIXAL3D_GGUF_DIR", "/mnt/hdd/3d/avatar-shootout/_weights/skin_vae_gguf", 0);
        std::vector<float> q_embed = rig::skin_embed_with_feats(verts.data(), norms.data(), N, scfg); // [in_feat, N]

        M1Harness Hs("skin_vae", 4096, use_cuda);
        ggml_context* ctx = Hs.ctx;
        int64_t zne[4] = { scfg.latent, Tz, 1, 1 };
        int64_t cne[4] = { scfg.latent, n_cond_skin, 1, 1 };
        int64_t qne[4] = { scfg.in_feat(), N, 1, 1 };
        ggml_tensor* z_in   = Hs.input("z_codes", 2, zne);
        ggml_tensor* cond_t = Hs.const_tensor("cond_lat", 2, cne, cond_lat);     // persistent [512,384]
        ggml_tensor* q_t    = Hs.const_tensor("q_embed", 2, qne, q_embed);       // persistent [in_feat,N]
        ggml_tensor* sp = rig::build_decoder_joint(Hs, ctx, scfg, z_in, cond_t, q_t);  // [1, N]
        ggml_set_output(sp);
        ggml_cgraph* gf = new_graph(ctx, 16384);
        ggml_build_forward_expand(gf, sp);
        Hs.alloc_and_upload(gf);

        std::vector<float> zbuf((size_t)scfg.latent * Tz), got(N);
        for (int j = 0; j < J; ++j) {
            for (int t = 0; t < Tz; ++t)
                memcpy(&zbuf[(size_t)t * scfg.latent], &codes[((size_t)j * Tz + t) * scfg.latent], scfg.latent * 4);
            ggml_backend_tensor_set(z_in, zbuf.data(), 0, zbuf.size() * 4);
            Hs.compute(gf);
            ggml_backend_tensor_get(sp, got.data(), 0, (size_t)N * 4);
            for (int n = 0; n < N; ++n) skin_pred[(size_t)n * J + j] = got[n];  // [N,J] row-major
        }
        printf("[STAGE R4] skin_pred computed [%d,%d]\n", N, J);
        npy_write("/tmp/skintokens_e2e/gen_skin_pred.npy", skin_pred.data(), 4, "<f4",
                  { (int64_t)N, (int64_t)J });
        printf("[STAGE R4] dumped skin_pred[%d,%d] -> /tmp/skintokens_e2e/gen_skin_pred.npy\n", N, J);
        // Sample VRAM while Hs (skin-VAE weights + activations) is still resident.
        vram_sample();

        // diff vs the fp32 oracle (PASS gate) + the bf16 golden (informational).
        auto diff_npy = [&](const std::string& path, const char* tag) {
            if (!file_exists(path)) return;
            NpyArray a = npy_load(path);
            if (a.numel() != (int64_t)skin_pred.size()) { printf("[STAGE R4] %s nelem mismatch\n", tag); return; }
            const float* g = a.f32(); double mx = 0, sm = 0;
            for (size_t i = 0; i < skin_pred.size(); ++i) { double d = std::fabs((double)skin_pred[i] - (double)g[i]); sm += d; if (d > mx) mx = d; }
            printf("[STAGE R4] skin_pred vs %s: max-abs=%.3e mean-abs=%.3e\n", tag, mx, sm / skin_pred.size());
        };
        diff_npy(e2e + "/r4_oracle_skin_pred_fp32.npy", "fp32 ORACLE");
        diff_npy(e2e + "/skin_pred.npy",                "bf16 golden");
    }

    t_r4 = ms_since(t_r4_0, clk::now());

    // ================= R6: write rigged GLB =========================================
    auto t_r6_0 = clk::now();
    std::vector<int64_t> faces;   // no faces from the sampled point cloud -> writer falls back to +Z normals
    bool wrote = glb::write_rigged_glb(out_glb.c_str(), verts, faces, sk.joints, sk.parents, skin_pred, &norms);
    printf("[STAGE R6] write_rigged_glb -> %s : %s\n", out_glb.c_str(), wrote ? "OK" : "FAIL");
    t_r6 = ms_since(t_r6_0, clk::now());

    // ================= PERF summary =================================================
    double t_total = ms_since(t_total0, clk::now());
    double r3_s = t_r3 / 1000.0;
    double tok_s = (r3_s > 0.0) ? (double)r3_generated / r3_s : 0.0;
    printf("[PERF] R1=%.1f ms  R3=%.1f ms (%d tokens, %.1f tok/s)  R5=%.1f ms  R4=%.1f ms  "
           "R6=%.1f ms  TOTAL=%.1f ms\n",
           t_r1, t_r3, r3_generated, tok_s, t_r5, t_r4, t_r6, t_total);
    if (vram_peak >= 0.0)
        printf("[PERF] VRAM: baseline_used=%.0f MiB  peak_used=%.0f MiB  (total=%.0f MiB) [cudaMemGetInfo, used=total-free]\n",
               vram_baseline, vram_peak, vram_total);
    else
        printf("[PERF] VRAM: unavailable (CPU build / no cudart) — query via nvidia-smi externally\n");

    printf("[e2e] DONE (R1->R3->R5->R4->R6). Validate the prints above on the GPU run.\n");
    return wrote ? 0 : 1;
}
