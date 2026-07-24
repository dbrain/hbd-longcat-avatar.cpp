// rig_pipeline.hpp — the SkinTokens auto-rig (R1->R3->R5->R4) as a reusable in-process function.
//
// Extracted verbatim from the validated skintokens_e2e.cpp "real" path (r1=real, cond=real, beam
// decode, bf16) so the inline image->rigged-asset API runs the SAME code as the scaffold, with no
// disk round-trip and no golden/validation cruft. Inputs are the mesh-sample outputs already in RAM
// (rig::prep_mesh_for_rig_inmem); outputs are joints[J,3] + parents[J] + skin_pred[N,J].
//
// Header-only; links ggml (CPU or CUDA) via the same recipe as skintokens_e2e. Weights:
//   r1w           : vecset encoder + output_proj.* per-tensor npy   (rig_audit/r1w_real)
//   qwen3_w       : transformer.model.* per-tensor npy              (rig_audit/inputs/qwen3_w)
//   skin_vae_gguf : dir holding skin_vae.gguf                       (_weights/skin_vae_gguf)
#pragma once
#include "qwen3_forward.hpp"
#include "rig_beam_generate.hpp"
#include "rig_beam_generate_batched.hpp"
#include "rig_sample.hpp"
#include "rig_grammar.hpp"
#include "detok_r5.hpp"
#include "rig_skin_decoder.hpp"
#include "rig_fsq.hpp"
#include "vecset_encoder.hpp"
#include "mesh_sample.hpp"     // rig::fps, prep result types
#include "gguf_reader.hpp"
#include "../../sparse_spike/npy.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <string>
#include <vector>

namespace rig {

struct RigOpts {
    std::string r1w;             // vecset + output_proj npy weights dir
    std::string qwen3_w;         // qwen3 per-tensor npy weights dir (embed_tokens etc.)
    std::string skin_vae_gguf;   // dir containing skin_vae.gguf
    bool use_cuda = true;
    int   num_beams = 10;        // HF-official beam-sample config (default rig recipe)
    float temp = 1.0f, reppen = 2.0f, topp = 0.95f;
    int   topk = 5;
    int   max_new = 1534;        // max_length 2048 - 514-token prefix
    ggml_type prec = GGML_TYPE_BF16;   // native AR regime (matches Python torch_dtype=bfloat16)
    bool  do_sample = true;
    uint64_t seed = 0;
    bool  verbose = true;
};

struct RigResult {
    std::vector<float> joints;     // [J,3]
    std::vector<int>   parents;    // [J]  root = -1
    std::vector<float> skin_pred;  // [N,J] row-major
    int J = 0, N = 0;
    bool ok = false;
};

// host FourierEmbedder (exact; mirrors skintokens_e2e::e2e_fourier_embed). [in_dim, Npts].
static inline std::vector<float> rig_fourier_embed(const float* pts, const float* feats, int64_t Npts,
                                                   const VecsetCfg& cfg) {
    const int F = cfg.num_freqs;
    std::vector<float> freq(F);
    for (int k = 0; k < F; ++k) {
        float f = std::pow(2.0f, (float)k);
        if (cfg.include_pi) f *= (float)M_PI;
        freq[k] = f;
    }
    const int in_dim = cfg.in_dim();
    std::vector<float> out((size_t)Npts * in_dim);
    for (int64_t i = 0; i < Npts; ++i) {
        const float* p = pts + i * 3;
        float* o = out.data() + i * in_dim;
        int w = 0;
        if (cfg.include_input) for (int c = 0; c < 3; ++c) o[w++] = p[c];
        for (int c = 0; c < 3; ++c) for (int k = 0; k < F; ++k) o[w++] = std::sin(p[c] * freq[k]);
        for (int c = 0; c < 3; ++c) for (int k = 0; k < F; ++k) o[w++] = std::cos(p[c] * freq[k]);
        if (feats) for (int c = 0; c < cfg.point_feats; ++c) o[w++] = feats[i * 3 + c];
    }
    return out;
}

// ---- R1 -> R3 -> R5 -> R4. Mirrors skintokens_e2e.cpp (r1=real, cond=real, beam). ----
inline RigResult run_rig_pipeline(const std::vector<float>& verts, const std::vector<float>& normals,
                                  const std::vector<float>& sampled_pc,
                                  const std::vector<float>& sampled_feats, const RigOpts& opt) {
    RigResult RR;
    const int N = (int)(verts.size() / 3);
    RR.N = N;
    auto log = [&](const char* fmt, auto... args) { if (opt.verbose) printf(fmt, args...); };

    Qwen3Cfg qcfg; qcfg.compute_type = opt.prec;
    rig::GrammarSpec gspec;
    const int hidden = qcfg.hidden;

    static const int TOKENIZER_VOCAB = 267;
    static const int VAE_VOCAB       = 32768;
    static const int TOK_EOS         = 258;

    // ================= R1: vecset mesh-condition encoder + output_proj ==========================
    std::vector<float> mesh_cond;
    int n_cond = 0;
    {
        VecsetCfg vcfg;
        const int64_t Q = (int64_t)(sampled_pc.size() / 3);
        std::vector<float> data_embed = rig_fourier_embed(verts.data(), normals.data(), N, vcfg);
        std::vector<float> samp_embed = rig_fourier_embed(sampled_pc.data(), sampled_feats.data(), Q, vcfg);

        M1Harness Hr(opt.r1w.c_str(), 2048, opt.use_cuda);
        ggml_context* rctx = Hr.ctx;
        int64_t dne[4] = { vcfg.in_dim(), N, 1, 1 };
        int64_t sne[4] = { vcfg.in_dim(), Q, 1, 1 };
        ggml_tensor* data_in = Hr.input("data_embed", 2, dne);
        ggml_tensor* samp_in = Hr.input("sampled_embed", 2, sne);
        ggml_tensor* latents = build_vecset_encoder(Hr, rctx, vcfg, data_in, samp_in);  // [512, Q]
        // output_proj.0 Linear(512->896,bias) + output_proj.1 RMSNorm(896)*gamma.
        ggml_tensor* cond_t = lin(rctx, Hr.weight("output_proj.0.weight"),
                                  Hr.weight("output_proj.0.bias"), latents);
        cond_t = ggml_rms_norm(rctx, cond_t, 1e-5f);
        cond_t = ggml_mul(rctx, cond_t, Hr.weight("output_proj.1.weight"));
        ggml_set_output(cond_t);
        ggml_cgraph* gf = new_graph(rctx, 16384);
        ggml_build_forward_expand(gf, cond_t);
        Hr.alloc_and_upload(gf);
        Hr.upload_input_raw(data_in, data_embed);
        Hr.upload_input_raw(samp_in, samp_embed);
        Hr.compute(gf);
        if ((int)cond_t->ne[0] != hidden) { fprintf(stderr, "[rig] R1 output dim %lld != %d\n",
                                                    (long long)cond_t->ne[0], hidden); return RR; }
        mesh_cond.resize((size_t)Q * hidden);
        ggml_backend_tensor_get(cond_t, mesh_cond.data(), 0, mesh_cond.size() * sizeof(float));
        n_cond = (int)Q;
        log("[rig R1] mesh_cond [%d,%d]\n", n_cond, hidden);
    }

    // ================= R3: AR generate (beam-sample, HF-official) ===============================
    std::vector<int> tokens;
    {
        NpyArray embed_a = npy_load(opt.qwen3_w + "/transformer.model.embed_tokens.weight.npy");
        if ((int)embed_a.shape[0] != qcfg.vocab || (int)embed_a.shape[1] != hidden) {
            fprintf(stderr, "[rig] embed_tokens shape mismatch\n"); return RR;
        }
        std::vector<float> embed_table(embed_a.f32(), embed_a.f32() + embed_a.numel());
        std::vector<int> start_tokens = { gspec.token_id_bos, gspec.token_id_cls_none };

        M1Harness Hq(opt.qwen3_w.c_str(), 64, opt.use_cuda);
        Hq.w_f16 = std::getenv("RIG_W_F16") && std::getenv("RIG_W_F16")[0] != '0';
        bool seq_beam = std::getenv("RIG_BEAM_SEQ") && std::getenv("RIG_BEAM_SEQ")[0] != '0';
        // rig_beam_generate_batched gained a trailing PrefixPrefillMode parameter
        // (a batched-only concept); rig_beam_generate did not.  Default arguments
        // are not part of a function's type, so the two can no longer be unified
        // into one function pointer via ?: (that was the build break).  Dispatch
        // with if/else instead — both trailing optionals (finished_out,
        // prefix_prefill) take their defaults, so the batched path keeps its
        // production SharedB1 prefill and behaviour is unchanged.
        if (seq_beam)
            tokens = rig::rig_beam_generate(Hq, qcfg, embed_table.data(), mesh_cond.data(), n_cond, start_tokens, gspec,
                         /*num_beams=*/opt.num_beams, /*max_new_tokens=*/opt.max_new,
                         /*length_penalty=*/1.0f, /*repetition_penalty=*/opt.reppen,
                         /*do_sample=*/opt.do_sample, /*temperature=*/opt.temp,
                         /*top_k=*/opt.topk, /*top_p=*/opt.topp, /*seed=*/opt.seed,
                         /*verbose=*/opt.verbose);
        else
            tokens = rig::rig_beam_generate_batched(Hq, qcfg, embed_table.data(), mesh_cond.data(), n_cond, start_tokens, gspec,
                         /*num_beams=*/opt.num_beams, /*max_new_tokens=*/opt.max_new,
                         /*length_penalty=*/1.0f, /*repetition_penalty=*/opt.reppen,
                         /*do_sample=*/opt.do_sample, /*temperature=*/opt.temp,
                         /*top_k=*/opt.topk, /*top_p=*/opt.topp, /*seed=*/opt.seed,
                         /*verbose=*/opt.verbose);
    }
    log("[rig R3] generated %zu tokens (last=%d)\n", tokens.size(), tokens.empty() ? -1 : tokens.back());

    // ================= R5: detokenize -> joints + parents ======================================
    std::vector<int64_t> tok64(tokens.begin(), tokens.end());
    detok::Spec dspec = detok::load_spec("");   // missing path -> golden-matching defaults
    detok::Skeleton sk = detok::detokenize(tok64.data(), (int64_t)tok64.size(), dspec);
    if (!sk.ok) { fprintf(stderr, "[rig] R5 detok failed: %s\n", sk.err.c_str()); return RR; }
    const int J = sk.J;
    RR.joints = sk.joints;
    RR.parents = sk.parents;
    RR.J = J;
    log("[rig R5] J=%d joints\n", J);

    // ================= R4: skin-weight decode ==================================================
    rig::SkinVaeCfg scfg;
    const int Tz = scfg.tokens_per_skin;
    // skin indices: all group tokens after the first tokenizer-eos. need = J*Tz.
    size_t eos_pos = tok64.size();
    for (size_t i = 0; i < tok64.size(); ++i) if (tok64[i] == TOK_EOS) { eos_pos = i; break; }
    const int need = J * Tz;
    bool have_skin = (eos_pos + 1 + need) <= tok64.size();
    if (!have_skin) {
        fprintf(stderr, "[rig] R4 WARN: not enough skin tokens (have %lld need %d) — gen truncated; "
                "skin_pred left zero.\n",
                (long long)((int64_t)tok64.size() - (int64_t)eos_pos - 1), need);
        RR.skin_pred.assign((size_t)N * J, 0.f);
        RR.ok = true;   // joints/parents are still valid
        return RR;
    }
    std::vector<int64_t> indices; indices.reserve(need);
    for (int i = 0; i < need; ++i) {
        int64_t tk = tok64[eos_pos + 1 + i] - TOKENIZER_VOCAB;
        if (tk < 0) tk = 0;
        if (tk >= VAE_VOCAB) tk -= VAE_VOCAB;
        indices.push_back(tk);
    }

    std::string gguf = opt.skin_vae_gguf + "/skin_vae.gguf";
    rig::Fsq fsq; fsq.init();
    { GgufReader gr(gguf.c_str());
      fsq.proj_out_w = gr.tensor_f32("model.FSQ.project_out.weight");
      fsq.proj_out_b = gr.tensor_f32("model.FSQ.project_out.bias"); }
    std::vector<float> codes = fsq.indices_to_codes(indices, true);

    // cond_latents = COMPUTE from THIS mesh (cond=real; mesh-agnostic, no giraffe leakage).
    setenv("PIXAL3D_GGUF_DIR", opt.skin_vae_gguf.c_str(), 1);
    const int SKIN_COND_TOKENS = 384;
    int n_cond_skin = 0;
    std::vector<float> cond_lat;
    {
        const int Qc = std::min(SKIN_COND_TOKENS, N);
        const int Kc = std::min(Qc * 4, N);
        std::vector<int> pre_idx(N);
        for (int i = 0; i < N; ++i) pre_idx[i] = i;
        std::mt19937_64 crng(opt.seed);
        for (int i = 0; i < Kc; ++i) {
            std::uniform_int_distribution<int> D(i, N - 1);
            std::swap(pre_idx[i], pre_idx[D(crng)]);
        }
        std::vector<float> pre_pts((size_t)Kc * 3), pre_nrm((size_t)Kc * 3);
        for (int i = 0; i < Kc; ++i)
            for (int k = 0; k < 3; ++k) {
                pre_pts[(size_t)i * 3 + k] = verts[(size_t)pre_idx[i] * 3 + k];
                pre_nrm[(size_t)i * 3 + k] = normals[(size_t)pre_idx[i] * 3 + k];
            }
        std::vector<int> fidx = rig::fps(pre_pts, Qc, 0);
        const int Q = (int)fidx.size();
        std::vector<float> q_pts((size_t)Q * 3), q_nrm((size_t)Q * 3);
        for (int i = 0; i < Q; ++i)
            for (int k = 0; k < 3; ++k) {
                q_pts[(size_t)i * 3 + k] = pre_pts[(size_t)fidx[i] * 3 + k];
                q_nrm[(size_t)i * 3 + k] = pre_nrm[(size_t)fidx[i] * 3 + k];
            }
        std::vector<float> cq_embed = rig::skin_embed_with_feats(q_pts.data(), q_nrm.data(), Q, scfg);
        std::vector<float> ckv_embed = rig::skin_embed_with_feats(verts.data(), normals.data(), N, scfg);

        M1Harness Hc("skin_vae", 4096, opt.use_cuda);
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
        n_cond_skin = Q;
        cond_lat.resize((size_t)scfg.latent * Q);
        ggml_backend_tensor_get(cl, cond_lat.data(), 0, cond_lat.size() * sizeof(float));
        log("[rig R4] cond_latents [%d,%d]\n", n_cond_skin, scfg.latent);
    }

    // skin decode: per-joint decoder over the FSQ codes + cond_latents + per-vertex query embed.
    RR.skin_pred.assign((size_t)N * J, 0.f);
    {
        std::vector<float> q_embed = rig::skin_embed_with_feats(verts.data(), normals.data(), N, scfg);
        M1Harness Hs("skin_vae", 4096, opt.use_cuda);
        ggml_context* ctx = Hs.ctx;
        int64_t zne[4] = { scfg.latent, Tz, 1, 1 };
        int64_t cne[4] = { scfg.latent, n_cond_skin, 1, 1 };
        int64_t qne[4] = { scfg.in_feat(), N, 1, 1 };
        ggml_tensor* z_in   = Hs.input("z_codes", 2, zne);
        ggml_tensor* cond_t = Hs.const_tensor("cond_lat", 2, cne, cond_lat);
        ggml_tensor* q_t    = Hs.const_tensor("q_embed", 2, qne, q_embed);
        ggml_tensor* sp = rig::build_decoder_joint(Hs, ctx, scfg, z_in, cond_t, q_t);   // [1, N]
        ggml_set_output(sp);
        ggml_cgraph* gf = new_graph(ctx, 16384);
        ggml_build_forward_expand(gf, sp);
        Hs.alloc_and_upload(gf);
        std::vector<float> zbuf((size_t)scfg.latent * Tz), got(N);
        for (int j = 0; j < J; ++j) {
            for (int t = 0; t < Tz; ++t)
                memcpy(&zbuf[(size_t)t * scfg.latent], &codes[((size_t)j * Tz + t) * scfg.latent],
                       scfg.latent * sizeof(float));
            ggml_backend_tensor_set(z_in, zbuf.data(), 0, zbuf.size() * sizeof(float));
            Hs.compute(gf);
            ggml_backend_tensor_get(sp, got.data(), 0, (size_t)N * sizeof(float));
            for (int n = 0; n < N; ++n) RR.skin_pred[(size_t)n * J + j] = got[n];
        }
        log("[rig R4] skin_pred [%d,%d]\n", N, J);
    }

    RR.ok = true;
    return RR;
}

}  // namespace rig
