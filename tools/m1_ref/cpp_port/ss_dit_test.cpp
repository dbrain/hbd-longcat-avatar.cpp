// SS dense flow DiT (SparseStructureFlowModel) single forward in ggml.
//
// Mirrors tools/m1_ref/ss_dit.py model_forward (validated fp32 numpy oracle, itself
// cross-checked vs the real torch model @ 1.5e-5). 30x ModulatedTransformerCrossBlock:
//   share_mod adaLN, full self-attn (qk-RMS-norm + complex interleaved RoPE),
//   ProjectAttention (cross-attn over 5 global tokens + per-block proj_linear add),
//   GELU-tanh MLP. norm1/norm3 non-affine eps1e-6, norm2 AFFINE eps1e-6, final LN eps1e-5.
//
// Validates (CPU backend, vs refs/): dit_block0 [1,4096,1536], dit_v [1,8,16,16,16].
#include "m1_ggml.hpp"
#include <cmath>

static const char* WDIR = "weights_npy/ss_flow";
static const char* REFS = "refs";

static const int C = 1536, NB = 30, NH = 12, HD = 128, SEQ = 4096, INCH = 8, MLP = 8192;
static const int TE_HALF = 128;  // timestep_embedding: dim=256, half=128 (NOT head_dim)

// complex interleaved RoPE: out=x*COS+rot(x)*SIN, rot[2p]=-x[2p+1],rot[2p+1]=x[2p].
// t [d=128, head, token]; COS/SIN [d=128, 1, token].
static ggml_tensor* rope_inter(ggml_context* ctx, ggml_tensor* t, ggml_tensor* COS, ggml_tensor* SIN) {
    int64_t d = t->ne[0], nh = t->ne[1], tok = t->ne[2];
    ggml_tensor* t5 = ggml_reshape_4d(ctx, t, 2, d / 2, nh, tok);  // [2,64,head,token]
    ggml_tensor* x0 = ggml_cont(ctx, ggml_view_4d(ctx, t5, 1, d / 2, nh, tok,
                                  t5->nb[1], t5->nb[2], t5->nb[3], 0));            // real
    ggml_tensor* x1 = ggml_cont(ctx, ggml_view_4d(ctx, t5, 1, d / 2, nh, tok,
                                  t5->nb[1], t5->nb[2], t5->nb[3], t5->nb[0]));    // imag
    ggml_tensor* rot5 = ggml_concat(ctx, ggml_neg(ctx, x1), x0, 0);  // [2,64,head,token]
    ggml_tensor* rot = ggml_reshape_3d(ctx, rot5, d, nh, tok);
    return ggml_add(ctx, ggml_mul(ctx, t, COS), ggml_mul(ctx, rot, SIN));
}

int main(int argc, char** argv) {
    bool use_cuda = (argc > 1 && std::string(argv[1]) == "cuda");
    M1Harness H(WDIR, 512, use_cuda);
    ggml_context* ctx = H.ctx;

    // ---- inputs ----
    int64_t x_ne[4] = {SEQ, INCH, 1, 1};     // dit_x.npy [1,8,16,16,16] flat = [ch,token]
    ggml_tensor* xin = H.input("x", 2, x_ne);
    int64_t t_ne[4] = {1, 1, 1, 1};
    ggml_tensor* tin = H.input("t", 1, t_ne);
    int64_t cs_ne[4] = {HD, 1, SEQ, 1};      // COS/SIN [d,1,token]
    ggml_tensor* COS = H.input("rope_cos", 3, cs_ne);
    ggml_tensor* SIN = H.input("rope_sin", 3, cs_ne);
    int64_t fr_ne[4] = {TE_HALF, 1, 1, 1};   // timestep freqs [128]
    ggml_tensor* freqs = H.input("freqs", 1, fr_ne);
    // cond (golden global/proj)
    int64_t g_ne[4] = {1024, 5, 1, 1};
    ggml_tensor* gctx = H.input("global", 2, g_ne);
    int64_t p_ne[4] = {1024, SEQ, 1, 1};
    ggml_tensor* pctx = H.input("proj", 2, p_ne);

    // ---- t embedding + share_mod global modulation ----
    ggml_tensor* args = ggml_mul(ctx, freqs, tin);              // [128] = freqs*t
    ggml_tensor* temb = ggml_concat(ctx, ggml_cos(ctx, args), ggml_sin(ctx, args), 0);  // [256]
    temb = lin(ctx, H.weight("t_embedder.mlp.0.weight"), H.weight("t_embedder.mlp.0.bias"), temb);
    temb = silu_(ctx, temb);
    temb = lin(ctx, H.weight("t_embedder.mlp.2.weight"), H.weight("t_embedder.mlp.2.bias"), temb);  // [1536]
    temb = silu_(ctx, temb);
    ggml_tensor* tmod = lin(ctx, H.weight("adaLN_modulation.1.weight"), H.weight("adaLN_modulation.1.bias"), temb); // [9216]

    // ---- input layer ----
    ggml_tensor* h = ggml_cont(ctx, ggml_transpose(ctx, xin));  // [ch=8, token]
    h = lin(ctx, H.weight("input_layer.weight"), H.weight("input_layer.bias"), h);  // [1536, token]

    ggml_tensor* block0_tap = nullptr;
    for (int bi = 0; bi < NB; bi++) {
        std::string bp = "blocks." + std::to_string(bi) + ".";
        // share_mod: mod6 = modulation[block] + tmod, chunk 6
        ggml_tensor* mod6 = ggml_add(ctx, H.weight(bp + "modulation"), tmod);  // [9216]
        auto chunk = [&](int i) { return ggml_view_1d(ctx, mod6, C, (size_t)i * C * ggml_element_size(mod6)); };
        ggml_tensor* s_msa = chunk(0), *sc_msa = chunk(1), *g_msa = chunk(2);
        ggml_tensor* s_mlp = chunk(3), *sc_mlp = chunk(4), *g_mlp = chunk(5);

        // --- self-attn (norm1 non-affine, modulate, qk-rms-norm + rope) ---
        ggml_tensor* a = layernorm(ctx, h, nullptr, nullptr, 1e-6f);
        a = modulate(ctx, a, sc_msa, s_msa);
        ggml_tensor* qkv = lin(ctx, H.weight(bp + "self_attn.to_qkv.weight"), H.weight(bp + "self_attn.to_qkv.bias"), a);
        qkv = ggml_reshape_4d(ctx, qkv, HD, NH, 3, SEQ);   // [d,head,three,token]
        auto take = [&](int three) {
            ggml_tensor* t = ggml_cont(ctx, ggml_view_4d(ctx, qkv, HD, NH, 1, SEQ,
                                       qkv->nb[1], qkv->nb[2], qkv->nb[3], (size_t)three * qkv->nb[2]));
            return ggml_reshape_3d(ctx, t, HD, NH, SEQ);
        };
        ggml_tensor* q = take(0), *k = take(1), *v = take(2);
        q = mh_rms_norm(ctx, q, H.weight(bp + "self_attn.q_rms_norm.gamma"));
        k = mh_rms_norm(ctx, k, H.weight(bp + "self_attn.k_rms_norm.gamma"));
        q = rope_inter(ctx, q, COS, SIN);
        k = rope_inter(ctx, k, COS, SIN);
        ggml_tensor* sa = attention(ctx, q, k, v, 1.0f / std::sqrt((float)HD));  // [1536, token]
        sa = lin(ctx, H.weight(bp + "self_attn.to_out.weight"), H.weight(bp + "self_attn.to_out.bias"), sa);
        sa = ggml_mul(ctx, sa, g_msa);   // gate
        h = ggml_add(ctx, h, sa);

        // --- ProjectAttention (norm2 AFFINE, cross-attn global + proj_linear) ---
        ggml_tensor* hn = layernorm(ctx, h, H.weight(bp + "norm2.weight"), H.weight(bp + "norm2.bias"), 1e-6f);
        // cross_attn over global (5 tokens)
        std::string cab = bp + "cross_attn.cross_attn_block.";
        ggml_tensor* cq = lin(ctx, H.weight(cab + "to_q.weight"), H.weight(cab + "to_q.bias"), hn);  // [1536,token]
        cq = ggml_reshape_3d(ctx, cq, HD, NH, SEQ);
        ggml_tensor* ckv = lin(ctx, H.weight(cab + "to_kv.weight"), H.weight(cab + "to_kv.bias"), gctx);  // [3072,5]
        ckv = ggml_reshape_4d(ctx, ckv, HD, NH, 2, 5);   // [d,head,two,5]
        auto takekv = [&](int two) {
            ggml_tensor* t = ggml_cont(ctx, ggml_view_4d(ctx, ckv, HD, NH, 1, 5,
                                       ckv->nb[1], ckv->nb[2], ckv->nb[3], (size_t)two * ckv->nb[2]));
            return ggml_reshape_3d(ctx, t, HD, NH, 5);
        };
        ggml_tensor* ck = takekv(0), *cv = takekv(1);
        cq = mh_rms_norm(ctx, cq, H.weight(cab + "q_rms_norm.gamma"));
        ck = mh_rms_norm(ctx, ck, H.weight(cab + "k_rms_norm.gamma"));
        ggml_tensor* ca = attention(ctx, cq, ck, cv, 1.0f / std::sqrt((float)HD));   // [1536,token]
        ca = lin(ctx, H.weight(cab + "to_out.weight"), H.weight(cab + "to_out.bias"), ca);
        // proj_linear (per-token linear add)
        ggml_tensor* pj = lin(ctx, H.weight(bp + "cross_attn.proj_linear.weight"), H.weight(bp + "cross_attn.proj_linear.bias"), pctx);  // [1536,token]
        h = ggml_add(ctx, h, ggml_add(ctx, ca, pj));

        // --- MLP (norm3 non-affine, modulate, gelu-tanh) ---
        ggml_tensor* m = layernorm(ctx, h, nullptr, nullptr, 1e-6f);
        m = modulate(ctx, m, sc_mlp, s_mlp);
        m = lin(ctx, H.weight(bp + "mlp.mlp.0.weight"), H.weight(bp + "mlp.mlp.0.bias"), m);  // [8192,token]
        m = gelu_tanh_(ctx, m);
        m = lin(ctx, H.weight(bp + "mlp.mlp.2.weight"), H.weight(bp + "mlp.mlp.2.bias"), m);  // [1536,token]
        m = ggml_mul(ctx, m, g_mlp);
        h = ggml_add(ctx, h, m);

        if (bi == 0) { h = ggml_cont(ctx, h); block0_tap = h; ggml_set_output(block0_tap); }
    }

    // final non-affine LN (eps 1e-5) + out_layer
    h = layernorm(ctx, h, nullptr, nullptr, 1e-5f);
    h = lin(ctx, H.weight("out_layer.weight"), H.weight("out_layer.bias"), h);  // [8, token]
    ggml_tensor* vout = ggml_cont(ctx, ggml_transpose(ctx, h));  // [token, 8] == dit_v flat
    ggml_set_output(vout);

    // ---- run ----
    ggml_cgraph* gf = new_graph(ctx, 32768);
    ggml_build_forward_expand(gf, vout);
    ggml_build_forward_expand(gf, block0_tap);
    H.alloc_and_upload(gf);

    // upload inputs
    H.upload_input_npy(xin, std::string(REFS) + "/dit_x.npy");
    H.upload_input_npy(tin, std::string(REFS) + "/dit_t.npy");
    H.upload_input_npy(gctx, "refs/dino_global.npy");   // == golden global (cond), C-order
    H.upload_input_npy(pctx, "refs/proj.npy");          // == golden proj cond (1.76e-5), C-order
    // freqs: exp(-ln(10000)*i/half), half=128
    std::vector<float> fr(TE_HALF);
    for (int i = 0; i < TE_HALF; i++) fr[i] = std::exp(-std::log(10000.0f) * i / TE_HALF);
    H.upload_input_raw(freqs, fr);
    // COS/SIN [d=128,1,token] from rope_phases [4096,64,2]; replicate per pair
    {
        NpyArray rp = npy_load(std::string(WDIR) + "/rope_phases.npy");  // [4096,64,2]
        const float* rpf = rp.f32();
        std::vector<float> cosb((size_t)HD * SEQ), sinb((size_t)HD * SEQ);
        for (int t = 0; t < SEQ; t++)
            for (int p = 0; p < HD / 2; p++) {
                float cz = rpf[((size_t)t * (HD / 2) + p) * 2 + 0];
                float sz = rpf[((size_t)t * (HD / 2) + p) * 2 + 1];
                size_t base = (size_t)t * HD;
                cosb[base + 2 * p] = cz; cosb[base + 2 * p + 1] = cz;
                sinb[base + 2 * p] = sz; sinb[base + 2 * p + 1] = sz;
            }
        H.upload_input_raw(COS, cosb);
        H.upload_input_raw(SIN, sinb);
    }

    H.compute(gf);

    printf("[ss_dit] backend=%s\n", use_cuda ? "cuda" : "cpu");
    CmpStats b0 = compare_to_npy(H, block0_tap, std::string(REFS) + "/dit_block0.npy", true, "block0");
    CmpStats vv = compare_to_npy(H, vout, std::string(REFS) + "/dit_v.npy", true, "v");
    bool ok = b0.maxabs < 5e-4 && vv.maxabs < 5e-4;
    printf("[ss_dit] %s (maxabs<5e-4)\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
