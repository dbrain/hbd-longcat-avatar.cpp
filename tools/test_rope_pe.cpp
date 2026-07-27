// Numerical equivalence test: ggml_rope_pe vs the apply_rope cont+repeat+mul chain.
// Build inside the builder image; links libggml. Run on CUDA.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Replicate the OLD interleaved chain (rope_interleaved=true) verbatim.
static ggml_tensor * apply_rope_chain(ggml_context * ctx, ggml_tensor * x, ggml_tensor * pe) {
    int64_t d_head = x->ne[0];
    int64_t L      = x->ne[2];
    int64_t n_head = x->ne[1];
    int64_t N      = x->ne[3];
    x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
    x = ggml_reshape_4d(ctx, x, 2, d_head / 2, L, n_head * N);
    x = ggml_cont(ctx, ggml_permute(ctx, x, 3, 0, 1, 2));
    int64_t offset = x->nb[2] * x->ne[2];
    ggml_tensor * x_0 = ggml_view_3d(ctx, x, x->ne[0], x->ne[1], x->ne[2], x->nb[1], x->nb[2], offset * 0);
    ggml_tensor * x_1 = ggml_view_3d(ctx, x, x->ne[0], x->ne[1], x->ne[2], x->nb[1], x->nb[2], offset * 1);
    x_0 = ggml_reshape_4d(ctx, x_0, 1, x_0->ne[0], x_0->ne[1], x_0->ne[2]);
    x_1 = ggml_reshape_4d(ctx, x_1, 1, x_1->ne[0], x_1->ne[1], x_1->ne[2]);
    ggml_tensor * temp_x = ggml_new_tensor_4d(ctx, x_0->type, 2, x_0->ne[1], x_0->ne[2], x_0->ne[3]);
    x_0 = ggml_repeat(ctx, x_0, temp_x);
    x_1 = ggml_repeat(ctx, x_1, temp_x);
    pe = ggml_cont(ctx, ggml_permute(ctx, pe, 3, 0, 1, 2));
    offset = pe->nb[2] * pe->ne[2];
    ggml_tensor * pe_0 = ggml_view_3d(ctx, pe, pe->ne[0], pe->ne[1], pe->ne[2], pe->nb[1], pe->nb[2], offset * 0);
    ggml_tensor * pe_1 = ggml_view_3d(ctx, pe, pe->ne[0], pe->ne[1], pe->ne[2], pe->nb[1], pe->nb[2], offset * 1);
    ggml_tensor * x_out = ggml_add_inplace(ctx, ggml_mul(ctx, x_0, pe_0), ggml_mul(ctx, x_1, pe_1));
    x_out = ggml_reshape_3d(ctx, x_out, d_head, L, n_head * N);
    return x_out;
}

// Replicate the NON-interleaved chain (rope_interleaved=false) verbatim from
// Rope::apply_rope: the pair is the first/second HALF of head_dim (not adjacent),
// with the extra trailing permute(1,0,2,3). torch_permute(0,2,3,1) == ggml_permute(0,3,1,2).
static ggml_tensor * apply_rope_chain_ni(ggml_context * ctx, ggml_tensor * x, ggml_tensor * pe) {
    int64_t d_head = x->ne[0];
    int64_t n_head = x->ne[1];
    int64_t L      = x->ne[2];
    int64_t N      = x->ne[3];
    x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
    x = ggml_reshape_4d(ctx, x, d_head / 2, 2, L, n_head * N);
    x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 3, 1, 2));  // torch_permute(0,2,3,1)
    int64_t offset = x->nb[2] * x->ne[2];
    ggml_tensor * x_0 = ggml_view_3d(ctx, x, x->ne[0], x->ne[1], x->ne[2], x->nb[1], x->nb[2], offset * 0);
    ggml_tensor * x_1 = ggml_view_3d(ctx, x, x->ne[0], x->ne[1], x->ne[2], x->nb[1], x->nb[2], offset * 1);
    x_0 = ggml_reshape_4d(ctx, x_0, 1, x_0->ne[0], x_0->ne[1], x_0->ne[2]);
    x_1 = ggml_reshape_4d(ctx, x_1, 1, x_1->ne[0], x_1->ne[1], x_1->ne[2]);
    ggml_tensor * temp_x = ggml_new_tensor_4d(ctx, x_0->type, 2, x_0->ne[1], x_0->ne[2], x_0->ne[3]);
    x_0 = ggml_repeat(ctx, x_0, temp_x);
    x_1 = ggml_repeat(ctx, x_1, temp_x);
    pe = ggml_cont(ctx, ggml_permute(ctx, pe, 3, 0, 1, 2));
    offset = pe->nb[2] * pe->ne[2];
    ggml_tensor * pe_0 = ggml_view_3d(ctx, pe, pe->ne[0], pe->ne[1], pe->ne[2], pe->nb[1], pe->nb[2], offset * 0);
    ggml_tensor * pe_1 = ggml_view_3d(ctx, pe, pe->ne[0], pe->ne[1], pe->ne[2], pe->nb[1], pe->nb[2], offset * 1);
    ggml_tensor * x_out = ggml_add_inplace(ctx, ggml_mul(ctx, x_0, pe_0), ggml_mul(ctx, x_1, pe_1));
    x_out = ggml_cont(ctx, ggml_permute(ctx, x_out, 1, 0, 2, 3));
    x_out = ggml_reshape_3d(ctx, x_out, d_head, L, n_head * N);
    return x_out;
}

int main() {
    // This is the exact layout used by LTX video self-attention after it folds
    // heads into L: [d_head, 1, tokens * heads, 1].
    const int64_t d_head = 128, n_head = 1, L = 257 * 32, N = 1;
    const int64_t half = d_head / 2;

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { printf("no cuda\n"); return 1; }

    ggml_init_params ip{ (size_t)64*1024*1024, NULL, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * x  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_head, n_head, L, N);
    ggml_tensor * pe = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 2, half, L);
    ggml_set_input(x); ggml_set_input(pe);

    ggml_tensor * out_chain = apply_rope_chain(ctx, x, pe);
    ggml_set_output(out_chain);
    ggml_tensor * out_fused = ggml_rope_pe(ctx, x, pe);
    ggml_set_output(out_fused);

    // non-interleaved (NeoX / LTX video) chain vs ggml_rope_pe_ni
    ggml_tensor * out_chain_ni = apply_rope_chain_ni(ctx, x, pe);
    ggml_set_output(out_chain_ni);
    ggml_tensor * out_fused_ni = ggml_rope_pe_ni(ctx, x, pe);
    ggml_set_output(out_fused_ni);

    // Compact basis has the full hidden-size pair dimension, indexed by the
    // original token and axis. Give all three axes the same entry in this
    // synthetic test so every dense value is reconstructed exactly.
    const int64_t rope_heads = 32;
    const int64_t tokens = L / rope_heads;
    const int64_t full_half = half * rope_heads;
    ggml_tensor * compact_basis = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 2, full_half, tokens);
    ggml_tensor * compact_index = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 3, tokens);
    ggml_set_input(compact_basis); ggml_set_input(compact_index);
    ggml_tensor * out_compact_ni = ggml_rope_pe_ni_compact(ctx, x, compact_basis, compact_index);
    ggml_set_output(out_compact_ni);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out_chain);
    ggml_build_forward_expand(gf, out_fused);
    ggml_build_forward_expand(gf, out_chain_ni);
    ggml_build_forward_expand(gf, out_fused_ni);
    ggml_build_forward_expand(gf, out_compact_ni);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    (void)buf;

    // fill inputs
    std::vector<float> xv(d_head*n_head*L*N), pv(2*2*half*L);
    std::vector<float> compact_pv(4*full_half*tokens, 0.f);
    std::vector<int32_t> compact_iv(3*tokens);
    srand(1234);
    for (auto & v : xv) v = (rand()/(float)RAND_MAX) * 2.f - 1.f;
    // pe per Rope::rope: result[4j..4j+3] = cos, -sin, sin, cos ; layout flat t*256... here width=4*half
    for (int64_t t = 0; t < L; ++t) {
        for (int64_t j = 0; j < half; ++j) {
            // Match LTX's two leading padding pairs in the full 4096-wide
            // RoPE grid; compact correctly treats those as identity.
            const int64_t global_pair = (t % rope_heads) * half + j;
            float ang = global_pair < (full_half % 3)
                ? 0.f
                : (0.3f * t) * powf(10000.f, -2.f*j/d_head);
            float c = cosf(ang), s = sinf(ang);
            int64_t base = t*(4*half) + 4*j;
            pv[base+0] = c; pv[base+1] = -s; pv[base+2] = s; pv[base+3] = c;
        }
    }
    for (int64_t token = 0; token < tokens; ++token) {
        compact_iv[3*token + 0] = token;
        compact_iv[3*token + 1] = token;
        compact_iv[3*token + 2] = token;
        for (int64_t head = 0; head < rope_heads; ++head) {
            for (int64_t j = 0; j < half; ++j) {
                const int64_t src = (token * rope_heads + head) * (4*half) + 4*j;
                const int64_t dst = token * (4*full_half) + 4*(head*half + j);
                for (int k = 0; k < 4; ++k) compact_pv[dst+k] = pv[src+k];
            }
        }
    }
    ggml_backend_tensor_set(x,  xv.data(), 0, xv.size()*sizeof(float));
    ggml_backend_tensor_set(pe, pv.data(), 0, pv.size()*sizeof(float));
    ggml_backend_tensor_set(compact_basis, compact_pv.data(), 0, compact_pv.size()*sizeof(float));
    ggml_backend_tensor_set(compact_index, compact_iv.data(), 0, compact_iv.size()*sizeof(int32_t));

    ggml_backend_graph_compute(backend, gf);

    std::vector<float> a(d_head*L*n_head*N), b(d_head*L*n_head*N);
    ggml_backend_tensor_get(out_chain, a.data(), 0, a.size()*sizeof(float));
    ggml_backend_tensor_get(out_fused, b.data(), 0, b.size()*sizeof(float));

    double maxabs = 0; int64_t worst = -1;
    for (size_t i = 0; i < a.size(); ++i) {
        double dd = fabs(a[i]-b[i]);
        if (dd > maxabs) { maxabs = dd; worst = (int64_t)i; }
    }
    printf("[interleaved]     n=%zu  max|chain-fused| = %.3e  at idx %lld (chain=%.6f fused=%.6f)\n",
           a.size(), maxabs, (long long)worst,
           worst>=0?a[worst]:0.f, worst>=0?b[worst]:0.f);

    std::vector<float> ani(d_head*L*n_head*N), bni(d_head*L*n_head*N);
    ggml_backend_tensor_get(out_chain_ni, ani.data(), 0, ani.size()*sizeof(float));
    ggml_backend_tensor_get(out_fused_ni, bni.data(), 0, bni.size()*sizeof(float));
    double maxabs_ni = 0; int64_t worst_ni = -1;
    for (size_t i = 0; i < ani.size(); ++i) {
        double dd = fabs(ani[i]-bni[i]);
        if (dd > maxabs_ni) { maxabs_ni = dd; worst_ni = (int64_t)i; }
    }
    printf("[non-interleaved] n=%zu  max|chain-fused| = %.3e  at idx %lld (chain=%.6f fused=%.6f)\n",
           ani.size(), maxabs_ni, (long long)worst_ni,
           worst_ni>=0?ani[worst_ni]:0.f, worst_ni>=0?bni[worst_ni]:0.f);
    std::vector<float> cni(d_head*L*n_head*N);
    ggml_backend_tensor_get(out_compact_ni, cni.data(), 0, cni.size()*sizeof(float));
    double maxabs_compact = 0; int64_t worst_compact = -1;
    for (size_t i = 0; i < bni.size(); ++i) {
        const double dd = fabs(bni[i] - cni[i]);
        if (dd > maxabs_compact) { maxabs_compact = dd; worst_compact = (int64_t) i; }
    }
    printf("[compact NeoX]    n=%zu  max|dense-compact| = %.3e  at idx %lld (dense=%.6f compact=%.6f)\n",
           cni.size(), maxabs_compact, (long long) worst_compact,
           worst_compact>=0?bni[worst_compact]:0.f, worst_compact>=0?cni[worst_compact]:0.f);
    // dump first 16 of each
    printf("chain[0..15]:");    for (int i=0;i<16 && i<(int)a.size();++i)   printf(" %.4f", a[i]);   printf("\n");
    printf("fused[0..15]:");    for (int i=0;i<16 && i<(int)b.size();++i)   printf(" %.4f", b[i]);   printf("\n");
    printf("chain_ni[0..15]:"); for (int i=0;i<16 && i<(int)ani.size();++i) printf(" %.4f", ani[i]); printf("\n");
    printf("fused_ni[0..15]:"); for (int i=0;i<16 && i<(int)bni.size();++i) printf(" %.4f", bni[i]); printf("\n");
    return (maxabs < 1e-5 && maxabs_ni < 1e-5 && maxabs_compact == 0.0) ? 0 : 2;
}
