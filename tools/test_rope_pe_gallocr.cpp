// Repro: does the fused ggml_rope_pe diverge from the chain UNDER GALLOCR
// (buffer reuse), as opposed to the isolation test which uses
// ggml_backend_alloc_ctx_tensors (every tensor its own persistent buffer)?
//
// Mirrors the real DiT self-attn topology:
//   x[d_head,n_head,L,N] -> rope (fused|chain) -> q_rope[d_head,L,HN]
//   -> view(cond/noise) -> cont -> flash_attn(q,k,v) -> proj matmul
// run BOTH a fused-rope branch and a chain-rope branch in ONE graph through
// ggml_gallocr (the real allocator), then compare the two final outputs.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static ggml_tensor * apply_rope_chain(ggml_context * ctx, ggml_tensor * x, ggml_tensor * pe) {
    int64_t d_head = x->ne[0], n_head = x->ne[1], L = x->ne[2], N = x->ne[3];
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

// Non-interleaved (NeoX / LTX video) chain replica.
static ggml_tensor * apply_rope_chain_ni(ggml_context * ctx, ggml_tensor * x, ggml_tensor * pe) {
    int64_t d_head = x->ne[0], n_head = x->ne[1], L = x->ne[2], N = x->ne[3];
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

// Build the consumer side that mirrors the real self_attn: split q_rope/k_rope
// into cond+noise, cont, flash-attn each, concat, then a proj matmul. Returns
// the proj output. `fused` selects fused op vs chain; `interleaved` selects the
// interleaved (GPT-J) vs non-interleaved (NeoX / LTX video) RoPE variant.
static ggml_tensor * build_attn(ggml_context * ctx, ggml_backend_t backend,
                                ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                                ggml_tensor * pe, ggml_tensor * proj_w,
                                bool fused, bool interleaved) {
    int64_t d_head = q->ne[0], n_head = q->ne[1], L = q->ne[2];
    int64_t HN = n_head;  // N==1
    ggml_tensor * q_rope = fused ? (interleaved ? ggml_rope_pe(ctx, q, pe) : ggml_rope_pe_ni(ctx, q, pe))
                                 : (interleaved ? apply_rope_chain(ctx, q, pe) : apply_rope_chain_ni(ctx, q, pe));
    ggml_tensor * k_rope = fused ? (interleaved ? ggml_rope_pe(ctx, k, pe) : ggml_rope_pe_ni(ctx, k, pe))
                                 : (interleaved ? apply_rope_chain(ctx, k, pe) : apply_rope_chain_ni(ctx, k, pe));
    // q_rope/k_rope: [d_head, L, HN]
    int64_t n_cond = 16;  // pretend cond-frame tokens
    int64_t L_noise = L - n_cond;
    float scale = 1.0f / sqrtf((float)d_head);

    auto run_fa = [&](ggml_tensor * qv, ggml_tensor * kv, ggml_tensor * vv) {
        // qv: [d_head, Lq, HN]; kv: [d_head, Lk, HN]; vv must be [d_head, n_head, Lk, N]
        ggml_tensor * kf = ggml_cast(ctx, kv, GGML_TYPE_F16);
        ggml_tensor * vp = ggml_cont(ctx, ggml_permute(ctx, vv, 0, 2, 1, 3));
        vp = ggml_reshape_3d(ctx, vp, d_head, vv->ne[2], n_head);
        ggml_tensor * vf = ggml_cast(ctx, vp, GGML_TYPE_F16);
        ggml_tensor * o = ggml_flash_attn_ext(ctx, qv, kf, vf, nullptr, scale, 0, 0);
        ggml_flash_attn_ext_set_prec(o, GGML_PREC_F32);
        return o;  // [d_head, n_head, Lq, ...]
    };

    // cond pass: q_cond x {k,v}_cond
    ggml_tensor * q_cond = ggml_cont(ctx, ggml_view_3d(ctx, q_rope, d_head, n_cond, HN, q_rope->nb[1], q_rope->nb[2], 0));
    ggml_tensor * k_cond = ggml_cont(ctx, ggml_view_3d(ctx, k_rope, d_head, n_cond, HN, k_rope->nb[1], k_rope->nb[2], 0));
    ggml_tensor * v_cond = ggml_cont(ctx, ggml_view_4d(ctx, v, v->ne[0], v->ne[1], n_cond, v->ne[3], v->nb[1], v->nb[2], v->nb[3], 0));
    ggml_tensor * x_cond = run_fa(q_cond, k_cond, v_cond);

    // noise pass: q_noise x {k,v}_full
    ggml_tensor * q_noise = ggml_cont(ctx, ggml_view_3d(ctx, q_rope, d_head, L_noise, HN, q_rope->nb[1], q_rope->nb[2], q_rope->nb[1] * n_cond));
    ggml_tensor * x_noise = run_fa(q_noise, k_rope, v);

    // x_cond/x_noise: [d_head, n_head, Lq]. concat along token axis (ne[2]) -> [d_head,n_head,L]
    ggml_tensor * out = ggml_concat(ctx, x_cond, x_noise, 2);
    out = ggml_reshape_3d(ctx, out, d_head * n_head, L, 1);  // [C, L, 1]
    // proj matmul: proj_w [C, C] x out -> [C, L, 1]
    out = ggml_mul_mat(ctx, proj_w, out);
    return out;
}

int main() {
    const int64_t d_head = 128, n_head = 32, L = 257, N = 1;
    const int64_t half = d_head / 2, C = d_head * n_head;

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { printf("no cuda\n"); return 1; }

    ggml_init_params ip{ (size_t)256*1024*1024, NULL, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * q  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_head, n_head, L, N);
    ggml_tensor * k  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_head, n_head, L, N);
    ggml_tensor * v  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_head, n_head, L, N);
    ggml_tensor * pe = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 2, half, L);
    ggml_tensor * pw = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, C);
    ggml_set_input(q); ggml_set_input(k); ggml_set_input(v);
    ggml_set_input(pe); ggml_set_input(pw);

    ggml_tensor * out_chain = build_attn(ctx, backend, q, k, v, pe, pw, false, true);
    ggml_set_output(out_chain);
    ggml_tensor * out_fused = build_attn(ctx, backend, q, k, v, pe, pw, true, true);
    ggml_set_output(out_fused);
    ggml_tensor * out_chain_ni = build_attn(ctx, backend, q, k, v, pe, pw, false, false);
    ggml_set_output(out_chain_ni);
    ggml_tensor * out_fused_ni = build_attn(ctx, backend, q, k, v, pe, pw, true, false);
    ggml_set_output(out_fused_ni);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(gf, out_chain);
    ggml_build_forward_expand(gf, out_fused);
    ggml_build_forward_expand(gf, out_chain_ni);
    ggml_build_forward_expand(gf, out_fused_ni);

    // REAL allocator (gallocr) — this is what the render uses, NOT alloc_ctx_tensors.
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_reserve(galloc, gf)) { printf("reserve failed\n"); return 1; }
    if (!ggml_gallocr_alloc_graph(galloc, gf)) { printf("alloc_graph failed\n"); return 1; }

    // fill inputs (need ggml_backend buffers for inputs — gallocr allocated them)
    std::vector<float> qv(C*L), kv(C*L), vv(C*L), pv(2*2*half*L), pwv(C*C);
    srand(1234);
    for (auto & z : qv) z = (rand()/(float)RAND_MAX)*2.f-1.f;
    for (auto & z : kv) z = (rand()/(float)RAND_MAX)*2.f-1.f;
    for (auto & z : vv) z = (rand()/(float)RAND_MAX)*2.f-1.f;
    for (auto & z : pwv) z = (rand()/(float)RAND_MAX)*0.05f;
    for (int64_t t = 0; t < L; ++t)
      for (int64_t j = 0; j < half; ++j) {
        float ang = (0.3f*t)*powf(10000.f, -2.f*j/d_head);
        float c = cosf(ang), s = sinf(ang);
        int64_t base = t*(4*half)+4*j;
        pv[base+0]=c; pv[base+1]=-s; pv[base+2]=s; pv[base+3]=c;
      }
    ggml_backend_tensor_set(q,  qv.data(), 0, qv.size()*sizeof(float));
    ggml_backend_tensor_set(k,  kv.data(), 0, kv.size()*sizeof(float));
    ggml_backend_tensor_set(v,  vv.data(), 0, vv.size()*sizeof(float));
    ggml_backend_tensor_set(pe, pv.data(), 0, pv.size()*sizeof(float));
    ggml_backend_tensor_set(pw, pwv.data(),0, pwv.size()*sizeof(float));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { printf("compute failed\n"); return 1; }

    std::vector<float> a(C*L), b(C*L), ani(C*L), bni(C*L);
    ggml_backend_tensor_get(out_chain, a.data(), 0, a.size()*sizeof(float));
    ggml_backend_tensor_get(out_fused, b.data(), 0, b.size()*sizeof(float));
    ggml_backend_tensor_get(out_chain_ni, ani.data(), 0, ani.size()*sizeof(float));
    ggml_backend_tensor_get(out_fused_ni, bni.data(), 0, bni.size()*sizeof(float));
    double maxabs=0; int64_t worst=-1; double suma=0,sumb=0;
    for (size_t i=0;i<a.size();++i){ double dd=fabs(a[i]-b[i]); if(dd>maxabs){maxabs=dd;worst=(int64_t)i;} suma+=fabs(a[i]); sumb+=fabs(b[i]); }
    printf("UNDER GALLOCR [interleaved]:     n=%zu max|chain-fused|=%.3e at %lld (chain=%.6f fused=%.6f) mean|a|=%.4f mean|b|=%.4f\n",
           a.size(), maxabs, (long long)worst, worst>=0?a[worst]:0.f, worst>=0?b[worst]:0.f, suma/a.size(), sumb/b.size());
    double maxabs_ni=0; int64_t worst_ni=-1; double suma_ni=0,sumb_ni=0;
    for (size_t i=0;i<ani.size();++i){ double dd=fabs(ani[i]-bni[i]); if(dd>maxabs_ni){maxabs_ni=dd;worst_ni=(int64_t)i;} suma_ni+=fabs(ani[i]); sumb_ni+=fabs(bni[i]); }
    printf("UNDER GALLOCR [non-interleaved]: n=%zu max|chain-fused|=%.3e at %lld (chain=%.6f fused=%.6f) mean|a|=%.4f mean|b|=%.4f\n",
           ani.size(), maxabs_ni, (long long)worst_ni, worst_ni>=0?ani[worst_ni]:0.f, worst_ni>=0?bni[worst_ni]:0.f, suma_ni/ani.size(), sumb_ni/bni.size());
    return (maxabs < 1e-4 && maxabs_ni < 1e-4) ? 0 : 2;
}
