// Minimal standalone repro for the Pixal3D flash-attn NaN. Calls ggml_flash_attn_ext on the exact
// slat-DiT shapes (D=128, n_head=12, gqa=1, N≈4734, cc8.6) with F16 K/V + F32 Q + a 256-padded F16
// mask, and compares the GPU result to a host CPU reference (same f16-rounded K/V). Prints NaN count
// + max error. Env knobs to BISECT the failing axis:
//   FA_D (head dim, def 128)  FA_H (q heads, def 12)  FA_N (n_q=n_kv, def 4734)
//   FA_GQA (q/kv head ratio, def 1)  FA_PAD (kv pad multiple, def 256)
//   FA_QF16 (1 = cast Q to F16, def 0)  FA_ZEROMASK (1 = all-zero mask, def 0)
//   FA_PREC32 (1 = set GGML_PREC_F32 on the op, def 1)
// Build: ./build.sh fa_repro cuda     Run (sanitizer): compute-sanitizer ./fa_repro
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

static int ienv(const char* k, int d){ const char* s=getenv(k); return s?atoi(s):d; }

int main() {
    const int D   = ienv("FA_D", 128);
    const int H   = ienv("FA_H", 12);
    const int N   = ienv("FA_N", 4734);
    const int GQA = ienv("FA_GQA", 1);
    const int PAD = ienv("FA_PAD", 256);
    const bool QF16 = ienv("FA_QF16", 0);
    const bool ZM   = ienv("FA_ZEROMASK", 0);
    const bool PREC32 = ienv("FA_PREC32", 1);
    const int Hk = H / GQA;
    const int Nkvpad = ((N + PAD - 1) / PAD) * PAD;
    const int Nqpad  = Nkvpad;
    float scale = 1.0f / std::sqrt((float)D);
    const float QKDOWN = getenv("FA_QKDOWN") ? atof(getenv("FA_QKDOWN")) : 1.0f;  // scale q,k by β, scale*=1/β² (result unchanged; tests f16 QK overflow)
    const float VDOWN  = getenv("FA_VDOWN")  ? atof(getenv("FA_VDOWN"))  : 1.0f;  // scale v down (tests PV overflow)
    printf("[fa] D=%d H=%d Hk=%d N=%d gqa=%d pad=%d Nkvpad=%d qF16=%d zeromask=%d prec32=%d\n",
           D, H, Hk, N, GQA, PAD, Nkvpad, (int)QF16, (int)ZM, (int)PREC32);

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { printf("[fa] CUDA init failed; CPU backend (kernel won't repro)\n"); backend = ggml_backend_cpu_init(); }

    ggml_init_params ip{ (size_t)2048*ggml_tensor_overhead() + 4*ggml_graph_overhead(), nullptr, true };
    ggml_context* ctx = ggml_init(ip);

    const bool REAL = ienv("FA_REALPATH", 0);  // replicate attention(): cont(permute)+pad+cast+flash
    ggml_tensor *q, *k, *v, *qfa, *kfa, *vfa;
    if (REAL) {
        // inputs in the pre-attention layout [d_head, n_head, n_token] (== q/k/v in slat_dit_graph)
        q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, H,  N);
        k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, Hk, N);
        v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, Hk, N);
        ggml_set_input(q); ggml_set_input(k); ggml_set_input(v);
        qfa = ggml_cont(ctx, ggml_permute(ctx, q, 0,2,1,3));   // [d, N, head]
        kfa = ggml_cont(ctx, ggml_permute(ctx, k, 0,2,1,3));
        vfa = ggml_cont(ctx, ggml_permute(ctx, v, 0,2,1,3));
        if (N < Nkvpad) { kfa = ggml_pad(ctx, kfa, 0, Nkvpad-N, 0, 0); vfa = ggml_pad(ctx, vfa, 0, Nkvpad-N, 0, 0); }
        kfa = ggml_cast(ctx, kfa, GGML_TYPE_F16);
        vfa = ggml_cast(ctx, vfa, GGML_TYPE_F16);
        if (QF16) qfa = ggml_cast(ctx, qfa, GGML_TYPE_F16);
    } else {
        q = ggml_new_tensor_4d(ctx, QF16?GGML_TYPE_F16:GGML_TYPE_F32, D, N, H, 1);
        k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, Nkvpad, Hk, 1);
        v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, Nkvpad, Hk, 1);
        ggml_set_input(q); ggml_set_input(k); ggml_set_input(v);
        qfa=q; kfa=k; vfa=v;
    }
    ggml_tensor* mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, Nkvpad, Nqpad, 1, 1);
    ggml_set_input(mask);
    // FA_BLOCKS>1: chain B flash ops sharing the SAME mask (mirrors the 30-block DiT). Each block's
    // [d,head,n] output feeds the next block's q (k,v reused). NaN-check only (no CPU ref for B>1).
    const int BLOCKS = ienv("FA_BLOCKS", 1);
    ggml_tensor* r = ggml_flash_attn_ext(ctx, qfa, kfa, vfa, mask, scale, 0.0f, 0.0f);
    if (PREC32) ggml_flash_attn_ext_set_prec(r, GGML_PREC_F32);
    for (int b=1; b<BLOCKS; b++) {
        // r is [d, head, n] (== attention input layout). permute+cont+pad+cast → next flash.
        ggml_tensor* qn = ggml_cont(ctx, ggml_permute(ctx, r, 0,2,1,3));    // [d, n, head]
        r = ggml_flash_attn_ext(ctx, qn, kfa, vfa, mask, scale, 0.0f, 0.0f);
        if (PREC32) ggml_flash_attn_ext_set_prec(r, GGML_PREC_F32);
    }
    ggml_set_output(r);

    // Optionally mirror the harness exactly: inputs+mask in a PERSISTENT buffer (like ctx_w), the
    // graph allocated by ggml_gallocr (reuses intermediate memory), and compute run TWICE (the
    // sampler calls forward() 2×/step → gallocr reuse across recomputes). FA_GALLOC=1 enables.
    const bool GALLOC = ienv("FA_GALLOC", 0);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, r);
    ggml_backend_buffer_t buf = nullptr;
    ggml_gallocr_t galloc = nullptr;
    if (GALLOC) {
        // gallocr allocates the whole graph (inputs kept persistent via ggml_set_input); mirrors
        // the harness's ggml_gallocr_alloc_graph path (intermediate-memory reuse).
        galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(galloc, gf)) { printf("[fa] gallocr alloc failed\n"); return 1; }
    } else {
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) { printf("[fa] alloc failed\n"); return 1; }
    }

    // canonical reference values qv[h][i][d], kv/vv[hk][j][d] (f16-rounded for K/V to match the cast)
    const float MAG = getenv("FA_MAG") ? atof(getenv("FA_MAG")) : 0.3f;   // q/k/v stddev (real data ~1-3)
    std::mt19937 rng(42); std::normal_distribution<float> nd(0.f, MAG);
    std::vector<float> qv((size_t)H*N*D), kf((size_t)Hk*N*D), vf((size_t)Hk*N*D);
    auto QI=[&](int h,int i,int d){ return ((size_t)h*N+i)*D+d; };
    auto KI=[&](int h,int j,int d){ return ((size_t)h*N+j)*D+d; };
    for (int h=0;h<H;h++)  for(int i=0;i<N;i++) for(int d=0;d<D;d++) qv[QI(h,i,d)]=nd(rng);
    for (int h=0;h<Hk;h++) for(int j=0;j<N;j++) for(int d=0;d<D;d++){
        kf[KI(h,j,d)]=ggml_fp16_to_fp32(ggml_fp32_to_fp16(nd(rng)));
        vf[KI(h,j,d)]=ggml_fp16_to_fp32(ggml_fp32_to_fp16(nd(rng))); }
    // FA_LOAD: replay REAL captured q/k/v (cap_{q,k,v}.bin, ggml [D,H,N] = mem idx (i*H+h)*D+d).
    // Requires FA_REALPATH=1 (bins are in the pre-attention [d,head,n] layout). f16-round K/V to
    // match the flash cast. This is the capture-the-failing-latents-and-replay test.
    if (getenv("FA_LOAD")) {
        auto rd=[&](const char* fn){ std::vector<float> b((size_t)D*H*N); FILE* f=fopen(fn,"rb");
            if(!f){printf("[fa] cannot open %s\n",fn);exit(1);} size_t n=fread(b.data(),4,b.size(),f); fclose(f);
            printf("[fa] loaded %s (%zu floats)\n",fn,n); return b; };
        std::vector<float> rq=rd("cap_q.bin"), rk=rd("cap_k.bin"), rv=rd("cap_v.bin");
        for (int h=0;h<H;h++) for(int i=0;i<N;i++) for(int d=0;d<D;d++) qv[QI(h,i,d)]=rq[((size_t)i*H+h)*D+d];
        for (int h=0;h<Hk;h++) for(int j=0;j<N;j++) for(int d=0;d<D;d++){
            kf[KI(h,j,d)]=ggml_fp16_to_fp32(ggml_fp32_to_fp16(rk[((size_t)j*H+h)*D+d]));
            vf[KI(h,j,d)]=ggml_fp16_to_fp32(ggml_fp32_to_fp16(rv[((size_t)j*H+h)*D+d])); }
    }
    if (QKDOWN != 1.0f) { for(auto&x:qv)x*=QKDOWN; for(auto&x:kf)x*=QKDOWN; scale /= (QKDOWN*QKDOWN); }
    if (VDOWN  != 1.0f) { for(auto&x:vf)x*=VDOWN; }

    // pack into the tensors in their actual layouts
    if (REAL) {
        // q [D,H,N]: idx(d,h,i)=(i*H+h)*D+d ; k,v [D,Hk,N]: idx(d,hk,j)=(j*Hk+hk)*D+d (F32, no pad)
        std::vector<float> qp((size_t)D*H*N), kp((size_t)D*Hk*N), vp((size_t)D*Hk*N);
        for(int h=0;h<H;h++) for(int i=0;i<N;i++) for(int d=0;d<D;d++) qp[((size_t)i*H+h)*D+d]=qv[QI(h,i,d)];
        for(int h=0;h<Hk;h++) for(int j=0;j<N;j++) for(int d=0;d<D;d++){ kp[((size_t)j*Hk+h)*D+d]=kf[KI(h,j,d)]; vp[((size_t)j*Hk+h)*D+d]=vf[KI(h,j,d)]; }
        ggml_backend_tensor_set(q, qp.data(),0,ggml_nbytes(q));
        ggml_backend_tensor_set(k, kp.data(),0,ggml_nbytes(k));
        ggml_backend_tensor_set(v, vp.data(),0,ggml_nbytes(v));
    } else {
        // q [D,N,H]: idx(d,i,h)=(h*N+i)*D+d ; k,v [D,Nkvpad,Hk] F16 (pad keys=0)
        std::vector<ggml_fp16_t> qh((size_t)D*N*H), kh((size_t)D*Nkvpad*Hk), vh((size_t)D*Nkvpad*Hk);
        std::vector<float> qf((size_t)D*N*H);
        for(int h=0;h<H;h++) for(int i=0;i<N;i++) for(int d=0;d<D;d++){ float x=qv[QI(h,i,d)]; size_t id=((size_t)h*N+i)*D+d; qf[id]=x; qh[id]=ggml_fp32_to_fp16(x); }
        for(int h=0;h<Hk;h++) for(int j=0;j<Nkvpad;j++) for(int d=0;d<D;d++){ size_t id=((size_t)h*Nkvpad+j)*D+d;
            kh[id]=ggml_fp32_to_fp16(j<N?kf[KI(h,j,d)]:0.f); vh[id]=ggml_fp32_to_fp16(j<N?vf[KI(h,j,d)]:0.f); }
        ggml_backend_tensor_set(q, QF16?(void*)qh.data():(void*)qf.data(),0,ggml_nbytes(q));
        ggml_backend_tensor_set(k, kh.data(),0,ggml_nbytes(k));
        ggml_backend_tensor_set(v, vh.data(),0,ggml_nbytes(v));
    }
    std::vector<ggml_fp16_t> mh((size_t)Nkvpad*Nqpad);
    const ggml_fp16_t M0=ggml_fp32_to_fp16(0.f), MINF=ggml_fp32_to_fp16(-65504.f);
    for(int jq=0;jq<Nqpad;jq++) for(int kk=0;kk<Nkvpad;kk++) mh[(size_t)jq*Nkvpad+kk]=(!ZM&&kk>=N)?MINF:M0;
    ggml_backend_tensor_set(mask, mh.data(),0,ggml_nbytes(mask));

    int ncompute = GALLOC ? 2 : 1;   // recompute to mirror the sampler's 2 forwards/step
    for (int it=0; it<ncompute; it++)
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { printf("[fa] compute FAILED\n"); return 1; }

    std::vector<float> out((size_t)D*H*N);   // r ne = [D, H, N, 1]
    ggml_backend_tensor_get(r, out.data(), 0, ggml_nbytes(r));

    // CPU reference: per (head, query) softmax over Nkvpad keys with the same mask + f16-rounded K/V.
    size_t nan_gpu=0; double maxerr=0, sumerr=0; size_t cmp=0;
    std::vector<float> ref(D);
    // check a sample of queries across all heads (full check is O(N^2 H D) — sample for speed)
    int qstep = N>512 ? N/512 : 1;
    for (int h=0; h<H; h++) {
        int hk = h / GQA;
        for (int i=0; i<N; i+=qstep) {
            // scores
            std::vector<float> sc(N); float mx=-1e30f;
            for (int j=0;j<N;j++){
                float dot=0; for (int d=0;d<D;d++){ float qx = qv[QI(h,i,d)];
                    if (QF16) qx = ggml_fp16_to_fp32(ggml_fp32_to_fp16(qx));
                    dot += qx * kf[KI(hk,j,d)]; }
                sc[j]=dot*scale; if(sc[j]>mx)mx=sc[j];
            }
            float den=0; for(int j=0;j<N;j++){ sc[j]=std::exp(sc[j]-mx); den+=sc[j]; }
            for(int d=0;d<D;d++) ref[d]=0;
            for(int j=0;j<N;j++){ float p=sc[j]/den; for(int d=0;d<D;d++) ref[d]+=p*vf[KI(hk,j,d)]; }
            for(int d=0;d<D;d++){ float g=out[((size_t)i*H+h)*D+d];   // r layout [D,H,N]
                if (std::isnan(g)||std::isinf(g)) nan_gpu++;
                else { double e=std::fabs(g-ref[d]); maxerr=e>maxerr?e:maxerr; sumerr+=e; cmp++; } }
        }
    }
    printf("[fa] GPU NaN/Inf elems (sampled)=%zu  maxerr=%.4e  meanerr=%.4e  (cmp=%zu)\n",
           nan_gpu, maxerr, cmp?sumerr/cmp:0.0, cmp);
    printf("[fa] VERDICT: %s\n", nan_gpu>0 ? "*** NaN/Inf — kernel FAILS on these shapes ***"
           : (maxerr<5e-2 ? "OK (matches CPU ref)" : "MISMATCH (no NaN but wrong values)"));
    ggml_backend_buffer_free(buf); ggml_free(ctx); ggml_backend_free(backend);
    return 0;
}
