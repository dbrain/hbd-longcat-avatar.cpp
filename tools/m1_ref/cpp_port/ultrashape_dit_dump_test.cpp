// Localize the e2e DiT divergence: run the UNCOND t=0 forward (cond=0) and compare every intermediate
// (rope, x_embedder, time token, each block output) to the banked golden (capture_dit_e2e_dump.py).
// First stage whose cosine collapses = the culprit.   ./build.sh ultrashape_dit_dump_test [cuda]
#include "ultrashape_dit.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>

static const char* GD  = "/mnt/hdd/3d/avatar-shootout/e2e_goldens";
static const char* DD  = "/mnt/hdd/3d/avatar-shootout/dit_dump_goldens";
static const char* WDIR = "/mnt/hdd/3d/avatar-shootout/ultrashape_goldens/weights_npy/dit";

static void cmp(const char* tag, const float* a, const float* b, int64_t n) {
    double dot=0,na=0,nb=0,s=0; for (int64_t i=0;i<n;i++){ dot+=(double)a[i]*b[i]; na+=(double)a[i]*a[i]; nb+=(double)b[i]*b[i]; s+=std::fabs((double)a[i]-b[i]); }
    double cc=dot/(std::sqrt(na)*std::sqrt(nb)+1e-30);
    printf("  [%-10s] cosine=%.6f meanabs=%.3e  |mine|=%.4f |gold|=%.4f %s\n",
           tag, cc, s/n, std::sqrt(na/n), std::sqrt(nb/n), cc<0.999?"  <-- DIVERGES":"");
}

int main(int argc, char** argv) {
    bool use_cuda = (argc>1 && std::string(argv[1])=="cuda");
    setenv("PIXAL3D_DIT_DUMP", "1", 1);
    UsDitCfg cfg;
    M1Harness H(WDIR, 8192, use_cuda);
    ggml_context* ctx = H.ctx;

    NpyArray xin = npy_load(std::string(GD)+"/init_latents.npy");  int64_t N=xin.shape[1];
    NpyArray vox = npy_load(std::string(GD)+"/voxel_cond.npy");   // f32 (capture save()s float)
    std::vector<float> voxf(vox.f32(), vox.f32() + vox.numel());
    NpyArray unc = npy_load(std::string(GD)+"/uncond_main.npy");  int64_t Tc=unc.shape[1];
    std::vector<float> uncond_main(unc.f32(), unc.f32()+unc.numel());
    std::vector<float> xv(xin.f32(), xin.f32()+xin.numel());

    // --- rope host-check vs golden (voxel tokens only) ---
    std::vector<float> rcos, rsin; us_rope_3d(voxf.data(), N, cfg, rcos, rsin);
    int dim=cfg.head_dim();
    { NpyArray gc=npy_load(std::string(DD)+"/rope_cos.npy"); NpyArray gs=npy_load(std::string(DD)+"/rope_sin.npy");
      cmp("rope_cos", rcos.data()+dim, gc.f32(), (int64_t)N*dim);   // skip token-0 identity
      cmp("rope_sin", rsin.data()+dim, gs.f32(), (int64_t)N*dim); }

    int64_t S=1+N;
    int64_t rope_ne[4]={dim,1,S,1};
    ggml_tensor* rcos_t=H.const_tensor("rope_cos",3,rope_ne,rcos);
    ggml_tensor* rsin_t=H.const_tensor("rope_sin",3,rope_ne,rsin);
    int64_t ts_ne[4]={cfg.hidden,1,1,1}; ggml_tensor* ts_t=H.input("ts_embed",2,ts_ne);
    int64_t x_ne[4]={cfg.in_channels,N,1,1}; ggml_tensor* x_lat=H.input("x_lat",2,x_ne);
    int64_t c_ne[4]={cfg.context_dim,Tc,1,1}; ggml_tensor* cond=H.input("cond",2,c_ne);
    ggml_tensor* out=us_refine_dit(H,ctx,cfg,x_lat,ts_t,cond,rcos_t,rsin_t);
    ggml_set_output(out);
    ggml_cgraph* gf=new_graph(ctx,65536);
    ggml_build_forward_expand(gf,out);
    H.alloc_and_upload(gf);
    std::vector<float> ts=us_timesteps_embed(0.0f,cfg.hidden);
    H.upload_input_raw(ts_t,ts); H.upload_input_raw(x_lat,xv); H.upload_input_raw(cond,uncond_main);
    H.compute(gf);

    printf("[dump] backend=%s N=%lld Tc=%lld (uncond t=0)\n", use_cuda?"cuda":"cpu",(long long)N,(long long)Tc);
    auto getd=[&](const char* nm)->ggml_tensor*{ return ggml_graph_get_tensor(gf,nm); };
    auto cmpd=[&](const char* dbg, const char* gold){
        ggml_tensor* t=getd(dbg); if(!t){ printf("  [%s] MISSING\n",dbg); return; }
        std::vector<float> buf(ggml_nelements(t));
        ggml_backend_tensor_get(t, buf.data(), 0, buf.size()*sizeof(float));
        NpyArray g=npy_load(std::string(DD)+"/"+gold+".npy");
        if ((int64_t)buf.size()!=g.numel()){ printf("  [%s] SIZE mine=%zu gold=%lld\n",dbg,buf.size(),(long long)g.numel()); return; }
        cmp(gold, buf.data(), g.f32(), g.numel());
    };
    cmpd("dbg_c","time_token");
    cmpd("dbg_xe","xe");
    for (int i=0;i<cfg.depth;i++){ char d[24],g[24]; snprintf(d,24,"dbg_blk%d",i); snprintf(g,24,"blk%02d",i); cmpd(d,g); }
    { std::vector<float> o(ggml_nelements(out)); ggml_backend_tensor_get(out,o.data(),0,o.size()*sizeof(float));
      NpyArray g=npy_load(std::string(DD)+"/pred.npy"); cmp("pred", o.data(), g.f32(), g.numel()); }
    return 0;
}
