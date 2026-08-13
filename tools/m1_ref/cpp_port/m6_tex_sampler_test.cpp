// M6 Stage-4 Texture SLat DiT sampler + denorm from GOLDEN cond (grid 64), CFG OFF.
// Same DiT graph as M3b (build_slat_dit_forward) but in_ch 64: each forward concats
// [tex_noise(32) || RE-normalized golden shape_slat(32)] -> [64,M]; out 32. CFG off
// (GS1.0/GR0.0, interval [0.6,0.9], RT3, 12 steps). tex noise = 4th seed-42 draw (torch_randn).
// Targets: pre-denorm vs torch_tex_slat fp32 (tight) + denorm vs bf16 golden stage4_out (cosine).
// Also validates torch_randn's 4th draw vs refs/tex_noise.npy. Build: ./build.sh m6_tex_sampler_test [cuda]
#include "slat_dit_graph.hpp"
#include "geometry_e2e.hpp"
#include "torch_randn.hpp"
#include <cstdio>
#include <cmath>

static const char* FLOW_W = "weights_npy/slat_flow_imgshape2tex_1024";
static const char* REFS = "refs";
static const char* GOLD = "../../sparse_spike/golden_stages";

static void cmp(const char* tag, const std::vector<float>& a, const std::vector<float>& b) {
    double ma=0,sum=0,dot=0,na=0,nb=0; int n=(int)a.size();
    for (int i=0;i<n;i++){ double d=std::fabs((double)a[i]-b[i]); ma=std::max(ma,d); sum+=d;
        dot+=(double)a[i]*b[i]; na+=(double)a[i]*a[i]; nb+=(double)b[i]*b[i]; }
    printf("  [%s] maxabs=%.3e meanabs=%.3e cosine=%.6f\n", tag, ma, sum/n, dot/(std::sqrt(na*nb)+1e-12));
}

int main(int argc, char** argv) {
    bool use_cuda = (argc > 1 && std::string(argv[1]) == "cuda");
    const float SM=1e-5f, GS=1.0f, GR=0.0f;   // CFG OFF
    const double RT=3.0, IV0=0.6, IV1=0.9;
    const int STEPS=12, INCH=32, IN64=64;

    NpyArray cN = npy_load(std::string(GOLD)+"/stage4_cond/proj_coords.npy");
    int M=(int)cN.shape[0]; const int32_t* c4=cN.i32();
    std::vector<int32_t> coords_xyz((size_t)M*3);
    for (int n=0;n<M;n++) for (int j=0;j<3;j++) coords_xyz[n*3+j]=c4[n*4+1+j];
    const int NEL = M*INCH;

    // re-normalized golden shape_slat (concat_cond): (shape_slat - mean)/std
    NpyArray ssN = npy_load(std::string(GOLD)+"/stage3b_out/shape_slat_feats.npy");  // [M,32] denorm
    NpyArray smN = npy_load(std::string(REFS)+"/shape_slat_norm_mean.npy");
    NpyArray sdN = npy_load(std::string(REFS)+"/shape_slat_norm_std.npy");
    std::vector<float> shape_renorm((size_t)NEL);
    for (int i=0;i<NEL;i++){ int c=i%INCH; shape_renorm[i]=(ssN.f32()[i]-smN.f32()[c])/sdN.f32()[c]; }

    // tex noise = 4th seed-42 draw; validate vs refs/tex_noise.npy
    trandn::Generator gen(42);
    gen.randn((int64_t)1*8*16*16*16);                 // stage1
    int N2=(int)npy_load(std::string(GOLD)+"/stage2_cond/proj_coords.npy").shape[0];
    gen.randn((int64_t)N2*32);                          // stage2
    gen.randn((int64_t)M*32);                           // stage3b
    std::vector<float> tex_noise = gen.randn((int64_t)M*32);  // tex (4th)
    { NpyArray tn=npy_load(std::string(REFS)+"/tex_noise.npy"); double ma=0;
      for (int i=0;i<NEL;i++) ma=std::max(ma,(double)std::fabs(tex_noise[i]-tn.f32()[i]));
      printf("[m6tex] tex_noise (4th seed-42 draw) vs ref: maxabs=%.3e %s\n", ma, ma<5e-6?"OK":"**CHECK**"); }

    // DiT graph (in_ch 64)
    M1Harness Hf(FLOW_W, 2048, use_cuda);
    ggml_context* cf = Hf.ctx;
    int64_t x_ne[4]={IN64,M,1,1};                ggml_tensor* xin=Hf.input("x",2,x_ne);
    int64_t t_ne[4]={1,1,1,1};                   ggml_tensor* tin=Hf.input("t",1,t_ne);
    int64_t g_ne[4]={1024,5,1,1};                ggml_tensor* gin=Hf.input("global",2,g_ne);
    int64_t p_ne[4]={slatdit::PROJ_IN,M,1,1};    ggml_tensor* pin=Hf.input("proj",2,p_ne);
    ggml_tensor* vout = slatdit::build_slat_dit_forward(cf, Hf, M, xin, tin, gin, pin, coords_xyz.data());
    ggml_set_output(vout);
    ggml_cgraph* gff = new_graph(cf, 32768); ggml_build_forward_expand(gff, vout);
    Hf.alloc_and_upload(gff);

    NpyArray gN = npy_load(std::string(GOLD)+"/stage4_cond/global.npy");
    NpyArray pN = npy_load(std::string(GOLD)+"/stage4_cond/proj_feats.npy");
    std::vector<float> cond_g(gN.f32(), gN.f32()+gN.numel());
    std::vector<float> cond_p(pN.f32(), pN.f32()+pN.numel());

    // forward over x_32: build x_64 = [x_32 || shape_renorm] per voxel, upload, compute -> v_32
    std::vector<float> x64((size_t)M*IN64);
    auto forward = [&](const std::vector<float>& x32, float ts, bool /*cond always true: CFG off*/){
        for (int n=0;n<M;n++){ for (int c=0;c<INCH;c++) x64[(size_t)n*IN64+c]=x32[(size_t)n*INCH+c];
                               for (int c=0;c<INCH;c++) x64[(size_t)n*IN64+INCH+c]=shape_renorm[(size_t)n*INCH+c]; }
        Hf.upload_input_raw(xin,x64); std::vector<float> tv{ts}; Hf.upload_input_raw(tin,tv);
        Hf.upload_input_raw(gin,cond_g); Hf.upload_input_raw(pin,cond_p);
        Hf.compute(gff); std::vector<float> v(NEL); ggml_backend_tensor_get(vout,v.data(),0,NEL*4); return v; };

    std::vector<float> tex_raw = geo::flow_sampler(NEL, tex_noise, SM, GS, GR, RT, IV0, IV1, STEPS, forward, "m6tex");

    // denorm with tex_slat_normalization
    NpyArray tmN=npy_load(std::string(REFS)+"/tex_slat_norm_mean.npy");
    NpyArray tsN=npy_load(std::string(REFS)+"/tex_slat_norm_std.npy");
    std::vector<float> tex_den(NEL);
    for (int i=0;i<NEL;i++){ int c=i%INCH; tex_den[i]=tex_raw[i]*tsN.f32()[c]+tmN.f32()[c]; }

    printf("[m6tex] backend=%s M=%d\n", use_cuda?"cuda":"cpu", M);
    NpyArray orefN=npy_load(std::string(REFS)+"/torch_tex_slat_fp32.npy");
    NpyArray odenN=npy_load(std::string(REFS)+"/torch_tex_slat_denorm_fp32.npy");
    NpyArray goldN=npy_load(std::string(GOLD)+"/stage4_out/tex_slat_feats.npy");
    std::vector<float> oref(orefN.f32(),orefN.f32()+NEL), oden(odenN.f32(),odenN.f32()+NEL), gold(goldN.f32(),goldN.f32()+NEL);
    cmp("pre-denorm vs torch_fp32 (TIGHT)", tex_raw, oref);
    cmp("denorm    vs torch_fp32 (TIGHT)", tex_den, oden);
    cmp("denorm    vs golden bf16     ", tex_den, gold);

    double ma=0; for (int i=0;i<NEL;i++) ma=std::max(ma,(double)std::fabs(tex_raw[i]-oref[i]));
    double dot=0,na=0,nb=0; for (int i=0;i<NEL;i++){dot+=(double)tex_den[i]*gold[i]; na+=(double)tex_den[i]*tex_den[i]; nb+=(double)gold[i]*gold[i];}
    double cosg=dot/(std::sqrt(na*nb)+1e-12);
    bool ok=(ma<(use_cuda?3e-1:5e-3)) && cosg>0.99;
    printf("[m6tex] %s (pre-denorm maxabs vs fp32=%.3e, cosine vs golden=%.6f)\n", ok?"PASS":"FAIL", ma, cosg);
    return ok?0:1;
}
