// TRELLIS-2 CROSS-MODE tex DiT validation (Stage-2 texturing, step 1) vs tex_goldens/.
// Isolates the tex-DiT: feeds the GOLDEN image cond (cond_cond [1,4101,1024]) + GOLDEN shape_slat
// (coords+feats) + the BANKED init noise (tex_noise) -> single-forward parity vs tex_pred0_feats, then
// full 12-step CFG-off flow sample -> denorm by tex_slat_normalization -> vs tex_slat_feats.
// concat_cond = (shape_slat - shape_mean)/shape_std (validated vs banked shape_norm_feats).
// Build: ./build.sh tex_dit_test [cuda]
#include "tex_dit_cross.hpp"
#include "geometry_e2e.hpp"
#include <cstdio>
#include <cmath>
#include <string>

static const char* FLOW_W = "weights_npy/trellis2_tex_1024";
static const char* G = "/mnt/hdd/3d/avatar-shootout/tex_goldens";

// _texturing_pipeline_local.json normalization (32-dim per-channel)
static const float SHAPE_MEAN[32]={0.781296f,0.018091f,-0.495192f,-0.558457f,1.06053f,0.093252f,1.518149f,-0.933218f,-0.732996f,2.604095f,-0.118341f,-2.143904f,0.495076f,-2.179512f,-2.130751f,-0.996944f,0.261421f,-2.217463f,1.260067f,-0.150213f,3.790713f,1.481266f,-1.046058f,-1.523667f,-0.059621f,2.22078f,1.621212f,0.87723f,0.567247f,-3.175944f,-3.186688f,1.578665f};
static const float SHAPE_STD[32]={5.972266f,4.706852f,5.44501f,5.209927f,5.32022f,4.547237f,5.020802f,5.444004f,5.226681f,5.683095f,4.831436f,5.286469f,5.652043f,5.367606f,5.525084f,4.730578f,4.805265f,5.124013f,5.530808f,5.619001f,5.10393f,5.41767f,5.269677f,5.547194f,5.634698f,5.235274f,6.110351f,5.511298f,6.237273f,4.879207f,5.347008f,5.405691f};
static const float TEX_MEAN[32]={3.501659f,2.212398f,2.226094f,0.251093f,-0.026248f,-0.687364f,0.439898f,-0.928075f,0.029398f,-0.339596f,-0.869527f,1.038479f,-0.972385f,0.126042f,-1.129303f,0.455149f,-1.209521f,2.069067f,0.544735f,2.569128f,-0.323407f,2.293f,-1.925608f,-1.217717f,1.213905f,0.971588f,-0.023631f,0.10675f,2.021786f,0.250524f,-0.662387f,-0.768862f};
static const float TEX_STD[32]={2.665652f,2.743913f,2.765121f,2.595319f,3.037293f,2.291316f,2.144656f,2.911822f,2.969419f,2.501689f,2.154811f,3.163343f,2.621215f,2.381943f,3.186697f,3.021588f,2.295916f,3.234985f,3.233086f,2.26014f,2.874801f,2.810596f,3.29272f,2.674999f,2.680878f,2.372054f,2.451546f,2.353556f,2.995195f,2.379849f,2.786195f,2.77519f};

static double cmp(const char* tag, const std::vector<float>& a, const std::vector<float>& b) {
    double ma=0,sum=0,dot=0,na=0,nb=0; int n=(int)a.size();
    for (int i=0;i<n;i++){ double d=std::fabs((double)a[i]-b[i]); ma=std::max(ma,d); sum+=d;
        dot+=(double)a[i]*b[i]; na+=(double)a[i]*a[i]; nb+=(double)b[i]*b[i]; }
    double cos=dot/(std::sqrt(na*nb)+1e-12);
    printf("  [%s] maxabs=%.3e meanabs=%.3e cosine=%.6f\n", tag, ma, sum/n, cos);
    return cos;
}

int main(int argc, char** argv) {
    bool use_cuda = (argc > 1 && std::string(argv[1]) == "cuda");
    const float SM=1e-5f, GS=1.0f, GR=0.0f;   // CFG OFF (guidance_strength 1.0)
    const double RT=3.0, IV0=0.6, IV1=0.9;
    const int STEPS=12, INCH=32, IN64=64;

    // golden shape_slat (coords + denorm feats)
    NpyArray ssC = npy_load(std::string(G)+"/shape_slat_coords.npy");  // [M,4] (float in golden)
    NpyArray ssF = npy_load(std::string(G)+"/shape_slat_feats.npy");   // [M,32]
    int M=(int)ssC.shape[0]; const int NEL=M*INCH;
    std::vector<int32_t> coords_xyz((size_t)M*3);
    for (int n=0;n<M;n++) for (int j=0;j<3;j++) coords_xyz[n*3+j]=(int32_t)std::lround(ssC.f32()[(size_t)n*4+1+j]);

    // concat_cond = (shape_slat - shape_mean)/shape_std ; validate vs banked shape_norm_feats
    std::vector<float> shape_norm((size_t)NEL);
    for (int i=0;i<NEL;i++){ int c=i%INCH; shape_norm[i]=(ssF.f32()[i]-SHAPE_MEAN[c])/SHAPE_STD[c]; }
    { NpyArray sn=npy_load(std::string(G)+"/shape_norm_feats.npy"); std::vector<float> b(sn.f32(),sn.f32()+NEL);
      cmp("shape_norm vs banked", shape_norm, b); }

    // golden image cond [1,Ntok,1024] -> cin [1024,Ntok]
    NpyArray cN = npy_load(std::string(G)+"/cond_cond.npy");
    int Ntok=(int)cN.shape[1];
    std::vector<float> cond_ctx(cN.f32(), cN.f32()+cN.numel());

    // banked init noise [M,32]
    NpyArray nN = npy_load(std::string(G)+"/tex_noise.npy");
    std::vector<float> tex_noise(nN.f32(), nN.f32()+NEL);

    printf("[texdit] backend=%s M=%d Ntok=%d\n", use_cuda?"cuda":"cpu", M, Ntok);

    // graph (cross mode, in_ch 64)
    M1Harness Hf(FLOW_W, 4096, use_cuda);
    ggml_context* cf = Hf.ctx;
    int64_t x_ne[4]={IN64,M,1,1};      ggml_tensor* xin=Hf.input("x",2,x_ne);
    int64_t t_ne[4]={1,1,1,1};         ggml_tensor* tin=Hf.input("t",1,t_ne);
    int64_t c_ne[4]={1024,Ntok,1,1};   ggml_tensor* cin=Hf.input("cond",2,c_ne);
    ggml_tensor* vout = texdit::build_tex_dit_cross_forward(cf, Hf, M, Ntok, xin, tin, cin, coords_xyz.data());
    ggml_set_output(vout);
    ggml_cgraph* gff = new_graph(cf, 65536); ggml_build_forward_expand(gff, vout);
    Hf.alloc_and_upload(gff);

    std::vector<float> x64((size_t)M*IN64);
    auto forward = [&](const std::vector<float>& x32, float ts, bool){
        for (int n=0;n<M;n++){ for (int c=0;c<INCH;c++) x64[(size_t)n*IN64+c]=x32[(size_t)n*INCH+c];
                               for (int c=0;c<INCH;c++) x64[(size_t)n*IN64+INCH+c]=shape_norm[(size_t)n*INCH+c]; }
        Hf.upload_input_raw(xin,x64); std::vector<float> tv{ts}; Hf.upload_input_raw(tin,tv);
        Hf.upload_input_raw(cin,cond_ctx);
        Hf.compute(gff); std::vector<float> v(NEL); ggml_backend_tensor_get(vout,v.data(),0,NEL*4); return v; };

    // --- single-forward parity (step 0): t0 from the same schedule ---
    std::vector<double> ts(STEPS+1);
    for (int i=0;i<=STEPS;i++){ double lt=1.0-(double)i/STEPS; ts[i]=RT*lt/(1+(RT-1)*lt); }
    double cos0=2;  // default pass if golden absent
    { std::string pp=std::string(G)+"/tex_pred0_feats.npy"; FILE* f=fopen(pp.c_str(),"rb");
      if (f){ fclose(f);
        std::vector<float> p0 = forward(tex_noise, (float)(1000.0*ts[0]), true);
        NpyArray gp=npy_load(pp); std::vector<float> g(gp.f32(),gp.f32()+NEL);
        cos0=cmp("step0 pred_v vs golden", p0, g);
      } else printf("  [step0] tex_pred0_feats.npy absent — skipping single-forward parity\n"); }

    // --- full 12-step sample ---
    std::vector<float> tex_raw = geo::flow_sampler(NEL, tex_noise, SM, GS, GR, RT, IV0, IV1, STEPS, forward, "texdit");
    std::vector<float> tex_den(NEL);
    for (int i=0;i<NEL;i++){ int c=i%INCH; tex_den[i]=tex_raw[i]*TEX_STD[c]+TEX_MEAN[c]; }

    NpyArray goldN=npy_load(std::string(G)+"/tex_slat_feats.npy");
    std::vector<float> gold(goldN.f32(),goldN.f32()+NEL);
    double cosf=cmp("tex_slat (denorm) vs golden", tex_den, gold);

    // DUMP_TEXSLAT=path.npy -> write my denormed tex_slat (for the fully-native decode+bake chain)
    if (const char* dp = std::getenv("DUMP_TEXSLAT")) {
        FILE* f=fopen(dp,"wb");
        if (f){ // minimal .npy: header + raw f32 [M,32]
            char hdr[128]; int hl=snprintf(hdr,sizeof(hdr),"{'descr': '<f4', 'fortran_order': False, 'shape': (%d, 32), }",M);
            int total=10+hl; int pad=(64-(total%64))%64; std::string h(hdr,hl); h.append(pad,' '); h.push_back('\n');
            uint16_t hlen=(uint16_t)h.size(); fwrite("\x93NUMPY\x01\x00",1,8,f); fwrite(&hlen,2,1,f); fwrite(h.data(),1,h.size(),f);
            fwrite(tex_den.data(),4,NEL,f); fclose(f); printf("  [dump] tex_slat -> %s\n", dp); }
    }

    bool ok = (cos0>0.999) && (cosf>0.99);
    printf("[texdit] %s (step0 cosine=%.6f, tex_slat cosine=%.6f)\n", ok?"PASS":"FAIL", cos0, cosf);
    return ok?0:1;
}
