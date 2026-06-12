// Rung-1 DELIVERABLE: full Stage-1 image -> occupancy coords, entirely in C++/ggml.
//   image -> DINOv3 ViT-L/16 -> proj grid_sample cond -> SS dense DiT (12-step
//   FlowEulerGuidanceIntervalSampler) -> SS VAE decode -> coords.
// Reuses the per-stage-validated graph builders. cond is produced by the C++ DINOv3
// (== golden cond @1.5e-5), so the result matches the torch fp32 closure:
//   coords N=1120, IoU 0.9859 vs the bf16 golden (1126).
#include "dinov3_graph.hpp"
#include "ss_dit_graph.hpp"
#include "ss_vae_graph.hpp"
#include <cmath>

static const char* DINO_W = "weights_npy/dinov3";
static const char* FLOW_W = "weights_npy/ss_flow";
static const char* DEC_W = "weights_npy/ss_dec";
static const char* REFS = "refs";
static const char* GOLD = "../../sparse_spike/golden_stages";
static const int NEL = ssdit::SEQ * ssdit::INCH;

// camera scalars from golden_stages/cam.json (MoGe, miku decode)
static const float CAM = 0.7332379387484828f, DIST = 1.3021559715270996f, MS = 1.0f;

static void inv4x4(const float A[16], float out[16]) {
    float m[4][8];
    for (int r = 0; r < 4; r++) { for (int c = 0; c < 4; c++) m[r][c] = A[r*4+c]; for (int c = 0; c < 4; c++) m[r][4+c] = (r==c)?1.f:0.f; }
    for (int col = 0; col < 4; col++) {
        int piv = col; for (int r = col+1; r < 4; r++) if (std::fabs(m[r][col]) > std::fabs(m[piv][col])) piv = r;
        for (int c = 0; c < 8; c++) std::swap(m[col][c], m[piv][c]);
        float d = m[col][col]; for (int c = 0; c < 8; c++) m[col][c] /= d;
        for (int r = 0; r < 4; r++) { if (r==col) continue; float f = m[r][col]; for (int c = 0; c < 8; c++) m[r][c] -= f*m[col][c]; }
    }
    for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) out[r*4+c] = m[r][4+c];
}

static double vstd(const std::vector<float>& x) {
    double m = 0; for (float v : x) m += v; m /= x.size();
    double s = 0; for (float v : x) s += (v-m)*(v-m); return std::sqrt(s / x.size());
}

int main(int argc, char** argv) {
    bool use_cuda = (argc > 1 && std::string(argv[1]) == "cuda");
    printf("[stage1_e2e] backend=%s\n", use_cuda ? "cuda" : "cpu");

    // ========== (1) DINOv3 + proj grid_sample -> cond ==========
    std::vector<float> cond_g, cond_p;
    {
        M1Harness H(DINO_W, 256, use_cuda);
        ggml_context* ctx = H.ctx;
        int64_t img_ne[4] = {dino::IMG, dino::IMG, 3, 1};
        ggml_tensor* img = H.input("image", 4, img_ne);
        int64_t cs_ne[4] = {dino::HEAD_DIM, 1, dino::NP, 1};
        ggml_tensor* cosT = H.input("rope_cos", 3, cs_ne), *sinT = H.input("rope_sin", 3, cs_ne);
        ggml_tensor *global = nullptr, *patchmap = nullptr;
        dino::build_dinov3(ctx, H, img, cosT, sinT, &global, &patchmap);

        // proj: host corner idx + weights, then get_rows gather (== proj_grid_test)
        const int R = 16, img_res = 512, W = 32, Hh = 32, N = R*R*R;
        float T[16] = {1,0,0,0, 0,0,-1,-DIST, 0,1,0,0, 0,0,0,1}, Winv[16]; inv4x4(T, Winv);
        const float focal = 16.f/std::tan(CAM/2.f), focal_px = focal*img_res/32.f;
        auto lin1 = [&](int i){ return -1.f + 2.f*i/(R-1); };
        std::vector<int32_t> i00(N),i01(N),i10(N),i11(N);
        std::vector<float> w00(N),w01(N),w10(N),w11(N);
        int tt = 0;
        for (int i=0;i<R;i++) for (int j=0;j<R;j++) for (int k=0;k<R;k++) {
            float x=lin1(i),y=lin1(j),z=lin1(k);
            float gx=x/MS/2.f, gy=(-z)/MS/2.f, gz=(y)/MS/2.f;
            float xc=Winv[0]*gx+Winv[1]*gy+Winv[2]*gz+Winv[3];
            float yc=Winv[4]*gx+Winv[5]*gy+Winv[6]*gz+Winv[7];
            float zc=Winv[8]*gx+Winv[9]*gy+Winv[10]*gz+Winv[11];
            float xn=focal_px*xc/(-zc+1e-8f), yn=focal_px*yc/(-zc+1e-8f);
            float xp=xn+img_res/2.f, yp=-yn+img_res/2.f;
            float gnx=(xp+0.5f)/img_res*2.f-1.f, gny=(yp+0.5f)/img_res*2.f-1.f;
            float ix=(gnx+1.f)*0.5f*W-0.5f, iy=(gny+1.f)*0.5f*Hh-0.5f;
            int x0=(int)std::floor(ix),y0=(int)std::floor(iy),x1=x0+1,y1=y0+1;
            float wx1=ix-x0,wy1=iy-y0,wx0=1.f-wx1,wy0=1.f-wy1;
            auto cl=[](int v,int hi){return v<0?0:(v>hi?hi:v);};
            int x0c=cl(x0,W-1),x1c=cl(x1,W-1),y0c=cl(y0,Hh-1),y1c=cl(y1,Hh-1);
            i00[tt]=y0c*W+x0c; w00[tt]=wy0*wx0; i01[tt]=y0c*W+x1c; w01[tt]=wy0*wx1;
            i10[tt]=y1c*W+x0c; w10[tt]=wy1*wx0; i11[tt]=y1c*W+x1c; w11[tt]=wy1*wx1; tt++;
        }
        auto mki=[&](const char*nm){ auto t=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,N); ggml_set_name(t,nm); ggml_set_input(t); return t; };
        auto mkw=[&](const char*nm){ int64_t e[4]={1,N,1,1}; auto t=ggml_new_tensor(ctx,GGML_TYPE_F32,2,e); ggml_set_name(t,nm); ggml_set_input(t); return t; };
        ggml_tensor *ti00=mki("i00"),*ti01=mki("i01"),*ti10=mki("i10"),*ti11=mki("i11");
        ggml_tensor *ta00=mkw("a00"),*ta01=mkw("a01"),*ta10=mkw("a10"),*ta11=mkw("a11");
        ggml_tensor* fmap = ggml_reshape_2d(ctx, patchmap, dino::HID, (int64_t)Hh*W);
        auto gat=[&](ggml_tensor*idx,ggml_tensor*wg){ return ggml_mul(ctx, ggml_get_rows(ctx,fmap,idx), wg); };
        ggml_tensor* proj = ggml_add(ctx, ggml_add(ctx,gat(ti00,ta00),gat(ti01,ta01)), ggml_add(ctx,gat(ti10,ta10),gat(ti11,ta11)));
        proj = ggml_cont(ctx, proj);  // [1024,4096]
        ggml_set_output(proj);

        ggml_cgraph* gf = new_graph(ctx, 16384);
        ggml_build_forward_expand(gf, global);
        ggml_build_forward_expand(gf, proj);
        H.alloc_and_upload(gf);
        H.upload_input_npy(img, std::string(REFS) + "/image_chw.npy");
        std::vector<float> cosb, sinb; dino::rope_cos_sin(cosb, sinb);
        H.upload_input_raw(cosT, cosb); H.upload_input_raw(sinT, sinb);
        ggml_backend_tensor_set(ti00,i00.data(),0,N*4); ggml_backend_tensor_set(ti01,i01.data(),0,N*4);
        ggml_backend_tensor_set(ti10,i10.data(),0,N*4); ggml_backend_tensor_set(ti11,i11.data(),0,N*4);
        H.upload_input_raw(ta00,w00); H.upload_input_raw(ta01,w01); H.upload_input_raw(ta10,w10); H.upload_input_raw(ta11,w11);
        H.compute(gf);
        cond_g.resize(ggml_nelements(global)); cond_p.resize(ggml_nelements(proj));
        ggml_backend_tensor_get(global, cond_g.data(), 0, cond_g.size()*4);
        ggml_backend_tensor_get(proj, cond_p.data(), 0, cond_p.size()*4);
        // sanity vs golden cond
        NpyArray gg = npy_load(std::string(REFS)+"/dino_global.npy"), gp = npy_load(std::string(REFS)+"/proj.npy");
        double dg=0,dp=0; for (size_t i=0;i<cond_g.size();i++) dg=std::max(dg,(double)std::fabs(cond_g[i]-gg.f32()[i]));
        for (size_t i=0;i<cond_p.size();i++) dp=std::max(dp,(double)std::fabs(cond_p[i]-gp.f32()[i]));
        printf("  [cond] global maxabs vs golden=%.3e  proj=%.3e\n", dg, dp);
    }

    // ========== (2) SS DiT 12-step sampler ==========
    std::vector<float> z_s(NEL);
    {
        const float SM=1e-5f, GS=7.5f, GR=0.7f, RT=5.0f, IV0=0.6f, IV1=1.0f; const int STEPS=12;
        M1Harness Hf(FLOW_W, 512, use_cuda);
        ggml_context* cf = Hf.ctx;
        int64_t x_ne[4]={ssdit::SEQ,ssdit::INCH,1,1}; ggml_tensor* xin=Hf.input("x",2,x_ne);
        int64_t t_ne[4]={1,1,1,1}; ggml_tensor* tin=Hf.input("t",1,t_ne);
        int64_t g_ne[4]={1024,5,1,1}; ggml_tensor* gin=Hf.input("global",2,g_ne);
        int64_t p_ne[4]={1024,ssdit::SEQ,1,1}; ggml_tensor* pin=Hf.input("proj",2,p_ne);
        ggml_tensor* vout = ssdit::build_ss_dit_forward(cf, Hf, xin, tin, gin, pin);
        ggml_set_output(vout);
        ggml_cgraph* gff = new_graph(cf, 32768); ggml_build_forward_expand(gff, vout);
        Hf.alloc_and_upload(gff);
        std::vector<float> zero_g(cond_g.size(),0.f), zero_p(cond_p.size(),0.f);
        auto forward=[&](const std::vector<float>&xx,float ts,bool cond){
            Hf.upload_input_raw(xin,xx); std::vector<float> tv{ts}; Hf.upload_input_raw(tin,tv);
            Hf.upload_input_raw(gin,cond?cond_g:zero_g); Hf.upload_input_raw(pin,cond?cond_p:zero_p);
            Hf.compute(gff); std::vector<float> v(NEL); ggml_backend_tensor_get(vout,v.data(),0,NEL*4); return v;
        };
        auto p2x=[&](const std::vector<float>&xt,float t,const std::vector<float>&pr){ std::vector<float> o(NEL); float a=1-SM,b=SM+(1-SM)*t; for(int i=0;i<NEL;i++)o[i]=a*xt[i]-b*pr[i]; return o; };
        auto x2p=[&](const std::vector<float>&xt,float t,const std::vector<float>&x0){ std::vector<float> o(NEL); float a=1-SM,b=SM+(1-SM)*t; for(int i=0;i<NEL;i++)o[i]=(a*xt[i]-x0[i])/b; return o; };
        NpyArray noiseN = npy_load(std::string(REFS)+"/noise_seed42.npy");
        std::vector<float> x(noiseN.f32(), noiseN.f32()+NEL);
        std::vector<float> tseq(STEPS+1);
        for (int i=0;i<=STEPS;i++){ float lt=1.f-(float)i/STEPS; tseq[i]=RT*lt/(1+(RT-1)*lt); }
        for (int s=0;s<STEPS;s++){
            float t=tseq[s],tp=tseq[s+1]; std::vector<float> v;
            if (t>=IV0&&t<=IV1){
                auto vp=forward(x,1000.f*t,true), vn=forward(x,1000.f*t,false);
                std::vector<float> pred(NEL); for(int i=0;i<NEL;i++)pred[i]=GS*vp[i]+(1-GS)*vn[i];
                auto x0p=p2x(x,t,vp), x0c=p2x(x,t,pred); double r=vstd(x0p)/vstd(x0c);
                std::vector<float> x0(NEL); for(int i=0;i<NEL;i++){float rc=x0c[i]*(float)r; x0[i]=GR*rc+(1-GR)*x0c[i];}
                v=x2p(x,t,x0);
            } else v=forward(x,1000.f*t,true);
            for(int i=0;i<NEL;i++)x[i]-=(t-tp)*v[i];
            printf("  step %2d/%d t=%.4f %s\n",s+1,STEPS,t,(t>=IV0&&t<=IV1)?"[cfg]":"[cond]"); fflush(stdout);
        }
        z_s = x;
        NpyArray gold = npy_load(std::string(GOLD)+"/stage1_ssdec/z_s.npy");
        double ma=0,sum=0; for(int i=0;i<NEL;i++){double d=std::fabs(z_s[i]-gold.f32()[i]);ma=std::max(ma,d);sum+=d;}
        printf("  z_s vs golden(bf16): maxabs=%.3e meanabs=%.3e (expect ~1.2 / ~3.6e-3)\n", ma, sum/NEL);
        std::ifstream tf(std::string(REFS)+"/torch_z_s_fp32.npy");
        if (tf.good()){ NpyArray tz=npy_load(std::string(REFS)+"/torch_z_s_fp32.npy"); double m2=0,s2=0; for(int i=0;i<NEL;i++){double d=std::fabs(z_s[i]-tz.f32()[i]);m2=std::max(m2,d);s2+=d;} printf("  z_s vs torch_fp32: maxabs=%.3e meanabs=%.3e\n",m2,s2/NEL); }
    }

    // ========== (3) SS VAE decode -> coords ==========
    {
        M1Harness Hd(DEC_W, 256, use_cuda);
        ggml_context* cd = Hd.ctx;
        int64_t z_ne[4]={16,16,16,8}; ggml_tensor* zin=Hd.input("z_s",4,z_ne);
        ggml_tensor* logits = ssvae::build_ss_vae_decode(cd, Hd, zin);
        ggml_set_output(logits);
        ggml_cgraph* gfd = new_graph(cd, 8192); ggml_build_forward_expand(gfd, logits);
        Hd.alloc_and_upload(gfd); Hd.upload_input_raw(zin, z_s); Hd.compute(gfd);
        std::vector<float> L((size_t)64*64*64); ggml_backend_tensor_get(logits,L.data(),0,L.size()*4);
        auto mine = ssvae::logits_to_coords(L);
        auto gold = ssvae::load_golden_coords(std::string(GOLD)+"/stage1_out/coords.npy");
        int inter=0; for(auto&c:mine) if(gold.count(c)) inter++;
        int uni=(int)mine.size()+(int)gold.size()-inter; double iou=(double)inter/uni;
        printf("  coords mine N=%zu  golden N=%zu  inter=%d  IoU=%.4f\n", mine.size(), gold.size(), inter, iou);
        bool ok = (mine.size()>=1100 && mine.size()<=1135 && iou>0.97);
        printf("[stage1_e2e] %s (expect N~1120, IoU~0.986 vs bf16 golden)\n", ok?"PASS":"FAIL");
        return ok?0:1;
    }
}
