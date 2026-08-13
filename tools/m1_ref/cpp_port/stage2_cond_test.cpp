// M5 building block: shape-stage proj conditioning (grid 32) — the TRUE cond for M2.
//
// get_proj_cond_shape: DinoV3ProjFeatureExtractor produces z_proj = concat[proj_lr, proj_hr]:
//   proj_lr = proj_grid32(DINOv3 patchmap [32x32, 1024])     (lr branch)
//   proj_hr = proj_grid32(NAF hr feature map [512x512, 1024]) (hr branch)
//   -> [grid32^3, 2048], gathered at the voxel coords.
// Same projection as the validated SS proj_grid (img_res 512); only R=32 + the two feature
// maps (32x32 lr / 512x512 hr) differ. Validates concat[proj_lr,proj_hr] at the golden coords
// vs golden_stages/stage2_cond/proj_feats, using the already-validated DINOv3 patchmap + NAF hr.
//
// Host C++ (the cond feeds the M2 DiT graph as an uploaded input, like the golden cond).
// Build: ./build.sh stage2_cond_test   (no cuda needed; pure host gather)
#include "../../sparse_spike/npy.hpp"
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

static const char* REFS = "refs";
static const char* GOLD = "../../sparse_spike/golden_stages";

static void inv4x4(const float A[16], float out[16]) {
    float m[4][8];
    for (int r=0;r<4;r++){ for(int c=0;c<4;c++) m[r][c]=A[r*4+c]; for(int c=0;c<4;c++) m[r][4+c]=(r==c)?1.f:0.f; }
    for (int col=0;col<4;col++){
        int piv=col; for(int r=col+1;r<4;r++) if(std::fabs(m[r][col])>std::fabs(m[piv][col])) piv=r;
        for(int c=0;c<8;c++) std::swap(m[col][c],m[piv][c]);
        float d=m[col][col]; for(int c=0;c<8;c++) m[col][c]/=d;
        for(int r=0;r<4;r++){ if(r==col) continue; float f=m[r][col]; for(int c=0;c<8;c++) m[r][c]-=f*m[col][c]; }
    }
    for(int r=0;r<4;r++) for(int c=0;c<4;c++) out[r*4+c]=m[r][4+c];
}

int main() {
    // camera scalars (cam.json — same camera for ss/shape stages; from the miku decode)
    const float cam = 0.7332379387484828f, dist = 1.3021559715270996f, ms = 1.0f;
    const int R = 32, img_res = 512, C = 1024;
    const int Hlr = 32, Wlr = 32, Hhr = 512, Whr = 512;

    // host projection: T (camera), focal — identical to the validated proj_grid_test
    float T[16] = {1,0,0,0,  0,0,-1,-dist,  0,1,0,0,  0,0,0,1};
    float Winv[16]; inv4x4(T, Winv);
    const float focal = 16.f / std::tan(cam / 2.f);
    const float focal_px = focal * img_res / 32.f;
    auto lin1 = [&](int i){ return R==1 ? -1.f : -1.f + 2.f*i/(R-1); };

    // load coords + validated feature maps
    NpyArray cN = npy_load(std::string(GOLD) + "/stage2_cond/proj_coords.npy");
    int N = (int)cN.shape[0]; const int32_t* co = cN.i32();
    NpyArray pmN = npy_load(std::string(REFS) + "/dino_patchmap.npy");   // [1,32,32,1024] BHWC
    NpyArray hrN = npy_load(std::string(REFS) + "/naf_hr_fp32.npy");     // [1,1024,512,512] BCHW
    const float* pm = pmN.f32();   // pm[h*32*1024 + w*1024 + c]
    const float* hr = hrN.f32();   // hr[c*512*512 + h*512 + w]
    printf("[s2cond] N=%d  patchmap%s  hr%s\n", N,
           (pmN.shape==std::vector<int64_t>{1,32,32,1024})?"[1,32,32,1024]":"?",
           (hrN.shape==std::vector<int64_t>{1,1024,512,512})?"[1,1024,512,512]":"?");

    std::vector<float> cond((size_t)N * 2048);

    // per coord: project its grid cell to pixel, bilinear-gather both maps
    for (int n = 0; n < N; n++) {
        int xi = co[n*4+1], yi = co[n*4+2], zi = co[n*4+3];
        float x = lin1(xi), y = lin1(yi), z = lin1(zi);
        float gx = x/ms/2.f, gy = (-z)/ms/2.f, gz = (y)/ms/2.f;
        float xc = Winv[0]*gx + Winv[1]*gy + Winv[2]*gz + Winv[3];
        float yc = Winv[4]*gx + Winv[5]*gy + Winv[6]*gz + Winv[7];
        float zc = Winv[8]*gx + Winv[9]*gy + Winv[10]*gz + Winv[11];
        float x_ndc = focal_px * xc / (-zc + 1e-8f);
        float y_ndc = focal_px * yc / (-zc + 1e-8f);
        float x_pix = x_ndc + img_res/2.f;
        float y_pix = -y_ndc + img_res/2.f;
        float gnx = (x_pix + 0.5f)/img_res*2.f - 1.f;
        float gny = (y_pix + 0.5f)/img_res*2.f - 1.f;

        // bilinear corners for a given map (H,W), align_corners=False, border clamp
        auto corners = [&](int Hm, int Wm, int& a, int& b, int& c2, int& d2, float& wa, float& wb, float& wc, float& wd) {
            float ix = (gnx+1.f)*0.5f*Wm - 0.5f, iy = (gny+1.f)*0.5f*Hm - 0.5f;
            int x0 = (int)std::floor(ix), y0 = (int)std::floor(iy), x1 = x0+1, y1 = y0+1;
            float wx1 = ix-x0, wy1 = iy-y0, wx0 = 1.f-wx1, wy0 = 1.f-wy1;
            auto cl=[](int v,int hi){ return v<0?0:(v>hi?hi:v); };
            int x0c=cl(x0,Wm-1),x1c=cl(x1,Wm-1),y0c=cl(y0,Hm-1),y1c=cl(y1,Hm-1);
            a=y0c*Wm+x0c; b=y0c*Wm+x1c; c2=y1c*Wm+x0c; d2=y1c*Wm+x1c;
            wa=wy0*wx0; wb=wy0*wx1; wc=wy1*wx0; wd=wy1*wx1;
        };
        // lr: BHWC patchmap, pos*1024 + c
        { int a,b,c2,d2; float wa,wb,wc,wd; corners(Hlr,Wlr,a,b,c2,d2,wa,wb,wc,wd);
          const float *pa=pm+(size_t)a*C,*pb=pm+(size_t)b*C,*pc=pm+(size_t)c2*C,*pd=pm+(size_t)d2*C;
          float* out = cond.data()+(size_t)n*2048;
          for (int c=0;c<C;c++) out[c] = wa*pa[c]+wb*pb[c]+wc*pc[c]+wd*pd[c]; }
        // hr: BCHW map, c*HW + pos
        { int a,b,c2,d2; float wa,wb,wc,wd; corners(Hhr,Whr,a,b,c2,d2,wa,wb,wc,wd);
          const size_t HW=(size_t)Hhr*Whr;
          float* out = cond.data()+(size_t)n*2048 + C;
          for (int c=0;c<C;c++){ const float* hc=hr+(size_t)c*HW; out[c]=wa*hc[a]+wb*hc[b]+wc*hc[c2]+wd*hc[d2]; } }
    }

    // compare vs golden stage2_cond proj_feats
    NpyArray gN = npy_load(std::string(GOLD) + "/stage2_cond/proj_feats.npy");
    const float* g = gN.f32();
    double ma=0,sum=0,dot=0,na=0,nb=0; size_t worst=0;
    // split lr/hr error to localise
    double malr=0,mahr=0;
    for (size_t i=0;i<cond.size();i++){ double d=std::fabs((double)cond[i]-g[i]); if(d>ma){ma=d;worst=i;} sum+=d;
        dot+=(double)cond[i]*g[i]; na+=(double)cond[i]*cond[i]; nb+=(double)g[i]*g[i];
        if ((i%2048)<1024) malr=std::max(malr,d); else mahr=std::max(mahr,d); }
    printf("[s2cond] vs golden proj_feats[%d,2048]: maxabs=%.3e meanabs=%.3e cosine=%.6f (lr-maxabs=%.3e hr-maxabs=%.3e) worst@%zu\n",
           N, ma, sum/cond.size(), dot/(std::sqrt(na*nb)+1e-12), malr, mahr, worst);
    bool ok = ma < 5e-3;  // fp32 grid_sample vs fp16-golden NAF/DINOv3 maps; judge by cosine+lr
    printf("[s2cond] %s (cosine should be ~1; lr branch should be very tight)\n", ok?"PASS":"CHECK");
    return ok?0:1;
}
