// Host helpers for the Pixal3D geometry E2E driver (A1 chain assembly):
//   - ProjCam + proj projection (shared by stage1 dense grid16 and shape coord-gather grid32/64)
//   - flow_sampler (the FlowEulerGuidanceIntervalSampler loop, double t_seq — interval-safe)
//   - grid64 quantize + unique (sorted, == torch quant_coords.unique(dim=0))
// All validated math, lifted verbatim from the per-stage tests (proj == stage2/3b_cond_test,
// sampler == m2/m3b_sampler_test). No ggml/torch deps — pure host.
#pragma once
#include <cstdint>
#include <cmath>
#include <vector>
#include <functional>
#include <algorithm>
#include <set>
#include <array>

namespace geo {

static inline void inv4x4(const float A[16], float out[16]) {
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

// Camera unprojection (== proj_grid_test / stage*_cond_test). camera_angle_x, distance, mesh_scale.
struct ProjCam {
    float cam, dist, ms;
    float Winv[16];
    ProjCam(float camera_angle_x, float distance, float mesh_scale)
        : cam(camera_angle_x), dist(distance), ms(mesh_scale) {
        float T[16] = {1,0,0,0, 0,0,-1,-dist, 0,1,0,0, 0,0,0,1};
        inv4x4(T, Winv);
    }
    static inline float lin1(int i, int R) { return R==1 ? -1.f : -1.f + 2.f*i/(R-1); }
    // grid cell (xi,yi,zi) in [0,R) -> NDC (gnx,gny) on an img_res image
    void project(int xi, int yi, int zi, int R, int img_res, float& gnx, float& gny) const {
        const float focal_px = (img_res * 0.5f) / std::tan(cam * 0.5f);
        float x = lin1(xi,R), y = lin1(yi,R), z = lin1(zi,R);
        float gx = x/ms/2.f, gy = (-z)/ms/2.f, gz = (y)/ms/2.f;
        float xc = Winv[0]*gx + Winv[1]*gy + Winv[2]*gz + Winv[3];
        float yc = Winv[4]*gx + Winv[5]*gy + Winv[6]*gz + Winv[7];
        float zc = Winv[8]*gx + Winv[9]*gy + Winv[10]*gz + Winv[11];
        float x_ndc = focal_px * xc / (-zc + 1e-8f), y_ndc = focal_px * yc / (-zc + 1e-8f);
        float x_pix = x_ndc + img_res/2.f, y_pix = -y_ndc + img_res/2.f;
        gnx = (x_pix + 0.5f)/img_res*2.f - 1.f;
        gny = (y_pix + 0.5f)/img_res*2.f - 1.f;
    }
};

// bilinear corners (align_corners=False, border clamp) for a (Hm x Wm) map
static inline void bilinear(float gnx, float gny, int Hm, int Wm, int idx[4], float w[4]) {
    float ix = (gnx+1.f)*0.5f*Wm - 0.5f, iy = (gny+1.f)*0.5f*Hm - 0.5f;
    int x0=(int)std::floor(ix), y0=(int)std::floor(iy), x1=x0+1, y1=y0+1;
    float wx1=ix-x0, wy1=iy-y0, wx0=1.f-wx1, wy0=1.f-wy1;
    auto cl=[](int v,int hi){ return v<0?0:(v>hi?hi:v); };
    int x0c=cl(x0,Wm-1),x1c=cl(x1,Wm-1),y0c=cl(y0,Hm-1),y1c=cl(y1,Hm-1);
    idx[0]=y0c*Wm+x0c; idx[1]=y0c*Wm+x1c; idx[2]=y1c*Wm+x0c; idx[3]=y1c*Wm+x1c;
    w[0]=wy0*wx0; w[1]=wy0*wx1; w[2]=wy1*wx0; w[3]=wy1*wx1;
}

// Stage-1 SS proj cond (DINO-only, dense over the R^3 grid). patchmap BHWC [Hlr*Wlr, C].
// out[cell, C] for cell = (i*R+j)*R+k iterating i,j,k ascending (== grid flatten in stage1_e2e).
static inline std::vector<float> proj_cond_ss(const float* patchmap, int Hlr, int Wlr, int C,
        int R, int img_res, const ProjCam& cam) {
    std::vector<float> out((size_t)R*R*R * C);
    int cell = 0;
    for (int i=0;i<R;i++) for (int j=0;j<R;j++) for (int k=0;k<R;k++) {
        float gnx, gny; cam.project(i, j, k, R, img_res, gnx, gny);
        int idx[4]; float w[4]; bilinear(gnx, gny, Hlr, Wlr, idx, w);
        const float *p0=patchmap+(size_t)idx[0]*C, *p1=patchmap+(size_t)idx[1]*C,
                    *p2=patchmap+(size_t)idx[2]*C, *p3=patchmap+(size_t)idx[3]*C;
        float* o = out.data()+(size_t)cell*C;
        for (int c=0;c<C;c++) o[c] = w[0]*p0[c]+w[1]*p1[c]+w[2]*p2[c]+w[3]*p3[c];
        cell++;
    }
    return out;
}

// Shape proj cond at voxel coords: concat[proj_lr(DINOv3 patchmap BHWC [Hlr*Wlr,C]),
//   proj_hr(NAF hr BCHW [C,Hhr*Whr])] -> [N, 2*C]. coords [N,4]=(b,x,y,z).
static inline std::vector<float> proj_cond_shape(const int32_t* coords, int N,
        const float* patchmap, int Hlr, int Wlr, const float* naf_hr, int Hhr, int Whr, int C,
        int R, int img_res, const ProjCam& cam) {
    std::vector<float> out((size_t)N * (2*C));
    const size_t HW = (size_t)Hhr*Whr;
    #pragma omp parallel for schedule(static)
    for (int n=0;n<N;n++) {
        int xi=coords[n*4+1], yi=coords[n*4+2], zi=coords[n*4+3];
        float gnx, gny; cam.project(xi, yi, zi, R, img_res, gnx, gny);
        float* o = out.data()+(size_t)n*(2*C);
        // lr: BHWC
        { int idx[4]; float w[4]; bilinear(gnx, gny, Hlr, Wlr, idx, w);
          const float *p0=patchmap+(size_t)idx[0]*C, *p1=patchmap+(size_t)idx[1]*C,
                      *p2=patchmap+(size_t)idx[2]*C, *p3=patchmap+(size_t)idx[3]*C;
          for (int c=0;c<C;c++) o[c]=w[0]*p0[c]+w[1]*p1[c]+w[2]*p2[c]+w[3]*p3[c]; }
        // hr: BCHW
        { int idx[4]; float w[4]; bilinear(gnx, gny, Hhr, Whr, idx, w);
          float* oh=o+C; for (int c=0;c<C;c++){ const float* hc=naf_hr+(size_t)c*HW;
            oh[c]=w[0]*hc[idx[0]]+w[1]*hc[idx[1]]+w[2]*hc[idx[2]]+w[3]*hc[idx[3]]; } }
    }
    return out;
}

// FlowEulerGuidanceIntervalSampler — host loop (== m2/m3b_sampler_test). forward(x,t_scaled,cond)
// runs the (ggml) DiT; t_seq + interval test in DOUBLE (RT=3 lands a step exactly at t=0.6).
static inline double vstd(const std::vector<float>& x) {  // population std over all elems
    double m=0; for (float v:x) m+=v; m/=x.size();
    double s=0; for (float v:x) s+=(v-m)*(v-m); return std::sqrt(s/x.size());
}
static inline std::vector<float> flow_sampler(int NEL, const std::vector<float>& noise,
        float SM, float GS, float GR, double RT, double IV0, double IV1, int STEPS,
        const std::function<std::vector<float>(const std::vector<float>&, float, bool)>& forward,
        const char* tag = nullptr) {
    std::vector<float> x = noise;
    auto pred_to_x0 = [&](const std::vector<float>& xt, float t, const std::vector<float>& pr){
        std::vector<float> o(NEL); float a=(1-SM), b=(SM+(1-SM)*t);
        for(int i=0;i<NEL;i++) o[i]=a*xt[i]-b*pr[i];
        return o; };
    auto x0_to_pred = [&](const std::vector<float>& xt, float t, const std::vector<float>& x0){
        std::vector<float> o(NEL); float a=(1-SM), b=(SM+(1-SM)*t);
        for(int i=0;i<NEL;i++) o[i]=(a*xt[i]-x0[i])/b;
        return o; };
    std::vector<double> tseq(STEPS+1);
    for (int i=0;i<=STEPS;i++){ double lt=1.0-(double)i/STEPS; tseq[i]=RT*lt/(1+(RT-1)*lt); }
    const bool cfg_off = (GS == 1.0f);  // tex: GS1.0/GR0 -> CFG branch == cond-only; skip neg forward
    for (int s=0;s<STEPS;s++){
        double t=tseq[s], tp=tseq[s+1];
        bool in_iv = (t>=IV0 && t<=IV1) && !cfg_off;
        std::vector<float> v;
        if (in_iv){
            std::vector<float> vp=forward(x,(float)(1000.0*t),true), vn=forward(x,(float)(1000.0*t),false);
            std::vector<float> pred(NEL); for(int i=0;i<NEL;i++) pred[i]=GS*vp[i]+(1-GS)*vn[i];
            std::vector<float> x0p=pred_to_x0(x,(float)t,vp), x0c=pred_to_x0(x,(float)t,pred);
            double r = vstd(x0p)/vstd(x0c);
            std::vector<float> x0(NEL);
            for(int i=0;i<NEL;i++){ float rc=x0c[i]*(float)r; x0[i]=GR*rc+(1-GR)*x0c[i]; }
            v=x0_to_pred(x,(float)t,x0);
        } else v=forward(x,(float)(1000.0*t),true);
        for(int i=0;i<NEL;i++) x[i]-=(float)(t-tp)*v[i];
        if (tag) { printf("  [%s] step %2d/%d t=%.6f %s\n", tag, s+1, STEPS, t, in_iv?"[cfg]":"[cond]"); fflush(stdout); }
    }
    return x;
}

// per-channel denorm: slat = slat*std + mean  (32-dim, channel c = i % C)
static inline void denorm_inplace(std::vector<float>& x, const float* mean, const float* sd, int C) {
    for (size_t i=0;i<x.size();i++){ int c=(int)(i % C); x[i] = x[i]*sd[c] + mean[c]; }
}

// grid64 quantize + unique (== torch run(): q = round((hr_xyz+0.5)/lr_res*(grid_res-1)),
// then quant_coords.unique(dim=0) — sorted lexicographically by (b,x,y,z)).
// hr_coords [Nh,4]=(b,x,y,z). Returns sorted-unique coords [M,4].
static inline std::vector<int32_t> quantize_grid_unique(const int32_t* hr_coords, int Nh,
        int lr_resolution, int grid_res) {
    std::set<std::array<int32_t,4>> uniq;
    const float lr = (float)lr_resolution, gm1 = (float)(grid_res - 1);
    for (int i=0;i<Nh;i++) {
        int32_t b = hr_coords[i*4+0];
        // match torch op order + round-half-to-even: round(((v+0.5)/lr_res)*(grid_res-1))
        auto q = [&](int v){ return (int32_t)std::nearbyintf((((float)v + 0.5f) / lr) * gm1); };
        uniq.insert({b, q(hr_coords[i*4+1]), q(hr_coords[i*4+2]), q(hr_coords[i*4+3])});
    }
    std::vector<int32_t> out; out.reserve(uniq.size()*4);
    for (auto& c : uniq) { out.push_back(c[0]); out.push_back(c[1]); out.push_back(c[2]); out.push_back(c[3]); }
    return out;  // std::set iterates ascending == unique(dim=0) sorted order
}

}  // namespace geo
