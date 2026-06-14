// normal_bake.hpp — bake a TANGENT-SPACE normal map from a dense high-poly mesh onto a low-poly
// (retopo) mesh's UV atlas. The "sludge → diamonds" move: the riggable low-poly renders with the dense
// mesh's surface detail. Self-contained (reuses texrp::DenseHash for closest-point); does NOT touch the
// shared texatlas::bake(). CPU.
//
// Per atlas texel: rasterize the low-poly UV triangle → barycentric 3D position P + interpolated
// low-poly normal N + per-triangle tangent T (from UV/pos gradients). Find the closest front-facing
// point on the dense shell and interpolate the dense NORMAL Nd there. Encode Nd in the texel's tangent
// basis (T,B,N) → RGB. Uncovered/gutter texels get flat (128,128,255) then a small dilation.
#pragma once
#include "tex_reproject.hpp"     // texrp::DenseHash
#include "tex_atlas.hpp"         // texatlas::vert_normals
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace nrmbake {

static inline void norm3(float* v){ float l=std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); if(l>1e-20f){v[0]/=l;v[1]/=l;v[2]/=l;} }

// low-poly verts/normals/uvs [V*3,V*3,V*2], faces uint32 [F*3]; dense dverts/dfaces. Returns RGB
// tangent-space normal map [tw*th*3]. ncell = dense hash resolution (256 like the reproject).
inline std::vector<uint8_t> bake_normal_map(
        const std::vector<float>& verts, const std::vector<float>& normals,
        const std::vector<float>& uvs, const std::vector<uint32_t>& faces,
        const std::vector<float>& dverts, const std::vector<int64_t>& dfaces,
        int tw, int th, int raster_ss=2, int ncell=256) {

    // dense per-vertex normals (the detail source)
    std::vector<float> dnrm = texatlas::vert_normals(dverts, dfaces);
    texrp::DenseHash hash(dverts.data(), dfaces.data(), (int64_t)dfaces.size()/3, ncell);

    const size_t T = (size_t)tw*th;
    std::vector<float> accum(T*3, 0.f);     // accumulated tangent-space normal (for SS averaging)
    std::vector<float> wsum(T, 0.f);
    const uint32_t F = (uint32_t)faces.size()/3;

    for (uint32_t f=0; f<F; f++) {
        uint32_t ia=faces[f*3], ib=faces[f*3+1], ic=faces[f*3+2];
        const float *Pa=&verts[ia*3], *Pb=&verts[ib*3], *Pc=&verts[ic*3];
        const float *Na=&normals[ia*3], *Nb=&normals[ib*3], *Nc=&normals[ic*3];
        float ua=uvs[ia*2], va=uvs[ia*2+1], ub=uvs[ib*2], vb=uvs[ib*2+1], uc=uvs[ic*2], vc=uvs[ic*2+1];
        // pixel-space UV
        float ax=ua*tw, ay=va*th, bx=ub*tw, by=vb*th, cx=uc*tw, cy=vc*th;
        // per-triangle tangent (pos gradient w.r.t. u): T = (e1*duv2.y - e2*duv1.y)/det
        float e1[3]={Pb[0]-Pa[0],Pb[1]-Pa[1],Pb[2]-Pa[2]}, e2[3]={Pc[0]-Pa[0],Pc[1]-Pa[1],Pc[2]-Pa[2]};
        float du1=ub-ua, dv1=vb-va, du2=uc-ua, dv2=vc-va;
        float det=du1*dv2-du2*dv1; if (std::fabs(det)<1e-20f) det=1e-20f; float r=1.f/det;
        float Tg[3]={ (e1[0]*dv2-e2[0]*dv1)*r, (e1[1]*dv2-e2[1]*dv1)*r, (e1[2]*dv2-e2[2]*dv1)*r };
        // bbox in texels
        int x0=(int)std::floor(std::min({ax,bx,cx})), x1=(int)std::ceil(std::max({ax,bx,cx}));
        int y0=(int)std::floor(std::min({ay,by,cy})), y1=(int)std::ceil(std::max({ay,by,cy}));
        x0=std::max(0,x0); y0=std::max(0,y0); x1=std::min(tw-1,x1); y1=std::min(th-1,y1);
        float area=(bx-ax)*(cy-ay)-(cx-ax)*(by-ay); if (std::fabs(area)<1e-12f) continue; float invA=1.f/area;
        for (int y=y0;y<=y1;y++) for (int x=x0;x<=x1;x++) {
            for (int sy=0;sy<raster_ss;sy++) for (int sx=0;sx<raster_ss;sx++) {
                float px=x+((float)sx+0.5f)/raster_ss, py=y+((float)sy+0.5f)/raster_ss;
                float l0=((bx-px)*(cy-py)-(cx-px)*(by-py))*invA;
                float l1=((cx-px)*(ay-py)-(ax-px)*(cy-py))*invA;
                float l2=1.f-l0-l1;
                if (l0<-1e-4f||l1<-1e-4f||l2<-1e-4f) continue;
                float P[3]={l0*Pa[0]+l1*Pb[0]+l2*Pc[0], l0*Pa[1]+l1*Pb[1]+l2*Pc[1], l0*Pa[2]+l1*Pb[2]+l2*Pc[2]};
                float N[3]={l0*Na[0]+l1*Nb[0]+l2*Nc[0], l0*Na[1]+l1*Nb[1]+l2*Nc[1], l0*Na[2]+l1*Nb[2]+l2*Nc[2]};
                norm3(N);
                // orthonormal tangent basis at this texel
                float Tt[3]={Tg[0],Tg[1],Tg[2]}; float d=Tt[0]*N[0]+Tt[1]*N[1]+Tt[2]*N[2];
                Tt[0]-=N[0]*d; Tt[1]-=N[1]*d; Tt[2]-=N[2]*d; norm3(Tt);
                float B[3]={N[1]*Tt[2]-N[2]*Tt[1], N[2]*Tt[0]-N[0]*Tt[2], N[0]*Tt[1]-N[1]*Tt[0]};
                // dense normal at closest front-facing point
                float Nd[3]={0,0,0}, snapped[3];
                bool ok=hash.sample(P, N, dnrm.data(), 3, Nd, snapped, /*front_dot*/0.0f, /*max_ring*/6,
                                    /*max_dist2*/-1.f, /*back_fallback*/true);
                if (!ok) continue; norm3(Nd);
                // tangent-space encode
                float ts[3]={ Nd[0]*Tt[0]+Nd[1]*Tt[1]+Nd[2]*Tt[2],
                              Nd[0]*B[0]+Nd[1]*B[1]+Nd[2]*B[2],
                              Nd[0]*N[0]+Nd[1]*N[1]+Nd[2]*N[2] };
                if (ts[2] < 0.f) { ts[0]=-ts[0]; ts[1]=-ts[1]; ts[2]=-ts[2]; }   // keep z+ (texel front)
                size_t p=(size_t)y*tw+x; accum[p*3]+=ts[0]; accum[p*3+1]+=ts[1]; accum[p*3+2]+=ts[2]; wsum[p]+=1.f;
            }
        }
    }

    std::vector<uint8_t> out(T*3); std::vector<uint8_t> valid(T,0);
    auto enc=[](float v){ int q=(int)std::lround((v*0.5f+0.5f)*255.f); return (uint8_t)std::max(0,std::min(255,q)); };
    for (size_t p=0;p<T;p++) {
        if (wsum[p]>0.f) { float n[3]={accum[p*3]/wsum[p],accum[p*3+1]/wsum[p],accum[p*3+2]/wsum[p]}; norm3(n);
            out[p*3]=enc(n[0]); out[p*3+1]=enc(n[1]); out[p*3+2]=enc(n[2]); valid[p]=1; }
        else { out[p*3]=128; out[p*3+1]=128; out[p*3+2]=255; }   // flat
    }
    // dilate valid normals into the gutter a few px (avoid flat-seam bilinear bleed at chart edges)
    for (int it=0; it<4; it++) {
        std::vector<uint8_t> nv=valid;
        for (int y=0;y<th;y++) for (int x=0;x<tw;x++) { size_t p=(size_t)y*tw+x; if (valid[p]) continue;
            int sx=0,sy=0,sz=0,c=0;
            for (int dy=-1;dy<=1;dy++) for (int dx=-1;dx<=1;dx++){ int xx=x+dx,yy=y+dy; if(xx<0||yy<0||xx>=tw||yy>=th)continue;
                size_t q=(size_t)yy*tw+xx; if(valid[q]){ sx+=out[q*3];sy+=out[q*3+1];sz+=out[q*3+2];c++; } }
            if (c){ out[p*3]=(uint8_t)(sx/c); out[p*3+1]=(uint8_t)(sy/c); out[p*3+2]=(uint8_t)(sz/c); nv[p]=1; } }
        valid.swap(nv);
    }
    return out;
}

}  // namespace nrmbake
