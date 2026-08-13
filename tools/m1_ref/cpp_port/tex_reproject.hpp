// lap-18 BVH-REPROJECT: bake a UV PBR texture for the QEM mesh by snapping each rasterised texel's
// 3D position onto the smooth DENSE outer-shell mesh (closest-POINT-on-triangle, NOT nearest-vertex —
// nearest-vertex jumps across the thin twin-tail gaps), with FRONT-FACE rejection (a dense triangle
// whose normal opposes the texel's surface normal is the back side → skip it), then barycentric-interp
// the dense per-vertex PBR (6-ch). This is what the per-vertex colour was approximating, but:
//   - on a continuous UV texture (no facet cracks — the "smashed porcelain" goes away),
//   - sampled on the OUTER shell only (no teal-interior splatter from folded charts),
//   - front-face filtered (no dark back-of-hair bleed onto the front of the tails),
//   - carries metallic/roughness (per-vertex COLOR_0 dropped them).
// Colour space is handled by the baseColor TEXTURE being glTF-sRGB (the renderer sRGB→linear-decodes
// it, matching Python's baseColorTexture) — which fixes the COLOR_0 "washed out" (COLOR_0 is linear,
// so it skipped that decode and rendered too bright).
#pragma once
#include <cstdint>
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>

namespace texrp {

// --- closest point on triangle (Ericson, Real-Time Collision Detection §5.1.5) ---
// returns squared distance; writes barycentric (u,v,w) for (a,b,c) of the closest point.
static inline float closest_pt_tri(const float* p, const float* a, const float* b, const float* c,
                                   float& u, float& v, float& w) {
    float ab[3]={b[0]-a[0],b[1]-a[1],b[2]-a[2]};
    float ac[3]={c[0]-a[0],c[1]-a[1],c[2]-a[2]};
    float ap[3]={p[0]-a[0],p[1]-a[1],p[2]-a[2]};
    float d1=ab[0]*ap[0]+ab[1]*ap[1]+ab[2]*ap[2];
    float d2=ac[0]*ap[0]+ac[1]*ap[1]+ac[2]*ap[2];
    if (d1<=0.f && d2<=0.f){ u=1;v=0;w=0; float dx=p[0]-a[0],dy=p[1]-a[1],dz=p[2]-a[2]; return dx*dx+dy*dy+dz*dz; }
    float bp[3]={p[0]-b[0],p[1]-b[1],p[2]-b[2]};
    float d3=ab[0]*bp[0]+ab[1]*bp[1]+ab[2]*bp[2];
    float d4=ac[0]*bp[0]+ac[1]*bp[1]+ac[2]*bp[2];
    if (d3>=0.f && d4<=d3){ u=0;v=1;w=0; float dx=p[0]-b[0],dy=p[1]-b[1],dz=p[2]-b[2]; return dx*dx+dy*dy+dz*dz; }
    float vc=d1*d4-d3*d2;
    if (vc<=0.f && d1>=0.f && d3<=0.f){ float t=d1/(d1-d3); u=1-t;v=t;w=0;
        float q[3]={a[0]+t*ab[0],a[1]+t*ab[1],a[2]+t*ab[2]}; float dx=p[0]-q[0],dy=p[1]-q[1],dz=p[2]-q[2]; return dx*dx+dy*dy+dz*dz; }
    float cp[3]={p[0]-c[0],p[1]-c[1],p[2]-c[2]};
    float d5=ab[0]*cp[0]+ab[1]*cp[1]+ab[2]*cp[2];
    float d6=ac[0]*cp[0]+ac[1]*cp[1]+ac[2]*cp[2];
    if (d6>=0.f && d5<=d6){ u=0;v=0;w=1; float dx=p[0]-c[0],dy=p[1]-c[1],dz=p[2]-c[2]; return dx*dx+dy*dy+dz*dz; }
    float vb=d5*d2-d1*d6;
    if (vb<=0.f && d2>=0.f && d6<=0.f){ float t=d2/(d2-d6); u=1-t;v=0;w=t;
        float q[3]={a[0]+t*ac[0],a[1]+t*ac[1],a[2]+t*ac[2]}; float dx=p[0]-q[0],dy=p[1]-q[1],dz=p[2]-q[2]; return dx*dx+dy*dy+dz*dz; }
    float va=d3*d6-d5*d4;
    if (va<=0.f && (d4-d3)>=0.f && (d5-d6)>=0.f){ float t=(d4-d3)/((d4-d3)+(d5-d6)); u=0;v=1-t;w=t;
        float q[3]={b[0]+t*(c[0]-b[0]),b[1]+t*(c[1]-b[1]),b[2]+t*(c[2]-b[2])}; float dx=p[0]-q[0],dy=p[1]-q[1],dz=p[2]-q[2]; return dx*dx+dy*dy+dz*dz; }
    float denom=1.f/(va+vb+vc); float vv=vb*denom, ww=vc*denom; u=1-vv-ww; v=vv; w=ww;
    float q[3]={a[0]+ab[0]*vv+ac[0]*ww, a[1]+ab[1]*vv+ac[1]*ww, a[2]+ab[2]*vv+ac[2]*ww};
    float dx=p[0]-q[0],dy=p[1]-q[1],dz=p[2]-q[2]; return dx*dx+dy*dy+dz*dz;
}

// uniform spatial hash over the DENSE mesh triangles (verts in [-0.5,0.5]). A triangle is registered
// in every cell its 3 verts touch (triangles are ~1 grid-1024 unit, so this covers them). Query
// expands cell rings around P until a hit (+ one extra ring) and keeps the nearest FRONT-FACING tri.
struct DenseHash {
    const float* V; const int64_t* F; int64_t NF;
    int n;                       // cells per axis
    float inv;                   // n (cell index = (p+0.5)*n)
    std::vector<int> cell_start; // CSR over cells
    std::vector<int> tri_idx;    // triangle ids, grouped by cell
    std::vector<float> fn;       // per-face unit normal [NF*3]

    inline int cidx(int x,int y,int z) const { return (x*n+y)*n+z; }
    inline void clamp3(int& x,int& y,int& z) const {
        x=x<0?0:(x>=n?n-1:x); y=y<0?0:(y>=n?n-1:y); z=z<0?0:(z>=n?n-1:z); }

    DenseHash(const float* verts, const int64_t* faces, int64_t nf, int ncell)
        : V(verts), F(faces), NF(nf), n(ncell) {
        inv=(float)n;
        fn.resize((size_t)NF*3);
        std::vector<int> count((size_t)n*n*n+1, 0);
        auto cellOf=[&](const float* p, int& cx,int& cy,int& cz){
            cx=(int)((p[0]+0.5f)*inv); cy=(int)((p[1]+0.5f)*inv); cz=(int)((p[2]+0.5f)*inv); clamp3(cx,cy,cz); };
        // face normals + count cell occupancy (each tri → its 3 distinct vertex cells)
        for (int64_t t=0;t<NF;t++){ const float*a=&V[F[t*3]*3],*b=&V[F[t*3+1]*3],*c=&V[F[t*3+2]*3];
            float e1[3]={b[0]-a[0],b[1]-a[1],b[2]-a[2]}, e2[3]={c[0]-a[0],c[1]-a[1],c[2]-a[2]};
            float nx=e1[1]*e2[2]-e1[2]*e2[1], ny=e1[2]*e2[0]-e1[0]*e2[2], nz=e1[0]*e2[1]-e1[1]*e2[0];
            float L=std::sqrt(nx*nx+ny*ny+nz*nz); if(L<1e-30f)L=1; fn[t*3]=nx/L; fn[t*3+1]=ny/L; fn[t*3+2]=nz/L;
            int last=-1; for(int k=0;k<3;k++){ int cx,cy,cz; cellOf(&V[F[t*3+k]*3],cx,cy,cz); int ci=cidx(cx,cy,cz);
                if(ci!=last){ count[ci+1]++; last=ci; } }
        }
        for (size_t i=1;i<count.size();i++) count[i]+=count[i-1];
        cell_start=count;                      // CSR offsets
        tri_idx.resize(count.back());
        std::vector<int> cur(count.begin(), count.end()-1);
        for (int64_t t=0;t<NF;t++){ int last=-1;
            for(int k=0;k<3;k++){ int cx,cy,cz; cellOf(&V[F[t*3+k]*3],cx,cy,cz); int ci=cidx(cx,cy,cz);
                if(ci!=last){ tri_idx[cur[ci]++]=(int)t; last=ci; } }
        }
    }

    // closest front-facing point to P (query normal qn). Writes the snapped on-shell 3D point to
    // out_pt[3] (the caller then grid_samples the VOLUME there — trilinear-stable, so no speckle from
    // per-texel triangle-choice flips), AND the 6-ch barycentric dense attr to out[C] (cheap; used
    // when the caller prefers the mesh attr). Returns false if no triangle found at all (caller
    // inpaints). front_dot = min dot to count as same-facing; falls back to nearest-any if no front.
    inline bool sample(const float* P, const float* qn, const float* attr, int C,
                       float* out, float* out_pt, float front_dot, int max_ring,
                       float max_dist2 = -1.f, bool allow_back_fallback = true) const {
        int bx=(int)((P[0]+0.5f)*inv), by=(int)((P[1]+0.5f)*inv), bz=(int)((P[2]+0.5f)*inv); clamp3(bx,by,bz);
        float bestd=1e30f, bestu=0,bestv=0,bestw=0; int bestt=-1;
        float bestd_any=1e30f, bu_any=0,bv_any=0,bw_any=0; int bt_any=-1;
        int found_ring=-1;
        for (int r=0;r<=max_ring;r++){
            if (found_ring>=0 && r>found_ring+1) break;   // one ring past first hit suffices
            int x0=bx-r,x1=bx+r,y0=by-r,y1=by+r,z0=bz-r,z1=bz+r;
            for (int x=x0;x<=x1;x++){ if(x<0||x>=n) continue;
              for (int y=y0;y<=y1;y++){ if(y<0||y>=n) continue;
                for (int z=z0;z<=z1;z++){ if(z<0||z>=n) continue;
                  if (std::max(std::max(std::abs(x-bx),std::abs(y-by)),std::abs(z-bz))!=r) continue; // shell only
                  int ci=cidx(x,y,z);
                  for (int ii=cell_start[ci]; ii<cell_start[ci+1]; ii++){ int t=tri_idx[ii];
                    float uu,vv,ww; float d=closest_pt_tri(P,&V[F[t*3]*3],&V[F[t*3+1]*3],&V[F[t*3+2]*3],uu,vv,ww);
                    if (d<bestd_any){ bestd_any=d; bu_any=uu;bv_any=vv;bw_any=ww; bt_any=t; }
                    float fd=fn[t*3]*qn[0]+fn[t*3+1]*qn[1]+fn[t*3+2]*qn[2];
                    if (fd>=front_dot && d<bestd){ bestd=d; bestu=uu;bestv=vv;bestw=ww; bestt=t; }
                  }
                }
              }
            }
            if ((bestt>=0||bt_any>=0) && found_ring<0) found_ring=r;
        }
        int t=bestt; float U=bestu,Vv=bestv,Wv=bestw;
        if (t<0 && allow_back_fallback){ t=bt_any; U=bu_any;Vv=bv_any;Wv=bw_any; bestd=bestd_any; }   // no front tri → fall back to nearest any
        if (t<0) return false;
        if (max_dist2 >= 0.f && bestd > max_dist2) return false;
        const float* a=&V[F[t*3]*3]; const float* b=&V[F[t*3+1]*3]; const float* c=&V[F[t*3+2]*3];
        if (out_pt) for (int d=0;d<3;d++) out_pt[d]=U*a[d]+Vv*b[d]+Wv*c[d];   // snapped on-shell 3D point
        if (out && attr){ const float* fa=&attr[(size_t)F[t*3]*C]; const float* fb=&attr[(size_t)F[t*3+1]*C]; const float* fc=&attr[(size_t)F[t*3+2]*C];
            for (int cc=0;cc<C;cc++) out[cc]=U*fa[cc]+Vv*fb[cc]+Wv*fc[cc]; }
        return true;
    }
};

}  // namespace texrp
