// UV-atlas PBR texture bake — the C++/host port of the Python texture post-process
// (Trellis2TexturingPipeline.postprocess_mesh, the no-remesh/no-decimate path that
// o_voxel.postprocess.to_glb reduces to without remeshing). Pipeline:
//   xatlas uv_unwrap(mesh) -> per-vertex UVs + chart layout (re-indexed verts/faces + xref)
//   CPU rasterize each UV triangle -> per-texel barycentric 3D surface position
//   grid_sample_3d trilinear (tex_grid_sample.hpp) the per-voxel 6-ch PBR volume at that position
//   inpaint the UV gutter -> baseColor(RGBA) + metallicRoughness(R0,G=rough,B=metal) atlases
// Produces exactly the textures the glTF needs; the per-voxel 6-ch PBR is already validated
// bit-exact (m6_tex_decode). Replaces the interim per-vertex COLOR_0 bake.
#pragma once
#include <string>
#include "tex_grid_sample.hpp"
#include "tex_reproject.hpp"     // lap-18: closest-point-on-dense-mesh reproject (kills splatter/cracks)
#ifdef TEXATLAS_NATIVE_CUMESH
#include "native_cumesh_bridge.hpp"
#endif
#include "../../../thirdparty/xatlas.h"
#include "../../../thirdparty/meshoptimizer/meshoptimizer.h"
#include <cstdint>
#include <cmath>
#include <vector>
#include <cstdio>
#include <algorithm>
#include <chrono>
#include <array>
#include <unordered_map>
#include <cstdlib>
#include <cstring>
#include <queue>
#include <limits>

namespace texatlas {

static inline double _now(){ using namespace std::chrono; return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count(); }
static double _phase_t0 = 0;
static bool _xatlas_progress(xatlas::ProgressCategory cat, int progress, void*) {
    if (progress==0) { _phase_t0=_now(); }
    if (progress==100) printf("[atlas]   %-14s done (%.1fs)\n",
        cat==xatlas::ProgressCategory::AddMesh?"AddMesh":
        cat==xatlas::ProgressCategory::ComputeCharts?"ComputeCharts":
        cat==xatlas::ProgressCategory::PackCharts?"PackCharts":"BuildOutputMeshes",
        _now()-_phase_t0);
    return true;
}

// Quadric edge-collapse decimation to ~target_faces (meshopt_simplify), then compact unused
// vertices. Matches the role of Python to_glb's cumesh.simplify (decimation_target): makes xatlas
// tractable + the GLB web-sized. The PBR is sampled at the decimated surface — a sub-voxel shift
// from the original (Python corrects via a BVH reproject; the smooth volume makes it negligible).
static inline void decimate(const std::vector<float>& vin, const std::vector<int64_t>& fin,
                            size_t target_faces, std::vector<float>& vout, std::vector<int64_t>& fout,
                            bool verbose=true) {
    const size_t Vin=vin.size()/3, Fin=fin.size()/3;
    std::vector<uint32_t> idx(fin.size());
    for (size_t i=0;i<idx.size();i++) idx[i]=(uint32_t)fin[i];
    std::vector<uint32_t> dst(idx.size());
    float err=0.f;
    // pass 1: topology-preserving quadric collapse (best quality)
    size_t new_ic = meshopt_simplify(dst.data(), idx.data(), idx.size(), vin.data(), Vin, 3*sizeof(float),
                                     target_faces*3, /*target_error*/1.0f, /*options*/0, &err);
    bool used_sloppy = false;
    // pass 2 (fallback): the dual-grid mesh has many non-manifold/border edges that block quality
    // collapses (it stalls ~2x the target) -> meshopt_simplifySloppy ignores topology and reliably
    // hits the target, which is what keeps xatlas tractable. Texture is volume-sampled so the
    // slightly looser silhouette is invisible.
    if (new_ic > target_faces*3 * 5/4) {
        float serr=0.f;
        size_t sic = meshopt_simplifySloppy(dst.data(), idx.data(), idx.size(), vin.data(), Vin, 3*sizeof(float),
                                            target_faces*3, /*target_error*/1.0f, &serr);
        if (sic > 0) { new_ic = sic; err = serr; used_sloppy = true; }
    }
    dst.resize(new_ic);
    // compact: keep only referenced verts, first-seen order
    std::vector<int> remap(Vin, -1); vout.clear(); fout.clear(); fout.reserve(new_ic);
    for (uint32_t oi : dst) {
        if (remap[oi] < 0) { remap[oi]=(int)(vout.size()/3);
            vout.push_back(vin[(size_t)oi*3]); vout.push_back(vin[(size_t)oi*3+1]); vout.push_back(vin[(size_t)oi*3+2]); }
        fout.push_back(remap[oi]);
    }
    if (verbose) printf("[atlas] decimate%s: %zu->%zu faces, %zu->%zu verts (err=%.4f)\n",
                        used_sloppy?"(sloppy)":"", Fin, fout.size()/3, Vin, vout.size()/3, err);
}

struct BakedTexture {
    int tw = 0, th = 0;                 // atlas dimensions (texels)
    std::vector<uint8_t> base_color;    // [tw*th*4] RGBA  (RGB = base_color, A = PBR alpha)
    std::vector<uint8_t> metal_rough;   // [tw*th*3] RGB   (R=0, G=roughness, B=metallic)  per glTF
    std::vector<float> verts;           // [Vout*3]  unwrapped (re-indexed) positions
    std::vector<float> normals;         // [Vout*3]
    std::vector<float> uvs;             // [Vout*2]  normalized [0,1], glTF convention
    std::vector<uint32_t> faces;        // [Fout*3]
    int chart_count = 0, atlas_count = 0;
};

// Sparse 3-D colour outlier cleanup for a decoded PBR surface. Texture-flow occasionally emits
// isolated black/bright voxels inside otherwise coherent cloth or skin; UV-space filtering cannot
// reliably remove those once a chart has been split. Only RGB is touched, and only when a voxel
// differs materially from its occupied 3x3x3-neighbour median.
static inline size_t filter_pbr_rgb_outliers(std::vector<float>& pbr, const std::vector<int32_t>& coords,
                                             float threshold) {
    const size_t N=coords.size()/4;
    if (threshold<=0.f || N==0 || pbr.size()!=N*6) return 0;
    auto key=[](int x,int y,int z)->uint64_t {
        return (uint64_t)(uint32_t)(x+2048) | ((uint64_t)(uint32_t)(y+2048)<<13) |
               ((uint64_t)(uint32_t)(z+2048)<<26);
    };
    std::unordered_map<uint64_t,uint32_t> index;
    index.reserve(N*2);
    for (uint32_t i=0;i<(uint32_t)N;i++)
        index.emplace(key(coords[(size_t)i*4+1],coords[(size_t)i*4+2],coords[(size_t)i*4+3]),i);
    const std::vector<float> src=pbr;
    std::vector<uint8_t> replace(N,0);
    std::vector<std::array<float,3>> med(N);
    #pragma omp parallel for schedule(static)
    for (int64_t ii=0;ii<(int64_t)N;ii++) {
        size_t i=(size_t)ii;
        float vals[3][27]; int n=0;
        const int x=coords[i*4+1],y=coords[i*4+2],z=coords[i*4+3];
        for(int dz=-1;dz<=1;dz++) for(int dy=-1;dy<=1;dy++) for(int dx=-1;dx<=1;dx++) {
            auto it=index.find(key(x+dx,y+dy,z+dz)); if(it==index.end()) continue;
            const float* q=&src[(size_t)it->second*6];
            for(int c=0;c<3;c++) vals[c][n]=q[c];
            n++;
        }
        if (n<7) continue;
        float delta=0.f;
        for(int c=0;c<3;c++) { std::nth_element(vals[c],vals[c]+n/2,vals[c]+n); med[i][c]=vals[c][n/2];
            delta=std::max(delta,std::fabs(src[i*6+c]-med[i][c])); }
        replace[i]=delta>threshold;
    }
    size_t changed=0;
    for(size_t i=0;i<N;i++) if(replace[i]) { for(int c=0;c<3;c++) pbr[i*6+c]=med[i][c]; changed++; }
    return changed;
}

// area-weighted per-vertex normals (matches glb_writer; robust to dual-grid winding)
static inline std::vector<float> vert_normals(const std::vector<float>& v, const std::vector<int64_t>& f) {
    const size_t V=v.size()/3, F=f.size()/3;
    std::vector<float> n(V*3,0.f);
    for (size_t t=0;t<F;t++){ int64_t a=f[t*3],b=f[t*3+1],c=f[t*3+2];
        const float*pa=&v[a*3],*pb=&v[b*3],*pc=&v[c*3];
        float e1[3]={pb[0]-pa[0],pb[1]-pa[1],pb[2]-pa[2]}, e2[3]={pc[0]-pa[0],pc[1]-pa[1],pc[2]-pa[2]};
        float fn[3]={e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0]};
        for (int64_t vi:{a,b,c}) for(int d=0;d<3;d++) n[vi*3+d]+=fn[d]; }
    for (size_t i=0;i<V;i++){ float*p=&n[i*3]; float L=std::sqrt(p[0]*p[0]+p[1]*p[1]+p[2]*p[2]);
        if(L>1e-20f){p[0]/=L;p[1]/=L;p[2]/=L;} else {p[0]=0;p[1]=0;p[2]=1;} }
    return n;
}

static inline std::vector<float> welded_vert_normals(const std::vector<float>& v, const std::vector<uint32_t>& f) {
    const size_t V=v.size()/3, F=f.size()/3;
    std::vector<float> n(V*3,0.f);
    struct Key { int64_t x,y,z; bool operator==(const Key& o) const { return x==o.x && y==o.y && z==o.z; } };
    struct Hash {
        size_t operator()(const Key& k) const {
            uint64_t h=(uint64_t)k.x*0x9E3779B185EBCA87ull;
            h ^= (uint64_t)k.y + 0x9E3779B97F4A7C15ull + (h<<6) + (h>>2);
            h ^= (uint64_t)k.z + 0xC2B2AE3D27D4EB4Full + (h<<6) + (h>>2);
            return (size_t)h;
        }
    };
    const double S = 10000000.0; // weld exact/split copies without merging distinct nearby surface verts.
    std::unordered_map<Key, std::vector<uint32_t>, Hash> groups;
    groups.reserve(V);
    for (uint32_t i=0;i<(uint32_t)V;i++) {
        Key k{(int64_t)llround((double)v[(size_t)i*3+0]*S),
              (int64_t)llround((double)v[(size_t)i*3+1]*S),
              (int64_t)llround((double)v[(size_t)i*3+2]*S)};
        groups[k].push_back(i);
    }
    for (size_t t=0;t<F;t++){ uint32_t a=f[t*3],b=f[t*3+1],c=f[t*3+2];
        const float*pa=&v[(size_t)a*3],*pb=&v[(size_t)b*3],*pc=&v[(size_t)c*3];
        float e1[3]={pb[0]-pa[0],pb[1]-pa[1],pb[2]-pa[2]}, e2[3]={pc[0]-pa[0],pc[1]-pa[1],pc[2]-pa[2]};
        float fn[3]={e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0]};
        for (uint32_t vi:{a,b,c}) {
            Key k{(int64_t)llround((double)v[(size_t)vi*3+0]*S),
                  (int64_t)llround((double)v[(size_t)vi*3+1]*S),
                  (int64_t)llround((double)v[(size_t)vi*3+2]*S)};
            auto it = groups.find(k);
            if (it == groups.end()) continue;
            for (uint32_t wi: it->second) for(int d=0;d<3;d++) n[(size_t)wi*3+d]+=fn[d];
        }
    }
    for (size_t i=0;i<V;i++){ float*p=&n[i*3]; float L=std::sqrt(p[0]*p[0]+p[1]*p[1]+p[2]*p[2]);
        if(L>1e-20f){p[0]/=L;p[1]/=L;p[2]/=L;} else {p[0]=0;p[1]=0;p[2]=1;} }
    return n;
}

// fill invalid (mask==0) texels of a C-channel float atlas from valid neighbours (multi-pass
// dilation; the cheap cv2.inpaint analog used only to seal the UV gutter so bilinear filtering at
// chart edges doesn't sample background). Iterates `iters` rings outward.
static inline void inpaint(std::vector<float>& img, std::vector<uint8_t>& mask, int W, int H, int C, int iters) {
    for (int it=0; it<iters; it++) {
        std::vector<uint8_t> newly; newly.reserve(1024);
        std::vector<int> coords; coords.reserve(1024);
        for (int y=0;y<H;y++) for (int x=0;x<W;x++) {
            if (mask[(size_t)y*W+x]) continue;
            float acc[8]={0}; int cnt=0;
            for (int dy=-1;dy<=1;dy++) for (int dx=-1;dx<=1;dx++){
                if(!dx&&!dy) continue;
                int nx=x+dx,ny=y+dy;
                if(nx<0||ny<0||nx>=W||ny>=H) continue;
                if(mask[(size_t)ny*W+nx]){ const float*s=&img[((size_t)ny*W+nx)*C]; for(int c=0;c<C;c++) acc[c]+=s[c]; cnt++; }
            }
            if(cnt){ float*d=&img[((size_t)y*W+x)*C]; for(int c=0;c<C;c++) d[c]=acc[c]/cnt; coords.push_back(y*W+x); }
        }
        if (coords.empty()) break;
        for (int idx : coords) mask[idx]=1;   // promote after the full pass (ring-by-ring)
    }
}

static inline void dilate_background(std::vector<float>& img, std::vector<uint8_t>& mask, int W, int H, int C) {
    if (W <= 0 || H <= 0 || C <= 0) return;
    std::queue<int> q;
    for (int p=0; p<W*H; p++) if (mask[(size_t)p]) q.push(p);
    while (!q.empty()) {
        int p=q.front(); q.pop();
        int x=p%W, y=p/W;
        const float* src=&img[(size_t)p*C];
        for (int dy=-1; dy<=1; dy++) for (int dx=-1; dx<=1; dx++) {
            if (!dx && !dy) continue;
            int nx=x+dx, ny=y+dy;
            if (nx<0 || ny<0 || nx>=W || ny>=H) continue;
            int n=ny*W+nx;
            if (mask[(size_t)n]) continue;
            float* dst=&img[(size_t)n*C];
            for (int c=0; c<C; c++) dst[c]=src[c];
            mask[(size_t)n]=1;
            q.push(n);
        }
    }
}

// Small native approximation of OpenCV's Telea inpaint: advance invalid pixels from the
// valid boundary by distance and estimate each new texel from an already-known radius
// neighbourhood. This is intentionally bounded by fill_rings for atlas gutters; it avoids
// turning the whole background into meaningful texture while giving seam pixels the same
// distance-ordered treatment that hides tiny cracks in the Python bake.
static inline void inpaint_telea(std::vector<float>& img, std::vector<uint8_t>& mask,
        int W, int H, int C, int fill_rings, int radius) {
    if (W <= 0 || H <= 0 || C <= 0 || fill_rings <= 0 || radius <= 0) return;
    const int N = W * H;
    const float INF = std::numeric_limits<float>::infinity();
    std::vector<float> dist((size_t)N, INF);
    std::vector<uint8_t> state((size_t)N, 2); // 0 known, 1 band, 2 unknown
    struct Node { float d; int p; bool operator<(const Node& o) const { return d > o.d; } };
    std::priority_queue<Node> pq;
    auto push_band = [&](int p, float d) {
        if (state[(size_t)p] == 0 || d >= dist[(size_t)p]) return;
        state[(size_t)p] = 1; dist[(size_t)p] = d; pq.push({d, p});
    };
    for (int p=0;p<N;p++) {
        if (mask[(size_t)p]) { state[(size_t)p]=0; dist[(size_t)p]=0.f; }
    }
    for (int y=0;y<H;y++) for (int x=0;x<W;x++) {
        int p=y*W+x;
        if (state[(size_t)p] != 0) continue;
        for (int dy=-1;dy<=1;dy++) for (int dx=-1;dx<=1;dx++) {
            if(!dx&&!dy) continue;
            int nx=x+dx, ny=y+dy;
            if(nx<0||ny<0||nx>=W||ny>=H) continue;
            int q=ny*W+nx;
            if (state[(size_t)q] == 2) push_band(q, dx && dy ? 1.41421356f : 1.f);
        }
    }
    const float maxd = (float)fill_rings + 0.01f;
    while (!pq.empty()) {
        Node n = pq.top(); pq.pop();
        int p = n.p;
        if (state[(size_t)p] == 0 || n.d != dist[(size_t)p]) continue;
        if (n.d > maxd) break;
        int x=p%W, y=p/W;

        float gx=0.f, gy=0.f;
        if (x > 0 && dist[(size_t)p-1] < INF) gx += dist[(size_t)p] - dist[(size_t)p-1];
        if (x+1 < W && dist[(size_t)p+1] < INF) gx += dist[(size_t)p+1] - dist[(size_t)p];
        if (y > 0 && dist[(size_t)p-W] < INF) gy += dist[(size_t)p] - dist[(size_t)p-W];
        if (y+1 < H && dist[(size_t)p+W] < INF) gy += dist[(size_t)p+W] - dist[(size_t)p];
        float gl=std::sqrt(gx*gx+gy*gy);
        if (gl < 1e-6f) { gx=0.f; gy=1.f; gl=1.f; }
        gx/=gl; gy/=gl;

        float acc[8]={0}; float wsum=0.f;
        for (int dy=-radius; dy<=radius; dy++) for (int dx=-radius; dx<=radius; dx++) {
            if (!dx && !dy) continue;
            int nx=x+dx, ny=y+dy;
            if(nx<0||ny<0||nx>=W||ny>=H) continue;
            int q=ny*W+nx;
            if (state[(size_t)q] != 0) continue;
            float d2=(float)(dx*dx+dy*dy);
            if (d2 < 1e-6f || d2 > (float)(radius*radius)) continue;
            float dl=std::sqrt(d2);
            float dir=std::fabs((-(float)dx/dl)*gx + (-(float)dy/dl)*gy);
            float lev=1.f/(1.f + std::fabs(dist[(size_t)p]-dist[(size_t)q]));
            float w=(0.001f + dir) * lev / d2;
            const float* s=&img[(size_t)q*C];
            for (int c=0;c<C;c++) acc[c]+=w*s[c];
            wsum+=w;
        }
        if (wsum <= 1e-20f) {
            for (int dy=-1;dy<=1;dy++) for (int dx=-1;dx<=1;dx++) {
                int nx=x+dx, ny=y+dy;
                if(nx<0||ny<0||nx>=W||ny>=H) continue;
                int q=ny*W+nx;
                if (state[(size_t)q] != 0) continue;
                const float* s=&img[(size_t)q*C];
                for (int c=0;c<C;c++) acc[c]+=s[c];
                wsum+=1.f;
            }
        }
        if (wsum > 1e-20f) {
            float* d=&img[(size_t)p*C];
            for (int c=0;c<C;c++) d[c]=acc[c]/wsum;
        }
        state[(size_t)p]=0; mask[(size_t)p]=1;
        for (int dy=-1;dy<=1;dy++) for (int dx=-1;dx<=1;dx++) {
            if(!dx&&!dy) continue;
            int nx=x+dx, ny=y+dy;
            if(nx<0||ny<0||nx>=W||ny>=H) continue;
            int q=ny*W+nx;
            if (state[(size_t)q] == 0) continue;
            push_band(q, n.d + (dx && dy ? 1.41421356f : 1.f));
        }
    }
}

static inline uint8_t u8(float v){ v=v*255.f+0.5f; return (uint8_t)(v<0?0:(v>255?255:v)); }

static inline std::vector<uint8_t> resize_u8_bilinear(const std::vector<uint8_t>& src,
        int sw, int sh, int dw, int dh, int C) {
    std::vector<uint8_t> dst((size_t)dw*dh*C);
    for (int y=0;y<dh;y++) {
        float sy = ((y + 0.5f) * sh / (float)dh) - 0.5f;
        int y0 = (int)std::floor(sy), y1 = y0 + 1;
        float fy = sy - y0;
        y0 = std::max(0, std::min(sh-1, y0));
        y1 = std::max(0, std::min(sh-1, y1));
        for (int x=0;x<dw;x++) {
            float sx = ((x + 0.5f) * sw / (float)dw) - 0.5f;
            int x0 = (int)std::floor(sx), x1 = x0 + 1;
            float fx = sx - x0;
            x0 = std::max(0, std::min(sw-1, x0));
            x1 = std::max(0, std::min(sw-1, x1));
            for (int c=0;c<C;c++) {
                float a = src[((size_t)y0*sw+x0)*C+c] * (1.f-fx) + src[((size_t)y0*sw+x1)*C+c] * fx;
                float b = src[((size_t)y1*sw+x0)*C+c] * (1.f-fx) + src[((size_t)y1*sw+x1)*C+c] * fx;
                dst[((size_t)y*dw+x)*C+c] = u8((a * (1.f-fy) + b * fy) / 255.f);
            }
        }
    }
    return dst;
}

static inline std::vector<uint8_t> resize_u8_area(const std::vector<uint8_t>& src,
        int sw, int sh, int dw, int dh, int C) {
    if (dw >= sw || dh >= sh) return resize_u8_bilinear(src, sw, sh, dw, dh, C);
    std::vector<uint8_t> dst((size_t)dw*dh*C);
    for (int y=0;y<dh;y++) {
        double y0 = (double)y * sh / dh, y1 = (double)(y+1) * sh / dh;
        int iy0 = (int)std::floor(y0), iy1 = (int)std::ceil(y1);
        for (int x=0;x<dw;x++) {
            double x0 = (double)x * sw / dw, x1 = (double)(x+1) * sw / dw;
            int ix0 = (int)std::floor(x0), ix1 = (int)std::ceil(x1);
            double acc[8]={0}, wsum=0;
            for (int sy=iy0; sy<iy1; sy++) {
                if (sy < 0 || sy >= sh) continue;
                double wy = std::max(0.0, std::min(y1, (double)sy+1.0) - std::max(y0, (double)sy));
                if (wy <= 0) continue;
                for (int sx=ix0; sx<ix1; sx++) {
                    if (sx < 0 || sx >= sw) continue;
                    double wx = std::max(0.0, std::min(x1, (double)sx+1.0) - std::max(x0, (double)sx));
                    double w = wx * wy;
                    if (w <= 0) continue;
                    const uint8_t* p=&src[((size_t)sy*sw+sx)*C];
                    for (int c=0;c<C;c++) acc[c] += w * p[c];
                    wsum += w;
                }
            }
            for (int c=0;c<C;c++) dst[((size_t)y*dw+x)*C+c] = (uint8_t)std::max(0.0, std::min(255.0, acc[c]/std::max(wsum,1e-12)+0.5));
        }
    }
    return dst;
}

static inline void blur_u8_gaussian_rgb(std::vector<uint8_t>& img, int W, int H, int C, float sigma) {
    if (sigma <= 0.f || W <= 1 || H <= 1 || C < 3) return;
    int r = std::max(1, (int)std::ceil(sigma * 3.f));
    std::vector<float> k((size_t)r*2+1);
    float sum = 0.f;
    for (int i=-r;i<=r;i++) {
        float v = std::exp(-(float)(i*i) / (2.f*sigma*sigma));
        k[(size_t)(i+r)] = v; sum += v;
    }
    for (float& v:k) v /= sum;
    std::vector<uint8_t> tmp(img.size());
    for (int y=0;y<H;y++) for (int x=0;x<W;x++) {
        for (int c=0;c<C;c++) {
            if (c >= 3) { tmp[((size_t)y*W+x)*C+c] = img[((size_t)y*W+x)*C+c]; continue; }
            float acc=0.f;
            for (int dx=-r;dx<=r;dx++) {
                int xx = std::max(0, std::min(W-1, x+dx));
                acc += k[(size_t)(dx+r)] * img[((size_t)y*W+xx)*C+c];
            }
            tmp[((size_t)y*W+x)*C+c] = (uint8_t)std::max(0.f, std::min(255.f, acc + 0.5f));
        }
    }
    for (int y=0;y<H;y++) for (int x=0;x<W;x++) {
        for (int c=0;c<C;c++) {
            if (c >= 3) { img[((size_t)y*W+x)*C+c] = tmp[((size_t)y*W+x)*C+c]; continue; }
            float acc=0.f;
            for (int dy=-r;dy<=r;dy++) {
                int yy = std::max(0, std::min(H-1, y+dy));
                acc += k[(size_t)(dy+r)] * tmp[((size_t)yy*W+x)*C+c];
            }
            img[((size_t)y*W+x)*C+c] = (uint8_t)std::max(0.f, std::min(255.f, acc + 0.5f));
        }
    }
}

// Remove isolated generator speckle without blending across UV charts.  Unlike the optional
// full-atlas Gaussian blur, this only samples texels that were rasterised from the SAME connected
// UV chart; that remains true even for the direct reference packer's zero-pixel chart padding.
// A 3x3 median preserves material boundaries (buttons, hair silhouettes) while rejecting one-pixel
// dark/light freckles.
static inline void median_surface_rgb(std::vector<float>& atl, const std::vector<uint8_t>& surface_mask,
                                      const std::vector<int32_t>& surface_chart,
                                      int W, int H, int radius) {
    if (radius <= 0 || W <= 0 || H <= 0 || atl.size() != (size_t)W*H*6 ||
        surface_mask.size() != (size_t)W*H || surface_chart.size() != (size_t)W*H) return;
    const std::vector<float> src=atl;
    #pragma omp parallel for schedule(static)
    for (int y=0;y<H;y++) for (int x=0;x<W;x++) {
        const size_t p=(size_t)y*W+x;
        if (!surface_mask[p]) continue;
        float values[81];
        for (int c=0;c<3;c++) {
            int n=0;
            for (int dy=-radius;dy<=radius;dy++) for (int dx=-radius;dx<=radius;dx++) {
                int nx=x+dx, ny=y+dy;
                if (nx<0 || ny<0 || nx>=W || ny>=H) continue;
                size_t q=(size_t)ny*W+nx;
                if (surface_mask[q] && surface_chart[q] == surface_chart[p]) values[n++]=src[q*6+c];
            }
            // Do not distort isolated slivers where a neighbourhood is not meaningful.
            if (n >= 5) {
                std::nth_element(values, values+n/2, values+n);
                atl[p*6+c]=values[n/2];
            }
        }
    }
}

// Bake. in_verts [Vin*3] in [-0.5,0.5]; in_faces int64 [Fin*3]; PBR volume feats [N*6] + coords
// [N*4] (b,x,y,z) at grid `grid_res`; texture_size target; padding texels for the gutter.
// Normal-cone CHART PRE-CLUSTER (lap-17 P1). xatlas' ComputeCharts segmentation is pathologically
// slow on the QEM remesh (its ~50k non-manifold edges blow up the half-edge build → minutes / hangs).
// Since the texture bakes VOLUMETRICALLY (per-texel 3D position → grid_sample), UV layout quality is
// irrelevant — we only need disjoint, low-distortion islands to pack. So we build the charts
// ourselves: region-grow faces over edge adjacency while their normals stay within a cone of the
// chart seed normal, planar-project each chart, and hand xatlas pre-made UV islands via AddUvMesh +
// faceMaterialData → it only PACKS (seconds), skipping segmentation entirely. Robust to non-manifold
// (our own face adjacency). Returns per-uv-vertex 2D coords + uv→orig-vertex map + remapped faces +
// per-face chart/material id. cone_cos = cos(half-angle); lower = bigger charts/fewer islands.
static inline void precluster_charts(const std::vector<float>& V, const std::vector<int64_t>& F,
        float cone_cos, std::vector<float>& uv, std::vector<int>& uv2orig,
        std::vector<uint32_t>& uvfaces, std::vector<uint32_t>& facemat, int& n_charts) {
    const int64_t Nf = (int64_t)F.size()/3;
    // face normals
    std::vector<float> fn((size_t)Nf*3);
    for (int64_t t=0;t<Nf;t++){ const float*a=&V[F[t*3]*3],*b=&V[F[t*3+1]*3],*c=&V[F[t*3+2]*3];
        float e1[3]={b[0]-a[0],b[1]-a[1],b[2]-a[2]}, e2[3]={c[0]-a[0],c[1]-a[1],c[2]-a[2]};
        float n[3]={e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0]};
        float L=std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]); if(L<1e-20f)L=1;
        fn[t*3]=n[0]/L; fn[t*3+1]=n[1]/L; fn[t*3+2]=n[2]/L; }
    // edge -> faces (for face adjacency)
    std::unordered_map<int64_t,std::vector<int>> e2f; e2f.reserve((size_t)Nf*3);
    auto ukey=[](int64_t a,int64_t b){ int64_t lo=a<b?a:b,hi=a<b?b:a; return (lo<<32)|(uint32_t)hi; };
    for (int64_t t=0;t<Nf;t++){ int64_t v[3]={F[t*3],F[t*3+1],F[t*3+2]};
        for(int e=0;e<3;e++) e2f[ukey(v[e],v[(e+1)%3])].push_back((int)t); }
    // region-grow charts (BFS). A face joins iff within the cone of the chart's SEED normal — the SAME
    // normal used as the projection plane below. So every face in a chart projects with area scale
    // >= cone_cos (cos of the cone half-angle): at 40° that's 0.77 (no degenerate slivers → no UV
    // streaks); at 80° it'd be 0.17 (the garbled-atlas bug). Keep the cone tight (~40°).
    std::vector<int> chart((size_t)Nf,-1); std::vector<int> stack; n_charts=0;
    std::vector<std::array<float,3>> seedn;   // each chart's seed (= projection) normal
    for (int64_t s=0;s<Nf;s++){ if(chart[s]>=0) continue;
        int c=n_charts++; const float* sn=&fn[s*3]; seedn.push_back({sn[0],sn[1],sn[2]});
        chart[s]=c; stack.push_back((int)s);
        while(!stack.empty()){ int t=stack.back(); stack.pop_back(); int64_t v[3]={F[t*3],F[t*3+1],F[t*3+2]};
            for(int e=0;e<3;e++){ for(int nf: e2f[ukey(v[e],v[(e+1)%3])]){ if(chart[nf]>=0) continue;
                if (fn[nf*3]*sn[0]+fn[nf*3+1]*sn[1]+fn[nf*3+2]*sn[2] >= cone_cos){ chart[nf]=c; stack.push_back(nf); } } } }
    }
    // planar-project each chart onto its SEED-normal plane; new uv-vertex per (chart, orig-vertex)
    std::vector<std::array<float,6>> basis((size_t)n_charts);   // tangent(3)+bitangent(3)
    for (int c=0;c<n_charts;c++){ double n[3]={seedn[c][0],seedn[c][1],seedn[c][2]};
        double L=std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]); if(L<1e-20)L=1; n[0]/=L;n[1]/=L;n[2]/=L;
        double ref[3]; if (std::fabs(n[1])<0.9){ ref[0]=0;ref[1]=1;ref[2]=0; } else { ref[0]=1;ref[1]=0;ref[2]=0; }
        double t1[3]={ ref[1]*n[2]-ref[2]*n[1], ref[2]*n[0]-ref[0]*n[2], ref[0]*n[1]-ref[1]*n[0] };
        double tl=std::sqrt(t1[0]*t1[0]+t1[1]*t1[1]+t1[2]*t1[2]); if(tl<1e-20)tl=1; t1[0]/=tl;t1[1]/=tl;t1[2]/=tl;
        double b1[3]={ n[1]*t1[2]-n[2]*t1[1], n[2]*t1[0]-n[0]*t1[2], n[0]*t1[1]-n[1]*t1[0] };
        basis[c]={(float)t1[0],(float)t1[1],(float)t1[2],(float)b1[0],(float)b1[1],(float)b1[2]};
    }
    uv.clear(); uv2orig.clear(); uvfaces.resize((size_t)Nf*3); facemat.resize((size_t)Nf);
    std::unordered_map<int64_t,int> seen; seen.reserve((size_t)Nf*3);
    for (int64_t t=0;t<Nf;t++){ int c=chart[t]; facemat[t]=(uint32_t)c; const auto&bs=basis[c];
        for(int k=0;k<3;k++){ int64_t ov=F[t*3+k]; int64_t key=((int64_t)c<<34)^ov;
            auto it=seen.find(key); int id;
            if(it==seen.end()){ id=(int)uv2orig.size(); seen[key]=id; uv2orig.push_back((int)ov);
                const float* p=&V[ov*3];
                uv.push_back(p[0]*bs[0]+p[1]*bs[1]+p[2]*bs[2]);
                uv.push_back(p[0]*bs[3]+p[1]*bs[4]+p[2]*bs[5]);
            } else id=it->second;
            uvfaces[t*3+k]=(uint32_t)id;
        }
    }
}

// Same normal-cone clustering as precluster_charts, but instead of planar-projecting each chart,
// split vertices on chart boundaries and feed the resulting disconnected islands back to xatlas
// AddMesh. This removes the global non-manifold topology that made xatlas fragment/hang, while still
// letting xatlas do its own conformal parameterization per island (LSCM/piecewise) instead of our
// folding planar UVs.
static inline void precluster_split_mesh(const std::vector<float>& V, const std::vector<int64_t>& F,
        float cone_cos, std::vector<float>& out_v, std::vector<uint32_t>& out_f,
        std::vector<int>& split2orig, int& n_charts) {
    const int64_t Nf = (int64_t)F.size()/3;
    std::vector<float> fn((size_t)Nf*3);
    for (int64_t t=0;t<Nf;t++){ const float*a=&V[F[t*3]*3],*b=&V[F[t*3+1]*3],*c=&V[F[t*3+2]*3];
        float e1[3]={b[0]-a[0],b[1]-a[1],b[2]-a[2]}, e2[3]={c[0]-a[0],c[1]-a[1],c[2]-a[2]};
        float n[3]={e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0]};
        float L=std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]); if(L<1e-20f)L=1;
        fn[t*3]=n[0]/L; fn[t*3+1]=n[1]/L; fn[t*3+2]=n[2]/L; }

    std::unordered_map<int64_t,std::vector<int>> e2f; e2f.reserve((size_t)Nf*3);
    auto ukey=[](int64_t a,int64_t b){ int64_t lo=a<b?a:b,hi=a<b?b:a; return (lo<<32)|(uint32_t)hi; };
    for (int64_t t=0;t<Nf;t++){ int64_t v[3]={F[t*3],F[t*3+1],F[t*3+2]};
        for(int e=0;e<3;e++) e2f[ukey(v[e],v[(e+1)%3])].push_back((int)t); }

    std::vector<int> chart((size_t)Nf,-1), stack; n_charts=0;
    for (int64_t s=0;s<Nf;s++){ if(chart[s]>=0) continue;
        int c=n_charts++; const float* sn=&fn[s*3];
        chart[s]=c; stack.push_back((int)s);
        while(!stack.empty()){ int t=stack.back(); stack.pop_back(); int64_t v[3]={F[t*3],F[t*3+1],F[t*3+2]};
            for(int e=0;e<3;e++){ for(int nf: e2f[ukey(v[e],v[(e+1)%3])]){ if(chart[nf]>=0) continue;
                if (fn[nf*3]*sn[0]+fn[nf*3+1]*sn[1]+fn[nf*3+2]*sn[2] >= cone_cos){ chart[nf]=c; stack.push_back(nf); } } } }
    }

    out_v.clear(); out_f.resize((size_t)Nf*3); split2orig.clear();
    out_v.reserve(V.size()); split2orig.reserve(V.size()/3);
    std::unordered_map<int64_t,int> seen; seen.reserve((size_t)Nf*3);
    for (int64_t t=0;t<Nf;t++){ int c=chart[t];
        for(int k=0;k<3;k++){ int64_t ov=F[t*3+k]; int64_t key=((int64_t)c<<34)^ov;
            auto it=seen.find(key); int id;
            if(it==seen.end()){ id=(int)split2orig.size(); seen[key]=id; split2orig.push_back((int)ov);
                out_v.push_back(V[(size_t)ov*3]); out_v.push_back(V[(size_t)ov*3+1]); out_v.push_back(V[(size_t)ov*3+2]);
            } else id=it->second;
            out_f[(size_t)t*3+k]=(uint32_t)id;
        }
    }
}

struct ClusterMesh {
    std::vector<float> verts;
    std::vector<uint32_t> faces;
    std::vector<int> vmap;
};

struct FaceAdj {
    int f;
    float len;
};

static inline int compress_chart_ids(std::vector<int>& chart) {
    std::unordered_map<int,int> remap; remap.reserve(chart.size());
    int n=0;
    for (int& c: chart) {
        auto it=remap.find(c);
        if (it==remap.end()) { remap[c]=n; c=n++; }
        else c=it->second;
    }
    return n;
}

static inline void split_disconnected_charts(const std::vector<std::vector<FaceAdj>>& adj, std::vector<int>& chart) {
    const int F=(int)chart.size();
    std::vector<char> seen(F,0);
    std::vector<int> out(F,-1), stack;
    int next=0;
    for (int s=0;s<F;s++) {
        if (seen[s]) continue;
        int old=chart[s];
        seen[s]=1; out[s]=next; stack.push_back(s);
        while(!stack.empty()) {
            int f=stack.back(); stack.pop_back();
            for (const FaceAdj& a: adj[f]) if (!seen[a.f] && chart[a.f]==old) {
                seen[a.f]=1; out[a.f]=next; stack.push_back(a.f);
            }
        }
        next++;
    }
    chart.swap(out);
}

static inline std::vector<ClusterMesh> build_cluster_meshes_from_ids(const std::vector<float>& V, const std::vector<int64_t>& F,
        const std::vector<int>& chart, int n_charts) {
    const int64_t Nf=(int64_t)F.size()/3;
    std::vector<ClusterMesh> out((size_t)n_charts);
    std::vector<std::unordered_map<int64_t,uint32_t>> remap((size_t)n_charts);
    for (int64_t t=0;t<Nf;t++){ int c=chart[(size_t)t]; ClusterMesh& cm=out[(size_t)c]; auto& rm=remap[(size_t)c];
        for(int k=0;k<3;k++){ int64_t ov=F[t*3+k]; auto it=rm.find(ov); uint32_t id;
            if (it==rm.end()) {
                id=(uint32_t)cm.vmap.size(); rm[ov]=id; cm.vmap.push_back((int)ov);
                cm.verts.push_back(V[(size_t)ov*3]); cm.verts.push_back(V[(size_t)ov*3+1]); cm.verts.push_back(V[(size_t)ov*3+2]);
            } else id=it->second;
            cm.faces.push_back(id);
        }
    }
    out.erase(std::remove_if(out.begin(), out.end(), [](const ClusterMesh& c){ return c.faces.empty(); }), out.end());
    return out;
}

static inline std::vector<ClusterMesh> precluster_meshes_cumesh_cpu(const std::vector<float>& V, const std::vector<int64_t>& F,
        float threshold_rad, int refine_iterations=100, int global_iterations=3,
        float smooth_strength=1.f, float area_weight=0.1f, float perim_weight=0.0001f) {
    const int Nf=(int)F.size()/3;
    std::vector<float> fn((size_t)Nf*3), farea((size_t)Nf);
    for (int t=0;t<Nf;t++){ const float*a=&V[(size_t)F[t*3]*3],*b=&V[(size_t)F[t*3+1]*3],*c=&V[(size_t)F[t*3+2]*3];
        float e1[3]={b[0]-a[0],b[1]-a[1],b[2]-a[2]}, e2[3]={c[0]-a[0],c[1]-a[1],c[2]-a[2]};
        float n[3]={e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0]};
        float L=std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]); farea[(size_t)t]=0.5f*L; if(L<1e-20f)L=1;
        fn[(size_t)t*3]=n[0]/L; fn[(size_t)t*3+1]=n[1]/L; fn[(size_t)t*3+2]=n[2]/L; }

    std::vector<std::vector<FaceAdj>> adj((size_t)Nf);
    std::unordered_map<int64_t,std::vector<int>> e2f; e2f.reserve((size_t)Nf*3);
    auto ukey=[](int64_t a,int64_t b){ int64_t lo=a<b?a:b,hi=a<b?b:a; return (lo<<32)|(uint32_t)hi; };
    for (int t=0;t<Nf;t++){ int64_t v[3]={F[(size_t)t*3],F[(size_t)t*3+1],F[(size_t)t*3+2]};
        for(int e=0;e<3;e++) e2f[ukey(v[e],v[(e+1)%3])].push_back(t); }
    for (const auto& kv: e2f) {
        const std::vector<int>& fs=kv.second;
        if (fs.size()<2) continue;
        int64_t lo=(int64_t)(kv.first>>32), hi=(int64_t)(uint32_t)kv.first;
        const float* a=&V[(size_t)lo*3]; const float* b=&V[(size_t)hi*3];
        float dx=a[0]-b[0],dy=a[1]-b[1],dz=a[2]-b[2]; float len=std::sqrt(dx*dx+dy*dy+dz*dz);
        for (size_t i=0;i<fs.size();i++) for (size_t j=i+1;j<fs.size();j++) {
            adj[(size_t)fs[i]].push_back({fs[j],len});
            adj[(size_t)fs[j]].push_back({fs[i],len});
        }
    }

    std::vector<int> chart((size_t)Nf);
    for (int i=0;i<Nf;i++) chart[(size_t)i]=i;

    struct Stats { float ax=0,ay=0,az=1,half=0,area=0,perim=0; };
    auto compute_stats = [&](int C, std::vector<Stats>& st, std::unordered_map<uint64_t,float>* edge_len) {
        st.assign((size_t)C, Stats{});
        std::vector<float> sx((size_t)C,0), sy((size_t)C,0), sz((size_t)C,0);
        for (int f=0; f<Nf; f++) { int c=chart[(size_t)f]; float a=farea[(size_t)f];
            sx[(size_t)c]+=fn[(size_t)f*3]*a; sy[(size_t)c]+=fn[(size_t)f*3+1]*a; sz[(size_t)c]+=fn[(size_t)f*3+2]*a; st[(size_t)c].area+=a; }
        for (int c=0;c<C;c++) { float L=std::sqrt(sx[c]*sx[c]+sy[c]*sy[c]+sz[c]*sz[c]);
            if (L>1e-20f){ st[c].ax=sx[c]/L; st[c].ay=sy[c]/L; st[c].az=sz[c]/L; } }
        for (int f=0; f<Nf; f++) { int c=chart[(size_t)f];
            float d=std::max(-1.f,std::min(1.f, st[(size_t)c].ax*fn[(size_t)f*3]+st[(size_t)c].ay*fn[(size_t)f*3+1]+st[(size_t)c].az*fn[(size_t)f*3+2]));
            st[(size_t)c].half=std::max(st[(size_t)c].half, std::acos(d)); }
        if (edge_len) {
            edge_len->clear(); edge_len->reserve((size_t)Nf*2);
            for (int f=0; f<Nf; f++) for (const FaceAdj& a: adj[(size_t)f]) if (f<a.f) {
                int c0=chart[(size_t)f], c1=chart[(size_t)a.f]; if(c0==c1) continue;
                int lo=std::min(c0,c1), hi=std::max(c0,c1);
                (*edge_len)[((uint64_t)(uint32_t)lo<<32)|(uint32_t)hi] += a.len;
            }
            for (auto& kv:*edge_len) { int c0=(int)(kv.first>>32), c1=(int)(uint32_t)kv.first; st[(size_t)c0].perim+=kv.second; st[(size_t)c1].perim+=kv.second; }
        }
    };
    auto merged_half = [](const Stats& a, const Stats& b) {
        float dot=std::max(-1.f,std::min(1.f,a.ax*b.ax+a.ay*b.ay+a.az*b.az));
        float axis_angle=std::acos(dot);
        float low=std::min(-a.half, axis_angle-b.half);
        float high=std::max(a.half, axis_angle+b.half);
        return (high-low)*0.5f;
    };

    int C=Nf;
    for (int gi=0; gi<global_iterations; gi++) {
        while (true) {
            std::vector<Stats> st; std::unordered_map<uint64_t,float> edges;
            compute_stats(C, st, &edges);
            if (edges.empty()) break;
            std::vector<uint64_t> best((size_t)C, UINT64_MAX);
            std::vector<float> ecost; ecost.reserve(edges.size());
            std::vector<uint64_t> ekeys; ekeys.reserve(edges.size());
            for (auto& kv: edges) {
                int c0=(int)(kv.first>>32), c1=(int)(uint32_t)kv.first;
                float new_area=st[(size_t)c0].area+st[(size_t)c1].area;
                float new_perim=st[(size_t)c0].perim+st[(size_t)c1].perim-2.f*kv.second;
                float cost=merged_half(st[(size_t)c0],st[(size_t)c1]) + area_weight*new_area + perim_weight*(new_perim*new_perim/std::max(new_area,1e-20f));
                uint32_t ei=(uint32_t)ecost.size(); ecost.push_back(cost); ekeys.push_back(kv.first);
                uint32_t cv; std::memcpy(&cv,&cost,4); uint64_t packed=((uint64_t)cv<<32)|ei;
                if (packed<best[(size_t)c0]) best[(size_t)c0]=packed;
                if (packed<best[(size_t)c1]) best[(size_t)c1]=packed;
            }
            std::vector<int> parent((size_t)C); for(int i=0;i<C;i++) parent[(size_t)i]=i;
            int merges=0;
            for (uint32_t ei=0; ei<ecost.size(); ei++) {
                if (ecost[ei] > threshold_rad) continue;
                int c0=(int)(ekeys[ei]>>32), c1=(int)(uint32_t)ekeys[ei];
                uint32_t cv; std::memcpy(&cv,&ecost[ei],4); uint64_t packed=((uint64_t)cv<<32)|ei;
                if (best[(size_t)c0]==packed && best[(size_t)c1]==packed) { parent[(size_t)c1]=c0; merges++; }
            }
            if (!merges) break;
            for (int& c: chart) while(parent[(size_t)c]!=c) c=parent[(size_t)c];
            C=compress_chart_ids(chart);
        }
        for (int ri=0; ri<refine_iterations; ri++) {
            std::vector<Stats> st; compute_stats(C, st, nullptr);
            std::vector<int> next=chart;
            for (int f=0; f<Nf; f++) {
                int cand[16]; float smooth[16]; int nc=1; cand[0]=chart[(size_t)f]; smooth[0]=0;
                for (const FaceAdj& a: adj[(size_t)f]) {
                    int c=chart[(size_t)a.f], idx=-1; for(int i=0;i<nc;i++) if(cand[i]==c){idx=i;break;}
                    if(idx<0 && nc<16){idx=nc; cand[nc]=c; smooth[nc]=0; nc++;}
                    if(idx>=0) smooth[idx]+=a.len;
                }
                int bestc=chart[(size_t)f]; float bests=-1e9f;
                for(int i=0;i<nc;i++){ const Stats& s=st[(size_t)cand[i]];
                    float geo=s.ax*fn[(size_t)f*3]+s.ay*fn[(size_t)f*3+1]+s.az*fn[(size_t)f*3+2]; if(geo<=0) continue;
                    float score=geo + smooth_strength*smooth[i];
                    if(score>bests+1e-5f || (std::fabs(score-bests)<=1e-5f && cand[i]<bestc)){ bests=score; bestc=cand[i]; }
                }
                next[(size_t)f]=bestc;
            }
            chart.swap(next); C=compress_chart_ids(chart);
        }
        split_disconnected_charts(adj, chart);
        C=compress_chart_ids(chart);
    }
    return build_cluster_meshes_from_ids(V, F, chart, C);
}

static inline std::vector<ClusterMesh> precluster_meshes(const std::vector<float>& V, const std::vector<int64_t>& F,
        float cone_cos) {
    const int64_t Nf = (int64_t)F.size()/3;
    std::vector<float> fn((size_t)Nf*3);
    for (int64_t t=0;t<Nf;t++){ const float*a=&V[F[t*3]*3],*b=&V[F[t*3+1]*3],*c=&V[F[t*3+2]*3];
        float e1[3]={b[0]-a[0],b[1]-a[1],b[2]-a[2]}, e2[3]={c[0]-a[0],c[1]-a[1],c[2]-a[2]};
        float n[3]={e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0]};
        float L=std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]); if(L<1e-20f)L=1;
        fn[t*3]=n[0]/L; fn[t*3+1]=n[1]/L; fn[t*3+2]=n[2]/L; }

    std::unordered_map<int64_t,std::vector<int>> e2f; e2f.reserve((size_t)Nf*3);
    auto ukey=[](int64_t a,int64_t b){ int64_t lo=a<b?a:b,hi=a<b?b:a; return (lo<<32)|(uint32_t)hi; };
    for (int64_t t=0;t<Nf;t++){ int64_t v[3]={F[t*3],F[t*3+1],F[t*3+2]};
        for(int e=0;e<3;e++) e2f[ukey(v[e],v[(e+1)%3])].push_back((int)t); }

    std::vector<int> chart((size_t)Nf,-1), stack; int n_charts=0;
    for (int64_t s=0;s<Nf;s++){ if(chart[s]>=0) continue;
        int c=n_charts++; const float* sn=&fn[s*3];
        chart[s]=c; stack.push_back((int)s);
        while(!stack.empty()){ int t=stack.back(); stack.pop_back(); int64_t v[3]={F[t*3],F[t*3+1],F[t*3+2]};
            for(int e=0;e<3;e++){ for(int nf: e2f[ukey(v[e],v[(e+1)%3])]){ if(chart[nf]>=0) continue;
                if (fn[nf*3]*sn[0]+fn[nf*3+1]*sn[1]+fn[nf*3+2]*sn[2] >= cone_cos){ chart[nf]=c; stack.push_back(nf); } } } }
    }

    std::vector<ClusterMesh> out((size_t)n_charts);
    std::vector<std::unordered_map<int64_t,uint32_t>> remap((size_t)n_charts);
    for (int64_t t=0;t<Nf;t++){ int c=chart[t]; ClusterMesh& cm=out[(size_t)c]; auto& rm=remap[(size_t)c];
        for(int k=0;k<3;k++){ int64_t ov=F[t*3+k]; auto it=rm.find(ov); uint32_t id;
            if (it==rm.end()) {
                id=(uint32_t)cm.vmap.size(); rm[ov]=id; cm.vmap.push_back((int)ov);
                cm.verts.push_back(V[(size_t)ov*3]); cm.verts.push_back(V[(size_t)ov*3+1]); cm.verts.push_back(V[(size_t)ov*3+2]);
            } else id=it->second;
            cm.faces.push_back(id);
        }
    }
    out.erase(std::remove_if(out.begin(), out.end(), [](const ClusterMesh& c){ return c.faces.empty(); }), out.end());
    return out;
}

static inline BakedTexture bake(const std::vector<float>& in_verts0, const std::vector<int64_t>& in_faces0,
                                const std::vector<float>& pbr_feats, const std::vector<int32_t>& pbr_coords,
                                int grid_res, int texture_size, int decimate_target_faces=0,
                                int padding=4, bool verbose=true, int sample_fallback_r=0,
                                bool precluster=false, float cone_deg=55.f,
                                // lap-18 BVH-reproject: if reproject, snap each texel onto the DENSE
                                // outer-shell mesh (closest-pt-on-tri + front-face reject) and barycentric
                                // -interp dense_attr (6-ch PBR per dense vert) instead of grid_sample'ing
                                // the volume at the raw (possibly interior) rasterised position.
                                const std::vector<float>* dense_verts=nullptr,
                                const std::vector<int64_t>* dense_faces=nullptr,
                                const std::vector<float>* dense_attr=nullptr,
                                bool reproject=false) {
    // optional decimation (Python decimates to ~1M verts before unwrap; keeps xatlas tractable)
    std::vector<float> dverts; std::vector<int64_t> dfaces;
    const bool deci = (decimate_target_faces>0 && (int)in_faces0.size()/3 > decimate_target_faces);
    if (deci) decimate(in_verts0, in_faces0, (size_t)decimate_target_faces, dverts, dfaces, verbose);
    const std::vector<float>& in_verts = deci ? dverts : in_verts0;
    const std::vector<int64_t>& in_faces = deci ? dfaces : in_faces0;

    const int Vin=(int)in_verts.size()/3, Fin=(int)in_faces.size()/3, C=6;
    std::vector<float> in_norm = vert_normals(in_verts, in_faces);
    std::vector<uint32_t> idx32((size_t)Fin*3);
    for (size_t i=0;i<idx32.size();i++) idx32[i]=(uint32_t)in_faces[i];

    // Direct xatlas segmentation is excellent on the normal closed meshes, but a QEM/sloppy
    // decimation can leave a large non-manifold edge fan.  On those meshes its half-edge build
    // grows pathologically (minutes for Miku) even though the material is sampled directly from
    // the PBR volume.  Detect the condition on the *actual decimated bake mesh*, then use the
    // existing conformal pre-cluster path: xatlas still parameterizes each disconnected island,
    // but never sees the bad global topology.  This is topology-driven, not model-name-driven.
    bool auto_precluster=false;
    size_t nonmanifold_edges=0;
    if (!precluster && deci && !std::getenv("ATL_FORCE_DIRECT") && !std::getenv("ATL_AUTO_PRECLUSTER_OFF")) {
        std::unordered_map<uint64_t,uint32_t> edge_use;
        edge_use.reserve((size_t)Fin*3);
        for (int f=0;f<Fin;f++) for (int k=0;k<3;k++) {
            uint32_t a=(uint32_t)in_faces[(size_t)f*3+k];
            uint32_t b=(uint32_t)in_faces[(size_t)f*3+(k+1)%3];
            if (a>b) std::swap(a,b);
            ++edge_use[((uint64_t)a<<32)|b];
        }
        for (const auto& e:edge_use) if (e.second>2) ++nonmanifold_edges;
        auto_precluster=nonmanifold_edges > std::max<size_t>(2048,(size_t)Fin/100);
        if (verbose && auto_precluster)
            printf("[atlas] auto precluster: %zu non-manifold edges / %d faces (direct xatlas safety gate)\n",
                   nonmanifold_edges,Fin);
    }
    const bool effective_precluster=precluster || auto_precluster;

    // ---- xatlas UV unwrap ----
    xatlas::Atlas* atlas = xatlas::Create();
    if (verbose) xatlas::SetProgressCallback(atlas, _xatlas_progress, nullptr);
    std::vector<std::vector<int>> xref_maps;   // per xatlas mesh: output xref -> original in_verts index
    std::vector<std::vector<float>> cluster_source_verts;
    std::vector<std::vector<float>> cluster_source_norms;
    xatlas::PackOptions po;
    po.resolution=(uint32_t)texture_size; po.padding=(uint32_t)padding;
    po.bilinear=true; po.bruteForce=false; po.blockAlign=true; po.createImage=false;
    if (std::getenv("ATL_NO_BLOCK_ALIGN")) po.blockAlign=false;
    if (std::getenv("ATL_NO_BILINEAR")) po.bilinear=false;
    if (std::getenv("ATL_PYREF_XATLAS")) {
        po.padding=0; po.blockAlign=false;
    }
    auto pack_charts = [&]() {
        xatlas::PackCharts(atlas, po);
        if (!std::getenv("ATL_FIT_RES") || texture_size <= 0) return;
        for (int it=0; it<6; it++) {
            if (atlas->atlasCount <= 1 && atlas->width <= (uint32_t)texture_size && atlas->height <= (uint32_t)texture_size)
                break;
            float sx = (float)texture_size / std::max(1u, atlas->width);
            float sy = (float)texture_size / std::max(1u, atlas->height);
            float scale = std::min(sx, sy) * 0.97f;
            if (atlas->atlasCount > 1)
                scale *= 0.45f / std::sqrt((float)atlas->atlasCount);
            scale = std::max(0.05f, std::min(0.98f, scale));
            po.texelsPerUnit = std::max(1e-6f, atlas->texelsPerUnit * scale);
            if (verbose) printf("[atlas] repack fit-res: pass %d scale=%.3f texelsPerUnit=%.3f from %ux%u atlases=%u\n",
                                it+1, scale, po.texelsPerUnit, atlas->width, atlas->height, atlas->atlasCount);
            xatlas::PackCharts(atlas, po);
        }
    };
    if (effective_precluster && !std::getenv("ATL_PLANAR")) {
        double tp=_now();
        float cdeg = getenv("ATL_CONE") ? atof(getenv("ATL_CONE")) : (std::getenv("ATL_PYREF_XATLAS") ? 90.f : cone_deg);
        std::vector<ClusterMesh> clusters;
        const char* cluster_mode = "cumesh-cpu";
        bool use_cluster_sources = false;
        if (std::getenv("ATL_BFS")) {
            cluster_mode = "BFS";
            clusters = precluster_meshes(in_verts, in_faces, std::cos(cdeg*3.14159265f/180.f));
#ifdef TEXATLAS_NATIVE_CUMESH
        } else if (std::getenv("ATL_NATIVE_CUMESH")) {
            cluster_mode = "native-cumesh";
            use_cluster_sources = true;
            std::vector<native_cumesh::ClusterMesh> nc = native_cumesh::compute_clusters(
                in_verts, in_faces, cdeg*3.14159265f/180.f,
                std::getenv("ATL_REFINE") ? atoi(std::getenv("ATL_REFINE")) : 100,
                std::getenv("ATL_GLOBAL") ? atoi(std::getenv("ATL_GLOBAL")) : 3,
                std::getenv("ATL_SMOOTH") ? (float)atof(std::getenv("ATL_SMOOTH")) : 1.f,
                std::getenv("ATL_AREA") ? (float)atof(std::getenv("ATL_AREA")) : 0.1f,
                std::getenv("ATL_PERIM") ? (float)atof(std::getenv("ATL_PERIM")) : 0.0001f);
            clusters.resize(nc.size());
            for (size_t i=0;i<nc.size();i++) {
                clusters[i].verts.swap(nc[i].verts);
                clusters[i].faces.swap(nc[i].faces);
                clusters[i].vmap.swap(nc[i].vmap);
            }
#endif
        } else {
            clusters = precluster_meshes_cumesh_cpu(in_verts, in_faces, cdeg*3.14159265f/180.f,
                    std::getenv("ATL_REFINE") ? atoi(std::getenv("ATL_REFINE")) : 100,
                    std::getenv("ATL_GLOBAL") ? atoi(std::getenv("ATL_GLOBAL")) : 3,
                    std::getenv("ATL_SMOOTH") ? (float)atof(std::getenv("ATL_SMOOTH")) : 1.f,
                    std::getenv("ATL_AREA") ? (float)atof(std::getenv("ATL_AREA")) : 0.1f,
                    std::getenv("ATL_PERIM") ? (float)atof(std::getenv("ATL_PERIM")) : 0.0001f);
        }
        size_t splitv=0; for (const auto& c:clusters) splitv += c.vmap.size();
        if (verbose){ printf("[atlas] %s precluster: %zu xatlas meshes (cone %.0f°, %.2fs), %zu split-verts\n",
                              cluster_mode, clusters.size(), cdeg, _now()-tp, splitv); fflush(stdout); }
        xref_maps.reserve(clusters.size());
        if (use_cluster_sources) {
            cluster_source_verts.reserve(clusters.size());
            cluster_source_norms.reserve(clusters.size());
        }
        for (const ClusterMesh& cm : clusters) {
            xatlas::MeshDecl md;
            md.vertexCount = (uint32_t)cm.vmap.size(); md.vertexPositionData = cm.verts.data();
            md.vertexPositionStride = 3*sizeof(float); md.indexCount = (uint32_t)cm.faces.size();
            md.indexData = cm.faces.data(); md.indexFormat = xatlas::IndexFormat::UInt32;
            xatlas::AddMeshError e = xatlas::AddMesh(atlas, md, (uint32_t)clusters.size());
            if (e != xatlas::AddMeshError::Success) fprintf(stderr,"[atlas] AddMesh(cluster) error: %s\n", xatlas::StringForEnum(e));
            xref_maps.push_back(cm.vmap);
            if (use_cluster_sources) {
                cluster_source_verts.push_back(cm.verts);
                std::vector<int64_t> lf(cm.faces.begin(), cm.faces.end());
                cluster_source_norms.push_back(vert_normals(cm.verts, lf));
            }
        }
        xatlas::AddMeshJoin(atlas);
        xatlas::ChartOptions co;
        if (std::getenv("ATL_PYREF_XATLAS")) {
            // Match cumesh.xatlas.Atlas.compute_charts defaults used by Python CuMesh.uv_unwrap().
            co.maxCost = getenv("ATL_MAXCOST") ? atof(getenv("ATL_MAXCOST")) : 2.0f;
            co.normalDeviationWeight = getenv("ATL_NDW") ? atof(getenv("ATL_NDW")) : 2.0f;
            co.normalSeamWeight = getenv("ATL_NSW") ? atof(getenv("ATL_NSW")) : 4.0f;
            co.straightnessWeight = getenv("ATL_SW") ? atof(getenv("ATL_SW")) : 6.0f;
            co.roundnessWeight = getenv("ATL_RW") ? atof(getenv("ATL_RW")) : 0.01f;
            co.textureSeamWeight = getenv("ATL_TSW") ? atof(getenv("ATL_TSW")) : 0.5f;
            co.maxIterations = getenv("ATL_XITERS") ? atoi(getenv("ATL_XITERS")) : 1;
            xatlas::ComputeCharts(atlas, co);
        } else if (use_cluster_sources && !std::getenv("ATL_FORCE_CUSTOM_CHARTS")) {
            xatlas::ComputeCharts(atlas);
        } else {
            co.maxCost = getenv("ATL_MAXCOST") ? atof(getenv("ATL_MAXCOST")) : 16.0f;
            co.normalDeviationWeight = getenv("ATL_NDW") ? atof(getenv("ATL_NDW")) : 1.0f;
            co.normalSeamWeight = 1.0f; co.straightnessWeight = 1.0f; co.roundnessWeight = 0.1f; co.maxIterations = 1;
            xatlas::ComputeCharts(atlas, co);
        }
        pack_charts();
    } else if (effective_precluster) {
        // Legacy diagnostic path: build our own charts (normal-cone region-grow), hand xatlas pre-made
        // planar UV islands. Fast, but folds on curved sheets; keep only for A/B with ATL_PLANAR=1.
        std::vector<float> uv; std::vector<int> uv2orig; std::vector<uint32_t> uvfaces, facemat; int ncl=0;
        double tp=_now();
        float cdeg = getenv("ATL_CONE") ? atof(getenv("ATL_CONE")) : cone_deg;
        precluster_charts(in_verts, in_faces, std::cos(cdeg*3.14159265f/180.f), uv, uv2orig, uvfaces, facemat, ncl);
        if (verbose){ printf("[atlas] precluster: %d charts (cone %.0f°, %.2fs), %zu uv-verts\n",
                              ncl, cdeg, _now()-tp, uv2orig.size()); fflush(stdout); }
        xatlas::UvMeshDecl um; um.vertexCount=(uint32_t)uv2orig.size(); um.vertexUvData=uv.data();
        um.vertexStride=2*sizeof(float); um.indexCount=(uint32_t)uvfaces.size(); um.indexData=uvfaces.data();
        um.indexFormat=xatlas::IndexFormat::UInt32; um.faceMaterialData=facemat.data();
        xatlas::AddMeshError e = xatlas::AddUvMesh(atlas, um);
        if (e != xatlas::AddMeshError::Success) fprintf(stderr,"[atlas] AddUvMesh error: %s\n", xatlas::StringForEnum(e));
        xatlas::ComputeCharts(atlas);   // for UV meshes: just groups existing islands (fast)
        pack_charts();
        xref_maps.push_back(uv2orig);
    } else {
        xatlas::MeshDecl md;
        md.vertexCount = (uint32_t)Vin; md.vertexPositionData = in_verts.data();
        md.vertexPositionStride = 3*sizeof(float); md.indexCount = (uint32_t)Fin*3;
        md.indexData = idx32.data(); md.indexFormat = xatlas::IndexFormat::UInt32;
        xatlas::AddMeshError e = xatlas::AddMesh(atlas, md);
        if (e != xatlas::AddMeshError::Success) { fprintf(stderr,"[atlas] AddMesh error: %s\n", xatlas::StringForEnum(e)); }
        xatlas::AddMeshJoin(atlas);
        xatlas::ChartOptions co;
        if (std::getenv("ATL_PYREF_XATLAS")) {
            // Reference mode must match CuMesh's *chart* parameters as well as its
            // pack settings.  This direct-mesh branch previously retained the loose
            // historic C++ values, so the advertised reference unwrap was not actually
            // the same recipe as the Python reference.
            co.maxCost = getenv("ATL_MAXCOST") ? atof(getenv("ATL_MAXCOST")) : 2.0f;
            co.normalDeviationWeight = getenv("ATL_NDW") ? atof(getenv("ATL_NDW")) : 2.0f;
            co.normalSeamWeight = getenv("ATL_NSW") ? atof(getenv("ATL_NSW")) : 4.0f;
            co.straightnessWeight = getenv("ATL_SW") ? atof(getenv("ATL_SW")) : 6.0f;
            co.roundnessWeight = getenv("ATL_RW") ? atof(getenv("ATL_RW")) : 0.01f;
            co.textureSeamWeight = getenv("ATL_TSW") ? atof(getenv("ATL_TSW")) : 0.5f;
            co.maxIterations = getenv("ATL_XITERS") ? atoi(getenv("ATL_XITERS")) : 1;
        } else {
            co.maxCost = getenv("ATL_MAXCOST") ? atof(getenv("ATL_MAXCOST")) : 16.0f;
            co.normalDeviationWeight = getenv("ATL_NDW") ? atof(getenv("ATL_NDW")) : 1.0f;
            co.normalSeamWeight = 1.0f; co.straightnessWeight = 1.0f; co.roundnessWeight = 0.1f; co.maxIterations = 1;
        }
        if (verbose){ printf("[atlas] unwrapping %d verts / %d faces ...\n", Vin, Fin); fflush(stdout); }
        xatlas::ComputeCharts(atlas, co);
        pack_charts();
        xref_maps.emplace_back(Vin); for (int i=0;i<Vin;i++) xref_maps.back()[i]=i;
    }
    int W=(int)atlas->width, Ht=(int)atlas->height;
    uint32_t total_v=0,total_i=0; for(uint32_t mi=0; mi<atlas->meshCount; mi++){ total_v+=atlas->meshes[mi].vertexCount; total_i+=atlas->meshes[mi].indexCount; }
    if (verbose) printf("[atlas] %ux%u  charts=%u sub-atlases=%u meshes=%u out: %u verts / %u tris\n",
                        atlas->width, atlas->height, atlas->chartCount, atlas->atlasCount,
                        atlas->meshCount, total_v, total_i/3);

    BakedTexture bt; bt.tw=W; bt.th=Ht; bt.chart_count=(int)atlas->chartCount; bt.atlas_count=(int)atlas->atlasCount;
    const int Vout=(int)total_v, Fout=(int)total_i/3;
    bt.verts.resize((size_t)Vout*3); bt.normals.resize((size_t)Vout*3); bt.uvs.resize((size_t)Vout*2);
    bt.faces.resize((size_t)Fout*3);
    std::vector<float> px(Vout), py(Vout);   // pixel-space UV for rasterization
    uint32_t voff=0, ioff=0;
    for (uint32_t mi=0; mi<atlas->meshCount; mi++) {
        const xatlas::Mesh& om = atlas->meshes[mi];
        const std::vector<int>& xref2orig = xref_maps[mi];
        const bool use_cluster_source = mi < cluster_source_verts.size() && !cluster_source_verts[mi].empty();
        const std::vector<float>* src_v = use_cluster_source ? &cluster_source_verts[mi] : &in_verts;
        const std::vector<float>* src_n = use_cluster_source ? &cluster_source_norms[mi] : &in_norm;
        for (uint32_t i=0;i<om.vertexCount;i++){
            const xatlas::Vertex& v = om.vertexArray[i];
            uint32_t r = use_cluster_source ? v.xref : (uint32_t)xref2orig[v.xref];
            uint32_t oi = voff + i;
            for (int d=0;d<3;d++){ bt.verts[(size_t)oi*3+d]=(*src_v)[(size_t)r*3+d]; bt.normals[(size_t)oi*3+d]=(*src_n)[(size_t)r*3+d]; }
            px[oi]=v.uv[0]; py[oi]=v.uv[1];
            bt.uvs[(size_t)oi*2+0]=v.uv[0]/(float)W; bt.uvs[(size_t)oi*2+1]=v.uv[1]/(float)Ht;
        }
        for (uint32_t i=0;i<om.indexCount;i++) bt.faces[(size_t)ioff+i]=voff+om.indexArray[i];
        voff += om.vertexCount; ioff += om.indexCount;
    }
    if (std::getenv("TEX_TOPO_NORMALS")) {
        // Python parity: vertex normals on the OUTPUT (chart-split) topology, exactly like bake_uv.py's
        // trimesh.Trimesh(verts,faces).vertex_normals. Area-weighted, NOT position-welded — welding
        // averages across the <1-voxel twin-tail front/back gap and chart-split seams, tilting normals
        // ~16deg off the surface (the "random triangles" under lighting). Topology normals = ~Python.
        double tn=_now();
        std::vector<int64_t> f64(bt.faces.begin(), bt.faces.end());
        bt.normals = vert_normals(bt.verts, f64);
        if (verbose) printf("[atlas] output-topology normals (%.2fs)\n", _now()-tn);
    } else if (!std::getenv("TEX_NO_WELD_NORMALS")) {
        double tn=_now();
        bt.normals = welded_vert_normals(bt.verts, bt.faces);
        if (verbose) printf("[atlas] welded output normals (%.2fs)\n", _now()-tn);
    }

    // A chart is a connected component in xatlas' OUTPUT topology.  xatlas splits a shared
    // source vertex at every UV seam, so faces that still share an output vertex are guaranteed
    // to be safe neighbours in atlas space.  Keep this identity through rasterisation for the
    // optional surface-only cleanup; a binary coverage mask alone is insufficient when direct
    // reference packing deliberately uses zero chart padding.
    std::vector<int32_t> face_parent((size_t)Fout);
    for (int t=0;t<Fout;t++) face_parent[(size_t)t]=t;
    auto face_root = [&](int32_t x) {
        while (face_parent[(size_t)x] != x) {
            face_parent[(size_t)x] = face_parent[(size_t)face_parent[(size_t)x]];
            x = face_parent[(size_t)x];
        }
        return x;
    };
    std::vector<int32_t> first_face((size_t)Vout,-1);
    for (int t=0;t<Fout;t++) for (int k=0;k<3;k++) {
        const uint32_t v=bt.faces[(size_t)t*3+k];
        int32_t& first=first_face[(size_t)v];
        if (first<0) first=t;
        else {
            int32_t a=face_root(t), b=face_root(first);
            if (a!=b) face_parent[(size_t)a]=b;
        }
    }
    std::vector<int32_t> face_chart((size_t)Fout);
    for (int t=0;t<Fout;t++) face_chart[(size_t)t]=face_root(t);

    // ---- rasterize (serial; writes per-texel 3D position + interpolated normal + mask) ----
    double t_ras = _now();
    std::vector<float> pos((size_t)W*Ht*3, 0.f);
    std::vector<float> nrm((size_t)W*Ht*3, 0.f);   // lap-18: texel normal for front-face reproject
    std::vector<uint8_t> mask((size_t)W*Ht, 0);
    std::vector<int32_t> surface_chart((size_t)W*Ht, -1);
    int raster_ss = std::getenv("TEX_RASTER_SS") ? atoi(std::getenv("TEX_RASTER_SS")) : 1;
    raster_ss = std::max(1, std::min(4, raster_ss));
    if (verbose && raster_ss > 1) printf("[atlas] raster supersample: %dx%d\n", raster_ss, raster_ss);
    for (int t=0;t<Fout;t++){
        uint32_t a=bt.faces[(size_t)t*3], b=bt.faces[(size_t)t*3+1], c=bt.faces[(size_t)t*3+2];
        float ax=px[a],ay=py[a], bx=px[b],by=py[b], cx=px[c],cy=py[c];
        float area = (bx-ax)*(cy-ay)-(by-ay)*(cx-ax);
        if (std::fabs(area)<1e-9f) continue;
        float inv=1.f/area;
        int x0=(int)std::floor(std::min({ax,bx,cx})), x1=(int)std::ceil(std::max({ax,bx,cx}));
        int y0=(int)std::floor(std::min({ay,by,cy})), y1=(int)std::ceil(std::max({ay,by,cy}));
        x0=std::max(0,x0); y0=std::max(0,y0); x1=std::min(W-1,x1); y1=std::min(Ht-1,y1);
        const float *Pa=&bt.verts[(size_t)a*3], *Pb=&bt.verts[(size_t)b*3], *Pc=&bt.verts[(size_t)c*3];
        const float *Na=&bt.normals[(size_t)a*3], *Nb=&bt.normals[(size_t)b*3], *Nc=&bt.normals[(size_t)c*3];
        for (int y=y0;y<=y1;y++) for (int x=x0;x<=x1;x++){
            float bw0=0.f,bw1=0.f,bw2=0.f; int hits=0;
            for (int syi=0; syi<raster_ss; syi++) for (int sxi=0; sxi<raster_ss; sxi++) {
                float sx=x+((float)sxi+0.5f)/(float)raster_ss;
                float sy=y+((float)syi+0.5f)/(float)raster_ss;
                float w0=((bx-sx)*(cy-sy)-(by-sy)*(cx-sx))*inv;   // bary for vertex a
                float w1=((cx-sx)*(ay-sy)-(cy-sy)*(ax-sx))*inv;   // for b
                float w2=1.f-w0-w1;                                // for c
                if (w0<0||w1<0||w2<0) continue;                    // outside (winding-normalized by inv)
                bw0+=w0; bw1+=w1; bw2+=w2; hits++;
            }
            if (!hits) continue;
            float invh=1.f/(float)hits;
            float w0=bw0*invh, w1=bw1*invh, w2=bw2*invh;
            float* P=&pos[((size_t)y*W+x)*3]; float* Nn=&nrm[((size_t)y*W+x)*3];
            for (int d=0;d<3;d++){ P[d]=w0*Pa[d]+w1*Pb[d]+w2*Pc[d]; Nn[d]=w0*Na[d]+w1*Nb[d]+w2*Nc[d]; }
            mask[(size_t)y*W+x]=1;
            surface_chart[(size_t)y*W+x]=face_chart[(size_t)t];
        }
    }
    int covered=0; for (auto m:mask) covered+=m;
    if (verbose) printf("[atlas] rasterized: %d / %d texels covered (%.1f%%) (%.2fs)\n", covered, W*Ht, 100.0*covered/(W*Ht), _now()-t_ras);
    double t_attr = _now();

    // ---- per-texel attribute: either reproject onto the dense shell (lap-18) or grid_sample the
    //      PBR volume at the rasterised position (the legacy/teal-splatter path) ----
    std::vector<float> atl((size_t)W*Ht*C, 0.f);
    const bool do_reproject = reproject && dense_verts && dense_faces && dense_attr && !dense_faces->empty();
    if (do_reproject) {
        const int ncell  = std::getenv("RP_NCELL")  ? atoi(std::getenv("RP_NCELL"))  : 256;
        const int maxring= std::getenv("RP_MAXRING")? atoi(std::getenv("RP_MAXRING")): 12;
        const float fdot = std::getenv("RP_FRONTDOT")? (float)atof(std::getenv("RP_FRONTDOT")) : 0.0f;
        const float maxdist_vox = std::getenv("RP_MAXDIST_VOX") ? (float)atof(std::getenv("RP_MAXDIST_VOX")) : -1.f;
        const float maxdist = maxdist_vox >= 0.f ? maxdist_vox / (float)grid_res : -1.f;
        const float maxdist2 = maxdist >= 0.f ? maxdist * maxdist : -1.f;
        const bool allow_back = std::getenv("RP_NO_BACKFALLBACK") == nullptr;
        // SAMPLE MODE: default = snap the texel onto the dense shell (closest-pt-on-tri, front-face
        // reject) then trilinear grid_sample the VOLUME there (stable → no per-texel speckle from
        // triangle-choice flips on fine hair; == pyref's sampling, but on the correct on-shell point).
        // RP_ATTR=1 uses the barycentric dense-mesh attr instead (speckles on fine strand detail).
        const bool use_attr = std::getenv("RP_ATTR")!=nullptr;
        // RP_DIRECT=1 (--tex-volume-direct): THE FRAY FIX. Both snap modes above read the volume at a
        // point found by composing TWO closest-point projections (texel P → coarse shell → volume).
        // That composition SLIDES ALONG THE SURFACE: measured on inline_soldier1536, 9.2% of texels end
        // up reading a voxel >4 voxels away from the voxel actually nearest to them (p99 = 13.2 vox) —
        // which at a material boundary means fetching the wrong material. That sliding IS the fraying;
        // it is not per-vertex quantisation (snap+volume scores the same as RP_ATTR: 4.73% vs 4.84%
        // texels >0.15 off the volume) and it is not the back-fallback (0.1% of bad texels).
        // Direct mode reads the volume AT THE TEXEL'S OWN rasterised position — one projection, no
        // slide. The reason that was abandoned (the BLACK-TEXTURE bug: the refined surface sits ~2.5
        // vox off the sparse PBR shell, so all 8 trilinear corners are empty → black) is really the
        // `sample_fallback_r` argument being 0 at the production call site: at fallback_r=0 direct
        // sampling is 47.6% wrong, at fallback_r=8 it is 1.34% wrong (vs RP_ATTR's 4.87%). The snap is
        // kept as the guard for the ~1.9% of texels the volume has no data near at all.
        const bool volume_direct = std::getenv("RP_DIRECT")!=nullptr;
        texgs::VolIndex vol(pbr_coords.data(), (int)pbr_coords.size()/4, 4, 1);
        double tbh=_now();
        texrp::DenseHash dh(dense_verts->data(), dense_faces->data(), (int64_t)dense_faces->size()/3, ncell);
        if (verbose) printf("[atlas] reproject: dense %zu v / %zu f, hash %d^3 cells (%.2fs build), front_dot=%.2f, maxdist=%.2f vox, back_fallback=%s, mode=%s, fallback_r=%d\n",
                            dense_verts->size()/3, dense_faces->size()/3, ncell, _now()-tbh, fdot, maxdist_vox,
                            allow_back?"on":"off",
                            volume_direct ? (use_attr?"volume-direct (guard: mesh-attr)":"volume-direct (guard: snap+volume)")
                                          : (use_attr?"mesh-attr":"snap+volume"),
                            sample_fallback_r);
        size_t miss=0, guard=0;
        #pragma omp parallel for schedule(dynamic, 2048) reduction(+:miss) reduction(+:guard)
        for (int p=0;p<W*Ht;p++){
            if (!mask[p]) continue;
            const float* P=&pos[(size_t)p*3]; const float* Nn=&nrm[(size_t)p*3];
            // VOLUME-DIRECT: read the volume at the texel's OWN position first. No shell round-trip →
            // no slide → the volume's own boundary sharpness survives. Only if the volume has nothing
            // within sample_fallback_r of this texel do we fall through to the snap (the black-texture guard).
            if (volume_direct) {
                float q0=(P[0]+0.5f)*grid_res, q1=(P[1]+0.5f)*grid_res, q2=(P[2]+0.5f)*grid_res;
                texgs::sample_one(vol, pbr_feats.data(), C, q0,q1,q2, &atl[(size_t)p*C], sample_fallback_r);
                bool any=false; for (int c=0;c<C;c++) if (atl[(size_t)p*C+c]!=0.f){ any=true; break; }
                if (any) continue;
                guard++;
            }
            float qn[3]={Nn[0],Nn[1],Nn[2]}; float L=std::sqrt(qn[0]*qn[0]+qn[1]*qn[1]+qn[2]*qn[2]);
            if (L>1e-20f){ qn[0]/=L;qn[1]/=L;qn[2]/=L; }
            float snap[3];
            if (!dh.sample(P, qn, use_attr?dense_attr->data():nullptr, C, use_attr?&atl[(size_t)p*C]:nullptr, snap, fdot, maxring, maxdist2, allow_back)) { miss++; continue; }
            if (!use_attr){ float q0=(snap[0]+0.5f)*grid_res, q1=(snap[1]+0.5f)*grid_res, q2=(snap[2]+0.5f)*grid_res;
                texgs::sample_one(vol, pbr_feats.data(), C, q0,q1,q2, &atl[(size_t)p*C], sample_fallback_r); }
        }
        if (verbose && volume_direct) printf("[atlas] volume-direct: %zu texels (%.3f%% of covered) had no voxel within fallback_r=%d -> snap guard\n",
                                             guard, 100.0*guard/(double)std::max(1,covered), sample_fallback_r);
        if (verbose && miss) printf("[atlas] reproject misses (no dense tri in range): %zu (%.3f%% of covered)\n",
                                    miss, 100.0*miss/(double)std::max(1,covered));
    } else {
        texgs::VolIndex vol(pbr_coords.data(), (int)pbr_coords.size()/4, 4, 1);
        #pragma omp parallel for schedule(dynamic, 4096)
        for (int p=0;p<W*Ht;p++){
            if (!mask[p]) continue;
            const float* P=&pos[(size_t)p*3];
            float q0=(P[0]+0.5f)*grid_res, q1=(P[1]+0.5f)*grid_res, q2=(P[2]+0.5f)*grid_res;
            texgs::sample_one(vol, pbr_feats.data(), C, q0,q1,q2, &atl[(size_t)p*C], sample_fallback_r);
        }
    }

    if (verbose) printf("[atlas] per-texel attr (%s): (%.2fs)\n", do_reproject?"reproject":"grid_sample", _now()-t_attr);
    double t_inp = _now();

    // ---- inpaint: gutter + INTERIOR HOLES. A texel can be covered (inside a chart triangle) yet
    // grid_sample misses (the remeshed surface is too far from the sparse PBR shell in deep
    // concavities — skirt folds, underarm) -> all-zero -> a black hole in the render. Treat those
    // (covered AND all-channels-zero) as NOT-valid so the inpaint fills them from valid chart
    // neighbours (correct local colour, unlike grabbing a far/wrong voxel via a bigger fallback).
    std::vector<uint8_t> mask2(mask.size());
    size_t holes=0;
    for (size_t p=0;p<(size_t)W*Ht;p++){
        bool any=false; const float* a=&atl[p*C]; for(int c=0;c<C;c++) if(a[c]!=0.f){any=true;break;}
        mask2[p] = (mask[p] && any) ? 1 : 0;
        if (mask[p] && !any) holes++;
    }
    if (verbose && holes) printf("[atlas] interior holes (covered but unsampled): %zu (%.2f%% of covered) -> inpainting\n",
                                 holes, 100.0*holes/(double)covered);
    // Gutter dilation. NON-precluster (few large charts, far apart): a big fill (64) also seals
    // interior grid_sample-miss holes. PRECLUSTER (thousands of tiny charts packed ~padding apart):
    // a big fill BLEEDS each chart's colour across the gutter into its neighbours → teal "peeking
    // through the cuts" (the seam-glitch). Cap the fill at padding-1 so each chart only fills its OWN
    // gutter and never reaches a neighbour's rendered texels (neighbours are `padding` apart). Interior
    // holes are instead prevented by the nearest-voxel sample fallback, not by dilation.
    int inp_iters = std::getenv("TEX_INPAINT_ITERS") ? atoi(std::getenv("TEX_INPAINT_ITERS"))
                                                     : (precluster ? std::max(1, padding-1) : 64);
    if (std::getenv("TEX_TELEA_INPAINT")) {
        int tr = std::getenv("TEX_TELEA_RADIUS") ? atoi(std::getenv("TEX_TELEA_RADIUS")) : 3;
        int rings = precluster ? inp_iters : std::max(padding+2, inp_iters);
        if (verbose) printf("[atlas] telea-style inpaint rings=%d radius=%d\n", rings, tr);
        inpaint_telea(atl, mask2, W, Ht, C, rings, tr);
    } else {
        inpaint(atl, mask2, W, Ht, C, precluster ? inp_iters : std::max(padding+2, inp_iters));
    }
    if (std::getenv("TEX_FILL_BACKGROUND")) {
        if (verbose) printf("[atlas] filling remaining atlas background by nearest valid dilation\n");
        dilate_background(atl, mask2, W, Ht, C);
    }
    if (const char* m = std::getenv("TEX_BASE_MEDIAN")) {
        int radius=std::max(0,std::min(4,atoi(m)));
        if (radius > 0) {
            if (verbose) printf("[atlas] surface-only RGB median radius=%d\n", radius);
            median_surface_rgb(atl, mask, surface_chart, W, Ht, radius);
        }
    }

    // ---- pack to uint8 textures (Python layout) ----
    // baseColor colourspace lever (owner judges — do not declare a winner). The tex-DiT baseColor sits
    // in a dark/linear space; an encode brightens it to the spec-correct stored value. The decoder emits
    // LINEAR baseColor but glTF tags the texture sRGB, so a conformant viewer double-converts -> ~54% too
    // dark. The sRGB OETF (linear->sRGB) here makes the stored texel round-trip correctly (verified
    // numerically 2026-06-20: spec-viewer decode recovers the true albedo to <0.002). DEFAULT ON now;
    // set ATL_BASECOLOR_SRGB=0 to disable. ATL_BASECOLOR_GAMMA=g instead applies pow(c,g) (g<1 brightens).
    // Applied to the RGB baseColor channels only (not alpha/PBR).
    const char* bc_srgb_env = std::getenv("ATL_BASECOLOR_SRGB");
    const bool bc_srgb = bc_srgb_env ? (std::atoi(bc_srgb_env) != 0) : true;
    const float bc_gamma = std::getenv("ATL_BASECOLOR_GAMMA") ? (float)atof(std::getenv("ATL_BASECOLOR_GAMMA")) : 0.f;
    auto enc_bc = [&](float c)->float{
        if (c<0.f) c=0.f;
        if (c>1.f) c=1.f;
        if (bc_srgb) return c<=0.0031308f ? 12.92f*c : 1.055f*std::pow(c,1.f/2.4f)-0.055f;
        if (bc_gamma>0.f) return std::pow(c, bc_gamma);
        return c;
    };
    if (verbose && (bc_srgb || bc_gamma>0.f))
        printf("[atlas] baseColor encode: %s\n", bc_srgb ? "sRGB OETF" : ("gamma " + std::to_string(bc_gamma)).c_str());
    bt.base_color.resize((size_t)W*Ht*4);
    bt.metal_rough.resize((size_t)W*Ht*3);
    for (size_t p=0;p<(size_t)W*Ht;p++){
        const float* a=&atl[p*C];
        bt.base_color[p*4+0]=u8(enc_bc(a[0])); bt.base_color[p*4+1]=u8(enc_bc(a[1])); bt.base_color[p*4+2]=u8(enc_bc(a[2]));
        bt.base_color[p*4+3]=u8(a[5]);                       // alpha
        bt.metal_rough[p*3+0]=0;                              // R unused
        bt.metal_rough[p*3+1]=u8(a[4]);                       // G = roughness
        bt.metal_rough[p*3+2]=u8(a[3]);                       // B = metallic
    }
    bool blur_done = false;
    if (std::getenv("TEX_BASE_BLUR_PRE_RESIZE")) {
        if (const char* b = std::getenv("TEX_BASE_BLUR")) {
            float sigma = (float)atof(b);
            if (verbose && sigma > 0.f) printf("[atlas] baseColor pre-resize gaussian blur sigma=%.2f\n", sigma);
            blur_u8_gaussian_rgb(bt.base_color, W, Ht, 4, sigma);
            blur_done = sigma > 0.f;
        }
    }
    int final_size = std::getenv("TEX_FINAL_SIZE") ? atoi(std::getenv("TEX_FINAL_SIZE")) : texture_size;
    if (!std::getenv("TEX_KEEP_ATLAS_SIZE") && final_size > 0 && (W > final_size || Ht > final_size)) {
        int nw = final_size;
        int nh = std::max(1, (int)std::lround((double)Ht * final_size / (double)W));
        if (verbose) printf("[atlas] resize final textures: %dx%d -> %dx%d\n", W, Ht, nw, nh);
        bt.base_color = resize_u8_area(bt.base_color, W, Ht, nw, nh, 4);
        bt.metal_rough = resize_u8_area(bt.metal_rough, W, Ht, nw, nh, 3);
        bt.tw = nw; bt.th = nh;
    }
    if (!blur_done && (std::getenv("TEX_BASE_BLUR_PRE_RESIZE") == nullptr)) if (const char* b = std::getenv("TEX_BASE_BLUR")) {
        float sigma = (float)atof(b);
        if (verbose && sigma > 0.f) printf("[atlas] baseColor gaussian blur sigma=%.2f\n", sigma);
        blur_u8_gaussian_rgb(bt.base_color, bt.tw, bt.th, 4, sigma);
    }
    xatlas::Destroy(atlas);
    if (verbose) printf("[atlas] inpaint+encode+resize (%.2fs)\n", _now()-t_inp);
    return bt;
}

}  // namespace texatlas
