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
#include "tex_grid_sample.hpp"
#include "tex_reproject.hpp"     // lap-18: closest-point-on-dense-mesh reproject (kills splatter/cracks)
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
                if(!dx&&!dy) continue; int nx=x+dx,ny=y+dy; if(nx<0||ny<0||nx>=W||ny>=H) continue;
                if(mask[(size_t)ny*W+nx]){ const float*s=&img[((size_t)ny*W+nx)*C]; for(int c=0;c<C;c++) acc[c]+=s[c]; cnt++; }
            }
            if(cnt){ float*d=&img[((size_t)y*W+x)*C]; for(int c=0;c<C;c++) d[c]=acc[c]/cnt; coords.push_back(y*W+x); }
        }
        if (coords.empty()) break;
        for (int idx : coords) mask[idx]=1;   // promote after the full pass (ring-by-ring)
    }
}

static inline uint8_t u8(float v){ v=v*255.f+0.5f; return (uint8_t)(v<0?0:(v>255?255:v)); }

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

    // ---- xatlas UV unwrap ----
    xatlas::Atlas* atlas = xatlas::Create();
    if (verbose) xatlas::SetProgressCallback(atlas, _xatlas_progress, nullptr);
    std::vector<int> xref2orig;   // output-vertex xref -> original in_verts index (for 3D position)
    xatlas::PackOptions po; po.resolution=(uint32_t)texture_size; po.padding=(uint32_t)padding;
    po.bilinear=true; po.bruteForce=false; po.blockAlign=true; po.createImage=false;
    if (precluster) {
        // P1: build our own charts (normal-cone region-grow), hand xatlas pre-made UV islands ->
        // AddUvMesh skips the slow segmentation (xatlas hangs on the QEM mesh's non-manifold edges).
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
        xatlas::PackCharts(atlas, po);
        xref2orig = uv2orig;
    } else {
        xatlas::MeshDecl md;
        md.vertexCount = (uint32_t)Vin; md.vertexPositionData = in_verts.data();
        md.vertexPositionStride = 3*sizeof(float); md.indexCount = (uint32_t)Fin*3;
        md.indexData = idx32.data(); md.indexFormat = xatlas::IndexFormat::UInt32;
        xatlas::AddMeshError e = xatlas::AddMesh(atlas, md);
        if (e != xatlas::AddMeshError::Success) { fprintf(stderr,"[atlas] AddMesh error: %s\n", xatlas::StringForEnum(e)); }
        xatlas::AddMeshJoin(atlas);
        xatlas::ChartOptions co;
        co.maxCost = getenv("ATL_MAXCOST") ? atof(getenv("ATL_MAXCOST")) : 16.0f;
        co.normalDeviationWeight = getenv("ATL_NDW") ? atof(getenv("ATL_NDW")) : 1.0f;
        co.normalSeamWeight = 1.0f; co.straightnessWeight = 1.0f; co.roundnessWeight = 0.1f; co.maxIterations = 1;
        if (verbose){ printf("[atlas] unwrapping %d verts / %d faces ...\n", Vin, Fin); fflush(stdout); }
        xatlas::ComputeCharts(atlas, co);
        xatlas::PackCharts(atlas, po);
        xref2orig.resize(Vin); for (int i=0;i<Vin;i++) xref2orig[i]=i;
    }
    const xatlas::Mesh& om = atlas->meshes[0];
    int W=(int)atlas->width, Ht=(int)atlas->height;
    if (verbose) printf("[atlas] %ux%u  charts=%u sub-atlases=%u  out: %u verts / %u tris\n",
                        atlas->width, atlas->height, atlas->chartCount, atlas->atlasCount,
                        om.vertexCount, om.indexCount/3);

    BakedTexture bt; bt.tw=W; bt.th=Ht; bt.chart_count=(int)atlas->chartCount; bt.atlas_count=(int)atlas->atlasCount;
    const int Vout=(int)om.vertexCount, Fout=(int)om.indexCount/3;
    bt.verts.resize((size_t)Vout*3); bt.normals.resize((size_t)Vout*3); bt.uvs.resize((size_t)Vout*2);
    bt.faces.resize((size_t)Fout*3);
    std::vector<float> px(Vout), py(Vout);   // pixel-space UV for rasterization
    for (int i=0;i<Vout;i++){
        const xatlas::Vertex& v = om.vertexArray[i];
        uint32_t r = (uint32_t)xref2orig[v.xref];
        for (int d=0;d<3;d++){ bt.verts[(size_t)i*3+d]=in_verts[(size_t)r*3+d]; bt.normals[(size_t)i*3+d]=in_norm[(size_t)r*3+d]; }
        px[i]=v.uv[0]; py[i]=v.uv[1];
        bt.uvs[(size_t)i*2+0]=v.uv[0]/(float)W; bt.uvs[(size_t)i*2+1]=v.uv[1]/(float)Ht;
    }
    for (size_t i=0;i<bt.faces.size();i++) bt.faces[i]=om.indexArray[i];

    // ---- rasterize (serial; writes per-texel 3D position + interpolated normal + mask) ----
    std::vector<float> pos((size_t)W*Ht*3, 0.f);
    std::vector<float> nrm((size_t)W*Ht*3, 0.f);   // lap-18: texel normal for front-face reproject
    std::vector<uint8_t> mask((size_t)W*Ht, 0);
    for (int t=0;t<Fout;t++){
        uint32_t a=om.indexArray[t*3], b=om.indexArray[t*3+1], c=om.indexArray[t*3+2];
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
            float sx=x+0.5f, sy=y+0.5f;
            float w0=((bx-sx)*(cy-sy)-(by-sy)*(cx-sx))*inv;   // bary for vertex a
            float w1=((cx-sx)*(ay-sy)-(cy-sy)*(ax-sx))*inv;   // for b
            float w2=1.f-w0-w1;                                // for c
            if (w0<0||w1<0||w2<0) continue;                    // outside (winding-normalized by inv)
            float* P=&pos[((size_t)y*W+x)*3]; float* Nn=&nrm[((size_t)y*W+x)*3];
            for (int d=0;d<3;d++){ P[d]=w0*Pa[d]+w1*Pb[d]+w2*Pc[d]; Nn[d]=w0*Na[d]+w1*Nb[d]+w2*Nc[d]; }
            mask[(size_t)y*W+x]=1;
        }
    }
    int covered=0; for (auto m:mask) covered+=m;
    if (verbose) printf("[atlas] rasterized: %d / %d texels covered (%.1f%%)\n", covered, W*Ht, 100.0*covered/(W*Ht));

    // ---- per-texel attribute: either reproject onto the dense shell (lap-18) or grid_sample the
    //      PBR volume at the rasterised position (the legacy/teal-splatter path) ----
    std::vector<float> atl((size_t)W*Ht*C, 0.f);
    const bool do_reproject = reproject && dense_verts && dense_faces && dense_attr && !dense_faces->empty();
    if (do_reproject) {
        const int ncell  = std::getenv("RP_NCELL")  ? atoi(std::getenv("RP_NCELL"))  : 256;
        const int maxring= std::getenv("RP_MAXRING")? atoi(std::getenv("RP_MAXRING")): 12;
        const float fdot = std::getenv("RP_FRONTDOT")? (float)atof(std::getenv("RP_FRONTDOT")) : 0.0f;
        // SAMPLE MODE: default = snap the texel onto the dense shell (closest-pt-on-tri, front-face
        // reject) then trilinear grid_sample the VOLUME there (stable → no per-texel speckle from
        // triangle-choice flips on fine hair; == pyref's sampling, but on the correct on-shell point).
        // RP_ATTR=1 uses the barycentric dense-mesh attr instead (speckles on fine strand detail).
        const bool use_attr = std::getenv("RP_ATTR")!=nullptr;
        texgs::VolIndex vol(pbr_coords.data(), (int)pbr_coords.size()/4, 4, 1);
        double tbh=_now();
        texrp::DenseHash dh(dense_verts->data(), dense_faces->data(), (int64_t)dense_faces->size()/3, ncell);
        if (verbose) printf("[atlas] reproject: dense %zu v / %zu f, hash %d^3 cells (%.2fs build), front_dot=%.2f, mode=%s\n",
                            dense_verts->size()/3, dense_faces->size()/3, ncell, _now()-tbh, fdot, use_attr?"mesh-attr":"snap+volume");
        size_t miss=0;
        #pragma omp parallel for schedule(dynamic, 2048) reduction(+:miss)
        for (int p=0;p<W*Ht;p++){
            if (!mask[p]) continue;
            const float* P=&pos[(size_t)p*3]; const float* Nn=&nrm[(size_t)p*3];
            float qn[3]={Nn[0],Nn[1],Nn[2]}; float L=std::sqrt(qn[0]*qn[0]+qn[1]*qn[1]+qn[2]*qn[2]);
            if (L>1e-20f){ qn[0]/=L;qn[1]/=L;qn[2]/=L; }
            float snap[3];
            if (!dh.sample(P, qn, use_attr?dense_attr->data():nullptr, C, use_attr?&atl[(size_t)p*C]:nullptr, snap, fdot, maxring)) { miss++; continue; }
            if (!use_attr){ float q0=(snap[0]+0.5f)*grid_res, q1=(snap[1]+0.5f)*grid_res, q2=(snap[2]+0.5f)*grid_res;
                texgs::sample_one(vol, pbr_feats.data(), C, q0,q1,q2, &atl[(size_t)p*C], sample_fallback_r); }
        }
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
    inpaint(atl, mask2, W, Ht, C, precluster ? inp_iters : std::max(padding+2, inp_iters));

    // ---- pack to uint8 textures (Python layout) ----
    bt.base_color.resize((size_t)W*Ht*4);
    bt.metal_rough.resize((size_t)W*Ht*3);
    for (size_t p=0;p<(size_t)W*Ht;p++){
        const float* a=&atl[p*C];
        bt.base_color[p*4+0]=u8(a[0]); bt.base_color[p*4+1]=u8(a[1]); bt.base_color[p*4+2]=u8(a[2]);
        bt.base_color[p*4+3]=u8(a[5]);                       // alpha
        bt.metal_rough[p*3+0]=0;                              // R unused
        bt.metal_rough[p*3+1]=u8(a[4]);                       // G = roughness
        bt.metal_rough[p*3+2]=u8(a[3]);                       // B = metallic
    }
    xatlas::Destroy(atlas);
    return bt;
}

}  // namespace texatlas
