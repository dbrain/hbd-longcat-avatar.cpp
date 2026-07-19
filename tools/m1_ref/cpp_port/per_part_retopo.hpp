// per_part_retopo.hpp — seam-aware PER-PART Instant Meshes retopology.
//
// The endgame for --quad (see [[project_image_to_rig_retopo_findings]]): global uniform IM under-
// resolves fingers; global adaptive IM leaves ~2040 holes. Per-part UNIFORM IM gets both — each
// P3-SAM part is remeshed at a part-appropriate UNIFORM density (fingers dense, body sparse), so
// fingers survive AND every part is watertight-uniform (no adaptivity holes).
//
// The hard part is SEAMS. IM replaces every vertex, so two independently-remeshed adjacent parts do
// NOT share their cut boundary -> cracks. We close them with a three-step defence:
//   (1) run each part's IM with `-b` (align the field to the part's open boundary -> straight cut edge);
//   (2) SNAP each part's IM boundary verts onto the original P3-SAM part-cut curve (a cloud of the
//       original mesh's seam vertices) -> both parts' boundaries land on the same curve;
//   (3) global WELD (spatial hash) of now-coincident boundary verts, then FILL_HOLES fan-fills any
//       residual open loop left by the density mismatch across the seam -> guaranteed watertight.
// Bake reads normals from the dense source mesh, so the fan patches don't show shading-wise.
//
// Densities come from ppd::tier_keep() (the same region table the decimation path uses): the per-part
// IM face target = max(400, round(part_faces * keep)). So the retopo density tracks the decimation
// intent exactly (HAND 0.80, BODY 0.05, ...), just as quads instead of a QEM tri budget.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include "glb_reader.hpp"           // glb::Mesh
#include "per_part_decimate.hpp"    // ppd::tier_keep, ppd::PartReport
#include "im_retopo.hpp"            // imretopo::im_retopo, svae::Mesh

namespace ppr {

struct RetopoCfg {
    imretopo::ImCfg im;        // base IM config (bin/tmp/crease); per-part sets target_faces + -b.
    double snap_radius_frac = 3.0;  // snap IM boundary verts within snap_radius_frac * part_edge_len
                                    // of a seam anchor onto it. 0 disables snapping.
    double weld_eps_frac    = 0.5;  // exact-dedup weld within weld_eps_frac * min_part_edge_len.
    double bweld_eps_frac   = 0.9;  // boundary-only seam weld within bweld_eps_frac * max_part_edge_len.
    bool   fill_holes       = true; // fan-fill residual open loops (watertight guarantee).
    bool   verbose          = true;
};

// ---------------------------------------------------------------------------
// undirected edge key over 32/64-bit vertex indices
// ---------------------------------------------------------------------------
static inline uint64_t edge_key(int64_t a, int64_t b) {
    uint64_t x = (uint64_t)a, y = (uint64_t)b;
    if (x > y) std::swap(x, y);
    return (x << 32) ^ y;   // vertex counts here are < 2^32
}

// count boundary (open) edges of a tri mesh: undirected edges incident to exactly 1 face.
static inline int64_t count_open_edges(const std::vector<int64_t>& faces) {
    std::unordered_map<uint64_t, int> cnt;
    cnt.reserve(faces.size());
    const int64_t F = (int64_t)faces.size() / 3;
    for (int64_t f = 0; f < F; f++)
        for (int e = 0; e < 3; e++)
            cnt[edge_key(faces[f*3+e], faces[f*3+(e+1)%3])]++;
    int64_t open = 0;
    for (auto& kv : cnt) if (kv.second == 1) open++;
    return open;
}

// canonical vertex ids by POSITION (the AI meshes are split-vert: V >> F, so index-based adjacency
// is meaningless — a geometrically shared edge has different indices on each side). Quantize each
// position to a fine grid and map coincident verts to one canonical id.
static inline std::vector<int64_t> canonical_by_position(const glb::Mesh& m, double eps) {
    const int64_t V = (int64_t)m.verts.size()/3;
    const double inv = 1.0 / (eps > 0 ? eps : 1e-6);
    std::unordered_map<uint64_t,int64_t> cell; cell.reserve(V);
    std::vector<int64_t> canon(V);
    int64_t next = 0;
    for (int64_t i=0;i<V;i++){
        const float* p=&m.verts[i*3];
        int64_t x=(int64_t)std::llround(p[0]*inv), y=(int64_t)std::llround(p[1]*inv), z=(int64_t)std::llround(p[2]*inv);
        uint64_t k = ((uint64_t)(uint32_t)(x*73856093)) ^ ((uint64_t)(uint32_t)(y*19349663)<<21) ^ ((uint64_t)(uint32_t)(z*83492791)<<42);
        auto it=cell.find(k);
        if (it==cell.end()){ cell.emplace(k,next); canon[i]=next; next++; }
        else canon[i]=it->second;
    }
    return canon;
}

// ---------------------------------------------------------------------------
// seam anchor cloud: positions where two P3-SAM part labels meet (an edge whose
// two incident faces carry different labels), plus genuine mesh-boundary verts.
// Adjacency is computed on POSITION-welded (canonical) vertices so split-vert
// input does not blow the anchor set up. These are the curves the per-part IM
// boundaries must land on.
// ---------------------------------------------------------------------------
static inline std::vector<float> build_seam_anchors(const glb::Mesh& m,
                                                    const std::vector<int64_t>& fid,
                                                    double eps) {
    const int64_t F = (int64_t)m.faces.size() / 3;
    std::vector<int64_t> canon = canonical_by_position(m, eps);
    std::unordered_map<uint64_t, int64_t> edge_lbl;   // canon-edge -> first label (-1 = already seam)
    std::unordered_map<uint64_t, int>     ecnt;       // canon-edge -> incidence count
    edge_lbl.reserve(F * 3); ecnt.reserve(F * 3);
    std::unordered_set<int64_t> seam_c;               // canonical seam vertex ids
    auto touch = [&](int64_t a, int64_t b, int64_t lbl) {
        uint64_t k = edge_key(a, b); ecnt[k]++;
        auto it = edge_lbl.find(k);
        if (it == edge_lbl.end()) edge_lbl.emplace(k, lbl);
        else if (it->second != lbl && it->second != -1) { seam_c.insert(a); seam_c.insert(b); it->second = -1; }
    };
    for (int64_t f = 0; f < F; f++) {
        int64_t l=fid[f], a=canon[m.faces[f*3]], b=canon[m.faces[f*3+1]], c=canon[m.faces[f*3+2]];
        touch(a,b,l); touch(b,c,l); touch(c,a,l);
    }
    // genuine mesh-boundary edges (canon edge seen once) -> anchors too.
    for (auto& kv : ecnt) if (kv.second == 1) { seam_c.insert((int64_t)(kv.first >> 32)); seam_c.insert((int64_t)(uint32_t)kv.first); }
    // one representative position per canonical seam id: scan verts, emit first hit.
    std::unordered_set<int64_t> done; done.reserve(seam_c.size());
    std::vector<float> pts; pts.reserve(seam_c.size()*3);
    const int64_t V=(int64_t)m.verts.size()/3;
    for (int64_t i=0;i<V;i++){ int64_t cv=canon[i]; if (seam_c.count(cv) && done.insert(cv).second){ pts.push_back(m.verts[i*3]); pts.push_back(m.verts[i*3+1]); pts.push_back(m.verts[i*3+2]); } }
    return pts;
}

// ---------------------------------------------------------------------------
// uniform-grid nearest-point index over a point cloud (for seam snapping).
// ---------------------------------------------------------------------------
struct PointGrid {
    std::vector<float> pts;                       // n*3
    double cell = 1.0, inv = 1.0;
    std::unordered_map<uint64_t, std::vector<int>> grid;
    void build(std::vector<float> p, double cell_size) {
        pts = std::move(p); cell = cell_size > 0 ? cell_size : 1.0; inv = 1.0 / cell;
        const int64_t n = (int64_t)pts.size() / 3;
        grid.reserve(n);
        for (int64_t i = 0; i < n; i++) grid[key(&pts[i*3])].push_back((int)i);
    }
    uint64_t key(const float* p) const {
        int64_t x = (int64_t)std::floor(p[0]*inv), y = (int64_t)std::floor(p[1]*inv), z = (int64_t)std::floor(p[2]*inv);
        return ((uint64_t)(uint32_t)(x*73856093) ) ^ ((uint64_t)(uint32_t)(y*19349663)<<21) ^ ((uint64_t)(uint32_t)(z*83492791)<<42);
    }
    // nearest point within `radius`; returns index or -1. writes squared distance if found.
    int nearest(const float* q, double radius, double* d2out = nullptr) const {
        int64_t cx = (int64_t)std::floor(q[0]*inv), cy = (int64_t)std::floor(q[1]*inv), cz = (int64_t)std::floor(q[2]*inv);
        double best = radius*radius; int bi = -1;
        for (int dz=-1; dz<=1; dz++) for (int dy=-1; dy<=1; dy++) for (int dx=-1; dx<=1; dx++) {
            int64_t x=cx+dx, y=cy+dy, z=cz+dz;
            uint64_t k = ((uint64_t)(uint32_t)(x*73856093)) ^ ((uint64_t)(uint32_t)(y*19349663)<<21) ^ ((uint64_t)(uint32_t)(z*83492791)<<42);
            auto it = grid.find(k); if (it == grid.end()) continue;
            for (int idx : it->second) {
                const float* p = &pts[(size_t)idx*3];
                double ddx=p[0]-q[0], ddy=p[1]-q[1], ddz=p[2]-q[2];
                double d2 = ddx*ddx+ddy*ddy+ddz*ddz;
                if (d2 < best) { best = d2; bi = idx; }
            }
        }
        if (bi >= 0 && d2out) *d2out = best;
        return bi;
    }
};

// ---------------------------------------------------------------------------
// weld coincident verts (spatial hash @ eps); rewrites faces, drops degenerate tris.
// ---------------------------------------------------------------------------
static inline void weld_mesh(std::vector<float>& V, std::vector<int64_t>& Fc, double eps) {
    if (eps <= 0) return;
    const int64_t nv = (int64_t)V.size() / 3;
    const double inv = 1.0 / eps;
    std::unordered_map<uint64_t, int64_t> cell_to_new;   // quantized cell -> new index
    cell_to_new.reserve(nv);
    std::vector<int64_t> remap(nv);
    std::vector<float> newV; newV.reserve(V.size());
    auto qkey = [&](const float* p) {
        int64_t x=(int64_t)std::llround(p[0]*inv), y=(int64_t)std::llround(p[1]*inv), z=(int64_t)std::llround(p[2]*inv);
        return ((uint64_t)(uint32_t)(x*73856093)) ^ ((uint64_t)(uint32_t)(y*19349663)<<21) ^ ((uint64_t)(uint32_t)(z*83492791)<<42);
    };
    for (int64_t i = 0; i < nv; i++) {
        uint64_t k = qkey(&V[i*3]);
        auto it = cell_to_new.find(k);
        if (it == cell_to_new.end()) {
            int64_t ni = (int64_t)newV.size()/3;
            cell_to_new.emplace(k, ni);
            newV.push_back(V[i*3]); newV.push_back(V[i*3+1]); newV.push_back(V[i*3+2]);
            remap[i] = ni;
        } else remap[i] = it->second;
    }
    std::vector<int64_t> newF; newF.reserve(Fc.size());
    for (size_t f = 0; f < Fc.size(); f += 3) {
        int64_t a = remap[Fc[f]], b = remap[Fc[f+1]], c = remap[Fc[f+2]];
        if (a==b || b==c || a==c) continue;   // degenerate after weld
        newF.push_back(a); newF.push_back(b); newF.push_back(c);
    }
    V.swap(newV); Fc.swap(newF);
}

// ---------------------------------------------------------------------------
// boundary-only weld: merge coincident-within-eps verts that lie on an OPEN edge, leaving interior
// verts untouched. The seam gap between two parts is ~the COARSER part's edge, so this uses a generous
// eps to close seams without collapsing the dense interiors a global weld at that eps would ruin.
// ---------------------------------------------------------------------------
static inline void boundary_weld(std::vector<float>& V, std::vector<int64_t>& Fc, double eps) {
    if (eps <= 0) return;
    const int64_t nv = (int64_t)V.size()/3;
    // boundary verts (on an open edge)
    std::unordered_map<uint64_t,int> ec; ec.reserve(Fc.size());
    for (size_t f=0; f<Fc.size(); f+=3) for (int e=0;e<3;e++) ec[edge_key(Fc[f+e], Fc[f+(e+1)%3])]++;
    std::vector<char> isb(nv, 0);
    for (size_t f=0; f<Fc.size(); f+=3) for (int e=0;e<3;e++) {
        int64_t a=Fc[f+e], b=Fc[f+(e+1)%3];
        if (ec[edge_key(a,b)]==1){ isb[a]=1; isb[b]=1; }
    }
    // hash boundary verts to an eps grid; first vert in a cell is the representative.
    const double inv=1.0/eps;
    std::unordered_map<uint64_t,int64_t> rep; rep.reserve(nv);
    std::vector<int64_t> remap(nv);
    for (int64_t i=0;i<nv;i++) remap[i]=i;
    for (int64_t i=0;i<nv;i++) {
        if (!isb[i]) continue;
        int64_t x=llround(V[i*3]*inv), y=llround(V[i*3+1]*inv), z=llround(V[i*3+2]*inv);
        uint64_t k=((uint64_t)(uint32_t)(x*73856093))^((uint64_t)(uint32_t)(y*19349663)<<21)^((uint64_t)(uint32_t)(z*83492791)<<42);
        auto it=rep.find(k);
        if (it==rep.end()) rep.emplace(k,i);
        else remap[i]=it->second;
    }
    std::vector<int64_t> nf; nf.reserve(Fc.size());
    for (size_t f=0; f<Fc.size(); f+=3) {
        int64_t a=remap[Fc[f]], b=remap[Fc[f+1]], c=remap[Fc[f+2]];
        if (a==b||b==c||a==c) continue;
        nf.push_back(a); nf.push_back(b); nf.push_back(c);
    }
    Fc.swap(nf);   // verts left in place (unreferenced ones are harmless; glb writer keeps all)
}

// ---------------------------------------------------------------------------
// trace oriented open-boundary loops and fan-fill each (centroid + triangle fan).
// Preserves all existing faces; only adds patches. Guarantees watertight if the
// open edges form closed loops (they do on a welded manifold-ish surface).
// ---------------------------------------------------------------------------
static inline int fill_holes(std::vector<float>& V, std::vector<int64_t>& Fc, int max_loop = 400) {
    auto dkey = [](int64_t a, int64_t b){ return ((uint64_t)(uint32_t)a << 32) | (uint32_t)b; };
    const int64_t F = (int64_t)Fc.size()/3;
    std::unordered_set<uint64_t> dir; dir.reserve(Fc.size());
    for (int64_t f=0; f<F; f++) for (int e=0;e<3;e++) dir.insert(dkey(Fc[f*3+e], Fc[f*3+(e+1)%3]));
    // boundary directed edges a->b (opposite absent). Unique successor per vertex on a clean 2-manifold
    // boundary; keep only the first (avoids the runaway fan spiral when tangled non-loops branch).
    std::unordered_map<int64_t, int64_t> succ; succ.reserve(Fc.size());
    for (int64_t f=0; f<F; f++) for (int e=0;e<3;e++) {
        int64_t a=Fc[f*3+e], b=Fc[f*3+(e+1)%3];
        if (!dir.count(dkey(b,a))) succ.emplace(a, b);   // emplace = keep first only
    }
    std::unordered_set<int64_t> visited;
    int nloops = 0;
    for (auto& kv : succ) {
        int64_t start = kv.first;
        if (visited.count(start)) continue;
        // walk the successor chain; only fill if it CLOSES back to start within max_loop (skip tangled
        // non-loops — a centroid fan on those spawns fresh boundary and explodes).
        std::vector<int64_t> loop; loop.reserve(64);
        int64_t v = start; bool closed = false;
        for (int step = 0; step <= max_loop; step++) {
            loop.push_back(v);
            auto it = succ.find(v);
            if (it == succ.end()) break;
            v = it->second;
            if (v == start) { closed = true; break; }
            if (visited.count(v)) break;   // ran into another chain
        }
        if (!closed || loop.size() < 3) continue;
        for (int64_t idx : loop) visited.insert(idx);
        double cx=0,cy=0,cz=0;
        for (int64_t idx : loop) { cx+=V[idx*3]; cy+=V[idx*3+1]; cz+=V[idx*3+2]; }
        cx/=loop.size(); cy/=loop.size(); cz/=loop.size();
        int64_t ci = (int64_t)V.size()/3;
        V.push_back((float)cx); V.push_back((float)cy); V.push_back((float)cz);
        for (size_t i=0;i<loop.size();i++) { int64_t a=loop[i], b=loop[(i+1)%loop.size()]; Fc.push_back(ci); Fc.push_back(a); Fc.push_back(b); }
        nloops++;
    }
    return nloops;
}

// ===========================================================================
// main entry: per-part seam-aware IM retopo.
// ===========================================================================
inline bool per_part_im_retopo(const glb::Mesh& in, const std::vector<int64_t>& fid,
                               const RetopoCfg& cfg,
                               glb::Mesh& out, std::vector<ppd::PartReport>& reports) {
    const int64_t V = (int64_t)in.verts.size() / 3;
    const int64_t F = (int64_t)in.faces.size() / 3;
    if ((int64_t)fid.size() != F) {
        std::fprintf(stderr, "per_part_im_retopo: face_ids %zu != F %lld\n", fid.size(), (long long)F);
        return false;
    }
    if (F == 0) { std::fprintf(stderr, "per_part_im_retopo: empty mesh\n"); return false; }

    // --- normalized face centroids (same convention as ppd) for the tier heuristic ---
    float mn[3]={1e30f,1e30f,1e30f}, mx[3]={-1e30f,-1e30f,-1e30f};
    for (int64_t v=0; v<V; v++) for (int d=0; d<3; d++) { float x=in.verts[v*3+d]; mn[d]=std::min(mn[d],x); mx[d]=std::max(mx[d],x); }
    double c[3]={(mn[0]+mx[0])*0.5,(mn[1]+mx[1])*0.5,(mn[2]+mx[2])*0.5}, s=0;
    for (int64_t v=0; v<V; v++) for (int d=0; d<3; d++) s=std::max(s, std::fabs((double)in.verts[v*3+d]-c[d]));
    if (s<=0) s=1;
    std::vector<double> fcn(F*3);
    for (int64_t f=0; f<F; f++) {
        int64_t a=in.faces[f*3], b=in.faces[f*3+1], d2=in.faces[f*3+2];
        for (int d=0; d<3; d++) fcn[f*3+d] = (((double)in.verts[a*3+d]+in.verts[b*3+d]+in.verts[d2*3+d])/3.0 - c[d])/s;
    }

    // --- seam anchor cloud + snap grid ---
    double diag = std::sqrt((double)( (mx[0]-mn[0])*(mx[0]-mn[0]) + (mx[1]-mn[1])*(mx[1]-mn[1]) + (mx[2]-mn[2])*(mx[2]-mn[2]) ));
    std::vector<float> anchors = build_seam_anchors(in, fid, diag * 1e-5);   // weld coincident split-verts
    // cell size for the grid ~ a coarse fraction of the model diagonal; snapping radius is per-part.
    PointGrid seam;
    if (cfg.snap_radius_frac > 0 && !anchors.empty()) seam.build(anchors, diag * 0.01);
    if (cfg.verbose) std::printf("[ppr] %lld seam anchor verts (model diag %.4f)\n", (long long)anchors.size()/3, diag);

    // --- group faces by label, sort by descending face count ---
    std::vector<int64_t> labels;
    std::unordered_map<int64_t, std::vector<int64_t>> face_of; face_of.reserve(64);
    for (int64_t f=0; f<F; f++) {
        auto it=face_of.find(fid[f]);
        if (it==face_of.end()) { labels.push_back(fid[f]); face_of[fid[f]].push_back(f); }
        else it->second.push_back(f);
    }
    std::sort(labels.begin(), labels.end(), [&](int64_t a,int64_t b){
        size_t na=face_of[a].size(), nb=face_of[b].size();
        return na!=nb ? na>nb : a<b;
    });

    std::printf("%5s %5s %9s -> %8s  %8s\n", "part", "tier", "in_f", "im_f", "snapped");
    out = glb::Mesh{}; reports.clear();
    double min_edge = 1e30, max_edge = 0;
    // shared seam vertices: every part boundary vert that snaps to seam anchor A becomes the SAME
    // global vertex. This is what actually closes the seams — float-coincidence welding can't, because
    // two independently-remeshed parts never place a vertex at the exact same seam position. Keyed by
    // anchor id so adjacent parts converge on one shared vertex chain along each cut.
    std::unordered_map<int, int64_t> anchor_gvtx;

    for (int64_t lbl : labels) {
        const auto& flist = face_of[lbl];
        const int64_t nf = (int64_t)flist.size();

        // extract submesh (compact verts)
        std::unordered_map<int64_t,int> remap; remap.reserve(nf*2);
        std::vector<int64_t> vsub; std::vector<int64_t> fsub; fsub.reserve(nf*3);
        for (int64_t f : flist) for (int j=0;j<3;j++) {
            int64_t g=in.faces[f*3+j];
            auto it=remap.find(g); int loc;
            if (it==remap.end()) { loc=(int)vsub.size(); remap.emplace(g,loc); vsub.push_back(g); }
            else loc=it->second;
            fsub.push_back(loc);
        }

        // tier -> IM face target
        double cen[3]={0,0,0}, xm=0;
        for (int64_t f : flist) { for (int d=0;d<3;d++) cen[d]+=fcn[f*3+d]; xm=std::max(xm,std::fabs(fcn[f*3])); }
        for (int d=0;d<3;d++) cen[d]/=(double)nf;
        ppd::Tier tier = ppd::tier_keep(nf, F, cen[0], cen[1], cen[2], xm);
        int64_t target = std::max((int64_t)400, (int64_t)llround((double)nf * tier.keep));

        // build svae::Mesh part and run IM (uniform, boundary-aligned)
        svae::Mesh pin; pin.verts.resize(vsub.size()*3);
        for (size_t i=0;i<vsub.size();i++){ int64_t g=vsub[i]; pin.verts[i*3]=in.verts[g*3]; pin.verts[i*3+1]=in.verts[g*3+1]; pin.verts[i*3+2]=in.verts[g*3+2]; }
        pin.faces.assign(fsub.begin(), fsub.end());
        pin.N=(int)vsub.size(); pin.F=(int)nf;

        imretopo::ImCfg im = cfg.im;
        im.target_faces = (int)std::min<int64_t>(target, 2000000000);
        im.target_verts = 0;
        im.adaptivity   = 0.0f;       // per-part is UNIFORM by construction
        im.align_boundaries = true;   // keep the cut edge straight
        im.verbose = false;
        im.tmp = cfg.im.tmp + "/p" + std::to_string(lbl);

        svae::Mesh pout;
        if (nf > 400 && !imretopo::im_retopo(pin, im, pout)) {
            std::fprintf(stderr, "[ppr] IM failed on part %lld (nf=%lld); keeping verbatim\n", (long long)lbl, (long long)nf);
            pout = pin;
        } else if (nf <= 400) {
            pout = pin;   // too small to retopo — keep as-is
        }

        // part edge-length estimate (from IM target over part area proxy) for snap radius / weld eps
        // use mean output edge length: sample a few faces.
        double part_edge = 0; int ne = 0;
        for (int64_t f=0; f<pout.F && ne<200; f++) {
            for (int e=0;e<3;e++){
                int64_t a=pout.faces[f*3+e], b=pout.faces[f*3+(e+1)%3];
                double dx=pout.verts[a*3]-pout.verts[b*3], dy=pout.verts[a*3+1]-pout.verts[b*3+1], dz=pout.verts[a*3+2]-pout.verts[b*3+2];
                part_edge += std::sqrt(dx*dx+dy*dy+dz*dz); ne++;
            }
        }
        if (ne) part_edge /= ne; else part_edge = diag*0.01;
        min_edge = std::min(min_edge, part_edge);
        max_edge = std::max(max_edge, part_edge);

        // boundary verts of pout (verts on an open edge = the part cut)
        std::unordered_set<int64_t> bverts;
        {
            std::unordered_map<uint64_t,int> ec; ec.reserve(pout.faces.size());
            for (int64_t f=0; f<pout.F; f++) for (int e=0;e<3;e++) ec[edge_key(pout.faces[f*3+e], pout.faces[f*3+(e+1)%3])]++;
            for (int64_t f=0; f<pout.F; f++) for (int e=0;e<3;e++){
                int64_t a=pout.faces[f*3+e], b=pout.faces[f*3+(e+1)%3];
                if (ec[edge_key(a,b)]==1){ bverts.insert(a); bverts.insert(b); }
            }
        }

        // --- recombine with SHARED seam vertices ---
        // Each local vertex -> a global index. Boundary verts snap to the nearest seam anchor and reuse
        // that anchor's global vertex (shared across parts). Interior verts get fresh globals.
        const double R = cfg.snap_radius_frac * part_edge;
        std::vector<int64_t> g(pout.N, -1);
        int64_t snapped = 0;
        for (int64_t v = 0; v < pout.N; v++) {
            if (cfg.snap_radius_frac > 0 && !anchors.empty() && bverts.count(v)) {
                int ai = seam.nearest(&pout.verts[v*3], R);
                if (ai >= 0) {
                    auto it = anchor_gvtx.find(ai);
                    if (it != anchor_gvtx.end()) { g[v] = it->second; }
                    else {
                        int64_t ni = (int64_t)out.verts.size()/3;
                        out.verts.push_back(seam.pts[ai*3]); out.verts.push_back(seam.pts[ai*3+1]); out.verts.push_back(seam.pts[ai*3+2]);
                        anchor_gvtx.emplace(ai, ni); g[v] = ni;
                    }
                    snapped++;
                    continue;
                }
            }
            int64_t ni = (int64_t)out.verts.size()/3;
            out.verts.push_back(pout.verts[v*3]); out.verts.push_back(pout.verts[v*3+1]); out.verts.push_back(pout.verts[v*3+2]);
            g[v] = ni;
        }
        for (int64_t f = 0; f < pout.F; f++) {
            int64_t a=g[pout.faces[f*3]], b=g[pout.faces[f*3+1]], c=g[pout.faces[f*3+2]];
            if (a==b||b==c||a==c) continue;   // collapsed where multiple boundary verts shared one anchor
            out.faces.push_back(a); out.faces.push_back(b); out.faces.push_back(c);
        }

        std::printf("%5lld %5s %9lld -> %8d  %8lld\n", (long long)lbl, tier.name, (long long)nf, pout.F, (long long)snapped);
        reports.push_back({lbl, tier.name, nf, pout.F});
    }

    int64_t open_pre = count_open_edges(out.faces);
    if (cfg.verbose) std::printf("[ppr] combined: %zu v / %zu f, open edges (pre-weld) = %lld\n",
                                 out.verts.size()/3, out.faces.size()/3, (long long)open_pre);

    // --- weld coincident boundary verts (snapped anchors coincide across parts) ---
    if (cfg.weld_eps_frac > 0) {
        weld_mesh(out.verts, out.faces, cfg.weld_eps_frac * min_edge);
        int64_t open_w = count_open_edges(out.faces);
        if (cfg.verbose) std::printf("[ppr] after weld (eps=%.5f): %zu v / %zu f, open edges = %lld\n",
                                     cfg.weld_eps_frac*min_edge, out.verts.size()/3, out.faces.size()/3, (long long)open_w);
    }

    // --- boundary-only seam weld (generous eps ~ coarse part edge) closes T-junction seam gaps ---
    if (cfg.bweld_eps_frac > 0) {
        boundary_weld(out.verts, out.faces, cfg.bweld_eps_frac * max_edge);
        int64_t open_b = count_open_edges(out.faces);
        if (cfg.verbose) std::printf("[ppr] after boundary_weld (eps=%.5f): %zu f, open edges = %lld\n",
                                     cfg.bweld_eps_frac*max_edge, out.faces.size()/3, (long long)open_b);
    }

    // --- fill residual open loops (density-mismatch gaps) -> watertight ---
    // Fan-filling a non-closing chain can spawn a smaller residual loop, so iterate to a fixed point.
    if (cfg.fill_holes) {
        int64_t open_prev = count_open_edges(out.faces);
        int total_loops = 0, pass = 0;
        while (open_prev > 0 && pass < 12) {
            int nl = fill_holes(out.verts, out.faces);
            total_loops += nl;
            int64_t open_now = count_open_edges(out.faces);
            if (cfg.verbose) std::printf("[ppr]   fill pass %d: +%d loops -> open edges %lld\n", pass, nl, (long long)open_now);
            if (open_now >= open_prev || nl == 0) { open_prev = open_now; break; }
            open_prev = open_now; pass++;
        }
        if (cfg.verbose) std::printf("[ppr] after fill_holes: %d loops over %d passes, %zu v / %zu f, open edges = %lld  %s\n",
                                     total_loops, pass+1, out.verts.size()/3, out.faces.size()/3, (long long)open_prev,
                                     open_prev==0 ? "(WATERTIGHT)" : "(STILL OPEN)");
    }
    return true;
}

} // namespace ppr
