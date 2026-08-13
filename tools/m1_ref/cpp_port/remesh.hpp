// Proper watertight REMESH for the Pixal3D port (A1) — replaces the close_surface + ear-fill +
// mirror-cap hack (which reached boundary==0 only by ADDING ~14.6k degenerate flap tris and left
// the mesh non-manifold, so meshopt fell back to simplifySloppy and xatlas never collapsed charts).
//
// Algorithm = MARCHING TETRAHEDRA on the grid-1024 binary occupancy (the ~1.5M occupied voxel
// coords the M4 decoder grows). Why MT over classic Marching Cubes: a tetrahedron has NO ambiguous
// cases (unlike the 6 ambiguous MC cases that crack/non-manifold without a topology-correct 256-
// entry table). With the translation-invariant Kuhn 6-tet decomposition (all tets share the cube
// main diagonal 0-7), adjacent cubes split every shared face along the SAME world diagonal, so the
// surface is GUARANTEED watertight (0 boundary edges) AND 2-manifold (every edge has exactly 2
// incident faces) — derivable from first principles, no magic table. Vertices on shared grid edges
// are deduplicated by a canonical edge key, so there are no T-junctions or duplicate verts either.
//
// The output is the boundary between occupied and empty voxels at the 1/1024 lattice (sub-mm), so
// it loses the learned sub-voxel dual-grid vertices, but at grid1024 that displacement is invisible
// (handoff: "sub-mm = invisible"); the clean manifold topology unblocks quality decimation + a tight
// 2048² atlas. Textures are baked volumetrically (grid_sample at the rasterized 3D position), so the
// remeshed topology is independent of texturing — the PBR volume bake works unchanged.
#pragma once
#include "sparse_vae.hpp"     // svae::Mesh, coord_key, boundary_edge_count
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <functional>
#include <cstdio>
#include <cstdlib>           // getenv/atoi/abort (REMESH_CLOSE_R seal knob)
#include <cstring>           // memset (separable box morphology)

namespace svae {

inline void orient_consistent(Mesh& m);   // defined below — fixes MC winding flips

// Kuhn (Freudenthal) triangulation of the unit cube into 6 tetrahedra, each sharing the main
// diagonal corner0(0,0,0)->corner7(1,1,1). Corner index c packs bits (cx,cy,cz)=(c&1,(c>>1)&1,(c>>2)&1).
// Each tet = {0, +firstAxis, +firstAxis+secondAxis, 7} over the 6 axis orderings. Being a pure
// translation-invariant pattern, neighbouring cubes agree on every shared-face diagonal => conforming.
static const int MT_TETS[6][4] = {
    {0,1,3,7},  // x,y,z
    {0,1,5,7},  // x,z,y
    {0,2,3,7},  // y,x,z
    {0,2,6,7},  // y,z,x
    {0,4,5,7},  // z,x,y
    {0,4,6,7},  // z,y,x
};

// Marching Tetrahedra on a binary occupancy set. coords: int32 [N,4] = (b,x,y,z) occupied voxels
// (batch ignored — single object). grid_size = lattice resolution (1024). Returns a clean manifold
// watertight triangle Mesh; vertex world pos matches the dual-grid frame ( (c+0.5)/grid - 0.5 voxel
// centres ) so the tex-atlas volumetric bake ( q=(P+0.5)*grid ) samples the right voxels.
inline Mesh marching_tetrahedra(const int32_t* coords, int N, int grid_size) {
    const float voxel = 1.0f / (float)grid_size, aabb0 = -0.5f;

    // occupancy set (x,y,z) -> index unused; we only need membership.
    std::unordered_map<int64_t, char> occ;
    occ.reserve((size_t)N * 2);
    for (int i = 0; i < N; i++)
        occ.emplace(coord_key(0, coords[i*4+1], coords[i*4+2], coords[i*4+3]), (char)1);
    auto is_occ = [&](int x, int y, int z) -> bool {
        return occ.find(coord_key(0, x, y, z)) != occ.end();
    };

    // candidate cells = every cube incident to an occupied voxel (the voxel is one of its 8 corners).
    // A cube's min-corner is (x-dx,y-dy,z-dz) for d in {0,1}^3. Dedup via a set of cube min-corner keys.
    std::unordered_map<int64_t, char> cells;
    cells.reserve((size_t)N * 4);
    for (int i = 0; i < N; i++) {
        int x = coords[i*4+1], y = coords[i*4+2], z = coords[i*4+3];
        for (int dz = -1; dz <= 0; dz++)
            for (int dy = -1; dy <= 0; dy++)
                for (int dx = -1; dx <= 0; dx++)
                    cells.emplace(coord_key(0, x+dx, y+dy, z+dz), (char)1);
    }

    Mesh m; m.N = 0; m.F = 0;
    // edge-midpoint vertex dedup: canonical key from the min grid endpoint + axis -> vertex index.
    std::unordered_map<int64_t, int> vmap;
    vmap.reserve(cells.size() * 3);

    // grid-point world position (voxel centre).
    auto gp_world = [&](int x, int y, int z, int d) -> float {
        int c = (d==0)?x : (d==1)?y : z;
        return ((float)c + 0.5f) * voxel + aabb0;
    };
    // get-or-create the vertex on a TET edge between grid points A and B. NB tet edges are not only
    // axis-aligned cube edges — the Kuhn tets also have face-diagonal and main-diagonal edges (A,B
    // differ on up to 3 axes). Canonicalise the unordered global pair as (lo-corner, direction-mask):
    // lo = componentwise min, dmask = axes on which they differ (1..7). This dedups the SAME grid edge
    // across every cube/tet that touches it (the watertight+manifold guarantee depends on this).
    auto edge_vert = [&](int ax,int ay,int az, int bx,int by,int bz) -> int {
        int lx=ax<bx?ax:bx, ly=ay<by?ay:by, lz=az<bz?az:bz;
        int dm = (ax!=bx?1:0) | (ay!=by?2:0) | (az!=bz?4:0);
        // offset +2 (handles min-corner -1) -> 12-bit fields; pack (lo,dmask) into one int64
        int64_t key = ((((int64_t)(lx+2)*4096 + (ly+2))*4096 + (lz+2)) * 8) + dm;
        auto it = vmap.find(key);
        if (it != vmap.end()) return it->second;
        int idx = (int)(m.verts.size()/3);
        // midpoint of the two grid-point centres
        m.verts.push_back(0.5f*(gp_world(ax,ay,az,0)+gp_world(bx,by,bz,0)));
        m.verts.push_back(0.5f*(gp_world(ax,ay,az,1)+gp_world(bx,by,bz,1)));
        m.verts.push_back(0.5f*(gp_world(ax,ay,az,2)+gp_world(bx,by,bz,2)));
        vmap.emplace(key, idx);
        return idx;
    };

    // corner offset (cx,cy,cz) per corner index
    auto cxyz = [](int c, int& x, int& y, int& z){ x=c&1; y=(c>>1)&1; z=(c>>2)&1; };

    for (auto& kv : cells) {
        // unpack cube min-corner (mask back the 20-bit packed coord; coords are non-negative here)
        int64_t k = kv.first;
        int z0 = (int)( k        & 0xFFFFF);
        int y0 = (int)((k >> 20) & 0xFFFFF);
        int x0 = (int)((k >> 40) & 0xFFFFF);
        // 8 corner grid coords + occupancy
        int cgx[8],cgy[8],cgz[8]; bool cin[8]; int nin=0;
        for (int c=0;c<8;c++){ int ox,oy,oz; cxyz(c,ox,oy,oz);
            cgx[c]=x0+ox; cgy[c]=y0+oy; cgz[c]=z0+oz;
            cin[c]=is_occ(cgx[c],cgy[c],cgz[c]); if(cin[c]) nin++; }
        if (nin==0 || nin==8) continue;            // fully outside / inside -> no surface

        for (int t=0;t<6;t++) {
            const int* T = MT_TETS[t];
            int in[4]; int ni=0;
            for (int j=0;j<4;j++){ in[j]=cin[T[j]]?1:0; ni+=in[j]; }
            if (ni==0 || ni==4) continue;

            // outward direction = (sum outside corner pos) - (sum inside corner pos)
            float od[3]={0,0,0};
            for (int j=0;j<4;j++){ int c=T[j]; float s=in[j]?-1.f:1.f;
                od[0]+=s*cgx[c]; od[1]+=s*cgy[c]; od[2]+=s*cgz[c]; }

            auto emit_tri = [&](int v0,int v1,int v2){
                const float *p0=&m.verts[(size_t)v0*3],*p1=&m.verts[(size_t)v1*3],*p2=&m.verts[(size_t)v2*3];
                float e1[3]={p1[0]-p0[0],p1[1]-p0[1],p1[2]-p0[2]}, e2[3]={p2[0]-p0[0],p2[1]-p0[1],p2[2]-p0[2]};
                float nx=e1[1]*e2[2]-e1[2]*e2[1], ny=e1[2]*e2[0]-e1[0]*e2[2], nz=e1[0]*e2[1]-e1[1]*e2[0];
                if (nx*od[0]+ny*od[1]+nz*od[2] < 0.f) std::swap(v1,v2);   // face outward
                m.faces.push_back(v0); m.faces.push_back(v1); m.faces.push_back(v2);
            };
            auto ev = [&](int ca,int cb){ return edge_vert(cgx[ca],cgy[ca],cgz[ca], cgx[cb],cgy[cb],cgz[cb]); };

            if (ni==1 || ni==3) {
                // one corner on the minority side -> triangle on its 3 edges
                int s = (ni==1)?1:0;                  // minority value
                int a=-1; for (int j=0;j<4;j++) if (in[j]==s){ a=j; break; }
                int b[3],bi=0; for (int j=0;j<4;j++) if (j!=a) b[bi++]=j;
                int v0=ev(T[a],T[b[0]]), v1=ev(T[a],T[b[1]]), v2=ev(T[a],T[b[2]]);
                emit_tri(v0,v1,v2);
            } else {
                // ni==2: quad across the 4 in-out edges -> 2 triangles
                int insd[2],outd[2],ii=0,oo=0;
                for (int j=0;j<4;j++){ if(in[j]) insd[ii++]=j; else outd[oo++]=j; }
                int e00=ev(T[insd[0]],T[outd[0]]);
                int e01=ev(T[insd[0]],T[outd[1]]);
                int e10=ev(T[insd[1]],T[outd[0]]);
                int e11=ev(T[insd[1]],T[outd[1]]);
                emit_tri(e00,e01,e11);
                emit_tri(e00,e11,e10);
            }
        }
    }
    m.N = (int)(m.verts.size()/3);
    m.F = (int)(m.faces.size()/3);
    return m;
}

// ---------------------------------------------------------------------------------------------
// SMOOTHED-FIELD Marching Tetrahedra (A1 follow-up — the tight-atlas fix).
//
// The plain `marching_tetrahedra` above runs on BINARY occupancy: every grid point is 0 or 1, so
// every iso-crossing lands at the edge MIDPOINT → a diagonally-faceted "staircase" surface whose
// per-triangle normals are noisy. That noise (not topology) is what defeats meshopt quality
// decimation (it stalls ~13.8M faces) and makes xatlas seed ~48k charts. Taubin-smoothing the
// 8M-vertex staircase after the fact barely helps and is slow.
//
// FIX: build a SMOOTHED scalar field f(x,y,z) ∈ [0,1] = a box-blur of the binary occupancy, then
// march on f with iso=0.5 and LINEAR edge interpolation (t = (iso-fa)/(fb-fa)). The crossing now
// moves continuously with the field → a SMOOTH low-curvature surface whose quadric decimation WORKS
// and whose normals are coherent → tight atlas. The Kuhn-tet topology guarantee is UNCHANGED:
//   * classification cin[c] = (f(corner) >= iso) is a per-grid-point predicate, identical from every
//     cube that shares the corner → conforming sign field → still watertight + 2-manifold.
//   * the interpolated vertex position depends only on the two endpoint field values + iso, which
//     are deterministic per shared grid edge → the (lo-corner,dmask) dedup gives ONE vertex per
//     edge (no T-junctions); and t↔(1-t) under endpoint swap yields the SAME world point, so the
//     emit order doesn't matter.
// blur_radius r → (2r+1)³ box kernel over the binary occupancy (r=1 → 3³, r=2 → 5³). Larger r =
// smoother (rounder) surface, fewer charts, but more rounding of fine features. iso=0.5 keeps the
// surface centred on the occupied boundary (box-blur of a half-space is symmetric about it).
inline Mesh marching_tetrahedra_field(const int32_t* coords, int N, int grid_size,
                                      int blur_radius = 1, float iso = 0.5f) {
    const float voxel = 1.0f / (float)grid_size, aabb0 = -0.5f;

    std::unordered_map<int64_t, char> occ;
    occ.reserve((size_t)N * 2);
    for (int i = 0; i < N; i++)
        occ.emplace(coord_key(0, coords[i*4+1], coords[i*4+2], coords[i*4+3]), (char)1);
    auto is_occ = [&](int x, int y, int z) -> bool {
        return occ.find(coord_key(0, x, y, z)) != occ.end();
    };

    // memoized smoothed field at a grid point: fraction of occupied voxels in the (2r+1)³ box.
    const int r = blur_radius;
    const float inv_box = 1.0f / (float)((2*r+1)*(2*r+1)*(2*r+1));
    std::unordered_map<int64_t, float> fmap;
    fmap.reserve((size_t)N * 4);
    auto field_at = [&](int x, int y, int z) -> float {
        int64_t k = coord_key(0, x, y, z);
        auto it = fmap.find(k);
        if (it != fmap.end()) return it->second;
        int cnt = 0;
        for (int dz=-r; dz<=r; dz++)
            for (int dy=-r; dy<=r; dy++)
                for (int dx=-r; dx<=r; dx++)
                    if (is_occ(x+dx, y+dy, z+dz)) cnt++;
        float f = (float)cnt * inv_box;
        fmap.emplace(k, f);
        return f;
    };

    // candidate cells: cubes incident to an occupied voxel (covers the iso=0.5 boundary, which lives
    // within ~1 voxel of the occupied set for a symmetric blur).
    std::unordered_map<int64_t, char> cells;
    cells.reserve((size_t)N * 4);
    for (int i = 0; i < N; i++) {
        int x = coords[i*4+1], y = coords[i*4+2], z = coords[i*4+3];
        for (int dz = -1; dz <= 0; dz++)
            for (int dy = -1; dy <= 0; dy++)
                for (int dx = -1; dx <= 0; dx++)
                    cells.emplace(coord_key(0, x+dx, y+dy, z+dz), (char)1);
    }

    Mesh m; m.N = 0; m.F = 0;
    std::unordered_map<int64_t, int> vmap;
    vmap.reserve(cells.size() * 3);

    auto gp_world = [&](int c, int d) -> float { (void)d; return ((float)c + 0.5f) * voxel + aabb0; };
    // get-or-create the interpolated vertex on the tet edge A->B (A,B grid coords). Canonical key as
    // before. Position = linear iso-crossing using the smoothed field at A and B.
    auto edge_vert = [&](int ax,int ay,int az, int bx,int by,int bz) -> int {
        int lx=ax<bx?ax:bx, ly=ay<by?ay:by, lz=az<bz?az:bz;
        int dm = (ax!=bx?1:0) | (ay!=by?2:0) | (az!=bz?4:0);
        int64_t key = ((((int64_t)(lx+2)*4096 + (ly+2))*4096 + (lz+2)) * 8) + dm;
        auto it = vmap.find(key);
        if (it != vmap.end()) return it->second;
        float fa = field_at(ax,ay,az), fb = field_at(bx,by,bz);
        float denom = fb - fa;
        float t = (fabsf(denom) < 1e-6f) ? 0.5f : (iso - fa) / denom;
        if (t < 0.01f) t = 0.01f; else if (t > 0.99f) t = 0.99f;   // avoid degenerate (zero-len) tris
        int idx = (int)(m.verts.size()/3);
        m.verts.push_back(gp_world(ax,0) + t*(gp_world(bx,0)-gp_world(ax,0)));
        m.verts.push_back(gp_world(ay,1) + t*(gp_world(by,1)-gp_world(ay,1)));
        m.verts.push_back(gp_world(az,2) + t*(gp_world(bz,2)-gp_world(az,2)));
        vmap.emplace(key, idx);
        return idx;
    };

    auto cxyz = [](int c, int& x, int& y, int& z){ x=c&1; y=(c>>1)&1; z=(c>>2)&1; };

    for (auto& kv : cells) {
        int64_t k = kv.first;
        int z0 = (int)( k        & 0xFFFFF);
        int y0 = (int)((k >> 20) & 0xFFFFF);
        int x0 = (int)((k >> 40) & 0xFFFFF);
        int cgx[8],cgy[8],cgz[8]; bool cin[8]; int nin=0;
        for (int c=0;c<8;c++){ int ox,oy,oz; cxyz(c,ox,oy,oz);
            cgx[c]=x0+ox; cgy[c]=y0+oy; cgz[c]=z0+oz;
            cin[c]=(field_at(cgx[c],cgy[c],cgz[c]) >= iso); if(cin[c]) nin++; }
        if (nin==0 || nin==8) continue;

        for (int t=0;t<6;t++) {
            const int* T = MT_TETS[t];
            int in[4]; int ni=0;
            for (int j=0;j<4;j++){ in[j]=cin[T[j]]?1:0; ni+=in[j]; }
            if (ni==0 || ni==4) continue;
            float od[3]={0,0,0};
            for (int j=0;j<4;j++){ int c=T[j]; float s=in[j]?-1.f:1.f;
                od[0]+=s*cgx[c]; od[1]+=s*cgy[c]; od[2]+=s*cgz[c]; }
            auto emit_tri = [&](int v0,int v1,int v2){
                const float *p0=&m.verts[(size_t)v0*3],*p1=&m.verts[(size_t)v1*3],*p2=&m.verts[(size_t)v2*3];
                float e1[3]={p1[0]-p0[0],p1[1]-p0[1],p1[2]-p0[2]}, e2[3]={p2[0]-p0[0],p2[1]-p0[1],p2[2]-p0[2]};
                float nx=e1[1]*e2[2]-e1[2]*e2[1], ny=e1[2]*e2[0]-e1[0]*e2[2], nz=e1[0]*e2[1]-e1[1]*e2[0];
                if (nx*od[0]+ny*od[1]+nz*od[2] < 0.f) std::swap(v1,v2);
                m.faces.push_back(v0); m.faces.push_back(v1); m.faces.push_back(v2);
            };
            auto ev = [&](int ca,int cb){ return edge_vert(cgx[ca],cgy[ca],cgz[ca], cgx[cb],cgy[cb],cgz[cb]); };
            if (ni==1 || ni==3) {
                int s = (ni==1)?1:0;
                int a=-1; for (int j=0;j<4;j++) if (in[j]==s){ a=j; break; }
                int b[3],bi=0; for (int j=0;j<4;j++) if (j!=a) b[bi++]=j;
                int v0=ev(T[a],T[b[0]]), v1=ev(T[a],T[b[1]]), v2=ev(T[a],T[b[2]]);
                emit_tri(v0,v1,v2);
            } else {
                int insd[2],outd[2],ii=0,oo=0;
                for (int j=0;j<4;j++){ if(in[j]) insd[ii++]=j; else outd[oo++]=j; }
                int e00=ev(T[insd[0]],T[outd[0]]);
                int e01=ev(T[insd[0]],T[outd[1]]);
                int e10=ev(T[insd[1]],T[outd[0]]);
                int e11=ev(T[insd[1]],T[outd[1]]);
                emit_tri(e00,e01,e11);
                emit_tri(e00,e11,e10);
            }
        }
    }
    m.N = (int)(m.verts.size()/3);
    m.F = (int)(m.faces.size()/3);
    return m;
}

// ---------------------------------------------------------------------------------------------
// COARSE-GRID Marching Tetrahedra (the actual tight-atlas win). Instead of marching the full 1024
// lattice (12M faces → must decimate, and quadric decimation STALLS on the lattice-regular surface)
// then chasing decimation, march a DOWNSAMPLED field directly: build a coarse occupancy fraction
// field at grid (1024/stride) and march it. Output is a LOW-POLY (~10²–10³ k faces, set by stride),
// SMOOTH (the box-average is the low-pass), WATERTIGHT, COHERENT-NORMAL manifold mesh — exactly what
// xatlas wants → charts in the tens-hundreds, no decimation step at all. This is the modern
// occupancy→clean-mesh recipe (coarse SDF + MC), and it's seconds on CPU.
//
// stride = fine voxels per coarse cell (1024/stride = coarse grid; e.g. stride 4 → grid 256). The
// coarse field at point X = (# occupied fine voxels in the surrounding coarse-cell box) / box_vox,
// optionally box-blurred over `blur` coarse neighbours for extra smoothing. iso=0.5. Candidate cells
// are properly DILATED (cheap at coarse res) so the iso surface is never clipped → boundary==0.
inline Mesh marching_cubes_coarse(const int32_t* coords, int N, int fine_grid, int stride,
                                  int blur = 0, float iso = 0.5f) {
    const int Gc = fine_grid / stride;                 // coarse grid resolution
    const float voxel_c = (float)stride / (float)fine_grid, aabb0 = -0.5f;
    const float inv_cell = 1.0f / (float)(stride*stride*stride);

    // coarse occupancy COUNT: fine voxel (x,y,z) → coarse (x/stride,...). count[coarse]++
    std::unordered_map<int64_t, int> ccount;
    ccount.reserve((size_t)N);
    for (int i = 0; i < N; i++) {
        int X = coords[i*4+1]/stride, Y = coords[i*4+2]/stride, Z = coords[i*4+3]/stride;
        ccount[coord_key(0, X, Y, Z)] += 1;
    }
    // raw coarse field (fraction occupied in the cell)
    auto raw_field = [&](int X,int Y,int Z) -> float {
        auto it = ccount.find(coord_key(0,X,Y,Z));
        return it == ccount.end() ? 0.0f : (float)it->second * inv_cell;
    };
    // optional box-blur over coarse neighbours (memoized)
    const int r = blur;
    const float inv_box = 1.0f / (float)((2*r+1)*(2*r+1)*(2*r+1));
    std::unordered_map<int64_t, float> fmap;
    fmap.reserve(ccount.size()*4);
    auto field_at = [&](int X,int Y,int Z) -> float {
        if (r == 0) return raw_field(X,Y,Z);
        int64_t k = coord_key(0,X,Y,Z);
        auto it = fmap.find(k); if (it!=fmap.end()) return it->second;
        float s=0.f; for(int dz=-r;dz<=r;dz++)for(int dy=-r;dy<=r;dy++)for(int dx=-r;dx<=r;dx++) s+=raw_field(X+dx,Y+dy,Z+dz);
        float f=s*inv_box; fmap.emplace(k,f); return f;
    };

    // candidate cells: every coarse cube incident to a coarse grid point with field could-cross.
    // Dilate the occupied-coarse set by (blur+1) so no surface cube is missed (cheap at coarse res).
    std::unordered_map<int64_t,char> cells; cells.reserve(ccount.size()*8);
    const int d = r + 1;
    for (auto& kv : ccount) {
        int64_t k = kv.first;
        int Z0 = (int)( k        & 0xFFFFF), Y0 = (int)((k>>20)&0xFFFFF), X0=(int)((k>>40)&0xFFFFF);
        for (int dz=-d; dz<=d-1; dz++) for (int dy=-d; dy<=d-1; dy++) for (int dx=-d; dx<=d-1; dx++)
            cells.emplace(coord_key(0, X0+dx, Y0+dy, Z0+dz), (char)1);
    }

    Mesh m; m.N=0; m.F=0;
    std::unordered_map<int64_t,int> vmap; vmap.reserve(cells.size()*3);
    auto gp_world = [&](int c) -> float { return ((float)c + 0.5f) * voxel_c + aabb0; };
    auto edge_vert = [&](int ax,int ay,int az,int bx,int by,int bz)->int{
        int lx=ax<bx?ax:bx, ly=ay<by?ay:by, lz=az<bz?az:bz;
        int dm=(ax!=bx?1:0)|(ay!=by?2:0)|(az!=bz?4:0);
        int64_t key=((((int64_t)(lx+2)*4096+(ly+2))*4096+(lz+2))*8)+dm;
        auto it=vmap.find(key); if(it!=vmap.end()) return it->second;
        float fa=field_at(ax,ay,az), fb=field_at(bx,by,bz), den=fb-fa;
        float t=(fabsf(den)<1e-6f)?0.5f:(iso-fa)/den; if(t<0.01f)t=0.01f; else if(t>0.99f)t=0.99f;
        int idx=(int)(m.verts.size()/3);
        m.verts.push_back(gp_world(ax)+t*(gp_world(bx)-gp_world(ax)));
        m.verts.push_back(gp_world(ay)+t*(gp_world(by)-gp_world(ay)));
        m.verts.push_back(gp_world(az)+t*(gp_world(bz)-gp_world(az)));
        vmap.emplace(key,idx); return idx;
    };
    auto cxyz=[](int c,int&x,int&y,int&z){x=c&1;y=(c>>1)&1;z=(c>>2)&1;};
    for (auto& kv : cells) {
        int64_t k=kv.first; int z0=(int)(k&0xFFFFF), y0=(int)((k>>20)&0xFFFFF), x0=(int)((k>>40)&0xFFFFF);
        int cgx[8],cgy[8],cgz[8]; bool cin[8]; int nin=0;
        for(int c=0;c<8;c++){int ox,oy,oz;cxyz(c,ox,oy,oz);cgx[c]=x0+ox;cgy[c]=y0+oy;cgz[c]=z0+oz;
            cin[c]=(field_at(cgx[c],cgy[c],cgz[c])>=iso); if(cin[c])nin++;}
        if(nin==0||nin==8) continue;
        for(int t=0;t<6;t++){ const int* T=MT_TETS[t]; int in[4],ni=0;
            for(int j=0;j<4;j++){in[j]=cin[T[j]]?1:0;ni+=in[j];} if(ni==0||ni==4)continue;
            float od[3]={0,0,0}; for(int j=0;j<4;j++){int c=T[j];float s=in[j]?-1.f:1.f;od[0]+=s*cgx[c];od[1]+=s*cgy[c];od[2]+=s*cgz[c];}
            auto emit_tri=[&](int v0,int v1,int v2){
                const float*p0=&m.verts[(size_t)v0*3],*p1=&m.verts[(size_t)v1*3],*p2=&m.verts[(size_t)v2*3];
                float e1[3]={p1[0]-p0[0],p1[1]-p0[1],p1[2]-p0[2]},e2[3]={p2[0]-p0[0],p2[1]-p0[1],p2[2]-p0[2]};
                float nx=e1[1]*e2[2]-e1[2]*e2[1],ny=e1[2]*e2[0]-e1[0]*e2[2],nz=e1[0]*e2[1]-e1[1]*e2[0];
                if(nx*od[0]+ny*od[1]+nz*od[2]<0.f) std::swap(v1,v2);
                m.faces.push_back(v0);m.faces.push_back(v1);m.faces.push_back(v2);};
            auto ev=[&](int ca,int cb){return edge_vert(cgx[ca],cgy[ca],cgz[ca],cgx[cb],cgy[cb],cgz[cb]);};
            if(ni==1||ni==3){int s=(ni==1)?1:0;int a=-1;for(int j=0;j<4;j++)if(in[j]==s){a=j;break;}
                int b[3],bi=0;for(int j=0;j<4;j++)if(j!=a)b[bi++]=j;
                emit_tri(ev(T[a],T[b[0]]),ev(T[a],T[b[1]]),ev(T[a],T[b[2]]));}
            else{int insd[2],outd[2],ii=0,oo=0;for(int j=0;j<4;j++){if(in[j])insd[ii++]=j;else outd[oo++]=j;}
                int e00=ev(T[insd[0]],T[outd[0]]),e01=ev(T[insd[0]],T[outd[1]]),e10=ev(T[insd[1]],T[outd[0]]),e11=ev(T[insd[1]],T[outd[1]]);
                emit_tri(e00,e01,e11);emit_tri(e00,e11,e10);}
        }
    }
    m.N=(int)(m.verts.size()/3); m.F=(int)(m.faces.size()/3);
    (void)Gc;
    orient_consistent(m);
    return m;
}

// ---------------------------------------------------------------------------------------------
// SOLIDIFY + COARSE Marching Cubes — the correct shell→clean-mesh recipe (THE tight-atlas win).
//
// CRITICAL realisation: the M4 occupancy is a thin SURFACE SHELL (~1.47M voxels = surface, not a
// filled solid). Box-averaging a thin shell and thresholding at iso=0.5 only keeps cells where the
// shell is locally dense → it SHATTERS into ~1200 disconnected fragments (genus≈-814), and xatlas
// then needs a chart per fragment → 24k charts. The fix is to mesh the OUTER boundary of the SOLID
// the shell encloses, not the shell itself:
//   1. coarse-rasterise: coarse cell = occupied if ANY fine voxel in it is occupied (thickens the
//      shell → seals the small gaps the M4 extractor leaves, so the flood-fill can't leak inside).
//   2. SEAL (REMESH_CLOSE_R, below) — the coarse rasterise ALONE does NOT stop the leak.
//   3. flood-fill the EXTERIOR on the coarse grid (6-connectivity from the padded bbox shell).
//   4. solid = NOT exterior (fills the interior cavity the shell encloses) → one connected volume.
//   5. box-blur the binary solid → smooth field, MC at iso=0.5 + linear interp → a SMOOTH,
//      WATERTIGHT, SINGLE-COMPONENT, coherent-normal low-poly outer surface → tight atlas, no
//      decimation step. Seconds on CPU. (This is what cumesh's dual-contour remesh achieves; same
//      end product, open recipe.)
//
// ============================ THE LEAK, AND WHY THE SEAL IS SHAPED LIKE THIS ==================
// Step 1's comment ("so the flood-fill can't leak inside") was WRONG, and the bug it let through
// shipped: the exterior flood LEAKED into the interior, so `solid` collapsed back onto the seed
// shell and MC meshed the SHELL — an outer skin PLUS a never-visible inner skin ~1 wall beneath it
// (a "vacuum-formed toy"). Measured on the soldier @res1536/stride2 (grid 768³): solid/wall = 1.000.
// ~37% of the shipped mesh's area was interior, explaining ~73% of the texture-bake holes.
//
// WHY it leaks — digital topology, not a coding slip: the coarse shell is ~1 cell thick, and a
// 1-cell-thick 26-connected (diagonal/staircase) surface does NOT separate a 6-connected
// background. The 6-neighbour flood walks straight through the diagonal steps. This is the
// standard (26-object ⇒ 6-background) pairing: such a shell is NOT 6-separating, by construction.
//
// WHY A PLAIN "CLOSE BEFORE THE FILL" DOES NOT FIX IT (the intuitive fix — it is a NO-OP here):
// a morphological close (dilate∘erode) fills gaps SMALLER THAN the structuring element, but a
// diagonal staircase has no such gap — it is already closed w.r.t. a box SE. Worked 2-D example,
// X = {(i,i)}, B = 3×3: dilate(X,B) = {|x−y| ≤ 2}; erode(that,B) = {|x−y| ≤ 0} = X exactly. The
// close returns the shell UNCHANGED, the flood still leaks, and solid/wall stays 1.00. (Verified
// in-grid — see the R sweep in the handoff.) The close only helps against genuine holes.
//
// WHAT ACTUALLY WORKS = DILATE → FILL → ERODE (note the fill is INSIDE the pair):
//   dilate by R : the shell becomes ≥(2R+1) thick in Chebyshev ⇒ 6-tunnel-free ⇒ the flood CANNOT
//                 leak, whatever the staircase or the M4 extractor's holes do.
//   flood+invert: the interior cavity is filled with BULK material.
//   erode by R  : peels the R layers back off the OUTSIDE, restoring the silhouette — but the bulk
//                 interior is not a thin shell, so erosion does not re-open it. The fill STICKS.
// Net: outer surface back where it was, interior solid. is_solid then answers about a BODY, not a
// shell, so MC emits ONE skin.
//   * Silhouette safety: erode∘dilate = the morphological CLOSING of the seed (⊒ seed, and it can
//     never exceed the seed's bbox: a point R+1 outside would need its whole SE-box inside
//     dilate(X), which only reaches R). So the outside moves by ≤0 cells — verified empirically
//     (REMESH_SEAL_VERBOSE prints the 3 silhouette projections before/after; delta must be 0).
//   * The real cost of R is FEATURE FUSION, not silhouette: a closing bridges gaps ≤ ~2R coarse
//     cells (= 2·R·stride fine voxels). That is the twintail/finger risk — keep R as small as
//     seals, hence the sweep. R is a knob: REMESH_CLOSE_R=0 restores the exact legacy (leaky) path.
// =============================================================================================

// separable (2r+1)³ BOX morphology on a dense binary volume; index = (x*ny + y)*nz + z (z
// contiguous). A box max/min is SEPARABLE per axis, so an r=3 close is 6 linear passes (~7
// taps/cell/axis) rather than 343 taps/cell, and needs ONE temp buffer instead of a 7³ gather.
// Out-of-volume counts as EMPTY for both ops; callers pad by ≥ r+2 so real content is never
// clipped. dilate=true → OR (max), false → AND (min).
inline void box_morph3(std::vector<uint8_t>& v, int nx, int ny, int nz, int r, bool dilate) {
    if (r <= 0) return;
    std::vector<uint8_t> t(v.size());
    const size_t sy = (size_t)nz, sx = (size_t)ny * (size_t)nz;
    const uint8_t init = dilate ? 0 : 1;
    // ---- Z pass (the varying axis is the contiguous one)
    #pragma omp parallel for schedule(static)
    for (int x = 0; x < nx; x++)
        for (int y = 0; y < ny; y++) {
            const uint8_t* s = &v[(size_t)x*sx + (size_t)y*sy];
            uint8_t* d = &t[(size_t)x*sx + (size_t)y*sy];
            for (int z = 0; z < nz; z++) {
                int lo = z-r, hi = z+r; uint8_t a;
                if (dilate) { a = 0; if (lo<0) lo=0; if (hi>=nz) hi=nz-1;
                              for (int k=lo;k<=hi;k++) a = (uint8_t)(a | s[k]); }
                else if (lo<0 || hi>=nz) { a = 0; }
                else { a = 1; for (int k=lo;k<=hi;k++) a = (uint8_t)(a & s[k]); }
                d[z] = a;
            }
        }
    v.swap(t);
    // ---- Y pass (accumulate whole z-rows so the inner loop stays contiguous)
    #pragma omp parallel for schedule(static)
    for (int x = 0; x < nx; x++)
        for (int y = 0; y < ny; y++) {
            uint8_t* d = &t[(size_t)x*sx + (size_t)y*sy];
            int lo = y-r, hi = y+r;
            if (!dilate && (lo<0 || hi>=ny)) { memset(d, 0, (size_t)nz); continue; }
            if (lo<0) lo=0;
            if (hi>=ny) hi=ny-1;
            memset(d, init, (size_t)nz);
            for (int k=lo;k<=hi;k++) { const uint8_t* s=&v[(size_t)x*sx + (size_t)k*sy];
                if (dilate) for (int z=0;z<nz;z++) d[z] = (uint8_t)(d[z] | s[z]);
                else        for (int z=0;z<nz;z++) d[z] = (uint8_t)(d[z] & s[z]); }
        }
    v.swap(t);
    // ---- X pass
    #pragma omp parallel for schedule(static)
    for (int x = 0; x < nx; x++)
        for (int y = 0; y < ny; y++) {
            uint8_t* d = &t[(size_t)x*sx + (size_t)y*sy];
            int lo = x-r, hi = x+r;
            if (!dilate && (lo<0 || hi>=nx)) { memset(d, 0, (size_t)nz); continue; }
            if (lo<0) lo=0;
            if (hi>=nx) hi=nx-1;
            memset(d, init, (size_t)nz);
            for (int k=lo;k<=hi;k++) { const uint8_t* s=&v[(size_t)k*sx + (size_t)y*sy];
                if (dilate) for (int z=0;z<nz;z++) d[z] = (uint8_t)(d[z] | s[z]);
                else        for (int z=0;z<nz;z++) d[z] = (uint8_t)(d[z] & s[z]); }
        }
    v.swap(t);
}

// Default SEAL radius in COARSE cells. *** DEFAULT 0 = OFF, AND THAT IS THE CORRECT DEFAULT. ***
// MEASURED IN-GRID (soldier @res1536/stride2, the real 768³ production grid, REMESH_CLOSE_R=0):
//     seed(wall) = 999,290 cells   solid = 6,240,249 cells   solid/wall = 6.245
//     emitted surface signed volume = 0.013778  vs  solid*cellvol = 0.013776   (match to 0.01%)
//                                              vs  seed*cellvol   = 0.002206   (6.2x off)
// i.e. the flood fill DOES NOT LEAK, `solid` IS a filled body, and the surface this function emits
// bounds THAT BODY — a SINGLE skin. The "the fill is a no-op / solid == 1.00x wall / the mesh is
// double-walled because of this function" theory is REFUTED at the real resolution. (Control: this
// path reproduces production's 4,831,250 verts exactly, so it is the real legacy path.)
// The seal below is therefore INSURANCE, not a fix — it costs feature fusion (it bridges gaps of
// ≤ ~2*R coarse cells; at stride 2 R=1 already fuses ~4 fine voxels, which is the twintail/finger
// risk the mc_blur comment warns about) and buys nothing on inputs that already fill correctly.
// Turn it on ONLY for an input the QC warning below actually fires on.
#ifndef REMESH_CLOSE_R_DEFAULT
#define REMESH_CLOSE_R_DEFAULT 0
#endif

inline Mesh marching_cubes_solid(const int32_t* coords, int N, int fine_grid, int stride,
                                 int blur = 1, float iso = 0.5f) {
    const int Gc = fine_grid / stride;
    const float voxel_c = (float)stride / (float)fine_grid, aabb0 = -0.5f;

    // SEAL knobs. REMESH_CLOSE_R=0 reproduces the legacy (double-walled) mesh bit-for-bit for A/B.
    const char* cr_env  = std::getenv("REMESH_CLOSE_R");
    const int   close_r = cr_env ? atoi(cr_env) : REMESH_CLOSE_R_DEFAULT;
    const bool  seal_strict  = std::getenv("REMESH_SEAL_STRICT")  != nullptr;
    const bool  seal_verbose = std::getenv("REMESH_SEAL_VERBOSE") != nullptr;

    // 1. coarse occupied seed + bbox (in coarse coords). Two linear passes over `coords` — no hash
    //    map: the dense bbox grid IS the membership structure (and it is what the seal needs).
    int mnx=Gc,mny=Gc,mnz=Gc,mxx=-1,mxy=-1,mxz=-1;
    for (int i=0;i<N;i++){ int X=coords[i*4+1]/stride, Y=coords[i*4+2]/stride, Z=coords[i*4+3]/stride;
        mnx=X<mnx?X:mnx; mny=Y<mny?Y:mny; mnz=Z<mnz?Z:mnz;
        mxx=X>mxx?X:mxx; mxy=Y>mxy?Y:mxy; mxz=Z>mxz?Z:mxz; }
    // pad must clear the dilation: with pad = 2+R the corner seed stays empty after dilate by R.
    const int pad = 2 + (close_r > 0 ? close_r : 0);
    int bx0=mnx-pad, by0=mny-pad, bz0=mnz-pad;
    int nbx=(mxx-mnx)+1+2*pad, nby=(mxy-mny)+1+2*pad, nbz=(mxz-mnz)+1+2*pad;
    const size_t ncell = (size_t)nbx*(size_t)nby*(size_t)nbz;
    auto bidx=[&](int X,int Y,int Z)->int64_t{ return ((int64_t)(X-bx0)*nby + (Y-by0))*nbz + (Z-bz0); };
    std::vector<uint8_t> occ(ncell, 0);
    for (int i=0;i<N;i++) occ[bidx(coords[i*4+1]/stride, coords[i*4+2]/stride, coords[i*4+3]/stride)] = 1;
    int64_t seed_cells=0; for (size_t i=0;i<ncell;i++) seed_cells += occ[i];

    // 2. SEAL step A — DILATE the shell so it is 6-tunnel-free (skipped entirely when close_r==0).
    std::vector<uint8_t> work = occ;
    if (close_r > 0) box_morph3(work, nbx, nby, nbz, close_r, /*dilate=*/true);

    // 3. flood-fill exterior (6-conn) from the (bx0,by0,bz0) corner == index 0 (empty: pad ≥ R+2).
    //    state: 0=unknown(interior candidate), 1=shell, 2=exterior. int64 ids — a coarse bbox can
    //    exceed 2^31 cells at small stride, which the old `int` stack would have silently wrapped.
    std::vector<uint8_t> st(ncell, 0);
    for (size_t i=0;i<ncell;i++) if (work[i]) st[i]=1;
    {
        std::vector<int64_t> stack; stack.reserve(1u<<16);
        auto push=[&](int64_t id){ if(st[id]==0){ st[id]=2; stack.push_back(id);} };
        push(0);
        const int64_t sy=nbz, sx=(int64_t)nby*nbz;
        while(!stack.empty()){ int64_t id=stack.back(); stack.pop_back();
            int64_t z=id%nbz, y=(id/nbz)%nby, x=id/sx;
            if(z>0)      push(id-1);
            if(z<nbz-1)  push(id+1);
            if(y>0)      push(id-sy);
            if(y<nby-1)  push(id+sy);
            if(x>0)      push(id-sx);
            if(x<nbx-1)  push(id+sx);
        }
    }
    std::vector<uint8_t>().swap(work);   // free before the erode allocates its temp

    // 4. solid = NOT exterior ... then SEAL step B — ERODE by the same R (restores the silhouette;
    //    the now-BULK interior does not re-open). Union the seed back in so the seal is extensive:
    //    a genuinely sub-R-thin feature can never be erased, only failed-to-fill.
    std::vector<uint8_t> solid(ncell);
    for (size_t i=0;i<ncell;i++) solid[i] = (st[i]!=2) ? 1 : 0;
    std::vector<uint8_t>().swap(st);
    if (close_r > 0) {
        box_morph3(solid, nbx, nby, nbz, close_r, /*dilate=*/false);
        for (size_t i=0;i<ncell;i++) if (occ[i]) solid[i]=1;
    }
    int64_t solid_cells=0; for (size_t i=0;i<ncell;i++) solid_cells += solid[i];
    const double ratio = seed_cells ? (double)solid_cells/(double)seed_cells : 0.0;

    // ---- QC: THE check that was missing. boundary==0, nonmanifold==0 and single-component ALL
    // PASS PERFECTLY on a leaked double-walled shell — every acceptance test we had was blind to
    // it, which is exactly how it shipped. solid/wall is the one number that sees it: a filled
    // body is many times its own shell; a leak pins the ratio at ~1.0.
    if (ratio < 1.2) {
        fprintf(stderr,
            "[remesh] *** QC: solid/wall = %.3f (solid=%lld seed=%lld, close_r=%d, grid=%d) ***\n"
            "[remesh]     The exterior flood-fill LEAKED through the occupancy shell, so `solid`\n"
            "[remesh]     collapsed onto the shell and MC will emit a DOUBLE-WALLED mesh (outer\n"
            "[remesh]     skin + a never-visible inner skin) — ~37%% wasted polys/atlas and ~73%%\n"
            "[remesh]     of the texture holes. Raise REMESH_CLOSE_R (0 = leaky legacy path).\n"
            "[remesh]     NB boundary/nonmanifold/component checks CANNOT see this. Expected for a\n"
            "[remesh]     genuinely sheet-like (not closed) input; otherwise it is the bug.\n"
            "[remesh]     (REMESH_SEAL_STRICT=1 turns this warning into an abort.)\n",
            ratio, (long long)solid_cells, (long long)seed_cells, close_r, Gc);
        if (seal_strict) { fprintf(stderr, "[remesh] REMESH_SEAL_STRICT: aborting.\n"); abort(); }
    }
    if (seal_verbose) {
        // SILHOUETTE SAFETY: project seed and solid down each axis. The seal is a closing on the
        // outside, so the projections must be IDENTICAL — any growth here means the fix moved the
        // outer surface and is wrong.
        int64_t ps[3]={0,0,0}, pl[3]={0,0,0};
        auto proj=[&](int axis, const std::vector<uint8_t>& v)->int64_t{
            int d0 = axis==0?nby:nbx, d1 = axis==2?nby:nbz;
            std::vector<uint8_t> msk((size_t)d0*d1, 0);
            for (int x=0;x<nbx;x++) for (int y=0;y<nby;y++) for (int z=0;z<nbz;z++)
                if (v[((size_t)x*nby+y)*nbz+z]) {
                    int u = axis==0?y:x, w = axis==2?y:z;
                    msk[(size_t)u*d1+w] = 1; }
            int64_t c=0; for (size_t i=0;i<msk.size();i++) c+=msk[i]; return c; };
        for (int ax=0; ax<3; ax++){ ps[ax]=proj(ax,occ); pl[ax]=proj(ax,solid); }
        fprintf(stderr, "[remesh] SEAL close_r=%d grid=%d bbox=%dx%dx%d (%.1fM cells)\n"
                        "[remesh]   seed(wall)=%lld solid=%lld  solid/wall=%.3f\n"
                        "[remesh]   silhouette proj (seed -> solid): X %lld -> %lld | Y %lld -> %lld | Z %lld -> %lld  (delta must be 0)\n",
                close_r, Gc, nbx, nby, nbz, (double)ncell/1e6,
                (long long)seed_cells, (long long)solid_cells, ratio,
                (long long)ps[0],(long long)pl[0],(long long)ps[1],(long long)pl[1],(long long)ps[2],(long long)pl[2]);
    }

    // 5. solid predicate: in-bbox and filled.
    auto is_solid=[&](int X,int Y,int Z)->bool{
        if(X<bx0||Y<by0||Z<bz0||X>=bx0+nbx||Y>=by0+nby||Z>=bz0+nbz) return false;
        return solid[bidx(X,Y,Z)] != 0; };
    // 4. smoothed field = box-blur of binary solid
    const int r=blur; const float inv_box=1.0f/(float)((2*r+1)*(2*r+1)*(2*r+1));
    auto field_at=[&](int X,int Y,int Z)->float{
        int c=0; for(int dz=-r;dz<=r;dz++)for(int dy=-r;dy<=r;dy++)for(int dx=-r;dx<=r;dx++) if(is_solid(X+dx,Y+dy,Z+dz)) c++;
        return (float)c*inv_box; };

    // candidate cells = cubes within the bbox (cheap; bbox is small at coarse res)
    Mesh m; m.N=0; m.F=0;
    std::unordered_map<int64_t,int> vmap; vmap.reserve((size_t)nbx*nby);
    auto gp_world=[&](int c)->float{ return ((float)c+0.5f)*voxel_c+aabb0; };
    auto edge_vert=[&](int ax,int ay,int az,int bx,int by,int bz)->int{
        int lx=ax<bx?ax:bx,ly=ay<by?ay:by,lz=az<bz?az:bz;
        int dm=(ax!=bx?1:0)|(ay!=by?2:0)|(az!=bz?4:0);
        int64_t key=((((int64_t)(lx+2)*4096+(ly+2))*4096+(lz+2))*8)+dm;
        auto it=vmap.find(key); if(it!=vmap.end())return it->second;
        float fa=field_at(ax,ay,az),fb=field_at(bx,by,bz),den=fb-fa;
        float t=(fabsf(den)<1e-6f)?0.5f:(iso-fa)/den; if(t<0.01f)t=0.01f; else if(t>0.99f)t=0.99f;
        int idx=(int)(m.verts.size()/3);
        m.verts.push_back(gp_world(ax)+t*(gp_world(bx)-gp_world(ax)));
        m.verts.push_back(gp_world(ay)+t*(gp_world(by)-gp_world(ay)));
        m.verts.push_back(gp_world(az)+t*(gp_world(bz)-gp_world(az)));
        vmap.emplace(key,idx); return idx; };
    auto cxyz=[](int c,int&x,int&y,int&z){x=c&1;y=(c>>1)&1;z=(c>>2)&1;};
    for(int X=bx0;X<bx0+nbx-1;X++)for(int Y=by0;Y<by0+nby-1;Y++)for(int Z=bz0;Z<bz0+nbz-1;Z++){
        int cgx[8],cgy[8],cgz[8]; bool cin[8]; int nin=0;
        for(int c=0;c<8;c++){int ox,oy,oz;cxyz(c,ox,oy,oz);cgx[c]=X+ox;cgy[c]=Y+oy;cgz[c]=Z+oz;
            cin[c]=(field_at(cgx[c],cgy[c],cgz[c])>=iso); if(cin[c])nin++;}
        if(nin==0||nin==8)continue;
        for(int t=0;t<6;t++){const int*T=MT_TETS[t];int in[4],ni=0;
            for(int j=0;j<4;j++){in[j]=cin[T[j]]?1:0;ni+=in[j];} if(ni==0||ni==4)continue;
            float od[3]={0,0,0};for(int j=0;j<4;j++){int c=T[j];float s=in[j]?-1.f:1.f;od[0]+=s*cgx[c];od[1]+=s*cgy[c];od[2]+=s*cgz[c];}
            auto emit_tri=[&](int v0,int v1,int v2){
                const float*p0=&m.verts[(size_t)v0*3],*p1=&m.verts[(size_t)v1*3],*p2=&m.verts[(size_t)v2*3];
                float e1[3]={p1[0]-p0[0],p1[1]-p0[1],p1[2]-p0[2]},e2[3]={p2[0]-p0[0],p2[1]-p0[1],p2[2]-p0[2]};
                float nx=e1[1]*e2[2]-e1[2]*e2[1],ny=e1[2]*e2[0]-e1[0]*e2[2],nz=e1[0]*e2[1]-e1[1]*e2[0];
                if(nx*od[0]+ny*od[1]+nz*od[2]<0.f)std::swap(v1,v2);
                m.faces.push_back(v0);m.faces.push_back(v1);m.faces.push_back(v2);};
            auto ev=[&](int ca,int cb){return edge_vert(cgx[ca],cgy[ca],cgz[ca],cgx[cb],cgy[cb],cgz[cb]);};
            if(ni==1||ni==3){int s=(ni==1)?1:0;int a=-1;for(int j=0;j<4;j++)if(in[j]==s){a=j;break;}
                int b[3],bi=0;for(int j=0;j<4;j++)if(j!=a)b[bi++]=j;
                emit_tri(ev(T[a],T[b[0]]),ev(T[a],T[b[1]]),ev(T[a],T[b[2]]));}
            else{int insd[2],outd[2],ii=0,oo=0;for(int j=0;j<4;j++){if(in[j])insd[ii++]=j;else outd[oo++]=j;}
                int e00=ev(T[insd[0]],T[outd[0]]),e01=ev(T[insd[0]],T[outd[1]]),e10=ev(T[insd[1]],T[outd[0]]),e11=ev(T[insd[1]],T[outd[1]]);
                emit_tri(e00,e01,e11);emit_tri(e00,e11,e10);}
        }
    }
    m.N=(int)(m.verts.size()/3); m.F=(int)(m.faces.size()/3);
    (void)Gc;
    orient_consistent(m);
    return m;
}

// Taubin (λ|μ) smoothing of a manifold mesh — moves vertices only (topology/watertightness/
// manifoldness preserved). The marching-tet surface is a diagonally-faceted "staircase" at the
// voxel lattice; its per-triangle normals are noisy, which makes xatlas seed a chart per few faces
// (defeating the atlas). Taubin's alternating positive(λ)/negative(μ) Laplacian passes low-pass the
// surface WITHOUT the shrinkage of plain Laplacian smoothing -> coherent normals -> charts collapse.
// At grid1024 the sub-mm smoothing is invisible (and actually closer to the organic original than
// the blocky voxels). Uses uniform (umbrella) weights over the 1-ring.
inline void taubin_smooth(Mesh& m, int iters, float lambda = 0.5f, float mu = -0.53f) {
    const int Vn = (int)(m.verts.size()/3);
    const int64_t Fn = (int64_t)(m.faces.size()/3);
    // build 1-ring adjacency from the (manifold) faces.
    std::vector<std::vector<int>> adj(Vn);
    for (int64_t t=0;t<Fn;t++){ int v[3]={(int)m.faces[t*3],(int)m.faces[t*3+1],(int)m.faces[t*3+2]};
        for (int e=0;e<3;e++){ int a=v[e], b=v[(e+1)%3];
            adj[a].push_back(b); adj[b].push_back(a); } }
    // dedup neighbours (a manifold edge appears from its 2 faces -> each neighbour listed twice)
    for (int i=0;i<Vn;i++){ auto& nb=adj[i]; std::sort(nb.begin(),nb.end()); nb.erase(std::unique(nb.begin(),nb.end()),nb.end()); }

    std::vector<float> tmp(m.verts.size());
    auto pass = [&](float w){
        #pragma omp parallel for schedule(static)
        for (int i=0;i<Vn;i++){
            const auto& nb=adj[i];
            if (nb.empty()){ for(int d=0;d<3;d++) tmp[(size_t)i*3+d]=m.verts[(size_t)i*3+d]; continue; }
            float c[3]={0,0,0};
            for (int j:nb) for(int d=0;d<3;d++) c[d]+=m.verts[(size_t)j*3+d];
            float inv=1.f/(float)nb.size();
            for(int d=0;d<3;d++){ float lap=c[d]*inv - m.verts[(size_t)i*3+d];
                tmp[(size_t)i*3+d]=m.verts[(size_t)i*3+d] + w*lap; }
        }
        m.verts.swap(tmp);
    };
    for (int it=0; it<iters; it++){ pass(lambda); pass(mu); }
}

// make_manifold — convert a non-manifold triangle soup (the O-Voxel dual-grid mesh has ~162k
// non-manifold edges where >2 quads meet at thin sheets) into an edge+vertex manifold mesh by
// VERTEX-UMBRELLA SPLITTING. For each vertex, its incident faces are grouped into "umbrellas"
// (fans) connected through MANIFOLD edges (edges with exactly 2 faces); each umbrella becomes its
// own copy of the vertex. After the split every edge has <=2 incident faces and every vertex a
// single fan → meshopt QUALITY (QEM) decimation is no longer locked by the non-manifold skeleton,
// so it flows all the way to the target instead of stalling (~1.08M) and falling to sloppy. Former
// non-manifold edges + the dual-grid frontier become boundaries (decimate with SimplifyLockBorder
// to keep them, or close with fill_holes). Pure host; adds vertices only where topology demands.
// (This is the principled "feed QEM a sharp MANIFOLD mesh" fix — keeps the dual-grid's QEF vertex
// positions, i.e. the sharp heel spikes / fingers, while making it decimatable.)
inline int64_t make_manifold(Mesh& m) {
    int64_t F = (int64_t)m.faces.size()/3;
    // 0. drop degenerate faces (a repeated index → not a real triangle, pollutes edge counts)
    {
        std::vector<int64_t> ff; ff.reserve(m.faces.size());
        for (int64_t t=0;t<F;t++){ int64_t a=m.faces[t*3],b=m.faces[t*3+1],c=m.faces[t*3+2];
            if(a!=b&&b!=c&&a!=c){ ff.push_back(a);ff.push_back(b);ff.push_back(c);} }
        m.faces.swap(ff); F=(int64_t)m.faces.size()/3;
    }
    const int64_t V = (int64_t)m.verts.size()/3;
    // global edge incidence: how many faces share each undirected edge (==2 manifold, >2 non-manifold)
    auto ukey=[](int64_t a,int64_t b){ int64_t lo=a<b?a:b,hi=a<b?b:a; return (lo<<32)|(uint32_t)hi; };
    std::unordered_map<int64_t,int> ec; ec.reserve((size_t)F*3);
    for (int64_t t=0;t<F;t++){ int64_t v[3]={m.faces[t*3],m.faces[t*3+1],m.faces[t*3+2]};
        for(int e=0;e<3;e++) ec[ukey(v[e],v[(e+1)%3])]++; }
    // per-vertex incident faces (faceid)
    std::vector<std::vector<int>> vf(V);
    for (int64_t t=0;t<F;t++) for(int c=0;c<3;c++) vf[m.faces[t*3+c]].push_back((int)t);
    // PER-VERTEX umbrella split: for each vertex v, locally union its incident faces that share an
    // edge (v,x) which is MANIFOLD (exactly 2 faces). Faces meeting only at a non-manifold edge of v
    // land in different umbrellas → get distinct vertex copies → that edge ends up with <=2 faces.
    // (The earlier bug was a GLOBAL face union-find: faces could be merged through a manifold edge NOT
    // incident to v, re-sharing v's non-manifold edge. Local grouping fixes that.)
    std::vector<float> nv = m.verts;                   // appended copies below
    std::vector<int64_t> nf(m.faces.begin(), m.faces.end());
    // for each (faceid, corner) we record the new vertex id; default = original (group 0 keeps v)
    for (int64_t v=0; v<V; v++){
        auto& inc = vf[v]; const int n=(int)inc.size(); if(n==0) continue;
        // local union-find over inc[0..n-1]
        std::vector<int> uf(n); for(int i=0;i<n;i++) uf[i]=i;
        std::function<int(int)> find=[&](int x){ while(uf[x]!=x){ uf[x]=uf[uf[x]]; x=uf[x]; } return x; };
        // neighbor x -> local face indices that have edge (v,x); union pairs on a MANIFOLD edge
        std::unordered_map<int64_t,int> firstFace;     // x -> local idx of first face seen (only if edge manifold)
        for (int i=0;i<n;i++){ int t=inc[i]; int64_t a=m.faces[t*3],b=m.faces[t*3+1],c=m.faces[t*3+2];
            int64_t other[2]; int oc=0;
            if(a==v){other[0]=b;other[1]=c;} else if(b==v){other[0]=a;other[1]=c;} else {other[0]=a;other[1]=b;}
            for(oc=0;oc<2;oc++){ int64_t x=other[oc]; if(ec[ukey(v,x)]!=2) continue;   // only manifold edges merge
                auto it=firstFace.find(x);
                if(it==firstFace.end()) firstFace[x]=i;
                else { int a1=find(i),b1=find(it->second); if(a1!=b1) uf[a1]=b1; } }
        }
        // assign a new vertex per local group; group of the FIRST face keeps original index v
        std::unordered_map<int,int> groupVid; int firstRoot=find(0); groupVid[firstRoot]=(int)v;
        for (int i=0;i<n;i++){ int r=find(i); int vid;
            auto it=groupVid.find(r);
            if(it==groupVid.end()){ vid=(int)(nv.size()/3);
                nv.push_back(m.verts[v*3]); nv.push_back(m.verts[v*3+1]); nv.push_back(m.verts[v*3+2]);
                groupVid[r]=vid; } else vid=it->second;
            // set this corner of face inc[i]
            int t=inc[i]; for(int c=0;c<3;c++) if(m.faces[t*3+c]==v) nf[t*3+c]=vid;
        }
    }
    int64_t added=(int64_t)(nv.size()/3) - V;
    m.verts.swap(nv); m.faces.swap(nf); m.N=(int)(m.verts.size()/3); m.F=(int)(m.faces.size()/3);
    return added;
}

// Make triangle winding globally CONSISTENT by BFS over face adjacency, then flip globally so the
// outward normal points out (signed volume > 0). The per-tet `od` sign test in the MC emit is noisy
// on skewed interpolated triangles → ~10% of faces wind backwards → adjacent-face normals flip →
// xatlas treats every flip edge as a hard seam → a chart per facet (18k charts). A watertight
// 2-manifold mesh has exactly 2 faces per undirected edge, so consistent orientation is unique up to
// a global flip: pick a seed, propagate (neighbour must traverse the shared edge in the OPPOSITE
// direction), flipping any face that disagrees. This is the lever that collapses the chart count.
inline void orient_consistent(Mesh& m) {
    const int64_t F = (int64_t)m.faces.size()/3;
    if (F == 0) return;
    // undirected edge -> up to 2 (face, which-edge). manifold => exactly 2.
    std::unordered_map<int64_t, std::pair<int,int>> e2f;   // key -> (face0<<2|e0)  ... store first
    std::unordered_map<int64_t, std::vector<int>> emap;    // key -> face indices (size 2)
    emap.reserve((size_t)F*3);
    auto ukey=[](int64_t a,int64_t b){ int64_t lo=a<b?a:b,hi=a<b?b:a; return (lo<<32)|(uint32_t)hi; };
    for (int64_t t=0;t<F;t++){ int64_t v[3]={m.faces[t*3],m.faces[t*3+1],m.faces[t*3+2]};
        for (int e=0;e<3;e++) emap[ukey(v[e],v[(e+1)%3])].push_back((int)t); }
    (void)e2f;
    std::vector<char> visited(F,0);
    std::vector<int> stack;
    // helper: does face t currently contain the DIRECTED edge (a->b)?
    auto has_dir=[&](int t,int64_t a,int64_t b)->bool{
        int64_t v[3]={m.faces[(int64_t)t*3],m.faces[(int64_t)t*3+1],m.faces[(int64_t)t*3+2]};
        for(int e=0;e<3;e++) if(v[e]==a && v[(e+1)%3]==b) return true;
        return false; };
    auto flip=[&](int t){ std::swap(m.faces[(int64_t)t*3+1], m.faces[(int64_t)t*3+2]); };
    for (int64_t s=0;s<F;s++){
        if (visited[s]) continue;
        visited[s]=1; stack.push_back((int)s);
        while(!stack.empty()){ int t=stack.back(); stack.pop_back();
            int64_t v[3]={m.faces[(int64_t)t*3],m.faces[(int64_t)t*3+1],m.faces[(int64_t)t*3+2]};
            for(int e=0;e<3;e++){ int64_t a=v[e], b=v[(e+1)%3];
                auto& fs = emap[ukey(a,b)];
                for(int nf: fs){ if(nf==t||visited[nf]) continue;
                    // consistent orientation: neighbour must contain (b->a). If it has (a->b), flip it.
                    if (has_dir(nf,a,b)) flip(nf);
                    visited[nf]=1; stack.push_back(nf);
                }
            }
        }
    }
    // global flip so outward normals point out (signed volume > 0 about the centroid).
    double cx=0,cy=0,cz=0; const int64_t Vn=(int64_t)m.verts.size()/3;
    for(int64_t i=0;i<Vn;i++){ cx+=m.verts[i*3]; cy+=m.verts[i*3+1]; cz+=m.verts[i*3+2]; }
    cx/=Vn; cy/=Vn; cz/=Vn;
    double vol=0;
    for(int64_t t=0;t<F;t++){ int64_t i0=m.faces[t*3],i1=m.faces[t*3+1],i2=m.faces[t*3+2];
        double a[3]={m.verts[i0*3]-cx,m.verts[i0*3+1]-cy,m.verts[i0*3+2]-cz};
        double b[3]={m.verts[i1*3]-cx,m.verts[i1*3+1]-cy,m.verts[i1*3+2]-cz};
        double c[3]={m.verts[i2*3]-cx,m.verts[i2*3+1]-cy,m.verts[i2*3+2]-cz};
        double cr[3]={b[1]*c[2]-b[2]*c[1],b[2]*c[0]-b[0]*c[2],b[0]*c[1]-b[1]*c[0]};
        vol += a[0]*cr[0]+a[1]*cr[1]+a[2]*cr[2]; }
    if (vol < 0) for(int64_t t=0;t<F;t++) flip((int)t);
}

// count edges whose incident-face count != 2 (boundary or non-manifold). manifold+watertight <=> 0.
inline void mesh_topology_stats(const Mesh& m, int64_t& boundary, int64_t& nonmanifold) {
    const int64_t F = (int64_t)m.faces.size()/3;
    std::unordered_map<int64_t,int> ec; ec.reserve((size_t)F*3*2);
    auto ukey=[](int64_t a,int64_t b){ int64_t lo=a<b?a:b,hi=a<b?b:a; return (lo<<32)|(uint32_t)hi; };
    for (int64_t t=0;t<F;t++){ int64_t v[3]={m.faces[t*3],m.faces[t*3+1],m.faces[t*3+2]};
        for (int e=0;e<3;e++) ec[ukey(v[e],v[(e+1)%3])]++; }
    boundary=0; nonmanifold=0;
    for (auto& kv:ec){ if(kv.second==1) boundary++; else if(kv.second>2) nonmanifold++; }
}

} // namespace svae
