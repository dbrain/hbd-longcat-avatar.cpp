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
#include <cstdio>

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
//   2. flood-fill the EXTERIOR on the coarse grid (6-connectivity from the padded bbox shell).
//   3. solid = NOT exterior (fills the interior cavity the shell encloses) → one connected volume.
//   4. box-blur the binary solid → smooth field, MC at iso=0.5 + linear interp → a SMOOTH,
//      WATERTIGHT, SINGLE-COMPONENT, coherent-normal low-poly outer surface → tight atlas, no
//      decimation step. Seconds on CPU. (This is what cumesh's dual-contour remesh achieves; same
//      end product, open recipe.)
inline Mesh marching_cubes_solid(const int32_t* coords, int N, int fine_grid, int stride,
                                 int blur = 1, float iso = 0.5f) {
    const int Gc = fine_grid / stride;
    const float voxel_c = (float)stride / (float)fine_grid, aabb0 = -0.5f;

    // 1. coarse occupied seed + bbox (in coarse coords)
    int mnx=Gc,mny=Gc,mnz=Gc,mxx=-1,mxy=-1,mxz=-1;
    std::unordered_map<int64_t,char> seed; seed.reserve((size_t)N);
    for (int i=0;i<N;i++){ int X=coords[i*4+1]/stride, Y=coords[i*4+2]/stride, Z=coords[i*4+3]/stride;
        seed.emplace(coord_key(0,X,Y,Z),(char)1);
        mnx=X<mnx?X:mnx; mny=Y<mny?Y:mny; mnz=Z<mnz?Z:mnz;
        mxx=X>mxx?X:mxx; mxy=Y>mxy?Y:mxy; mxz=Z>mxz?Z:mxz; }
    const int pad=2;
    int bx0=mnx-pad, by0=mny-pad, bz0=mnz-pad;
    int nbx=(mxx-mnx)+1+2*pad, nby=(mxy-mny)+1+2*pad, nbz=(mxz-mnz)+1+2*pad;
    auto bidx=[&](int X,int Y,int Z)->int64_t{ return ((int64_t)(X-bx0)*nby + (Y-by0))*nbz + (Z-bz0); };
    // dense state: 0=unknown(interior candidate), 1=occupied-seed(solid), 2=exterior
    std::vector<uint8_t> st((size_t)nbx*nby*nbz, 0);
    for (auto& kv: seed){ int64_t k=kv.first; int Z=(int)(k&0xFFFFF),Y=(int)((k>>20)&0xFFFFF),X=(int)((k>>40)&0xFFFFF);
        st[bidx(X,Y,Z)] = 1; }
    // 2. flood-fill exterior from (bx0,by0,bz0) corner (guaranteed empty due to pad)
    std::vector<int> stack; stack.reserve(nbx*nby);
    auto push=[&](int x,int y,int z){ size_t id=(size_t)(x-bx0)*nby*nbz+(size_t)(y-by0)*nbz+(z-bz0);
        if(st[id]==0){ st[id]=2; stack.push_back((int)id);} };
    push(bx0,by0,bz0);
    const int dirs[6][3]={{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
    while(!stack.empty()){ int id=stack.back(); stack.pop_back();
        int z=id%nbz, y=(id/nbz)%nby, x=id/(nbz*nby);
        int X=x+bx0,Y=y+by0,Z=z+bz0;
        for(auto&dd:dirs){ int nX=X+dd[0],nY=Y+dd[1],nZ=Z+dd[2];
            if(nX<bx0||nY<by0||nZ<bz0||nX>=bx0+nbx||nY>=by0+nby||nZ>=bz0+nbz) continue;
            push(nX,nY,nZ); } }
    // 3. solid predicate: in-bbox and not exterior
    auto is_solid=[&](int X,int Y,int Z)->bool{
        if(X<bx0||Y<by0||Z<bz0||X>=bx0+nbx||Y>=by0+nby||Z>=bz0+nbz) return false;
        return st[bidx(X,Y,Z)] != 2; };
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
        for(int e=0;e<3;e++) if(v[e]==a && v[(e+1)%3]==b) return true; return false; };
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
