// Validate the native marching-cubes surface extractor (ultrashape_mc.hpp) against the cubvh golden
// (capture_mc.py: get_sparse_valid_voxels -> cubvh.sparse_marching_cubes(mc_level=0) -> bbox scale).
// The cubvh GPU kernel emits verts/faces in nondeterministic order, so we validate GEOMETRICALLY as
// SETS: (1) vertex point-set equality, (2) face-set equality (each face = canonical sorted triple of
// its 3 vertex POSITIONS). Also writes the mesh to GLB.   ./build.sh ultrashape_mc_test
#include "ultrashape_mc.hpp"
#include "glb_writer.hpp"
#include "../../sparse_spike/npy.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <set>
#include <map>
#include <array>
#include <cmath>

static const char* GDIR = "/mnt/hdd/3d/avatar-shootout/volume_goldens";
static const char* MDIR = "/mnt/hdd/3d/avatar-shootout/mc_goldens";

// quantize a position to an integer key (1e-4 grid) for order-independent set comparison
static inline std::array<int64_t,3> qkey(float x, float y, float z) {
    auto q=[&](float v){ return (int64_t)std::llround((double)v * 10000.0); };
    return {q(x), q(y), q(z)};
}
typedef std::array<int64_t,3> VKey;
// canonical face key = the 3 vertex position-keys, sorted (winding/order-independent)
typedef std::array<VKey,3> FKey;
static FKey fkey(const VKey& a, const VKey& b, const VKey& c) {
    FKey f{a,b,c}; std::sort(f.begin(), f.end()); return f;
}

int main(int argc, char** argv) {
    const int OCT = 64; const float BNDS = 1.01f; const float MC_LEVEL = 0.0f;

    NpyArray gl = npy_load(std::string(GDIR) + "/grid_logits.npy");   // [1,G,G,G]
    int G = (int)gl.shape[gl.shape.size()-1];
    printf("[us_mc] grid_logits %s G=%d\n",
           [&]{ static char b[64]; snprintf(b,64,"[%lld,%lld,%lld,%lld]",
                (long long)gl.shape[0],(long long)(gl.shape.size()>1?gl.shape[1]:1),
                (long long)(gl.shape.size()>2?gl.shape[2]:1),(long long)(gl.shape.size()>3?gl.shape[3]:1)); return b; }(), G);

    us_mc::Mesh m = us_mc::marching_cubes(gl.f32(), G, MC_LEVEL);
    int V = (int)(m.verts.size()/3), F = (int)(m.faces.size()/3);
    printf("[us_mc] mine (voxel space): verts=%d faces=%d\n", V, F);

    // ---- golden (raw, voxel space) ----
    NpyArray gv = npy_load(std::string(MDIR) + "/verts_raw.npy");     // [Vg,3] f32
    NpyArray gf = npy_load(std::string(MDIR) + "/faces.npy");         // [Fg,3] i64
    int Vg = (int)gv.shape[0], Fg = (int)gf.shape[0];
    const float* gvp = gv.f32(); const int64_t* gfp = gf.i64();
    printf("[us_mc] golden: verts=%d faces=%d\n", Vg, Fg);

    // ---- vertex set comparison ----
    std::multiset<VKey> mine_vk, gold_vk;
    for (int i=0;i<V;i++)  mine_vk.insert(qkey(m.verts[i*3],m.verts[i*3+1],m.verts[i*3+2]));
    for (int i=0;i<Vg;i++) gold_vk.insert(qkey(gvp[i*3],gvp[i*3+1],gvp[i*3+2]));
    int vmiss=0; for (auto&k:gold_vk) if (!mine_vk.count(k)) vmiss++;
    int vextra=0; for (auto&k:mine_vk) if (!gold_vk.count(k)) vextra++;
    bool verts_ok = (V==Vg) && vmiss==0 && vextra==0;
    printf("[us_mc] vertex-set: count %s (%d vs %d), golden-not-in-mine=%d  mine-not-in-golden=%d -> %s\n",
           V==Vg?"OK":"DIFF", V, Vg, vmiss, vextra, verts_ok?"MATCH":"MISMATCH");

    // ---- face set comparison (canonical sorted position triples) ----
    std::multiset<FKey> mine_fk, gold_fk;
    for (int i=0;i<F;i++) {
        int a=(int)m.faces[i*3], b=(int)m.faces[i*3+1], c=(int)m.faces[i*3+2];
        mine_fk.insert(fkey(qkey(m.verts[a*3],m.verts[a*3+1],m.verts[a*3+2]),
                            qkey(m.verts[b*3],m.verts[b*3+1],m.verts[b*3+2]),
                            qkey(m.verts[c*3],m.verts[c*3+1],m.verts[c*3+2])));
    }
    for (int i=0;i<Fg;i++) {
        int64_t a=gfp[i*3], b=gfp[i*3+1], c=gfp[i*3+2];
        gold_fk.insert(fkey(qkey(gvp[a*3],gvp[a*3+1],gvp[a*3+2]),
                            qkey(gvp[b*3],gvp[b*3+1],gvp[b*3+2]),
                            qkey(gvp[c*3],gvp[c*3+1],gvp[c*3+2])));
    }
    int fmiss=0; { std::multiset<FKey> tmp=mine_fk; for (auto&k:gold_fk){ auto it=tmp.find(k); if(it==tmp.end()) fmiss++; else tmp.erase(it);} }
    int fextra=0; { std::multiset<FKey> tmp=gold_fk; for (auto&k:mine_fk){ auto it=tmp.find(k); if(it==tmp.end()) fextra++; else tmp.erase(it);} }
    bool faces_ok = (F==Fg) && fmiss==0 && fextra==0;
    printf("[us_mc] face-set: count %s (%d vs %d), golden-not-in-mine=%d  mine-not-in-golden=%d -> %s\n",
           F==Fg?"OK":"DIFF", F, Fg, fmiss, fextra, faces_ok?"MATCH":"MISMATCH");

    // ---- scaling check vs verts_scaled ----
    std::vector<float> vs = m.verts;
    us_mc::scale_to_bbox(vs, G, BNDS);
    NpyArray gsc = npy_load(std::string(MDIR) + "/verts_scaled.npy"); // [Vg,3]
    const float* gsp = gsc.f32();
    std::multiset<VKey> mine_sk, gold_sk;
    for (int i=0;i<V;i++)  mine_sk.insert(qkey(vs[i*3],vs[i*3+1],vs[i*3+2]));
    for (int i=0;i<Vg;i++) gold_sk.insert(qkey(gsp[i*3],gsp[i*3+1],gsp[i*3+2]));
    int smiss=0; for (auto&k:gold_sk) if (!mine_sk.count(k)) smiss++;
    bool scale_ok = (V==Vg) && smiss==0;
    printf("[us_mc] scaled-vertex-set: golden-not-in-mine=%d -> %s\n", smiss, scale_ok?"MATCH":"MISMATCH");

    // ---- write GLB (scaled, web-ready) ----
    const char* out = "/mnt/hdd/3d/avatar-shootout/_shootout_out/us_mc_native.glb";
    if (glb::write_glb(out, vs, m.faces)) printf("[us_mc] wrote %s\n", out);
    else printf("[us_mc] GLB write FAILED\n");

    bool pass = verts_ok && faces_ok && scale_ok;
    printf("[us_mc] %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
