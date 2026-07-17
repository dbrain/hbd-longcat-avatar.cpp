// synth_face_ids — VALIDATION helper (not part of the port). Load a GLB, assign each face a synthetic
// part label by spatial partition (8-region grid: y-band x sign(x) x sign(z)), write face_ids.npy
// (int32) + print per-label face counts. Mimics P3-SAM body-part labels so per_part_decimate can be
// exercised end-to-end without P3-SAM. Also designed so the HAND tier can fire: a small far-X mid-Y
// region is carved out as its own label.
//   build: g++ -O2 -std=c++17 synth_face_ids.cpp -o synth_face_ids
//   run:   ./synth_face_ids <mesh.glb> <out_face_ids.npy>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <map>
#include <cmath>
#include "glb_reader.hpp"

static void write_npy_i32(const char* path, const std::vector<int32_t>& v) {
    FILE* f = std::fopen(path, "wb");
    char dict[128];
    int dl = std::snprintf(dict, sizeof dict, "{'descr': '<i4', 'fortran_order': False, 'shape': (%zu,), }", v.size());
    int total = 10 + dl;
    int pad = (64 - (total % 64)) % 64;          // pad header so data starts 64-aligned
    int hdr = dl + pad + 1;                       // +1 for trailing '\n'
    std::fwrite("\x93NUMPY", 1, 6, f);
    unsigned char ver[2] = {1, 0}; std::fwrite(ver, 1, 2, f);
    unsigned char hl[2] = {(unsigned char)(hdr & 0xff), (unsigned char)((hdr >> 8) & 0xff)};
    std::fwrite(hl, 1, 2, f);
    std::fwrite(dict, 1, dl, f);
    for (int i = 0; i < pad; i++) std::fputc(' ', f);
    std::fputc('\n', f);
    std::fwrite(v.data(), 4, v.size(), f);
    std::fclose(f);
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: synth_face_ids <mesh.glb> <out.npy>\n"); return 1; }
    glb::Mesh m;
    if (!glb::read_glb(argv[1], m)) { std::fprintf(stderr, "read failed\n"); return 1; }
    const int64_t V = (int64_t)m.verts.size() / 3, F = (int64_t)m.faces.size() / 3;

    float mn[3] = {1e30f,1e30f,1e30f}, mx[3] = {-1e30f,-1e30f,-1e30f};
    for (int64_t v = 0; v < V; v++) for (int d = 0; d < 3; d++) {
        float x = m.verts[v*3+d]; if (x<mn[d]) mn[d]=x; if (x>mx[d]) mx[d]=x; }
    double c[3] = {(mn[0]+mx[0])*0.5,(mn[1]+mx[1])*0.5,(mn[2]+mx[2])*0.5};
    double s = 0; for (int64_t v=0;v<V;v++) for (int d=0;d<3;d++) s=std::max(s,std::fabs((double)m.verts[v*3+d]-c[d]));
    if (s<=0) s=1;

    std::vector<int32_t> fid(F);
    std::map<int32_t,int64_t> cnt;
    for (int64_t f = 0; f < F; f++) {
        int64_t a=m.faces[f*3],b=m.faces[f*3+1],cc=m.faces[f*3+2];
        double fx=0,fy=0,fz=0;
        for (int64_t g : {a,b,cc}) { fx+=m.verts[g*3]; fy+=m.verts[g*3+1]; fz+=m.verts[g*3+2]; }
        fx=((fx/3)-c[0])/s; fy=((fy/3)-c[1])/s; fz=((fz/3)-c[2])/s;
        int32_t lbl;
        // carve a small far-X mid-Y "HAND" pocket so the HAND tier exercises
        if (std::fabs(fx) > 0.55 && fy > -0.30 && fy < 0.30) lbl = (fx>0) ? 100 : 101;
        else {
            int yb = (fy < -0.33) ? 0 : (fy < 0.33 ? 1 : 2);   // 3 y-bands
            int xs = (fx >= 0) ? 1 : 0;
            int zs = (fz >= 0) ? 1 : 0;
            lbl = yb*4 + xs*2 + zs;                            // 0..11
        }
        fid[f]=lbl; cnt[lbl]++;
    }
    write_npy_i32(argv[2], fid);
    std::printf("F=%lld  n_labels=%zu  -> %s\n", (long long)F, cnt.size(), argv[2]);
    for (auto&kv:cnt) std::printf("  label %d : %lld faces (%.1f%%)\n", kv.first, (long long)kv.second, 100.0*kv.second/F);
    return 0;
}
