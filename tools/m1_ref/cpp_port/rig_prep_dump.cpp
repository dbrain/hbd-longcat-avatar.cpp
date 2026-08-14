// rig_prep_dump.cpp — dump the EXACT rig conditioning point cloud that image_to_rig builds.
//
// image_to_rig feeds the rigger `rig::prep_mesh_for_rig_inmem(mesh, N=8192, M=512, seed=0)`.
// skintokens_e2e (the chain driver, which owns the structural-selection gate) reads the same
// cloud from `vertices.npy` + `normals.npy`. This tool is the bridge: given any GLB it writes a
// rig-input dir that skintokens_e2e can consume, so a mesh change can be A/B'd through the rig
// WITHOUT re-running diffusion, DC or the texture bake.
//
//   ./rig_prep_dump <mesh.glb> <out_dir> [N=8192] [M=512] [proxy=full|central-core|side-band]
//
// CPU only, no ggml.
#include "mesh_sample.hpp"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <sys/stat.h>

static void npy_write_f32(const std::string& path, const float* data, int64_t rows, int64_t cols) {
    std::string hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
                      std::to_string(rows) + ", " + std::to_string(cols) + "), }";
    size_t pre = 10 + hdr.size() + 1;
    while (pre % 64) { hdr += ' '; pre = 10 + hdr.size() + 1; }
    hdr += '\n';
    std::ofstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot write %s\n", path.c_str()); std::exit(2); }
    const char magic[] = "\x93NUMPY";
    f.write(magic, 6);
    uint8_t ver[2] = { 1, 0 }; f.write((const char*)ver, 2);
    uint16_t hlen = (uint16_t)hdr.size(); f.write((const char*)&hlen, 2);
    f.write(hdr.data(), hdr.size());
    f.write((const char*)data, (size_t)rows * cols * sizeof(float));
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <mesh.glb> <out_dir> [N=8192] [M=512] [proxy=full] [seed=0]\n", argv[0]);
        return 2;
    }
    const char* glb = argv[1];
    std::string outdir = argv[2];
    int N = argc > 3 ? std::atoi(argv[3]) : 8192;
    int M = argc > 4 ? std::atoi(argv[4]) : 512;
    rig::RigGuideProxy proxy = rig::RigGuideProxy::Full;
    if (argc > 5 && !rig::parse_rig_guide_proxy(argv[5], proxy)) {
        std::fprintf(stderr, "bad proxy '%s'\n", argv[5]); return 2;
    }
    const uint64_t seed = argc > 6 ? std::strtoull(argv[6], nullptr, 10) : 0;
    rig::PrepResult P = rig::prep_mesh_for_rig(glb, N, M, seed, proxy);
    if (!P.ok) { std::fprintf(stderr, "prep failed for %s\n", glb); return 1; }
    mkdir(outdir.c_str(), 0755);
    npy_write_f32(outdir + "/vertices.npy", P.vertices.data(), P.N, 3);
    npy_write_f32(outdir + "/normals.npy",  P.normals.data(),  P.N, 3);
    npy_write_f32(outdir + "/sampled_pc.npy",    P.sampled_pc.data(),    P.M, 3);
    npy_write_f32(outdir + "/sampled_feats.npy", P.sampled_feats.data(), P.M, 3);
    // A quick shape summary so two clouds can be compared without loading numpy.
    double mn[3] = { 1e30, 1e30, 1e30 }, mx[3] = { -1e30, -1e30, -1e30 }, mean[3] = { 0, 0, 0 };
    for (int i = 0; i < P.N; ++i) for (int k = 0; k < 3; ++k) {
        double v = P.vertices[(size_t)i * 3 + k];
        mn[k] = std::min(mn[k], v); mx[k] = std::max(mx[k], v); mean[k] += v / P.N;
    }
    std::printf("prep %s -> %s  N=%d M=%d proxy=%s\n", glb, outdir.c_str(), P.N, P.M,
                rig::rig_guide_proxy_name(proxy));
    std::printf("  bbox x[%.3f %.3f] y[%.3f %.3f] z[%.3f %.3f]  mean=(%.4f %.4f %.4f)\n",
                mn[0], mx[0], mn[1], mx[1], mn[2], mx[2], mean[0], mean[1], mean[2]);
    return 0;
}
