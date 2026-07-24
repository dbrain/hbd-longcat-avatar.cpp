// mesh_sample_main.cpp — CLI front-end for the auto-rig mesh preprocessing.
//
//   mesh_sample <mesh.glb> <out_dir> [N=8192] [M=512] [seed=0]
//               [guide=full|central-core|side-band]
//
// Loads an arbitrary GLB, normalizes to [-1,1]^3, area-weighted surface-samples N points + normals,
// and derives the R1 VecSet query sampled_pc[M] (choice N->4M then FPS 4M->M from idx 0). Writes:
//   <out_dir>/vertices.npy      [N,3] f32
//   <out_dir>/normals.npy       [N,3] f32
//   <out_dir>/sampled_pc.npy    [M,3] f32  (R1 VecSet query points)
//   <out_dir>/sampled_feats.npy [M,3] f32  (the normals AT those query points — R1 feats, fresh)
// These are drop-in replacements for the banked giraffe npys consumed by skintokens_e2e (r1=real):
//   vertices.npy/normals.npy       -> the e2e dir (-e arg, default /tmp/skintokens_e2e)
//   sampled_pc.npy/sampled_feats.npy -> the r1 weight dir (-r1w arg, default /tmp/vecset_r0)
// With sampled_feats emitted here, r1=real no longer needs ANY banked giraffe npys for the query
// feats (only the output_proj.* ckpt weights, which are mesh-agnostic).
//
// Pure CPU. No ggml/CUDA. Build:  ./build.sh mesh_sample_main
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include "mesh_sample.hpp"

// minimal little-endian .npy writer (2-D shape), mirrors skintokens_e2e.cpp::npy_write.
static void npy_write(const std::string& path, const float* data,
                      const std::vector<int64_t>& shape) {
    std::string sh = "(";
    for (size_t i = 0; i < shape.size(); ++i) { sh += std::to_string(shape[i]); sh += ","; }
    sh += ")";
    std::string hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': " + sh + ", }";
    size_t base = 10 + hdr.size() + 1;
    size_t pad = (64 - (base % 64)) % 64;
    hdr.append(pad, ' '); hdr += '\n';
    uint16_t hlen = (uint16_t)hdr.size();
    int64_t numel = 1; for (auto d : shape) numel *= d;
    std::ofstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "npy_write: cannot open %s\n", path.c_str()); return; }
    f.write("\x93NUMPY", 6);
    char ver[2] = { 1, 0 }; f.write(ver, 2);
    f.write(reinterpret_cast<const char*>(&hlen), 2);
    f.write(hdr.data(), hdr.size());
    f.write(reinterpret_cast<const char*>(data), (size_t)numel * sizeof(float));
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <mesh.glb> <out_dir> [N=8192] [M=512] [seed=0]\n", argv[0]);
        return 2;
    }
    const char* glb = argv[1];
    std::string out_dir = argv[2];
    int N = argc > 3 ? std::atoi(argv[3]) : 8192;
    int M = argc > 4 ? std::atoi(argv[4]) : 512;
    uint64_t seed = argc > 5 ? (uint64_t)std::strtoull(argv[5], nullptr, 10) : 0;
    rig::RigGuideProxy guide = rig::RigGuideProxy::Full;
    if (argc > 6 && !rig::parse_rig_guide_proxy(argv[6], guide)) {
        std::fprintf(stderr, "unknown guide '%s' (expected full, central-core, or side-band)\n", argv[6]);
        return 2;
    }

    rig::PrepResult R = rig::prep_mesh_for_rig(glb, N, M, seed, guide);
    if (!R.ok) { std::fprintf(stderr, "mesh_sample: prep failed\n"); return 1; }

    npy_write(out_dir + "/vertices.npy",      R.vertices.data(),      { R.N, 3 });
    npy_write(out_dir + "/normals.npy",       R.normals.data(),       { R.N, 3 });
    npy_write(out_dir + "/sampled_pc.npy",    R.sampled_pc.data(),    { R.M, 3 });
    npy_write(out_dir + "/sampled_feats.npy", R.sampled_feats.data(), { R.M, 3 });
    // The sampled cloud and its learned skin field share the coordinate frame
    // of this exact mesh input.  Keep that source beside the arrays so the
    // transfer stage cannot accidentally combine an old sample set with a
    // differently transformed delivery GLB (for example an FBX scene-node
    // transform newly honoured by the GLB reader).
    {
        std::ofstream provenance(out_dir + "/sampling_provenance.txt");
        if (!provenance) {
            std::fprintf(stderr, "mesh_sample: cannot write sampling provenance\n");
            return 1;
        }
        provenance << "schema_version=1\n"
                   << "mesh_source_path=" << glb << "\n"
                   << "normalization=bbox-midpoint; global-max-absolute-extent\n"
                   << "sample_count=" << R.N << "\n"
                   << "query_count=" << R.M << "\n"
                   << "seed=" << seed << "\n"
                   << "guide=" << rig::rig_guide_proxy_name(guide) << "\n";
    }

    // ---- sanity stats ----
    auto bbox = [](const std::vector<float>& p, int n, double mn[3], double mx[3]) {
        for (int c = 0; c < 3; ++c) { mn[c] = 1e300; mx[c] = -1e300; }
        for (int i = 0; i < n; ++i)
            for (int c = 0; c < 3; ++c) {
                double x = p[(size_t)i * 3 + c];
                if (x < mn[c]) mn[c] = x;
                if (x > mx[c]) mx[c] = x;
            }
    };
    double smn[3], smx[3];
    bbox(R.vertices, R.N, smn, smx);

    // normal unit-length check
    double nmin = 1e300, nmax = -1e300, nsum = 0;
    int nbad = 0;
    for (int i = 0; i < R.N; ++i) {
        double a = R.normals[(size_t)i * 3 + 0], b = R.normals[(size_t)i * 3 + 1], c = R.normals[(size_t)i * 3 + 2];
        double L = std::sqrt(a * a + b * b + c * c);
        nsum += L; if (L < nmin) nmin = L; if (L > nmax) nmax = L;
        if (std::fabs(L - 1.0) > 1e-3) nbad++;
    }

    // FPS spread: mean nearest-neighbor spacing within the M selected points.
    double nn_sum = 0; double nn_min = 1e300, nn_max = -1e300;
    for (int i = 0; i < R.M; ++i) {
        double best = 1e300;
        const float* pi = &R.sampled_pc[(size_t)i * 3];
        for (int j = 0; j < R.M; ++j) {
            if (j == i) continue;
            const float* pj = &R.sampled_pc[(size_t)j * 3];
            double dx = (double)pi[0] - pj[0], dy = (double)pi[1] - pj[1], dz = (double)pi[2] - pj[2];
            double d = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (d < best) best = d;
        }
        nn_sum += best; if (best < nn_min) nn_min = best; if (best > nn_max) nn_max = best;
    }

    std::printf("[mesh_sample] mesh=%s -> %s\n", glb, out_dir.c_str());
    std::printf("  N=%d  M=%d  seed=%llu  guide=%s\n", R.N, R.M,
                (unsigned long long)seed, rig::rig_guide_proxy_name(guide));
    std::printf("  samples bbox: [%.4f %.4f %.4f] .. [%.4f %.4f %.4f]\n",
                smn[0], smn[1], smn[2], smx[0], smx[1], smx[2]);
    std::printf("  normals |n|: min=%.5f max=%.5f mean=%.5f  off-unit(>1e-3)=%d/%d %s\n",
                nmin, nmax, nsum / R.N, nbad, R.N, nbad == 0 ? "OK" : "WARN");
    std::printf("  FPS nn-spacing: mean=%.5f min=%.5f max=%.5f (well-spread if min is sizeable)\n",
                nn_sum / R.M, nn_min, nn_max);
    std::printf("  wrote vertices.npy [%d,3], normals.npy [%d,3], sampled_pc.npy [%d,3]\n",
                R.N, R.N, R.M);
    return 0;
}
