// Standalone CPU reference of 3D sparse convolution, mirroring
// sparse_conv_ref.py EXACTLY (same offset order, f64 accumulate, same
// output ordering) so it matches the goldens to ~0.  This is the eventual
// ggml-op compute kernel, proved correct in C++ before CUDA/perf work.
//
// Build (CPU-only, no ggml/CUDA):
//   g++ -O2 -std=c++17 sparse_conv.cpp -o sparse_conv_test
// Run:
//   ./sparse_conv_test golden_ref          # validate all cases in a dir
//
// See HANDOFF-sparse-conv-spike.md "SKIMP LOG": this is correctness-only;
// rulebook reuse / tensor-core GEMM / fp16 / coalesced scatter are the perf
// phase against the FlexGEMM baseline.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

#include "npy.hpp"

// pack (b,z,y,x) like sparse_conv_ref._coord_key (20-bit fields)
static inline int64_t coord_key(int32_t b, int32_t z, int32_t y, int32_t x) {
    auto m = [](int32_t v) { return (int64_t)(v & 0xFFFFF); };
    return (((((int64_t)b << 20) | m(z)) << 20 | m(y)) << 20) | m(x);
}

struct KOff { int dz, dy, dx; };
static std::vector<KOff> kernel_offsets(int K, int dilation) {
    int c = (K - 1) / 2;
    std::vector<KOff> offs;
    for (int kz = 0; kz < K; kz++)
        for (int ky = 0; ky < K; ky++)
            for (int kx = 0; kx < K; kx++)
                offs.push_back({(kz - c) * dilation, (ky - c) * dilation, (kx - c) * dilation});
    return offs;  // index == weight first dim, z-major
}

// SubMConv3d: out coords == in coords; only active neighbours contribute.
static std::vector<float> submconv3d(const int32_t* coords, int N,
                                     const float* feats, int Cin,
                                     const float* weight, int Cout,
                                     const float* bias, int K, int dilation) {
    auto offs = kernel_offsets(K, dilation);
    std::unordered_map<int64_t, int> index;
    index.reserve(N * 2);
    for (int i = 0; i < N; i++)
        index[coord_key(coords[i*4+0], coords[i*4+1], coords[i*4+2], coords[i*4+3])] = i;

    std::vector<float> out((size_t)N * Cout);
    std::vector<double> acc(Cout);
    for (int i = 0; i < N; i++) {
        std::fill(acc.begin(), acc.end(), 0.0);
        int32_t b = coords[i*4+0], z = coords[i*4+1], y = coords[i*4+2], x = coords[i*4+3];
        for (size_t k = 0; k < offs.size(); k++) {
            auto it = index.find(coord_key(b, z + offs[k].dz, y + offs[k].dy, x + offs[k].dx));
            if (it == index.end()) continue;
            int j = it->second;
            const float* fj = feats + (size_t)j * Cin;
            const float* wk = weight + k * (size_t)Cin * Cout;  // [Cin,Cout]
            for (int ci = 0; ci < Cin; ci++) {
                double fv = fj[ci];
                const float* wrow = wk + (size_t)ci * Cout;
                for (int co = 0; co < Cout; co++) acc[co] += fv * wrow[co];
            }
        }
        for (int co = 0; co < Cout; co++)
            out[(size_t)i * Cout + co] = (float)(acc[co] + (bias ? bias[co] : 0.0));
    }
    return out;
}

static const uint32_t SENTINEL = 0xFFFFFFFFu;

// Rulebook build — mirrors sparse_conv_ref.build_neighbor_map (and the CUDA
// hashmap_build kernel we'll port). neighbor_map[n,v] = input index or SENTINEL.
static std::vector<uint32_t> build_neighbor_map(const int32_t* coords, int N,
                                                int K, int dilation) {
    auto offs = kernel_offsets(K, dilation);
    int V = (int)offs.size();
    std::unordered_map<int64_t, int> index;
    index.reserve(N * 2);
    for (int i = 0; i < N; i++)
        index[coord_key(coords[i*4+0], coords[i*4+1], coords[i*4+2], coords[i*4+3])] = i;
    std::vector<uint32_t> nmap((size_t)N * V, SENTINEL);
    for (int i = 0; i < N; i++) {
        int32_t b = coords[i*4+0], z = coords[i*4+1], y = coords[i*4+2], x = coords[i*4+3];
        for (int v = 0; v < V; v++) {
            auto it = index.find(coord_key(b, z + offs[v].dz, y + offs[v].dy, x + offs[v].dx));
            if (it != index.end()) nmap[(size_t)i * V + v] = (uint32_t)it->second;
        }
    }
    return nmap;
}

// Implicit-GEMM dataflow: gather via rulebook, accumulate (no im2col buffer).
// This is exactly the FlexGEMM/CUDA inner loop, on CPU.
static std::vector<float> submconv3d_via_rulebook(const float* feats, int N, int Cin,
                                                  const float* weight, int Cout,
                                                  const float* bias, const uint32_t* nmap, int V) {
    std::vector<float> out((size_t)N * Cout);
    std::vector<double> acc(Cout);
    for (int i = 0; i < N; i++) {
        std::fill(acc.begin(), acc.end(), 0.0);
        for (int v = 0; v < V; v++) {
            uint32_t j = nmap[(size_t)i * V + v];
            if (j == SENTINEL) continue;
            const float* fj = feats + (size_t)j * Cin;
            const float* wk = weight + (size_t)v * Cin * Cout;
            for (int ci = 0; ci < Cin; ci++) {
                double fv = fj[ci];
                const float* wrow = wk + (size_t)ci * Cout;
                for (int co = 0; co < Cout; co++) acc[co] += fv * wrow[co];
            }
        }
        for (int co = 0; co < Cout; co++)
            out[(size_t)i * Cout + co] = (float)(acc[co] + (bias ? bias[co] : 0.0));
    }
    return out;
}

// SparseConv3d strided/padded: new down-sampled active set (sorted by b,z,y,x).
struct OutVox { int32_t b, z, y, x; std::vector<double> acc; };
static void sparseconv3d(const int32_t* coords, int N, const float* feats, int Cin,
                         const float* weight, int Cout, const float* bias,
                         int K, int sz, int sy, int sx, int pz, int py, int px,
                         std::vector<int32_t>& out_coords, std::vector<float>& out_feats) {
    auto offs = kernel_offsets(K, 1);
    int c = (K - 1) / 2;
    std::unordered_map<int64_t, int> oindex;  // key -> position in vox
    std::vector<OutVox> vox;
    for (int i = 0; i < N; i++) {
        int32_t b = coords[i*4+0], z = coords[i*4+1], y = coords[i*4+2], x = coords[i*4+3];
        const float* fi = feats + (size_t)i * Cin;
        for (size_t k = 0; k < offs.size(); k++) {
            int kz = offs[k].dz + c, ky = offs[k].dy + c, kx = offs[k].dx + c;  // [0,K)
            int nz = z + pz - (kz - c), ny = y + py - (ky - c), nx = x + px - (kx - c);
            if (nz % sz || ny % sy || nx % sx) continue;
            int oz = nz / sz, oy = ny / sy, ox = nx / sx;
            if (oz < 0 || oy < 0 || ox < 0) continue;
            int64_t key = coord_key(b, oz, oy, ox);
            auto it = oindex.find(key);
            int pos;
            if (it == oindex.end()) {
                pos = (int)vox.size();
                oindex[key] = pos;
                vox.push_back({b, oz, oy, ox, std::vector<double>(Cout, 0.0)});
            } else pos = it->second;
            const float* wk = weight + k * (size_t)Cin * Cout;
            for (int ci = 0; ci < Cin; ci++) {
                double fv = fi[ci];
                const float* wrow = wk + (size_t)ci * Cout;
                for (int co = 0; co < Cout; co++) vox[pos].acc[co] += fv * wrow[co];
            }
        }
    }
    std::sort(vox.begin(), vox.end(), [](const OutVox& a, const OutVox& b) {
        if (a.b != b.b) return a.b < b.b;
        if (a.z != b.z) return a.z < b.z;
        if (a.y != b.y) return a.y < b.y;
        return a.x < b.x;
    });
    out_coords.resize(vox.size() * 4);
    out_feats.resize(vox.size() * Cout);
    for (size_t i = 0; i < vox.size(); i++) {
        out_coords[i*4+0] = vox[i].b; out_coords[i*4+1] = vox[i].z;
        out_coords[i*4+2] = vox[i].y; out_coords[i*4+3] = vox[i].x;
        for (int co = 0; co < Cout; co++)
            out_feats[i * Cout + co] = (float)(vox[i].acc[co] + (bias ? bias[co] : 0.0));
    }
}

// --- tiny JSON field readers (manifest is flat) ---
#include <fstream>
#include <sstream>
static std::string read_file(const std::string& p) {
    std::ifstream f(p); std::stringstream ss; ss << f.rdbuf(); return ss.str();
}
static int json_int(const std::string& s, const std::string& key, int def) {
    auto p = s.find("\"" + key + "\"");
    if (p == std::string::npos) return def;
    p = s.find(':', p) + 1;
    while (p < s.size() && (s[p] == ' ' || s[p] == '\t')) p++;
    if (p < s.size() && (s[p] == 'n')) return def;  // null
    return std::atoi(s.c_str() + p);
}
static std::string json_str(const std::string& s, const std::string& key) {
    auto p = s.find("\"" + key + "\"");
    if (p == std::string::npos) return "";
    p = s.find(':', p); auto q1 = s.find('"', p + 1); auto q2 = s.find('"', q1 + 1);
    return s.substr(q1 + 1, q2 - q1 - 1);
}

static int run_case(const std::string& dir) {
    std::string man = read_file(dir + "/manifest.json");
    std::string op = json_str(man, "op");
    int K = json_int(man, "kernel_size", 3);
    auto in_coords = npy_load(dir + "/in_coords.npy");
    auto in_feats  = npy_load(dir + "/in_feats.npy");
    auto weight    = npy_load(dir + "/weight.npy");
    auto bias      = npy_load(dir + "/bias.npy");
    auto out_feats = npy_load(dir + "/out_feats.npy");
    auto out_coords = npy_load(dir + "/out_coords.npy");
    int N = (int)in_coords.shape[0], Cin = (int)in_feats.shape[1], Cout = (int)weight.shape[2];

    double maxerr = 0.0; bool coords_ok = true; size_t nout = out_feats.shape[0];
    if (op == "subm") {
        auto pred = submconv3d(in_coords.i32(), N, in_feats.f32(), Cin,
                               weight.f32(), Cout, bias.f32(), K, 1);
        for (size_t i = 0; i < pred.size(); i++)
            maxerr = std::max(maxerr, (double)std::fabs(pred[i] - out_feats.f32()[i]));
        // also validate the FlexGEMM rulebook + implicit-GEMM dataflow on CPU
        int V = K * K * K;
        auto nmap = build_neighbor_map(in_coords.i32(), N, K, 1);
        if (std::ifstream(dir + "/neighbor_map.npy")) {
            auto g = npy_load(dir + "/neighbor_map.npy");
            const uint32_t* gp = reinterpret_cast<const uint32_t*>(g.raw.data());
            for (size_t i = 0; i < nmap.size(); i++)
                if (nmap[i] != gp[i]) { coords_ok = false; break; }  // rulebook must match golden
        }
        auto rb = submconv3d_via_rulebook(in_feats.f32(), N, Cin, weight.f32(), Cout,
                                          bias.f32(), nmap.data(), V);
        double rberr = 0.0;
        for (size_t i = 0; i < rb.size(); i++)
            rberr = std::max(rberr, (double)std::fabs(rb[i] - out_feats.f32()[i]));
        maxerr = std::max(maxerr, rberr);  // fold rulebook-path error into the case result
    } else {
        int s = json_int(man, "stride", 2), p = json_int(man, "padding", 0);
        std::vector<int32_t> oc; std::vector<float> of;
        sparseconv3d(in_coords.i32(), N, in_feats.f32(), Cin, weight.f32(), Cout,
                     bias.f32(), K, s, s, s, p, p, p, oc, of);
        if (oc.size() != (size_t)out_coords.numel()) { coords_ok = false; }
        else for (size_t i = 0; i < oc.size(); i++)
            if (oc[i] != out_coords.i32()[i]) { coords_ok = false; break; }
        size_t n = std::min(of.size(), (size_t)out_feats.numel());
        for (size_t i = 0; i < n; i++)
            maxerr = std::max(maxerr, (double)std::fabs(of[i] - out_feats.f32()[i]));
        nout = oc.size() / 4;
    }
    bool pass = (maxerr < 1e-4) && coords_ok;
    printf("  %-16s op=%-6s N=%-4d Cin=%-3d Cout=%-3d K=%d out=%zu  maxerr=%.3e coords=%s  %s\n",
           dir.substr(dir.find_last_of('/') + 1).c_str(), op.c_str(), N, Cin, Cout, K,
           nout, maxerr, coords_ok ? "ok" : "MISMATCH", pass ? "PASS" : "*** FAIL ***");
    return pass ? 0 : 1;
}

#include <dirent.h>
int main(int argc, char** argv) {
    std::string root = argc > 1 ? argv[1] : "golden_ref";
    DIR* d = opendir(root.c_str());
    if (!d) { printf("cannot open %s\n", root.c_str()); return 2; }
    std::vector<std::string> cases;
    for (dirent* e; (e = readdir(d));) {
        std::string n = e->d_name;
        if (n == "." || n == "..") continue;
        if (std::ifstream(root + "/" + n + "/manifest.json")) cases.push_back(n);
    }
    closedir(d);
    std::sort(cases.begin(), cases.end());
    printf("sparse_conv_test: %zu cases in %s\n", cases.size(), root.c_str());
    int fails = 0;
    for (auto& c : cases) fails += run_case(root + "/" + c);
    printf("%s (%d/%zu passed)\n", fails ? "FAILURES" : "ALL PASS",
           (int)cases.size() - fails, cases.size());
    return fails ? 1 : 0;
}
