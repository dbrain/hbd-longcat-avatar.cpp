// Clean-surface skin-field refiner.
//
// The textured delivery mesh is UV-chart split: vertices on the same physical
// surface can be duplicated, so smoothing weights there creates chart-edge
// artifacts and can exchange influence across coincident/overlapping sheets.
// This utility instead performs all interpolation and smoothing on the clean,
// watertight, pre-texture surface.  Its reduced clean vertices/weights are then
// handed to combine_rig_tex_main for the already-validated textured export.
//
// Usage:
//   rig_clean_surface_skin <clean.glb> <seed_vertices.npy> <seed_skin.npy>
//       <joints.npy> <parents.npy> <out_dir>
//       [--faces 30000] [--rounds 16] [--blend 0.55] [--knn 4]
//
// This is an explicit candidate generator. It never bypasses the subsequent
// actual-GLB pose gate or rendered review.
#include "glb_reader.hpp"
#include "rig_transfer.hpp"
#include "../../sparse_spike/npy.hpp"
#include "../../../thirdparty/meshoptimizer/meshoptimizer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

static void write_npy(const std::string& path, const void* data, size_t elem_size,
                      const char* descr, const std::vector<int64_t>& shape) {
    std::string dims = "(";
    for (int64_t d : shape) { dims += std::to_string(d); dims += ","; }
    dims += ")";
    std::string header = std::string("{'descr': '") + descr + "', 'fortran_order': False, 'shape': " + dims + ", }";
    header.append((64 - ((10 + header.size() + 1) % 64)) % 64, ' ');
    header += '\n';
    const uint16_t header_len = (uint16_t)header.size();
    int64_t count = 1; for (int64_t d : shape) count *= d;
    std::ofstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot write " + path);
    file.write("\x93NUMPY", 6);
    const char version[2] = {1, 0}; file.write(version, 2);
    file.write(reinterpret_cast<const char*>(&header_len), 2);
    file.write(header.data(), (std::streamsize)header.size());
    file.write(reinterpret_cast<const char*>(data), (std::streamsize)((size_t)count * elem_size));
    if (!file) throw std::runtime_error("short write " + path);
}

static bool load_f32_2d(const std::string& path, std::vector<float>& values, int& rows, int& cols) {
    try {
        const NpyArray a = npy_load(path);
        if (a.descr != "<f4" || a.shape.size() != 2 || a.shape[0] <= 0 || a.shape[1] <= 0) return false;
        rows = (int)a.shape[0]; cols = (int)a.shape[1];
        values.assign(a.f32(), a.f32() + a.numel());
        return true;
    } catch (const std::exception& e) { std::fprintf(stderr, "read %s: %s\n", path.c_str(), e.what()); return false; }
}

static bool load_i64_1d(const std::string& path, std::vector<int64_t>& values) {
    try {
        const NpyArray a = npy_load(path);
        if (a.descr != "<i8" || a.shape.size() != 1 || a.shape[0] <= 0) return false;
        values.assign(a.i64(), a.i64() + a.numel());
        return true;
    } catch (const std::exception& e) { std::fprintf(stderr, "read %s: %s\n", path.c_str(), e.what()); return false; }
}

static void normalize(std::vector<float>& v) {
    double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
    for (size_t i = 0; i < v.size() / 3; ++i) for (int d = 0; d < 3; ++d) {
        lo[d] = std::min(lo[d], (double)v[i * 3 + d]); hi[d] = std::max(hi[d], (double)v[i * 3 + d]);
    }
    const double centre[3] = {(lo[0] + hi[0]) * .5, (lo[1] + hi[1]) * .5, (lo[2] + hi[2]) * .5};
    double extent = 0.0;
    for (size_t i = 0; i < v.size() / 3; ++i) for (int d = 0; d < 3; ++d)
        extent = std::max(extent, std::fabs((double)v[i * 3 + d] - centre[d]));
    const double scale = extent > 0 ? 1.0 / extent : 1.0;
    for (size_t i = 0; i < v.size() / 3; ++i) for (int d = 0; d < 3; ++d)
        v[i * 3 + d] = (float)(((double)v[i * 3 + d] - centre[d]) * scale);
}

static bool qem_reduce(const std::vector<float>& input_v, const std::vector<int64_t>& input_f,
                       size_t target_faces, std::vector<float>& out_v, std::vector<int64_t>& out_f) {
    const size_t V = input_v.size() / 3, F = input_f.size() / 3;
    if (V == 0 || F == 0) return false;
    std::vector<unsigned int> in(input_f.size());
    for (size_t i = 0; i < in.size(); ++i) {
        if (input_f[i] < 0 || (size_t)input_f[i] >= V) return false;
        in[i] = (unsigned int)input_f[i];
    }
    std::vector<unsigned int> simplified;
    if (target_faces >= F) simplified = in;
    else {
        simplified.resize(in.size()); float err = 0.f;
        const size_t count = meshopt_simplify(simplified.data(), in.data(), in.size(), input_v.data(), V,
            3 * sizeof(float), target_faces * 3, 1e-2f, meshopt_SimplifyLockBorder, &err);
        simplified.resize(count);
        std::printf("  QEM surface: %zuF -> %zuF (error %.5g)\n", F, count / 3, err);
    }
    if (simplified.size() < 3) return false;
    std::vector<unsigned int> remap(V, std::numeric_limits<unsigned int>::max());
    out_v.clear(); out_f.resize(simplified.size());
    for (size_t i = 0; i < simplified.size(); ++i) {
        const unsigned int old = simplified[i];
        if (remap[old] == std::numeric_limits<unsigned int>::max()) {
            remap[old] = (unsigned int)(out_v.size() / 3);
            out_v.insert(out_v.end(), {input_v[old * 3], input_v[old * 3 + 1], input_v[old * 3 + 2]});
        }
        out_f[i] = remap[old];
    }
    return true;
}

static void smooth_clean_surface(const std::vector<int64_t>& faces, size_t V, int J,
                                 std::vector<float>& weights, int rounds, float blend) {
    std::vector<std::vector<int>> neighbours(V);
    for (size_t f = 0; f + 2 < faces.size(); f += 3) {
        const int a = (int)faces[f], b = (int)faces[f + 1], c = (int)faces[f + 2];
        for (const auto pair : {std::pair<int,int>{a,b}, {b,c}, {c,a}}) {
            neighbours[(size_t)pair.first].push_back(pair.second);
            neighbours[(size_t)pair.second].push_back(pair.first);
        }
    }
    for (auto& row : neighbours) {
        std::sort(row.begin(), row.end()); row.erase(std::unique(row.begin(), row.end()), row.end());
    }
    std::vector<float> next(weights.size());
    for (int pass = 0; pass < rounds; ++pass) {
        #pragma omp parallel for schedule(static)
        for (long long vi = 0; vi < (long long)V; ++vi) {
            const auto& nb = neighbours[(size_t)vi];
            const float* old = &weights[(size_t)vi * J]; float* dst = &next[(size_t)vi * J];
            if (nb.empty()) { std::copy(old, old + J, dst); continue; }
            float sum = 0.f;
            for (int j = 0; j < J; ++j) {
                double avg = 0.0;
                for (int other : nb) avg += weights[(size_t)other * J + j];
                dst[j] = (1.f - blend) * old[j] + blend * (float)(avg / nb.size());
                sum += dst[j];
            }
            if (sum > 1e-12f) for (int j = 0; j < J; ++j) dst[j] /= sum;
        }
        weights.swap(next);
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 7) {
        std::fprintf(stderr, "usage: %s <clean.glb> <seed_vertices.npy> <seed_skin.npy> <joints.npy> <parents.npy> <out_dir> [--faces 30000] [--rounds 16] [--blend 0.55] [--knn 4]\n", argv[0]);
        return 2;
    }
    int target_faces = 30000, rounds = 16, knn = 4; float blend = .55f;
    for (int i = 7; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--faces" && i + 1 < argc) target_faces = std::atoi(argv[++i]);
        else if (flag == "--rounds" && i + 1 < argc) rounds = std::atoi(argv[++i]);
        else if (flag == "--blend" && i + 1 < argc) blend = std::atof(argv[++i]);
        else if (flag == "--knn" && i + 1 < argc) knn = std::atoi(argv[++i]);
        else { std::fprintf(stderr, "bad option %s\n", flag.c_str()); return 2; }
    }
    if (target_faces < 100 || rounds < 0 || rounds > 128 || !(blend >= 0.f && blend <= 1.f) || knn < 1 || knn > 16) {
        std::fprintf(stderr, "invalid refinement options\n"); return 2;
    }
    glb::Mesh mesh;
    if (!glb::read_glb(argv[1], mesh) || mesh.verts.empty() || mesh.faces.empty()) return 1;
    std::vector<float> seed_v, seed_w, joints; std::vector<int64_t> parents;
    int Ns = 0, xyz = 0, skin_rows = 0, J = 0, joint_rows = 0, joint_xyz = 0;
    if (!load_f32_2d(argv[2], seed_v, Ns, xyz) || xyz != 3 || !load_f32_2d(argv[3], seed_w, skin_rows, J) || skin_rows != Ns ||
        !load_f32_2d(argv[4], joints, joint_rows, joint_xyz) || joint_xyz != 3 || joint_rows != J ||
        !load_i64_1d(argv[5], parents) || (int)parents.size() != J) {
        std::fprintf(stderr, "invalid source skin/skeleton NPY shapes\n"); return 1;
    }
    normalize(mesh.verts);
    std::vector<float> solve_v; std::vector<int64_t> solve_f;
    if (!qem_reduce(mesh.verts, mesh.faces, (size_t)target_faces, solve_v, solve_f)) return 1;
    const size_t V = solve_v.size() / 3;
    std::vector<float> refined;
    rig::BBox source_box, solve_box;
    const auto started = std::chrono::steady_clock::now();
    rig::transfer_skin(seed_v, seed_w, J, solve_v, refined, knn, &source_box, &solve_box);
    smooth_clean_surface(solve_f, V, J, refined, rounds, blend);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    double dominant = 0.0, sum_error = 0.0;
    for (size_t v = 0; v < V; ++v) {
        float peak = 0.f, sum = 0.f;
        for (int j = 0; j < J; ++j) { const float x = refined[v * J + j]; peak = std::max(peak, x); sum += x; }
        dominant += peak; sum_error = std::max(sum_error, std::fabs(1.0 - sum));
    }
    try {
        std::filesystem::create_directories(argv[6]); const std::string out = argv[6];
        write_npy(out + "/vertices.npy", solve_v.data(), sizeof(float), "<f4", {(int64_t)V, 3});
        write_npy(out + "/gen_skin_pred_clean_surface.npy", refined.data(), sizeof(float), "<f4", {(int64_t)V, J});
        write_npy(out + "/gen_joints.npy", joints.data(), sizeof(float), "<f4", {J, 3});
        write_npy(out + "/gen_parents.npy", parents.data(), sizeof(int64_t), "<i8", {J});
        std::ofstream meta(out + "/clean_surface_metadata.txt");
        meta << "method=learned_skin_kNN_then_clean_watertight_surface_smoothing\n"
             << "source_samples=" << Ns << " joints=" << J << " clean_vertices=" << V << " clean_faces=" << solve_f.size() / 3 << "\n"
             << "rounds=" << rounds << " blend=" << blend << " knn=" << knn << " seconds=" << seconds << "\n"
             << "mean_dominant=" << dominant / V << " sum_max_error=" << sum_error << "\n"
             << "requires=combine_rig_tex_main and rig_pose_smoke rendered pose gate\n";
    } catch (const std::exception& e) { std::fprintf(stderr, "write failed: %s\n", e.what()); return 1; }
    std::printf("== clean-surface skin refinement ==\n  clean solve V=%zu F=%zu J=%d\n", V, solve_f.size() / 3, J);
    std::printf("  rounds=%d blend=%.3f kNN=%d time=%.2fs mean-dominant=%.4f sum-error=%.2g\n",
                rounds, blend, knn, seconds, dominant / V, sum_error);
    std::printf("  wrote %s (then combine + pose-gate; not self-promoting)\n", argv[6]);
    return 0;
}
