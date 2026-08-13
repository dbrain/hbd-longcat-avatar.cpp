// Topology-aware generic skin-weight fallback.
//
// `SkinTokens` supplies the skeleton; this tool derives a separate skin field
// from the *surface connectivity* of a clean source mesh.  It intentionally
// does not make creature anatomy claims: joints retain the caller's generated
// tree/names and are used only as point/bone interpolation handles.
//
// We solve second-order harmonic (biharmonic) coordinates on a QEM-reduced
// copy of the mesh, then export its vertices + weights for the existing
// kNN transfer in combine_rig_tex_main.  That makes the expensive sparse solve
// bounded while transfer still reaches every vertex of the textured delivery
// mesh.  Crucially, weights propagate along connected surface paths: a ponytail
// connected at the head cannot acquire arm weight merely because it is close in
// Euclidean 3-D space.
//
// This is an experiment/promotion candidate, never an implicit replacement for
// learned SkinTokens weights.  A caller must run rig_pose_smoke.py after combine
// and visually inspect its skeleton/pose render before publishing an artifact.
//
// Usage:
//   rig_harmonic_weights <clean_surface.glb> <joints.npy> <parents.npy> <out_dir>
//       [--faces 20000] [--bone-samples 2] [--anchor-neighbours 2] [--harmonic 2] [--uniform]
//
// Output out_dir/vertices.npy, gen_skin_pred_harmonic.npy,
// gen_joints.npy, gen_parents.npy, harmonic_metadata.txt.
//
// libigl is already vendored under thirdparty/ManifoldPlus. Its headers are
// MPL-2.0; we call its public `igl::harmonic` API without copying it here.
#include "glb_reader.hpp"
#include "../../sparse_spike/npy.hpp"
#include "../../../thirdparty/meshoptimizer/meshoptimizer.h"

#include <igl/harmonic.h>
#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace {

static void usage(const char* exe) {
    std::fprintf(stderr,
        "usage: %s <clean_surface.glb> <joints.npy> <parents.npy> <out_dir>\n"
        "       [--faces 20000] [--bone-samples 2] [--anchor-neighbours 2] [--harmonic 2] [--uniform]\n",
        exe);
}

static void npy_write(const std::string& path, const void* data, size_t elem_bytes,
                      const char* descr, const std::vector<int64_t>& shape) {
    std::string sh = "(";
    for (int64_t d : shape) { sh += std::to_string(d); sh += ","; }
    sh += ")";
    std::string hdr = std::string("{'descr': '") + descr + "', 'fortran_order': False, 'shape': " + sh + ", }";
    const size_t base = 10 + hdr.size() + 1; // magic + version + uint16 header len + newline
    hdr.append((64 - (base % 64)) % 64, ' ');
    hdr += '\n';
    const uint16_t hlen = (uint16_t)hdr.size();
    int64_t count = 1;
    for (int64_t d : shape) count *= d;
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + path);
    f.write("\x93NUMPY", 6);
    const char version[2] = {1, 0}; f.write(version, 2);
    f.write(reinterpret_cast<const char*>(&hlen), sizeof(hlen));
    f.write(hdr.data(), (std::streamsize)hdr.size());
    f.write(reinterpret_cast<const char*>(data), (std::streamsize)((size_t)count * elem_bytes));
    if (!f) throw std::runtime_error("short write " + path);
}

static bool load_f32_2d(const std::string& path, std::vector<float>& out, int& rows, int& cols) {
    try {
        const NpyArray a = npy_load(path);
        if (a.descr != "<f4" || a.shape.size() != 2 || a.shape[0] <= 0 || a.shape[1] <= 0) {
            std::fprintf(stderr, "expected float32 [N,M] NPY: %s\n", path.c_str());
            return false;
        }
        rows = (int)a.shape[0]; cols = (int)a.shape[1];
        out.assign(a.f32(), a.f32() + a.numel());
        return true;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "cannot read %s: %s\n", path.c_str(), e.what());
        return false;
    }
}

static bool load_i64_1d(const std::string& path, std::vector<int64_t>& out) {
    try {
        const NpyArray a = npy_load(path);
        if (a.descr != "<i8" || a.shape.size() != 1 || a.shape[0] <= 0) {
            std::fprintf(stderr, "expected int64 [N] NPY: %s\n", path.c_str());
            return false;
        }
        out.assign(a.i64(), a.i64() + a.numel());
        return true;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "cannot read %s: %s\n", path.c_str(), e.what());
        return false;
    }
}

static void normalize_mesh(std::vector<float>& verts) {
    if (verts.empty()) return;
    double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
    for (size_t v = 0; v < verts.size() / 3; ++v) for (int d = 0; d < 3; ++d) {
        lo[d] = std::min(lo[d], (double)verts[v * 3 + d]);
        hi[d] = std::max(hi[d], (double)verts[v * 3 + d]);
    }
    const double c[3] = {(lo[0] + hi[0]) * .5, (lo[1] + hi[1]) * .5, (lo[2] + hi[2]) * .5};
    double amax = 0.0;
    for (size_t v = 0; v < verts.size() / 3; ++v) for (int d = 0; d < 3; ++d)
        amax = std::max(amax, std::fabs((double)verts[v * 3 + d] - c[d]));
    const double inv = amax > 0.0 ? 1.0 / amax : 1.0;
    for (size_t v = 0; v < verts.size() / 3; ++v) for (int d = 0; d < 3; ++d)
        verts[v * 3 + d] = (float)(((double)verts[v * 3 + d] - c[d]) * inv);
}

// QEM reduction only bounds the solver.  We preserve original normalized
// vertex positions rather than creating synthetic centroids, so the kNN
// transfer that follows stays in the exact delivery coordinate frame.
static bool qem_reduce(const std::vector<float>& input_v, const std::vector<int64_t>& input_f,
                       size_t target_faces, std::vector<float>& out_v, std::vector<int>& out_f) {
    const size_t vin = input_v.size() / 3, fin = input_f.size() / 3;
    if (vin == 0 || fin == 0) return false;
    std::vector<unsigned int> indices(input_f.size());
    for (size_t i = 0; i < input_f.size(); ++i) {
        if (input_f[i] < 0 || (size_t)input_f[i] >= vin) return false;
        indices[i] = (unsigned int)input_f[i];
    }
    if (target_faces == 0 || target_faces >= fin) {
        out_v = input_v; out_f.resize(indices.size());
        for (size_t i = 0; i < indices.size(); ++i) out_f[i] = (int)indices[i];
        return true;
    }
    std::vector<unsigned int> simplified(indices.size());
    float error = 0.f;
    const size_t written = meshopt_simplify(simplified.data(), indices.data(), indices.size(),
        input_v.data(), vin, 3 * sizeof(float), target_faces * 3, 1e-2f,
        meshopt_SimplifyLockBorder, &error);
    if (written < 3) return false;
    simplified.resize(written);
    std::vector<unsigned int> remap(vin, std::numeric_limits<unsigned int>::max());
    out_v.clear(); out_v.reserve(written);
    out_f.resize(written);
    for (size_t i = 0; i < written; ++i) {
        const unsigned int old = simplified[i];
        if (remap[old] == std::numeric_limits<unsigned int>::max()) {
            remap[old] = (unsigned int)(out_v.size() / 3);
            out_v.push_back(input_v[old * 3]); out_v.push_back(input_v[old * 3 + 1]); out_v.push_back(input_v[old * 3 + 2]);
        }
        out_f[i] = (int)remap[old];
    }
    std::printf("  QEM reduced %zuF/%zuV -> %zuF/%zuV (error %.5g)\n", fin, vin, out_f.size() / 3, out_v.size() / 3, error);
    return true;
}

static std::vector<int> nearest_vertices(const std::vector<float>& verts, const double p[3], int count) {
    std::vector<std::pair<double, int>> best;
    best.reserve((size_t)count);
    for (int vi = 0; vi < (int)verts.size() / 3; ++vi) {
        const double dx = verts[(size_t)vi * 3] - p[0];
        const double dy = verts[(size_t)vi * 3 + 1] - p[1];
        const double dz = verts[(size_t)vi * 3 + 2] - p[2];
        const double d2 = dx * dx + dy * dy + dz * dz;
        if ((int)best.size() < count) {
            best.emplace_back(d2, vi);
            std::sort(best.begin(), best.end());
        } else if (d2 < best.back().first) {
            best.back() = {d2, vi};
            std::sort(best.begin(), best.end());
        }
    }
    std::vector<int> out; out.reserve(best.size());
    for (const auto& e : best) out.push_back(e.second);
    return out;
}

static void add_anchor(std::map<int, std::vector<double>>& constraints, const std::vector<float>& verts,
                       const double p[3], const std::vector<double>& row, int neighbour_count) {
    for (int vi : nearest_vertices(verts, p, neighbour_count)) {
        auto& value = constraints[vi];
        if (value.empty()) value.assign(row.size(), 0.0);
        for (size_t j = 0; j < row.size(); ++j) value[j] += row[j];
    }
}

static bool valid_tree(const std::vector<int64_t>& parents) {
    int roots = 0;
    for (int j = 0; j < (int)parents.size(); ++j) {
        if (parents[(size_t)j] == -1) { ++roots; continue; }
        if (parents[(size_t)j] < 0 || parents[(size_t)j] >= (int)parents.size()) return false;
        int p = j;
        for (int depth = 0; depth <= (int)parents.size(); ++depth) {
            p = (int)parents[(size_t)p];
            if (p == -1) break;
            if (p < 0 || p >= (int)parents.size()) return false;
            if (depth == (int)parents.size()) return false;
        }
    }
    return roots >= 1;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 5) { usage(argv[0]); return 2; }
    const std::string mesh_path = argv[1], joints_path = argv[2], parents_path = argv[3], out_dir = argv[4];
    int target_faces = 20000, bone_samples = 2, anchor_neighbours = 2, harmonic_order = 2;
    bool uniform_laplacian = false;
    for (int i = 5; i < argc; ++i) {
        const std::string f = argv[i];
        auto next = [&]() -> int { return ++i < argc ? std::atoi(argv[i]) : -1; };
        if (f == "--faces") target_faces = next();
        else if (f == "--bone-samples") bone_samples = next();
        else if (f == "--anchor-neighbours") anchor_neighbours = next();
        else if (f == "--harmonic") harmonic_order = next();
        else if (f == "--uniform") uniform_laplacian = true;
        else { std::fprintf(stderr, "unknown argument: %s\n", f.c_str()); return 2; }
    }
    if (target_faces < 100 || bone_samples < 0 || anchor_neighbours < 1 || anchor_neighbours > 8 || harmonic_order != 1 && harmonic_order != 2) {
        std::fprintf(stderr, "invalid solver options\n"); return 2;
    }

    glb::Mesh mesh;
    if (!glb::read_glb(mesh_path.c_str(), mesh) || mesh.verts.empty() || mesh.faces.empty()) {
        std::fprintf(stderr, "cannot read a triangle surface from %s\n", mesh_path.c_str()); return 1;
    }
    std::vector<float> joints; int joint_rows = 0, joint_cols = 0;
    std::vector<int64_t> parents;
    if (!load_f32_2d(joints_path, joints, joint_rows, joint_cols) || joint_cols != 3 ||
        !load_i64_1d(parents_path, parents) || (int)parents.size() != joint_rows || !valid_tree(parents)) {
        std::fprintf(stderr, "invalid skeleton inputs (need joints [J,3] and an acyclic parent forest)\n"); return 1;
    }
    const int J = joint_rows;
    normalize_mesh(mesh.verts); // exact mesh_sample/combine frame convention

    std::vector<float> solve_v;
    std::vector<int> solve_f;
    if (!qem_reduce(mesh.verts, mesh.faces, (size_t)target_faces, solve_v, solve_f)) {
        std::fprintf(stderr, "QEM reduction failed\n"); return 1;
    }
    const int V = (int)solve_v.size() / 3, F = (int)solve_f.size() / 3;
    if (V < J * 4 || F < J * 4) { std::fprintf(stderr, "reduced mesh is too small for %d joints\n", J); return 1; }

    std::map<int, std::vector<double>> constraints;
    // Every joint is a one-hot handle. Each parent->child segment adds evenly
    // interpolated handles, which avoids the nearest-joint Voronoi discontinuity.
    for (int j = 0; j < J; ++j) {
        std::vector<double> row((size_t)J, 0.0); row[(size_t)j] = 1.0;
        const double p[3] = {joints[(size_t)j * 3], joints[(size_t)j * 3 + 1], joints[(size_t)j * 3 + 2]};
        add_anchor(constraints, solve_v, p, row, anchor_neighbours);
    }
    for (int child = 0; child < J; ++child) {
        const int parent = (int)parents[(size_t)child];
        if (parent < 0) continue;
        for (int s = 1; s <= bone_samples; ++s) {
            const double t = (double)s / (bone_samples + 1.0);
            std::vector<double> row((size_t)J, 0.0);
            row[(size_t)parent] = 1.0 - t; row[(size_t)child] = t;
            const double p[3] = {
                (1.0 - t) * joints[(size_t)parent * 3] + t * joints[(size_t)child * 3],
                (1.0 - t) * joints[(size_t)parent * 3 + 1] + t * joints[(size_t)child * 3 + 1],
                (1.0 - t) * joints[(size_t)parent * 3 + 2] + t * joints[(size_t)child * 3 + 2]};
            add_anchor(constraints, solve_v, p, row, anchor_neighbours);
        }
    }
    if ((int)constraints.size() < J) { std::fprintf(stderr, "anchor collapse left only %zu constraints for %d joints\n", constraints.size(), J); return 1; }

    Eigen::MatrixXd EV(V, 3);
    Eigen::MatrixXi EF(F, 3);
    for (int v = 0; v < V; ++v) for (int d = 0; d < 3; ++d) EV(v, d) = solve_v[(size_t)v * 3 + d];
    for (int f = 0; f < F; ++f) for (int c = 0; c < 3; ++c) EF(f, c) = solve_f[(size_t)f * 3 + c];
    Eigen::VectorXi b((int)constraints.size());
    Eigen::MatrixXd bc((int)constraints.size(), J);
    int r = 0;
    for (auto& [vi, row] : constraints) {
        double sum = 0; for (double x : row) sum += x;
        if (!(sum > 0.0)) { std::fprintf(stderr, "invalid anchor row\n"); return 1; }
        b(r) = vi;
        for (int j = 0; j < J; ++j) bc(r, j) = row[(size_t)j] / sum;
        ++r;
    }
    std::printf("== topology-aware harmonic skin ==\n");
    std::printf("  source=%s normalized V=%zu F=%zu -> solve V=%d F=%d\n", mesh_path.c_str(), mesh.verts.size() / 3, mesh.faces.size() / 3, V, F);
    std::printf("  skeleton J=%d; anchors=%d (%d nearest surface vertices/handle; %d interior samples/bone)\n",
                J, b.size(), anchor_neighbours, bone_samples);
    std::printf("  solve: %d-harmonic coordinates (libigl %s surface operator)\n", harmonic_order,
                uniform_laplacian ? "uniform-topology" : "cotangent");
    const auto begun = std::chrono::steady_clock::now();
    Eigen::MatrixXd W;
    const bool solve_ok = uniform_laplacian
        ? igl::harmonic(EF, b, bc, harmonic_order, W)
        : igl::harmonic(EV, EF, b, bc, harmonic_order, W);
    if (!solve_ok || W.rows() != V || W.cols() != J || !W.allFinite()) {
        std::fprintf(stderr, "harmonic solve failed or returned non-finite weights\n"); return 1;
    }
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - begun).count();
    std::vector<float> weights((size_t)V * J);
    size_t negative = 0; double min_raw = 0.0, max_residual = 0.0, dominant_sum = 0.0;
    for (int v = 0; v < V; ++v) {
        double total = 0.0;
        for (int j = 0; j < J; ++j) {
            const double x = W(v, j);
            min_raw = std::min(min_raw, x);
            if (x < 0.0) ++negative;
            const double clamped = std::max(0.0, x);
            weights[(size_t)v * J + j] = (float)clamped;
            total += clamped;
        }
        if (!(total > 1e-12)) { std::fprintf(stderr, "weight row %d lost all mass after non-negative projection\n", v); return 1; }
        float dominant = 0.f;
        for (int j = 0; j < J; ++j) {
            float& x = weights[(size_t)v * J + j]; x = (float)(x / total);
            dominant = std::max(dominant, x);
        }
        dominant_sum += dominant;
        double check = 0.0; for (int j = 0; j < J; ++j) check += weights[(size_t)v * J + j];
        max_residual = std::max(max_residual, std::fabs(1.0 - check));
    }
    std::filesystem::create_directories(out_dir);
    try {
        npy_write(out_dir + "/vertices.npy", solve_v.data(), sizeof(float), "<f4", {V, 3});
        npy_write(out_dir + "/gen_skin_pred_harmonic.npy", weights.data(), sizeof(float), "<f4", {V, J});
        npy_write(out_dir + "/gen_joints.npy", joints.data(), sizeof(float), "<f4", {J, 3});
        npy_write(out_dir + "/gen_parents.npy", parents.data(), sizeof(int64_t), "<i8", {J});
        std::ofstream meta(out_dir + "/harmonic_metadata.txt");
        meta << "method=libigl " << (uniform_laplacian ? "uniform-topology" : "cotangent") << " surface " << harmonic_order << "-harmonic coordinates\n"
             << "source_vertices=" << mesh.verts.size() / 3 << " source_faces=" << mesh.faces.size() / 3 << "\n"
             << "solve_vertices=" << V << " solve_faces=" << F << " joints=" << J << " anchors=" << b.size() << "\n"
             << "bone_samples=" << bone_samples << " anchor_neighbours=" << anchor_neighbours << "\n"
             << "solve_seconds=" << elapsed << " raw_negative_entries=" << negative << " raw_min=" << min_raw << "\n"
             << "mean_dominant=" << dominant_sum / V << " normalized_sum_max_residual=" << max_residual << "\n"
             << "requires=combine_rig_tex_main then rig_pose_smoke.py --generic-all-influential --pose-gate --show-skeleton\n";
    } catch (const std::exception& e) {
        std::fprintf(stderr, "output failed: %s\n", e.what()); return 1;
    }
    std::printf("  solved in %.2fs; raw negatives=%zu (min %.4g), mean dominant=%.4f, row-sum residual %.3g\n",
                elapsed, negative, min_raw, dominant_sum / V, max_residual);
    std::printf("  wrote %s (use --skin gen_skin_pred_harmonic.npy with combine_rig_tex_main)\n", out_dir.c_str());
    return 0;
}
