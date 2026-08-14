// rig_weight_cleanup.cpp — run the skin-weight cleanup (rig_weight_cleanup.hpp) on a rigged GLB
// that already exists, and write a cleaned one out.
//
//   ./rig_weight_cleanup <rigged.glb> [cleaned.glb] [options]
//
//   RULE 1 — far outlier (drop an influence only if it fails BOTH tests)
//     --far-ratio F        > F x this vertex's own nearest-bone distance                    (4.0)
//     --far-frac-diag F    AND > F x the mesh bbox diagonal                                 (0.12)
//     --min-weight F       influences at or below this are left to rule 2                   (0.01)
//     --inpaint-rounds N   Jacobi waves available to refill emptied vertices                (64)
//     --smooth-rounds N    restricted Laplacian passes over the CHANGED vertices only       (2)
//     --smooth-blend F     per pass: (1-F)*own + F*mean(welded neighbours)                  (0.5)
//   RULE 2 — Lipschitz repair
//     --max-tension F      smooth an edge's endpoints above this tension; 0 disables        (4.0)
//     --tension-rounds N   smoothing waves (the over-tension set shrinks each round)        (24)
//     --tension-blend F    per wave: (1-F)*own + F*mean(welded neighbours)                  (0.6)
//   ACCEPTANCE
//     --tolerance F        no candidate may push either pose-gate number above F x before   (1.001)
//     --unverified         skip the acceptance check entirely (diagnostics only)
//   REPORTING
//     --dry-run            measure and report, write nothing
//     --gate               also print the pose gate + weight health, before and after
//     --json PATH          write the before/after measurement set as JSON
//
// The output GLB is the input file with ONLY its JOINTS_0/WEIGHTS_0 bytes replaced — the texture
// atlas, UV layout, materials, node tree and every accessor offset are bit-identical. Re-runnable on
// any asset the pipeline (or anything else) has already produced.
//
// This is a QUALITY OFFER, not a gate: it never fails, never refuses to write, and never rejects an
// asset. Exit code is 0 unless the file could not be read or written.
#include "rig_glb_skin_io.hpp"
#include "rig_pose_gate.hpp"
#include "rig_weight_cleanup.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

void print_gate(const char* tag, rigqc::SkinnedRig& R) {
    rigqc::WeightHealth H = rigqc::run_weight_health(R);
    std::printf("  [%s] weight health: J=%d V=%d influential=%d mass>=1%%=%d dominant>=1%%verts=%d "
                "biggest_joint_share=%.1f%% %s\n",
                tag, H.J, H.V, H.influential, H.mass_1pct, H.dominant_1pct,
                H.single_share * 100.0, H.pass ? "PASS" : "FAIL");
    rigqc::PoseGateOpts d;                    // the shipped default (named arms, else heaviest joint)
    rigqc::PoseGateOpts a; a.mode = rigqc::PoseGateOpts::GenericAllInfluential;
    rigqc::PoseGateResult Gd = rigqc::run_pose_gate(R, d);
    rigqc::PoseGateResult Ga = rigqc::run_pose_gate(R, a);
    if (Gd.ok) std::printf("  [%s] %s\n", tag, rigqc::pose_gate_line(Gd).c_str());
    if (Ga.ok) std::printf("  [%s] %s\n", tag, rigqc::pose_gate_line(Ga).c_str());
}

std::string json_stats(const char* key, const rigclean::Stats& s) {
    char b[2048];
    std::snprintf(b, sizeof(b),
        "\"%s\":{\"V\":%d,\"J\":%d,\"diag\":%.6f,"
        "\"dom_p50\":%.6f,\"dom_p99\":%.6f,\"dom_p999\":%.6f,\"dom_max\":%.6f,"
        "\"dom_over25\":%ld,\"dom_over35\":%ld,\"dom_over50\":%ld,"
        "\"bone_p50\":%.6f,\"bone_p99\":%.6f,\"bone_p999\":%.6f,\"bone_max\":%.6f,"
        "\"tension_p50\":%.4f,\"tension_p99\":%.4f,\"tension_p999\":%.4f,\"tension_max\":%.4f,"
        "\"tension_over\":%ld,\"tension_verts\":%ld,"
        "\"mean_influences\":%.4f,\"blended_frac\":%.6f,\"rigid_seam_edges\":%ld,"
        "\"influences_dropped\":%ld,\"dropped_weight_total\":%.4f,\"vertices_changed\":%ld,"
        "\"vertices_changed_material\":%ld,\"vertices_emptied\":%ld,"
        "\"vertices_inpainted\":%ld,\"vertices_restored\":%ld,\"max_dropped_weight\":%.6f,"
        "\"max_weight_delta\":%.6f}",
        key, s.V, s.J, s.diag, s.dom_p50, s.dom_p99, s.dom_p999, s.dom_max,
        s.dom_over25, s.dom_over35, s.dom_over50,
        s.bone_p50, s.bone_p99, s.bone_p999, s.bone_max,
        s.tension_p50, s.tension_p99, s.tension_p999, s.tension_max, s.tension_over, s.tension_verts,
        s.mean_influences, s.blended_frac, s.rigid_seam_edges,
        s.influences_dropped, s.dropped_weight_total, s.vertices_changed,
        s.vertices_changed_material, s.vertices_emptied,
        s.vertices_inpainted, s.vertices_restored, s.max_dropped_weight, s.max_weight_delta);
    return b;
}

std::string json_gate(const char* key, const rigqc::PoseGateResult& G, const rigqc::WeightHealth& H) {
    char b[1024];
    std::snprintf(b, sizeof(b),
        "\"%s\":{\"moved\":%.6f,\"p99\":%.6f,\"p995\":%.6f,\"p999\":%.6f,\"max_stretch\":%.6f,"
        "\"over5\":%.6f,\"over10\":%.6f,\"worst_audit\":%.6f,\"worst_component\":%.6f,"
        "\"influential\":%d,\"single_share\":%.6f,\"pose\":\"%s\"}",
        key, G.moved, G.p99, G.p995, G.p999, G.max_stretch, G.over5, G.over10,
        G.worst_audit_p999, G.worst_component_p999, H.influential, H.single_share,
        G.pose_label.c_str());
    return b;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "usage: %s <rigged.glb> [cleaned.glb] [--far-ratio F] [--far-frac-diag F] "
            "[--smooth-rounds N] [--smooth-blend F] [--inpaint-rounds N] [--dry-run] [--gate] "
            "[--json PATH]\n", argv[0]);
        return 2;
    }
    const char* in_path  = argv[1];
    const char* out_path = nullptr;
    const char* json_path = nullptr;
    bool dry = false, want_gate = false, unverified = false;
    double tolerance = 1.001;
    rigclean::Options opt;

    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--dry-run") dry = true;
        else if (a == "--gate") want_gate = true;
        else if (a == "--far-ratio" && i + 1 < argc) opt.far_ratio = (float)std::atof(argv[++i]);
        else if (a == "--far-frac-diag" && i + 1 < argc) opt.far_frac_diag = (float)std::atof(argv[++i]);
        else if (a == "--smooth-rounds" && i + 1 < argc) opt.smooth_rounds = std::atoi(argv[++i]);
        else if (a == "--smooth-blend" && i + 1 < argc) opt.smooth_blend = (float)std::atof(argv[++i]);
        else if (a == "--inpaint-rounds" && i + 1 < argc) opt.inpaint_rounds = std::atoi(argv[++i]);
        else if (a == "--min-weight" && i + 1 < argc) opt.min_weight = (float)std::atof(argv[++i]);
        else if (a == "--max-tension" && i + 1 < argc) opt.max_tension = (float)std::atof(argv[++i]);
        else if (a == "--tension-rounds" && i + 1 < argc) opt.tension_rounds = std::atoi(argv[++i]);
        else if (a == "--tension-blend" && i + 1 < argc) opt.tension_blend = (float)std::atof(argv[++i]);
        else if (a == "--unverified") unverified = true;
        else if (a == "--tolerance" && i + 1 < argc) tolerance = std::atof(argv[++i]);
        else if (a == "--json" && i + 1 < argc) json_path = argv[++i];
        else if (a.rfind("--", 0) == 0) { std::fprintf(stderr, "unknown flag %s\n", a.c_str()); return 2; }
        else if (!out_path) out_path = argv[i];
        else { std::fprintf(stderr, "unexpected argument %s\n", a.c_str()); return 2; }
    }
    if (!(opt.far_ratio >= 1.f) || !(opt.far_frac_diag > 0.f) || opt.smooth_rounds < 0 ||
        opt.smooth_rounds > 32 || !(opt.smooth_blend >= 0.f && opt.smooth_blend <= 1.f) ||
        opt.inpaint_rounds < 1 || opt.inpaint_rounds > 4096) {
        std::fprintf(stderr, "invalid cleanup options\n"); return 2;
    }
    if (!out_path) dry = true;

    rigio::SkinnedGlb g;
    std::string err;
    if (!rigio::load_skinned_glb(in_path, g, err)) {
        std::fprintf(stderr, "load %s: %s\n", in_path, err.c_str());
        return 2;
    }
    const double disagree = rigio::joint_bind_disagreement(g);
    if (!(disagree < 0.02)) {
        // The skeleton and the mesh are not in the same space (or the rig is not in bind pose), so
        // every distance below would be measured against the wrong skeleton. Refuse LOUDLY rather
        // than silently "cleaning" against nonsense.
        std::fprintf(stderr,
            "refusing: node-tree joint positions disagree with the inverse bind matrices by %.4f of "
            "the bbox diagonal — the rig is not in bind pose and distances would be meaningless\n",
            disagree);
        return 2;
    }

    rigqc::SkinnedRig R_before = g.rig;      // kept for the before/after gate print
    rigqc::SkinnedRig R_after  = g.rig;      // edited in place to the accepted rung

    rigclean::VerifiedResult vr;
    const auto t0 = std::chrono::steady_clock::now();
    if (unverified) {
        rigclean::Stats b, a;
        std::vector<int32_t> ji = g.rig.jidx;
        std::vector<float>   wv = g.rig.jw;
        if (!rigclean::clean_skin_weights(g.rig.vertices, g.rig.faces, g.joint_pos, g.rig.parent,
                                          ji, wv, opt, &b, &a)) {
            std::fprintf(stderr, "cleanup refused: malformed rig arrays\n");
            return 2;
        }
        R_after.jidx = ji; R_after.jw = wv;
        vr.rung = 3; vr.rung_name = "far-outlier + Lipschitz (UNVERIFIED)";
        vr.before = b; vr.after = a;
        vr.note = "--unverified: the pose-gate acceptance check was skipped";
    } else {
        if (!rigclean::clean_skin_weights_verified(R_after, g.joint_pos, opt, vr, tolerance)) {
            std::fprintf(stderr, "cleanup refused: malformed rig arrays\n");
            return 2;
        }
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const rigclean::Stats& before = vr.before;
    const rigclean::Stats& after  = vr.after;
    const std::vector<int32_t>& jidx = R_after.jidx;
    const std::vector<float>&   jw   = R_after.jw;

    std::printf("== skin-weight cleanup: %s ==\n", in_path);
    std::printf("  rule 1 (far outlier): drop influence when  d(v,bone_j) > %.2f x d(v,nearest bone)"
                "  AND  > %.3f x diag(%.4f) = %.4f\n",
                opt.far_ratio, opt.far_frac_diag, before.diag, opt.far_frac_diag * before.diag);
    std::printf("  rule 2 (Lipschitz):   smooth an edge's endpoints when tension > %.2f "
                "(<= ~%.1fx stretch at a 45deg audit rotation)\n",
                opt.max_tension, 1.0 + 0.785398 * opt.max_tension);
    std::printf("  %s\n", rigclean::stats_line("before", before).c_str());
    std::printf("  %s\n", rigclean::stats_line("after ", after).c_str());
    std::printf("  moved: %ld influences dropped (total weight %.1f, max single %.4f) across "
                "%ld/%d vertices (%.4f%%), of which %ld MATERIAL (L1>=0.05, %.4f%%); "
                "%ld emptied -> %ld inpainted, %ld restored; max per-vertex L1 delta %.4f; %.2fs\n",
                after.influences_dropped, after.dropped_weight_total, after.max_dropped_weight,
                after.vertices_changed, before.V,
                100.0 * (double)after.vertices_changed / std::max(1, before.V),
                after.vertices_changed_material,
                100.0 * (double)after.vertices_changed_material / std::max(1, before.V),
                after.vertices_emptied, after.vertices_inpainted, after.vertices_restored,
                after.max_weight_delta, secs);
    std::printf("  tension: %ld over-tension edges before -> %ld after; %ld vertices smoothed by "
                "rule 2\n", before.tension_over, after.tension_over, after.tension_verts);
    std::printf("  ACCEPTED rung %d (%s): pose gate default %.3f -> %.3f, all-influential %.3f -> "
                "%.3f (tolerance x%.2f)%s%s\n",
                vr.rung, vr.rung_name, vr.before_default, vr.after_default, vr.before_allinf,
                vr.after_allinf, tolerance, vr.note.empty() ? "" : "\n     ", vr.note.c_str());

    rigqc::PoseGateResult Gb, Ga, GbA, GaA;
    rigqc::WeightHealth   Hb, Ha;
    if (want_gate || json_path) {
        rigqc::PoseGateOpts d;
        rigqc::PoseGateOpts ai; ai.mode = rigqc::PoseGateOpts::GenericAllInfluential;
        Gb  = rigqc::run_pose_gate(R_before, d);
        Ga  = rigqc::run_pose_gate(R_after, d);
        GbA = rigqc::run_pose_gate(R_before, ai);
        GaA = rigqc::run_pose_gate(R_after, ai);
        Hb = rigqc::run_weight_health(R_before);
        Ha = rigqc::run_weight_health(R_after);
    }
    if (want_gate) {
        print_gate("before", R_before);
        print_gate("after ", R_after);
    }

    if (!dry) {
        if (!rigio::write_patched_glb(g, jidx, jw, out_path, err)) {
            std::fprintf(stderr, "write %s: %s\n", out_path, err.c_str());
            return 2;
        }
        std::printf("  wrote %s (skin bytes patched in place; everything else byte-identical)\n", out_path);
    } else {
        std::printf("  dry run — nothing written\n");
    }

    if (json_path) {
        FILE* jf = std::fopen(json_path, "w");
        if (!jf) { std::fprintf(stderr, "cannot write %s\n", json_path); return 2; }
        std::fprintf(jf, "{\"input\":\"%s\",\"far_ratio\":%.4f,\"far_frac_diag\":%.4f,"
                         "\"smooth_rounds\":%d,\"seconds\":%.3f,%s,%s,%s,%s}\n",
                     in_path, opt.far_ratio, opt.far_frac_diag, opt.smooth_rounds, secs,
                     json_stats("before", before).c_str(), json_stats("after", after).c_str(),
                     (json_gate("gate_before", Gb, Hb) + "," +
                      json_gate("gate_before_allinf", GbA, Hb)).c_str(),
                     (json_gate("gate_after", Ga, Ha) + "," +
                      json_gate("gate_after_allinf", GaA, Ha)).c_str());
        std::fclose(jf);
    }
    return 0;
}
