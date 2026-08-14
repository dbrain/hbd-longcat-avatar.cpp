// rig_pose_gate_main.cpp — run the NATIVE pose gate (rig_pose_gate.hpp) on a rigged GLB, and print
// rig_pose_smoke.py's `pose gate:` line character for character.
//
// This binary exists to FALSIFY the port: run it and the Python side by side on the same asset and
// diff the two lines. It is not part of the pipeline — image_to_rig calls the header directly on
// the arrays it already holds.
//
//   ./rig_pose_gate <rigged.glb> [--generic-all-influential | --generic-extremity |
//                                 --generic-joint N] [--weight-health]
//
// Exit code mirrors the Python: 0 = gate passed, 1 = gate failed, 2 = could not run.
//
// The skinned-GLB loader used to live inline here. It now lives in rig_glb_skin_io.hpp, shared with
// rig_weight_cleanup — the tool that EDITS a skin and the instrument that JUDGES it must agree on
// what the asset is, and two copies of that rule would drift. The move was verified line-for-line
// against this binary's own output on all eight campaign assets.
#include "rig_glb_skin_io.hpp"
#include "rig_pose_gate.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "usage: %s <rigged.glb> [--generic-all-influential | --generic-extremity | "
            "--generic-joint N] [--weight-health]\n", argv[0]);
        return 2;
    }
    rigqc::PoseGateOpts opt;
    bool want_health = false;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--generic-all-influential") opt.mode = rigqc::PoseGateOpts::GenericAllInfluential;
        else if (a == "--generic-extremity")  opt.mode = rigqc::PoseGateOpts::GenericExtremity;
        else if (a == "--generic-joint" && i + 1 < argc) {
            opt.mode = rigqc::PoseGateOpts::GenericJoint; opt.requested_joint = std::atoi(argv[++i]);
        } else if (a == "--weight-health") want_health = true;
        else { std::fprintf(stderr, "unknown flag %s\n", a.c_str()); return 2; }
    }

    rigio::SkinnedGlb g;
    std::string err;
    if (!rigio::load_skinned_glb(argv[1], g, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 2; }
    rigqc::SkinnedRig& R = g.rig;

    if (want_health) {
        rigqc::WeightHealth H = rigqc::run_weight_health(R);
        std::printf("%s J=%d V=%d influential=%d (>= 8%% of peak) mass>=1%%=%d dominant>=1%%verts=%d "
                    "biggest_joint_share=%.1f%%\n",
                    argv[1], H.J, H.V, H.influential, H.mass_1pct, H.dominant_1pct, H.single_share * 100.0);
        if (H.review)
            std::printf("  REVIEW: one joint holds %.1f%% of all skin mass. Legitimate for a large "
                        "rigid prop (hat, helmet, shell) — confirm on the pose-gate render before "
                        "treating it as a defect.\n", H.single_share * 100.0);
        if (!H.pass)
            std::printf("  WEIGHT-HEALTH FAIL: only %d/%d joints carry >=8%% of peak mass (need >= 4) "
                        "— nothing meaningful articulates and the pose gate would audit almost nothing\n",
                        H.influential, H.J);
        else
            std::printf("  weight health: PASS\n");
    }

    rigqc::PoseGateResult G = rigqc::run_pose_gate(R, opt);
    if (!G.ok) { std::fprintf(stderr, "pose gate could not run: %s\n", G.err.c_str()); return 2; }
    std::printf("%s\n", rigqc::pose_gate_line(G).c_str());
    std::printf("native V=%d J=%d max_disp=%.4g\n", R.V(), R.J(), G.max_disp);
    return G.pass ? 0 : 1;
}
