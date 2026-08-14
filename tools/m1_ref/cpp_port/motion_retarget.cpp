// motion_retarget.cpp — rigged GLB + motion clip  ->  animated GLB, natively.
//
// The C++ replacement for anim/tools/build_motion_ab.py: it loads each rigged delivery, maps its
// bones to SMPL-22 (names first, topology walk as the fallback), decides the L/R swap by
// measurement, retargets every motion in every variant, and bakes the lot into one animated GLB
// per subject.  No Python, no subprocess, no GPU.
//
//   ./motion_retarget --out DIR [--subjects a,b] [--smooth 2] [--fps 20]
//   ./motion_retarget --out DIR --verify /path/to/python/motion_ab_20260813
//   ./motion_retarget --roundtrip                 # the invariant: rest motion -> our own rest
//   ./motion_retarget --diff-glb A.glb B.glb      # per-channel float32 comparison of two bakes
//   ./motion_retarget --bake-smpl-rest            # regenerate smpl_rest_joints.hpp from the .npz
//   ./motion_retarget --check-smpl-rest           # baked table vs the .npz, bit-for-bit
//
// VARIANTS (ids are the Python ones, so the eye page and the manifest keep working):
//   fix      positional retarget, facing decided by measurement
//   nofix    the other L/R labelling — the A/B that shows the mirror
//   base     positional + retarget base pose on the arms (rest_align arm=1)
//   rot      ROTATION-NATIVE, rest_align 0 — isolates bridge-vs-native and nothing else
//   rotbase  rotation-native + base pose — THE DEFAULT, the combination that must be right
// `rot`/`rotbase` only exist for a rotation-carrying source; a positional clip has nothing to
// offer them and they are skipped, exactly as in the prototype.
#include "motion_glb_anim.hpp"
#include "motion_retarget.hpp"
#include "rig_exercise.hpp"      // --exercise: the native rig_exercise_anim.py
#include "smpl_rest_joints.hpp"   // the 22x3 constants, so the delivery path needs no SMPL npz

#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <string>
#include <sys/stat.h>
#include <vector>

using namespace mret;

// ---------------------------------------------------------------------------
// paths (the same trees build_motion_ab.py reads)
// ---------------------------------------------------------------------------
static const char* SMPLX_NPZ =
    "/mnt/hdd/3d/avatar-shootout/EMAGE/emage_evaltools/smplx_models/smplx/SMPLX_NEUTRAL_2020.npz";
static const char* NPY_DIR = "/mnt/hdd/3d/avatar-shootout/web/c1/clips";
static const char* HYM_DIR = "/mnt/hdd/3d/avatar-shootout/_shootout_out/hymotion_20260813";
static const char* NEW_TREE = "/mnt/hdd/3d/avatar-shootout/_shootout_out/onmaster_20260813";
static const char* OLD_TREE = "/mnt/hdd/3d/avatar-shootout/_shootout_out/final_e2e_20260725";

struct Subject { const char* name; std::string glb; const char* note; };

// soldier comes from the OLD tree on purpose: the 2026-08-13 soldier rig has no 4-joint leg
// chain, so the namer correctly refuses to name anything (0/22) and there is no facing cue.
// soldier_proto is the anonymous-bone_N July prototype: it exercises the TOPOLOGY FALLBACK
// end to end and is the rig every number in the July report was measured on.
static std::vector<Subject> default_subjects() {
    return {
        {"miku", std::string(NEW_TREE) + "/miku/rigged.glb", "onmaster_20260813 · named 21/22 (no LeftToeBase)"},
        {"char1", std::string(NEW_TREE) + "/char1/rigged.glb", "onmaster_20260813 · named 22/22"},
        {"gilly", std::string(NEW_TREE) + "/gilly/rigged.glb", "onmaster_20260813 · named 19/22 (left leg is a stump)"},
        {"soldier", std::string(OLD_TREE) + "/soldier/rigged.glb",
         "final_e2e_20260725 · named 18/22 — the 08-13 soldier is 0/22"},
        {"soldier_proto", "/home/dbrain/dev/puppy-eyetest/anim/soldier.glb",
         "proj_v1 prototype rig · 0/22 named — TOPOLOGY FALLBACK, 21 slots, both arms"},
    };
}
static const char* MOMASK_MOTIONS[] = {"walk", "dance", "cheer", "kick", "jump_joy", "wave", "clap", "bow"};

// the variant table now lives in motion_retarget.hpp (mret::variants), so the shipped inline
// pipeline and this A/B harness cannot disagree about what "rotbase" means.
struct MotionSrc {
    std::string name, path, source;
    bool        hymo = false;
    double      fps = 20;
};

static bool exists(const std::string& p) { struct stat st; return ::stat(p.c_str(), &st) == 0; }
static void mkdirs(const std::string& p) {
    std::string cur;
    for (size_t i = 0; i < p.size(); i++) {
        cur.push_back(p[i]);
        if (p[i] == '/' || i + 1 == p.size()) ::mkdir(cur.c_str(), 0755);
    }
}

// numpy's np.round: half-to-even at the tie.  Used so "does my clip JSON match Python's" is a
// crisp equality test rather than a threshold argument.
static double round5(double v) {
    const double s = v * 1e5;
    double r = std::nearbyint(s);          // default FE_TONEAREST == half-to-even
    return r / 1e5;
}
// Angle between two orientations, in degrees.
//
// NOT 2*acos(|dot|): near-identical quaternions put dot at 1-delta, where acos' derivative is
// infinite, so the answer degrades to 2*sqrt(2*delta) and the STORAGE quantisation of whatever
// you are comparing against dominates the number.  Concretely, a reference rounded to 5 dp reads
// as ~0.5 deg of "deviation" and a float32 one as ~0.04 deg — entirely artefact.  The half-angle
// tangent form below is well-conditioned at 0, which is the only place this metric is used.
// Both sides are normalised first because a 5-dp-rounded quaternion is not unit.
static double quat_angle_deg(const Q4& a0, const Q4& b0) {
    const Q4 a = q_norm(a0);
    Q4 b = q_norm(b0);
    if (a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w < 0) b = Q4{-b.x, -b.y, -b.z, -b.w};  // double cover
    const double dm = std::sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) +
                                (a.z - b.z) * (a.z - b.z) + (a.w - b.w) * (a.w - b.w));
    const double dp = std::sqrt((a.x + b.x) * (a.x + b.x) + (a.y + b.y) * (a.y + b.y) +
                                (a.z + b.z) * (a.z + b.z) + (a.w + b.w) * (a.w + b.w));
    return 2.0 * std::atan2(dm, dp) * 180.0 / M_PI;
}
static double quat_comp_maxabs(const Q4& a, const Q4& b) {
    Q4 c = b;
    if (a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w < 0) c = Q4{-b.x, -b.y, -b.z, -b.w};
    return std::max(std::max(std::fabs(a.x - c.x), std::fabs(a.y - c.y)),
                    std::max(std::fabs(a.z - c.z), std::fabs(a.w - c.w)));
}

// ---------------------------------------------------------------------------
// verification against the Python reference
// ---------------------------------------------------------------------------
struct DevStat {
    double max_deg = 0, sum_deg = 0, max_comp = 0;
    long   n = 0, ncomp_mismatch = 0, ncomp = 0;
    void add(double d) { if (d > max_deg) max_deg = d; sum_deg += d; n++; }
    double mean() const { return n ? sum_deg / n : 0.0; }
};

// Compare an in-memory clip against Python's clips/<anim>.json (quats rounded to 5 dp).
static bool verify_clip_json(const std::string& path, const Clip& c, DevStat& st, std::string& err) {
    std::vector<uint8_t> buf;
    if (!read_file(path.c_str(), buf)) { err = "missing " + path; return false; }
    glb::detail::JVal d;
    glb::detail::JParser p((const char*)buf.data(), buf.size());
    if (!p.parse(d)) { err = "parse failed " + path; return false; }
    const auto* q = d.find("quats");
    const auto* bn = d.find("bones");
    if (!q || !q->is_arr() || (int)q->arr.size() != c.T) { err = "frame count differs in " + path; return false; }
    if (!bn || (int)bn->arr.size() != c.N) { err = "bone count differs in " + path; return false; }
    for (int t = 0; t < c.T; t++)
        for (int b = 0; b < c.N; b++) {
            const auto& e = q->arr[(size_t)t].arr[(size_t)b];
            const Q4 ref{e.arr[0].as_double(), e.arr[1].as_double(), e.arr[2].as_double(), e.arr[3].as_double()};
            const Q4& mine = c.quats[(size_t)t * c.N + (size_t)b];
            st.add(quat_angle_deg(mine, ref));
            st.max_comp = std::max(st.max_comp, quat_comp_maxabs(mine, ref));
            const double mr[4] = {round5(mine.x), round5(mine.y), round5(mine.z), round5(mine.w)};
            const double rr[4] = {ref.x, ref.y, ref.z, ref.w};
            for (int k = 0; k < 4; k++) { st.ncomp++; if (mr[k] != rr[k]) st.ncomp_mismatch++; }
        }
    return true;
}

// Per-channel float32 comparison of two baked GLBs — the tightest available resolution (~1e-5 deg)
// because it compares the exact bytes a viewer will play, not a 5-dp text round.
static bool read_anims(const char* path, std::map<std::string, std::map<int, std::vector<Q4>>>& out,
                       std::string& err) {
    std::vector<uint8_t> file, bin;
    glb::detail::JVal g;
    if (!glb_split(path, file, g, bin, err)) return false;
    const auto* accs = g.find("accessors");
    const auto* bvs = g.find("bufferViews");
    const auto* anims = g.find("animations");
    if (!anims || !anims->is_arr()) { err = std::string("no animations in ") + path; return false; }
    auto read_vec4 = [&](int ai, std::vector<Q4>& q) {
        const auto& a = accs->arr[(size_t)ai];
        const int bv = (int)a.find("bufferView")->as_int();
        const int64_t cnt = a.find("count")->as_int();
        const auto& v = bvs->arr[(size_t)bv];
        size_t off = (size_t)(v.find("byteOffset") ? v.find("byteOffset")->as_int() : 0);
        if (const auto* ao = a.find("byteOffset")) off += (size_t)ao->as_int();
        q.resize((size_t)cnt);
        for (int64_t i = 0; i < cnt; i++) {
            float f[4];
            std::memcpy(f, &bin[off + (size_t)i * 16], 16);
            q[(size_t)i] = Q4{f[0], f[1], f[2], f[3]};
        }
    };
    for (const auto& an : anims->arr) {
        const std::string nm = an.find("name") ? an.find("name")->str : "";
        const auto* chs = an.find("channels");
        const auto* sms = an.find("samplers");
        for (const auto& ch : chs->arr) {
            const auto* tg = ch.find("target");
            if (!tg || !tg->find("path") || tg->find("path")->str != "rotation") continue;
            const int node = (int)tg->find("node")->as_int();
            const int si = (int)ch.find("sampler")->as_int();
            const int oa = (int)sms->arr[(size_t)si].find("output")->as_int();
            read_vec4(oa, out[nm][node]);
        }
    }
    return true;
}

static int diff_glbs(const char* a, const char* b) {
    std::map<std::string, std::map<int, std::vector<Q4>>> A, B;
    std::string err;
    if (!read_anims(a, A, err) || !read_anims(b, B, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 2; }
    double gmax = 0, gsum = 0, gcmax = 0;
    long gn = 0, gbit = 0;
    int missing = 0;
    bool verbose = ::getenv("MRET_DIFF_VERBOSE") != nullptr;
    std::string worst;
    for (auto& an : A) {
        auto it = B.find(an.first);
        if (it == B.end()) { std::printf("  !! %s missing on the other side\n", an.first.c_str()); missing++; continue; }
        double amax = 0, asum = 0, acmax = 0;
        long an_n = 0, abit = 0;
        for (auto& chn : an.second) {
            auto jt = it->second.find(chn.first);
            if (jt == it->second.end()) { std::printf("  !! %s node %d channel missing\n", an.first.c_str(), chn.first); missing++; continue; }
            if (jt->second.size() != chn.second.size()) { std::printf("  !! %s node %d length differs\n", an.first.c_str(), chn.first); missing++; continue; }
            for (size_t t = 0; t < chn.second.size(); t++) {
                const Q4& x = chn.second[t];
                const Q4& y = jt->second[t];
                amax = std::max(amax, quat_angle_deg(x, y));
                asum += quat_angle_deg(x, y);
                acmax = std::max(acmax, quat_comp_maxabs(x, y));
                if (x.x == y.x && x.y == y.y && x.z == y.z && x.w == y.w) abit++;
                an_n++;
            }
        }
        gsum += asum; gn += an_n; gbit += abit;
        gcmax = std::max(gcmax, acmax);
        if (amax > gmax) { gmax = amax; worst = an.first; }
        if (verbose)
            std::printf("  %-28s channels=%2d  max=%.4g deg  mean=%.4g deg  max|dq|=%.3g  bit-identical %ld/%ld\n",
                        an.first.c_str(), (int)an.second.size(), amax, an_n ? asum / an_n : 0.0, acmax, abit, an_n);
    }
    for (auto& bn : B) if (!A.count(bn.first)) { std::printf("  !! %s only on the other side\n", bn.first.c_str()); missing++; }
    std::printf("  GLB TOTAL  animations=%d  keys=%ld  max=%.4g deg (%s)  mean=%.4g deg  max|dq|=%.3g  "
                "bit-identical %ld/%ld  missing=%d\n",
                (int)A.size(), gn, gmax, worst.c_str(), gn ? gsum / gn : 0.0, gcmax, gbit, gn, missing);
    return missing ? 1 : 0;
}

// ---------------------------------------------------------------------------
// --bake-smpl-rest / --check-smpl-rest
//
// The ONLY thing the animation path ever wanted from SMPLX_NEUTRAL_2020.npz (167 MB) is
// smpl_rest_joints(): 22 joint positions, 528 bytes.  Bake them into a header so a delivery
// machine needs no SMPL weights tree at all, and keep a checker so the bake can be proved
// bit-for-bit equal to the file it came from rather than merely "close".
// ---------------------------------------------------------------------------
static int smpl_rest_tool(bool emit_header) {
    std::vector<V3> j;
    std::string err;
    if (!smpl_rest_joints(SMPLX_NPZ, j, err)) {
        std::fprintf(stderr, "cannot read %s: %s\n", SMPLX_NPZ, err.c_str());
        return 2;
    }
    if (emit_header) {
        std::printf(
            "// smpl_rest_joints.hpp — SMPL-H 22 REST (\"zero-pose\") joint positions, GENERATED.\n"
            "//\n"
            "// PROVENANCE\n"
            "//   file   %s\n"
            "//   quantity  J = J_regressor @ v_template, first 22 rows, pelvis translated to the\n"
            "//             origin.  float32 on disk, widened to float64 before the product — exactly\n"
            "//             `d['J_regressor'].astype(np.float64) @ d['v_template'].astype(np.float64)`.\n"
            "//   emitted with %%.17g, i.e. the shortest form that round-trips to the same double.\n"
            "//\n"
            "// WHY IT IS BAKED.  The retarget's only use of the 167 MB npz was these 528 bytes, so a\n"
            "// delivery box was carrying an entire SMPL-X weights tree to compute a constant.  The\n"
            "// numbers are a property of the MODEL, not of a run: nothing here is a tuned value.\n"
            "//\n"
            "// REGENERATE / FALSIFY (both need the npz; neither is on the delivery path)\n"
            "//   ./motion_retarget --bake-smpl-rest > smpl_rest_joints.hpp\n"
            "//   ./motion_retarget --check-smpl-rest      # bit-for-bit vs the npz, exit 0 = identical\n"
            "// The checker is the acceptance test: `identical 66/66` means the baked path and the\n"
            "// npz path cannot diverge, so the retarget stays bit-exact.\n"
            "#pragma once\n"
            "\n"
            "namespace mret {\n"
            "\n"
            "// [22][3], metres, pelvis at the origin.  Order = SMPL-H 22 (rig::kSmplNames).\n"
            "inline const double (*smpl_rest_joints_baked())[3] {\n"
            "    static const double J[22][3] = {\n",
            SMPLX_NPZ);
        for (int i = 0; i < 22; i++)
            std::printf("        {%.17g, %.17g, %.17g},   // %2d %s\n", j[(size_t)i].x, j[(size_t)i].y,
                        j[(size_t)i].z, i, rig::kSmplNames[i]);
        std::printf("    };\n"
                    "    return J;\n"
                    "}\n"
                    "\n"
                    "// The shape retarget_delta / decide_lr_swap want.\n"
                    "inline std::vector<V3> smpl_rest_joints_baked_v3() {\n"
                    "    const double (*J)[3] = smpl_rest_joints_baked();\n"
                    "    std::vector<V3> out(22);\n"
                    "    for (int i = 0; i < 22; i++) out[(size_t)i] = V3{J[i][0], J[i][1], J[i][2]};\n"
                    "    return out;\n"
                    "}\n"
                    "\n"
                    "}  // namespace mret\n");
        return 0;
    }
    const std::vector<V3> b = smpl_rest_joints_baked_v3();
    int same = 0, diff = 0;
    double worst = 0;
    for (int i = 0; i < 22; i++)
        for (int k = 0; k < 3; k++) {
            const double a = j[(size_t)i][k], c = b[(size_t)i][k];
            if (a == c) same++;
            else { diff++; worst = std::max(worst, std::fabs(a - c)); }
        }
    std::printf("smpl rest joints: baked vs %s -> identical %d/66, differing %d (max |d| = %.3g)  %s\n",
                SMPLX_NPZ, same, diff, worst, diff ? "FAIL" : "PASS");
    return diff ? 1 : 0;
}

// ---------------------------------------------------------------------------
// the roundtrip invariant
// ---------------------------------------------------------------------------
static int roundtrip(const std::vector<V3>& src_rest, const std::vector<Subject>& subs) {
    // Feed the SOURCE'S OWN REST POSE in as the motion.  Every D is identity, so every local quat
    // must come back EXACTLY equal to the rig's own bind local.  It is necessary, not sufficient
    // (it cannot see a rest MISMATCH — at source-rest eq (1) returns Grest for any theta), which
    // is precisely why rest_align exists and why this is only half the acceptance test.
    int fails = 0;
    for (const auto& s : subs) {
        if (!exists(s.glb)) { std::printf("  %-14s SKIP (missing)\n", s.name); continue; }
        Prep p;
        std::string err;
        if (!prepare(s.glb, src_rest, p, err)) { std::printf("  %-14s FAIL prepare: %s\n", s.name, err.c_str()); fails++; continue; }
        std::vector<V3> rest = src_rest;
        if (p.swap.swap) lr_swap_inplace(rest, 1);
        const int T = 3;
        std::vector<V3> motion((size_t)T * 22);
        for (int t = 0; t < T; t++) for (int j = 0; j < 22; j++) motion[(size_t)t * 22 + (size_t)j] = rest[(size_t)j];
        for (int mode = 0; mode < 2; mode++) {           // 0 = positional, 1 = rotation-native
            std::vector<Q4> rot;
            if (mode == 1) rot.assign((size_t)T * 22, q_identity());
            Clip c;
            RestAlign ra;
            if (!retarget_delta(p.rig, p.map, motion, rest, 30, false, ra, mode ? &rot : nullptr, c, err)) {
                std::printf("  %-14s FAIL: %s\n", s.name, err.c_str()); fails++; continue;
            }
            double worst = 0;
            for (int t = 0; t < c.T; t++)
                for (int b = 0; b < c.N; b++)
                    worst = std::max(worst, quat_angle_deg(c.quats[(size_t)t * c.N + (size_t)b],
                                                           q_norm(p.rig.Lr[(size_t)c.bones[(size_t)b]])));
            std::printf("  %-14s %-16s worst |q - q_bind| = %.6f deg   %s\n", s.name,
                        mode ? "rotation-native" : "positional", worst, worst < 1e-6 ? "PASS" : "FAIL");
            if (!(worst < 1e-6)) fails++;
        }
    }
    return fails;
}

// ---------------------------------------------------------------------------
// SINGLE-SHOT: one rig + N motion clips -> one animated GLB.
//
// This is the shape the end-to-end chain needs — `image_to_rig` hands over a rigged.glb, the
// hymotion binary hands over raw smplh22 clips, and this turns the pair into the deliverable.
// The A/B harness above exists to prove this path is right; this is the path that ships.
// ---------------------------------------------------------------------------
static int single_shot(const std::string& rig_path, const std::vector<std::string>& motions,
                       const std::string& out_glb, const std::vector<std::string>& variants,
                       int smooth, double npy_fps, const std::vector<V3>& src_rest0) {
    Prep p;
    std::string err;
    if (!prepare(rig_path, src_rest0, p, err)) { std::fprintf(stderr, "%s: %s\n", rig_path.c_str(), err.c_str()); return 1; }
    std::printf("%s: map=%s mapped=%d/22 swap=%s  rest mismatch arm=%.1f leg=%.1f spine=%.1f deg\n",
                rig_path.c_str(), p.map.which.c_str(), p.map.mapped(), p.swap.swap ? "on" : "off",
                p.rest_arm_mean, p.rest_leg_mean, p.rest_spine_mean);

    std::vector<BakedClip> baked;
    for (const auto& mp : motions) {
        const bool is_json = mp.size() > 5 && mp.compare(mp.size() - 5, 5, ".json") == 0;
        std::vector<V3> pos;
        std::vector<Q4> rot;
        double mfps = npy_fps;
        std::string base = mp.substr(mp.find_last_of('/') + 1);
        base = base.substr(0, base.find_last_of('.'));
        if (is_json) {
            HymoClip hc;
            if (!load_hymo_json(mp.c_str(), hc, err)) { std::fprintf(stderr, "%s: %s\n", mp.c_str(), err.c_str()); return 1; }
            mfps = hc.fps;                       // fps is carried PER CLIP, never resampled
            pos = hymo_positions(hc, src_rest0);
            rot = smpl_global_rots(hc);
        } else {
            NpyArray a;
            if (!npy_load(mp.c_str(), a, err)) { std::fprintf(stderr, "%s: %s\n", mp.c_str(), err.c_str()); return 1; }
            const int64_t J = a.shape.size() >= 3 ? a.shape[1] : (a.shape[1] / 3), T = a.shape[0];
            pos.resize((size_t)T * 22);
            for (int64_t t = 0; t < T; t++)
                for (int64_t j = 0; j < 22; j++)
                    pos[(size_t)(t * 22 + j)] = V3{a.data[(size_t)((t * J + j) * 3 + 0)],
                                                   a.data[(size_t)((t * J + j) * 3 + 1)],
                                                   a.data[(size_t)((t * J + j) * 3 + 2)]};
        }
        const int T = (int)(pos.size() / 22);
        for (const auto& vid : variants) {
            const Variant* v = mret::find_variant(vid);
            if (!v) { std::fprintf(stderr, "unknown variant '%s'\n", vid.c_str()); return 2; }
            if (v->rot && rot.empty()) {
                std::fprintf(stderr, "%s: '%s' needs a rotation-carrying source; %s has positions only\n",
                             base.c_str(), vid.c_str(), mp.c_str());
                return 1;
            }
            const bool swap = (vid == "nofix") ? !p.swap.swap : p.swap.swap;
            std::vector<V3> mm = pos, rest = src_rest0;
            std::vector<Q4> mr = rot;
            if (swap) { lr_swap_inplace(mm, T); lr_swap_inplace(rest, 1); if (!mr.empty()) lr_swap_inplace(mr, T); }
            RestAlign ra;
            ra.a[GRP_ARM] = v->arm_alpha;
            Clip c;
            if (!retarget_delta(p.rig, p.map, mm, rest, mfps, false, ra, v->rot ? &mr : nullptr, c, err)) {
                std::fprintf(stderr, "%s__%s: %s\n", base.c_str(), vid.c_str(), err.c_str()); return 1;
            }
            smooth_quats(c, smooth);
            BakedClip bc;
            bc.name = (variants.size() == 1) ? base : base + "__" + vid;
            bc.bones = c.bones; bc.fps = mfps; bc.T = c.T; bc.N = c.N; bc.quats = c.quats;
            baked.push_back(std::move(bc));
            std::printf("  %-30s %d frames @ %g fps\n", baked.back().name.c_str(), c.T, mfps);
        }
    }
    const int n = bake_animations(p.rig, baked, out_glb.c_str(), err);
    if (n < 0) { std::fprintf(stderr, "bake failed: %s\n", err.c_str()); return 1; }
    std::printf("wrote %s (%d animations)\n", out_glb.c_str(), n);
    return 0;
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    std::string out, verify, subjects_filter, diffA, diffB, rig_path, out_glb, smpl_npz, exA, exB;
    rigex::Options exopt;
    std::vector<std::string> one_motions, one_variants;
    int smooth = 2;
    double momask_fps = 20;
    bool do_roundtrip = false, bake_smpl_rest = false, check_smpl_rest = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--out" && i + 1 < argc) out = argv[++i];
        else if (a == "--verify" && i + 1 < argc) verify = argv[++i];
        else if (a == "--subjects" && i + 1 < argc) subjects_filter = argv[++i];
        else if (a == "--smooth" && i + 1 < argc) smooth = std::atoi(argv[++i]);
        else if (a == "--fps" && i + 1 < argc) momask_fps = std::atof(argv[++i]);
        else if (a == "--roundtrip") do_roundtrip = true;
        else if (a == "--smpl-npz" && i + 1 < argc) smpl_npz = argv[++i];
        else if (a == "--exercise" && i + 2 < argc) { exA = argv[++i]; exB = argv[++i]; }
        else if (a == "--degrees" && i + 1 < argc) exopt.degrees = std::atof(argv[++i]);
        else if (a == "--slot" && i + 1 < argc) exopt.slot = std::atof(argv[++i]);
        else if (a == "--stride" && i + 1 < argc) exopt.stride = std::atof(argv[++i]);
        else if (a == "--max-joints" && i + 1 < argc) exopt.max_joints = std::atoi(argv[++i]);
        else if (a == "--bake-smpl-rest") bake_smpl_rest = true;
        else if (a == "--check-smpl-rest") check_smpl_rest = true;
        else if (a == "--diff-glb" && i + 2 < argc) { diffA = argv[++i]; diffB = argv[++i]; }
        else if (a == "--rig" && i + 1 < argc) rig_path = argv[++i];
        else if (a == "--motion" && i + 1 < argc) one_motions.push_back(argv[++i]);
        else if (a == "--out-glb" && i + 1 < argc) out_glb = argv[++i];
        else if (a == "--variant" && i + 1 < argc) {
            std::string v = argv[++i];
            for (size_t s = 0, e; s <= v.size(); s = e + 1) {
                e = v.find(',', s);
                if (e == std::string::npos) e = v.size();
                if (e > s) one_variants.push_back(v.substr(s, e - s));
            }
        }
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    if (!diffA.empty()) return diff_glbs(diffA.c_str(), diffB.c_str());
    if (bake_smpl_rest || check_smpl_rest) return smpl_rest_tool(bake_smpl_rest);
    if (!exA.empty()) {
        rigex::Result R;
        std::string e;
        if (!rigex::bake_exercise(exA.c_str(), exB.c_str(), exopt, R, e)) {
            std::fprintf(stderr, "%s: %s\n", exA.c_str(), e.c_str());
            return 2;
        }
        std::printf("%s: exercised %d influential joints, %.1fs clip, +-%g deg\n", exB.c_str(),
                    R.joints, R.duration, exopt.degrees);
        return 0;
    }

    // The source rest pose is a CONSTANT of the SMPL model, so it is baked (smpl_rest_joints.hpp)
    // and the 167 MB npz is not on the delivery path at all.  `--smpl-npz PATH` reads the file
    // instead, which is the A/B behind `--check-smpl-rest` (today: identical 66/66).
    std::string err;
    std::vector<V3> src_rest0 = smpl_rest_joints_baked_v3();
    if (!smpl_npz.empty() && !smpl_rest_joints(smpl_npz.c_str(), src_rest0, err)) {
        std::fprintf(stderr, "source rest unavailable: %s\n", err.c_str());
        return 2;
    }

    if (!rig_path.empty()) {
        if (one_motions.empty() || out_glb.empty()) {
            std::fprintf(stderr, "usage: %s --rig R.glb --motion M.json [--motion ...] --out-glb OUT.glb "
                                 "[--variant rotbase] [--smooth 2] [--fps 20]\n", argv[0]);
            return 2;
        }
        if (one_variants.empty()) one_variants.push_back("rotbase");   // the owner's chosen default
        return single_shot(rig_path, one_motions, out_glb, one_variants, smooth, momask_fps, src_rest0);
    }

    std::vector<Subject> subs = default_subjects();
    if (!subjects_filter.empty()) {
        std::vector<Subject> keep;
        for (auto& s : subs)
            if (("," + subjects_filter + ",").find("," + std::string(s.name) + ",") != std::string::npos) keep.push_back(s);
        subs.swap(keep);
    }
    if (do_roundtrip) {
        std::printf("=== roundtrip identity (source rest in -> our own rest out) ===\n");
        const int f = roundtrip(src_rest0, subs);
        std::printf("roundtrip: %s\n", f ? "FAILED" : "ALL PASS");
        if (out.empty()) return f;
    }
    if (out.empty()) { std::fprintf(stderr, "usage: %s --out DIR [--verify REFDIR] [--subjects a,b] [--roundtrip]\n", argv[0]); return 2; }

    // ---- motion list: MoMask .npy first, then raw HY-Motion (fps carried PER CLIP) ----
    std::vector<MotionSrc> motions;
    for (const char* m : MOMASK_MOTIONS) {
        std::string p = std::string(NPY_DIR) + "/_" + m + ".npy";
        if (exists(p)) motions.push_back({m, p, "MoMask/HumanML3D", false, momask_fps});
    }
    {
        // readdir, not `ls` through popen — a port whose whole point is "no subprocess" should
        // not shell out to enumerate a directory.
        std::vector<std::string> files;
        if (DIR* dp = ::opendir(HYM_DIR)) {
            while (struct dirent* de = ::readdir(dp)) {
                std::string n = de->d_name;
                if (n.size() > 5 && n.compare(n.size() - 5, 5, ".json") == 0)
                    files.push_back(std::string(HYM_DIR) + "/" + n);
            }
            ::closedir(dp);
        }
        std::sort(files.begin(), files.end());
        for (auto& f : files) {
            HymoClip hc;
            std::string e;
            if (!load_hymo_json(f.c_str(), hc, e)) continue;    // not a raw smplh22 clip: skip
            std::string base = f.substr(f.find_last_of('/') + 1);
            base = base.substr(0, base.size() - 5);
            motions.push_back({base, f, "HY-Motion 2026-08-13 (raw, rotation-native)", true, hc.fps});
        }
    }
    std::printf("motions:");
    for (auto& m : motions) std::printf(" %s(%s,%gfps)", m.name.c_str(), m.hymo ? "hymo" : "npy", m.fps);
    std::printf("\n");

    mkdirs(out);
    std::string manifest = "{\n \"built\": \"cpp_port\",\n \"fps\": " + jnum(momask_fps) + ",\n \"motions\": [\n";
    for (size_t i = 0; i < motions.size(); i++) {
        manifest += "  {\"name\": \"" + motions[i].name + "\", \"source\": \"" + motions[i].source +
                    "\", \"fps\": " + jnum(motions[i].fps) + ", \"kind\": \"" + (motions[i].hymo ? "hymo" : "npy") + "\"}";
        manifest += (i + 1 < motions.size()) ? ",\n" : "\n";
    }
    manifest += " ],\n \"subjects\": [\n";

    DevStat grand;
    int verify_files = 0, verify_missing = 0, map_mismatch = 0;
    bool first_subject = true;

    for (const auto& sub : subs) {
        if (!exists(sub.glb)) { std::printf("!! %s: %s missing, skipped\n", sub.name, sub.glb.c_str()); continue; }
        std::printf("==============================================================================\n");
        std::printf("%s  %s\n", sub.name, sub.glb.c_str());
        Prep p;
        if (!prepare(sub.glb, src_rest0, p, err)) { std::fprintf(stderr, "  FAIL: %s\n", err.c_str()); return 1; }
        std::printf("  map=%s  mapped=%d/22  swap=%s (%s, rig %s / src %s)\n", p.map.which.c_str(), p.map.mapped(),
                    p.swap.swap ? "True" : "False", p.swap.decided ? "measured" : p.swap.reason.c_str(),
                    p.swap.rig_cue.c_str(), p.swap.src_cue.c_str());
        std::printf("  topology walk: %s\n", p.topo_ok ? "ok" : ("FAIL — " + p.topo_err).c_str());
        if (p.named_ok && p.topo_ok) {
            std::map<int, int> ns, ts;
            for (auto& kv : p.named.b2s) ns[kv.second] = kv.first;
            for (auto& kv : p.topo.b2s) ts[kv.second] = kv.first;
            std::string differ;
            int common = 0;
            for (int s = 0; s < 22; s++) {
                if (ns.count(s) && ts.count(s)) common++;
                const int a = ns.count(s) ? ns[s] : -1, b = ts.count(s) ? ts[s] : -1;
                if (a != b) { differ += std::string(differ.empty() ? "" : ",") + rig::kSmplNames[s]; }
            }
            std::printf("  named vs topology: %d joints in common, differ on %s\n", common,
                        differ.empty() ? "nothing" : differ.c_str());
        }
        std::printf("  rest mismatch (deg, mean/max): arm=%.1f/%.1f leg=%.1f spine=%.1f\n",
                    p.rest_arm_mean, p.rest_arm_max, p.rest_leg_mean, p.rest_spine_mean);
        // load_rig composes rest globals from translation+rotation only. Neither branch is
        // exercised by this fleet, but a silent wrong rest is the worst failure mode there is.
        if (p.rig.saw_scale) std::printf("  !! SCALED joint node — the rest walk ignores scale (prototype parity)\n");
        if (p.rig.saw_matrix) std::printf("  !! matrix-transform node — rest taken via the matrix branch\n");

        const std::string subdir = out + "/" + sub.name;
        const std::string clipdir = subdir + "/clips";
        mkdirs(clipdir);
        if (!write_alias_skeleton(p.rig, (subdir + "/rig_alias.glb").c_str(), err))
            std::fprintf(stderr, "  alias skeleton: %s\n", err.c_str());

        // map.json — the exact correspondence, so a checker can validate the MAPPER independently
        // of the RETARGETER.
        {
            std::string j = "{\n \"b2s\": {";
            bool f1 = true;
            for (auto& kv : p.map.b2s) { j += (f1 ? "" : ", "); f1 = false; j += "\"" + std::to_string(kv.first) + "\": " + std::to_string(kv.second); }
            j += "},\n \"aim\": {";
            f1 = true;
            for (auto& kv : p.map.aim) { j += (f1 ? "" : ", "); f1 = false; j += "\"" + std::to_string(kv.first) + "\": " + (kv.second < 0 ? "null" : std::to_string(kv.second)); }
            j += "},\n \"orig_names\": {";
            f1 = true;
            for (int n : p.rig.joints) { j += (f1 ? "" : ", "); f1 = false; std::string e2; jesc(p.rig.orig[(size_t)n], e2); j += "\"" + std::to_string(n) + "\": " + e2; }
            j += "},\n \"which\": \"" + p.map.which + "\",\n \"swap\": " + (p.swap.swap ? "true" : "false") + "\n}\n";
            FILE* fh = std::fopen((subdir + "/map.json").c_str(), "w");
            if (fh) { std::fwrite(j.data(), 1, j.size(), fh); std::fclose(fh); }
        }

        // ---- cross-check the MAPPER against Python's map.json, independently of the retarget ----
        if (!verify.empty()) {
            std::vector<uint8_t> buf;
            const std::string rp = verify + "/" + sub.name + "/map.json";
            if (read_file(rp.c_str(), buf)) {
                glb::detail::JVal d;
                glb::detail::JParser jp((const char*)buf.data(), buf.size());
                if (jp.parse(d)) {
                    int diff = 0;
                    const auto* rb = d.find("b2s");
                    for (auto& kv : rb->obj) {
                        const int node = std::atoi(kv.first.c_str());
                        auto it = p.map.b2s.find(node);
                        if (it == p.map.b2s.end() || it->second != (int)kv.second.as_int()) diff++;
                    }
                    for (auto& kv : p.map.b2s) if (!rb->obj.count(std::to_string(kv.first))) diff++;
                    const auto* ra = d.find("aim");
                    for (auto& kv : ra->obj) {
                        const int node = std::atoi(kv.first.c_str());
                        const int want = kv.second.type == glb::detail::JVal::Null ? -1 : (int)kv.second.as_int();
                        auto it = p.map.aim.find(node);
                        if (it == p.map.aim.end() || it->second != want) diff++;
                    }
                    const bool wsw = d.find("swap") && d.find("swap")->b;
                    const std::string wwh = d.find("which") ? d.find("which")->str : "";
                    std::printf("  map.json vs Python: %s  (which %s/%s, swap %d/%d, %d entries differ)\n",
                                (diff == 0 && wsw == p.swap.swap && wwh == p.map.which) ? "IDENTICAL" : "DIFFERS",
                                p.map.which.c_str(), wwh.c_str(), (int)p.swap.swap, (int)wsw, diff);
                    if (diff || wsw != p.swap.swap || wwh != p.map.which) map_mismatch++;
                }
            } else {
                std::printf("  map.json vs Python: reference missing\n");
            }
        }

        std::vector<BakedClip> baked;
        DevStat subj;
        for (const auto& mo : motions) {
            std::vector<V3> pos;
            std::vector<Q4> rot;
            double mfps = mo.fps;
            if (mo.hymo) {
                HymoClip hc;
                if (!load_hymo_json(mo.path.c_str(), hc, err)) { std::fprintf(stderr, "  %s: %s\n", mo.name.c_str(), err.c_str()); continue; }
                mfps = hc.fps;
                pos = hymo_positions(hc, src_rest0);       // the legacy float32 position bridge
                rot = smpl_global_rots(hc);                // the thing the bridge throws away
            } else {
                NpyArray a;
                if (!npy_load(mo.path.c_str(), a, err)) { std::fprintf(stderr, "  %s: %s\n", mo.name.c_str(), err.c_str()); continue; }
                const int64_t J = a.shape.size() >= 3 ? a.shape[1] : (a.shape[1] / 3);
                const int64_t T = a.shape[0];
                pos.resize((size_t)T * 22);
                for (int64_t t = 0; t < T; t++)
                    for (int64_t j = 0; j < 22; j++)
                        pos[(size_t)(t * 22 + j)] = V3{a.data[(size_t)((t * J + j) * 3 + 0)],
                                                       a.data[(size_t)((t * J + j) * 3 + 1)],
                                                       a.data[(size_t)((t * J + j) * 3 + 2)]};
            }
            const int T = (int)(pos.size() / 22);
            int NV = 0;
            const Variant* VA = mret::variants(NV);
            for (int vi = 0; vi < NV; vi++) {
                const Variant& v = VA[vi];
                if (v.hymo_only && !mo.hymo) continue;
                const bool swap = (std::strcmp(v.id, "nofix") == 0) ? !p.swap.swap : p.swap.swap;
                std::vector<V3> mm = pos, rest = src_rest0;
                std::vector<Q4> mr = rot;
                if (swap) {
                    // relabel motion AND source rest identically — lr_swap indexes the JOINT axis,
                    // so it works unchanged on rotations.
                    lr_swap_inplace(mm, T);
                    lr_swap_inplace(rest, 1);
                    if (!mr.empty()) lr_swap_inplace(mr, T);
                }
                RestAlign ra;
                ra.a[GRP_ARM] = v.arm_alpha;
                Clip c;
                if (!retarget_delta(p.rig, p.map, mm, rest, mfps, false, ra, v.rot ? &mr : nullptr, c, err)) {
                    std::fprintf(stderr, "  %s__%s: %s\n", mo.name.c_str(), v.id, err.c_str());
                    continue;
                }
                smooth_quats(c, smooth);
                const std::string anim = mo.name + "__" + v.id;

                // clip JSON (5 dp, as the prototype writes it)
                {
                    std::string j = "{\"name\": \"" + anim + "\", \"fps\": " + jnum(mfps) + ", \"bones\": [";
                    for (int b = 0; b < c.N; b++) j += (b ? ", " : "") + std::string("\"bone_") + std::to_string(c.bones[(size_t)b]) + "\"";
                    j += "], \"quats\": [";
                    for (int t = 0; t < c.T; t++) {
                        j += (t ? ", [" : "[");
                        for (int b = 0; b < c.N; b++) {
                            const Q4& q = c.quats[(size_t)t * c.N + (size_t)b];
                            j += (b ? ", [" : "[");
                            j += jnum(round5(q.x)) + ", " + jnum(round5(q.y)) + ", " + jnum(round5(q.z)) + ", " + jnum(round5(q.w)) + "]";
                        }
                        j += "]";
                    }
                    j += "], \"root_pos\": [";
                    for (int t = 0; t < c.T; t++) {
                        const V3& r = c.root_pos[(size_t)t];
                        j += (t ? ", [" : "[");
                        j += jnum(round5(r.x)) + ", " + jnum(round5(r.y)) + ", " + jnum(round5(r.z)) + "]";
                    }
                    j += "]}";
                    FILE* fh = std::fopen((clipdir + "/" + anim + ".json").c_str(), "w");
                    if (fh) { std::fwrite(j.data(), 1, j.size(), fh); std::fclose(fh); }
                }

                if (!verify.empty()) {
                    DevStat st;
                    const std::string rp = verify + "/" + sub.name + "/clips/" + anim + ".json";
                    std::string e2;
                    if (verify_clip_json(rp, c, st, e2)) {
                        verify_files++;
                        subj.max_deg = std::max(subj.max_deg, st.max_deg);
                        subj.sum_deg += st.sum_deg; subj.n += st.n;
                        subj.ncomp += st.ncomp; subj.ncomp_mismatch += st.ncomp_mismatch;
                        grand.max_deg = std::max(grand.max_deg, st.max_deg);
                        grand.sum_deg += st.sum_deg; grand.n += st.n;
                        grand.ncomp += st.ncomp; grand.ncomp_mismatch += st.ncomp_mismatch;
                        subj.max_comp = std::max(subj.max_comp, st.max_comp);
                        grand.max_comp = std::max(grand.max_comp, st.max_comp);
                        if (st.max_deg > 1e-2 || st.ncomp_mismatch)
                            std::printf("    !! %-28s max=%.3e deg mean=%.3e deg  5dp-mismatch %ld/%ld\n",
                                        anim.c_str(), st.max_deg, st.mean(), st.ncomp_mismatch, st.ncomp);
                    } else {
                        verify_missing++;
                        std::printf("    %-28s VERIFY: %s\n", anim.c_str(), e2.c_str());
                    }
                }

                BakedClip bc;
                bc.name = anim; bc.bones = c.bones; bc.fps = mfps; bc.T = c.T; bc.N = c.N; bc.quats = c.quats;
                baked.push_back(std::move(bc));
            }
        }
        const int n = bake_animations(p.rig, baked, (subdir + "/anim.glb").c_str(), err);
        if (n < 0) { std::fprintf(stderr, "  bake failed: %s\n", err.c_str()); return 1; }
        std::printf("  baked %d animations into %s/anim.glb\n", n, subdir.c_str());
        if (!verify.empty() && subj.n)
            std::printf("  SUBJECT %s vs Python clips/*.json: max=%.3e deg  mean=%.3e deg  max|dq|=%.3e  "
                        "5dp-identical %ld/%ld components\n",
                        sub.name, subj.max_deg, subj.mean(), subj.max_comp,
                        subj.ncomp - subj.ncomp_mismatch, subj.ncomp);

        manifest += std::string(first_subject ? "" : ",\n") + "  {\"name\": \"" + sub.name + "\", \"note\": \"" + sub.note +
                    "\", \"glb\": \"" + sub.name + "/anim.glb\", \"rig_alias\": \"" + sub.name + "/rig_alias.glb\"" +
                    ", \"source_glb\": \"" + sub.glb + "\", \"joints\": " + std::to_string(p.rig.joints.size()) +
                    ", \"mapped\": " + std::to_string(p.map.mapped()) + ", \"map\": \"" + p.map.which +
                    "\", \"named_ok\": " + (p.named_ok ? "true" : "false") + ", \"topo_ok\": " + (p.topo_ok ? "true" : "false") +
                    ", \"swap\": " + (p.swap.swap ? "true" : "false") + ", \"animations\": [";
        for (size_t i = 0; i < baked.size(); i++) manifest += (i ? ", " : "") + std::string("\"") + baked[i].name + "\"";
        manifest += "]}";
        first_subject = false;

        // finest-resolution check: our baked float32 channels vs Python's
        if (!verify.empty()) {
            const std::string ra = verify + "/" + sub.name + "/anim.glb";
            if (exists(ra)) {
                std::printf("  --- baked GLB channel diff (float32, the bytes a viewer plays) ---\n");
                diff_glbs((subdir + "/anim.glb").c_str(), ra.c_str());
            }
        }
    }
    manifest += "\n ]\n}\n";
    { FILE* fh = std::fopen((out + "/manifest.json").c_str(), "w");
      if (fh) { std::fwrite(manifest.data(), 1, manifest.size(), fh); std::fclose(fh); } }
    std::printf("\nmanifest: %s/manifest.json\n", (out).c_str());
    if (!verify.empty())
        std::printf("VERIFY TOTAL: %d clips compared, %d missing, map mismatches %d, "
                    "max=%.3e deg mean=%.3e deg max|dq|=%.3e, 5dp-identical %ld/%ld components\n",
                    verify_files, verify_missing, map_mismatch, grand.max_deg, grand.mean(),
                    grand.max_comp, grand.ncomp - grand.ncomp_mismatch, grand.ncomp);
    return 0;
}
