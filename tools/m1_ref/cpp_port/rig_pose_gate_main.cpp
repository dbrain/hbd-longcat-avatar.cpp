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
#include "glb_reader.hpp"     // JSON parser + accessor plumbing
#include "rig_pose_gate.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using glb::detail::JVal;

// ---------------------------------------------------------------------------
// Minimal skinned-GLB loader. Mirrors rig_pose_smoke.mesh_and_skin: pick the first node that has
// BOTH a mesh and a skin and whose primitive carries POSITION/JOINTS_0/WEIGHTS_0 + indices, then
// concatenate every identity-transform primitive sharing that same skin.
// ---------------------------------------------------------------------------
struct Glb {
    JVal                 root;
    std::vector<uint8_t> bin;
};

static bool load_glb(const char* path, Glb& out) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); return false; }
    std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> raw((size_t)std::max(0L, n));
    if (n <= 12 || std::fread(raw.data(), 1, (size_t)n, f) != (size_t)n) {
        std::fclose(f); std::fprintf(stderr, "short read\n"); return false;
    }
    std::fclose(f);
    uint32_t magic, ver, len;
    std::memcpy(&magic, &raw[0], 4); std::memcpy(&ver, &raw[4], 4); std::memcpy(&len, &raw[8], 4);
    if (magic != 0x46546C67u || ver != 2) { std::fprintf(stderr, "not a GLB v2\n"); return false; }
    size_t off = 12;
    bool have_json = false;
    while (off + 8 <= raw.size()) {
        uint32_t clen, ctype;
        std::memcpy(&clen, &raw[off], 4); std::memcpy(&ctype, &raw[off + 4], 4);
        off += 8;
        if (off + clen > raw.size()) break;
        if (ctype == 0x4E4F534Au) {
            glb::detail::JParser p((const char*)&raw[off], clen);
            if (!p.parse(out.root)) { std::fprintf(stderr, "bad JSON chunk\n"); return false; }
            have_json = true;
        } else if (ctype == 0x004E4942u) {
            out.bin.assign(raw.begin() + (long)off, raw.begin() + (long)(off + clen));
        }
        off += clen + ((4 - (clen & 3)) & 3);
    }
    return have_json;
}

// Read accessor `idx` into doubles (ncomp columns). Tight or strided; no sparse accessors.
static bool read_accessor(const Glb& g, int idx, std::vector<double>& out, int& ncomp, int64_t& count) {
    const JVal* accs = g.root.find("accessors");
    const JVal* bvs  = g.root.find("bufferViews");
    if (!accs || !accs->is_arr() || idx < 0 || idx >= (int)accs->arr.size()) return false;
    const JVal& a = accs->arr[(size_t)idx];
    const JVal* bvi = a.find("bufferView");
    if (!bvi || !bvs || !bvs->is_arr()) return false;
    const JVal& bv = bvs->arr[(size_t)bvi->as_int()];
    const int ct = (int)a.find("componentType")->as_int();
    const int cs = glb::detail::comp_size(ct);
    ncomp = glb::detail::type_count(a.find("type")->str);
    count = a.find("count")->as_int();
    if (cs == 0 || ncomp == 0) return false;
    size_t base = 0, stride = 0;
    if (const JVal* o = bv.find("byteOffset")) base += (size_t)o->as_int();
    if (const JVal* o = a.find("byteOffset"))  base += (size_t)o->as_int();
    if (const JVal* s = bv.find("byteStride")) stride = (size_t)s->as_int();
    if (stride == 0) stride = (size_t)cs * (size_t)ncomp;
    if (base + (size_t)(count - 1) * stride + (size_t)cs * ncomp > g.bin.size()) return false;
    out.resize((size_t)count * ncomp);
    for (int64_t i = 0; i < count; ++i) {
        const uint8_t* p = &g.bin[base + (size_t)i * stride];
        for (int c = 0; c < ncomp; ++c) {
            const uint8_t* q = p + (size_t)c * cs;
            double v = 0;
            switch (ct) {
                case 5120: { int8_t   x; std::memcpy(&x, q, 1); v = x; break; }
                case 5121: { uint8_t  x; std::memcpy(&x, q, 1); v = x; break; }
                case 5122: { int16_t  x; std::memcpy(&x, q, 2); v = x; break; }
                case 5123: { uint16_t x; std::memcpy(&x, q, 2); v = x; break; }
                case 5125: { uint32_t x; std::memcpy(&x, q, 4); v = x; break; }
                case 5126: { float    x; std::memcpy(&x, q, 4); v = x; break; }
                default: return false;
            }
            out[(size_t)i * ncomp + c] = v;
        }
    }
    return true;
}

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

    Glb g;
    if (!load_glb(argv[1], g)) return 2;
    const JVal* nodes = g.root.find("nodes");
    const JVal* meshes = g.root.find("meshes");
    const JVal* skins = g.root.find("skins");
    if (!nodes || !meshes || !skins || !nodes->is_arr() || !skins->is_arr()) {
        std::fprintf(stderr, "no nodes/meshes/skins\n"); return 2;
    }

    auto usable_prim = [&](const JVal& prim) {
        const JVal* at = prim.find("attributes");
        return at && at->find("POSITION") && at->find("JOINTS_0") && at->find("WEIGHTS_0") &&
               prim.find("indices");
    };

    int skin_i = -1;
    for (size_t ni = 0; ni < nodes->arr.size() && skin_i < 0; ++ni) {
        const JVal& nd = nodes->arr[ni];
        if (!nd.find("mesh") || !nd.find("skin")) continue;
        const JVal* prims = meshes->arr[(size_t)nd.find("mesh")->as_int()].find("primitives");
        if (!prims || !prims->is_arr()) continue;
        for (const JVal& p : prims->arr) if (usable_prim(p)) { skin_i = (int)nd.find("skin")->as_int(); break; }
    }
    if (skin_i < 0) { std::fprintf(stderr, "no indexed skinned primitive\n"); return 2; }

    rigqc::SkinnedRig R;
    int32_t vbase = 0;
    for (size_t ni = 0; ni < nodes->arr.size(); ++ni) {
        const JVal& nd = nodes->arr[ni];
        if (!nd.find("skin") || (int)nd.find("skin")->as_int() != skin_i || !nd.find("mesh")) continue;
        if (nd.find("matrix") || nd.find("translation") || nd.find("rotation") || nd.find("scale")) {
            std::fprintf(stderr, "skinned mesh node carries a transform — unsupported (as in the Python)\n");
            return 2;
        }
        const JVal* prims = meshes->arr[(size_t)nd.find("mesh")->as_int()].find("primitives");
        for (const JVal& p : prims->arr) {
            if (!usable_prim(p)) continue;
            const JVal* at = p.find("attributes");
            std::vector<double> pos, jj, ww, idx;
            int nc; int64_t cnt;
            if (!read_accessor(g, (int)at->find("POSITION")->as_int(), pos, nc, cnt) || nc != 3) return 2;
            const int64_t nv = cnt;
            for (int64_t i = 0; i < nv * 3; ++i) R.vertices.push_back((float)pos[(size_t)i]);
            if (!read_accessor(g, (int)at->find("JOINTS_0")->as_int(), jj, nc, cnt) || nc != 4) return 2;
            for (int64_t i = 0; i < nv * 4; ++i) R.jidx.push_back((int32_t)jj[(size_t)i]);
            if (!read_accessor(g, (int)at->find("WEIGHTS_0")->as_int(), ww, nc, cnt) || nc != 4) return 2;
            for (int64_t i = 0; i < nv * 4; ++i) R.jw.push_back((float)ww[(size_t)i]);
            if (!read_accessor(g, (int)p.find("indices")->as_int(), idx, nc, cnt) || nc != 1) return 2;
            for (int64_t i = 0; i < cnt; ++i) R.faces.push_back((int32_t)idx[(size_t)i] + vbase);
            vbase += (int32_t)nv;
        }
    }

    // Skeleton: skin.joints (node indices) -> joint-space local/ibm/parent.
    const JVal& sk = skins->arr[(size_t)skin_i];
    const JVal* jl = sk.find("joints");
    if (!jl || !jl->is_arr()) { std::fprintf(stderr, "skin has no joints\n"); return 2; }
    const int J = (int)jl->arr.size();
    std::vector<int> jnode((size_t)J);
    for (int j = 0; j < J; ++j) jnode[(size_t)j] = (int)jl->arr[(size_t)j].as_int();

    // node parents
    std::vector<int> nparent(nodes->arr.size(), -1);
    for (size_t ni = 0; ni < nodes->arr.size(); ++ni)
        if (const JVal* ch = nodes->arr[ni].find("children"))
            for (const JVal& c : ch->arr) nparent[(size_t)c.as_int()] = (int)ni;
    std::vector<int> node_to_joint(nodes->arr.size(), -1);
    for (int j = 0; j < J; ++j) node_to_joint[(size_t)jnode[(size_t)j]] = j;

    // node local matrices, exactly as the Python builds them: `matrix` transposed, else TRANSLATION
    // ONLY (rig_pose_smoke ignores node rotation/scale on the skeleton; our writer emits neither).
    auto node_local = [&](int n) {
        rigqc::M4 m = rigqc::m4_identity();
        const JVal& nd = nodes->arr[(size_t)n];
        if (const JVal* mm = nd.find("matrix")) {
            for (int c = 0; c < 4; ++c) for (int r = 0; r < 4; ++r)
                m[(size_t)r * 4 + c] = (float)mm->arr[(size_t)(c * 4 + r)].num;   // col-major -> row-major
        } else if (const JVal* t = nd.find("translation")) {
            m[3] = (float)t->arr[0].num; m[7] = (float)t->arr[1].num; m[11] = (float)t->arr[2].num;
        }
        return m;
    };

    R.parent.assign((size_t)J, -1);
    R.local.assign((size_t)J, rigqc::m4_identity());
    for (int j = 0; j < J; ++j) {
        // Walk up to the nearest ancestor that is itself a joint, accumulating the chain of
        // intermediate node transforms into this joint's effective local.
        rigqc::M4 acc = node_local(jnode[(size_t)j]);
        int cur = nparent[(size_t)jnode[(size_t)j]];
        while (cur >= 0 && node_to_joint[(size_t)cur] < 0) {
            acc = rigqc::m4_mul(node_local(cur), acc);
            cur = nparent[(size_t)cur];
        }
        R.parent[(size_t)j] = (cur >= 0) ? node_to_joint[(size_t)cur] : -1;
        R.local[(size_t)j] = acc;
    }

    std::vector<double> ibm;
    int nc; int64_t cnt;
    R.ibm.assign((size_t)J, rigqc::m4_identity());
    if (const JVal* ibmi = sk.find("inverseBindMatrices")) {
        if (!read_accessor(g, (int)ibmi->as_int(), ibm, nc, cnt) || nc != 16 || cnt != J) {
            std::fprintf(stderr, "bad inverseBindMatrices\n"); return 2;
        }
        for (int j = 0; j < J; ++j)
            for (int c = 0; c < 4; ++c) for (int r = 0; r < 4; ++r)
                R.ibm[(size_t)j][(size_t)r * 4 + c] = (float)ibm[(size_t)j * 16 + (size_t)(c * 4 + r)];
    }

    R.names.assign((size_t)J, std::string());
    for (int j = 0; j < J; ++j)
        if (const JVal* nm = nodes->arr[(size_t)jnode[(size_t)j]].find("name")) R.names[(size_t)j] = nm->str;

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
    std::printf("native V=%d J=%d max_disp=%.4g\n", R.V(), J, G.max_disp);
    return G.pass ? 0 : 1;
}
