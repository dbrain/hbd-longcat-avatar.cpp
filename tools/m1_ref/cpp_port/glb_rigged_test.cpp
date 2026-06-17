// CPU self-test for the rigged-GLB writer (glb_rigged.hpp).
//   1) Load the SkinTokens goldens from /tmp/skintokens_e2e/ (tiny self-contained .npy reader:
//      C-order float32 '<f4' and int64 '<i8' only), rig the 8192 sampled points + 60 joints,
//      write /tmp/rigged_miku.glb, then independently re-parse the GLB to assert the skin
//      encoding (JOINTS_0/WEIGHTS_0 counts == V, weight rows sum ~1, skins[0].joints==60,
//      node tree matches `parents`, inverseBindMatrices count==60, world-joint reconstruction).
//   2) Real-mesh test: read native_v7.glb via glb::read_glb, rig with the golden 60 joints +
//      synthetic nearest-joint weights, write /tmp/rigged_real.glb, assert it parses.
// No ggml, no CUDA. Build: ./build.sh glb_rigged_test
#include "glb_reader.hpp"   // glb::read_glb + the JSON parser (detail::JParser/JVal)
#include "glb_rigged.hpp"   // glb::write_rigged_glb
#include "glb_writer.hpp"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>

// ---------------------------------------------------------------------------
// Tiny .npy reader: '\x93NUMPY' magic, header dict for shape + dtype. Only the
// two dtypes we need ('<f4' C-order float32, '<i8' C-order int64). Returns the
// flat data; shape returned via `shape`.
// ---------------------------------------------------------------------------
static bool npy_header(FILE* f, std::string& dtype, std::vector<int64_t>& shape, size_t& data_off) {
    unsigned char magic[8];
    if (std::fread(magic, 1, 8, f) != 8) return false;
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) return false;
    int ver_major = magic[6];
    uint32_t hlen;
    if (ver_major == 1) {
        uint16_t h16; if (std::fread(&h16, 2, 1, f) != 1) return false; hlen = h16; data_off = 10 + hlen;
    } else {
        uint32_t h32; if (std::fread(&h32, 4, 1, f) != 1) return false; hlen = h32; data_off = 12 + hlen;
    }
    std::string hdr(hlen, '\0');
    if (std::fread(&hdr[0], 1, hlen, f) != hlen) return false;
    // descr
    auto dpos = hdr.find("'descr'");
    if (dpos == std::string::npos) return false;
    auto q1 = hdr.find('\'', hdr.find(':', dpos) + 1);
    auto q2 = hdr.find('\'', q1 + 1);
    dtype = hdr.substr(q1 + 1, q2 - q1 - 1);
    // shape
    auto spos = hdr.find("'shape'");
    auto p1 = hdr.find('(', spos);
    auto p2 = hdr.find(')', p1);
    std::string sh = hdr.substr(p1 + 1, p2 - p1 - 1);
    shape.clear();
    size_t i = 0;
    while (i < sh.size()) {
        while (i < sh.size() && (sh[i] == ' ' || sh[i] == ',')) i++;
        if (i >= sh.size()) break;
        char* end = nullptr;
        long v = std::strtol(sh.c_str() + i, &end, 10);
        if (end == sh.c_str() + i) break;
        shape.push_back((int64_t)v);
        i = (size_t)(end - sh.c_str());
    }
    return true;
}

static bool load_npy_f32(const char* path, std::vector<float>& out, std::vector<int64_t>& shape) {
    FILE* f = std::fopen(path, "rb"); if (!f) return false;
    std::string dtype; size_t off;
    if (!npy_header(f, dtype, shape, off)) { std::fclose(f); return false; }
    if (dtype != "<f4") { std::fprintf(stderr, "npy %s: dtype %s != <f4\n", path, dtype.c_str()); std::fclose(f); return false; }
    size_t n = 1; for (int64_t d : shape) n *= (size_t)d;
    out.resize(n);
    std::fseek(f, (long)off, SEEK_SET);
    bool ok = std::fread(out.data(), 4, n, f) == n;
    std::fclose(f);
    return ok;
}

static bool load_npy_i64(const char* path, std::vector<int64_t>& out, std::vector<int64_t>& shape) {
    FILE* f = std::fopen(path, "rb"); if (!f) return false;
    std::string dtype; size_t off;
    if (!npy_header(f, dtype, shape, off)) { std::fclose(f); return false; }
    if (dtype != "<i8") { std::fprintf(stderr, "npy %s: dtype %s != <i8\n", path, dtype.c_str()); std::fclose(f); return false; }
    size_t n = 1; for (int64_t d : shape) n *= (size_t)d;
    out.resize(n);
    std::fseek(f, (long)off, SEEK_SET);
    bool ok = std::fread(out.data(), 8, n, f) == n;
    std::fclose(f);
    return ok;
}

// ---------------------------------------------------------------------------
// Independent GLB skin re-parser (uses glb_reader's JSON parser + the BIN blob)
// to verify accessor counts, weight-row sums, skins, node tree, IBM.
// ---------------------------------------------------------------------------
struct RigCheck {
    int64_t joints0_count = -1, weights0_count = -1, ibm_count = -1, skin_joints = -1;
    double  max_weight_sum_err = 0.0;
    bool    node_tree_ok = false;
    bool    parsed = false;
};

static bool read_file(const char* path, std::vector<uint8_t>& buf) {
    FILE* f = std::fopen(path, "rb"); if (!f) return false;
    std::fseek(f, 0, SEEK_END); long sz = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    buf.resize((size_t)sz);
    bool ok = std::fread(buf.data(), 1, (size_t)sz, f) == (size_t)sz;
    std::fclose(f);
    return ok;
}

static RigCheck inspect_rigged_glb(const char* path, const std::vector<int>& parents_expected) {
    using namespace glb::detail;
    RigCheck rc;
    std::vector<uint8_t> buf;
    if (!read_file(path, buf) || buf.size() < 12) return rc;
    auto rd32 = [&](size_t o) { uint32_t v; std::memcpy(&v, buf.data() + o, 4); return v; };
    uint32_t total = rd32(8);
    const char* json_ptr = nullptr; size_t json_len = 0;
    const uint8_t* bin_ptr = nullptr; size_t bin_len = 0;
    size_t off = 12;
    while (off + 8 <= total) {
        uint32_t clen = rd32(off), ctype = rd32(off + 4); size_t cs = off + 8;
        if (ctype == 0x4E4F534Au) { json_ptr = (const char*)(buf.data() + cs); json_len = clen; }
        else if (ctype == 0x004E4942u) { bin_ptr = buf.data() + cs; bin_len = clen; }
        off = cs + clen;
    }
    if (!json_ptr || !bin_ptr) return rc;
    JVal root;
    if (!JParser(json_ptr, json_len).parse(root) || !root.is_obj()) return rc;

    const JVal* accessors = root.find("accessors");
    const JVal* bufviews  = root.find("bufferViews");
    const JVal* meshes    = root.find("meshes");
    const JVal* skins     = root.find("skins");
    const JVal* nodes     = root.find("nodes");
    if (!accessors || !bufviews || !meshes || !skins || !nodes) return rc;

    // locate the skinned primitive's attribute accessor indices
    int joints0_acc = -1, weights0_acc = -1;
    const JVal* prims = meshes->arr[0].find("primitives");
    const JVal* attrs = prims->arr[0].find("attributes");
    if (const JVal* a = attrs->find("JOINTS_0"))  joints0_acc  = (int)a->as_int(-1);
    if (const JVal* a = attrs->find("WEIGHTS_0")) weights0_acc = (int)a->as_int(-1);

    auto acc_count = [&](int i) -> int64_t {
        if (i < 0 || i >= (int)accessors->arr.size()) return -1;
        const JVal* c = accessors->arr[i].find("count"); return c ? c->as_int(-1) : -1;
    };
    rc.joints0_count  = acc_count(joints0_acc);
    rc.weights0_count = acc_count(weights0_acc);

    // skins[0]: joints array length + inverseBindMatrices accessor count
    const JVal& sk = skins->arr[0];
    if (const JVal* jr = sk.find("joints"); jr && jr->is_arr()) rc.skin_joints = (int64_t)jr->arr.size();
    if (const JVal* ibmj = sk.find("inverseBindMatrices")) rc.ibm_count = acc_count((int)ibmj->as_int(-1));

    // WEIGHTS_0 row sums: resolve the accessor -> bufferView -> BIN, read VEC4 f32.
    if (weights0_acc >= 0) {
        const JVal& acc = accessors->arr[weights0_acc];
        int64_t bv = acc.find("bufferView")->as_int(-1);
        int64_t count = acc.find("count")->as_int(0);
        const JVal& view = bufviews->arr[bv];
        int64_t bvoff = view.find("byteOffset") ? view.find("byteOffset")->as_int(0) : 0;
        int64_t accoff = acc.find("byteOffset") ? acc.find("byteOffset")->as_int(0) : 0;
        const uint8_t* base = bin_ptr + bvoff + accoff;
        for (int64_t v = 0; v < count; v++) {
            float w[4];
            std::memcpy(w, base + v * 16, 16);
            double s = (double)w[0] + w[1] + w[2] + w[3];
            double e = std::fabs(s - 1.0);
            if (e > rc.max_weight_sum_err) rc.max_weight_sum_err = e;
        }
        (void)bin_len;
    }

    // node tree consistency: derive parent[] from each node's children[] and compare to expected.
    size_t J = parents_expected.size();
    std::vector<int> derived(J, -1);
    bool ok_tree = nodes->arr.size() >= J;  // there should be J joint nodes + 1 mesh node
    if (ok_tree) {
        for (size_t n = 0; n < J; n++) {
            const JVal* ch = nodes->arr[n].find("children");
            if (ch && ch->is_arr())
                for (const JVal& c : ch->arr) {
                    int ci = (int)c.as_int(-1);
                    if (ci >= 0 && ci < (int)J) derived[ci] = (int)n;
                }
        }
        for (size_t j = 0; j < J; j++) if (derived[j] != parents_expected[j]) { ok_tree = false; break; }
    }
    rc.node_tree_ok = ok_tree;
    rc.parsed = true;
    return rc;
}

static long file_size(const char* p) {
    FILE* f = std::fopen(p, "rb"); if (!f) return -1;
    std::fseek(f, 0, SEEK_END); long s = std::ftell(f); std::fclose(f); return s;
}

int main() {
    const char* DIR = "/tmp/skintokens_e2e";
    int fails = 0;
    auto check = [&](const char* name, bool ok) {
        std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", name);
        if (!ok) fails++;
    };

    // ===================== GOLDEN TEST =====================
    std::printf("== golden rig test (%s) ==\n", DIR);
    std::vector<float> verts, normals, skin;
    std::vector<int64_t> jointsd, sh;
    std::vector<int64_t> parents_i64;
    if (!load_npy_f32((std::string(DIR) + "/vertices.npy").c_str(), verts, sh)) { std::fprintf(stderr, "load vertices failed\n"); return 1; }
    uint32_t V = (uint32_t)(verts.size() / 3);
    load_npy_f32((std::string(DIR) + "/normals.npy").c_str(), normals, sh);
    if (!load_npy_f32((std::string(DIR) + "/skin_pred.npy").c_str(), skin, sh)) { std::fprintf(stderr, "load skin_pred failed\n"); return 1; }
    uint32_t J = (uint32_t)sh[1];
    std::vector<float> joints;
    if (!load_npy_f32((std::string(DIR) + "/detok_joints.npy").c_str(), joints, sh)) { std::fprintf(stderr, "load joints failed\n"); return 1; }
    if (!load_npy_i64((std::string(DIR) + "/detok_parents.npy").c_str(), parents_i64, sh)) { std::fprintf(stderr, "load parents failed\n"); return 1; }
    std::vector<int> parents(parents_i64.begin(), parents_i64.end());

    std::printf("  V=%u  J=%u  (skin %ux%u)\n", V, J, V, J);

    // self-check: local->world reconstruction error (recompute world from locals + parents)
    std::vector<float> locals, ibm;
    bool rig_ok = glb::rig_local_and_ibm(joints, parents, locals, ibm);
    check("rig_local_and_ibm (forest valid)", rig_ok);
    double max_recon_err = 0.0;
    if (rig_ok) {
        std::vector<float> world(J * 3, 0.f);
        // parents are topologically ordered enough (each parent index < child for this tree,
        // except verify by iterating to fixpoint). Use a parent-first walk via chain sum.
        for (uint32_t j = 0; j < J; j++) {
            // accumulate up the chain
            float acc[3] = {0, 0, 0};
            int cur = (int)j;
            int guard = 0;
            while (cur != -1 && guard++ <= (int)J) {
                acc[0] += locals[cur*3+0]; acc[1] += locals[cur*3+1]; acc[2] += locals[cur*3+2];
                cur = parents[cur];
            }
            world[j*3+0] = acc[0]; world[j*3+1] = acc[1]; world[j*3+2] = acc[2];
        }
        for (uint32_t k = 0; k < J * 3; k++) {
            double e = std::fabs((double)world[k] - joints[k]);
            if (e > max_recon_err) max_recon_err = e;
        }
    }
    std::printf("  world-joint reconstruction max err = %.3e\n", max_recon_err);
    check("local->world reconstruction matches input joints (<1e-4)", max_recon_err < 1e-4);

    // write the rigged GLB
    std::vector<int64_t> faces;  // sampled point set -> no faces (degenerate); writer must accept this
    const char* OUT_MIKU = "/tmp/rigged_miku.glb";
    bool wrote = glb::write_rigged_glb(OUT_MIKU, verts, faces, joints, parents, skin,
                                       normals.size() == (size_t)V * 3 ? &normals : nullptr);
    check("write_rigged_glb (golden)", wrote);

    // re-read via glb::read_glb for a POSITION-count sanity check
    glb::Mesh rm;
    bool rr = glb::read_glb(OUT_MIKU, rm);
    check("glb::read_glb re-parse (golden)", rr);
    check("read_glb POSITION count == V", rr && (uint32_t)(rm.verts.size() / 3) == V);

    // independent skin re-parse
    RigCheck rc = inspect_rigged_glb(OUT_MIKU, parents);
    check("independent re-parse of rigged GLB", rc.parsed);
    std::printf("  JOINTS_0 count=%lld  WEIGHTS_0 count=%lld  IBM count=%lld  skin.joints=%lld\n",
                (long long)rc.joints0_count, (long long)rc.weights0_count,
                (long long)rc.ibm_count, (long long)rc.skin_joints);
    std::printf("  max weight-row-sum error = %.3e\n", rc.max_weight_sum_err);
    check("JOINTS_0 count == V",  rc.joints0_count  == (int64_t)V);
    check("WEIGHTS_0 count == V", rc.weights0_count == (int64_t)V);
    check("inverseBindMatrices count == J(60)", rc.ibm_count == (int64_t)J);
    check("skins[0].joints == J(60)", rc.skin_joints == (int64_t)J);
    check("weight rows sum ~ 1.0 (err < 1e-4)", rc.max_weight_sum_err < 1e-4);
    check("node tree parent/children consistent with parents[]", rc.node_tree_ok);

    long miku_sz = file_size(OUT_MIKU);
    std::printf("  wrote %s (%ld bytes)\n", OUT_MIKU, miku_sz);

    // ===================== REAL-MESH TEST =====================
    std::printf("== real-mesh rig test (native_v7.glb) ==\n");
    glb::Mesh real;
    bool real_read = glb::read_glb("native_v7.glb", real);
    check("read native_v7.glb", real_read);
    if (real_read) {
        uint32_t RV = (uint32_t)(real.verts.size() / 3);
        std::printf("  real mesh: V=%u  F=%zu\n", RV, real.faces.size() / 3);
        // synthetic weights: each vertex fully weighted to nearest golden joint.
        std::vector<float> rweights((size_t)RV * J, 0.f);
        for (uint32_t v = 0; v < RV; v++) {
            const float* p = &real.verts[(size_t)v * 3];
            int best = 0; float bd = 1e30f;
            for (uint32_t j = 0; j < J; j++) {
                float dx = p[0]-joints[j*3+0], dy = p[1]-joints[j*3+1], dz = p[2]-joints[j*3+2];
                float d = dx*dx + dy*dy + dz*dz;
                if (d < bd) { bd = d; best = (int)j; }
            }
            rweights[(size_t)v * J + best] = 1.f;
        }
        const char* OUT_REAL = "/tmp/rigged_real.glb";
        bool rw = glb::write_rigged_glb(OUT_REAL, real.verts, real.faces, joints, parents, rweights,
                                        real.normals.size() == (size_t)RV * 3 ? &real.normals : nullptr);
        check("write_rigged_glb (real mesh)", rw);
        glb::Mesh back;
        bool rb = glb::read_glb(OUT_REAL, back);
        check("glb::read_glb re-parse (real mesh)", rb);
        check("real re-parse POSITION count == V", rb && (uint32_t)(back.verts.size()/3) == RV);
        RigCheck rrc = inspect_rigged_glb(OUT_REAL, parents);
        check("real independent re-parse", rrc.parsed);
        check("real JOINTS_0 count == V", rrc.joints0_count == (int64_t)RV);
        check("real weight rows sum ~ 1.0", rrc.max_weight_sum_err < 1e-4);
        check("real node tree consistent", rrc.node_tree_ok);
        long real_sz = file_size(OUT_REAL);
        std::printf("  wrote %s (%ld bytes)\n", OUT_REAL, real_sz);
    }

    std::printf("\n%s (%d failure%s)\n", fails == 0 ? "OK" : "FAILED", fails, fails == 1 ? "" : "s");
    return fails == 0 ? 0 : 1;
}
