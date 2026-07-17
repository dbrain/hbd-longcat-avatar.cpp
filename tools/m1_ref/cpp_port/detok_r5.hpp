// detok_r5.hpp — R5 host skeleton de-tokenizer, header-only (no ggml/GPU). The VALIDATED core of
// detok_r5.cpp factored out so the e2e driver can call it. Mirrors TokenizerPart.detokenize +
// make_skeleton: parse the skeleton token stream -> joints (undiscretized xyz) + parent tree.
//
// The algorithm is byte-identical to detok_r5.cpp (which remains the standalone validation harness
// against the goldens); this header only changes the I/O boundary (in: token ids; out: joints+parents).
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <array>
#include <fstream>
#include <sstream>
#include <set>

namespace detok {

struct Spec {
    int num_discrete = 256, bos = 257, eos = 258, pad = 259, branch = 256, spring = 260,
        cls_none = 263, vocab = 267;
    float lo = -1.f, hi = 1.f;
    std::set<int> cls_ids, parts_ids;
};

// Load tok_spec.txt (the golden run spec). If the file is absent, the defaults above (== the
// SkinTokens golden) are used.
inline Spec load_spec(const std::string& path) {
    Spec s;
    std::ifstream f(path);
    if (!f) {  // defaults already match the golden; populate the id sets.
        s.cls_ids = {264, 265, 266}; s.parts_ids = {261, 262};
        return s;
    }
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream is(line); std::string k; is >> k;
        if (k == "num_discrete") is >> s.num_discrete; else if (k == "cont_lo") is >> s.lo;
        else if (k == "cont_hi") is >> s.hi; else if (k == "bos") is >> s.bos; else if (k == "eos") is >> s.eos;
        else if (k == "pad") is >> s.pad; else if (k == "branch") is >> s.branch; else if (k == "spring") is >> s.spring;
        else if (k == "cls_none") is >> s.cls_none; else if (k == "vocab_size") is >> s.vocab;
        else if (k == "cls_ids") { int v; while (is >> v) s.cls_ids.insert(v); }
        else if (k == "parts_ids") { int v; while (is >> v) s.parts_ids.insert(v); }
    }
    return s;
}

typedef std::array<float, 3> V3;
inline V3 undiscretize(const int64_t* t, const Spec& s) {
    V3 o;
    for (int c = 0; c < 3; c++) { float v = ((float)t[c] + 0.5f) / s.num_discrete; o[c] = v * (s.hi - s.lo) + s.lo; }
    return o;
}
inline float d2(const V3& a, const V3& b) { float x = a[0] - b[0], y = a[1] - b[1], z = a[2] - b[2]; return x * x + y * y + z * z; }

// Result of de-tokenization.
struct Skeleton {
    std::vector<float> joints;   // J*3 row-major
    std::vector<int>   parents;  // J, root = -1
    int J = 0;
    bool ok = false;
    std::string err;
};

// Detokenize a FULL model-vocab output_ids sequence (must begin with bos; first tokenizer-eos marks
// the skeleton/skin boundary). Returns joints[J,3] + parents[J]. Skin tokens after the eos are
// ignored here (handled by R4).
inline Skeleton detokenize(const int64_t* all, int64_t n_all, const Spec& s) {
    Skeleton r;
    int64_t eos_pos = -1;
    for (int64_t i = 0; i < n_all; i++) if (all[i] == s.eos) { eos_pos = i; break; }
    if (eos_pos < 0) { r.err = "no eos"; return r; }
    if (all[0] != s.bos) { r.err = "first token not bos"; return r; }
    std::vector<int64_t> ids(all + 1, all + eos_pos);   // strip bos + eos

    std::vector<V3> joints, p_joints;
    bool is_branch = false; V3 last_joint{};
    size_t i = 0;
    while (i < ids.size()) {
        int64_t id = ids[i];
        if (id < s.num_discrete) {
            V3 current;
            if (is_branch) {
                // a branch joint needs a full 6-token group (parent xyz ++ child xyz). On a malformed/
                // truncated stream the tail may be short -> reading ids[i+3..i+5] would run off the end
                // (garbage int64 -> ~1e12 joint coords). Stop parsing cleanly instead of emitting junk.
                if (i + 6 > ids.size()) break;
                V3 p_joint = undiscretize(&ids[i], s);
                current = undiscretize(&ids[i + 3], s);
                joints.push_back(current); p_joints.push_back(p_joint); i += 6;
            } else {
                if (i + 3 > ids.size()) break;   // need a full xyz triplet; tail too short -> stop
                current = undiscretize(&ids[i], s);
                joints.push_back(current);
                if (p_joints.empty()) p_joints.push_back(current);   // root
                else p_joints.push_back(last_joint);
                i += 3;
            }
            last_joint = current; is_branch = false;
        } else if (id == s.branch) { is_branch = true; i += 1; }
        else if (id == s.spring || s.parts_ids.count((int)id)) { i += 1; }
        else if (s.cls_ids.count((int)id) || id == s.cls_none) { i += 1; }
        else { r.err = "unexpected token " + std::to_string((long long)id); return r; }
    }
    const int J = (int)joints.size();

    // make_skeleton parents: joint 0 = root(-1); for k>0, parent = argmin_{j<k, reversed} |joints[j]-p_joints[k]|^2
    std::vector<int> parents(J, -1);
    for (int k = 1; k < J; k++) {
        float best = 1e30f; int pid = -1;
        for (int j = k - 1; j >= 0; j--) { float nd = d2(joints[j], p_joints[k]); if (nd < best) { best = nd; pid = j; } }
        parents[k] = pid;
    }

    r.joints.resize((size_t)J * 3);
    for (int k = 0; k < J; k++) for (int c = 0; c < 3; c++) r.joints[(size_t)k * 3 + c] = joints[k][c];
    r.parents = std::move(parents);
    r.J = J; r.ok = true;
    return r;
}

} // namespace detok
