// detok_r5.cpp — R5: host skeleton de-tokenizer (no ggml/GPU). Mirrors TokenizerPart.detokenize +
// make_skeleton: parse the skeleton token stream -> joints (undiscretized xyz) + parent tree. Validates
// vs the Python golden (detok_joints.npy / detok_parents.npy). Reads tok_spec.txt + output_ids.npy.
//   build: ./build.sh detok_r5   |   run: ./detok_r5 [golden=/tmp/skintokens_e2e]
#include "../../sparse_spike/npy.hpp"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <cmath>
#include <fstream>
#include <sstream>
#include <set>

struct Spec {
    int num_discrete=256, bos=257, eos=258, pad=259, branch=256, spring=260, cls_none=263, vocab=267;
    float lo=-1.f, hi=1.f;
    std::set<int> cls_ids, parts_ids;
};

static Spec load_spec(const std::string& path) {
    Spec s; std::ifstream f(path); std::string line;
    while (std::getline(f, line)) {
        std::istringstream is(line); std::string k; is >> k;
        if (k=="num_discrete") is>>s.num_discrete; else if (k=="cont_lo") is>>s.lo;
        else if (k=="cont_hi") is>>s.hi; else if (k=="bos") is>>s.bos; else if (k=="eos") is>>s.eos;
        else if (k=="pad") is>>s.pad; else if (k=="branch") is>>s.branch; else if (k=="spring") is>>s.spring;
        else if (k=="cls_none") is>>s.cls_none; else if (k=="vocab_size") is>>s.vocab;
        else if (k=="cls_ids") { int v; while (is>>v) s.cls_ids.insert(v); }
        else if (k=="parts_ids") { int v; while (is>>v) s.parts_ids.insert(v); }
    }
    return s;
}

typedef std::array<float,3> V3;
static V3 undiscretize(const int64_t* t, const Spec& s) {
    V3 o;
    for (int c=0;c<3;c++) { float v = ((float)t[c] + 0.5f) / s.num_discrete; o[c] = v*(s.hi-s.lo)+s.lo; }
    return o;
}
static float d2(const V3&a, const V3&b){ float x=a[0]-b[0],y=a[1]-b[1],z=a[2]-b[2]; return x*x+y*y+z*z; }

int main(int argc, char** argv) {
    std::string g = argc>1 ? argv[1] : "/tmp/skintokens_e2e";
    Spec s = load_spec(g + "/tok_spec.txt");
    NpyArray oi = npy_load(g + "/output_ids.npy");
    const int64_t* all = oi.i64();
    int64_t n_all = oi.numel();
    // skeleton = [.. first eos]; strip bos (front) + eos (back)
    int64_t eos_pos = -1;
    for (int64_t i=0;i<n_all;i++) if (all[i]==s.eos) { eos_pos=i; break; }
    if (eos_pos<0) { printf("no eos\n"); return 1; }
    if (all[0]!=s.bos) { printf("first token not bos\n"); return 1; }
    std::vector<int64_t> ids(all+1, all+eos_pos);   // strip bos + eos

    std::vector<V3> joints, p_joints;
    bool is_branch=false; bool have_last=false; V3 last_joint{};
    size_t i=0;
    while (i < ids.size()) {
        int64_t id = ids[i];
        if (id < s.num_discrete) {
            V3 current;
            if (is_branch) {
                V3 p_joint = undiscretize(&ids[i], s);
                current = undiscretize(&ids[i+3], s);
                joints.push_back(current); p_joints.push_back(p_joint); i += 6;
            } else {
                current = undiscretize(&ids[i], s);
                joints.push_back(current);
                if (p_joints.empty()) p_joints.push_back(current);   // root
                else p_joints.push_back(last_joint);
                i += 3;
            }
            last_joint = current; have_last = true; is_branch = false;
        } else if (id == s.branch) { is_branch = true; have_last = false; i += 1; }
        else if (id == s.spring || s.parts_ids.count((int)id)) { i += 1; }
        else if (s.cls_ids.count((int)id) || id == s.cls_none) { i += 1; }
        else { printf("unexpected token %lld at %zu\n", (long long)id, i); return 1; }
    }
    int J = (int) joints.size();

    // make_skeleton parents: joint 0 = root(-1); for i>0, parent = argmin_{j<i, reversed} |joints[j]-p_joints[i]|^2
    std::vector<int> parents(J, -1);
    for (int k=1;k<J;k++) {
        float best=1e30f; int pid=-1;
        for (int j=k-1;j>=0;j--) { float nd=d2(joints[j], p_joints[k]); if (nd<best){best=nd;pid=j;} }
        parents[k]=pid;
    }

    // validate
    NpyArray gj = npy_load(g + "/detok_joints.npy");   // [J,3]
    NpyArray gp = npy_load(g + "/detok_parents.npy");  // [J]
    int Jg = (int) gj.shape[0];
    printf("[detok_r5] J=%d (golden %d), tokens parsed from %lld skeleton ids\n", J, Jg, (long long)ids.size());
    if (J != Jg) { printf("[detok_r5] FAIL joint count\n"); return 1; }
    const float* gjp = gj.f32(); const int64_t* gpp = gp.i64();
    float maxabs=0; int pbad=0;
    for (int k=0;k<J;k++) {
        for (int c=0;c<3;c++) maxabs = std::max(maxabs, std::fabs(joints[k][c]-gjp[k*3+c]));
        if (parents[k] != (int)gpp[k]) pbad++;
    }
    printf("[detok_r5] joints maxabs=%.3e  parents mismatch=%d/%d\n", maxabs, pbad, J);
    bool ok = (maxabs < 1e-5) && (pbad == 0);
    printf("[detok_r5] %s\n", ok ? "PASS (joints exact + parent tree identical)" : "FAIL");
    return ok ? 0 : 1;
}
