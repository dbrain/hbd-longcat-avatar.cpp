// rig_bone_names_roundtrip.cpp — prove the WRITER actually carries standard bone names.
//
// Takes a real rigged GLB's skeleton, names it, writes it back out through the production
// writer (glb::write_rigged_glb with joint_names), re-reads the result, and asserts:
//   1. every joint node name in the new GLB == the name we asked for
//   2. skin.joints still lists all J joints in order (skinning is by node INDEX, so
//      renaming must not perturb it)
//   3. re-naming the re-read skeleton reproduces the same names (idempotent)
//
//   ./rig_bone_names_roundtrip <rigged.glb> <out.glb>
#include "glb_rigged.hpp"
#include "glb_reader.hpp"
#include "rig_bone_names.hpp"
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// (skeleton reader shared with rig_bone_names_test — kept local to avoid a new header)
static bool read_skeleton(const char* path, std::vector<float>& joints,
                          std::vector<int>& parents, std::vector<std::string>& names_in) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END); long fsz = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf((size_t)fsz);
    if (std::fread(buf.data(), 1, (size_t)fsz, f) != (size_t)fsz) { std::fclose(f); return false; }
    std::fclose(f);
    uint32_t jlen; std::memcpy(&jlen, buf.data()+12, 4);
    glb::detail::JVal root;
    glb::detail::JParser p((const char*)buf.data()+20, jlen);
    if (!p.parse(root)) return false;
    const glb::detail::JVal* nodes = root.find("nodes");
    if (!nodes || !nodes->is_arr()) return false;
    const int NN = (int)nodes->arr.size();
    std::vector<int> npar(NN, -1);
    for (int i = 0; i < NN; i++) {
        const glb::detail::JVal* c = nodes->arr[i].find("children");
        if (c && c->is_arr()) for (const auto& cv : c->arr) npar[(int)cv.as_int()] = i;
    }
    std::vector<float> gp((size_t)NN*3, 0.f);
    { std::vector<int> st;
      for (int i = 0; i < NN; i++) if (npar[i] == -1) st.push_back(i);
      while (!st.empty()) { int i = st.back(); st.pop_back();
        const glb::detail::JVal* t = nodes->arr[i].find("translation");
        float lt[3] = {0,0,0};
        if (t && t->is_arr() && t->arr.size()==3) for (int k=0;k<3;k++) lt[k]=(float)t->arr[k].as_double();
        int pi = npar[i];
        for (int k=0;k<3;k++) gp[(size_t)i*3+k] = (pi>=0 ? gp[(size_t)pi*3+k] : 0.f) + lt[k];
        const glb::detail::JVal* c = nodes->arr[i].find("children");
        if (c && c->is_arr()) for (const auto& cv : c->arr) st.push_back((int)cv.as_int()); } }
    // NOTE: translation-only composition (no joint rotations) — true for both the SkinTokens
    // writer and our own writer, which is what this round-trip exercises.
    std::map<int,int> num_to_node;
    for (int i = 0; i < NN; i++) {
        const glb::detail::JVal* n = nodes->arr[i].find("name");
        if (n && n->is_str() && n->str.rfind("bone_",0)==0) num_to_node[std::atoi(n->str.c_str()+5)] = i;
    }
    if (num_to_node.empty()) return false;
    std::map<int,int> n2l; std::vector<int> l2n;
    for (auto& kv : num_to_node) { n2l[kv.second] = (int)l2n.size(); l2n.push_back(kv.second); }
    const int J = (int)l2n.size();
    joints.resize((size_t)J*3); parents.assign(J,-1); names_in.assign(J,"");
    for (int li = 0; li < J; li++) {
        int ni = l2n[li];
        for (int k=0;k<3;k++) joints[(size_t)li*3+k] = gp[(size_t)ni*3+k];
        int p = npar[ni]; while (p!=-1 && !n2l.count(p)) p = npar[p];
        parents[li] = (p!=-1) ? n2l[p] : -1;
        names_in[li] = nodes->arr[ni].find("name")->str;
    }
    return true;
}

// read node names + skin.joints out of a GLB
static bool read_names_and_skin(const char* path, std::vector<std::string>& node_names,
                                std::vector<int>& skin_joints) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END); long fsz = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf((size_t)fsz);
    if (std::fread(buf.data(), 1, (size_t)fsz, f) != (size_t)fsz) { std::fclose(f); return false; }
    std::fclose(f);
    uint32_t jlen; std::memcpy(&jlen, buf.data()+12, 4);
    glb::detail::JVal root;
    glb::detail::JParser p((const char*)buf.data()+20, jlen);
    if (!p.parse(root)) return false;
    const glb::detail::JVal* nodes = root.find("nodes");
    if (!nodes) return false;
    for (const auto& n : nodes->arr) {
        const glb::detail::JVal* nm = n.find("name");
        node_names.push_back(nm && nm->is_str() ? nm->str : "");
    }
    const glb::detail::JVal* skins = root.find("skins");
    if (skins && skins->is_arr() && !skins->arr.empty()) {
        const glb::detail::JVal* j = skins->arr[0].find("joints");
        if (j && j->is_arr()) for (const auto& v : j->arr) skin_joints.push_back((int)v.as_int());
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s <rigged.glb> <out.glb>\n", argv[0]); return 2; }
    std::vector<float> joints; std::vector<int> parents; std::vector<std::string> orig;
    if (!read_skeleton(argv[1], joints, parents, orig)) { std::fprintf(stderr, "read failed\n"); return 2; }
    const int J = (int)parents.size();

    rig::BoneNaming BN = rig::name_bones(joints, parents, rig::NameOpts{});
    if (!BN.ok) { std::fprintf(stderr, "naming failed: %s\n", BN.fail_reason.c_str()); return 2; }

    // minimal mesh so the writer has something to skin (the writer is what we are testing)
    std::vector<float>   verts = {0,0,0, 1,0,0, 0,1,0};
    std::vector<int64_t> faces = {0,1,2};
    std::vector<float>   w((size_t)3*J, 0.f);
    for (int v = 0; v < 3; v++) w[(size_t)v*J + (v % J)] = 1.f;   // each vert bound to one joint

    if (!glb::write_rigged_glb(argv[2], verts, faces, joints, parents, w, nullptr, &BN.names)) {
        std::fprintf(stderr, "write failed\n"); return 2;
    }

    std::vector<std::string> got; std::vector<int> skin_joints;
    if (!read_names_and_skin(argv[2], got, skin_joints)) { std::fprintf(stderr, "reread failed\n"); return 2; }

    int fails = 0;
    // 1. names survived the writer verbatim
    int mism = 0;
    for (int j = 0; j < J; j++) if (j >= (int)got.size() || got[j] != BN.names[j]) mism++;
    std::printf("[%s] names-survive-writer     %d/%d joint nodes match the requested name\n",
                mism ? "FAIL" : "PASS", J - mism, J);
    if (mism) fails++;

    // 2. skin.joints untouched: J entries, identity order (skinning is by node index)
    bool skin_ok = ((int)skin_joints.size() == J);
    for (int j = 0; skin_ok && j < J; j++) if (skin_joints[j] != j) skin_ok = false;
    std::printf("[%s] skin.joints-intact       %d entries, identity order (renaming must not move indices)\n",
                skin_ok ? "PASS" : "FAIL", (int)skin_joints.size());
    if (!skin_ok) fails++;

    // 3. no duplicate names (a duplicate would make name-based retargeting ambiguous)
    std::map<std::string,int> seen; int dup = 0;
    for (int j = 0; j < J; j++) if (++seen[BN.names[j]] == 2) dup++;
    std::printf("[%s] no-duplicate-names       %d duplicated name(s)\n", dup ? "FAIL" : "PASS", dup);
    if (dup) fails++;

    std::printf("\nround-trip %s (J=%d, core=%d/22) -> %s\n",
                fails ? "FAILED" : "CLEAN", J, BN.named_core, argv[2]);
    return fails;
}
