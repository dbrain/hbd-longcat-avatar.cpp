// rig_bone_names_test.cpp — name the bones of an existing rigged GLB and run the falsifier.
//
// Reads the bone_* node tree straight out of the GLB JSON chunk (reusing glb_reader's
// JSON parser), composes rest GLOBAL joint positions, names them, prints the evidence.
//
// Exit code = number of failed falsifier checks (so it can gate a script).
//
//   ./rig_bone_names_test <rigged.glb> [--smpl] [--facing +z|-z] [--quiet]
#include "glb_reader.hpp"
#include "rig_bone_names.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// ---- minimal quaternion/vector helpers for the TRS walk ----
struct Q { float x=0,y=0,z=0,w=1; };
static Q qmul(const Q& a, const Q& b) {
    return Q{ a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
              a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
              a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
              a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z };
}
static void qrot(const Q& q, const float v[3], float out[3]) {
    // out = v + 2*cross(q.xyz, cross(q.xyz, v) + q.w*v)
    float tx = 2.f*(q.y*v[2] - q.z*v[1]);
    float ty = 2.f*(q.z*v[0] - q.x*v[2]);
    float tz = 2.f*(q.x*v[1] - q.y*v[0]);
    out[0] = v[0] + q.w*tx + (q.y*tz - q.z*ty);
    out[1] = v[1] + q.w*ty + (q.z*tx - q.x*tz);
    out[2] = v[2] + q.w*tz + (q.x*ty - q.y*tx);
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <rigged.glb> [--smpl] [--facing +z|-z] [--quiet]\n", argv[0]); return 2; }
    const char* path = argv[1];
    rig::NameOpts opt;
    bool quiet = false;
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--smpl") opt.style = rig::NameStyle::SmplH;
        else if (a == "--mixamo") opt.style = rig::NameStyle::Mixamo;
        else if (a == "--quiet") quiet = true;
        else if (a == "--facing" && i+1 < argc) { opt.facing_override = (argv[++i][0]=='-') ? -1 : +1; }
    }

    // ---- read the GLB JSON chunk ----
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); return 2; }
    std::fseek(f, 0, SEEK_END); long fsz = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf((size_t)fsz);
    if (std::fread(buf.data(), 1, (size_t)fsz, f) != (size_t)fsz) { std::fclose(f); return 2; }
    std::fclose(f);
    if (fsz < 20 || std::memcmp(buf.data(), "glTF", 4) != 0) { std::fprintf(stderr, "not a GLB\n"); return 2; }
    uint32_t jlen; std::memcpy(&jlen, buf.data()+12, 4);
    glb::detail::JVal root;
    { glb::detail::JParser p((const char*)buf.data()+20, jlen);
      if (!p.parse(root)) { std::fprintf(stderr, "JSON parse failed\n"); return 2; } }

    const glb::detail::JVal* nodes = root.find("nodes");
    if (!nodes || !nodes->is_arr()) { std::fprintf(stderr, "no nodes[]\n"); return 2; }
    const int NN = (int)nodes->arr.size();

    // node parents
    std::vector<int> npar(NN, -1);
    for (int i = 0; i < NN; i++) {
        const glb::detail::JVal* c = nodes->arr[i].find("children");
        if (!c || !c->is_arr()) continue;
        for (const auto& cv : c->arr) npar[(int)cv.as_int()] = i;
    }
    // local TRS
    std::vector<float> lt((size_t)NN*3, 0.f);
    std::vector<Q>     lq(NN);
    for (int i = 0; i < NN; i++) {
        const glb::detail::JVal* t = nodes->arr[i].find("translation");
        if (t && t->is_arr() && t->arr.size() == 3)
            for (int k = 0; k < 3; k++) lt[(size_t)i*3+k] = (float)t->arr[k].as_double();
        const glb::detail::JVal* r = nodes->arr[i].find("rotation");
        if (r && r->is_arr() && r->arr.size() == 4)
            lq[i] = Q{(float)r->arr[0].as_double(),(float)r->arr[1].as_double(),
                      (float)r->arr[2].as_double(),(float)r->arr[3].as_double()};
        // NOTE: `matrix` nodes are not handled; the SkinTokens writer emits translation-only
        // joint nodes. If a rig ever uses matrices this must be extended (it would show up
        // as a nonsense skeleton, and the falsifier would catch it).
    }
    // rest global positions
    std::vector<float> gp((size_t)NN*3, 0.f);
    std::vector<Q>     gq(NN);
    {
        std::vector<int> stack;
        for (int i = 0; i < NN; i++) if (npar[i] == -1) stack.push_back(i);
        // iterative DFS so deep chains cannot blow the stack
        std::vector<std::pair<int,std::pair<std::array<float,3>,Q>>> work;
        for (int r : stack) work.push_back({r, {{{0,0,0}}, Q{}}});
        while (!work.empty()) {
            auto [i, pp] = work.back(); work.pop_back();
            const auto& ppos = pp.first; const Q& pq = pp.second;
            float rt[3];
            qrot(pq, &lt[(size_t)i*3], rt);
            gp[(size_t)i*3+0] = ppos[0] + rt[0];
            gp[(size_t)i*3+1] = ppos[1] + rt[1];
            gp[(size_t)i*3+2] = ppos[2] + rt[2];
            gq[i] = qmul(pq, lq[i]);
            const glb::detail::JVal* c = nodes->arr[i].find("children");
            if (c && c->is_arr())
                for (const auto& cv : c->arr)
                    work.push_back({(int)cv.as_int(),
                                    {{{gp[(size_t)i*3+0],gp[(size_t)i*3+1],gp[(size_t)i*3+2]}}, gq[i]}});
        }
    }

    // ---- collect the bone_* skeleton, ordered by bone NUMBER ----
    std::map<int,int> num_to_node;
    for (int i = 0; i < NN; i++) {
        const glb::detail::JVal* n = nodes->arr[i].find("name");
        if (!n || !n->is_str()) continue;
        if (n->str.rfind("bone_", 0) != 0) continue;
        num_to_node[std::atoi(n->str.c_str()+5)] = i;
    }
    if (num_to_node.empty()) { std::fprintf(stderr, "no bone_* nodes in %s\n", path); return 2; }

    std::map<int,int> node_to_local;
    std::vector<int> local_to_node;
    for (auto& kv : num_to_node) { node_to_local[kv.second] = (int)local_to_node.size();
                                   local_to_node.push_back(kv.second); }
    const int J = (int)local_to_node.size();
    std::vector<float> joints((size_t)J*3);
    std::vector<int>   parents(J, -1);
    for (int li = 0; li < J; li++) {
        int ni = local_to_node[li];
        for (int k = 0; k < 3; k++) joints[(size_t)li*3+k] = gp[(size_t)ni*3+k];
        int p = npar[ni];
        while (p != -1 && !node_to_local.count(p)) p = npar[p];   // skip non-bone ancestors
        parents[li] = (p != -1) ? node_to_local[p] : -1;
    }

    std::printf("%s: nodes=%d bones=J=%d\n", path, NN, J);

    // ---- name ----
    rig::BoneNaming R = rig::name_bones(joints, parents, opt);
    if (!quiet) {
        std::printf("\n=== bone_N -> name (measured rest position) ===\n");
        int bi = 0;
        for (auto& kv : num_to_node) {
            int li = node_to_local[kv.second];
            std::printf("  bone_%-3d -> %-30s pos=(%+.4f, %+.4f, %+.4f)\n",
                        kv.first, R.names[li].empty() ? "(unnamed)" : R.names[li].c_str(),
                        joints[(size_t)li*3+0], joints[(size_t)li*3+1], joints[(size_t)li*3+2]);
            bi++;
        }
    }
    int fails = rig::falsify_bone_names(joints, parents, R, true);
    std::printf("\nRESULT %s: core=%d/22 fingers=%d extra=%d fails=%d\n",
                fails == 0 ? "CLEAN" : "FALSIFIED", R.named_core, R.named_fingers, R.n_extra, fails);
    return fails;
}
