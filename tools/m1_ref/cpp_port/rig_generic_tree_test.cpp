// rig_generic_tree_test.cpp — focused invariants for the non-humanoid rig gate.
//
// Build/run: ./build.sh rig_generic_tree_test && ./rig_generic_tree_test
#include "rig_bone_names.hpp"

#include <cstdio>
#include <vector>

static std::vector<float> points(int n) {
    std::vector<float> p((size_t)n * 3, 0.f);
    for (int i = 0; i < n; ++i) p[(size_t)i * 3] = (float)i;
    return p;
}

int main() {
    int failures = 0;
    auto expect = [&](bool ok, const char* what) {
        std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what);
        if (!ok) ++failures;
    };

    // A generic tree can be smaller than a humanoid, and may have more than
    // seven immediate limbs.  Semantic/quality policy is applied separately.
    {
        auto n = rig::name_generic_bones(points(10), {-1, 0, 0, 0, 0, 0, 0, 0, 0, 0});
        expect(n.ok && n.root == 0 && n.max_fan == 9, "9-limb generic root is structurally valid");
        expect(n.names[0] == "skintokens:Root_000" && n.names[9] == "skintokens:Joint_009",
               "generic names retain stable source indexes");
    }
    expect(rig::generic_max_fan_limit(10) == 8, "small creature fan limit permits 8 branches");
    expect(rig::generic_max_fan_limit(40) == 8, "bird-scale creature fan limit permits 8 branches");
    expect(rig::generic_max_fan_limit(45) == 9, "large creature fan limit scales with joint count");
    expect(rig::generic_max_fan_limit(173) == 35, "collapsed Tira-style fan remains rejectable");

    // A genuine decoder star remains rejected by the ordinary gate, but the
    // narrow native normalizer may reparent only its overflow children.  It
    // preserves every source joint/index and produces a size-aware tree;
    // callers still have to clear the real LBS audit after skinning.
    {
        std::vector<int> star(56, 0); star[0] = -1;
        std::vector<float> cloud = points(56);
        std::string detail;
        const bool repaired = rig::normalize_generic_parent_fan(cloud, star, &detail);
        auto named = rig::name_generic_bones(cloud, star);
        expect(repaired && named.ok && named.max_fan <= rig::generic_max_fan_limit(56),
               "pathological generic fan is locally normalized without dropping joints");
    }

    expect(!rig::name_generic_bones(points(2), {-1, -1}).ok, "multiple roots are rejected");
    expect(!rig::name_generic_bones(points(4), {-1, 0, 3, 2}).ok,
           "disconnected cyclic component is rejected");
    expect(!rig::name_generic_bones(points(2), {-1, -2}).ok, "parent values below -1 are rejected");

    // A long generic creature is allowed when its joints advance through
    // space; a decoder runaway that merely repeats quantised coordinates is
    // not a rig and must fail before it reaches the skinning fallback.
    {
        auto long_chain = points(201);
        expect(rig::name_generic_bones(long_chain, [&] { std::vector<int> p(201); p[0] = -1; for (int i=1;i<201;i++) p[i]=i-1; return p; }()).ok,
               "deep articulated generic chain remains valid");
        std::fill(long_chain.begin() + 3, long_chain.end(), 0.f);
        expect(!rig::name_generic_bones(long_chain, [&] { std::vector<int> p(201); p[0] = -1; for (int i=1;i<201;i++) p[i]=i-1; return p; }()).ok,
               "duplicate-position generic runaway is rejected");
    }

    return failures == 0 ? 0 : 1;
}
