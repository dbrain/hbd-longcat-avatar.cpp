// detok_grammar_test.cpp — fail-closed grammar checks matching TokenizerPart.detokenize.
//
// Build/run: ./build.sh detok_grammar_test && ./detok_grammar_test
#include "detok_r5.hpp"
#include "rig_grammar.hpp"

#include <cstdio>
#include <vector>

int main() {
    int failures = 0;
    auto expect = [&](bool ok, const char* what) {
        std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what);
        if (!ok) ++failures;
    };
    const detok::Spec s;
    auto parse = [&](std::vector<int64_t> ids) {
        return detok::detokenize(ids.data(), (int64_t)ids.size(), s);
    };

    expect(parse({257, 263, 1, 2, 3, 258}).ok,
           "complete root coordinate group is accepted");
    expect(!parse({257, 263, 1, 2, 258}).ok,
           "truncated ordinary coordinate group is rejected");
    expect(!parse({257, 263, 256, 1, 2, 3, 4, 5, 258}).ok,
           "truncated branch coordinate group is rejected");
    expect(!parse({257, 263, 1, 2, 3, 256, 258}).ok,
           "branch token without coordinates is rejected");
    expect(!parse({257, 263, 258}).ok,
           "empty skeleton is rejected");

    const rig::GrammarSpec g;
    const auto after_branch_parent = rig::allowed_next_tokens({257, 263, 1, 2, 3, 256, 4, 5, 6}, g);
    const bool only_coordinate = after_branch_parent.size() == (size_t)g.num_discrete
        && after_branch_parent.front() == 0 && after_branch_parent.back() == g.num_discrete - 1;
    expect(only_coordinate,
           "grammar requires the child coordinate triple after a branch parent triple");
    const auto complete_branch = rig::allowed_next_tokens({257, 263, 1, 2, 3, 256, 4, 5, 6, 7, 8, 9}, g);
    expect(std::find(complete_branch.begin(), complete_branch.end(), g.token_id_eos) != complete_branch.end(),
           "grammar allows skeleton switch only after a complete branch payload");
    expect(rig::bones_in_sequence({257, 263, 1, 2, 3, 256, 4, 5, 6, 7, 8, 9, 258}, g) == 2,
           "branch child contributes one skin-weight row");
    return failures == 0 ? 0 : 1;
}
