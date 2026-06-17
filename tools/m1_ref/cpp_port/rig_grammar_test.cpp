// rig_grammar_test.cpp — CPU self-test for rig_grammar.hpp.
//
// Loads the golden emitted sequence (/tmp/skintokens_e2e/output_ids.npy, int64 [567])
// and REPLAYS it: at every step t, asserts golden[t] is in allowed_next_tokens(golden[0:t]).
// This proves the ported grammar accepts the real model output at every step.
//
// Build: ./build.sh rig_grammar_test  (CPU-only, header-only, no ggml)

#include "rig_grammar.hpp"

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <unordered_set>

// ---- Minimal .npy reader: int64 ('<i8'), 1-D, C-contiguous. -----------------
static std::vector<int64_t> load_npy_i8(const std::string& path, std::vector<size_t>& shape) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "FAIL: cannot open %s\n", path.c_str()); exit(2); }
    char magic[6];
    f.read(magic, 6);
    if (memcmp(magic, "\x93NUMPY", 6) != 0) { fprintf(stderr, "FAIL: bad npy magic\n"); exit(2); }
    uint8_t major, minor;
    f.read((char*)&major, 1);
    f.read((char*)&minor, 1);
    uint32_t header_len = 0;
    if (major == 1) {
        uint16_t hl16 = 0;
        f.read((char*)&hl16, 2);
        header_len = hl16;
    } else {
        f.read((char*)&header_len, 4);
    }
    std::string header(header_len, '\0');
    f.read(&header[0], header_len);

    // dtype check
    if (header.find("<i8") == std::string::npos) {
        fprintf(stderr, "FAIL: expected dtype '<i8' in header: %s\n", header.c_str());
        exit(2);
    }
    // parse shape tuple e.g. 'shape': (567,)
    size_t sp = header.find("'shape'");
    if (sp == std::string::npos) sp = header.find("\"shape\"");
    size_t lp = header.find('(', sp);
    size_t rp = header.find(')', lp);
    std::string shp = header.substr(lp + 1, rp - lp - 1);
    shape.clear();
    size_t pos = 0;
    while (pos < shp.size()) {
        while (pos < shp.size() && (shp[pos] == ' ' || shp[pos] == ',')) pos++;
        if (pos >= shp.size()) break;
        size_t end = pos;
        while (end < shp.size() && shp[end] >= '0' && shp[end] <= '9') end++;
        if (end > pos) shape.push_back((size_t)std::stoul(shp.substr(pos, end - pos)));
        pos = end + 1;
    }
    size_t n = 1;
    for (size_t d : shape) n *= d;
    std::vector<int64_t> data(n);
    f.read((char*)data.data(), (std::streamsize)(n * sizeof(int64_t)));
    if (!f) { fprintf(stderr, "FAIL: short read of npy data\n"); exit(2); }
    return data;
}

int main() {
    const char* npy = "/tmp/skintokens_e2e/output_ids.npy";
    std::vector<size_t> shape;
    std::vector<int64_t> g64 = load_npy_i8(npy, shape);
    if (shape.size() != 1) { fprintf(stderr, "FAIL: expected 1-D, got %zu dims\n", shape.size()); return 2; }
    const int N = (int)g64.size();
    printf("loaded %s: %d tokens\n", npy, N);

    rig::GrammarSpec g; // defaults match tok_spec.txt / meta.json
    printf("spec: num_discrete=%d branch=%d bos=%d eos(switch)=%d pad=%d spring=%d "
           "cls_none=%d tokenizer_vocab=%d model_vocab=%d model_eos=%d tps=%d\n",
           g.num_discrete, g.token_id_branch, g.token_id_bos, g.token_id_eos, g.token_id_pad,
           g.token_id_spring, g.token_id_cls_none, g.tokenizer_vocab, g.model_vocab,
           g.model_eos, g.tokens_per_skin);

    std::vector<int> emitted;
    emitted.reserve(N);

    int first_fail = -1;
    for (int t = 0; t < N; ++t) {
        std::vector<int> allowed = rig::allowed_next_tokens(emitted, g);
        int tok = (int)g64[t];
        std::unordered_set<int> aset(allowed.begin(), allowed.end());
        if (aset.find(tok) == aset.end()) {
            first_fail = t;
            // summarize allowed set
            int amin = allowed.empty() ? -1 : allowed[0];
            int amax = amin;
            for (int v : allowed) { amin = std::min(amin, v); amax = std::max(amax, v); }
            fprintf(stderr,
                "FAIL at step t=%d: golden token %d NOT in allowed set "
                "(size=%zu, min=%d, max=%d). prev token=%d. phase=%s\n",
                t, tok, allowed.size(), amin, amax,
                t > 0 ? (int)g64[t - 1] : -1,
                ([&]() {
                    for (int e : emitted) if (e == g.token_id_eos) return "skin";
                    return "skeleton";
                })());
            // print a small head of the allowed set for debugging
            fprintf(stderr, "  allowed head:");
            for (size_t i = 0; i < allowed.size() && i < 24; ++i) fprintf(stderr, " %d", allowed[i]);
            fprintf(stderr, "\n");
            break;
        }
        emitted.push_back(tok);
    }

    if (first_fail >= 0) {
        fprintf(stderr, "REPLAY FAILED at step %d\n", first_fail);
        return 1;
    }
    printf("REPLAY: all %d steps validated (golden token in allowed set at every step)\n", N);

    // bones_in_sequence on the full sequence must equal n_joints = 60.
    int J = rig::bones_in_sequence(emitted, g);
    printf("bones_in_sequence(full) = %d (expected 60)\n", J);
    if (J != 60) { fprintf(stderr, "FAIL: bones_in_sequence != 60\n"); return 1; }

    // Spot-check mid-sequence: at the switch token (idx 326) the count must already be 60,
    // since bones_in_sequence breaks at eos and the whole skeleton precedes it.
    {
        std::vector<int> upto_switch(emitted.begin(), emitted.begin() + 327); // includes eos at 326
        int Jmid = rig::bones_in_sequence(upto_switch, g);
        printf("bones_in_sequence(up to+incl switch @326) = %d (expected 60)\n", Jmid);
        if (Jmid != 60) { fprintf(stderr, "FAIL: mid-sequence bone count != 60\n"); return 1; }
    }
    // Spot-check earlier: a partial skeleton must yield fewer bones, monotonically rising.
    {
        std::vector<int> partial(emitted.begin(), emitted.begin() + 50);
        int Jp = rig::bones_in_sequence(partial, g);
        printf("bones_in_sequence(first 50 tokens) = %d (expected 0 < J < 60)\n", Jp);
        if (!(Jp > 0 && Jp < 60)) { fprintf(stderr, "FAIL: partial bone count out of range\n"); return 1; }
    }

    // Sanity: empty input must offer exactly [bos].
    {
        std::vector<int> empty;
        std::vector<int> a = rig::next_possible_tokens(empty, g);
        if (a.size() != 1 || a[0] != g.token_id_bos) {
            fprintf(stderr, "FAIL: next_possible_tokens([]) != [bos]\n");
            return 1;
        }
    }

    printf("OK\n");
    return 0;
}
