// rig_grammar.hpp — header-only port of the SkinTokens / TokenRig "R3" constrained
// decoding grammar. PURE CPU LOGIC. No GPU, no model weights.
//
// Ports:
//   - tokenizer_part.py : TokenizerPart.next_posible_token() + bones_in_sequence()
//   - tokenrig.py       : VocabSwitchingLogitsProcessor (the two-phase mask logic)
//
// Two vocab spaces (see meta.json / tok_spec.txt for the golden run):
//
//   SKELETON / TOKENIZER vocab  (tokenizer_vocab = 267 ids, [0,267)):
//      [0, num_discrete)        discretized joint coordinates   (num_discrete=256)
//      branch  = num_discrete+0 = 256
//      bos     = num_discrete+1 = 257
//      eos     = num_discrete+2 = 258   (tokenizer-eos == the "switch" token)
//      pad     = num_discrete+3 = 259
//      spring  = num_discrete+4 = 260
//      parts   = 261, 262                (parts_token_id, count = 2)
//      cls_none= 263
//      cls_ids = 264, 265, 266           (cls_token_id, count = 3)
//      => vocab_size (tokenizer) = 267
//
//   MODEL vocab (vocab_size = tokenizer_vocab + vae_vocab + 1 = 267 + 32768 + 1 = 33036):
//      [0, 267)                 skeleton/tokenizer ids (as above)
//      [267, 267+32768)         VAE skin codes: skin_token = vae_code + tokenizer_vocab
//      model-eos = 33035        (= vocab_size - 1; emitted to terminate the skin phase)
//
// Phase 1 (skeleton): logits are masked to next_possible_tokens(sequence). The grammar
//   walks a small state machine. It terminates when the tokenizer-eos (258, the "switch"
//   token) is emitted (next_possible_tokens offers it in expect_part_or_joint /
//   expect_branch_or_part_or_joint states).
// Phase 2 (skin): once the switch token (258) is present in the sequence, every step is
//   masked to ONLY the skin vocab [267, 33036) (i.e. mask[switch_token_id:] = 0). When
//   exactly J*tokens_per_skin skin tokens have been emitted (length-where == J*tps), the
//   mask collapses to the single model-eos id. J = bones_in_sequence(full sequence).

#pragma once
#include <vector>
#include <cstdint>
#include <algorithm>

namespace rig {

// ---- Token-id scheme (mirrors TokenizerPart.parse). Defaults = the golden run spec. ----
struct GrammarSpec {
    int num_discrete = 256;
    int n_parts      = 2;   // len(parts_token_id)
    int n_cls        = 3;   // len(cls_token_id)

    // tokenizer / skeleton vocab ids (derived; matches tok_spec.txt)
    int token_id_branch;    // 256
    int token_id_bos;       // 257
    int token_id_eos;       // 258  (== switch token in the logits processor)
    int token_id_pad;       // 259
    int token_id_spring;    // 260
    int parts_lo, parts_hi; // [261, 263)  i.e. ids 261,262
    int token_id_cls_none;  // 263
    int cls_lo, cls_hi;     // [264, 267)  i.e. ids 264,265,266
    int tokenizer_vocab;    // 267

    // model vocab (skin phase)
    int vae_vocab    = 32768;
    int model_vocab;        // tokenizer_vocab + vae_vocab + 1 = 33036
    int model_eos;          // model_vocab - 1 = 33035
    int tokens_per_skin = 4;

    GrammarSpec() { recompute(); }

    void recompute() {
        int off = num_discrete;
        token_id_branch = off + 0;
        token_id_bos    = off + 1;
        token_id_eos    = off + 2;
        token_id_pad    = off + 3;
        off += 4;
        token_id_spring = off + 0;
        off += 1;
        parts_lo = off;
        off += n_parts;
        parts_hi = off;            // exclusive
        token_id_cls_none = off + 0;
        off += 1;
        cls_lo = off;
        off += n_cls;
        cls_hi = off;              // exclusive
        tokenizer_vocab = off;     // 267

        model_vocab = tokenizer_vocab + vae_vocab + 1;
        model_eos   = model_vocab - 1;
    }

    bool is_cls(int id)  const { return id >= cls_lo  && id < cls_hi; }
    bool is_part(int id) const { return id >= parts_lo && id < parts_hi; }
};

// ---------------------------------------------------------------------------
// next_possible_tokens — faithful port of TokenizerPart.next_posible_token().
//
// `emitted` is the FULL sequence so far in MODEL-vocab ids (must begin with bos).
// Returns the set of valid next ids (skeleton phase only). Empty input => [bos].
// ---------------------------------------------------------------------------
inline std::vector<int> next_possible_tokens(const std::vector<int>& emitted,
                                             const GrammarSpec& g = GrammarSpec()) {
    if (emitted.empty()) return { g.token_id_bos };

    enum State {
        EXPECT_BOS,
        EXPECT_CLS_OR_PART_OR_JOINT,
        EXPECT_PART_OR_JOINT,
        EXPECT_JOINT_2,
        EXPECT_JOINT_3,
        EXPECT_BRANCH_OR_PART_OR_JOINT,
        // A branch serializes an explicit parent coordinate triple followed
        // by the child triple.  Treating it as an ordinary three-coordinate
        // joint makes EOS legal halfway through that six-token payload; R5
        // then correctly rejects the supposedly completed sequence.
        EXPECT_BRANCH_PARENT_1,
        EXPECT_BRANCH_PARENT_2,
        EXPECT_BRANCH_PARENT_3,
        EXPECT_BRANCH_CHILD_1,
        EXPECT_BRANCH_CHILD_2,
        EXPECT_BRANCH_CHILD_3,
        EXPECT_JOINT,
        EXPECT_CLS,   // present in the python emit-switch but never reached by the walk
    };

    State state = EXPECT_BOS;
    for (int id : emitted) {
        switch (state) {
            case EXPECT_BOS:
                // python: assert id == bos. We trust the caller (replay guarantees it).
                state = EXPECT_CLS_OR_PART_OR_JOINT;
                break;
            case EXPECT_CLS_OR_PART_OR_JOINT:
                if (id < g.num_discrete)                              state = EXPECT_JOINT_2;
                else if (id == g.token_id_cls_none || g.is_cls(id))   state = EXPECT_PART_OR_JOINT;
                else                                                  state = EXPECT_JOINT; // a part (spring/parts)
                break;
            case EXPECT_PART_OR_JOINT:
                if (id < g.num_discrete) state = EXPECT_JOINT_2;
                else                     state = EXPECT_PART_OR_JOINT;
                break;
            case EXPECT_JOINT_2:
                state = EXPECT_JOINT_3;
                break;
            case EXPECT_JOINT_3:
                state = EXPECT_BRANCH_OR_PART_OR_JOINT;
                break;
            case EXPECT_BRANCH_OR_PART_OR_JOINT:
                if (id == g.token_id_branch) state = EXPECT_BRANCH_PARENT_1;
                else if (id < g.num_discrete) state = EXPECT_JOINT_2;
                else                          state = EXPECT_JOINT; // a part
                break;
            case EXPECT_BRANCH_PARENT_1: state = EXPECT_BRANCH_PARENT_2; break;
            case EXPECT_BRANCH_PARENT_2: state = EXPECT_BRANCH_PARENT_3; break;
            case EXPECT_BRANCH_PARENT_3: state = EXPECT_BRANCH_CHILD_1; break;
            case EXPECT_BRANCH_CHILD_1:  state = EXPECT_BRANCH_CHILD_2; break;
            case EXPECT_BRANCH_CHILD_2:  state = EXPECT_BRANCH_CHILD_3; break;
            case EXPECT_BRANCH_CHILD_3:  state = EXPECT_BRANCH_OR_PART_OR_JOINT; break;
            case EXPECT_JOINT:
                state = EXPECT_JOINT_2;
                break;
            default:
                break; // unreachable
        }
    }

    std::vector<int> s;
    auto add_cls = [&]() {
        s.push_back(g.token_id_cls_none);
        for (int v = g.cls_lo; v < g.cls_hi; ++v) s.push_back(v);
    };
    auto add_part = [&]() {
        s.push_back(g.token_id_spring);
        for (int v = g.parts_lo; v < g.parts_hi; ++v) s.push_back(v);
    };
    auto add_joint = [&]() {
        for (int i = 0; i < g.num_discrete; ++i) s.push_back(i);
    };
    auto add_branch = [&]() { s.push_back(g.token_id_branch); };
    auto add_eos    = [&]() { s.push_back(g.token_id_eos); };
    auto add_bos    = [&]() { s.push_back(g.token_id_bos); };

    switch (state) {
        case EXPECT_BOS:                 add_bos(); break;
        case EXPECT_CLS_OR_PART_OR_JOINT: add_cls(); add_part(); add_joint(); break;
        case EXPECT_CLS:                 add_cls(); break;
        case EXPECT_PART_OR_JOINT:       add_part(); add_joint(); add_eos(); break;
        case EXPECT_JOINT_2:             add_joint(); break;
        case EXPECT_JOINT_3:             add_joint(); break;
        case EXPECT_BRANCH_PARENT_1:     add_joint(); break;
        case EXPECT_BRANCH_PARENT_2:     add_joint(); break;
        case EXPECT_BRANCH_PARENT_3:     add_joint(); break;
        case EXPECT_BRANCH_CHILD_1:      add_joint(); break;
        case EXPECT_BRANCH_CHILD_2:      add_joint(); break;
        case EXPECT_BRANCH_CHILD_3:      add_joint(); break;
        case EXPECT_BRANCH_OR_PART_OR_JOINT: add_joint(); add_part(); add_branch(); add_eos(); break;
        case EXPECT_JOINT:               add_joint(); break;
    }
    return s;
}

// ---------------------------------------------------------------------------
// bones_in_sequence — faithful port of TokenizerPart.bones_in_sequence().
// Counts completed (non-branch-mate) bones. Breaks at the tokenizer-eos token.
// `emitted` is the FULL sequence (must begin with bos).
// ---------------------------------------------------------------------------
inline int bones_in_sequence(const std::vector<int>& emitted,
                             const GrammarSpec& g = GrammarSpec()) {
    enum State {
        EXPECT_BOS,
        EXPECT_CLS_OR_PART_OR_JOINT,
        EXPECT_PART_OR_JOINT,
        EXPECT_JOINT_2,
        EXPECT_JOINT_3,
        EXPECT_BRANCH_OR_PART_OR_JOINT,
        EXPECT_BRANCH_PARENT_1,
        EXPECT_BRANCH_PARENT_2,
        EXPECT_BRANCH_PARENT_3,
        EXPECT_BRANCH_CHILD_1,
        EXPECT_BRANCH_CHILD_2,
        EXPECT_BRANCH_CHILD_3,
        EXPECT_JOINT,
    };
    int s = 0;
    bool is_branch = false;
    State state = EXPECT_BOS;
    for (int id : emitted) {
        switch (state) {
            case EXPECT_BOS:
                state = EXPECT_CLS_OR_PART_OR_JOINT;
                break;
            case EXPECT_CLS_OR_PART_OR_JOINT:
                if (id < g.num_discrete)                              state = EXPECT_JOINT_2;
                else if (id == g.token_id_cls_none || g.is_cls(id))   state = EXPECT_PART_OR_JOINT;
                else                                                  state = EXPECT_JOINT;
                break;
            case EXPECT_PART_OR_JOINT:
                if (id < g.num_discrete) state = EXPECT_JOINT_2;
                else                     state = EXPECT_PART_OR_JOINT;
                break;
            case EXPECT_JOINT_2:
                state = EXPECT_JOINT_3;
                break;
            case EXPECT_JOINT_3:
                if (!is_branch) s += 1;
                is_branch = false;
                state = EXPECT_BRANCH_OR_PART_OR_JOINT;
                break;
            case EXPECT_BRANCH_OR_PART_OR_JOINT:
                if (id == g.token_id_branch) { state = EXPECT_BRANCH_PARENT_1; is_branch = true; }
                else if (id < g.num_discrete) state = EXPECT_JOINT_2;
                else                          state = EXPECT_JOINT;
                break;
            case EXPECT_BRANCH_PARENT_1: state = EXPECT_BRANCH_PARENT_2; break;
            case EXPECT_BRANCH_PARENT_2: state = EXPECT_BRANCH_PARENT_3; break;
            case EXPECT_BRANCH_PARENT_3: state = EXPECT_BRANCH_CHILD_1; break;
            case EXPECT_BRANCH_CHILD_1:  state = EXPECT_BRANCH_CHILD_2; break;
            case EXPECT_BRANCH_CHILD_2:  state = EXPECT_BRANCH_CHILD_3; break;
            case EXPECT_BRANCH_CHILD_3:
                // A six-coordinate branch payload serializes one additional
                // child joint. It must contribute to J just like an ordinary
                // coordinate triple, otherwise the skin-phase mask requests
                // too few 4-token skin codes and accepts model EOS early.
                s += 1;
                is_branch = false;
                state = EXPECT_BRANCH_OR_PART_OR_JOINT;
                break;
            case EXPECT_JOINT:
                state = EXPECT_JOINT_2;
                break;
        }
        if (id == g.token_id_eos) break;
    }
    return s;
}

// ---------------------------------------------------------------------------
// allowed_next_tokens — full two-phase mask, port of VocabSwitchingLogitsProcessor.
//
// Given the FULL model-vocab sequence so far (init + generated), returns the set of
// valid next ids. This is exactly the set of indices the python sets to mask=0.
//
//   Phase 1 (no switch token in sequence yet):
//       -> next_possible_tokens(sequence)   [skeleton ids]
//   Phase 2 (switch token present):
//       J = bones_in_sequence(sequence); length = len(sequence)
//       where = index of first switch token
//       if (length - where) == J*tokens_per_skin  ->  { model_eos }
//       else                                       ->  all skin ids [tokenizer_vocab, model_vocab)
//
// Mirrors the python: mask[switch_token_id:] = 0 means "every id >= switch_token_id",
// i.e. tokenizer-eos(258)..model_vocab-1. In practice the model only ever emits ids in
// the skin range [267, 33035) or the model-eos (33035); the lower tail (258..266) is
// vestigial. We replicate the python literally: start at switch_token_id.
// ---------------------------------------------------------------------------
inline std::vector<int> allowed_next_tokens(const std::vector<int>& sequence,
                                            const GrammarSpec& g = GrammarSpec()) {
    int switch_tok = g.token_id_eos; // tokenizer.eos
    int where = -1;
    for (int i = 0; i < (int)sequence.size(); ++i) {
        if (sequence[i] == switch_tok) { where = i; break; }
    }
    if (where < 0) {
        return next_possible_tokens(sequence, g);
    }
    int length = (int)sequence.size();
    int J = bones_in_sequence(sequence, g);
    if ((length - where) == J * g.tokens_per_skin) {
        return { g.model_eos };
    }
    // python: mask[switch_token_id:] = 0  THEN  mask[eos] = -inf  ->  ids [switch_tok, model_vocab) \ {model_eos}.
    // (eos is FORBIDDEN until exactly J*tokens_per_skin skin tokens are emitted; the == case above forces it.)
    std::vector<int> s;
    s.reserve(g.model_vocab - switch_tok);
    for (int v = switch_tok; v < g.model_vocab; ++v) if (v != g.model_eos) s.push_back(v);
    return s;
}

} // namespace rig
