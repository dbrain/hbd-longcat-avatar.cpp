// rig_structural_select.hpp — pick a skeleton from the COMPLETED beam pool by anatomy, not by score.
//
// WHY THIS EXISTS
// ---------------
// The beam search returns up to `num_beams` completed hypotheses. The language-model winner (max
// length-normalized score) is NOT automatically an anatomically valid skeleton: the decoder
// periodically emits a joint cloud whose de-tokenized parent tree collapses into a star ("head fan"
// / "runaway"), and the score has no term for that. `rig_texture_chain.sh` has always defended
// against this by running skintokens_e2e with `structural-select`, which walks the pool in score
// order and takes the FIRST hypothesis that names a complete 22-bone Mixamo skeleton and survives
// the falsifier.
//
// The inline `image_to_rig` path (rig_pipeline.hpp) did NOT have that gate — it took the top beam
// unconditionally — so the two production entry points into the same decoder disagreed. This header
// is that gate, lifted verbatim out of skintokens_e2e.cpp so BOTH callers run the same code and the
// divergence cannot come back.
//
// The gate is a SELECTOR, never a repairer: it only chooses among sequences the model actually
// produced. `normalize_generic_parent_fan` (generic profile only) is the one exception and is
// itself a deterministic re-parent of overflow children, documented in rig_bone_names.hpp.
#pragma once
#include "detok_r5.hpp"
#include "rig_beam_generate.hpp"   // rig::RigHyp
#include "rig_bone_names.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace rig {

struct StructuralPick {
    int  index = -1;             // index into the pool, -1 == nothing passed the gate
    std::vector<int> tokens;     // the selected hypothesis' full token sequence
    int  J = 0;
    bool repairable = false;     // accepted in one of the narrow post-skin repair forms
    bool used_generic = false;   // accepted by the creature gate; caller must re-run
                                 // normalize_generic_parent_fan on the re-detokenized parents
    double normscore = 0.0;
};

// `generic` selects the creature gate (well-formed tree + size-aware fan limit) instead of the
// 22-bone humanoid contract. `candidate_rank >= 0` restricts the generic gate to that one pool
// index (a diagnostic; humanoid selection ignores it). Pool must be sorted best-score-first, which
// is what rig_beam_generate*/finished_out guarantees.
inline StructuralPick structural_select(const std::vector<RigHyp>& pool, const detok::Spec& dspec,
                                        bool generic, int candidate_rank = -1, bool verbose = true) {
    StructuralPick pick;
    auto say = [&](const char* fmt, auto... args) { if (verbose) printf(fmt, args...); };
    for (size_t i = 0; i < pool.size(); ++i) {
        const RigHyp& hyp = pool[i];
        if (!hyp.eos) continue;
        std::vector<int64_t> candidate(hyp.sequence.begin(), hyp.sequence.end());
        detok::Skeleton test = detok::detokenize(candidate.data(), (int64_t)candidate.size(), dspec);
        if (!test.ok) {
            say("[STAGE R3] structural candidate %zu rejected: detokenize %s\n", i, test.err.c_str());
            continue;
        }
        if (generic) {
            if (candidate_rank >= 0 && (int)i != candidate_rank) continue;
            std::string normalized;
            if (rig::normalize_generic_parent_fan(test.joints, test.parents, &normalized))
                say("[STAGE R3] generic candidate %zu: %s\n", i, normalized.c_str());
            rig::GenericBoneNaming gen = rig::name_generic_bones(test.joints, test.parents);
            if (!gen.ok) {
                say("[STAGE R3] generic candidate %zu rejected: %s\n", i, gen.fail_reason.c_str());
                continue;
            }
            const int fan_limit = rig::generic_max_fan_limit(test.J);
            if (gen.max_fan > fan_limit) {
                say("[STAGE R3] generic candidate %zu rejected: maxfan=%d exceeds size-aware limit=%d for J=%d\n",
                    i, gen.max_fan, fan_limit, test.J);
                continue;
            }
            pick.index = (int)i; pick.tokens = hyp.sequence; pick.J = test.J; pick.normscore = hyp.normscore;
            pick.used_generic = true;
            say("[STAGE R3] generic structural-select accepted candidate %zu/%zu: J=%d score=%.5f root=%d maxfan=%d/%d\n",
                i + 1, pool.size(), test.J, hyp.normscore, gen.root, gen.max_fan, fan_limit);
            return pick;
        }
        rig::NameOpts opts;
        if (const char* facing = std::getenv("RIG_BONE_FACING"))
            opts.facing_override = facing[0] == '-' ? -1 : +1;
        rig::BoneNaming named = rig::name_bones(test.joints, test.parents, opts);
        if (!named.ok) {
            say("[STAGE R3] structural candidate %zu rejected: core=%d/22 (%s)\n", i,
                named.named_core, named.fail_reason.c_str());
            if (named.named_core >= 18) {
                bool present[22] = {};
                for (const std::string& name : named.names)
                    for (int s = 0; s < 22; ++s)
                        if (name == rig::kMixamoNames[s]) present[s] = true;
                say("[STAGE R3] structural candidate %zu missing slots:", i);
                for (int s = 0; s < 22; ++s) if (!present[s]) say(" %s", rig::kMixamoNames[s]);
                say("\n");
            }
            continue;
        }
        // The only incomplete form allowed through this pre-skin gate is
        // the already-audited short torso / terminal-head form.  R4 plus
        // combine_rig_tex_main will apply its geometry-and-skin-supported
        // repair and re-run the full falsifier. A separate exact 21/22
        // terminal-toe form is also allowed: combine derives it from the
        // mirrored existing toe plus actual Foot skin mass. Missing a
        // limb, hand, or any other semantic node is never repairable.
        bool present[22] = {};
        int hips = -1, spine1 = -1, spine2 = -1, neck = -1;
        for (int j = 0; j < (int)named.names.size(); ++j) {
            for (int s = 0; s < 22; ++s) if (named.names[j] == rig::kMixamoNames[s]) {
                present[s] = true;
                if (s == 0) hips = j;
                else if (s == 6) spine1 = j;
                else if (s == 9) spine2 = j;
                else if (s == 12) neck = j;
            }
        }
        bool missing_only_spine_head = named.named_core == 20 || named.named_core == 21;
        for (int s = 0; s < 22; ++s)
            if (!present[s] && s != 3 && s != 15) missing_only_spine_head = false;
        bool needs_spine = !present[3], needs_head = !present[15];
        bool repairable = missing_only_spine_head &&
                          (!needs_spine || (hips >= 0 && spine1 >= 0 && spine2 >= 0 &&
                                             test.parents[spine1] == hips)) &&
                          (!needs_head || neck >= 0);
        bool missing_only_one_toe = named.named_core == 21 && (present[10] != present[11]);
        for (int s = 0; s < 22; ++s)
            if (!present[s] && s != 10 && s != 11) missing_only_one_toe = false;
        const int own_foot_slot = present[10] ? 8 : 7;
        const int other_foot_slot = present[10] ? 7 : 8;
        const int other_toe_slot = present[10] ? 10 : 11;
        int own_foot=-1, other_foot=-1, other_toe=-1;
        for (int j=0;j<(int)named.smpl.size();++j) {
            if (named.smpl[j]==own_foot_slot) own_foot=j;
            if (named.smpl[j]==other_foot_slot) other_foot=j;
            if (named.smpl[j]==other_toe_slot) other_toe=j;
        }
        bool toe_repairable = missing_only_one_toe && own_foot>=0 && other_foot>=0 && other_toe>=0 &&
                              test.parents[other_toe] == other_foot;
        bool missing_only_right_arm = named.named_core == 19;
        for(int s=0;s<22;s++) if(!present[s] && s!=17 && s!=19 && s!=21) missing_only_right_arm=false;
        int ls=-1,rs=-1,la=-1,lf=-1,lh=-1;
        for(int j=0;j<(int)named.smpl.size();j++){ if(named.smpl[j]==13)ls=j; if(named.smpl[j]==14)rs=j; if(named.smpl[j]==16)la=j; if(named.smpl[j]==18)lf=j; if(named.smpl[j]==20)lh=j; }
        bool right_arm_repairable=missing_only_right_arm && ls>=0&&rs>=0&&la>=0&&lf>=0&&lh>=0&&test.parents[la]==ls&&test.parents[lf]==la&&test.parents[lh]==lf;
        repairable = repairable || toe_repairable || right_arm_repairable;
        if (named.named_core != 22 && !repairable) {
            say("[STAGE R3] structural candidate %zu rejected: core=%d/22 (not the narrow Spine/Head repair form)\n",
                i, named.named_core);
            say("[STAGE R3] structural candidate %zu missing slots:", i);
            for (int s = 0; s < 22; ++s) if (!present[s]) say(" %s", rig::kMixamoNames[s]);
            say("\n");
            continue;
        }
        if (named.named_core == 22) {
            int fails = rig::falsify_bone_names(test.joints, test.parents, named, false);
            if (fails != 0) {
                say("[STAGE R3] structural candidate %zu rejected: bone falsifier=%d\n", i, fails);
                continue;
            }
        }
        pick.index = (int)i; pick.tokens = hyp.sequence; pick.J = test.J;
        pick.repairable = repairable; pick.normscore = hyp.normscore;
        say("[STAGE R3] structural-select accepted candidate %zu/%zu: J=%d score=%.5f%s\n",
            i + 1, pool.size(), test.J, hyp.normscore,
            repairable ? (right_arm_repairable ? " (exact mirrored right-arm repair required post-skin)" : (toe_repairable ? " (exact one-ToeBase repair required post-skin)" : " (narrow Spine/Head repair required post-skin)")) : "");
        return pick;
    }
    return pick;
}

}  // namespace rig
