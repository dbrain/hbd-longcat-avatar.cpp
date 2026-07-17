#ifndef __SD_MODEL_TE_HYMOTION_TEXT_HPP__
#define __SD_MODEL_TE_HYMOTION_TEXT_HPP__

// ---------------------------------------------------------------------------
// HY-Motion 1.0 text-encoder contract  --  native ggml, no python at runtime.
//
// This file's ONLY job is to reproduce `HYTextModel.encode()` exactly:
//
//     hymotion/network/text_encoders/text_encoder.py:205-208
//         encode(text) -> (vtxt_raw, ctxt_raw, ctxt_length)
//
// Both encoders are REUSED, not ported -- they already exist in this tree:
//   ctxt  <- Qwen3-8B  via LLM::LLMRunner (src/model/te/llm.hpp, LLMArch::QWEN3)
//   vtxt  <- CLIP-L    via CLIPTextModelRunner (src/model/te/clip.hpp)
//
// ===========================================================================
// THE CONTRACT, WITH CITATIONS. Every line below is load-bearing.
// ===========================================================================
//
// 1. ENCODERS -- text_encoder.py:20-24, 26-47
//      llm            = Qwen/Qwen3-8B, AutoModelForCausalLM, torch_dtype=bfloat16
//      sentence_emb   = openai/clip-vit-large-patch14, CLIPTextModel
//    config.yml:31-34 confirms `llm_type: qwen3`, `max_length_llm: 128`.
//    (The "Qwen3-8B + CLIP-L" claim in the brief is CORRECT -- verified at source.)
//
// 2. WHICH LAYER  ***the classic silent bug; settled by reading transformers***
//      text_encoder.py:155   ctxt_raw = llm_outputs.hidden_states[-1]
//    In HF (transformers 4.53.3 is the pin; read 4.51.3, identical shape), Qwen3Model
//    applies the final norm BEFORE appending the last entry:
//      modeling_qwen3.py:593   hidden_states = self.norm(hidden_states)
//      modeling_qwen3.py:597   all_hidden_states += (hidden_states,)
//    => hidden_states[-1] IS post-final-norm, and is byte-identical to
//       last_hidden_state. There is NO off-by-one here: "the last hidden state" is
//       the right answer, and it is the POST-norm one.
//    Our llm.hpp returns exactly that when out_layers is EMPTY:
//      llm.hpp:1266  auto normed_x = norm->forward(ctx, x);
//      llm.hpp:1281  } else { x = normed_x; }        <-- empty out_layers path
//    So: pass {} for out_layers. Passing {36} would give the PRE-norm layer-36
//    output -- same shape, wrong values.
//
// 3. POOLING (vtxt) -- text_encoder.py:44 pooling_mode = "pooler_output"
//    HF CLIPTextModel.pooler_output = final_layer_norm(last_hidden)[argmax(input_ids)]
//    with NO projection (modeling_clip.py:967-979; the projection exists only in
//    CLIPTextModelWithProjection, modeling_clip.py:1410).
//    *** projection_dim == hidden_size == 768 for CLIP-L, so a wrongly-applied
//        text_projection is SHAPE-CORRECT and silently wrong. tools/convert_hymotion_clip.py
//        does not emit the tensor at all, so it cannot happen. ***
//    argmax(input_ids) == first occurrence of 49407 (EOS is the highest CLIP id);
//    sd.cpp's max_token_idx convention is the same (conditioner.hpp:417-422).
//
// 4. PROMPT TEMPLATE -- text_encoder.py:29-33 + model_constants.py:6-8
//    A Qwen3 chat template, enable_thinking=False, add_generation_prompt=False:
//        <|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n
//    SYSTEM_PROMPT is a python triple-quoted literal, so it *begins with a newline
//    and four spaces and ends with a newline* -- 449 chars,
//    sha256 b9c55290b778c01739a337a36315c2065ef6ccfbfa0a1898b727377ed4965283.
//    That whitespace is part of the tokenised string; do not "tidy" it.
//
// 5. CROP -- text_encoder.py:109-110, 159-164, 222-246
//      crop_start     = token index at which the user text begins
//      max_length_llm = 128 + crop_start          (the tokeniser's max_length)
//      ctxt_raw       = hidden[:, crop_start : crop_start+128]
//      ctxt_length    = clamp(attention_mask.sum() - crop_start, 0, 128)
//
// 6. PADDING -- text_encoder.py:131-141, padding="max_length", padding_side="right"
//    We DO NOT pad the LLM input, and that is not a shortcut -- it is provably
//    identical:
//      (a) Qwen3 is causal and HF derives position_ids from arange, not from the
//          attention mask, so right-hand pad tokens cannot influence any real
//          token's hidden state.
//      (b) The rows we would be dropping (positions >= ctxt_length) are consumed
//          only behind the mask: the refiner's context pooling is a MASKED mean
//          (token_refiner.py:182-184, `(x*mask).sum(1)/denom`) and its self-attention
//          mask is `mask_1 & mask_2` (token_refiner.py:107-111), which blocks
//          pad->real in both directions.
//    So zero-filling those rows == running them as pad tokens. We zero-fill.
//
// 7. CFG -- motion_diffusion.py:536-537, 559-570; config.yml:30
//      text_guidance_scale = 5.0                     (test_cfg.text_guidance_scale)
//      do_cfg = scale > 1.0 and not uncondition_mode
//      uncond branch: null_vtxt_feat, and (enable_ctxt_null_feat: true)
//        null_ctxt_input EXPANDED over the *cond* Lc, sharing the *cond's* mask
//        (ctxt_mask_temporal = cat([mask]*2)).
//      x_pred = basic + scale * (text - basic)
//    5.0 is upstream's number. It is not invented here.
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "core/tensor.hpp"
#include "model/te/clip.hpp"
#include "model/te/llm.hpp"
#include "tokenizers/clip_tokenizer.h"
#include "tokenizers/qwen2_tokenizer.h"

namespace HYMotionText {

// model_constants.py:6-8, verbatim (449 chars incl. the leading "\n    " and trailing "\n").
static const char* SYSTEM_PROMPT =
    "\n    Summarize human motion only from the user text for representation: action "
    "categories, key body-part movements, order/transitions, trajectory/direction, posture; "
    "include style/emotion/speed only if present. Explicitly capture laterality (left/right) "
    "when mentioned; do not guess. If multiple actions are described, indicate the count of "
    "distinct actions (e.g., actions=3) and their order. Do not invent missing info. Keep one "
    "concise paragraph.\n";

static const size_t SYSTEM_PROMPT_LEN = 449;  // guards against an editor eating the whitespace

// Everything in the templated string that precedes the user's text.
inline std::string chat_prefix() {
    return std::string("<|im_start|>system\n") + SYSTEM_PROMPT + "<|im_end|>\n<|im_start|>user\n";
}

// The full string HF's apply_chat_template() produces for this template with
// enable_thinking=False, add_generation_prompt=False.
inline std::string chat_string(const std::string& user_text) {
    return chat_prefix() + user_text + "<|im_end|>\n";
}

// ---------------------------------------------------------------------------
// crop_start
//
// Computed TWO independent ways, which must agree:
//   (A) upstream's marker algorithm (text_encoder.py:222-246): template a "<BOC>"
//       marker, tokenise, and find the marker's token subsequence.
//   (B) |tokenize(chat_prefix())| -- valid because the Qwen2 pre-tokeniser splits at
//       "\n" (`\s*[\r\n]+`, tokenize_util.cpp:815) and the prefix ends with "user\n",
//       so no token can straddle the boundary.
// If they disagree, the template or the tokeniser is wrong and we must not proceed
// silently -- a wrong crop_start feeds the DiT the SYSTEM PROMPT's hidden states
// instead of the user's, at a perfectly valid shape.
// ---------------------------------------------------------------------------
inline int64_t compute_crop_start(Qwen2Tokenizer& tok, std::string* report = nullptr) {
    const std::string marker = "<BOC>";

    std::vector<int> full   = tok.encode(chat_string(marker));
    std::vector<int> mtoks  = tok.encode(marker);
    std::vector<int> prefix = tok.encode(chat_prefix());

    int64_t via_marker = -1;
    if (!mtoks.empty() && full.size() >= mtoks.size()) {
        auto it = std::search(full.begin(), full.end(), mtoks.begin(), mtoks.end());
        if (it != full.end()) {
            via_marker = (int64_t)std::distance(full.begin(), it);
        }
    }
    const int64_t via_prefix = (int64_t)prefix.size();

    if (report) {
        *report = "crop_start: marker=" + (via_marker < 0 ? std::string("<not found>") : std::to_string(via_marker)) +
                  " prefix_len=" + std::to_string(via_prefix);
    }
    if (via_marker >= 0 && via_marker != via_prefix) {
        LOG_ERROR("hymotion-text: crop_start disagreement: marker=%lld prefix=%lld -- "
                  "the chat template or the tokenizer is wrong; refusing to guess",
                  (long long)via_marker, (long long)via_prefix);
        return -1;
    }
    // Upstream's fallback when the marker cannot be located is max(0, len(full)-1).
    // We never take it: (B) is exact and we cross-check it above.
    return via_prefix;
}

// ---------------------------------------------------------------------------
// ctxt  --  Qwen3-8B hidden_states[-1], cropped + masked.
//
// out: ctxt [max_length_llm * hidden]  (position-major, hidden contiguous -- the
//      layout HYMotionRunner::ctxt_vec wants), ctxt_length in [0, max_length_llm].
// ---------------------------------------------------------------------------
inline bool encode_ctxt(LLM::LLMRunner& llm,
                        Qwen2Tokenizer& tok,
                        const std::string& user_text,
                        int64_t crop_start,
                        int64_t max_length_llm,  // 128
                        int64_t hidden,          // 4096
                        int n_threads,
                        std::vector<float>& ctxt_out,
                        int64_t& ctxt_length_out,
                        bool verbose = false) {
    std::vector<int> tokens = tok.encode(chat_string(user_text));

    // truncation=True, max_length = crop_start + _orig_max_length_llm  (text_encoder.py:138)
    const size_t max_len = (size_t)(crop_start + max_length_llm);
    if (tokens.size() > max_len) {
        LOG_WARN("hymotion-text: prompt truncated %zu -> %zu tokens", tokens.size(), max_len);
        tokens.resize(max_len);
    }
    if ((int64_t)tokens.size() <= crop_start) {
        LOG_ERROR("hymotion-text: tokenised prompt (%zu) does not extend past crop_start (%lld)",
                  tokens.size(), (long long)crop_start);
        return false;
    }

    sd::Tensor<int32_t> input_ids({(int64_t)tokens.size(), 1},
                                  std::vector<int32_t>(tokens.begin(), tokens.end()));

    // out_layers = {} -> post-final-norm == HF hidden_states[-1]. See note 2 above.
    auto hs = llm.compute(n_threads, input_ids, sd::Tensor<float>(), {}, {});
    if (hs.empty()) {
        LOG_ERROR("hymotion-text: Qwen3 forward produced nothing");
        return false;
    }
    // sd::Tensor::shape() is GGML ne-ORDER -- fastest-varying FIRST -- so this is
    // [hidden, n_token, batch], NOT {batch, n_token, hidden}. MEASURED: a 106-token prompt at
    // hidden=4096 returns [4096, 106, 1], which admits only one reading. The original guard
    // checked the last two dims and so demanded [.., 106, 4096]; it rejected every correct
    // output. The DATA layout was right all along (ne[0] contiguous => element (t,h) at
    // t*hidden + h, exactly what the memcpy below assumes) -- only the check was inverted.
    const auto& shp = hs.shape();
    if (shp.size() < 2 || shp[0] != hidden || shp[1] != (int64_t)tokens.size()) {
        std::string got;
        for (size_t d = 0; d < shp.size(); d++) {
            got += (d ? ", " : "") + std::to_string((long long)shp[d]);
        }
        LOG_ERROR("hymotion-text: unexpected Qwen3 output shape: got [%s], want [.., %lld, %lld]",
                  got.c_str(), (long long)tokens.size(), (long long)hidden);
        return false;
    }

    ctxt_out.assign((size_t)(max_length_llm * hidden), 0.0f);
    const int64_t n_valid = std::min<int64_t>((int64_t)tokens.size() - crop_start, max_length_llm);
    std::memcpy(ctxt_out.data(),
                hs.data() + (size_t)(crop_start * hidden),
                (size_t)(n_valid * hidden) * sizeof(float));
    // Rows [n_valid, max_length_llm) stay zero -- provably equivalent to upstream's
    // pad-token rows, see note 6.
    ctxt_length_out = n_valid;

    if (verbose) {
        LOG_INFO("hymotion-text: %zu tokens, crop_start=%lld, ctxt_length=%lld/%lld",
                 tokens.size(), (long long)crop_start, (long long)n_valid, (long long)max_length_llm);
    }
    return true;
}

// ---------------------------------------------------------------------------
// vtxt  --  CLIP-L pooler_output [768].
//
// The reference tokenises with padding=True (i.e. pad-to-longest; for a single
// prompt that is no padding at all) and truncation to 77. We pad to 77, which is
// equivalent: CLIP's text tower is causal, so tokens after the real EOS cannot
// influence the pooled position, and we pool at the FIRST 49407 either way.
// ---------------------------------------------------------------------------
inline bool encode_vtxt(CLIPTextModelRunner& clip,
                        CLIPTokenizer& tok,
                        const std::string& user_text,
                        int64_t vtxt_dim,  // 768
                        int n_threads,
                        std::vector<float>& vtxt_out,
                        bool verbose = false) {
    std::vector<int> tokens = tok.encode(user_text);
    std::vector<float> weights(tokens.size(), 1.0f);
    tok.pad_tokens(tokens, &weights, nullptr, /*min_length*/ 77, /*max_length*/ 77, false);

    // argmax(input_ids) == first EOS (49407 is the highest id in the CLIP vocab).
    size_t max_token_idx = 0;
    auto it              = std::find(tokens.begin(), tokens.end(), tok.EOS_TOKEN_ID);
    if (it != tokens.end()) {
        max_token_idx = std::min<size_t>((size_t)std::distance(tokens.begin(), it), tokens.size() - 1);
    } else {
        LOG_ERROR("hymotion-text: no EOS in CLIP tokens -- cannot locate the pooled position");
        return false;
    }

    sd::Tensor<int32_t> input_ids({(int64_t)tokens.size(), 1},
                                  std::vector<int32_t>(tokens.begin(), tokens.end()));
    auto pooled = clip.compute(n_threads, input_ids, 0, nullptr, max_token_idx,
                               /*return_pooled*/ true, /*clip_skip*/ -1);
    if (pooled.empty() || pooled.numel() != vtxt_dim) {
        LOG_ERROR("hymotion-text: CLIP pooled has %lld elems, want %lld",
                  (long long)(pooled.empty() ? 0 : pooled.numel()), (long long)vtxt_dim);
        return false;
    }
    vtxt_out.assign(pooled.data(), pooled.data() + vtxt_dim);
    if (verbose) {
        LOG_INFO("hymotion-text: CLIP pooled at token %zu of %zu", max_token_idx, tokens.size());
    }
    return true;
}

}  // namespace HYMotionText

#endif  // __SD_MODEL_TE_HYMOTION_TEXT_HPP__
