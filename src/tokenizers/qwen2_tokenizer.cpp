#include "qwen2_tokenizer.h"

#include "core/util.h"
#include "vocab/vocab.h"

void Qwen2Tokenizer::load_from_merges(const std::string& merges_utf8_str) {
    auto byte_unicode_pairs = bytes_to_unicode();
    byte_encoder            = std::map<int, std::u32string>(byte_unicode_pairs.begin(), byte_unicode_pairs.end());
    for (auto& pair : byte_unicode_pairs) {
        byte_decoder[pair.second] = pair.first;
    }

    std::vector<std::u32string> merges = split_utf32(merges_utf8_str);
    LOG_DEBUG("merges size %zu", merges.size());
    std::vector<std::pair<std::u32string, std::u32string>> merge_pairs;
    for (const auto& merge : merges) {
        size_t space_pos = merge.find(' ');
        merge_pairs.emplace_back(merge.substr(0, space_pos), merge.substr(space_pos + 1));
    }

    std::vector<std::u32string> tokens;
    for (const auto& pair : byte_unicode_pairs) {
        tokens.push_back(pair.second);
    }
    for (const auto& merge : merge_pairs) {
        tokens.push_back(merge.first + merge.second);
    }
    for (auto& special_token : special_tokens) {
        tokens.push_back(utf8_to_utf32(special_token));
    }

    int i = 0;
    for (const auto& token : tokens) {
        encoder[token] = i;
        decoder[i]     = token;
        i++;
    }
    encoder_len = i;
    LOG_DEBUG("vocab size: %d", encoder_len);

    int rank = 0;
    for (const auto& merge : merge_pairs) {
        bpe_ranks[merge] = rank++;
    }
    bpe_len = rank;
}

Qwen2Tokenizer::Qwen2Tokenizer(const std::string& merges_utf8_str, SpecialTail special_tail) {
    UNK_TOKEN = "<|endoftext|>";
    EOS_TOKEN = "<|endoftext|>";
    PAD_TOKEN = "<|endoftext|>";

    UNK_TOKEN_ID = 151643;
    EOS_TOKEN_ID = 151643;
    PAD_TOKEN_ID = 151643;

    special_tokens = {
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|object_ref_start|>",
        "<|object_ref_end|>",
        "<|box_start|>",
        "<|box_end|>",
        "<|quad_start|>",
        "<|quad_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|vision_pad|>",
        "<|image_pad|>",
        "<|video_pad|>",
        "<tool_call>",
        "</tool_call>",
        "<|fim_prefix|>",
        "<|fim_middle|>",
        "<|fim_suffix|>",
        "<|fim_pad|>",
        "<|repo_name|>",
        "<|file_sep|>",
        "<tool_response>",
        "</tool_response>",
        "<think>",
        "</think>",
    };
    // Everything above is 151643-151668 and is IDENTICAL in every Qwen2/3 vocabulary
    // this engine loads, MiniMax-H3's included (verified token-by-token against the
    // shipped FL2VA/tokenizer/ with transformers 4.57.3). Only the tail differs, and
    // because load_from_merges() assigns ids by POSITION, the tail cannot be a union
    // of both sets -- appending one would shift the other's ids off the trained
    // embeddings. Select it instead.
    switch (special_tail) {
        case SpecialTail::MINIMAX_H3:
            // 151669-151675, in this exact order, from MiniMax-H3's
            // FL2VA/tokenizer/tokenizer_config.json `additional_special_tokens`.
            // These are absent from vocab.json, so HF appends them at load time and
            // they land here; vocab_size 151936 means the embeddings are trained.
            // <d> ... </d> is the wrapper H3 keys ALL spoken and sung content on --
            // without it a dialogue line byte-BPEs to ordinary tokens and the trained
            // id is never emitted (README.md:119, and MiniMax's own ref2va request).
            special_tokens.insert(special_tokens.end(),
                                  {
                                      "<d>",
                                      "</d>",
                                      "<|cutoff|>",
                                      "<|lyrics_start|>",
                                      "<|lyrics_end|>",
                                      "<|caption_start|>",
                                      "<|caption_end|>",
                                  });
            break;
        case SpecialTail::DEFAULT:
        default:
            // 151669-151673. HiDream-O1 encodes <|boi_token|><|tms_token|> as literal
            // prompt text, so these must keep their ids for every non-H3 consumer.
            special_tokens.insert(special_tokens.end(),
                                  {
                                      "<|boi_token|>",
                                      "<|bor_token|>",
                                      "<|eor_token|>",
                                      "<|bot_token|>",
                                      "<|tms_token|>",
                                  });
            break;
    }

    if (merges_utf8_str.size() > 0) {
        load_from_merges(merges_utf8_str);
    } else {
        load_from_merges(load_qwen2_merges());
    }
}
