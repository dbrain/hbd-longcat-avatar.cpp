#ifndef __SD_TOKENIZERS_QWEN2_TOKENIZER_H__
#define __SD_TOKENIZERS_QWEN2_TOKENIZER_H__

#include <string>

#include "bpe_tokenizer.h"

class Qwen2Tokenizer : public BPETokenizer {
protected:
    void load_from_merges(const std::string& merges_utf8_str);

public:
    // Everything up to and including </think> (id 151668) is common to every Qwen2/3
    // vocabulary this engine loads. PAST that point the table is NOT shared: each
    // checkpoint appends its own `additional_special_tokens`, and ids here are
    // POSITIONAL -- an entry's index in `special_tokens` IS its token id -- so the
    // tail has to be selected per checkpoint family rather than unioned.
    enum class SpecialTail {
        // <|boi_token|> .. <|tms_token|> at 151669-151673. HiDream-O1 emits
        // <|boi_token|><|tms_token|> literally (hidream_o1.hpp:538,623); every other
        // consumer never encodes these, so this stays the default.
        DEFAULT,
        // <d> .. <|caption_end|> at 151669-151675, per MiniMax-H3's shipped
        // FL2VA/tokenizer/tokenizer_config.json. Displaces the five above.
        MINIMAX_H3,
    };

    explicit Qwen2Tokenizer(const std::string& merges_utf8_str = "",
                            SpecialTail special_tail           = SpecialTail::DEFAULT);
};

#endif  // __SD_TOKENIZERS_QWEN2_TOKENIZER_H__
