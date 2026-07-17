#!/usr/bin/env python3
"""
FALSIFIER for the HY-Motion text-encoder contract. Designed to FAIL, loudly.

Why this exists
---------------
"The tensors load and the shapes match" is a check that CANNOT FAIL. Both real bugs
found in this port so far had perfectly matching shapes (`d_head = 64/20 = 3` by
integer division; two 488x488 masks that `ggml_add` broadcast against each other).
Shape-compatible does not mean correct.

So this compares our C++ against something INDEPENDENT:

  our C++ Qwen2Tokenizer  builds its vocab from src/tokenizers/vocab/qwen_merges.hpp
                          (byte encoder + merge list + a hardcoded special-token order)
  the Qwen3-8B GGUF       carries HF's OWN explicit id -> token table

Our C++ never reads the GGUF's token table. If the two disagree by ONE id, crop_start
shifts and the DiT is silently fed the SYSTEM PROMPT's hidden states instead of the
user's -- at a perfectly valid [128, 4096] shape, with no error anywhere.

TEST 1  vocab parity      our id -> GGUF token, byte-decoded, must rebuild the exact
                          templated string byte-for-byte.
TEST 2  crop_start        marker-algorithm (upstream's) == |tokenize(prefix)| (ours),
                          AND the tokens before crop_start must decode to exactly the
                          chat prefix, AND the tokens at/after it must start with the
                          user's text.
TEST 3  prompt sensitivity  a different prompt must move crop_start NOT AT ALL and
                          must change the token tail.
TEST 4  CLIP pooling idx  the pooled position must be the FIRST 49407 and must equal
                          argmax(input_ids) -- HF's actual rule (modeling_clip.py:976-979).

Each test prints PASS/FAIL. Exit code 1 if any fail.

Usage:
    uv run --with gguf --with numpy python3 tools/falsify_hymotion_text.py \
        --bin  /path/to/build/bin/hymotion \
        --qwen3 /home/dbrain/dev/flux2.cpp/models/text_encoders/Qwen3-8B-UD-Q4_K_XL.gguf

No GPU. No weights loaded. `--selftest-text` never initialises a backend.
"""
import argparse
import subprocess
import sys

# ---------------------------------------------------------------------------
# GPT-2 / Qwen byte-level BPE: the inverse of bytes_to_unicode()
# ---------------------------------------------------------------------------


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + list(
        range(ord("\xae"), ord("\xff") + 1)
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


BYTE_ENCODER = bytes_to_unicode()
BYTE_DECODER = {v: k for k, v in BYTE_ENCODER.items()}


def token_to_bytes(tok: str) -> bytes:
    """GGUF stores byte-level-BPE tokens in the 'unicode' alphabet; invert it."""
    out = bytearray()
    for ch in tok:
        if ch not in BYTE_DECODER:
            raise KeyError(f"char {ch!r} (U+{ord(ch):04X}) not in the byte-level alphabet")
        out.append(BYTE_DECODER[ch])
    return bytes(out)


# ---------------------------------------------------------------------------
# An INDEPENDENT reference tokenizer.
#
# This is the check with teeth. A byte round-trip can pass with wrong ids (two
# different tokenisations of "    Summarize" decode to the same bytes but are
# different token sequences, and the DiT sees the ids, not the bytes). So we
# re-implement Qwen2/3 BPE here from the real regex + the GGUF's OWN merge list,
# and compare IDS against the C++.
#
# Qwen2 pre-tokenizer regex (tokenizer.json, "pre_tokenizer": Split pattern):
#   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
# ---------------------------------------------------------------------------

QWEN_PAT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


class RefTokenizer:
    def __init__(self, vocab, merges, specials):
        self.encoder = {t: i for i, t in enumerate(vocab)}
        self.ranks = {tuple(m.split(" ")): i for i, m in enumerate(merges)}
        self.specials = specials
        import regex

        self.pat = regex.compile(QWEN_PAT)
        # longest-first so <|im_start|> wins over any prefix
        self.special_pat = regex.compile(
            "(" + "|".join(regex.escape(s) for s in sorted(specials, key=len, reverse=True)) + ")"
        )

    def _bpe(self, word):
        parts = list(word)
        if len(parts) == 1:
            return parts
        while True:
            best, best_rank = None, None
            for i in range(len(parts) - 1):
                r = self.ranks.get((parts[i], parts[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = i, r
            if best is None:
                break
            parts[best : best + 2] = [parts[best] + parts[best + 1]]
        return parts

    def encode(self, text):
        ids = []
        for chunk in self.special_pat.split(text):
            if not chunk:
                continue
            if chunk in self.specials:
                ids.append(self.encoder[chunk])
                continue
            for piece in self.pat.findall(chunk):
                word = "".join(BYTE_ENCODER[b] for b in piece.encode("utf-8"))
                for sub in self._bpe(word):
                    if sub not in self.encoder:
                        raise KeyError(f"BPE produced {sub!r}, absent from the vocab")
                    ids.append(self.encoder[sub])
        return ids


# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "\n    Summarize human motion only from the user text for representation: action "
    "categories, key body-part movements, order/transitions, trajectory/direction, posture; "
    "include style/emotion/speed only if present. Explicitly capture laterality (left/right) "
    "when mentioned; do not guess. If multiple actions are described, indicate the count of "
    "distinct actions (e.g., actions=3) and their order. Do not invent missing info. Keep one "
    "concise paragraph.\n"
)
SYSTEM_PROMPT_SHA = "b9c55290b778c01739a337a36315c2065ef6ccfbfa0a1898b727377ed4965283"


def chat_prefix():
    return "<|im_start|>system\n" + SYSTEM_PROMPT + "<|im_end|>\n<|im_start|>user\n"


def chat_string(user):
    return chat_prefix() + user + "<|im_end|>\n"


def run_selftest(binary, prompt):
    r = subprocess.run([binary, "--selftest-text", "--prompt", prompt],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        raise SystemExit(f"FATAL: {binary} --selftest-text exited {r.returncode}")
    out = {}
    for line in r.stdout.splitlines():
        if not line.strip():
            continue
        k, _, v = line.partition(" ")
        out[k] = v
    return out


class Results:
    def __init__(self):
        self.failed = 0

    def check(self, name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
        if not ok:
            self.failed += 1
        return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="path to the hymotion binary")
    ap.add_argument("--qwen3", required=True, help="path to the Qwen3-8B gguf (for its vocab)")
    ap.add_argument("--prompt", default="kick a ball")
    ap.add_argument("--prompt2", default="pushes a barbell overhead")
    args = ap.parse_args()

    import hashlib

    from gguf import GGUFReader

    R = Results()

    print("== TEST 0: the system prompt literal is byte-exact ==")
    R.check("SYSTEM_PROMPT sha256",
            hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest() == SYSTEM_PROMPT_SHA,
            "this script's copy matches model_constants.py")
    st = run_selftest(args.bin, args.prompt)
    got_len, want_len = st["SYSTEM_PROMPT_LEN"].split()
    R.check("C++ SYSTEM_PROMPT length", got_len == want_len == "449", f"C++={got_len} expect={want_len}")
    R.check("C++ chat string length", int(st["CHAT_BYTES"]) == len(chat_string(args.prompt)),
            f"C++={st['CHAT_BYTES']} python={len(chat_string(args.prompt))}")

    print("\n== loading the GGUF's own vocab (independent of our C++) ==")
    r = GGUFReader(args.qwen3)
    toks = r.fields["tokenizer.ggml.tokens"]
    vocab = [str(bytes(toks.parts[idx]), "utf-8") for idx in toks.data]
    print(f"  GGUF vocab: {len(vocab)} tokens")

    ids = [int(x) for x in st["IDS"].split()]
    crop = int(st["CROP_START"])
    print(f"  our C++: {len(ids)} tokens, crop_start={crop}  ({st['CROP_REPORT']})")

    print("\n== TEST 1: our token ids, decoded through the GGUF's table, rebuild the string ==")
    specials = {}
    for i, t in enumerate(vocab):
        if t.startswith("<|") or t in ("<think>", "</think>", "<tool_call>", "</tool_call>",
                                        "<tool_response>", "</tool_response>"):
            specials[i] = t
    rebuilt = bytearray()
    bad = None
    for tid in ids:
        if tid >= len(vocab):
            bad = f"id {tid} >= vocab {len(vocab)}"
            break
        if tid in specials:
            rebuilt += specials[tid].encode()
        else:
            try:
                rebuilt += token_to_bytes(vocab[tid])
            except KeyError as e:
                bad = f"id {tid} token {vocab[tid]!r}: {e}"
                break
    if bad:
        R.check("byte-decode our ids via GGUF vocab", False, bad)
    else:
        want = chat_string(args.prompt).encode()
        ok = bytes(rebuilt) == want
        detail = "exact" if ok else f"got {len(rebuilt)}B want {len(want)}B"
        if not ok:
            for i in range(min(len(rebuilt), len(want))):
                if rebuilt[i] != want[i]:
                    detail += f"; first diff at byte {i}: {bytes(rebuilt[i:i+24])!r} vs {want[i:i+24]!r}"
                    break
        R.check("our ids -> GGUF vocab -> exact templated string", ok, detail)

    print("\n== TEST 1b: our ids == an INDEPENDENT reference BPE's ids ==")
    print("   (the byte round-trip above CANNOT catch a mis-split: '    Summarize' as")
    print("    ['    ','Summarize'] and ['   ',' Summarize'] decode to identical bytes")
    print("    but are different ids, and the DiT consumes ids.)")
    merges_f = r.fields.get("tokenizer.ggml.merges")
    merges = [str(bytes(merges_f.parts[idx]), "utf-8") for idx in merges_f.data]
    ref = RefTokenizer(vocab, merges, set(specials.values()))
    ref_ids = ref.encode(chat_string(args.prompt))
    ok = ref_ids == ids
    detail = f"{len(ids)} tokens" if ok else f"C++ {len(ids)} vs ref {len(ref_ids)}"
    if not ok:
        for i in range(min(len(ids), len(ref_ids))):
            if ids[i] != ref_ids[i]:
                detail += (f"; first diff at token {i}: C++ {ids[i]}={vocab[ids[i]]!r} "
                           f"vs ref {ref_ids[i]}={vocab[ref_ids[i]]!r}")
                break
    R.check("C++ token ids == reference BPE token ids", ok, detail)
    R.check("reference agrees on crop_start", len(ref.encode(chat_prefix())) == crop,
            f"ref={len(ref.encode(chat_prefix()))} C++={crop}")

    print("\n== TEST 2: crop_start is where the user's text actually begins ==")
    R.check("marker algorithm == prefix length", "not found" not in st["CROP_REPORT"],
            st["CROP_REPORT"])
    pre = bytearray()
    for tid in ids[:crop]:
        pre += specials[tid].encode() if tid in specials else token_to_bytes(vocab[tid])
    R.check("tokens[:crop_start] == chat prefix exactly",
            bytes(pre) == chat_prefix().encode(),
            f"{len(pre)}B vs {len(chat_prefix().encode())}B")
    post = bytearray()
    for tid in ids[crop:]:
        post += specials[tid].encode() if tid in specials else token_to_bytes(vocab[tid])
    R.check("tokens[crop_start:] starts with the user prompt",
            bytes(post).startswith(args.prompt.encode()),
            repr(bytes(post)[:48]))
    # ctxt_length the C++ will compute
    ctxt_len = min(len(ids) - crop, 128)
    R.check("ctxt_length is sane (0 < n <= 128)", 0 < ctxt_len <= 128, f"ctxt_length={ctxt_len}")
    print(f"       -> the DiT sees {ctxt_len} text tokens: prompt + <|im_end|> + newline")

    print("\n== TEST 3: a different prompt changes the tail, not the prefix ==")
    st2 = run_selftest(args.bin, args.prompt2)
    ids2 = [int(x) for x in st2["IDS"].split()]
    crop2 = int(st2["CROP_START"])
    R.check("crop_start is prompt-independent", crop == crop2, f"{crop} vs {crop2}")
    R.check("prefix tokens identical", ids[:crop] == ids2[:crop2])
    R.check("tail tokens differ", ids[crop:] != ids2[crop2:],
            f"{len(ids)-crop} vs {len(ids2)-crop2} tokens")

    print("\n== TEST 4: CLIP pools at HF's argmax(input_ids) ==")
    cids = [int(x) for x in st["CLIP_IDS"].split()]
    eos_idx = int(st["CLIP_EOS_IDX"])
    R.check("CLIP padded to 77", len(cids) == 77, f"n={len(cids)}")
    R.check("pooled idx == first 49407", eos_idx == cids.index(49407), f"idx={eos_idx}")
    # HF: pooled_output = last_hidden[argmax(input_ids)] because EOS is the max id.
    R.check("pooled idx == argmax(input_ids) (HF's actual rule)",
            eos_idx == max(range(len(cids)), key=lambda i: cids[i]),
            f"argmax={max(range(len(cids)), key=lambda i: cids[i])}")
    R.check("BOS is 49406", cids[0] == 49406, f"cids[0]={cids[0]}")

    print()
    if R.failed:
        print(f"FALSIFIED: {R.failed} check(s) FAILED")
        return 1
    print("All checks passed.")
    print("NOTE: this validates the TEMPLATE/CROP/TOKENISER contract -- the part that fails")
    print("      silently. It does NOT validate Qwen3's hidden-state numerics; nothing on this")
    print("      box can (no torch, no reference Qwen3-8B). See the report.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
