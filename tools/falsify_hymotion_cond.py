#!/usr/bin/env python3
"""
FALSIFIER #2 -- the conditioning tensors themselves. Runs on dumps from
`hymotion --encode-only --dump-cond PFX`, so it needs no GPU of its own.

This is the check that "the tensors load and the shapes match" cannot be. It asks:
does the text encoder actually READ the prompt?

The sharpest test here is CHECK 5. Every prompt's token stream ends with the same
two tokens -- `<|im_end|>` and a newline -- because the chat template appends them.
So the row at ctxt_length-2 is the SAME TOKEN for every prompt. Its hidden state may
only differ because Qwen3 attended to the preceding user text. Therefore:

    if that row is identical across two different prompts, the encoder is not
    conditioning on the prompt at all -- and every shape would still be perfect.

An encoder that returned a constant, or that cropped at the wrong offset and handed
back system-prompt states, or that never ran the user tokens, all pass a shape check
and all FAIL check 5.

Usage:
    # main agent runs the two encodes first (see the report), then:
    uv run --with numpy python3 tools/falsify_hymotion_cond.py \
        --a /tmp/hm_kick --b /tmp/hm_barbell [--repeat-of-a /tmp/hm_kick2]
"""
import argparse
import os
import sys

import numpy as np

CTXT_DIM = 4096
CTXT_ROWS = 128
VTXT_DIM = 768


def load(prefix):
    meta = {}
    with open(prefix + ".meta.txt") as f:
        for line in f:
            k, _, v = line.strip().partition("=")
            meta[k] = v
    ctxt = np.fromfile(prefix + ".ctxt.f32", dtype=np.float32)
    vtxt = np.fromfile(prefix + ".vtxt.f32", dtype=np.float32)
    rows = int(meta.get("ctxt_rows", CTXT_ROWS))
    dim = int(meta.get("ctxt_dim", CTXT_DIM))
    if ctxt.size != rows * dim:
        raise SystemExit(f"FATAL: {prefix}.ctxt.f32 has {ctxt.size} floats, expected {rows*dim}")
    return meta, ctxt.reshape(rows, dim), vtxt


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


class R:
    n = 0

    @classmethod
    def check(cls, name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
        if not ok:
            cls.n += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="dump prefix for prompt A")
    ap.add_argument("--b", required=True, help="dump prefix for prompt B (a DIFFERENT prompt)")
    ap.add_argument("--repeat-of-a", help="optional: a second dump of prompt A, for determinism")
    args = ap.parse_args()

    ma, ca, va = load(args.a)
    mb, cb, vb = load(args.b)
    la, lb = int(ma["ctxt_length"]), int(mb["ctxt_length"])
    print(f"A: {ma['prompt']!r}  ctxt_length={la}")
    print(f"B: {mb['prompt']!r}  ctxt_length={lb}")
    if ma["prompt"] == mb["prompt"]:
        raise SystemExit("FATAL: --a and --b must be DIFFERENT prompts, else this proves nothing")

    print("\n== CHECK 1: the conditioning is finite and non-degenerate ==")
    for tag, c, l in (("A", ca, la), ("B", cb, lb)):
        R.check(f"{tag} ctxt all finite", bool(np.isfinite(c).all()),
                f"{int((~np.isfinite(c)).sum())} non-finite")
        valid = c[:l]
        R.check(f"{tag} valid rows are non-zero", float(np.abs(valid).max()) > 0,
                f"max|x|={float(np.abs(valid).max()):.4f}")
        R.check(f"{tag} valid rows are not all identical",
                float(np.std(valid, axis=0).mean()) > 1e-6,
                f"mean std across rows={float(np.std(valid, axis=0).mean()):.5f}")
    for tag, v in (("A", va), ("B", vb)):
        R.check(f"{tag} vtxt finite and non-zero, {VTXT_DIM}-d",
                v.size == VTXT_DIM and bool(np.isfinite(v).all()) and float(np.abs(v).max()) > 0,
                f"n={v.size} max|x|={float(np.abs(v).max()):.4f}")

    print("\n== CHECK 2: rows past ctxt_length are exactly zero (our padding contract) ==")
    # These rows are masked by the DiT (masked mean + mask_1&mask_2 attention), so their
    # content is irrelevant -- but if they are NOT zero, our zero-fill assumption and the
    # code that produced them disagree, and one of them is wrong.
    for tag, c, l in (("A", ca, la), ("B", cb, lb)):
        tail = c[l:]
        R.check(f"{tag} rows[{l}:] == 0", float(np.abs(tail).max()) == 0.0,
                f"max|tail|={float(np.abs(tail).max()):.6f}")

    print("\n== CHECK 3: different prompts produce different conditioning ==")
    pa, pb = ca[:la].mean(axis=0), cb[:lb].mean(axis=0)
    c_ctxt = cos(pa, pb)
    R.check("cos(mean-pooled ctxt A, B) is well below 1.0", c_ctxt < 0.99, f"cos={c_ctxt:.6f}")
    c_vtxt = cos(va, vb)
    R.check("cos(vtxt A, vtxt B) is well below 1.0", c_vtxt < 0.99, f"cos={c_vtxt:.6f}")

    print("\n== CHECK 4: ctxt_length matches the token count the tokenizer selftest predicts ==")
    R.check("A and B have different ctxt_length (different prompt lengths)", la != lb,
            f"{la} vs {lb} -- if equal, check the prompts really differ in length")

    print("\n== CHECK 5: the SAME token in a different context has a DIFFERENT state ==")
    print("   Both prompts end with <|im_end|>\\n, so row[ctxt_length-2] is the identical")
    print("   token in both. It can only differ via attention over the user text.")
    if la >= 2 and lb >= 2:
        ra, rb = ca[la - 2], cb[lb - 2]
        c_end = cos(ra, rb)
        R.check("<|im_end|> row differs across prompts", c_end < 0.999, f"cos={c_end:.6f}")
        R.check("<|im_end|> row is not bit-identical across prompts",
                not np.array_equal(ra, rb),
                "identical => the encoder is ignoring the prompt entirely")
    else:
        R.check("<|im_end|> row comparison", False, f"ctxt_length too small ({la},{lb})")

    if args.repeat_of_a:
        print("\n== CHECK 6: determinism -- the same prompt twice is bit-identical ==")
        mc, cc, vc = load(args.repeat_of_a)
        if mc["prompt"] != ma["prompt"]:
            raise SystemExit("FATAL: --repeat-of-a must use the SAME prompt as --a")
        R.check("ctxt bit-identical across runs", np.array_equal(ca, cc),
                f"max|diff|={float(np.abs(ca-cc).max()):.3e}")
        R.check("vtxt bit-identical across runs", np.array_equal(va, vc),
                f"max|diff|={float(np.abs(va-vc).max()):.3e}")

    print()
    if R.n:
        print(f"FALSIFIED: {R.n} check(s) FAILED")
        return 1
    print("All checks passed: the encoder reads the prompt, and the padding contract holds.")
    print("NOT PROVEN by this script: that the hidden-state VALUES match HF's Qwen3-8B.")
    print("That needs a torch reference this box does not have. See the report.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
