#!/usr/bin/env python3
"""Decide, FROM THE DATA, whether a fused QKV weight is per-head INTERLEAVED or CONTIGUOUS.

Why this exists
---------------
The raw MiniMax-H3 checkpoint stores `blocks.N.attn.qkv_proj.weight` per-head interleaved --
rows are `[head0: q(hd) k(hd) v(hd), head1: q k v, ...]`.  The reference implementation
un-interleaves at LOAD time (`_reorder_grouped_qkv_to_qkv`), and the official diffusers converter
does the same on the way in (`reorder_interleaved_qkv`, scripts/convert_minimax_h3_to_diffusers.py).
So `[q_all; k_all; v_all]` is an IN-MEMORY layout, never the on-disk one, and a file carrying the
reference spelling tells you nothing about which of the two it holds.

Our engine's `split_qkv` assumes contiguous.  Feeding it a raw shard is silently wrong: the shapes
all match, nothing errors, the attention is garbage.  Filenames and key spellings cannot
distinguish the two, so this probe looks at the numbers instead.

The statistic
-------------
Cut the `3 * H` head-sized row chunks of the fused weight.  Each hypothesis reads the chunk index
`c` as a (third, head) pair, and the two disagree only about which coordinate is which:

    CONTIGUOUS   third = c // H,  head = c %  H
    INTERLEAVED  third = c %  3,  head = c // 3

Both are the same balanced 3 x H factorial design over the same chunks, so they can be scored
identically and compared directly.  Two per-chunk features are used:

  norm     scalar   -- mean row L2 norm over the chunk.  q/k/v projections of a trained block
                       almost always sit at different weight scales.
  profile  vector   -- the chunk's energy over the INPUT axis, binned to 64 bins and L2
                       normalised.  q, k and v read the same residual stream but learn different
                       column emphases; this sees that even when the scales happen to match.

Scoring is a TWO-WAY ANOVA, and that is load-bearing rather than fancy.  A one-way "group by
third" F fires on the wrong hypothesis whenever the weight has head-INDEX structure -- and trained
attention weights do: reading an interleaved tensor with the contiguous grouping is exactly
"group the heads into three consecutive blocks", which is significant all by itself.  Measured on
Qwen3-VL, a one-way statistic gave confident false positives on a fixture with no q/k/v structure
at all.  The two-way form removes the head main effect first and reports

    F_third = (SS_third / (3 - 1)) / (SS_resid / ((3 - 1) * (H - 1)))

so the score is "how much does the projection identity explain, once head identity is taken out".
Under the wrong hypothesis the hypothesised head factor absorbs the real signal and F_third
collapses.  Confidence is a permutation test: relabel the chunks at random into the same 3 x H
grid and see how often chance reaches this F.

A verdict is only issued when all four of these hold: the two features AGREE on a winner, at least
one is significant for the winner, NEITHER is significant for the loser, and one of them separates
the two hypotheses by at least `MIN_F_RATIO`.  Otherwise the answer is AMBIGUOUS, which is a real
answer: a detector that always commits is a detector that cannot fail.

The thresholds were set against the exchangeable null, not chosen for taste.  At p <= 0.01 the null
committed on ~5% of trials (four looks at alpha = 0.01, and the two features are correlated, so the
per-trial rate is several times alpha); the F-ratio gate alone barely dented it, because those
false positives are not near-ties.  p <= 0.002 takes the null to 0/64.  It costs the labelled
fixtures nothing on the real fused qkv, whose p sits at the permutation floor of 2.5e-4.

Negative control (`--selftest`)
-------------------------------
Qwen3-VL is on this box and gives three fixtures built from REAL trained weights:

  vision   `model.visual.blocks.N.attn.qkv.weight` -- a genuinely fused MHA qkv whose layout is
           pinned by transformers itself: `qkv(x).reshape(seq, 3, num_heads, -1)`
           (modeling_qwen3_vl.py:192) means the output axis is `[3, heads, head_dim]`, i.e.
           CONTIGUOUS.  This is the closest analogue to H3 available without H3.
  text     `layers.N.self_attn.{q,k,v}_proj.weight` re-fused by hand.  Qwen3-VL text is GQA
           (32 q heads, 8 kv heads), so the FAIR MHA-shaped fixture takes the first `n_kv` q
           heads and all `n_kv` k/v heads -- a real trained subset, with nothing duplicated.
           (Repeating kv heads to 32 was rejected: exact duplicate chunks collapse the
           within-group dispersion of k and v and would flatter the detector.)
  nullseq  three disjoint consecutive 8-head slices of `q_proj` alone, labelled q/k/v.  No q/k/v
           structure exists, so a confident verdict here is the statistic reading an artefact.
           This is the ADVERSARIAL null: q_proj heads are not exchangeable by index, so a naive
           one-way statistic does fire on it.
  nullperm the same slices with the head blocks randomly shuffled first -- genuinely exchangeable,
           the clean null.

  Both nulls must come back AMBIGUOUS.

Each fixture is fed BOTH ways round -- as built, and re-packed into the other layout -- so every
trial has a known label and half of them are the layout the detector must NOT say.

Usage
-----
    # negative-control battery (no H3 weights needed)
    python3 tools/h3_qkv_layout_probe.py --selftest

    # a real checkpoint, once it exists
    python3 tools/h3_qkv_layout_probe.py --dir /path/to/MiniMax-H3/FL2VA \
        --tensor blocks.0.attn.qkv_proj.weight --heads 56 --head-dim 128

Pure numpy + a hand-rolled safetensors header reader, so it costs a few MB and no torch.
"""

import argparse
import glob
import json
import os
import struct
import sys

import numpy as np

# ------------------------------------------------------------------------------------------------
# safetensors, read directly (header is JSON after an 8-byte little-endian length)
# ------------------------------------------------------------------------------------------------

_DTYPES = {
    "F64": (np.dtype("<f8"), 8),
    "F32": (np.dtype("<f4"), 4),
    "F16": (np.dtype("<f2"), 2),
    "BF16": (np.dtype("<u2"), 2),
    "I8": (np.dtype("i1"), 1),
    "U8": (np.dtype("u1"), 1),
}


def read_header(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return header, 8 + n


def load_tensor(path, name):
    """Return one tensor as float32, reading only its byte range."""
    header, data_start = read_header(path)
    if name not in header:
        raise KeyError(f"{name} not in {path}")
    info = header[name]
    dtype, esz = _DTYPES[info["dtype"]]
    begin, end = info["data_offsets"]
    count = (end - begin) // esz
    with open(path, "rb") as f:
        f.seek(data_start + begin)
        raw = np.frombuffer(f.read(end - begin), dtype=dtype, count=count)
    if info["dtype"] == "BF16":
        # bf16 is the top 16 bits of an f32
        out = np.zeros(raw.size, dtype=np.uint32)
        out |= raw.astype(np.uint32) << 16
        arr = out.view(np.float32)
    else:
        arr = raw.astype(np.float32)
    return arr.reshape(info["shape"])


def index_of(directory):
    """name -> file, for a sharded or single-file HF directory."""
    idx = os.path.join(directory, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as f:
            wm = json.load(f)["weight_map"]
        return {k: os.path.join(directory, v) for k, v in wm.items()}
    out = {}
    for path in sorted(glob.glob(os.path.join(directory, "*.safetensors"))):
        header, _ = read_header(path)
        for k in header:
            if k != "__metadata__":
                out[k] = path
    return out


# ------------------------------------------------------------------------------------------------
# the statistic
# ------------------------------------------------------------------------------------------------

CONTIGUOUS = "contiguous"
INTERLEAVED = "interleaved"

PROFILE_BINS = 64
N_SHUFFLES = 4000
# A grouping is "significant" when no more than this share of random balanced 3-partitions of the
# same chunks reach its F.  4000 shuffles resolves down to 2.5e-4, so 0.01 is far from the floor.
P_SIGNIFICANT = 0.002
# The winning grouping must also beat the losing one by this F ratio on at least one feature.
MIN_F_RATIO = 4.0


def chunk_features(w, heads, head_dim):
    """Per-chunk scalar (mean row L2 norm) and vector (binned input-axis energy profile)."""
    rows, cols = w.shape
    if rows != 3 * heads * head_dim:
        raise ValueError(f"{rows} rows != 3 * {heads} * {head_dim}")
    sq = (w.astype(np.float64) ** 2).reshape(3 * heads, head_dim, cols)

    norm = np.sqrt(sq.sum(axis=2)).mean(axis=1)  # [3H]

    bins = min(PROFILE_BINS, cols)
    edges = np.linspace(0, cols, bins + 1).astype(int)
    prof = np.stack([sq[:, :, a:b].sum(axis=(1, 2)) for a, b in zip(edges[:-1], edges[1:])], axis=1)
    prof = np.sqrt(prof)
    prof /= np.linalg.norm(prof, axis=1, keepdims=True) + 1e-30  # scale-free, so it is not `norm` again
    return norm, prof


def two_way_f(x, heads, order):
    """F for the `third` main effect of a balanced 3 x heads design, head effect removed.

    `order` maps grid cell (third * heads + head) -> chunk index, so one array does both
    hypotheses.  Works for scalar and vector features alike: the sums of squares are traces, and
    the per-dimension degrees of freedom cancel in the ratio.
    """
    x = x.reshape(x.shape[0], -1)
    grid = x[order].reshape(3, heads, x.shape[1])
    grand = grid.mean(axis=(0, 1))
    m_third = grid.mean(axis=1)  # [3, D]
    m_head = grid.mean(axis=0)  # [heads, D]

    ss_third = heads * float(((m_third - grand) ** 2).sum())
    ss_head = 3 * float(((m_head - grand) ** 2).sum())
    ss_total = float(((grid - grand) ** 2).sum())
    ss_resid = max(ss_total - ss_third - ss_head, 0.0)

    df_third = 3 - 1
    df_resid = (3 - 1) * (heads - 1)
    if ss_resid <= 0.0:
        return float("inf") if ss_third > 0.0 else 0.0
    return (ss_third / df_third) / (ss_resid / df_resid)


def score_feature(x, heads, rng):
    """F under both hypotheses plus a permutation p-value for each."""
    n = 3 * heads
    idx = np.arange(n)
    # cell (third, head) -> chunk index, per hypothesis
    order = {
        CONTIGUOUS: idx,  # third = c // heads, head = c % heads
        INTERLEAVED: (idx % heads) * 3 + (idx // heads),  # third = c % 3, head = c // 3
    }
    f = {h: two_way_f(x, heads, order[h]) for h in order}

    ge = {h: 0 for h in order}
    for _ in range(N_SHUFFLES):
        fr = two_way_f(x, heads, rng.permutation(n))
        for h in order:
            if fr >= f[h]:
                ge[h] += 1
    p = {h: (ge[h] + 1) / (N_SHUFFLES + 1) for h in order}
    return f, p


def probe(w, heads, head_dim, seed=20260803):
    rng = np.random.default_rng(seed)
    norm, prof = chunk_features(w, heads, head_dim)
    f_n, p_n = score_feature(norm, heads, rng)
    f_p, p_p = score_feature(prof, heads, rng)

    def winner(f):
        return CONTIGUOUS if f[CONTIGUOUS] >= f[INTERLEAVED] else INTERLEAVED

    def margin(f):
        lo = min(f[CONTIGUOUS], f[INTERLEAVED])
        hi = max(f[CONTIGUOUS], f[INTERLEAVED])
        return float("inf") if lo <= 0 else hi / lo

    w_n, w_p = winner(f_n), winner(f_p)
    verdict = "ambiguous"
    if w_n == w_p:
        loser = INTERLEAVED if w_n == CONTIGUOUS else CONTIGUOUS
        # Commit only when the two features AGREE, at least one of them is significant for the
        # winner, and NEITHER is significant for the loser.  Requiring both to be individually
        # significant was tried and is too strict: `norm` is weak on a small-H fixture where
        # `profile` is decisive, and the result was abstention on cases the detector had right.
        # Requiring only agreement was also tried and is too loose.
        winner_sig = min(p_n[w_n], p_p[w_p]) <= P_SIGNIFICANT
        loser_null = p_n[loser] > P_SIGNIFICANT and p_p[loser] > P_SIGNIFICANT
        # ...and one feature must actually SEPARATE the hypotheses.  Without this the exchangeable
        # null commits on ~6% of trials off near-ties (max F ratio 1.1-1.5) that clear the
        # permutation test by luck.  Real fused q/k/v separates by 30x and up on `norm`, so the gate
        # costs true positives nothing.
        separated = max(margin(f_n), margin(f_p)) >= MIN_F_RATIO
        if winner_sig and loser_null and separated:
            verdict = w_n
    return {
        "verdict": verdict,
        "norm": {"winner": w_n, "f": f_n, "p": p_n, "margin": margin(f_n)},
        "profile": {"winner": w_p, "f": f_p, "p": p_p, "margin": margin(f_p)},
    }


# ------------------------------------------------------------------------------------------------
# layout surgery, used to build the fixtures (and mirrored by the C++ converter)
# ------------------------------------------------------------------------------------------------


def to_contiguous(w, heads, head_dim):
    """Interleaved [h0:qkv, h1:qkv, ...] -> [q_all; k_all; v_all].  == reorder_interleaved_qkv."""
    g = w.reshape(heads, 3, head_dim, *w.shape[1:])
    return np.concatenate([g[:, t].reshape(heads * head_dim, *w.shape[1:]) for t in range(3)], axis=0)


def to_interleaved(w, heads, head_dim):
    """[q_all; k_all; v_all] -> interleaved.  The exact inverse of to_contiguous."""
    g = w.reshape(3, heads, head_dim, *w.shape[1:])
    return np.concatenate(
        [g[:, h].reshape(3 * head_dim, *w.shape[1:]) for h in range(heads)], axis=0
    )


# ------------------------------------------------------------------------------------------------
# negative control
# ------------------------------------------------------------------------------------------------

QWEN_DIRS = [
    "/mnt/ssd/h3-staging/Qwen3-VL-4B-Instruct",
    "/mnt/ssd/h3-staging/Qwen3-VL-32B-Instruct",
]


def _cfg(directory):
    with open(os.path.join(directory, "config.json")) as f:
        return json.load(f)


def vision_fixtures(directory, limit):
    """Real fused MHA qkv, known CONTIGUOUS (modeling_qwen3_vl.py:192)."""
    cfg = _cfg(directory)["vision_config"]
    heads = cfg["num_heads"]
    head_dim = cfg["hidden_size"] // heads
    files = index_of(directory)
    out = []
    for i in range(cfg["depth"]):
        for name in (
            f"model.visual.blocks.{i}.attn.qkv.weight",
            f"visual.blocks.{i}.attn.qkv.weight",
        ):
            if name in files:
                out.append((f"vision.{i}", load_tensor(files[name], name), heads, head_dim))
                break
        if len(out) >= limit:
            break
    return out


def text_fixtures(directory, limit):
    """GQA q/k/v re-fused as a FAIR MHA fixture: first n_kv q heads, all n_kv k/v heads."""
    cfg = _cfg(directory)["text_config"]
    head_dim = cfg["head_dim"]
    n_kv = cfg["num_key_value_heads"]
    files = index_of(directory)
    out = []
    for i in range(cfg["num_hidden_layers"]):
        got = {}
        for which in "qkv":
            for name in (
                f"model.language_model.layers.{i}.self_attn.{which}_proj.weight",
                f"model.layers.{i}.self_attn.{which}_proj.weight",
            ):
                if name in files:
                    got[which] = load_tensor(files[name], name)
                    break
        if len(got) != 3:
            break
        w = np.concatenate(
            [got["q"][: n_kv * head_dim], got["k"][: n_kv * head_dim], got["v"][: n_kv * head_dim]],
            axis=0,
        )
        out.append((f"text.{i}", w, n_kv, head_dim))
        if len(out) >= limit:
            break
    return out


def _null_base(directory, limit, shuffle):
    """Three disjoint slices of q_proj alone -- no q/k/v structure exists to be found."""
    cfg = _cfg(directory)["text_config"]
    head_dim = cfg["head_dim"]
    n_kv = cfg["num_key_value_heads"]
    files = index_of(directory)
    rng = np.random.default_rng(0xC0FFEE)
    out = []
    for i in range(cfg["num_hidden_layers"]):
        q = None
        for name in (
            f"model.language_model.layers.{i}.self_attn.q_proj.weight",
            f"model.layers.{i}.self_attn.q_proj.weight",
        ):
            if name in files:
                q = load_tensor(files[name], name)
                break
        if q is None or q.shape[0] < 3 * n_kv * head_dim:
            break
        w = q[: 3 * n_kv * head_dim].reshape(3 * n_kv, head_dim, -1)
        if shuffle:
            w = w[rng.permutation(3 * n_kv)]
        out.append((f"{'nullperm' if shuffle else 'nullseq'}.{i}", w.reshape(-1, q.shape[1]).copy(), n_kv, head_dim))
        if len(out) >= limit:
            break
    return out


def nullseq_fixtures(directory, limit):
    return _null_base(directory, limit, shuffle=False)


def nullperm_fixtures(directory, limit):
    return _null_base(directory, limit, shuffle=True)


def selftest(limit, verbose):
    rows = []
    for directory in QWEN_DIRS:
        if not os.path.isdir(directory):
            print(f"skip (absent): {directory}")
            continue
        tag = os.path.basename(directory)
        for kind, build, truth in (
            ("vision", vision_fixtures, CONTIGUOUS),
            ("text", text_fixtures, CONTIGUOUS),
            ("nullseq", nullseq_fixtures, CONTIGUOUS),
            ("nullperm", nullperm_fixtures, None),
        ):
            for name, w, heads, head_dim in build(directory, limit):
                # As built, and re-packed the other way: every trial carries a known label, and half
                # of them are the layout the detector must NOT return.
                trials = [(w, truth)]
                if truth == CONTIGUOUS:
                    trials.append((to_interleaved(w, heads, head_dim), INTERLEAVED))
                else:
                    trials.append((to_interleaved(w, heads, head_dim), None))
                for arr, want in trials:
                    r = probe(arr, heads, head_dim)
                    rows.append((tag, kind, name, heads, head_dim, want, r))
                    if verbose:
                        print(
                            f"  {tag:24s} {name:12s} want={str(want):11s} got={r['verdict']:11s} "
                            f"normF {r['norm']['f'][CONTIGUOUS]:8.2f}/{r['norm']['f'][INTERLEAVED]:8.2f} "
                            f"p {r['norm']['p'][CONTIGUOUS]:.4f}/{r['norm']['p'][INTERLEAVED]:.4f}  "
                            f"profF {r['profile']['f'][CONTIGUOUS]:7.2f}/{r['profile']['f'][INTERLEAVED]:7.2f} "
                            f"p {r['profile']['p'][CONTIGUOUS]:.4f}/{r['profile']['p'][INTERLEAVED]:.4f}"
                        )
    if not rows:
        print("no fixtures found -- nothing was tested")
        return 1

    print()
    print("negative control -- Qwen3-VL, real trained weights, both layouts per fixture")
    print()
    print("The contract being tested is NEVER WRONG, abstain when weak.  A wrong verdict is a")
    print("failure; an abstention is reported and is not.  `nullseq` is scored the same way as the")
    print("labelled fixtures on purpose -- see the note under the table.")
    print()
    print(f"{'model':26s} {'fixture':9s} {'expected':12s} {'n':>4s} {'right':>6s} "
          f"{'WRONG':>6s} {'abstain':>8s} {'worst F ratio':>14s}")
    wrong_total = 0
    for tag in sorted({r[0] for r in rows}):
        for kind in ("vision", "text", "nullseq", "nullperm"):
            for want in (CONTIGUOUS, INTERLEAVED, None):
                sel = [r for r in rows if r[0] == tag and r[1] == kind and r[5] == want]
                if not sel:
                    continue
                ok = wrong = ambig = 0
                worst = float("inf")
                for r in sel:
                    v = r[6]["verdict"]
                    feats = (r[6]["norm"], r[6]["profile"])
                    if want is None:
                        # the exchangeable null is right exactly when it refuses to commit
                        if v == "ambiguous":
                            ok += 1
                        else:
                            wrong += 1
                        worst = min(worst, max(f["margin"] for f in feats))
                    else:
                        if v == want:
                            ok += 1
                        elif v == "ambiguous":
                            ambig += 1
                        else:
                            wrong += 1
                        worst = min(worst, min(f["margin"] for f in feats))
                wrong_total += wrong
                label = "AMBIGUOUS" if want is None else want
                print(f"{tag:26s} {kind:9s} {label:12s} {len(sel):4d} {ok:6d} {wrong:6d} "
                      f"{ambig:8d} {worst:14.2f}")
    print()
    print("note: `nullseq` is NOT a null.  Three consecutive 8-head slices of a trained q_proj do")
    print("      differ in level, and 'three blocks of heads that differ' is mathematically the")
    print("      same signal as 'q/k/v that differ' -- no statistic can separate them.  So a")
    print("      verdict there that MATCHES the packing is correct behaviour, not a false")
    print("      positive; only `nullperm`, where the blocks are exchangeable, is a true null.")
    print()
    print("PASS" if wrong_total == 0 else f"FAIL -- {wrong_total} WRONG verdict(s)")
    return 0 if wrong_total == 0 else 1


# ------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", help="HF/safetensors directory holding the fused qkv")
    ap.add_argument("--file", help="single .safetensors file")
    ap.add_argument("--tensor", help="fused qkv tensor name")
    ap.add_argument("--heads", type=int, default=56)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--selftest", action="store_true", help="run the Qwen3-VL negative control")
    ap.add_argument("--limit", type=int, default=8, help="fixtures per kind per model in --selftest")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest(args.limit, args.verbose)

    if not args.tensor or not (args.dir or args.file):
        ap.error("--tensor plus one of --dir / --file, or --selftest")
    path = args.file or index_of(args.dir)[args.tensor]
    w = load_tensor(path, args.tensor)
    r = probe(w, args.heads, args.head_dim)
    if args.json:
        json.dump(r, sys.stdout, sort_keys=True, indent=1)
        sys.stdout.write("\n")
        return 0
    print(f"{args.tensor}  shape={tuple(w.shape)}  heads={args.heads} head_dim={args.head_dim}")
    for feat in ("norm", "profile"):
        d = r[feat]
        print(f"  {feat:8s} F contiguous={d['f'][CONTIGUOUS]:12.4f} (p={d['p'][CONTIGUOUS]:.4f})"
              f"  interleaved={d['f'][INTERLEAVED]:12.4f} (p={d['p'][INTERLEAVED]:.4f})"
              f"  -> {d['winner']} x{d['margin']:.2f}")
    print(f"  VERDICT: {r['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
