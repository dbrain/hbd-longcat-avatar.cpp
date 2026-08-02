#!/usr/bin/env python3
"""Numeric proof that the MiniMax-H3 q/k head-channel permutation is an identity.

H3 rotates the leading ``rot_dim = 2 * 3 * rope_freq_dim`` channels of every head with the
split-half convention taken over the ROTARY SUB-BLOCK -- pair ``(c, c + rot_dim/2)`` -- and passes
the remaining ``head_dim - rot_dim`` channels through.  ggml's ``Rope::apply_rope`` /
``ggml_rope_pe_ni`` pair ``(c, c + head_dim/2)`` over the WHOLE head, so the port has to slice, rope
and concat.

Permuting the head-channel axis of the q and k halves of ``qkv_proj.weight`` (and of
``q_norm.weight`` / ``k_norm.weight``) at conversion time makes full-width split-half reproduce
H3's pairing exactly, provided the rotation table is widened from ``rot_dim/2`` pairs to
``head_dim/2`` pairs with an IDENTITY rotation (cos=1, sin=0) in the tail.

    new[c]      = old[c]           c in [0, rot_dim/2)
    new[c + hd] = old[c + rot_dim/2]
    new[rot_dim/2 + j]      = old[rot_dim + j]              j in [0, (hd - rot_dim)/2)
    new[hd + rot_dim/2 + j] = old[rot_dim + (hd-rot_dim)/2 + j]

At the shipping sizes (head_dim 128, rot_dim 96) that is exactly

    new[0..47] = old[0..47], new[64..111] = old[48..95],
    new[48..63] = old[96..111], new[112..127] = old[112..127]

Run with the system python (numpy only), or with a torch-capable interpreter to additionally check
the diffusers reference verbatim:

    python3 tools/ref_qk_permute.py
    /mnt/hdd/3d/avatar-shootout/Pixal3D/.venv/bin/python tools/ref_qk_permute.py
"""

import ast
import os
import sys

import numpy as np

HEAD_DIM = 128
ROPE_FREQ_DIM = 16
ROT_DIM = 2 * 3 * ROPE_FREQ_DIM  # 96
THETA = 10000.0

HIDDEN = 256  # a small stand-in for 5376; the identity does not depend on it
HEADS = 4
SEQ = 11


# ---------------------------------------------------------------------------------------------
# reference (diffusers PR #14355, transformer_minimax_h3.py)
# ---------------------------------------------------------------------------------------------


def ref_rope_tables(position_ids):
    """MiniMaxH3RotaryPosEmbed.forward, numpy transcription. -> cos/sin (S, rot_dim)."""
    inv_freq = 1.0 / (THETA ** (np.arange(0, 2 * ROPE_FREQ_DIM, 2, dtype=np.float64) / (2 * ROPE_FREQ_DIM)))
    freqs = position_ids[:, :, None] * inv_freq[None, None, :]  # (S, 3, freq_dim)
    freqs = freqs.reshape(position_ids.shape[0], 3 * ROPE_FREQ_DIM)
    freqs = np.concatenate((freqs, freqs), axis=-1)  # (S, rot_dim)
    return np.cos(freqs), np.sin(freqs)


def ref_apply_rotary_emb(x, cos, sin):
    """_apply_rotary_emb, numpy transcription. x is (S, heads, head_dim)."""
    rotary_dim = cos.shape[-1]
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    c = cos[:, None, :]
    s = sin[:, None, :]
    x1, x2 = np.split(x_rot, 2, axis=-1)
    x_rotated = np.concatenate((-x2, x1), axis=-1)
    return np.concatenate((x_rot * c + x_rotated * s, x_pass), axis=-1)


def rms_norm(x, weight, eps=1e-5):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * weight


# ---------------------------------------------------------------------------------------------
# the engine side
# ---------------------------------------------------------------------------------------------


def gen_pe(position_ids, inv_freq_len, theta, pairs_out):
    """gen_minimax_h3_pe(), including the identity tail. -> (S, pairs_out, 2, 2)."""
    s = position_ids.shape[0]
    pe = np.zeros((s, pairs_out, 2, 2), dtype=np.float64)
    pe[:, :, 0, 0] = 1.0  # identity everywhere; the real rotations overwrite the first 3*inv_len
    pe[:, :, 1, 1] = 1.0
    inv_freq = 1.0 / (theta ** (np.arange(inv_freq_len, dtype=np.float64) / inv_freq_len))
    for axis in range(3):
        ang = position_ids[:, axis][:, None] * inv_freq[None, :]
        base = axis * inv_freq_len
        pe[:, base : base + inv_freq_len, 0, 0] = np.cos(ang)
        pe[:, base : base + inv_freq_len, 0, 1] = -np.sin(ang)
        pe[:, base : base + inv_freq_len, 1, 0] = np.sin(ang)
        pe[:, base : base + inv_freq_len, 1, 1] = np.cos(ang)
    return pe


def apply_rope_split_half(x, pe):
    """ggml_rope_pe_ni: pair (c, c + d/2) over the full width of x. x is (S, heads, d)."""
    d = x.shape[-1]
    half = d // 2
    assert pe.shape[1] == half, (pe.shape, d)
    x0 = x[..., :half]
    x1 = x[..., half:]
    c00 = pe[:, None, :, 0, 0]
    c01 = pe[:, None, :, 0, 1]
    c10 = pe[:, None, :, 1, 0]
    c11 = pe[:, None, :, 1, 1]
    return np.concatenate((x0 * c00 + x1 * c01, x0 * c10 + x1 * c11), axis=-1)


def port_rope_partial(x, pe_rot):
    """h3_rope_partial(): slice -> full-width split-half on the slice -> concat."""
    rot = pe_rot.shape[1] * 2
    return np.concatenate((apply_rope_split_half(x[..., :rot], pe_rot), x[..., rot:]), axis=-1)


def head_channel_permutation(head_dim, rot_dim):
    """new[i] = old[perm[i]]."""
    rh = rot_dim // 2
    ph = (head_dim - rot_dim) // 2
    perm = np.empty(head_dim, dtype=np.int64)
    perm[0:rh] = np.arange(0, rh)
    perm[head_dim // 2 : head_dim // 2 + rh] = np.arange(rh, rot_dim)
    perm[rh : rh + ph] = np.arange(rot_dim, rot_dim + ph)
    perm[head_dim // 2 + rh : head_dim // 2 + rh + ph] = np.arange(rot_dim + ph, head_dim)
    return perm


def permute_qkv_weight(w, heads, head_dim, perm):
    """w is ggml [in, 3*inner] i.e. numpy (3*inner, in). Permute the q and k thirds only."""
    inner = heads * head_dim
    out = w.copy()
    for third in (0, 1):  # q, k -- v is left alone
        block = w[third * inner : (third + 1) * inner].reshape(heads, head_dim, -1)
        out[third * inner : (third + 1) * inner] = block[:, perm, :].reshape(inner, -1)
    return out


# ---------------------------------------------------------------------------------------------


def report(label, a, b):
    d = float(np.max(np.abs(a - b)))
    scale = float(np.max(np.abs(b))) or 1.0
    print(f"  {label:<58} max|diff| = {d:.3e}   rel = {d / scale:.3e}")
    return d


def main():
    rng = np.random.default_rng(20260802)
    inner = HEADS * HEAD_DIM

    # a real packed 3-axis position grid, including a repeated (t, h) block
    position_ids = np.stack(
        [
            np.repeat(np.arange(SEQ) // 4, 1).astype(np.float64),
            (np.arange(SEQ) % 3).astype(np.float64),
            (np.arange(SEQ) % 5).astype(np.float64) * 0.5,
        ],
        axis=-1,
    )

    x = rng.standard_normal((SEQ, HIDDEN))
    w_qkv = rng.standard_normal((3 * inner, HIDDEN)) / np.sqrt(HIDDEN)
    w_qn = rng.standard_normal(HEAD_DIM) * 0.1 + 1.0
    w_kn = rng.standard_normal(HEAD_DIM) * 0.1 + 1.0

    perm = head_channel_permutation(HEAD_DIM, ROT_DIM)
    print(f"permutation (head_dim={HEAD_DIM}, rot_dim={ROT_DIM}), new[i] = old[perm[i]]:")
    for lo, hi in ((0, 48), (48, 64), (64, 112), (112, 128)):
        print(f"  new[{lo}..{hi - 1}] = old[{perm[lo]}..{perm[hi - 1]}]")
    assert sorted(perm.tolist()) == list(range(HEAD_DIM)), "permutation is not a bijection"

    def project(weight):
        qkv = x @ weight.T
        q, k, v = np.split(qkv, 3, axis=-1)
        return (
            q.reshape(SEQ, HEADS, HEAD_DIM),
            k.reshape(SEQ, HEADS, HEAD_DIM),
            v.reshape(SEQ, HEADS, HEAD_DIM),
        )

    # --- A: the reference ---------------------------------------------------------------------
    cos, sin = ref_rope_tables(position_ids)
    q, k, v = project(w_qkv)
    q_ref = ref_apply_rotary_emb(rms_norm(q, w_qn), cos, sin)
    k_ref = ref_apply_rotary_emb(rms_norm(k, w_kn), cos, sin)
    scores_ref = np.einsum("shd,thd->hst", q_ref, k_ref)

    # --- B: the port as it stands (slice / rope / concat, 48-pair pe) --------------------------
    pe48 = gen_pe(position_ids, ROPE_FREQ_DIM, THETA, 3 * ROPE_FREQ_DIM)
    q_port = port_rope_partial(rms_norm(q, w_qn), pe48)
    k_port = port_rope_partial(rms_norm(k, w_kn), pe48)
    scores_port = np.einsum("shd,thd->hst", q_port, k_port)

    # --- C: permuted weights + ONE full-width split-half rope, 64-pair pe ----------------------
    pe64 = gen_pe(position_ids, ROPE_FREQ_DIM, THETA, HEAD_DIM // 2)
    qp, kp, vp = project(permute_qkv_weight(w_qkv, HEADS, HEAD_DIM, perm))
    q_new = apply_rope_split_half(rms_norm(qp, w_qn[perm]), pe64)
    k_new = apply_rope_split_half(rms_norm(kp, w_kn[perm]), pe64)
    scores_new = np.einsum("shd,thd->hst", q_new, k_new)

    print("\nA. reference vs the port as it stands")
    fail = report("q after rope", q_port, q_ref) > 1e-12
    fail |= report("k after rope", k_port, k_ref) > 1e-12
    fail |= report("attention scores q.k", scores_port, scores_ref) > 1e-10

    print("\nB. reference vs permuted weights + full-width split-half")
    fail |= report("attention scores q.k", scores_new, scores_ref) > 1e-10
    fail |= report("v (must be untouched by the permutation)", vp, v) > 0.0
    # the rotated q/k are the SAME VECTORS, permuted -- so out_proj sees identical input only
    # because it consumes `v`, never q or k.
    fail |= report("q after rope, un-permuted", q_new, q_ref[..., perm]) > 1e-12
    fail |= report("k after rope, un-permuted", k_new, k_ref[..., perm]) > 1e-12

    # --- D: the pe tail really is the identity ------------------------------------------------
    tail = pe64[:, 3 * ROPE_FREQ_DIM :]
    eye = np.broadcast_to(np.eye(2), tail.shape)
    print("\nC. widened rotation table")
    fail |= report("pe[0:48] identical to the 48-pair table", pe64[:, : 3 * ROPE_FREQ_DIM], pe48) > 0.0
    fail |= report("pe[48:64] == identity rotation", tail, eye) > 0.0

    torch_ok = check_torch(position_ids, x, w_qkv, w_qn, w_kn, perm)

    print("\n" + ("FAIL" if fail else "PASS") + f"  (heads={HEADS}, seq={SEQ}, hidden={HIDDEN})")
    return 1 if fail else (0 if torch_ok is not False else 1)


# ---------------------------------------------------------------------------------------------
# optional: run the diffusers reference VERBATIM under real torch
# ---------------------------------------------------------------------------------------------

REF_PATH = os.path.expanduser(
    "~/handoffs/longcat-avatar.cpp/minimax-h3/ref/diffusers/"
    "src__diffusers__models__transformers__transformer_minimax_h3.py"
)


def check_torch(position_ids, x, w_qkv, w_qn, w_kn, perm):
    try:
        import torch
    except ImportError:
        print("\nD. torch not available in this interpreter -- numpy transcription only")
        return None
    if not os.path.exists(REF_PATH):
        print(f"\nD. reference source not found at {REF_PATH}")
        return None

    # Pull `_apply_rotary_emb` and `MiniMaxH3RotaryPosEmbed` out of the PR source by AST name so
    # nothing is retyped (same technique as tools/ref_layout_torch.py).
    tree = ast.parse(open(REF_PATH).read())
    wanted = {"_apply_rotary_emb", "MiniMaxH3RotaryPosEmbed"}
    picked = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in wanted]
    assert len(picked) == len(wanted), [n.name for n in picked]
    ns = {"torch": torch, "nn": torch.nn}
    exec(compile(ast.Module(body=picked, type_ignores=[]), REF_PATH, "exec"), ns)

    inner = w_qkv.shape[0] // 3
    heads = inner // HEAD_DIM
    seq = x.shape[0]
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(torch.float64)

    rope = ns["MiniMaxH3RotaryPosEmbed"](rope_freq_dim=ROPE_FREQ_DIM, rope_theta=THETA).to(torch.float64)
    rope.inv_freq = rope.inv_freq.to(torch.float64)
    cos, sin = rope(t(position_ids))

    def run(weight, qn, kn):
        qkv = t(x) @ t(weight).T
        q, k, _ = qkv.split(inner, dim=-1)
        q = q.view(1, seq, heads, HEAD_DIM)
        k = k.view(1, seq, heads, HEAD_DIM)
        rms = lambda a, w: a / torch.sqrt(a.pow(2).mean(-1, keepdim=True) + 1e-5) * t(w)
        q = ns["_apply_rotary_emb"](rms(q, qn), cos, sin)
        k = ns["_apply_rotary_emb"](rms(k, kn), cos, sin)
        return torch.einsum("bshd,bthd->bhst", q, k)

    ref_scores = run(w_qkv, w_qn, w_kn)[0].numpy()

    def engine_scores(weight, qn, kn, pe, rope):
        qkv = (x @ weight.T).reshape(seq, 3, inner)
        q = rms_norm(qkv[:, 0].reshape(seq, heads, HEAD_DIM), qn)
        k = rms_norm(qkv[:, 1].reshape(seq, heads, HEAD_DIM), kn)
        return np.einsum("shd,thd->hst", rope(q, pe), rope(k, pe))

    pe48 = gen_pe(position_ids, ROPE_FREQ_DIM, THETA, 3 * ROPE_FREQ_DIM)
    pe64 = gen_pe(position_ids, ROPE_FREQ_DIM, THETA, HEAD_DIM // 2)
    old = engine_scores(w_qkv, w_qn, w_kn, pe48, port_rope_partial)
    new = engine_scores(
        permute_qkv_weight(w_qkv, heads, HEAD_DIM, perm), w_qn[perm], w_kn[perm], pe64, apply_rope_split_half
    )

    # MiniMaxH3RotaryPosEmbed.forward casts position_ids to float32 before building the angles, so
    # BOTH engine paths sit ~1e-8 from the reference no matter what.  The claim under test is that
    # the permutation adds nothing to that floor -- hence the control line.
    print("\nD. VERBATIM diffusers reference under torch %s" % torch.__version__)
    d_old = report("control: TODAY's slice/rope/concat path vs reference", old, ref_scores)
    d_new = report("attention scores: permuted engine path vs reference", new, ref_scores)
    d_pair = report("permuted path vs today's path", new, old)
    return d_pair <= 1e-10 and d_new <= max(1e-10, 4.0 * d_old)


if __name__ == "__main__":
    sys.exit(main())
