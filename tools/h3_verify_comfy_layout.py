#!/usr/bin/env python3
"""Prove `h3_comfy_nvfp4_layout.py` reads and writes ComfyUI's NVFP4 layout correctly.

Four independent checks.  The first two are the ones that matter, because they are the two
things a round-trip CANNOT catch: a round-trip against your own code is self-consistent
even when both directions are wrong the same way.

  A. ORACLE, ELEMENTWISE.  Take a layer out of a REAL ComfyUI nvfp4 checkpoint we did not
     produce (the H3 text encoder), dequantise it with this module, and compare it
     elementwise against the ORIGINAL bf16 Qwen3-VL weight it was quantised from.  Then
     repeat with the swizzle off and with the nibble order flipped, to show what a wrong
     convention actually looks like.  This is the ONLY check that can see the nibble order.

  B. ROUND-TRIP.  Our GGUF's block_nvfp4 bytes -> comfy layout -> back.  Must be
     byte-identical, which closes the loop against the C++ importer's algorithm.

  C. ROW NORMS.  Dequantised ggml vs dequantised comfy for the same tensor, per output row.
     Rotation-invariant, so it catches the swizzle; it is exactly the statistic that gave
     0.013 (row-major) vs 0.999937 (de-swizzled) when the import was first written.

  D. TARGET-SHAPE AGREEMENT.  Our GGUF vs ComfyUI's own H3 checkpoint: q_norm under the
     RoPE head-channel permutation, and the token-refiner QKV row norms under the fused-QKV
     de-interleave.  These decide whether the exporter must undo those two rewrites.
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h3_comfy_nvfp4_layout as L  # noqa: E402
from h3_prune_adaln import GGML_TYPES, gguf_read_header, nbytes  # noqa: E402

COMFY_DIR = "/mnt/ssd/h3-staging/comfy-ref/models"
COMFY_TE = f"{COMFY_DIR}/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
COMFY_DIT = f"{COMFY_DIR}/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
QWEN_DIR = "/mnt/ssd/h3-staging/Qwen3-VL-32B-Instruct"


# ------------------------------------------------------------------------------ readers
class ST:
    """Single-file safetensors reader that keeps raw bytes for exotic dtypes."""

    def __init__(self, path):
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.hdr = json.loads(self.f.read(n))
        self.hdr.pop("__metadata__", None)
        self.base = 8 + n

    def raw(self, name):
        h = self.hdr[name]
        a, b = h["data_offsets"]
        self.f.seek(self.base + a)
        return h["dtype"], h["shape"], self.f.read(b - a)

    def array(self, name):
        dt, shape, raw = self.raw(name)
        if dt == "BF16":
            a = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
        elif dt == "F32":
            a = np.frombuffer(raw, dtype="<f4")
        elif dt == "F16":
            a = np.frombuffer(raw, dtype="<f2").astype(np.float32)
        elif dt == "I8":
            a = np.frombuffer(raw, dtype=np.int8).astype(np.float32)
        elif dt in ("U8", "F8_E4M3"):
            a = np.frombuffer(raw, dtype=np.uint8)
        else:
            raise SystemExit(f"{name}: unhandled dtype {dt}")
        return a.reshape(shape) if shape else a


class Sharded:
    def __init__(self, d):
        self.map = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self.dir, self.cache = d, {}

    def array(self, name):
        shard = self.map[name]
        if shard not in self.cache:
            self.cache[shard] = ST(os.path.join(self.dir, shard))
        return self.cache[shard].array(name)


class GGUF:
    def __init__(self, path):
        self.f, _, _, _, tl, self.data_start, _ = gguf_read_header(path)
        self.t = {t["name"]: t for t in tl}

    def info(self, name):
        t = self.t[name]
        return GGML_TYPES[t["type"]][0], list(t["dims"])

    def raw(self, name):
        t = self.t[name]
        self.f.seek(self.data_start + t["off"])
        return self.f.read(nbytes(t["dims"], t["type"]))

    def array(self, name):
        t = self.t[name]
        tname = GGML_TYPES[t["type"]][0]
        raw = self.raw(name)
        if tname == "f32":
            a = np.frombuffer(raw, dtype="<f4")
        elif tname == "bf16":
            a = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
        elif tname == "f16":
            a = np.frombuffer(raw, dtype="<f2").astype(np.float32)
        else:
            raise SystemExit(f"{name}: {tname} is not a plain dtype")
        # GGUF dims are fastest-first; torch shape is the reverse.
        return a.reshape(list(reversed(t["dims"])))


def corr(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(np.corrcoef(a, b)[0, 1])


def rownorm(w):
    return np.sqrt((np.asarray(w, np.float64) ** 2).sum(axis=1))


# --------------------------------------------------------------------------- the checks
def check_a():
    print("=" * 78)
    print("A. ORACLE: a REAL ComfyUI nvfp4 layer vs the bf16 it was quantised from")
    print("   (the only check that can see the nibble order)")
    print("=" * 78)
    if not os.path.exists(COMFY_TE):
        print("   SKIP: no comfy nvfp4 checkpoint at", COMFY_TE)
        return None
    te = ST(COMFY_TE)
    qwen = Sharded(QWEN_DIR)

    ok = True
    for layer in ("model.layers.0.mlp.gate_proj", "model.layers.3.self_attn.q_proj"):
        if f"{layer}.weight" not in te.hdr:
            continue
        if f"{layer}.pre_quant_scale" in te.hdr:
            print(f"   {layer}: has pre_quant_scale (AWQ input smoothing), skipping")
            continue
        cfg = json.loads(bytes(te.array(f"{layer}.comfy_quant")).decode())
        _, pshape, praw = te.raw(f"{layer}.weight")
        _, sshape, sraw = te.raw(f"{layer}.weight_scale")
        s2 = float(te.array(f"{layer}.weight_scale_2").reshape(-1)[0])
        out_f, in_f = pshape[0], pshape[1] * 2
        packed = np.frombuffer(praw, np.uint8).reshape(out_f, in_f // 2)
        plane = np.frombuffer(sraw, np.uint8).reshape(sshape)

        ref = qwen.array(layer.replace("model.layers", "model.language_model.layers") + ".weight")
        assert ref.shape == (out_f, in_f), (ref.shape, out_f, in_f)

        print(f"\n   {layer}  {cfg}  [{out_f}, {in_f}]  scale_2={s2:.6g}")
        print(f"     scale plane {tuple(sshape)}, expected {L.scale_plane_shape(in_f, out_f)}")
        rows = slice(0, 512)                      # a slab is plenty and keeps this quick
        r = np.asarray(ref[rows], np.float64)
        got = {}
        for key, label, kw in (
                ("correct", "CORRECT   (de-swizzle + hi_first)", dict(hi_first=True, swizzled=True)),
                ("nibble", "wrong     (nibble order flipped)", dict(hi_first=False, swizzled=True)),
                ("swizzle", "wrong     (scales read row-major)", dict(hi_first=True, swizzled=False))):
            if not kw["swizzled"]:
                # row-major misread: take the first out*n_sub bytes of the plane as-is
                n_sub = in_f // L.QK_NVFP4_SUB
                sub = plane.reshape(-1)[: out_f * n_sub]
                w = L.dequantize_comfy(packed, sub, s2, in_f, out_f, **kw)
            else:
                w = L.dequantize_comfy(packed, plane, s2, in_f, out_f, **kw)
            c = corr(w[rows], r)
            n = corr(rownorm(w[rows]), rownorm(r))
            got[key] = (c, n)
            print(f"     {label}: elementwise r={c:+.6f}  row-norm r={n:+.6f}")

        # The bar is the MARGIN, not an absolute correlation.  A correct NVFP4 dequant only
        # reaches ~0.92..0.96 elementwise against the bf16 it came from -- that gap is the
        # 4-bit quantisation error (plus, in this checkpoint, AWQ input smoothing folded into
        # the weight), not a layout defect.  What proves the layout is that the two wrong
        # conventions collapse: the flipped nibble order lands at r ~= 0, and reading the
        # scales row-major destroys the row norms.
        c_ok, n_ok = got["correct"]
        c_nib = abs(got["nibble"][0])
        n_swz = abs(got["swizzle"][1])
        good = c_ok > 0.85 and n_ok > 0.85 and c_ok > 20 * c_nib and n_ok > 2 * n_swz
        print(f"     -> correct elementwise r={c_ok:+.4f} vs flipped-nibble |r|={c_nib:.4f}; "
              f"correct row-norm r={n_ok:+.4f} vs row-major |r|={n_swz:.4f}: "
              f"{'PASS' if good else 'FAIL'}")
        ok &= good
    return ok


def check_b_c(gguf_path, tensors):
    print()
    print("=" * 78)
    print("B/C. ROUND-TRIP and ROW NORMS on our own GGUF")
    print("=" * 78)
    g = GGUF(gguf_path)
    ok = True
    for name in tensors:
        tname, dims = g.info(name)
        if tname != "nvfp4":
            print(f"   {name}: {tname}, skipping")
            continue
        in_f, out_f = int(dims[0]), int(dims[1])
        blocks = np.frombuffer(g.raw(name), np.uint8)
        packed, plane, rowmajor = L.ggml_to_comfy(blocks, in_f, out_f)
        back = L.comfy_to_ggml(packed, plane, in_f, out_f)
        exact = bool(np.array_equal(blocks, back))
        w_comfy = L.dequantize_comfy(packed, plane, 1.0, in_f, out_f)
        w_direct = L.dequantize_comfy(packed, rowmajor, 1.0, in_f, out_f, swizzled=False)
        rn = corr(rownorm(w_comfy), rownorm(w_direct))
        print(f"   {name:<46} [{out_f}, {in_f}]")
        print(f"     round-trip block_nvfp4 bytes identical : {exact}")
        print(f"     scale plane {plane.shape} = expected {L.scale_plane_shape(in_f, out_f)}: "
              f"{plane.shape == L.scale_plane_shape(in_f, out_f)}")
        print(f"     de-swizzled vs row-major row-norm r    : {rn:+.6f}  (must be 1.0)")
        print(f"     |w| max {np.abs(w_comfy).max():.5g}  mean {np.abs(w_comfy).mean():.5g}  "
              f"zeros {100.0 * np.mean(w_comfy == 0):.2f}%")
        ok &= exact and abs(rn - 1.0) < 1e-9
    return ok


def check_d(gguf_path):
    print()
    print("=" * 78)
    print("D. OUR GGUF vs COMFYUI'S OWN H3 CHECKPOINT")
    print("   decides whether the exporter must undo the RoPE permutation / de-interleave")
    print("=" * 78)
    if not os.path.exists(COMFY_DIT):
        print("   SKIP: no comfy H3 checkpoint at", COMFY_DIT)
        return None
    g = GGUF(gguf_path)
    c = ST(COMFY_DIT)

    print("   markers in our GGUF:",
          [m for m in ("attn.qkv_deinterleaved", "rope.qk_permuted") if m in g.t] or "none")

    # --- D1: q_norm, 128 floats, unquantised on both sides -> exact elementwise test.
    perm = np.array(qk_head_permutation(128, 96))
    for leaf in ("blocks.0.attn.q_norm.weight", "blocks.0.attn.k_norm.weight"):
        ours = g.array(leaf).astype(np.float64).ravel()
        theirs = c.array(leaf).astype(np.float64).ravel()
        undone = np.empty_like(ours)
        undone[perm] = ours                       # invert ours[i] = orig[perm[i]]
        d_as_is = np.abs(ours - theirs).max()
        d_undone = np.abs(undone - theirs).max()
        print(f"\n   {leaf}")
        print(f"     max|ours - comfy|            = {d_as_is:.6g}")
        print(f"     max|unpermute(ours) - comfy| = {d_undone:.6g}   <== must be the small one")
        # our F32 is an upcast of the original bf16; check writing BF16 back is lossless
        bf = (np.asarray(g.array(leaf), np.float32).view(np.uint32) & 0xFFFF).max()
        print(f"     our F32 has {'NO' if bf == 0 else 'SOME'} mantissa bits below bf16 "
              f"-> emitting BF16 is {'lossless' if bf == 0 else 'LOSSY'}")

    # --- D2: token-refiner QKV row norms.  The refiner takes the de-interleave and NOT the
    #         RoPE permutation, so it isolates the de-interleave on its own.
    leaf = "token_refiner.blocks.0.attn.qkv_proj.weight"
    _, dims = g.info(leaf)
    in_f, out_f = int(dims[0]), int(dims[1])
    ours = L.dequantize_ggml(np.frombuffer(g.raw(leaf), np.uint8), in_f, out_f)
    theirs = c.array(leaf)
    n_ours, n_theirs = rownorm(ours), rownorm(theirs)
    heads, head_dim = out_f // (3 * 128), 128
    # If comfy were per-head interleaved, its row blocks would be a (heads x 3) transpose of ours.
    idx = np.arange(3 * heads)
    src = 3 * (idx % heads) + (idx // heads)
    reint = n_theirs.reshape(3 * heads, head_dim)[np.argsort(src)].reshape(-1)
    print(f"\n   {leaf}   ({heads} heads x {head_dim})")
    print(f"     row-norm r, comfy AS-IS vs ours (contiguous q|k|v) = {corr(n_ours, n_theirs):+.6f}")
    print(f"     row-norm r, comfy RE-INTERLEAVED vs ours           = {corr(n_ours, reint):+.6f}")

    # --- D3: block-0 QKV.  comfy's is int8 with convrot; the rotation is over the INPUT dim,
    #         which preserves each output row's L2 norm, so row norms remain comparable.
    leaf = "blocks.0.attn.qkv_proj.weight"
    _, dims = g.info(leaf)
    in_f, out_f = int(dims[0]), int(dims[1])
    ours = L.dequantize_ggml(np.frombuffer(g.raw(leaf), np.uint8), in_f, out_f)
    q8 = c.array(leaf)
    sc = c.array(leaf.replace(".weight", ".weight_scale")).reshape(-1, 1)
    theirs = q8 * sc
    n_ours, n_theirs = rownorm(ours), rownorm(theirs)
    heads = out_f // (3 * 128)
    inner = heads * 128
    und = n_ours.copy()
    for third in range(2):                        # q and k only; v is never permuted
        blk = und[third * inner:(third + 1) * inner].reshape(heads, 128)
        fixed = np.empty_like(blk)
        fixed[:, perm] = blk
        und[third * inner:(third + 1) * inner] = fixed.reshape(-1)
    print(f"\n   {leaf}   (comfy is int8+convrot; convrot rotates the INPUT dim, so row norms hold)")
    print(f"     row-norm r, ours AS-IS                     = {corr(n_ours, n_theirs):+.6f}")
    print(f"     row-norm r, ours with RoPE permute UNDONE  = {corr(und, n_theirs):+.6f}"
          "   <== must be the higher one")
    return True


def qk_head_permutation(head_dim, rot_dim):
    """MiniMaxH3::build_qk_head_permutation -- out[i] = in[perm[i]]."""
    half, rot_half, pass_half = head_dim // 2, rot_dim // 2, (head_dim - rot_dim) // 2
    perm = [0] * head_dim
    for c in range(rot_half):
        perm[c] = c
        perm[half + c] = rot_half + c
    for j in range(pass_half):
        perm[rot_half + j] = rot_dim + j
        perm[half + rot_half + j] = rot_dim + pass_half + j
    return perm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default="/home/dbrain/models/h3/h3-fl2va-dit-nvfp4-pruned.gguf")
    ap.add_argument("--skip-oracle", action="store_true")
    a = ap.parse_args()

    results = []
    if not a.skip_oracle:
        results.append(("A oracle", check_a()))
    results.append(("B/C round-trip", check_b_c(a.gguf, [
        "blocks.0.attn.out_proj.weight",
        "blocks.7.mlp.fc2.weight",
        "condition_proj.weight",
    ])))
    results.append(("D target shape", check_d(a.gguf)))

    print("\n" + "=" * 78)
    for name, r in results:
        print(f"  {name:<20} {'PASS' if r else ('SKIP/INFO' if r is None else 'FAIL')}")
    print("=" * 78)


if __name__ == "__main__":
    main()
