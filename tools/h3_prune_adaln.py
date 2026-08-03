#!/usr/bin/env python3
"""Rewrite a MiniMax-H3 DiT GGUF into the PRUNED (rank-8 curve-table) AdaLN form.

WHAT THIS IS
------------
H3's AdaLN modulation is a function of the TIMESTEP ALONE:

    t_emb(t) = time_embedder(sinusoid(t))            # no text, no image, no guidance
    mod_b(t) = W_b . silu(t_emb(t)) + b_b            # per block, [96768] or [10752]

`silu(t_emb(t))` is a ONE-DIMENSIONAL CURVE in R^2688 -- it has 1025 useful samples and
2688 coordinates, but it does not fill them.  Measured on our own bf16 FL2VA weights, the
mean-centred curve matrix [1025 x 2688] has singular values

    s1 7.08e+00   s4 4.70e-01   s8 2.07e-03   s12 9.52e-06

i.e. 99.9999998% of its energy lives in 8 dimensions.  So write the curve as

    silu(t_emb(t)) ~= mu + c(t) U        U: [8, 2688] orthonormal,  c(t): [1025, 8]

and every block's modulation factorises EXACTLY through that 8-dim bottleneck:

    mod_b(t) = (W_b U^T) c(t) + (W_b mu + b_b)
             =    W'_b    c(t) +      b'_b        W'_b: [out, 8]

The AdaLN projection weights collapse from 13,034,520,576 parameters (50 blocks x
96768x2688, plus final_layer 10752x2688 -- 39% of the whole DiT) to 38,793,216: a factor of
336.  Measured modulation-space error is 1.1e-5 .. 1.5e-5 relative, against the ~4.5-bit
NVFP4 those 13.0B parameters are otherwise quantised to.  This is not a compression
tradeoff -- the pruned form is MORE accurate AND 336x smaller, because the full form was
storing a rank-8 map in a rank-2688 container.

On the real FL2VA nvfp4 DiT that is 18.74 GB -> 11.47 GB of tensor payload.

This is the same factorisation Comfy-Org's `..._pruned_...` checkpoints ship.  Verified:
the 8-dim span derived here from our weights and the span of their `adaln_t_table` agree to
principal cosines of 1.000000 (all eight).  We derive our own rather than consume theirs
because (a) it works for ref2va, which they do not publish, (b) we can keep the factors at
F32 where they store F16 -- their checkpoint reproduces the exact modulation to only
1.9e-4, which is F16 rounding noise, where ours reaches 1.3e-5.

The ENGINE SIDE ALREADY EXISTS and is selected by TENSOR PRESENCE, not by a flag: see
MiniMaxH3Config::use_adaln_curves() / detect_from_weights() in
src/model/diffusion/minimax_h3.hpp.  A file carrying `adaln_t_table` takes the curve path
(Linear(8, N), no SiLU, table lerp by timestep); a file without it is unchanged.

WHAT IT DOES
------------
Copies the input GGUF tensor-for-tensor and byte-for-byte EXCEPT:
  * drops    time_embedder.*                    (the curve replaces it)
  * replaces *.adaln_proj.linear.weight         [2688, out] -> [8, out]   F32
  * replaces *.adaln_proj.linear.bias           [out]       -> [out]      F32
  * adds     adaln_t_table                      [8, 1025]                 F32
Everything else -- attention, MLP, the fp32 islands, the de-interleave/RoPE markers -- is
copied verbatim, so an A/B against the input file isolates the AdaLN change exactly.

The factors are derived from the RAW bf16 safetensors, never from the quantised GGUF: the
whole point is to not inherit NVFP4 error into the factorisation.

USAGE
-----
    h3_prune_adaln.py --src   /mnt/ssd/h3-staging/weights/MiniMax-H3/FL2VA/transformer \
                      --in    /home/dbrain/models/h3/h3-fl2va-dit-nvfp4.gguf \
                      --out   /home/dbrain/models/h3/h3-fl2va-dit-nvfp4-pruned.gguf

    h3_prune_adaln.py ... --dry-run       # report the factorisation + sizes, write nothing
    h3_prune_adaln.py ... --verify        # after writing, read the file back and compare
                                          # its reconstructed modulation to the exact one
    h3_prune_adaln.py ... --verify-only   # just check an --out that already exists
    h3_prune_adaln.py ... --verify-rows 0 # exhaustive rather than sampled (slow: hours)

Ref2VA works the same way -- point --src at its own transformer/ dir.  The basis is
per-checkpoint (it is derived from THAT file's time embedder) and must never be shared
between variants.

numpy only -- no torch, no `gguf`, no `safetensors` package.
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

GRID = 1025          # rows of adaln_t_table; matches Comfy's, and lerp error is 2.5e-6
RANK = 8             # curve dimension

# ggml_type -> (name, block elems, bytes per block).  From ggml/include/ggml.h in THIS tree.
# NOTE: 30 is BF16 and 39 is MXFP4.  tools/h3_gguf_probe.py has these two wrong (it calls 30
# "i32" and 39 "bf16") and therefore computes wrong byte sizes for every bf16 tensor -- do not
# copy its table.
GGML_TYPES = {
    0: ("f32", 1, 4), 1: ("f16", 1, 2), 2: ("q4_0", 32, 18), 3: ("q4_1", 32, 20),
    6: ("q5_0", 32, 22), 7: ("q5_1", 32, 24), 8: ("q8_0", 32, 34), 9: ("q8_1", 32, 40),
    10: ("q2_K", 256, 84), 11: ("q3_K", 256, 110), 12: ("q4_K", 256, 144),
    13: ("q5_K", 256, 176), 14: ("q6_K", 256, 210), 15: ("q8_K", 256, 292),
    16: ("iq2_xxs", 256, 66), 17: ("iq2_xs", 256, 74), 18: ("iq3_xxs", 256, 98),
    19: ("iq1_s", 256, 50), 20: ("iq4_nl", 32, 18), 21: ("iq3_s", 256, 110),
    22: ("iq2_s", 256, 82), 23: ("iq4_xs", 256, 136),
    24: ("i8", 1, 1), 25: ("i16", 1, 2), 26: ("i32", 1, 4), 27: ("i64", 1, 8),
    28: ("f64", 1, 8), 29: ("iq1_m", 256, 56), 30: ("bf16", 1, 2),
    34: ("tq1_0", 256, 54), 35: ("tq2_0", 256, 66),
    39: ("mxfp4", 32, 17), 40: ("nvfp4", 64, 36), 41: ("q1_0", 32, 6), 42: ("q2_0", 32, 10),
}
GGML_TYPE_F32 = 0

ADALN_W = ".adaln_proj.linear.weight"
ADALN_B = ".adaln_proj.linear.bias"


# ---------------------------------------------------------------- safetensors (read-only)
class Safetensors:
    """Minimal sharded-safetensors reader.  bf16 is widened to f32 by bit-shift."""

    def __init__(self, path):
        self.files = {}
        if os.path.isdir(path):
            idx = os.path.join(path, "model.safetensors.index.json")
            if not os.path.exists(idx):
                raise SystemExit(f"{path}: no model.safetensors.index.json")
            self.map = json.load(open(idx))["weight_map"]
            self.dir = path
        else:
            raise SystemExit(f"{path}: expected the transformer DIRECTORY")

    def _hdr(self, shard):
        if shard not in self.files:
            f = open(os.path.join(self.dir, shard), "rb")
            n = int.from_bytes(f.read(8), "little")
            self.files[shard] = (f, json.loads(f.read(n)), 8 + n)
        return self.files[shard]

    def get(self, name):
        f, hdr, base = self._hdr(self.map[name])
        h = hdr[name]
        beg, end = h["data_offsets"]
        f.seek(base + beg)
        raw = f.read(end - beg)
        if h["dtype"] == "BF16":
            a = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
        elif h["dtype"] == "F32":
            a = np.frombuffer(raw, dtype="<f4")
        elif h["dtype"] == "F16":
            a = np.frombuffer(raw, dtype="<f2").astype(np.float32)
        else:
            raise SystemExit(f"{name}: unhandled safetensors dtype {h['dtype']}")
        return a.reshape(h["shape"])

    def has(self, name):
        return name in self.map


# ------------------------------------------------------------------------ GGUF read/write
def gguf_read_header(path):
    f = open(path, "rb")

    def u32():
        return struct.unpack("<I", f.read(4))[0]

    def u64():
        return struct.unpack("<Q", f.read(8))[0]

    def string():
        return f.read(u64())

    if f.read(4) != b"GGUF":
        raise SystemExit(f"{path}: not a GGUF file")
    version = u32()
    n_tensors = u64()
    n_kv = u64()
    kv_start = f.tell()

    SIMPLE = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
              7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

    def read_value(t):
        if t in SIMPLE:
            fmt = SIMPLE[t]
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
        if t == 8:
            return string()
        if t == 9:
            et = u32()
            n = u64()
            for _ in range(n):
                read_value(et)
            return None
        raise SystemExit(f"unknown GGUF value type {t}")

    align = 32
    for _ in range(n_kv):
        key = string()
        val = read_value(u32())
        # Honour a declared alignment: every tensor offset is a multiple of it, and getting
        # this wrong silently shifts every tensor's data.
        if key == b"general.alignment" and isinstance(val, int) and val > 0:
            align = val
    kv_end = f.tell()
    f.seek(kv_start)
    kv_blob = f.read(kv_end - kv_start)

    tensors = []
    for _ in range(n_tensors):
        name = string().decode("utf-8")
        nd = u32()
        dims = [u64() for _ in range(nd)]
        ttype = u32()
        off = u64()
        tensors.append({"name": name, "dims": dims, "type": ttype, "off": off})
    pos = f.tell()
    data_start = pos + ((-pos) % align)
    return f, version, kv_blob, n_kv, tensors, data_start, align


def nbytes(dims, ttype):
    tname, blk, tsz = GGML_TYPES.get(ttype, (None, None, None))
    if tname is None:
        raise SystemExit(f"unknown ggml type {ttype} -- refusing to guess its size")
    n = 1
    for d in dims:
        n *= d
    if n % blk:
        raise SystemExit(f"type {tname} needs a multiple of {blk} elements, got {n}")
    return (n // blk) * tsz


def pack_str(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


# ------------------------------------------------------------------------ the maths
def silu(x):
    return x / (1.0 + np.exp(-x))


def sinusoid(t, dim):
    """comfy/ldm/minimax/model.py:127-133 -- COS BEFORE SIN, t consumed UNSCALED in [0,1].

    Must match MiniMaxH3::timestep_sinusoid() in minimax_h3_sched.hpp exactly."""
    half = dim // 2
    freq = np.exp(-np.log(10000.0) * np.arange(half, dtype=np.float64) / half)
    arg = t[:, None] * freq[None, :]
    return np.concatenate([np.cos(arg), np.sin(arg)], axis=1)


def build_curve_basis(st, rank, grid, verbose=True):
    """Return (table [grid, rank] f64, U [rank, 2688] f64, mu [2688] f64, report str)."""
    pi_w = st.get("time_embedder.proj_in.weight").astype(np.float64)
    pi_b = st.get("time_embedder.proj_in.bias").astype(np.float64)
    po_w = st.get("time_embedder.proj_out.weight").astype(np.float64)
    po_b = st.get("time_embedder.proj_out.bias").astype(np.float64)

    t = np.arange(grid, dtype=np.float64) / (grid - 1)
    temb = silu(sinusoid(t, pi_w.shape[1]) @ pi_w.T + pi_b) @ po_w.T + po_b
    F = silu(temb)                                     # [grid, t_dim] -- adaln_proj input

    mu = F.mean(0)
    U_, S, Vt = np.linalg.svd(F - mu, full_matrices=False)
    U = Vt[:rank]                                      # [rank, t_dim], orthonormal rows
    c = (F - mu) @ U.T                                 # [grid, rank]

    # Balance the coordinates to unit std and fold the INVERSE into the basis, so that the
    # product table @ U_scaled is unchanged.  The two scalings pull in OPPOSITE directions --
    # divide the table by d, MULTIPLY the basis by d -- and getting that backwards is silent:
    # it produces a well-formed file whose modulation is wrong by up to 1/d^2 (~2e8 here).
    # Hence the reconstruction assert below; do not remove it.
    d = c.std(0)
    d[d == 0] = 1.0
    table = c / d                                      # [grid, rank], unit std per column
    U_scaled = U * d[:, None]                          # so that  f ~= mu + table @ U_scaled

    recon = np.linalg.norm((mu + table @ U_scaled) - F) / np.linalg.norm(F)
    resid = np.sqrt((S[rank:] ** 2).sum() / (S ** 2).sum())
    # The reconstruction can only be as good as the rank-`rank` truncation; anything much
    # worse means the table/basis scaling is inconsistent, not that the curve is high-rank.
    if not np.isfinite(recon) or recon > max(10.0 * resid, 1e-6):
        raise SystemExit(f"INTERNAL: curve reconstruction is {recon:.3e} but the rank-{rank} "
                         f"truncation alone is {resid:.3e} -- the table/basis scaling is wrong")

    rep = [f"  curve   f(t) = silu(time_embedder(sinusoid(t)))  [{grid} x {F.shape[1]}]",
           f"  sigma   " + "  ".join(f"s{i+1}={S[i]:.3e}" for i in range(min(4, len(S)))),
           f"          " + "  ".join(f"s{i+1}={S[i]:.3e}" for i in range(rank - 1, min(rank + 3, len(S)))),
           f"  rank {rank} captures 1 - {resid:.3e} of the centred curve energy",
           f"  reconstruction  ||mu + table @ U - f|| / ||f||  =  {recon:.3e}"]
    if verbose:
        print("\n".join(rep))
    if resid > 1e-3:
        raise SystemExit(f"REFUSING: rank {rank} leaves {resid:.3e} residual -- this checkpoint's "
                         f"AdaLN curve is NOT rank {rank}.  Do not prune it.")
    return table, U_scaled, mu, S


def factorise(W, b, U_scaled, mu):
    """[out, t_dim] bf16-widened -> (W' [out, rank] f32, b' [out] f32)."""
    W64 = W.astype(np.float64)
    Wp = W64 @ U_scaled.T                      # [out, rank]
    bp = W64 @ mu + b.astype(np.float64)       # [out]
    return Wp.astype(np.float32), bp.astype(np.float32)


# ------------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser(description="Prune MiniMax-H3 AdaLN to the rank-8 curve-table form")
    ap.add_argument("--src", required=True, help="RAW safetensors transformer/ dir (the factors come from here)")
    ap.add_argument("--in", dest="src_gguf", required=True, help="input DiT GGUF")
    ap.add_argument("--out", help="output GGUF (required unless --dry-run)")
    ap.add_argument("--rank", type=int, default=RANK)
    ap.add_argument("--grid", type=int, default=GRID)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true", help="read the result back and check it")
    ap.add_argument("--verify-only", action="store_true",
                    help="skip writing; just verify the existing --out")
    ap.add_argument("--verify-rows", type=int, default=4096,
                    help="output rows sampled per tensor when verifying (0 = all, slow)")
    a = ap.parse_args()
    if not a.dry_run and not a.out:
        ap.error("--out is required unless --dry-run")
    if a.verify_only:
        if not a.out:
            ap.error("--verify-only needs --out (the file to check)")
        verify(a.out, a.src, a.rank, sample=a.verify_rows, in_path=a.src_gguf)
        return

    st = Safetensors(a.src)
    f, version, kv_blob, n_kv, tensors, data_start, align = gguf_read_header(a.src_gguf)
    print(f"input  {a.src_gguf}")
    print(f"       GGUF v{version}, {len(tensors)} tensors, {n_kv} kv, data at 0x{data_start:x}, align {align}")

    print(f"\nfactorising the AdaLN curve from {a.src}")
    table, U_scaled, mu, S = build_curve_basis(st, a.rank, a.grid)

    # ---- plan the output tensor list -------------------------------------------------
    out = []          # each: dict(name, dims, type, src=('copy', off, nbytes) | ('data', ndarray))
    dropped, rewritten = [], []
    for t in tensors:
        nm = t["name"]
        if nm.startswith("time_embedder."):
            dropped.append(nm)
            continue
        if nm.endswith(ADALN_W) or nm.endswith(ADALN_B):
            rewritten.append(nm)
            continue
        out.append({"name": nm, "dims": t["dims"], "type": t["type"],
                    "src": ("copy", data_start + t["off"], nbytes(t["dims"], t["type"]))})

    prefixes = sorted({nm[: -len(ADALN_W)] for nm in rewritten if nm.endswith(ADALN_W)})
    if not prefixes:
        raise SystemExit("no *.adaln_proj.linear.weight in the input -- already pruned?")
    for nm in rewritten:
        if nm.endswith(ADALN_B) and nm[: -len(ADALN_B)] not in prefixes:
            raise SystemExit(f"{nm} has no matching weight")

    # adaln_t_table first, so it is cheap to find in a hexdump.
    out.insert(0, {"name": "adaln_t_table",
                   "dims": [a.rank, a.grid],                  # ggml ne: [k, grid]
                   "type": GGML_TYPE_F32,
                   "src": ("data", np.ascontiguousarray(table, dtype=np.float32))})

    print(f"\n  {len(dropped)} tensors dropped ({', '.join(sorted(dropped)) or 'none'})")
    print(f"  {len(prefixes)} adaln projections rewritten to rank {a.rank}")

    old_bytes = sum(nbytes(t["dims"], t["type"]) for t in tensors)
    for p in prefixes:
        W = st.get(p + ADALN_W)                                   # [out, t_dim] raw
        b = st.get(p + ADALN_B)
        Wp, bp = factorise(W, b, U_scaled, mu)
        out.append({"name": p + ADALN_W, "dims": [a.rank, Wp.shape[0]], "type": GGML_TYPE_F32,
                    "src": ("data", np.ascontiguousarray(Wp))})
        out.append({"name": p + ADALN_B, "dims": [bp.shape[0]], "type": GGML_TYPE_F32,
                    "src": ("data", np.ascontiguousarray(bp))})
        print(f"    {p or '<root>':<16} out={Wp.shape[0]:>6}  ->  W'[{Wp.shape[0]}, {a.rank}]"
              f" + b'[{bp.shape[0]}]  f32", flush=True)
        del W, b

    new_bytes = 0
    for t in out:
        new_bytes += nbytes(t["dims"], t["type"]) if t["src"][0] == "copy" else t["src"][1].nbytes
    print(f"\n  tensor payload {old_bytes/1e9:.2f} GB -> {new_bytes/1e9:.2f} GB "
          f"({100.0*new_bytes/old_bytes:.1f}%)")

    if a.dry_run:
        print("\ndry run -- nothing written.")
        return

    # ---- write ------------------------------------------------------------------------
    tmp = a.out + ".partial"
    with open(tmp, "wb") as g:
        g.write(b"GGUF")
        g.write(struct.pack("<I", version))
        g.write(struct.pack("<Q", len(out)))
        g.write(struct.pack("<Q", n_kv))
        g.write(kv_blob)
        # offsets are relative to data_start and must be `align`-aligned
        off, entries = 0, []
        for t in out:
            sz = nbytes(t["dims"], t["type"]) if t["src"][0] == "copy" else t["src"][1].nbytes
            entries.append((t, off, sz))
            off += sz + ((-sz) % align)
        for t, o, _sz in entries:
            g.write(pack_str(t["name"]))
            g.write(struct.pack("<I", len(t["dims"])))
            for d in t["dims"]:
                g.write(struct.pack("<Q", int(d)))
            g.write(struct.pack("<I", t["type"]))
            g.write(struct.pack("<Q", o))
        pos = g.tell()
        g.write(b"\x00" * ((-pos) % align))
        base = g.tell()
        for i, (t, o, sz) in enumerate(entries):
            assert g.tell() - base == o, f"offset drift at {t['name']}"
            if t["src"][0] == "copy":
                _, src_off, n = t["src"]
                f.seek(src_off)
                left = n
                while left:
                    chunk = f.read(min(left, 1 << 24))
                    if not chunk:
                        raise SystemExit(f"{t['name']}: short read from the input GGUF")
                    g.write(chunk)
                    left -= len(chunk)
            else:
                g.write(t["src"][1].tobytes())
            g.write(b"\x00" * ((-sz) % align))
            if (i % 50) == 0:
                print(f"    ... {i}/{len(entries)}", end="\r", flush=True)
    os.replace(tmp, a.out)
    print(f"\nwrote {a.out}  ({os.path.getsize(a.out)/1e9:.2f} GB)")

    if a.verify:
        verify(a.out, a.src, a.rank, sample=a.verify_rows, in_path=a.src_gguf)


def verify_copies(out_path, in_path, n_check=12, seed=0):
    """Prove the tensors we did NOT touch came across byte-for-byte.

    The factorisation check below only looks at AdaLN.  An offset bug in the writer would
    leave AdaLN perfect and silently shift the attention/MLP payload -- which is 11 GB of
    NVFP4 that no numerical check here would ever look at.  Compare raw bytes instead."""
    fi, _v, _kv, _n, ti, di, _ai = gguf_read_header(in_path)
    fo, _v, _kv, _n, to, do, _ao = gguf_read_header(out_path)
    src_by = {t["name"]: t for t in ti}
    copied = [t for t in to if t["name"] in src_by
              and not t["name"].endswith((ADALN_W, ADALN_B)) and t["name"] != "adaln_t_table"]
    rng = np.random.default_rng(seed)
    pick = [copied[i] for i in rng.choice(len(copied), size=min(n_check, len(copied)), replace=False)]
    # always include the largest copied tensor -- the most offset-sensitive one
    pick.append(max(copied, key=lambda t: nbytes(t["dims"], t["type"])))
    print(f"\n  byte-identity of untouched tensors ({len(copied)} copied, checking {len(pick)}):")
    bad = 0
    for t in pick:
        s = src_by[t["name"]]
        if s["dims"] != t["dims"] or s["type"] != t["type"]:
            print(f"    MISMATCH shape/type: {t['name']}")
            bad += 1
            continue
        n = nbytes(t["dims"], t["type"])
        fi.seek(di + s["off"])
        fo.seek(do + t["off"])
        left, ok = n, True
        while left:
            k = min(left, 1 << 22)
            if fi.read(k) != fo.read(k):
                ok = False
                break
            left -= k
        tname = GGML_TYPES[t["type"]][0]
        print(f"    {'OK ' if ok else 'BAD'} {t['name']:<52} {tname:<6} {n/1e6:9.2f} MB")
        bad += 0 if ok else 1
    if bad:
        raise SystemExit(f"{bad} copied tensors do NOT match the input -- the writer is wrong")
    print("    all sampled copies are byte-identical")


def verify(path, src, rank, sample=4096, seed=0, in_path=None):
    """Read the written file back and compare its modulation to the exact one.

    Compares on a fixed random SUBSET of `sample` output rows per tensor.  The full product
    is [1025 x 96768] per tensor -- 266 GFLOP x 51 -- which this box's reference BLAS would
    grind on for over an hour.  Relative Frobenius error is a ratio of sums over independent
    output rows, so a few thousand of them pin it down far tighter than the 1e-3 threshold
    below; `sample=0` forces the exhaustive comparison if you ever want it."""
    print(f"\n=== verify {path} ===")
    st = Safetensors(src)
    f, version, kv_blob, n_kv, tensors, data_start, align = gguf_read_header(path)
    by = {t["name"]: t for t in tensors}
    if "adaln_t_table" not in by:
        raise SystemExit("adaln_t_table missing from the output -- the engine would take the FULL path")
    if any(n.startswith("time_embedder.") for n in by):
        raise SystemExit("time_embedder.* survived -- the output is inconsistent")

    def read(nm):
        t = by[nm]
        tname, blk, tsz = GGML_TYPES[t["type"]]
        assert tname == "f32", f"{nm} is {tname}, expected f32"
        f.seek(data_start + t["off"])
        n = 1
        for d in t["dims"]:
            n *= d
        return np.frombuffer(f.read(n * 4), dtype="<f4").reshape(list(reversed(t["dims"])))

    table = read("adaln_t_table")                     # [grid, rank]
    grid = table.shape[0]
    print(f"  adaln_t_table {table.shape}  rank={table.shape[1]}  rms={table.std():.5f}")

    # rebuild the exact curve independently, the way the ENGINE would index it
    pi_w = st.get("time_embedder.proj_in.weight").astype(np.float64)
    pi_b = st.get("time_embedder.proj_in.bias").astype(np.float64)
    po_w = st.get("time_embedder.proj_out.weight").astype(np.float64)
    po_b = st.get("time_embedder.proj_out.bias").astype(np.float64)
    t = np.arange(grid, dtype=np.float64) / (grid - 1)
    F = silu(silu(sinusoid(t, pi_w.shape[1]) @ pi_w.T + pi_b) @ po_w.T + po_b)

    worst = 0.0
    rng = np.random.default_rng(seed)
    for nm in sorted(n for n in by if n.endswith(ADALN_W)):
        p = nm[: -len(ADALN_W)]
        Wp = read(nm)                                  # [out, rank]
        bp = read(p + ADALN_B)
        W = st.get(p + ADALN_W)                        # [out, t_dim]
        b = st.get(p + ADALN_B)
        n_out = W.shape[0]
        rows = (slice(None) if not sample or sample >= n_out
                else rng.choice(n_out, size=sample, replace=False))
        Ws = W[rows].astype(np.float64)
        exact = F @ Ws.T + b[rows].astype(np.float64)                       # [grid, n_sel]
        got = (table.astype(np.float64) @ Wp[rows].T.astype(np.float64)
               + bp[rows].astype(np.float64))
        rel = np.linalg.norm(exact - got) / np.linalg.norm(exact)
        worst = max(worst, rel)
        tag = "all" if isinstance(rows, slice) else f"{sample} of {n_out}"
        print(f"  {p or '<root>':<16} out={n_out:>6} ({tag:>14} rows)  rel L2 = {rel:.4e}"
              f"   max abs = {np.abs(exact - got).max():.3e}", flush=True)
        del W, Ws, exact, got
    print(f"\n  WORST relative modulation error: {worst:.4e}")
    if worst > 1e-3:
        raise SystemExit("verification FAILED")
    if in_path:
        verify_copies(path, in_path)
    print("  OK")


if __name__ == "__main__":
    main()
