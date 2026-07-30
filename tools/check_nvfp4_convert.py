#!/usr/bin/env python3
"""Validate a flat `-M convert --type nvfp4` gguf against its bf16 safetensors source.

WHY THIS EXISTS: `nvfp4-CLEAN.gguf` was NOT produced by a convert. Git history (d685d8c)
shows it came from `tools/import_ltx_nvfp4.py`, a transcoder for NVIDIA's *official ModelOpt
NVFP4* release -- which is why it carries 1344 `<weight>.wglobal` sidecars. A flat ggml
convert from bf16 is a DIFFERENT code path with no precedent for LTX in this tree, so its
output needs checking rather than trusting.

A byte-diff proves nothing (see project_nvfp4_lora_unfold_is_inert). Project instead.

Usage:
  # the control: convert the OFFICIAL distilled bf16 and check it against its own source,
  # then against the known-good import. If the method is sound, both agree.
  ./check_nvfp4_convert.py <converted.gguf> /mnt/hdd/ltx2-official/ltx-2.3-22b-distilled-1.1.safetensors
  ./check_nvfp4_convert.py <converted.gguf> --vs-gguf models/ltx2/nvfp4-CLEAN.gguf

Reference numbers to beat: the import scored ratio 0.9963 against its source
(project_ltx_dev050_was_never_half_distilled). A sound flat convert should land in the
same neighbourhood. Projection well under ~0.99, or a rel-error above a few percent, means
the convert is losing the weights and the fallback is a K-quant/Q8_0 convert instead.
"""
import argparse, collections, json, struct, sys
import numpy as np

QK, SUB = 64, 16
GGML_F32, GGML_BF16, GGML_NVFP4 = 0, 30, 40

# ---- E2M1 (4-bit) and UE4M3 (8-bit, unsigned) decode tables -------------------
_E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
_E2M1 = np.concatenate([_E2M1, -_E2M1])  # sign bit is bit 3


def _ue4m3_table():
    u = np.arange(256, dtype=np.int32)
    e, m = (u >> 3) & 0xF, u & 0x7
    sub = np.ldexp(m.astype(np.float64), -9)          # e == 0: 2^-6 * m/8
    nrm = np.ldexp(1.0 + m / 8.0, e - 7)
    return np.where(e == 0, sub, nrm).astype(np.float32)


_UE4M3 = _ue4m3_table()


def dequant_nvfp4(raw: np.ndarray, n_elem: int) -> np.ndarray:
    """block_nvfp4 = 4x UE4M3 sub-block scales + 32B of packed E2M1 (ggml-common.h:221)."""
    nb = n_elem // QK
    b = raw[: nb * 36].reshape(nb, 36)
    scales = _UE4M3[b[:, :4]]                              # [nb, 4]
    qs = b[:, 4:]                                          # [nb, 32]
    out = np.empty((nb, QK), dtype=np.float32)
    for s in range(QK // SUB):                             # 4 sub-blocks of 16
        byts = qs[:, s * 8:(s + 1) * 8]                    # 8 bytes -> 16 values
        out[:, s * SUB: s * SUB + 8] = _E2M1[byts & 0x0F]  # low nibble  = first half
        out[:, s * SUB + 8: (s + 1) * SUB] = _E2M1[byts >> 4]
        out[:, s * SUB:(s + 1) * SUB] *= scales[:, s: s + 1]
    return out.reshape(-1)[:n_elem]


# ---- readers -----------------------------------------------------------------
def read_gguf(path):
    f = open(path, "rb")
    assert f.read(4) == b"GGUF", "not a gguf"
    struct.unpack("<I", f.read(4))
    nt, nkv = struct.unpack("<QQ", f.read(16))

    def rstr():
        n, = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8", "replace")

    def skipv(t):
        fixed = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if t in fixed:
            f.read(fixed[t])
        elif t == 8:
            rstr()
        elif t == 9:
            et, = struct.unpack("<I", f.read(4))
            n, = struct.unpack("<Q", f.read(8))
            for _ in range(n):
                skipv(et)
        else:
            raise ValueError(f"unknown gguf value type {t}")

    align = 32
    for _ in range(nkv):
        k = rstr()
        t, = struct.unpack("<I", f.read(4))
        if k == "general.alignment" and t == 4:
            align, = struct.unpack("<I", f.read(4))
        else:
            skipv(t)

    infos = {}
    for _ in range(nt):
        name = rstr()
        nd, = struct.unpack("<I", f.read(4))
        dims = struct.unpack("<%dQ" % nd, f.read(8 * nd))
        ty, = struct.unpack("<I", f.read(4))
        off, = struct.unpack("<Q", f.read(8))
        infos[name] = (dims, ty, off)
    pos = f.tell()
    data_start = (pos + align - 1) // align * align
    return f, data_start, infos


def gguf_tensor(f, base, infos, name):
    dims, ty, off = infos[name]
    n = int(np.prod(dims))
    f.seek(base + off)
    if ty == GGML_NVFP4:
        return dequant_nvfp4(np.frombuffer(f.read(n // QK * 36), dtype=np.uint8), n), dims
    if ty == GGML_F32:
        return np.frombuffer(f.read(n * 4), dtype=np.float32).copy(), dims
    if ty == GGML_BF16:
        u = np.frombuffer(f.read(n * 2), dtype=np.uint16).astype(np.uint32) << 16
        return u.view(np.float32).copy(), dims
    raise ValueError(f"{name}: unhandled ggml type {ty}")


def st_open(path):
    f = open(path, "rb")
    n, = struct.unpack("<Q", f.read(8))
    h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return f, 8 + n, h


def st_tensor(f, base, h, key):
    m = h[key]
    a, b = m["data_offsets"]
    f.seek(base + a)
    raw = np.frombuffer(f.read(b - a), dtype=np.uint8)
    if m["dtype"] == "BF16":
        u = raw.view(np.uint16).astype(np.uint32) << 16
        return u.view(np.float32).copy()
    if m["dtype"] == "F32":
        return raw.view(np.float32).copy()
    raise ValueError(f"{key}: unhandled safetensors dtype {m['dtype']}")


# ---- main --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("source", nargs="?", help="bf16 safetensors the gguf was converted from")
    ap.add_argument("--vs-gguf", help="compare against a known-good gguf instead")
    ap.add_argument("--limit", type=int, default=24, help="tensors to sample (0 = all)")
    ap.add_argument("--prefix", default="model.diffusion_model.")
    a = ap.parse_args()

    f, base, infos = read_gguf(a.gguf)
    nvfp4 = [n for n, (_, t, _) in infos.items() if t == GGML_NVFP4]
    types = collections.Counter(t for _, t, _ in infos.values())
    has_wglobal = any(n.endswith(".wglobal") for n in infos)
    print(f"{a.gguf}: {len(infos)} tensors, types={dict(types)}, "
          f"nvfp4={len(nvfp4)}, wglobal sidecars={'YES' if has_wglobal else 'NONE'}")
    if not nvfp4:
        print("!! no nvfp4 tensors -- the convert did not apply the type")
        return 1

    if a.vs_gguf:
        g2, b2, i2 = read_gguf(a.vs_gguf)

        # ★ The reference's OWN `.wglobal` must be applied, exactly as it is for the
        # primary file below. Omitting it silently rescales the reference by
        # 1/wglobal (~2.7e3 for LTX), so an IDENTICAL pair projects at ~4e-4 and the
        # tool reports "the convert is losing the weights" on a perfect file. Both
        # nvfp4-CLEAN.gguf and every route-B build carry sidecars, so this hit the
        # exact comparison the docstring recommends.
        def get_ref(n):
            r = gguf_tensor(g2, b2, i2, n)[0]
            if n + ".wglobal" in i2:
                r = r * gguf_tensor(g2, b2, i2, n + ".wglobal")[0][0]
            return r

        missing = lambda n: n not in i2
        label = a.vs_gguf
    else:
        if not a.source:
            print("!! need a source safetensors or --vs-gguf")
            return 2
        sf, sb, sh = st_open(a.source)

        # Naming is NOT consistent between producers: the ModelOpt import writes BARE
        # names (`transformer_blocks.0...`), while `-M convert -m <full checkpoint>`
        # preserves the source's `model.diffusion_model.` prefix. Try both.
        def _key(n):
            for cand in (n, a.prefix + n, n[len(a.prefix):] if n.startswith(a.prefix) else None):
                if cand is not None and cand in sh:
                    return cand
            return None

        get_ref = lambda n: st_tensor(sf, sb, sh, _key(n))
        missing = lambda n: _key(n) is None
        label = a.source

    picks = nvfp4 if a.limit == 0 else nvfp4[:: max(1, len(nvfp4) // a.limit)][: a.limit]
    print(f"comparing {len(picks)} tensor(s) against {label}\n")
    print(f"{'tensor':62s} {'projection':>11s} {'rel_err':>9s}")
    projections = []
    for name in picks:
        if missing(name):
            print(f"{name:62s} {'MISSING IN REF':>21s}")
            continue
        q, dims = gguf_tensor(f, base, infos, name)
        w = infos.get(name + ".wglobal")
        if w is not None:
            q = q * gguf_tensor(f, base, infos, name + ".wglobal")[0][0]
        ref = get_ref(name)
        if ref.size != q.size:
            print(f"{name:62s} size {ref.size} vs {q.size} -- SHAPE MISMATCH")
            continue
        denom = float(ref @ ref)
        proj = float(q @ ref) / denom if denom > 0 else float("nan")
        rel = float(np.linalg.norm(q - ref) / (np.linalg.norm(ref) + 1e-12))
        projections.append(proj)
        print(f"{name:62s} {proj:11.4f} {rel:9.4f}")

    if projections:
        p = np.array(projections)
        print(f"\nprojection: mean {p.mean():.4f}  min {p.min():.4f}  max {p.max():.4f}")
        print("reference: the ModelOpt import scored 0.9963 against its source.")
        if p.mean() < 0.98:
            print("VERDICT: SUSPECT -- the convert is losing the weights. Try a K-quant/Q8_0 "
                  "convert, or remap one of the ready-made Q8_0 GGUFs instead.")
        else:
            print("VERDICT: in family with the import. Proceed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
