#!/usr/bin/env python3
"""VERIFY a Q8_0 LoRA gguf against the BF16 safetensors it came from.

The lesson from the 2026-07 inert-fold family: coverage counts and byte-diffs prove nothing.
So this checks the only thing that matters — that the DELTA THE ENGINE WILL APPLY, B@A, is
still the delta the LoRA encodes.

  L1 STRUCTURE  every source tensor is present in the gguf, with the right reversed dims.
  L2 FIDELITY   per-layer relative error of the reconstructed delta (B@A), Q8 vs BF16.
  L3 DIRECTION  projection <d_q8, d_bf16>/<d_bf16, d_bf16> and cosine.
                Unlike an nvfp4 FOLD (~0.36, round-to-nearest kills sub-half-step components),
                a Q8 requant of A and B should land ~1.000 with cosine ~1.000. Anything else
                means the container, the dim order or the quantiser is wrong.

Validate this script the way verify_fold.py was validated — make it FAIL a known-bad file
(--corrupt flips one Q8 scale) before trusting it to PASS a real one.

Usage:
  verify_lora_q8.py --src <in.safetensors> --gguf <out.gguf> [--tensors 12] [--corrupt]
"""
import sys, json, struct, argparse
import numpy as np

GGML_TYPE_Q8_0 = 8
QK8_0 = 32


def st_open(p):
    f = open(p, 'rb'); n = struct.unpack('<Q', f.read(8))[0]
    h = json.loads(f.read(n)); h.pop('__metadata__', None); return f, 8 + n, h


def st_get(f, base, h, k):
    m = h[k]; a, b = m['data_offsets']; f.seek(base + a)
    raw = np.frombuffer(f.read(b - a), dtype=np.uint8)
    x = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32) if m['dtype'] == 'BF16' \
        else raw.view(np.float32)
    return x.reshape(m['shape']).astype(np.float64)


def parse_gguf(path):
    f = open(path, 'rb')
    assert f.read(4) == b'GGUF', f"{path}: not a gguf"
    struct.unpack('<I', f.read(4))[0]
    nt = struct.unpack('<Q', f.read(8))[0]; nkv = struct.unpack('<Q', f.read(8))[0]
    assert nkv == 0, f"{path}: expected nkv=0, got {nkv}"
    infos = {}
    for _ in range(nt):
        kl = struct.unpack('<Q', f.read(8))[0]; name = f.read(kl).decode()
        nd = struct.unpack('<I', f.read(4))[0]
        ne = [struct.unpack('<Q', f.read(8))[0] for _ in range(nd)]
        tt = struct.unpack('<I', f.read(4))[0]; off = struct.unpack('<Q', f.read(8))[0]
        infos[name] = (ne, tt, off)
    cur = f.tell(); f.close()
    return infos, cur + ((-cur) % 32)


def dequant_q8(raw, ne):
    """raw bytes -> [rows, k] float64 (k = ne[0])."""
    k = ne[0]; rows = int(np.prod(ne[1:])) if len(ne) > 1 else 1
    nb = k // QK8_0
    b = np.frombuffer(raw, dtype=np.uint8).reshape(rows * nb, 2 + QK8_0)
    d = b[:, 0:2].copy().view(np.float16).astype(np.float64).reshape(-1, 1)
    q = b[:, 2:].view(np.int8).astype(np.float64)
    return (q * d).reshape(rows, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--gguf', required=True)
    ap.add_argument('--tensors', type=int, default=12)
    ap.add_argument('--min-projection', type=float, default=0.97)
    ap.add_argument('--max-relerr', type=float, default=0.02)
    ap.add_argument('--corrupt', choices=['inert', 'gain', 'bitrot'],
                    help='self-test: damage lora_B in memory; the run MUST fail. '
                         'inert = zero it (the 2026-07 silently-inert-fold failure); '
                         'gain = halve every block scale (wrong-strength fold); '
                         'bitrot = flip 5%% of block scales (container/offset damage). '
                         'A self-test that still PASSES means the checker is worthless.')
    a = ap.parse_args()

    fs_, base, h = st_open(a.src)
    infos, dstart = parse_gguf(a.gguf)
    fails = []

    missing = [k for k in h if k not in infos]
    if missing:
        fails.append(f"L1 STRUCTURE: {len(missing)} source tensors absent from gguf "
                     f"(e.g. {missing[:2]})")
        print(f"  L1 STRUCTURE: FAIL ({len(missing)} missing)")
    else:
        bad_dims = [k for k in h if list(infos[k][0]) != list(reversed(h[k]['shape']))]
        if bad_dims:
            fails.append(f"L1 STRUCTURE: {len(bad_dims)} tensors have wrong ggml dim order "
                         f"(e.g. {bad_dims[0]})")
            print(f"  L1 STRUCTURE: FAIL (dim order, {len(bad_dims)} tensors)")
        else:
            print(f"  L1 STRUCTURE: PASS ({len(h)} tensors, dims reversed correctly)")

    stems = sorted({k[:-len('.lora_A.weight')] for k in h if k.endswith('.lora_A.weight')})
    stems = [s for s in stems if s + '.lora_B.weight' in h]
    idx = np.linspace(0, len(stems) - 1, min(a.tensors, len(stems))).astype(int)
    sample = [stems[i] for i in sorted(set(idx.tolist()))]
    print(f"\n--- L2 FIDELITY / L3 DIRECTION over {len(sample)} of {len(stems)} modules ---")
    print(f"{'module':<52}{'rank':>6}{'relerr':>10}{'proj':>8}{'cos':>8}")

    fg = open(a.gguf, 'rb')

    def load_q8(name):
        ne, tt, off = infos[name]
        assert tt == GGML_TYPE_Q8_0, f"{name}: type {tt} != Q8_0"
        nb = int(np.prod(ne)) // QK8_0 * (2 + QK8_0)
        fg.seek(dstart + off)
        raw = bytearray(fg.read(nb))
        w = dequant_q8(bytes(raw), ne)
        if a.corrupt and name.endswith('.lora_B.weight'):
            if a.corrupt == 'inert':
                w = np.zeros_like(w)
            elif a.corrupt == 'gain':
                w = w * 0.5
            else:  # bitrot: damage 5% of rows, deterministically
                w = w.copy()
                w[::20] = 0.0
        return w

    rel, prj, cs = [], [], []
    for s in sample:
        A0 = st_get(fs_, base, h, s + '.lora_A.weight')
        B0 = st_get(fs_, base, h, s + '.lora_B.weight')
        A1 = load_q8(s + '.lora_A.weight')
        B1 = load_q8(s + '.lora_B.weight')
        d0 = B0 @ A0
        d1 = B1 @ A1
        n0 = np.linalg.norm(d0)
        e = float(np.linalg.norm(d1 - d0) / n0) if n0 else float('nan')
        dd = float((d0 * d0).sum())
        p = float((d1 * d0).sum() / dd) if dd else float('nan')
        c = float((d1 * d0).sum() / (np.linalg.norm(d1) * n0)) if n0 else float('nan')
        rel.append(e); prj.append(p); cs.append(c)
        print(f"{s[len('diffusion_model.'):][:51]:<52}{A0.shape[0]:>6}{e:>10.4f}{p:>8.4f}{c:>8.4f}")

    mr, mp, mc = float(np.mean(rel)), float(np.mean(prj)), float(np.mean(cs))
    print(f"\n  L2 FIDELITY : mean relative error {mr:.4f}  (max {max(rel):.4f})")
    if mr > a.max_relerr:
        fails.append(f"L2 FIDELITY: mean relative error {mr:.4f} > {a.max_relerr}")
        print(f"  L2 FIDELITY : FAIL")
    else:
        print(f"  L2 FIDELITY : PASS (<= {a.max_relerr})")
    print(f"  L3 DIRECTION: mean projection {mp:.4f}, mean cosine {mc:.4f}")
    if mp < a.min_projection:
        fails.append(f"L3 DIRECTION: projection {mp:.4f} < {a.min_projection}")
        print(f"  L3 DIRECTION: FAIL")
    else:
        print(f"  L3 DIRECTION: PASS (>= {a.min_projection})")

    print()
    if fails:
        print("=========== VERIFY: FAIL ===========")
        for f_ in fails:
            print("  " + f_)
        sys.exit(1)
    print("=========== VERIFY: PASS ===========")
    sys.exit(0)


if __name__ == '__main__':
    main()
