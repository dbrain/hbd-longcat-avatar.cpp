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

GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8
QK8_0 = 32


def st_open(p):
    f = open(p, 'rb'); n = struct.unpack('<Q', f.read(8))[0]
    h = json.loads(f.read(n)); h.pop('__metadata__', None); return f, 8 + n, h


def st_get(f, base, h, k):
    m = h[k]; a, b = m['data_offsets']; f.seek(base + a)
    raw = np.frombuffer(f.read(b - a), dtype=np.uint8)
    if m['dtype'] == 'BF16':
        x = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    elif m['dtype'] in ('F16', 'FP16'):
        # ai-toolkit writes the krea2_edit adapters in F16; view(float32) here would
        # silently halve the tensor and read pairs of halves as garbage floats.
        x = raw.view(np.float16).astype(np.float32)
    else:
        x = raw.view(np.float32)
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
        assert tt in (GGML_TYPE_Q8_0, GGML_TYPE_F16), f"{name}: type {tt} not Q8_0/F16"
        n = int(np.prod(ne))
        if tt == GGML_TYPE_F16:
            fg.seek(dstart + off)
            w = np.frombuffer(fg.read(n * 2), dtype=np.float16).astype(np.float64)
            w = w.reshape(ne[1], ne[0]) if len(ne) > 1 else w
            # F16 has 5 exponent bits vs BF16's 8: in-range values gain precision (10 vs 7
            # mantissa bits) but anything past +-65504 saturates and denormals below ~6e-8
            # flush. LoRA weights sit around 1e-3 so both are far away — but CHECK, don't assume.
            if not np.isfinite(w).all():
                raise AssertionError(f"{name}: non-finite after F16 cast (BF16 range overflow)")
        else:
            nb = n // QK8_0 * (2 + QK8_0)
            fg.seek(dstart + off)
            w = dequant_q8(fg.read(nb), ne)
        if a.corrupt and name.endswith('.lora_B.weight'):
            if a.corrupt == 'inert':
                w = np.zeros_like(w)
            elif a.corrupt == 'gain':
                w = w * 0.5
            else:  # bitrot: damage 5% of rows, deterministically
                w = w.copy()
                w[::20] = 0.0
        return w

    # A module whose SOURCE delta is exactly zero (lora_B never trained off its zero init —
    # Best_FaceID_v1.0 ships 864 such modules) has no direction to compare against: relerr,
    # projection and cosine are all 0/0 = nan. Those must NOT enter the means, because every
    # threshold here is written `if mean > limit: FAIL` and **`nan > x` is False** — a single
    # nan used to sail through as a green PASS. Inert modules are checked separately (Q8 must
    # keep them exactly zero) and any nan from a LIVE module is a hard failure.
    rel, prj, cs = [], [], []
    inert_ok, inert_bad = 0, []
    for s in sample:
        A0 = st_get(fs_, base, h, s + '.lora_A.weight')
        B0 = st_get(fs_, base, h, s + '.lora_B.weight')
        A1 = load_q8(s + '.lora_A.weight')
        B1 = load_q8(s + '.lora_B.weight')
        d0 = B0 @ A0
        d1 = B1 @ A1
        n0 = np.linalg.norm(d0)
        label = s[len('diffusion_model.'):][:51]
        if not n0:
            # inert by construction: the only correct Q8 image of 0 is 0
            if np.any(d1):
                inert_bad.append(s)
                print(f"{label:<52}{A0.shape[0]:>6}{'INERT':>10}{'NONZERO':>8}{'BAD':>8}")
            else:
                inert_ok += 1
                print(f"{label:<52}{A0.shape[0]:>6}{'inert':>10}{'0':>8}{'0':>8}")
            continue
        e = float(np.linalg.norm(d1 - d0) / n0)
        dd = float((d0 * d0).sum())
        p = float((d1 * d0).sum() / dd)
        c = float((d1 * d0).sum() / (np.linalg.norm(d1) * n0))
        rel.append(e); prj.append(p); cs.append(c)
        print(f"{label:<52}{A0.shape[0]:>6}{e:>10.4f}{p:>8.4f}{c:>8.4f}")

    if inert_ok or inert_bad:
        print(f"\n  L0 INERT    : {inert_ok} zero-delta module(s) preserved exactly"
              + (f", {len(inert_bad)} CORRUPTED" if inert_bad else ""))
    if inert_bad:
        fails.append(f"L0 INERT: {len(inert_bad)} zero-delta module(s) became non-zero in the gguf")
        print(f"  L0 INERT    : FAIL")

    if not rel:
        fails.append("L2/L3: every sampled module was inert — nothing was actually verified")
        print("\n  L2 FIDELITY : FAIL (no live modules in the sample)")
        print("  L3 DIRECTION: FAIL (no live modules in the sample)")
        mr = mp = mc = float('nan')
    else:
        mr, mp, mc = float(np.mean(rel)), float(np.mean(prj)), float(np.mean(cs))
        if not all(np.isfinite([mr, mp, mc])):
            fails.append(f"L2/L3: non-finite statistic (relerr {mr}, proj {mp}, cos {mc})")
            print(f"\n  L2/L3       : FAIL (non-finite statistic — refusing to pass)")
        else:
            print(f"\n  L2 FIDELITY : mean relative error {mr:.4f}  (max {max(rel):.4f})"
                  f"  [{len(rel)} live]")
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
