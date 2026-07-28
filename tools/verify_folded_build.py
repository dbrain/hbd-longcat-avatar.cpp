#!/usr/bin/env python3
"""VERIFY a route-B folded build (tools/build_folded_nvfp4.py) against its base gguf.

WHY NOT verify_fold.py: that script reads the .wglobal sidecars from the BASE file and uses
them to dequantise BOTH files. A rebuild-from-bf16 legitimately RECOMPUTES every wglobal, so
verify_fold.py would dequantise the folded file with the wrong per-tensor scale (garbage dW)
and its C5 "wglobal must not change" would fail by design. This script reads each file's OWN
wglobal, and covers BF16 tensors too — which a frozen-grid fold can never touch.

  B1 STRUCTURE   same tensor names/dims/types (wglobal VALUES may differ; that is the point)
  B2 CHANGED     % of dequantised elements that moved, in TRUE units, per storage class
  B3 DIRECTION   projection <dW, mult*(B@A)>/<delta,delta> and cosine. Route B should land
                 ~1.0; route A measures ~0.36. Anything near 0 means inert/wrong scale.
  B4 COLLATERAL  tensors the LoRA does NOT target must be near-identical (they are requantised
                 from the same bf16 source, so expect ~0 change, not necessarily bit-equality)

Usage:
  verify_folded_build.py --base <base.gguf> --folded <folded.gguf> \
                         --lora <lora.safetensors> --mult 1.5
"""
import argparse, json, os, struct, sys
import numpy as np

GT_F32, GT_BF16, GT_NVFP4 = 0, 30, 40
LORA_PFX = "diffusion_model."

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_fold import st_open, st_raw, bf16_to_f32, dequant_block, parse_gguf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--folded", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--mult", type=float, required=True)
    ap.add_argument("--tensors", type=int, default=10)
    ap.add_argument("--min-projection", type=float, default=0.85)
    ap.add_argument("--min-changed", type=float, default=20.0)
    a = ap.parse_args()
    fails = []

    bi, bstart = parse_gguf(a.base)
    fi, fstart = parse_gguf(a.folded)
    bmap = {n: (d, t, o) for (n, d, t, o) in bi}
    fmap = {n: (d, t, o) for (n, d, t, o) in fi}
    print(f"base   {len(bi)} tensors   folded {len(fi)} tensors")

    missing = [n for n in bmap if n not in fmap]
    shape_bad = [n for n in bmap if n in fmap and
                 (bmap[n][0] != fmap[n][0] or bmap[n][1] != fmap[n][1])]
    if missing or shape_bad:
        fails.append(f"B1 STRUCTURE: {len(missing)} missing, {len(shape_bad)} dim/type mismatch")
        print(f"  B1 STRUCTURE: FAIL ({len(missing)} missing, {len(shape_bad)} mismatched)"
              + (f"  e.g. {(missing or shape_bad)[0]}" if (missing or shape_bad) else ""))
    else:
        print(f"  B1 STRUCTURE: PASS (names/dims/types identical)")

    fb, ff = open(a.base, "rb"), open(a.folded, "rb")

    def wg(fh, start, mp, name):
        k = name + ".wglobal"
        if k not in mp:
            return 1.0
        fh.seek(start + mp[k][2]); return struct.unpack("<f", fh.read(4))[0]

    def load(fh, start, mp, name):
        d, t, o = mp[name]
        if t == GT_NVFP4:
            inn, out = d[0], d[1]; nblk = inn // 64
            fh.seek(start + o)
            raw = np.frombuffer(fh.read(out * nblk * 36), dtype=np.uint8).reshape(out, nblk, 36)
            return (dequant_block(raw, out, inn, nblk) * wg(fh, start, mp, name)).astype(np.float64)
        ne = int(np.prod(d))
        fh.seek(start + o)
        if t == GT_BF16:
            x = bf16_to_f32(np.frombuffer(fh.read(ne * 2), dtype=np.uint8))
        else:
            x = np.frombuffer(fh.read(ne * 4), dtype=np.float32)
        return x.reshape(d[1], d[0]).astype(np.float64) if len(d) == 2 else x.astype(np.float64)

    fl, bl, hl = st_open(a.lora)
    stems = {k[:-len(".lora_A.weight")] for k in hl if k.endswith(".lora_A.weight")}
    tgt = [n for n in bmap if not n.endswith(".wglobal")
           and n.endswith(".weight") and LORA_PFX + n[:-len(".weight")] in stems]
    non = [n for n in bmap if not n.endswith(".wglobal")
           and n.endswith(".weight") and LORA_PFX + n[:-len(".weight")] not in stems
           and bmap[n][1] == GT_NVFP4]
    nv = [n for n in tgt if bmap[n][1] == GT_NVFP4]
    bf = [n for n in tgt if bmap[n][1] != GT_NVFP4]
    print(f"  lora targets: {len(tgt)}  ({len(nv)} nvfp4, {len(bf)} bf16/f32 — "
          f"the {len(bf)} a frozen-grid fold CANNOT touch)")

    def sample(lst, k):
        if not lst: return []
        idx = np.linspace(0, len(lst) - 1, min(k, len(lst))).astype(int)
        return [lst[i] for i in sorted(set(idx.tolist()))]

    print(f"\n{'tensor':<50}{'cls':>7}{'%chg':>9}{'proj':>8}{'cos':>8}")
    pcts, projs, coss = [], [], []
    for name in sample(nv, a.tensors) + sample(bf, max(2, a.tensors // 3)):
        Wb = load(fb, bstart, bmap, name)
        Wf = load(ff, fstart, fmap, name)
        stem = LORA_PFX + name[:-len(".weight")]
        A = bf16_to_f32(st_raw(fl, bl, hl, stem + ".lora_A.weight")[0]).reshape(
            hl[stem + ".lora_A.weight"]["shape"]).astype(np.float64)
        B = bf16_to_f32(st_raw(fl, bl, hl, stem + ".lora_B.weight")[0]).reshape(
            hl[stem + ".lora_B.weight"]["shape"]).astype(np.float64)
        delta = a.mult * (B @ A)
        dW = Wf - Wb
        pct = float((dW != 0).mean()) * 100.0
        dd = float((delta * delta).sum())
        proj = float((dW * delta).sum() / dd) if dd else 0.0
        nd, nl = np.linalg.norm(dW), np.linalg.norm(delta)
        cos = float((dW * delta).sum() / (nd * nl)) if nd and nl else 0.0
        cls = "nvfp4" if bmap[name][1] == GT_NVFP4 else "bf16"
        pcts.append(pct); projs.append(proj); coss.append(cos)
        print(f"{name[:49]:<50}{cls:>7}{pct:>8.2f}%{proj:>8.3f}{cos:>8.3f}")

    mp_, mc, mchg = float(np.mean(projs)), float(np.mean(coss)), float(np.mean(pcts))
    print(f"\n  B2 CHANGED   : mean {mchg:.2f}% of elements moved")
    if mchg < a.min_changed:
        fails.append(f"B2 CHANGED: {mchg:.2f}% < {a.min_changed}%"); print("  B2 CHANGED   : FAIL")
    else:
        print(f"  B2 CHANGED   : PASS (>= {a.min_changed}%)")
    print(f"  B3 DIRECTION : projection {mp_:.4f}  cosine {mc:.4f}")
    print(f"       route A (frozen grid) measures ~0.36 here; route B should be ~1.0.")
    if mp_ < a.min_projection:
        fails.append(f"B3 DIRECTION: projection {mp_:.4f} < {a.min_projection}")
        print("  B3 DIRECTION : FAIL")
    else:
        print(f"  B3 DIRECTION : PASS (>= {a.min_projection})")

    worst = 0.0
    for name in sample(non, 5):
        Wb = load(fb, bstart, bmap, name); Wf = load(ff, fstart, fmap, name)
        den = np.sqrt((Wb * Wb).mean())
        worst = max(worst, float(np.sqrt(((Wf - Wb) ** 2).mean()) / den) if den else 0.0)
    print(f"  B4 COLLATERAL: worst untargeted relative change {worst:.4f}")
    if worst > 0.05:
        fails.append(f"B4 COLLATERAL: untargeted tensors moved {worst:.4f}")
        print("  B4 COLLATERAL: FAIL")
    else:
        print("  B4 COLLATERAL: PASS (<= 0.05)")

    fb.close(); ff.close(); print()
    if fails:
        print("=========== VERIFY: FAIL ===========")
        for x in fails: print("  " + x)
        sys.exit(1)
    print("=========== VERIFY: PASS ===========")
    sys.exit(0)


if __name__ == "__main__":
    main()
