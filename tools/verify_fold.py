#!/usr/bin/env python3
"""VERIFY that an nvfp4 LoRA fold actually changed the model. Run this on EVERY folded
gguf before deploying it.

WHY THIS EXISTS
---------------
2026-07-16..27: every LTX-2.3 nvfp4 "variant" gguf on disk (dev050, lipdub, audioreactive,
omninft, faceid-sheet) was byte-different from its base but WEIGHT-IDENTICAL to it. The fold
tool printed "1344/1344 folded, success" and a byte-diff showed 6-7% of bytes differing, so
nothing looked wrong. In fact 100% of the differing bytes were redundant +0/-0 e2m1 encodings
and not one dequantised weight had moved.

Root cause: the base ggufs use the UNFOLDED nvfp4 convention where
    true_weight = dequant_block(...) * wglobal        (wglobal ~ 1e-5 .. 4e-3)
and the fold tool added the LoRA delta — which is in TRUE weight units — straight onto the
BLOCK-domain dequant, attenuating it by wglobal (~2000-19000x). The delta ended up at
0.0003%-0.018% of one quantisation step, so no code could ever move.

THE LESSONS THIS SCRIPT ENCODES
-------------------------------
1. A byte-diff proves NOTHING. Compare DEQUANTISED VALUES.
2. Coverage counts ("480/480 modules folded") measure targets, not effect.
3. Dequantise in TRUE units — block_dq * wglobal — or you are measuring the wrong domain.
4. "Something changed" is not enough either: check the change is the RIGHT change, by
   projecting the observed weight delta onto the LoRA's expected delta.

CHECKS
------
  C1 STRUCTURE   folded gguf has the same tensor list / dims / types / offsets as base.
  C2 CHANGED     % of dequantised elements (TRUE units) that moved, over sampled tensors.
                 A fold that changes ~0% is inert -> FAIL.
  C3 DIRECTION   cosine(observed dW, expected LoRA delta) and the projection ratio
                 <dW,delta>/<delta,delta>. Proves the applied change IS the LoRA, at the
                 right scale and sign — catches wrong --scale, transposed A/B, wrong prefix.
                 (needs --lora)
  C4 COLLATERAL  nvfp4 tensors the LoRA does NOT target must be bit-identical. (needs --lora)
  C5 WGLOBAL     .wglobal sidecars must be unchanged (the fold must not touch the GEMM alpha).

  ROUTE A ONLY: C4 and C5 assume an IN-PLACE fold onto the frozen fp4 grid. A route-B build
  (build_folded_nvfp4.py — merge in bf16, then requantise) legitimately re-derives every block
  scale, so it reports "C5 WGLOBAL: FAIL (~1342/1344 changed)" and trips C4 by design. On a
  route-B variant, judge by C2 + C3 alone: projection ~0.99 is a correct fold, ~0.36 is a
  frozen-grid fold, ~0.0 is inert. Verified 2026-07-29 — every shipped route-B variant fails
  C5 identically, including known-good ones, so C5 on a route-B file is noise, not a defect.

USAGE
  python3 tools/verify_fold.py \
      --base   models/ltx2/nvfp4-CLEAN.gguf \
      --folded models/ltx2/nvfp4-dev050-faceid-sheet.gguf \
      --lora   /mnt/ssd/models/ltx-best-face-id/Best_FaceID_CharacterSheet_v1.0_LoRA.safetensors \
      --scale  1.0
Exit code 0 = PASS, 1 = FAIL. Sampling is deterministic (evenly spaced), so runs are comparable.
"""
import sys, os, json, struct, argparse
import numpy as np

GT_F32 = 0
GT_NVFP4 = 40

# ---------------- nvfp4 primitives (must match import_ltx_nvfp4.py / fold_generic_lora.py) ---
def st_open(p):
    f = open(p, 'rb'); n = struct.unpack('<Q', f.read(8))[0]; h = json.loads(f.read(n)); return f, 8 + n, h

def st_raw(f, base, h, key):
    m = h[key]; a, b = m['data_offsets']; f.seek(base + a)
    return np.frombuffer(f.read(b - a), dtype=np.uint8), m['dtype'], m['shape']

def bf16_to_f32(u8):
    return (u8.view(np.uint16).astype(np.uint32) << 16).view(np.float32)

def e4m3_dec_byte(u8):
    u = u8.astype(np.int32); e = (u >> 3) & 0xF; m = u & 0x7
    v = np.where(e == 0, np.ldexp(m.astype(np.float64), -9), np.ldexp(1.0 + m / 8.0, e - 7))
    return np.where((u & 0x7F) == 0, 0.0, v)

LUTm = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

def e2m1_decode(nib):
    return np.where((nib & 8) != 0, -1.0, 1.0) * LUTm[nib & 7]

def dequant_block(raw, out, inn, nblk):
    """Block-domain dequant (NOT true units — multiply by wglobal for that)."""
    dv = e4m3_dec_byte(raw[:, :, 0:4]).astype(np.float32)
    qs = raw[:, :, 4:36]
    W = np.zeros((out, inn), np.float32); idx = np.arange(nblk) * 64
    for s in range(4):
        q8 = qs[:, :, s * 8:s * 8 + 8]
        W[:, np.add.outer(idx, s * 16 + np.arange(8)).ravel()] = \
            (e2m1_decode(q8 & 0x0F).astype(np.float32) * dv[:, :, s, None]).reshape(out, -1)
        W[:, np.add.outer(idx, s * 16 + 8 + np.arange(8)).ravel()] = \
            (e2m1_decode(q8 >> 4).astype(np.float32) * dv[:, :, s, None]).reshape(out, -1)
    return W

def parse_gguf(path):
    f = open(path, 'rb')
    assert f.read(4) == b'GGUF', f"{path}: not a gguf"
    struct.unpack('<I', f.read(4))[0]
    nt = struct.unpack('<Q', f.read(8))[0]; nkv = struct.unpack('<Q', f.read(8))[0]
    assert nkv == 0, f"{path}: expected nkv=0, got {nkv}"
    infos = []
    for _ in range(nt):
        kl = struct.unpack('<Q', f.read(8))[0]; name = f.read(kl).decode()
        nd = struct.unpack('<I', f.read(4))[0]
        dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(nd)]
        tt = struct.unpack('<I', f.read(4))[0]; off = struct.unpack('<Q', f.read(8))[0]
        infos.append((name, dims, tt, off))
    cur = f.tell(); data_start = cur + ((-cur) % 32)
    f.close()
    return infos, data_start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--folded', required=True)
    ap.add_argument('--lora', help='the safetensors that was folded (enables C3/C4)')
    ap.add_argument('--scale', type=float, default=1.0, help='the --scale the fold used')
    ap.add_argument('--prefix', default='diffusion_model.')
    ap.add_argument('--tensors', type=int, default=12, help='how many nvfp4 tensors to sample')
    ap.add_argument('--min-changed', type=float, default=0.5,
                    help='FAIL if mean %% of dequantised elements changed is below this')
    ap.add_argument('--min-projection', type=float, default=0.15,
                    help='FAIL if <dW,delta>/<delta,delta> is below this (needs --lora). '
                         'CALIBRATION: a CORRECT rank-128 fold onto this nvfp4 grid measures '
                         '~0.36 mean, because round-to-nearest discards every sub-half-step '
                         'delta. ~0.0 means inert/wrong; 1.0 is unreachable with RTN.')
    args = ap.parse_args()

    fails = []

    # ---------------- C1 STRUCTURE ----------------
    bi, bstart = parse_gguf(args.base)
    fi, fstart = parse_gguf(args.folded)
    print(f"base   : {args.base}\n         {len(bi)} tensors, data_start={bstart}, {os.path.getsize(args.base)} B")
    print(f"folded : {args.folded}\n         {len(fi)} tensors, data_start={fstart}, {os.path.getsize(args.folded)} B")
    if bi != fi or bstart != fstart:
        fails.append("C1 STRUCTURE: tensor table or data_start differs between base and folded")
        print("  C1 STRUCTURE: FAIL (tensor table differs)")
    else:
        print("  C1 STRUCTURE: PASS (identical tensor table)")

    bmap = {n: (d, t, o) for (n, d, t, o) in bi}
    nvfp4 = [(n, d, o) for (n, d, t, o) in bi if t == GT_NVFP4]

    # wglobal sidecars
    wg_off = {n[:-len('.wglobal')]: o for (n, d, t, o) in bi if t == GT_F32 and n.endswith('.wglobal')}
    fb = open(args.base, 'rb'); ff = open(args.folded, 'rb')

    def rd_wg(fh, start, stem):
        fh.seek(start + wg_off[stem]); return struct.unpack('<f', fh.read(4))[0]

    wglobals = {}
    if wg_off:
        for stem in wg_off:
            wglobals[stem] = rd_wg(fb, bstart, stem)
        print(f"  convention: UNFOLDED, {len(wg_off)} .wglobal sidecars -> true = block_dq * wglobal")
    else:
        print("  convention: FOLDED (legacy), no .wglobal -> true = block_dq")

    # ---------------- C5 WGLOBAL ----------------
    if wg_off:
        nwg_diff = sum(1 for stem in wg_off if rd_wg(fb, bstart, stem) != rd_wg(ff, fstart, stem))
        if nwg_diff:
            fails.append(f"C5 WGLOBAL: {nwg_diff}/{len(wg_off)} .wglobal sidecars changed")
            print(f"  C5 WGLOBAL: FAIL ({nwg_diff}/{len(wg_off)} changed)")
        else:
            print(f"  C5 WGLOBAL: PASS (0/{len(wg_off)} changed)")

    # ---------------- which tensors the LoRA targets ----------------
    hl = None
    targets, nontargets = [], []
    if args.lora:
        fl, bl, hl = st_open(args.lora)
        for (name, dims, off) in nvfp4:
            stem = args.prefix + name[:-len('.weight')]
            if stem + '.lora_A.weight' in hl and stem + '.lora_B.weight' in hl:
                targets.append((name, dims, off))
            else:
                nontargets.append((name, dims, off))
        print(f"  lora targets: {len(targets)} nvfp4 tensors; non-targets: {len(nontargets)}")
        alphas = [k for k in hl if k.endswith('.alpha')]
        print(f"  lora '.alpha' tensors: {len(alphas)} "
              f"({'engine will use alpha/rank' if alphas else 'none -> engine scale = multiplier'})")
    else:
        targets = nvfp4

    if not targets:
        print("no nvfp4 fold targets found — check --prefix/--lora")
        sys.exit(1)

    # deterministic even sampling across depth
    idx = np.linspace(0, len(targets) - 1, min(args.tensors, len(targets))).astype(int)
    sample = [targets[i] for i in sorted(set(idx.tolist()))]

    # ---------------- C2 CHANGED + C3 DIRECTION ----------------
    print(f"\n--- C2 CHANGED / C3 DIRECTION over {len(sample)} sampled tensors "
          f"(TRUE units = block_dq * wglobal) ---")
    hdr = f"{'tensor':<52}{'%chg':>8}{'mean|dW|':>12}{'max|dW|':>12}"
    if args.lora:
        hdr += f"{'cos':>8}{'proj':>8}"
    print(hdr)

    pct_list, proj_list, cos_list = [], [], []
    for (name, dims, off) in sample:
        inn, out = dims[0], dims[1]; nblk = inn // 64; nbytes = out * nblk * 36
        wg = wglobals.get(name, 1.0)
        fb.seek(bstart + off); rb = np.frombuffer(fb.read(nbytes), dtype=np.uint8).reshape(out, nblk, 36)
        ff.seek(fstart + off); rf = np.frombuffer(ff.read(nbytes), dtype=np.uint8).reshape(out, nblk, 36)
        Wb = dequant_block(rb, out, inn, nblk) * wg          # TRUE units
        Wf = dequant_block(rf, out, inn, nblk) * wg          # TRUE units
        dW = (Wf - Wb).astype(np.float64)
        pct = float((dW != 0).mean()) * 100.0
        pct_list.append(pct)
        line = f"{name[:51]:<52}{pct:>7.3f}%{np.abs(dW).mean():>12.3e}{np.abs(dW).max():>12.3e}"

        if args.lora:
            stem = args.prefix + name[:-len('.weight')]
            ka, kb = stem + '.lora_A.weight', stem + '.lora_B.weight'
            r = hl[ka]['shape'][0]
            A = bf16_to_f32(st_raw(fl, bl, hl, ka)[0].view(np.uint16)).reshape(r, inn)
            B = bf16_to_f32(st_raw(fl, bl, hl, kb)[0].view(np.uint16)).reshape(out, r)
            delta = (args.scale * (B @ A)).astype(np.float64)     # TRUE units
            dd = float((delta * delta).sum())
            proj = float((dW * delta).sum()) / dd if dd > 0 else 0.0
            nd = np.linalg.norm(dW); nl = np.linalg.norm(delta)
            cos = float((dW * delta).sum() / (nd * nl)) if nd > 0 and nl > 0 else 0.0
            proj_list.append(proj); cos_list.append(cos)
            line += f"{cos:>8.3f}{proj:>8.3f}"
        print(line)

    mean_pct = float(np.mean(pct_list))
    print(f"\n  C2 CHANGED: mean {mean_pct:.4f}%  median {float(np.median(pct_list)):.4f}%  "
          f"min {min(pct_list):.4f}%  max {max(pct_list):.4f}%")
    if mean_pct < args.min_changed:
        fails.append(f"C2 CHANGED: mean {mean_pct:.4f}% < {args.min_changed}% — THE FOLD IS INERT")
        print(f"  C2 CHANGED: FAIL — this fold did not move the weights. DO NOT DEPLOY.")
    else:
        print(f"  C2 CHANGED: PASS (>= {args.min_changed}%)")

    if args.lora:
        mp = float(np.mean(proj_list)); mc = float(np.mean(cos_list))
        print(f"  C3 DIRECTION: mean projection {mp:.4f} (1.0 = delta captured exactly), "
              f"mean cosine {mc:.4f}")
        print(f"      projection = the fraction of the LoRA delta that survives fp4 rounding.")
        print(f"      It is NOT expected to reach 1.0: round-to-nearest maps any element whose")
        print(f"      delta is under half a quantisation step back onto the SAME code, so the")
        print(f"      fold is a biased, per-layer-uneven attenuation of the LoRA. Measured ~0.36")
        print(f"      mean for a correct rank-128 fold here (per-tensor 0.002 .. 0.83). Treat a")
        print(f"      folded variant as 'the LoRA at a reduced and layer-dependent strength',")
        print(f"      never as equivalent to runtime LoRA. ~0.0 = inert or wrong scale/orientation.")
        if mp < args.min_projection:
            fails.append(f"C3 DIRECTION: projection {mp:.4f} < {args.min_projection} — "
                         f"the change does not match the LoRA (wrong scale/prefix/orientation?)")
            print(f"  C3 DIRECTION: FAIL")
        else:
            print(f"  C3 DIRECTION: PASS")

        # ---------------- C4 COLLATERAL ----------------
        if nontargets:
            nidx = np.linspace(0, len(nontargets) - 1, min(6, len(nontargets))).astype(int)
            bad = 0
            for i in sorted(set(nidx.tolist())):
                name, dims, off = nontargets[i]
                inn, out = dims[0], dims[1]; nbytes = out * (inn // 64) * 36
                fb.seek(bstart + off); a = fb.read(nbytes)
                ff.seek(fstart + off); b = ff.read(nbytes)
                if a != b:
                    bad += 1
            if bad:
                fails.append(f"C4 COLLATERAL: {bad} non-target nvfp4 tensors were modified")
                print(f"  C4 COLLATERAL: FAIL ({bad} untargeted tensors changed)")
            else:
                print(f"  C4 COLLATERAL: PASS (sampled untargeted tensors bit-identical)")

    fb.close(); ff.close()
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
