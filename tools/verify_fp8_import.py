#!/usr/bin/env python3
"""VERIFY an fp8 (GGML_TYPE_F8_E4M3) LTX gguf against the bf16 checkpoint it came from,
and against the nvfp4 build of the same weights.

Checks:
  F1 STRUCTURE  every fp8 tensor has a matching `.wglobal` sibling, and the scales are sane.
  F2 ACCURACY   relative error of dequant(e4m3) * wglobal vs the bf16 source.
                This is the number that justifies fp8 existing at all: it should be several
                times SMALLER than the nvfp4 build's error on the same tensors (~9.4e-2).
  F3 SANITY     no NaN / Inf in the dequantised weights (the e4m3 0x7F NaN-code trap).

Usage:
  verify_fp8_import.py --gguf <fp8.gguf> --bf16 <source.safetensors> [--nvfp4 <nvfp4.gguf>]
"""
import sys, json, struct, argparse
import numpy as np

GT_F32, GT_F8_E4M3, GT_NVFP4 = 0, 42, 40
BASE_PFX = 'model.diffusion_model.'

sys.path.insert(0, '/home/dbrain/dev/longcat-avatar.cpp/tools')
from verify_fold import st_open, st_raw, bf16_to_f32, dequant_block, parse_gguf


def e4m3_dec(u8):
    u = u8.astype(np.int32); s = (u >> 7) & 1; e = (u >> 3) & 0xF; m = u & 0x7
    v = np.where(e == 0, np.ldexp(m.astype(np.float64), -9), np.ldexp(1.0 + m / 8.0, e - 7))
    v = np.where((u & 0x7F) == 0x7F, np.nan, v)      # e4m3fn NaN code — must NOT appear
    return np.where(s == 1, -v, v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gguf', required=True)
    ap.add_argument('--bf16', required=True)
    ap.add_argument('--nvfp4')
    ap.add_argument('--tensors', type=int, default=8)
    # e4m3 carries 3 mantissa bits -> relative step 1/16, so a correct import lands at
    # ~1/16/sqrt(3) = 3.6% RMS. 5% leaves headroom without accepting a broken scale
    # (a wrong/absent wglobal shows up as an error of order 1.0, not 0.04).
    ap.add_argument('--max-relerr', type=float, default=0.05)
    a = ap.parse_args()
    fails = []

    infos, dstart = parse_gguf(a.gguf)
    fp8 = [(n, d, o) for (n, d, t, o) in infos if t == GT_F8_E4M3]
    wg_off = {n[:-len('.wglobal')]: o for (n, d, t, o) in infos
              if t == GT_F32 and n.endswith('.wglobal')}
    print(f"gguf: {len(infos)} tensors, {len(fp8)} fp8, {len(wg_off)} wglobal")

    nowg = [n for (n, d, o) in fp8 if n not in wg_off]
    if nowg:
        fails.append(f"F1 STRUCTURE: {len(nowg)} fp8 tensors have no .wglobal (e.g. {nowg[0]})")
        print(f"  F1 STRUCTURE: FAIL ({len(nowg)} missing .wglobal)")
    else:
        print(f"  F1 STRUCTURE: PASS (every fp8 tensor has a .wglobal)")

    fb, bb, hb = st_open(a.bf16)
    fp8 = [(n, d, o) for (n, d, o) in fp8 if BASE_PFX + n in hb]
    idx = np.linspace(0, len(fp8) - 1, min(a.tensors, len(fp8))).astype(int)
    sample = [fp8[i] for i in sorted(set(idx.tolist()))]
    fg = open(a.gguf, 'rb')

    nv_map, nvstart, fnv = None, None, None
    if a.nvfp4:
        ninfos, nvstart = parse_gguf(a.nvfp4)
        nv_map = {n: (d, t, o) for (n, d, t, o) in ninfos}
        fnv = open(a.nvfp4, 'rb')

    print(f"\n{'tensor':<44}{'fp8 err':>11}{'nvfp4 err':>11}{'gain':>7}{'nan':>6}")
    e8, e4, nnan = [], [], 0
    for (name, dims, off) in sample:
        ne = int(np.prod(dims))
        fg.seek(dstart + wg_off[name]); wg = struct.unpack('<f', fg.read(4))[0]
        fg.seek(dstart + off)
        W8 = e4m3_dec(np.frombuffer(fg.read(ne), dtype=np.uint8)) * wg

        u, dt, shp = st_raw(fb, bb, hb, BASE_PFX + name)
        W = bf16_to_f32(u).reshape(shp).astype(np.float64).ravel()
        nb = int(np.isnan(W8).sum()) + int(np.isinf(W8).sum())
        nnan += nb
        den = np.sqrt((W * W).mean())
        r8 = float(np.sqrt((np.nan_to_num(W8 - W) ** 2).mean()) / den)
        e8.append(r8)

        r4 = float('nan')
        if nv_map and name in nv_map and nv_map[name][1] == GT_NVFP4:
            nd, nt_, no = nv_map[name]
            inn, out = nd[0], nd[1]; nblk = inn // 64
            fnv.seek(nvstart + nv_map[name + '.wglobal'][2])
            wg4 = struct.unpack('<f', fnv.read(4))[0]
            fnv.seek(nvstart + no)
            raw = np.frombuffer(fnv.read(out * nblk * 36), dtype=np.uint8).reshape(out, nblk, 36)
            W4 = (dequant_block(raw, out, inn, nblk) * wg4).astype(np.float64).ravel()
            r4 = float(np.sqrt(((W4 - W) ** 2).mean()) / den)
            e4.append(r4)
        g = (r4 / r8) if r8 and r4 == r4 else float('nan')
        print(f"{name[:43]:<44}{r8:>11.4e}{r4:>11.4e}{g:>7.2f}{nb:>6}")

    m8 = float(np.mean(e8))
    print(f"\n  F2 ACCURACY: fp8 mean relative error {m8:.4e}")
    if e4:
        m4 = float(np.mean(e4))
        print(f"               nvfp4 same tensors    {m4:.4e}   -> fp8 is {m4/m8:.2f}x more accurate")
    if m8 > a.max_relerr:
        fails.append(f"F2 ACCURACY: {m8:.4e} > {a.max_relerr}")
        print(f"  F2 ACCURACY: FAIL")
    else:
        print(f"  F2 ACCURACY: PASS (<= {a.max_relerr})")
    if nnan:
        fails.append(f"F3 SANITY: {nnan} NaN/Inf values in dequantised fp8 weights")
        print(f"  F3 SANITY: FAIL ({nnan} NaN/Inf)")
    else:
        print(f"  F3 SANITY: PASS (no NaN/Inf)")

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
