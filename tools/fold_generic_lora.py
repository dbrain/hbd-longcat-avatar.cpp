#!/usr/bin/env python3
"""Fold an arbitrary LTX-2.3 LoRA into an nvfp4 gguf, producing a self-contained
fp4 gguf so the runtime never needs the LoRA (saves the resident base + LoRA VRAM).

Generalized from tools/fold_lipdub_lora.py — same byte-exact nvfp4 block primitives,
but with a dynamic LoRA rank, a --scale multiplier, and tolerant coverage:
  * folds every nvfp4 Linear (gguf type 40) that has a matching lora_A/lora_B pair,
  * leaves any nvfp4 tensor the LoRA does NOT touch as a byte-exact dq(base) copy,
  * LOGS (does not silently drop) any LoRA target that is NOT an nvfp4 tensor in the
    gguf (e.g. bf16 MoE to_gate_logits) — those deltas are not folded by this tool.

LoRA key convention (matches the engine's native convention, verified on lipdub +
fal audio-reactive): keys are `<prefix><layer>.lora_A.weight` / `.lora_B.weight`,
delta[out,in] = scale * (B @ A), B=lora_B[out,r], A=lora_A[r,in].

=============================== .wglobal / UNITS ===============================
CRITICAL, and the cause of a whole family of silently-inert folds (2026-07-16..27).

There are TWO on-disk nvfp4 conventions and they need DIFFERENT arithmetic:

  FOLDED (legacy, 4444 tensors, no `.wglobal`):
      true_weight == dequant_block(...)
      The ModelOpt per-tensor global (weight_scale_2) is baked into the per-16 e4m3
      block scale. dq(base) is already in TRUE weight units, so a LoRA delta — which
      is always in true units — is added directly. fold_lipdub_lora.py assumed this
      and was CORRECT for its inputs.

  UNFOLDED (current, 5788 tensors, one `<weight>.wglobal` F32 sidecar per nvfp4
  tensor — see import_ltx_nvfp4.py):
      true_weight == dequant_block(...) * wglobal          (wglobal ~= 5e-5 .. 5e-4)
      The global is kept OUT of the block scale (folding it underflows ~85% of blocks
      into e4m3 subnormals = the patchy-colour bug) and is applied at the GEMM as
      `out = ggml_mul(out, wglobal)` (ggml_extend.hpp:4330).
      dq(base) is therefore ~2000-19000x LARGER than the true weight. Adding a
      true-units delta straight onto it attenuates the delta by wglobal, making the
      fold a NO-OP: measured 0.0003%-0.018% of one quantisation step, i.e. 0.0000%
      of elements change.

This tool now reads the `.wglobal` sidecars and converts the delta into whichever
domain the base file uses:  W_block += delta_true / wglobal   (wglobal defaults to
1.0 when absent, preserving the legacy behaviour exactly).

ALWAYS run tools/verify_fold.py on the output before deploying. Coverage counts
("480/480 folded") measure targets, not effect.
===============================================================================

See fold_lipdub_lora.py's docstring for WHY we fold onto dq(base), not bf16.
Usage:
  python3 tools/fold_generic_lora.py \
      --base models/ltx2/nvfp4-CLEAN.gguf \
      --lora models/ltx2/loras/ltx2.3-audio-reactive-v2.safetensors \
      --scale 1.5 \
      --out  models/ltx2/nvfp4-CLEAN-audioreactive.gguf
"""
import sys, os, json, struct, shutil, argparse
import numpy as np

GT_NVFP4 = 40
GT_F32 = 0

# ---------- primitives (byte-identical to fold_lipdub_lora.py) ----------
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
_pb = np.arange(128, dtype=np.uint8); _pv = e4m3_dec_byte(_pb); _order = np.argsort(_pv)
_sv = _pv[_order]; _sb = _pb[_order]
def e4m3_enc_pos(x):
    x = np.minimum(np.asarray(x, dtype=np.float64), 448.0)
    idx = np.searchsorted(_sv, x); idx = np.clip(idx, 0, len(_sv) - 1); lo = np.clip(idx - 1, 0, len(_sv) - 1)
    pick = np.where(np.abs(_sv[idx] - x) < np.abs(_sv[lo] - x), idx, lo); return _sb[pick].astype(np.uint8)
LUTm = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E2M1_MAX = 6.0; E4M3_MAX = 448.0; E4M3_MIN_SUB = 2.0 ** -9
def e2m1_decode(nib):
    return np.where((nib & 8) != 0, -1.0, 1.0) * LUTm[nib & 7]
def e2m1_nearest(t):
    s = (t < 0).astype(np.uint8); m = np.abs(t)
    idx = np.searchsorted(LUTm, m); idx = np.clip(idx, 0, 7); lo = np.clip(idx - 1, 0, 7)
    pick = np.where(np.abs(LUTm[idx] - m) < np.abs(LUTm[lo] - m), idx, lo).astype(np.uint8)
    return (s << 3) | pick

def dequant_block(raw, out, inn, nblk):
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

def quant_folded(W, out, inn):
    nb16 = inn // 16
    b16 = W.reshape(out, nb16, 16)
    amax = np.abs(b16).max(axis=2)
    scale = np.clip(amax / E2M1_MAX, E4M3_MIN_SUB, E4M3_MAX)
    byte = e4m3_enc_pos(scale)
    eff = np.repeat(e4m3_dec_byte(byte), 16, axis=1)
    t = np.where(eff > 0.0, W / eff, 0.0)
    nib = e2m1_nearest(t)
    return nib, byte

def pack(nib, byte, out, inn):
    nblk = inn // 64
    oc = nib.reshape(out, nblk, 64); qs = np.empty((out, nblk, 32), dtype=np.uint8)
    for s in range(4):
        sub = oc[:, :, s * 16:s * 16 + 16]
        qs[:, :, s * 8:s * 8 + 8] = sub[:, :, 0:8] | (sub[:, :, 8:16] << 4)
    db = byte.reshape(out, nblk, 4).astype(np.uint8)
    return np.concatenate([db, qs], axis=2).tobytes()

def parse_gguf(path):
    f = open(path, 'rb'); assert f.read(4) == b'GGUF'; ver = struct.unpack('<I', f.read(4))[0]
    nt = struct.unpack('<Q', f.read(8))[0]; nkv = struct.unpack('<Q', f.read(8))[0]
    assert nkv == 0, f"expected nkv=0, got {nkv}"
    infos = []
    for _ in range(nt):
        kl = struct.unpack('<Q', f.read(8))[0]; name = f.read(kl).decode(); nd = struct.unpack('<I', f.read(4))[0]
        dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(nd)]
        tt = struct.unpack('<I', f.read(4))[0]; off = struct.unpack('<Q', f.read(8))[0]
        infos.append((name, dims, tt, off))
    cur = f.tell(); data_start = cur + ((-cur) % 32)
    f.close()
    return infos, data_start

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True, help='nvfp4 gguf to fold into (e.g. nvfp4-CLEAN.gguf)')
    ap.add_argument('--lora', required=True, help='safetensors LoRA (lora_A/lora_B keys)')
    ap.add_argument('--out',  required=True, help='output gguf path')
    ap.add_argument('--scale', type=float, required=True, help='LoRA strength multiplier (e.g. 1.5)')
    ap.add_argument('--prefix', default='diffusion_model.',
                    help="lora key prefix prepended to the gguf tensor stem (default 'diffusion_model.')")
    ap.add_argument('--dry-run', action='store_true', help='report coverage and exit without writing')
    args = ap.parse_args()

    infos, data_start = parse_gguf(args.base)
    nvfp4 = [(n, d, o) for (n, d, t, o) in infos if t == GT_NVFP4]
    print(f"base: {args.base}\n  {len(infos)} tensors, {len(nvfp4)} nvfp4, data_start={data_start}")

    # ---- .wglobal sidecars: decide which domain dq(base) is in (see docstring) ----
    wg_off = {n[:-len('.wglobal')]: o for (n, d, t, o) in infos
              if t == GT_F32 and n.endswith('.wglobal')}
    wglobals = {}
    if wg_off:
        with open(args.base, 'rb') as fw:
            for stem, o in wg_off.items():
                fw.seek(data_start + o)
                wglobals[stem] = struct.unpack('<f', fw.read(4))[0]
    if wglobals:
        vals = np.array(list(wglobals.values()), dtype=np.float64)
        print(f"  convention: UNFOLDED — {len(wglobals)} .wglobal sidecars "
              f"(min={vals.min():.3e} max={vals.max():.3e}); delta will be divided by wglobal")
        bad = [k for k, v in wglobals.items() if not np.isfinite(v) or v == 0.0]
        if bad:
            sys.exit(f"FATAL: {len(bad)} .wglobal are zero/non-finite, e.g. {bad[:3]}")
    else:
        print("  convention: FOLDED (legacy) — no .wglobal; dq(base) is already in true units")

    fl, bl, hl = st_open(args.lora)
    lora_mods = set()
    for k in hl:
        if k.endswith('.lora_A.weight'):
            lora_mods.add(k[:-len('.lora_A.weight')])

    # map each nvfp4 gguf tensor -> its lora A/B keys (tolerant: skip missing)
    targets, uncovered = [], []
    for (name, dims, off) in nvfp4:
        stem = args.prefix + name[:-len('.weight')]
        ka = stem + '.lora_A.weight'; kb = stem + '.lora_B.weight'
        if ka in hl and kb in hl:
            targets.append((name, dims, off, ka, kb))
        else:
            uncovered.append(name)
    covered_stems = set(args.prefix + n[:-len('.weight')] for (n, _, _, _, _) in targets)
    lora_only = sorted(lora_mods - covered_stems)

    print(f"  fold targets (nvfp4 ∩ lora): {len(targets)}/{len(nvfp4)}")
    print(f"  nvfp4 NOT in lora (left as dq(base) copy): {len(uncovered)}")
    print(f"  lora targets NOT nvfp4 (deltas NOT folded by this tool): {len(lora_only)}")
    if lora_only:
        from collections import Counter
        c = Counter(s.split('.')[-1] for s in lora_only)
        print("    dropped-delta module kinds: " + ", ".join(f"{k}×{v}" for k, v in c.most_common()))
    ranks = set()
    for (_, _, _, ka, _) in targets:
        ranks.add(hl[ka]['shape'][0])
    print(f"  lora rank(s) over targets: {sorted(ranks)}   scale={args.scale}")

    if args.dry_run:
        print("dry-run: no output written"); return
    if not targets:
        sys.exit("no fold targets — check --prefix / lora naming")

    print(f"copying {args.base} -> {args.out} ...")
    shutil.copyfile(args.base, args.out)

    fbase = open(args.base, 'rb')
    o = open(args.out, 'r+b')
    done = 0
    changed_frac = []      # per-tensor % of elements whose DEQUANTISED value moved
    for (name, dims, off, ka, kb) in targets:
        inn, out = dims[0], dims[1]; nblk = inn // 64; nbytes = out * nblk * 36
        r = hl[ka]['shape'][0]                                                # dynamic rank
        fbase.seek(data_start + off)
        raw = np.frombuffer(fbase.read(nbytes), dtype=np.uint8).reshape(out, nblk, 36)
        Wcl = dequant_block(raw, out, inn, nblk)                             # base = dq(base)
        A = bf16_to_f32(st_raw(fl, bl, hl, ka)[0].view(np.uint16)).reshape(r, inn)
        B = bf16_to_f32(st_raw(fl, bl, hl, kb)[0].view(np.uint16)).reshape(out, r)
        delta = (args.scale * (B @ A)).astype(np.float32)                    # [out,in], TRUE units
        if wglobals:
            if name not in wglobals:
                sys.exit(f"FATAL: {name} is nvfp4 and the file has .wglobal sidecars, "
                         f"but this tensor has none — refusing to fold in an unknown domain")
            delta = delta / np.float32(wglobals[name])                       # -> block domain
        Wm = (Wcl + delta).astype(np.float64)
        nib, byte = quant_folded(Wm, out, inn)
        data = pack(nib, byte, out, inn)
        assert len(data) == nbytes, f"{name}: {len(data)} vs {nbytes}"
        o.seek(data_start + off); o.write(data)

        # effect check: compare DEQUANTISED values, not bytes (+0/-0 codes are redundant)
        nib0, byte0 = quant_folded(Wcl.astype(np.float64), out, inn)
        v0 = e2m1_decode(nib0) * np.repeat(e4m3_dec_byte(byte0), 16, axis=1)
        v1 = e2m1_decode(nib) * np.repeat(e4m3_dec_byte(byte), 16, axis=1)
        changed_frac.append(float((v0 != v1).mean()) * 100.0)

        done += 1
        if done % 200 == 0:
            print(f"  ...{done}/{len(targets)}  (running mean changed = {np.mean(changed_frac):.3f}%)")
    o.close(); fbase.close()

    cf = np.array(changed_frac)
    print(f"wrote {args.out} ({os.path.getsize(args.out)} bytes); folded {done} nvfp4 tensors @ scale {args.scale}")
    print(f"EFFECT: dequantised elements changed — mean {cf.mean():.4f}%  "
          f"median {np.median(cf):.4f}%  min {cf.min():.4f}%  max {cf.max():.4f}%")
    if cf.mean() < 0.01:
        print("*** FAIL: this fold is a NO-OP. The weights did not move. Do NOT deploy. ***")
        print("*** Check the .wglobal/units domain and the --scale. See verify_fold.py. ***")
        sys.exit(2)
    print("run tools/verify_fold.py on this output before deploying.")

if __name__ == '__main__':
    main()
