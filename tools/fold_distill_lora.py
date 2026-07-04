#!/usr/bin/env python3
"""Fold an LTX-2.3 LoRA into the nvfp4-CLEAN gguf at an ARBITRARY (incl. NEGATIVE)
strength, producing a new self-contained fp4 gguf.  Generalized, strength-parameterized
sibling of tools/fold_lipdub_lora.py (read that first — this reuses its exact nvfp4 block
primitives and its "fold into dq(CLEAN), requantize to the same layout, copy everything
non-nvfp4 + the whole header verbatim" approach).

WHY NEGATIVE STRENGTH ("de-distillation"):
  nvfp4-CLEAN.gguf == dev + distill-lora@1.0  (that is exactly what "distilled-1.1" IS).
  Therefore, algebraically:
      dev + distill@s  ==  distilled-1.1 + distill-lora@(s - 1.0)
  To obtain "dev + distill@0.65" (the Denoise-AI S2 base stage) we fold THIS distill LoRA
  onto nvfp4-CLEAN at strength = 0.65 - 1.0 = -0.35.  No 46 GB dev download, no re-quant of
  the whole model, same architecture / param count / nvfp4 format / kernels => same VRAM &
  per-step speed as current prod, only less distillation baked in.
      strength =  0.0  -> byte-identical to CLEAN (no change)
      strength = -0.35 -> effective distill 0.65   (S2 base)
      strength = -1.00 -> recovers ~pure dev (assumption to verify: distilled-1.1 = dev + lora@1.0)
      strength = +X    -> over-distill (also supported, delta = X * (B@A))

FOLD MATH (delta = strength * (B @ A); identical grid handling to fold_lipdub_lora.py):
  For each nvfp4 Linear in CLEAN that has a matching LoRA target (a bijection on the nvfp4
  set — see coverage report below):
      W_merged = dequant(CLEAN_block) + strength * (B @ A)
      re-quantize W_merged to the SAME folded block_nvfp4 layout (per-16 e4m3 scale =
      e4m3_enc_pos(block_amax/6), codes = e2m1_nearest(W/eff)); overwrite in place.
  Non-nvfp4 tensors and the entire gguf header/framing are copied VERBATIM (output starts as
  a byte copy of CLEAN).  See fold_lipdub_lora.py for the dq(CLEAN)-not-bf16 rationale (keeps
  codes byte-exact where delta=0; fold deviates from a runtime "CLEAN + f32 LoRA" reference
  only by the delta's fp4 grid error).

LIMITATION — non-nvfp4 LoRA targets are NOT folded:
  The distill LoRA also targets bf16/f32 layers (adaln modulation, timestep embedders, the
  audio path).  Those base tensors are copied verbatim (this script, like the lipdub
  precedent, only rewrites nvfp4 type-40 tensors).  Their share of the distillation delta is
  therefore left baked in.  They are reported as "unmatched (non-nvfp4)" so the coverage gap
  is explicit.  Fold the load-bearing attn/ffn Linears where the mush lives; the modulation
  layers are small and the eye-test gates whether their residual distillation matters.

LoRA convention (matches fold_lipdub_lora.py):  base key 'diffusion_model.<name-w/o-.weight>',
  '.lora_A.weight' = A[rank,in], '.lora_B.weight' = B[out,rank];  delta[out,in] = B @ A.
  RANK IS READ PER-TENSOR from A.shape[0] (this distill LoRA mixes rank 384/256/128/32 — do
  NOT hardcode a single rank the way the lipdub script did).
"""
import sys, os, json, struct, shutil, argparse
import numpy as np

GT_NVFP4 = 40

# Defaults for this repo (longcat-avatar.cpp checkout).
DEF_BASE = 'models/ltx2/nvfp4-CLEAN.gguf'
DEF_LORA = 'models/ltx2/loras/_hf_ltx23/ltx-2.3-22b-distilled-lora-384-1.1.safetensors'
DEF_PREFIX = 'diffusion_model.'   # LoRA-side prefix; gguf tensor name = base_key[len(prefix):] + '.weight'

# ---------- nvfp4 block primitives (copied verbatim from fold_lipdub_lora.py) ----------
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
def swz_index(out, nbk):
    R, RC, CC = 128, 32, 4; rows = np.arange(out)[:, None]; cols = np.arange(nbk)[None, :]
    rb = rows // R; rem = rows % R; d4 = rem // RC; d3 = rem % RC; cbg = cols // CC; d5 = cols % CC
    cbg_cnt = (nbk + CC - 1) // CC
    return (((rb * cbg_cnt + cbg) * RC + d3) * 16 + d4 * CC + d5)
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
    """Dequantize a CLEAN block_nvfp4 tensor (raw uint8 [out,nblk,36]) -> W[out,inn] f32,
    matching ggml dequantize_row_nvfp4 exactly (element = e2m1_code * ue4m3(d), global=1)."""
    dv = e4m3_dec_byte(raw[:, :, 0:4]).astype(np.float32)   # (out,nblk,4)
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
    """Single-level folded NVFP4 quant matching CLEAN's stored convention.
    Returns (nib[out,inn] uint8, byte[out,nb16] uint8). Element scale = e4m3(block_amax/6)."""
    nb16 = inn // 16
    b16 = W.reshape(out, nb16, 16)
    amax = np.abs(b16).max(axis=2)                       # (out,nb16)
    scale = np.clip(amax / E2M1_MAX, E4M3_MIN_SUB, E4M3_MAX)
    byte = e4m3_enc_pos(scale)                           # (out,nb16) folded e4m3 scale
    eff = np.repeat(e4m3_dec_byte(byte), 16, axis=1)     # (out,inn)
    t = np.where(eff > 0.0, W / eff, 0.0)
    nib = e2m1_nearest(t)                                # (out,inn)
    return nib, byte

def pack(nib, byte, out, inn):
    """CLEAN block layout: per 64-elem block -> 4 scale bytes + 32 data bytes."""
    nblk = inn // 64
    oc = nib.reshape(out, nblk, 64); qs = np.empty((out, nblk, 32), dtype=np.uint8)
    for s in range(4):
        sub = oc[:, :, s * 16:s * 16 + 16]
        qs[:, :, s * 8:s * 8 + 8] = sub[:, :, 0:8] | (sub[:, :, 8:16] << 4)
    db = byte.reshape(out, nblk, 4).astype(np.uint8)
    return np.concatenate([db, qs], axis=2).tobytes()

# ---------- gguf header (copied from fold_lipdub_lora.py) ----------
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

# ---------- mapping ----------
def build_mapping(infos, lora_hdr, prefix):
    """Return (targets, report).
      targets = list of (name, dims, off, ka, kb, rank) for every nvfp4 Linear that has a
                matching LoRA A/B pair (the fold set).
      report  = dict of coverage counters + lists.
    Convention: gguf tensor '<x>.weight' <-> LoRA base '<prefix><x>' with '.lora_A.weight'/'.lora_B.weight'.
    """
    lkeys = set(k for k in lora_hdr if k != '__metadata__')
    # enumerate lora targets present (base -> has A / has B)
    lora_bases = {}
    for k in lkeys:
        if k.endswith('.lora_A.weight'):
            lora_bases.setdefault(k[:-len('.lora_A.weight')], {})['A'] = True
        elif k.endswith('.lora_B.weight'):
            lora_bases.setdefault(k[:-len('.lora_B.weight')], {})['B'] = True
    targets = []
    nvfp4_no_lora = []      # nvfp4 gguf tensors with no matching lora target
    matched_nonnvfp4 = []   # lora targets that hit a non-nvfp4 gguf tensor (skipped in fold)
    lora_hit = set()
    gguf_weight_bases = set()
    for (name, dims, tt, off) in infos:
        if not name.endswith('.weight'):
            continue
        lb = prefix + name[:-len('.weight')]
        gguf_weight_bases.add(lb)
        if lb in lora_bases:
            lora_hit.add(lb)
            if tt == GT_NVFP4:
                ka = lb + '.lora_A.weight'; kb = lb + '.lora_B.weight'
                assert ka in lora_hdr and kb in lora_hdr, f"partial lora pair for {name}"
                rank = lora_hdr[ka]['shape'][0]
                targets.append((name, dims, off, ka, kb, rank))
            else:
                matched_nonnvfp4.append((name, tt))
        elif tt == GT_NVFP4:
            nvfp4_no_lora.append(name)
    # lora targets that map to no gguf tensor at all
    lora_unmatched = sorted(b for b in lora_bases if b not in gguf_weight_bases)
    n_nvfp4 = sum(1 for (_, _, tt, _) in infos if tt == GT_NVFP4)
    report = dict(
        n_gguf=len(infos), n_nvfp4=n_nvfp4, n_lora_targets=len(lora_bases),
        folded=len(targets), nvfp4_no_lora=nvfp4_no_lora,
        matched_nonnvfp4=matched_nonnvfp4, lora_unmatched=lora_unmatched,
        bijection=(len(nvfp4_no_lora) == 0 and len(targets) == n_nvfp4),
    )
    return targets, report

def print_report(report, strength, base, lora, prefix):
    print("=" * 72)
    print(f"base   : {base}")
    print(f"lora   : {lora}")
    print(f"prefix : {prefix}")
    print(f"strength requested : {strength}   (delta = strength * B@A)")
    if strength < 0:
        print(f"  -> effective distill fraction ~= {1.0 + strength:.3f}  (CLEAN==distill@1.0)")
    print("-" * 72)
    print(f"gguf tensors total            : {report['n_gguf']}")
    print(f"gguf nvfp4 (type 40) Linears  : {report['n_nvfp4']}")
    print(f"lora targets (A/B pairs)      : {report['n_lora_targets']}")
    print(f"nvfp4 Linears FOLDED          : {report['folded']}")
    print(f"nvfp4 Linears with NO lora    : {len(report['nvfp4_no_lora'])}")
    print(f"lora targets on NON-nvfp4     : {len(report['matched_nonnvfp4'])}  (bf16/f32, copied verbatim, NOT de-distilled)")
    print(f"lora targets on NO gguf tensor: {len(report['lora_unmatched'])}")
    print(f"nvfp4 fold is a clean bijection: {report['bijection']}")
    if report['nvfp4_no_lora']:
        print("  !! nvfp4 Linears missing a lora target (will be copied unchanged):")
        for n in report['nvfp4_no_lora'][:20]:
            print("       ", n)
    if report['matched_nonnvfp4']:
        print("  -- non-nvfp4 lora targets skipped (sample):")
        for n, tt in report['matched_nonnvfp4'][:12]:
            print(f"        {n}  (ggufType {tt})")
        if len(report['matched_nonnvfp4']) > 12:
            print(f"        ... +{len(report['matched_nonnvfp4']) - 12} more")
    if report['lora_unmatched']:
        print("  ?? lora targets with no gguf tensor (sample):")
        for n in report['lora_unmatched'][:12]:
            print("       ", n)
    print("=" * 72)

def main():
    ap = argparse.ArgumentParser(description="Fold an LTX LoRA into nvfp4-CLEAN gguf at arbitrary/negative strength.")
    ap.add_argument('--base', default=DEF_BASE, help='base nvfp4 gguf (default: %(default)s)')
    ap.add_argument('--lora', default=DEF_LORA, help='LoRA safetensors (default: distill-384-1.1)')
    ap.add_argument('--strength', type=float, required=True,
                    help='delta = strength * (B@A). NEGATIVE allowed: -0.35 => effective distill 0.65 (de-distill).')
    ap.add_argument('--out', help='output gguf path (required unless --dry-run)')
    ap.add_argument('--prefix', default=DEF_PREFIX, help="LoRA-side key prefix (default: %(default)r)")
    ap.add_argument('--dry-run', action='store_true',
                    help='validate name mapping + shapes + rank + coverage ONLY; no quantization, no output written.')
    args = ap.parse_args()
    if not args.dry_run and not args.out:
        ap.error("--out is required unless --dry-run")

    infos, data_start = parse_gguf(args.base)
    fl, bl, hl = st_open(args.lora)
    targets, report = build_mapping(infos, hl, args.prefix)
    print_report(report, args.strength, args.base, args.lora, args.prefix)

    # Validate every folded target's shapes/rank against the gguf dims (cheap; no data read).
    rank_hist = {}
    shape_errors = 0
    for (name, dims, off, ka, kb, rank) in targets:
        inn, out = dims[0], dims[1]
        Ashp = hl[ka]['shape']; Bshp = hl[kb]['shape']
        rank_hist[rank] = rank_hist.get(rank, 0) + 1
        ok = (list(Ashp) == [rank, inn] and list(Bshp) == [out, rank])
        if not ok:
            shape_errors += 1
            if shape_errors <= 10:
                print(f"  SHAPE MISMATCH {name}: gguf(inn,out)=({inn},{out}) A={Ashp} B={Bshp} rank={rank}")
    print(f"rank histogram over folded set: {dict(sorted(rank_hist.items()))}")
    print(f"shape validation errors: {shape_errors}")
    if shape_errors:
        print("ABORT: shape mismatches present.", file=sys.stderr); sys.exit(2)

    if args.dry_run:
        print(f"[dry-run] mapping valid. Would fold {len(targets)} nvfp4 Linears at strength={args.strength}. "
              f"No output written.")
        return

    if abs(args.strength) == 0.0:
        print("WARNING: strength==0 produces a byte-identical copy of the base.", file=sys.stderr)

    # copy base -> out (header + all non-nvfp4 tensors verbatim; nvfp4 overwritten below)
    print(f"copying {args.base} -> {args.out} ...")
    shutil.copyfile(args.base, args.out)

    fbase = open(args.base, 'rb')
    o = open(args.out, 'r+b')
    done = 0
    dmin = np.inf; dmax = 0.0; dsum = 0.0; dcnt = 0
    for (name, dims, off, ka, kb, rank) in targets:
        inn, out = dims[0], dims[1]; nblk = inn // 64; nbytes = out * nblk * 36
        fbase.seek(data_start + off)
        raw = np.frombuffer(fbase.read(nbytes), dtype=np.uint8).reshape(out, nblk, 36)
        Wcl = dequant_block(raw, out, inn, nblk)                           # base = dq(CLEAN)
        A = bf16_to_f32(st_raw(fl, bl, hl, ka)[0].view(np.uint16)).reshape(rank, inn)
        B = bf16_to_f32(st_raw(fl, bl, hl, kb)[0].view(np.uint16)).reshape(out, rank)
        delta = (args.strength * (B @ A)).astype(np.float32)               # [out,in]
        ad = np.abs(delta)
        dmin = min(dmin, float(ad.min())); dmax = max(dmax, float(ad.max()))
        dsum += float(ad.sum()); dcnt += ad.size
        Wm = (Wcl + delta).astype(np.float64)
        nib, byte = quant_folded(Wm, out, inn)
        data = pack(nib, byte, out, inn)
        assert len(data) == nbytes, f"{name}: {len(data)} vs {nbytes}"
        o.seek(data_start + off); o.write(data)
        done += 1
        if done % 200 == 0:
            print(f"  ...{done}/{len(targets)}")
    o.close(); fbase.close()
    print(f"wrote {args.out} ({os.path.getsize(args.out)} bytes)")
    print(f"folded {done} nvfp4 Linears; |delta| min={dmin:.3e} max={dmax:.3e} "
          f"mean={dsum / max(dcnt, 1):.3e}  effective_strength={args.strength}")

if __name__ == '__main__':
    main()
