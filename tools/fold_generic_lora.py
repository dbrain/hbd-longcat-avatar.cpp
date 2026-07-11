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
    for (name, dims, off, ka, kb) in targets:
        inn, out = dims[0], dims[1]; nblk = inn // 64; nbytes = out * nblk * 36
        r = hl[ka]['shape'][0]                                                # dynamic rank
        fbase.seek(data_start + off)
        raw = np.frombuffer(fbase.read(nbytes), dtype=np.uint8).reshape(out, nblk, 36)
        Wcl = dequant_block(raw, out, inn, nblk)                             # base = dq(base)
        A = bf16_to_f32(st_raw(fl, bl, hl, ka)[0].view(np.uint16)).reshape(r, inn)
        B = bf16_to_f32(st_raw(fl, bl, hl, kb)[0].view(np.uint16)).reshape(out, r)
        delta = (args.scale * (B @ A)).astype(np.float32)                    # [out,in]
        Wm = (Wcl + delta).astype(np.float64)
        nib, byte = quant_folded(Wm, out, inn)
        data = pack(nib, byte, out, inn)
        assert len(data) == nbytes, f"{name}: {len(data)} vs {nbytes}"
        o.seek(data_start + off); o.write(data)
        done += 1
        if done % 200 == 0:
            print(f"  ...{done}/{len(targets)}")
    o.close(); fbase.close()
    print(f"wrote {args.out} ({os.path.getsize(args.out)} bytes); folded {done} nvfp4 tensors @ scale {args.scale}")

if __name__ == '__main__':
    main()
