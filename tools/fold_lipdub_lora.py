#!/usr/bin/env python3
"""Fold the LTX-2.3 lipdub IC-LoRA into the nvfp4-CLEAN gguf, producing a new
self-contained fp4 gguf so the runtime never needs the LoRA (saves the resident
base + LoRA VRAM).

CLEAN's proven on-disk convention (reverse-engineered, byte-exact on many tensors):
  - block_nvfp4: per 64-elem block = 4 UE4M3 scale bytes (one per 16-elem sub-block)
    + 32 packed E2M1 nibble bytes (build_nvfp4 / pack_nvfp4 layout).
  - the per-16 e4m3 scale d = e4m3_enc_pos(official_block_scale * weight_scale_2),
    i.e. the ModelOpt per-tensor global is FOLDED into the block scale (element scale
    = amax_block/6, fully self-contained). There are NO ".wglobal" tensors, and the
    CUDA runtime nvfp4_weight_global_for() therefore returns 1.0 -> the block scale
    already carries the whole magnitude. (Confirmed vs ggml-cuda/nvfp4-cublaslt.cu:363.)

LoRA convention (src/model/adapter/lora.hpp get_lora_weight_diff + ggml_ext_merge_lora):
  delta[out,in] = multiplier * (alpha/rank or 1) * (B @ A);  no alpha tensor + <lora:..:1.0>
  => scale = 1.0, delta = B @ A  (B=lora_B[out,r], A=lora_A[r,in]).  Verified orientation.
Runtime apply (patch_weight): dequant nvfp4->f32, add f32 diff, run linear in f32.

Fold (minimal-deviation; base = dq(CLEAN), NOT bf16):
  For each nvfp4 Linear (== every LoRA target; 1344 of them, a bijection):
    W_merged = dequant(CLEAN_block) + B@A
    re-quantize W_merged to the SAME folded block_nvfp4 layout (per-16 e4m3 scale =
    e4m3_enc_pos(block_amax/6), codes = e2m1_nearest(W/eff)); overwrite in place.
  Every non-nvfp4 tensor (406 bf16 + 2694 f32) and the entire gguf header/framing are
  copied VERBATIM from CLEAN (output starts as a byte copy of CLEAN).

  WHY dq(CLEAN) and not bf16: this golden does NOT keep official e2m1 codes unfolded; it
  is itself a fresh fp4 quantization (~13% weight-level error vs true bf16, inherent to a
  22B nvfp4). Re-quantizing from bf16 produces a DIFFERENT 13% error realization, so the
  unchanged part of every weight would diverge from CLEAN by ~13% at GEMM level (measured)
  -- larger than the LoRA effect itself. Re-quantizing dq(CLEAN)+delta keeps codes byte-exact
  to CLEAN where delta=0 (dq(CLEAN) already sits on its own fp4 grid), so the folded model
  deviates from the runtime "CLEAN + f32 LoRA" reference ONLY by the delta's fp4 grid error
  (~5.5% GEMM-level on the large-delta layers, == CLEAN's own fp4 floor). The fold is LOSSY:
  the rank-128 delta is partially below fp4 granularity on small-delta layers; eye-test gates.
"""
import sys, os, json, struct, shutil
import numpy as np

CLEAN = 'models/ltx2/nvfp4-CLEAN.gguf'
BF16  = '/mnt/hdd/ltx2-official/ltx-2.3-22b-distilled-1.1.safetensors'
LORA  = 'models/ltx2/loras/ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors'
OFF   = '/mnt/ssd/models/ltx-fp4/ltx-2.3-22b-distilled-1.1-nvfp4.safetensors'   # for validation only
OUT   = 'models/ltx2/nvfp4-CLEAN-lipdub.gguf'
PFX   = 'model.diffusion_model.'
GT_NVFP4 = 40

# ---------- primitives ----------
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

# ---------- parse CLEAN header ----------
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
    infos, data_start = parse_gguf(CLEAN)
    nvfp4 = [(n, d, o) for (n, d, t, o) in infos if t == GT_NVFP4]
    print(f"CLEAN: {len(infos)} tensors, {len(nvfp4)} nvfp4, data_start={data_start}")

    fl, bl, hl = st_open(LORA)

    # map each nvfp4 gguf tensor -> its lora A/B keys
    targets = []
    for (name, dims, off) in nvfp4:
        base = 'diffusion_model.' + name[:-len('.weight')]  # strip '.weight'
        ka = base + '.lora_A.weight'; kb = base + '.lora_B.weight'
        assert ka in hl and kb in hl, f"missing lora for {name}"
        targets.append((name, dims, off, ka, kb))
    assert len(targets) == len(nvfp4)
    print(f"mapped {len(targets)} nvfp4 tensors to lora A/B (bijection OK)")

    # copy CLEAN -> OUT (header + all non-nvfp4 tensors verbatim; nvfp4 overwritten below)
    print(f"copying {CLEAN} -> {OUT} ...")
    shutil.copyfile(CLEAN, OUT)

    fclean = open(CLEAN, 'rb')
    o = open(OUT, 'r+b')
    done = 0
    for (name, dims, off, ka, kb) in targets:
        inn, out = dims[0], dims[1]; nblk = inn // 64; nbytes = out * nblk * 36
        fclean.seek(data_start + off)
        raw = np.frombuffer(fclean.read(nbytes), dtype=np.uint8).reshape(out, nblk, 36)
        Wcl = dequant_block(raw, out, inn, nblk)                              # base = dq(CLEAN)
        A = bf16_to_f32(st_raw(fl, bl, hl, ka)[0].view(np.uint16)).reshape(128, inn)
        B = bf16_to_f32(st_raw(fl, bl, hl, kb)[0].view(np.uint16)).reshape(out, 128)
        delta = (B @ A).astype(np.float32)                                   # [out,in], scale=1.0
        Wm = (Wcl + delta).astype(np.float64)
        nib, byte = quant_folded(Wm, out, inn)
        data = pack(nib, byte, out, inn)
        assert len(data) == nbytes, f"{name}: {len(data)} vs {nbytes}"
        o.seek(data_start + off); o.write(data)
        done += 1
        if done % 200 == 0:
            print(f"  ...{done}/{len(targets)}")
    o.close(); fclean.close()
    print(f"wrote {OUT} ({os.path.getsize(OUT)} bytes)")

if __name__ == '__main__':
    main()
