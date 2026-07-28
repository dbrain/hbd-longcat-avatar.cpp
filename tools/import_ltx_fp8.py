#!/usr/bin/env python3
"""Import ComfyUI's dev-fp8 LTX-2.3 DiT safetensors -> a ggml gguf that longcat-avatar.cpp's
LTXAV loader reads, storing the heavy Linears NATIVELY as GGML_TYPE_F8_E4M3 (1 byte/elem,
standard __nv_fp8_e4m3) so they GEMM on the Blackwell FP8 tensor cores (no requantise).

Mirrors tools/import_ltx_nvfp4.py's structure:
  - SRC ggml gguf (default models/ltx2/nvfp4-CLEAN.gguf) supplies the EXACT DiT tensor
    list + dims + KV our loader expects (bare names, ggml dim order). We keep that list
    verbatim and look each name up in the fp8 safetensors as 'model.diffusion_model.<name>'.
  - Per tensor, the OUTPUT ggml type is chosen from the safetensors dtype:
        F8_E4M3 -> GGML_TYPE_F8_E4M3 (repacked verbatim: comfy e4m3fn == std __nv_fp8_e4m3)
        F32     -> F32
        BF16    -> F16
    EXCEPT the modulation/norm/bias class (keepf32) which is forced F32 regardless — those
    feed element-wise/binbcast ops that assert on non-F32 (the LTX-NVFP4 load-crash class).
    (No safetensors norm/scale tensor is fp8, so F8-first ordering is safe either way.)

ComfyUI's dev-fp8 weights are NOT raw casts: quant_ops stores fp8_bytes = weight/scale with
scale = amax(|weight|)/448 (filled to the ±448 e4m3 range), and the scale lives in a runtime
Params object — NOT in the safetensors. So we MUST recover a per-tensor A-scale
`wglobal = amax(|bf16_weight|)/448` from the ORIGINAL bf16 checkpoint and persist it as a sibling
`<name>.wglobal` F32 tensor (the exact mechanism the NVFP4 unfolded-import uses for ModelOpt
weight_scale_2). The loader registers it (stable-diffusion.cpp load_nvfp4_weight_globals +
ggml_cuda_nvfp4_register_weight_global) and the FP8 cuBLASLt GEMM applies it as the weight A-scale
(A_scale=1.0 would make weights ~448/amax x too large -> blown-white output).

Usage: import_ltx_fp8.py [<src_ggml_gguf> <fp8_safetensors> <out_gguf> <bf16_src_safetensors>]
Defaults: nvfp4-CLEAN.gguf, ltx-2.3-22b-dev-fp8.safetensors,
          /mnt/ssd/models/ltx2/ltx-2.3-22b-dev-fp8.gguf, ltx23-src/ltx-2.3-22b-dev.safetensors
"""
import sys, json, struct, numpy as np, os, collections

# F32LIN=1: map the non-fp8, non-keepf32 Linears (comfy's BF16 layers) to F32 instead of F16.
# Widens their range past F16 max (65504). Default output name gets a -f32lin suffix.
F32LIN = os.environ.get("F32LIN", "0") not in ("0", "", "false", "False")

# ATTN_F16=1: store the ATTENTION Linears (to_q/k/v/out, gate_logits — any "attn" name) as F16
# (dequant comfy e4m3 × wglobal -> f16) instead of native e4m3, keeping FFN native e4m3 for speed.
# Fixes the native-e4m3-attention NaN (e4m3 GEMM -> F16 output overflows on attn; softmax -> NaN;
# blocks 0/1 which comfy already stores F16 stay clean). Output name gets an -attnf16 suffix.
ATTN_F16 = os.environ.get("ATTN_F16", "0") not in ("0", "", "false", "False")
def is_attn(name):
    return "attn" in name and name.endswith(".weight")

# FOLD_LORA=<path>: fold a distill LoRA into the weights at convert time (dev050 recipe), so the
# render needs NO runtime LoRA (whose F8_E4M3 merge is broken: it dequants e4m3 to raw ±448 codes
# without the wglobal scale). For each lora-matched weight W (dequantized e4m3×wglobal, or bf16/f32):
# W' = W + LORA_MULT·(B@A); F8 weights are then re-quantized to e4m3 with a fresh wglobal'=amax(W')/448.
# The engine's own scale is multiplier-only (no per-tensor .alpha tensors in this LoRA), so
# LORA_MULT (default 0.5) matches runtime `<lora:...:0.5>`.
FOLD_LORA = os.environ.get("FOLD_LORA", "")
LORA_MULT = float(os.environ.get("LORA_MULT", "0.5"))

SRC = sys.argv[1] if len(sys.argv) > 1 else "/home/dbrain/dev/longcat-avatar.cpp/models/ltx2/nvfp4-CLEAN.gguf"
OFF = sys.argv[2] if len(sys.argv) > 2 else "/mnt/ssd/models/ltx23-comfy/_dl/ltx23fp8/ltx-2.3-22b-dev-fp8.safetensors"
if FOLD_LORA:
    _default_out = "/home/dbrain/dev/longcat-avatar.cpp/models/ltx2/ltx-2.3-22b-dev-fp8-dev050.gguf"
elif F32LIN:
    _default_out = "/mnt/ssd/models/ltx2/ltx-2.3-22b-dev-fp8-f32lin.gguf"
else:
    _default_out = "/mnt/ssd/models/ltx2/ltx-2.3-22b-dev-fp8.gguf"
if ATTN_F16:
    _default_out = _default_out.replace(".gguf", "-attnf16.gguf")
OUT = sys.argv[3] if len(sys.argv) > 3 else _default_out
BF  = sys.argv[4] if len(sys.argv) > 4 else "/mnt/ssd/models/ltx23-src/ltx-2.3-22b-dev.safetensors"
PFX = "model.diffusion_model."
LORA_PFX = "diffusion_model."   # the distill LoRA uses this prefix (NOT model.diffusion_model.)
FP8_E4M3_MAX = 448.0

GT_F32 = 0
GT_F16 = 1
GT_BF16 = 30
GT_F8_E4M3 = 42  # must match GGML_TYPE_F8_E4M3 in ggml/include/ggml.h

# modulation / norm / bias / scale class: MUST stay F32 (element-wise + binbcast asserts).
# Same list as tools/import_ltx_nvfp4.py's keepf32.
KEEPF32_KEYS = ('scale_shift', 'adaln', 'norm', 'embedder', '.bias', '.scale')

def is_keepf32(name):
    return any(k in name for k in KEEPF32_KEYS)

# ---- safetensors reader (raw; no safetensors/torch dep) ----
def st_open(p):
    f = open(p, 'rb'); n = struct.unpack('<Q', f.read(8))[0]; h = json.loads(f.read(n)); return f, 8 + n, h

def st_raw(f, base, h, key):
    m = h[key]; a, b = m['data_offsets']; f.seek(base + a)
    return np.frombuffer(f.read(b - a), dtype=np.uint8), m['dtype'], m['shape']

def bf16_to_f32(u8):
    return (u8.view(np.uint16).astype(np.uint32) << 16).view(np.float32).astype(np.float32)

def f32_to_f16(x):
    return np.ascontiguousarray(x, dtype=np.float32).astype(np.float16)

def st_f32(f, base, h, key):
    # read a safetensors tensor as float32 (handles BF16 / F32), returned in its declared shape.
    u, dt, sh = st_raw(f, base, h, key)
    if dt == 'BF16':   x = bf16_to_f32(u)
    elif dt == 'F32':  x = u.view(np.float32)
    elif dt == 'F16':  x = u.view(np.float16).astype(np.float32)
    else: raise Exception(f"{key}: unsupported dtype {dt}")
    return np.ascontiguousarray(x, dtype=np.float32).reshape(sh)

# ---- e4m3 (standard fn) decode/encode, matching ggml_e4m3_to_fp32 / __nv_fp8_e4m3 ----
def e4m3_dec_byte(u8):
    u = u8.astype(np.int32); s = (u >> 7) & 1; e = (u >> 3) & 0xF; m = u & 0x7
    v = np.where(e == 0, np.ldexp(m.astype(np.float64), -9), np.ldexp(1.0 + m / 8.0, e - 7))
    v = np.where((u & 0x7F) == 0x7F, 0.0, v)          # NaN -> 0
    return np.where(s == 1, -v, v)
# codes 0..126 ONLY — EXCLUDE 0x7F (e4m3fn NaN). 0x7F decodes to 0.0 here, so including it let
# searchsorted pick the NaN code for near-zero weights -> NaN bytes in the gguf -> the GEMM
# faithfully produced NaN output. 0x7E (448) is the max finite and stays in the table.
_pb = np.arange(127, dtype=np.uint8); _pv = e4m3_dec_byte(_pb)          # positive code -> value
_order = np.argsort(_pv); _sv = _pv[_order]; _sb = _pb[_order]
def e4m3_enc_signed(x):
    xa = np.minimum(np.abs(np.asarray(x, dtype=np.float64)), 448.0)
    idx = np.searchsorted(_sv, xa); idx = np.clip(idx, 0, len(_sv) - 1); lo = np.clip(idx - 1, 0, len(_sv) - 1)
    pick = np.where(np.abs(_sv[idx] - xa) < np.abs(_sv[lo] - xa), idx, lo)
    c = _sb[pick].astype(np.uint8)
    return np.where(np.asarray(x) < 0, c | 0x80, c).astype(np.uint8)

# ---- LoRA fold: ΔW = LORA_MULT · (B @ A), [out,in]; None if this weight has no LoRA pair ----
if FOLD_LORA:
    fl, bl, hl = st_open(FOLD_LORA)
    print(f"fold: LoRA {FOLD_LORA}  mult={LORA_MULT}  ({sum(1 for k in hl if k.endswith('.lora_A.weight'))} pairs)")
def lora_delta(name, sh):
    # name = gguf bare weight name incl '.weight'; sh = [out, in]. LoRA key = diffusion_model.<name-.weight>.
    if not FOLD_LORA or len(sh) != 2:
        return None
    base = LORA_PFX + name[:-len('.weight')]
    ka, kb = base + '.lora_A.weight', base + '.lora_B.weight'
    if ka not in hl or kb not in hl:
        return None
    A = st_f32(fl, bl, hl, ka)   # [rank, in]
    B = st_f32(fl, bl, hl, kb)   # [out, rank]
    dW = (B @ A).astype(np.float32)
    assert dW.shape == (sh[0], sh[1]), f"{name}: dW {dW.shape} vs {tuple(sh)}"
    return (LORA_MULT * dW).astype(np.float32)

# ---- parse SRC gguf: KV blob verbatim + tensor (name, dims) list ----
f = open(SRC, 'rb'); magic = f.read(4); ver = struct.unpack('<I', f.read(4))[0]
nt = struct.unpack('<Q', f.read(8))[0]; nkv = struct.unpack('<Q', f.read(8))[0]
kv_start = f.tell()

def rs():
    n = struct.unpack('<Q', f.read(8))[0]; return f.read(n)

def sv(t):
    if t in (0, 1, 7): f.read(1)
    elif t in (2, 3): f.read(2)
    elif t in (4, 5, 6): f.read(4)
    elif t in (10, 11, 12): f.read(8)
    elif t == 8: rs()
    elif t == 9:
        et = struct.unpack('<I', f.read(4))[0]; nn = struct.unpack('<Q', f.read(8))[0]
        [(rs() if et == 8 else sv(et)) for _ in range(nn)]

align = 32
for _ in range(nkv):
    kl = struct.unpack('<Q', f.read(8))[0]; k = f.read(kl); t = struct.unpack('<I', f.read(4))[0]
    if k == b'general.alignment': align = struct.unpack('<I', f.read(4))[0]
    else: sv(t)
kv_end = f.tell(); f.seek(kv_start); KV = f.read(kv_end - kv_start)

infos = []
for _ in range(nt):
    kl = struct.unpack('<Q', f.read(8))[0]; name = f.read(kl).decode(); nd = struct.unpack('<I', f.read(4))[0]
    dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(nd)]
    tt = struct.unpack('<I', f.read(4))[0]; off = struct.unpack('<Q', f.read(8))[0]
    infos.append([name, dims])
f.close()

fo, bo, ho = st_open(OFF)
fb, bb, hb = st_open(BF)   # original bf16 checkpoint -> per-tensor recovery scale

def bf16_amax(oname):
    # amax(|weight|) over the ORIGINAL bf16 tensor (the fp8 file has no scale). Streamed as
    # uint16 -> f32 magnitude; bf16 |x| is monotone in (u16 & 0x7fff), so we can take the max
    # exponent/mantissa bit-pattern and decode just that (avoids a full f32 expansion of 16M+ elems).
    u, dt, sh = st_raw(fb, bb, hb, oname)
    if dt == 'BF16':
        w = u.view(np.uint16)
        mag = (w & 0x7FFF).max()                 # largest |bf16| bit pattern
        f = np.array([mag], dtype=np.uint16)
        return float((f.astype(np.uint32) << 16).view(np.float32)[0])
    elif dt == 'F32':
        return float(np.abs(u.view(np.float32)).max())
    raise Exception(f"{oname}: bf16-src dtype {dt}")

def nbytes_of(tt, dims):
    ne = int(np.prod(dims)) if dims else 1
    if tt == GT_F32: return ne * 4
    if tt == GT_F16: return ne * 2
    if tt == GT_F8_E4M3: return ne * 1
    raise Exception(f"unexpected out type {tt}")

# ---- plan: choose out type per SRC tensor from the safetensors dtype ----
plan = []
hist = collections.Counter()
missing = []
n_ws_from_file = 0   # how many per-tensor scales came from the file vs bf16-amax recovery
wglobals = {}   # bare weight name -> recovery scale (amax(|bf16|)/448); folded into the FP8 GEMM A-scale
for name, dims in infos:
    if name.endswith('.wglobal'):
        continue  # nvfp4-unfolded sibling from the SRC list; we regenerate our own below
    oname = PFX + name
    if oname not in ho:
        missing.append(name); continue
    dt = ho[oname]['dtype']
    if dt == 'F8_E4M3' and ATTN_F16 and is_attn(name):
        ot = GT_F16          # dequant e4m3×wglobal -> F16 (comfy-style); no wglobal sibling needed
    elif dt == 'F8_E4M3':
        ot = GT_F8_E4M3
    elif is_keepf32(name):
        ot = GT_F32
    elif dt == 'F32':
        ot = GT_F32
    elif dt == 'BF16':
        ot = GT_F32 if F32LIN else GT_F16   # F32LIN widens comfy's BF16 Linears past F16 range
    else:
        raise Exception(f"{oname}: unexpected safetensors dtype {dt}")
    plan.append((name, oname, dims, ot, dt))
    hist[ot] += 1
    if ot == GT_F8_E4M3:
        # Lightricks' OFFICIAL fp8 release (Lightricks/LTX-2.3-fp8) is a ModelOpt-style export
        # and ships the exact per-tensor `<name>.weight_scale` alongside each fp8 weight.
        # ComfyUI's dev-fp8 does NOT (its scale lives in a runtime Params object), which is why
        # this tool recovers amax(|bf16|)/448 from the original checkpoint. Prefer the file's own
        # scale whenever it is present: the recovery is only accurate to ~0.3% (measured ratio
        # 0.9975..1.0031 on the official distilled-fp8), and that error lands straight on the
        # GEMM A-scale.
        wsk = oname.replace('.weight', '.weight_scale')
        if wsk in ho:
            wglobals[name] = float(st_raw(fo, bo, ho, wsk)[0].view(np.float32).ravel()[0])
            n_ws_from_file += 1
        else:
            amax = bf16_amax(oname)
            wglobals[name] = (amax / FP8_E4M3_MAX) if amax > 0.0 else 1.0

if missing:
    print(f"[WARN] {len(missing)} SRC tensors missing from fp8 safetensors (first 10): {missing[:10]}")

# sibling per-tensor recovery scales: one 1-elem F32 "<weight>.wglobal" per F8_E4M3 tensor.
# F16-stored Linears (comfy kept them bf16) need none -> unregistered -> A-scale defaults to 1.0.
for wname in wglobals:
    plan.append((wname + '.wglobal', None, [1], GT_F32, 'WGLOBAL'))

nt_out = len(plan)
_sc = np.array(list(wglobals.values()), dtype=np.float64) if wglobals else np.array([1.0])
print(f"plan: {nt_out} tensors  |  F8_E4M3={hist[GT_F8_E4M3]}  F16={hist[GT_F16]}  F32={hist[GT_F32]}  wglobal={len(wglobals)}")
print(f"wglobal source: {n_ws_from_file} from file weight_scale, {len(wglobals)-n_ws_from_file} from bf16 amax/448")
print(f"wglobal scale stats: min={_sc.min():.6g}  median={np.median(_sc):.6g}  max={_sc.max():.6g}  mean={_sc.mean():.6g}")

def build_data(name, oname, dims, ot, dt):
    u, sdt, sh = st_raw(fo, bo, ho, oname)
    # byte-count sanity: safetensors element count must equal ggml dims product
    ne = int(np.prod(dims)) if dims else 1
    ne_st = int(np.prod(sh)) if sh else 1
    assert ne == ne_st, f"{oname}: elem mismatch ggml {ne} (dims {dims}) vs st {ne_st} (shape {sh})"
    dW = lora_delta(name, sh)   # None unless FOLD_LORA and this weight has a matching LoRA pair
    if ot == GT_F8_E4M3:
        assert dt == 'F8_E4M3'
        assert len(u) == ne, f"{oname}: fp8 byte len {len(u)} != {ne}"
        if dW is None:
            return u.tobytes()  # verbatim e4m3 bytes (row-major [out,in] == ggml [in,out])
        # dequant comfy e4m3 × wglobal -> f32, add ΔW, requant e4m3 with a fresh per-tensor wglobal'.
        W = e4m3_dec_byte(u.reshape(sh)) * wglobals[name] + dW
        amax = float(np.abs(W).max()); wg2 = (amax / FP8_E4M3_MAX) if amax > 0.0 else 1.0
        wglobals[name] = wg2                       # updated BEFORE the sibling '.wglobal' is written (it is later in plan order)
        return e4m3_enc_signed(W / wg2).astype(np.uint8).tobytes()
    # decode to f32
    if dt == 'F32':
        x = u.view(np.float32).astype(np.float32)
    elif dt == 'BF16':
        x = bf16_to_f32(u)
    elif dt == 'F8_E4M3':
        # ATTN_F16 path: comfy e4m3 codes -> f32 via the per-tensor recovery scale (amax(|bf16|)/448),
        # so the F16 store carries the true weight magnitude (NOT the raw ±448 codes).
        amax = bf16_amax(oname); sc = (amax / FP8_E4M3_MAX) if amax > 0.0 else 1.0
        x = (e4m3_dec_byte(u.reshape(sh)) * sc).reshape(-1)
    else:
        raise Exception(f"{oname}: cannot decode dtype {dt}")
    if dW is not None:
        x = (x.reshape(sh) + dW).reshape(-1)
    if ot == GT_F32:
        return np.ascontiguousarray(x, dtype=np.float32).tobytes()
    if ot == GT_F16:
        return f32_to_f16(x).tobytes()
    raise Exception(ot)

# ---- write gguf (same magic/ver/KV; nt = plan length) ----
o = open(OUT, 'wb')
o.write(magic); o.write(struct.pack('<I', ver)); o.write(struct.pack('<Q', nt_out)); o.write(struct.pack('<Q', nkv)); o.write(KV)
off = 0
for name, oname, dims, ot, dt in plan:
    nbk = name.encode(); o.write(struct.pack('<Q', len(nbk))); o.write(nbk)
    o.write(struct.pack('<I', len(dims)))
    for d in dims: o.write(struct.pack('<Q', d))
    nb = nbytes_of(ot, dims); o.write(struct.pack('<I', ot)); o.write(struct.pack('<Q', off))
    off += nb + ((-nb) % align)
cur = o.tell(); o.write(b'\x00' * ((-cur) % align))
for name, oname, dims, ot, dt in plan:
    if dt == 'WGLOBAL':
        # read AFTER all main tensors are built (siblings are last in plan) -> sees folded wglobal'
        data = struct.pack('<f', wglobals[name[:-len('.wglobal')]])
    else:
        data = build_data(name, oname, dims, ot, dt)
    nb = nbytes_of(ot, dims); assert len(data) == nb, f"{name}: {len(data)} vs {nb}"
    o.write(data); o.write(b'\x00' * ((-len(data)) % align))
o.close()
print(f"wrote {OUT} ({os.path.getsize(OUT)} bytes)")
print(f"tensor-type histogram: F8_E4M3={hist[GT_F8_E4M3]}  F16={hist[GT_F16]}  F32={hist[GT_F32]}")
if FOLD_LORA:
    _sc2 = np.array(list(wglobals.values()), dtype=np.float64)
    print(f"post-fold wglobal stats: min={_sc2.min():.6g}  median={np.median(_sc2):.6g}  max={_sc2.max():.6g}")
