#!/usr/bin/env python3
"""Import a lightx2v Wan2.2-A14B NVFP4(-Sparse) safetensors -> a ggml block_nvfp4 gguf that
longcat-avatar.cpp's Wan loader reads. The lightx2v checkpoint is a STANDARD ModelOpt NVFP4
export of the SAME 4-step MoE distill we have in bf16 (verified: sign 100%, codes 98.8%, global
0.2%). We transcode its EXACT e2m1 codes + per-16 e4m3 block scales into ggml block_nvfp4.

Format (verified numerically against the 151 bf16, per leg):
  <name>.weight              U8   [out, in/2]    packed e2m1, LOW nibble = first element
  <name>.weight_scale        F8_E4M3 [out, in/16]  per-16 block scale (LINEAR, not swizzled)
  <name>.alpha               F32  []             per-tensor  ─┐ product = weight global scale
  <name>.input_global_scale  F32  []             per-tensor  ─┘ ( = amax(W)/(6*448) )
Dequant: W = e2m1(code) * weight_scale * (alpha * input_global_scale).
ggml block_nvfp4: d[4] = per-16 UE4M3 scale (=weight_scale verbatim), qs = packed e2m1 (the ggml
2x kvalues_mxfp4 vs std-e2m1 and the /2 UE4M3 decode cancel, so codes+scale bytes pass verbatim).
The per-tensor global (alpha*igs) is carried as a sibling "<weight>.wglobal" F32 and folded into the
cuBLASLt GEMM alpha (NOT into the block scale — folding underflows e4m3 = the patchy-colour bug).

Non-fp4 tensors (modulation/norm/embedding/time_projection/head/bias) -> F32 (the model allocates
them F32; ggml_concat/add assert on uniform type). SRC ggml gguf supplies the tensor list/dims/KV
(flat Wan names == lightx2v names, so PFX=''); the f16 patch_embedding squeeze is inherited.

Usage: import_wan_nvfp4.py <src_ggml_gguf> <lightx2v_nvfp4_st> <out_gguf>
"""
import sys, json, struct, numpy as np, os
SRC, OFF, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
PFX = ''                              # lightx2v names are FLAT, == ggml bare names
GT_F32 = 0; GT_BF16 = 30; GT_NVFP4 = 40
NONFP4_TYPE = GT_F32                  # all non-fp4 -> F32 (safe: loader down-converts to wtype)

def st_open(p):
    f = open(p, 'rb'); n = struct.unpack('<Q', f.read(8))[0]; h = json.loads(f.read(n)); h.pop('__metadata__', None); return f, 8 + n, h
def st_raw(f, base, h, key):
    m = h[key]; a, b = m['data_offsets']; f.seek(base + a); return np.frombuffer(f.read(b - a), dtype=np.uint8), m['dtype'], m['shape']
def e4m3_dec_byte(u8):
    u = u8.astype(np.int32); e = (u >> 3) & 0xF; m = u & 0x7
    v = np.where(e == 0, np.ldexp(m.astype(np.float64), -9), np.ldexp(1.0 + m / 8.0, e - 7))
    return np.where((u & 0x7F) == 0, 0.0, v)
_pb = np.arange(128, dtype=np.uint8); _pv = e4m3_dec_byte(_pb)
_order = np.argsort(_pv); _sv = _pv[_order]; _sb = _pb[_order]
def e4m3_enc_pos(x):
    x = np.minimum(np.asarray(x, dtype=np.float64), 448.0)
    idx = np.searchsorted(_sv, x); idx = np.clip(idx, 0, len(_sv) - 1); lo = np.clip(idx - 1, 0, len(_sv) - 1)
    pick = np.where(np.abs(_sv[idx] - x) < np.abs(_sv[lo] - x), idx, lo); return _sb[pick].astype(np.uint8)
def bf16_to_f32(u8): return (u8.view(np.uint16).astype(np.uint32) << 16).view(np.float32).astype(np.float32)
def f32_to_bf16(x):
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32); r = ((u >> 16) & 1) + 0x7FFF; return ((u + r) >> 16).astype(np.uint16)
def swz_index(out, nb):
    # cuBLASLt SWIZZLE_32_4_4 block-scale layout -> linear [out, nb] gather indices. lightx2v
    # stores weight_scale swizzled (despite the [out, in/16] shape label); reading it linearly
    # gives relL2 ~0.49 garbage, un-swizzling gives ~0.10 (proper FP4). Same as LTX official.
    R, RC, CC = 128, 32, 4; rows = np.arange(out)[:, None]; cols = np.arange(nb)[None, :]
    rb = rows // R; rem = rows % R; d4 = rem // RC; d3 = rem % RC; cbg = cols // CC; d5 = cols % CC; cbg_cnt = (nb + CC - 1) // CC
    return (((rb * cbg_cnt + cbg) * RC + d3) * 16 + d4 * CC + d5)

fo, bo, ho = st_open(OFF)

def read_wglobal(oname):
    base = oname[:-len('.weight')]
    alpha = float(st_raw(fo, bo, ho, base + '.alpha')[0].view(np.float32)[0])
    igs   = float(st_raw(fo, bo, ho, base + '.input_global_scale')[0].view(np.float32)[0])
    return alpha * igs                # per-tensor weight global = amax(W)/(6*448)

def build_nvfp4(oname, dims):
    wu, _, wsh = st_raw(fo, bo, ho, oname)
    out, inn = wsh[0], wsh[1] * 2; nb16 = inn // 16
    W = wu.reshape(out, inn // 2); code = np.empty((out, inn), dtype=np.uint8)
    code[:, 0::2] = W & 0xF; code[:, 1::2] = W >> 4          # LO nibble = first element (Wan/lightx2v)
    su, _, ssh = st_raw(fo, bo, ho, oname.replace('.weight', '.weight_scale'))
    assert ssh[0] == out and ssh[1] == nb16, f"{oname} weight_scale {ssh} vs [{out},{nb16}]"
    sc = e4m3_dec_byte(su).ravel()[swz_index(out, nb16)]    # un-swizzle SWIZZLE_32_4_4 -> linear [out,nb16]
    d = e4m3_enc_pos(sc).reshape(out, nb16)                 # round-trip into ggml d encoding
    nblk = inn // 64
    oc = code.reshape(out, nblk, 64); qs = np.empty((out, nblk, 32), dtype=np.uint8)
    for s in range(4):
        sub = oc[:, :, s * 16:s * 16 + 16]; qs[:, :, s * 8:s * 8 + 8] = sub[:, :, 0:8] | (sub[:, :, 8:16] << 4)
    db = d.reshape(out, nblk, 4).astype(np.uint8)
    blk = np.concatenate([db, qs], axis=2)
    assert dims[0] == inn and dims[1] == out, f"{oname} dims {dims} vs in{inn} out{out}"
    return blk.tobytes()

def official_nonfp4(oname, ot):
    u, dt, _ = st_raw(fo, bo, ho, oname)
    if dt == 'BF16': x = bf16_to_f32(u)
    elif dt == 'F32': x = u.view(np.float32)
    elif dt == 'F16': x = u.view(np.float16).astype(np.float32)
    else: raise Exception(f"{oname} unexpected non-fp4 dtype {dt}")
    return f32_to_bf16(x).tobytes() if ot == GT_BF16 else np.ascontiguousarray(x, dtype=np.float32).tobytes()

# ---- parse SRC gguf (KV verbatim) ----
f = open(SRC, 'rb'); magic = f.read(4); ver = struct.unpack('<I', f.read(4))[0]
nt = struct.unpack('<Q', f.read(8))[0]; nkv = struct.unpack('<Q', f.read(8))[0]
kv_start = f.tell()
def rs(): n = struct.unpack('<Q', f.read(8))[0]; return f.read(n)
def sv(t):
    if t in (0, 1, 7): f.read(1)
    elif t in (2, 3): f.read(2)
    elif t in (4, 5, 6): f.read(4)
    elif t in (10, 11, 12): f.read(8)
    elif t == 8: rs()
    elif t == 9:
        et = struct.unpack('<I', f.read(4))[0]; nn = struct.unpack('<Q', f.read(8))[0]; [(rs() if et == 8 else sv(et)) for _ in range(nn)]
align = 32
for _ in range(nkv):
    kl = struct.unpack('<Q', f.read(8))[0]; k = f.read(kl); t = struct.unpack('<I', f.read(4))[0]
    if k == b'general.alignment': align = struct.unpack('<I', f.read(4))[0]
    else: sv(t)
kv_end = f.tell(); f.seek(kv_start); KV = f.read(kv_end - kv_start)
infos = []
for _ in range(nt):
    kl = struct.unpack('<Q', f.read(8))[0]; name = f.read(kl).decode(); nd = struct.unpack('<I', f.read(4))[0]
    dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(nd)]; tt = struct.unpack('<I', f.read(4))[0]; off = struct.unpack('<Q', f.read(8))[0]
    infos.append([name, dims])

def nbytes_of(tt, dims):
    ne = int(np.prod(dims))
    if tt == GT_F32: return ne * 4
    if tt == GT_BF16: return ne * 2
    if tt == GT_NVFP4: nb = dims[0] // 64; ne1 = ne // dims[0]; return ne1 * nb * 36
    raise Exception(tt)

plan = []; nfp4 = 0; nf32 = 0; wglobals = {}
for name, dims in infos:
    oname = PFX + name
    assert oname in ho, f"missing official {oname}"
    if ho[oname]['dtype'] == 'U8':
        plan.append((name, oname, dims, GT_NVFP4)); nfp4 += 1
        wglobals[name] = read_wglobal(oname)
    else:
        plan.append((name, oname, dims, NONFP4_TYPE)); nf32 += 1
for wname in wglobals:
    plan.append((wname + '.wglobal', None, [1], GT_F32))
nt = nt + len(wglobals)
print(f"plan: {nfp4} nvfp4 + {nf32} nonfp4 + {len(wglobals)} wglobal = {len(plan)} tensors")

o = open(OUT, 'wb')
o.write(magic); o.write(struct.pack('<I', ver)); o.write(struct.pack('<Q', nt)); o.write(struct.pack('<Q', nkv)); o.write(KV)
off = 0
for name, oname, dims, ot in plan:
    nbk = name.encode(); o.write(struct.pack('<Q', len(nbk))); o.write(nbk); o.write(struct.pack('<I', len(dims)))
    for d in dims: o.write(struct.pack('<Q', d))
    nb = nbytes_of(ot, dims); o.write(struct.pack('<I', ot)); o.write(struct.pack('<Q', off)); off += nb + ((-nb) % align)
cur = o.tell(); o.write(b'\x00' * ((-cur) % align))
for i, (name, oname, dims, ot) in enumerate(plan):
    if name.endswith('.wglobal'):
        data = struct.pack('<f', wglobals[name[:-len('.wglobal')]])
    elif ot == GT_NVFP4:
        data = build_nvfp4(oname, dims)
    else:
        data = official_nonfp4(oname, ot)
    nb = nbytes_of(ot, dims); assert len(data) == nb, f"{name} {len(data)} vs {nb}"
    o.write(data); o.write(b'\x00' * ((-len(data)) % align))
    if i % 200 == 0: print(f"  .. {i}/{len(plan)}")
o.close()
print(f"wrote {OUT} ({os.path.getsize(OUT)} bytes)")
