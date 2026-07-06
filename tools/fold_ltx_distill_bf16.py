#!/usr/bin/env python3
"""Fold the LTX-2.3 distill LoRA into the bf16 DEV DiT safetensors, producing a
high-precision merged checkpoint (dev + distill@<mult>) suitable as the SOURCE
for an imatrix-calibrated nvfp4 `-M convert`. CPU-only, streaming per-tensor.

WHY bf16 (not dq(CLEAN)): imatrix quant wants a pristine high-precision source so
the ONLY quant error is the single imatrix-minimised nvfp4 rounding. Folding into
an already-nvfp4 gguf then re-quantizing compounds two ~13% errors.

Math (verified against tools/fold_lipdub_lora.py + lora metadata):
  lora_alpha == lora_rank == 384  ->  lora scale = alpha/rank = 1.0
  delta[out,in] = mult * (B[out,r] @ A[r,in])        (no per-key alpha tensors)
  W_merged = W + delta                                (same [out,in] orientation)
Name map: lora key 'diffusion_model.X.lora_{A,B}.weight'
          -> base  'model.diffusion_model.X.weight'   (dev safetensors prefix)

Usage:
  fold_ltx_distill_bf16.py <dev.safetensors> <distill_lora.safetensors> <out.safetensors> [mult=0.5]
  # add --dry-run to only validate name/shape matching (no output written)
"""
import sys, json, struct, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("dev"); ap.add_argument("lora"); ap.add_argument("out")
ap.add_argument("mult", nargs="?", type=float, default=0.5)
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

BASE_PFX = "model.diffusion_model."
LORA_PFX = "diffusion_model."

def st_header(path):
    f = open(path, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return f, 8 + n, h

# numpy dtype <-> safetensors dtype string
NP = {"F64": np.float64, "F32": np.float32, "F16": np.float16,
      "BF16": None, "I64": np.int64, "I32": np.int32, "I16": np.int16,
      "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_}

def bf16_to_f32(u8):
    return (u8.view(np.uint16).astype(np.uint32) << 16).view(np.float32)

def f32_to_bf16(x):
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    r = ((u >> 16) & 1) + 0x7FFF
    return ((u + r) >> 16).astype(np.uint16)

def load(f, base, meta, key):
    dt = meta[key]["dtype"]; shp = meta[key]["shape"]
    a0, b0 = meta[key]["data_offsets"]
    f.seek(base + a0); raw = np.frombuffer(f.read(b0 - a0), dtype=np.uint8)
    if dt == "BF16":
        x = bf16_to_f32(raw).reshape(shp)
    else:
        x = raw.view(NP[dt]).reshape(shp)
    return x, dt, shp

fd, bd, hd = st_header(a.dev)
fl, bl, hl = st_header(a.lora)

# discover lora targets
stems = set()
for k in hl:
    if ".lora_A.weight" in k:
        stems.add(k[:-len(".lora_A.weight")])
targets = {}  # base_key -> (A_key, B_key)
missing = []
for stem in sorted(stems):
    bare = stem[len(LORA_PFX):] if stem.startswith(LORA_PFX) else stem
    base_key = BASE_PFX + bare + ".weight"
    ak = stem + ".lora_A.weight"; bk = stem + ".lora_B.weight"
    if base_key not in hd or bk not in hl:
        missing.append((stem, base_key in hd, bk in hl)); continue
    # shape sanity: base [out,in], A [r,in], B [out,r]
    out_, in_ = hd[base_key]["shape"]
    r1, inA = hl[ak]["shape"]; outB, r2 = hl[bk]["shape"]
    if not (inA == in_ and outB == out_ and r1 == r2):
        missing.append((stem, "SHAPE", (out_, in_), (r1, inA), (outB, r2))); continue
    targets[base_key] = (ak, bk)

print(f"[fold] lora stems={len(stems)}  matched targets={len(targets)}  unmatched={len(missing)}")
for m in missing[:20]:
    print("  UNMATCHED:", m)
if a.dry_run:
    tot = len(hd); folded = len(targets)
    print(f"[dry-run] dev tensors={tot}  would-fold={folded}  passthrough={tot-folded}  mult={a.mult}")
    sys.exit(0 if not missing else 1)

# ---- streaming write: header then data, tensors in dev order ----
names = list(hd.keys())
# assemble output header
out_meta = {}; off = 0
for name in names:
    dt = hd[name]["dtype"]; shp = hd[name]["shape"]
    n = int(np.prod(shp)) if shp else 1
    esz = 2 if dt == "BF16" else np.dtype(NP[dt]).itemsize
    nb = n * esz
    out_meta[name] = {"dtype": dt, "shape": shp, "data_offsets": [off, off + nb]}
    off += nb
hdr = json.dumps(out_meta).encode()
pad = (8 - (len(hdr) % 8)) % 8
hdr += b" " * pad
o = open(a.out, "wb")
o.write(struct.pack("<Q", len(hdr))); o.write(hdr)
folded = 0
for name in names:
    x, dt, shp = load(fd, bd, hd, name)
    if name in targets:
        ak, bk = targets[name]
        A, _, _ = load(fl, bl, hl, ak)   # [r,in] f32
        B, _, _ = load(fl, bl, hl, bk)   # [out,r] f32
        delta = a.mult * (B.astype(np.float32) @ A.astype(np.float32))  # [out,in]
        x = x.astype(np.float32) + delta
        if dt == "BF16":
            data = f32_to_bf16(x).tobytes()
        else:
            data = np.ascontiguousarray(x, dtype=NP[dt]).tobytes()
        folded += 1
    else:
        # passthrough verbatim
        fd.seek(bd + hd[name]["data_offsets"][0])
        data = fd.read(hd[name]["data_offsets"][1] - hd[name]["data_offsets"][0])
    o.write(data)
o.close()
print(f"[fold] wrote {a.out}  folded={folded}/{len(targets)} targets  mult={a.mult}")
