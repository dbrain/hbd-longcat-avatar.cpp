#!/usr/bin/env python3
"""Remap a HF PEFT LoRA adapter (adapter_model.safetensors) into the engine's
native comfy convention so fold_generic_lora.py can fold it into an nvfp4 gguf.

PEFT stores keys as   base_model.model.<module>.lora_A.weight  (F32, alpha NOT baked).
The engine / fold_generic_lora.py expects  diffusion_model.<gguf-stem>.lora_A.weight
(BF16), and applies delta = scale*(B@A) with scale supplied on the CLI.

For OmniNFT (LTX-2.3 RL-LoRA) the module paths already match the gguf tensor stems
(transformer_blocks.N.{attn1,attn2,audio_attn1,...,ff.net.2,ff.net.0.proj}), so the
remap is a pure prefix swap  base_model.model. -> diffusion_model.  plus F32->BF16.
Native PEFT strength is alpha/r (OmniNFT: 64/32 = 2.0) -> pass --scale 2.0 to the fold.

Usage:
  python3 tools/remap_peft_lora.py \
      --in  models/ltx2/loras/_hf_omninft/LTX-2.3-RL-Lora/adapter_model.safetensors \
      --out models/ltx2/loras/omninft-2.3-diffconv.safetensors
"""
import sys, os, json, struct, argparse
import numpy as np

SRC_PREFIX = "base_model.model."
DST_PREFIX = "diffusion_model."


def f32_to_bf16_u16(f32):
    """Round-to-nearest-even f32 -> bf16, returned as uint16."""
    u = f32.astype(np.float32).view(np.uint32)
    # round-to-nearest-even on the truncated 16 low bits
    lsb = (u >> 16) & 1
    bias = 0x7FFF + lsb
    u = (u + bias) >> 16
    return u.astype(np.uint16)


def st_read(path):
    f = open(path, 'rb')
    n = struct.unpack('<Q', f.read(8))[0]
    h = json.loads(f.read(n))
    base = 8 + n
    return f, base, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--src-prefix', default=SRC_PREFIX)
    ap.add_argument('--dst-prefix', default=DST_PREFIX)
    args = ap.parse_args()

    f, base, h = st_read(args.inp)
    keys = [k for k in h if k != '__metadata__']
    renamed, skipped = {}, []
    for k in keys:
        if not k.startswith(args.src_prefix):
            skipped.append(k)
            continue
        nk = args.dst_prefix + k[len(args.src_prefix):]
        renamed[nk] = k

    print(f"in : {args.inp}  ({len(keys)} tensors)")
    print(f"remap {len(renamed)} keys  '{args.src_prefix}' -> '{args.dst_prefix}'")
    if skipped:
        print(f"  WARNING: {len(skipped)} keys did not start with src-prefix (skipped):")
        for k in skipped[:8]:
            print("    ", k)

    # build output header + data (BF16)
    out_header = {}
    blobs = []
    offset = 0
    for nk in sorted(renamed):
        ok = renamed[nk]
        m = h[ok]
        a, b = m['data_offsets']
        f.seek(base + a)
        raw = f.read(b - a)
        dt = m['dtype']
        if dt == 'F32':
            arr = np.frombuffer(raw, dtype=np.float32)
            u16 = f32_to_bf16_u16(arr)
        elif dt in ('BF16', 'F16'):
            u16 = np.frombuffer(raw, dtype=np.uint16).copy()
        else:
            sys.exit(f"unsupported dtype {dt} for {ok}")
        data = u16.tobytes()
        out_header[nk] = {'dtype': 'BF16', 'shape': m['shape'],
                          'data_offsets': [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)

    hdr = json.dumps(out_header).encode('utf-8')
    pad = (-len(hdr)) % 8
    hdr += b' ' * pad
    with open(args.out, 'wb') as o:
        o.write(struct.pack('<Q', len(hdr)))
        o.write(hdr)
        for d in blobs:
            o.write(d)
    print(f"wrote {args.out}  ({os.path.getsize(args.out)} bytes, {len(out_header)} tensors BF16)")


if __name__ == '__main__':
    main()
