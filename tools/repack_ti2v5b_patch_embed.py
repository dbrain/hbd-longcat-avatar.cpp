#!/usr/bin/env python3
"""
Re-pack a city96/Kiijoku Wan2.2-TI2V-5B-Turbo GGUF so it loads in our sd.cpp port.

The only incompatibility: `patch_embedding.weight` is stored 5D
  gguf dims [2,2,1,48,3072]  (= numpy [out=3072, in=48, kt=1, kh=2, kw=2])
but ggml is 4D-max and our loader/CausalConv3d expects the out-major packed form
  gguf dims [2,2,1,147456]   (= numpy [out*in=147456, kt=1, kh=2, kw=2])
(same convention as our working A14B GGUF: [2,2,1,184320]=5120*36, and
tools/convert_wan22_vae.py: "5D [out,in,kt,kh,kw] -> 4D [out*in,kt,kh,kw] out-major").

patch_embedding.weight is F32 and C-contiguous with `out` outermost, so the merge
is a pure numpy reshape — ZERO byte movement, bit-identical weights. Detection then
sees ne[3]==147456 -> VERSION_WAN2_2_TI2V; num_layers==30 -> the 5B config branch.
Every other tensor (Linears 2D, norms 1D) already matches the loader; quantized
tensors are copied byte-exact via raw_dtype (Turbo distill + quant preserved).

Usage:
  /mnt/hdd/nava/.venv/bin/python tools/repack_ti2v5b_patch_embed.py \
      --src models/Wan2_2-TI2V-5B-Turbo-Q8_0.gguf \
      --out models/wan22-ti2v-5b-turbo-q8_0.gguf
"""
import argparse
import numpy as np
import gguf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    r = gguf.GGUFReader(args.src)
    arch = r.fields["general.architecture"].contents()
    w = gguf.GGUFWriter(args.out, arch)

    # carry over the (minimal) KV metadata
    if "general.quantization_version" in r.fields:
        w.add_quantization_version(r.fields["general.quantization_version"].contents())
    if "general.file_type" in r.fields:
        w.add_file_type(r.fields["general.file_type"].contents())

    F32 = gguf.GGMLQuantizationType.F32
    F16 = gguf.GGMLQuantizationType.F16
    n_fixed = 0
    for t in r.tensors:
        if t.name == "patch_embedding.weight":
            a = np.ascontiguousarray(t.data)              # numpy [out,in,kt,kh,kw]=(3072,48,1,2,2)
            out, inn, kt, kh, kw = a.shape
            a = a.reshape(out * inn, kt, kh, kw)          # out-major merge -> (147456,1,2,2)
            w.add_tensor(t.name, a)                        # F32 -> gguf dims [kw,kh,kt,147456]
            print(f"  repacked {t.name}: {(out,inn,kt,kh,kw)} -> {a.shape}  ne[3]={out*inn}")
            n_fixed += 1
        elif t.tensor_type in (F32, F16):
            w.add_tensor(t.name, np.ascontiguousarray(t.data))
        else:
            # quantized: copy raw bytes byte-exact. t.data is uint8 with the *byte*
            # shape; gguf-py derives the logical dims from it via raw_dtype (leave
            # raw_shape defaulted to tensor.shape = the byte shape).
            w.add_tensor(
                t.name,
                np.ascontiguousarray(t.data),
                raw_dtype=t.tensor_type,
            )

    assert n_fixed == 1, f"expected exactly 1 patch_embedding, fixed {n_fixed}"
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
