#!/usr/bin/env python3
"""Build the quality-preserving pruned MiniMax-H3 NVFP4 GGUF.

The old H3 conversion applied flat, absolute E4M3 group scales to every eligible
matrix.  This builder instead matches TensorCore NVFP4's two-level scale:

    weight ~= E2M1_value * E4M3_group_scale * F32_tensor_global

The global keeps the E4M3 scale plane well-conditioned instead of rounding
absolute scales directly.  Only the 200 residual-stack attention/MLP matrices
are NVFP4.  The condition projection and two token-refiner blocks are restored
from the BF16 donor because they are one-shot, quality-sensitive paths.

All remaining tensors and metadata are copied byte-for-byte from an existing
pruned GGUF, including AdaLN's compact timestep table and H3 marker tensors.
"""

import argparse
import os
import re
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_folded_nvfp4 import quant_nvfp4_unfolded  # noqa: E402
from h3_prune_adaln import gguf_read_header, nbytes, pack_str  # noqa: E402

F32 = 0
F16 = 1
BF16 = 30
NVFP4 = 40
SELECT = re.compile(
    r"^blocks\.\d+\.(?:attn\.(?:qkv_proj|out_proj)|mlp\.(?:fc1|fc2))\.weight$"
)
PROTECT = re.compile(
    r"^(?:condition_proj|token_refiner\.blocks\.\d+\."
    r"(?:attn\.(?:qkv_proj|out_proj)|mlp\.(?:fc1|fc2)))\.weight$"
)


def dense_item_size(t):
    if t == F32:
        return 4
    if t in (F16, BF16):
        return 2
    raise SystemExit(f"dense donor type {t} is not F32/F16/BF16")


def read_dense_rows(f, data_start, tensor, row0, rows):
    k, n = tensor["dims"]
    if row0 < 0 or rows <= 0 or row0 + rows > n:
        raise ValueError("row range outside tensor")
    item = dense_item_size(tensor["type"])
    f.seek(data_start + tensor["off"] + row0 * k * item)
    raw = f.read(rows * k * item)
    if len(raw) != rows * k * item:
        raise SystemExit(f"{tensor['name']}: short donor read")
    if tensor["type"] == BF16:
        out = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
    elif tensor["type"] == F16:
        out = np.frombuffer(raw, dtype="<f2").astype(np.float32)
    else:
        out = np.frombuffer(raw, dtype="<f4").copy()
    return out.reshape(rows, k)


def tensor_amax(f, data_start, tensor, rows_per_chunk):
    _k, n = tensor["dims"]
    result = 0.0
    for row0 in range(0, n, rows_per_chunk):
        rows = min(rows_per_chunk, n - row0)
        result = max(result, float(np.abs(read_dense_rows(
            f, data_start, tensor, row0, rows)).max()))
    return result


def copy_bytes(src, dst, offset, size):
    src.seek(offset)
    left = size
    while left:
        chunk = src.read(min(left, 1 << 24))
        if not chunk:
            raise SystemExit("short read while copying tensor payload")
        dst.write(chunk)
        left -= len(chunk)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="pruned GGUF supplying structure")
    ap.add_argument("--donor", required=True, help="BF16/F16/F32 GGUF supplying weights")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows-per-chunk", type=int, default=32)
    ap.add_argument("--scale-search-radius", type=int, default=0, choices=range(33),
                    metavar="N", help="MSE-refine each E4M3 group scale over +/- N codes "
                                     "(default: 0, legacy amax quantizer)")
    args = ap.parse_args()
    if args.rows_per_chunk <= 0:
        ap.error("--rows-per-chunk must be positive")

    base_f, version, kv_blob, n_kv, base_ts, base_data, align = gguf_read_header(args.base)
    donor_f, donor_version, _kv, _nkv, donor_ts, donor_data, _align = gguf_read_header(args.donor)
    if donor_version != version:
        raise SystemExit(f"GGUF version mismatch: base v{version}, donor v{donor_version}")
    donor_by = {t["name"]: t for t in donor_ts}

    plan = []
    selected = []
    protected = []
    for bt in base_ts:
        name = bt["name"]
        if name.endswith(".wglobal"):
            continue
        if SELECT.match(name):
            dt = donor_by.get(name)
            if dt is None or dt["dims"] != bt["dims"] or len(dt["dims"]) != 2:
                raise SystemExit(f"{name}: donor missing or incompatible")
            if dt["dims"][0] % 64:
                raise SystemExit(f"{name}: input width is not divisible by 64")
            item = {"name": name, "dims": bt["dims"], "type": NVFP4,
                    "kind": "quant", "donor": dt}
            plan.append(item)
            selected.append(item)
            continue
        if PROTECT.match(name):
            dt = donor_by.get(name)
            if dt is None or dt["dims"] != bt["dims"] or dt["type"] not in (F32, F16, BF16):
                raise SystemExit(f"{name}: protected donor missing or incompatible")
            item = {"name": name, "dims": dt["dims"], "type": dt["type"],
                    "kind": "donor-copy", "offset": donor_data + dt["off"]}
            item["size"] = nbytes(item["dims"], item["type"])
            plan.append(item)
            protected.append(name)
            continue
        size = nbytes(bt["dims"], bt["type"])
        plan.append({"name": name, "dims": bt["dims"], "type": bt["type"],
                     "kind": "base-copy", "offset": base_data + bt["off"], "size": size})

    if len(selected) != 200:
        raise SystemExit(f"expected 200 residual-stack linears, found {len(selected)}")
    if len(protected) != 9:
        raise SystemExit(f"expected 9 protected matrices, found {len(protected)}")

    for index, item in enumerate(selected, 1):
        dt = item["donor"]
        amax = tensor_amax(donor_f, donor_data, dt, args.rows_per_chunk)
        # The sidecar is F32, so quantize against that exact rounded value too.
        # Otherwise a group sitting precisely on an E4M3 midpoint can be encoded
        # from the Python F64 value but decoded with a slightly different scalar.
        item["wglobal"] = float(np.float32(
            amax / (6.0 * 448.0) if amax > 0 else 1.0))
        plan.append({"name": item["name"] + ".wglobal", "dims": [1], "type": F32,
                     "kind": "data", "data": struct.pack("<f", item["wglobal"])})
        print(f"scale {index:3d}/200  {item['name']:<52} "
              f"amax={amax:.7g} global={item['wglobal']:.9g}", flush=True)

    entries = []
    offset = 0
    for item in plan:
        size = nbytes(item["dims"], item["type"])
        entries.append((item, offset, size))
        offset += size + (-size % align)

    temp = args.out + ".partial"
    with open(temp, "wb") as out:
        out.write(b"GGUF")
        out.write(struct.pack("<I", version))
        out.write(struct.pack("<Q", len(plan)))
        out.write(struct.pack("<Q", n_kv))
        out.write(kv_blob)
        for item, tensor_offset, _size in entries:
            out.write(pack_str(item["name"]))
            out.write(struct.pack("<I", len(item["dims"])))
            for dim in item["dims"]:
                out.write(struct.pack("<Q", int(dim)))
            out.write(struct.pack("<I", item["type"]))
            out.write(struct.pack("<Q", tensor_offset))
        out.write(b"\0" * (-out.tell() % align))
        output_data = out.tell()

        for index, (item, tensor_offset, size) in enumerate(entries, 1):
            if out.tell() - output_data != tensor_offset:
                raise SystemExit(f"offset drift at {item['name']}")
            if item["kind"] in ("base-copy", "donor-copy"):
                src = base_f if item["kind"] == "base-copy" else donor_f
                copy_bytes(src, out, item["offset"], item["size"])
            elif item["kind"] == "data":
                out.write(item["data"])
            else:
                dt = item["donor"]
                k, n = dt["dims"]
                wrote = 0
                for row0 in range(0, n, args.rows_per_chunk):
                    rows = min(args.rows_per_chunk, n - row0)
                    w = read_dense_rows(donor_f, donor_data, dt, row0, rows)
                    packed, wg = quant_nvfp4_unfolded(
                        w, flat=False, wglobal=item["wglobal"],
                        scale_search_radius=args.scale_search_radius)
                    if wg != item["wglobal"]:
                        raise SystemExit("internal weight-global drift")
                    out.write(packed)
                    wrote += len(packed)
                if wrote != size:
                    raise SystemExit(f"{item['name']}: wrote {wrote}, expected {size}")
            out.write(b"\0" * (-size % align))
            if index % 25 == 0 or item["kind"] == "quant":
                print(f"write {index:3d}/{len(entries)}  {item['name']}", flush=True)

    os.replace(temp, args.out)
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e9:.2f} GB)")
    print("recipe: 200 unfolded NVFP4 residual linears, 9 donor-precision protected matrices, "
          f"E4M3 scale MSE search radius {args.scale_search_radius}")


if __name__ == "__main__":
    main()
