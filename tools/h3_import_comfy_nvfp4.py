#!/usr/bin/env python3
"""Import ComfyUI NVFP4 residual weights into a pruned MiniMax-H3 GGUF.

Comfy checkpoints store packed FP4 values and FP8 group scales in the blocked
cuBLAS layout.  The engine's GGUF uses interleaved ``block_nvfp4`` records
instead.  This tool converts the 200 hot attention/MLP matrices while retaining
the base GGUF's exact pruned AdaLN tables and protected conditioning tensors.

The source may be a "mixed" checkpoint containing other Comfy quant formats;
only linears explicitly selected below must declare ``{"format":"nvfp4"}``.
"""

import argparse
import json
import os
import re
import struct

import numpy as np

from h3_prune_adaln import gguf_read_header, nbytes, pack_str


NVFP4 = 40
SELECT = re.compile(
    r"^blocks\.\d+\.(?:attn\.(?:qkv_proj|out_proj)|mlp\.(?:fc1|fc2))\.weight$"
)


def read_safetensors_header(path):
    f = open(path, "rb")
    header_size_raw = f.read(8)
    if len(header_size_raw) != 8:
        raise SystemExit(f"{path}: truncated safetensors header")
    header_size = struct.unpack("<Q", header_size_raw)[0]
    header = json.loads(f.read(header_size))
    data_start = 8 + header_size
    header.pop("__metadata__", None)
    for value in header.values():
        begin, end = value["data_offsets"]
        value["offset"] = data_start + begin
        value["nbytes"] = end - begin
    return f, header


def read_entry(f, entry):
    f.seek(entry["offset"])
    data = f.read(entry["nbytes"])
    if len(data) != entry["nbytes"]:
        raise SystemExit("short safetensors read")
    return data


def copy_bytes(src, dst, offset, size):
    src.seek(offset)
    left = size
    while left:
        data = src.read(min(left, 1 << 24))
        if not data:
            raise SystemExit("short GGUF read")
        dst.write(data)
        left -= len(data)


def swizzled_scale_offsets(rows, groups, n_col_blocks):
    """Return Comfy's blocked scale-plane offsets for a row chunk."""
    row = np.asarray(rows, dtype=np.int64)[:, None]
    group = np.arange(groups, dtype=np.int64)[None, :]
    rb = row // 128
    rem = row % 128
    d4 = rem // 32
    d3 = rem % 32
    cbg = group // 4
    d5 = group % 4
    return (((rb * n_col_blocks + cbg) * 32 + d3) * 16 +
            d4 * 4 + d5)


def write_comfy_nvfp4(out, source_path, weight, scale, in_features,
                      out_features, rows_per_chunk, row_map=None):
    """Convert one Comfy packed/swizzled matrix to ggml block_nvfp4."""
    packed = np.memmap(
        source_path, mode="r", dtype=np.uint8, offset=weight["offset"],
        shape=(out_features, in_features // 2),
    )
    scales = np.memmap(
        source_path, mode="r", dtype=np.uint8, offset=scale["offset"],
        shape=(scale["nbytes"],),
    )
    nsub = in_features // 16
    nblk = in_features // 64
    n_col_blocks = (nsub + 3) // 4

    for row0 in range(0, out_features, rows_per_chunk):
        row1 = min(row0 + rows_per_chunk, out_features)
        source_rows = (np.arange(row0, row1) if row_map is None
                       else row_map[row0:row1])
        source = np.asarray(packed[source_rows]).reshape(row1 - row0, nsub, 8)

        # Comfy: element 2j in the high nibble, 2j+1 in the low nibble.
        # ggml: element j in the low nibble, element j+8 in the high nibble.
        first = source[:, :, :4]
        second = source[:, :, 4:]
        quants = np.empty_like(source)
        quants[:, :, 0::2] = (first >> 4) | (second & 0xF0)
        quants[:, :, 1::2] = (first & 0x0F) | ((second & 0x0F) << 4)

        scale_offsets = swizzled_scale_offsets(
            source_rows, nsub, n_col_blocks
        )
        group_scales = np.asarray(scales[scale_offsets])

        blocks = np.empty((row1 - row0, nblk, 36), dtype=np.uint8)
        blocks[:, :, :4] = group_scales.reshape(row1 - row0, nblk, 4)
        blocks[:, :, 4:] = quants.reshape(row1 - row0, nblk, 32)
        out.write(blocks.tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="pruned GGUF supplying structure")
    parser.add_argument("--source", required=True, help="Comfy mixed-NVFP4 safetensors")
    parser.add_argument("--out", required=True)
    parser.add_argument("--rows-per-chunk", type=int, default=128)
    args = parser.parse_args()
    if args.rows_per_chunk <= 0:
        parser.error("--rows-per-chunk must be positive")

    base_f, version, kv_blob, n_kv, base_tensors, base_data, align = (
        gguf_read_header(args.base)
    )
    source_f, source = read_safetensors_header(args.source)
    base_by_name = {tensor["name"]: tensor for tensor in base_tensors}
    if ("attn.qkv_deinterleaved" not in base_by_name or
            "rope.qk_permuted" not in base_by_name):
        raise SystemExit(
            "base GGUF must carry qkv-deinterleaved and q/k-RoPE-permuted markers"
        )
    head_dim = base_by_name["blocks.0.attn.q_norm.weight"]["dims"][0]
    rot_dim = 2 * 3 * base_by_name["rope.inv_freq"]["dims"][0]
    half = head_dim // 2
    rot_half = rot_dim // 2
    pass_half = (head_dim - rot_dim) // 2
    qk_perm = np.empty(head_dim, dtype=np.int64)
    for channel in range(rot_half):
        qk_perm[channel] = channel
        qk_perm[half + channel] = rot_half + channel
    for channel in range(pass_half):
        qk_perm[rot_half + channel] = rot_dim + channel
        qk_perm[half + rot_half + channel] = rot_dim + pass_half + channel

    imports = {}
    for tensor in base_tensors:
        name = tensor["name"]
        if not SELECT.match(name):
            continue
        stem = name[:-len(".weight")]
        weight = source.get(name)
        scale = source.get(stem + ".weight_scale")
        global_scale = source.get(stem + ".weight_scale_2")
        quant = source.get(stem + ".comfy_quant")
        if any(value is None for value in (weight, scale, global_scale, quant)):
            raise SystemExit(f"{stem}: incomplete Comfy NVFP4 tensor family")
        try:
            quant_config = json.loads(read_entry(source_f, quant))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise SystemExit(f"{stem}: malformed comfy_quant: {error}") from error
        if quant_config != {"format": "nvfp4"}:
            raise SystemExit(f"{stem}: unsupported comfy_quant {quant_config!r}")
        in_features, out_features = tensor["dims"]
        expected_scale_shape = [
            ((out_features + 127) // 128) * 128,
            (((in_features // 16) + 3) // 4) * 4,
        ]
        if (tensor["type"] != NVFP4 or weight["dtype"] != "U8" or
                weight["shape"] != [out_features, in_features // 2] or
                scale["dtype"] != "F8_E4M3" or
                scale["shape"] != expected_scale_shape or
                global_scale["dtype"] != "F32" or
                global_scale["shape"] != [] or global_scale["nbytes"] != 4):
            raise SystemExit(f"{stem}: incompatible Comfy NVFP4 layout")
        imports[name] = (weight, scale, global_scale)

    if len(imports) != 200:
        raise SystemExit(f"expected 200 residual NVFP4 matrices, found {len(imports)}")

    plan = []
    for tensor in base_tensors:
        source_name = tensor["name"]
        item = {
            "name": source_name,
            "dims": tensor["dims"],
            "type": tensor["type"],
            "kind": "copy",
            "offset": base_data + tensor["off"],
        }
        if source_name in imports:
            item["kind"] = "comfy"
            item["import"] = imports[source_name]
            if source_name.endswith(".attn.qkv_proj.weight"):
                out_features = tensor["dims"][1]
                inner = out_features // 3
                heads = inner // head_dim
                row_map = np.arange(out_features, dtype=np.int64)
                for third in range(2):
                    start = third * inner
                    rows = row_map[start:start + inner].reshape(heads, head_dim)
                    rows[:] = (start + np.arange(heads)[:, None] * head_dim +
                               qk_perm[None, :])
                item["row_map"] = row_map
        elif source_name.endswith(".weight.wglobal"):
            weight_name = source_name[:-len(".wglobal")]
            if weight_name in imports:
                item["kind"] = "global"
                item["global"] = imports[weight_name][2]
        plan.append(item)

    temp = args.out + ".partial"
    with open(temp, "wb") as out:
        out.write(b"GGUF")
        out.write(struct.pack("<I", version))
        out.write(struct.pack("<Q", len(plan)))
        out.write(struct.pack("<Q", n_kv))
        out.write(kv_blob)

        offset = 0
        entries = []
        for item in plan:
            size = nbytes(item["dims"], item["type"])
            entries.append((item, offset, size))
            offset += size + (-size % align)

        for item, tensor_offset, _size in entries:
            out.write(pack_str(item["name"]))
            out.write(struct.pack("<I", len(item["dims"])))
            for dim in item["dims"]:
                out.write(struct.pack("<Q", int(dim)))
            out.write(struct.pack("<I", item["type"]))
            out.write(struct.pack("<Q", tensor_offset))

        out.write(b"\0" * (-out.tell() % align))
        data_start = out.tell()
        imported = 0
        for index, (item, tensor_offset, size) in enumerate(entries):
            if out.tell() - data_start != tensor_offset:
                raise SystemExit(f"offset drift at {item['name']}")
            if item["kind"] == "copy":
                copy_bytes(base_f, out, item["offset"], size)
            elif item["kind"] == "global":
                copy_bytes(source_f, out, item["global"]["offset"], 4)
            else:
                weight, scale, _global = item["import"]
                before = out.tell()
                write_comfy_nvfp4(
                    out, args.source, weight, scale, item["dims"][0],
                    item["dims"][1], args.rows_per_chunk, item.get("row_map"),
                )
                if out.tell() - before != size:
                    raise SystemExit(f"{item['name']}: converted size mismatch")
                imported += 1
            out.write(b"\0" * (-size % align))
            if index % 50 == 0:
                print(f"{index}/{len(entries)}", flush=True)

    os.replace(temp, args.out)
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e9:.2f} GB)")
    print(f"imported {imported} Comfy NVFP4 matrices")


if __name__ == "__main__":
    main()
