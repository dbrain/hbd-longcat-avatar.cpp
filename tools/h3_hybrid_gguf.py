#!/usr/bin/env python3
"""Build a MiniMax-H3 hybrid GGUF by taking selected tensors from a second GGUF.

This is an isolation tool for quantization-sensitive components. The output keeps the
base file's metadata and tensor order, but selected tensors use the donor's dtype and
payload. Tensor names and logical dimensions must match exactly.
"""

import argparse
import os
import re
import struct

from h3_prune_adaln import gguf_read_header, nbytes, pack_str


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="GGUF supplying metadata and unselected tensors")
    parser.add_argument("--donor", required=True, help="GGUF supplying selected tensors")
    parser.add_argument("--out", required=True)
    parser.add_argument("--select", required=True, help="regular expression matched against tensor names")
    args = parser.parse_args()

    select = re.compile(args.select)
    base_f, version, kv_blob, n_kv, base_tensors, base_data, align = gguf_read_header(args.base)
    donor_f, donor_version, _kv, _n_kv, donor_tensors, donor_data, _align = gguf_read_header(args.donor)
    if donor_version != version:
        raise SystemExit(f"GGUF version mismatch: base v{version}, donor v{donor_version}")
    donor_by_name = {tensor["name"]: tensor for tensor in donor_tensors}

    plan = []
    selected = []
    for base_tensor in base_tensors:
        source_f, source_data, source_tensor = base_f, base_data, base_tensor
        if select.search(base_tensor["name"]):
            donor_tensor = donor_by_name.get(base_tensor["name"])
            if donor_tensor is None:
                raise SystemExit(f"selected tensor missing from donor: {base_tensor['name']}")
            if donor_tensor["dims"] != base_tensor["dims"]:
                raise SystemExit(
                    f"dimension mismatch for {base_tensor['name']}: "
                    f"{base_tensor['dims']} vs {donor_tensor['dims']}"
                )
            source_f, source_data, source_tensor = donor_f, donor_data, donor_tensor
            selected.append((base_tensor["name"], base_tensor["type"], donor_tensor["type"]))
        plan.append({
            "name": base_tensor["name"],
            "dims": base_tensor["dims"],
            "type": source_tensor["type"],
            "source_f": source_f,
            "source_offset": source_data + source_tensor["off"],
        })

    if not selected:
        raise SystemExit("--select matched no tensors")

    temp = args.out + ".partial"
    with open(temp, "wb") as out:
        out.write(b"GGUF")
        out.write(struct.pack("<I", version))
        out.write(struct.pack("<Q", len(plan)))
        out.write(struct.pack("<Q", n_kv))
        out.write(kv_blob)

        offset = 0
        entries = []
        for tensor in plan:
            size = nbytes(tensor["dims"], tensor["type"])
            entries.append((tensor, offset, size))
            offset += size + (-size % align)

        for tensor, tensor_offset, _size in entries:
            out.write(pack_str(tensor["name"]))
            out.write(struct.pack("<I", len(tensor["dims"])))
            for dim in tensor["dims"]:
                out.write(struct.pack("<Q", int(dim)))
            out.write(struct.pack("<I", tensor["type"]))
            out.write(struct.pack("<Q", tensor_offset))

        out.write(b"\0" * (-out.tell() % align))
        data_start = out.tell()
        for index, (tensor, tensor_offset, size) in enumerate(entries):
            if out.tell() - data_start != tensor_offset:
                raise SystemExit(f"offset drift at {tensor['name']}")
            copy_bytes(tensor["source_f"], out, tensor["source_offset"], size)
            out.write(b"\0" * (-size % align))
            if index % 50 == 0:
                print(f"{index}/{len(entries)}", flush=True)

    os.replace(temp, args.out)
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e9:.2f} GB)")
    print(f"selected {len(selected)} tensors")
    for name, old_type, new_type in selected[:8]:
        print(f"  {name}: type {old_type} -> {new_type}")


if __name__ == "__main__":
    main()
