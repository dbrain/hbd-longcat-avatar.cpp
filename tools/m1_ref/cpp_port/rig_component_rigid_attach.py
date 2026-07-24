#!/usr/bin/env python3
"""Diagnostic rigid binding for one position-welded disconnected GLB component.

Used to evaluate a component-aware prop repair after the ordinary learned
transfer has failed its real LBS gate.  It never chooses a component or joint
itself; callers must supply both and still run the complete pose gate.
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
from pathlib import Path

import numpy as np

from rig_pose_smoke import edge_component_ids, mesh_and_skin


COMPONENT = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}


def glb_doc_and_bin_offset(path: Path):
    raw = bytearray(path.read_bytes())
    magic, version, length = struct.unpack_from("<III", raw, 0)
    if magic != 0x46546C67 or version != 2 or length != len(raw):
        raise ValueError("not a GLB v2")
    json_len, json_type = struct.unpack_from("<II", raw, 12)
    if json_type != 0x4E4F534A:
        raise ValueError("missing JSON chunk")
    doc = json.loads(raw[20:20 + json_len])
    bin_header = 20 + json_len
    bin_len, bin_type = struct.unpack_from("<II", raw, bin_header)
    if bin_type != 0x004E4942 or bin_header + 8 + bin_len != len(raw):
        raise ValueError("missing BIN chunk")
    return doc, raw, bin_header + 8


def writable_accessor(doc, raw, bin_offset, index):
    acc = doc["accessors"][index]
    view = doc["bufferViews"][acc["bufferView"]]
    dtype = COMPONENT[acc["componentType"]]
    columns = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[acc["type"]]
    packed = np.dtype(dtype).itemsize * columns
    stride = view.get("byteStride", packed)
    if stride != packed:
        raise ValueError("interleaved rig accessor unsupported")
    offset = bin_offset + view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    return np.ndarray((acc["count"], columns), dtype=dtype, buffer=raw, offset=offset)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--component", type=int, required=True, help="welded component root ID from the pose diagnostic")
    parser.add_argument("--joint", type=int, required=True, help="existing skin joint index")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite: {args.output}")
    doc, vertices, faces, _, _, skin_nodes, _, _ = mesh_and_skin(str(args.input))
    components = edge_component_ids(vertices, faces)
    selected_faces = components == args.component
    if not selected_faces.any():
        raise SystemExit(f"no faces in component {args.component}")
    selected_vertices = np.unique(faces[selected_faces])
    if args.joint < 0 or args.joint >= len(skin_nodes):
        raise SystemExit(f"joint {args.joint} out of range [0,{len(skin_nodes)})")
    json_doc, raw, bin_offset = glb_doc_and_bin_offset(args.input)
    node = next((n for n in json_doc["nodes"] if "mesh" in n and "skin" in n), None)
    if node is None:
        raise SystemExit("no skinned mesh node")
    primitives = json_doc["meshes"][node["mesh"]].get("primitives", [])
    if len(primitives) != 1:
        raise SystemExit("diagnostic currently requires one output primitive")
    attrs = primitives[0]["attributes"]
    joints = writable_accessor(json_doc, raw, bin_offset, attrs["JOINTS_0"])
    weights = writable_accessor(json_doc, raw, bin_offset, attrs["WEIGHTS_0"])
    if len(joints) != len(vertices) or weights.shape != joints.shape:
        raise SystemExit("skin accessors do not match mesh vertices")
    joints[selected_vertices] = 0
    joints[selected_vertices, 0] = args.joint
    weights[selected_vertices] = 0.0
    weights[selected_vertices, 0] = 1.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(raw)
    print(f"rigid-attach: component={args.component} faces={int(selected_faces.sum())} vertices={len(selected_vertices)} joint={args.joint} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
