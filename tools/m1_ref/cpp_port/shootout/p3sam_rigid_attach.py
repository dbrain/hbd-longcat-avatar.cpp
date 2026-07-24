#!/usr/bin/env python3
"""Rigidly attach explicitly selected P3-SAM face regions to one rig bone.

This is a conservative post-transfer policy for secondary geometry.  It does
not call a region "hair" and it does not remove geometry: the caller supplies
the P3-SAM labels and the target bone.  The reference P3-SAM mesh may have a
different tessellation from the rigged delivery mesh; classification therefore
uses nearest tail-vs-body face centroids after a documented bbox alignment.

It is suitable for an attachment which should remain stable at its anchor in a
baseline animation.  Sway requires a separately validated secondary chain.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


COMPONENT = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def read_glb(path: Path) -> tuple[dict, bytearray]:
    raw = bytearray(path.read_bytes())
    magic, version, length = struct.unpack_from("<III", raw, 0)
    if magic != 0x46546C67 or version != 2 or length != len(raw):
        raise ValueError(f"not a valid GLB v2: {path}")
    offset = 12
    json_len, json_type = struct.unpack_from("<II", raw, offset); offset += 8
    if json_type != 0x4E4F534A:
        raise ValueError("missing JSON chunk")
    doc = json.loads(raw[offset:offset + json_len]); offset += json_len
    bin_len, bin_type = struct.unpack_from("<II", raw, offset); offset += 8
    if bin_type != 0x004E4942:
        raise ValueError("missing BIN chunk")
    return doc, raw


def accessor_view(doc: dict, raw: bytearray, index: int) -> np.ndarray:
    acc = doc["accessors"][index]
    view = doc["bufferViews"][acc["bufferView"]]
    dtype = COMPONENT[acc["componentType"]]
    cols = WIDTH[acc["type"]]
    packed = np.dtype(dtype).itemsize * cols
    if view.get("byteStride", packed) != packed:
        raise ValueError("interleaved GLB accessor is unsupported")
    # GLB header + JSON header/chunk are not needed: bufferView byte offsets are
    # relative to the BIN chunk, so find it from the file chunks.
    offset = 12 + 8 + len(json.dumps(doc, separators=(",", ":")).encode("utf-8"))
    # The JSON chunk uses 4-byte padding and may retain whitespace.  Locate BIN
    # by parsing the actual chunk headers rather than relying on a re-serialize.
    json_len = struct.unpack_from("<I", raw, 12)[0]
    bin_start = 12 + 8 + json_len + 8
    start = bin_start + view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    return np.ndarray((acc["count"], cols), dtype=dtype, buffer=raw, offset=start)


def one_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False)
    if not isinstance(loaded, trimesh.Scene) or len(loaded.geometry) != 1:
        raise ValueError(f"expected one mesh geometry in {path}")
    return next(iter(loaded.geometry.values()))


def normalized(points: np.ndarray, basis_vertices: np.ndarray) -> np.ndarray:
    lo, hi = basis_vertices.min(axis=0), basis_vertices.max(axis=0)
    span = hi - lo
    if np.any(span <= 1e-8):
        raise ValueError("degenerate reference/target bbox")
    return (points - lo) / span


def global_node_positions(doc: dict) -> np.ndarray:
    """Compose node transforms; the writer normally uses translations only."""
    nodes = doc["nodes"]
    parent = np.full(len(nodes), -1, dtype=np.int64)
    local = np.repeat(np.eye(4, dtype=np.float32)[None], len(nodes), axis=0)
    for i, node in enumerate(nodes):
        for child in node.get("children", []):
            parent[child] = i
        if "matrix" in node:
            local[i] = np.asarray(node["matrix"], dtype=np.float32).reshape(4, 4).T
        else:
            local[i, :3, 3] = node.get("translation", [0.0, 0.0, 0.0])
    global_ = np.zeros_like(local)
    pending = set(range(len(nodes)))
    while pending:
        progressed = False
        for i in list(pending):
            p = parent[i]
            if p < 0 or p not in pending:
                global_[i] = local[i] if p < 0 else global_[p] @ local[i]
                pending.remove(i)
                progressed = True
        if not progressed:
            raise ValueError("cyclic glTF node hierarchy")
    return global_[:, :3, 3]


def selected_boundary_vertices(faces: np.ndarray, selected: np.ndarray) -> np.ndarray:
    """Vertices on a selected/non-selected P3-SAM label boundary."""
    edges: dict[tuple[int, int], list[bool]] = {}
    for face, keep in zip(faces, selected, strict=True):
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = (int(a), int(b)) if a < b else (int(b), int(a))
            edges.setdefault(key, []).append(bool(keep))
    boundary = []
    for edge, sides in edges.items():
        if any(sides) and not all(sides):
            boundary.extend(edge)
    if not boundary:
        raise ValueError("selected labels have no boundary with the remaining P3-SAM mesh")
    return np.unique(np.asarray(boundary, dtype=np.int64))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rigged_glb", type=Path)
    parser.add_argument("p3sam_mesh", type=Path)
    parser.add_argument("face_ids", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--labels", nargs="+", required=True, type=np.int64)
    parser.add_argument("--bone", default="auto", help="exact glTF bone node name, or 'auto' (default)")
    parser.add_argument("--margin", type=float, default=0.90,
                        help="require tail distance < margin * body distance (default 0.90)")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    if not 0.0 < args.margin < 1.0:
        raise SystemExit("--margin must be in (0,1)")

    reference = one_mesh(args.p3sam_mesh)
    labels = np.load(args.face_ids)
    if labels.shape != (len(reference.faces),):
        raise SystemExit(f"face-label shape {labels.shape} does not match {len(reference.faces)} faces")
    selected = np.isin(labels, np.asarray(args.labels, dtype=np.int64))
    if not selected.any() or selected.all():
        raise SystemExit("selected labels must cover a non-empty proper subset of reference faces")
    centroids = np.asarray(reference.triangles_center, dtype=np.float32)
    tail_tree = cKDTree(normalized(centroids[selected], np.asarray(reference.vertices)))
    body_tree = cKDTree(normalized(centroids[~selected], np.asarray(reference.vertices)))

    doc, raw = read_glb(args.rigged_glb)
    mesh_node = next((node for node in doc["nodes"] if "mesh" in node and "skin" in node), None)
    if mesh_node is None:
        raise SystemExit("rigged GLB has no skinned mesh node")
    primitive = doc["meshes"][mesh_node["mesh"]]["primitives"][0]
    attrs = primitive["attributes"]
    if not {"POSITION", "JOINTS_0", "WEIGHTS_0"}.issubset(attrs):
        raise SystemExit("rigged GLB lacks POSITION/JOINTS_0/WEIGHTS_0")
    positions = accessor_view(doc, raw, attrs["POSITION"]).astype(np.float32, copy=False)
    joint_rows = accessor_view(doc, raw, attrs["JOINTS_0"])
    weight_rows = accessor_view(doc, raw, attrs["WEIGHTS_0"])
    skin = doc["skins"][mesh_node["skin"]]
    bone_name = args.bone
    if bone_name == "auto":
        boundary_ids = selected_boundary_vertices(np.asarray(reference.faces), selected)
        boundary = normalized(np.asarray(reference.vertices)[boundary_ids], np.asarray(reference.vertices))
        target_lo, target_hi = positions.min(axis=0), positions.max(axis=0)
        boundary_target = boundary * (target_hi - target_lo) + target_lo
        node_positions = global_node_positions(doc)
        skinned_nodes = np.asarray(skin["joints"], dtype=np.int64)
        _, near = cKDTree(node_positions[skinned_nodes]).query(boundary_target, workers=-1)
        counts = np.bincount(near, minlength=len(skinned_nodes))
        bone_index = int(np.argmax(counts))
        node_index = int(skinned_nodes[bone_index])
        bone_name = str(doc["nodes"][node_index].get("name", f"node_{node_index}"))
        anchor_method = f"auto boundary vote ({int(counts[bone_index])}/{len(boundary_target)})"
    else:
        try:
            node_index = next(i for i, node in enumerate(doc["nodes"]) if node.get("name") == bone_name)
            bone_index = skin["joints"].index(node_index)
        except StopIteration as exc:
            raise SystemExit(f"bone {bone_name!r} is not present") from exc
        except ValueError as exc:
            raise SystemExit(f"bone {bone_name!r} is not in this skin") from exc
        anchor_method = "explicit"

    target_mesh = trimesh.Trimesh(vertices=positions, faces=np.empty((0, 3), dtype=np.int64), process=False)
    query = normalized(np.asarray(target_mesh.vertices), np.asarray(target_mesh.vertices))
    tail_dist, _ = tail_tree.query(query, workers=-1)
    body_dist, _ = body_tree.query(query, workers=-1)
    attached = tail_dist < args.margin * body_dist
    if not attached.any():
        raise SystemExit("no delivery vertices classified as selected attachment")
    joint_rows[attached] = 0
    joint_rows[attached, 0] = bone_index
    weight_rows[attached] = 0.0
    weight_rows[attached, 0] = 1.0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(raw)
    report = args.out.with_suffix(args.out.suffix + ".attachment-report.txt")
    report.write_text(
        f"reference={args.p3sam_mesh}\nface_ids={args.face_ids}\nlabels={list(map(int, args.labels))}\n"
        f"bone={bone_name}\nbone_index={bone_index}\nanchor_method={anchor_method}\nmargin={args.margin}\n"
        f"reference_faces={len(labels)}\nreference_selected_faces={int(selected.sum())}\n"
        f"delivery_vertices={len(positions)}\ndelivery_rigidly_attached={int(attached.sum())}\n",
        encoding="utf-8",
    )
    print(f"[p3sam-attach] selected faces={int(selected.sum())}/{len(labels)}; "
          f"rigid {int(attached.sum())}/{len(positions)} delivery vertices to {bone_name} ({anchor_method}) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
