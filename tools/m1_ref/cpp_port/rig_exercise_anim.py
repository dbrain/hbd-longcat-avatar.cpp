#!/usr/bin/env python3
"""Bake a "skeleton exercise" animation clip into a rigged GLB.

A still pose gate proves one joint moves. This walks every materially weighted
joint in turn so the whole rig can be eye-tested in motion in any glTF viewer:
each influential joint swings about an axis perpendicular to its own bone
direction, overlapping slightly into the next, producing a wave down the rig.

Bad weights are obvious here in a way they are not in a still: a joint that
drags unrelated geometry, an appendage nothing moves, a limb that rubber-bands.

usage: rig_exercise_anim.py <rigged.glb> <out.glb> [--degrees 25] [--slot 1.2]
                            [--stride 0.6] [--max-joints 40]
"""

from __future__ import annotations

import importlib.util
import json
import os
import struct
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("rig_pose_smoke",
                                               os.path.join(_HERE, "rig_pose_smoke.py"))
_ps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ps)


def read_glb(path: str):
    raw = open(path, "rb").read()
    magic, version, length = struct.unpack_from("<III", raw, 0)
    if magic != 0x46546C67 or version != 2 or length != len(raw):
        raise ValueError("not a GLB v2")
    off = 12
    jlen, jtype = struct.unpack_from("<II", raw, off); off += 8
    if jtype != 0x4E4F534A:
        raise ValueError("missing JSON chunk")
    doc = json.loads(raw[off:off + jlen]); off += jlen
    blob = b""
    if off < len(raw):
        blen, btype = struct.unpack_from("<II", raw, off); off += 8
        if btype != 0x004E4942:
            raise ValueError("second chunk is not BIN")
        blob = raw[off:off + blen]
    return doc, bytearray(blob)


def write_glb(path: str, doc: dict, blob: bytearray) -> None:
    while len(blob) % 4:
        blob.append(0)
    js = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    while len(js) % 4:
        js += b" "
    total = 12 + 8 + len(js) + 8 + len(blob)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(js), 0x4E4F534A)); f.write(js)
        f.write(struct.pack("<II", len(blob), 0x004E4942)); f.write(blob)


def add_accessor(doc, blob, array, comp_type, type_name, with_minmax=False):
    while len(blob) % 4:
        blob.append(0)
    offset = len(blob)
    data = np.ascontiguousarray(array, dtype=np.float32)
    blob.extend(data.tobytes())
    doc.setdefault("bufferViews", []).append(
        {"buffer": 0, "byteOffset": offset, "byteLength": data.nbytes})
    acc = {"bufferView": len(doc["bufferViews"]) - 1, "componentType": comp_type,
           "count": int(data.shape[0]), "type": type_name}
    if with_minmax:
        acc["min"] = [float(data.min())]
        acc["max"] = [float(data.max())]
    doc.setdefault("accessors", []).append(acc)
    return len(doc["accessors"]) - 1


def quat_from_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(axis)
    if n < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = axis / n
    s = np.sin(angle * 0.5)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(angle * 0.5)])


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])


def main() -> int:
    args = sys.argv[1:]
    if len(args) < 2:
        print(__doc__)
        return 2
    src, dst = args[0], args[1]
    degrees, slot, stride, max_joints = 25.0, 1.2, 0.6, 40
    for i, a in enumerate(args):
        if a == "--degrees":
            degrees = float(args[i + 1])
        elif a == "--slot":
            slot = float(args[i + 1])
        elif a == "--stride":
            stride = float(args[i + 1])
        elif a == "--max-joints":
            max_joints = int(args[i + 1])

    doc, blob = read_glb(src)
    if not doc.get("skins"):
        print(f"{src}: no skin, nothing to exercise", file=sys.stderr)
        return 2

    # Reuse the gate's own notion of "materially weighted" so the clip exercises
    # exactly the joints the pose gate claims to audit.
    _d, vertices, _f, joints, weights, skin_nodes, _ibm, _l = _ps.mesh_and_skin(src)
    mass, centres = _ps.generic_joint_masses_and_centres(vertices, joints, weights, len(skin_nodes))
    mass = np.asarray(mass, dtype=np.float64)
    order = [i for i in np.argsort(-mass) if mass[i] >= 0.08 * mass.max()][:max_joints]

    nodes = doc["nodes"]
    parent = np.full(len(nodes), -1, dtype=np.int64)
    for ni, node in enumerate(nodes):
        for c in node.get("children", []):
            parent[c] = ni
    node_to_joint = {int(n): j for j, n in enumerate(skin_nodes)}
    roots = {j for j, n in enumerate(skin_nodes) if parent[n] not in node_to_joint}
    order = [j for j in order if j not in roots]
    if not order:
        print(f"{src}: no non-root influential joint", file=sys.stderr)
        return 2

    samplers, channels = [], []
    for slot_i, j in enumerate(order):
        node_i = int(skin_nodes[j])
        node = nodes[node_i]
        # Bone direction: mean child offset, else the offset to this joint's own
        # influence centroid. Swing about an axis perpendicular to it so the
        # motion is maximally visible rather than a twist about the bone.
        kids = [nodes[c].get("translation", [0, 0, 0]) for c in node.get("children", [])]
        d = np.mean(np.asarray(kids, dtype=np.float64), axis=0) if kids else np.zeros(3)
        if np.linalg.norm(d) < 1e-6:
            d = np.asarray(centres[j], dtype=np.float64) - np.asarray(
                node.get("translation", [0, 0, 0]), dtype=np.float64)
        if np.linalg.norm(d) < 1e-6:
            d = np.array([0.0, 1.0, 0.0])
        ref = np.array([0.0, 0.0, 1.0]) if abs(d[2] / max(np.linalg.norm(d), 1e-9)) < 0.9 \
            else np.array([1.0, 0.0, 0.0])
        axis = np.cross(d, ref)

        base = np.asarray(node.get("rotation", [0.0, 0.0, 0.0, 1.0]), dtype=np.float64)
        ang = np.radians(degrees)
        t0 = slot_i * stride
        times = np.array([t0, t0 + slot * 0.25, t0 + slot * 0.5, t0 + slot * 0.75, t0 + slot])
        quats = np.stack([
            quat_mul(base, quat_from_axis_angle(axis, 0.0)),
            quat_mul(base, quat_from_axis_angle(axis, ang)),
            quat_mul(base, quat_from_axis_angle(axis, 0.0)),
            quat_mul(base, quat_from_axis_angle(axis, -ang)),
            quat_mul(base, quat_from_axis_angle(axis, 0.0)),
        ])
        t_acc = add_accessor(doc, blob, times.reshape(-1, 1), 5126, "SCALAR", with_minmax=True)
        q_acc = add_accessor(doc, blob, quats, 5126, "VEC4")
        samplers.append({"input": t_acc, "output": q_acc, "interpolation": "LINEAR"})
        channels.append({"sampler": len(samplers) - 1,
                         "target": {"node": node_i, "path": "rotation"}})

    doc.setdefault("animations", []).append(
        {"name": "skeleton-exercise", "samplers": samplers, "channels": channels})
    doc["buffers"][0]["byteLength"] = len(blob)
    write_glb(dst, doc, blob)
    duration = (len(order) - 1) * stride + slot
    print(f"{dst}: exercised {len(order)} influential joints, {duration:.1f}s clip, "
          f"+-{degrees:g} deg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
