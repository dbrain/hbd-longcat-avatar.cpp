#!/usr/bin/env python3
"""Conservatively repair a malformed generic skin field on separate mesh pieces.

SkinTokens occasionally assigns one disconnected compact piece (hair shell,
halo, ornament, etc.) to joints on *different skeleton branches*.  That is not
ordinary flexible skinning: a real limb/wing or carried prop may span a chain
of nearby bones, but cannot remain coherent when it is simultaneously bound to
two unrelated limbs.  The result is the familiar exploding-hair/attachment
failure even when every individual mesh component has low internal stretch.

For a bounded generic fallback, identify only those branch-spanning pieces and
rigidly attach each to the geometrically nearest existing joint.  The source
GLB is immutable, every repair is written to a separate GLB, and the caller
must still run the ordinary whole-tree LBS pose gate on that exact output.

This intentionally does not invent semantic names ("head", "hair", "wing",
etc.) and never replaces a normal chain-local multi-bone skin field.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

from rig_pose_smoke import (
    deform,
    edge_component_ids,
    global_transforms,
    mesh_and_skin,
    rotation_z,
    skin_parent_indices,
)


COMPONENT = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


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
    columns = WIDTH[acc["type"]]
    packed = np.dtype(dtype).itemsize * columns
    stride = view.get("byteStride", packed)
    if stride != packed:
        raise ValueError("interleaved rig accessor unsupported")
    offset = bin_offset + view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    return np.ndarray((acc["count"], columns), dtype=dtype, buffer=raw, offset=offset)


def tree_path(parent: np.ndarray, first: int, second: int) -> list[int]:
    """Return the undirected skeleton path between two joint indices."""
    first_path = []
    node = first
    while node >= 0:
        first_path.append(node)
        node = int(parent[node])
    first_index = {node: i for i, node in enumerate(first_path)}
    second_path = []
    node = second
    while node not in first_index:
        second_path.append(node)
        node = int(parent[node])
        if node < 0:
            raise ValueError("skin joints are not a rooted tree")
    return first_path[:first_index[node] + 1] + list(reversed(second_path))


def support_topology(parent: np.ndarray, active: list[int]) -> tuple[int, bool]:
    """Return support-tree diameter and whether its minimal subtree branches."""
    if len(active) < 2:
        return 0, False
    edges: set[tuple[int, int]] = set()
    diameter = 0
    for index, first in enumerate(active):
        for second in active[index + 1:]:
            path = tree_path(parent, first, second)
            diameter = max(diameter, len(path) - 1)
            edges.update(tuple(sorted((left, right))) for left, right in zip(path, path[1:]))
    degree: dict[int, int] = {}
    for left, right in edges:
        degree[left] = degree.get(left, 0) + 1
        degree[right] = degree.get(right, 0) + 1
    return diameter, any(value >= 3 for value in degree.values())


def repair_candidates(vertices, faces, joints, weights, skin_nodes, local, doc,
                      min_faces: int, max_component_face_fraction: float,
                      active_fraction: float, min_span: int):
    components = edge_component_ids(vertices, faces)
    parents = skin_parent_indices(doc, skin_nodes)
    joint_positions = global_transforms(doc, local)[skin_nodes, :3, 3]
    face_limit = int(len(faces) * max_component_face_fraction)
    candidates = []
    for component in np.unique(components):
        selected_faces = components == component
        face_count = int(selected_faces.sum())
        if face_count < min_faces or face_count > face_limit:
            continue
        selected_vertices = np.unique(faces[selected_faces])
        mass = np.zeros(len(skin_nodes), dtype=np.float64)
        for slot in range(4):
            np.add.at(mass, joints[selected_vertices, slot], weights[selected_vertices, slot])
        total_mass = float(mass.sum())
        if total_mass <= 0:
            continue
        active = np.flatnonzero(mass >= total_mass * active_fraction).astype(int).tolist()
        span, branched = support_topology(parents, active)
        # A compact piece flexing across a *single* bone chain is normal skinning
        # (sleeve, feather, blade, tail section). Branch-spanning support is the
        # malformed case: it ties the same surface to incompatible limbs.
        if not branched or span < min_span:
            continue
        centre = vertices[selected_vertices].mean(axis=0)
        distances = np.linalg.norm(joint_positions - centre, axis=1)
        anchor = int(np.argmin(distances))
        candidates.append({
            "component": int(component), "face_count": face_count,
            "vertex_count": int(len(selected_vertices)), "face_fraction": face_count / len(faces),
            "active_joints": active, "support_span": span,
            "anchor_joint": anchor, "anchor_distance": float(distances[anchor]),
            "repair_kind": "branch-spanning-support", "vertices": selected_vertices,
            "face_indices": np.flatnonzero(selected_faces),
        })
    return candidates


def apply_rigid(candidates, joints, weights):
    """Return an in-memory skin field with each candidate rigidly anchored."""
    repaired_joints = joints.copy()
    repaired_weights = weights.copy()
    for candidate in candidates:
        selected = candidate["vertices"]
        repaired_joints[selected] = 0
        repaired_joints[selected, 0] = candidate["anchor_joint"]
        repaired_weights[selected] = 0.0
        repaired_weights[selected, 0] = 1.0
    return repaired_joints, repaired_weights


def local_pose_candidates(vertices, faces, joints, weights, skin_nodes, ibm, local, doc,
                          excluded_components: set[int], min_faces: int,
                          max_component_face_fraction: float, active_fraction: float,
                          pose_limit: float = 6.0):
    """Find a tiny detached root/joint blend that *still* fails the real pose test.

    This is deliberately a second, narrower class than branch repair.  A
    detached surface with a root-plus-one-joint blend can be a valid flexible
    detail, so topology alone is insufficient reason to change it.  It is
    considered only after the branch repair has been applied in memory, when
    rotating its sole material non-root joint demonstrably exceeds the same
    per-component LBS limit used by the publication gate.  Requiring that the
    sole joint is also the geometric nearest joint makes the change a rigid
    attachment to the place the model already chose, not a semantic guess.
    """
    components = edge_component_ids(vertices, faces)
    parents = skin_parent_indices(doc, skin_nodes)
    roots = {joint_i for joint_i, parent in enumerate(parents) if parent < 0}
    joint_positions = global_transforms(doc, local)[skin_nodes, :3, 3]
    face_limit = int(len(faces) * max_component_face_fraction)

    # Match the generic all-influential gate: only joints with material global
    # support are audited, and use real GLB LBS/IBMs rather than a rest-tree
    # proxy.  The geometry is unchanged by the first repair, so the welded
    # component IDs remain stable for the output write below.
    total_mass = np.zeros(len(skin_nodes), dtype=np.float64)
    for slot in range(4):
        np.add.at(total_mass, joints[:, slot], weights[:, slot])
    non_roots = [joint_i for joint_i in range(len(skin_nodes)) if joint_i not in roots]
    peak_mass = max((float(total_mass[joint_i]) for joint_i in non_roots), default=0.0)
    if peak_mass <= 0:
        return []
    rest = global_transforms(doc, local)
    rest_vertices = deform(vertices, joints, weights, skin_nodes, ibm, rest)
    edge_pairs = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    mesh_diag = float(np.linalg.norm(rest_vertices.max(axis=0) - rest_vertices.min(axis=0)))
    rest_length = np.linalg.norm(rest_vertices[edge_pairs[:, 0]] - rest_vertices[edge_pairs[:, 1]], axis=1)
    valid_edges = rest_length >= mesh_diag * 1e-4
    edge_components = np.concatenate([components, components, components])[valid_edges]
    candidates = []
    for component in np.unique(components):
        if int(component) in excluded_components:
            continue
        selected_faces = components == component
        face_count = int(selected_faces.sum())
        if face_count < min_faces or face_count > face_limit:
            continue
        selected_vertices = np.unique(faces[selected_faces])
        mass = np.zeros(len(skin_nodes), dtype=np.float64)
        for slot in range(4):
            np.add.at(mass, joints[selected_vertices, slot], weights[selected_vertices, slot])
        support_mass = float(mass.sum())
        if support_mass <= 0:
            continue
        active = np.flatnonzero(mass >= support_mass * active_fraction).astype(int).tolist()
        material_non_roots = [joint_i for joint_i in active if joint_i not in roots]
        if len(active) != 2 or len(material_non_roots) != 1 or not any(joint_i in roots for joint_i in active):
            continue
        target = material_non_roots[0]
        if total_mass[target] < peak_mass * 0.08:
            continue
        centre = vertices[selected_vertices].mean(axis=0)
        distances = np.linalg.norm(joint_positions - centre, axis=1)
        anchor = int(np.argmin(distances))
        if anchor != target:
            continue
        local_pose = local.copy()
        local_pose[int(skin_nodes[target])] = local_pose[int(skin_nodes[target])] @ rotation_z(45)
        posed_vertices = deform(vertices, joints, weights, skin_nodes, ibm, global_transforms(doc, local_pose))
        posed_length = np.linalg.norm(posed_vertices[edge_pairs[:, 0]] - posed_vertices[edge_pairs[:, 1]], axis=1)
        stretch = posed_length[valid_edges] / np.maximum(rest_length[valid_edges], 1e-9)
        component_edges = edge_components == component
        if int(component_edges.sum()) < 30:
            continue
        component_p999 = float(np.quantile(stretch[component_edges], 0.999))
        if component_p999 <= pose_limit:
            continue
        candidates.append({
            "component": int(component), "face_count": face_count,
            "vertex_count": int(len(selected_vertices)), "face_fraction": face_count / len(faces),
            "active_joints": active, "support_span": 1, "anchor_joint": anchor,
            "anchor_distance": float(distances[anchor]), "trigger_joint": target,
            "trigger_component_p999": component_p999, "repair_kind": "tiny-detached-pose-failure",
            "vertices": selected_vertices, "face_indices": np.flatnonzero(selected_faces),
        })
    return candidates


def raw_topology_component_ids(faces: np.ndarray, vertex_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Label authored index-connected pieces without position welding.

    The publication gate deliberately welds coincident positions, because UV
    chart splits must not turn a normal textured character into hundreds of
    false components.  A generated asset can also contain separately indexed
    pieces which merely touch at a point.  This helper preserves that authored
    separation only for the narrowly evidenced pose-spike fallback below.
    """
    parent = np.arange(vertex_count, dtype=np.int64)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    for triangle in faces:
        root = find(int(triangle[0]))
        for vertex in triangle[1:]:
            other = find(int(vertex))
            if other != root:
                parent[other] = root
    vertex_components = np.asarray([find(index) for index in range(vertex_count)], dtype=np.int64)
    return vertex_components[faces[:, 0]], vertex_components


def raw_topology_pose_spike_candidates(vertices, faces, joints, weights, skin_nodes, ibm, local, doc,
                                       min_faces: int, max_component_face_fraction: float,
                                       pose_limit: float = 6.0):
    """Rigidly attach only small authored pieces proven to shred under LBS.

    This is intentionally *after* normal branch/local repair.  It does not
    treat separate raw topology as defective: a piece is eligible only when an
    actual all-influential 45-degree pose has a per-piece p999 stretch above
    the publication limit and contains a >limit edge.  The nearest existing
    rest joint is geometric provenance, not a guessed anatomical label.
    """
    face_components, vertex_components = raw_topology_component_ids(faces, len(vertices))
    parents = skin_parent_indices(doc, skin_nodes)
    roots = {joint_i for joint_i, parent in enumerate(parents) if parent < 0}
    total_mass = np.zeros(len(skin_nodes), dtype=np.float64)
    for slot in range(4):
        np.add.at(total_mass, joints[:, slot], weights[:, slot])
    non_roots = [joint_i for joint_i in range(len(skin_nodes)) if joint_i not in roots]
    peak_mass = max((float(total_mass[joint_i]) for joint_i in non_roots), default=0.0)
    if peak_mass <= 0:
        return []
    audit_joints = [joint_i for joint_i in non_roots if total_mass[joint_i] >= peak_mass * 0.08]
    if not audit_joints:
        return []
    rest = global_transforms(doc, local)
    rest_vertices = deform(vertices, joints, weights, skin_nodes, ibm, rest)
    edge_pairs = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    mesh_diag = float(np.linalg.norm(rest_vertices.max(axis=0) - rest_vertices.min(axis=0)))
    rest_length = np.linalg.norm(rest_vertices[edge_pairs[:, 0]] - rest_vertices[edge_pairs[:, 1]], axis=1)
    valid_edges = rest_length >= mesh_diag * 1e-4
    raw_edge_components = np.concatenate([face_components, face_components, face_components])[valid_edges]
    face_limit = int(len(faces) * max_component_face_fraction)
    joint_positions = rest[skin_nodes, :3, 3]
    candidates: dict[int, dict] = {}
    for target in audit_joints:
        posed_local = local.copy()
        posed_local[int(skin_nodes[target])] = posed_local[int(skin_nodes[target])] @ rotation_z(45)
        posed_vertices = deform(vertices, joints, weights, skin_nodes, ibm, global_transforms(doc, posed_local))
        posed_length = np.linalg.norm(posed_vertices[edge_pairs[:, 0]] - posed_vertices[edge_pairs[:, 1]], axis=1)
        stretch = posed_length[valid_edges] / np.maximum(rest_length[valid_edges], 1e-9)
        face_stretch = np.zeros((3, len(faces)), dtype=np.float64)
        face_stretch.reshape(-1)[valid_edges] = stretch
        bad_faces = np.any(face_stretch > pose_limit, axis=0)
        for component in np.unique(face_components[bad_faces]):
            selected_faces = face_components == component
            face_count = int(selected_faces.sum())
            if face_count < min_faces or face_count > face_limit:
                continue
            component_edges = raw_edge_components == component
            if int(component_edges.sum()) < 30:
                continue
            component_p999 = float(np.quantile(stretch[component_edges], 0.999))
            if component_p999 <= pose_limit:
                continue
            selected_vertices = np.flatnonzero(vertex_components == component)
            centre = vertices[selected_vertices].mean(axis=0)
            distances = np.linalg.norm(joint_positions - centre, axis=1)
            candidate = {
                "component": int(component), "face_count": face_count,
                "vertex_count": int(len(selected_vertices)), "face_fraction": face_count / len(faces),
                "anchor_joint": int(np.argmin(distances)),
                "anchor_distance": float(distances.min()), "trigger_joint": int(target),
                "trigger_bad_faces": int(np.sum(selected_faces & bad_faces)),
                "trigger_component_p999": component_p999,
                "repair_kind": "raw-topology-pose-spike",
                "vertices": selected_vertices, "face_indices": np.flatnonzero(selected_faces),
            }
            previous = candidates.get(int(component))
            if previous is None or component_p999 > previous["trigger_component_p999"]:
                candidates[int(component)] = candidate
    return list(candidates.values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, help="new repaired GLB; required unless --dry-run")
    parser.add_argument("--report", type=Path, required=True, help="new JSON provenance report")
    parser.add_argument("--dry-run", action="store_true", help="report candidates without writing a GLB")
    parser.add_argument("--min-faces", type=int, default=20)
    parser.add_argument("--max-component-face-fraction", type=float, default=0.15,
                        help="only consider one disconnected piece up to this share of all faces")
    parser.add_argument("--max-total-face-fraction", type=float, default=0.40,
                        help="refuse repair if all selected pieces exceed this share of all faces")
    parser.add_argument("--active-fraction", type=float, default=0.05)
    parser.add_argument("--min-span", type=int, default=5)
    parser.add_argument("--max-local-component-face-fraction", type=float, default=0.01,
                        help="after branch repair, inspect only tiny detached pose-failing pieces up to this share")
    parser.add_argument("--max-raw-topology-component-face-fraction", type=float, default=0.10,
                        help="after ordinary repair, inspect only bounded raw-index pieces proven to pose-spike")
    args = parser.parse_args()
    if not args.input.is_file():
        raise SystemExit(f"missing input: {args.input}")
    if args.output.exists() or args.report.exists():
        raise SystemExit("refusing to overwrite output or report")
    if (args.min_faces < 1 or not 0 < args.max_component_face_fraction <= 1
            or not 0 < args.max_total_face_fraction <= 1
            or not 0 < args.active_fraction < 1 or args.min_span < 1
            or not 0 < args.max_local_component_face_fraction <= 1
            or not 0 < args.max_raw_topology_component_face_fraction <= 1):
        raise SystemExit("invalid repair thresholds")

    doc, vertices, faces, joints, weights, skin_nodes, ibm, local = mesh_and_skin(str(args.input))
    branch_candidates = repair_candidates(vertices, faces, joints, weights, skin_nodes, local, doc,
                                          args.min_faces, args.max_component_face_fraction,
                                          args.active_fraction, args.min_span)
    provisional_joints, provisional_weights = apply_rigid(branch_candidates, joints, weights)
    local_candidates = local_pose_candidates(
        vertices, faces, provisional_joints, provisional_weights, skin_nodes, ibm, local, doc,
        {candidate["component"] for candidate in branch_candidates}, args.min_faces,
        args.max_local_component_face_fraction, args.active_fraction,
    )
    provisional_joints, provisional_weights = apply_rigid(local_candidates, provisional_joints, provisional_weights)
    raw_topology_candidates = raw_topology_pose_spike_candidates(
        vertices, faces, provisional_joints, provisional_weights, skin_nodes, ibm, local, doc,
        args.min_faces, args.max_raw_topology_component_face_fraction,
    )
    candidates = branch_candidates + local_candidates + raw_topology_candidates
    repaired_face_indices = np.unique(np.concatenate(
        [candidate["face_indices"] for candidate in candidates], dtype=np.int64,
    )) if candidates else np.empty(0, dtype=np.int64)
    repaired_faces = int(len(repaired_face_indices))
    repaired_fraction = repaired_faces / max(1, len(faces))
    if repaired_fraction > args.max_total_face_fraction:
        raise SystemExit(f"refusing broad component repair: {repaired_fraction:.3f} > {args.max_total_face_fraction:.3f}")
    report = {
        "schema_version": 2,
        "input": str(args.input.resolve()),
        "mode": "dry-run" if args.dry_run else "rigid-branch-repair",
        "thresholds": {
            "min_faces": args.min_faces,
            "max_component_face_fraction": args.max_component_face_fraction,
            "max_total_face_fraction": args.max_total_face_fraction,
            "active_fraction": args.active_fraction, "min_span": args.min_span,
            "max_local_component_face_fraction": args.max_local_component_face_fraction,
            "max_raw_topology_component_face_fraction": args.max_raw_topology_component_face_fraction,
        },
        "mesh_faces": int(len(faces)), "repaired_faces": repaired_faces,
        "repaired_face_fraction": repaired_fraction,
        "repairs": [{key: value for key, value in candidate.items()
                     if key not in {"vertices", "face_indices"}} for candidate in candidates],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, sort_keys=True))
        return 0

    json_doc, raw, bin_offset = glb_doc_and_bin_offset(args.input)
    node = next((node for node in json_doc["nodes"] if "mesh" in node and "skin" in node), None)
    if node is None:
        raise SystemExit("no skinned mesh node")
    primitives = json_doc["meshes"][node["mesh"]].get("primitives", [])
    if len(primitives) != 1:
        raise SystemExit("component repair requires exactly one output primitive")
    attrs = primitives[0].get("attributes", {})
    if "JOINTS_0" not in attrs or "WEIGHTS_0" not in attrs:
        raise SystemExit("skinned primitive lacks JOINTS_0/WEIGHTS_0")
    mutable_joints = writable_accessor(json_doc, raw, bin_offset, attrs["JOINTS_0"])
    mutable_weights = writable_accessor(json_doc, raw, bin_offset, attrs["WEIGHTS_0"])
    if len(mutable_joints) != len(vertices) or mutable_weights.shape != mutable_joints.shape:
        raise SystemExit("skin accessors do not match output mesh vertices")
    repaired_joints, repaired_weights = apply_rigid(candidates, mutable_joints, mutable_weights)
    mutable_joints[:] = repaired_joints
    mutable_weights[:] = repaired_weights
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(raw)
    report["output"] = str(args.output.resolve())
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"component branch repair: pieces={len(candidates)} faces={repaired_faces}/{len(faces)} -> {args.output}")
    for candidate in report["repairs"]:
        print("  kind={repair_kind} component={component} faces={face_count} anchor={anchor_joint} distance={anchor_distance:.5f}".format(**candidate))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
