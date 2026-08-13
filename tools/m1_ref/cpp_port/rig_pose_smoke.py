#!/usr/bin/env python3
"""Render rest vs. a deterministic articulation LBS pose from a rigged GLB.

This is deliberately a promotion gate, not an exporter: a valid rest skeleton
is insufficient when a secondary attachment has been weighted to an arm. For a
generic creature namespace it normally rotates the most materially weighted
non-root joint. The explicit ``--generic-extremity`` diagnostic instead chooses
a materially influential joint whose influence centroid is geometrically distal
from the mesh centre; it does not claim a semantic wing/tail/arm label.
"""

from __future__ import annotations

import json
import os
import struct
import sys

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import pyrender
import trimesh
from PIL import Image, ImageDraw


COMPONENT = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def read_glb(path: str):
    raw = open(path, "rb").read()
    magic, version, length = struct.unpack_from("<III", raw, 0)
    if magic != 0x46546C67 or version != 2 or length != len(raw):
        raise ValueError("not a GLB v2")
    offset = 12
    json_len, json_type = struct.unpack_from("<II", raw, offset); offset += 8
    if json_type != 0x4E4F534A:
        raise ValueError("missing JSON chunk")
    doc = json.loads(raw[offset:offset + json_len]); offset += json_len
    bin_len, bin_type = struct.unpack_from("<II", raw, offset); offset += 8
    if bin_type != 0x004E4942:
        raise ValueError("missing BIN chunk")
    return doc, memoryview(raw[offset:offset + bin_len])


def accessor(doc, blob, index):
    a = doc["accessors"][index]
    view = doc["bufferViews"][a["bufferView"]]
    dtype = COMPONENT[a["componentType"]]
    cols = WIDTH[a["type"]]
    count = a["count"]
    start = view.get("byteOffset", 0) + a.get("byteOffset", 0)
    stride = view.get("byteStride", np.dtype(dtype).itemsize * cols)
    if stride != np.dtype(dtype).itemsize * cols:
        raise ValueError("interleaved accessor unsupported in smoke renderer")
    return np.frombuffer(blob, dtype=dtype, count=count * cols, offset=start).reshape(count, cols).copy()


def rotation_z(degrees: float) -> np.ndarray:
    a = np.deg2rad(degrees); c, s = np.cos(a), np.sin(a)
    out = np.eye(4, dtype=np.float32)
    out[:2, :2] = [[c, -s], [s, c]]
    return out


def global_transforms(doc, local):
    parent = np.full(len(doc["nodes"]), -1, dtype=np.int32)
    for i, node in enumerate(doc["nodes"]):
        for child in node.get("children", []):
            parent[child] = i
    global_ = np.zeros_like(local)
    pending = set(range(len(local)))
    while pending:
        progressed = False
        for i in list(pending):
            p = parent[i]
            if p < 0 or p not in pending:
                global_[i] = local[i] if p < 0 else global_[p] @ local[i]
                pending.remove(i); progressed = True
        if not progressed:
            raise ValueError("cyclic node tree")
    return global_


def mesh_and_skin(path: str):
    doc, blob = read_glb(path)
    # A source asset can carry a non-skinned accessory mesh before its actual
    # skinned character mesh in node order.  Select the first *usable skinned*
    # primitive instead of assuming node zero is the delivery geometry.  The
    # generated one-mesh GLBs still follow the same path.  This lets the smoke
    # test independently inspect a retained authored rig, while production
    # generic delivery still has to pass its whole-mesh transfer gate.
    mesh_i = None
    primitive = None
    for node_i, node in enumerate(doc["nodes"]):
        if "mesh" not in node or "skin" not in node:
            continue
        for candidate in doc["meshes"][node["mesh"]].get("primitives", []):
            attrs = candidate.get("attributes", {})
            if all(key in attrs for key in ("POSITION", "JOINTS_0", "WEIGHTS_0")) and "indices" in candidate:
                mesh_i, primitive = node_i, candidate
                break
        if primitive is not None:
            break
    if mesh_i is None or primitive is None:
        raise ValueError("no indexed skinned mesh primitive with POSITION/JOINTS_0/WEIGHTS_0")
    selected_skin_i = doc["nodes"][mesh_i]["skin"]
    # Source-retained rigs often split a character into body/head material
    # primitives.  Test all identity-transform primitives sharing the selected
    # skin, otherwise a valid-looking tiny accessory can mask a bad body skin.
    # A transformed multi-node source needs a scene-aware renderer; fail rather
    # than silently applying its node transform with the wrong skin convention.
    candidates = []
    for node_i, node in enumerate(doc["nodes"]):
        if node.get("skin") != selected_skin_i or "mesh" not in node:
            continue
        if any(key in node for key in ("matrix", "translation", "rotation", "scale")):
            raise ValueError("multi-primitive source smoke requires identity skinned mesh-node transforms")
        for candidate in doc["meshes"][node["mesh"]].get("primitives", []):
            attrs = candidate.get("attributes", {})
            if all(key in attrs for key in ("POSITION", "JOINTS_0", "WEIGHTS_0")) and "indices" in candidate:
                candidates.append(candidate)
    if not candidates:
        raise ValueError("selected skin has no usable primitives")
    vertex_blocks, joint_blocks, weight_blocks, face_blocks = [], [], [], []
    vertex_offset = 0
    for candidate in candidates:
        attrs = candidate["attributes"]
        block_vertices = accessor(doc, blob, attrs["POSITION"]).astype(np.float32)
        block_joints = accessor(doc, blob, attrs["JOINTS_0"]).astype(np.int64)
        block_weights = accessor(doc, blob, attrs["WEIGHTS_0"]).astype(np.float32)
        block_faces = accessor(doc, blob, candidate["indices"]).reshape(-1, 3) + vertex_offset
        vertex_blocks.append(block_vertices); joint_blocks.append(block_joints)
        weight_blocks.append(block_weights); face_blocks.append(block_faces)
        vertex_offset += len(block_vertices)
    vertices = np.concatenate(vertex_blocks)
    joints = np.concatenate(joint_blocks)
    weights = np.concatenate(weight_blocks)
    faces = np.concatenate(face_blocks)
    skin = doc["skins"][selected_skin_i]
    skin_nodes = np.asarray(skin["joints"], dtype=np.int64)
    ibm = accessor(doc, blob, skin["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
    local = np.repeat(np.eye(4, dtype=np.float32)[None], len(doc["nodes"]), axis=0)
    for i, node in enumerate(doc["nodes"]):
        if "matrix" in node:
            local[i] = np.asarray(node["matrix"], dtype=np.float32).reshape(4, 4).T
        else:
            local[i, :3, 3] = node.get("translation", [0, 0, 0])
    return doc, vertices, faces, joints, weights, skin_nodes, ibm, local


def deform(vertices, joints, weights, skin_nodes, ibm, global_):
    """Apply glTF LBS using its column-vector matrix convention.

    glTF stores matrices column-major and transforms a homogeneous point as
    ``M @ p``.  ``m`` is therefore [joint, row, column].  The former row-vector
    einsum happened to preserve the rest pose for translation-only binds, but
    applied an animated rotation about the model origin instead of its joint.
    That made the pose gate reject otherwise valid rigs.  Keep this operation
    explicit: the exporter and this gate must exercise the same glTF semantics.
    """
    m = global_[skin_nodes] @ ibm
    h = np.concatenate([vertices, np.ones((len(vertices), 1), dtype=np.float32)], axis=1)
    out = np.zeros((len(vertices), 3), dtype=np.float32)
    for k in range(4):
        transformed = np.einsum("nij,nj->ni", m[joints[:, k]], h)[:, :3]
        out += weights[:, k:k + 1] * transformed
    return out


def deformation_convention_self_test():
    """Guard the glTF column-vector convention with a joint-pivot example."""
    # A point one unit along +X from a joint at (1, 1) must rotate to one unit
    # along +Y from that same joint.  The old row-vector expression instead
    # rotated the point about the world origin.
    vertices = np.asarray([[2.0, 1.0, 0.0]], dtype=np.float32)
    joints = np.zeros((1, 4), dtype=np.int64)
    weights = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    skin_nodes = np.asarray([0], dtype=np.int64)
    global_ = np.eye(4, dtype=np.float32)[None]
    global_[0, :2, 3] = [1.0, 1.0]
    global_[0] = global_[0] @ rotation_z(90.0)
    ibm = np.eye(4, dtype=np.float32)[None]
    ibm[0, :2, 3] = [-1.0, -1.0]
    got = deform(vertices, joints, weights, skin_nodes, ibm, global_)[0]
    expected = np.asarray([1.0, 2.0, 0.0], dtype=np.float32)
    if not np.allclose(got, expected, atol=1e-5):
        raise AssertionError(f"glTF column-vector LBS self-test failed: got {got}, expected {expected}")
    print("glTF column-vector LBS self-test: PASS")


def camera(yaw, centre, extent):
    """Frame arbitrary-scale source and normalized delivery meshes alike."""
    a = np.deg2rad(yaw)
    # The former fixed origin/distance worked only because generated delivery
    # meshes are normalized near [-1,1]. Source-authored GLBs may retain FBX
    # scale and offset, so preserve the old 2.4 eye distance for a 2-unit wide
    # delivery mesh while looking at the actual geometry centre.
    scale = max(float(extent), 1e-5)
    eye = centre + np.array([1.2 * scale * np.sin(a), 0.05 * scale, 1.2 * scale * np.cos(a)])
    forward = centre - eye; forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 1, 0]); right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    pose = np.eye(4); pose[:3, 0] = right; pose[:3, 1] = up; pose[:3, 2] = -forward; pose[:3, 3] = eye
    return pose


def skin_parent_indices(doc, skin_nodes):
    node_parent = np.full(len(doc["nodes"]), -1, dtype=np.int64)
    for node_i, node in enumerate(doc["nodes"]):
        for child in node.get("children", []):
            node_parent[child] = node_i
    node_to_joint = {int(node): joint_i for joint_i, node in enumerate(skin_nodes)}
    return np.asarray([node_to_joint.get(int(node_parent[node]), -1) for node in skin_nodes], dtype=np.int64)


def bone_transform(start, end):
    """Return a transform that maps a Z-axis cylinder onto start -> end."""
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length <= 1e-8:
        return None
    z = np.array([0.0, 0.0, 1.0])
    unit = direction / length
    axis = np.cross(z, unit)
    sine = float(np.linalg.norm(axis))
    cosine = float(np.dot(z, unit))
    if sine <= 1e-8:
        rotation = np.eye(3) if cosine >= 0.0 else np.diag([1.0, -1.0, -1.0])
    else:
        skew = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
        rotation = np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / (sine * sine))
    out = np.eye(4)
    out[:3, :3] = rotation
    out[:3, 3] = (start + end) * 0.5
    return out, length


def add_skeleton(scene, positions, parents, mesh_diag):
    radius = max(mesh_diag * 0.006, 0.003)
    joint_colour = [255, 96, 48, 255]
    bone_colour = [255, 168, 72, 255]
    for joint_i, point in enumerate(positions):
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=radius * 1.45)
        sphere.apply_translation(point)
        sphere.visual.vertex_colors = joint_colour
        scene.add(pyrender.Mesh.from_trimesh(sphere, smooth=False))
        parent = int(parents[joint_i])
        if parent < 0:
            continue
        transformed = bone_transform(positions[parent], point)
        if transformed is None:
            continue
        matrix, length = transformed
        cylinder = trimesh.creation.cylinder(radius=radius, height=length, sections=8)
        cylinder.apply_transform(matrix)
        cylinder.visual.vertex_colors = bone_colour
        scene.add(pyrender.Mesh.from_trimesh(cylinder, smooth=False))


def render(vertices, faces, yaw, skeleton=None):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual.vertex_colors = [190, 205, 215, 255]
    scene = pyrender.Scene(bg_color=[0.08, 0.08, 0.09, 1], ambient_light=[0.45] * 3)
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))
    if skeleton is not None:
        positions, parents = skeleton
        mesh_diag = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
        add_skeleton(scene, positions, parents, mesh_diag)
    bounds = np.asarray([vertices.min(axis=0), vertices.max(axis=0)])
    pose = camera(yaw, bounds.mean(axis=0), (bounds[1] - bounds[0]).max())
    scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 3.2), pose=pose)
    scene.add(pyrender.DirectionalLight(intensity=3.0), pose=pose)
    renderer = pyrender.OffscreenRenderer(480, 640)
    color, _ = renderer.render(scene); renderer.delete()
    return Image.fromarray(color)


def generic_joint_masses_and_centres(vertices, joints, weights, joint_count):
    """Return per-joint influence mass and its weighted vertex centroid."""
    mass = np.zeros(joint_count, dtype=np.float64)
    weighted_pos = np.zeros((joint_count, 3), dtype=np.float64)
    for slot in range(4):
        ids = joints[:, slot]
        w = weights[:, slot].astype(np.float64)
        np.add.at(mass, ids, w)
        for axis in range(3):
            np.add.at(weighted_pos[:, axis], ids, w * vertices[:, axis])
    centres = weighted_pos / np.maximum(mass[:, None], 1e-12)
    return mass, centres


def edge_component_ids(vertices, faces):
    """Label edges by position-welded surface piece, not UV-split indices."""
    parent = np.arange(len(vertices), dtype=np.int64)

    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    # glTF commonly duplicates a position at every UV/chart boundary. Weld
    # those exact/near-exact positions first; otherwise every chart becomes a
    # false "component" and a clean textured character fails this gate. The
    # tolerance is microscopic relative to the delivery diagonal, so separate
    # props only join when they genuinely share geometry.
    extent = float((vertices.max(axis=0) - vertices.min(axis=0)).max())
    quantum = max(extent * 1e-6, 1e-8)
    welded = {}
    for vertex_i, point in enumerate(vertices):
        key = tuple(np.rint(point / quantum).astype(np.int64))
        other = welded.get(key)
        if other is None:
            welded[key] = vertex_i
        else:
            root, other_root = find(vertex_i), find(other)
            if root != other_root:
                parent[root] = other_root
    for triangle in faces:
        root = find(int(triangle[0]))
        for vertex in triangle[1:]:
            other = find(int(vertex))
            if other != root:
                parent[other] = root
    return np.asarray([find(int(face[0])) for face in faces], dtype=np.int64)


def main():
    if sys.argv[1:] == ["--self-test"]:
        deformation_convention_self_test()
        return 0
    if len(sys.argv) < 3:
        raise SystemExit(f"usage: {sys.argv[0]} <rigged.glb> <out.png> [pose options], or --self-test")
    extra = sys.argv[3:]
    generic_extremity = False
    generic_all_influential = False
    generic_joint = None
    show_skeleton = False
    pose_gate = False
    while extra:
        flag = extra.pop(0)
        if flag == "--generic-extremity":
            generic_extremity = True
        elif flag == "--generic-all-influential":
            generic_all_influential = True
        elif flag == "--generic-joint" and extra:
            generic_joint = int(extra.pop(0))
        elif flag == "--show-skeleton":
            show_skeleton = True
        elif flag == "--pose-gate":
            pose_gate = True
        else:
            raise SystemExit(
                f"usage: {sys.argv[0]} <rigged.glb> <out.png> "
                "[--generic-extremity | --generic-joint N | --generic-all-influential] [--show-skeleton]",
            )
    if sum(flag is not None and flag is not False for flag in (generic_extremity, generic_all_influential, generic_joint)) > 1:
        raise SystemExit("choose at most one generic pose selector")
    doc, vertices, faces, joints, weights, skin_nodes, ibm, local = mesh_and_skin(sys.argv[1])
    name_to_node = {n.get("name"): i for i, n in enumerate(doc["nodes"])}
    required = ["mixamorig:LeftArm", "mixamorig:RightArm"]
    rest = global_transforms(doc, local)
    posed_local = local.copy()
    audit_joints = []
    # An explicit generic target is an audit request, even on an otherwise valid
    # Mixamo rig. In particular, appendage-tip recovery must not silently fall
    # back to the pleasant-looking arm pose and leave the recovered joint untested.
    if generic_joint is None and not generic_extremity and not generic_all_influential and all(name in name_to_node for name in required):
        posed_local[name_to_node["mixamorig:LeftArm"]] = posed_local[name_to_node["mixamorig:LeftArm"]] @ rotation_z(-90)
        posed_local[name_to_node["mixamorig:RightArm"]] = posed_local[name_to_node["mixamorig:RightArm"]] @ rotation_z(90)
        pose_label = "arms raised"
    else:
        parent = np.full(len(doc["nodes"]), -1, dtype=np.int64)
        for node_i, node in enumerate(doc["nodes"]):
            for child in node.get("children", []):
                parent[child] = node_i
        skin_node_to_joint = {int(node): joint_i for joint_i, node in enumerate(skin_nodes)}
        roots = {joint_i for joint_i, node in enumerate(skin_nodes) if parent[node] not in skin_node_to_joint}
        mass, centres = generic_joint_masses_and_centres(vertices, joints, weights, len(skin_nodes))
        candidates = [joint_i for joint_i in range(len(skin_nodes)) if joint_i not in roots]
        if not candidates:
            raise ValueError("generic pose smoke requires a non-root skinned joint")
        peak_mass = max(float(mass[joint_i]) for joint_i in candidates)
        influential = [joint_i for joint_i in candidates if mass[joint_i] >= 0.08 * peak_mass]
        if generic_joint is not None:
            if generic_joint not in candidates:
                raise ValueError(f"generic joint {generic_joint} is not a non-root skinned joint")
            chosen_joint = generic_joint
            pose_label = f"generic requested joint {chosen_joint:02d} rotated"
        elif generic_extremity:
            mesh_centre = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
            mesh_scale = max(float((vertices.max(axis=0) - vertices.min(axis=0)).max()), 1e-6)
            # Ignore numerical slivers, then prefer a substantial cluster that
            # is far from the overall mesh centre. This is geometry-only: a
            # sword, antenna, wing, tail, or other distal part may win.
            viable = influential
            chosen_joint = max(
                viable,
                key=lambda joint_i: float(np.linalg.norm(centres[joint_i] - mesh_centre) / mesh_scale)
                * (0.25 + 0.75 * np.sqrt(float(mass[joint_i]) / peak_mass)),
            )
            pose_label = f"generic distal joint {chosen_joint:02d} rotated"
        elif generic_all_influential:
            # A single pleasant distal articulation is not evidence that the
            # whole generic skeleton is usable. Audit every joint carrying a
            # material amount of skin support. Render the strongest supported
            # one so the visible-motion threshold is not decided by a tiny,
            # otherwise healthy distal tip; the promotion decision remains
            # the worst actual LBS seam over the complete set.
            chosen_joint = max(influential, key=lambda joint_i: float(mass[joint_i]))
            audit_joints = influential
            pose_label = f"generic materially weighted joint {chosen_joint:02d} rotated; all {len(audit_joints)} influential joints audited"
        else:
            chosen_joint = max(candidates, key=lambda joint_i: float(mass[joint_i]))
            pose_label = f"generic joint {chosen_joint:02d} rotated"
        chosen_node = int(skin_nodes[chosen_joint])
        posed_local[chosen_node] = posed_local[chosen_node] @ rotation_z(45)
    posed = global_transforms(doc, posed_local)
    rest_v = deform(vertices, joints, weights, skin_nodes, ibm, rest)
    pose_v = deform(vertices, joints, weights, skin_nodes, ibm, posed)
    mesh_diag = float(np.linalg.norm(rest_v.max(axis=0) - rest_v.min(axis=0)))
    displacement = np.linalg.norm(pose_v - rest_v, axis=1)
    moved = float(np.mean(displacement > max(1e-6, mesh_diag * 0.02)))
    # A clean rest pose can conceal a skin field that connects a large part to
    # the wrong ancestor. Measure the actual glTF LBS matrices, rather than a
    # tree heuristic, on the same geometry-based generic articulation used by
    # this smoke render. Ignore numerical sliver edges but fail a visible
    # stretched seam. Numerical controls are recorded in the runbook; they
    # must be regenerated whenever this renderer changes.
    edge_pairs = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    rest_len = np.linalg.norm(rest_v[edge_pairs[:, 0]] - rest_v[edge_pairs[:, 1]], axis=1)
    pose_len = np.linalg.norm(pose_v[edge_pairs[:, 0]] - pose_v[edge_pairs[:, 1]], axis=1)
    valid_edges = rest_len >= mesh_diag * 1e-4
    # A global p999 can conceal a small but clearly visible disconnected prop
    # (weapon, halo, wing panel, or generated attachment) being shredded.  Its
    # few bad edges disappear among a dense character body. Every meaningful
    # connected surface component therefore gets its own p999 in the generic
    # all-joint audit. Tiny numeric fragments are ignored only below 30 edges.
    face_components = edge_component_ids(rest_v, faces)
    edge_components = np.concatenate([face_components, face_components, face_components])
    component_min_edges = 30
    stretch = pose_len[valid_edges] / np.maximum(rest_len[valid_edges], 1e-9)
    stretch_p99 = float(np.quantile(stretch, 0.99)) if len(stretch) else float("inf")
    stretch_p995 = float(np.quantile(stretch, 0.995)) if len(stretch) else float("inf")
    stretch_p999 = float(np.quantile(stretch, 0.999)) if len(stretch) else float("inf")
    stretch_max = float(stretch.max()) if len(stretch) else float("inf")
    stretch_gt5 = float(np.mean(stretch > 5.0)) if len(stretch) else 1.0
    stretch_gt10 = float(np.mean(stretch > 10.0)) if len(stretch) else 1.0
    worst_audit_p999 = stretch_p999
    worst_audit_joint = None
    worst_component_p999 = stretch_p999
    worst_component_joint = None
    worst_component_id = None
    if audit_joints:
        for audit_joint in audit_joints:
            audit_local = local.copy()
            audit_node = int(skin_nodes[audit_joint])
            audit_local[audit_node] = audit_local[audit_node] @ rotation_z(45)
            audit_v = deform(vertices, joints, weights, skin_nodes, ibm, global_transforms(doc, audit_local))
            audit_len = np.linalg.norm(audit_v[edge_pairs[:, 0]] - audit_v[edge_pairs[:, 1]], axis=1)
            audit_stretch = audit_len[valid_edges] / np.maximum(rest_len[valid_edges], 1e-9)
            audit_p999 = float(np.quantile(audit_stretch, 0.999)) if len(audit_stretch) else float("inf")
            if audit_p999 > worst_audit_p999:
                worst_audit_p999 = audit_p999
                worst_audit_joint = audit_joint
            audit_components = edge_components[valid_edges]
            for component in np.unique(audit_components):
                component_stretch = audit_stretch[audit_components == component]
                if len(component_stretch) < component_min_edges:
                    continue
                component_p999 = float(np.quantile(component_stretch, 0.999))
                if component_p999 > worst_component_p999:
                    worst_component_p999 = component_p999
                    worst_component_joint = audit_joint
                    worst_component_id = int(component)
    # Whole-tree generic stress necessarily includes joints at narrow wing or
    # feather transitions. It uses a slightly wider publication threshold than
    # a single named humanoid-arm diagnostic; controls are documented in the
    # runbook and must use this exact column-vector implementation.
    gate_stretch_limit = 6.0 if audit_joints else 5.0
    if pose_gate:
        print(
            f"pose gate: pose={pose_label!r} moved(>2%diag)={moved:.3f} "
            f"p99/p995/p999_stretch={stretch_p99:.3f}/{stretch_p995:.3f}/{stretch_p999:.3f} "
            f"over5/over10={stretch_gt5:.5f}/{stretch_gt10:.5f} max_stretch={stretch_max:.3f} "
            f"all-influential-worst={worst_audit_p999:.3f} "
            f"component-worst={worst_component_p999:.3f}"
            + (f"@joint{worst_audit_joint:02d} " if worst_audit_joint is not None else " ")
            + (f"component={worst_component_id}@joint{worst_component_joint:02d} " if worst_component_joint is not None else " ")
            + f"(gate: moved>=0.010, p999<={gate_stretch_limit:.3f})",
        )
        if moved < 0.010 or worst_audit_p999 > gate_stretch_limit or worst_component_p999 > gate_stretch_limit:
            return 1
    parents = skin_parent_indices(doc, skin_nodes)
    rest_skeleton = (rest[skin_nodes, :3, 3], parents) if show_skeleton else None
    posed_skeleton = (posed[skin_nodes, :3, 3], parents) if show_skeleton else None
    tiles = [
        render(rest_v, faces, 0, rest_skeleton), render(pose_v, faces, 0, posed_skeleton),
        render(rest_v, faces, 35, rest_skeleton), render(pose_v, faces, 35, posed_skeleton),
    ]
    canvas = Image.new("RGB", (960, 1280), "black")
    labels = ["rest front", f"{pose_label} front", "rest side", f"{pose_label} side"]
    for i, tile in enumerate(tiles):
        x, y = (i % 2) * 480, (i // 2) * 640
        canvas.paste(tile, (x, y)); ImageDraw.Draw(canvas).text((x + 8, y + 8), labels[i], fill="yellow")
    canvas.save(sys.argv[2])
    print(
        f"wrote {sys.argv[2]} V={len(vertices)} J={len(skin_nodes)} "
        f"pose={pose_label!r} skeleton={show_skeleton} moved(>2%diag)={moved:.3f} "
        f"p99/p995/p999_stretch={stretch_p99:.3f}/{stretch_p995:.3f}/{stretch_p999:.3f} "
        f"over5/over10={stretch_gt5:.5f}/{stretch_gt10:.5f} max_stretch={stretch_max:.3f} max_disp={displacement.max():.4g}",
    )


if __name__ == "__main__":
    raise SystemExit(main())
