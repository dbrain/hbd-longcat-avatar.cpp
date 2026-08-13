#!/usr/bin/env python3
"""Diagnostic: SkinTokens skin-only mode on a supplied named biped skeleton.

This deliberately does *not* join the image-to-rig wrapper.  It tests the
upstream ``--use_skeleton`` escape hatch: retain every mesh component (hair,
wings, props, etc.), supply a known skeleton, and ask TokenRig to generate
only the SkinTokens weights.  Promotion still requires full-mesh transfer,
bone-name validation and a rendered deformation check.

The template skeleton is read from a validated rigged GLB.  ``--fit-bbox`` is
only a controlled diagnostic alignment between two normalized point clouds;
it is not a generic skeleton fitter.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh


MIXAMO_CORE = [
    "mixamorig:Hips", "mixamorig:Spine", "mixamorig:Spine1", "mixamorig:Spine2",
    "mixamorig:LeftShoulder", "mixamorig:LeftArm", "mixamorig:LeftForeArm", "mixamorig:LeftHand",
    "mixamorig:RightShoulder", "mixamorig:RightArm", "mixamorig:RightForeArm", "mixamorig:RightHand",
    "mixamorig:Neck", "mixamorig:Head",
    "mixamorig:LeftUpLeg", "mixamorig:LeftLeg", "mixamorig:LeftFoot", "mixamorig:LeftToeBase",
    "mixamorig:RightUpLeg", "mixamorig:RightLeg", "mixamorig:RightFoot", "mixamorig:RightToeBase",
]


def samples(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.load(path / "vertices.npy").astype(np.float32, copy=False)
    normals = np.load(path / "normals.npy").astype(np.float32, copy=False)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or normals.shape != vertices.shape:
        raise ValueError(f"invalid sampled rig input: {path}")
    return vertices, normals


def load_core(glb: Path) -> tuple[np.ndarray, np.ndarray]:
    scene = trimesh.load(glb, process=False)
    if not isinstance(scene, trimesh.Scene):
        raise ValueError(f"expected a GLB scene: {glb}")
    names = set(scene.graph.nodes)
    missing = [name for name in MIXAMO_CORE if name not in names]
    if missing:
        raise ValueError(f"template is missing Mixamo core nodes: {missing}")
    parent_by_node = scene.graph.transforms.parents
    joints = np.stack([scene.graph.get(name)[0][:3, 3] for name in MIXAMO_CORE]).astype(np.float32)
    index = {name: i for i, name in enumerate(MIXAMO_CORE)}
    parents = []
    for name in MIXAMO_CORE:
        parent = parent_by_node[name]
        if name == "mixamorig:Hips":
            if parent != "world":
                raise ValueError(f"Hips must be a root below world, got {parent!r}")
            parents.append(-1)
        elif parent not in index:
            raise ValueError(f"core node {name} has non-core parent {parent!r}")
        else:
            parents.append(index[parent])
    return joints, np.asarray(parents, dtype=np.int64)


def fit_bbox(joints: np.ndarray, source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    src_min, src_max = source_points.min(axis=0), source_points.max(axis=0)
    dst_min, dst_max = target_points.min(axis=0), target_points.max(axis=0)
    src_extent = src_max - src_min
    if np.any(src_extent <= 1e-6):
        raise ValueError("source point cloud has a degenerate axis")
    return ((joints - src_min) / src_extent * (dst_max - dst_min) + dst_min).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skintokens-root", type=Path, required=True)
    parser.add_argument("--template-glb", type=Path, required=True)
    parser.add_argument("--template-samples", type=Path, required=True)
    parser.add_argument("--target-samples", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fit-bbox", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
    os.environ.setdefault("XFORMERS_IGNORE_FLASH_VERSION_CHECK", "1")
    root = args.skintokens_root.resolve()
    sys.path.insert(0, str(root))
    os.chdir(root)
    from src.server.spec import get_model
    from src.tokenizer.spec import TokenizeInput

    target_vertices, target_normals = samples(args.target_samples)
    joints, parents = load_core(args.template_glb)
    if args.fit_bbox:
        template_vertices, _ = samples(args.template_samples)
        joints = fit_bbox(joints, template_vertices, target_vertices)

    model = get_model("experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt").eval()
    skeleton_tokens = model.tokenizer.tokenize(
        TokenizeInput(joints=joints, parents=parents.tolist(), cls=None, joint_names=None)
    )
    print(f"[template-skin] template J={len(joints)} tokens={len(skeleton_tokens)} fit_bbox={args.fit_bbox}", flush=True)
    vertices_t = torch.tensor(target_vertices, dtype=torch.float32, device="cuda")
    normals_t = torch.tensor(target_normals, dtype=torch.float32, device="cuda")
    result = model.generate(
        vertices_t,
        normals_t,
        skeleton_tokens=skeleton_tokens,
        return_decode_dict=False,
        max_length=2048,
        repetition_penalty=2.0,
        num_beams=1,
        do_sample=False,
        num_return_sequences=1,
    )
    if result.detokenize_output is None or result.skin_pred is None:
        raise RuntimeError("upstream skin-only generation did not produce skeleton + skin weights")
    decoded = result.detokenize_output
    raw_skin = result.skin_pred.detach().float().cpu().numpy()
    if decoded.J != len(MIXAMO_CORE) or raw_skin.shape != (len(target_vertices), decoded.J):
        raise RuntimeError(f"unexpected output shapes: J={decoded.J}, skin={raw_skin.shape}")
    if not np.isfinite(raw_skin).all() or (raw_skin < 0).any():
        raise RuntimeError("skin output contains NaN/Inf")
    raw_sums = raw_skin.sum(axis=1, keepdims=True)
    if (raw_sums <= 1e-8).any():
        raise RuntimeError("skin output has an all-zero row")
    # The upstream Asset export path normalizes decoded VAE weights before it
    # writes a rig.  Make that contract explicit here: retain the raw decoder
    # tensor for diagnosis but hand an exporter-ready matrix to the GLB glue.
    skin = raw_skin / raw_sums
    args.out.mkdir(parents=True, exist_ok=False)
    np.save(args.out / "sampled_vertices.npy", target_vertices)
    np.save(args.out / "gen_joints.npy", decoded.joints.astype(np.float32))
    np.save(args.out / "gen_parents.npy", np.asarray(decoded.parents, dtype=np.int64))
    np.save(args.out / "raw_gen_skin_pred.npy", raw_skin)
    np.save(args.out / "gen_skin_pred.npy", skin)
    (args.out / "template_names.txt").write_text("\n".join(MIXAMO_CORE) + "\n", encoding="utf-8")
    print(
        f"[template-skin] PASS J={decoded.J} skin={skin.shape} "
        f"raw-row-sum-maxerr={np.abs(raw_sums[:, 0] - 1).max():.6g} "
        f"normalized-row-sum-maxerr={np.abs(skin.sum(axis=1) - 1).max():.6g} -> {args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
