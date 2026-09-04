#!/usr/bin/env python3
"""Stage the ours-vs-trellis.cpp A/B into one web-servable tree + a manifest.

Both halves start from the SAME matte, so the comparison isolates geometry, texture and —
because the trellis.cpp mesh is put through OUR rig stage — how riggable each mesh is. The
per-stage ladder is the point: a fault has to be attributed to the stage that produced it, not
the stage where it became visible, and fingers/thin sheets are exactly what a later stage eats.

usage: trellis_ab_stage.py <out-root> [subject ...]
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys

SHOOT = "/mnt/hdd/3d/avatar-shootout/_shootout_out"
OURS = f"{SHOOT}/final_e2e_20260725"
OURS_OVERRIDE = {"miku": f"{SHOOT}/repro_20260813/miku"}   # re-run after the namer fix
TRELLIS = f"{SHOOT}/trellis_cpp_20260813"
CP = "/home/dbrain/dev/longcat-sparse-spike/tools/m1_ref/cpp_port"

# (subject, our run dir name, input matte)
SUBJECTS = [
    ("miku",    "miku",        f"{SHOOT}/v8_production_e2e_20260724/input_birefnet_rgba.png"),
    ("soldier", "soldier",     f"{SHOOT}/soldier_matte.png"),
    ("gilly",   "gilly",       f"{SHOOT}/runbook_image_to_rig/gilly/gilly_matte.png"),
    ("char1",   "char1",       f"{SHOOT}/char1_matte.png"),
    ("toy1",    "toy1_matted", f"{OURS}/toy1_matted/input_birefnet_rgba.png"),
    ("toy2",    "toy2",        f"{SHOOT}/toy2_matte.png"),
]

# stage ladder. (key, label, note, filename-in-staged-subject-dir)
OURS_STAGES = [
    ("input",  "input matte",   "the same RGBA cutout feeds both pipelines",      "input.png"),
    ("coarse", "1 · O-Voxel raw", "decoder dual-grid mesh, straight out of M4 — a soup, not a surface", "ours_coarse.glb"),
    ("dc",     "2 · narrow-band DC", "+ Taubin x2 — the parity mesh",             "ours_dc.glb"),
    ("hero",   "3 · hero, baked", "decimated 220k, direct volume PBR bake + normal map", "ours_hero.glb"),
    ("rig",    "4 · rigged",      "SkinTokens skeleton + skin on the hero",       "ours_hero.glb"),
    ("anim",   "5 · exercise",    "every materially weighted joint swings in turn", "ours_anim.glb"),
    ("relief", "relief · front",  "normals-as-colour, FLAT — geometry with NO texture and no normal map", "relief_0.png"),
    ("relief2","relief · back",   "the same, yaw 180",                            "relief_180.png"),
]
TRELLIS_STAGES = [
    ("input",  "input matte",   "identical input",                                "input.png"),
    ("hero",   "1 · trellis 1024", "weld→fill→DC remesh→QEM 300k→xatlas→PBR bake; textures are LOSSY WEBP and there is no normal map", "trellis.glb"),
    ("hq",     "1b · res 1536",  "their top tier, PNG textures — miku only so far", "trellis_1536.glb"),
    ("rig",    "2 · rigged",      "OUR SkinTokens stage, run on their mesh",      "trellis_rigged.glb"),
    ("anim",   "3 · exercise",    "same exercise clip generator",                 "trellis_anim.glb"),
    ("relief", "relief · front",  "normals-as-colour, FLAT — the honest geometry A/B", "relief_0.png"),
    ("relief2","relief · back",   "the same, yaw 180",                            "relief_180.png"),
]


def link(src: str, dst: str) -> bool:
    if not os.path.exists(src):
        return False
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(src, dst)
    return True


def grab(path: str, pattern: str, group: int = 1, cast=str):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            m = re.search(pattern, f.read(), re.M)
    except OSError:
        return None
    if not m:
        return None
    try:
        return cast(m.group(group))
    except (ValueError, IndexError):
        return None


def mesh_counts(glb: str) -> dict:
    """verts/faces straight from the GLB, so the table never quotes a log instead of the asset."""
    try:
        import trimesh
        scene = trimesh.load(glb, process=False)
        mesh = scene.dump(concatenate=True) if hasattr(scene, "dump") else scene
        return {"verts": int(len(mesh.vertices)), "faces": int(len(mesh.faces))}
    except Exception:
        return {}


def detail(glb: str) -> dict:
    """mesh_detail_metric.py — dihedral tail + Taubin roughness, both tessellation-independent."""
    try:
        out = subprocess.run(
            [sys.executable, f"{CP}/shootout/mesh_detail_metric.py", glb],
            capture_output=True, text=True, timeout=900,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    d = {}
    for key, pat in (("dihedral_p95", r"dihedral.*?p95[= ]+([0-9.]+)"),
                     ("rough_p95", r"rough.*?p95[= ]+([0-9.eE+-]+)")):
        m = re.search(pat, out, re.I | re.S)
        if m:
            d[key] = float(m.group(1))
    d["raw"] = out.strip()
    return d


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    root = sys.argv[1]
    want = set(sys.argv[2:])
    os.makedirs(root, exist_ok=True)

    manifest = {"subjects": [], "ours_stages": OURS_STAGES, "trellis_stages": TRELLIS_STAGES}

    for subj, ours_name, matte in SUBJECTS:
        if want and subj not in want:
            continue
        od = OURS_OVERRIDE.get(subj, f"{OURS}/{ours_name}")
        sd = os.path.join(root, subj)
        os.makedirs(sd, exist_ok=True)

        have = {}
        have["input"] = link(matte, f"{sd}/input.png")
        have["ours_coarse"] = link(f"{od}/stage/coarse.glb", f"{sd}/ours_coarse.glb")
        have["ours_dc"] = link(f"{od}/stage/dc.glb", f"{sd}/ours_dc.glb")
        have["ours_hero"] = link(f"{od}/rigged.glb", f"{sd}/ours_hero.glb")
        have["ours_anim"] = link(f"{od}/rigged.anim.glb", f"{sd}/ours_anim.glb")
        have["trellis"] = link(f"{TRELLIS}/{subj}.glb", f"{sd}/trellis.glb")
        have["trellis_rigged"] = link(f"{TRELLIS}/{subj}.rigged.glb", f"{sd}/trellis_rigged.glb")
        have["trellis_anim"] = link(f"{TRELLIS}/{subj}.rigged.anim.glb", f"{sd}/trellis_anim.glb")
        have["trellis_1536"] = link(f"{TRELLIS}/{subj}_1536.glb", f"{sd}/trellis_1536.glb")
        # relief sheets are written in place by render_relief.py, not linked
        have["relief_0"] = os.path.exists(f"{sd}/relief_0.png")
        have["relief_180"] = os.path.exists(f"{sd}/relief_180.png")

        ours = {
            "run": od,
            "wall_s": grab(f"{od}/metrics.txt", r"^wall_total_s=(\d+)", cast=int),
            "peak_mib": grab(f"{od}/metrics.txt", r"peak_mib=(\d+)", cast=int),
            "joints": grab(f"{od}/qc_rig_score.txt", r"J=(\d+)", cast=int),
            "rig_score": grab(f"{od}/qc_rig_score.txt", r"TOTAL=([0-9.]+)", cast=float),
            "named_core": grab(f"{od}/qc_rig_score.txt", r"named_core=(\d+/\d+)"),
            "symmetry": grab(f"{od}/qc_rig_score.txt", r"symmetry=([0-9.]+)", cast=float),
            "weight_health": grab(f"{od}/qc_weight_health.txt", r"weight health: (\w+)"),
            "pose_p999": grab(f"{od}/qc_pose_gate.txt", r"p99/p995/p999_stretch=[0-9.]+/[0-9.]+/([0-9.]+)", cast=float),
        }
        tre = {
            "wall_s": grab(f"{TRELLIS}/{subj}.log", r"done in ([0-9.]+)s", cast=lambda s: int(float(s))),
            "rig_wall_s": grab(f"{TRELLIS}/{subj}.rig_timing.txt", r"rig_wall_s=(\d+)", cast=int),
            "components": grab(f"{TRELLIS}/{subj}.log", r"uv_bake: (\d+) components", cast=int),
            "joints": grab(f"{TRELLIS}/{subj}.qc_rig_score.txt", r"J=(\d+)", cast=int),
            "rig_score": grab(f"{TRELLIS}/{subj}.qc_rig_score.txt", r"TOTAL=([0-9.]+)", cast=float),
            "named_core": grab(f"{TRELLIS}/{subj}.qc_rig_score.txt", r"named_core=(\d+/\d+)"),
            "symmetry": grab(f"{TRELLIS}/{subj}.qc_rig_score.txt", r"symmetry=([0-9.]+)", cast=float),
            "weight_health": grab(f"{TRELLIS}/{subj}.qc_weight_health.txt", r"weight health: (\w+)"),
            "pose_p999": grab(f"{TRELLIS}/{subj}.qc_pose_gate.txt", r"p99/p995/p999_stretch=[0-9.]+/[0-9.]+/([0-9.]+)", cast=float),
        }
        for side, path in (("ours", f"{sd}/ours_hero.glb"), ("trellis", f"{sd}/trellis.glb")):
            if os.path.exists(path):
                (ours if side == "ours" else tre).update(mesh_counts(path))

        manifest["subjects"].append({"name": subj, "have": have, "ours": ours, "trellis": tre})
        print(f"[stage] {subj}: " + " ".join(k for k, v in have.items() if v))

    with open(os.path.join(root, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print(f"[stage] manifest -> {root}/manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
