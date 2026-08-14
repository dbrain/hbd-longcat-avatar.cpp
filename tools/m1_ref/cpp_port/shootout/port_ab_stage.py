#!/usr/bin/env python3
"""Stage the spike-vs-ported-onto-master A/B into one web-servable tree + a manifest.

The acceptance test for the port is an EYE TEST, not bit-exactness: master's ggml moved two
upstream bumps, so one voxel moving at the sparse-structure DiT is expected and cascades into a
different mesh and a different skeleton. What has to hold is that the delivery still looks right
and still rigs, on every subject — n=1 is worthless on this stack.

Both sides run the identical ladder, so the stage rails line up 1:1.

usage: port_ab_stage.py <out-root> [subject ...]
"""

from __future__ import annotations

import json
import os
import re
import sys

SHOOT = "/mnt/hdd/3d/avatar-shootout/_shootout_out"
# LEFT = the tree that has always worked (spike/sparse-conv-3d @ 3a7c436). miku was re-run today
# after the namer fix; the other five are the 2026-07-25 delivery from the same code.
SPIKE = f"{SHOOT}/final_e2e_20260725"
SPIKE_OVERRIDE = {"miku": f"{SHOOT}/repro_20260813/miku"}
# RIGHT = spike/on-master @ ebbf007, ggml 9847065d.
PORTED = f"{SHOOT}/onmaster_20260813"

SUBJECTS = [
    ("miku",    "miku",        f"{SHOOT}/v8_production_e2e_20260724/input_birefnet_rgba.png"),
    ("soldier", "soldier",     f"{SHOOT}/soldier_matte.png"),
    ("gilly",   "gilly",       f"{SHOOT}/runbook_image_to_rig/gilly/gilly_matte.png"),
    ("char1",   "char1",       f"{SHOOT}/char1_matte.png"),
    ("toy1",    "toy1_matted", f"{SPIKE}/toy1_matted/input_birefnet_rgba.png"),
    ("toy2",    "toy2",        f"{SHOOT}/toy2_matte.png"),
]

STAGES = [
    ("input",  "input matte",     "the same RGBA cutout feeds both trees",            "{s}_input.png"),
    ("coarse", "1 · O-Voxel raw", "decoder dual-grid mesh — a soup, not a surface",   "{s}_coarse.glb"),
    ("dc",     "2 · narrow-band DC", "+ Taubin x2 — the parity mesh",                 "{s}_dc.glb"),
    ("hero",   "3 · hero, baked", "decimated 220k + direct volume PBR bake + normals", "{s}_hero.glb"),
    ("rig",    "4 · rigged",      "SkinTokens skeleton + skin",                       "{s}_hero.glb"),
    ("anim",   "5 · exercise",    "every materially weighted joint swings in turn",   "{s}_anim.glb"),
    ("relief", "relief · front",  "normals-as-colour, FLAT — geometry, no texture",   "relief_0.png"),
    ("relief2","relief · back",   "the same, yaw 180",                                "relief_180.png"),
]


def link(src: str, dst: str) -> bool:
    if not os.path.exists(src):
        return False
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(src, dst)
    return True


def grab(path: str, pattern: str, cast=str):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            m = re.search(pattern, f.read(), re.M)
    except OSError:
        return None
    if not m:
        return None
    try:
        return cast(m.group(1))
    except (ValueError, IndexError):
        return None


def side_metrics(run_dir: str) -> dict:
    return {
        "run": run_dir,
        "wall_s": grab(f"{run_dir}/metrics.txt", r"^wall_total_s=(\d+)", int),
        "peak_mib": grab(f"{run_dir}/metrics.txt", r"peak_mib=(\d+)", int),
        "n1": grab(f"{run_dir}/metrics.txt", r"geometry: N1=(\d+)", int),
        "verts": grab(f"{run_dir}/metrics.txt", r"DONE ->.*verts=(\d+)", int),
        "faces": grab(f"{run_dir}/metrics.txt", r"DONE ->.*faces=(\d+)", int),
        "joints": grab(f"{run_dir}/qc_rig_score.txt", r"J=(\d+)", int),
        "rig_score": grab(f"{run_dir}/qc_rig_score.txt", r"TOTAL=([0-9.]+)", float),
        "named_core": grab(f"{run_dir}/qc_rig_score.txt", r"named_core=(\d+/\d+)"),
        "symmetry": grab(f"{run_dir}/qc_rig_score.txt", r"symmetry=([0-9.]+)", float),
        "biggest_share": grab(f"{run_dir}/qc_weight_health.txt", r"biggest_joint_share=([0-9.]+)%", float),
        "weight_health": grab(f"{run_dir}/qc_weight_health.txt", r"weight health: (\w+)"),
        "pose_p999": grab(f"{run_dir}/qc_pose_gate.txt",
                          r"p99/p995/p999_stretch=[0-9.]+/[0-9.]+/([0-9.]+)", float),
    }


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    root = sys.argv[1]
    want = set(sys.argv[2:])
    os.makedirs(root, exist_ok=True)
    manifest = {"sides": [{"key": "spike", "label": "spike 3a7c436", "note": "the tree that has always worked"},
                          {"key": "ported", "label": "on-master ebbf007", "note": "master cce6ee4 + ggml 9847065d"}],
                "stages": STAGES, "subjects": []}

    for subj, spike_name, matte in SUBJECTS:
        if want and subj not in want:
            continue
        sd = os.path.join(root, subj)
        os.makedirs(sd, exist_ok=True)
        runs = {"spike": SPIKE_OVERRIDE.get(subj, f"{SPIKE}/{spike_name}"),
                "ported": f"{PORTED}/{subj}"}
        have = {"input": link(matte, f"{sd}/input.png"),
                "relief_0": os.path.exists(f"{sd}/relief_0.png"),
                "relief_180": os.path.exists(f"{sd}/relief_180.png")}
        for side, rd in runs.items():
            have[f"{side}_input"] = have["input"]
            have[f"{side}_coarse"] = link(f"{rd}/stage/coarse.glb", f"{sd}/{side}_coarse.glb")
            have[f"{side}_dc"] = link(f"{rd}/stage/dc.glb", f"{sd}/{side}_dc.glb")
            have[f"{side}_hero"] = link(f"{rd}/rigged.glb", f"{sd}/{side}_hero.glb")
            have[f"{side}_anim"] = link(f"{rd}/rigged.anim.glb", f"{sd}/{side}_anim.glb")
            have[f"{side}_rig"] = have[f"{side}_hero"]

        manifest["subjects"].append({
            "name": subj, "have": have,
            "spike": side_metrics(runs["spike"]), "ported": side_metrics(runs["ported"]),
        })
        print(f"[stage] {subj}: " + " ".join(k for k, v in have.items() if v))

    with open(os.path.join(root, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print(f"[stage] manifest -> {root}/manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
