#!/usr/bin/env python3
"""Scan a stage_explorer_e2e.sh output root and emit stages.json for the eye-test page.

Records, per subject and per pipeline stage: the artifact, its geometry size,
atlas size, the stage's own wall time from the run log, the subject's peak VRAM,
and any gate verdicts that exist. Missing stages are recorded as missing rather
than skipped, so a run that died mid-pipeline is visible as such.

usage: stage_explorer_manifest.py <out-root> [> stages.json]
"""

from __future__ import annotations

import json
import os
import re
import struct
import sys


def glb_stats(path: str):
    """Face/vertex counts and largest embedded image, without a full parse."""
    try:
        with open(path, "rb") as f:
            head = f.read(12)
            magic, _v, _l = struct.unpack("<III", head)
            if magic != 0x46546C67:
                return None
            jlen, _jt = struct.unpack("<II", f.read(8))
            doc = json.loads(f.read(jlen))
    except Exception:
        return None
    acc = doc.get("accessors", [])
    faces = verts = 0
    for mesh in doc.get("meshes", []):
        for prim in mesh.get("primitives", []):
            if "indices" in prim and prim["indices"] < len(acc):
                faces += acc[prim["indices"]]["count"] // 3
            pos = prim.get("attributes", {}).get("POSITION")
            if pos is not None and pos < len(acc):
                verts += acc[pos]["count"]
    atlas = ""
    for img in doc.get("images", []):
        bv = img.get("bufferView")
        if bv is not None and bv < len(doc.get("bufferViews", [])):
            atlas = f"{doc['bufferViews'][bv]['byteLength'] // 1024} KiB embedded"
    return {"faces": faces, "verts": verts, "atlas": atlas,
            "joints": len(doc["skins"][0]["joints"]) if doc.get("skins") else 0,
            "animations": len(doc.get("animations", [])),
            "bytes": os.path.getsize(path)}


MESH_STAGES = [
    ("01-source",   "01 · authored source mesh",   "source_mesh.glb",                  "model",
     "these subjects have no input image — they enter at the rig stage from an authored mesh"),
    ("08-rig",      "02 · Rigged (generic)",       "generic_rigged.glb",               "model",
     "native SkinTokens generic rig; absent means the structural or deformation gate rejected it"),
    ("08b-raw",     "02b · raw pre-gate candidate", "generic_rigged.learned-smooth.raw-before-component-repair.glb", "model",
     "what the decoder produced before native component repair and the gate — kept so a rejection can be looked at"),
    ("09-skeleton", "03 · Skeleton + pose gate",   "generic_rigged.pose-gate.png",     "image",
     "rest vs posed with the skeleton drawn"),
    ("10-exercise", "04 · Skeleton exercising",    "generic_rigged.exercise.glb",      "model",
     "every materially weighted joint swings in turn — press play"),
]

STAGES = [
    ("01-input",    "01 · input image",            "input_matte.png",                 "image",
     "the exact model-facing matte; everything downstream is conditioned on this"),
    ("02-coarse",   "02 · Pixal3D coarse seed",    "coarse_geometry.glb",             "model",
     "first geometry. Lumpy is expected; torn or hollow here is a fault that propagates"),
    ("03-refined",  "03 · UltraShape refined",     "refined_geometry.glb",            "model",
     "detail pass over the seed. This exact mesh is the texture and rig source"),
    ("04-hero",     "04 · Hero textured",          "native_hero_textured.glb",        "model",
     "full refined mesh, 8K atlas. Visual master; never auto-rigged"),
    ("05-high",     "05 · High textured (300k)",   "native_high_textured.glb",        "model",
     "the rig source tier"),
    ("06-medium",   "06 · Medium textured (150k)", "native_medium_textured.glb",      "model",
     "general game character tier"),
    ("07-low",      "07 · Low textured (50k)",     "native_low_textured.glb",         "model",
     "background/distant tier"),
    ("08-rig",      "08 · Rigged hand-off",        "hymotion_rigged.glb",             "model",
     "the published rig, if it passed every gate"),
    ("09-skeleton", "09 · Skeleton + pose gate",   "hymotion_rigged.pose-gate.png",   "image",
     "rest vs posed with the skeleton drawn; this is the deformation evidence"),
    ("10-exercise", "10 · Skeleton exercising",    "hymotion_rigged.exercise.glb",    "model",
     "every materially weighted joint swings in turn — press play"),
]

ATLASES = {"04-hero": "native_hero_textured_atlas.png",
           "05-high": "native_high_textured_atlas.png",
           "06-medium": "native_medium_textured_atlas.png",
           "07-low": "native_low_textured_atlas.png"}


def stage_times(log_path: str):
    """`[N] Stage name (12.3s)` lines the native pipeline already prints."""
    out = []
    try:
        text = open(log_path, encoding="utf-8", errors="replace").read()
    except OSError:
        return out
    for m in re.finditer(r"^\[(\d+)\]\s+(.+?)\s+\(([\d.]+)s\)\s*$", text, re.M):
        out.append({"n": int(m.group(1)), "label": m.group(2), "seconds": float(m.group(3))})
    return out


def read_kv(path: str):
    d = {}
    try:
        for line in open(path, encoding="utf-8", errors="replace"):
            if "=" in line:
                k, v = line.split("=", 1)
                d[k.strip()] = v.strip()
    except OSError:
        pass
    return d


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    root = sys.argv[1]
    subjects = []
    for name in sorted(os.listdir(root)):
        sub = os.path.join(root, name)
        if not os.path.isdir(sub) or name.startswith("."):
            continue
        timing = read_kv(os.path.join(root, f"{name}.timing.txt"))
        mesh_kind = timing.get("kind") == "mesh" or os.path.exists(
            os.path.join(sub, "source_mesh.glb"))
        stage_list = MESH_STAGES if mesh_kind else STAGES
        rig_stem = "generic_rigged" if mesh_kind else "hymotion_rigged"
        entry = {"id": name,
                 "kind": "mesh" if mesh_kind else "image",
                 "wall_seconds": int(timing.get("wall_seconds", 0) or 0),
                 "peak_vram_mib": int(timing.get("peak_vram_mib", 0) or 0),
                 "exit": int(timing.get("exit", -1) or -1),
                 "stage_times": stage_times(os.path.join(root, f"{name}.e2e.log")),
                 "stages": []}
        for sid, label, fname, kind, note in stage_list:
            # The raw pre-gate candidate has been written under two naming
            # conventions (with and without the skin-mode infix); accept either
            # rather than silently reporting the stage as missing.
            for cand in (fname, fname.replace(".learned-smooth", "")):
                if os.path.exists(os.path.join(sub, cand)):
                    fname = cand
                    break
            path = os.path.join(sub, fname)
            rec = {"id": sid, "label": label, "file": fname, "kind": kind,
                   "note": note, "present": os.path.exists(path)}
            if rec["present"] and kind == "model":
                rec["stats"] = glb_stats(path)
            if sid in ATLASES and os.path.exists(os.path.join(sub, ATLASES[sid])):
                rec["atlas_file"] = ATLASES[sid]
            for suffix, key in (("pose-gate.txt", "gate"), ("weight-health.txt", "health")):
                f = os.path.join(sub, f"{rig_stem}.{suffix}")
                if sid == "08-rig" and os.path.exists(f):
                    rec[key] = open(f, encoding="utf-8", errors="replace").read().strip()
            # A texture stage's file existing does NOT mean it was accepted: the
            # bake writes the GLB and its QC, and the gate rejects afterwards.
            # Surface that verdict so a rejected asset can never be displayed as
            # a delivered stage.
            qc = os.path.join(sub, f"{fname[:-4]}.glb.texture-qc.txt") if fname.endswith(".glb") else ""
            if qc and os.path.exists(qc):
                text = open(qc, encoding="utf-8", errors="replace").read()
                rec["texture_qc"] = text.strip()
                verdict = re.search(r"^sampling_verdict=(.*)$", text, re.M)
                missing = re.search(r"^unresolved_surface_fraction_after_chart_repair=([\d.]+)$", text, re.M)
                rec["texture_verdict"] = verdict.group(1).strip() if verdict else "unknown"
                rec["texture_unresolved"] = float(missing.group(1)) if missing else 0.0
                if rec["texture_verdict"] != "complete-after-chart-repair" or rec["texture_unresolved"] > 0.001:
                    rec["rejected"] = (f"texture REJECTED: {rec['texture_verdict']}, "
                                       f"{rec['texture_unresolved']*100:.1f}% of the surface unresolved")
            entry["stages"].append(rec)
        subjects.append(entry)
    json.dump({"schema_version": 1, "root": os.path.basename(root), "subjects": subjects},
              sys.stdout, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
