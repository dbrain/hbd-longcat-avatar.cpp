#!/usr/bin/env python3
"""Skin-weight distribution gate for a rigged GLB.

`rig_score` validates the skeleton and `rig_pose_smoke.py --pose-gate` validates
deformation, but neither catches a *concentrated* weight field: a 50-bone rig
whose mass sits almost entirely on one joint still poses "within tolerance"
because the pose gate only exercises joints holding >= 8% of peak mass. Fewer
usable joints therefore means a weaker audit and an easier pass — backwards.

This gate closes that hole. It fails a rig whose articulation is nominal.

usage: rig_weight_health.py <rigged.glb> [--min-influential 4]
                                         [--review-single-share 0.35]

Calibrated against rigs whose pose-gate renders were inspected by eye on
2026-07-24, not chosen a priori:

    artifact                              influential  single-share  render
    fresh e2e miku/hymotion_rigged.glb        25/37        16.4%     clean
    fresh e2e gilly/hymotion_rigged.glb       10/56        26.1%     clean
    fresh e2e soldier/hymotion_rigged.glb      4/25        46.3%     CLEAN (see below)
    miku attachment_recovery_experimental      2/50        58.8%     broken

**A high single-joint share is a review flag, not a failure.** The soldier is
the calibration case: 44.7% of its mesh sits above the Head joint and 100% of
that is Head-dominant, because the model wears a bearskin busby. A large rigid
prop *should* concentrate on one bone. An earlier version of this gate failed
the soldier for it; that was wrong.

Only a genuinely low influential-joint count is treated as a failure, and even
that separates the known-good and known-broken cases by a thin margin (4 vs 2).
**These metrics are weak discriminators. The pose-gate render is the authority**
— use the stage explorer page and look at the model. This tool exists to stop a
rig from being published without anyone looking, not to replace looking.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("rig_pose_smoke",
                                               os.path.join(_HERE, "rig_pose_smoke.py"))
_ps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ps)


def health(path: str):
    _doc, vertices, _faces, joints, weights, skin_nodes, _ibm, _local = _ps.mesh_and_skin(path)
    n = len(skin_nodes)
    mass, _ = _ps.generic_joint_masses_and_centres(vertices, joints, weights, n)
    mass = np.asarray(mass, dtype=np.float64)
    total = max(mass.sum(), 1e-12)
    frac = mass / total
    dominant = np.bincount(joints[np.arange(len(joints)), weights.argmax(1)],
                           minlength=n) / max(len(vertices), 1)
    return dict(J=n, V=len(vertices),
                influential=int((mass >= 0.08 * mass.max()).sum()),
                mass_1pct=int((frac >= 0.01).sum()),
                dominant_1pct=int((dominant >= 0.01).sum()),
                single_share=float(frac.max()))


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    path = args[0]
    min_influential = 4
    review_single_share = 0.35
    for i, a in enumerate(args):
        if a == "--min-influential":
            min_influential = int(args[i + 1])
        elif a == "--review-single-share":
            review_single_share = float(args[i + 1])

    h = health(path)
    print(f"{path} J={h['J']} V={h['V']} influential={h['influential']} "
          f"(>= 8% of peak) mass>=1%={h['mass_1pct']} dominant>=1%verts={h['dominant_1pct']} "
          f"biggest_joint_share={h['single_share'] * 100:.1f}%")

    if h["single_share"] > review_single_share:
        print(f"  REVIEW: one joint holds {h['single_share'] * 100:.1f}% of all skin mass. "
              f"Legitimate for a large rigid prop (hat, helmet, shell) — confirm on the "
              f"pose-gate render before treating it as a defect.")
    if h["influential"] < min_influential:
        print(f"  WEIGHT-HEALTH FAIL: only {h['influential']}/{h['J']} joints carry >=8% of "
              f"peak mass (need >= {min_influential}) — nothing meaningful articulates and "
              f"the pose gate would audit almost nothing")
        return 1
    print("  weight health: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
