#!/usr/bin/env python3
"""Compare NAVA per-step audio trajectory dump directories.

Both directories should contain sd.cpp-style tensor bins with audio tensors in
ne layout [128, T], as written by C++ NAVA_DUMP_AUDIO_TRAJ and Python
NAVA_DUMP_AUDIO_TRAJ. The report is branch-oriented so a run can quickly show
whether cond, uncond, mmask, cfg, or the scheduler step is the first divergence.
"""
import argparse
import re
from pathlib import Path

import numpy as np


STEP_RE = re.compile(r"^(?P<branch>.+)_(?P<step>\d\d)\.bin$")


def read_bin(path):
    with open(path, "rb") as f:
        n_dims, name_len, typ = np.fromfile(f, np.int32, 3)
        dims = [int(x) for x in np.fromfile(f, np.int32, int(n_dims))]
        name = f.read(int(name_len)).decode("utf-8", errors="replace")
        data = np.fromfile(f, np.float32)
    if typ != 0:
        raise ValueError(f"{path}: expected f32 type 0, got {typ}")
    if data.size != int(np.prod(dims)):
        raise ValueError(f"{path}: payload has {data.size} floats, dims imply {np.prod(dims)}")
    if len(dims) != 2 or dims[0] != 128:
        raise ValueError(f"{path}: expected dims [128, T], got {dims}")
    return name, dims, data.astype(np.float64)


def corrcoef(a, b):
    aa = a - a.mean()
    bb = b - b.mean()
    den = np.linalg.norm(aa) * np.linalg.norm(bb)
    return float(np.dot(aa, bb) / den) if den else 0.0


def classify(filename):
    if filename == "aud_noise.bin":
        return "aud_noise", -1
    m = STEP_RE.match(filename)
    if not m:
        return filename[:-4], -1
    return m.group("branch"), int(m.group("step"))


def compare_file(ref_path, cand_path):
    _, ref_dims, ref = read_bin(ref_path)
    _, cand_dims, cand = read_bin(cand_path)
    if ref_dims != cand_dims:
        raise ValueError(f"{ref_path.name}: dims differ {ref_dims} vs {cand_dims}")
    diff = cand - ref
    return {
        "dims": ref_dims,
        "corr": corrcoef(ref, cand),
        "rmsdiff": float(np.sqrt(np.mean(diff * diff))),
        "reldiff": float(np.linalg.norm(diff) / (np.linalg.norm(ref) + 1e-12)),
        "std_ratio": float(cand.std() / (ref.std() + 1e-12)),
        "ref_std": float(ref.std()),
        "cand_std": float(cand.std()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference_dir", help="reference trajectory directory, usually Python")
    ap.add_argument("candidate_dir", help="candidate trajectory directory, usually C++")
    ap.add_argument("--top", type=int, default=12, help="number of worst files to print")
    args = ap.parse_args()

    ref_dir = Path(args.reference_dir)
    cand_dir = Path(args.candidate_dir)
    ref_files = {p.name for p in ref_dir.glob("*.bin")}
    cand_files = {p.name for p in cand_dir.glob("*.bin")}
    common = sorted(ref_files & cand_files, key=lambda n: (classify(n)[1], classify(n)[0], n))
    missing_ref = sorted(cand_files - ref_files)
    missing_cand = sorted(ref_files - cand_files)

    if not common:
        raise SystemExit("no common .bin files")

    rows = []
    for name in common:
        branch, step = classify(name)
        stats = compare_file(ref_dir / name, cand_dir / name)
        rows.append((name, branch, step, stats))

    print(f"reference: {ref_dir}")
    print(f"candidate: {cand_dir}")
    print(f"common={len(common)} missing_reference={len(missing_ref)} missing_candidate={len(missing_cand)}")
    if missing_ref:
        print("missing in reference: " + " ".join(missing_ref[:20]))
    if missing_cand:
        print("missing in candidate: " + " ".join(missing_cand[:20]))

    print("\nper-file in sampler order:")
    for name, branch, step, s in rows:
        print(
            f"  {name:18s} corr={s['corr']:.6f} rmsdiff={s['rmsdiff']:.6f} "
            f"reldiff={s['reldiff']:.6f} std_ratio={s['std_ratio']:.6f}"
        )

    print(f"\nworst by reldiff (top {min(args.top, len(rows))}):")
    for name, branch, step, s in sorted(rows, key=lambda r: -r[3]["reldiff"])[: args.top]:
        print(
            f"  {name:18s} branch={branch:10s} step={step:02d} "
            f"corr={s['corr']:.6f} reldiff={s['reldiff']:.6f} "
            f"std={s['ref_std']:.6f}/{s['cand_std']:.6f}"
        )

    print("\nbranch summary:")
    branches = sorted({branch for _, branch, _, _ in rows})
    for branch in branches:
        vals = [s for _, b, _, s in rows if b == branch]
        print(
            f"  {branch:10s} files={len(vals):2d} "
            f"max_reldiff={max(v['reldiff'] for v in vals):.6f} "
            f"mean_reldiff={np.mean([v['reldiff'] for v in vals]):.6f} "
            f"min_corr={min(v['corr'] for v in vals):.6f}"
        )


if __name__ == "__main__":
    main()
