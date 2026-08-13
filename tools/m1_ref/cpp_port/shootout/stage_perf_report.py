#!/usr/bin/env python3
"""Per-stage wall time and peak VRAM for stage-explorer runs; diffs two roots.

    stage_perf_report.py <root>                 # one run
    stage_perf_report.py <baseline> <candidate> # A/B, stage by stage

Stage labels come from the pipeline's own `[n] label (12.3s)` lines, so this
reports what the code actually did rather than a hand-maintained stage list.
"""

from __future__ import annotations

import os
import re
import sys

STAGE_RE = re.compile(r"^\[(\d+[a-z]*)\]\s+(.+?)\s+\(([\d.]+)s\)\s*$", re.M)


def load(root: str):
    runs = {}
    for f in sorted(os.listdir(root)):
        if not f.endswith(".e2e.log"):
            continue
        subject = f[: -len(".e2e.log")]
        text = open(os.path.join(root, f), encoding="utf-8", errors="replace").read()
        stages = []
        for m in STAGE_RE.finditer(text):
            label = m.group(2)
            # collapse the noisy per-run detail so two runs' stages line up
            label = re.sub(r"\s*\(.*", "", label)
            label = re.sub(r"[:=].*", "", label).strip()
            stages.append((f"[{m.group(1)}] {label}", float(m.group(3))))
        meta = {}
        tf = os.path.join(root, f"{subject}.timing.txt")
        if os.path.exists(tf):
            for line in open(tf):
                if "=" in line:
                    k, v = line.split("=", 1)
                    meta[k.strip()] = v.strip()
        runs[subject] = {"stages": stages, "meta": meta}
    return runs


def secs(v):
    return f"{v/60:.1f}m" if v >= 90 else f"{v:.1f}s"


def one(root):
    runs = load(root)
    for subject, r in runs.items():
        total = sum(s for _, s in r["stages"])
        m = r["meta"]
        print(f"\n=== {subject}   wall={secs(int(m.get('wall_seconds',0) or 0))}"
              f"  peak_vram={m.get('peak_vram_mib','?')} MiB  exit={m.get('exit','?')}")
        if not r["stages"]:
            print("    (no timed stages logged)")
            continue
        for label, s in sorted(r["stages"], key=lambda x: -x[1]):
            bar = "#" * max(1, round(40 * s / max(total, 1e-9)))
            print(f"    {label:52s}{secs(s):>8s}  {100*s/total:4.1f}%  {bar}")
        print(f"    {'TIMED TOTAL':52s}{secs(total):>8s}")


def ab(base_root, cand_root):
    base, cand = load(base_root), load(cand_root)
    print(f"baseline  = {os.path.basename(base_root)}")
    print(f"candidate = {os.path.basename(cand_root)}")
    for subject in sorted(set(base) & set(cand)):
        b, c = base[subject], cand[subject]
        bm, cm = b["meta"], c["meta"]
        print(f"\n=== {subject}")
        bw = int(bm.get("wall_seconds", 0) or 0); cw = int(cm.get("wall_seconds", 0) or 0)
        bv = int(bm.get("peak_vram_mib", 0) or 0); cv = int(cm.get("peak_vram_mib", 0) or 0)
        print(f"    {'wall clock':40s}{secs(bw):>10s}{secs(cw):>10s}"
              f"{(cw-bw)/max(bw,1)*100:+9.1f}%")
        print(f"    {'peak VRAM (MiB)':40s}{bv:>10d}{cv:>10d}{(cv-bv)/max(bv,1)*100:+9.1f}%")
        bs = dict(b["stages"]); cs = dict(c["stages"])
        print(f"    {'-'*40}{'baseline':>10s}{'cand':>10s}{'delta':>10s}")
        for label in sorted(set(bs) | set(cs), key=lambda k: -max(bs.get(k, 0), cs.get(k, 0))):
            x, y = bs.get(label), cs.get(label)
            if x is None or y is None:
                print(f"    {label:40s}{secs(x) if x else '-':>10s}{secs(y) if y else '-':>10s}"
                      f"{'only one':>10s}")
            else:
                print(f"    {label:40s}{secs(x):>10s}{secs(y):>10s}{(y-x)/max(x,1e-9)*100:+9.1f}%")
        tb, tc = sum(bs.values()), sum(cs.values())
        print(f"    {'TIMED TOTAL':40s}{secs(tb):>10s}{secs(tc):>10s}"
              f"{(tc-tb)/max(tb,1e-9)*100:+9.1f}%")


def main() -> int:
    if len(sys.argv) == 2:
        one(sys.argv[1])
    elif len(sys.argv) == 3:
        ab(sys.argv[1], sys.argv[2])
    else:
        print(__doc__, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
