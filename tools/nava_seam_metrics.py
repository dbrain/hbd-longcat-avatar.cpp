#!/usr/bin/env python3
"""
Automated continuation-seam cleanliness probe for NAVA multi-segment chains.

The meatbag eye-test is the final judge; this is a PRE-FILTER to rank warm-start
strengths objectively before burning eyeballs. Audio numbers lie, but VIDEO
lighting / sharpness / motion discontinuity at the seam are honest flags.

Given the per-segment clips (in temporal order), it decodes frames losslessly,
concatenates them, locates each seam (frame boundary between segment k and k+1),
and reports, for every seam:

  motion delta  d[i] = mean|frame[i+1]-frame[i]|  (luma)         -> motion continuity
  luma  L[i]    = mean luma per frame                            -> lighting/exposure jump
  sharp s[i]    = var(Laplacian) per frame                       -> focus pop / "plastic" drift

Headline per seam:
  seam_jump   = d[seam] / median(intra d)        ~1 good, >>1 = hard cut
  jerk        = how much the local motion curve kinks at the seam (2nd-diff z-score)
  luma_jump   = |L[seam+1]-L[seam]| / median(intra |dL|)
  sharp_step  = median(sharp seg_{k+1}) / median(sharp seg_k)   1=stable, >1 plastic drift

Also writes a horizontal filmstrip PNG (last 2 frames of seg k + first 2 of seg k+1)
per seam so the cut is eyeballable at a glance.

Usage:
  python3 tools/nava_seam_metrics.py --label warm0.7 \
      --out /mnt/hdd/nava/warm_exp/metrics/warm0.7 \
      seg0.webm seg1.webm [seg2.webm ...]
"""
import argparse, os, subprocess, sys, tempfile, json
import numpy as np
from PIL import Image


def decode_frames(clip):
    """Decode all frames of a clip to a uint8 [N,H,W,3] array (lossless rawvideo)."""
    d = tempfile.mkdtemp(prefix="seamfr_")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", clip, "-pix_fmt", "rgb24",
         os.path.join(d, "f%04d.png")],
        check=True)
    files = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".png"))
    arr = np.stack([np.asarray(Image.open(f).convert("RGB")) for f in files]).astype(np.float32)
    for f in files:
        os.remove(f)
    os.rmdir(d)
    return arr  # [N,H,W,3]


def luma(frames):
    return frames @ np.array([0.299, 0.587, 0.114], dtype=np.float32)  # [N,H,W]


def lap_var(gray):
    """var of 4-neighbour Laplacian, per frame. gray [N,H,W]."""
    lp = (-4 * gray[:, 1:-1, 1:-1]
          + gray[:, :-2, 1:-1] + gray[:, 2:, 1:-1]
          + gray[:, 1:-1, :-2] + gray[:, 1:-1, 2:])
    return lp.reshape(lp.shape[0], -1).var(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--label", default="chain")
    ap.add_argument("--out", default=None, help="output dir for filmstrip + metrics.json")
    ap.add_argument("--drop-head", type=int, default=0,
                    help="drop first N pixel frames of every clip AFTER the first "
                         "(the warm-start overlap region; 1+(O-1)*4 for latent overlap O)")
    args = ap.parse_args()

    segs = [decode_frames(c) for c in args.clips]
    if args.drop_head > 0:
        segs = [segs[0]] + [s[args.drop_head:] for s in segs[1:]]
    seg_lens = [s.shape[0] for s in segs]
    frames = np.concatenate(segs, axis=0)         # [Ntot,H,W,3]
    g = luma(frames)
    N = frames.shape[0]

    d = np.abs(g[1:] - g[:-1]).reshape(N - 1, -1).mean(axis=1)   # motion delta, len N-1
    L = g.reshape(N, -1).mean(axis=1)                            # luma, len N
    dL = np.abs(L[1:] - L[:-1])                                  # luma delta
    sharp = lap_var(g)                                          # len N

    # seam indices: boundary i means frames[i] is last of seg k, frames[i+1] first of seg k+1
    seams = np.cumsum(seg_lens)[:-1] - 1   # index of the LAST frame of each non-final segment

    # intra-segment deltas (exclude the seam transitions themselves)
    seam_set = set(int(s) for s in seams)
    intra_d = np.array([d[i] for i in range(len(d)) if i not in seam_set])
    intra_dL = np.array([dL[i] for i in range(len(dL)) if i not in seam_set])
    med_d = float(np.median(intra_d)) if len(intra_d) else 0.0
    med_dL = float(np.median(intra_dL)) if len(intra_dL) else 0.0

    report = {"label": args.label, "seg_lens": seg_lens, "n_frames": int(N),
              "median_intra_motion": med_d, "median_intra_luma_step": med_dL,
              "seams": []}

    for si, seam in enumerate(seams):
        seam = int(seam)
        # motion jerk: 2nd difference of d around the seam, z-scored by intra std
        lo, hi = max(0, seam - 3), min(len(d), seam + 4)
        win = d[lo:hi]
        d2 = np.abs(np.diff(win, n=2)) if len(win) >= 3 else np.array([0.0])
        jerk = float(d2.max() / (intra_d.std() + 1e-6))
        seg_k = sharp[seam - seg_lens[si] + 1: seam + 1]
        seg_k1 = sharp[seam + 1: seam + 1 + seg_lens[si + 1]]
        report["seams"].append({
            "seam_index": seam,
            "seam_jump_ratio": float(d[seam] / (med_d + 1e-6)),
            "jerk_z": jerk,
            "luma_jump_ratio": float(dL[seam] / (med_dL + 1e-6)),
            "sharp_step_ratio": float(np.median(seg_k1) / (np.median(seg_k) + 1e-6)),
            "motion_window": [round(float(x), 3) for x in d[lo:hi]],
            "sharp_seg_k_med": float(np.median(seg_k)),
            "sharp_seg_k1_med": float(np.median(seg_k1)),
        })

    # one combined seam score (lower=smoother): geometric-ish blend, |.-1| penalties
    sc = 0.0
    for s in report["seams"]:
        sc += abs(s["seam_jump_ratio"] - 1.0) + 0.5 * s["jerk_z"] \
              + abs(s["luma_jump_ratio"] - 1.0) + abs(s["sharp_step_ratio"] - 1.0)
    report["seam_score"] = round(sc / max(1, len(report["seams"])), 4)

    print(json.dumps(report, indent=2))

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        json.dump(report, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
        # filmstrip: [seg k -2,-1 | seg k+1 0,1] per seam, stacked vertically
        rows = []
        for seam in seams:
            seam = int(seam)
            idx = [seam - 1, seam, seam + 1, seam + 2]
            idx = [i for i in idx if 0 <= i < N]
            strip = np.concatenate([frames[i].astype(np.uint8) for i in idx], axis=1)
            rows.append(strip)
        film = np.concatenate(rows, axis=0) if rows else frames[0].astype(np.uint8)
        Image.fromarray(film).save(os.path.join(args.out, "seam_filmstrip.png"))
        print(f"\nfilmstrip -> {os.path.join(args.out, 'seam_filmstrip.png')}", file=sys.stderr)


if __name__ == "__main__":
    main()
