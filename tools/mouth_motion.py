#!/usr/bin/env python3
"""Quantify lip-sync: per-frame motion in a lower-face (mouth) ROI vs a control
(forehead/background) ROI, correlated with the audio RMS envelope. If the mouth
ROI moves with the audio and far more than the control ROI, the audio graft is
driving the mouth."""
import sys, glob, os, wave
import numpy as np
from PIL import Image


def frames_motion(d):
    paths = sorted(glob.glob(os.path.join(d, "*.png")), key=lambda p: (len(p), p))
    imgs = [np.asarray(Image.open(p).convert("L")).astype(np.float32) for p in paths]
    H, W = imgs[0].shape
    # mouth ROI: lower-center; control ROI: top strip (forehead/background)
    my0, my1, mx0, mx1 = int(0.60 * H), int(0.82 * H), int(0.30 * W), int(0.70 * W)
    cy0, cy1, cx0, cx1 = int(0.02 * H), int(0.18 * H), int(0.20 * W), int(0.80 * W)
    mouth, ctrl = [], []
    for a, b in zip(imgs[:-1], imgs[1:]):
        dmap = np.abs(b - a)
        mouth.append(dmap[my0:my1, mx0:mx1].mean())
        ctrl.append(dmap[cy0:cy1, cx0:cx1].mean())
    return np.array(mouth), np.array(ctrl), len(imgs)


def audio_env(wav_path, n_trans, fps):
    w = wave.open(wav_path)
    sr, n = w.getframerate(), w.getnframes()
    raw = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32)
    if w.getnchannels() == 2:
        raw = raw.reshape(-1, 2).mean(1)
    spf = sr / fps
    env = []
    for i in range(n_trans):
        s, e = int(i * spf), int((i + 1) * spf)
        seg = raw[s:e]
        env.append(np.sqrt((seg ** 2).mean()) if len(seg) else 0.0)
    return np.array(env)


def main():
    d, wav = sys.argv[1], sys.argv[2]
    fps = float(sys.argv[3]) if len(sys.argv) > 3 else 25.0
    mouth, ctrl, nf = frames_motion(d)
    env = audio_env(wav, len(mouth), fps)

    def z(x):
        return (x - x.mean()) / (x.std() + 1e-8)
    corr = float((z(mouth) * z(env)).mean())
    ratio = mouth.mean() / (ctrl.mean() + 1e-8)
    print(f"{d}: {nf} frames, fps={fps}")
    print(f"  mouth-ROI motion mean={mouth.mean():.2f}  control-ROI mean={ctrl.mean():.2f}  "
          f"mouth/control={ratio:.2f}x")
    print(f"  corr(mouth_motion, audio_rms) = {corr:+.3f}")
    print("  frame:  mouth  ctrl  audio")
    for i, (m, c, e) in enumerate(zip(mouth, ctrl, env)):
        bar = "#" * int(m / (mouth.max() + 1e-8) * 30)
        print(f"   {i:3d}  {m:6.2f} {c:5.2f}  {e:6.0f}  {bar}")


main()
