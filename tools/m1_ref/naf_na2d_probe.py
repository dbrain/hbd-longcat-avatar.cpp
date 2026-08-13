#!/usr/bin/env python3
"""
Pin down natten na2d (recent, cutlass-fna) border + indexing semantics for the NAF
conditioner, on the EXACT config that matters (512x512 grid, dilation 16, kernel 9x9),
with tiny channels so it's cheap + dumpable. Compares natten's na2d against an explicit
vectorized numpy gather (81 offsets) under candidate border modes:
  - "clamp": neighbor index clamped to [0,N) (natten classic sliding-window-inward feel,
    realized via per-position window-start clamp).
  - "mask": out-of-bounds neighbors excluded from softmax (renormalize over valid).
The matching mode is what the C++/ggml na2d must implement.

GPU (cutlass-fna needs CUDA):
  /mnt/hdd/3d/avatar-shootout/Pixal3D/.venv/bin/python naf_na2d_probe.py
"""
import numpy as np
import torch
from natten import na2d

H = W = 512
NH = 4
DQK = 8     # tiny head dim
DV = 8
KS = 9
DIL = 16
R = KS // 2  # 4
SCALE = DQK ** -0.5


def natten_ref(q, k, v):
    out = na2d(q, k, v, kernel_size=(KS, KS), dilation=(DIL, DIL), stride=1, backend="cutlass-fna")
    return out.detach().cpu().numpy()


def _neighbor_index(N, d, mode):
    """For each query position p in [0,N), return the K absolute neighbor indices [N,K]
    along one axis under `mode`. r=K//2.
      clamp: per-tap index clamp(p + d*(j-r), 0, N-1)        (repeats at border)
      shift: NATTEN slide-inward within dilation phase: phase=p%d, cls=p//d (L=N//d),
             center=clamp(cls, r, L-1-r), neighbor = phase + d*(center + j - r)  (no repeats)
      mask:  like clamp but caller marks OOB invalid
    """
    p = np.arange(N)
    out = np.zeros((N, KS), np.int64)
    valid = np.ones((N, KS), bool)
    if mode == "shift":
        L = N // d
        phase = p % d
        cls = p // d
        center = np.clip(cls, R, L - 1 - R)
        for j in range(KS):
            out[:, j] = phase + d * (center + (j - R))
    else:
        for j in range(KS):
            raw = p + d * (j - R)
            if mode == "mask":
                valid[:, j] = (raw >= 0) & (raw < N)
            out[:, j] = np.clip(raw, 0, N - 1)
    return out, valid


def numpy_na2d(q, k, v, mode):
    # q,k,v: [1,H,W,NH,D]
    q = q[0].numpy(); k = k[0].numpy(); v = v[0].numpy()
    iy, vy = _neighbor_index(H, DIL, mode)   # [H,KS]
    ix, vx = _neighbor_index(W, DIL, mode)   # [W,KS]
    out = np.zeros((H, W, NH, DV), np.float32)
    scores = np.full((H, W, NH, KS * KS), -1e30, np.float32)
    for a in range(KS):
        for b in range(KS):
            pos = a * KS + b
            ii = iy[:, a][:, None]            # [H,1]
            jj = ix[:, b][None, :]            # [1,W]
            kk = k[ii, jj]                     # [H,W,NH,D]
            s = (q * kk).sum(-1) * SCALE       # [H,W,NH]
            val = vy[:, a][:, None] & vx[:, b][None, :]
            s = np.where(val[..., None], s, -1e30)
            scores[..., pos] = s
    scores -= scores.max(-1, keepdims=True)
    w = np.exp(scores); w /= w.sum(-1, keepdims=True)
    for a in range(KS):
        for b in range(KS):
            pos = a * KS + b
            ii = iy[:, a][:, None]; jj = ix[:, b][None, :]
            out += w[..., pos:pos + 1] * v[ii, jj]
    return out[None]


def main():
    torch.manual_seed(0)
    dev = "cuda"
    q = torch.randn(1, H, W, NH, DQK, device=dev)
    k = torch.randn(1, H, W, NH, DQK, device=dev)
    v = torch.randn(1, H, W, NH, DV, device=dev)
    ref = natten_ref(q, k, v)
    print(f"[probe] natten out {ref.shape} range [{ref.min():.4f},{ref.max():.4f}]")
    qc, kc, vc = q.cpu(), k.cpu(), v.cpu()
    for mode in ("clamp", "mask", "shift"):
        mine = numpy_na2d(qc, kc, vc, mode)
        d = np.abs(mine - ref)
        # interior vs border breakdown
        bm = np.zeros((H, W), bool); bm[R*DIL:H-R*DIL, R*DIL:W-R*DIL] = True
        di = d[0][bm].max() if bm.any() else 0
        dbord = d[0][~bm].max()
        print(f"[probe] mode={mode:6s} maxabs={d.max():.3e} interior_max={di:.3e} border_max={dbord:.3e}")
    # also test dilation-1 small grid to sanity check pure indexing
    print("[probe] done — the mode with ~0 maxabs is the natten semantics to port.")


if __name__ == "__main__":
    main()
