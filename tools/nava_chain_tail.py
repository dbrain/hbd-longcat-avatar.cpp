#!/usr/bin/env python3
"""Extract clip-continuation anchors from a prior NAVA segment's dumped latents.

Reads the prior segment's final video latent (NAVA_DUMP_LATENT, ggml ne [W,H,F,48])
and/or final audio latent (NAVA_DUMP_AUDIO_LATENT, ggml ne [128,L]), and writes the
trailing N video frames / K audio tokens in the same .bin format, ready to feed the
next render via --video-anchor / --audio-anchor.

Usage:
  nava_chain_tail.py --vid <final_latent.bin> --n 1 --vid-out <anchor_vid.bin>
  nava_chain_tail.py --aud <audio_latent.bin> --k 13 --aud-out <anchor_aud.bin>
(either or both)

.bin layout: int32 n_dims,name_len,type(0=f32); int32 dims[n_dims] (ggml ne order,
ne[0] fastest); name bytes; f32 payload (ne[0] fastest).
"""
import argparse
import struct

import numpy as np


def read_bin(path):
    with open(path, "rb") as f:
        n_dims, name_len, typ = np.fromfile(f, np.int32, 3)
        dims = [int(x) for x in np.fromfile(f, np.int32, int(n_dims))]
        f.read(int(name_len))
        data = np.fromfile(f, np.float32)
    if typ != 0:
        raise ValueError(f"{path}: expected f32 type 0, got {typ}")
    return dims, data


def write_bin(path, dims, arr_c, name):
    """dims: ggml ne (ne[0] fastest). arr_c: numpy array in C-order == REVERSED ne."""
    payload = arr_c.astype("<f4").ravel(order="C")
    nb = name.encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<iii", len(dims), len(nb), 0))
        for d in dims:
            f.write(struct.pack("<i", int(d)))
        f.write(nb)
        f.write(payload.tobytes())
    print(f"wrote {path}  ne={dims}  ({payload.size} floats)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vid")
    ap.add_argument("--n", type=int, default=1, help="trailing video frames to keep")
    ap.add_argument("--vid-out")
    ap.add_argument("--aud")
    ap.add_argument("--k", type=int, default=0, help="trailing audio tokens to keep")
    ap.add_argument("--aud-out")
    args = ap.parse_args()

    if args.vid and args.vid_out:
        dims, data = read_bin(args.vid)  # ne [W,H,F,48]
        W, H, F, C = dims
        a = data.reshape(C, F, H, W)  # C-order == reversed ne
        n = max(1, min(args.n, F))
        tail = a[:, F - n:, :, :]  # [C,n,H,W]
        write_bin(args.vid_out, [W, H, n, C], tail, "video_anchor")
        print(f"  video: kept last {n}/{F} frames")

    if args.aud and args.aud_out:
        dims, data = read_bin(args.aud)  # ne [128,L]
        D, L = dims
        a = data.reshape(L, D)  # [L,128]
        k = max(1, min(args.k if args.k > 0 else L, L))
        tail = a[L - k:, :]  # [k,128]
        write_bin(args.aud_out, [D, k], tail, "audio_anchor")
        print(f"  audio: kept last {k}/{L} tokens")


if __name__ == "__main__":
    main()
