#!/usr/bin/env python3
"""
Convert the PyTorch reference npz (from NAVA/nava_dump_ref.py) into the exact
input .bin files that examples/nava/main.cpp reads via sd.cpp
`load_tensor_from_file_as_tensor`.

HEADER format (little-endian):
  int32 n_dims, int32 name_len, int32 ggml_type(=0 for F32),
  int32 dim[n_dims]  (ggml ne-order, ne[0] = fastest-varying),
  char  name[name_len],
  then the f32 payload, ne[0]-fastest.

INPUT bins the harness expects (ggml ne-order):
  video.bin    [W, H, F, 48]    pre-patchify video latent
  audio.bin    [128, Lraw]      audio latent (channels-first)
  context.bin  [3072, 512]      POST text_embedding context (harness passes
                                through when ne[0]==3072), OR [4096,512] raw
  timestep.bin [1]              scalar diffusion timestep

npz key layouts (torch, C-contiguous => last axis is fastest = ggml ne[0]):
  input_vid              [48, F, H, W]   -> ne [W,H,F,48]  (same bytes)
  input_audio            [L, 128]        -> ne [128, L]    (same bytes)
  context_vid            [512, 3072]     -> ne [3072, 512] (same bytes)  [post-embed]
  input_vid_context_raw  [L, 4096]       -> ne [4096, L]   (raw umT5)
  input_t                [1]             -> ne [1]

Usage:
  python3 tools/nava_npz_to_bin.py <ref_tensors.npz> <out_dir> [--raw-context] [--i2v-clean-timestep]
"""
import sys, os, argparse
import numpy as np

GGML_TYPE_F32 = 0


def write_bin(path, arr, ne_dims, name):
    """arr: C-contiguous float32 whose flat byte order already equals ne[0]-fastest.
    ne_dims: list of ints in ggml ne-order (ne[0] first)."""
    arr = np.ascontiguousarray(arr.astype(np.float32))
    assert arr.size == int(np.prod(ne_dims)), (
        f"{name}: {arr.size} elems vs ne {ne_dims} ({np.prod(ne_dims)})")
    with open(path, "wb") as f:
        np.array([len(ne_dims)], dtype=np.int32).tofile(f)
        np.array([len(name)], dtype=np.int32).tofile(f)
        np.array([GGML_TYPE_F32], dtype=np.int32).tofile(f)
        np.array(ne_dims, dtype=np.int32).tofile(f)
        f.write(name.encode("utf-8"))
        arr.tofile(f)
    print(f"wrote {path}  ne={ne_dims}  ({arr.size} f32)")


def squeeze_lead_batch(a):
    # drop a leading singleton batch dim if present
    while a.ndim > 1 and a.shape[0] == 1:
        a = a[0]
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("outdir")
    ap.add_argument("--raw-context", action="store_true",
                    help="emit raw umT5 [4096,512] context instead of post-embed [3072,512]")
    ap.add_argument("--i2v-clean-timestep", action="store_true",
                    help="emit per-token timestep with video frame 0 set to 0 when meta_first_frame_is_clean is true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    z = np.load(args.npz, allow_pickle=True)

    def get(k):
        return np.asarray(z[k])

    # ---- video: input_vid [48,F,H,W] -> ne [W,H,F,48] (identical bytes) ----
    vid = get("input_vid")          # [48, F, H, W] (or [1,48,F,H,W])
    vid = squeeze_lead_batch(vid)
    assert vid.ndim == 4, f"input_vid ndim {vid.shape}"
    C, F, H, W = vid.shape
    write_bin(os.path.join(args.outdir, "video.bin"), vid, [W, H, F, C], "video")

    # ---- audio: input_audio [L,128] -> ne [128, L] (identical bytes) ----
    aud = get("input_audio")        # [L, 128]
    aud = squeeze_lead_batch(aud)
    assert aud.ndim == 2, f"input_audio ndim {aud.shape}"
    L, AC = aud.shape
    write_bin(os.path.join(args.outdir, "audio.bin"), aud, [AC, L], "audio")

    # ---- context ----
    if args.raw_context:
        ctx = get("input_vid_context_raw")  # [Ltok, 4096]
        ctx = squeeze_lead_batch(ctx)
        Lt, D = ctx.shape
        write_bin(os.path.join(args.outdir, "context.bin"), ctx, [D, Lt], "context")
    else:
        ctx = get("context_vid")            # [512, 3072] (post text_embedding)
        ctx = squeeze_lead_batch(ctx)
        assert ctx.ndim == 2, f"context_vid ndim {ctx.shape}"
        Lt, D = ctx.shape
        write_bin(os.path.join(args.outdir, "context.bin"), ctx, [D, Lt], "context")

    # ---- timestep ----
    t = get("input_t").astype(np.float32).reshape(-1)
    if args.i2v_clean_timestep:
        first_clean = bool(np.asarray(z.get("meta_first_frame_is_clean", [0.0])).reshape(-1)[0] > 0.5)
        if not first_clean:
            raise SystemExit("--i2v-clean-timestep requested but meta_first_frame_is_clean is false/missing")
        if C != 48:
            raise SystemExit(f"--i2v-clean-timestep expects video C=48, got {C}")
        L_vid = F * (H // 2) * (W // 2)
        frame_tokens = (H // 2) * (W // 2)
        ts = np.full((L_vid + L,), t[0], dtype=np.float32)
        ts[:frame_tokens] = 0.0
        write_bin(os.path.join(args.outdir, "timestep.bin"), ts, [ts.size], "timestep")
    else:
        write_bin(os.path.join(args.outdir, "timestep.bin"), t[:1], [1], "timestep")

    print("\nDerived grid:")
    print(f"  video ne [W={W}, H={H}, F={F}, C={C}]  -> L_vid = F*(H/2)*(W/2) = "
          f"{F * (H // 2) * (W // 2)}")
    print(f"  audio ne [128, L={L}]                  -> L_audio = {L}")
    print(f"  context ne [{D}, {Lt}]")


if __name__ == "__main__":
    main()
