"""
Isolate the 3D-RoPE: take the C++ pre-rope q (after q_norm), apply the REFERENCE
rope_3d, and compare against the C++ post-rope q. If pre-rope matches the
reference q but post-rope diverges, the C++ RoPE convention is the bug.
"""
import os, sys, struct
import numpy as np
sys.path.insert(0, os.path.expanduser("~/dev/longcat-video-ref"))
import torch

D = "models/_dump"


def rd(f):
    with open(os.path.join(D, f), "rb") as fh:
        nd = struct.unpack("<q", fh.read(8))[0]
        dims = [struct.unpack("<q", fh.read(8))[0] for _ in range(nd)]
        data = np.frombuffer(fh.read(), dtype="<f4").copy()
    return data, dims


def cos(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


# ---- reference 3D RoPE (verbatim from modules/avatar/rope_3d.py) ----
from einops import rearrange, repeat


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


def precompute_freqs_cis_3d(grid_size, head_dim=128, base=10000):
    num_frames, height, width = grid_size
    dim_t = head_dim - 4 * (head_dim // 6)
    dim_h = 2 * (head_dim // 6)
    dim_w = 2 * (head_dim // 6)
    freqs_t = 1.0 / (base ** (torch.arange(0, dim_t, 2)[:(dim_t // 2)].float() / dim_t))
    freqs_h = 1.0 / (base ** (torch.arange(0, dim_h, 2)[:(dim_h // 2)].float() / dim_h))
    freqs_w = 1.0 / (base ** (torch.arange(0, dim_w, 2)[:(dim_w // 2)].float() / dim_w))
    grid_t = torch.from_numpy(np.linspace(0, num_frames, num_frames, endpoint=False, dtype=np.float32)).float()
    grid_h = torch.from_numpy(np.linspace(0, height, height, endpoint=False, dtype=np.float32)).float()
    grid_w = torch.from_numpy(np.linspace(0, width, width, endpoint=False, dtype=np.float32)).float()
    freqs_t = torch.einsum("..., f -> ... f", grid_t, freqs_t)
    freqs_h = torch.einsum("..., f -> ... f", grid_h, freqs_h)
    freqs_w = torch.einsum("..., f -> ... f", grid_w, freqs_w)
    freqs_t = repeat(freqs_t, "... n -> ... (n r)", r=2)
    freqs_h = repeat(freqs_h, "... n -> ... (n r)", r=2)
    freqs_w = repeat(freqs_w, "... n -> ... (n r)", r=2)
    # broadcast concat -> (T,H,W,D)
    ft = freqs_t[:, None, None, :].expand(num_frames, height, width, freqs_t.shape[-1])
    fh = freqs_h[None, :, None, :].expand(num_frames, height, width, freqs_h.shape[-1])
    fw = freqs_w[None, None, :, :].expand(num_frames, height, width, freqs_w.shape[-1])
    freqs = torch.cat([ft, fh, fw], dim=-1)
    freqs = rearrange(freqs, "T H W D -> (T H W) D")
    return freqs


def ref_rope(q, freqs):
    # q: [B, head, seq, head_dim]
    cos_, sin_ = freqs.cos(), freqs.sin()
    cos_ = rearrange(cos_, "n d -> 1 1 n d"); sin_ = rearrange(sin_, "n d -> 1 1 n d")
    return (q * cos_) + (rotate_half(q) * sin_)


def main():
    N_t, N_h, N_w = 2, 52, 30
    n_token = N_t * N_h * N_w
    head_dim, num_heads = 128, 32

    # C++ pre-rope q: ggml ne=[128, 32, 3120] -> np (3120, 32, 128) = (token, head, d)
    qpre, dpre = rd("sa_q_prerope.bin")
    qpre = qpre.reshape(n_token, num_heads, head_dim)
    # C++ post-rope q: ggml ne=[128, 3120, 32] -> np (32, 3120, 128) = (head, token, d)
    qpost, dpost = rd("sa_q_postrope.bin")
    qpost = qpost.reshape(num_heads, n_token, head_dim)
    print("pre dims(ggml)", dpre, "post dims(ggml)", dpost)

    # reference rope on the C++ pre-rope q
    q = torch.from_numpy(qpre).permute(1, 0, 2).unsqueeze(0).float()  # [1, head, token, d]
    freqs = precompute_freqs_cis_3d((N_t, N_h, N_w))
    qr = ref_rope(q, freqs)[0].numpy()  # [head, token, d]

    print("cos(cpp_postrope, ref_rope)        =", cos(qpost, qr))
    # also: does cpp post == cpp pre (i.e. is rope a no-op)?
    print("cos(cpp_postrope, cpp_prerope)     =", cos(qpost.transpose(1,0,2), qpre))

    # Try alternative conventions to identify the mismatch:
    # (A) NeoX / half-split rotate instead of interleaved
    def rotate_half_neox(x):
        d = x.shape[-1] // 2
        x1 = x[..., :d]; x2 = x[..., d:]
        return torch.cat([-x2, x1], dim=-1)
    def neox_freqs(grid_size, head_dim=128, base=10000):
        num_frames, height, width = grid_size
        dim_t = head_dim - 4*(head_dim//6); dim_h = 2*(head_dim//6); dim_w = 2*(head_dim//6)
        ft = 1.0/(base**(torch.arange(0,dim_t,2)[:dim_t//2].float()/dim_t))
        fh = 1.0/(base**(torch.arange(0,dim_h,2)[:dim_h//2].float()/dim_h))
        fw = 1.0/(base**(torch.arange(0,dim_w,2)[:dim_w//2].float()/dim_w))
        gt = torch.arange(num_frames).float(); gh=torch.arange(height).float(); gw=torch.arange(width).float()
        ft = torch.einsum("a,f->af", gt, ft); fh=torch.einsum("a,f->af",gh,fh); fw=torch.einsum("a,f->af",gw,fw)
        # NeoX: concat then duplicate halves [f, f]
        ft = ft[:,None,None,:].expand(num_frames,height,width,ft.shape[-1])
        fh = fh[None,:,None,:].expand(num_frames,height,width,fh.shape[-1])
        fw = fw[None,None,:,:].expand(num_frames,height,width,fw.shape[-1])
        f = torch.cat([ft,fh,fw],dim=-1)
        f = rearrange(f, "T H W D -> (T H W) D")
        return torch.cat([f, f], dim=-1)
    fz = neox_freqs((N_t,N_h,N_w))
    cosz, sinz = fz.cos(), fz.sin()
    cosz = rearrange(cosz,"n d -> 1 1 n d"); sinz=rearrange(sinz,"n d->1 1 n d")
    qz = (q*cosz + rotate_half_neox(q)*sinz)[0].numpy()
    print("cos(cpp_postrope, NEOX rope)       =", cos(qpost, qz))


if __name__ == "__main__":
    main()
