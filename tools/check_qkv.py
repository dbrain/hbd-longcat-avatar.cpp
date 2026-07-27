"""
Memory-light block-0 qkv check: load ONLY block-0 attn weights from the int8
shard (+ fold the DMD lora), compute qkv on the C++-dumped x_m, split q/k/v the
REFERENCE way, and compare to the C++ sa_q_prerope / sa_v dumps. Pinpoints the
qkv/head-split layout bug without constructing the 27GB full model.
"""
import os, struct, json
import numpy as np
import torch
from safetensors.torch import load_file

_OUT = open("/tmp/qkv_result.txt", "w")
_print = print


def print(*a, **k):  # noqa: A001 - flush to a file so nothing is lost
    _print(*a, **k)
    _print(*a, file=_OUT, **k); _OUT.flush(); os.fsync(_OUT.fileno())

QD = "/mnt/hdd/longcat/avatar-1.5/base_model_int8"
LORA = "/mnt/hdd/longcat/avatar-1.5/lora/dmd_lora.safetensors"
D = "models/_dump"
H, d = 32, 128
C = H * d


def rd(f):
    with open(os.path.join(D, f), "rb") as fh:
        nd = struct.unpack("<q", fh.read(8))[0]
        dims = [struct.unpack("<q", fh.read(8))[0] for _ in range(nd)]
        return np.frombuffer(fh.read(), dtype="<f4").copy(), dims


def cos(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)


def main():
    sd = load_file(os.path.join(QD, "quantized_model-00001-of-00004.safetensors"), device="cpu")
    wi = sd["blocks.0.attn.qkv.weight_int8"].float()      # [3C, C]
    ws = sd["blocks.0.attn.qkv.weight_scale"].float()     # [3C]
    W = wi * ws[:, None]                                  # [3C, C] dequant
    b = sd["blocks.0.attn.qkv.bias"].float()              # [3C]
    qn_w = sd["blocks.0.attn.q_norm.weight"].float()      # [d]
    kn_w = sd["blocks.0.attn.k_norm.weight"].float()

    # fold DMD lora into W: W += multiplier(1.0) * alpha_scale(0.5) * (up @ down),
    # n_seperate=3 block-diagonal (qkv).
    lf = load_file(LORA, device="cpu")
    pre = "lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___attn___lorahyphen___qkv"
    down = lf[pre + ".lora_down.weight"].float()          # [3*128, C]
    alpha_scale = float(lf[pre + ".alpha_scale"])         # 0.5
    print("alpha_scale", alpha_scale, "down", tuple(down.shape))
    dim = 128
    delta = torch.zeros_like(W)
    for i in range(3):
        up_i = lf[pre + f".lora_up.blocks.{i}.weight"].float()   # [C, 128]
        down_i = down[i * dim:(i + 1) * dim]                     # [128, C]
        delta[i * C:(i + 1) * C] = up_i @ down_i                 # [C, C]
    W = W + 1.0 * alpha_scale * delta

    # x_m from C++ dump (matched the reference cos 1.0): ggml [C, n_token] -> (n_token, C)
    xm, xd = rd("b0_x_m_attn.bin")
    n_token = xd[1]
    x = torch.from_numpy(xm.reshape(n_token, C))          # (N, C)

    qkv = x @ W.T + b                                     # (N, 3C)

    def rmsnorm(t, w, eps=1e-6):
        # t: (..., d)
        return t / torch.sqrt((t ** 2).mean(-1, keepdim=True) + eps) * w

    # ---- REFERENCE split: view(N, 3, H, d) ----
    qkv5 = qkv.view(n_token, 3, H, d)
    qr, kr, vr = qkv5[:, 0], qkv5[:, 1], qkv5[:, 2]       # each (N, H, d)
    qr_n = rmsnorm(qr, qn_w)                               # (N, H, d)

    cpp_q, _ = rd("sa_q_prerope.bin")  # ggml [d,H,token] -> (token,H,d)
    cpp_q = cpp_q.reshape(n_token, H, d)
    cpp_v, _ = rd("sa_v.bin")          # ggml [d,H,token,N] -> (N,token,H,d)->(token,H,d)
    cpp_v = cpp_v.reshape(1, n_token, H, d)[0]

    print("=== REFERENCE split view(N,3,H,d) ===")
    print("  q_prerope cos:", cos(cpp_q, qr_n.numpy()))
    print("  v         cos:", cos(cpp_v, vr.numpy()))

    # ---- ALT split A: view(N, 3, d, H) (head_dim/heads swapped within each third) ----
    qkv5a = qkv.view(n_token, 3, d, H)
    qa, va = qkv5a[:, 0], qkv5a[:, 2]                      # (N, d, H)
    qa = qa.permute(0, 2, 1); va = va.permute(0, 2, 1)    # (N, H, d)
    qa_n = rmsnorm(qa, qn_w)
    print("=== ALT-A view(N,3,d,H) ===")
    print("  q_prerope cos:", cos(cpp_q, qa_n.numpy()))
    print("  v         cos:", cos(cpp_v, va.numpy()))

    # ---- ALT split B: interleaved thirds view(N, C, 3) (q=::3) ----
    qkv_i = qkv.view(n_token, C, 3)
    qb = qkv_i[:, :, 0].reshape(n_token, H, d)
    vb = qkv_i[:, :, 2].reshape(n_token, H, d)
    qb_n = rmsnorm(qb, qn_w)
    print("=== ALT-B interleaved view(N,C,3) q=[:,:,0] ===")
    print("  q_prerope cos:", cos(cpp_q, qb_n.numpy()))
    print("  v         cos:", cos(cpp_v, vb.numpy()))

    # ---- raw correlation: does C++ qkv row order differ? Check q vs each third ----
    for name, third in [("first", qkv5[:, 0]), ("second", qkv5[:, 1]), ("third", qkv5[:, 2])]:
        print(f"  cpp_v vs ref {name}-third (N,H,d) cos:", cos(cpp_v, third.numpy()))

    # ====================================================================
    # FULL block-0 self-attention from standalone weights (REFERENCE math),
    # compare to C++ b0_attn_out (post-proj). This is the reliable check
    # (independent weight load + C++ x_m input).
    # ====================================================================
    print("\n=== FULL self-attn (standalone, ref math) vs C++ ===")
    N_t, N_h, N_w = 2, 52, 30
    ncl_thw = 1 * (n_token // N_t)  # 1560
    kr_n = rmsnorm(kr, kn_w)        # (N, H, d)
    # 3D rope (interleaved) on q,k
    from einops import rearrange, repeat

    def rope3d_freqs():
        base = 10000
        dim_t = d - 4 * (d // 6); dim_h = 2 * (d // 6); dim_w = 2 * (d // 6)
        ft = 1.0 / (base ** (torch.arange(0, dim_t, 2)[:dim_t // 2].float() / dim_t))
        fh = 1.0 / (base ** (torch.arange(0, dim_h, 2)[:dim_h // 2].float() / dim_h))
        fw = 1.0 / (base ** (torch.arange(0, dim_w, 2)[:dim_w // 2].float() / dim_w))
        gt = torch.arange(N_t).float(); gh = torch.arange(N_h).float(); gw = torch.arange(N_w).float()
        ft = torch.einsum("a,f->af", gt, ft); fh = torch.einsum("a,f->af", gh, fh); fw = torch.einsum("a,f->af", gw, fw)
        ft = repeat(ft, "a n -> a (n r)", r=2); fh = repeat(fh, "a n -> a (n r)", r=2); fw = repeat(fw, "a n -> a (n r)", r=2)
        ft = ft[:, None, None, :].expand(N_t, N_h, N_w, ft.shape[-1])
        fh = fh[None, :, None, :].expand(N_t, N_h, N_w, fh.shape[-1])
        fw = fw[None, None, :, :].expand(N_t, N_h, N_w, fw.shape[-1])
        f = torch.cat([ft, fh, fw], dim=-1)
        return rearrange(f, "T H W D -> (T H W) D")  # (n_token, d)

    def rot_half(x):
        x = rearrange(x, "... (e r) -> ... e r", r=2)
        x1, x2 = x.unbind(-1)
        return rearrange(torch.stack((-x2, x1), -1), "... e r -> ... (e r)")

    freqs = rope3d_freqs()
    cosf = freqs.cos()[None]; sinf = freqs.sin()[None]   # (1, n_token, d)
    # q,k: (N_token, H, d) -> (H, N_token, d)
    qh = qr_n.permute(1, 0, 2); kh = kr_n.permute(1, 0, 2); vh = vr.permute(1, 0, 2)
    qh = qh * cosf + rot_half(qh) * sinf
    kh = kh * cosf + rot_half(kh) * sinf

    def attn(q, k, v):
        s = (q @ k.transpose(1, 2)) / np.sqrt(d)
        s = s - s.max(-1, keepdim=True).values
        w = torch.softmax(s, -1)
        return w @ v

    # cond split: q_cond x {k,v}_cond ; q_noise x {k,v}_full
    xc = attn(qh[:, :ncl_thw], kh[:, :ncl_thw], vh[:, :ncl_thw])
    xnz = attn(qh[:, ncl_thw:], kh, vh)
    xo = torch.cat([xc, xnz], dim=1)                     # (H, n_token, d)
    # back to (n_token, H*d) head-major, then proj
    xo = xo.permute(1, 0, 2).reshape(n_token, H * d)
    proj_wi = sd["blocks.0.attn.proj.weight_int8"].float()
    proj_ws = sd["blocks.0.attn.proj.weight_scale"].float()
    proj_w = proj_wi * proj_ws[:, None]
    proj_b = sd["blocks.0.attn.proj.bias"].float()
    # fold proj lora
    pp = "lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___attn___lorahyphen___proj"
    if pp + ".lora_down.weight" in lf:
        pd = lf[pp + ".lora_down.weight"].float(); pu = lf[pp + ".lora_up.weight"].float()
        proj_w = proj_w + 0.5 * (pu @ pd)
    attn_out = xo @ proj_w.T + proj_b
    cpp_ao, _ = rd("b0_attn_out.bin")
    print("  attn_out cos:", cos(cpp_ao, attn_out.numpy()),
          "std cpp", float(cpp_ao.std()), "ref", float(attn_out.std()))

    # ====================================================================
    # FFN (SwiGLU) from standalone weights on the C++-dumped x_m_ffn input.
    # Isolates whether the C++ FFN explosion (gu std 230, out std 1373) is
    # genuine model behaviour or a C++ bug.
    # ====================================================================
    print("\n=== FFN (standalone) on C++ x_m_ffn vs C++ ===")

    def lin(name, x_in, with_lora=True):
        wi = sd[f"blocks.0.ffn.{name}.weight_int8"].float()
        ws = sd[f"blocks.0.ffn.{name}.weight_scale"].float()
        W_ = wi * ws[:, None]
        lp = f"lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___ffn___lorahyphen___{name}"
        if with_lora and lp + ".lora_down.weight" in lf:
            ld = lf[lp + ".lora_down.weight"].float(); lu = lf[lp + ".lora_up.weight"].float()
            W_ = W_ + 0.5 * (lu @ ld)
        return x_in @ W_.T

    xmf, xfd = rd("b0_x_m_ffn.bin")          # ggml [C, n_token] -> (n_token, C)
    xmf = torch.from_numpy(xmf.reshape(n_token, C))
    g = torch.nn.functional.silu(lin("w1", xmf))
    u = lin("w3", xmf)
    gu = g * u
    cpp_gu, _ = rd("b0_ffn_gu.bin")
    print("  ffn_gu  cos:", cos(cpp_gu, gu.numpy()), "std cpp", float(cpp_gu.std()), "ref", float(gu.std()))
    fo = lin("w2", gu)
    cpp_fo, _ = rd("b0_ffn_out.bin")
    print("  ffn_out cos:", cos(cpp_fo, fo.numpy()), "std cpp", float(cpp_fo.std()), "ref", float(fo.std()))

    # ====================================================================
    # End-to-end block-0 residual chain from the C++ inputs, compare to the
    # C++ tap_block0 (b0_after_ffn). Validates the residual adds + gates.
    # x0 = tap_patch_embed ; reuse the dumped gate_msa/gate_mlp.
    # ====================================================================
    print("\n=== block-0 residual chain (standalone) vs C++ tap_block0 ===")
    T = N_t
    x0n, x0d = rd("tap_patch_embed.bin")        # (n_token, C)
    x0 = torch.from_numpy(x0n.reshape(n_token, C))
    gmsa, _ = rd("b0_gate_msa.bin")             # ggml [C,1,T] -> (T, C)
    gmsa = torch.from_numpy(gmsa.reshape(T, C))
    gmlp, _ = rd("b0_gate_mlp.bin")
    gmlp = torch.from_numpy(gmlp.reshape(T, C))
    spatial = n_token // T
    frame = torch.arange(n_token) // spatial
    xa = x0 + gmsa[frame] * attn_out            # after_attn
    cpp_aa, _ = rd("b0_after_attn.bin")
    print("  after_attn cos:", cos(cpp_aa, xa.numpy()), "std cpp", float(cpp_aa.std()), "ref", float(xa.std()))
    # text cross attn output already folded into C++ b0_after_text; here just
    # validate after_ffn = after_text + gate_mlp*ffn_out using C++ after_text.
    cpp_at, _ = rd("b0_after_text.bin")
    xt = torch.from_numpy(cpp_at[0].reshape(n_token, C) if False else cpp_at.reshape(n_token, C))
    xf = xt + gmlp[frame] * fo
    cpp_b0, _ = rd("tap_block0.bin")
    print("  after_ffn(tap_block0) cos:", cos(cpp_b0, xf.numpy()),
          "std cpp", float(cpp_b0.std()), "ref", float(xf.std()))


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()
