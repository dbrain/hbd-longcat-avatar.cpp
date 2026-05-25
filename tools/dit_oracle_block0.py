"""
Focused block-0 oracle: load the int8 DiT (+DMD lora), then reproduce ONLY
block 0's forward (audio-skipped) on the C++-dumped inputs, comparing each
sub-step. Avoids the full 48-block forward (lighter on RAM / faster). Results
are appended to OUT with explicit flush so nothing is lost on a kill.
"""
import os, sys, struct
os.environ.setdefault("RANK", "0"); os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0"); os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29599")
import types, importlib.machinery as _M
import numpy as np

REF = os.path.expanduser("~/dev/longcat-video-ref")
sys.path.insert(0, REF)


def _stub(n, a=None):
    m = types.ModuleType(n); m.__spec__ = _M.ModuleSpec(n, loader=None); m.__version__ = "0"
    for k, v in (a or {}).items():
        setattr(m, k, v)
    sys.modules[n] = m; return m


_stub("longcat_video.block_sparse_attention", {"flash_attn_bsa_3d": None})
_stub("longcat_video.block_sparse_attention.bsa_interface", {"flash_attn_bsa_3d": None})
_stub("flash_attn", {"flash_attn_func": None, "flash_attn_varlen_func": None})
_stub("flash_attn_interface", {"flash_attn_func": None, "flash_attn_varlen_func": None})
_xf = _stub("xformers"); _xf.ops = _stub("xformers.ops")
_stub("longcat_video.audio_process", {"get_attn_map_with_target": None})
_stub("longcat_video.audio_process.torch_utils", {"get_attn_map_with_target": None})

import torch
import torch.nn.functional as F

CKPT = "/mnt/hdd/longcat/avatar-1.5"
D = "models/_dump"
OUT = open("/tmp/block0_oracle.txt", "w")


def log(*a):
    print(*a); print(*a, file=OUT); OUT.flush(); os.fsync(OUT.fileno())


def rd(f):
    with open(os.path.join(D, f), "rb") as fh:
        nd = struct.unpack("<q", fh.read(8))[0]
        dims = [struct.unpack("<q", fh.read(8))[0] for _ in range(nd)]
        data = np.frombuffer(fh.read(), dtype="<f4").copy()
    return data, dims


def diff(name, cpp, ref):
    a = cpp.astype(np.float64).ravel(); b = ref.astype(np.float64).ravel()
    if a.size != b.size:
        log(f"  [{name}] SIZE cpp={a.size} ref={b.size}"); return
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    log(f"  [{name:14s}] cos={cos:.6f} cpp.std={a.std():.4g} ref.std={b.std():.4g}")


def main():
    dtype = torch.float32
    from longcat_video.modules.quantization import load_quantized_dit
    log("[b0] loading int8 DiT...")
    prev = torch.get_default_dtype(); torch.set_default_dtype(torch.bfloat16)
    try:
        dit = load_quantized_dit(CKPT, subfolder="base_model_int8", cp_split_hw=[1, 1])
    finally:
        torch.set_default_dtype(prev)
    log("[b0] applying DMD lora...")
    dit.load_lora(os.path.join(CKPT, "lora", "dmd_lora.safetensors"), "dmd",
                  multiplier=1.0, lora_network_dim=128, lora_network_alpha=64)
    dit.enable_loras(["dmd"])
    # Keep loaded dtype (bf16 norms/biases, int8 quant weights) to avoid a full
    # fp32 RAM blowup. QuantizedLinear dequants to the fp32 input dtype per-call;
    # modulate_fp32 / LayerNorm_FP32 already compute in fp32.
    dit = dit.to("cpu").eval()

    # dense cross-attn patch
    from longcat_video.modules.attention import MultiHeadCrossAttention

    def dense_cross(self, x, cond, kv_seqlen):
        B, N, Cn = x.shape
        q = self.q_linear(x).view(1, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        q, k = self.q_norm(q), self.k_norm(k)
        q = q.permute(0, 2, 1, 3); k = k.permute(0, 2, 1, 3); v = v.permute(0, 2, 1, 3)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.permute(0, 2, 1, 3).reshape(B, -1, Cn)
        return self.proj(x)

    MultiHeadCrossAttention._process_cross_attn = dense_cross

    from longcat_video.modules.avatar.attention import Attention
    Attention._process_attn = lambda self, q, k, v, shape: F.scaled_dot_product_attention(q, k, v)

    # ---- inputs ----
    xpe, _ = rd("tap_patch_embed.bin")        # ggml [C, n_token]
    C, n_token = 4096, 3120; T = 2; ncl = 1
    N_t, N_h, N_w = 2, 52, 30
    x = torch.from_numpy(xpe.reshape(n_token, C)).to(dtype).unsqueeze(0)  # [1, N, C]
    # ref t-embed: feed dumped tap_t_embed directly to bypass embedding mismatch
    temb, td = rd("tap_t_embed.bin")          # ggml [512, T] -> (T,512)
    t = torch.from_numpy(temb.reshape(T, 512)).to(dtype).unsqueeze(0)     # [1, T, 512]
    # y_embed (already projected) as the cross-attn context [1, L, C]
    yemb, yd = rd("tap_y_embed.bin")          # ggml [C, L] -> (L, C)
    L = yd[1]
    y = torch.from_numpy(yemb.reshape(L, C)).to(dtype).reshape(1, L, C)
    y_seqlens = [L]

    blk = dit.blocks[0]
    import torch.amp as amp
    from longcat_video.modules.blocks import modulate_fp32

    with torch.no_grad():
        with amp.autocast(device_type="cpu", dtype=torch.float32):
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
                blk.adaLN_modulation(t).unsqueeze(2).chunk(6, dim=-1)
        log(f"[b0] scale_msa std={scale_msa.std():.4g} scale_mlp std={scale_mlp.std():.4g}")
        cpp_sc, _ = rd("b0_scale_msa.bin")
        diff("scale_msa", cpp_sc, scale_msa.float().cpu().numpy())

        B, N, _ = x.shape
        xn = blk.mod_norm_attn(x.view(B, T, -1, C).float())
        log(f"[b0] xn(mod_norm) per-token-std(over C) mean={xn.reshape(-1,C).std(dim=1).mean():.4g} "
            f"overall std={xn.std():.4g}")
        x_m = modulate_fp32(blk.mod_norm_attn, x.view(B, T, -1, C), shift_msa, scale_msa).view(B, N, C)
        log(f"[b0] x_m_attn ref std={x_m.std():.4g}")
        cpp_xm, _ = rd("b0_x_m_attn.bin")
        diff("x_m_attn", cpp_xm, x_m.float().cpu().numpy())

        # also recompute x_m manually the C++ way to cross-check
        scale_bc = scale_msa  # [1,T,1,C]
        xm_manual = (xn * (scale_bc + 1) + shift_msa).view(B, N, C)
        diff("x_m_manual", cpp_xm, xm_manual.float().cpu().numpy())

        # ---- reference self-attn qkv + q_norm (pre-rope), compare to C++ ----
        a = blk.attn
        qkv = a.qkv(x_m)  # [B, N, 3C]
        # save the raw reference qkv output + reference q (pre-norm) for offline
        # permutation analysis vs the C++ dumps.
        np.save(os.path.join(D, "ref_qkv.npy"), qkv[0].float().cpu().numpy())  # (N, 3C)
        qkv5 = qkv.view(B, N, 3, a.num_heads, a.head_dim).permute(2, 0, 3, 1, 4)
        qh, kh, vh = qkv5.unbind(0)  # each [B, H, N, d]
        np.save(os.path.join(D, "ref_q_raw.npy"), qh[0].permute(1, 0, 2).float().cpu().numpy())  # (N,H,d)
        qh = a.q_norm(qh); kh = a.k_norm(kh)
        # C++ sa_q_prerope ggml [d, H, token] -> np (token, H, d)
        cpp_qpre, _ = rd("sa_q_prerope.bin")
        cpp_qpre = cpp_qpre.reshape(N, a.num_heads, a.head_dim)        # (token, H, d)
        ref_qpre = qh[0].permute(1, 0, 2).float().cpu().numpy()        # (token, H, d)
        diff("q_prerope", cpp_qpre, ref_qpre)
        cpp_v, _ = rd("sa_v.bin")  # ggml [d, H, token, N] -> np (N, token, H, d)
        cpp_v = cpp_v.reshape(1, N, a.num_heads, a.head_dim)[0]        # (token, H, d)
        ref_v = vh[0].permute(1, 0, 2).float().cpu().numpy()
        diff("v", cpp_v, ref_v)

        # ---- full block-0 forward on dumped inputs (audio-skipped) ----
        latent_shape = (N_t, N_h, N_w)
        attn_out, _ = blk.attn(x_m, shape=latent_shape, num_cond_latents=ncl,
                               return_kv=False)
        log(f"[b0] attn_out ref std={attn_out.std():.4g}")
        cpp_ao, _ = rd("b0_attn_out.bin")
        diff("attn_out", cpp_ao, attn_out.float().cpu().numpy())

        xa = x + (gate_msa * attn_out.view(B, -1, N // T, C)).view(B, -1, C)
        cpp_aa, _ = rd("b0_after_attn.bin")
        diff("after_attn", cpp_aa, xa.float().cpu().numpy())

        # text cross attn (cond rows zeroed per num_cond_latents)
        xt = xa + blk.cross_attn(blk.pre_crs_attn_norm(xa), y, y_seqlens,
                                 num_cond_latents=ncl, shape=latent_shape)
        cpp_at, _ = rd("b0_after_text.bin")
        diff("after_text", cpp_at, xt.float().cpu().numpy())

        # ffn (swiglu) with mlp modulation
        x_m_ffn = modulate_fp32(blk.mod_norm_ffn, xt.view(B, -1, N // T, C),
                                shift_mlp, scale_mlp).view(B, -1, C)
        log(f"[b0] x_m_ffn ref std={x_m_ffn.std():.4g}")
        cpp_xmf, _ = rd("b0_x_m_ffn.bin")
        diff("x_m_ffn", cpp_xmf, x_m_ffn.float().cpu().numpy())

        gu = F.silu(blk.ffn.w1(x_m_ffn)) * blk.ffn.w3(x_m_ffn)
        log(f"[b0] ffn_gu ref std={gu.std():.4g} max={gu.abs().max():.4g}")
        cpp_gu, _ = rd("b0_ffn_gu.bin")
        diff("ffn_gu", cpp_gu, gu.float().cpu().numpy())

        ffn_out = blk.ffn.w2(gu)
        log(f"[b0] ffn_out ref std={ffn_out.std():.4g}")
        cpp_fo, _ = rd("b0_ffn_out.bin")
        diff("ffn_out", cpp_fo, ffn_out.float().cpu().numpy())

        xf = xt + (gate_mlp * ffn_out.view(B, -1, N // T, C)).view(B, -1, C)
        log(f"[b0] after_ffn ref std={xf.std():.4g}")
        cpp_af, _ = rd("b0_after_ffn.bin")
        diff("after_ffn", cpp_af, xf.float().cpu().numpy())

    log("[b0] DONE")
    OUT.close()


if __name__ == "__main__":
    main()
