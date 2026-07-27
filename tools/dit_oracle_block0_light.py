"""
LIGHT block-0 oracle: construct a SINGLE LongCatAvatarSingleStreamBlock and load
ONLY its weights (+ the embedder taps are read from the C++ dump directly), so we
never materialize the full 48-block / 13.6B model in host RAM (~1-2 GB peak here).

Block-0 weights are dequantized straight from the int8 safetensors shards
(w = int8 * scale) and the DMD LoRA delta is folded in, reusing the Safetensors /
DmdLora readers from tools/convert_longcat_avatar.py. Forward + diff logic mirrors
tools/dit_oracle_block0.py (which loaded the full model). Results stream to OUT
with fsync so a kill loses nothing.

Run UNDER A MEMORY CAP (one heavy job at a time):
    TMPDIR=$PWD/.convert-tmp systemd-run --user --scope \
        -p MemoryMax=14G -p MemorySwapMax=0 nice -n 15 \
        .venv-oracle/bin/python tools/dit_oracle_block0_light.py
"""
import os, sys, struct
os.environ.setdefault("RANK", "0"); os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0"); os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29599")
import types, importlib.machinery as _M
import numpy as np

REF = os.path.expanduser("~/dev/longcat-video-ref")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REF)
sys.path.insert(0, HERE)


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

from convert_longcat_avatar import Safetensors, DmdLora, load_index

CKPT = "/mnt/hdd/longcat/avatar-1.5"
D = os.path.join(os.path.dirname(HERE), "models", "_dump")
OUT = open("/tmp/block0_oracle_light.txt", "w")


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
    denom = np.maximum(np.abs(b), 1e-3)
    maxrel = float(np.max(np.abs(a - b) / denom))
    mae = float(np.max(np.abs(a - b)))
    log(f"  [{name:14s}] cos={cos:.6f} maxrel={maxrel:.3e} mae={mae:.4e} | "
        f"cpp(m={a.mean():+.4g},s={a.std():.4g}) ref(m={b.mean():+.4g},s={b.std():.4g})")


def build_block0():
    """Construct one avatar block and load block-0 weights from the int8 shards."""
    from longcat_video.modules.avatar.longcat_video_dit_avatar import LongCatAvatarSingleStreamBlock
    # Match the full model construction (longcat_video_dit_avatar.py L249-266):
    # hidden 4096, heads 32, mlp_ratio 4 (-> SwiGLU inner 11008), adaln 512,
    # output_dim 768, audio_prenorm False (avatar default).
    blk = LongCatAvatarSingleStreamBlock(
        hidden_size=4096, num_heads=32, mlp_ratio=4, adaln_tembed_dim=512,
        cp_split_hw=[1, 1],
        output_dim=768, audio_prenorm=False, class_range=24, class_interval=4,
    ).eval()

    name2shard = load_index(os.path.join(CKPT, "base_model_int8"))
    shards = {}

    def shard(p):
        if p not in shards:
            shards[p] = Safetensors(p)
        return shards[p]

    dmd = DmdLora(os.path.join(CKPT, "lora", "dmd_lora.safetensors"), 1.0)

    PFX = "blocks.0."
    sd = blk.state_dict()
    new_sd = {}
    n_int8 = n_plain = n_folded = 0
    for key in sd:                                   # e.g. "attn.qkv.weight"
        full = PFX + key
        if full in name2shard:                       # plain BF16 copy-through (norms/affine/bias)
            new_sd[key] = torch.from_numpy(shard(name2shard[full]).get(full).astype(np.float32))
            n_plain += 1
            continue
        # Linear weight -> dequant int8*scale (+ DMD delta)
        i8 = full[:-len(".weight")] + ".weight_int8" if full.endswith(".weight") else None
        if i8 and i8 in name2shard:
            base = full[:-len(".weight")]            # "blocks.0.attn.qkv"
            w8 = shard(name2shard[i8]).get(i8)
            sc = shard(name2shard[base + ".weight_scale"]).get(base + ".weight_scale")
            w = w8.astype(np.float32) * sc.reshape(-1, *([1] * (w8.ndim - 1)))
            if dmd.has(base):
                d = dmd.delta(base)
                assert d.shape == w.shape, f"{base}: lora {d.shape} != w {w.shape}"
                w = w + d
                n_folded += 1
            new_sd[key] = torch.from_numpy(w)
            n_int8 += 1
            continue
        raise KeyError(f"block-0 tensor not found in shards: {full}")

    missing, unexpected = blk.load_state_dict(new_sd, strict=False)
    # mod_norm_* / pre_audio_crs (Identity) have no params; tolerate.
    real_missing = [m for m in missing if "mod_norm" not in m and "pre_audio" not in m]
    if real_missing or unexpected:
        log(f"[b0] load WARN missing={real_missing[:6]} unexpected={unexpected[:6]}")
    log(f"[b0] loaded block-0: {n_int8} dequant-int8 ({n_folded} DMD-folded) + {n_plain} bf16")
    for st in shards.values():
        st.close()
    dmd.close()
    return blk


def main():
    dtype = torch.float32
    torch.set_grad_enabled(False)

    log("[b0] constructing + loading single block 0 (no full-model load)...")
    blk = build_block0()

    # dense attention patches (skip flash/xformers/bsa) — same as the full oracle.
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

    from longcat_video.modules.blocks import modulate_fp32

    # ---- inputs (read from the C++ dump; same layout the full oracle used) ----
    inx, ixd = rd("in_x.bin")                 # ggml [W,H,T,C]
    W, H, T, Cin = ixd
    N_t = T
    N_h = H // 2
    N_w = W // 2
    C = 4096
    n_token = N_t * N_h * N_w
    log(f"[b0] in_x ggml {ixd} -> latent grid (T,Hp,Wp)=({N_t},{N_h},{N_w}) n_token={n_token} ncl=1")

    xpe, xpd = rd("tap_patch_embed.bin")      # ggml [C, n_token] -> (n_token, C)
    assert xpe.size == n_token * C, f"patch_embed size {xpe.size} != {n_token*C}"
    x = torch.from_numpy(xpe.reshape(n_token, C)).to(dtype).unsqueeze(0)  # [1, N, C]

    temb, td = rd("tap_t_embed.bin")          # ggml [512, T] -> (T, 512)
    t = torch.from_numpy(temb.reshape(T, 512)).to(dtype).unsqueeze(0)     # [1, T, 512]

    yemb, yd = rd("tap_y_embed.bin")          # ggml [C, L] -> (L, C)
    L = yd[1]
    y = torch.from_numpy(yemb.reshape(L, C)).to(dtype).reshape(1, L, C)
    y_seqlens = [L]
    ncl = 1

    B, N, _ = x.shape
    latent_shape = (N_t, N_h, N_w)
    import torch.amp as amp

    log("[b0] reference block-0 forward in FORWARD ORDER...")
    with torch.no_grad():
        with amp.autocast(device_type="cpu", dtype=torch.float32):
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
                blk.adaLN_modulation(t).unsqueeze(2).chunk(6, dim=-1)   # each [1,T,1,C]

        # --- adaLN modulation params ---
        diff("scale_msa", rd("b0_scale_msa.bin")[0], scale_msa.float().cpu().numpy())
        diff("gate_msa", rd("b0_gate_msa.bin")[0], gate_msa.float().cpu().numpy())
        diff("scale_mlp", rd("b0_scale_mlp.bin")[0], scale_mlp.float().cpu().numpy())
        diff("gate_mlp", rd("b0_gate_mlp.bin")[0], gate_mlp.float().cpu().numpy())

        # --- modulate(x) for self-attn ---
        x_m = modulate_fp32(blk.mod_norm_attn, x.view(B, T, -1, C), shift_msa, scale_msa).view(B, N, C)
        diff("x_m_attn", rd("b0_x_m_attn.bin")[0], x_m.float().cpu().numpy())

        # --- self-attn internals: q (pre-rope) and v ---
        a = blk.attn
        qkv = a.qkv(x_m)
        qkv5 = qkv.view(B, N, 3, a.num_heads, a.head_dim).permute(2, 0, 3, 1, 4)
        qh, kh, vh = qkv5.unbind(0)
        qh = a.q_norm(qh); kh = a.k_norm(kh)
        cpp_qpre, _ = rd("sa_q_prerope.bin")              # ggml [d,H,token] -> (token,H,d)
        diff("q_prerope", cpp_qpre.reshape(N, a.num_heads, a.head_dim),
             qh[0].permute(1, 0, 2).float().cpu().numpy())
        cpp_v, _ = rd("sa_v.bin")                         # ggml [d,H,token,N] -> (token,H,d)
        diff("v", cpp_v.reshape(1, N, a.num_heads, a.head_dim)[0],
             vh[0].permute(1, 0, 2).float().cpu().numpy())

        # --- self-attn post-rope q/k (the reference ropes inside attn; recompute) ---
        q_r, k_r = a.rope_3d(qh, kh, latent_shape, None, None)
        cpp_qpost, _ = rd("sa_q_postrope.bin")            # ggml [d,token,H] -> (H,token,d)
        diff("q_postrope", cpp_qpost.reshape(a.num_heads, N, a.head_dim),
             q_r[0].float().cpu().numpy())
        cpp_kpost, _ = rd("sa_k_postrope.bin")
        diff("k_postrope", cpp_kpost.reshape(a.num_heads, N, a.head_dim),
             k_r[0].float().cpu().numpy())

        # --- full self-attn out ---
        attn_out, _ = a(x_m, shape=latent_shape, num_cond_latents=ncl, return_kv=False)
        diff("attn_out", rd("b0_attn_out.bin")[0], attn_out.float().cpu().numpy())

        xa = x + (gate_msa * attn_out.view(B, -1, N // T, C)).view(B, -1, C)
        diff("after_attn", rd("b0_after_attn.bin")[0], xa.float().cpu().numpy())

        xt = xa + blk.cross_attn(blk.pre_crs_attn_norm(xa), y, y_seqlens,
                                 num_cond_latents=ncl, shape=latent_shape)
        diff("after_text", rd("b0_after_text.bin")[0], xt.float().cpu().numpy())

        x_m_ffn = modulate_fp32(blk.mod_norm_ffn, xt.view(B, -1, N // T, C),
                                shift_mlp, scale_mlp).view(B, -1, C)
        diff("x_m_ffn", rd("b0_x_m_ffn.bin")[0], x_m_ffn.float().cpu().numpy())

        gu = F.silu(blk.ffn.w1(x_m_ffn)) * blk.ffn.w3(x_m_ffn)
        diff("ffn_gu", rd("b0_ffn_gu.bin")[0], gu.float().cpu().numpy())

        ffn_out = blk.ffn.w2(gu)
        diff("ffn_out", rd("b0_ffn_out.bin")[0], ffn_out.float().cpu().numpy())

        xf = xt + (gate_mlp * ffn_out.view(B, -1, N // T, C)).view(B, -1, C)
        diff("after_ffn", rd("b0_after_ffn.bin")[0], xf.float().cpu().numpy())

        # --- whole block-0 output vs the C++ tap_block0 ---
        diff("tap_block0", rd("tap_block0.bin")[0], xf.float().cpu().numpy())

    log("[b0] DONE")
    OUT.close()


if __name__ == "__main__":
    main()
