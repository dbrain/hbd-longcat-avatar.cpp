"""
Numerical reference oracle for the LongCat-Video-Avatar DiT C++ port.

Loads the int8 base DiT + folds the DMD LoRA (matching proof_gen.py), then calls
the reference `.forward` DIRECTLY with the C++-dumped inputs (no RNG / encode in
the loop). Forward hooks capture the SAME intermediates the C++ taps, and the
script diffs them (cosine sim + max rel err + mean/std) in forward order to
LOCALIZE the first divergence.

Runs on CPU in float32 (most faithful: no nf4, full exponent range — matches the
C++ "force F32 matmul" intent). Slow but only one forward over 5 frames.

Usage (validation venv):
  .venv-oracle/bin/python tools/dit_oracle.py --dump models/_dump
"""
import os, sys, json, math, argparse, struct
import numpy as np

# Pin single-thread distributed env BEFORE importing the reference package.
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29588")

REF = os.path.expanduser("~/dev/longcat-video-ref")
sys.path.insert(0, REF)

import types as _types


import importlib.machinery as _mach


def _stub(name, attrs=None):
    """Register a dummy module so the reference's heavy/CUDA-only imports
    (triton, flash_attn, xformers, librosa, imageio, ...) don't fail. We
    monkeypatch the attention to dense SDPA anyway, so these are never called.
    Give it a real ModuleSpec so importlib.util.find_spec (diffusers probes) works."""
    m = _types.ModuleType(name)
    m.__spec__ = _mach.ModuleSpec(name, loader=None)
    m.__version__ = "0.0.0"
    for k, v in (attrs or {}).items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


# block_sparse_attention.bsa_interface has a module-level @torch.compile that
# pulls in torch inductor -> real triton. We never use BSA (dense SDPA below),
# so stub the whole subpackage + the one symbol attention.py imports from it.
_stub("longcat_video.block_sparse_attention", {"flash_attn_bsa_3d": None})
_stub("longcat_video.block_sparse_attention.bsa_interface", {"flash_attn_bsa_3d": None})
# flash attention / xformers (replaced by dense SDPA below)
_stub("flash_attn", {"flash_attn_func": None, "flash_attn_varlen_func": None})
_stub("flash_attn_interface", {"flash_attn_func": None, "flash_attn_varlen_func": None})
_xf = _stub("xformers")
_xfops = _stub("xformers.ops")
_xf.ops = _xfops
# audio_process pulls librosa / imageio / soundfile / pyloudnorm / av — never
# used for the DiT forward. Stub the package + the one symbol attention.py
# imports (get_attn_map_with_target; only called when ref_target_masks is set,
# i.e. multitalk — never in our single-talk path).
_stub("longcat_video.audio_process", {"get_attn_map_with_target": None})
_stub("longcat_video.audio_process.torch_utils", {"get_attn_map_with_target": None})

import torch
import torch.nn.functional as F

CKPT = "/mnt/hdd/longcat/avatar-1.5"


def read_bin(path):
    """Read a C++ dump: int64 ndim, ndim int64 ggml-dims (fastest first), f32 data.
    Returns (np.ndarray in ggml order ne[0..n-1], dims list)."""
    with open(path, "rb") as fh:
        ndim = struct.unpack("<q", fh.read(8))[0]
        dims = [struct.unpack("<q", fh.read(8))[0] for _ in range(ndim)]
        data = np.frombuffer(fh.read(), dtype="<f4").copy()
    # ggml ne[] is fastest-varying first; numpy wants slowest-first → reverse.
    arr = data.reshape(list(reversed(dims)))
    return arr, dims


def diff(name, cpp_flat, ref_flat):
    """cosine sim + max abs rel err + mean/std on flattened tensors."""
    a = cpp_flat.astype(np.float64).ravel()
    b = ref_flat.astype(np.float64).ravel()
    if a.size != b.size:
        print(f"  [{name}] SIZE MISMATCH cpp={a.size} ref={b.size}")
        return
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    cos = float(a @ b / (na * nb + 1e-30))
    denom = np.maximum(np.abs(b), 1e-3)
    maxrel = float(np.max(np.abs(a - b) / denom))
    mae = float(np.mean(np.abs(a - b)))
    print(f"  [{name:18s}] cos={cos:.6f}  maxrel={maxrel:.3e}  mae={mae:.4e} "
          f"| cpp(m={a.mean():+.4g},s={a.std():.4g}) ref(m={b.mean():+.4g},s={b.std():.4g})")
    return cos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="models/_dump")
    ap.add_argument("--no-lora", action="store_true", help="skip DMD lora fold (debug)")
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = "cpu"
    dtype = torch.float32

    # ---- load inputs dumped by the C++ run ----
    d = args.dump
    x_g, x_dims = read_bin(os.path.join(d, "in_x.bin"))           # ggml [W,H,T,C]
    ts_g, _ = read_bin(os.path.join(d, "in_timestep.bin"))        # [T]
    ctx_g, ctx_dims = read_bin(os.path.join(d, "in_context.bin")) # ggml [C=4096, L=512]
    print(f"[oracle] in_x ggml dims {x_dims} -> np {x_g.shape}")
    print(f"[oracle] in_context ggml dims {ctx_dims} -> np {ctx_g.shape}")
    print(f"[oracle] timestep {ts_g.tolist()}")

    # ggml in_x ne=[W,H,T,C] => np shape (C,T,H,W). torch DiT wants [B,C,T,H,W].
    C, T, H, W = x_g.shape
    hidden_states = torch.from_numpy(x_g).to(dtype).reshape(1, C, T, H, W)
    # ggml in_context ne=[4096, 512] => np (512, 4096) = [L tokens, dim]. torch
    # y_embedder expects caption [B, 1, N_token, C].
    L, Cc = ctx_g.shape
    encoder_hidden_states = torch.from_numpy(ctx_g).to(dtype).reshape(1, 1, L, Cc)
    timestep = torch.from_numpy(ts_g).to(dtype).reshape(-1)  # [T]

    # ---- load reference DiT (int8 base, bf16-construct to avoid 54GB fp32 spike) ----
    from longcat_video.modules.quantization import load_quantized_dit

    print("[oracle] loading int8 DiT (bf16 construct)...", flush=True)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        dit = load_quantized_dit(CKPT, subfolder="base_model_int8", cp_split_hw=[1, 1])
    finally:
        torch.set_default_dtype(prev)

    if not args.no_lora:
        print("[oracle] applying DMD lora (multiplier=1.0, dim128/alpha64)...", flush=True)
        lora_path = os.path.join(CKPT, "lora", "dmd_lora.safetensors")
        dit.load_lora(lora_path, "dmd", multiplier=1.0, lora_network_dim=128, lora_network_alpha=64)
        dit.enable_loras(["dmd"])

    # Cast everything (incl. dequant compute) to float32 on CPU for a clean
    # reference (no nf4, full precision). QuantizedLinear dequants to x.dtype.
    dit = dit.to(device=dev, dtype=dtype)
    dit.eval()

    # Force a dense attention fallback (reference has only flash/xformers/bsa).
    patch_attention(dit)

    # ---- register hooks capturing the SAME intermediates as the C++ taps ----
    taps = {}

    def cap(name):
        def hook(mod, inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            taps[name] = t.detach().float().cpu().numpy()
        return hook

    dit.x_embedder.register_forward_hook(cap("patch_embed"))
    dit.t_embedder.register_forward_hook(cap("t_embed"))
    dit.y_embedder.register_forward_hook(cap("y_embed"))
    dit.blocks[0].register_forward_hook(cap("block0"))
    dit.blocks[1].register_forward_hook(cap("block1"))
    dit.final_layer.register_forward_hook(cap("final_layer"))

    # num_cond_latents: leading zero-timestep frames (C++ derives the same way).
    ncl = int((timestep == 0).sum().item())
    if ncl == timestep.numel():
        ncl = 0
    print(f"[oracle] num_cond_latents={ncl}")

    # audio_embs: the C++ stubs audio (no audio cross-attn). To MATCH that we run
    # a trimmed top-level forward (reproduces the reference forward minus audio
    # token-prep + minus per-block audio cross-attn). patch_block_skip_audio
    # rewrites each block.forward to skip audio.
    patch_block_skip_audio(dit)

    print("[oracle] running reference forward (audio-skipped, matches C++)...", flush=True)
    with torch.no_grad():
        out = run_dit_forward(dit, hidden_states, timestep, encoder_hidden_states, ncl)
    out_np = out.detach().float().cpu().numpy()  # [B, C_out, T, H, W]
    print(f"[oracle] forward done. out shape {out_np.shape} mean={out_np.mean():.5f} std={out_np.std():.5f}")

    # ---- diff against the C++ dumps, in forward order ----
    print("\n==== DIFF (C++ vs torch reference) ====")
    # patch_embed: ggml [hidden, thw] = np (thw, hidden); ref [B, N, C] = (1, thw, hidden)
    cmp_tap(d, "tap_patch_embed", taps.get("patch_embed"))
    cmp_tap(d, "tap_t_embed", taps.get("t_embed"))
    cmp_tap(d, "tap_y_embed", taps.get("y_embed"))
    cmp_tap(d, "tap_block0", taps.get("block0"))
    cmp_tap(d, "tap_block1", taps.get("block1"))
    cmp_tap(d, "tap_final_layer", taps.get("final_layer"))
    # output: ggml [W,H,T,C] = np (C,T,H,W); ref [B,C,T,H,W]
    cpp_out, _ = read_bin(os.path.join(d, "tap_output.bin"))
    diff("output", cpp_out.ravel(), out_np.ravel())

    # persist ref block-0 captures for offline analysis
    np.savez(os.path.join(d, "ref_b0taps.npz"), **{k: v for k, v in B0TAPS.items()})

    # ---- block-0 sub-step diffs (localize within the first divergent block) ----
    print("\n==== BLOCK-0 SUB-STEP DIFF ====")
    for name in ["b0_shift_msa", "b0_scale_msa", "b0_gate_msa", "b0_scale_mlp", "b0_gate_mlp",
                 "b0_x_m_attn", "b0_attn_out", "b0_after_attn", "b0_after_text",
                 "b0_x_m_ffn", "b0_ffn_gu", "b0_ffn_out", "b0_after_ffn"]:
        ref_arr = B0TAPS.get(name)
        path = os.path.join(d, name + ".bin")
        if ref_arr is None or not os.path.exists(path):
            print(f"  [{name}] (missing cpp or ref)")
            continue
        cpp_arr, _ = read_bin(path)
        diff(name, cpp_arr.ravel(), np.ascontiguousarray(ref_arr).ravel())


def cmp_tap(dump_dir, fname, ref_arr):
    if ref_arr is None:
        print(f"  [{fname}] (no ref capture)")
        return
    cpp_arr, dims = read_bin(os.path.join(dump_dir, fname + ".bin"))
    # Compare by total flatten; both should be element-order identical after we
    # account for layout. For [hidden, thw] vs [B, thw, hidden] the element sets
    # match if we flatten the ref in (thw, hidden) C-order == cpp np (thw,hidden).
    name = fname.replace("tap_", "")
    diff(name, cpp_arr.ravel(), np.ascontiguousarray(ref_arr).ravel())


# ---------------------------------------------------------------------------
# Trimmed top-level forward: reproduces LongCatVideoAvatarTransformer3DModel.
# forward EXACTLY (patch_embed -> t_embed -> y_embed -> blocks -> final_layer
# -> unpatchify) but skips the audio token-prep + per-block audio cross-attn,
# so it matches the C++ stub (which builds no audio_hidden_states). Inputs are
# the C++-dumped tensors.
# ---------------------------------------------------------------------------
def run_dit_forward(dit, hidden_states, timestep, encoder_hidden_states, num_cond_latents):
    import torch
    import torch.amp as amp

    B, _, T, H, W = hidden_states.shape
    N_t = T // dit.patch_size[0]
    N_h = H // dit.patch_size[1]
    N_w = W // dit.patch_size[2]

    if len(timestep.shape) == 1:
        # reference: timestep.unsqueeze(0)->[1,T] if scalar-per-frame already len T.
        # Our dumped timestep is [T] (one per frame), so treat as [B=1, T].
        timestep = timestep.unsqueeze(0)  # [1, T]

    dtype = dit.x_embedder.proj.weight.dtype
    hidden_states = hidden_states.to(dtype)
    timestep = timestep.to(dtype)
    encoder_hidden_states = encoder_hidden_states.to(dtype)

    hidden_states = dit.x_embedder(hidden_states)  # [B, N, C]

    with amp.autocast(device_type='cpu', dtype=torch.float32):
        t = dit.t_embedder(timestep.float().flatten(), dtype=torch.float32).reshape(B, N_t, -1)

    encoder_hidden_states = dit.y_embedder(encoder_hidden_states)  # [B, 1, N_token, C]

    # text-token handling (no encoder_attention_mask in our path)
    y_seqlens = [encoder_hidden_states.shape[2]] * encoder_hidden_states.shape[0]
    encoder_hidden_states = encoder_hidden_states.squeeze(1).view(1, -1, hidden_states.shape[-1])

    for blk in dit.blocks:
        hidden_states = blk(
            hidden_states, encoder_hidden_states, t, y_seqlens,
            (N_t, N_h, N_w), num_cond_latents,
        )

    hidden_states = dit.final_layer(hidden_states, t, (N_t, N_h, N_w))
    hidden_states = dit.unpatchify(hidden_states, N_t, N_h, N_w)
    hidden_states = hidden_states.to(torch.float32)
    return hidden_states


# ---------------------------------------------------------------------------
# Monkeypatches: dense attention + audio-skip so the reference matches the
# C++ stub exactly.
# ---------------------------------------------------------------------------
def patch_attention(dit):
    """Replace flash/xformers attention with dense SDPA (math) in fp32."""
    import torch
    from longcat_video.modules.avatar.attention import Attention, SingleStreamAttention
    from longcat_video.modules.attention import MultiHeadCrossAttention

    def dense_process_attn(self, q, k, v, shape):
        # q,k,v: [B, H, S, D]
        return F.scaled_dot_product_attention(q, k, v)

    Attention._process_attn = dense_process_attn

    # MultiHeadCrossAttention._process_cross_attn uses xformers; replace with dense.
    def dense_cross(self, x, cond, kv_seqlen):
        B, N, Cn = x.shape
        q = self.q_linear(x).view(1, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        q, k = self.q_norm(q), self.k_norm(k)
        # [1, S, H, D] -> [1, H, S, D]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        x = F.scaled_dot_product_attention(q, k, v)  # [1, H, N, D]
        x = x.permute(0, 2, 1, 3).reshape(B, -1, Cn)
        x = self.proj(x)
        return x

    MultiHeadCrossAttention._process_cross_attn = dense_cross


B0TAPS = {}  # block-0 sub-step captures, mirroring the C++ b0_* taps


def patch_block_skip_audio(dit):
    """Wrap each block.forward to skip the audio cross-attn (the C++ stubs it)."""
    import types
    import torch
    import torch.amp as amp
    from longcat_video.modules.blocks import modulate_fp32

    def block_forward(self, x, y, t, y_seqlen, latent_shape, num_cond_latents=None,
                      return_kv=False, kv_cache=None, skip_crs_attn=False,
                      num_ref_latents=None, audio_hidden_states=None, ref_img_index=None,
                      mask_frame_range=None, token_ref_target_masks=None, human_num=None):
        t0 = getattr(self, "_tap0", False)

        def cap(name, tt):
            if t0:
                B0TAPS[name] = tt.detach().float().cpu().numpy()

        x_dtype = x.dtype
        B, N, C = x.shape
        Tt, _, _ = latent_shape
        with amp.autocast(device_type='cpu', dtype=torch.float32):
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
                self.adaLN_modulation(t).unsqueeze(2).chunk(6, dim=-1)
        cap("b0_shift_msa", shift_msa); cap("b0_scale_msa", scale_msa); cap("b0_gate_msa", gate_msa)
        cap("b0_scale_mlp", scale_mlp); cap("b0_gate_mlp", gate_mlp)
        if t0:
            cap("b0_in_x", x)
            cap("b0_xn_attn", self.mod_norm_attn(x.view(B, Tt, -1, C).to(torch.float32)).view(B, N, C))
        x_m = modulate_fp32(self.mod_norm_attn, x.view(B, Tt, -1, C), shift_msa, scale_msa).view(B, N, C)
        cap("b0_x_m_attn", x_m)
        attn_outputs = self.attn(x_m, shape=latent_shape, num_cond_latents=num_cond_latents,
                                 return_kv=False, num_ref_latents=num_ref_latents,
                                 ref_img_index=ref_img_index, mask_frame_range=mask_frame_range,
                                 ref_target_masks=token_ref_target_masks)
        x_s, _ = attn_outputs
        cap("b0_attn_out", x_s)
        with amp.autocast(device_type='cpu', dtype=torch.float32):
            x = x + (gate_msa * x_s.view(B, -1, N // Tt, C)).view(B, -1, C)
        x = x.to(x_dtype)
        cap("b0_after_attn", x)
        # text cross attn
        x = x + self.cross_attn(self.pre_crs_attn_norm(x), y, y_seqlen,
                                num_cond_latents=num_cond_latents, shape=latent_shape)
        cap("b0_after_text", x)
        # AUDIO CROSS ATTN SKIPPED (matches C++ stub)
        x_m = modulate_fp32(self.mod_norm_ffn, x.view(B, -1, N // Tt, C), shift_mlp, scale_mlp).view(B, -1, C)
        cap("b0_x_m_ffn", x_m)
        gu = F.silu(self.ffn.w1(x_m)) * self.ffn.w3(x_m)
        cap("b0_ffn_gu", gu)
        x_s = self.ffn.w2(gu)
        cap("b0_ffn_out", x_s)
        with amp.autocast(device_type='cpu', dtype=torch.float32):
            x = x + (gate_mlp * x_s.view(B, -1, N // Tt, C)).view(B, -1, C)
        x = x.to(x_dtype)
        cap("b0_after_ffn", x)
        return x

    for i, blk in enumerate(dit.blocks):
        blk._tap0 = (i == 0)
        blk.forward = types.MethodType(block_forward, blk)


if __name__ == "__main__":
    main()
