#!/usr/bin/env python3
"""
M3a capture + weight export for shape_slat_decoder.upsample(x4) -> hr_coords.

Two jobs in one script:
  (1) EXPORT shape_dec weights to cpp_port/weights_npy/shape_dec/ in the per-tensor
      .npy format the C++ port loads. Conv weights are CONVERTED to the spike's
      canonical [V=27, Cin, Cout] via transpose(1,2,3,4,0).reshape — the SAME
      convention golden_hook.py used + validated bit-exact vs flex_gemm (~1e-7).
  (2) CAPTURE the per-stage goldens by running decoder.upsample on the golden
      DENORM lr_slat (stage2_out). Two modes:
        --mode fp32  : real decoder modules in FP32 on CPU, with ONLY SparseConv3d
                       monkeypatched to the spike's fp32 gather-matmul (== exactly
                       what the C++ port computes). This is the TRUE-FP32 ORACLE.
        --mode gpu   : real decoder + real flex_gemm in fp16 on CUDA. Sanity set
                       (must reproduce the existing golden hr_coords = 382554).

Per stage L in 0..3 (the 4 C2S up-blocks) we dump:
  s{L}_in_coords/in_feats   (input to the C2S block = output of level-L ConvNeXt)
  s{L}_subdiv_logits/sub    ([N,8] logits + bool>0)
  s{L}_out_coords/out_feats (after C2S = input to level L+1)
plus from_latent_out (feats+coords, input to level 0) and hr_coords (final).

Run (oracle, the one the C++ port validates against):
  OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES="" \
    /mnt/hdd/3d/avatar-shootout/Pixal3D/.venv/bin/python stage3a_capture.py --mode fp32 --export
GPU sanity:
  /mnt/hdd/3d/avatar-shootout/Pixal3D/.venv/bin/python stage3a_capture.py --mode gpu
"""
import os, sys, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CPP = os.path.join(HERE, "cpp_port")
WOUT = os.path.join(CPP, "weights_npy", "shape_dec")
ROUT = os.path.join(CPP, "refs", "stage3a")
GOLD = "/home/dbrain/dev/longcat-sparse-spike/tools/sparse_spike/golden_stages"
SNAP = ("/home/dbrain/.cache/huggingface/hub/models--TencentARC--Pixal3D/"
        "snapshots/0b31f9160aa400719af409098bff7936a932f726/ckpts")
SHAPE_DEC = os.path.join(SNAP, "shape_dec_next_dc_f16c32_fp16")

sys.path.insert(0, "/mnt/hdd/3d/avatar-shootout/Pixal3D")
os.environ.setdefault("ATTN_BACKEND", "sdpa")

SENTINEL = np.uint32(0xFFFFFFFF)


def kernel_offsets(K=3, dilation=1):
    c = (K - 1) // 2
    offs = []
    for kz in range(K):
        for ky in range(K):
            for kx in range(K):
                offs.append(((kz - c) * dilation, (ky - c) * dilation, (kx - c) * dilation))
    return offs


def coord_key_arr(coords):
    # coords [N,4] int -> int64 packed key (b,z,y,x), 20-bit fields
    b = coords[:, 0].astype(np.int64); z = coords[:, 1].astype(np.int64)
    y = coords[:, 2].astype(np.int64); x = coords[:, 3].astype(np.int64)
    M = np.int64(0xFFFFF)
    return (((((b << 20) | (z & M)) << 20) | (y & M)) << 20) | (x & M)


def build_neighbor_map(coords):
    """[N,27] uint32, nmap[i,v] = index of coord[i]+offs[v] or SENTINEL. Vectorized."""
    N = coords.shape[0]
    offs = kernel_offsets()
    keys = coord_key_arr(coords)
    order = np.argsort(keys, kind="stable")
    skeys = keys[order]
    nmap = np.full((N, 27), SENTINEL, dtype=np.uint32)
    for v, (dz, dy, dx) in enumerate(offs):
        nc = coords.copy()
        nc[:, 1] += dz; nc[:, 2] += dy; nc[:, 3] += dx
        qk = coord_key_arr(nc)
        pos = np.searchsorted(skeys, qk)
        pos_cl = np.clip(pos, 0, N - 1)
        hit = skeys[pos_cl] == qk
        nmap[hit, v] = order[pos_cl[hit]].astype(np.uint32)
    return nmap


_NMAP_CACHE = {}


def spike_conv_fp32(coords, feats, weight_vcc, bias):
    """fp32 gather-matmul == the C++ spike kernel. weight_vcc [27,Cin,Cout]."""
    import torch
    N = feats.shape[0]
    cnp = coords.detach().cpu().numpy().astype(np.int32)
    ks = coord_key_arr(cnp)
    sig = (N, int(ks.sum() & 0x7FFFFFFFFFFFFFFF), int(ks[0]), int(ks[-1]))  # content signature
    nm = _NMAP_CACHE.get(sig)
    if nm is None:
        nm = build_neighbor_map(cnp)
        _NMAP_CACHE[sig] = nm
    nm_t = torch.from_numpy(nm.astype(np.int64))
    Cin = feats.shape[1]; Cout = weight_vcc.shape[2]
    ff = feats.float()
    # pad a zero row for SENTINEL gather
    fpad = torch.cat([ff, torch.zeros(1, Cin, dtype=ff.dtype)], dim=0)
    idx = nm_t.clone()
    idx[idx == np.int64(0xFFFFFFFF)] = N  # sentinel -> zero row
    out = torch.zeros(N, Cout, dtype=torch.float32)
    W = torch.from_numpy(weight_vcc.astype(np.float32))  # [27,Cin,Cout]
    for v in range(27):
        g = fpad[idx[:, v]]               # [N,Cin]
        out += g @ W[v]                   # [N,Cout]
    if bias is not None:
        out += torch.from_numpy(bias.astype(np.float32))[None]
    return out


def conv_weight_to_vcc(w):
    """torch SparseConv3d weight [Co,Kd,Kh,Kw,Ci] -> spike [27,Cin,Cout]."""
    w = np.asarray(w, dtype=np.float32)
    Co = w.shape[0]; Ci = w.shape[4]
    return w.transpose(1, 2, 3, 4, 0).reshape(27, Ci, Co)


def export_weights(decoder):
    """Dump every decoder param to per-tensor .npy; conv weights -> [27,Cin,Cout]."""
    os.makedirs(WOUT, exist_ok=True)
    sd = decoder.state_dict()
    n = 0
    for k, v in sd.items():
        a = v.detach().cpu().float().numpy()
        if k.endswith(".conv.weight") or k.endswith(".conv1.weight") or k.endswith(".conv2.weight"):
            a = conv_weight_to_vcc(a)  # [27,Cin,Cout]
        np.save(os.path.join(WOUT, k + ".npy"), a.astype(np.float32))
        n += 1
    meta = {k: [int(x) for x in v.shape] for k, v in sd.items()}
    json.dump(meta, open(os.path.join(WOUT, "_keys.json"), "w"), indent=2)
    print(f"[export] {n} tensors -> {WOUT}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fp32", "gpu"], default="fp32")
    ap.add_argument("--export", action="store_true")
    args = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import pixal3d.models as models
    import pixal3d.modules.sparse as sp
    from pixal3d.modules.sparse.conv import conv as conv_mod
    os.makedirs(ROUT, exist_ok=True)

    fp32 = (args.mode == "fp32")
    # Keep CUDA visible so `import flex_gemm` succeeds; in fp32 mode the decoder runs
    # entirely on CPU with the conv monkeypatched, so no flex_gemm kernel ever fires.
    dev = "cpu" if fp32 else "cuda"

    print(f"[s3a] loading shape_dec (mode={args.mode}, dev={dev})...", flush=True)
    decoder = models.from_pretrained(SHAPE_DEC)
    decoder.eval()
    if fp32:
        decoder = decoder.float()          # ALL params -> fp32 (weights are fp16 in ckpt)
        decoder.dtype = torch.float32
        decoder.use_fp16 = False
    decoder.to(dev)

    if args.export:
        export_weights(decoder)

    # ---- monkeypatch conv to the spike fp32 path (oracle only) ----
    if fp32:
        _wcache = {}
        def hooked_conv(self, x):
            wid = id(self)
            wv = _wcache.get(wid)
            if wv is None:
                wv = conv_weight_to_vcc(self.weight.detach().cpu().numpy())
                _wcache[wid] = wv
            b = None if getattr(self, "bias", None) is None else self.bias.detach().cpu().numpy()
            out = spike_conv_fp32(x.coords, x.feats, wv, b)
            return x.replace(out.to(x.feats.dtype))
        conv_mod.SparseConv3d.forward = hooked_conv
        print("[s3a] SparseConv3d -> spike fp32 gather-matmul (ORACLE)", flush=True)

    # ---- instrument the 4 C2S up-blocks ----
    from pixal3d.models.sc_vaes.sparse_unet_vae import SparseResBlockC2S3d
    _orig_c2s = SparseResBlockC2S3d.forward
    _stage = {"i": 0}
    def np_(t):
        return t.detach().cpu().float().numpy()
    def hooked_c2s(self, x, subdiv=None):
        L = _stage["i"]; _stage["i"] += 1
        in_coords = np_(x.coords).astype(np.int32)
        in_feats = np_(x.feats)
        subdiv_t = self.to_subdiv(x)                  # SparseLinear C->8
        sub_logits = np_(subdiv_t.feats)              # [N,8]
        sub_bool = (sub_logits > 0)
        out = _orig_c2s(self, x, subdiv)
        h_out = out[0] if isinstance(out, tuple) else out
        out_coords = np_(h_out.coords).astype(np.int32)
        out_feats = np_(h_out.feats)
        tag = f"s{L}"
        np.save(os.path.join(ROUT, f"{tag}_in_coords.npy"), in_coords)
        np.save(os.path.join(ROUT, f"{tag}_in_feats.npy"), in_feats.astype(np.float32))
        np.save(os.path.join(ROUT, f"{tag}_subdiv_logits.npy"), sub_logits.astype(np.float32))
        np.save(os.path.join(ROUT, f"{tag}_sub.npy"), sub_bool)
        np.save(os.path.join(ROUT, f"{tag}_out_coords.npy"), out_coords)
        np.save(os.path.join(ROUT, f"{tag}_out_feats.npy"), out_feats.astype(np.float32))
        print(f"[s3a] {tag}: in N={in_coords.shape[0]} -> out N={out_coords.shape[0]} "
              f"(children sum={int(sub_bool.sum())}) logit[min,max]=[{sub_logits.min():.3f},{sub_logits.max():.3f}]",
              flush=True)
        return out
    SparseResBlockC2S3d.forward = hooked_c2s

    # ---- capture from_latent output by wrapping upsample's first ops ----
    # build the lr_slat SparseTensor from the golden (DENORM)
    lr_coords = np.load(os.path.join(GOLD, "stage2_out", "lr_slat_coords.npy")).astype(np.int32)
    lr_feats = np.load(os.path.join(GOLD, "stage2_out", "lr_slat_feats.npy")).astype(np.float32)
    print(f"[s3a] lr_slat: N={lr_coords.shape[0]} C={lr_feats.shape[1]} "
          f"coord-range x[{lr_coords[:,1].min()},{lr_coords[:,1].max()}]", flush=True)
    ct = torch.from_numpy(lr_coords).to(dev)
    # feed slat at from_latent's dtype (pipeline feeds fp32; internal .type(dtype) casts
    # to fp16 AFTER from_latent in the gpu path).
    ft = torch.from_numpy(lr_feats).to(dev).to(decoder.from_latent.weight.dtype)
    slat = sp.SparseTensor(feats=ft, coords=ct)

    # capture from_latent out (input to level 0)
    fl = decoder.from_latent(slat)
    fl = fl.type(decoder.dtype)
    np.save(os.path.join(ROUT, "from_latent_coords.npy"), np_(fl.coords).astype(np.int32))
    np.save(os.path.join(ROUT, "from_latent_feats.npy"), np_(fl.feats).astype(np.float32))

    print(f"[s3a] running upsample (mode={args.mode})...", flush=True)
    hr = decoder.upsample(slat, upsample_times=4)
    hr_np = hr.detach().cpu().numpy().astype(np.int32)
    suff = "_fp32" if fp32 else "_gpu"
    np.save(os.path.join(ROUT, f"hr_coords{suff}.npy"), hr_np)
    print(f"[s3a] hr_coords{suff}: {hr_np.shape}", flush=True)

    # ---- validate vs existing golden ----
    gold = np.load(os.path.join(GOLD, "stage3a_up", "hr_coords.npy")).astype(np.int32)
    def setof(a):
        return set(map(tuple, a.tolist()))
    sg = setof(gold); sh = setof(hr_np)
    inter = len(sg & sh); union = len(sg | sh)
    print(f"\n[s3a] hr_coords{suff} N={hr_np.shape[0]} vs golden N={gold.shape[0]}: "
          f"inter={inter} IoU={inter/union:.6f} (golden-only={len(sg-sh)}, mine-only={len(sh-sg)})", flush=True)

    # quantize to grid64 -> M (== stage3b coords)
    hr_res, lr_res = 1024, 512
    q = np.concatenate([hr_np[:, :1],
                        ((hr_np[:, 1:] + 0.5) / lr_res * (hr_res // 16)).astype(np.int32)], axis=1)
    qu = np.unique(q, axis=0)
    s3b = np.load(os.path.join(GOLD, "stage3b_cond", "proj_coords.npy")).astype(np.int32) \
        if os.path.exists(os.path.join(GOLD, "stage3b_cond", "proj_coords.npy")) else None
    msg = f"[s3a] quantized grid64 -> M={qu.shape[0]}"
    if s3b is not None:
        si = setof(s3b); sq = setof(qu)
        msg += f"  vs stage3b M={s3b.shape[0]}: inter={len(si&sq)} IoU={len(si&sq)/len(si|sq):.6f}"
    print(msg, flush=True)
    print("[s3a] DONE", flush=True)


if __name__ == "__main__":
    main()
