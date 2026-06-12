#!/usr/bin/env python3
"""
M6 Stage-4 TEXTURE DECODE capture + tex_dec weight export.
  (1) EXPORT tex_dec (SparseUnetVaeDecoder, out 6, pred_subdiv=false) weights ->
      cpp_port/weights_npy/tex_dec/  (conv weights -> spike [27,Cin,Cout], like shape_dec).
  (2) fp32 ORACLE: decode the golden tex_slat (stage4_out) using the golden shape subs
      (stage5_mesh/sub{0..3}) as guide_subs -> per-voxel PBR (6ch) ·0.5+0.5. SparseConv3d
      monkeypatched to the spike fp32 gather-matmul (== the C++ port). Saves per-level coords
      + the PBR voxels [N,6] for the C++ tex-decoder validation.

CPU fp32. Run AFTER the tex DiT oracle (serial — one heavy torch at a time):
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 \
    /mnt/hdd/3d/avatar-shootout/Pixal3D/.venv/bin/python stage4_decode_capture.py --export
"""
import os, sys, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CPP = os.path.join(HERE, "cpp_port")
WOUT = os.path.join(CPP, "weights_npy", "tex_dec")
ROUT = os.path.join(CPP, "refs", "stage4")
GOLD = "/home/dbrain/dev/longcat-sparse-spike/tools/sparse_spike/golden_stages"
SNAP = ("/home/dbrain/.cache/huggingface/hub/models--TencentARC--Pixal3D/"
        "snapshots/0b31f9160aa400719af409098bff7936a932f726/ckpts")
TEX_DEC = os.path.join(SNAP, "tex_dec_next_dc_f16c32_fp16")
sys.path.insert(0, "/mnt/hdd/3d/avatar-shootout/Pixal3D")
import stage3a_capture as s3a  # conv_weight_to_vcc, spike_conv_fp32


def export_weights(decoder):
    os.makedirs(WOUT, exist_ok=True)
    sd = decoder.state_dict(); n = 0
    for k, v in sd.items():
        a = v.detach().cpu().float().numpy()
        if k.endswith(".conv.weight") or k.endswith(".conv1.weight") or k.endswith(".conv2.weight"):
            a = s3a.conv_weight_to_vcc(a)  # [27,Cin,Cout]
        np.save(os.path.join(WOUT, k + ".npy"), a.astype(np.float32)); n += 1
    json.dump({k: [int(x) for x in v.shape] for k, v in sd.items()},
              open(os.path.join(WOUT, "_keys.json"), "w"), indent=2)
    print(f"[s4dec export] {n} tensors -> {WOUT}", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--export", action="store_true"); args = ap.parse_args()
    os.makedirs(ROUT, exist_ok=True)
    import torch; torch.set_grad_enabled(False)
    import pixal3d.models as models
    import pixal3d.modules.sparse as sp
    from pixal3d.modules.sparse.conv import conv as conv_mod
    from pixal3d.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder, SparseResBlockC2S3d

    print("[s4dec] loading tex_dec (fp32, CPU)...", flush=True)
    decoder = models.from_pretrained(TEX_DEC)
    decoder.eval(); decoder = decoder.float(); decoder.dtype = torch.float32
    decoder.use_fp16 = False; decoder.to("cpu")
    print(f"[s4dec] out_channels={decoder.out_channels} pred_subdiv={getattr(decoder,'pred_subdiv','?')}", flush=True)
    if args.export:
        export_weights(decoder)

    # conv -> spike fp32 (oracle)
    _wcache = {}
    def hooked_conv(self, x):
        wid = id(self); wv = _wcache.get(wid)
        if wv is None:
            wv = s3a.conv_weight_to_vcc(self.weight.detach().cpu().numpy()); _wcache[wid] = wv
        b = None if getattr(self, "bias", None) is None else self.bias.detach().cpu().numpy()
        return x.replace(s3a.spike_conv_fp32(x.coords, x.feats, wv, b).to(x.feats.dtype))
    conv_mod.SparseConv3d.forward = hooked_conv
    print("[s4dec] SparseConv3d -> spike fp32 (ORACLE)", flush=True)

    # per-level coord capture
    _orig = SparseResBlockC2S3d.forward; _stage = {"i": 0}
    def np_(t): return t.detach().cpu().float().numpy()
    def hooked_c2s(self, x, subdiv=None):
        L = _stage["i"]; _stage["i"] += 1
        ic = np_(x.coords).astype(np.int32)
        out = _orig(self, x, subdiv)
        oc = np_(out[0].coords if isinstance(out, tuple) else out.coords).astype(np.int32)
        np.save(os.path.join(ROUT, f"tex_s{L}_in_coords.npy"), ic)
        np.save(os.path.join(ROUT, f"tex_s{L}_out_coords.npy"), oc)
        print(f"[s4dec] tex s{L}: in N={ic.shape[0]} -> out N={oc.shape[0]}", flush=True)
        return out
    SparseResBlockC2S3d.forward = hooked_c2s

    # inputs: golden tex_slat (stage4_out) @ grid64 + golden shape subs as guide_subs
    tc = np.load(os.path.join(GOLD, "stage4_out", "tex_slat_coords.npy")).astype(np.int32)
    tf = np.load(os.path.join(GOLD, "stage4_out", "tex_slat_feats.npy")).astype(np.float32)
    tex_slat = sp.SparseTensor(feats=torch.from_numpy(tf), coords=torch.from_numpy(tc))
    subs = []
    for L in range(4):
        sc = np.load(os.path.join(GOLD, "stage5_mesh", f"sub{L}_coords.npy")).astype(np.int32)
        sfe = np.load(os.path.join(GOLD, "stage5_mesh", f"sub{L}_feats.npy")).astype(np.float32)
        subs.append(sp.SparseTensor(feats=torch.from_numpy(sfe), coords=torch.from_numpy(sc)))
    print(f"[s4dec] tex_slat N={tc.shape[0]} + {len(subs)} guide_subs; decoding (fp32)...", flush=True)

    out = SparseUnetVaeDecoder.forward(decoder, tex_slat, guide_subs=subs)
    raw = np_(out.feats)                 # [N,6] raw
    pbr = raw * 0.5 + 0.5                # ·0.5+0.5 (decode_tex_slat)
    oc = np_(out.coords).astype(np.int32)
    np.save(os.path.join(ROUT, "tex_out_coords.npy"), oc)
    np.save(os.path.join(ROUT, "tex_pbr_raw.npy"), raw.astype(np.float32))
    np.save(os.path.join(ROUT, "tex_pbr.npy"), pbr.astype(np.float32))
    print(f"[s4dec] PBR voxels N={raw.shape[0]} ch={raw.shape[1]} "
          f"base[{pbr[:,:3].min():.3f},{pbr[:,:3].max():.3f}] -> {ROUT}/tex_pbr.npy", flush=True)


if __name__ == "__main__":
    main()
