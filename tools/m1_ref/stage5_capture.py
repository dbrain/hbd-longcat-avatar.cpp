#!/usr/bin/env python3
"""
M4 capture: decoder.forward (full, grid64->grid1024) fp32 oracle + mesh oracle.

Reuses stage3a_capture's fp32-conv monkeypatch (the REAL decoder modules in fp32 with only
SparseConv3d replaced by the spike gather-matmul). Runs the FULL SparseUnetVaeDecoder.forward
(no early-exit) on the golden HR shape_slat (stage3b_out, denorm, M=4734 @grid64) -> 7-channel
head at grid1024. Splits the FDG head, then runs the REAL o_voxel flexible_dual_grid_to_mesh
(CUDA) on the fp32 head inputs -> verts/faces "mesh oracle" (the tight target for the C++ port).

Captures to cpp_port/refs/stage5/:
  per-level: s{L}_in_coords, s{L}_subdiv_logits, s{L}_out_coords  (L=0..3)  [coord growth]
  head:      head_coords [N,4]@1024, dual_vertices [N,3], intersected [N,3] int8, quad_lerp [N,1], h7 [N,7]
  mesh:      oracle_verts [N,3], oracle_faces [F,3] int64

Run (CPU fp32 decoder, GPU only for the o_voxel mesh extractor):
  OMP_NUM_THREADS=12 ATTN_BACKEND=sdpa \
    /mnt/hdd/3d/avatar-shootout/Pixal3D/.venv/bin/python stage5_capture.py
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CPP = os.path.join(HERE, "cpp_port")
ROUT = os.path.join(CPP, "refs", "stage5")
GOLD = "/home/dbrain/dev/longcat-sparse-spike/tools/sparse_spike/golden_stages"
SNAP = ("/home/dbrain/.cache/huggingface/hub/models--TencentARC--Pixal3D/"
        "snapshots/0b31f9160aa400719af409098bff7936a932f726/ckpts")
SHAPE_DEC = os.path.join(SNAP, "shape_dec_next_dc_f16c32_fp16")
sys.path.insert(0, "/mnt/hdd/3d/avatar-shootout/Pixal3D")
os.environ.setdefault("ATTN_BACKEND", "sdpa")

import stage3a_capture as s3a   # reuse conv_weight_to_vcc, spike_conv_fp32, build_neighbor_map


def main():
    os.makedirs(ROUT, exist_ok=True)
    import torch
    torch.set_grad_enabled(False)
    import pixal3d.models as models
    import pixal3d.modules.sparse as sp
    from pixal3d.modules.sparse.conv import conv as conv_mod
    from pixal3d.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder, SparseResBlockC2S3d

    print("[s5] loading shape_dec (fp32, CPU)...", flush=True)
    decoder = models.from_pretrained(SHAPE_DEC)
    decoder.eval()
    decoder = decoder.float()
    decoder.dtype = torch.float32
    decoder.use_fp16 = False
    decoder.to("cpu")
    decoder.set_resolution(1024)
    vmargin = decoder.voxel_margin
    print(f"[s5] resolution={decoder.resolution} voxel_margin={vmargin}", flush=True)

    # conv -> spike fp32 (oracle)
    _wcache = {}
    def hooked_conv(self, x):
        wid = id(self)
        wv = _wcache.get(wid)
        if wv is None:
            wv = s3a.conv_weight_to_vcc(self.weight.detach().cpu().numpy()); _wcache[wid] = wv
        b = None if getattr(self, "bias", None) is None else self.bias.detach().cpu().numpy()
        out = s3a.spike_conv_fp32(x.coords, x.feats, wv, b)
        return x.replace(out.to(x.feats.dtype))
    conv_mod.SparseConv3d.forward = hooked_conv
    print("[s5] SparseConv3d -> spike fp32 (ORACLE)", flush=True)

    # instrument the 4 C2S up-blocks for coord/subdiv growth
    _orig = SparseResBlockC2S3d.forward
    _stage = {"i": 0}
    def np_(t): return t.detach().cpu().float().numpy()
    def hooked_c2s(self, x, subdiv=None):
        L = _stage["i"]; _stage["i"] += 1
        in_coords = np_(x.coords).astype(np.int32)
        sub_logits = np_(self.to_subdiv(x).feats)
        out = _orig(self, x, subdiv)
        h_out = out[0] if isinstance(out, tuple) else out
        out_coords = np_(h_out.coords).astype(np.int32)
        np.save(os.path.join(ROUT, f"s{L}_in_coords.npy"), in_coords)
        np.save(os.path.join(ROUT, f"s{L}_subdiv_logits.npy"), sub_logits.astype(np.float32))
        np.save(os.path.join(ROUT, f"s{L}_out_coords.npy"), out_coords)
        print(f"[s5] s{L}: in N={in_coords.shape[0]} -> out N={out_coords.shape[0]}", flush=True)
        return out
    SparseResBlockC2S3d.forward = hooked_c2s

    # input: golden HR shape_slat (denorm) @ grid64
    coords = np.load(os.path.join(GOLD, "stage3b_out", "shape_slat_coords.npy")).astype(np.int32)
    feats = np.load(os.path.join(GOLD, "stage3b_out", "shape_slat_feats.npy")).astype(np.float32)
    print(f"[s5] HR shape_slat: N={coords.shape[0]} C={feats.shape[1]} grid-range[{coords[:,1].min()},{coords[:,1].max()}]", flush=True)
    slat = sp.SparseTensor(feats=torch.from_numpy(feats), coords=torch.from_numpy(coords))

    print("[s5] running FULL decoder.forward (fp32)...", flush=True)
    h, subs = SparseUnetVaeDecoder.forward(decoder, slat, return_subs=True)
    h7 = np_(h.feats)            # [N,7]
    hc = np_(h.coords).astype(np.int32)
    N = h7.shape[0]
    print(f"[s5] head: N={N} @grid1024 range[{hc[:,1].min()},{hc[:,1].max()}]  h7[min,max]=[{h7.min():.3f},{h7.max():.3f}]", flush=True)

    # FDG head split (eval): vertices=(1+2m)sigmoid-m, intersected=h[3:6]>0, quad_lerp=softplus(h[6:7])
    import torch.nn.functional as F
    ht = torch.from_numpy(h7)
    dual_vertices = ((1 + 2 * vmargin) * torch.sigmoid(ht[:, 0:3]) - vmargin)
    intersected = (ht[:, 3:6] > 0)
    quad_lerp = F.softplus(ht[:, 6:7])
    np.save(os.path.join(ROUT, "head_coords.npy"), hc)
    np.save(os.path.join(ROUT, "head_h7.npy"), h7.astype(np.float32))
    np.save(os.path.join(ROUT, "head_dual_vertices.npy"), np_(dual_vertices).astype(np.float32))
    np.save(os.path.join(ROUT, "head_intersected.npy"), np_(intersected).astype(np.int8))
    np.save(os.path.join(ROUT, "head_quad_lerp.npy"), np_(quad_lerp).astype(np.float32))
    print(f"[s5] saved head: dual_vertices{dual_vertices.shape} intersected sum={int(intersected.sum())} quad_lerp[min,max]=[{quad_lerp.min():.3f},{quad_lerp.max():.3f}]", flush=True)

    # ---- mesh oracle: real o_voxel on the fp32 head inputs (CUDA) ----
    from o_voxel.convert import flexible_dual_grid_to_mesh
    print("[s5] running real o_voxel flexible_dual_grid_to_mesh on fp32 head (GPU)...", flush=True)
    c_cu = torch.from_numpy(hc[:, 1:]).int().cuda()
    verts, faces = flexible_dual_grid_to_mesh(
        c_cu, dual_vertices.cuda(), intersected.cuda(), quad_lerp.cuda(),
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], grid_size=1024, train=False)
    verts_np = verts.detach().cpu().numpy().astype(np.float32)
    faces_np = faces.detach().cpu().numpy().astype(np.int64)
    np.save(os.path.join(ROUT, "oracle_verts.npy"), verts_np)
    np.save(os.path.join(ROUT, "oracle_faces.npy"), faces_np)
    print(f"[s5] MESH ORACLE: verts={verts_np.shape} faces={faces_np.shape}", flush=True)

    # ---- compare to fp16 golden ----
    gv = np.load(os.path.join(GOLD, "stage5_mesh", "vertices.npy")).astype(np.float32)
    gf = np.load(os.path.join(GOLD, "stage5_mesh", "faces.npy")).astype(np.int64)
    print(f"\n[s5] vs fp16 golden: verts mine={verts_np.shape[0]} golden={gv.shape[0]}  "
          f"faces mine={faces_np.shape[0]} golden={gf.shape[0]}", flush=True)
    # geometric vertex-set IoU at grid1024 (round world->voxel via the inverse transform)
    def vox(v):  # world vertex -> nearest grid1024 voxel index set
        q = np.round((v + 0.5) * 1024 - 0.5).astype(np.int64)
        return set(map(tuple, q.tolist()))
    # cheaper: compare the per-voxel coord set (head_coords vs golden sub-derived) — use vertex round
    sm, sg = vox(verts_np), vox(gv)
    inter = len(sm & sg)
    print(f"[s5] vertex voxel-set IoU vs golden: {inter/len(sm|sg):.6f} (mine={len(sm)} golden={len(sg)})", flush=True)
    print("[s5] DONE", flush=True)


if __name__ == "__main__":
    main()
