#!/usr/bin/env python3
"""Dump reference I/O (.npy) from the VALIDATED m1_ref numpy oracles, so the C++/ggml
port has tight per-stage targets. All fp32, numpy-only (no torch model needed).

Produces (in cpp_port/refs/):
  DINOv3:
    image_chw.npy        [3,512,512]  preprocessed input (ImageNet-normed)
    dino_global.npy      [1,5,1024]   z_global  (== stage1_cond/global.npy)
    dino_patchmap.npy    [1,32,32,1024] patch-token spatial map (pre grid_sample)
    dino_emb.npy         [1,1029,1024] embeddings (cls+reg+patch) pre-blocks  [tap]
    dino_block0.npy      [1,1029,1024] hidden after block 0                   [tap]
  proj:
    proj.npy             [1,4096,1024] z_proj (== stage1_cond/proj.npy)       [from patchmap]
  SS DiT single forward (deterministic case):
    dit_x.npy            [1,8,16,16,16]  input latent (RandomState(0))
    dit_t.npy            [1]             t_scaled = 537.0
    dit_v.npy            [1,8,16,16,16]  v-prediction output
    dit_block0.npy       [1,4096,1536]  hidden after block 0                  [tap]
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
M1 = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, M1)
sys.path.insert(0, os.path.join(M1, "..", "proj_cond"))

REFS = os.path.join(HERE, "refs")
os.makedirs(REFS, exist_ok=True)

GOLD = "/home/dbrain/dev/longcat-sparse-spike/tools/sparse_spike/golden_stages"


def save(name, a):
    a = np.asarray(a, dtype=np.float32)
    np.save(os.path.join(REFS, name + ".npy"), a)
    print(f"  saved {name} {a.shape}", flush=True)


def dump_dinov3():
    import json
    import dinov3_proj as dp
    print("[dinov3] loading weights + image...", flush=True)
    W = dp.Weights()
    img = os.path.join(GOLD, "pre", "preprocessed.png")
    cam = json.load(open(os.path.join(GOLD, "cam.json")))
    image_chw = dp.load_preprocess(img, 512)
    save("image_chw", image_chw)

    # taps: embeddings + block0
    x = dp.embeddings(image_chw, W)
    save("dino_emb", x)
    cos, sin = dp.rope_cos_sin(512, 512)
    n_prefix = 1 + dp.N_REG
    x0 = dp.block(x, W, 0, cos, sin, n_prefix)
    save("dino_block0", x0)

    # full extract
    z_global, z_proj, z_patchmap = dp.extract(img, cam, weights=W)
    save("dino_global", z_global)
    save("dino_patchmap", z_patchmap)
    save("proj", z_proj)
    # sanity vs golden
    g_ref = np.load(os.path.join(GOLD, "stage1_cond", "global.npy"))
    p_ref = np.load(os.path.join(GOLD, "stage1_cond", "proj.npy"))
    print(f"  dino_global vs golden maxabs={np.abs(z_global-g_ref).max():.3e}", flush=True)
    print(f"  proj       vs golden maxabs={np.abs(z_proj-p_ref).max():.3e}", flush=True)


def dump_dit():
    import ss_dit as sd
    print("[ss_dit] loading weights...", flush=True)
    W = sd.Weights(os.path.join(M1, "weights", "ss_flow.npz"))
    cond = {
        "global": np.load(os.path.join(GOLD, "stage1_cond", "global.npy")).astype(np.float32),
        "proj": np.load(os.path.join(GOLD, "stage1_cond", "proj.npy")).astype(np.float32),
    }
    rng = np.random.RandomState(0)
    x = rng.randn(1, 8, 16, 16, 16).astype(np.float32)
    t = np.array([537.0], dtype=np.float32)
    save("dit_x", x)
    save("dit_t", t)

    # tap after block 0: replicate the head of model_forward
    B = x.shape[0]
    h = x.reshape(B, sd.IN_CHANNELS, -1).transpose(0, 2, 1)
    h = sd.linear(h, W["input_layer.weight"], W["input_layer.bias"])
    t_emb = sd.t_embedder(W, t)
    t_emb = sd.silu(t_emb)
    t_emb_mod = sd.linear(t_emb, W["adaLN_modulation.1.weight"], W["adaLN_modulation.1.bias"])
    h0 = sd.block_forward(W, 0, h, t_emb_mod, cond["global"], cond["proj"], W.rope_phases)
    save("dit_block0", h0)

    print("[ss_dit] full forward...", flush=True)
    v = sd.model_forward(W, x, t, cond)
    save("dit_v", v)


def dump_vae():
    import ss_vae_decode as vd
    import numpy as np
    print("[ss_vae] decoding z_s (fp32)...", flush=True)
    z_s = np.load(os.path.join(GOLD, "stage1_ssdec", "z_s.npy"))
    logits = vd.decode(z_s)              # fp32 [1,1,64,64,64]
    save("ss_logits_fp32", logits)
    coords = vd.to_coords(logits)        # [N,4] int
    np.save(os.path.join(REFS, "coords_fp32.npy"), coords.astype(np.int32))
    print(f"  coords_fp32 N={coords.shape[0]}", flush=True)
    gold = np.load(os.path.join(GOLD, "stage1_out", "coords.npy"))
    print(f"  golden coords N={gold.shape[0]}", flush=True)


if __name__ == "__main__":
    which = sys.argv[1:] or ["dinov3", "dit"]
    if "dinov3" in which:
        dump_dinov3()
    if "dit" in which:
        dump_dit()
    if "vae" in which:
        dump_vae()
    print("DUMP DONE")
