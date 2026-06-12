#!/usr/bin/env python3
"""
Precompute Wan CLIP-ViT-H/14 image tokens -> clip_fea .bin for sd-infinitetalk --clip-fea.
Sidesteps the C++ name-map: runs the Wan `.pth` `visual.*` vision tower directly and dumps the
257x1280 hidden states (the clip_fea the DiT img_emb consumes).

Wan visual tower (verified from the .pth): patch_embedding Conv2d(3,1280,k14,s14) -> +cls_embedding
-> +pos_embedding[257] -> pre_norm(LN) -> N pre-norm blocks {norm1, attn.to_qkv(FUSED 3x1280),
attn.proj, norm2, mlp.0/mlp.2 GELU} -> post_norm(LN) -> [257,1280]. (head/projection unused.)

Usage:
  uv run --with numpy --with torch --with pillow python3 tools/precompute_wan_clip.py \
    --clip models/dl/wan21-i2v-14b-480p/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    --image models/ref_singer.png --out models/ref_singer_clipfea.bin
"""
import argparse, re, struct
import numpy as np


def write_bin(path, ne0, ne1, data_f32, name="clip_fea"):
    # matches load_tensor_from_file_as_tensor: n_dims,i32 name_len,i32 ttype,i32 dims[n_dims],i32 name data
    with open(path, "wb") as f:
        f.write(struct.pack("<i", 2))            # n_dims
        f.write(struct.pack("<i", len(name)))    # name length
        f.write(struct.pack("<i", 0))            # ttype = GGML_TYPE_F32 (0)
        f.write(struct.pack("<i", ne0))          # dims[0] (fastest)
        f.write(struct.pack("<i", ne1))          # dims[1]
        f.write(name.encode())
        f.write(np.ascontiguousarray(data_f32, dtype=np.float32).tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=224)  # ViT-H/14 -> 16x16 patches + cls = 257
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from PIL import Image

    sd = torch.load(args.clip, map_location="cpu", weights_only=False)
    V = {k[len("visual."):]: v.float() for k, v in sd.items() if k.startswith("visual.")}
    nl = 1 + max(int(re.match(r"transformer\.(\d+)\.", k).group(1)) for k in V if k.startswith("transformer."))
    dim = V["cls_embedding"].shape[-1]
    n_head = 16
    hd = dim // n_head
    print(f"Wan CLIP visual: dim={dim} layers={nl} heads={n_head} head_dim={hd}")

    # preprocess: short-side resize -> center-crop size -> CLIP normalize
    img = Image.open(args.image).convert("RGB")
    w, h = img.size
    s = args.size / min(w, h)
    img = img.resize((round(w * s), round(h * s)), Image.BICUBIC)
    w, h = img.size
    L, T = (w - args.size) // 2, (h - args.size) // 2
    img = img.crop((L, T, L + args.size, T + args.size))
    x = torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    x = ((x - mean) / std).unsqueeze(0)  # [1,3,size,size]

    with torch.no_grad():
        h_ = F.conv2d(x, V["patch_embedding.weight"], stride=14)            # [1,dim,16,16]
        h_ = h_.flatten(2).transpose(1, 2)                                   # [1,256,dim]
        cls = V["cls_embedding"].reshape(1, 1, dim).expand(1, 1, dim)
        h_ = torch.cat([cls, h_], dim=1) + V["pos_embedding"]                # [1,257,dim]
        h_ = F.layer_norm(h_, (dim,), V["pre_norm.weight"], V["pre_norm.bias"])
        N = h_.shape[1]
        for i in range(nl):
            p = f"transformer.{i}."
            y = F.layer_norm(h_, (dim,), V[p + "norm1.weight"], V[p + "norm1.bias"])
            qkv = y @ V[p + "attn.to_qkv.weight"].t() + V[p + "attn.to_qkv.bias"]  # [1,N,3*dim]
            q, k, v = qkv.split(dim, dim=-1)
            q = q.view(1, N, n_head, hd).transpose(1, 2)
            k = k.view(1, N, n_head, hd).transpose(1, 2)
            v = v.view(1, N, n_head, hd).transpose(1, 2)
            a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(1, N, dim)
            a = a @ V[p + "attn.proj.weight"].t() + V[p + "attn.proj.bias"]
            h_ = h_ + a
            y = F.layer_norm(h_, (dim,), V[p + "norm2.weight"], V[p + "norm2.bias"])
            y = y @ V[p + "mlp.0.weight"].t() + V[p + "mlp.0.bias"]
            y = F.gelu(y)
            y = y @ V[p + "mlp.2.weight"].t() + V[p + "mlp.2.bias"]
            h_ = h_ + y
        h_ = F.layer_norm(h_, (dim,), V["post_norm.weight"], V["post_norm.bias"])  # [1,257,dim]

    feat = h_[0].numpy().astype(np.float32)  # [257, dim] tok-major, c-fastest within tok
    print(f"clip_fea: shape={feat.shape} mean={feat.mean():.4f} std={feat.std():.4f} "
          f"min={feat.min():.3f} max={feat.max():.3f}")
    # save ne [dim,257]: data order = for tok(ne1) for c(ne0) = feat C-order flatten
    write_bin(args.out, dim, N, feat.flatten())
    print(f"wrote {args.out} ne=[{dim},{N}]")


if __name__ == "__main__":
    main()
