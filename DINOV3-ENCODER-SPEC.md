# DINOv3 ViT-L/16 encoder — port spec (M1 conditioner backbone)

The image cond backbone for ALL 4 proj conditioners (ss / shape_512 / shape_1024 /
tex_1024). HF model `camenduru/dinov3-vitl16-pretrain-lvd1689m`, class `DINOv3ViTModel`
(`transformers/models/dinov3_vit/modeling_dinov3_vit.py`). The base sd.cpp repo has a
CLIP ViT (`clip.hpp`) as a structural template; the deltas below are what make it DINOv3.

Used via Pixal3D's `DinoV3ProjFeatureExtractor.extract_features` (`image_conditioned_proj.py:450`)
which **bypasses `model.forward`** and does: `embeddings → rope_embeddings → 24×layer →
F.layer_norm` (NOTE: plain unweighted final norm, NOT the trained `self.norm`).

## Exact config (camenduru/dinov3-vitl16 config.json — VERIFIED on disk)
```
hidden_size 1024, num_hidden_layers 24, num_attention_heads 16 (head_dim 64),
patch_size 16, num_register_tokens 4, intermediate_size 4096, hidden_act gelu,
use_gated_mlp FALSE, rope_theta 100.0, layerscale_value 1.0 (learned), layer_norm_eps 1e-5,
query_bias true, KEY_BIAS FALSE, value_bias true, proj_bias true, mlp_bias true,
attention_dropout 0, drop_path 0.  image_size in config=224 but extractor passes 512/1024.
```

## Forward (extractor path, B images, image already resized to `image_size`²)
1. **Preprocess**: `resize(image_size, LANCZOS)` → `/255` → ImageNet normalize
   `mean=[.485,.456,.406] std=[.229,.224,.225]`. (extractor `transform`.)
2. **Embeddings** (`DINOv3ViTEmbeddings`): `Conv2d(3→1024, k16, s16)` patchify →
   `flatten(2).transpose` → `[B, P, 1024]` (P = (image_size/16)² ; 1024 @512, 4096 @1024).
   Prepend `cls_token(1)` + `register_tokens(4)` → seq `[B, 5+P, 1024]`.
   Order: **[cls, reg×4, patches]**. (mask_token unused at inference.)
3. **2D-axial RoPE** (`DINOv3ViTRopePositionEmbedding`) — **VALIDATED numpy↔torch GPU-free**
   (`tools/proj_cond/test_dinov3_rope.py`, cos/sin maxabs 6e-8). Depends only on (H,W):
   - `inv_freq = 1 / 100^arange(0, 1, 4/64)` → 16 freqs.
   - patch-center coords: `(arange(nph)+0.5)/nph`, `(arange(npw)+0.5)/npw`, meshgrid `ij`
     → `[P,2]` (y,x) → `*2-1` (→ [-1,1]).
   - `angles = 2π · coords[:,:,None] · inv_freq` `[P,2,16]` → `flatten(1,2)` `[P,32]` →
     `tile(2)` `[P,64]`. `cos=cos(angles), sin=sin(angles)` (fp32).
4. **24× `DINOv3ViTLayer`** (pre-norm, LayerScale, drop_path=Identity at inference):
   ```
   r=x;  h=norm1(x);  h=attn(h, cos,sin);  h=layer_scale1(h);  x = r + h
   r=x;  h=norm2(x);  h=mlp(h);            h=layer_scale2(h);  x = r + h
   ```
   - **attn** (`DINOv3ViTAttention`): SEPARATE `q_proj/k_proj/v_proj/o_proj` (NOT fused
     qkv). **k_proj has NO bias**; q/v/o have bias. reshape `[B,heads,P+5,64]`.
     RoPE applied to q,k **patch tokens only** (cls+reg excluded): split off first 5
     tokens, `q_patch = q*cos + rotate_half(q)*sin` (`rotate_half(x)=cat(-x2,x1)`,
     halves), recombine. softmax(QKᵀ·64^-0.5)·V. **No QK-norm.** `o_proj`.
   - **layer_scale**: per-channel `h * lambda1` (`lambda1` learned [1024], init 1.0).
   - **mlp** (`DINOv3ViTMLP`, NOT gated): `down_proj(gelu(up_proj(x)))`, 1024→4096→1024,
     bias on both.
   - **norm1/norm2**: standard affine `LayerNorm(1024, eps=1e-5)`.
5. **Final norm**: `F.layer_norm(h, [1024])` — **unweighted/plain** (eps default 1e-5),
   NOT the model's trained `self.norm`. (Deliberate, per extractor `extract_features`.)
6. **Split** (`forward`): `z_clstoken = z[:,0:1]`; `z_regtokens = z[:,1:5]`;
   `z_patchtokens = z[:,5:]` → reshape `[B, nph, npw, 1024]` (spatial map for ProjGrid).
   `z_global = cat[cls, reg] = [B,5,1024]`.

## Reuse / net-new (vs base CLIP ViT)
- Reuse: patch Conv2d, LayerNorm (affine), MHA + softmax, gelu, Linear, residual.
- **Net-new deltas**: 2D-axial RoPE (validated above; reuse base RoPE-apply with these
  phases), LayerScale (per-channel mul — trivial), 4 register tokens + cls (concat),
  asymmetric bias (k_proj biasless), final plain `F.layer_norm` (not trained norm),
  RoPE on patch tokens ONLY (skip first 5). No gated MLP, no QK-norm here.

## Weights → GGUF
Convert `camenduru/dinov3-vitl16` safetensors → gguf (base `src/convert.cpp`). Tensor
names: `embeddings.{cls_token,register_tokens,patch_embeddings.{weight,bias}}`,
`layer.{i}.{norm1,norm2}.{weight,bias}`, `layer.{i}.attention.{q,k,v,o}_proj.{weight,bias}`
(no `k_proj.bias`), `layer.{i}.{layer_scale1,layer_scale2}.lambda1`,
`layer.{i}.mlp.{up,down}_proj.{weight,bias}`. Skip trained `norm.*` (extractor unused).
For pure-op validation first, load from M0 `golden_stages/stage1_cond/` (z_global/z_proj
are the stage boundary) rather than gguf.

## Validation ladder (GPU when M0 done)
1. [DONE, GPU-free] 2D-axial RoPE numpy↔torch — `test_dinov3_rope.py` PASS (6e-8).
2. [DONE, GPU-free] proj grid_sample+unproject numpy/C++↔torch — `tools/proj_cond/` PASS (~1e-5).
3. [GPU/CPU-load] full DINOv3 encoder: feed M0 preprocessed image → compare z_global/z_proj
   vs `golden_stages/stage1_cond/`. (Can run DINOv3 on CPU with CUDA hidden using the
   cached weights — no GPU needed, just ~1.1GB host load.)
