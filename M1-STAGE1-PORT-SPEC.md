# M1 port spec — Stage 1 (Sparse Structure) image → occupancy coords

First real E2E slice. Image → SS dense DiT (sampler) → SS VAE decode → coords[N,4].
Exercises: DINOv3 ViT cond, proj grid_sample + camera unproject, dense flow DiT (proj
mode), FlowEulerGuidanceIntervalSampler, dense 3D-conv VAE + pixel_shuffle_3d. **NO
sparse attention, NO NAF, NO mesh** — the smallest loop that produces real geometry.

All exact shapes/params are GROUND TRUTH from `CONFIGS-RESOLVED.md` (configs on disk).
Validate against M0 goldens: `golden_stages/{stage1_cond,stage1_ssdec,stage1_out}/`.

Port everything **fp32** (correctness-first). Python runs the DiT torso bf16 + VAE
fp16; goldens carry that noise → expect tol ~1e-2 rel on z_s, exact-ish on coords
(occupancy is a `>0` threshold, robust to small noise — the real M1 pass/fail signal).

---

## 1. Conditioning — `get_proj_cond_ss` → (z_global, z_proj)   [golden: stage1_cond/]

Model: `DinoV3ProjFeatureExtractor(model=dinov3-vitl16, image_size=512, grid_res=16,
no NAF)`. Camera scalars from MoGe (or `--fov`): `(camera_angle_x, distance,
mesh_scale=1.0)` — captured in `golden_stages/cam.json`.

### 1a. DINOv3 ViT-L/16 feature map  (portable, standard ViT + 3 quirks)
- Input: preprocessed square RGB → `resize(512,512, LANCZOS)` → `/255` →
  ImageNet-normalize `mean=[.485,.456,.406] std=[.229,.224,.225]`.
- ViT-L/16: hidden **1024**, 24 layers, 16 heads (head_dim 64), patch 16 →
  `32×32 = 1024` patch tokens; **4 register tokens**; 1 CLS. Seq = 1+4+1024.
- **Quirks vs vanilla ViT** (net-new bits #4,#5): **2D-axial RoPE on patch tokens
  only** (CLS+4 reg excluded), **LayerScale** (per-channel `λ` ×2/block), gated-MLP
  OFF. Final norm = **plain `F.layer_norm(h, [1024])`** (unweighted — NOT the trained
  final LN). HF class `DINOv3ViTModel`; replicate `embeddings → rope_embeddings →
  24× layer → F.layer_norm`.
- Outputs: `z_global = cat[CLS(1), reg(4)] = [B,5,1024]`; patch tokens →
  `[B,32,32,1024]` spatial map for projection.

### 1b. Proj grid_sample + camera unproject  (net-new #2/#3 — pure arithmetic)
`ProjGrid(grid_res=16, image_res=512)`, `project_points_to_image_batch`,
`sample_features` (`image_conditioned_proj.py:27-232`). Exact recipe:

1. **grid_points** (precompute, constant): `linspace(-1,1,16)` ⊗³ via
   `meshgrid(...,indexing='ij')` → stack `[16³,3]` (x,y,z order) → rotate to Blender
   frame `gp = gp @ Rᵀ`, `R=[[1,0,0],[0,0,-1],[0,1,0]]`. Order is meshgrid-ij so
   token t = (i*16+j)*16+k over (x,y,z) — match exactly for voxel alignment.
2. `gp = gp / mesh_scale / 2`.
3. **transform_matrix** = front view, then `[1,3] = -distance`:
   `T=[[1,0,0,0],[0,0,-1,-distance],[0,1,0,0],[0,0,0,1]]`.
   `world_to_camera = inv(T)` (do in fp32).
4. homogeneous `[16³,4]` → `pc = points_h @ world_to_cameraᵀ` → `[...,:3]`.
   `depth = -z_cam`.
5. `focal = 16/tan(angle/2)`, `focal_px = focal*image_res/32`  (sensor 32mm,
   image_res=512). `x_ndc = focal_px*x_cam/(-z_cam+1e-8)`, same for y.
   `x_pix = x_ndc + 256`, `y_pix = -y_ndc + 256` (Y flip).
6. **normalize for grid_sample**: `g = (pix + 0.5)/512*2 - 1`.
7. `F.grid_sample(fmap[B,1024,32,32], grid[B,16³,1,2], mode=bilinear,
   align_corners=False, padding_mode='border')` → `[B,1024,16³]` → permute →
   `z_proj = [B, 16³, 1024]`. (No NAF at grid 16 → proj C = 1024.)

> grid_sample(align_corners=False, border pad) must match torch's pixel-center
> convention EXACTLY: src coord `= (g+1)/2*size - 0.5`, clamp to `[0,size-1]` for
> border, bilinear. The longcat/sd.cpp base likely has a grid_sample helper — reuse it
> and unit-test against a tiny golden before trusting.

`neg_cond` = zeros_like(global), zeros_like(proj). (CFG negative = all-zeros cond.)

---

## 2. Flow sampling — `sample_sparse_structure`   [golden: stage1_ssdec/z_s]

Model `sparse_structure_flow_model` = `SparseStructureFlowModel` (config in §SS of
CONFIGS-RESOLVED): res 16, in/out 8, model_ch **1536**, cond_ch 1024, **30 blocks**,
**12 heads** (head_dim 128), mlp_ratio 5.3334, **pe_mode rope**, **share_mod true**,
qk_rms_norm true (self+cross), image_attn_mode proj.

`noise = randn(1, 8, 16,16,16)` (seed 42, reseeded in run() right before Stage 1).

### DiT forward (`sparse_structure_flow.py:244`)
```
h = x.view(1,8,4096).permute(0,2,1)            # [1,4096,8] tokens (z,y,x flatten of 16³)
h = Linear(8→1536)(h)
t_emb = t_embedder(t*1000)                       # sinusoidal(256)+MLP → [1,1536]
t_emb = adaLN_modulation(t_emb)                  # share_mod: SiLU+Linear(1536→9216)  → [1,9216]
for blk in 30: h = blk(h, t_emb, (global,proj), rope_phases)
h = F.layer_norm(h, [1536])                      # non-affine, fp32
h = Linear(1536→8)(h)
z_s = h.permute(0,2,1).view(1,8,16,16,16)
```
- **TimestepEmbedder**: `freqs=exp(-ln(10000)*arange(128)/128)`, `emb=cat[cos,sin]`
  (256) → `Linear(256→1536)+SiLU+Linear(1536→1536)`. NOTE `t = t*1000` applied by the
  sampler (`_inference_model`), DiT receives the ×1000 value.
- **RoPE phases**: `RotaryPositionEmbedder(head_dim=128, 3)` over the 16³ grid coords
  (meshgrid ij, [4096,3]) → precomputed phase table. Applied to q,k AFTER qk-rms-norm.

### Block — `ModulatedTransformerCrossBlock._forward` (`modulated.py:180`)  share_mod
```
s_msa,sc_msa,g_msa,s_mlp,sc_mlp,g_mlp = (block.modulation[9216] + t_emb).chunk(6)   # per-block learned bias added to shared mod
h = norm1(x)                       # LayerNorm32 NON-affine eps1e-6
h = h*(1+sc_msa) + s_msa
h = self_attn(h, phases)           # full MHA, qk-rms-norm + rope, head_dim 128
x = x + g_msa*h
h = norm2(x)                       # LayerNorm32 AFFINE eps1e-6  (has weight+bias!)
h = cross_attn(h, (global,proj))   # ProjectAttention — NO modulation, plain residual
x = x + h
h = norm3(x)                       # LayerNorm32 NON-affine
h = h*(1+sc_mlp) + s_mlp
h = mlp(h)                         # FeedForwardNet: Linear(1536→8192)+GELU+Linear(8192→1536)  (mlp_ratio 5.3334)
x = x + g_mlp*h
```
- **self_attn** (`modules.py:66`, type self, full): `to_qkv` Linear(1536→4608) →
  reshape [B,L,3,12,128]; if qk_rms_norm: `MultiHeadRMSNorm` per head (`normalize(x.float())
  *gamma*sqrt(128)`); rope on q,k; `scaled_dot_product_attention(q,k,v)` (softmax(QKᵀ/√128)V);
  `to_out` Linear(1536→1536).
- **cross_attn = ProjectAttention** (`proj_attention.py:38`):
  `global_out = cross_attn_block(h, global_context)` — MHA cross, q from h[1536],
  kv from global[B,5,1024] via `to_kv`(1024→3072), qk_rms_norm_cross on q,k.
  `proj_out = proj_linear(proj_context)` — **per-block** `Linear(1024→1536)` on
  z_proj[B,4096,1024]. `return proj_out + global_out`. ← proj is a per-token linear
  add (one proj feature per voxel token, aligned), global is real cross-attn over 5.
- `mlp` FeedForwardNet (`blocks.py`): `Linear(C, C*mlp_ratio)` + GELU + `Linear(back)`.
  mlp_ratio 5.3334 → hidden ≈ 8192 (confirm exact int from a weight shape in golden).

### CFG / sampler (§3 of E2E-PORT-MAP + `flow_euler.py`) — SS params
steps 12, strength 7.5, rescale 0.7, **interval [0.6,1.0]**, rescale_t 5.0,
sigma_min 1e-5. `t = linspace(1,0,13)`, warp `t = 5t/(1+4t)`. Per step: if
`0.6≤t≤1.0` do 2 forwards (cond+neg) → `v = 7.5*vp + (1-6.5)*vn`... `= vn+7.5*(vp-vn)`;
else 1 forward (cond). Then std-rescale (gr=0.7, σ=1e-5): x0p/x0c via
`_v_to_xstart_eps`, rescale x0c std→x0p std, blend 0.7, back to v. Euler:
`x -= (t-t_prev)*v`. (Pseudocode already in E2E-PORT-MAP §3 — matches verbatim.)

---

## 3. SS VAE decode + occupancy → coords   [golden: stage1_ssdec/ss_logits, stage1_out/coords]

Model `sparse_structure_decoder` = `SparseStructureDecoder`: out 1, latent 8,
res_blocks 2, middle 2, channels **[512,128,32]**, fp16 torso, norm "layer"
(`ChannelLayerNorm32`). (`sparse_structure_vae.py:210`)
```
h = Conv3d(8→512, k3,p1)(z_s)            # [1,512,16,16,16]
h = 2× ResBlock3d(512)                    # middle
# stage loop over channels [512,128,32]:
#   512: 2×ResBlock3d(512) → UpsampleBlock3d(512→128)   # → 32³
#   128: 2×ResBlock3d(128) → UpsampleBlock3d(128→32)    # → 64³
#    32: 2×ResBlock3d(32)   (last, no upsample)
h = ChannelLayerNorm32(32) → SiLU → Conv3d(32→1, k3,p1)  # [1,1,64,64,64]
```
- **ResBlock3d**: `norm1(layer) → SiLU → Conv3d(k3p1) → norm2 → SiLU →
  Conv3d(k3p1, zero-init) + skip(1×1 conv if ch change else identity)`.
- **UpsampleBlock3d** (mode conv): `Conv3d(ch → next*8, k3p1)` then
  **`pixel_shuffle_3d(·, 2)`** (`spatial.py:4`): `[B,C,H,W,D]`, C'=C/8, reshape
  `[B,C',2,2,2,H,W,D]` → permute `(0,1,5,2,6,3,7,4)` → `[B,C',2H,2W,2D]`. **VALIDATED
  GPU-free (bit-exact maxabs 0.0 vs torch):** it's a pure gather — viewing input as
  `[B,C',s,s,s,H',W',D']`, `out[b,c,h,w,d] = in[b, c, h%s, w%s, d%s, h//s, w//s, d//s]`
  (s=2). i.e. in-channel `= c*s³ + (h%s)*s² + (w%s)*s + (d%s)`, in-spatial `=(h//s,w//s,
  d//s)`. Author as a gather kernel (do NOT assume the base's `depth_to_space_3d` uses the
  same s³→channel ordering — this one is spatial-interleaved; verify or just use this map).
- 16³ → ×2 → 32³ → ×2 → 64³. Output logits `[1,1,64,64,64]`.

### Occupancy → coords (`pixal3d_image_to_3d.py:340-346`)
```
decoded = (ss_logits > 0)                 # bool [1,1,64,64,64]
# resolution=32 != 64 → ratio = 64//32 = 2
decoded = max_pool3d(decoded.float(), 2, 2, 0) > 0.5    # [1,1,32,32,32]
coords  = argwhere(decoded)[:, [0,2,3,4]].int()         # [N,4] = (batch, x,y,z) @ res 32
```
**M1 success = coords match the golden** (same N, same voxel set). Occupancy is a
threshold so it's robust to fp32-vs-bf16 noise — the clean pass/fail signal. Also
check z_s rel-err ~1e-2 as a secondary diagnostic.

---

## 4. Reuse inventory (do this before writing kernels)
The worktree is off `longcat-avatar.cpp@5e26fc5` (sd.cpp/ggml). Most of M1 is standard
DiT/ViT/VAE — likely already in the base. Grep the base for, and reuse:
`ggml_conv_3d` / conv3d, layer_norm (affine + non-affine), `scaled_dot_product`/flash,
RoPE apply, SiLU/GELU, grid_sample. **Net-new to author**: pixel_shuffle_3d (reshape),
the proj grid_sample+unproject arithmetic (#2/#3), DINOv3 2D-axial-RoPE+LayerScale+4reg
(if no DINOv3 in base), ProjectAttention (= cross-attn + per-block proj_linear add).
GGUF: convert the 3 Stage-1 ckpts (ss_flow, ss_dec, DINOv3) safetensors→gguf — needed
once running in C++; for pure-op validation, load weights from the goldens' npy first.

## 5. Open items / to confirm at M0 capture time
- exact mlp hidden width (1536*5.3334 = 8192.1 → almost certainly 8192; read a weight
  shape from the gguf/ckpt to be sure).
- DINOv3 `num_register_tokens` (code default 4 — confirm from HF config → z_global=5).
- DINOv3 2D-axial-RoPE base/inv_freq exact formula (read HF `modeling_dinov3_vit.py`
  in the venv `transformers` — port carefully; it's the one non-vanilla ViT detail).
- grid_sample voxel-token ordering (meshgrid-ij x,y,z) must equal the SS-DiT token
  flatten order `x.view(B,C,-1)` of the 16³ latent — verify with a golden (z_proj
  token t aligns with z_s token t).
