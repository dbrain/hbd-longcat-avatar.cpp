# Pixal3D configs RESOLVED (ground truth) — corrects E2E-PORT-MAP guesses

The map (`E2E-PORT-MAP.md`) flagged several channel widths / sampler params as `?`
"read from code, confirm against pipeline.json (fetched from HF, not on disk)".

**`pipeline.json` + all per-model configs ARE on disk** in the HF cache (no GPU, no
network needed) — they were pulled during the 2026-06-11 golden-conv decode:

```
/home/dbrain/.cache/huggingface/hub/models--TencentARC--Pixal3D/
    snapshots/0b31f9160aa400719af409098bff7936a932f726/
        pipeline.json
        ckpts/{ss_flow,ss_dec,slat_flow_*512,slat_flow_*1024,shape_dec,tex_dec,
               slat_flow_imgshape2tex_*1024}.json
```

The M0 dumper (`golden_stage_hook.py`) also copies them into `golden_stages/configs/`.

---

## ⚠️ Corrections to E2E-PORT-MAP.md

| Map said | TRUTH (from configs) |
|---|---|
| shape slat = `8` ch (latent), decoder head `7` | **shape/tex SLat latent = 32 ch** (`in/out_channels: 32`; normalization mean/std are 32-dim). FDG decoder head **is** 7 (3 vert + 3 intersected + 1 quad_lerp). SS latent (`z_s`) **is** 8. |
| SS DiT: "APE (sinusoidal) or RoPE" | **`pe_mode: "rope"`** for ALL flow models (SS + both shape + tex). No APE at inference. |
| share_mod "read from config" | **`share_mod: true`** for ALL flow models → ONE global `SiLU+Linear(C,6C)` adaLN, per-block applies it (no per-block modulation MLP). |
| SLat self-attn full vs windowed "config-driven" | **`attn_mode='full'`** hard-coded in `SLatFlowModel.__init__` (`structured_latent_flow.py:84`). NOT config-driven. **No windowed/double-windowed attention on the geometry path** → primitive #9 (windowed sparse attn) is NOT needed for M1–M5. Only #8 (full varlen sparse attn) + #10 (sparse RoPE). |
| flow model class `SLatFlowModel` | class is **`ElasticSLatFlowModel`** = `SparseTransformerElasticMixin + SLatFlowModel`. The mixin only chunks the forward for low-VRAM *training*; inference forward == `SLatFlowModel.forward`. Port `SLatFlowModel.forward` verbatim. |
| decoder blocks "ResBlock + UpsampleBlock" | decoder block_type = **`SparseConvNeXtBlock3d`** (ConvNeXt, not ResBlock); up_block_type = **`SparseResBlockC2S3d`** (channel→spatial reshuffle, primitive #14 — NOT pixel-shuffle). |
| guidance_interval "read it" | SS/shape `[0.6, 1.0]`; **tex `[0.6, 0.9]`** (different). |

---

## pipeline.json (the loaded config; class name in file = `Trellis2ImageTo3DPipeline`)

Samplers — all `FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)`:

| stage | steps | guidance_strength | guidance_rescale | guidance_interval | rescale_t |
|---|---|---|---|---|---|
| sparse_structure | 12 | 7.5 | 0.7 | [0.6, 1.0] | 5.0 |
| shape_slat (LR+HR) | 12 | 7.5 | 0.5 | [0.6, 1.0] | 3.0 |
| tex_slat (PHASE-2) | 12 | 1.0 (CFG off) | 0.0 | **[0.6, 0.9]** | 3.0 |

> Note: `inference.run_inference()` passes per-stage `*_sampler_params` overrides
> that re-supply steps/strength/rescale/rescale_t with the SAME values as
> pipeline.json — but it does **NOT** override `guidance_interval`, so the interval
> comes from pipeline.json (above). The overrides also DON'T pass guidance_interval,
> so `[0.6,1.0]`/`[0.6,0.9]` hold.

`shape_slat_normalization` and `tex_slat_normalization` each have **32-dim** `mean`
and `std` (denorm: `slat = slat*std + mean`). Full arrays are in pipeline.json /
`golden_stages/configs/pipeline.json`. (Confirms slat = 32 ch.)

image_cond_model (pipeline.json) = `DinoV3FeatureExtractor(facebook/dinov3-vitl16...)`
— but **inference.py overrides this**: it builds 4 separate
`DinoV3ProjFeatureExtractor` (proj mode) from `IMAGE_COND_CONFIGS` (below), NOT the
pipeline.json one. The pipeline.json image_cond_model is unused in proj mode.

rembg = `BiRefNet(briaai/RMBG-2.0)`. default_pipeline_type = `1536_cascade`
(but `--low_vram` / our runner forces `1024_cascade`, hr_resolution=1024).

---

## Per-model ckpt configs

### SS flow — `ss_flow_img_dit_1_3B_64_bf16` → `SparseStructureFlowModel`
```
resolution 16, in_channels 8, out_channels 8, model_channels 1536, cond_channels 1024,
num_blocks 30, num_heads 12 (head_dim 128), mlp_ratio 5.3334, pe_mode rope,
share_mod true, qk_rms_norm true, qk_rms_norm_cross true, image_attn_mode proj,
dtype bfloat16.  (NO proj_in_channels key → proj_in defaults; SS proj cond is DENSE.)
```
Dense DiT: flatten 16³=4096 tokens → Linear(8→1536) → 30× ModulatedTransformerCrossBlock
(adaLN full self-attn + proj cross-attn + MLP) → F.layer_norm(non-affine) → Linear(1536→8)
→ reshape [1,8,16,16,16]. RoPE phases precomputed on the 16³ grid.

### SS decoder — `ss_dec_conv3d_16l8_fp16` → `SparseStructureDecoder`
```
out_channels 1, latent_channels 8, num_res_blocks 2, num_res_blocks_middle 2,
channels [512, 128, 32], use_fp16 true, norm_type "layer" (default ChannelLayerNorm32).
```
Dense 3D-conv VAE decoder (`sparse_structure_vae.py:210`):
`Conv3d(8→512,k3p1)` → 2× ResBlock3d(512) middle → for ch in [512,128,32]:
2× ResBlock3d(ch) + (UpsampleBlock3d(ch→next) if not last) → out:
`ChannelLayerNorm32 + SiLU + Conv3d(32→1,k3p1)`.
UpsampleBlock3d = `Conv3d(ch→next*8,k3p1)` + **`pixel_shuffle_3d(·,2)`** (#7).
Input 16³ → ×2 ×2 = **64³** logits [1,1,64,64,64]. `>0` → occ; max_pool3d ratio
64//32=2 → res-32 occupancy → coords [N,4]. (torso fp16, in/out cast to fp32.)

### Shape SLat flow LR/HR — `slat_flow_img2shape_dit_1_3B_{512,1024}` → `ElasticSLatFlowModel`
```
resolution {32, 64}, in_channels 32, out_channels 32, model_channels 1536,
cond_channels 1024, num_blocks 30, num_heads 12, mlp_ratio 5.3334, pe_mode rope,
share_mod true, qk_rms_norm true, qk_rms_norm_cross true, image_attn_mode proj,
proj_in_channels 2048, dtype bfloat16.
```
Sparse DiT (`structured_latent_flow.py:190`): SparseLinear(32→1536) → 30×
ModulatedSparseTransformerCrossBlock (sparse adaLN **full varlen** self-attn + proj
cross-attn + MLP) → sparse F.layer_norm(non-affine) → SparseLinear(1536→32).
`resolution` only sizes the (unused at inference, rope) PE; coords drive sparse RoPE.

### Tex SLat flow — `slat_flow_imgshape2tex_dit_1_3B_1024` → `ElasticSLatFlowModel`  [PHASE-2]
Same as above but **in_channels 64** (= 32 noise + 32 channel-concat shape_slat,
re-normalized), out_channels 32. concat done inside forward (`:212-213`).

### Shape decoder — `shape_dec_next_dc_f16c32_fp16` → `FlexiDualGridVaeDecoder`
```
resolution 256 (overridden per-decode via set_resolution(actual_hr_resolution)),
model_channels [1024,512,256,128,64], latent_channels 32, num_blocks [4,16,8,4,0],
block_type 5× SparseConvNeXtBlock3d, up_block_type 4× SparseResBlockC2S3d,
voxel_margin 0.5 (default), use_fp16 true.  out head = 7 ch.
```
Head split (`fdg_vae.py:100-102`, eval): `vertices = (1+2·0.5)·sigmoid(h[...,0:3]) − 0.5`
= `2·sigmoid − 0.5`; `intersected = (h[...,3:6] > 0)`; `quad_lerp = softplus(h[...,6:7])`.
→ `flexible_dual_grid_to_mesh(coords[:,1:], vertices, intersected, quad_lerp,
aabb=[[-.5]*3,[.5]*3], grid_size=resolution)` (O-Voxel, #16).
`.upsample(slat, upsample_times=4)` reuses this backbone, early-exits returning coords.

### Tex decoder — `tex_dec_next_dc_f16c32_fp16` → `SparseUnetVaeDecoder`  [PHASE-2]
Same backbone shape, **out_channels 6** (base_color3+metallic1+roughness1+alpha1),
**pred_subdiv false** (reuses shape's `subs` as `guide_subs`). `·0.5+0.5`.

---

## IMAGE_COND_CONFIGS (inference.py:26-55) — the 4 proj conditioners

All `DinoV3ProjFeatureExtractor` over `camenduru/dinov3-vitl16-pretrain-lvd1689m`
(DINOv3 ViT-L/16: hidden 1024, 24 layers, 16 heads, patch 16, 4 register tokens,
2D-axial RoPE on patch tokens only, LayerScale; final = plain `F.layer_norm`):

| key | image_size | grid_resolution | NAF | proj C |
|---|---|---|---|---|
| ss | 512 | **16** | none | 1024 (DINO only) |
| shape_512 | 512 | 32 | NAF→512 | **2048** (raw 1024 ‖ NAF 1024) |
| shape_1024 | 1024 | 64 (override → actual_hr//16) | NAF→512 | 2048 |
| tex_1024 | 1024 | 64 | NAF→512 (env-capped) | 2048 |

**M1 takes only `ss` (grid 16, NO NAF)** → proj C = 1024 (matches SS DiT cond/proj
path; ss config has no `proj_in_channels` so it ingests the 1024-wide DINO map
directly via the proj cross-attn). NAF (#6) only enters at M2 (shape_512+).

---

## Decode-shape data flow (token counts; from run())
1. SS coords: occupancy @ res-32 grid → N voxels (few k–tens of k).
2. LR slat @ grid-32 (lr_resolution 512): SparseTensor[N, 32].
3. upsample ×4 → hr_coords; quantize to `grid_res = actual_hr//16` via
   `((hr_coords[:,1:]+0.5)/512*(grid_res−1)).round().int()`, `.unique(dim=0)` → M tokens.
   Token-budget loop steps actual_hr down by 128 (floor 1024) if M > PIXAL3D_MAX_TOKENS
   (49152) — for low_vram/1024 path, `actual_hr_resolution <= 1024` breaks immediately
   so actual_hr = 1024, grid_res = 64.
4. HR slat @ grid-64: SparseTensor[M, 32] → denorm → **shape_slat** (final geom latent).
5. decode_shape_slat(set_resolution(1024)) → mesh.

## Precision note for port validation
Flow torsos run **bf16**, VAE/decoders **fp16** (in/out layers cast to fp32). Goldens
carry that noise. C++ port stays fp32 (correctness-first). Expect validation tol vs
goldens ~1e-2 rel on flow-model outputs (z_s, slats), ~1e-3 on fp16 decoder paths.
Judge correctness by E2E mesh agreement (verts/faces), not a fixed elementwise tol.
