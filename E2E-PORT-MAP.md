# Pixal3D (TRELLIS-2) Image→3D — End-to-End Port Map

Stage-by-stage map of the **geometry-only** path (image → untextured mesh / GLB) for a
methodical C++/ggml port. Texture/PBR stages are noted but marked **PHASE-2**.

- **Python source:** `/mnt/hdd/3d/avatar-shootout/Pixal3D/`, package `pixal3d/`.
- **Live pipeline file:** `pixal3d/pipelines/pixal3d_image_to_3d.py`, class `Pixal3DImageTo3DPipeline`.
  (NOTE: `trellis2_image_to_3d.py` is the older sibling — the *same algorithm* but the entry
  point `inference.py` imports `Pixal3DImageTo3DPipeline`. All `run()` line numbers below are
  from `pixal3d_image_to_3d.py`. Some sub-agent citations reference `trellis2_image_to_3d.py`
  for the model/module internals, which are shared verbatim.)
- **Entry point:** `inference.py` → `run_inference()` → `pipeline.run(...)` (`inference.py:248`).
- **Default pipeline type:** `'1536_cascade'` (standard) / `'1024_cascade'` (low_vram).
  `inference.py:246`: `pipeline_type = f"{1024 if low_vram else 1536}_cascade"`.

This pipeline is **TRELLIS-2 "proj" mode**: a 4-tier DINOv3 image conditioner with
camera-aware voxel projection, a dense sparse-structure DiT, a cascaded sparse SLat DiT
(LR 512 → HR 1024/1536), a sparse VAE decoder, and an O-Voxel flexible-dual-grid mesh
extractor. The "shape" path is geometry; the "tex" path is PHASE-2.

---

## 0. Pipeline Overview (geometry path)

```
image (PIL)
  │
  ├─[PRE-A] preprocess_image  → rembg (BiRefNet) + alpha bbox crop + recomposite on bg  → square RGB
  │          (inference.py runs this with preprocess_image=False inside run())
  ├─[PRE-B] MoGe-2 camera estimation → {camera_angle_x, distance, mesh_scale}   (inference.py only)
  │
  ▼  pipeline.run(image, camera_params, pipeline_type, max_num_tokens)
  │
  STAGE 1  Sparse Structure (DENSE DiT @ 16³)
    get_proj_cond_ss → DINOv3+proj cond     (image_cond_model_ss, grid 16, 512px, no NAF)
    sample_sparse_structure:
       flow:  noise [1,Cin,16,16,16] --flow-ODE--> z_s [1,8,16,16,16]
       decode: sparse_structure_decoder(z_s)>0 --> occ [1,1,64,64,64] bool
       → coords [N,4] int32  (batch,x,y,z) of occupied voxels @ res 32 (after maxpool)
  │
  STAGE 2  Shape SLat LR @ 512  (SPARSE DiT)
    get_proj_cond_shape (image_cond_model_shape_512, grid 32, 512px, NAF→512)
    sample_shape_slat: SparseTensor noise[N,Cin] --flow-ODE--> lr_slat[N,8] (denorm)
  │
  STAGE 3a  Upsample LR→HR coords
    shape_slat_decoder.upsample(lr_slat, ×4)  → hr_coords [N',4]
    quantize/unique → hr_coords_unique [M,4] @ grid_res = hr_res//16 (token-budget loop)
  │
  STAGE 3b  Shape SLat HR @ 1024/1536  (SPARSE DiT)
    get_proj_cond_shape (image_cond_model_shape_1024, grid 64, 1024px, NAF→512)
    flow: SparseTensor noise[M,Cin] --flow-ODE--> hr_slat[M,8] → shape_slat (denorm)
  │
  STAGE 4  Texture SLat  ──────────────────── PHASE-2 (skip for geometry)
    get_proj_cond_shape (image_cond_model_tex_1024) ; sample_tex_slat (concat_cond=shape_slat)
  │
  STAGE 5  Decode → mesh
    decode_shape_slat: shape_slat_decoder(shape_slat, return_subs=True)
       → Mesh (vertices, faces) via O-Voxel flexible_dual_grid_to_mesh ; subs (for tex)
    [PHASE-2] decode_tex_slat(tex_slat, guide_subs=subs) → per-voxel PBR attrs
    m.fill_holes()
  │
  ▼  o_voxel.postprocess.to_glb(...)   → GLB  (inference.py:265)   ─── PHASE-2 for textured GLB;
                                              geometry-only can export vertices/faces directly.
```

**Flow models (the 3 DiTs) and the VAE/decoders are the heavy compute. The DINOv3
conditioner runs 4× (once per stage). Mesh extraction is GPU-hashmap + custom CUDA.**

---

## 1. Stage table (I/O tensors, model key, arch, ops, VRAM)

Channel counts marked `?` come from the HF `pipeline.json` (fetched at load, **not on disk** —
see Open Questions). The latent channel widths used below (`8` for shape slat, `7` for the
shape-decoder output head) are read from code; confirm exact `in_channels`/`model_channels`
against the downloaded config before porting.

---

### PRE-A — Image preprocessing (`preprocess_image`, `pixal3d_image_to_3d.py:144-186`)

| | |
|---|---|
| **Input** | PIL image (RGB or RGBA), arbitrary size |
| **Output** | square RGB PIL image, object centered, black bg |
| **Model** | `rembg_model` = **BiRefNet** (`pipelines/rembg/BiRefNet.py`) — only if no usable alpha |
| **Ops** | LANCZOS resize (≤1024), BiRefNet matting → alpha, `alpha>0.8` bbox, center+1.1× crop, `rgb*a + bg*(1-a)` recomposite |
| **Portable?** | BiRefNet = standard conv/transformer segmentation net. **PHASE-2-ish:** can be replaced by host-side rembg or skipped if input already has alpha. Not on the critical geometry math path — recommend running matting outside the ggml core (CPU/host or a separate model). |
| **VRAM** | BiRefNet loaded to GPU for this call then offloaded (low_vram). Light. |

### PRE-B — Camera estimation (`inference.py:139-157`, MoGe-2)

| | |
|---|---|
| **Input** | preprocessed RGB image |
| **Output** | `{camera_angle_x (rad), distance, mesh_scale}` |
| **Model** | **MoGe-2** (`Ruicheng/moge-2-vitl`) — external monocular-geometry ViT; only intrinsics `fx` is used → `camera_angle_x = 2·atan(W/(2·fx))`; `distance` from `distance_from_fov()` |
| **Portable?** | **Optional / replaceable.** `--fov` CLI flag bypasses MoGe entirely with a manual FOV. For the port, treat camera params as **scalar inputs** (estimate them however; MoGe need not be ported). This is the cleanest cut-line: the ggml core takes `(camera_angle_x, distance, mesh_scale)` as floats. |
| **VRAM** | MoGe loaded after rembg offloads (never co-resident); freed before pipeline. Light. |

---

### STAGE 1 — Sparse Structure (dense DiT @ 16³)

**Driver:** `run()` `:673-686`; `get_proj_cond_ss` `:192-227`; `sample_sparse_structure` `:301-348`.

**1a. Conditioning** — `get_proj_cond_ss` → `image_cond_model_ss`
| | |
|---|---|
| **Model** | `DinoV3ProjFeatureExtractor` (config `ss`: `dinov3-vitl16`, image 512, grid_res **16**, **no NAF**) |
| **Input** | `[image]` (PIL), `camera_angle_x`, `distance`, `mesh_scale` (scalars) |
| **Output** | `cond = {'global': z_global [B,5,1024], 'proj': z_proj [B, 16³, C_proj]}` ; `neg_cond` = zeros_like |
| **Arch** | DINOv3-ViT-L/16: hidden 1024, 24 layers, 16 heads, patch 16, **4 register tokens**, **2D axial RoPE on patch tokens only**, **LayerScale** ×2/block, gated-MLP OFF. Final norm = unweighted `F.layer_norm` (NOT the trained final LN — deliberate). `z_global` = `cat[CLS, 4 register tokens]` = `[B,5,1024]`. |
| **proj op** | voxel grid `linspace(-1,1,16)³` → axis-swap to Blender frame → `/mesh_scale/2` → perspective project (focal `16/tan(angle/2)·res/32`) → pixel → normalize `[-1,1]` → **`F.grid_sample(fmap, pts, mode=bilinear, align_corners=False, padding_mode='border')`** → `[B,C,16³]`. (No NAF at grid 16.) |
| **File** | `pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py` (extractor `DinoV3ProjFeatureExtractor`, `ProjGrid`, projection math `:27-135, 151-232`) |

**1b. Flow sampling** — `sample_sparse_structure`, model key **`sparse_structure_flow_model`**
| | |
|---|---|
| **Input** | `noise = randn(1, in_channels, 16,16,16)` (dense) ; `cond/neg_cond` from 1a |
| **Output** | `z_s [1, 8, 16, 16, 16]` (dense SS latent) |
| **Arch** | `SparseStructureFlowModel` (`models/sparse_structure_flow.py:56`). Dense DiT: flatten 16³→4096 tokens, `Linear(in→model_ch)`, APE (sinusoidal) or RoPE, N× `ModulatedTransformerCrossBlock` (adaLN self-attn **full** + proj cross-attn + MLP), final `F.layer_norm` + `Linear(→out)`, reshape to `[B,out,16,16,16]`. `image_attn_mode='proj'`. |
| **Sampler** | `FlowEulerGuidanceIntervalSampler` (see §3). steps=12 default, CFG strength 7.5, rescale 0.7, rescale_t 5.0. |

**1c. SS decode + occupancy** — model key **`sparse_structure_decoder`**
| | |
|---|---|
| **Input** | `z_s [1,8,16,16,16]` |
| **Output** | `decoded = decoder(z_s) > 0` → bool `[1,1,64,64,64]`; max_pool3d ↓ to res 32; `coords [N,4] int32` via `argwhere` |
| **Arch** | `SparseStructureDecoder` (`models/sparse_structure_vae.py:210`). Dense 3D conv VAE decoder: `Conv3d(in)`, middle ResBlock3d×2, per-stage ResBlock3d + **`UpsampleBlock3d`** (`Conv3d→pixel_shuffle_3d ×2`), out `GroupNorm/LayerNorm + SiLU + Conv3d`. Std dense 3D convs + **`pixel_shuffle_3d`** (`modules/spatial.py:4-13`, net-new reshape). |
| **VRAM** | Flow model + DINOv3 are the load here. Decoder small. Output coords typically a few-thousand to ~tens-of-thousands voxels @ res 32. |

---

### STAGE 2 — Shape SLat LR @ 512 (sparse DiT)

**Driver:** `run()` `:688-700`; `get_proj_cond_shape` `:229-295`; `sample_shape_slat` `:350-388`.

**2a. Conditioning** — `get_proj_cond_shape(image_cond_model_shape_512, coords)`
| | |
|---|---|
| **Model** | `DinoV3ProjFeatureExtractor` (config `shape_512`: image 512, grid_res **32**, **NAF→512**) |
| **Input** | `[image]`, `coords [N,4]` (Stage-1 occupied voxels), scalars |
| **Output** | `cond = {'global': z_global [B,5,1024], 'proj': SparseTensor(feats=z_proj_sparse[N,C], coords)}` ; neg = zeros |
| **proj op** | Same projection as Stage 1, but feature map first **NAF-upsampled** to 512×512 (guide = RGB), then the LR (raw) + HR (NAF) sampled features are **concatenated** → `C_proj = 2·1024 = 2048`. z_proj reshaped to `[B,32,32,32,C]` and **gathered at `coords`** → per-voxel SparseTensor (`:275-281`). |
| **NAF** | `valeoai/NAF` "Neighborhood Attention Feature" upsampler (`torch.hub`). Guide-conditioned: 2 conv encoders + RoPE + **9×9 neighborhood (windowed) cross-attention** via `natten.na2d` CUDA. **Net-new + heavy.** |

**2b. Flow sampling** — `sample_shape_slat`, model key **`shape_slat_flow_model_512`**
| | |
|---|---|
| **Input** | `noise = SparseTensor(randn(N, in_channels), coords)` ; cond from 2a |
| **Output** | `lr_slat = SparseTensor[N, 8]`, then **denormalized** `slat = slat*std + mean` (`shape_slat_normalization`) |
| **Arch** | `SLatFlowModel` (`models/structured_latent_flow.py:15`). Sparse DiT: `SparseLinear(in→model_ch)`, sparse APE on `coords[:,1:]`, N× **`ModulatedSparseTransformerCrossBlock`** (sparse adaLN self-attn + proj cross-attn + MLP), final sparse layer_norm + `SparseLinear(→out)`. `image_attn_mode='proj'`. Self-attn = **full varlen** (per-batch all-to-all) for `attn_mode='full'`; some configs use windowed/double-windowed (config-driven). |
| **Sampler** | `FlowEulerGuidanceIntervalSampler`. shape params: steps 12, strength 7.5, rescale 0.5, rescale_t 3.0. |
| **VRAM** | Flow model + DINOv3+NAF. NAF na2d is a VRAM spike. N voxels @ res-32 grid. |

---

### STAGE 3a — Upsample LR→HR coords (no flow)

**Driver:** `run()` `:702-734`.
| | |
|---|---|
| **Model** | **`shape_slat_decoder`** `.upsample(lr_slat, upsample_times=4)` |
| **Input** | `lr_slat [N,8]` (denorm SparseTensor @ res 32 grid, lr_resolution=512) |
| **Output** | `hr_coords [N',4]` — runs the decoder's sparse-conv backbone to predict per-voxel **subdivision masks** (`to_subdiv` SparseLinear C→8, `>0`), splitting voxels ×2 four times; returns coords only (early-exit at stage `i==upsample_times`). |
| **Post** | token-budget loop (`:713-728`): quantize `hr_coords` to `grid_res = hr_res//16`, `unique`, count tokens; if `> max_num_tokens` (env `PIXAL3D_MAX_TOKENS`) step `hr_res -= 128` down to floor (env `PIXAL3D_HR_RES_FLOOR=1024`). Result `hr_coords_unique [M,4]` + `actual_hr_resolution`. |
| **Arch** | Sparse VAE decoder backbone (`FlexiDualGridVaeDecoder`→`SparseUnetVaeDecoder`, `models/sc_vaes/`). Submanifold sparse conv (flex_gemm), `SparseResBlockUpsample3d` / `SparseResBlockC2S3d`, `SparseUpsample`/`SparseChannel2Spatial` (coords-growing). |
| **VRAM** | Decoder loaded; M can reach tens-of-thousands → hundreds-of-thousands of voxels. This is where token budget guards OOM. |

### STAGE 3b — Shape SLat HR @ 1024/1536 (sparse DiT)

**Driver:** `run()` `:736-766` (inlined, mirrors `sample_shape_slat`).
| | |
|---|---|
| **Conditioning** | `get_proj_cond_shape(image_cond_model_shape_1024, hr_coords_unique, grid_resolution_override=actual_hr_res//16)` — config `shape_1024`: image **1024**, grid_res **64**, NAF→512. Proj grid resolution overridden to match HR. Output `proj` SparseTensor[M, 2048], `global [B,5,1024]`. |
| **Flow model** | **`shape_slat_flow_model_1024`** — same `SLatFlowModel` arch as Stage 2, HR token count M. |
| **Input/Output** | `noise = SparseTensor(randn(M, in_channels), hr_coords_unique)` → `hr_slat[M,8]` → denorm → **`shape_slat`** (the final geometry latent). |
| **Sampler** | `FlowEulerGuidanceIntervalSampler`, shape params (same as Stage 2). |
| **VRAM** | **Heaviest stage.** HR sparse DiT over M tokens with full/windowed attention + DINOv3@1024 + NAF. Token-budget loop exists precisely to keep this within VRAM. |

---

### STAGE 4 — Texture SLat — **PHASE-2** (skip for geometry)

**Driver:** `run()` `:768-782`; `sample_tex_slat` `:510-551`.
| | |
|---|---|
| **Conditioning** | `get_proj_cond_shape(image_cond_model_tex_1024, shape_slat.coords)` — config `tex_1024`: image 1024, grid 64, NAF→512 (NAF target env-capped to 512: na2d@1024 OOMs 12GB — see VRAM notes). |
| **Flow model** | **`tex_slat_flow_model_1024`** (`SLatFlowModel`). noise has `in_channels − shape_slat.feats.shape[1]` channels; `concat_cond=shape_slat` channel-concatenated **inside model forward** (`structured_latent_flow.py:212-213`). shape_slat is re-normalized `(x-mean)/std` before concat. |
| **Sampler** | `FlowEulerGuidanceIntervalSampler`, tex params: steps 12, strength **1.0** (→ CFG off, single forward), rescale 0.0, rescale_t 3.0. |
| **Output** | `tex_slat` (denorm with `tex_slat_normalization`). |
| **Note** | Needed only for textured/PBR GLB. Geometry-only port can stop after Stage 3b. |

---

### STAGE 5 — Decode → mesh (geometry) + PHASE-2 texture

**Driver:** `decode_latent` `:574-606`; `decode_shape_slat` `:485-508`; `decode_tex_slat` `:553-572`.

**5a. Shape decode** — model key **`shape_slat_decoder`** (`FlexiDualGridVaeDecoder`)
| | |
|---|---|
| **Input** | `shape_slat [M, 8]` (SparseTensor); `set_resolution(actual_hr_resolution)` first |
| **Backbone** | `decoder(slat, return_subs=True)` → `SparseUnetVaeDecoder.forward` (`models/sc_vaes/sparse_unet_vae.py:478`): `from_latent` SparseLinear → fp16 torso → per-stage sparse-conv ResBlocks + up-blocks (each up-block returns a subdivision `sub`; collected into `subs`) → fp32 → non-affine sparse `F.layer_norm` → `output_layer` SparseLinear → **`[M', 7]`**. |
| **Head split** | `FlexiDualGridVaeDecoder.forward` (`models/sc_vaes/fdg_vae.py:97-110`): ch 0:3 → `vertices = (1+2·margin)·sigmoid(.) − margin` (dual-vertex offset); ch 3:6 → `intersected = (. > 0)` bool per-axis; ch 6:7 → `quad_lerp = softplus(.)` split weight. |
| **Mesh extract** | **`flexible_dual_grid_to_mesh(coords, vertices, intersected, quad_lerp, aabb=[[-.5]*3,[.5]*3], grid_size=res)`** (`o_voxel/convert/flexible_dual_grid.py:142-283`). Dual-contouring-style: 1 vertex per occupied voxel at learned offset; GPU hashmap (`_C.hashmap_insert_3d_idx_as_val_cuda`, `_C.hashmap_lookup_3d_cuda`) gathers edge-neighbor quads on `intersected` edges; quad→2 triangles via `quad_lerp` diagonal choice. → `Mesh(vertices[V,3], faces[T,3])`. |
| **Output** | `meshes` (list of `Mesh`), `subs` (subdivision SparseTensors, fed to tex decode). |

**5b. Texture decode** — **PHASE-2** — model key **`tex_slat_decoder`** (`FlexiDualGridVaeDecoder`, `pred_subdiv=False`)
| | |
|---|---|
| | `tex_slat_decoder(tex_slat, guide_subs=subs)*0.5+0.5` → per-voxel PBR attrs `[M', tex_ch]` (base_color/metallic/roughness/alpha per `pbr_attr_layout`). Uses **shape's** subdivisions (identical topology). No mesh extraction. Same sparse-conv backbone as 5a. |

**5c. Assemble + GLB**
| | |
|---|---|
| | `m.fill_holes()`; wrap as `MeshWithVoxel(vertices, faces, coords, attrs, ...)`. Then (`inference.py:265`) `o_voxel.postprocess.to_glb(vertices, faces, attr_volume, coords, ..., remesh=True, texture_size=4096)` → GLB; apply 4×4 axis rotation; export. **GLB bake (texture, remesh, UV) is PHASE-2.** Geometry-only: export `vertices`/`faces` directly (e.g. OBJ/PLY). |

---

## 2. Net-new primitives to port

Custom CUDA / non-standard ops on the **geometry** path, in rough port order. `[DONE]` = already
implemented in this spike (`HANDOFF-sparse-conv-spike.md`, `PORT-SPEC-flexgemm-submanifold.md`).

| # | Primitive | File:line | Symbol | Changes coords? | Notes |
|---|---|---|---|---|---|
| 1 | **Submanifold sparse conv3d** (3³, stride1, hashmap neighbor-gather→GEMM; weight `(Co,Kd,Kh,Kw,Ci)`) | `modules/sparse/conv/conv_flex_gemm.py:46` | `sparse_submanifold_conv3d` (flex_gemm) | No | **[DONE]** — the spike's core. Backbone of SS/shape decoders. |
| 2 | **2D bilinear `grid_sample`** (mode=bilinear, align_corners=False, padding_mode=border) | `image_conditioned_proj.py:111-135` | `sample_features` | No | **The proj-cond core op.** Sample R³ (or occupied-only) voxel projections from `[B,C,h,w]` DINO map. |
| 3 | **Camera unprojection** (Blender pinhole: voxel grid → world_to_camera `inv(4×4)` → focal `16/tan(α/2)·res/32` → persp-divide → pixel → normalize) | `image_conditioned_proj.py:27-108, 208-223` | `project_points_to_image_batch`, `ProjGrid.forward` | n/a | Pure arithmetic, no lib. Pairs with #2. |
| 4 | **DINOv3 2D axial RoPE** (patch-center coords, `inv_freq` head_dim/4, axial H/W, `.tile(2)`; **patch tokens only**, CLS+4 reg excluded) | `transformers/.../modeling_dinov3_vit.py:75-101,133-250` | `apply_rotary_pos_emb` | n/a | Non-vanilla ViT detail. |
| 5 | **LayerScale** (per-channel `lambda1` ×2/block) | `modeling_dinov3_vit.py:319-325` | `DINOv3ViTLayerScale` | n/a | Trivial but must be present. |
| 6 | **NAF guide-conditioned upsampler** (2 conv encoders + RoPE + **9×9 neighborhood/local cross-attention** via `natten.na2d`) | `~/.cache/torch/hub/valeoai_NAF_main/src/model/naf.py:72-117`, `.../layers/attentions.py:32-76` | `naf`, `CrossAttention` | n/a | **Net-new windowed-attention kernel.** Only for grid 32/64 tiers (Stages 2/3b/4). Biggest conditioner risk. **Tier without NAF (Stage 1, grid 16) skips it — ship that first.** |
| 7 | **pixel_shuffle_3d** (3D depth-to-space; 8-D reshape/permute, `C→C/s³`, spatial `×s`) | `modules/spatial.py:4-13` | `pixel_shuffle_3d` | n/a | SS decoder upsample. Custom op (>4 dims for ggml). |
| 8 | **Full sparse attention (varlen)** — per-batch all-to-all, `cu_seqlens` from layout | `modules/sparse/attention/full_attn.py:184-195` | `sparse_scaled_dot_product_attention` | No | Needs varlen FA kernel. Used by SLat DiTs (self + cross). |
| 9 | **Windowed / double-windowed sparse self-attn** (window-index argsort + varlen FA + un-sort scatter; shifted-window head split) | `modules/sparse/attention/windowed_attn.py:14-130`, `modules.py:117-126` | `sparse_windowed_scaled_dot_product_self_attention`, `calc_window_partition` | No (sort/unsort) | Config-driven per block. Plain window-bucket sort (NOT Z-order/Hilbert). |
| 10 | **3D sparse RoPE** (polar phases from `coords[:,1:]`, cached; interleaved complex rotate) | `modules/sparse/attention/rope.py:23-57` | `SparseRotaryPositionEmbedder` | No | Per-token complex multiply. |
| 11 | **Per-voxel proj-feature linear + additive inject** (`proj_linear(proj.feats) + cross_attn_out`) | `modules/sparse/attention/proj_attention.py:41-42` (gated `:93-97`) | `SparseProjectAttention` | No | Trivial math; **non-standard dict-context `{global,proj}` + strict voxel-alignment**. Dense analogue `ProjectAttention` (`modules/transformer/proj_attention.py:38-48`). |
| 12 | **Sparse upsample (nearest)** — subdivision-mask-driven voxel split (repeat_interleave coords) | `modules/sparse/spatial/basic.py:71-109` | `SparseUpsample` | **Yes** | Decoder up-blocks + `.upsample()`. |
| 13 | **Sparse downsample (avg/max pool)** — `coord//f`, unique, scatter_reduce | `modules/sparse/spatial/basic.py:12-68` | `SparseDownsample` | **Yes** | (Encoder-side; decode uses up only — but needed if encoder ported.) |
| 14 | **Spatial↔Channel reshuffle** (pack/unpack `f³` neighbors into channels; subdivision-gated) | `modules/sparse/spatial/spatial2channel.py:7-94` | `SparseSpatial2Channel` / `SparseChannel2Spatial` | **Yes** | Alt up-block path (`SparseResBlockC2S3d`). |
| 15 | **Subdivision prediction** (SparseLinear C→8 logits, `>0` → 8-bit child occupancy) | `models/sc_vaes/sparse_unet_vae.py:151,156,159` | `to_subdiv` + binarize | drives #12/#14 | Determines coords growth in decode/upsample. |
| 16 | **O-Voxel flexible-dual-grid mesh extraction** (GPU hashmap voxel insert/lookup + edge-quad assembly + `quad_lerp` diagonal split) | `o_voxel/convert/flexible_dual_grid.py:142-283` | `flexible_dual_grid_to_mesh` (`_C.hashmap_insert_3d_idx_as_val_cuda`, `_C.hashmap_lookup_3d_cuda`) | n/a | **Biggest single net-new component.** CUDA `_C.so` source not in tree — reimplement from Python wrapper semantics. |
| 17 | **Vertex/flag/split head split** (sigmoid-offset / `>0` / softplus on 7-ch output) | `models/sc_vaes/fdg_vae.py:100-102` | inline `FlexiDualGridVaeDecoder.forward` | n/a | Trivial pointwise; feeds #16. |
| 18 | **Per-scale spatial cache** (memoized neighbor maps, up/down index maps, subdivision masks keyed by `_scale`) | `modules/sparse/basic.py:773-795` | `register/get_spatial_cache` | n/a | Index plumbing under #1/#8/#9/#12-14. |
| 19 | **Gather/scatter-by-coords + per-batch broadcast + layout (bincount/cumsum)** | `modules/sparse/basic.py:467-471,719-721`; `windowed_attn.py:45-47` | `__cal_layout`, `batch_boardcast_map`, fwd/bwd indices | n/a | Connective tissue for everything sparse. |

**PHASE-2 net-new (texture path):** texture-tier NAF na2d@1024 (#6 at higher res — OOM source),
texture decoder head, `o_voxel.postprocess.to_glb` (remesh/UV/texture bake `rasterize.py`,
`serialize.py`).

---

## 3. The flow-matching sampler (PORTABLE — replicate exactly)

**Class:** all three stages use **`FlowEulerGuidanceIntervalSampler`**
(`pipelines/samplers/flow_euler.py:169`; MRO `GuidanceInterval → CFG → FlowEuler`). Constructor
arg: `sigma_min`. (The exact `name` strings are in the HF-fetched `pipeline.json`; the call
signatures admit no other class — confirm if byte-certainty needed.)

**Schedule** (`flow_euler.py:114-118`): uniform `t = linspace(1, 0, steps+1)`, then warped
`t = rescale_t·t / (1 + (rescale_t−1)·t)` (Möbius warp, fixes 0 and 1; `r>1` packs steps at
high noise). Walk adjacent pairs `(t, t_prev)`, t: 1→0.

**Model call:** timestep is **×1000** before the model's `t_embedder`
(`flow_euler.py:44-46`); the ODE math uses un-scaled `t∈[0,1]`.

**Euler update** (`flow_euler.py:79-81`): `x = x − (t − t_prev)·v`, where `v = model(x, t·1000, cond)`.

**CFG** (`classifier_free_guidance_mixin.py:17`): `v = s·v_pos + (1−s)·v_neg`
(`= v_neg + s·(v_pos−v_neg)`), `s = guidance_strength`. `s=1`→cond-only single forward;
`s=0`→neg-only. **Guidance interval** (`guidance_interval_mixin.py:9-13`): CFG (2 forwards)
only when `g0 ≤ t ≤ g1`; outside → `s=1` (cond-only). **Guidance rescale** (`mixin:20-27`,
if `>0`): convert v_pos and v to x0-space, rescale guided-x0 std to match pos-x0 std, blend by
`guidance_rescale`, convert back.

**Dense vs sparse:** one code path — `SparseTensor` overloads `+ − * .std`, so the loop is
identical; only noise construction differs (Stage 1 dense `[1,C,16,16,16]`; Stages 2/3b/4
`SparseTensor(randn(N,Cin), coords)`).

**concat_cond** (tex only): passed through `**kwargs` to the model, channel-concatenated onto
`x` **inside** `SLatFlowModel.forward` (`structured_latent_flow.py:212-213`); applied on both
pos and neg forwards.

**Per-stage params** (from `inference.py` defaults):

| stage | steps | guidance_strength | guidance_rescale | rescale_t |
|---|---|---|---|---|
| sparse_structure (SS) | 12 | 7.5 | 0.7 | 5.0 |
| shape_slat (LR+HR) | 12 | 7.5 | 0.5 | 3.0 |
| tex_slat (PHASE-2) | 12 | 1.0 (CFG off) | 0.0 | 3.0 |

`guidance_interval` is set in the HF `pipeline.json` `params` (not a CLI default) — read it.

**Port pseudocode** (interval + CFG + rescale inlined):

```
for i in 0..steps:  t_lin[i] = 1 - i/steps
for i in 0..steps:  t_seq[i] = rescale_t*t_lin[i] / (1 + (rescale_t-1)*t_lin[i])
x = noise
for k in 0..steps-1:
    t, t_prev = t_seq[k], t_seq[k+1]
    in_interval = (g0 <= t <= g1)
    tm = t * 1000
    if s==1 or not in_interval:   v = MODEL(x, tm, cond,     concat_cond)
    elif s==0:                    v = MODEL(x, tm, neg_cond,  concat_cond)
    else:
        vp = MODEL(x, tm, cond,    concat_cond)
        vn = MODEL(x, tm, neg_cond, concat_cond)
        v  = s*vp + (1-s)*vn
        if gr > 0:                                    # std-rescale, σ=sigma_min
            x0p = (1-σ)*x - (σ+(1-σ)*t)*vp
            x0c = (1-σ)*x - (σ+(1-σ)*t)*v
            x0  = gr*(x0c*(std(x0p)/std(x0c))) + (1-gr)*x0c   # std over all non-batch dims, per sample, unbiased
            v   = ((1-σ)*x - x0) / (σ+(1-σ)*t)
    x = x - (t - t_prev) * v
samples = x
```
(`std` = torch default unbiased N−1; ×1000 before model; concat_cond inside model; one path
for dense/sparse.)

---

## 4. Portable components (standard ops — already in existing ggml ViT/DiT ports)

- **Dense DiT block** `ModulatedTransformerCrossBlock` (`modules/transformer/modulated.py:180-198`):
  adaLN 6×C (shift/scale/gate × {msa,mlp}), `norm1` non-affine + `scale_msa`/`shift_msa`,
  full self-attn, `gate_msa` residual; `norm2` **affine** + plain (un-modulated, un-gated)
  cross-attn residual; `norm3` non-affine + `scale_mlp`/`shift_mlp` + MLP + `gate_mlp`. Two
  modes: `share_mod` (global `Linear(C,6C)` + per-block bias) vs per-block `SiLU+Linear(C,6C)`.
- **Full MHA** (`modules/attention/full_attn.py` + `modules.py:66-102`): `softmax(QKᵀ/√d)·V`,
  no mask. flash_attn/xformers/sdpa/naive backends — math identical.
- **QK RMS-norm** (`modules/attention/modules.py:9-16`): per-head L2 over head_dim in fp32,
  `× gamma × √head_dim`. Optional.
- **Sinusoidal abs-pos** `AbsolutePositionEmbedder` (`modules/transformer/blocks.py:8-46`):
  per-coord `[sin‖cos]`, base 10000, zero-pad. For 16³ grid → bake the table.
- **3D RoPE (dense)** `RotaryPositionEmbedder` (`modules/attention/rope.py`): axial 3-way,
  interleaved complex, fp32, zero-pad.
- **fp32 norms** `LayerNorm32`/`GroupNorm32`/`ChannelLayerNorm32` (`modules/norm.py`): stock
  norms with forced fp32 (ggml already accumulates fp32). `eps=1e-6` in DiT.
- **Dense 3D conv VAE** `SparseStructureEncoder`/`Decoder` (`models/sparse_structure_vae.py`):
  `Conv3d` + `ResBlock3d` + `Down/UpsampleBlock3d` + GroupNorm/LayerNorm + SiLU. Standard
  except `pixel_shuffle_3d` (#7) and `pixel_unshuffle`.
- **Sparse per-feature ops**: `SparseLinear`, `SparseReLU/SiLU/GELU`, `SparseLayerNorm`,
  `SparseFeedForwardNet`, `SparseMultiHeadRMSNorm` — all operate on `.feats` then `.replace`.
- **TimestepEmbedder** (`models/sparse_structure_flow.py:12`): sinusoidal + 2-layer MLP. Standard.
- **The whole flow-matching sampler** (§3) — pure tensor math.
- **MoGe-2** (camera) and **BiRefNet** (rembg) — replaceable / host-side; not on the geometry
  math path.

---

## 5. Data layout the port must replicate

`SparseTensor` (`modules/sparse/basic.py:343`): `{feats [N,C], coords [N,4]}` where
`coords[:,0]`=batch, `coords[:,1:4]`=`(x,y,z)` int32 in `[0,1023]` (10 bits/axis). **Same-batch
voxels must be contiguous** (varlen attention + norms depend on it). `layout` = per-batch
`[start,stop)` slices from `bincount(coords[:,0]).cumsum`. `_scale` (Fraction triple) tracks
cumulative resolution and keys the `_spatial_cache` (neighbor maps, serialization orders, RoPE
phases, up/down index maps). `.replace(feats)` = same topology, new feats (used after every
linear/norm/attn). Submanifold conv keeps coords fixed → **active-voxel set only grows at
up-blocks**, driven by predicted subdivision masks.

---

## 6. Per-stage VRAM notes (low_vram = load-per-stage, offload-after)

- **low_vram** (`inference.py:86`): models stay on CPU, each loaded to GPU per stage then
  `.cpu()` — peak ≈ one flow model + one DINOv3, not all ~18GB. NAF weights pre-downloaded.
- **Stage 3b (Shape HR)** is the **heaviest**: HR sparse DiT over M tokens (full/windowed attn)
  + DINOv3@1024 + NAF. The **token-budget loop** (`run():713-728`, env `PIXAL3D_MAX_TOKENS`
  default 49152, `PIXAL3D_HR_RES_FLOOR` default 1024) steps `hr_res` down by 128 to avoid OOM on
  complex geometry / 12GB cards. `1536_cascade` is std; `1024_cascade` for low_vram.
- **Texture NATTEN (PHASE-2) is the canonical OOM**: `tex_1024` NAF `na2d` at 1024 OOMs a 12GB
  card → `PIXAL3D_TEX_NAF_TARGET` env-capped to 512 (`inference.py:46-54`). The 9×9 neighborhood
  attention is "the single heaviest op."
- **NAF na2d** (Stages 2/3b/4) is a recurring VRAM spike at the conditioner.
- **rembg / MoGe** are sequenced so they never co-reside with each other or the pipeline
  (`inference.py:191-226`): rembg offloads before MoGe; MoGe freed before `pipeline.run`.
- Decode (Stage 5): `shape_slat_decoder` to GPU with `low_vram=True` flag (weight residency
  only, no algorithm change). Mesh extraction is GPU-hashmap, fp32.

---

## 7. Open questions / things to capture before porting Stage 1

1. **`pipeline.json` is fetched from HF at load** (`TencentARC/Pixal3D`), not on disk
   (`base.py:31-37`). It carries the exact per-model `in_channels`/`model_channels`/`num_blocks`/
   `attn_mode`/`pe_mode`, the **sampler `name`s + `params` (incl. `guidance_interval`)**, and the
   `shape_slat_normalization`/`tex_slat_normalization` mean/std. **Download and capture it first** —
   the `8`/`7` channel widths above are read from code paths, not from the config.
2. **Self-attn mode per SLat block** (full vs windowed vs double_windowed) is config-driven — read
   `attn_mode` for `shape_slat_flow_model_512/1024` from the config. Determines whether #9 (windowed)
   is needed or only #8 (full).
3. **`share_mod`** flag per flow model — changes adaLN wiring (global vs per-block). Read from config.
4. **`o_voxel` and `flex_gemm` CUDA `_C.so` are compiled-only** (no source in tree). Mesh extraction
   (#16) and submanifold conv (#1 [DONE]) semantics were reconstructed from Python wrappers — golden
   tensors are essential for validating #16.
5. **Golden-tensor capture points** (suggested, per stage): (1a) z_global/z_proj; (1b) z_s; (1c)
   coords; (2) lr_slat; (3a) hr_coords_unique + actual_hr_resolution; (3b) shape_slat; (5a)
   vertices/faces. Capture with `torch.manual_seed(seed)` fixed (run() reseeds at `:671`).
6. **`max_pool3d` ratio** in SS decode (`:343-345`): decoder outputs 64³, pooled to res 32 → verify
   the exact ratio/threshold against config `resolution`.
7. **NAF tier** (#6) blocks Stages 2+. **Recommend porting Stage 1 (grid-16, no-NAF) end-to-end
   first** — it exercises the dense DiT, proj grid_sample (#2/#3), SS VAE decode (pixel_shuffle_3d
   #7), and the sampler, without NAF or sparse attention — then add the sparse DiT + NAF for Stages 2/3.
