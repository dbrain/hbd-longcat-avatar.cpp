# shape_slat_encoder — native-texturing rung-1 ENCODER port

`shape_slat = shape_slat_encoder(voxelized_mesh)` — the **encoder** half of the FlexiDualGrid
sparse-VAE. Its DECODER (`shape_dec` / `tex_dec`, `next_dc_f16c32`) is already ported & validated
(`sparse_vae.hpp` + `sparse_vae_pipeline.hpp`). This encoder is the decoder run **in reverse**:
`input_layer → 5 down-stages → final LayerNorm → to_latent → mean`.

## Files added
| file | role |
|------|------|
| `extract_shape_enc.py` | safetensors → per-tensor fp32 `.npy` (CPU/numpy, no torch). Transposes the 40 sparse-conv kernels torch `[OC,3,3,3,IC]` → spike `[V=27,IC,OC]`; Linears/norms kept verbatim. |
| `shape_slat_encoder.hpp` | the encoder graph (`senc::shape_slat_encode`), mirroring the decoder. Reuses `sparse_vae.hpp`'s validated ops; adds the net-new `s2c_shrink`/`s2c_scatter`/`s2c_skip_mean` (Spatial2Channel downsample). |
| `shape_slat_encoder_test.cpp` | CPU compile + gguf/npy load + tensor-shape + s2c sanity check (`./build.sh shape_slat_encoder_test`). |

## Source weights & GGUF
- safetensors: `/mnt/hdd/pixal3d_tex/trellis2_4b/ckpts/shape_enc_next_dc_f16c32_fp16.safetensors`
  — **708,797,208 bytes (~676 MB), 284 tensors, F16**. Config: `…/shape_enc_next_dc_f16c32_fp16.json`.
- per-tensor npy: `weights_npy/shape_enc/` (284 `.npy`, fp32).
- **GGUF (fp32): `/mnt/hdd/pixal3d/weights_gguf/shape_enc.gguf` — 1,417,558,016 bytes (~1417 MB), 284 tensors.**
  (fp32, same dir convention as `shape_dec.gguf`; loads via the dependency-free `gguf_reader.hpp`,
  env `PIXAL3D_GGUF_DIR=/mnt/hdd/pixal3d/weights_gguf`.)

Load check (`shape_slat_encoder_test`): **284/284 tensors present with the exact element counts the
arch predicts**, via BOTH the GGUF and the npy dir; s2c downsample sane. PASS.

## Architecture (encoder vs decoder)
Config `FlexiDualGridVaeEncoder` (`SparseUnetVaeEncoder`):
`model_channels=[64,128,256,512,1024]`, `latent_channels=32`, `num_blocks=[0,4,8,16,4]`,
block=`SparseConvNeXtBlock3d`, down_block=`SparseResBlockS2C3d`.

| | DECODER (`tex_dec`/`shape_dec`, ported) | ENCODER (`shape_enc`, this port) |
|--|--|--|
| class | `SparseUnetVaeDecoder` | `SparseUnetVaeEncoder` |
| entry | `from_latent` Linear **32→1024** | `input_layer` Linear **6→64** |
| channels | **1024→64** (descend) | **64→1024** (ascend) |
| num_blocks order | `[4,16,8,4,0]` | `[0,4,8,16,4]` |
| resample block | `SparseResBlockC2S3d` (Channel2Spatial **UP**, `c2s_grow`) | `SparseResBlockS2C3d` (Spatial2Channel **DOWN**, `s2c_shrink`) |
| exit | LN(non-affine) + `output_layer` 64→7 (FDG) / 64→6 (PBR) | LN(non-affine) + `to_latent` 1024→**64** → chunk(mean[32],logvar[32]) |
| output | mesh / PBR voxels | `shape_slat` feats `[N,32]` = **mean** (`sample_posterior=False`) |

Encoder forward (`senc::shape_slat_encode`), 5 stages, stage `i`:
1. `input_layer` 6→64. (Input feats6 = `[vertices.feats-0.5 (3), intersected.feats.float()-0.5 (3)]`,
   concatenated by the caller, per `FlexiDualGridVaeEncoder.forward`.)
2. `num_blocks[i]` ConvNeXt blocks @ `model_channels[i]` (conv→LN(affine)→MLP(Lin,SiLU,Lin)→residual).
3. if `i<4`: one `SparseResBlockS2C3d` `MC[i]→MC[i+1]`, named `blocks.i.num_blocks[i]`
   (same boundary-block naming as the decoder; confirmed against the safetensors keys):
   `h=conv1(silu(norm1(x)))[C→Cout/8]; h=S2C(h),x=S2C(x)[downsample ×8 ch]; h=conv2(silu(norm2(h)))[Cout→Cout]; h+=skip(x)`,
   `skip = reshape(x_down,[M,Cout,8C/Cout]).mean(-1)`.
4. after stage 4 (no down): final `F.layer_norm` (non-affine, eps 1e-5) + `to_latent` 1024→64 +
   chunk → `mean[32]` = `shape_slat`.

### The net-new op: Spatial2Channel downsample (exact inverse of decoder `c2s_grow`)
`SparseSpatial2Channel(factor=2)`: child voxel `(c1,c2,c3)` → parent `(c1//2, c2//2, c3//2)` (floor div),
slot `subidx = (c1%2) + (c2%2)*2 + (c3%2)*4`. Output voxels = the **unique** parent coords, sorted by
linearized z-major code (matches torch `code.unique()`). Each child scatters its `Cin`-vector into
slot `subidx` of its parent's 8-slot block (missing slots zero) → feats `[M, 8*Cin]`. Implemented in
`shape_slat_encoder.hpp` as `s2c_shrink` (rulebook) + `s2c_scatter` + `s2c_skip_mean`. Tensor-level
ConvNeXt/conv/LN/SiLU/Linear are reused verbatim from `sparse_vae.hpp` (validated ~1e-7 on these layers).

## Tensor name groups (284 total)
- `input_layer.{weight[64,6],bias[64]}` — entry Linear 6→64.
- `to_latent.{weight[64,1024],bias[64]}` — exit Linear 1024→2·32.
- ConvNeXt blocks `blocks.{i}.{j}.` for `j<num_blocks[i]`: `conv.{weight[27,C,C],bias}`,
  `norm.{weight,bias}[C]`, `mlp.0.{weight[4C,C],bias}`, `mlp.2.{weight[C,4C],bias}`.
- S2C down blocks `blocks.{i}.{num_blocks[i]}.` (i=0..3): `norm1.{weight,bias}[C]`,
  `conv1.{weight[27,C,Cout/8],bias[Cout/8]}`, `conv2.{weight[27,Cout,Cout],bias[Cout]}` (norm2 is non-affine → no weights).
- (conv weights stored spike `[27,Cin,Cout]`; Linear weights torch `[out,in]`.)

## Validation plan (later, on GPU)
The encoder is **authored + compiles + loads**, but **not yet run end-to-end** — it needs a voxelized
input that does not exist as a banked artifact. The banked goldens are encoder **outputs** only:

- `/mnt/hdd/pixal3d_tex/golden_{69k,usorig,usdense}_1024/shape_slat_{feats,coords}.npy`
  (+ `golden_69k_512`, `golden_ultrashape_512`). e.g. `golden_69k_1024`: `coords [7390,4] int32`
  (grid-64, range 0..63), `feats [7390,32] fp32` = the `shape_slat` mean.

To validate `senc::shape_slat_encode`:
1. **Produce the voxelized input the goldens were made from.** Run the Python reference
   (`FlexiDualGridVaeEncoder`) on the SAME source mesh at grid-1024, capturing the encoder's TRUE
   input — the sparse tensor fed to `input_layer`: `coords [N,4]` (grid-1024) and the 6-channel feats
   `[vertices.feats-0.5, intersected.feats-0.5]` (i.e. the o_voxel `dual_vertices` + `intersected`
   outputs). Bank these as `refs/shape_enc/voxel_{coords,feats6}.npy` (mirrors how the decoder banks
   `refs/stage4/*`). NOTE the goldens were generated with `_512` and `_1024` variants — capture the
   matching grid so the down-stages land on the golden grid-64 coords.
2. **Run the C++ encoder** with that voxel input and compare `out_coords` (coord-set IoU, expect 1.0)
   and `feats[N,32]` vs the golden `shape_slat_{coords,feats}` (expect maxabs ~1e-5 fp32 / ~5e-2 on
   the spike CUDA conv — same tolerance band as `m6_tex_decode_test`). A `shape_enc_e2e_test.cpp`
   would follow the `m6_tex_decode_test.cpp` pattern: load voxel input + golden, call
   `senc::shape_slat_encode`, diff.

## Remaining blocker: the o_voxel forward voxelizer (OUT OF SCOPE here)
The full native-texturing path is `mesh → voxelize → shape_slat_encoder → … → texture`. The
**voxelizer** `o_voxel.mesh_to_flexible_dual_grid` (`mesh_to_flexible_dual_grid` →
`_C.mesh_to_flexible_dual_grid_cpu` in `o_voxel/_C…so`) is **not ported** and is the remaining blocker
for an end-to-end texture run. It takes `(vertices, faces, grid_size/voxel_size, aabb)` and returns the
sparse occupied-voxel `coords [N,3]`, per-voxel `dual_vertices`, and `intersected` flags — i.e. the
exact `coords` + the 6-channel feats that feed this encoder.

Porting it entails a **dual-grid QEF solver**: for each grid cell intersected by the mesh, gather the
incident triangle planes + boundary/regularization terms (`face_weight`, `boundary_weight`,
`regularization_weight=0.1`) and solve a small per-voxel quadratic-error-function least-squares for the
dual vertex position, plus the per-axis edge-intersection (`intersected`) flags via mesh–edge tests.
The reverse op `flexible_dual_grid_to_mesh` is ALREADY ported (`svae::flexible_dual_grid_to_mesh` in
`sparse_vae.hpp`); the forward solver is the missing piece. (It runs on CPU in the reference despite
the name; a from-scratch C++ port — triangle-cell rasterization + per-voxel 3×3 normal-equation solve —
is the work. Until then, validate the encoder against banked Python-captured voxel inputs as above.)
