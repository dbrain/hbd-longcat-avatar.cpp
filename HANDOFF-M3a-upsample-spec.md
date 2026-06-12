> ✅ **DONE + VALIDATED 2026-06-12** (bit-exact CPU+CUDA — see E2E-PORT-KICKOFF PROGRESS LOG). This spec
> was the map; the implemented port is `cpp_port/{m3a_upsample.cpp, sparse_vae.hpp}` + `stage3a_capture.py`.
> Confirmed conventions: C2S packing is CHILD-MAJOR; conv weight `[Co,Kd,Kh,Kw,Ci]`→spike `[27,Cin,Cout]`.
> Kept for reference only.

# M3a — `shape_slat_decoder.upsample(×4)` → hr_coords  (sparse-VAE backbone, the next big net-new)

Reconstructed from `pixal3d/models/sc_vaes/` (Explore, 2026-06-12). **VERIFY every state_dict
name + the C2S feature-packing reshape order against the real code before trusting** — this is
a map, not validated. Golden to hit: `golden_stages/stage3a_up/hr_coords` **[382554, 4] int32**
(input `stage2_out/lr_slat_*` [1126,32]; the upsample runs on the DENORM lr_slat).

## What it computes
`FlexiDualGridVaeDecoder.upsample(slat, upsample_times=4)` (`sparse_unet_vae.py:509-522`):
```
h = from_latent(slat)              # SparseLinear 32->1024, coords unchanged
for i, res in enumerate(self.blocks):     # 5 levels, model_channels [1024,512,256,128,64]
    if i == upsample_times: return h.coords   # early-exit at i==4 (returns coords only)
    for j, block in enumerate(res):
        if i < len(blocks)-1 and j == last:  h, sub = block(h)   # SparseResBlockC2S3d (grows coords ×2)
        else:                                h = block(h)        # SparseConvNeXtBlock3d (coords fixed)
```
num_blocks per level = [4,16,8,4,0]; each level i<4 ends with ONE C2S up-block. So 4 coord-
growth steps (×2 each → ×16 total). Level 4 + output_layer NEVER run in upsample().
**Geometry depends only on `to_subdiv` (per up-block) + coords; features feed `to_subdiv`.**

## Blocks
**SparseConvNeXtBlock3d** (`sparse_unet_vae.py:265-294`), coords fixed:
`h = SparseConv3d(C,C,k3)(x); h = LayerNorm32(C, affine)(h); h = MLP(Linear C->4C, SiLU, Linear 4C->C)(h); return h + x`
(verify norm/conv/mlp order — subagent showed conv→norm→mlp; CHECK the real `_forward`.)

**SparseResBlockC2S3d** (`sparse_unet_vae.py:217-262`), grows coords ×2:
```
subdiv = to_subdiv(x)                       # SparseLinear C->8  (the OCTREE-CHILD logits)
h = SparseConv3d(C, C_out*8, k3)(SiLU(norm1(x)))    # norm1 affine, eps1e-6
sub_b = (subdiv.feats > 0)                  # [N,8] bool child-occupancy
h = SparseChannel2Spatial(2)(h, sub_b)      # COORDS ×2 + unpack 8 packed children -> [M, C_out]
x = SparseChannel2Spatial(2)(x, sub_b)      # skip path upsampled too
h = SparseConv3d(C_out, C_out, k3)(SiLU(norm2(h)))  # norm2 NON-affine, eps1e-6, conv2 zero-init
h = h + skip_connection(x)                  # skip = repeat_interleave feats to C_out
return h, subdiv
```

## The coord-growth crux: SparseChannel2Spatial(factor=2) (`spatial2channel.py:58-94`)
```
sub:[N,8] bool;  subidx = sub.nonzero()[:,-1] (child 0..7);  N_leaf = sub.sum(-1)
new_coords = x.coords.clone();  new_coords[:,1:] *= 2
new_coords = repeat_interleave(new_coords, N_leaf)          # [M,4]
for d in 0..2:  new_coords[:, 1+d] += (subidx // 2**d) % 2  # child bit -> +0/+1 per axis
idx = repeat_interleave(arange(N), N_leaf)                  # parent of each child
feats: x.feats[N, C_out*8].reshape(N*8, C_out); new_feats = feats[idx*8 + subidx]  # [M,C_out]
```
So child k (0..7): offset = (k&1, (k>>1)&1, (k>>2)&1) on (x,y,z); coord = parent*2 + offset.
**VERIFY** the feature reshape: is it `reshape(N*8, C_out)` (child-major within the C_out*8
block, i.e. channel c of child k at packed index k*C_out+c) or `reshape(N, C_out, 8)`? This
ordering must match `conv1`'s `C_out*8` output layout — CHECK `spatial2channel.py` exactly.
Also `_scale /= factor` per step (cache key); index cache keyed by `_scale`.

## Weights to export (`shape_dec_next_dc_f16c32_fp16.safetensors`)
from_latent [1024,32]; per level i, num_blocks[i]× ConvNeXt {norm,conv,mlp.0,mlp.2} + 1 C2S
{norm1,norm2,conv1 (C->C_out*8),conv2 (C_out->C_out),to_subdiv (8,C)}. output_layer [7,64]
(unused by upsample). SparseConv3d weight is flex_gemm native `[Co,Kw,Kh,Kd,Ci]` (spike) — the
spike kernel already validated; reuse it. Add to export_weights.py: `shape_dec` torch path.

## Port plan (correctness-first, like every prior rung)
1. **numpy ref** reproducing upsample() vs the real torch decoder.upsample on CPU (the sparse
   decoder runs CPU — sdpa/flex_gemm? flex_gemm needs CUDA; may need a CPU sparse-conv ref =
   the spike's CPU oracle `tools/sparse_spike/sparse_conv.cpp`/numpy). Validate hr_coords SET-EQUAL
   golden (382554). Coords are integer → exact match expected (subdiv is `>0`, deterministic).
2. **ggml graph**: integrate the spike submanifold conv (CUDA kernel + CPU mirror) into the
   M-harness as an op; build ConvNeXt + C2S(+subdiv) + 4-stage loop. The C2S coord-growth is
   host-side index arithmetic (like na2d indices) given the subdiv masks — but subdiv depends
   on the conv features, so it's data-dependent → can't fully precompute on host; need to read
   subdiv back per stage OR build a custom ggml op. Likely: run per-stage, read subdiv to host,
   compute next-stage coords + gather indices on host, feed next stage. (4 stages, sequential.)
3. Then **M3b is DONE** (shape-HR DiT, this session) — chain M3a→M3b for the HR shape_slat.

## Risks / unknowns to retire first
- flex_gemm is CUDA-only → the torch numpy ref for upsample may need CUDA (run decoder.upsample
  on GPU once to capture hr_coords + per-stage intermediate coords/subdiv as goldens). Capture
  per-stage subdiv masks + coords so the ggml port validates stage-by-stage.
- The exact ConvNeXt `_forward` op order + whether norm is pre/post conv.
- C2S feature packing order (child-major vs channel-major) — get it from source, not the map.
- Whether `from_latent` input is the DENORM lr_slat or re-normalized (pipeline passes denorm slat
  to upsample; CHECK if decoder re-normalizes — likely uses slat as-is).
