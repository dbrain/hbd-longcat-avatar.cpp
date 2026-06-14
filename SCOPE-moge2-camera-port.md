# SCOPE — MoGe-2 camera estimator → ggml (replace estimate_camera.py)

Scoping pass 2026-06-14 (no code yet). Goal: nuke the last heavy Python front-end, `estimate_camera.py`,
which runs MoGe-2 (`Ruicheng/moge-2-vitl`, PyTorch) to predict per-image FOV+distance for arbitrary-photo
uploads. Bypass today = fixed `--fov`/`--cam` (works for training-like framing). This doc = what a real
native port costs.

## What we ACTUALLY need (the big simplification)
`estimate_camera.py` consumes ONLY `out["intrinsics"]` → `fx` → `camera_angle_x = 2·atan(W/(2fx))` → then
its OWN pure-host distance math (already mirrored in the script). So we do **not** need MoGe's depth, point
map export, normal, or metric scale as outputs — only the camera intrinsics.

BUT: MoGe-2 has **no dedicated camera/FOV head**. Intrinsics are recovered from the predicted dense POINT
MAP via `recover_focal_shift(points, mask)` (moge/utils/geometry_torch.py). So "intrinsics" still requires
the point-map pipeline. The FOV-only subgraph is:

    image → DINOv2 ViT-L/14 encoder → (UV concat) → shared neck → points_head + mask_head
          → recover_focal_shift(points, mask) → focal → fx → camera_angle_x

`normal_head` and `scale_head` are NOT needed for FOV → SKIP them (metric_scale only rescales depth, not
the angle). Run at MIN `num_tokens` (~1200): FOV is a global quantity and the solver downsamples the point
map to 64×64 anyway, so fine detail/high token counts are wasted here.

## The chain, piece by piece (moge/model/v2.py forward())
1. **Input prep**: bilinear-resize image to `token_rows*14 × token_cols*14` (antialias), ImageNet
   normalize (mean .485/.456/.406, std .229/.224/.225). token grid from num_tokens + aspect ratio.
2. **DINOv2 ViT-L/14 encoder** (the long pole) — standard DINOv2: patch_embed conv k14 s14, cls token
   (+ register tokens — CONFIRM count from checkpoint), interpolated learned pos-embed, 24 blocks
   (LN → MHSA → LayerScale → LN → MLP(GELU) → LayerScale), MLP ffn (not SwiGLU for vitl), NO RoPE.
   `get_intermediate_layers(n=intermediate_layers)` returns features from N specific blocks (CONFIRM
   indices) — these feed the neck pyramid. dim 1024.
3. **output_projections**: one 1×1 conv (1024→dim_out) per intermediate layer. trivial.
4. **UV concat**: `normalized_view_plane_uv` appended at 5 pyramid levels (encodes aspect ratio). cheap.
5. **shared neck**: Resampler / ConvStack / ResidualConvBlock pyramid (Conv2d + GroupNorm + ReLU +
   ConvTranspose/pixel-shuffle upsamplers). Standard conv FPN.
6. **points_head + mask_head**: ConvStack DPT-style heads → 3-ch point map + 1-ch mask. Bilinear
   interpolate to image res; `_remap_points` (elementwise); mask sigmoid.
7. **recover_focal_shift**: downsample point map+mask to 64×64, per-image least-squares for (focal, shift)
   over ~4096 masked points (currently numpy on CPU). focal → fx,fy → intrinsics.

## Port plan & effort (≈ DINOv3-encoder + DPT-head lap already done here)
| piece | effort | notes |
|---|---|---|
| DINOv2 ViT-L/14 encoder | ~55% | reuse ggml ViT scaffolding from `dinov3_graph.hpp`; ADAPT: LayerScale, learned+interpolated pos-embed, register tokens, no-RoPE, patch14, get_intermediate_layers |
| neck + points_head + mask_head | ~25% | ggml conv2d / `ggml_group_norm` / relu / transpose-conv / bilinear; all standard ops |
| UV concat + input resize/normalize | ~5% | cheap host/ggml |
| recover_focal_shift + camera math | ~5% | CPU; 64×64 LSQ; camera math already in estimate_camera.py |
| GGUF pack + stage validation | ~10% | pack moge-2-vitl weights to GGUF; match fp32 oracle stage-by-stage |

All standard ggml ops; builds directly on the existing pixal3d ggml infra (already runs ViT/DiT/conv
graphs from GGUF). The encoder is the bulk — call it a focused multi-session lap, the encoder being the
long pole, the conv heads + solver quick.

## Reuse already in-tree
- `tools/m1_ref/cpp_port/dinov3_graph.hpp` + `dinov3_test` — a working ggml ViT encoder (DINOv3). DINOv2
  differs (LayerScale, learned pos-embed interp, registers, no RoPE) but the patch-embed/attention/MLP/LN
  scaffolding + GGUF load path transfer. `DINOV3-ENCODER-SPEC.md` documents that work.
- GGUF tooling: `pack_gguf.cpp`, `gguf_reader.hpp`, `PIXAL3D_GGUF_DIR` (the 7-model A2 pipeline).
- `image_io.hpp` (stb + Lanczos resize) for input prep.

## Validation plan
Dump MoGe-2 intermediates from Python on 3–5 varied test photos (encoder feats per intermediate layer,
neck out, points, mask, recovered focal, final camera_angle_x). Match C++ stage-by-stage to the **fp32
oracle** (NOT tf32/bf16), per the project rule. Final acceptance: `camera_angle_x` within ~0.5° of Python
— geometry is robust to small FOV error, so this is a tolerance match, not bit-exact.

## Alternatives (decreasing nativeness)
1. **Full ggml port** (above) — true native, arbitrary-photo, one binary. Recommended if "upload any
   photo" is a real product requirement.
2. **ONNX route** — MoGe-2 has `onnx_compatible_mode`; export ONNX, run via onnxruntime C++. No port, but
   adds a heavy non-ggml runtime dep (like a host service, but ORT not ggml). Middle ground.
3. **Bypass (zero cost)** — keep fixed `--fov`/`--cam`. Fine for controlled/training-like framing; only
   "drop in literally any photo" needs the estimate. This is the current state and is not a blocker for
   "model looks good".

## Encoder config — PINNED (from moge/model/dinov2/hub/backbones.py)
DINOv2 `vit_large`, patch14, embed_dim 1024, depth 24, **ffn=mlp** (NOT swiglu), LayerScale (DINOv2),
learned interpolated pos-embed, NO RoPE. The ONLY open backbone bit: **register tokens 0 vs 4** — MoGe-2's
`backbone` config string picks `dinov2_vitl14` (0 regs) or `dinov2_vitl14_reg` (4 regs). Affects token
layout + pos-embed slicing only.

## Open items to confirm at build time (need the checkpoint, not downloaded yet)
- register tokens 0 vs 4 (above) and exact `intermediate_layers` indices (which blocks feed the neck).
- `dim_out`, neck channel widths, points/mask head ConvStack configs — read from the moge-2-vitl config.
Weights download from HF `Ruicheng/moge-2-vitl` on first `estimate_camera.py` run; grab config.json then.
