# LTX-2.3 nvfp4 → ComfyUI parity — RESULTS

**Goal:** match ComfyUI's dev-fp8 two-stage quality in our engine, at ≥ our current speed. **Achieved.**

## The winning config (comfy-grade quality, faster than our current)
```
RES=speed MAXV=9 TBF=4 VWT=4 VHT=2 bash run_parity_nvfp4.sh
# = nvfp4-CLEAN-dev050, base 640x352 -> x2 -> 1280x704, 121f@24fps,
#   8-step base + 3-step refine (comfy sigmas), cfg=1,
#   VAE: LTX_VAE_HEAD_F32=1 + LTX_VAE_TEMPORAL_BLEND=1 (TBF=4,O=2) + CONV3D_WTILES=4 + DECODE_F16=1
```
**135s** · 1280×704 · both artifacts fixed · comfy-grade. (vs our old 170s, vs comfy ~400s.)

## What was actually wrong (the whole gap = VAE decode, not the model)
Our nvfp4 DiT was fine. Two VAE-decode defects made it look worse than comfy:
1. **Striping** = screen-door mesh from the f16 head conv magnified by the pixel-shuffle → **`LTX_VAE_HEAD_F32=1`**.
2. **Ghosting** (bg people phasing) = hard-cut temporal tiling with no feather (cache breaks under offload) →
   **`LTX_VAE_TEMPORAL_BLEND=1`** (quintic-smootherstep feathered decode). Cross-model win: same 22B VAE is
   used by prod relip/dub, so this upgrades that pipeline too. Fix shipped as master commit `31f99f5`.

## Speed ladder (all nvfp4, both fixes, 24fps/121f)
| res | wall | notes |
|---|---|---|
| **1280×704** | **135s** | ★ recommended — beats our 170s current, comfy-grade |
| 1280×704 (WT16) | 184s | over-tiled decode (slow); VWT=4 fixes it |
| 1664×960 | 315s | +pixels, if a shot needs it |
| 1920×1088 | 449s | comfy's exact res; overkill for most |

**Key lever:** at a given res, `VWT` (spatial decode tiles) is the speed knob — fewer = faster (bit-exact),
only raise it if the decode OOMs. `TBF` (temporal tile) trades memory vs blend granularity. `RES` is the
pixel-budget/speed lever (24fps pays for it).

## Speed-vs-quality menu
- **Max speed** (~93–120s): old hard-cut decode + `HEAD_F32` only (fixes striping, some ghosting remains).
- **Recommended** (135s): full blend, both fixed, 1280×704 — the sweet spot.
- **Max quality** (315–449s): 1664–1920 for hero shots.

dev-fp8 never needed. Compare: http://10.0.0.208:8077/ltx_denoise/compare.html
