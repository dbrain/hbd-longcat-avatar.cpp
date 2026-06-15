# V6 dense-bake recipe (the clean, fingered, "hero" Miku) — DON'T LOSE THIS

The **dense v6 bake is the cleanest asset we produce** — full detail, fingers intact, no teal seams at
4096 (owner verdict 2026-06-14: "the cleanest model I've seen"). It is the deliverable for **static /
display / hero** use, and the **colour + normal source-of-truth** for every retopo/LOD downstream. This
file records the exact recipe so it survives.

Recipe is also self-documented in `tex_reproject.cpp` (header + `main()` defaults); this is the prose copy.

## Two steps (both GPU, in `tools/m1_ref/cpp_port/`)

### 1. Generate the bake inputs from pixal3d (the dump)
```bash
./build.sh pixal3d cuda
PIXAL3D_FORCE_UVATLAS=1 PIXAL3D_DUMP_BAKE=1 \
  ./pixal3d --model <gguf-dir> --image <input.png> --out miku.glb
```
Dumps the ALIGNED bake inputs into the cwd:
- `dump_mesh_*.bin` + `dump_bake.txt` — the QEM mesh = bake target / UV atlas
- `dump_dense_*.bin` + `dump_dense.txt` — the smooth DENSE outer-shell mesh = **colour + normal source-of-truth**
- `dump_pbr_{f,c}.bin` — the per-voxel 6-channel PBR volume
(`pixal3d.cpp:154` + `pixal3d_chain.hpp:393` gate these on `PIXAL3D_DUMP_BAKE`.)

### 2. Bake the v6 texture
```bash
./build.sh tex_reproject cuda
./tex_reproject 4096 native_v6.glb        # v6 parity defaults are BAKED IN
```
Reads `dump_dense_*` (colour), `dump_pbr_*` (PBR), `dump_mesh_*` (target). Writes `native_v6.glb`
(~86 MB uncompressed-ish at 4096).

## The v6 flag-set (FINDINGS-A) — DEFAULT in `tex_reproject.cpp main()`
The 5 diagnosed fixes that reach Python parity, now defaults (any env var overrides one):
| flag | value | why |
|---|---|---|
| `RP_OFF` | `1` | reproject/normal-offset OFF (FINDINGS-A: offset specks the aligned mesh) |
| `TEX_FBR` | `0` | **no dark fallback** — kills the dark specks |
| `TEX_RASTER_SS` | `2` | 2× supersampled raster (AA edges, matches Python ssaa2) |
| `TEX_KEEP_ATLAS_SIZE` | `1` | full-res parity, **no atlas downsample** (set only if `TEX_FINAL_SIZE` unset) |
| topo-normals + full bg-fill + TELEA inpaint + pyref-xatlas + native-cumesh-500k | (baked) | the rest of the parity set |

Other env knobs: `ATL_CONE` (55), `ATL_PAD` (4), `RP_NCELL`, `RP_MAXRING`, `RP_FRONTDOT`,
`TEX_INPAINT_ITERS`, `VCOLOR_FB`.

## Variants
- **Game LOD** (cheaper, smaller texture): `CUMESH_TARGET=70000 TEX_FINAL_SIZE=1024 ./tex_reproject 4096 lod.glb`
- **Compressed shippable**: pack with the in-process meshopt+KTX2 packer (`--pack hero`) — see HANDOFF-B / `project_pixal3d_inprocess_glb_pack`.

## Where v6 lives
`native_v6.glb` / `native_v6_inproc.glb` (packed) are in this dir and on the compare page
(`http://10.0.0.208:8011/compare.html`, "Game-asset downscale" tab B, and the "Retopo" tab C as the
quality ceiling). See `project_pixal3d_inprocess_glb_pack`, `project_pixal3d_retopo_manifoldplus`.
