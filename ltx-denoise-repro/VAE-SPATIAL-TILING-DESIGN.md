# LTX-2.3 VAE: Comfy-style FEATHERED SPATIAL tiling (replace temporal tiling)

Design + env-gated code prototype. CPU/design analysis only — **nothing here was run on
the GPU or built** (shared 16 GB card is in use by the main thread). The prototype is
additive and gated behind `LTX_VAE_SPATIAL_TILES`; every existing decode path is byte-for-byte
untouched when that env is unset.

Files touched: `src/model/vae/ltx_vae.hpp` (new `decode_spatial_blend()` + a gated dispatch
in `_compute`). No other file changed.

---

## 0. The goal, restated

Our 1920×1088×97f decode currently uses **whole-spatial + TEMPORAL tiling** (either
`decode_temporal_blend`, `LTX_VAE_TEMPORAL_BLEND`, or the streaming
`decode_temporal_tiled_streaming`, `LTX_VAE_WHOLEFRAME`). Chopping the *time* axis into
2–4-latent-frame windows is what produces **motion ghosting / phasing** — even the feathered
temporal blend fabricates cross-tile temporal context at every window edge, and moving
background content has nothing to anchor to across the seam.

ComfyUI's `LTXVTiledVAEDecode` does the opposite and is seamless: it tiles **space**
(2×2, overlap 6 latent, feathered) and decodes **all frames at once per tile** — the time
axis is never chopped, so motion is continuous by construction. Space is a much more
forgiving axis to seam because the feather blend operates on static geometry, not on
temporally-evolving content.

This doc explains why *our* old spatial tiling seamed, and adds a Comfy-style feathered
spatial path that fits ≤ 11.5 GB at 1920×1088×97f on the 16 GB card.

---

## 1. Why the OLD spatial tiling seamed (root cause, in code)

The old spatial tiling is the generic `VAE::decode → tiled_compute → process_tiles_2d`
machinery (`src/model/vae/vae.hpp:197`, `src/core/ggml_extend.hpp:949`), driven by
`--vae-tiling --vae-relative-tile-size`. It already applies a **smootherstep feather**
(`sd_tensor_merge_2d`, `ggml_extend.hpp:896`), so "no feather at all" is NOT the root cause.
The seams come from **four** distinct, compounding defects:

### (a) The blend is ADDITIVE and NOT normalized — it relies on an analytic sum-to-1 that breaks at snapped edge tiles
`sd_tensor_merge_2d` writes `out += new * smootherstep(x_f) * smootherstep(y_f)` with **no
weight accumulation / divide** (`ggml_extend.hpp:938`). It is only correct because
smootherstep has the identity `S(x) + S(1-x) = 1`, so two complementary ramps in a regular
overlap band sum to exactly 1. **That identity is violated at the last row/column.**
`process_tiles_2d` *snaps* the final tile back to fit
(`x = small_width - tile_size_x; dx = original_x - x; x_skip = dx`,
`ggml_extend.hpp:1027-1035`), so the last tile's real overlap with its predecessor is
**irregular (larger)** than the standard `overlap_x`. But the ramp still divides by the
*standard* `overlap_x` (`ggml_extend.hpp:932-935`), so the two ramps no longer complement →
the summed weight in that band is ≠ 1 → a **bright/dark seam stripe at the final row and
column**. With no normalization there is nothing to correct it.

### (b) Edge-guard booleans misfire on the snapped tile
The decision to ramp an edge is `(overlap_x>0 && x>0)` on the left and
`x < (img_width - width)` on the right (`ggml_extend.hpp:932-933`). For the snapped last tile
`x == img_width - width` **exactly**, so the right-edge guard is false and that edge is
forced to weight 1 (no ramp) even though it overlaps a real neighbour → double-counted band.

### (c) Overlap too NARROW to cover the 32× decoder's receptive field → conv-border leak (the "grid-cross")
Each spatial tile is decoded as a standalone image. Every `CausalConv3d` **zero-pads the
tile's spatial borders** (there is no cross-tile latent context), so decoded pixels within
~the receptive field of a tile edge are **fabricated**, not real. The feather only hides
them if the overlap is **wider than the receptive field**, so the corrupted band sits in the
low-weight zone while the neighbour's real-context interior dominates. The old path used the
generic `target_overlap` (a small fraction, ~0.25 → a handful of latent px), which for LTX's
deep 32×-upsampling decoder is **too narrow** → fabricated borders leak into the blend =
the visible **grid-cross seam**. Comfy deliberately uses **overlap 6 latent (= 192 px)** to
cover it.

### (d) (Historical, now fixed, but the name stuck) the "screen-door MESH" was NOT tiling
The in-code comment at `ltx_vae.hpp:1309` blames "screen-door / grid-cross seams" on spatial
tiling, but `TILING-FIX.md` established that the **fixed 8-px screen-door mesh** was a
per-frame VAE defect: the missing LTX-0.9.5 up-block **pixel-shuffle residual** + an f16 head
conv (fixed in `31f99f5`, meshscore 52.7 → 12.1, `LTX_VAE_HEAD_F32`). That is orthogonal to
tiling and already resolved on this branch — but its "screen-door/grid-cross" name got
attached to the whole-spatial comment, conflating the two and helping drive the code to flee
to whole-spatial decode.

**Summary:** the old spatial path seams because of (a)+(b) a non-normalized blend that breaks
at snapped edge tiles, and (c) an overlap too small to cover the conv receptive field.
Comfy avoids all three: **normalized** weight-accumulate-then-divide, and a **wide (6 latent)**
overlap. The prototype below copies both.

---

## 2. The ShotStream "whole-frame" machinery — reusable?

`grep` findings: the LTX whole-frame lever is `LTX_VAE_WHOLEFRAME` (`ltx_vae.hpp:1308-1324`),
and it is **not** a spatial-tiling engine — it just forces the **temporal-streaming** path
(`decode_temporal_tiled_streaming`) to run whole-spatial with a bounded *frame* count. The
`SHOTSTREAM_*` hits are all in `src/model/diffusion/wan*.hpp` (the DiT KV-cache), unrelated to
the VAE. So there is **no existing spatial-feather engine to reuse** — the only spatial tiler
in the tree is the generic `process_tiles_2d`, which is exactly the one that seams (§1). The
prototype therefore adds a **new** spatial-blend routine rather than reusing either. It does
reuse the orthogonal per-tile levers: `GGML_CUDNN_CONV3D`, `LTX_VAE_CONV3D_WTILES/_HTILES`
(bounds the im2col *inside* each spatial tile, `ggml_extend.hpp:1362`), and
`LTX_VAE_DECODE_F16` (halves the activation floor, `ltx_vae.hpp:1172`).

---

## 3. Comfy's `LTXVTiledVAEDecode` algorithm (the reference we copy)

- Tile the latent **H×W** into `horizontal_tiles × vertical_tiles` (default **2×2**),
  each tile spanning the **full temporal dim** (all frames — no temporal chop).
- Overlap = **6 latent** on shared edges.
- Blend = **feathered weighted average**: build a per-pixel weight that ramps
  `torch.linspace(0,1,overlap)` in from each shared edge (1.0 in the interior and at true
  image borders), accumulate `output += tile * weight` **and** `weights += weight`, then
  `output /= weights + 1e-8`. The explicit normalization is what makes it robust to
  non-uniform tile placement (contrast §1a).

The prototype mirrors this exactly, with two deliberate deltas: (i) it uses **smootherstep**
instead of a raw linear ramp (matches the rest of this codebase, `sd_tensor_merge_2d`), and
(ii) it drives tile count/overlap from env so the VRAM knob is tunable.

---

## 4. Design of the new path: `decode_spatial_blend`

**Layout.** `NX × NY` tiles over the latent (W = dim0, H = dim1). Non-overlap core per tile
`coreW = ceil(Wl/NX)`, `coreH = ceil(Hl/NY)`. Each tile is the core **extended by `O` latent
px on every interior side** (clamped at the image border): latent range
`x∈[max(0,cx0-O), min(Wl,cx1+O))`, same for y. Interior tiles thus overlap their neighbour by
`~2O` latent (an `O`-px feather band on each side of the shared boundary).

**All frames per tile / no temporal chop.** Each tile holds **all** latent frames (we only
slice dims 0/1) and is decoded as **ONE chunk** via `build_temporal_tile_graph(z_tile, 0, 0)`
— the *same* graph builder `decode_temporal_blend` uses (`decode_tiled_chunk` →
`forward_tiled_frame`). `chunk_overlap=0` means no time-axis trim beyond the causal drop-first,
and because every spatial tile passes `chunk_idx=0` the drop-first fires identically for all of
them → every tile yields the identical output frame count `Tp = f*8-7` (= the full clip
length; spatial dims differ only). No `LTX_VAE_TBLEND` temporal windowing is involved.

> **Why this builder and not `build_graph()`/`vae.decode()`?** (fixed after the first GPU
> test.) The plain whole-decode conv path hits
> `im2col.cu:548 GGML_ASSERT(src1->type == GGML_TYPE_F32)` under the prod env
> `LTX_VAE_DECODE_F16=1 + GGML_CUDNN_CONV3D=1`: a stride/dilation-1 conv3d the cuDNN matcher
> rejects falls back to `ggml_conv_3d` (im2col), which asserts an **F32** activation while the
> stream is **F16**. `decode_temporal_blend`'s `forward_tiled_frame` conv routing does **not**
> hit this, so routing spatial tiles through the identical builder inherits the fix — no extra
> cast needed. (First run: path fired, peaked ~11.9 GB at 4×4/O6 as estimated, then crashed on
> this assert before writing the webm.)

**Feather / blend weights.** In pixel space (`scale` = decoder spatial upsample, 32 for LTX),
each tile occupies `[X0, X0+tw) × [Y0, Y0+th)` where `X0 = x0*scale`. The feather band widths
are `leftband=(cx0-x0)*scale`, `rightband=(x1-cx1)*scale` (0 at a true image edge), sim. top/
bottom. Weight is **separable**: `w(lx,ly) = smoother(wx(lx)) * smoother(wy(ly))` where
`wx = min(1, lx/leftband, (tw-1-lx)/rightband)` (a band is skipped if its width is 0, i.e. at
the image border → weight 1 there). We accumulate `out += w*tile` and `wsum += w` (weight is
frame/channel-independent, accumulated once), then **normalize** `out[px] /= wsum[px]`. This
is the Comfy weighted-average — robust to the snapped-edge irregularity that seams the old
path (§1a/b).

**Where the blend runs.** On the CPU host, after each tile's GPU decode is copied back and its
compute buffer freed (`free_cache_ctx_and_buffer()` between tiles). Peak **GPU** VRAM =
one resident 1385 MB param set + one tile's decode compute buffer; peak **CPU** RAM = the full
output buffer (~2.4 GB f32 at 1920×1088×97×3) + one tile + an 8 MB weight plane.

**Param residency (fixed after the 2nd GPU test).** The tiles run
`GGMLRunner::compute(..., free_immediately=true)`, whose `free_compute_buffer()` calls
`restore_all_params()` when `!keep_params_resident_` — so **each tile would re-offload the full
1385 MB param set** (16 tiles × 1385 MB → cudaMalloc OOM; observed in the 2nd test).
`decode_temporal_blend` sidesteps this because its prod recipe runs with `--vae-tiling` **enabled**
(1×1), so the outer `VAE::decode` calls `set_keep_params_resident(true)`
(`vae.hpp:192`, gated on `tiling_params.enabled && LONGCAT_VAE_KEEP_RESIDENT`). Our path runs with
tiling **disabled** (§4.1), so it manages residency itself: `decode_spatial_blend` saves the
prior `keep_params_resident_`, `set_keep_params_resident(true)` before the loop (offload happens
**once**), and restores the prior state at every exit (`false` → frees the resident params now;
`true` → leaves them for an outer keep-resident recipe). Peak GPU is thus **one** param set +
one tile — no accumulation.

**Returns** the pre-`[0,1]`-scale decoder output (same contract as `decode_temporal_blend`);
the outer `VAE::decode` applies `scale_tensor_to_0_1` to the merged result.

### 4.1 CRITICAL routing note (read before testing)
`decode_spatial_blend` lives **inside `LTXVideoVAE::_compute`**. But `VAE::decode`
(`vae.hpp:197`) only calls `_compute` **directly when `vae_tiling_params.enabled == false`**;
when tiling is enabled it routes through the *outer* `tiled_compute → process_tiles_2d` (the
old spatial path). So to hit the new path you must **disable `--vae-tiling`** (and
`--temporal-tiling`). With tiling disabled, `_compute` receives the whole latent and does its
own spatial split — there is **no** 47 GB whole-graph risk because the split happens *before*
any graph is built.

---

## 5. VRAM model (tile count → peak) — pick N×M for ≤ 11.5 GB

Anchor (measured, from memory/handoffs): whole-spatial × **TBF=3** latent frames ≈ **9819 MB**
at 1920×1088×97f. Latent grid = **60×34** (= 1920/32 × 1088/32) = 2040 latent-px;
**all frames = 13** (= (97-1)/8 + 1). The per-graph activation floor scales ~
`tile_latent_px × frames_in_graph`. Reference unit = 2040×3 = 6120 → 9819 MB, i.e.
**≈ 1.60 MB / (latent-px·frame)** *treated as fully proportional* — this is a **conservative
upper bound** (it folds the fixed ~1.4 GB params + workspace floor into the slope, so it
over-estimates sub-reference tiles; real peak will be **lower**).

Interior-tile latent area = `(coreW+2O) × (coreH+2O)` (clamped ≤ 60×34), × 13 frames:

| N×N | O (lat) | tile latent px | ×13 frames | est. peak (upper-bound) | note |
|-----|---------|----------------|------------|-------------------------|------|
| 2×2 | 6 | 42×29 = 1218 | 15834 | ~25.4 GB | too big |
| 2×2 | 4 | 38×25 = 950 | 12350 | ~19.8 GB | too big |
| 3×3 | 6 | 32×24 = 768 | 9984 | ~16.0 GB | over |
| 3×3 | 4 | 28×20 = 560 | 7280 | **~11.7 GB** | marginal |
| **4×4** | **6** | 27×21 = 567 | 7371 | **~11.8 GB** | **start here** (real < est.) |
| **4×4** | **4** | 23×17 = 391 | 5083 | **~8.2 GB** | safe |
| 5×5 | 6 | 24×19 = 456 | 5928 | ~9.5 GB | safe |
| 5×5 | 4 | 20×15 = 300 | 3900 | ~6.3 GB | very safe |

Because all-frames = 13 is 4.3× the TBF=3 reference, spatial area must drop hard to
compensate — hence **≥ 4×4** is the practical starting band. Recommended first shot:
**`4×4`, overlap `6`** (Comfy-parity feather; est. 11.8 GB upper-bound → very likely under
11.5 in reality). If nvidia-smi shows a spike, step to **`4×4` overlap `4`** (~8 GB) or add
**`LTX_VAE_DECODE_F16=1`** (~0.6× the activation floor) and/or
**`LTX_VAE_CONV3D_WTILES=2`** to bound each tile's im2col — both orthogonal, no extra spatial
tiles needed. Non-square (e.g. `4x3`) is supported since W (60) > H (34).

**Rectangular tip:** the latent is wider than tall (60×34), so `NX ≥ NY` (e.g. `5x3` or
`4x3`) balances tile aspect and can beat square counts for a given peak.

---

## 6. The prototype (already applied, additive + gated)

### New env flags
- `LTX_VAE_SPATIAL_TILES=NxM` — e.g. `4x4` (or a single `N` → `NxN`). **Enables the path.**
- `LTX_VAE_SPATIAL_OVERLAP=O` — latent-px overlap on shared edges (default **6**, Comfy-parity).

### Dispatch (`LTXVideoVAE::_compute`, highest priority, before `LTX_VAE_TEMPORAL_BLEND`)
```cpp
if (decode_graph && input.dim() >= 4 && input.shape()[2] >= 1 &&
    getenv("LTX_VAE_SPATIAL_TILES") != nullptr) {
    const std::string spec = getenv("LTX_VAE_SPATIAL_TILES");
    int NX = std::max(1, atoi(spec.c_str()));       // atoi stops at 'x'
    int NY = NX;
    const size_t xpos = spec.find_first_of("xX");
    if (xpos != std::string::npos) NY = std::max(1, atoi(spec.c_str() + xpos + 1));
    const int O = wholeframe_env_int("LTX_VAE_SPATIAL_OVERLAP", 6);
    return decode_spatial_blend(n_threads, input, expected_dim, NX, NY, O);
}
```

### `decode_spatial_blend(...)`
See `ltx_vae.hpp` (inserted just before `build_latent_statistics_graph`). Behaviour: overlap-
extended NX×NY latent tiles → per-tile `build_temporal_tile_graph(z_tile,0,0)` whole-decode
(all frames, one chunk — same builder/conv-routing as `decode_temporal_blend`, F16+cuDNN-safe)
→ normalized separable smootherstep feather → weighted-average merge on the host. Mirrors the
proven `decode_temporal_blend` for the compute/free/`restore_trailing_singleton_dims` plumbing
and slicing idioms (`sd::ops::slice(input, dim, start, end)`), so it should compile against the
surrounding patterns.

### Things the main thread must VERIFY at build/run
1. **Compiles** — the routine only uses idioms already present in `decode_temporal_blend`
   (`sd::ops::slice`, `GGMLRunner::compute<float>`, `restore_trailing_singleton_dims`,
   `sd::Tensor` ctor/`fill_`/`data`/`numel`/`shape`/`empty`, `free_cache_ctx_and_buffer`,
   `cache_tensor_map`). Not yet compiled here (no build during GPU use).
2. **`scale == 32`** — computed at runtime as `tw / (x1-x0)` from the first decoded tile; if
   the LTX decoder's spatial upsample isn't a clean 32× (it should be: 4× unpatch × 2×2×2
   depth-to-space), the integer division would truncate. Log it / assert on first tile.
3. **`build_temporal_tile_graph(z_tile, 0, 0)` decodes all frames** for a sliced-in-space tile
   as a single chunk (it does — same builder `decode_temporal_blend` uses; slice only touches
   dims 0/1, so the full temporal range is intact). Verified indirectly by the retest.
4. **Routing** — must run with `--vae-tiling` **disabled** (§4.1), else the outer
   `tiled_compute` double-tiles.

---

## 7. Test recipe (for the main thread, on GPU)

Route through `_compute` (tiling **off**) and enable the spatial path. Start conservative on
tiles, measure VRAM, then relax:

```sh
# 4×4 feathered spatial tiles, overlap 6 latent, all frames per tile, NO temporal tiling.
# NOTE: no --vae-tiling / no --temporal-tiling (so VAE::decode calls _compute directly).
export LTX_VAE_SPATIAL_TILES=4x4
export LTX_VAE_SPATIAL_OVERLAP=6
export GGML_CUDNN_CONV3D=1        # im2col-free conv3d (keeps the per-tile floor down)
export LTX_VAE_HEAD_F32=1         # keep the screen-door-mesh fix on (orthogonal, cheap)
# optional VRAM relief if 4x4/O6 spikes over 11.5 GB:
# export LTX_VAE_DECODE_F16=1     # ~0.6x activation floor
# export LTX_VAE_CONV3D_WTILES=2  # bound each tile's im2col
# and/or step overlap down: LTX_VAE_SPATIAL_OVERLAP=4  (or tiles up: 4x4 -> 5x4)
# make sure the OLD/other paths are OFF:
unset LTX_VAE_TEMPORAL_BLEND LTX_VAE_WHOLEFRAME
# ... run the usual 1920x1088x97f t2v/i2v decode, WITHOUT --vae-tiling/--temporal-tiling ...
```

**What to look for**
- **Log line** `LTX VAE spatial-blend decode: tiles=4x4 overlap=6 ...` — confirms the path
  fired (if absent, tiling wasn't disabled → the old `process_tiles_2d` ran instead).
- **VRAM** via `nvidia-smi dmon` / peak: target ≤ 11.5 GB. If over, apply the relief knobs
  above. If comfortably under, you can *lower* the tile count (e.g. `3x3 O4`) for speed.
- **Seams** in a STILL frame: no grid-cross / brightness stripes at the 4×4 tile boundaries
  (the normalized wide-overlap feather should be clean; if a faint seam remains, raise
  `LTX_VAE_SPATIAL_OVERLAP` to 8).
- **Motion** in the eye-test clip (deliver via `:8077`): the whole point — background crowds /
  camera pans should no longer ghost/phase (there is no temporal seam by construction). A/B
  against the current `LTX_VAE_TEMPORAL_BLEND` clip.
- **Perf**: NX·NY tile decodes run sequentially → wall ≈ NX·NY × (small-tile decode). More
  tiles = more but smaller graphs; watch total decode time vs the temporal path.

---

## 8. Open items / follow-ups
- The host-side blend loop is single-threaded O(tiles × tile_px × frames×ch) (~hundreds of M
  MAdds) — a couple seconds; parallelize with the frame/channel loop if it shows up.
- Param residency across the tile loop is now handled internally (§4, offload once); no
  `LONGCAT_VAE_KEEP_RESIDENT` needed for this path. If per-tile graph-BUILD overhead (not param
  offload) shows up at high tile counts, that's the next thing to profile.
- Consider a `LTX_VAE_SPATIAL_FRAMES` cap as a safety valve to also bound frames per tile if a
  future resolution makes even 1/16-area × 13-frames too big — but that reintroduces a temporal
  seam, so leave it off by default (defeats the purpose).
```
