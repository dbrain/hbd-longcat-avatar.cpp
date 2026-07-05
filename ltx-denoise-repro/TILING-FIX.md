# LTX VAE decode: striping + ghosting root-cause + fix

Investigation of the two background artifacts (horizontal **striping/banding** and
background-people **phasing/ghosting**) that our LTX-2.3 renders show but ComfyUI's
same recipe does not. CPU/design only — nothing here was run on the GPU or built.

## TL;DR

- The artifacts are **two independent problems**, and **both are already fixed on
  `master`** in commit `31f99f5` — the `ltx-denoise-workflow` **worktree binary predates
  that commit**, so the running `sd-cli` is missing the fixes.
- **Striping/banding = VAE "screen-door mesh"** — the worktree binary omits the LTX-0.9.5
  up-block **pixel-shuffle upsample residual** (and the F32 head-conv), so the decoder
  emits a fixed grid that the 4× unpatchify magnifies. This is a per-frame **spatial**
  VAE-decode defect, *not* a tiling seam.
- **Ghosting/phasing = temporal tiling seam.** Our decode runs
  `decode_temporal_tiled_streaming` in **4-latent-frame windows, 1-frame overlap**, and
  stitches by **DROP-overlap + hard `ggml_concat` — NO feather/blend**. It relies on a
  cross-chunk causal `feat_map` cache that is unreliable under `--offload-to-cpu` (which
  prod uses), so each of the 8 chunk boundaries drifts → background "phases and blurs."
- **ComfyUI feathers every tile boundary** (linear `linspace` crossfade, spatial *and*
  temporal) and uses **much larger temporal tiles** (16 latent frames vs our 4).
- **Recommended fix (code, needs rebuild): cherry-pick `31f99f5` onto the worktree** (it
  applies cleanly) and run with `LTX_VAE_TEMPORAL_BLEND=1`. That commit adds the mesh fix
  *and* an offload-safe, smootherstep-**feathered** temporal-blend decode that mirrors
  Comfy. **No pure-flag fix is possible on the current binary** — the levers don't exist
  in it yet.

---

## 1. What actually runs in prod (confirmed from render logs)

Prod flags (`run_ablation.sh` / `../longcat-avatar.cpp/run_ltx_t2v.sh`):
```
--vae-tiling --vae-relative-tile-size 1x1 --temporal-tiling \
--extra-tiling-args temporal_tile_frames=4,temporal_tile_overlap=1  --offload-to-cpu
```

From `_ablation_out/s1_t2v_r0_baseline/log` (and the i2v logs):
```
ltx_vae.hpp - Using streaming temporal tiling: temporal_tile_frames=4, temporal_tile_overlap=1, total latent frames=25, resulting in 8 tiles
ltx_vae.hpp - LTX VAE temporal tile 1/8: latent frames [0, 4), overlap=1
ltx_vae.hpp - LTX VAE temporal tile 2/8: latent frames [3, 7), overlap=1
...
ltx_vae.hpp - LTX VAE temporal tile 8/8: latent frames [21, 25), overlap=0
vae.hpp:205 - VAE Tile size: 40x22          # single spatial tile = whole 1280x704 frame
offload params ...                          # --offload-to-cpu active
```

So the real decode is:
- **Spatial: ONE tile** (`--vae-relative-tile-size 1x1` → latent 40×22 = full frame) →
  **no spatial seams**. So spatial tiling is *not* the striping source.
- **Temporal: `decode_temporal_tiled_streaming`** in 4-frame windows, stride 3, over 8
  chunks, under CPU offload.

### 1a. How temporal tiling chunks + stitches — HARD-CUT, no blend

`src/model/vae/ltx_vae.hpp` (worktree), `decode_temporal_tiled_streaming` (~L1366) →
`build_temporal_tile_graph` → `VideoVAE::decode_tiled_chunk` → `forward_tiled_frame`:

- Windows: `for start in 0.. step (frames-overlap=3)`, chunk `[start, start+4)`.
- Per chunk decoded through the full decoder; causal state carried across chunks via a
  `feat_map` **cache** (`temporal_feat_cache_name`, cached between graphs).
- After decode: `if chunk_overlap>0: out_chunk = slice(out_chunk, 2, 0, ne[2]-overlap*8)`
  — **drops** the trailing overlap output frames — then
  `output = ggml_concat(output, out_chunk, 2)` — **hard concatenation**.
- **There is NO weighting, feather, crossfade, or normalization across the join.** It is a
  drop-overlap + butt-splice. Continuity depends entirely on the causal `feat_map` cache
  being correct across chunk boundaries.

Why that produces ghosting in *this* config:
- **Offload fragility.** The main-branch author's own comment (in the fixed code) says the
  streaming cache path has an "offload feat_map-cache issue" that the new independent-tile
  path "sidesteps entirely." Prod runs `--offload-to-cpu`, so the carried causal context
  is the exact thing that breaks → per-chunk context loss.
- **Tiny 4-frame windows.** 25 latent frames → **8 chunk boundaries**. Every boundary is a
  fresh chance for a brightness/detail step. Low-structure regions (out-of-focus background
  crowd) have nothing to anchor to, so they "phase and blur."
- **Hard splice.** With no feather, any residual per-chunk DC/exposure difference lands as
  a visible temporal seam (also a plausible contributor to horizontal banding at the split
  frames).

### 1b. Striping/banding — separate VAE-decode defect (missing upsample residual)

`diff` of worktree `ltx_vae.hpp` vs `master` shows the worktree passes `false` for the
up-block residual and uses a plain `conv_out`:
```
worktree:  Upsample(..., /*up_residual=*/false)   conv_out = CausalConv3d(hidden, 3*p*p, 3)
master:    Upsample(..., up_residual)              conv_out = CausalConv3d(..., head_f32)   + LTX_VAE_HEAD_F32
```
master's comment on the added residual:
> "LTX 0.9.5+ VAE decoder up-blocks carry a parameter-free pixel-shuffle RESIDUAL skip
> (upsample_residual=(True,True,True)). … Omitting it leaves the conv-only branch, whose
> per-sub-pixel DC offset forms a 2px checkerboard at quarter-res that the final 4x unpatch
> magnifies into the fixed 8px 'mesh' (meshscore 52.7 → 12.1)."

This is a **fixed spatial grid** baked into every decoded frame — exactly a
striping/banding signature — and it is absent from ComfyUI because Comfy runs the real
diffusers `AutoencoderKLLTXVideo`, which *has* the residual. **This is unrelated to
tiling.**

---

## 2. What ComfyUI does differently (the golden reference)

`ComfyUI-LTXVideo/tiled_vae_decode.py`:

**`LTXVTiledVAEDecode` (spatial):** tiles H×V, each tile decoded over the **full temporal
dim**, and overlaps are **feathered**: `torch.linspace(0,1,overlap)` ramps on left/right
and top/bottom edges, accumulated into a weight buffer and divided out
(`output /= weights + 1e-8`). → weighted-average blend, no seam.

**`LTXVSpatioTemporalTiledVAEDecode` (temporal):**
- Default **`temporal_tile_length=16` latent frames**, `temporal_overlap=1` (we use 4/1).
- Each temporal chunk is **re-decoded fresh** (re-includes overlap latents; **no carried
  causal cache**).
- The overlap band is **crossfaded**:
  ```
  frame_weights = linspace(0, 1, overlap_frames+2)[1:-1]
  output[overlap] = (1 - w)*existing + w*new_tile        # linear temporal feather
  output[after_overlap] = new_tile[overlap:]             # hard only outside the band
  ```
  i.e. the outgoing tile's fabricated-context edge fades out as the incoming tile's
  real-context interior fades in.

**Delta vs us:** (a) Comfy **feathers** temporal (and spatial) boundaries; we **hard-cut**.
(b) Comfy uses **16-frame** temporal tiles (4× fewer boundaries). (c) Comfy re-decodes
overlaps **independently** instead of trusting a fragile carried causal cache. (d) Comfy's
VAE has the **upsample residual** (no mesh).

---

## 3. Fix options (ranked)

### Option A — flags only, current worktree binary: NOT SUFFICIENT (be honest)

The current binary exposes **only** `temporal_tile_frames` / `temporal_tile_overlap`. The
feathering/whole-spatial/mesh levers (`LTX_VAE_TEMPORAL_BLEND`, `LTX_VAE_WHOLEFRAME`,
`LTX_VAE_HEAD_F32`, upsample residual) **do not exist in this binary** — setting those envs
is a no-op. So with the current binary you **cannot** add feathering and **cannot** fix the
mesh. The only thing flags can do is reduce the *number* of temporal seams:

```
# A1 — fewer, larger temporal tiles (ghosting only, marginal; does NOT fix striping):
#   replace the --extra-tiling-args value:
--extra-tiling-args temporal_tile_frames=8,temporal_tile_overlap=2
```
- Effect: 25 latent frames → ~4 tiles instead of 8 → half the boundaries, more causal
  context per chunk. Still hard-cut, still offload-fragile, still meshed.
- **VRAM:** each graph now decodes up to **8** latent frames whole-spatial (≈64 output
  frames at 1280×704) vs 4 today — roughly **~2× the per-graph full-res activation floor**.
  With the prod `--max-vram 11` target on a 16 GB card this is plausibly OK but must be
  watched; `temporal_tile_frames=16` (Comfy parity) will almost certainly **OOM** — that
  4-frame window is exactly why tiles were made this small.
- **Verdict:** stop-gap at best; will not reach Comfy quality. Do not ship as "the fix."

### Option B — code (cherry-pick the existing master fix), rebuild: RECOMMENDED

The complete fix already exists and is proven on `master` as **commit `31f99f5`**
("ltxv 0.9.x VAE: fix screen-door mesh (upsample residual) + conv3d W/H-tiling +
length-independent temporal-blend decode"). It brings:
- **upsample residual + `LTX_VAE_HEAD_F32`** → kills the screen-door mesh (**striping**),
  meshscore 52.7 → 12.1. The residual defaults **ON** for this timestep-conditioned 0.9.5
  VAE, so the mesh fix is automatic once built.
- **`decode_temporal_blend` (`LTX_VAE_TEMPORAL_BLEND=1`)** → decodes **independent**
  temporal tiles (no carried causal cache → offload-safe) and stitches with an
  **overlap + quintic smootherstep feather**, normalized to a proper weighted average.
  This is the Comfy approach (feathered temporal blend), bounding the activation floor to a
  T-frame tile — so **VRAM ≈ current** (T defaults to 4). Fixes **ghosting/phasing**.

It **cherry-picks cleanly** onto the worktree (verified `git cherry-pick --no-commit
31f99f5` → auto-merges `src/core/ggml_extend.hpp`, `src/model/vae/ltx_vae.hpp`,
`src/stable-diffusion.cpp`, **no conflicts**; reverted after test).

**Apply (worktree, build deferred):**
```
cd /home/dbrain/dev/longcat-avatar-ltxdenoise
git cherry-pick 31f99f5        # clean; or `git rebase master` (3 commits) as an alternative
```

**Rebuild (deferred — do NOT run while the GPU is in use):** incremental, the existing
`build-cudnn/` is already configured (Unix Makefiles, Release, CUDA arch 120/cuDNN):
```
docker run --rm --gpus '"device=1"' \
  -v /home/dbrain/dev/longcat-avatar-ltxdenoise:/src -w /src \
  longcat-avatar-dev:builder-cudnn-ff  cmake --build build-cudnn -j
```

**Run after rebuild — swap the decode flags to the feathered path:**
```
# replace, in run_ablation.sh / run_ltx_t2v.sh, the current
#   --vae-tiling --vae-relative-tile-size 1x1 --temporal-tiling \
#   --extra-tiling-args temporal_tile_frames=4,temporal_tile_overlap=1
# with the whole-spatial + feathered temporal-blend decode, driven by env:
#
#   -e LTX_VAE_TEMPORAL_BLEND=1 -e LTX_VAE_TBLEND_FRAMES=4 -e LTX_VAE_TBLEND_OVERLAP=2
#   -e LTX_VAE_HEAD_F32=1
#   (keep --vae-tiling --vae-relative-tile-size 1x1 for the whole-spatial single tile;
#    TEMPORAL_BLEND takes priority over the streaming/hard-cut path in _compute)
```
- **Coherence expectation:** mesh striping removed (residual + F32 head); temporal seams
  removed (smootherstep feather across the 2-frame overlap band, offload-safe independent
  tiles) → should match Comfy's background stability.
- **VRAM:** unchanged floor (T=4 tile). If a lap wants Comfy-parity smoothness with more
  headroom, raise `LTX_VAE_TBLEND_FRAMES` (watch VRAM as in A1). `LTX_VAE_TBLEND_OVERLAP=2`
  gives a wider feather than the current overlap=1.
- **nvfp4-safe:** this is entirely VAE *decode*, independent of DiT weight quant — the
  nvfp4 DiT recipe is untouched.

### Option B′ — hand-implement feather in the worktree (only if cherry-pick is undesirable)

If for some reason the branch must not take `31f99f5`, the minimal change is to add a
`decode_temporal_blend`-style merge to `decode_temporal_tiled_streaming`: instead of
`slice(drop overlap) + ggml_concat`, keep the overlap frames from **both** neighbours and
host-side blend them with a linear/smootherstep ramp over `overlap*8` output frames,
dividing by accumulated weight — and clear the `feat_map` cache per tile so tiles are
independent (offload-safe). This re-derives exactly what `31f99f5` already ships, so
cherry-pick is strictly preferable; not re-implemented here to avoid divergence from the
proven version.

---

## 4. Verification plan (when GPU is free)

1. Rebuild worktree with `31f99f5` applied.
2. Re-run `run_ablation.sh 1 t2v 0` **A/B**: (a) old flags, (b) `LTX_VAE_TEMPORAL_BLEND=1
   LTX_VAE_HEAD_F32=1`. Eye-test on `:8077/ltx_denoise/`.
3. Confirm on the r0 clip: (i) no fixed 8px grid/striping in still frames, (ii) background
   crowd stable across frames (no phasing) — matching the Comfy reference clip.
4. Check peak VRAM stays ≤ ~11.5 GB (temporal-blend floor = T=4 tile, expected ≈ current).
