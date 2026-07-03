# HANDOFF — Wan2.2 VAE temporal streaming (zero spatial seams at ≤11.5 GB)

Status: **implemented, NOT built, NOT GPU-validated, NOT committed.** Main agent builds
(docker) + GPU-validates. Env-gated, OFF by default — the committed `0.5×0.5` path is
untouched until validated.

## Goal recap

Decode/encode the Wan2.2 VAE at **1×1 spatial tiling (zero spatial seams)** within
≤11.5 GB. Today `--vae-relative-tile-size 1x1` OOMs at a ~15.7 GB compute buffer because a
full-frame × all-frames decode allocates one giant activation tensor. Tiling space fits but
leaves visible seams. **Temporal streaming** processes the video in chunks of frames at FULL
spatial resolution, bounding the activation memory to one chunk while the *causal* feat_cache
makes the chunk boundaries seamless.

---

## 1. Root cause of the disabled partial-graph path ("chunk 1 result is weird")

The disabled path was `WanVAERunner::_compute`'s old `else { /* chunk 1 result is weird */ }`
branch → `build_graph_partial()` → `decode_partial()` / `encode_partial()`, carrying
`_feat_map` / `_enc_feat_map` across `compute()` calls via the runner's named cache.

The streaming *plumbing* (reload feat_cache by name → run one chunk → re-cache → persist) was
correct and is the same pattern the **working LTX** path uses
(`LTXVideoVAE::build_temporal_tile_graph` + `decode_tiled_chunk`, `ltx_vae.hpp`). The bug was
a **view-vs-contiguous / graph-liveness mismatch at the cache-persist boundary**:

- `WAN::CausalConv3d::forward` stores its causal cache slot as a **`ggml_ext_slice` VIEW**
  into the current graph's activations, e.g. `wan_vae.hpp` line ~177:
  `auto cache_x = ggml_ext_slice(ctx->ggml_ctx, x, 2, -CACHE_T, x->ne[2]); ... feat_cache[idx] = cache_x;`
  (same shape at lines ~386, ~684, ~739, ~858, ~913). **It is a view, not a cont.**
- The old `build_graph_partial` persist loop did:
  `cache(name, feat_cache); ggml_build_forward_expand(gf, feat_cache);` on that **view**.
- `GGMLRunner::cache()` (`ggml_extend.hpp` ~4544) conts a view **into a NEW tensor** stored in
  `cache_tensor_map`, but the loop then `ggml_build_forward_expand`s the **VIEW**, not the
  cont. So the cont node was **never wired into the cgraph** and **never computed**.
- After compute, `copy_cache_tensors_to_cache_buffer()` (`ggml_extend.hpp` ~2418) reads the
  `cache_tensor_map` entries (the uncomputed cont) and DMAs them into the persistent cache
  buffer → it **persisted uninitialized memory**.

Why this presents as *"chunk 0 fine, chunk 1 weird"*: chunk 0's decoded **output** reads no
cache (history-less I-frame), so the picture is correct — but the cache it **stores** is
garbage. Chunk 1 **reads** that garbage cache for its causal context → corrupt output.

LTX never hit this because `LTXVAE::CausalConv3d::forward` (`ltx_vae.hpp` ~167) stores
`ggml_cont(...)` in the cache slot up front (comment there: *"a contiguous copy (not a view)
so that the large intermediate `x` can be freed"*), so `cache()` + `forward_expand` operate
on the **same** contiguous node.

(The task's "gallocr clobbers inputs across compute() passes" gotcha is the same family of
bug: the cure is that persisted cache must live in the non-gallocr `cache_buffer`. The
mechanism for that — `cache()` + `copy_cache_tensors_to_cache_buffer()` — was already correct;
the defect was that the node being persisted was never computed.)

---

## 2. Design (the fix / clean reimplementation)

I reimplemented streaming cleanly, mirroring the LTX structure, and fixed the root cause **at
the persist boundary** (so the committed conv `forward` stays byte-identical). All new code is
in `src/model/vae/wan_vae.hpp`.

### Why per-chunk == whole-decode (numerically)
- `conv2` (decode entry) and `conv1` (encode exit) are `CausalConv3d(.., {1,1,1})` — **pointwise
  in time** → applying per temporal chunk == applying to the whole latent, frame-by-frame.
- `unpatchify` / `patchify` mix **spatial** dims only, per-frame independent.
- The only cross-frame coupling is the decoder/encoder's causal `feat_cache`, which we carry
  across chunks. So chunked output == single-graph output (modulo run-to-run kernel
  nondeterminism, which is not chunk-specific).

### Decode (`WanVAE::decode_chunk` + runner streaming)
- `WanVAE::decode_chunk(ctx, z, frame_base)` = `decode()`'s per-frame loop, restricted to the
  frames present in `z` (a temporal slice of the latent), **without** the `clear_cache()` at
  either end. `frame_base` is the **global** latent-frame index of `z`'s first frame — the
  decoder's `CausalConv3d` branches on `chunk_idx` (`==0` / `==1` / `>=2`), so each frame must
  be fed its global index.
- `WanVAERunner::decode_temporal_streaming(n_threads, z, chunk_frames)` loops
  `start = 0, chunk_frames, 2*chunk_frames, …` over the latent frames; each iteration:
  - `sd::ops::slice(z, 2, start, end)` (host-side, so only this chunk uploads),
  - `build_graph_temporal_decode_chunk` → reload `_feat_map` from named cache → `decode_chunk`
    → `persist_feat_map` → expand output,
  - `restore_trailing_singleton_dims(compute(...), z.dim())`, `concat` on dim 2.
  - `free_cache_ctx_and_buffer()` + `cache_tensor_map.clear()` at both ends.

### Encode (`encode_partial` reused + runner streaming)
- Reuses the existing `WanVAE::encode_partial(ctx, x, group_idx)` (patchify + encoder(group) +
  `conv1` + channel-chunk → this group's `mu`; carries `_enc_feat_map`, no `clear_cache`).
- `WanVAERunner::encode_temporal_streaming` streams `encode()`'s natural pixel-frame groups
  — group 0 = pixel frame `[0,1)`, group `i` = `[1+4(i-1), 1+4i)` — one group per `compute()`,
  `conv1` is pointwise so per-group `mu` == the monolithic encode, `concat` on dim 2.

### The fix: `persist_feat_map(gf, fmap, name_of)`
For every non-null cache slot: if it's a view / non-contiguous, **`ggml_cont` it ourselves**,
then `cache(name, fc)` **AND** `ggml_build_forward_expand(gf, fc)` on the **same cont node** so
it is computed before `copy_cache_tensors_to_cache_buffer` reads it. (Contiguous slots — e.g.
a `concat` result — are cached/expanded directly.) Cache names: `wan_dec_feat:<i>` /
`wan_enc_feat:<i>`.

### Chunk-size knob (env, OFF by default)
`LONGCAT_VAE_TEMPORAL_CHUNK=N`:
- `0` / unset → disabled; committed full-frame path runs unchanged.
- `N≥1` → decode streams `N` latent frames per `compute()` pass (encode streams its natural
  `1+4k` groups regardless of `N`, since the encoder's chunking is fixed).
- `LONGCAT_` prefix chosen so `run_wan22_i2v_nvfp4.sh`'s env-forwarding regex
  (`^(GGML_|LONGCAT_|LTX_)`) passes it into the container automatically.

Smaller `N` → lower peak VRAM, more `compute()` passes (more per-pass alloc/offload overhead +
more cache round-trips). Larger `N` → higher peak, fewer passes. Start at `N=1` (max memory
headroom); raise to 2–4 if wall-time matters and VRAM allows.

The intercept sits in `WanVAERunner::_compute`, gated on
`chunk>=1 && src.dim()==5 && src.shape()[2] > 1` (genuine multi-frame video only; single
frames and the image path fall through to the committed graph). It composes with spatial
tiling: at `1x1` there is exactly one spatial tile, so `_compute` is called once with all
frames and streams them temporally. (`make_ggml_tensor` maps `shape()[2] → ggml ne[2]` = the
temporal axis the model counts frames on, so the gate matches the committed path's own frame
convention.)

---

## 3. Files changed

- `src/model/vae/wan_vae.hpp` only:
  - added includes `<cstdlib> <functional> <string>`.
  - `WanVAE::decode_chunk(ctx, z, frame_base, b=1)` — new (mirrors `decode()` loop, no
    clear_cache, global frame index).
  - `WanVAERunner`: `wan_vae_temporal_chunk()` (env), `wan_dec_feat_name`/`wan_enc_feat_name`,
    `persist_feat_map()` (the fix), `build_graph_temporal_decode_chunk()`,
    `build_graph_temporal_encode_chunk()`, `decode_temporal_streaming()`,
    `encode_temporal_streaming()`, and a rewritten `_compute()` with the streaming intercept +
    the original full-frame path as fallback.
  - The dead/buggy `build_graph_partial()` / `decode_partial()` remain (now unreferenced) as
    documentation of the root-caused bug; they are no longer wired into `_compute`.

No call-site changes needed: the knob is env-driven and the run script already passes
`--vae-tiling --temporal-tiling --vae-relative-tile-size`.

---

## 4. Validation plan (for the main agent)

### Build
`make`-equivalent docker build of `build/bin/sd-cli` (the CUDA-13.3 + cuDNN builder used by
`run_wan22_i2v_nvfp4.sh`). Warnings are errors — watch for any unused-variable in the new code.

### Run command
Use `./run_wan22_i2v_nvfp4.sh` with 1×1 spatial tiling + the new chunk knob:

```
VAE_TILE=1x1 LONGCAT_VAE_TEMPORAL_CHUNK=1 \
GGML_CUDNN_CONV3D=1 MAXV=11.2 FP8_LAYERS=blocks. TWOLEVEL=0 CUDNN_OFF=0 FUSE_OFF=1 \
./run_wan22_i2v_nvfp4.sh
```

(`LONGCAT_VAE_TEMPORAL_CHUNK` is auto-forwarded into the container by the script's
`^(GGML_|LONGCAT_|LTX_)` env pass-through.) If `N=1` leaves comfortable VRAM headroom, retry
`LONGCAT_VAE_TEMPORAL_CHUNK=2` (and 4) for fewer passes / less wall time.

### Expected
- **Compute buffer**: the VAE-decode alloc should drop from ~15.7 GB to roughly one chunk's
  worth (≈ `15.7 / n_latent_frames` GB for `N=1`, ~sub-GB), so 1×1 no longer OOMs. Confirm in
  the log's VAE compute-buffer alloc line and that peak (script's `peak …MiB`) stays
  < 11.5 GB. Watch the **persistent cache buffer** size (`… cache backend buffer size = … MB`
  in the log) — it holds the last `CACHE_T=2` temporal frames per conv slot; it is bounded but
  the near-output full-res slots are the biggest single contributor; it should be ~1–2 GB, not
  tens of GB.
- **No spatial seams** (the whole point): eye-test the decoded video — the tile seams visible
  in the `0.5×0.5` baseline must be gone.
- **No temporal-chunk discontinuity**: inspect frames at chunk boundaries. For `N=1` every
  decoded pixel-frame boundary is a chunk boundary, so any flicker/strobe/brightness-step
  between adjacent frames would indicate the causal cache is not carrying correctly (i.e. the
  fix regressed). It should be smooth — a boundary frame must match its neighbour.

### Numerical checks
- Decode output `std` ≈ **0.76**, `nnan = 0` (per-segment range/NaN scan if available).
- **Parity** (strongest gate): decode the SAME latents twice — once with
  `LONGCAT_VAE_TEMPORAL_CHUNK=1` (streamed, 1×1) and once with the committed full-frame path
  at a tile size that fits (e.g. `0.5×0.5`, or full-frame on a bigger card). The streamed
  decode should match the monolithic decode within kernel-nondeterminism noise (cuDNN/cuBLAS
  algo pick — same tolerance already accepted elsewhere in this repo), NOT differ
  structurally. If you can bank the latents, decode-only replay both ways and diff
  (mean-abs / SSIM); a structural mismatch localized to frame indices ≥ the second chunk would
  re-implicate the cache carry.
- Sanity: `LONGCAT_VAE_TEMPORAL_CHUNK=2` and `=4` must produce the **same** video as `=1`
  (chunk size is a memory/throughput knob only, not a quality knob).

### Encode (optional, if the i2v path encodes multi-frame input)
Single init-image encode is 1 frame → streaming auto-skips (gate `shape()[2]>1`), so the i2v
default is unaffected. To exercise encode streaming, run a multi-frame encode (v2v / chain
continuation) with the same env var and check the round-trip (`encode`→`decode`) is seamless.

---

## 5. Risks / things to watch
- **Persistent cache-buffer VRAM**: bounded (`CACHE_T=2` frames × 34 conv slots) but the
  last full-res slots are non-trivial; if peak is tight, it's the second lever after the
  compute buffer. It's freed at both ends of each streaming call.
- **Kernel nondeterminism**: parity is "within noise", not bit-exact (consistent with prior
  LTX/Wan findings in this repo). The boundary-flicker eye-test is the real seam gate.
- If streaming ever returns empty, `_compute` logs a warning and falls back to the committed
  full-frame graph (which will OOM at 1×1 — that's the pre-existing behaviour, not a new
  regression).
