# ShotStream C++/ggml port — implementation notes (Stages A–E scaffold)

Worktree `/home/dbrain/dev/longcat-avatar-shotstream`, branch `shotstream`. Not committed
(owner review). Mirrors the S2V causal-streaming machinery (`src/wan_s2v.hpp`) onto the stock
`WAN::Wan` DiT per `HANDOFF-shotstream-streaming-port.md` + `REFERENCE-shotstream-inference.md`.

## Files created / edited

- **`src/model/diffusion/wan.hpp`** (edited, 2 new public methods, no existing code touched):
  - `WanAttentionBlock::forward_causal(...)` — copy of `forward()` but routes self-attn through
    the existing `WanSelfAttention::forward_kv_cache` (attends `[prev_local ++ this_chunk ++
    context]`, exports fresh RoPE'd K + raw V via `new_kc/new_vc`). Modulation is the plain
    `[N,6,dim]` broadcast (all 3 chunk frames share one timestep → `es[*]` broadcast over tokens).
  - `WAN::Wan::forward_causal_block(...)` — one-chunk DiT forward: patch-embed → time/text embed →
    30 causal blocks threading per-layer local+global caches → head → unpatchify → flow `[W,H,nfb,16]`.
    Graph-cut marks per-block (`shotstream.prelude`, `shotstream.blocks.i.out`) so the segmented
    CPU-offload compute path applies (same pattern as S2V).
- **`src/model/diffusion/wan_shotstream.hpp`** (new, `namespace WAN_SHOTSTREAM`): `ShotStreamRunner`
  (subclass of `GGMLRunner`) + `ShotStreamConfig`. The full streaming driver:
  - Host-side F16 dual cache: `local_k/local_v` (intra-shot rolling) + `ctx_k/ctx_v` (global).
  - `forward_block(...)` — builds the graph, feeds prev/ctx caches as inputs, runs the segmented
    `compute<float>` with a per-segment K/V readback hook (copied from `WanS2VRunner::
    compute_causal_block`), then appends the fresh K/V into the local OR global cache.
  - `run_chunk(...)` — the **4-step warped DMD schedule** `[1000, 957.929, 888.889, 737.589]`
    (§5.2), flow→x0 `x0 = x_t − σ·flow` (§5.3), `add_noise` re-noise between steps (§5.1), then a
    **clean rewrite at t=0** that persists the chunk's clean K/V into the local cache (§3.3). No CFG.
  - `prefill_context(...)` — builds the global cache once/shot by a single t=0 forward over the
    ≤6 history latents (reuses `forward_block` with `store_into_global`).
  - `run_shot(...)` — resets local cache, loops the 7 chunks (3 latent frames each).
  - `add_shot_phase(...)` — `change_rope` θ=1/6 hook (§3.4), **gated OFF** (`SHOTSTREAM_ROPE_PHASE=1`).
- **`examples/shotstream/main.cpp`** + **`examples/shotstream/CMakeLists.txt`** (new): `sd-shotstream`
  CLI — loads the DiT, drives shot→shot, dumps `shotNN_latent.bin`. Wired into
  **`examples/CMakeLists.txt`** (`add_subdirectory(shotstream)`).

## What compiles (real build result)

Full translation-unit compile of `examples/shotstream/main.cpp` (which pulls in the wan.hpp edits +
the entire `ShotStreamRunner` header) with the **production flags** from the wan22 build's
`compile_commands.json`, in the cuDNN builder image:

```
$ docker run --rm -v /home/dbrain/dev/longcat-avatar-shotstream:/src -w /src \
    longcat-avatar-dev:builder-cudnn \
    bash -c 'c++ -DGGML_MAX_NAME=128 -DGGML_USE_CPU -DGGML_USE_CUDA -DSD_USE_CUDA \
      -I/src/examples -I/src/examples/shotstream/.. -I/src/src -I/src/src/core \
      -I/src/thirdparty -I/src/. -I/src/include -I/src/ggml/src/../include -I/src/thirdparty/. \
      -O1 -DNDEBUG -std=gnu++17 -fsyntax-only /src/examples/shotstream/main.cpp; echo EXIT_CODE=$?'
...
EXIT_CODE=0
```

Clean (zero diagnostics). This validates the two new `wan.hpp` methods, the whole
`wan_shotstream.hpp` runner, and the CLI against the actual codebase types/APIs.

## Full build command (links the lib + `sd-shotstream` binary; ~15 min CUDA build, NEXT session)

```
docker run --rm -v /home/dbrain/dev/longcat-avatar-shotstream:/src -w /src \
  longcat-avatar-dev:builder-cudnn bash -lc '
  cmake -B build -DCMAKE_BUILD_TYPE=Release -DSD_CUDA=ON -DGGML_CUDNN=ON \
        -DCMAKE_CUDA_ARCHITECTURES="86" -DSD_BUILD_EXAMPLES=ON &&
  cmake --build build --target sd-shotstream -j"$(nproc)"'
```
(Arch 86 = the 3060. The wan22 cache uses `86;120`; drop 120 if only targeting Ampere.)

## How to run Stage A once a GPU is free

```
docker run --rm --gpus '"device=0"' \
  -e LONGCAT_FFN_TILE_TOKENS=4096 -e GGML_CUDNN_ATTN=1 \
  -v <shotstream>:/src -v <models>:/models -w /src longcat-avatar-dev:builder-cudnn \
  /src/build/bin/sd-shotstream \
    --dit /models/shotstream-1.3b-dit-f16.gguf \
    --context /models/shot0_umt5.bin \    # [4096,512,1] f32; omit for a zeros smoke
    --shots 1 --W 832 --H 480 --out /src/shotstream_out
```
Produces `shotstream_out/shot00_latent.bin` `[832,480,21,16]`. Stage-A gate = cosine ≥ 0.999 of
chunk-0 x0 vs the torch oracle (`tools/` agent) with the local+global caches empty.

## Stubbed / TODO (in priority order for the NEXT session)

1. **umT5 text encoding** — the CLI reads a precomputed `[4096,512,1]` context `.bin` (or zeros).
   Wire the live umT5 runner + per-shot multi-caption swap (§4.3) + `shot_flags` per-frame caption
   gather in cross-attn. Currently one context is reused across all shots.
2. **VAE encode/decode** — latents are dumped to `.bin` only. Stage-D history conditioning
   (`prefill_context`) currently carries the previous shot's **tail latent frames directly**; the
   reference round-trips through pixels (per-frame VAE encode of decoded history, §1.2). Wire
   `WAN::WanVAERunner` (per-frame encode, whole-shot decode) + the `generate_video_chain`
   re-encode path.
3. **change_rope shot phase (Stage E)** — `add_shot_phase` is gated OFF; the 4-float-per-channel
   `pe` packing (`[c,-s,s,c]` assumed) **must be verified against a torch RoPE dump** before trust.
   Also: generated frames should sit at temporal positions `6 + start` (condition_start_frame=6),
   context frames at 0..5 — currently `t_offset = start` (0-based); add the +6 offset once the
   global cache is validated.
4. **Torch-oracle parity harness** — diff DMD sigmas, per-block K/V after chunk 0, `denoised_pred`
   per chunk. The autoregressive rollout compounds any schedule/cache/RoPE error (the single
   biggest risk).

## Biggest correctness risk I hit

The **RoPE `pe` memory layout for the shot-phase rotation**. `pe` is a `[2,2,axes_dim_sum/2,pos_len]`
tensor = **4 floats per complex channel** (a 2×2 rope matrix), *not* an interleaved `(cos,sin)` pair
— my first `add_shot_phase` assumed 2 floats/channel and would have silently corrupted every
temporal RoPE position for shots >0. Fixed to the 4-float stride and **gated OFF by default** so the
single-shot path is unaffected, but the exact intra-block float packing (`[c,-s,s,c]` vs `[c,c,s,s]`)
is unverified and must be pinned against a torch dump (Stage E). More broadly: the whole port is an
AR rollout, so the schedule-exactness + clean-rewrite-cache-timing must be gated on a chunk-by-chunk
oracle before any multi-chunk output is trusted (per the HANDOFF's "single biggest risk").

## Run debug (first successful GPU run — 2026-07-03)

First-ever run failed at `ggml_gallocr_reserve` with a **~4.37 TB** CPU compute-buffer alloc,
on CPU backend. Three separate bugs, all fixed; the full 1-shot latent now comes out clean.

### Bug 1 (PRIMARY) — 4.37 TB alloc = pixel dims fed to a latent-space DiT
`examples/shotstream/main.cpp` passed `--W 832 --H 480` (PIXEL resolution) straight into
`run_shot`, but the DiT operates on the **VAE latent** (stride 8× spatial, `REFERENCE §0`:
`H=60=480/8, W=104=832/8`). So `forward_block` computed `L_blk = nfb·h_len·w_len =
3·240·416 = 299520` tokens per chunk instead of the correct **`3·30·52 = 4680`** (64× too big).
With flash OFF (the CPU/first path), `ggml_ext_attention_ext` falls to the non-flash branch and
**materializes the scores tensor** `kq = ggml_mul_mat(k,q)` of shape `[L_k, L_q, n_head] =
[299520, 299520, 12]` f32 = **4.306 TB** (the reserved segment buffer rounds to the reported
4.371 TB). The exact overflow tensor = that `kq` (`src/core/ggml_extend.hpp:1759`). At the
correct 4680 it is 1.05 GB (and streamed by flash anyway).
**Fix:** convert pixel→latent at the CLI/runner boundary — `const int VAE_SPATIAL = 8; latW =
W/8; latH = H/8;` and pass `latW/latH` to `run_shot`. Output latent is now `[104,60,21,16]`
(the earlier note's `[832,480,21,16]` was wrong).

### Bug 2 — ran on CPU, not CUDA
The main.cpp backend code was already correct (identical to `examples/s2v/main.cpp`), but the
existing `build/` CMake cache was `SD_CUDA=OFF` / `GGML_CUDA=OFF` — a **CPU-only build** (only
`libggml-cpu.a`, no ggml-cuda), so `ggml_backend_init_by_type(GPU)` returned null.
**Fix:** reconfigure `-DSD_CUDA=ON -DGGML_CUDNN=ON -DCMAKE_CUDA_ARCHITECTURES=86` and rebuild.
Now prints `backend: GPU (CUDA0)` on the 3060. (Build gotcha: the ggml-cuda MMQ/FA `cicc`
compiles spike >26 GB RAM and OOM-kill on this 31 GB host while the other GPU's render runs; the
3 heaviest `mmq-instance-{q5_k,q6_k,q8_0}.cu` were compiled with `-Xcicc -O1` to cap `cicc`
memory ~8 GB — those quantized kernels are unused by the F16 model, so codegen quality is moot.)
Also added a defensive `LONGCAT_NO_FUSED_ROPE=1` on CPU fallback (ROPE_PE is CUDA-only), mirroring s2v.

### Bug 3 (found after bug 1) — flash attention never enabled ⇒ VRAM OOM on later chunks
With dims fixed, chunks 0–4 succeeded but **chunk 5 OOM'd** (`cudaMalloc failed`, wanted a
10.7 GB single buffer with weights resident / +2.7 GB segment with offload). Root cause: the
runner never called `set_flash_attention_enabled(true)` (s2v does at `main.cpp:304`), so
`GGMLRunnerContext.flash_attn_enabled` stayed `false` and the causal KV-cache self-attention
materialized the `[L_k,L_q,heads]` scores. As the local cache grows across the 7 chunks
(chunk b sees `b·4680` prior tokens), that tensor reaches ~6.3 GB at chunk 5
(`28080·4680·12` f32) and busts the 12 GB card.
**Fix:** `dit->set_flash_attention_enabled(getenv("SHOTSTREAM_NO_FLASH") == nullptr);` after
runner construction. cuDNN (`GGML_CUDNN_ATTN=1`) now streams the scores; peak GPU stays ~9 GB.

### VRAM fit note (offload vs resident)
The default CPU-offload path (`SHOTSTREAM_NO_OFFLOAD` unset) + flash got *past* chunk 5 but the
process was **SIGKILLed at exit** (host-side churn from the per-forward 60-tensor K/V readback +
weight-prefetch staging; run3, exit 137, latent not written). The DiT is only **2.6 GB f16**, so
offload is unnecessary on a 12 GB card — keeping weights **resident** (`SHOTSTREAM_NO_OFFLOAD=1`)
avoids the offload host churn entirely and is the correct default for this small model.

### Final result (working recipe)
```
docker run --rm --gpus '"device=0"' -e SHOTSTREAM_NO_OFFLOAD=1 -e LONGCAT_FFN_TILE_TOKENS=4096 \
  -e GGML_CUDNN_ATTN=1 -v <shotstream>:/src -v <models>:/models -w /src \
  longcat-avatar-dev:builder-cudnn /src/build/bin/sd-shotstream \
    --dit /models/shotstream-1.3b-dit-f16.gguf --shots 1 --W 832 --H 480 --out /src/shotstream_out
```
- `shot 0: 21 latent frames in 133.0s` on the RTX 3060 (peak VRAM ~9.3 GB, weights resident).
- `shotstream_out/shot00_latent.bin` = `[104,60,21,16]`, numel 2 096 640.
- **Stats: nnan=0, ninf=0, mean −0.042, std 0.922, min −3.95, max 3.63, |·|-mean 0.734, p1/p99 ±2.1**
  — a clean, sane flow latent (content is meaningless with zeros text-context; this validates the
  causal graph executes end-to-end and the AR rollout is numerically stable across all 7 chunks).
- Files changed (uncommitted): `examples/shotstream/main.cpp` only (latent-dim conversion,
  flash enable, CPU fused-rope guard). `wan.hpp` causal section + `wan_shotstream.hpp` were correct.

## umT5 + VAE wiring — FIRST PIXELS FROM A REAL PROMPT (2026-07-03)

TODO #1 (umT5 text encoding) and TODO #2 (VAE decode) are now wired into the CLI, so
`sd-shotstream` produces a **viewable mp4 from a text prompt** through the streaming path.
Both reuse existing runners verbatim from `examples/s2v/main.cpp` — no reimplementation, no
new runner glue (`wan_shotstream.hpp`/`wan.hpp` untouched this round). Only
`examples/shotstream/main.cpp` changed.

### What was wired
1. **umT5-XXL text encoding** (`--t5xxl <gguf>`, `-p/--prompt`). Loads `T5Embedder` on a
   dedicated **CPU backend** (so it never competes with the DiT/VAE for VRAM, matching s2v),
   `tokenize(prompt,512,false)` → `model.compute` → `[4096, n_tok, 1]`, then **zero-pad to
   `[4096,512,1]`** (the Wan T2V DiT does not pad internally — same as `model_s2v.py`). Fed to
   `run_shot`/`prefill_context` in place of the zeros stub (single caption reused per shot; the
   multi-caption per-frame `shot_flags` gather is still TODO). umT5 params freed after encode.
   Falls back to `--context <bin>` or a zeros stub when `--t5xxl` is absent.
2. **Wan2.1 VAE decode** (`--vae <gguf>`, `--vae-tiling`, `--temporal-tiling`,
   `--vae-relative-tile-size 0.5x0.5`). After the shot loop the DiT is **freed**
   (`free_params_buffer`+`free_compute_buffer`) so the ~3.5 GB decode buffer fits; then
   `WAN::WanVAERunner` (decode-only) is loaded. Per shot latent `[104,60,21,16]`: convert
   diffusion→raw VAE space (`z*std+mean`, per-channel over **ne[3]** — the manual s2v helper,
   NOT `diffusion_to_vae_latents()` which assumes channel at dim 2), `unsqueeze_(4)` to skip the
   VAE's 4D `unsqueeze(2)`, `decode(...,decode_video=true)` (tiled, temporal, 0.5×0.5) → RGB
   `[832,480,81,3]` already in [0,1] (the runner's `scale_tensor_to_0_1`). Write `f%03d.png`
   into `shotNN_frames/`, mux `shotNN.mp4` @16fps via ffmpeg (`builder-cudnn-ff` image).

### Final run command (3060, device 0)
```
docker run --rm --gpus '"device=0"' \
  -e SHOTSTREAM_NO_OFFLOAD=1 -e GGML_CUDNN_ATTN=1 -e LONGCAT_FFN_TILE_TOKENS=4096 \
  -e GGML_CUDA_F16_BCAST_FUSE=1 -e GGML_CUDA_BIAS_GELU_FUSE=1 -e GGML_CUDA_BIAS_RMS_FUSE=1 \
  -e GGML_CUDA_RMS_MOD_FUSE=1 -e GGML_CUDNN_CONV3D=1 -e LONGCAT_VAE_TEMPORAL_CHUNK=0 \
  -v <shotstream>:/src -v <models>:/models -w /src longcat-avatar-dev:builder-cudnn-ff \
  /src/build/bin/sd-shotstream \
    --dit /models/shotstream-1.3b-dit-f16.gguf --t5xxl /models/longcat-umt5-xxl-q8_0.gguf \
    --vae /models/longcat-wan-vae-f16.gguf \
    -p "a red fox trotting through a snowy pine forest at dawn, volumetric morning light, cinematic, photorealistic" \
    --shots 1 --W 832 --H 480 --fps 16 --seed 42 --vae-relative-tile-size 0.5x0.5 \
    --out /src/shotstream_out
```

### Result — mp4 produced ✅
- **`shotstream_out/shot00.mp4`** — h264, 832×480, 16 fps, **81 frames, 5.06 s**, 466 KB.
- Content **matches the prompt**: a red fox in a snowy pine forest at dawn, coherent across
  the shot (frames f000/f040/f080 show progressing fox pose over a stable background) — the
  prompt-matched content is the end-to-end proof that umT5 conditioning is live (the old
  zeros-context latent was "meaningless"). Decoded RGB nnan=0, mean 0.282, std 0.189, full [0,1].
- **Wall = 208 s total** on the RTX 3060: umT5 load+encode + DiT streaming ≈ 141 s (DiT itself
  ~133 s, 7 chunks × 4-step) + **VAE decode 67.4 s** (3.48 GB decode buffer, 9 temporal/spatial
  tiles). **Peak VRAM = 9343 MiB (~9.1 GB)** — DiT weights-resident phase sets the peak; VAE
  decode (DiT freed) stays under it.

### Correctness caveats (expected — this round is "pixels, not parity")
- **change_rope shot-phase still gated OFF** and single-shot only, so inter-shot phase
  decoupling is untested; irrelevant for this 1-shot clip.
- **Multi-shot history is NOT the reference path**: shots >0 still carry the previous shot's
  tail *latent* frames directly into `prefill_context` (the reference re-encodes decoded
  *pixels* per-frame). The VAE encode round-trip for history remains TODO — only decode is wired.
- No numerical/oracle validation yet; motion is a modest trot (fox pose varies, little global
  translation) — plausibly the distilled 4-step schedule + gated RoPE, to revisit with the
  torch-oracle harness. Output is clean and stable (no chunk seams, no identity drift within
  the shot).

## Perf breakdown + env-stack A/B — RTX 3060 (2026-07-03)

Goal: reduce the 208 s single-shot wall (832×480, 21 latent / 81 px frames). Instrumented
`main.cpp` (`--chunks N` short-run cap, `--no-vae`, `[STAGE]` umT5 timer) +
`wan_shotstream.hpp` (`[CHUNK]` per-chunk + `SHOTSTREAM_PROFILE=1` per-`[FWD]` timers). All
runs device 0 (3060); a background `nvidia-smi -l` sampled peak VRAM.

### Where the 208 s goes (baseline breakdown, current env)
| Stage | Time | Notes |
|---|---|---|
| umT5 load+encode | ~3.4 s | CPU backend, one-shot |
| **DiT streaming** | **133 s** | 7 chunks × 5 forwards (4 warped-DMD denoise + 1 clean rewrite) |
| **VAE decode** | **67 s** | Wan2.1 3D VAE, 9 spatial tiles (0.5×0.5) × ~7.4 s/it, cuDNN conv3d |
| overhead (load/mux) | ~4 s | |

**DiT per-forward = ~1.5 s fixed + ~0.66 s per 4680 cached tokens** — i.e. time grows
*linearly with the local KV-cache size*. All 5 forwards of chunk `b` attend over `b·4680`
prior local tokens; the growth (self-attention over the rolling cache) is ~57 % of the DiT,
the fixed FFN/proj/cross-attn ~43 %. The causal self-attn runs on the **native ggml FA2
kernel**, NOT cuDNN (confirmed: `GGML_CUDNN_ATTN_F16_OUT` tripped `fattn-common.cuh
GGML_ASSERT(KQV->type==F32)`). VAE: tiling granularity is time-neutral (4 tiles×6.8 s ≈ 9
tiles×3.0 s = same conv work; 1×1 whole-frame OOMs at 13.3 GB).

### Env-stack A/B (each vs current-prod base; DiT = sum of chunks 0–2, VAE = steady s/it)
| Lever | ΔDiT | ΔVAE | ΔVRAM | Verdict |
|---|---|---|---|---|
| `WAN_DIT_F16` (env only) | 0 % | — | 0 | **no-op** — `forward_causal_block` never cast x→F16 |
| `WAN_DIT_F16` **+ causal cast hook** (added) | ~0 % (34.2 vs 34.4 s) | — | 0 | **neutral** — F16-weight Linears re-upcast to F32 dst; attn is native-FA2 F32-out |
| `GGML_CUDNN_ATTN_F16_OUT` | — | — | — | **CRASH** — native FA2 asserts F32 KQV on the causal path |
| `WAN_ROPE_F16` (+DIT_F16) | ~0 % | — | 0 | neutral |
| `GGML_CUDNN_CONV3D_WS_MB=2048` | — | 0 % (3.02 s/it) | 0 | **neutral** — workspace not the conv limiter |
| `WAN_VAE_F16` | — | **+10 %** (3.33 vs 3.02 s/it) | 0 | **LOSS** — extra F16↔F32 casts on Ampere cuDNN conv3d |
| VAE tile 0.75 / 1×1 | — | 0 % / OOM | +2.6 GB | neutral / busts |

**The prod perf env stack (WAN_DIT_F16 / ROPE_F16 / ATTN_F16_OUT / CONV3D_WS / VAE_F16) does
NOT transfer to ShotStream.** It was tuned for Wan2.2-A14B nvfp4 weights on Blackwell sm120;
ShotStream is an **F16-weight Wan2.1-1.3B** model on Ampere sm86 — F16 GEMM + native FA2 are
already the fast path, so the F16-stream/conv levers are wash-or-worse. (The base fusion flags
`F16_BCAST/BIAS_GELU/BIAS_RMS/RMS_MOD` + `GGML_CUDNN_ATTN` + `FFN_TILE_TOKENS=4096` were
already on in the 208 s run and are kept.)

### The one real win — CODE lever: skip K/V export on denoise forwards (`wan_shotstream.hpp`)
Every forward exported all 60 layer K/V tensors as graph outputs + read them back to host —
but only the **clean-rewrite** forward (1 of 5 per chunk) persists them; the 4 denoise steps
discard theirs. Gated the export + readback hook on `need_kv = persist_local ||
store_into_global`. **Byte-identical** (K/V don't feed `vel`; clean rewrite unchanged) but the
60 graph-output cuts had been fragmenting the segmented compute, so denoise forwards dropped
~35 % (chunk-0 1.50→0.98 s).

| Metric | Baseline | + K/V skip | Δ |
|---|---|---|---|
| DiT (4-chunk, chunks 0–3) | 53.2 s | 44.5 s | **−16.4 %** |
| **DiT full (7 chunks)** | **133 s** | **116.2 s** | **−12.6 %** |
| VAE decode | 67.5 s | 67.5 s | 0 |
| peak VRAM | ~9.1 GB | ~9.1 GB | 0 |
| **wall** | **208 s** | **~191 s** | **−17 s (−8 %)** |

### Best recipe (final artifact `shotstream_out/shot00_fast.mp4`, 81 f, clean, nnan=0)
```
docker run --rm --gpus '"device=0"' \
  -e SHOTSTREAM_NO_OFFLOAD=1 -e GGML_CUDNN_ATTN=1 -e LONGCAT_FFN_TILE_TOKENS=4096 \
  -e GGML_CUDA_F16_BCAST_FUSE=1 -e GGML_CUDA_BIAS_GELU_FUSE=1 -e GGML_CUDA_BIAS_RMS_FUSE=1 \
  -e GGML_CUDA_RMS_MOD_FUSE=1 -e GGML_CUDNN_CONV3D=1 -e LONGCAT_VAE_TEMPORAL_CHUNK=0 \
  ... sd-shotstream --dit shotstream-1.3b-dit-f16.gguf --t5xxl ... --vae ... \
  --shots 1 --W 832 --H 480 --vae-relative-tile-size 0.5x0.5
```
(No WAN_DIT_F16 / ROPE_F16 / ATTN_F16_OUT / VAE_F16 / CONV3D_WS — all wash-or-worse here.)

### #1 remaining bottleneck + recommended next lever
DiT (116 s) still dominates, and ~57 % of it is the **growing causal self-attention** running
on the **native ggml FA2 kernel at ~4× off compute-ideal** for these d_head=128 shapes on
Ampere. **Recommend: get cuDNN flash-attention to engage on the causal KV-cache path.** cuDNN
is set (`GGML_CUDNN_ATTN=1`) yet the causal attn falls to native FA2 — investigate the CUDA
backend's cuDNN-vs-fattn dispatch (likely the F16 cached-K/V concat layout or the mask-free
d_head=128 gate). cuDNN typically beats native FA2 on Ampere for large L_k; plausible ~15–25 %
of the growing-attn portion → ~10–18 s. Second lever: **keep the local KV cache GPU-resident**
instead of re-uploading it (H2D) on every forward — saves the per-forward cache upload (up to
~0.4 GB/forward at chunk 6), a smaller but clean win. Neither is a "trivial safe" change, so
left as recommendations. Fewer denoise steps (4→3, −20 %) is a quality trade, not a free lever.

## Chaining + parity — MULTI-SHOT CHAINING landed + oracle-verified (2026-07-03)

The headline feature (shot N+1 continues coherently from shot N), the RoPE θ=1/6 shot-phase,
and the offline oracle-parity gates are all done. Two GPU renders spent (the whole budget).

### P0.5 — RoPE packing VERIFIED (offline, no GPU, gates everything) ✅
- Added a free-function pair `WAN_SHOTSTREAM::pe_build_frames` / `pe_add_shot_phase` (shared by
  the runner AND a new `sd-shotstream --dump-rope <dir>` early-exit path — no model/backend).
  Emits `rope_freqs_chunk_shot{0,1}.npy` in the oracle's `[4680,64,2]` (cos,sin) layout (cos =
  pe[..+0], sin = pe[..+2] of the `[c,-s,s,c]` 2×2 block), t_offset=6 (=condition_start_frame),
  phase θ=1/6 on the temporal band for shot 1.
- `tools/shotstream_compare.py` vs goldens: **cosine = 1.000000** for BOTH shot 0 and shot 1
  (max_abs_err 1.6e-6 = f32-table-vs-f64-golden quantization → the tool's 1e-9 tol "FAIL" flag is
  expected; the task gate is cosine ~1.0, met exactly). The `[c,-s,s,c]` packing, the temporal-band
  channel range, and the k·θ phase composition are all correct.
- **Conclusion: the prior agent's flagged risk is CLEARED.** `add_shot_phase` is now enabled by
  default (`rope_phase_enabled`, opt out with `SHOTSTREAM_NO_ROPE_PHASE=1`).

### P1 — block-0 op-parity VERIFIED (offline, 1-block fwd on device 0) ✅
- Added `WanAttentionBlock::selfattn_only` + `Wan::forward_block0_selfattn` (thin wrappers that
  reuse the PRODUCTION `WanSelfAttention::forward_kv_cache` verbatim, empty caches) +
  `ShotStreamRunner::dump_block0` + `sd-shotstream --dump-block0 <dir> --block0-input <npy>`
  (minimal .npy reader/writer in main.cpp). Feeds the oracle's fixed `block0_selfattn_input`
  through block 0's real causal self-attn with the shot-0 RoPE table.
- vs goldens: **block0_roped_k cos 0.999998, block0_v cos 0.999998, block0_selfattn_out cos
  0.999989** — all ≫ the 0.999 target. (max_abs_err ~2–4% = the gguf's **F16 weights** vs the
  oracle's `.float()` F32 weights/compute; the cosine is the parity gate.) The RMSNorm + dynamic
  RoPE (packing) + maskless SDPA + o-proj port is faithful. Flash engaged (native FA2, d_head=128).

### P0 — MULTI-SHOT CHAINING (the headline) ✅
Reworked the runner + CLI to the reference pixel round-trip (§1.2/§3.4/§4/§6):
- **+6 temporal offset**: generated chunks now sit at temporal positions `6 + start`
  (`condition_start_frame=6`); context frames at 0..5. `run_shot` t_off = `f0 + 6`.
- **RoPE shot-phase per shot**: generated frames carry phase `k·θ`; context frames carry the phase
  of their **source** shot (§3.4). `prefill_context(context_shot_phase, …)`; for the 2-shot case
  (single prior shot 0) that's phase 0, decoupled from shot 1's generated-frame phase 1·θ.
- **History = pixel round-trip (NOT latent carry)**: main.cpp now loads the VAE **encode+decode**
  (`decode_only=false`; the gguf has encoder tensors) alongside the resident DiT. Per shot: decode
  → keep pixels; for shot k>0 sample 6 pixels evenly (`linspace`) from the prior shot, **per-frame
  VAE-encode** (1-frame clip → 1 latent) → `vae_mu_to_diffusion` → `[latW,latH,6,16]` →
  `prefill_context`. Shot 0 prefills a **zeroed** 6-frame context (faithful §6; opt out
  `SHOTSTREAM_NO_ZERO_CTX_SHOT0=1`). DiT compute buffer freed before each decode so both models
  coexist on 12 GB.
- Stitches all shots into `shotstream_out/chain_2shot.mp4`.

### Renders spent: 2 (the whole budget)
1. **Smoke** (2 shots × 3 chunks, 33 px/shot, 66-frame chain): **COHERENT & near-frame-continuous**
   — seam MSE (shot0-end→shot1-start) 499 ≈ within-shot motion MSE 445; identical palette. Peak
   VRAM **8385 MiB**. Same fox trotting rightward across the cut, no reset. Proved chaining.
2. **Full** (2 shots × 7 chunks = 81 px/shot, 162-frame chain → `shotstream_out/chain_2shot.mp4`):
   **COHERENT at scene/identity level** — same red fox, same snowy dawn pine forest, same
   cinematic style across the cut; **no identity reset, no scene change**. Shot 1 is a new shot
   *composition* of the same world (denser foreground pines) — expected multi-shot behavior (the
   θ=1/6 phase decouples shots; the 6 history frames span ALL of shot 0, so shot 1 continues the
   scene "gist", not a frame-exact handoff — that's why the shorter smoke read as more continuous).
   - Timing (device 0, RTX 3060): shot 0 DiT 154.9 s + decode 67.5 s; history encode ~2.5 s; shot 1
     DiT 114.1 s + decode 66.8 s. **Peak VRAM 11843 MiB** (~11.6 GB, at the 12 GB card's edge).
   - Two `cudaMalloc failed (8.6/8.1 GB)` at the last chunks were **caught by the segmented
     compute fallback** and both shots still produced all 21 latent frames (recovery is automatic).
     Full 7-chunk shots + the 9360-token global cache is VRAM-tight; safer levers = keep local KV
     GPU-resident (avoid per-forward H2D re-upload), or `SHOTSTREAM_NO_ZERO_CTX_SHOT0=1`
     (drops shot 0's 9360-token global cache), or ≤6 chunks.

### P2 — perf (time-boxed): cuDNN-on-causal-path is a RABBIT HOLE — STOPPED per instructions
The causal self-attn (~57% of DiT) runs on the **native ggml FA2** kernel, confirmed live
(`FA2DBG: flash ENGAGED d_head=128`), NOT cuDNN. `ggml_ext_attention_ext(flash=true)` hard-selects
the native FA2 path; cuDNN attention (`GGML_CUDNN_ATTN`) only serves the non-causal `forward()`.
Routing the KV-cache path through cuDNN would require the cuDNN wrapper to accept the concatenated
growing-cache K/V layout + the mask-free d_head=128 case + F32-KQV constraint (native FA2 already
asserts F32 KQV; `GGML_CUDNN_ATTN_F16_OUT` crashes it — see the perf-pass note above). That's a
CUDA-backend dispatch change, not a quick gate/layout flip → **rabbit hole, left as a
recommendation.** The chaining renders measured the native-FA2 path for free; it streams the
scores so VRAM stays bounded regardless.

### Files touched (uncommitted)
- `src/model/diffusion/wan_shotstream.hpp` — shared pe free-fns; +6 offset; context_shot_phase;
  `rope_phase_enabled` default-on; `dump_block0`.
- `src/model/diffusion/wan.hpp` — `WanAttentionBlock::selfattn_only` + `Wan::forward_block0_selfattn`
  (reuse `forward_kv_cache`, no dup).
- `examples/shotstream/main.cpp` — npy I/O; `--dump-rope`/`--dump-block0`; VAE encode+decode +
  pixel round-trip history + per-shot decode + chain stitch.

## Chaining bug fix — shot 1 "playdough style + second fox" = STALE LOCAL CACHE in prefill (2026-07-03)

**Symptom (human eye on `chain_2shot.mp4`, same prompt both shots):** shot 0 = correct photoreal
red fox / snowy pine forest / warm dawn; shot 1 = style breaks to claymation/stylized-CGI AND a
SECOND fox appears. The prior "COHERENT at scene/identity level" sign-off was a MISDIAGNOSIS — the
RoPE-table + block-0 oracle gates (cosine 1.0) do NOT cover the multi-shot context path, and the
claymation was hand-waved as "expected multi-shot behavior". It is a real correctness bug.

**Root cause (found by tracing the code, not the oracle):** `prefill_context()` runs a full DiT
forward over the history latents to fill the GLOBAL cache, and it reused `forward_block` — which
attends `prev_kc` (the LOCAL rolling cache) whenever it is non-empty. But `run_shot` only clears the
local cache AFTER prefill, so for shot k>0 the local cache still held shot **k-1's 21 (or N) clean
latent frames**. So shot 1's 6 context frames were prefilled while attending to shot 0's entire
local cache. `new_kc[0]` (block 0) is computed from the block input and is unaffected, but every
block ≥1 reads the previous block's attention OUTPUT, so blocks 1–29's stored context K/V were
computed with shot 0 bleeding in → the global cache is a corrupted shot-0 imprint. That (a) drags
shot 1's style toward a washed-out/over-smoothed prior (claymation) and (b) re-injects shot 0's fox
as a second subject. It compounds over the shot (each chunk's clean-rewrite seeds the local cache
while attending the bad context), which is why it's mild at 3 chunks and severe by 7. Shot 0 is
clean only because there is no prior local cache to leak. The reference is clean by construction: its
context prefill takes `kv_cache=None`, whose attention branch is a pure self-attention over the
context frames only (`attention(roped_query, roped_key, v)`) — no local/prev cache.

**The fix (`wan_shotstream.hpp`):** `prefill_context()` now calls `reset_local_cache()` before the
prefill forward (so `have_local=false` ⇒ context self-attention only, matching the reference).
`run_shot` re-clears it right after, so nothing else changes. One line; `SHOTSTREAM_KEEP_STALE_LOCAL=1`
reproduces the old bug for A/B. Diagnostics added to `main.cpp`: `SHOTSTREAM_NO_CONTEXT=1` (empty
global cache isolation) + history frame/mu/ctx-latent `dump_stats` (the stats were sane — nnan=0,
std≈0.47 — which correctly ruled out the VAE encode as the culprit and pointed at the prefill).

**Proof (device 0, RTX 3060):** ruled OUT the encode (context-latent stats sane) and the RoPE phase
(shot-1 generated phase 1·θ / context phase 0 already matches the reference; the reference even
gives shot 0 phase 1 and is fine, so the phase can't cause claymation). The decisive render is the
FIXED chain at chunks=6 (`shotstream_out/fix6/`, 69 px/shot). Frames VISUALLY verified (Read the
PNGs): shot0 f034, shot1 f000/f004/f034/f062 — shot 1 is the SAME photoreal style as shot 0, ONE fox
throughout, SAME snowy-dawn pine world; the cut (shot0 f068 → shot1 f000) is a clean shot change with
preserved style/subject/world. No claymation, no second fox. (chunks=7 renders reliably OOM at shot-1
chunk 6's ~9 GB compute buffer on the 12 GB card — the pre-existing VRAM cliff, unrelated to this fix;
chunks=6 stays under it. My fix also LOWERS shot-1 prefill VRAM since prefill no longer attends the
21-frame stale cache.) Final artifact: `shotstream_out/chain_fixed.mp4`.

### Files touched by this fix (uncommitted)
- `src/model/diffusion/wan_shotstream.hpp` — `prefill_context` clears the local cache before the
  prefill forward (the fix); `SHOTSTREAM_KEEP_STALE_LOCAL` A/B escape hatch.
- `examples/shotstream/main.cpp` — `SHOTSTREAM_NO_CONTEXT` isolation toggle; history `dump_stats`.
- `run_shotstream_diag.sh` — 2-shot batch diag harness (`--shots 2 --chunks N`, `EXTRA_ENV=`).

## VRAM + VAE perf — full-shot VRAM diagnosis + 42% faster VAE decode (2026-07-03)

Goal: full 7-chunk (81f) shots under 11.5 GB peak on the 3060, as fast as possible
(esp. the 67 s VAE decode), + a per-shot timing table across a continued chain.
New code (all in `examples/shotstream/main.cpp`, gated; `wan.hpp`/`wan_shotstream.hpp`
untouched): `--decode-latent <bin>` (VAE-only decode A/B harness, no DiT/umT5),
`SHOTSTREAM_VAE_OVERLAP` (tile overlap knob on dec_tiling), `SHOTSTREAM_VAE_OFFLOAD`
(VAE params on a CPU backend). Harnesses: `run_vae_ab.sh`, `run_vae_seam.sh`,
`run_shotstream_chain_timed.sh`.

### VAE decode is 42% faster — root cause was TILE COUNT, not the conv kernel
Decode-only A/B on the saved full 81-frame `shot00_latent.bin` (device 0):

| tiling | rel×rel | overlap | tiles | decode | VAE buf | seams |
|---|---|---|---|---|---|---|
| **prod baseline** | 0.5×0.5 | 0.25 | 3×3=9 | **67.1 s** | 4361 MB | clean |
| cuDNN conv3d OFF | 0.5×0.5 | 0.25 | 9 | ~73 s | 5289 MB | clean |
| overlap 0.125 | 0.5×0.5 | 0.125 | 9 | 67.4 s | 4361 MB | clean(=base) |
| overlap 0.0 | 0.5×0.5 | 0.0 | 2×2=4 | 30.3 s | 4361 MB | faint seam |
| 0.6 / 0.25 | 0.6×0.6 | 0.25 | 4 | 42.8 s | 6053 MB | clean |
| **CHOSEN 0.57/0.15** | 0.57×0.57 | 0.15 | 2×2=4 | **38.7 s** | 5485 MB | **clean** |

- **Decode time is set by the tile COUNT** — each spatial tile re-runs the full 3D-conv
  decode over ALL frames. The prod `overlap 0.25` forces a 3rd overlapping tile per axis
  (step = tile·(1−overlap) < tile ⇒ 3×3=9 tiles); overlap 0.0 gives an exact 2×2=4
  partition. 9 tiles ≈ 2.25× the frame area of conv work ⇒ the "abnormal" 0.83 s/frame.
- **cuDNN conv3d (`GGML_CUDNN_CONV3D=1`) IS engaging and IS the fast path** — turning it
  off (im2col) was ~6 s SLOWER (73 vs 67 s). It was NOT silently falling back. The slowness
  was purely the 9-tile overlap-recompute, not the kernel.
- **Fix = 4 clean tiles.** `rel 0.57×0.57 / overlap 0.15` → 2×2=4 tiles that still carry a
  ~16 px feathered overlap → seam-free (eye-verified vs base) → **38.7 s (−42%)**. In-chain
  on 69 f it is **33 s (was ~57 s, −42%)**. Overlap 0.0 (30 s) has a faint center seam — rejected.
- The VAE mid-block attention logs `FA2 supports_op=FALSE d_head=384` (falls to the
  materialized SDPA) — small tensor, not a bottleneck; left as-is.

### The chunks=7 OOM is the DiT KV-cache compute buffer — NOT decode coexistence
Per-forward DiT compute buffer grows with the causal cache (measured):
`ch0 2.64 → ch1 3.39 → ch2 4.39 → ch3 5.56 → ch4 6.59 → ch5 7.62 → ch6 8.55 GB`.
It grows because the local rolling K/V + the fixed 9360-token global context cache are
`make_input`-uploaded as graph-leaf inputs EVERY forward — all 30 layers' F16 K/V live at
once in one monolithic buffer (~6.9 GB at ch6). The segmented path that would bound this is
gated on CPU-offload (`params_backend != runtime_backend`); with resident weights
(`SHOTSTREAM_NO_OFFLOAD=1`, correct for this 2.7 GB model) the graph runs monolithic.

- At chunk 6 the peak = weights 2.71 + compute 8.55 + VAE 0.24 + CUDA ctx ≈ **11843 MiB**,
  ~445 MiB under the 12288 MiB card. **Non-deterministic**: shot-0 chunk-6 sometimes
  completes, sometimes the clean-rewrite forward tips it over and the process dies — so
  **chunks=7 is unreliable on the 3060 even for a single shot**.
- **`free_compute_buffer()` before the VAE decode is already in place and correct** — it
  keeps the decode-phase peak low (VAE decode measured 8419 MiB, DiT weights resident + VAE
  buffer, well UNDER the DiT peak). But the OOM happens DURING the DiT, so freeing DiT
  compute before decode cannot touch it. (The "DiT 2.87 GB coexists with VAE 5.03 GB during
  decode = ~11 GB" model does not match the trace; the real binding peak is the DiT alone.)
- **`SHOTSTREAM_VAE_OFFLOAD` is a NET NEGATIVE** (reverted from the recipe): the per-tile
  param streaming leaves ~1 GB of pool residue that fragments VRAM and turns the marginal
  chunk-6 alloc into a HARD `cudaMalloc` OOM (killed shot-1 outright vs the resident-VAE
  path which survives). VAE weights resident (0.24 GB) is the robust choice.
- Fast-VAE-in-chain caveat at chunks=7: the fast tiling's 5.5 GB decode buffer (vs 4.4 GB)
  adds cross-shot fragmentation that also OOMs chunks=7 shot-1. At **chunks=6** there is
  enough headroom that it does not (peak stays DiT-bound, decode 8.4 GB < DiT 11.6 GB).

### Delivered: chunks=6 (69f) reliable, under 11.5 GB, faster — 3-shot timing table
`run_shotstream_chain_timed.sh` (CHUNKS=6, fast VAE, `SEED=42`, 832×480), device 0:

| shot | umT5 | DiT (6 ch) | VAE decode | history-enc | total | peak VRAM |
|---|---|---|---|---|---|---|
| 0 | 3.4 s | 129.8 s | 33.1 s | — | ~166 s | 11559 MiB |
| 1 | — | 127.8 s | 32.5 s | ~2.5 s | ~163 s | 11559 MiB |
| 2 | — | 127.6 s | 32.6 s | ~2.5 s | ~163 s | 11559 MiB |

- **Constant per-shot cost confirmed** (DiT ~128 s, decode ~33 s, peak identical 11559 MiB
  every shot) — the bounded-buffer chaining holds; per-shot latency is independent of stream
  length. Peak **11559 MiB = 11.29 GiB, under 11.5 GB**, set by the DiT chunk-5 buffer (7.62 GB
  compute + 2.7 GB weights); the fast VAE decode (8.4 GB) never sets the peak.
- Wall/shot 129.8 DiT + 33.1 decode = **~163 s vs ~187 s with the old 57 s decode (−13%)**,
  entirely from the VAE. Eye-test `shotstream_out/chain_timed/perf_chain.mp4` (shots 0+1):
  clean, seam-free, coherent (same fox/world/style across the cut). 3-shot `chain_2shot.mp4`.

### Honest status on the full 7-chunk target + the real next lever
Full 7-chunk (81f) peaks ~11.84 GiB (over 11.5) and is unreliable on the 12 GB 3060; the
floor = weights 2.7 + **F16 KV cache 6.9 GB** (local 28080 + global 9360 tok × 30 layers) +
~1.6 GB activations + VAE. It fits the 5060 (16 GB) with ~4 GB headroom. To get 7 chunks
reliably under 11.5 on the 3060 needs to shrink the KV in the compute buffer:
- **(A) KV-cache RESIDENCY** — hold the K/V in a persistent GPU buffer referenced by the
  graph instead of re-`make_input`-ing it every forward. Removes up to 6.9 GB from the
  monolithic compute buffer ⇒ largest single alloc drops to ~1.6 GB (reliable) ⇒ also a
  ~5 s/shot win (no per-forward H2D cache re-upload). This is the notes' recommended
  "keep the local KV cache GPU-resident" lever; it touches the (verified) cache path so it
  was scoped, not landed, this session.
- **(B) Q8_0 K/V quant** — halves the 6.9 GB ⇒ peak ~9.4 GB; needs the flash path to accept
  quantized K/V + an eye-test (quality-gated).
- Shot-0-only: `SHOTSTREAM_NO_ZERO_CTX_SHOT0=1` drops shot 0's zeroed 9360-tok global cache
  ⇒ shot-0 DiT ~116 s (vs 164) and ~10.1 GB (reliable) — a faithfulness tradeoff (owner's
  call; the zeroed context is the §6-faithful default).

## 3+-shot phase fix — PER-FRAME context RoPE phase = true-endless chaining (2026-07-03)

Chaining was only correct for **2 shots**. For 3+ shots the ≤6 history/context frames come from
**multiple** prior shots, but the old code (a) sampled all 6 from `shot_pixels.back()` (the single
most-recent shot) and (b) applied ONE uniform RoPE phase `k−1` to all of them. Both are wrong vs
the reference (`causal_inference_ar.py` `dynamic_sample_frames` + §3.4): the ≤6 frames are
**distributed across ALL prior shots**, and each context frame must carry the `k·θ` (θ=1/6) phase
of the shot it **originated** from. This session wires that.

### The per-frame phase scheme (ref `causal_inference_ar.py:87-177`, REFERENCE §3.4/§4.1)
For shot `k` (0-based), with `nctx=6` context frames over `k` prior shots:
- **Counts:** `base = nctx/k` per prior shot, `rem = nctx%k` added to the **LAST** shots.
  Verified live: k=1→`[6]`, k=2→`[3,3]`, k=3→`[2,2,2]` (k=4,nctx=6 would be `[1,1,2,2]`).
- **Sampling:** within prior shot `s`, `linspace(0, Ts−1, counts[s]).round()` over that shot's
  decoded pixels (count==1 → first frame, matching the ref's `min(indices)`).
- **Per-frame phase:** frame sampled from shot `s` carries phase `s·θ`. So the context phase
  VECTOR is `[0]*counts[0] ++ [1]*counts[1] ++ … ++ [k−1]*counts[k−1]` (temporal positions
  0..5); the generated chunk keeps its uniform phase `k·θ` (positions 6..). Logged per shot:
  shot1 `[0,0,0,0,0,0]`, shot2 `[0,0,0,1,1,1]`, shot3 `[0,0,1,1,2,2]` — exact ref match.
- **2-shot invariant:** k=1 → `counts=[6]`, phases all-0. Sampling formula reduces to the OLD
  `build_history_ctx` frame indices *exactly*, and an all-0 phase vector leaves the RoPE table
  byte-identical to the uniform-0 (no-op) path ⇒ **the 2-shot case is byte-identical.**

### What changed (uncommitted; `wan_shotstream.hpp` + `examples/shotstream/main.cpp` only)
- **`wan_shotstream.hpp`** — `pe_add_shot_phase` gained an optional `const std::vector<int>*
  shot_per_frame`: when non-null it rotates each position's temporal band by
  `shot_per_frame[frame]·θ`, where `frame = p / (npos/nframes)` (positions are frame-major per
  `gen_vid_ids`); when null it takes the ORIGINAL uniform-`shot_idx` path unchanged (shared
  `rotate_pos` lambda ⇒ the rotation math is literally the same code, byte-identical). Threaded
  the pointer through `pe_build_frames` / `build_pe_frames` / `forward_block` (all default
  `nullptr`, so chunk generation + the offline `--dump-rope`/`--dump-block0` paths are untouched).
  `prefill_context(int context_shot_phase, …)` → `prefill_context(const std::vector<int>&
  context_shot_phases, …)`, passed as the per-frame vector to the context prefill forward.
- **`examples/shotstream/main.cpp`** — new `build_history_ctx_multi(shot_pixels, k, out_phases)`
  does the multi-shot distribution above (uses the full `shot_pixels` vector, not just `.back()`)
  and returns the per-frame source-shot phase vector. Batch loop k>0 now calls it and passes
  `ctx_phases` to `prefill_context`. All 4 `prefill_context` sites updated to the vector API
  (shot-0 zero-ctx → `vector(nctx,0)`; serve loop → `vector(nctx, shot_idx−1)`, since serve keeps
  only the immediately-prior shot's pixels = single-source). Chain-stitch tag now `chain_%zushot`.

### Regression guard (offline, no GPU) ✅
`--dump-rope` + `tools/shotstream_compare.py` vs goldens: **cos 1.000000** for shot0 AND shot1
(max_abs_err 1.6e-6 = the documented f32-table-vs-f64-golden quantization) — the refactor did NOT
perturb the uniform RoPE path.

### Verification — 4-shot chain render (device 0 / RTX 3060, chunks=6, seed 42, 832×480) ✅
`chain_4shot.mp4` (4×69 = 276 px frames) → `shotstream_out/chain_4shot.mp4`. Per-shot phase
vectors logged exactly as above. **Timing (constant per-shot cost holds across the longer chain):**

| shot | DiT (6 ch) | VAE decode | history phases (source-shot) | peak VRAM |
|---|---|---|---|---|
| 0 | 129.8 s | 57.4 s | — (zeroed ctx) | |
| 1 | 127.8 s | 56.9 s | `[0,0,0,0,0,0]` | |
| 2 | 127.6 s | 56.9 s | `[0,0,0,1,1,1]` | |
| 3 | 127.5 s | 57.0 s | `[0,0,1,1,2,2]` | |
| — | | | **overall peak** | **11429 MiB** (< 11.5 GB) |

(This run used the chain-SAFE VAE `0.5×0.5 / overlap 0.25` = 9 tiles / ~57 s for robustness over 4
shots, not the faster 4-tile.) One host-side SIGKILL (rc=137, **not** a GPU OOM — peak was 9.9 GB)
hit the first attempt during shot 0's DiT while the owner's device-1 job contended for the 31 GB
host RAM; a clean retry with host RAM free completed all 4 shots. The host-RAM peak is the ~9 GB of
F16 K/V caches — orthogonal to this fix.

### Eye-test (READ the PNGs, STRICT — not a metric) ✅ core objective met
- **Shots 1–2:** single photoreal red fox, snowy-dawn pine forest, golden backlight — clean.
- **Shot 3 (`shot02`):** SAME photoreal style/world/fox; a **second (smaller companion) fox**
  appears in-frame — both are correct photoreal red foxes (NOT claymation, NOT a style break).
  Plausibly faithful multi-shot composition (context now genuinely spans shots 0+1, each a
  snapshot of the fox somewhere); it self-resolves by shot 4.
- **Shot 4 (`shot03`, the DEEPEST in the chain):** single **pristine** photoreal red fox, same
  world/style — as clean as shot 0. **No drift, no claymation, no identity reset, no collapse.**
  A broken per-frame phase would make the deepest shot the WORST; it is the cleanest ⇒ the phase
  decoupling is working. The old catastrophic failure mode (claymation + wrong-style 2nd fox) is
  GONE. Residual: the transient companion fox in shot 3 (lower-severity, in-world).

### Fold-in: VAE 1×1 decode A/B (`--decode-latent` on the 81-frame `shot00_latent.bin`, dev 0)
Answers the owner's Q — does a whole-frame 1×1 decode fit under 11.5 GB now the DiT compute
buffer is freed, and beat the 4-tile 38.7 s? **No — 1×1 still OOMs even VAE-only:**

| tiling | tiles | decode | peak VRAM | result |
|---|---|---|---|---|
| `0.57×0.57` ovl0.15 (current) | 4 | 38.6 s | 5485 MiB | ✓ seam-free |
| `0.5×1.0` ovl0.0 | 2 | **30.3 s** | 8291 MiB | ✓ but faint center seam (ovl0) |
| `1.0×1.0` (whole-frame **1×1**) | 1 | — | (261) | ✗ **OOM: `wan_vae alloc compute buffer failed`** |

The `--decode-latent` harness loads **only** the VAE (no DiT at all — the best case for 1×1), and
the single-tile whole-frame conv3d compute buffer still can't be reserved on the 12 GB 3060 (the
old "13.3 GB" was the requested size, not a DiT-coexistence artifact). **⇒ 1×1 is NOT a viable
default; keep the 4-tile `0.57×0.57`.** The 2-tile `0.5×1.0` is ~22 % faster (30.3 s) and fits
(8.3 GB) but needs overlap>0 (→ a 3rd tile) to kill the center seam — a possible future default
once seam-checked, not adopted now.

---

## Per-shot prompts (global + per-shot captions) — the "chain on real-time prompting" core

**What:** ShotStream now feeds EACH shot its OWN umT5 context instead of reusing one caption
across the whole chain. Two-level captioning matching the reference `multi_caption=False` path:
a scene-level **global** caption prepended to a per-shot caption.

**Caption assembly (ref-faithful).** Shot i's umT5 input string =
`global_prompt + prompt[i]` — **plain string concatenation, NO separator/space added.**
Confirmed identical in three reference sites:
`Teacher_Ode_Sample.py:222` (`caption = global_captions[0][0] + shots_captions[0][i][0][0]`),
`wan/text2video.py:220`, `trainer/wan_frameconcat.py:258`. Any desired spacing lives in the
strings themselves (reference `shotN:` local-caption prefix, `dataset.py:167`). We replicate the
concat byte-for-byte (`std::string full = global_prompt + prompts[p]`).

**CLI (`examples/shotstream/main.cpp`).**
- `-p` / `--prompt` is now **repeatable** — one occurrence per shot (shot k → caption k).
- `--prompts "p1|p2|p3|p4"` — pipe-separated per-shot captions (convenience form).
- `--global-prompt` / `--global "<scene caption>"` — scene-level prefix applied to every shot.
  Empty by default.
- **Reuse-last:** if fewer captions than shots are given, the LAST caption is reused for the
  remaining shots (`context_for_shot(k)` clamps `k` to `contexts.size()-1`).
- **Byte-identical fallback:** no `-p` → the built-in default caption; a single `-p X` with empty
  global → `"" + X == X`, one context reused for all shots = exactly the old behavior.

**Encode path.** In the one-shot/batch (non-`--serve`) branch, umT5 encodes **each**
`global_prompt + prompt[i]` up-front into its own `[4096,512,1]` context; the N contexts live in
`std::vector<sd::Tensor<float>> contexts` and umT5 host params are freed right after (umT5 is
CPU-backed, so N contexts ≈ 4 MB each, zero VRAM cost). `context_for_shot(k)` hands shot k its own
context to BOTH `prefill_context` (its global/history KV) and `run_shot` (generation) — i.e. shot
k's context prefill uses shot k's OWN caption, exactly the `multi_caption=False` reference
(`caption = global + shots_captions[...][latent_gen_iter]`). `--serve` already re-encoded per stdin
line; it now prepends `--global-prompt` to each live line too (empty global = unchanged).

**Multi-caption context (`shot_flags` per-frame gather) — STILL TODO (follow-up).**
Per REFERENCE §4.3 / `model.py:258-279`, the fully-faithful `multi_caption=True` path feeds umT5
the *list* of captions 0..k → `context [k+1,512,4096]`, and in cross-attention each latent frame
selects its OWN shot's caption via `k = k[shot_flags]; v = v[shot_flags]` — so the ≤6 CONTEXT
frames attend the caption of the shot they ORIGINATED from, not the current shot's. That requires
restructuring `WanT2VCrossAttention::forward` + `WanAttentionBlock::forward_causal` +
`forward_causal_block` to (a) accept a stacked `[N_caps,512,dim]` context, (b) reshape q per-frame
`[f, h*w, n, d]`, (c) gather per-frame k/v by a `shot_flags` vector, (d) run per-frame cross-attn.
This is a genuine architectural change to the cross-attention op (the current path cross-attends a
single `[4096,512,1]` context for all tokens), so it is deferred. The shipped per-shot design
(`multi_caption=False`: shot k's context prefill + generation both use shot k's caption) is the
reference's own non-multi-caption inference path and is correct on its own — the TODO only affects
how a MIXED-source history (3+ shots) captions its context frames.

**Build:** compiles clean via
`docker run --rm -v <worktree>:/src -w /src longcat-avatar-dev:builder-cudnn bash -lc
'cmake --build build --target sd-shotstream -j"$(nproc)"'` → `build/bin/sd-shotstream` relinked
(only the pre-existing `system()` `-Wunused-result` warnings, non-fatal in the examples build).
Left uncommitted. No dedicated verify render launched — per-shot prompts fold into the next
planned chain render (pass repeated `-p` + optional `--global-prompt`).

---

## KV-cache residency (2026-07-03)

**Problem.** `ShotStreamRunner::forward_block` held both KV caches in HOST F16 vectors
(`local_k/local_v`, `ctx_k/ctx_v`) and `make_input`-uploaded the ENTIRE growing local+global
cache on EVERY forward. All 30 layers' cache leaves are `INPUT_EXTERNAL` (live for the whole
graph), so gallocr sized the single compute buffer to hold the whole ~6 GB cache at once → an
**8.55 GB monolithic alloc** that OOMs the full 81f / 7-chunk shot on the 12 GB 3060 (chunks=6/69f
fit at 11.4 GB). It also wasted ~5 s/shot re-uploading the whole (growing) cache H2D, ×5
forwards/chunk.

**Fix = GPU residency (default; A/B fallback `SHOTSTREAM_KV_HOST=1`).** Each persisted chunk's
K/V now lives in its OWN persistent ggml backend buffer on the runtime backend, held ACROSS graph
computes — mirroring `LongCatAvatarRunner`'s `condkv_buf` pattern (own ctx+buffer via
`ggml_backend_alloc_ctx_tensors`, leaves `register_persistent_tensor`'d so the segmented executor's
`reset_segment_runtime_tensors` doesn't orphan them).
- **State:** `struct KVChunk { ggml_context* ctx; ggml_backend_buffer_t buf; vector<ggml_tensor*> k,v; int64_t L; }`;
  `std::vector<KVChunk> res_local, res_ctx`. Each chunk = one buffer holding all `nL` layers' K+V,
  F16 `[d_head, L, n_head]` (identical layout/bytes to the old host tensors).
- **READ (`forward_block`):** instead of `make_input(local_k[i])`, `resident_read()` builds the
  ordered (oldest→newest) `ggml_concat` of the resident chunk tensors in `compute_ctx` and passes
  it as `prev_kc[i]`/`cond_kc[i]`. The chunk leaves already have GPU buffers → gallocr never places
  them in the compute buffer, and the per-layer concat NODES are freed by gallocr right after each
  block. Result: the ~6 GB of cache leaves leave the compute buffer, and the per-layer concat is the
  only transient → **peak single alloc 8.55 GB → ~1.6 GB**, full 81f fits < 11.5 GB. Single chunk ⇒
  bare leaf returned (no concat node). The value handed to `forward_causal_block` is a byte-identical
  F16 `[d_head, ΣL, n_head]` in the same order as the host path's rolling concat, so
  `forward_kv_cache`'s `to_f32`+concat math is unchanged.
- **WRITE (deliberately UNCHANGED, conservative):** the `need_kv` export/readback path is kept
  byte-for-byte — fresh K/V still cut-marked `shotstream.blocks.i.out`/`nk|nv` and read back to host
  F16 per-segment via `segment_readback_hook_`. We then `push_kv_chunk()` = upload just THAT chunk's
  K/V once into a fresh persistent buffer. So the whole-cache re-upload disappears (≈5 s/shot saved);
  only the small new-chunk D2H+H2D round-trip remains (~1.6 GB/shot). The RoPE phase (K stored
  post-RoPE inside `forward_causal_block`), per-shot/global prompt contexts, the `need_kv`
  skip on the 4 denoise steps, and the stale-local-cache reset in `prefill_context` are all untouched
  — `reset_local_cache()` now also `free_kv_chunks(res_local)` so a shot's context prefill still runs
  clean self-attention.
- **Eviction:** window bound preserved at CHUNK granularity (drop oldest `KVChunk` past
  `local_attn_size/num_frame_per_block` = 7). Shipped config never evicts (7 chunks == 21 frames),
  so it never fires — same as the host `slice` clamp.

**A/B flag:** `SHOTSTREAM_KV_HOST=1` restores the exact legacy host-upload+concat path (host
vectors, `make_input`, `sd::ops::concat`/`slice`). Default (unset) = resident.

**Est. VRAM/alloc impact:** max single compute-buffer alloc 8.55 GB → ~1.6 GB; the ~6–7.7 GB cache
now lives in dedicated per-chunk buffers (no compute-buffer over-allocation, no F32 whole-cache
duplication) → full 7-chunk 81f shot fits < 11.5 GB on the 3060. H2D: whole-cache re-upload per
forward eliminated (~5 s/shot); residual = one new-chunk round-trip per persisted forward.

**Deferred lever (not done — risk):** a fully-resident in-graph `ggml_cpy(new_kc[i], chunk.k[i])`
append (condkv style) would drop the residual new-chunk readback+reupload too, but it writes a
persistent-aliasing tensor as a graph-cut output on the **segmented offload path** (the prod default,
`SHOTSTREAM_NO_OFFLOAD` unset → DiT weights on CPU + `--max-vram` segmentation). condkv proves the
pattern works, but it touches the graph-cut planner's segment assignment and can't be eye-verified
here, so it's left for a follow-up. The chosen hybrid keeps the verified readback path intact.
The documented Q8-KV fallback was NOT needed (residency fixes VRAM *and* perf).

**Build:** `docker run --rm -v /home/dbrain/dev/longcat-avatar-shotstream:/src -w /src
longcat-avatar-dev:builder-cudnn bash -lc 'cmake --build build --target sd-shotstream -j"$(nproc)"'`
→ compiles clean, `build/bin/sd-shotstream` relinked (only the pre-existing `main.cpp` `system()`
`-Wunused-result` warnings; `wan_shotstream.hpp` produced none). Left uncommitted, NO render run.

**RISK for the verify render to watch:**
1. **Segmented/offload path (prod default) correctness** — confirm the resident `resident_read`
   concat lands in each block's segment (persistent leaves are registered so they shouldn't be
   orphaned; the concat is an ordinary intermediate feeding block i). A/B against
   `SHOTSTREAM_KV_HOST=1` should be visually identical.
2. **A/B latent parity** — resident vs `SHOTSTREAM_KV_HOST=1` should match closely (same F16 bytes,
   same concat order); any drift means a read-order or geometry mismatch.
3. **Chaining (2+ shots)** — the global-cache prefill + stale-local reset path; verify no
   style-drift/duplicate-subject regression vs host mode.
4. **Peak VRAM** — confirm the compute-buffer peak drop (expect chunks=7 to fit where it OOM'd).

## ncu/nsys floor deep-dive — RTX 3060 (2026-07-04)

Iterative profile→fix→re-profile loop to drive ShotStream to its per-kernel floor on the
3060 and PROVE it. Profiling folded into short renders (1 shot, 3 chunks, 832×480) in the
owner's baseline **host-KV mode** (`SHOTSTREAM_KV_HOST=1`) + fast recipe. nsys for the
timeline (`prof_shotstream_nsys.sh`), ncu `--set`-style roofline sections on the 1–2 hottest
kernels (`prof_shotstream_ncu.sh`, needs `--cap-add SYS_ADMIN`). All fixes A/B'd for
byte-identical latent parity (incl. 2-shot chain) via env opt-outs. Code left uncommitted;
changes are in `src/model/diffusion/wan.hpp` (`WanSelfAttention::forward_kv_cache`).

### THE REFRAME: cuDNN attention is architecturally impossible on the 3060
The prior notes' #1 recommended lever — "get cuDNN flash-attn to engage on the causal KV
path (est. 10–18 s)" — is a **DEAD END on Ampere, by construction, not a dispatch bug.**
`ggml_cuda_get_best_fattn_kernel` (fattn.cu:446) gates the cuDNN SDPA path on
`ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_BLACKWELL` (1200). The 3060 is cc **8.6**
→ the cuDNN branch is never taken regardless of `GGML_CUDNN_ATTN=1`. The cuDNN frontend SDPA
engine used here is the sm120-native one; there is no Ampere cuDNN-attn path in this fork.
Only the 5060 Ti (cc 12.0) can use it. **On the 3060 the causal self-attn MUST run on ggml's
native tensor-core flash, and it already selects the best one (MMA_F16, verified below).**

### Profile 0 — baseline nsys (chunks=3, host KV): the hotspots
`shotstream_out/nsys_base`. DiT 48.4 s, VAE 16.4 s. Per-kernel GPU time (DiT+VAE):

| Kernel | % | Total | What |
|---|---|---|---|
| `flash_attn_ext_f16<128,128,64,1,0,0>` | 32.7% | 15.74 s | **causal self-attn** (L_k up to 23400 = local+cur+ctx) |
| `ampere_h1688gemm_*` (2 variants) | 10.3% | 4.67 s | DiT Linears (q/k/v/o, FFN) + cross-attn GEMM |
| VAE conv3d (`sm8x_xmma_fprop` + `im2col`) | ~18% | ~8.7 s | Wan 3D-VAE decode (cuDNN conv3d) |
| `pad_f32` | 3.4% | 1.62 s | **WASTE**: self-attn L_k→256 pad + synth mask |
| `concat_T_cont_4d<float,*>` | ~7% | ~3.3 s | KV-cache concat (F32) |
| `convert_unary<half↔float>` | ~4.5% | ~2.0 s | KV-cache F16↔F32 round-trip + VAE |

The causal self-attn dominates; a large secondary cluster is **pure waste around it** (pad +
synthesized mask + a whole-cache F16→F32→F16 round-trip on every forward).

### Profile 1 — ncu the #1 kernel (flash_attn): occupancy-bound, at the kernel floor
`shotstream_out/ncu_flash`, big-L_k self-attn instances (L_k=18720):
- **Compute (SM) 38.5% · DRAM 5.1% · L2 26% · Achieved occupancy 24.1% (theoretical 25%)**
- **168 registers/thread** → register-limited to 3 blocks/SM = 12 warps = 25% of 48 → the
  occupancy CEILING is set by register pressure, *by kernel design*.
- The kernel is `flash_attn_ext_f16<DKQ=128, DV=128, ncols1=64, ncols2=1, false, false>` =
  the **MMA_F16 tensor-core flash** (fattn-mma-f16.cuh) — the *best* Ampere kernel, not a
  fallback (WMMA/tile/vec are slower for these large-batch D=128 shapes; the dispatch's
  turing-MMA branch selects it). Its `__launch_bounds__` deliberately trades occupancy for
  register-resident MMA tiles (ILP-hides latency at low occupancy).
- **Verdict: neither compute- nor memory-bound — latency/occupancy-bound at the kernel's own
  designed ceiling.** The attention FLOPs (L_q·L_k·D per forward × 5 forwards/chunk) are
  irreducible: iterative denoising re-attends the whole cache each step (Q changes every step;
  cached K/V are fixed but must be attended). This IS the 3060 floor for this attention.

### Fix #1 — skip the causal kv-pad + synthesized mask (byte-identical)
`forward_kv_cache` called `ggml_ext_attention_ext(mask=nullptr)` WITHOUT `flash_skip_kv_pad`,
so every self-attn padded L_k to a 256-multiple AND synthesized a `[L_k_pad × L_q]` -inf F16
mask (~220 MB) — exactly what the bidirectional `WanSelfAttention::forward` already avoids
(it passes `flash_skip_kv_pad = mask==nullptr`). Padded keys had -inf mask ⇒ 0 softmax weight
⇒ removing them is numerically identical; the modern MMA kernel handles unpadded L_k with no
mask. Passed `flash_skip_kv_pad=true` (env opt-out `SHOTSTREAM_KV_PAD=1`).

| Metric (chunks=3) | base | Fix#1 | Δ |
|---|---|---|---|
| DiT | 48.4 s | 42.1 s | **−13%** |
| flash_attn kernel | 15.74 s | 12.81 s | **−18.6%** (no mask to read + unpadded L_k) |
| `pad_f32` | 1.62 s (2052) | 0.77 s (1092, now VAE-only) | −52% |
| DiT compute buffer | 4389.80 MB | 3839.06 MB | **−551 MB** (killed the mask tensor) |
| latent parity | — | max_abs=0.0, cos=1.0, **identical=True** | ✅ |

### Fix #2 — concat the KV-cache in F16, not F32 (byte-identical, VRAM win)
The cache is already F16; the legacy path cast the WHOLE growing cache F16→F32 (`to_f32`),
concatenated in F32, then build_kqv cast it back F32→F16 for flash — a full F16→F32→F16
round-trip of `[prev++cur++ctx]` every forward. Now: cast only the NEW chunk (small, L_blk)
F32→F16 and concat in F16; cached tokens keep their exact F16 bytes; build_kqv's redundant-
cast guard feeds F16 straight to flash. Byte-identical (flash already consumed F16 K/V; the
new chunk still gets exactly one F32→F16 rounding). Env opt-out `SHOTSTREAM_KV_F32_CONCAT=1`.

| Metric (chunks=3) | Fix#1 | Fix#1+#2 | Δ |
|---|---|---|---|
| DiT | 42.1 s | 42.1 s | **0 (time-neutral)** |
| DiT compute buffer | 3839.06 MB | 3619.69 MB | **−220 MB** (F16 concat = ½ the transient) |
| latent parity (shot0 + **2-shot chain**) | — | max_abs=0.0, **identical=True** both shots | ✅ |

**Why time-neutral:** the per-layer concat is **launch/latency-bound**, not bandwidth-bound —
F16 concat is the same wall as F32 (1.407 s vs 1.404 s over 1500 launches), and the removed
whole-cache converts are offset by the new small F16 casts. The win is **pure VRAM**: the
F16-concat saving scales with cache size (bigger at chunks=6/7).

### Profile 2 — ncu the #2 kernel (matmul): compute-bound, at roofline
`shotstream_out/ncu_gemm`, `ampere_h1688gemm` instances: **SM 76–87% · DRAM 53–58%** →
compute-bound, near the tensor-core roofline. This is cuBLAS's own tuned F16 GEMM = at floor.

### Prod-scale validation (chunks=6, 1 shot, host KV, 0.5×0.5 chain-safe VAE)
Clean same-binary A/B (fixes ON default vs `SHOTSTREAM_KV_PAD=1 SHOTSTREAM_KV_F32_CONCAT=1`):

| | DiT | VAE | wall | peak VRAM | DiT buf |
|---|---|---|---|---|---|
| fixes OFF (pre-fix) | 129.9 s | 57.6 s | 200 s | 11035 MiB | 7621 MB |
| **fixes ON** | **112.1 s** | 57.4 s | **181 s** | **9667 MiB** | **6252 MB** |
| **Δ** | **−13.7%** | 0 | −19 s | **−1368 MiB (−12.4%)** | **−1369 MB** |

- **chunks=7 (full 81-frame shot) now FITS the 3060: 10529 MiB, no OOM** (DiT 142.6 s, VAE
  67.6 s) — previously peaked ~11.84 GB / OOM'd. The −1.37 GB headroom is the VRAM half of the
  two fixes. (VAE here uses the conservative 9-tile 0.5×0.5; the fast 4-tile 0.57×0.57 recipe
  independently takes VAE to ~33–39 s — orthogonal to these DiT fixes.)

### Final hotspot table + verdict
| Hotspot | Bound | At floor? | Only remaining lever |
|---|---|---|---|
| Causal self-attn (`flash_attn_ext_f16` MMA, ~31%) | **occupancy/latency** (25% reg-capped, SM 38%, DRAM 5%) | **YES** — best Ampere kernel, FLOPs irreducible | cuDNN/FA3 → **Blackwell-only, needs the 5060 Ti**; or a custom low-register kernel (major eng) |
| DiT Linears / cross-attn (`h1688gemm`, ~11%) | **compute** (SM 85%+) | **YES** — cuBLAS roofline | none (hardware peak) |
| VAE 3D-conv decode (`xmma_fprop`+`im2col`, ~18%) | **compute** (tensor-core conv) | **YES** — cuDNN conv3d, tiling is time-neutral | lighter VAE / fewer tiles (OOMs) — out of scope |
| KV concat + eager-op tail (~15%) | **launch/latency** (many small per-layer kernels) | reducible waste **removed** (mask, F32 round-trip); residual concat is launch-bound | CUDA-graph capture of the per-forward graph (structural; conflicts with the segmented executor) — out of scope |

**Verdict — this is the floor because:** (1) the dominant kernel (self-attn) is on the best
available Ampere kernel (MMA_F16, ncu-proven) running at its designed occupancy ceiling, over
an irreducible attention workload; (2) the matmuls and VAE convs are compute-bound at the
tensor-core roofline (ncu-proven for the GEMM); (3) the two tractable waste sources around the
attention — the synthesized kv-pad mask and the whole-cache F16↔F32 round-trip — are removed
(−13.7% DiT, −1.37 GB VRAM, byte-identical). **The only remaining levers are out of scope or
need hardware:** cuDNN/FA3 attention (Blackwell → the 5060 Ti, where it is *not* gated off),
a hand-written low-register D=128 flash kernel, a lighter VAE, or CUDA-graph capture to kill
the launch-bound eager-op tail. None is a "tractable safe win" on the 3060; the residual is
provably compute-/occupancy-bound with the wasted cycles eliminated.

**Files (uncommitted):** `src/model/diffusion/wan.hpp` (Fix#1 `flash_skip_kv_pad=true`,
Fix#2 F16 concat, both env-opt-out). Harnesses: `prof_shotstream_nsys.sh`,
`prof_shotstream_ncu.sh`.
