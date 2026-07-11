# The +1.7GB continuation context reserve — Codex continuation brief (2026-07-07)

**Status: NOT the cuDNN plans (disproven with hard data). It's a uniform ~1.4-1.7GB context-level
reserve** that appears in seg-2 of a chained render and is untouchable by every cuDNN lever AND
every ggml/CUDA pool trim. This supersedes CUDNN-STUCK-PLANS.md (that theory is dead — see below).
Goal: localize the owner of the reserve (needs a real memory profiler, not env flags) and reclaim
or prevent it. Cross-project: any warm worker that streams a big offloaded model across phases.

## The exact measurements (LTX-2.3 2-seg 97f 1080p continuation, LONGCAT_VRAM_BREAKDOWN=1)
Per-phase `driver_used` (= total 15889 − cudaMemGetInfo free), seg-1 (clean) vs seg-2:
```
                  seg-1    seg-2    delta   notes
base   (used)     7537     8941     +1404   same base size; reserve already present at seg-2's FIRST phase
refine (16f)      10513    12977    +2464   = frames (+770) + reserve (+1694)
refine (13f*)     10531    12207    +1676   *frame-fix ON: SAME 13 frames, same compute_buf 2765/2767 → pure reserve
decode            11009    12747    +1738   same VAE buffer 6532, same runtime 1385 → pure reserve
```
Attribution at seg-2 refine (13f): `12207 = compute_buf 2765 + resident 5471 + partial 209 + prefetched 209 + UNACCOUNTED 3553`. Seg-1: `10531 = 2767 + 5471 + 418 + UNACCOUNTED 1857`. **The +1696 is entirely in the UNACCOUNTED bucket** (not compute, resident, streaming, or ggml pool — `pool=0` on every line).

Boundary behavior (RUN A monotonic `used`): seg-1 climbs to 10513, drops to ~8941 at the seam (~1572 freed), but **~1404 persists into seg-2's base** and stays for all of seg-2. So the reserve is allocated *during seg-1's execution* and is *not returned* at the chain boundary.

Baseline decomposition of UNACCOUNTED: seg-1 ~798 at base entry (CUDA context + cuDNN handle init) → grows to ~2202 by seg-2 base = **+1404 allocated during seg-1 that never frees.**

## Everything tried — ALL null or crash (this is the key input)
| intervention | commit | result |
|---|---|---|
| clear fe::graph plan caches | ggml 85c61afa | **+0.0 MB** freed |
| cudnnDestroy thread-local handles | ggml 74cf75e6 | **+0.0 MB** (delta logged) |
| cudaMemPoolTrimTo(0) default async mempool | ggml 9193d986 | **+0.0 MB** |
| per-plan build cudaMemGetInfo deltas | ggml 9910454d | **every build +0.0** — plans allocate ~nothing; workspaces ws=0-133MB |
| set_max_workspace_allowed / deselect_workspace_greater_than pre-build | ggml 5b06871e | **no change** to seg-2 |
| GGML_CUDNN_ATTN=0 (native flash-attn) | — | **CRASH** ~78s (compute_buf=384 partial=5887 prefetched=4248) |
| GGML_CUDNN_CONV3D=0 | — | **CRASH** ~149s (same signature) |
| ggml_backend_cuda_trim_pools (VMM pool) | existing | pool=0 already — doesn't see it |
| MAXV / shared-resident knobs | — | orthogonal (DiT weight residency, separate) |

Conclusion: the reserve is **outside** the ggml VMM pool (pool=0), **outside** the CUDA default async mempool (trim +0.0), and **not** the cuDNN plans (build delta +0.0, workspace cap no-op). cuDNN handle destroy freed nothing. Both cuDNN-disable escape hatches crash (their own bug — the graph-cut/streaming mis-plans without cuDNN → the partial=5887 OOM signature).

## Current best hypothesis
**CUDA VMM / raw-cudaMalloc fragmentation from the DiT param-streaming churn.** Seg-1 streams the 16GB offloaded DiT to GPU in chunks (the offload/graph-cut system does many alloc/free cycles). This churn likely fragments the CUDA context's internal heap or leaves VMM reservations that `cudaMemGetInfo` counts as used but no trim returns. It's ~1404MB, allocated during seg-1, persistent. Not cuDNN (every cuDNN lever null; disable crashes for an unrelated reason).

## Concrete next diagnostics (the real ones — env flags are exhausted)
1. **`nsys profile --cuda-memory-usage=true`** on the 2-seg repro → every cudaMalloc/cuMemCreate/cuMemMap with size + call stack. Find the allocation(s) totaling ~1404 that are made in seg-1 and NOT freed by seg-2's base. That call stack IS the owner. (nsys is available; see reference_nsys_container_fix — CUDA-13 builder needs `apt install cuda-nsight-systems-13-0`.)
2. **Instrument the gap directly** in ggml: at each phase reserve, log `used = total - free`, `sum_of_all_ggml_backend_buffer_bytes`, `vmm_pool_reserved`. The residual `used - sum_buffers - vmm_reserved` IS the reserve; watch it step up during seg-1 to pinpoint which op/phase grows it.
3. **`compute-sanitizer --tool memcheck --leak-check full`** (in the devel container) — will it flag ~1404 of leaked/unfreed device allocations across the seam?
4. **Test the streaming-churn hypothesis:** run a 2-seg chain with the DiT NOT offloaded (if it fits a smaller test, or with LONGCAT_SHARED_RESIDENT forcing more resident / less streaming) — if the seg-2 reserve shrinks with less streaming churn, that confirms the allocator-fragmentation origin.
5. **CUDA VMM introspection:** if ggml uses cuMemCreate/cuMemMap (ggml_cuda_pool_vmm), query the reserved-vs-mapped high-water; a `cuMemUnmap`+`cuMemRelease` of the unmapped high-water at the seam may reclaim it (different from cudaMemPoolTrimTo which only hits the stream-ordered pool).

## Reproduce + build
- Engine at LTX_REF (kobbler compose, `LTX_REF`). All the diagnostic commits are on master (ggml + parent). `LONGCAT_VRAM_BREAKDOWN=1` gives the per-phase reserve lines + the per-plan build deltas.
- 2-seg repro: `ltx-denoise-repro/../scratchpad/renders/cont2_1080.json` via koblem `POST :8090/api/v1/ltx-video/generate` (Bearer) or engine-direct `:8096`.
- **Build container:** `nvidia/cuda:13.3.0-devel-ubuntu24.04` + apt `cudnn9-cuda-13 libcudnn9-dev-cuda-13` + cudnn-frontend @ `9782b855ddecefe1646b00bb0cfd9870c381e391` at `/opt/cudnn-frontend/include`. Archs `86;120`, `-DGGML_CUDNN=ON`. (nsys/compute-sanitizer: add the CUDA nsight packages.)
- Key files: streaming/offload + graph-cut in `src/core/ggml_extend.hpp` (the param streaming, partial/prefetch buffers, VMM pool); `src/ggml-cuda/ggml-cuda.cu` (release_cudnn_plans + the pool/VMM allocator); `fattn-cudnn.cu` / `conv3d-cudnn.cu` (the per-plan logs).
- **The frame-fix** (`LTX_REFINE_CONTEXT_FRAMES`, in the tree) reduces the frame/compute part but NOT the reserve — set it to isolate the reserve at constant frames (seg-1 13f vs seg-2 13f = pure +1676).

## Also worth a fix (separate)
The `GGML_CUDNN_ATTN=0` and `GGML_CUDNN_CONV3D=0` paths crash (partial=5887 streaming blowup) — fixing them gives a clean cuDNN-off escape hatch AND would let us definitively confirm/deny cuDNN as the reserve owner.

## HOW TO BUILD, RUN, AND ITERATE (operational playbook)
All of this runs on the GPU box (the 5060 Ti host). Repos:
- engine: `/home/dbrain/dev/longcat-avatar-ltxdenoise` (branch `ltx-denoise-workflow`, pushed to `master`)
- ggml submodule: `.../ggml` (branch `ltx-denoise-workflow` → `master`)
- deploy config: `/home/dbrain/dev/kobbler` (docker-compose.yml + docker/ltx-video/Dockerfile)

**The Docker build clones from git**, so every code change must be committed + pushed before building:
```bash
# 1. edit ggml/src/ggml-cuda/*.cu (or engine src/)
# 2. commit + push ggml:
cd /home/dbrain/dev/longcat-avatar-ltxdenoise/ggml
git add -A && git commit -m "..." && git push origin ltx-denoise-workflow:master
# 3. bump the submodule pointer in the parent + push:
cd /home/dbrain/dev/longcat-avatar-ltxdenoise
git add ggml && git commit -m "bump ggml" && git push origin ltx-denoise-workflow:master
NEW=$(git rev-parse --short HEAD)
# 4. point the deploy at the new SHA (Docker builds at LTX_REF) + push:
cd /home/dbrain/dev/kobbler
sed -i -E "s|LTX_REF: \"\\\$\{LTX_REF:-[a-f0-9]+\}\"|LTX_REF: \"\${LTX_REF:-$NEW}\"|" docker-compose.yml
git add docker-compose.yml && git commit -m "bump LTX_REF" && git push
# 5. build (~25min, includes the ggml rebuild):
docker compose build ltx-video
```

**ENV-PLUMBING GOTCHA (bit us 3×):** a new env var only reaches the container if it's in the
docker-compose.yml `ltx-video` `environment:` block. Inline `VAR=x docker compose up` ONLY works for
vars already interpolated there (e.g. `GGML_CUDA_ALLOC_TRACE: "${LTX_CUDA_ALLOC_TRACE:-}"`). Add new
knobs to that block first, or the run silently uses the default. Verify with:
`docker exec kobbler-ltx-video-1 sh -c 'echo $YOUR_VAR'`.

**Deploy + run the 2-seg repro (~11min):**
```bash
cd /home/dbrain/dev/kobbler
# deploy with the trace on (frame-fix off via N=99 to isolate the reserve at constant frames):
LTX_CUDA_ALLOC_TRACE=1 LTX_REFINE_CONTEXT_FRAMES=99 LTX_VRAM_BREAKDOWN=1 docker compose up -d ltx-video
# wait for health, then submit:
K=<koblem bearer key>   # see below
R=<scratchpad>/renders/cont2_1080.json
JOB=$(curl -s -X POST http://localhost:8090/api/v1/ltx-video/generate \
      -H "Authorization: Bearer $K" -H "Content-Type: application/json" -d @"$R" \
      | python3 -c 'import json,sys;print(json.load(sys.stdin)["job_id"])')
# poll: curl -s http://localhost:8090/api/v1/ltx-video/jobs/$JOB -H "Authorization: Bearer $K"
```
Engine-direct (no auth): `POST http://localhost:8096/ltx/v1/generate` same body; `/v1/admin/unload` to free the GPU child between runs.

**Read the trace:**
```bash
docker logs kobbler-ltx-video-1 --since 14m 2>&1 | grep -aE 'cuda-alloc' | grep -av gemma
```
Correlate with the phase markers (`generate_video …`, `latent spatial upscale`, `predecode`,
`decode`) and the per-phase `[VRAM] ltxav reserve` lines. **What localizes it:** a `cudaMalloc` or
`vmm-map` during SEG-1 with size ~1.4-1.7GB (or several summing to it) that has NO matching
`cudaFree-buffer`/`vmm-unmap` before the seg-2 `base sample` — or a free whose `delta` (free-returned)
is much smaller than its `size` (→ VMM high-water not returned to the driver). If NOTHING large shows
in seg-1 without a matching free, the reserve is **below allocator visibility** (driver/context
internal) → escalate to `nsys profile --cuda-memory-usage=true` or `compute-sanitizer --leak-check`.

**Timings:** full rebuild ~25min (ggml dominates); deploy ~1min; 2-seg 97f repro ~11min (seg-1 ~5min,
seg boundary + seg-2 refine is where the reserve is). For a faster reserve-only check you still need a
real chain (single renders can't reach the 16-latent seg-2 state) — the 2-seg is the minimum.

**Credentials:** koblem bearer key + the exact cont2_1080.json path are in the session scratchpad /
the repo owner. The 2-seg repro is no-audio t2v (a "convertible pulls up to a bar at dusk" prompt).

## RESOLVED (diagnosis) — driver-heap fragmentation from raw-cudaMalloc churn (2026-07-07)
The `GGML_CUDA_ALLOC_TRACE` run answered it. It is **NOT a leak and NOT cuDNN** — it's **CUDA
driver-heap fragmentation** from re-allocating multi-GB buffers via raw `cudaMalloc`.

Evidence (2-seg 97f 1080p, ALLOC_TRACE=1):
- **Allocator is net-0 balanced:** 2090 cudaMalloc = 2090 cudaFree, byte-for-byte (1,821,735 MB each). All 6 vmm-map have matching vmm-unmap. Nothing leaks.
- **The "everything-freed" floor climbs monotonically** — `used` after each phase-boundary vmm-unmap: `1247 → 2027 → 2949 → 3727 MB` (+2480 across 4 phases). Classic fragmentation accumulation.
- **The churn:** 40 distinct sizes cycling — the big graph-cut/refine **compute buffers go through raw cudaMalloc**: `2766 MB ×141`, `3386 MB ×141` (the 13f/16f refine gallocr), interleaved with streaming chunks `209×656, 900×376, 850×184, 418×176`. Only 6 allocations use the VMM pool; the other 2090 are raw cudaMalloc. Mixed-size raw malloc/free interleaving = heap fragmentation.

**Why every prior lever was +0.0:** there is nothing allocated to reclaim — the 1.4-1.7GB is stranded
in the driver's fragmented free-list, invisible to pool trims / handle destroy / mempool trim.

**FIX DIRECTIONS (for Codex):**
1. **Stop re-allocating the big graph-cut compute buffer per segment via raw cudaMalloc.** Reserve the
   max-size compute buffer ONCE (persistent gallocr / a fixed scratch buffer sized to the largest
   graph) and reuse it across all graph-cut segments + phases. The 141× churn of the 2766/3386 MB
   buffer is the dominant fragmenter. Find where the gallocr/compute buffer is allocated per
   graph-cut segment in `src/core/ggml_extend.hpp` (the graph-cut compute path) and make it persistent.
2. **Route large buffers through the VMM pool** (`cuMemCreate`/`cuMemMap`, already used for 6 allocs)
   instead of raw `cudaMalloc` — VMM grows/shrinks contiguously without fragmenting the raw heap.
3. **Streaming chunk pool:** the 209/900/850 MB streaming chunks (656+376+184 events) should come from
   a fixed ring/pool, not fresh cudaMalloc per chunk.
4. **Fallback:** worker-recycle (fresh CUDA context) between chains eliminates fragmentation but reloads
   the model — expensive, last resort.
Verify with the same ALLOC_TRACE run: after the fix, the post-unmap floor should stop climbing and
seg-2 phases should read seg-1's clean levels (~10531 refine, ~11009 decode → peak ~11009, fits ≤11.5).

## UPDATE 2 — fragmentation RULED OUT; it's per-unique-shape cuDNN exec memory (2026-07-07)
The compute-buffer-reuse fix (parent 205eb40 — reuse the graph-cut compute allocator across the
segment loop) **worked on the churn but NOT the reserve**:
```
2766/3386 MB malloc:  141× → 3×      (churn collapsed — good hygiene, KEEP the fix)
total cudaMalloc:     2090 → 1248
post-unmap FLOOR:     1247→2027→2949→3727   IDENTICAL to before
seg-2 peak:           12996                  unchanged
```
**Removing ~840 mallocs moved the floor by 0.** So it is NOT raw-cudaMalloc fragmentation.

New diagnosis: the floor climbs **~800MB per PHASE, deterministically** (+780/+922/+778 across the 4
phases seg1-refine / seg1-decode / seg2-refine / seg2-decode). Four unique large shapes (refine
Lq=26520/32640, decode conv3d N=13/16) → ~800MB × 4 ≈ +2480 = the floor climb. VMM unmaps return their
pool (delta +416/+458/+510/+458), so it's not the pool.

**⇒ It's cuDNN plan EXECUTION memory, allocated at first-compute (NOT at build), cached per shape,
context-level.** This reconciles ALL prior nulls: build-delta +0.0 (allocated at exec, not build);
handle-destroy +0.0 at the boundary (seg-2 re-executes its shapes AFTER, rebuilding the reservation);
every pool/mempool trim null (it's cuDNN-internal, cached, keyed by shape).

**Definitive next tool: `nsys profile --cuda-memory-usage=true`** on the 2-seg repro — it captures the
COMPUTE-TIME allocation (below the cudaMalloc layer GGML_CUDA_ALLOC_TRACE covers) + call stack. Look for
a ~800MB allocation at the first execution of each new SDPA/conv3d shape that is never freed. If it's
cuDNN-internal: the levers are (a) reduce unique shapes (pad/bucket shapes so cuDNN reuses one plan —
e.g. always run refine at a fixed max Lq), (b) a deterministic/limited-memory cuDNN engine config that
doesn't cache exec memory, (c) the crash-fix for GGML_CUDNN_ATTN/CONV3D=0 so cuDNN-off is a real escape.
Keep the compute-buffer-reuse fix regardless (it's correct + reduces churn).

## UPDATE 3 — nsys CONFIRMS: reserve is cuDNN-internal unfreed workspace (2026-07-07)
Full-2-seg nsys capture (builder nsys 2026.2, imported via QdstrmImporter + libdw1 — the recurring
importer bug is just a missing libdw.so.1). Query on CUDA_GPU_MEMORY_USAGE_EVENTS:
- Large allocs (5471/5886/7564 MB) are all **balanced** (alloc count == dealloc count) — no big leak.
- **NO async mempool** (localMemoryPoolSize empty) — rules out pool retention (matches mempool-trim +0.0).
- **Unfreed allocations >100MB: 216MB×2 + 151MB×1 + 108MB×2 = ~799 MB**, and **pc=0 (NO ggml backtrace)**
  → these are allocated *inside cuDNN*, not through our code, and never freed.

**CONCLUSION: the +800/shape reserve is cuDNN-INTERNAL cached workspace (108-216MB chunks).** cuDNN
allocates these directly (no ggml stack) at first execution of each new SDPA/conv3d shape and never
releases them — which is why every ggml/pool/handle/mempool lever returned +0.0.

**Fix levers (cuDNN-internal, for Codex):**
1. Correlate the 108/151/216 MB chunk allocations' timestamps with the cuDNN API timeline (correlationId
   in CUDA_GPU_MEMORY_USAGE_EVENTS → CUPTI_ACTIVITY_KIND_RUNTIME) to pin each chunk to SDPA vs conv3d.
2. Shape bucketing/padding so cuDNN reuses ONE plan+workspace across seg-1/seg-2 (fixed max Lq for the
   refine, fixed conv3d shapes) — fewer unique shapes = fewer cached chunks. Likely the cleanest fix.
3. cuDNN heuristic/engine that doesn't cache exec workspace, or an explicit workspace-provided execution
   path (provide the workspace from a reused ggml buffer so cuDNN doesn't allocate+cache its own).
4. Fix the GGML_CUDNN_ATTN/CONV3D=0 crash → clean cuDNN-off escape (confirms + is a fallback).
Standard nsys harness (with the libdw1 fix baked in): ltx-denoise-repro/nsys_profile.sh. The imported
report is at _ablation_out/nsys_chain/prof.nsys-rep (open in nsys-ui for the timeline).

## UPDATE 4 — the reserve is cuDNN CONV (implicit-GEMM) workspace; conv3d-D-bucket didn't touch it
nsys timeline correlation of the unfreed >100MB chunks (216MB×2, 108MB×2, 151MB) to nearby kernels:
- **216MB / 108MB allocs run next to `sm80_xmma_fprop_implicit_gemm_..._nhwckrsc_nhwc` kernels** = cuDNN
  CONVOLUTION (implicit-GEMM), NOT SDPA. (The sdpa_sm120 flash kernels are present but not adjacent to
  the persistent chunks.) So the ~800 reserve is cuDNN conv workspace, cached/unfreed.
- Codex's GGML_CUDNN_CONV3D_BUCKET_D=16 (pad conv3d depth) — GPU test: decode peak UNCHANGED (12771),
  overall 13068. So D-bucketing did NOT collapse these workspaces. Hypotheses (for Codex):
  1. These come from a DIFFERENT cuDNN conv path than conv3d-cudnn.cu (the `sm80_xmma_fprop_implicit_gemm`
     is an implicit-GEMM conv — check whether the VAE decode's convs route through conv3d-cudnn.cu or a
     generic cudnn conv). Bucket the path that actually allocates the 216/108MB workspace.
  2. Workspace is keyed by spatial/channel (seg-1 & seg-2 decode share 1920×1088 spatial, differ only in
     D) — if D-independent, per-D bucketing is a no-op. Then the seg-2 +800 isn't D-variance; find what
     conv shape seg-2 sees that seg-1 doesn't (the +K keyframe changes the temporal conv extent?).
  3. Verify BUCKET_D even engaged (the cudnn-conv3d-plan logs rotate out of the 150MB docker window; use
     the cont97-style file capture or a short repro to confirm planD=16 on the D=13 shape).
The imported nsys report (_ablation_out/nsys_chain/prof.nsys-rep) has the full conv-op timeline — open in
nsys-ui, or query CUDA_GPU_MEMORY_USAGE_EVENTS joined to CUPTI_ACTIVITY_KIND_KERNEL by time window (see
nsys_profile.sh) to pin the exact conv descriptor that allocates + caches each chunk.

## UPDATE 5 — reserve is NOT at cuDNN build OR execute (exec-delta proof)
Codex added exec-time free-delta logs. GPU test (FILTER_ENGINES=1):
- 36596 [cudnn-conv3d-exec] + 20160 [cudnn-sdpa-exec] lines, **largest free-delta = -2.0 MB**. NO
  cuDNN execute allocates >150MB. Build-delta was already +0.0. So the ~800 is allocated at NEITHER
  the cuDNN plan-build NOR the execute — the two points we can bracket.
- **0 [cudnn-conv2d-exec] lines** — conv2d-cudnn.cu path is NOT used; the sm80_xmma_fprop kernels from
  UPDATE 4 were temporally adjacent to the chunks, not causal (the ±20ms nsys correlation was too loose).
- GGML_CUDNN_CONV_FILTER_ENGINES=1: decode 12771 / peak 13068 UNCHANGED.
- Also note: the 216/151/108MB chunks did NOT appear in GGML_CUDA_ALLOC_TRACE (>128MB) either → they're
  cuDNN-internal cudaMallocs bypassing ggml's allocator, made OUTSIDE build/execute.

**⇒ The reserve is allocated at cuDNN PLAN FINALIZATION or HANDLE/ENGINE INIT** (persistent per-shape
exec memory reserved there), which is why every reclaim (cache-clear, handle-destroy-at-boundary,
mempool-trim, workspace-cap, engine-filter, conv3d-bucket) returned +0.0 / no change.

**8 approaches tried, all null.** Remaining:
1. DEFINITIVE probe: `nsys --trace=cudnn` API timeline — correlate the 216MB cudaMalloc (correlationId
   in CUDA_GPU_MEMORY_USAGE_EVENTS) to the exact cudnnBackend* API call in CUPTI_ACTIVITY_KIND_RUNTIME
   that allocates it. Names the precise cuDNN allocation site (harness: nsys_profile.sh, --trace=cuda,cudnn).
   Caveat: the earlier nsys unfreed-set may partly be capture-end-live (capture stopped mid-seg-2) — run
   a FULL 2-seg capture and confirm the chunks persist past seg-2's start before trusting them.
2. ESCAPE: fix the GGML_CUDNN_ATTN=0 / CONV3D=0 crash (partial=5887 streaming blowup) → cuDNN-off mode
   provably reclaims the reserve (the ONLY lever that frees it), a cross-project fallback.
Pragmatic ship today: 12.7GB works on 16GB; frame-fix + x1.75 upscale fits ≤11.5 despite the reserve.

## RESOLVED — the reserve is CUDA kernel-CODE memory (cuLibraryLoadData), not an allocation (2026-07-07)
Definitive from the nsys capture (CUPTI_ACTIVITY_KIND_RUNTIME):
- **cuLibraryLoadData = 98 calls, cuLibraryUnload = 4 → 94 cuDNN kernel-variant libraries RESIDENT.**
- Timing: 85 loaded progressively across seg-1's phases, +13 in seg-2 — matches the floor climbing
  1247→2027→2949→3727 (each phase touches new shapes → loads more kernel code libraries).
- ~18 MB/library × 94 ≈ the ~1.7GB. 8 distinct cuDNN kernel families, 158 distinct kernels total
  (sm80_xmma_fprop implicit-GEMM conv, fort_native_sdpa flash, etc.).
- CUDNN_LOGINFO_DBG/LOGLEVEL_DBG do NOT emit anything on cuDNN 9's frontend path (dead diagnostic).

**Why all 11 prior approaches were +0.0:** they all targeted *allocations* (cudaMalloc/pool/plan/handle/
mempool/workspace). This is `cuLibraryLoadData` = kernel CODE in the CUDA context — invisible to every
allocation tracker, ~0 at plan-build AND execute (loads at first kernel *dispatch*), but counted by
cudaMemGetInfo. Not a leak, not a workspace — cuDNN's lazy per-shape kernel-library loading.

**Correctly-scoped fix levers (for Codex):**
1. **Cut the number of unique cuDNN kernel variants loaded.** Shape bucketing (pad refine to a fixed Lq,
   fixed conv shapes) so seg-1/seg-2 reuse the SAME cuDNN kernels → fewer libraries. NOTE: this is
   different from the workspace-bucket that failed — the goal is fewer distinct KERNELS, verify via the
   cuLibraryLoadData count dropping (query above), not the workspace.
2. **Engine/heuristic limiting** so cuDNN selects from a smaller kernel set (fewer sm80/sm120/tile
   variants). GGML_CUDNN_CONV_FILTER_ENGINES is the start; extend to SDPA + verify the load count.
3. **CUDA_MODULE_LOADING** — already LAZY by default; won't shrink the total (loads are demand-driven).
4. **ESCAPE:** fix the GGML_CUDNN_ATTN/CONV3D=0 crash → no cuDNN = no kernel-variant loading = the reserve
   is gone (the only lever that provably frees it; also the cleanest cross-project fallback).
VERIFY any fix by re-running the nsys harness and checking cuLibraryLoadData drops (query in this doc).

## UPDATE 6 — NVTX attribution: the kernel-library loads are 80% SDPA (bucket the Lq)
NVTX cudnn-op ranges (id/kind/shape) on execute + GGML_CUDNN_OP_TRACE, attributed each cuLibraryLoadData
to the cudnn-op whose build triggered it (loads fire at plan-FINALIZE, just before the op's first execute,
so join each load to its NEXT cudnn-op NVTX range). Result on the 2-seg reduced capture (97 loads total):
```
  by kind:   SDPA = 78 loads,  conv3d = 19 loads
  top shapes:
    sdpa Lq=8160  Lkv=8160  D=128 -> 25  (first/base 16-frame self-attn: cuDNN's big initial variant set)
    sdpa Lq=1024  Lkv=1024  D=128 -> 14
    conv3d D=18 H=21 W=34 -> 9,  conv3d D=16 H=7 W=6 -> 6
    sdpa Lq=32640 (seg-2 refine), Lq=38760, Lq=127/152 (cross-attn), Lq=1024 D=64 ... -> 2-3 each
```
**⇒ The ~1.7GB kernel-code reserve is DOMINANTLY the SDPA path (78/97 loads).** conv3d/conv2d bucketing
(already committed) only touches 19 loads — a minor win. The real fix is **SDPA shape bucketing**: pad the
attention to a fixed set of (Lq,Lkv,D) so cuDNN loads ONE kernel variant set instead of one per distinct
shape. This is the K/V-masked padding Codex deferred — now CONFIRMED as the 4×-bigger lever.

**Fix plan (Codex):** bucket SDPA Q/K/V to fixed max Lq (self-attn) + fixed cross-attn K length, with a
padding MASK so extra keys don't change the softmax denominator (cuDNN SDPA supports an attn mask /
padding). Verify by re-running nsys_profile.sh and checking `cuLibraryLoadData` (by-kind sdpa) drops.
NVTX query is in this run's analysis (join CUPTI_ACTIVITY_KIND_RUNTIME cuLibraryLoadData to next NVTX
cudnn-op by start time). Tooling: nsys_profile.sh (add --trace=...,nvtx + GGML_CUDNN_OP_TRACE=1).

## UPDATE 7 — SDPA bucketing works but PARTIAL (needs tighter bucket selection)
GGML_CUDNN_ATTN_BUCKET=1 (buckets 160,1024,8160,38760, set_padding_mask). nsys re-profile:
```
                baseline   ATTN_BUCKET=1
cuLibraryLoad:  98         87    (-11)
SDPA loads:     78         68    (-10)
net resident:   94         83    (-11)  ~= -200MB (~1/8 of the reserve)
```
Render completes fine (mask keeps softmax exact). But the collapse is partial — the op-trace still shows
NON-bucketed Lq: **127, 152 (below min bucket 160), 9690, 32640 (between 8160↔38760 — should round UP to
38760 but load their own kernels).** Only the exact-bucket shapes (1024/160/8160/38760) collapsed.

**Tuning for Codex to get the full drop:**
1. Bucket-select must be "**smallest bucket ≥ len**" (round UP) for ALL lengths — 32640/9690→38760,
   127/152→160. The current logic misses in-between + sub-min lengths.
2. **Bucket Lkv too, not just Lq** (cross-attn has Lq≠Lkv; both drive kernel variants).
3. D=64 vs D=128 (two head dims) and the padding_mask flag each still fork variants — check the mask
   path isn't adding as many kernels as bucketing saves (net was only -10 SDPA).
4. Consider a single "≥max seen" bucket per (D) to force ONE self-attn kernel + ONE cross-attn kernel.
Verify: re-run nsys_profile.sh, watch SDPA cuLibraryLoadData drop toward ~single-digits. The knob +
mechanism are proven correct; it's a bucket-coverage tuning problem now.

## UPDATE 8 — FINAL / CORRECTED: bucketing is a dead end; the "reserve" is the large-seqlen hires plans
Definitive analysis of the captured traces (nsys_bucket/prof.sqlite + op-trace, no re-render). **The whole "bucket the shapes to reclaim the reserve" premise was wrong.**
- Plan cache already collapses 22 real shapes → **12 bucketed plans**. cuLibraryLoadData only 97→87.
- **cuDNN variable-len SDPA loads a kernel per REAL seq_len**, independent of the bucketed graph dim. Proven: same plan `(38760,38760,D128)`, real 32640 loads a kernel AND real 38760 loads another, zero plan rebuilds between. Leak = `fattn-cudnn.cu:595-596` (seq_len tensor = REAL len). This is **correctness-required**: the padding mask uses seq_len to exclude zero-padded K cols; force it to the bucket and exp(0)=1 pad cols inflate the softmax denominator → wrong output. A −∞ bias-mask avoids the seq_len tensor but the colliding bucket (38760) → bias ≈3GB. **No cheap correct cuDNN-layer fix.**
- **Memory is concentrated, not diffuse:** every plan build +0.0MB EXCEPT the 38760-token hires plans (build #9 used 7325→11057MB, #10 →12269MB). The reserve == the **1080p refine attention at 38760 tokens** + audio a2v (~+3GB). Not reclaimable kernel-code trivia — it's the real cost of large-seq attention.
- **Verdict:** 1080p+audio continuation over 16GB is a CAPACITY limit. Stop chasing reclaim (plan-clear/mempool-trim/bucketing all proven null). Real levers: (1) **tile the hires refine attention** (reuse VAE spatial-tiling → smaller per-tile seq_len — the proper fix), (2) lower upscale (x1.5 has its own seg-0 death to debug), (3) split audio at full res (loses lipdub). `GGML_CUDNN_ATTN_BUCKET` stays as-is (harmless, ~10 fewer loads, off by default) — do NOT invest more in it for VRAM.
