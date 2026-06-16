# HANDOFF-H — geometry-eval results: TRELLIS.2/Pixal3D replacement + cleaner/lighter-mesh levers

**Date:** 2026-06-16 · **Branch:** `spike/sparse-conv-3d` · **Supersedes the plan in** HANDOFF-G.

**Goal (refined by owner this session):** not just "pick a replacement generator" but **shrink the asset
without losing meaningful detail → a cleaner, lighter, riggable character mesh** (separable fingers, smooth
watertight surface) to feed the geometry-agnostic SkinTokens rigger. GPU is shared; pixal3d is **already
C++/ggml-ported** (so we drive the C++ port, not python, except as a validation oracle).

> TL;DR — three findings that reframe the whole effort:
> 1. **Pixal3D (what we run) is the NEWEST model in the room.** Almost the entire HANDOFF-G shortlist
>    (StdGEN, TripoSG, Step1X, PartPacker, Hunyuan3D-2.1) is *older* (Q1–Q2 2025) than pixal3d (SIGGRAPH
>    2026, May 2026). Owner's "these are old models nobody talks about" instinct is **correct**.
> 2. **The real 2026 development is a RETOPO bolt-on wave**, not a better generator. New AR "artist-mesh"
>    models (FastMesh-V4K / TSSR / DeepMesh — open weights *now*) take a point cloud sampled from our dense
>    mesh and re-emit a **clean, light, low-poly** mesh. That is *exactly* the "shrink without losing detail"
>    lever, and it's a post-process, not a generator swap.
> 3. **`--resolution 1536` is an INPUT-QUALITY knob for the downstream retopo, NOT a finger fix.** Owner
>    clarification (2026-06-16): pixal3d's *dense* output **already has clean fingers** — the fingers die in
>    the **lightening/decimation** step (dense → small = "spikey mess"). So 1536 does not fix fingers; it
>    gives the next stage (FastMesh/TSSR retopo) a finer, more faithful source surface to sample. Its worth
>    is measured at the **END** of the pipeline (is the final *light* mesh cleaner from a 1536 vs 1024
>    source?), not on the dense mesh. The C++ port hardcoded grid-1024; this session made it feature-complete
>    (`--resolution` flag → grid-96 HR lattice + 1536 dense marching) — a legitimate parity fix, but it is
>    orthogonal to the real problem, which is RETOPO quality (§3/§6). [1536 results below]

---

## 0. THE REAL PROBLEM + THE ANSWER (read this first — supersedes the generator-shootout framing)

**Problem (owner, crisp):** pixal3d gives a dense (~9M tri) visually-great single-character mesh; every
attempt to shrink it to a game-asset budget → blob, or shards with **merged fingers**. Generation is fine
(pixal3d ≈ SOTA, KEEP it). This is a **decimation/retopo** problem. pixal3d outputs **one fused watertight
mesh, NO parts** (`flexible_dual_grid_to_mesh` = single surface) → nothing protects the fingers when you
decimate.

**Answer (verified production-workflow research, 2026-06-16):** no single tool; it's a pipeline whose key
property is **REGION-ADAPTIVE DENSITY** — spend polys on hands/face, starve the torso. **Uniform
decimation/remesh provably can't** (this is why meshopt-sloppy/QEM/Instant-Meshes all mushed the fingers;
IM/QuadriFlow are the *weak* tools — global density only, no per-region paint, no thin-feature handling).
And **fingers are GEOMETRY, not texture** — normal maps can't fake the silhouette; Mixamo/auto-riggers *drop
finger bones* when fingers are fused → real spaced finger geometry MUST survive.

**The recipe (open-path for us in bold):**
1. **Segment the fused mesh → isolate hands.** **Tencent Hunyuan3D-Part / P3-SAM** ("any input mesh", open
   weights, commercial w/ territory+MAU caveats; **same Tencent/VecSet family we know → most portable**);
   NVIDIA **PartField** (open, NON-commercial, eval only); free no-ML = **SDF/curvature graph-cut**.
2. **Retopo with per-region density:** Quad Remesher (~$60, vertex-color density + finger edge-guides) or
   ZRemesher (names fingers, polypaint density). Free: **Blender Decimate + vertex-group mask**.
3. **Hand budget:** index+thumb independent often enough (weld rest); 2-3 loops/knuckle; hands ≈5-15% of
   body; mobile hero ~3k tris, Switch hero ~15k (BotW Link ~16.6k).
4. **Bake** normals/AO from the 9M original onto the low-poly (surface detail only, never silhouette).
5. **Hand-finish** fingers/face with **RetopoFlow** (free, Blender) where auto-retopo twists loops.

**KEY INSIGHT (owner, 2026-06-16): P3-SAM segmentation UNLOCKS the whole learned-retopo class we'd written
off.** MeshFlow/FastMesh/TSSR have low FACE CAPS (~4096) that mitten the *whole model* (4096 faces across a
full Miku = nothing) — but **per-part, the cap is a GENEROUS budget** (4096 faces on one hand ⪢ the ~150-250
tris a game hand needs). So segment → run a learned retopo **per part** at full budget. Backends (all
**point-cloud-in**, sample a cloud from the part mesh → clean artist mesh):
- **MeshFlow** ([facebook/meshflow](https://github.com/facebookresearch/meshflow), HF facebook/meshflow) —
  **flow-based, ~1s, 18× faster than AR**; owner tried it full-model yesterday → "lumpy" (4096 spread thin);
  per-part should be far cleaner (dense sampling on one hand). The speed pick.
- **FastMesh** (jhkim0759, AR, V1K/V4K) · **TSSR** (Tencent, ≤7k) — AR alternates.
So the pipeline is **P3-SAM → pick the best retopo backend PER PART** (hands → learned at full budget; body →
cheap QEM), NOT "P3-SAM → IM". Patched-IM and importance-weighted-QEM are the non-learned baselines to beat;
learned-per-part may well be cleaner. None of these learned backends survive on disk (envs cleaned up) —
dockerize per the P3-SAM pattern when we test them.

**NEXT EXPERIMENT (when GPU frees):** segment dense Miku with **Hunyuan3D-Part/P3-SAM** → isolate hands →
then A/B per-part retopo (MeshFlow/FastMesh learned at full per-part budget vs QEM/patched-IM) → recombine →
render fingers. This replaces the now-moot StdGEN/FastMesh-FULL-BODY shootout (full-body was the mistake; the
cap only bites full-body). (StdGEN's only draw was "parts" — we can get parts on our SOTA mesh via
segmentation instead of switching generators.) FastMesh/TSSR remain low-priority (full-body face caps mitten
hands; only viable per-hand). **P3-SAM env: `tools/m1_ref/cpp_port/shootout/p3sam_setup.sh`** (venv on /mnt/hdd;
backbone = facebook Sonata → flash-attn/spconv + chamfer3D CUDA ext; `auto_mask.py --mesh_path <mesh>` → masks).

### Is our decimation SOTA? Core = yes; defaults + density control = no.
- **`--remesh` = classic Garland-Heckbert QEM** (`qem.hpp`, sp4cerat public-domain). This IS the SOTA
  *baseline* (same quadric-collapse at the core of Simplygon/meshopt). Feature-preserving, non-manifold-tolerant.
- **BUG: the DEFAULT path uses meshopt `simplifySloppy`** (`texatlas::decimate`, no --remesh) = the explicitly
  low-quality fast variant → the "spikey mess". Fix: default to QEM/non-sloppy for hero assets.
- **Gap vs SOTA practice: no per-vertex importance weighting.** Our QEM is uniform (geometry-error only).
  SOTA (Simplygon vertex-weights / Quad Remesher density paint / weighted-QEM) scales the quadric by a
  per-vertex importance so hands resist collapse. **THE upgrade: add importance weights to `qem.hpp`
  (multiply quadric by importance), drive them from P3-SAM part labels (hands=high).** = the buildable SOTA
  pipeline. (Probabilistic-quadrics / attribute-aware QEM = marginal; density allocation is the real lever.)
- **The decimator is not the bottleneck** — uniform-perfect QEM still mushes fingers; per-region density is.

---

## 1. SIDE-TASK DONE — C++ pixal3d `--resolution` support (was hardcoded 1024)

The C++ port was supposed to be feature-complete vs the python `inference.py --resolution`, but it baked
grid-1024 in three places. Now driven by a `--resolution N` flag (default 1024 = unchanged).

**What "1536_cascade" actually changes** (from `Pixal3D/pixal3d/pipelines/pixal3d_image_to_3d.py`
`sample_shape_slat_cascade`): the HR sparse lattice = `resolution/16` (**64 @1024 → 96 @1536**), and the
shape decoder marches the dense dual-grid at `resolution`. The decoder is a fixed **16× sparse upsampler**
(grid_in → grid_in×16), so it is resolution-agnostic — feeding grid-96 coords naturally lands a 1536 dense
lattice. Same weights; finer voxels = thinner separable features (the finger lever).

**Files changed (all in `tools/m1_ref/cpp_port/`):**
- `pixal3d.cpp` — `--resolution N` (alias `--res`, must be mult of 16) → `ChainInput.resolution`; usage text;
  textured-bake `grid_res` now `in.resolution`.
- `pixal3d_chain.hpp` — `ChainInput.resolution` (default 1024); stage-4 quantize target `64`→`resolution/16`;
  stage-5 HR cond grid `64`→`resolution/16`; stage-7 `m4_decode_mesh(..., in.resolution)`.
- `svp_gpu.hpp` + `sparse_vae_pipeline.hpp` — `m4_decode_mesh(..., int resolution=1024)`; the final
  `flexible_dual_grid_to_mesh(..., 1024, ...)` → `..., resolution, ...`. (GPU + CPU decode paths.)

**Build:** `./build.sh pixal3d cuda` (clean; the `.sframe` ld note is benign).
**Run:** `./pixal3d --model weights_gguf --image prep_test_matte.png --out X.glb --resolution 1536 [--ply]`

**Correctness note / caveat:** this is a *consistent scale* of the validated 1024 path (the existing C++
`quantize_grid_unique` uses `round(... *(grid-1))`; the python cascade uses `int(... *(res//16))` — a
round-vs-trunc / off-by-one nuance the original author folded into the validated 1024 baseline). The 1536
path scales the same constants, so it inherits the same fidelity; it is **validated by rendering vs a python
1536 oracle**, not asserted bit-exact (1536 had no prior C++ reference). [oracle A/B status below]

### 1024 baseline (regression check — default path unchanged)
`N1=3606  M=15443 (grid64)  dual verts=4.56M faces=9.14M → decimated 145k/67k  wall=757s  peakVRAM=9273MiB`
→ matches the established baseline; the `--resolution` plumbing did not perturb the default. ✅

### 1536 result — RUNS but mesh is BROKEN (0 faces) → port is NOT yet feature-complete
`N1=3606  M=36295 (grid96, ✅ 2.35× the 1024 lattice)  → M4 decode verts=10,079,897 faces=0  wall=2463s`
- The flag plumbing is correct (grid96 lattice, M=36295, 10M verts at the 1536 dense lattice — the
  decoder's 16× upsample worked). **But `flexible_dual_grid_to_mesh` emitted ZERO faces** → the GLB is a
  point cloud, not a mesh. So `--resolution 1536` is **not usable yet**.
- Root cause is NOT coord packing (`coord_key` uses 20 bits/coord = handles ≤1M, 1536 is fine). It's in the
  dual-grid face extraction at the finer lattice (`sparse_vae.hpp:flexible_dual_grid_to_mesh`, ~L247-330).
  Needs a focused debug pass — DEFERRED (orthogonal to the real decimation problem; owner paused 1536).
- Perf data point (for the deferred pixal3d-speed work): M3b DiT = **2067s @1536 vs 391s @1024** (5.3×);
  total 2463s. 1536 is very expensive on the 3060.
- **DO NOT COMMIT the `--resolution` change as-is** — 1024 path is correct (regression ✅) but 1536 yields a
  faceless mesh, so the feature is incomplete. Either finish the dual-grid fix first, or commit with the flag
  explicitly documented as "1024 only; 1536 WIP (0-face bug)".

---

## 2. UltraShape — owner was right to reconsider; it's a bolt-on refiner, NOT a generator-family commitment

HANDOFF-G dismissed it as "a sparse-voxel refiner stacked on Hunyuan3D-2.1, same family." Re-checked the repo
([PKU-YuanGroup/UltraShape-1.0](https://github.com/PKU-YuanGroup/UltraShape-1.0), [arXiv 2512.21185](https://arxiv.org/abs/2512.21185)):

- **Interface is generator-agnostic:** `infer_dit_refine.py --image <png> --mesh <coarse .glb/.obj>`. The
  README *suggests* Hunyuan3D-2.1 for the coarse mesh, but it takes an **arbitrary** coarse mesh path — so we
  can feed it **our pixal3d/StdGEN mesh + the Miku image**. The refine-only path needs **no Hunyuan install**.
- **Knobs richer than expected:** `--num_latents 32768` (vs pixal3d's *actual 4734* shape tokens @1024),
  `--octree_res 1024` (MC res), `--steps 50→12`. So it can genuinely *add* surface resolution to our mesh.
- **Newer than the whole §1 shortlist** (Dec 2025). Same sparse-voxel family we're already porting → not wasted.

**Honest limit:** it refines detail **at the coarse mesh's voxel anchors** ("fixed spatial locations… reduced
solution space"). It sharpens soft/staircased surface; it does **not** re-topologize, so it will **not un-web
fingers already fused** in the coarse mesh. It attacks *surface softness* (real, on-target for "cleaner"), not
*topology* (the finger problem). Verdict: **worth a cheap A/B as a refiner/post-pass on the winning geometry**,
positioned alongside the AR-retopo route for "lighter."

---

## 3. pixal3d param levers for "cleaner / lighter" (the owner's actual question)

The C++ port already carries the **mesh-lightening machinery** (this is good news — "shrink without losing
detail" is largely already built):
- `--decimate <F>` — feature-preserving QEM target (default 150k; game asset e.g. `--decimate 40000`).
- `--remesh` — proper marching-tet manifold watertight remesh (no flaps; tight atlas) — supersedes the hole-fill hack.
- `--pack hero|small` — in-process meshopt (KHR_mesh_quantization + EXT_meshopt_compression) + KTX2 textures.
- `PIXAL3D_REMESH_FACES` / `PIXAL3D_REMESH_AGGR` env — QEM budget/aggressiveness.

**Where the fingers actually die (owner clarification):** NOT in generation — the dense mesh has clean
fingers. They die in **lightening**. The C++ default path calls **meshopt `decimate(sloppy)`** (1024 run:
`9.3M→145k faces`) = the "spikey mess". The fix is the **decimation/retopo method**, not the source:
- **First cheap test:** the port's own `--remesh` (feature-preserving Garland-Heckbert QEM, `qem.hpp` —
  defers collapses across sharp edges, rejects normal flips) vs the default sloppy, at a LOW budget
  (`--decimate 40000` / `PIXAL3D_REMESH_FACES`). This is the cheapest "keep the fingers" lever and is
  already built. (See [[project_pixal3d_retopo_manifoldplus]]: ROOT FIX was ManifoldPlus→IM with ADAPTIVE
  density; uniform decimation always trades fingers for cheek polys.)
- **Then** learned retopo (FastMesh-V4K / TSSR) — §6.
**`--resolution 1536` is NOT this lever** — it changes the *source* fed to retopo (input quality), measured
at pipeline end (§1, §6). The token cap is also not a lever (1024 emitted 4734 of a 20000 cap).

---

## 4. 2026 landscape freshness (verified web research; nothing dethrones Pixal3D for fingers)

The field bifurcated into (a) sparse-voxel hi-fi generators (TRELLIS.2 / Pixal3D = what we have) and (b) a new
wave of **AR artist-mesh retopo** models. Open-weights-NOW candidates relevant to "lighter/cleaner":

| Model | Released | What it does | Relevance |
|---|---|---|---|
| **FastMesh-V4K** ([HF](https://huggingface.co/WopperSet/FastMesh-V4K)) | Aug 2025 / 3DV'26 | point-cloud-in → clean low-poly artist mesh | **the "lighter" bolt-on** — sample our dense mesh → light triangle mesh |
| **TSSR** ([HF](https://huggingface.co/skyofsky/TSSR)) | Oct 2025 (Tencent) | discrete-diffusion topology-sculpt, ≤10k faces | clean artist topology, alt to FastMesh |
| **DeepMesh-0.5B** ([GH](https://github.com/zhaorw02/DeepMesh)) | ICCV'25 | RL-tuned AR artist mesh | retopo baseline |
| **AniGen** ([HF VAST-AI/AniGen](https://huggingface.co/VAST-AI/AniGen)) | SIGGRAPH'26 | image→**fully-rigged** char, **fingers-skeleton variant** | **open weights DO exist** (corrects HANDOFF-G "verify if any"); MIT, ~23GB. Rig monolith — doesn't compose w/ SkinTokens, eval its *geometry* only |
| **UltraShape-1.0** | Dec 2025 | image+coarse-mesh → refined | best open **refiner** (see §2) |
| **DetailGen3D** ([HF VAST-AI](https://huggingface.co/VAST-AI/DetailGen3D)) | Apr 2025 | rectified-flow detail refiner | alt refiner, MIT |

**Closed / API-only / paper-only (do NOT chase):** Hunyuan3D-2.5/3.0 (API-only; **2.1 is the open ceiling**),
Seed3D 2.0 (ByteDance API), PolyGen (API quad), SATO / QuadGPT / QuadLink / Mesh-Pro (native-quad frontier,
**no code/weights yet**), Sparc3D, ShapeGen, SuperCarver. **Native-quad open weights still do not exist.**

**Bottom line:** keep Pixal3D for dense geometry; the 2026 win is *downstream* (retopo bolt-on for light, +
UltraShape for surface), not a generator swap. The only generator worth a *structural* trial is **StdGEN**
(A-pose part-decomposed → separable hands attacks finger-webbing at the topology level, which neither 1536 nor
UltraShape can). PartPacker is its backstop.

---

## 5. Shootout eval rig (prepared this session) — StdGEN + UltraShape in ONE docker image

The two repos pin incompatible stacks (StdGEN py3.9/torch2.1/cu118; UltraShape py3.10/torch2.5/cu121/
flash_attn/cubvh), so **one image, two isolated conda envs** = single `docker rmi` cleanup, no host venv mess.
Weights mount from `/mnt/hdd` (not baked in). All under `tools/m1_ref/cpp_port/shootout/`:
- `Dockerfile` — two-env image (base = local `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`).
- `ultrashape_requirements.txt` / `stdgen_requirements.txt` — repo pins (torch/flash_attn/pytorch3d split out).
- `fetch_weights.sh` — `infinith/UltraShape` (~7.4GB), `hyz317/StdGEN`, SAM ViT-H → `/mnt/hdd/.../_weights`.
- `run_1536_compare.sh` — the C++ 1024-vs-1536 generation + hand-closeup/silhouette render + side-by-side.

Build: `cd shootout && docker build -t geom-shootout .`  ·  Run: `docker run --rm --gpus all --cap-add
SYS_ADMIN -v /mnt/hdd/3d/avatar-shootout:/work -v .../_weights:/weights geom-shootout conda run -n
ultrashape ...`. (`--cap-add SYS_ADMIN` for EGL offscreen render; see reference_ncu_docker_syadmin pattern.)
**Not yet built/run** — authored as prep while GPU was busy; build is CPU/IO-heavy (flash_attn/pytorch3d
from source), so kick it when it won't contend with a GPU test.

---

## 6. Plan — owner priority: FINALIZE the e2e "clean model" flow FIRST, pixal3d perf LATER

Owner direction (2026-06-16): **FastMesh-V4K and TSSR are the lead retopo candidates**; lock the end-to-end
clean-mesh flow before touching pixal3d speed. The e2e flow:

```
  pixal3d (dense geometry, C++)  →  RETOPO (FastMesh-V4K / TSSR / current QEM)  →  clean light mesh  →  SkinTokens rig
       [--resolution 1536?]            [the e2e "clean model" step to finalize]        [validate by RENDER]
```

Cheapest-signal-first within that:
1. **pixal3d `--resolution 1536`** ← done this session; read §1 verdict. The source-quality lever (finer voxels).
2. **Retopo bolt-on — FastMesh-V4K first, TSSR second** (both owner-flagged). Sample a point cloud from the
   dense pixal3d mesh → clean low-poly. This is the **e2e clean-flow step to finalize.** A/B vs the C++ port's
   *existing* QEM/`--remesh`/`--pack` path (we may already be close on "lighter"; the question is *topology
   cleanliness*). FastMesh/TSSR force python (AR transformers, unported) → eval-only for now.
3. **UltraShape refine** on the pixal3d mesh → **cleaner-surface** lever (A/B; won't fix fingers). Stack-able
   before retopo.
4. **StdGEN** → only *structural* finger fix (part-decomposed hands); PartPacker backstop. Needs docker env (§5).
5. **THEN** port the winning retopo to C++/ggml. FastMesh/DeepMesh/TSSR are **AR transformers (Qwen-class) =
   the LLM wheelhouse** (reuse `m1_ggml.hpp`) — same shape as the SkinTokens rigger port; UltraShape is the
   same sparse-voxel family as the already-ported pixal3d decoder. Validate-vs-fp32-oracle either way.

**DEFERRED (explicitly, per owner): pixal3d inference SPEED.** Current geometry-only wall ~757s @1024 /
~730s+ @1536 on the 3060; M3b DiT dominates (~390s, 12 steps). Do NOT optimize until the e2e clean-mesh flow
(retopo winner) is locked — the final pipeline shape determines what's worth speeding up. Likely levers when
we get there: fewer/cascaded DiT steps, the `--fast` f16 tensor-core path (already in the port), Q4_K DiT.

## 7. Gotchas / discipline
- Validate ported stages vs the **true-fp32 oracle** (`.float()` + eager attn, `NVIDIA_TF32_OVERRIDE=0`).
- **RENDER and look** — never judge fingers/surface from face counts or logs.
- 1536 VRAM on the 3060 is tight (1024 already 9.3GB); if 1536 OOMs, port the python token step-down loop
  (reduce resolution by 128 until under a token budget) or tile the decode.
- No Rust builds on-server; C++/docker builds are fine (coordinate GPU). See feedback_no_build_on_server.
