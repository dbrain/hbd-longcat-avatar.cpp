# HANDOFF — Pixal3D C++/ggml port: back to FEATURE-COMPLETE (perf parked)

---
## ✅✅ STATUS 2026-06-12 (this run) — UV-ATLAS PBR TEXTURES DONE
**The headline gap is CLOSED.** `pixal3d --tex` now emits real UV-atlas PBR (baseColorTexture +
metallicRoughnessTexture + TEXCOORD_0), replacing the interim COLOR_0. Full chain validated E2E
(`miku_uvatlas_e2e.glb` + a non-Miku turtle), and a per-voxel parity check vs the Python reference.
Details: **`FINDINGS-13-uvatlas-textures.md`**. Done this run:
1. **UV-atlas PBR textures** (task #1) — `tex_grid_sample.hpp` (grid_sample_3d, bit-exact 4.77e-7 vs
   flex_gemm) + bundled `thirdparty/xatlas.{h,cpp}` + `thirdparty/meshoptimizer/` (decimation) +
   `tex_atlas.hpp` (unwrap+raster+bake) + `glb_textured.hpp` (nlohmann json + embedded PNGs). Wired
   into `pixal3d --tex` (`--texsize`, `--decimate`, `--vcolor`). Validated visually + atlas-mean parity.
2. **glb_writer JSON cleanup** (task #3) — the textured writer uses nlohmann (`thirdparty/json.hpp`).
   (The untextured/COLOR_0 snprintf writer is left as-is; it's tiny and still correct.)
3. **Raw-photo front-end** (task #2) — `preprocess_photo.py` (host rembg BiRefNet RMBG-2.0 + crop +
   premultiply-on-black) = the upload-PNG→matte API ingestion step. Camera = `--fov` cut-line (MoGe
   auto-FOV is the unported nicety). Owner confirmed the end goal is an API (upload PNG → GLB → rig).
4. **Parity sweep** (task #4) — C++ chain matches `run()` stage-for-stage; bake matches
   `MeshWithVoxel`→`to_glb` (same attr_volume/coords/grid/layout/packing). Only Python-side diff =
   cumesh remesh/decimate/hole-fill/BVH-reproject (proprietary CUDA, intentionally skipped). See
   FINDINGS-13 "Parity statement".

**Multi-image (owner asked, then deferred):** NOT a shipped capability — Pixal3D proj mode is
single-image/front-view by design (ProjGrid asserts front-view-only; the "2 views" is training
augmentation; the multi-image mixin is a different non-proj variant). It's a real model change, not
"just wiring". Owner: skip for now, explore later (needs front+back image pairs to even validate).

**TOP PARKED ITEM (perf/quality of the new feature):** the O-Voxel mesh has ~24k tiny holes →
xatlas makes a chart per hole-boundary → the unwrap is ~100-140s and inflates the atlas (4324² @
30k charts). Python avoids this by remeshing to watertight first (cumesh). Fix = a normal-cone face
pre-cluster (cumesh-style) or a hole-fill/weld pre-pass. Banked with the other perf levers below.
---


**Owner directive (2026-06-12):** the Phase-C performance run is good enough for now — **park the
remaining perf tasks as "later" (they're banked in `PERF-NOTES-pixal3d.md`) and move back to
FEATURE-COMPLETE vs the Python library.** The headline gap is **full textures** (we currently ship an
interim *vertex-color* GLB, not the real UV-atlas PBR bake), plus any other parity gaps with the core
Python model. Run autonomously; decide, do, golden-validate, document, continue.

START by reading: this file · `HANDOFF-NEXT.md` (the product-goal ladder) · `CONFIGS-RESOLVED.md`
(tex decoder = `SparseUnetVaeDecoder`, out 6 PBR) · `PERF-NOTES-pixal3d.md` (perf state, parked).
Memory: `project_3dgen_cpp_port`, `project_avatar_rig_path`, `feedback_correctness_before_perf`.

---
## ✅ WHAT'S DONE + VALIDATED — do NOT redo

- **Geometry + texture MATH is fully ported + validated** (image → mesh + per-voxel PBR), pure
  C++/ggml from GGUF, matching the Python fp32 oracle. Stages: DINOv3@512/1024, NAF, SS DiT + SS VAE,
  M2/M3b sparse DiTs, M3a upsample, M4 O-Voxel mesh, M6 tex DiT + tex decoder (out 6 PBR).
- **CLI:** `pixal3d --model <gguf_dir> --image <png> --out <glb> [--tex] [--fov F | --cam ...] [--fast]`.
- **PERF (Phase C, DONE + parked) — textured E2E 509.7s → ~216s (−58%), mesh near-identical:**
  - LAP 1: GPU-resident sparse-VAE decode (`svp_gpu.hpp` + `svae_cuda.cu`) — decodes 254→50s, bit-exact.
  - LAP 2: F16 DiT weights (`pack_gguf --type f16`, `PIXAL3D_FAST`) — −30%, near-lossless. Q8 rejected
    (distributed loss). LAP 3: f16 tensor-core attention — −12%/DiT.
  - **`pixal3d --fast`** = the perf config (needs the `weights_gguf_f16/` set). Default path stays bit-exact.
  - **Measured peak VRAM (--fast run): 8199 MiB (~8.0 GB), at the M3b DiT** (M=4633 tokens → largest
    dense-attention working set). 12 GB card so ample headroom; ~0.5 GB over the 7.5 GB target. **Owner:
    8 GB is fine — not a wall ("we've fixed much worse"). Do feature-complete FIRST: it may change the
    perf/VRAM picture (the UV-atlas raster adds a stage), so re-measure perf AFTER features land.**
- **Weights live on `/mnt/hdd/pixal3d/`** (`weights_npy`, `weights_gguf` f32, `weights_gguf_f16`),
  symlinked back into `tools/m1_ref/cpp_port/`. (Owner moved them off the `/` SSD; ~24s/run HDD load tax.)
- Build: `cd tools/m1_ref/cpp_port && ./build.sh <test> [cuda]`. Live compare page (3-up: this-lap vs
  fp32 vs golden) at `http://10.0.0.208:8011/compare.html` (python http.server on :8011 from cpp_port/).

---
## ★ THE WORK — feature-complete vs Python (in priority order)

### 1. FULL UV-ATLAS PBR TEXTURES (the headline gap — replaces the vertex-color interim)
**Current state:** `m6_tex_decode` produces the per-voxel 6-ch PBR (base_color3 / metallic / roughness /
alpha) — **validated bit-exact vs the oracle (maxabs 3.5e-6)**. The GLB then bakes ONLY `base_color` as
per-vertex `COLOR_0` (the interim). **What's missing = `o_voxel.postprocess.to_glb`:**
- **xatlas `uv_unwrap`** of the mesh → per-vertex UV coords + a chart layout.
- **Rasterize/bake a 2048² PBR texture atlas** from the per-voxel attrs (nvdiffrast does this
  differentiably in Python; we need a plain CUDA rasterizer — sample each texel's surface point, look up
  the nearest voxel's 6-ch PBR). Produces **baseColor** (sRGB) + **metallicRoughness** (G=roughness,
  B=metallic) + alpha textures.
- **glTF embed**: emit `images` + `textures` + `samplers` + `pbrMetallicRoughness{baseColorTexture,
  metallicRoughnessTexture}` + per-vertex `TEXCOORD_0` (instead of `COLOR_0`).
- **Goldens:** `golden_stages/stage4_{cond,out}` (tex cond + tex_slat). For the atlas itself, capture
  the Python `to_glb` output (run `pixal3d/.../postprocess` on the golden mesh+PBR) → compare the baked
  atlas + UVs. Validate by re-rendering (the atlas is resolution/seam-dependent, so judge by visual +
  attribute-sampling agreement, not bit-exact).
- **Port surface:** xatlas (C++ lib, bundleable) + a CUDA triangle rasterizer (barycentric attr
  interp + voxel-attr lookup). This is the largest net-new feature piece. Scope it first; the PBR data
  is already in hand, so it's "geometry→UV→raster→embed", no new model math.

### 2. RAW-PHOTO FRONT-END (ingestion parity — currently needs a preprocessed matte)
Python takes a raw photo (rembg `BiRefNet(RMBG-2.0)` cut-out → crop → MoGe camera). The C++ CLI
currently needs a **preprocessed square matte** + `--fov`/`--cam`. For true "photo → GLB" parity:
- Host-side **rembg** (BiRefNet RMBG-2.0) for background removal + square crop (the kickoff's
  recommended cut-line: do this host-side, keep the C++ on the ggml math path).
- Camera: either port **MoGe** (monocular geometry → camera) or keep `--fov` as the documented bypass.
- Decide with the owner whether full MoGe parity is wanted or `--fov` is the cut-line. Cheapest =
  ship a small host `rembg.py` preprocessor + `--fov`.

### 3. glb_writer JSON cleanup (small)
`glb_writer.hpp` hand-rolls the glTF JSON with `snprintf`. Swap for the **bundled
`thirdparty/json.hpp`** (nlohmann; the parent fork already uses it in
`src/model_io/safetensors_io.cpp`). Do it alongside the richer UV-atlas glTF (#1), since that JSON
gets much more complex (images/textures/samplers).

### 4. Parity sweep (confirm nothing else is unported)
Diff the C++ chain against `pixal3d/pipelines/pixal3d_image_to_3d.py` end to end: mesh post-processing
(any smoothing/decimation/normal recompute the Python `to_glb` does), the exact glTF attributes Python
emits, and any config knobs we hard-coded. Note web-decimation (production-web nicety, not Python parity).

---
## PARKED — performance "later" (all banked in PERF-NOTES-pixal3d.md)
> ⚠️ Owner: do **feature-complete first** — the UV-atlas/raster work may shift the perf+VRAM profile,
> so don't re-optimize until features land, then re-profile. 8 GB peak is acceptable for now.
1. **Spike conv kernel** (~18s, the decode floor): naive fp32 implicit-GEMM, no tensor cores. Bit-exact
   win = better fp32 tiling / shared-mem weight staging. Tensor-core (tf32/f16) is faster but perturbs
   subdiv (judge by mesh IoU). Validate via `m4_gpu_test`.
2. **SS DiT** (42s, biggest stage; also the VRAM peak): flash-attn / persistent step-cgraph.
3. **VRAM < 7.5 GB** (measured peak ~8 GB still over budget): the f16-attn casts + the 4096/4633-token
   attention scores dominate the DiT peak; the decode holds shape_dec (1.9 GB f32) resident. Levers:
   stream/quantize the decode weights, shrink attention working set, or imatrix-Q4 the DiTs (the
   distributed quant error needs imatrix — capture per-channel act² from the 12-step forward →
   `ggml_quantize_chunk`). Sub-4 GB is the stretch.

---
## NON-NEGOTIABLES / GOTCHAS
Correctness vs the TRUE-fp32 oracle (NOT the bf16 golden); judge E2E by mesh agreement. Persistent-
weights buffer (gallocr recompute→NaN); fp32 tanh-GELU; float64 t_seq; C2S child-major; conv weight
[Co,Kd,Kh,Kw,Ci]→spike [27,Cin,Cout]; tex decode oracle needs GPU VISIBLE (flex_gemm Triton import).
**The `--fast` perf path is opt-in** — keep the default bit-exact for validation. C++/CUDA builds fine
here; NO Rust builds. No `pkill -f`, no rm-globs. Long jobs `run_in_background:true`. Sub-agents
DEADLOCK on background completion → drive heavy runs from the main loop. No multi-agent Workflow for
research. Keep PROGRESS LOG + PERF-NOTES + memory current.

## DEFINITION OF DONE (this run)
A **fully-textured** `pixal3d --tex` output — image → glTF with embedded UV-atlas PBR (baseColor +
metallicRoughness), matching the Python `to_glb` — plus a clear parity statement (what, if anything,
still differs from the core Python pipeline). Get as far as possible toward that; the rigging/AniRig
north-star (memory `project_avatar_rig_path`) is the phase after feature-complete.
