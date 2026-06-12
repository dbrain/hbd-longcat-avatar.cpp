# E2E port kickoff — Pixal3D/TRELLIS.2 → C++/ggml (image → mesh, geometry-first)

Handoff for a fresh agent taking over the **end-to-end functional port**. The
sparse-conv spike (the hardest novel primitive) is DONE and proven; now port the
rest of the pipeline stage-by-stage until image→GLB runs in C++ matching Python.

## North star (don't lose this)
Productionise Pixal3D as a **mute 3D-game-asset generator** — image → textured-
rigged GLB — as a koblem heavy GPU engine (worker-isolation, idle-unload, REST +
panel), like acestep/flux2/longcat. C++/ggml port is the target (no heavy Python on
the shared 3060). Pipeline: **Pixal3D mesh → SkinTokens rig → motion**. AniGen
(one-shot gen+rig) shares the same structured-latent shapes, so this port largely
transfers to it later. Full strategy: memory `project_3dgen_cpp_port.md`,
`project_avatar_rig_path.md`.

## The operating philosophy (from the owner — obey these)
1. **Functional E2E first, performance LAST.** Get "image → correct mesh GLB"
   working under C++, slow is fine. Only AFTER it's feature-complete do a
   performance run. See memory `feedback_correctness_before_perf`.
2. **Correctness == numeric precision.** Keep high precision (fp32/f64) during the
   functional port; validate each stage tight against Python goldens. Perf tuning
   later legitimately loosens precision (tf32/fp16 → ~1e-3); that's rounding, judged
   by E2E output, not breakage. Don't drop precision for speed now.
3. **VRAM is the real risk.** Replicate Pixal3D `low_vram`: sequential stages, each
   model resident on GPU only during its stage, offloaded after → peak = max(stage),
   not sum (Python fits the 3060 ~7.9 GB this way). **Instrument peak VRAM per stage
   from the first capture.** 7.5 GB co-resident is a perf-phase target, not now.
4. **Golden-tensor methodology** (proven in the spike): monkeypatch the Python model
   to dump each stage's input/output tensors → port the stage in C++/ggml → validate
   bit/tol-exact vs the goldens. The model run is GPU-but-one-time; C++ dev is then
   offline.
5. No multi-agent **Workflow** for research (`feedback_no_workflow_for_basic_research`)
   — but Explore/general-purpose sub-agents for bounded reads are encouraged. No
   `rm`-globs, no `pkill -f` (kills own shell — use PIDs). C++/cpp builds fine on this
   host; NO Rust builds here. Coordinate GPU with the owner (it's shared).

## What's DONE (the spike) — read these
- `HANDOFF-sparse-conv-spike.md` — full spike status.
- `PORT-SPEC-flexgemm-submanifold.md` — the sparse-conv algorithm + ggml mapping.
- `PERF-NOTES-sparse-conv.md` — perf intel (PARKED until the perf run).
- **Submanifold sparse conv3d is ported + validated**: `tools/sparse_spike/` —
  CPU f64 oracle (`sparse_conv.cpp`), CUDA implicit kernel (`sparse_subm_conv.cu`,
  fp32, ~1e-7 on all 9 real layers + synthetic), goldens + flex_gemm baseline. This
  is ONE of the net-new primitives; the rest are below.

## The pipeline (read `E2E-PORT-MAP.md` for the full stage-by-stage detail)
Live pipeline file: **`pixal3d/pipelines/pixal3d_image_to_3d.py`** (NOT
`trellis2_image_to_3d.py` — that's an older near-identical sibling). Geometry path:
PRE-A rembg/crop → PRE-B MoGe camera → (1) Sparse-Structure dense DiT @16³ + 3D-conv
VAE → coords → (2) Shape-SLat LR @512 sparse DiT → (3a) decoder.upsample×4 → HR
coords → (3b) Shape-SLat HR sparse DiT → shape_slat → (5) sparse-VAE decode →
O-Voxel mesh → GLB. (Stage 4 texture = PHASE-2, skip for first E2E.)

### Net-new primitives still to port (geometry) — see map for file:line
- submanifold sparse conv3d — **DONE** (spike).
- 2D bilinear grid_sample + camera unprojection (the "proj" image conditioning).
- DINOv3 2D-axial-RoPE + LayerScale (image-cond ViT).
- pixel_shuffle_3d (SS VAE decode).
- varlen + windowed/double-windowed **sparse attention** + sparse 3D RoPE (shape DiT).
- sparse up/downsample + spatial↔channel reshuffle + subdivision prediction.
- NAF guide-upsampler w/ 9×9 **natten** neighborhood attention (only grid-32/64 tiers
  — avoidable in the first slice).
- **O-Voxel flexible-dual-grid mesh extractor** — biggest net-new piece; CUDA source
  NOT in tree (compiled `_C.so` only) → MUST validate against golden tensors.

### Portable (your existing wheelhouse)
DINOv3 ViT, dense flow DiT, dense 3D conv, layernorm/matmul/standard attention, and
the **FlowEulerGuidanceIntervalSampler** (Euler flow-matching + CFG + guidance-
interval + std-rescale + Möbius t-warp; portable; pseudocode in the map).

### ⚠️ Gotchas (from the pipeline mapping)
- **`pipeline.json` is fetched from HF at runtime, not on disk** — holds exact channel
  widths, sampler `guidance_interval`, normalization mean/std, per-block attn_mode/
  share_mod. **Capture it first** (it's in the HF cache after a run, or print it).
- `o_voxel` and `flex_gemm` ship as compiled `_C.so` only (no source) → golden-validate
  the mesh extractor and (already done) the conv.
- "proj" image cond is injected as a **plain additive residual** (proj_linear + add),
  not cross-attention.

## Recommended milestone ladder (functional, golden-validated each step)
- **M0 — capture goldens for every stage boundary.** Extend the spike's
  `golden_hook.py` pattern to dump: preprocessed image, image-cond features, SS noise+
  coords out, shape-slat in/out, mesh verts/faces. + per-stage peak VRAM. One GPU run.
  Also grab `pipeline.json`.
- **M1 — FIRST SLICE: Stage 1 (grid-16, no-NAF) E2E.** Image → SS dense DiT (sampler)
  → SS VAE decode (pixel_shuffle_3d) → coords. Exercises dense DiT + proj grid_sample +
  the sampler with NO sparse-attention/NAF/mesh dependency. Smallest real E2E loop.
- **M2 — Shape SLat sparse DiT** (sparse attention + sparse RoPE) → match shape latent.
- **M3 — Shape sparse-VAE decode** (conv DONE + up/down + subdivision) → match substruct.
- **M4 — O-Voxel mesh extraction** → match verts/faces → write untextured GLB.
- **M5 — wire full geometry E2E**: image → untextured mesh GLB, numerically matching
  Python. ← "functionally complete" geometry milestone.
- **M6+** — texture branch (NATTEN etc.); then integrate into longcat-avatar.cpp +
  docker build; THEN the performance run.

## Environment / how to run (all verified this session)
- **Model + venv**: `/mnt/hdd/3d/avatar-shootout/Pixal3D` (+ `.venv` py3.10, flex_gemm
  installed, only-backend). Run a decode: `run_pixal3d.sh <img> <res>` (low_vram).
  Test image e.g. `/mnt/hdd/3d/avatar-shootout/assets/miku.png`.
- **C++ port worktree**: `/home/dbrain/dev/longcat-sparse-spike` (branch
  `spike/sparse-conv-3d`, off longcat-avatar.cpp@5e26fc5; UNCOMMITTED). ggml submodule
  checked out. The sd.cpp/ggml base has DiT/ViT/VAE/attention kernels to reuse.
- **Host has NO system nvcc** → use the toolchain: `/mnt/hdd/3d/avatar-shootout/
  toolchain/bin/nvcc` (CUDA 12.4, `-arch=sm_86 -ccbin <toolchain>/bin/g++`), run with
  `LD_LIBRARY_PATH=<toolchain>/lib`. Compile is GPU-free; see
  `tools/sparse_spike/run_bench.sh` for the pattern. Canonical integration build is
  the longcat-avatar **docker** builder (owner prefers docker for the real subsystem).
- **GGUF**: NOT needed yet — we validate ops against captured `.npy` goldens, not by
  loading the model in C++. GGUF (safetensors→GGUF) only matters once running the
  whole pipeline in C++ (a later milestone).
- Goldens so far: `tools/sparse_spike/golden_model/` (real conv layers, gitignored
  2.1 GB) + `flexgemm_timing.json` + 3060 autotune cache.

## PROGRESS LOG

### 2026-06-12 (latest) — PHASE B / M6 TEXTURES ✅ (tex DiT + tex decoder validated; vertex-color textured GLB)
**The texture branch is ported + validated — image → TEXTURED (colored) Miku.** Both net-new tex models
exported (export_weights.py + stage4_decode_capture.py), ported, and validated vs fp32 oracles:
- **tex SLat DiT** (`slat_flow_imgshape2tex_1024`, in_ch 64): each forward concats [tex_noise(32) ‖
  RE-normalized shape_slat(32) `(x-mean)/std`] → build_slat_dit_forward (UNCHANGED — input_layer is [1536,64],
  so a [64,M] input just works), out 32. **CFG OFF** (GS1.0/GR0.0 → CFG branch == cond-only, added a fast-path
  to geo::flow_sampler that skips the neg forward), interval **[0.6,0.9]**, RT3, 12 steps. cond = REUSE stage3b
  cond (stage4_cond == stage3b_cond, maxabs 0.0 — same DINOv3@1024+NAF+proj_grid64). **tex noise = 4th
  continuous seed-42 draw** randn(M,32) (torch_randn 4th draw vs ref maxabs 2e-6). tex denorm = tex_slat_norm.
  `m6_tex_sampler_test` (golden cond + golden shape_slat + 4th noise): pre-denorm vs torch_tex_slat fp32
  **cosine 1.000000**, vs golden bf16 0.9999. (torch_stage4_ref.py oracle: cosine 0.999913, NOISE CONFIRMED.)
- **tex decoder** (`tex_dec`, SparseUnetVaeDecoder, out 6, pred_subdiv=false): reuses the M4 backbone but the
  C2S up-blocks use **guide_subs** (the shape decoder's per-level subdiv `>0`, captured by m4_decode_mesh's
  out_subs; isolated test uses golden stage5_mesh/sub{0..3}) instead of to_subdiv; head output_layer 64→6 PBR
  (base_color[0:3]/metallic[3]/roughness[4]/alpha[5]) ·0.5+0.5. `m6_tex_decode` + `m6_tex_decode_test` vs the
  fp32 oracle (stage4_decode_capture.py: real tex_dec fp32 + spike conv): **out_coords IoU 1.0 (1547076 voxels),
  PBR maxabs 4.7e-6** (bit-exact). conv weights exported in spike [27,Cin,Cout] (conv_weight_to_vcc), 284 tensors.
- **Textured GLB:** glb_writer gained a COLOR_0 (per-vertex base_color VEC3) path; mesh vert i ↔ PBR voxel i
  (both grow from the same subs). Isolated bake (golden mesh + my PBR) → **unmistakably teal Miku** (base_color
  mean [0.19,0.42,0.39] = her palette; render `miku_textured_render.png`). E2E wired into `pixal3d --tex`
  (run_geometry textured path: hoist stage3b cond, m4 returns subs, tex DiT reuses cond + 4th noise, tex decode
  → per-vertex color → colored .glb).
- **Net-new M6 files:** torch_stage4_ref.py, stage4_decode_capture.py, m6_tex_sampler_test.cpp,
  m6_tex_decode_test.cpp, svp::m6_tex_decode, pix tex stage, glb COLOR_0. GOTCHA: tex decode oracle needs the
  GPU VISIBLE (flex_gemm Triton import needs a CUDA driver even though the conv is monkeypatched to CPU spike).
- **FULL TEXTURED CLI E2E — DONE ✅:** `pixal3d --model weights_gguf --image preprocessed.png --out
  miku_cli_tex.glb --tex` → **509.7s, teal Miku** (N1=1120, M=4633, 1.47M v / 3.05M f). COLOR_0 mean
  [0.20,0.41,0.38] == the isolated golden-input bake (E2E tex stage correct). All 9 models now packed to GGUF
  (slat_flow_imgshape2tex_1024 + tex_dec added, bit-exact). compare.html updated → 2-up textured (mine vs
  golden+PBR) + a geometry toggle, served on :8011. (model-viewer applies COLOR_0×baseColorFactor → teal in
  browser; pyrender ignores COLOR_0 on a material, hence the earlier grey screenshot.)
- **REMAINING M6 (faithful-match, deferred — large):** the proper **UV-atlas to_glb** (o_voxel: xatlas
  uv_unwrap + nvdiffrast differentiable-raster bake of a 2048² PBR texture atlas + glTF embedded
  baseColor/metallicRoughness textures — multi-day: xatlas + CUDA rasterizer port). Vertex-color GLB is the
  working interim. JSON cleanup: glb_writer → bundled `thirdparty/json.hpp`.
- **PHASE C (perf) KICKED OFF — `PERF-NOTES-pixal3d.md`:** full timing breakdown banked. **#1 = the host
  sparse-VAE decodes (M3a+M4+tex ≈ 255s, 50% of wall — CPU-bound: spike conv on GPU but dense
  linear/layernorm/silu + nmap-hashmap + coord-growth on CPU OpenMP; move dense ops to GPU; validate cheaply
  via m4_mesh/m6_tex_decode_test golden-input, no full E2E).** #2 = quantize the 4 DiTs via GGUF Q-types
  (193s, 38%; leverage A2; judge by E2E mesh IoU) + re-enable tf32. #3 = VRAM 7.5GB co-residency.

### 2026-06-12 — PHASE A COMPLETE ✅✅ A2 GGUF + A3 `pixal3d` CLI (GGUF + photo → GLB)
**THE PRODUCT GOAL (geometry half) IS MET: `pixal3d --model <gguf> --image <png> --out <glb>` runs the
whole image→mesh pipeline from GGUF weights, in pure C++/ggml, matching the Python library.** Validated
end-to-end by a full CLI run (337.9s/3060): `--model weights_gguf` (all 7 models from GGUF) `--image
preprocessed.png` (a PNG, loaded+resized in C++) → N1=1120, M=4633, 1.47M verts/3.05M faces, clean Miku.
- **A2 — GGUF pack + GGUF-backed weight() (the dev→product bridge):**
  - `pack_gguf.cpp` (ggml's gguf writer): weights_npy/<model> → one `<model>.gguf`, ne=reversed(npy shape);
    the only >4D case (ss_dec's 20 conv3d weights [OC,IC,KD,KH,KW]) collapses to 4D [KW,KH,KD,OC*IC] ==
    exactly weight_conv3d()'s ne. ALL 7 models packed (~19GB): dinov3/naf/ss_flow/ss_dec/slat_flow_512/
    slat_flow_1024/shape_dec. `gguf_validate.cpp` (ggml reader) + `gguf_reader.hpp`/`gguf_reader_test.cpp`
    (dependency-free mmap reader, for the no-ggml m3a/m4 standalone build): **every model BIT-EXACT vs .npy.**
  - **Loader:** M1Harness gains a GGUF path gated by `PIXAL3D_GGUF_DIR` (env) — weight()/weight_conv3d()
    fetch the stored tensor (gguf_fetch), upload from gguf host data. `svp::WLoad` (sparse-VAE host ops)
    gains a parallel GGUF path via the standalone reader. **dinov3_test + ss_vae_test (incl conv3d) give
    BIT-IDENTICAL output npy vs GGUF** (global maxabs 1.335e-5 same to the digit; ss_logits worst@124907
    got=6.139988 same; coords 1126==1126). Zero code changes to the tests — env-var transparent.
- **A3 — `pixal3d` CLI:** `pixal3d.cpp` (args --model/--image/--out/--fov/--cam/--cpu/--ply) + `image_io.hpp`
  (stb_image load + a PIL-matching Lanczos-3 resize, antialiased) + `pixal3d_chain.hpp` (the chain extracted
  from geometry_e2e.cpp into `pix::run_geometry(ChainInput)→Mesh`, shared by CLI + harness). Image path
  validated: C++ Lanczos-3 resize(preprocessed.png,512/1024) vs PIL-derived refs = **meanabs 2e-4** (max ~0.04
  at sharp edges = PIL 8-bit fixed-point vs float, far below model noise floor). Camera = --fov (deg→rad) or
  --cam, default miku cam.json. Output = web-ready .glb (C++ glb_writer now bakes vertex normals + a
  doubleSided PBR material; `ply2glb.cpp` re-bakes any .ply). Build `./build.sh pixal3d cuda`. The CLI run's
  M=4633 (vs npy-ref geometry_e2e 4690) is the only delta = C++-Lanczos vs PIL-Lanczos input → a valid
  alternate realization, same Miku. shape_slat_norm_{mean,std} (config consts) ship in the model dir.
- **Web rendering (owner-requested):** `glb_writer` emits web-ready GLBs; `bake_web_glb.py` + `compare.html`
  (2-up <model-viewer>, mine vs golden, synced cameras — camera-sync gated on user-interaction to avoid a
  CPU feedback loop) served from a host http.server. Full-res GLBs ~70MB (decimation = production-web TODO).
- **NEXT = PHASE B / M6 (textures = feature-complete):** export tex models (slat_flow_imgshape2tex_1024 +
  tex_dec — the latter needs the shape_dec spike-conv export path) → tex DiT (in_ch 64 = 32 noise ‖ 32
  RE-normed shape_slat, CFG off GS1.0 interval [0.6,0.9], RT3; tex noise = **4th** seed-42 draw randn(M,32);
  tex_slat denorm) + tex decoder (out 6 PBR base_color[0:3]/metallic[3]/roughness[4]/alpha[5], ·0.5+0.5,
  pred_subdiv=false → reuses shape decode's per-level `subs` as guide_subs) + NAF@1024 (reused) +
  textured-GLB bake (o_voxel to_glb: uv_unwrap + atlas, OR a simpler vertex-color GLB first). Goldens
  stage4_{cond,out} captured. Then Phase C (perf: GGUF Q-types/fuse/fit 7.5GB).

### 2026-06-12 — A1 GEOMETRY E2E CHAIN ASSEMBLY DONE ✅ (image → mesh GLB, pure C++/ggml)
**PHASE A item 1 COMPLETE: one driver, image → untextured Miku mesh, reproducing the seed-42 python
output.** `geometry_e2e.cpp` wires the per-stage-validated programs into ONE chain; new artifacts
`geometry_e2e.{cpp,hpp}`, `torch_randn.hpp`, `sparse_vae_pipeline.hpp`, `glb_writer.hpp`, `randn_test.cpp`,
`render_mesh.py`; build.sh gained a `geometry_e2e cuda` branch (ggml-cuda graphs + spike conv in one link).
- **Net-new plumbing, all validated:**
  - **torch CPU randn ported** (`torch_randn.hpp`): the cross-stage seed-42 noise. The float32 path is
    torch's `normal_fill`/`normal_fill_AVX2` (NOT per-elem `normal_distribution<double>`): fill `size`
    uniforms via `random()` (24-bit), then in-place block-of-16 Box-Muller (`normal_fill_16`). MT19937 =
    `at::mt19937_engine`. ONE continuous stream: `randn(1,8,16^3)`;`randn(N1,32)`;`randn(M,32)` over the
    COMPUTED counts. `randn_test` vs refs/{noise_seed42,stage2_noise,stage3b_noise}: **algorithm bit-exact
    (>60% values identical), maxabs 2.2e-6** = venv-torch AVX2-Sleef sin/cos/log vs scalar libm (sub-ULP,
    4 orders below the model's fp32-vs-bf16 floor → zero added voxel flips). GOTCHA: `uniform_real<double>`
    uses `random64()` not two `random()`; normal uses `log1p(-u2)`; theta uses pi-as-double.
  - **grid64 quantize+unique** (`geo::quantize_grid_unique`): `round(((v+0.5)/512)*(grid_res-1=63))` torch
    op-order + round-half-to-even (`nearbyintf`), then `std::set` = `unique(dim=0)` sorted (b,x,y,z). Coord
    ORDER matters (noise[i]↔coord[i]); pipeline uses `argwhere` (SS, sorted) + `unique` (HR, sorted), both
    matched by my `std::set` extraction → noise aligns per-voxel.
  - **GLB writer** (`glb_writer.hpp`): glTF 2.0 binary, single untextured mesh (POSITION + uint32 indices).
- **Chain (each = validated reuse):** DINOv3@512 (run ONCE, shared by stage1 proj16 + stage2 cond) → SS DiT
  12-step (GS7.5/GR0.7/RT5/[0.6,1]) → SS VAE → coords1 → NAF@512 + proj_grid32 → M2 DiT (RT3, t=0.6 lands
  on interval → CFG, double t_seq) → denorm → M3a upsample → quantize grid64 → DINOv3@1024 + NAF@1024 +
  proj_grid64 → M3b DiT → denorm → M4 decoder+mesh. VRAM low_vram-style (each harness opens/closes; peak =
  max(stage)). Run: `NVIDIA_TF32_OVERRIDE=0 ./geometry_e2e cuda`, **350.5s** on the 3060.
- **RESULT (fp32 chain, seed-42):** N1=**1120** (== validated stage1_e2e exactly), M=**4690** grid64 tokens
  (golden 4734, −0.9%), mesh **1500691 verts / 3071494 faces** (golden 1547076/3251686 = 97.0% / 94.5%).
- **VALIDATION (vs fp16 golden, no heavy fp32-python oracle needed):** vertex-voxel IoU ladder **0.969@grid16,
  0.936@grid32, 0.829@grid64, 0.657@grid128** + **near-identical AABB** (mine [-0.30,-0.42,-0.13]→[0.28,0.40,0.22]
  vs golden [-0.30,-0.42,-0.13]→[0.28,0.40,0.22]). Monotone-rising-as-coarser + matching AABB = SAME OBJECT,
  fine-boundary realization noise (a plumbing bug → low IoU at ALL scales). The grid64 0.83 (vs hoped ~0.97)
  is the data-dependent-noise realization shift: my N1=1120 vs golden 1126 (6-voxel stage1 diff) shifts the
  noise→voxel alignment downstream → a faithful but DIFFERENT realization than the fp16 golden (a tight ~1.0
  match needs a fp32-python oracle, same N1; deferred per "avoid heavy runs"). **Render (`render_mesh.py`,
  pyrender/EGL) = unmistakably Miku across 3 views** (floor-length twin-tails, skirt, heeled boots) → clean
  complete 3D mesh. `miku_geometry_e2e.{ply,glb}` + `_render.png` written.
- **NEXT = A2 (GGUF)**: pack the safetensors/weights_npy → GGUF + a GGUF-backed `weight()` in M1Harness
  (validate GGUF==.npy bit-exact); then A3 (CLI `pixal3d --model --image --out`). Base repo has the
  converter (`src/convert.cpp`, `model_loader.cpp`, `ggml/include/gguf.h`).

### 2026-06-12 — M5 SHAPE CONDITIONING VALIDATED ✅ (true cond for both shape stages)
**Every geometry COMPONENT of the pipeline is now ported + validated.** The M5 true-cond machinery
(the last net-new model surface) is done; only chain ASSEMBLY remains. New: `stage2_cond_test.cpp`,
`stage3b_cond_test.cpp`, `dinov3_1024_test.cpp`, `naf_1024_test.cpp`, `stage3b_cond_capture.py`;
parameterized `dinov3_graph.hpp` (Cfg CFG512/CFG1024) + `naf_graph.hpp` (Cfg CFG512/CFG1024).
- **get_proj_cond_shape recipe** (confirmed from `inference.py` IMAGE_COND_CONFIGS + the extractor):
  `z_proj = concat[proj_grid(DINOv3 patchmap, lr), proj_grid(NAF hr, hr)]` → `[gridR³, 2048]`, gathered
  at the voxel coords. ss=grid16/img512/noNAF; shape_512=grid32/img512/NAF-target512; shape_1024=
  grid64/**img1024**/NAF-target512. The proj projects each grid cell to a pixel (host scalar math:
  `focal_px=(img_res/2)/tan(cam/2)`, same camera as ss) + bilinear-gathers the map — only R, img_res,
  and the two map H,W vary; project ONLY at the coords (≡ dense-then-gather).
- **stage2 cond (grid32)** `stage2_cond_test`: proj_grid32 over the validated DINOv3 patchmap@512 (lr) +
  NAF hr@512 (hr), concat, at golden coords → vs golden stage2_cond proj_feats: **cosine 1.000000,
  lr-branch maxabs 2.19e-5** (proj_grid + DINOv3 essentially exact), hr-branch 2.6e-3 (the NAF's tf32-
  golden noise, M2 lesson).
- **DINOv3@1024** `dinov3_1024_test`: parameterized `dino::build_dinov3(CFG1024)` (64×64 patches, 4096
  patch tokens, same 2D-axial RoPE). vs captured golden: **patchmap maxabs 6.5e-5, global 2.67e-5**
  (meanabs ~1.6e-6). @512 path unchanged (dinov3_test still PASSes — regression clean).
- **NAF@1024** `naf_1024_test`: parameterized `naf::build_naf_forward(CFG1024)` — guide@1024 →
  avg-pool(2) to 512 → SRC=64, UP=8 (vs @512's identity-pool/SRC32/UP16); na2d shift-rule identical,
  just dil8/block8 over the 64² source. vs tf32 golden: **meanabs 1.22e-5, maxabs 3.64e-3** (tighter
  than M2 @512's 5.3e-3 — correct within tf32 noise, M2 standard).
- **stage3b cond (grid64)** `stage3b_cond_test`: proj_grid64 over the captured patchmap_1024 + naf_hr_1024,
  concat, at golden coords → vs golden stage3b_cond proj_feats: **cosine 1.000000, lr 2.2e-5, hr 1.29e-4.**
- **Capture** `stage3b_cond_capture.py`: ran the REAL shape_1024 DinoV3ProjFeatureExtractor on the golden
  preprocessed image → **reproduced golden stage3b_cond proj_feats EXACTLY (maxabs 0.0)**, confirming the
  preprocessing (preprocessed.png → resize-1024-LANCZOS → DINOv3@1024 + NAF@1024 + proj_grid64) and saving
  validated goldens (image_1024_chw, patchmap_1024, naf_hr_1024, global_1024) to refs/stage3b/.
- **STATUS: every geometry component validated** — Stage1 (image→coords, M1), shape cond stage2/stage3b
  (DINOv3@512/1024 + NAF@512/1024 + proj_grid16/32/64), M2 (shape-LR DiT+sampler+denorm), M3a (upsample,
  bit-exact), M3b (shape-HR DiT+sampler+denorm), M4 (decoder + O-Voxel mesh, bit-exact). The cascade was
  validated stage-by-stage with golden hand-offs.
- **NEXT = M5 CHAIN ASSEMBLY** (wire the validated components into one image→mesh-GLB driver): the only
  net-new work is plumbing (cross-stage seed-42 noise reproduction — torch CPU MT19937 randn over the
  COMPUTED coord counts, NOT golden; denorm; the grid64 quantization+unique; an OBJ/PLY/GLB writer). All
  the heavy graphs are validated programs to call. See `HANDOFF-M5-assembly.md` for the precise plan. Then
  **M6** (texture branch = feature-complete): tex SLat DiT (in_ch 64, reuses M2/M3 DiT) + tex decoder
  (out 6 PBR, reuses M3/M4 sparse-VAE, pred_subdiv=false→guide_subs) + NAF@1024 (reused) + textured-GLB
  bake (o_voxel.postprocess.to_glb, UV/atlas — the largest net-new piece in M6).

### 2026-06-12 — M4 O-Voxel mesh extraction VALIDATED ✅ (decoder bit-exact + mesh BIT-EXACT vs o_voxel)
**Geometry is functionally complete: HR shape_slat → textured-ready mesh (verts+faces).** Two halves,
each independently validated. `m4_mesh.cpp` (full decoder→head→mesh) + `m4_mesh_only.cpp` (extractor
isolation) + `stage5_capture.py` (fp32 oracle + real-o_voxel mesh oracle) + `sparse_vae.hpp` gained
`flexible_dual_grid_to_mesh`.
- **(a) Full decoder.forward** (= M3a backbone, NO early-exit, +final F.layer_norm(non-affine,eps1e-5)
  +output_layer SparseLinear 64→7 + FDG head) on the golden HR shape_slat (M=4734 @grid64): grows
  4734→20533→87441→367657→**1547112 voxels @grid1024**. **Every per-stage out_coords EXACT (IoU 1.0);
  head_h7 maxabs 5.2e-4, dual_vertices 1.1e-5, quad_lerp 2.3e-4** vs the fp32 oracle (real decoder
  modules in fp32, conv monkeypatched — same recipe as M3a). FDG head: vertices=2·sigmoid(h[0:3])−0.5,
  intersected=h[3:6]>0, quad_lerp=softplus(h[6:7]).
- **(b) Mesh extractor** `flexible_dual_grid_to_mesh` (host C++, reconstructed from the o_voxel python
  wrapper): per-voxel vertex `(coord+dual)/1024−0.5`; 3 edge-dirs × 4-neighbor offsets; `intersected`
  selects (voxel,dir) quads; hashmap neighbor lookup keeps quads where all 4 corners exist; split each
  into 2 triangles by the diagonal with larger `quad_lerp` product (split_1 `[0,1,2,0,2,3]` if
  qlerp[0]·qlerp[2]>qlerp[1]·qlerp[3] else split_2 `[0,1,3,3,1,2]`). **m4_mesh_only: fed the oracle's
  exact head inputs → verts maxabs 0.0, faces elementwise-diff 0 (1547112 verts / 3251950 faces
  IDENTICAL to the real o_voxel `_C.so` CUDA kernel) → BIT-EXACT.** No learned weights — pure geometry.
- **Mesh oracle vs fp16 golden**: my fp32-head→o_voxel mesh = 1547112 verts / 3251950 faces vs the fp16
  golden 1547076 / 3251686; vertex voxel-set IoU **0.999** (boundary noise, M1 lesson, NOT a bug).
- **Integrated** (m4_mesh CUDA: my decoder→my head→my mesh): per-stage coords EXACT, head_h7 2.7e-4,
  **intersected diff=0**, verts maxabs 5.96e-08, **faces 3251950==3251950 with elementwise-diff=8** (8 of
  9.75M index entries — split-diagonal ties where `quad_lerp[0]·[2]≈[1]·[3]`). As bit-exact as fp32 allows.
  Both CPU and CUDA validated.
- **Tooling**: `npy.hpp` gained int8/int64/uint dtype support (+`i8()`/`i64()`; fixed a latent int64-as-
  int32 read bug). `build.sh` cuda branch now covers m3a+m4 (nvcc spike .cu + g++ host, no ggml).
- **NEXT = M5** (full geometry E2E): chain Stage1→M2→M3a→M3b→M4 in C++ with TRUE cond (NAF, not golden
  cond) → untextured mesh GLB matching python. Then M6 (texture branch = feature-complete).

### 2026-06-12 — M3a upsample (sparse-VAE LR→HR coords) VALIDATED ✅ CPU+CUDA (bit-exact)
**The mesh gateway is open.** `decoder.upsample(lr_slat, x4) → hr_coords` ported + validated bit-exact
on BOTH backends. `m3a_upsample.cpp` + `sparse_vae.hpp` + `stage3a_capture.py`.
- **fp32 ORACLE** (`stage3a_capture.py --mode fp32`): the REAL `FlexiDualGridVaeDecoder.upsample` run
  in fp32 on CPU with ONLY `SparseConv3d` monkeypatched to the spike's fp32 gather-matmul (so the
  oracle uses the REAL LayerNorm32 / SparseLinear / to_subdiv / SparseChannel2Spatial code — zero
  reimplementation risk for everything but the conv, which the spike already validated vs flex_gemm
  ~1e-7). hr_coords_fp32 = **382584** vs the existing fp16 golden **382554**: **IoU 0.999018** (≈190
  boundary voxels flip — fp16-vs-fp32 subdiv-logit noise, the exact M1 lesson; NOT a bug). Dumps
  per-stage in/out coords + subdiv logits + feats to `cpp_port/refs/stage3a/`.
- **C++ PORT** (`m3a_upsample.cpp`, imperative fp32 host): from_latent (Linear 32→1024) → 4 levels of
  [ConvNeXt×{4,16,8,4} + SparseResBlockC2S3d], coords grow ×2 per level (32→64→128→256→512). The one
  net-new op = the spike submanifold conv (`sparse_vae.hpp::subm_conv` → CPU mirror OR CUDA kernel,
  selectable). Dense ops (LayerNorm32 / SiLU / Linear, already ggml-proven in M1/M2) = host fp32 loops.
  Coord-growth (SparseChannel2Spatial) = host index arithmetic. **CPU**: every per-stage out_coords
  **EXACT (IoU 1.000000)**, per-stage feats maxabs ≤3e-4; **hr_coords SET-EQUAL the fp32 oracle
  (382584==382584, mine-only=0, oracle-only=0)**. **CUDA** (conv via spike GPU kernel): IDENTICAL —
  every coord EXACT, hr_coords SET-EQUAL oracle. Quantize to grid64 → M=4865 (the M3a→M3b gateway;
  vs golden stage3b M=4734, again fp16/fp32 amplified at the coarse any-occupant grid).
- **Design call (load-bearing)**: M3a is data-dependent (coords grow from `subdiv>0` decisions), so a
  static ggml graph can't express the loop. Implemented it IMPERATIVE host-orchestrated — the natural
  shape for a sparse VAE decoder, and it mirrors the real pipeline's flow-model-vs-VAE split. The spike
  conv IS integrated as the `subm_conv` op (CPU+CUDA). Full ggml-graph wiring of the dense ops is an
  M5-integration concern. Correctness-first: CPU fp32 == oracle, exactly.
- **VALIDATED conventions** (all confirmed against source, not the spec map): C2S feature packing is
  CHILD-MAJOR (`reshape [N,C]→[N*8,C/8]`, child k = contiguous block k); ConvNeXt order = conv→norm→
  mlp→residual; C2S order = to_subdiv(x) → norm1(affine)→silu→conv1(C→out*8) → C2S → norm2(NON-affine)
  →silu→conv2 → +skip(repeat_interleave×4); conv weight `[Co,Kd,Kh,Kw,Ci]`→spike `[27,Cin,Cout]` via
  `transpose(1,2,3,4,0).reshape` (golden_hook convention, validated vs flex_gemm); child k offset =
  `(k&1,(k>>1)&1,(k>>2)&1)` on (c1,c2,c3). 292 shape_dec weights exported to `weights_npy/shape_dec/`.
- **NEXT = M4** (O-Voxel `flexible_dual_grid_to_mesh`): the full decoder.forward (= M3a backbone, NO
  early-exit, +output_layer 7ch + FDG head) on HR shape_slat → grid1024 voxels → per-edge dual-grid
  quad assembly (hashmap neighbor lookup) → verts/faces. Mesh-extractor spec mapped from the o_voxel
  python wrapper (`o_voxel/convert/flexible_dual_grid.py`) — verify against source before porting.

### 2026-06-12 — M3b shape-HR DiT VALIDATED ✅ + M3a upsample SPEC'D
- **M3b** (`m3b_sampler_test.cpp` + `torch_stage3b_ref.py`): shape-SLat HR DiT+sampler+denorm at
  grid-64. Arch IDENTICAL to M2 — only weights (`slat_flow_1024`, exported+unpacked), HR coords
  (golden M=4734) and cond (`stage3b_cond`) differ; reuses `build_slat_dit_forward` verbatim.
  stage-3b noise = 3rd seed-42 CPU draw (`randn(1,8,16,16,16)`;`randn(1126,32)`;`randn(4734,32)`) —
  NOISE CONFIRMED (fp32 oracle vs bf16 golden cosine 0.9974). **C++ CUDA: pre-denorm vs fp32 oracle
  cosine 0.999996 (maxabs 0.28/meanabs 6.6e-4); denorm vs bf16 golden cosine 0.997410 (== oracle's
  own).** PASS. Cascade (LR→HR shape DiT) holds.
- **M3a** (upsample LR→HR coords, the sparse-VAE backbone) — NOT ported; full reconstructed spec in
  `HANDOFF-M3a-upsample-spec.md`. It's the next big net-new: SparseConvNeXtBlock3d + SparseResBlockC2S3d
  (channel→spatial coord-growth ×2 driven by `to_subdiv` octree-child logits) + 4-stage loop →
  hr_coords [382554,4]. Needs the spike sparse-conv integrated into the M-harness as a ggml op + a
  GPU golden capture of per-stage coords/subdiv (flex_gemm is CUDA-only). Data-dependent (subdiv
  read-back per stage). Validate hr_coords SET-EQUAL golden (integer coords → exact).

### 2026-06-12 — M2 NAF conditioner VALIDATED ✅ (ggml) → M2 FULLY COMPLETE
NAF (valeoai/NAF) guide-upsampler ported to ggml + validated. **M2 now complete: DiT +
sampler + denorm + NAF all validated.**
- **na2d semantics pinned** (`naf_na2d_probe.py`): natten (0.21, recent/cutlass-fna) uses the
  "shift" border rule — per dilation phase the 9×9 window slides inward to stay in-bounds
  (NOT per-index clamp, NOT mask). Probe matched @2.5e-6 everywhere; clamp/mask diverge at
  borders (3.6/3.2). For NAF (dil 16, k/v block-16 nearest-upsampled from 32²): na2d ==
  per-output-pixel 9×9 over the SOURCE grid centered at `clamp(i//16, 4, 27)`.
- **reference** `naf_ref.py` (torch functional, explicit ops): ImageEncoder bit-exact (0.0) and
  na2d 5.3e-5 vs the captured golden → full math de-risked. Saves a TRUE-fp32 oracle
  (`naf_ie_fp32.npy`/`naf_hr_fp32.npy`, tf32 disabled).
- **capture** `naf_capture.py`: NAF lr_features == the validated stage-1 DINOv3 patchmap
  (`dino_patchmap`, SAME dinov3-vitl16 @512) → captured standalone (no full-pipeline replay).
  Goldens `refs/naf_{guide,lr_features,hr_golden,ie_out,...}.npy`; weights `weights/naf.npz`
  (37 tensors) → `weights_npy/naf/`.
- **ggml** `naf_graph.hpp` + `naf_test.cpp`: ImageEncoder (2 conv encoders k1/k3 w/ reflect
  pad via `ggml_pad_reflect_1d`×2-axis; GroupNorm8 affine; SiLU; cat; axial RoPE half-split) +
  na2d (81 `get_rows` taps over flattened 32² source grid, host-precomputed shift indices as
  I32 consts; softmax; weighted V). **CPU ie 1.19e-5 / hr 1.34e-5; CUDA ie 6.5e-5 / hr 4.1e-5
  vs the true-fp32 oracle** → PASS. (vs the tf32-CUDA golden: ie 2.1e-2 / hr 5.3e-3 — the
  golden's tf32 noise, NOT error; M1 lesson reconfirmed: validate vs fp32, not the tf32 golden.)
- GOTCHAS: (1) `ggml_soft_max` CUDA grid = `(ne01,ne02,ne03)`; put the big dim (HW=262144) in
  ne01 (gridDim.x, ≤2³¹) not ne02 (gridDim.y, ≤65535) → scores laid out `[81,HW,4]` not
  `[81,4,HW]` (else "invalid configuration argument" abort). (2) harness gained
  `const_tensor_i32` for get_rows indices (mesh stages will reuse). (3) na2d block-equivalence:
  with block-16 nearest-upsample, clamp@512-grid == clamp source class to [0,32).

### 2026-06-12 — M2 sampler + denorm VALIDATED ✅ (lr_slat[N,32])
Full M2 Stage-2 shape-SLat LR is done: noise → 12-step FlowEuler → denorm → lr_slat[1126,32].
- **torch fp32 oracle** `torch_stage2_ref.py` — real `ElasticSLatFlowModel`(512) + real
  `FlowEulerGuidanceIntervalSampler` (GS 7.5 / GR 0.5 / RT 3.0 / interval [0.6,1.0] / 12 steps),
  golden cond (stage2_cond) over the golden 1126 coords, **seed-42 stage-2 noise** =
  `manual_seed(42); randn(1,8,16,16,16)` (discard) `; randn(1126,32)` on CPU (continues the global
  RNG — confirmed nothing between draws RNG: get_proj_cond_ss/shape, NAF, samplers don't).
  Denorm fp32 vs bf16 golden: meanabs 0.16 / cosine **0.9984** (range [-23,26]) → **NOISE
  CONFIRMED**; the oracle is the tight C++ target. Saved `refs/stage2_noise.npy`,
  `torch_lr_slat_fp32.npy` (pre-denorm), `torch_lr_slat_denorm_fp32.npy`, `shape_slat_norm_{mean,std}.npy`.
- **C++/ggml** `m2_sampler_test.cpp` — sampler loop (host scalar math) around `slat_dit_graph`
  (N=1126, in/out 32, proj 2048, sparse coord-RoPE) + per-channel denorm (`slat*std+mean`). The
  sparse `.std`-rescale = POPULATION std over all N·32 elems (`VarLenTensor.std` = `sqrt(E[x²]−E[x]²)`,
  ratio-invariant so `vstd` matches). **CPU: pre-denorm vs fp32 oracle maxabs 2.475e-4 / cosine
  1.000000; denorm vs bf16 golden cosine 0.998402 (== the oracle's own agreement)** → PASS.
- **CRITICAL GOTCHA — t_seq / guidance-interval must be float64.** torch builds `t_seq` via numpy
  float64; stage-2's Möbius warp (RT=3) lands a step EXACTLY at t=0.6, the interval boundary. In
  float32 that t = 0.5999999 → CFG is SKIPPED where torch applies it (guidance 7.5 vs 1) and the
  flow ODE amplifies the single flipped step to ~1.3 maxabs (cosine still 0.99). Fix: compute
  `t_seq` + the `g0≤t≤g1` decision in `double`. (Stage-1 RT=5 never hit 0.6, so float32 was fine —
  this only bites stage-2/3b.) Build/run: `./build.sh m2_sampler_test [cuda]`.

### 2026-06-12 — M2 sparse SLat DiT VALIDATED (single forward)
Started M2 (Shape-SLat LR sparse DiT, grid32). The DiT block math == M1 (confirmed:
ModulatedSparseTransformerCrossBlock identical), so it reuses the M1 infra.
- **numpy ref** `tools/m1_ref/slat_dit.py` — reuses ss_dit.py's validated primitives + new
  3D **sparse RoPE** (phases from `coords[:,1:]`: `phase[n,axis*21+f]=coord[n,axis]/10000^(f/21)`,
  3 axes×21 + 1 zero-pad pair = 64; interleaved-complex rotate). **Cross-checked vs the REAL
  torch ElasticSLatFlowModel on CPU (the sparse DiT runs CPU-only — sdpa attn, no conv):
  maxabs 6.0e-6.** Dumps `refs/slat_dit_{x,t,v}.npy` + `slat_coords.npy` (v is **float64** —
  `compare_to_npy` now handles `<f8`).
- **C++/ggml** `slat_dit_graph.hpp` (`build_slat_dit_forward`) — M1 DiT parameterized: in/out
  32, proj_in **2048** (proj_linear 2048→1536), N=1126 voxel tokens, coord-rope consts
  (`fill_sparse_cos_sin`), layout `[ch,N]` (slat_dit_v is token-major c-fastest, no transpose).
  `slat_dit_test`: **CPU v maxabs 6.2e-6, CUDA (tf32 off) 7.6e-4** vs ref — PASS both. M2
  weights `weights_npy/slat_flow_512` (export `export_weights.py slat_flow_512`).
- **REMAINING M2**: full FlowEuler sampler (shape params guidance_rescale **0.5**, rescale_t
  **3.0**, interval [0.6,1.0]) + denorm (`slat*std+mean`, 32-dim `shape_slat_normalization`
  from pipeline.json) → lr_slat[N,32]. Golden reproduction needs the exact stage2 noise
  (`randn(N,32)` — continues the global RNG after stage1, NOT a fresh reseed; dump it from a
  replay or the golden hook). Then the **NAF conditioner** (9×9 neighborhood attn → proj
  2048; heavy net-new) — port AFTER, validating the DiT against golden cond first.

### 2026-06-12 — RUNG-1 DONE ✅: C++/ggml Stage-1 image→coords == torch fp32
Full Stage-1 ported to **real ggml ops** and validated tight against the `tools/m1_ref`
fp32 numpy/torch oracles, on the ggml **CPU backend** (per-op) AND end-to-end on **CUDA**.
Work in `tools/m1_ref/cpp_port/` (standalone ggml programs, NOT the framework runner yet —
that's integration/M5). Build: `./build.sh <test> [cuda]` — CPU links `ggml/build-cpu`
(system g++ glibc2.43 + `-fopenmp`); CUDA links `ggml/build-cuda` (toolchain nvcc12.4 +
g++12.4, sm_86, +`-lcuda -lcudart -lcublas`, run with `LD_LIBRARY_PATH=$TOOL/lib:/usr/lib`).
Weights: npz→per-tensor `.npy` (`unpack_weights.py` → `weights_npy/{dinov3,ss_flow,ss_dec}`);
refs `dump_refs.py` + `torch_stage1_ref.py` (torch fp32 z_s oracle) → `refs/`.

| component (test) | result vs fp32 ref | status |
|---|---|---|
| **proj grid_sample** (`proj_grid_test`) | z_proj maxabs **1.1e-5** vs proj.npy | ✅ CPU |
| **DINOv3 ViT-L/16** (`dinov3_test`) | global **5.7e-6**, patchmap **1.5e-5** | ✅ CPU |
| **SS dense DiT** 1-fwd (`ss_dit_test`) | v **2.7e-5**, block0 rel ~6e-7 | ✅ CPU |
| **SS VAE decode** (`ss_vae_test`) | ss_logits **9.3e-4**; **coords SET-EQUAL golden (1126)** | ✅ CPU |
| **sampler→VAE** (`sampler_test`, golden cond) | z_s vs torch fp32 **2.7e-4**; **coords 1120, IoU 0.9859** | ✅ CUDA |
| **FULL image→coords** (`stage1_e2e`) | cond **1.3e-5**, z_s vs torch **3.4e-4**, **coords 1120, IoU 0.9859** | ✅ CUDA |

**`stage1_e2e cuda` (with `NVIDIA_TF32_OVERRIDE=0`) reproduces the torch fp32 pipeline
EXACTLY: 1120 coords, IoU 0.9859 vs the bf16 golden (1126), ~68s.** The 6-voxel gap to
1126 is the bf16-torso rounding (predicted). Without the tf32 override → 1119/0.9832
(cublas tf32 noise in DINOv3, 1e-2; harmless threshold flips).

TWO CRITICAL FIXES (both bit the sampler, which is the first multi-`graph_compute` path):
- **Repeated `graph_compute` corruption**: if weights AND the compute graph share one
  gallocr pool, gallocr reuses the weight buffers for intermediates → run 1 OK, run 2
  garbage, run 3 NaN. FIX: weights live in a SEPARATE persistent buffer (`ctx_w` +
  `ggml_backend_alloc_ctx_tensors`); gallocr only manages the compute graph (it skips
  pre-allocated weights). Likewise rope cos/sin/freqs are PERSISTENT consts (`const_tensor`
  in `ctx_w`) — gallocr-managed *inputs* get their buffers reused after last-consumer, so
  one-time-uploaded inputs are clobbered on recompute. (`M1Harness` in `m1_ggml.hpp`.)
- **fp32 on CUDA**: `ggml_mul_mat_set_prec(GGML_PREC_F32)` does NOT disable tf32 for
  F32×F32 cublas. Use the runtime env `NVIDIA_TF32_OVERRIDE=0` for correctness-first fp32.

Key port decisions / gotchas (all load-bearing):
- **grid_sample = 4× `ggml_get_rows` + weighted sum** (corner idx+weights are host constants
  from the 3 camera scalars). No custom CUDA kernel needed for correctness; a fused
  grid_sample kernel is a perf-phase option. Camera scalars come from `cam.json`
  (angle 0.7332379, dist 1.3021560) — NOT the proj_cond synthetic case.
- **ggml layout rule**: numpy C-order shape `[d0..dk]` ↔ ggml `ne={dk..d0}` (reverse),
  identical bytes. Linear: torch `[out,in]` → ggml `[in,out]`, `ggml_mul_mat(W,x)`.
- **tanh-GELU MUST be built in fp32** (`0.5*(x+x*tanh(...))`): ggml's `ggml_gelu` uses an
  F16 lookup table on CPU (~1e-3 error) — that was the DiT v 1.7e-3 miss; explicit fp32 → 2.7e-5.
- **qk-RMS-norm** = `ggml_rms_norm(x,1e-12)*gamma` (== `F.normalize(x)*gamma*sqrt(D)`).
- **complex interleaved RoPE** (DiT): `x*COS + rot(x)*SIN`, rot swaps even/odd pairs with
  sign; COS/SIN replicated per pair from saved `rope_phases[4096,64,2]`. (DINOv3 uses the
  half-split rotate_half RoPE on patch tokens only.)
- **conv3d**: `ggml_conv_3d` data layout `[W,H,D,C]` (channel ne3), F32 im2col since F32
  kernel (exact). 5D torch kernel `[OC,IC,KD,KH,KW]` → ggml `[KW,KH,KD,IC*OC]`.
- **pixel_shuffle_3d == base `depth_to_space_3d`** (ported): channel order `(c,p1=D,p2=H,
  p3=W)` exactly matches numpy `c*8+4a+2e+f`. ChannelLayerNorm32 = permute C→ne0, `ggml_norm`.
- DINOv3 has **massive-activation outlier channels** (|x|~1e5) → judge intermediates by
  meanabs, not maxabs (1.5e-2 abs on a 155299 value = ~1e-7 rel).

**NEXT**: M2 — Shape SLat LR sparse DiT (grid32). For batch=1 the "full varlen sparse
attention" is just plain attention over N=1126 voxel tokens (no varlen split); sparse 3D
RoPE = the SAME interleaved complex rope but with per-voxel COS/SIN from coords
(`phase[n,axis*21+f]=coord[n,axis]·freqs[f]`, freqs=`1/10000^(i/21)`, 3 axes×21+1 zero-pad
pair=64); SparseLinear=per-token Linear; proj_in=2048. Block math == M1 (confirmed:
ModulatedSparseTransformerCrossBlock norm1/3 nonaffine, norm2 affine, qk-rms self-attn +
rope, ProjectAttention, MLP). So M2 reuses `ss_dit_graph.hpp` nearly verbatim. Validate the
sparse DiT vs golden cond (`golden_stages/stage2_cond` proj_feats[N,2048]/global) →
lr_slat[N,32] (`stage2_out`); the heavy NAF conditioner (9×9 neighborhood attn) is separate
— port it AFTER, validating the DiT against golden cond first. Then M3→M4→M5.
Harness (`m1_ggml.hpp`) + graph headers (`dinov3_graph`/`ss_dit_graph`/`ss_vae_graph.hpp`)
reusable. GGUF (task #5) deferred — per-tensor .npy is the validated weight format; GGUF
only matters at framework integration (M5).

### 2026-06-11 — M1 Rung-0 reference modules VALIDATED (numpy fp32, CPU)
Per-module fp32 numpy references in `tools/m1_ref/` (weights exported safetensors→npz
via `export_weights.py`: ss_flow/ss_dec/dinov3 + `*_keys.json`; rope_phases kept complex
as [...,2]). Each validated against the real torch math:
- **SS VAE decode** (`ss_vae_decode.py`): z_s golden → ss_logits (median abs 1.75e-2 vs
  fp16-torso golden) → **coords SET-EQUAL to golden, N=1126** ✅ (self-verified).
- **SS DiT + FlowEuler sampler** (`ss_dit.py`): numpy `model_forward` vs the REAL
  `SparseStructureFlowModel` on CPU (single forward, shared cond) = **maxabs 1.5e-5** ✅
  — proves the block math (complex RoPE, qk-rms-norm, share_mod per-block modulation add,
  ProjectAttention, 30 blocks). E2E 12-step-vs-golden tail is correct-but-slow naive numpy;
  killed mid-run to spare swap (cross-check already proved correctness; noise repro =
  `torch.manual_seed(42);randn(1,8,16,16,16)`).
- **DINOv3+proj** (`dinov3_proj.py`): image→(z_global,z_proj) vs golden = maxabs 1.5e-5 ✅
  (reuses validated proj_cond_ref + dinov3 RoPE; gotchas: RoPE on patch tokens only,
  k_proj biasless, plain final F.layer_norm, erf-gelu). [re-confirm in main loop pending]
Gotchas captured in each file's report.

**M1 STAGE-1 E2E COMPLETE ✅ (reference level, 2026-06-12)** — `stage1_e2e.py` chains
image → dinov3_proj → ss_dit(sampler) → ss_vae_decode → coords with NO golden inputs
(only the preprocessed image + cam scalars). Result: cond vs golden 1.5e-5; z_s vs golden
median 3.6e-3 / maxabs 1.23; **coords 1120 vs golden 1126, IoU 0.9859**.
DEFINITIVE fp32 closure (`torch_stage1_ref.py`): the REAL torch model + REAL
FlowEulerGuidanceIntervalSampler in fp32 (same seed-42 noise + golden cond) produces the
**IDENTICAL** result — z_s maxabs 1.226/median 3.607e-3, **coords 1120, IoU 0.9859**. So
the fp32 path (mine == torch) lands on 1120; the golden's 1126 is purely its **bf16 torso**
(11 coarse res-32 boundary voxels flip at the occupancy threshold — bf16 rounding, not a
bug; these get refined away in Stage 2/3). The numpy reference is bit-faithful to the real
fp32 pipeline incl. the 12-step sampler. naive-numpy E2E = ~63 min (attention/python-loop
bound); torch fp32 = ~9.4 min. Perf is the C++/ggml port's job, not the ref's.

**REMAINING:** C++/ggml port of Stage-1 (Rung-1 — reuse base kernels per the inventory;
only grid_sample + DINOv3 encoder are genuinely new); then M2 (shape sparse-DiT, full
varlen sparse attn + sparse RoPE) → M3 (shape sparse-VAE: sparse conv DONE + up/C2S +
subdiv) → M4 (O-Voxel mesh) → M5 (full geometry GLB). All M2+ goldens already captured in
golden_stages/. NOTE: ran two heavy torch subagents in parallel earlier → contended w/
owner's GPU test + swap → [[feedback_no_heavy_parallel_subagents_during_gpu_test]]; heavy
CPU work strictly sequential/one-at-a-time.

### 2026-06-11 (later) — M0 CAPTURED ✅ (GPU run, one decode, miku.png, res 1024)
`golden_stage_runner.py` ran clean (exit 0, full decode + GLB export). All stage
goldens in `tools/sparse_spike/golden_stages/` (409 MB, gitignored) + `vram.json` +
`stages_manifest.json` + copied `configs/`. Every shape matched CONFIGS-RESOLVED
predictions (slat=32ch, SS proj dense[1,4096,1024], shape proj 2048, z_s=8). Goldens
verified loadable, value ranges sane (coords in-grid, verts in aabb[-0.5,0.5]).

**Per-stage peak VRAM (MiB alloc / reserved) — the low_vram budget the port must hold:**
| stage | alloc | reserved | what |
|---|---|---|---|
| stage1_cond | 1212 | 1244 | DINOv3 ss (grid16, no NAF) |
| stage1 | 2805 | 2848 | SS flow(30blk) + VAE decode |
| stage2_cond | 4384 | 4628 | DINOv3 shape_512 + NAF |
| stage2 | 2759 | 4630 | shape LR flow |
| stage3a | 1433 | 1562 | upsample decoder |
| **stage3b_cond** | **6334** | **7632** | DINOv3@1024 + NAF (HR cond) |
| **stage4_cond** | **6339** | **7634** | tex cond (PHASE-2) — **PEAK** |
| stage4 | 2993 | 7642 | tex flow |
| stage5_decode | 2371 | 2624 | shape decode + O-Voxel mesh |
→ **Peak ≈ 6.3 GB alloc / 7.6 GB reserved at the DINOv3@1024+NAF conditioner** (matches
the map's "Stage 3b heaviest"). Geometry-only (skip tex stage4) peak is the same 6.3/7.6
at stage3b_cond. Perf-phase target 7.5 GB co-resident is plausible. NAF is the spike.

**Real data-flow (miku.png) — golden token/voxel counts for sizing the port:**
SS coords 1126 @res32 → LR slat[1126,32] → upsample×4 → hr_coords[382554,4] @lr-res512
→ quantize/unique → **4734 tokens** @grid64 → HR shape_slat[4734,32] → mesh 1,547,076
verts / 3,251,686 faces (pre fill_holes); final 1,549,929 / 3,269,048 (post + GLB).
subs (4 levels): 4734 → 20533 → 87442 → 367673 voxels. (hr_coords N=382554 ≈ the
sparse-conv spike's heaviest layer N=382533 — consistent.)

### 2026-06-11 (earlier) — CPU prep (GPU was busy; no GPU touched)
- **M0 dumper BUILT (ready to fire, GPU-gated):** `tools/sparse_spike/golden_stage_hook.py`
  + `golden_stage_runner.py`. Monkeypatches the pipeline to dump EVERY stage boundary
  (pre image, SS cond global/proj, z_s, ss_logits, coords, lr_slat, hr_coords,
  shape_slat, tex_slat, mesh verts/faces+subs) + **per-stage peak VRAM** (`vram.json`)
  + copies all configs. Saves incrementally (survives a PHASE-2 GLB-export crash).
  Both byte-compile clean. **To run (coordinate GPU first):**
  `cd /mnt/hdd/3d/avatar-shootout/Pixal3D && source .venv/bin/activate &&
   ATTN_BACKEND=sdpa python <spike>/tools/sparse_spike/golden_stage_runner.py
   --image /mnt/hdd/3d/avatar-shootout/assets/miku.png --resolution 1024`
- **`pipeline.json` + ALL per-model configs FOUND ON DISK** (no GPU/network needed):
  `~/.cache/huggingface/hub/models--TencentARC--Pixal3D/snapshots/0b31.../{pipeline.json,
  ckpts/*.json}`. Open Questions #1/#2/#3 from the map are RESOLVED → **`CONFIGS-RESOLVED.md`**.
  Key map corrections: **shape/tex SLat latent = 32 ch (not 8)**; SS latent z_s = 8;
  ALL flow models **pe_mode=rope, share_mod=true, attn_mode='full'** (NO windowed attn
  on geometry path → primitive #9 NOT needed for M1–M5); flow class is
  `ElasticSLatFlowModel` (inference == `SLatFlowModel.forward`); decoder blocks =
  `SparseConvNeXtBlock3d` + `SparseResBlockC2S3d` (channel↔spatial, not pixel-shuffle);
  tex guidance_interval = [0.6,0.9] (vs [0.6,1.0]).
- **M1 spec WRITTEN:** `M1-STAGE1-PORT-SPEC.md` — exact block math (ModulatedTransformer
  CrossBlock + ProjectAttention), proj grid_sample/unproject recipe, SS VAE +
  pixel_shuffle_3d, sampler params, reuse inventory. Mechanical to port once M0 goldens
  exist. NOTE: norm2 in the DiT block is AFFINE (norm1/norm3 non-affine).
- **Base-repo kernel inventory (Explore):** almost ALL of M1 is reusable in the
  sd.cpp/ggml base — `ggml_ext_conv_3d`, affine+non-affine LayerNorm (`ggml_ext_layer_norm`),
  `GroupNorm32`, cross-attn (`ggml_ext_attention_ext`), generic RoPE-apply
  (`Rope::apply_rope` / `ggml_rope_pe`), per-head `RMSNorm`, SiLU/GELU, `Linear`,
  `ggml_ext_timestep_embedding`, adaLN modulation (`z_image.hpp`), pixel-shuffle
  (`PixelShuffleND`) + `depth_to_space_3d`, GGUF load+convert (`src/convert.cpp`).
  **GAPS / net-new to author:** (1) **grid_sample** (bilinear/border/align_corners=False —
  no dedicated op; only `ggml_upscale`); (2) **DINOv3 encoder** (only CLIP ViT exists —
  template to adapt: add 2D-axial RoPE + 4 reg tokens + LayerScale); (3) ProjectAttention
  (trivial compose = cross-attn + per-block `proj_linear` add). ggml submodule @ 19727d01.
- **PROJ-COND net-new RISK RETIRED (GPU-free), `tools/proj_cond/`:** the camera-unproject
  is host scalar arithmetic (grid + 3 camera scalars are fixed per stage → it produces the
  grid_sample coords on the host; only the bilinear gather needs a GPU kernel). Built
  numpy ref (`proj_cond_ref.py`) + **validated bit/tol-exact vs the REAL torch `ProjGrid`
  on CPU** (`test_proj_cond.py`, CUDA hidden) across SS grid16 / shape grid32 / grid64 —
  **maxabs ~1e-5** (fp32 ordering noise). C++ mirror (`proj_cond.cpp`, 4x4 inv + CPU
  grid_sample) vs the dumped torch golden = **maxabs 1.53e-5 PASS**. So the proj geometry +
  bilinear sampler are locked; the only remaining proj work is a CUDA grid_sample kernel
  (validate vs `golden_ref/proj_case0/` + the M0 `stage1_cond` real DINOv3 map).
- **NOT touched:** no DiT/VAE C++ yet (correctness-first needs the M0 goldens — author
  after the GPU capture). Nothing committed (worktree still UNCOMMITTED).

## First actions for the fresh agent
1. Read `E2E-PORT-MAP.md` (the detailed stage map) + this doc + the 3 spike docs.
2. M0: write the staged golden dumper (generalize `tools/sparse_spike/golden_hook.py`),
   run ONE Pixal3D decode to capture all stage boundaries + `pipeline.json` + per-stage
   VRAM. (Needs GPU once — coordinate with owner.)
3. M1: port Stage 1 (grid-16 SS) and validate vs golden. Then march the ladder.
Keep the handoff docs updated as you go. Save durable findings to memory
(`/home/dbrain/.claude/projects/-home-dbrain-dev-kobbler/memory/`).
