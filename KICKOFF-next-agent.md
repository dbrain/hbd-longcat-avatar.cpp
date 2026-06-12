Continue the C++/ggml port of Pixal3D (TRELLIS.2). The geometry + texture pipeline is DONE and
FEATURE-COMPLETE: `pixal3d --model <gguf> --image <png> --out <glb> [--tex]` produces an untextured
or textured 3D model, pure C++/ggml, from GGUF weights, matching the Python library. **Your job now,
in order: (1) a PERFORMANCE RUN — make it fast; (2) finish the faithful texture bake (UV-atlas);
(3) head toward RIGGING (the north-star).** Run FULLY AUTONOMOUSLY (owner away; GPU + CPU 100% free).
Decide, do, profile/golden-validate, document, continue. Only halt if genuinely hard-stuck.

START by reading IN FULL, in /home/dbrain/dev/longcat-sparse-spike/:
  PERF-NOTES-pixal3d.md (the perf breakdown + prioritized levers — your immediate roadmap)
  · E2E-PORT-KICKOFF.md PROGRESS LOG (top ~4 entries: Phase A + M6 — what's validated, the gotchas)
  · HANDOFF-NEXT.md (★ THE PRODUCT GOAL + FULL SCOPE LADDER) · CONFIGS-RESOLVED.md.
  Memory: project_3dgen_cpp_port, project_avatar_rig_path, feedback_correctness_before_perf,
  reference_subagent_background_stall, reference_ncu_docker_syadmin, feedback_no_workflow_for_basic_research.

DONE + VALIDATED — do NOT redo: spike sparse-conv · M0–M5 geometry · A1 chain (geometry_e2e.cpp) ·
A2 GGUF (all 9 models packed `weights_gguf/`, BIT-EXACT vs .npy; harness `gguf_fetch` + standalone
`gguf_reader.hpp`; env `PIXAL3D_GGUF_DIR`) · A3 CLI (`pixal3d.cpp` + `image_io.hpp` Lanczos +
`pixal3d_chain.hpp::run_geometry`) · M6 textures (tex DiT cosine 1.0, tex decoder maxabs 4.7e-6 vs
fp32 oracles; `svp::m6_tex_decode`; `--tex` → teal Miku, COLOR_0 vertex base_color). Build:
`cd tools/m1_ref/cpp_port && ./build.sh <test> [cuda]`. CLI: `./build.sh pixal3d cuda`, run
`LD_LIBRARY_PATH=/mnt/hdd/3d/avatar-shootout/toolchain/lib:/usr/lib NVIDIA_TF32_OVERRIDE=0 ./pixal3d --model weights_gguf ...`.
A live web compare page (mine vs golden, textured + geometry tabs) is served at
http://10.0.0.208:8011/compare.html (python http.server on :8011 from cpp_port/).

THE WORK, in order:

PHASE C — PERFORMANCE RUN (make it fast; correctness loosens, judge by E2E mesh IoU vs fp32, NOT tight tol):
  - **Profile FIRST with a MINIMAL run — do NOT keep re-running the full 510s E2E.** The isolated tests run
    each hot stage on golden input WITHOUT the full chain: `m4_mesh` / `m6_tex_decode_test` (the host decode,
    ~110s each — the #1 target), a single DiT forward (slat_dit_test / ss_dit_test), naf/dinov3 tests. Use
    **nsys** (timeline) + **ncu** (kernel: occupancy/DRAM/MMQ) on these. ncu in the toolchain/docker needs
    `--cap-add SYS_ADMIN` (memory reference_ncu_docker_syadmin). Bank a per-stage VRAM breakdown too
    (golden_stages/vram.json = Python baseline ~6.3GB alloc / 7.6GB reserved).
  - **#1 target = the host sparse-VAE decodes (M3a + M4 + tex ≈ 255s, ~50% of wall).** The spike conv is on
    GPU but the dense per-voxel ops (`svae::linear`/`layernorm`/`silu` over up to 1.47M voxels), the per-level
    `build_nmap` hashmap, and `c2s_grow` all run on CPU OpenMP → the GPU idles between conv launches. Move the
    dense ops to GPU (keep feats RESIDENT across the decode; only pull back the subdiv decisions for coord-
    growth), faster nmap (Morton/sorted vs unordered_map), fuse conv+bias+act. Validate via `m4_mesh` (bit-exact
    for host-only changes) / IoU (for precision changes) — no full E2E per iteration.
  - **#2 = quantize the 4 DiTs (193s, 38%) via GGUF Q-types** (Q8_0 near-lossless / Q4_K) — leverage the A2
    infra: add a `--type` to `pack_gguf.cpp` (quantize 2D matmul weights; keep 1D norms/conv f32); the harness
    `gguf_fetch` + `lin()` (ggml_mul_mat) handle quant weights natively. + re-enable tf32 (drop
    NVIDIA_TF32_OVERRIDE=0). Judge by one E2E mesh IoU ≈ 1.0.
  - **#3 = VRAM 7.5GB co-residency** (quantized → keep resident, skip per-stage reload). Bank all intel back
    into PERF-NOTES-pixal3d.md as you go.

FEATURE-COMPLETE POLISH (faithful texture match):
  - **UV-atlas to_glb** (the deferred large M6 piece): o_voxel's `to_glb` = xatlas `uv_unwrap` + nvdiffrast
    differentiable-raster bake of a 2048² PBR texture atlas from the voxel volume → glTF embedded baseColor +
    metallicRoughness textures (all 6 PBR ch, not just vertex base_color). Goldens stage4_{cond,out}. The
    vertex-color GLB (current) is the working interim. Needs an xatlas + a CUDA rasterizer port — large.
  - **JSON**: swap glb_writer's hand-rolled snprintf for the BUNDLED `thirdparty/json.hpp` (nlohmann; the
    parent fork already uses it in src/model_io/safetensors_io.cpp). Do it alongside the richer UV-atlas glTF.

NORTH-STAR — RIGGING + MOTION (the real end goal: image → textured RIGGED GLB):
  - The product is a mute 3D-game-asset generator: Pixal3D mesh → **SkinTokens rig** (Qwen3-0.6B core → cheap
    ggml port) → body motion (HY-Motion or Mixamo/AMASS retarget). See memory project_avatar_rig_path.
  - **Trial AniRig** (auto-rigging) — owner wants it evaluated; "in theory plugs into this codebase" (shares
    the structured-latent shapes / consumes this mesh output). Scope it: does AniRig take our mesh/GLB → rig?
    What's the port surface (model size, ops, ggml-portability)? This is the next major chain after the
    textured mesh is fast.

NON-NEGOTIABLES / GOTCHAS: correctness validated vs the TRUE-fp32 oracle (real module CPU, conv→spike fp32),
NOT the tf32/bf16 golden; judge E2E by mesh IoU. float64 t_seq (a step lands at t=0.6); persistent-weights
buffer (gallocr recompute→NaN); fp32 tanh-GELU; ggml_soft_max big dim in ne01; randn uniform<double>=
random64()+log1p; tex decode oracle needs GPU VISIBLE (flex_gemm Triton import); C2S CHILD-MAJOR; conv weight
[Co,Kd,Kh,Kw,Ci]→spike [27,Cin,Cout]. WORKING STYLE: parallelize PREP w/ read-only Explore subagents;
SERIALIZE heavy validation in the MAIN loop (single GPU; one torch oracle at a time; subagents DEADLOCK on
run_in_background → drive long builds/runs from the main loop w/ run_in_background:true). No multi-agent
Workflow for research. C++/CUDA builds fine; NO Rust builds. No pkill -f, no rm-globs. Long jobs
run_in_background:true, not detached &. PREFER minimal "backdoor scraping" profiling over long full runs.
Worktree UNCOMMITTED-but-checkpointed (commit when it helps). Keep PROGRESS LOG + PERF-NOTES + memory current;
note exactly where you stopped.

DEFINITION OF DONE (this run): a measurably FASTER pixal3d (profiled hot-spots fixed, intel banked), and clear
forward motion toward the UV-atlas bake + the rigging/AniRig evaluation. Get as far as possible.
