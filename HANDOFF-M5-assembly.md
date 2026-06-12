# HANDOFF — Phase A: image → untextured mesh GLB, as a CLI from GGUF

**The product goal is a usable CLI** (`pixal3d --model m.gguf --image in.png --out out.glb`), matching
the Python library. As of 2026-06-12 **every geometry COMPONENT is ported + validated** (see
E2E-PORT-KICKOFF PROGRESS LOG top entries). Phase A = ASSEMBLY + PACKAGING, in 3 steps:
  - **A1** the chain driver (this doc, §"The chain") — wire the validated programs into `geometry_e2e.cpp`
    (image → mesh), validate vs a Python fp32 E2E oracle (IoU, per the fp32-vs-fp16 M1 lesson).
  - **A2** GGUF (§"GGUF") — replace the per-tensor `.npy` dev format with GGUF load (the bridge to product).
  - **A3** CLI + front-end (§"CLI") — turn the driver into `pixal3d`, add host rembg + `--fov`.
The driver from A1 IS the CLI core; do A1 first (proves the geometry), then A2/A3 make it a real tool.

## The chain (each step = a VALIDATED program/graph to reuse)
1. **Stage1** (image→coords) — `stage1_e2e.cpp` is already a working E2E program. image@512 →
   `dino::build_dinov3(CFG512)`(ss) → proj_grid16 → SS dense DiT (12-step FlowEuler, GS7.5/GR0.7/
   RT5.0/[0.6,1.0]) → SS VAE decode → coords[N1,4] @grid32. (RT=5 ⇒ float32 t_seq OK here.)
2. **stage2 cond** — `dino::build_dinov3(CFG512)` patchmap + `naf::build_naf_forward(CFG512)` hr →
   proj_grid32 at the Stage1 coords (see `stage2_cond_test.cpp` for the proj host-math) → cond
   `{global[5,1024], proj[N1,2048]}`. (lr branch tight 2e-5; hr branch = NAF.)
3. **M2 DiT** — `slat_dit_graph.hpp build_slat_dit_forward` + the sampler loop in `m2_sampler_test.cpp`
   (GS7.5/GR0.5/RT3.0/[0.6,1.0]/12, **double t_seq** — RT3 hits t=0.6) + per-ch denorm (shape_slat_norm)
   → lr_slat[N1,32].
4. **M3a** — `sparse_vae.hpp` upsample (see `m3a_upsample.cpp`) → hr_coords → quantize grid64:
   `q=((hr+0.5)/512*64).int()`, `.unique` → coords[M,4] @grid64. (bit-exact.)
5. **stage3b cond** — `dino::build_dinov3(CFG1024)` patchmap + `naf::build_naf_forward(CFG1024)` hr →
   proj_grid64 at the M3a coords (see `stage3b_cond_test.cpp`) → cond[M,2048].
6. **M3b DiT** — same as M2 (`build_slat_dit_forward`, weights slat_flow_1024) + sampler (see
   `m3b_sampler_test.cpp`) + denorm → shape_slat[M,32].
7. **M4** — `sparse_vae.hpp` decoder forward + `flexible_dual_grid_to_mesh` (see `m4_mesh.cpp`) →
   verts/faces. Write an OBJ/PLY (trivial) or GLB.

## The only net-new plumbing (everything else is validated reuse)
- **Cross-stage noise**: torch draws, in order, `manual_seed(42); randn(1,8,16,16,16)` [stage1];
  `randn(N1,32)` [stage2]; `randn(M,32)` [stage3b] — on **CPU** (the refs reproduced the golden with
  CPU seed-42). N1 and M are the **COMPUTED** coord counts (differ from golden: C++ Stage1 gives ~1120
  not 1126). So the noise SHAPE is data-dependent → must be generated AFTER each stage's count is known.
  Options: (a) **port torch CPU randn** = MT19937 (std::mt19937? NO — torch uses its own mt19937 + a
  specific Box-Muller/normal_distribution; match `at::normal_` exactly) — bounded but fiddly; OR
  (b) a tiny torch helper called from the driver to emit the 3 noise tensors for the computed counts
  (hybrid; simplest for a first validated E2E). Recommend (b) first, then (a) for self-containment.
- **grid64 quantize+unique** (step 4): host sort/unique of int4 coords (sparse_vae has the machinery).
- **GLB writer**: verts[V,3] f32 + faces[F,3] i64 → a single-mesh .glb (or .obj for a quick eyeball).
  Untextured. (Textured GLB w/ UV/atlas is M6.)

## Validation strategy
- **Oracle**: run the REAL Python pipeline in **fp32 on CPU** (force all torsos fp32, tf32 off, conv
  monkeypatched to the spike fp32 path like stage3a/stage5 captures) end-to-end → fp32 mesh. Slow
  (~tens of min) but it's the tight target. Compare the C++ chain mesh to it: coords SET-IoU (~1.0 if
  same noise), verts tight, faces near-exact.
- **Sanity**: vs the existing fp16 golden stage5_mesh (IoU ~0.99 = fp16/fp32 boundary noise).
- If the noise matches (same CPU seed-42 over the computed counts) and every stage is validated, the
  C++ chain mesh == the fp32 oracle within fp32-accum noise (a few boundary voxels), == the fp16 golden
  within IoU ~0.99.

## A2 — GGUF (replace the per-tensor .npy dev format)
The ports load `weights_npy/<model>/<key>.npy` (one file per tensor; the `*_capture.py`/`unpack_weights.py`
exports). For a shippable CLI, convert the source safetensors → GGUF and load in C++.
- **Models to pack** (HF cache snapshot `0b31f916...`): `ss_flow`, `ss_dec`, `dinov3` (camenduru/dinov3-vitl16),
  `slat_flow_img2shape_512`, `slat_flow_img2shape_1024`, `shape_dec`, NAF (valeoai/NAF) — and for M6
  `slat_flow_imgshape2tex_1024`, `tex_dec`. Either one GGUF with name-prefixed tensors, or one per model.
- **Conv weights**: the sparse-conv weights must be packed in the spike `[V=27, Cin, Cout]` layout (the
  `transpose(1,2,3,4,0).reshape` from stage3a_capture / golden_hook). Stash dims in GGUF kv-metadata.
- **rope_phases** (ss_flow DiT) is complex → keep as a trailing `[...,2]` real/imag tensor (as the npz export does).
- **Loader**: base repo `src/convert.cpp` / GGUF reader (sd.cpp/ggml). Map GGUF tensors → the same names the
  graph builders request via `M1Harness::weight(key)` (i.e. add a GGUF-backed `weight()` path to the harness,
  keeping the `.npy` path for dev validation). Validate: GGUF-loaded run == `.npy`-loaded run, bit-identical.
- Keep fp32 for now (Phase A correctness). GGUF Q-types are a PERF-phase lever (Phase C).

## A3 — CLI + front-end
`pixal3d --model <gguf> --image <png> --out <glb> [--fov <rad> | --cam <angle> <dist> <scale>] [--resolution 1024]`
- **Camera scalars** (camera_angle_x, distance, mesh_scale) are inputs to every proj cond (host scalar math).
  The golden used cam.json (0.7332 / 1.3022 / 1.0, from MoGe). Recommended cut-line: accept `--fov` (→ derive
  camera_angle_x) + sane defaults, OR `--cam`. Porting MoGe-2 (camera estimation) is optional/host-side.
- **rembg / matting**: Pixal3D runs BiRefNet (briaai/RMBG-2.0) to crop+matte before the cond models (the golden
  used `pre/preprocessed.png`). Recommended cut-line: host-side (accept an already-matted RGBA/RGB PNG), or a
  thin host-Python pre-step. Only port BiRefNet for a self-contained "raw photo → GLB" binary.
- The image preprocessing the cond models expect: resize to image_size (512/1024) LANCZOS, /255; DINOv3 gets
  ImageNet-normalize, NAF gets the [0,1] guide (see stage3b_cond_capture.py for the exact steps).
- Output: glTF 2.0 `.glb` (binary, single mesh). Untextured for Phase A (positions + indices); M6 adds
  UV + a base-color/metallic-roughness texture. (`write_ply` in sparse_vae.hpp is the quick-eyeball stand-in.)

## Then M6 (texture = feature-complete)
tex SLat DiT (`slat_flow_imgshape2tex_dit_1_3B_1024`, **in_ch 64** = 32 noise ‖ 32 shape_slat re-normed,
out 32; CFG off GS1.0, interval **[0.6,0.9]**; reuses build_slat_dit_forward) + tex decoder
(`tex_dec_next_dc_f16c32_fp16`, **out 6** PBR, **pred_subdiv=false** → reuses shape's `subs` as guide_subs;
`·0.5+0.5`; reuses the M3a/M4 sparse-VAE backbone) + NAF@1024 (reused) + **textured-GLB bake**
(`o_voxel.postprocess.to_glb` — UV unwrap + atlas + sample attrs from the volume; the largest net-new
M6 piece, _C.so-backed → golden-validate). tex cond = get_proj_cond_shape(tex_1024) (== shape_1024
recipe). Goldens already captured: golden_stages/stage4_{cond,out}.
