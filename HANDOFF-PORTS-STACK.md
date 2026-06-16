# HANDOFF — the C++/ggml port stack for the image→rigged-avatar pipeline

**Date:** 2026-06-16 · **Branch:** `spike/sparse-conv-3d` · **Worktree:** `~/dev/longcat-sparse-spike`
All C++ ports live in `tools/m1_ref/cpp_port/` (call it `$CP`), reuse `m1_ggml.hpp` (M1Harness:
lin/layernorm/attention/gelu/rope) + the sparse-conv spike, and validate **against an fp32 oracle**
(`NVIDIA_TF32_OVERRIDE=0`, eager attn). Build via `$CP/build.sh <tool> [cuda]` (Docker/C++ builds are
fine on this host — [[feedback_no_build_on_server]] is Rust-only). GPU = one RTX 3060 12GB (coordinate).
Big weights/goldens live on **/mnt/hdd** (SSD is for final models — owner pref).

## The pipeline today — what is C++ vs still Python

```
image ─►[matte]─► pixal3d (geom+PBR) ─► UltraShape (clean) ─► P3-SAM (segment) ─► per-part decimate ─► [TEXTURE] ─► [RIG]
        C++        C++ (gguf)            PYTHON/docker         PYTHON/docker        C++ obj_decimate     2 paths      SkinTokens
                   + MoGe FOV = PYTHON                         + py split/recombine                                  (partial C++)
```

The **geometry generator (pixal3d)** and the **decimator (obj_decimate)** are already C++/ggml. Everything
else with a model in it is still Python. This doc is the ladder to finish the stack. Ordered by
value × tractability.

---

## SHARED PREREQ — a C++ GLB *reader* (unblocks several ports)
The port has GLB **writers** (`glb_writer.hpp`, `glb_textured.hpp`, `glb_packed.hpp`) but **no reader**.
That's the single reason `per_part_decimate.py`, the stage hand-offs, and the texturing front-end still
need trimesh. A small glTF-2.0 reader (parse JSON chunk + BIN accessors for POSITION/indices/UV; ~a day,
no deps or vendor `cgltf`) removes Python from every mesh-I/O seam below. **Do this first** — it pays off
across ports 1, 4, 5.

---

## 1. NATIVE MESH TEXTURING (TRELLIS.2) — highest value, mostly already ported
**What:** re-generate a texture on an *arbitrary* mesh's own voxels (no reproject, no thin-feature bleed).
Validated 2026-06-16 (owner: "much better"). Full writeup: [[project_pixal3d_native_mesh_texturing]].
**Status:** runs in Python via the dockerized `pixal3d-tex` image (`$CP/shootout/{Dockerfile.pixal3d-tex,
texture_mesh.py,pixal3d_tex_run.sh}`), weights = `microsoft/TRELLIS.2-4B` texturing set on
`/mnt/hdd/pixal3d_tex`. **Goldens banked** at `/mnt/hdd/pixal3d_tex/golden_{69k,ultrashape,usdense}_*`
(shape_slat feats+coords).
**To port (small!):** the C++ `pixal3d --tex` path ALREADY has the tex DiT (`slat_flow_imgshape2tex_1024`),
tex decoder, and cumesh bake. The ONLY new pieces are:
  - **`shape_slat_encoder`** — the encoder half of the `shape_dec` VAE we already ported (same
    `next_dc_f16c32` arch, sparse U-Net; mirror the decoder with `shape_enc` weights). Convert
    `TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16.safetensors` → gguf (pack_gguf.cpp pattern).
  - **`mesh_to_flexible_dual_grid`** voxelizer (the `o_voxel` op: mesh → voxel_indices + dual_vertices +
    intersected). The port has the *inverse* (`flexible_dual_grid_to_mesh`); port the forward.
  - wire: encode mesh → shape_slat → existing tex DiT (concat-cond on shape_slat) → tex dec → bake.
**Validate** the encoder vs the banked shape_slat goldens (same mesh in → match feats/coords).
**Gotchas:** run inference under no-grad equivalent (the Python OOM'd at 1024 purely from autograd
retention — the C++ has no autograd so 1024 is free). Latent res 512/1024 = which flow (use 1024). Texture
is generative (seeded) — not bit-reproducible vs Python; validate the *encoder* (deterministic), eyeball
the bake. Optional: keep pixal3d's fine-tuned tex flow (local gguf) for the parent style vs TRELLIS.2 base.

## 2. UltraShape — refiner (clean/watertight/hole-fill pre-pass)
**What:** image + coarse mesh → refined watertight mesh (sparse-voxel DiT + marching cubes). Densifies
~7.5×; the clean source for segment+decimate. [[project_pixal3d_dense_to_lowpoly]].
**Status:** Python/docker (`$CP/shootout/{Dockerfile.ultrashape,ultrashape_run.sh}`, ckpt
`/mnt/hdd/3d/avatar-shootout/_weights/ultrashape/ultrashape_v1.pt`). Patched runtime files in the repo
(low_vram CPU-keep, cpu generator) — keep them.
**To port:** same **sparse-voxel family** as the pixal3d decoder we ported (VAE + a ~3B DiT + conditioner,
all-in-one ckpt). Heavy but architecturally known. Reuse the sparse-conv spike + `m1_ggml.hpp` attention.
The win is removing a brittle 3060-OOM-prone docker stack. **Feed it the dense 3.46M shell** (via
`dump_to_glb` / `pixal3d --decimate 0`), not the 138k sloppy-decimate — modestly cleaner output (verified).
**Validate** vs the Python refined mesh (geometry, not bit-exact — it's a diffusion sample; compare
chamfer / render).

## 3. MoGe — "the FOV thing" (camera estimation for arbitrary images)
**What:** monocular geometry (`Ruicheng/moge-2-vitl`) → camera FOV + distance, so pixal3d frames a *wild*
photo correctly. Python only (`inference.py: get_camera_params_wild_moge`, `distance_from_fov`). The C++
`pixal3d` takes `--fov`/`--cam` **manually** (defaults to the miku cam) — fine for controlled input,
wrong for arbitrary photos.
**To port:** a ViT-L depth/geometry net → FOV solve. Smallest of the model ports (single ViT + a
closed-form camera fit). Alternatively keep it as a thin pre-step that just emits `--fov`/`--dist` numbers
(it's a one-shot scalar estimate, not a per-frame pipeline stage) — lower priority than 1/2.

## 4. P3-SAM — part segmentation (hands isolate for region-adaptive decimate)
**What:** any-mesh → per-face part labels; hands isolate as parts so decimate keeps fingers.
[[project_pixal3d_dense_to_lowpoly]] (P3-SAM VALIDATED section).
**Status:** Python/docker (`$CP/shootout/{Dockerfile.p3sam,p3sam_run.sh}`), Sonata backbone +
patched `auto_mask*.py` (prompt_bs threading) — keep. `PROMPT_BS=2` for 3060.
**To port:** same Tencent/VecSet family (most portable per the research) — Sonata (point-transformer)
backbone + a seg head. Real work; do after 1/2. Until then it stays the docker oracle.

## 5. per-part decimate split/recombine (mesh-op glue)
**What:** `per_part_decimate.py` splits the segmented mesh by part label, shells each to the C++
`obj_decimate` (the actual QEM is already C++), recombines, and emits the 69k (+ optional dump_mesh bins).
**To port:** pure mesh I/O + label-split + concat — trivial in C++ **once the GLB reader (prereq) exists**.
The decimation compute is already C++. This is the cheapest python removal after the reader lands.

## 6. SkinTokens (TokenRig) — rigging — PARTIAL, finish it
**What:** geometry-agnostic auto-rigger: mesh → 567 tokens → 60 joints + skin weights [V,60].
**Status (HANDOFF-F):** **R1 (VecSet encoder) + R2 (Qwen3-0.6B AR core) + R5 (skeleton de-tokenizer)
ported & validated** (CPU/CUDA vs fp32 golden; R5 bit-exact). The hard/novel GPU primitives are proven.
Files in `$CP`: `vecset_encoder.hpp`, `qwen3_forward.hpp`, `detok_r5.cpp` (+ `capture_*.py` in the
SkinTokens repo). Golden source `/mnt/hdd/3d/avatar-shootout/SkinTokens` (ckpt
`experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt`).
**Remaining (R3/R4/R6/R7 — "known-portable engineering"):** the generate loop tying R1+R2 (AR sampling
into the token stream), the **skin-weight decoder**, and the **rigged-GLB writer** (joints+weights+skel).
Read `HANDOFF-RIGGING-skintokens.md` (R0–R7 spec) + `HANDOFF-F-skintokens-port-R1-R2-R5-done.md`. See
[[project_3dgen_cpp_port]], [[project_avatar_rig_path]].

---

## Suggested order
0. **GLB reader** (unblocks 1/4/5). →
1. **Native texturing** (encoder + voxelizer onto the existing C++ tex path — smallest model port, biggest
   quality win, goldens ready). →
6. **Finish SkinTokens** (only engineering left, gets us to a rigged asset). →
5. **per-part split/recombine** (trivial post-reader). →
2/4. **UltraShape / P3-SAM** (the two heavy family ports). →
3. **MoGe** (or leave as a scalar pre-step).

## Conventions (all ports)
- Validate vs **true-fp32** oracle (`.float()` + eager attn, `NVIDIA_TF32_OVERRIDE=0`), NOT tf32/bf16.
- gguf pack via `pack_gguf.cpp`; standalone load via `gguf_reader.hpp` (env `PIXAL3D_GGUF_DIR`).
- RENDER and look — never judge geometry/texture from logs ([[feedback_no_declaring_winners]]).
- Persistent-weights buffer (gallocr recompute→NaN), fp32 tanh-GELU, `t_seq` float64 — see
  [[project_3dgen_cpp_port]] gotchas.
- Generative stages (texture flow, UltraShape DiT) are **seeded, not bit-reproducible** vs Python — validate
  the deterministic encoders/decoders bit-wise, eyeball the sampled output (render ≥2 seeds for A/B).
