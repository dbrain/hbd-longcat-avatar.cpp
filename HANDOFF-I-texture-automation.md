# HANDOFF-I — automate the image→textured-lowpoly pipeline (the texture stage + single coordinate frame)

**Goal:** turn the *validated-but-hand-stitched* `image → riggable low-poly with fingers + texture` workflow
into a **fully automated, reproducible pipeline** — no manual one-off glue. The geometry half is DONE and
proven; the **texture-onto-final-mesh stage is the open problem**, blocked by a coordinate-frame mismatch
that must be solved automatically (single canonical frame), not by hand-aligning meshes.

**HARD RULE (owner, load-bearing):** *everything automated, no manual stitching.* No hand-typed coordinate
alignments, no manually-written binary dumps, no swapping intermediate files by hand. Every step = a committed,
parameterized component. **Python is allowed ONLY as the thin wrapper around the not-yet-ported models
(P3-SAM, UltraShape)**; all mesh ops / decimate / normalize / bake / format-conversion are real stages (C++
for prod via the pixal3d port; a committed driver script to orchestrate). See
`memory/feedback_automate_no_manual_stitching.md` + `feedback_no_build_on_server.md`. C++/docker builds are
fine on-server; NO Rust builds. GPU is the RTX 3060 12GB (shared — coordinate, but currently free).

Work in the worktree `~/dev/longcat-sparse-spike` (branch `spike/sparse-conv-3d`), tooling in
`tools/m1_ref/cpp_port/` (call it `$CP`). Commit scripts there (`$CP/.gitignore` is `*` + whitelist
`.cpp/.hpp/.sh/.py` + the `shootout/` dir); .md at repo root. Read first: this doc, `HANDOFF-H-geometry-eval-results.md`,
and memory `project_pixal3d_dense_to_lowpoly`, `feedback_automate_no_manual_stitching`.

---

## 1. THE VALIDATED PIPELINE (each stage works; commands below are the ground truth)

```
image.png ─► [matte] ─► pixal3d ─┬─ geometry .glb ─► UltraShape ─► P3-SAM ─► per-part decimate ─► [TEXTURE BAKE] ─► asset
                                 └─ --tex: PBR + dump_*.bin (the colour source)
```
End result so far: **1.04M→69,404-face Miku, fingers intact, 1.3 MB** (geometry ✅). Texture: generation ✅,
transfer-onto-69k ❌ (§3).

### Inputs / assets
- Test image: `10.0.0.163:~/Downloads/miku-try-this.png` (RGBA 640×1024, arms-out A-pose — **USE A/T-POSE
  inputs**: P3-SAM only isolates hands when limbs are spatially separated). `scp` it to `/mnt/hdd/3d/avatar-shootout/_shootout_out/`.
- Working/output dir: `/mnt/hdd/3d/avatar-shootout/_shootout_out/` (call it `$OUT`).
- pixal3d weights: `$CP/weights_gguf` (geometry + tex DiT). Build: `cd $CP && ./build.sh pixal3d cuda`.

### Stage commands (validated this session)
**(0) Matte** — pixal3d wants a SQUARE BLACK-bg RGB matte (like the turtle `$CP/prep_test_matte.png`, 944²).
From the RGBA source: crop to alpha bbox → composite on black → pad to square (+~10% margin). *This is
currently an inline python heredoc — MAKE IT A COMMITTED STAGE* (`$CP/shootout/make_matte.{py or a C++ tool}`).
Output e.g. `$OUT/miku_try_matte.png`.

**(1) pixal3d geometry** (~350s @1024, 8.6GB):
```
cd $CP && ./pixal3d --model weights_gguf --image $OUT/miku_try_matte.png --out $OUT/miku_try.glb --ply
```
(`--resolution 1536` exists but is a WEIGHTS-limit dead end → 0-face mesh; stay at 1024. See HANDOFF-H §1.)

**(1b) pixal3d --tex** (PBR + the reproject dump; ~560s, 8.2GB):
```
cd $CP && PIXAL3D_FORCE_UVATLAS=1 PIXAL3D_DUMP_BAKE=1 ./pixal3d --model weights_gguf \
  --image $OUT/miku_try_matte.png --out $OUT/miku_try_tex.glb --tex --texsize 2048
```
Writes a textured GLB **and** the bake dump to `$CP/`: `dump_dense_{v,f}.bin` (the colour-source dense shell),
`dump_pbr_{f,c}.bin` (per-voxel 6-ch PBR volume), `dump_mesh_{v,f}.bin` (pixal3d QEM), `dump_bake.txt`
(`NVq NFq NP`), `dump_dense.txt` (`NVd NFd`). Formats: verts=float32, **faces=int64**.

**(2) UltraShape refine** (clean + fill holes; docker `ultrashape`; ~CPU-load-slow + ~50 DiT steps):
```
cd $CP && ./shootout/ultrashape_run.sh $OUT/miku_try.glb $OUT/miku_try_matte.png mikutry
# -> $OUT/ultrashape_mikutry/miku_try_matte_refined.glb  (1.04M f, watertight)
```

**(3) P3-SAM segment** (docker `p3sam`; hands isolate as parts):
```
cd $CP && POINT_NUM=100000 PROMPT_BS=2 ./shootout/p3sam_run.sh \
  $OUT/ultrashape_mikutry/miku_try_matte_refined.glb mikutry_refined
# -> $OUT/p3sam_mikutry_refined/auto_mask_mesh_final.glb + auto_mask_mesh_final_face_ids.npy (per-face part id)
# 18 parts, 95.6% coverage; hands = the small parts at extreme |x|, mid-y (e.g. p184 right / p192 left)
```

**(4) Per-part region-adaptive decimate** (C++ QEM via `obj_decimate`; python only splits/recombines):
```
cd $CP && /mnt/hdd/3d/avatar-shootout/Pixal3D/.venv/bin/python shootout/per_part_decimate.py \
  $OUT/p3sam_mikutry_refined/auto_mask_mesh_final.glb \
  $OUT/p3sam_mikutry_refined/auto_mask_mesh_final_face_ids.npy  $OUT/miku_lowpoly.glb
# importance tiers: HAND keep 0.80, HAIR 0.025, LIMB 0.10, FOOT 0.12, HEAD 0.15, BODY 0.05 -> 1.04M->69k
```
(`per_part_decimate.py` is committed in `shootout/`. The split/recombine should become a C++ pipeline stage
for prod — it's mesh-label split + concat, the pixal3d port has the GLB writers + qem.hpp.)

---

## 2. DOCKER + WEIGHTS (built/fetched this session; reuse, don't rebuild)
- **Image `p3sam`** (`$CP/shootout/Dockerfile.p3sam`): P3-SAM seg. nvidia/cuda:12.4.1-cudnn-devel + py3.10/cu124,
  torch2.4, spconv-cu124, torch_scatter/cluster, flash_attn, chamfer3D, vendored Sonata. Weights:
  `/mnt/hdd/3d/avatar-shootout/Hunyuan3D-Part/P3-SAM/weights/p3sam/p3sam.safetensors` (fetched); Sonata
  auto-downloads once to `/mnt/hdd/3d/avatar-shootout/_sonata_cache` (mounted to /root/sonata).
- **Image `ultrashape`** (`$CP/shootout/Dockerfile.ultrashape`): UltraShape refine. py3.10/cu121 torch2.5.1 +
  flash_attn/cubvh/sageattention/diso/torch_cluster. Weights: `/mnt/hdd/3d/avatar-shootout/_weights/ultrashape/
  ultrashape_v1.pt` (7.37GB, all-in-one vae+dit+conditioner).
- Repos under `/mnt/hdd/3d/avatar-shootout/`: `Hunyuan3D-Part/` (P3-SAM+XPart), `UltraShape-1.0/`. Both have
  **patched files mounted at runtime** (see §4) — keep them.

---

## 3. THE OPEN PROBLEM: texture onto the final mesh (frame mismatch)
**Texture GENERATION works** (`miku_try_tex.glb`, native UV-atlas PBR, ~0.9% inpaint).
**Texture TRANSFER onto the 69k FAILED**: reprojecting pixal3d's PBR onto the UltraShape-derived 69k gave
**97.8% of texels un-snapped (inpainted = wrong)**. Root cause = **coordinate-FRAME mismatch**, NOT tuning:
- pixal3d's PBR dump (`dump_dense_*`) is in pixal3d's **internal** frame.
- pixal3d's `.glb` output applies a fixed **rotation** (the `rot` matrix near the end of `pixal3d.cpp` to_glb),
  so `miku_try.glb` (→ UltraShape input) is ROTATED vs the dump.
- UltraShape **re-normalizes** (scale ~0.99, centered) → yet another frame. (`infer_dit_refine.py --scale 0.99`.)
- Net: final 69k ≈ `(UltraShape-norm ∘ pixal3d-to_glb-rot)` of the dump frame. bbox sizes differ ~2.3–2.7×
  *per-axis* AND there's a rotation → a bbox-scale can't reconcile it → the bake's closest-point snap misses.

**`tex_reproject` (the C++ bake)** is the right tool: reads `dump_dense_*` (colour source) + `dump_pbr_*` +
`dump_mesh_*` (bake TARGET), colours the dense verts via grid_sample then bakes a UV PBR for the target by
snapping target texels onto the dense shell. **`CUMESH_TARGET=0` is REQUIRED** to make it bake onto the
provided `dump_mesh` (else its v6 default `CUMESH_TARGET=500000` REBUILDS the target from the dense, ignoring
your mesh). Build: `./build.sh tex_reproject cuda`. Run: `CUMESH_TARGET=0 TEX_FINAL_SIZE=2048 ./tex_reproject 2048 out.glb`.

### THE TASK — automate texturing via a SINGLE CANONICAL FRAME
Make the whole pipeline carry meshes in ONE documented frame (centered, unit-scaled, fixed up/forward axis),
so pixal3d's PBR and the final 69k coincide *by construction* and the bake needs **zero alignment**. Concretely,
pick & implement one (automated, committed — not hand-typed):
- **(A) Canonical-frame contract (preferred):** add an automated `normalize-to-canonical` at every stage
  boundary (a tiny C++/committed tool: center + unit-scale + fix axis). pixal3d emits canonical; UltraShape
  output → canonical; P3-SAM/decimate preserve it; the PBR dump is also expressed canonical (record pixal3d's
  to_glb rotation and apply its inverse to the dump, or dump in canonical frame directly). Then
  `tex_reproject CUMESH_TARGET=0` on the canonical 69k = correct (<~1% inpaint, like the native bake).
- **(B) Composed inverse transform:** since pixal3d's to_glb rotation is fixed/known and UltraShape's
  normalization is deterministic from the input bbox + `--scale`, compute the exact mesh→PBR-frame transform
  and apply it automatically before bake. (Riskier: must get the composition + UltraShape's exact norm right.)
- **(C) Auto registration stage:** ICP/principal-axis+centroid+scale align of the 69k to `dump_dense`
  (robust to any frame diff, fully automatic). Heaviest but most general.

Validate: bake → final textured 69k should have **<~5% inpaint** and colours that match `miku_try_tex.glb`
(teal hair, grey/black outfit) when zoomed on `compare.html`. Then **wire the whole thing into ONE reproducible
driver** (`$CP/shootout/run_pipeline.sh <image>` → textured lowpoly), each stage a real call, no manual steps.

---

## 4. GOTCHAS / PATCHES ALREADY MADE (keep these; they're load-bearing)
- **P3-SAM** (`Hunyuan3D-Part/P3-SAM/`): docker adds `omegaconf tqdm` (a late layer). **3060 OOM** at the
  `[point_num,K,518]` seg tensor → use `PROMPT_BS=2`. **Repo bug:** `main()` parses `--prompt_bs` but never
  passes it to `predict_aabb` → PATCHED `demo/auto_mask.py` + `demo/auto_mask_no_postprocess.py` (added
  `prompt_bs=args.prompt_bs,` to the predict_aabb call). `point_num` does NOT drive VRAM (prompt_bs does).
  Auto-merges to ~7–21 parts; `no_postprocess` also merges (~same). Coverage better on the clean refined mesh.
- **UltraShape** (`UltraShape-1.0/`): Dockerfile needs `ENV CPATH=/usr/local/cuda/include` (diso host C++
  compile can't find cuda_runtime.h otherwise); diso+sageattention installed `--no-build-isolation`. diso's
  `_C.so` has an undefined-symbol ABI bug **but it's unused** — the default `mc_algo='mc'` path uses cubvh, so
  ignore it. RUNTIME PATCHES to `scripts/infer_dit_refine.py`: (1) `load_models(..., low_vram)` keeps modules
  on CPU when low_vram — the stock code `.to(device)`'d the full ~3B DiT before `enable_model_cpu_offload()` →
  load-OOM on 12GB; (2) `generator = torch.Generator('cpu')` not cuda (offload device mismatch w/ randn_tensor);
  (3) run with `PYTHONPATH=/work/UltraShape-1.0`. Run via `--low_vram --num_latents 8192 --chunk_size 2048
  --octree_res 512`. Model load is slow (7.37GB off /mnt/hdd; GPU idle during load — normal).
- **tex_reproject**: `CUMESH_TARGET=0` (else it rebuilds the target, ignoring dump_mesh). dump faces = int64.
- The `.sframe` linker note on every C++ build is benign.

---

## 5. compare.html VIEWER (to show results; http.server on :8011 serving `$CP`)
- Server (already running; restart if needed): `cd $CP && python3 -m http.server 8011 --bind 0.0.0.0`.
  Open **http://10.0.0.208:8011/compare.html**.
- It's `<model-viewer>` (3 synced viewers a/b/c) with tabbed modes in a `SRC={}` map + `setMode()`. To add/edit
  a comparison: copy the GLB into `$CP/` (the served dir — gitignored, fine), add/update a `SRC.<mode>` entry
  (`a/b/c` = GLB filenames relative to `$CP`, with `?v=<cachebust>` + `capA/B/C` HTML), add a `<button>` in
  `.tabs`, and a `tX.className=...` line in `setMode`. **Hard-reload (Ctrl-Shift-R)** — model-viewer caches.
- The **"UltraShape clean ✨"** tab (`us` mode) currently shows: A=`us_tex.glb` (pixal3d textured, good),
  B=`us_lowpoly_tex.glb` (the FAILED 97.8%-inpaint transfer probe), C=`us_lowpoly.glb` (the 69k geometry).
  Replace B with the correctly-textured 69k once the canonical-frame bake works.

## 6. Current artifacts in `$OUT` (so you can resume without re-running)
`miku_try_matte.png` (matte) · `miku_try.glb` (pixal3d geom 138k) · `miku_try_tex.glb` (pixal3d TEXTURED) ·
`ultrashape_mikutry/miku_try_matte_refined.glb` (clean 1.04M) · `p3sam_mikutry_refined/auto_mask_mesh_final.glb`
+ `_face_ids.npy` (18-part seg) · `miku_lowpoly.glb` (FINAL 69k geometry, fingers intact) ·
`miku_lowpoly_tex.glb` (the failed transfer). The pixal3d `--tex` dump (`dump_*.bin`) is in `$CP/` — but it was
clobbered by the failed-probe's hand-aligned mesh; **re-run §1b to regenerate a clean dump.**

## 7. Definition of done
One command `run_pipeline.sh <image.png>` → a textured, ~tens-of-k-face, riggable GLB with fingers, produced
fully automatically (no manual steps), viewable on compare.html with correct colours. Bonus: source-image
finger-parting for cleaner gen; lighter budgets; then C++-port the decimate+bake stages (P3-SAM/UltraShape
get ggml ports as their turn comes — they're the same VecSet/sparse-voxel + transformer families already ported).
