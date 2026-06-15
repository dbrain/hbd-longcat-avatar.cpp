# HANDOFF-E — IM curvature-adaptive retopo (RENDER-blocked) + SkinTokens R1 VecSet encoder (validated)

Continues HANDOFF-D. Both tracks were pushed to their **CPU limit** and are now **GPU-blocked**. This
doc has: (§1) what each track does + how it was built/run, (§2) the exact GPU commands needed and why,
(§3) queued render/validation commands ready to paste. Everything here is committed on
`spike/sparse-conv-3d` (source only; IM src lives in a gitignored fresh clone, carried as a patch in
`build_instant_meshes.sh`).

Read first: `HANDOFF-D-IM-adaptive-and-skintokens.md`, `HANDOFF-C-retopo-quadriflow.md`,
`HANDOFF-RIGGING-skintokens.md`, `V6-DENSE-RECIPE.md`.

---

## TRACK 1 — IM curvature-adaptive retopo (DONE on CPU; needs the RENDER eyeball)

### What it is
wjakob/instant-meshes is **UNIFORM** — one global lattice spacing — so a fixed quad budget spreads
evenly and thin/high-curvature parts (fingers) web together before the flat torso ever needs that
density. HANDOFF-D's premise ("IM core is already curvature-adaptive, just plumb it") was **wrong**:
this IM has NO adaptive sizing field (verified — `hierarchy.h` carries a single `mScale`, no per-vertex
sizing array; no "adaptive"/"curvature" code anywhere). So this lap *implemented* one.

### The implementation (the key insight)
The lattice-math functors (`compat_position_*`, `position_round_*`) **already take `(scale, inv_scale)`
as parameters**, so a per-vertex sizing field can be injected at the CALL SITES without rewriting any
lattice math. Added behind one knob (`IM_ADAPTIVITY` env or `-A <float>`; **default 0 = the uniform
path, bit-for-bit — cannot regress**):

- `hierarchy.{h,cpp}` — per-level per-vertex sizing field `mS` (mirrors `mA`); `build()` downsamples it
  area-weighted alongside the geometry; `setS()` / `S(level)` / `hasSizingField()`.
- `batch.{h,cpp}` — `compute_sizing_field()`: 1-ring normal-variation curvature → smooth → robust
  area-weighted standardize → density multiplier (`d = 1 + adaptivity·GAMMA·cn`) → `s = base/√d` →
  **gradation limiting** (the load-bearing robustness step, GRAD=1.5) → **single global-scalar budget
  conservation** (same total quad count at a fixed `-f`). `batch_process()` gains a `float adaptivity` arg.
- `field.cpp` (`optimize_positions_impl`, both solver lambdas) — per-vertex `s_i` for rounding, per-edge
  mean `s_e` for the compat/translate terms.
- `extract.cpp` (`extract_graph`) — per-edge `s_e` at the **collapse decision** (this is what actually
  redistributes density) + a per-extracted-vertex sizing field `S_new` so the sliver-snap threshold
  (`0.3·scale`) becomes local (else the global threshold re-merges fingers in dense regions, undoing it).

**Two bugs found + fixed during the run** (both were the same root cause — incommensurate neighbouring
lattices when `s` jumps abruptly): (a) "Internal error in extraction" (a degenerate face survived
pure-quad subdivision), (b) vertex count *inflating* instead of conserving. **Gradation limiting** (bound
the field's local rate of change) fixed both: extraction succeeds and final face counts conserve.

### Build / run (CPU, on-server OK)
```bash
cd tools/m1_ref/cpp_port
./build_instant_meshes.sh                      # clones IM, applies the adaptive patch, builds
BIN=$(./build_instant_meshes.sh -p)
# A/B at a FIXED budget: adaptivity 0 (uniform) / 0.5 / 1.0, deterministic
"$BIN" -i /tmp/det_mani_d9.obj -o out_a0.obj   -f 60000 -D -A 0
"$BIN" -i /tmp/det_mani_d9.obj -o out_a0.5.obj -f 60000 -D -A 0.5
"$BIN" -i /tmp/det_mani_d9.obj -o out_a1.0.obj -f 60000 -D -A 1.0
```
Inputs are the ManifoldPlus-cleaned dense Miku: `/tmp/det_mani_d9.obj` (grid-512, 2.1M v) and
`/tmp/det_mani_d10.obj` (grid-1024, 5.2M v, crispest fingers). ~25s (d9) / ~65s (d10) per run.

### CPU results (6 variants written to `/tmp/im_adapt/`, all VALID, finite, ~budget-conserved)
| variant | final F | %verts in baseline's top-curvature decile |
|---|---|---|
| d9  a0 (uniform) | 215,597 | 10.00% (1.00× by def) |
| d9  a0.5 | 238,398 | 16.67% (**1.67×**) |
| d9  a1.0 | 235,332 | 21.41% (**2.14×**) |
| d10 a0 (uniform) | 212,223 | 10.00% |
| d10 a0.5 | 227,597 | 16.19% (**1.62×**) |
| d10 a1.0 | 211,850 | 20.87% (**2.09×**) |

So at a **fixed `-f`** (d10 a1.0 final F = 211,850 vs uniform 212,223 — essentially identical budget),
the adaptive field puts **~2× more vertices in high-curvature regions** (fingers/edges) and output mean
curvature rises (CPU check: `im_adapt_density_check.py`). **This is a quantitative INDICATOR, not proof
of finger separation** — that is a visual question for the render (HARD RULE: never claim a mesh "works"
from vertex counts).

### Renders DONE (2026-06-15 GPU run) — density redistribution CONFIRMED; finger verdict = owner's
```bash
cd tools/m1_ref/cpp_port && ./render_im_adapt.sh     # 6 variants: hand sweeps + body + detail crops
# -> /tmp/im_adapt/CMP_{hands,geo}_{d9,d10}.png, geo_*_detail.png, FULL_d10.png
```
**What the renders show:** at a FIXED `-f` (final face counts ~equal), the adaptive (a1.0) mesh has
**visibly denser, cleaner quad flow on the high-curvature regions** — the boots/heels and the skirt
frills are noticeably finer than uniform (see `geo_d10_a{0,1}_detail.png`), and the **full-figure
silhouette is preserved** (`FULL_d10.png`), confirming the budget-conserving shape-safety property.
So the lever provably moves density to curvature — exactly the mechanism. **The finger-specific verdict
is NOT closed:** this Miku's huge twintail hair dominates the side extremities, so the auto hand-framing
(`render_hands_auto.py`) kept hitting hair, not fingers. The owner — who knows the asset — should judge
finger separation directly on `compare.html` (add an "IM adaptive" tab; the OBJ variants are in
`/tmp/im_adapt/`). **Success target:** fingers separate at a0.5/a1.0 where uniform (a0) webs them.
**If it helps:** wire `IM_ADAPTIVITY` into the live pixal3d retopo driver + re-bake (normal map off the
V6 dense shell, HANDOFF-C R2). **If not:** the normal-mapped uniform path already ships acceptable
fingers — this is polish. Tuning knobs in `compute_sizing_field` (the patch heredoc in
`build_instant_meshes.sh`): `GAMMA` (aggressiveness), `GRAD` (gradation ratio; raise → more contrast but
risks the extraction error returning), `DMAX/DMIN`.

---

## TRACK 2 — SkinTokens R1 VecSet mesh-condition encoder (WRITTEN, COMPILES, CPU-VALIDATED)

### What it is
The R1 spike from `HANDOFF-RIGGING-skintokens.md`: port the VecSet mesh-condition encoder (the one
genuinely-new GPU primitive; also Hunyuan3D shape-VAE groundwork) and validate vs golden BEFORE
committing to the rest. **Confirmed exact architecture from the proven checkpoint** (CPU static
inspection of `SkinTokens/experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt`,
`model_config.mesh_encoder`, `__target__=michelangelo_encoder` ⇒ `ShapeAsLatentPerceiverEncoder`,
`no_query=True`):

> FourierEmbedder(F=8, include_pi=FALSE, include_input=TRUE) ⊕ normals(3) → input_proj(54→512) →
> ResidualCrossAttentionBlock(query=FPS-sampled, kv=full) → 8× ResidualAttentionBlock → ln_post →
> latents **[width=512, Q=512]**.  heads=8 (head_dim=64), **qkv_bias=FALSE**, mlp×4, GELU=erf,
> attn scale=1/√64, fp32 softmax, ln eps=1e-5. (`num_latents=256` in cfg is unused for the query —
> `no_query=True`; the query is the FPS-sampled point embeddings, `token_num=512`.)

### What was built (all reusable; `tools/m1_ref/cpp_port/`)
- **`vecset_encoder.hpp`** — the ggml graph (`build_vecset_encoder`), config-struct driven, reusing
  `m1_ggml.hpp` (`lin`/`layernorm`/`gelu_erf_`/`attention`). The error-prone part (packed-QKV per-head
  slicing) is isolated in `vs_take_head_slice`. Weight keys = PyTorch state_dict suffixes under
  `mesh_encoder.encoder.` so the GGUF packer writes them verbatim.
- **`vecset_encode.cpp`** — R1 harness: host Fourier embed (exact, implemented) → graph → compare to
  golden latents. Build: `./build.sh vecset_encode [cuda]` (default ggml branch; no script edit).
- **`vecset_synth_gen.py`** — CPU-only synthetic parity oracle: instantiates the EXACT
  `CrossAttentionEncoder` submodules on CPU/fp32 with random weights + random points and dumps
  weights+inputs+latents as npy. **NOT** the R0 golden (no GPU, no FPS/rng/bf16) — a unit test of the
  ggml graph's numerics.

### ✅ VALIDATED — ggml graph matches the REAL model on both backends (2026-06-15, GPU run)
Two independent checks, both PASS the 2e-3 fp32-oracle tol with large margin:

1. **CPU synthetic oracle** (random weights, fp32, no GPU): ggml == PyTorch to **maxabs 2.1e-6** —
   fp32-exact. Proves the graph math (QKV slicing, both attentions, 8-layer stack, GELU, scale,
   ln_post, host Fourier embed).
   ```bash
   cd /mnt/hdd/3d/avatar-shootout/SkinTokens
   CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD .venv/bin/python \
     ~/dev/longcat-sparse-spike/tools/m1_ref/cpp_port/vecset_synth_gen.py /tmp/vecset_synth
   cd ~/dev/longcat-sparse-spike/tools/m1_ref/cpp_port
   ./build.sh vecset_encode && ./vecset_encode /tmp/vecset_synth        # PASS maxabs 2.1e-6
   ```
2. **R0 real-model golden** (REAL trained weights from the proven ckpt + REAL rng.choice(0)+FPS
   sampling + GPU fp32 oracle, NVIDIA_TF32_OVERRIDE=0): ggml **CPU maxabs 1.2e-4 / CUDA maxabs 5.6e-5**
   (meanabs ~2-6e-6) vs the real encoder output. So the graph is correct with the actual weight
   distributions and the real FPS-sampled-query path, on both backends. **R1 (the de-risk spike) is
   DONE — the VecSet encoder is portable.**
   ```bash
   cd /mnt/hdd/3d/avatar-shootout/SkinTokens
   NVIDIA_TF32_OVERRIDE=0 PYTHONPATH=$PWD .venv/bin/python \
     ~/dev/longcat-sparse-spike/tools/m1_ref/cpp_port/capture_vecset_r0.py /tmp/vecset_r0   # giraffe.glb
   cd ~/dev/longcat-sparse-spike/tools/m1_ref/cpp_port
   ./vecset_encode /tmp/vecset_r0                 # CPU  PASS maxabs 1.2e-4
   ./build.sh vecset_encode cuda
   NVIDIA_TF32_OVERRIDE=0 ./vecset_encode /tmp/vecset_r0 cuda   # CUDA PASS maxabs 5.6e-5
   ```
   (Golden npys live in `/tmp/vecset_r0`, regenerable via `capture_vecset_r0.py`; not committed —
   cpp_port keeps only source. The encoder uses the custom pure-torch `fps` in `src/model/utils.py`,
   NOT torch_cluster.)

### Remaining Track-2 follow-ons (R1 closed; these are R1-polish → R2…R7)
- **GGUF pack (optional, for prod/perf):** write `mesh_encoder.encoder.*` (106 tensors) from the .ckpt
  as `<dir>/skintokens.gguf` (keys already match), then `PIXAL3D_GGUF_DIR=<dir> ./vecset_encode …`.
  Not needed for correctness — npy mode already validates.
- **Host sampling port (CPU):** numpy PCG64 `rng.choice(N, 2048, seed=0)` + the pure-torch
  `fps(ratio=1/4)` to derive `sampled_pc` from `pc` in C++ (currently injected from golden). Validate
  sampled indices vs golden, then drop the `sampled_*` inputs.
- `output_proj` (Linear 512→896 + LayerNorm; `output_proj.{0,1}` in the ckpt) is the rig's, not the
  encoder's — R2+.
- Then R2 Qwen3-0.6B core, R3 beam decode, R4 FSQ skin decoder, R5 tokenizer, R6 rigged-GLB, R7 E2E.

### Then (R1 follow-ons → R2…R7, see the rigging handoff)
- **Host sampling port (CPU):** numpy PCG64 `rng.choice(N, 2048, seed=0)` + `fps(ratio=1/4)` to derive
  `sampled_pc` from `pc` (currently injected from golden). Validate sampled indices vs golden, then drop
  the `sampled_*` inputs. Standing gotcha: "point sampling must match Python exactly or condition drifts."
- `output_proj` (Linear 512→896 + LayerNorm; `output_proj.{0,1}` in the ckpt) is the rig's, not the
  encoder's — R2+. R1's validation target is the encoder `latents`.
- Then R2 Qwen3-0.6B core, R3 beam decode, R4 FSQ skin decoder, R5 tokenizer, R6 rigged-GLB, R7 E2E.

---

## Files added/changed this lap (committed on `spike/sparse-conv-3d`)
- `tools/m1_ref/cpp_port/`:
  - `build_instant_meshes.sh` (now applies the adaptive **patch heredoc** after clone, idempotent)
  - `im_batch_main.cpp` (`-A`/`IM_ADAPTIVITY` knob)
  - `render_im_adapt.sh` (GPU renders), `render_hands_auto.py` (hand-locator render), `im_adapt_density_check.py` (CPU density check)
  - `vecset_encoder.hpp`, `vecset_encode.cpp`, `vecset_synth_gen.py` (CPU oracle), `capture_vecset_r0.py` (GPU real-model golden) — R1, VALIDATED
- IM src changes (`batch.{h,cpp}`, `hierarchy.{h,cpp}`, `field.cpp`, `extract.cpp`) live in the
  gitignored fresh clone `thirdparty/instant-meshes/` and are carried as the patch in
  `build_instant_meshes.sh` (verified: applies cleanly to a fresh upstream clone, builds, idempotent).
- This doc (repo root).

## Standing gotchas (inherited): validate vs TRUE-fp32 oracle (not bf16); fp32 tanh/erf-GELU;
`NVIDIA_TF32_OVERRIDE=0`; long GPU runs `run_in_background:true` from the MAIN loop (sub-agents deadlock
on background completion); no multi-agent Workflow for research; no `pkill -f`; no rm-globs. C++/CUDA
build fine on-server; **NO Rust builds**.
