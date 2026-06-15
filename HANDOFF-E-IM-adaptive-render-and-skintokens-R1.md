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

### ⛔ GPU STOP POINT (Track 1): the render-and-look (owner's job)
Renders are GPU (pyrender/EGL). **Queued, ready to paste** when the GPU is free:
```bash
cd tools/m1_ref/cpp_port
./render_im_adapt.sh        # renders all 6 variants' hand close-ups + body geometry, builds montages
# -> /tmp/im_adapt/CMP_hands_d9.png, CMP_hands_d10.png  (a0|a0.5|a1.0 side by side)
#    /tmp/im_adapt/CMP_geo_d9.png,   CMP_geo_d10.png    (uniform vs adaptive body — torso should coarsen)
```
**Success target (owner verdict):** fingers SEPARATE at adaptivity 0.5/1.0 where the uniform (0.0) path
webs them, with torso density visibly dropping to pay for it — same budget, better spent. Then optionally
add an "IM adaptive" tab to `compare.html`. If the lever helps, wire `IM_ADAPTIVITY` into the live
pixal3d retopo driver and re-bake (normal map off the V6 dense shell, per HANDOFF-C R2).
**If it disappoints:** the normal-mapped uniform path already ships acceptable fingers — this is polish.
Tuning knobs (all in `compute_sizing_field`, `batch.cpp` / the patch heredoc): `GAMMA` (aggressiveness),
`GRAD` (gradation ratio; raise → more contrast but risk the extraction error returns), `DMAX/DMIN`.

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

### ✅ CPU VALIDATION (no GPU): ggml graph == PyTorch oracle to **maxabs 2.1e-6** (fp32-exact)
```bash
cd /mnt/hdd/3d/avatar-shootout/SkinTokens
CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD .venv/bin/python \
  ~/dev/longcat-sparse-spike/tools/m1_ref/cpp_port/vecset_synth_gen.py /tmp/vecset_synth
cd ~/dev/longcat-sparse-spike/tools/m1_ref/cpp_port
./build.sh vecset_encode && ./vecset_encode /tmp/vecset_synth      # -> PASS, maxabs 2.146e-06
```
This proves the encoder GRAPH (QKV slicing, both attentions, the 8-layer stack, GELU, scale, ln_post,
the host Fourier embed) is correct, entirely on CPU. The remaining R1 unknowns are the *real-data*
plumbing (below), not the math.

### ⛔ GPU/torch STOP POINTS (Track 2)
1. **R0 golden capture (GPU).** Run the proven SkinTokens command on a validated pixal3d mesh and
   monkeypatch-dump (per `HANDOFF-RIGGING-skintokens.md` R0): the input points `pc[N,3]` + normals,
   the FPS-sampled `sampled_pc[512,3]` + `sampled_feats` (post `np.random.default_rng(0).choice` +
   `fps(ratio=1/4)`, eval mode), and the encoder output `latents[512,512]`. Save as
   `pc.npy/feats.npy/sampled_pc.npy/sampled_feats.npy/latents.npy` in a golden dir. *Why GPU:* the
   proven run is bf16+flash-attn on the GPU; this is the real condition-feature target.
2. **GGUF pack (CPU torch load, but pairs with R0).** Extend `pack_gguf.cpp`/a script to write
   `mesh_encoder.encoder.*` (106 tensors) from the .ckpt as `<dir>/skintokens.gguf`. Keys already match.
3. **Then validate end-to-end:**
   ```bash
   PIXAL3D_GGUF_DIR=<gguf-dir> ./vecset_encode <golden-dir> [cuda]   # expect maxabs < 2e-3 vs golden
   ```
   Validate vs the **TRUE-fp32 oracle**, not the bf16 golden (re-dump the reference in fp32 if needed).

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
  - `render_im_adapt.sh` (queued GPU renders), `im_adapt_density_check.py` (CPU density check)
  - `vecset_encoder.hpp`, `vecset_encode.cpp`, `vecset_synth_gen.py` (R1)
- IM src changes (`batch.{h,cpp}`, `hierarchy.{h,cpp}`, `field.cpp`, `extract.cpp`) live in the
  gitignored fresh clone `thirdparty/instant-meshes/` and are carried as the patch in
  `build_instant_meshes.sh` (verified: applies cleanly to a fresh upstream clone, builds, idempotent).
- This doc (repo root).

## Standing gotchas (inherited): validate vs TRUE-fp32 oracle (not bf16); fp32 tanh/erf-GELU;
`NVIDIA_TF32_OVERRIDE=0`; long GPU runs `run_in_background:true` from the MAIN loop (sub-agents deadlock
on background completion); no multi-agent Workflow for research; no `pkill -f`; no rm-globs. C++/CUDA
build fine on-server; **NO Rust builds**.
