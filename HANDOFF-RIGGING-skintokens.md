# HANDOFF — RIGGING (next major phase after pixal3d feature-complete + perf)

**This is a head-start spec, NOT active work.** pixal3d/TRELLIS.2 is on its final laps
(UV-atlas PBR textures + parked perf — see `HANDOFF-NEXT-feature-complete.md` /
`KICKOFF-next-agent.md`). Rigging is the **next subsystem** and is a clean BOLT-ON: it
consumes the mesh pixal3d already produces and adds a skeleton + skinning weights. It does
**not** touch the geometry/texture/perf code in flight — finish those first, then start here.

Goal: image → **rigged, animation-ready** GLB on the RTX 3060, pure C++/ggml, lighter+faster
than the Python. The north-star the whole 3D effort was pivoted toward (see memory
`project_avatar_rig_path`).

---

## 0. The architecture decision (read first — it's the load-bearing idea)

Treat the pipeline as **two swappable slots joined by a plain mesh (GLB / verts+faces):**

```
[ GEOMETRY slot ]            mesh           [ RIGGING slot ]
image → static mesh    ───────────────→   mesh → skeleton + skin
  • pixal3d / TRELLIS.2  (sparse voxel, DONE)   • SkinTokens / TokenRig  (AR transformer) ← PORT THIS
  • Hunyuan3D-2.1        (vecset DiT, future)    • UniRig (older, weaker — skip)
```

**The rigger to port is SkinTokens (a.k.a. TokenRig), VAST-AI, MIT license** (arXiv
2602.04805; repo `VAST-AI-Research/SkinTokens`). Why it's the right pick:

- **Geometry-agnostic.** It conditions on whatever mesh you hand it → port it ONCE and it rigs
  pixal3d's output, a future Hunyuan3D port, anything. This is the leverage that future-proofs
  "best geometry model varies per asset."
- **MIT-licensed** → clean for the commercial game-asset prod pivot (pixal3d/TRELLIS license is
  un-vetted; flag that separately). 
- **Already proven on the 3060 in Python:** peak **3.2 GB / 95 s**, **52 bones + full per-finger
  articulation**, clean skin — end-to-end on a real pixal3d mesh (memory `project_skintokens_3060`).
  That run is your golden reference. Working dir: `/mnt/hdd/3d/avatar-shootout/SkinTokens/`.
- **Lighter port than pixal3d was:** NO sparse conv (the existential risk that dominated pixal3d),
  **no custom CUDA ops at inference**, and the heavy compute is a Qwen3-0.6B transformer — your
  single strongest porting area (qwen3-tts.cpp, llama-turboquant). 

**Do NOT architect around AniGen** (arXiv 2604.08746, VAST, SIGGRAPH 2026). It fuses geometry+rig
into one monolith (S³ Fields) → it does **not** compose, and its rig is locked to its own
(older, TRELLIS.1-era) geometry — you can't point it at pixal3d's better TRELLIS.2 mesh. Keep it
as a standalone eval/eyeball only.

**Hunyuan3D-2.1 (future alt geometry, arXiv 2506.15442):** 3.3B flow-matching DiT over a **VecSet
latent → SDF → marching cubes**, + 2B Hunyuan3D-Paint (PBR, CUDA raster). It is **vecset, not
sparse-voxel** — a different island from pixal3d. The bonus: **SkinTokens' mesh encoder is the
same VecSet family**, so the VecSet primitive you build for SkinTokens (R1 below) is a
down-payment on a future Hunyuan3D shape port. License `tencent-hunyuan-community` (commercial
with restrictions). Not in scope here; noted so R1 is built reusably.

---

## 1. SkinTokens — confirmed inference architecture

Source: arXiv 2602.04805 (validate every number against the repo + your golden capture — paper
summaries drift). Input: mesh normalized to `[-1,1]³`. Output: skeleton (joint positions + bone
hierarchy/tree) + skinning-weight matrix `N×J`.

1. **Mesh front-end (host):** normalize to `[-1,1]³`; sample a surface point set (+normals). Their
   working command rigs a **decimated** proxy then transfers weights to the full mesh — note the
   `--use_transfer` flag in `run_skintokens.sh`. pixal3d emits ~1.5M verts; the rigger runs on a
   decimated mesh and the skin is transferred back. So the front-end = **decimate + sample points**,
   and the back-end = **weight transfer to full-res** (nearest / barycentric). Capture both ends'
   exact behavior from Python.
2. **VecSet mesh encoder (110M)** — `3DShape2Vecset` design (Zhang 2023): 2 encoder + 10 decoder
   layers, PMPE positional encoding, processes geometry as an unordered point set → shape condition
   features for the transformer. **This is the one genuinely-new GPU primitive (the R1 spike).**
3. **FSQ-CVAE skin codec** — Finite Scalar Quantization, levels **`[8,8,8,5,5,5]` = 64,000 entries,
   NO learnable codebook** (round-to-grid; dequant is pure scalar arithmetic). VecSet encoders;
   decoder → per-vertex weights with sigmoid `[0,1]`. Up to **`TD=32` skin tokens per bone**.
4. **Qwen3-0.6B autoregressive transformer** — GQA + RoPE (standard decoder-only). Vocabulary =
   quantized joint-coordinate integer tokens + type tokens + the ~64K FSQ skin tokens + specials
   `<bos> <eos> <type>`. Generates the **whole rig as one interleaved sequence**:
   `<bos> <type₁> dx₁ dy₁ dz₁ … <typeₖ> dxₖ dyₖ dzₖ … 𝒟₁,₀…𝒟₁,TD … 𝒟ₖ,₀…𝒟ₖ,TD <eos>`
   — i.e. **skeleton tokens first, then per-bone skin tokens in canonical order.** Decoding uses
   **beam search (beams=10)** in the proven run.
5. **Skeleton tok/detok (host):** joint coords uniformly quantized → integer tokens; bone order from
   templates (bipeds) or chain-partitioning. Detok = dequant coords + reconstruct parent-child edges
   from the type/sequence structure.
6. **No custom CUDA ops at inference.** (LBS appears only in training reward.)

---

## 2. Reuse inventory — what's already in `tools/m1_ref/cpp_port/` you lean on

- **`m1_ggml.hpp`** — `attention()`, `lin()`, RoPE apply, RMSNorm, GGUF-backed weight load, the
  persistent-weights-buffer pattern. Covers ~all of the Qwen3 core AND the VecSet attention blocks.
- **`gguf_reader.hpp` + `pack_gguf.cpp`** — pack SkinTokens' 3 weight sets (VecSet encoder, FSQ-CVAE,
  Qwen3-0.6B) → GGUF; `--type f16/q8_0/q4_k` already there for later quant.
- **`glb_writer.hpp` / `glb_textured.hpp`** — extend for the glTF **skin** (joint node tree,
  `inverseBindMatrices`, `JOINTS_0` + `WEIGHTS_0` vertex attrs). The static/textured GLB path is done.
- **`build.sh`** — add `rig_*` / `vecset_*` test branches (CPU `build-cpu`, CUDA `build-cuda`; same
  toolchain).
- **The whole methodology**: golden_stage_hook-style monkeypatch dumps, per-op validation vs a
  **TRUE-fp32 oracle** (NOT the bf16/tf32 golden), set-equal / cosine / maxabs tolerances.

**Net-new GPU work** = (a) the **VecSet cross-attention encoder** [main piece, shared with future
Hunyuan3D], (b) **beam-search KV management** (your TTS loops are greedy/sampling — beam is a new
wrinkle), (c) FSQ dequant (trivial). Everything else is reuse or host-side.

---

## 3. Milestone ladder (mirror pixal3d: de-risk the new primitive FIRST)

Exactly as sparse-conv was spiked before the E2E port, **spike the VecSet encoder before
committing to the rest** — it's the only existential-ish unknown.

- **R0 — Golden capture.** From `SkinTokens/.venv`, run the proven command on a validated pixal3d
  mesh; monkeypatch-dump every stage boundary: sampled points (+normals), VecSet condition features,
  the full token sequence, FSQ skin-token ids, decoded per-vertex weights, final joints + bone tree,
  and the rigged GLB. Also dump the decimation + weight-transfer in/out. This is the foundation for
  every R-stage's validation. (Mirror `tools/sparse_spike/golden_stage_hook.py`.)
- **R1 — VecSet mesh encoder (THE SPIKE).** Port `3DShape2Vecset` (2 enc / 10 dec, PMPE) in ggml;
  validate the condition features vs R0 golden to fp32-oracle tol. **If this lands, the rest is
  known-portable.** Build it cleanly reusable (it doubles as Hunyuan3D shape-VAE groundwork).
- **R2 — Qwen3-0.6B core + custom vocab.** Transformer via `m1_ggml.hpp` (GQA + RoPE + RMSNorm +
  SwiGLU); custom embedding table + output head over the joint/type/skin vocab. Validate **teacher-
  forced**: feed golden tokens, check next-token logits match golden — isolates the transformer from
  the search.
- **R3 — Beam-search decode + KV cache.** beams=10 AR loop. Beam search is nondeterministic across
  impls → **validate the RIG (joints/weights/bend), not bit-exact tokens.**
- **R4 — FSQ-CVAE skin decoder.** Scalar dequant (levels `[8,8,8,5,5,5]`) + decoder net → per-vertex
  weights+sigmoid; validate vs R0 weights.
- **R5 — Tokenizer/detokenizer (host).** Skeleton coord quant/dequant grid (must match Python
  **exactly** — off-by-one ⇒ wrong joints), bone-order template, parent-child tree reconstruction;
  validate joints + hierarchy vs golden.
- **R6 — Rigged-GLB writer.** Extend `glb_writer.hpp` with glTF skin; validate the GLB loads and
  bends — mirror their `rig_probe.py` / `rig_wave.py` per-region bend test (probe bone-local axes;
  glTF bone axes are arbitrary).
- **R7 — E2E.** Standalone `pixalrig <in.glb> <out.glb>` (geometry-agnostic — the universal-rigger
  primitive) + a convenience `pixal3d --rig` that chains geometry→rig. Compare to the Python rigged
  GLB: joint positions, weight maps, visual bend test.
- **Then PERF** (quantize Qwen3 + VecSet via the existing GGUF Q-types; judge by rig agreement) and
  **PROD** (koblem heavy-GPU engine slot like acestep/flux2: worker-isolation, idle-unload true-0,
  REST + panel).

---

## 4. KEY FACTS / paths

- **Python reference (golden source):** `/mnt/hdd/3d/avatar-shootout/SkinTokens/` — `.venv` (py3.11,
  torch2.7 cu128, **flash-attn 2.8.3** prebuilt wheel is load-bearing for THEIR run, bpy 5.0.1 via
  out-of-process `bpy_server`). Scripts: `setup_skintokens.sh`, `run_skintokens.sh`, `inspect_rig.py`,
  `rig_probe.py`, `rig_wave.py`. Proven output: `results/cutout-1_1024_rigged.glb` (52 bones, beams=10,
  `--use_transfer`). Full detail: `~/dev/avatar-shootout/HANDOFF.md` §PHASE 3.
- **Port lives here:** `tools/m1_ref/cpp_port/`; build `./build.sh <test> [cuda]`; CUDA runs need
  `NVIDIA_TF32_OVERRIDE=0` + the toolchain `LD_LIBRARY_PATH`. GGUF weights dir via `PIXAL3D_GGUF_DIR`.
- **Memory to read:** `project_skintokens_3060`, `project_avatar_rig_path`, `project_3dgen_cpp_port`,
  `feedback_no_build_on_server` (C++/CUDA build fine here; **NO Rust builds**),
  `reference_subagent_background_stall`.

## 5. GOTCHAS

**Rigging-specific:**
- **R1 (VecSet) is the spike — prove it before anything else.** It's the only real GPU unknown.
- **Point sampling + decimation + weight-transfer must match Python exactly** (sample scheme, count,
  normals, `[-1,1]³` normalize, decimation target, transfer rule) or the condition/skin drifts.
- **Joint-coord quantization grid must match exactly** — off-by-one ⇒ displaced joints.
- **Beam search is nondeterministic across impls** → validate the rig output (joints/weights/bend
  render), NOT bit-exact token equality.
- **flash-attn "load-bearing" in their Python is just their attn impl** — use your own ggml attention
  and validate numerically; it is not a porting constraint.
- glTF bone-local axes are arbitrary — **PROBE before choreographing** any bend test (their note:
  +Z raised char-right arm for that rig).

**Standing project gotchas (inherited):** persistent-weights buffer or gallocr recompute → NaN;
fp32 tanh-GELU (not the F16 LUT); validate vs the **TRUE-fp32 oracle**, not the bf16/tf32 golden;
`NVIDIA_TF32_OVERRIDE=0`. Long runs `run_in_background:true` (NOT detached `&`); **sub-agents
DEADLOCK on background-job completion** → drive heavy runs from the main loop; **no multi-agent
Workflow for research**; no `pkill -f`; no rm-globs (delete named files only).

## 6. DEFINITION OF DONE

`pixalrig in.glb → rigged out.glb` (and `pixal3d --rig` chaining geometry→rig): a glTF skin with
skeleton + per-vertex `JOINTS_0`/`WEIGHTS_0` that **matches the Python SkinTokens output** (joint
positions, weight maps, and a passing per-region bend test), pure C++/ggml from GGUF, on the 3060
at ≤ the Python's 3.2 GB / 95 s — then quantize for perf and wrap as a koblem engine.
