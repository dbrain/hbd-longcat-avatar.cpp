# HANDOFF-F — SkinTokens rigging port: R1+R2+R5 ported & validated; R3/R4/R6/R7 remain

Continues HANDOFF-E. The SkinTokens (TokenRig) C++/ggml port is underway. **The full Python pipeline
runs e2e headless here** and the two hardest GPU nets + the skeleton output are ported and validated
against the real model. This doc: (§1) what's done, (§2) the remaining ladder with exact Python refs +
validation plans, (§3) all capture/build/run lines + golden locations.

Read first: `HANDOFF-RIGGING-skintokens.md` (R0–R7 spec), `HANDOFF-E-...` (R1 detail), memories
`project_avatar_rig_path`, `project_3dgen_cpp_port`.

Golden source: `/mnt/hdd/3d/avatar-shootout/SkinTokens/` (`.venv`, ckpt
`experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt`). All ports live in
`tools/m1_ref/cpp_port/` and reuse `m1_ggml.hpp` (M1Harness, lin/layernorm/attention/gelu) + npy goldens.

---

## 1. DONE + VALIDATED this lap (commits 724a990, bb3a487, 0dc5d99)

| rung | what | validation (vs Python) | files |
|---|---|---|---|
| **R0-full** | run the real pipeline headless (no bpy; `generate(vertices,normals)`) → 567 tokens, **60 joints**, skin [8192,60]; dump all stage goldens | pipeline runs; oracle captured | `capture_skintokens_e2e.py` → `/tmp/skintokens_e2e/` |
| **R1** | VecSet mesh-condition encoder (Michelangelo perceiver: Fourier+normals→input_proj→cross-attn(FPS query)→8× self-attn→ln_post; w512 h8 hd64) | **CPU 1.2e-4 / CUDA 5.6e-5 maxabs** vs real-weight fp32 golden; +2.1e-6 synthetic | `vecset_encoder.hpp`, `vecset_encode.cpp`, `capture_vecset_r0.py`, `vecset_synth_gen.py` |
| **R2** | Qwen3-0.6B AR core (hidden 896, 28 layers, 16q/8kv GQA, hd128, qk-norm, RoPE θ1e6, SwiGLU, causal, tied embeds) | **meanabs 1.6e-3, 99.07% argmax**, all 10 mismatches near-ties (= fp32 accumulation, not a bug) | `qwen3_forward.hpp`, `qwen3_r2_test.cpp`, `capture_qwen3_r2.py` |
| **R5** | host skeleton de-tokenizer (parse stream → undiscretized joints + parent tree via make_skeleton) | **bit-exact**: 60 joints maxabs 0, parents 0/60 mismatch | `detok_r5.cpp`, `capture_tokenizer_r5.py` |

The genuinely-novel/risky GPU primitive (R1 VecSet, also Hunyuan3D groundwork) and the heavy compute
(R2 Qwen3) are proven portable on CPU+CUDA. R5 gives the skeleton (joints+tree). What remains is
"known-portable" engineering: the generate loop that ties R1+R2 together, the skin decoder, and the
GLB writer.

### Build/run the validated rungs
```bash
cd tools/m1_ref/cpp_port
./build.sh vecset_encode cuda  && NVIDIA_TF32_OVERRIDE=0 ./vecset_encode /tmp/vecset_r0 cuda   # R1
./build.sh qwen3_r2_test cuda  && NVIDIA_TF32_OVERRIDE=0 ./qwen3_r2_test /tmp/qwen3_r2 /tmp/qwen3_w cuda  # R2
./build.sh detok_r5            && ./detok_r5 /tmp/skintokens_e2e                                # R5
```
(Re-capture goldens with the `capture_*.py` from the SkinTokens repo: `NVIDIA_TF32_OVERRIDE=0
PYTHONPATH=$PWD .venv/bin/python <script> ...`. The pure-torch `fps` is in `src/model/utils.py`;
no torch_cluster. Goldens live in `/tmp/{skintokens_e2e,vecset_r0,qwen3_r2,qwen3_w}` — regenerable,
not committed.)

---

## 2. REMAINING LADDER (R3, R4, R6, R7)

### R4 — FSQ-CVAE skin decoder (next; clean numerical port, reuses R1 VecSet machinery)
Produces the per-vertex skin weights `skin_pred [N, J]`. Python: `tokenrig.py:decode()` (L536) +
`SkinFSQCVAEModel` (`src/model/skin_vae/autoencoders/skin_fsq_cvae_model.py`). Three parts:
1. **FSQ.indices_to_codes** (`.../FSQ.py`) — levels **`[8,8,8,5,5,5]`** (=15000? no: 8·8·8·5·5·5=64000;
   vae.vocab=32768 ⇒ confirm actual levels from the ckpt buffer `_levels`). Pure scalar: code = per-digit
   dequant of the mixed-radix index, then a half-level shift/scale (`half_l`, `offset`, `_scale`). Trivial.
2. **cond encoder** (`vae.model._encode`, Tripo2Encoder) — produces `cond_latents [tokens_skin_cond=384,
   dim]` from `cond=[vertices,normals]`. **Same VecSet family as R1** — reuse `vecset_encoder.hpp` patterns
   (different dims: width_encoder=512; check `is_learned_queries`, FrequencyPositionalEmbedding vs the R1
   FourierEmbedder). Run once per mesh.
3. **decoder** (`vae.decode` → Tripo2Decoder, width_decoder=1024) — per joint: `z = FSQ codes [b,4,dim]`,
   cross-attends to `cond` + `cond_latents` → `logits [b, N, 1]` = that joint's skin column. Another
   VecSet-family cross-attn decoder.
- **Golden:** extend `capture_skintokens_e2e.py` to also dump `cond_latents` (have it), the per-joint
  `skin_tokens` (FSQ indices) and `skin_pred` (have it). Validate C++ skin_pred vs golden to fp32 tol.
- Weights: `vae.*` (252 tensors) — dump npy like R2 (`capture_qwen3_r2.py` pattern) or pack GGUF.

### R3 — AR generate loop + constrained decoding (ties R1+R2 together → token sequence)
Python: `self.transformer.generate(inputs_embeds=cat[mesh_cond, start_embed], num_beams=10, ...)` +
**`VocabSwitchingLogitsProcessor`** (`tokenrig.py:33`) + `get_logits_processor` (L525). The processor
constrains tokens by a grammar: before the switch token (=eos of skeleton) only
`tokenizer.next_posible_token(...)` are allowed (skeleton grammar); after it, only skin tokens
(`>= switch_token_id`); after exactly `J*tokens_per_skin` skin tokens, force eos.
- Port as a C++ AR loop over the validated R2 transformer **with a KV cache** (incremental decode — new
  vs R2's single teacher-forced forward; the new wrinkle). Beam search (beams=10) is the proven config
  but **nondeterministic across impls** → a greedy/sampling loop is an acceptable first e2e.
  **HANDOFF-RIGGING rule: validate the RIG (joints/weights/bend), NOT bit-exact tokens.**
- The grammar mask needs the tokenizer `next_posible_token` (`tokenizer_part.py:119`) + `bones_in_sequence`
  ported to host C++ (small, mirrors the detok state machine in `detok_r5.cpp`).
- inputs_embeds prefix = `cat[learned_mesh_cond (R1→output_proj), embed(start_tokens)]`. output_proj =
  `output_proj.0` (Linear 512→896) + `output_proj.1` (RMSNorm 896) — 3 tensors, trivial; append to R1.

### R6 — rigged-GLB writer (host)
Extend `glb_writer.hpp`/`glb_textured.hpp` with a glTF **skin**: joint node tree (from R5 joints+parents),
`inverseBindMatrices`, per-vertex `JOINTS_0` + `WEIGHTS_0` (top-4 weights per vertex from R4 skin_pred).
Python ref: `Asset` + `BpyParser.export_asset` (`_debug_export`, L280) and `run_rig` transfer/export
(`demo.py:241`). bpy is only used for I/O — the glTF skin can be written directly (trimesh/numpy or C++).

### R7 — E2E `pixalrig in.glb → rigged out.glb`
Chain: mesh → sample pts+normals → R1 encoder + R4 cond encoder → R3 generate → R5 detok (skeleton) +
R4 FSQ decode (skin) → R6 GLB. Plus `pixal3d --rig`. Validate vs the Python rigged GLB: joint positions,
weight maps, and a per-region bend test (their `rig_probe.py`/`rig_wave.py`; glTF bone axes are arbitrary
— PROBE before choreographing). Then PERF (quantize Qwen3+VecSet) + PROD (koblem heavy-GPU engine).

---

## 3. Gotchas / notes
- Headless core needs **no bpy** — `model.generate(vertices, normals, cls="", **gk)` (cls="" ⇒ free
  skeleton; cls names: rignet/vroid/articulation). bpy is only mesh load + export.
- Validate vs the **TRUE-fp32 oracle** (`.float()` + eager attn, `NVIDIA_TF32_OVERRIDE=0`), not bf16/flash.
- AR-LM correctness criterion = meanabs tiny + every argmax mismatch a near-tie (R2 pattern), NOT
  bit-exact logits — non-associative fp accumulation over 28 layers drifts ~0.25 on near-tied tokens.
- Qwen3 GQA repeat_kv interleave: insert n_rep as the INNER dim, repeat, reshape (see
  `qwen3_forward.hpp:qw_repeat_kv`). Causal mask built host-side (shared `attention()` is bidirectional).
- ckpt module prefixes + counts: `vae` 252, `mesh_encoder` 106, `transformer` 311, `output_proj` 3.
- Standing: C++/CUDA build fine on-server; NO Rust builds; drive long GPU runs from the MAIN loop
  (sub-agents deadlock on bg completion); no `pkill -f` / rm-globs.

## Files this lap (committed on `spike/sparse-conv-3d`)
`tools/m1_ref/cpp_port/`: `vecset_encoder.hpp`, `vecset_encode.cpp`, `capture_vecset_r0.py`,
`vecset_synth_gen.py` (R1); `qwen3_forward.hpp`, `qwen3_r2_test.cpp`, `capture_qwen3_r2.py`,
`capture_skintokens_e2e.py` (R2 + e2e); `detok_r5.cpp`, `capture_tokenizer_r5.py` (R5). Docs at repo
root (HANDOFF-E, this HANDOFF-F).
