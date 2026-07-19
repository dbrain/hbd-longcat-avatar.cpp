# Upstream merge working notes — 2026-07-19

Worktree: `../longcat-avatar-mergewt`, branch `upstream-merge-2026-07-19`
Strategy: **MERGE** (owner confirmed). ggml first, sd.cpp second.

## Stage 1 — ggml merge ✅ DONE + BUILDS (committed 531a2c72, branch upstream-merge-2026-07-19)
All 13 conflicts resolved. Builds clean two-arch (sm86;120) + GGML_SAGEATTENTION3=ON, 0 errors.
fattn-sa3.cu compiles. GPU determinism/smoke tests PENDING (need GPU slot).

## Stage 1 — ggml merge (detail below)
merge-base `e705c5fe` (v0.13.0) → upstream tip `9be31331` (v0.17.0). Fork 131 / upstream 304.
`git merge upstream/master` started; 13 conflicts.

### Conflict resolution log (ggml)
| File | hunks | status | notes |
|---|---|---|---|
| include/ggml-rpc.h | 1 | DONE | assert→107 (merged op count). RPC not compiled in fork; cosmetic. |
| include/ggml.h | 3 | DONE | F8_E4M3 keeps slot 42, upstream Q2_0→43, COUNT→44; ftype F8_E4M3=28,Q2_0→29; col2im_1d dedup (kept upstream enum pos); removed fork's trailing COL2IM_1D. COUNT=107. |
| src/ggml.c | 3 | DONE | NAME/SYMBOL arrays dedup'd (removed trailing col2im), asserts→107, conv_2d took fork branched ver, removed dup ggml_col2im_1d (kept upstream's w/ better asserts). |
| src/ggml-cpu/ops.cpp | 2 | DONE | concat took fork N-way memcpy ver (dropped upstream concat_any); col2im kept UPSTREAM impl (time-axis threading better for mono audio). |
| src/ggml-cpu/ggml-cpu.c | (auto-merge!) | DONE | NOT flagged UU but had DUP COL2IM_1D case labels in forward-dispatch + n_tasks switches. Removed fork's dup cases, kept upstream placement. |
| src/ggml-cuda/col2im-1d.cu | 6 (add/add) | DONE | took UPSTREAM wholesale (adopted fork op + fast_div_modulo opt; numerically identical). |
| src/ggml-cuda/concat.cu | 6 | DONE | took FORK (superset: templated + 4D single-launch fold + profiling). Added BF16 case→half path (bit-copy) for new upstream models. |
| src/ggml-cuda/cpy.cu | 2 | DONE | H1: fork z-chunk transpose (LTX long-video fix) over upstream generic fallback. H2: kept BOTH decl sets — upstream memcpy2D fast-path + fork perm/coal transpose. Verified predicate safe (true transpose nb00!=elsz breaks at d=0). |
| src/ggml-cuda/fattn-common.cuh | 2 | DONE | H1: kept BOTH fork per-head-mask guard (ne[2]==1) + upstream int64_t s31/s33 (kernel sig confirms int64_t). H2: took fork WAN SLA bsa_sync. |
| src/ggml-cuda/fattn-tile.cuh | 4 | DONE | all additive: fork 384 (Wan VAE) config cases + upstream 512 cases. Both kept, 4 tables each. |
| src/ggml-cuda/fattn.cu | 2 | DONE | H1: kept fork cuDNN env helpers + upstream kv_type_supported (used@567). H2: kept upstream get_alloc_size (REQUIRED by ggml-cuda.cu:1082) + fork _dispatch; dropped upstream dup main entry (fork's SA3 entry survives). |
| src/ggml-cuda/mmq.cu | 1 | DONE | trivial: kept fork NVFP4 comment; body calls mul_mat_q_case<T> (upstream mmq.cuh compatible). |
| src/ggml-cuda/mmq.cuh | 4 | DONE (UPSTREAM WHOLESALE) | fork=4199L old-structure, upstream=1495L total rewrite w/ NATIVE NVFP4/MXFP4. All 4 fork hunks were old-API (get_mmq_x_max_host, mmq_x/need_check tmpl, flux2 occupancy EXPERIMENTS default=upstream) — incompatible. **codex-rescue verifying NVFP4 layout compat + landmine gates.** |
| src/ggml-cuda/ggml-cuda.cu | 8 | DONE | H1 includes both. H2+H4: took UPSTREAM linear dispatch, DROPPED fork op_mul_mat multi-GPU-split machinery (0 callers, single-GPU engine). H3: kept fork NVFP4/FP8 cuBLASLt early-returns + upstream Hadamard FWHT. H5/H8: kept fork SNAKE/CONV_1D_DIRECT + upstream LIGHTNING_INDEXER; removed DUP COL2IM_1D (upstream's canonical @2671/@6279 kept). H6: kept BOTH fork FP8-epilogue fusion + upstream gated_delta_net fusion (added 2 braces to close fork scope). H7: kept fork concat supports_op predicate (ships fork concat.cu). |

**Landmine preservation check (all confirmed present in resolved tree):**
- sm120 determinism: fattn bsa_sync + int64 KV_max stride kept.
- fp8 F16-store overflow: ef4435d fork commit is pre-merge-base history, untouched.
- flux2-white sm86: `blackwell_mma_available(cc)` gates ALL fork FP8/FP4 cuBLASLt paths (hunk 3) → sm86 falls back. Kept.
- NVFP4 cuBLASLt repack: nvfp4-cublaslt.cu not conflicted, untouched.
- LTX no-F16-Q sm86: engine-side, not ggml.
- SA3 arch-gating: fattn.cu SA3 entry (GGML_LTX_SA3) kept.

**KEY ARCH DECISION:** MMQ NVFP4 now UPSTREAM-native (dropped fork's hand-rolled mmq.cuh); fork's cuBLASLt FP8/FP4 GEMM (separate path, ggml-cuda.cu hunk 3 + nvfp4-cublaslt.cu) KEPT. Two independent NVFP4 paths coexist. **codex-rescue verifying layout compat — CHECK before declaring safe.**

### Fork custom ops to preserve (from handoff)
SNAKE, CONV_1D_DIRECT, ROPE_PE, COL2IM_1D, CONV_2D_DEFORM, RMS_MODULATE,
NVFP4/FP8 cuBLASLt GEMM, native SageAttention3 backend, cuDNN attn/conv3d fusion,
sm120 Blackwell determinism, VAE-decode kernels, fusion passes.

## Stage 1 BUILD (in longcat-avatar-dev:builder, archs 86;120, SD_CUDA=ON)
- ggml-base: BUILDS ✓
- ggml-cpu: BUILDS ✓ (benign -Wswitch warnings: RMS_MODULATE/RMS_NORM_CHANNELS/F8_E4M3 not in some switches — pre-existing fork behavior, warnings not errors)
- ggml-cuda: fixed 1 real break — upstream removed `ggml_backend_buft_is_cuda_split` + all split-buffer infra; fork had 4 callers. Re-added as stub returning false (replaced dead `ggml_cuda_op_mul_mat_t` typedef) + removed now-unused `const bool split` var. REBUILDING.
- Build cmd: `docker run -v <wt>:/src longcat-avatar-dev:builder` → `cmake -S /src -B build-wtcuda -DSD_CUDA=ON -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES=86;120` → `cmake --build --target ggml-base ggml-cpu ggml-cuda`. Logs in scratchpad ggml_build*.log.

## ✅✅ CORRECTED STRATEGY (owner, 2026-07-19) — this supersedes the pivot below
**"Fork exists because upstream was slow/missing things. When upstream adds/does-better, ADOPT
upstream and shed the hack. Keep fork ONLY where genuinely unique + upstream still lacks it."**
"I skipped it because upstream also has it" is BACKWARDS. Adopt upstream's new models + shared
improvements; keep fork only for: graph-cut VRAM/offload, LTX/Wan/audio video engine, NVFP4/FP8
cuBLASLt GEMM, SA3, longcat-avatar.

### EVIDENCE — the fork is CLOSE to upstream, not divergent (re-measured)
- Fork's dispatch ALREADY uses `backend_for(SDBackendModule::DIFFUSION)` (fork method @ sd.cpp:326,
  used 121×) — the same routing upstream uses. NOT a divergent loader.
- Fork ALREADY integrates: ideogram4, PiD, z-image, ernie-image, lens, hidream, anima, qwen-image,
  flux2 (VERSION_FLUX2/KLEIN, sd_version_uses_flux2_vae all in fork model.h).
- New model .hpp use GGMLBlock/UnaryBlock/GGMLRunnerContext (incl circular_x/y) — fork HAS these.
- The 434 build errors were MY mechanical merge scripts (Frankenstein), NOT real incompatibility.

### THE GENUINE GAP = 7 upstream models the fork has 0 refs to (+ shared improvements)
krea2, boogu, hunyuan (video 1.5), minit2i, sefi_image, animatediff, lingbot_video. Headers already
present at src/model/diffusion/*.hpp + src/model/vae/hunyuan_vae.hpp (committed in 6c0fdf2).

### ⚠️ KEY GOTCHA (measured on krea2) — Runner glue needs per-model adaptation
The new model .hpp are written for UPSTREAM's Runner-construction convention:
  `Krea2Runner(backend, tensor_storage_map, prefix, model_manager)`  ← ModelManager (fork lacks)
The FORK convention threads params_backend for its offload/VRAM system:
  `Ideogram4Runner(backend_for(DIFFUSION), params_backend_for(DIFFUSION), tensor_storage_map, prefix)`
So per model you MUST edit the Runner class in <model>.hpp to the fork's
`DiffusionModelRunner(backend, params_backend, prefix)` base convention (constructor sig + base init +
drop model_manager, add params_backend) AND the conditioner (LLMEmbedder etc.). The DiT BLOCKS port
clean (GGMLBlock/UnaryBlock/GGMLRunnerContext all shared); the Runner WRAPPER is the real work.
Also per-model VAE quirks: krea2 uses `sd_version_uses_wan_vae` (WAN VAE, not flux2). Realistic
effort ≈ 1–2h/model. Model-def blocks compile against fork primitives (verified krea2 uses shared base).

### PORT RECIPE (per model — dispatch uses backend_for/params_backend_for like the fork)
From upstream/master:src/stable-diffusion.cpp, graft onto the fork engine:
1. `#include "model/diffusion/<model>.hpp"` at top.
2. `VERSION_<MODEL>` enum in model.h + name string in the version-name array (sd.cpp ~line 117).
3. `sd_version_is_<model>()` helper in model.h.
4. Version detection (from tensor names) — find upstream's detection for the model.
5. Dispatch case: `else if (sd_version_is_<model>(version)) { diffusion_model =
   make_shared<<Model>::<Model>Runner>(backend_for(SDBackendModule::DIFFUSION), ...); }` (~sd.cpp:1000).
6. Per-model branches: conditioner naming (~2915), VAE (krea2/flux2→flux VAE fork has; hunyuan→
   hunyuan_vae.hpp), scheduler, timestep/edit quirks (grep upstream for sd_version_is_<model>).
7. Build (sd-cli+sd-server, sm86;120, SA3). Then GPU smoke-test.
Krea2 example integration points in upstream sd.cpp: include@35, name@117, dispatch@1001-1008,
conditioner@2915, timestep@3016.

### REMAINING WORK (multi-session, ~the handoff's 1-week estimate)
- Port the 7 models (recipe above), each building. [NOT STARTED]
- Adopt upstream's genuine shared-code improvements (per-hunk, prefer-upstream-where-it-supersedes).
  Fork tracks upstream closely so this delta is smaller than the raw 60-hunk count suggests.
- GPU validation (determinism + per-model smoke renders) — needs a slot.
Fallback tag `wip-fork-plus-ggml-building` = current building fork+ggml base.

## ⚠️ STAGE 2 STRATEGY PIVOT (SUPERSEDED by the corrected strategy above)
The fork and upstream **sd.cpp engine architectures are fundamentally incompatible** — NOT a
conflict-resolution task. Upstream rewrote the loader/runner around `ModelManager` /
`backend_for(SDBackendModule)` / multi-device layer-split; the fork rewrote the SAME layer around
its own single-device graph-cut/offload runner. Per-hunk merging produced a non-building Frankenstein
(467 then 434 errors: mixed class members, ambiguous overloads, `else` without `if`). A true
keep-both here = **per-model PORTING** of each upstream model onto the fork runner, not merging.

**DELIVERED INSTEAD (buildable + honest):** fork sd.cpp (exact) + **upstream ggml** (the 304-commit
freshness — new MMQ/NVFP4, schedulers-in-ggml, etc.). Upstream's new sd.cpp MODEL HEADERS kept as
uncompiled REFERENCE for porting: krea2.hpp, boogu.hpp, hunyuan.hpp, minit2i.hpp, sefi_image.hpp,
animatediff.hpp, lingbot_video.hpp, hunyuan_vae.hpp + docs/assets. REMOVED upstream infra .cpp that
break glob-compile against the fork engine: model_manager, backend_fit, layer_split_partition,
detailer, pulid_extension, imatrix, weight_manager.h.
NOTE: fork's model.h ALREADY has VERSION_FLUX2/FLUX2_KLEIN/IDEOGRAM4/LENS/ERNIE_IMAGE +
sd_version_uses_flux2_vae — the fork already tracked much upstream model support independently.
The mask-broadcast graft was reverted with the fork restore (re-add during porting if wanted).

## Stage 2 — sd.cpp merge (RESOLVED as fork+ggml, full build running)
21 conflicts resolved. ggml submodule pointer → stage-1 merge 531a2c72.

### Resolution summary
- **Additive (kept BOTH):** .gitignore, model.h (VERSION enum), ggml_graph_cut.cpp (includes),
  model.hpp (variant list union), rope.hpp H2 (gen_lingbot_video_ids), conditioner.hpp H1
  (fork VRAM-free + upstream multi-device virtuals), convert.cpp includes, **stable-diffusion.h
  (all 4 — includes FLUX2_SCHEDULER + circular_x/y the owner wants)**, **denoiser.hpp (fork LTX
  + upstream flux/logit-normal schedulers)**, common.cpp (hires + circular), async_jobs includes.
- **Fork (HEAD):** ltxv.hpp (RMS_MODULATE fusion), model_loader.cpp (fork imatrix resolver),
  wan_vae.hpp (fork video VAE + fork compute() sig; upstream is_2D dropped — image models use the
  separate flux VAE), convert.cpp logic, wan.hpp (WAN SLA), rope.hpp H1/H3/H4, examples/cli+server
  (fork CLI/server, coupled to fork generate_image API), **stable-diffusion.cpp (ALL 60 hunks)**.
- **Graft:** ggml_extend.hpp — took fork wholesale (21 hunks, the graph-cut/VRAM/Q-clamp/BSA
  superset) + grafted upstream's broadcast-mask query-dim materialization (Chroma/T5 NaN fix,
  identity for fork masks).

### ⚠️ KNOWN GAP — new-model WIRING in stable-diffusion.cpp NOT ported (scoped follow-up)
The fork rewrote the loader/runner architecture (ModelManager → fork routing; the handoff's "10
refactor commits"). Upstream's new-model wiring in stable-diffusion.cpp (qwen-image-layered decode,
AnimateDiff gen, LTX-upsampler-via-ModelManager, and the load-dispatch for krea2/ideogram4/boogu)
uses upstream's `ModelManager`/`backend_for(SDBackendModule)` architecture — NOT a clean additive
graft into the fork engine. Took HEAD (fork) for all 60 stable-diffusion.cpp hunks → fork engine
is self-consistent + builds, but these new-model LOAD PATHS are NOT wired. What IS present: upstream
model-class *definitions* (auto-merged model/diffusion/*.hpp), new schedulers (flux2!), circular RoPE
API. Substantial-upstream hunks deferred: H32 (qwen-layered), H43 (LTX upsampler refactor), H45
(animatediff), H11/16/17/39/46/51/56/60. Porting each = per-model work + GPU validation.

### Build (stage 2) — fixes applied
1. **ggml_extend.hpp Frankenstein (467 errors):** per-hunk HEAD produced a broken 7202-line mix
   (fork bodies + auto-merged upstream bodies, inconsistent GGMLRunner member decls). FIX: took the
   fork's COMPLETE file (`git checkout HEAD -- src/core/ggml_extend.hpp`, 6800L, self-consistent) +
   re-grafted the broadcast-mask fix. Upstream's ggml_extend changes are architecturally
   incompatible with the fork runner (that's why 21 conflicts) — correct to drop wholesale.
2. **layer_registry.{h,cpp} deleted by merge (8 errors):** merge staged a deletion of these fork
   files (upstream has no equivalent); ggml_extend.hpp includes layer_registry.h. FIX: restored
   from HEAD (`git checkout HEAD -- src/core/layer_registry.*`). CMake globs src/core/*.cpp.
Rebuilding sd-cli+sd-server. NOT committed until clean.
