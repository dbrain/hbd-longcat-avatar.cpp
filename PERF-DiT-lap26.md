# DiT perf — lap-26 floor-proof (2026-05-28)

Fresh ncu ground-truth on the RTX 3060 (sm_86, 28 SM, 48 warps/SM max). Standard render,
`--steps 1`, 25f/480×832 → ~10920 self-attn tokens (7 latent frames × 1560 tok/frame),
n_cond_tokens=1560 (1 ref frame), L_noise=9360.

## DiT step op-profile (LONGCAT_OP_PROFILE, serialized/inflated; proportions exact)
graph nodes=10775, one step ≈ **17.28 s/it** wall (×8 ≈ 138 s).

| op | % | calls |
|---|---|---|
| MUL_MAT | 46.0 | 826 |
| FLASH_ATTN_EXT | 33.7 | 144 |
| ADD | 5.8 | 1020 |
| MUL | 3.6 | 434 |
| SCALE | 2.7 | 819 |
| CONT | 2.4 | 1543 |  ← lap-24 win held (was 4.3%)
| NORM | 1.5 | 242 |
| CONCAT | 1.2 | 145 |
| ROPE_PE | 1.2 | 96 |

144 flash calls = 48 blocks × 3 (cond self-attn + noise self-attn + text cross-attn; audio
cross-attn is non-flash materialized).

## ncu — FLASH_ATTN_EXT (the 33.7%)
Kernel **`flash_attn_ext_f16<128,128,64,1,0,0>` = the MMA kernel** (DKQ=128, DV=128, ncols1=64,
ncols2=1, softcap=0, V_is_K_view=0). NOT the WMMA kernel (its 3rd param is nwarps, can't be 64) —
dispatch is correct (`turing_mma_available(sm_86)`=true → MMA path, fattn.cu:478).

Dominant noise self-attn launch (grid 4704): **gpu_time 141 ms** (serialized).
- sm__throughput **30%**, dram **2–13%**, tensor_op_hmma **30%**
- **warps_active 16.6%** (2 blocks/SM), **registers/thread 191** → register-capped occupancy
- ⇒ **latency/occupancy-bound, NOT compute-bound.** lap-21's "63% of roofline" was wrong.

## ncu — MUL_MAT (the 46%)
Kernel **`mul_mat_q<12,128,0>`** = Q4_K MMQ (type 12 = Q4_K). Dominant per-block launches
(grid 2752) **13.3 ms each**; cold/larger one 35 ms.
- sm__throughput **54%**, dram **39%**, tensor_op_hmma **0** (expected — MMQ is *integer* IMMA/dp4a, not HMMA)
- warps_active **16.67%** (1 block/SM of 8 warps)
- `quantize_mmq_q8_1` (Q8_1 activation quant) healthy: 93% DRAM, 82% occupancy, ~0.7 ms.

### pf32 precision lever — DEAD (proven, not asserted)
mul_mat dispatch (ggml-cuda.cu:2551–2607) picks MMQ purely on `ggml_is_quantized(src0)` + shapes;
**it never reads `dst->op_params[0]` (the prec flag).** So `GGML_PREC_F32` does NOT push Q4_K onto a
dequant→cuBLAS path — Q4_K always uses MMQ. The prec flag only affects the F16-weight matmuls
(F32- vs F16-accumulate), which on Ampere tensor cores is the same speed. ⇒ dropping pf32 saves
~nothing. Handoff lever-3 "precision" angle closed.

MUL_MAT is fundamental FFN(w1/w2/w3) + qkv/proj compute at the established MMQ floor (turboquant laps:
Q4_K is the Ampere floor, Q3_K +7% slower). Irreducible without changing the model.

## Verdict so far
- **MUL_MAT 46% = genuine floor** (MMQ, irreducible compute, precision lever dead).
- **FLASH 33.7% = the only real headroom.** Kernel is occupancy-starved (16.6%), register-bound.
  Two angles: (a) MMA tile/occupancy tuning — bit-exact but fork-class register surgery; (b)
  **block-sparse work elimination** (~8× FLOP cut, algorithmic, quality trade, handoff lever-1).
## Flash MMA tile sweep (bit-exact A/B) — DEAD, and it proves the floor
Env knob `LONGCAT_FA_NCOLS1` added to fattn.cu's `switch_ncols1` (default-off diagnostic; forces a
smaller query-tile, bit-exact same softmax math). FLASH_ATTN_EXT serialized ms @ 25f/steps=1:

| ncols1 | FLASH ms | vs 64 |
|---|---|---|
| 64 (auto) | 5640 | — |
| 32 | 9380 | +66% |
| 16 | 12197 | +116% |

**Monotonic: the kernel strictly wants the BIGGEST tile.** Smaller tiles raise occupancy but lose K/V
reuse → more total K/V streaming. So flash is NOT occupancy-starved in a usable way; ncols1=64 is the
tradeoff optimum. Bit-exact tile lever closed.

## Flash stall-reason ncu (the "why latency-bound") — floor PROVEN
Dominant noise self-attn kernel, warp-stall sample counts:
- **long_scoreboard 1.82M (~62%)** — memory latency on K/V loads (dominant)
- **math_pipe_throttle 853K (~29%)** — tensor/ALU pipe busy
- wait 452K, barrier 173K
- l1tex 24.9%, **lts (L2) 19–26%**, smem_per_block_static=0 (dynamic smem)
- occupancy capped at **2 blocks/SM by BOTH registers and shared_mem**

⇒ memory-latency-bound on K/V streaming for dense O(N²) attention, tensor pipe also well-fed. The big
tile already minimizes K/V traffic (attacking the long_scoreboard root cause) at the cost of occupancy;
ggml's auto-choice is the optimum (measured). The only way to cut the K/V-streaming root cause is
**less K/V data = sparsity**, which the reference (`proof_gen.py:61-62`) DISABLES for the avatar
(off-distribution). So flash is at its floor for the avatar's dense design on sm_86.

## FLOOR VERDICT (both fish, with receipts)
- **MUL_MAT 46%**: Q4_K MMQ, 54% SM / 39% DRAM, pf32-irrelevant (dispatch ignores prec). Fundamental
  FFN+qkv+proj compute (M=10920). Q4_K is the established Ampere floor (Q3_K +7% slower). Irreducible.
- **FLASH 33.7%**: MMA kernel, memory-latency-bound on K/V streaming, tile-config-optimal (monotonic),
  dense = the reference's own avatar setting. At floor.
- **Cross-attn KV cache (lever 2)**: cacheable matmuls are kv_linear at M=512 (text) / M=32 (audio) vs
  the M=10920 FFN — by FLOP arithmetic <1% of the step, and needs cross-graph persistence. Skip (matches
  handoff's "<2% → skip").
- **Block-sparse (lever 1)**: reference disables it for the avatar; off-distribution; modest at 25f
  (only 7 latent frames). Not a safe win.

## ★★★ HEADLINE LEVER — cond-frame cross-step recompute (12% bit-exact, the missed stone)
The "MUL_MAT is floored" verdict proved the KERNEL is efficient (cuBLAS-corroborated 36 TF), but NOT
that the WORK is necessary. It isn't. ~14% of every big matmul is **7× redundant**:

**Proof (by construction, airtight):**
- `stable-diffusion.cpp:4863-4864` (ai2v) / `:4932` (avc) / `:4967`: `denoise_mask = full(1)` then
  `fill_slice(dim2=temporal, [0,num_cond), 0.0)` → cond frames masked to **exactly 0**.
- `:2076`: `noised_input = noised_input*denoise_mask + init_latent*(1-denoise_mask)` → cond frame DiT
  input = **`init_latent`, constant every step** (no noise, no per-step c_in effect — it's replaced).
- `:1706`: cond timestep pinned to **0** every step → cond adaLN(silu(t=0)) modulation constant.
- DiT is deterministic ⇒ the cond frame's ENTIRE 48-block forward (qkv, cond-self-attn, FFN, proj,
  per-block residual stream + cond K/V) is **bit-identical across all 8 steps**. Noise tokens attend to
  that step-invariant cond K/V.
- For 25f ai2v: num_cond_latents=1 latent frame = n_per_frame=1560 of 10920 tokens = **14.3%**.

**The win:** compute the cond forward ONCE; steps 1–7 process only the 9360 noise tokens, pulling cached
per-block cond K/V for the noise self-attn. Bit-exact (PSNR 99 gate IS the empirical proof).
- Cost model: baseline 8×100%. Cached = step0 100% + 7× ~86% (noise-only) = 702% ⇒ **−12.25% of DiT
  sampling wall** (= ~−9.5% clip wall). Stacks on FFN, qkv, proj, cond-attn-pass, cond modulation.

**Implementation (multi-session, the hard plumbing the handoff flagged for lever-2 but bigger):**
1. Persistent cross-`compute()` buffers for per-block cond K/V (post-RoPE): 48 × 2 × [128,32,1560] F16
   ≈ 1.2 GB (fits in ~5.9 GB headroom; or cache cond_x per block ≈ 613 MB and recompute cheap cond qkv).
2. Split DiT forward into cond-only (run once, fills cache) + noise-only (per step, reads cache). The
   self_attn noise pass already slices q_noise; extend it to concat [cached cond k/v ++ fresh noise k/v].
3. Sampler: detect cond is fixed (denoise_mask has a 0-frame prefix) → enable the cache path; invalidate
   if init_latent/num_cond change. Guard: only when cond frames are contiguous prefix + mask exactly 0.
4. Gate: PSNR 99.00 vs 8-step baseline `step8_s42`-equiv (bit-exact) — that gate passing PROVES the
   redundancy. Then 8-step sampling-wall A/B for the measured win.
Risk: the graph-rebuild-per-step model + ggml-alloc view-liveness (see lap-20) — persistent buffers must
live outside the per-step gallocr. Tag the work-elim carefully; verify last-frame coherence too.

### Turnkey injection points (surface mapped lap-26)
- **`diffusion_model.hpp` DiffusionParams (struct @ :18)**: add `int step = -1;` (and optionally
  `bool cond_kv_cache = false;`). Set `step` in the denoise lambda (`stable-diffusion.cpp:2043`, the
  lambda already has `step`) where diffusion_params is populated before `diffusion_model->compute`.
- **`diffusion_model.hpp` LongCatAvatarModel::compute (:484)**: already derives `num_cond_latents` from
  leading-zero timesteps (:492-509) — reuse that as the cond-fixed signal. Thread `diffusion_params.step`
  → `avatar.cur_step` before calling `avatar.compute`.
- **`longcat_avatar.hpp` LongCatAvatarRunner**: add persistent cond-K/V cache members + a dedicated
  `ggml_backend_buffer_t` (allocated once in alloc_params_buffer or lazily; NOT the per-graph gallocr).
  `build_graph` (:1148): branch on `cur_step` — step 0 (or cache empty) = full graph but capture each
  block's post-RoPE cond k/v into the persistent buffer (graph output → `ggml_backend_tensor_get`/copy);
  step>0 = noise-only graph, cond k/v injected as leaf inputs from the persistent buffer.
- **`self_attn` (:209)**: the cond/noise split already exists. Cache path: noise pass concats
  [cached cond k/v ++ fresh noise k/v]; skip the cond query pass + cond proj/FFN output on step>0. The
  cond residual stream is step-invariant so the whole cond column can be frozen (cache its per-block x or
  just its per-block k/v — k/v-only avoids re-deriving but needs 48×2 buffers).
- **Gate**: env `LONGCAT_COND_CACHE=1` (default off → byte-identical). PSNR 99.00 vs 8-step baseline = the
  empirical proof of step-invariance. Then 8-step sampling-wall A/B for the measured ~12%.
- **Discipline**: build green + default byte-identical every commit (flag default-off), so cross-session
  work never leaves a broken tree.

### Persistence mechanism — CONFIRMED works on the resident (25f) path (traced lap-26)
The runner already has cross-`compute()` tensor persistence (the graph-cut/offload cache), usable directly:
- `GGMLRunnerContext::persist_cache_tensor(name, t)` (ggml_extend.hpp:1712) → registers t in `cache_tensor_map`.
- `GGMLRunnerContext::load_cache_tensor(name)` (:1705) → `get_cache_tensor_by_name`, returns the persisted
  device tensor (lives in `cache_ctx`/`cache_buffer`, SEPARATE from the per-step gallocr → lap-20-safe).
- End of EVERY `compute()` (resident path: `execute_graph` @2905 passes no `cache_keep_names` ⇒ **nullptr**),
  `copy_cache_tensors_to_cache_buffer(nullptr)` (:2591) persists ALL registered + **merges forward all
  prior** cache tensors (:1962-1976). So: persist cond k/v at step 0 only; they auto-survive steps 1–7.
- Ordering: load during build_graph (reads prior step's cache_buffer, still valid) → persist/merge at end
  of compute. Cached leaf has a buffer ⇒ gallocr won't realloc it (same as weights). Round-trip is clean.

**Build recipe (resident ai2v, num_cond_latents=1):**
1. step 0 (cur_step<=0): full graph as today; in `self_attn` 2-way split, after building `k_cond`/`v_cond`
   (the cont'd [0,n_cond) slices, post-RoPE for k), `ctx->persist_cache_tensor("b{i}_ck"/"b{i}_cv", …)`.
   Also persist the final cond-output slice ("cond_out") to restore output shape on step>0.
2. step>0 (cache present): build a NOISE-ONLY graph — residual stream x = noise tokens (9360) only.
   Each block: modulate/qkv/proj/FFN on noise only; self_attn noise pass attends
   `concat(load_cache_tensor("b{i}_ck"), noise_k)` / same for v. RoPE pe sliced to noise positions
   [n_cond,n_token). Skip cond query pass + cond proj/FFN entirely. At output, concat cached "cond_out"
   ++ noise_out to restore full token count (cond region is overwritten by init_latent in the sampler
   anyway, so exactness there is free).
3. Gate `LONGCAT_COND_CACHE=1`; PSNR 99 vs 8-step baseline = proof; then 8-step wall A/B.
Watch: RoPE position offset for noise (must match the full-graph positions), concat order (cond-then-noise),
n_per_frame divisibility, and the audio/text cross-attn cond-zeroing (already cond-aware — noise-only graph
just drops the cond rows it used to zero).

### Build status (lap-26, in progress)
- DONE + built green: plumbing (`DiffusionParams.step`→`runner.cur_step`→`GGMLRunnerContext.sampler_step`
  + `cond_kv_cache`, env `LONGCAT_COND_CACHE`) and the PERSIST half — `self_attn` 2-way split persists
  `k_cond`/`v_cond` as `longcat.condkv.b{i}.{k,v}` at sampler_step<=0 when enabled. Default off = byte-identical.
- NEXT (consume path, the hard 80%): in `LongCatAvatar::forward` (block loop @1066), when
  `ctx->cond_kv_cache && ctx->sampler_step>0`: slice x→noise [n_cond:], t_emb→noise frames, pe→noise
  positions [n_cond:], T→T_noise, run blocks noise-only; in `self_attn` add a consume branch that loads
  `load_cache_tensor("longcat.condkv.b{i}.{k,v}")` and does the noise pass as
  `q_noise × concat(cond_cached_krope, noise_krope)` (concat reconstructs the exact full K/V because the
  cached cond k was RoPE'd with positions [0,n_cond) at step 0). Restore output shape after final_layer by
  concatenating a cached cond-output slice (persist "longcat.cond_out" at step 0); sampler overwrites the
  cond region with init_latent anyway so exactness there is free. Gate: PSNR 99 vs step-8 baseline.

## (dropped) DMD step count (4≈8) — OWNER SAYS NO
The model is DMD-distilled. Reference default = 8 steps (`proof_gen.py:182`). DiT sampling scales ~linearly
with steps, so 8→4 is a ~2× sampling-wall win IF quality holds — a sampling-loop knob, NOT a kernel fight,
and it dwarfs any kernel-level lever. Sweeping 8/6/4/2 (× 2 seeds), gating on ac16 flatness incl. last
frames + visual. See step_sweep results below.
