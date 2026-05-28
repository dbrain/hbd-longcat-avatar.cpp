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

### VERIFY BLOCKER (lap-26): cache buffer OOM — F32 too big
8-step A/B: baseline OK (137.77s). Flag-on **OOM** at `copy_cache_tensors_to_cache_buffer` (ggml_extend
.hpp:2006) allocating **2.45 GB** — the cond K/V are post-RoPE **F32**: k_cond [128,1560,32] + v_cond
[128,32,1560] = 6.39M elts ×2 ×48 blocks ×4B ≈ 2.45 GB, on top of 8.5 GB resident weights on a 12 GB card.
Consume logic did NOT crash (no assert/garbage) — pure footprint. Fix options (next tick):
- **(A, preferred) F16 PRE-SCALED cache + direct flash in consume.** Cache `F16(k_cond*kv_scale)` /
  `F16(v_cond*kv_scale)` (kv_scale=1/256 — required so v's ~1e6 magnitude doesn't overflow F16) → 1.23 GB.
  Consume: `F16(k_noise*kv_scale)`/`F16(v_noise*kv_scale)`, concat F16, then call `ggml_flash_attn_ext`
  DIRECTLY (mirror `ggml_ext_attention_ext`'s build_kqv: softmax scale = (1/√d)/kv_scale, out *= 1/kv_scale).
  CANNOT reuse the wrapper with pre-scaled k/v — its kv_scale both scales k/v AND folds into the softmax
  scale + output rescale, so injecting pre-scaled k/v double-applies. Bit-exact because the wrapper casts
  to the same F16 internally anyway; the /256 is an exact F16 exponent shift (mantissa preserved).
- **(B, simpler/robust) cache cond_x (block input) F32** = [4096,1560]×48 ≈ 1.23 GB, bit-exact, reuses
  the unchanged attention path. Cost: consume recomputes cond qkv+RoPE (1560 tok, ~14% of qkv — erodes
  the qkv portion of the win, but FFN/proj/attn-output savings remain, the bigger chunks).
- (C) cache_buffer on CPU backend + per-step upload — likely too slow / not how the mechanism binds.
Lean (A) for full win + bit-exact; (B) if (A)'s direct-flash replication proves fiddly to verify.

### VRAM wall — the deeper blocker (lap-26, root cause)
F16 prescaled cache (1.23 GB) STILL OOMs, now at the **compute buffer** (4.18 GB) at step 1, not the
cache. Root cause = the capture/persist mechanism marks cond K/V as graph **outputs** (`set_output`),
which the allocator CANNOT recycle. So step 1 must hold simultaneously: full-step compute transients
(~2.5-3 GB, dominated by the FFN `[11008×10920]` SwiGLU triple) + 1.23 GB non-recyclable cond-K/V
outputs + (at copy time) the 1.23 GB cache buffer — on top of 8.5 GB resident weights. Peak ~13.9 GB
≫ 11.9 GB card. (The F16 prescale's `scale→cast` even adds F32 intermediates, making the compute
buffer worse, not better.) This is why "work already being done" OOMs: the *store* forces the cond
state to be RESIDENT (all 48 blocks at once) where the dense path kept it TRANSIENT (one block,
recycled). It's a recompute→store trade and the 8.5 GB weights leave too little room for the store.

**Proper fix (next): direct-to-persistent-buffer write + FFN tiling.**
- Avoid the `set_output` double-residency: pre-allocate a persistent buffer (own ggml_context +
  `ggml_backend_alloc_ctx_tensors`, 96 named F16 tensors [d_head,n_cond,heads] / [d_head,heads,n_cond,N],
  ~1.23 GB) ONCE at step 1; in-graph `ggml_cpy` the (prescaled-F16) cond K/V directly into those
  persistent tensors. They're not compute-buffer outputs → the cond K/V stay transient/recyclable; only
  the 1.23 GB persistent buffer is added. Consume references those persistent tensors directly (no
  load_cache_tensor / no capture path).
- Force FFN token-tiling whenever the cache is active (bit-exact, per-token) to shrink the step compute
  buffer. Then step-1 peak ≈ 8.5 (weights) + ~1.5 (tiled compute) + 1.23 (persistent cache) ≈ 11.2 GB —
  fits with ~0.7 GB headroom. Consume steps (noise-only 9360 tok, tiled) fit similarly.
- Fallback if still too tight: --offload-to-cpu (frees weight VRAM) but adds per-block weight streaming
  that likely eats the 12% — measure before committing to that path.
The capture/`cache_tensor` mechanism is wrong for 48×2 large per-block tensors (built for a few graph-cut
boundaries); the direct persistent buffer is the right tool. Lever is still a real ~12% bit-exact win
in COMPUTE; the fight is purely fitting the 1.23 GB store on a 12 GB card w/ 8.5 GB weights.

### ROOT CAUSE FOUND (lap-26): cache mechanism wrong for resident path
Bisected at 320×448 (fits, so correctness-comparable): cache-on PSNR **12 dB** (garbage) and F16 mode
**segfaults (exit 139)**. Tap dumps (`b0_xm`/`b0_xs` via `ctx->capture_tensor`): block-0 modulate INPUT
matches (max 0), self-attn OUTPUT diverges **30×**. Decisive test `LONGCAT_COND_NOCAT` (noise-only,
NO cond concat) runs **clean (exit 0)** → the noise-only STRUCTURE (forward slicing of x/t_emb/pe/audio,
rope on sliced pe, attention, ffn) is SOUND. The bug is purely the **cached-cond concat**: the runner's
`persist_cache_tensor`/`load_cache_tensor` (graph-cut/offload cache) does NOT yield a valid resident-graph
leaf — the offload path rebinds buffers via `bind_segment_cached_inputs`; the resident path references the
`cache_ctx` tensor raw and it reads garbage / faults. F32 bisect also wrong → not the F16 round-trip.

**FIX (compact direct persistent buffer, ~45 LoC):**
- `GGMLRunnerContext`: add `std::vector<ggml_tensor*>* condkv_k/condkv_v` + `std::vector<ggml_tensor*>* cache_writes`.
- Runner: members `ggml_context* condkv_ctx; ggml_backend_buffer_t condkv_buf; vector<ggml_tensor*> condkv_k/v; int64_t condkv_ncond`;
  `ensure_condkv(d_head,n_cond,heads,N,n_layers)` — lazily ggml_new F16 tensors in own ctx +
  `ggml_backend_alloc_ctx_tensors(runtime_backend)` (allocated like params → gallocr respects as leaf).
- build_graph: `ensure_condkv(...)` when cache active; set ctx ptrs; after `build_forward_expand(gf,out)`,
  loop `build_forward_expand(gf, w)` for each collected cache_write.
- self_attn step1: `auto w = ggml_cpy(prescaled_f16_kcond, (*ctx->condkv_k)[block_idx]); ctx->cache_writes->push_back(w);` (+v).
- self_attn step>1: use `(*ctx->condkv_k)[block_idx]` directly (cast F32, unscale ×256, concat). No load_cache_tensor.
- This kills the double-residency (cond k/v written straight to persistent buf, not set_output) AND fixes the
  segfault (proper leaf). Then re-verify PSNR99 @320×448, then FFN-tile for 480 VRAM fit. Diag toggles
  (LONGCAT_COND_CACHE_F32 / _NOCAT, b0_x* taps) can be removed after.

## ⚠️ OFFLOAD IS BUGGERED (lap-26 finding — FIX IN A FUTURE LAP)
`--offload-to-cpu` + the cond-K/V cache = **PSNR 12 dB** (garbage), vs 99 dB bit-exact on the resident
path. This is the **recurring offload/gallocr PSNR-ruining bug** (same class as the lap-20 ggml-alloc
view-output liveness bug — "why does this keep happening"). The offload SEGMENTED graph-cut path
(`bind_segment_cached_inputs`) does not correctly bind cross-`compute()` persistent leaves: my direct
condkv buffer (and likely any persistent tensor referenced as a resident-style leaf) gets clobbered /
mis-bound when each block becomes a segment. Symptom: massive PSNR loss, not a crash.
- **For now:** cond-cache is RESIDENT-ONLY (gated; offload disables it implicitly — actually it does NOT,
  so the prod gate must also exclude offload, OR offload must be fixed). TODO: make `cond_kv_cache` also
  require `!offload` until the offload path is fixed.
- **Future lap:** audit the graph-cut segmented path's handling of persistent/cross-graph leaves
  (bind_segment_cached_inputs + the cache_buffer realloc-per-compute) — this bug recurs every time
  something persistent meets offload. The resident path is clean; offload is the liability.
- Also: offload VRAM peak measured ~3.8 GB but that was a sampling trough (offload streams per-segment;
  true peak likely ~8 GB). Offload step overhead measured ~12% here (uncharacterized recently — owner
  has avoided offload by design). Not the path to optimize around.

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
