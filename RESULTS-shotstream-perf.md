# ShotStream perf session — results (2026-07-04, 3060 / device 0)

Branch `shotstream-master` = cherry-pick of `07573ea` (the port) onto `origin/master` (`05191b5`,
post wan22+longcat merge), ggml synced to `a6398094`. Built arch **86;120** (runs on 3060 + 5060).

## STEP 1 — rebase + parity (DONE)
- `wan.hpp` merged additively (forward_kv_cache + forward_causal/_block + flash_skip_kv_pad + F16 KV concat all present).
- Block-0 op-parity vs the torch oracle: **cos 0.99999** (matches the port's original figure) — ggml bump preserved forward numerics.
- 2-shot fox chain visually clean.

## Baseline (resident KV, safe 9-tile VAE)
| | value |
|---|---|
| DiT | 83.5 / 84.3 s per shot (faster than old notes' 112–146 s — resident KV + newer ggml) |
| VAE decode | 67.8 s/shot (0.5×0.5, 9 tiles) |
| wall | ~152 s/shot · peak 11905 MiB |

## Fixes / levers this session (all env-gated)

### 1. VRAM decode-OOM fix — EXACT (main.cpp, default on; opt-out `SHOTSTREAM_KEEP_KV_FOR_DECODE=1`)
The ggml bump pushed the at-the-edge chunks=7 chain over 12 GB during VAE decode (resident DiT weights +
~6–8 GB resident KV + 3.5 GB decode buffer). Fix: free this shot's resident KV caches (dead until next
shot — global rebuilt from decoded-pixel round-trip, local reset by run_shot) before the decode. Unblocks
the chain; also lowers peak.

### 2. Exact last-chunk-rewrite skip — EXACT (default on; opt-out `SHOTSTREAM_LASTCHUNK_REWRITE=1`)
The **last** chunk of a shot persists clean K/V that are **never read** (no next chunk; next shot resets the
local cache; decode consumes the latent, not K/V). Skipping that forward is **bit-identical** — verified:
shot-0 latent byte-for-byte equal to the baseline. Drops 1 forward/shot AND removes the chunk-6 alloc that
fragments the pool + OOMs after a fast VAE decode. (Also applied to Lever A's last-denoise persist.)

### 3. Fast 4-tile VAE in-chain — enabled by #1+#2 (`--vae-relative-tile-size 0.57x0.57 SHOTSTREAM_VAE_OVERLAP=0.15`)
| | value |
|---|---|
| VAE decode | **39 s/shot** (was 67.8) — −43% |
| wall | **~120 s/shot** (was 152) — −21% |
| peak | 11.28 GB · no OOM · no tile seams · shot-0 latent bit-identical |
Clip: `perf_fastvae_robust_2shot.mp4` on `:8077/shotstream/`.

### 4. Lever A — skip clean rewrite (`SHOTSTREAM_SKIP_CLEAN_REWRITE=1`) — APPROXIMATION, owner to judge
Drops the 5th (clean-rewrite) forward/chunk; caches the last DENOISE step's K/V instead. Those come from a
σ≈0.74 (~74%-noise) input, **not** the clean x0 — so NOT bit-exact.
| | value |
|---|---|
| DiT | **67 s/shot** (was 84) — −19% |
| wall | ~107 s/shot (with fast VAE) — −30% vs baseline |
- **REJECTED (owner motion eye-test).** Static frames looked clean, but in motion the **background
  "re-renders as it goes" — flashes, sharpens, changes** across the chain. Mechanism: the clean rewrite
  caches CLEAN K/V so past frames are a *stable* memory; Lever A's noisy K/V make the bg memory unstable →
  the model re-interprets/re-sharpens it each chunk. The clean rewrite is **load-bearing** for temporal
  consistency. Env gate defaults OFF (clean rewrite runs) — nothing to revert. Do not re-chase this.
  (Another "pixel-metrics / static-frame lie" — always motion-judge chaining.)
Clip: `perf_leverA_2shot.mp4` on the page (marked ❌).

## Profiling (SHOTSTREAM_PROFILE, chunks=2)
- denoise forward: 0.87 s (loc=0) → 1.13 s (loc=4680). **clean-rewrite/persist forward: 1.41 s** = +0.54 s
  pure overhead from the K/V D2H readback + host re-upload + sync → the target for exact lever **E5**.
- last-chunk skip verified live (last chunk runs 4 forwards, no 5th).

### 5. E5 — in-graph K/V append — EXACT (default on under SHOTSTREAM_NO_OFFLOAD; opt-out `SHOTSTREAM_INGRAPH_KV=0`)
Persist the fresh K/V by an in-graph `ggml_cpy` straight into a pre-allocated persistent chunk buffer,
instead of exporting them as graph outputs → D2H readback → host re-upload. Removes the GPU→host→GPU
round-trip + its sync on every persist forward.
| | value |
|---|---|
| persist forward | **1.42 → 0.88 s** (= a denoise forward; the +0.54 s round-trip gone) |
| per shot | ~3.8 s (6 clean-rewrites + 1 prefill × 0.54 s) — ~4.5% DiT |
| parity | **latent BIT-IDENTICAL** (chunks=2 A/B) |
Kept the per-chunk resident buffers (did not need E4's contiguous-buffer refactor first).

## Post-E5 nsys — the 3060 DiT is at its practical floor
DiT-only chunks=3, resident + E5 (`shotstream_out/nsys_e5`). Kernel GPU time 15.33 s vs real
(non-nsys) DiT wall ~15.7 s → **GPU ~97% saturated** (no launch-bound idle → CUDA graphs give ~nothing).
| kernel | % | verdict |
|---|---|---|
| `flash_attn_ext_f16` (attention) | 36.8% | Ampere floor (occupancy-bound MMA, register-capped by design) |
| `ampere_h1688gemm` ×2 (FFN/proj/cross) | 24.7% | cuBLAS roofline |
| `convert_unary` F16↔F32 casts | 10.5% | WAN_DIT_F16 recovers only ~2% on the causal path (native FA2 needs F32 KQV → attention boundary keeps the casts) |
| E5 `cpy_scalar` (K/V append) | 3.4% | replaced the worse host round-trip |
| `concat_T_cont_4d` (KV concat, E4) | 2.7% | too small to justify E4's strided-buffer refactor |
| norms/adds/rope/glue | ~13% | GPU-executing, not launch-idle |

**Conclusion:** attention (floor) + GEMM (roofline) = 62% at hardware limits; the rest is GPU-bound glue.
E4 (2.7%, risky), WAN_DIT_F16 (2%, approximation), CUDA graphs (~0, GPU saturated), custom attention
kernel (weeks, kernel already at its occupancy ceiling — cuDNN gives it free on Blackwell). **The 3060
DiT is floored. The −26% wall this session (fast VAE + E5 + last-chunk skip, all bit-exact) is the 3060
result; the next big step is the 5060 (2× compute + free cuDNN SDPA + FP4 + 16 GB).**

## Remaining levers (see tasks / perf-ideas.md)
- **E4** fixed contiguous resident KV buffers (kill per-forward concat, launch-bound ~5-7% DiT) → **E5**
  in-graph K/V append (remove host round-trip, ~4-5% DiT). Meatier exact DiT levers.
- E1 text hoist / E3 timestep hoist / E6 RoPE cache — small exact.
- E2 need_flow=false (skip head/unpatchify on K/V-only forwards) — tiny (head is 1536→64).
- **5060/Blackwell** (ON HOLD — owner using device 1): cuDNN SDPA gate at `fattn.cu:446` matches the
  shotstream self-attn shape EXACTLY (D=128, mask-free, F16 KV, gqa=1) → `GGML_CUDNN_ATTN=1` should route
  self-attn to fused cuDNN SDPA for free. Then WAN_DIT_F16 + FP4 + bigger VAE tiles (16 GB).
- 3060 DiT is otherwise at the Ampere attention floor (occupancy-bound MMA, register-capped by design).
