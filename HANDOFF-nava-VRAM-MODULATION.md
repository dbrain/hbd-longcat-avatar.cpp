# NAVA DiT VRAM — modulation 2-timestep collapse (bit-exact) — HANDOFF 2026-06-05

Branch `nava-port`. **For a FRESH session** — implement the rewrite below from the spec, drive the
build→render→measure loop from the MAIN LOOP (subagents stall on GPU jobs). The DESIGN is done
(a fresh read-only subagent produced the spec, reproduced verbatim in §SPEC). Implementation is
mechanical-but-delicate index bookkeeping — do it fresh, validate bit-exactness hard.

## Why (measured this session)
The DiT compute buffer is **1771 MB** (q5_K DiT peak 6855 MiB = weights 4850 + buffer 1771 + ctx 227).
`NAVA_DIT_VRAM_HIST=1` (env-gated probe, in src/nava.hpp build_graph) shows it's dominated by the
AdaLN **modulation** tensor `[3072, 6, 5148]` = **379.6 MB each, ~40 of them, 4-5 live at peak**.
It's redundant: the per-token timestep has only **2 unique values** (0 for the i2v clean-anchor
prefix, `tval` for the rest — main.cpp:1057-1067). Computing modulation on 2 timesteps and expanding
piecewise is **bit-identical**. Tier-2 below it is the FFN `[18432,5148]` (379.6 MB, real, not
dedupable) → so the fix should drop the buffer 1771 → ~800-1000 MB → **DiT peak ~6000-6150 MiB on
q5_K** (vs 6855). Double-dips: those `[3072,6,5148]` ADDs are also part of the k_bin_bcast DiT mass.

## Repro / mechanics
- Build: `export PATH=/mnt/hdd/3d/avatar-shootout/toolchain/bin:$PATH; export LD_LIBRARY_PATH=/mnt/hdd/3d/avatar-shootout/toolchain/lib; cmake --build build-nava --target nava -j8`
- Probe: `NAVA_DIT_VRAM_HIST=1 ... nava render ... --steps 1` → top-35 tensors + op-bytes + "nava compute buffer size".
- Locked render config (q5_K is the chosen prod quant this session): see HANDOFF-nava-PERF-BIBLE-KICKOFF.md, swap `--gguf models/nava-dit-q5_k.gguf`.
- VRAM peak: sample `nvidia-smi --query-gpu=memory.used` during a real render (NOT profile mode).
- Bit-exact validation: `NAVA_DUMP_AUDIO_LATENT=a.bin` + `NAVA_DUMP_WAV=w.bin`, same seed, current-vs-patched, assert audio-latent cos≈1.0 and ~0 waveform diff. Compare CUDA-patched vs CUDA-current (CUDA attention has pre-existing nondeterminism; don't compare to CPU).

## SPEC (verbatim from the design subagent)

### A. Origin of `[3072,6,5148]`
`time_embed` (nava.hpp:687-701) runs on the per-token timestep `t` (ne=[L_total]=5148 for i2v) →
`e0 = [dim,6,5148]` (379.6 MB). Sliced per stream in build_graph:839-844 (e_vid/e_audio/e_single).
The 379.6 MB **ADD** the profiler flags is `e = ggml_add(e, modparam)` inside `chunk6`
(nava.hpp:326-337) and the inline equivalent in `SingleBlock::forward` (464-471), created ONCE PER
BLOCK: 10 DoubleBlocks × 2 (chunk6 for e_vid+e_audio) + 20 SingleBlocks + 2 NavaHeads = ~40.
Each chunk6: `add[dim,6,L]` → `ggml_ext_chunk(e,6,1)` (6× cont [dim,1,L]) → 6× cont(permute→[dim,L,1]).

### B. Rewrite — Strategy (i): keep `e` compact `[dim,6,2]`, expand chunks per-token at the end.
Chosen over Strategy (ii) (split every modulate into 2 token-blocks) because (i) is localized to the
4 chunk-builders; **block bodies (mod_add/mod_mul/modulate/gated-residual) are UNTOUCHED** — they
still receive `[dim,L,1]` chunks, so the downstream op graph is byte-identical.

Prereq in build_graph: build compact `t2 = [0.0, tval]`, run time_embed on it → `e0c=[dim,6,2]`,
`e_time_c=[dim,2]`. Convention: **column 0 = anchor (t=0), column 1 = tval**.

New `chunk6(ctx, e/*[dim,6,2]*/, which, n_anchor, L_stream)`:
```
e  = ggml_add(ctx, e, modparam)            // [dim,6,2]+[dim,6,1] -> [dim,6,2]  (0.15 MB)
es = ggml_ext_chunk(ctx, e, 6, 1)          // 6× [dim,1,2]
for each chunk c:
    c  = ggml_ext_cont(ggml_ext_torch_permute(c, 0,2,1,3))   // [dim,2,1]
    if e->ne[2]==1: KEEP OLD BODY (text mode, no-op)         // <-- fast path, gate on ne[2]>1
    elif n_anchor==0:        out = ggml_repeat_4d(col1 -> [dim,L_stream,1])
    elif n_anchor==L_stream: out = ggml_repeat_4d(col0 -> [dim,L_stream,1])
    else:
        c0 = ggml_view_3d(c, dim,1,1, c->nb[1],c->nb[2], 0)         // anchor col view
        c1 = ggml_view_3d(c, dim,1,1, c->nb[1],c->nb[2], c->nb[1])  // rest col view
        a  = ggml_repeat_4d(ctx, c0, dim, n_anchor, 1, 1)          // [dim,n_anchor,1]
        r  = ggml_repeat_4d(ctx, c1, dim, L_stream-n_anchor, 1, 1) // [dim,L_stream-n_anchor,1]
        out = ggml_concat(ctx, a, r, 1)                            // [dim,L_stream,1]
```
repeat/concat are pure copies (no arithmetic) → bit-preserving. The `+modparam` is per-(dim,k)
column-independent, so width-2 add == width-L add for every token mapping to that column → identical
bits. Proof in full: the te0…tp1 embedding pipeline is column-wise pure & deterministic, so t∈{0,tval}
yields exactly 2 distinct columns E[:,:,0], E[:,:,1]; running it on t2 reproduces them bit-for-bit.

### C. Edge cases (CRITICAL)
1. **Text/T2V (Lt==1, per_token==false, build_graph:833):** gate on `e->ne[2]>1`; keep old chunk6
   body verbatim → literal no-op. Verify by diffing NAVA_DIT_VRAM_HIST node list.
2. **Audio anchor `n_anchor_a` is INDEPENDENT of video `n_clean_i`** and **can be 0** (→ pure col1
   broadcast). i2v can have n_clean_i>0 with n_anchor_a==0 and vice versa — handle both single-anchor
   branches. video chunk6 uses (n_clean_i, L_vid); audio chunk6 uses (n_anchor_a, L_aud).
3. **SingleBlock `e_single` `[dim,6,L_total]` = TWO anchor runs** (video prefix THEN audio prefix in
   the concatenated sequence): expand to `concat(repeat(c0,n_clean_i), repeat(c1,L_vid-n_clean_i),
   repeat(c0,n_anchor_a), repeat(c1,L_aud-n_anchor_a), dim=1)`. **Highest-risk piece — validate with
   a single-block hidden-state capture vs baseline.**
4. **NavaHead (537-559, 2-chunk):** same structure, Lt=L_stream; keep Lt==1 fast path; video head uses
   n_clean_i, audio head n_anchor_a.
5. Truly only 2 values (main.cpp:1080-1082 assigns 0 or tval, nothing else). Add GGML_ASSERT n_anchor<L_stream.
6. **Recover n_clean_i/n_anchor_a in build_graph host-side from `t`** (they currently live only in
   main.cpp): scan the timestep tensor — `n_clean_i = leading zeros in t[0:L_vid]`, `n_anchor_a =
   leading zeros in t[L_vid:L_total]`, `tval = first nonzero`. Build t2={0,tval}, make_input(t2).
   L_vid/audio_len already computed in build_graph (814-816) before timestep is consumed. No API change.

### D. Effort/risk
~4 functions in src/nava.hpp (time_embed call site, chunk6, SingleBlock inline chunk, NavaHead),
+ build_graph plumbing (~4 lines, the 3 block loops at 854/863/877 are driven once each). Medium.
Highest risk = SingleBlock 4-segment. Keep modulation params F32 (init_params:291) so add stays F32.

## STATUS at checkpoint
- q5_K LOCKED as prod quant (−724 MiB vs q6_K, faster, ear-clean). q4_K@25 ≈ q5_K@25 (cos .968) —
  open ear-test for −1400 MiB (clips q4k_25 vs cache25_ref on :8097).
- Step-cache (NAVA_CACHE_THRESH, default off) committed as opt-in: −43% @25 but trades audio
  refinement (narrow value for audio-sensitive use). mmq occupancy + align-off both measured-dead.
- This modulation rewrite is the LAST identified lever that moves BOTH VRAM and speed at zero quality
  cost. After it: the FFN `[18432,5148]` is the buffer floor (tile-only, low odds); video-VAE
  direct-conv3d is the biggest remaining bit-exact SPEED lever (~−10-15s/clip, deep kernel work).
- Owner wants a **1280×704** stress render after the VRAM win (higher res → more tokens → bigger
  everything; the modulation tensors scale with token count, so this fix helps that case MORE).
