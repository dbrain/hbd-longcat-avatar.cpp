# Wan2.2-VACE-Fun-A14B — wiring sketch (for test runs later)

The velocity-preserving continuation path. Status: **DiT support already exists** (`wan.hpp`),
**control-tensor builder written** (`src/vace.hpp`, compiles). Not yet wired into a runnable
example, and the **model is not downloaded**. This is the "scribble" of how it sticks together.

## What's already in place
- **DiT branch (`src/model/diffusion/wan.hpp`)** — `VaceWanAttentionBlock` (before_proj/after_proj),
  `vace_patch_embedding` (Conv3d `vace_in_dim=96`→dim), `vace_context`/`vace_strength` threaded
  through `forward_orig` (a vace_block injects into the main stream every `num_layers/vace_layers`
  layers). Auto-detected: `WanConfig::detect_from_weights` sets `vace_layers` from `vace_blocks.*`
  in the GGUF; `WanRunner::compute(..., vace_context, vace_strength)` already takes it. **Zero DiT
  code to write.**
- **Control builder (`src/vace.hpp`)** — `VACE::build_vace_context(vae, control_px, mask, ...)` →
  `[W_lat,H_lat,T_lat,96]` = inactive_lat(16) ++ reactive_lat(16) ++ space-to-depth mask(64), plus
  `VACE::make_continuation_source()` (context frames + gray placeholder + mask). Mirrors
  `wan/vace.py` (vace_encode_frames + vace_encode_masks + vace_latent).

## Model (downloading → `models/dl/wan22-vace-fun-a14b/`)
`alibaba-pai/Wan2.2-VACE-Fun-A14B` — grabbing only `high_noise_model/` + `low_noise_model/` (each a
SINGLE `diffusion_pytorch_model.safetensors`, ~28GB; skipped the repo's Wan2.1_VAE.pth + umt5, have them).
**Config CONFIRMED** (`high_noise_model/config.json`): `_class_name=VaceWanModel`, `model_type=vace`,
`dim=5120 num_layers=40 num_heads=40 ffn_dim=13824`, **`in_dim=16`** (T2V base — x is plain 16ch noisy,
NO i2v c_concat), **`vace_in_dim=96`**, **`vace_layers=[0,5,10,15,20,25,30,35]`** (8 blocks every 5 —
matches `wan.hpp` `step=num_layers/vace_layers`). So our DiT auto-detects it correctly.
- **CONVERT-TIME CAVEAT:** it's diffusers-format (`_diffusers_version 0.30.0`). Check the safetensors
  tensor names — if they're true-diffusers (`blocks.N.attn1.to_q`) not flat-Wan (`blocks.N.self_attn.q`),
  `convert_wan_dit.py` (verbatim copy) won't match the C++ loader; needs name-mapping (the A14B distill
  experts were already flat-Wan, so this MAY be too — verify before converting).

## What's required to run a test
1. **Download** — in progress (see above). Reuse our umT5 + Wan2.1 VAE.
2. **Convert** each expert with `tools/convert_wan_dit.py` (already patched):
   - `vace_blocks.*` pass through verbatim (no filter).
   - **Both** `patch_embedding.weight` AND `vace_patch_embedding.weight` get the out*in merge —
     the fix's `name.endswith("patch_embedding.weight")` matches both (verify the assert: vace conv
     is `[dim,96,1,2,2]`, base is `[dim,16,1,2,2]` for the T2V base; in_dim=16 not 36 for VACE).
   - Then `sd-cli -M convert --type q4_K` per expert (like the A14B I2V experts).
   - **Distill:** VACE-Fun base is full-step. If no distilled checkpoint exists, extend
     `convert_wan_dit.py` to fold the lightx2v A14B step/cfg LoRA (copy `load_lora()` from
     `convert_infinitetalk_dit.py`; the vace_blocks are untouched, only base blocks get deltas).
     **UNVERIFIED that the A14B distill folds cleanly onto VACE-Fun — confirm before trusting 4-step.**
3. **Build the control video per window** (continuation):
   - `ctx_frames` = last N pixel frames of the previous generated clip (N≈ a few; the *given* frames).
   - `VACE::make_continuation_source(ctx_frames, frame_num=81, gray=0.5, control_px, mask)`.
   - `vace_context = VACE::build_vace_context(vae, n, control_px, mask, tiling, 4, 8, WAN_VAE_MEAN, WAN_VAE_STD)`.
4. **Sample** — VACE is a T2V base, so `x` is a plain 16ch noisy latent (NO i2v c_concat), umT5
   context, and the 96ch `vace_context` carries all the conditioning. MoE high/low switch as for A14B.

## Inference loop (how it sticks together)
```cpp
// load two WanRunners (the GGUF auto-detects vace_layers -> desc "Wan2.x-VACE-14B", in_dim=16):
//   low  = WanRunner(... "model.diffusion_model.")          // low-noise expert
//   high = WanRunner(... "model.high_noise_diffusion_model.")// high-noise expert
auto context = umt5_encode(prompt);                          // [4096,512,1]
auto sigmas  = distilled_sigmas(4, /*shift=*/5);             // 4-step LCM (if distill folds)
sd::Tensor<float> x(/*[W_lat,H_lat,T_lat,16]*/); randn(x);

for (int i = 0; i < steps; ++i) {
    float s = sigmas[i], s_next = sigmas[i+1];
    auto ts = Tensor::from_vector({ s * 1000.f });
    auto* rn = (s >= moe_boundary /*0.875*/) ? high.get() : low.get();   // MoE expert switch
    auto v  = rn->compute(n_threads, x, ts, context,
                          /*clip_fea=*/{}, /*c_concat=*/{}, /*time_dim_concat=*/{},
                          vace_context, /*vace_strength=*/1.0f);
    for (k) x[k] += v[k] * (s_next - s);                     // euler (same convention as s2v/IT)
}
// VACE has no motion-frame pin — the *given* context frames live in vace_context (inactive+mask),
// so continuity is the model's job, not a host-side latent overwrite.
auto rgb = vae_decode_video(x);                              // [W,H,T,3]
// drop the first N pixel frames on windows>0 (the regenerated context), keep the rest.
```

Then the outer window loop is the same shape as the InfiniteTalk streamer: slide by
`frame_num - N`, feed the new tail as `ctx_frames`. **The win over our InfiniteTalk overlap:** VACE
was *trained* to honor multiple given frames, so velocity carries across the seam instead of resetting.

## Two ways to drive it
- **(a) sd-cli** — if it already exposes a VACE control-video/mask input + the `--diffusion-model`
  / `--high-noise-diffusion-model` / `--moe-boundary` flags, point them at the converted experts and
  a prepared control clip. Check `examples/cli` for VACE flags first.
- **(b) custom `examples/vace/main.cpp`** — clone the InfiniteTalk example shell (load/VAE/umT5/decode),
  swap the DiT for two `WAN::WanRunner`s + the MoE switch above, and feed `VACE::build_vace_context`.
  Cleaner for a controlled first test; ~the s2v/infinitetalk example size.

## Open questions to settle on the first render (oracle vs wan/vace.py)
1. inactive/reactive sign + whether the to-generate region is gray(0.5) or black(0).
2. VAE latent space: raw vs (mu-mean)/std for the vace latents (we apply the normalization).
3. mask temporal resample (nearest vs the exact `(depth+3)//4` + nearest-exact interpolate).
4. ref-image prepend (reference-guided VACE prepends a ref frame along T; continuation doesn't need it).
5. distill: does lightx2v-A14B fold onto VACE-Fun and stay coherent at 4 steps?
