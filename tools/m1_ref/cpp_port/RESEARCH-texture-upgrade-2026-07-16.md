# Texture-upgrade research — image→3D PBR texturing SOTA (2026-07-16)

Deep-research (verified: 22 confirmed / 3 refuted claims, 23 sources). GOAL: replace pixal3d's soft
volumetric (SLAT-baked) texture with a crisp, hands-off PBR texture stage, keeping TRELLIS-2 geometry.
Owner: **license does NOT matter — "whatever the best is"; want a dedicated texture stage good enough to
never touch up.** Hard requirement: output a **baked UV texture** (animation-safe under the rig), not a
screen-space projection.

## Options (all keep TRELLIS-2 geometry)

| Option | PBR? | Core arch → ggml portability | License | VRAM | Date |
|---|---|---|---|---|---|
| **TRELLIS.2 native texturing** (`app_texturing.py`, Trellis2TexturingPipeline) — SEPARATE from the soft SLAT bake; basecolor/rough/metal/opacity; textures external meshes | ✅ | flow-transformer (SAME family we already ported for the trellis2 geometry DiT) — portability UNCONFIRMED | MIT | unstated | Dec 2025 |
| **Hunyuan3D-Paint 2.1** (of Hunyuan3D 2.1; "powered by RomanTex + MaterialMVP") — mesh-conditioned 6-view 512-res diffusion + back-projection | ✅ full Disney BRDF (albedo/rough/metal) | **SD-2.1 UNet** (CONFIRMED — the "2B DiT" claim was REFUTED) → most portable via our Flux/SD infra | Tencent Community (territorial excl. EU/UK/SK, ≤1M MAU commercial) — *owner doesn't care* | 21 GB tex (≈6 GB CPU-offloaded) | Jun 2025 |
| **MVPainter** (arXiv 2505.12635, amap-cvlab/MV-Painter) — **union-ControlNet on normal+depth** (directly targets softness), IDArb PBR extractor 256→512 | ✅ basecolor/metal/rough | SD-family + ControlNet | unspecified | unstated | May 2025 |
| Step1X-3D texture | ❌ albedo-only (PBR = stated future work) | SD-XL UNet (very portable) | — | — | May 2025 |
| Paint3D | ❌ albedo-only | SD1.5 UNet + ControlNet (very portable) | Apache-2.0 | — | CVPR 2024 |
| Hunyuan3D 2.5 | ✅ | novel MV arch ext. of 2.0 Paint | initially closed/API-gated | — | Jun 2025 |
| MVPaint | unstated | SMG + S3I + **UVR (UV super-res + seam-smooth)** | — | — | CVPR 2025 |

## Front-project + generate-the-back hybrid = a real published class
- **TEXGen** (NeurIPS 2024): projects the input view into UV → 700M diffusion fills ONLY the unseen parts.
  Exactly "real front + generate the back," and cheaper (only synthesize occluded regions). Uses our real
  sharp front image. Front-projection agent already PROVED the front aligns pixel-perfect on our camera.
- **Im2SurfTex** (CGF 2025): drop-in trained neural back-projection (texture-space cross-attention) that
  replaces ad-hoc backproject/averaging → fewer seams/softness. Layerable onto ANY multi-view generator.

## Verified caveats
- **NO quality benchmark survived** — the "Hunyuan3D-Paint beats rivals on CLIP-FID/LPIPS" claim was REFUTED
  (0-3). Pick on PBR + arch portability + ecosystem, then **A/B render on OUR geometry** to choose a winner.
- Hunyuan3D-Paint core = SD-2.1 UNet (good for us), NOT a 2B DiT (refuted). No subsurface scattering (refuted);
  it's albedo/rough/metal metallic-roughness workflow.
- TRELLIS.2 texture-generator arch/portability = the key OPEN QUESTION (likely a structured-latent flow
  transformer, i.e. the same family as our geometry port, NOT an SD UNet).

## Recommendation (license-agnostic, best-quality, hands-off)
1. **First determine what we're ACTUALLY doing** — is our texture the TRELLIS.2 SLAT appearance decode (soft),
   or its dedicated `app_texturing` PBR pipeline? If we're on the SLAT bake, switching to TRELLIS.2 native
   texturing is the same-ecosystem, same-arch-family upgrade (we already ported the trellis2 flow transformer).
2. **Highest-quality dedicated stage** = Hunyuan3D-Paint 2.1 (SD-2.1-UNet, portable via our Flux/SD infra,
   full PBR) or MVPainter (normal+depth ControlNet directly targets sharpness). Owner license-agnostic → both open.
3. Front-project (real image) + Flux-inpaint-the-back (TEXGen-style) = smallest new build, uses our sharp
   source; must bake to UV (animation-safe). Im2SurfTex to clean the back-projection.

Key sources: TRELLIS.2 github.com/microsoft/TRELLIS.2; Hunyuan3D-Paint arXiv 2506.15442 + Tencent-Hunyuan/
Hunyuan3D-2.1; MVPainter arXiv 2505.12635; TEXGen arXiv 2411.14740; Im2SurfTex arXiv 2502.14006; MaterialMVP
arXiv 2503.10289; Hunyuan3D 2.5 arXiv 2506.16504; Step1X-3D arXiv 2505.07747; Paint3D arXiv 2312.13913.
