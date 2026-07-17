# Image-Conditioned 3D Mesh Texturing — Model Survey (replace TRELLIS-2 texturing)

Date: 2026-06-18. Goal: **(existing mesh + reference character image) → clean PBR texture (base color at minimum, ideally metallic/roughness)** on a single **RTX 3060 (12 GB)**. Geometry already exists (cleaned/UltraShape'd mesh). We need a texture-only path, NOT geometry+texture coupled. End goal is a ggml/C++ port (as with TRELLIS-2), so port effort/risk matters.

## The key filter

Most published "texture" methods are **text→texture** or are **coupled to their own generated geometry**. Two filters applied:
1. Takes an **arbitrary input mesh + a reference IMAGE** (image-conditioned, not text-only).
2. Fits (or can be made to fit) **12 GB** VRAM.

Result: the real contenders are **Hunyuan3D-Paint (2.0 / 2.1)**, **MV-Adapter (Image2Texture)**, **FlexPainter**, and **TRELLIS.2's own texturing pipeline** (the incumbent). Most academic methods (TEXGen, Paint3D, SyncMVD, FlashTex, Text2Tex, TEXTure, Paint-it, EASI-Tex, TriTex, Step1X-3D) are either text-only, geometry-coupled, 40 GB-class, or have no usable image-conditioned path.

---

## Candidate detail

### 1. Hunyuan3D-Paint 2.0 (hy3dpaint, the "paint" half of Hunyuan3D-2.0)  — STRONG FIT
- Repo: https://github.com/Tencent-Hunyuan/Hunyuan3D-2 — `Hunyuan3DPaintPipeline`
- Model: https://huggingface.co/tencent/Hunyuan3D-2 — paint model ~1.3B (SD-based multiview UNet + ControlNet-style geometry conditioning + IP-Adapter image conditioning + delight)
- Paper: Hunyuan3D 2.0, arXiv:2501.12202 (Jan 2025)
- License: **Tencent Hunyuan Community License** — usable but encumbered: commercial use allowed *if you don't redistribute the weights and don't claim ownership*; **explicitly excludes EU / UK / South Korea**; >100M MAU needs a separate license. Not OSI-open. (https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/main/LICENSE)
- **Image-conditions an existing mesh?** YES. `paint_pipeline(mesh, image=...)` takes a hand-crafted/given mesh + a reference image; this is exactly our interface. Decoupled from shape generation.
- PBR: 2.0 paint produces **Albedo + (via the PBR variant) Normal / Roughness / Metallic**; the headline "delight" removes baked lighting → illumination-invariant albedo. (2.0 base is strong on albedo; full metallic/roughness is the 2.1 story below.)
- **VRAM on 3060: FITS.** Full shape+texture is ~12 GB; texture-only is the lighter half. The community **Hunyuan3D-2GP** fork (https://github.com/deepbeepmeep/Hunyuan3D-2GP, mmgp offload) runs texture in **6 GB** with 32–48 GB system RAM (profiles 4–5). 12 GB is comfortable without aggressive offload.
- Quality: SOTA-class among open texturing as of 2025; delight + multiview + geometry-aware attention give good view-consistency and few seams. Strong on characters (it is the de-facto open baseline everyone benchmarks against).
- Architecture: **multiview diffusion (SD UNet) conditioned on rendered normal+position maps of the mesh + reference-image (double-stream reference-net / IP-Adapter) + delight module → multi-view RGB → back-project & bake to UV.**
- ggml-port effort: **Medium.** SD-class UNet + VAE + CLIP/IP-Adapter image encoder — all op families we already have in our cpp stack (longcat/flux2/sd.cpp). The non-trivial bits are the **geometry conditioning render (normal/position maps), the delight model, multiview attention coupling, and the back-projection/UV-bake** (rasterization + visibility weighting — CPU/GL, not ggml). Comparable surface to a flux2/longcat port plus a bake stage.

### 2. Hunyuan3D-Paint 2.1 (hy3dpaint in Hunyuan3D-2.1) — BEST QUALITY, but VRAM-heavy
- Repo: https://github.com/tencent-hunyuan/hunyuan3d-2.1 — Paint-v2-1 (**2B**)
- Paper: arXiv:2506.15442 (Jun 2025), "Production-Ready PBR Material." Built on **RomanTex** (3D-aware RoPE multi-attention) + **MaterialMVP** (illumination-invariant multi-view PBR diffusion, https://github.com/ZebinHe/MaterialMVP).
- License: same Tencent community license (EU/UK/KR excluded).
- Image-conditions an existing mesh? YES, same interface as 2.0.
- PBR: **full Disney Principled BRDF — Albedo + Metallic + Roughness + Normal**, dual-branch UNet, illumination-invariant. This is the *most production-ready PBR* of the open field.
- **VRAM: ~21 GB** for max_num_view=6 @ 512 — **does NOT fit a 3060 out of the box.** Would need view-count reduction, tiling, FP8/Q-quant, and offload to fit 12 GB; non-trivial. (Our cpp port could quantize it like we do longcat, but it's a heavier lift than 2.0.)
- ggml-port effort: **Medium-High** — same shape as 2.0 but 2B, dual-branch PBR UNet, plus RomanTex 3D-RoPE (we already ported fused RoPE variants for LTX, so feasible) and the multi-channel material VAE.

### 3. MV-Adapter — Image2Texture — LIGHTEST FIT, base-color-leaning
- Repo: https://github.com/huanngzh/MV-Adapter (ICCV 2025). HF: https://huggingface.co/huanngzh/mv-adapter, demo https://huggingface.co/spaces/VAST-AI/MV-Adapter-Img2Texture
- Paper: arXiv:2412.03632 (Dec 2024). Pipeline released 2025-03-31 (img2tex) / 2025-05-15 (full).
- License: **Apache-2.0** — cleanest license of the whole field (no territory carve-outs, fully commercial). Big advantage.
- Image-conditions an existing mesh? YES — `--image ... --mesh ...`; renders 6-view normals of the given mesh, IP-Adapter-style image condition drives a fine-tuned T2I model, back-projects + incidence-weighted blend to UV.
- PBR: **base color / shaded GLB primarily** — it is a *multiview adapter*, not a material model; native metallic/roughness is not its headline output (the 3D demos emit a shaded/base-color GLB). PBR would need bolting on (e.g. a MaterialMVP-style second pass). This is the main quality gap vs Hunyuan.
- **VRAM: the SD2.1 variant runs in `<10 GB` (`--variant sd21`)**; SDXL variant needs `>16 GB`. **SD2.1 fits the 3060 with headroom** — the most comfortable fit here.
- Quality: very good multiview consistency (that's the whole paper); base-color quality strong. As a *plug-in adapter* it's smaller/simpler than Hunyuan's full stack.
- Architecture: **plug-and-play attention adapter on a frozen SD2.1/SDXL T2I → geometry(normal)-conditioned + image-conditioned 6-view RGB → backproject/blend to UV.** No delight/material branch.
- ggml-port effort: **Low-Medium** — smallest model, SD2.1 base is exactly sd.cpp territory; the adapter is extra attention layers + a conditioning path. Easiest port; but you inherit the "base color only" limitation and would have to add PBR yourself.

### 4. TRELLIS.2 texturing (the INCUMBENT we're replacing)
- Repo: https://github.com/microsoft/TRELLIS.2 — `Trellis2TexturingPipeline` (`app_texturing.py`). Model: https://huggingface.co/microsoft/TRELLIS.2-4B
- Image-conditions an existing mesh? YES — mesh + image → **PBR (BaseColor + Roughness + Metallic + Opacity)** on the input mesh's own voxels. This is precisely what our pipeline already uses.
- License: MIT (Microsoft) — clean.
- Architecture: **structured-latent / sparse-voxel** PBR generation (not multiview-diffusion-bake). Different paradigm: textures via the shape encoder + sparse latents directly, which is why it avoids reproject leaks when run natively on the target mesh's voxels.
- Why look elsewhere: it's a **4B** model and the reproject/UV path has been the pain point in our notes; we're evaluating whether a multiview-bake approach is cheaper to port / fits 3060 better / gives cleaner faces.

### 5. FlexPainter (newer, 2025) — flexible image+text, watch-list
- Repo: https://github.com/StarRealMan/FlexPainter — Paper arXiv:2506.02620 (Jun 2025).
- Image-conditions an existing mesh? YES — mesh path + image prompt; **image-based CFG** decomposes structure vs style for reference-image stylization. Multi-view grid + reprojection sync + completion/enhancement.
- PBR: multi-view RGB → UV; PBR not the headline (closer to MV-Adapter than Hunyuan-2.1). License/VRAM not clearly documented yet (SDXL-grade backbone → likely ~12–16 GB).
- Status: promising for *reference-image-driven* stylization (our exact "character image → mesh" use), but newer/less battle-tested; verify license + VRAM before committing. ggml effort similar to MV-Adapter.

### Ruled out (and why)
- **TEXGen** (https://github.com/CVMI-Lab/TEXGen, SIGGRAPH Asia 2024, 700M UV-domain diffusion): genuinely image-conditioned UV diffusion and architecturally elegant for a port — **but training/inference needs >40 GB (A100)** as published, and it's albedo-only. VRAM kills it for 3060 without heavy surgery.
- **Paint3D** (https://github.com/OpenTexture/Paint3D, CVPR 2024): coarse-to-fine 2K UV, lighting-less; image OR text condition exists, but quality/consistency is behind Hunyuan/MV-Adapter and it's an older SD1.5-era stack.
- **SyncMVD** (https://github.com/LIU-Yuxin/SyncMVD, MIT): **text-guided only**; no image-conditioning interface.
- **Step1X-3D** (https://github.com/stepfun-ai/Step1X-3D, Apache-2.0): texture stage is **coupled to its geometry stage** and needs **24 GB**; not a drop-in texture-only path for our mesh.
- **FlashTex** (Roblox, https://github.com/Roblox/FlashTex), **Paint-it**, **TEXTure**, **Text2Tex**, **Fantasia3D**: **text-prompt** texturing (SDS / backprojection). FlashTex does emit PBR (kd/roughness/metallic/normal) and relightability, but it's text-driven, not image-conditioned.
- **EASI-Tex** (https://github.com/sairajk/easi-tex, SIGGRAPH 2024): *is* single-image mesh texturing, but it's a per-mesh **SDS/ControlNet optimization** (slow, no PBR, non-feed-forward) — wrong shape for a fast production stage.
- **TriTex** (NVIDIA, CVPR 2025): learns texture from a single *textured mesh* exemplar (texture transfer), **code not released**, not an image→texture interface.
- **Meta 3D TextureGen / AssetGen**: **no open weights released**; research-only. Out.
- **TexFusion**: NVIDIA, no public weights. Out.

---

## Ranked short-list (top 3)

1. **Hunyuan3D-Paint 2.0 (hy3dpaint, via Hunyuan3D-2.0)** — best quality-per-VRAM that actually fits a 3060 (~12 GB, 6 GB with the 2GP offload fork), true image→existing-mesh interface, delighted albedo + a PBR path, the de-facto open baseline. Main catch: encumbered Tencent license (no EU/UK/KR). *Prototype this first.*
2. **MV-Adapter Image2Texture (SD2.1 variant)** — the safest *fit and license*: Apache-2.0, `<10 GB`, smallest/easiest port (sd.cpp + adapter), excellent view-consistency. Catch: base-color-leaning, you'd have to add a PBR pass for metallic/roughness.
3. **Hunyuan3D-Paint 2.1** — the **best PBR quality** of the open field (full Disney BRDF, illumination-invariant), but **~21 GB** out of the box → needs quantization/offload/view-reduction to fit 12 GB. Worth it only if 2.0's PBR proves insufficient and we're willing to spend the VRAM-shrink effort (which our cpp/quant tooling is good at).

## Final recommendation — prototype first

**Prototype Hunyuan3D-Paint 2.0 (hy3dpaint) first.** It is the only candidate that simultaneously (a) takes our exact interface — *given mesh + reference character image → textured mesh*, decoupled from geometry; (b) fits the 3060 today (~12 GB stock, 6 GB with the 2GP mmgp fork) so we can validate end-to-end before any porting; (c) gives delighted albedo plus a PBR path and is the strongest open character-texturing quality at this VRAM; and (d) is a familiar ggml port — SD-class multiview UNet + VAE + image encoder (longcat/flux2/sd.cpp op families), the only genuinely new work being the geometry-render conditioning, delight, and the back-projection/UV-bake stage (CPU/GL, outside ggml).

Run the **MV-Adapter SD2.1 (Apache-2.0, `<10 GB`)** path in parallel as the *license-clean, lowest-port-risk* fallback and as the easiest first ggml port to stand up — accept that PBR (metallic/roughness) would be a follow-on pass on top of its base-color output. Escalate to **Hunyuan3D-Paint 2.1** only if/when 2.0's PBR quality is the bottleneck, using our quant/offload tooling to bring the ~21 GB model under 12 GB.

If the Tencent license territory exclusion (EU/UK/KR) is a hard blocker for distribution, **invert the order**: lead with **MV-Adapter (Apache-2.0)** and add a MaterialMVP-style PBR pass, keeping Hunyuan as the quality reference.
