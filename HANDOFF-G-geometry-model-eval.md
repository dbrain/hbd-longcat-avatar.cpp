# HANDOFF-G — geometry-model eval: pick the TRELLIS.2 replacement (2026 open image→3D)

**Why:** pixal3d / TRELLIS.2 is a **sparse-voxel** generator (occupancy grid → marching cubes). Its surface
is inherently soft/staircased and voxel-resolution-limited on thin features — **fingers web together**,
silhouettes are lumpy, and no retopo/smoothing/normal-map fixes it (the softness is upstream; see
HANDOFF-E/F + the IM-adaptive saga). We want a generator whose **surface is fundamentally cleaner** for
**riggable character meshes** (separable fingers, smooth watertight/manifold). The rigger we're porting
(**SkinTokens**, R1+R2+R5 done — see HANDOFF-F) is **geometry-agnostic**, so any clean mesh drops in.
**Baseline to beat: Hunyuan3D-2.1** (VecSet→SDF→marching cubes).

**Filters used:** weights must be publicly downloadable. License + VRAM NOT filtered (owner: "find a way").
Mesh-first; splat only matters if splat→mesh is mature (it isn't — see §4).

> Method note: this was salvaged from a killed deep-research workflow + verified by one normal subagent
> (~56k tokens). Every nontrivial claim has a source link. Re-verify "coming soon" repos before relying.

---

## 1. SHORTLIST — evaluate in this order

| # | Name | What it is | Arch family | Output | Release | Verified weights | finger/char | rig-ready |
|---|------|-----------|-------------|--------|---------|------------------|:---:|:---:|
| 1 | **StdGEN** ⭐ | single image → **A-pose, semantically part-decomposed** character | S-LRM + multiview | mesh (parts separable, A-pose) | CVPR 2025 | [hyz317/StdGEN](https://huggingface.co/hyz317/StdGEN) · [GH](https://github.com/hyz317/StdGEN) | 5 | 5 |
| 2 | **Step1X-3D** | two-stage native-3D: VecSet VAE-DiT → watertight TSDF (+texture) | **VecSet-latent VAE+DiT → TSDF→mesh** | watertight mesh | May 2025 | [stepfun-ai/Step1X-3D](https://huggingface.co/stepfun-ai/Step1X-3D) (5.27GB geom) · [GH](https://github.com/stepfun-ai/Step1X-3D) | 4 | 4 |
| 3 | **TripoSG** | large rectified-flow VecSet, image→SDF→mesh | **flow-matching VecSet-latent** (1.5B MoE DiT, DINOv2) | mesh (VecSet→SDF) | ~Mar 2025 | [VAST-AI/TripoSG](https://huggingface.co/VAST-AI/TripoSG) (5.76GB) · [GH](https://github.com/VAST-AI-Research/TripoSG) | 4 | 4 |
| 4 | **PartPacker** | single image → arbitrary **separable parts** (dual-volume packing) | VAE + flow (part-level) | mesh (GLB, parts) | Jun 2025 (NVlabs) | [NVlabs/PartPacker](https://github.com/NVlabs/PartPacker) (`vae.pt`+`flow.pt` on HF) | 4 | 4 |
| 5 | **Hunyuan3D-2.1** *(baseline)* | flow-matching DiT over ShapeVAE VecSet → SDF → MC | VecSet flow → **SDF→MC** | mesh (+PBR) | Jun 2025 | [tencent/Hunyuan3D-2.1](https://huggingface.co/tencent/Hunyuan3D-2.1) | 3 | 3 |
| 6 | **Hi3DGen** | image→normal→geometry (normal bridging) | TRELLIS/SLAT **sparse-voxel** + flow | geometry-only mesh | ~Apr 2025 | [Stable-X/trellis-normal-v0-1](https://huggingface.co/Stable-X) · [GH Stable3DGen](https://github.com/Stable-X/Stable3DGen) | 3 | 3 |
| 7 | **Direct3D-S2** | gigascale sparse-volume SDF diffusion, 1024³, Spatial Sparse Attn | **sparse-volume + SDF** DiT | mesh (SDF→obj) | May 2025 | [wushuang98/Direct3D-S2](https://huggingface.co/wushuang98/Direct3D-S2) · [GH](https://github.com/DreamTechAI/Direct3D-S2) | 3 | 3 |

Scores 1–5 (subagent judgment from docs, NOT from running them — that's the eval's job).
VRAM where known: Direct3D-S2 ~10GB @512-res / ~24GB @1024-res; TripoSG ~8GB-class (community, uncited);
others not surfaced. None VRAM-blocked given "we'll make it work."

---

## 2. CONFIRMED-with-weights vs GHOSTS (don't chase ghosts)

**Confirmed downloadable now:** StdGEN, Step1X-3D, TripoSG, PartPacker, Hunyuan3D-2.0/2.1, Hi3DGen
(Stable-X), Direct3D-S2, UltraShape-1.0 ([infinith/UltraShape](https://huggingface.co/infinith/UltraShape),
7.37GB — but it's a **sparse-voxel refiner stacked on Hunyuan3D-2.1**, same family we're leaving → low
priority), and the current baseline TRELLIS.2-4B ([microsoft/TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B),
Dec 16 2025).

**Ghosts / no open weights / out-of-scope:**
- **MoCA** — real repo ([lizhiqi49/MoCA](https://github.com/lizhiqi49/MoCA)) but placeholder, "coming soon," no weights.
- **FaithC** — real (CVPR'26 oral, [Luo-Yihao/FaithC](https://github.com/Luo-Yihao/FaithC)) but it's a **mesh representation/VAE, not a generator**; no weights.
- **Sparc3D** — closed/monetized; HF "space" is a commercial iframe, no checkpoint.
- **Hunyuan3D-2.5** (~10B, arXiv 2506.16504) and **3.0** — **API/Studio only, NOT open weights**. So 2.1 is the open ceiling in that family. (Don't confuse with the unrelated open *HunyuanImage-3.0* 2D model.)
- **PartCrafter** — paper only ([arXiv 2506.05573](https://arxiv.org/abs/2506.05573)), weights "coming soon" — re-poll.
- **"Lattice"** (Reddit) — did not resolve to a model; almost certainly the *structured-latent (SLAT)* concept from TRELLIS, not a separate gen.
- **Riggers (OUT OF SCOPE — we have SkinTokens):** "AniRig" = **Anymate** ([yfde/Anymate](https://github.com/yfde/Anymate)); also RigAnything, **UniRig** ([VAST-AI/UniRig](https://huggingface.co/VAST-AI/UniRig)), RigNet — all mesh→skeleton+skin, none generate geometry.

**AniGen** (the owner's actual interest — arXiv ~2604.08746, VAST, SIGGRAPH 2026): a geometry+rig **monolith**
(S³ Fields). Per the rigging handoff it **does NOT compose** — geometry and rig are fused and the rig is
locked to its own (older, TRELLIS.1-era) geometry. For us (we have our own geometry-agnostic rigger) its
only value would be its *geometry*, which is hard to extract from the monolith and is TRELLIS.1-era (likely
NOT cleaner than TRELLIS.2). **Action for the eval: verify whether open weights exist; if so, eyeball its
geometry quality only — but it is a low-priority long-shot vs the §1 shortlist.**

---

## 3. Recommendation — try FIRST

**StdGEN first, then Step1X-3D (or TripoSG) as the surface engine.** StdGEN is the only confirmed-open model
purpose-built for the exact target — single image → **A-pose, part-decomposed character mesh** (trained on
10k+ VRoid anime/human). Part-decomposition is the structural attack on finger-webbing (separable hands)
AND it lands the mesh already A-posed = what a rigger wants. In parallel test **Step1X-3D** (watertight TSDF
+ sharp-edge sampling + detail LoRA) or **TripoSG** (pure flow-matching VecSet→SDF) as the clean-continuous-
surface engine — the most direct test of the "VecSet/SDF beats sparse-voxel on thin features" hypothesis.
Likely winner = the combo (VecSet surface quality + part separability).

**Honest caveat:** TRELLIS.2-4B *itself* claims to handle sharp/non-manifold features better than SDF
iso-surface methods ([source](https://huggingface.co/microsoft/TRELLIS.2-4B)) — so the voxel-vs-VecSet gap
may be narrower than assumed. **The A/B on a finger-heavy character is the real test, not the arch label.**

---

## 4. Splat→mesh — NO (skip for this use case)
"TripSplat" = **TripoSplat** ([VAST-AI-Research/TripoSplat](https://github.com/VAST-AI-Research/TripoSplat),
real, MIT) but outputs **only Gaussians, no mesh**. Splat→mesh extractors (SuGaR/2DGS/GOF/RaDe-GS/MILo) are
multi-view scene tools that extract surfaces via TSDF/MC/Poisson and **erode/fuse thin structures**
([MILo, arXiv 2506.24096](https://arxiv.org/pdf/2506.24096)) — *worse* on our exact problem, plus a lossy
stage. Direct SDF/VecSet mesh gen is clearly more mature for riggable characters.

---

## 5. How this plugs into OUR setup (eval-light → port the winner)
- **Eval phase = minimal Python inference, NOT a full stack.** Each shortlist model ships an HF checkpoint +
  an image→mesh inference script (several also have ComfyUI nodes). Goal of the eval: clone repo → fetch HF
  weights → run their one inference command on a **finger-heavy test character** → dump GLB → eyeball vs the
  TRELLIS.2 baseline on `compare.html`/`imadapt_web`. Reuse the per-model-venv pattern from
  `/mnt/hdd/3d/avatar-shootout/` rather than building infra. Pick the winner on **finger/silhouette quality**.
- **Then port the winner to C++/ggml** (the `tools/m1_ref/cpp_port/` pattern — M1Harness, GGUF weights,
  validate-vs-fp32-oracle). The shortlist is dominated by the **VecSet-latent / flow-matching / SDF** family,
  which is the **same family as the R1 VecSet encoder already ported & validated this session** (HANDOFF-F) —
  so that work is a down-payment regardless of which we pick.
- **Rigger is unaffected** — SkinTokens consumes whatever clean mesh the new geometry slot emits.

## 6. Build/eval reminders
- GPU may be shared — coordinate / ask before heavy generation runs. C++/CUDA build fine on-server; NO Rust builds.
- Validate ported stages vs the TRUE-fp32 oracle (`.float()` + eager attn, `NVIDIA_TF32_OVERRIDE=0`).
- The IM-adaptive retopo viewer (clay A/B/C + textured V6 hero) is at
  `http://10.0.0.208:8011/imadapt_web/im_adapt.html` for baseline calibration.
