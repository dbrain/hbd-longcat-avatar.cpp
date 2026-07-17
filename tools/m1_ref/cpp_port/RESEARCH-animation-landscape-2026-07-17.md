# Animation landscape for the image→rigged-3D pipeline — recon report
**Date:** 2026-07-17 · **Scope:** prompt→animation, face/"moving skin" emoting, fingers, and the naming/retargeting bridge.
**Method:** 6 parallel recon sweeps (on-disk + web) + adversarial verification. **No GPU was touched — nothing here is render-verified.**
**Tags:** `[VERIFIED]` = I (or a sweep) opened the file / hit the API / read the exact line. `[INFERRED]` = reasoned, not measured. `[UNVERIFIED]` = could not check.

---

## 1. TL;DR — the five things that actually matter

**1. Two "we already have X" claims on disk are FALSE, and one "fact" in the brief is BACKWARDS.**
- **EMAGE weights are gone.** `$AS/EMAGE/SETUP_REPORT.md` says they are cached at `~/.cache/huggingface/hub/models--H-Liu1997--emage_audio`. **That directory does not exist** — the hub cache holds only Wan/LTX/NAVA/gemma. Three independent sweeps confirmed. Only survivor: `SMPLX_NEUTRAL_2020.npz` (167MB). `[VERIFIED]`
- **The brief's finger premise is inverted.** Measured from the GLBs: **the soldier has 1 connected component and 3 real digit chains per hand with real skin weight**; his source PNG shows silhouette-separated digits with deep concave notches. **Gilly's GLB has 16 connected components — her "hands" are 4 detached blobs floating ~4cm off her arm stumps**, one solid paddle in cross-section, zero finger lobes; her source art draws fingers as *line-art strokes on a flat paddle* (same paint-not-geometry pattern as her face). SkinTokens hallucinated 19 finger bones/hand into stray geometry. `[VERIFIED — measured, but see §8: gilly's GLB is from make_gilly.sh (pixal3d@1024, NO UltraShape, dated Jun 1) and may simply be stale]`
- **HY-Motion's "26GB" is not a floor** — but it also isn't disproven. See #3.

**2. Nothing in text→motion does fingers. Nothing. Including HY-Motion.**
The paper says it verbatim: *"We employ the skeleton definition of SMPL-H (22 joints without hands)"*, 201-dim/frame. Its `joint_names.json` lists 52 SMPL-H joints incl. Thumb3 — **that is a trap**: `body_model.py` fills hands from `LEFT_HAND_MEAN_AA` **constants**. 52 joints would need input_dim 471, not 201. MoMask: same 22-joint wall. `[VERIFIED]`

**3. HY-Motion 1.0 is real, ungated, and its output format is the best possible news for us — with two live blockers.**
Tencent, released 2025-12-30, `tencent/HY-Motion-1.0`, `gated: False`, 1.0B MMDiT + flow matching. **It emits rot6d LOCAL rotations + root transl** (not MoMask's positions) → feeds our new delta retargeter with *no IK step*. The 26GB README figure decomposes as Qwen3-8B fp32 (16.4GB) + CLIP-L (1.7GB) + DiT (4.2GB) + 4 seeds — and a third party has **already published Qwen3-8B as GGUF Q4_K_M (5.03GB)** next to the ckpts. Sequential encode→free→DiT lands ≈5–6GB `[INFERRED — arithmetic from verified file sizes, NOT measured]`. **Blocker A:** the licence says *"THIS LICENSE AGREEMENT DOES NOT APPLY IN THE EUROPEAN UNION, UNITED KINGDOM AND SOUTH KOREA."* If you're in the UK/EU that is a lawyer question, not mine. **Blocker B:** nobody has verified whether Qwen3-8B is a one-shot freeable encoder or re-invoked during denoising — if the latter, the 12GB argument collapses.

**4. The dead-hands question has a worse answer than "fingers sit at bind pose" — the WRIST is dead too.**
`bonemap.py`'s `SMPL_CHILD` has no entry for joints 20/21 (the wrists), so `AIM[wrist]=None` → rest quat on every frame. Measured on `g_dance.json`: **18/61 bones move, 43 static**; every finger bone deviates **<0.5° over 140 frames**; `bone_9` (L elbow) rotates 71.4° while `bone_10` (L **hand**) is static. Identical in the new `gil_dance_deltafix.json` (elbow 124.2°, hand static). SMPL-22 ends at the wrist *position* — wrist **orientation is absent from the data**, not merely unmapped. **No SMPL-22 source can ever drive it.** This is a free, visible quality win independent of everything else in this doc. `[VERIFIED]`

**5. "MoMask was a bit jank" is probably partly our fault — and has never been re-judged.**
`_built.tsv` proves MoMask generated 8 of the 9 clips (it *is* the prompt list). Every clip you judged (`web/c1/clips/g_*.json`) went through the **stock look-at retargeter** — i.e. through the two bugs root-caused this week. `$AS/HANDOFF.md:153` itself splits the blame: *"MoMask motion is mid-tier"* AND *"look-at retarget has NO foot-planting/IK... NO head-aim"*. Only **3 of 8** motions were rebuilt as `_deltafix`. **Nobody has rendered them.** That is the cheapest experiment on this page (§7).

> **Naming caveat:** phonetically, "emotion or ezmotion" is a much better match for **EMAGE** than for MoMask — and EMAGE *is* on disk. But `_built.tsv` proves **MoMask made gilly's clips**. Best reading: you're remembering EMAGE's *name* and MoMask's *output*. Both are on disk; only MoMask's weights survive.

---

## 2a. Prompt → animation ("swat a fly away", "do a barrel roll")

### The honest answer to the fine-grained-prompt question
**No source I found demonstrates any model nailing fine-grained non-locomotion prompts.** `[VERIFIED — as an absence]`

HY-Motion's own limitations section concedes trouble with *"highly detailed or complex instructions"* and weak human-object interaction, and **explicitly rules out non-humanoid, in-place motion, seamless loops, and emotion/appearance**. Its strongest demonstrated capability is *laterality* ("left" vs "right hand").

Read that against your two examples:
- **"do a barrel roll"** — a whole-body locomotion-adjacent stunt, plausibly in AMASS/mocap corpora. **Plausible.** `[INFERRED]`
- **"swat a fly away"** — fine-grained, directed at an imaginary object, dominated by wrist and finger motion (which HY-Motion does not emit and our retargeter freezes anyway). **Coin flip at best.** `[INFERRED]`

The only quality evidence anywhere is Tencent's **SSAE**: HY-Motion **78.6%**, MoMask **58.0%**, GoToZero 52.7%, DART 42.7%. That is **their metric, their eval, their VLM judge, in their paper** — structurally the same circular validation that burned us with "cos 0.9998 proves texturing is faithful". **Do not weight it.** This needs a render, not more searching.

So: **they are largely walk/run/dance machines with a good laterality prior.** HY-Motion is genuinely better than MoMask at following text, but "prompt anything" is not what the field delivers today.

### The named candidates
| | |
|---|---|
| **HYMotion = HY-Motion 1.0** | Real. Tencent Hunyuan 3D Digital Human, 2025-12-30, arXiv 2512.23464. `tencent/HY-Motion-1.0`, ungated, 6.05GB (1.0B `latest.ckpt` 4.172GB; Lite 0.46B 1.844GB). MMDiT 27 layers / feat_dim 1280 / 20 heads / RoPE + flow matching, Euler 50 steps, 30fps, max 360 frames (12s). Dual text conditioning: Qwen3-8B token hidden states (4096-d) + CLIP-L pooled (768-d). `[VERIFIED via HF API + raw configs]` |
| **"emotion/ezmotion" = MoMask** (output) / **EMAGE** (probably the name) | MoMask: `EricGuo5513/momask-codes` @94a6636, **MIT** (read the LICENSE file — the cleanest licence in this survey), 194–201MB of checkpoints **on disk and complete**, `.venv` intact. RVQ-VAE(6×512×512) + 8-layer masked transformer + residual transformer + length estimator, CLIP ViT-B/32 text encoder. Emits `(T,22,3)` **positions** @20fps. `[VERIFIED]` |
| **`ZeyuLing/motius` + `hftrainer`** | ~44 ungated HF repos re-hosting nearly your whole named list (MDM, T2M-GPT, MoMask, MotionLCM, MoGenTS, FlowMDM, MotionGPT/3, MLD, DART, MotionCLR, MotionLab, GoToZero, MotionStreamer, PRISM) as **safetensors + Mean/Std** — far better porting substrate than the upstream gdown/.tar-pickle mess. **Caveat: third-party reimplementations, ~0 downloads, none verified to load or match their papers.** This is exactly our historical burn shape. `[VERIFIED files exist; UNVERIFIED that they work]` |

### The one architectural fact that decides a lot
**MoMask emits positions; HY-Motion emits rotations.** MoMask's position-only output is *precisely why* `retarget.py` became position-based look-at — i.e. **MoMask's representation is the upstream cause of the jank you root-caused**. HY-Motion drops straight into the rest-relative delta retargeter. That is a bigger practical difference than the SSAE gap.

Independent corroboration that the ecosystem has *not* solved this: **HY-Motion GitHub issue #42** is a user reporting SMPL-H→Mixamo retargeting failing on root/hips + rest-pose differences — **open since Feb 2026, zero maintainer response**. That is the bug we fixed this week. We start ahead. `[VERIFIED]`

### The free prompt-rewriter trick
HY-Motion ships `Text2MotionPrompter` (Qwen3-30B-A3B fine-tune, **apache-2.0**, 61GB) which turns a terse prompt into a standardised caption + a frame count. **Do not download it.** The rewrite prompt (`REWRITE_AND_INFER_TIME_PROMPT_FORMAT`) is **plain text in the repo** — lift it and run it on a Qwen3 we already have in llama.cpp. Note its prompt engineering explicitly *forbids* hallucinating sub-actions and mandates preserving laterality — so it will not paper over a weak base model. `[VERIFIED]`

### Dead ends — and why they died (this section saves the next agent a day)
| Model | Why it's dead |
|---|---|
| **SAMoR** (arXiv 2607.02148, 2026) | Perfect problem-shape: motion for *"any skeleton and topology"*, 8 part tokens, cross-topology, no retraining — the only thing here that doesn't assume SMPL. **No weights, no code link.** Project page shows a "⌘ Code" affordance pointing nowhere. Watch; don't plan. |
| **How to Move Your Dragon / T2M4LVO** (ICML 2025) | Reads like a bullseye (text + arbitrary skeletons + 70 species). The only public repo is **the project webpage** (56.6% JavaScript). The only released artifact is 1135 **captions** (CC-BY-NC-4.0) that *explicitly ship no motion data* "due to licensing restrictions". Truebones' commercial licence looks structural, so a release isn't obviously coming. **This is the #1 trap for anyone searching "text to motion non-humanoid" — it surfaces first and is unusable.** |
| **UniMoGen** (Autodesk Research, 2505.21837) | Architecturally the best match for our variable-J rig (no max-J padding, real-time). **No repo, no weights.** Corporate research → likely a Maya/MotionBuilder feature, not a release. Also conditioned on style/trajectory, **not free text** — wouldn't do "swat a fly" even if released. |
| **X-MoGen** (2508.05162) | Cross-species text→motion, 115 species. **No code, no weights.** Also uses a *shared* skeletal topology, so it wouldn't sidestep our naming work anyway. |
| **OmniMotion-X** (CVPR 2026) | The only candidate that would answer **both** fingers *and* face (SMPL-X 322-dim). Repo is a placeholder; **code release explicitly gated behind an unreleased dataset**. Highest-value watch-item. Note even on release the face half wouldn't transfer — our mesh has no blendshapes and our face is paint. |
| **MotionDreamer / SkelMo** (2606.01518, one month old) | Topology-agnostic, takes the asset's own skeleton **+ mesh** (no naming needed) — genuinely nice. But **video**-driven not text, no weights, and would be the heaviest port here. Recheck in ~3 months. |
| **SAME** (SIGGRAPH Asia 2023) | Solves *retargeting* — which we already solved **better** (roundtrip 0.000000° vs stock 51–135°). Would reintroduce torch **and** a T-pose assumption we know is false for the 45° A-pose soldier. Canonical hit for "skeleton-agnostic"; a solution to yesterday's problem. |
| **AnyTop** (SIGGRAPH 2025) | *Not dead — but not what you asked for.* Weights are real, public, **MIT**, 121MB, **2.28M params** (trains in 24h on one A6000). **But it is NOT text-conditioned** — its "textual joint descriptions" are T5 embeddings for *joint identity*, not action prompting. It generates in-distribution Truebones motion from a skeleton alone. Cannot do "swat a fly away". Also **requires joint names** + 4 identified joints (R/L hip, R/L shoulder) — the same naming prerequisite as Mixamo. Licence smell: MIT-tagged weights trained on a **commercial Gumroad asset pack**, while the *processed dataset* is withheld "due to licensing clarification". |
| **The 2025 academic cohort** (MDM, MotionGPT, T2M-GPT, MotionDiffuse, PriorMDM, FlowMDM, OmniControl, MotionLCM, StableMoFusion, MoGenTS, DART, GoToZero…) | **Superseded.** All ≤MoMask-era instruction-following, all positions-not-rotations, no fingers, no face, SMPL-only. Two genuine outliers worth remembering: **MotionLCM** (latent consistency, few-step — if *speed* ever becomes the constraint) and **OmniControl / PriorMDM** (explicit per-joint / end-effector spatial control — arguably a **better tool for "swat a fly"** than pure text, because you can just *say where the hand goes*). |

**Nothing in this field is trained on chibi.** Every candidate is SMPL, i.e. photoreal adult proportions. Stylised retargeting is *our* problem — which we have already largely solved, and which the literature cannot help us assess.

---

## 2b. Face / "moving skin" emoting

### The dichotomy that decides everything
**Every geometry-driven face rigger reads GEOMETRY. Our eyes and mouth are PAINT (0.16mm ≈ 1.7× the noise floor; paint-luminance↔depth correlation +0.059 = a decal).** They are not "hard" for that class of model — **there is no signal**.

This is not just our measurement. Independent corroboration:
- **RigAnyFace's own stated failure mode** is *"shell-like meshes lacking fine-grained geometric details"* — a verbatim description of our two-layer painted shell. A 2025 SOTA rigger, explicitly built for topological diversity including non-humanoids, **predicts failure on us, by the authors.** `[VERIFIED]`
- **NFR's README** demands *"the mouth and eyes need to be cut for correct global solving"*. Amusingly we **pass** its other requirement (*"no mouth/eye/nose sockets and eye balls inside the face"* — that's our shell exactly) and **fail** this one: there is nothing to cut. `[VERIFIED]`

**Corollary — and it's the whole survey: the only models that can see our face are ones that read the TEXTURE or a rendered image.**

### What is IMPOSSIBLE (settled — the previous agent was right)
- **Shutting a painted eye.** No deformer can. `FACE_RIG_REPORT.md`'s own verdict: the squash *"cannot fully 'shut' a painted eye… use it as a squint/blink accent."*
- **Opening a mouth that isn't modelled.** A visible jaw-open needs ~2.24mm of chin travel; mouth relief is 0.16mm — 14× too small.
- **Anything FLAME/ICT-topology.** Dead on topology *before* the paint problem even bites: FaceFormer / VOCA / EMOTE / EmoTalk / UniTalker / CodeTalker / SelfTalk (all FLAME, **5,023 verts**), AU-Blendshape / AUBlendNet (FLAME), ICT-FaceKit (own fixed 14,062-vert topology). To use any of them you must first FLAME-register a painted chibi — which requires the facial *geometry landmarks* we don't have. Clean no, all of them.

### What is ACHIEVABLE — and cheap
**libigl BBW + ARAP. It is already vendored in our tree.** `[VERIFIED]`
`/home/dbrain/dev/longcat-sparse-spike/thirdparty/ManifoldPlus/3rd_party/libigl/include/igl/` — 822 headers; `bbw.h`, `arap.h`, `harmonic.h`, `biharmonic_coordinates.h`, `boundary_conditions.h`, `lbs_matrix.h`, `slim.h` all present. Eigen alongside. `bbw.h` uses libigl's internal `igl::active_set` — **not MOSEK** (grepped). Licence **MPL2**; the GPL files are confined to `igl/copyleft/{tga,quadprog}.cpp`, which BBW/ARAP don't need (grep-verified, though the full transitive include graph was **not** audited).

Header-only C++, CPU, **zero weights, zero VRAM, deterministic** — and it arrived free with the ManifoldPlus retopo step you already run. Bounded Biharmonic Weights binds a mesh to points/bones/**cages** in arbitrary configuration: exactly the cage/lattice/RBF/ARAP family, and rig-agnostic (no names, no J, no SMPL, no T-pose).

**Can do:** brow raise (**brows are real geometry — 0.40mm, 4× noise floor**), cheek puff, squash/stretch, jaw-region skin travel, head tilt.
**Cannot do:** the two impossibles above.

**And you already have the proof it works.** `$AS/add_face_bones.py` + `FACE_RIG_REPORT.md`: **57,582 verts move on the jaw** (44,958 at w>0.4), render-verified, **no tearing**; jaw = local X, sign −1, ~30° clean (45° max); blink = eye bone scale-Y 0.05–0.20. Its blockers are **not** technique — they are (a) hand-tuned gilly constants (`JAW_HEAD=[0,0.025,0.332]`, lip line z 0.318→0.345, `EYE_L/R=±0.026`) and a hardcoded `bone_5` = head, and (b) bpy. **Both are fixable without a model:**
- Landmarks from the **texture** are free — per your own measurement, eyes are 150 luminance below skin, and an eyes-anchored valley detector hits the mouth line to **0.27mm in ~60 lines of numpy**. No LLM needed.
- Find the head bone **topologically**, not by name.
- Do the weight painting with **libigl BBW in C++** instead of bpy vertex groups.

That stack — auto-locate handles from texture → BBW → drive — is **the cheapest credible answer to "moving skin", is pure C++, needs no download, and costs no VRAM.**

### The bonus ask: can anything CREATE a mouth?
**Two real options, both honest about their cost.**

**(i) Talking Head Anime 3 (THA3) — paint the mouth, don't carve it.** `[VERIFIED weights exist; UNVERIFIED on our characters]`
Weights real: official Dropbox + HF mirror `OktayAlpk/talking-head-anime-3` (4 variants). **Models CC-BY-4.0 → commercial OK. Code MIT. Best licence position in this entire survey.** Its 45-param vector is almost a transcription of your ask: eyebrow `troubled/angry/lowered/raised/happy/serious`, eye `wink/happy_wink/surprised/relaxed/unimpressed`, `iris_rotation_x/y`, mouth `aaa/iii/uuu/eee/ooo/delta/smirk`.

It's 2D — **but our face IS 2D. That's the point, not the objection.** And it answers the bonus directly: **it paints an open mouth, with interior, onto a closed-mouth character — closed-at-rest for free, no geometry, no agape problem.**

Concrete fit with machinery we already have: you already do reproject-bake through a camera projection (RP_ATTR). So: render front → THA3 with all *rotation* params zeroed (only the face morpher fires; changes stay inside the 128×128 face box) → re-project the morphed face back through **the same camera** into the atlas → a texture-space expression set.

**Named risks, undressed:** (1) trained on anime — gilly plausibly in-distribution, **the toy soldier likely NOT**; (2) it wants a full standing-character framing, not a face crop; (3) front-view only (fine, since our mouth is paint, but it breaks at grazing angles); (4) it's PyTorch — a **build-time** python step, not a runtime dep, which is offensive but survivable; (5) **VRAM never stated** (author says ~20fps on a Titan RTX). Cheap to falsify: one image, one forward pass.

**(ii) Seam-cut + hidden cavity — your blocker is the CARVE, not the mouth.** `[INFERRED — this is a DESIGN PROPOSAL, not found in a paper, not prototyped]`
"Permanently agape" is a property of **SDF carving**, not of mouth creation. The industry answer is: don't carve a hole — **cut a seam and hide a cavity behind closed lips.** Locate the mouth line from texture (proven, 0.27mm) → **split verts along it AFTER quad retopo** (a topological cut has **zero feature size**, so the 0.776mm mean-edge floor **does not apply** — that constraint only kills a *modelled slot*) → procedural pocket behind → weight the lower lip to the jaw. **At rest the cut is coincident = invisible = closed.**
Encouragement, not evidence: **OmniFaceRig fits its inner mouth with ARAP + SDF — and we have both** (libigl ARAP on disk; UltraShape decoded logits). Unverified risks I can already name: z-fighting at rest; the interior needs its own texture (from where?); whether the cut survives GLB export/skinning; how SkinTokens' weights behave on split verts.

### More dead ends
| | Why |
|---|---|
| **OmniFaceRig** (arXiv 2606.08043, ~6 weeks old) | Maddening. Its stated input is **literally our asset**: *"static surface-only 3D character mesh, with no pre-modeled oral cavity"* → 155 FACS blendshapes + procedural teeth/gums/tongue + UV repack, fully automatic, targets stylised topologies. And it works by **segmenting rendered 2D views** — the one mechanism that *can* see paint. **But: no code, no weights, dataset "Coming Soon", and it depends on unreleased custom Sapiens-1B decoders + an unnamed VLM + SAM 3; 20–30s on an A100.** It is a **blueprint, not software**. Unresolved: its segmentation looks for a "mouth opening" — ours is a flat decal, so it might not find our mouth at all. Recheck in 3–6 months. |
| **RigAnyFace** | No code, no weights, no "coming soon". 5.4M params (would be a trivial port if they ever ship). Listed mainly as the **strongest independent evidence for the impossibility claim** — see above. |
| **THA4** | *Looks* like the dream (<2MB SIREN, browser, no PyTorch at runtime) and is a **trap** on three independent grounds, any one fatal: **CC-BY-NonCommercial**; **~30 GPU-hours per character** on an A6000 to distill (violates hands-off automation *and* the single 12GB 3060); and **the TF.js converter is not included** — the portable artifact the headline promises is not in the release. Use THA3. |
| **T2Bs** (ICCV 2025, Snap) | Textbook trap. Title says text→character blendshapes. README says: *"We use per-expression meshes to generate multi-view videos (the mesh-generation stage is not included)."* **It requires the answer as input.** It's a registration tool. Plus non-commercial (INRIA/MPII Gaussian rasterizer) and A100-scale. |

### Two things worth keeping
- **VRM 1.0 `textureTransformBind`** — a *shipping standard* built for exactly this problem (painted anime faces), with presets happy/angry/sad/relaxed/surprised, visemes aa/ih/ou/ee/oh, blink/blinkL/R, lookUp/Down/L/R. Zero VRAM, it's glTF JSON + a UV offset in the shader. **But it corrects the obvious plan: `textureTransformBind` is PER-MATERIAL** — one planar face chart is *not enough*, because a UV offset would move the whole face at once and eyes couldn't blink independently of the mouth. The industry answer (what VRoid does) is **separate materials/submeshes for face / eyes / mouth**, each with its own sprite sheet. Our confetti atlas (2,999–4,403 charts; face 159, mouth 17) is *less* of an obstacle than it looks — the charts are already tiny and disposable, so this is a re-atlas, not a fight. Caveat: the spec deliberately declines to define what the presets should *look* like — it gives us transport, not content. **Pair with THA3 to generate the sprite frames.** `[VERIFIED spec]`
- **NVIDIA Audio2Face-3D** — genuinely open, recent weights (NVIDIA Open Model License; regression v2.2 / diffusion v3.0). **It does not build a rig** — it emits 52 ARKit coefficients. As a *signal source* (jawOpen, eyeBlinkL/R, mouthSmileL/R, browInnerUp → our BBW handles + texture swaps) it is real and useful for the audio-driven half. Secondary, since you want **emotes** more than lip-sync.

---

## 2c. Fingers

### The dead-hands question — ANSWERED, empirically
**What happens to a 61-joint rig's 39 finger joints when you retarget a SMPL-22 clip?**

They get the bone's **rest local quaternion written on every frame.** `retarget.py::retarget()`:
```python
if aimc is None or B2S.get(b) is None or B2S.get(aimc) is None:
    ... quats[t, bidx[b]] = Lrest.as_quat(); continue
```
Its own docstring says it: *"Unmapped bones (fingers, leaves) stay at rest."*

**Measured** (max angular deviation vs frame 0, across all frames):
- `g_dance.json`: **18/61 bones move, 43 static**; every finger bone **<0.5° over 140 frames**. Same for `g_wave`.
- `gil_dance_deltafix.json` (the *new* delta retargeter): **identical 18/43 split**. The delta fix did not change this — it's a mapping gap, not a math bug.

**So: dead hands, rigidly welded to the forearm. And the wrist is dead too** (see TL;DR #4) — `bone_10`/`bone_33` static while `bone_9`/`bone_32` swing 62–124°. **Fixing the wrist is probably a bigger visible win than fingers, and it's free.** Honest limit: SMPL-22 has no joint past the wrist, so orientation is genuinely absent — options are (a) synthesise a plausible wrist from the forearm+elbow frame so the hand at least *follows*, or (b) move to a source with hand joints and aim the wrist at the middle-finger root. `[all VERIFIED]`

### Case 1 — the mesh HAS fingers (this is the soldier, not gilly)
- **Soldier** `_shootout_out/proj_v1/soldier_projected_rigged.glb`: **1 connected component**, clean. SkinTokens gave **3 digit chains/hand** — `bone_9 → {10-11, 12-14, 15-17}`, mirrored `bone_21 → {22-23, 24-26, 27-29}` — with **real skin weights** (dominant-vert counts: bone_12=193, bone_17=90, bone_15=77, bone_11=76, bone_10=62, bone_13=60, bone_14=20). Tips spatially distinct (bone_11↔bone_17 = 6.59% of body height). **Drivable today.** `[VERIFIED]`
- **Cheapest path to "move fingers", needing no model, no VRAM, no python: procedurally curl the digit chains SkinTokens already emitted.** Derive each chain from topology (hand bone's children = digit roots; follow single-child chains), compute the chain's own **bend plane from its rest geometry**, rotate about that axis by a curl parameter. **No bone names needed. Works with emergent J. Degrades gracefully** (no chains → nothing to curl). Tens of lines of C++. This is also your "move skin" idea applied to hands: curling bones that already own weights deforms real skin.
  **Honest risk:** on a stylised 3-digit toy hand this may read as a **deforming blob** rather than fingers. **That is an eye-test question and I cannot answer it.**
- **Real finger motion, if you want data not procedure:** **EMAGE** is the only runnable hands-capable model — and *we already run it and throw the hands away.* `emage_gen.py:45` loads `emage_vq/hands`; line 117 reconstructs `right_hand_pose = aa[:, 40:55]` (15 MANO joints/hand); **line 142 does `joints = out["joints"][:, :22, :]`** — the hands are **computed then discarded**. `:22`→`:55` recovers 30 finger joints. **Three caveats:** it's **audio-driven co-speech gesture, not promptable** (no "swat a fly"); its **weights are gone** (re-downloadable: `H-Liu1997/emage_audio`, public, **apache-2.0**, 556MB); and **nobody has ever run it on this box** — its own SETUP_REPORT admits *"everything verified except the actual GPU forward pass"*, so **its finger quality is completely unknown; it could be mush.** Also needs a 5→3 correspondence for the soldier. `[VERIFIED code; UNVERIFIED quality]`
- **Motion-X / Motion-X++** (15.6M SMPL-X whole-body poses) is a legit **data** route: mine a small library of hand curl trajectories offline, blend procedurally onto our chains. Ports for free (it's data). `[UNVERIFIED downloadability — access form not submitted]`
- **HandMDM** (2508.15902) is the only *text-promptable hands-only* model found — could layer prompted fingers on a HY-Motion body, which is architecturally exactly right. But weights are **promised, not confirmed**, and its motion prior is **sign language** (the non-sign generalisation claim is the authors' own).

### Case 2 — the mesh does NOT have fingers
**Honest answer: no.** Hand-mesh completion exists (**HHMR** inpaints missing fingers; **MMHMR**) but operates in **MANO space** — it would graft a **photoreal adult human hand** onto a chibi toy. Style collision, and MANO's finger diameter sits below quad-retopo's 0.776mm mean-edge floor at toy scale anyway. **HoloPart** completes *occluded* parts, not absent ones. **Nothing found generates stylised digits on an arbitrary stylised mesh.** The correct lever is **upstream** — the geometry stage or the input image — not a completion model.

### …but Case 2 is largely MOOT, and this is the load-bearing finding
**Gilly** `web/c1/gilly_rigged.glb` — from `make_gilly.sh` = **pixal3d@1024 + SkinTokens, NO UltraShape, dated Jun 1**:
- **16 connected components.** Body = comp0, ending at **x=±0.363**. comp1 (7,267 verts) + comp2 (5,900) at x=+0.390…+0.486; comp3 (5,525) + comp4 (5,069) at x=−0.490…−0.403. **Her hands are 4 detached blobs floating ~4cm off the arm stumps.**
- Confirmed independently by a top-down vertex-occupancy projection: **zero vertices in x=[−0.398,−0.36]**.
- Fingertip cross-section (5,556 verts): **one solid paddle, 0.033 × 0.065, zero finger lobes.**
- Her source art draws fingers as **line-art strokes on a flat edge-on paddle** — **the same paint-not-geometry pattern as her face.**
- SkinTokens **hallucinated 19 finger bones per hand into stray blobs.**
- **This is the brief's own warning firing:** *"T-pose subjects make pixal3d spawn stray bits at limb tips."* **Gilly is T-posed. Her hands ARE the stray bits.** `[VERIFIED — two independent methods]`

**Consequences:** (1) finger work should be judged on **the soldier**; (2) gilly is **not** evidence SkinTokens rigs fingers — she's evidence it **invents** them; (3) **gilly's hands are a geometry bug to re-run**, not an animation problem, and her GLB may simply be **stale** (the current pipeline has UltraShape welding; `make_gilly.sh` didn't). See §8 — this contradicts a stated "verified fact" in the brief and deserves a sceptical re-check **by someone who can render her.**

Ironic corroboration: `bonemap_v2.py` documents that stock `bonemap.py` **crashes** on the 38-bone soldier (`TypeError: cannot unpack non-iterable NoneType`) because his **hanging fingertips dip below the hips** and his arms get misclassified as legs. The crash is *caused by the soldier having fingers.* `[VERIFIED as a claim in the file; the repro was NOT re-run]`

---

## 3. THE BRIDGE: naming + retargeting

### The case FOR naming — and it's stronger than "it'd be convenient"
**Your instinct is right, and the decisive fact is not about Mixamo at all: every downstream auto-mapper is NAME-based, not topology-based.** `[VERIFIED — vendor docs]`
- **Godot**: auto-map is *"pattern matching for the bone names"*.
- **Unreal**: Auto Retarget Chains *"analyze[s] the hierarchy … and tr[ies] to match bone chains with pre-configured templates **based on the bone names**"* (exact matches not required).
- **Unity**: *"name your bones in a way that reflects the body parts"*.
- **Blender** ARP / Rokoko ship **Mixamo presets**.
- **glTF mandates nothing** — names are *"application-specific … for display"*.

So `bone_%u` (emitted at `glb_rigged.hpp:220` / `glb_rigged_textured.hpp:147`) is **not a cosmetic problem — it is the single thing that locks us out of every tool, and a name string is the whole key.** No model, no GPU, no weights. It is the cheapest, highest-leverage item in this document, and **you already made it a requirement for other reasons** (`$CP/STATUS-WIP-2026-07-17.md:18` — port `bonemap.py` + `rename_to_mixamo.py` into the C++ pipeline).

**And it pays twice:** AnyTop *also* requires joint names + 4 identified joints (R/L hip, R/L shoulder). Naming unlocks Mixamo **and** the skeleton-agnostic branch simultaneously.

**Evidence it works:** `MIXAMO_REPORT.md` — verified on gilly's 61 joints: all 22 core humanoid bones + 12 finger bones/hand, no duplicates, `skins==1`, `skin.joints` still 61, geometry/skinning **untouched** (only `node.name` changes). Leftovers become harmless `mixamorig:Extra_*` leaves. **Note our own `bonemap.py` beat the vendored upstream renamer**, which mislabeled gilly's right arm as the spine (it scored the climbing arm chain above the short neck) and dumped all right-hand fingers into "Extra". `[VERIFIED]`

### The case AGAINST / the honest catches
1. **It has been verified on exactly ONE character.** The soldier — 45° A-pose, different J, fingers that dip below the hips — **has never been through the naming path**, and stock `bonemap.py` reportedly *crashes* on him. "Naming is the unlock" rests on n=1. **This is the first thing to test and it is CPU-only.**
2. **A fixed library does not answer your actual ask.** Mixamo gives you whatever Adobe stocks; "swat a fly away" is prompt-driven generation. **Naming is complementary to a text→motion model, not a replacement** — it's the *quality floor* and the fastest route to a good render.
3. **Mixamo's institutional health is murky and the sourcing is weak.** *Free with an Adobe ID* is from Adobe's own FAQ (solid). *"Not supported anymore"* is **a forum report of what one support worker said** during a June 2025 outage — hearsay; **no official deprecation notice found**. Also: **its EULA forbids using the content to train ML models**, custom-model hosting is discontinued, and it needs an **FBX reader we don't have** (`MIXAMO_REPORT.md §Option B`: no pure-python FBX reader installed; would need Blender). **If we commit, mirror the clips.** A web service + Adobe login is fundamentally incompatible with hands-off automation and an inline C++ API — treat it as a **one-time offline harvest**, never a pipeline dependency. Also: I could **not** confirm Mixamo clips carry per-frame *finger keyframes* vs a static posed hand. `[UNVERIFIED]`
4. **Which standard?** Tolerance to our J-variance (28–61), ranked: **UE5 IK Retargeter** is *chain*-based and explicitly transfers *"between skeletons with varying numbers of bones, bone names, and orientations"* — most tolerant. **VRM 1.0** needs only **15 required bones** (no neck, no shoulders, no toes) and explicitly permits non-humanoid nodes *between* humanoid bones — very tolerant. **Unity Humanoid** needs 15 required bones **and a T-pose** (it ships "Enforce T-Pose" precisely because rigs aren't). **Mixamo's naming** is tolerant (extras → `Extra_*`; gilly's missing pinky is fine) but **its clips presume its own T-pose**. Practical: **emit `mixamorig:*` as the interop key, keep our own chain/aim map internally.**

### The retarget base pose — a human judgement, CONFIRMED, with one useful nuance
**Our conclusion holds. But "nothing automatic exists" is too strong, and the nuance is the payoff.** `[VERIFIED — vendor docs]`

Both Unreal and Godot **do** ship base-pose auto-solvers — and **both are exactly our `alpha=1`**:
- **Godot's Silhouette Fixer**: *"attempts to make the model's silhouette match … the reference poses"*.
- **UE's Auto Align**: *"automatically aligns all bones of the Target … to match the position and rotation of the Source"*, methods **Direction / Local+Global Rotation Axes / Mesh** (*"generates direction vectors based on vertex weighting"*).

And both ship **the same admissions we measured**:
- Godot: it *"cannot fix silhouettes which are too different, and it may not work for fixing bone roll"*; its Rest Fixer's Overwrite Axis gives *"**horrible results if the original Bone Rest set externally is important**"* — **that sentence is gilly's bow legs.**
- UE: Auto Align is needed *"when your Target … has very different mesh geometry"*.

**Critically: UE's "Mesh" method IS our `check_mesh_axis.py`** — direction vectors from vertex weighting — **and we already ran that experiment. The mesh AGREES with the bones on both rigs** (gilly's legs track her own mesh to **2.5°**; the soldier's arms to **14°**). **So the industry's best auto-solver would return the same answer ours does, and would not save us.**

**Verdict: it is a human judgement.** The industry's actual answer is: **default to alpha=1, then let a human EXCLUDE bones.** Godot has a `filter` array; UE lets you align *all / selected / selected+children*.

> **Actionable delta:** our `alpha` is **per-chain-group** (arm/leg/spine). The industry granularity is **per-bone with an exclusion list.** Move to that.

**And kill the popular assumption:** UE5.4's "one-click" retargeting is real, but it automates **chain mapping**, not the pose — the docs are explicit that Auto Retarget Chains *"does not automatically solve retarget poses."*

---

## 4. Candidate table

**Weights column is the decisive one.** `✅` = verified downloadable/on-disk. `❌` = does not exist today.

### Prompt → motion
| Name | What | Weights | VRAM | Rig assumption | Portability | Verdict |
|---|---|---|---|---|---|---|
| **HY-Motion 1.0** | Text→motion, 1B MMDiT + flow matching, 30fps/12s | ✅ `tencent/HY-Motion-1.0`, ungated, 6.05GB (Lite 1.844GB) | README says 26GB; **≈5-6GB sequential w/ Q4 Qwen3 GGUF** `[INFERRED, NOT MEASURED]` | SMPL-H **22 joints, NO hands**. Emits **rot6d + transl** → no IK | **Best reuse here**: Qwen3 GGUF exists + llama.cpp ✓, CLIP ✓, MMDiT≈Flux ✓, Euler ✓. Skip proprietary `fbxsdkpy`, use `output_format='dict'` | Strong candidate — **licence excludes EU/UK/SK** |
| **MoMask** | Text→motion, RVQ-6 + masked transformer | ✅ **ON DISK**, 194-201MB, `.venv` intact | <2GB; **README says it runs on CPU** | SMPL-22, **positions only** → needs IK. No hands/face | Easiest port here (tiny transformers + CLIP ✓); safetensors mirror exists | **The free control experiment** — never re-judged through the fixed retargeter |
| **Text2MotionPrompter** | Terse prompt → caption + frame count | ✅ ungated, **apache-2.0**, 61GB | 61GB fp16 | N/A | **Don't download** — the rewrite prompt is plain text in the repo; run it on our Qwen3 | Lift the prompt, skip the model |
| **`ZeyuLing/motius`+`hftrainer`** | ~44 mirrors of the named list, safetensors + Mean/Std | ✅ files exist (HF API) — **⚠ ~0 downloads, unverified they load** | varies (MotionStreamer **1.27GB**; GoToZero-7B 27GB; PRISM 32.7GB) | mostly HumanML3D-263 → SMPL-22 | Best porting substrate found | Weights **source** + A/B harness, **not** a product dep |
| **AnyTop** | Skeleton-agnostic diffusion, **2.28M params** | ✅ `Inbar2344/AnyTop`, **MIT**, 121MB | <2GB (demoed on a 2080 Ti) | **Arbitrary topology** — but needs **joint names** + R/L hip + R/L shoulder + BVH | Port = a weekend (T5 joint-name embeds are precomputable → `cond.npy`) | **NOT text-conditioned** → can't do the ask. Curiosity/hedge |
| **MotionLCM / OmniControl / PriorMDM** | latent-consistency speed / per-joint spatial control | ✅ via mirrors `[UNVERIFIED]` | tiny | SMPL-22 | tiny transformers | **Remember OmniControl**: explicit end-effector control may beat text for "swat a fly" |
| ~~2025 academic cohort~~ | MDM/T2M-GPT/MoGenTS/DART/FlowMDM/… | mixed | tiny | SMPL-22 positions | easy | **DEAD: superseded.** ≤MoMask-era, no fingers, no face |
| ~~**SAMoR**~~ | *any* skeleton+topology, 8 part tokens | ❌ **no weights, no code link** | — | arbitrary — *the dream* | — | **DEAD: paper only.** Watch |
| ~~**T2M4LVO / "Dragon"**~~ | text + 70 species | ❌ **the repo is a webpage**; only 1135 captions (NC), *"does not provide motion data"* | — | arbitrary | — | **DEAD: #1 search trap.** Truebones licence looks structural |
| ~~**UniMoGen**~~ | no-padding variable-J, real-time | ❌ no repo | — | arbitrary — best-matched design | — | **DEAD: Autodesk corporate.** Also **not text**-conditioned |
| ~~**X-MoGen**~~ | cross-species text→motion | ❌ | — | *shared* topology → wouldn't skip naming anyway | — | **DEAD: no code** |
| ~~**OmniMotion-X**~~ | SMPL-X whole-body, text+music+speech | ❌ **gated behind unreleased dataset** | — | SMPL-X (**hands + jaw**) | — | **DEAD today — best watch-item.** Only one that'd do fingers *and* face |
| ~~**MotionDreamer**~~ | topology-agnostic from **video** | ❌ | heaviest here | takes our mesh — no naming needed | bad | **DEAD (1mo old).** Video not text. Recheck |
| ~~**SAME**~~ | skeleton-agnostic embedding | ✅ in-repo ckpt | small | **needs T-pose** | poor | **DEAD: solves yesterday's problem, worse than ours** (0.000000° roundtrip) |
| **Mixamo / AMASS retarget** | Pro clips, no model | ⚠ **ZERO clips on disk** (only UniRig *rig* FBX + 1 template BVH) | **0** | needs **names** | no model to port | **Strong** — the quality floor. Blocked on an Adobe login + an FBX reader |
| **Procedural / IK / spring-bone** | FK/IK/motion-matching/jiggle | **N/A — we write it** | **0** | **NONE — zero assumptions** | **it IS C++** | **Strong complement.** Can't be prompted; fixes foot-plant; only thing that touches non-humanoid today |

### Face
| Name | What | Weights | VRAM | Rig assumption | Portability | Verdict |
|---|---|---|---|---|---|---|
| **libigl BBW + ARAP** | handle/cage/bone deformation | **N/A — ✅ ALREADY VENDORED** (822 headers, MPL2, no MOSEK) | **0 (CPU)** | **NONE** | **zero — it's already in the tree** | **The honest "moving skin" answer** |
| **`add_face_bones.py`** | jaw+eyeL+eyeR on gilly | N/A — ✅ on disk, **render-verified** (57,582 verts, no tearing) | 0 | **hardcodes `bone_5` + gilly world constants** → violates hands-off | technique ports; bpy is incidental | **Proof it works.** Auto-derive the constants from texture |
| **THA3** | 45-param 2D anime face morpher — **paints mouths** | ✅ Dropbox + `OktayAlpk/talking-head-anime-3`. **Models CC-BY-4.0 (commercial OK), code MIT** | unstated; ~20fps Titan RTX `[UNVERIFIED]` | **NONE — never sees a rig** | PyTorch, **build-time only** | **Answers the bonus.** Soldier likely out-of-distribution |
| **VRM 1.0 `textureTransformBind`** | shipping standard for painted faces | **N/A — spec** | **0** | none | trivial (glTF JSON + UV offset) | **Validates texture-space — but it's PER-MATERIAL: split eyes/mouth into their own materials** |
| **Audio2Face-3D** | audio → 52 ARKit coeffs | ✅ HF, NVIDIA Open Model License | small `[UNVERIFIED]` | assumes an ARKit rig — **doesn't build one** | plausible ONNX | **Driver, not rigger.** Secondary (you want emotes > lip-sync) |
| ~~**OmniFaceRig**~~ | **exactly our input spec** → 155 FACS + procedural cavity | ❌ no code, dataset "Coming Soon", deps unreleased | A100-class | none of ours | worst-case python | **DEAD: blueprint only.** Its ARAP+SDF cavity recipe is reusable — **we own both** |
| ~~**RigAnyFace**~~ | neural auto-rig, 5.4M params | ❌ no link at all | tiny | topology-flexible | trivial *if* shipped | **DEAD — and it's our best independent evidence: its OWN failure mode is "shell-like meshes lacking fine-grained geometric details"** |
| ~~**THA4**~~ | <2MB SIREN, browser | ✅ Dropbox | tiny | none | **converter NOT included** | **DEAD ×3: NonCommercial + ~30 A6000-hours/character + no converter** |
| ~~**NFR**~~ | topology-flexible deformation AE | ✅ Google Drive | small | **"mouth and eyes need to be cut"** — we have nothing to cut | DiffusionNet + pytorch3d, pinned torch 1.12/cu113 | **DEAD.** Superseded by RigAnyFace (2.77mm vs 1.01mm) |
| ~~**T2Bs**~~ | "text→character blendshapes" | ❌ (code ≠ title) | A100 | head meshes | non-commercial rasterizer | **DEAD: requires per-expression meshes as INPUT** — it needs the answer |
| ~~FLAME/ICT family~~ | FaceFormer/VOCA/EMOTE/EmoTalk/UniTalker/CodeTalker/SelfTalk/AU-Blendshape/ICT-FaceKit | mostly ✅ | small | **FLAME 5,023 verts / ICT 14,062** | moot | **DEAD on topology, before the paint problem even applies** |

### Fingers
| Name | What | Weights | VRAM | Rig assumption | Verdict |
|---|---|---|---|---|---|
| **Procedural digit curl** | curl the chains SkinTokens already emitted | **N/A** | **0** | **NONE** | **Only thing that works on the soldier today.** Eye-test risk: may read as a blob |
| **Fix the dead wrist** | `SMPL_CHILD` has no key for joints 20/21 | **N/A — our bug** | 0 | none | **Free, probably the biggest visible win.** Data limit: SMPL-22 has no post-wrist joint |
| **EMAGE** | audio→co-speech gesture, **SMPL-X 55 incl. 30 hand joints** | ❌ **NOT on disk** (SETUP_REPORT is false) — re-downloadable `H-Liu1997/emage_audio`, **apache-2.0**, 556MB | ~2-3GB `[UNVERIFIED]` | SMPL-X. We compute the hands then **discard them at `emage_gen.py:142`** | **The only runnable hands model — and NEVER EXECUTED on this box.** Audio-driven ≠ promptable |
| **Motion-X / X++** | 15.6M SMPL-X poses (**data**) | ⚠ access form, `[UNVERIFIED]` | N/A | SMPL-X | Mine a hand-pose library offline; **data ports for free** |
| **HandMDM** | **text→hand motion** (MDM-class) | ⚠ "will make available" `[UNVERIFIED]` | small | SMPL-X hands + 13 upper body | Only text-promptable hands model. **Prior is sign language** |
| ~~**HHMR / MMHMR**~~ | hand-mesh completion / inpaint fingers | not chased | — | **MANO — photoreal adult hand** | **DEAD: would graft a real human hand on a chibi toy.** Also below the 0.776mm retopo floor at toy scale |
| ~~**HoloPart**~~ | part amodal completion | — | — | — | **DEAD: completes OCCLUDED parts, doesn't invent absent ones** |

---

## 5. What we already have vs what needs building

### ON DISK, VERIFIED
| Asset | Status |
|---|---|
| **MoMask** + 194-201MB checkpoints + `.venv` | ✅ **complete and runnable.** `momask_setup.log` ends in an error — **that log is STALE**, the weights landed later |
| **8 MoMask clips** (`_{wave,jump_joy,clap,bow,cheer,walk,dance,kick}.npy`, (T,22,3) @20fps, T=80-199) + `_built.tsv` (**the prompt list**) | ✅ |
| **Retargeted outputs**: 52-bone `*.json`, 61-bone `g_*.json`, 64-bone `talk_hello`/`tc_*` @30fps w/ `scales`+`audio` | ✅ |
| **`puppy-eyetest/anim/clips/`**: walk/cheer/dance × {stock, delta, **deltafix**} × {gilly 61, soldier 38} | ✅ **3 of 8 motions rebuilt; NONE rendered** |
| **`retarget_delta.py` + `bonemap_v2.py`** (roundtrip **0.000000°**; rig-invariant **0.1°** spread; base-pose fix 100%→0% crossing frames) | ✅ |
| **`bonemap.py` / `rename_to_mixamo.py` / `MIXAMO_REPORT.md`** + mixamo/ue5/vroid yamls + `gilly_mixamo.glb` | ✅ **verified on gilly only** |
| **three.js r160 rig player** (self-hosted, offline) at `puppy-eyetest/anim/` | ✅ |
| **libigl + Eigen** (822 headers, BBW/ARAP/harmonic/SLIM, MPL2, internal active_set) in `thirdparty/ManifoldPlus/3rd_party/` | ✅ **free, already there, unnoticed** |
| **`add_face_bones.py` / `audio_to_jaw.py` / `FACE_RIG_REPORT.md`** + `gilly_face.glb` (64 bones) + verify PNGs + `jaw_talk.mp4` | ✅ render-verified, gilly-hardcoded |
| **`SMPLX_NEUTRAL_2020.npz`** (167MB) + EMAGE code + `.venv` | ✅ code/venv/SMPL-X — **but see below** |

### CLAIMED-BUT-ABSENT (the traps)
| Claim | Reality |
|---|---|
| "EMAGE weights are pre-downloaded, 14 files, all fetched" (`SETUP_REPORT.md`) | ❌ **`~/.cache/huggingface/hub/models--H-Liu1997--emage_audio` DOES NOT EXIST.** Hub holds only Wan/LTX/NAVA/gemma |
| "EMAGE is set up and verified" | ❌ Its own report admits **the GPU forward pass was never run.** *(Though `out/hello_joints.npy` (102,22,3) + `clips/talk_body.json` fps=30/T=102 prove it DID run once, after the report was written — the report is falsified in both directions)* |
| "HY-Motion is 24GB, needs fitting" (`HANDOFF.md:153`) | Contradicted by our own memory (`4.17GB ckpt; the 24GB is multi-seed activations`). **Both were second-hand.** Now resolved: 6.05GB repo, 26GB README = fp32 text encoders |
| "REMAINING HUMAN STEP: drop a few Mixamo FBX somewhere" (`HANDOFF.md:155`) | ❌ **Never happened.** Zero animation FBX on disk. The only FBX are UniRig **rig** examples (giraffe/carrot/tira/bird/miku); the only BVH is `MoMask/visualization/data/template.bvh` (a template) |
| The brief: "gilly has fingers (61 joints); the soldier has mitten hands" | ❌ **Backwards** (§2c) |
| The brief: "SkinTokens beam=20" | ⚠ `$CP/rig_pipeline.hpp:40` says `int num_beams = 10; // HF-official beam-sample config (default rig recipe)`. Call sites not traced. Doesn't change any finger conclusion (those come from shipped GLBs) — but "deterministic at beam=20" may describe a config we don't default to |
| The brief: "J ∈ {28,29,33,38,61}" | ⚠ `walk.json` has **52 bones**. The list is incomplete |

### NEEDS DOWNLOADING / BUILDING
| Item | Cost |
|---|---|
| HY-Motion 1.0 (+ Lite) | 6.05GB, ungated |
| Qwen3-8B GGUF Q4_K_M via `Aero-Ex/Hy-Motion1.0` (997 dl) + CLIP-L | 5.03 + 1.71GB |
| EMAGE weights (re-download) | 556MB, apache-2.0 |
| Mixamo clips | **an Adobe login + an FBX reader we don't have** |
| Port `bonemap` + `rename_to_mixamo` → C++ | **already a requirement** (`STATUS-WIP:18`) |
| Auto-derive face landmarks from texture (~60 lines numpy) | free per your own measurement |
| Per-bone alpha + exclusion list in the retargeter | small |
| Wrist aim fix | small |
| Procedural digit curl | tens of lines |
| Connected-component QA gate (~40 lines) | **would have caught gilly's blob hands months ago** |

---

## 6. Recommended sequencing (perfect the model FIRST; animation is a small chunk at the end)

**Stage 0 — free, CPU-only, no downloads, no GPU. Do these regardless of everything below.**
| # | Item | Effort |
|---|---|---|
| 0.1 | **Render the 3 `_deltafix` clips on the eye-test page.** Re-judge MoMask honestly (§7) | ~1h |
| 0.2 | **Fix the dead wrist** (`SMPL_CHILD` missing 20/21) | ~1h |
| 0.3 | **Connected-component QA gate** on every generated mesh (`components > 1 → fail`) | ~1h |
| 0.4 | **Re-run gilly through the CURRENT pipeline** (with UltraShape) and re-check the hands | GPU, small |
| 0.5 | **Run `bonemap.py`/`rename_to_mixamo.py` on the SOLDIER.** n=1 → n=2. It reportedly **crashes** on him | ~half a day |
| 0.6 | **Per-bone alpha + exclusion list** (the industry granularity) | ~half a day |

**Stage 1 — the model (your stated priority). Nothing below competes with this.**
Finish the C++ port (wire UltraShape, P3-SAM GPU heads, quad retopo), and **port `bonemap` + `rename_to_mixamo` into it so the shipped GLB carries `mixamorig:*` — which you already decided.** *That single string change is the whole animation bridge.*

**Stage 2 — animation, in cost order:**
| # | Item | Effort | Risk |
|---|---|---|---|
| 2.1 | **Procedural digit curl** on the soldier's chains | ~1 day | may read as a blob — eye-test |
| 2.2 | **Spring/jiggle secondary motion** (post-pass on *any* clip; exaggerated overshoot is *stylistically correct* on a chibi toy) | ~2 days | params are per-rig unless derived from bone length/depth |
| 2.3 | **Mixamo harvest** (one-time, offline, mirrored) — the quality floor + real finger keys | blocked on Adobe login + FBX reader | service health `[UNVERIFIED]`; clips presume Mixamo's T-pose |
| 2.4 | **HY-Motion in stock python, off the critical path**, generating SMPL-22 npy → our native retargeter | days | **VRAM unproven; licence excludes EU/UK/SK** |
| 2.5 | **Face "moving skin"**: texture landmarks → libigl BBW → brow/cheek/jaw. Pure C++, 0 VRAM | ~1 week | BBW convergence/time on ~694k verts is **untested** |
| 2.6 | **THA3 expression bake** (build-time python) → texture-space emotes + a *painted* mouth | ~1 week | **soldier likely out-of-distribution** |
| 2.7 | **Native HY-Motion port** (MMDiT ≈ Flux ✓, Qwen3 GGUF ✓, CLIP ✓, Euler ✓) | **weeks — only after 2.4 proves quality** | the Qwen3 hidden-state extraction is the real unknown |

**Explicitly NOT recommended:** porting MoMask (HY-Motion supersedes it *and* reuses more of our infra); downloading Text2MotionPrompter (61GB — lift the prompt); THA4; anything with ❌ in the weights column.

---

## 7. The cheap proof of concept — run this next

**Zero downloads. Zero GPU. Everything already on disk. It tests the single most consequential open question in this document: was the jank MoMask, or was it us?**

1. Take `/home/dbrain/dev/puppy-eyetest/anim/clips/` — walk/cheer/dance × {**stock**, **delta**, **deltafix**} × {gilly 61-bone, soldier 38-bone}. **Already built. Never rendered.**
2. Load them in the three.js r160 player at `/home/dbrain/dev/puppy-eyetest/anim/` (self-hosted, offline).
3. Put them **side by side on the eye-test page**: same motion, same character, stock vs delta vs deltafix. **You judge.**
4. **Read the answer:**
   - *deltafix looks fine* → **MoMask was never the problem; the jank was ours.** The whole "we need a better model" premise weakens, Mixamo/procedural becomes the fast path, and HY-Motion drops to a nice-to-have.
   - *deltafix still looks jank* → the model **is** mid-tier, and HY-Motion (better instruction-following **and** native rotations) earns its download.
   - Either way you will see the **dead hands and the dead wrist** immediately — 43/61 bones frozen — which tells you whether that's a real visual problem or something nobody would notice on a chibi toy.

**Bonus, still free:** rebuild the other 5 MoMask motions (bow/clap/kick/wave/jump_joy) through `retarget_delta.py` — the `.npy` sources are on disk and MoMask's own README says it runs on CPU. Full 8-clip A/B, **no GPU contention.**

**Second cheap PoC, ~1 hour, CPU:** run `bonemap.py` on the soldier. It reportedly crashes (arms misclassified as legs because his fingertips hang below his hips). That is a **CPU bug fix** that takes "naming is the unlock" from n=1 to n=2 and de-risks Stage 1.

---

## 8. Open questions / what nobody verified

**Nobody touched the GPU. Nothing here is render-verified. Every "would work" is an argument, not a result.** That is the exact gap this project keeps getting burned in.

1. **Is gilly's blob-hand finding real, or is her GLB just stale?** `make_gilly.sh` = pixal3d@1024 + SkinTokens, **no UltraShape**, dated Jun 1; the soldier came from the newer `proj_v1` path and is clean. UltraShape welds stray bits. **So "pixal3d can't do hands" may be entirely the wrong conclusion — the right one may be "gilly's GLB is 6 weeks stale, re-run her."** The measurement is stark (4 detached blobs, 4cm gap, two independent methods) but it **contradicts a stated "verified fact" in the brief**, so it deserves a sceptical re-check **by someone who can render her**.
2. **Would pixal3d at 1536 give gilly fingers?** Probably not — **her source art has no silhouette evidence of fingers at all** (line strokes on a flat paddle). Resolution cannot invent silhouette. The soldier's fingers *are* in his silhouette, which is plausibly *why* he has them. **Hypothesis, untested.**
3. **Does the soldier's 3-digit hand READ as fingers when curled?** His tips own only 20–193 verts; the 0.776mm retopo floor may or may not survive them. **The entire procedural-curl recommendation hinges on this and it is unknown until rendered.**
4. **HY-Motion's real VRAM.** The 26GB→~6GB argument is **arithmetic from verified file sizes, not a measurement.** The decisive unknown: **is Qwen3-8B a one-shot freeable encoder, or re-invoked during denoising?** If the latter the whole argument collapses. Note the README's own mitigations (`--num_seeds=1`, <30 words, <5s) imply **26GB is a real observed number for someone**, not a typo. **Do not let anyone (including this doc) claim "HY-Motion fits on the 3060" until it has been run.**
5. **The Qwen3-8B hidden-state extraction is the biggest porting risk, and it is unexamined.** Which layer? What pooling? `crop_start: 0`, `enable_llm_padding`, the `PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION` system prompt, the chat template — and whether **llama.cpp Q4 hidden states are numerically close enough to torch fp16** not to wreck conditioning. If this doesn't reproduce, the "we already have Qwen3" advantage partly evaporates.
6. **`enable_special_game_feat: true` and `mask_mode: narrowband`** appear in both shipped HY-Motion configs, are **undocumented**, and nobody examined them. `enable_special_game_feat` in particular sounds load-bearing for game-asset use.
7. **Inference SPEED is unverified for every candidate.** Tencent publishes no timings anywhere I could find.
8. **Licences — where the owner sits decides one of these.** HY-Motion: *"THIS LICENSE AGREEMENT DOES NOT APPLY IN THE EUROPEAN UNION, UNITED KINGDOM AND SOUTH KOREA"* — **a lawyer question if you're in the UK/EU**; only the definitions section was read. MoMask: **MIT, verified by reading the file** — but **HumanML3D derives from AMASS = non-commercial**, so **the model licence can be clean while the DATA licence is not.** SMPL-X: non-commercial + normally registration-gated (EMAGE's mirror needing no registration is a **smell, not a clearance**). AnyTop: **MIT-tagged weights trained on a commercial Gumroad pack**, while the processed dataset is withheld "due to licensing clarification". Mixamo: **forbids using the content to train ML models**. libigl: MPL2, but **the full transitive include graph of `bbw.h`/`arap.h` was not audited** for `igl/copyleft/` reachability.
9. **`ZeyuLing/motius` mirrors: files exist; nothing more.** ~44 repos, ~0 downloads, third-party reimplementations. **Confirmed via HF API that the files exist with plausible sizes. NOT confirmed that they load, run, or match their papers.** This is precisely our historical burn pattern, flagged rather than dressed up.
10. **The `awesome-text-to-motion` "Weights ❌" column is unreliable** — it marks ❌ for essentially every 2025 model, yet **MoMask demonstrably has weights (we hold them)**. First-party gdown links for MDM/MotionGPT/T2M-GPT/MotionDiffuse/PriorMDM/FlowMDM/OmniControl/StableMoFusion/MoGenTS were **not tested**. So "weights exist" for the named list rests on **unverified mirrors**, not **unchecked first-party links**. Weakest part of this report.
11. **HumanTOMATO deserves 10 minutes.** It is the best-shaped candidate (text **+** hands, ICML 2024) and I'm **most likely to be wrong about it** — the README was fetched *rendered* and a checkpoint link may be hiding in a collapsed `<details>`. Its releases are OpenTMA (alignment) + `tomato` (representation); **no H2VQ / Hierarchical-GPT generation weights found.** Check the repo's releases/issues directly.
12. **libigl was never compiled** (and won't be, per the build rules). Two live unknowns: does that **old vendored snapshot** (its CMakeLists still references deprecated python bindings) build against our toolchain, and **does `igl::active_set` converge on a ~694k-vert mesh in acceptable time?** BBW may need decimation. Genuinely unknown.
13. **THA3's training set and VRAM are both unstated.** VRoid-rendered is *assumed*, not confirmed. The repo has **no limitations section** — absence of documented failure modes is not evidence of robustness.
14. **No download link was tested** — THA3/THA4 Dropbox, NFR Google Drive, Audio2Face-3D HF ids. Confirmed *published*, not that they *resolve*. Dropbox/Drive links rot. The THA3 HF mirror shows the 4 expected variant dirs; **completeness not verified.**
15. **The painted-face measurements (0.16mm, +0.059, 0.43/0.40mm, 2.24mm, 0.776mm) and the atlas numbers (2,999-4,403 charts) are taken from the brief, not reproduced.** The central impossibility claim rests on them — though **RigAnyFace corroborates it from outside**.
16. **The seam-cut + hidden-cavity mouth is my reasoning, not a paper and not a prototype.** Named risks: z-fighting at rest; the interior's texture source; whether the cut survives GLB export/skinning; how SkinTokens' weights behave on split verts.
17. **`retarget_delta.py`/`bonemap_v2.py` were read, not re-run.** The claimed `bonemap.py` soldier crash was **not** reproduced.
18. **Never verified:** that Mixamo clips carry per-frame *finger* keyframes (vs a static posed hand); that our three.js player can do per-material UV offset or morph targets (the VRM route **assumes a player-side change nobody scoped**); whether EMAGE's HF repo is still ungated; Motion-X's gating; **the 4 small stray components near gilly's face** (comp5-7, 283-378 verts) — noticed, uninvestigated, possibly relevant to the parked face work.
19. **Not surveyed at all:** Rigel3D, RigMo, AniGen, ViPS, SKDream, TapMo, BiMotion, PALUM, MotionBricks, "Semantic-Aware Motion Encoding for Topology-Agnostic Character Animation" (2605.27055 — sounds on-topic), Neural Face Skinning (2505.22416), Live2D-style 2D rigging, Wan2.2-Animate-14B, and the SD/Flux-inpaint viseme-sprite route (**deliberately skipped**: "we have Flux inpaint" is on the record as a previously-false claim, and I did not check the weights myself).
