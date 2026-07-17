# STATUS / WIP BOARD — image→rigged-3D — 2026-07-17

Companion to `HANDOFF-2026-07-17-texture-perf-threads.md` (which has the evidence + repro commands).
This doc answers three owner questions: **what's outstanding**, **is it automatic or manual**, and
**what must land before the perf phase**.

**Legend — status:** ✅ done · 🟠 built, not validated · 🔍 diagnosed, not fixed · 🔴 broken/unknown · ⬜ not started
**Legend — automation:** 🤖 = automatic in the C++ pipeline (images in, asset out) ·
🎚️ = automatic *after* a one-time decision · 🖐️ = **needs a human every time** · 🚫 = not in the pipeline at all

---
## 🔔 OWNER GOALS CLARIFIED (late 2026-07-17) — THESE CHANGE THE BOARD

1. **ANIMATION IS A PIPELINE REQUIREMENT.** *"animation is definitely a pipeline requirement - or at
   least getting the rigging 'animation ready' (named properly to some kind of standard)… the actual
   goal included animation using something like **HYMotion**… if we have sane rigging can just grab
   animations online."* ⇒ **port `bonemap.py` + `rename_to_mixamo.py` INTO the C++ pipeline** so the
   shipped GLB carries `mixamorig:*` names, not `bone_%u`. **The rig is not "done" until it is
   animation-ready.** Sequencing (owner): perfect image→rigged-model first; animation is *"a small
   chunk at the end"*.
2. **3D PRINTABILITY is a goal** — *"the benefit of filling the mesh is the models become 3d printable
   easily."* ⇒ `REMESH_CLOSE_R=3` is a **product feature**, not a perf nicety. Raises the stakes on the
   **inverted winding** too (slicers care about orientation).
3. **Fix the winding, don't just document it.** (in flight)
4. **Seam blending** front↔back wanted: *"be nice if it didn't feel like there was a seam."*
5. Owner texture verdict: fray fix **"much better"**, texfix **"much better, usable"**, camo **"gone"**.
   Net: *"usable - not perfect."* Remaining: eyes (INTRINSIC — the volume never drew them), pants/boots
   blend, crossover seam.
6. **The `solid` eye-test's noise/ripple is a TEST ARTIFACT, not the fix** — it was run `--no-refine`
   (raw marching cubes, stride 2, no smoothing). Owner: *"feels like something ultrashape would smooth
   out"* — correct. **Re-run R=3 WITH refine before judging.**

## ⚠️ THE HEADLINE ANSWER: "are the animation ticks automatic?"

**No. NONE of the animation work is in the pipeline.** It is **all Python side-tools** under
`/home/dbrain/dev/puppy-eyetest/anim/tools/` (`retarget_delta.py`, `bonemap_v2.py`, `build_ab.py`,
`check_*.py` …), run by hand, emitting clip JSON for a web page. `grep` for retarget/bonemap in
`$CP` → **zero real hits**. `image_to_rig` cannot animate anything and does not know clips exist.

**The ticks mean "proven correct in a script", NOT "the product does it."** That distinction was not
made clearly enough in conversation and it matters: the rig the pipeline emits is a **bind-pose
skeleton + skin weights**, nothing more.

**Conversely — the CORE product IS automatic.** `image → geometry → refine → quad → projected texture
→ auto-rig → rigged textured GLB` runs in ONE C++ process with **no human input beyond the images**.
Everything measured at runtime (normal convention, silhouette fits, head-bone derivation). No
per-model tuning exists anywhere in it.

---
## TEXTURE

| thread | status | auto? | note |
|---|---|---|---|
| Front-projection (`--tex-project`) | ✅ | 🤖 | owner: *"pretty damn good"*. Needs front+back images = **the API contract**, not manual intervention |
| 3D-aware hole fill (the camo) | ✅ | 🤖 | owner: *"shoes and head is fixed"*. 97.8% of holes |
| Normal-convention detection (`Cam::nsign`) | ✅ | 🤖 | measured vs z-buffer, 96.1% vs 3.9% |
| Back-view auto-align | ✅ | 🤖 | silhouette-bbox fit, scale 1.0252 on the soldier |
| Front align (`TEXPROJ_FRONT_ALIGN`) | 🟠 | 🤖 | **A/B RUNNING.** Only buys ~1.2px at the chest — may not reach the buttons |
| BG-sample reject (`TEXPROJ_BG_REJECT`) | 🟠 | 🤖 | **A/B RUNNING.** For the black streaks |
| **Bake frays the volume's clean cuts (5.13%)** | 🔍 | — | **NOT FIXED. Our defect**, bimodal at material boundaries. `RP_ATTR` default is *documented* as speckle-prone; the alternative is worse (14.86%) |
| **Tex-DiT regression (proj vs cross)** | 🟠 | 🎚️ | **wired, NOT A/B'd.** Possibly **revert-level**. One-time decision, then automatic |
| Gold sits off the bumps | 🔍 | — | registration, mostly **predates refine** (coarse is already 0.5% narrow). May be beyond a bbox fit |
| "Softness is intrinsic" | ✅ | — | **half-false**; detail loss IS intrinsic ⇒ front-projection justified |

## MODEL / GEOMETRY

| thread | status | auto? | note |
|---|---|---|---|
| Double-walled shells (diagnosis) | ✅ | — | settled after 3 agents + 2 tautologies. Fill leaks **regionally**; only the HAT seals |
| **`REMESH_CLOSE_R=3` fix** | 🟠 | 🎚️ | **coded, default OFF, NOT GPU-TESTED.** −57% verts, −73% holes *if it's a body not a blob*. **Eye-test decides** |
| **Winding flip at source (`usr::refine`)** | 🔍 | — | **NOT FIXED.** Shimmed in texture only. Latent landmine — masked today by `doubleSided:true` |
| MoGe: 2× DiT cost + OOM | 🔍 | 🤖* | works, but 46.5° vs 42° → +20.6% verts → **CUDA OOM**. *Automatic but currently unusable at res1536* |
| **Refine emits 668k vs reference 165k verts** | 🔴 | — | **UNEXPLAINED, NEW.** May *be* why the refine costs 3339s. **Chase before optimising** |

## ANIMATION / RIG — ⚠️ read the headline above

| thread | status | auto? | note |
|---|---|---|---|
| Auto-rig (SkinTokens, in-pipeline) | ✅ | 🤖 | skeleton + skin from one Qwen3 token stream. **This IS in the product** |
| Wobble page | ✅ | 🚫 | a **demo**, not a feature. Owner: *"best animation ive seen from this"* |
| Delta retarget | ✅ | 🚫🖐️ | **Python side-tool.** Correct (roundtrip 0.000000°, rig-invariant) but **not wired to anything** |
| Soldier's crossing arms | ✅ | 🚫🖐️ | fixed via **retarget base pose** (100%→0%). **α (arm=1/leg=0) is a HUMAN JUDGEMENT** — nothing geometric picks it; that's why Unreal/Unity ask |
| `bonemap_v2` (soldier crash fix) | ✅ | 🚫 | Python. Stock `derive_map` **crashes** on the soldier |
| Mixamo bone naming | 🟡 | 🚫🖐️ | **exists on `/mnt/hdd`, never wired, never run on the soldier.** Would make clips portable |
| Head lean | ✅ | — | **not a bug** — it's in MoMask's motion data |
| Gilly's bow legs | ✅ | — | **her design** (bones track her own mesh to 2.5°). We are NOT imposing a pose |
| SkinTokens vs UniRig | ✅ | — | SkinTokens does both; **UniRig is dead** |

## PERF (owner's roadmap step 4 — **do not start until the above is green**)

| thread | status | note |
|---|---|---|
| HQ baseline measured | ✅ | **4498s total.** Refine **3339s = 74.3%**. VRAM peak **10751 MiB = 87.5%** |
| Refine optimisation | ⬜ | **the whole story** — 3/4 of runtime + owns the VRAM ceiling |
| `mc_stride` lever | 🔍 | stride 3 → −63.6% verts; volume inflates = quality tradeoff → **owner judges** |
| 72s volume bake runs **only for UVs** | 🔍 | when `--tex-project`; projection core = 0.16s |
| flux2 173s/step | 🚫 | **OUT OF SCOPE** (owner, 2026-07-17) |

## ⬜ "COMPLETE THE PORT" — the original mini-prompt's glue gaps, LARGELY UNTOUCHED

| thread | status | note |
|---|---|---|
| Inline `obj_decimate` | ⬜ | forks a binary + OBJ round-trip; meshoptimizer already linked |
| quadwild in-process | ⬜ | `std::system()` shell-out + OBJ round-trip |
| Retire `shootout/run_pipeline.sh` | ⬜ | legacy driver still shells to docker/python; supersede it |
| Validate `--part-retopo` e2e | ⬜ | P3-SAM GPU heads built, never exercised |

---
## HOW MUCH NEEDS A HUMAN? (the honest audit)

**The core API — `images → clean textured rigged model` — is FULLY AUTOMATIC today.** No per-model
tuning. The three "decisions" below are **one-time project settings**, not per-asset work:
1. **tex-DiT: proj vs cross** — pending the A/B. Then baked in.
2. **`REMESH_CLOSE_R`: 0 or 3** — pending the eye-test. Then baked in.
3. **`--tex-snap-volume` / bake mode** — both modes are defective; a real fix is outstanding.

**By design, not a gap:** the owner supplies **front+back images** (that IS the contract), and the
owner **judges quality on the eye-test**.

**Genuinely manual / missing:**
- **ALL animation** (🚫) — retarget, clips, Mixamo naming. Python, hand-run, not in the pipeline. If
  animated output is ever a product requirement, this is **net-new work**, not a wiring job.
- **α base-pose per rig family** (🖐️) — a human judgement, unavoidable in principle (both rigs' quirks
  are real; only a human knows which pose was meant as "neutral"). Industry tools ask too.
- **A back view for an existing front-only subject** — was flux2-generated; **flux2 is now out of
  scope**, so the back must come from the owner. `back_v1.png` is on disk for the soldier.

---
## SUGGESTED ORDER (owner: *"anything without a tick we should probably fix before performance"*)

1. **Land the two texture A/Bs** (front-align + bg-reject) — running now.
2. **Tex-DiT A/B** (`weights_gguf_f16`, both sides) — may be a **revert**, may invalidate other work.
3. **GPU-test `REMESH_CLOSE_R=3`** + eye-test — blob or body?
4. **Explain the 668k-vs-165k refine anomaly** — gates any refine perf work.
5. **Fix the bake fraying** (5.13%) — the one texture defect with no fix at all.
6. **Fix the winding at source** in `usr::refine` — cheap now, expensive when it bites.
7. **Close the 4 glue gaps** — this is what "complete the port" means.
8. **THEN perf**: the refine (74.3%), `mc_stride`, the UV-only bake, MoGe's OOM.
