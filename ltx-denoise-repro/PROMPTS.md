# LTX-2.3 prompting + the pain-point scenario set

Sources: the workflow author's own in-graph "LTX 2.3 Prompt Enhancer v2.1" note (`workflows/` ver1),
the official LTX-2.3 prompt guide, and community guides (links at bottom). They all agree.

## Cheat-sheet — how LTX-2.3 wants to be prompted
- **One flowing paragraph. Present tense. 4–8 sentences.** No lists, no line breaks, no fragments.
- **Order:** (1) establish the shot/framing + lens, (2) set the scene — lighting, colour palette, surface
  textures, atmosphere, (3) describe the action as ONE sequence flowing start→end, (4) define the
  character — age, hair, clothing, distinguishing details, (5) camera relationship to subject, (6) audio.
- **Fill the duration.** A short prompt on a 4-second clip makes the model *rush* the action → jitter/mush.
  Our clips are ~4 s (193f@48fps) — the prompt must give enough described motion to occupy that time.
- **One primary motion focus at a time.** If the *subject* does a strong action, keep the *camera* simple
  (locked or slow). If the *camera* makes a big move, simplify the subject. Never both chaotically.
- **Complete big camera moves before precise facial acting.** Let the face resolve while the camera is calm.
- **Emotion through visible behaviour** — gaze, posture, breath, gesture — not abstract labels ("happy").
- **Cinematography terms work**: `35mm lens`, `shallow depth of field`, `tracking shot`, `low angle`,
  `golden hour`, `locked-off`, `rim light`. Match **detail density to shot scale** (wide → describe more
  environment; close → describe the face).
- **Avoid**: readable text/logos as focal points, many simultaneous actions, conflicting lighting,
  impossible physics, contradictory camera instructions.
- **i2v**: the image defines the scene — write **motion-only**: preserve identity/clothing/pose, animate
  plausibly from the visible starting pose, don't redesign. Describe the transition *from stillness*.

## Why this targets the "distant-character poison"
Distant/small faces go to mush partly because a thin prompt lets the model rush and improvise the tiny
figure. The fixes: (a) make the figure's **gross motion explicit and sequential** so the model has a clear
path, (b) **simplify the camera** so all the model's budget goes to the subject, (c) for the walk-toward
case, describe the **distant→resolve arc** so the model knows the face should sharpen as it nears.

---

## Scenario set (the three pain points), each t2v + i2v
Hold ONE scenario fixed across the whole recipe ablation (seed 42). **Recommended first scenario = #3
(static camera, distant crossing)** — the purest "distant small moving figure" poison. Then #1, then #2.

### 1 — Person walking toward camera (distant → face fills frame)
**t2v:**
> Wide cinematic shot on a rain-slicked city street at night, shot on a 35mm lens with a shallow depth of field. Neon signage in magenta and cyan reflects across the wet asphalt while a faint mist softens the distant traffic lights. A man in his early thirties in a dark wool coat over a grey hoodie starts far down the block as a small silhouette and walks steadily toward the camera, his stride even and unhurried, hands loose at his sides. As he closes the distance his features resolve into focus — a short beard, tired eyes, breath fogging faintly in the cold air — until his face nearly fills the frame. The camera holds a slow, locked eyeline and lets him come to it rather than moving to meet him. Ambient city sound of tyres hissing on wet road, a low neon hum, and footsteps growing louder.

**i2v** (from `flux_neon_seed7.png`, motion-only):
> The figure begins to walk steadily toward the camera, closing the distance with an even, unhurried stride, hands loose at his sides. As he approaches, his face resolves into sharper focus and his expression softens into a faint, tired half-smile, breath fogging in the cold air. The camera stays locked on his eyeline, holding still as he comes to it, while neon reflections slide across the wet street behind him. Ambient sound of tyres on wet asphalt and footsteps growing louder.

### 2 — Person dancing in front of a crowd (mid-distance subject, crowd behind)
**t2v:**
> Medium-wide shot at a crowded outdoor night concert, shot on a 50mm lens with warm stage light spilling from behind. A young woman with long dark hair in a fringed denim jacket dances at the centre of the frame, spinning on one heel, stepping side to side and throwing her arms up to the rhythm, her movements fluid and full-bodied. Behind her a dense crowd of silhouetted onlookers sways and claps, kept soft and slightly out of focus by the shallow depth of field so the eye stays on her. Amber and gold light rims her figure while cooler blues wash the background and dust motes drift through the beams. The camera holds an almost-static frame with only the faintest push-in, letting her motion carry the shot. Ambient sound of a live crowd, rhythmic music, and cheering.

**i2v** (needs a matching still — dancer + crowd; flux one or supply):
> The woman breaks into an energetic dance, spinning on one heel, stepping side to side and throwing her arms up to the beat, her movements fluid and full-bodied. The crowd behind her sways and claps, kept soft by the shallow depth of field. Warm stage light rims her as she moves and the camera holds an almost-static frame with the faintest push-in. Ambient live-crowd noise, rhythmic music, and cheering.

### 3 — Static camera, person crossing the footpath / road (distant, small in frame)  ★ run first
**t2v:**
> Locked-off wide shot of a pedestrian crossing on a busy daytime city street, shot on a 35mm lens at eye level from the far kerb; the camera does not move. Cool overcast daylight flattens the scene and wet pavement holds pale reflections of the surrounding buildings. A woman in a red raincoat and jeans walks briskly from the left of the frame across the zebra crossing to the right, her gait natural and purposeful, a canvas bag swinging at her shoulder. Behind her, blurred traffic waits at the light and a few other pedestrians drift at the edges of the frame. She stays at a middle distance, small in the wide composition, never dominating the frame, her figure crisp against the muted street. Ambient city sound of idling engines, distant chatter, and the rhythmic tap of her footsteps on the crossing.

**i2v** (needs a matching still — empty crossing / figure at the kerb; flux one or supply):
> The woman steps off the kerb and walks briskly across the zebra crossing from left to right, her gait natural and purposeful, a canvas bag swinging at her shoulder. The camera stays completely locked-off and does not move while blurred traffic waits behind her. She holds a middle distance, small and crisp in the wide frame. Ambient city sound of idling engines, distant chatter, and footsteps tapping on the crossing.

---
## Notes for the ablation
- **t2v** covers all three scenarios out of the box. **i2v** #1 uses `flux_neon_seed7.png`; #2/#3 need a
  matching start still — quickest is to flux one (separate GPU step) or drop one in `models/ltx2/_inputs/`.
- Keep the prompt **byte-identical across all recipe rows** of a scenario — only the model/flags change.
- If a scene still rushes/jitters, the first lever is *more described motion to fill the 4 s*, not more steps.

Sources: [LTX-2.3 prompt guide](https://ltx.io/model/model-blog/ltx-2-3-prompt-guide) ·
[LTX-2 audio-to-motion prompting](https://ltx.io/model/model-blog/prompting-guide-for-ltx-2) ·
[RunDiffusion LTX-2 guide](https://www.rundiffusion.com/ltx-2-prompt-guide) ·
[fal LTX i2v prompt guide](https://fal.ai/learn/devs/ltx-video-2-pro-image-to-video-prompt-guide)
