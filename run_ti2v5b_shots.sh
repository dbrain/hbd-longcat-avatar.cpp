#!/usr/bin/env bash
# Wan2.2-TI2V-5B Turbo — T2V eval on the SAME 8-shot LTX-2.3 stress set (run_musicvideo_fixed.sh).
# Pure t2v (NO init-img) — the real "one-stop-shop" test: faces, motion, camera, neon. Even a bad
# result is a useful quantified data point. Loops each shot through run_ti2v5b.sh (preset/LoRA/VRAM
# /mp4 plumbing reused). Builds a per-shot montage for the eye-test page.
#
# Usage:
#   PRESET=combo SHOTS_IDX=0 FR=49 ./run_ti2v5b_shots.sh        # CANARY: shot 0 only, measure VRAM/wall
#   PRESET=turbo SHOTS_IDX=0 FR=49 ./run_ti2v5b_shots.sh        # canary the other preset (A/B)
#   PRESET=combo FR=49 ./run_ti2v5b_shots.sh                    # all 8 shots, winner preset
#   PRESET=combo FR=49 SEEDS="42 7" ./run_ti2v5b_shots.sh       # multiple seeds per shot
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PRESET="${PRESET:-combo}"
FR="${FR:-49}"; W="${W:-1280}"; H="${H:-704}"; FPS="${FPS:-24}"
SEEDS="${SEEDS:-42}"
OUTROOT="$PWD/ti2v5b_out"; MONT="$OUTROOT/montage_${PRESET}_${W}x${H}_${FR}f"; mkdir -p "$MONT"

# Same 8 shots as run_musicvideo_fixed.sh ([0]=face close-up stress test).
SHOTS=(
"Tight close-up of a man with dark stubble singing into a vintage microphone, eyes half-closed, mouthing lyrics, sweat glistening, slight head bob. Shallow depth of field. Warm amber stage light, dark background, cinematic, high detail, sharp focus on the face."
"A vintage car rolls to a stop outside a neon-lit corner bar at dusk, headlights sweeping the wet asphalt, tyres easing to a halt. Slow tracking shot alongside the car. Warm amber and magenta neon reflecting on the damp street, cinematic, volumetric light, shallow depth of field, high detail."
"A man swings the car door open, steps out and sings toward the camera, gesturing to the beat, the door swinging shut behind him. Handheld medium shot, slow push-in. Warm amber neon glow from the bar sign, rain-slick reflective street, cinematic, volumetric light, high detail."
"A man pushes open the door of a dim empty bar and walks inside past rows of glowing bottles, dust drifting in a shaft of light. Camera dollies in behind him through the doorway. Moody warm tungsten light, deep shadows, cinematic, volumetric light, high detail."
"A man sits on a stool at an empty bar counter singing energetically, shoulders bouncing, tapping the counter to the beat. Static medium shot. Warm tungsten light over the bar, glowing bottles behind him, cinematic, shallow depth of field, high detail."
"Medium close-up of a woman with curly hair singing backup, swaying side to side, snapping her fingers, smiling. Soft rim light, blurred neon bokeh behind. Cinematic, shallow depth of field, high detail."
"A drummer plays an energetic beat in a smoky bar, arms blurring with motion, cymbals flashing under a spotlight. Dynamic medium shot. Moody warm light, volumetric haze, cinematic, high detail."
"A man walks down a rain-slick neon street at night singing to the camera, reflections shimmering, breath visible in the cold air. Slow steadicam tracking shot. Saturated cyan and magenta neon, cinematic, volumetric light, shallow depth of field, high detail."
)

# Which shots to run (default all; SHOTS_IDX="0" or "0 2 5" for a subset/canary)
IDXS="${SHOTS_IDX:-$(seq 0 $((${#SHOTS[@]}-1)))}"

echo "== TI2V-5B T2V eval: preset=$PRESET ${W}x${H} ${FR}f seeds='$SEEDS' shots=[$IDXS] =="
for si in $IDXS; do
  for seed in $SEEDS; do
    LBL="shot${si}_${PRESET}_seed${seed}_${W}x${H}_${FR}f"
    PRESET="$PRESET" MODE=t2v PROMPT="${SHOTS[$si]}" SEED="$seed" \
      W="$W" H="$H" FR="$FR" FPS="$FPS" LABEL="$LBL" \
      ./run_ti2v5b.sh
    # collect the per-shot mp4 into the montage dir
    [ -f "$OUTROOT/$LBL.mp4" ] && cp "$OUTROOT/$LBL.mp4" "$MONT/$LBL.mp4"
  done
done
echo "== per-shot clips in $MONT =="
ls -la "$MONT"/*.mp4 2>/dev/null
