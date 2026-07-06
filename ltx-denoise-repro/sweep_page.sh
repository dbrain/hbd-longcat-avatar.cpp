#!/usr/bin/env bash
# Regenerates the model-sweep eye-test page from the manifest + clips. Grid: scenario rows x model columns.
set -u
D=/home/dbrain/dev/longcat-avatar-wan22/perf_out/ltx_denoise
C=$D/clips; MAN=$D/sweep_manifest.tsv
meta(){ # $1=modeltag $2=scenario -> "wall | peakVRAM" or ""
  grep -P "^$1\ts$2\t" "$MAN" 2>/dev/null | head -1 | awk -F'\t' '{print $3" · "$4}'
}
cell(){ # $1=clipfile $2=metastr $3=label
  if [ -f "$C/$1" ]; then
    echo "<figure><video src=\"clips/$1\" controls loop playsinline muted></video><figcaption><b>$3</b><br><span class=m>$2</span></figcaption></figure>"
  else
    echo "<figure class=pend><div class=ph>pending</div><figcaption><b>$3</b></figcaption></figure>"
  fi
}
{
cat <<'H'
<!doctype html><meta charset=utf-8><title>LTX-2.3 quant sweep — 1920×1088</title>
<style>
 body{background:#0d0d0d;color:#e2e2e2;font:13px/1.5 system-ui,sans-serif;margin:0;padding:18px 20px;max-width:2200px}
 h1{font-size:21px;margin:0 0 3px} .sub{color:#8a8a8a;font-size:12.5px;max-width:1400px;margin:0 0 4px}
 h2{font-size:15px;color:#9cf;margin:22px 0 2px;border-bottom:1px solid #282828;padding-bottom:4px}
 .row{display:flex;gap:10px;flex-wrap:nowrap;overflow-x:auto;margin:8px 0;padding-bottom:8px}
 figure{margin:0;background:#000;padding:6px;border:1px solid #2b2b2b;border-radius:7px;flex:0 0 auto;width:340px}
 figure.ref{border-color:#3a6ea5} figure.win{border-color:#3a8a3a} figure.pend{border-color:#333;opacity:.5}
 video{width:100%;display:block;border-radius:4px;background:#000}
 .ph{height:190px;display:flex;align-items:center;justify-content:center;color:#555;font-size:13px}
 figcaption{margin-top:5px;font-size:12px;color:#ccc} .m{color:#f5b942;font-size:11.5px}
 .leg{color:#888;font-size:12px}
</style>
<h1>LTX-2.3 quant sweep — all dev+distill@0.5, 1920×1088</h1>
<p class=sub>Same recipe/seed across all weights. Columns: <b>comfy-fp8</b> (reference) → <b>our-fp8</b> (is-our-code-sane) → <b>imatrix-fp4</b> (the quality bet) → <b>q4km</b> (K-quant, timing/vram) → <b>old-nvfp4</b> (RTN baseline). <span class=leg>Timing · peak-VRAM under each. Play in motion.</span></p>
H
declare -A NAME=( [comfy]="COMFY fp8 (ref)" [fp8]="OUR fp8" [imatrix]="OUR imatrix-fp4" [q4km]="Q4_K_M" [oldfp4]="old nvfp4 (RTN)" )
for s in 1 2 3; do
  case $s in 1) t="S1 — walk toward camera";; 2) t="S2 — dance in crowd";; 3) t="S3 — distant crossing";; esac
  echo "<h2>$t</h2><div class=row>"
  # comfy reference (fixed filenames)
  cell "COMFY_s${s}_t2v_2stage.mp4" "1920×1088 · comfy" "COMFY fp8 (ref)" | sed 's/<figure>/<figure class=ref>/'
  cell "SWEEP_fp8_s${s}.mp4"     "$(meta fp8 $s)"     "OUR fp8"
  cell "SWEEP_imatrix_s${s}.mp4" "$(meta imatrix $s)" "OUR imatrix-fp4" | sed 's/<figure>/<figure class=win>/'
  cell "SWEEP_q4km_s${s}.mp4"    "$(meta q4km $s)"    "Q4_K_M"
  cell "SWEEP_oldfp4_s${s}.mp4"  "$(meta oldfp4 $s)"  "old nvfp4 (RTN)"
  echo "</div>"
done
cat <<'SEED'
<h2>🎲 Comfy s3 seed variants — find the composition closest to ours (for a fair A/B)</h2>
<p class=sub>6 seeds of comfy dev-fp8 s3. Comfy is deterministic + cleaner but always a different scene than our nvfp4 — pick the one whose framing/distance is closest to <b>our nvfp4/imatrix</b> (last two) so you can judge quality on a like-for-like composition.</p>
<div class=row>
SEED
for sd in 7 13 101 202 303 404; do
  echo " <figure class=ref><video src=\"clips/COMFY_s3_seed${sd}.mp4\" controls loop playsinline muted></video><figcaption>comfy seed $sd</figcaption></figure>"
done
echo " <figure><video src=\"clips/SWEEP_oldfp4_s3.mp4\" controls loop playsinline muted></video><figcaption><b>OURS nvfp4 (RTN)</b></figcaption></figure>"
echo " <figure class=win><video src=\"clips/SWEEP_imatrix_s3.mp4\" controls loop playsinline muted></video><figcaption><b>OURS imatrix-fp4</b></figcaption></figure>"
echo "</div>"
cat <<'NVCODE'
<h2>🔬 Same nvfp4 weights, comfy pipeline vs ours — is the "dit dotty in motion" our code?</h2>
<p class=sub>Identical nvfp4 weights fed through <b>comfy's</b> pipeline (left, seeds 101/202/303) vs <b>ours</b> (right). Comfy is a different composition per seed but same-weights — if comfy-nvfp4 is clean in motion and ours is dotty, the artifact is our code path, not the 4-bit weights. Seed 101 = owner's best-match composition.</p>
<div class=row>
NVCODE
for sd in 101 202 303; do
  cl="COMFY_nvfp4_s3_seed${sd}.mp4"
  if [ -f "$C/$cl" ]; then
    echo " <figure class=ref><video src=\"clips/$cl\" controls loop playsinline muted></video><figcaption>comfy nvfp4 seed $sd</figcaption></figure>"
  else
    echo " <figure class=pend><div class=ph>pending</div><figcaption>comfy nvfp4 seed $sd</figcaption></figure>"
  fi
done
echo " <figure><video src=\"clips/SWEEP_oldfp4_s3.mp4\" controls loop playsinline muted></video><figcaption><b>OURS nvfp4 (RTN)</b></figcaption></figure>"
echo " <figure class=win><video src=\"clips/SWEEP_imatrix_s3.mp4\" controls loop playsinline muted></video><figcaption><b>OURS imatrix-fp4</b></figcaption></figure>"
echo "</div>"
cat <<'VAEAB'
<h2>🎬 VAE A/B — temporal-tiling vs comfy-style feathered SPATIAL tiling (imatrix, seed 42, 1920×1088×97f)</h2>
<p class=sub>Both clean in a STILL; the difference is IN MOTION. <b>temporal</b> = our old default (97f chopped into 11 time-tiles → ghosting on crowds/pans). <b>spatial</b> = comfy-style feathered spatial tiles, ALL frames per tile, NO temporal chop (seam-free). <b>2×2 + F16</b> is the candidate lock-in: fastest tiling (316s) + offload eviction (VAE freed during sampling, DiT freed during decode) + conv3d 32×16 → peak <b>11670MB, under the 11.5GB cap</b>. Watch background crowds / the far cars for phase/ghost, and compare 2×2 vs 4×4 for any spatial-tile seam.</p>
<div class=row>
 <figure><video src="clips/AB_imatrix_matched_s3_42.mp4" controls loop playsinline muted></video><figcaption><b>temporal tiling</b> (old)</figcaption></figure>
 <figure class=win style="border-color:#3a8a3a;border-width:2px"><video src="clips/AB_imatrix_spatial_2x2_ov4_s3_42.mp4" controls loop playsinline muted></video><figcaption>✅ <b>LOCKED — spatial 2×2 ov4</b> (283s · 11164MB · seam-free)</figcaption></figure>
 <figure class=win><video src="clips/AB_imatrix_spatial_2x2_f16_s3_42.mp4" controls loop playsinline muted></video><figcaption><b>spatial 2×2 conv32×16 ov6</b> (316s · 11670MB · fallback)</figcaption></figure>
 <figure class=win><video src="clips/AB_imatrix_spatial_f16_s3_42.mp4" controls loop playsinline muted></video><figcaption><b>spatial 4×4 + F16</b> (more tiles)</figcaption></figure>
 <figure class=win><video src="clips/AB_imatrix_spatial_s3_42.mp4" controls loop playsinline muted></video><figcaption><b>spatial 4×4</b> F32 (older)</figcaption></figure>
 <figure class=ref><video src="clips/COMFY_s3_seed101.mp4" controls loop playsinline muted></video><figcaption>comfy (ref)</figcaption></figure>
</div>
<p class=sub>👁️ <b>Where the 2×2 seams would be</b> (red = tile boundaries): a vertical line down frame-centre (x≈960) and a horizontal line across frame-centre (y≈544). In the <b>ov4</b> clip, watch for a faint brightness step / texture discontinuity along these lines — worst in smooth areas (sky, wet pavement reflections) and as the woman / background cars cross the centre. Clean = no visible line there.</p>
<div class=row>
 <figure style="width:520px"><img src="clips/SEAMGUIDE_2x2_ov4.png" style="width:100%;border-radius:4px"><figcaption>ov4 frame — <b>red lines = the two 2×2 tile edges</b> (look here)</figcaption></figure>
 <figure style="width:520px"><img src="clips/SEAMGUIDE_2x2_ov4_clean.png" style="width:100%;border-radius:4px"><figcaption>same frame, unmarked — is the centre cross visible?</figcaption></figure>
</div>
VAEAB
cat <<'AB'
<h2>⚖️ CLEAN A/B — fp8 vs imatrix, IDENTICAL pipeline (same script/settings/seed 42, only the weights differ)</h2>
<p class=sub>Both via run_parity, RES=parity 1920×1088×97f, TBF=3 VWT=16 VHT=8, seed 42. Same VAE tiling. <b>imatrix-fp4 is crisp; our fp8 is dotty/ghosty</b> (26MB vs 8MB h264 = the dot-noise). So the fuzz is NOT the tiling — it's fp8-specific: our GEMM quantizes ACTIVATIONS to e4m3 too, noisier than nvfp4's two-level. Higher-precision-weights via this fp8 path = WORSE, not better. Play in motion.</p>
<div class=row>
 <figure class=win><video src="clips/AB_imatrix_matched_s3_42.mp4" controls loop playsinline muted></video><figcaption><b>imatrix-fp4</b> (clean)</figcaption></figure>
 <figure><video src="clips/AB_fp8_matched_s3_42.mp4" controls loop playsinline muted></video><figcaption><b>OUR fp8</b> per-tensor (dotty)</figcaption></figure>
 <figure><video src="clips/AB_mxfp8_matched_s3_42.mp4" controls loop playsinline muted></video><figcaption><b>OUR fp8 MXFP8</b> (block-scaled, still dotty)</figcaption></figure>
 <figure class=ref><video src="clips/COMFY_s3_seed101.mp4" controls loop playsinline muted></video><figcaption>comfy fp8 (ref)</figcaption></figure>
</div>
AB
cat <<'FP8'
<h2>🎯 Native fp8 seeds (⚠ over-tiled TBO=1 — mesh is my VAE settings, ignore vs the clean A/B above)</h2>
<p class=sub>Our own <b>native e4m3 fp8</b> (dev+distill@0.5, NaN-code + workspace-align bugs fixed) at 1920×1088×97f, three seeds. fp8 ≈ high precision down the SAME code path as nvfp4. If fp8 is clean in motion and <b>imatrix-fp4</b> (right) is mushy → the mush is the 4-bit weights; if fp8 is ALSO mushy → it's our pipeline. <b>comfy-fp8</b> (far right) = external reference.</p>
<div class=row>
FP8
for sd in 42 101 202; do
  cl="SWEEP_fp8nanfix_s3_seed${sd}.mp4"
  if [ -f "$C/$cl" ]; then
    echo " <figure class=win><video src=\"clips/$cl\" controls loop playsinline muted></video><figcaption><b>OUR fp8</b> seed $sd</figcaption></figure>"
  else
    echo " <figure class=pend><div class=ph>pending</div><figcaption>OUR fp8 seed $sd</figcaption></figure>"
  fi
done
echo " <figure><video src=\"clips/SWEEP_imatrix_s3.mp4\" controls loop playsinline muted></video><figcaption><b>OURS imatrix-fp4</b></figcaption></figure>"
echo " <figure class=ref><video src=\"clips/COMFY_s3_seed101.mp4\" controls loop playsinline muted></video><figcaption>comfy fp8 (s101)</figcaption></figure>"
echo "</div>"
echo "<h2>Timing / VRAM table</h2><pre style='color:#bbb;font-size:12px'>"
printf "%-10s %-4s %-8s %-10s %-12s %s\n" MODEL SCN WALL PEAK-VRAM RES STATUS
[ -f "$MAN" ] && sort "$MAN" | awk -F'\t' '{printf "%-10s %-4s %-8s %-10s %-12s %s\n",$1,$2,$3,$4,$5,$6}'
echo "</pre>"
} > "$D/sweep.html"
echo "regenerated sweep.html"
