#!/usr/bin/env bash
# Regenerates fidelity.html from the manifest — every experiment, labeled, next to the comfy target.
set -u
D=/home/dbrain/dev/longcat-avatar-wan22/perf_out/ltx_denoise
MAN=$D/fid_manifest.tsv
PSNR=$(cat $D/.fid_noisefloor 2>/dev/null || echo "?")
{
cat <<'HEAD'
<!doctype html><meta charset=utf-8><title>LTX nvfp4 fidelity hunt</title>
<style>
 body{background:#0d0d0d;color:#e2e2e2;font:14px/1.5 system-ui,sans-serif;margin:0;padding:20px 24px;max-width:1700px}
 h1{font-size:22px;margin:0 0 4px} h2{font-size:16px;color:#9cf;margin:26px 0 4px;border-bottom:1px solid #282828;padding-bottom:5px}
 .sub{color:#8a8a8a;font-size:13px;max-width:1150px}
 .row{display:flex;gap:14px;flex-wrap:wrap;align-items:flex-start;margin:12px 0}
 figure{margin:0;background:#000;padding:8px;border:1px solid #2b2b2b;border-radius:8px;width:440px}
 figure.target{border-color:#3a6ea5} figure.noise{border-color:#7a6a2a}
 video{width:100%;display:block;border-radius:5px;background:#000}
 figcaption{margin-top:7px;font-size:12.5px;color:#cfcfcf}
 .lbl{font-weight:700;color:#9d9} .cfg{color:#f5b942;font-family:monospace;font-size:11.5px} .meta{color:#888;font-size:11.5px}
 .warn{color:#f88} code{background:#1e1e1e;padding:1px 5px;border-radius:3px}
</style>
<h1>LTX-2.3 nvfp4 — fidelity hunt (you judge, one variable per clip)</h1>
<p class=sub>Everything renders at <b>1280×704, seed 42, 24fps</b>. Each experiment changes <b>ONE</b> thing vs the baseline. <b>Play in motion.</b> Note: our pipeline may be nondeterministic — the <b>noise floor</b> below shows how much two <i>identical</i> runs differ, so you can tell a real change from run-to-run luck. This path has fooled us before; trust your eyes, not the labels.</p>

<h2>🔑 THE FINDING — nvfp4 is reproducible; OUR engine isn't</h2>
<p class=sub>The reason this hunt kept failing: <b>our nvfp4 renders aren't deterministic</b>, so every A/B was measuring noise. Proof below — same seed, run twice, each pair:</p>
<div class=row>
 <figure class=target><video src="clips/COMFY_nvfp4_s3.mp4" controls loop playsinline muted></video><figcaption><span class=lbl style="color:#9cf">COMFY nvfp4</span> · two seed-42 runs = <b style="color:#7d7">PIXEL-IDENTICAL (PSNR ∞)</b><br><span class=meta>reference impl is deterministic → nvfp4 CAN be reproducible</span></figcaption></figure>
 <figure class=noise><video src="clips/FID_noise_A.mp4" controls loop playsinline muted></video><figcaption><span class=lbl>OURS nvfp4 · run A</span></figcaption></figure>
 <figure class=noise><video src="clips/FID_noise_B.mp4" controls loop playsinline muted></video><figcaption><span class=lbl>OURS nvfp4 · run B</span> · identical config to A → <b class=warn>only __PSNR__ dB apart (different scene!)</b></figcaption></figure>
</div>
<p class=sub><b>Conclusion:</b> the 23 dB wobble is a <b>bug in our kernels</b>, not "NVIDIA fluff" and not inherent to nvfp4 — comfy proves determinism is achievable on this exact GPU. Fix our determinism → the fidelity hunt becomes possible. Everything below this line is on hold until then (those single clips are one draw from a noisy process).</p>

<h2>🎯 Quant quality control — comfy dev-fp8 vs comfy-nvfp4 (both deterministic)</h2>
<p class=sub>Since comfy is deterministic, THIS is a valid quant comparison: does 4-bit nvfp4 lose quality vs fp8, holding the pipeline fixed?</p>
<div class=row>
 <figure class=target><video src="clips/COMFY_s3_t2v_2stage.mp4" controls loop playsinline muted></video><figcaption><span class=lbl style="color:#9cf">COMFY dev-fp8</span> · 1920×1088</figcaption></figure>
 <figure class=target><video src="clips/COMFY_nvfp4_s3.mp4" controls loop playsinline muted></video><figcaption><span class=lbl style="color:#9cf">COMFY dev-nvfp4</span> · same recipe, 4-bit weights</figcaption></figure>
</div>

<h2>🧪 Mixed-precision validation (comfy, distilled-1.1, seed 42, same recipe)</h2>
<p class=sub>The decision test: does keeping the sensitive attn/FF Linears at <b>BF16</b> (nvfp4mixed) visibly beat uniform 4-bit? If yes, it's buildable in our engine (no new type). <b>Judge in motion</b> — same seed but precision diverges the scene, so don't trust a single frame. Both deterministic (real single-variable).</p>
<div class=row>
 <figure class=noise><video src="clips/COMFY_distilled_uniform.mp4" controls loop playsinline muted></video><figcaption><span class=lbl>UNIFORM nvfp4</span> · distilled-1.1, all 4-bit · 396s</figcaption></figure>
 <figure class=target><video src="clips/COMFY_distilled_mixed.mp4" controls loop playsinline muted></video><figcaption><span class=lbl style="color:#7d7">nvfp4MIXED</span> · +BF16 on sensitive attn/FF Linears · 390s</figcaption></figure>
</div>

<h2>🔬 Experiments — one variable each</h2>
<div class=row>
HEAD
if [ -f "$MAN" ]; then
  while IFS=$'\t' read -r label desc cfg wall res; do
    [ "$label" = noise_A ] && continue; [ "$label" = noise_B ] && continue
    echo " <figure><video src=\"clips/FID_$label.mp4\" controls loop playsinline muted></video><figcaption><span class=lbl>$desc</span><br><span class=cfg>$cfg</span><br><span class=meta>$wall · $res</span></figcaption></figure>"
  done < "$MAN"
fi
cat <<'FOOT'
</div>
<p class=sub>Baseline = <code>euler · F16 decode · nvfp4-dev050 · VWT=4</code>. Older parity summary: <a href="compare.html" style="color:#9cf">compare.html</a>.</p>
FOOT
} > "$D/fidelity.html"
sed -i "s/__PSNR__/$PSNR/" "$D/fidelity.html"
echo "regenerated fidelity.html ($([ -f "$MAN" ] && wc -l < "$MAN" || echo 0) experiments)"
