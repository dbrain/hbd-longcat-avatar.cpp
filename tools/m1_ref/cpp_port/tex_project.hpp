// tex_project.hpp — TEXTURE BY PROJECTING THE REAL IMAGES onto the mesh's UV atlas.
//
// The volume bake (texatlas::bake) regenerates appearance from TRELLIS-2's ~64³ PBR latent: correct
// colours, ZERO detail (no eyes, no buttons) — that softness is intrinsic to the model, not a setting.
// The one thing that beats every generative painter on the FRONT is the REAL input image. This bake
// re-samples the source matte (and any number of extra generated views) through the exact pixal3d camera
// and writes the projected pixels into the atlas that texatlas::bake already unwrapped.
//
// Self-contained, mirrors normal_bake.hpp's contract: reuses the shared helpers (texatlas::inpaint_telea,
// texatlas::dilate_background, imgio::load_rgb01) and does NOT touch texatlas::bake() — that function is
// shared with the production pixal3d --tex path and must stay untouched.
//
// USAGE: run texatlas::bake first (gives UVs + metal_rough + a volume base_color), then call
// texproj::project_onto(bt, cfg). It OVERWRITES bt.base_color's RGB in place; bt.metal_rough is KEPT
// from the volume bake and so is the alpha byte. (See "THE METAL_ROUGH CAVEAT" below — keeping the
// volume metal_rough is a MEASURED quality limit on metallic trim, not a free pass.)
//
// PIPELINE
//   1. UV-raster bt.verts/normals/uvs/faces -> per atlas texel: 3D position, interpolated normal, coverage
//      (same edge-function/inv-winding/bbox/supersample structure as tex_atlas.hpp:917-953).
//   2. Depth prepass: rasterize EVERY mesh triangle through the camera into an image-space z-buffer
//      (perspective-correct, min-depth). Once per VIEW.
//   3. Per-view auto-align: fit the view image's subject bbox onto the mesh's silhouette bbox for that
//      view (uniform scale + translate). THE FRONT IS FITTED TOO — see BUG 1 below; the old claim that
//      "the front IS the matte, so it needs no fit" is FALSE and measurably so.
//   4. Per texel: project -> (u,v,depth); visible iff not occluded by the z-buffer AND facing the camera.
//      THE BUG THIS FIXES: the prototype gated ONLY on normal.z>0 with NO depth test, so the face painted
//      straight through onto the bearskin hat ("double face in the hat") and onto tucked-behind surfaces.
//   4b. SUBJECT-MASK REJECT (BUG 2 below): a texel whose sample lands outside the view image's eroded
//      subject mask is NOT painted by that view — it becomes a hole for the 3D fill to serve.
//   5. Confidence = smoothstep(nz_lo, nz_hi, facing) — a grazing-angle ramp. It was ALSO supposed to stop
//      grazing texels sampling background, and it demonstrably does not — that is BUG 2.
//   6. Blend all views by confidence IN LINEAR LIGHT.
//   7. 3D-AWARE HOLE FILL (see below), then the atlas-space Telea/dilate for whatever is left.
//
// COLOUR SPACE: the source PNG is sRGB-encoded, glTF baseColorTexture is implicitly sRGB, and
// bt.base_color holds sRGB BYTES (tex_atlas.hpp:1048-1068 applies the sRGB OETF at pack time). We write
// bt.base_color DIRECTLY, bypassing that pack, so a straight sRGB byte copy is the correct, lossless
// answer — no decode, no re-encode. (The old prototype DID decode to linear, but only because it wrote
// glTF COLOR_0, which is linear. That does not apply here.) The ONE exception is the weighted view blend
// and the hole fills, which are only physically meaningful in linear light: so we decode to linear,
// blend/fill there, and re-encode with the sRGB OETF on the way to the byte. Single-source texels
// round-trip decode->encode to within <1/255 of the original sample.
//
// ---------------------------------------------------------------------------------------------
// THE 3D-AWARE HOLE FILL (TEXPROJ_FILL_3D, default ON)
//
// ~50% of covered texels are seen by NO view (soldier: front 25.7% / back 23.9% / holes 50.4%). Filling
// those with texatlas::inpaint_telea + dilate_background is an ATLAS-SPACE operation, and the precluster
// atlas packs ~14k tiny charts next to each other with no 3D relationship whatsoever. So a hole took the
// colour of whatever chart happened to be packed nearby in UV: red/black/skin mashed together = the
// owner's "camo pattern" on the side views, top of head, bottom of feet and under the jacket, and the
// "black lines" down the sides (a boot/hat chart bleeding into a torso chart).
//
// The fix: for every covered-but-unpainted texel, find the k nearest PAINTED texels BY 3D POSITION and
// blend them inverse-distance-weighted. Two properties make this work where atlas-Telea cannot:
//   - NEAREST IN 3D, not in UV: a hole on the side of the torso pulls from the torso texels just over the
//     silhouette (which the grazing-angle ramp painted at facing just above nz_lo), not from whichever
//     chart shares its atlas neighbourhood. The silhouette rim IS the correct colour to continue.
//   - A NORMAL-AGREEMENT GATE: dot(n_hole, n_painted) > TEXPROJ_FILL_MINDOT. Without it a hole on the
//     INSIDE of the jacket would steal the OUTSIDE surface's colour straight across the thin shell (they
//     are ~1mm apart in 3D but opposite-facing) — which would look exactly like the bug we are fixing.
//     Both normals come from the same mesh, so the inward/outward convention cancels in the dot product
//     and this gate needs no nsign.
// k-nearest (not single-nearest) because a single nearest paints visible Voronoi facets — and "blotchy
// misc colouring" is the complaint, so we must not trade camo for facets.
//
// Whatever the gate + ring cap cannot serve falls through to the ORIGINAL Telea + dilate. That is the
// right fallback for real UV GUTTERS (mask=0 texels between charts), which are a genuinely different
// problem: they exist so a renderer's bilinear filter has something to bleed from, and they are correctly
// solved in atlas space. Gutter handling is untouched.
//
// ---------------------------------------------------------------------------------------------
// BUG 1 — THE FRONT *DOES* NEED THE SILHOUETTE FIT (TEXPROJ_FRONT_ALIGN, default ON)
//
// This file used to special-case the front to skip the auto-align, on the claim that the front image "IS
// the matte the mesh was built from, verified pixel-perfect". THAT CLAIM IS FALSE. Measured with
// texproj_probe (this file's own Cam + raster_depth + subject_mask, soldier @ 1060x1060):
//     matte subject bbox (soldier_matte.png, nonzero):  w=0.5547  h=0.9094
//     refined.glb silhouette (z-buffer, yaw=0):         w=0.5491  h=0.9009  -> w-ratio 1.0103  h-ratio 1.0094
//     coarse.glb  silhouette (z-buffer, yaw=0):         w=0.5519  h=0.9019  -> w-ratio 1.0051  h-ratio 1.0084
//
// READ THOSE TWO MESH ROWS CAREFULLY — they refute the obvious causal story (which the handoff spec, and an
// earlier version of this comment, both told). The blame does NOT land mainly on the UltraShape refine or
// on image_to_rig.cpp's bbox_canon_onto():
//   - The COARSE mesh — pixal3d's own geometry, the thing built directly from this matte — is ALREADY 0.5%
//     narrow and 0.8% SHORT of the matte's silhouette. Nearly ALL of the vertical error and half the
//     horizontal error predate refine entirely. It is the geometry model's silhouette fidelity, full stop,
//     and nothing in this file or upstream of it is going to "fix" that.
//   - The refine then adds ~0.5% more NARROWING (1.0051 -> 1.0103) and ~0.1% vertically. That asymmetry is
//     bbox_canon_onto's signature and confirms how it works: it pins the LONGEST axis (y), so refined and
//     coarse have bit-identical y extents ([-0.4219,0.4206] both) while x/z drift uncorrected.
// So: the mesh is not the matte's projection, for two independent reasons, and only ONE of them is even
// arguably a bug elsewhere. Either way this bake must fit the front, because it is the only stage that can.
//
// ~1% at 1060px is several pixels at the silhouette — the sort of thing that slides a gold button off the
// geometric bump that IS modelled underneath it, which is the owner's report ("they show as gold, they're
// just... a little off the bumps"). ⚠ BUT KNOW THE SIZE OF WHAT THIS BUYS BEFORE BELIEVING IT: the fitted
// transform (scale 1.0094, translate (-6.00,-4.49)px) moves the sample point by only
//     chest/buttons (530,403): (-1.01, -0.69) px      hand (318,583): (-3.01, +1.00) px
//     boot sole     (530,975): (-1.01, +4.70) px      centre          (530,530): (-1.01, +0.50) px
// i.e. ~1.2px at the chest on a ~25px button. It is a REAL correction in the right direction and it is free,
// but if the owner's perceived button offset is bigger than a pixel or two, THE REST IS NOT A GLOBAL 2D
// TRANSFORM — it is local geometry drift (the chest surface's own 3D position), and no bbox fit of any kind
// will reach it. Do not keep tuning this fit expecting the buttons to snap on.
//
// WHY min() IS STILL RIGHT HERE, with a measured number behind it: on the REFINED mesh the two axis ratios
// are w 1.0103 and h 1.0094 — they agree to 0.09%. A near-isotropic error is exactly what a uniform 2D
// similarity is for, so min() costs essentially nothing here (it picks 1.0094 vs aniso's 1.0103/1.0094) and
// it keeps the mapped box inside the subject bbox on BOTH axes — so a silhouette texel can never map out
// into the background, which is the whole reason min() was chosen for the back. Keep it.
// (TEXPROJ_ALIGN_ANISO=1 fits the axes independently: an A/B lever, NOT the default. It buys <0.1% here,
// and it makes the mapped box EQUAL the subject box, putting silhouette samples exactly on the outline
// where bilinear straddles — survivable only because TEXPROJ_BG_REJECT now backstops it. Note the ratios
// are NOT as isotropic on the coarse mesh (1.0051 vs 1.0084), so do not assume isotropy for other subjects.)
//
// ---------------------------------------------------------------------------------------------
// BUG 2 — BLACK STREAKS DOWN THE SIDES = GRAZING-ANGLE BACKGROUND SAMPLING (TEXPROJ_BG_REJECT, default ON)
//
// Owner: "left and right side of model still has black streaks (down ears -> end of hands)". That path IS
// the silhouette edge. A texel whose normal grazes the camera projects to within ~a pixel of the subject's
// outline, and sample_bilinear then blends in the matte's BLACK BACKGROUND. The texel is painted black; the
// 3D fill happily propagates that black along the side, because as far as it knows black is a real sample.
// The nz_lo=0.05 confidence ramp was supposed to prevent this and does not — a texel at facing=0.06 is
// "confident enough" to paint, and (being the only view that sees it) its weight CANCELS in the blend, so
// the ramp cannot attenuate a lone bad sample at all. Tuning nz_lo is the wrong lever: it trades black
// streaks for a bald silhouette.
//
// FIX: reject background samples outright. The source image knows exactly where the subject is, so build a
// per-view subject mask from the image itself, ERODE it by TEXPROJ_BG_ERODE px (default 2, in SOURCE-IMAGE
// pixels) so no bilinear tap can straddle the outline, and drop any texel sampling outside it. A dropped
// texel contributes ZERO confidence — exactly like failing the z-test — so it becomes a hole and the 3D
// fill paints it from a confident neighbour. That is the right owner for it: the same fill already fixed
// the soles and the hat.
//
// ⚠ THE LANDMINE (measured — do NOT undo this): the obvious subject detector "max(r,g,b) > 0.05" (the
// pre-existing TEXPROJ_BACK_BG_THRESH default) WOULD PUNCH A HOLE THROUGH THE BOOTS AND THE BEARSKIN. On
// soldier_matte.png the true background is EXACTLY (0,0,0) over 77.35% of the image, and:
//     nonzero (= true subject)                 254476 px
//     thresh 0.05 + hole-fill  -> subject      234480 px  =>  20016 SUBJECT px lost (7.9%), 86% of them at
//                                                             v>0.7 — i.e. THE BLACK BOOTS
//     thresh 1/255 + hole-fill -> subject      253494 px  =>   1048 px lost (0.4%) — every one of them a
//                                                             code==1 matte rim pixel, which the erosion
//                                                             removes anyway. (1442 px are code==1; 460 of
//                                                             them the hole-fill reclaims.)
// (Both rows reproduced independently in numpy and by texproj_probe against this file's own subject_mask —
//  they agree to the pixel. NOTE the 1/255 boundary is `mx <= thresh`, so it discards code 0 AND code 1.)
//
// PREDICTIONS for the first run with BG_REJECT on, so a surprise is recognisable as one:
//   - the erosion removes 10288 px, 4.06% of the front's subject PIXELS (r=1 -> 2.04%, r=3 -> 6.07%).
//   - the front's painted-texel count DROPS — expected and good, we are refusing to paint texels black.
//     Expect the rejected set to be a good deal larger than 4%: a grazing surface compresses many texels
//     into few pixels, and for facing below ~0.2 essentially the whole band sits within 2px of the outline.
//     Rough expectation ~5-15% of the texels that pass facing+depth; the run prints the real number.
//   - THE FALSIFIABLE PART: the rejected set's MEAN FACING must come out far below the kept set's. That is
//     the grazing signature and the entire basis of this diagnosis. If they are comparable, the reject is
//     firing in the surface interior and the mask or the fit is wrong — project_onto prints a WARN for
//     exactly that case, and you should believe it over this comment.
//   - holes rise by roughly what the front loses; the 3D fill should absorb nearly all of it (its
//     neighbours just inboard are painted and near-coplanar, so the mindot gate passes easily). Watch
//     n_telea: if the Telea fallback share jumps, the camo risk is back and BG_ERODE is too aggressive.
// So the front detector's threshold defaults to 1/255 (TEXPROJ_FRONT_BG_THRESH), not 0.05: a matte
// composited on exact black needs a threshold that only has to beat rounding, and anything higher starts
// eating the subject's own dark materials. Two more guards on top:
//   - HOLE FILL (TEXPROJ_BG_FILL_HOLES, default ON): "background" = the near-black region REACHABLE FROM
//     THE IMAGE BORDER. A dark region enclosed by subject (the bearskin's core: 5710 px at v 0.1-0.3 on the
//     soldier) is therefore subject by construction, whatever the threshold. This is what makes a
//     threshold safe against dark materials in the INTERIOR; it cannot save a dark region that touches the
//     silhouette (a boot sole), which is why the threshold itself must stay low.
//   - a per-view diagnostic line prints subject%, hole-fill recovery and the discarded-nonzero count, so a
//     view whose background is NOT near-black (a noisy generated view) is visible immediately instead of
//     silently fitting its bbox to the whole image.
// NON-FRONT views keep TEXPROJ_BACK_BG_THRESH=0.05 (unchanged, so this cannot regress the back's existing
// alignment) — but a generated back view of the SAME soldier has the SAME black boots, so it is very likely
// to want 0.004 too. That is an A/B for the owner, not a silent default change.
//
// ---------------------------------------------------------------------------------------------
// BUG 3 — THE FRONT/BACK SEAM. Owner: "the crossover front to back has 'defects' (colours dont blend
// perfectly etc.) ... be nice if it didn't feel like there was a seam".
//
// ⚠ FIRST, KILL THE OBVIOUS STORY, BECAUSE IT IS FALSE. The natural diagnosis — "front and back are two
// independently-generated images, they differ in exposure/WB, and the confidence-weighted mean smears
// them across the transition band" — is wrong twice over, and BOTH halves are measured:
//
//   (1) THERE IS NO TRANSITION BAND. Cam::facing reduces to +nz*nsign at yaw=0 and -nz*nsign at yaw=180,
//       so facing_back == -facing_front EXACTLY. A texel cannot have both confidences > 0. Measured on
//       the soldier: of 165296 refined.glb verts, the front paints 37673 and the back 30772 and BOTH
//       paint 0 — and on the real 4096^2 atlas, proj_seam.png's red and green channels overlap in
//       exactly 0 texels. So step 6's "blend all views by confidence" NEVER BLENDS front and back: every
//       texel is painted by exactly one view or by none. The weight cancels (acc/wsum = the sample), so
//       in the shipped front+back config TEXPROJ_NZ_HI is DEAD CODE — it cannot move a single pixel, and
//       TEXPROJ_NZ_LO does one thing only: it sets where painting stops and the hole band begins.
//       COROLLARY: you cannot fix this seam by reshaping the confidence ramp. There is nothing to ramp.
//
//   (2) THERE IS NO EXPOSURE/WB STEP TO CORRECT. Measured three ways, all agreeing:
//         - 398 verts where BOTH cameras have an unoccluded, in-subject sample of the SAME 3D point
//           (zero registration error by construction): mean front-back = R -0.4, G -3.3, B -3.7 sRGB
//           bytes, on a per-pair scatter of std 43/36/24. The mean step is ~0 to within +-4 bytes.
//         - on the real atlas, over the 131694 seam texels: |viewA-viewB| = 36.7/255 but the SIGNED bias
//           is only (-11.4, +2.1, +1.4) -> bias/|d| = 0.14.
//         - the bias is not even sign-stable across sampling schemes (per-vertex says R +5.0, per-texel
//           says R -11.4), which is what "no real global offset" looks like.
//       So ~85% of what you see at the seam is ZERO-MEAN SCATTER, and a global shift/scale colour match
//       (the textbook first move) can only ever remove the bias — i.e. at most ~15-20% of the defect,
//       fitted on the least reliable samples in the pipeline (grazing rim texels). NOT IMPLEMENTED ON
//       PURPOSE. If you are about to add TEXPROJ_SEAM_MATCH, re-measure bias/|d| first; if it is still
//       ~0.15, you are building a lever that cannot move the thing being complained about.
//       (Same verdict, for the same reason, on gradient-domain/Poisson and multi-band: seamless cloning
//       exists to remove a STEP at a boundary. There is no step. It would spread the scatter and cost a
//       surface-Laplacian solve over ~4M texels — and the atlas is confetti, ~3000-4400 charts of median
//       4-7 verts, so any UV-space kernel is the camo bug all over again.)
//
// WHAT THE SEAM ACTUALLY IS: the front paints facing > NZ_LO, the back paints facing < -NZ_LO, and the
// band between them is 100% hole (measured: the |facing|<0.05 bands are 100% unpainted on the outer
// wall) — widened further by the grazing-angle bg-reject, which fires on ~13% of the back's facing>0.05
// texels. So the crossover is a STRIP OF INVENTED COLOUR ~20-25 degrees of arc wide, and the 3D fill owns
// it. The fill is a k-NN inverse-distance estimator, so inside that strip whichever side happens to be
// nearest wins: the colour FLIPS rather than ramps, and it mottles. That is the "colours dont blend".
//
// TWO LEVERS, BOTH MEASURED, BOTH DEFAULT-OFF (the owner judges on the render):
//
//   TEXPROJ_BACK_BG=auto — BY FAR THE BIGGER ONE, and it is really a MASK BUG, not a blend problem.
//     The non-front detector defaults to `black` at 0.05, but a flux2-generated view is not a matte: its
//     background carries 1-2/255 dither (only 47% of back_v1_rgba.png's background is exactly 0), and its
//     bearskin and boots are BLACK and TOUCH the silhouette, so the border flood-fill guard cannot save
//     them. Measured on back_v1_rgba.png, silhouette-vs-subject IoU:
//         black @0.05  (the default) : IoU 0.857 — 27832 px = 12.37% of the mesh's back silhouette lands
//                                      on "background" and is refused -> holes -> invented crossover
//         black @1/255 (the front's) : IoU 0.288 — CATASTROPHE. The dither blocks the flood fill, subject
//                                      becomes 989082 px (94% of the image) and the bbox fit blows up to
//                                      scale 1.145. Do NOT copy the front's threshold onto a generated view.
//         alpha        (RMBG's matte): IoU 0.945 / 2.73%  <- vs the FRONT's own ceiling of 0.971 / 1.40%
//     i.e. `auto` (= alpha when the file has one) takes the back from 8.8x the front's registration error
//     down to 2.0x.
//     ⚠⚠ THIS FILE'S OWN BUG-2 ALARM WAS ALREADY SAYING SO, AND WAS IGNORED. BUG 2 above promises that the
//     rejected set's MEAN FACING must come out far below the kept set's, "and if they are comparable, the
//     reject is firing in the surface interior and the mask or the fit is wrong — project_onto prints a
//     WARN for exactly that case, and you should believe it over this comment." It does, and you should:
//         back, black@0.05 : bg-rejected 24.46%, mean facing rejected 0.491 vs kept 0.695
//                            -> "[WARN: rejected texels are NOT grazing — mask/fit suspect, not streaks]"
//         back, alpha      : bg-rejected 14.95%, mean facing rejected 0.294 vs kept 0.707 -> "[OK]"
//         front (control)  : bg-rejected 13.07%, mean facing rejected 0.273 vs kept 0.657 -> "[OK]"
//     The back's reject was firing in the SURFACE INTERIOR at half the kept facing — i.e. it was never a
//     grazing/streak problem at all — and `alpha` lands it exactly on the front's own profile. The alarm
//     worked; nobody read it.
//     On the real atlas: back coverage 18.7% -> 21.0%, holes 58.4% -> 56.1% (-91898 texels),
//     and 69.9% of covered texels change by a mean of 10.7/255. It resolves PER VIEW and degrades to
//     `black` when a view has no alpha, so it cannot be the wrong answer — but the DEFAULT STAYS `black`
//     so it cannot silently change an existing run. ⚠ It needs an RGBA back (back_v1_rgba.png, not
//     back_v1.png): they carry identical RGB, and only the RGBA one carries the matte.
//
//   TEXPROJ_SEAM_BLEND=1 — the honest, small one. Splits the fill's OWN k-NN set by which view painted
//     each neighbour, reconstructs each view's estimate separately, and cross-fades them by
//     smoothstep(-BAND,+BAND,facing) — a partition of unity across the silhouette, so the crossover is
//     MONOTONE front-to-back by construction instead of a nearest-neighbour lottery. This is the only
//     place the ramp CAN live, because the k-NN neighbourhood is the only overlap that exists (see (1)).
//     Surgical and cheap: it changes exactly the seam strip (131696 texels changed vs 131694 seam texels)
//     by a mean of 30.5/255 each, 3.3% of covered, and costs no measurable time. It does NOT change
//     coverage. It does NOT remove the scatter — nothing can, the two images genuinely disagree there —
//     it only stops the transition flipping.
//
// WHAT IS LEFT AND IS NOT A BLEND PROBLEM: the strip is dark, because both images' rim pixels are the
// shaded, foreshortened grazing edge of a LIT render. Projecting a lit image bakes that shading in. No
// blend fixes it; it needs either a rim-shading compensation or an unlit source.
//
// SEE proj_seam.png (R=front-painted, G=back-painted, B=the seam strip) and the `seam:` stage line.
// texproj_seam_probe.cpp reproduces every number above on the CPU, and SEAM_REAL_ATLAS=1 drives the REAL
// project_onto through a synthetic confetti atlas with no GPU and no bake.
// ---------------------------------------------------------------------------------------------
// THE METAL_ROUGH CAVEAT (measured 2026-07-17 on the soldier, proj_v1) — NOT THE GOLD-BUTTON COMPLAINT
//
// ⚠ READ BUG 1 FIRST. The owner has since judged the build and the gold DOES show: "they show as gold
// theyre just maybe not 'deep enough' into the model ... they sit a little off the bumps". That is a
// REGISTRATION complaint (BUG 1), not a material one. Everything below is still true as measurement, but it
// is NOT the reported defect — do not chase metallicRoughness on the strength of it.
//
// We keep bt.metal_rough from the volume bake. On the soldier's gold jacket buttons this is a MEASURED
// quality ceiling, not a neutral choice. Sampling the owner's proj_v1/proj_tex.glb at the face centroids
// of the 806 front-facing chest-button triangles (material has metallicFactor=roughnessFactor=1.0, so the
// texture bytes are used verbatim):
//     baseColor at those texels = sRGB(218,171,130)   source matte there = sRGB(219,172,131)
//        -> the projected GOLD ALBEDO is essentially exact (L1 median 2/765; only 1.5% deviate >90)
//     metallic  at those texels = median 0.000, mean 0.148, only 14.8% above 0.5
//     roughness at those texels = mean 0.564
// i.e. the atlas carries perfect gold albedo but tells the renderer "rough non-metallic dielectric" — so
// the buttons render as flat mustard PAINT and cannot throw a gold specular. They are in fact LESS
// metallic than the surrounding red jacket (median 41/255) and the black boots (median 80/255): the ~64³
// PBR latent simply has no idea a 25px button is there. THIS is "the gold doesn't hold", and it is NOT
// fixable from this file — no amount of projection quality changes metallicRoughness. A projected-
// metalness heuristic is a decision for the owner, not something to invent here.
// ---------------------------------------------------------------------------------------------
#pragma once
#include "tex_atlas.hpp"      // texatlas::BakedTexture, inpaint_telea, dilate_background
#include "image_io.hpp"       // imgio::load_rgb01 + stb_image (STB_IMAGE_IMPLEMENTATION lives there)
#include "glb_textured.hpp"   // glb::encode_png (debug dumps only)
#include <vector>
#include <string>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <limits>
#include <sys/stat.h>

namespace texproj {

// Upper bound on the number of views the SEAM cross-fade + seam metric track per fill texel. The seam
// machinery keeps a small fixed per-view accumulator on the stack inside the fill's inner loop, so it
// must be a compile-time bound. Views beyond this still project and still fill normally — they simply
// don't get a per-view estimate — so raising it is a perf/stack decision, not a correctness one.
// 8 is far past the front+back contract (and past the 4-5 views a turnaround sheet could ever give).
#define TEXPROJ_MAX_SEAM_VIEWS 8

// ---------------------------------------------------------------------------------------------
// config / results
// ---------------------------------------------------------------------------------------------
// An extra camera: a yaw about +Y (0 = front, 180 = back) plus the image taken from there.
struct ViewSpec {
    float yaw_deg = 0.f;
    std::string img;
};

struct Cfg {
    // Camera. Defaults = DEF_CAM / DEF_DIST / DEF_MS (image_to_rig.cpp:49) = the miku cam.
    float cam = 0.7332379387484828f;   // camera_angle_x (rad)
    float dist = 1.3021559715270996f;  // camera distance
    float ms = 1.0f;                   // mesh_scale
    std::string front_img;             // REQUIRED — the --image matte (the frame the mesh was built from)
    std::string back_img;              // optional ("" = none) — sugar for a view at yaw=180
    std::vector<ViewSpec> views;       // optional EXTRA views (any yaw); front + back sugar are implicit
    // Keep the volume bake for texels no supplied camera can observe. This prevents a front-only image
    // from inventing noisy colour behind hair, under arms, or around a missing back/side view.
    bool preserve_base_for_holes = false;
    bool verbose = true;
    std::string debug_dir;             // optional: dump z-buffers / confidence / align / fill-source PNGs
};

struct Stats {
    int covered = 0;                   // atlas texels inside a chart triangle
    double front_pct = 0;              // % of covered painted (at least partly) by the front image
    double back_pct = 0;               // % of covered painted (at least partly) by the yaw=180 view
    double hole_pct = 0;               // % of covered with NO view -> filled
    // per-view coverage, index-parallel to the assembled view list (0 = front)
    std::vector<float> view_yaw;
    std::vector<double> view_pct;
    // per-view background-sample rejection (BUG 2), index-parallel to view_yaw
    std::vector<int>   view_bgrej;        // texels that passed facing+depth but sampled outside the subject
    std::vector<float> view_bgrej_facing; // mean facing of those texels (LOW = the grazing signature)
    std::vector<float> view_kept_facing;  // mean facing of the texels this view actually painted
    // per-view silhouette fit (index-parallel; index 0 = the FRONT, fitted since TEXPROJ_FRONT_ALIGN)
    std::vector<float> view_scale;
    std::vector<float> view_tx, view_ty;
    // hole fill accounting
    int n_hole = 0;                    // covered texels no view could paint
    int n_fill3d = 0;                  // of those, filled by the 3D k-nearest fill
    int n_telea = 0;                   // of those, fell through to the atlas-space Telea/dilate
    int n_base = 0;                    // of those, retained from the pre-projection volume bake
    // BUG 3 — THE SEAM. n_seam = 3D-fill texels whose k-NN spans >=2 views: that set IS the front/back
    // crossover, and it is the ONLY place the two views meet (the projection's confidence overlap band is
    // empty by construction — see Cam::facing). seam_absdiff = mean |viewA-viewB| there, in sRGB units;
    // seam_bias = the same difference SIGNED. |d| is what the eye sees; the bias is the only component a
    // global exposure/WB match could remove, so bias/|d| << 1 means colour matching is the wrong tool.
    int   n_seam = 0;
    float seam_absdiff[3] = {0, 0, 0};
    float seam_bias[3] = {0, 0, 0};
    float seam_mean_absdiff = 0.f;
    bool  seam_blend_on = false;
    // silhouette fit of the yaw=180 view (identity when absent / disabled / degenerate)
    bool  back_aligned = false;
    float back_scale = 1.f;            // uniform scale, view-image px per silhouette px
    float back_tx = 0.f, back_ty = 0.f;// translation in view-image px
    // per-phase timings (the owner wants a perf hunt after this lands — instrument up front)
    float normal_sign = 1.f;   // measured mesh normal convention: +1 outward, -1 inward
    double t_uv_raster = 0, t_zbuf_front = 0, t_zbuf_back = 0, t_align = 0, t_project = 0,
           t_fill3d = 0, t_inpaint = 0;
};

// ---------------------------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------------------------
static inline float srgb_to_linear(float c) {
    if (c <= 0.f) return 0.f;
    if (c >= 1.f) return 1.f;
    return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}
static inline float linear_to_srgb(float c) {
    if (c <= 0.f) return 0.f;
    if (c >= 1.f) return 1.f;
    return c <= 0.0031308f ? 12.92f * c : 1.055f * std::pow(c, 1.f / 2.4f) - 0.055f;
}
static inline float smoothstep(float lo, float hi, float x) {
    if (hi <= lo) return x >= hi ? 1.f : 0.f;
    float t = (x - lo) / (hi - lo);
    t = t < 0.f ? 0.f : (t > 1.f ? 1.f : t);
    return t * t * (3.f - 2.f * t);
}
static inline uint8_t u8f(float v) { int q = (int)std::lround(v * 255.f); return (uint8_t)(q < 0 ? 0 : (q > 255 ? 255 : q)); }
static inline float envf(const char* k, float d) { const char* e = std::getenv(k); return e ? (float)std::atof(e) : d; }
static inline int   envi(const char* k, int d)   { const char* e = std::getenv(k); return e ? std::atoi(e) : d; }

// Exact sin/cos for a yaw in DEGREES, with the quadrantal angles SNAPPED.
//
// WHY THE SNAP IS LOAD-BEARING: the front/back cameras must be reproduced BIT-EXACTLY by the generalized
// yaw camera (yaw=0 and yaw=180), and naive trig cannot do that — sinf(pi) is -8.742278e-08, not 0, so a
// yaw=180 view would compute xc = -x + 8.7e-8*z and drift off the old back camera in the last bits.
// Snapping 0/90/180/270 makes yaw=0 and yaw=180 collapse to the exact old expressions (see Cam::project).
static inline void sincos_deg(float deg, float& s, float& c) {
    float d = std::fmod(deg, 360.f);
    if (d < 0.f) d += 360.f;
    if (d == 0.f)   { s =  0.f; c =  1.f; return; }
    if (d == 90.f)  { s =  1.f; c =  0.f; return; }
    if (d == 180.f) { s =  0.f; c = -1.f; return; }
    if (d == 270.f) { s = -1.f; c =  0.f; return; }
    const float r = d * (float)M_PI / 180.f;
    s = std::sin(r); c = std::cos(r);
}

// ---------------------------------------------------------------------------------------------
// THE CAMERA — derived from geometry_e2e.hpp ProjCam::project (constructed pixal3d_chain.hpp:173 as
// ProjCam(cam,dist,ms)); mesh verts are ALREADY in this frame ([-0.5,0.5], marching-cubes world).
//
// GENERALIZED TO AN ARBITRARY YAW ABOUT +Y. The camera sits at (dist*sin(yaw), 0, dist*cos(yaw)) looking
// at the origin, up +y. Equivalently: rotate the vertex into the camera frame with Ry(-yaw), then run the
// original pinhole math. yaw=0 is the FRONT, yaw=180 is the BACK.
//
//   t = tan(cam*0.5)
//   xc = cos(yaw)*x - sin(yaw)*z          // Ry(-yaw) applied to (x,y,z)
//   zc = sin(yaw)*x + cos(yaw)*z
//   depth  = dist - zc
//   u_norm = 0.5 + 0.5*xc/(t*depth)       // * image_width  -> pixel col
//   v_norm = 0.5 - 0.5*y /(t*depth)       // * image_height -> pixel row (y-flipped)
//
// REDUCTIONS (verified bit-exact thanks to sincos_deg's quadrant snap):
//   yaw=0   : sin=0, cos=1  -> xc=x, zc=z   -> depth=dist-z, u=0.5+0.5*x*inv   == the old FRONT
//   yaw=180 : sin=0, cos=-1 -> xc=-x, zc=-z -> depth=dist+z, u=0.5+0.5*(-x)*inv == the old BACK
//             (the old back camera's `xs=-x` and `depth=dist+z` fall straight out; -(-z) is exact, and
//              0.f*z contributes an exact 0, so no bits move.)
//
// NOTE ON ms: the spec handed to this file originally omitted the mesh_scale divisor (a no-op at the
// default ms=1.0). ProjCam does gx=x/ms/2 etc, so the general form divides vx,vy,vz by ms. Kept general
// so `--cam <ang> <dist> <scale>` with scale!=1 stays correct.
// ---------------------------------------------------------------------------------------------
struct Cam {
    float t, dist, ms;
    float yaw_deg, sy, cy;
    // Sign of the mesh's normal convention: +1 = OUTWARD (front surface has n.z>0), -1 = INWARD.
    // MEASURED, not assumed — see detect_normal_sign(). The handoff and the old prototype both assert
    // "front-facing = vertex_normal.z > 0", and that is BACKWARDS for every mesh downstream of the
    // UltraShape refine: refined/quad/atlas normals point INWARD (coarse.glb is outward — refine flips
    // the winding). Proven on the soldier: the 200 frontmost verts (z=+0.146, nearest the camera) have
    // mean n.z = -0.9977, the 200 backmost have +0.9908. The ~48%-front-facing sanity check in the
    // handoff cannot catch this: it is ~50/50 either way. This is why the prototype painted the model's
    // far/inner surface ("a double face projected into the hat", "drawing partially inside the model")
    // while its own preview looked correct — the preview only rendered the same nz>0 set it painted.
    float nsign = 1.f;
    Cam(float fov, float d, float m, float yaw_degrees)
        : t(std::tan(fov * 0.5f)), dist(d), ms(m > 1e-8f ? m : 1.f), yaw_deg(yaw_degrees) {
        sincos_deg(yaw_degrees, sy, cy);
    }
    // -> normalized image coords u,v in [0,1] (0 = left/top edge, 1 = right/bottom edge). Returns depth.
    inline float project(const float* v, float& u, float& vv) const {
        const float x = v[0] / ms, y = v[1] / ms, z = v[2] / ms;
        const float xc = cy * x - sy * z;
        const float zc = sy * x + cy * z;
        const float depth = dist - zc;
        const float dd = depth > 1e-6f ? depth : 1e-6f;
        const float inv = 1.f / (t * dd);
        u = 0.5f + 0.5f * xc * inv;
        vv = 0.5f - 0.5f * y * inv;
        return depth;
    }
    // Camera-space facing of a texel normal, >0 = this camera can see the surface. The direction from the
    // surface TOWARD the camera is (sin(yaw), 0, cos(yaw)), so facing = nsign * dot(n, that). `nsign`
    // folds in the measured normal convention, so callers never need to know which way normals point.
    // Reduces to the old cases: yaw=0 -> n[2]*nsign; yaw=180 -> -n[2]*nsign.
    inline float facing(const float* n) const { return (sy * n[0] + cy * n[2]) * nsign; }
};

// ---------------------------------------------------------------------------------------------
// image loading
//
// FINDING (worth flagging): imgio::load_rgb01 (image_io.hpp:19) calls stbi_load(..., &c, 3) — it FORCES
// 3 channels, so it DROPS ALPHA, and it cannot tell you whether the file HAD any. stb reports the FILE's
// native channel count in `c` even when you request 4, so a local RGBA loader gives us both the pixels and
// an honest has_alpha answer. That's what load_rgba01 is for.
//
// EVERY view loads through it now, the front included. The front used to use load_rgb01 on the reasoning
// that "the front needs no subject detection" — that reasoning died with BUG 1 (it needs a fit) and BUG 2
// (it needs a mask). The RGB channels are bit-identical either way: stb's channel conversion does not
// depend on req_comp, so this is a capability change, not a numerical one.
// ---------------------------------------------------------------------------------------------
static inline bool load_rgba01(const std::string& path, imgio::Image& im, std::vector<float>& alpha, bool& has_alpha) {
    int w = 0, h = 0, c = 0;
    unsigned char* d = stbi_load(path.c_str(), &w, &h, &c, 4);   // always RGBA out; c = channels IN FILE
    if (!d) { std::fprintf(stderr, "[texproj] failed to load image: %s (%s)\n", path.c_str(), stbi_failure_reason()); return false; }
    has_alpha = (c == 4 || c == 2);
    im.w = w; im.h = h; im.rgb.resize((size_t)w * h * 3);
    if (has_alpha) alpha.resize((size_t)w * h);
    for (size_t i = 0; i < (size_t)w * h; i++) {
        for (int k = 0; k < 3; k++) im.rgb[i * 3 + k] = d[i * 4 + k] / 255.f;
        if (has_alpha) alpha[i] = d[i * 4 + 3] / 255.f;
    }
    stbi_image_free(d);
    return true;
}

// ---------------------------------------------------------------------------------------------
// bilinear sample, align_corners=False + border clamp (mirrors geometry_e2e.hpp:56 bilinear()).
// u,v are normalized [0,1] edge coords -> index coord fx = u*W - 0.5.
//
// NOTE: geometry_e2e's ProjCam+bilinear COMPOSITION effectively samples at index u*W (it adds a
// half-pixel then undoes align_corners=False), i.e. it carries a +0.5px bias. That composition is
// torch-parity for the DINO cond path and is bit-frozen; it is NOT the right convention for resampling
// an image for display. We use the standard u*W-0.5 here (as specified). The difference is half a pixel
// at 1024-1536px — far below the overlay-verified alignment bar.
// ---------------------------------------------------------------------------------------------
static inline void sample_bilinear(const imgio::Image& im, float u, float v, float out[3]) {
    const float fx = u * im.w - 0.5f, fy = v * im.h - 0.5f;
    const int x0 = (int)std::floor(fx), y0 = (int)std::floor(fy);
    const float wx = fx - (float)x0, wy = fy - (float)y0;
    auto cl = [](int a, int hi) { return a < 0 ? 0 : (a > hi ? hi : a); };
    const int x0c = cl(x0, im.w - 1), x1c = cl(x0 + 1, im.w - 1);
    const int y0c = cl(y0, im.h - 1), y1c = cl(y0 + 1, im.h - 1);
    const float* p00 = &im.rgb[((size_t)y0c * im.w + x0c) * 3];
    const float* p01 = &im.rgb[((size_t)y0c * im.w + x1c) * 3];
    const float* p10 = &im.rgb[((size_t)y1c * im.w + x0c) * 3];
    const float* p11 = &im.rgb[((size_t)y1c * im.w + x1c) * 3];
    for (int c = 0; c < 3; c++)
        out[c] = (1.f - wy) * ((1.f - wx) * p00[c] + wx * p01[c])
               +        wy  * ((1.f - wx) * p10[c] + wx * p11[c]);
}

// ---------------------------------------------------------------------------------------------
// DEPTH PREPASS — rasterize every mesh triangle through `cam` into an image-space z-buffer at the
// source image's resolution, keeping the MIN depth per pixel. Standard edge-function raster + bbox clip
// (same structure as tex_atlas.hpp:917-953), pixel centres at (x+0.5, y+0.5) so the mapping u -> pixel
// index floor(u*W) is exactly consistent with the texel lookup below.
//
// Depth is interpolated PERSPECTIVE-CORRECTLY (1/depth is what's linear in screen space), which is exact
// for a planar triangle. No backface culling: on an open/non-manifold mesh the frontmost surface may be
// wound either way, and we want the frontmost surface whatever its winding.
//
// SERIAL BY DESIGN. ~330k tris is cheap (well under a second) and a threaded min-z-buffer needs either
// atomics or per-thread buffers + merge; correct beats clever here (and this runs once per view).
// ---------------------------------------------------------------------------------------------
struct ZBuf {
    int w = 0, h = 0;
    std::vector<float> d;   // min depth per pixel, +inf = uncovered (background)
    inline float at(int x, int y) const { return d[(size_t)y * w + x]; }
};

static inline ZBuf raster_depth(const std::vector<float>& verts, const std::vector<uint32_t>& faces,
                                const Cam& cam, int W, int H) {
    ZBuf z; z.w = W; z.h = H;
    z.d.assign((size_t)W * H, std::numeric_limits<float>::infinity());
    const size_t F = faces.size() / 3;
    for (size_t f = 0; f < F; f++) {
        float X[3], Y[3], D[3];
        bool bad = false;
        for (int k = 0; k < 3; k++) {
            const float* p = &verts[(size_t)faces[f * 3 + k] * 3];
            float u, v;
            D[k] = cam.project(p, u, v);
            if (!(D[k] > 1e-4f)) { bad = true; break; }   // behind / on the camera plane (never happens
            X[k] = u * (float)W;                          // for a [-0.5,0.5] mesh at dist~1.3, but guard)
            Y[k] = v * (float)H;
        }
        if (bad) continue;
        const float area = (X[1] - X[0]) * (Y[2] - Y[0]) - (Y[1] - Y[0]) * (X[2] - X[0]);
        if (std::fabs(area) < 1e-12f) continue;
        const float inv = 1.f / area;                     // normalizes winding, like the atlas raster
        int x0 = (int)std::floor(std::min({X[0], X[1], X[2]})), x1 = (int)std::ceil(std::max({X[0], X[1], X[2]}));
        int y0 = (int)std::floor(std::min({Y[0], Y[1], Y[2]})), y1 = (int)std::ceil(std::max({Y[0], Y[1], Y[2]}));
        x0 = std::max(0, x0); y0 = std::max(0, y0); x1 = std::min(W - 1, x1); y1 = std::min(H - 1, y1);
        if (x1 < x0 || y1 < y0) continue;
        const float IW[3] = { 1.f / D[0], 1.f / D[1], 1.f / D[2] };
        for (int y = y0; y <= y1; y++) for (int x = x0; x <= x1; x++) {
            const float sx = (float)x + 0.5f, sy = (float)y + 0.5f;
            const float w0 = ((X[1] - sx) * (Y[2] - sy) - (Y[1] - sy) * (X[2] - sx)) * inv;
            const float w1 = ((X[2] - sx) * (Y[0] - sy) - (Y[2] - sy) * (X[0] - sx)) * inv;
            const float w2 = 1.f - w0 - w1;
            if (w0 < 0.f || w1 < 0.f || w2 < 0.f) continue;
            const float iw = w0 * IW[0] + w1 * IW[1] + w2 * IW[2];
            if (!(iw > 1e-9f)) continue;
            const float dep = 1.f / iw;
            float& slot = z.d[(size_t)y * W + x];
            if (dep < slot) slot = dep;
        }
    }
    return z;
}

// Occluder depth at (u,v): the MAX over the 3x3 neighbourhood of the frontmost-surface depths, so a
// silhouette texel isn't spuriously rejected by raster aliasing (the exact pixel may have been claimed
// by a triangle a hair in front). Uncovered (+inf) neighbours are SKIPPED, not maxed in — otherwise a
// single background pixel next to the silhouette would make the occluder infinitely far and wave
// everything through. If the whole neighbourhood is uncovered nothing can occlude this texel anyway
// (returns visible); that's the honest answer, not a leak: the hat/face case has the hat solidly
// rasterized, so the buried face texel always finds a much nearer occluder and is rejected.
static inline bool visible(const ZBuf& z, float u, float v, float depth, float eps) {
    const int ix = (int)std::floor(u * (float)z.w), iy = (int)std::floor(v * (float)z.h);
    if (ix < 0 || iy < 0 || ix >= z.w || iy >= z.h) return false;   // projects off-image
    float occ = -std::numeric_limits<float>::infinity();
    bool any = false;
    for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
        const int nx = ix + dx, ny = iy + dy;
        if (nx < 0 || ny < 0 || nx >= z.w || ny >= z.h) continue;
        const float dd = z.at(nx, ny);
        if (!std::isfinite(dd)) continue;
        if (dd > occ) occ = dd;
        any = true;
    }
    if (!any) return true;
    return depth <= occ + eps;
}

// ---------------------------------------------------------------------------------------------
// NORMAL-CONVENTION DETECTION (self-calibrating; never trust a documented convention)
//
// Which normal set is the surface the camera actually SEES? Ask the z-buffer — it is ground truth and
// we already built it. For each candidate sign, count verts that both face the camera AND sit on the
// frontmost surface. The real convention wins by a landslide (soldier: -1 scores 96.1% vs +1's 3.9%),
// so this is robust; a near-tie would mean the mesh is degenerate, and we keep +1 and say so.
//
// This exists because the "front-facing = n.z > 0" claim in the handoff is false for refined/quad/atlas
// meshes. Rather than hard-code the flip (coarse.glb really IS outward — the two disagree), measure it.
// Override with TEXPROJ_NORMAL_SIGN=1|-1; `auto` (default) measures.
// ---------------------------------------------------------------------------------------------
static inline float detect_normal_sign(const texatlas::BakedTexture& bt, const ZBuf& z, const Cam& cam,
                                       float eps, bool verbose, double* score_pos, double* score_neg) {
    const size_t V = bt.verts.size() / 3;
    if (V == 0 || bt.normals.size() != bt.verts.size()) return 1.f;
    // Stride so huge meshes stay cheap; a few 10k samples decide this overwhelmingly.
    const size_t stride = std::max<size_t>(1, V / 60000);
    size_t n_pos = 0, n_neg = 0, n_tot = 0;
    for (size_t i = 0; i < V; i += stride) {
        const float* P = &bt.verts[i * 3];
        const float* N = &bt.normals[i * 3];
        float u, v; const float d = cam.project(P, u, v);
        if (!visible(z, u, v, d, eps)) continue;   // only the frontmost surface votes
        n_tot++;
        if (N[2] > 0.f) n_pos++;                   // outward convention would call this front-facing
        else if (N[2] < 0.f) n_neg++;              // inward convention would
    }
    const double sp = n_tot ? (double)n_pos / (double)n_tot : 0.0;
    const double sn = n_tot ? (double)n_neg / (double)n_tot : 0.0;
    if (score_pos) *score_pos = sp;
    if (score_neg) *score_neg = sn;
    const float sign = (sn > sp) ? -1.f : 1.f;
    if (verbose)
        std::printf("[texproj] normal convention: %s (z-buffer agreement +1:%.1f%% vs -1:%.1f%% over %zu "
                    "frontmost samples)%s\n",
                    sign > 0 ? "OUTWARD (front = n.z>0)" : "INWARD (front = n.z<0)",
                    100.0 * sp, 100.0 * sn, n_tot,
                    (std::fabs(sp - sn) < 0.15) ? "  [WARN: near-tie, mesh may be degenerate]" : "");
    return sign;
}

// ---------------------------------------------------------------------------------------------
// PER-VIEW AUTO-ALIGN (silhouette-bbox fit) — EVERY view, including the FRONT
//
// A generated view is NOT pixel-registered with the mesh: it is either a flux2 view edit (framing and
// scale drift) or a panel split out of a turnaround sheet (crop framing drifts). AND NEITHER IS THE FRONT:
// the mesh is the REFINED mesh bbox-canon'd onto the coarse frame, not the matte's exact projection —
// measured ~1% narrow / ~0.9% short on the soldier. See BUG 1 at the top. TEXPROJ_FRONT_ALIGN=0 restores
// the old unfitted front for A/B.
//
// That view's z-buffer's covered pixels ARE the mesh's silhouette from that yaw, so we already have the
// target for free. Fit a 2D SIMILARITY (uniform scale + translate — never anisotropic, that would shear
// the subject's aspect) mapping the view IMAGE's subject bbox onto that silhouette bbox:
//
//     x_img = scale * x_sil + tx,   tx = sub_cx - scale * sil_cx     (same for y)
//
// SCALE = MIN(sub_w/sil_w, sub_h/sil_h), not the mean. Justification: with the min, the mapped box is
// contained in the subject bbox on BOTH axes, so a silhouette texel can never map outside the subject
// and sample background — which is exactly the failure the grazing-angle ramp exists to suppress, and
// exactly what the old prototype's speckle was. The mean splits an aspect mismatch and WILL bleed
// background on the looser axis. The min errs toward a hair of magnification, which is invisible.
// ---------------------------------------------------------------------------------------------
struct BBox {
    int x0 = 0, y0 = 0, x1 = -1, y1 = -1;
    inline bool valid() const { return x1 >= x0 && y1 >= y0; }
    inline float w() const { return (float)(x1 - x0 + 1); }
    inline float h() const { return (float)(y1 - y0 + 1); }
    inline float cx() const { return 0.5f * (float)(x0 + x1 + 1); }   // in continuous pixel-edge coords
    inline float cy() const { return 0.5f * (float)(y0 + y1 + 1); }
};

// bbox of the mesh's silhouette from a view = the covered (finite-depth) pixels of that view's z-buffer
static inline BBox silhouette_bbox(const ZBuf& z) {
    BBox b; b.x0 = z.w; b.y0 = z.h; b.x1 = -1; b.y1 = -1;
    for (int y = 0; y < z.h; y++) for (int x = 0; x < z.w; x++) {
        if (!std::isfinite(z.at(x, y))) continue;
        b.x0 = std::min(b.x0, x); b.x1 = std::max(b.x1, x);
        b.y0 = std::min(b.y0, y); b.y1 = std::max(b.y1, y);
    }
    return b;
}

// bbox of a binary mask's set pixels
static inline BBox bbox_of_mask(const std::vector<uint8_t>& m, int w, int h) {
    BBox b; b.x0 = w; b.y0 = h; b.x1 = -1; b.y1 = -1;
    for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
        if (!m[(size_t)y * w + x]) continue;
        b.x0 = std::min(b.x0, x); b.x1 = std::max(b.x1, x);
        b.y0 = std::min(b.y0, y); b.y1 = std::max(b.y1, y);
    }
    return b;
}

// Accounting for one view's subject mask — printed so a bad detector is visible instead of silent.
//
// ⚠ n_nonzero / n_lost ARE ONLY MEANINGFUL FOR THE `black` DETECTOR. They exist to catch the LANDMINE:
// a luminance threshold quietly eating a dark material. Their premise is "nonzero RGB = subject", which
// holds for a make_matte matte (composited on EXACT black) and is FALSE for anything else. On a flux2
// generated view the background carries 1-2/255 of dither, so 64.5% of its "nonzero" pixels are
// background — and in `alpha` mode those are correctly excluded by RMBG's own matte. Reporting that as
// "DISCARDED-nonzero 64.53% [WARN: lower the threshold]" is not just noise, it is advice that INVERTS the
// right answer: lowering the black threshold to 1/255 on that view takes silhouette-vs-subject IoU from
// 0.945 to 0.288 (the dither blocks the border flood fill and 94% of the image becomes "subject").
// So `mode_is_black` gates them, and the caller must not print them otherwise. See BUG 3.
struct MaskStats {
    int n_subject = 0;      // final subject px (after hole fill, before erosion)
    int n_nonzero = 0;      // px with ANY nonzero rgb — the honest upper bound on "subject", BLACK MODE ONLY
    int n_holefill = 0;     // px the border flood-fill RECLAIMED (dark, but enclosed by subject)
    int n_lost = 0;         // nonzero px STILL called background — the dark-material (boot) risk number.
                            // BLACK MODE ONLY: meaningless when the detector is not luminance-based.
    int n_eroded = 0;       // px the erosion removed from the final subject
    bool mode_is_black = true;  // false => n_nonzero/n_lost are NOT interpretable; do not warn on them
};

// PER-VIEW SUBJECT MASK — where the image actually has subject. Detector `mode`:
//   "black" (default) — candidate background = max(r,g,b) <= thresh  (the RMBG/make_matte convention)
//   "alpha"           — candidate background = a <= 0.5, only when the FILE actually carried alpha
//
// Then, if `fill_holes`, the mask is NOT simply !candidate_bg: true background is the candidate-background
// region REACHABLE FROM THE IMAGE BORDER (4-connected flood fill). Anything dark but ENCLOSED by subject is
// subject. This is what protects a dark material in the interior (the soldier's bearskin: 5710 enclosed px)
// from being punched out of the texture — see THE LANDMINE at the top of this file. It cannot protect a
// dark region that TOUCHES the silhouette (a boot sole), which is why `thresh` must also stay low.
static inline void subject_mask(const imgio::Image& im, const std::vector<float>& alpha, bool has_alpha,
                                const std::string& mode, float thresh, bool fill_holes,
                                std::vector<uint8_t>& out, MaskStats* ms = nullptr) {
    const int w = im.w, h = im.h;
    const size_t N = (size_t)w * h;
    const bool use_alpha = (mode == "alpha") && has_alpha && alpha.size() == N;
    std::vector<uint8_t> bg(N, 0);      // 1 = candidate background
    int n_nonzero = 0;
    for (size_t i = 0; i < N; i++) {
        const float* p = &im.rgb[i * 3];
        const float mx = std::max({p[0], p[1], p[2]});
        if (mx > 0.f) n_nonzero++;
        bg[i] = use_alpha ? (alpha[i] <= 0.5f) : (mx <= thresh);
    }
    if (fill_holes) {
        // BFS from every border pixel that is candidate-background. Explicit stack (no recursion: a
        // 1060x1060 background is ~890k px deep in the worst case and would blow the real stack).
        std::vector<uint8_t> reach(N, 0);
        std::vector<int> stk;
        stk.reserve(N / 4);
        auto push = [&](int x, int y) {
            const size_t i = (size_t)y * w + x;
            if (bg[i] && !reach[i]) { reach[i] = 1; stk.push_back((int)i); }
        };
        for (int x = 0; x < w; x++) { push(x, 0); push(x, h - 1); }
        for (int y = 0; y < h; y++) { push(0, y); push(w - 1, y); }
        while (!stk.empty()) {
            const int i = stk.back(); stk.pop_back();
            const int x = i % w, y = i / w;
            if (x > 0)     push(x - 1, y);
            if (x < w - 1) push(x + 1, y);
            if (y > 0)     push(x, y - 1);
            if (y < h - 1) push(x, y + 1);
        }
        bg.swap(reach);                 // background = ONLY what the border can reach
    }
    out.assign(N, 0);
    for (size_t i = 0; i < N; i++) out[i] = bg[i] ? 0 : 1;
    if (ms) {
        ms->mode_is_black = !use_alpha;
        ms->n_nonzero = n_nonzero;
        ms->n_subject = 0; ms->n_lost = 0; ms->n_holefill = 0;
        for (size_t i = 0; i < N; i++) {
            const float* p = &im.rgb[i * 3];
            const bool nz = std::max({p[0], p[1], p[2]}) > 0.f;
            if (out[i]) ms->n_subject++;
            if (nz && !out[i]) ms->n_lost++;
            // reclaimed = subject now, but the raw detector called it background
            const bool raw_bg = use_alpha ? (alpha[i] <= 0.5f) : (std::max({p[0], p[1], p[2]}) <= thresh);
            if (out[i] && raw_bg) ms->n_holefill++;
        }
    }
}

// ERODE a binary mask by `r` px, L-infinity (square structuring element), separable min-filter.
//
// r is in SOURCE-IMAGE pixels. WHY IT MUST BE >= 1: sample_bilinear taps floor(u*W-0.5) and +1, i.e. the
// four taps for a texel whose lookup pixel is (ix,iy) all lie within [ix-1,ix+1] x [iy-1,iy+1]. Eroding by
// r>=1 therefore guarantees that a texel passing the eroded mask has NO tap in the background. r=2 buys one
// more pixel of margin against the matte's own antialiased rim, whose pixels ARE partial composites against
// black (i.e. already darkened) even though they read as subject.
//
// The image border is treated as BACKGROUND, so a subject running off the edge of the frame erodes inward.
// That is the honest reading: we cannot know what is outside the frame, so we must not sample its clamp.
static inline void erode_mask(std::vector<uint8_t>& m, int w, int h, int r) {
    if (r <= 0) return;
    std::vector<uint8_t> tmp((size_t)w * h, 0);
    for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {          // horizontal pass
        uint8_t v = 1;
        for (int d = -r; d <= r && v; d++) {
            const int xx = x + d;
            if (xx < 0 || xx >= w || !m[(size_t)y * w + xx]) v = 0;
        }
        tmp[(size_t)y * w + x] = v;
    }
    for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {          // vertical pass
        uint8_t v = 1;
        for (int d = -r; d <= r && v; d++) {
            const int yy = y + d;
            if (yy < 0 || yy >= h || !tmp[(size_t)yy * w + x]) v = 0;
        }
        m[(size_t)y * w + x] = v;
    }
}

// similarity transform: silhouette pixel space -> view-image pixel space.
// sx==sy for the default uniform fit; TEXPROJ_ALIGN_ANISO=1 lets them differ (A/B lever only).
struct Fit {
    float sx = 1.f, sy = 1.f, tx = 0.f, ty = 0.f;
    bool fitted = false;
    inline void apply(float x, float y, float& ox, float& oy) const { ox = sx * x + tx; oy = sy * y + ty; }
};

// `aniso` fits the axes INDEPENDENTLY. It is an A/B lever, not a default: on the front the two axis ratios
// already agree to 0.3% (1.0119 vs 1.0090), and that 0.3% is real perspective — the x-extremes (hands) and
// y-extremes (hat/soles) sit at different depths — not aspect error. Anisotropic would fit a fake shear to
// it. It also makes the mapped box EQUAL the subject box, putting silhouette samples exactly on the outline
// where bilinear straddles; that is only survivable because TEXPROJ_BG_REJECT now backstops it.
static inline Fit fit_similarity(const BBox& sil, const BBox& sub, bool aniso = false) {
    Fit f;
    if (!sil.valid() || !sub.valid() || sil.w() < 2.f || sil.h() < 2.f || sub.w() < 2.f || sub.h() < 2.f) return f;
    const float rw = sub.w() / sil.w(), rh = sub.h() / sil.h();
    if (aniso) { f.sx = rw; f.sy = rh; }
    else       { f.sx = f.sy = std::min(rw, rh); }
    f.tx = sub.cx() - f.sx * sil.cx();
    f.ty = sub.cy() - f.sy * sil.cy();
    f.fitted = true;
    return f;
}

// ---------------------------------------------------------------------------------------------
// UV RASTER — per atlas texel 3D position + interpolated normal + coverage, from the BAKED (chart-split)
// topology. Structure copied from tex_atlas.hpp:917-953. bt.uvs are NORMALIZED (tex_atlas.hpp:897 divides
// by the pre-resize atlas dims), so pixel space = uv * (tw, th).
//
// NOTE: bt.tw/th may be SMALLER than the atlas the bake rasterized at (tex_atlas.hpp:1085 resizes the
// packed textures down to TEX_FINAL_SIZE). Because the UVs are normalized, re-rastering at bt.tw/th is
// self-consistent — we simply project at the FINAL atlas resolution, which is exactly the resolution the
// pixels we are about to write live at. (Corollary, measured: the TEX_FINAL_SIZE downsample happens
// INSIDE bake() and therefore BEFORE us; nothing downsamples what we write. It cannot be blamed for
// projected detail loss — see THE METAL_ROUGH CAVEAT at the top.)
// ---------------------------------------------------------------------------------------------
static inline void raster_uv(const texatlas::BakedTexture& bt, int ss,
                             std::vector<float>& pos, std::vector<float>& nrm, std::vector<uint8_t>& mask) {
    const int W = bt.tw, H = bt.th;
    pos.assign((size_t)W * H * 3, 0.f);
    nrm.assign((size_t)W * H * 3, 0.f);
    mask.assign((size_t)W * H, 0);
    const size_t F = bt.faces.size() / 3;
    const bool have_nrm = bt.normals.size() == bt.verts.size();
    for (size_t t = 0; t < F; t++) {
        const uint32_t a = bt.faces[t * 3], b = bt.faces[t * 3 + 1], c = bt.faces[t * 3 + 2];
        const float ax = bt.uvs[(size_t)a * 2] * W, ay = bt.uvs[(size_t)a * 2 + 1] * H;
        const float bx = bt.uvs[(size_t)b * 2] * W, by = bt.uvs[(size_t)b * 2 + 1] * H;
        const float cx = bt.uvs[(size_t)c * 2] * W, cy = bt.uvs[(size_t)c * 2 + 1] * H;
        const float area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
        if (std::fabs(area) < 1e-9f) continue;
        const float inv = 1.f / area;
        int x0 = (int)std::floor(std::min({ax, bx, cx})), x1 = (int)std::ceil(std::max({ax, bx, cx}));
        int y0 = (int)std::floor(std::min({ay, by, cy})), y1 = (int)std::ceil(std::max({ay, by, cy}));
        x0 = std::max(0, x0); y0 = std::max(0, y0); x1 = std::min(W - 1, x1); y1 = std::min(H - 1, y1);
        const float* Pa = &bt.verts[(size_t)a * 3]; const float* Pb = &bt.verts[(size_t)b * 3]; const float* Pc = &bt.verts[(size_t)c * 3];
        const float* Na = have_nrm ? &bt.normals[(size_t)a * 3] : nullptr;
        const float* Nb = have_nrm ? &bt.normals[(size_t)b * 3] : nullptr;
        const float* Nc = have_nrm ? &bt.normals[(size_t)c * 3] : nullptr;
        for (int y = y0; y <= y1; y++) for (int x = x0; x <= x1; x++) {
            float bw0 = 0.f, bw1 = 0.f, bw2 = 0.f; int hits = 0;
            for (int syi = 0; syi < ss; syi++) for (int sxi = 0; sxi < ss; sxi++) {
                const float sx = x + ((float)sxi + 0.5f) / (float)ss;
                const float sy = y + ((float)syi + 0.5f) / (float)ss;
                const float w0 = ((bx - sx) * (cy - sy) - (by - sy) * (cx - sx)) * inv;
                const float w1 = ((cx - sx) * (ay - sy) - (cy - sy) * (ax - sx)) * inv;
                const float w2 = 1.f - w0 - w1;
                if (w0 < 0.f || w1 < 0.f || w2 < 0.f) continue;   // outside (winding-normalized by inv)
                bw0 += w0; bw1 += w1; bw2 += w2; hits++;
            }
            if (!hits) continue;
            const float ih = 1.f / (float)hits;
            const float w0 = bw0 * ih, w1 = bw1 * ih, w2 = bw2 * ih;
            float* P = &pos[((size_t)y * W + x) * 3];
            float* N = &nrm[((size_t)y * W + x) * 3];
            for (int d = 0; d < 3; d++) P[d] = w0 * Pa[d] + w1 * Pb[d] + w2 * Pc[d];
            if (have_nrm) for (int d = 0; d < 3; d++) N[d] = w0 * Na[d] + w1 * Nb[d] + w2 * Nc[d];
            else {   // no baked normals: fall back to the geometric face normal
                const float e1[3] = {Pb[0]-Pa[0], Pb[1]-Pa[1], Pb[2]-Pa[2]};
                const float e2[3] = {Pc[0]-Pa[0], Pc[1]-Pa[1], Pc[2]-Pa[2]};
                N[0] = e1[1]*e2[2] - e1[2]*e2[1]; N[1] = e1[2]*e2[0] - e1[0]*e2[2]; N[2] = e1[0]*e2[1] - e1[1]*e2[0];
            }
            const float L = std::sqrt(N[0]*N[0] + N[1]*N[1] + N[2]*N[2]);
            if (L > 1e-20f) { N[0] /= L; N[1] /= L; N[2] /= L; }
            mask[(size_t)y * W + x] = 1;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// 3D POINT GRID over the PAINTED texels — a uniform spatial hash + ring search.
//
// WHY NOT texrp::DenseHash (tex_reproject.hpp:57)? It was the first thing checked, and it does NOT fit:
// it indexes TRIANGLES (int64 face array + per-face normals) and answers closest-POINT-ON-TRIANGLE
// queries via closest_pt_tri. We need nearest-POINT-among-a-sparse-texel-set, k of them, gated on an
// interpolated per-texel normal that DenseHash has no slot for. Reusing it would mean synthesizing a
// degenerate triangle per texel and fighting its single-best/front-dot contract. So: a local grid in the
// same idiom (CSR cells, ring-expanding search, capped ring), which is ~60 lines and exactly the shape
// of the problem.
// ---------------------------------------------------------------------------------------------
// PER-CELL NORMAL CONE — the difference between "fast" and "looks like a hang".
//
// MEASURED: without it, a 9.0M-texel atlas (= the soldier's real count) at ~50% holes did not finish the
// fill in 400s. Cause: a hole whose normal has NO gate-passing painted texel nearby cannot terminate
// early — it scans every ring to the cap — and at 9M surface texels a 128-grid cell holds ~175 points, so
// each such hole burned ~10^5-10^6 point tests. Storing a cone (axis + half-angle) per cell lets the
// search reject a whole cell in ~10 flops when NO normal inside it could ever pass the gate.
//
// The test is trig-free. A cell survives iff some n in its cone can satisfy dot(nh,n) > mindot, i.e.
//     angle(nh, axis) < acos(mindot) + H          (H = cone half-angle)
// Taking cos of both sides (cos is decreasing on [0,pi]) and expanding cos(A+H):
//     dot(nh, axis) > mindot*cos(H) - sqrt(1-mindot^2)*sin(H)
// so we store cos(H) per cell and the whole test is a dot product and two multiplies.
struct PointGrid {
    int nx = 1, ny = 1, nz = 1;
    float cell = 1.f, inv = 1.f;
    float mn[3] = {0, 0, 0};
    std::vector<int> cell_start;   // CSR offsets over nx*ny*nz cells
    std::vector<int> ids;          // texel ids, grouped by cell
    std::vector<float> cone;       // [cells*4] per cell: axis xyz + cos(half-angle)

    inline int cidx(int x, int y, int z) const { return (x * ny + y) * nz + z; }
    inline void cell_of(const float* p, int& cx, int& cy, int& cz) const {
        cx = (int)((p[0] - mn[0]) * inv); cy = (int)((p[1] - mn[1]) * inv); cz = (int)((p[2] - mn[2]) * inv);
        cx = cx < 0 ? 0 : (cx >= nx ? nx - 1 : cx);
        cy = cy < 0 ? 0 : (cy >= ny ? ny - 1 : cy);
        cz = cz < 0 ? 0 : (cz >= nz ? nz - 1 : cz);
    }

    // Build over `src` texel ids, whose positions live in pos[id*3] and normals in nrm[id*3].
    //
    // `per_cell_cap` (0 = keep everything) is a PERFORMANCE-CRITICAL density cap, and it is the one
    // deliberate approximation in the fill. MEASURED on a 9.0M-texel proxy (= the soldier's real count)
    // with ~50% holes: uncapped took 310s, which is unshippable next to a 5.6s bake. The cause is not the
    // algorithm but the input: 4.5M painted texels on a ~0.86-wide body means ~175 sources per cell, all
    // within ~6mm of each other and (being neighbouring texels of one continuous surface) essentially the
    // same colour. The k=6 nearest of those are sub-millimetre apart, so testing all 175 buys nothing and
    // costs everything. Keeping a strided <=cap per cell leaves the sources ~2mm apart — still far finer
    // than the feature scale of a region we are INVENTING colour for — and makes the cost O(cells*cap).
    // Strided (not random) so the result stays deterministic. knn_gated remains EXACT over whatever the
    // grid holds; the approximation lives here and only here.
    PointGrid(const std::vector<int>& src, const std::vector<float>& pos, const std::vector<float>& nrm,
              int cells_long_axis, int per_cell_cap = 0) {
        float mx[3] = {-1e30f, -1e30f, -1e30f};
        mn[0] = mn[1] = mn[2] = 1e30f;
        for (int id : src) {
            const float* p = &pos[(size_t)id * 3];
            for (int c = 0; c < 3; c++) { mn[c] = std::min(mn[c], p[c]); mx[c] = std::max(mx[c], p[c]); }
        }
        if (src.empty()) { mn[0] = mn[1] = mn[2] = 0.f; mx[0] = mx[1] = mx[2] = 1.f; }
        float ext = 0.f;
        for (int c = 0; c < 3; c++) ext = std::max(ext, mx[c] - mn[c]);
        if (!(ext > 1e-9f)) ext = 1.f;
        cells_long_axis = std::max(1, std::min(512, cells_long_axis));
        cell = ext / (float)cells_long_axis;
        inv = 1.f / cell;
        auto dim = [&](int c) { return std::max(1, std::min(512, (int)((mx[c] - mn[c]) * inv) + 1)); };
        nx = dim(0); ny = dim(1); nz = dim(2);
        std::vector<int> count((size_t)nx * ny * nz + 1, 0);
        for (int id : src) { int cx, cy, cz; cell_of(&pos[(size_t)id * 3], cx, cy, cz); count[cidx(cx, cy, cz) + 1]++; }
        for (size_t i = 1; i < count.size(); i++) count[i] += count[i - 1];
        cell_start = count;
        ids.resize(count.back());
        std::vector<int> cur(count.begin(), count.end() - 1);
        for (int id : src) { int cx, cy, cz; cell_of(&pos[(size_t)id * 3], cx, cy, cz); ids[cur[cidx(cx, cy, cz)]++] = id; }

        const int NC = nx * ny * nz;
        // ---- density cap: keep at most `per_cell_cap` strided sources per cell (see the ctor comment) ----
        if (per_cell_cap > 0) {
            std::vector<int> ns((size_t)NC + 1, 0), keep;
            keep.reserve(ids.size());
            for (int c = 0; c < NC; c++) {
                const int b = cell_start[c], e = cell_start[c + 1], n = e - b;
                ns[c + 1] = ns[c];
                if (n <= 0) continue;
                if (n <= per_cell_cap) { for (int i = b; i < e; i++) keep.push_back(ids[i]); ns[c + 1] += n; }
                else {
                    // stride so the kept subset spans the whole cell rather than clustering at its head
                    const int stride = (n + per_cell_cap - 1) / per_cell_cap;
                    int kept = 0;
                    for (int i = b; i < e && kept < per_cell_cap; i += stride) { keep.push_back(ids[i]); kept++; }
                    ns[c + 1] += kept;
                }
            }
            ids.swap(keep);
            cell_start.swap(ns);
        }

        // ---- per-cell normal cone: axis = normalized mean normal, cos(H) = min dot to that axis ----
        // Built from the FINAL (post-cap) ids so the cone describes exactly what the search will test.
        cone.assign((size_t)NC * 4, 0.f);
        #pragma omp parallel for schedule(dynamic, 1024)
        for (int c = 0; c < NC; c++) {
            const int b = cell_start[c], e = cell_start[c + 1];
            if (b >= e) continue;
            float ax = 0, ay = 0, az = 0;
            for (int i = b; i < e; i++) {
                const float* n = &nrm[(size_t)ids[i] * 3];
                ax += n[0]; ay += n[1]; az += n[2];
            }
            float L = std::sqrt(ax*ax + ay*ay + az*az);
            if (!(L > 1e-20f)) {   // normals cancel exactly (e.g. a cell straddling a thin shell):
                cone[(size_t)c*4+0] = 1.f; cone[(size_t)c*4+1] = 0.f;   // a degenerate all-directions cone
                cone[(size_t)c*4+2] = 0.f; cone[(size_t)c*4+3] = -1.f;  // cos(H)=-1 -> never rejects
                continue;
            }
            ax /= L; ay /= L; az /= L;
            float cosH = 1.f;
            for (int i = b; i < e; i++) {
                const float* n = &nrm[(size_t)ids[i] * 3];
                const float nl = std::sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
                if (!(nl > 1e-20f)) { cosH = -1.f; break; }
                const float d = (ax*n[0] + ay*n[1] + az*n[2]) / nl;
                cosH = std::min(cosH, d);
            }
            cone[(size_t)c*4+0] = ax; cone[(size_t)c*4+1] = ay; cone[(size_t)c*4+2] = az;
            cone[(size_t)c*4+3] = cosH < -1.f ? -1.f : (cosH > 1.f ? 1.f : cosH);
        }
    }
};

// Exact k-nearest search over a PointGrid, gated on normal agreement.
//
// Returns the count found (<= k) and writes ids + squared distances. Candidates must satisfy
// dot(nh, normalize(nrm[q])) > mindot. `maxring` hard-caps the search so it always terminates; anything
// it cannot serve is the caller's problem (we fall back to the atlas-space Telea).
//
// Factored out of project_onto so it can be brute-force verified — the ring-termination bound is exactly
// the sort of thing that silently returns non-nearest neighbours. See cam/knn tests.
static inline int knn_gated(const PointGrid& grid, const std::vector<float>& pos, const std::vector<float>& nrm,
                            const float* P, const float* nh, int k, float mindot, int maxring,
                            int* out_ids, float* out_d2) {
    int cnt = 0; float worst = 1e30f;
    int bx, by, bz; grid.cell_of(P, bx, by, bz);
    // per-query constants for the trig-free cone rejection (see PointGrid): a cell survives iff
    //   dot(nh, axis) > mindot*cos(H) - sqrt(1-mindot^2)*sin(H)
    const float md = mindot < -1.f ? -1.f : (mindot > 1.f ? 1.f : mindot);
    const float s_md = std::sqrt(std::max(0.f, 1.f - md * md));
    for (int r = 0; r <= maxring; r++) {
        // EXACT k-NN termination. At the START of ring r we have searched rings 0..r-1. P sits somewhere
        // inside its own cell, so every point in a ring-r cell is at Euclidean distance >= (r-1)*cell
        // from P. Once we hold k candidates all closer than THAT, no later ring can improve on them.
        // In practice this fires at r<=2. (Using r*cell here instead of (r-1)*cell would break one ring
        // too early and quietly return non-nearest neighbours.)
        if (cnt == k && r >= 2) {
            const float lim = (float)(r - 1) * grid.cell;
            if (worst <= lim * lim) break;
        }
        for (int dx = -r; dx <= r; dx++) {
            const int x = bx + dx; if (x < 0 || x >= grid.nx) continue;
            for (int dy = -r; dy <= r; dy++) {
                const int y = by + dy; if (y < 0 || y >= grid.ny) continue;
                // Iterate the SHELL only: O(r^2) per ring instead of O(r^3). When |dx| or |dy| is already
                // r we are on a side face -> the whole dz column belongs to the shell; otherwise only the
                // dz=+-r caps do.
                const bool side = (std::abs(dx) == r || std::abs(dy) == r);
                const int zstep = side ? 1 : (r > 0 ? 2 * r : 1);
                for (int dz = -r; dz <= r; dz += zstep) {
                    const int z = bz + dz; if (z < 0 || z >= grid.nz) continue;
                    const int ci = grid.cidx(x, y, z);
                    const int cb = grid.cell_start[ci], ce = grid.cell_start[ci + 1];
                    if (cb >= ce) continue;                      // empty cell
                    {   // CONE REJECT: can ANY normal in this cell pass the gate? If not, skip all of it.
                        // This is what keeps a hole with no gate-passing neighbour from scanning ~10^6
                        // points on its way to the ring cap (measured: 400s+ -> seconds).
                        //
                        // Let A = angle(nh, axis), H = cone half-angle. The best a cell can do is
                        // max_dot = cos(max(0, A-H)), so it survives iff cos(max(0,A-H)) > mindot.
                        // TWO GUARDS BEFORE the trig-free form, both load-bearing (a brute-force test
                        // caught their absence as 15/2000 wrong answers + a wrecked gate-off control):
                        //   (1) A <= H  (D >= cosH): nh points INTO the cone -> max_dot = 1 -> keep.
                        //   (2) acos(mindot)+H >= pi  (cosH <= -mindot): the cone is so wide that some
                        //       normal passes for ANY nh -> keep. Without this the expansion below is
                        //       evaluated outside the range where cos is monotonically decreasing, and
                        //       cos(A) > cos(acos(md)+H) stops implying A < acos(md)+H -> over-rejection.
                        const float* cn = &grid.cone[(size_t)ci * 4];
                        const float cosH = cn[3];
                        const float D = nh[0]*cn[0] + nh[1]*cn[1] + nh[2]*cn[2];
                        if (!(D >= cosH || cosH <= -md)) {
                            const float sinH = std::sqrt(std::max(0.f, 1.f - cosH * cosH));
                            const float thr = md * cosH - s_md * sinH;   // = cos(acos(md) + H)
                            if (!(D > thr)) continue;
                        }
                    }
                    for (int ii = cb; ii < ce; ii++) {
                        const int q = grid.ids[ii];
                        // DISTANCE FIRST, normals second. Once we hold k candidates, `worst` prunes the
                        // overwhelming majority of points in ~6 flops, and touching nrm[] at all would be
                        // a second random cache line per point for nothing. (Measured: reordering this
                        // plus dropping the redundant re-normalize below is a ~3x cut on the fill.)
                        const float* QP = &pos[(size_t)q * 3];
                        const float dxp = QP[0]-P[0], dyp = QP[1]-P[1], dzp = QP[2]-P[2];
                        const float d2 = dxp*dxp + dyp*dyp + dzp*dzp;
                        if (cnt == k && !(d2 < worst)) continue;
                        // Normal-agreement gate. Both normals come from the SAME mesh, so the
                        // inward/outward convention cancels in the dot and no nsign is needed. This is
                        // what stops a hole on the inside of the jacket from stealing the outside
                        // surface's colour straight across the thin shell.
                        // nh and nrm[] are BOTH unit (raster_uv normalizes; callers must too), so the dot
                        // IS the cosine — no length division.
                        const float* Q = &nrm[(size_t)q * 3];
                        const float dot = nh[0]*Q[0] + nh[1]*Q[1] + nh[2]*Q[2];
                        if (!(dot > mindot)) continue;
                        if (cnt < k) {
                            out_d2[cnt] = d2; out_ids[cnt] = q; cnt++;
                            if (cnt == k) { worst = 0.f; for (int j = 0; j < cnt; j++) worst = std::max(worst, out_d2[j]); }
                        } else if (d2 < worst) {
                            int wj = 0; for (int j = 1; j < cnt; j++) if (out_d2[j] > out_d2[wj]) wj = j;
                            out_d2[wj] = d2; out_ids[wj] = q;
                            worst = 0.f; for (int j = 0; j < cnt; j++) worst = std::max(worst, out_d2[j]);
                        }
                    }
                }
            }
        }
    }
    return cnt;
}

// ---------------------------------------------------------------------------------------------
// debug dumps
// ---------------------------------------------------------------------------------------------
static inline void dump_png(const std::string& path, const uint8_t* d, int w, int h, int comp) {
    std::vector<uint8_t> png = glb::encode_png(d, w, h, comp);
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { std::fprintf(stderr, "[texproj] cannot write %s\n", path.c_str()); return; }
    std::fwrite(png.data(), 1, png.size(), f);
    std::fclose(f);
}

// z-buffer -> grayscale PNG: near = white, far = dark, uncovered background = black.
static inline void dump_zbuf(const std::string& path, const ZBuf& z) {
    float dmin = std::numeric_limits<float>::infinity(), dmax = -std::numeric_limits<float>::infinity();
    for (float d : z.d) { if (!std::isfinite(d)) continue; dmin = std::min(dmin, d); dmax = std::max(dmax, d); }
    if (!std::isfinite(dmin) || dmax <= dmin) { dmin = 0.f; dmax = 1.f; }
    std::vector<uint8_t> g((size_t)z.w * z.h, 0);
    for (size_t i = 0; i < z.d.size(); i++) {
        if (!std::isfinite(z.d[i])) continue;
        g[i] = u8f(1.f - (z.d[i] - dmin) / (dmax - dmin));   // near = 255
    }
    dump_png(path, g.data(), z.w, z.h, 1);
}

static inline void draw_rect(std::vector<uint8_t>& img, int W, int H, const BBox& b,
                             uint8_t r, uint8_t g, uint8_t bl, int thick) {
    if (!b.valid()) return;
    auto px = [&](int x, int y) {
        if (x < 0 || y < 0 || x >= W || y >= H) return;
        uint8_t* p = &img[((size_t)y * W + x) * 3]; p[0] = r; p[1] = g; p[2] = bl;
    };
    for (int t = 0; t < thick; t++) {
        for (int x = b.x0 - t; x <= b.x1 + t; x++) { px(x, b.y0 - t); px(x, b.y1 + t); }
        for (int y = b.y0 - t; y <= b.y1 + t; y++) { px(b.x0 - t, y); px(b.x1 + t, y); }
    }
}

// One camera + its image + its z-buffer + its alignment fit + its subject mask.
struct View {
    Cam cam;
    imgio::Image img;
    std::vector<float> alpha;
    bool has_alpha = false;
    bool is_front = false;         // yaw==0 AND it is the source matte (fitted too — see BUG 1)
    ZBuf z;
    Fit fit;
    BBox sil, sub;
    std::vector<uint8_t> subj;     // ERODED subject mask (BUG 2); empty = reject disabled for this view
    MaskStats ms;
    std::string path;
    int n_painted = 0;
    int n_bgrej = 0;               // texels dropped because they sampled outside `subj`
    double bgrej_facing_sum = 0, kept_facing_sum = 0;
    View(const Cam& c) : cam(c) {}
};

// ---------------------------------------------------------------------------------------------
// project_onto — OVERWRITES bt.base_color's RGB in place (alpha + bt.metal_rough preserved).
// Returns false on error (missing/unloadable front image, empty/mismatched atlas).
//
// ENV KNOBS (quality levers — the owner judges; nothing here declares a winner):
//   TEXPROJ_DEPTH_EPS      0.004   depth-test slack, in mesh units (mesh spans [-0.5,0.5])
//   TEXPROJ_NZ_LO          0.05    grazing-angle ramp start (confidence 0 below this facing)
//   TEXPROJ_NZ_HI          0.35    grazing-angle ramp end   (confidence 1 above this facing)
//   TEXPROJ_RASTER_SS      (TEX_RASTER_SS, else 1)   atlas raster supersample, 1..4
//   --- 3D hole fill (the anti-camo path) ---
//   TEXPROJ_FILL_3D        1       0 = OFF, i.e. the old UV-Telea-only behaviour (A/B lever)
//   TEXPROJ_FILL_MINDOT    0.3     normal-agreement gate: dot(n_hole, n_painted) must exceed this
//   TEXPROJ_FILL_K         6       k nearest painted texels blended (inverse-distance weights)
//   TEXPROJ_FILL_MAXDIST   0.08    how far a fill may reach, in MESH UNITS (mesh spans [-0.5,0.5]).
//                                  THE distance knob — grid-independent, unlike a raw ring count. It is
//                                  converted to a ring count (ceil(MAXDIST/cell)), and rings are cubic
//                                  shells, so the true reach is this along an axis and up to ~1.7x it on
//                                  the cell diagonal. It is a cheap termination bound, not an exact leash.
//   TEXPROJ_FILL_CELLCAP   12      max sources kept per grid cell (0 = uncapped). Perf-critical:
//                                  uncapped was 310s on a 9M-texel proxy vs ~2s capped. See PointGrid.
//   TEXPROJ_FILL_GRID      128     grid cells along the mesh's longest axis (cell = extent/this)
//   TEXPROJ_FILL_MAXRING   0       0 = derive rings from MAXDIST (correct). >0 forces a raw ring count
//                                  and makes the reach depend on TEXPROJ_FILL_GRID — debug only.
//   --- gutters + whatever the 3D fill could not serve ---
//   TEXPROJ_INPAINT_RINGS  8       telea rings for gutters + leftovers
//   TEXPROJ_INPAINT_RADIUS 3       telea estimation radius
//   TEXPROJ_NO_DILATE      unset   skip the nearest-valid background flood (leaves big holes black)
//   --- silhouette auto-align ---
//   TEXPROJ_BACK_ALIGN     1       0 = identity (no silhouette fit) for A/B; applies to ALL non-front views
//   TEXPROJ_FRONT_ALIGN    1       BUG 1 FIX. 0 = restore the old unfitted front ("[FRONT: exact framing]")
//   TEXPROJ_ALIGN_ANISO    0       1 = per-axis scale instead of uniform min(). A/B lever — see fit_similarity
//   --- BUG 3: the seam ---
//   TEXPROJ_SEAM_BLEND     0       1 = the 3D fill CROSS-FADES the views across the seam instead of running
//                                  one inverse-distance mean over a mixed neighbour set. Default OFF.
//   TEXPROJ_SEAM_BAND      0.35    cross-fade half-width in FACING units: weight = smoothstep(-band,+band,
//                                  facing). Bigger = the crossover is spread over more surface.
//   TEXPROJ_BACK_BG        black   NON-FRONT subject detector: `black` | `alpha` | `auto`.
//                                  `auto` = the file's own alpha when it has one, else `black`. MEASURED
//                                  the single biggest seam lever on the soldier — see BUG 3.
//   TEXPROJ_BACK_BG_THRESH 0.05    NON-FRONT `black` detector: background = max(r,g,b) <= thresh
//   TEXPROJ_FRONT_BG_THRESH 0.0039 FRONT `black` detector (= 1/255). MEASURED: the matte's background is
//                                  EXACTLY 0, and 0.05 would discard 7.9% of the subject — the BOOTS. See
//                                  THE LANDMINE at the top before raising this.
//   --- BUG 2: background-sample rejection ---
//   TEXPROJ_BG_REJECT      1       0 = OFF (old behaviour: grazing texels sample the black background)
//   TEXPROJ_BG_ERODE       2       erode the subject mask by this many SOURCE-IMAGE px before testing.
//                                  Must be >=1 to cover the bilinear tap footprint; 2 also clears the
//                                  matte's antialiased (already-darkened) rim.
//   TEXPROJ_BG_FILL_HOLES  1       background = only what the image BORDER can reach. Protects dark
//                                  materials enclosed by subject (the bearskin). 0 = raw threshold.
//   TEXPROJ_NORMAL_SIGN    auto    1|-1 forces the mesh normal convention (default = measure it)
//   TEXPROJ_DEBUG_DIR      unset   overrides Cfg::debug_dir
// ---------------------------------------------------------------------------------------------
inline bool project_onto(texatlas::BakedTexture& bt, const Cfg& cfg, Stats* out_stats = nullptr) {
    const int W = bt.tw, H = bt.th;
    if (W <= 0 || H <= 0 || bt.faces.empty() || bt.uvs.size() != (bt.verts.size() / 3) * 2) {
        std::fprintf(stderr, "[texproj] bad BakedTexture (tw=%d th=%d verts=%zu uvs=%zu faces=%zu)\n",
                     W, H, bt.verts.size() / 3, bt.uvs.size(), bt.faces.size() / 3);
        return false;
    }
    if (bt.base_color.size() != (size_t)W * H * 4) {
        std::fprintf(stderr, "[texproj] base_color size %zu != tw*th*4 (%zu)\n", bt.base_color.size(), (size_t)W * H * 4);
        return false;
    }
    if (cfg.front_img.empty()) { std::fprintf(stderr, "[texproj] front_img is required\n"); return false; }

    const float eps   = envf("TEXPROJ_DEPTH_EPS", 0.004f);
    const float nz_lo = envf("TEXPROJ_NZ_LO", 0.05f);
    const float nz_hi = envf("TEXPROJ_NZ_HI", 0.35f);
    int ss = envi("TEXPROJ_RASTER_SS", envi("TEX_RASTER_SS", 1));
    ss = std::max(1, std::min(4, ss));
    const bool do_align = envi("TEXPROJ_BACK_ALIGN", 1) != 0;
    const bool front_align = envi("TEXPROJ_FRONT_ALIGN", 1) != 0;   // BUG 1: the front DOES need the fit
    const bool align_aniso = envi("TEXPROJ_ALIGN_ANISO", 0) != 0;
    // `auto` = use the file's OWN alpha when it has one, else fall back to `black`. See BUG 3: on the
    // soldier's flux2 back view this is the single largest measured seam contributor, because `black` at
    // 0.05 eats the bearskin and the boots exactly where they touch the silhouette. Default stays `black`
    // so this cannot silently change an existing run.
    // DEFAULT = `auto` since 2026-07-17: OWNER-JUDGED on the eye-test 4-way (A/B/C/D). `auto` uses the
    // file's own alpha when it has one, else `black`. Measured on the flux2-generated back view:
    // silhouette-vs-subject IoU alpha 0.945 vs black@0.05 0.857 (black refuses 12.37% of the back
    // silhouette — the bearskin and boots, which touch the outline so the flood-fill guard can't save
    // them). Back coverage 18.0% -> 20.4%, and the grazing falsifier flips WARN -> OK. `black` remains
    // right for a COMPOSITED matte (exact-zero background); `auto` picks per-view instead of guessing.
    const char* bg_env = std::getenv("TEXPROJ_BACK_BG");
    const std::string bg_mode = bg_env ? bg_env : "auto";
    const float bg_thresh = envf("TEXPROJ_BACK_BG_THRESH", 0.05f);
    // The FRONT matte is composited on EXACT black (measured: 77.35% of soldier_matte.png is (0,0,0)), so
    // its detector only needs to beat rounding. 0.05 here would eat the boots — see THE LANDMINE.
    const float front_bg_thresh = envf("TEXPROJ_FRONT_BG_THRESH", 1.f / 255.f);
    const bool  bg_reject = envi("TEXPROJ_BG_REJECT", 1) != 0;      // BUG 2
    const int   bg_erode  = std::max(0, envi("TEXPROJ_BG_ERODE", 2));
    const bool  bg_holes  = envi("TEXPROJ_BG_FILL_HOLES", 1) != 0;
    const char* dbg_env = std::getenv("TEXPROJ_DEBUG_DIR");
    const std::string debug_dir = dbg_env ? dbg_env : cfg.debug_dir;
    // --- BUG 3: the seam. TEXPROJ_SEAM_BLEND makes the 3D fill CROSS-FADE the views instead of
    // running one inverse-distance mean over a mixed neighbour set.
    // DEFAULT = ON since 2026-07-17: OWNER-JUDGED (config C on the eye-test 4-way). Verdict verbatim:
    // "seam kind of moves in C/B, blends better on the worse arm but worse on the better arm ... so its
    // kind of on average cleaner ... C is probably the sane default even if its a little 'not perfect'."
    // NOTE what this can and cannot do: front paints facing>+0.05 and back facing<-0.05 and the two
    // views overlap in EXACTLY ZERO texels (facing_back == -facing_front identically), so nothing is
    // ever "blended" between them — the ~20-25deg strip between is 100% HOLE, invented by the k-NN fill.
    // This makes that invented strip RAMP instead of FLIP. It cannot fix the strip being DARK: both
    // images' rim pixels are the shaded, foreshortened grazing edge of a LIT render, and projecting a
    // lit image bakes that in. That needs de-lighting, not blending. (`hunyuan3d-delight-v2-0` is on
    // disk, 4.1GB; or estimate shading as lowpass(projected / volume_albedo) — TRELLIS's baseColor IS
    // unlit albedo — and divide it out, which needs no new model.)
    const bool  seam_blend = envi("TEXPROJ_SEAM_BLEND", 1) != 0;
    const float seam_band  = envf("TEXPROJ_SEAM_BAND", 0.35f);
    const bool  fill3d   = envi("TEXPROJ_FILL_3D", 1) != 0;
    const float fill_dot = envf("TEXPROJ_FILL_MINDOT", 0.3f);
    const int   fill_k   = std::max(1, std::min(32, envi("TEXPROJ_FILL_K", 6)));
    const int   fill_grid= envi("TEXPROJ_FILL_GRID", 128);
    const int   fill_cap = std::max(0, envi("TEXPROJ_FILL_CELLCAP", 12));
    // The search radius is specified in WORLD units and converted to rings once the cell size is known.
    // TEXPROJ_FILL_MAXRING still forces a raw ring count, but it must NOT be the primary knob: rings are
    // grid-relative, so tuning the grid would silently change how far a fill may reach. Measured: at a
    // fixed 12 rings, grid 128 -> 320 dropped the 3D-fill share from 69% to 44% purely because the cells
    // got smaller. In world units the share holds (69.0% vs 68.3%) across grids, as it must.
    const float fill_dist = envf("TEXPROJ_FILL_MAXDIST", 0.08f);   // mesh spans [-0.5,0.5]
    const int   fill_ring_force = envi("TEXPROJ_FILL_MAXRING", 0); // 0 = derive from MAXDIST

    Stats st;

    // ---- assemble the view list: front (yaw 0, the matte) + back sugar (yaw 180) + any extra views ----
    std::vector<ViewSpec> specs;
    specs.push_back({0.f, cfg.front_img});
    if (!cfg.back_img.empty()) specs.push_back({180.f, cfg.back_img});
    for (const ViewSpec& v : cfg.views) specs.push_back(v);

    std::vector<View> views;
    views.reserve(specs.size());
    for (size_t i = 0; i < specs.size(); i++) {
        View v(Cam(cfg.cam, cfg.dist, cfg.ms, specs[i].yaw_deg));
        v.path = specs[i].img;
        v.is_front = (i == 0);
        // ALL views go through the RGBA loader now: the front needs subject detection too (BUG 1's fit and
        // BUG 2's mask), so it needs an honest has_alpha like everyone else. Numerically identical to
        // load_rgb01 for the RGB channels — stb's channel conversion does not depend on req_comp.
        if (!load_rgba01(v.path, v.img, v.alpha, v.has_alpha)) return false;
        views.push_back(std::move(v));
    }
    const int NV = (int)views.size();
    // index of the yaw=180 view, if any — Stats' back_* fields report ITS fit (API compat)
    int back_i = -1;
    for (int i = 1; i < NV; i++) if (views[i].cam.yaw_deg == 180.f) { back_i = i; break; }

    if (cfg.verbose) {
        for (int i = 0; i < NV; i++)
            std::printf("[texproj] view %d: yaw=%6.1fdeg  %s (%dx%d)%s%s\n", i, views[i].cam.yaw_deg,
                        views[i].path.c_str(), views[i].img.w, views[i].img.h,
                        views[i].is_front ? (front_align ? "  [FRONT: silhouette-fitted]"
                                                         : "  [FRONT: exact framing, no fit — TEXPROJ_FRONT_ALIGN=0]")
                                          : "",
                        views[i].has_alpha ? "  [file HAS alpha]" : "");
        if (NV == 1) std::printf("[texproj] only the front view — everything it can't see becomes a hole\n");
        std::printf("[texproj] cam fov=%.4frad dist=%.4f scale=%.2f  atlas=%dx%d  ss=%d  eps=%.4f nz=[%.2f,%.2f]\n",
                    cfg.cam, cfg.dist, cfg.ms, W, H, ss, eps, nz_lo, nz_hi);
        // (the derived ring count + actual grid dims are printed once the grid exists, below)
        if (fill3d) std::printf("[texproj] 3D hole fill: ON  k=%d mindot=%.2f maxdist=%.3f grid=%d cellcap=%d\n",
                                fill_k, fill_dot, fill_dist, fill_grid, fill_cap);
        else        std::printf("[texproj] 3D hole fill: OFF (TEXPROJ_FILL_3D=0) — atlas-space Telea only\n");
        if (bg_reject) std::printf("[texproj] bg-sample reject: ON  erode=%dpx holefill=%d  thresh front=%.4f "
                                   "non-front=%.4f  align: front=%d aniso=%d\n",
                                   bg_erode, (int)bg_holes, front_bg_thresh, bg_thresh, (int)front_align, (int)align_aniso);
        else           std::printf("[texproj] bg-sample reject: OFF (TEXPROJ_BG_REJECT=0) — grazing texels may "
                                   "sample background  align: front=%d aniso=%d\n", (int)front_align, (int)align_aniso);
        std::fflush(stdout);
    }
    if (bg_mode == "alpha") for (int i = 1; i < NV; i++) if (!views[i].has_alpha)
        std::fprintf(stderr, "[texproj] WARN: TEXPROJ_BACK_BG=alpha but %s has no alpha channel — falling back to `black`\n",
                     views[i].path.c_str());
    // `auto` resolves per view, so it can never be the wrong answer for a view that lacks alpha.
    if (bg_mode == "auto" && cfg.verbose) for (int i = 1; i < NV; i++)
        std::printf("[texproj] view %d BACK_BG=auto -> `%s` (%s)\n", i,
                    views[i].has_alpha ? "alpha" : "black",
                    views[i].has_alpha ? "the file carries an alpha matte — trust it over a luminance guess"
                                       : "no alpha in the file");

    // ---- 1. UV raster ----
    double t0 = texatlas::_now();
    std::vector<float> pos, nrm; std::vector<uint8_t> mask;
    raster_uv(bt, ss, pos, nrm, mask);
    int covered = 0; for (uint8_t m : mask) covered += m;
    st.t_uv_raster = texatlas::_now() - t0;
    if (cfg.verbose) std::printf("[texproj] uv raster: %d / %d texels covered (%.1f%%, %.2fs)\n",
                                 covered, W * H, 100.0 * covered / (double)(W * H), st.t_uv_raster);
    if (!covered) { std::fprintf(stderr, "[texproj] atlas raster covered 0 texels\n"); return false; }

    // ---- 2. depth prepass (one z-buffer per view, at that view image's own resolution) ----
    for (int i = 0; i < NV; i++) {
        t0 = texatlas::_now();
        views[i].z = raster_depth(bt.verts, bt.faces, views[i].cam, views[i].img.w, views[i].img.h);
        const double dt = texatlas::_now() - t0;
        if (i == 0) st.t_zbuf_front = dt;
        else if (i == back_i) st.t_zbuf_back = dt;
        if (cfg.verbose)
            std::printf("[texproj] depth prepass view %d (yaw %.1f): %zu tris -> %dx%d  (%.2fs)\n",
                        i, views[i].cam.yaw_deg, bt.faces.size() / 3, views[i].z.w, views[i].z.h, dt);
    }

    // ---- 2b. measure the mesh's normal convention against the FRONT z-buffer (NEVER assume it) ----
    {
        const int forced = envi("TEXPROJ_NORMAL_SIGN", 0);
        float s;
        if (forced == 1 || forced == -1) {
            s = (float)forced;
            if (cfg.verbose) std::printf("[texproj] normal convention: FORCED %+d via TEXPROJ_NORMAL_SIGN\n", forced);
        } else {
            double sp = 0, sn = 0;
            s = detect_normal_sign(bt, views[0].z, views[0].cam, eps, cfg.verbose, &sp, &sn);
        }
        for (View& v : views) v.cam.nsign = s;
        st.normal_sign = s;
    }

    // ---- 3. per-view subject mask + silhouette-bbox auto-align. THE FRONT IS INCLUDED NOW (BUG 1). ----
    // The mask serves BOTH jobs, so they cannot disagree: its bbox is the align target, and its eroded form
    // is BUG 2's background-sample reject. A normal geometry matte is black-composited, while --tex-front
    // can be an RGBA cutout in the SAME frame; that alpha is the only reliable subject mask for raw inputs.
    t0 = texatlas::_now();
    for (int i = 0; i < NV; i++) {
        View& v = views[i];
        const bool is_front = v.is_front;
        // `auto` -> the file's own alpha when it has one. A generated view's RGB background is NOT exact
        // black (flux2 output carries 1-2/255 dither), so a luminance threshold is a guess; its alpha, when
        // RMBG wrote one, is the real matte. MEASURED on the soldier's back_v1_rgba.png: black@0.05 gives
        // silhouette-vs-subject IoU 0.857 and throws away 12.37% of the mesh's back silhouette (the
        // bearskin + boots, which touch the outline so the hole-fill guard cannot save them); alpha gives
        // IoU 0.945 / 2.73%, i.e. essentially the front's own 0.971 / 1.40%.
        const std::string mode = is_front ? (v.has_alpha ? std::string("alpha") : std::string("black"))
                               : (bg_mode == "auto" ? (v.has_alpha ? std::string("alpha") : std::string("black"))
                                                    : bg_mode);
        const float thr = is_front ? front_bg_thresh : bg_thresh;
        const bool want_align = is_front ? front_align : do_align;
        if (!want_align && !bg_reject) {
            if (cfg.verbose)
                std::printf("[texproj] view %d align: DISABLED (%s=0) — identity, no mask\n", i,
                            is_front ? "TEXPROJ_FRONT_ALIGN" : "TEXPROJ_BACK_ALIGN");
            continue;
        }
        subject_mask(v.img, v.alpha, v.has_alpha, mode, thr, bg_holes, v.subj, &v.ms);
        const size_t NP = (size_t)v.img.w * v.img.h;
        if (cfg.verbose) {
            if (v.ms.mode_is_black)
                // The luminance detector: n_lost IS the LANDMINE number (dark subject eaten by thresh).
                std::printf("[texproj] view %d mask (%s thresh=%.4f holefill=%d): subject %d px (%.1f%% of image), "
                            "hole-fill reclaimed %d, DISCARDED-nonzero %d (%.2f%% of nonzero)%s\n",
                            i, mode.c_str(), thr, (int)bg_holes, v.ms.n_subject, 100.0 * v.ms.n_subject / (double)NP,
                            v.ms.n_holefill, v.ms.n_lost,
                            v.ms.n_nonzero ? 100.0 * v.ms.n_lost / v.ms.n_nonzero : 0.0,
                            (v.ms.n_nonzero && 100.0 * v.ms.n_lost / v.ms.n_nonzero > 2.0)
                                ? "  [WARN: threshold is eating DARK SUBJECT (boots/hair?) — lower it]" : "");
            else
                // The alpha detector reads RMBG's own matte, so "nonzero RGB" is not evidence of anything
                // and MUST NOT be warned on — see MaskStats. A generated view's background dither makes
                // that number ~64% and the old WARN told you to lower a threshold this mode doesn't use.
                std::printf("[texproj] view %d mask (%s, holefill=%d): subject %d px (%.1f%% of image) from the "
                            "file's own alpha matte — thresh/nonzero stats N/A for this detector\n",
                            i, mode.c_str(), (int)bg_holes, v.ms.n_subject, 100.0 * v.ms.n_subject / (double)NP);
        }
        if (v.ms.n_subject > (int)(0.9 * NP))
            std::fprintf(stderr, "[texproj] WARN: view %d subject mask covers %.1f%% of the image — the "
                                 "background is probably NOT near-black, so the bbox fit and the bg-reject "
                                 "are both meaningless for this view. Raise its BG_THRESH.\n",
                         i, 100.0 * v.ms.n_subject / (double)NP);

        if (want_align) {
            v.sil = silhouette_bbox(v.z);
            v.sub = bbox_of_mask(v.subj, v.img.w, v.img.h);
            v.fit = fit_similarity(v.sil, v.sub, align_aniso);
            if (!v.fit.fitted) {
                std::fprintf(stderr, "[texproj] WARN: view %d (yaw %.1f) auto-align SKIPPED (degenerate bbox: "
                                     "silhouette %.0fx%.0f, subject %.0fx%.0f) — using identity\n",
                             i, v.cam.yaw_deg, v.sil.valid() ? v.sil.w() : 0.f, v.sil.valid() ? v.sil.h() : 0.f,
                             v.sub.valid() ? v.sub.w() : 0.f, v.sub.valid() ? v.sub.h() : 0.f);
            } else if (cfg.verbose) {
                std::printf("[texproj] view %d align: silhouette [%d,%d..%d,%d] %.0fx%.0f  subject [%d,%d..%d,%d] %.0fx%.0f\n",
                            i, v.sil.x0, v.sil.y0, v.sil.x1, v.sil.y1, v.sil.w(), v.sil.h(),
                            v.sub.x0, v.sub.y0, v.sub.x1, v.sub.y1, v.sub.w(), v.sub.h());
                std::printf("[texproj] view %d align: scale=%.4f/%.4f (w %.4f / h %.4f -> %s)  translate=(%+.2f, %+.2f) px%s\n",
                            i, v.fit.sx, v.fit.sy, v.sub.w() / v.sil.w(), v.sub.h() / v.sil.h(),
                            align_aniso ? "aniso" : "min", v.fit.tx, v.fit.ty,
                            is_front ? "   <- FRONT (BUG 1: expected ~1.009 on the soldier)" : "");
            }
        } else if (cfg.verbose) {
            std::printf("[texproj] view %d align: DISABLED (%s=0) — identity (mask still built for bg-reject)\n",
                        i, is_front ? "TEXPROJ_FRONT_ALIGN" : "TEXPROJ_BACK_ALIGN");
        }

        // Erode LAST: the align bbox must come from the un-eroded mask (eroding first would shrink the
        // subject bbox by ~2px on every side and bias the fitted scale DOWN by ~0.7% — which is the same
        // order as the drift we are correcting, i.e. it would silently eat BUG 1's fix).
        if (bg_reject) {
            const int before = v.ms.n_subject;
            erode_mask(v.subj, v.img.w, v.img.h, bg_erode);
            int after = 0; for (uint8_t m : v.subj) after += m;
            v.ms.n_eroded = before - after;
            if (cfg.verbose)
                std::printf("[texproj] view %d mask: erode %dpx removed %d -> %d subject px (rim %.2f%%)\n",
                            i, bg_erode, v.ms.n_eroded, after, before ? 100.0 * v.ms.n_eroded / before : 0.0);
        } else {
            v.subj.clear();          // reject disabled: an empty mask means "never reject" downstream
        }
    }
    st.t_align = texatlas::_now() - t0;
    if (back_i >= 0) {
        st.back_aligned = views[back_i].fit.fitted;
        st.back_scale = views[back_i].fit.sx; st.back_tx = views[back_i].fit.tx; st.back_ty = views[back_i].fit.ty;
    }

    // ---- 4-6. per texel: project through EVERY view, occlusion-test, confidence-weight, blend (LINEAR) ----
    t0 = texatlas::_now();
    std::vector<float> rgb_lin((size_t)W * H * 3, 0.f);
    // Decode before projected samples replace it. This is the already-baked volume atlas, retained only
    // where no real supplied view can vouch for the surface.
    std::vector<float> base_lin;
    if (cfg.preserve_base_for_holes) {
        base_lin.resize((size_t)W * H * 3);
        #pragma omp parallel for schedule(static)
        for (int p = 0; p < W * H; p++)
            for (int c = 0; c < 3; c++)
                base_lin[(size_t)p * 3 + c] = srgb_to_linear(bt.base_color[(size_t)p * 4 + c] / 255.f);
    }
    std::vector<uint8_t> valid((size_t)W * H, 0);
    // per-texel provenance for the fill-source debug PNG: 0=none 1=projected 2=3D-filled 3=telea/dilate
    std::vector<uint8_t> src_of((size_t)W * H, 0);
    // WHICH view painted this texel (255 = none). The seam cross-fade and the seam map both need to know
    // a painted texel's provenance, not just that it was painted. NV is tiny so a byte is plenty.
    // Costs W*H bytes — the same as src_of, which is already unconditional.
    std::vector<uint8_t> view_of((size_t)W * H, 255);
    std::vector<uint8_t> dbg;                       // RGB: R=view0 conf, G=max other-view conf, B=hole flag
    if (!debug_dir.empty()) dbg.assign((size_t)W * H * 3, 0);
    std::vector<int> n_view(NV, 0), n_bgrej(NV, 0);
    std::vector<double> f_bgrej(NV, 0.0), f_kept(NV, 0.0);
    // BUG-2 debug map buffers — only worth their memory when we are actually dumping (a 4096^2 atlas would
    // otherwise burn ~84MB of float+byte per run for nothing). Per-texel writes are race-free: the omp for
    // below partitions `p`, so exactly one thread owns each texel.
    const bool want_bgmap = bg_reject && !debug_dir.empty();
    std::vector<uint8_t> bgrej_any;    // this texel lost >=1 view to the bg reject
    std::vector<float>   bgrej_maxf;   // the HIGHEST facing among the views that rejected it
    if (want_bgmap) { bgrej_any.assign((size_t)W * H, 0); bgrej_maxf.assign((size_t)W * H, 0.f); }
    int n_hole = 0;
    const int T = W * H;
    #pragma omp parallel
    {
        std::vector<int> loc(NV, 0), loc_rej(NV, 0);
        std::vector<double> locf_rej(NV, 0.0), locf_kept(NV, 0.0);
        int loc_hole = 0;
        #pragma omp for schedule(dynamic, 1024) nowait
        for (int p = 0; p < T; p++) {
            if (!mask[(size_t)p]) continue;                 // gutter / background: not our texel
            const float* P = &pos[(size_t)p * 3];
            const float* N = &nrm[(size_t)p * 3];

            float wsum = 0.f, acc[3] = {0, 0, 0};
            float c0 = 0.f, cmax_other = 0.f;
            float best_c = 0.f; int best_v = 255;   // the highest-confidence view = this texel's provenance
            for (int i = 0; i < NV; i++) {
                const View& v = views[i];
                float u, vv; const float d = v.cam.project(P, u, vv);
                const float nz = v.cam.facing(N);
                if (!(nz > 0.f) || !visible(v.z, u, vv, d, eps)) continue;
                const float c = smoothstep(nz_lo, nz_hi, nz);
                if (!(c > 0.f)) continue;
                // The occlusion test stays in SILHOUETTE space (the z-buffer is the mesh's own depth);
                // only the image lookup goes through the fitted similarity. EVERY view is fitted now,
                // including the front — see BUG 1.
                float su = u, sv = vv;
                if (v.fit.fitted) {
                    float ix, iy; v.fit.apply(u * v.img.w, vv * v.img.h, ix, iy);
                    su = ix / (float)v.img.w; sv = iy / (float)v.img.h;
                }
                // BUG 2 — BACKGROUND-SAMPLE REJECT. Sampling here would bilinear-blend the matte's black
                // background into a silhouette texel and paint it black; the 3D fill would then propagate
                // that black down the side ("black streaks, ears -> hands"). Contribute ZERO instead —
                // identical to failing the z-test — and let the hole fill serve it from a confident
                // neighbour. Pixel index convention matches visible(): floor(u*W).
                if (!v.subj.empty()) {
                    const int ix = (int)std::floor(su * (float)v.img.w);
                    const int iy = (int)std::floor(sv * (float)v.img.h);
                    if (ix < 0 || iy < 0 || ix >= v.img.w || iy >= v.img.h ||
                        !v.subj[(size_t)iy * v.img.w + ix]) {
                        loc_rej[i]++; locf_rej[i] += nz;
                        if (want_bgmap) {
                            bgrej_any[(size_t)p] = 1;
                            bgrej_maxf[(size_t)p] = std::max(bgrej_maxf[(size_t)p], nz);
                        }
                        continue;
                    }
                }
                float col[3]; sample_bilinear(v.img, su, sv, col);
                for (int k = 0; k < 3; k++) acc[k] += c * srgb_to_linear(col[k]);
                wsum += c;
                loc[i]++; locf_kept[i] += nz;
                if (c > best_c) { best_c = c; best_v = i; }
                if (i == 0) c0 = c; else cmax_other = std::max(cmax_other, c);
            }

            if (wsum > 1e-4f) {
                // Blend in LINEAR light (a confidence-weighted mean of sRGB bytes would darken the seam).
                // The sRGB->linear->sRGB round-trip is lossless to <1/255 for a single-source texel — and
                // note that for a single view the weight CANCELS (acc/wsum = the sample itself), so the
                // grazing ramp never tints a texel only one camera can see; it only sets hole-vs-painted
                // and the relative weight between overlapping views.
                float* o = &rgb_lin[(size_t)p * 3];
                for (int c = 0; c < 3; c++) o[c] = acc[c] / wsum;
                valid[(size_t)p] = 1;
                src_of[(size_t)p] = 1;
                view_of[(size_t)p] = (uint8_t)best_v;
            } else {
                loc_hole++;   // no view can see this texel (under arms, occluded, sides with no side view)
            }
            if (!dbg.empty()) {
                dbg[(size_t)p * 3 + 0] = u8f(c0);
                dbg[(size_t)p * 3 + 1] = u8f(cmax_other);
                dbg[(size_t)p * 3 + 2] = valid[(size_t)p] ? 0 : 255;
            }
        }
        #pragma omp critical
        {
            for (int i = 0; i < NV; i++) {
                n_view[i] += loc[i]; n_bgrej[i] += loc_rej[i];
                f_bgrej[i] += locf_rej[i]; f_kept[i] += locf_kept[i];
            }
            n_hole += loc_hole;
        }
    }
    st.t_project = texatlas::_now() - t0;
    if (cfg.verbose) {
        for (int i = 0; i < NV; i++) {
            std::printf("[texproj] project: view %d (yaw %6.1f) painted %8d (%5.1f%%)\n",
                        i, views[i].cam.yaw_deg, n_view[i], 100.0 * n_view[i] / covered);
            // THE BUG-2 SIGNATURE. `mean facing` of the rejected set is the whole diagnosis: these texels
            // are supposed to be the ones whose normal GRAZES the camera, i.e. mean facing well BELOW the
            // kept set's (and near nz_lo=%.2f). If the rejected set's facing is comparable to the kept
            // set's, the reject is firing in the SURFACE INTERIOR, not at the silhouette — which means the
            // subject mask or the fit is wrong, NOT that we found the streaks. Say so loudly if so.
            if (bg_reject) {
                const double mf_rej = n_bgrej[i] ? f_bgrej[i] / n_bgrej[i] : 0.0;
                const double mf_kep = n_view[i]  ? f_kept[i]  / n_view[i]  : 0.0;
                const int tested = n_view[i] + n_bgrej[i];
                std::printf("[texproj] project: view %d bg-rejected %8d (%5.2f%% of %d that passed facing+depth)"
                            "  mean facing: rejected %.3f vs kept %.3f%s\n",
                            i, n_bgrej[i], tested ? 100.0 * n_bgrej[i] / tested : 0.0, tested, mf_rej, mf_kep,
                            (n_bgrej[i] > 100 && mf_rej > 0.5 * mf_kep)
                                ? "   [WARN: rejected texels are NOT grazing — mask/fit suspect, not streaks]"
                                : (n_bgrej[i] > 100 ? "   [grazing signature OK]" : ""));
            }
        }
        std::printf("[texproj] project: holes %d (%.1f%%) of %d covered (%.2fs)\n",
                    n_hole, 100.0 * n_hole / covered, covered, st.t_project);
    }
    for (int i = 0; i < NV; i++) {
        views[i].n_painted = n_view[i]; views[i].n_bgrej = n_bgrej[i];
        views[i].bgrej_facing_sum = f_bgrej[i]; views[i].kept_facing_sum = f_kept[i];
    }

    // A front photo cannot see behind hair, under an arm, or around the back. Do not turn those visibility
    // holes into nearest-neighbour/Telea noise: retain the coherent volume albedo. Supplied back/side
    // views still replace it normally above. `n_hole` stays the honest source-coverage metric.
    int n_base = 0;
    if (cfg.preserve_base_for_holes) {
        #pragma omp parallel for reduction(+:n_base) schedule(static)
        for (int p = 0; p < T; p++) {
            if (!mask[(size_t)p] || valid[(size_t)p]) continue;
            for (int c = 0; c < 3; c++) rgb_lin[(size_t)p * 3 + c] = base_lin[(size_t)p * 3 + c];
            valid[(size_t)p] = 1;
            src_of[(size_t)p] = 4;
            n_base++;
        }
        if (cfg.verbose)
            std::printf("[texproj] hybrid fallback: retained volume base for %d / %d unobserved texels (%.1f%%)\n",
                        n_base, n_hole, n_hole ? 100.0 * n_base / n_hole : 0.0);
    }

    // ---- 6b. 3D-AWARE HOLE FILL: k-nearest painted texel in 3D, normal-gated, inverse-distance blend ----
    int n_fill3d = 0;
    // BUG 3 accounting. seam_of marks the fill texels that see >=2 views — i.e. THE SEAM, the only place
    // front and back ever meet (the projection's overlap band is provably empty; see the cross-fade
    // comment below). loc_sd = mean |view_a - view_b|, loc_ss = the SIGNED mean, both in sRGB units.
    // The signed/unsigned split is the whole diagnosis: |d| is what you SEE, and the signed mean is the
    // only part a global colour match could ever remove.
    long loc_seam = 0;
    double loc_sd[3] = {0, 0, 0}, loc_ss[3] = {0, 0, 0};
    std::vector<uint8_t> seam_of((size_t)W * H, 0);
    t0 = texatlas::_now();
    if (fill3d && n_hole > 0) {
        std::vector<int> painted, holes;
        painted.reserve(covered - n_hole); holes.reserve(n_hole);
        for (int p = 0; p < T; p++) {
            if (!mask[(size_t)p]) continue;
            if (valid[(size_t)p]) painted.push_back(p); else holes.push_back(p);
        }
        if (!painted.empty()) {
            PointGrid grid(painted, pos, nrm, fill_grid, fill_cap);
            const int fill_ring = fill_ring_force > 0
                                ? fill_ring_force
                                : std::max(1, (int)std::ceil(fill_dist / std::max(grid.cell, 1e-9f)));
            if (cfg.verbose)
                std::printf("[texproj] 3D fill grid: %dx%dx%d cells, cell=%.5f, %zu sources (cap %d/cell), "
                            "maxdist=%.3f -> %d rings\n", grid.nx, grid.ny, grid.nz, grid.cell,
                            grid.ids.size(), fill_cap, fill_dist, fill_ring);
            // Write into a side buffer + flag array so the grid's source set stays EXACTLY the originally
            // painted texels: a hole must never seed another hole (that would smear colour arbitrarily far
            // and make the result thread-order dependent). Deterministic, race-free.
            std::vector<float> fill_rgb((size_t)holes.size() * 3, 0.f);
            std::vector<uint8_t> got((size_t)holes.size(), 0);
            const int NH = (int)holes.size();
            #pragma omp parallel for schedule(dynamic, 512) reduction(+:n_fill3d,loc_seam) \
                reduction(+:loc_sd[:3]) reduction(+:loc_ss[:3])
            for (int hi = 0; hi < NH; hi++) {
                const int p = holes[hi];
                const float* P = &pos[(size_t)p * 3];
                const float* N = &nrm[(size_t)p * 3];
                const float nl = std::sqrt(N[0]*N[0] + N[1]*N[1] + N[2]*N[2]);
                if (!(nl > 1e-12f)) continue;              // degenerate normal -> leave for telea
                const float nh[3] = {N[0]/nl, N[1]/nl, N[2]/nl};

                // bounded k-best by squared distance (k <= 32, so a linear insert beats a heap)
                float bd[32]; int bi[32];
                const int cnt = knn_gated(grid, pos, nrm, P, nh, fill_k, fill_dot, fill_ring, bi, bd);
                if (!cnt) continue;                        // gate + ring cap found nothing -> telea's turn
                // Inverse-distance (Shepard) blend in LINEAR light. k-nearest rather than single-nearest:
                // one nearest paints visible Voronoi facets, and "blotchy misc colouring" is the bug.
                //
                // BUG 3 (THE SEAM) — split the SAME k-NN set by WHICH VIEW painted each neighbour. This is
                // the only place in the whole file where the front and the back actually meet: a hole on
                // the silhouette has front-painted neighbours on one side and back-painted ones on the
                // other. Accumulating per-view costs one extra array and nothing else.
                float acc[3] = {0, 0, 0}, wsum = 0.f;
                float vacc[TEXPROJ_MAX_SEAM_VIEWS][3] = {{0}}, vw[TEXPROJ_MAX_SEAM_VIEWS] = {0};
                int n_src_views = 0;
                for (int j = 0; j < cnt; j++) {
                    const float w = 1.f / (bd[j] + 1e-12f);
                    const float* s = &rgb_lin[(size_t)bi[j] * 3];
                    for (int c = 0; c < 3; c++) acc[c] += w * s[c];
                    wsum += w;
                    const uint8_t vo = view_of[(size_t)bi[j]];
                    if (vo < TEXPROJ_MAX_SEAM_VIEWS) {
                        if (vw[vo] == 0.f) n_src_views++;
                        for (int c = 0; c < 3; c++) vacc[vo][c] += w * s[c];
                        vw[vo] += w;
                    }
                }
                if (!(wsum > 0.f)) continue;
                // SEAM METRIC (always measured, even when the fix is off): this hole sees >=2 views, so it
                // IS the seam. |view_a - view_b| is how far apart the two images are where they meet — the
                // number a colour match would have to shrink, and the number the stage line reports.
                if (n_src_views >= 2) {
                    int va = -1, vb = -1;
                    for (int i = 0; i < NV && i < TEXPROJ_MAX_SEAM_VIEWS; i++)
                        if (vw[i] > 0.f) { if (va < 0) va = i; else if (vb < 0) vb = i; }
                    if (va >= 0 && vb >= 0) {
                        loc_seam++;
                        for (int c = 0; c < 3; c++) {
                            const float ca = vacc[va][c] / vw[va], cb = vacc[vb][c] / vw[vb];
                            loc_sd[c] += std::fabs((double)linear_to_srgb(ca) - linear_to_srgb(cb));
                            loc_ss[c] += (double)linear_to_srgb(ca) - linear_to_srgb(cb);
                        }
                    }
                }
                if (!seam_blend || n_src_views < 2) {
                    for (int c = 0; c < 3; c++) fill_rgb[(size_t)hi * 3 + c] = acc[c] / wsum;
                } else {
                    // CROSS-FADE. One inverse-distance mean over a mixed set makes the transition a
                    // nearest-neighbour lottery: whichever side happens to be closest wins, so the colour
                    // flips rather than ramps and the strip mottles. Instead: reconstruct each view's OWN
                    // estimate of this point's colour from its own neighbours, then weight the estimates by
                    // how much this texel's normal faces that view. smoothstep(-band,+band,facing) is a
                    // partition of unity across the silhouette (0.5/0.5 at facing=0, pure at +-band), so
                    // the result is guaranteed MONOTONE from front to back.
                    //
                    // WHY THE RAMP MUST LIVE HERE AND NOT IN THE PROJECTION: facing_back == -facing_front
                    // exactly (Cam::facing reduces to +-n.z*nsign at yaw 0/180), so no texel can ever have
                    // both confidences > 0 — the projection's confidence-weighted mean has an EMPTY overlap
                    // band and cannot blend anything. Measured: 0 of 165296 verts. The k-NN neighbourhood is
                    // the only overlap that exists.
                    float g[TEXPROJ_MAX_SEAM_VIEWS], gsum = 0.f;
                    for (int i = 0; i < NV && i < TEXPROJ_MAX_SEAM_VIEWS; i++) {
                        g[i] = 0.f;
                        if (!(vw[i] > 0.f)) continue;
                        g[i] = smoothstep(-seam_band, seam_band, views[i].cam.facing(nh));
                        gsum += g[i];
                    }
                    if (!(gsum > 1e-6f)) {   // every contributing view grazes past the band -> plain mean
                        for (int c = 0; c < 3; c++) fill_rgb[(size_t)hi * 3 + c] = acc[c] / wsum;
                    } else {
                        for (int c = 0; c < 3; c++) {
                            float o = 0.f;
                            for (int i = 0; i < NV && i < TEXPROJ_MAX_SEAM_VIEWS; i++)
                                if (vw[i] > 0.f) o += g[i] * (vacc[i][c] / vw[i]);
                            fill_rgb[(size_t)hi * 3 + c] = o / gsum;
                        }
                    }
                }
                if (n_src_views >= 2) seam_of[(size_t)p] = 1;
                got[hi] = 1;
                n_fill3d++;
            }
            // merge (serial, deterministic)
            for (int hi = 0; hi < NH; hi++) {
                if (!got[hi]) continue;
                const int p = holes[hi];
                for (int c = 0; c < 3; c++) rgb_lin[(size_t)p * 3 + c] = fill_rgb[(size_t)hi * 3 + c];
                valid[(size_t)p] = 1;
                src_of[(size_t)p] = 2;
            }
        }
    }
    st.t_fill3d = texatlas::_now() - t0;
    const int n_telea = n_hole - n_fill3d - n_base;
    if (cfg.verbose && fill3d)
        std::printf("[texproj] 3D fill: %d / %d holes filled (%.1f%%), %d fell through to telea (%.1f%%)  (%.2fs)\n",
                    n_fill3d, n_hole, n_hole ? 100.0 * n_fill3d / n_hole : 0.0,
                    n_telea, n_hole ? 100.0 * n_telea / n_hole : 0.0, st.t_fill3d);
    // ---- BUG 3: the seam metric ----
    st.n_seam = (int)loc_seam;
    for (int c = 0; c < 3; c++) {
        st.seam_absdiff[c] = loc_seam ? (float)(255.0 * loc_sd[c] / loc_seam) : 0.f;
        st.seam_bias[c]    = loc_seam ? (float)(255.0 * loc_ss[c] / loc_seam) : 0.f;
    }
    st.seam_mean_absdiff = (st.seam_absdiff[0] + st.seam_absdiff[1] + st.seam_absdiff[2]) / 3.f;
    st.seam_blend_on = seam_blend;
    if (cfg.verbose && fill3d && NV > 1) {
        std::printf("[texproj] SEAM: %d fill texels see >=2 views (%.2f%% of 3D fills) — the ONLY overlap "
                    "front and back have (the confidence ramp's own overlap band is empty by construction)\n",
                    st.n_seam, n_fill3d ? 100.0 * st.n_seam / n_fill3d : 0.0);
        if (st.n_seam) {
            std::printf("[texproj] SEAM: |viewA-viewB| = R%.1f G%.1f B%.1f (mean %.1f/255)   SIGNED bias = "
                        "R%+.1f G%+.1f B%+.1f   bias/|d| = %.2f\n",
                        st.seam_absdiff[0], st.seam_absdiff[1], st.seam_absdiff[2], st.seam_mean_absdiff,
                        st.seam_bias[0], st.seam_bias[1], st.seam_bias[2],
                        st.seam_mean_absdiff > 1e-6f
                            ? (std::fabs(st.seam_bias[0]) + std::fabs(st.seam_bias[1]) + std::fabs(st.seam_bias[2]))
                              / (3.f * st.seam_mean_absdiff) : 0.f);
            std::printf("[texproj] SEAM: cross-fade %s (TEXPROJ_SEAM_BLEND=%d, band=%.2f). bias/|d| well "
                        "below 1 means the views DISAGREE without an OFFSET — a global colour match has "
                        "almost nothing to remove.\n", seam_blend ? "ON" : "OFF", (int)seam_blend, seam_band);
        }
    }
    // mark the leftovers so the debug PNG can show what telea had to invent
    for (int p = 0; p < T; p++) if (mask[(size_t)p] && !valid[(size_t)p]) src_of[(size_t)p] = 3;

    // ---- 7. gutters + leftovers: reuse the shared fillers, in LINEAR light ----
    // UNCHANGED and still correct for its real job: true UV GUTTERS (mask=0 texels between charts) exist
    // so a renderer's bilinear filter has something to bleed from — an atlas-space problem with an
    // atlas-space answer. inpaint_telea is distance-ordered from the valid boundary, so each chart mostly
    // fills its OWN gutter; keep the ring count modest (charts are packed ~padding apart in the precluster
    // atlas, and an over-long fill is what makes a neighbour's colour peek through the cuts —
    // tex_atlas.hpp:1020). dilate_background then nearest-valids EVERYTHING still unfilled, which
    // guarantees no black texels rather than pretending to invent detail.
    const int rings  = std::max(0, envi("TEXPROJ_INPAINT_RINGS", 8));
    const int radius = std::max(1, envi("TEXPROJ_INPAINT_RADIUS", 3));
    t0 = texatlas::_now();
    std::vector<uint8_t> fill_mask = valid;   // both fillers MUTATE the mask (1 = known/valid)
    if (rings > 0) texatlas::inpaint_telea(rgb_lin, fill_mask, W, H, 3, rings, radius);
    if (!std::getenv("TEXPROJ_NO_DILATE")) texatlas::dilate_background(rgb_lin, fill_mask, W, H, 3);
    st.t_inpaint = texatlas::_now() - t0;
    if (cfg.verbose) std::printf("[texproj] gutter inpaint rings=%d radius=%d + background dilate (%.2fs)\n",
                                 rings, radius, st.t_inpaint);

    // ---- write back: linear -> sRGB byte. Alpha (and bt.metal_rough) are KEPT from the volume bake. ----
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < T; p++) {
        const float* s = &rgb_lin[(size_t)p * 3];
        for (int c = 0; c < 3; c++) bt.base_color[(size_t)p * 4 + c] = u8f(linear_to_srgb(s[c]));
    }

    // ---- debug dumps ----
    if (!debug_dir.empty()) {
        mkdir(debug_dir.c_str(), 0755);
        dump_png(debug_dir + "/proj_confidence.png", dbg.data(), W, H, 3);          // R=front G=other B=hole
        dump_png(debug_dir + "/proj_base_color.png", bt.base_color.data(), W, H, 4);
        // PROVENANCE MAP — the one image that answers "is the camo gone?" without guessing:
        //   RED   = a real projected pixel     GREEN = 3D-filled (k-NN, normal-gated)
        //   BLUE  = fell through to atlas Telea/dilate  (blue on a large smooth region = camo risk)
        //   black = not covered (gutter/background)
        {
            std::vector<uint8_t> fs((size_t)W * H * 3, 0);
            for (int p = 0; p < T; p++) {
                uint8_t* o = &fs[(size_t)p * 3];
                switch (src_of[(size_t)p]) {
                    case 1: o[0] = 255; break;
                    case 2: o[1] = 255; break;
                    case 3: o[2] = 255; break;
                    case 4: o[0] = 255; o[1] = 255; break; // yellow = original volume fallback
                    default: break;
                }
            }
            dump_png(debug_dir + "/proj_fill_source.png", fs.data(), W, H, 3);
        }
        // SEAM MAP (BUG 3) — "where is the crossover, and how wide is it?"
        //   RED   = painted by view 0 (the FRONT matte)
        //   GREEN = painted by any other view (the generated back / extra yaws)
        //   BLUE  = a 3D-fill texel whose neighbours span >=2 views = THE SEAM STRIP itself
        //   black = not covered, or a fill that only one view could reach
        // A correct run: two big flat fields of red and green with a THIN blue ribbon between them tracing
        // the silhouette. A WIDE blue ribbon means the two views' painted regions do not meet and the fill
        // is inventing the crossover over a large area — which is exactly what a bad subject mask on the
        // generated view causes (see BUG 3: black@0.05 vs alpha).
        {
            std::vector<uint8_t> sm((size_t)W * H * 3, 0);
            for (int p = 0; p < T; p++) {
                if (!mask[(size_t)p]) continue;
                uint8_t* o = &sm[(size_t)p * 3];
                if (src_of[(size_t)p] == 1) { if (view_of[(size_t)p] == 0) o[0] = 255; else o[1] = 255; }
                if (seam_of[(size_t)p]) o[2] = 255;
            }
            dump_png(debug_dir + "/proj_seam.png", sm.data(), W, H, 3);
        }
        // BUG-2 MAP — "are the black streaks gone, and did we reject the right texels?"
        //   RED   = at least one view refused to paint this texel because it sampled background
        //   GREEN = how hard that view was facing us when it did (0 = perfectly grazing = expected;
        //           BRIGHT green = a HIGH-facing texel got rejected = the mask or the fit is wrong)
        //   BLUE  = this texel ended up a hole (so the 3D fill / telea owns it)
        // A correct run: a thin RED rim tracing the silhouette, essentially BLACK green channel.
        if (want_bgmap) {
            std::vector<uint8_t> bm((size_t)W * H * 3, 0);
            for (int p = 0; p < T; p++) {
                if (!mask[(size_t)p]) continue;
                uint8_t* o = &bm[(size_t)p * 3];
                o[0] = bgrej_any[(size_t)p] ? 255 : 0;
                o[1] = bgrej_any[(size_t)p] ? u8f(bgrej_maxf[(size_t)p]) : 0;
                o[2] = src_of[(size_t)p] >= 2 ? 255 : 0;
            }
            dump_png(debug_dir + "/proj_bg_reject.png", bm.data(), W, H, 3);
        }
        for (int i = 0; i < NV; i++) {
            char nm[64];
            if (i == 0) std::snprintf(nm, sizeof(nm), "front");
            else std::snprintf(nm, sizeof(nm), "yaw%03d", (int)std::lround(views[i].cam.yaw_deg));
            dump_zbuf(debug_dir + "/proj_zbuf_" + nm + ".png", views[i].z);
            // the view image with the FITTED silhouette bbox (red) over the detected subject bbox (green).
            // If the fit is sane the red box hugs the green one; if it's wild you can see it immediately.
            // The FRONT gets this too now — it is fitted like everything else (BUG 1).
            const View& v = views[i];
            if (!v.fit.fitted && !v.sub.valid()) continue;
            std::vector<uint8_t> ov((size_t)v.img.w * v.img.h * 3);
            for (size_t k = 0; k < (size_t)v.img.w * v.img.h * 3; k++) ov[k] = u8f(v.img.rgb[k]);
            draw_rect(ov, v.img.w, v.img.h, v.sub, 0, 255, 0, 2);
            BBox msil = v.sil;
            if (v.fit.fitted) {
                float ax, ay, bxx, byy;
                v.fit.apply((float)v.sil.x0, (float)v.sil.y0, ax, ay);
                v.fit.apply((float)v.sil.x1, (float)v.sil.y1, bxx, byy);
                msil.x0 = (int)std::lround(ax); msil.y0 = (int)std::lround(ay);
                msil.x1 = (int)std::lround(bxx); msil.y1 = (int)std::lround(byy);
            }
            draw_rect(ov, v.img.w, v.img.h, msil, 255, 0, 0, 2);
            dump_png(debug_dir + "/proj_align_" + nm + ".png", ov.data(), v.img.w, v.img.h, 3);
        }
        if (cfg.verbose)
            std::printf("[texproj] debug -> %s: proj_fill_source.png (R=projected G=3D-filled B=telea), "
                        "proj_seam.png (R=front-painted G=back-painted B=THE SEAM STRIP), "
                        "proj_confidence.png (R=front G=other B=hole), proj_base_color.png, proj_zbuf_*.png"
                        "%s\n", debug_dir.c_str(),
                        NV > 1 ? ", proj_align_*.png (red=fitted silhouette, green=detected subject)" : "");
    }

    st.covered = covered;
    st.n_hole = n_hole; st.n_fill3d = n_fill3d; st.n_telea = n_telea; st.n_base = n_base;
    st.front_pct = 100.0 * n_view[0] / covered;
    st.back_pct  = (back_i >= 0) ? 100.0 * n_view[back_i] / covered : 0.0;
    st.hole_pct  = 100.0 * n_hole / covered;
    for (int i = 0; i < NV; i++) {
        st.view_yaw.push_back(views[i].cam.yaw_deg);
        st.view_pct.push_back(100.0 * n_view[i] / covered);
        st.view_bgrej.push_back(n_bgrej[i]);
        st.view_bgrej_facing.push_back(n_bgrej[i] ? (float)(f_bgrej[i] / n_bgrej[i]) : 0.f);
        st.view_kept_facing.push_back(n_view[i] ? (float)(f_kept[i] / n_view[i]) : 0.f);
        st.view_scale.push_back(views[i].fit.fitted ? views[i].fit.sx : 1.f);
        st.view_tx.push_back(views[i].fit.fitted ? views[i].fit.tx : 0.f);
        st.view_ty.push_back(views[i].fit.fitted ? views[i].fit.ty : 0.f);
    }
    if (out_stats) *out_stats = st;
    return true;
}

}  // namespace texproj
