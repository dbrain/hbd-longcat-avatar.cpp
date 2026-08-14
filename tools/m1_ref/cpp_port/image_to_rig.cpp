// image_to_rig — the INLINE C++ API (TASK B / END GOAL): one in-process binary that turns a
// background-removed character matte into a TEXTURED + RIGGED (skinned) GLB, with NO bash stitching
// and NO python. Replaces image2rig.sh: every stage is a native call in this single process, the GPU
// to itself, weights freed between stages (M1Harness scope), fits one RTX 3060 (12GB).
//
//   matte.png
//     -> pix::run_geometry(--tex)            : DINOv3 + DiTs/VAEs -> dense mesh + per-voxel PBR
//     -> texatlas::bake                      : xatlas UV unwrap + grid_sample -> TEXTURED mesh (in RAM)
//     -> rig::prep_mesh_for_rig_inmem        : normalize + area-sample + FPS query (no GLB round-trip)
//     -> rig::run_rig_pipeline               : R1 VecSet -> R3 Qwen3 beam -> R5 detok -> R4 skin-VAE
//     -> rig::transfer_skin                  : kNN skin transfer sampled -> full textured mesh
//     -> glb::write_rigged_textured_glb      : TEXTURED + SKINNED GLB
//
// Usage:
//   image_to_rig --model <geo_gguf_dir> --image <matte.png> --out <out.glb>
//                [--fov <deg>] [--cam <ang_rad> <dist> <scale>] [--texsize N] [--decimate F]
//                [--resolution N] [--seed N] [--fast]
//                [--r1w <dir>] [--qwen3 <dir>] [--skin-vae <dir>] [--beams N] [--rig-seed N] [--cpu]
//
// Camera: --fov/--cam set the geometry camera (default = miku cam). Native MoGe fold-in (image->FOV
// in-process) is moge_neck.hpp+moge_fov.hpp, validated standalone; wiring it as a callable here is the
// remaining sub-step (it needs MoGe weights as a GGUF + variable-res preprocess). Until then pass --fov.
//
// Build: ./build.sh image_to_rig cuda   (reuses the pixal3d link recipe: ggml-cuda + spike conv +
// cumesh + basisu + xatlas + meshopt). CPU build is impractical (geometry needs the GPU).
#include "image_to_rig_options.hpp"    // the flags as a STRUCT: main() parses into it, a library fills it
#include "cancel_hook.hpp"             // cooperative cancellation for the long GPU loops
#include "pixal3d_chain.hpp"
#include "image_io.hpp"
#include "matte_native_imgio.hpp"     // in-process RMBG-2.0 matte (replaces the docker/vision-cli hop)
#include "tex_atlas.hpp"
#include "tex_project.hpp"             // --tex-project: project the real images into the UV atlas
#include "glb_textured.hpp"            // encode_png
#include "glb_rigged_textured.hpp"     // write_rigged_textured_glb
#include "rig_transfer.hpp"            // transfer_skin
#include "rig_pipeline.hpp"            // run_rig_pipeline
#include "rig_pose_gate.hpp"           // native pose gate: the selector's deformation term
#include "rig_stage_report.hpp"        // what the rig stage decided, for a library caller
#include "rig_bone_names.hpp"          // name_bones + falsify_bone_names (standard bone naming)
#include "mesh_sample.hpp"             // prep_mesh_for_rig_inmem, normalize_mesh
#include "p3sam_segment.hpp"           // P3-SAM part segmentation (native, --part-retopo)
#include "per_part_decimate.hpp"       // region-adaptive per-part decimation (hands kept dense)
#include "moge_cam.hpp"                // native MoGe camera (FOV) estimation (--moge)
#include "ultrashape_refine.hpp"       // native UltraShape refine (clean/watertight densify, --refine)
#include "quad_retopo.hpp"             // rung 2: quadwild-bimdf quad retopology (--quad, shell-out)
#include "im_retopo.hpp"               // rung 2 (default): Instant Meshes retopo (clean organic flow)
#include "normal_bake.hpp"             // tangent-space normal map: dense high-poly detail -> retopo low-poly
#include "narrow_band_dc.hpp"          // --dc-remesh: Python-parity narrow-band DC of the O-Voxel mesh
#include "mesh_taubin.hpp"             // --dc-remesh: feature-preserving Taubin post-smooth
#ifdef PIXAL3D_PACK
#include "glb_packed.hpp"              // packed writer (KTX2+meshopt) — the only textured writer that carries a normal map
#endif
#include <functional>
#include <unordered_map>
#include <thread>
#include <algorithm>
#include "glb_writer.hpp"              // glb::write_glb (intermediate stage artifacts)
#include "glb_reader.hpp"              // glb::read_glb (--from-refined resume)
#include "../../sparse_spike/npy.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <sys/stat.h>

// DEF_CAM / DEF_DIST / DEF_MS now live in image_to_rig_options.hpp — they are the DEFAULT VALUES of
// Options::cam/dist/ms, and a default belongs with the field it defaults.

static void usage() {
    printf("usage: image_to_rig --model <geo_gguf_dir> --image <matte.png> --out <out.glb>\n"
           "        [--fov <deg>] [--cam <ang_rad> <dist> <scale>] [--texsize N] [--decimate F]\n"
           "        [--resolution N] [--seed N] [--fast] [--cpu]\n"
           "        [--r1w <dir>] [--qwen3 <dir>] [--skin-vae <dir>] [--beams N] [--rig-seed N]\n"
           "        [--rig-retries N]     (default 0. BEST-OF-N over conditioning draws: re-draw the\n"
           "                          8192-point cloud and decode again, up to N extra times, until a\n"
           "                          draw passes the selector; the BEST-ranked draw ships either way.\n"
           "                          The decode is deterministic given the cloud, so a fresh DRAW is\n"
           "                          the only way to ask again. Accept = skin weights present AND\n"
           "                          (22-bone gate OR (creature gate AND pose gate)) AND enough named\n"
           "                          core bones to retarget.)\n"
           "        [--rig-pose-gate | --no-rig-pose-gate]\n"
           "                         (the native deformation audit (rig_pose_gate.hpp = a port of\n"
           "                          rig_pose_smoke.py --pose-gate --generic-all-influential). Default\n"
           "                          ON iff --rig-retries > 0; with no re-draw there is no choice to\n"
           "                          make. Turning it OFF means a creature-gate draw can never be\n"
           "                          accepted, since the pose gate is that branch's safety check.)\n"
           "        [--rig-pose-gate-strict]  (also require the pose gate on the 22-BONE branch of the\n"
           "                          accept predicate, not only the creature branch. Off by default; on,\n"
           "                          a rig that names all 22 bones but shreds a component is rejected.)\n"
           "        [--rig-min-named-core N]  (default 19. A draw must NAME at least N of the SMPL-22\n"
           "                          core bones to be accepted: the retargeter maps through names and\n"
           "                          its topology fallback fails outright on some rigs, so a rig that\n"
           "                          names 14 animates worse than one that names 21 whatever it scores.)\n"
           "        [--rig-sample]  (stochastic scaffold recipe do_sample=true beams=10; default is\n"
           "                         deterministic beam=20 = fan-free rig)\n"
           "        [--no-bone-names] [--bone-names mixamo|smpl] [--bone-facing +z|-z]\n"
           "                         (standard bone naming, ON by default: anonymous bone_N is\n"
           "                          retargetable by nothing. Names are derived from skeleton\n"
           "                          structure + rest geometry and falsified before the write.\n"
           "                          --bone-facing overrides the auto-derived facing, which is what\n"
           "                          decides LEFT vs RIGHT; env RIG_BONE_FACING does the same)\n"
           "        [--part-retopo] [--p3sam-weights <dir>] [--obj-decimate <path>]\n"
           "                         (native P3-SAM finger-preserving per-part decimation before bake;\n"
           "                          CPU correctness-port, slow, OFF by default — GPU port = perf TODO)\n"
           "        [--moge] [--moge-weights <dir>]\n"
           "                         (native MoGe-2 camera/FOV estimation from the image, replaces --fov;\n"
           "                          makes the inline API 100%% camera-native)\n"
           "        [--refine | --no-refine]  (native UltraShape refine between geometry and bake; ON by\n"
           "                          default — the clean/watertight ~7.5x densify)\n"
           "        [--dc-remesh] [--dc-band N=1] [--dc-taubin N=2]\n"
           "                         (PYTHON-PARITY COARSE. Keeps the smooth O-Voxel dual-grid decoder\n"
           "                          mesh and narrow-band dual-contours THAT (= Python o_voxel\n"
           "                          to_glb(remesh=True)) instead of marching-cubes-soliding the binary\n"
           "                          occupancy (a lego staircase), then Taubin-smooths it. Bake + rig\n"
           "                          then run on the parity mesh in ONE call. Implies --no-refine (a\n"
           "                          generative densify no longer aligns to the PBR volume, so texturing\n"
           "                          it slides colour) and raises texsize/decimate defaults to 2048/220k.)\n"
           "        [--us-octree N] [--us-latents N] [--us-steps N] [--us-guidance F] [--us-chunk N]\n"
           "        [--us-gguf <dir>] [--us-dit-w <dir>] [--us-vae-w <dir>] [--us-cnd-w <dir>] [--us-meta <npy>]\n"
           "        [--rig-cache <dir>]   (write the skeleton+skin field on the first run, reuse it on every\n"
           "                          later run. LOD tiers of one asset MUST share a skeleton or a clip\n"
           "                          authored for the hero cannot play on the game tier; re-rigging per\n"
           "                          tier also re-rolls the rig's failure modes. Also skips ~16-30s/tier.)\n"
           "        [--stage-dir <dir>]   (emit coarse/refined/decimated GLB intermediates + resume caches)\n"
           "        [--solid]             (--dc-remesh only, OPT-IN. Sign the field against the decoder\n"
           "                          occupancy so the inward wall of the shrink-wrap envelope is never\n"
           "                          contoured, and drop buried cavity shells: ONE watertight solid.\n"
           "                          The geometry is strictly better, but the auto-rig regresses hard on\n"
           "                          it (miku maxfan 4->13, rig_score 0.897->0.000; gilly fails the pose\n"
           "                          gate) because the rigger was tuned on the two-walled cloud -- see the\n"
           "                          solid_mesh comment. Use it for geometry-only or externally-rigged\n"
           "                          deliveries. --no-solid is the default.)\n"
           "        [--image-model-ready] (diagnostic only: input already is Pixal's final square/black frame;\n"
           "                          bypass native alpha/crop framing to isolate downstream geometry)\n"
           "        [--from-refined <dir>] (resume: skip geometry+refine, load refined.glb + PBR cache)\n"
           "        [--geometry-only]      (write the clean refined geometry cache; skip legacy PBR/atlas/rig)\n"
           "        [--material-cache-only] (write clean refined geometry + native Pixal PBR cache; skip atlas/rig)\n"
           "        [--tex-dit proj|cross] [--tex-dit-w <dir>]\n"
           "                         (WHICH generative tex DiT paints the PBR volume. cross = DEFAULT =\n"
           "                          trellis2_tex_1024 (TRELLIS-2 texturing model, full 4101-token DINOv3\n"
           "                          cross-attn cond). proj = slat_flow_imgshape2tex_1024 = the model\n"
           "                          pixal3d's OWN python pipeline runs (5 global tokens + proj_linear over\n"
           "                          the stage3b proj cond). Same cond/normalization/decoder/bake either way,\n"
           "                          so it is a clean single-variable A/B. proj is also the cheaper graph.)\n"
           "        [--tex-snap-volume]   (reproject-bake mode for refined mesh: snap+volume instead of the\n"
           "                          default mesh-attr; steadier on fine hair, more holes on smooth bodies)\n"
           "        [--tex-volume-direct] [--tex-fallback-r <voxels>]\n"
           "                         (THE FRAY FIX, default OFF. Both snap modes above fetch colour via TWO\n"
           "                          chained closest-point projections (texel -> coarse shell -> volume),\n"
           "                          which slides along the surface: 9.2%% of texels read a voxel >4 vox\n"
           "                          away on the model (p99 13.2), so material boundaries the volume draws\n"
           "                          cleanly come out frayed. --tex-volume-direct reads the volume at the\n"
           "                          TEXEL'S OWN position (snap kept only as the no-data guard) and cuts\n"
           "                          volume-residual texels >0.15 from 4.87%% to 1.34%%. --tex-fallback-r\n"
           "                          sets the nearest-voxel search radius (default 8 in direct mode, 0 in\n"
           "                          the snap modes = today's behaviour); r=0 in direct mode reproduces\n"
           "                          the historic BLACK-TEXTURE bug.)\n"
           "        [--tex-project] [--tex-project-overlay] [--tex-front <img>] [--tex-back <img>] [--tex-view <yaw_deg> <img>]...\n"
           "                         (texture by PROJECTING the real images into the UV atlas instead of\n"
           "                          sampling TRELLIS's soft PBR volume: front = --image (or --tex-front,\n"
           "                          a TEXTURE-ONLY front source in the SAME camera frame -- e.g. an\n"
           "                          sd-delight de-lit --image; geometry always keeps --image), back = optional\n"
           "                          --tex-back (sugar for --tex-view 180). --tex-view is REPEATABLE and\n"
           "                          takes any yaw about +Y (e.g. --tex-view 90 right.png), so side views\n"
           "                          can be added as they become available. Z-buffered occlusion +\n"
           "                          grazing-angle confidence ramp; holes get a 3D-aware k-nearest fill\n"
           "                          (TEXPROJ_FILL_3D=0 for the old atlas-Telea A/B). EVERY view is\n"
           "                          silhouette-bbox auto-aligned, the FRONT included (the refined mesh is\n"
           "                          NOT the matte's exact projection: measured ~1%% narrow -> gold buttons\n"
           "                          slid off their bumps; TEXPROJ_FRONT_ALIGN=0 to A/B). Samples landing\n"
           "                          outside the view's eroded subject mask are REJECTED rather than\n"
           "                          bilinear-blended with the black background (that was the black streaks\n"
           "                          down the sides); TEXPROJ_BG_REJECT=0 to A/B. Keeps the volume\n"
           "                          metallicRoughness — see tex_project.hpp's METAL_ROUGH CAVEAT, but note\n"
           "                          the owner's gold complaint is REGISTRATION, not metalness.\n"
           "                          Near-source detail on the front. --tex-project-overlay retains the\n"
           "                          baked volume texture for unobserved texels rather than inventing it;\n"
           "                          use it for a partial turnaround / occluded hair-and-arms hybrid.)\n"
           "        [--quad] [--quadwild-repo <dir>]\n"
           "                         (rung-2 quad retopology via quadwild-bimdf on the refined mesh: clean\n"
           "                          field-aligned topology, triangulated for the downstream; shell-out, CPU)\n");
}


// ---------------------------------------------------------------------------
// argv -> Options. THE ONLY PARSER: main() runs it, the library skips it and fills the struct
// directly. Flag for flag, message for message, exit code for exit code, this is the loop that
// used to sit inline in main() with `x =` rewritten as `o.x =`.
//
// It APPLIES ONTO whatever `o` already holds, so a caller can fill the typed fields and then hand
// through an escape-hatch flag list -- later still wins, exactly as a later argv element does.
// ---------------------------------------------------------------------------
int i2r::parse_args(int argc, char** argv, i2r::Options& o) {
    auto nextf = [&](int& i){ return (float)std::atof(argv[++i]); };
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--model" && i+1 < argc) o.model = argv[++i];
        else if (a == "--image" && i+1 < argc) o.image = argv[++i];
        else if (a == "--out"   && i+1 < argc) o.out = argv[++i];
        // cam and dist are THE SAME PARAMETER: DEF_DIST == 0.5/tan(DEF_CAM/2) to 7 d.p. (1.3021560).
        // DEF_DIST was never a measured constant -- it is the canonical "unit-diameter object exactly
        // fills the frame" render convention. Setting cam WITHOUT dist breaks the identity and turns a
        // perspective change into a ZOOM: measured on the soldier, --moge's 46.5deg made the model
        // +16% BIGGER (extents x1.164/1.159/1.146, shape unchanged to ~1%) and pushed 23,336 verts
        // through the canonical box floor at Y <= -0.4999 -- a flat plane where the boots should be.
        // That is what OOM'd the refine (bigger subject -> N1 1227->1701 -> M 12541->18674 -> M^2).
        else if (a == "--fov"   && i+1 < argc) { o.cam = std::atof(argv[++i]) * (float)M_PI / 180.0f; o.dist = 0.5f / std::tan(o.cam * 0.5f); }
        else if (a == "--cam"   && i+3 < argc) { o.cam = std::atof(argv[++i]); o.dist = std::atof(argv[++i]); o.ms = std::atof(argv[++i]); }
        else if (a == "--texsize" && i+1 < argc) { o.texsize = std::atoi(argv[++i]); o.texsize_set = true; }
        else if (a == "--decimate" && i+1 < argc) { o.decimate = std::atoi(argv[++i]); o.decimate_set = true; }
        else if ((a == "--resolution" || a == "--res") && i+1 < argc) {
            const int r = std::atoi(argv[++i]);
            if (r % 16 != 0) { printf("--resolution must be a multiple of 16\n"); return 1; }
            o.geo_resolution = r;
        }
        else if (a == "--seed" && i+1 < argc) o.geo_seed = std::atoi(argv[++i]);
        else if (a == "--fast") o.fast = true;
        else if (a == "--cpu")  o.use_cuda = false;
        else if (a == "--r1w"   && i+1 < argc) o.r1w = argv[++i];
        else if (a == "--qwen3" && i+1 < argc) o.qwen3_w = argv[++i];
        else if (a == "--skin-vae" && i+1 < argc) o.skinvae = argv[++i];
        else if (a == "--beams" && i+1 < argc) o.num_beams = std::atoi(argv[++i]);
        else if (a == "--rig-sample") { o.rig_sample = true; if (o.num_beams == 20) o.num_beams = 10; }
        else if (a == "--rig-seed" && i+1 < argc) o.rig_seed = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--no-rig-select") o.rig_structural_select = false;
        else if (a == "--rig-retries" && i+1 < argc) o.rig_retries = std::max(0, std::atoi(argv[++i]));
        else if (a == "--allow-zero-skin") o.allow_zero_skin = true;
        else if (a == "--rig-pose-gate") o.rig_pose_gate = true;
        else if (a == "--no-rig-pose-gate") o.rig_pose_gate = false;
        else if (a == "--rig-min-named-core" && i+1 < argc) o.rig_min_named_core = std::atoi(argv[++i]);
        else if (a == "--rig-pose-gate-strict") { o.rig_pose_gate_strict = true; if (!o.rig_pose_gate.set) o.rig_pose_gate = true; }
        else if (a == "--no-bone-names") o.bone_names = false;
        else if (a == "--bone-names" && i+1 < argc) { std::string v = argv[++i]; o.bone_names_smpl = (v == "smpl"); }
        else if (a == "--bone-facing" && i+1 < argc) { o.bone_facing = (argv[++i][0] == '-') ? -1 : +1; }
        else if (a == "--part-retopo") o.part_retopo = true;
        else if (a == "--p3sam-weights" && i+1 < argc) o.p3sam_w = argv[++i];
        else if (a == "--obj-decimate" && i+1 < argc) o.obj_decimate = argv[++i];
        else if (a == "--moge") o.use_moge = true;
        else if (a == "--moge-weights" && i+1 < argc) o.moge_w = argv[++i];
        // ---- UltraShape refine stage ----
        else if (a == "--refine") { o.refine = true; o.refine_set = true; }
        else if (a == "--no-refine") { o.refine = false; o.refine_set = true; }
        else if (a == "--dc-remesh") o.dc_remesh = true;
        else if (a == "--no-solid") o.solid_mesh = false;
        else if (a == "--solid") o.solid_mesh = true;
        else if (a == "--dc-band" && i+1 < argc) o.dc_band = std::atoi(argv[++i]);
        else if (a == "--dc-taubin" && i+1 < argc) o.dc_taubin = std::atoi(argv[++i]);
        else if (a == "--us-octree" && i+1 < argc) o.us_octree = std::atoi(argv[++i]);
        else if (a == "--us-latents" && i+1 < argc) o.us_num_latents = std::atoi(argv[++i]);
        else if (a == "--us-steps" && i+1 < argc) o.us_steps = std::atoi(argv[++i]);
        else if (a == "--us-guidance" && i+1 < argc) o.us_guidance = (float)std::atof(argv[++i]);
        else if (a == "--us-chunk" && i+1 < argc) o.us_chunk = (int64_t)std::atoll(argv[++i]);
        else if (a == "--us-gguf" && i+1 < argc) o.us_gguf_dir = std::string(argv[++i]);
        else if (a == "--us-dit-w" && i+1 < argc) o.us_dit_w = std::string(argv[++i]);
        else if (a == "--us-vae-w" && i+1 < argc) o.us_vae_w = std::string(argv[++i]);
        else if (a == "--us-cnd-w" && i+1 < argc) o.us_cnd_w = std::string(argv[++i]);
        else if (a == "--us-meta" && i+1 < argc) o.us_meta = std::string(argv[++i]);
        else if (a == "--stage-dir" && i+1 < argc) o.stage_dir = argv[++i];
        else if (a == "--image-model-ready") o.image_model_ready = true;
        else if (a == "--from-refined" && i+1 < argc) o.from_refined = argv[++i];
        else if (a == "--tex-snap-volume") o.tex_snap_volume = true;
        else if (a == "--tex-volume-direct") o.tex_volume_direct = true;
        else if (a == "--tex-fallback-r" && i+1 < argc) o.tex_fallback_r = std::atoi(argv[++i]);
        else if (a == "--tex-project") o.tex_project = true;
        else if (a == "--tex-project-overlay") { o.tex_project = true; o.tex_project_overlay = true; }
        else if (a == "--tex-front" && i+1 < argc) o.tex_front = argv[++i];
        else if (a == "--tex-back" && i+1 < argc) o.tex_back = argv[++i];
        else if (a == "--tex-view" && i+2 < argc) {
            i2r::TexView vs;
            vs.yaw_deg = (float)std::atof(argv[++i]);
            vs.img = argv[++i];
            // `--tex-view 180 x.png` and `--tex-back x.png` are the same view; route both through tex_back
            // so exactly one yaw=180 camera exists and Stats' back_* fields stay meaningful.
            if (std::fmod(vs.yaw_deg, 360.f) == 180.f && o.tex_back.empty()) o.tex_back = vs.img;
            else if (std::fmod(vs.yaw_deg, 360.f) == 0.f)
                printf("NOTE: --tex-view %g is the FRONT view (--image); ignoring the duplicate\n", vs.yaw_deg);
            else o.tex_views.push_back(vs);
        }
        // --tex-dit proj|cross: WHICH generative tex DiT paints the PBR volume. `cross` (DEFAULT =
        // unchanged behaviour) = trellis2_tex_1024, the TRELLIS-2 texturing model the tex_goldens came
        // from. `proj` = slat_flow_imgshape2tex_1024 = the model pixal3d's OWN Python pipeline runs
        // (Trellis2TexturingPipeline is dead code there), i.e. the one that painted gilly. Same cond the
        // geometry stage already computes, same tex_slat normalization, same tex decoder, same bake — so
        // this is a clean single-variable A/B. --tex-dit-w overrides the weight dir/basename.
        else if (a == "--tex-dit" && i+1 < argc) {
            std::string m = argv[++i];
            if (m == "proj") o.geo_tex_proj = true;
            else if (m == "cross") o.geo_tex_proj = false;
            else { printf("--tex-dit must be 'proj' or 'cross' (got '%s')\n", m.c_str()); return 1; }
        }
        else if (a == "--tex-dit-w" && i+1 < argc) o.geo_tex_flow_w = std::string(argv[++i]);
        else if (a == "--quad") o.do_quad = true;
        else if (a == "--no-quad") o.do_quad = false;
        else if (a == "--retopo" && i+1 < argc) o.retopo_tool = argv[++i];   // im (default) | quadwild
        else if (a == "--im-adaptivity" && i+1 < argc) o.im_adaptivity = (float)std::atof(argv[++i]);
        else if (a == "--im-verts" && i+1 < argc) o.im_target_verts = std::atoi(argv[++i]);
        else if (a == "--quadwild-repo" && i+1 < argc) o.quad_repo = std::string(argv[++i]);
        else if (a == "--no-clean") o.clean = false;
        else if (a == "--mc-stride" && i+1 < argc) o.mc_stride = std::atoi(argv[++i]);
        else if (a == "--mc-blur" && i+1 < argc) o.mc_blur = std::atoi(argv[++i]);
        else if (a == "--mc-smooth" && i+1 < argc) o.mc_smooth = std::atoi(argv[++i]);
        else if (a == "--rig-cache" && i+1 < argc) o.rig_cache = argv[++i];
        else if (a == "--dump-geo" && i+1 < argc) o.dump_geo = argv[++i];
        else if (a == "--from-geo" && i+1 < argc) o.from_geo = argv[++i];
        else if (a == "--no-rig") o.no_rig = true;
        else if (a == "--geometry-only") o.geometry_only = true;
        else if (a == "--material-cache-only") o.material_cache_only = true;
        // geometry sampler knobs (forwarded to ChainInput; defaults already = inference.py)
        else if (a == "--guidance" && i+1 < argc) { float g=nextf(i); o.geo_guidance = g; }
        else if (a == "--steps" && i+1 < argc) { int s=std::atoi(argv[++i]); o.geo_steps = s; }
        else { printf("unknown/incomplete arg: %s\n", a.c_str()); usage(); return 1; }
    }
    return -1;   // parsed; carry on
}

static std::vector<float> load_norm(const std::string& model, const char* kind, const char* which) {
    for (std::string p : {model + "/" + kind + "_norm_" + which + ".npy",
                          std::string("refs/") + kind + "_norm_" + which + ".npy"}) {
        FILE* f = fopen(p.c_str(), "rb");
        if (f) { fclose(f); NpyArray a = npy_load(p); return std::vector<float>(a.f32(), a.f32() + a.numel()); }
    }
    throw std::runtime_error(std::string(kind) + "_norm_" + which + " not found in model dir or refs/");
}

// bbox of an interleaved [n*3] vertex array
static void mesh_bbox(const std::vector<float>& v, float mn[3], float mx[3]) {
    mn[0]=mn[1]=mn[2]=1e30f; mx[0]=mx[1]=mx[2]=-1e30f;
    for (size_t i=0; i+2 < v.size(); i+=3) for (int c=0;c<3;c++) { float x=v[i+c]; mn[c]=std::min(mn[c],x); mx[c]=std::max(mx[c],x); }
}

// Map mesh `v`'s bbox onto reference bbox `ref` by center + UNIFORM scale (longest-axis extent ratio).
// This is exactly run_pipeline.sh's RP_CANON_TO_DENSE=1: it puts the UltraShape-frame refined mesh (±1)
// into the pixal3d coarse-mesh frame ([-0.5,0.5]) so texatlas::bake samples the PBR volume correctly.
static void bbox_canon_onto(std::vector<float>& v, const std::vector<float>& ref) {
    float vmn[3],vmx[3],rmn[3],rmx[3];
    mesh_bbox(v, vmn, vmx); mesh_bbox(ref, rmn, rmx);
    float vc[3],rc[3]; float vext=0.f, rext=0.f;
    for (int c=0;c<3;c++){ vc[c]=0.5f*(vmn[c]+vmx[c]); rc[c]=0.5f*(rmn[c]+rmx[c]);
                           vext=std::max(vext, vmx[c]-vmn[c]); rext=std::max(rext, rmx[c]-rmn[c]); }
    float s = (vext > 1e-9f) ? rext/vext : 1.0f;
    for (size_t i=0; i+2 < v.size(); i+=3) for (int c=0;c<3;c++) v[i+c] = (v[i+c]-vc[c])*s + rc[c];
}

// ---------------------------------------------------------------------------
// The rig-draw selector's verdict on ONE conditioning draw, and the ordering it induces.
// ---------------------------------------------------------------------------
namespace i2r {

// Last rig stage's decision, for a caller that got only an int back. See rig_stage_report.hpp for
// why this is a last-run global and the conditions under which that is well defined.
static rig::StageReport g_rig_report;

struct DrawVerdict {
    int  index = 0, J = 0;
    bool skin_ok = false, humanoid = false, generic = false;
    bool naming_evaluated = false;
    int  named_core = 0;
    int  falsifier = 999;          // 999 = naming failed / not run: ranks below any real count
    bool pose_ran = false, pose_pass = false;
    double pose_worst = 0, pose_moved = 0;
    bool accept = false;

    // `strict` also demands the deformation check on the HUMANOID branch. Default off because the
    // 22-bone gate is a strong anatomy claim on its own — but it is not a deformation claim, and
    // char1 is the counter-example: its shipped rig names 22/22, passes the humanoid gate and scores
    // the HIGHEST rig_score of any subject we ship (0.921), while the pose gate reads 10.454
    // all-influential and 77.864 on one component (25.877 even on the plain arms-raised pose). The
    // lenient predicate accepts that rig at draw 0 and never looks further. See the report.
    void compute_accept(int min_named_core, bool strict) {
        const bool named_ok = !naming_evaluated || named_core >= min_named_core;
        const bool deform_ok = !pose_ran || pose_pass;   // no gate run => no evidence against
        accept = skin_ok && named_ok &&
                 ((humanoid && (!strict || deform_ok)) || (generic && pose_pass));
    }

    // STRICT ORDERING, and the reasoning for each rung:
    //  1. skin_ok      — a rig that deforms nothing is never a delivery, whatever else it scores.
    //  2. accept       — a draw that cleared the whole predicate beats one that did not.
    //  3. named_core   — THE TIE-BREAK, and deliberately not the pose-gate number. Both are about
    //                    "will this animate", but the retargeter maps through NAMES: 21/22 named
    //                    with pose 4.127 (miku draw 0) animates better than 14/22 named with a
    //                    prettier score (draw 2), because the missing names are missing bones. So
    //                    rank on retargetability first and on deformation quality second.
    //  4. falsifier    — among equally-named rigs prefer the one whose names survive the falsifier.
    //  5. pose_worst   — then the smaller worst-case LBS seam.
    //  6. index        — then the EARLIER draw, so the result is deterministic and draw 0 (the exact
    //                    NumPy-parity path) wins every tie it is in.
    bool better_than(const DrawVerdict& o) const {
        if (skin_ok    != o.skin_ok)    return skin_ok;
        if (accept     != o.accept)     return accept;
        if (named_core != o.named_core) return named_core > o.named_core;
        if (falsifier  != o.falsifier)  return falsifier < o.falsifier;
        if (pose_ran && o.pose_ran && pose_worst != o.pose_worst) return pose_worst < o.pose_worst;
        return index < o.index;
    }

    std::string describe() const {
        char b[512];
        std::string gate = humanoid ? "humanoid" : (generic ? "generic" : "NO gate (top beam kept)");
        std::string pose = !pose_ran ? std::string("pose=n/a")
                                     : (std::string(pose_pass ? "pose PASS " : "pose FAIL ") +
                                        [&]{ char t[64]; std::snprintf(t, sizeof(t), "%.3f/6.0 moved=%.3f",
                                                                       pose_worst, pose_moved); return std::string(t); }());
        std::snprintf(b, sizeof(b), "%s gate=%s core=%d/22 falsifier=%s %s -> %s",
                      skin_ok ? "skin ok" : "SKIN ZERO", gate.c_str(),
                      naming_evaluated ? named_core : -1,
                      falsifier == 999 ? "n/a" : std::to_string(falsifier).c_str(),
                      pose.c_str(), accept ? "ACCEPT" : "reject");
        return b;
    }
};

}  // namespace i2r

const rig::StageReport& image_to_rig_last_rig_report() { return i2r::g_rig_report; }

// IMAGE_TO_RIG_LIB: this translation unit is BOTH the program and a callable stage, so there is one
// implementation and not a library copy that can drift from the binary. The stage is
// i2r::run(Options) — a typed struct, no argv round-trip: avatar::Engine fills the fields it means
// and calls this directly, and main() (at the bottom of the file) parses argv into the same struct
// first. Everything below the binding block is the pipeline exactly as it was.
static int i2r_run_body(i2r::Options opt) {
    // Correctness-first: fp32 matmul accumulation, no TF32 (TF32's ~1e-2 noise flips occupancy-threshold
    // voxels). overwrite=0 so a caller CAN relax it (export NVIDIA_TF32_OVERRIDE=1) for the perf A/B --
    // pixal3d.cpp:67's sibling comment already says "perf phase relaxes". Default stays 0.
    setenv("NVIDIA_TF32_OVERRIDE", "0", 0);
    // UltraShape refine perf defaults (overwrite=0 -> caller's env always wins = escape hatch):
    //  - USR_DIT_FLASH / USR_DECODE_FLASH: flash the refine DiT (self+cross attn) and the VAE decode.
    //    ~2x faster refine (172s vs 334s @ N=8192), sub-voxel (0.61 MC cells vs the fp32 path) and it IS
    //    the reference regime (bf16+flash_attn). `USR_DIT_FLASH=0` / `USR_DECODE_FLASH=0` restore fp32.
    //  - USR_MOE_CHUNK=8192: token-tile the MoE ONLY above 8192 latents (untiled at N<=8192 -> no cost),
    //    so N=16384/32768 fit the 3060's 12GB (the mul_mat_id pool OOMs untiled). Lossless. `=0` disables.
    // Library defaults stay OFF so the fp32 parity tests (which call the DiT/decode directly) are unchanged.
    setenv("USR_DIT_FLASH",    "1",    0);
    setenv("USR_DECODE_FLASH", "1",    0);
    setenv("USR_MOE_CHUNK",    "8192", 0);
    //  - USR_GEO_FLASH=1 enables F16-Q/K/V flash attention for the geometry DiTs (M3b slat + texture).
    //    It is a throughput option only.  The frozen M3b contract on the RTX 3060 measures mean error
    //    1.129e-2 versus Python FP32 with flash, compared with 1.991e-3 for dense F32; that divergence
    //    is large enough to alter face/clothing geometry.  High-quality production therefore defaults
    //    to dense F32.  Set =1 deliberately for a performance-only run and label it as such.
    setenv("USR_GEO_FLASH",    "0",    0);

    // ---- the option NAMES the pipeline below uses, bound to the struct ------------------------
    // References, not copies: the body mutates several of these while resolving implied defaults
    // (--dc-remesh raising texsize, MoGe replacing the camera), and it must keep doing exactly that.
    // Binding by name is also what keeps this refactor reviewable — the ~700 lines of pipeline that
    // follow are untouched, and a merge against another agent's edits to them is a clean one.
    std::string& model   = opt.model;
    std::string& image   = opt.image;
    std::string& out     = opt.out;
    std::string& r1w     = opt.r1w;
    std::string& qwen3_w = opt.qwen3_w;
    std::string& skinvae = opt.skinvae;
    float& cam  = opt.cam;
    float& dist = opt.dist;
    float& ms   = opt.ms;
    bool&  use_cuda = opt.use_cuda;
    bool&  fast     = opt.fast;
    bool&  rig_sample = opt.rig_sample;
    bool&  rig_structural_select = opt.rig_structural_select;
    int&   rig_retries = opt.rig_retries;
    bool&  allow_zero_skin = opt.allow_zero_skin;
    // The pose gate is ON exactly when there is a choice to make. At --rig-retries 0 the loop breaks
    // on the first draw whatever the verdict, so running a ~4 s deformation audit could only slow
    // the shipped path down and change nothing — and the default path must stay bit-identical.
    const bool rig_pose_gate = opt.rig_pose_gate.set ? opt.rig_pose_gate.v : (opt.rig_retries > 0);
    const int  rig_min_named_core = opt.rig_min_named_core;
    const bool rig_pose_gate_strict = opt.rig_pose_gate_strict;
    bool&  clean = opt.clean;
    int&   mc_stride = opt.mc_stride;
    int&   mc_blur   = opt.mc_blur;
    int&   mc_smooth = opt.mc_smooth;
    bool&  dc_remesh = opt.dc_remesh;
    int&   dc_band   = opt.dc_band;
    int&   dc_taubin = opt.dc_taubin;
    bool&  solid_mesh = opt.solid_mesh;
    int&   texsize   = opt.texsize;
    int&   decimate  = opt.decimate;
    int&   num_beams = opt.num_beams;
    bool&  texsize_set  = opt.texsize_set;
    bool&  decimate_set = opt.decimate_set;
    bool&  refine_set   = opt.refine_set;
    std::string& dump_geo  = opt.dump_geo;
    std::string& from_geo  = opt.from_geo;
    std::string& rig_cache = opt.rig_cache;
    bool& no_rig = opt.no_rig;
    bool& geometry_only = opt.geometry_only;
    bool& material_cache_only = opt.material_cache_only;
    uint64_t& rig_seed = opt.rig_seed;
    bool& bone_names = opt.bone_names;
    bool& bone_names_smpl = opt.bone_names_smpl;
    int&  bone_facing = opt.bone_facing;
    bool& part_retopo = opt.part_retopo;
    std::string& p3sam_w = opt.p3sam_w;
    std::string& obj_decimate = opt.obj_decimate;
    bool& use_moge = opt.use_moge;
    std::string& moge_w = opt.moge_w;
    bool& refine = opt.refine;
    bool& tex_snap_volume = opt.tex_snap_volume;
    bool& tex_volume_direct = opt.tex_volume_direct;
    int&  tex_fallback_r = opt.tex_fallback_r;
    bool& tex_project = opt.tex_project;
    bool& tex_project_overlay = opt.tex_project_overlay;
    std::string& tex_back  = opt.tex_back;
    std::string& tex_front = opt.tex_front;
    bool& do_quad = opt.do_quad;
    std::string& retopo_tool = opt.retopo_tool;
    std::string& stage_dir    = opt.stage_dir;
    std::string& from_refined = opt.from_refined;
    bool& image_model_ready = opt.image_model_ready;

    // ---- the mirrored options, applied onto the config structs that own their defaults ---------
    // `.set` is the whole point: an UNSET mirror leaves that struct's own default alone, so each
    // default lives in exactly one place and the two cannot drift (see image_to_rig_options.hpp).
    pix::ChainInput in;
    if (opt.geo_resolution.set)  in.resolution = opt.geo_resolution.v;
    if (opt.geo_seed.set)        in.seed       = opt.geo_seed.v;
    if (opt.geo_guidance.set)  { in.ss.guidance = opt.geo_guidance.v; in.shape.guidance = opt.geo_guidance.v; }
    if (opt.geo_steps.set)     { in.ss.steps = in.shape.steps = in.tex.steps = opt.geo_steps.v; }
    if (opt.geo_tex_proj.set)    in.tex_proj   = opt.geo_tex_proj.v;
    if (opt.geo_tex_flow_w.set)  in.tex_flow_w = opt.geo_tex_flow_w.v;

    usr::RefineCfg uscfg;
    if (opt.us_octree.set)      uscfg.octree      = opt.us_octree.v;
    if (opt.us_num_latents.set) uscfg.num_latents = opt.us_num_latents.v;
    if (opt.us_steps.set)       uscfg.steps       = opt.us_steps.v;
    if (opt.us_guidance.set)    uscfg.guidance    = opt.us_guidance.v;
    if (opt.us_chunk.set)       uscfg.chunk       = opt.us_chunk.v;
    if (opt.us_gguf_dir.set)    uscfg.gguf_dir    = opt.us_gguf_dir.v;
    if (opt.us_dit_w.set)       uscfg.dit_w       = opt.us_dit_w.v;
    if (opt.us_vae_w.set)       uscfg.vae_w       = opt.us_vae_w.v;
    if (opt.us_cnd_w.set)       uscfg.cnd_w       = opt.us_cnd_w.v;
    if (opt.us_meta.set)        uscfg.meta        = opt.us_meta.v;

    qr::QuadCfg qcfg;
    if (opt.quad_repo.set) qcfg.repo = opt.quad_repo.v;
    imretopo::ImCfg imcfg;
    if (opt.im_adaptivity.set)   imcfg.adaptivity   = opt.im_adaptivity.v;
    if (opt.im_target_verts.set) imcfg.target_verts = opt.im_target_verts.v;

    std::vector<texproj::ViewSpec> tex_views;
    tex_views.reserve(opt.tex_views.size());
    for (const auto& v : opt.tex_views) { texproj::ViewSpec s; s.yaw_deg = v.yaw_deg; s.img = v.img; tex_views.push_back(s); }
    if (model.empty() || image.empty() || out.empty()) { usage(); return 1; }
    if ((geometry_only || material_cache_only) && stage_dir.empty()) { printf("--geometry-only/--material-cache-only requires --stage-dir\n"); return 1; }
    if (geometry_only && material_cache_only) { printf("--geometry-only and --material-cache-only are mutually exclusive\n"); return 1; }
    if (fast) { setenv("PIXAL3D_FAST", "1", 1); setenv("GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F", "1", 1); }

    printf("==== image_to_rig (inline native image -> textured+rigged GLB) ====\n");
    // --dc-remesh resolves its implied defaults here so the banner below reports what will actually run.
    if (dc_remesh) {
        if (!refine_set)   refine = false;      // parity mesh IS the coarse O-Voxel-DC surface
        if (!texsize_set)  texsize = 2048;
        if (!decimate_set) decimate = 220000;
    }

    printf("  model=%s\n  image=%s\n  out=%s\n  cam: fov=%.4frad dist=%.4f scale=%.2f  backend=%s\n",
           model.c_str(), image.c_str(), out.c_str(), cam, dist, ms, use_cuda ? "cuda" : "cpu");
    if (dc_remesh)
        printf("  dc-remesh=yes (Python-parity O-Voxel narrow-band DC, band=%d, taubin=%d; refine=%s, texsize=%d, decimate=%d)\n",
               dc_band, dc_taubin, refine ? "ON (explicit)" : "off", texsize, decimate);
    if (dc_remesh)
        printf("  solid=%s\n", solid_mesh ? "yes (signed field + buried-shell drop -> ONE watertight manifold solid)"
                                           : "NO (default: historical two-walled DC envelope -- pass --solid for the signed one-wall solid; it breaks the rig)");
    if (geometry_only) {
        printf("  geometry-only=yes (legacy PBR material, UV bake, and rig are skipped)\n");
    } else if (material_cache_only) {
        printf("  material-cache-only=yes (native Pixal PBR volume retained; UV bake and rig are skipped)\n");
    } else {
        printf("  tex-dit=%s%s%s\n", in.tex_proj ? "proj (slat_flow_imgshape2tex_1024)" : "cross (trellis2_tex_1024)",
               in.tex_flow_w.empty() ? "" : " w=", in.tex_flow_w.c_str());
    }

    // ---------- [0/4] native MoGe camera estimation (image -> FOV), 100% camera-native ----------
    if (use_moge) {
        double t_m = pix::now_s();
        float ang = moge::estimate_cam_angle_x(image, moge_w, use_cuda, true);
        // MUST set dist with cam -- they are the same parameter (see the --fov note above). MoGe even
        // COMPUTES this and we were throwing it away: its logged fx_norm=1.163723 IS 0.5/tan(ang/2)
        // = 1.163724. Without this line --moge is a zoom, not a camera, and it clips the model through
        // the floor of the canonical box and then OOMs the refine.
        if (ang > 0.05f && ang < 3.0f) { cam = ang; dist = 0.5f / std::tan(cam * 0.5f);
            printf("  [0/4] MoGe camera: fov=%.4frad (%.2fdeg) dist=%.6f  (%.1fs)\n", cam, cam*180.0f/(float)M_PI, dist, pix::now_s()-t_m); }
        else printf("  [0/4] MoGe returned implausible fov=%.4f — keeping cam=%.4f\n", ang, cam);
    }

    // A later observed-view overlay must use the same camera convention that
    // created/refined this mesh. Persist the resolved values with the stage
    // cache instead of requiring a fragile launcher-log scrape. This record is
    // CPU-only provenance: it neither changes geometry nor invokes another
    // model, and it is useful even when the default/caller camera was used.
    if (!stage_dir.empty()) {
        mkdir(stage_dir.c_str(), 0755);
        std::ofstream cf(stage_dir + "/camera_provenance.txt", std::ios::trunc);
        if (!cf) { std::fprintf(stderr, "WARN: could not write camera provenance in %s\n", stage_dir.c_str()); }
        else {
            cf << "schema_version=1\n"
               << "camera_source=" << (use_moge ? "moge" : "caller-or-default") << '\n'
               << "cam_angle_x_rad=" << cam << '\n'
               << "camera_distance=" << dist << '\n'
               << "mesh_scale=" << ms << '\n'
               << "image=" << image << '\n';
            std::printf("  [stage] wrote %s/camera_provenance.txt\n", stage_dir.c_str());
        }
    }

    cancelhook::check();   // stage boundary
    // ---------- [1/4] geometry + texture (in-process, GPU) ----------
    setenv("PIXAL3D_GGUF_DIR", model.c_str(), 1);
    try {
        // matte_native:: is a DROP-IN for imgio::load_pixal_matte_chw with the same framing contract.
        // The difference is what happens to an input that is NOT already a matte: the border-flood
        // heuristic picks a different silhouette than Python's (63.6% stage-1 agreement — a collapse),
        // so that case used to be handed to a CONTAINER (`docker run … vision-cli`) per image. This
        // runs RMBG-2.0 in-process instead, under the same 3060 lock as the rest of the pipeline:
        // 2.4 s -> 0.79 s with the model resident, and identical alpha>0.8 crop box (silhouette
        // IoU 1.0000) so nothing downstream moves. Inputs already classified rgba-cutout/black-matte
        // never touch the model at all, which is why the existing corpus is unaffected.
        in.img512_raw  = image_model_ready ? imgio::load_chw(image, 512) : matte_native::load_pixal_matte_chw(image, 512);
        in.img1024_raw = image_model_ready ? imgio::load_chw(image, 1024) : matte_native::load_pixal_matte_chw(image, 1024);
    } catch (const cancelhook::Cancelled&) { throw;   // a cancel is not a load failure
    } catch (const std::exception& e) { printf("image load failed: %s\n", e.what()); return 1; }
    in.cam = cam; in.dist = dist; in.ms = ms; in.use_cuda = use_cuda; in.verbose = true;
    // Keep the established M4 mesh-decode layout (including its auxiliary subdivision output) exactly
    // as the textured path. Geometry-only suppresses M6 by passing no texture output buffers below;
    // flipping this flag changes M4's CUDA path and is not a harmless performance optimisation.
    in.textured = true; in.watertight = true; in.remesh = false;
    // --dc-remesh keeps the raw O-Voxel dual-grid mesh out of the chain (mc_remesh would replace it
    // with the MC-solid staircase); the DC remesh below is what makes it manifold + smooth.
    in.mc_remesh = clean && !dc_remesh; in.mc_stride = mc_stride; in.mc_blur = mc_blur; in.mc_post_smooth = mc_smooth;
    in.norm_mean = load_norm(model, "shape_slat", "mean");
    in.norm_std  = load_norm(model, "shape_slat", "std");
    in.tex_mean  = load_norm(model, "tex_slat", "mean");
    in.tex_std   = load_norm(model, "tex_slat", "std");

    double t_geo = pix::now_s();
    pix::ChainStats st;
    std::vector<float> pbr_feats;
    std::vector<int32_t> pbr_coords, coords1024;
    svae::Mesh mesh;
    // small binary blob I/O for the geometry cache (size-prefixed contiguous arrays)
    auto sv_vec = [](const std::string& p, const void* d, size_t bytes){ FILE* f=fopen(p.c_str(),"wb"); if(!f) return false; size_t n=bytes; fwrite(&n,sizeof(n),1,f); fwrite(d,1,bytes,f); fclose(f); return true; };
    auto ld_vec = [](const std::string& p, std::vector<uint8_t>& out)->bool{ FILE* f=fopen(p.c_str(),"rb"); if(!f) return false; size_t n=0; if(fread(&n,sizeof(n),1,f)!=1){fclose(f);return false;} out.resize(n); size_t r=fread(out.data(),1,n,f); fclose(f); return r==n; };

    // little-endian int32 header I/O for the resume caches (resolution tag)
    auto sv_i32 = [&](const std::string& p, int32_t v){ return sv_vec(p, &v, sizeof(v)); };
    auto ld_i32 = [&](const std::string& p, int32_t& v)->bool{ std::vector<uint8_t> b; if(!ld_vec(p,b)||b.size()!=sizeof(v)) return false; v=*(int32_t*)b.data(); return true; };
    bool did_refine_load = false;   // --from-refined skips BOTH geometry diffusion and UltraShape refine
    // Reproject dense shell = the pixal COARSE mesh (kept in the volume [-0.5,0.5] frame). When UltraShape
    // refine runs, the refined surface differs proportionally from the coarse mesh (UltraShape is a
    // GENERATIVE densify, not a rigid subdivide) — a uniform bbox-canon aligns the longest axis but leaves
    // per-axis residuals (e.g. +20% depth), so a DIRECT PBR-volume grid_sample on the refined surface
    // misses the sparse voxel shell and bakes BLACK. Fix (mirrors run_pipeline.sh's tex_reproject
    // RP_CANON_TO_DENSE): bake the refined mesh with reproject=true against this coarse shell — each texel
    // snaps to the nearest coarse-surface point (which IS aligned to the volume) and samples there.
    std::vector<float>   shell_verts;
    std::vector<int64_t> shell_faces;

    if (!from_refined.empty()) {
        // ---- re-enter from a cached REFINED mesh: skip geometry + UltraShape entirely (iterate bake/rig).
        //      Loads the PBR volume + resolution tag + the refined mesh GLB (already in the pixal frame). ----
        std::vector<uint8_t> b;
        if (!ld_vec(from_refined + "/pbr_feats.bin", b)) { printf("FAIL: --from-refined missing pbr_feats.bin in %s\n", from_refined.c_str()); return 1; }
        pbr_feats.assign((float*)b.data(), (float*)(b.data()+b.size()));
        if (!ld_vec(from_refined + "/pbr_coords.bin", b)) { printf("FAIL: --from-refined missing pbr_coords.bin\n"); return 1; }
        pbr_coords.assign((int32_t*)b.data(), (int32_t*)(b.data()+b.size()));
        int32_t rres = 0; if (ld_i32(from_refined + "/resolution.bin", rres) && rres > 0) in.resolution = rres;
        glb::Mesh gm;
        if (!glb::read_glb((from_refined + "/refined.glb").c_str(), gm)) { printf("FAIL: --from-refined missing refined.glb\n"); return 1; }
        mesh.verts = gm.verts; mesh.faces = gm.faces; mesh.N = (int)(gm.verts.size()/3); mesh.F = (int)(gm.faces.size()/3);
        did_refine_load = true;
        // reproject shell for the refined mesh = the cached coarse.glb (volume frame), if present.
        glb::Mesh cm;
        if (glb::read_glb((from_refined + "/coarse.glb").c_str(), cm)) { shell_verts = cm.verts; shell_faces = cm.faces; }
        printf("  [1/4] from-refined: loaded refined mesh (%d v / %d f) + %zu PBR voxels @res%d + shell(%zu v) from %s\n",
               mesh.N, mesh.F, pbr_coords.size()/4, in.resolution, shell_verts.size()/3, from_refined.c_str());
    } else if (!from_geo.empty()) {
        // ---- re-enter from cached geometry: skip the ~420s diffusion, rebuild the manifold mesh ----
        std::vector<uint8_t> b;
        if (!ld_vec(from_geo + "/coords1024.bin", b)) { printf("FAIL: --from-geo missing coords1024.bin in %s\n", from_geo.c_str()); return 1; }
        coords1024.assign((int32_t*)b.data(), (int32_t*)(b.data()+b.size()));
        if (!ld_vec(from_geo + "/pbr_feats.bin", b)) { printf("FAIL: --from-geo missing pbr_feats.bin\n"); return 1; }
        pbr_feats.assign((float*)b.data(), (float*)(b.data()+b.size()));
        if (!ld_vec(from_geo + "/pbr_coords.bin", b)) { printf("FAIL: --from-geo missing pbr_coords.bin\n"); return 1; }
        pbr_coords.assign((int32_t*)b.data(), (int32_t*)(b.data()+b.size()));
        printf("  [1/4] from-geo: loaded %zu occ voxels, %zu PBR voxels from %s\n",
               coords1024.size()/4, pbr_coords.size()/4, from_geo.c_str());
        if (!clean) { printf("FAIL: --from-geo only supports the clean (MC-remesh) path\n"); return 1; }
        // The geo cache holds the binary occupancy, not the O-Voxel decoder mesh, so a DC of it would
        // contour the STAIRCASE — the exact mistake the parity work root-caused. Refuse instead.
        if (dc_remesh) { printf("FAIL: --dc-remesh needs the O-Voxel decoder mesh; --from-geo only caches the occupancy grid\n"); return 1; }
        mesh = pix::build_mc_remesh(coords1024, in.resolution, mc_stride, mc_blur, mc_smooth, true);
    } else {
        // The occupancy is wanted whenever --dump-geo caches it AND whenever --dc-remesh is going to
        // sign its distance field with it. It is a plain copy of a vector the decoder already holds.
        mesh = pix::run_geometry(in, &st, nullptr,
                                 geometry_only ? nullptr : &pbr_feats,
                                 geometry_only ? nullptr : &pbr_coords,
                                 (dump_geo.empty() && !dc_remesh) ? nullptr : &coords1024);
        printf("  [1/4] geometry: N1=%d M=%d verts=%d faces=%d  (%.1fs)\n",
               st.N1, st.M, mesh.N, mesh.F, pix::now_s() - t_geo);
        if (!dump_geo.empty()) {
            std::string d = dump_geo;
            mkdir(d.c_str(), 0755);   // create the cache dir if missing (idempotent)
            bool ok1 = sv_vec(d + "/coords1024.bin", coords1024.data(), coords1024.size()*sizeof(int32_t));
            bool ok2 = sv_vec(d + "/pbr_feats.bin",  pbr_feats.data(),  pbr_feats.size()*sizeof(float));
            bool ok3 = sv_vec(d + "/pbr_coords.bin", pbr_coords.data(), pbr_coords.size()*sizeof(int32_t));
            if (!(ok1 && ok2 && ok3)) { printf("FAIL: --dump-geo could not write to %s\n", d.c_str()); return 1; }
            printf("  [1/4] dump-geo: cached coords1024(%zu)/pbr_feats(%zu)/pbr_coords(%zu) -> %s\n",
                   coords1024.size(), pbr_feats.size(), pbr_coords.size(), d.c_str());
        }
    }

    cancelhook::check();   // stage boundary
    // ---------- [1a0/4] --dc-remesh: Python-parity narrow-band DC of the O-Voxel mesh ----------
    // `mesh` here is the raw dual-grid decoder surface (smooth QEF vertices, but non-manifold and
    // ~3.4M faces). Dual-contouring its own narrow-band UDF is what Python's to_glb(remesh=True)
    // does, and it yields the smooth watertight parity surface. The RAW mesh is kept as the bake's
    // reproject shell: it is the surface the PBR volume was decoded on, so it stays volume-aligned.
    if (dc_remesh && !did_refine_load) {
        double t_dc = pix::now_s();
        printf("  [1a0/4] dc-remesh: narrow-band DC of the O-Voxel mesh (%d v / %d f, res=%d band=%d)...\n",
               mesh.N, mesh.F, in.resolution, dc_band);
        shell_verts = mesh.verts; shell_faces = mesh.faces;   // volume-aligned generation surface
        // SOLID DELIVERY: build the inside/outside oracle from the decoder's own occupancy and hand
        // it to the DC, which then contours a SIGNED field.  Without it the DC contours {udf == eps}
        // of an unsigned field -- two level sets, hence the two-walled envelope.  See solid_field.hpp
        // for why a voxel occupancy is accurate enough (the sign only matters where udf > eps).
        solidfield::SolidField interior;
        if (solid_mesh) {
            if (coords1024.empty()) {
                printf("  [1a0/4] solid: no occupancy available (resumed geometry?) -- keeping the "
                       "unsigned two-walled DC. Pass --no-solid to silence this.\n");
            } else {
                const double t_sf = pix::now_s();
                const int seal  = std::getenv("NBDC_SOLID_SEAL")  ? atoi(std::getenv("NBDC_SOLID_SEAL"))  : 1;
                const int erode = std::getenv("NBDC_SOLID_ERODE") ? atoi(std::getenv("NBDC_SOLID_ERODE")) : 1;
                interior = solidfield::build(coords1024.data(), (int)(coords1024.size()/4),
                                             in.resolution, seal, erode, true);
                printf("  [1a0/4] solid: inside/outside oracle from %zu occupancy voxels (%.1fs)%s\n",
                       coords1024.size()/4, pix::now_s()-t_sf,
                       interior.empty() ? "  -- LEAKED, falling back to the unsigned field" : "");
            }
        }
        std::vector<float> dcv; std::vector<int64_t> dcf;
        if (!nbdc::remesh(mesh.verts, mesh.faces, in.resolution, (float)dc_band, dcv, dcf, true,
                          interior.empty() ? nullptr : &interior)) {
            printf("FAIL: --dc-remesh narrow-band dual contour\n"); return 1;
        }
        if (!interior.empty()) {
            // Drop BURIED SHELLS.  What is left of the inward wall comes off as closed components
            // with INWARD normals, i.e. negative signed volume -- they bound a void inside the body
            // rather than a body.  That is a containment test, not a size test, which is the point:
            // "keep the biggest component" would delete the legitimately separate parts of an
            // articulated subject, whereas a separate part is a real body and its volume is
            // POSITIVE however small it is.  The DC's winding follows the field's sign globally, so
            // the test is exact rather than heuristic.
            const double t_fl = pix::now_s();
            int64_t dropped_f = 0, dropped_c = 0;
            {
                const int64_t V = (int64_t)(dcv.size()/3), F = (int64_t)(dcf.size()/3);
                std::vector<int64_t> par(V); for (int64_t i=0;i<V;i++) par[i]=i;
                std::function<int64_t(int64_t)> find = [&](int64_t x){
                    while (par[x]!=x){ par[x]=par[par[x]]; x=par[x]; } return x; };
                auto uni=[&](int64_t a,int64_t b){ int64_t ra=find(a),rb=find(b); if(ra!=rb) par[ra]=rb; };
                for (int64_t t=0;t<F;t++){ uni(dcf[t*3],dcf[t*3+1]); uni(dcf[t*3+1],dcf[t*3+2]); }
                std::unordered_map<int64_t,double> vol;
                std::unordered_map<int64_t,int64_t> cnt;
                for (int64_t t=0;t<F;t++){
                    const int64_t i0=dcf[t*3],i1=dcf[t*3+1],i2=dcf[t*3+2];
                    const float *p0=&dcv[i0*3],*p1=&dcv[i1*3],*p2=&dcv[i2*3];
                    const double cr[3]={ (double)p1[1]*p2[2]-(double)p1[2]*p2[1],
                                         (double)p1[2]*p2[0]-(double)p1[0]*p2[2],
                                         (double)p1[0]*p2[1]-(double)p1[1]*p2[0] };
                    const int64_t root = find(i0);
                    vol[root] += (p0[0]*cr[0]+p0[1]*cr[1]+p0[2]*cr[2])/6.0;
                    cnt[root] += 1;
                }
                std::vector<int64_t> keep;
                keep.reserve(dcf.size());
                for (int64_t t=0;t<F;t++){
                    const int64_t root = find(dcf[t*3]);
                    if (vol[root] > 0.0) { keep.push_back(dcf[t*3]); keep.push_back(dcf[t*3+1]); keep.push_back(dcf[t*3+2]); }
                }
                for (const auto& kv : vol) if (kv.second <= 0.0) { ++dropped_c; dropped_f += cnt[kv.first]; }
                dcf.swap(keep);
                // compact vertices
                std::vector<int64_t> remap(V, -1); std::vector<float> nv; nv.reserve(dcv.size());
                for (int64_t& idx : dcf) { int64_t& r = remap[idx];
                    if (r < 0) { r = (int64_t)(nv.size()/3); nv.push_back(dcv[idx*3]); nv.push_back(dcv[idx*3+1]); nv.push_back(dcv[idx*3+2]); }
                    idx = r; }
                dcv.swap(nv);
            }
            printf("  [1a0/4] solid: dropped %lld buried shells (%lld faces, inward normals) (%.1fs)\n",
                   (long long)dropped_c, (long long)dropped_f, pix::now_s()-t_fl);
        }
        if (dc_taubin > 0) meshsmooth::taubin(dcv, dcf, dc_taubin);
        mesh.verts = std::move(dcv); mesh.faces = std::move(dcf);
        mesh.N = (int)(mesh.verts.size()/3); mesh.F = (int)(mesh.faces.size()/3);
        if (!interior.empty()) {
            // NO orient_consistent() here.  The DC's winding already follows the sign of the field,
            // so a signed field yields globally outward normals for free -- measured: main body
            // +0.0142, every dropped shell negative.  Running svae::orient_consistent on top MAKES
            // IT WORSE: its BFS propagates orientation across the ~200 remaining non-manifold edges,
            // where "the neighbour across this edge" is not well defined, and it flipped enough of
            // the body to cut the enclosed volume from +0.0142 to +0.0059 (miku) and +0.0101 to
            // +0.0013 (gilly).  Weld only.
            const mesh_exact_clean::Report r = mesh_exact_clean::clean(mesh.verts, mesh.faces);
            mesh.N = (int)(mesh.verts.size()/3); mesh.F = (int)(mesh.faces.size()/3);
            int64_t b = 0, nm = 0; svae::mesh_topology_stats(mesh, b, nm);
            printf("  [1a0/4] solid: welded (welded %lld, degenerate %lld, duplicate %lld)"
                   " -> boundary=%lld nonmanifold=%lld\n",
                   (long long)r.welded_vertices, (long long)r.dropped_degenerate_faces,
                   (long long)r.dropped_duplicate_faces, (long long)b, (long long)nm);
        }
        printf("  [1a0/4] dc-remesh: -> %d v / %d f (taubin x%d)  (%.1fs)\n", mesh.N, mesh.F, dc_taubin, pix::now_s()-t_dc);
        if (!stage_dir.empty()) {
            mkdir(stage_dir.c_str(), 0755);
            // coarse.glb = the RAW O-Voxel generation surface (what the PBR volume was decoded on and
            // what the bake reprojects against); dc.glb = the parity delivery mesh. They are also
            // written under the names --from-refined expects, so `--dc-remesh --from-refined <dir>`
            // re-enters at the bake with no second diffusion — that is the LOD-tier / retexture loop.
            if (glb::write_glb((stage_dir + "/coarse.glb").c_str(), shell_verts, shell_faces))
                printf("  [stage] wrote %s/coarse.glb (raw O-Voxel mesh = bake shell)\n", stage_dir.c_str());
            if (glb::write_glb((stage_dir + "/dc.glb").c_str(), mesh.verts, mesh.faces))
                printf("  [stage] wrote %s/dc.glb (parity delivery mesh, %d v / %d f)\n", stage_dir.c_str(), mesh.N, mesh.F);
            glb::write_glb((stage_dir + "/refined.glb").c_str(), mesh.verts, mesh.faces);   // --from-refined alias
            sv_vec(stage_dir + "/pbr_feats.bin",  pbr_feats.data(),  pbr_feats.size()*sizeof(float));
            sv_vec(stage_dir + "/pbr_coords.bin", pbr_coords.data(), pbr_coords.size()*sizeof(int32_t));
            sv_i32(stage_dir + "/resolution.bin", (int32_t)in.resolution);
            printf("  [stage] pbr cache written (resume: --dc-remesh --from-refined %s)\n", stage_dir.c_str());
        }
    }

    // stage-dir: snapshot the coarse (pre-refine) mesh as a named eye-test artifact.
    // (--dc-remesh staged its own coarse.glb above — the RAW mesh, not the DC output.)
    if (!stage_dir.empty() && !did_refine_load && !dc_remesh) {
        mkdir(stage_dir.c_str(), 0755);
        if (glb::write_glb((stage_dir + "/coarse.glb").c_str(), mesh.verts, mesh.faces))
            printf("  [stage] wrote %s/coarse.glb (%d v / %d f)\n", stage_dir.c_str(), mesh.N, mesh.F);
        // The PBR voxel coords are indices on the grid-`in.resolution` lattice, so the lattice scale
        // has to be staged with the volume for any consumer (--from-refined resume, the CPU texture
        // rebake) to sample it in the right coordinate space.  Tag it here rather than only on the
        // refine path so a --no-refine stage cache is never left untagged.
        sv_i32(stage_dir + "/resolution.bin", (int32_t)in.resolution);
    }

    cancelhook::check();   // stage boundary
    // ---------- [1a/4] UltraShape refine (native, in-process): clean/watertight ~7.5x densify ----------
    if (refine && !did_refine_load) {
        double t_ref = pix::now_s();
        printf("  [1a/4] UltraShape refine: coarse %d v / %d f -> refined ...\n", mesh.N, mesh.F);
        std::vector<float> coarse_ref = mesh.verts;   // canon target (pixal [-0.5,0.5] frame)
        shell_verts = mesh.verts; shell_faces = mesh.faces;   // reproject dense shell (coarse, volume frame)
        usr::RefineCfg rc = uscfg;
        svae::Mesh refined;
        try {
            refined = usr::refine(mesh.verts, mesh.faces, image, rc, use_cuda);
        } catch (const cancelhook::Cancelled&) { throw;   // a cancel is not a refine failure
        } catch (const std::exception& e) { printf("FAIL: UltraShape refine: %s\n", e.what()); return 1; }
        // re-canonicalize the ±1 UltraShape mesh onto the coarse-mesh bbox so the PBR volume bakes onto it.
        bbox_canon_onto(refined.verts, coarse_ref);
        // The refinement kernel already removes bit-identical vertices in its ±1
        // frame.  Re-canonicalising to the Pixal frame is another float transform,
        // though, and can make two formerly distinct endpoints collapse when the
        // GLB's f32 positions are written.  Repeat the exact-only cleanup *after*
        // that final coordinate transform so the staged delivery mesh itself, not
        // merely the internal UltraShape mesh, is the topology contract.
        {
            const mesh_exact_clean::Report clean = mesh_exact_clean::clean(refined.verts, refined.faces);
            refined.N = (int)(refined.verts.size() / 3);
            refined.F = (int)(refined.faces.size() / 3);
            if (clean.welded_vertices || clean.dropped_degenerate_faces || clean.dropped_duplicate_faces)
                printf("  [1a/4] final-frame exact cleanup: V %lld->%lld (welded %lld), F %lld->%lld (degenerate %lld, duplicate %lld)\n",
                       (long long)clean.input_vertices, (long long)clean.output_vertices,
                       (long long)clean.welded_vertices, (long long)clean.input_faces,
                       (long long)clean.output_faces, (long long)clean.dropped_degenerate_faces,
                       (long long)clean.dropped_duplicate_faces);
        }
        mesh = std::move(refined);
        printf("  [1a/4] UltraShape refine: -> %d v / %d f  (%.1fs)\n", mesh.N, mesh.F, pix::now_s() - t_ref);
        // stage-dir: emit refined.glb + cache the PBR volume + resolution for --from-refined resume.
        if (!stage_dir.empty()) {
            mkdir(stage_dir.c_str(), 0755);
            glb::write_glb((stage_dir + "/refined.glb").c_str(), mesh.verts, mesh.faces);
            sv_vec(stage_dir + "/pbr_feats.bin",  pbr_feats.data(),  pbr_feats.size()*sizeof(float));
            sv_vec(stage_dir + "/pbr_coords.bin", pbr_coords.data(), pbr_coords.size()*sizeof(int32_t));
            sv_i32(stage_dir + "/resolution.bin", (int32_t)in.resolution);
            printf("  [stage] wrote %s/refined.glb + pbr cache (resume: --from-refined %s)\n",
                   stage_dir.c_str(), stage_dir.c_str());
        }
    }

    if (geometry_only || material_cache_only) {
        // The stage cache is the delivery boundary for the native texture runner.  No legacy UV bake,
        // PNG atlas, or skeleton is generated here, so those artefacts cannot be accidentally promoted.
        if (!did_refine_load && !refine) {
            glb::write_glb((stage_dir + "/refined.glb").c_str(), mesh.verts, mesh.faces);
        }
        printf("==== DONE (--%s) -> %s/refined.glb  (verts=%d faces=%d, %.1fs total) ====\n",
               geometry_only ? "geometry-only" : "material-cache-only", stage_dir.c_str(), mesh.N, mesh.F, pix::now_s() - t_geo);
        return 0;
    }

    // Dense high-poly kept for the normal-map bake (retopo/decimation drops fine relief — anime faces,
    // fingers — which the retopo mesh then re-acquires as a tangent-space normal map baked from here).
    std::vector<float>   nrm_src_verts;
    std::vector<int64_t> nrm_src_faces;

    cancelhook::check();   // stage boundary
    // ---------- [1a2/4] quad retopology (rung 2, quadwild-bimdf shell-out; CPU) ----------
    // Runs on the refined mesh (fresh-refine or --from-refined). Produces clean field-aligned quads,
    // TRIANGULATED into `mesh` for the tri-only bake/rig/glb path (the field-aligned edge flow survives
    // the fan-triangulation). The quad mesh IS the clean retopo, so the bake must not re-decimate it.
    if (do_quad) {
        double t_q = pix::now_s();
        const bool use_im = (retopo_tool != "quadwild");
        printf("  [1a2/4] retopo (%s) on %d v / %d f (refined)...\n", use_im ? "instant-meshes" : "quadwild", mesh.N, mesh.F);
        svae::Mesh quad;
        bool ok = use_im ? imretopo::im_retopo(mesh, imcfg, quad) : qr::quad_retopo(mesh, qcfg, quad);
        if (!ok) { printf("FAIL: retopo (%s)\n", use_im ? "im" : "quadwild"); return 1; }
        nrm_src_verts = mesh.verts; nrm_src_faces = mesh.faces;   // dense pre-retopo mesh -> normal-map source
        mesh = std::move(quad);
        decimate = 0;   // retopo mesh is the intended final topology — keep it through the bake
        printf("  [1a2/4] retopo: -> %d v / %d f  (%.1fs)\n", mesh.N, mesh.F, pix::now_s() - t_q);
        if (!stage_dir.empty()) {
            mkdir(stage_dir.c_str(), 0755);
            if (glb::write_glb((stage_dir + "/quad.glb").c_str(), mesh.verts, mesh.faces))
                printf("  [stage] wrote %s/quad.glb (%d v / %d f, triangulated quads)\n", stage_dir.c_str(), mesh.N, mesh.F);
        }
    }

    cancelhook::check();   // stage boundary
    // ---------- [1b/4] optional P3-SAM part-retopo (native, before bake) ----------
    if (part_retopo) {
        double t_pr = pix::now_s();
        printf("  [1b/4] part-retopo: P3-SAM segmenting %d faces (native CPU; slow, GPU port pending)...\n", mesh.F);
        auto fids = p3sam::segment_mesh(mesh.verts.data(), mesh.N, mesh.faces.data(), mesh.F, p3sam_w, in.seed);
        nrm_src_verts = mesh.verts; nrm_src_faces = mesh.faces;   // dense pre-decimate mesh -> normal-map source
        glb::Mesh gm; gm.verts = mesh.verts; gm.faces = mesh.faces;
        glb::Mesh dec; std::vector<ppd::PartReport> rep;
        if (!ppd::per_part_decimate(gm, fids, obj_decimate, dec, rep)) { printf("FAIL: part-retopo\n"); return 1; }
        mesh.verts = dec.verts; mesh.faces = dec.faces;
        mesh.N = (int)(dec.verts.size() / 3); mesh.F = (int)(dec.faces.size() / 3);
        decimate = 0;   // bake must NOT re-decimate the already part-retopo'd mesh
        size_t nparts = 0; { std::vector<int64_t> u(fids); std::sort(u.begin(), u.end()); u.erase(std::unique(u.begin(), u.end()), u.end()); nparts = u.size(); }
        printf("  [1b/4] part-retopo: %zu parts, faces -> %d  (%.0fs)\n", nparts, mesh.F, pix::now_s() - t_pr);
        if (!stage_dir.empty()) {
            mkdir(stage_dir.c_str(), 0755);
            if (glb::write_glb((stage_dir + "/decimated.glb").c_str(), mesh.verts, mesh.faces))
                printf("  [stage] wrote %s/decimated.glb (%d v / %d f)\n", stage_dir.c_str(), mesh.N, mesh.F);
        }
    }

    // UV-atlas PBR bake -> TEXTURED mesh in RAM. CLEAN (manifold) path: precluster=true (normal-cone
    // clustering grows large charts on the manifold surface → ~18s bake, ~14k clusters, not the >400s /
    // tens-of-thousands-of-charts shatter on the non-manifold staircase). Raw path keeps precluster=false.
    double t_bake = pix::now_s();
    // LOD / "game asset" tiers: the bake decimates internally, and decimation is exactly where the
    // dense surface relief goes. Keep the pre-bake mesh as the normal-map source whenever the target
    // is a real reduction, so the low-poly tier renders WITH that detail instead of losing it. The
    // retopo paths above already set this; RETOPO_NO_NORMAL=1 opts out of the bake either way.
    if (nrm_src_verts.empty() && decimate > 0 && mesh.F > 2 * decimate && !std::getenv("RETOPO_NO_NORMAL")) {
        nrm_src_verts = mesh.verts; nrm_src_faces = mesh.faces;
        printf("  [1c/4] normal-map source: pre-decimation mesh (%d v / %d f -> ~%d faces)\n",
               mesh.N, mesh.F, decimate);
    }
    // The DC remesh output is manifold and clean, so it takes the precluster path too (the raw
    // O-Voxel mesh is what shatters xatlas, and that mesh is now only the reproject shell).
    const bool precluster = clean || dc_remesh;
    // REPROJECT bake when we have a coarse dense shell (i.e. UltraShape refine ran): snap each texel of
    // the refined/decimated mesh onto the coarse shell (volume frame) then grid_sample the PBR volume at
    // the snapped point. Robust to the refined mesh's proportional drift from the coarse mesh (the black-
    // texture fix). Without a shell (coarse-only / --no-refine) the mesh IS the volume surface → the
    // direct volume grid_sample aligns, so keep the cheaper non-reproject path.
    const bool use_reproject = !shell_faces.empty();
    // Per-vertex PBR on the coarse shell (6-ch). The shell sits in the volume [-0.5,0.5] frame, so a texel
    // maps to grid coord (v+0.5)*grid_res — sample the PBR volume there. This colours the shell (== the
    // production dump_dense), which the reproject bake snaps refined texels onto. Required because bake's
    // reproject gate needs a non-null dense_attr (tex_atlas.hpp:960).
    std::vector<float> shell_attr;
    if (use_reproject) {
        const int C = 6, SV = (int)(shell_verts.size()/3);
        shell_attr.resize((size_t)SV * C);
        texgs::VolIndex vol(pbr_coords.data(), (int)pbr_coords.size()/4, 4, 1);
        #pragma omp parallel for schedule(dynamic, 4096)
        for (int i = 0; i < SV; i++) {
            float q0=(shell_verts[i*3]+0.5f)*in.resolution, q1=(shell_verts[i*3+1]+0.5f)*in.resolution, q2=(shell_verts[i*3+2]+0.5f)*in.resolution;
            texgs::sample_one(vol, pbr_feats.data(), C, q0,q1,q2, &shell_attr[(size_t)i*C], 2);
        }
        if (dc_remesh && !refine) {
            // PARITY BAKE (= texture_rebake_native --bake volume-trilinear, which is what produced the
            // proven parity.glb). The mesh being textured is the DC of the shell, i.e. essentially the
            // same surface, so the volume is read DIRECT at each texel's own position — one projection,
            // no cross-surface slide (RP_ATTR/snap is what smeared tie colour onto the collar). The
            // shell stays only as the guard for texels the sparse volume has no data near.
            unsetenv("RP_ATTR"); setenv("RP_DIRECT", "1", 1);
            setenv("RP_FRONTDOT", "-1", 0);        // guard snap takes the globally closest tri, like Python's BVH
            setenv("ATL_BASECOLOR_SRGB", "0", 0);  // Python stores the RAW volume colour; sRGB OETF washes darks
            setenv("TEX_TELEA_INPAINT", "1", 0);
            setenv("TEX_TELEA_RADIUS", "3", 0);
            setenv("TEX_FILL_BACKGROUND", "1", 0); // fill the gutter or chart edges bleed black = seam web
        } else {
            // default reproject mode = mesh-attr (barycentric shell colours); --tex-snap-volume opts out.
            if (!tex_snap_volume) setenv("RP_ATTR", "1", 1); else unsetenv("RP_ATTR");
            // --tex-volume-direct: read the volume at each texel's own position; the mode above becomes the
            // guard for texels the volume has no data near. Default OFF = today's behaviour.
            if (tex_volume_direct) setenv("RP_DIRECT", "1", 1); else unsetenv("RP_DIRECT");
        }
    }
    // sample_fallback_r: 0 keeps today's snap-mode behaviour bit-for-bit; volume-direct needs a real
    // radius or every texel starves to black (that starvation IS the "black texture bug").
    const int bake_fbr = tex_fallback_r >= 0 ? tex_fallback_r
                       : ((tex_volume_direct || (dc_remesh && !refine)) ? 8 : 0);
    texatlas::BakedTexture bt = use_reproject
        ? texatlas::bake(mesh.verts, mesh.faces, pbr_feats, pbr_coords, in.resolution, texsize, decimate,
                         4, true, bake_fbr, precluster, 55.f, &shell_verts, &shell_faces, &shell_attr, /*reproject=*/true)
        : texatlas::bake(mesh.verts, mesh.faces, pbr_feats, pbr_coords, in.resolution, texsize, decimate,
                         4, true, bake_fbr, precluster);
    printf("  [1/4] tex bake%s: %dx%d, %d charts, %d verts / %zu faces  (%.1fs)\n",
           use_reproject ? " (reproject/shell)" : "", bt.tw, bt.th, bt.chart_count,
           (int)bt.verts.size()/3, bt.faces.size()/3, pix::now_s()-t_bake);

    cancelhook::check();   // stage boundary
    // ---------- [1b/4] tex PROJECTION (--tex-project) ----------
    // Overwrite the volume bake's baseColor with the REAL images projected through the pixal3d camera:
    // front = --image (the matte the mesh was built from = the exact frame, no alignment needed), back =
    // --tex-back (optional, silhouette-bbox auto-aligned). Z-buffered occlusion kills the prototype's
    // "double face in the hat". metallicRoughness + alpha stay from the volume bake. See tex_project.hpp.
    if (tex_project) {
        double t_proj = pix::now_s();
        texproj::Cfg pcfg;
        pcfg.cam = cam; pcfg.dist = dist; pcfg.ms = ms;
        // --tex-front overrides the front source; geometry above still ran on `image` (the lit original).
        pcfg.front_img = tex_front.empty() ? image : tex_front;
        pcfg.back_img  = tex_back;
        pcfg.views     = tex_views;              // extra non-front, non-back yaws (repeatable --tex-view)
        pcfg.preserve_base_for_holes = tex_project_overlay;
        pcfg.verbose   = true;
        pcfg.debug_dir = stage_dir;              // empty = no debug dumps (env TEXPROJ_DEBUG_DIR overrides)
        texproj::Stats ps;
        if (!texproj::project_onto(bt, pcfg, &ps)) { printf("FAIL: tex projection\n"); return 1; }
        if (!tex_front.empty())
            printf("         front source: %s (--tex-front; geometry used %s)\n", tex_front.c_str(), image.c_str());
        std::string vdesc = "front";
        if (!tex_back.empty()) vdesc += "+back";
        for (const auto& v : tex_views) vdesc += "+yaw" + std::to_string((int)std::lround(v.yaw_deg));
        printf("  [1b/4] tex project (%s): front %.1f%% / back %.1f%% / holes %.1f%% of %d covered texels  (%.1fs)\n",
               vdesc.c_str(), ps.front_pct, ps.back_pct, ps.hole_pct, ps.covered, pix::now_s()-t_proj);
        if (ps.view_pct.size() > 2)
            for (size_t v = 0; v < ps.view_pct.size(); v++)
                printf("         view %zu: yaw %+7.1fdeg painted %.1f%% of covered\n", v, ps.view_yaw[v], ps.view_pct[v]);
        // Hole accounting: 3D-filled = respects the surface; telea = the old atlas-space fill (the camo
        // risk). A high telea share on a big smooth region is what to look for in proj_fill_source.png.
        printf("         holes: %d total -> %d 3D-filled (%.1f%%) / %d volume-retained (%.1f%%) / %d telea-fallback (%.1f%%)  [fill %.2fs]\n",
               ps.n_hole, ps.n_fill3d, ps.n_hole ? 100.0*ps.n_fill3d/ps.n_hole : 0.0,
               ps.n_base, ps.n_hole ? 100.0*ps.n_base/ps.n_hole : 0.0,
               ps.n_telea, ps.n_hole ? 100.0*ps.n_telea/ps.n_hole : 0.0, ps.t_fill3d);
        if (!tex_back.empty())
            printf("         back align: %s scale=%.4f translate=(%+.1f, %+.1f) px\n",
                   ps.back_aligned ? "fitted" : "IDENTITY (disabled or degenerate)",
                   ps.back_scale, ps.back_tx, ps.back_ty);
        // SEAM (BUG 3). n_seam = the fill texels where front and back actually meet — the crossover strip.
        // |d| is the disagreement you SEE there; bias is the part a global colour match could remove. A
        // small bias next to a big |d| means the views disagree WITHOUT an exposure offset, so widening/
        // cross-fading the crossover is the lever, not colour matching. A GROWING n_seam means the two
        // painted regions are drifting apart and the fill is inventing more of the crossover.
        if (ps.n_seam)
            printf("         seam: %d fill texels span >=2 views (%.1f%% of 3D fills)  |viewA-viewB| = %.1f/255"
                   " (R%.1f G%.1f B%.1f)  signed bias %+.1f/%+.1f/%+.1f  bias/|d| %.2f  cross-fade=%s\n",
                   ps.n_seam, ps.n_fill3d ? 100.0*ps.n_seam/ps.n_fill3d : 0.0, ps.seam_mean_absdiff,
                   ps.seam_absdiff[0], ps.seam_absdiff[1], ps.seam_absdiff[2],
                   ps.seam_bias[0], ps.seam_bias[1], ps.seam_bias[2],
                   ps.seam_mean_absdiff > 1e-6f
                       ? (std::fabs(ps.seam_bias[0])+std::fabs(ps.seam_bias[1])+std::fabs(ps.seam_bias[2]))
                         / (3.f*ps.seam_mean_absdiff) : 0.f,
                   ps.seam_blend_on ? "ON" : "off");
        printf("         phases: uv-raster %.2fs | zbuf front %.2fs | zbuf back %.2fs | align %.2fs | project %.2fs | fill3d %.2fs | inpaint %.2fs\n",
               ps.t_uv_raster, ps.t_zbuf_front, ps.t_zbuf_back, ps.t_align, ps.t_project, ps.t_fill3d, ps.t_inpaint);
        if (!stage_dir.empty()) {
            mkdir(stage_dir.c_str(), 0755);
            bool have_nrm_p = bt.normals.size() == bt.verts.size();
            if (glb::write_glb_textured((stage_dir + "/proj_tex.glb").c_str(), bt.verts,
                                        have_nrm_p ? bt.normals : std::vector<float>(), bt.uvs, bt.faces,
                                        bt.base_color, bt.metal_rough, bt.tw, bt.th))
                printf("  [stage] wrote %s/proj_tex.glb\n", stage_dir.c_str());
        }
    }

    cancelhook::check();   // stage boundary
    // ---------- [1c/4] normal-map bake (the standard retopo detail lift) ----------
    // The retopo/decimate mesh is intentionally low-poly and drops fine relief (Instant Meshes resamples
    // anime faces to a blank egg; decimation coarsens fingers). Bake a TANGENT-SPACE normal map from the
    // kept dense high-poly onto the low-poly UV atlas so it renders WITH the dense surface detail. Skipped
    // when no retopo ran (nrm_src empty) or the bake has no UVs/normals. RETOPO_NO_NORMAL=1 forces off.
    std::vector<uint8_t> nmap;
    if (!nrm_src_verts.empty() && bt.normals.size() == bt.verts.size() && !bt.uvs.empty()
        && !std::getenv("RETOPO_NO_NORMAL")) {
        double t_nm = pix::now_s();
        std::vector<uint32_t> nf(bt.faces.begin(), bt.faces.end());
        nmap = nrmbake::bake_normal_map(bt.verts, bt.normals, bt.uvs, nf, nrm_src_verts, nrm_src_faces, bt.tw, bt.th);
        printf("  [1c/4] normal map baked from dense high-poly (%zu v / %zu f) -> %dx%d  (%.1fs)\n",
               nrm_src_verts.size()/3, nrm_src_faces.size()/3, bt.tw, bt.th, pix::now_s() - t_nm);
    }

    // --no-rig: write the TEXTURED-only GLB (skip the 159s auto-rig) for fast mesh/texture A/B.
    if (no_rig) {
        std::vector<uint32_t> f32(bt.faces.begin(), bt.faces.end());
        bool have_nrm = bt.normals.size() == bt.verts.size();
        bool okt;
#ifdef PIXAL3D_PACK
        if (!nmap.empty()) {
            // packed writer carries the normal map (KTX2 + meshopt; model-viewer renders it).
            int threads = (int)std::max(1u, std::thread::hardware_concurrency());
            okt = glb::write_glb_textured_packed(out.c_str(), bt.verts, have_nrm ? bt.normals : std::vector<float>(),
                                                 bt.uvs, f32, bt.base_color, bt.metal_rough, bt.tw, bt.th,
                                                 /*uastc*/true, 192, threads, &nmap);
        } else
#endif
        {
            if (!nmap.empty()) printf("  NOTE: normal map baked but binary lacks PIXAL3D_PACK; writing uncompressed WITHOUT normal map\n");
            okt = glb::write_glb_textured(out.c_str(), bt.verts, have_nrm ? bt.normals : std::vector<float>(),
                                          bt.uvs, f32, bt.base_color, bt.metal_rough, bt.tw, bt.th);
        }
        if (!okt) { printf("FAIL: write textured GLB\n"); return 1; }
        printf("==== DONE (--no-rig, textured only%s) -> %s  (verts=%zu faces=%zu, %.1fs total) ====\n",
               nmap.empty() ? "" : " + normal map", out.c_str(), bt.verts.size()/3, bt.faces.size()/3, pix::now_s() - t_geo);
        return 0;
    }

    cancelhook::check();   // stage boundary
    // ---------- [2/4] rig cond points (native, in RAM) ----------
    std::vector<int64_t> faces64(bt.faces.begin(), bt.faces.end());
    rig::PrepResult P;
    rig::RigResult  R;
    // Clear the last-run report here, so a --rig-cache hit (which decodes nothing) cannot leave a
    // previous request's verdict standing as if it were this one's.
    i2r::g_rig_report = rig::StageReport{};

    // --rig-cache: SHARE ONE SKELETON ACROSS LOD TIERS.
    // Re-running the rig per tier gives every tier a DIFFERENT skeleton (measured on the first
    // ladder: miku 34/44/60/42 joints, char1 57/20/58/207 across hero/high/medium/game). That is
    // wrong twice over — LOD tiers of one asset MUST share a skeleton or a clip authored for the
    // hero cannot play on the game tier, and re-rolling the rig per tier re-rolls its failure
    // modes too (char1's game tier landed on the runaway attractor, J=207 maxfan=128 score 0.000).
    // So rig ONCE, cache the skeleton + the skin field it is defined on, and transfer that same
    // field onto each tier's mesh. Skipping the rig also drops ~16-30s per tier.
    bool rig_from_cache = false;
    if (!rig_cache.empty()) {
        std::vector<uint8_t> bj, bp, bs, bv;
        if (ld_vec(rig_cache + "/rig_joints.bin",  bj) && ld_vec(rig_cache + "/rig_parents.bin", bp) &&
            ld_vec(rig_cache + "/rig_skin.bin",    bs) && ld_vec(rig_cache + "/rig_points.bin",  bv)) {
            R.joints.assign((float*)bj.data(), (float*)(bj.data()+bj.size()));
            R.parents.assign((int*)bp.data(),  (int*)(bp.data()+bp.size()));
            R.skin_pred.assign((float*)bs.data(), (float*)(bs.data()+bs.size()));
            P.vertices.assign((float*)bv.data(), (float*)(bv.data()+bv.size()));
            R.J = (int)(R.joints.size()/3);
            R.N = (int)(P.vertices.size()/3);
            if (R.J > 0 && R.N > 0 && R.parents.size() == (size_t)R.J &&
                R.skin_pred.size() == (size_t)R.N * R.J) {
                R.ok = rig_from_cache = true;
                printf("  [2/4] rig: loaded cached skeleton J=%d over N=%d skin points from %s"
                       " (shared across tiers; rig stage skipped)\n", R.J, R.N, rig_cache.c_str());
            } else {
                printf("  [2/4] rig cache in %s is malformed (J=%d N=%d) — regenerating\n",
                       rig_cache.c_str(), R.J, R.N);
                R = rig::RigResult{}; P = rig::PrepResult{};
            }
        }
    }

    // The delivery mesh in the rig frame. HOISTED above the draw loop because the pose gate must
    // judge the asset that would actually SHIP — the transferred skin on the textured mesh — not the
    // 8192 sample points the decoder saw (which have no faces, so no edges, so no stretch).
    // prep_mesh_for_rig_inmem normalized its own copy the same way, so joints/skin (sampled in
    // normalized [-1,1]) align with the geometry we write out.
    std::vector<float> verts_norm = bt.verts;
    rig::normalize_mesh(verts_norm);
    std::vector<float> dst_w;
    bool dst_w_valid = false;

    if (!rig_from_cache) {
        // BEST-OF-N OVER CONDITIONING DRAWS (--rig-retries, default 0 = exactly the shipped
        // one-shot path). The decode is deterministic given the conditioning cloud, so the ONLY way
        // to ask the decoder a second time is to draw a second cloud: the 8192 conditioning points
        // are one area-weighted sample of the surface, and re-drawing them is the same mesh asked
        // again, not a different asset.
        //
        // WHY BEST-OF-N AT ALL. Six draws of the SHIPPED --no-solid miku mesh, same weights, same
        // beam config, same decode seed, only the draw varying: J = 37/153/37/161/196/192, and FOUR
        // of the six welded every vertex to one joint (R3 ran to the token cap, no beam reached eos,
        // R4 never saw a skin token). A new subject is a coin flip, which is exactly what an API
        // cannot be. The rig stage is only ~20-27 s of a ~450 s pipeline, so a re-draw is cheap.
        //
        // THE PREDICATE, and why each term is load-bearing:
        //   skin_ok                         — a weightless rig is never a delivery. Non-negotiable.
        //   humanoid OR (generic AND pose)  — keying on the 22-bone gate ALONE would have discarded
        //                                     miku draw 2 (rig_score 0.915, the best rig measured in
        //                                     the whole campaign), which passed only the CREATURE
        //                                     gate, and it can never accept a creature like gilly at
        //                                     all. The pose gate is the deformation check that makes
        //                                     letting a non-humanoid through safe: across every rig
        //                                     in the campaign it read 2.2-4.1 on the good ones and
        //                                     9.2-46.7 on the bad ones, with no overlap.
        //   named_core >= N                 — EASY TO MISS AND IT MATTERS. The retargeter maps
        //                                     through NAMES (mret::derive_map_named) and its
        //                                     topology fallback fails outright on miku, whose legs
        //                                     parent to Spine and not to Hips. miku draw 2 scores
        //                                     0.915 but names only 14/22, while draw 0 scores 0.897
        //                                     and names 21/22 — the higher-scoring rig ANIMATES
        //                                     WORSE. Selecting on a score alone quietly degrades
        //                                     animation, so the floor is a real term, not a polish.
        //                                     19 is not a new constant: it is the lowest core count
        //                                     the existing humanoid gate is ever willing to accept
        //                                     (its narrow mirrored-right-arm repair form).
        double t_rig = pix::now_s();
        int rig_attempt = 0;
        i2r::DrawVerdict best_v;
        rig::RigResult  best_R;
        rig::PrepResult best_P;
        std::vector<float> best_w;
        bool have_best = false;
        for (;; ++rig_attempt) {
            const uint64_t draw_seed = rig_seed + (uint64_t)rig_attempt;
            P = rig::prep_mesh_for_rig_inmem(bt.verts, faces64, 8192, 512, draw_seed);
            if (!P.ok) { printf("FAIL: mesh prep failed\n"); return 1; }
            printf("  [2/4] rig prep: N=%d sampled, M=%d FPS queries%s\n", P.N, P.M,
                   rig_attempt ? "  (re-draw)" : "");

            // ---------- [3/4] auto-rig (R1->R3->R5->R4) ----------
            rig::RigOpts ro;
            ro.r1w = r1w; ro.qwen3_w = qwen3_w; ro.skin_vae_gguf = skinvae;
            ro.use_cuda = use_cuda; ro.num_beams = num_beams; ro.seed = rig_seed; ro.verbose = true;
            ro.do_sample = rig_sample;   // default false = deterministic fan-free beam (gilly audit)
            ro.structural_select = rig_structural_select;
            R = rig::run_rig_pipeline(P.vertices, P.normals, P.sampled_pc, P.sampled_feats, ro);
            if (!R.ok) { printf("FAIL: rig pipeline failed\n"); return 1; }

            i2r::DrawVerdict v;
            v.index    = rig_attempt;
            v.J        = R.J;
            v.skin_ok  = R.skin_ok;
            v.humanoid = R.humanoid_gate_ok;
            v.generic  = R.generic_gate_ok;
            v.naming_evaluated = bone_names;
            if (bone_names) {
                rig::NameOpts no;
                no.style = bone_names_smpl ? rig::NameStyle::SmplH : rig::NameStyle::Mixamo;
                no.facing_override = bone_facing;
                rig::BoneNaming BN = rig::name_bones(R.joints, R.parents, no);
                if (BN.ok) {
                    v.named_core = BN.named_core;
                    v.falsifier  = rig::falsify_bone_names(R.joints, R.parents, BN, false);
                }
            }
            // The pose gate. Run on the DELIVERY mesh, in the --generic-all-influential form: the
            // default single-joint pose passed all four assets in the campaign table including the
            // two bad ones, so it is not a sufficient gate. Skipped when the skin is zero (the
            // gate's own `moved` term would report 0.000 and we already know why).
            if (rig_pose_gate && R.skin_ok) {
                rig::transfer_skin(P.vertices, R.skin_pred, R.J, verts_norm, dst_w, 4);
                dst_w_valid = true;
                rigqc::SkinnedRig SR;
                if (rigqc::pose_gate_from_rig(verts_norm, faces64, R.joints, R.parents, dst_w,
                                              std::vector<std::string>{}, SR)) {
                    rigqc::PoseGateOpts po;
                    po.mode = rigqc::PoseGateOpts::GenericAllInfluential;
                    rigqc::PoseGateResult PG = rigqc::run_pose_gate(SR, po);
                    if (PG.ok) {
                        v.pose_ran   = true;
                        v.pose_pass  = PG.pass;
                        v.pose_worst = PG.worst();
                        v.pose_moved = PG.moved;
                        printf("  [3/4] %s\n", rigqc::pose_gate_line(PG).c_str());
                    } else {
                        printf("  [3/4] pose gate could not run: %s\n", PG.err.c_str());
                    }
                }
            }
            v.compute_accept(rig_min_named_core, rig_pose_gate_strict);
            printf("  [3/4] draw %d/%d: J=%d %s\n", rig_attempt + 1, rig_retries + 1, R.J,
                   v.describe().c_str());

            if (!have_best || v.better_than(best_v)) {
                best_v = v; best_R = R; best_P = P; have_best = true;
                best_w = (v.pose_ran && dst_w_valid) ? dst_w : std::vector<float>();
            }
            if (v.accept || rig_attempt >= rig_retries) break;
            printf("  [3/4] rig: re-drawing the conditioning cloud (attempt %d/%d)\n",
                   rig_attempt + 2, rig_retries + 1);
        }
        // Ship the BEST draw, not the last one. Before this the loop fell out holding whatever the
        // final attempt produced, so exhausting the retries actively degraded the result: draw 0
        // could be the only usable rig and draw N a 196-joint runaway, and the runaway shipped.
        R = best_R; P = best_P;
        if (!best_w.empty()) { dst_w = best_w; dst_w_valid = true; } else { dst_w.clear(); dst_w_valid = false; }
        i2r::g_rig_report = rig::StageReport{};
        i2r::g_rig_report.valid            = true;
        i2r::g_rig_report.draws            = rig_attempt + 1;
        i2r::g_rig_report.accepted         = best_v.accept;
        i2r::g_rig_report.accepted_draw    = best_v.accept ? best_v.index : -1;
        i2r::g_rig_report.skin_ok          = best_v.skin_ok;
        i2r::g_rig_report.humanoid_gate_ok = best_v.humanoid;
        i2r::g_rig_report.generic_gate_ok  = best_v.generic;
        i2r::g_rig_report.named_core       = best_v.named_core;
        i2r::g_rig_report.falsifier_fails  = best_v.falsifier;
        i2r::g_rig_report.pose_gate_ran    = best_v.pose_ran;
        i2r::g_rig_report.pose_gate_pass   = best_v.pose_pass;
        i2r::g_rig_report.pose_gate_worst  = best_v.pose_worst;
        i2r::g_rig_report.pose_gate_moved  = best_v.pose_moved;
        i2r::g_rig_report.J                = best_v.J;
        i2r::g_rig_report.summary          = best_v.describe();
        // FAIL CLOSED on a weightless rig. Measured 2026-08-14: five draws of the SHIPPED mesh gave
        // 3 runaways (J=153/161/196, maxfan up to 138), and every one of them delivered skin_pred all
        // zeros — an asset that deforms nothing — behind a single WARN. A warning is the wrong
        // response to that for two reasons:
        //   1. the GLB is publishable-looking, and both QC gates live in a wrapper script that an API
        //      caller need not run, so the corruption is silent at this boundary;
        //   2. worse, the block below banks R.skin_pred into --rig-cache, and every LOD tier then
        //      REUSES it — one bad draw poisons the whole ladder with zero weights.
        // There is no case where a rig that deforms nothing is a wanted delivery, so refusing is
        // strictly safer than emitting. --allow-zero-skin keeps the old behaviour for debugging.
        if (!R.skin_ok) {
            if (!allow_zero_skin) {
                printf("FAIL: rig produced ZERO skin weights (R3 ran away before the skin tokens; "
                       "J=%d). This asset would not deform, and --rig-cache would propagate it to "
                       "every LOD tier. Re-draw with --rig-retries N, or pass --allow-zero-skin to "
                       "write it anyway.\n", R.J);
                return 1;
            }
            printf("  [3/4] rig: WARNING — skin weights are ZERO (R3 ran away before the skin "
                   "tokens). This asset will not deform. Written only because --allow-zero-skin.\n");
        }
        printf("  [3/4] rig: J=%d joints  (%.1fs, %d draw%s%s)\n", R.J, pix::now_s() - t_rig,
               rig_attempt + 1, rig_attempt ? "s" : "",
               rig_attempt == 0 ? ""
                                : (best_v.accept ? (best_v.index ? ", accepted on a re-draw"
                                                                 : ", the FIRST draw still won")
                                                 : ", NO draw was accepted — shipping the best-ranked"));

        if (!rig_cache.empty()) {
            mkdir(rig_cache.c_str(), 0755);
            bool ok1 = sv_vec(rig_cache + "/rig_joints.bin",  R.joints.data(),    R.joints.size()*sizeof(float));
            bool ok2 = sv_vec(rig_cache + "/rig_parents.bin", R.parents.data(),   R.parents.size()*sizeof(int));
            bool ok3 = sv_vec(rig_cache + "/rig_skin.bin",    R.skin_pred.data(), R.skin_pred.size()*sizeof(float));
            bool ok4 = sv_vec(rig_cache + "/rig_points.bin",  P.vertices.data(),  P.vertices.size()*sizeof(float));
            if (ok1 && ok2 && ok3 && ok4)
                printf("  [3/4] rig cache written -> %s (reuse with --rig-cache for every other tier)\n",
                       rig_cache.c_str());
            else
                printf("  WARN: could not write the rig cache to %s\n", rig_cache.c_str());
        }
    }

    cancelhook::check();   // stage boundary
    // ---------- [4/4] combine: kNN skin-transfer onto the textured mesh + write ----------
    // verts_norm is hoisted above the rig block (the pose gate needs it). The transfer is skipped
    // only when the winning draw's transfer is already in hand from the gate — same call, same
    // inputs, so this is a cache and not a second code path.
    if (!dst_w_valid) rig::transfer_skin(P.vertices, R.skin_pred, R.J, verts_norm, dst_w, 4);

    // ---------- name the bones to a standard humanoid convention (automatic) ----------
    // SkinTokens emits anonymous bone_0..bone_N, which no Mixamo/AMASS clip can retarget
    // onto. Derive standard names from the skeleton's structure + rest geometry, then run
    // the falsifier so a wrong answer is LOUD instead of silent. Naming never blocks the
    // write: a rig we cannot name still ships with bone_N names, just un-retargetable.
    std::vector<std::string> jnames;
    {
        rig::NameOpts no;
        no.style = bone_names_smpl ? rig::NameStyle::SmplH : rig::NameStyle::Mixamo;
        no.facing_override = bone_facing;
        if (!bone_names) {
            printf("  [4/4] bone naming: DISABLED (--no-bone-names) -> anonymous bone_N\n");
        } else {
            rig::BoneNaming BN = rig::name_bones(R.joints, R.parents, no);
            if (!BN.ok) {
                printf("  [4/4] bone naming: FAILED (%s) -> falling back to anonymous bone_N\n",
                       BN.fail_reason.c_str());
            } else {
                int fails = rig::falsify_bone_names(R.joints, R.parents, BN, true);
                printf("  [4/4] bone naming: %s core=%d/22 fingers=%d extra=%d facing=%+dZ\n",
                       fails ? "FALSIFIED (names kept; TRUST THEM AT YOUR OWN RISK)" : "clean",
                       BN.named_core, BN.named_fingers, BN.n_extra, BN.facing);
                jnames = BN.names;
            }
        }
    }

    // baseColor RGBA atlas -> PNG (the only texture the rigged-textured writer carries, matching combine).
    std::vector<uint8_t> base_png = glb::encode_png(bt.base_color.data(), bt.tw, bt.th, 4);
    // The tangent-space normal map baked above is what makes a decimated tier still look dense;
    // carry it into the rigged delivery (the packed --no-rig writer already did).
    std::vector<uint8_t> nrm_png;
    if (!nmap.empty()) nrm_png = glb::encode_png(nmap.data(), bt.tw, bt.th, 3);
    bool have_nrm = bt.normals.size() == bt.verts.size();
    bool ok = glb::write_rigged_textured_glb(out.c_str(), verts_norm, faces64, bt.uvs,
                                             R.joints, R.parents, dst_w,
                                             have_nrm ? &bt.normals : nullptr,
                                             base_png.data(), base_png.size(), "image/png",
                                             jnames.empty() ? nullptr : &jnames,
                                             nrm_png.empty() ? nullptr : nrm_png.data(),
                                             nrm_png.size());
    if (!ok) { printf("FAIL: write_rigged_textured_glb\n"); return 1; }
    printf("==== DONE -> %s  (verts=%zu faces=%zu J=%d, %.1fs total) ====\n",
           out.c_str(), verts_norm.size()/3, faces64.size()/3, R.J, pix::now_s() - t_geo);
    return 0;
}


// ---------------------------------------------------------------------------
// The stage's public entry point: the pipeline above, plus the cancellation boundary.
//
// cancelhook::check() THROWS from deep inside the DiT sampler / the DC ladder / the beam decode
// (see cancel_hook.hpp for why a return code could not work there). The throw unwinds through
// M1Harness, whose destructor frees the CUDA weight buffers and the backend — which is what makes a
// cancel actually give the card back rather than merely stop using it. It is caught HERE, once, at
// the stage boundary, so no half-run pipeline state escapes.
//
// Nothing has been written to `out` at any point a check() can fire: the GLB writers are the last
// thing the pipeline does and there is no cancellation point inside them.
// ---------------------------------------------------------------------------
int i2r::run(i2r::Options opt) {
    try {
        return i2r_run_body(std::move(opt));
    } catch (const cancelhook::Cancelled&) {
        printf("==== image_to_rig CANCELLED (no output written; GPU buffers released on unwind) ====\n");
        fflush(stdout);
        return i2r::RC_CANCELLED;
    } catch (const std::exception& e) {
        // THE ONE DELIBERATE DEVIATION FROM THE OLD CLI BEHAVIOUR, and it is why it is here.
        //
        // Nothing used to catch here, so a throw from inside the chain reached the top and
        // std::terminate() ABORTED THE PROCESS. For a program that is fine — the shell driver sees a
        // non-zero exit either way. For a LIBRARY inside a long-lived service it is fatal: one
        // request takes the whole service down with it, and every queued request dies too.
        //
        // MEASURED, not hypothetical: with `llama-server` (a co-tenant that does NOT take the 3060
        // flock) holding 8976 MiB, the geometry stage threw `gallocr_alloc_graph failed` out of a
        // 904 MiB cudaMalloc and the process core-dumped. An engine that shares a card WILL meet
        // this. Now it is an error return, the stack unwinds through M1Harness (so the partial
        // allocations are freed), and the service survives to answer the next request.
        //
        // Revert = delete this clause. The exit code changes from 134 (SIGABRT) to 1; both are
        // non-zero, and the ggml diagnostic that precedes it is unchanged, so the A/B drivers'
        // `EXIT=` checks and log greps read the same either way.
        printf("FAIL: image_to_rig stage threw: %s\n", e.what());
        fflush(stdout);
        return i2r::RC_FAIL;
    }
}

// ---------------------------------------------------------------------------
// The program. Parse argv into the struct, then run the stage — so the CLI and the library differ
// in nothing but how the struct got filled.
// ---------------------------------------------------------------------------
#ifdef IMAGE_TO_RIG_LIB
int image_to_rig_main(int argc, char** argv) {
#else
int main(int argc, char** argv) {
#endif
    // A production run can spend many minutes in M4/UltraShape.  When stdout is
    // redirected to the per-candidate run log, libc otherwise fully buffers
    // stage lines and leaves the operator unable to tell a live decode from a
    // dead process.  Preserve every existing message, but make newline-delimited
    // progress authoritative in the log.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    i2r::Options opt;
    const int rc = i2r::parse_args(argc, argv, opt);
    if (rc >= 0) return rc;
    return i2r::run(std::move(opt));
}