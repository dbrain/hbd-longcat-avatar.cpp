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
#include "pixal3d_chain.hpp"
#include "image_io.hpp"
#include "tex_atlas.hpp"
#include "tex_project.hpp"             // --tex-project: project the real images into the UV atlas
#include "glb_textured.hpp"            // encode_png
#include "glb_rigged_textured.hpp"     // write_rigged_textured_glb
#include "rig_transfer.hpp"            // transfer_skin
#include "rig_pipeline.hpp"            // run_rig_pipeline
#include "rig_bone_names.hpp"          // name_bones + falsify_bone_names (standard bone naming)
#include "mesh_sample.hpp"             // prep_mesh_for_rig_inmem, normalize_mesh
#include "p3sam_segment.hpp"           // P3-SAM part segmentation (native, --part-retopo)
#include "per_part_decimate.hpp"       // region-adaptive per-part decimation (hands kept dense)
#include "moge_cam.hpp"                // native MoGe camera (FOV) estimation (--moge)
#include "ultrashape_refine.hpp"       // native UltraShape refine (clean/watertight densify, --refine)
#include "quad_retopo.hpp"             // rung 2: quadwild-bimdf quad retopology (--quad, shell-out)
#include "im_retopo.hpp"               // rung 2 (default): Instant Meshes retopo (clean organic flow)
#include "normal_bake.hpp"             // tangent-space normal map: dense high-poly detail -> retopo low-poly
#ifdef PIXAL3D_PACK
#include "glb_packed.hpp"              // packed writer (KTX2+meshopt) — the only textured writer that carries a normal map
#endif
#include <thread>
#include <algorithm>
#include "glb_writer.hpp"              // glb::write_glb (intermediate stage artifacts)
#include "glb_reader.hpp"              // glb::read_glb (--from-refined resume)
#include "../../sparse_spike/npy.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <sys/stat.h>

static const float DEF_CAM = 0.7332379387484828f, DEF_DIST = 1.3021559715270996f, DEF_MS = 1.0f;

static void usage() {
    printf("usage: image_to_rig --model <geo_gguf_dir> --image <matte.png> --out <out.glb>\n"
           "        [--fov <deg>] [--cam <ang_rad> <dist> <scale>] [--texsize N] [--decimate F]\n"
           "        [--resolution N] [--seed N] [--fast] [--cpu]\n"
           "        [--r1w <dir>] [--qwen3 <dir>] [--skin-vae <dir>] [--beams N] [--rig-seed N]\n"
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
           "        [--us-octree N] [--us-latents N] [--us-steps N] [--us-guidance F] [--us-chunk N]\n"
           "        [--us-gguf <dir>] [--us-dit-w <dir>] [--us-vae-w <dir>] [--us-cnd-w <dir>] [--us-meta <npy>]\n"
           "        [--stage-dir <dir>]   (emit coarse/refined/decimated GLB intermediates + resume caches)\n"
           "        [--from-refined <dir>] (resume: skip geometry+refine, load refined.glb + PBR cache)\n"
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

int main(int argc, char** argv) {
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
    //  - USR_GEO_FLASH=1: flash the GEOMETRY DiTs' self-attn (M3b slat + tex-proj + tex-cross), same deal
    //    as USR_DIT_FLASH but for the pixal3d geo stages. M3b 280->192s, tex 160->111s (-138s, 31%), peak
    //    4.3GB. Effectively lossless: coarse.glb A/B adds only ~0.016% chamfer OVER the geo DiTs' own
    //    ~0.08% run-to-run nondeterminism floor (f16 Q/K/V, GGML_PREC_F32 accum; lin/FFN/MoE stay fp32).
    //    `USR_GEO_FLASH=0` restores the fp32 dense-tiled path.
    setenv("USR_GEO_FLASH",    "1",    0);
    std::string model, image, out;
    std::string r1w     = "/home/dbrain/models/3d/rig/r1w_real";
    std::string qwen3_w = "/home/dbrain/models/3d/rig/qwen3_w";
    std::string skinvae = "/home/dbrain/models/3d/rig/skin_vae_gguf";
    float cam = DEF_CAM, dist = DEF_DIST, ms = DEF_MS;
    bool use_cuda = true, fast = false;
    // Inline-API rig default = DETERMINISTIC beam (do_sample=false) + beams=20 — the fan-free recipe
    // (gilly audit). On the canonical standing-Miku input the stochastic scaffold default (sample,
    // beams=10) hit the runaway attractor (J=178 maxfan=103 rig_score 0.000); deterministic = J=56
    // maxfan=5 rig_score 0.908. A one-shot API must be robust out of the box. `--rig-sample` restores
    // the stochastic scaffold recipe (do_sample=true, beams=10) for owner A/B comparison.
    bool rig_sample = false;
    // --clean (default ON): MC manifold remesh (de-spike) + precluster atlas bake. The raw dual-grid
    // O-Voxel mesh is non-manifold (spikey render + a shattered >400s xatlas atlas); the manifold
    // remesh fixes both and the precluster bake drops to ~18s. `--no-clean` restores the raw path for
    // A/B comparison.
    bool clean = true;
    // MC-remesh recipe (validated on standing-Miku): stride 2 (grid 512) + blur 0 + 1 Taubin iter =
    // de-spiked manifold that KEEPS thin features (twintails). blur>0 box-bridges thin gaps (fuses the
    // twintails into one cape); the finer stride-1 grid preserves them too but is ~12x slower for the
    // same result, so blur=0 at stride 2 is the default. --mc-blur/--mc-stride/--mc-smooth override.
    int mc_stride = 2, mc_blur = 0, mc_smooth = 1;
    int texsize = 1024, decimate = 150000, num_beams = 20;
    // --dump-geo/--from-geo: cache the geometry volume (coords1024 + per-voxel PBR) so the cheap
    // post-geometry stages (MC-remesh + bake + rig, ~3min) can be iterated without re-paying the ~420s
    // diffusion. --no-rig writes a textured-only GLB (skips the 159s auto-rig) for fast mesh/texture A/B.
    std::string dump_geo, from_geo;
    bool no_rig = false;
    uint64_t rig_seed = 0;
    // Standard bone naming (ON by default -- an anonymous bone_N rig cannot take a Mixamo or
    // AMASS clip without a human hand-mapping it, which is the whole point of rigging).
    //   --no-bone-names   keep anonymous bone_N
    //   --bone-names smpl|mixamo   emit SMPL-H 22 names instead of Mixamo (default mixamo)
    //   --bone-facing +z|-z        override the auto-derived facing (which decides LEFT/RIGHT)
    bool bone_names = true, bone_names_smpl = false;
    int  bone_facing = 0;   // 0 = auto-derive from the feet
    // --part-retopo: native P3-SAM part segmentation -> region-adaptive per-part decimation
    // (hands/fingers kept dense, hair/torso crushed) BEFORE texture bake, replacing the flat decimate.
    // CPU correctness-first port (validated cos 1.0 backbone/heads); GPU port = perf follow-up, so it
    // is OFF by default and currently slow. See p3sam_retopo.cpp for the standalone path.
    bool part_retopo = false;
    std::string p3sam_w = "/mnt/hdd/3d/avatar-shootout/p3sam_goldens/weights_npy";
    std::string obj_decimate = "./obj_decimate";
    // --moge: native MoGe-2 camera estimation (image -> FOV) replaces the default/--fov camera, making
    // the inline API 100% camera-native. Square-matte path (rows=cols=60, AR=1). See moge_cam.hpp.
    bool use_moge = false;
    std::string moge_w = "/mnt/hdd/3d/avatar-shootout/moge_goldens/weights_npy";
    // --refine (default ON): native UltraShape refine between pixal3d geometry and the bake/rig — the
    // clean/watertight ~7.5x densify. `--no-refine` restores the coarse-only path (fast A/B). Config +
    // weight dirs via --us-*. The refined mesh is re-canonicalized onto the coarse-mesh bbox so the PBR
    // volume bakes onto it (RP_CANON_TO_DENSE). See ultrashape_refine.hpp.
    bool refine = true;
    usr::RefineCfg uscfg;
    // Reproject-bake mode (only when refine ran → a coarse shell exists). Default = mesh-attr (barycentric
    // interp of the shell's per-vertex PBR): best texel coverage (~71% direct vs ~39% for snap+volume on
    // toy2) → clean uniform texture after inpaint. --tex-snap-volume forces the snap-onto-shell +
    // volume-grid_sample path, which is steadier on FINE HAIR/STRANDS (no triangle-choice speckle) but
    // leaves more holes on smooth bodies. See tex_atlas.hpp:969-973.
    bool tex_snap_volume = false;
    // --tex-volume-direct: THE FRAY FIX (default OFF — owner A/Bs). Both reproject modes above find the
    // colour by composing TWO closest-point projections (texel -> coarse shell -> volume), which slides
    // along the surface and fetches the wrong material at a colour boundary — 9.2% of texels read a voxel
    // >4 vox away on the model (p99 13.2 vox). Direct mode reads the volume at the TEXEL'S OWN position.
    // The BLACK-TEXTURE bug that motivated the snap is really --tex-fallback-r being 0 here (see below):
    // direct sampling is 47.6% wrong at r=0 and 1.34% wrong at r=8, vs RP_ATTR's 4.87%. Measured on
    // inline_soldier1536; see tex_atlas.hpp's RP_DIRECT comment.
    bool tex_volume_direct = false;
    // --tex-fallback-r: `sample_fallback_r` for texatlas::bake — the radius (in voxels) of the
    // nearest-occupied-voxel search that texgs::sample_one falls back to when all 8 trilinear corners are
    // empty. -1 = mode default: 0 for the legacy snap modes (TODAY'S BEHAVIOUR, preserved), 8 for
    // --tex-volume-direct (below r=8 the direct read is starved: r=1 -> 40.4% wrong, r=4 -> 11.9%,
    // r=8 -> 1.3%). NB the legacy driver tex_reproject.cpp:172 has always defaulted this to 16; only
    // this production call site passes 0, which is why the direct path "baked black" here.
    int tex_fallback_r = -1;
    // --tex-project: after the bake, OVERWRITE the atlas baseColor by projecting the real images onto the
    // mesh (front = --image, the matte the mesh was built from = the exact camera frame; back = optional
    // --tex-back, a generated 180deg view). The volume bake's colours are right but carry ZERO detail (the
    // PBR is generated in a ~64^3 latent → band-limited by design); the projection keeps near-source
    // detail. metallicRoughness stays from the volume bake. See tex_project.hpp.
    // Views are generalized to an arbitrary yaw about +Y: --tex-back is sugar for --tex-view 180, and
    // --tex-view is repeatable (the owner called out SIDE views as a problem area, so adding e.g.
    // `--tex-view 90 right.png --tex-view -90 left.png` needs no code change here).
    bool tex_project = false;
    bool tex_project_overlay = false;
    std::string tex_back;
    // --tex-front: use a DIFFERENT image than --image as the front projection source. Geometry and
    // texturing already consume different tensors (geometry gets the resized + ImageNet-normalized
    // img512_raw/img1024_raw; projection gets pcfg.front_img -> stbi_load at native res); they share
    // only this path variable. Splitting it lets the front be de-lit (sd-delight) while geometry keeps
    // the LIT original it was validated on -- i.e. de-lighting cannot regress TRELLIS by construction.
    // MUST be the same camera frame as --image (same crop/scale), so in practice: a processed --image.
    std::string tex_front;
    std::vector<texproj::ViewSpec> tex_views;
    // --quad: rung-2 quad retopology (quadwild-bimdf) on the refined mesh -> clean field-aligned topology,
    // TRIANGULATED for the tri-only bake/rig/glb downstream. Shell-out (no linking). OFF by default while
    // bringing up; --quadwild-repo overrides the built-repo path. See quad_retopo.hpp.
    bool do_quad = false;
    qr::QuadCfg qcfg;
    // --retopo <im|quadwild>: which retopology tool the --quad stage uses. Default IM (Instant Meshes) —
    // clean organic quad flow + watertight; quadwild tears organic flat regions (validated on a robot).
    std::string retopo_tool = "im";
    imretopo::ImCfg imcfg;
    // Staging: --stage-dir writes every intermediate GLB (coarse/refined/decimated) as a named artifact
    // for the eye-test page AND the .bin caches to resume. --from-refined <dir> skips geometry+refine
    // (loads refined.glb + the cached PBR volume). --from-geo (below) skips only the geometry diffusion.
    std::string stage_dir, from_refined;
    pix::ChainInput in;

    auto nextf = [&](int& i){ return (float)std::atof(argv[++i]); };
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--model" && i+1 < argc) model = argv[++i];
        else if (a == "--image" && i+1 < argc) image = argv[++i];
        else if (a == "--out"   && i+1 < argc) out = argv[++i];
        // cam and dist are THE SAME PARAMETER: DEF_DIST == 0.5/tan(DEF_CAM/2) to 7 d.p. (1.3021560).
        // DEF_DIST was never a measured constant -- it is the canonical "unit-diameter object exactly
        // fills the frame" render convention. Setting cam WITHOUT dist breaks the identity and turns a
        // perspective change into a ZOOM: measured on the soldier, --moge's 46.5deg made the model
        // +16% BIGGER (extents x1.164/1.159/1.146, shape unchanged to ~1%) and pushed 23,336 verts
        // through the canonical box floor at Y <= -0.4999 -- a flat plane where the boots should be.
        // That is what OOM'd the refine (bigger subject -> N1 1227->1701 -> M 12541->18674 -> M^2).
        else if (a == "--fov"   && i+1 < argc) { cam = std::atof(argv[++i]) * (float)M_PI / 180.0f; dist = 0.5f / std::tan(cam * 0.5f); }
        else if (a == "--cam"   && i+3 < argc) { cam = std::atof(argv[++i]); dist = std::atof(argv[++i]); ms = std::atof(argv[++i]); }
        else if (a == "--texsize" && i+1 < argc) texsize = std::atoi(argv[++i]);
        else if (a == "--decimate" && i+1 < argc) decimate = std::atoi(argv[++i]);
        else if ((a == "--resolution" || a == "--res") && i+1 < argc) {
            in.resolution = std::atoi(argv[++i]);
            if (in.resolution % 16 != 0) { printf("--resolution must be a multiple of 16\n"); return 1; }
        }
        else if (a == "--seed" && i+1 < argc) in.seed = std::atoi(argv[++i]);
        else if (a == "--fast") fast = true;
        else if (a == "--cpu")  use_cuda = false;
        else if (a == "--r1w"   && i+1 < argc) r1w = argv[++i];
        else if (a == "--qwen3" && i+1 < argc) qwen3_w = argv[++i];
        else if (a == "--skin-vae" && i+1 < argc) skinvae = argv[++i];
        else if (a == "--beams" && i+1 < argc) num_beams = std::atoi(argv[++i]);
        else if (a == "--rig-sample") { rig_sample = true; if (num_beams == 20) num_beams = 10; }
        else if (a == "--rig-seed" && i+1 < argc) rig_seed = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--no-bone-names") bone_names = false;
        else if (a == "--bone-names" && i+1 < argc) { std::string v = argv[++i]; bone_names_smpl = (v == "smpl"); }
        else if (a == "--bone-facing" && i+1 < argc) { bone_facing = (argv[++i][0] == '-') ? -1 : +1; }
        else if (a == "--part-retopo") part_retopo = true;
        else if (a == "--p3sam-weights" && i+1 < argc) p3sam_w = argv[++i];
        else if (a == "--obj-decimate" && i+1 < argc) obj_decimate = argv[++i];
        else if (a == "--moge") use_moge = true;
        else if (a == "--moge-weights" && i+1 < argc) moge_w = argv[++i];
        // ---- UltraShape refine stage ----
        else if (a == "--refine") refine = true;
        else if (a == "--no-refine") refine = false;
        else if (a == "--us-octree" && i+1 < argc) uscfg.octree = std::atoi(argv[++i]);
        else if (a == "--us-latents" && i+1 < argc) uscfg.num_latents = std::atoi(argv[++i]);
        else if (a == "--us-steps" && i+1 < argc) uscfg.steps = std::atoi(argv[++i]);
        else if (a == "--us-guidance" && i+1 < argc) uscfg.guidance = (float)std::atof(argv[++i]);
        else if (a == "--us-chunk" && i+1 < argc) uscfg.chunk = std::atoll(argv[++i]);
        else if (a == "--us-gguf" && i+1 < argc) uscfg.gguf_dir = argv[++i];
        else if (a == "--us-dit-w" && i+1 < argc) uscfg.dit_w = argv[++i];
        else if (a == "--us-vae-w" && i+1 < argc) uscfg.vae_w = argv[++i];
        else if (a == "--us-cnd-w" && i+1 < argc) uscfg.cnd_w = argv[++i];
        else if (a == "--us-meta" && i+1 < argc) uscfg.meta = argv[++i];
        else if (a == "--stage-dir" && i+1 < argc) stage_dir = argv[++i];
        else if (a == "--from-refined" && i+1 < argc) from_refined = argv[++i];
        else if (a == "--tex-snap-volume") tex_snap_volume = true;
        else if (a == "--tex-volume-direct") tex_volume_direct = true;
        else if (a == "--tex-fallback-r" && i+1 < argc) tex_fallback_r = std::atoi(argv[++i]);
        else if (a == "--tex-project") tex_project = true;
        else if (a == "--tex-project-overlay") { tex_project = true; tex_project_overlay = true; }
        else if (a == "--tex-front" && i+1 < argc) tex_front = argv[++i];
        else if (a == "--tex-back" && i+1 < argc) tex_back = argv[++i];
        else if (a == "--tex-view" && i+2 < argc) {
            texproj::ViewSpec vs;
            vs.yaw_deg = (float)std::atof(argv[++i]);
            vs.img = argv[++i];
            // `--tex-view 180 x.png` and `--tex-back x.png` are the same view; route both through tex_back
            // so exactly one yaw=180 camera exists and Stats' back_* fields stay meaningful.
            if (std::fmod(vs.yaw_deg, 360.f) == 180.f && tex_back.empty()) tex_back = vs.img;
            else if (std::fmod(vs.yaw_deg, 360.f) == 0.f)
                printf("NOTE: --tex-view %g is the FRONT view (--image); ignoring the duplicate\n", vs.yaw_deg);
            else tex_views.push_back(vs);
        }
        // --tex-dit proj|cross: WHICH generative tex DiT paints the PBR volume. `cross` (DEFAULT =
        // unchanged behaviour) = trellis2_tex_1024, the TRELLIS-2 texturing model the tex_goldens came
        // from. `proj` = slat_flow_imgshape2tex_1024 = the model pixal3d's OWN Python pipeline runs
        // (Trellis2TexturingPipeline is dead code there), i.e. the one that painted gilly. Same cond the
        // geometry stage already computes, same tex_slat normalization, same tex decoder, same bake — so
        // this is a clean single-variable A/B. --tex-dit-w overrides the weight dir/basename.
        else if (a == "--tex-dit" && i+1 < argc) {
            std::string m = argv[++i];
            if (m == "proj") in.tex_proj = true;
            else if (m == "cross") in.tex_proj = false;
            else { printf("--tex-dit must be 'proj' or 'cross' (got '%s')\n", m.c_str()); return 1; }
        }
        else if (a == "--tex-dit-w" && i+1 < argc) in.tex_flow_w = argv[++i];
        else if (a == "--quad") do_quad = true;
        else if (a == "--no-quad") do_quad = false;
        else if (a == "--retopo" && i+1 < argc) retopo_tool = argv[++i];   // im (default) | quadwild
        else if (a == "--im-adaptivity" && i+1 < argc) imcfg.adaptivity = (float)std::atof(argv[++i]);
        else if (a == "--im-verts" && i+1 < argc) imcfg.target_verts = std::atoi(argv[++i]);
        else if (a == "--quadwild-repo" && i+1 < argc) qcfg.repo = argv[++i];
        else if (a == "--no-clean") clean = false;
        else if (a == "--mc-stride" && i+1 < argc) mc_stride = std::atoi(argv[++i]);
        else if (a == "--mc-blur" && i+1 < argc) mc_blur = std::atoi(argv[++i]);
        else if (a == "--mc-smooth" && i+1 < argc) mc_smooth = std::atoi(argv[++i]);
        else if (a == "--dump-geo" && i+1 < argc) dump_geo = argv[++i];
        else if (a == "--from-geo" && i+1 < argc) from_geo = argv[++i];
        else if (a == "--no-rig") no_rig = true;
        // geometry sampler knobs (forwarded to ChainInput; defaults already = inference.py)
        else if (a == "--guidance" && i+1 < argc) { float g=nextf(i); in.ss.guidance=g; in.shape.guidance=g; }
        else if (a == "--steps" && i+1 < argc) { int s=std::atoi(argv[++i]); in.ss.steps=in.shape.steps=in.tex.steps=s; }
        else { printf("unknown/incomplete arg: %s\n", a.c_str()); usage(); return 1; }
    }
    if (model.empty() || image.empty() || out.empty()) { usage(); return 1; }
    if (fast) { setenv("PIXAL3D_FAST", "1", 1); setenv("GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F", "1", 1); }

    printf("==== image_to_rig (inline native image -> textured+rigged GLB) ====\n");
    printf("  model=%s\n  image=%s\n  out=%s\n  cam: fov=%.4frad dist=%.4f scale=%.2f  backend=%s\n",
           model.c_str(), image.c_str(), out.c_str(), cam, dist, ms, use_cuda ? "cuda" : "cpu");
    printf("  tex-dit=%s%s%s\n", in.tex_proj ? "proj (slat_flow_imgshape2tex_1024)" : "cross (trellis2_tex_1024)",
           in.tex_flow_w.empty() ? "" : " w=", in.tex_flow_w.c_str());

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

    // ---------- [1/4] geometry + texture (in-process, GPU) ----------
    setenv("PIXAL3D_GGUF_DIR", model.c_str(), 1);
    try {
        in.img512_raw  = imgio::load_chw(image, 512);
        in.img1024_raw = imgio::load_chw(image, 1024);
    } catch (const std::exception& e) { printf("image load failed: %s\n", e.what()); return 1; }
    in.cam = cam; in.dist = dist; in.ms = ms; in.use_cuda = use_cuda; in.verbose = true;
    in.textured = true; in.watertight = true; in.remesh = false;
    in.mc_remesh = clean; in.mc_stride = mc_stride; in.mc_blur = mc_blur; in.mc_post_smooth = mc_smooth;
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
        mesh = pix::build_mc_remesh(coords1024, in.resolution, mc_stride, mc_blur, mc_smooth, true);
    } else {
        mesh = pix::run_geometry(in, &st, nullptr, &pbr_feats, &pbr_coords, dump_geo.empty() ? nullptr : &coords1024);
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

    // stage-dir: snapshot the coarse (pre-refine) mesh as a named eye-test artifact.
    if (!stage_dir.empty() && !did_refine_load) {
        mkdir(stage_dir.c_str(), 0755);
        if (glb::write_glb((stage_dir + "/coarse.glb").c_str(), mesh.verts, mesh.faces))
            printf("  [stage] wrote %s/coarse.glb (%d v / %d f)\n", stage_dir.c_str(), mesh.N, mesh.F);
    }

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
        } catch (const std::exception& e) { printf("FAIL: UltraShape refine: %s\n", e.what()); return 1; }
        // re-canonicalize the ±1 UltraShape mesh onto the coarse-mesh bbox so the PBR volume bakes onto it.
        bbox_canon_onto(refined.verts, coarse_ref);
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

    // Dense high-poly kept for the normal-map bake (retopo/decimation drops fine relief — anime faces,
    // fingers — which the retopo mesh then re-acquires as a tangent-space normal map baked from here).
    std::vector<float>   nrm_src_verts;
    std::vector<int64_t> nrm_src_faces;

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
    const bool precluster = clean;
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
        // default reproject mode = mesh-attr (barycentric shell colours); --tex-snap-volume opts out.
        if (!tex_snap_volume) setenv("RP_ATTR", "1", 1); else unsetenv("RP_ATTR");
        // --tex-volume-direct: read the volume at each texel's own position; the mode above becomes the
        // guard for texels the volume has no data near. Default OFF = today's behaviour.
        if (tex_volume_direct) setenv("RP_DIRECT", "1", 1); else unsetenv("RP_DIRECT");
    }
    // sample_fallback_r: 0 keeps today's snap-mode behaviour bit-for-bit; volume-direct needs a real
    // radius or every texel starves to black (that starvation IS the "black texture bug").
    const int bake_fbr = tex_fallback_r >= 0 ? tex_fallback_r : (tex_volume_direct ? 8 : 0);
    texatlas::BakedTexture bt = use_reproject
        ? texatlas::bake(mesh.verts, mesh.faces, pbr_feats, pbr_coords, in.resolution, texsize, decimate,
                         4, true, bake_fbr, precluster, 55.f, &shell_verts, &shell_faces, &shell_attr, /*reproject=*/true)
        : texatlas::bake(mesh.verts, mesh.faces, pbr_feats, pbr_coords, in.resolution, texsize, decimate,
                         4, true, bake_fbr, precluster);
    printf("  [1/4] tex bake%s: %dx%d, %d charts, %d verts / %zu faces  (%.1fs)\n",
           use_reproject ? " (reproject/shell)" : "", bt.tw, bt.th, bt.chart_count,
           (int)bt.verts.size()/3, bt.faces.size()/3, pix::now_s()-t_bake);

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

    // ---------- [2/4] rig cond points (native, in RAM) ----------
    std::vector<int64_t> faces64(bt.faces.begin(), bt.faces.end());
    rig::PrepResult P = rig::prep_mesh_for_rig_inmem(bt.verts, faces64, 8192, 512, rig_seed);
    if (!P.ok) { printf("FAIL: mesh prep failed\n"); return 1; }
    printf("  [2/4] rig prep: N=%d sampled, M=%d FPS queries\n", P.N, P.M);

    // ---------- [3/4] auto-rig (R1->R3->R5->R4) ----------
    rig::RigOpts ro;
    ro.r1w = r1w; ro.qwen3_w = qwen3_w; ro.skin_vae_gguf = skinvae;
    ro.use_cuda = use_cuda; ro.num_beams = num_beams; ro.seed = rig_seed; ro.verbose = true;
    ro.do_sample = rig_sample;   // default false = deterministic fan-free beam (gilly audit)
    double t_rig = pix::now_s();
    rig::RigResult R = rig::run_rig_pipeline(P.vertices, P.normals, P.sampled_pc, P.sampled_feats, ro);
    if (!R.ok) { printf("FAIL: rig pipeline failed\n"); return 1; }
    printf("  [3/4] rig: J=%d joints  (%.1fs)\n", R.J, pix::now_s() - t_rig);

    // ---------- [4/4] combine: kNN skin-transfer onto the textured mesh + write ----------
    // Normalize the textured mesh into the rig frame (prep_mesh_for_rig_inmem normalized its copy the
    // same way), so joints/skin (sampled in normalized [-1,1]) align with the geometry we write out.
    std::vector<float> verts_norm = bt.verts;
    rig::normalize_mesh(verts_norm);
    std::vector<float> dst_w;
    rig::transfer_skin(P.vertices, R.skin_pred, R.J, verts_norm, dst_w, 4);

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
    bool have_nrm = bt.normals.size() == bt.verts.size();
    bool ok = glb::write_rigged_textured_glb(out.c_str(), verts_norm, faces64, bt.uvs,
                                             R.joints, R.parents, dst_w,
                                             have_nrm ? &bt.normals : nullptr,
                                             base_png.data(), base_png.size(), "image/png",
                                             jnames.empty() ? nullptr : &jnames);
    if (!ok) { printf("FAIL: write_rigged_textured_glb\n"); return 1; }
    printf("==== DONE -> %s  (verts=%zu faces=%zu J=%d, %.1fs total) ====\n",
           out.c_str(), verts_norm.size()/3, faces64.size()/3, R.J, pix::now_s() - t_geo);
    return 0;
}
