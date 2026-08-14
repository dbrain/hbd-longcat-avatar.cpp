// image_to_rig_options.hpp — image_to_rig's ~90 flags as a STRUCT, so the stage can be called with
// a typed request instead of a synthesised argv.
//
// WHY. image_to_rig.cpp is both the A/B harness binary and (compiled -DIMAGE_TO_RIG_LIB) the rig
// stage of avatar::Engine. The library used to reach it by marshalling its typed RigRequest into an
// argv array and letting the CLI parser re-parse it: every option stringified and re-scanned on
// every request, and any option without a flag simply unreachable. Now there is one Options struct;
// main() parses argv INTO it and the library FILLS IT DIRECTLY. There is still exactly one parser
// and one implementation, so the CLI cannot drift from the library.
//
// THIS HEADER IS DELIBERATELY LIGHT: <string>, <vector>, <cstdint> and nothing else. It is included
// by avatar_pipeline.cpp, which is compiled against sd.cpp's ggml (GGML_MAX_NAME=160). Pulling
// pixal3d_chain.hpp / ultrashape_refine.hpp in here would drag the OTHER ggml's headers into that
// translation unit, and sizeof(ggml_tensor) differs between the two — a silent ABI corruption, not
// a compile error. That is why the handful of options that belong to those config structs are
// MIRRORED here as Opt<T> instead of embedding the structs themselves.
//
// Opt<T> means "the flag was not given, keep whatever that config struct's own default is". Copying
// the defaults into this header instead would create two sources of truth for e.g. the refine
// octree size, and they would drift.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace i2r {

// The default geometry camera (the miku cam). cam and dist are ONE parameter:
// DEF_DIST == 0.5/tan(DEF_CAM/2). Setting one without the other turns a perspective change into a
// zoom — see the --fov comment in image_to_rig.cpp.
inline constexpr float DEF_CAM  = 0.7332379387484828f;
inline constexpr float DEF_DIST = 1.3021559715270996f;
inline constexpr float DEF_MS   = 1.0f;

// "unset" without inventing a sentinel value that could also be a legitimate setting.
template <class T>
struct Opt {
    T    v{};
    bool set = false;
    Opt& operator=(const T& x) { v = x; set = true; return *this; }
    explicit operator bool() const { return set; }
};

struct TexView {           // mirrors texproj::ViewSpec
    float       yaw_deg = 0.f;
    std::string img;
};

// Exit codes image_to_rig's stage can return, beyond 0 = ok / 1 = failed.
enum : int { RC_OK = 0, RC_FAIL = 1, RC_CANCELLED = 130 };

struct Options {
    // ---- required ----
    std::string model;                 // --model : the geometry GGUF dir
    std::string image;                 // --image : any photo (RMBG-2.0 mattes it in-process)
    std::string out;                   // --out   : the textured + rigged GLB

    // ---- rig weights ----
    std::string r1w     = "/home/dbrain/models/3d/rig/r1w_real";
    std::string qwen3_w = "/home/dbrain/models/3d/rig/qwen3_w";
    std::string skinvae = "/home/dbrain/models/3d/rig/skin_vae_gguf";

    // ---- camera / backend ----
    float cam = DEF_CAM, dist = DEF_DIST, ms = DEF_MS;
    bool  use_cuda = true;             // --cpu
    bool  fast     = false;            // --fast

    // ---- rig recipe ----
    // Inline-API default = DETERMINISTIC beam (do_sample=false) + beams=20, the fan-free recipe.
    // --rig-sample restores the stochastic scaffold recipe (do_sample=true, beams=10).
    bool     rig_sample = false;
    bool     rig_structural_select = true;   // --no-rig-select
    int      rig_retries = 0;                // extra conditioning DRAWS when the anatomy gate rejects all
    // --rig-extra-retries: ONE MORE DRAW BEYOND THE BUDGET, granted only when the budget ran out
    // with NOTHING accepted and nothing even passing the pose gate. "If none pass, is it worth one
    // more?" — yes, but only in that corner: a rejected draw is typically a ~214 s runaway, so this
    // is the difference between a worst case of (retries+1) and (retries+2) draws and it must never
    // fire on a run that already has a usable rig. 0 disables.
    int      rig_extra_retries = 1;
    bool     allow_zero_skin = false;        // escape hatch for the weightless-rig guard
    // --rig-pose-gate / --no-rig-pose-gate. UNSET = on iff rig_retries > 0: with no re-draw there is
    // no choice to make, so the audit would cost ~4 s and change nothing. It is the deformation term
    // of the selector predicate, and without it a creature-gate draw can never be accepted.
    Opt<bool> rig_pose_gate;
    // --rig-min-named-core: the floor on SMPL-22 core bones a draw must NAME to be accepted. The
    // retargeter maps through names and its topology fallback fails on miku, so a high-scoring rig
    // that names 14 bones animates worse than a lower-scoring one that names 21. 19 = the lowest
    // core count the existing humanoid structural gate ever accepts.
    int      rig_min_named_core = 19;
    // --rig-pose-gate-strict: also require the pose gate on the HUMANOID branch of the accept
    // predicate, not only on the creature branch. Off by default (the spec'd predicate); on, char1's
    // 22/22-named, rig_score-0.921, pose-gate-77.864 draw 0 stops being an accept.
    bool     rig_pose_gate_strict = false;
    int      num_beams = 20;
    uint64_t rig_seed = 0;
    bool     no_rig = false;
    std::string rig_cache;                   // one skeleton shared across LOD tiers

    // ---- bone naming ----
    bool bone_names = true, bone_names_smpl = false;
    int  bone_facing = 0;              // 0 = auto-derive from the feet

    // ---- coarse mesh ----
    bool clean = true;                 // --no-clean
    int  mc_stride = 2, mc_blur = 0, mc_smooth = 1;
    bool dc_remesh = false;            // the Python-parity O-Voxel narrow-band DC path
    int  dc_band = 1, dc_taubin = 2;
    bool solid_mesh = false;           // --solid: signed field. Better geometry, BREAKS the rig.

    // ---- bake / decimate ----
    int  texsize = 1024, decimate = 150000;
    // --dc-remesh raises texsize/decimate/refine defaults ONLY if the caller did not set them.
    // A library caller that means 2048/220000 must set both the value AND the _set flag, exactly as
    // passing --texsize on the command line does.
    bool texsize_set = false, decimate_set = false, refine_set = false;
    bool refine = true;                // native UltraShape refine

    // ---- texture ----
    bool tex_snap_volume = false, tex_volume_direct = false;
    int  tex_fallback_r = -1;          // -1 = mode default (0 for snap modes, 8 for direct)
    bool tex_project = false, tex_project_overlay = false;
    std::string tex_front, tex_back;
    std::vector<TexView> tex_views;

    // ---- retopology ----
    bool part_retopo = false;
    std::string p3sam_w = "/mnt/hdd/3d/avatar-shootout/p3sam_goldens/weights_npy";
    std::string obj_decimate = "./obj_decimate";
    bool do_quad = false;
    std::string retopo_tool = "im";    // im (default) | quadwild

    // ---- camera estimation ----
    bool use_moge = false;
    std::string moge_w = "/mnt/hdd/3d/avatar-shootout/moge_goldens/weights_npy";

    // ---- staging / resume ----
    std::string stage_dir, from_refined, dump_geo, from_geo;
    bool geometry_only = false, material_cache_only = false;
    bool image_model_ready = false;    // diagnostic only

    // ---- mirrored: pix::ChainInput (the geometry chain's own defaults win when unset) ----
    Opt<int>         geo_resolution;   // --resolution / --res (must be a multiple of 16)
    Opt<int>         geo_seed;         // --seed
    Opt<float>       geo_guidance;     // --guidance  (ss + shape)
    Opt<int>         geo_steps;        // --steps     (ss + shape + tex)
    Opt<bool>        geo_tex_proj;     // --tex-dit proj|cross
    Opt<std::string> geo_tex_flow_w;   // --tex-dit-w

    // ---- mirrored: usr::RefineCfg ----
    Opt<int>         us_octree;
    Opt<int>         us_num_latents;
    Opt<int>         us_steps;
    Opt<float>       us_guidance;
    Opt<int64_t>     us_chunk;
    Opt<std::string> us_gguf_dir, us_dit_w, us_vae_w, us_cnd_w, us_meta;

    // ---- mirrored: qr::QuadCfg / imretopo::ImCfg ----
    Opt<std::string> quad_repo;
    Opt<float>       im_adaptivity;
    Opt<int>         im_target_verts;
};

// argv -> Options. Applies ON TOP of whatever `o` already holds, so a library caller can fill the
// typed fields and then hand through an escape-hatch flag list (later wins, exactly as a later argv
// element wins today).
//   returns -1  : parsed, keep going
//   returns >=0 : the process exit code the CLI would return (usage/erroneous flag already printed)
int parse_args(int argc, char** argv, Options& o);

// The stage itself: everything image_to_rig's main() does after parsing. Taken BY VALUE because
// the body resolves implied defaults into its own copy (--dc-remesh raising texsize, MoGe replacing
// the camera, ...). Returns RC_OK / RC_FAIL / RC_CANCELLED.
int run(Options o);

}  // namespace i2r

// The pre-struct entry point, kept so nothing that already calls it breaks: parse + run.
int image_to_rig_main(int argc, char** argv);
