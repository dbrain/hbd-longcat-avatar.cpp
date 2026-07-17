// Validate the MoGe-2 conv stack (neck + points_head + mask_head) against the banked goldens.
//   DINOv2-L (dinov2_graph) -> output_projections sum -> UV pyramid -> neck -> points/mask heads.
// Compares neck_out [1,32,960,960], points_head_out [1,3,960,960], mask_head_out [1,1,960,960].
//
//   ./build.sh moge_test          # CPU
//   ./build.sh moge_test cuda     # CUDA
#include "dinov2_graph.hpp"
#include "moge_neck.hpp"
#include "moge_fov.hpp"
#include <cstdio>
#include <string>

static const char* DIR  = "/mnt/hdd/3d/avatar-shootout/moge_goldens";
static const char* WDIR = "/mnt/hdd/3d/avatar-shootout/moge_goldens/weights_npy";

int main(int argc, char** argv) {
    bool use_cuda = (argc > 1 && std::string(argv[1]) == "cuda");
    const int rows = 60, cols = 60;        // 840/14
    const double AR = 1793.0 / 1793.0;     // image aspect ratio W/H (meta: 1793x1793)
    M1Harness H(WDIR, 2048, use_cuda);
    ggml_context* ctx = H.ctx;

    int64_t img_ne[4] = {(int64_t)cols * 14, (int64_t)rows * 14, 3, 1};
    ggml_tensor* img = H.input("image", 4, img_ne);
    NpyArray pe = npy_load(std::string(WDIR) + "/encoder.backbone.pos_embed.npy");
    std::vector<float> pos = dinov2::interp_pos_embed(pe.f32(), rows, cols);
    int64_t pos_ne[4] = {dinov2::HID, (int64_t)(1 + rows * cols), 1, 1};
    ggml_tensor* pos_const = H.const_tensor("pos_embed_interp", 2, pos_ne, pos);

    // DINOv2 backbone -> final-normed patch tokens at blocks 5/11/17/23.
    std::vector<int> capture = {5, 11, 17, 23};
    std::vector<ggml_tensor*> inter; ggml_tensor* cls_out = nullptr;
    dinov2::build(ctx, H, img, pos_const, rows, cols, capture, inter, &cls_out);

    // output_projections: 1x1 conv (== per-pixel linear) on each normed patch map, SUM the 4.
    const int NP = rows * cols;
    ggml_tensor* enc = nullptr;
    for (int k = 0; k < 4; k++) {
        std::string pk = "encoder.output_projections." + std::to_string(k);
        ggml_tensor* w = ggml_reshape_2d(ctx, H.weight(pk + ".weight"), dinov2::HID, dinov2::HID);  // [in,out]
        ggml_tensor* proj = lin(ctx, w, H.weight(pk + ".bias"), inter[k]);                          // [HID, NP]
        enc = enc ? ggml_add(ctx, enc, proj) : proj;
    }
    // [HID, NP] (channel-fast) -> spatial [cols, rows, HID] (channel-slow) via transpose+reshape.
    ggml_tensor* enc_t = ggml_cont(ctx, ggml_transpose(ctx, enc));            // [NP, HID]
    ggml_tensor* enc_sp = ggml_reshape_3d(ctx, enc_t, cols, rows, dinov2::HID); // [cols,rows,HID]
    enc_sp = ggml_reshape_4d(ctx, enc_sp, cols, rows, dinov2::HID, 1);

    // UV pyramid: level0 concat with enc; levels 1..4 are uv-only (2ch).
    std::vector<ggml_tensor*> in_feats(5, nullptr);
    for (int L = 0; L < 5; L++) {
        int w = cols << L, h = rows << L;
        std::vector<float> uvd = moge::view_plane_uv(w, h, AR);
        int64_t uv_ne[4] = {w, h, 2, 1};
        ggml_tensor* uv = H.const_tensor(("uv" + std::to_string(L)).c_str(), 4, uv_ne, uvd);
        in_feats[L] = (L == 0) ? ggml_concat(ctx, enc_sp, uv, 2) : uv;  // concat along channel
    }

    const int neck_res[5]  = {0, 2, 2, 2, 0};
    const int head_res[5]  = {0, 1, 1, 1, 0};
    std::vector<ggml_tensor*> nf = moge::convstack(ctx, H, "neck", in_feats, neck_res, /*has_out4=*/false);
    ggml_tensor* neck_out = nf[4]; ggml_set_output(neck_out);
    std::vector<ggml_tensor*> ph = moge::convstack(ctx, H, "points_head", nf, head_res, true);
    ggml_tensor* points = ph[4]; ggml_set_output(points);
    std::vector<ggml_tensor*> mh = moge::convstack(ctx, H, "mask_head", nf, head_res, true);
    ggml_tensor* mask = mh[4]; ggml_set_output(mask);

    ggml_cgraph* gf = new_graph(ctx, 65536);
    ggml_build_forward_expand(gf, neck_out);
    ggml_build_forward_expand(gf, points);
    ggml_build_forward_expand(gf, mask);
    H.alloc_and_upload(gf);
    H.upload_input_npy(img, std::string(DIR) + "/image_14_norm.npy");
    H.compute(gf);

    printf("[moge] backend=%s\n", use_cuda ? "cuda" : "cpu");
    CmpStats n = compare_to_npy(H, neck_out, std::string(DIR) + "/neck_out.npy", true, "neck_out");
    CmpStats p = compare_to_npy(H, points,   std::string(DIR) + "/points_head_out.npy", true, "points_head");
    CmpStats m = compare_to_npy(H, mask,     std::string(DIR) + "/mask_head_out.npy", true, "mask_head");
    // neck/points are O(1) feature/point maps (gate on meanabs); mask is a pre-sigmoid LOGIT map
    // (|x|~14) -> gate on maxabs logit tolerance (0.2 logit => <1e-2 sigmoid in the transition band).
    bool ok = n.meanabs < 1e-3 && p.meanabs < 1e-3 && m.maxabs < 0.2;

    // End-to-end Stage 1: feed the ACTUAL ggml head outputs into the host FOV recovery.
    // ggml points [W,H,C,1] read back == CHW (c outer, h, w) — exactly recover_fov_from_heads' layout.
    const int Whd = (int)points->ne[0], Hhd = (int)points->ne[1];
    std::vector<float> pbuf((size_t)3 * Whd * Hhd), mbuf((size_t)Whd * Hhd);
    ggml_backend_tensor_get(points, pbuf.data(), 0, pbuf.size() * 4);
    ggml_backend_tensor_get(mask,   mbuf.data(), 0, mbuf.size() * 4);
    moge::FovResult R = moge::recover_fov_from_heads(pbuf.data(), mbuf.data(), Hhd, Whd, 1793, 1793, AR);
    NpyArray fr = npy_load(std::string(DIR) + "/fov_result.npy");
    const double* g = fr.f64();
    printf("[moge] FOV  got cam_x=%.6f rad (%.4f deg) fx=%.6f  |  ref %.4f deg fx=%.6f\n",
           R.camera_angle_x, R.camera_angle_x * 180.0 / M_PI, R.fx_norm, g[1], g[2]);
    // FOV from our own (cosine-1.0 but not bit-exact) heads drifts ~0.13deg vs ref through the
    // sensitive LM; the optimizer itself is exact (3e-7) on the golden heads. ~0.3% is parity for a
    // recovered camera angle that only sets reconstruction framing.
    bool fov_ok = std::fabs(R.camera_angle_x - g[0]) < 5e-3;
    ok = ok && fov_ok;
    printf("[moge] %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
