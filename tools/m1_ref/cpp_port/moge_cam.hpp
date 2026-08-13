// moge_cam.hpp — native MoGe-2 camera (FOV) estimation as a CALLABLE (TASK B fold-in).
// Wraps the validated standalone graph from moge_test.cpp (DINOv2-L backbone -> output_projections
// -> UV pyramid -> neck -> points/mask heads -> host LM FOV recovery) into one function that takes
// an image path and returns camera_angle_x (radians), so image_to_rig is 100% camera-native.
//
// Our inputs are always SQUARE character mattes (image_to_rig feeds a 512/1024 square matte), so the
// MoGe token grid is fixed at rows=cols=60 (840/14) and AR=1 — no variable-resolution path needed.
// Preprocess = resize to 840x840 [0,1] (imgio::load_chw) + DINOv2 ImageNet normalization
// (mean [0.485,0.456,0.406] std [0.229,0.224,0.225]) — verified bit-exact vs the banked image_14_norm.
#pragma once
#include "dinov2_graph.hpp"
#include "moge_neck.hpp"
#include "moge_fov.hpp"
#include "image_io.hpp"
#include <string>
#include <vector>
#include <cstdio>

namespace moge {

// weights_dir = MoGe weights_npy dir (default /mnt/hdd/3d/avatar-shootout/moge_goldens/weights_npy).
inline float estimate_cam_angle_x(const std::string& img_path, const std::string& weights_dir,
                                  bool use_cuda, bool verbose = false) {
    const int rows = 60, cols = 60;
    const double AR = 1.0;                       // square matte
    const int IMG = cols * 14;                   // 840
    M1Harness H(weights_dir, 2048, use_cuda);
    ggml_context* ctx = H.ctx;

    int64_t img_ne[4] = {(int64_t)IMG, (int64_t)IMG, 3, 1};
    ggml_tensor* img = H.input("image", 4, img_ne);
    NpyArray pe = npy_load(weights_dir + "/encoder.backbone.pos_embed.npy");
    std::vector<float> pos = dinov2::interp_pos_embed(pe.f32(), rows, cols);
    int64_t pos_ne[4] = {dinov2::HID, (int64_t)(1 + rows * cols), 1, 1};
    ggml_tensor* pos_const = H.const_tensor("pos_embed_interp", 2, pos_ne, pos);

    std::vector<int> capture = {5, 11, 17, 23};
    std::vector<ggml_tensor*> inter; ggml_tensor* cls_out = nullptr;
    dinov2::build(ctx, H, img, pos_const, rows, cols, capture, inter, &cls_out);

    const int NP = rows * cols;
    ggml_tensor* enc = nullptr;
    for (int k = 0; k < 4; k++) {
        std::string pk = "encoder.output_projections." + std::to_string(k);
        ggml_tensor* w = ggml_reshape_2d(ctx, H.weight(pk + ".weight"), dinov2::HID, dinov2::HID);
        ggml_tensor* proj = lin(ctx, w, H.weight(pk + ".bias"), inter[k]);
        enc = enc ? ggml_add(ctx, enc, proj) : proj;
    }
    ggml_tensor* enc_t = ggml_cont(ctx, ggml_transpose(ctx, enc));
    ggml_tensor* enc_sp = ggml_reshape_3d(ctx, enc_t, cols, rows, dinov2::HID);
    enc_sp = ggml_reshape_4d(ctx, enc_sp, cols, rows, dinov2::HID, 1);

    std::vector<ggml_tensor*> in_feats(5, nullptr);
    for (int L = 0; L < 5; L++) {
        int w = cols << L, h = rows << L;
        std::vector<float> uvd = moge::view_plane_uv(w, h, AR);
        int64_t uv_ne[4] = {w, h, 2, 1};
        ggml_tensor* uv = H.const_tensor(("uv" + std::to_string(L)).c_str(), 4, uv_ne, uvd);
        in_feats[L] = (L == 0) ? ggml_concat(ctx, enc_sp, uv, 2) : uv;
    }
    const int neck_res[5] = {0, 2, 2, 2, 0};
    const int head_res[5] = {0, 1, 1, 1, 0};
    std::vector<ggml_tensor*> nf = moge::convstack(ctx, H, "neck", in_feats, neck_res, false);
    std::vector<ggml_tensor*> ph = moge::convstack(ctx, H, "points_head", nf, head_res, true);
    ggml_tensor* points = ph[4]; ggml_set_output(points);
    std::vector<ggml_tensor*> mh = moge::convstack(ctx, H, "mask_head", nf, head_res, true);
    ggml_tensor* mask = mh[4]; ggml_set_output(mask);

    ggml_cgraph* gf = new_graph(ctx, 65536);
    ggml_build_forward_expand(gf, points);
    ggml_build_forward_expand(gf, mask);
    H.alloc_and_upload(gf);

    // preprocess: resize square matte -> 840x840 [0,1] CHW, DINOv2 ImageNet normalize
    std::vector<float> chw = imgio::load_chw(img_path, IMG);
    const float mean[3] = {0.485f, 0.456f, 0.406f}, sd[3] = {0.229f, 0.224f, 0.225f};
    const size_t HW = (size_t)IMG * IMG;
    for (int c = 0; c < 3; c++)
        for (size_t i = 0; i < HW; i++) chw[(size_t)c * HW + i] = (chw[(size_t)c * HW + i] - mean[c]) / sd[c];
    H.upload_input_raw(img, chw);
    H.compute(gf);

    const int Whd = (int)points->ne[0], Hhd = (int)points->ne[1];
    std::vector<float> pbuf((size_t)3 * Whd * Hhd), mbuf((size_t)Whd * Hhd);
    ggml_backend_tensor_get(points, pbuf.data(), 0, pbuf.size() * 4);
    ggml_backend_tensor_get(mask,   mbuf.data(), 0, mbuf.size() * 4);
    moge::FovResult R = moge::recover_fov_from_heads(pbuf.data(), mbuf.data(), Hhd, Whd, IMG, IMG, AR);
    if (verbose)
        printf("[moge] cam_angle_x=%.6f rad (%.4f deg) fx_norm=%.6f n_valid=%d\n",
               R.camera_angle_x, R.camera_angle_x * 180.0 / M_PI, R.fx_norm, R.n_valid);
    return (float)R.camera_angle_x;
}

} // namespace moge
