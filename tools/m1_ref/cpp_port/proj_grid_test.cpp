// Stage-1 "proj" image conditioning (net-new op #2/#3) in ggml.
//
// Mirrors proj_cond_ref.proj_grid_forward: camera-unproject the constant 16^3 grid to
// pixel coords (host scalar arithmetic, depends only on 3 camera scalars), then bilinear
// grid_sample(align_corners=False, border) the DINOv3 patch map. The bilinear gather is
// expressed in ggml as 4 ggml_get_rows (the 4 corners) + weighted sum — the corner
// indices + weights are host constants (don't depend on feature values), so NO custom
// CUDA kernel is needed for correctness. (A fused grid_sample CUDA kernel is a perf-phase
// option, noted in the handoff.)
//
// Validates z_proj [1,4096,1024] vs refs/proj.npy (== golden_stages/stage1_cond/proj.npy).
#include "m1_ggml.hpp"
#include <cmath>

static const char* WDIR = "weights_npy/dinov3";  // unused (no weights); kept for harness
static const char* REFS = "refs";

// 4x4 inverse (Gauss-Jordan, fp32) — matches proj_cond.cpp / torch.linalg.inv(fp32).
static void inv4x4(const float A[16], float out[16]) {
    float m[4][8];
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) m[r][c] = A[r * 4 + c];
        for (int c = 0; c < 4; c++) m[r][4 + c] = (r == c) ? 1.f : 0.f;
    }
    for (int col = 0; col < 4; col++) {
        int piv = col;
        for (int r = col + 1; r < 4; r++)
            if (std::fabs(m[r][col]) > std::fabs(m[piv][col])) piv = r;
        for (int c = 0; c < 8; c++) std::swap(m[col][c], m[piv][c]);
        float d = m[col][col];
        for (int c = 0; c < 8; c++) m[col][c] /= d;
        for (int r = 0; r < 4; r++) {
            if (r == col) continue;
            float f = m[r][col];
            for (int c = 0; c < 8; c++) m[r][c] -= f * m[col][c];
        }
    }
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) out[r * 4 + c] = m[r][4 + c];
}

int main(int argc, char** argv) {
    bool use_cuda = (argc > 1 && std::string(argv[1]) == "cuda");
    // case params: ss conditioner (grid 16, image 512); cam from golden cam.json
    const int R = 16, img_res = 512, H = 32, W = 32, C = 1024;
    // camera scalars from golden_stages/cam.json (MoGe, miku decode)
    const float cam = 0.7332379387484828f, dist = 1.3021559715270996f, ms = 1.0f;
    const int N = R * R * R;  // 4096

    // ---- host: build grid points (Blender frame), project, corner idx + weights ----
    float T[16] = {1,0,0,0,  0,0,-1,-dist,  0,1,0,0,  0,0,0,1};
    float Winv[16]; inv4x4(T, Winv);
    const float focal = 16.f / std::tan(cam / 2.f);
    const float focal_px = focal * img_res / 32.f;
    auto lin1 = [&](int i) { return R == 1 ? -1.f : -1.f + 2.f * i / (R - 1); };

    std::vector<int32_t> idx00(N), idx01(N), idx10(N), idx11(N);
    std::vector<float> w00(N), w01(N), w10(N), w11(N);
    int t = 0;
    for (int i = 0; i < R; i++)
        for (int j = 0; j < R; j++)
            for (int k = 0; k < R; k++) {
                float x = lin1(i), y = lin1(j), z = lin1(k);
                // gp @ ROT.T -> (x, -z, y); then / ms / 2
                float gx = x / ms / 2.f, gy = (-z) / ms / 2.f, gz = (y) / ms / 2.f;
                float xc = Winv[0]*gx + Winv[1]*gy + Winv[2]*gz + Winv[3];
                float yc = Winv[4]*gx + Winv[5]*gy + Winv[6]*gz + Winv[7];
                float zc = Winv[8]*gx + Winv[9]*gy + Winv[10]*gz + Winv[11];
                float x_ndc = focal_px * xc / (-zc + 1e-8f);
                float y_ndc = focal_px * yc / (-zc + 1e-8f);
                float x_pix = x_ndc + img_res / 2.f;
                float y_pix = -y_ndc + img_res / 2.f;
                float gnx = (x_pix + 0.5f) / img_res * 2.f - 1.f;
                float gny = (y_pix + 0.5f) / img_res * 2.f - 1.f;
                // bilinear, align_corners=False, border
                float ix = (gnx + 1.f) * 0.5f * W - 0.5f;
                float iy = (gny + 1.f) * 0.5f * H - 0.5f;
                int x0 = (int)std::floor(ix), y0 = (int)std::floor(iy);
                int x1 = x0 + 1, y1 = y0 + 1;
                float wx1 = ix - x0, wy1 = iy - y0, wx0 = 1.f - wx1, wy0 = 1.f - wy1;
                auto cl = [](int v, int hi) { return v < 0 ? 0 : (v > hi ? hi : v); };
                int x0c = cl(x0, W - 1), x1c = cl(x1, W - 1), y0c = cl(y0, H - 1), y1c = cl(y1, H - 1);
                idx00[t] = y0c * W + x0c; w00[t] = wy0 * wx0;
                idx01[t] = y0c * W + x1c; w01[t] = wy0 * wx1;
                idx10[t] = y1c * W + x0c; w10[t] = wy1 * wx0;
                idx11[t] = y1c * W + x1c; w11[t] = wy1 * wx1;
                t++;
            }

    // ---- ggml graph ----
    M1Harness Hn(WDIR, 256, use_cuda);
    ggml_context* ctx = Hn.ctx;

    int64_t pne[4] = {C, W, H, 1};
    ggml_tensor* patch = Hn.input("patchmap", 3, pne);     // [ch,W,H]
    // corner index tensors (I32) + weight tensors (F32)
    auto mk_i32 = [&](const char* nm) {
        ggml_tensor* t2 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
        ggml_set_name(t2, nm); ggml_set_input(t2); return t2;
    };
    auto mk_w = [&](const char* nm) {
        int64_t wne[4] = {1, N, 1, 1};
        ggml_tensor* t2 = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, wne);
        ggml_set_name(t2, nm); ggml_set_input(t2); return t2;
    };
    ggml_tensor* i00 = mk_i32("i00"), *i01 = mk_i32("i01"), *i10 = mk_i32("i10"), *i11 = mk_i32("i11");
    ggml_tensor* a00 = mk_w("a00"), *a01 = mk_w("a01"), *a10 = mk_w("a10"), *a11 = mk_w("a11");

    ggml_tensor* fmap = ggml_reshape_2d(ctx, patch, C, (int64_t)H * W);  // [ch, pos]
    auto gather = [&](ggml_tensor* idx, ggml_tensor* wgt) {
        ggml_tensor* g = ggml_get_rows(ctx, fmap, idx);   // [ch, N]
        return ggml_mul(ctx, g, wgt);                     // broadcast wgt over ch
    };
    ggml_tensor* proj = ggml_add(ctx, ggml_add(ctx, gather(i00, a00), gather(i01, a01)),
                                      ggml_add(ctx, gather(i10, a10), gather(i11, a11)));
    proj = ggml_cont(ctx, proj);  // [ch=1024, N=4096]  == reverse([1,4096,1024])
    ggml_set_output(proj);

    ggml_cgraph* gf = new_graph(ctx);
    ggml_build_forward_expand(gf, proj);
    Hn.alloc_and_upload(gf);

    // upload inputs
    Hn.upload_input_npy(patch, std::string(REFS) + "/dino_patchmap.npy");
    ggml_backend_tensor_set(i00, idx00.data(), 0, N * 4);
    ggml_backend_tensor_set(i01, idx01.data(), 0, N * 4);
    ggml_backend_tensor_set(i10, idx10.data(), 0, N * 4);
    ggml_backend_tensor_set(i11, idx11.data(), 0, N * 4);
    Hn.upload_input_raw(a00, w00); Hn.upload_input_raw(a01, w01);
    Hn.upload_input_raw(a10, w10); Hn.upload_input_raw(a11, w11);

    Hn.compute(gf);

    printf("[proj_grid] backend=%s\n", use_cuda ? "cuda" : "cpu");
    CmpStats s = compare_to_npy(Hn, proj, std::string(REFS) + "/proj.npy", true, "z_proj");
    bool ok = s.maxabs < 1e-4;
    printf("[proj_grid] %s (tol 1e-4)\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
