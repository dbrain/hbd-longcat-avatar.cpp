// C++ reference for the Pixal3D "proj" image-conditioning geometry (net-new op #2/#3:
// camera unproject + 2D bilinear grid_sample). Mirrors proj_cond_ref.py and is the
// host-side production candidate for the unproject (grid_sample coords are computed
// from 3 camera scalars on the host; only the bilinear gather needs a GPU kernel in
// prod). Validated vs the torch ProjGrid golden (golden_ref/proj_case0/).
//
// Build:  g++ -O2 -std=c++17 proj_cond.cpp -o proj_cond_test
// Run:    ./proj_cond_test golden_ref/proj_case0
//
// fp32 throughout to mirror torch fp32. All math matches proj_cond_ref.py line-for-line.
#include "../sparse_spike/npy.hpp"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using std::vector;

// ---- 4x4 inverse (Gauss-Jordan, fp32) — matches torch.linalg.inv(fp32) to ~1e-6 ----
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

// Build [R^3,3] Blender-frame grid points (meshgrid ij, x,y,z; then @ ROT.T).
static vector<float> build_grid_points(int R) {
    // ROT = [[1,0,0],[0,0,-1],[0,1,0]]; gp_rot = gp @ ROT.T:
    //   x' = x*1            = x
    //   y' = x*0+y*0+z*(-1) = -z   (ROT row1 = [0,0,-1])
    //   z' = x*0+y*1+z*0    = y    (ROT row2 = [0,1,0])
    vector<float> gp((size_t)R * R * R * 3);
    auto lin = [&](int i) { return R == 1 ? -1.f : -1.f + 2.f * i / (R - 1); };
    size_t t = 0;
    for (int i = 0; i < R; i++)
        for (int j = 0; j < R; j++)
            for (int k = 0; k < R; k++) {
                float x = lin(i), y = lin(j), z = lin(k);
                gp[t * 3 + 0] = x;
                gp[t * 3 + 1] = -z;
                gp[t * 3 + 2] = y;
                t++;
            }
    return gp;
}

// Bilinear grid_sample, align_corners=False, padding_mode=border. fmap [C,H,W] (chw),
// gx/gy in [-1,1]. Writes out[C] for one query.
static inline void sample_one(const float* fmap_chw, int C, int H, int W,
                              float gx, float gy, float* out) {
    float ix = (gx + 1.f) * 0.5f * W - 0.5f;
    float iy = (gy + 1.f) * 0.5f * H - 0.5f;
    int x0 = (int)std::floor(ix), y0 = (int)std::floor(iy);
    int x1 = x0 + 1, y1 = y0 + 1;
    float wx1 = ix - x0, wy1 = iy - y0, wx0 = 1.f - wx1, wy0 = 1.f - wy1;
    auto cl = [](int v, int hi) { return v < 0 ? 0 : (v > hi ? hi : v); };
    int x0c = cl(x0, W - 1), x1c = cl(x1, W - 1), y0c = cl(y0, H - 1), y1c = cl(y1, H - 1);
    const size_t HW = (size_t)H * W;
    for (int c = 0; c < C; c++) {
        const float* p = fmap_chw + (size_t)c * HW;
        float f00 = p[(size_t)y0c * W + x0c], f01 = p[(size_t)y0c * W + x1c];
        float f10 = p[(size_t)y1c * W + x0c], f11 = p[(size_t)y1c * W + x1c];
        out[c] = f00 * (wy0 * wx0) + f01 * (wy0 * wx1) + f10 * (wy1 * wx0) + f11 * (wy1 * wx1);
    }
}

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : "golden_ref/proj_case0";
    // case params (must match dump_golden in test_proj_cond.py)
    const int R = 16, img_res = 512;
    const float cam = 0.8575560450553894f, dist = 2.0f, ms = 1.0f;

    NpyArray fmap = npy_load(dir + "/fmap_hwc.npy");   // [H,W,C]
    NpyArray exp = npy_load(dir + "/expected.npy");    // [R^3,C]
    int H = (int)fmap.shape[0], W = (int)fmap.shape[1], C = (int)fmap.shape[2];
    const float* fhwc = fmap.f32();

    // to CHW
    vector<float> fchw((size_t)C * H * W);
    for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++)
            for (int c = 0; c < C; c++)
                fchw[(size_t)c * H * W + (size_t)h * W + w] = fhwc[((size_t)h * W + w) * C + c];

    // transform + inverse
    float T[16] = {1,0,0,0,  0,0,-1,-dist,  0,1,0,0,  0,0,0,1};
    float Winv[16];
    inv4x4(T, Winv);

    vector<float> gp = build_grid_points(R);
    int N = R * R * R;
    const float focal = 16.f / std::tan(cam / 2.f);
    const float focal_px = focal * img_res / 32.f;

    vector<float> out((size_t)N * C);
    vector<float> qv(C);
    for (int n = 0; n < N; n++) {
        float gx = gp[n * 3 + 0] / ms / 2.f;
        float gy = gp[n * 3 + 1] / ms / 2.f;
        float gz = gp[n * 3 + 2] / ms / 2.f;
        // homogeneous [x,y,z,1] @ Winv^T  -> camera coords (first 3)
        float xc = Winv[0] * gx + Winv[1] * gy + Winv[2] * gz + Winv[3];
        float yc = Winv[4] * gx + Winv[5] * gy + Winv[6] * gz + Winv[7];
        float zc = Winv[8] * gx + Winv[9] * gy + Winv[10] * gz + Winv[11];
        float x_ndc = focal_px * xc / (-zc + 1e-8f);
        float y_ndc = focal_px * yc / (-zc + 1e-8f);
        float x_pix = x_ndc + img_res / 2.f;
        float y_pix = -y_ndc + img_res / 2.f;
        float gnx = (x_pix + 0.5f) / img_res * 2.f - 1.f;
        float gny = (y_pix + 0.5f) / img_res * 2.f - 1.f;
        sample_one(fchw.data(), C, H, W, gnx, gny, qv.data());
        for (int c = 0; c < C; c++) out[(size_t)n * C + c] = qv[c];
    }

    // compare
    const float* e = exp.f32();
    double maxabs = 0; int nb = (int)exp.numel();
    for (int i = 0; i < nb; i++) maxabs = std::max(maxabs, (double)std::fabs(out[i] - e[i]));
    printf("[proj_cond] N=%d C=%d  maxabs=%.3e  %s\n", N, C, maxabs,
           maxabs < 1e-4 ? "PASS" : "FAIL");
    return maxabs < 1e-4 ? 0 : 1;
}
