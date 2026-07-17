// Validate moge::recover_fov (host recover_focal_shift + FOV) against the banked golden.
// Feeds the golden full-res point map + mask -> intrinsics fx_norm + camera_angle_x, compares to
// fov_result.npy [camera_angle_x_rad, deg, fx_norm] (= 0.71999 rad / 41.252 deg / 1.32838).
//   g++ -O2 -std=c++17 moge_fov_test.cpp -o moge_fov_test   (no ggml; built via build.sh moge_fov_test)
#include "moge_fov.hpp"
#include "../../sparse_spike/npy.hpp"
#include <cstdio>
#include <string>

static const char* DIR = "/mnt/hdd/3d/avatar-shootout/moge_goldens";

int main() {
    std::string d = DIR;
    // Faithful MoGe path: feed the RAW head outputs (points_head_out/mask_head_out, 960^2 CHW),
    // resize->remap->recover internally. (out["points"] is post-shift/metric_scale -> wrong input.)
    NpyArray ph = npy_load(d + "/points_head_out.npy");  // [1,3,960,960]
    NpyArray mh = npy_load(d + "/mask_head_out.npy");     // [1,1,960,960]
    NpyArray fr = npy_load(d + "/fov_result.npy");        // [3] f64
    int Hh = (int)ph.shape[2], Wh = (int)ph.shape[3];
    const int Himg = 1793, Wimg = 1793;                   // infer resolution (meta)
    double AR = (double)Wimg / (double)Himg;              // 1.0 (square)
    moge::FovResult R = moge::recover_fov_from_heads(ph.f32(), mh.f32(), Hh, Wh, Himg, Wimg, AR);

    const double* g = fr.f64();
    printf("[moge_fov] n_valid=%d  focal=%.6f shift=%.6f\n", R.n_valid, R.focal, R.shift);
    printf("[moge_fov]  got: cam_x=%.6f rad (%.4f deg)  fx_norm=%.6f\n", R.camera_angle_x,
           R.camera_angle_x * 180.0 / M_PI, R.fx_norm);
    printf("[moge_fov]  ref: cam_x=%.6f rad (%.4f deg)  fx_norm=%.6f\n", g[0], g[1], g[2]);
    double drad = std::fabs(R.camera_angle_x - g[0]);
    double dfx  = std::fabs(R.fx_norm - g[2]);
    bool ok = drad < 2e-3 && dfx < 5e-3;   // ~0.1 deg
    printf("[moge_fov] dcam=%.2e dfx=%.2e  %s\n", drad, dfx, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
