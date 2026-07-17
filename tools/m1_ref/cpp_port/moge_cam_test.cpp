// Validate moge_cam.hpp callable: estimate_cam_angle_x on the golden source image (compare to
// fov_result.npy = 0.71999 rad / 41.25 deg) and on the actual matte input. CUDA build:
//   ./build.sh moge_cam_test cuda
#include "moge_cam.hpp"
#include "../../sparse_spike/npy.hpp"
#include <cstdio>
#include <cmath>
#include <string>

int main(int argc, char** argv) {
    bool cuda = !(argc > 1 && std::string(argv[1]) == "cpu");
    std::string W = "/mnt/hdd/3d/avatar-shootout/moge_goldens/weights_npy";
    NpyArray fr = npy_load("/mnt/hdd/3d/avatar-shootout/moge_goldens/fov_result.npy");
    double ref = fr.f64()[0];

    printf("== golden source image ==\n");
    float a1 = moge::estimate_cam_angle_x("_moge_golden_src.png", W, cuda, true);
    printf("  got %.5f rad (%.3f deg)  ref %.5f rad (%.3f deg)  d=%.3f deg\n",
           a1, a1*180/M_PI, ref, ref*180/M_PI, std::fabs(a1-ref)*180/M_PI);

    printf("== matte (_xtex_matte.png) ==\n");
    float a2 = moge::estimate_cam_angle_x("_xtex_matte.png", W, cuda, true);
    printf("  matte cam_angle_x = %.5f rad (%.3f deg)\n", a2, a2*180/M_PI);

    bool ok = std::fabs(a1 - ref) < (2.0 * M_PI / 180.0) && a2 > 0.05f && a2 < 3.0f;
    printf("%s\n", ok ? "MOGE_CAM PASS" : "MOGE_CAM CHECK");
    return ok ? 0 : 1;
}
