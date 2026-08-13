// Validate the native ImageProcessorV2 port (us_image_proc.hpp): raw PNG -> proc_image/proc_mask vs the
// banked cond_full_goldens (proc_image [1,3,1024,1024], proc_mask [1,1,1024,1024], both [-1,1]).
// char1_matte.png is RGB (no alpha) -> exercises the whole-image recenter path. Pass an alpha matte as
// argv[1] (+ its golden dir as argv[2]) to also cover the bbox-crop path.
//   ./build.sh us_image_proc_test
#include "us_image_proc.hpp"
#include "../../sparse_spike/npy.hpp"
#include <cstdio>
#include <string>
#include <cmath>

static const char* IMG  = "/mnt/hdd/3d/avatar-shootout/_shootout_out/char1_matte.png";
static const char* GDIR = "/mnt/hdd/3d/avatar-shootout/cond_full_goldens";

static void cmp(const char* tag, const float* a, const float* b, int64_t n, double& cos_out) {
    double dot=0,na=0,nb=0,maxa=0,sum=0;
    for (int64_t i=0;i<n;i++){ double d=std::fabs((double)a[i]-b[i]); if(d>maxa)maxa=d; sum+=d;
        dot+=(double)a[i]*b[i]; na+=(double)a[i]*a[i]; nb+=(double)b[i]*b[i]; }
    cos_out = dot/(std::sqrt(na)*std::sqrt(nb)+1e-30);
    printf("  [%s] n=%lld cosine=%.6f maxabs=%.3e meanabs=%.3e\n", tag,(long long)n,cos_out,maxa,sum/n);
}

int main(int argc, char** argv) {
    std::string img = argc > 1 ? argv[1] : IMG;
    std::string gdir = argc > 2 ? argv[2] : GDIR;
    printf("[imgproc] image=%s golden=%s\n", img.c_str(), gdir.c_str());

    us_imgproc::Processed p = us_imgproc::process(img, 1024, 0.15);

    NpyArray pim = npy_load(gdir + "/proc_image.npy");   // [1,3,1024,1024]
    NpyArray pmk = npy_load(gdir + "/proc_mask.npy");     // [1,1,1024,1024]
    printf("[imgproc] mine image=%zu mask=%zu  golden image=%lld mask=%lld\n",
           p.image.size(), p.mask.size(), (long long)pim.numel(), (long long)pmk.numel());

    double ci=0, cm=0;
    cmp("proc_image vs golden", p.image.data(), pim.f32(), pim.numel(), ci);
    cmp("proc_mask  vs golden", p.mask.data(),  pmk.f32(), pmk.numel(), cm);

    // mask is binary {-1,+1}; report exact-match fraction too (NEAREST should be near-perfect).
    int64_t n = pmk.numel(); long mism = 0;
    for (int64_t i=0;i<n;i++) if ((p.mask[i] > 0) != (pmk.f32()[i] > 0)) mism++;
    printf("[imgproc] mask sign-mismatch %ld / %lld (%.4f%%)\n", mism, (long long)n, 100.0*mism/n);

    bool pass = ci > 0.999 && cm > 0.999;
    printf("[imgproc] %s (image cosine=%.6f, mask cosine=%.6f)\n", pass?"PASS":"FAIL", ci, cm);
    return pass ? 0 : 1;
}
