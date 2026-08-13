#include "image_io.hpp"
#include "../../sparse_spike/npy.hpp"
#include <cstdio>
static double cmp(const std::vector<float>& a, const std::string& path){
    NpyArray r=npy_load(path); const float* y=r.f32();
    double ma=0,sum=0; for(size_t i=0;i<a.size();i++){double d=std::fabs((double)a[i]-y[i]); ma=std::max(ma,d); sum+=d;}
    printf("  vs %s: maxabs=%.4f meanabs=%.5f (n=%zu)\n", path.c_str(), ma, sum/a.size(), a.size()); return ma;
}
int main(){
    auto c512 = imgio::load_chw("../../sparse_spike/golden_stages/pre/preprocessed.png", 512);
    auto c1024 = imgio::load_chw("../../sparse_spike/golden_stages/pre/preprocessed.png", 1024);
    printf("[resize_test] C++ Lanczos-3 vs PIL-derived refs:\n");
    cmp(c512, "refs/naf_guide.npy");
    cmp(c1024, "refs/stage3b/image_1024_chw.npy");
    return 0;
}
