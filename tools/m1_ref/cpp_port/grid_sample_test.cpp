// Validate the host trilinear grid_sample (tex_grid_sample.hpp) vs the flex_gemm oracle on the
// real PBR volume. Inputs from tex_grid_sample_capture.py (refs/stage4/). CPU-only, no ggml/GPU.
//   ./build.sh grid_sample_test && ./grid_sample_test
#include "tex_grid_sample.hpp"
#include "../../sparse_spike/npy.hpp"
#include <cstdio>
#include <cmath>
#include <string>

static const char* REFS = "refs/stage4";

int main() {
    NpyArray feN = npy_load(std::string(REFS)+"/tex_pbr.npy");          // [N,6] f32
    NpyArray coN = npy_load(std::string(REFS)+"/tex_out_coords.npy");   // [N,4] i32
    NpyArray qN  = npy_load(std::string(REFS)+"/tex_gs_query.npy");     // [Q,3] f32
    NpyArray oN  = npy_load(std::string(REFS)+"/tex_gs_oracle.npy");    // [Q,6] f32
    int N = (int)feN.shape[0], C = (int)feN.shape[1];
    int Q = (int)qN.shape[0];
    printf("[gs] volume N=%d C=%d, queries Q=%d\n", N, C, Q);

    std::vector<float> out((size_t)Q*C);
    texgs::grid_sample_trilinear(feN.f32(), coN.i32(), N, /*stride*/4, /*xoff*/1, C, qN.f32(), Q, out.data());

    const float* o = oN.f32();
    double ma=0, sum=0; size_t worst=0;
    for (size_t i=0;i<out.size();i++){ double d=std::fabs((double)out[i]-o[i]); if(d>ma){ma=d;worst=i;} sum+=d; }
    // also count how many queries hit nothing (tw==0 -> all-zero row, should match oracle zeros)
    int empties=0; for (int q=0;q<Q;q++){ bool z=true; for(int c=0;c<C;c++) if(out[(size_t)q*C+c]!=0.f){z=false;break;} if(z) empties++; }
    printf("[gs] C++ vs flex_gemm: maxabs=%.3e meanabs=%.3e worst@%zu got=%.6f ref=%.6f\n",
           ma, sum/out.size(), worst, out[worst], o[worst]);
    printf("[gs] empty(all-zero) queries: %d / %d\n", empties, Q);
    printf(ma < 1e-4 ? "[gs] PASS\n" : "[gs] FAIL (maxabs too high)\n");
    return ma < 1e-4 ? 0 : 1;
}
