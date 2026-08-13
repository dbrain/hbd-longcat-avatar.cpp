#include "tex_atlas_cuda.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

int main() {
    // Unit square split exactly as a packed atlas chart. This checks coverage,
    // barycentric position interpolation and normal interpolation without
    // needing a model or Python fixture.
    const std::vector<float> v = {0,0,0, 1,0,0, 1,1,0, 0,1,0};
    const std::vector<float> n = {0,0,1, 0,0,1, 0,0,1, 0,0,1};
    const std::vector<uint32_t> f = {0,1,2, 0,2,3};
    const std::vector<float> uv = {0,0, 8,0, 8,6, 0,6};
    std::vector<float> pos, nr; std::vector<uint8_t> mask;
    texatlas_cuda::RasterStats s; std::string err;
    if (!texatlas_cuda::raster_uv(v,n,f,uv,8,6,1,pos,nr,mask,&s,&err)) {
        std::fprintf(stderr,"FAIL CUDA raster: %s\n",err.c_str()); return 1;
    }
    int covered=0; float worst=0.f;
    for (int y=0;y<6;y++) for(int x=0;x<8;x++) {
        const size_t p=(size_t)y*8+x;
        if (!mask[p]) { std::fprintf(stderr,"FAIL uncovered pixel %d,%d\n",x,y); return 1; }
        ++covered;
        const float ex=((float)x+.5f)/8.f, ey=((float)y+.5f)/6.f;
        worst=std::max(worst,std::fabs(pos[p*3]-ex));
        worst=std::max(worst,std::fabs(pos[p*3+1]-ey));
        worst=std::max(worst,std::fabs(pos[p*3+2]));
        worst=std::max(worst,std::fabs(nr[p*3+2]-1.f));
    }
    std::printf("[tex-atlas-cuda] covered=%d/48 worst=%.8f upload=%.4fs kernel=%.4fs download=%.4fs\n",
                covered,worst,s.upload_seconds,s.kernel_seconds,s.download_seconds);
    return worst <= 2e-6f ? 0 : 1;
}
