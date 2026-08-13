// Standalone driver for the narrow-band UDF dual contouring remesh (Pixal3D/CuMesh
// `remesh_narrow_band_dc`). The algorithm itself now lives in narrow_band_dc_cuda.cu so
// image_to_rig --dc-remesh runs the SAME code in-process; this stays as the GLB-in/GLB-out
// probe used by shootout/native_ovoxel_dc_parity.sh and for A/B checks.
//
// Usage: narrow_band_dc_probe input.glb output.glb [resolution=1024] [band=1]
#include "glb_reader.hpp"
#include "glb_writer.hpp"
#include "narrow_band_dc.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3 || argc > 5) {
        std::fprintf(stderr, "usage: %s input.glb output.glb [resolution=1024] [band=1]\n", argv[0]);
        return 2;
    }
    const int resolution = argc >= 4 ? std::atoi(argv[3]) : 1024;
    const float band = argc == 5 ? std::strtof(argv[4], nullptr) : 1.f;

    glb::Mesh in;
    if (!glb::read_glb(argv[1], in)) return 1;

    // CUDA_VISIBLE_DEVICES pins the 3060, so device 0 is the intended card (see the runbook).
    const cudaError_t e = cudaSetDevice(0);
    if (e != cudaSuccess) { std::fprintf(stderr, "cudaSetDevice(0): %s\n", cudaGetErrorString(e)); return 1; }

    std::vector<float> out_verts;
    std::vector<int64_t> out_faces;
    if (!nbdc::remesh(in.verts, in.faces, resolution, band, out_verts, out_faces)) return 1;
    if (!glb::write_glb(argv[2], out_verts, out_faces)) return 1;
    return 0;
}
