// Standalone driver for the narrow-band UDF dual contouring remesh (Pixal3D/CuMesh
// `remesh_narrow_band_dc`). The algorithm itself now lives in narrow_band_dc_cuda.cu so
// image_to_rig --dc-remesh runs the SAME code in-process; this stays as the GLB-in/GLB-out
// probe used by shootout/native_ovoxel_dc_parity.sh and for A/B checks.
//
// Usage: narrow_band_dc_probe input.glb output.glb [resolution=1024] [band=1] [occupancy.bin]
//
// `occupancy.bin` is the size-prefixed int32 [N,4] O-Voxel occupancy that image_to_rig stages as
// pbr_coords.bin.  Supplying it SIGNS the distance field (see narrow_band_dc.hpp), which is what
// turns the two-walled shrink-wrap envelope into a single solid surface.  Omitting it reproduces
// the historical unsigned output bit-for-bit, which is what the res-1024 parity A/B needs.
// Env: NBDC_SOLID_SEAL / NBDC_SOLID_ERODE override the occupancy seal / pull-back radii.
#include "glb_reader.hpp"
#include "glb_writer.hpp"
#include "narrow_band_dc.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3 || argc > 6) {
        std::fprintf(stderr,
                     "usage: %s input.glb output.glb [resolution=1024] [band=1] [occupancy.bin]\n",
                     argv[0]);
        return 2;
    }
    const int resolution = argc >= 4 ? std::atoi(argv[3]) : 1024;
    const float band = argc >= 5 ? std::strtof(argv[4], nullptr) : 1.f;
    const char* occ_path = argc >= 6 ? argv[5] : nullptr;

    glb::Mesh in;
    if (!glb::read_glb(argv[1], in)) return 1;

    // CUDA_VISIBLE_DEVICES pins the 3060, so device 0 is the intended card (see the runbook).
    const cudaError_t e = cudaSetDevice(0);
    if (e != cudaSuccess) { std::fprintf(stderr, "cudaSetDevice(0): %s\n", cudaGetErrorString(e)); return 1; }

    solidfield::SolidField interior;
    if (occ_path) {
        FILE* f = std::fopen(occ_path, "rb");
        if (!f) { std::fprintf(stderr, "cannot open %s\n", occ_path); return 1; }
        size_t nbytes = 0;
        if (std::fread(&nbytes, sizeof(nbytes), 1, f) != 1) { std::fclose(f); return 1; }
        std::vector<int32_t> coords(nbytes / sizeof(int32_t));
        if (std::fread(coords.data(), 1, nbytes, f) != nbytes) { std::fclose(f); return 1; }
        std::fclose(f);
        const int seal = std::getenv("NBDC_SOLID_SEAL") ? std::atoi(std::getenv("NBDC_SOLID_SEAL")) : 1;
        const int erode = std::getenv("NBDC_SOLID_ERODE") ? std::atoi(std::getenv("NBDC_SOLID_ERODE")) : 1;
        interior = solidfield::build(coords.data(), (int)(coords.size() / 4), resolution, seal, erode, true);
    }

    std::vector<float> out_verts;
    std::vector<int64_t> out_faces;
    if (!nbdc::remesh(in.verts, in.faces, resolution, band, out_verts, out_faces, true,
                      interior.empty() ? nullptr : &interior)) return 1;
    if (!glb::write_glb(argv[2], out_verts, out_faces)) return 1;
    return 0;
}
