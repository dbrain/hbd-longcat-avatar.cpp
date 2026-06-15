// im_batch_main — headless CLI front-end for Instant Meshes' batch_process() (the field-aligned
// quad-retopo core, NO GUI/OpenGL/nanogui). Mirrors upstream main.cpp's batch path. We need a robust
// remesher: QuadriFlow OOM-blows on the dense AI mesh's non-manifold/holey topology; Instant Meshes
// ingests the dense MANIFOLD source directly and emits clean field-aligned topology that preserves
// fingers. Reads .ply/.obj, writes .ply/.obj.
//   build: ./build_instant_meshes.sh   (bootstraps classic TBB 2020 + compiles the GUI-free core)
//   run:   instant_meshes_batch -i in.ply -o out.obj -f 60000 [-S smooth=2] [-D] [-d] [-b] [-c deg]
#include "batch.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

// field.cpp's Optimizer references this global (normally defined in the GUI main.cpp we exclude).
int nprocs = -1;   // -1 => tbb::task_scheduler_init::automatic

int main(int argc, char** argv) {
    std::string input, output;
    int rosy = 4, posy = 4, face_count = -1, vertex_count = -1;
    int smooth_iter = 2, knn_points = 10;
    float scale = -1, crease_angle = -1;
    bool extrinsic = true, align_to_boundaries = false, deterministic = false, dominant = false;
    // Curvature-adaptive sizing (default off => uniform path, cannot regress).
    // Precedence: -A <float> CLI flag, else IM_ADAPTIVITY env, else 0.
    float adaptivity = 0.0f;
    if (const char* env = getenv("IM_ADAPTIVITY")) adaptivity = (float) atof(env);

    for (int i = 1; i < argc; i++) {
        const char* a = argv[i];
        auto next = [&](){ return (i+1 < argc) ? argv[++i] : (const char*)""; };
        if      (!strcmp(a,"-i")) input = next();
        else if (!strcmp(a,"-o")) output = next();
        else if (!strcmp(a,"-f")) face_count = atoi(next());
        else if (!strcmp(a,"-v")) vertex_count = atoi(next());
        else if (!strcmp(a,"-s")) scale = atof(next());
        else if (!strcmp(a,"-S")) smooth_iter = atoi(next());
        else if (!strcmp(a,"-c")) crease_angle = atof(next());     // crease angle in degrees
        else if (!strcmp(a,"-k")) knn_points = atoi(next());
        else if (!strcmp(a,"-r")) rosy = atoi(next());             // 2/4/6 rotational symmetry
        else if (!strcmp(a,"-p")) posy = atoi(next());             // 3/4 positional symmetry
        else if (!strcmp(a,"-b")) align_to_boundaries = true;
        else if (!strcmp(a,"-D")) deterministic = true;
        else if (!strcmp(a,"-d")) dominant = true;                 // quad-DOMINANT (mixed tri/quad) vs pure quad
        else if (!strcmp(a,"-I")) extrinsic = false;               // intrinsic smoothing
        else if (!strcmp(a,"-A")) adaptivity = atof(next());        // curvature adaptivity [0,1] (0=uniform)
        else { fprintf(stderr, "unknown arg %s\n", a); return 1; }
    }
    if (input.empty() || output.empty()) {
        fprintf(stderr, "usage: instant_meshes_batch -i in.ply -o out.obj (-f faces | -v verts) "
                        "[-S smooth=2] [-c creaseDeg] [-d dominant] [-D deterministic] [-b boundaries] "
                        "[-A adaptivity=0] (or env IM_ADAPTIVITY; 0=uniform, ~0.5-1.0=curvature-adaptive)\n");
        return 1;
    }
    fprintf(stderr, "[im] in=%s out=%s rosy=%d posy=%d faces=%d verts=%d smooth=%d crease=%g "
                    "pure_quad=%d deterministic=%d boundaries=%d adaptivity=%g\n",
            input.c_str(), output.c_str(), rosy, posy, face_count, vertex_count, smooth_iter,
            crease_angle, !dominant, deterministic, align_to_boundaries, adaptivity);

    try {
        batch_process(input, output, rosy, posy, scale, face_count, vertex_count, crease_angle,
                      extrinsic, align_to_boundaries, smooth_iter, knn_points,
                      /*pure_quad=*/!dominant, deterministic, adaptivity);
    } catch (const std::exception& e) {
        fprintf(stderr, "[im] FAILED: %s\n", e.what());
        return 1;
    }
    fprintf(stderr, "[im] wrote %s\n", output.c_str());
    return 0;
}
