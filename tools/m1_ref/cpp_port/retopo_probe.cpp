// retopo_probe — load an OBJ (tris and/or quads), triangulate, run xatlas ComputeCharts, report chart
// count + atlas util. The R1/R2 de-risk: does a QuadriFlow'd mesh unwrap into FEW charts (vs the 182k
// the non-manifold cumesh/QEM mesh produces)? CPU-only, no GPU.
//   build:  g++ -O2 -std=c++17 retopo_probe.cpp ../../../thirdparty/xatlas.cpp -o retopo_probe -lpthread
//   run:    ./retopo_probe mesh.obj [pyref]
#include "../../../thirdparty/xatlas.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: retopo_probe mesh.obj [pyref]\n"); return 1; }
    bool pyref = argc > 2 && !strcmp(argv[2], "pyref");
    FILE* f = fopen(argv[1], "r"); if (!f) { printf("open failed\n"); return 1; }
    std::vector<float> verts;
    std::vector<uint32_t> idx;
    int nquad = 0, ntri = 0;
    char line[512];
    while (fgets(line, sizeof line, f)) {
        if (line[0] == 'v' && line[1] == ' ') {
            float x, y, z; sscanf(line + 2, "%f %f %f", &x, &y, &z);
            verts.push_back(x); verts.push_back(y); verts.push_back(z);
        } else if (line[0] == 'f' && line[1] == ' ') {
            // parse up to 4 vertex indices (ignore /vt/vn); OBJ is 1-based
            int v[4], n = 0; char* p = line + 2;
            while (n < 4) { while (*p == ' ') p++; if (*p == 0 || *p == '\n') break;
                v[n++] = atoi(p); while (*p && *p != ' ' && *p != '\n') p++; }
            if (n == 4) { nquad++;
                idx.push_back(v[0]-1); idx.push_back(v[1]-1); idx.push_back(v[2]-1);
                idx.push_back(v[0]-1); idx.push_back(v[2]-1); idx.push_back(v[3]-1);
            } else if (n == 3) { ntri++;
                idx.push_back(v[0]-1); idx.push_back(v[1]-1); idx.push_back(v[2]-1);
            }
        }
    }
    fclose(f);
    uint32_t nv = (uint32_t)(verts.size() / 3), ntris = (uint32_t)(idx.size() / 3);
    printf("[probe] %s: %u verts, %d quads + %d tris -> %u triangles\n", argv[1], nv, nquad, ntri, ntris);

    xatlas::Atlas* atlas = xatlas::Create();
    xatlas::MeshDecl m;
    m.vertexCount = nv; m.vertexPositionData = verts.data(); m.vertexPositionStride = 12;
    m.indexCount = (uint32_t)idx.size(); m.indexData = idx.data(); m.indexFormat = xatlas::IndexFormat::UInt32;
    xatlas::AddMeshError e = xatlas::AddMesh(atlas, m);
    if (e != xatlas::AddMeshError::Success) { printf("AddMesh error: %s\n", xatlas::StringForEnum(e)); return 1; }

    xatlas::ChartOptions co;
    if (pyref) { // Python/cumesh xatlas defaults (matches the bake's ATL_PYREF_XATLAS)
        co.maxCost = 2.0f; co.normalDeviationWeight = 2.0f; co.normalSeamWeight = 4.0f;
        co.straightnessWeight = 6.0f; co.roundnessWeight = 0.01f; co.textureSeamWeight = 0.5f; co.maxIterations = 1;
    }
    xatlas::Generate(atlas, co);  // ComputeCharts + PackCharts → populates chartCount + atlas dims
    printf("[probe] xatlas%s: CHARTS=%u  atlas=%ux%u  util=%.1f%%\n",
           pyref ? " (pyref opts)" : " (default)", atlas->chartCount, atlas->width, atlas->height,
           100.0f * atlas->utilization[0]);
    xatlas::Destroy(atlas);
    return 0;
}
