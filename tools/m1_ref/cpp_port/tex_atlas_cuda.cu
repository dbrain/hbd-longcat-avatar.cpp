#include "tex_atlas_cuda.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>

namespace texatlas_cuda {
namespace {

__global__ void raster_kernel(const float* verts, const float* nrms,
                              const uint32_t* faces, const float* uv,
                              int tri_count, int width, int height, int ss,
                              float* out_pos, float* out_nrm, uint8_t* out_mask) {
    const int t = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (t >= tri_count) return;
    const uint32_t ia = faces[t * 3], ib = faces[t * 3 + 1], ic = faces[t * 3 + 2];
    const float ax = uv[ia * 2], ay = uv[ia * 2 + 1];
    const float bx = uv[ib * 2], by = uv[ib * 2 + 1];
    const float cx = uv[ic * 2], cy = uv[ic * 2 + 1];
    const float area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    if (fabsf(area) < 1e-9f) return;
    const float inv = 1.f / area;
    const int x0 = max(0, (int)floorf(fminf(ax, fminf(bx, cx))));
    const int x1 = min(width - 1, (int)ceilf(fmaxf(ax, fmaxf(bx, cx))));
    const int y0 = max(0, (int)floorf(fminf(ay, fminf(by, cy))));
    const int y1 = min(height - 1, (int)ceilf(fmaxf(ay, fmaxf(by, cy))));
    for (int y = y0; y <= y1; ++y) for (int x = x0; x <= x1; ++x) {
        float w0sum = 0.f, w1sum = 0.f, w2sum = 0.f; int hits = 0;
        for (int sy = 0; sy < ss; ++sy) for (int sx = 0; sx < ss; ++sx) {
            const float px = x + ((float)sx + .5f) / (float)ss;
            const float py = y + ((float)sy + .5f) / (float)ss;
            const float w0 = ((bx-px)*(cy-py) - (by-py)*(cx-px)) * inv;
            const float w1 = ((cx-px)*(ay-py) - (cy-py)*(ax-px)) * inv;
            const float w2 = 1.f - w0 - w1;
            if (w0 < 0.f || w1 < 0.f || w2 < 0.f) continue;
            w0sum += w0; w1sum += w1; w2sum += w2; ++hits;
        }
        if (!hits) continue;
        const float scale = 1.f / (float)hits;
        const float w0 = w0sum * scale, w1 = w1sum * scale, w2 = w2sum * scale;
        const size_t p = (size_t)y * (size_t)width + (size_t)x;
        // Packed charts have disjoint interiors.  At a shared edge either
        // triangle gives the same interpolated surface value, so no atomic
        // blend is necessary and this remains a one-thread-per-triangle pass.
        for (int d = 0; d < 3; ++d) {
            out_pos[p*3+d] = w0*verts[ia*3+d] + w1*verts[ib*3+d] + w2*verts[ic*3+d];
            out_nrm[p*3+d] = w0*nrms[ia*3+d] + w1*nrms[ib*3+d] + w2*nrms[ic*3+d];
        }
        out_mask[p] = 1;
    }
}

template <class T> struct Dev {
    T* p = nullptr;
    ~Dev() { if (p) cudaFree(p); }
};

static bool ck(cudaError_t e, const char* what, std::string* err) {
    if (e == cudaSuccess) return true;
    if (err) *err = std::string(what) + ": " + cudaGetErrorString(e);
    return false;
}
static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

} // namespace

bool raster_uv(const std::vector<float>& vertices, const std::vector<float>& normals,
               const std::vector<uint32_t>& faces, const std::vector<float>& uv_px,
               int width, int height, int supersample, std::vector<float>& position,
               std::vector<float>& normal, std::vector<uint8_t>& mask,
               RasterStats* stats, std::string* error) {
    if (width < 1 || height < 1 || supersample < 1 || vertices.size() != normals.size() ||
        vertices.size() % 3 || uv_px.size() != vertices.size() / 3 * 2 || faces.size() % 3) {
        if (error) *error = "invalid UV raster input";
        return false;
    }
    const size_t pixels = (size_t)width * (size_t)height;
    const size_t vbytes = vertices.size() * sizeof(float), fbytes = faces.size() * sizeof(uint32_t);
    const size_t ubytes = uv_px.size() * sizeof(float), pbytes = pixels * 3 * sizeof(float), mbytes = pixels;
    Dev<float> dv, dn, du, dp, do_n; Dev<uint32_t> df; Dev<uint8_t> dm;
    const double t0 = now();
    if (!ck(cudaMalloc(&dv.p, vbytes), "cudaMalloc verts", error) || !ck(cudaMalloc(&dn.p, vbytes), "cudaMalloc normals", error) ||
        !ck(cudaMalloc(&df.p, fbytes), "cudaMalloc faces", error) || !ck(cudaMalloc(&du.p, ubytes), "cudaMalloc uvs", error) ||
        !ck(cudaMalloc(&dp.p, pbytes), "cudaMalloc pos", error) || !ck(cudaMalloc(&do_n.p, pbytes), "cudaMalloc nrm", error) ||
        !ck(cudaMalloc(&dm.p, mbytes), "cudaMalloc mask", error)) return false;
    if (!ck(cudaMemcpy(dv.p, vertices.data(), vbytes, cudaMemcpyHostToDevice), "upload verts", error) ||
        !ck(cudaMemcpy(dn.p, normals.data(), vbytes, cudaMemcpyHostToDevice), "upload normals", error) ||
        !ck(cudaMemcpy(df.p, faces.data(), fbytes, cudaMemcpyHostToDevice), "upload faces", error) ||
        !ck(cudaMemcpy(du.p, uv_px.data(), ubytes, cudaMemcpyHostToDevice), "upload uvs", error) ||
        !ck(cudaMemset(dp.p, 0, pbytes), "clear pos", error) || !ck(cudaMemset(do_n.p, 0, pbytes), "clear nrm", error) ||
        !ck(cudaMemset(dm.p, 0, mbytes), "clear mask", error)) return false;
    const double t1 = now();
    const int tri_count = (int)(faces.size() / 3);
    raster_kernel<<<(tri_count + 127) / 128, 128>>>(dv.p, dn.p, df.p, du.p, tri_count, width, height, supersample, dp.p, do_n.p, dm.p);
    if (!ck(cudaGetLastError(), "raster kernel launch", error) || !ck(cudaDeviceSynchronize(), "raster kernel", error)) return false;
    const double t2 = now();
    position.resize(pixels * 3); normal.resize(pixels * 3); mask.resize(pixels);
    if (!ck(cudaMemcpy(position.data(), dp.p, pbytes, cudaMemcpyDeviceToHost), "download pos", error) ||
        !ck(cudaMemcpy(normal.data(), do_n.p, pbytes, cudaMemcpyDeviceToHost), "download nrm", error) ||
        !ck(cudaMemcpy(mask.data(), dm.p, mbytes, cudaMemcpyDeviceToHost), "download mask", error)) return false;
    if (stats) { stats->upload_seconds = t1 - t0; stats->kernel_seconds = t2 - t1; stats->download_seconds = now() - t2; }
    return true;
}

} // namespace texatlas_cuda
