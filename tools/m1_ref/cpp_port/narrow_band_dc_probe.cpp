// Native diagnostic implementation of Pixal3D/CuMesh's remesh_narrow_band_dc.
//
// This is intentionally a probe, not a delivery-path replacement: it owns a compact native CUDA
// BVH so the narrow-band UDF can be checked without Python, Torch, or the fragile CuBVH extension.
//
// Usage: narrow_band_dc_probe input.glb output.glb [resolution=1024] [band=1]
#include "glb_reader.hpp"
#include "glb_writer.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

struct Coord { int x, y, z; };

// Coordinates are in [0, 1024], so 11 bits per component is sufficient for both voxel and
// grid-vertex maps.  Keeping this representation native makes the sparse topology deterministic.
uint64_t key(int x, int y, int z) {
    return (uint64_t)(uint32_t)x | ((uint64_t)(uint32_t)y << 11) | ((uint64_t)(uint32_t)z << 22);
}

void cuda_ok(cudaError_t e, const char* where) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", where, cudaGetErrorString(e));
        std::exit(1);
    }
}

struct NativeTri { float p[9]; };
struct NativeNode { float mn[3], mx[3]; int left, right, begin, count; };

__device__ inline float dot3(const float* a, const float* b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
__device__ inline float point_aabb_sq(const float* p, const NativeNode& n) {
    float d2 = 0.f;
    for (int k = 0; k < 3; ++k) { float d = p[k] < n.mn[k] ? n.mn[k]-p[k] : (p[k] > n.mx[k] ? p[k]-n.mx[k] : 0.f); d2 += d*d; }
    return d2;
}

// Christer Ericson, Real-Time Collision Detection, point-to-triangle squared distance.  Scalar
// math deliberately avoids Eigen in CUDA device code (the standalone CuBVH route faulted under
// CUDA 13.3 even on a cube, despite correct input pointers).
__device__ inline float point_tri_sq(const float* p, const NativeTri& t) {
    const float *a=t.p, *b=t.p+3, *c=t.p+6;
    float ab[3]={b[0]-a[0],b[1]-a[1],b[2]-a[2]}, ac[3]={c[0]-a[0],c[1]-a[1],c[2]-a[2]}, ap[3]={p[0]-a[0],p[1]-a[1],p[2]-a[2]};
    float d1=dot3(ab,ap), d2=dot3(ac,ap);
    if (d1<=0.f && d2<=0.f) return dot3(ap,ap);
    float bp[3]={p[0]-b[0],p[1]-b[1],p[2]-b[2]}; float d3=dot3(ab,bp), d4=dot3(ac,bp);
    if (d3>=0.f && d4<=d3) return dot3(bp,bp);
    float vc=d1*d4-d3*d2;
    if (vc<=0.f && d1>=0.f && d3<=0.f) { float v=d1/(d1-d3), q[3]={a[0]+v*ab[0],a[1]+v*ab[1],a[2]+v*ab[2]}; float x[3]={p[0]-q[0],p[1]-q[1],p[2]-q[2]}; return dot3(x,x); }
    float cp[3]={p[0]-c[0],p[1]-c[1],p[2]-c[2]}; float d5=dot3(ab,cp), d6=dot3(ac,cp);
    if (d6>=0.f && d5<=d6) return dot3(cp,cp);
    float vb=d5*d2-d1*d6;
    if (vb<=0.f && d2>=0.f && d6<=0.f) { float w=d2/(d2-d6), q[3]={a[0]+w*ac[0],a[1]+w*ac[1],a[2]+w*ac[2]}; float x[3]={p[0]-q[0],p[1]-q[1],p[2]-q[2]}; return dot3(x,x); }
    float va=d3*d6-d5*d4;
    if (va<=0.f && (d4-d3)>=0.f && (d5-d6)>=0.f) { float w=(d4-d3)/((d4-d3)+(d5-d6)); float q[3]={b[0]+w*(c[0]-b[0]),b[1]+w*(c[1]-b[1]),b[2]+w*(c[2]-b[2])}; float x[3]={p[0]-q[0],p[1]-q[1],p[2]-q[2]}; return dot3(x,x); }
    float denom=1.f/(va+vb+vc), v=vb*denom, w=vc*denom;
    float q[3]={a[0]+ab[0]*v+ac[0]*w,a[1]+ab[1]*v+ac[1]*w,a[2]+ab[2]*v+ac[2]*w}; float x[3]={p[0]-q[0],p[1]-q[1],p[2]-q[2]}; return dot3(x,x);
}

__global__ void native_bvh_udf(const float* positions, int n, const NativeNode* nodes, const NativeTri* tris, float* out) {
    int i=(int)(blockIdx.x*blockDim.x+threadIdx.x); if (i>=n) return;
    const float* p=positions+(size_t)i*3; float best=1e6f; int stack[64], sp=0; stack[sp++]=0;
    while (sp) {
        const NativeNode& node=nodes[stack[--sp]];
        if (point_aabb_sq(p,node)>best) continue;
        if (node.count) { for (int j=0;j<node.count;++j) { float d=point_tri_sq(p,tris[node.begin+j]); if(d<best) best=d; } }
        else { if (sp<=62) { stack[sp++]=node.left; stack[sp++]=node.right; } }
    }
    out[i]=sqrtf(best);
}

class Udf {
public:
    explicit Udf(const glb::Mesh& mesh) {
        triangles_.resize(mesh.faces.size() / 3);
        for (size_t i = 0; i < triangles_.size(); ++i) {
            const int64_t ia = mesh.faces[i * 3], ib = mesh.faces[i * 3 + 1], ic = mesh.faces[i * 3 + 2];
            if (ia < 0 || ib < 0 || ic < 0 || (size_t)ia >= mesh.verts.size() / 3 ||
                (size_t)ib >= mesh.verts.size() / 3 || (size_t)ic >= mesh.verts.size() / 3) {
                throw std::runtime_error("input GLB has an invalid face index");
            }
            for (int k=0;k<3;++k) { triangles_[i].p[k]=mesh.verts[(size_t)ia*3+k]; triangles_[i].p[3+k]=mesh.verts[(size_t)ib*3+k]; triangles_[i].p[6+k]=mesh.verts[(size_t)ic*3+k]; }
        }
        std::fprintf(stderr, "[narrow-band] building native BVH for %zu triangles (CPU build, GPU queries)\n", triangles_.size());
        build_node(0, (int)triangles_.size());
        cuda_ok(cudaMalloc(&d_tris_, triangles_.size()*sizeof(NativeTri)), "cudaMalloc triangles");
        cuda_ok(cudaMalloc(&d_nodes_, nodes_.size()*sizeof(NativeNode)), "cudaMalloc nodes");
        cuda_ok(cudaMemcpy(d_tris_,triangles_.data(),triangles_.size()*sizeof(NativeTri),cudaMemcpyHostToDevice), "cudaMemcpy triangles");
        cuda_ok(cudaMemcpy(d_nodes_,nodes_.data(),nodes_.size()*sizeof(NativeNode),cudaMemcpyHostToDevice), "cudaMemcpy nodes");
    }
    ~Udf() { if(d_tris_) cudaFree(d_tris_); if(d_nodes_) cudaFree(d_nodes_); }

    std::vector<float> distances(const std::vector<Coord>& coords, int resolution, float scale, bool cell_centres) const {
        const size_t n = coords.size();
        std::vector<float> host_pos(n * 3), out(n);
        for (size_t i = 0; i < n; ++i) {
            const float o = cell_centres ? .5f : 0.f;
            host_pos[i * 3] = (((float)coords[i].x + o) / resolution - .5f) * scale;
            host_pos[i * 3 + 1] = (((float)coords[i].y + o) / resolution - .5f) * scale;
            host_pos[i * 3 + 2] = (((float)coords[i].z + o) / resolution - .5f) * scale;
        }
        float *d_pos=nullptr,*d_dist=nullptr;
        cuda_ok(cudaMalloc(&d_pos,n*3*sizeof(float)), "cudaMalloc positions"); cuda_ok(cudaMalloc(&d_dist,n*sizeof(float)), "cudaMalloc distances");
        cuda_ok(cudaMemcpy(d_pos,host_pos.data(),n*3*sizeof(float),cudaMemcpyHostToDevice), "cudaMemcpy positions");
        native_bvh_udf<<<(unsigned)((n+127)/128),128>>>(d_pos,(int)n,d_nodes_,d_tris_,d_dist);
        cuda_ok(cudaGetLastError(), "native BVH UDF launch"); cuda_ok(cudaDeviceSynchronize(), "native BVH UDF");
        cuda_ok(cudaMemcpy(out.data(),d_dist,n*sizeof(float),cudaMemcpyDeviceToHost), "cudaMemcpy distances"); cudaFree(d_pos); cudaFree(d_dist);
        return out;
    }

private:
    int build_node(int begin, int end) {
        const int idx=(int)nodes_.size(); nodes_.push_back({}); NativeNode& n=nodes_.back();
        for(int k=0;k<3;++k) { n.mn[k]=1e30f; n.mx[k]=-1e30f; }
        for(int i=begin;i<end;++i) for(int v=0;v<3;++v) for(int k=0;k<3;++k) { float x=triangles_[i].p[v*3+k]; n.mn[k]=std::min(n.mn[k],x); n.mx[k]=std::max(n.mx[k],x); }
        const int count=end-begin; if(count<=8) { n.begin=begin; n.count=count; n.left=n.right=-1; return idx; }
        float cmn[3]={1e30f,1e30f,1e30f},cmx[3]={-1e30f,-1e30f,-1e30f}; for(int i=begin;i<end;++i) for(int k=0;k<3;++k) { float c=(triangles_[i].p[k]+triangles_[i].p[3+k]+triangles_[i].p[6+k])/3.f; cmn[k]=std::min(cmn[k],c); cmx[k]=std::max(cmx[k],c); }
        int axis=(cmx[1]-cmn[1]>cmx[0]-cmn[0])?1:0; if(cmx[2]-cmn[2]>cmx[axis]-cmn[axis]) axis=2; const int mid=(begin+end)/2;
        std::nth_element(triangles_.begin()+begin,triangles_.begin()+mid,triangles_.begin()+end,[axis](const NativeTri&a,const NativeTri&b){return (a.p[axis]+a.p[3+axis]+a.p[6+axis])<(b.p[axis]+b.p[3+axis]+b.p[6+axis]);});
        const int l=build_node(begin,mid), r=build_node(mid,end); nodes_[idx].left=l; nodes_[idx].right=r; nodes_[idx].begin=0; nodes_[idx].count=0; return idx;
    }
    std::vector<NativeTri> triangles_; std::vector<NativeNode> nodes_; NativeTri* d_tris_=nullptr; NativeNode* d_nodes_=nullptr;
};

std::vector<Coord> all_cells(int r) {
    std::vector<Coord> out;
    out.reserve((size_t)r * r * r);
    for (int x = 0; x < r; ++x) for (int y = 0; y < r; ++y) for (int z = 0; z < r; ++z) out.push_back({x, y, z});
    return out;
}

void compact_mesh(std::vector<float>& verts, std::vector<int64_t>& faces) {
    std::vector<int32_t> remap(verts.size() / 3, -1);
    std::vector<float> compact;
    compact.reserve(verts.size());
    for (int64_t& f : faces) {
        int32_t& r = remap[(size_t)f];
        if (r < 0) {
            r = (int32_t)(compact.size() / 3);
            compact.insert(compact.end(), verts.begin() + (size_t)f * 3, verts.begin() + (size_t)f * 3 + 3);
        }
        f = r;
    }
    verts.swap(compact);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 5) {
        std::fprintf(stderr, "usage: %s input.glb output.glb [resolution=1024] [band=1]\n", argv[0]);
        return 2;
    }
    const int resolution = argc >= 4 ? std::atoi(argv[3]) : 1024;
    const float band = argc == 5 ? std::strtof(argv[4], nullptr) : 1.f;
    if (resolution < 32 || (resolution & (resolution - 1)) != 0 || band <= 0) {
        std::fprintf(stderr, "resolution must be a power of two >= 32 and band must be positive\n");
        return 2;
    }
    glb::Mesh in;
    if (!glb::read_glb(argv[1], in)) return 1;
    try {
        cuda_ok(cudaSetDevice(0), "cudaSetDevice(0)");
        Udf udf(in);
        const float scale = (float)(resolution + 3 * band) / resolution;  // Pixal3D postprocess.py exactly
        const float eps = band * scale / resolution;
        int r = 32;
        std::vector<Coord> cells = all_cells(r);
        while (true) {
            std::fprintf(stderr, "[narrow-band] querying %zu cell centres at %d^3\n", cells.size(), r);
            std::vector<float> d = udf.distances(cells, r, scale, /*cell_centres=*/true);
            std::vector<Coord> kept;
            kept.reserve(cells.size() / 2);
            const float limit = .87f * scale / r;
            for (size_t i = 0; i < cells.size(); ++i) if (std::fabs(d[i] - eps) < limit) kept.push_back(cells[i]);
            cells.swap(kept);
            std::fprintf(stderr, "[narrow-band] kept %zu active cells\n", cells.size());
            if (r == resolution) break;
            r *= 2;
            std::vector<Coord> children;
            children.reserve(cells.size() * 8);
            for (const Coord& c : cells) for (int dx = 0; dx <= 1; ++dx) for (int dy = 0; dy <= 1; ++dy) for (int dz = 0; dz <= 1; ++dz)
                children.push_back({c.x * 2 + dx, c.y * 2 + dy, c.z * 2 + dz});
            cells.swap(children);
        }
        if (cells.empty()) throw std::runtime_error("narrow band unexpectedly contains no cells");

        std::unordered_map<uint64_t, int> cell_index;
        cell_index.reserve(cells.size() * 2);
        std::unordered_set<uint64_t> vertex_keys;
        vertex_keys.reserve(cells.size() * 4);
        for (size_t i = 0; i < cells.size(); ++i) {
            cell_index.emplace(key(cells[i].x, cells[i].y, cells[i].z), (int)i);
            for (int dx = 0; dx <= 1; ++dx) for (int dy = 0; dy <= 1; ++dy) for (int dz = 0; dz <= 1; ++dz)
                vertex_keys.insert(key(cells[i].x + dx, cells[i].y + dy, cells[i].z + dz));
        }
        std::vector<Coord> grid_vertices;
        grid_vertices.reserve(vertex_keys.size());
        std::unordered_map<uint64_t, int> value_index;
        value_index.reserve(vertex_keys.size() * 2);
        for (uint64_t k : vertex_keys) {
            Coord c{(int)(k & 2047), (int)((k >> 11) & 2047), (int)((k >> 22) & 2047)};
            value_index.emplace(k, (int)grid_vertices.size());
            grid_vertices.push_back(c);
        }
        std::fprintf(stderr, "[narrow-band] querying %zu deduplicated grid vertices\n", grid_vertices.size());
        std::vector<float> values = udf.distances(grid_vertices, resolution, scale, /*cell_centres=*/false);
        for (float& d : values) d -= eps;

        std::vector<float> dual(cells.size() * 3);
        std::vector<int8_t> crossings(cells.size() * 3, 0);
        auto val = [&](int x, int y, int z) { return values[(size_t)value_index.at(key(x, y, z))]; };
        for (size_t i = 0; i < cells.size(); ++i) {
            const Coord c = cells[i]; float sum[3] = {0, 0, 0}; int count = 0;
            auto edge = [&](int axis, int x, int y, int z, bool canonical) {
                int x2 = x + (axis == 0), y2 = y + (axis == 1), z2 = z + (axis == 2);
                const float a = val(x, y, z), b = val(x2, y2, z2);
                if ((a < 0) == (b < 0)) return;
                const float t = -a / (b - a);
                sum[0] += x + (axis == 0 ? t : 0); sum[1] += y + (axis == 1 ? t : 0); sum[2] += z + (axis == 2 ? t : 0); ++count;
                if (canonical) crossings[i * 3 + axis] = a < 0 ? 1 : -1;
            };
            for (int u = 0; u <= 1; ++u) for (int v = 0; v <= 1; ++v) edge(0, c.x, c.y + u, c.z + v, u == 1 && v == 1);
            for (int u = 0; u <= 1; ++u) for (int v = 0; v <= 1; ++v) edge(1, c.x + u, c.y, c.z + v, u == 1 && v == 1);
            for (int u = 0; u <= 1; ++u) for (int v = 0; v <= 1; ++v) edge(2, c.x + u, c.y + v, c.z, u == 1 && v == 1);
            dual[i * 3] = count ? sum[0] / count : c.x + .5f;
            dual[i * 3 + 1] = count ? sum[1] / count : c.y + .5f;
            dual[i * 3 + 2] = count ? sum[2] / count : c.z + .5f;
        }

        std::vector<float> out_verts(cells.size() * 3);
        for (size_t i = 0; i < cells.size(); ++i) for (int d = 0; d < 3; ++d)
            out_verts[i * 3 + d] = (dual[i * 3 + d] / resolution - .5f) * scale;
        const int OFF[3][4][3] = {{{0,0,0},{0,0,1},{0,1,1},{0,1,0}}, {{0,0,0},{1,0,0},{1,0,1},{0,0,1}}, {{0,0,0},{0,1,0},{1,1,0},{1,0,0}}};
        const int S1N[6] = {0,1,2,0,2,3}, S1P[6] = {0,2,1,0,3,2}, S2N[6] = {0,1,3,3,1,2}, S2P[6] = {0,3,1,3,2,1};
        std::vector<int64_t> out_faces;
        out_faces.reserve(cells.size() * 6);
        auto align = [&](const int q[4], const int s[6]) {
            auto n = [&](int a, int b, int c, float out[3]) {
                const float *p = &out_verts[(size_t)q[a] * 3], *u = &out_verts[(size_t)q[b] * 3], *v = &out_verts[(size_t)q[c] * 3];
                float e0[3] = {u[0]-p[0],u[1]-p[1],u[2]-p[2]}, e1[3] = {v[0]-p[0],v[1]-p[1],v[2]-p[2]};
                out[0]=e0[1]*e1[2]-e0[2]*e1[1]; out[1]=e0[2]*e1[0]-e0[0]*e1[2]; out[2]=e0[0]*e1[1]-e0[1]*e1[0];
            };
            float a[3], b[3]; n(s[0],s[1],s[2],a); n(s[3],s[4],s[5],b); return std::fabs(a[0]*b[0]+a[1]*b[1]+a[2]*b[2]);
        };
        for (size_t i = 0; i < cells.size(); ++i) for (int e = 0; e < 3; ++e) if (crossings[i * 3 + e]) {
            int q[4]; bool ok = true;
            for (int j = 0; j < 4; ++j) {
                const Coord& c = cells[i]; auto it = cell_index.find(key(c.x + OFF[e][j][0], c.y + OFF[e][j][1], c.z + OFF[e][j][2]));
                if (it == cell_index.end()) { ok = false; break; } q[j] = it->second;
            }
            if (!ok) continue;
            const bool pos = crossings[i * 3 + e] > 0;
            const int* a = pos ? S1P : S1N; const int* b = pos ? S2P : S2N;
            const int* s = align(q, a) > align(q, b) ? a : b;
            for (int j = 0; j < 6; ++j) out_faces.push_back(q[s[j]]);
        }
        compact_mesh(out_verts, out_faces);
        std::fprintf(stderr, "[narrow-band] result V=%zu F=%zu (raw V=%zu F=%zu)\n", out_verts.size()/3, out_faces.size()/3, in.verts.size()/3, in.faces.size()/3);
        if (!glb::write_glb(argv[2], out_verts, out_faces)) return 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "narrow-band probe failed: %s\n", e.what());
        return 1;
    }
    return 0;
}
