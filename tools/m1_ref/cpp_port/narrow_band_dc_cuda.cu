// Native implementation of Pixal3D/CuMesh's remesh_narrow_band_dc.
//
// Moved verbatim out of narrow_band_dc_probe.cpp (2026-07-25) so image_to_rig's --dc-remesh mode
// and the standalone probe run the SAME code. It owns a compact native CUDA BVH so the narrow-band
// UDF can be computed without Python, Torch, or the fragile CuBVH extension.
#include "narrow_band_dc.hpp"
#include "cancel_hook.hpp"   // cooperative cancellation between ladder levels

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

struct Coord { int x, y, z; };

// 11 bits per component, so coordinates must fit [0, 2047] -- grid vertices reach `resolution`
// INCLUSIVE (cell corner c+1 of the last cell), hence the hard KEY_MAX_COORD ceiling below.
// 1024 and 1536 both fit; 2048 does NOT (it would alias onto y=0 of the next component).
// Keeping this representation native makes the sparse topology deterministic.
constexpr int KEY_MAX_COORD = 2047;
uint64_t key(int x, int y, int z) {
    return (uint64_t)(uint32_t)x | ((uint64_t)(uint32_t)y << 11) | ((uint64_t)(uint32_t)z << 22);
}

void cuda_ok(cudaError_t e, const char* where) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(e));
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
    Udf(const std::vector<float>& verts, const std::vector<int64_t>& faces, bool verbose) {
        triangles_.resize(faces.size() / 3);
        for (size_t i = 0; i < triangles_.size(); ++i) {
            const int64_t ia = faces[i * 3], ib = faces[i * 3 + 1], ic = faces[i * 3 + 2];
            if (ia < 0 || ib < 0 || ic < 0 || (size_t)ia >= verts.size() / 3 ||
                (size_t)ib >= verts.size() / 3 || (size_t)ic >= verts.size() / 3) {
                throw std::runtime_error("input mesh has an invalid face index");
            }
            for (int k=0;k<3;++k) { triangles_[i].p[k]=verts[(size_t)ia*3+k]; triangles_[i].p[3+k]=verts[(size_t)ib*3+k]; triangles_[i].p[6+k]=verts[(size_t)ic*3+k]; }
        }
        if (verbose) std::fprintf(stderr, "[narrow-band] building native BVH for %zu triangles (CPU build, GPU queries)\n", triangles_.size());
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

}  // namespace

namespace nbdc {

bool remesh(const std::vector<float>& in_verts, const std::vector<int64_t>& in_faces,
            int resolution, float band,
            std::vector<float>& out_verts_arg, std::vector<int64_t>& out_faces_arg,
            bool verbose, const solidfield::SolidField* interior) {
    // The refinement ladder below starts on a coarse grid and DOUBLES until it reaches
    // `resolution`, so the only structural requirement is that `resolution` be reachable by
    // doubling from a grid that is (a) at least 32 and (b) small enough to enumerate densely.
    // That is NOT the same as "power of two" -- the old guard was over-strict and locked out 1536.
    //
    // Peel factors of two off `resolution` while the coarse grid stays >= 32.  For every power of
    // two >= 32 this lands on exactly 32, so `ladder_start`/`levels` reproduce the old hardcoded
    // `int r = 32` bit-for-bit (1024 -> 32 with 5 doublings).  1536 lands on 48 (1536, 768, 384,
    // 192, 96, 48), likewise 5 doublings back up.  `ladder_start << levels == resolution` holds by
    // construction, so the loop always terminates exactly on `resolution`.
    int ladder_start = resolution, levels = 0;
    while ((ladder_start & 1) == 0 && (ladder_start >> 1) >= 32) { ladder_start >>= 1; ++levels; }
    // The coarsest level is enumerated densely (ladder_start^3 cells), so a `resolution` with too
    // few factors of two (e.g. an odd one) is refused rather than allowed to eat host RAM.  Every
    // multiple of 16 below 2048 -- which is all image_to_rig will pass -- peels to <= 127.
    constexpr int MAX_LADDER_START = 128;
    if (resolution < 32 || resolution > KEY_MAX_COORD || band <= 0 || ladder_start > MAX_LADDER_START) {
        std::fprintf(stderr,
                     "[narrow-band] resolution must be in [32, %d] and reach a coarse grid of "
                     "<= %d by halving (got %d -> %d), and band must be positive\n",
                     KEY_MAX_COORD, MAX_LADDER_START, resolution, ladder_start);
        return false;
    }
    if (in_verts.empty() || in_faces.empty()) {
        std::fprintf(stderr, "[narrow-band] empty input mesh\n");
        return false;
    }
    try {
        Udf udf(in_verts, in_faces, verbose);
        // Both derived quantities are ratios in `resolution` with no power-of-two dependence:
        // `scale` widens the [-0.5, 0.5] cube by 1.5 band-widths per side and `eps` is the band
        // offset expressed in that widened frame.  They stay exact at 1536.
        const float scale = (float)(resolution + 3 * band) / resolution;  // Pixal3D postprocess.py exactly
        const float eps = band * scale / resolution;
        // Only meaningful when the field is signed; 0 keeps the historical cell set exactly.
        // 4 is measured, not guessed: at 2 the dropped-quad counter below still fires (miku 2660
        // boundary edges, gilly 536) because the interior flag's boundary can sit a couple of voxels
        // off the eps level set; at 4 it reaches zero on both heroes.
        const int grow = (interior && !interior->empty())
                       ? (std::getenv("NBDC_BAND_GROW") ? atoi(std::getenv("NBDC_BAND_GROW")) : 4) : 0;
        int r = ladder_start;
        std::vector<Coord> cells = all_cells(r);
        while (true) {
            // CANCELLATION POINT — one per ladder level. The quantum is a whole level's UDF query,
            // and the levels grow 8x in cells as the grid doubles, so the LAST level dominates the
            // worst case. Everything live here is a std::vector or the RAII BVH, so the throw is
            // safe; it is re-thrown past the catch below rather than reported as a DC failure.
            cancelhook::check();
            if (verbose) std::fprintf(stderr, "[narrow-band] querying %zu cell centres at %d^3\n", cells.size(), r);
            std::vector<float> d = udf.distances(cells, r, scale, /*cell_centres=*/true);
            std::vector<Coord> kept;
            kept.reserve(cells.size() / 2);
            const float limit = .87f * scale / r;
            for (size_t i = 0; i < cells.size(); ++i) if (std::fabs(d[i] - eps) < limit) kept.push_back(cells[i]);
            cells.swap(kept);
            if (verbose) std::fprintf(stderr, "[narrow-band] kept %zu active cells\n", cells.size());
            if (r >= resolution) break;   // == by construction; >= so a bad ladder cannot run away
            r *= 2;
            std::vector<Coord> children;
            children.reserve(cells.size() * 8);
            for (const Coord& c : cells) for (int dx = 0; dx <= 1; ++dx) for (int dy = 0; dy <= 1; ++dy) for (int dz = 0; dz <= 1; ++dz)
                children.push_back({c.x * 2 + dx, c.y * 2 + dy, c.z * 2 + dz});
            cells.swap(children);
        }
        if (cells.empty()) throw std::runtime_error("narrow band unexpectedly contains no cells");

        // BAND EXPANSION.  A quad is emitted only when all four cells around its grid edge are in
        // the band; otherwise it is silently dropped, which tears a boundary edge.  With the
        // UNSIGNED field that never fires -- every sign change sits within 0.87 voxels of the eps
        // level set, so its four cells are in the band by construction, and the delivered mesh has
        // boundary == 0.  SIGNING moves some crossings off that level set (the interior flag's
        // boundary is where they land), so the band has to be grown to cover them.  Growing it
        // cannot change the unsigned result: the added cells contain no sign change, so they emit
        // nothing and compact_mesh drops their unreferenced dual vertices.
        if (grow > 0) {
            std::unordered_set<uint64_t> have;
            have.reserve(cells.size() * 2);
            for (const Coord& c : cells) have.insert(key(c.x, c.y, c.z));
            for (int g = 0; g < grow; ++g) {
                std::vector<Coord> added;
                for (const Coord& c : cells) {
                    const int nb[6][3] = {{c.x + 1, c.y, c.z}, {c.x - 1, c.y, c.z},
                                          {c.x, c.y + 1, c.z}, {c.x, c.y - 1, c.z},
                                          {c.x, c.y, c.z + 1}, {c.x, c.y, c.z - 1}};
                    for (const auto& q : nb) {
                        if (q[0] < 0 || q[1] < 0 || q[2] < 0) continue;
                        if (q[0] >= resolution || q[1] >= resolution || q[2] >= resolution) continue;
                        if (have.insert(key(q[0], q[1], q[2])).second) added.push_back({q[0], q[1], q[2]});
                    }
                }
                cells.insert(cells.end(), added.begin(), added.end());
            }
            if (verbose) std::fprintf(stderr, "[narrow-band] band grown by %d -> %zu cells\n", grow, cells.size());
        }

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
        cancelhook::check();   // before the single biggest query: every grid vertex of the band
        if (verbose) std::fprintf(stderr, "[narrow-band] querying %zu deduplicated grid vertices\n", grid_vertices.size());
        std::vector<float> values = udf.distances(grid_vertices, resolution, scale, /*cell_centres=*/false);
        for (float& d : values) d -= eps;

        // SIGN THE FIELD (opt-in; `interior == nullptr` leaves everything above and below exactly
        // as it was).  Only values that are still POSITIVE can flip, and a positive value means the
        // vertex is more than `eps` from the input surface -- so the flip never touches a crossing
        // that defines the outward wall.  What it removes is the deep-interior positive region, and
        // with it the inward wall that region's boundary was.
        //
        // The oracle is a VOXEL set closed with box morphology, so its boundary wanders about a
        // voxel either side of the true surface.  Two ways that hurts, and they are not symmetric:
        //   * flagging an OUTSIDE vertex interior pushes the outward wall out by a fraction of a
        //     voxel -- harmless, and with erode >= 1 the flag does not reach a positive vertex at
        //     all (a vertex less than eps outside is already negative and cannot flip);
        //   * MISSING an inside vertex leaves a positive island buried in the body, which contours
        //     into a stray closed sheet.  Measured raw: 8013 components and 106k boundary edges.
        // So the raw labels are median-filtered over the {f > 0} vertex graph, which erases isolated
        // misses without moving the large-scale boundary.  (A single vote per connected {f > 0}
        // region was tried first and is WRONG: gilly's interior and exterior positive regions are
        // joined through a hole in the O-Voxel surface, so one vote decides both and the whole
        // inward wall survives -- 25k flips instead of 900k.)
        if (interior && !interior->empty()) {
            const size_t nv = values.size();
            const int smooth = std::getenv("NBDC_SIGN_SMOOTH") ? atoi(std::getenv("NBDC_SIGN_SMOOTH")) : 3;
            std::vector<uint8_t> lab(nv, 0), pos(nv, 0);
            for (size_t i = 0; i < nv; ++i) {
                if (values[i] <= 0.f) continue;
                pos[i] = 1;
                const Coord& c = grid_vertices[i];
                lab[i] = interior->inside(c.x, c.y, c.z) ? 1 : 0;
            }
            // 6-neighbour adjacency among the positive vertices, materialised once (CSR).
            std::vector<int32_t> off(nv + 1, 0), adj;
            adj.reserve(nv * 3);
            for (size_t i = 0; i < nv; ++i) {
                off[i] = (int32_t)adj.size();
                if (!pos[i]) continue;
                const Coord& c = grid_vertices[i];
                const int nb[6][3] = {{c.x + 1, c.y, c.z}, {c.x - 1, c.y, c.z},
                                      {c.x, c.y + 1, c.z}, {c.x, c.y - 1, c.z},
                                      {c.x, c.y, c.z + 1}, {c.x, c.y, c.z - 1}};
                for (const auto& q : nb) {
                    if (q[0] < 0 || q[1] < 0 || q[2] < 0) continue;
                    if (q[0] > KEY_MAX_COORD || q[1] > KEY_MAX_COORD || q[2] > KEY_MAX_COORD) continue;
                    auto it = value_index.find(key(q[0], q[1], q[2]));
                    if (it != value_index.end() && pos[(size_t)it->second]) adj.push_back(it->second);
                }
            }
            off[nv] = (int32_t)adj.size();
            int64_t changed_total = 0;
            for (int pass = 0; pass < smooth; ++pass) {
                std::vector<uint8_t> next = lab;
                int64_t changed = 0;
                for (size_t i = 0; i < nv; ++i) {
                    if (!pos[i]) continue;
                    int in = lab[i] ? 1 : 0, tot = 1;
                    for (int32_t k = off[i]; k < off[i + 1]; ++k) { in += lab[(size_t)adj[k]] ? 1 : 0; ++tot; }
                    const uint8_t v = (uint8_t)(2 * in > tot ? 1 : 0);
                    if (v != lab[i]) ++changed;
                    next[i] = v;
                }
                lab.swap(next);
                changed_total += changed;
                if (!changed) break;
            }
            // DIAGNOSTIC (NBDC_SIGN_DEBUG=1): the connected {f>0} regions with their oracle vote.
            // This is what tells you whether a bad result is a mislabelled region or two regions
            // that should be separate having merged.
            if (std::getenv("NBDC_SIGN_DEBUG")) {
                std::vector<int32_t> parent(nv);
                for (size_t i = 0; i < nv; ++i) parent[i] = (int32_t)i;
                std::function<int32_t(int32_t)> find = [&](int32_t x) {
                    while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
                for (size_t i = 0; i < nv; ++i) if (pos[i])
                    for (int32_t k = off[i]; k < off[i + 1]; ++k) {
                        const int32_t ra = find((int32_t)i), rb = find(adj[k]); if (ra != rb) parent[ra] = rb; }
                std::unordered_map<int32_t, std::pair<int64_t,int64_t>> vote;
                for (size_t i = 0; i < nv; ++i) if (pos[i]) {
                    auto& t = vote[find((int32_t)i)];
                    const Coord& c = grid_vertices[i];
                    if (interior->inside(c.x, c.y, c.z)) ++t.first; else ++t.second; }
                std::vector<std::pair<int64_t,std::pair<int64_t,int64_t>>> big;
                for (const auto& kv : vote) big.push_back({kv.second.first + kv.second.second, kv.second});
                std::sort(big.begin(), big.end(), [](auto& a, auto& b){ return a.first > b.first; });
                std::fprintf(stderr, "[narrow-band] sign-debug: %zu positive regions; largest:\n", vote.size());
                for (size_t i = 0; i < big.size() && i < 8; ++i)
                    std::fprintf(stderr, "    n=%-9lld  interior-flagged=%-9lld outside-flagged=%-9lld -> %s\n",
                                 (long long)big[i].first, (long long)big[i].second.first,
                                 (long long)big[i].second.second,
                                 big[i].second.first > big[i].second.second ? "INTERIOR" : "exterior");
            }
            int64_t flipped = 0;
            for (size_t i = 0; i < nv; ++i) if (pos[i] && lab[i]) { values[i] = -values[i]; ++flipped; }
            if (verbose)
                std::fprintf(stderr,
                             "[narrow-band] signed field: flipped %lld of %zu grid vertices to "
                             "interior-negative (%lld labels changed by %d median passes)\n",
                             (long long)flipped, nv, (long long)changed_total, smooth);
        }

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
        int64_t dropped_quads = 0;
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
            if (!ok) { ++dropped_quads; continue; }
            const bool pos = crossings[i * 3 + e] > 0;
            const int* a = pos ? S1P : S1N; const int* b = pos ? S2P : S2N;
            const int* s = align(q, a) > align(q, b) ? a : b;
            for (int j = 0; j < 6; ++j) out_faces.push_back(q[s[j]]);
        }
        compact_mesh(out_verts, out_faces);
        if (verbose && dropped_quads)
            std::fprintf(stderr, "[narrow-band] *** %lld quads dropped (a crossing whose 4 cells are "
                                 "not all in the band) -- each one tears boundary edges ***\n",
                         (long long)dropped_quads);
        if (verbose) std::fprintf(stderr, "[narrow-band] result V=%zu F=%zu (raw V=%zu F=%zu)\n",
                                  out_verts.size()/3, out_faces.size()/3, in_verts.size()/3, in_faces.size()/3);
        out_verts_arg.swap(out_verts);
        out_faces_arg.swap(out_faces);
        return true;
    } catch (const cancelhook::Cancelled&) {
        throw;   // a cancel is not a DC failure: let it reach the stage boundary
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[narrow-band] failed: %s\n", e.what());
        return false;
    }
}

}  // namespace nbdc
