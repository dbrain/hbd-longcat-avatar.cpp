// voxelizer.hpp — source-level port of o_voxel's flexible dual-grid conversion.
//
// Keep the QEF representation deliberately close to
// o-voxel/src/convert/flexible_dual_grid.cpp: source-space float32 vertices,
// float32 4x4 QEF accumulation, and the source constrained solve.  The learned
// encoder is sensitive to these otherwise-small arithmetic differences.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <unordered_map>
#include <vector>
// The frozen reference wheel was built from Eigen commit 21e4582d.  Keep that
// exact header-only dependency as a pinned submodule: Eigen's QR and vector
// normalization changed enough after the older ManifoldPlus copy to move QEF
// vertices and, consequently, the learned shape lattice.
#include "../../../thirdparty/eigen_reference/Eigen/Dense"

namespace vox {

struct VoxelOut {
    std::vector<int32_t> coords;       // [N,3]
    std::vector<float> dual;           // [N,3], local offset in [0,1]
    std::vector<int8_t> intersected;   // [N,3]
    int N = 0;
};

static inline int64_t cell_key(int32_t x, int32_t y, int32_t z) {
    auto m=[](int32_t v){ return (int64_t)(v & 0xFFFFF); };
    return (((m(x) << 20) | m(y)) << 20) | m(z);
}

// This is intentionally not a hand-expanded normal equation.  o_voxel stores
// and accumulates Matrix4f QEFs, including their homogeneous terms.
struct Cell {
    Eigen::Matrix4f qef = Eigen::Matrix4f::Zero();
    Eigen::Vector3f mean = Eigen::Vector3f::Zero();
    float count = 0.f;
};

template <typename T, typename U>
static inline U qef_lerp(const T& a, const T& b, const T& t, const U& va, const U& vb) {
    if (a == b) return va;
    const T alpha = (t - a) / (b - a);
    return (1 - alpha) * va + alpha * vb;
}

// Direct transcription of the source's final constrained solve, preserving
// float arithmetic, candidate order, strict bounds and QEF error evaluation.
static inline Eigen::Vector3f solve_qef(const Eigen::Matrix4f& qef,
                                        const Eigen::Vector3f& min_corner,
                                        const Eigen::Vector3f& max_corner) {
    Eigen::Matrix3f A = qef.topLeftCorner<3, 3>();
    Eigen::Vector3f b = -qef.block<3, 1>(0, 3);
    Eigen::Vector3f v = A.colPivHouseholderQr().solve(b);
    if (v.x() >= min_corner.x() && v.x() <= max_corner.x() &&
        v.y() >= min_corner.y() && v.y() <= max_corner.y() &&
        v.z() >= min_corner.z() && v.z() <= max_corner.z()) return v;

    float best = std::numeric_limits<float>::infinity();
    auto accept = [&](const Eigen::Vector4f& p) {
        if (p.x() < min_corner.x() || p.x() > max_corner.x() ||
            p.y() < min_corner.y() || p.y() > max_corner.y() ||
            p.z() < min_corner.z() || p.z() > max_corner.z()) return;
        const float err = p.transpose() * qef * p;
        if (err < best) { best = err; v << p[0], p[1], p[2]; }
    };

    for (int fixed = 0; fixed < 3; ++fixed) {
        const int a1 = (fixed + 1) % 3, a2 = (fixed + 2) % 3;
        Eigen::Matrix2f m;
        m << qef(a1,a1), qef(a1,a2), qef(a2,a1), qef(a2,a2);
        Eigen::Matrix2f B;
        B << qef(a1,fixed), qef(a1,3), qef(a2,fixed), qef(a2,3);
        const auto sol = m.colPivHouseholderQr();
        for (float bound : {min_corner[fixed], max_corner[fixed]}) {
            Eigen::Vector2f q(bound, 1.f), x = sol.solve(-B * q);
            Eigen::Vector4f p; p[fixed] = bound; p[a1] = x.x(); p[a2] = x.y(); p[3] = 1.f;
            accept(p);
        }
    }
    for (int free_axis = 0; free_axis < 3; ++free_axis) {
        const int a1 = (free_axis + 1) % 3, a2 = (free_axis + 2) % 3;
        const float a = qef(free_axis, free_axis);
        const Eigen::Vector3f b3(qef(free_axis,a1), qef(free_axis,a2), qef(free_axis,3));
        for (float x1 : {min_corner[a1], max_corner[a1]}) for (float x2 : {min_corner[a2], max_corner[a2]}) {
            const float x = -b3.dot(Eigen::Vector3f(x1, x2, 1.f)) / a;
            Eigen::Vector4f p; p[free_axis] = x; p[a1] = x1; p[a2] = x2; p[3] = 1.f;
            accept(p);
        }
    }
    for (int x = 0; x < 2; ++x) for (int y = 0; y < 2; ++y) for (int z = 0; z < 2; ++z) {
        Eigen::Vector4f p;
        p << (x ? min_corner.x() : max_corner.x()),
             (y ? min_corner.y() : max_corner.y()),
             (z ? min_corner.z() : max_corner.z()), 1.f;
        accept(p);
    }
    return v;
}

inline VoxelOut mesh_to_flexible_dual_grid(
        const float* verts, int V, const int32_t* faces, int F,
        const int grid_size[3], const float aabb_min[3], const float aabb_max[3],
        float face_weight, float boundary_weight, float regularization_weight) {
    Eigen::Vector3f voxel_size;
    for (int a=0; a<3; ++a) voxel_size[a] = (aabb_max[a] - aabb_min[a]) / grid_size[a];
    const Eigen::Vector3i grid_min(0,0,0), grid_max(grid_size[0],grid_size[1],grid_size[2]);
    std::vector<Eigen::Vector3f> vertices((size_t)V);
    for (int i=0; i<V; ++i) for (int a=0; a<3; ++a)
        vertices[(size_t)i][a] = verts[(size_t)i*3+a] - aabb_min[a];

    std::unordered_map<int64_t,int> index; index.reserve((size_t)F*4);
    std::vector<int32_t> coords; coords.reserve((size_t)F*6);
    std::vector<Cell> cells; cells.reserve((size_t)F*6);
    std::vector<int8_t> flags;
    auto get_cell = [&](int x, int y, int z) {
        const int64_t key = cell_key(x,y,z);
        auto it=index.find(key); if (it != index.end()) return it->second;
        const int id=(int)cells.size(); index.emplace(key,id);
        coords.insert(coords.end(), {x,y,z}); cells.emplace_back(); flags.insert(flags.end(), {0,0,0});
        return id;
    };

    // o_voxel::intersect_qef.
    for (int fi=0; fi<F; ++fi) {
        const Eigen::Vector3f& v0=vertices[(size_t)faces[fi*3+0]];
        const Eigen::Vector3f& v1=vertices[(size_t)faces[fi*3+1]];
        const Eigen::Vector3f& v2=vertices[(size_t)faces[fi*3+2]];
        const Eigen::Vector3f e0=v1-v0, e1=v2-v1;
        const Eigen::Vector3f n=e0.cross(e1).normalized();
        if (!std::isfinite(n.x()) || !std::isfinite(n.y()) || !std::isfinite(n.z())) continue;
        Eigen::Vector4f plane; plane << n.x(), n.y(), n.z(), -n.dot(v0);
        const Eigen::Matrix4f Q=plane*plane.transpose();
        for (int ax2=0; ax2<3; ++ax2) {
            const int ax0=(ax2+1)%3, ax1=(ax2+2)%3;
            std::array<Eigen::Vector3d,3> t = {
                Eigen::Vector3d(v0[ax0],v0[ax1],v0[ax2]),
                Eigen::Vector3d(v1[ax0],v1[ax1],v1[ax2]),
                Eigen::Vector3d(v2[ax0],v2[ax1],v2[ax2])};
            std::sort(t.begin(),t.end(),[](const Eigen::Vector3d& a,const Eigen::Vector3d& b){ return a.y()<b.y(); });
            const int start=std::clamp(int(t[0].y()/voxel_size[ax1]),grid_min[ax1],grid_max[ax1]-1);
            const int mid=std::clamp(int(t[1].y()/voxel_size[ax1]),grid_min[ax1],grid_max[ax1]-1);
            const int end=std::clamp(int(t[2].y()/voxel_size[ax1]),grid_min[ax1],grid_max[ax1]-1);
            auto half = [&](int row_start,int row_end,const Eigen::Vector3d& t0,const Eigen::Vector3d& t1,const Eigen::Vector3d& t2) {
                for (int y_idx=row_start; y_idx<row_end; ++y_idx) {
                    const double y=(y_idx+1)*voxel_size[ax1];
                    Eigen::Vector2d t3=qef_lerp(t0.y(),t1.y(),y,Eigen::Vector2d(t0.x(),t0.z()),Eigen::Vector2d(t1.x(),t1.z()));
                    Eigen::Vector2d t4=qef_lerp(t0.y(),t2.y(),y,Eigen::Vector2d(t0.x(),t0.z()),Eigen::Vector2d(t2.x(),t2.z()));
                    if (t3.x()>t4.x()) std::swap(t3,t4);
                    const int line_start=std::clamp(int(t3.x()/voxel_size[ax0]),grid_min[ax0],grid_max[ax0]-1);
                    const int line_end=std::clamp(int(t4.x()/voxel_size[ax0]),grid_min[ax0],grid_max[ax0]-1);
                    for (int x_idx=line_start; x_idx<line_end; ++x_idx) {
                        const double x=(x_idx+1)*voxel_size[ax0];
                        const double z=qef_lerp(t3.x(),t4.x(),x,t3.y(),t4.y());
                        const int z_idx=int(z/voxel_size[ax2]);
                        if (z_idx<grid_min[ax2] || z_idx>=grid_max[ax2]) continue;
                        Eigen::Vector3d intersect; intersect[ax0]=x; intersect[ax1]=y; intersect[ax2]=z;
                        for (int dx=0; dx<2; ++dx) for (int dy=0; dy<2; ++dy) {
                            int c[3]; c[ax0]=x_idx+dx; c[ax1]=y_idx+dy; c[ax2]=z_idx;
                            const int id=get_cell(c[0],c[1],c[2]); Cell& cell=cells[(size_t)id];
                            cell.mean+=intersect.cast<float>(); cell.count+=1.f; cell.qef+=Q;
                            if (dx==0 && dy==0) flags[(size_t)id*3+ax2]=1;
                        }
                    }
                }
            };
            half(start,mid,t[0],t[1],t[2]); half(mid,end,t[2],t[1],t[0]);
        }
    }

    // o_voxel::face_qef.
    if (face_weight>0.f) for (int fi=0; fi<F; ++fi) {
        const Eigen::Vector3f& v0=vertices[(size_t)faces[fi*3+0]];
        const Eigen::Vector3f& v1=vertices[(size_t)faces[fi*3+1]];
        const Eigen::Vector3f& v2=vertices[(size_t)faces[fi*3+2]];
        const Eigen::Vector3f e0=v1-v0,e1=v2-v1,e2=v0-v2,n=e0.cross(e1).normalized();
        if (!std::isfinite(n.x()) || !std::isfinite(n.y()) || !std::isfinite(n.z())) continue;
        Eigen::Vector4f plane; plane << n.x(),n.y(),n.z(),-n.dot(v0); const Eigen::Matrix4f Q=plane*plane.transpose();
        const Eigen::Vector3f bb0=v0.cwiseMin(v1).cwiseMin(v2).cwiseQuotient(voxel_size);
        const Eigen::Vector3f bb1=v0.cwiseMax(v1).cwiseMax(v2).cwiseQuotient(voxel_size);
        const Eigen::Vector3i lo(std::max(int(bb0.x()),0),std::max(int(bb0.y()),0),std::max(int(bb0.z()),0));
        const Eigen::Vector3i hi(std::min(int(bb1.x()+1),grid_max.x()),std::min(int(bb1.y()+1),grid_max.y()),std::min(int(bb1.z()+1),grid_max.z()));
        const Eigen::Vector3f c(n.x()>0.f?voxel_size.x():0.f,n.y()>0.f?voxel_size.y():0.f,n.z()>0.f?voxel_size.z():0.f);
        const float d1=n.dot(c-v0),d2=n.dot(voxel_size-c-v0);
        const int mxy=n.z()<0.f?-1:1, myz=n.x()<0.f?-1:1, mzx=n.y()<0.f?-1:1;
        const Eigen::Vector2f nxy0(-mxy*e0.y(),mxy*e0.x()),nxy1(-mxy*e1.y(),mxy*e1.x()),nxy2(-mxy*e2.y(),mxy*e2.x());
        const Eigen::Vector2f nyz0(-myz*e0.z(),myz*e0.y()),nyz1(-myz*e1.z(),myz*e1.y()),nyz2(-myz*e2.z(),myz*e2.y());
        const Eigen::Vector2f nzx0(-mzx*e0.x(),mzx*e0.z()),nzx1(-mzx*e1.x(),mzx*e1.z()),nzx2(-mzx*e2.x(),mzx*e2.z());
        const Eigen::Vector2f vsxy(voxel_size.x(),voxel_size.y()),vsyz(voxel_size.y(),voxel_size.z()),vszx(voxel_size.z(),voxel_size.x());
        const float dxy0=-nxy0.dot(v0.head<2>())+nxy0.cwiseMax(0.f).dot(vsxy), dxy1=-nxy1.dot(v1.head<2>())+nxy1.cwiseMax(0.f).dot(vsxy), dxy2=-nxy2.dot(v2.head<2>())+nxy2.cwiseMax(0.f).dot(vsxy);
        const float dyz0=-nyz0.dot(Eigen::Vector2f(v0.y(),v0.z()))+nyz0.cwiseMax(0.f).dot(vsyz), dyz1=-nyz1.dot(Eigen::Vector2f(v1.y(),v1.z()))+nyz1.cwiseMax(0.f).dot(vsyz), dyz2=-nyz2.dot(Eigen::Vector2f(v2.y(),v2.z()))+nyz2.cwiseMax(0.f).dot(vsyz);
        const float dzx0=-nzx0.dot(Eigen::Vector2f(v0.z(),v0.x()))+nzx0.cwiseMax(0.f).dot(vszx), dzx1=-nzx1.dot(Eigen::Vector2f(v1.z(),v1.x()))+nzx1.cwiseMax(0.f).dot(vszx), dzx2=-nzx2.dot(Eigen::Vector2f(v2.z(),v2.x()))+nzx2.cwiseMax(0.f).dot(vszx);
        for (int z=lo.z();z<hi.z();++z) for (int y=lo.y();y<hi.y();++y) for (int x=lo.x();x<hi.x();++x) {
            const Eigen::Vector3f p=voxel_size.cwiseProduct(Eigen::Vector3f(x,y,z));
            if ((n.dot(p)+d1)*(n.dot(p)+d2)>0.f) continue;
            const Eigen::Vector2f pxy(p.x(),p.y()),pyz(p.y(),p.z()),pzx(p.z(),p.x());
            if (nxy0.dot(pxy)+dxy0<0.f || nxy1.dot(pxy)+dxy1<0.f || nxy2.dot(pxy)+dxy2<0.f ||
                nyz0.dot(pyz)+dyz0<0.f || nyz1.dot(pyz)+dyz1<0.f || nyz2.dot(pyz)+dyz2<0.f ||
                nzx0.dot(pzx)+dzx0<0.f || nzx1.dot(pzx)+dzx1<0.f || nzx2.dot(pzx)+dzx2<0.f) continue;
            auto it=index.find(cell_key(x,y,z)); if (it!=index.end()) cells[(size_t)it->second].qef+=Q;
        }
    }

    // o_voxel::boundry_qef (the misspelling is present in the upstream source).
    if (boundary_weight>0.f) {
        // Upstream uses std::map, so retain its lexicographic edge order.  The
        // order is observable because Matrix4f additions are not associative.
        std::map<std::pair<int,int>,int> edges;
        for (int fi=0;fi<F;++fi) for (int k=0;k<3;++k) { int a=faces[fi*3+k],b=faces[fi*3+(k+1)%3]; if(a>b)std::swap(a,b); ++edges[{a,b}]; }
        for (const auto& e:edges) if (e.second==1) {
            const Eigen::Vector3f& v0=vertices[(size_t)e.first.first]; const Eigen::Vector3f& v1=vertices[(size_t)e.first.second];
            Eigen::Vector3d dir(v1.x()-v0.x(),v1.y()-v0.y(),v1.z()-v0.z()); const double length=dir.norm(); if(length<1e-6d) continue; dir.normalize();
            const Eigen::Matrix3f A=Eigen::Matrix3f::Identity()-(dir*dir.transpose()).cast<float>(); const Eigen::Vector3f b=-A*v0;
            Eigen::Matrix4f Q=Eigen::Matrix4f::Zero(); Q.topLeftCorner<3,3>()=A; Q.block<3,1>(0,3)=b; Q.block<1,3>(3,0)=b.transpose(); Q(3,3)=v0.transpose()*A*v0;
            Eigen::Vector3i current=(v0.cwiseQuotient(voxel_size)).array().floor().cast<int>(); const Eigen::Vector3i step=(dir.array()>0).select(Eigen::Vector3i(1,1,1),Eigen::Vector3i(-1,-1,-1));
            Eigen::Vector3d tmax,tdelta;
            for(int a=0;a<3;++a) if(dir[a]==0.d) tmax[a]=tdelta[a]=std::numeric_limits<double>::infinity(); else { const float border=voxel_size[a]*(current[a]+(step[a]>0?1:0)); tmax[a]=(border-v0[a])/dir[a]; tdelta[a]=voxel_size[a]/std::abs(dir[a]); }
            std::vector<Eigen::Vector3i> traversed; traversed.push_back(current);
            while(true) { const int a=tmax.x()<tmax.y()?(tmax.x()<tmax.z()?0:2):(tmax.y()<tmax.z()?1:2); if(tmax[a]>length) break; current[a]+=step[a];tmax[a]+=tdelta[a];traversed.push_back(current); }
            for(const auto& c:traversed) { if((c.array()<grid_min.array()).any()||(c.array()>=grid_max.array()).any())continue; auto it=index.find(cell_key(c.x(),c.y(),c.z()));if(it!=index.end())cells[(size_t)it->second].qef+=boundary_weight*Q; }
        }
    }

    VoxelOut out; out.N=(int)cells.size(); out.coords=std::move(coords); out.intersected=std::move(flags); out.dual.resize((size_t)out.N*3);
    for (int i=0;i<out.N;++i) {
        Cell& cell=cells[(size_t)i]; Eigen::Matrix4f Q=cell.qef;
        if (regularization_weight>0.f) { const Eigen::Vector3f p=cell.mean/cell.count; Eigen::Matrix4f reg=Eigen::Matrix4f::Zero();reg.topLeftCorner<3,3>()=Eigen::Matrix3f::Identity();reg.block<3,1>(0,3)=-p;reg.block<1,3>(3,0)=-p.transpose();reg(3,3)=p.dot(p);Q+=regularization_weight*cell.count*reg; }
        const Eigen::Vector3i coord(out.coords[(size_t)i*3],out.coords[(size_t)i*3+1],out.coords[(size_t)i*3+2]);
        const Eigen::Vector3f min=voxel_size.cwiseProduct(coord.cast<float>()), max=voxel_size.cwiseProduct((coord+Eigen::Vector3i::Ones()).cast<float>());
        const Eigen::Vector3f v=solve_qef(Q,min,max);
        for(int a=0;a<3;++a) out.dual[(size_t)i*3+a]=v[a]/voxel_size[a]-coord[a];
    }
    return out;
}

} // namespace vox
