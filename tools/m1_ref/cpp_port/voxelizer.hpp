// voxelizer.hpp — native, source-level port of o_voxel's flexible dual-grid
// forward conversion.  This is the input contract for the native Pixal3D
// texture path: no Python/o_voxel is used at production time.
//
// The important detail is that o_voxel's active cells and QEFs come from three
// passes: scan-line edge intersections (which establish topology and Hermite
// samples), face QEFs, and optional open-boundary QEFs.  A triangle/AABB-only
// approximation looks superficially similar but shifts the dual coordinates
// enough to perturb the learned shape encoder.  Keep this in lockstep with
// o-voxel/src/convert/flexible_dual_grid.cpp.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

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

// Symmetric QEF stored as A (upper triangular) and b where the objective is
// x^T A x - 2 b^T x + constant.  sum/count are o_voxel's intersect means.
struct Cell {
    double A[6] = {0,0,0,0,0,0}; // 00,01,02,11,12,22
    double b[3] = {0,0,0};
    double sum[3] = {0,0,0};
    int count = 0;
    void add_plane(const double n[3], double d, const double p[3], bool sample) {
        A[0]+=n[0]*n[0]; A[1]+=n[0]*n[1]; A[2]+=n[0]*n[2];
        A[3]+=n[1]*n[1]; A[4]+=n[1]*n[2]; A[5]+=n[2]*n[2];
        b[0]+=n[0]*d; b[1]+=n[1]*d; b[2]+=n[2]*d;
        if (sample) { for(int a=0;a<3;a++) sum[a]+=p[a]; ++count; }
    }
    void add_line(const double a[6], const double bb[3], double weight) {
        for(int i=0;i<6;i++) A[i]+=weight*a[i];
        for(int i=0;i<3;i++) b[i]+=weight*bb[i];
    }
};

static inline void solve3(const double A[6], const double b[3], double x[3]) {
    double m[3][3]={{A[0],A[1],A[2]},{A[1],A[3],A[4]},{A[2],A[4],A[5]}};
    double det=m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
              -m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
              +m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]);
    if (std::fabs(det)<1e-20) { x[0]=x[1]=x[2]=0; return; }
    double inv=1.0/det;
    double c00=m[1][1]*m[2][2]-m[1][2]*m[2][1];
    double c01=m[1][2]*m[2][0]-m[1][0]*m[2][2];
    double c02=m[1][0]*m[2][1]-m[1][1]*m[2][0];
    double c11=m[0][0]*m[2][2]-m[0][2]*m[2][0];
    double c12=m[0][2]*m[1][0]-m[0][0]*m[2][1];
    double c22=m[0][0]*m[1][1]-m[0][1]*m[1][0];
    x[0]=inv*(c00*b[0]+c01*b[1]+c02*b[2]);
    x[1]=inv*(c01*b[0]+c11*b[1]+c12*b[2]);
    x[2]=inv*(c02*b[0]+c12*b[1]+c22*b[2]);
}

// Exact unit-cell constrained minimizer.  o_voxel tries each boundary
// dimensionality after its unconstrained QR solve; enumerating all active sets
// yields the same minimizer and is stable for the tiny regularized system.
static inline void solve_unit_box(const double A[6], const double b[3], double out[3]) {
    double m[3][3]={{A[0],A[1],A[2]},{A[1],A[3],A[4]},{A[2],A[4],A[5]}};
    double best=1e300, ans[3]={.5,.5,.5};
    for(int sx=-1;sx<=1;sx++) for(int sy=-1;sy<=1;sy++) for(int sz=-1;sz<=1;sz++) {
        int st[3]={sx,sy,sz}, fr[3], nf=0; double x[3]={0,0,0};
        for(int a=0;a<3;a++) { if(st[a]<0) fr[nf++]=a; else x[a]=(double)st[a]; }
        double rhs[3]={0,0,0};
        for(int i=0;i<nf;i++) { int a=fr[i]; rhs[i]=b[a]; for(int q=0;q<3;q++) if(st[q]>=0) rhs[i]-=m[a][q]*x[q]; }
        bool ok=true;
        if(nf==1) { double d=m[fr[0]][fr[0]]; if(std::fabs(d)<1e-20) ok=false; else x[fr[0]]=rhs[0]/d; }
        else if(nf==2) { int a=fr[0], c=fr[1]; double d=m[a][a]*m[c][c]-m[a][c]*m[a][c]; if(std::fabs(d)<1e-20) ok=false; else { x[a]=(rhs[0]*m[c][c]-m[a][c]*rhs[1])/d; x[c]=(m[a][a]*rhs[1]-m[a][c]*rhs[0])/d; } }
        else if(nf==3) solve3(A,rhs,x);
        if(!ok) continue;
        for(int a=0;a<3;a++) if(x[a]<-1e-8 || x[a]>1.0+1e-8) ok=false;
        if(!ok) continue;
        double q=0; for(int a=0;a<3;a++) { q-=2*b[a]*x[a]; for(int c=0;c<3;c++) q+=x[a]*m[a][c]*x[c]; }
        if(q<best) { best=q; for(int a=0;a<3;a++) ans[a]=std::max(0.0,std::min(1.0,x[a])); }
    }
    for(int a=0;a<3;a++) out[a]=ans[a];
}

static inline double lerp(double a,double b,double t,double va,double vb) {
    return a==b ? va : (1.0-(t-a)/(b-a))*va + ((t-a)/(b-a))*vb;
}

inline VoxelOut mesh_to_flexible_dual_grid(
        const float* verts, int V, const int32_t* faces, int F,
        const int grid_size[3], const float aabb_min[3], const float aabb_max[3],
        float face_weight, float boundary_weight, float regularization_weight) {
    // Work in grid units. This is algebraically the source's shifted-world
    // formulation with voxel_size=(aabb_max-aabb_min)/grid_size.
    double scale[3]; for(int a=0;a<3;a++) scale[a]=grid_size[a]/(double)(aabb_max[a]-aabb_min[a]);
    std::vector<double> g((size_t)V*3);
    for(int i=0;i<V;i++) for(int a=0;a<3;a++) g[(size_t)i*3+a]=((double)verts[(size_t)i*3+a]-aabb_min[a])*scale[a];

    std::unordered_map<int64_t,int> index; index.reserve((size_t)F*4);
    std::vector<int32_t> coords; coords.reserve((size_t)F*6);
    std::vector<Cell> cells; cells.reserve((size_t)F*6);
    std::vector<int8_t> flags;
    auto get_cell=[&](int x,int y,int z) {
        int64_t k=cell_key(x,y,z); auto it=index.find(k); if(it!=index.end()) return it->second;
        int id=(int)cells.size(); index.emplace(k,id); coords.push_back(x); coords.push_back(y); coords.push_back(z);
        cells.emplace_back(); flags.insert(flags.end(),{0,0,0}); return id;
    };

    // o_voxel::intersect_qef: scan all three axis projections. Besides exact
    // occupancy/intersection flags, these repeated plane samples are the QEF's
    // primary term and its regularization target.
    for(int fi=0;fi<F;fi++) {
        const double* v0=&g[(size_t)faces[fi*3+0]*3]; const double* v1=&g[(size_t)faces[fi*3+1]*3]; const double* v2=&g[(size_t)faces[fi*3+2]*3];
        double e0[3]={v1[0]-v0[0],v1[1]-v0[1],v1[2]-v0[2]}, e1[3]={v2[0]-v1[0],v2[1]-v1[1],v2[2]-v1[2]};
        double n[3]={e0[1]*e1[2]-e0[2]*e1[1],e0[2]*e1[0]-e0[0]*e1[2],e0[0]*e1[1]-e0[1]*e1[0]};
        double nl=std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]); if(nl<1e-30) continue;
        for(double& q:n) q/=nl;
        double d=n[0]*v0[0]+n[1]*v0[1]+n[2]*v0[2];
        for(int ax2=0;ax2<3;ax2++) {
            int ax0=(ax2+1)%3, ax1=(ax2+2)%3;
            struct T { double x,y,z; } t[3]={{v0[ax0],v0[ax1],v0[ax2]},{v1[ax0],v1[ax1],v1[ax2]},{v2[ax0],v2[ax1],v2[ax2]}};
            std::sort(t,t+3,[](const T& a,const T& b){return a.y<b.y;});
            int start=std::clamp((int)t[0].y,0,grid_size[ax1]-1), mid=std::clamp((int)t[1].y,0,grid_size[ax1]-1), end=std::clamp((int)t[2].y,0,grid_size[ax1]-1);
            auto half=[&](int ys,int ye,const T& a,const T& b,const T& c) {
                for(int yi=ys;yi<ye;yi++) {
                    double y=yi+1; double t3x=lerp(a.y,b.y,y,a.x,b.x), t3z=lerp(a.y,b.y,y,a.z,b.z), t4x=lerp(a.y,c.y,y,a.x,c.x), t4z=lerp(a.y,c.y,y,a.z,c.z);
                    if(t3x>t4x){std::swap(t3x,t4x);std::swap(t3z,t4z);} int xs=std::clamp((int)t3x,0,grid_size[ax0]-1), xe=std::clamp((int)t4x,0,grid_size[ax0]-1);
                    for(int xi=xs;xi<xe;xi++) { double x=xi+1, z=lerp(t3x,t4x,x,t3z,t4z); int zi=(int)z; if(zi<0||zi>=grid_size[ax2]) continue;
                        double p[3]; p[ax0]=x;p[ax1]=y;p[ax2]=z;
                        for(int dx=0;dx<2;dx++) for(int dy=0;dy<2;dy++) { int cc[3];cc[ax0]=xi+dx;cc[ax1]=yi+dy;cc[ax2]=zi; int id=get_cell(cc[0],cc[1],cc[2]); cells[id].add_plane(n,d,p,true); if(dx==0&&dy==0) flags[(size_t)id*3+ax2]=1; }
                    }
                }
            };
            half(start,mid,t[0],t[1],t[2]); half(mid,end,t[2],t[1],t[0]);
        }
    }

    // o_voxel::face_qef. The source uses face_weight as an on/off gate (the
    // face Q itself is unscaled), so preserve that contract exactly.
    if(face_weight>0) for(int fi=0;fi<F;fi++) {
        const double* v0=&g[(size_t)faces[fi*3+0]*3]; const double* v1=&g[(size_t)faces[fi*3+1]*3]; const double* v2=&g[(size_t)faces[fi*3+2]*3];
        double e0[3]={v1[0]-v0[0],v1[1]-v0[1],v1[2]-v0[2]}, e1[3]={v2[0]-v1[0],v2[1]-v1[1],v2[2]-v1[2]}, e2[3]={v0[0]-v2[0],v0[1]-v2[1],v0[2]-v2[2]};
        double n[3]={e0[1]*e1[2]-e0[2]*e1[1],e0[2]*e1[0]-e0[0]*e1[2],e0[0]*e1[1]-e0[1]*e1[0]}, nl=std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]); if(nl<1e-30) continue; for(double& q:n)q/=nl; double d=n[0]*v0[0]+n[1]*v0[1]+n[2]*v0[2];
        int lo[3],hi[3]; for(int a=0;a<3;a++){lo[a]=std::max((int)std::min({v0[a],v1[a],v2[a]}),0);hi[a]=std::min((int)(std::max({v0[a],v1[a],v2[a]})+1),grid_size[a]);}
        double c[3]={n[0]>0?1.0:0.0,n[1]>0?1.0:0.0,n[2]>0?1.0:0.0}, d1=n[0]*(c[0]-v0[0])+n[1]*(c[1]-v0[1])+n[2]*(c[2]-v0[2]), d2=n[0]*(1-c[0]-v0[0])+n[1]*(1-c[1]-v0[1])+n[2]*(1-c[2]-v0[2]);
        int pairs[3][2]={{0,1},{1,2},{2,0}};
        for(int z=lo[2];z<hi[2];z++)for(int y=lo[1];y<hi[1];y++)for(int x=lo[0];x<hi[0];x++) {
            double p[3]={(double)x,(double)y,(double)z}, ndp=n[0]*x+n[1]*y+n[2]*z; if((ndp+d1)*(ndp+d2)>0) continue; bool pass=true;
            // The three projected triangle half-space tests from source.
            for(int proj=0;proj<3&&pass;proj++) { int a=pairs[proj][0],b=pairs[proj][1], drop=(proj+2)%3; int mul=n[drop]<0?-1:1; const double* vv[3]={v0,v1,v2}; const double* ee[3]={e0,e1,e2};
                for(int k=0;k<3;k++){ double nx=-mul*ee[k][b], ny=mul*ee[k][a]; double dd=-(nx*vv[k][a]+ny*vv[k][b])+(nx>0?nx:0)+(ny>0?ny:0); if(nx*p[a]+ny*p[b]+dd<0){pass=false;break;} }
            }
            if(!pass) continue;
            auto it=index.find(cell_key(x,y,z));
            if(it!=index.end()) cells[it->second].add_plane(n,d,nullptr,false);
        }
    }

    // Boundary line QEFs are relevant for open input meshes. Generated assets
    // are normally closed, but retaining the source behaviour keeps this path
    // model agnostic.
    if(boundary_weight>0) {
        std::unordered_map<uint64_t,int> edges; edges.reserve((size_t)F*3);
        for(int fi=0;fi<F;fi++) for(int k=0;k<3;k++){ uint32_t a=(uint32_t)faces[fi*3+k],b=(uint32_t)faces[fi*3+(k+1)%3];if(a>b)std::swap(a,b);edges[((uint64_t)a<<32)|b]++; }
        for(const auto& kv:edges) if(kv.second==1) { int ia=(int)(kv.first>>32),ib=(int)(uint32_t)kv.first; const double* p0=&g[(size_t)ia*3];const double* p1=&g[(size_t)ib*3]; double dir[3]={p1[0]-p0[0],p1[1]-p0[1],p1[2]-p0[2]}, len=std::sqrt(dir[0]*dir[0]+dir[1]*dir[1]+dir[2]*dir[2]);if(len<1e-6)continue;for(double& q:dir)q/=len;double aa[6]={1-dir[0]*dir[0],-dir[0]*dir[1],-dir[0]*dir[2],1-dir[1]*dir[1],-dir[1]*dir[2],1-dir[2]*dir[2]}, bb[3]={aa[0]*p0[0]+aa[1]*p0[1]+aa[2]*p0[2],aa[1]*p0[0]+aa[3]*p0[1]+aa[4]*p0[2],aa[2]*p0[0]+aa[4]*p0[1]+aa[5]*p0[2]};
            int cur[3]={(int)std::floor(p0[0]),(int)std::floor(p0[1]),(int)std::floor(p0[2])}, step[3]={dir[0]>0?1:-1,dir[1]>0?1:-1,dir[2]>0?1:-1}; double tmax[3],tdelta[3]; for(int a=0;a<3;a++){if(dir[a]==0){tmax[a]=tdelta[a]=INFINITY;}else{tmax[a]=((cur[a]+(step[a]>0?1:0))-p0[a])/dir[a];tdelta[a]=1/std::fabs(dir[a]);}}
            for(;;){auto it=index.find(cell_key(cur[0],cur[1],cur[2]));if(it!=index.end())cells[it->second].add_line(aa,bb,boundary_weight);int ax=(tmax[0]<tmax[1])?(tmax[0]<tmax[2]?0:2):(tmax[1]<tmax[2]?1:2);if(tmax[ax]>len)break;cur[ax]+=step[ax];tmax[ax]+=tdelta[ax];}
        }
    }

    VoxelOut out; out.N=(int)cells.size(); out.coords=std::move(coords); out.dual.resize((size_t)out.N*3); out.intersected=std::move(flags);
    for(int i=0;i<out.N;i++) { Cell& c=cells[i]; double A[6]={c.A[0],c.A[1],c.A[2],c.A[3],c.A[4],c.A[5]}, b[3]={c.b[0],c.b[1],c.b[2]}; if(regularization_weight>0&&c.count>0)for(int a=0;a<3;a++){A[a==0?0:a==1?3:5]+=regularization_weight*c.count;b[a]+=regularization_weight*c.sum[a];}
        int32_t* cc=&out.coords[(size_t)i*3]; double local_b[3]={b[0]-(A[0]*cc[0]+A[1]*cc[1]+A[2]*cc[2]),b[1]-(A[1]*cc[0]+A[3]*cc[1]+A[4]*cc[2]),b[2]-(A[2]*cc[0]+A[4]*cc[1]+A[5]*cc[2])}, x[3]; solve_unit_box(A,local_b,x);for(int a=0;a<3;a++)out.dual[(size_t)i*3+a]=(float)x[a]; }
    return out;
}

} // namespace vox
