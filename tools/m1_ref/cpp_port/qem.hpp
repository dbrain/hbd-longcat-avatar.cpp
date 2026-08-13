// qem.hpp — feature-preserving Quadric Error Metric mesh decimation that TOLERATES non-manifold
// input. (lap-17 P0 crispness.) meshopt's QUALITY simplifier locks on the O-Voxel dual-grid mesh's
// ~162k non-manifold edges + ~58k boundary edges and stalls at ~1.08M faces, then falls to
// simplifySloppy (noisy/mottled normals → the "putty blob"). Classic Garland-Heckbert QEM instead
// accumulates a per-vertex error quadric from ALL incident faces and collapses each edge
// independently — non-manifold edges are no obstacle — so it decimates the SHARP dual mesh straight
// to a web-sized poly budget while preserving the silhouette + sharp features (heel spikes, fingers,
// chin gap), because collapsing across a sharp edge has high quadric cost and is deferred.
//
// Self-contained (no deps beyond <vector>/<cmath>); structure follows the well-known sp4cerat
// "Fast-Quadric-Mesh-Simplification" iterative-threshold scheme (public domain), reimplemented here
// for this port. Pure host. Operates on the svae::Mesh layout (float xyz verts, int64 tri indices).
#pragma once
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include "sparse_vae.hpp"   // svae::Mesh, now_s

namespace qem {

struct SymMat {                       // symmetric 4x4 quadric, 10 unique entries
    double m[10];
    SymMat(double c=0){ for(double&x:m) x=c; }
    SymMat(double a,double b,double c,double d){            // plane (a,b,c,d) → outer product Kp
        m[0]=a*a; m[1]=a*b; m[2]=a*c; m[3]=a*d;
        m[4]=b*b; m[5]=b*c; m[6]=b*d;
        m[7]=c*c; m[8]=c*d; m[9]=d*d;
    }
    double operator[](int i) const { return m[i]; }
    SymMat operator+(const SymMat&o) const { SymMat r; for(int i=0;i<10;i++) r.m[i]=m[i]+o.m[i]; return r; }
    SymMat& operator+=(const SymMat&o){ for(int i=0;i<10;i++) m[i]+=o.m[i]; return *this; }
    // determinant of the 3x3 minor formed by columns (a,b,c) rows (0..2) — for the optimal-point solve
    double det(int a11,int a12,int a13,int a21,int a22,int a23,int a31,int a32,int a33) const {
        return m[a11]*(m[a22]*m[a33]-m[a23]*m[a32]) - m[a12]*(m[a21]*m[a33]-m[a23]*m[a31])
             + m[a13]*(m[a21]*m[a32]-m[a22]*m[a31]);
    }
};

struct Vec3 { double x,y,z;
    Vec3(double a=0,double b=0,double c=0):x(a),y(b),z(c){}
    Vec3 operator-(const Vec3&o)const{return {x-o.x,y-o.y,z-o.z};}
    Vec3 operator+(const Vec3&o)const{return {x+o.x,y+o.y,z+o.z};}
    Vec3 operator*(double s)const{return {x*s,y*s,z*s};}
    Vec3 cross(const Vec3&o)const{return {y*o.z-z*o.y, z*o.x-x*o.z, x*o.y-y*o.x};}
    double dot(const Vec3&o)const{return x*o.x+y*o.y+z*o.z;}
    double len()const{return std::sqrt(x*x+y*y+z*z);}
    Vec3 norm()const{double l=len(); return l>1e-20?Vec3{x/l,y/l,z/l}:Vec3{0,0,0};}
};

struct Tri { int v[3]; double err[4]; int deleted, dirty; Vec3 n; };
struct Vert { Vec3 p; int tstart, tcount, border; SymMat q; Vec3 col; };  // col = carried per-vertex colour
struct Ref { int tid, tv; };

struct Simplify {
    std::vector<Tri> tris; std::vector<Vert> verts; std::vector<Ref> refs;

    // quadric error of position p against quadric q
    static double verr(const SymMat&q, double x,double y,double z){
        return q[0]*x*x + 2*q[1]*x*y + 2*q[2]*x*z + 2*q[3]*x
             + q[4]*y*y + 2*q[5]*y*z + 2*q[6]*y
             + q[7]*z*z + 2*q[8]*z + q[9];
    }
    // optimal contraction point + its error for edge (id_v1,id_v2)
    double calc_error(int id_v1,int id_v2, Vec3& result){
        SymMat q = verts[id_v1].q + verts[id_v2].q;
        bool border = verts[id_v1].border && verts[id_v2].border;
        double error=0; double det = q.det(0,1,2, 1,4,5, 2,5,7);
        if (det!=0 && !border){
            // optimal vertex = -A^-1 b  (Cramer's rule on the 3x3 quadric)
            double x = -1/det*(q.det(1,2,3, 4,5,6, 5,7,8));
            double y =  1/det*(q.det(0,2,3, 1,5,6, 2,7,8));
            double z = -1/det*(q.det(0,1,3, 1,4,6, 2,5,8));
            result = {x,y,z}; error = verr(q,x,y,z);
        } else {
            Vec3 p1=verts[id_v1].p, p2=verts[id_v2].p, p3=(p1+p2)*0.5;
            double e1=verr(q,p1.x,p1.y,p1.z), e2=verr(q,p2.x,p2.y,p2.z), e3=verr(q,p3.x,p3.y,p3.z);
            error=std::min(e1,std::min(e2,e3));
            if(error==e1) result=p1; else if(error==e2) result=p2; else result=p3;
        }
        return error;
    }
    // would moving vertex i0 to p flip any of i0's triangles (excluding the collapsing edge)?
    bool flipped(Vec3 p,int i0,int i1,Vert&v0, std::vector<int>&deleted){
        for(int k=0;k<v0.tcount;k++){
            Tri&t=tris[refs[v0.tstart+k].tid]; if(t.deleted) continue;
            int s=refs[v0.tstart+k].tv; int id1=t.v[(s+1)%3], id2=t.v[(s+2)%3];
            if(id1==i1||id2==i1){ deleted[k]=1; continue; }       // this tri dies in the collapse
            Vec3 d1=(verts[id1].p-p).norm(), d2=(verts[id2].p-p).norm();
            if(std::fabs(d1.dot(d2))>0.999) return true;          // degenerate (near-colinear)
            Vec3 n=d1.cross(d2).norm();
            deleted[k]=0;
            if(n.dot(t.n)<0.2) return true;                       // normal would flip
        }
        return false;
    }
    void update_tris(int i0,Vert&v,std::vector<int>&deleted,int&deleted_tris){
        Vec3 p;
        for(int k=0;k<v.tcount;k++){ Ref r=refs[v.tstart+k]; Tri&t=tris[r.tid];
            if(t.deleted) continue;
            if(deleted[k]){ t.deleted=1; deleted_tris++; continue; }
            t.v[r.tv]=i0; t.dirty=1;
            t.err[0]=calc_error(t.v[0],t.v[1],p); t.err[1]=calc_error(t.v[1],t.v[2],p);
            t.err[2]=calc_error(t.v[2],t.v[0],p); t.err[3]=std::min(t.err[0],std::min(t.err[1],t.err[2]));
            refs.push_back(r);
        }
    }
    void update_mesh(int iter){
        if(iter>0){ int dst=0; for(auto&t:tris) if(!t.deleted) tris[dst++]=t; tris.resize(dst); }
        // recompute vertex quadrics + face normals on iter 0
        if(iter==0){
            for(auto&v:verts) v.q=SymMat(0);
            for(auto&t:tris){ Vec3 p0=verts[t.v[0]].p,p1=verts[t.v[1]].p,p2=verts[t.v[2]].p;
                Vec3 n=(p1-p0).cross(p2-p0).norm(); t.n=n;
                SymMat kp(n.x,n.y,n.z,-n.dot(p0));
                for(int j=0;j<3;j++) verts[t.v[j]].q+=kp;
            }
            for(auto&t:tris){ Vec3 p; for(int j=0;j<3;j++) t.err[j]=calc_error(t.v[j],t.v[(j+1)%3],p);
                t.err[3]=std::min(t.err[0],std::min(t.err[1],t.err[2])); }
        }
        // rebuild refs (vertex→triangle adjacency)
        for(auto&v:verts){ v.tstart=0; v.tcount=0; }
        for(auto&t:tris) for(int j=0;j<3;j++) verts[t.v[j]].tcount++;
        int start=0; for(auto&v:verts){ v.tstart=start; start+=v.tcount; v.tcount=0; }
        refs.resize(tris.size()*3);
        for(int i=0;i<(int)tris.size();i++) for(int j=0;j<3;j++){ Vert&v=verts[tris[i].v[j]];
            refs[v.tstart+v.tcount]={i,j}; v.tcount++; }
        // border classification on iter 0: an edge used by exactly one incident triangle of the
        // vertex pair → boundary. (open dual-grid frontier — preserve it.)
        if(iter==0){
            std::vector<int> vcount, vids;
            for(auto&v:verts) v.border=0;
            for(auto&v:verts){ vcount.clear(); vids.clear();
                for(int k=0;k<v.tcount;k++){ Tri&t=tris[refs[v.tstart+k].tid];
                    for(int j=0;j<3;j++){ int id=t.v[j]; int ofs=0; while(ofs<(int)vids.size()&&vids[ofs]!=id) ofs++;
                        if(ofs==(int)vids.size()){ vcount.push_back(1); vids.push_back(id); } else vcount[ofs]++; } }
                for(size_t k=0;k<vids.size();k++) if(vcount[k]==1) verts[vids[k]].border=1;
            }
        }
    }
    void compact(){
        int dst=0; for(auto&v:verts) v.tcount=0;
        for(auto&t:tris) if(!t.deleted){ tris[dst++]=t; for(int j=0;j<3;j++) verts[t.v[j]].tcount=1; }
        tris.resize(dst); dst=0;
        for(auto&v:verts) if(v.tcount){ v.tstart=dst; verts[dst].p=v.p; verts[dst].col=v.col; dst++; }
        for(auto&t:tris) for(int j=0;j<3;j++) t.v[j]=verts[t.v[j]].tstart;
        verts.resize(dst);
    }
    // target_count = triangles to keep; aggressiveness ~7 (sp4cerat default)
    void simplify(int target_count, double aggressiveness=7.0, int max_iters=200){
        for(auto&t:tris) t.deleted=0;
        int deleted_tris=0; int Ntri=(int)tris.size();
        std::vector<int> deleted0, deleted1;
        int prev_deleted=-1, stall=0;
        for(int iter=0; iter<max_iters; iter++){
            if(Ntri-deleted_tris<=target_count) break;
            // early-exit: once the threshold is large yet collapses have stalled for 3 rounds, the
            // remaining edges are all feature/border (collapsing them would flip normals) — the
            // feature-preserving floor. Stop instead of burning iterations to max_iters.
            if(iter>4){ if(deleted_tris==prev_deleted){ if(++stall>=3) break; } else stall=0; }
            prev_deleted=deleted_tris;
            if(iter%5==0) update_mesh(iter); else if(iter>0){} // refs stay valid between non-compaction iters
            for(auto&t:tris) t.dirty=0;
            double threshold = 0.000000001*std::pow(double(iter+3), aggressiveness);
            for(int i=0;i<(int)tris.size();i++){ Tri&t=tris[i];
                if(t.err[3]>threshold||t.deleted||t.dirty) continue;
                for(int j=0;j<3;j++){ if(t.err[j]>threshold) continue;
                    int i0=t.v[j], i1=t.v[(j+1)%3];
                    if(verts[i0].border!=verts[i1].border) continue;   // don't collapse boundary→interior
                    Vec3 p; calc_error(i0,i1,p);
                    deleted0.assign(verts[i0].tcount,0); deleted1.assign(verts[i1].tcount,0);
                    if(flipped(p,i0,i1,verts[i0],deleted0)) continue;
                    if(flipped(p,i1,i0,verts[i1],deleted1)) continue;
                    verts[i0].p=p; verts[i0].q=verts[i1].q+verts[i0].q;
                    verts[i0].col=(verts[i0].col+verts[i1].col)*0.5;   // carry/average per-vertex colour
                    int rstart=(int)refs.size();
                    update_tris(i0,verts[i0],deleted0,deleted_tris);
                    update_tris(i0,verts[i1],deleted1,deleted_tris);
                    int rcount=(int)refs.size()-rstart;
                    verts[i0].tstart=rstart; verts[i0].tcount=rcount;
                    break;
                }
                if(Ntri-deleted_tris<=target_count) break;
            }
        }
        compact();
    }
};

// Decimate a (possibly non-manifold) svae::Mesh to ~target_faces, preserving features. Returns the
// simplified mesh. aggr controls speed/quality (lower = gentler/slower/cleaner; 7 = sp4cerat default).
// in_col/out_col (optional): per-vertex RGB [V*3] carried through the collapses (averaged on merge),
// so the decimated mesh inherits the DENSE mesh's exact, smooth, hole-free colour — instead of
// re-sampling the sparse PBR volume at the moved/sparse output verts (which gives the noisy
// teal/black "zombie" patchwork). The dense source verts sit exactly on the shell so their colour is
// correct; averaging down a local cluster stays on that surface and is smooth.
inline svae::Mesh qem_simplify(const svae::Mesh& in, int target_faces, double aggr=7.0,
                               const std::vector<float>* in_col=nullptr, std::vector<float>* out_col=nullptr){
    Simplify s;
    const int64_t V=(int64_t)in.verts.size()/3, F=(int64_t)in.faces.size()/3;
    s.verts.resize(V);
    for(int64_t i=0;i<V;i++){ s.verts[i].p={in.verts[i*3],in.verts[i*3+1],in.verts[i*3+2]};
        s.verts[i].tstart=0; s.verts[i].tcount=0; s.verts[i].border=0;
        s.verts[i].col = in_col ? Vec3{(*in_col)[i*3],(*in_col)[i*3+1],(*in_col)[i*3+2]} : Vec3{0,0,0}; }
    s.tris.resize(F);
    for(int64_t i=0;i<F;i++){ s.tris[i].v[0]=(int)in.faces[i*3]; s.tris[i].v[1]=(int)in.faces[i*3+1];
        s.tris[i].v[2]=(int)in.faces[i*3+2]; s.tris[i].deleted=0; s.tris[i].dirty=0; }
    s.simplify(target_faces, aggr);
    svae::Mesh out; out.verts.resize(s.verts.size()*3); out.faces.resize(s.tris.size()*3);
    for(size_t i=0;i<s.verts.size();i++){ out.verts[i*3]=(float)s.verts[i].p.x;
        out.verts[i*3+1]=(float)s.verts[i].p.y; out.verts[i*3+2]=(float)s.verts[i].p.z; }
    for(size_t i=0;i<s.tris.size();i++) for(int j=0;j<3;j++) out.faces[i*3+j]=s.tris[i].v[j];
    out.N=(int)s.verts.size(); out.F=(int)s.tris.size();
    if(out_col){ out_col->resize(s.verts.size()*3);
        for(size_t i=0;i<s.verts.size();i++){ (*out_col)[i*3]=(float)s.verts[i].col.x;
            (*out_col)[i*3+1]=(float)s.verts[i].col.y; (*out_col)[i*3+2]=(float)s.verts[i].col.z; } }
    return out;
}

} // namespace qem
