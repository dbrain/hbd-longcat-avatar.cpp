// mesh_topo — position-welded topology stats (open + non-manifold edges) of a GLB.
#include <cstdio>
#include <cmath>
#include <unordered_map>
#include <vector>
#include "glb_reader.hpp"
int main(int argc,char**argv){
    if(argc<2){printf("usage: mesh_topo a.glb [eps_frac=1e-5]\n");return 1;}
    double epsf=argc>2?atof(argv[2]):1e-5;
    glb::Mesh m; if(!glb::read_glb(argv[1],m)){printf("read fail\n");return 1;}
    int64_t V=m.verts.size()/3,F=m.faces.size()/3;
    double mn[3]={1e30,1e30,1e30},mx[3]={-1e30,-1e30,-1e30};
    for(int64_t i=0;i<V;i++)for(int d=0;d<3;d++){float x=m.verts[i*3+d];mn[d]=std::min(mn[d],(double)x);mx[d]=std::max(mx[d],(double)x);}
    double diag=std::sqrt((mx[0]-mn[0])*(mx[0]-mn[0])+(mx[1]-mn[1])*(mx[1]-mn[1])+(mx[2]-mn[2])*(mx[2]-mn[2]));
    double inv=1.0/(diag*epsf);
    std::unordered_map<uint64_t,int64_t> cell; std::vector<int64_t> canon(V); int64_t nx=0;
    for(int64_t i=0;i<V;i++){int64_t x=llround(m.verts[i*3]*inv),y=llround(m.verts[i*3+1]*inv),z=llround(m.verts[i*3+2]*inv);
        uint64_t k=((uint64_t)(uint32_t)(x*73856093))^((uint64_t)(uint32_t)(y*19349663)<<21)^((uint64_t)(uint32_t)(z*83492791)<<42);
        auto it=cell.find(k); if(it==cell.end()){cell.emplace(k,nx);canon[i]=nx++;}else canon[i]=it->second;}
    std::unordered_map<uint64_t,int> ec; ec.reserve(F*3);
    for(int64_t f=0;f<F;f++)for(int e=0;e<3;e++){int64_t a=canon[m.faces[f*3+e]],b=canon[m.faces[f*3+(e+1)%3]];if(a==b)continue;uint64_t k=(uint64_t)std::min(a,b)<<32^std::max(a,b);ec[k]++;}
    int64_t op=0,nm=0; for(auto&kv:ec){if(kv.second==1)op++;else if(kv.second>2)nm++;}
    printf("%s: V=%lld uniqV=%lld F=%lld  open=%lld nonmanifold=%lld (eps=%.1e*diag)\n",argv[1],(long long)V,(long long)nx,(long long)F,(long long)op,(long long)nm,epsf);
    return 0;
}
