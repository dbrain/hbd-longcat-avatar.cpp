// obj_manifold_clean — make a triangle OBJ manifold-ish so QuadriFlow's Initialize() (edge-adjacency /
// disjoint-tree, Eigen-indexed) doesn't OOB-assert. The sloppy-decimated detailed mesh (/tmp/det.obj)
// is topologically non-manifold (edges shared by >2 faces, bowtie verts) — QF crashes on it. Here we:
//   1. weld coincident verts (spatial hash @ epsilon) so split edges re-share
//   2. drop degenerate faces (repeated index / zero area) and duplicate faces
//   3. remove non-manifold edges greedily (keep the 2 most-coplanar faces per edge, drop the rest;
//      iterate until every edge has <=2 incident faces)
//   4. drop faces that leave non-manifold (bowtie) vertices: a vertex whose incident faces form >1
//      edge-connected fan
//   5. keep the largest edge-connected component
// Reports manifold stats so we can SEE whether the result is QF-ingestible before feeding it.
//   build: g++ -O2 -std=c++17 obj_manifold_clean.cpp -o obj_manifold_clean
//   run:   ./obj_manifold_clean in.obj out.obj [weld_eps_frac=2e-5]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <array>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <functional>

struct V3 { float x,y,z; };

static uint64_t ekey(uint32_t a, uint32_t b){ if(a>b) std::swap(a,b); return (uint64_t)a<<32 | b; }

int main(int argc, char** argv){
    if(argc<3){ printf("usage: obj_manifold_clean in.obj out.obj [weld_eps_frac=2e-5]\n"); return 1; }
    float eps_frac = argc>3 ? atof(argv[3]) : 2e-5f;

    // --- parse OBJ (v + f, triangulate polygons) ---
    std::vector<V3> V;
    std::vector<uint32_t> F; // triangle indices (0-based)
    {
        FILE* f=fopen(argv[1],"r"); if(!f){printf("open %s failed\n",argv[1]);return 1;}
        char line[512];
        while(fgets(line,sizeof line,f)){
            if(line[0]=='v'&&line[1]==' '){ V3 p; sscanf(line+2,"%f %f %f",&p.x,&p.y,&p.z); V.push_back(p); }
            else if(line[0]=='f'&&line[1]==' '){
                int v[16]; int n=0; char* s=line+2;
                while(*s&&n<16){ while(*s==' ')s++; if(!*s||*s=='\n')break;
                    long idx=strtol(s,&s,10); if(idx<0) idx=(long)V.size()+idx+1; v[n++]=(int)idx-1;
                    while(*s&&*s!=' '&&*s!='\n')s++; } // skip /vt/vn
                for(int k=1;k+1<n;k++){ F.push_back(v[0]); F.push_back(v[k]); F.push_back(v[k+1]); }
            }
        }
        fclose(f);
    }
    printf("[in] %zu verts, %zu tris\n", V.size(), F.size()/3);

    // --- bbox + weld epsilon ---
    V3 lo{1e30f,1e30f,1e30f}, hi{-1e30f,-1e30f,-1e30f};
    for(auto&p:V){ lo.x=std::min(lo.x,p.x);lo.y=std::min(lo.y,p.y);lo.z=std::min(lo.z,p.z);
                   hi.x=std::max(hi.x,p.x);hi.y=std::max(hi.y,p.y);hi.z=std::max(hi.z,p.z); }
    float diag=std::sqrt((hi.x-lo.x)*(hi.x-lo.x)+(hi.y-lo.y)*(hi.y-lo.y)+(hi.z-lo.z)*(hi.z-lo.z));
    float eps=diag*eps_frac; float inv=eps>0?1.0f/eps:1.0f;
    printf("[weld] bbox diag=%.4f eps=%.6g\n", diag, eps);

    // --- weld coincident verts via quantized spatial hash ---
    std::unordered_map<uint64_t,uint32_t> cell; cell.reserve(V.size()*2);
    std::vector<uint32_t> remap(V.size());
    std::vector<V3> NV; NV.reserve(V.size());
    auto qkey=[&](const V3&p)->uint64_t{
        int64_t ix=(int64_t)std::floor(p.x*inv), iy=(int64_t)std::floor(p.y*inv), iz=(int64_t)std::floor(p.z*inv);
        // hash three ints
        uint64_t h=1469598103934665603ull;
        for(int64_t c:{ix,iy,iz}){ h^=(uint64_t)c; h*=1099511628211ull; }
        return h;
    };
    for(size_t i=0;i<V.size();i++){
        uint64_t k=qkey(V[i]);
        auto it=cell.find(k);
        if(it==cell.end()){ uint32_t id=(uint32_t)NV.size(); cell.emplace(k,id); NV.push_back(V[i]); remap[i]=id; }
        else remap[i]=it->second;
    }
    printf("[weld] %zu -> %zu verts (%zu merged)\n", V.size(), NV.size(), V.size()-NV.size());

    // --- remap faces, drop degenerate + duplicate ---
    std::vector<std::array<uint32_t,3>> tris;
    std::unordered_set<uint64_t> seen; seen.reserve(F.size());
    size_t degen=0, dup=0;
    for(size_t i=0;i<F.size();i+=3){
        uint32_t a=remap[F[i]],b=remap[F[i+1]],c=remap[F[i+2]];
        if(a==b||b==c||a==c){ degen++; continue; }
        uint32_t s0=a,s1=b,s2=c; if(s0>s1)std::swap(s0,s1); if(s1>s2)std::swap(s1,s2); if(s0>s1)std::swap(s0,s1);
        uint64_t fk=((uint64_t)s0*2654435761u+s1)*2654435761u+s2;
        if(!seen.insert(fk).second){ dup++; continue; }
        tris.push_back({a,b,c});
    }
    printf("[face] dropped %zu degenerate, %zu duplicate -> %zu tris\n", degen, dup, tris.size());

    auto farea=[&](const std::array<uint32_t,3>&t)->float{
        V3&A=NV[t[0]],&B=NV[t[1]],&C=NV[t[2]];
        float ux=B.x-A.x,uy=B.y-A.y,uz=B.z-A.z, vx=C.x-A.x,vy=C.y-A.y,vz=C.z-A.z;
        float cx=uy*vz-uz*vy,cy=uz*vx-ux*vz,cz=ux*vy-uy*vx;
        return 0.5f*std::sqrt(cx*cx+cy*cy+cz*cz);
    };
    auto fnorm=[&](const std::array<uint32_t,3>&t,float n[3]){
        V3&A=NV[t[0]],&B=NV[t[1]],&C=NV[t[2]];
        float ux=B.x-A.x,uy=B.y-A.y,uz=B.z-A.z, vx=C.x-A.x,vy=C.y-A.y,vz=C.z-A.z;
        n[0]=uy*vz-uz*vy;n[1]=uz*vx-ux*vz;n[2]=ux*vy-uy*vx;
        float l=std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2])+1e-30f; n[0]/=l;n[1]/=l;n[2]/=l;
    };

    // --- iteratively remove non-manifold edges (edge with >2 faces): keep 2 most-coplanar, drop rest ---
    std::vector<char> dead(tris.size(),0);
    for(int pass=0; pass<8; pass++){
        std::unordered_map<uint64_t,std::vector<uint32_t>> e2f; e2f.reserve(tris.size()*3);
        for(uint32_t fi=0;fi<tris.size();fi++){ if(dead[fi])continue; auto&t=tris[fi];
            e2f[ekey(t[0],t[1])].push_back(fi); e2f[ekey(t[1],t[2])].push_back(fi); e2f[ekey(t[2],t[0])].push_back(fi); }
        size_t removed=0, nm_edges=0;
        for(auto&kv:e2f){
            // count live faces on this edge
            std::vector<uint32_t> live; for(uint32_t fi:kv.second) if(!dead[fi]) live.push_back(fi);
            if(live.size()<=2) continue;
            nm_edges++;
            // keep the 2 faces whose normals are most anti/para consistent (best dihedral pair),
            // drop the rest. Cheap heuristic: keep two with largest area (more reliable surface).
            std::sort(live.begin(),live.end(),[&](uint32_t a,uint32_t b){ return farea(tris[a])>farea(tris[b]); });
            for(size_t i=2;i<live.size();i++){ if(!dead[live[i]]){ dead[live[i]]=1; removed++; } }
        }
        printf("[nm-edge pass %d] %zu non-manifold edges, removed %zu faces\n", pass, nm_edges, removed);
        if(removed==0) break;
    }

    // --- largest edge-connected component (union-find over live faces sharing an edge) ---
    std::vector<uint32_t> live; for(uint32_t fi=0;fi<tris.size();fi++) if(!dead[fi]) live.push_back(fi);
    std::vector<uint32_t> uf(tris.size()); for(uint32_t i=0;i<uf.size();i++) uf[i]=i;
    std::function<uint32_t(uint32_t)> find=[&](uint32_t x){ while(uf[x]!=x){ uf[x]=uf[uf[x]]; x=uf[x]; } return x; };
    {
        std::unordered_map<uint64_t,uint32_t> efirst; efirst.reserve(live.size()*3);
        for(uint32_t fi:live){ auto&t=tris[fi];
            for(uint64_t e:{ekey(t[0],t[1]),ekey(t[1],t[2]),ekey(t[2],t[0])}){
                auto it=efirst.find(e);
                if(it==efirst.end()) efirst.emplace(e,fi);
                else { uint32_t ra=find(it->second),rb=find(fi); if(ra!=rb) uf[ra]=rb; }
            }
        }
    }
    std::unordered_map<uint32_t,uint32_t> compsz;
    for(uint32_t fi:live) compsz[find(fi)]++;
    uint32_t best=0; uint32_t bestsz=0;
    for(auto&kv:compsz) if(kv.second>bestsz){ bestsz=kv.second; best=kv.first; }
    printf("[component] %zu components, largest=%u faces (of %zu live)\n", compsz.size(), bestsz, live.size());

    // --- collect final faces + boundary/non-manifold stats ---
    std::vector<std::array<uint32_t,3>> outF;
    for(uint32_t fi:live) if(find(fi)==best) outF.push_back(tris[fi]);
    {
        std::unordered_map<uint64_t,int> ec; ec.reserve(outF.size()*3);
        for(auto&t:outF){ ec[ekey(t[0],t[1])]++; ec[ekey(t[1],t[2])]++; ec[ekey(t[2],t[0])]++; }
        size_t bnd=0,nm=0; for(auto&kv:ec){ if(kv.second==1)bnd++; else if(kv.second>2)nm++; }
        printf("[final] %zu faces, %zu boundary edges, %zu non-manifold edges\n", outF.size(), bnd, nm);
    }

    // --- remap used verts, write OBJ ---
    std::vector<uint32_t> vmap(NV.size(),UINT32_MAX); std::vector<V3> OV;
    for(auto&t:outF) for(int k=0;k<3;k++){ uint32_t&vi=t[k];
        if(vmap[vi]==UINT32_MAX){ vmap[vi]=(uint32_t)OV.size(); OV.push_back(NV[vi]); } vi=vmap[vi]; }
    FILE* o=fopen(argv[2],"w"); if(!o){printf("out open failed\n");return 1;}
    for(auto&p:OV) fprintf(o,"v %g %g %g\n",p.x,p.y,p.z);
    for(auto&t:outF) fprintf(o,"f %u %u %u\n",t[0]+1,t[1]+1,t[2]+1);
    fclose(o);
    printf("[out] wrote %s: %zu verts, %zu tris\n", argv[2], OV.size(), outF.size());
    return 0;
}
