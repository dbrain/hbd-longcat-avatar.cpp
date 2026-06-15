// obj_decimate — meshopt quality (manifold-preserving, feature-aware QEM) decimation of an OBJ to a
// target triangle count. Unlike the raw marching-tet voxel lattice (which stalled meshopt at ~13.8M,
// FINDINGS-15), the ManifoldPlus output is a smooth watertight octree surface that QEM decimates
// cleanly — so we can shrink a high-depth (finger-preserving) manifold down to a size QuadriFlow's
// hierarchy can ingest without its 26GB memory blowup. Triangulates quads on input.
//   build: g++ -O2 -std=c++17 obj_decimate.cpp ../../../thirdparty/meshoptimizer/*.cpp -o obj_decimate
//   run:   ./obj_decimate in.obj out.obj <target_tris> [target_error=1e-2]
#include "../../../thirdparty/meshoptimizer/meshoptimizer.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>

int main(int argc, char** argv){
    if(argc<4){ printf("usage: obj_decimate in.obj out.obj <target_tris> [target_error=1e-2]\n"); return 1; }
    size_t target_tris=(size_t)atol(argv[3]);
    float target_error=argc>4?(float)atof(argv[4]):1e-2f;

    std::vector<float> V; std::vector<uint32_t> idx;
    { FILE* f=fopen(argv[1],"r"); if(!f){printf("open failed\n");return 1;} char line[1024];
      while(fgets(line,sizeof line,f)){
        if(line[0]=='v'&&line[1]==' '){ float x,y,z; sscanf(line+2,"%f %f %f",&x,&y,&z); V.push_back(x);V.push_back(y);V.push_back(z); }
        else if(line[0]=='f'&&line[1]==' '){ int v[16],n=0; char* s=line+2;
            while(*s&&n<16){ while(*s==' ')s++; if(!*s||*s=='\n')break; long a=strtol(s,&s,10); if(a<0)a=(long)V.size()/3+a+1; v[n++]=(int)a-1; while(*s&&*s!=' '&&*s!='\n')s++; }
            for(int k=1;k+1<n;k++){ idx.push_back(v[0]);idx.push_back(v[k]);idx.push_back(v[k+1]); } } }
      fclose(f); }
    size_t nv=V.size()/3;
    printf("[in] %zu verts, %zu tris\n", nv, idx.size()/3);

    std::vector<uint32_t> out(idx.size()); float err=0;
    size_t n=meshopt_simplify(out.data(), idx.data(), idx.size(), V.data(), nv, 12,
                              target_tris*3, target_error, meshopt_SimplifyLockBorder, &err);
    out.resize(n);
    printf("[simplify] %zu -> %zu tris (err=%.5f)\n", idx.size()/3, n/3, err);

    // compact used verts
    std::vector<uint32_t> remap(nv, UINT32_MAX); std::vector<float> NV;
    for(uint32_t& vi:out){ if(remap[vi]==UINT32_MAX){ remap[vi]=(uint32_t)(NV.size()/3);
        NV.push_back(V[vi*3]);NV.push_back(V[vi*3+1]);NV.push_back(V[vi*3+2]); } vi=remap[vi]; }
    FILE* o=fopen(argv[2],"w"); if(!o){printf("out open failed\n");return 1;}
    for(size_t i=0;i<NV.size();i+=3) fprintf(o,"v %g %g %g\n",NV[i],NV[i+1],NV[i+2]);
    for(size_t i=0;i<out.size();i+=3) fprintf(o,"f %u %u %u\n",out[i]+1,out[i+1]+1,out[i+2]+1);
    fclose(o);
    printf("[out] wrote %s: %zu verts, %zu tris\n", argv[2], NV.size()/3, out.size()/3);
    return 0;
}
