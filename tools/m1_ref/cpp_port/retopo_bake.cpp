// retopo_bake — bake the cached PBR volume onto a QuadriFlow retopo mesh's UVs, pack to a compressed
// GLB. Closes the Miku retopo chain on CPU (no GPU): occupancy→coarse manifold MC→QuadriFlow gave the
// quad mesh; here we xatlas-unwrap it (real ComputeCharts, NOT cumesh precluster), grid_sample the
// cached PBR volume (dump_pbr_*.bin) onto it, TELEA-inpaint, and meshopt+KTX2 pack.
//   build: ./build.sh retopo_bake
//   run:   ./retopo_bake <quad.obj> <out.glb> [texsize=2048] [small]
#include "tex_atlas.hpp"
#include "glb_packed.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <thread>

static std::vector<uint8_t> rd(const char* p){ FILE* f=fopen(p,"rb"); if(!f){printf("missing %s\n",p);exit(1);}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); std::vector<uint8_t> b(n); size_t r=fread(b.data(),1,n,f);(void)r; fclose(f); return b; }

int main(int argc, char** argv) {
    if (argc < 3) { printf("usage: retopo_bake quad.obj out.glb [texsize=2048] [small]\n"); return 1; }
    const char* objp = argv[1]; const char* out = argv[2];
    int TS = argc > 3 ? atoi(argv[3]) : 2048;
    bool uastc = !(argc > 4 && !strcmp(argv[4], "small"));

    // v6 parity bake flags (FINDINGS-A) — real xatlas pyref opts + topo normals + AA raster + no dark
    // fallback + full bg fill + TELEA. setenv(...,0) so an explicit env still overrides.
    auto def=[](const char*k,const char*v){ setenv(k,v,0); };
    def("ATL_PYREF_XATLAS","1"); def("TEX_RASTER_SS","2"); def("TEX_TOPO_NORMALS","1");
    def("TEX_KEEP_ATLAS_SIZE","1"); def("TEX_FILL_BACKGROUND","1");
    def("TEX_TELEA_INPAINT","1"); def("TEX_TELEA_RADIUS","4"); def("TEX_INPAINT_ITERS","16");

    // --- load QuadriFlow OBJ (verts + quads/tris → triangulated int64 faces) ---
    std::vector<float> verts; std::vector<int64_t> faces;
    { FILE* f=fopen(objp,"r"); if(!f){printf("open %s failed\n",objp);return 1;} char line[512];
      while(fgets(line,sizeof line,f)){
        if(line[0]=='v'&&line[1]==' '){ float x,y,z; sscanf(line+2,"%f %f %f",&x,&y,&z); verts.push_back(x);verts.push_back(y);verts.push_back(z); }
        else if(line[0]=='f'&&line[1]==' '){ long v[4]; int n=0; char*p=line+2;
          while(n<4){ while(*p==' ')p++; if(!*p||*p=='\n')break; v[n++]=atol(p); while(*p&&*p!=' '&&*p!='\n')p++; }
          if(n>=3){ faces.push_back(v[0]-1);faces.push_back(v[1]-1);faces.push_back(v[2]-1);
            if(n==4){ faces.push_back(v[0]-1);faces.push_back(v[2]-1);faces.push_back(v[3]-1); } } }
      } fclose(f); }
    printf("[retopo_bake] mesh: %zu verts %zu tris\n", verts.size()/3, faces.size()/3);

    // --- load cached PBR volume (dump_pbr_*.bin; NP from dump_bake.txt 3rd field) ---
    size_t NVq,NFq,NP; { FILE* f=fopen("dump_bake.txt","r"); if(!f){printf("no dump_bake.txt\n");return 1;}
        if(fscanf(f,"%zu %zu %zu",&NVq,&NFq,&NP)!=3){fclose(f);return 1;} fclose(f); }
    auto pf=rd("dump_pbr_f.bin"); auto pc=rd("dump_pbr_c.bin");
    std::vector<float>   pbr((float*)pf.data(),(float*)pf.data()+NP*6);
    std::vector<int32_t> coords((int32_t*)pc.data(),(int32_t*)pc.data()+NP*4);
    printf("[retopo_bake] PBR volume: %zu voxels @ grid1024\n", NP);

    // --- bake: real xatlas ComputeCharts (precluster=false) + direct volume grid_sample (reproject=false) ---
    double t0=texatlas::_now();
    texatlas::BakedTexture bt = texatlas::bake(verts, faces, pbr, coords, /*grid_res*/1024, TS,
        /*decimate*/0, /*padding*/4, /*verbose*/true, /*fallback_r*/0, /*precluster*/false);
    printf("[retopo_bake] baked %dx%d, %d charts, %zu out-verts (%.1fs)\n",
           bt.tw, bt.th, bt.chart_count, bt.verts.size()/3, texatlas::_now()-t0);

    int threads = (int)std::thread::hardware_concurrency();
    bool ok = glb::write_glb_textured_packed(out, bt.verts, bt.normals, bt.uvs, bt.faces,
                                             bt.base_color, bt.metal_rough, bt.tw, bt.th, uastc, 192, threads);
    printf("[retopo_bake] %s -> %s (%s)\n", ok?"wrote":"FAILED", out, uastc?"hero/UASTC":"small/ETC1S");
    return ok?0:1;
}
