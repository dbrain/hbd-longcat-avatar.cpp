// Write a pixal3d PIXAL3D_DUMP_BAKE binary mesh dump (float32 verts + int64 faces) out as a GLB.
// Lets us feed the FULL dense pre-decimate shell (dump_dense_*, ~3.46M f) into the next stage (UltraShape)
// instead of pixal3d's default 138k decimate — whose sloppy-decimate defects otherwise leak downstream.
//   build: ./build.sh dump_to_glb
//   run:   ./dump_to_glb <verts.bin> <faces.bin> <NV> <NF> <out.glb>
//          (NV/NF from the matching dump_*.txt, e.g. `dump_to_glb dump_dense_v.bin dump_dense_f.bin $(cat dump_dense.txt) out.glb`)
#include "glb_writer.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstdint>

static std::vector<uint8_t> rd(const char* p){ FILE* f=fopen(p,"rb"); if(!f){printf("missing %s\n",p);exit(1);}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); std::vector<uint8_t> b(n);
    size_t r=fread(b.data(),1,n,f); (void)r; fclose(f); return b; }

int main(int argc,char**argv){
    if(argc<6){ printf("usage: dump_to_glb <verts.bin> <faces.bin> <NV> <NF> <out.glb>\n"); return 1; }
    size_t NV=strtoull(argv[3],0,10), NF=strtoull(argv[4],0,10);
    auto vb=rd(argv[1]); auto fb=rd(argv[2]);
    std::vector<float>   verts((float*)vb.data(),   (float*)vb.data()+NV*3);
    std::vector<int64_t> faces((int64_t*)fb.data(), (int64_t*)fb.data()+NF*3);
    if(!glb::write_glb(argv[5], verts, faces, nullptr)){ printf("write failed %s\n", argv[5]); return 1; }
    printf("[dump_to_glb] %zu v / %zu f -> %s\n", NV, NF, argv[5]);
    return 0;
}
