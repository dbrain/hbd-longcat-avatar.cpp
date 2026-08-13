// Tiny utility: read a binary-LE PLY (verts f32 + faces uchar-list int) -> web-ready GLB
// via glb_writer.hpp (normals + double-sided material). Also validates the C++ glb_writer
// without re-running the full chain.  Build: g++ -O2 -std=c++17 ply2glb.cpp -o ply2glb
#include "glb_writer.hpp"
#include <cstdio>
#include <cstdint>
#include <vector>
#include <string>
#include <cstring>

int main(int argc, char** argv) {
    if (argc < 3) { printf("usage: ply2glb in.ply out.glb\n"); return 1; }
    FILE* f = std::fopen(argv[1], "rb");
    if (!f) { printf("cannot open %s\n", argv[1]); return 1; }
    std::fseek(f, 0, SEEK_END); long sz = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf(sz); if (std::fread(buf.data(),1,sz,f)!=(size_t)sz){return 1;} std::fclose(f);
    // parse header
    const char* hdr = "end_header\n";
    uint8_t* he = (uint8_t*)memmem(buf.data(), sz, hdr, strlen(hdr));
    if (!he) { printf("no end_header\n"); return 1; }
    size_t off = (he - buf.data()) + strlen(hdr);
    std::string head((char*)buf.data(), off);
    int nv=0, nf=0;
    { size_t p=head.find("element vertex"); if(p!=std::string::npos) nv=atoi(head.c_str()+p+14);
      size_t q=head.find("element face");   if(q!=std::string::npos) nf=atoi(head.c_str()+q+12); }
    printf("ply: %d verts %d faces\n", nv, nf);
    std::vector<float> verts((size_t)nv*3);
    memcpy(verts.data(), buf.data()+off, (size_t)nv*3*4);
    off += (size_t)nv*3*4;
    std::vector<int64_t> faces((size_t)nf*3);
    for (int i=0;i<nf;i++) {
        uint8_t cnt = buf[off]; off+=1;                  // list count (uchar) = 3
        for (int k=0;k<(int)cnt;k++){ int32_t idx; memcpy(&idx, buf.data()+off, 4); off+=4;
            if (k<3) faces[(size_t)i*3+k]=idx; }
    }
    bool ok = glb::write_glb(argv[2], verts, faces);
    printf("%s %s\n", ok?"wrote":"FAILED", argv[2]);
    return ok?0:1;
}
