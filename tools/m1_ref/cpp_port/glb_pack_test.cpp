// glb_pack_test — validate glb_packed.hpp end-to-end: write a compressed textured GLB (meshopt + KTX2)
// for a small mesh, then re-read it and meshopt-DECODE the streams to prove the layout/offsets are valid.
//   ./build.sh glb_pack_test && ./glb_pack_test
#include "glb_packed.hpp"
#include "../../../thirdparty/json.hpp"
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    // unit cube: 8 verts, 12 tris
    std::vector<float> verts = {-1,-1,-1, 1,-1,-1, 1,1,-1, -1,1,-1, -1,-1,1, 1,-1,1, 1,1,1, -1,1,1};
    std::vector<float> normals; for (int i=0;i<8;i++){ float l=1.f/std::sqrt(3.f); normals.insert(normals.end(),{verts[i*3]*l,verts[i*3+1]*l,verts[i*3+2]*l}); }
    std::vector<float> uvs = {0,0, 1,0, 1,1, 0,1, 0,0, 1,0, 1,1, 0,1};
    std::vector<uint32_t> faces = {0,1,2, 0,2,3, 4,6,5, 4,7,6, 0,4,5, 0,5,1, 1,5,6, 1,6,2, 2,6,7, 2,7,3, 3,7,4, 3,4,0};
    const int TW=64, TH=64;
    std::vector<uint8_t> base(TW*TH*4), mr(TW*TH*3);
    for (int y=0;y<TH;y++) for (int x=0;x<TW;x++){ int i=y*TW+x; base[i*4]=x*4; base[i*4+1]=y*4; base[i*4+2]=128; base[i*4+3]=255; mr[i*3]=0; mr[i*3+1]=200; mr[i*3+2]=30; }

    const char* OUT="glb_pack_test.glb";
    if (!glb::write_glb_textured_packed(OUT, verts, normals, uvs, faces, base, mr, TW, TH, /*uastc*/true)) { printf("WRITE FAILED\n"); return 1; }

    // re-read + validate
    FILE* f=fopen(OUT,"rb"); fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    std::vector<uint8_t> d(n); size_t rd=fread(d.data(),1,n,f); (void)rd; fclose(f);
    uint32_t jlen; memcpy(&jlen,&d[12],4);
    auto j=nlohmann::json::parse(d.begin()+20, d.begin()+20+jlen);
    uint32_t bin_off = 20 + jlen + (4-(jlen&3))%4 + 8;  // skip JSON chunk + pad + BIN chunk header

    int ktx=0;
    static const uint8_t MAGIC[12]={0xAB,0x4B,0x54,0x58,0x20,0x32,0x30,0xBB,0x0D,0x0A,0x1A,0x0A};
    for (size_t i=0;i+12<=d.size();i++){ if(!memcmp(&d[i],MAGIC,12)) ktx++; }

    auto& bvs=j["bufferViews"];
    auto decode_ext=[&](int bv)->std::vector<uint8_t>{
        auto e=bvs[bv]["extensions"]["EXT_meshopt_compression"];
        uint32_t bo=e["byteOffset"], bl=e["byteLength"];
        return std::vector<uint8_t>(d.begin()+bin_off+bo, d.begin()+bin_off+bo+bl);
    };
    // indices: bufferView 5, decode TRIANGLES
    uint32_t F3=j["accessors"][3]["count"]; uint32_t V=j["accessors"][0]["count"];
    auto cidx=decode_ext(5); std::vector<uint32_t> didx(F3);
    int ri=meshopt_decodeIndexBuffer(didx.data(), F3, 4, cidx.data(), cidx.size());
    // meshopt's index codec preserves triangles up to CYCLIC ROTATION (winding kept), so compare each
    // tri as a rotation-equivalent triple, not element-wise.
    bool idx_ok = (ri==0);
    for (uint32_t t=0; t<F3 && idx_ok; t+=3){
        uint32_t a=faces[t],b=faces[t+1],c=faces[t+2], x=didx[t],y=didx[t+1],z=didx[t+2];
        idx_ok = (x==a&&y==b&&z==c)||(x==b&&y==c&&z==a)||(x==c&&y==a&&z==b);
    }
    // positions: bufferView 2, decode ATTRIBUTES stride 8
    auto cpos=decode_ext(2); std::vector<uint8_t> dpos((size_t)V*8);
    int rp=meshopt_decodeVertexBuffer(dpos.data(), V, 8, cpos.data(), cpos.size());
    bool pos_ok = (rp==0);
    // normals: bufferView 3, decode ATTRIBUTES stride 4 then OCTAHEDRAL filter -> int8 xyz, compare ~original
    auto cnrm=decode_ext(3); std::vector<uint8_t> dnrm((size_t)V*4);
    int rn=meshopt_decodeVertexBuffer(dnrm.data(), V, 4, cnrm.data(), cnrm.size());
    meshopt_decodeFilterOct(dnrm.data(), V, 4);
    bool nrm_ok = (rn==0);
    for (uint32_t i=0;i<V && nrm_ok;i++){
        int8_t* n=(int8_t*)&dnrm[i*4];
        for (int d=0;d<3;d++){ float got=n[d]/127.f, want=normals[i*3+d]; if (std::fabs(got-want)>0.06f) nrm_ok=false; }
    }

    bool ext_ok = j["extensionsRequired"].size()==3;
    printf("[glb_pack_test] V=%u F3=%u  ktx2_payloads=%d  idx_roundtrip=%s  pos_decode=%s  nrm_oct=%s  ext=%s  size=%ld\n",
           V, F3, ktx, idx_ok?"OK":"BAD", pos_ok?"OK":"BAD", nrm_ok?"OK":"BAD", ext_ok?"OK":"BAD", n);
    bool ok = (ktx==2) && idx_ok && pos_ok && nrm_ok && ext_ok;
    printf("[glb_pack_test] %s\n", ok?"ALL VALID":"FAILED");
    return ok?0:1;
}
