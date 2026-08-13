// Textured glTF 2.0 binary (.glb) writer — UV-atlas PBR. Emits POSITION/NORMAL/TEXCOORD_0 +
// indices, two embedded PNG textures (baseColor RGBA + metallicRoughness RGB), samplers,
// pbrMetallicRoughness{baseColorTexture, metallicRoughnessTexture}. JSON built with nlohmann
// (thirdparty/json.hpp) since the textured glTF is far richer than the snprintf untextured writer.
// PNGs encoded in-memory via stb_image_write. Web-ready (<model-viewer>/three.js).
#pragma once
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../../thirdparty/stb_image_write.h"
#include "../../../thirdparty/json.hpp"
#include <cstdint>
#include <cstdio>
#include <vector>
#include <string>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include "../../../thirdparty/meshoptimizer/meshoptimizer.h"

namespace glb {

static inline void png_collect(void* ctx, void* data, int size) {
    auto* v = (std::vector<uint8_t>*)ctx;
    v->insert(v->end(), (uint8_t*)data, (uint8_t*)data + size);
}
// encode HWC uint8 image -> PNG bytes
static inline std::vector<uint8_t> encode_png(const uint8_t* data, int w, int h, int comp) {
    std::vector<uint8_t> out;
    stbi_write_png_to_func(png_collect, &out, w, h, comp, data, w*comp);
    return out;
}

// verts/normals [V*3], uvs [V*2] (normalized, glTF convention), faces uint32 [F*3].
// base_color RGBA8 [tw*th*4], metal_rough RGB8 [tw*th*3]. alphaMode OPAQUE (matches Python).
inline bool write_glb_textured(const char* path,
        const std::vector<float>& verts, const std::vector<float>& normals,
        const std::vector<float>& uvs, const std::vector<uint32_t>& faces,
        const std::vector<uint8_t>& base_color, const std::vector<uint8_t>& metal_rough, int tw, int th) {
    using json = nlohmann::json;
    const uint32_t V=(uint32_t)(verts.size()/3), F3=(uint32_t)faces.size();
    std::vector<uint8_t> png0 = encode_png(base_color.data(), tw, th, 4);
    std::vector<uint8_t> png1 = encode_png(metal_rough.data(), tw, th, 3);

    if (std::getenv("GLB_QUANTIZE")) {
        std::vector<int16_t> qpos((size_t)V*3);
        std::vector<int8_t> qnrm((size_t)V*3);
        std::vector<uint16_t> quv((size_t)V*2);

        float mn[3]={FLT_MAX,FLT_MAX,FLT_MAX}, mx[3]={-FLT_MAX,-FLT_MAX,-FLT_MAX};
        for (uint32_t i=0;i<V;i++) for(int d=0;d<3;d++){ float v=verts[(size_t)i*3+d]; if(v<mn[d])mn[d]=v; if(v>mx[d])mx[d]=v; }
        float center[3]={(mn[0]+mx[0])*0.5f,(mn[1]+mx[1])*0.5f,(mn[2]+mx[2])*0.5f};
        float radius=std::max({mx[0]-center[0], mx[1]-center[1], mx[2]-center[2], 1e-6f});
        float qmn[3]={FLT_MAX,FLT_MAX,FLT_MAX}, qmx[3]={-FLT_MAX,-FLT_MAX,-FLT_MAX};
        for (uint32_t i=0;i<V;i++) {
            for (int d=0;d<3;d++) {
                float v=(verts[(size_t)i*3+d]-center[d])/radius;
                v = std::max(-1.f, std::min(1.f, v));
                qpos[(size_t)i*3+d]=(int16_t)meshopt_quantizeSnorm(v, 16);
                if(v<qmn[d]) qmn[d]=v;
                if(v>qmx[d]) qmx[d]=v;
            }
            for (int d=0;d<3;d++) {
                float n=normals[(size_t)i*3+d];
                n = std::max(-1.f, std::min(1.f, n));
                qnrm[(size_t)i*3+d]=(int8_t)meshopt_quantizeSnorm(n, 8);
            }
            for (int d=0;d<2;d++) {
                float u=uvs[(size_t)i*2+d];
                u = std::max(0.f, std::min(1.f, u));
                quv[(size_t)i*2+d]=(uint16_t)meshopt_quantizeUnorm(u, 16);
            }
        }

        auto pad4=[](uint32_t n){ return (4 - (n & 3)) & 3; };
        const uint32_t POS=V*3*2, NRM=V*3, UV=V*2*2, IDX=F3*4;
        const uint32_t P0=(uint32_t)png0.size(), P1=(uint32_t)png1.size();
        uint32_t off=0;
        auto alloc=[&](uint32_t len){ uint32_t o=off; off+=len+pad4(len); return o; };
        const uint32_t POS_OFF=alloc(POS);
        const uint32_t NRM_OFF=alloc(NRM);
        const uint32_t UV_OFF=alloc(UV);
        const uint32_t IDX_OFF=alloc(IDX);
        const uint32_t P0_OFF=alloc(P0);
        const uint32_t P1_OFF=alloc(P1);
        const uint32_t BIN_LEN=off;

        json j;
        j["asset"]={{"version","2.0"},{"generator","pixal3d-cpp"}};
        j["extensionsUsed"]={"KHR_mesh_quantization"};
        j["scene"]=0; j["scenes"]={{{"nodes",{0}}}};
        j["nodes"]={{{"mesh",0},{"translation",{center[0],center[1],center[2]}},{"scale",{radius,radius,radius}}}};
        j["materials"]={{
            {"name","pbr"},{"doubleSided",true},{"alphaMode","OPAQUE"},
            {"pbrMetallicRoughness",{
                {"baseColorTexture",{{"index",0},{"texCoord",0}}},
                {"baseColorFactor",{1,1,1,1}},
                {"metallicRoughnessTexture",{{"index",1},{"texCoord",0}}},
                {"metallicFactor",1.0},{"roughnessFactor",1.0}
            }}
        }};
        j["meshes"]={{{"primitives",{{
            {"attributes",{{"POSITION",0},{"NORMAL",1},{"TEXCOORD_0",2}}},
            {"indices",3},{"material",0},{"mode",4}
        }}}}};
        int min_filter = std::getenv("GLB_MIPMAPS") ? 9987 : 9729;
        j["samplers"]={{{"magFilter",9729},{"minFilter",min_filter},{"wrapS",33071},{"wrapT",33071}}};
        j["textures"]={{{"source",0},{"sampler",0}},{{"source",1},{"sampler",0}}};
        j["images"]={{{"bufferView",4},{"mimeType","image/png"}},{{"bufferView",5},{"mimeType","image/png"}}};
        j["buffers"]={{{"byteLength",BIN_LEN}}};
        j["bufferViews"]={
            {{"buffer",0},{"byteOffset",POS_OFF},{"byteLength",POS},{"target",34962}},
            {{"buffer",0},{"byteOffset",NRM_OFF},{"byteLength",NRM},{"target",34962}},
            {{"buffer",0},{"byteOffset",UV_OFF}, {"byteLength",UV}, {"target",34962}},
            {{"buffer",0},{"byteOffset",IDX_OFF},{"byteLength",IDX},{"target",34963}},
            {{"buffer",0},{"byteOffset",P0_OFF},{"byteLength",P0}},
            {{"buffer",0},{"byteOffset",P1_OFF},{"byteLength",P1}}
        };
        j["accessors"]={
            {{"bufferView",0},{"componentType",5122},{"count",V},{"type","VEC3"},{"normalized",true},
             {"min",{qmn[0],qmn[1],qmn[2]}},{"max",{qmx[0],qmx[1],qmx[2]}}},
            {{"bufferView",1},{"componentType",5120},{"count",V},{"type","VEC3"},{"normalized",true}},
            {{"bufferView",2},{"componentType",5123},{"count",V},{"type","VEC2"},{"normalized",true}},
            {{"bufferView",3},{"componentType",5125},{"count",F3},{"type","SCALAR"}}
        };
        std::string js = j.dump();
        uint32_t json_len=(uint32_t)js.size(), json_pad=pad4(json_len), json_chunk=json_len+json_pad;
        uint32_t bin_pad=pad4(BIN_LEN), bin_chunk=BIN_LEN+bin_pad;
        uint32_t total=12+8+json_chunk+8+bin_chunk;

        FILE* f=std::fopen(path,"wb"); if(!f) return false;
        auto w32=[&](uint32_t v){ std::fwrite(&v,4,1,f); };
        auto wz=[&](uint32_t n){ for(uint32_t i=0;i<n;i++) std::fputc(0,f); };
        w32(0x46546C67u); w32(2u); w32(total);
        w32(json_chunk); w32(0x4E4F534Au);
        std::fwrite(js.data(),1,json_len,f); for(uint32_t i=0;i<json_pad;i++) std::fputc(' ',f);
        w32(bin_chunk); w32(0x004E4942u);
        std::fwrite(qpos.data(),1,POS,f); wz(pad4(POS));
        std::fwrite(qnrm.data(),1,NRM,f); wz(pad4(NRM));
        std::fwrite(quv.data(),1,UV,f); wz(pad4(UV));
        std::fwrite(faces.data(),1,IDX,f); wz(pad4(IDX));
        std::fwrite(png0.data(),1,P0,f); wz(pad4(P0));
        std::fwrite(png1.data(),1,P1,f); wz(pad4(P1));
        for(uint32_t i=0;i<bin_pad;i++) std::fputc(0,f);
        std::fclose(f);
        return true;
    }

    auto pad4=[](uint32_t n){ return (4 - (n & 3)) & 3; };
    const uint32_t POS=V*3*4, NRM=V*3*4, UV=V*2*4, IDX=F3*4;
    const uint32_t P0=(uint32_t)png0.size(), P1=(uint32_t)png1.size();
    const uint32_t P0pad=pad4(P0), P1pad=pad4(P1);
    uint32_t off=0;
    const uint32_t POS_OFF=off; off+=POS;
    const uint32_t NRM_OFF=off; off+=NRM;
    const uint32_t UV_OFF=off;  off+=UV;
    const uint32_t IDX_OFF=off; off+=IDX;
    const uint32_t P0_OFF=off;  off+=P0+P0pad;
    const uint32_t P1_OFF=off;  off+=P1+P1pad;
    const uint32_t BIN_LEN=off;

    float mn[3]={FLT_MAX,FLT_MAX,FLT_MAX}, mx[3]={-FLT_MAX,-FLT_MAX,-FLT_MAX};
    for (uint32_t i=0;i<V;i++) for(int d=0;d<3;d++){ float v=verts[(size_t)i*3+d]; if(v<mn[d])mn[d]=v; if(v>mx[d])mx[d]=v; }

    json j;
    j["asset"]={{"version","2.0"},{"generator","pixal3d-cpp"}};
    j["scene"]=0; j["scenes"]={{{"nodes",{0}}}}; j["nodes"]={{{"mesh",0}}};
    j["materials"]={{
        {"name","pbr"},{"doubleSided",true},{"alphaMode","OPAQUE"},
        {"pbrMetallicRoughness",{
            {"baseColorTexture",{{"index",0},{"texCoord",0}}},
            {"baseColorFactor",{1,1,1,1}},
            {"metallicRoughnessTexture",{{"index",1},{"texCoord",0}}},
            {"metallicFactor",1.0},{"roughnessFactor",1.0}
        }}
    }};
    j["meshes"]={{{"primitives",{{
        {"attributes",{{"POSITION",0},{"NORMAL",1},{"TEXCOORD_0",2}}},
        {"indices",3},{"material",0},{"mode",4}
    }}}}};
    int min_filter = std::getenv("GLB_MIPMAPS") ? 9987 : 9729;
    j["samplers"]={{{"magFilter",9729},{"minFilter",min_filter},{"wrapS",33071},{"wrapT",33071}}};
    j["textures"]={{{"source",0},{"sampler",0}},{{"source",1},{"sampler",0}}};
    j["images"]={{{"bufferView",4},{"mimeType","image/png"}},{{"bufferView",5},{"mimeType","image/png"}}};
    j["buffers"]={{{"byteLength",BIN_LEN}}};
    j["bufferViews"]={
        {{"buffer",0},{"byteOffset",POS_OFF},{"byteLength",POS},{"target",34962}},
        {{"buffer",0},{"byteOffset",NRM_OFF},{"byteLength",NRM},{"target",34962}},
        {{"buffer",0},{"byteOffset",UV_OFF}, {"byteLength",UV}, {"target",34962}},
        {{"buffer",0},{"byteOffset",IDX_OFF},{"byteLength",IDX},{"target",34963}},
        {{"buffer",0},{"byteOffset",P0_OFF},{"byteLength",P0}},
        {{"buffer",0},{"byteOffset",P1_OFF},{"byteLength",P1}}
    };
    j["accessors"]={
        {{"bufferView",0},{"componentType",5126},{"count",V},{"type","VEC3"},
         {"min",{mn[0],mn[1],mn[2]}},{"max",{mx[0],mx[1],mx[2]}}},
        {{"bufferView",1},{"componentType",5126},{"count",V},{"type","VEC3"}},
        {{"bufferView",2},{"componentType",5126},{"count",V},{"type","VEC2"}},
        {{"bufferView",3},{"componentType",5125},{"count",F3},{"type","SCALAR"}}
    };
    std::string js = j.dump();
    uint32_t json_len=(uint32_t)js.size(), json_pad=pad4(json_len), json_chunk=json_len+json_pad;
    uint32_t bin_pad=pad4(BIN_LEN), bin_chunk=BIN_LEN+bin_pad;
    uint32_t total=12+8+json_chunk+8+bin_chunk;

    FILE* f=std::fopen(path,"wb"); if(!f) return false;
    auto w32=[&](uint32_t v){ std::fwrite(&v,4,1,f); };
    w32(0x46546C67u); w32(2u); w32(total);
    w32(json_chunk); w32(0x4E4F534Au);                  // "JSON"
    std::fwrite(js.data(),1,json_len,f); for(uint32_t i=0;i<json_pad;i++) std::fputc(' ',f);
    w32(bin_chunk); w32(0x004E4942u);                   // "BIN\0"
    std::fwrite(verts.data(),4,(size_t)V*3,f);
    std::fwrite(normals.data(),4,(size_t)V*3,f);
    std::fwrite(uvs.data(),4,(size_t)V*2,f);
    std::fwrite(faces.data(),4,F3,f);
    std::fwrite(png0.data(),1,P0,f); for(uint32_t i=0;i<P0pad;i++) std::fputc(0,f);
    std::fwrite(png1.data(),1,P1,f); for(uint32_t i=0;i<P1pad;i++) std::fputc(0,f);
    for(uint32_t i=0;i<bin_pad;i++) std::fputc(0,f);
    std::fclose(f);
    return true;
}

}  // namespace glb
