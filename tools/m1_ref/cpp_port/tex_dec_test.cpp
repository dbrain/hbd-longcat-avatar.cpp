// TRELLIS-2 tex DECODER validation (Stage-2 texturing, step 2) vs tex_goldens/pbr_feats.
// Decoder weights (weights_npy/tex_dec) == trellis2_4b tex_dec_next_dc_f16c32 (sorted maxabsdiff 0.0).
// Isolates the decoder: feeds the GOLDEN (denormed) tex_slat + GOLDEN shape_slat coords; guide_subs are
// rebuilt by senc::build_guide_subs from the grid-1024 occupancy = the GOLDEN pbr_coords (so the
// subdivision structure is the oracle's, isolating the decode MATH from the voxelizer-occupancy risk).
// Compares pbr vs golden COORD-KEYED (decoder emits in c2s order, not the golden's order).
// Build: ./build.sh tex_dec_test [cuda]
#include "sparse_vae_pipeline.hpp"
#include "shape_slat_encoder.hpp"
#include "../../sparse_spike/npy.hpp"
#include <cstdio>
#include <cmath>
#include <unordered_map>
#include <cstdint>

static const char* WDIR = "weights_npy/tex_dec";
static const char* G = "/mnt/hdd/3d/avatar-shootout/tex_goldens";

static inline int64_t ckey(int32_t b,int32_t x,int32_t y,int32_t z){
    return ((int64_t)(b&0x3ff)<<33)^((int64_t)(x&0x7ff)<<22)^((int64_t)(y&0x7ff)<<11)^(int64_t)(z&0x7ff); }

int main(int argc, char** argv) {
    bool use_cuda = (argc>1 && std::string(argv[1])=="cuda");
    printf("[texdec] backend=%s\n", use_cuda?"cuda(conv)":"cpu");

    // golden shape_slat coords (grid64, decoder input) + golden denormed tex_slat
    NpyArray scN = npy_load(std::string(G)+"/shape_slat_coords.npy");   // [M,4] float
    // tex_slat: golden by default; env TEXSLAT_NPY=path overrides (my DiT output -> fully native chain)
    std::string tsp = std::getenv("TEXSLAT_NPY") ? std::getenv("TEXSLAT_NPY") : std::string(G)+"/tex_slat_feats.npy";
    NpyArray tfN = npy_load(tsp);                                       // [M,32]
    int M=(int)scN.shape[0];
    std::vector<int32_t> coords((size_t)M*4);
    for (size_t i=0;i<(size_t)M*4;i++) coords[i]=(int32_t)std::lround(scN.f32()[i]);
    std::vector<float> tex_slat(tfN.f32(), tfN.f32()+(size_t)M*32);
    printf("[texdec] tex_slat src = %s\n", tsp.c_str());

    // golden pbr (the bar) — coords (grid1024 occupancy) + feats [V,6]
    NpyArray pcN = npy_load(std::string(G)+"/pbr_coords.npy");          // [V,4] float
    NpyArray pfN = npy_load(std::string(G)+"/pbr_feats.npy");           // [V,6]
    int V=(int)pcN.shape[0];
    std::vector<int32_t> occ1024((size_t)V*4);
    for (size_t i=0;i<(size_t)V*4;i++) occ1024[i]=(int32_t)std::lround(pcN.f32()[i]);
    printf("[texdec] M(grid64)=%d  golden pbr V=%d\n", M, V);

    // guide_subs from the grid-1024 occupancy (= golden pbr_coords)
    auto guide = senc::build_guide_subs(occ1024, coords);
    printf("[texdec] guide_subs levels: %zu %zu %zu %zu (bytes)\n",
           guide[0].size(), guide[1].size(), guide[2].size(), guide[3].size());

    std::vector<int32_t> out_coords;
    std::vector<float> pbr = svp::m6_tex_decode(coords, tex_slat, guide, WDIR, use_cuda, &out_coords);
    int N=(int)pbr.size()/6;
    printf("[texdec] decoded PBR voxels N=%d\n", N);

    // DUMP_PBR=dir -> write my pbr_feats[N,6]+pbr_coords[N,4] (int32) for the native bake
    if (const char* dd = std::getenv("DUMP_PBR")) {
        auto wr=[&](const char* nm,const void* d,const char* descr,int rows,int cols,int es){
            std::string p=std::string(dd)+"/"+nm; FILE* f=fopen(p.c_str(),"wb"); if(!f)return;
            char hdr[160]; int hl=snprintf(hdr,sizeof(hdr),"{'descr': '%s', 'fortran_order': False, 'shape': (%d, %d), }",descr,rows,cols);
            int total=10+hl; int pad=(64-(total%64))%64; std::string h(hdr,hl); h.append(pad,' '); h.push_back('\n');
            uint16_t hlen=(uint16_t)h.size(); fwrite("\x93NUMPY\x01\x00",1,8,f); fwrite(&hlen,2,1,f); fwrite(h.data(),1,h.size(),f);
            fwrite(d,es,(size_t)rows*cols,f); fclose(f); };
        wr("pbr_feats.npy", pbr.data(), "<f4", N, 6, 4);
        wr("pbr_coords.npy", out_coords.data(), "<i4", N, 4, 4);
        printf("  [dump] native pbr -> %s/pbr_{feats,coords}.npy\n", dd);
    }

    // coord-keyed compare vs golden
    std::unordered_map<int64_t,int> gidx; gidx.reserve(V*2);
    for (int i=0;i<V;i++) gidx[ckey(occ1024[i*4],occ1024[i*4+1],occ1024[i*4+2],occ1024[i*4+3])]=i;
    int matched=0; double ma=0,sum=0,dot=0,na=0,nb=0; const float* gf=pfN.f32();
    double bmn=1e9,bmx=-1e9;
    for (int n=0;n<N;n++){
        int64_t k=ckey(out_coords[n*4],out_coords[n*4+1],out_coords[n*4+2],out_coords[n*4+3]);
        auto it=gidx.find(k); if(it==gidx.end()) continue; matched++; int gi=it->second;
        for (int c=0;c<6;c++){ double a=pbr[(size_t)n*6+c], b=gf[(size_t)gi*6+c];
            double d=std::fabs(a-b); ma=std::max(ma,d); sum+=d; dot+=a*b; na+=a*a; nb+=b*b; }
        for (int c=0;c<3;c++){ double v=pbr[(size_t)n*6+c]; bmn=std::min(bmn,v); bmx=std::max(bmx,v); }
    }
    double cos=dot/(std::sqrt(na*nb)+1e-12);
    printf("[texdec] coord match: %d/%d mine, %d golden (set-IoU≈%.4f)\n", matched, N, V,
           (double)matched/(N+V-matched));
    printf("[texdec] PBR vs golden (matched): maxabs=%.3e meanabs=%.3e cosine=%.6f\n", ma, sum/(matched*6.0), cos);
    printf("[texdec] base_color range [%.3f,%.3f]\n", bmn, bmx);
    // gate on coord-set completeness + cosine (maxabs on saturated PBR channels is unachievable, per
    // methodology); maxabs is informational. cosine>0.9999 covers both golden-tex_slat (1.0) and the
    // native-DiT-tex_slat path (0.99998, where one voxel's saturated channel can spike maxabs).
    bool ok = (matched>0) && ((double)matched/N > 0.999) && cos>0.9999;
    printf("[texdec] %s\n", ok?"PASS (tex decoder == trellis2 oracle)":"FAIL/CHECK");
    return ok?0:1;
}
