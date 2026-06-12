// M6 tex decoder validation: svp::m6_tex_decode (out 6 PBR, guide_subs) vs the fp32 oracle
// (stage4_decode_capture.py: real tex_dec fp32 w/ spike conv). Input = golden tex_slat
// (stage4_out) + golden shape subs (stage5_mesh/sub{0..3}_feats>0) as guide_subs.
// Build: ./build.sh m6_tex_decode_test [cuda]
#include "sparse_vae_pipeline.hpp"
#include "../../sparse_spike/npy.hpp"
#include <cstdio>
#include <cmath>
#include <set>
#include <array>

static const char* WDIR = "weights_npy/tex_dec";
static const char* GOLD = "../../sparse_spike/golden_stages";
static const char* REFS = "refs/stage4";

typedef std::array<int32_t,4> C4;
static std::set<C4> cset(const int32_t* c, int N){ std::set<C4> s; for(int i=0;i<N;i++) s.insert({c[i*4],c[i*4+1],c[i*4+2],c[i*4+3]}); return s; }

int main(int argc, char** argv) {
    bool use_cuda = (argc>1 && std::string(argv[1])=="cuda");
    printf("[m6dec] backend=%s\n", use_cuda?"cuda(conv)":"cpu");

    NpyArray tcN = npy_load(std::string(GOLD)+"/stage4_out/tex_slat_coords.npy");
    NpyArray tfN = npy_load(std::string(GOLD)+"/stage4_out/tex_slat_feats.npy");
    int M=(int)tcN.shape[0];
    std::vector<int32_t> coords(tcN.i32(), tcN.i32()+(size_t)M*4);
    std::vector<float> tex_slat(tfN.f32(), tfN.f32()+(size_t)M*32);
    printf("[m6dec] tex_slat M=%d\n", M);

    // guide_subs = golden shape subs (logits > 0)
    std::vector<std::vector<uint8_t>> guide(4);
    for (int L=0;L<4;L++){
        NpyArray sf = npy_load(std::string(GOLD)+"/stage5_mesh/sub"+std::to_string(L)+"_feats.npy");
        guide[L].resize(sf.numel());
        for (int64_t i=0;i<sf.numel();i++) guide[L][i] = (sf.f32()[i] > 0.f) ? 1 : 0;
    }

    std::vector<int32_t> out_coords;
    std::vector<float> pbr = svp::m6_tex_decode(coords, tex_slat, guide, WDIR, use_cuda, &out_coords);
    int N=(int)pbr.size()/6;
    printf("[m6dec] PBR voxels N=%d\n", N);

    // vs oracle
    NpyArray opN = npy_load(std::string(REFS)+"/tex_pbr.npy");
    NpyArray ocN = npy_load(std::string(REFS)+"/tex_out_coords.npy");
    int oN=(int)opN.shape[0];
    printf("[m6dec] oracle N=%d\n", oN);
    // coord set IoU
    auto sm=cset(out_coords.data(),N), so=cset(ocN.i32(),(int)ocN.shape[0]);
    int inter=0; for(auto&c:sm) if(so.count(c)) inter++;
    printf("[m6dec] out_coords: mine=%d oracle=%d IoU=%.6f\n", N, oN, (double)inter/(sm.size()+so.size()-inter));
    // PBR compare (elementwise if counts match — coords in same order)
    bool ok=false;
    if (N==oN) {
        const float* o=opN.f32(); double ma=0,sum=0; size_t worst=0;
        for (size_t i=0;i<pbr.size();i++){ double d=std::fabs((double)pbr[i]-o[i]); if(d>ma){ma=d;worst=i;} sum+=d; }
        printf("[m6dec] PBR vs oracle: maxabs=%.3e meanabs=%.3e worst@%zu got=%.5f ref=%.5f\n",
               ma, sum/pbr.size(), worst, pbr[worst], o[worst]);
        // base_color range sanity
        double bmn=1e9,bmx=-1e9; for(int i=0;i<N;i++) for(int c=0;c<3;c++){ float v=pbr[(size_t)i*6+c]; bmn=std::min(bmn,(double)v); bmx=std::max(bmx,(double)v); }
        printf("[m6dec] base_color range [%.3f,%.3f]\n", bmn, bmx);
        ok = ma < (use_cuda ? 5e-2 : 5e-3);
    } else printf("[m6dec] COUNT MISMATCH\n");
    printf("[m6dec] %s\n", ok?"PASS (tex decoder == fp32 oracle)":"FAIL/CHECK");
    return ok?0:1;
}
