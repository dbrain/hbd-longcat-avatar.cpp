// Diagnostic: does recomputing the SAME DiT graph multiple times (the sampler pattern)
// stay correct? Builds the forward once, computes 3x with identical inputs, checks each
// output is finite, identical across runs, and matches dit_v.npy.
#include "ss_dit_graph.hpp"
#include <cmath>

static const char* FLOW_W = "weights_npy/ss_flow";
static const char* REFS = "refs";
static const int NEL = ssdit::SEQ * ssdit::INCH;

static int count_nan(const std::vector<float>& v) { int n=0; for(float x:v) if(std::isnan(x)||std::isinf(x)) n++; return n; }

int main(int argc, char** argv) {
    bool use_cuda = (argc > 1 && std::string(argv[1]) == "cuda");
    M1Harness Hf(FLOW_W, 512, use_cuda);
    ggml_context* cf = Hf.ctx;
    int64_t x_ne[4]={ssdit::SEQ,ssdit::INCH,1,1}; ggml_tensor* xin=Hf.input("x",2,x_ne);
    int64_t t_ne[4]={1,1,1,1}; ggml_tensor* tin=Hf.input("t",1,t_ne);
    int64_t g_ne[4]={1024,5,1,1}; ggml_tensor* gin=Hf.input("global",2,g_ne);
    int64_t p_ne[4]={1024,ssdit::SEQ,1,1}; ggml_tensor* pin=Hf.input("proj",2,p_ne);
    ggml_tensor* vout = ssdit::build_ss_dit_forward(cf, Hf, xin, tin, gin, pin);
    ggml_set_output(vout);
    ggml_cgraph* gff = new_graph(cf, 32768); ggml_build_forward_expand(gff, vout);
    Hf.alloc_and_upload(gff);

    NpyArray xN = npy_load(std::string(REFS)+"/dit_x.npy");
    std::vector<float> x(xN.f32(), xN.f32()+NEL);
    NpyArray gN = npy_load(std::string(REFS)+"/dino_global.npy");
    NpyArray pN = npy_load(std::string(REFS)+"/proj.npy");
    std::vector<float> cg(gN.f32(),gN.f32()+gN.numel()), cp(pN.f32(),pN.f32()+pN.numel());

    printf("[dit_repeat] backend=%s\n", use_cuda?"cuda":"cpu");
    std::vector<float> prev;
    for (int run=0; run<3; run++) {
        Hf.upload_input_raw(xin,x); std::vector<float> tv{537.f}; Hf.upload_input_raw(tin,tv);
        Hf.upload_input_raw(gin,cg); Hf.upload_input_raw(pin,cp);
        Hf.compute(gff);
        std::vector<float> v(NEL); ggml_backend_tensor_get(vout,v.data(),0,NEL*4);
        NpyArray ref = npy_load(std::string(REFS)+"/dit_v.npy");
        double ma=0; for(int i=0;i<NEL;i++) ma=std::max(ma,(double)std::fabs(v[i]-ref.f32()[i]));
        double dprev=0; if(!prev.empty()) for(int i=0;i<NEL;i++) dprev=std::max(dprev,(double)std::fabs(v[i]-prev[i]));
        printf("  run %d: nan/inf=%d  maxabs vs dit_v=%.3e  maxdiff vs prev=%.3e\n",
               run, count_nan(v), ma, run? dprev:0.0);
        prev = v;
    }
    return 0;
}
