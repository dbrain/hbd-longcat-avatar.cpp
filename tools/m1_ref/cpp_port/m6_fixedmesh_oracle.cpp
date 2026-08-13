// Exact-boundary M6 sampler oracle for the frozen fixed-mesh diagnostic.
// Reads official Python shape latent, projection condition and texture noise; it does not encode
// a mesh or generate geometry. Build: ./build.sh m6_fixedmesh_oracle cuda
#include "slat_dit_graph.hpp"
#include "geometry_e2e.hpp"
#include <cstdio>
#include <cmath>

static void report_trace(const char* tag, int step, const std::vector<float>& actual,
                         const std::string& root, const char* name) {
    char file[64]; std::snprintf(file, sizeof(file), "%s_%02d.npy", name, step);
    NpyArray ref = npy_load(root + "/" + file);
    if (ref.numel() != (int64_t)actual.size()) throw std::runtime_error("M6 trace shape mismatch: " + std::string(file));
    double mae=0, rmse=0, mx=0, dot=0, aa=0, bb=0;
    for (size_t i=0;i<actual.size();++i) { const double a=actual[i], b=ref.f32()[i], e=a-b;
        mae+=std::fabs(e); rmse+=e*e; mx=std::max(mx,std::fabs(e)); dot+=a*b; aa+=a*a; bb+=b*b; }
    const double n=actual.size();
    std::printf("M6_TRACE %s step=%02d mae=%.9g rmse=%.9g maxabs=%.9g cosine=%.9g\n",
                tag,step,mae/n,std::sqrt(rmse/n),mx,dot/(std::sqrt(aa*bb)+1e-30));
}

static const char* W="weights_npy/slat_flow_imgshape2tex_1024";
static const char* R="refs";
int main(int argc,char**argv){
    // The official fixed-input capture disables TF32.  Keep this standalone
    // oracle on the same math mode, independent of its caller's environment.
    setenv("NVIDIA_TF32_OVERRIDE", "0", 1);
    // Strict M6 parity must never inherit model-family experiments from a
    // shell that was used for LTX/LongCat work.  These are presence-based or
    // cached by ggml, so setting them to "0" is not a safe substitute.
    const bool bf16_tensorcore_f32_accum = std::getenv("PIXAL3D_BF16_TENSORCORE_F32_ACCUM") != nullptr;
    for (const char * key : {"PIXAL3D_FAST", "PIXAL3D_FLASH", "USR_GEO_FLASH",
                             "GGML_CUDNN_ATTN", "GGML_LTX_SA3", "PIXAL3D_FA_WMMA",
                             "PIXAL3D_FA_TILE", "PIXAL3D_FA_VEC", "LONGCAT_FA_NCOLS1",
                             "GGML_CUDA_BF16_F32_OUT", "GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F",
                             "GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F"}) unsetenv(key);
    // PyTorch's BF16 Linear preserves the tensor-core FP32 accumulator through
    // the bias add, then rounds the module output once.  The normal ggml BF16
    // cuBLAS route stores the product to BF16 first, which makes this a
    // deliberately isolated parity A/B (not a production default).
    if (bf16_tensorcore_f32_accum) setenv("GGML_CUDA_BF16_F32_OUT", "1", 1);
    bool cuda=argc>1 && std::string(argv[1])=="cuda";
    const char* root=std::getenv("M6_FIXEDMESH_ORACLE_DIR");
    if(!root||!*root){ std::fprintf(stderr,"set M6_FIXEDMESH_ORACLE_DIR to the official Python oracle directory\n"); return 2; }
    const std::string d=root;
    const char* trace_root=std::getenv("M6_FIXEDMESH_TRACE_DIR");
    NpyArray c=npy_load(d+"/shape_slat_coords.npy"), s=npy_load(d+"/shape_slat_feats.npy"),
             g=npy_load(d+"/cond_global.npy"), p=npy_load(d+"/cond_proj.npy"),
             nz=npy_load(d+"/tex_noise.npy"), ref=npy_load(d+"/tex_slat.npy"),
             mean=npy_load(std::string(R)+"/shape_slat_norm_mean.npy"), stdv=npy_load(std::string(R)+"/shape_slat_norm_std.npy"),
             tmean=npy_load(std::string(R)+"/tex_slat_norm_mean.npy"), tstd=npy_load(std::string(R)+"/tex_slat_norm_std.npy");
    int M=(int)c.shape[0], N=M*32; if(s.numel()!=N||p.numel()!=M*2048||g.numel()!=5*1024||nz.numel()!=N||ref.numel()!=N){std::fprintf(stderr,"invalid oracle dimensions M=%d\n",M);return 2;}
    std::vector<int32_t> xyz((size_t)M*3); for(int i=0;i<M;i++)for(int j=0;j<3;j++)xyz[(size_t)i*3+j]=c.i32()[(size_t)i*4+j+1];
    std::vector<float> shape((size_t)N), noise(nz.f32(),nz.f32()+N), cg(g.f32(),g.f32()+g.numel()), cp(p.f32(),p.f32()+p.numel());
    for(int i=0;i<N;i++)shape[i]=(s.f32()[i]-mean.f32()[i%32])/stdv.f32()[i%32];
    M1Harness h(W,2048,cuda); ggml_context* ctx=h.ctx;
    int64_t xn[4]={64,M,1,1},tn[4]={1,1,1,1},gn[4]={1024,5,1,1},pn[4]={2048,M,1,1};
    auto*x=h.input("x",2,xn);auto*t=h.input("t",1,tn);auto*gi=h.input("global",2,gn);auto*pi=h.input("proj",2,pn);
    ggml_tensor* block0=nullptr; ggml_tensor* input_layer=nullptr; ggml_tensor* self_attn0=nullptr; ggml_tensor* self_attn0_preout=nullptr; ggml_tensor* self_attn0_qkv=nullptr; ggml_tensor* self_attn0_qkv_input=nullptr; ggml_tensor* tmod=nullptr; ggml_tensor* self_attn0_norm1=nullptr; ggml_tensor* cross_attn0=nullptr; ggml_tensor* proj_linear0=nullptr; ggml_tensor* mlp0=nullptr; ggml_tensor* mlp0_input=nullptr; ggml_tensor* mlp0_linear0=nullptr; ggml_tensor* mlp0_hidden=nullptr; ggml_tensor* cross_attn0_preout=nullptr;
    auto*out=slatdit::build_slat_dit_forward(ctx,h,M,x,t,gi,pi,xyz.data(),&block0,&input_layer,&self_attn0,&self_attn0_preout,&self_attn0_qkv,&self_attn0_qkv_input,&tmod,&self_attn0_norm1,&cross_attn0,&proj_linear0,&mlp0,&mlp0_input,&mlp0_linear0,&mlp0_hidden,&cross_attn0_preout);ggml_set_output(out);auto*gf=new_graph(ctx,32768);ggml_build_forward_expand(gf,out);h.alloc_and_upload(gf);
    int trace_step=0;
    std::vector<float>x64((size_t)M*64);auto forward=[&](const std::vector<float>& q,float ts,bool){
        if(trace_root&&*trace_root) report_trace("input",trace_step,q,trace_root,"python_m6_x");
        for(int i=0;i<M;i++)for(int j=0;j<32;j++){x64[(size_t)i*64+j]=q[(size_t)i*32+j];x64[(size_t)i*64+32+j]=shape[(size_t)i*32+j];}
        h.upload_input_raw(x,x64);h.upload_input_raw(t,std::vector<float>{ts});h.upload_input_raw(gi,cg);h.upload_input_raw(pi,cp);h.compute(gf);
        if(trace_root&&*trace_root&&trace_step==0) { std::vector<float>embed((size_t)M*slatdit::C); ggml_backend_tensor_get(input_layer,embed.data(),0,embed.size()*4); report_trace("torso_input",0,embed,trace_root,"python_m6_torso_input_step"); std::vector<float>tm(6*slatdit::C); ggml_backend_tensor_get(tmod,tm.data(),0,tm.size()*4); report_trace("tmod",0,tm,trace_root,"python_m6_tmod_step"); std::vector<float>n1((size_t)M*slatdit::C); ggml_backend_tensor_get(self_attn0_norm1,n1.data(),0,n1.size()*4); report_trace("self_attn_norm1",0,n1,trace_root,"python_m6_self_attn_norm1_step"); std::vector<float>qkvin((size_t)M*slatdit::C); ggml_backend_tensor_get(self_attn0_qkv_input,qkvin.data(),0,qkvin.size()*4); report_trace("self_attn_qkv_input",0,qkvin,trace_root,"python_m6_self_attn_qkv_input_step"); std::vector<float>qkv((size_t)M*slatdit::C*3); ggml_backend_tensor_get(self_attn0_qkv,qkv.data(),0,qkv.size()*4); report_trace("self_attn_qkv",0,qkv,trace_root,"python_m6_self_attn_qkv_step"); if (std::getenv("PIXAL3D_FA_CAPTURE")) for (const char *nm : {"cap_q", "cap_k", "cap_v"}) { ggml_tensor *cap = ggml_get_tensor(ctx, nm); if (!cap) throw std::runtime_error(std::string("missing ") + nm); std::vector<float>cv((size_t)ggml_nelements(cap)); ggml_backend_tensor_get(cap, cv.data(), 0, cv.size() * sizeof(float)); report_trace(nm, 0, cv, trace_root, (std::string("python_m6_self_attn_sdpa_") + (nm + 4) + "_step").c_str()); } std::vector<float>pre((size_t)M*slatdit::C); ggml_backend_tensor_get(self_attn0_preout,pre.data(),0,pre.size()*4); report_trace("self_attn_preout",0,pre,trace_root,"python_m6_self_attn_preout_step"); std::vector<float>self((size_t)M*slatdit::C); ggml_backend_tensor_get(self_attn0,self.data(),0,self.size()*4); report_trace("self_attn00",0,self,trace_root,"python_m6_self_attn_00_step"); std::vector<float>crosspre((size_t)M*slatdit::C); ggml_backend_tensor_get(cross_attn0_preout,crosspre.data(),0,crosspre.size()*4); report_trace("cross_attn_preout",0,crosspre,trace_root,"python_m6_cross_attn_preout_step"); std::vector<float>cross((size_t)M*slatdit::C); ggml_backend_tensor_get(cross_attn0,cross.data(),0,cross.size()*4); report_trace("cross_attn00",0,cross,trace_root,"python_m6_cross_attn_00_step"); std::vector<float>pj((size_t)M*slatdit::C); ggml_backend_tensor_get(proj_linear0,pj.data(),0,pj.size()*4); report_trace("proj_linear00",0,pj,trace_root,"python_m6_proj_linear_00_step"); std::vector<float>mlpin((size_t)M*slatdit::C); ggml_backend_tensor_get(mlp0_input,mlpin.data(),0,mlpin.size()*4); report_trace("mlp00_input",0,mlpin,trace_root,"python_m6_mlp_00_input_step"); std::vector<float>mlp0lin((size_t)ggml_nelements(mlp0_linear0)); ggml_backend_tensor_get(mlp0_linear0,mlp0lin.data(),0,mlp0lin.size()*4); report_trace("mlp00_linear0",0,mlp0lin,trace_root,"python_m6_mlp_00_linear0_step"); std::vector<float>mlph((size_t)ggml_nelements(mlp0_hidden)); ggml_backend_tensor_get(mlp0_hidden,mlph.data(),0,mlph.size()*4); report_trace("mlp00_hidden",0,mlph,trace_root,"python_m6_mlp_00_hidden_step"); std::vector<float>mlp((size_t)M*slatdit::C); ggml_backend_tensor_get(mlp0,mlp.data(),0,mlp.size()*4); report_trace("mlp00",0,mlp,trace_root,"python_m6_mlp_00_step"); std::vector<float>b0((size_t)M*slatdit::C); ggml_backend_tensor_get(block0,b0.data(),0,b0.size()*4); report_trace("block00",0,b0,trace_root,"python_m6_block_00_step"); }
        std::vector<float>v(N);ggml_backend_tensor_get(out,v.data(),0,(size_t)N*4);
        if(trace_root&&*trace_root) report_trace("pred_v",trace_step,v,trace_root,"python_m6_pred_v");
        ++trace_step; return v;
    };
    auto raw=geo::flow_sampler(N,noise,1e-5f,1.f,0.f,3.0,0.6,0.9,12,forward,"m6-fixed-oracle");
    double mae=0,rmse=0,mx=0,dot=0,aa=0,bb=0;for(int i=0;i<N;i++){double a=raw[i]*tstd.f32()[i%32]+tmean.f32()[i%32],b=ref.f32()[i],e=a-b;mae+=fabs(e);rmse+=e*e;mx=std::max(mx,fabs(e));dot+=a*b;aa+=a*a;bb+=b*b;}mae/=N;rmse=sqrt(rmse/N);double cos=dot/(sqrt(aa*bb)+1e-30);std::printf("M6_FIXEDMESH_ORACLE M=%d mae=%.9g rmse=%.9g maxabs=%.9g cosine=%.9g\n",M,mae,rmse,mx,cos);return (mae<0.01&&cos>0.999)?0:1;
}
