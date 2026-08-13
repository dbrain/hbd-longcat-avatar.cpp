// Full UltraShape refine e2e (native sampling orchestration) — validates the glue that ties the
// already-proven pieces together: flow-matching Euler sampling loop (50 steps, CFG g=5) over
// us_refine_dit, then export (1/scale -> us_vae_transformer -> dense volume decode -> marching cubes).
// Feeds the BANKED fp32 cond/voxel_cond/init_latents/sigmas from capture_e2e.py (the conditioner &
// voxelizer are validated separately), reproduces the fp32 oracle, and checks final_latents +
// grid_logits cosine + symmetric Chamfer vs the golden mesh, then writes the GLB.
//   ./build.sh ultrashape_e2e cuda  +  PIXAL3D_GGUF_DIR=<gguf-with-dit.gguf>  -> F16 DiT on the 3060
//   (~seconds; F16 forward == fp32). CPU fp32 (no gguf env) = the npy parity path (~10min).
// VALIDATED 2026-06-18: final_latents cos 0.9999, grid cos 0.99995, Chamfer 0.03% bbox-diag,
// mesh 9100 vs golden 9054 verts. (Two bugs fixed: voxel_cond read as i32 not f32; gallocr
// input-clobber between the cond/uncond CFG passes — re-upload all inputs before each compute.)
#include "ultrashape_dit.hpp"
#include "ultrashape_vae.hpp"
#include "ultrashape_mc.hpp"
#include "dinov2_graph.hpp"
#include "ultrashape_cond.hpp"
#include "us_image_proc.hpp"
#include "us_voxelize.hpp"
#include "torch_randn.hpp"
#include "glb_writer.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
static double now_s(){ return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static const char* GDIR = "/mnt/hdd/3d/avatar-shootout/e2e_goldens";
static const char* DIT_W = "/mnt/hdd/3d/avatar-shootout/ultrashape_goldens/weights_npy/dit";
static const char* VAE_W = "/mnt/hdd/3d/avatar-shootout/ultrashape_goldens/weights_npy/vae";
static const char* CND_W = "/mnt/hdd/3d/avatar-shootout/ultrashape_goldens/weights_npy/conditioner";
// --native inputs: the RAW matte PNG (ImageProcessorV2 runs natively inside) + the coarse mesh GLB.
static const char* RAW_IMG    = "/mnt/hdd/3d/avatar-shootout/_shootout_out/char1_matte.png";
static const char* COARSE_GLB = "/mnt/hdd/3d/avatar-shootout/_shootout_out/char1_coarse.glb";

// Full native image conditioner: RAW matte PNG -> ImageProcessorV2 (native) -> proc_image/proc_mask ->
// host preprocess + dinov2::encode + foreground-mask token gather -> cond_main [1024, Tc] (HID fastest).
// (ImageProcessorV2 validated in us_image_proc_test; conditioner in ultrashape_cond_test.)
static std::vector<float> run_native_conditioner(bool use_cuda, int& Tc_out, const char* img_path) {
    const int OUT = 1022, PATCH = 14, rows = 73, cols = 73, HID = dinov2::HID, SEQ = 1 + rows*cols;
    us_imgproc::Processed pp = us_imgproc::process(img_path, 1024, 0.15);   // raw PNG -> proc_image/mask
    const int PS = pp.size;
    std::vector<float> inputs = us_cond::preprocess(pp.image.data(), 3, PS, PS, OUT,
                                                    us_cond::IMAGENET_MEAN, us_cond::IMAGENET_STD);
    M1Harness H(CND_W, 2048, use_cuda);
    ggml_context* ctx = H.ctx;
    int64_t img_ne[4] = {(int64_t)cols*PATCH, (int64_t)rows*PATCH, 3, 1};
    ggml_tensor* img = H.input("image", 4, img_ne);
    NpyArray pe = npy_load(std::string(CND_W) + "/main_image_encoder.model.embeddings.position_embeddings.npy");
    std::vector<float> pos = dinov2::interp_pos_embed(pe.f32(), rows, cols, 0.1);
    int64_t pos_ne[4] = {HID, (int64_t)SEQ, 1, 1};
    ggml_tensor* pos_const = H.const_tensor("pos_embed_interp", 2, pos_ne, pos);
    ggml_tensor* out = dinov2::encode(ctx, H, img, pos_const, rows, cols, dinov2::HF, "main_image_encoder.model.");
    ggml_cgraph* gf = new_graph(ctx, 32768);
    ggml_build_forward_expand(gf, out);
    H.alloc_and_upload(gf);
    H.upload_input_raw(img, inputs);
    H.compute(gf);
    std::vector<float> hidden((size_t)HID*SEQ);
    ggml_backend_tensor_get(out, hidden.data(), 0, hidden.size()*sizeof(float));
    std::vector<int> valid = us_cond::mask_valid_tokens(pp.mask.data(), PS, PS, OUT, PATCH, true);
    Tc_out = (int)valid.size();
    printf("[native] conditioner: %d valid tokens\n", Tc_out);
    return us_cond::gather_tokens(hidden.data(), HID, SEQ, valid);
}

static double cosine(const float* a, const float* b, int64_t n) {
    double dot=0,na=0,nb=0;
    for (int64_t i=0;i<n;i++){ dot+=(double)a[i]*b[i]; na+=(double)a[i]*a[i]; nb+=(double)b[i]*b[i]; }
    return dot/(std::sqrt(na)*std::sqrt(nb)+1e-30);
}
static double meanabs(const float* a, const float* b, int64_t n) {
    double s=0; for (int64_t i=0;i<n;i++) s+=std::fabs((double)a[i]-b[i]); return s/n;
}

int main(int argc, char** argv) {
    bool use_cuda = false, native = false;
    // Configurable like the Python infer_dit_refine.py: octree_res, num_latents, guidance, steps, io.
    // Defaults = PRODUCTION (match the Python pipeline: guidance 7.5, octree 512). The banked-parity run
    // must pass --guidance 5.0 --octree 64 (the goldens were captured by capture_e2e.py at those values).
    float GUIDANCE   = 7.5f;       // pipelines.py default (single-CFG); capture_e2e.py used 5.0
    int   OCT        = 512;        // octree_resolution (DENSE grid here; Python uses a hierarchical octree)
    int   cli_latents = -1;        // override token_num/num_latents (default = meta[0])
    int   cli_steps   = -1;        // override steps (default = meta[2])
    const char* cli_img = nullptr; const char* cli_mesh = nullptr; const char* cli_out = nullptr;
    int64_t CHUNK = 16384;
    const float BNDS = 1.0f;       // box_v=1.0
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "cuda") use_cuda = true;
        else if (a == "--native") native = true;
        else if (a == "--guidance" && i+1<argc) GUIDANCE = (float)atof(argv[++i]);
        else if ((a == "--octree" || a == "--octree-res") && i+1<argc) OCT = atoi(argv[++i]);
        else if ((a == "--num-latents" || a == "--token-num") && i+1<argc) cli_latents = atoi(argv[++i]);
        else if (a == "--steps" && i+1<argc) cli_steps = atoi(argv[++i]);
        else if (a == "--chunk" && i+1<argc) CHUNK = atoll(argv[++i]);
        else if (a == "--image" && i+1<argc) cli_img = argv[++i];
        else if (a == "--mesh" && i+1<argc) cli_mesh = argv[++i];
        else if (a == "--out" && i+1<argc) cli_out = argv[++i];
    }
    const char* IMG_PATH  = cli_img  ? cli_img  : RAW_IMG;
    const char* MESH_PATH = cli_mesh ? cli_mesh : COARSE_GLB;

    // ---- config (meta = run config, not a parity golden) ----
    NpyArray meta = npy_load(std::string(GDIR) + "/meta.npy");   // f64 [8]
    const double* mp = meta.f64();
    int   token_num = (cli_latents > 0) ? cli_latents : (int)mp[0];
    int   steps     = (cli_steps   > 0) ? cli_steps   : (int)mp[2];
    int   seed      = (int)mp[3];
    float scale_factor = (float)mp[6];   // VAE constant
    int   vox_res   = (int)mp[7];
    printf("[e2e] mode=%s token_num=%d steps=%d guidance=%.2f scale_factor=%.8f oct=%d chunk=%lld\n",
           native?"NATIVE":"banked", token_num, steps, GUIDANCE, scale_factor, OCT, (long long)CHUNK);

    int64_t N, Tc;
    std::vector<float> voxf, cond_main, uncond_main, sigmas, x;
    if (native) {
        // ---- FULL NATIVE: image+coarse-mesh -> cond/voxel/init/sigmas, NO banked model outputs ----
        // (proc_image/proc_mask = the ImageProcessorV2 "processed matte" boundary; coarse GLB = the mesh.)
        int Tc_i = 0;
        cond_main = run_native_conditioner(use_cuda, Tc_i, IMG_PATH);
        Tc = Tc_i;
        uncond_main.assign((size_t)Tc * dinov2::HID, 0.0f);                 // unconditional = zero tokens
        us_vox::VoxelCond vc;
        if (!us_vox::voxelize_mesh(MESH_PATH, token_num, vox_res, 409600, (uint64_t)seed, vc)) {
            printf("[native] voxelize_mesh FAILED\n"); return 1; }
        N = token_num;
        voxf.resize((size_t)N*3);
        for (size_t i = 0; i < voxf.size(); i++) voxf[i] = (float)vc.coords[i];
        printf("[native] voxelizer: %d occupied -> %lld latent voxels\n", vc.n_occupied, (long long)N);
        trandn::Generator g((uint64_t)seed);                               // torch-faithful CPU randn
        x = g.randn((int64_t)N * 64);                                      // init_latents [k*64+c]
        sigmas.resize(steps + 1);                                          // linspace(0,1,steps) + append 1.0
        for (int i = 0; i < steps; i++) sigmas[i] = (float)i / (steps - 1);
        sigmas[steps] = 1.0f;
    } else {
        NpyArray voxN = npy_load(std::string(GDIR) + "/voxel_cond.npy");   // f32 [1,N,3] (capture save()s float)
        N = voxN.shape[1];
        voxf.assign(voxN.f32(), voxN.f32() + voxN.numel());
        NpyArray condN = npy_load(std::string(GDIR) + "/cond_main.npy");   // f32 [1,Tc,1024]
        Tc = condN.shape[1];
        cond_main.assign(condN.f32(), condN.f32()+condN.numel());
        NpyArray uncN = npy_load(std::string(GDIR) + "/uncond_main.npy");  // f32 [1,Tc,1024]
        uncond_main.assign(uncN.f32(), uncN.f32()+uncN.numel());
        NpyArray sigN = npy_load(std::string(GDIR) + "/sigmas_full.npy");  // f32 [steps+1]
        sigmas.assign(sigN.f32(), sigN.f32()+sigN.numel());
        NpyArray initN = npy_load(std::string(GDIR) + "/init_latents.npy");// f32 [1,N,64]
        x.assign(initN.f32(), initN.f32()+initN.numel());                  // [k*64+c]
    }
    printf("[e2e] N=%lld Tc=%lld sigmas=%zu  s[0..2]=%.4f,%.4f,%.4f  s[-2:]=%.4f,%.4f\n",
           (long long)N,(long long)Tc, sigmas.size(), sigmas[0],sigmas[1],sigmas[2],
           sigmas[sigmas.size()-2], sigmas[sigmas.size()-1]);

    // DiT scope: the F16 DiT weights (6.1GB) free at the end of this block, BEFORE the VAE harness is
    // built, so the two are never co-resident (mirrors Python --low_vram cpu_offload). Without this the
    // VAE transformer's O(N^2) self-attention over N=8192 latents OOMs on the 3060 (see PERF B3).
    double fl_cos = 1.0;
    {
    UsDitCfg dcfg;
    const int hidden = dcfg.hidden, ich = dcfg.in_channels;

    // ---- DiT graph (built once; x_lat/cond/ts as inputs, rope const) ----
    M1Harness Hd(DIT_W, 8192, use_cuda);
    ggml_context* cd = Hd.ctx;
    std::vector<float> rcos, rsin;
    us_rope_3d(voxf.data(), N, dcfg, rcos, rsin);
    int dim = dcfg.head_dim(); int64_t S = 1 + N;
    int64_t rope_ne[4] = {dim, 1, S, 1};
    ggml_tensor* rcos_t = Hd.const_tensor("rope_cos", 3, rope_ne, rcos);
    ggml_tensor* rsin_t = Hd.const_tensor("rope_sin", 3, rope_ne, rsin);
    int64_t ts_ne[4] = {hidden, 1, 1, 1};
    ggml_tensor* ts_t = Hd.input("ts_embed", 2, ts_ne);
    int64_t x_ne[4] = {ich, N, 1, 1};
    ggml_tensor* x_lat = Hd.input("x_lat", 2, x_ne);
    int64_t c_ne[4] = {dcfg.context_dim, Tc, 1, 1};
    ggml_tensor* cond = Hd.input("cond", 2, c_ne);
    ggml_tensor* dout = us_refine_dit(Hd, cd, dcfg, x_lat, ts_t, cond, rcos_t, rsin_t);  // [64,N]
    ggml_set_output(dout);
    ggml_cgraph* gd = new_graph(cd, 65536);
    ggml_build_forward_expand(gd, dout);
    Hd.alloc_and_upload(gd);

    // ---- sampling loop ----
    std::vector<float> pred_c((size_t)N*ich), pred_u((size_t)N*ich);
    const int num_train = 1000;
    double t_sample0 = now_s();
    for (int i=0;i<steps;i++) {
        float tnorm = sigmas[i];   // t/num_train == sigmas[i] (shift=1)
        std::vector<float> ts = us_timesteps_embed(tnorm, hidden);
        // cond pass — re-upload ALL inputs before each compute: gallocr may reuse an input's buffer as
        // scratch after its last graph-use, so a prior compute can clobber x_lat/ts for the next pass.
        Hd.upload_input_raw(ts_t, ts);
        Hd.upload_input_raw(x_lat, x);
        Hd.upload_input_raw(cond, cond_main);
        Hd.compute(gd);
        ggml_backend_tensor_get(dout, pred_c.data(), 0, pred_c.size()*sizeof(float));
        // uncond pass (zero cond tokens)
        Hd.upload_input_raw(ts_t, ts);
        Hd.upload_input_raw(x_lat, x);
        Hd.upload_input_raw(cond, uncond_main);
        Hd.compute(gd);
        ggml_backend_tensor_get(dout, pred_u.data(), 0, pred_u.size()*sizeof(float));
        // CFG + Euler step
        float dsig = sigmas[i+1] - sigmas[i];
        for (size_t j=0;j<x.size();j++) {
            float pred = pred_u[j] + GUIDANCE*(pred_c[j]-pred_u[j]);
            x[j] += dsig * pred;
        }
        if (i%10==0 || i==steps-1) {
            double m=0; for (float v:x) m+=std::fabs(v); m/=x.size();
            printf("  [e2e] step %2d t=%.4f dsig=%+.4f |x|=%.4f\n", i, tnorm, dsig, m);
        }
        // compare to banked trajectory checkpoints (traj_k = x AFTER step i where k=i+1)
        int k = i + 1;
        if (!native && (k==1 || k==5 || k==10 || k==25 || k==49 || k==50)) {
            char p[256]; snprintf(p, sizeof(p), "%s/traj_%02d.npy", GDIR, k);
            NpyArray tg = npy_load(p);
            double c = cosine(x.data(), tg.f32(), (int64_t)x.size());
            double ma = meanabs(x.data(), tg.f32(), (int64_t)x.size());
            double mg=0; for (int64_t j=0;j<(int64_t)x.size();j++) mg+=std::fabs(tg.f32()[j]); mg/=x.size();
            printf("    [traj k=%2d] cosine=%.6f meanabs=%.3e  |mine|=%.4f |gold|=%.4f\n",
                   k, c, ma, [&]{double s=0;for(float v:x)s+=std::fabs(v);return s/x.size();}(), mg);
        }
    }

    printf("[perf] DiT sampling: %d steps x2(CFG) over N=%lld latents = %.1fs (%.2fs/step)\n",
           steps, (long long)N, now_s()-t_sample0, (now_s()-t_sample0)/std::max(1,steps));

    // ---- validate final latents (banked mode only; native diverges by RNG inputs, judge mesh) ----
    if (!native) {
        NpyArray flN = npy_load(std::string(GDIR) + "/final_latents.npy");
        fl_cos = cosine(x.data(), flN.f32(), (int64_t)x.size());
        double fl_ma = meanabs(x.data(), flN.f32(), (int64_t)x.size());
        printf("[e2e] final_latents cosine=%.6f meanabs=%.3e\n", fl_cos, fl_ma);
    }
    }  // end DiT scope — Hd (6.1GB F16 weights) freed here before the VAE harness is built

    // ---- export: 1/scale -> vae transformer -> dense volume decode ----
    std::vector<float> lat_scaled(x.size());
    for (size_t j=0;j<x.size();j++) lat_scaled[j] = x[j] / scale_factor;

    UsVaeCfg vcfg;

    // ---- VAE transformer: computed ONCE (depends only on the latents), not per query-chunk. ----
    // The 16-layer O(N^2) self-attention over N latents used to be built into the geo_decoder chunk
    // graph and thus recomputed ~Ngrid/CHUNK times (~1k+ at octree 256) — the dense-decode bottleneck.
    // Hoisting it out is bit-exact (same fp32 weights, same latents) and removes it from the chunk loop.
    std::vector<float> tr_host((size_t)vcfg.width * N);
    {
        M1Harness Ht(VAE_W, 4096, use_cuda);
        int64_t vlat_ne[4] = {vcfg.embed_dim, N, 1, 1};
        ggml_tensor* vlat = Ht.input("vlatents", 2, vlat_ne);
        ggml_tensor* tr = us_vae_transformer(Ht, Ht.ctx, vcfg, vlat);       // [width, N]
        ggml_set_output(tr);
        ggml_cgraph* gt = new_graph(Ht.ctx, 32768);
        ggml_build_forward_expand(gt, tr);
        Ht.alloc_and_upload(gt);
        Ht.upload_input_raw(vlat, lat_scaled);
        double t_tr0 = now_s();
        Ht.compute(gt);
        ggml_backend_tensor_get(tr, tr_host.data(), 0, tr_host.size()*sizeof(float));
        printf("[perf] VAE transformer (16 layers over N=%lld, once): %.1fs\n", (long long)N, now_s()-t_tr0);
    }   // Ht (transformer weights) freed before the decoder harness is built

    // ---- geo_decoder: per query-chunk, reading the cached transformer output as a persistent const. ----
    // tr_cached lives in ctx_w (const_tensor): uploaded once, never reused by gallocr as scratch, so
    // there is no per-chunk re-upload and no input-clobber between computes.
    M1Harness Hv(VAE_W, 4096, use_cuda);
    ggml_context* cv = Hv.ctx;
    int64_t tr_ne[4] = {vcfg.width, N, 1, 1};
    ggml_tensor* tr_in = Hv.const_tensor("tr_cached", 2, tr_ne, std::move(tr_host));
    int64_t qe_ne[4] = {vcfg.fourier_out(), CHUNK, 1, 1};
    ggml_tensor* query_embed = Hv.input("query_embed", 2, qe_ne);
    ggml_tensor* occ = us_geo_decoder(Hv, cv, vcfg, query_embed, tr_in);   // [1,CHUNK]
    ggml_set_output(occ);
    ggml_cgraph* gv = new_graph(cv, 32768);
    ggml_build_forward_expand(gv, occ);
    Hv.alloc_and_upload(gv);

    int G;
    std::vector<float> grid = us_dense_grid_queries(OCT, BNDS, G);      // grid spans ±box_v (=1.0, matches capture_e2e)
    int64_t Ngrid = (int64_t)G*G*G;
    std::vector<float> logits(Ngrid, 0.0f);
    double t_dec0 = now_s();
    printf("[perf] VAE dense decode: G=%d -> %lld queries / %lld-chunk = %lld computes ...\n",
           G, (long long)Ngrid, (long long)CHUNK, (long long)((Ngrid+CHUNK-1)/CHUNK));
    for (int64_t s=0;s<Ngrid;s+=CHUNK) {
        int64_t n = std::min(CHUNK, Ngrid-s);
        std::vector<float> qchunk((size_t)CHUNK*3, 0.0f);
        std::copy(grid.begin()+s*3, grid.begin()+(s+n)*3, qchunk.begin());
        std::vector<float> qe = us_fourier_embed(qchunk.data(), CHUNK, vcfg);
        Hv.upload_input_raw(query_embed, qe);
        Hv.compute(gv);
        std::vector<float> got(CHUNK);
        ggml_backend_tensor_get(occ, got.data(), 0, CHUNK*sizeof(float));
        std::copy(got.begin(), got.begin()+n, logits.begin()+s);
    }
    printf("[perf] VAE dense decode: %.1fs\n", now_s()-t_dec0);

    // ---- marching cubes (bounds box_v=1.0) + GLB (always) ----
    us_mc::Mesh m = us_mc::marching_cubes(logits.data(), G, 0.0f);
    us_mc::scale_to_bbox(m.verts, G, BNDS);
    int V=(int)(m.verts.size()/3), F=(int)(m.faces.size()/3);
    printf("[e2e] mesh: verts=%d faces=%d  (G=%d, %lld grid queries)\n", V, F, G, (long long)Ngrid);
    const char* out = cli_out ? cli_out
                     : (native ? "/mnt/hdd/3d/avatar-shootout/_shootout_out/us_e2e_FULLNATIVE.glb"
                               : "/mnt/hdd/3d/avatar-shootout/_shootout_out/us_e2e_native.glb");
    if (glb::write_glb(out, m.verts, m.faces)) printf("[e2e] wrote %s\n", out);

    // ---- golden grid validation: ONLY when this run's grid matches the banked golden's G ----
    // (the goldens were captured at OCT=64; a production OCT (e.g. 512) has no matching golden and
    //  parity is judged externally vs the Python docker refine. Guards an OOB read on G mismatch.)
    double g_cos = 1.0, chrel = 0.0; bool golden_ok = false;
    NpyArray glN = npy_load(std::string(GDIR) + "/grid_logits.npy");    // [1,Gg,Gg,Gg]
    int Gg = (int)glN.shape[glN.shape.size()-1];
    if (Gg == G && (int64_t)glN.numel() == Ngrid) {
        golden_ok = true;
        g_cos = cosine(logits.data(), glN.f32(), Ngrid);
        double g_ma = meanabs(logits.data(), glN.f32(), Ngrid);
        printf("[e2e] grid_logits G=%d cosine=%.6f meanabs=%.3e\n", G, g_cos, g_ma);
        us_mc::Mesh mg = us_mc::marching_cubes(glN.f32(), G, 0.0f);
        us_mc::scale_to_bbox(mg.verts, G, BNDS);
        int Vg=(int)(mg.verts.size()/3);
        auto nn_mean = [](const std::vector<float>& A, const std::vector<float>& B){
            size_t na=A.size()/3, nb=B.size()/3; double s=0;
            for (size_t i=0;i<na;i++){ double best=1e30;
                for (size_t j=0;j<nb;j++){ double dx=A[i*3]-B[j*3],dy=A[i*3+1]-B[j*3+1],dz=A[i*3+2]-B[j*3+2];
                    double d=dx*dx+dy*dy+dz*dz; if(d<best)best=d; }
                s+=std::sqrt(best); }
            return na? s/na : 0.0; };
        double chamfer = 0.5*(nn_mean(m.verts, mg.verts)+nn_mean(mg.verts, m.verts));
        float mn[3]={1e30f,1e30f,1e30f}, mx[3]={-1e30f,-1e30f,-1e30f};
        for (int i=0;i<Vg;i++) for(int d=0;d<3;d++){ float v=mg.verts[i*3+d]; mn[d]=std::min(mn[d],v); mx[d]=std::max(mx[d],v); }
        double diag=std::sqrt((double)(mx[0]-mn[0])*(mx[0]-mn[0])+(mx[1]-mn[1])*(mx[1]-mn[1])+(mx[2]-mn[2])*(mx[2]-mn[2]));
        chrel = chamfer / (diag + 1e-9);
        printf("[e2e] golden mesh verts=%d; symmetric Chamfer=%.5f (%.2f%% of bbox-diag %.4f)\n",
               Vg, chamfer, 100.0*chrel, diag);
    } else {
        printf("[e2e] no golden at G=%d (golden Gg=%d) -> skip golden parity (judge vs Python docker externally)\n", G, Gg);
    }

    bool pass = golden_ok ? (V > 0 && (native ? chrel < 0.05 : (fl_cos > 0.999 && chrel < 0.01)))
                          : (V > 0);   // production OCT: just require a non-empty mesh; parity judged externally
    printf("[e2e] %s  (mode=%s octree=%d guidance=%.2f final_latents cos %.6f, grid cos %.6f, Chamfer %.3f%% diag)\n",
           pass?"PASS":"FAIL", native?"NATIVE":"banked", OCT, GUIDANCE, fl_cos, g_cos, 100.0*chrel);
    return pass ? 0 : 1;
}
