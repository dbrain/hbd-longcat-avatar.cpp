// Validate the FULL native UltraShape image conditioner: proc_image[-1,1] + proc_mask -> cond_main.
// (1) host preprocess (resize+ImageNet norm) vs banked `inputs`; (2) dinov2::encode vs last_hidden_full;
// (3) host mask token-gather vs valid_bool + cond_main. Proves image->cond_main with NO banked goldens
// in the chain (we feed only proc_image/proc_mask, the ImageProcessorV2 output = the make_matte boundary).
//   ./build.sh ultrashape_cond_test [cuda]    (USE_BANKED_INPUTS=1 feeds the `inputs` golden to isolate the ViT)
#include "dinov2_graph.hpp"
#include "ultrashape_cond.hpp"
#include "us_image_proc.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cmath>

static const char* GDIR = "/mnt/hdd/3d/avatar-shootout/cond_full_goldens";
static const char* WDIR = "/mnt/hdd/3d/avatar-shootout/ultrashape_goldens/weights_npy/conditioner";
static const char* PREFIX = "main_image_encoder.model.";

static void cmp(const char* tag, const float* a, const float* b, int64_t n) {
    double dot=0,na=0,nb=0,maxa=0,sum=0;
    for (int64_t i=0;i<n;i++){ double d=std::fabs((double)a[i]-b[i]); maxa=std::max(maxa,d); sum+=d;
        dot+=(double)a[i]*b[i]; na+=(double)a[i]*a[i]; nb+=(double)b[i]*b[i]; }
    printf("  [%s] n=%lld cosine=%.6f maxabs=%.3e meanabs=%.3e\n", tag,(long long)n,
           dot/(std::sqrt(na)*std::sqrt(nb)+1e-30), maxa, sum/n);
}

int main(int argc, char** argv) {
    bool use_cuda = (argc > 1 && std::string(argv[1]) == "cuda");
    bool use_banked = std::getenv("USE_BANKED_INPUTS") != nullptr;
    // RAW=<matte.png> runs the native ImageProcessorV2 front-end (raw PNG -> proc_image/mask) instead of
    // loading the proc_image/proc_mask goldens -> measures the FULL Stage-1 contract: raw PNG -> cond_main.
    const char* raw = std::getenv("RAW");
    const int OUT = 1022, PATCH = 14, rows = 73, cols = 73;
    const int HID = dinov2::HID;

    // ---- (1) host preprocess: proc_image -> inputs ----
    std::vector<float> proc_image, proc_mask; int H, W;
    if (raw) {
        us_imgproc::Processed pp = us_imgproc::process(raw, 1024, 0.15);
        H = W = pp.size; proc_image = pp.image; proc_mask = pp.mask;
        printf("[cond] RAW front-end: %s -> proc_image %dx%d\n", raw, H, W);
    } else {
        NpyArray pim = npy_load(std::string(GDIR) + "/proc_image.npy");   // [1,3,1024,1024]
        H = (int)pim.shape[2]; W = (int)pim.shape[3];
        proc_image.assign(pim.f32(), pim.f32() + pim.numel());
        NpyArray pmk = npy_load(std::string(GDIR) + "/proc_mask.npy");    // [1,1,1024,1024]
        proc_mask.assign(pmk.f32(), pmk.f32() + pmk.numel());
    }
    std::vector<float> my_inputs = us_cond::preprocess(proc_image.data(), 3, H, W, OUT,
                                                       us_cond::IMAGENET_MEAN, us_cond::IMAGENET_STD);
    NpyArray inpG = npy_load(std::string(GDIR) + "/inputs.npy");      // [1,3,1022,1022]
    cmp("preprocess vs inputs", my_inputs.data(), inpG.f32(), inpG.numel());

    // ---- (2) DINOv2 ViT ----
    M1Harness Hh(WDIR, 2048, use_cuda);
    ggml_context* ctx = Hh.ctx;
    int64_t img_ne[4] = {(int64_t)cols*PATCH, (int64_t)rows*PATCH, 3, 1};
    ggml_tensor* img = Hh.input("image", 4, img_ne);
    NpyArray pe = npy_load(std::string(WDIR) + "/" + PREFIX + "embeddings.position_embeddings.npy");
    std::vector<float> pos = dinov2::interp_pos_embed(pe.f32(), rows, cols, 0.1);
    int64_t pos_ne[4] = {HID, (int64_t)(1+rows*cols), 1, 1};
    ggml_tensor* pos_const = Hh.const_tensor("pos_embed_interp", 2, pos_ne, pos);
    ggml_tensor* out = dinov2::encode(ctx, Hh, img, pos_const, rows, cols, dinov2::HF, PREFIX);
    ggml_cgraph* gf = new_graph(ctx, 32768);
    ggml_build_forward_expand(gf, out);
    Hh.alloc_and_upload(gf);
    if (use_banked) Hh.upload_input_npy(img, std::string(GDIR) + "/inputs.npy");
    else            Hh.upload_input_raw(img, my_inputs);
    Hh.compute(gf);
    int SEQ = 1 + rows*cols;
    printf("[cond] backend=%s inputs=%s SEQ=%d\n", use_cuda?"cuda":"cpu", use_banked?"banked":"native", SEQ);
    std::vector<float> hidden((size_t)HID*SEQ);
    ggml_backend_tensor_get(out, hidden.data(), 0, hidden.size()*sizeof(float));
    NpyArray lhG = npy_load(std::string(GDIR) + "/last_hidden_full.npy");  // [1,5330,1024]
    cmp("last_hidden vs golden", hidden.data(), lhG.f32(), lhG.numel());

    // ---- (3) mask token gather -> cond_main ----
    std::vector<int> valid = us_cond::mask_valid_tokens(proc_mask.data(), H, W, OUT, PATCH, true);
    NpyArray vbG = npy_load(std::string(GDIR) + "/valid_bool.npy");   // [1,5330]
    int gold_valid = 0; for (int64_t i=0;i<vbG.numel();i++) if (vbG.f32()[i] > 0) gold_valid++;
    // exact index-set check
    bool idx_ok = ((int)valid.size() == gold_valid);
    if (idx_ok) for (int t : valid) if (vbG.f32()[t] <= 0) { idx_ok = false; break; }
    printf("[cond] valid tokens: mine=%zu golden=%d  index-set %s\n",
           valid.size(), gold_valid, idx_ok?"EXACT":"MISMATCH");

    std::vector<float> cond_main = us_cond::gather_tokens(hidden.data(), HID, SEQ, valid);
    NpyArray cmG = npy_load(std::string(GDIR) + "/cond_main.npy");    // [1,3970,1024]
    if (cond_main.size() == (size_t)cmG.numel())
        cmp("cond_main vs golden", cond_main.data(), cmG.f32(), cmG.numel());
    else printf("  [cond_main] SIZE MISMATCH mine=%zu golden=%lld\n", cond_main.size(), (long long)cmG.numel());

    // PASS: cosine on cond_main >= 0.999 and exact token selection.
    double dot=0,na=0,nb=0; int64_t n=cmG.numel();
    if (cond_main.size()==(size_t)n) for (int64_t i=0;i<n;i++){ dot+=(double)cond_main[i]*cmG.f32()[i];
        na+=(double)cond_main[i]*cond_main[i]; nb+=(double)cmG.f32()[i]*cmG.f32()[i]; }
    double cos = dot/(std::sqrt(na)*std::sqrt(nb)+1e-30);
    bool pass = idx_ok && cos > 0.999;
    printf("[cond] %s (cond_main cosine=%.6f)\n", pass?"PASS":"FAIL", cos);
    return pass ? 0 : 1;
}
