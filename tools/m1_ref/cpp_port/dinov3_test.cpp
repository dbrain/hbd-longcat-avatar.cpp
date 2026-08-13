// DINOv3 ViT-L/16 encoder (Stage-1 image conditioner backbone) in ggml.
//
// Mirrors tools/m1_ref/dinov3_proj.py (the validated fp32 numpy oracle):
//   patch Conv2d(3->1024,k16,s16) -> [cls, reg*4, patch*1024] -> 24x DINOv3 layer
//   (pre-norm, separate q/k/v/o with k_proj biasless, 2D-axial RoPE on PATCH tokens
//    only, LayerScale, erf-GELU MLP) -> plain F.layer_norm -> split global/patchmap.
//
// Validates (CPU backend, tight vs numpy fp32 refs in refs/):
//   dino_emb [1,1029,1024]      after embeddings (cls+reg+patch)
//   dino_block0 [1,1029,1024]   after layer 0
//   dino_global [1,5,1024]      final cls+reg  (== golden_stages/stage1_cond/global.npy)
//   dino_patchmap [1,32,32,1024] final patch map (pre grid_sample)
#include "m1_ggml.hpp"
#include <cmath>

static const char* WDIR = "weights_npy/dinov3";
static const char* REFS = "refs";

static const int HID = 1024, N_LAYERS = 24, N_HEADS = 16, HEAD_DIM = 64;
static const int PATCH = 16, N_REG = 4, INTER = 4096, IMG = 512;
static const int NPH = IMG / PATCH, NPW = IMG / PATCH;        // 32,32
static const int NP = NPH * NPW;                              // 1024 patch tokens
static const int SEQ = 1 + N_REG + NP;                        // 1029
static const float ROPE_THETA = 100.0f, LN_EPS = 1e-5f;
static const int N_PREFIX = 1 + N_REG;                        // 5

// ---- precompute 2D-axial RoPE cos/sin, layout [d=64, 1, token=1024] (ne0=d, ne2=token) ----
static void rope_cos_sin(std::vector<float>& cosb, std::vector<float>& sinb) {
    cosb.assign((size_t)HEAD_DIM * NP, 0.f);
    sinb.assign((size_t)HEAD_DIM * NP, 0.f);
    int nf = HEAD_DIM / 4;  // 16
    std::vector<float> inv_freq(nf);
    for (int f = 0; f < nf; f++) inv_freq[f] = 1.0f / std::pow(ROPE_THETA, (4.0f / HEAD_DIM) * f);
    for (int i = 0; i < NPH; i++)
        for (int j = 0; j < NPW; j++) {
            int p = i * NPW + j;
            float cy = ((i + 0.5f) / NPH) * 2.f - 1.f;  // y coord
            float cx = ((j + 0.5f) / NPW) * 2.f - 1.f;  // x coord
            // angles[p, axis*16+f] = 2pi*coord[axis]*inv_freq[f]; axis0=y,axis1=x; tile to 64
            for (int f = 0; f < nf; f++) {
                float ay = 2.f * (float)M_PI * cy * inv_freq[f];
                float ax = 2.f * (float)M_PI * cx * inv_freq[f];
                int d0 = f;            // y block [0..15]
                int d1 = nf + f;       // x block [16..31]
                float ca_y = std::cos(ay), sa_y = std::sin(ay);
                float ca_x = std::cos(ax), sa_x = std::sin(ax);
                size_t base = (size_t)p * HEAD_DIM;
                cosb[base + d0] = ca_y;       sinb[base + d0] = sa_y;
                cosb[base + d1] = ca_x;       sinb[base + d1] = sa_x;
                cosb[base + 32 + d0] = ca_y;  sinb[base + 32 + d0] = sa_y;  // tile
                cosb[base + 32 + d1] = ca_x;  sinb[base + 32 + d1] = sa_x;
            }
        }
}

int main(int argc, char** argv) {
    bool use_cuda = (argc > 1 && std::string(argv[1]) == "cuda");
    M1Harness H(WDIR, 256, use_cuda);
    ggml_context* ctx = H.ctx;

    // ---- inputs ----
    int64_t img_ne[4] = {IMG, IMG, 3, 1};                 // [W,H,C,N]
    ggml_tensor* img = H.input("image", 4, img_ne);
    int64_t cs_ne[4] = {HEAD_DIM, 1, NP, 1};              // cos/sin [d,1,token]
    ggml_tensor* cosT = H.input("rope_cos", 3, cs_ne);
    ggml_tensor* sinT = H.input("rope_sin", 3, cs_ne);

    // ---- weights ----
    ggml_tensor* pw = H.weight("embeddings.patch_embeddings.weight");  // [16,16,3,1024]
    ggml_tensor* pb = H.weight("embeddings.patch_embeddings.bias");    // [1024]
    ggml_tensor* cls = H.weight("embeddings.cls_token");               // [1024,1,1]
    ggml_tensor* reg = H.weight("embeddings.register_tokens");         // [1024,4,1]

    // ---- patch embed: conv2d (F32 im2col since F32 kernel) ----
    ggml_tensor* conv = ggml_conv_2d(ctx, pw, img, PATCH, PATCH, 0, 0, 1, 1); // [OW,OH,OC,N]
    // [OW=32,OH=32,OC=1024] -> tokens p=oh*32+ow with ch as feature
    ggml_tensor* emb = ggml_reshape_2d(ctx, conv, (int64_t)NPW * NPH, HID);    // [p, ch]
    emb = ggml_cont(ctx, ggml_transpose(ctx, emb));                           // [ch, p]
    emb = ggml_add(ctx, emb, pb);                                             // + bias (per ch)
    // prepend cls + reg -> [ch, 1029]
    ggml_tensor* clsr = ggml_reshape_2d(ctx, cls, HID, 1);
    ggml_tensor* regr = ggml_reshape_2d(ctx, reg, HID, N_REG);
    ggml_tensor* x = ggml_concat(ctx, ggml_concat(ctx, clsr, regr, 1), emb, 1); // [ch,1029]
    ggml_set_output(x);
    ggml_tensor* emb_tap = x;  // dino_emb

    // ---- 24 layers ----
    ggml_tensor* block0_tap = nullptr;
    for (int li = 0; li < N_LAYERS; li++) {
        std::string p = "layer." + std::to_string(li) + ".";
        // attn sub-block (pre-norm affine)
        ggml_tensor* h = layernorm(ctx, x, H.weight(p + "norm1.weight"), H.weight(p + "norm1.bias"), LN_EPS);
        ggml_tensor* q = lin(ctx, H.weight(p + "attention.q_proj.weight"), H.weight(p + "attention.q_proj.bias"), h);
        ggml_tensor* k = lin(ctx, H.weight(p + "attention.k_proj.weight"), nullptr, h);  // k: NO bias
        ggml_tensor* v = lin(ctx, H.weight(p + "attention.v_proj.weight"), H.weight(p + "attention.v_proj.bias"), h);
        q = ggml_reshape_3d(ctx, q, HEAD_DIM, N_HEADS, SEQ);  // [d,head,token]
        k = ggml_reshape_3d(ctx, k, HEAD_DIM, N_HEADS, SEQ);
        v = ggml_reshape_3d(ctx, v, HEAD_DIM, N_HEADS, SEQ);
        // RoPE on patch tokens only (tokens N_PREFIX..end)
        auto apply_rope = [&](ggml_tensor* t) {
            ggml_tensor* pre = ggml_cont(ctx, ggml_view_3d(ctx, t, HEAD_DIM, N_HEADS, N_PREFIX,
                                          t->nb[1], t->nb[2], 0));
            ggml_tensor* pat = ggml_cont(ctx, ggml_view_3d(ctx, t, HEAD_DIM, N_HEADS, NP,
                                          t->nb[1], t->nb[2], (size_t)N_PREFIX * t->nb[2]));
            ggml_tensor* rot = ggml_add(ctx, ggml_mul(ctx, pat, cosT),
                                             ggml_mul(ctx, rotate_half(ctx, pat), sinT));
            return ggml_concat(ctx, pre, rot, 2);  // [d,head,token]
        };
        q = apply_rope(q);
        k = apply_rope(k);
        // attention: permute to [d,token,head]
        ggml_tensor* qp = ggml_permute(ctx, q, 0, 2, 1, 3);  // [d,token,head]
        ggml_tensor* kp = ggml_permute(ctx, k, 0, 2, 1, 3);
        ggml_tensor* kq = ggml_mul_mat(ctx, kp, qp);          // [token_k,token_q,head]
        kq = ggml_soft_max_ext(ctx, kq, nullptr, 1.0f / std::sqrt((float)HEAD_DIM), 0.0f);
        ggml_tensor* vp = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));  // [token,d,head]
        ggml_tensor* kqv = ggml_mul_mat(ctx, vp, kq);         // [d,token_q,head]
        kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3);             // [d,head,token]
        ggml_tensor* attn = ggml_cont_2d(ctx, kqv, HID, SEQ); // [ch,token]
        attn = lin(ctx, H.weight(p + "attention.o_proj.weight"), H.weight(p + "attention.o_proj.bias"), attn);
        attn = ggml_mul(ctx, attn, H.weight(p + "layer_scale1.lambda1"));  // LayerScale
        x = ggml_add(ctx, x, attn);
        // mlp sub-block
        ggml_tensor* h2 = layernorm(ctx, x, H.weight(p + "norm2.weight"), H.weight(p + "norm2.bias"), LN_EPS);
        h2 = lin(ctx, H.weight(p + "mlp.up_proj.weight"), H.weight(p + "mlp.up_proj.bias"), h2);
        h2 = gelu_erf_(ctx, h2);
        h2 = lin(ctx, H.weight(p + "mlp.down_proj.weight"), H.weight(p + "mlp.down_proj.bias"), h2);
        h2 = ggml_mul(ctx, h2, H.weight(p + "layer_scale2.lambda1"));
        x = ggml_add(ctx, x, h2);
        if (li == 0) { x = ggml_cont(ctx, x); block0_tap = x; ggml_set_output(block0_tap); }
    }
    // final plain (unweighted) layer_norm
    x = layernorm(ctx, x, nullptr, nullptr, LN_EPS);          // [ch,1029]
    ggml_set_output(x);
    // split
    ggml_tensor* global = ggml_cont(ctx, ggml_view_2d(ctx, x, HID, N_PREFIX, x->nb[1], 0)); // [ch,5]
    ggml_set_output(global);
    ggml_tensor* patch = ggml_cont(ctx, ggml_view_2d(ctx, x, HID, NP, x->nb[1], (size_t)N_PREFIX * x->nb[1]));
    ggml_tensor* patchmap = ggml_reshape_3d(ctx, patch, HID, NPW, NPH);  // [ch,w,h]
    ggml_set_output(patchmap);

    // ---- run ----
    ggml_cgraph* gf = new_graph(ctx, 16384);
    ggml_build_forward_expand(gf, global);
    ggml_build_forward_expand(gf, patchmap);
    ggml_build_forward_expand(gf, emb_tap);
    ggml_build_forward_expand(gf, block0_tap);
    H.alloc_and_upload(gf);
    // upload inputs
    H.upload_input_npy(img, std::string(REFS) + "/image_chw.npy");
    std::vector<float> cosb, sinb; rope_cos_sin(cosb, sinb);
    H.upload_input_raw(cosT, cosb);
    H.upload_input_raw(sinT, sinb);

    H.compute(gf);

    printf("[dinov3] backend=%s\n", use_cuda ? "cuda" : "cpu");
    CmpStats e = compare_to_npy(H, emb_tap, std::string(REFS) + "/dino_emb.npy", true, "emb");
    CmpStats b0 = compare_to_npy(H, block0_tap, std::string(REFS) + "/dino_block0.npy", true, "block0");
    CmpStats g = compare_to_npy(H, global, std::string(REFS) + "/dino_global.npy", true, "global");
    CmpStats pm = compare_to_npy(H, patchmap, std::string(REFS) + "/dino_patchmap.npy", true, "patchmap");
    // emb/block0 carry DINOv3 massive-activation outlier channels (|x|~1e5) where fp32 ulp
    // dominates maxabs -> judge them by meanabs. global/patchmap are the real O(1-10) outputs.
    bool ok = g.maxabs < 1e-4 && pm.maxabs < 1e-4 && e.meanabs < 1e-5 && b0.meanabs < 1e-5;
    printf("[dinov3] %s (global/patchmap<1e-4, emb/block0 meanabs<1e-5)\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
