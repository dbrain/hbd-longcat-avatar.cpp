// Shared ggml-graph harness for the Pixal3D Stage-1 C++ port validation.
//
// Correctness-first: build the real ggml ops, run on the CPU backend, compare to the
// fp32 numpy reference outputs (tools/m1_ref/cpp_port/refs/) and the captured goldens.
// CUDA backend reuses the SAME graph builders (select at runtime). No framework deps —
// only ggml + the spike's npy.hpp. The numpy refs (dinov3_proj.py / ss_dit.py /
// ss_vae_decode.py / proj_cond_ref.py) are the exact spec each helper mirrors.
//
// ggml layout rule: a torch/numpy tensor with C-order shape [d0,d1,...,dk] maps to a
// ggml tensor with ne = {dk,...,d1,d0} (reverse) and IDENTICAL raw bytes. So a torch
// nn.Linear weight [out,in] loads into ggml ne={in,out}; ggml_mul_mat(W,x) with x
// ne0=in then yields the Linear (y[out]=sum_in W[out,in]x[in]).
#pragma once
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#ifdef M1_USE_CUDA
#include "ggml-cuda.h"
#endif
#include "../../sparse_spike/npy.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <stdexcept>

// ----------------------------------------------------------------------------
// reversed-shape helper
// ----------------------------------------------------------------------------
static inline void rev_ne(const std::vector<int64_t>& shape, int64_t ne[4], int& nd) {
    nd = (int)shape.size();
    if (nd > 4) throw std::runtime_error("rev_ne: >4 dims");
    for (int i = 0; i < 4; i++) ne[i] = 1;
    for (int i = 0; i < nd; i++) ne[i] = shape[nd - 1 - i];  // reverse
}

// ----------------------------------------------------------------------------
// Harness: holds a metadata ctx (no_alloc), a backend, queues weight/input uploads.
// ----------------------------------------------------------------------------
struct M1Harness {
    ggml_context* ctx = nullptr;    // compute graph (inputs + ops) — gallocr-managed
    ggml_context* ctx_w = nullptr;  // weights — persistent buffer (NOT gallocr-managed)
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;  // weights buffer
    ggml_gallocr_t galloc = nullptr;
    std::string wdir;  // weight .npy dir
    // GGUF mode (A2): if PIXAL3D_GGUF_DIR is set and <dir>/<model>.gguf exists, weights load
    // from one GGUF instead of per-tensor .npy. weight()/weight_conv3d() fetch the stored tensor
    // (already 4D-collapsed for conv3d at pack time) -> bit-identical to the .npy path.
    gguf_context* gctx = nullptr;
    ggml_context* gctx_data = nullptr;
    bool use_gguf = false;
    std::vector<std::pair<ggml_tensor*, const void*>> gguf_uploads;  // (ctx_w dst, gguf host src)
    // tensors that need host data uploaded after alloc: (tensor, npy path)
    std::vector<std::pair<ggml_tensor*, std::string>> uploads;
    // input tensors filled from an in-memory float buffer (gallocr-managed)
    std::vector<std::pair<ggml_tensor*, std::vector<float>>> raw_uploads;
    // PERSISTENT constants (ctx_w, host data) — survive across graph_compute calls,
    // unlike gallocr-managed inputs whose buffers get reused for intermediates.
    std::vector<std::pair<ggml_tensor*, std::vector<float>>> const_uploads;
    // PERSISTENT I32 constants (e.g. get_rows gather indices for na2d / mesh).
    std::vector<std::pair<ggml_tensor*, std::vector<int32_t>>> const_uploads_i32;

    M1Harness(const std::string& weight_dir, size_t mem_mb = 512, bool use_cuda = false) {
        wdir = weight_dir;
        // GGUF override: PIXAL3D_GGUF_DIR=<dir> + <dir>/<model>.gguf present -> use it.
        if (const char* gdir = std::getenv("PIXAL3D_GGUF_DIR")) {
            std::string base = weight_dir.substr(weight_dir.find_last_of('/') + 1);
            std::string gpath = std::string(gdir) + "/" + base + ".gguf";
            gguf_init_params gp{ /*no_alloc=*/false, /*ctx=*/&gctx_data };
            gctx = gguf_init_from_file(gpath.c_str(), gp);
            if (gctx) { use_gguf = true; }
            else { gctx_data = nullptr; }  // not packed yet -> fall back to .npy
        }
#ifdef M1_USE_CUDA
        if (use_cuda) backend = ggml_backend_cuda_init(0);
#endif
        if (!backend) backend = ggml_backend_cpu_init();
        ggml_init_params ip{ mem_mb * 1024 * 1024, nullptr, true /*no_alloc*/ };
        ctx = ggml_init(ip);
        // weights metadata ctx: ~1100 tensors max -> generous overhead
        ggml_init_params wp{ (size_t)4 * 1024 * 1024, nullptr, true };
        ctx_w = ggml_init(wp);
    }
    ~M1Harness() {
        if (galloc) ggml_gallocr_free(galloc);
        if (wbuf) ggml_backend_buffer_free(wbuf);
        if (ctx) ggml_free(ctx);
        if (ctx_w) ggml_free(ctx_w);
        if (gctx) gguf_free(gctx);
        if (gctx_data) ggml_free(gctx_data);
        if (backend) ggml_backend_free(backend);
    }

    // GGUF-backed fetch: create a ctx_w tensor mirroring the stored tensor's ne, queue an
    // upload from the gguf host data. conv3d weights were 4D-collapsed at pack time, so this
    // serves both weight() and weight_conv3d() identically.
    ggml_tensor* gguf_fetch(const std::string& key) {
        ggml_tensor* src = ggml_get_tensor(gctx_data, key.c_str());
        if (!src) throw std::runtime_error("gguf tensor not found: " + key);
        int nd = ggml_n_dims(src); if (nd < 1) nd = 1;
        ggml_tensor* t = ggml_new_tensor(ctx_w, src->type, nd, src->ne);
        ggml_set_name(t, key.c_str());
        gguf_uploads.push_back({t, src->data});
        wcache[key] = t;
        return t;
    }
    bool is_cuda() const {
#ifdef M1_USE_CUDA
        return ggml_backend_is_cuda(backend);
#else
        return false;
#endif
    }

    // weight cache: a graph may reference the same weight twice (e.g. rope tables) —
    // return the existing tensor instead of creating a duplicate.
    std::map<std::string, ggml_tensor*> wcache;

    // Create a weight tensor from a .npy (reversed shape) in ctx_w, queue its upload.
    ggml_tensor* weight(const std::string& key) {
        auto it = wcache.find(key);
        if (it != wcache.end()) return it->second;
        if (use_gguf) return gguf_fetch(key);
        std::string path = wdir + "/" + key + ".npy";
        NpyArray a = npy_load(path);
        int64_t ne[4]; int nd;
        rev_ne(a.shape, ne, nd);
        ggml_tensor* t = ggml_new_tensor(ctx_w, GGML_TYPE_F32, nd, ne);
        ggml_set_name(t, key.c_str());
        uploads.push_back({t, path});
        wcache[key] = t;
        return t;
    }

    // Create a weight tensor with an explicit ggml ne (caller-supplied) in ctx_w.
    // Used for conv3d kernels: npy [OC,IC,KD,KH,KW] -> ggml ne=[KW,KH,KD,IC*OC].
    ggml_tensor* weight_shaped(const std::string& key, int nd, const int64_t ne[4]) {
        auto it = wcache.find(key);
        if (it != wcache.end()) return it->second;
        std::string path = wdir + "/" + key + ".npy";
        ggml_tensor* t = ggml_new_tensor(ctx_w, GGML_TYPE_F32, nd, ne);
        ggml_set_name(t, key.c_str());
        uploads.push_back({t, path});
        wcache[key] = t;
        return t;
    }
    // conv3d kernel loader: npy [OC,IC,KD,KH,KW] -> ggml [KW,KH,KD,IC*OC].
    ggml_tensor* weight_conv3d(const std::string& key) {
        if (use_gguf) return weight(key);  // stored 4D-collapsed at pack time == this ne
        NpyArray a = npy_load(wdir + "/" + key + ".npy");
        if (a.shape.size() != 5) throw std::runtime_error("conv3d weight not 5D: " + key);
        int64_t OC = a.shape[0], IC = a.shape[1], KD = a.shape[2], KH = a.shape[3], KW = a.shape[4];
        int64_t ne[4] = {KW, KH, KD, IC * OC};
        return weight_shaped(key, 4, ne);
    }

    // Persistent constant tensor (in ctx_w, host data) — for values that DON'T change
    // across recomputes (e.g. rope cos/sin/freqs). Uploaded once after wbuf alloc.
    ggml_tensor* const_tensor(const char* name, int nd, const int64_t ne[4], std::vector<float> data) {
        ggml_tensor* t = ggml_new_tensor(ctx_w, GGML_TYPE_F32, nd, ne);
        ggml_set_name(t, name);
        const_uploads.push_back({t, std::move(data)});
        return t;
    }

    // Persistent I32 constant (ctx_w) — for get_rows gather indices (na2d, mesh).
    ggml_tensor* const_tensor_i32(const char* name, int nd, const int64_t ne[4], std::vector<int32_t> data) {
        ggml_tensor* t = ggml_new_tensor(ctx_w, GGML_TYPE_I32, nd, ne);
        ggml_set_name(t, name);
        const_uploads_i32.push_back({t, std::move(data)});
        return t;
    }

    // Create an input tensor with an explicit ggml ne (already reversed by caller).
    ggml_tensor* input(const char* name, int nd, const int64_t ne[4]) {
        ggml_tensor* t = ggml_new_tensor(ctx, GGML_TYPE_F32, nd, ne);
        ggml_set_name(t, name);
        ggml_set_input(t);
        return t;
    }

    // Queue a raw float upload into a tensor (e.g. precomputed cos/sin/indices).
    void set_raw(ggml_tensor* t, std::vector<float> data) {
        raw_uploads.push_back({t, std::move(data)});
    }

    // Allocate weights in their OWN persistent buffer (so gallocr never reuses their
    // memory for intermediates — critical: otherwise the 2nd graph_compute reads
    // clobbered weights). Then gallocr-allocate the compute graph (intermediates reused;
    // inputs persist). Weights pre-allocated => gallocr skips them.
    void alloc_and_upload(ggml_cgraph* gf) {
        wbuf = ggml_backend_alloc_ctx_tensors(ctx_w, backend);  // weights + consts persistent
        for (auto& gu : gguf_uploads) {  // GGUF-backed weights: upload from the gguf host data
            ggml_backend_tensor_set(gu.first, gu.second, 0, ggml_nbytes(gu.first));
        }
        for (auto& u : uploads) {
            NpyArray a = npy_load(u.second);
            if (a.descr != "<f4")
                throw std::runtime_error("weight not f32: " + u.second + " (" + a.descr + ")");
            size_t bytes = (size_t)a.numel() * 4;
            if (bytes != ggml_nbytes(u.first))
                throw std::runtime_error("size mismatch " + u.second);
            ggml_backend_tensor_set(u.first, a.raw.data(), 0, bytes);
        }
        for (auto& c : const_uploads) {
            if (c.second.size() * 4 != ggml_nbytes(c.first))
                throw std::runtime_error(std::string("const size mismatch for ") + ggml_get_name(c.first));
            ggml_backend_tensor_set(c.first, c.second.data(), 0, c.second.size() * 4);
        }
        for (auto& c : const_uploads_i32) {
            if (c.second.size() * 4 != ggml_nbytes(c.first))
                throw std::runtime_error(std::string("const i32 size mismatch for ") + ggml_get_name(c.first));
            ggml_backend_tensor_set(c.first, c.second.data(), 0, c.second.size() * 4);
        }
        galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(galloc, gf))
            throw std::runtime_error("gallocr_alloc_graph failed");
        for (auto& r : raw_uploads) {
            if (r.second.size() * 4 != ggml_nbytes(r.first))
                throw std::runtime_error(std::string("raw size mismatch for ") + ggml_get_name(r.first));
            ggml_backend_tensor_set(r.first, r.second.data(), 0, r.second.size() * 4);
        }
    }

    void upload_input_npy(ggml_tensor* t, const std::string& npy_path) {
        NpyArray a = npy_load(npy_path);
        size_t bytes = (size_t)a.numel() * 4;
        if (bytes != ggml_nbytes(t)) throw std::runtime_error("input size mismatch " + npy_path);
        ggml_backend_tensor_set(t, a.raw.data(), 0, bytes);
    }
    void upload_input_raw(ggml_tensor* t, const std::vector<float>& data) {
        ggml_backend_tensor_set(t, data.data(), 0, data.size() * 4);
    }

    void compute(ggml_cgraph* gf) {
        ggml_backend_graph_compute(backend, gf);
    }
};

// ----------------------------------------------------------------------------
// graph-builder helpers (mirror the numpy refs)
// ----------------------------------------------------------------------------

// Linear y = x @ W^T + b.  W ggml ne={in,out}, x ne0=in.  b ggml ne={out} or null.
// Force fp32 accumulation (correctness-first): CUDA defaults to tf32 for fp32 matmul,
// which adds ~1e-2 noise that flips a few occupancy-threshold voxels. The perf phase
// can re-enable tf32.
static inline ggml_tensor* lin(ggml_context* ctx, ggml_tensor* W, ggml_tensor* b, ggml_tensor* x) {
    ggml_tensor* y = ggml_mul_mat(ctx, W, x);  // [out, ...]
    ggml_mul_mat_set_prec(y, GGML_PREC_F32);
    if (b) y = ggml_add(ctx, y, b);            // b broadcasts over ne0
    return y;
}

// LayerNorm over ne0 (last torch dim). weight/bias ggml ne={C} or null for plain.
static inline ggml_tensor* layernorm(ggml_context* ctx, ggml_tensor* x,
                                     ggml_tensor* w, ggml_tensor* b, float eps) {
    x = ggml_norm(ctx, x, eps);  // normalizes over ne0, mean/var population
    if (w) x = ggml_mul(ctx, x, w);
    if (b) x = ggml_add(ctx, x, b);
    return x;
}

static inline ggml_tensor* silu_(ggml_context* ctx, ggml_tensor* x) { return ggml_silu(ctx, x); }

// erf GELU (exact, == torch nn.GELU()).  ggml_gelu_erf if present, else gelu.
static inline ggml_tensor* gelu_erf_(ggml_context* ctx, ggml_tensor* x) {
    return ggml_gelu_erf(ctx, x);
}
// tanh-approx GELU (== nn.GELU(approximate="tanh")), built in fp32. ggml_gelu uses an
// F16 lookup table on CPU (~1e-3 error) — too lossy for correctness-first validation.
// 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))) = 0.5*(x + x*tanh(inner)).
static inline ggml_tensor* gelu_tanh_(ggml_context* ctx, ggml_tensor* x) {
    const float c = 0.7978845608028654f;  // sqrt(2/pi)
    ggml_tensor* x3 = ggml_mul(ctx, ggml_mul(ctx, x, x), x);
    ggml_tensor* inner = ggml_scale(ctx, ggml_add(ctx, x, ggml_scale(ctx, x3, 0.044715f)), c);
    ggml_tensor* t = ggml_tanh(ctx, inner);
    return ggml_scale(ctx, ggml_add(ctx, x, ggml_mul(ctx, x, t)), 0.5f);
}

// Scaled-dot-product attention. q,k,v: [d_head, n_head, n_token]. Returns [C=d*head, tq].
static inline ggml_tensor* attention(ggml_context* ctx, ggml_tensor* q, ggml_tensor* k,
                                     ggml_tensor* v, float scale) {
    int64_t d = q->ne[0], nh = q->ne[1], tq = q->ne[2];
    ggml_tensor* qp = ggml_permute(ctx, q, 0, 2, 1, 3);  // [d, tq, head]
    ggml_tensor* kp = ggml_permute(ctx, k, 0, 2, 1, 3);  // [d, tk, head]
    ggml_tensor* kq = ggml_mul_mat(ctx, kp, qp);         // [tk, tq, head]
    ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
    kq = ggml_soft_max_ext(ctx, kq, nullptr, scale, 0.0f);
    ggml_tensor* vp = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));  // [tk, d, head]
    ggml_tensor* kqv = ggml_mul_mat(ctx, vp, kq);        // [d, tq, head]
    ggml_mul_mat_set_prec(kqv, GGML_PREC_F32);
    kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3);            // [d, head, tq]
    return ggml_cont_2d(ctx, kqv, d * nh, tq);           // [C, tq]
}

// MultiHeadRMSNorm: F.normalize(x,dim=-1)*gamma*sqrt(D) == rms_norm(x)*gamma.
// x [d, head, token]; gamma [d, head].
static inline ggml_tensor* mh_rms_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* gamma) {
    x = ggml_rms_norm(ctx, x, 1e-12f);
    return ggml_mul(ctx, x, gamma);
}

// adaLN modulate: h*(1+scale)+shift = h*scale + h + shift.  scale/shift [C] over [C,token].
static inline ggml_tensor* modulate(ggml_context* ctx, ggml_tensor* h, ggml_tensor* scale, ggml_tensor* shift) {
    return ggml_add(ctx, ggml_add(ctx, ggml_mul(ctx, h, scale), h), shift);
}

// torch-style permute (ported from base ggml_ext_torch_permute).
static inline ggml_tensor* torch_permute(ggml_context* ctx, ggml_tensor* x, int a0, int a1, int a2, int a3) {
    int ta[4] = {a0, a1, a2, a3}, ga[4] = {0};
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            if (ta[j] == i) { ga[i] = j; break; }
    return ggml_permute(ctx, x, ga[0], ga[1], ga[2], ga[3]);
}
static inline ggml_tensor* ext_cont(ggml_context* ctx, ggml_tensor* x) {
    return ggml_is_contiguous(x) ? x : ggml_cont(ctx, x);
}

// pixel_shuffle_3d / depth_to_space (ported verbatim from base ltx_vae.hpp).
// x ggml [W,H,T,C8] (C8 = c*p1*p2*p3, p1=T-bit p2=H-bit p3=W-bit) -> [W*fs, H*fs, T*ft, c].
// Channel ordering (c p1 p2 p3) == numpy pixel_shuffle_3d c*8+4a+2e+f (a=T,e=H,f=W).
static inline ggml_tensor* depth_to_space_3d(ggml_context* ctx, ggml_tensor* x, int64_t c,
                                             int ft, int fs, bool drop_first = false) {
    const int64_t T = x->ne[2], H = x->ne[1], W = x->ne[0];
    x = ext_cont(ctx, torch_permute(ctx, x, 0, 1, 3, 2));
    x = ggml_reshape_4d(ctx, x, W, H, fs, (int64_t)fs * ft * c * T);
    x = ext_cont(ctx, torch_permute(ctx, x, 2, 0, 1, 3));
    x = ggml_reshape_4d(ctx, x, (int64_t)fs * W, H, fs, (int64_t)ft * c * T);
    x = ext_cont(ctx, torch_permute(ctx, x, 0, 2, 1, 3));
    x = ggml_reshape_4d(ctx, x, (int64_t)fs * W * fs * H, ft, c, T);
    x = ext_cont(ctx, torch_permute(ctx, x, 0, 1, 3, 2));
    x = ggml_reshape_4d(ctx, x, (int64_t)fs * W, (int64_t)fs * H, (int64_t)ft * T, c);
    if (drop_first && ft > 1) x = ggml_cont(ctx, ggml_view_4d(ctx, x, x->ne[0], x->ne[1], x->ne[2] - 1, x->ne[3],
                                                              x->nb[1], x->nb[2], x->nb[3], x->nb[2]));
    return x;
}

// rotate_half over ne0: concat(-x[half:], x[:half]).
static inline ggml_tensor* rotate_half(ggml_context* ctx, ggml_tensor* x) {
    int64_t d = x->ne[0]; int64_t h = d / 2;
    ggml_tensor* x1 = ggml_view_4d(ctx, x, h, x->ne[1], x->ne[2], x->ne[3],
                                   x->nb[1], x->nb[2], x->nb[3], 0);
    ggml_tensor* x2 = ggml_view_4d(ctx, x, h, x->ne[1], x->ne[2], x->ne[3],
                                   x->nb[1], x->nb[2], x->nb[3], h * x->nb[0]);
    x2 = ggml_neg(ctx, ggml_cont(ctx, x2));
    return ggml_concat(ctx, x2, ggml_cont(ctx, x1), 0);
}

// ----------------------------------------------------------------------------
// compare util
// ----------------------------------------------------------------------------
struct CmpStats { double maxabs, meanabs, maxrel; int64_t n; };

static inline CmpStats compare_to_npy(M1Harness& H, ggml_tensor* t, const std::string& ref_npy,
                                      bool print = true, const char* tag = "") {
    NpyArray ref = npy_load(ref_npy);
    int64_t n = ref.numel();
    if (n != ggml_nelements(t)) {
        printf("  [%s] NELEM MISMATCH got=%lld ref=%lld\n", tag, (long long)ggml_nelements(t), (long long)n);
        return {1e30, 1e30, 1e30, n};
    }
    std::vector<float> got(n);
    ggml_backend_tensor_get(t, got.data(), 0, n * sizeof(float));
    bool ref_f64 = (ref.descr == "<f8");
    const float* r = ref_f64 ? nullptr : ref.f32();
    const double* rd = ref_f64 ? ref.f64() : nullptr;
    auto rv = [&](int64_t i) { return ref_f64 ? rd[i] : (double)r[i]; };
    double maxabs = 0, sum = 0, maxrel = 0;
    int64_t worst = -1;
    for (int64_t i = 0; i < n; i++) {
        double d = std::fabs((double)got[i] - rv(i));
        if (d > maxabs) { maxabs = d; worst = i; }
        sum += d;
        double rel = d / (std::fabs(rv(i)) + 1e-6);
        maxrel = std::max(maxrel, rel);
    }
    CmpStats s{maxabs, sum / n, maxrel, n};
    if (print) {
        printf("  [%s] n=%lld maxabs=%.3e meanabs=%.3e maxrel=%.3e  worst@%lld got=%.6f ref=%.6f\n",
               tag, (long long)n, s.maxabs, s.meanabs, s.maxrel, (long long)worst,
               worst >= 0 ? got[worst] : 0.f, worst >= 0 ? rv(worst) : 0.0);
    }
    return s;
}

// Build a graph, alloc, upload, compute, return. Caller sets inputs via H before/after.
static inline ggml_cgraph* new_graph(ggml_context* ctx, int size = 8192) {
    return ggml_new_graph_custom(ctx, size, false);
}
