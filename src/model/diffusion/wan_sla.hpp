#pragma once

// ============================================================================
// WAN SLA — lightx2v "SLA" (dynamic block-sparse attention) on our ggml runtime.
// ============================================================================
//
// This is the host side of the Wan2.2 self-attention sparse-attention port. It
// REUSES the existing block-sparse K-tile skip that already lives in the MMA
// flash kernel (ggml/src/ggml-cuda/fattn-mma-f16.cuh) and is fed by a tiny
// per-(Q-tile, K-tile) bitmap. What this file adds is the *content-based*
// selector that lightx2v calls "SLA":
//
//   1. mean-pool Q into 64-token blocks, K (smooth-k: k - mean_token(k)) into
//      64-token blocks  (sla_util.py get_block_map / mean_pool)
//   2. coarse score = pooled_q · pooled_kᵀ   [n_qblk × n_kblk]   (per head)
//   3. keep the top (1 - sparsity) K-blocks per Q-block  (default 0.8 ⇒ top 20%)
//   4. pack the kept blocks into the kernel's bitmap and run only those tiles.
//
// STAGE 0 (this file, runnable): the pooled Q/K are exported from ONE selector
// block per step (the graph writes them into the persistent pooled_q/pooled_k
// tensors). After each step the host reads them back, builds ONE bitmap (heads
// collapsed by summing per-head scores, then a single top-k), and feeds it to
// ALL blocks of the NEXT step (one-step-stale, shared across blocks + heads).
// That is the cheapest *content-derived* probe to answer the make-or-break
// question — does our un-finetuned Wan tolerate 80% block-sparsity? — before
// investing in the per-head GPU selector.
//
// STAGE 1: make the selector a GPU kernel,
// per head and per block, recomputed in-graph each step → a [n_heads, n_qtiles,
// n_kwords] bitmap, kernel indexed with a head dim. This file's CPU selector is
// the reference the GPU kernel must match.
//
// Default OFF (WAN_SLA unset) ⇒ none of this runs, byte-identical to prod.
// ----------------------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#ifdef SD_USE_CUDA
#include "ggml-cuda.h"  // ggml_cuda_set_longcat_fa_bsa_bitmap / _mask_free
#endif

namespace sd {

// Block size — must match the FA template (ncols1 = nbatch_fa = 64) the skip is
// compiled for AND lightx2v's bq = bkv = 64.
static constexpr int WAN_SLA_BLK = 64;

struct WanSlaConfig {
    bool  enabled        = false;
    float sparsity       = 0.8f;  // fraction of K-blocks DROPPED (keep top 1-sparsity)
    int   selector_block = 0;     // which DiT block exports pooled Q/K
    int   warmup_steps   = 1;     // (stale mode only) first N steps run dense+capture; sparse after
    // Stage 1 levers (the ghosting fix). Default OFF = the Stage-0 stale/shared path.
    bool  current_step   = false; // WAN_SLA_CURRENT_STEP: build the bitmap from THIS step's
                                  // pooled Q/K (2-pass: pass-1 dense harvests block-`sel`'s
                                  // Q/K, pass-2 runs sparse). Kills the one-step-stale echo.
    bool  per_head       = false; // WAN_SLA_PERHEAD: a separate bitmap per attention head.

    static WanSlaConfig from_env() {
        WanSlaConfig c;
        if (const char* s = std::getenv("WAN_SLA")) {
            c.enabled = (s[0] == '1');
        }
        if (const char* s = std::getenv("WAN_SLA_SPARSITY"); s && s[0]) {
            c.sparsity = std::strtof(s, nullptr);
            if (!(c.sparsity > 0.0f && c.sparsity < 1.0f)) c.sparsity = 0.8f;
        }
        if (const char* s = std::getenv("WAN_SLA_SELECTOR_BLOCK"); s && s[0]) {
            c.selector_block = std::atoi(s);
        }
        if (const char* s = std::getenv("WAN_SLA_WARMUP"); s && s[0]) {
            c.warmup_steps = std::atoi(s);
        }
        if (const char* s = std::getenv("WAN_SLA_CURRENT_STEP")) {
            c.current_step = (s[0] == '1');
        }
        if (const char* s = std::getenv("WAN_SLA_PERHEAD")) {
            c.per_head = (s[0] == '1');
        }
        return c;
    }
};

// Set the bitmap bits for ONE Q-block from its K-block scores (top-k membership).
// `bitmap_row` points at this Q-block's n_kwords words. Always keeps the final
// K-tile (the kernel runs it unconditionally) so the live-fraction stat is honest.
inline void wan_sla_topk_row(const float* score, int n_kblk, int topk,
                             std::vector<int>& idx_scratch, uint32_t* bitmap_row) {
    for (int kb = 0; kb < n_kblk; ++kb) idx_scratch[kb] = kb;
    std::nth_element(idx_scratch.begin(), idx_scratch.begin() + (topk - 1), idx_scratch.begin() + n_kblk,
                     [&](int a, int b) { return score[a] > score[b]; });
    for (int t = 0; t < topk; ++t) {
        const int kb = idx_scratch[t];
        bitmap_row[kb >> 5] |= (1u << (kb & 31));
    }
    bitmap_row[(n_kblk - 1) >> 5] |= (1u << ((n_kblk - 1) & 31));
}

// ----------------------------------------------------------------------------
// Pure-CPU selector — faithful to sla_util.py get_block_map, with the heads
// collapsed (Stage 0 uses ONE shared bitmap, so per-head scores are summed and a
// single top-k is taken). No GPU; unit-testable in isolation by the main agent.
//
//   pooled_q / pooled_k : host F32, layout [d_head, n_blk, n_head] (ne0 fastest)
//                         pooled_k is already smooth-k (k - mean_token(k)).
//   out_bitmap          : resized to n_qtiles * n_kwords, indexed
//                         [jt * n_kwords + (kb >> 5)] |= (1u << (kb & 31)).
//   keep_frac           : 1 - sparsity (fraction of K-blocks kept).
// ----------------------------------------------------------------------------
inline void wan_sla_build_bitmap(const float* pooled_q,
                                 const float* pooled_k,
                                 int d_head,
                                 int n_qblk,
                                 int n_kblk,
                                 int n_head,
                                 float keep_frac,
                                 int n_kwords,
                                 std::vector<uint32_t>& out_bitmap) {
    const int n_qtiles = n_qblk;  // self-attn: Q-tile == Q-block (64 wide)
    out_bitmap.assign((size_t)n_qtiles * n_kwords, 0u);

    int topk = (int)std::floor(keep_frac * (double)n_kblk);
    if (topk < 1)      topk = 1;
    if (topk > n_kblk) topk = n_kblk;

    // [d, blk, h] strides: d fastest, then blk, then head.
    auto qv = [&](int d, int b, int h) {
        return pooled_q[((size_t)h * n_qblk + b) * d_head + d];
    };
    auto kv = [&](int d, int b, int h) {
        return pooled_k[((size_t)h * n_kblk + b) * d_head + d];
    };

    std::vector<float> score((size_t)n_kblk);
    std::vector<int>   idx((size_t)n_kblk);
    for (int qb = 0; qb < n_qblk; ++qb) {
        // Combined coarse score: sum over heads of pooled_q[qb]·pooled_k[kb].
        // (Stage-0 head collapse — Stage 1 / per-head keeps these separate.)
        for (int kb = 0; kb < n_kblk; ++kb) score[kb] = 0.0f;
        for (int h = 0; h < n_head; ++h) {
            for (int kb = 0; kb < n_kblk; ++kb) {
                float acc = 0.0f;
                for (int d = 0; d < d_head; ++d) acc += qv(d, qb, h) * kv(d, kb, h);
                score[kb] += acc;
            }
        }
        wan_sla_topk_row(score.data(), n_kblk, topk, idx, &out_bitmap[(size_t)qb * n_kwords]);
    }
}

// ----------------------------------------------------------------------------
// Stage 1 per-head selector: a SEPARATE bitmap per attention head (faithful to
// sla_util.py, which selects per head). Same pooled inputs [d_head, n_blk, n_head].
//   out_bitmap : resized to n_head * n_qtiles * n_kwords, head-major:
//                [(h*n_qtiles + jt)*n_kwords + (kb>>5)]  (matches the kernel's
//                per-head offset head*n_qtiles*n_kwords).
// ----------------------------------------------------------------------------
inline void wan_sla_build_bitmap_perhead(const float* pooled_q,
                                         const float* pooled_k,
                                         int d_head,
                                         int n_qblk,
                                         int n_kblk,
                                         int n_head,
                                         float keep_frac,
                                         int n_kwords,
                                         std::vector<uint32_t>& out_bitmap) {
    const int n_qtiles = n_qblk;
    out_bitmap.assign((size_t)n_head * n_qtiles * n_kwords, 0u);

    int topk = (int)std::floor(keep_frac * (double)n_kblk);
    if (topk < 1)      topk = 1;
    if (topk > n_kblk) topk = n_kblk;

    auto qv = [&](int d, int b, int h) {
        return pooled_q[((size_t)h * n_qblk + b) * d_head + d];
    };
    auto kv = [&](int d, int b, int h) {
        return pooled_k[((size_t)h * n_kblk + b) * d_head + d];
    };

    std::vector<float> score((size_t)n_kblk);
    std::vector<int>   idx((size_t)n_kblk);
    for (int h = 0; h < n_head; ++h) {
        uint32_t* head_base = &out_bitmap[(size_t)h * n_qtiles * n_kwords];
        for (int qb = 0; qb < n_qblk; ++qb) {
            for (int kb = 0; kb < n_kblk; ++kb) {
                float acc = 0.0f;
                for (int d = 0; d < d_head; ++d) acc += qv(d, qb, h) * kv(d, kb, h);
                score[kb] = acc;
            }
            wan_sla_topk_row(score.data(), n_kblk, topk, idx, head_base + (size_t)qb * n_kwords);
        }
    }
}

// ----------------------------------------------------------------------------
// Per-render device state: owns the persistent pooled_q / pooled_k export
// tensors (written by the selector block each step) and the bitmap tensor
// (rebuilt on the host each step from the previous step's pooled values, then
// pushed to the FA kernel via the device-symbol setters). Lives across steps;
// the GGMLRunner reset path must NOT free these (register them persistent).
// ----------------------------------------------------------------------------
struct WanSlaState {
    WanSlaConfig          cfg;
    ggml_backend_t        backend  = nullptr;
    ggml_context*         ctx      = nullptr;
    ggml_backend_buffer_t buf      = nullptr;

    ggml_tensor* pooled_q = nullptr;  // [d_head, n_blk, n_head] F32
    ggml_tensor* pooled_k = nullptr;  // [d_head, n_blk, n_head] F32
    ggml_tensor* bitmap   = nullptr;  // [n_kwords, n_qtiles]  F32-typed (u32 bits)

    int64_t n_token = -1;
    int     d_head  = 0;
    int     n_head  = 0;
    int     n_blk   = 0;   // ceil(n_token / 64)  (== n_qblk == n_kblk == n_qtiles == n_ktiles)
    int     n_kwords = 0;

    bool    bitmap_ready = false;  // a bitmap has been built (≥1 dense capture done)
    int     step         = 0;      // 0-based denoise step for THIS runner instance
    double  last_live_frac = 1.0;  // fraction of K-tiles kept by the last bitmap (≈ keep_frac)
    std::vector<uint32_t> host_bitmap;
    std::vector<float>    host_pq, host_pk;

    WanSlaState()                              = default;
    WanSlaState(const WanSlaState&)            = delete;  // owns a backend buffer
    WanSlaState& operator=(const WanSlaState&) = delete;

    bool active() const { return cfg.enabled; }

    // (Re)allocate for a given self-attn sequence length. Idempotent on shape.
    // Returns the freshly-created tensors so the runner can register them as
    // persistent. n_tok must equal the selector block's WanSelfAttention n_token.
    void ensure(ggml_backend_t be, int64_t n_tok, int n_heads, int dh) {
        if (buf && n_token == n_tok && n_head == n_heads && d_head == dh) return;
        free_state();
        backend  = be;
        n_token  = n_tok;
        n_head   = n_heads;
        d_head   = dh;
        n_blk    = (int)((n_tok + WAN_SLA_BLK - 1) / WAN_SLA_BLK);
        const int n_ktiles = n_blk;
        n_kwords = (n_ktiles + 31) / 32;

        // Per-head (Stage 1) needs one bitmap slab per head: [n_kwords, n_qtiles*n_head],
        // head-major. Shared (Stage 0) is one slab [n_kwords, n_qtiles].
        const int bitmap_rows = cfg.per_head ? (n_blk * n_head) : n_blk;

        ggml_init_params p = { ggml_tensor_overhead() * 4 + 1024, nullptr, /*no_alloc=*/true };
        ctx = ggml_init(p);
        pooled_q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_head, n_blk, n_head);
        pooled_k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_head, n_blk, n_head);
        // F32-typed but holds 32-bit bitmap words (bit-cast), exactly like the
        // avatar BSA bitmap so the existing setter consumes it unchanged.
        bitmap   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_kwords, bitmap_rows);
        ggml_set_name(pooled_q, "wan_sla.pooled_q");
        ggml_set_name(pooled_k, "wan_sla.pooled_k");
        ggml_set_name(bitmap,   "wan_sla.bitmap");
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

        bitmap_ready = false;
        step         = 0;
        host_pq.assign((size_t)d_head * n_blk * n_head, 0.0f);
        host_pk.assign((size_t)d_head * n_blk * n_head, 0.0f);
        host_bitmap.assign((size_t)bitmap_rows * n_kwords, 0u);
    }

    // After a compute: read the pooled Q/K the graph just wrote and (re)build the
    // bitmap. In stale mode this feeds the NEXT step; in current_step mode it feeds
    // THIS step's pass-2 (the pooled values are identical in both passes — block-`sel`'s
    // Q/K depend only on x, which is unchanged — so the selection is exactly current-step).
    void update_from_pooled() {
        if (!active() || pooled_q == nullptr) return;
        ggml_backend_tensor_get(pooled_q, host_pq.data(), 0, ggml_nbytes(pooled_q));
        ggml_backend_tensor_get(pooled_k, host_pk.data(), 0, ggml_nbytes(pooled_k));
        if (getenv("WAN_SLA_POOLDBG")) {
            // Confirm the pooled export actually landed (non-zero) and carries per-head
            // variation. host_pq/pk layout [d_head, n_blk, n_head] (d fastest).
            auto l2 = [&](const std::vector<float>& v, int h) {
                double s = 0.0; size_t base = (size_t)h * n_blk * d_head;
                for (size_t i = 0; i < (size_t)n_blk * d_head; ++i) s += (double)v[base+i]*v[base+i];
                return std::sqrt(s);
            };
            double crossqd = 0.0;  // max |head0 - head1| of pooled_q
            if (n_head > 1) for (size_t i = 0; i < (size_t)n_blk*d_head; ++i)
                crossqd = std::max(crossqd, (double)std::fabs(host_pq[i] - host_pq[(size_t)n_blk*d_head + i]));
            fprintf(stderr, "[SLA-POOL] step=%d pq_l2[h0]=%.4g pq_l2[h1]=%.4g pk_l2[h0]=%.4g cross_head_q_maxabs=%.4g\n",
                    step, l2(host_pq,0), n_head>1?l2(host_pq,1):0.0, l2(host_pk,0), crossqd);
        }
        const float keep = 1.0f - cfg.sparsity;
        if (cfg.per_head) {
            wan_sla_build_bitmap_perhead(host_pq.data(), host_pk.data(), d_head, n_blk, n_blk,
                                         n_head, keep, n_kwords, host_bitmap);
        } else {
            wan_sla_build_bitmap(host_pq.data(), host_pk.data(), d_head, n_blk, n_blk,
                                 n_head, keep, n_kwords, host_bitmap);
        }
        ggml_backend_tensor_set(bitmap, host_bitmap.data(), 0, ggml_nbytes(bitmap));
        size_t bits = 0;
        for (uint32_t w : host_bitmap) bits += (size_t)__builtin_popcount(w);
        const double n_tiles = (double)n_blk * (double)n_blk * (cfg.per_head ? (double)n_head : 1.0);
        last_live_frac = (double)bits / n_tiles;
        bitmap_ready = true;
        ++step;
    }

    // True when the NEXT compute should run sparse (stale mode: warmup elapsed + bitmap ready).
    bool sparse_now() const {
        return active() && bitmap_ready && step >= cfg.warmup_steps;
    }

    // Push (sparse=true) or clear (false) the device-symbol state. Call before each
    // compute. When sparse the kernel consults the bitmap on the maskless self-attn
    // path, scoped to this sequence's K-tile count (cross-attn untouched) and, when
    // per_head, head-indexed.
    void apply_device_sparse(bool sparse) {
#ifdef SD_USE_CUDA
        if (getenv("WAN_SLA_HOSTDBG")) {
            fprintf(stderr, "[SLA-HOST] apply(sparse=%d) bitmap_ready=%d n_blk=%d n_kwords=%d n_qtiles=%d bitmap_data=%p per_head=%d\n",
                    (int)sparse, (int)bitmap_ready, n_blk, n_kwords, n_blk,
                    (void*)(bitmap ? bitmap->data : nullptr), (int)cfg.per_head);
        }
        if (sparse && bitmap_ready) {
            ggml_cuda_set_longcat_fa_bsa_bitmap(bitmap->data, n_blk, n_kwords);
            ggml_cuda_set_longcat_fa_bsa_mask_free(1, /*n_ktiles=*/n_blk);
            ggml_cuda_set_longcat_fa_bsa_n_heads(cfg.per_head ? n_head : 0);
        } else {
            ggml_cuda_set_longcat_fa_bsa_bitmap(nullptr, 0, 0);
            ggml_cuda_set_longcat_fa_bsa_mask_free(0, 0);
            ggml_cuda_set_longcat_fa_bsa_n_heads(0);
        }
#else
        (void)sparse;
#endif
    }

    void clear_device() {
#ifdef SD_USE_CUDA
        ggml_cuda_set_longcat_fa_bsa_bitmap(nullptr, 0, 0);
        ggml_cuda_set_longcat_fa_bsa_mask_free(0, 0);
        ggml_cuda_set_longcat_fa_bsa_n_heads(0);
#endif
    }

    void free_state() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
        buf = nullptr;
        ctx = nullptr;
        pooled_q = pooled_k = bitmap = nullptr;
        n_token = -1;
        bitmap_ready = false;
    }

    ~WanSlaState() { free_state(); }
};

}  // namespace sd
