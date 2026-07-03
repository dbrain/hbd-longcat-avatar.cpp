#ifndef __WAN_SHOTSTREAM_HPP__
#define __WAN_SHOTSTREAM_HPP__

#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "ggml_extend.hpp"
#include "model/common/rope.hpp"
#include "model/diffusion/wan.hpp"  // WAN::Wan, WanConfig, WanAttentionBlock::forward_causal

// ---------------------------------------------------------------------------
// KlingTeam ShotStream — causal streaming multi-shot T2V (Wan2.1-T2V-1.3B fork).
//
// Ground truth: REFERENCE-shotstream-inference.md + /tmp/shotstream-ref
//   (pipeline/causal_inference_ar.py, wan/modules/causal_model_change_rope.py).
//
// This runner drives the AR streaming loop on top of the stock WAN::Wan DiT:
//   shot loop → per-shot: build global(context) cache once (t=0) → 7 chunks of 3
//   latent frames → per chunk: 4-step warped DMD denoise + a clean (t=0) rewrite
//   that persists this chunk's CLEAN K/V into the rolling LOCAL cache.
//
// Dual cache (per §2):
//   * LOCAL  (kv_cache1)          — this shot's own chunks' clean K/V, grows per
//                                   chunk, reset each shot. 21 latent frames max.
//   * GLOBAL (kv_cache1_context)  — ≤6 clean history latent frames of prior shots,
//                                   prefilled once per shot, read-only.
// Attention = maskless SDPA over concat(local ++ current ++ global); block-causal
// semantics come structurally from cache contents (no explicit mask, no eviction
// for the shipped config — see §2.4).
// ---------------------------------------------------------------------------
namespace WAN_SHOTSTREAM {

    // Warped DMD timesteps (nominal [1000,740,500,260], warp+shift=8). §5.2.
    static const float SHOTSTREAM_WARPED_TS[4] = {1000.0f, 957.929f, 888.889f, 737.589f};

    struct ShotStreamConfig {
        int num_frame_per_block = 3;   // chunk = 3 latent frames
        int frames_per_shot     = 21;  // 7 chunks
        int local_attn_size     = 21;  // local cache = 7 chunks (no eviction)
        int context_frames      = 6;   // global cache = 2 chunks
        int condition_start_frame = 6; // generated frames sit at temporal pos 6+start (=9360/1560)
        int sink_size           = 0;   // dead code for shipped config
        float timestep_shift    = 8.0f;
        float rope_shot_theta    = 1.0f / 6.0f;  // θ per shot index (change_rope)
        int  n_steps            = 4;
    };

    // change_rope (§3.4): add the shot phase k·θ to the temporal RoPE band, in place.
    // embed_nd packs each position as (axes_dim_sum/2) complex channels × a 2×2 rotation
    // block (4 floats/channel, layout [c,-s,s,c] — see Rope::rope()); the first
    // axes_dim[0]/2 channels are the temporal band. Rotating a rope block by Δ=k·θ composes
    // the rotations: cos/sin → cos(φ+Δ)/sin(φ+Δ) (mathematically exact). Free function so the
    // offline RoPE-parity dump (examples/shotstream/main.cpp --dump-rope) exercises the SAME
    // code the runner uses. VERIFIED against the torch oracle goldens (rope_freqs_chunk_*).
    //
    // TWO phase modes:
    //   * UNIFORM (shot_per_frame == nullptr) — one phase shot_idx·θ over EVERY temporal
    //     frame. Used for generated chunks (all 3 frames = shot k) and single-source history.
    //     Byte-identical to the original code.
    //   * PER-FRAME (shot_per_frame != nullptr) — the ≤6 context latent frames each carry the
    //     k·θ phase of the shot they ORIGINATED from (3+-shot histories mix source shots; §3.4).
    //     Positions are frame-major (gen_vid_ids: idx = frame·h_len·w_len + …), so position p
    //     belongs to temporal frame p/(npos/nframes); its phase is shot_per_frame[frame]·θ.
    //     A frame with source-shot 0 gets no rotation ⇒ an all-shot-0 history (the 2-shot case)
    //     leaves pe byte-identical to the uniform-0 (no-op) path.
    inline void pe_add_shot_phase(std::vector<float>& pe, int shot_idx,
                                  const WAN::WanConfig& config, float rope_shot_theta, bool enabled,
                                  const std::vector<int>* shot_per_frame = nullptr) {
        if (!enabled || rope_shot_theta == 0.0f) return;
        const int chans  = (int)config.axes_dim_sum / 2;  // complex channels per position (64)
        const int t_half = config.axes_dim[0] / 2;        // temporal channels (22)
        const int stride = chans * 4;                     // floats per position (4/channel)
        const int npos   = (int)(pe.size() / stride);
        auto rotate_pos = [&](int p, float cd, float sd_) {
            for (int c = 0; c < t_half; ++c) {
                float* b  = &pe[(size_t)p * stride + (size_t)c * 4];  // [c, -s, s, c]
                float co  = b[0], si = b[2];
                float co2 = co * cd - si * sd_;  // cos(φ+Δ)
                float si2 = si * cd + co * sd_;  // sin(φ+Δ)
                b[0] = co2; b[1] = -si2; b[2] = si2; b[3] = co2;
            }
        };

        // PER-FRAME context phase: each temporal frame carries its source shot's k·θ.
        if (shot_per_frame != nullptr && !shot_per_frame->empty()) {
            const int nframes          = (int)shot_per_frame->size();
            const int tokens_per_frame = nframes > 0 ? npos / nframes : npos;
            for (int p = 0; p < npos; ++p) {
                int frame = tokens_per_frame > 0 ? p / tokens_per_frame : 0;
                if (frame >= nframes) frame = nframes - 1;
                const int sidx = (*shot_per_frame)[frame];
                if (sidx == 0) continue;  // phase 0 = identity rotation → pe unchanged
                const float delta = sidx * rope_shot_theta;
                rotate_pos(p, std::cos(delta), std::sin(delta));
            }
            return;
        }

        // UNIFORM phase (original path — byte-identical).
        if (shot_idx == 0) return;
        const float delta = shot_idx * rope_shot_theta;
        const float cd = std::cos(delta), sd_ = std::sin(delta);
        for (int p = 0; p < npos; ++p) rotate_pos(p, cd, sd_);
    }

    // Build the RoPE `pe` table for a `t`-latent-frame grid whose temporal positions start at
    // t_offset, then apply the change_rope shot phase (shot_idx·θ). Free function (shared by
    // the runner and the offline dump). config defaults (patch {1,2,2}, theta 10000,
    // axes_dim {44,42,42}) are exactly the ShotStream RoPE params.
    inline std::vector<float> pe_build_frames(const WAN::WanConfig& config, float rope_shot_theta,
                                              bool phase_enabled, int t, int h, int w,
                                              int t_offset, int shot_idx,
                                              const std::vector<int>* shot_per_frame = nullptr) {
        int pt = std::get<0>(config.patch_size);
        int ph = std::get<1>(config.patch_size);
        int pw = std::get<2>(config.patch_size);
        auto ids = Rope::gen_vid_ids(t, h, w, pt, ph, pw, 1, t_offset, 0, 0);
        auto pe  = Rope::embed_nd(ids, 1, static_cast<float>(config.theta), config.axes_dim);
        pe_add_shot_phase(pe, shot_idx, config, rope_shot_theta, phase_enabled, shot_per_frame);
        return pe;
    }

    struct ShotStreamRunner : public GGMLRunner {
        WAN::WanConfig config;
        WAN::Wan wan;
        ShotStreamConfig ss;
        std::string desc = "ShotStream-1.3B";

        // Persistent host-side KV caches (F16, per layer), ne [d_head, L, n_head].
        // ONLY used on the legacy host path (SHOTSTREAM_KV_HOST=1). The default path
        // holds the K/V RESIDENT on the GPU (res_local / res_ctx below).
        std::vector<sd::Tensor<ggml_fp16_t>> local_k, local_v;   // intra-shot rolling
        std::vector<sd::Tensor<ggml_fp16_t>> ctx_k, ctx_v;       // inter-shot global

        // ------------------------------------------------------------------
        // KV-CACHE RESIDENCY (default; A/B fallback SHOTSTREAM_KV_HOST=1).
        //
        // The legacy path stores each persisted chunk's clean K/V in a HOST F16 vector
        // and `make_input`-uploads the ENTIRE (growing) local+global cache on EVERY
        // forward. By chunk 6 that placed the whole ~6 GB cache into the single
        // compute-buffer as input leaves (all 30 layers live simultaneously) → an
        // 8.55 GB monolithic alloc that OOMs the full 81f (7-chunk) shot on the 12 GB
        // card, plus ~5 s/shot of wasted whole-cache H2D per forward (×5 forwards/chunk).
        //
        // Residency instead keeps each persisted chunk's K/V in its OWN persistent
        // ggml backend buffer on the runtime backend (mirrors LongCatAvatarRunner's
        // condkv_buf pattern): the buffers live ACROSS graph computes, so forward_block
        // READS the cache by concatenating the resident chunk tensors in-graph (leaves
        // that gallocr never places in the compute buffer, and per-layer concats that
        // gallocr frees after each block) instead of uploading host vectors. That moves
        // the ~6 GB of cache leaves OUT of the compute buffer (peak single alloc
        // 8.55 GB → ~1.6 GB) and eliminates the per-forward whole-cache re-upload.
        //
        // WRITE side is intentionally UNCHANGED byte-for-byte: the fresh K/V are still
        // exported as cut-marked outputs and read back to host F16 per-segment (need_kv
        // path preserved). We then upload just THAT chunk's K/V once into a new resident
        // buffer — so the verified readback/RoPE/prompt-context/chaining logic is
        // identical; only the STORAGE + the READ change. (A fully-resident in-graph
        // ggml_cpy append — condkv style — would additionally drop the ~1.6 GB/shot
        // new-chunk round-trip; deferred as it touches the segmented graph-cut planner.)
        struct KVChunk {
            ggml_context*         ctx = nullptr;
            ggml_backend_buffer_t buf = nullptr;
            std::vector<ggml_tensor*> k, v;  // [nL], each F16 [d_head, L, n_head]
            int64_t L = 0;
        };
        std::vector<KVChunk> res_local;  // intra-shot rolling (reset each shot)
        std::vector<KVChunk> res_ctx;    // inter-shot global (prefilled once per shot)
        // Default = resident. SHOTSTREAM_KV_HOST=1 forces the legacy host-upload path.
        bool kv_host_mode = (std::getenv("SHOTSTREAM_KV_HOST") != nullptr);

        // input holders that must outlive the get_graph lambda.
        std::vector<float> pe_hold;
        sd::Tensor<float>  ts_hold;

        std::mt19937 rng{1234};

        // change_rope shot-phase gate (§3.4). Now VERIFIED against the torch oracle
        // (rope_freqs_chunk_shot{0,1}); enabled by default. Set SHOTSTREAM_NO_ROPE_PHASE=1
        // to force the single-shot (no-phase) path for A/B.
        bool rope_phase_enabled = (std::getenv("SHOTSTREAM_NO_ROPE_PHASE") == nullptr);

        ShotStreamRunner(ggml_backend_t backend,
                         ggml_backend_t params_backend,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "model.diffusion_model")
            : GGMLRunner(backend, params_backend),
              config(WAN::WanConfig::detect_from_weights(tensor_storage_map, prefix)) {
            // Wan2.1-T2V-1.3B (stock flat-Wan naming; the ShotStream weights are merged).
            config.model_type = "t2v";
            config.dim        = 1536;
            config.eps        = 1e-06f;
            config.ffn_dim    = 8960;
            config.freq_dim   = 256;
            config.in_dim     = 16;
            config.num_heads  = 12;
            config.out_dim    = 16;
            config.text_len   = 512;
            wan = WAN::Wan(config);
            wan.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override { return desc; }
        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string prefix) {
            wan.get_param_tensors(tensors, prefix);
        }

        void reset_local_cache() { local_k.clear(); local_v.clear(); free_kv_chunks(res_local); }
        void reset_global_cache() { ctx_k.clear(); ctx_v.clear(); free_kv_chunks(res_ctx); }

        // ---- KV residency helpers -----------------------------------------
        // Free every resident chunk buffer in `cache` (unregisters its persistent
        // leaves first so the segmented executor doesn't hold stale pointers).
        void free_kv_chunks(std::vector<KVChunk>& cache) {
            for (auto& c : cache) {
                for (ggml_tensor* t : c.k) unregister_persistent_tensor(t);
                for (ggml_tensor* t : c.v) unregister_persistent_tensor(t);
                if (c.buf != nullptr) ggml_backend_buffer_free(c.buf);
                if (c.ctx != nullptr) ggml_free(c.ctx);
            }
            cache.clear();
        }

        // Upload one just-computed chunk's K/V (host F16, read back verbatim by the
        // need_kv path) into a fresh persistent GPU buffer and append it to `cache`.
        // Layout [d_head, L, n_head] matches read_named() and the host-path tensors, so
        // the resident bytes are identical to what the legacy cache would have held.
        void push_kv_chunk(std::vector<KVChunk>& cache,
                           std::vector<sd::Tensor<ggml_fp16_t>>& nk,
                           std::vector<sd::Tensor<ggml_fp16_t>>& nv) {
            int nL       = config.num_layers;
            int d_head   = (int)(config.dim / config.num_heads);
            int n_head   = (int)config.num_heads;
            KVChunk c;
            c.L = nk.empty() ? 0 : nk[0].shape()[1];
            ggml_init_params p = { ggml_tensor_overhead() * (size_t)(2 * nL + 8), nullptr, /*no_alloc=*/true };
            c.ctx = ggml_init(p);
            GGML_ASSERT(c.ctx != nullptr);
            c.k.resize(nL); c.v.resize(nL);
            for (int i = 0; i < nL; i++) {
                c.k[i] = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F16, d_head, c.L, n_head);
                c.v[i] = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F16, d_head, c.L, n_head);
                ggml_set_name(c.k[i], ("shotstream.reskv.k." + std::to_string(i)).c_str());
                ggml_set_name(c.v[i], ("shotstream.reskv.v." + std::to_string(i)).c_str());
            }
            c.buf = ggml_backend_alloc_ctx_tensors(c.ctx, runtime_backend);
            GGML_ASSERT(c.buf != nullptr);
            for (int i = 0; i < nL; i++) {
                ggml_backend_tensor_set(c.k[i], nk[i].data(), 0, ggml_nbytes(c.k[i]));
                ggml_backend_tensor_set(c.v[i], nv[i].data(), 0, ggml_nbytes(c.v[i]));
                // Register as runner-persistent so reset_segment_runtime_tensors (offload
                // path) leaves their buffer/data pointers intact between segments.
                register_persistent_tensor(c.k[i]);
                register_persistent_tensor(c.v[i]);
            }
            cache.push_back(std::move(c));
        }

        // Build the in-graph read view of a resident cache for layer `i`: the ordered
        // concat (oldest→newest) of its chunk tensors, byte-identical to what the host
        // path's rolling `concat(local_k[i], nk[i], 1)` produced. Returns the lone chunk
        // tensor directly when only one chunk is resident (no concat node). nullptr if
        // the cache is empty. The concat nodes live in compute_ctx and read persistent
        // leaves, so gallocr keeps only ~one live per block (freed after each block).
        ggml_tensor* resident_read(const std::vector<KVChunk>& cache, bool want_k, int i) {
            if (cache.empty()) return nullptr;
            ggml_tensor* acc = want_k ? cache[0].k[i] : cache[0].v[i];
            for (size_t c = 1; c < cache.size(); c++) {
                ggml_tensor* t = want_k ? cache[c].k[i] : cache[c].v[i];
                acc = ggml_concat(compute_ctx, acc, t, 1);
            }
            return acc;
        }

        // Token counts for logging (mode-agnostic).
        int64_t local_cache_tokens() const {
            if (kv_host_mode) return local_k.empty() ? 0 : local_k[0].shape()[1];
            int64_t n = 0; for (const auto& c : res_local) n += c.L; return n;
        }
        int64_t ctx_cache_tokens() const {
            if (kv_host_mode) return ctx_k.empty() ? 0 : ctx_k[0].shape()[1];
            int64_t n = 0; for (const auto& c : res_ctx) n += c.L; return n;
        }

        // Build the RoPE table for `t` latent frames whose temporal grid starts at
        // t_offset, with the change_rope shot phase (shot_idx·θ) applied to the temporal
        // band. Delegates to the shared free functions (see pe_build_frames above).
        std::vector<float> build_pe_frames(int t, int h, int w, int t_offset, int shot_idx,
                                           const std::vector<int>* shot_per_frame = nullptr) {
            return pe_build_frames(config, ss.rope_shot_theta, rope_phase_enabled,
                                   t, h, w, t_offset, shot_idx, shot_per_frame);
        }

        // Gaussian noise chunk [W,H,T,16] matching the latent layout.
        sd::Tensor<float> randn_chunk(int W, int H, int T, int C = 16) {
            sd::Tensor<float> n({(int64_t)W, (int64_t)H, (int64_t)T, (int64_t)C});
            std::normal_distribution<float> nd(0.0f, 1.0f);
            for (int64_t i = 0; i < n.numel(); ++i) n.data()[i] = nd(rng);
            return n;
        }

        // -------------------------------------------------------------------
        // Run one causal-block forward (one generator pass). Reads the local +
        // global caches, computes this chunk's flow/velocity, and (if persist_local)
        // appends this chunk's freshly-computed K/V to the LOCAL rolling cache.
        //   store_into_global — used by the context prefill: append K/V to the GLOBAL
        //                       cache instead of the local one (persist_local ignored).
        // Returns flow [W,H,T,16].
        sd::Tensor<float> forward_block(int n_threads,
                                        int t_offset,
                                        int shot_idx,
                                        const sd::Tensor<float>& x_block,
                                        float timestep,
                                        const sd::Tensor<float>& context,
                                        bool persist_local,
                                        bool store_into_global,
                                        const std::vector<int>* shot_phase_per_frame = nullptr) {
            int T = (int)x_block.shape()[2], H = (int)x_block.shape()[1], W = (int)x_block.shape()[0];
            int pt = std::get<0>(config.patch_size), ph = std::get<1>(config.patch_size), pw = std::get<2>(config.patch_size);
            int nfb = (T + pt / 2) / pt, h_len = (H + ph / 2) / ph, w_len = (W + pw / 2) / pw;
            int L_blk = nfb * h_len * w_len;
            int nL = config.num_layers;
            int d_head = (int)(config.dim / config.num_heads);
            int n_head = (int)config.num_heads;

            bool have_local = kv_host_mode ? !local_k.empty() : !res_local.empty();
            bool have_ctx   = kv_host_mode ? !ctx_k.empty()   : !res_ctx.empty();
            // Perf: the fresh K/V only need exporting + host readback when this forward
            // actually persists them (the clean rewrite → local, or the context prefill →
            // global). The 4 warped-DMD DENOISE steps per chunk (persist=false, store=false)
            // discard their K/V, so exporting them there was pure waste: 60 F16 cast ops +
            // ~0.9 GB D2H readback per forward, ×4 of every 5 forwards. Byte-identical to the
            // old path — new_kc/new_vc are still computed as the attention K/V; we just stop
            // marking them as graph outputs / reading them back when unused.
            const bool need_kv = persist_local || store_into_global;

            auto get_graph = [&]() -> ggml_cgraph* {
                ggml_cgraph* gf = new_graph_custom(WAN::WAN_GRAPH_SIZE);

                ggml_tensor* gx   = make_input(x_block);
                ggml_tensor* gctx = make_input(context);

                ts_hold = sd::Tensor<float>::from_vector(std::vector<float>{timestep});
                ggml_tensor* gts = make_input(ts_hold);

                pe_hold = build_pe_frames(T, H, W, t_offset, shot_idx, shot_phase_per_frame);
                int pos_len = (int)(pe_hold.size() / config.axes_dim_sum / 2);
                auto pe = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, config.axes_dim_sum / 2, pos_len);
                set_backend_tensor_data(pe, pe_hold.data());

                // Read the local + global caches as attention inputs. Legacy path uploads
                // the whole host vector per forward (make_input → compute buffer); the
                // resident path references the persistent chunk buffers in-graph (no
                // upload, chunk leaves stay OUT of the compute buffer). Both yield the
                // same F16 [d_head, L, n_head] tensor in the same oldest→newest order, so
                // forward_causal_block's to_f32+concat math is byte-identical.
                std::vector<ggml_tensor*> prev_kc, prev_vc, cond_kc, cond_vc;
                if (have_local) {
                    prev_kc.resize(nL); prev_vc.resize(nL);
                    for (int i = 0; i < nL; i++) {
                        if (kv_host_mode) { prev_kc[i] = make_input(local_k[i]); prev_vc[i] = make_input(local_v[i]); }
                        else              { prev_kc[i] = resident_read(res_local, true, i); prev_vc[i] = resident_read(res_local, false, i); }
                    }
                }
                if (have_ctx) {
                    cond_kc.resize(nL); cond_vc.resize(nL);
                    for (int i = 0; i < nL; i++) {
                        if (kv_host_mode) { cond_kc[i] = make_input(ctx_k[i]); cond_vc[i] = make_input(ctx_v[i]); }
                        else              { cond_kc[i] = resident_read(res_ctx, true, i); cond_vc[i] = resident_read(res_ctx, false, i); }
                    }
                }

                auto rctx = get_context();
                std::vector<ggml_tensor*> new_kc, new_vc;
                ggml_tensor* vel = wan.forward_causal_block(&rctx, gx, gts, gctx, pe,
                                                            prev_kc, prev_vc, cond_kc, cond_vc,
                                                            new_kc, new_vc);

                // Export each layer's fresh K/V (F16) as cut-marked, per-segment-readable
                // outputs (matches the S2V rolling-cache readback pattern) — ONLY when this
                // forward persists them (skip on the discarded denoise-step forwards).
                if (need_kv) {
                    for (int i = 0; i < nL; i++) {
                        ggml_tensor* nk1 = ggml_cast(rctx.ggml_ctx, new_kc[i], GGML_TYPE_F16);
                        ggml_tensor* nv1 = ggml_cast(rctx.ggml_ctx, new_vc[i], GGML_TYPE_F16);
                        ggml_set_output(nk1);
                        ggml_set_output(nv1);
                        ggml_build_forward_expand(gf, nk1);
                        ggml_build_forward_expand(gf, nv1);
                        sd::ggml_graph_cut::mark_graph_cut(nk1, "shotstream.blocks." + std::to_string(i) + ".out", "nk");
                        sd::ggml_graph_cut::mark_graph_cut(nv1, "shotstream.blocks." + std::to_string(i) + ".out", "nv");
                    }
                }
                ggml_set_output(vel);
                ggml_build_forward_expand(gf, vel);
                return gf;
            };

            std::vector<sd::Tensor<ggml_fp16_t>> nk(nL), nv(nL);
            std::vector<bool> got_k(nL, false), got_v(nL, false);
            auto cut_name = [](const std::string& g, const std::string& o) {
                return sd::ggml_graph_cut::make_graph_cut_name(g, o);
            };
            auto read_named = [&](ggml_cgraph* sg, const std::string& name,
                                  sd::Tensor<ggml_fp16_t>& dst) -> bool {
                int nn = ggml_graph_n_nodes(sg);
                for (int n = 0; n < nn; n++) {
                    ggml_tensor* t = ggml_graph_node(sg, n);
                    if (t && strcmp(t->name, name.c_str()) == 0 && t->buffer != nullptr) {
                        dst = sd::Tensor<ggml_fp16_t>({(int64_t)d_head, (int64_t)L_blk, (int64_t)n_head});
                        ggml_backend_tensor_get(t, dst.data(), 0, ggml_nbytes(t));
                        return true;
                    }
                }
                return false;
            };
            if (need_kv) {
                segment_readback_hook_ = [&](ggml_cgraph* sg) {
                    for (int i = 0; i < nL; i++) {
                        std::string g = "shotstream.blocks." + std::to_string(i) + ".out";
                        if (!got_k[i]) got_k[i] = read_named(sg, cut_name(g, "nk"), nk[i]);
                        if (!got_v[i]) got_v[i] = read_named(sg, cut_name(g, "nv"), nv[i]);
                    }
                };
            }

            static const bool ss_profile = (std::getenv("SHOTSTREAM_PROFILE") != nullptr);
            int64_t tf0 = ss_profile ? ggml_time_ms() : 0;
            auto vel_opt = GGMLRunner::compute<float>(get_graph, n_threads, true, false);
            if (ss_profile) {
                printf("  [FWD] t=%.0f persist=%d store_g=%d L_blk=%d loc=%d ctx=%d: %.2fs\n",
                       timestep, (int)persist_local, (int)store_into_global, L_blk,
                       (int)local_cache_tokens(), (int)ctx_cache_tokens(),
                       (ggml_time_ms() - tf0) / 1000.0);
            }
            segment_readback_hook_ = nullptr;
            if (!vel_opt.has_value() || vel_opt->empty()) return {};
            sd::Tensor<float> vel = std::move(*vel_opt);
            vel.reshape_({(int64_t)W, (int64_t)H, (int64_t)nfb, 16});

            if (store_into_global) {
                // append into the GLOBAL/context cache (context prefill path).
                if (kv_host_mode) {
                    if (ctx_k.empty()) { ctx_k = std::move(nk); ctx_v = std::move(nv); }
                    else {
                        for (int i = 0; i < nL; i++) {
                            ctx_k[i] = sd::ops::concat(ctx_k[i], nk[i], 1);
                            ctx_v[i] = sd::ops::concat(ctx_v[i], nv[i], 1);
                        }
                    }
                } else {
                    push_kv_chunk(res_ctx, nk, nv);  // resident: append one persistent chunk
                }
            } else if (persist_local) {
                if (kv_host_mode) {
                    if (local_k.empty()) { local_k = std::move(nk); local_v = std::move(nv); }
                    else {
                        for (int i = 0; i < nL; i++) {
                            local_k[i] = sd::ops::concat(local_k[i], nk[i], 1);
                            local_v[i] = sd::ops::concat(local_v[i], nv[i], 1);
                        }
                    }
                    // Window bound: local_attn_size chunks (7). Shipped config never evicts
                    // within a shot (21 latent frames == the cache size), so this is a safety
                    // clamp only; sink_size pinning is dead code (§2.4).
                    int64_t window_tokens = (int64_t)(ss.local_attn_size / ss.num_frame_per_block) * (int64_t)L_blk;
                    for (int i = 0; i < nL; i++) {
                        int64_t cur = local_k[i].shape()[1];
                        if (window_tokens > 0 && cur > window_tokens) {
                            local_k[i] = sd::ops::slice(local_k[i], 1, cur - window_tokens, cur);
                            local_v[i] = sd::ops::slice(local_v[i], 1, cur - window_tokens, cur);
                        }
                    }
                } else {
                    push_kv_chunk(res_local, nk, nv);  // resident: append one persistent chunk
                    // Window bound: same safety clamp as the host path, applied at chunk
                    // granularity. Shipped config = 7 chunks == local_attn_size (21 frames),
                    // so this never fires; if it did, drop the oldest resident chunk(s).
                    int max_chunks = ss.num_frame_per_block > 0
                                         ? ss.local_attn_size / ss.num_frame_per_block : 0;
                    while (max_chunks > 0 && (int)res_local.size() > max_chunks) {
                        KVChunk& old = res_local.front();
                        for (ggml_tensor* t : old.k) unregister_persistent_tensor(t);
                        for (ggml_tensor* t : old.v) unregister_persistent_tensor(t);
                        if (old.buf != nullptr) ggml_backend_buffer_free(old.buf);
                        if (old.ctx != nullptr) ggml_free(old.ctx);
                        res_local.erase(res_local.begin());
                    }
                }
            }
            return vel;
        }

        // Flow → x0: x0 = x_t − σ·flow (§5.3).
        static void to_x0(const sd::Tensor<float>& x_t, const sd::Tensor<float>& flow,
                          float sigma, sd::Tensor<float>& x0) {
            x0 = sd::Tensor<float>({x_t.shape()[0], x_t.shape()[1], x_t.shape()[2], x_t.shape()[3]});
            for (int64_t i = 0; i < x_t.numel(); ++i) x0.data()[i] = x_t.data()[i] - sigma * flow.data()[i];
        }

        // add_noise (§5.1): x_t = (1−σ)·x0 + σ·noise.
        void add_noise(const sd::Tensor<float>& x0, float sigma, sd::Tensor<float>& out) {
            auto noise = randn_chunk((int)x0.shape()[0], (int)x0.shape()[1], (int)x0.shape()[2]);
            out = sd::Tensor<float>({x0.shape()[0], x0.shape()[1], x0.shape()[2], x0.shape()[3]});
            for (int64_t i = 0; i < x0.numel(); ++i)
                out.data()[i] = (1.0f - sigma) * x0.data()[i] + sigma * noise.data()[i];
        }

        // -------------------------------------------------------------------
        // Denoise ONE chunk (3 latent frames) with the 4-step warped DMD schedule,
        // then persist its CLEAN K/V into the local cache (§3.3). Returns x0.
        sd::Tensor<float> run_chunk(int n_threads, int t_offset, int shot_idx,
                                    const sd::Tensor<float>& noise_chunk,
                                    const sd::Tensor<float>& context) {
            sd::Tensor<float> x = noise_chunk;  // σ = 1.0 at step 0
            sd::Tensor<float> x0;
            for (int i = 0; i < ss.n_steps; ++i) {
                float t = SHOTSTREAM_WARPED_TS[i];
                float sigma = t / 1000.0f;
                // denoise-step forwards read caches but do NOT persist (snapshot-free).
                auto flow = forward_block(n_threads, t_offset, shot_idx, x, t, context,
                                          /*persist_local=*/false, /*store_into_global=*/false);
                if (flow.empty()) return {};
                to_x0(x, flow, sigma, x0);
                if (i < ss.n_steps - 1) {
                    float sigma_next = SHOTSTREAM_WARPED_TS[i + 1] / 1000.0f;
                    add_noise(x0, sigma_next, x);  // re-noise to the next timestep
                }
            }
            // clean rewrite at t=0: overwrites this chunk's local-cache slot with clean K/V.
            (void)forward_block(n_threads, t_offset, shot_idx, x0, 0.0f, context,
                                /*persist_local=*/true, /*store_into_global=*/false);
            return x0;
        }

        // -------------------------------------------------------------------
        // Prefill the GLOBAL/context cache once per shot from `context_latents`
        // ([W,H,Nctx,16], clean), at timestep 0, temporal positions 0..Nctx-1 (§2.1).
        // context_shot_phases[i] = the SOURCE-shot index of context latent frame i; per §3.4
        // each frame carries the k·θ RoPE phase of the shot it ORIGINATED from. For a 2-shot
        // chain all 6 frames come from shot 0 → phases {0,0,0,0,0,0} (byte-identical to the old
        // uniform-0 path); shot 0's own zeroed context is also all-0. For 3+ shots the history
        // mixes source shots (e.g. 4th shot → {0,0,1,1,2,2}) so the phase is PER-FRAME — this
        // is the true-endless-chain fix. Size must equal context_latents temporal frames.
        void prefill_context(int n_threads, const std::vector<int>& context_shot_phases,
                             const sd::Tensor<float>& context_latents,
                             const sd::Tensor<float>& context) {
            reset_global_cache();
            // CRITICAL: the reference context prefill (kv_cache=None branch) computes the context
            // K/V by pure SELF-attention over the context frames ONLY — it does NOT attend to any
            // local/rolling cache. Our forward_block attends `prev_kc` (the local cache) whenever it
            // is non-empty, so if a PRIOR shot's local cache is still resident here (it is: run_shot
            // only resets it AFTER prefill), shot k>0's context frames would attend to shot k-1's
            // 21-frame local cache and every block after block 0 would store a CORRUPTED context K/V
            // (blocks read the previous block's attention output). That bleeds the prior shot into
            // the global cache → style drift + duplicated subject. Clear it so prefill is a clean
            // context self-attention, matching the reference. (run_shot re-clears it right after.)
            // SHOTSTREAM_KEEP_STALE_LOCAL=1 reproduces the pre-fix bug for A/B (do NOT clear).
            if (std::getenv("SHOTSTREAM_KEEP_STALE_LOCAL") == nullptr) reset_local_cache();
            if (context_latents.numel() == 0) return;
            // One t=0 forward over all context frames; store K/V into the global cache.
            // scalar shot_idx is unused when a per-frame phase vector is supplied (§3.4).
            (void)forward_block(n_threads, /*t_offset=*/0, /*shot_idx=*/0, context_latents, 0.0f, context,
                                /*persist_local=*/false, /*store_into_global=*/true,
                                /*shot_phase_per_frame=*/&context_shot_phases);
        }

        // -------------------------------------------------------------------
        // Generate ONE shot (21 latent frames = 7 chunks). Local cache reset here;
        // the global cache must already be prefilled by the caller (prefill_context).
        //   out_latent — [W,H,frames_per_shot,16] filled in place.
        sd::Tensor<float> run_shot(int n_threads, int shot_idx, int W, int H,
                                   const sd::Tensor<float>& context) {
            reset_local_cache();
            int nfb = ss.num_frame_per_block;
            int n_chunks = ss.frames_per_shot / nfb;
            sd::Tensor<float> out({(int64_t)W, (int64_t)H, (int64_t)ss.frames_per_shot, 16});
            for (int b = 0; b < n_chunks; ++b) {
                int f0 = b * nfb;
                // Generated frames sit at temporal positions condition_start_frame + start
                // (=6 + f0): the ≤6 context frames occupy 0..5 (§3.4). Positions reset each
                // shot; shots are separated in PHASE (k·θ), not by an ever-growing index.
                int t_off = f0 + ss.condition_start_frame;
                auto noise = randn_chunk(W, H, nfb);
                int64_t tc0 = ggml_time_ms();
                auto x0 = run_chunk(n_threads, t_off, shot_idx, noise, context);
                int64_t tc1 = ggml_time_ms();
                if (x0.empty()) { printf("ERROR: shot %d chunk %d empty\n", shot_idx, b); return {}; }
                int64_t local_tokens = local_cache_tokens();
                printf("[CHUNK] shot %d chunk %d/%d: %.2fs (local_cache=%lld tok)\n",
                       shot_idx, b, n_chunks, (tc1 - tc0) / 1000.0, (long long)local_tokens);
                sd::ops::slice_assign(&out, 2, f0, f0 + nfb, x0);
            }
            return out;
        }

        // -------------------------------------------------------------------
        // P1 block-0 op-parity probe (OFFLINE). Feed the torch oracle's fixed
        // block0_selfattn_input [dim, L_blk, 1] and run ONLY block 0's causal self-attn
        // with the shot-0 RoPE table (t_offset=6, phase 0), empty caches. Reads back:
        //   selfattn_out  ne[dim, L_blk, 1]        → golden block0_selfattn_out
        //   roped_k       ne[d_head, n_head, L_blk] → golden block0_roped_k (perm to [d,h,L])
        //   v             ne[d_head, n_head, L_blk] → golden block0_v
        // (ne is written by main as the reversed numpy shape). Cosine ≥ 0.999 = faithful.
        bool dump_block0(int n_threads, const sd::Tensor<float>& x_in,
                         sd::Tensor<float>& out_selfattn,
                         sd::Tensor<float>& out_roped_k,
                         sd::Tensor<float>& out_v) {
            int d_head = (int)(config.dim / config.num_heads);
            int n_head = (int)config.num_heads;
            // Golden geometry: 3 latent frames @ 60x104 → 3x30x52 = 4680 tokens, start=6.
            const int T = 3, H = 60, W = 104;
            int L_blk = 3 * 30 * 52;

            sd::Tensor<ggml_fp16_t> dummy;  // unused
            got_k_ = got_v_ = false;
            auto get_graph = [&]() -> ggml_cgraph* {
                ggml_cgraph* gf   = new_graph_custom(WAN::WAN_GRAPH_SIZE);
                ggml_tensor* gx   = make_input(x_in);
                pe_hold           = build_pe_frames(T, H, W, /*t_offset=*/ss.condition_start_frame, /*shot_idx=*/0);
                int pos_len       = (int)(pe_hold.size() / config.axes_dim_sum / 2);
                auto pe = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, config.axes_dim_sum / 2, pos_len);
                set_backend_tensor_data(pe, pe_hold.data());

                auto rctx = get_context();
                ggml_tensor *nk = nullptr, *nv = nullptr;
                ggml_tensor* out = wan.forward_block0_selfattn(&rctx, gx, pe, nk, nv);
                // nk/nv are ne[d_head, L_blk, n_head]; permute to [d_head, n_head, L_blk] so a
                // ne0-fastest flatten matches the golden numpy [.,L,n_head,d_head] (d fastest).
                ggml_tensor* nkp = ggml_cont(rctx.ggml_ctx, ggml_permute(rctx.ggml_ctx, nk, 0, 2, 1, 3));
                ggml_tensor* nvp = ggml_cont(rctx.ggml_ctx, ggml_permute(rctx.ggml_ctx, nv, 0, 2, 1, 3));
                ggml_set_name(nkp, "b0_roped_k");
                ggml_set_name(nvp, "b0_v");
                ggml_set_output(nkp); ggml_set_output(nvp);
                ggml_build_forward_expand(gf, nkp);
                ggml_build_forward_expand(gf, nvp);
                ggml_set_output(out);
                ggml_build_forward_expand(gf, out);
                return gf;
            };

            auto read_named = [&](ggml_cgraph* sg, const char* name, sd::Tensor<float>& dst,
                                  int64_t a, int64_t b, int64_t c) -> bool {
                int nn = ggml_graph_n_nodes(sg);
                for (int n = 0; n < nn; n++) {
                    ggml_tensor* t = ggml_graph_node(sg, n);
                    if (t && strcmp(t->name, name) == 0 && t->buffer != nullptr) {
                        dst = sd::Tensor<float>({a, b, c});
                        ggml_backend_tensor_get(t, dst.data(), 0, ggml_nbytes(t));
                        return true;
                    }
                }
                return false;
            };
            segment_readback_hook_ = [&](ggml_cgraph* sg) {
                if (!got_k_) got_k_ = read_named(sg, "b0_roped_k", out_roped_k, d_head, n_head, L_blk);
                if (!got_v_) got_v_ = read_named(sg, "b0_v", out_v, d_head, n_head, L_blk);
            };
            auto out_opt = GGMLRunner::compute<float>(get_graph, n_threads, true, false);
            segment_readback_hook_ = nullptr;
            if (!out_opt.has_value() || out_opt->empty()) return false;
            out_selfattn = std::move(*out_opt);
            out_selfattn.reshape_({(int64_t)config.dim, (int64_t)L_blk, 1});
            return got_k_ && got_v_;
        }
        bool got_k_ = false, got_v_ = false;
    };

}  // namespace WAN_SHOTSTREAM

#endif  // __WAN_SHOTSTREAM_HPP__
