// rig_beam_generate_batched.hpp — BATCHED-BEAM variant of rig_beam_generate. Identical HF 5.9.0
// beam-search host logic (warpers, joint candidate pool, finished/early-stop bookkeeping — all reused
// from rig_beam_generate.hpp), but the GPU decode runs ALL active beams in ONE batched forward
// (qwen3_batched.hpp) instead of num_beams sequential single-token decodes.
//
// State layout: one shared read-only prefix cache `pre` ([hd,nkv,P]) + a DOUBLE-BUFFERED pair of
// batched suffix cache (`cur`, [hd,nkv,suffix_max,num_beams]) + a 1-column temp. A beam is a COLUMN index
// into `cur`. Each step: REORDER survivors' parent columns (suffix [0,step)) IN-PLACE within `cur` (1 temp
// column breaks cycles — saves a whole 2nd double-buffer), then batched-decode the S new tokens at suffix
// offset `step`. The forward is BIT-EXACT to the sequential shared-prefix decode, so results match exactly.
#pragma once
#include "qwen3_batched.hpp"
#include "rig_beam_generate.hpp"   // reuse ALL warper/grammar/log-softmax helpers + rig_vram_used_mib
#include "rig_grammar.hpp"
#ifdef M1_USE_CUDA
#include <cuda_runtime.h>
#endif
#include <vector>
#include <string>
#include <limits>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <memory>
#include <random>
#include <stdexcept>
#include <chrono>

namespace rig {

inline std::vector<int> rig_beam_generate_batched(M1Harness& H, const Qwen3Cfg& cfg,
                                                  const float* embed_table, const float* mesh_cond, int n_cond,
                                                  const std::vector<int>& start_tokens,
                                                  const GrammarSpec& spec = GrammarSpec(),
                                                  int num_beams = 10, int max_new_tokens = 600,
                                                  float length_penalty = 1.0f, float repetition_penalty = 2.0f,
                                                  bool do_sample = true, float temperature = 1.5f,
                                                  int top_k = 10, float top_p = 0.95f,
                                                  uint64_t seed = 0, bool verbose = true) {
    const int hidden = cfg.hidden, hd = cfg.head_dim, V = cfg.vocab;
    const int P = n_cond + (int)start_tokens.size();
    const int suffix_max = max_new_tokens + 4;
    std::mt19937_64 rng(seed);
    auto embed_row = [&](int tok) -> const float* { return embed_table + (size_t)tok * hidden; };

    // ---- shared prefix cache + ONE batched suffix cache + a 1-column temp (single-buffer in-place
    //      reorder: the 2nd full double-buffer was pure write-safety overhead ~= one suffix cache; an
    //      in-place column reorder with 1 temp column (cycle-break) does the same beam permutation). ----
    auto pre_u = std::make_unique<Qwen3KvCache>();          pre_u->init(H, cfg, P);
    auto bufA  = std::make_unique<Qwen3KvCacheBatched>();   bufA->init(H, cfg, suffix_max, num_beams);
    auto tmp_u = std::make_unique<Qwen3KvCacheBatched>();   tmp_u->init(H, cfg, suffix_max, 1);   // 1 col
    Qwen3KvCache* pre = pre_u.get();
    Qwen3KvCacheBatched* cur = bufA.get();
    Qwen3KvCacheBatched* tmpcol = tmp_u.get();

    bool weights_ready = false;
    double vram_peak = rig_vram_used_mib();
    long vram_calls = 0;
    auto vram_bump = [&]() { if (vram_calls++ < 8 || (vram_calls & 63) == 0) { double u = rig_vram_used_mib(); if (u > vram_peak) vram_peak = u; } };
    if (verbose) printf("[rig_beam_b] VRAM after pool alloc (1 batched suffix @ %dx%d + 1 temp col + shared prefix @ %d): %.0f MiB\n",
                        suffix_max, num_beams, P, vram_peak);

    // ============================ PREFILL the shared prefix cache (sequential, L=P) ============
    std::vector<float> prefix_embeds; prefix_embeds.reserve((size_t)P * hidden);
    prefix_embeds.insert(prefix_embeds.end(), mesh_cond, mesh_cond + (size_t)n_cond * hidden);
    for (int t : start_tokens) { const float* r = embed_row(t); prefix_embeds.insert(prefix_embeds.end(), r, r + hidden); }
    std::vector<float> prefix_logits;
    {
        std::vector<float> cosb, sinb; beam_rope_tables(cfg, 0, P, cosb, sinb);
        std::vector<float> maskb((size_t)P * P, 0.f);
        for (int q = 0; q < P; q++) for (int k = 0; k < P; k++) maskb[(size_t)q * P + k] = (k <= q) ? 0.0f : -INFINITY;
        ggml_init_params ip{ (size_t)512 * 1024 * 1024, nullptr, true };
        ggml_context* cctx = ggml_init(ip);
        int64_t ine[4] = { hidden, P, 1, 1 }; ggml_tensor* inp = ggml_new_tensor(cctx, GGML_TYPE_F32, 2, ine);
        ggml_set_input(inp);
        int64_t cne[4] = { hd, P, 1, 1 };
        ggml_tensor* cosT = ggml_new_tensor(cctx, GGML_TYPE_F32, 2, cne); ggml_set_input(cosT);
        ggml_tensor* sinT = ggml_new_tensor(cctx, GGML_TYPE_F32, 2, cne); ggml_set_input(sinT);
        ggml_tensor* cos3 = ggml_reshape_3d(cctx, cosT, hd, 1, P);
        ggml_tensor* sin3 = ggml_reshape_3d(cctx, sinT, hd, 1, P);
        int64_t mne[4] = { P, P, 1, 1 }; ggml_tensor* maskT = ggml_new_tensor(cctx, GGML_TYPE_F32, 2, mne); ggml_set_input(maskT);
        std::vector<ggml_tensor*> writes;
        ggml_tensor* logits = build_qwen3_cached(H, cctx, cfg, *pre, inp, cos3, sin3, 0, P, maskT, writes, nullptr, 0);
        ggml_set_output(logits);
        ggml_cgraph* gf = new_graph(cctx, 32768);
        ggml_build_forward_expand(gf, logits);
        for (ggml_tensor* w : writes) ggml_build_forward_expand(gf, w);
        if (!weights_ready) { upload_weights_maybe_f16(H); weights_ready = true; }
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(H.backend));
        if (!ggml_gallocr_alloc_graph(ga, gf)) { ggml_gallocr_free(ga); ggml_free(cctx); throw std::runtime_error("batched prefill gallocr failed"); }
        ggml_backend_tensor_set(inp, prefix_embeds.data(), 0, (size_t)P * hidden * sizeof(float));
        ggml_backend_tensor_set(cosT, cosb.data(), 0, cosb.size() * sizeof(float));
        ggml_backend_tensor_set(sinT, sinb.data(), 0, sinb.size() * sizeof(float));
        ggml_backend_tensor_set(maskT, maskb.data(), 0, maskb.size() * sizeof(float));
        ggml_backend_graph_compute(H.backend, gf);
        vram_bump();
        std::vector<float> all((size_t)P * V);
        ggml_backend_tensor_get(logits, all.data(), 0, (size_t)P * V * sizeof(float));
        prefix_logits.assign(all.begin() + (size_t)(P - 1) * V, all.begin() + (size_t)P * V);
        ggml_gallocr_free(ga); ggml_free(cctx);
        pre->pos = P;
    }
    if (verbose) printf("[rig_beam_b] prefill P=%d (n_cond=%d), suffix_max=%d, beams=%d seed=%llu\n",
                        P, n_cond, suffix_max, num_beams, (unsigned long long)seed);

    // ---- batched decode: B beams' next token at suffix offset `step`. embeds = [B*hidden] (col-major
    //      per beam). Writes into `suf` at col 0..B-1. Returns logits [B*V] (beam b at [b*V, b*V+V)). ----
    double t_build = 0, t_compute = 0; long n_decode = 0;   // per-step CPU build vs GPU compute split
    auto run_batched = [&](Qwen3KvCacheBatched& suf, int step, int B,
                           const std::vector<float>& embeds) -> std::vector<float> {
        auto _t0 = std::chrono::steady_clock::now();
        const int abs0 = P + step;
        std::vector<float> c1, s1; beam_rope_tables(cfg, abs0, 1, c1, s1);   // [hd]
        std::vector<float> cosb((size_t)hd * B), sinb((size_t)hd * B);
        for (int b = 0; b < B; ++b) { std::copy(c1.begin(), c1.end(), cosb.begin() + (size_t)b * hd);
                                      std::copy(s1.begin(), s1.end(), sinb.begin() + (size_t)b * hd); }
        ggml_init_params ip{ (size_t)512 * 1024 * 1024, nullptr, true };
        ggml_context* cctx = ggml_init(ip);
        int64_t ine[4] = { hidden, B, 1, 1 }; ggml_tensor* inp = ggml_new_tensor(cctx, GGML_TYPE_F32, 2, ine); ggml_set_input(inp);
        int64_t cne[4] = { hd, B, 1, 1 };
        ggml_tensor* cosT = ggml_new_tensor(cctx, GGML_TYPE_F32, 2, cne); ggml_set_input(cosT);
        ggml_tensor* sinT = ggml_new_tensor(cctx, GGML_TYPE_F32, 2, cne); ggml_set_input(sinT);
        ggml_tensor* cos3 = ggml_reshape_3d(cctx, cosT, hd, 1, B);
        ggml_tensor* sin3 = ggml_reshape_3d(cctx, sinT, hd, 1, B);
        std::vector<ggml_tensor*> writes;
        ggml_tensor* logits = build_qwen3_batched(H, cctx, cfg, suf, *pre, P, inp, cos3, sin3, step, B, writes);
        ggml_set_output(logits);
        ggml_cgraph* gf = new_graph(cctx, 32768);
        ggml_build_forward_expand(gf, logits);
        for (ggml_tensor* w : writes) ggml_build_forward_expand(gf, w);
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(H.backend));
        if (!ggml_gallocr_alloc_graph(ga, gf)) { ggml_gallocr_free(ga); ggml_free(cctx); throw std::runtime_error("batched decode gallocr failed"); }
        ggml_backend_tensor_set(inp, embeds.data(), 0, (size_t)B * hidden * sizeof(float));
        ggml_backend_tensor_set(cosT, cosb.data(), 0, cosb.size() * sizeof(float));
        ggml_backend_tensor_set(sinT, sinb.data(), 0, sinb.size() * sizeof(float));
        auto _t1 = std::chrono::steady_clock::now();
        ggml_backend_graph_compute(H.backend, gf);
        auto _t2 = std::chrono::steady_clock::now();
        vram_bump();
        std::vector<float> out((size_t)B * V);
        ggml_backend_tensor_get(logits, out.data(), 0, (size_t)B * V * sizeof(float));
        ggml_gallocr_free(ga); ggml_free(cctx);
        suf.pos = step + 1;
        t_build   += std::chrono::duration<double,std::milli>(_t1 - _t0).count();
        t_compute += std::chrono::duration<double,std::milli>(_t2 - _t1).count();
        n_decode++;
        return out;
    };

    // ---- copy suffix [0,step) of column src_col (in `src`) -> column dst_col (in `dst`); per layer K&V. ----
    std::vector<char> gat_host;
    auto copy_col = [&](Qwen3KvCacheBatched& dst, int dst_col, Qwen3KvCacheBatched& src, int src_col, int step) {
        if (step <= 0) return;
        for (int l = 0; l < cfg.n_layers; ++l) {
            ggml_tensor* sk = src.k_cache[l]; ggml_tensor* sv = src.v_cache[l];
            ggml_tensor* dk = dst.k_cache[l]; ggml_tensor* dv = dst.v_cache[l];
            size_t bytes = (size_t)step * sk->nb[2];
            size_t soff = (size_t)src_col * sk->nb[3], doff = (size_t)dst_col * dk->nb[3];
#ifdef M1_USE_CUDA
            cudaMemcpy((char*)dk->data + doff, (char*)sk->data + soff, bytes, cudaMemcpyDeviceToDevice);
            cudaMemcpy((char*)dv->data + doff, (char*)sv->data + soff, bytes, cudaMemcpyDeviceToDevice);
#else
            if (gat_host.size() < bytes) gat_host.resize(bytes);
            ggml_backend_tensor_get(sk, gat_host.data(), soff, bytes); ggml_backend_tensor_set(dk, gat_host.data(), doff, bytes);
            ggml_backend_tensor_get(sv, gat_host.data(), soff, bytes); ggml_backend_tensor_set(dv, gat_host.data(), doff, bytes);
#endif
        }
    };

    // ---- IN-PLACE beam reorder of `cur`: column c <- OLD column parent_col[c] (c in [0,S)), suffix [0,step).
    //      Single-buffer safe: never overwrite a column still needed as a source; break cycles with the
    //      1-column temp. Handles duplicate parents (one parent -> many children) and dropped beams.
    //      INVARIANT: buf[j] holds OLD[j] while pending[j]>0 and temp_src!=j (we only overwrite a column
    //      once nothing unwritten still needs its original value, or after it's been saved to temp). ----
    auto reorder_inplace = [&](Qwen3KvCacheBatched& buf, const std::vector<int>& parent_col, int prevS, int step) {
        if (step <= 0) return;
        const int S = (int)parent_col.size();
        std::vector<int> pending(std::max(prevS, S), 0);   // pending[j] = #unwritten dsts needing OLD[j]
        for (int c = 0; c < S; ++c) pending[parent_col[c]]++;
        std::vector<char> written(S, 0);
        int temp_src = -1, nwritten = 0;                   // temp_src = OLD col currently saved in tmpcol
        while (nwritten < S) {
            bool did = false;
            for (int c = 0; c < S; ++c) {
                if (written[c]) continue;
                int j = parent_col[c];
                if (c == j) {                              // identity: no copy, always safe
                    written[c] = 1; nwritten++; pending[j]--; did = true;
                    if (temp_src == j && pending[j] == 0) temp_src = -1;
                    continue;
                }
                if (c < prevS && pending[c] > 0 && temp_src != c) continue;  // would clobber a live source
                copy_col(buf, c, (temp_src == j ? *tmpcol : buf), (temp_src == j ? 0 : j), step);
                written[c] = 1; nwritten++; pending[j]--; did = true;
                if (temp_src == j && pending[j] == 0) temp_src = -1;
            }
            if (!did) {                                    // pure cycle -> save one live source to temp
                int pick = -1;
                for (int c = 0; c < S; ++c) if (!written[c] && c < prevS && pending[c] > 0) { pick = c; break; }
                if (pick < 0) break;                       // unreachable
                copy_col(*tmpcol, 0, buf, pick, step);
                temp_src = pick;
            }
        }
    };

    struct BBeam { int col; std::vector<int> sequence; std::vector<float> last_logits; double score; };

    // ============================ STEP 0: expand prefix -> beams ========================
    std::vector<RigHyp> finished;
    {
        std::vector<int> allowed = allowed_next_tokens(start_tokens, spec);
        std::vector<float> s0 = prefix_logits;
        beam_log_softmax_inplace(s0);
        beam_repetition_penalty(s0, start_tokens, repetition_penalty);
        apply_grammar_mask(s0, allowed);
        if (do_sample) { beam_warp_temperature(s0, temperature); beam_warp_top_k(s0, top_k); beam_warp_top_p(s0, top_p); }
        std::vector<std::pair<float,int>> cands;
        if (do_sample) {
            std::vector<int> drawn = beam_sample_distinct(s0, num_beams, rng);
            for (int t : drawn) if (std::isfinite(s0[t])) cands.push_back({ s0[t], t });
            std::sort(cands.begin(), cands.end(), [](auto&a, auto&b){ return a.first > b.first; });
        } else {
            for (int t : allowed) if (std::isfinite(s0[t])) cands.push_back({ s0[t], t });
            std::sort(cands.begin(), cands.end(), [](auto&a, auto&b){ return a.first > b.first; });
        }
        int nb = std::min((int)cands.size(), num_beams);
        if (nb == 0) throw std::runtime_error("rig_beam_generate_batched: no allowed tokens at step 0");

        // batched-decode the nb first tokens (suffix offset 0, fresh suffix in `cur`).
        cur->reset();
        std::vector<float> embeds((size_t)nb * hidden);
        for (int b = 0; b < nb; ++b) { const float* r = embed_row(cands[b].second); std::copy(r, r + hidden, embeds.begin() + (size_t)b * hidden); }
        std::vector<float> lg = run_batched(*cur, 0, nb, embeds);
        std::vector<BBeam> beams; beams.reserve(nb);
        for (int b = 0; b < nb; ++b) {
            BBeam beam; beam.col = b;
            beam.sequence = start_tokens; beam.sequence.push_back(cands[b].second);
            beam.last_logits.assign(lg.begin() + (size_t)b * V, lg.begin() + (size_t)(b + 1) * V);
            beam.score = cands[b].first;
            beams.push_back(std::move(beam));
        }
        if (verbose) printf("[rig_beam_b] step 0: spawned %zu beams (top token %d)\n", beams.size(), beams.empty() ? -1 : beams[0].sequence.back());
        if (std::getenv("RIG_LOGIT_PROBE") && !beams.empty()) {  // step-0 beam-0 decode-logit parity probe
            double s=0,ss=0; int am=0; float mx=-1e30f;
            for (size_t i=0;i<beams[0].last_logits.size();++i){ float v=beams[0].last_logits[i];
                s+=v; ss+=(double)v*v; if(v>mx){mx=v;am=(int)i;} }
            printf("[probe] BATCHED step0 beam0 tok=%d : sum=%.5f sumsq=%.5f argmax=%d max=%.6f\n",
                   beams[0].sequence.back(), s, ss, am, mx);
        }

        bool improvement_possible = true, stopped_early = false;
        const int beams_to_keep = 2 * num_beams;
        auto norm = [&](double score, int gen_len) { return score / std::pow((double)std::max(1, gen_len), (double)length_penalty); };
        auto worst_finished_norm = [&]() -> double { if (finished.empty()) return -1e30; double w = std::numeric_limits<double>::infinity(); for (auto& h : finished) w = std::min(w, h.normscore); return w; };
        auto bank_finished = [&](std::vector<int> seq, double normscore, bool eos) {
            finished.push_back({ std::move(seq), normscore, eos });
            if ((int)finished.size() > num_beams) { auto worst = std::min_element(finished.begin(), finished.end(), [](const RigHyp&a, const RigHyp&b){ return a.normscore < b.normscore; }); finished.erase(worst); }
        };

        for (int step = 1; step < max_new_tokens; ++step) {
            const int gen_len = step + 1;
            int active = (int)beams.size();
            if (active == 0) break;

            // ---- 1+2: joint candidate pool (acc = warped_logp + beam_score) ----
            struct JC { double acc; int parent; int tok; };
            std::vector<JC> jc;
            for (int i = 0; i < active; ++i) {
                std::vector<float> s = beams[i].last_logits;
                std::vector<int> allowed2 = allowed_next_tokens(beams[i].sequence, spec);
                beam_log_softmax_inplace(s);
                beam_repetition_penalty(s, beams[i].sequence, repetition_penalty);
                apply_grammar_mask(s, allowed2);
                if (do_sample) { beam_warp_temperature(s, temperature); beam_warp_top_k(s, top_k); beam_warp_top_p(s, top_p); }
                for (size_t t = 0; t < s.size(); ++t) if (std::isfinite(s[t])) jc.push_back({ beams[i].score + (double)s[t], i, (int)t });
            }
            if (jc.empty()) break;

            // ---- 3: pick beams_to_keep candidates from the joint pool ----
            int want = std::min((int)jc.size(), beams_to_keep);
            std::vector<int> picks; picks.reserve(want);
            if (do_sample) {
                std::vector<char> used(jc.size(), 0);
                double mx = -std::numeric_limits<double>::infinity();
                for (auto& c : jc) mx = std::max(mx, c.acc);
                std::uniform_real_distribution<double> uni(0.0, 1.0);
                for (int n = 0; n < want; ++n) {
                    double sum = 0.0; for (size_t k = 0; k < jc.size(); ++k) if (!used[k]) sum += std::exp(jc[k].acc - mx);
                    if (sum <= 0.0) break;
                    double u = uni(rng) * sum, accum = 0.0; int pick = -1;
                    for (size_t k = 0; k < jc.size(); ++k) { if (used[k]) continue; accum += std::exp(jc[k].acc - mx); if (u < accum) { pick = (int)k; break; } }
                    if (pick < 0) for (size_t k = jc.size(); k-- > 0; ) if (!used[k]) { pick = (int)k; break; }
                    if (pick < 0) break;
                    used[pick] = 1; picks.push_back(pick);
                }
            } else {
                std::vector<int> idx(jc.size()); for (size_t k = 0; k < jc.size(); ++k) idx[k] = (int)k;
                std::partial_sort(idx.begin(), idx.begin() + want, idx.end(), [&](int a, int b){ return jc[a].acc > jc[b].acc; });
                picks.assign(idx.begin(), idx.begin() + want);
            }

            // ---- 4: eos -> finished; non-eos top num_beams -> next active ----
            std::vector<std::pair<double,int>> noneos; noneos.reserve(picks.size());
            for (int pk : picks) {
                const JC& c = jc[pk];
                if (c.tok == spec.model_eos) { std::vector<int> seq = beams[c.parent].sequence; seq.push_back(spec.model_eos); bank_finished(std::move(seq), norm(c.acc, gen_len), true); }
                else noneos.push_back({ c.acc, pk });
            }
            std::sort(noneos.begin(), noneos.end(), [](const auto&a, const auto&b){ return a.first > b.first; });
            int nkeep = std::min((int)noneos.size(), num_beams);
            struct Survivor { int parent_col; int tok; double score; std::vector<int> seq; };
            std::vector<Survivor> survivors; survivors.reserve(nkeep);
            for (int r = 0; r < nkeep; ++r) {
                const JC& c = jc[noneos[r].second];
                std::vector<int> seq = beams[c.parent].sequence; seq.push_back(c.tok);
                survivors.push_back({ beams[c.parent].col, c.tok, c.acc, std::move(seq) });
            }
            const int S = (int)survivors.size();
            if (S == 0) { stopped_early = true; break; }

            // ---- REORDER parent columns IN-PLACE (suffix [0,step)) in `cur`, then BATCHED-decode S tokens. ----
            std::vector<int> parent_col(S); for (int k = 0; k < S; ++k) parent_col[k] = survivors[k].parent_col;
            reorder_inplace(*cur, parent_col, active, step);   // active = prev beam count
            std::vector<float> embeds2((size_t)S * hidden);
            for (int k = 0; k < S; ++k) { const float* r = embed_row(survivors[k].tok); std::copy(r, r + hidden, embeds2.begin() + (size_t)k * hidden); }
            std::vector<float> lg2 = run_batched(*cur, step, S, embeds2);
            std::vector<BBeam> next_beams; next_beams.reserve(S);
            for (int k = 0; k < S; ++k) {
                BBeam nbm; nbm.col = k; nbm.sequence = std::move(survivors[k].seq);
                nbm.last_logits.assign(lg2.begin() + (size_t)k * V, lg2.begin() + (size_t)(k + 1) * V);
                nbm.score = survivors[k].score;
                next_beams.push_back(std::move(nbm));
            }
            beams = std::move(next_beams);     // cur holds this step's beams (single buffer, reordered in place)

            // ---- 5: early-stop heuristic ----
            double best_active = -std::numeric_limits<double>::infinity();
            for (auto& b : beams) best_active = std::max(best_active, b.score);
            double best_active_norm = beams.empty() ? -1e30 : norm(best_active, gen_len);
            improvement_possible = improvement_possible && (best_active_norm > worst_finished_norm());
            if (verbose && (step < 3 || step % 64 == 0))
                printf("[rig_beam_b] step %d: active=%d finished=%zu worst_fin=%.4f best_act_norm=%.4f\n",
                       step, (int)beams.size(), finished.size(), finished.empty() ? 0.0 : worst_finished_norm(), best_active_norm);
            if (!improvement_possible) { stopped_early = true; break; }
            if (beams.empty())         { stopped_early = true; break; }
        }

        // ---- FINAL: bank still-running beams as max-len truncations only if budget exhausted ----
        bool any_eos = false; for (auto& h : finished) any_eos |= h.eos;
        if (!stopped_early) for (auto& b : beams) { int gl = (int)b.sequence.size() - (int)start_tokens.size(); bank_finished(b.sequence, norm(b.score, gl), false); }
        if (finished.empty()) throw std::runtime_error("rig_beam_generate_batched: no hypotheses produced");
        auto best = std::max_element(finished.begin(), finished.end(), [](const RigHyp& a, const RigHyp& b){ return a.normscore < b.normscore; });
        if (verbose) {
            int gl = (int)best->sequence.size() - (int)start_tokens.size();
            printf("[rig_beam_b] DONE: %zu hyps (eos in pool=%d), best %s normscore=%.4f, len=%zu (gen=%d), last=%d\n",
                   finished.size(), (int)any_eos, best->eos ? "EOS" : "trunc", best->normscore, best->sequence.size(), gl, best->sequence.empty() ? -1 : best->sequence.back());
            printf("[rig_beam_b] VRAM beam-phase peak: %.0f MiB used (1 batched suffix @ %dx%d + 1 temp col + shared prefix @ %d)\n",
                   vram_peak, suffix_max, num_beams, P);
            printf("[rig_beam_b] per-step split: build+alloc+upload=%.2f ms/step  compute=%.2f ms/step  (%ld decodes; build=%.0f%%)\n",
                   n_decode ? t_build/n_decode : 0.0, n_decode ? t_compute/n_decode : 0.0, n_decode,
                   (t_build+t_compute)>0 ? 100.0*t_build/(t_build+t_compute) : 0.0);
        }
        return best->sequence;
    }
}

} // namespace rig
