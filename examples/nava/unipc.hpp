#ifndef __NAVA_UNIPC_HPP__
#define __NAVA_UNIPC_HPP__

// C++ port of NAVA's FlowUniPCMultistepScheduler (UniPC multistep sampler).
//
// Faithful port of nava_src/models/nava/utils/fm_solvers_unipc.py with the
// fixed NAVA config:
//   solver_order=2, predict_x0=True, solver_type="bh2", lower_order_final=True,
//   disable_corrector=[], final_sigmas_type="zero", prediction_type="flow_prediction",
//   num_train_timesteps=1000, sigma_min=0.003/1.002, sigma_max=1.0.
//
// All tensor math is elementwise over sd::Tensor<float>::data()[i] with scalar
// coefficients. model_outputs history is kept as two tensors (older + recent);
// emptiness flags the initial null state.
//
// Validated numerically against PyTorch to <1e-4 max-abs-error (see
// `nava unipc-test <dir>` in main.cpp).

#include <cmath>
#include <cstddef>
#include <vector>

#include "tensor.hpp"

struct UniPCSched {
    // public schedule, mirrors FlowMatchSched in main.cpp
    std::vector<float> sigmas;     // length N+1, sigmas[N] == 0
    std::vector<float> timesteps;  // length N (== sigmas[k]*1000)

    // ---- algorithm state ----
    int N            = 0;  // number of inference timesteps
    int step_index   = 0;
    int lower_order_nums = 0;
    int this_order   = 0;  // order chosen by the PREVIOUS step (used by corrector)

    // model_outputs history: index 0 = older, index 1 = most-recent x0.
    // empty() == null (initial warmup state).
    sd::Tensor<float> m_out_0;
    sd::Tensor<float> m_out_1;
    sd::Tensor<float> last_sample;  // empty() == null

    static float lam(float sig) {
        // lambda(sigma) = log(alpha) - log(sigma), alpha = 1 - sigma
        return std::log(1.0f - sig) - std::log(sig);
    }

    void set_timesteps(int n, float shift) {
        N = n;

        // sigma_max / sigma_min are NOT 1.0 / (0.003/1.002): they come from the
        // constructor's already-shift-transformed 1000-step training schedule
        // (fm_solvers_unipc.py __init__). The training raw sigmas are
        // (1 - alpha) for alpha = linspace(1, 1/T, T)[::-1], i.e. raw[0]=1-1/T,
        // raw[-1]=0. They get the shift transform once, then sigma_max/min are
        // taken from the ends. set_timesteps then linspaces between those and
        // applies the shift transform a SECOND time. Matched bit-for-bit (f32).
        const int T = 1000;
        // raw_first = 1 - 1/T (largest training sigma, computed in f32 to match torch)
        const float raw_first = 1.0f - 1.0f / static_cast<float>(T);
        const float raw_last  = 0.0f;
        const float sigma_max = shift * raw_first / (1.0f + (shift - 1.0f) * raw_first);
        const float sigma_min = shift * raw_last / (1.0f + (shift - 1.0f) * raw_last);  // == 0

        // s = linspace(sigma_max, sigma_min, N+1)[:-1]  -> N values, shift-transformed again
        std::vector<float> s(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            // np.linspace in f64 then cast; step = (min-max)/N
            double v = static_cast<double>(sigma_max) +
                       (static_cast<double>(sigma_min) - static_cast<double>(sigma_max)) *
                           (static_cast<double>(i) / static_cast<double>(n));
            float vf = static_cast<float>(v);
            s[static_cast<size_t>(i)] = shift * vf / (1.0f + (shift - 1.0f) * vf);
        }

        timesteps.resize(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            timesteps[static_cast<size_t>(i)] = s[static_cast<size_t>(i)] * 1000.0f;
        }

        // sigmas = concat([s, 0.0]) -> length N+1
        sigmas.resize(static_cast<size_t>(n) + 1);
        for (int i = 0; i < n; ++i) {
            sigmas[static_cast<size_t>(i)] = s[static_cast<size_t>(i)];
        }
        sigmas[static_cast<size_t>(n)] = 0.0f;

        // reset state
        m_out_0          = sd::Tensor<float>();
        m_out_1          = sd::Tensor<float>();
        last_sample      = sd::Tensor<float>();
        step_index       = 0;
        lower_order_nums = 0;
        this_order       = 0;
    }

    // convert_model_output: predict_x0 + flow_prediction -> x0 = sample - sigma_t*model_output
    sd::Tensor<float> convert(const sd::Tensor<float>& model_output,
                              const sd::Tensor<float>& sample,
                              int si) const {
        const float sigma = sigmas[static_cast<size_t>(si)];
        sd::Tensor<float> x0(sample.shape());
        const float* mo = model_output.data();
        const float* sm = sample.data();
        float* o        = x0.data();
        for (int64_t i = 0; i < x0.numel(); ++i) {
            o[i] = sm[i] - sigma * mo[i];
        }
        return x0;
    }

    // multistep_uni_p_bh_update (predictor); runs AFTER history append.
    // m0 = m_out_1 (most-recent x0), mi = m_out_0 (older).
    sd::Tensor<float> predictor(const sd::Tensor<float>& x, int order) const {
        const sd::Tensor<float>& m0 = m_out_1;

        const float sigma_s0 = sigmas[static_cast<size_t>(step_index)];
        const float sigma_t  = sigmas[static_cast<size_t>(step_index + 1)];
        const float alpha_t  = 1.0f - sigma_t;
        const float lambda_s0 = lam(sigma_s0);
        const float lambda_t  = lam(sigma_t);
        const float h        = lambda_t - lambda_s0;
        const float hh       = -h;  // predict_x0
        const float h_phi_1  = std::expm1(hh);
        const float B_h      = std::expm1(hh);  // bh2

        const float coef_x  = sigma_t / sigma_s0;
        const float coef_m0 = alpha_t * h_phi_1;

        sd::Tensor<float> x_t(x.shape());
        const float* xd  = x.data();
        const float* m0d = m0.data();
        float* o         = x_t.data();

        if (order >= 2) {
            const int si       = step_index - 1;
            const float rk     = (lam(sigmas[static_cast<size_t>(si)]) - lambda_s0) / h;
            const float* mid   = m_out_0.data();
            // D1 = (mi - m0)/rk ; rhos_p = [0.5] for order 2
            const float coef_D1 = alpha_t * B_h * 0.5f;
            for (int64_t i = 0; i < x_t.numel(); ++i) {
                const float D1 = (mid[i] - m0d[i]) / rk;
                o[i] = coef_x * xd[i] - coef_m0 * m0d[i] - coef_D1 * D1;
            }
        } else {
            for (int64_t i = 0; i < x_t.numel(); ++i) {
                o[i] = coef_x * xd[i] - coef_m0 * m0d[i];
            }
        }
        return x_t;
    }

    // multistep_uni_c_bh_update (corrector); runs BEFORE history append.
    // m0 = m_out_1 (still the previous step's most-recent x0), mi = m_out_0.
    sd::Tensor<float> corrector(const sd::Tensor<float>& model_t,
                                const sd::Tensor<float>& last_x,
                                int order) const {
        const sd::Tensor<float>& m0 = m_out_1;

        const float sigma_t  = sigmas[static_cast<size_t>(step_index)];
        const float sigma_s0 = sigmas[static_cast<size_t>(step_index - 1)];
        const float alpha_t  = 1.0f - sigma_t;
        const float lambda_s0 = lam(sigma_s0);
        const float lambda_t  = lam(sigma_t);
        const float h        = lambda_t - lambda_s0;
        const float hh       = -h;  // predict_x0
        const float h_phi_1  = std::expm1(hh);
        const float B_h      = std::expm1(hh);  // bh2

        const float coef_x  = sigma_t / sigma_s0;
        const float coef_m0 = alpha_t * h_phi_1;

        sd::Tensor<float> x_t(last_x.shape());
        const float* xd  = last_x.data();
        const float* m0d = m0.data();
        const float* mtd = model_t.data();
        float* o         = x_t.data();

        if (order == 1) {
            const float coef_D1t = alpha_t * B_h * 0.5f;  // rhos_c = [0.5]
            for (int64_t i = 0; i < x_t.numel(); ++i) {
                const float D1_t = mtd[i] - m0d[i];
                o[i] = coef_x * xd[i] - coef_m0 * m0d[i] - coef_D1t * D1_t;
            }
        } else {
            // order == 2: solve 2x2 R rhos_c = b
            const int si     = step_index - 2;
            const float rk   = (lam(sigmas[static_cast<size_t>(si)]) - lambda_s0) / h;
            const float* mid = m_out_0.data();

            float h_phi_k = h_phi_1 / hh - 1.0f;
            const float b1 = h_phi_k / B_h;
            h_phi_k        = h_phi_k / hh - 0.5f;
            const float b2 = h_phi_k * 2.0f / B_h;

            const float det = 1.0f - rk;
            const float rc0 = (b1 - b2) / det;
            const float rc1 = (b2 - rk * b1) / det;

            const float cAB = alpha_t * B_h;
            for (int64_t i = 0; i < x_t.numel(); ++i) {
                const float D1   = (mid[i] - m0d[i]) / rk;
                const float D1_t = mtd[i] - m0d[i];
                o[i] = coef_x * xd[i] - coef_m0 * m0d[i] - cAB * (rc0 * D1 + rc1 * D1_t);
            }
        }
        return x_t;
    }

    // step: returns prev_sample.
    sd::Tensor<float> step(const sd::Tensor<float>& model_output,
                           const sd::Tensor<float>& sample_in) {
        sd::Tensor<float> sample = sample_in;

        const bool use_corrector = (step_index > 0) && !last_sample.empty();

        sd::Tensor<float> x0 = convert(model_output, sample, step_index);

        if (use_corrector) {
            sample = corrector(x0, last_sample, this_order);
        }

        // shift history, append
        m_out_0 = m_out_1;
        m_out_1 = x0;

        int order;
        // lower_order_final: this_order = min(solver_order=2, N - step_index)
        order = std::min(2, N - step_index);
        order = std::min(order, lower_order_nums + 1);
        this_order = order;

        last_sample = sample;

        sd::Tensor<float> prev = predictor(sample, this_order);

        if (lower_order_nums < 2) {
            lower_order_nums++;
        }
        step_index++;
        return prev;
    }
};

#endif  // __NAVA_UNIPC_HPP__
