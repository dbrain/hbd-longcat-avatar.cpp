#ifndef __SD_MODEL_DIFFUSION_HYMOTION_DECODE_HPP__
#define __SD_MODEL_DIFFUSION_HYMOTION_DECODE_HPP__

// ---------------------------------------------------------------------------
// HY-Motion latent [T,201] -> SMPL-H local joint rotations + root translation.
// Pure CPU, no ggml. Mirrors MotionFlowMatching._decode_o6dp.
//
// THE 201-DIM FRAME:
//     [  0:  3]  root translation
//     [  3:  9]  root_rot6d (global body orientation)
//     [  9:135]  21 local joint rotations, rot6d
//     [135:201]  22 joint positions -- THE REFERENCE DECODER NEVER READS THESE
//                (they are auxiliary training supervision; note std[135..137]==0
//                because joint 0 is the pelvis at the origin by construction)
//
// See hymotion.hpp TRAP 2 for why the rot6d layout is interleaved-columns and
// not the pytorch3d [0:3]/[3:6] convention. Getting it wrong yields a valid
// rotation matrix that is simply the wrong rotation.
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace HYMotion {

    // SMPL/SMPL-H/SMPL-X share these first 22 body joints, in this order.
    // This is the same basis puppy-eyetest/anim/tools/retarget_delta.py already
    // pins ("SMPL-X's first 22 body joints are the same joints, in the same
    // order, as SMPL/HumanML3D's 22"), which is why no adapter is needed.
    static const char* const SMPL22_JOINT_NAMES[22] = {
        "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
        "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
        "neck", "left_collar", "right_collar", "head", "left_shoulder",
        "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"};

    static const int SMPL22_PARENTS[22] = {
        -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19};

    struct Mat3 {
        // column-major-ish accessor: m[r][c]
        float m[3][3];
    };

    struct Quat {
        float w, x, y, z;  // pytorch3d / matrix_to_quaternion order
    };

    // rot6d -> R, EXACTLY utils/geometry.py:rot6d_to_rotation_matrix.
    //   x  = rot6d.view(3,2)      (row-major)  => x[i][j] = rot6d[2i+j]
    //   a1 = x[...,0] = (rot6d[0], rot6d[2], rot6d[4])
    //   a2 = x[...,1] = (rot6d[1], rot6d[3], rot6d[5])
    //   b1 = normalize(a1); b2 = normalize(a2 - (b1.a2) b1); b3 = b1 x b2
    //   R  = stack((b1,b2,b3), dim=-1)  => COLUMNS are b1,b2,b3
    inline Mat3 rot6d_to_matrix(const float* r) {
        const float a1[3] = {r[0], r[2], r[4]};
        const float a2[3] = {r[1], r[3], r[5]};

        float b1[3], b2[3], b3[3];
        float n1 = std::sqrt(a1[0] * a1[0] + a1[1] * a1[1] + a1[2] * a1[2]);
        // F.normalize's eps guard (default 1e-12)
        n1 = std::max(n1, 1e-12f);
        for (int i = 0; i < 3; ++i) {
            b1[i] = a1[i] / n1;
        }

        const float dot = b1[0] * a2[0] + b1[1] * a2[1] + b1[2] * a2[2];
        for (int i = 0; i < 3; ++i) {
            b2[i] = a2[i] - dot * b1[i];
        }
        float n2 = std::sqrt(b2[0] * b2[0] + b2[1] * b2[1] + b2[2] * b2[2]);
        n2       = std::max(n2, 1e-12f);
        for (int i = 0; i < 3; ++i) {
            b2[i] /= n2;
        }

        b3[0] = b1[1] * b2[2] - b1[2] * b2[1];
        b3[1] = b1[2] * b2[0] - b1[0] * b2[2];
        b3[2] = b1[0] * b2[1] - b1[1] * b2[0];

        Mat3 R;
        for (int i = 0; i < 3; ++i) {
            R.m[i][0] = b1[i];
            R.m[i][1] = b2[i];
            R.m[i][2] = b3[i];
        }
        return R;
    }

    // pytorch3d matrix_to_quaternion, returning (w,x,y,z). Uses the numerically
    // stable "largest diagonal" branch rather than the naive trace formula.
    inline Quat matrix_to_quaternion(const Mat3& R) {
        const float m00 = R.m[0][0], m01 = R.m[0][1], m02 = R.m[0][2];
        const float m10 = R.m[1][0], m11 = R.m[1][1], m12 = R.m[1][2];
        const float m20 = R.m[2][0], m21 = R.m[2][1], m22 = R.m[2][2];

        const float trace = m00 + m11 + m22;
        Quat q;
        if (trace > 0.0f) {
            float s = std::sqrt(trace + 1.0f) * 2.0f;
            q.w     = 0.25f * s;
            q.x     = (m21 - m12) / s;
            q.y     = (m02 - m20) / s;
            q.z     = (m10 - m01) / s;
        } else if (m00 > m11 && m00 > m22) {
            float s = std::sqrt(1.0f + m00 - m11 - m22) * 2.0f;
            q.w     = (m21 - m12) / s;
            q.x     = 0.25f * s;
            q.y     = (m01 + m10) / s;
            q.z     = (m02 + m20) / s;
        } else if (m11 > m22) {
            float s = std::sqrt(1.0f + m11 - m00 - m22) * 2.0f;
            q.w     = (m02 - m20) / s;
            q.x     = (m01 + m10) / s;
            q.y     = 0.25f * s;
            q.z     = (m12 + m21) / s;
        } else {
            float s = std::sqrt(1.0f + m22 - m00 - m11) * 2.0f;
            q.w     = (m10 - m01) / s;
            q.x     = (m02 + m20) / s;
            q.y     = (m12 + m21) / s;
            q.z     = 0.25f * s;
        }
        return q;
    }

    inline Mat3 quaternion_to_matrix(const Quat& q) {
        const float n = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
        const float w = q.w / n, x = q.x / n, y = q.y / n, z = q.z / n;
        Mat3 R;
        R.m[0][0] = 1 - 2 * (y * y + z * z);
        R.m[0][1] = 2 * (x * y - z * w);
        R.m[0][2] = 2 * (x * z + y * w);
        R.m[1][0] = 2 * (x * y + z * w);
        R.m[1][1] = 1 - 2 * (x * x + z * z);
        R.m[1][2] = 2 * (y * z - x * w);
        R.m[2][0] = 2 * (x * z - y * w);
        R.m[2][1] = 2 * (y * z + x * w);
        R.m[2][2] = 1 - 2 * (x * x + y * y);
        return R;
    }

    // Symmetric 4x4 eigen-decomposition by cyclic Jacobi; returns the eigenvector
    // for the LARGEST eigenvalue. Replaces np.linalg.eigh(A)[1][:, -1].
    // NB: the eigenvector's SIGN is arbitrary in both implementations (q and -q
    // are the same rotation) -- never element-wise compare quaternions against a
    // reference; compare rotation matrices or |dot|.
    inline void jacobi_eigh4_max(const double A_in[4][4], double out[4]) {
        double a[4][4], v[4][4];
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                a[i][j] = A_in[i][j];
                v[i][j] = (i == j) ? 1.0 : 0.0;
            }
        }
        for (int sweep = 0; sweep < 64; ++sweep) {
            double off = 0.0;
            for (int i = 0; i < 4; ++i) {
                for (int j = i + 1; j < 4; ++j) {
                    off += a[i][j] * a[i][j];
                }
            }
            if (off < 1e-24) {
                break;
            }
            for (int p = 0; p < 4; ++p) {
                for (int q = p + 1; q < 4; ++q) {
                    if (std::fabs(a[p][q]) < 1e-30) {
                        continue;
                    }
                    const double theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                    const double t     = (theta >= 0 ? 1.0 : -1.0) /
                                     (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                    const double c = 1.0 / std::sqrt(t * t + 1.0);
                    const double s = t * c;
                    for (int k = 0; k < 4; ++k) {
                        const double akp = a[k][p], akq = a[k][q];
                        a[k][p] = c * akp - s * akq;
                        a[k][q] = s * akp + c * akq;
                    }
                    for (int k = 0; k < 4; ++k) {
                        const double apk = a[p][k], aqk = a[q][k];
                        a[p][k] = c * apk - s * aqk;
                        a[q][k] = s * apk + c * aqk;
                    }
                    for (int k = 0; k < 4; ++k) {
                        const double vkp = v[k][p], vkq = v[k][q];
                        v[k][p] = c * vkp - s * vkq;
                        v[k][q] = s * vkp + c * vkq;
                    }
                }
            }
        }
        int best = 0;
        for (int i = 1; i < 4; ++i) {
            if (a[i][i] > a[best][best]) {
                best = i;
            }
        }
        for (int k = 0; k < 4; ++k) {
            out[k] = v[k][best];
        }
    }

    // utils/geometry.py:wavg_quaternion_markley
    inline Quat wavg_quaternion_markley(const std::vector<Quat>& Q, const std::vector<float>& weights) {
        double A[4][4] = {{0}};
        double wSum    = 0.0;
        for (size_t i = 0; i < Q.size(); ++i) {
            double q[4] = {Q[i].w, Q[i].x, Q[i].y, Q[i].z};
            if (q[0] < 0.0) {  // antipodal handling keys on q[0] == w
                for (int k = 0; k < 4; ++k) {
                    q[k] = -q[k];
                }
            }
            const double w_i = weights[i];
            for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 4; ++c) {
                    A[r][c] += w_i * q[r] * q[c];
                }
            }
            wSum += w_i;
        }
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                A[r][c] /= wSum;
            }
        }
        double e[4];
        jacobi_eigh4_max(A, e);
        Quat out;
        out.w = (float)e[0];
        out.x = (float)e[1];
        out.y = (float)e[2];
        out.z = (float)e[3];
        return out;
    }

    // utils/geometry.py:gaussian_kernel1d(sigma, order=0, radius)
    inline std::vector<float> gaussian_kernel1d(float sigma, int radius) {
        std::vector<float> phi((size_t)(2 * radius + 1));
        double sum = 0.0;
        for (int i = -radius; i <= radius; ++i) {
            const double v          = std::exp(-0.5 / (sigma * sigma) * (double)i * i);
            phi[(size_t)(i + radius)] = (float)v;
            sum += v;
        }
        for (auto& v : phi) {
            v = (float)(v / sum);
        }
        return phi;
    }

    // utils/geometry.py:quaternion_fix_continuity -- cumulative sign flip so
    // consecutive frames stay on the same hemisphere.
    inline void quaternion_fix_continuity(std::vector<Quat>& q) {
        if (q.size() <= 1) {
            return;
        }
        // The reference takes the dot products of consecutive RAW (unflipped)
        // frames, cumsum's the negative-dot flags, and applies that parity to
        // frames 1..T-1. Do exactly that -- comparing against the already-flipped
        // previous frame instead is a subtly different filter.
        std::vector<int> parity(q.size(), 0);
        int acc = 0;
        for (size_t t = 1; t < q.size(); ++t) {
            const float d = q[t].w * q[t - 1].w + q[t].x * q[t - 1].x + q[t].y * q[t - 1].y + q[t].z * q[t - 1].z;
            acc += (d < 0.0f) ? 1 : 0;
            parity[t] = acc % 2;
        }
        for (size_t t = 1; t < q.size(); ++t) {
            if (parity[t]) {
                q[t].w = -q[t].w;
                q[t].x = -q[t].x;
                q[t].y = -q[t].y;
                q[t].z = -q[t].z;
            }
        }
    }

    // utils/geometry.py:slice_seq_with_padding -- edge-replicate padding.
    inline std::vector<Quat> slice_seq_with_padding(const std::vector<Quat>& seq, int64_t mid, int64_t len) {
        std::vector<Quat> out((size_t)len);
        const int64_t half = len / 2;
        for (int64_t i = 0; i < len; ++i) {
            int64_t idx = mid - half + i;
            idx         = std::max<int64_t>(0, std::min<int64_t>((int64_t)seq.size() - 1, idx));
            out[(size_t)i] = seq[(size_t)idx];
        }
        return out;
    }

    // utils/motion_process.py:smooth_quats (sigma=1.0 -> radius 4, 9-tap)
    inline std::vector<Quat> smooth_quats(std::vector<Quat> q, float sigma) {
        if (q.empty() || sigma <= 0.0f) {
            return q;
        }
        quaternion_fix_continuity(q);

        const int radius     = (int)(4.0f * sigma + 0.5f);  // truncate=4.0
        const auto weights   = gaussian_kernel1d(sigma, radius);
        const int64_t klen   = (int64_t)weights.size();
        std::vector<Quat> res = q;

        for (int64_t fr = 0; fr < (int64_t)q.size(); ++fr) {
            auto win        = slice_seq_with_padding(q, fr, klen);
            const Quat& ref = win[(size_t)(klen / 2)];
            for (auto& w : win) {
                const float d = w.w * ref.w + w.x * ref.x + w.y * ref.y + w.z * ref.z;
                if (d < 0.0f) {
                    w.w = -w.w;
                    w.x = -w.x;
                    w.y = -w.y;
                    w.z = -w.z;
                }
            }
            res[(size_t)fr] = wavg_quaternion_markley(win, weights);
        }
        return res;
    }

    // Savitzky-Golay, scipy semantics for mode='interp': a polyorder polynomial is
    // least-squares fitted to each centred window; interior samples take the fit at
    // the centre, and the first/last half-window take the fit of the first/last
    // full window evaluated at their own offset (NOT edge padding).
    inline std::vector<float> savgol_filter(const std::vector<float>& y, int window, int polyorder) {
        const int64_t n = (int64_t)y.size();
        if (n < window) {
            return y;  // scipy would raise; be permissive for very short clips
        }
        const int half = window / 2;
        const int nc   = polyorder + 1;

        // Normal equations for the Vandermonde A[i][j] = x_i^j, x = -half..half.
        std::vector<double> AtA((size_t)(nc * nc), 0.0);
        for (int i = -half; i <= half; ++i) {
            double xp[16];
            xp[0] = 1.0;
            for (int j = 1; j < nc; ++j) {
                xp[j] = xp[j - 1] * i;
            }
            for (int r = 0; r < nc; ++r) {
                for (int c = 0; c < nc; ++c) {
                    AtA[(size_t)(r * nc + c)] += xp[r] * xp[c];
                }
            }
        }
        // Invert AtA (nc <= 6) by Gauss-Jordan.
        std::vector<double> inv((size_t)(nc * nc), 0.0);
        for (int i = 0; i < nc; ++i) {
            inv[(size_t)(i * nc + i)] = 1.0;
        }
        std::vector<double> M = AtA;
        for (int col = 0; col < nc; ++col) {
            int piv = col;
            for (int r = col + 1; r < nc; ++r) {
                if (std::fabs(M[(size_t)(r * nc + col)]) > std::fabs(M[(size_t)(piv * nc + col)])) {
                    piv = r;
                }
            }
            for (int c = 0; c < nc; ++c) {
                std::swap(M[(size_t)(col * nc + c)], M[(size_t)(piv * nc + c)]);
                std::swap(inv[(size_t)(col * nc + c)], inv[(size_t)(piv * nc + c)]);
            }
            const double d = M[(size_t)(col * nc + col)];
            for (int c = 0; c < nc; ++c) {
                M[(size_t)(col * nc + c)] /= d;
                inv[(size_t)(col * nc + c)] /= d;
            }
            for (int r = 0; r < nc; ++r) {
                if (r == col) {
                    continue;
                }
                const double f = M[(size_t)(r * nc + col)];
                for (int c = 0; c < nc; ++c) {
                    M[(size_t)(r * nc + c)] -= f * M[(size_t)(col * nc + c)];
                    inv[(size_t)(r * nc + c)] -= f * inv[(size_t)(col * nc + c)];
                }
            }
        }

        // coeffs(eval_x)[i] = v(eval_x)^T (AtA)^-1 A^T e_i : the linear weights that
        // map a window of samples to the fitted polynomial evaluated at eval_x.
        auto window_weights = [&](double ex) {
            std::vector<double> vx((size_t)nc);
            vx[0] = 1.0;
            for (int j = 1; j < nc; ++j) {
                vx[(size_t)j] = vx[(size_t)(j - 1)] * ex;
            }
            std::vector<double> u((size_t)nc, 0.0);  // u = (AtA)^-1 v
            for (int r = 0; r < nc; ++r) {
                for (int c = 0; c < nc; ++c) {
                    u[(size_t)r] += inv[(size_t)(r * nc + c)] * vx[(size_t)c];
                }
            }
            std::vector<double> w((size_t)window, 0.0);
            for (int i = -half; i <= half; ++i) {
                double xp = 1.0, acc = 0.0;
                for (int j = 0; j < nc; ++j) {
                    acc += u[(size_t)j] * xp;
                    xp *= i;
                }
                w[(size_t)(i + half)] = acc;
            }
            return w;
        };

        std::vector<float> out((size_t)n);
        const auto w_centre = window_weights(0.0);
        for (int64_t t = half; t < n - half; ++t) {
            double acc = 0.0;
            for (int i = 0; i < window; ++i) {
                acc += w_centre[(size_t)i] * y[(size_t)(t - half + i)];
            }
            out[(size_t)t] = (float)acc;
        }
        for (int k = 0; k < half; ++k) {
            const auto wl = window_weights((double)(k - half));  // first window, offset k
            double accl   = 0.0;
            for (int i = 0; i < window; ++i) {
                accl += wl[(size_t)i] * y[(size_t)i];
            }
            out[(size_t)k] = (float)accl;

            const auto wr = window_weights((double)(half - k));  // last window, offset from end
            double accr   = 0.0;
            for (int i = 0; i < window; ++i) {
                accr += wr[(size_t)i] * y[(size_t)(n - window + i)];
            }
            out[(size_t)(n - 1 - k)] = (float)accr;
        }
        return out;
    }

    struct DecodedMotion {
        int64_t frames = 0;
        int64_t joints = 22;
        int fps        = 30;
        std::vector<Quat> quats;   // [T*22] local, (w,x,y,z); index 0 is the root
        std::vector<float> root;   // [T*3] root translation
    };

    // MotionFlowMatching.decode_motion_from_latent + _decode_o6dp(num_joints=22,
    // rel_trans=False, should_apply_smooothing=True).
    //
    // latent: [T, 201] raw model output (NOT denormalised)
    // mean/std: [201] from the checkpoint buffers
    //
    // NOTE ON GROUND ALIGNMENT: the reference additionally shifts transl[...,1] by
    // the min-y of its WOODEN MANNEQUIN's posed vertices (body_model.py WoodenMesh).
    // We deliberately do not: that offset is a property of Tencent's mannequin mesh,
    // not of the motion, and we ground-align against the user's own rig downstream.
    // Consequence: our `root` differs from the reference's `transl` by a per-clip
    // constant in Y. Any golden captured on transl therefore embeds the mannequin
    // offset -- compare root VELOCITY, or re-add the offset, but do not expect the
    // absolute Y to match.
    inline DecodedMotion decode(const float* latent,
                                int64_t T,
                                const float* mean,
                                const float* std_,
                                int fps            = 30,
                                bool apply_smooth  = true,
                                float slerp_sigma  = 1.0f) {
        const int64_t D  = 201;
        const int64_t NJ = 22;

        // denorm, replicating the std<1e-3 -> 0 guard exactly (motion_diffusion.py:209)
        std::vector<float> den((size_t)(T * D));
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t d = 0; d < D; ++d) {
                const float s              = (std_[d] < 1e-3f) ? 0.0f : std_[d];
                den[(size_t)(t * D + d)] = latent[t * D + d] * s + mean[d];
            }
        }

        // rot6d = cat([root_rot6d (1), body_rot6d (21)]) -> [T, 22, 6]
        std::vector<Mat3> R((size_t)(T * NJ));
        for (int64_t t = 0; t < T; ++t) {
            const float* f = &den[(size_t)(t * D)];
            R[(size_t)(t * NJ + 0)] = rot6d_to_matrix(f + 3);  // root_rot6d [3:9]
            for (int64_t j = 1; j < NJ; ++j) {
                R[(size_t)(t * NJ + j)] = rot6d_to_matrix(f + 9 + (j - 1) * 6);  // [9:135]
            }
        }

        DecodedMotion out;
        out.frames = T;
        out.joints = NJ;
        out.fps    = fps;
        out.quats.assign((size_t)(T * NJ), Quat{1, 0, 0, 0});

        // slerp smoothing, per joint over time
        for (int64_t j = 0; j < NJ; ++j) {
            std::vector<Quat> track((size_t)T);
            for (int64_t t = 0; t < T; ++t) {
                track[(size_t)t] = matrix_to_quaternion(R[(size_t)(t * NJ + j)]);
            }
            if (apply_smooth) {
                // The reference fixes continuity once over the (T,J) block and then
                // again inside smooth_quats per joint; the first pass is idempotent
                // w.r.t. the second for a single track, so one call here matches.
                track = smooth_quats(track, slerp_sigma);
            }
            for (int64_t t = 0; t < T; ++t) {
                out.quats[(size_t)(t * NJ + j)] = track[(size_t)t];
            }
        }

        // root translation + savgol(window=11, polyorder=5)
        out.root.assign((size_t)(T * 3), 0.0f);
        for (int c = 0; c < 3; ++c) {
            std::vector<float> comp((size_t)T);
            for (int64_t t = 0; t < T; ++t) {
                comp[(size_t)t] = den[(size_t)(t * D + c)];
            }
            if (apply_smooth) {
                comp = savgol_filter(comp, 11, 5);
            }
            for (int64_t t = 0; t < T; ++t) {
                out.root[(size_t)(t * 3 + c)] = comp[(size_t)t];
            }
        }
        return out;
    }

    // Emit the clip JSON that puppy-eyetest/anim already plays, with quats as
    // xyzw (the player's order) rather than the wxyz used internally here.
    inline std::string to_clip_json(const DecodedMotion& m, const std::string& name) {
        std::string s;
        s.reserve((size_t)(m.frames * m.joints * 48));
        char buf[256];
        s += "{\n  \"name\": \"" + name + "\",\n";
        snprintf(buf, sizeof(buf), "  \"fps\": %d,\n", m.fps);
        s += buf;
        s += "  \"skeleton\": \"smplh22\",\n";
        s += "  \"bones\": [";
        for (int64_t j = 0; j < m.joints; ++j) {
            s += (j ? ", \"" : "\"");
            s += SMPL22_JOINT_NAMES[j];
            s += "\"";
        }
        s += "],\n  \"parents\": [";
        for (int64_t j = 0; j < m.joints; ++j) {
            snprintf(buf, sizeof(buf), "%s%d", j ? ", " : "", SMPL22_PARENTS[j]);
            s += buf;
        }
        s += "],\n  \"quats\": [";
        for (int64_t t = 0; t < m.frames; ++t) {
            s += (t ? ",\n    [" : "\n    [");
            for (int64_t j = 0; j < m.joints; ++j) {
                const Quat& q = m.quats[(size_t)(t * m.joints + j)];
                snprintf(buf, sizeof(buf), "%s[%.6f,%.6f,%.6f,%.6f]", j ? "," : "", q.x, q.y, q.z, q.w);
                s += buf;
            }
            s += "]";
        }
        s += "\n  ],\n  \"root_pos\": [";
        for (int64_t t = 0; t < m.frames; ++t) {
            snprintf(buf, sizeof(buf), "%s[%.6f,%.6f,%.6f]", t ? ", " : "",
                     m.root[(size_t)(t * 3 + 0)], m.root[(size_t)(t * 3 + 1)], m.root[(size_t)(t * 3 + 2)]);
            s += buf;
        }
        s += "]\n}\n";
        return s;
    }

}  // namespace HYMotion

#endif  // __SD_MODEL_DIFFUSION_HYMOTION_DECODE_HPP__
