// MoGe-2 recover_focal_shift -> camera FOV. Pure host math (geometry_{torch,numpy}.py §5 of MOGE-ARCH.md).
// Given the dense point map + mask, recovers the camera focal (relative to image width) and z-shift by
// a 1-D Levenberg-Marquardt fit of a pinhole reprojection, then -> camera_angle_x.
#pragma once
#include <cmath>
#include <vector>
#include <cstdint>

namespace moge {

// nearest-neighbour downsample of a [H,W,C] map (row-major, C-fastest) to outxout, torch 'nearest'
// (src = floor(o * in/out)). Returns [out*out*C].
static inline std::vector<float> nearest_down(const float* src, int Hin, int Win, int C, int out) {
    std::vector<float> d((size_t)out * out * C);
    for (int oy = 0; oy < out; oy++) {
        int sy = (int)std::floor((double)oy * Hin / out);
        for (int ox = 0; ox < out; ox++) {
            int sx = (int)std::floor((double)ox * Win / out);
            const float* s = src + ((size_t)sy * Win + sx) * C;
            float* o = &d[((size_t)oy * out + ox) * C];
            for (int c = 0; c < C; c++) o[c] = s[c];
        }
    }
    return d;
}

// analytic view-plane uv at a single (col,row) of a (width,height) grid, AR = W/H. Matches
// normalized_view_plane_uv linspace endpoints.
static inline void uv_at(int col, int row, int width, int height, double AR, double& u, double& v) {
    double span_x = AR / std::sqrt(1.0 + AR * AR);
    double span_y = 1.0 / std::sqrt(1.0 + AR * AR);
    double ax = -span_x * (width - 1) / width,  bx = span_x * (width - 1) / width;
    double ay = -span_y * (height - 1) / height, by = span_y * (height - 1) / height;
    u = width  == 1 ? ax : ax + (bx - ax) * (double)col / (width - 1);
    v = height == 1 ? ay : ay + (by - ay) * (double)row / (height - 1);
}

// closed-form optimal focal for a given shift: f = sum(xy_proj*uv)/sum(xy_proj^2).
// xy_proj = xy/(z+shift). Returns f and fills residual r (len 2M: [fx*xp-u, fy*yp-v] interleaved per pt).
static inline double focal_for_shift(const std::vector<double>& xy, const std::vector<double>& z,
                                     const std::vector<double>& uv, double shift,
                                     std::vector<double>* r = nullptr) {
    int M = (int)z.size();
    double num = 0, den = 0;
    std::vector<double> xp(2 * M);
    for (int i = 0; i < M; i++) {
        double inv = 1.0 / (z[i] + shift);
        double xpx = xy[2 * i] * inv, xpy = xy[2 * i + 1] * inv;
        xp[2 * i] = xpx; xp[2 * i + 1] = xpy;
        num += xpx * uv[2 * i] + xpy * uv[2 * i + 1];
        den += xpx * xpx + xpy * xpy;
    }
    double f = num / den;
    if (r) {
        r->resize(2 * M);
        for (int i = 0; i < 2 * M; i++) (*r)[i] = f * xp[i] - uv[i];
    }
    return f;
}

static inline double cost_of(const std::vector<double>& r) {
    double c = 0; for (double v : r) c += v * v; return 0.5 * c;
}

// 1-D Levenberg-Marquardt on `shift` minimizing ||f(shift)*xy_proj - uv||^2 (numeric jacobian,
// matching scipy least_squares(method='lm', x0=0, ftol=1e-3)). Returns shift; sets *focal.
static inline double solve_focal_shift(const std::vector<double>& xy, const std::vector<double>& z,
                                       const std::vector<double>& uv, double* focal) {
    double shift = 0.0;
    std::vector<double> r; focal_for_shift(xy, z, uv, shift, &r);
    double cost = cost_of(r);
    double lambda = 1e-3;
    const double eps = 1e-6;
    for (int it = 0; it < 100; it++) {
        std::vector<double> rp, rm;
        focal_for_shift(xy, z, uv, shift + eps, &rp);
        focal_for_shift(xy, z, uv, shift - eps, &rm);
        // numeric jacobian J_i = d r_i / d shift; build J^T J (scalar) and J^T r.
        double JtJ = 0, Jtr = 0;
        for (size_t i = 0; i < r.size(); i++) {
            double j = (rp[i] - rm[i]) / (2 * eps);
            JtJ += j * j; Jtr += j * r[i];
        }
        bool stepped = false;
        for (int inner = 0; inner < 30; inner++) {
            double step = -Jtr / (JtJ + lambda);
            double ns = shift + step;
            std::vector<double> nr; focal_for_shift(xy, z, uv, ns, &nr);
            double nc = cost_of(nr);
            if (nc < cost) {
                double rel = (cost - nc) / (cost + 1e-30);
                shift = ns; r = nr; cost = nc; lambda = std::max(lambda * 0.5, 1e-12); stepped = true;
                if (rel < 1e-3 || std::fabs(step) < 1e-9) { it = 1000; }  // ftol-style stop
                break;
            }
            lambda *= 4.0;
        }
        if (!stepped) break;  // can't improve
    }
    *focal = focal_for_shift(xy, z, uv, shift, nullptr);
    return shift;
}

// bilinear sample a CHW map at fractional (fy,fx), border-clamped (torch align_corners=False).
static inline float bilinear_chw(const float* m, int C, int c, int Hh, int Wh, double fy, double fx) {
    double y = fy < 0 ? 0 : (fy > Hh - 1 ? Hh - 1 : fy);
    double x = fx < 0 ? 0 : (fx > Wh - 1 ? Wh - 1 : fx);
    int y0 = (int)std::floor(y), x0 = (int)std::floor(x);
    int y1 = y0 + 1 < Hh ? y0 + 1 : y0, x1 = x0 + 1 < Wh ? x0 + 1 : x0;
    double wy = y - y0, wx = x - x0;
    auto at = [&](int yy, int xx) { return (double)m[((size_t)c * Hh + yy) * Wh + xx]; };
    double top = at(y0, x0) * (1 - wx) + at(y0, x1) * wx;
    double bot = at(y1, x0) * (1 - wx) + at(y1, x1) * wx;
    return (float)(top * (1 - wy) + bot * wy);
}

// Full recovery from a full-res point map [H,W,3] + mask [H,W] (1=valid). AR = imgW/imgH.
// Returns fx_norm (= focal/2 * sqrt(1+AR^2)/AR) and camera_angle_x (radians).
struct FovResult { double focal, shift, fx_norm, camera_angle_x; int n_valid; };
static inline FovResult recover_fov(const float* points, const float* mask, int Himg, int Wimg, double AR) {
    const int LR = 64;
    std::vector<float> pts_lr = nearest_down(points, Himg, Wimg, 3, LR);
    // mask nearest-down then threshold (>0.5): matches interpolate(mask.float(),'nearest')>0 on the 0/1 map.
    std::vector<float> mask_lr = nearest_down(mask, Himg, Wimg, 1, LR);
    std::vector<double> xy, z, uv;
    for (int oy = 0; oy < LR; oy++) {
        for (int ox = 0; ox < LR; ox++) {
            if (mask_lr[(size_t)oy * LR + ox] <= 0.5f) continue;
            const float* p = &pts_lr[((size_t)oy * LR + ox) * 3];
            double u, v; uv_at(ox, oy, LR, LR, AR, u, v);
            xy.push_back(p[0]); xy.push_back(p[1]);
            z.push_back(p[2]);
            uv.push_back(u); uv.push_back(v);
        }
    }
    FovResult R{1.0, 0.0, 0.0, 0.0, (int)z.size()};
    if (R.n_valid < 2) { R.focal = 1.0; R.shift = 0.0; }
    else R.shift = solve_focal_shift(xy, z, uv, &R.focal);
    R.fx_norm = R.focal / 2.0 * std::sqrt(1.0 + AR * AR) / AR;
    R.camera_angle_x = 2.0 * std::atan(Wimg / (2.0 * R.fx_norm * Wimg));  // = 2*atan(1/(2*fx_norm))
    return R;
}

// Faithful recover from the RAW head outputs (the MoGe infer path): resize head Hh->Himg bilinear,
// remap(exp), recover_focal_shift (nearest-downsample to 64). remap is pointwise and the final
// downsample is nearest, so sample only the 64x64 needed points. points_chw/mask_chw = [3,Hh,Wh]/[1,Hh,Wh].
static inline FovResult recover_fov_from_heads(const float* points_chw, const float* mask_chw,
        int Hh, int Wh, int Himg, int Wimg, double AR) {
    const int LR = 64;
    std::vector<double> xy, z, uv;
    for (int oy = 0; oy < LR; oy++) {
        int sy1 = (int)std::floor((double)oy * Himg / LR);          // nearest Himg->64
        for (int ox = 0; ox < LR; ox++) {
            int sx1 = (int)std::floor((double)ox * Wimg / LR);
            double fy = (double)Hh / Himg * (sy1 + 0.5) - 0.5;      // bilinear head(Hh)->Himg coord
            double fx = (double)Wh / Wimg * (sx1 + 0.5) - 0.5;
            double m = bilinear_chw(mask_chw, 1, 0, Hh, Wh, fy, fx);// mask logit; sigmoid>0.5 == logit>0
            if (m <= 0.0) continue;
            double px = bilinear_chw(points_chw, 3, 0, Hh, Wh, fy, fx);
            double py = bilinear_chw(points_chw, 3, 1, Hh, Wh, fy, fx);
            double pz = bilinear_chw(points_chw, 3, 2, Hh, Wh, fy, fx);
            double zexp = std::exp(pz);                             // remap exp
            double u, v; uv_at(sx1, sy1, Wimg, Himg, AR, u, v);     // uv at full-res then nearest->64
            xy.push_back(px * zexp); xy.push_back(py * zexp);
            z.push_back(zexp);
            uv.push_back(u); uv.push_back(v);
        }
    }
    FovResult R{1.0, 0.0, 0.0, 0.0, (int)z.size()};
    if (R.n_valid >= 2) R.shift = solve_focal_shift(xy, z, uv, &R.focal);
    R.fx_norm = R.focal / 2.0 * std::sqrt(1.0 + AR * AR) / AR;
    R.camera_angle_x = 2.0 * std::atan(1.0 / (2.0 * R.fx_norm));
    return R;
}

}  // namespace moge
