// motion_retarget.hpp — REST-RELATIVE ("delta") motion retargeting, in C++.
//
// A faithful port of /home/dbrain/dev/puppy-eyetest/anim/tools/{retarget_delta,bonemap_names,
// bonemap_v2}.py and the load_rig/quat_between/body_frame/smooth_quats half of
// /mnt/hdd/3d/avatar-shootout/retarget.py.  Header-only, no dependencies beyond glb_reader.hpp's
// JSON parser and rig_bone_names.hpp's SMPL-22 / Mixamo name tables.  Pure maths + glTF I/O: no
// GPU, no ggml, no Python.
//
// THE ALGORITHM, IN ONE LINE
//   stock look-at:   our_bone_global_aim := smpl_absolute_dir       (our rest is overwritten)
//   delta (here):    our_bone_global_rot := D_smpl * our_rest_rot   (our rest is preserved)
// with D_smpl = the SOURCE's swing from the SOURCE's own rest, in world space.  Feed the source's
// own rest in as the motion and every D is identity, so our rig sits at its own rest EXACTLY —
// the roundtrip-identity invariant (0.000000 deg) that makes this fully automatic.
//
// TWO WAYS TO GET D, both implemented, both the same D:
//  * POSITIONAL (MoMask/HumanML3D .npy — joint positions only): D = quat_between(u_s, v_s), the
//    minimal swing from the source's rest bone direction to its posed one.  Carries no twist and
//    leaves LEAF joints (head, wrists, feet) with no aim child and therefore no D at all.
//  * ROTATION-NATIVE (HY-Motion — local rotations): SMPL's rest IS its zero pose, so the source's
//    global rest rotation is identity at every joint and D_s(t) = Gsmpl_s(t) * Gsmpl_s(rest)^-1
//    = Gsmpl_s(t).  The FK'd global rotation already IS the quantity eq (1) wants.  Twist and leaf
//    joints survive.  Not a new algorithm — the same D, read instead of estimated.
//
// THE RETARGET BASE POSE (`rest_align`) is orthogonal to both and still required: eq (1)
// transplants a world-space swing measured from the SOURCE's rest onto OUR rest, which is only
// valid where the two rests point the same way.  A_b = slerp(I, K_b, alpha) with
// K_b = quat_between(our rest dir, SMPL rest dir) makes that assumption explicit:
//     Gpose[b] = D_s * A_b * Grest[b]                                          (eq 1')
// alpha=0 reproduces eq (1) bit-exactly; alpha=1 lands the bone on the source's absolute
// direction.  Per chain group (arm/leg/spine) because the trade differs per chain.
//
// NUMERICAL FIDELITY.  Every quaternion primitive below reproduces scipy.spatial.transform's
// exact formulation (composition without renormalisation, from_matrix's decision-matrix branch,
// from_rotvec/as_rotvec's small-angle series, apply() through the rotation MATRIX) because the
// acceptance test is "reproduce the Python output", and a mathematically-equivalent-but-different
// formula shows up as a real deviation once you look at 1e-7 rad.
//
// KNOWN FAITHFUL-TO-THE-PROTOTYPE BEHAVIOURS (reproduced deliberately, flagged here):
//  * load_rig composes rest globals from translation+rotation ONLY — node SCALE is ignored.
//    Harmless on every rig in this fleet (all five have no scaled joint node) but it is an
//    assumption, not a law.
//  * bone numbering: callers alias every joint node's name to `bone_<node index>` in memory, so a
//    "bone number" here IS a glTF node index.  Nothing on disk is renamed.
#pragma once

#include "glb_reader.hpp"      // glb::detail::JVal / JParser — the glTF JSON chunk
#include "rig_bone_names.hpp"  // rig::kSmplNames / kMixamoNames / kLRPairs — one table, not two

#include <algorithm>
#include <cmath>
#include <functional>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace mret {

// ===========================================================================
// 1.  Vector / quaternion maths — scipy.spatial.transform.Rotation, verbatim
// ===========================================================================

struct V3 {
    double x = 0, y = 0, z = 0;
    double  operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
    double& operator[](int i) { return i == 0 ? x : (i == 1 ? y : z); }
};
inline V3 operator+(const V3& a, const V3& b) { return V3{a.x + b.x, a.y + b.y, a.z + b.z}; }
inline V3 operator-(const V3& a, const V3& b) { return V3{a.x - b.x, a.y - b.y, a.z - b.z}; }
inline V3 operator*(const V3& a, double s) { return V3{a.x * s, a.y * s, a.z * s}; }
inline double dot(const V3& a, const V3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline V3 cross(const V3& a, const V3& b) {
    return V3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline double norm(const V3& a) { return std::sqrt(dot(a, a)); }

// xyzw, i.e. scipy's storage order.
struct Q4 {
    double x = 0, y = 0, z = 0, w = 1;
};

inline Q4 q_identity() { return Q4{0, 0, 0, 1}; }

// R.from_quat(): scipy NORMALISES on construction and preserves the sign.
inline Q4 q_norm(const Q4& q) {
    double n = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    if (n == 0.0) return q_identity();
    return Q4{q.x / n, q.y / n, q.z / n, q.w / n};
}

// scipy `_compose_quat` (Hamilton p*q).  NOT renormalised — scipy does not, and renormalising
// here would drift away from the reference in the last bits of every composed chain.
inline Q4 q_mul(const Q4& p, const Q4& q) {
    return Q4{p.w * q.x + q.w * p.x + p.y * q.z - p.z * q.y,
              p.w * q.y + q.w * p.y + p.z * q.x - p.x * q.z,
              p.w * q.z + q.w * p.z + p.x * q.y - p.y * q.x,
              p.w * q.w - (p.x * q.x + p.y * q.y + p.z * q.z)};
}

// Rotation.inv() for a unit quaternion = conjugate (scipy negates xyz).
inline Q4 q_inv(const Q4& q) { return Q4{-q.x, -q.y, -q.z, q.w}; }

// Rotation.as_matrix(), row-major m[3][3] — scipy's exact expression.
inline void q_matrix(const Q4& q, double m[9]) {
    const double x = q.x, y = q.y, z = q.z, w = q.w;
    const double x2 = x * x, y2 = y * y, z2 = z * z, w2 = w * w;
    const double xy = x * y, zw = z * w, xz = x * z, yw = y * w, yz = y * z, xw = x * w;
    m[0] = x2 - y2 - z2 + w2; m[1] = 2 * (xy - zw);      m[2] = 2 * (xz + yw);
    m[3] = 2 * (xy + zw);     m[4] = -x2 + y2 - z2 + w2; m[5] = 2 * (yz - xw);
    m[6] = 2 * (xz - yw);     m[7] = 2 * (yz + xw);      m[8] = -x2 - y2 + z2 + w2;
}

// Rotation.apply(v) — scipy builds the matrix and multiplies, so we do too (a direct
// v + 2*cross(...) form is algebraically identical but not bitwise identical).
inline V3 q_apply(const Q4& q, const V3& v) {
    double m[9];
    q_matrix(q, m);
    return V3{m[0] * v.x + m[1] * v.y + m[2] * v.z,
              m[3] * v.x + m[4] * v.y + m[5] * v.z,
              m[6] * v.x + m[7] * v.y + m[8] * v.z};
}

// Rotation.from_matrix() — scipy's decision-matrix branch, row-major input.
inline Q4 q_from_matrix(const double m[9]) {
    double d[4] = {m[0], m[4], m[8], m[0] + m[4] + m[8]};
    int choice = 0;
    for (int i = 1; i < 4; i++)
        if (d[i] > d[choice]) choice = i;
    double q[4];
    if (choice != 3) {
        const int i = choice, j = (i + 1) % 3, k = (j + 1) % 3;
        q[i] = 1 - d[3] + 2 * m[i * 3 + i];
        q[j] = m[j * 3 + i] + m[i * 3 + j];
        q[k] = m[k * 3 + i] + m[i * 3 + k];
        q[3] = m[k * 3 + j] - m[j * 3 + k];
    } else {
        q[0] = m[2 * 3 + 1] - m[1 * 3 + 2];
        q[1] = m[0 * 3 + 2] - m[2 * 3 + 0];
        q[2] = m[1 * 3 + 0] - m[0 * 3 + 1];
        q[3] = 1 + d[3];
    }
    const double n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    return Q4{q[0] / n, q[1] / n, q[2] / n, q[3] / n};
}

// Rotation.as_rotvec() — including scipy's Taylor branch below 1e-3 rad.
inline V3 q_as_rotvec(const Q4& qin) {
    Q4 q = qin;
    if (q.w < 0) { q.x = -q.x; q.y = -q.y; q.z = -q.z; q.w = -q.w; }
    const double n = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z);
    const double angle = 2.0 * std::atan2(n, q.w);
    double scale;
    if (angle <= 1e-3) {
        const double a2 = angle * angle;
        scale = 2 + a2 / 12 + 7 * a2 * a2 / 2880;
    } else {
        scale = angle / std::sin(angle / 2);
    }
    return V3{scale * q.x, scale * q.y, scale * q.z};
}

// Rotation.from_rotvec() — same Taylor branch.
inline Q4 q_from_rotvec(const V3& r) {
    const double n = norm(r);
    double scale;
    if (n <= 1e-3) {
        const double n2 = n * n;
        scale = 0.5 - n2 / 48 + n2 * n2 / 3840;
    } else {
        scale = std::sin(n / 2) / n;
    }
    return Q4{scale * r.x, scale * r.y, scale * r.z, std::cos(n / 2)};
}

// retarget.py's quat_between: shortest-arc rotation taking unit u to unit v.
// The two guard bands (parallel / antiparallel) and the fallback axis choice are load-bearing —
// they are what stops a degenerate bone from producing a NaN that then poisons a whole chain.
inline Q4 quat_between(const V3& uin, const V3& vin) {
    V3 u = uin * (1.0 / norm(uin));
    V3 v = vin * (1.0 / norm(vin));
    const double d = dot(u, v);
    if (d > 0.999999) return q_identity();
    if (d < -0.999999) {
        V3 axis = cross(u, V3{1, 0, 0});
        if (norm(axis) < 1e-6) axis = cross(u, V3{0, 1, 0});
        axis = axis * (1.0 / norm(axis));
        return q_from_rotvec(axis * M_PI);
    }
    V3 axis = cross(u, v);
    axis = axis * (1.0 / norm(axis));
    const double ang = std::acos(std::max(-1.0, std::min(1.0, d)));
    return q_from_rotvec(axis * ang);
}

// retarget.py's body_frame: world<-frame orthonormal basis, columns (right, up, fwd).
// Returned row-major so it can go straight into q_from_matrix.
inline void body_frame(const V3& pelvis, const V3& spine3, const V3& lhip, const V3& rhip,
                       double out[9]) {
    V3 up = spine3 - pelvis;
    up = up * (1.0 / norm(up));
    V3 right = rhip - lhip;
    right = right - up * dot(right, up);
    right = right * (1.0 / norm(right));
    V3 fwd = cross(right, up);
    fwd = fwd * (1.0 / norm(fwd));
    right = cross(up, fwd);
    out[0] = right.x; out[1] = up.x; out[2] = fwd.x;
    out[3] = right.y; out[4] = up.y; out[5] = fwd.y;
    out[6] = right.z; out[7] = up.z; out[8] = fwd.z;
}
inline void mat3_mul_bT(const double a[9], const double b[9], double out[9]) {  // a @ b.T
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            out[r * 3 + c] = a[r * 3 + 0] * b[c * 3 + 0] + a[r * 3 + 1] * b[c * 3 + 1] +
                             a[r * 3 + 2] * b[c * 3 + 2];
}

// ===========================================================================
// 2.  SMPL-22 constants
// ===========================================================================

// The kinematic child that defines each joint's aim direction.  -1 = leaf (no aim, no D).
// One table, shared by the named mapper, the topology walk and rest_align — bonemap_v2.py,
// bonemap_names.py and check_arms.py each carry a copy and they MUST agree.
inline const int* smpl_child_table() {
    static const int t[22] = {
        /*0 pelvis*/ -1, /*1 L_hip*/ 4,  /*2 R_hip*/ 5,  /*3 spine1*/ 6,
        /*4 L_knee*/ 7,  /*5 R_knee*/ 8, /*6 spine2*/ 9, /*7 L_ankle*/ 10,
        /*8 R_ankle*/ 11,/*9 spine3*/ 12,/*10 L_foot*/ -1,/*11 R_foot*/ -1,
        /*12 neck*/ 15,  /*13 L_collar*/ 16, /*14 R_collar*/ 17, /*15 head*/ -1,
        /*16 L_shoulder*/ 18, /*17 R_shoulder*/ 19, /*18 L_elbow*/ 20, /*19 R_elbow*/ 21,
        /*20 L_wrist*/ -1, /*21 R_wrist*/ -1};
    return t;
}

// Chain group of each DRIVEN (non-leaf) joint, for rest_align.  0=arm 1=leg 2=spine, -1=leaf.
enum { GRP_ARM = 0, GRP_LEG = 1, GRP_SPINE = 2 };
inline int smpl_group(int s) {
    switch (s) {
        case 13: case 16: case 18: case 14: case 17: case 19: return GRP_ARM;
        case 1: case 4: case 7: case 2: case 5: case 8:       return GRP_LEG;
        case 0: case 3: case 6: case 9: case 12:              return GRP_SPINE;
        default: return GRP_SPINE;   // matches SMPL_GROUP.get(s, 'spine') in retarget_delta.py
    }
}

// L/R mirror pairs — reuse rig_bone_names.hpp's table rather than restating it.
inline void lr_swap_inplace(std::vector<V3>& j, int stride22_count) {
    for (int t = 0; t < stride22_count; t++) {
        V3* p = &j[(size_t)t * 22];
        for (int k = 0; k < rig::LR_PAIR_N; k++) std::swap(p[rig::kLRPairs[k][0]], p[rig::kLRPairs[k][1]]);
    }
}
inline void lr_swap_inplace(std::vector<Q4>& q, int stride22_count) {
    for (int t = 0; t < stride22_count; t++) {
        Q4* p = &q[(size_t)t * 22];
        for (int k = 0; k < rig::LR_PAIR_N; k++) std::swap(p[rig::kLRPairs[k][0]], p[rig::kLRPairs[k][1]]);
    }
}

// ===========================================================================
// 3.  .npy / .npz readers (numpy's own container formats, no dependency)
// ===========================================================================

struct NpyArray {
    std::string          descr;        // e.g. "<f4"
    bool                 fortran = false;
    std::vector<int64_t> shape;
    std::vector<double>  data;         // always widened to double
    int64_t count() const {
        int64_t n = 1;
        for (int64_t s : shape) n *= s;
        return n;
    }
};

inline bool npy_parse(const uint8_t* buf, size_t len, NpyArray& out, std::string& err) {
    if (len < 10 || std::memcmp(buf, "\x93NUMPY", 6) != 0) { err = "not a .npy"; return false; }
    const int major = buf[6];
    size_t hlen, hoff;
    if (major == 1) { hlen = (size_t)buf[8] | ((size_t)buf[9] << 8); hoff = 10; }
    else            { hlen = (size_t)buf[8] | ((size_t)buf[9] << 8) | ((size_t)buf[10] << 16) |
                             ((size_t)buf[11] << 24); hoff = 12; }
    if (hoff + hlen > len) { err = "truncated .npy header"; return false; }
    std::string hdr((const char*)buf + hoff, hlen);
    auto grab = [&](const char* key) -> std::string {
        size_t p = hdr.find(key);
        if (p == std::string::npos) return "";
        p = hdr.find(':', p);
        return p == std::string::npos ? "" : hdr.substr(p + 1);
    };
    { std::string d = grab("'descr'");
      size_t a = d.find('\''); size_t b = d.find('\'', a + 1);
      if (a == std::string::npos || b == std::string::npos) { err = "no descr"; return false; }
      out.descr = d.substr(a + 1, b - a - 1); }
    out.fortran = hdr.find("'fortran_order': True") != std::string::npos;
    { std::string s = grab("'shape'");
      size_t a = s.find('('), b = s.find(')');
      if (a == std::string::npos || b == std::string::npos) { err = "no shape"; return false; }
      std::string body = s.substr(a + 1, b - a - 1);
      out.shape.clear();
      for (size_t i = 0; i < body.size();) {
          while (i < body.size() && (body[i] == ' ' || body[i] == ',')) i++;
          if (i >= body.size()) break;
          out.shape.push_back(std::strtoll(body.c_str() + i, nullptr, 10));
          while (i < body.size() && body[i] != ',') i++;
      } }
    const uint8_t* p = buf + hoff + hlen;
    const size_t   avail = len - (hoff + hlen);
    const int64_t  n = out.count();
    out.data.resize((size_t)n);
    if (out.descr == "<f4") {
        if (avail < (size_t)n * 4) { err = "truncated f4 data"; return false; }
        for (int64_t i = 0; i < n; i++) { float v; std::memcpy(&v, p + i * 4, 4); out.data[(size_t)i] = (double)v; }
    } else if (out.descr == "<f8") {
        if (avail < (size_t)n * 8) { err = "truncated f8 data"; return false; }
        for (int64_t i = 0; i < n; i++) { double v; std::memcpy(&v, p + i * 8, 8); out.data[(size_t)i] = v; }
    } else {
        err = "unsupported dtype " + out.descr;
        return false;
    }
    return true;
}

inline bool read_file(const char* path, std::vector<uint8_t>& out) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    out.resize((size_t)sz);
    bool ok = sz == 0 || std::fread(out.data(), 1, (size_t)sz, f) == (size_t)sz;
    std::fclose(f);
    return ok;
}

inline bool npy_load(const char* path, NpyArray& out, std::string& err) {
    std::vector<uint8_t> buf;
    if (!read_file(path, buf)) { err = std::string("cannot open ") + path; return false; }
    return npy_parse(buf.data(), buf.size(), out, err);
}

// Minimal STORED-member .npz reader.  np.savez writes members uncompressed (verified on
// SMPLX_NEUTRAL_2020.npz: every member has compress_type 0), so no inflate is needed and this
// stays dependency-free.  Zip64 extra fields are honoured because numpy writes them for large
// members even when the archive itself fits in 32 bits.
inline bool npz_read_member(const char* path, const char* member, NpyArray& out, std::string& err) {
    std::vector<uint8_t> z;
    if (!read_file(path, z)) { err = std::string("cannot open ") + path; return false; }
    auto u16 = [&](size_t o) { return (uint32_t)z[o] | ((uint32_t)z[o + 1] << 8); };
    auto u32 = [&](size_t o) { return (uint32_t)z[o] | ((uint32_t)z[o + 1] << 8) |
                                      ((uint32_t)z[o + 2] << 16) | ((uint32_t)z[o + 3] << 24); };
    auto u64 = [&](size_t o) { return (uint64_t)u32(o) | ((uint64_t)u32(o + 4) << 32); };

    // locate the End Of Central Directory record (scan back over the max 64k comment)
    size_t eocd = SIZE_MAX;
    for (size_t i = z.size() >= 22 ? z.size() - 22 : 0; ; i--) {
        if (i + 4 <= z.size() && z[i] == 'P' && z[i + 1] == 'K' && z[i + 2] == 5 && z[i + 3] == 6) { eocd = i; break; }
        if (i == 0 || (z.size() - i) > 66000) break;
    }
    if (eocd == SIZE_MAX) { err = "no zip EOCD"; return false; }
    uint64_t cd_off = u32(eocd + 16), n_ent = u16(eocd + 10);
    if (cd_off == 0xFFFFFFFFu) {  // zip64 EOCD locator sits 20 bytes before the EOCD
        if (eocd < 20) { err = "bad zip64 locator"; return false; }
        uint64_t z64 = u64(eocd - 20 + 8);
        cd_off = u64(z64 + 48); n_ent = (uint16_t)u64(z64 + 32);
    }
    size_t p = (size_t)cd_off;
    for (uint32_t e = 0; e < n_ent && p + 46 <= z.size(); e++) {
        if (!(z[p] == 'P' && z[p + 1] == 'K' && z[p + 2] == 1 && z[p + 3] == 2)) break;
        const uint32_t nlen = u16(p + 28), elen = u16(p + 30), clen = u16(p + 32);
        const uint32_t method = u16(p + 10);
        uint64_t csize = u32(p + 20), lho = u32(p + 42);
        std::string nm((const char*)&z[p + 46], nlen);
        if (nm == member) {
            // zip64 extra: fields appear in a fixed order, only for the members that need them
            if (csize == 0xFFFFFFFFu || lho == 0xFFFFFFFFu) {
                size_t ep = p + 46 + nlen, eend = ep + elen;
                while (ep + 4 <= eend) {
                    const uint32_t tag = u16(ep), sz = u16(ep + 2);
                    if (tag == 0x0001) {
                        size_t q = ep + 4;
                        /* uncompressed size */ if (q + 8 <= eend) q += 8;
                        if (csize == 0xFFFFFFFFu && q + 8 <= eend) { csize = u64(q); q += 8; }
                        if (lho   == 0xFFFFFFFFu && q + 8 <= eend) { lho   = u64(q); }
                        break;
                    }
                    ep += 4 + sz;
                }
            }
            if (method != 0) { err = "npz member is compressed (inflate not implemented)"; return false; }
            if ((size_t)lho + 30 > z.size()) { err = "bad local header offset"; return false; }
            const uint32_t lnlen = u16((size_t)lho + 26), lelen = u16((size_t)lho + 28);
            const size_t   doff = (size_t)lho + 30 + lnlen + lelen;
            if (doff + (size_t)csize > z.size()) { err = "member data out of range"; return false; }
            return npy_parse(&z[doff], (size_t)csize, out, err);
        }
        p += 46 + nlen + elen + clen;
    }
    err = std::string("npz member not found: ") + member;
    return false;
}

// SMPL(-X) rest ("zero-pose") joint positions, first 22, pelvis at the origin:
//     J = J_regressor @ v_template          (the model's own definition)
// float32 on disk, widened to float64 before the product — exactly what
// `d['J_regressor'].astype(np.float64) @ d['v_template'].astype(np.float64)` does.
inline bool smpl_rest_joints(const char* npz_path, std::vector<V3>& out, std::string& err) {
    NpyArray Jr, vt;
    if (!npz_read_member(npz_path, "J_regressor.npy", Jr, err)) return false;
    if (!npz_read_member(npz_path, "v_template.npy", vt, err)) return false;
    if (Jr.shape.size() != 2 || vt.shape.size() != 2 || vt.shape[1] != 3 ||
        Jr.shape[1] != vt.shape[0]) { err = "SMPL-X arrays have unexpected shapes"; return false; }
    const int64_t J = Jr.shape[0], V = Jr.shape[1];
    out.assign(22, V3{});
    for (int64_t j = 0; j < 22 && j < J; j++) {
        double acc[3] = {0, 0, 0};
        for (int64_t v = 0; v < V; v++) {
            // J_regressor is stored fortran_order=True in this file; index accordingly.
            const double w = Jr.fortran ? Jr.data[(size_t)(v * J + j)] : Jr.data[(size_t)(j * V + v)];
            if (w == 0.0) continue;
            acc[0] += w * vt.data[(size_t)(v * 3 + 0)];
            acc[1] += w * vt.data[(size_t)(v * 3 + 1)];
            acc[2] += w * vt.data[(size_t)(v * 3 + 2)];
        }
        out[(size_t)j] = V3{acc[0], acc[1], acc[2]};
    }
    const V3 root = out[0];
    for (auto& p : out) p = p - root;
    return true;
}

// ===========================================================================
// 4.  Rig loading — retarget.py's load_rig, plus bonemap_names.joint_node_indices
// ===========================================================================

struct Rig {
    glb::detail::JVal    gltf;        // the parsed JSON chunk (kept: the baker rewrites it)
    std::vector<uint8_t> bin;         // the BIN chunk, verbatim
    int                  N = 0;       // node count
    std::vector<std::string> orig;    // original node names ("" if unnamed)
    std::vector<int>     parent;      // node -> parent node, -1 at a root
    std::vector<V3>      Lt;          // local translation
    std::vector<Q4>      Lr;          // local rotation
    std::vector<V3>      Gpos;        // rest global position
    std::vector<Q4>      Grot;        // rest global rotation
    std::vector<int>     joints;      // skeleton node indices, ASCENDING (== the bone numbers)
    std::vector<char>    is_bone;     // node -> is it one of `joints`
    bool                 saw_scale = false;   // a scaled joint node would break the rest walk
    bool                 saw_matrix = false;  // matrix nodes take load_rig's other branch
};

inline bool glb_split(const char* path, std::vector<uint8_t>& file, glb::detail::JVal& json,
                      std::vector<uint8_t>& bin, std::string& err) {
    if (!read_file(path, file)) { err = std::string("cannot open ") + path; return false; }
    if (file.size() < 20 || std::memcmp(file.data(), "glTF", 4) != 0) { err = "not a GLB"; return false; }
    uint32_t off = 12;
    bool have_json = false;
    while (off + 8 <= file.size()) {
        uint32_t clen, ctype;
        std::memcpy(&clen, &file[off], 4);
        std::memcpy(&ctype, &file[off + 4], 4);
        const uint8_t* data = &file[off + 8];
        if (off + 8 + clen > file.size()) { err = "GLB chunk overruns file"; return false; }
        if (ctype == 0x4E4F534A) {  // 'JSON'
            glb::detail::JParser p((const char*)data, clen);
            if (!p.parse(json)) { err = "glTF JSON parse failed"; return false; }
            have_json = true;
        } else if (ctype == 0x004E4942) {  // 'BIN\0'
            bin.assign(data, data + clen);
        }
        off += 8 + clen + ((4 - clen % 4) % 4);
    }
    if (!have_json) { err = "GLB has no JSON chunk"; return false; }
    return true;
}

inline bool load_rig(const char* path, Rig& rig, std::string& err) {
    using glb::detail::JVal;
    std::vector<uint8_t> file;
    if (!glb_split(path, file, rig.gltf, rig.bin, err)) return false;
    const JVal* nodes = rig.gltf.find("nodes");
    if (!nodes || !nodes->is_arr()) { err = "glTF has no nodes[]"; return false; }
    rig.N = (int)nodes->arr.size();
    rig.orig.assign(rig.N, "");
    rig.parent.assign(rig.N, -1);
    rig.Lt.assign(rig.N, V3{});
    rig.Lr.assign(rig.N, q_identity());
    rig.Gpos.assign(rig.N, V3{});
    rig.Grot.assign(rig.N, q_identity());

    for (int i = 0; i < rig.N; i++) {
        const JVal& n = nodes->arr[(size_t)i];
        if (const JVal* nm = n.find("name")) if (nm->is_str()) rig.orig[(size_t)i] = nm->str;
        if (const JVal* c = n.find("children"))
            if (c->is_arr())
                for (const auto& cv : c->arr) rig.parent[(size_t)cv.as_int()] = i;
        if (const JVal* mj = n.find("matrix")) {
            // load_rig's matrix branch: glTF stores column-major, so M = reshape(4,4).T; the
            // rotation is the linear part with its COLUMNS normalised (scale divided out).
            if (mj->is_arr() && mj->arr.size() == 16) {
                rig.saw_matrix = true;
                double M[16];
                for (int k = 0; k < 16; k++) M[k] = mj->arr[(size_t)k].as_double();
                rig.Lt[(size_t)i] = V3{M[12], M[13], M[14]};
                double lin[9], cn[3];
                for (int c = 0; c < 3; c++) {
                    cn[c] = std::sqrt(M[c * 4 + 0] * M[c * 4 + 0] + M[c * 4 + 1] * M[c * 4 + 1] +
                                      M[c * 4 + 2] * M[c * 4 + 2]);
                    for (int r = 0; r < 3; r++) lin[r * 3 + c] = M[c * 4 + r] / (cn[c] == 0 ? 1 : cn[c]);
                }
                rig.Lr[(size_t)i] = q_from_matrix(lin);
            }
        } else {
            if (const JVal* t = n.find("translation"))
                if (t->is_arr() && t->arr.size() == 3)
                    rig.Lt[(size_t)i] = V3{t->arr[0].as_double(), t->arr[1].as_double(), t->arr[2].as_double()};
            if (const JVal* r = n.find("rotation"))
                if (r->is_arr() && r->arr.size() == 4)
                    rig.Lr[(size_t)i] = Q4{r->arr[0].as_double(), r->arr[1].as_double(),
                                           r->arr[2].as_double(), r->arr[3].as_double()};
            if (const JVal* s = n.find("scale"))
                if (s->is_arr() && s->arr.size() == 3)
                    for (int k = 0; k < 3; k++)
                        if (s->arr[(size_t)k].as_double() != 1.0) rig.saw_scale = true;
        }
    }
    // rest globals, composed down from every root.  NOTE: scale is deliberately not applied —
    // load_rig does not apply it either, and every rig in this fleet has none.
    std::vector<std::pair<int, std::pair<V3, Q4>>> work;
    for (int i = 0; i < rig.N; i++)
        if (rig.parent[(size_t)i] == -1) work.push_back({i, {V3{}, q_identity()}});
    while (!work.empty()) {
        auto [i, pp] = work.back();
        work.pop_back();
        const Q4 pq = q_norm(pp.second);
        rig.Grot[(size_t)i] = q_mul(pq, q_norm(rig.Lr[(size_t)i]));
        rig.Gpos[(size_t)i] = pp.first + q_apply(pq, rig.Lt[(size_t)i]);
        if (const JVal* c = nodes->arr[(size_t)i].find("children"))
            if (c->is_arr())
                for (const auto& cv : c->arr)
                    work.push_back({(int)cv.as_int(), {rig.Gpos[(size_t)i], rig.Grot[(size_t)i]}});
    }

    // bonemap_names.joint_node_indices: the skin's joint list if there is one, else every node
    // whose name we recognise as a bone.  Sorted ascending either way.
    std::set<int> js;
    const JVal* skins = rig.gltf.find("skins");
    if (skins && skins->is_arr() && !skins->arr.empty()) {
        for (const auto& sk : skins->arr)
            if (const JVal* jt = sk.find("joints"))
                if (jt->is_arr())
                    for (const auto& jv : jt->arr) js.insert((int)jv.as_int());
    } else {
        for (int i = 0; i < rig.N; i++) {
            const std::string& nm = rig.orig[(size_t)i];
            if (nm.rfind("bone_", 0) == 0 || nm.rfind("mixamorig:", 0) == 0 ||
                nm.rfind("skintokens:", 0) == 0)
                js.insert(i);
        }
    }
    rig.joints.assign(js.begin(), js.end());
    rig.is_bone.assign(rig.N, 0);
    for (int j : rig.joints) rig.is_bone[(size_t)j] = 1;
    return true;
}

// ===========================================================================
// 5.  Bone maps — names first, topology walk as the fallback
// ===========================================================================

struct BoneMap {
    std::map<int, int> b2s;   // joint node -> SMPL-22 slot
    std::map<int, int> aim;   // joint node -> the joint node that defines its aim (-1 = none)
    std::string        which; // "names" | "topology"
    int mapped() const { return (int)b2s.size(); }
};

inline void fill_aim(BoneMap& m) {
    std::map<int, int> s2b;
    for (auto& kv : m.b2s) s2b[kv.second] = kv.first;
    const int* child = smpl_child_table();
    m.aim.clear();
    for (auto& kv : m.b2s) {
        const int c = (kv.second >= 0 && kv.second < 22) ? child[kv.second] : -1;
        auto it = (c >= 0) ? s2b.find(c) : s2b.end();
        m.aim[kv.first] = (it == s2b.end()) ? -1 : it->second;
    }
}

// bonemap_names.derive_map_named — the correspondence read straight off the Mixamo names.
inline bool derive_map_named(const Rig& rig, BoneMap& out, std::string& err) {
    out = BoneMap{};
    std::map<std::string, int> want;   // "mixamorig:X" -> slot
    for (int s = 0; s < rig::SMPL_N; s++) want[rig::kMixamoNames[s]] = s;
    std::set<int> used;
    for (int node : rig.joints) {
        const std::string& nm = rig.orig[(size_t)node];
        if (nm.rfind("mixamorig:", 0) != 0) continue;
        auto it = want.find(nm);
        if (it == want.end()) continue;          // a finger / a bone SMPL has no slot for
        if (used.count(it->second)) { err = "two nodes claim " + nm; return false; }
        used.insert(it->second);
        out.b2s[node] = it->second;
    }
    if (!used.count(0) || !used.count(9)) {
        char b[128];
        std::snprintf(b, sizeof(b), "named map lacks pelvis and/or spine3 (%d/22 named)", (int)out.b2s.size());
        err = b;
        return false;
    }
    fill_aim(out);
    out.which = "names";
    return true;
}

// bonemap_v2.derive_map_v2 — the TOPOLOGY walk, with its two documented fixes:
//   FIX-1 spine = the root-child whose subtree CLIMBS HIGHEST (not "whatever is left after the
//         legs": with hanging arms every root child dips below the hips and the spine gets
//         classified as a leg).
//   FIX-2 legs = the OTHER descending root children.
// Kept as the fallback because it is the only thing that works on an UNNAMED rig (the toys are
// correctly unnamed and must stay that way).  It fails outright on miku, whose legs hang off
// Spine rather than Hips, so the root has one child and there are no two leg chains — that is
// exactly why names come first.
inline bool derive_map_topology(const Rig& rig, BoneMap& out, std::string& err) {
    out = BoneMap{};
    const int AX_LAT = 0, AX_UP = 1;
    const std::vector<int>& bones = rig.joints;
    if (bones.empty()) { err = "no bones"; return false; }
    std::map<int, std::vector<int>> ch;
    for (int b : bones) ch[b] = {};
    for (int b : bones)
        if (rig.parent[(size_t)b] >= 0 && rig.is_bone[(size_t)rig.parent[(size_t)b]])
            ch[rig.parent[(size_t)b]].push_back(b);
    int root = -1;
    for (int b : bones)
        if (rig.parent[(size_t)b] < 0 || !rig.is_bone[(size_t)rig.parent[(size_t)b]]) { root = b; break; }
    if (root < 0) { err = "no root bone"; return false; }

    auto subtree = [&](int b) {
        std::vector<int> outv, st{b};
        while (!st.empty()) { int x = st.back(); st.pop_back(); outv.push_back(x);
                              for (int c : ch[x]) st.push_back(c); }
        return outv;
    };
    auto st_up = [&](int b) { double m = -1e300; for (int n : subtree(b)) m = std::max(m, rig.Gpos[(size_t)n][AX_UP]); return m; };
    auto st_dn = [&](int b) { double m =  1e300; for (int n : subtree(b)) m = std::min(m, rig.Gpos[(size_t)n][AX_UP]); return m; };
    auto st_lat= [&](int b) { double m = -1e300; for (int n : subtree(b)) m = std::max(m, std::fabs(rig.Gpos[(size_t)n][AX_LAT])); return m; };

    const std::vector<int>& rc = ch[root];
    if (rc.empty()) { err = "root has no children"; return false; }
    int spine_start = rc[0];
    for (int c : rc) if (st_up(c) > st_up(spine_start)) spine_start = c;   // FIX-1, max() keeps the first on a tie
    std::vector<int> legs;
    for (int c : rc)
        if (c != spine_start && st_dn(c) < rig.Gpos[(size_t)root][AX_UP] - 1e-3) legs.push_back(c);
    std::stable_sort(legs.begin(), legs.end(), [&](int a, int b) { return st_lat(a) > st_lat(b); });
    if (legs.size() < 2) {
        char b[128]; std::snprintf(b, sizeof(b), "expected 2 leg chains under root, found %d", (int)legs.size());
        err = b; return false;
    }
    legs.resize(2);

    std::vector<int> up_chain;
    int cur = spine_start, chest = -1, Larm = -1, Rarm = -1;
    while (cur >= 0) {
        up_chain.push_back(cur);
        const std::vector<int>& kids = ch[cur];
        if (kids.empty()) break;
        int up_kid = kids[0];
        for (int k : kids) if (st_up(k) > st_up(up_kid)) up_kid = k;
        if (chest < 0) {
            std::vector<int> neg, pos;
            for (int k : kids) {
                if (k == up_kid) continue;
                if (rig.Gpos[(size_t)k][AX_LAT] < 0) neg.push_back(k);
                else if (rig.Gpos[(size_t)k][AX_LAT] > 0) pos.push_back(k);
            }
            if (!neg.empty() && !pos.empty()) {
                chest = cur;
                Larm = neg[0]; for (int k : neg) if (st_lat(k) > st_lat(Larm)) Larm = k;
                Rarm = pos[0]; for (int k : pos) if (st_lat(k) > st_lat(Rarm)) Rarm = k;
            }
        }
        cur = up_kid;
    }
    if (Larm < 0) { err = "could not locate a chest with arm children on both sides"; return false; }

    const int ci = (int)(std::find(up_chain.begin(), up_chain.end(), chest) - up_chain.begin());
    std::vector<int> spine_nodes(up_chain.begin(), up_chain.begin() + ci + 1);
    std::vector<int> neckhead(up_chain.begin() + ci + 1,
                             up_chain.begin() + std::min((size_t)ci + 3, up_chain.size()));

    auto chain = [&](int start, int n, bool lateral) {
        std::vector<int> outv{start};
        int c = start;
        for (int k = 1; k < n; k++) {
            const std::vector<int>& kids = ch[c];
            if (kids.empty()) break;
            int best = kids[0];
            for (int q : kids) {
                const double a = lateral ? st_lat(q) : -st_dn(q);
                const double b = lateral ? st_lat(best) : -st_dn(best);
                if (a > b) best = q;
            }
            c = best;
            outv.push_back(c);
        }
        return outv;
    };

    out.b2s[root] = 0;
    { std::vector<int> sp(spine_nodes.end() - std::min<size_t>(3, spine_nodes.size()), spine_nodes.end());
      const int smpl_spine_rev[3] = {9, 6, 3};
      for (size_t k = 0; k < sp.size(); k++) out.b2s[sp[sp.size() - 1 - k]] = smpl_spine_rev[k]; }
    out.b2s[chest] = 9;
    if (!neckhead.empty()) {
        out.b2s[neckhead[0]] = 12;
        if (neckhead.size() > 1) out.b2s[neckhead[1]] = 15;
    }
    { const int La[4] = {13, 16, 18, 20}, Ra[4] = {14, 17, 19, 21};
      std::vector<int> L = chain(Larm, 4, true), R = chain(Rarm, 4, true);
      for (size_t k = 0; k < L.size(); k++) out.b2s[L[k]] = La[k];
      for (size_t k = 0; k < R.size(); k++) out.b2s[R[k]] = Ra[k]; }
    { const bool first_is_left = rig.Gpos[(size_t)legs[0]][AX_LAT] < 0;
      std::vector<int> Ll = chain(first_is_left ? legs[0] : legs[1], 4, false);
      std::vector<int> Rl = chain(first_is_left ? legs[1] : legs[0], 4, false);
      const int Ls[4] = {1, 4, 7, 10}, Rs[4] = {2, 5, 8, 11};
      for (size_t k = 0; k < Ll.size(); k++) out.b2s[Ll[k]] = Ls[k];
      for (size_t k = 0; k < Rl.size(); k++) out.b2s[Rl[k]] = Rs[k]; }
    fill_aim(out);
    out.which = "topology";
    return true;
}

// ===========================================================================
// 6.  The L/R decision, measured (bonemap_names.decide_lr_swap)
// ===========================================================================
//
// forward := sum over feet of (toe - ankle), flattened horizontal;  right := cross(forward, up).
// swap needed  <=>  side_rig(SMPL_L's bone) != side_src(SMPL joint 1), each in its OWN body frame,
// so the sign convention of `right` cancels and the answer is immune to the cross(fwd,up) vs
// cross(up,fwd) argument.
//
// THE TRAP THIS AVOIDS: an earlier chirality metric assigned each joint a side from its mean
// POSED position.  On a kick the swinging ankle crosses the midline, so the motion was credited
// to the leg that did not move — it mislabelled exactly the asymmetric clips the test exists for.
// Sides are read from REST geometry, once.

struct SwapInfo {
    bool        swap = true;
    bool        decided = false;
    std::string reason, rig_cue, src_cue;
    V3          rig_forward{}, src_forward{};
    double      rig_L_side = 0, src_L_side = 0;
    std::vector<int> joints_used;
};

// `pos(slot) -> position, or false if this skeleton has no such joint`
template <typename PosFn>
inline bool body_side_frame(PosFn pos, V3& fwd, V3& right, std::string& cue) {
    const V3 UP{0, 1, 0};
    V3 acc{};
    int n = 0;
    const int pairs[2][2] = {{7, 10}, {8, 11}};
    for (auto& pr : pairs) {
        V3 a, t;
        if (!pos(pr[0], a) || !pos(pr[1], t)) continue;
        acc = acc + (t - a);
        n++;
    }
    bool have = false;
    if (n > 0) {
        V3 f = acc - UP * dot(acc, UP);
        if (norm(f) >= 1e-9) { fwd = f * (1.0 / norm(f)); have = true; }
        char b[32]; std::snprintf(b, sizeof(b), "feet(%d)", n); cue = b;
    }
    if (!have) {
        V3 h, nk;
        if (!pos(15, h) || !pos(12, nk)) { cue = "none"; return false; }
        V3 d = h - nk;
        d = d - UP * dot(d, UP);
        if (norm(d) < 1e-9) { cue = "none"; return false; }
        fwd = d * (1.0 / norm(d));
        cue = "head-vs-neck (WEAK)";
    }
    right = cross(fwd, UP);
    right = right * (1.0 / std::max(norm(right), 1e-12));
    return true;
}

inline SwapInfo decide_lr_swap(const Rig& rig, const BoneMap& map, const std::vector<V3>& src_rest) {
    SwapInfo out;
    std::map<int, int> s2b;
    for (auto& kv : map.b2s) if (kv.second >= 0) s2b[kv.second] = kv.first;
    auto rig_pos = [&](int s, V3& p) {
        auto it = s2b.find(s);
        if (it == s2b.end()) return false;
        p = rig.Gpos[(size_t)it->second];
        return true;
    };
    auto src_pos = [&](int s, V3& p) { p = src_rest[(size_t)s]; return true; };

    V3 fr, rr, fs, rs;
    const bool okr = body_side_frame(rig_pos, fr, rr, out.rig_cue);
    const bool oks = body_side_frame(src_pos, fs, rs, out.src_cue);
    if (!okr || !oks) {
        out.swap = true;
        out.reason = "no facing cue on the rig — falling back to swap=ON (the measured setting "
                     "for every rig on this tree)";
        return out;
    }
    for (int s : {1, 16, 20}) if (s2b.count(s)) out.joints_used.push_back(s);
    auto side_of = [&](auto posfn, const V3& right, const std::vector<int>& js) {
        double acc = 0; int n = 0;
        for (int j : js) { V3 p, p0; if (!posfn(0, p0) || !posfn(j, p)) continue; acc += dot(p - p0, right); n++; }
        return n ? acc / n : 0.0;
    };
    if (out.joints_used.empty()) {
        out.swap = true;
        out.reason = "rig L-side joints sit on the midline";
        return out;
    }
    out.rig_L_side = side_of(rig_pos, rr, out.joints_used);
    out.src_L_side = side_of(src_pos, rs, out.joints_used);
    if (std::fabs(out.rig_L_side) < 1e-4) {
        out.swap = true;
        out.reason = "rig L-side joints sit on the midline";
        return out;
    }
    out.decided = true;
    out.rig_forward = fr;
    out.src_forward = fs;
    out.swap = (out.rig_L_side > 0) != (out.src_L_side > 0);
    return out;
}

// ===========================================================================
// 7.  The retarget itself
// ===========================================================================

struct Clip {
    std::vector<int>    bones;       // joint node indices, ascending (== `bone_<node>` names)
    double              fps = 30;
    int                 T = 0, N = 0;
    std::vector<Q4>     quats;       // T*N, LOCAL (parent-relative) — glTF node rotations already
    std::vector<V3>     root_pos;    // T
};

struct RestAlign {
    double a[3] = {0, 0, 0};   // indexed by GRP_ARM / GRP_LEG / GRP_SPINE
    bool any() const { return a[0] != 0 || a[1] != 0 || a[2] != 0; }
};

// motion : (T,22) source joint POSITIONS, already in the same L/R labelling as src_rest.
// src_rot: (T,22) source GLOBAL rotations, or nullptr for the positional path.
// Exactly one of the two drives D; the positional path is untouched when src_rot is given.
inline bool retarget_delta(const Rig& rig, const BoneMap& map, const std::vector<V3>& motion,
                           const std::vector<V3>& src_rest, double fps, bool root_translate,
                           const RestAlign& ra, const std::vector<Q4>* src_rot, Clip& out,
                           std::string& err) {
    const std::vector<int>& bnums = rig.joints;      // sorted ascending, == python's bnums
    const int Nb = (int)bnums.size();
    std::map<int, int> bidx;
    for (int k = 0; k < Nb; k++) bidx[bnums[k]] = k;

    std::map<int, int> s2b;
    for (auto& kv : map.b2s) if (kv.second >= 0) s2b[kv.second] = kv.first;
    if (!s2b.count(0) || !s2b.count(9)) { err = "map lacks pelvis (0) and/or spine3 (9)"; return false; }
    const int root_b = s2b[0];

    // topological order (parents first), built by the same recursive `add` the prototype uses
    std::vector<int> topo;
    std::set<int>    seen;
    std::function<void(int)> add = [&](int b) {
        if (seen.count(b)) return;
        const int pn = rig.parent[(size_t)b];
        if (pn >= 0 && rig.is_bone[(size_t)pn]) add(pn);
        seen.insert(b);
        topo.push_back(b);
    };
    for (int b : bnums) add(b);

    // ---- source rest reference frames (constants; computed ONCE) ----
    double srest_frame[9];
    body_frame(src_rest[0], src_rest[9], src_rest[1], src_rest[2], srest_frame);

    // per-bone source REST direction u_s (world space, unit)
    std::map<int, V3> U;
    for (int b : bnums) {
        auto sb = map.b2s.find(b);
        auto ab = map.aim.find(b);
        if (sb == map.b2s.end() || ab == map.aim.end() || ab->second < 0) continue;
        auto ac_s = map.b2s.find(ab->second);
        if (ac_s == map.b2s.end() || ac_s->second < 0 || sb->second < 0) continue;
        V3 u = src_rest[(size_t)ac_s->second] - src_rest[(size_t)sb->second];
        const double n = norm(u);
        if (n < 1e-9) continue;
        U[b] = u * (1.0 / n);
    }

    std::map<int, Q4> Grest, Lrest;
    for (int b : bnums) {
        Grest[b] = q_norm(rig.Grot[(size_t)b]);
        Lrest[b] = q_norm(rig.Lr[(size_t)b]);
    }

    // ---- RETARGET BASE POSE (eq 1'): A_b = slerp(identity, K_b, alpha_group) ----
    std::map<int, Q4> A;
    if (ra.any()) {
        for (int b : bnums) {
            auto sb = map.b2s.find(b);
            if (!U.count(b) || sb == map.b2s.end() || sb->second < 0) continue;   // leaf/unmapped: keep our rest
            const double alpha = ra.a[smpl_group(sb->second)];
            if (alpha == 0.0) continue;
            const int ac = map.aim.at(b);
            V3 ours = rig.Gpos[(size_t)ac] - rig.Gpos[(size_t)b];
            const double n = norm(ours);
            if (n < 1e-9) continue;
            const Q4 K = quat_between(ours * (1.0 / n), U[b]);
            A[b] = q_from_rotvec(q_as_rotvec(K) * alpha);
        }
    }

    const double our_torso = norm(rig.Gpos[(size_t)s2b[9]] - rig.Gpos[(size_t)s2b[0]]);
    const int T = (int)(motion.size() / 22);

    out.bones = bnums;
    out.fps = fps;
    out.T = T;
    out.N = Nb;
    out.quats.assign((size_t)T * Nb, q_identity());
    out.root_pos.assign((size_t)T, V3{});
    const V3 m0 = motion[0];

    std::map<int, Q4> Gpose;
    for (int t = 0; t < T; t++) {
        const V3* m = &motion[(size_t)t * 22];
        Q4 D_root = q_identity();
        if (!src_rot) {
            // root: the body-frame delta of the SOURCE against the SOURCE'S OWN rest.  (Stock
            // measures the source's frame against OUR RIG's rest frame, which mixes the two
            // skeletons and is what leaves SMPL's frame baked in.)
            double tf[9], prod[9];
            body_frame(m[0], m[9], m[1], m[2], tf);
            mat3_mul_bT(tf, srest_frame, prod);
            D_root = q_from_matrix(prod);
        }
        Gpose.clear();
        for (int b : topo) {
            const int pn = rig.parent[(size_t)b];
            const bool parent_is_bone = pn >= 0 && rig.is_bone[(size_t)pn];
            Q4 Gp;
            if (parent_is_bone && Gpose.count(pn)) Gp = Gpose[pn];
            else Gp = (pn == -1) ? q_identity() : q_norm(rig.Grot[(size_t)pn]);  // parent node never animated

            bool have_D = false;
            Q4   D = q_identity();
            if (src_rot) {
                // rotation-native: D_s IS the source's global rotation.  Root included — the
                // pelvis's own global rotation is exact, so no body frame is estimated from
                // three joint positions.
                auto sb = map.b2s.find(b);
                if (sb != map.b2s.end() && sb->second >= 0) {
                    D = q_norm((*src_rot)[(size_t)t * 22 + (size_t)sb->second]);
                    have_D = true;
                }
            } else if (b == root_b) {
                D = D_root;
                have_D = true;
            } else if (U.count(b)) {
                const int ac = map.aim.at(b);
                V3 v = m[map.b2s.at(ac)] - m[map.b2s.at(b)];
                const double n = norm(v);
                if (n > 1e-9) { D = quat_between(U[b], v * (1.0 / n)); have_D = true; }
            }

            // eq (1'): our own rest rotation, first carried onto the retarget base pose by A_b
            // (identity unless rest_align), then moved by the source's rest-relative swing.
            // No D (leaf/unmapped) -> identity delta -> keep our rest local, inherit the parent's.
            Q4 G;
            if (!have_D)            G = q_mul(Gp, Lrest[b]);
            else if (A.count(b))    G = q_mul(q_mul(D, A[b]), Grest[b]);
            else                    G = q_mul(D, Grest[b]);
            Gpose[b] = G;
            out.quats[(size_t)t * Nb + (size_t)bidx[b]] = q_mul(q_inv(Gp), G);   // eq (2)
        }
        if (root_translate) {
            const double sc = our_torso / std::max(norm(m[9] - m[0]), 1e-6);
            out.root_pos[(size_t)t] = rig.Gpos[(size_t)root_b] + (m[0] - m0) * sc;
        } else {
            out.root_pos[(size_t)t] = rig.Gpos[(size_t)root_b];
        }
    }
    return true;
}

// retarget.py's smooth_quats: temporal nlerp over a +/-win window with 0.5^|k| weights,
// hemisphere-aligned to the CENTRE frame (not to the neighbour) before averaging.
inline void smooth_quats(Clip& c, int win) {
    if (win <= 0 || c.T < 3) return;
    const int T = c.T, N = c.N;
    std::vector<double> w((size_t)(2 * win + 1));
    double wsum = 0;
    for (int k = -win; k <= win; k++) { w[(size_t)(k + win)] = std::pow(0.5, std::abs(k)); wsum += w[(size_t)(k + win)]; }
    for (auto& v : w) v /= wsum;
    std::vector<Q4> out(c.quats);
    for (int n = 0; n < N; n++) {
        for (int t = 0; t < T; t++) {
            const Q4& ref = c.quats[(size_t)t * N + n];
            double acc[4] = {0, 0, 0, 0};
            for (int k = -win; k <= win; k++) {
                int idx = t + k;
                idx = std::max(0, std::min(T - 1, idx));
                const Q4& q = c.quats[(size_t)idx * N + n];
                double d = q.x * ref.x + q.y * ref.y + q.z * ref.z + q.w * ref.w;
                const double sign = (d > 0) ? 1.0 : (d < 0 ? -1.0 : 1.0);   // np.sign, 0 -> 1
                const double wk = w[(size_t)(k + win)];
                acc[0] += wk * q.x * sign; acc[1] += wk * q.y * sign;
                acc[2] += wk * q.z * sign; acc[3] += wk * q.w * sign;
            }
            double nr = std::sqrt(acc[0] * acc[0] + acc[1] * acc[1] + acc[2] * acc[2] + acc[3] * acc[3]);
            if (nr < 1e-9) nr = 1;
            out[(size_t)t * N + n] = Q4{acc[0] / nr, acc[1] / nr, acc[2] / nr, acc[3] / nr};
        }
    }
    c.quats.swap(out);
}

// ===========================================================================
// 8.  Sources
// ===========================================================================

struct HymoClip {
    std::string      name;
    double           fps = 30;
    int              T = 0;
    std::vector<int> parents;      // 22
    std::vector<Q4>  local;        // T*22, LOCAL xyzw
    std::vector<V3>  root_pos;     // T
};

// The parse, from memory. A generated clip never has to touch the disk to be retargeted: the
// in-process chain hands the JSON TEXT across, not a path. (Text, not a struct, deliberately —
// HYMotion::to_clip_json rounds to 6 dp, so parsing the same text the artifact holds is what makes
// the saved clip and the baked animation reproduce each other.)
inline bool load_hymo_string(const char* data, size_t len, HymoClip& out, std::string& err) {
    glb::detail::JVal d;
    glb::detail::JParser p(data, len);
    if (!p.parse(d)) { err = "JSON parse failed"; return false; }
    const auto* skel = d.find("skeleton");
    const auto* bones = d.find("bones");
    if (!skel || !skel->is_str() || skel->str != "smplh22" || !bones || !bones->is_arr() ||
        bones->arr.size() != 22) {
        // load_hymo() refuses anything that is not smplh22 on purpose: the older 38-bone
        // `bone_N` clips at 20 fps are already-RETARGETED OUTPUT, not a source.
        err = "not a raw smplh22 clip";
        return false;
    }
    if (const auto* n = d.find("name")) if (n->is_str()) out.name = n->str;
    out.fps = d.find("fps") ? d.find("fps")->as_double(30) : 30;
    const auto* par = d.find("parents");
    if (!par || !par->is_arr() || par->arr.size() != 22) { err = "clip has no parents array"; return false; }
    out.parents.clear();
    for (const auto& v : par->arr) out.parents.push_back((int)v.as_int());
    const auto* q = d.find("quats");
    if (!q || !q->is_arr()) { err = "clip has no quats"; return false; }
    out.T = (int)q->arr.size();
    out.local.resize((size_t)out.T * 22);
    for (int t = 0; t < out.T; t++)
        for (int j = 0; j < 22; j++) {
            const auto& e = q->arr[(size_t)t].arr[(size_t)j];
            out.local[(size_t)t * 22 + (size_t)j] =
                Q4{e.arr[0].as_double(), e.arr[1].as_double(), e.arr[2].as_double(), e.arr[3].as_double()};
        }
    out.root_pos.assign((size_t)out.T, V3{});
    if (const auto* rp = d.find("root_pos"))
        if (rp->is_arr() && (int)rp->arr.size() == out.T)
            for (int t = 0; t < out.T; t++)
                out.root_pos[(size_t)t] = V3{rp->arr[(size_t)t].arr[0].as_double(),
                                             rp->arr[(size_t)t].arr[1].as_double(),
                                             rp->arr[(size_t)t].arr[2].as_double()};
    return true;
}

inline bool load_hymo_json(const char* path, HymoClip& out, std::string& err) {
    std::vector<uint8_t> buf;
    if (!read_file(path, buf)) { err = std::string("cannot open ") + path; return false; }
    return load_hymo_string((const char*)buf.data(), buf.size(), out, err);
}

// build_motion_ab.smpl_global_rots: LOCAL -> GLOBAL down the SMPL kinematic tree.  SMPL's rest is
// its zero pose, so these globals ARE the rest-relative world swings retarget_delta wants.
inline std::vector<Q4> smpl_global_rots(const HymoClip& c) {
    std::vector<Q4> g((size_t)c.T * 22);
    for (int t = 0; t < c.T; t++) {
        const Q4* q = &c.local[(size_t)t * 22];
        Q4* gg = &g[(size_t)t * 22];
        gg[0] = q_norm(q[0]);
        for (int j = 1; j < 22; j++) gg[j] = q_mul(q_norm(gg[c.parents[(size_t)j]]), q_norm(q[j]));
    }
    return g;
}

// hymo_to_npy.py's FK: the LEGACY POSITION BRIDGE.  Still needed for the positional variants of a
// rotation-carrying clip (fix/nofix/base), and for MoMask sources, which have nothing else.
// NOTE the float32 round at the end — the bridge writes a float32 .npy and the retarget reads it
// back, so the positional path genuinely sees float32 positions.  Reproduced, not "fixed".
inline std::vector<V3> hymo_positions(const HymoClip& c, const std::vector<V3>& smpl_rest) {
    std::vector<V3> off(22);
    off[0] = smpl_rest[0];
    for (int j = 1; j < 22; j++) off[(size_t)j] = smpl_rest[(size_t)j] - smpl_rest[(size_t)c.parents[(size_t)j]];
    std::vector<V3> P((size_t)c.T * 22);
    std::vector<double> gR((size_t)22 * 9);
    for (int t = 0; t < c.T; t++) {
        double m[9];
        q_matrix(q_norm(c.local[(size_t)t * 22]), m);
        std::memcpy(&gR[0], m, sizeof(m));
        P[(size_t)t * 22] = c.root_pos[(size_t)t];
        for (int j = 1; j < 22; j++) {
            const int p = c.parents[(size_t)j];
            double lm[9];
            q_matrix(q_norm(c.local[(size_t)t * 22 + (size_t)j]), lm);
            double* gp = &gR[(size_t)p * 9];
            double* gj = &gR[(size_t)j * 9];
            for (int r = 0; r < 3; r++)
                for (int col = 0; col < 3; col++)
                    gj[r * 3 + col] = gp[r * 3 + 0] * lm[0 * 3 + col] + gp[r * 3 + 1] * lm[1 * 3 + col] +
                                      gp[r * 3 + 2] * lm[2 * 3 + col];
            const V3& o = off[(size_t)j];
            P[(size_t)t * 22 + (size_t)j] =
                P[(size_t)t * 22 + (size_t)p] +
                V3{gp[0] * o.x + gp[1] * o.y + gp[2] * o.z,
                   gp[3] * o.x + gp[4] * o.y + gp[5] * o.z,
                   gp[6] * o.x + gp[7] * o.y + gp[8] * o.z};
        }
    }
    for (auto& v : P) { v.x = (double)(float)v.x; v.y = (double)(float)v.y; v.z = (double)(float)v.z; }
    return P;
}

// ===========================================================================
// 9.  prepare() — load + map + swap decision, i.e. everything about a rig that does not depend on
//     which motion is being played.  Lives here rather than in the A/B driver because the inline
//     pipeline needs exactly the same answer: one implementation, or the shipped path and the
//     harness that validates it can disagree.
// ===========================================================================

struct Prep {
    Rig       rig;
    BoneMap   map, named, topo;
    bool      named_ok = false, topo_ok = false;
    std::string named_err, topo_err;
    SwapInfo  swap;
    // angle(our rest bone dir, SMPL rest bone dir) per chain group — the quantity delta silently
    // adds to every frame, and the number the base-pose knob exists for.
    double    rest_arm_mean = 0, rest_arm_max = 0, rest_leg_mean = 0, rest_spine_mean = 0;
};

inline bool prepare(const std::string& glb, const std::vector<V3>& src_rest, Prep& p, std::string& err) {
    if (!load_rig(glb.c_str(), p.rig, err)) return false;
    p.named_ok = derive_map_named(p.rig, p.named, p.named_err);
    p.topo_ok = derive_map_topology(p.rig, p.topo, p.topo_err);
    if (p.named_ok) p.map = p.named;
    else if (p.topo_ok) p.map = p.topo;
    else { err = "no usable map (names: " + p.named_err + "; topology: " + p.topo_err + ")"; return false; }
    p.swap = decide_lr_swap(p.rig, p.map, src_rest);

    std::vector<V3> src = src_rest;
    if (p.swap.swap) lr_swap_inplace(src, 1);
    std::map<int, int> s2b;
    for (auto& kv : p.map.b2s) if (kv.second >= 0) s2b[kv.second] = kv.first;
    double acc[3] = {0, 0, 0}, mx[3] = {0, 0, 0};
    int cnt[3] = {0, 0, 0};
    const int* child = smpl_child_table();
    for (int s = 0; s < 22; s++) {
        const int c = child[s];
        if (c < 0 || !s2b.count(s) || !s2b.count(c)) continue;
        V3 ours = p.rig.Gpos[(size_t)s2b[c]] - p.rig.Gpos[(size_t)s2b[s]];
        V3 smpl = src[(size_t)c] - src[(size_t)s];
        const double t = std::acos(std::max(-1.0, std::min(1.0, dot(ours * (1.0 / norm(ours)),
                                                                    smpl * (1.0 / norm(smpl)))))) * 180.0 / M_PI;
        const int gidx = smpl_group(s);
        acc[gidx] += t; cnt[gidx]++;
        mx[gidx] = std::max(mx[gidx], t);
    }
    p.rest_arm_mean = cnt[GRP_ARM] ? acc[GRP_ARM] / cnt[GRP_ARM] : 0;
    p.rest_arm_max = mx[GRP_ARM];
    p.rest_leg_mean = cnt[GRP_LEG] ? acc[GRP_LEG] / cnt[GRP_LEG] : 0;
    p.rest_spine_mean = cnt[GRP_SPINE] ? acc[GRP_SPINE] / cnt[GRP_SPINE] : 0;
    return true;
}

// The five A/B variants, by id.  `rot`/`rotbase` need a rotation-carrying source (HY-Motion); a
// positional clip has nothing to offer them.  rotbase is the shipped default.
struct Variant { const char* id; double arm_alpha; bool rot; bool hymo_only; };
inline const Variant* variants(int& n) {
    static const Variant V[] = {
        {"fix", 0.0, false, false},      // positional · facing decided by measurement
        {"nofix", 0.0, false, false},    // the other L/R labelling — the A/B that shows the mirror
        {"base", 1.0, false, false},     // + retarget base pose on the arms
        {"rot", 0.0, true, true},        // ROTATION-NATIVE, rest_align held at 0
        {"rotbase", 1.0, true, true},    // rotation-native + base pose — THE DEFAULT
    };
    n = 5;
    return V;
}
inline const Variant* find_variant(const std::string& id) {
    int n = 0;
    const Variant* V = variants(n);
    for (int i = 0; i < n; i++) if (id == V[i].id) return &V[i];
    return nullptr;
}

}  // namespace mret
