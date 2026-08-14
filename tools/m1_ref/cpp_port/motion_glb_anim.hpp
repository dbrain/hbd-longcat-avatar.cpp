// motion_glb_anim.hpp — bake retargeted clips into a skinned GLB as glTF animations, and write
// the meshless alias skeleton the Python verification tools expect.
//
// Port of build_motion_ab.py's bake() / write_alias_skeleton().  The clip's local quats ARE glTF
// node rotations (retarget_delta eq (2) already produces parent-relative locals), so baking is a
// straight copy into rotation channels — no conversion, no player code.
//
// TWO DECISIONS WORTH RESTATING, because they are easy to "fix" back into bugs:
//  * QUATERNIONS ARE HEMISPHERE-ALIGNED ALONG TIME before they are written.  A LINEAR sampler
//    interpolates the raw components, so an unflipped sign change makes the limb take the long
//    way round and spin.
//  * FPS IS CARRIED PER CLIP AND NEVER RESAMPLED.  glTF keys animations in SECONDS, so a 30 fps
//    clip and a 20 fps clip coexist in one GLB with no interpolation and no error.  Resampling
//    quaternions to a common rate would only add error and buy nothing.
//
// Rewriting the glTF JSON means re-serialising it.  Object member ORDER is not preserved (the
// parser stores members in a std::map) — that is legal JSON and legal glTF, and the acceptance
// test is on the animation SAMPLES, not on the byte layout of the container.  Numbers are printed
// with the shortest decimal form that round-trips, so no node transform is perturbed.
#pragma once

#include "motion_retarget.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace mret {

// ---------------------------------------------------------------------------
// JSON writing
// ---------------------------------------------------------------------------

// Shortest decimal string that parses back to the identical double (Python's repr, in effect).
// Integral values print without a fraction so `"componentType": 5126` does not become 5126.0.
inline std::string jnum(double v) {
    if (v == (double)(long long)v && std::fabs(v) < 9.0e15) {
        char b[32];
        std::snprintf(b, sizeof(b), "%lld", (long long)v);
        return b;
    }
    char b[64];
    for (int prec = 1; prec <= 17; prec++) {
        std::snprintf(b, sizeof(b), "%.*g", prec, v);
        if (std::strtod(b, nullptr) == v) return b;
    }
    std::snprintf(b, sizeof(b), "%.17g", v);
    return b;
}

inline void jesc(const std::string& s, std::string& out) {
    out.push_back('"');
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) { char b[8]; std::snprintf(b, sizeof(b), "\\u%04x", c); out += b; }
                else out.push_back((char)c);
        }
    }
    out.push_back('"');
}

inline void jdump(const glb::detail::JVal& v, std::string& out) {
    using J = glb::detail::JVal;
    switch (v.type) {
        case J::Null: out += "null"; break;
        case J::Bool: out += v.b ? "true" : "false"; break;
        case J::Num:  out += jnum(v.num); break;
        case J::Str:  jesc(v.str, out); break;
        case J::Arr: {
            out.push_back('[');
            for (size_t i = 0; i < v.arr.size(); i++) { if (i) out.push_back(','); jdump(v.arr[i], out); }
            out.push_back(']');
            break;
        }
        case J::Obj: {
            out.push_back('{');
            bool first = true;
            for (const auto& kv : v.obj) {
                if (!first) out.push_back(',');
                first = false;
                jesc(kv.first, out);
                out.push_back(':');
                jdump(kv.second, out);
            }
            out.push_back('}');
            break;
        }
    }
}

// small constructors
inline glb::detail::JVal jnumv(double d) { glb::detail::JVal v; v.type = glb::detail::JVal::Num; v.num = d; return v; }
inline glb::detail::JVal jstrv(const std::string& s) { glb::detail::JVal v; v.type = glb::detail::JVal::Str; v.str = s; return v; }
inline glb::detail::JVal jobjv() { glb::detail::JVal v; v.type = glb::detail::JVal::Obj; return v; }
inline glb::detail::JVal jarrv() { glb::detail::JVal v; v.type = glb::detail::JVal::Arr; return v; }

inline bool write_glb(const char* path, const glb::detail::JVal& json, const std::vector<uint8_t>& bin,
                      std::string& err) {
    std::string js;
    jdump(json, js);
    while (js.size() % 4) js.push_back(' ');
    std::vector<uint8_t> pad_bin(bin);
    while (pad_bin.size() % 4) pad_bin.push_back(0);
    const uint32_t total = 12 + 8 + (uint32_t)js.size() + (pad_bin.empty() ? 0 : 8 + (uint32_t)pad_bin.size());
    FILE* f = std::fopen(path, "wb");
    if (!f) { err = std::string("cannot write ") + path; return false; }
    auto w32 = [&](uint32_t v) { std::fwrite(&v, 4, 1, f); };
    std::fwrite("glTF", 1, 4, f);
    w32(2);
    w32(total);
    w32((uint32_t)js.size());
    w32(0x4E4F534A);
    std::fwrite(js.data(), 1, js.size(), f);
    if (!pad_bin.empty()) {
        w32((uint32_t)pad_bin.size());
        w32(0x004E4942);
        std::fwrite(pad_bin.data(), 1, pad_bin.size(), f);
    }
    std::fclose(f);
    return true;
}

// ---------------------------------------------------------------------------
// bake
// ---------------------------------------------------------------------------

struct BakedClip {
    std::string      name;      // "<motion>__<variant>"
    std::vector<int> bones;     // joint node indices
    double           fps = 30;
    int              T = 0, N = 0;
    std::vector<Q4>  quats;     // T*N LOCAL
};

// Returns the number of animations written.
inline int bake_animations(const Rig& src, const std::vector<BakedClip>& clips, const char* out_glb,
                           std::string& err) {
    using J = glb::detail::JVal;
    J g = src.gltf;                       // deep copy — the source rig is never modified
    std::vector<uint8_t> blob = src.bin;

    J* bufferViews = const_cast<J*>(g.find("bufferViews"));
    if (!bufferViews) { g.obj["bufferViews"] = jarrv(); bufferViews = &g.obj["bufferViews"]; }
    J* accessors = const_cast<J*>(g.find("accessors"));
    if (!accessors) { g.obj["accessors"] = jarrv(); accessors = &g.obj["accessors"]; }
    if (!g.find("animations")) g.obj["animations"] = jarrv();
    J& animations = g.obj["animations"];

    auto add_view = [&](const void* data, size_t nbytes) {
        while (blob.size() % 4) blob.push_back(0);
        const size_t off = blob.size();
        const uint8_t* p = (const uint8_t*)data;
        blob.insert(blob.end(), p, p + nbytes);
        J bv = jobjv();
        bv.obj["buffer"] = jnumv(0);
        bv.obj["byteOffset"] = jnumv((double)off);
        bv.obj["byteLength"] = jnumv((double)nbytes);
        bufferViews->arr.push_back(bv);
        return (int)bufferViews->arr.size() - 1;
    };

    const J* nodes = g.find("nodes");
    const int NN = (nodes && nodes->is_arr()) ? (int)nodes->arr.size() : 0;

    for (const auto& c : clips) {
        std::vector<float> times((size_t)c.T);
        for (int t = 0; t < c.T; t++) times[(size_t)t] = (float)((double)t / c.fps);
        const int ti = add_view(times.data(), times.size() * 4);
        J ta = jobjv();
        ta.obj["bufferView"] = jnumv(ti);
        ta.obj["componentType"] = jnumv(5126);
        ta.obj["count"] = jnumv(c.T);
        ta.obj["type"] = jstrv("SCALAR");
        { J mn = jarrv(); mn.arr.push_back(jnumv(times.front())); ta.obj["min"] = mn;
          J mx = jarrv(); mx.arr.push_back(jnumv(times.back()));  ta.obj["max"] = mx; }
        accessors->arr.push_back(ta);
        const int t_acc = (int)accessors->arr.size() - 1;

        J samplers = jarrv(), channels = jarrv();
        for (int bi = 0; bi < c.N; bi++) {
            const int node = c.bones[(size_t)bi];
            if (node >= NN) continue;
            std::vector<Q4> q((size_t)c.T);
            for (int t = 0; t < c.T; t++) q[(size_t)t] = q_norm(c.quats[(size_t)t * c.N + (size_t)bi]);
            for (int t = 1; t < c.T; t++) {   // hemisphere-align for LINEAR/slerp
                const Q4& p = q[(size_t)t - 1];
                Q4& n = q[(size_t)t];
                if (n.x * p.x + n.y * p.y + n.z * p.z + n.w * p.w < 0.0)
                    n = Q4{-n.x, -n.y, -n.z, -n.w};
            }
            double dev = 0;
            for (int t = 0; t < c.T; t++) {
                dev = std::max(dev, std::fabs(q[(size_t)t].x - q[0].x));
                dev = std::max(dev, std::fabs(q[(size_t)t].y - q[0].y));
                dev = std::max(dev, std::fabs(q[(size_t)t].z - q[0].z));
                dev = std::max(dev, std::fabs(q[(size_t)t].w - q[0].w));
            }
            if (dev < 1e-6) continue;         // constant bone: no channel needed
            std::vector<float> f((size_t)c.T * 4);
            for (int t = 0; t < c.T; t++) {
                f[(size_t)t * 4 + 0] = (float)q[(size_t)t].x;
                f[(size_t)t * 4 + 1] = (float)q[(size_t)t].y;
                f[(size_t)t * 4 + 2] = (float)q[(size_t)t].z;
                f[(size_t)t * 4 + 3] = (float)q[(size_t)t].w;
            }
            const int qv = add_view(f.data(), f.size() * 4);
            J qa = jobjv();
            qa.obj["bufferView"] = jnumv(qv);
            qa.obj["componentType"] = jnumv(5126);
            qa.obj["count"] = jnumv(c.T);
            qa.obj["type"] = jstrv("VEC4");
            accessors->arr.push_back(qa);
            J s = jobjv();
            s.obj["input"] = jnumv(t_acc);
            s.obj["output"] = jnumv((double)accessors->arr.size() - 1);
            s.obj["interpolation"] = jstrv("LINEAR");
            samplers.arr.push_back(s);
            J tgt = jobjv();
            tgt.obj["node"] = jnumv(node);
            tgt.obj["path"] = jstrv("rotation");
            J ch = jobjv();
            ch.obj["sampler"] = jnumv((double)samplers.arr.size() - 1);
            ch.obj["target"] = tgt;
            channels.arr.push_back(ch);
        }
        if (channels.arr.empty()) continue;
        J an = jobjv();
        an.obj["name"] = jstrv(c.name);
        an.obj["samplers"] = samplers;
        an.obj["channels"] = channels;
        animations.arr.push_back(an);
    }

    J* bufs = const_cast<J*>(g.find("buffers"));
    if (bufs && bufs->is_arr() && !bufs->arr.empty())
        bufs->arr[0].obj["byteLength"] = jnumv((double)blob.size());
    if (!write_glb(out_glb, g, blob, err)) return -1;
    return (int)animations.arr.size();
}

// A meshless copy of the joint hierarchy with names aliased to bone_<node index>, so the Python
// check_* tools (check_arms / check_tilt / diagnose_facing) run unmodified against a
// Mixamo-named rig.  Kept purely as a verification convenience.
inline bool write_alias_skeleton(const Rig& rig, const char* out_glb, std::string& err) {
    using J = glb::detail::JVal;
    const J* snodes = rig.gltf.find("nodes");
    if (!snodes) { err = "no nodes"; return false; }
    std::map<int, int> remap;
    for (size_t k = 0; k < rig.joints.size(); k++) remap[rig.joints[k]] = (int)k;

    J g = jobjv();
    J asset = jobjv();
    asset.obj["version"] = jstrv("2.0");
    asset.obj["generator"] = jstrv("motion_retarget (cpp_port)");
    g.obj["asset"] = asset;
    J nodes = jarrv();
    for (int n : rig.joints) {
        const J& sn = snodes->arr[(size_t)n];
        J o = jobjv();
        char nm[32];
        std::snprintf(nm, sizeof(nm), "bone_%d", n);
        o.obj["name"] = jstrv(nm);
        for (const char* key : {"translation", "rotation", "scale"})
            if (const J* a = sn.find(key)) o.obj[key] = *a;
        if (const J* c = sn.find("children")) {
            J kids = jarrv();
            for (const auto& cv : c->arr)
                if (remap.count((int)cv.as_int())) kids.arr.push_back(jnumv(remap[(int)cv.as_int()]));
            if (!kids.arr.empty()) o.obj["children"] = kids;
        }
        nodes.arr.push_back(o);
    }
    g.obj["nodes"] = nodes;
    J roots = jarrv();
    for (int n : rig.joints) {
        const int p = rig.parent[(size_t)n];
        if (p < 0 || !remap.count(p)) roots.arr.push_back(jnumv(remap[n]));
    }
    J scene = jobjv();
    scene.obj["nodes"] = roots;
    J scenes = jarrv();
    scenes.arr.push_back(scene);
    g.obj["scenes"] = scenes;
    g.obj["scene"] = jnumv(0);
    return write_glb(out_glb, g, {}, err);
}

}  // namespace mret
