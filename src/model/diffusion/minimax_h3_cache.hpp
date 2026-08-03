#ifndef __SD_MODEL_DIFFUSION_MINIMAX_H3_CACHE_HPP__
#define __SD_MODEL_DIFFUSION_MINIMAX_H3_CACHE_HPP__

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#include "conditioning/conditioner.hpp"
#include "core/tensor.hpp"
#include "core/util.h"
#include "model/diffusion/minimax_h3_host_layout.hpp"

// ─────────────────────────────────────────────────────────────────────────────
//  MiniMax-H3 STAGE CACHING / REPLAY.  Two independent, OFF-BY-DEFAULT tools.
//
//  1. CONDITIONING CACHE (on disk, content-addressed, automatic once enabled).
//     The H3 text encoder is a 50-layer Qwen3-VL-32B running on CPU and costs
//     ~55-65 s of EVERY render.  An A/B that changes a sampler flag re-pays it
//     for a bit-identical result.  Enabled with MINIMAX_H3_COND_CACHE_DIR, a
//     repeat costs a ~3 MB read.
//
//  2. LATENT REPLAY (explicit, user-named save/load slot -- NOT a cache).
//     Dump the sampled packed latent and later decode it again without
//     sampling, so a VAE-side investigation costs seconds instead of ~5 min.
//     Deliberately NOT content-addressed: a sampled latent depends on the DiT
//     weights, the quant, the sampler, the seed, the steps, SA3 policy, the MLP
//     chunk size and every other env knob in the path -- a key over that surface
//     could never be complete, and an incomplete key on a 5-minute artefact is
//     the worst possible trade.  So a hit REQUIRES the user to name the file,
//     exactly as the existing WAN_SAVE_LATENT / LTX_LOAD_LATENT pair does.
//
//  ★ WHY THE KEY DESIGN IS THE WHOLE FEATURE (PROGRESS W22).
//     encode_image_cached() once stored an entry WITHOUT its deepstack features
//     and every later hit silently conditioned on a starved hidden state.  It
//     rendered.  It just rendered wrong.  Nothing in the output said so.
//     Three defences are built in here, in order of how much they buy:
//
//       (a) The FULL KEY TEXT is stored inside the entry and re-compared
//           byte-for-byte on load.  A hash collision is therefore a MISS, never
//           a wrong hit, and `head -c 2000 <entry>` shows a human exactly what
//           the entry was keyed on.
//       (b) STORE-EVERYTHING-OR-REFUSE.  If the SDCondition carries any field
//           this serializer does not know about, the entry is NOT written and a
//           WARN says which field.  A future agent who adds c_input_ids to the
//           H3 conditioner turns the cache OFF rather than silently dropping it.
//           This is the exact inverse of the W22 bug.
//       (c) A BUILD FINGERPRINT is in the key by default, so an image rebuild
//           starts a fresh namespace.  The tokenizer's special-token table, the
//           prompt-weighting decision and the layer-selection logic are all
//           compiled-in and cannot otherwise be observed from here.
//
//     ⚠️ THE FINGERPRINT'S ONE HOLE, stated plainly: __DATE__/__TIME__ expand
//     when THIS HEADER'S translation unit is compiled.  A full Docker build
//     recompiles everything, so every image gets a new namespace -- that is the
//     workflow this is designed for.  But an INCREMENTAL local build that
//     recompiles only src/tokenizers/qwen2_tokenizer.cpp would leave the
//     fingerprint unchanged while changing the token ids, and every hit would
//     then return pre-fix conditioning.  That is precisely the live hazard
//     today (W74 #1: seven special tokens are being added, `<d>` carries all
//     speech).  Escape hatch: MINIMAX_H3_COND_CACHE_SALT=<anything> is mixed
//     into the key, and bumping kCondFormatVersion below invalidates the world.
//     If you are not sure, salt it -- a wasted 55 s beats a wrong measurement.
// ─────────────────────────────────────────────────────────────────────────────

namespace minimax_h3_cache {

// Bump to invalidate every entry ever written. Bump it whenever the SET of
// things the key covers changes -- not when the cache code is merely edited.
constexpr int kCondFormatVersion = 1;

constexpr const char* kCondMagic   = "H3COND\x01\x00";  // 8 bytes
constexpr const char* kLatentMagic = "H3LATENT";        // 8 bytes

// ── small helpers ────────────────────────────────────────────────────────────

// The same twin-FNV-1a 128-bit construction sd_cache::tensor_content_key uses.
// Reused rather than reinvented so both caches collide (or do not) identically.
inline std::string fnv128_hex(const void* data, size_t size) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    uint64_t h1       = 1469598103934665603ull;
    uint64_t h2       = 0x9e3779b97f4a7c15ull;
    for (size_t i = 0; i < size; ++i) {
        h1 = (h1 ^ bytes[i]) * 1099511628211ull;
        h2 = (h2 ^ bytes[i]) * 0x100000001b3ull;
        h2 ^= h2 >> 29;
    }
    char digest[64];
    snprintf(digest, sizeof(digest), "%016llx%016llx", (unsigned long long)h1, (unsigned long long)h2);
    return std::string(digest);
}

inline std::string fnv128_hex(const std::string& text) {
    return fnv128_hex(text.data(), text.size());
}

inline std::string env_or_empty(const char* name) {
    const char* value = getenv(name);
    return (value == nullptr) ? std::string() : std::string(value);
}

inline bool env_is_on(const char* name) {
    const std::string value = env_or_empty(name);
    return !value.empty() && value != "0";
}

inline std::string iso_now() {
    const time_t now = time(nullptr);
    struct tm utc {};
    gmtime_r(&now, &utc);
    char buf[32];
    strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &utc);
    return std::string(buf);
}

// path + size + mtime. Enough to notice a re-converted GGUF at the same path,
// which is the realistic way the text encoder changes under a warm cache dir.
inline std::string file_identity(const std::string& path) {
    if (path.empty()) {
        return "-";
    }
    struct stat st {};
    if (stat(path.c_str(), &st) != 0) {
        return path + ":missing";
    }
    char buf[128];
    snprintf(buf, sizeof(buf), ":%lld:%lld", (long long)st.st_size, (long long)st.st_mtime);
    return path + buf;
}

// ── text-encoder identity ────────────────────────────────────────────────────
//
// The engine does not retain sd_ctx_params, so init() hands the two text-encoder
// paths over here once. Process-global on purpose: the CUDA-owning worker is a
// forked child serving one context (examples/server/worker_supervisor.cpp).
// A second context in one process would share this, which is why a CHANGE is
// logged rather than silently accepted.
inline std::string& te_identity_storage() {
    static std::string identity = "unset";
    return identity;
}

inline void note_text_encoder(const char* llm_path, const char* llm_vision_path) {
    const std::string llm    = (llm_path == nullptr) ? std::string() : std::string(llm_path);
    const std::string vision = (llm_vision_path == nullptr) ? std::string() : std::string(llm_vision_path);
    std::string identity     = "llm=" + file_identity(llm) + "|vis=" + file_identity(vision);
    if (llm.empty() && vision.empty()) {
        // No separate text-encoder file was given, so nothing here can tell one
        // text encoder from another. Left as "unset", which disables the cache
        // -- see cond_cache_enabled().
        identity = "unset";
    }
    std::string& stored = te_identity_storage();
    if (stored != "unset" && stored != identity) {
        LOG_WARN("[H3-COND-CACHE] text encoder identity changed in-process; the key follows the new one");
    }
    stored = std::move(identity);
}

// ── conditioning cache: enablement ───────────────────────────────────────────

inline std::string cond_cache_dir() {
    return env_or_empty("MINIMAX_H3_COND_CACHE_DIR");
}

// ★ Requires a KNOWN text encoder. If the TE did not arrive as its own file
// (--llm / --llm_vision) there is nothing to put in the key that would tell one
// text encoder from another, and a cache that cannot see its own model swapping
// is a wrong-answer generator. Off is the only safe answer, and it says so.
inline bool cond_cache_enabled() {
    if (cond_cache_dir().empty()) {
        return false;
    }
    if (te_identity_storage() == "unset") {
        static bool warned = false;
        if (!warned) {
            warned = true;
            LOG_WARN("[H3-COND-CACHE] disabled: no --llm/--llm_vision path was given, so the text "
                     "encoder cannot be identified in the cache key");
        }
        return false;
    }
    return true;
}

// ── conditioning cache: the key ──────────────────────────────────────────────
//
// EVERY input to encode_minimax_h3() must appear below. What is deliberately
// ABSENT is the whole point of the feature: width, height, frames, steps, seed,
// cfg, flow_shift, sampler, DiT weights, SA3 policy and MLP chunk do not reach
// the text encoder, so changing any of them still HITS. That is the A/B win.
//
// ⚠️ IF THE KEY IS INCOMPLETE the failure is silent and total: the render
// succeeds, looks plausible, and every arm of the A/B is conditioned on a
// hidden state belonging to a different input. The concrete ways that could
// happen here, so a future reader can check them off:
//   * a new field added to ConditionerParams that encode_minimax_h3() reads
//     (add it below AND bump kCondFormatVersion);
//   * a change to the tokenizer table or to prompt-weighting policy without a
//     full rebuild (covered only by the build fingerprint -- see the header
//     comment's stated hole, and use the salt);
//   * a change to TEXT_ENCODER_LAYER or to llm.hpp's layer-selection semantics
//     (the layer number is in the key; the SEMANTICS are build-fingerprint only);
//   * swapping the TE GGUF at the same path with the same size and mtime.
inline std::string cond_key(const std::string& prompt,
                            const std::vector<sd::Tensor<float>>& fl2va_ref_images,
                            const std::vector<MiniMaxH3Reference>& ref2va_references,
                            const RefImageParams& ref_image_params,
                            const std::string& weight_adapter_signature,
                            int clip_skip,
                            int text_encoder_layer,
                            // OPTIONAL and empty today. Closes the build-fingerprint hole
                            // described at the top of this file by putting the ACTUAL tokenizer
                            // behaviour in the key instead of a proxy for it. Filling it needs a
                            // one-line virtual on Conditioner, which is a contended file -- the
                            // diff is in the handoff. Defaulted so applying it later is a
                            // one-argument change at the single call site.
                            const std::string& text_path_fingerprint = std::string()) {
    std::string key;
    key.reserve(2048);
    key += "h3cond/v" + std::to_string(kCondFormatVersion) + "\n";

    if (env_is_on("MINIMAX_H3_COND_CACHE_IGNORE_BUILD")) {
        // Opt-out for someone who KNOWS the text path is unchanged across two
        // builds and wants reuse across them. Named in the entry so a later
        // reader can see the assertion that was made.
        key += "build=IGNORED-BY-REQUEST\n";
    } else {
        key += "build=" __DATE__ "T" __TIME__ "\n";
    }
    key += "salt=" + env_or_empty("MINIMAX_H3_COND_CACHE_SALT") + "\n";
    key += "te=" + te_identity_storage() + "\n";
    key += "textpath=" +
           (text_path_fingerprint.empty() ? std::string("build-fingerprint-only")
                                          : fnv128_hex(text_path_fingerprint)) + "\n";
    key += "layer=" + std::to_string(text_encoder_layer) + "\n";
    key += "adapter=" + weight_adapter_signature + "\n";
    key += "clip_skip=" + std::to_string(clip_skip) + "\n";
    // The vision canvas policy. minimax_h3_vision_canvas() reads exactly these
    // three, and they decide the pixel grid every vision block is encoded on.
    key += "vlm=" + std::to_string(static_cast<int>(ref_image_params.vlm_resize_mode)) + ":" +
           std::to_string(ref_image_params.vlm_min_size) + ":" + std::to_string(ref_image_params.vlm_max_size) + "\n";

    // The prompt goes in by LENGTH + HASH rather than verbatim: a Context-IR
    // document is multi-kilobyte and the entry header should stay readable.
    // The length is there so a hash collision would also have to match it.
    key += "prompt=" + std::to_string(prompt.size()) + ":" + fnv128_hex(prompt) + "\n";

    // fl2va keyframes. Hashing the PIXEL TENSOR (not a path, not a request
    // field) is the ltx-video / krea2 precedent: every geometry decision made
    // upstream of the encode is in the key by construction.
    key += "fl2va=" + std::to_string(fl2va_ref_images.size()) + "\n";
    for (size_t i = 0; i < fl2va_ref_images.size(); ++i) {
        key += "  k" + std::to_string(i) + "=" + sd_cache::tensor_content_key(fl2va_ref_images[i]) + "\n";
    }

    // ref2va references. Kind, sound flag, ordinal position and the block
    // timestamps all change the LABELS in the presentation, not only the
    // pixels, so all of them are keyed. `has_audio` alone moves an "<Audio j>: "
    // label and renumbers every later ordinal.
    key += "ref2va=" + std::to_string(ref2va_references.size()) + "\n";
    for (size_t i = 0; i < ref2va_references.size(); ++i) {
        const MiniMaxH3Reference& reference = ref2va_references[i];
        key += "  r" + std::to_string(i) + "=kind" + std::to_string(static_cast<int>(reference.kind)) +
               ":aud" + std::to_string(reference.has_audio ? 1 : 0);
        if (reference.image != nullptr) {
            key += ":img" + sd_cache::tensor_content_key(*reference.image);
        }
        if (reference.frames != nullptr) {
            key += ":f" + std::to_string(reference.frames->size());
            for (const auto& frame : *reference.frames) {
                key += "," + sd_cache::tensor_content_key(frame);
            }
        }
        key += ":ts";
        for (double seconds : reference.block_timestamps) {
            char buf[32];
            snprintf(buf, sizeof(buf), "%.6f,", seconds);
            key += buf;
        }
        key += "\n";
    }
    return key;
}

inline std::string cond_entry_path(const std::string& key) {
    return cond_cache_dir() + "/" + fnv128_hex(key) + ".h3cond";
}

// A stable 8-char tag for logs, so a human can eyeball "same key" across runs.
inline std::string cond_key_tag(const std::string& key) {
    return fnv128_hex(key).substr(0, 8);
}

// ── conditioning cache: serialization ────────────────────────────────────────
//
// Entry layout (all little-endian, native):
//   char[8]  magic
//   u32      format version
//   u64      key length,  then key bytes (VERBATIM -- re-compared on load)
//   u64      note length, then note bytes (ISO timestamp + shapes; diagnostics)
//   u32      tensor count
//   per tensor: u32 name len, name, u32 dtype (0=f32,1=i32), u32 ndim,
//               i64 dims[ndim], u64 byte count, raw bytes

namespace detail {

inline void write_u32(FILE* f, uint32_t v) { fwrite(&v, sizeof(v), 1, f); }
inline void write_u64(FILE* f, uint64_t v) { fwrite(&v, sizeof(v), 1, f); }
inline void write_i64(FILE* f, int64_t v) { fwrite(&v, sizeof(v), 1, f); }

inline void write_string(FILE* f, const std::string& s) {
    write_u64(f, static_cast<uint64_t>(s.size()));
    if (!s.empty()) {
        fwrite(s.data(), 1, s.size(), f);
    }
}

inline bool read_exact(FILE* f, void* dst, size_t n) {
    return n == 0 || fread(dst, 1, n, f) == n;
}

inline bool read_u32(FILE* f, uint32_t& v) { return read_exact(f, &v, sizeof(v)); }
inline bool read_u64(FILE* f, uint64_t& v) { return read_exact(f, &v, sizeof(v)); }
inline bool read_i64(FILE* f, int64_t& v) { return read_exact(f, &v, sizeof(v)); }

inline bool read_string(FILE* f, std::string& out, uint64_t limit) {
    uint64_t n = 0;
    if (!read_u64(f, n) || n > limit) {
        return false;
    }
    out.assign(static_cast<size_t>(n), '\0');
    return read_exact(f, out.data(), static_cast<size_t>(n));
}

template <typename T>
inline void write_tensor(FILE* f, const std::string& name, uint32_t dtype, const sd::Tensor<T>& tensor) {
    write_u32(f, static_cast<uint32_t>(name.size()));
    fwrite(name.data(), 1, name.size(), f);
    write_u32(f, dtype);
    write_u32(f, static_cast<uint32_t>(tensor.dim()));
    for (int64_t extent : tensor.shape()) {
        write_i64(f, extent);
    }
    const uint64_t bytes = static_cast<uint64_t>(tensor.numel()) * sizeof(T);
    write_u64(f, bytes);
    if (bytes != 0) {
        fwrite(tensor.data(), 1, static_cast<size_t>(bytes), f);
    }
}

template <typename T>
inline bool read_tensor_body(FILE* f, uint32_t dtype, uint32_t expect_dtype, sd::Tensor<T>& out) {
    if (dtype != expect_dtype) {
        return false;
    }
    uint32_t ndim = 0;
    if (!read_u32(f, ndim) || ndim > 8) {
        return false;
    }
    std::vector<int64_t> shape(ndim, 1);
    for (uint32_t i = 0; i < ndim; ++i) {
        if (!read_i64(f, shape[i]) || shape[i] < 0) {
            return false;
        }
    }
    uint64_t bytes = 0;
    if (!read_u64(f, bytes)) {
        return false;
    }
    sd::Tensor<T> tensor(shape);
    if (static_cast<uint64_t>(tensor.numel()) * sizeof(T) != bytes) {
        return false;
    }
    if (!read_exact(f, tensor.data(), static_cast<size_t>(bytes))) {
        return false;
    }
    out = std::move(tensor);
    return true;
}

// Best-effort mkdir -p. A cache that cannot create its directory reports it
// once and stays off; it never fails a render.
inline bool ensure_dir(const std::string& path) {
    if (path.empty()) {
        return false;
    }
    struct stat st {};
    if (stat(path.c_str(), &st) == 0) {
        return S_ISDIR(st.st_mode);
    }
    std::string partial;
    for (size_t i = 0; i <= path.size(); ++i) {
        if (i == path.size() || path[i] == '/') {
            if (!partial.empty() && stat(partial.c_str(), &st) != 0) {
                mkdir(partial.c_str(), 0775);
            }
        }
        if (i < path.size()) {
            partial += path[i];
        }
    }
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

// Byte-budget eviction, oldest mtime first. One flat directory of one file per
// entry, so this is a single readdir + a few unlinks. Bounded by construction:
// nothing accumulates past the budget, and an entry is ~3 MB for a text-only
// prompt (5120 floats x ~140 rows) so the 4 GiB default holds hundreds.
inline void evict_to_budget(const std::string& dir, uint64_t budget_bytes) {
    DIR* handle = opendir(dir.c_str());
    if (handle == nullptr) {
        return;
    }
    struct Entry {
        std::string path;
        time_t mtime = 0;
        uint64_t size = 0;
    };
    std::vector<Entry> entries;
    uint64_t total = 0;
    const time_t now = time(nullptr);
    while (dirent* item = readdir(handle)) {
        const std::string name = item->d_name;
        if (name.size() < 8 || name.compare(name.size() - 7, 7, ".h3cond") != 0) {
            // Abandoned commit temporaries. A worker killed mid-write leaves one, and it is
            // invisible to the byte budget because it has no .h3cond suffix, so it would
            // otherwise accumulate forever. An hour is far longer than any write here takes.
            if (name.find(".h3cond.tmp.") != std::string::npos) {
                const std::string stale = dir + "/" + name;
                struct stat st {};
                if (stat(stale.c_str(), &st) == 0 && now - st.st_mtime > 3600) {
                    unlink(stale.c_str());
                }
            }
            continue;
        }
        Entry entry;
        entry.path = dir + "/" + name;
        struct stat st {};
        if (stat(entry.path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) {
            continue;
        }
        entry.mtime = st.st_mtime;
        entry.size  = static_cast<uint64_t>(st.st_size);
        total += entry.size;
        entries.push_back(std::move(entry));
    }
    closedir(handle);
    if (total <= budget_bytes) {
        return;
    }
    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) { return a.mtime < b.mtime; });
    for (const Entry& entry : entries) {
        if (total <= budget_bytes) {
            break;
        }
        if (unlink(entry.path.c_str()) == 0) {
            total -= entry.size;
            LOG_INFO("[H3-COND-CACHE] evicted %s (%.1f MiB over budget)", entry.path.c_str(),
                     (double)entry.size / (1024.0 * 1024.0));
        }
    }
}

// The ONLY fields encode_minimax_h3() populates. Anything else being non-empty
// means the conditioner grew a field this serializer would drop -- and a
// dropped field is the W22 bug verbatim, so the answer is to refuse.
inline const char* unsupported_condition_field(const SDCondition& cond) {
    if (!cond.c_vector.empty())          return "c_vector";
    if (!cond.c_concat.empty())          return "c_concat";
    if (!cond.c_t5_ids.empty())          return "c_t5_ids";
    if (!cond.c_t5_weights.empty())      return "c_t5_weights";
    if (!cond.c_input_ids.empty())       return "c_input_ids";
    if (!cond.c_position_ids.empty())    return "c_position_ids";
    if (!cond.c_vinput_mask.empty())     return "c_vinput_mask";
    if (!cond.c_image_embeds.empty())    return "c_image_embeds";
    if (!cond.c_ref_images.empty())      return "c_ref_images";
    if (!cond.extra_c_crossattns.empty()) return "extra_c_crossattns";
    if (!cond.c_token_pieces.empty())    return "c_token_pieces";
    return nullptr;
}

// The invariant encode_minimax_h3() asserts before it returns. Re-checked on
// BOTH store and load: a truncated or hand-edited entry that satisfies the
// magic must still describe a row-consistent conditioning or it is discarded.
inline bool condition_is_well_formed(const SDCondition& cond) {
    if (cond.c_crossattn.empty() || cond.c_token_types.empty()) {
        return false;
    }
    if (cond.c_crossattn.dim() < 2) {
        return false;
    }
    return cond.c_crossattn.shape()[1] == cond.c_token_types.numel();
}

}  // namespace detail

// Returns true and fills `out` only on a VERIFIED hit: magic ok, format ok,
// stored key byte-identical to `key`, both tensors present, invariant holds.
// Every other outcome is a miss and says why.
inline bool load_condition(const std::string& key, SDCondition& out) {
    const std::string path = cond_entry_path(key);
    FILE* f                = fopen(path.c_str(), "rb");
    if (f == nullptr) {
        LOG_INFO("[H3-COND-CACHE] MISS %s (no entry) -- encoding text (expect ~55-65 s on CPU)",
                 cond_key_tag(key).c_str());
        return false;
    }
    struct Closer {
        FILE* f;
        ~Closer() { if (f != nullptr) fclose(f); }
    } closer{f};

    char magic[8] = {0};
    uint32_t format = 0;
    std::string stored_key;
    std::string note;
    uint32_t count = 0;
    if (!detail::read_exact(f, magic, sizeof(magic)) || memcmp(magic, kCondMagic, 8) != 0 ||
        !detail::read_u32(f, format) || format != static_cast<uint32_t>(kCondFormatVersion) ||
        !detail::read_string(f, stored_key, 1u << 20) ||
        !detail::read_string(f, note, 1u << 16) ||
        !detail::read_u32(f, count) || count > 16) {
        LOG_WARN("[H3-COND-CACHE] MISS %s (unreadable/foreign entry %s) -- encoding text",
                 cond_key_tag(key).c_str(), path.c_str());
        return false;
    }
    // ★ The collision guard. The hash only picks the FILE; the key decides
    // whether it is ours.
    if (stored_key != key) {
        LOG_WARN("[H3-COND-CACHE] MISS %s (KEY MISMATCH inside %s -- hash collision or a stale "
                 "format; treating as a miss) -- encoding text",
                 cond_key_tag(key).c_str(), path.c_str());
        return false;
    }

    SDCondition loaded;
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t name_len = 0;
        if (!detail::read_u32(f, name_len) || name_len > 64) {
            return false;
        }
        std::string name(name_len, '\0');
        uint32_t dtype = 0;
        if (!detail::read_exact(f, name.data(), name_len) || !detail::read_u32(f, dtype)) {
            return false;
        }
        bool ok = false;
        if (name == "c_crossattn") {
            ok = detail::read_tensor_body<float>(f, dtype, 0, loaded.c_crossattn);
        } else if (name == "c_token_types") {
            ok = detail::read_tensor_body<int32_t>(f, dtype, 1, loaded.c_token_types);
        }
        if (!ok) {
            LOG_WARN("[H3-COND-CACHE] MISS %s (entry carries unknown/corrupt tensor '%s') -- encoding text",
                     cond_key_tag(key).c_str(), name.c_str());
            return false;
        }
    }
    if (!detail::condition_is_well_formed(loaded)) {
        LOG_WARN("[H3-COND-CACHE] MISS %s (entry failed the row-count invariant; deleting %s) -- encoding text",
                 cond_key_tag(key).c_str(), path.c_str());
        unlink(path.c_str());
        return false;
    }

    LOG_INFO("[H3-COND-CACHE] HIT %s -- TEXT ENCODE SKIPPED. %lld rows x %lld dim, %s",
             cond_key_tag(key).c_str(),
             (long long)loaded.c_crossattn.shape()[1],
             (long long)loaded.c_crossattn.shape()[0],
             note.c_str());
    out = std::move(loaded);
    return true;
}

inline void store_condition(const std::string& key, const SDCondition& cond) {
    if (env_is_on("MINIMAX_H3_COND_CACHE_RO")) {
        return;
    }
    if (const char* field = detail::unsupported_condition_field(cond); field != nullptr) {
        LOG_WARN("[H3-COND-CACHE] NOT STORING %s: the conditioning carries '%s', which this "
                 "serializer does not know about. Storing it would silently drop that field on "
                 "every later hit (PROGRESS W22). Teach minimax_h3_cache.hpp about it and bump "
                 "kCondFormatVersion.",
                 cond_key_tag(key).c_str(), field);
        return;
    }
    if (!detail::condition_is_well_formed(cond)) {
        LOG_WARN("[H3-COND-CACHE] NOT STORING %s: conditioning failed its own row-count invariant",
                 cond_key_tag(key).c_str());
        return;
    }
    const std::string dir = cond_cache_dir();
    if (!detail::ensure_dir(dir)) {
        LOG_WARN("[H3-COND-CACHE] cannot create MINIMAX_H3_COND_CACHE_DIR '%s'; cache stays off", dir.c_str());
        return;
    }

    const std::string final_path = cond_entry_path(key);
    // ★ Written to a temp file and rename()d. A worker that is killed or
    // /v1/admin/unload'd mid-write (which happens between every A/B arm here)
    // must never leave a truncated file that later reads as a hit.
    const std::string tmp_path = final_path + ".tmp." + std::to_string((long long)getpid());

    const std::string note = "stored " + iso_now() + ", " +
                             std::to_string((long long)cond.c_crossattn.shape()[1]) + " rows x " +
                             std::to_string((long long)cond.c_crossattn.shape()[0]) + " dim";

    FILE* f = fopen(tmp_path.c_str(), "wb");
    if (f == nullptr) {
        LOG_WARN("[H3-COND-CACHE] cannot write %s; cache stays off", tmp_path.c_str());
        return;
    }
    fwrite(kCondMagic, 1, 8, f);
    detail::write_u32(f, static_cast<uint32_t>(kCondFormatVersion));
    detail::write_string(f, key);
    detail::write_string(f, note);
    detail::write_u32(f, 2);
    detail::write_tensor<float>(f, "c_crossattn", 0, cond.c_crossattn);
    detail::write_tensor<int32_t>(f, "c_token_types", 1, cond.c_token_types);
    const bool ok = (ferror(f) == 0);
    fclose(f);
    if (!ok || rename(tmp_path.c_str(), final_path.c_str()) != 0) {
        unlink(tmp_path.c_str());
        LOG_WARN("[H3-COND-CACHE] failed to commit %s", final_path.c_str());
        return;
    }

    const uint64_t budget_mb = [] {
        const std::string value = env_or_empty("MINIMAX_H3_COND_CACHE_MAX_MB");
        if (value.empty()) {
            return static_cast<uint64_t>(4096);
        }
        const long long parsed = atoll(value.c_str());
        return parsed > 0 ? static_cast<uint64_t>(parsed) : static_cast<uint64_t>(4096);
    }();
    detail::evict_to_budget(dir, budget_mb * 1024ull * 1024ull);

    LOG_INFO("[H3-COND-CACHE] STORED %s -> %s (%s)", cond_key_tag(key).c_str(), final_path.c_str(), note.c_str());
}

// ─────────────────────────────────────────────────────────────────────────────
//  LATENT REPLAY
// ─────────────────────────────────────────────────────────────────────────────

inline std::string save_latent_prefix() { return env_or_empty("MINIMAX_H3_SAVE_LATENT"); }
inline std::string load_latent_prefix() { return env_or_empty("MINIMAX_H3_LOAD_LATENT"); }

inline bool replay_requested() { return !load_latent_prefix().empty(); }

// Accepts either the bare prefix ("/out/run7") or the full file
// ("/out/run7.h3latent"), because both spellings are the obvious one.
inline std::string latent_file(const std::string& prefix) {
    if (prefix.size() > 9 && prefix.compare(prefix.size() - 9, 9, ".h3latent") == 0) {
        return prefix;
    }
    return prefix + ".h3latent";
}

namespace detail {

// numpy .npy v1.0, dtype '<f4', C-order.
//
// ⚠️ THE SHAPE IS REVERSED ON PURPOSE. sd::Tensor / ggml put the FASTEST-VARYING
// axis at shape()[0]; numpy C-order puts it LAST. Writing the shape unreversed
// would produce a file that loads without error and is transposed -- the same
// class of silent damage as the {T,C,S} vs {T,S,C} audio swap this tree already
// documents. The log line prints both spellings so you can check.
inline bool write_npy_f32(const std::string& path, const sd::Tensor<float>& tensor) {
    if (tensor.empty()) {
        return false;
    }
    std::string shape_text = "(";
    for (int64_t axis = tensor.dim() - 1; axis >= 0; --axis) {
        shape_text += std::to_string(tensor.shape()[axis]) + ",";
    }
    shape_text += ")";

    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': " + shape_text + ", }";
    size_t prefix_len  = 10 + header.size() + 1;
    while (prefix_len % 64 != 0) {
        header += ' ';
        prefix_len++;
    }
    header += '\n';

    FILE* f = fopen(path.c_str(), "wb");
    if (f == nullptr) {
        return false;
    }
    const unsigned char magic[8] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
    const uint16_t header_len    = static_cast<uint16_t>(header.size());
    fwrite(magic, 1, sizeof(magic), f);
    fwrite(&header_len, sizeof(header_len), 1, f);
    fwrite(header.data(), 1, header.size(), f);
    fwrite(tensor.data(), sizeof(float), static_cast<size_t>(tensor.numel()), f);
    const bool ok = (ferror(f) == 0);
    fclose(f);
    return ok;
}

inline std::string shape_text(const sd::Tensor<float>& tensor) {
    std::string text;
    for (size_t axis = 0; axis < tensor.shape().size(); ++axis) {
        text += (axis != 0 ? "x" : "") + std::to_string(tensor.shape()[axis]);
    }
    return text;
}

}  // namespace detail

// Dump the sampled PACKED latent (video channels ++ trailing audio channels)
// plus both audio-latent spellings as .npy for numerical inspection.
//
// Called once, after sampling, before anything slices the latent -- so what is
// on disk is exactly what the audio unpack and the video decode both read.
inline void save_sampled_latent(const sd::Tensor<float>& packed,
                                int audio_t,
                                int latent_channels,
                                const std::string& prompt) {
    const std::string prefix = save_latent_prefix();
    if (prefix.empty() || packed.empty()) {
        return;
    }
    const std::string path = latent_file(prefix);

    const std::string stamp = "saved=" + iso_now() + " build=" __DATE__ "T" __TIME__ " shape=" +
                              detail::shape_text(packed) + " audio_t=" + std::to_string(audio_t) +
                              " latent_channels=" + std::to_string(latent_channels) +
                              " prompt_bytes=" + std::to_string(prompt.size()) +
                              " prompt_hash=" + fnv128_hex(prompt) +
                              " note=" + env_or_empty("MINIMAX_H3_LATENT_NOTE");

    FILE* f = fopen(path.c_str(), "wb");
    if (f == nullptr) {
        LOG_ERROR("[H3-LATENT] cannot write %s", path.c_str());
        return;
    }
    fwrite(kLatentMagic, 1, 8, f);
    detail::write_u32(f, 1);
    detail::write_string(f, stamp);
    detail::write_u32(f, static_cast<uint32_t>(audio_t));
    detail::write_u32(f, static_cast<uint32_t>(latent_channels));
    detail::write_tensor<float>(f, "packed_latent", 0, packed);
    const bool ok = (ferror(f) == 0);
    fclose(f);
    if (!ok) {
        LOG_ERROR("[H3-LATENT] failed writing %s", path.c_str());
        return;
    }
    LOG_INFO("[H3-LATENT] SAVED %s (%s) -- replay with MINIMAX_H3_LOAD_LATENT=%s",
             path.c_str(), stamp.c_str(), prefix.c_str());

    // The raw audio latent, in BOTH layouts, each named for what it is. The two
    // differ by one transposition, have identical element counts, and swapping
    // them silently smears the stereo field -- so the dump never makes the
    // reader guess which one they have.
    if (audio_t > 0) {
        auto audio_request = minimax_h3_unpack_audio_latent(packed, audio_t, latent_channels, 32);
        if (audio_request.empty()) {
            LOG_WARN("[H3-LATENT] audio latent could not be unpacked (audio_t=%d)", audio_t);
            return;
        }
        auto audio_vae = minimax_h3_swap_audio_axes(audio_request);
        const std::string request_path = prefix + ".audio_request_TSC.npy";
        const std::string vae_path     = prefix + ".audio_vae_TCS.npy";
        if (detail::write_npy_f32(request_path, audio_request)) {
            LOG_INFO("[H3-LATENT] wrote %s -- ggml {T=%d, S=2, C=32, 1}, numpy shape (1,32,2,%d)",
                     request_path.c_str(), audio_t, audio_t);
        }
        if (detail::write_npy_f32(vae_path, audio_vae)) {
            LOG_INFO("[H3-LATENT] wrote %s -- ggml {T=%d, C=32, S=2, 1}, numpy shape (1,2,32,%d) "
                     "(this is the layout the audio VAE decodes)",
                     vae_path.c_str(), audio_t, audio_t);
        }
    }
}

// Replace the would-be-sampled latent with a saved one.
//
// Returns false on ANY mismatch and the caller must fail the render: replay is
// only ever requested explicitly, so quietly falling back to a 5-minute sample
// would waste the render AND hide the mistake. Every refusal names the field.
inline bool load_sampled_latent(sd::Tensor<float>& out,
                                const sd::Tensor<float>& expected_shape_like,
                                int expected_audio_t,
                                int expected_latent_channels) {
    const std::string path = latent_file(load_latent_prefix());
    FILE* f                = fopen(path.c_str(), "rb");
    if (f == nullptr) {
        LOG_ERROR("[H3-LATENT] MINIMAX_H3_LOAD_LATENT is set but %s cannot be opened", path.c_str());
        return false;
    }
    struct Closer {
        FILE* f;
        ~Closer() { if (f != nullptr) fclose(f); }
    } closer{f};

    char magic[8]   = {0};
    uint32_t format = 0;
    std::string stamp;
    uint32_t audio_t         = 0;
    uint32_t latent_channels = 0;
    uint32_t name_len        = 0;
    uint32_t dtype           = 0;
    if (!detail::read_exact(f, magic, sizeof(magic)) || memcmp(magic, kLatentMagic, 8) != 0 ||
        !detail::read_u32(f, format) || format != 1 ||
        !detail::read_string(f, stamp, 1u << 16) ||
        !detail::read_u32(f, audio_t) || !detail::read_u32(f, latent_channels) ||
        !detail::read_u32(f, name_len) || name_len > 64) {
        LOG_ERROR("[H3-LATENT] %s is not a readable h3latent file", path.c_str());
        return false;
    }
    std::string name(name_len, '\0');
    sd::Tensor<float> latent;
    if (!detail::read_exact(f, name.data(), name_len) || !detail::read_u32(f, dtype) ||
        !detail::read_tensor_body<float>(f, dtype, 0, latent)) {
        LOG_ERROR("[H3-LATENT] %s has a corrupt tensor payload", path.c_str());
        return false;
    }

    if (static_cast<int>(audio_t) != expected_audio_t) {
        LOG_ERROR("[H3-LATENT] REFUSING %s: it holds audio_t=%u, this render wants %d. "
                  "Replaying it would decode a different clip length. (%s)",
                  path.c_str(), audio_t, expected_audio_t, stamp.c_str());
        return false;
    }
    if (static_cast<int>(latent_channels) != expected_latent_channels) {
        LOG_ERROR("[H3-LATENT] REFUSING %s: it holds latent_channels=%u, this model has %d. (%s)",
                  path.c_str(), latent_channels, expected_latent_channels, stamp.c_str());
        return false;
    }
    if (!expected_shape_like.empty() && latent.shape() != expected_shape_like.shape()) {
        LOG_ERROR("[H3-LATENT] REFUSING %s: it holds %s, this render's latent is %s. "
                  "Resolution/frames must match the run that saved it. (%s)",
                  path.c_str(), detail::shape_text(latent).c_str(),
                  detail::shape_text(expected_shape_like).c_str(), stamp.c_str());
        return false;
    }

    LOG_INFO("[H3-LATENT] REPLAY -- SAMPLING SKIPPED. Decoding %s (%s)", path.c_str(), stamp.c_str());
    out = std::move(latent);
    return true;
}

// ── audio-latent injection (VAE-vs-DiT isolation) ────────────────────────────
//
// Splice a KNOWN audio latent into the packed tensor, leaving the video
// channels alone. The point: round-trip a reference waveform through
// MiniMaxH3AudioVAERunner::encode, dump it, then inject it here. If the decode
// is clean, the audio VAE is exonerated and the DiT's audio velocity is the
// suspect (W81); if it is still noise, the VAE is.
//
// Input file: the raw float32 payload of a `.audio_request_TSC.npy` dump, i.e.
// {T, 2, 32} request layout, headerless. Element count must match EXACTLY --
// a short file is refused, never zero-padded.
inline bool inject_audio_latent(sd::Tensor<float>& packed, int audio_t, int latent_channels) {
    const std::string path = env_or_empty("MINIMAX_H3_INJECT_AUDIO_LATENT");
    if (path.empty()) {
        return true;  // not requested
    }
    if (packed.empty() || audio_t <= 0) {
        LOG_ERROR("[H3-LATENT] MINIMAX_H3_INJECT_AUDIO_LATENT set but this render has no audio latent");
        return false;
    }
    const size_t values = static_cast<size_t>(audio_t) * 2u * 32u;
    struct stat st {};
    if (stat(path.c_str(), &st) != 0 || static_cast<size_t>(st.st_size) != values * sizeof(float)) {
        LOG_ERROR("[H3-LATENT] REFUSING to inject %s: expected exactly %zu float32 (%zu bytes) for "
                  "audio_t=%d in {T,2,32} request layout, file is %lld bytes",
                  path.c_str(), values, values * sizeof(float), audio_t, (long long)st.st_size);
        return false;
    }
    std::vector<float> buffer(values);
    FILE* f = fopen(path.c_str(), "rb");
    if (f == nullptr || fread(buffer.data(), sizeof(float), values, f) != values) {
        if (f != nullptr) {
            fclose(f);
        }
        LOG_ERROR("[H3-LATENT] could not read %s", path.c_str());
        return false;
    }
    fclose(f);

    const int64_t spatial = packed.shape()[0] * packed.shape()[1] * packed.shape()[2];
    const int64_t total   = packed.shape()[3];
    if (total <= latent_channels ||
        (total - latent_channels) * spatial < static_cast<int64_t>(values)) {
        LOG_ERROR("[H3-LATENT] packed latent has no room for the injected audio latent");
        return false;
    }
    // Exact mirror of minimax_h3_unpack_audio_latent's flat read.
    std::copy_n(buffer.data(), values,
                packed.data() + static_cast<size_t>(latent_channels) * static_cast<size_t>(spatial));
    LOG_INFO("[H3-LATENT] INJECTED audio latent from %s (%zu floats) -- the video channels are "
             "untouched, so the decode isolates the audio VAE from the DiT",
             path.c_str(), values);
    return true;
}

}  // namespace minimax_h3_cache

#endif  // __SD_MODEL_DIFFUSION_MINIMAX_H3_CACHE_HPP__
