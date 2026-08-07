// Multipart media parts: collecting them off a request, and proving they arrived intact.
//
// `docs/media-transport.md` §4 — a media string in a request may read `part:<name>` instead of
// carrying base64, and the bytes ride as a binary part of the same multipart request. This file is
// the two ends of that: pulling the parts off an httplib request into the job record, and (§9.3)
// checking a hash-derived part name against what actually arrived.

#include <array>
#include <cstdint>
#include <cstring>

#include "routes.h"

namespace {

// ── SHA-256 ──────────────────────────────────────────────────────────────────
//
// Vendored rather than pulled in: the whole use is verifying a part name, and the alternative was a
// dependency on OpenSSL in a CUDA image for 60 lines of arithmetic. FIPS 180-4, unrolled the boring
// way.

constexpr uint32_t k_sha256_k[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

inline uint32_t rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

void sha256_block(const uint8_t* block, std::array<uint32_t, 8>& h) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
        w[i] = (uint32_t)block[i * 4] << 24 | (uint32_t)block[i * 4 + 1] << 16 | (uint32_t)block[i * 4 + 2] << 8 |
               (uint32_t)block[i * 4 + 3];
    }
    for (int i = 16; i < 64; ++i) {
        const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i]              = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
        const uint32_t s1    = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        const uint32_t ch    = (e & f) ^ (~e & g);
        const uint32_t temp1 = hh + s1 + ch + k_sha256_k[i] + w[i];
        const uint32_t s0    = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        const uint32_t maj   = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t temp2 = s0 + maj;
        hh                   = g;
        g                    = f;
        f                    = e;
        e                    = d + temp1;
        d                    = c;
        c                    = b;
        b                    = a;
        a                    = temp1 + temp2;
    }
    h[0] += a;
    h[1] += b;
    h[2] += c;
    h[3] += d;
    h[4] += e;
    h[5] += f;
    h[6] += g;
    h[7] += hh;
}

std::string sha256_hex(const std::string& data) {
    std::array<uint32_t, 8> h = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                                 0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    const uint8_t* p          = reinterpret_cast<const uint8_t*>(data.data());
    size_t remaining          = data.size();
    while (remaining >= 64) {
        sha256_block(p, h);
        p += 64;
        remaining -= 64;
    }
    uint8_t tail[128] = {0};
    std::memcpy(tail, p, remaining);
    tail[remaining]         = 0x80;
    const size_t tail_len   = (remaining < 56) ? 64 : 128;
    const uint64_t bit_len  = static_cast<uint64_t>(data.size()) * 8;
    for (int i = 0; i < 8; ++i) {
        tail[tail_len - 1 - i] = static_cast<uint8_t>((bit_len >> (8 * i)) & 0xff);
    }
    sha256_block(tail, h);
    if (tail_len == 128) {
        sha256_block(tail + 64, h);
    }
    static const char* hex = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (uint32_t word : h) {
        for (int shift = 28; shift >= 0; shift -= 4) {
            out.push_back(hex[(word >> shift) & 0xf]);
        }
    }
    return out;
}

// koblem's `stage_refs` names a staged asset `a_<first 16 hex of its sha256>`. Anything else --
// `ref_0`, `audio_full`, a caller's own name -- carries no claim about its contents and is left
// alone.
bool is_hash_named(const std::string& name, std::string& out_hex_prefix) {
    if (name.size() != 18 || name.compare(0, 2, "a_") != 0) {
        return false;
    }
    for (size_t i = 2; i < name.size(); ++i) {
        const char c = name[i];
        const bool hex = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        if (!hex) {
            return false;
        }
    }
    out_hex_prefix = name.substr(2);
    return true;
}

}  // namespace

MediaPartTable collect_media_parts(const httplib::Request& req) {
    MediaPartTable parts;
    if (!req.is_multipart_form_data()) {
        return parts;
    }
    for (const auto& entry : req.form.files) {
        // `files` is a multimap: repeated names are legal on the wire. First one wins, which
        // matches how every other reader here treats a duplicated field, and a duplicate under a
        // hash-derived name is the same bytes anyway.
        parts.emplace(entry.first, entry.second.content);
    }
    return parts;
}

std::string first_media_part_hash_mismatch(const MediaPartTable& parts) {
    for (const auto& [name, bytes] : parts) {
        std::string expected;
        if (!is_hash_named(name, expected)) {
            continue;
        }
        if (sha256_hex(bytes).compare(0, expected.size(), expected) != 0) {
            return name;
        }
    }
    return "";
}
