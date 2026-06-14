// ktx2_encode.hpp — in-process KTX2/BasisU texture encoder (native C++, no external gltfpack/toktx).
// Wraps the vendored basis_universal encoder (thirdparty/basis_universal). Feed an RGBA8 atlas, get
// back KTX2 file bytes ready to embed in a GLB image (KHR_texture_basisu).
//
//   hero  : UASTC LDR 4x4 + Zstd supercompression  — near-lossless, larger. Use for baseColor heroes.
//   small : ETC1S (BasisLZ)                          — supercompressed, much smaller. Use for LODs.
//
// Build: add thirdparty/basis_universal to the include path and compile the encoder+transcoder+zstd
// source set with -DBASISD_SUPPORT_KTX2=1 -DBASISD_SUPPORT_KTX2_ZSTD=1 -DBASISU_SUPPORT_OPENCL=0
// -DBASISU_SUPPORT_SSE=1 -msse4.1 -DBASISU_DISABLE_ANDROID_ASTC_DECOMP=1 (see build.sh basisu deps).
#pragma once
#include "encoder/basisu_comp.h"
#include <cstdint>
#include <mutex>
#include <vector>

namespace ktx2enc {

inline void ensure_init() {
    static std::once_flag once;
    std::call_once(once, [] { basisu::basisu_encoder_init(/*use_opencl*/ false); });
}

// rgba: tightly-packed w*h*4 bytes (R,G,B,A). uastc=true → near-lossless UASTC+Zstd (hero); false →
// ETC1S (small). quality: ETC1S level [1,255] (higher = better/larger; ignored for UASTC). srgb=true
// for color data (baseColor); false for linear data (metallicRoughness, normals). mipmaps generated
// for trilinear sampling on the asset. Returns the .ktx2 file bytes, or empty on failure.
// comps: source channel count (4 = RGBA baseColor; 3 = RGB metallicRoughness/normal). basisu reads
// `comps` bytes per pixel — passing the wrong count shifts every pixel (garbage roughness → shiny mess).
inline std::vector<uint8_t> encode(const uint8_t* rgba, int w, int h,
                                   bool uastc, int quality, bool srgb, int comps = 4, int threads = 0) {
    ensure_init();
    using namespace basisu;

    basis_compressor_params p;
    job_pool jpool(threads > 0 ? (uint32_t)threads : 1u);
    p.m_pJob_pool = &jpool;
    p.m_multithreading = threads != 1;          // pool of >=1; let basisu parallelize when >1
    p.m_read_source_images = false;             // we supply pixels directly, not filenames
    p.m_write_output_basis_or_ktx2_files = false;
    p.m_create_ktx2_file = true;                // emit .ktx2 (not .basis)
    p.m_perceptual = srgb;                       // sRGB color vs linear data metric
    p.set_srgb_options(srgb);                    // sets KTX2 transfer function + mip colorspace too
    p.m_mip_gen = true;                          // mipmaps (3D asset sampled at varying distance)

    p.m_source_images.resize(1);
    p.m_source_images[0].init(rgba, (uint32_t)w, (uint32_t)h, (uint32_t)comps);

    if (uastc) {
        p.set_format_mode(basist::basis_tex_format::cUASTC_LDR_4x4);
        p.m_pack_uastc_ldr_4x4_flags = cPackUASTCLevelDefault;
        p.m_rdo_uastc_ldr_4x4 = false;          // no rate-distortion: keep it near-lossless
        p.m_ktx2_uastc_supercompression = basist::KTX2_SS_ZSTANDARD;  // shrink the 8bpp UASTC payload
    } else {
        p.set_format_mode(basist::basis_tex_format::cETC1S);
        p.m_quality_level = (quality > 0) ? quality : 192;           // [1,255]; 192 ~= a clean default
    }

    basis_compressor c;
    if (!c.init(p)) return {};
    if (c.process() != basis_compressor::cECSuccess) return {};
    const uint8_vec& k = c.get_output_ktx2_file();
    return std::vector<uint8_t>(k.begin(), k.end());
}

}  // namespace ktx2enc
