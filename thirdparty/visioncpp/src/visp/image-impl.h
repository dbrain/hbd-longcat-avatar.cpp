#pragma once

#include "util/math.h"
#include "util/string.h"
#include "visp/image.h"

#include <algorithm>
#include <array>

namespace visp {

// uint8 <-> float32 pixel conversions
// - no sRGB gamma correction for now, ML pipelines usually have some form of normalization

i32x4 get_channel_map(image_format);

inline f32x4 image_load(uint8_t const* img, size_t i, i32x4) {
    float v = float(img[i]) / 255.0f;
    return f32x4{v};
}

inline f32x4 image_load(u8x3 const* img, size_t i, i32x4) {
    u8x3 u = img[i];
    f32x4 v = f32x4{float(u[0]), float(u[1]), float(u[2]), 1.0f};
    return v / 255.0f;
}

inline f32x4 image_load(u8x4 const* img, size_t i, i32x4 map) {
    u8x4 u = img[i];
    f32x4 v = f32x4{float(u[map[0]]), float(u[map[1]]), float(u[map[2]]), float(u[map[3]])};
    return v / 255.0f;
}

inline void image_store(uint8_t* img, size_t i, f32x4 value) {
    img[i] = uint8_t(clamp(value[0], 0.0f, 1.0f) * 255.0f);
}

inline void image_store(u8x4* img, size_t i, f32x4 value) {
    value = clamp(value, 0.0f, 1.0f) * 255.0f;
    img[i] = u8x4{uint8_t(value[0]), uint8_t(value[1]), uint8_t(value[2]), uint8_t(value[3])};
}

// float32 pixel load/store

inline f32x4 image_load(float const* img, size_t i, i32x4) {
    return f32x4{img[i]};
}

inline f32x4 image_load(f32x3 const* img, size_t i, i32x4) {
    return f32x4{img[i][0], img[i][1], img[i][2], 1.0f};
}

inline f32x4 image_load(f32x4 const* img, size_t i, i32x4) {
    return img[i];
}

inline void image_store(float* img, size_t i, f32x4 value) {
    img[i] = value[0];
}

inline void image_store(f32x3* img, size_t i, f32x4 value) {
    img[i] = f32x3{value[0], value[1], value[2]};
}

inline void image_store(f32x4* img, size_t i, f32x4 value) {
    img[i] = value;
}

// clang-format off
template <typename T> struct scalar_type_impl          { using type = typename T::value_type; };
template <>           struct scalar_type_impl<float>   { using type = float; };
template <>           struct scalar_type_impl<uint8_t> { using type = uint8_t; };
template <typename T> using scalar_type = typename scalar_type_impl<T>::type;

constexpr uint8_t one(uint8_t) { return 255; }
constexpr float one(float) { return 1.0f; }
// clang-format on

//
// Image source - supports reading u8/f32 images with any format

template <typename T> // uint8_t, u8x3, u8x4, float, f32x3, f32x4
struct image_source {
    i32x2 extent{};
    T const* data;
    size_t stride; // row stride in elements
    i32x4 channel_map{0, 1, 2, 3};

    using scalar = scalar_type<T>;
    static constexpr int n_channels = sizeof(T) / sizeof(scalar);

    image_source(i32x2 extent, T const* ptr, size_t stride)
        : extent(extent), data(ptr), stride(stride) {}

    image_source(image_view img)
        : image_source(img.extent, (T const*)img.data, img.stride / n_bytes(img.format)) {            
        ASSERT(visp::n_channels(img) == n_channels);
        ASSERT(is_float(img.format) == (std::is_same_v<scalar, float>));

        channel_map = get_channel_map(img.format);
    }

    T const& operator[](size_t i) const { return data[i]; }

    f32x4 load(size_t i) const { return image_load(data, i, channel_map); }
    f32x4 load(i32x2 c) const { return load(c[1] * stride + c[0]); }
};

template <typename T>
inline int n_pixels(image_source<T> const& img) {
    return img.extent[0] * img.extent[1];
}

//
// Image target - supports writing u8/f32 images with alpha or rgba format

template <typename T> // uint8_t, u8x4, float, f32x3, f32x4
struct image_target : image_source<T> {

    image_target(i32x2 extent, T* ptr, size_t stride) : image_source<T>(extent, ptr, stride) {}

    image_target(image_data& img) : image_source<T>(img) {
        ASSERT(
            img.format != image_format::argb_u8 && img.format != image_format::bgra_u8 &&
            img.format != image_format::rgb_u8); // not supported for writing
    }

    image_target(image_span img) : image_source<T>(img) {}

    T& operator[](size_t i) const { return ((T*)this->data)[i]; }

    void store(size_t i, f32x4 value) const { image_store((T*)this->data, i, value); }
    void store(i32x2 c, f32x4 value) const { store(c[1] * this->stride + c[0], value); }
};

} // namespace visp