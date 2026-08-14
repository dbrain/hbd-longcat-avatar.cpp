#pragma once

#include <cstdint>
#include <exception>
#include <string_view>

#ifdef _MSC_VER
#    ifdef VISP_API_EXPORT
#        define VISP_API __declspec(dllexport)
#    else
#        define VISP_API __declspec(dllimport)
#    endif
#else
#    define VISP_API __attribute__((visibility("default")))
#endif

namespace visp {

//
// Fixed string - fixed length, does not allocate, truncates if too long

template <size_t N>
struct fixed_string {
    char data[N] = {0};
    size_t length = 0;

    constexpr fixed_string() {}

    fixed_string(char const* str) {
        auto view = std::string_view(str);
        length = std::min(view.size(), N - 1);
        std::copy(view.begin(), view.begin() + length, data);
    }

    template <size_t M>
    constexpr fixed_string(char const (&str)[M]) {
        static_assert(M <= N, "String literal is too long for fixed_string");
        length = M - 1;
        std::copy(str, str + length, data);
    }

    char const* c_str() const { return data; }

    std::string_view view() const { return {data, length}; }

    explicit operator bool() const { return length > 0; }
};

//
// Exception type used in the library for recoverable errors

struct exception : std::exception {
    fixed_string<128> message;

    explicit exception(char const* msg) : message(msg) {}
    explicit exception(fixed_string<128> msg) : message(msg) {}

    char const* what() const noexcept override { return message.c_str(); }
};

//
// Simple vector types (fixed-size arrays)

template <typename T>
struct vec2 {
    using value_type = T;
    static constexpr int dim = 2;

    T v[2];

    constexpr vec2() : v{0, 0} {}
    constexpr vec2(T x, T y) : v{x, y} {}
    constexpr vec2(T x) : v{x, x} {}

    constexpr T& operator[](size_t i) { return v[i]; }
    constexpr T const& operator[](size_t i) const { return v[i]; }

    constexpr auto operator<=>(vec2 const&) const = default;
};

template <typename T>
struct vec3 {
    using value_type = T;
    static constexpr int dim = 3;

    T v[3];

    constexpr vec3() : v{0, 0, 0} {}
    constexpr vec3(T x, T y, T z) : v{x, y, z} {}
    constexpr vec3(T x) : v{x, x, x} {}

    constexpr T& operator[](size_t i) { return v[i]; }
    constexpr T const& operator[](size_t i) const { return v[i]; }

    constexpr auto operator<=>(vec3 const&) const = default;
};

template <typename T, size_t Align = alignof(T)>
struct vec4 {
    using value_type = T;
    static constexpr int dim = 4;

    alignas(Align) T v[4];

    constexpr vec4() : v{0, 0, 0, 0} {}
    constexpr vec4(T x, T y, T z, T w) : v{x, y, z, w} {}
    constexpr vec4(T x) : v{x, x, x, x} {}

    constexpr T& operator[](size_t i) { return v[i]; }
    constexpr T const& operator[](size_t i) const { return v[i]; }

    constexpr auto operator<=>(vec4 const&) const = default;
};

using u8x3 = vec3<uint8_t>;
using u8x4 = vec4<uint8_t>;

using i32x2 = vec2<int32_t>;
using i32x4 = vec4<int32_t>;

using i64x2 = vec2<int64_t>;
using i64x4 = vec4<int64_t>;

using f32x2 = vec2<float>;
using f32x3 = vec3<float>;
using f32x4 = vec4<float, 16>;

//
// Flags - extends enums with bit-wise operations to be used as bitmask

template <typename E>
struct flags {
    using enum_type = E;

    uint32_t value = 0;

    constexpr flags() = default;
    constexpr flags(E value) : value(uint32_t(value)) {}
    explicit constexpr flags(uint32_t value) : value(value) {}

    flags& operator|=(E other) {
        value |= uint32_t(other);
        return *this;
    }
    
    flags& operator|=(flags other) {
        value |= other.value;
        return *this;
    }

    constexpr flags operator~() const { return flags(~value); }
    explicit constexpr operator bool() const { return value != 0; }

    friend constexpr flags operator&(flags lhs, E rhs) { return flags(lhs.value & uint32_t(rhs)); }
    friend constexpr flags operator&(flags lhs, flags rhs) { return flags(lhs.value & rhs.value); }

    friend constexpr flags operator|(flags lhs, E rhs) { return flags(lhs.value | uint32_t(rhs)); }
    friend constexpr flags operator|(flags lhs, flags rhs) { return flags(lhs.value | rhs.value); }
};

} // namespace visp
