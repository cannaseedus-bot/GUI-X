#pragma once
#include <DirectXMath.h>
#include <cmath>

namespace mx2lm {

using namespace DirectX;

// Safe normalize — returns (0,0,1) when length < epsilon
inline XMFLOAT3 SafeNormalize(const XMFLOAT3& v) {
    float len = std::sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
    if (len < 1e-5f) return {0.f, 0.f, 1.f};
    return {v.x/len, v.y/len, v.z/len};
}

inline XMFLOAT3 SafeNormalize(float x, float y, float z) {
    return SafeNormalize({x, y, z});
}

// Guard NaN/Inf in float
inline float GuardNaN(float v) {
    return (std::isnan(v) || std::isinf(v)) ? 0.f : v;
}

inline XMFLOAT4 GuardNaN4(const XMFLOAT4& v) {
    return {GuardNaN(v.x), GuardNaN(v.y), GuardNaN(v.z), GuardNaN(v.w)};
}

// Clamp float3
inline XMFLOAT3 Clamp3(const XMFLOAT3& v, float lo, float hi) {
    return {
        std::fmaxf(lo, std::fminf(hi, v.x)),
        std::fmaxf(lo, std::fminf(hi, v.y)),
        std::fmaxf(lo, std::fminf(hi, v.z))
    };
}

// 3D Morton code (for spatial sort)
inline uint32_t MortonCode3(uint32_t x, uint32_t y, uint32_t z) {
    // Expand 10-bit components to 30 bits
    auto expand = [](uint32_t v) -> uint32_t {
        v &= 0x000003FF;
        v = (v | (v << 16)) & 0xFF0000FF;
        v = (v | (v << 8))  & 0x0F00F00F;
        v = (v | (v << 4))  & 0xC30C30C3;
        v = (v | (v << 2))  & 0x49249249;
        return v;
    };
    return expand(x) | (expand(y) << 1) | (expand(z) << 2);
}

} // namespace mx2lm
