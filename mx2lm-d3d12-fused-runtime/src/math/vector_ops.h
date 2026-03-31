#pragma once
#include <DirectXMath.h>

namespace mx2lm {

using namespace DirectX;

inline XMFLOAT4 operator+(const XMFLOAT4& a, const XMFLOAT4& b) {
    return {a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w};
}

inline XMFLOAT4 operator*(const XMFLOAT4& a, float s) {
    return {a.x*s, a.y*s, a.z*s, a.w*s};
}

inline XMFLOAT3 operator+(const XMFLOAT3& a, const XMFLOAT3& b) {
    return {a.x+b.x, a.y+b.y, a.z+b.z};
}

inline XMFLOAT3 operator*(const XMFLOAT3& a, float s) {
    return {a.x*s, a.y*s, a.z*s};
}

inline float Dot(const XMFLOAT4& a, const XMFLOAT4& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
}

inline float Length(const XMFLOAT3& v) {
    return std::sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
}

} // namespace mx2lm
