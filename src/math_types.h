#pragma once
/// Minimal math types for Vultorch (camera, transforms).
/// No external dependency – header-only, float32 only.

#include <cmath>
#include <cstring>

namespace vultorch {

// ── vec3 ──────────────────────────────────────────────────────────────
struct vec3 {
    float x, y, z;
    vec3() : x(0), y(0), z(0) {}
    vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    vec3 operator+(const vec3& b) const { return {x + b.x, y + b.y, z + b.z}; }
    vec3 operator-(const vec3& b) const { return {x - b.x, y - b.y, z - b.z}; }
    vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
    vec3 operator-() const { return {-x, -y, -z}; }

    float length() const { return std::sqrt(x * x + y * y + z * z); }
    vec3  normalized() const { float l = length(); return l > 1e-8f ? vec3{x / l, y / l, z / l} : vec3{0, 0, 0}; }
};

inline float dot(vec3 a, vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline vec3 cross(vec3 a, vec3 b) {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

// ── mat4 (column-major, matches GLSL / Vulkan) ──────────────────────
struct mat4 {
    float m[16]; // column-major: m[col*4 + row]

    mat4() { std::memset(m, 0, sizeof(m)); }

    static mat4 identity() {
        mat4 r;
        r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
        return r;
    }

    float& operator()(int row, int col) { return m[col * 4 + row]; }
    float  operator()(int row, int col) const { return m[col * 4 + row]; }

    mat4 operator*(const mat4& b) const {
        mat4 r;
        for (int c = 0; c < 4; c++)
            for (int row = 0; row < 4; row++) {
                float s = 0;
                for (int k = 0; k < 4; k++)
                    s += (*this)(row, k) * b(k, c);
                r(row, c) = s;
            }
        return r;
    }

    /// Perspective projection (Vulkan clip: Y-down, Z ∈ [0,1]).
    static mat4 perspective(float fov_y_rad, float aspect, float near_plane, float far_plane) {
        float t = std::tan(fov_y_rad * 0.5f);
        mat4 r;
        r(0, 0) = 1.0f / (aspect * t);
        r(1, 1) = -1.0f / t;           // Y-down for Vulkan
        r(2, 2) = far_plane / (near_plane - far_plane);
        r(2, 3) = (near_plane * far_plane) / (near_plane - far_plane);
        r(3, 2) = -1.0f;
        return r;
    }

    /// Look-at view matrix (right-handed).
    static mat4 look_at(vec3 eye, vec3 target, vec3 up) {
        vec3 f = (target - eye).normalized();
        vec3 r = cross(f, up).normalized();
        vec3 u = cross(r, f);

        mat4 m = identity();
        m(0, 0) =  r.x; m(0, 1) =  r.y; m(0, 2) =  r.z; m(0, 3) = -dot(r, eye);
        m(1, 0) =  u.x; m(1, 1) =  u.y; m(1, 2) =  u.z; m(1, 3) = -dot(u, eye);
        m(2, 0) = -f.x; m(2, 1) = -f.y; m(2, 2) = -f.z; m(2, 3) =  dot(f, eye);
        return m;
    }
};

constexpr float PI = 3.14159265358979323846f;
inline float radians(float deg) { return deg * (PI / 180.0f); }

} // namespace vultorch
