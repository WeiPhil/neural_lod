// Copyright 2023 Intel Corporation.
// SPDX-License-Identifier: MIT

#ifndef UTIL_GLSL
#define UTIL_GLSL

#include "language.glsl"
#include "defaults.glsl"

#ifndef FLT_EPSILON
#define FLT_EPSILON 1.192092896e-07
#endif

#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38F
#endif

#define INV_LN10 0.4342944819

inline float positive_pow(float base, float power)
{
    return pow(max(abs(base), float(FLT_EPSILON)), power);
}

// https://en.wikipedia.org/wiki/SRGB Linear to SRGB and SRGB to Linear
inline float linear_to_srgb(float x)
{
  return (x <= 0.0031308f) ? 12.92f * x : 1.055f * positive_pow(x, 1.f / 2.4f) - 0.055f;
}

inline vec3 linear_to_srgb(vec3 c)
{
  vec3 r;
  r.x = (c.x <= 0.0031308f) ? 12.92f * c.x : 1.055f * positive_pow(c.x, 1.f / 2.4f) - 0.055f;
  r.y = (c.y <= 0.0031308f) ? 12.92f * c.y : 1.055f * positive_pow(c.y, 1.f / 2.4f) - 0.055f;
  r.z = (c.z <= 0.0031308f) ? 12.92f * c.z : 1.055f * positive_pow(c.z, 1.f / 2.4f) - 0.055f;
  return r;
}

inline float srgb_to_linear(float x)
{
  return (x <= 0.04045) ? x / 12.92 : positive_pow((x + 0.055) / 1.055, 2.4);
}

inline vec3 srgb_to_linear(vec3 c)
{
  vec3 r;
  r.x = (c.x <= 0.04045) ? c.x / 12.92 : positive_pow((c.x + 0.055) / 1.055, 2.4);
  r.y = (c.y <= 0.04045) ? c.y / 12.92 : positive_pow((c.y + 0.055) / 1.055, 2.4);
  r.z = (c.z <= 0.04045) ? c.z / 12.92 : positive_pow((c.z + 0.055) / 1.055, 2.4);
  return r;
}

inline vec3 logc_to_linear(vec3 x)
{
  return (pow(vec3(10.0f), (x - vec3(0.386036)) / vec3(0.244161)) - vec3(0.047996)) / vec3(5.555556);
}

inline vec3 linear_to_logc(vec3 x)
{
  return vec3(0.244161) * log(max(vec3(5.555556) * x + vec3(0.047996), vec3(0.0))) * vec3(INV_LN10) + vec3(0.386036);
}

inline vec2 sample_tent(vec2 random) {
	random -= vec2(0.5f);
	vec2 off_center = vec2(1.0f) - sqrt(max(vec2(1.0f) - abs(random) * 2.0f, 0.0f));
    return sign(random) * off_center;
}
inline float pdf_tent(vec2 offset) {
	vec2 pdfs_1d = vec2(1.0f) - abs(offset);
	return pdfs_1d.x * pdfs_1d.y;
}

inline void ortho_basis(GLSL_out(vec3) v_x, GLSL_out(vec3) v_y, const vec3 n) {
	v_y = vec3(0, 0, 0);

	if (n.x < 0.6f && n.x > -0.6f) {
		v_y.x = 1.f;
	} else if (n.y < 0.6f && n.y > -0.6f) {
		v_y.y = 1.f;
	} else if (n.z < 0.6f && n.z > -0.6f) {
		v_y.z = 1.f;
	} else {
		v_y.x = 1.f;
	}
	v_x = normalize(cross(v_y, n));
	v_y = normalize(cross(n, v_x));
}

inline mat3 ortho_frame(const vec3 n) {
	vec3 v_x, v_y;
	ortho_basis(v_x, v_y, n);
	return mat3(v_x, v_y, n);
}

inline float luminance(const vec3 c) {
	return 0.2126f * c.r + 0.7152f * c.g + 0.0722f * c.b;
}

inline float pow2(float x) {
	return x * x;
}

inline float length2(vec2 x) {
	return dot(x, x);
}
inline float length2(vec3 x) {
	return dot(x, x);
}
inline float length2(vec4 x) {
	return dot(x, x);
}

inline float cos_half_angle(float cos_angle) {
    return (1.0f + cos_angle) / sqrt(2.0f + 2.0f * cos_angle);
}

inline float power_heuristic(float n_f, float pdf_f, float n_g, float pdf_g) {
	float f = n_f * pdf_f;
	float g = n_g * pdf_g;
	return (f * f) / (f * f + g * g);
}

inline vec4 quat_mul(vec4 p, vec4 q)
{
	vec4 r;
	r.w = p.w * q.w - p.x * q.x - p.y * q.y - p.z * q.z;
	r.x = p.w * q.x + p.x * q.w + p.y * q.z - p.z * q.y;
	r.y = p.w * q.y + p.y * q.w + p.z * q.x - p.x * q.z;
	r.z = p.w * q.z + p.z * q.w + p.x * q.y - p.y * q.x;
	return r;
}

inline vec3 quat_rot(vec4 q, vec3 p)
{
	return vec3( quat_mul(quat_mul(q, vec4(p, 0.0f)), vec4(-q.x, -q.y, -q.z, q.w)) );
}

inline vec4 quat_copysign(vec4 q, float sgn) {
	if (q.w * sgn < 0.0f)
		return -q;
	if (q.w == 0.0f)
		q.w = sgn * 1.175494351e-38f;
	return q;
}

/*! An implementation of mix() using two fused-multiply add instructions. Used
	because the native mix() implementation had stability issues in a few
	spots. Credit to Fabian Giessen's blog, see:
	https://fgiesen.wordpress.com/2012/08/15/linear-interpolation-past-present-and-future/
	*/
inline float mix_fma(float x, float y, float a) {
	return fma(a, y, fma(-a, x, x));
}

// https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
inline uint32_t Part1By1(uint32_t x) {
	x &= 0x0000ffff;                  // x = ---- ---- ---- ---- fedc ba98 7654 3210
	x = (x ^ (x <<  8)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
	x = (x ^ (x <<  4)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
	x = (x ^ (x <<  2)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
	x = (x ^ (x <<  1)) & 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
	return x;
}
inline uint32_t Part1By2(uint32_t x) {
  x &= 0x000003ff;                  // x = ---- ---- ---- ---- ---- --98 7654 3210
  x = (x ^ (x << 16)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
  x = (x ^ (x <<  8)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
  x = (x ^ (x <<  4)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
  x = (x ^ (x <<  2)) & 0x09249249; // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
  return x;
}
inline uint32_t encode_morton2(uint32_t x, uint32_t y) {
	return (Part1By1(y) << 1) + Part1By1(x);
}
inline uint32_t encode_morton3(uint32_t x, uint32_t y, uint32_t z) {
	return (Part1By2(z) << 2) + (Part1By2(y) << 1) + Part1By2(x);
}

inline float min3_(float a, float b, float c)
{
  return min(a, min(b, c));
}

inline float max3_(float a, float b, float c)
{
  return max(a, max(b, c));
}

inline float safe_acos(float x)
{
    if (x < -1.0)
        x = -1.0;
    else if (x > 1.0)
        x = 1.0;
    return acos(x);
}

inline float safe_sqrt(float x)
{
    return sqrt(max(0.f,x));
}


inline vec2 cartesian_to_spherical(vec3 v)
{
    // Setting wo
    float theta = safe_acos(v.z);  // theta

    float phi = 0.0f;
    if (abs(v.x) > 0 || abs(v.y) > 0 ) {
        phi = atan(v.y, v.x);
    } else {
        phi = 0.0;
    }
    return vec2(theta, phi);
}

inline vec3 spherical_to_cartesian(const float theta, const float phi)
{
    return vec3(cos(phi)*sin(theta),sin(phi)*sin(theta),cos(theta));
}

inline vec3 square_to_uniform_sphere(const vec2 rng)
{
    float phi = 2.0f * M_PI * rng.x;
    float cos_theta = rng.y * 2.0f - 1.0f;
    float sin_theta = sqrt(1.0f - cos_theta * cos_theta);
    return vec3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}

inline vec3 square_to_hg(const vec2 rng, const vec3 w_o, float g){
    float cos_theta;
    if(abs(g) < 1e-3){
        cos_theta = 1.0 - 2.0 * rng.x;
    }else{
        float sqr_term = (1.0 - g * g) / (1.0 - g + 2.0 * g * rng.x);
        cos_theta = (1.0 + g * g - sqr_term * sqr_term) / (2.0 * g);
    }

    float sin_theta = sqrt(max(0.f, 1.f - cos_theta * cos_theta));
    float phi = 2.0 * M_PI * rng.y;

    float sin_phi = sin(phi);
    float cos_phi = cos(phi);

    vec3 local = vec3(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);

    vec3 n = -w_o;

    vec3 s;
    if( abs(n.x) > abs(n.y) ){
        float inv_len = 1.f / sqrt(n.x * n.x + n.z * n.z);
        s = vec3(n.z * inv_len, 0.0, -n.x * inv_len);
    } else {
        float inv_len = 1.f / sqrt(n.y * n.y + n.z * n.z);
        s = vec3(0.0, n.z * inv_len, -n.y * inv_len);
    }
    vec3 t = cross(n,s);

    return local.x * s + local.y * t + local.z * n;
}

inline float square_to_hg_pdf(const float wo_dot_wi, const float g){
    float denom = 1.0 + g * g + 2.0 * g * wo_dot_wi;
    return 0.25 * M_1_PI * (1.0 - g * g) / (denom * sqrt(denom));
}

/// Sample a point on a 2D standard normal distribution. Internally uses the Box-Muller
/// transformation
inline vec2 square_to_std_normal(const vec2 rng)
{
    float r = sqrt(-2.f * log(1.f - rng.x));
    float phi = 2.f * M_PI * rng.y;

    float s = sin(phi);
    float c = cos(phi);
    return vec2(c * r, s * r);
}

#endif

