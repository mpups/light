#pragma once

#include <array>

// Use public domain xoroshiro128** PRNG implementation as it is
// faster than Mersenne twister: http://prng.di.unimi.it/xoroshiro128starstar.c
using XoshiroState = std::array<uint64_t, 2>;

inline std::uint64_t rotl(std::uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

inline uint64_t xoshiro128ss(XoshiroState &s) {
	const uint64_t s0 = s[0];
	uint64_t s1 = s[1];
	const uint64_t result = rotl(s0 * 5, 7) * 9;

	s1 ^= s0;
	s[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16); // a, b
	s[1] = rotl(s1, 37); // c

	return result;
}

inline double to_double(uint64_t x) {
	const union { uint64_t i; double d; } u = { .i = UINT64_C(0x3FF) << 52 | x >> 12 };
	return u.d - 1.0;
}

// Uniform [-1..1)
inline float rnd(XoshiroState& rngState) {
	return 2.0*to_double(xoshiro128ss(rngState)) - 1.0;
}

// Uniform [0..1)
inline float rnd2(XoshiroState& rngState) {
	return to_double(xoshiro128ss(rngState));
}
