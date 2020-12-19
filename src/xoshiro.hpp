#pragma once

#include <array>

namespace xoshiro {

// Use public domain xoroshiro128** PRNG implementation as it is
// faster than Mersenne twister: http://prng.di.unimi.it/xoroshiro128starstar.c
// Seeding is taken care of using a separate public doamin generator:
// splitmix64 http://prng.di.unimi.it/splitmix64.c
using State = std::array<uint64_t, 2>;

inline std::uint64_t rotl(std::uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

uint64_t splitmix64(uint64_t z) {
	z += 0x9e3779b97f4a7c15;
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
	return z ^ (z >> 31);
}

// Seed the generator using splitmix64:
void seed(State &s, uint64_t seed) {
	s[0] = splitmix64(seed);
	s[1] = splitmix64(s[0]);
}

inline uint64_t next128ss(State &s) {
	const uint64_t s0 = s[0];
	uint64_t s1 = s[1];
	const uint64_t result = rotl(s0 * 5, 7) * 9;

	s1 ^= s0;
	s[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16); // a, b
	s[1] = rotl(s1, 37); // c

	return result;
}

// This is the jump function for the generator. It is equivalent
// to 2^64 calls to next128ss(); it can be used to generate
// 2^64 non-overlapping subsequences for parallel computations.
void jump(State &s) {
	static const uint64_t JUMP[] = { 0xdf900294d8f554a5, 0x170865df4b3201fc };

	uint64_t s0 = 0;
	uint64_t s1 = 0;
	for(std::size_t i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
		for(int b = 0; b < 64; b++) {
			if (JUMP[i] & UINT64_C(1) << b) {
				s0 ^= s[0];
				s1 ^= s[1];
			}
			next128ss(s);
		}

	s[0] = s0;
	s[1] = s1;
}

inline double to_double(uint64_t x) {
	const union { uint64_t i; double d; } u = { .i = UINT64_C(0x3FF) << 52 | x >> 12 };
	return u.d - 1.0;
}

// Uniform [-1..1)
inline float uniform_neg1_1(State& rngState) {
	return 2.0*to_double(next128ss(rngState)) - 1.0;
}

// Uniform [0..1)
inline float uniform_0_1(State& rngState) {
	return to_double(next128ss(rngState));
}

} // end of namespace xorshiro
