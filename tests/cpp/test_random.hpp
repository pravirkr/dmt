#pragma once

// Fast random fills for test inputs.
//
// The tests feed millions of samples per block, and the C++ suite also runs
// unoptimized (-O0 coverage, -O1 sanitizers), where a std::*_distribution
// call per element is several non-inlined calls deep. These fills keep the
// generator (splitmix64) in the loop body: a handful of integer operations
// per element whatever the optimization level, and the same values on every
// standard library (std::uniform_int_distribution and friends are
// implementation-defined, so their values already differ between libstdc++
// and libc++).

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace dmt::test_random {

/// splitmix64 (Steele, Lea, Flood 2014): one 64-bit output per call.
struct SplitMix64 {
    uint64_t state;
    explicit SplitMix64(uint64_t seed) : state(seed) {}
    uint64_t operator()() {
        uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
        z          = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
        z          = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31U);
    }
};

/// Uniform integers in [lo, hi] (inclusive), converted to T. Spans of a
/// power of two use the low bits directly; others the multiply-shift range
/// reduction (bias below 2^-32 for 32-bit spans).
template <typename T>
void fill_uniform_int(std::span<T> out, uint64_t seed, int64_t lo, int64_t hi) {
    const auto span = static_cast<uint64_t>(hi - lo) + 1;
    SplitMix64 gen(seed);
    const bool pow2 = (span & (span - 1)) == 0;
    if (pow2 && span <= 256) {
        // Eight draws per generator call.
        const uint64_t mask = span - 1;
        const auto bits     = static_cast<unsigned>(std::bit_width(mask));
        const unsigned per  = bits == 0 ? 64 : 64 / bits;
        uint64_t word       = 0;
        unsigned left       = 0;
        for (auto& v : out) {
            if (left == 0) {
                word = gen();
                left = per;
            }
            v = static_cast<T>(lo + static_cast<int64_t>(word & mask));
            word >>= bits;
            --left;
        }
        return;
    }
    for (auto& v : out) {
        const uint64_t r = gen() >> 32U;
        v = static_cast<T>(lo + static_cast<int64_t>((r * span) >> 32U));
    }
}

template <typename T>
std::vector<T>
uniform_int(std::size_t n, uint64_t seed, int64_t lo, int64_t hi) {
    std::vector<T> v(n);
    fill_uniform_int<T>(v, seed, lo, hi);
    return v;
}

/// Uniform reals in [lo, hi): 24 (float) or 53 (double) random bits.
template <typename T>
void fill_uniform_real(std::span<T> out, uint64_t seed, T lo, T hi) {
    SplitMix64 gen(seed);
    const double scale = static_cast<double>(hi - lo);
    for (auto& v : out) {
        const double u =
            static_cast<double>(gen() >> 11U) * 0x1.0p-53; // [0, 1)
        v = static_cast<T>(static_cast<double>(lo) + (u * scale));
    }
}

template <typename T>
std::vector<T> uniform_real(std::size_t n, uint64_t seed, T lo, T hi) {
    std::vector<T> v(n);
    fill_uniform_real<T>(v, seed, lo, hi);
    return v;
}

/// Normal deviates (Box-Muller, both outputs used).
template <typename T>
void fill_normal(std::span<T> out, uint64_t seed, T mean = 0, T sigma = 1) {
    SplitMix64 gen(seed);
    constexpr double kTwoPi = 6.283185307179586;
    for (std::size_t i = 0; i < out.size(); i += 2) {
        // u1 in (0, 1] so the log is finite.
        const double u1 = (static_cast<double>(gen() >> 11U) + 1.0) * 0x1.0p-53;
        const double u2 = static_cast<double>(gen() >> 11U) * 0x1.0p-53;
        const double r  = std::sqrt(-2.0 * std::log(u1));
        const double a  = kTwoPi * u2;
        out[i] = static_cast<T>(static_cast<double>(mean) +
                                (static_cast<double>(sigma) * r * std::cos(a)));
        if (i + 1 < out.size()) {
            out[i + 1] =
                static_cast<T>(static_cast<double>(mean) +
                               (static_cast<double>(sigma) * r * std::sin(a)));
        }
    }
}

template <typename T>
std::vector<T> normal(std::size_t n, uint64_t seed, T mean = 0, T sigma = 1) {
    std::vector<T> v(n);
    fill_normal<T>(v, seed, mean, sigma);
    return v;
}

} // namespace dmt::test_random
