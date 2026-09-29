#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

/**
 * @file ddmt_cpu_kernels.hpp
 * @brief Delayed-sum kernels of the DDMT CPU engine.
 *
 * One kernel family, `dst[s] = (ACC ? dst[s] : 0) + p_0[s] + ... +
 * p_{K-1}[s]` for s in [0, n), K in 1..16, over float, uint16_t or uint32_t
 * lanes. Explicit AVX-512 (float, uint32_t; uint16_t needs AVX-512BW) and AVX2
 * paths are chosen at compile time, with an `omp simd` fallback.
 *
 * unpack_bytes_u16() widens whole bytes of 1/2/4-bit samples (LSB first) to
 * uint16_t lanes for the staged windows.
 *
 * Every buffer the engine passes has at least kPad elements of readable (and,
 * for @p dst, writable) slack past @p n: the vector paths round n up to whole
 * vectors instead of handling a tail. Lanes past n hold unspecified values.
 */

namespace dmt::algorithms::ddmt_cpu {

/// Slack (elements) the caller guarantees past n on every buffer.
inline constexpr int kPad = 64;

namespace detail {

#if defined(__AVX512F__)
template <typename T> struct Vec;
template <> struct Vec<float> {
    using R                  = __m512;
    static constexpr int kLn = 16;
    static R zero() { return _mm512_setzero_ps(); }
    static R load(const float* p) { return _mm512_loadu_ps(p); }
    static void store(float* p, R v) { _mm512_storeu_ps(p, v); }
    static R add(R a, R b) { return _mm512_add_ps(a, b); }
};
template <> struct Vec<uint32_t> {
    using R                  = __m512i;
    static constexpr int kLn = 16;
    static R zero() { return _mm512_setzero_si512(); }
    static R load(const uint32_t* p) { return _mm512_loadu_si512(p); }
    static void store(uint32_t* p, R v) { _mm512_storeu_si512(p, v); }
    static R add(R a, R b) { return _mm512_add_epi32(a, b); }
};
#if defined(__AVX512BW__)
template <> struct Vec<uint16_t> {
    using R                  = __m512i;
    static constexpr int kLn = 32;
    static R zero() { return _mm512_setzero_si512(); }
    static R load(const uint16_t* p) { return _mm512_loadu_si512(p); }
    static void store(uint16_t* p, R v) { _mm512_storeu_si512(p, v); }
    static R add(R a, R b) { return _mm512_add_epi16(a, b); }
};
#endif
#elif defined(__AVX2__)
template <typename T> struct Vec;
template <> struct Vec<float> {
    using R                  = __m256;
    static constexpr int kLn = 8;
    static R zero() { return _mm256_setzero_ps(); }
    static R load(const float* p) { return _mm256_loadu_ps(p); }
    static void store(float* p, R v) { _mm256_storeu_ps(p, v); }
    static R add(R a, R b) { return _mm256_add_ps(a, b); }
};
template <> struct Vec<uint32_t> {
    using R                  = __m256i;
    static constexpr int kLn = 8;
    static R zero() { return _mm256_setzero_si256(); }
    static R load(const uint32_t* p) {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    }
    static void store(uint32_t* p, R v) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);
    }
    static R add(R a, R b) { return _mm256_add_epi32(a, b); }
};
template <> struct Vec<uint16_t> {
    using R                  = __m256i;
    static constexpr int kLn = 16;
    static R zero() { return _mm256_setzero_si256(); }
    static R load(const uint16_t* p) {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    }
    static void store(uint16_t* p, R v) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);
    }
    static R add(R a, R b) { return _mm256_add_epi16(a, b); }
};
#endif

template <typename T, typename = void> struct HasVec : std::false_type {};
#if defined(__AVX512F__) || defined(__AVX2__)
template <typename T>
struct HasVec<T, std::void_t<decltype(Vec<T>::kLn)>> : std::true_type {};
#endif

template <typename T, int K, bool ACC>
inline void sum_k(T* __restrict__ dst, const T* const* p, int n) {
#if defined(__AVX512F__) || defined(__AVX2__)
    if constexpr (HasVec<T>::value) {
        using V           = Vec<T>;
        constexpr int kLn = V::kLn;
        const T* __restrict__ q[K];
        for (int k = 0; k < K; ++k) {
            q[k] = p[k];
        }
        for (int s = 0; s < n; s += kLn) {
            typename V::R x0 = V::load(q[0] + s);
            typename V::R x1 = K > 1 ? V::load(q[1] + s) : V::zero();
#pragma GCC unroll 16
            for (int k = 2; k < K; ++k) {
                if (k % 2 == 0) {
                    x0 = V::add(x0, V::load(q[k] + s));
                } else {
                    x1 = V::add(x1, V::load(q[k] + s));
                }
            }
            if constexpr (K > 1) {
                x0 = V::add(x0, x1);
            }
            if constexpr (ACC) {
                x0 = V::add(V::load(dst + s), x0);
            }
            V::store(dst + s, x0);
        }
        return;
    }
#endif
    const T* __restrict__ q[K];
    for (int k = 0; k < K; ++k) {
        q[k] = p[k];
    }
#pragma omp simd
    for (int s = 0; s < n; ++s) {
        T v = ACC ? dst[s] : T{0};
        for (int k = 0; k < K; ++k) {
            v = static_cast<T>(v + q[k][s]);
        }
        dst[s] = v;
    }
}

/// Jump table over K = 1..16; only K <= KMax is instantiated.
template <typename T, bool ACC, int KMax>
inline void sum_dispatch(T* dst, const T* const* p, int k, int n) {
    switch (k) {
    case 1:
        if constexpr (1 <= KMax) {
            sum_k<T, 1, ACC>(dst, p, n);
        }
        break;
    case 2:
        if constexpr (2 <= KMax) {
            sum_k<T, 2, ACC>(dst, p, n);
        }
        break;
    case 3:
        if constexpr (3 <= KMax) {
            sum_k<T, 3, ACC>(dst, p, n);
        }
        break;
    case 4:
        if constexpr (4 <= KMax) {
            sum_k<T, 4, ACC>(dst, p, n);
        }
        break;
    case 5:
        if constexpr (5 <= KMax) {
            sum_k<T, 5, ACC>(dst, p, n);
        }
        break;
    case 6:
        if constexpr (6 <= KMax) {
            sum_k<T, 6, ACC>(dst, p, n);
        }
        break;
    case 7:
        if constexpr (7 <= KMax) {
            sum_k<T, 7, ACC>(dst, p, n);
        }
        break;
    case 8:
        if constexpr (8 <= KMax) {
            sum_k<T, 8, ACC>(dst, p, n);
        }
        break;
    case 9:
        if constexpr (9 <= KMax) {
            sum_k<T, 9, ACC>(dst, p, n);
        }
        break;
    case 10:
        if constexpr (10 <= KMax) {
            sum_k<T, 10, ACC>(dst, p, n);
        }
        break;
    case 11:
        if constexpr (11 <= KMax) {
            sum_k<T, 11, ACC>(dst, p, n);
        }
        break;
    case 12:
        if constexpr (12 <= KMax) {
            sum_k<T, 12, ACC>(dst, p, n);
        }
        break;
    case 13:
        if constexpr (13 <= KMax) {
            sum_k<T, 13, ACC>(dst, p, n);
        }
        break;
    case 14:
        if constexpr (14 <= KMax) {
            sum_k<T, 14, ACC>(dst, p, n);
        }
        break;
    case 15:
        if constexpr (15 <= KMax) {
            sum_k<T, 15, ACC>(dst, p, n);
        }
        break;
    case 16:
        if constexpr (16 <= KMax) {
            sum_k<T, 16, ACC>(dst, p, n);
        }
        break;
    default:
        break;
    }
}

} // namespace detail

/// Maximum number of sources summed by one sum_into() call.
inline constexpr int kMaxSources = 16;

/**
 * @brief dst[s] = (accumulate ? dst[s] : 0) + sum_k p[k][s] for s < n, with
 * 1 <= k <= KMax sources (KMax <= kMaxSources: the size of the caller's
 * pointer array).
 */
template <int KMax, typename T>
inline void sum_into(T* dst, const T* const* p, int k, int n, bool accumulate) {
    static_assert(KMax >= 1 && KMax <= kMaxSources);
    if (accumulate) {
        detail::sum_dispatch<T, true, KMax>(dst, p, k, n);
    } else {
        detail::sum_dispatch<T, false, KMax>(dst, p, k, n);
    }
}

namespace detail {

template <unsigned NBITS> struct ByteLut {
    static constexpr unsigned kPer = 8 / NBITS;
    static constexpr auto kTable   = [] {
        std::array<std::array<uint16_t, kPer>, 256> t{};
        for (unsigned b = 0; b < 256; ++b) {
            for (unsigned k = 0; k < kPer; ++k) {
                t[b][k] = static_cast<uint16_t>((b >> (k * NBITS)) &
                                                ((1U << NBITS) - 1U));
            }
        }
        return t;
    }();
};

} // namespace detail

/**
 * @brief Unpacks @p nbytes whole bytes of NBITS-bit samples (NBITS = 1, 2
 * or 4, LSB first) into 8 / NBITS * nbytes uint16_t samples at @p out.
 */
template <unsigned NBITS>
inline void
unpack_bytes_u16(const uint8_t* in, std::size_t nbytes, uint16_t* out) {
    static_assert(NBITS == 1 || NBITS == 2 || NBITS == 4);
    constexpr std::size_t kPer = 8 / NBITS;
    std::size_t b              = 0;
#if defined(__AVX512BW__)
    if constexpr (NBITS == 4) {
        const __m512i lo = _mm512_set1_epi32(0x0F);
        for (; b + 16 <= nbytes; b += 16) {
            const __m512i x = _mm512_cvtepu8_epi32(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + b)));
            const __m512i w =
                _mm512_or_si512(_mm512_and_si512(x, lo),
                                _mm512_slli_epi32(_mm512_srli_epi32(x, 4), 16));
            _mm512_storeu_si512(out + (b * kPer), w);
        }
    } else if constexpr (NBITS == 2) {
        const __m512i m = _mm512_set1_epi64(3);
        for (; b + 8 <= nbytes; b += 8) {
            const __m512i x = _mm512_cvtepu8_epi64(
                _mm_loadl_epi64(reinterpret_cast<const __m128i*>(in + b)));
            __m512i w = _mm512_and_si512(x, m);
            w         = _mm512_or_si512(
                w, _mm512_slli_epi64(
                       _mm512_and_si512(_mm512_srli_epi64(x, 2), m), 16));
            w = _mm512_or_si512(
                w, _mm512_slli_epi64(
                       _mm512_and_si512(_mm512_srli_epi64(x, 4), m), 32));
            w = _mm512_or_si512(w,
                                _mm512_slli_epi64(_mm512_srli_epi64(x, 6), 48));
            _mm512_storeu_si512(out + (b * kPer), w);
        }
    } else {
        const __m512i one = _mm512_set1_epi16(1);
        for (; b + 4 <= nbytes; b += 4) {
            uint32_t m = 0;
            std::memcpy(&m, in + b, sizeof(m));
            _mm512_storeu_si512(out + (b * kPer),
                                _mm512_maskz_mov_epi16(m, one));
        }
    }
#elif defined(__AVX2__)
    if constexpr (NBITS == 4) {
        const __m256i lo = _mm256_set1_epi32(0x0F);
        for (; b + 8 <= nbytes; b += 8) {
            const __m256i x = _mm256_cvtepu8_epi32(
                _mm_loadl_epi64(reinterpret_cast<const __m128i*>(in + b)));
            const __m256i w =
                _mm256_or_si256(_mm256_and_si256(x, lo),
                                _mm256_slli_epi32(_mm256_srli_epi32(x, 4), 16));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + (b * kPer)),
                                w);
        }
    } else if constexpr (NBITS == 2) {
        const __m256i m = _mm256_set1_epi64x(3);
        for (; b + 4 <= nbytes; b += 4) {
            int32_t v = 0;
            std::memcpy(&v, in + b, sizeof(v));
            const __m256i x = _mm256_cvtepu8_epi64(_mm_cvtsi32_si128(v));
            __m256i w       = _mm256_and_si256(x, m);
            w               = _mm256_or_si256(
                w, _mm256_slli_epi64(
                       _mm256_and_si256(_mm256_srli_epi64(x, 2), m), 16));
            w = _mm256_or_si256(
                w, _mm256_slli_epi64(
                       _mm256_and_si256(_mm256_srli_epi64(x, 4), m), 32));
            w = _mm256_or_si256(w,
                                _mm256_slli_epi64(_mm256_srli_epi64(x, 6), 48));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + (b * kPer)),
                                w);
        }
    }
#endif
    const auto& lut = detail::ByteLut<NBITS>::kTable;
    for (; b < nbytes; ++b) {
        std::memcpy(out + (b * kPer), lut[in[b]].data(),
                    kPer * sizeof(uint16_t));
    }
}

} // namespace dmt::algorithms::ddmt_cpu
