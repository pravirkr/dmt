#pragma once

#include <cstddef>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include "dmt/common/types.hpp"

/**
 * @file fdmt_fft_cpu_kernels.hpp
 * @brief Complex tile kernels of the Fourier-domain CPU engines.
 *
 * A tile holds K consecutive frequency bins of one tree node (or one DM
 * trial) in split layout: K real parts followed by K imaginary parts
 * (`float[2 * K]`). All kernels are fixed-K loops the compiler turns into a
 * few whole-register operations (`#pragma omp simd`); FDMTFFT uses K = 16
 * (one AVX-512 register per component, two AVX2, four NEON), DDMTFFT K =
 * 512 (brute force) or 16 (NUFFT). Only the interleaved <-> split
 * conversions of the FFTW rows use intrinsics explicitly (AVX2 with
 * non-temporal stores, NEON ld2/st2), with portable fallbacks.
 */

namespace dmt::algorithms::fft_cpu {

/// out = x * w (split layout).
template <int K>
inline void cmul(float* __restrict__ out,
                 const float* __restrict__ x,
                 const float* __restrict__ w) noexcept {
#pragma omp simd
    for (int j = 0; j < K; ++j) {
        const float xr = x[j];
        const float xi = x[K + j];
        const float wr = w[j];
        const float wi = w[K + j];
        out[j]         = (xr * wr) - (xi * wi);
        out[K + j]     = (xr * wi) + (xi * wr);
    }
}

/// out = tail + head * p (split layout): one FDMT merge node.
template <int K>
inline void cmadd(float* __restrict__ out,
                  const float* __restrict__ tail,
                  const float* __restrict__ head,
                  const float* __restrict__ p) noexcept {
#pragma omp simd
    for (int j = 0; j < K; ++j) {
        const float hr = head[j];
        const float hi = head[K + j];
        const float pr = p[j];
        const float pi = p[K + j];
        out[j]         = tail[j] + ((hr * pr) - (hi * pi));
        out[K + j]     = tail[K + j] + ((hr * pi) + (hi * pr));
    }
}

/// out = x_t * w_t + (x_h * w_h) * p: a level-1 merge with both level-0
/// operands formed on the fly from the channel spectra.
template <int K>
inline void cmadd_level1(float* __restrict__ out,
                         const float* __restrict__ xt,
                         const float* __restrict__ wt,
                         const float* __restrict__ xh,
                         const float* __restrict__ wh,
                         const float* __restrict__ p) noexcept {
#pragma omp simd
    for (int j = 0; j < K; ++j) {
        const float tr = (xt[j] * wt[j]) - (xt[K + j] * wt[K + j]);
        const float ti = (xt[j] * wt[K + j]) + (xt[K + j] * wt[j]);
        const float hr = (xh[j] * wh[j]) - (xh[K + j] * wh[K + j]);
        const float hi = (xh[j] * wh[K + j]) + (xh[K + j] * wh[j]);
        const float pr = p[j];
        const float pi = p[K + j];
        out[j]         = tr + ((hr * pr) - (hi * pi));
        out[K + j]     = ti + ((hr * pi) + (hi * pr));
    }
}

/// out = broadcast(b) * r (split layout), b one complex scalar.
template <int K>
inline void cscale(float* __restrict__ out,
                   ComplexType b,
                   const float* __restrict__ r) noexcept {
    const float br = b.real();
    const float bi = b.imag();
#pragma omp simd
    for (int j = 0; j < K; ++j) {
        out[j]     = (br * r[j]) - (bi * r[K + j]);
        out[K + j] = (br * r[K + j]) + (bi * r[j]);
    }
}

/// Split -> interleaved (FFTW): K complex values to @p dst, with
/// non-temporal stores where the ISA has them: for write-once rows that are
/// read back only after the whole pass (skips the read-for-ownership of the
/// destination line). @p dst must be
/// 64-byte aligned when 8 * K is a multiple of 64. Call stream_fence()
/// before the data is read by another thread.
template <int K>
inline void stream_interleaved(ComplexType* __restrict__ dst,
                               const float* __restrict__ in) noexcept {
#if defined(__AVX2__)
    if constexpr (K % 8 == 0) {
        auto* f = reinterpret_cast<float*>(dst);
        for (int j = 0; j < K; j += 8) {
            const __m256 re = _mm256_loadu_ps(in + j);
            const __m256 im = _mm256_loadu_ps(in + K + j);
            const __m256 lo = _mm256_unpacklo_ps(re, im); // 0 1 4 5 (pairs)
            const __m256 hi = _mm256_unpackhi_ps(re, im); // 2 3 6 7
            _mm256_stream_ps(f + (2 * j), _mm256_permute2f128_ps(lo, hi, 0x20));
            _mm256_stream_ps(f + (2 * j) + 8,
                             _mm256_permute2f128_ps(lo, hi, 0x31));
        }
        return;
    }
#elif defined(__ARM_NEON)
    if constexpr (K % 4 == 0) {
        auto* f = reinterpret_cast<float*>(dst);
        for (int j = 0; j < K; j += 4) {
            const float32x4x2_t v{vld1q_f32(in + j), vld1q_f32(in + K + j)};
            vst2q_f32(f + (2 * j), v);
        }
        return;
    }
#endif
    auto* f = reinterpret_cast<float*>(dst);
    for (int j = 0; j < K; ++j) {
        f[2 * j]       = in[j];
        f[(2 * j) + 1] = in[K + j];
    }
}

/// Interleaved -> split with non-temporal stores: K complex values of
/// @p src to dst[0..K) (real) and dst[K..2K) (imaginary). @p dst must be
/// 32-byte aligned.
template <int K>
inline void stream_split(float* __restrict__ dst,
                         const ComplexType* __restrict__ src) noexcept {
    const auto* f = reinterpret_cast<const float*>(src);
#if defined(__AVX2__)
    if constexpr (K % 8 == 0) {
        for (int j = 0; j < K; j += 8) {
            const __m256 a = _mm256_loadu_ps(f + (2 * j));     // c0..c3
            const __m256 b = _mm256_loadu_ps(f + (2 * j) + 8); // c4..c7
            // shuffle within 128-bit lanes, then fix the lane order.
            const __m256 re = _mm256_shuffle_ps(a, b, 0x88);
            const __m256 im = _mm256_shuffle_ps(a, b, 0xDD);
            _mm256_stream_ps(dst + j, _mm256_castpd_ps(_mm256_permute4x64_pd(
                                          _mm256_castps_pd(re), 0xD8)));
            _mm256_stream_ps(dst + K + j,
                             _mm256_castpd_ps(_mm256_permute4x64_pd(
                                 _mm256_castps_pd(im), 0xD8)));
        }
        return;
    }
#elif defined(__ARM_NEON)
    if constexpr (K % 4 == 0) {
        for (int j = 0; j < K; j += 4) {
            const float32x4x2_t v = vld2q_f32(f + (2 * j));
            vst1q_f32(dst + j, v.val[0]);
            vst1q_f32(dst + K + j, v.val[1]);
        }
        return;
    }
#endif
    for (int j = 0; j < K; ++j) {
        dst[j]     = f[2 * j];
        dst[K + j] = f[(2 * j) + 1];
    }
}

inline void stream_fence() noexcept {
#if defined(__AVX2__)
    _mm_sfence();
#endif
}

/// out = in (split tile copy).
template <int K>
inline void copy_tile(float* __restrict__ out,
                      const float* __restrict__ in) noexcept {
#pragma omp simd
    for (int j = 0; j < 2 * K; ++j) {
        out[j] = in[j];
    }
}

} // namespace dmt::algorithms::fft_cpu
