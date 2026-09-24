#pragma once

#include <cstddef>

// Private SIMD helpers. Only operations the compiler cannot produce from a
// plain `#pragma omp simd` loop live here: plain vector adds/copies are left
// to the auto-vectorizer, which generates code as good as hand-written
// intrinsics for them. Build with -DDMT_SIMD_FORCE_PORTABLE to exercise the
// portable fallback on any machine.
#if !defined(DMT_SIMD_FORCE_PORTABLE) && defined(__AVX512F__)
#include <immintrin.h>
#define DMT_SIMD_AVX512 1
#elif !defined(DMT_SIMD_FORCE_PORTABLE) && defined(__AVX2__)
#include <immintrin.h>
#define DMT_SIMD_AVX2 1
#else
#define DMT_SIMD_PORTABLE 1
#endif

namespace dmt::simd {

/// @brief Active backend name, for diagnostics and benchmark labels.
constexpr const char* backend_name() noexcept {
#if defined(DMT_SIMD_AVX512)
    return "AVX-512";
#elif defined(DMT_SIMD_AVX2)
    return "AVX2";
#else
    return "portable";
#endif
}

/// @brief True when `add_stream_f32` really issues non-temporal stores.
/// Portable builds (including ARM, whose ISA has no cache-bypassing vector
/// store that compilers expose) fall back to ordinary stores.
inline constexpr bool kHasStreamingStores =
#if defined(DMT_SIMD_PORTABLE)
    false;
#else
    true;
#endif

/**
 * @brief out[i] = a[i] + b[i] with non-temporal (cache-bypassing) stores.
 *
 * Streaming stores skip the read-for-ownership of `out` and keep it from
 * evicting cached operands, but the written data then has to come back from
 * DRAM when it is next read. They only pay off when the data written would
 * not have survived in cache until its next use anyway (see
 * FDMTExecConfig::streaming_stores). Results are bit-identical to the plain
 * add. A trailing store fence orders the non-temporal stores before any
 * later access from other threads.
 */
inline void add_stream_f32(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ out,
                           std::size_t count) noexcept {
#if defined(DMT_SIMD_AVX512) || defined(DMT_SIMD_AVX2)
#if defined(DMT_SIMD_AVX512)
    constexpr std::size_t kLanes = 16;
#else
    constexpr std::size_t kLanes = 8;
#endif
    constexpr std::uintptr_t kAlign = kLanes * sizeof(float);
    std::size_t i                   = 0;
    // Scalar head up to the vector-aligned output address (a float pointer
    // is 4-byte aligned, so this always terminates within kLanes steps).
    while (i < count &&
           (reinterpret_cast<std::uintptr_t>(out + i) % kAlign) != 0) {
        out[i] = a[i] + b[i];
        ++i;
    }
    const std::size_t vec_end = i + (((count - i) / kLanes) * kLanes);
    for (; i < vec_end; i += kLanes) {
#if defined(DMT_SIMD_AVX512)
        _mm512_stream_ps(out + i, _mm512_add_ps(_mm512_loadu_ps(a + i),
                                                _mm512_loadu_ps(b + i)));
#else
        _mm256_stream_ps(out + i, _mm256_add_ps(_mm256_loadu_ps(a + i),
                                                _mm256_loadu_ps(b + i)));
#endif
    }
    for (; i < count; ++i) {
        out[i] = a[i] + b[i];
    }
    _mm_sfence();
#else
#if defined(_OPENMP)
#pragma omp simd
#endif
    for (std::size_t i = 0; i < count; ++i) {
        out[i] = a[i] + b[i];
    }
#endif
}

} // namespace dmt::simd
