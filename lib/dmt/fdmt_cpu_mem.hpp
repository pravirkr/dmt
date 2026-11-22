#pragma once

// Large page-backed buffers and non-temporal streaming helpers shared by the
// CPU FDMT engine (tree / fused paths) and the CPU skewed-window engine.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <type_traits>
#include <utility>

#if defined(__linux__)
#include <sys/mman.h>
#endif

#if defined(__AVX512F__) || defined(__AVX__)
#include <immintrin.h>
#endif

#include "dmt/common/types.hpp"

namespace dmt::algorithms::detail {

/**
 * @brief A large zero-initialised array for the CPU FDMT engines' tree state,
 * skewed state and tree history: 2 MiB-aligned, with transparent huge pages
 * requested on Linux, and zero-filled by `nthreads` threads with a static
 * schedule.
 *
 * Huge pages: a fused time tile writes short runs of thousands of rows that
 * lie a chunk apart, one 4 KiB page each, so with small pages nearly every
 * run misses the TLB. The parallel fill first-touches the pages across the
 * threads (and so across NUMA nodes) instead of on the constructing thread.
 */
template <typename T> class FDMTLargeArray {
public:
    FDMTLargeArray() = default;
    ~FDMTLargeArray() { std::free(m_data); } // NOLINT
    FDMTLargeArray(const FDMTLargeArray&)            = delete;
    FDMTLargeArray& operator=(const FDMTLargeArray&) = delete;
    FDMTLargeArray(FDMTLargeArray&&)                 = delete;
    FDMTLargeArray& operator=(FDMTLargeArray&&)      = delete;

    void assign(SizeType n, int nthreads) {
        std::free(m_data); // NOLINT
        m_data = nullptr;
        m_size = n;
        if (n == 0) {
            return;
        }
        constexpr SizeType kAlign = SizeType{2} << 20;
        const SizeType bytes =
            (((n * sizeof(T)) + kAlign - 1) / kAlign) * kAlign;
        m_data = static_cast<T*>(std::aligned_alloc(kAlign, bytes)); // NOLINT
        if (m_data == nullptr) {
            throw std::bad_alloc();
        }
#if defined(__linux__) && defined(MADV_HUGEPAGE)
        ::madvise(m_data, bytes, MADV_HUGEPAGE); // a hint; failure is harmless
#endif
        auto* raw            = reinterpret_cast<std::byte*>(m_data);
        const SizeType pages = bytes / kAlign;
#pragma omp parallel for num_threads(nthreads) schedule(static)
        for (SizeType i = 0; i < pages; ++i) {
            std::memset(raw + (i * kAlign), 0, kAlign);
        }
    }

    void swap(FDMTLargeArray& other) noexcept {
        std::swap(m_data, other.m_data);
        std::swap(m_size, other.m_size);
    }

    [[nodiscard]] T* data() noexcept { return m_data; }
    [[nodiscard]] const T* data() const noexcept { return m_data; }
    [[nodiscard]] SizeType size() const noexcept { return m_size; }
    [[nodiscard]] bool empty() const noexcept { return m_size == 0; }
    [[nodiscard]] T* begin() noexcept { return m_data; }
    [[nodiscard]] T* end() noexcept { return m_data + m_size; }
    [[nodiscard]] const T* begin() const noexcept { return m_data; }
    [[nodiscard]] const T* end() const noexcept { return m_data + m_size; }

private:
    T* m_data{nullptr};
    SizeType m_size{0};
};

/**
 * @brief out[i] = a[i] + b[i] for i < m with non-temporal (streaming) stores
 * where the target supports them, for rows that go to main memory anyway (a
 * level far larger than the caches): it skips the read-for-ownership of
 * every output line. Same arithmetic as the plain loop, so the result is
 * bit-identical. Callers must fdmt_stream_fence() before another thread
 * reads `out`.
 */
template <typename StateT>
inline void fdmt_stream_add(const StateT* __restrict__ a,
                            const StateT* __restrict__ b,
                            StateT* __restrict__ out,
                            SizeType m) noexcept {
    // NOLINTNEXTLINE(bugprone-branch-clone): SIMD path when AVX is on
    if constexpr (std::is_same_v<StateT, float>) {
#if defined(__AVX512F__) || defined(__AVX__)
#if defined(__AVX512F__)
        constexpr SizeType kLanes = 16;
#else
        constexpr SizeType kLanes = 8;
#endif
        constexpr std::uintptr_t kAlign = kLanes * sizeof(float);
        SizeType i                      = 0;
        for (;
             i < m && (reinterpret_cast<std::uintptr_t>(out + i) % kAlign) != 0;
             ++i) {
            out[i] = a[i] + b[i];
        }
        for (; i + kLanes <= m; i += kLanes) {
#if defined(__AVX512F__)
            _mm512_stream_ps(out + i, _mm512_add_ps(_mm512_loadu_ps(a + i),
                                                    _mm512_loadu_ps(b + i)));
#else
            _mm256_stream_ps(out + i, _mm256_add_ps(_mm256_loadu_ps(a + i),
                                                    _mm256_loadu_ps(b + i)));
#endif
        }
        for (; i < m; ++i) {
            out[i] = a[i] + b[i];
        }
#elif defined(__clang__) && __has_builtin(__builtin_nontemporal_store)
        for (SizeType i = 0; i < m; ++i) {
            __builtin_nontemporal_store(a[i] + b[i], out + i);
        }
#else
#pragma omp simd
        for (SizeType i = 0; i < m; ++i) {
            out[i] = a[i] + b[i];
        }
#endif
    } else {
#if defined(__AVX512F__)
        if constexpr (std::is_same_v<StateT, uint32_t>) {
            constexpr SizeType kLanes       = 16;
            constexpr std::uintptr_t kAlign = kLanes * sizeof(uint32_t);
            SizeType i                      = 0;
            for (; i < m &&
                   (reinterpret_cast<std::uintptr_t>(out + i) % kAlign) != 0;
                 ++i) {
                out[i] = a[i] + b[i];
            }
            for (; i + kLanes <= m; i += kLanes) {
                _mm512_stream_si512(
                    reinterpret_cast<__m512i*>(out + i),
                    _mm512_add_epi32(
                        _mm512_loadu_si512(
                            reinterpret_cast<const __m512i*>(a + i)),
                        _mm512_loadu_si512(
                            reinterpret_cast<const __m512i*>(b + i))));
            }
            for (; i < m; ++i) {
                out[i] = a[i] + b[i];
            }
            return;
        }
#endif
#pragma omp simd
        for (SizeType i = 0; i < m; ++i) {
            out[i] = a[i] + b[i];
        }
    }
}

/// Orders this thread's streaming stores before a following barrier.
inline void fdmt_stream_fence() noexcept {
#if defined(__AVX__)
    _mm_sfence();
#elif defined(__clang__) && __has_builtin(__builtin_nontemporal_store)
    std::atomic_thread_fence(std::memory_order_release);
#endif
}

} // namespace dmt::algorithms::detail
