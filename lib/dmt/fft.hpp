#pragma once

/**
 * @file fft.hpp
 * @brief FFTW manager for batched CPU FFT plans (private; the CUDA
 * counterpart is fft_cuda.cuh).
 */

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <new>
#include <span>
#include <vector>

#include "dmt/common/types.hpp"

namespace dmt::utils {

/**
 * @brief Kind of batched 1D transform owned by FFTWManager.
 */
enum class FFTKind : std::uint8_t {
    kC2CForward  = 0, ///< In-place complex-to-complex forward.
    kC2CBackward = 1, ///< In-place complex-to-complex backward (unnormalized).
    kR2C         = 2, ///< Out-of-place real-to-complex.
    kC2R         = 3, ///< Out-of-place complex-to-real (unnormalized).
};

/// @brief FFTW's own (deterministic, heuristic) cost estimate of a
/// length-@p n R2C transform, per n * log2(n): lengths whose factors need
/// slow codelets score higher. Used to choose transform lengths.
double r2c_cost_per_nlogn(SizeType n);

/// @brief FFTW planner flags for the process-wide planner effort
/// (dmt::fft::set_planner).
unsigned fftw_planner_flags() noexcept;

/**
 * @brief One batched 1D FFT, planned once and executed later.
 *
 * Plans are single-threaded FFTW (`FFTW_ESTIMATE`). At execution the batch
 * is split across @p nthreads OpenMP workers; each worker runs one plan on
 * its contiguous slice. The same plan may be executed concurrently on
 * non-overlapping slices.
 */
class FFTWManager {
public:
    /**
     * @brief Plan a batched 1D transform.
     * @param kind Transform kind.
     * @param length Real or complex length of one transform (`n` for C2C,
     *        real length for R2C/C2R).
     * @param howmany Number of contiguous transforms.
     * @param nthreads OpenMP workers used at execute time (at least 1).
     */
    FFTWManager(FFTKind kind, SizeType length, SizeType howmany, int nthreads);

    ~FFTWManager();
    FFTWManager(FFTWManager&&) noexcept;
    FFTWManager& operator=(FFTWManager&&) noexcept;
    FFTWManager(const FFTWManager&)            = delete;
    FFTWManager& operator=(const FFTWManager&) = delete;

    /**
     * @brief In-place C2C transform. @p data must hold `howmany * length`
     *        complex samples.
     */
    void execute(std::span<ComplexType> data) const;

    /**
     * @brief Out-of-place R2C or C2R.
     *
     * For R2C, @p real is the input (`howmany * length`) and @p freq is the
     * output (`howmany * (length/2+1)`). For C2R the roles are reversed.
     * C2R may overwrite @p freq.
     */
    void execute(std::span<float> real, std::span<ComplexType> freq) const;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

/// Alignment (bytes) of every array a FFTWRowPlan executes on.
inline constexpr std::size_t kFFTAlignment = 64;

/// @brief Allocator giving kFFTAlignment-aligned storage (FFTW SIMD codelets
/// and whole-cache-line rows).
template <typename T> struct FFTAllocator {
    using value_type = T;
    FFTAllocator()   = default;
    template <typename U> explicit FFTAllocator(const FFTAllocator<U>&) {}
    T* allocate(std::size_t n) {
        const std::size_t bytes =
            ((n * sizeof(T)) + kFFTAlignment - 1) / kFFTAlignment *
            kFFTAlignment;
        void* ptr = std::aligned_alloc(kFFTAlignment, bytes == 0 ? kFFTAlignment
                                                                 : bytes);
        if (ptr == nullptr) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
    }
    void deallocate(T* ptr, std::size_t /*n*/) noexcept { std::free(ptr); }
    template <typename U> bool operator==(const FFTAllocator<U>&) const {
        return true;
    }
};

template <typename T> using FFTVector = std::vector<T, FFTAllocator<T>>;

/// Rounds a row length (elements of T) up so consecutive rows stay
/// kFFTAlignment-aligned.
template <typename T> constexpr SizeType fft_row_stride(SizeType n) {
    constexpr SizeType kPer = kFFTAlignment / sizeof(T);
    return (n + kPer - 1) / kPer * kPer;
}

/**
 * @brief A batch of @p howmany 1D transforms, planned once (with the
 * process-wide planner effort, see dmt/common/fft_config.hpp) and executed on
 * any caller arrays ("new-array execute").
 *
 * Every array passed to execute must be kFFTAlignment-aligned, and the
 * distance between consecutive transforms is @p in_dist / @p out_dist
 * elements (0: packed rows). execute is thread-safe: threads may run the same
 * plan concurrently on different arrays. C2R overwrites its input.
 */
class FFTWRowPlan {
public:
    FFTWRowPlan(FFTKind kind,
                SizeType length,
                SizeType howmany  = 1,
                SizeType in_dist  = 0,
                SizeType out_dist = 0);
    ~FFTWRowPlan();
    FFTWRowPlan(FFTWRowPlan&&) noexcept;
    FFTWRowPlan& operator=(FFTWRowPlan&&) noexcept;
    FFTWRowPlan(const FFTWRowPlan&)            = delete;
    FFTWRowPlan& operator=(const FFTWRowPlan&) = delete;

    /// @brief R2C: @p in real rows -> @p out complex rows.
    void r2c(float* in, ComplexType* out) const noexcept;
    /// @brief C2R: @p in complex rows (destroyed) -> @p out real rows.
    void c2r(ComplexType* in, float* out) const noexcept;
    /// @brief In-place C2C in the plan's direction (C2C plans are in-place;
    /// @p in_dist must equal @p out_dist).
    void c2c(ComplexType* data) const noexcept;

    [[nodiscard]] SizeType length() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace dmt::utils
