#pragma once

/**
 * @file fft.hpp
 * @brief FFTW manager for batched CPU FFT plans (private; the CUDA
 * counterpart is fft_cuda.cuh).
 */

#include <cstdint>
#include <memory>
#include <span>

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

} // namespace dmt::utils
