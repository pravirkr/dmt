#pragma once

/**
 * @file fft_cuda.cuh
 * @brief cuFFT manager for batched GPU FFT plans (private).
 */

#include <memory>
#include <span>
#include <vector>

#include "dmt/gpu_compat.cuh"

#include "dmt/common/types.hpp"
#include "dmt/fft.hpp"
#include "dmt/gpu_utils.cuh"

namespace dmt::utils {

/**
 * @brief One batched 1D cuFFT, planned once and executed later.
 *
 * Encapsulates the creation and management of FFT plans and provides
 * methods for forward and backward transforms. Performs the FFT using
 * cuFFT. The caller passes the stream at execute time. Do not execute the
 * same CUFFTManager concurrently.
 */
class CUFFTManager {
public:
    /**
     * @brief Plan a batched 1D transform on @p device_id.
     * @param kind Transform kind.
     * @param length Real or complex length of one transform.
     * @param howmany Number of contiguous transforms.
     * @param device_id CUDA device that owns the plan and its workspace.
     */
    CUFFTManager(FFTKind kind,
                 SizeType length,
                 SizeType howmany,
                 int device_id);

    ~CUFFTManager();
    CUFFTManager(CUFFTManager&&) noexcept;
    CUFFTManager& operator=(CUFFTManager&&) noexcept;
    CUFFTManager(const CUFFTManager&)            = delete;
    CUFFTManager& operator=(const CUFFTManager&) = delete;

    /**
     * @brief In-place C2C transform. @p data must hold `howmany * length`
     *        complex samples.
     */
    void execute(cuda::std::span<ComplexTypeGPU> data,
                 cudaStream_t stream = nullptr) const;

    /**
     * @brief Out-of-place R2C or C2R.
     *
     * For R2C, @p real is the input and @p freq is the output. For C2R the
     * roles are reversed. C2R may overwrite @p freq.
     */
    void execute(cuda::std::span<float> real,
                 cuda::std::span<ComplexTypeGPU> freq,
                 cudaStream_t stream = nullptr) const;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace dmt::utils
