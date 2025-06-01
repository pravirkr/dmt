#pragma once

#include <memory>
#include <span>

#ifdef DMT_ENABLE_CUDA
#include <cuda/std/span>
#include <cuda_runtime_api.h>
#endif // DMT_ENABLE_CUDA

#include "dmt/common/types.hpp"

namespace dmt::utils {

/**
 * @brief Manages FFT plans and execution for CPU (FFTW).
 *
 * Encapsulates the creation and management of FFT plans and provides
 * methods for forward and backward transforms. Performs the FFT using
 * FFTW.
 */
class FFTManagerCPU {
public:
    /**
     * @brief Construct for CPU backend using FFTW.
     * @param nfft Size of the FFT.
     * @param nsub Number of subbands (for potential batching or planning).
     * @param nbin Number of bins (for potential batching or planning).
     * @param mbin Another dimension (for potential batching or planning).
     * @param nchan Number of channels (for potential batching or planning).
     * @param nthreads Number of threads for FFTW planning and execution.
     */
    FFTManagerCPU(
        int nfft, int nsub, int nbin, int mbin, int nchan, int nthreads = 1);

    ~FFTManagerCPU();
    FFTManagerCPU(FFTManagerCPU&&) noexcept;
    FFTManagerCPU& operator=(FFTManagerCPU&&) noexcept;
    FFTManagerCPU(const FFTManagerCPU&)            = delete;
    FFTManagerCPU& operator=(const FFTManagerCPU&) = delete;

    /**
     * @brief Initialize the FFT plans.
     *
     * @param unpack_buffer The buffer to unpack the data.
     * @param delay_buffer The buffer to delay the data.
     */
    void initialize_plans(std::span<ComplexType> unpack_buffer,
                          std::span<ComplexType> delay_buffer);

    /**
     * @brief Performs an in-place forward FFT on CPU data.
     * @param data1 Host data buffer 1 (must match planned dimensions/size).
     * @param data2 Host data buffer 2 (must match planned dimensions/size).
     */
    void forward_fft(std::span<ComplexType> data1,
                     std::span<ComplexType> data2) const;

    /**
     * @brief Performs an in-place backward FFT on CPU data.
     * @param data1 Host data buffer 1 (must match planned dimensions/size).
     * @param data2 Host data buffer 2 (must match planned dimensions/size).
     */
    void backward_fft(std::span<ComplexType> data1,
                      std::span<ComplexType> data2) const;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

#ifdef DMT_ENABLE_CUDA
/**
 * @brief Manages FFT plans and execution for CUDA (cuFFT).
 *
 * Encapsulates the creation and management of FFT plans and provides
 * methods for forward and backward transforms. Performs the FFT using
 * cuFFT.
 */
class FFTManagerCUDA {
public:
    /**
     * @brief Construct for CUDA backend using cuFFT.
     * @param nfft Size of the FFT.
     * @param nsub Number of subbands (for potential batching or planning).
     * @param nbin Number of bins (for potential batching or planning).
     * @param mbin Another dimension (for potential batching or planning).
     * @param nchan Number of channels (for potential batching or planning).
     * @param device_id CUDA device ID.
     */
    FFTManagerCUDA(
        int nfft, int nsub, int nbin, int mbin, int nchan, int device_id = 0);

    ~FFTManagerCUDA();
    FFTManagerCUDA(FFTManagerCUDA&&) noexcept;
    FFTManagerCUDA& operator=(FFTManagerCUDA&&) noexcept;
    FFTManagerCUDA(const FFTManagerCUDA&)            = delete;
    FFTManagerCUDA& operator=(const FFTManagerCUDA&) = delete;

    /**
     * @brief Performs an in-place forward FFT on GPU data.
     * @param data Device data buffer (must match planned dimensions/size).
     * @param stream CUDA stream for execution.
     */
    void forward_fft(cuda::std::span<ComplexTypeCUDA> data1,
                     cuda::std::span<ComplexTypeCUDA> data2,
                     cudaStream_t stream = nullptr) const;

    /**
     * @brief Performs an in-place backward FFT on GPU data.
     * @param data Device data buffer (must match planned dimensions/size).
     * @param stream CUDA stream for execution.
     */
    void backward_fft(cuda::std::span<ComplexTypeCUDA> data1,
                      cuda::std::span<ComplexTypeCUDA> data2,
                      cudaStream_t stream = nullptr) const;

    static void swap_spectrum(cuda::std::span<ComplexTypeCUDA> data,
                              SizeType nx,
                              SizeType ny);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

#endif // DMT_ENABLE_CUDA

} // namespace dmt::utils