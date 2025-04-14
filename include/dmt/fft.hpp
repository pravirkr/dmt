#pragma once

#include <memory>
#include <span>

#ifdef DMT_ENABLE_CUDA
#include <cuda/std/span>
#include <cuda_runtime_api.h>
#endif // DMT_ENABLE_CUDA

#include "dmt/common/types.hpp"

namespace dmt {

/**
 * @brief Manages FFT plans and execution for CPU (FFTW) or CUDA (cuFFT).
 *
 * Encapsulates the creation and management of FFT plans and provides
 * methods for forward and backward transforms.
 *
 * @tparam Backend The execution backend (dmt::backend::CPU or
 * dmt::backend::CUDA).
 */
template <backend::ExecutionBackend Backend = backend::CPU>
class FFTManager {
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
    template <std::same_as<backend::CPU> P = Backend>
    FFTManager(
        int nfft, int nsub, int nbin, int mbin, int nchan, int nthreads = 1);

#ifdef DMT_ENABLE_CUDA
    /**
     * @brief Construct for CUDA backend using cuFFT.
     * @param nfft Size of the FFT.
     * @param nsub Number of subbands (for potential batching or planning).
     * @param nbin Number of bins (for potential batching or planning).
     * @param mbin Another dimension (for potential batching or planning).
     * @param nchan Number of channels (for potential batching or planning).
     * @param device_id CUDA device ID.
     */
    template <std::same_as<backend::CUDA> P = Backend>
    FFTManager(
        int nfft, int nsub, int nbin, int mbin, int nchan, int device_id = 0);
#endif // DMT_ENABLE_CUDA

    ~FFTManager();
    FFTManager(FFTManager&&) noexcept;
    FFTManager& operator=(FFTManager&&) noexcept;
    FFTManager(const FFTManager&)            = delete;
    FFTManager& operator=(const FFTManager&) = delete;

    /**
     * @brief Initialize the FFT plans.
     *
     * @tparam P The backend type.
     * @param unpack_buffer The buffer to unpack the data.
     * @param delay_buffer The buffer to delay the data.
     */
    template <std::same_as<backend::CPU> P = Backend>
    void initialize_plans(std::span<ComplexType> unpack_buffer,
                          std::span<ComplexType> delay_buffer);

    /**
     * @brief Performs an in-place forward FFT on CPU data.
     * @param data1 Host data buffer 1 (must match planned dimensions/size).
     * @param data2 Host data buffer 2 (must match planned dimensions/size).
     */
    template <std::same_as<backend::CPU> P = Backend>
    void forward_fft(std::span<ComplexType> data1,
                     std::span<ComplexType> data2) const;

    /**
     * @brief Performs an in-place backward FFT on CPU data.
     * @param data1 Host data buffer 1 (must match planned dimensions/size).
     * @param data2 Host data buffer 2 (must match planned dimensions/size).
     */
    template <std::same_as<backend::CPU> P = Backend>
    void backward_fft(std::span<ComplexType> data1,
                      std::span<ComplexType> data2) const;

#ifdef DMT_ENABLE_CUDA
    /**
     * @brief Performs an in-place forward FFT on GPU data.
     * @param data Device data buffer (must match planned dimensions/size).
     * @param stream CUDA stream for execution.
     */
    template <std::same_as<backend::CUDA> P = Backend>
    void forward_fft(cuda::std::span<ComplexTypeCUDA> data,
                     cudaStream_t stream = nullptr) const;

    /**
     * @brief Performs an in-place backward FFT on GPU data.
     * @param data Device data buffer (must match planned dimensions/size).
     * @param stream CUDA stream for execution.
     */
    template <std::same_as<backend::CUDA> P = Backend>
    void backward_fft(cuda::std::span<ComplexTypeCUDA> data,
                      cudaStream_t stream = nullptr) const;

    template <std::same_as<backend::CUDA> P = Backend>
    static void swap_spectrum(cuda::std::span<ComplexTypeCUDA> data,
                              SizeType nx,
                              SizeType ny);
#endif // DMT_ENABLE_CUDA

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

// Type aliases for convenience
using FFTManagerCPU = FFTManager<backend::CPU>;
#ifdef DMT_ENABLE_CUDA
using FFTManagerCUDA = FFTManager<backend::CUDA>;
#endif // DMT_ENABLE_CUDA

} // namespace dmt