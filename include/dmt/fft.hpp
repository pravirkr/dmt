#pragma once

#include <memory>
#include <span>

#ifdef DMT_ENABLE_CUDA
#include <cuda/std/span>
#include <cuda_runtime_api.h>
#include <type_traits>
#endif // DMT_ENABLE_CUDA

#include "dmt/common/types.hpp"

namespace dmt {

template <backend::ExecutionBackend Backend = backend::CPU>
class FFTManager {
public:
    template <std::same_as<backend::CPU> P = Backend>
    FFTManager(SizeType nfft,
               SizeType nsub,
               SizeType nbin,
               SizeType mbin,
               SizeType nchan,
               int nthreads = 1);

#ifdef DMT_ENABLE_CUDA
    template <std::same_as<backend::CUDA> P = Backend>
    FFTManager(SizeType nfft,
               SizeType nsub,
               SizeType nbin,
               SizeType mbin,
               SizeType nchan);
#endif // DMT_ENABLE_CUDA

    ~FFTManager();
    FFTManager(FFTManager&&) noexcept;
    FFTManager& operator=(FFTManager&&) noexcept;
    FFTManager(const FFTManager&)            = delete;
    FFTManager& operator=(const FFTManager&) = delete;

    template <std::same_as<backend::CPU> P = Backend>
    void initialize_plans(std::span<ComplexType> unpack_buffer,
                          std::span<ComplexType> delay_buffer);

    template <std::same_as<backend::CPU> P = Backend>
    void forward_fft(std::span<ComplexType> data) const;

    template <std::same_as<backend::CPU> P = Backend>
    void backward_fft(std::span<ComplexType> data) const;

    template <std::same_as<backend::CPU> P = Backend>
    static void
    swap_spectrum(std::span<ComplexType> data, SizeType nx, SizeType ny);

#ifdef DMT_ENABLE_CUDA
    template <std::same_as<backend::CUDA> P = Backend>
    void initialize_plans(cuda::std::span<ComplexTypeCUDA> unpack_buffer,
                          cuda::std::span<ComplexTypeCUDA> delay_buffer);

    template <std::same_as<backend::CUDA> P = Backend>
    void forward_fft(cuda::std::span<ComplexTypeCUDA> data) const;

    template <std::same_as<backend::CUDA> P = Backend>
    void backward_fft(cuda::std::span<ComplexTypeCUDA> data) const;

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