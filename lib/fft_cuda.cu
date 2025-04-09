#include "dmt/fft.hpp"

#include <cstddef>
#include <utility>


#include <spdlog/spdlog.h>

namespace dmt {

template <>
class FFTManager<backend::CUDA>::Impl {
public:
    Impl(SizeType nfft,
         SizeType nsub,
         SizeType nbin,
         SizeType mbin,
         SizeType nchan)
        : m_nfft(nfft),
          m_nsub(nsub),
          m_nbin(nbin),
          m_mbin(mbin),
          m_nchan(nchan) {}

    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&)                 = delete;
    Impl& operator=(Impl&&)      = delete;
    ~Impl();
    void initialize_plans(std::span<ComplexTypeCUDA> unpack_buffer,
                          std::span<ComplexTypeCUDA> delay_buffer);
    void forward_fft(std::span<ComplexTypeCUDA> data) const;
    void backward_fft(std::span<ComplexTypeCUDA> data) const;
    static void
    swap_spectrum(std::span<ComplexTypeCUDA> data, SizeType nx, SizeType ny);

private:
    SizeType m_nfft;
    SizeType m_nsub;
    SizeType m_nbin;
    SizeType m_mbin;
    SizeType m_nchan;

    cufftHandle m_forward_plan  = 0;
    cufftHandle m_backward_plan = 0;

}; // End FFTManager<backend::CPU>::Impl definition

// CPU-specific constructor implementation
template <>
template <std::same_as<backend::CUDA> P>
FFTManager<backend::CUDA>::FFTManager(
    SizeType nfft, SizeType nsub, SizeType nbin, SizeType mbin, SizeType nchan)
    : m_impl(std::make_unique<Impl>(nfft, nsub, nbin, mbin, nchan)) {
    spdlog::debug("FFTManager<CUDA> object created.");
}
template <>
FFTManager<backend::CUDA>::~FFTManager() {
    spdlog::debug("FFTManager<CUDA> object destroyed.");
}
template <>
FFTManager<backend::CUDA>::FFTManager(FFTManager&& other) noexcept
    : m_impl(std::move(other.m_impl)) {
    spdlog::debug("FFTManager<CUDA> object moved.");
}
template <>
FFTManager<backend::CUDA>&
FFTManager<backend::CUDA>::operator=(FFTManager&& other) noexcept {
    if (this != &other) {
        m_impl = std::move(other.m_impl);
    }
    return *this;
}
template <>
template <std::same_as<backend::CUDA> P>
void FFTManager<backend::CUDA>::initialize_plans(
    std::span<ComplexTypeCUDA> unpack_buffer,
    std::span<ComplexTypeCUDA> delay_buffer) {
    m_impl->initialize_plans(unpack_buffer, delay_buffer);
}
template <>
template <std::same_as<backend::CUDA> P>
void FFTManager<backend::CUDA>::forward_fft(
    std::span<ComplexTypeCUDA> data) const {
    m_impl->forward_fft(data);
}
template <>
template <std::same_as<backend::CUDA> P>
void FFTManager<backend::CUDA>::backward_fft(
    std::span<ComplexTypeCUDA> data) const {
    m_impl->backward_fft(data);
}
template <>
template <std::same_as<backend::CUDA> P>
void FFTManager<backend::CUDA>::swap_spectrum(std::span<ComplexTypeCUDA> data,
                                              SizeType nx,
                                              SizeType ny) {
    dmt::FFTManager<>::Impl::swap_spectrum(data, nx, ny);
}
// Explicit instantiation (for linking)
template FFTManager<backend::CUDA>::FFTManager(
    SizeType, SizeType, SizeType, SizeType, SizeType, int);

} // namespace dmt
