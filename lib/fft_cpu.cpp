#include "dmt/fft.hpp"

#include <cstddef>
#include <utility>

#ifdef DMT_ENABLE_OPENMP
#include <omp.h>
#endif
#include <fftw3.h>

#include <spdlog/spdlog.h>

namespace dmt {

template <>
class FFTManager<backend::CPU>::Impl {
public:
    Impl(SizeType nfft,
         SizeType nsub,
         SizeType nbin,
         SizeType mbin,
         SizeType nchan,
         int nthreads)
        : m_nfft(nfft),
          m_nsub(nsub),
          m_nbin(nbin),
          m_mbin(mbin),
          m_nchan(nchan),
          m_nthreads(nthreads) {
#ifdef DMT_ENABLE_OPENMP
        if (m_nthreads <= 0) {
            m_nthreads = omp_get_max_threads();
        }
        // Set the number of threads
        if (fftwf_init_threads() == 0) {
            spdlog::error("FFTW failed to initialize threads support.");
            throw std::runtime_error("Failed to initialise FFTW threads");
        }
        fftwf_plan_with_nthreads(m_nthreads);
        spdlog::debug(
            "FFTManager<CPU>::Impl: FFTW threads initialized with {} threads.",
            m_nthreads);
#else
        // Warn if nthreads > 1 but OpenMP is not enabled
        if (m_nthreads > 1) {
            spdlog::warn(
                "FFTManager<CPU>::Impl: Warning - nthreads > 1 specified, but "
                "OpenMP is not enabled (DMT_ENABLE_OPENMP undefined). "
                "Using single thread.");
        }
        m_nthreads = 1;
#endif
        spdlog::debug("FFTManager<CPU>::Impl created.");
    }

    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&)                 = delete;
    Impl& operator=(Impl&&)      = delete;

    ~Impl() {
        if (m_forward_plan != nullptr) {
            fftwf_destroy_plan(m_forward_plan);
        }
        if (m_backward_plan != nullptr) {
            fftwf_destroy_plan(m_backward_plan);
        }
#ifdef DMT_ENABLE_OPENMP
        fftwf_cleanup_threads();
#endif
    }
    void initialize_plans(std::span<ComplexType> unpack_buffer,
                          std::span<ComplexType> delay_buffer) {
        spdlog::debug("Initializing FFTW plans...");
        // Ensure pointers are valid complex types for FFTW
        auto* unpack_buffer_ptr =
            reinterpret_cast<fftwf_complex*>(unpack_buffer.data());
        auto* delay_buffer_ptr =
            reinterpret_cast<fftwf_complex*>(delay_buffer.data());

        // Check buffer sizes (not needed)
        // --- Forward Plan ---
        // 1D FFT of size m_nbin
        // Batch size: m_nfft * m_nsub
        // Input stride = 1, Input distance = m_nbin
        // Output stride = 1, Output distance = m_nbin (in-place)
        const std::array<int, 1> fft_size_fw = {static_cast<int>(m_nbin)};
        int howmany_fw      = static_cast<int>(m_nfft * m_nsub);
        int istride_fw      = 1;
        int idist_fw        = static_cast<int>(m_nbin);
        int ostride_fw      = 1;
        int odist_fw        = static_cast<int>(m_nbin);
        unsigned plan_flags = FFTW_MEASURE;
        m_forward_plan      = fftwf_plan_many_dft(
            1,                  // rank
            fft_size_fw.data(), // n
            howmany_fw,         // howmany
            unpack_buffer_ptr,  // in
            nullptr,            // inembed (null for simple stride/dist)
            istride_fw,         // istride
            idist_fw,           // idist
            unpack_buffer_ptr,  // out (in-place)
            nullptr,            // onembed (null for simple stride/dist)
            ostride_fw,         // ostride
            odist_fw,           // odist
            FFTW_FORWARD,       // sign
            plan_flags          // flags
        );
        if (m_forward_plan == nullptr) {
            throw std::runtime_error("Failed to create forward FFTW plan");
        }
        spdlog::debug("Forward FFTW plan created: {}");

        // --- Backward Plan ---
        // 1D FFT of size m_mbin
        // Batch size: m_nfft * m_nsub * m_nchan
        // Input stride = 1, Input distance = m_mbin
        // Output stride = 1, Output distance = m_mbin (in-place)
        const std::array<int, 1> fft_size_bw = {static_cast<int>(m_mbin)};
        int howmany_bw = static_cast<int>(m_nfft * m_nsub * m_nchan);
        int istride_bw = 1;
        int idist_bw   = static_cast<int>(m_mbin);
        int ostride_bw = 1;
        int odist_bw   = static_cast<int>(m_mbin);

        m_backward_plan =
            fftwf_plan_many_dft(1,                  // rank
                                fft_size_bw.data(), // n
                                howmany_bw,         // howmany
                                delay_buffer_ptr,   // in
                                nullptr,            // inembed
                                istride_bw,         // istride
                                idist_bw,           // idist
                                delay_buffer_ptr,   // out (in-place)
                                nullptr,            // onembed
                                ostride_bw,         // ostride
                                odist_bw,           // odist
                                FFTW_BACKWARD,      // sign
                                plan_flags          // flags
            );
        if (m_backward_plan == nullptr) {
            throw std::runtime_error("Failed to create FFTW plans");
        }
        spdlog::debug("Backward FFTW plan created:");
        spdlog::info("FFTW plans initialized successfully.");
    }

    void forward_fft(std::span<ComplexType> data) const {
        if (m_forward_plan == nullptr) {
            throw std::logic_error("Forward FFT plan not initialized.");
        }
        auto* data_ptr = reinterpret_cast<fftwf_complex*>(data.data());
        fftwf_execute_dft(m_forward_plan, data_ptr, data_ptr);
        swap_spectrum(data, m_nbin, m_nfft * m_nsub);
    }

    void backward_fft(std::span<ComplexType> data) const {
        if (m_backward_plan == nullptr) {
            throw std::logic_error("Backward FFT plan not initialized.");
        }
        auto* data_ptr = reinterpret_cast<fftwf_complex*>(data.data());
        swap_spectrum(data, m_mbin, m_nfft * m_nsub * m_nchan);
        fftwf_execute_dft(m_backward_plan, data_ptr, data_ptr);
    }

    static void
    swap_spectrum(std::span<ComplexType> data, SizeType nx, SizeType ny) {
        if (nx == 0 || ny == 0) {
            throw std::invalid_argument("Invalid dimensions in swap_spectrum");
        }
        ComplexType* data_ptr = data.data();

        const SizeType total_elements = nx * ny;
        if (data.size() != total_elements) {
            throw std::invalid_argument(
                "Span size does not match dimensions in "
                "swap_spectrum");
        }
        if (nx % 2 != 0) {
            throw std::invalid_argument("nx must be even in swap_spectrum");
        }
        // Swap the halves along the last dimension
        const SizeType mid_point = nx / 2;
        if (mid_point == 0 && nx > 0) {
            return;
        }
        for (SizeType j = 0; j < ny; ++j) {
            const SizeType offset      = j * nx;
            ComplexType* row_start_ptr = data_ptr + offset;
            std::rotate(row_start_ptr, row_start_ptr + mid_point,
                        row_start_ptr + nx);
        }
    }

private:
    SizeType m_nfft;
    SizeType m_nsub;
    SizeType m_nbin;
    SizeType m_mbin;
    SizeType m_nchan;
    int m_nthreads;

    fftwf_plan m_forward_plan  = nullptr;
    fftwf_plan m_backward_plan = nullptr;

}; // End FFTManager<backend::CPU>::Impl definition

// CPU-specific constructor implementation
template <>
template <std::same_as<backend::CPU> P>
FFTManager<backend::CPU>::FFTManager(SizeType nfft,
                                     SizeType nsub,
                                     SizeType nbin,
                                     SizeType mbin,
                                     SizeType nchan,
                                     int nthreads)
    : m_impl(std::make_unique<Impl>(nfft, nsub, nbin, mbin, nchan, nthreads)) {
    spdlog::debug("FFTManager<CPU> object created.");
}
template <>
FFTManager<backend::CPU>::~FFTManager() {
    spdlog::debug("FFTManager<CPU> object destroyed.");
}
template <>
FFTManager<backend::CPU>::FFTManager(FFTManager&& other) noexcept
    : m_impl(std::move(other.m_impl)) {
    spdlog::debug("FFTManager<CPU> object moved.");
}
template <>
FFTManager<backend::CPU>&
FFTManager<backend::CPU>::operator=(FFTManager&& other) noexcept {
    if (this != &other) {
        m_impl = std::move(other.m_impl);
    }
    return *this;
}
template <>
template <std::same_as<backend::CPU> P>
void FFTManager<backend::CPU>::initialize_plans(
    std::span<ComplexType> unpack_buffer, std::span<ComplexType> delay_buffer) {
    m_impl->initialize_plans(unpack_buffer, delay_buffer);
}
template <>
template <std::same_as<backend::CPU> P>
void FFTManager<backend::CPU>::forward_fft(std::span<ComplexType> data) const {
    m_impl->forward_fft(data);
}
template <>
template <std::same_as<backend::CPU> P>
void FFTManager<backend::CPU>::backward_fft(std::span<ComplexType> data) const {
    m_impl->backward_fft(data);
}
template <>
template <std::same_as<backend::CPU> P>
void FFTManager<backend::CPU>::swap_spectrum(std::span<ComplexType> data,
                                             SizeType nx,
                                             SizeType ny) {
    dmt::FFTManager<>::Impl::swap_spectrum(data, nx, ny);
}
// Explicit instantiation (for linking)
template FFTManager<backend::CPU>::FFTManager(
    SizeType, SizeType, SizeType, SizeType, SizeType, int);

} // namespace dmt
