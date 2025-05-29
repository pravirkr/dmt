#include "dmt/utils/fft.hpp"

#include <utility>

#ifdef DMT_ENABLE_OPENMP
#include <omp.h>
#endif
#include <fftw3.h>

#include <spdlog/spdlog.h>

#include "dmt/bb_utils_cpu.hpp"

namespace dmt::utils {

template <>
class FFTManager<backend::CPU>::Impl {
public:
    Impl(int nfft, int nsub, int nbin, int mbin, int nchan, int nthreads)
        : m_nfft(nfft),
          m_nsub(nsub),
          m_nbin(nbin),
          m_mbin(mbin),
          m_nchan(nchan),
          m_nthreads(nthreads) {
        configure_threading();
        spdlog::debug("FFTManager<CPU>::Impl: Initialized with nfft={}, "
                      "nsub={}, nbin={}, mbin={}, nchan={}, nthreads={}",
                      nfft, nsub, nbin, mbin, nchan, m_nthreads);
    }

    ~Impl() noexcept {
        spdlog::debug("FFTManager<CPU>::Impl: Destroying instance");
        cleanup_resources();
    }
    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&)                 = delete;
    Impl& operator=(Impl&&)      = delete;

    void initialize_plans(std::span<ComplexType> unpack_buffer,
                          std::span<ComplexType> delay_buffer) {
        spdlog::debug("initialize_plans: Creating FFTW plans");
        // Ensure pointers are valid complex types for FFTW
        auto* unpack_buffer_ptr =
            reinterpret_cast<fftwf_complex*>(unpack_buffer.data());
        auto* delay_buffer_ptr =
            reinterpret_cast<fftwf_complex*>(delay_buffer.data());

        unsigned plan_flags = FFTW_MEASURE;
        const int rank      = 1;

        // --- Forward Plan ---
        // 1D FFT of size m_nbin
        // Batch size: m_nfft * m_nsub
        // Input stride = 1, Input distance = m_nbin
        // Output stride = 1, Output distance = m_nbin (in-place)
        const std::array<int, 1> fft_size_fw = {m_nbin};

        const int howmany_fw = (m_nfft * m_nsub);
        const int idist_fw   = m_nbin;
        const int odist_fw   = m_nbin;
        const int istride_fw = 1;
        const int ostride_fw = 1;

        m_forward_plan =
            fftwf_plan_many_dft(rank,               // rank
                                fft_size_fw.data(), // n
                                howmany_fw,         // howmany
                                unpack_buffer_ptr,  // in
                                nullptr,            // inembed
                                istride_fw,         // istride
                                idist_fw,           // idist
                                unpack_buffer_ptr,  // out (in-place)
                                nullptr,            // onembed
                                ostride_fw,         // ostride
                                odist_fw,           // odist
                                FFTW_FORWARD,       // sign
                                plan_flags          // flags
            );
        if (m_forward_plan == nullptr) {
            throw std::runtime_error("Failed to create forward FFTW plan");
        }
        spdlog::debug("Forward FFTW plan created");

        // --- Backward Plan ---
        // 1D FFT of size m_mbin
        // Batch size: m_nfft * m_nsub * m_nchan
        // Input stride = 1, Input distance = m_mbin
        // Output stride = 1, Output distance = m_mbin (in-place)
        const std::array<int, 1> fft_size_bw = {m_mbin};

        const int howmany_bw = (m_nfft * m_nsub * m_nchan);
        const int idist_bw   = m_mbin;
        const int odist_bw   = m_mbin;
        const int istride_bw = 1;
        const int ostride_bw = 1;

        m_backward_plan =
            fftwf_plan_many_dft(rank,               // rank
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
            // Clean up forward plan if backward fails
            if (m_forward_plan != nullptr) {
                fftwf_destroy_plan(m_forward_plan);
                m_forward_plan = nullptr;
            }
            throw std::runtime_error("Failed to create FFTW plans");
        }
        spdlog::debug("Backward FFTW plan created:");
    }

    void forward_fft(std::span<ComplexType> data1,
                     std::span<ComplexType> data2) const {
        if (m_forward_plan == nullptr) {
            throw std::logic_error("Forward FFT plan not initialized.");
        }
        auto* data1_ptr = reinterpret_cast<fftwf_complex*>(data1.data());
        auto* data2_ptr = reinterpret_cast<fftwf_complex*>(data2.data());
        fftwf_execute_dft(m_forward_plan, data1_ptr, data1_ptr);
        fftwf_execute_dft(m_forward_plan, data2_ptr, data2_ptr);
        bb_utils::swap_spectrum(data1, data2, m_nbin, m_nfft * m_nsub);
        spdlog::debug("forward_fft: Completed FFT and spectrum swap");
    }

    void backward_fft(std::span<ComplexType> data1,
                      std::span<ComplexType> data2) const {
        if (m_backward_plan == nullptr) {
            throw std::logic_error("Backward FFT plan not initialized.");
        }
        auto* data1_ptr = reinterpret_cast<fftwf_complex*>(data1.data());
        auto* data2_ptr = reinterpret_cast<fftwf_complex*>(data2.data());
        bb_utils::swap_spectrum(data1, data2, m_mbin,
                                m_nfft * m_nsub * m_nchan);
        fftwf_execute_dft(m_backward_plan, data1_ptr, data1_ptr);
        fftwf_execute_dft(m_backward_plan, data2_ptr, data2_ptr);
        spdlog::debug("backward_fft: Completed spectrum swap and FFT");
    }

private:
    const int m_nfft;
    const int m_nsub;
    const int m_nbin;
    const int m_mbin;
    const int m_nchan;
    int m_nthreads;

    fftwf_plan m_forward_plan  = nullptr;
    fftwf_plan m_backward_plan = nullptr;

    // Configures threading for FFTW
    void configure_threading() {
#ifdef DMT_ENABLE_OPENMP
        if (m_nthreads <= 0) {
            m_nthreads = omp_get_max_threads();
            spdlog::debug("configure_threading: Using max threads: {}",
                          m_nthreads);
        }
        if (fftwf_init_threads() == 0) {
            spdlog::error(
                "configure_threading: Failed to initialize FFTW threads");
            throw std::runtime_error("Failed to initialize FFTW threads");
        }
        fftwf_plan_with_nthreads(m_nthreads);
        spdlog::debug("configure_threading: FFTW initialized with {} threads",
                      m_nthreads);
#else
        if (m_nthreads > 1) {
            spdlog::warn("configure_threading: nthreads={} requested but "
                         "OpenMP disabled; using single thread",
                         m_nthreads);
            m_nthreads = 1;
        }
#endif
    }

    // Cleans up FFTW resources
    void cleanup_resources() noexcept {
        try {
            if (m_forward_plan != nullptr) {
                fftwf_destroy_plan(m_forward_plan);
                m_forward_plan = nullptr;
                spdlog::debug("cleanup_resources: Forward plan destroyed");
            }
            if (m_backward_plan != nullptr) {
                fftwf_destroy_plan(m_backward_plan);
                m_backward_plan = nullptr;
                spdlog::debug("cleanup_resources: Backward plan destroyed");
            }
#ifdef DMT_ENABLE_OPENMP
            fftwf_cleanup_threads();
            spdlog::debug("cleanup_resources: FFTW threads cleaned up");
#endif
            fftwf_cleanup();
            spdlog::debug("cleanup_resources: FFTW global cleanup completed");
        } catch (...) {
            spdlog::error(
                "cleanup_resources: Unexpected exception during cleanup");
        }
    }

}; // End FFTManager<backend::CPU>::Impl definition

// CPU-specific constructor implementation
template <>
template <std::same_as<backend::CPU> P>
FFTManager<backend::CPU>::FFTManager(
    int nfft, int nsub, int nbin, int mbin, int nchan, int nthreads)
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
void FFTManager<backend::CPU>::forward_fft(std::span<ComplexType> data1,
                                           std::span<ComplexType> data2) const {
    m_impl->forward_fft(data1, data2);
}
template <>
template <std::same_as<backend::CPU> P>
void FFTManager<backend::CPU>::backward_fft(
    std::span<ComplexType> data1, std::span<ComplexType> data2) const {
    m_impl->backward_fft(data1, data2);
}
// Explicit instantiation (for linking)
template FFTManager<backend::CPU>::FFTManager(int, int, int, int, int, int);

} // namespace dmt::utils
