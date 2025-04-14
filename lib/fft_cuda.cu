#include "dmt/fft.hpp"

#include <array>
#include <cstddef>
#include <utility>

#include <cuda/std/complex>
#include <cuda/std/span>
#include <cuda_runtime_api.h>
#include <cufft.h>

#include <spdlog/spdlog.h>

#include "dmt/cuda_utils.cuh"

namespace dmt {

template <>
class FFTManager<backend::CUDA>::Impl {
public:
    Impl(int nfft, int nsub, int nbin, int mbin, int nchan, int device_id)
        : m_nfft(nfft),
          m_nsub(nsub),
          m_nbin(nbin),
          m_mbin(mbin),
          m_nchan(nchan),
          m_device_id(device_id)
    // Plans and workspace initialized below
    {
        dmt::cuda::util::check_cuda_call(cudaSetDevice(m_device_id),
                                         std::source_location::current(),
                                         "Impl Constructor cudaSetDevice");
        spdlog::debug("FFTManager<CUDA>::Impl: Set device to {}.", m_device_id);

        try {
            // --- Create cuFFT Plans ---
            create_plans(); // Separate function for clarity

            // --- Allocate Workspace ---
            // Workspace size depends on the larger plan (check docs, usually
            // max is safe)
            size_t forward_work_size  = 0;
            size_t backward_work_size = 0;
            if (m_forward_plan) {
                dmt::cuda::util::check_cuda_call(
                    cufftGetSize(m_forward_plan, &forward_work_size),
                    std::source_location::current(), "cufftGetSize (forward)");
            }
            if (m_backward_plan) {
                dmt::cuda::util::check_cuda_call(
                    cufftGetSize(m_backward_plan, &backward_work_size),
                    std::source_location::current(), "cufftGetSize (backward)");
            }
            m_workspace_size = std::max(forward_work_size, backward_work_size);

            if (m_workspace_size > 0) {
                dmt::cuda::util::check_cuda_call(
                    cudaMalloc(&m_workspace_d, m_workspace_size),
                    std::source_location::current(), "Workspace cudaMalloc");
                spdlog::debug("FFTManager<CUDA>::Impl: Allocated {} bytes for "
                              "cuFFT workspace.",
                              m_workspace_size);
            } else {
                spdlog_debug(
                    "FFTManager<CUDA>::Impl: No cuFFT workspace needed.");
            }

        } catch (...) {
            // Ensure cleanup happens if constructor throws
            cleanup_resources();
            throw; // Re-throw the exception
        }

        spdlog::debug("FFTManager<CUDA>::Impl created and plans initialized.");
    }

    // Destructor
    ~Impl() {
        spdlog_debug("Destroying cuFFT resources.");
        cleanup_resources();
    }
    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&)                 = delete;
    Impl& operator=(Impl&&)      = delete;

    void forward_fft(cuda::std::span<ComplexTypeCUDA> data,
                     cudaStream_t stream) const {
        if (m_forward_plan == 0) {
            throw std::logic_error("Forward cuFFT plan not initialized.");
        }
        // Associate plan with the stream
        DMT_CHECK_CUFFT_CALL(cufftSetStream(m_forward_plan, stream),
                             "forward_fft: cufftSetStream");

        // Set workspace (required by cufftExec)
        if (m_workspace_size > 0) {
            DMT_CHECK_CUFFT_CALL(
                cufftSetWorkArea(m_forward_plan, m_workspace_d),
                "forward_fft: cufftSetWorkArea");
        }
        DMT_CHECK_CUFFT_CALL(
            cufftExecC2C(
                m_forward_plan, reinterpret_cast<cufftComplex*>(data.data()),
                reinterpret_cast<cufftComplex*>(data.data()), CUFFT_FORWARD),
            "forward_fft: cufftExecC2C");
        // Perform swap after FFT if desired
        swap_spectrum(data, m_nbin, m_nfft * m_nsub, stream);
    }

    void backward_fft(cuda::std::span<ComplexTypeCUDA> data,
                      cudaStream_t stream) const {
        if (m_backward_plan == 0) {
            throw std::logic_error("Backward cuFFT plan not initialized.");
        }
        // Perform swap before backward FFT if desired
        swap_spectrum(data, m_mbin, m_nfft * m_nsub * m_nchan, stream);

        // Associate plan with the stream
        DMT_CHECK_CUFFT_CALL(cufftSetStream(m_backward_plan, stream),
                             "backward_fft: cufftSetStream");

        // Set workspace
        if (m_workspace_size > 0) {
            DMT_CHECK_CUFFT_CALL(
                cufftSetWorkArea(m_backward_plan, m_workspace_d),
                "backward_fft: cufftSetWorkArea");
        }

        // Execute the FFT
        DMT_CHECK_CUFFT_CALL(
            cufftExecC2C(
                m_backward_plan, reinterpret_cast<cufftComplex*>(data.data()),
                reinterpret_cast<cufftComplex*>(data.data()), CUFFT_BACKWARD),
            "backward_fft: cufftExecC2C");
    }

private:
    int m_nfft;
    int m_nsub;
    int m_nbin;
    int m_mbin;
    int m_nchan;
    int m_device_id;

    cufftHandle m_forward_plan  = 0;
    cufftHandle m_backward_plan = 0;
    void* m_workspace_d         = nullptr;
    size_t m_workspace_size     = 0;

    void create_plans() {
        int rank = 1; // 1D FFTs

        // --- Forward Plan ---
        // 1D FFT of size m_nbin
        // Batch size: m_nfft * m_nsub
        // Input stride = 1, Input distance = m_nbin
        // Output stride = 1, Output distance = m_nbin (in-place)
        std::array<int, 1> n_fw = {m_nbin};
        int batch_fw            = m_nfft * m_nsub;
        int idist_fw            = m_nbin;
        int odist_fw            = m_nbin;
        int istride_fw          = 1;
        int ostride_fw          = 1;

        DMT_CHECK_CUFFT_CALL(cufftCreate(&m_forward_plan),
                             "cufftCreate (forward)");
        DMT_CHECK_CUFFT_CALL(
            cufftMakePlanMany(m_forward_plan,     // plan handle
                              rank,               // rank
                              n_fw.data(),        // n
                              nullptr,            // inembed
                              istride_fw,         // istride
                              idist_fw,           // idist
                              nullptr,            // onembed
                              ostride_fw,         // ostride
                              odist_fw,           // odist
                              CUFFT_C2C,          // type
                              batch_fw,           // batch size
                              &m_workspace_size), // Get workspace size estimate
            "cufftMakePlanMany (forward)");
        spdlog::debug("Forward cuFFT plan created.");

        // --- Backward Plan ---
        // 1D FFT of size m_mbin
        // Batch size: m_nfft * m_nsub * m_nchan
        // Input stride = 1, Input distance = m_mbin
        // Output stride = 1, Output distance = m_mbin (in-place)
        std::array<int, 1> n_bw = {m_mbin};
        int batch_bw            = m_nfft * m_nsub * m_nchan;
        int idist_bw            = m_mbin;
        int odist_bw            = m_mbin;
        int istride_bw          = 1;
        int ostride_bw          = 1;
        size_t workspace_bw = 0; // Need separate query for backward plan size

        DMT_CHECK_CUFFT_CALL(cufftCreate(&m_backward_plan),
                             "cufftCreate (backward)");
        DMT_CHECK_CUFFT_CALL(
            cufftMakePlanMany(m_backward_plan, // plan handle
                              rank,            // rank
                              n_bw.data(),     // n
                              nullptr,         // inembed
                              istride_bw,      // istride
                              idist_bw,        // idist
                              nullptr,         // onembed
                              ostride_bw,      // ostride
                              odist_bw,        // odist
                              CUFFT_C2C,       // type
                              batch_bw,        // batch size
                              &workspace_bw),  // Get workspace size estimate
            "cufftMakePlanMany (backward)");
        // Update overall workspace size needed
        m_workspace_size = std::max(m_workspace_size, workspace_bw);
        spdlog::debug("Backward cuFFT plan created.");
    }

    // Helper to clean up resources
    void cleanup_resources() noexcept {
        if (m_forward_plan != 0) {
            // cufftDestroy is synchronous and can return errors
            cufftDestroy(m_forward_plan); // Ignore error in destructor? Log it?
            m_forward_plan = 0;
        }
        if (m_backward_plan != 0) {
            cufftDestroy(m_backward_plan);
            m_backward_plan = 0;
        }
        if (m_workspace_d != nullptr) {
            // cudaFree is asynchronous wrt host, but sync wrt stream 0 usually
            cudaFree(m_workspace_d); // Ignore error in destructor? Log it?
            m_workspace_d    = nullptr;
            m_workspace_size = 0;
        }
    }

}; // End FFTManager<backend::CPU>::Impl definition

// CUDA-specific constructor implementation
template <>
template <std::same_as<backend::CUDA> P>
FFTManager<backend::CUDA>::FFTManager(
    int nfft, int nsub, int nbin, int mbin, int nchan, int device_id)
    : m_impl(std::make_unique<Impl>(nfft, nsub, nbin, mbin, nchan, device_id)) {
    spdlog::debug("FFTManager<CUDA> object created for device {}.", device_id);
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
void FFTManager<backend::CUDA>::forward_fft(
    cuda::std::span<ComplexTypeCUDA> data, cudaStream_t stream) const {
    m_impl->forward_fft(data, stream);
}
template <>
template <std::same_as<backend::CUDA> P>
void FFTManager<backend::CUDA>::backward_fft(
    cuda::std::span<ComplexTypeCUDA> data, cudaStream_t stream) const {
    m_impl->backward_fft(data, stream);
}
// Explicit instantiation (for linking)
template FFTManager<backend::CUDA>::FFTManager(int, int, int, int, int, int);

} // namespace dmt
