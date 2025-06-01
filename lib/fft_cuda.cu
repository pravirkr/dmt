#include "dmt/utils/fft.hpp"

#include <array>
#include <cstddef>

#include <cuda/std/complex>
#include <cuda/std/span>
#include <cuda_runtime_api.h>
#include <cufft.h>

#include <spdlog/spdlog.h>

#include "dmt/bb_utils_cuda.cuh"
#include "dmt/common/types.hpp"
#include "dmt/cuda_utils.cuh"

namespace dmt::utils {

class FFTManagerCUDA::Impl {
public:
    Impl(int nfft, int nsub, int nbin, int mbin, int nchan, int device_id)
        : m_nfft(nfft),
          m_nsub(nsub),
          m_nbin(nbin),
          m_mbin(mbin),
          m_nchan(nchan),
          m_device_id(device_id) {
        cuda_utils::set_device(m_device_id);
        spdlog::debug("FFTManagerCUDA::Impl: Initialized CUDA device {}",
                      m_device_id);

        try {
            create_plans();
            allocate_workspace();
        } catch (const std::exception& e) {
            spdlog::error("FFTManagerCUDA::Impl: Initialization failed: {}",
                          e.what());
            cleanup_resources();
            throw;
        }
        spdlog::debug("FFTManagerCUDA::Impl: Successfully created with "
                      "nfft={}, nsub={}, nbin={}, mbin={}, nchan={}",
                      m_nfft, m_nsub, m_nbin, m_mbin, m_nchan);
    }

    ~Impl() {
        spdlog::debug("FFTManagerCUDA::Impl: Destroying instance");
        cleanup_resources();
    }
    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&)                 = delete;
    Impl& operator=(Impl&&)      = delete;

    void forward_fft(cuda::std::span<ComplexTypeCUDA> data1,
                     cuda::std::span<ComplexTypeCUDA> data2,
                     cudaStream_t stream) const {
        cuda_utils::check_cuda_call(cufftSetStream(m_forward_plan, stream),
                                    "forward_fft: cufftSetStream");

        auto* d1 = reinterpret_cast<cufftComplex*>(data1.data());
        auto* d2 = reinterpret_cast<cufftComplex*>(data2.data());
        cuda_utils::check_cuda_call(
            cufftExecC2C(m_forward_plan, d1, d1, CUFFT_FORWARD),
            "forward_fft: data1");
        cuda_utils::check_cuda_call(
            cufftExecC2C(m_forward_plan, d2, d2, CUFFT_FORWARD),
            "forward_fft: data2");
        bb_utils::swap_spectrum(data1, data2, m_nbin, m_nfft * m_nsub, stream);
        spdlog::debug("forward_fft: Completed FFT and spectrum swap");
    }

    void backward_fft(cuda::std::span<ComplexTypeCUDA> data1,
                      cuda::std::span<ComplexTypeCUDA> data2,
                      cudaStream_t stream) const {
        cuda_utils::check_cuda_call(cufftSetStream(m_backward_plan, stream),
                                    "backward_fft: cufftSetStream");

        bb_utils::swap_spectrum(data1, data2, m_mbin, m_nfft * m_nsub * m_nchan,
                                stream);

        auto* d1 = reinterpret_cast<cufftComplex*>(data1.data());
        auto* d2 = reinterpret_cast<cufftComplex*>(data2.data());
        cuda_utils::check_cuda_call(
            cufftExecC2C(m_backward_plan, d1, d1, CUFFT_INVERSE),
            "backward_fft: data1");
        cuda_utils::check_cuda_call(
            cufftExecC2C(m_backward_plan, d2, d2, CUFFT_INVERSE),
            "backward_fft: data2");
        spdlog::debug("backward_fft: Completed spectrum swap and FFT");
    }

private:
    const int m_nfft;
    const int m_nsub;
    const int m_nbin;
    const int m_mbin;
    const int m_nchan;
    const int m_device_id;

    cufftHandle m_forward_plan  = 0;
    cufftHandle m_backward_plan = 0;
    void* m_workspace_d         = nullptr;
    size_t m_workspace_size     = 0;

    void create_plans() {
        const int rank = 1; // 1D FFTs

        // --- Forward Plan ---
        // 1D FFT of size m_nbin
        // Batch size: m_nfft * m_nsub
        // Input stride = 1, Input distance = m_nbin
        // Output stride = 1, Output distance = m_nbin (in-place)
        std::array<int, 1> n_fw = {m_nbin};
        const int batch_fw      = m_nfft * m_nsub;
        const int idist_fw      = m_nbin;
        const int odist_fw      = m_nbin;
        const int istride_fw    = 1;
        const int ostride_fw    = 1;

        SizeType forward_workspace_size = 0;
        cuda_utils::check_cuda_call(cufftCreate(&m_forward_plan),
                                    "create_plans: cufftCreate (forward)");
        cuda_utils::check_cuda_call(
            cufftMakePlanMany(m_forward_plan,           // plan handle
                              rank,                     // rank
                              n_fw.data(),              // n
                              nullptr,                  // inembed
                              istride_fw,               // istride
                              idist_fw,                 // idist
                              nullptr,                  // onembed
                              ostride_fw,               // ostride
                              odist_fw,                 // odist
                              CUFFT_C2C,                // type
                              batch_fw,                 // batch size
                              &forward_workspace_size), // Get workspace size
            "create_plans: cufftMakePlanMany (forward)");
        spdlog::debug(
            "create_plans: Forward plan created, workspace size: {} bytes",
            forward_workspace_size);

        // --- Backward Plan ---
        // 1D FFT of size m_mbin
        // Batch size: m_nfft * m_nsub * m_nchan
        // Input stride = 1, Input distance = m_mbin
        // Output stride = 1, Output distance = m_mbin (in-place)
        std::array<int, 1> n_bw = {m_mbin};
        const int batch_bw      = m_nfft * m_nsub * m_nchan;
        const int idist_bw      = m_mbin;
        const int odist_bw      = m_mbin;
        const int istride_bw    = 1;
        const int ostride_bw    = 1;

        SizeType backward_workspace_size = 0;
        cuda_utils::check_cuda_call(cufftCreate(&m_backward_plan),
                                    "create_plans: cufftCreate (backward)");
        cuda_utils::check_cuda_call(
            cufftMakePlanMany(m_backward_plan,           // plan handle
                              rank,                      // rank
                              n_bw.data(),               // n
                              nullptr,                   // inembed
                              istride_bw,                // istride
                              idist_bw,                  // idist
                              nullptr,                   // onembed
                              ostride_bw,                // ostride
                              odist_bw,                  // odist
                              CUFFT_C2C,                 // type
                              batch_bw,                  // batch size
                              &backward_workspace_size), // Get workspace size
            "create_plans: cufftMakePlanMany (backward)");
        spdlog::debug(
            "create_plans: Backward plan created, workspace size: {} bytes",
            backward_workspace_size);

        // Set maximum workspace size
        m_workspace_size =
            std::max(forward_workspace_size, backward_workspace_size);
    }

    // Allocates workspace memory if needed
    void allocate_workspace() {
        if (m_workspace_size > 0) {
            cuda_utils::check_cuda_call(
                cudaMalloc(&m_workspace_d, m_workspace_size),
                "allocate_workspace: cudaMalloc");
            cuda_utils::check_cuda_call(
                cufftSetWorkArea(m_forward_plan, m_workspace_d),
                "allocate_workspace: cufftSetWorkArea (forward)");
            cuda_utils::check_cuda_call(
                cufftSetWorkArea(m_backward_plan, m_workspace_d),
                "allocate_workspace: cufftSetWorkArea (backward)");
            spdlog::debug(
                "allocate_workspace: Allocated {} bytes for cuFFT workspace",
                m_workspace_size);
        }
    }

    // Cleans up CUDA resources
    void cleanup_resources() noexcept {
        if (m_forward_plan != 0) {
            cuda_utils::check_cuda_call(
                cufftDestroy(m_forward_plan),
                "cleanup_resources: cufftDestroy (forward)");
            m_forward_plan = 0;
        }
        if (m_backward_plan != 0) {
            cuda_utils::check_cuda_call(
                cufftDestroy(m_backward_plan),
                "cleanup_resources: cufftDestroy (backward)");
            m_backward_plan = 0;
        }
        if (m_workspace_d != nullptr) {
            cuda_utils::check_cuda_call(cudaFree(m_workspace_d),
                                        "cleanup_resources: cudaFree");
            m_workspace_d = nullptr;
        }
        m_workspace_size = 0;
    }

}; // End FFTManagerCUDA::Impl definition

FFTManagerCUDA::FFTManagerCUDA(
    int nfft, int nsub, int nbin, int mbin, int nchan, int device_id)
    : m_impl(std::make_unique<Impl>(nfft, nsub, nbin, mbin, nchan, device_id)) {
}
FFTManagerCUDA::~FFTManagerCUDA()                               = default;
FFTManagerCUDA::FFTManagerCUDA(FFTManagerCUDA&& other) noexcept = default;
FFTManagerCUDA&
FFTManagerCUDA::operator=(FFTManagerCUDA&& other) noexcept = default;
void FFTManagerCUDA::forward_fft(cuda::std::span<ComplexTypeCUDA> data1,
                                 cuda::std::span<ComplexTypeCUDA> data2,
                                 cudaStream_t stream) const {
    m_impl->forward_fft(data1, data2, stream);
}
void FFTManagerCUDA::backward_fft(cuda::std::span<ComplexTypeCUDA> data1,
                                  cuda::std::span<ComplexTypeCUDA> data2,
                                  cudaStream_t stream) const {
    m_impl->backward_fft(data1, data2, stream);
}

} // namespace dmt::utils
