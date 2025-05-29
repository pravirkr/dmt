#pragma once

#include <concepts>
#include <memory>
#include <span>

#ifdef DMT_ENABLE_CUDA
#include <cuda/std/span>
#include <cuda_runtime_api.h>
#endif // DMT_ENABLE_CUDA

#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"

namespace dmt::algorithms {

/**
 * @brief Computes the Coherent Fast Dispersion Measure Transform (CFDMT).
 *
 * This class provides a unified interface for performing the CFDMT on
 * either the CPU or a CUDA-enabled GPU. The backend is chosen at
 * compile time via the Backend template parameter.
 *
 * @tparam Backend A tag type specifying the execution backend (e.g.,
 * dmt::backend::CPU, dmt::backend::CUDA). Instantiation with dmt::backend::CUDA
 * requires the library to be compiled with CUDA support enabled
 * (DMT_ENABLE_CUDA defined).
 */
template <backend::ExecutionBackend Backend = backend::CPU>
class CohFDMT {
public:
    template <std::same_as<backend::CPU> P = Backend>
    CohFDMT(float f_center,
            float bw_sub,
            SizeType nsub,
            float tbin,
            SizeType nbin,
            SizeType nfft,
            float t_p,
            float dm_max,
            float dm_min                = 0.0F,
            SizeType noverlap           = 8192,
            std::string_view data_order = "PRITF",
            bool verbose                = false,
            int nthreads                = 1);

#ifdef DMT_ENABLE_CUDA
    template <std::same_as<backend::CUDA> P = Backend>
    CohFDMT(float f_center,
            float bw_sub,
            SizeType nsub,
            float tbin,
            SizeType nbin,
            SizeType nfft,
            float t_p,
            float dm_max,
            float dm_min                = 0.0F,
            SizeType noverlap           = 8192,
            std::string_view data_order = "PRITF",
            bool verbose                = false,
            int device_id               = 0);
#endif // DMT_ENABLE_CUDA

    ~CohFDMT();
    CohFDMT(CohFDMT&&) noexcept;
    CohFDMT& operator=(CohFDMT&&) noexcept;
    CohFDMT(const CohFDMT&)            = delete;
    CohFDMT& operator=(const CohFDMT&) = delete;

    const plans::CohFDMTPlan& get_plan() const;
    SizeType get_dmt_size() const;

    template <IntegralDataType DataType>
    void execute(std::span<const DataType> data_in, std::span<float> dmt) const;

#ifdef DMT_ENABLE_CUDA
    template <IntegralDataType DataType,
              std::same_as<backend::CUDA> P = Backend>
    void execute(cuda::std::span<const DataType> d_data_in,
                 cuda::std::span<float> d_dmt,
                 cudaStream_t stream = nullptr) const;
#endif // DMT_ENABLE_CUDA

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

// Type aliases for convenience
using CohFDMTCPU = CohFDMT<backend::CPU>;
#ifdef DMT_ENABLE_CUDA
using CohFDMTCUDA = CohFDMT<backend::CUDA>;
#endif // DMT_ENABLE_CUDA

} // namespace dmt::algorithms
