#pragma once

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
 * Performs the CFDMT algorithm using CPU cores with optional OpenMP
 * parallelization.
 */
class CohFDMTCPU {
public:
    CohFDMTCPU(float f_center,
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
    ~CohFDMTCPU();
    CohFDMTCPU(CohFDMTCPU&&) noexcept;
    CohFDMTCPU& operator=(CohFDMTCPU&&) noexcept;
    CohFDMTCPU(const CohFDMTCPU&)            = delete;
    CohFDMTCPU& operator=(const CohFDMTCPU&) = delete;

    const plans::CohFDMTPlan& get_plan() const noexcept;
    SizeType get_dmt_size() const noexcept;

    template <IntegralDataType DataType>
    void execute(std::span<const DataType> data_in, std::span<float> dmt) const;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

#ifdef DMT_ENABLE_CUDA
/**
 * @brief Computes the Coherent Fast Dispersion Measure Transform (CFDMT).
 *
 * Performs the CFDMT algorithm using CUDA cores with optional CUDA streams.
 */
class CohFDMTCUDA {
public:
    CohFDMTCUDA(float f_center,
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

    ~CohFDMTCUDA();
    CohFDMTCUDA(CohFDMTCUDA&&) noexcept;
    CohFDMTCUDA& operator=(CohFDMTCUDA&&) noexcept;
    CohFDMTCUDA(const CohFDMTCUDA&)            = delete;
    CohFDMTCUDA& operator=(const CohFDMTCUDA&) = delete;

    const plans::CohFDMTPlan& get_plan() const noexcept;
    SizeType get_dmt_size() const noexcept;

    template <IntegralDataType DataType>
    void execute(std::span<const DataType> data_in, std::span<float> dmt) const;
    template <IntegralDataType DataType>
    void execute(cuda::std::span<const DataType> d_data_in,
                 cuda::std::span<float> d_dmt,
                 cudaStream_t stream = nullptr) const;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

#endif // DMT_ENABLE_CUDA

} // namespace dmt::algorithms
