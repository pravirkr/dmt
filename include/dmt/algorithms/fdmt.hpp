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
 * @brief Computes the Fast Dispersion Measure Transform (FDMT).
 *
 * This class provides a unified interface for performing the FDMT on
 * either the CPU or a CUDA-enabled GPU. The backend is chosen at
 * compile time via the Backend template parameter.
 *
 * @tparam Backend A tag type specifying the execution backend (e.g.,
 * dmt::backend::CPU, dmt::backend::CUDA). Instantiation with dmt::backend::CUDA
 * requires the library to be compiled with CUDA support enabled
 * (DMT_ENABLE_CUDA defined).
 */
template <backend::ExecutionBackend Backend = backend::CPU>
class FDMT {
public:
    /**
     * @brief Constructs an FDMT object for the CPU backend.
     *
     * @tparam P Constraint ensuring this overload is only for the CPU backend.
     * @param f_min Frequency of the lowest channel (MHz).
     * @param f_max Frequency of the highest channel (MHz).
     * @param nchans Number of frequency channels.
     * @param nsamps Number of time samples per incoming block.
     * @param tsamp Sampling time (s).
     * @param dt_max Maximum delay trial (in samples).
     * @param dt_step Step size for delay trials (default: 1).
     * @param dt_min Minimum delay trial (in samples, default: 0).
     * @param use_history Whether to use previous blocks' data to initialise.
     * @param verbose Enable verbose output.
     * @param nthreads Number of OpenMP threads to use (default: 1).
     */
    template <std::same_as<backend::CPU> P = Backend>
    FDMT(float f_min,
         float f_max,
         SizeType nchans,
         SizeType nsamps,
         float tsamp,
         SizeType dt_max,
         SizeType dt_step = 1,
         SizeType dt_min  = 0,
         bool use_history = false,
         bool verbose     = false,
         int nthreads     = 1);

#ifdef DMT_ENABLE_CUDA
    /**
     * @brief Constructs an FDMT object for the CUDA backend.
     * @note This constructor is only available if compiled with CUDA support
     * (DMT_ENABLE_CUDA is defined).
     * @tparam P Constraint ensuring this overload is only for the CUDA backend.
     * @param device_id CUDA device ID to use (default: 0).
     * (Other parameters as before)
     */
    template <std::same_as<backend::CUDA> P = Backend>
    FDMT(float f_min,
         float f_max,
         SizeType nchans,
         SizeType nsamps,
         float tsamp,
         SizeType dt_max,
         SizeType dt_step = 1,
         SizeType dt_min  = 0,
         bool use_history = false,
         bool verbose     = false,
         int device_id    = 0);
#endif // DMT_ENABLE_CUDA

    ~FDMT();
    FDMT(FDMT&&) noexcept;
    FDMT& operator=(FDMT&&) noexcept;
    FDMT(const FDMT&)            = delete;
    FDMT& operator=(const FDMT&) = delete;

    /**
     * @brief Gets the FDMT plan details.
     * @return Constant reference to the FDMTPlan object.
     */
    const plans::FDMTPlan& get_plan() const;

    /**
     * @brief Executes the FDMT transform using host memory.
     *
     * Input and output data reside on the host. For the CUDA backend,
     * this involves transferring data to/from the device internally.
     *
     * @param waterfall Input waterfall data view (host memory).
     * @param dmt Output DM-time array view (host memory).
     */
    void execute(std::span<const float> waterfall, std::span<float> dmt);

#ifdef DMT_ENABLE_CUDA
    /**
     * @brief Executes the FDMT transform using device memory (CUDA backend
     * only).
     * @note This overload is only available if compiled with CUDA support
     * (DMT_ENABLE_CUDA is defined).
     *
     * Input and output data pointers refer to memory allocated on the CUDA
     * device. Uses libcudacxx span (`cuda::std::span`) for device memory views.
     * Allows specifying a CUDA stream for asynchronous execution.
     *
     * @tparam P Constraint ensuring this overload is only for the CUDA backend.
     * @param d_waterfall Input waterfall data view (device memory)
     * @param d_dmt Output DM-time array view (device memory)
     * @param stream The CUDA stream to execute the transform on.
     */
    template <std::same_as<backend::CUDA> P = Backend>
    void execute(cuda::std::span<const float> d_waterfall,
                 cuda::std::span<float> d_dmt,
                 cudaStream_t stream = nullptr);

#endif // DMT_ENABLE_CUDA

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

// Type aliases for convenience
using FDMTCPU = FDMT<backend::CPU>;
#ifdef DMT_ENABLE_CUDA
using FDMTCUDA = FDMT<backend::CUDA>;
#endif // DMT_ENABLE_CUDA

} // namespace dmt::algorithms
