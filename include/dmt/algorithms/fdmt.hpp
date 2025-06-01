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
 * @brief Fast Dispersion Measure Transform (FDMT) for CPU execution.
 *
 * Performs the FDMT algorithm using CPU cores with optional OpenMP
 * parallelization.
 */
class FDMTCPU {
public:
    /**
     * @brief Constructs an FDMT object for the CPU execution.
     *
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
    FDMTCPU(float f_min,
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

    ~FDMTCPU();
    FDMTCPU(FDMTCPU&&) noexcept;
    FDMTCPU& operator=(FDMTCPU&&) noexcept;
    FDMTCPU(const FDMTCPU&)            = delete;
    FDMTCPU& operator=(const FDMTCPU&) = delete;

    /**
     * @brief Gets the FDMT plan details.
     * @return Constant reference to the FDMTPlan object.
     */
    const plans::FDMTPlan& get_plan() const noexcept;

    /**
     * @brief Executes the FDMT transform.
     *
     * @param waterfall Input waterfall data view.
     * @param dmt Output DM-time array view.
     */
    void execute(std::span<const float> waterfall, std::span<float> dmt);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

#ifdef DMT_ENABLE_CUDA
/**
 * @brief Fast Dispersion Measure Transform (FDMT) for CUDA execution.
 *
 * Performs the FDMT algorithm using CUDA-enabled GPUs for high-performance
 * processing.
 */
class FDMTCUDA {
public:
    /**
     * @brief Constructs an FDMT object for the CUDA execution.
     *
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
     * @param device_id CUDA device ID to use (default: 0).
     */
    FDMTCUDA(float f_min,
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

    ~FDMTCUDA();
    FDMTCUDA(FDMTCUDA&&) noexcept;
    FDMTCUDA& operator=(FDMTCUDA&&) noexcept;
    FDMTCUDA(const FDMTCUDA&)            = delete;
    FDMTCUDA& operator=(const FDMTCUDA&) = delete;

    /**
     * @brief Gets the FDMT plan details.
     * @return Constant reference to the FDMTPlan object.
     */
    const plans::FDMTPlan& get_plan() const noexcept;

    /**
     * @brief Executes the FDMT transform using host memory.
     *
     * Input and output data reside on the host. This involves transferring data
     * to/from the device internally.
     *
     * @param waterfall Input waterfall data view (host memory).
     * @param dmt Output DM-time array view (host memory).
     */
    void execute(std::span<const float> waterfall, std::span<float> dmt);

    /**
     * @brief Executes the FDMT transform using device memory.
     *
     * Input and output data pointers refer to memory allocated on the CUDA
     * device. Uses libcudacxx span (`cuda::std::span`) for device memory views.
     * Allows specifying a CUDA stream for asynchronous execution.
     *
     * @param d_waterfall Input waterfall data view (device memory)
     * @param d_dmt Output DM-time array view (device memory)
     * @param stream The CUDA stream to execute the transform on.
     */
    void execute(cuda::std::span<const float> d_waterfall,
                 cuda::std::span<float> d_dmt,
                 cudaStream_t stream = nullptr);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

#endif // DMT_ENABLE_CUDA

} // namespace dmt::algorithms
