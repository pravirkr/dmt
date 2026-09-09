#pragma once

#include <memory>
#include <span>
#include <string_view>

#ifdef DMT_ENABLE_CUDA
#include <cuda/std/span>
#include <cuda_runtime_api.h>
#endif // DMT_ENABLE_CUDA

#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"

namespace dmt::algorithms {

/**
 * @brief Zero-copy read-only view of a sub-band's intermediate DM-time
 * transform.
 */
struct FDMTSubbandView {
    std::span<const float> data; ///< Contiguous DM-time data (ndt * nsamps)
    SizeType subband_idx;        ///< (0 <= subband_idx < num_subbands)
    SizeType ndt;                ///< Number of DM delay trials
    SizeType nsamps;             ///< Number of time samples per delay trial
    float f_start;               ///< Start frequency of this sub-band (MHz)
    float f_end;                 ///< End frequency of this sub-band (MHz)
    std::span<const SizeType> dt_grid; ///< Delay trials in samples
};

/**
 * @brief Fast Dispersion Measure Transform (FDMT) for CPU execution.
 *
 * Performs the FDMT algorithm using CPU cores with optional OpenMP
 * parallelization. Supports both single-shot transformation and stepped
 * execution for intermediate sub-band inspection.
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
     * @param dt_min Minimum delay trial (in samples, default: 0).
     * @param use_box_smearing Whether to account for intra-channel smearing
     * using boxcar summation (default: true).
     * @param mode Mode of the FDMT transform. Available modes are:
     * - "full": Full FDMT transform of every sample in the input waterfall.
     * - "valid": Valid FDMT transform for samples upto dt_max.
     * - "roll": Roll FDMT transform using rotation of the input waterfall.
     * (default: "full").
     * @param verbose Enable verbose output.
     * @param nthreads Number of OpenMP threads to use (default: 1).
     */
    FDMTCPU(float f_min,
            float f_max,
            SizeType nchans,
            SizeType nsamps,
            float tsamp,
            SizeType dt_max,
            SizeType dt_min       = 0,
            bool use_box_smearing = true,
            std::string_view mode = "full",
            bool verbose          = false,
            int nthreads          = 1);

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
     * @brief Executes the full FDMT transform in a single shot.
     *
     * @param waterfall Input waterfall data view.
     * @param dmt Output DM-time array view.
     */
    void execute(std::span<const float> waterfall, std::span<float> dmt);

    // =========================================================================
    // Stepper / Hierarchical DP Engine API
    // =========================================================================

    /**
     * @brief Resets and initializes the stepper with a new waterfall block and
     * caller-provided dmt buffer for zero-allocation ping-pong storage (Level
     * 0).
     *
     * @param waterfall Input waterfall data view (nchans * nsamps).
     * @param dmt Output DM-time array view (size >= plan.get_buffer_size()).
     *            Used as one of the two ping-pong scratch buffers and receives
     *            the final transform at finalize().
     */
    void reset(std::span<const float> waterfall, std::span<float> dmt);

    /**
     * @brief Steps forward by a given number of levels.
     * @param levels Number of levels to advance (default: 1).
     */
    void advance(SizeType levels = 1);

    /**
     * @brief Advances execution until N levels remain before the root.
     *
     * Examples:
     * - remaining_levels = 1: Stops 1 level before root (2 children sub-bands
     * remaining).
     * - remaining_levels = 2: Stops 2 levels before root (4 children sub-bands
     * remaining).
     * - remaining_levels = 0: Advances to completion (root level).
     *
     * @param remaining_levels Target number of levels remaining before root.
     */
    void advance_until_remaining(SizeType remaining_levels);

    /**
     * @brief Read-only view of all intermediate data across all sub-bands at
     * current level.
     */
    [[nodiscard]] std::span<const float> view_level_data() const;

    /**
     * @brief Read-only view of the data slice for a specific sub-band at
     * current level.
     * @param subband_idx Index of the sub-band (0 <= subband_idx <
     * num_subbands()).
     */
    [[nodiscard]] std::span<const float>
    view_subband_data(SizeType subband_idx) const;

    /**
     * @brief Detailed read-only view and metadata for a specific sub-band at
     * current level.
     * @param subband_idx Index of the sub-band (0 <= subband_idx <
     * num_subbands()).
     */
    [[nodiscard]] FDMTSubbandView view_subband(SizeType subband_idx) const;

    /**
     * @brief Current level index (0 = initialised waterfall, total_levels() - 1
     * = root).
     */
    [[nodiscard]] SizeType current_level() const noexcept;

    /**
     * @brief Total number of levels (niters + 1).
     */
    [[nodiscard]] SizeType total_levels() const noexcept;

    /**
     * @brief Number of levels remaining before root (total_levels() - 1 -
     * current_level()).
     */
    [[nodiscard]] SizeType remaining_levels() const noexcept;

    /**
     * @brief Number of active sub-bands at the current level.
     */
    [[nodiscard]] SizeType num_subbands() const;

    /**
     * @brief Check if execution has reached the final root level.
     */
    [[nodiscard]] bool is_finished() const noexcept;

    /**
     * @brief Finalizes execution to root level.
     *
     * Advances all remaining levels to the root. The final root transform is
     * guaranteed to reside in the dmt buffer provided during reset().
     */
    void finalize();

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

// Convenience function FDMT a block of data
[[nodiscard]] std::tuple<std::vector<float>, plans::FDMTPlan>
compute_fdmt(std::span<const float> waterfall,
             float f_min,
             float f_max,
             SizeType nchans,
             SizeType nsamps,
             float tsamp,
             SizeType dt_max,
             SizeType dt_min       = 0,
             bool use_box_smearing = true,
             std::string_view mode = "full",
             bool verbose          = false,
             int nthreads          = 1);

#ifdef DMT_ENABLE_CUDA
/**
 * @brief Zero-copy read-only view of a sub-band's intermediate DM-time
 * transform residing in CUDA device memory.
 */
struct FDMTSubbandViewCUDA {
    cuda::std::span<const float> data; ///< Contiguous DM-time (ndt * nsamps)
    SizeType subband_idx;              ///< (0 <= subband_idx < num_subbands)
    SizeType ndt;                      ///< Number of DM delay trials
    SizeType nsamps; ///< Number of time samples per delay trial
    float f_start;   ///< Start frequency of this sub-band (MHz)
    float f_end;     ///< End frequency of this sub-band (MHz)
    std::span<const SizeType> dt_grid; ///< Delay trials in samples
};

using FDMTCUDASubbandView = FDMTSubbandViewCUDA;

/**
 * @brief Fast Dispersion Measure Transform (FDMT) for CUDA execution.
 *
 * Performs the FDMT algorithm using CUDA-enabled GPUs for high-performance
 * processing. Supports single-shot execution (host/device memory) as well as
 * interactive hierarchical stepped execution on device memory for intermediate
 * sub-band matched-filtering pipelines.
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
     * @param dt_min Minimum delay trial (in samples, default: 0).
     * @param use_box_smearing Whether to account for intra-channel smearing
     * using boxcar summation (default: true).
     * @param mode Mode of the FDMT transform. Available modes are:
     * - "full": Full FDMT transform of every sample in the input waterfall.
     * - "valid": Valid FDMT transform for samples upto dt_max.
     * - "roll": Roll FDMT transform using rotation of the input waterfall.
     * (default: "full").
     * @param verbose Enable verbose output.
     * @param device_id CUDA device ID to use (default: 0).
     */
    FDMTCUDA(float f_min,
             float f_max,
             SizeType nchans,
             SizeType nsamps,
             float tsamp,
             SizeType dt_max,
             SizeType dt_min       = 0,
             bool use_box_smearing = true,
             std::string_view mode = "full",
             bool verbose          = false,
             int device_id         = 0);

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
     *
     * @note Work is submitted asynchronously to @p stream. Call
     *       cudaStreamSynchronize(stream) (or an equivalent ordering guarantee)
     *       before reading @p d_dmt on the host or reusing the buffers.
     */
    void execute(cuda::std::span<const float> d_waterfall,
                 cuda::std::span<float> d_dmt,
                 cudaStream_t stream = nullptr);

    /**
     * @brief Resets and initializes the stepped execution state using device
     * memory.
     *
     * Prepares the internal pipeline buffers for hierarchical stepping,
     * binding the input waterfall and output DM-time buffers directly on the
     * device.
     *
     * @param d_waterfall 2D input array on device (nchans * nsamps).
     * @param d_dmt 2D output array on device (capacity at least
     * get_dmt_size()).
     * @param stream CUDA stream for asynchronous execution (default: nullptr).
     * @throws std::invalid_argument If the input or output buffers are too
     * small.
     *
     * @note Kernels and memory copies are queued on @p stream. Synchronize
     *       before consuming @p d_dmt or calling reset() again on another
     *       stream unless ordering is guaranteed externally.
     */
    void reset(cuda::std::span<const float> d_waterfall,
               cuda::std::span<float> d_dmt,
               cudaStream_t stream = nullptr);

    /**
     * @brief Advances execution by the specified number of levels on the CUDA
     * device.
     *
     * @param levels Number of hierarchical levels to advance (default: 1).
     *               If levels exceeds the remaining levels, advances to the
     * root.
     * @param stream CUDA stream to use (default: nullptr, uses stream from
     * reset if nullptr).
     * @throws std::logic_error If reset() was not called first.
     *
     * @note Asynchronous on the active CUDA stream; synchronize before reading
     *       results.
     */
    void advance(SizeType levels = 1, cudaStream_t stream = nullptr);

    /**
     * @brief Advances execution until a given number of levels remain before
     * the root.
     *
     * Commonly used to pause 1 level before root (2 sub-bands) or 2 levels
     * before root (4 sub-bands) for sub-band search / matched filtering.
     *
     * @param remaining_levels Target remaining levels before the root.
     * @param stream CUDA stream to use (default: nullptr, uses stream from
     * reset if nullptr).
     * @throws std::logic_error If reset() was not called first.
     *
     * @note Asynchronous on the active CUDA stream; synchronize before reading
     *       results.
     */
    void advance_until_remaining(SizeType remaining_levels,
                                 cudaStream_t stream = nullptr);

    /**
     * @brief Obtains a read-only device span of the entire buffer at the
     * current level.
     *
     * @return Read-only device view of the active state buffer.
     * @throws std::logic_error If reset() was not called first.
     */
    cuda::std::span<const float> view_level_data() const;

    /**
     * @brief Obtains a read-only device span of a specific sub-band at the
     * current level.
     *
     * @param subband_idx Sub-band index in [0, num_subbands()).
     * @return Read-only device view of the sub-band's DM-time array (ndt *
     * nsamps floats).
     * @throws std::logic_error If reset() was not called first.
     * @throws std::out_of_range If subband_idx >= num_subbands().
     */
    cuda::std::span<const float> view_subband_data(SizeType subband_idx) const;

    /**
     * @brief Obtains a rich descriptor view of a specific sub-band on the
     * device.
     *
     * Provides frequency boundaries, DM trial counts, and a device span.
     *
     * @param subband_idx Sub-band index in [0, num_subbands()).
     * @return FDMTSubbandViewCUDA containing sub-band metadata and device span.
     * @throws std::logic_error If reset() was not called first.
     * @throws std::out_of_range If subband_idx >= num_subbands().
     */
    FDMTSubbandViewCUDA view_subband(SizeType subband_idx) const;

    /**
     * @brief Current hierarchical level of the transform (0 = initial
     * waterfall).
     */
    SizeType current_level() const noexcept;

    /**
     * @brief Total number of levels in the hierarchy (niters + 1).
     */
    SizeType total_levels() const noexcept;

    /**
     * @brief Number of levels remaining before reaching the root level.
     */
    SizeType remaining_levels() const noexcept;

    /**
     * @brief Number of sub-bands present at the current level.
     * @throws std::logic_error If reset() was not called first.
     */
    SizeType num_subbands() const;

    /**
     * @brief Checks if execution has reached the root level.
     */
    bool is_finished() const noexcept;

    /**
     * @brief Advances through any remaining levels to the root level.
     *
     * @param stream CUDA stream to use (default: nullptr, uses stream from
     * reset if nullptr).
     * @throws std::logic_error If reset() was not called first.
     *
     * @note Asynchronous on the active CUDA stream; synchronize before reading
     *       the final transform in @p d_dmt passed to reset().
     */
    void finalize(cudaStream_t stream = nullptr);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

// Convenience function FDMT a block of data
std::vector<float> compute_fdmt_cuda(std::span<const float> waterfall,
                                     float f_min,
                                     float f_max,
                                     SizeType nchans,
                                     SizeType nsamps,
                                     float tsamp,
                                     SizeType dt_max,
                                     SizeType dt_min       = 0,
                                     bool use_box_smearing = true,
                                     std::string_view mode = "full",
                                     bool verbose          = false,
                                     int device_id         = 0);

#endif // DMT_ENABLE_CUDA

} // namespace dmt::algorithms
