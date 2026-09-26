#pragma once

/**
 * @file fdmt_fft.hpp
 * @brief Fast Dispersion Measure Transform using Fourier phase rotations
 * (FDMT-FFT).
 */

#include <memory>
#include <span>
#include <string_view>
#include <tuple>
#include <vector>

#ifdef DMT_ENABLE_CUDA
#include <cuda/std/span>
#include <cuda_runtime.h>
#endif // DMT_ENABLE_CUDA

#include "dmt/algorithms/fdmt.hpp"
#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"

namespace dmt::algorithms {

/**
 * @brief Fast Dispersion Measure Transform using Fourier shifts (FDMT-FFT) for
 * CPU execution.
 *
 * Implements Algorithm 2 of Zackay & Ofek (2014). Converts time-domain
 * dedispersion shifts into complex phase rotations in the Fourier domain:
 * @f[
 * \widetilde{\text{out}}[k] = \widetilde{\text{in}}_{\text{tail}}[k] +
 * \widetilde{\text{in}}_{\text{head}}[k] \cdot \exp\left(-2\pi i \cdot k \cdot
 * \frac{\Delta t}{N_{\text{fft}}}\right)
 * @f]
 * Reuses @ref dmt::plans::FDMTPlan (same coordinate DAG as FDMTCPU). The FFT
 * length and padding depend on mode:
 * - "roll": cyclic, @f$ N_{\text{fft}} = N_{\text{samps}} @f$ (matches FDMTCPU
 * roll).
 * - "full": zero-pad; @f$ N_{\text{fft}} = N_{\text{samps}} + L +
 * \text{max\_shift} @f$.
 * - "valid": overlap-save of the linear operator across blocks. Overlap length
 * @f$ L = \max(|\Delta t_{\text{min}}|, |\Delta t_{\text{max}}|) @f$.
 */
class FDMTFFTCPU {
public:
    /**
     * @brief Constructs an FDMTFFTCPU engine with a linear delay grid.
     *
     * @param f_min Bottom edge frequency in MHz.
     * @param f_max Top edge frequency in MHz.
     * @param nchans Number of channels (power of 2).
     * @param nsamps Number of time samples per block.
     * @param tsamp Sampling interval in seconds.
     * @param dt_max Maximum delay trial in samples.
     * @param dt_min Minimum delay trial in samples (default: 0).
     * @param dt_step Stride between delay trials (default: 1).
     * @param use_box_smearing Whether to account for intra-channel smearing
     * (default: true).
     * @param mode Mode: "valid", "full", or "roll" (default: "valid").
     * @param verbose 0 = warnings, 1 = info, 2 = debug (process-wide).
     * @param nthreads Number of OpenMP worker threads (default: 1).
     * @param nbeams Number of batched beams (default: 1).
     */
    FDMTFFTCPU(float f_min,
               float f_max,
               SizeType nchans,
               SizeType nsamps,
               float tsamp,
               IndexType dt_max,
               IndexType dt_min      = 0,
               SizeType dt_step      = 1,
               bool use_box_smearing = true,
               std::string_view mode = "valid",
               int verbose           = 0,
               int nthreads          = 1,
               SizeType nbeams       = 1);

    /**
     * @brief Constructs an FDMTFFTCPU engine with a custom delay trial grid.
     *
     * @param f_min Bottom edge frequency in MHz.
     * @param f_max Top edge frequency in MHz.
     * @param nchans Number of channels.
     * @param nsamps Number of samples.
     * @param tsamp Sampling time in seconds.
     * @param dt_grid Explicit list of delay trials.
     * @param use_box_smearing Whether to account for intra-channel smearing.
     * @param mode Mode: "valid", "full", or "roll".
     * @param verbose 0 = warnings, 1 = info, 2 = debug (process-wide).
     * @param nthreads OpenMP threads.
     * @param nbeams Number of batched beams.
     */
    FDMTFFTCPU(float f_min,
               float f_max,
               SizeType nchans,
               SizeType nsamps,
               float tsamp,
               const std::vector<IndexType>& dt_grid,
               bool use_box_smearing = true,
               std::string_view mode = "valid",
               int verbose           = 0,
               int nthreads          = 1,
               SizeType nbeams       = 1);

    /**
     * @brief Constructs an FDMTFFTCPU engine with a custom DM trial grid.
     *
     * @param f_min Bottom edge frequency in MHz.
     * @param f_max Top edge frequency in MHz.
     * @param nchans Number of channels.
     * @param nsamps Number of samples.
     * @param tsamp Sampling time in seconds.
     * @param dm_grid Explicit list of DM trials in pc/cm^3.
     * @param use_box_smearing Whether to account for intra-channel smearing.
     * @param mode Mode: "valid", "full", or "roll".
     * @param verbose 0 = warnings, 1 = info, 2 = debug (process-wide).
     * @param nthreads OpenMP threads.
     * @param nbeams Number of batched beams.
     */
    FDMTFFTCPU(float f_min,
               float f_max,
               SizeType nchans,
               SizeType nsamps,
               float tsamp,
               const std::vector<float>& dm_grid,
               bool use_box_smearing = true,
               std::string_view mode = "valid",
               int verbose           = 0,
               int nthreads          = 1,
               SizeType nbeams       = 1);

    ~FDMTFFTCPU();
    FDMTFFTCPU(FDMTFFTCPU&&) noexcept;
    FDMTFFTCPU& operator=(FDMTFFTCPU&&) noexcept;
    FDMTFFTCPU(const FDMTFFTCPU&)            = delete;
    FDMTFFTCPU& operator=(const FDMTFFTCPU&) = delete;

    /// @brief Read-only reference to underlying FDMT plan
    [[nodiscard]] const plans::FDMTPlan& get_plan() const noexcept;
    /// @brief Number of beams processed together
    [[nodiscard]] SizeType get_nbeams() const noexcept;

    /**
     * @brief Executes the end-to-end FDMT-FFT transform in a single shot.
     *
     * @param waterfall Input waterfall, beam-major (nbeams * nchans * nsamps).
     * @param dmt Output DM-time, beam-major (nbeams * ndms * dmt_nsamps).
     */
    void execute(std::span<const float> waterfall, std::span<float> dmt);

    /// @brief Initializes stepper engine with input waterfall and scratch
    /// buffer
    void reset(std::span<const float> waterfall, std::span<float> dmt);
    /// @brief Advances stepper forward by given number of levels
    void advance(SizeType levels = 1);
    /// @brief Advances execution until specified remaining levels before root
    void advance_until_remaining(SizeType remaining_levels);

    /// @brief View all intermediate subband data at current level (IFFT
    /// computed on demand)
    [[nodiscard]] std::span<const float> view_level_data() const;
    /// @brief View data slice for a specific subband at current level
    [[nodiscard]] std::span<const float>
    view_subband_data(SizeType subband_idx) const;
    /// @brief Detailed view and metadata for a subband at current level
    [[nodiscard]] FDMTSubbandView view_subband(SizeType subband_idx) const;

    /// @brief Current level index (0 = level 0, total_levels() - 1 = root)
    [[nodiscard]] SizeType current_level() const noexcept;
    /// @brief Total levels in the tree (niters + 1)
    [[nodiscard]] SizeType total_levels() const noexcept;
    /// @brief Remaining levels before root
    [[nodiscard]] SizeType remaining_levels() const noexcept;
    /// @brief Number of active subbands at current level
    [[nodiscard]] SizeType num_subbands() const;
    /// @brief True if root level has been reached
    [[nodiscard]] bool is_finished() const noexcept;
    /// @brief Advances all remaining levels to root
    void finalize();

    /// @brief Theoretical noise variance for a DM trial and boxcar width
    [[nodiscard]] float get_effective_variance(SizeType dm_idx,
                                               SizeType boxcar_width = 1) const;
    /// @brief Theoretical noise sigma for a DM trial and boxcar width
    [[nodiscard]] float get_effective_sigma(SizeType dm_idx,
                                            SizeType boxcar_width = 1) const;
    /// @brief Theoretical noise variance grid across all DM trials
    [[nodiscard]] std::vector<float>
    get_effective_variance_grid(SizeType boxcar_width = 1) const;
    /// @brief Theoretical noise sigma grid across all DM trials
    [[nodiscard]] std::vector<float>
    get_effective_sigma_grid(SizeType boxcar_width = 1) const;

    /// @brief Resets cross-block streaming history to cold start
    void reset_history() noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

/**
 * @brief Convenience function to run FDMT-FFT with a linear delay grid.
 */
[[nodiscard]] std::tuple<std::vector<float>, plans::FDMTPlan>
compute_fdmt_fft(std::span<const float> waterfall,
                 float f_min,
                 float f_max,
                 SizeType nchans,
                 SizeType nsamps,
                 float tsamp,
                 IndexType dt_max,
                 IndexType dt_min      = 0,
                 SizeType dt_step      = 1,
                 bool use_box_smearing = true,
                 std::string_view mode = "valid",
                 int verbose           = 0,
                 int nthreads          = 1,
                 SizeType nbeams       = 1);

/**
 * @brief Convenience function to run FDMT-FFT with a custom delay grid.
 */
[[nodiscard]] std::tuple<std::vector<float>, plans::FDMTPlan>
compute_fdmt_fft(std::span<const float> waterfall,
                 float f_min,
                 float f_max,
                 SizeType nchans,
                 SizeType nsamps,
                 float tsamp,
                 const std::vector<IndexType>& dt_grid,
                 bool use_box_smearing = true,
                 std::string_view mode = "valid",
                 int verbose           = 0,
                 int nthreads          = 1,
                 SizeType nbeams       = 1);

/**
 * @brief Convenience function to run FDMT-FFT with a custom DM grid.
 */
[[nodiscard]] std::tuple<std::vector<float>, plans::FDMTPlan>
compute_fdmt_fft(std::span<const float> waterfall,
                 float f_min,
                 float f_max,
                 SizeType nchans,
                 SizeType nsamps,
                 float tsamp,
                 const std::vector<float>& dm_grid,
                 bool use_box_smearing = true,
                 std::string_view mode = "valid",
                 int verbose           = 0,
                 int nthreads          = 1,
                 SizeType nbeams       = 1);

#ifdef DMT_ENABLE_CUDA
/**
 * @brief Fast Dispersion Measure Transform using Fourier shifts on CUDA GPUs.
 */
class FDMTFFTCUDA {
public:
    /**
     * @brief Constructs an FDMTFFTCUDA engine with a linear delay grid.
     */
    FDMTFFTCUDA(float f_min,
                float f_max,
                SizeType nchans,
                SizeType nsamps,
                float tsamp,
                IndexType dt_max,
                IndexType dt_min      = 0,
                SizeType dt_step      = 1,
                bool use_box_smearing = true,
                std::string_view mode = "valid",
                int verbose           = 0,
                int device_id         = 0,
                SizeType nbeams       = 1);

    /**
     * @brief Constructs an FDMTFFTCUDA engine with a custom delay grid.
     */
    FDMTFFTCUDA(float f_min,
                float f_max,
                SizeType nchans,
                SizeType nsamps,
                float tsamp,
                const std::vector<IndexType>& dt_grid,
                bool use_box_smearing = true,
                std::string_view mode = "valid",
                int verbose           = 0,
                int device_id         = 0,
                SizeType nbeams       = 1);

    /**
     * @brief Constructs an FDMTFFTCUDA engine with a custom DM grid.
     */
    FDMTFFTCUDA(float f_min,
                float f_max,
                SizeType nchans,
                SizeType nsamps,
                float tsamp,
                const std::vector<float>& dm_grid,
                bool use_box_smearing = true,
                std::string_view mode = "valid",
                int verbose           = 0,
                int device_id         = 0,
                SizeType nbeams       = 1);

    ~FDMTFFTCUDA();
    FDMTFFTCUDA(FDMTFFTCUDA&&) noexcept;
    FDMTFFTCUDA& operator=(FDMTFFTCUDA&&) noexcept;
    FDMTFFTCUDA(const FDMTFFTCUDA&)            = delete;
    FDMTFFTCUDA& operator=(const FDMTFFTCUDA&) = delete;

    /// @brief Read-only reference to underlying FDMT plan
    [[nodiscard]] const plans::FDMTPlan& get_plan() const noexcept;
    /// @brief Number of beams processed together
    [[nodiscard]] SizeType get_nbeams() const noexcept;

    /// @brief Executes FDMT-FFT from host memory
    void execute(std::span<const float> waterfall, std::span<float> dmt);
    /// @brief Executes FDMT-FFT directly on device memory
    void execute(cuda::std::span<const float> d_waterfall,
                 cuda::std::span<float> d_dmt,
                 cudaStream_t stream = nullptr);

    /// @brief Initializes stepper from host memory
    void reset(std::span<const float> waterfall, std::span<float> dmt);
    /// @brief Initializes stepper from device memory
    void reset(cuda::std::span<const float> d_waterfall,
               cuda::std::span<float> d_dmt,
               cudaStream_t stream = nullptr);

    template <typename Alloc1 = std::allocator<float>,
              typename Alloc2 = std::allocator<float>>
    void execute(const std::vector<float, Alloc1>& waterfall,
                 std::vector<float, Alloc2>& dmt) {
        execute(std::span<const float>(waterfall), std::span<float>(dmt));
    }

    template <typename Alloc1 = std::allocator<float>,
              typename Alloc2 = std::allocator<float>>
    void reset(const std::vector<float, Alloc1>& waterfall,
               std::vector<float, Alloc2>& dmt) {
        reset(std::span<const float>(waterfall), std::span<float>(dmt));
    }
    /// @brief Advances stepper forward by given levels on GPU
    void advance(SizeType levels = 1, cudaStream_t stream = nullptr);
    /// @brief Advances execution until specified remaining levels before root
    void advance_until_remaining(SizeType remaining_levels,
                                 cudaStream_t stream = nullptr);

    /// @brief View all intermediate subband data at current level on device
    [[nodiscard]] cuda::std::span<const float> view_level_data() const;
    /// @brief View subband data slice at current level on device
    [[nodiscard]] cuda::std::span<const float>
    view_subband_data(SizeType subband_idx) const;
    /// @brief Detailed subband view at current level on device
    [[nodiscard]] FDMTSubbandViewCUDA view_subband(SizeType subband_idx) const;

    /// @brief Current level index
    [[nodiscard]] SizeType current_level() const noexcept;
    /// @brief Total levels in tree
    [[nodiscard]] SizeType total_levels() const noexcept;
    /// @brief Remaining levels before root
    [[nodiscard]] SizeType remaining_levels() const noexcept;
    /// @brief Active subbands at current level
    [[nodiscard]] SizeType num_subbands() const;
    /// @brief True if root level reached
    [[nodiscard]] bool is_finished() const noexcept;
    /// @brief Finalizes remaining levels to root on device
    void finalize(cudaStream_t stream = nullptr);

    /// @brief Theoretical noise variance
    [[nodiscard]] float get_effective_variance(SizeType dm_idx,
                                               SizeType boxcar_width = 1) const;
    /// @brief Theoretical noise sigma
    [[nodiscard]] float get_effective_sigma(SizeType dm_idx,
                                            SizeType boxcar_width = 1) const;
    /// @brief Theoretical noise variance grid
    [[nodiscard]] std::vector<float>
    get_effective_variance_grid(SizeType boxcar_width = 1) const;
    /// @brief Theoretical noise sigma grid
    [[nodiscard]] std::vector<float>
    get_effective_sigma_grid(SizeType boxcar_width = 1) const;

    /// @brief Resets cross-block streaming history on GPU
    void reset_history() noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};
#endif // DMT_ENABLE_CUDA

} // namespace dmt::algorithms
