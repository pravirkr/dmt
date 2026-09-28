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

#include "dmt/algorithms/fdmt.hpp"
#include "dmt/common/backend.hpp"
#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"

namespace dmt::algorithms {

/**
 * @brief Fast Dispersion Measure Transform using Fourier shifts (FDMT-FFT).
 *
 * Implements Algorithm 2 of Zackay & Ofek (2014). Converts time-domain
 * dedispersion shifts into complex phase rotations in the Fourier domain:
 * @f[
 * \widetilde{\text{out}}[k] = \widetilde{\text{in}}_{\text{tail}}[k] +
 * \widetilde{\text{in}}_{\text{head}}[k] \cdot \exp\left(-2\pi i \cdot k \cdot
 * \frac{\Delta t}{N_{\text{fft}}}\right)
 * @f]
 * Reuses @ref dmt::plans::FDMTPlan (same coordinate DAG as FDMT). The FFT
 * length and padding depend on mode:
 * - "roll": cyclic, @f$ N_{\text{fft}} = N_{\text{samps}} @f$ (matches FDMT
 * roll).
 * - "full": zero-pad; @f$ N_{\text{fft}} = N_{\text{samps}} + L +
 * \text{max\_shift} @f$.
 * - "valid": overlap-save of the linear operator across blocks. Overlap length
 * @f$ L = \max(|\Delta t_{\text{min}}|, |\Delta t_{\text{max}}|) @f$.
 *
 * Runs on the backend chosen by the `Exec` constructor argument. The output
 * is compact: nbeams * plan.get_dmt_size() floats. Host memory works on every
 * backend (a GPU backend stages it and blocks); device memory (`DeviceSpan`)
 * on GPU backends only, asynchronous on the given Stream.
 */
class FDMTFFT {
public:
    /**
     * @brief Constructs an FDMTFFT engine with a linear delay grid.
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
     * @param exec Backend and its resources (default: CPU, 1 thread).
     * @param nbeams Number of batched beams (default: 1).
     */
    FDMTFFT(float f_min,
            float f_max,
            SizeType nchans,
            SizeType nsamps,
            float tsamp,
            IndexType dt_max,
            IndexType dt_min      = 0,
            SizeType dt_step      = 1,
            bool use_box_smearing = true,
            std::string_view mode = "valid",
            Exec exec             = {},
            SizeType nbeams       = 1);

    /// @brief Constructs an FDMTFFT engine with a custom delay trial grid
    /// (see the first constructor for the other parameters).
    FDMTFFT(float f_min,
            float f_max,
            SizeType nchans,
            SizeType nsamps,
            float tsamp,
            const std::vector<IndexType>& dt_grid,
            bool use_box_smearing = true,
            std::string_view mode = "valid",
            Exec exec             = {},
            SizeType nbeams       = 1);

    /// @brief Constructs an FDMTFFT engine with a custom DM trial grid in
    /// pc/cm^3 (see the first constructor for the other parameters).
    FDMTFFT(float f_min,
            float f_max,
            SizeType nchans,
            SizeType nsamps,
            float tsamp,
            const std::vector<float>& dm_grid,
            bool use_box_smearing = true,
            std::string_view mode = "valid",
            Exec exec             = {},
            SizeType nbeams       = 1);

    ~FDMTFFT();
    FDMTFFT(FDMTFFT&&) noexcept;
    FDMTFFT& operator=(FDMTFFT&&) noexcept;
    FDMTFFT(const FDMTFFT&)            = delete;
    FDMTFFT& operator=(const FDMTFFT&) = delete;

    /// @brief Read-only reference to underlying FDMT plan
    [[nodiscard]] const plans::FDMTPlan& get_plan() const noexcept;
    /// @brief Number of beams processed together
    [[nodiscard]] SizeType get_nbeams() const noexcept;
    /// @brief Backend this instance runs on.
    [[nodiscard]] Backend backend() const noexcept;
    /// @brief OpenMP threads used (CPU backend); 1 on a GPU backend.
    [[nodiscard]] int nthreads() const noexcept;
    /// @brief Device ordinal used (GPU backend); -1 on the CPU backend.
    [[nodiscard]] int device() const noexcept;

    /**
     * @brief Executes the end-to-end FDMT-FFT transform in a single shot.
     *
     * @param waterfall Input waterfall, beam-major (nbeams * nchans * nsamps).
     * @param dmt Output DM-time, beam-major (nbeams * ndms * dmt_nsamps).
     */
    void execute(std::span<const float> waterfall, std::span<float> dmt);
    /// @brief Executes FDMT-FFT directly on device memory (GPU backends).
    void execute(DeviceSpan<const float> d_waterfall,
                 DeviceSpan<float> d_dmt,
                 Stream stream = {});

    /// @brief Initializes stepper engine with input waterfall and output
    /// buffer (host memory; a GPU backend stages them on the device)
    void reset(std::span<const float> waterfall, std::span<float> dmt);
    /// @brief Initializes stepper from device memory (GPU backends)
    void reset(DeviceSpan<const float> d_waterfall,
               DeviceSpan<float> d_dmt,
               Stream stream = {});

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

    /// @brief Advances stepper forward by given number of levels (`stream`:
    /// GPU queue, default the one given to reset(); empty on the CPU)
    void advance(SizeType levels = 1, Stream stream = {});
    /// @brief Advances execution until specified remaining levels before root
    void advance_until_remaining(SizeType remaining_levels, Stream stream = {});

    /// @brief Host view of all intermediate subband data (beam 0) at the
    /// current level (IFFT computed on demand; on a GPU backend a host
    /// snapshot)
    [[nodiscard]] std::span<const float> view_level_data() const;
    /// @brief Host view of a specific subband's data at the current level
    [[nodiscard]] std::span<const float>
    view_subband_data(SizeType subband_idx) const;
    /// @brief Detailed host view and metadata for a subband at current level
    [[nodiscard]] FDMTSubbandView view_subband(SizeType subband_idx) const;
    /// @brief Device view of the current level (GPU backends)
    [[nodiscard]] DeviceSpan<const float> view_level_data_device() const;
    /// @brief Device view and metadata of a subband (GPU backends)
    [[nodiscard]] FDMTSubbandDeviceView
    view_subband_device(SizeType subband_idx) const;

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
    /// @brief Advances all remaining levels to root and stores the result
    /// in the dmt buffer given to reset() (blocking after a host reset())
    void finalize(Stream stream = {});

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
                 Exec exec             = {},
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
                 Exec exec             = {},
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
                 Exec exec             = {},
                 SizeType nbeams       = 1);

} // namespace dmt::algorithms
