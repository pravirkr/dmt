#pragma once

/**
 * @file fdmt_fft.hpp
 * @brief Fast Dispersion Measure Transform using Fourier phase rotations
 * (FDMT-FFT).
 */

#include <cstdint>
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
 * - "full": zero-padded linear convolution: @f$ N_{\text{fft}} \ge
 * N_{\text{samps}} + S @f$, S the tree's delay support
 * (FDMTPlan::get_fft_support()).
 * - "valid": overlap-save of the linear operator across blocks. Overlap length
 * @f$ L = \max(|\Delta t_{\text{min}}|, |\Delta t_{\text{max}}|) @f$,
 * @f$ N_{\text{fft}} \ge N_{\text{samps}} + \max(L, S) @f$.
 *
 * Lengths are rounded up to products of 2, 3, 5 and 7. The CPU engine runs the
 * whole tree one tile of frequency bins at a time out of cache-resident
 * buffers, and transforms long valid/full blocks in overlap-save segments (the
 * same linear convolution; the stepper always uses the single transform).
 *
 * Fractional delays are the nature of the Fourier-domain tree and the
 * default (`fractional_delays = true`): every merge shifts its head by a
 * real-valued delay instead of the tree's rounded integer one (in the Fourier
 * domain a fractional shift costs the same): the shift that aligns the mean
 * timing of the head's channels with the tail's at the node's DM, given the
 * integer delay grids of the children (a least-squares alignment the
 * integer tree approximates by rounding). The shift is band-limited
 * interpolation, which needs a guard of fdmt_fft::kFractionalGuard (64)
 * samples of context on both sides: the overlap-save history grows by the
 * exact support plus the guard, and in valid mode the output lags the input
 * by the guard (get_output_latency()), so that the newest samples of a block
 * serve as look-ahead and streaming equals one long call. Level 0 (the
 * per-channel delay grid) keeps whole-sample boxcars. Output shapes are
 * unchanged.
 *
 * `fractional_delays = false` rounds every merge shift as the time-domain
 * FDMT does. That mode exists only for equivalence tests against FDMT
 * (identical output through the Fourier domain); rounding the shifts gives
 * up what the Fourier domain is for, so do not use it for searching.
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
     * @param fractional_delays True by default, and by the nature of the
     * Fourier-domain tree: every merge shifts its head by a real-valued,
     * least-squares delay (a phase ramp, i.e. band-limited interpolation)
     * instead of the rounded integer delay of the time-domain FDMT tree.
     * Valid mode then lags the input by fdmt_fft::kFractionalGuard samples
     * (get_output_latency()). `false` rounds the shifts like FDMT and exists
     * only for equivalence tests against FDMT. See the class notes.
     * @param kill_mask Optional per-channel mask (size nchans, 1 = keep,
     * 0 = kill; empty keeps every channel). Killed channels read as zero.
     */
    FDMTFFT(float f_min,
            float f_max,
            SizeType nchans,
            SizeType nsamps,
            float tsamp,
            IndexType dt_max,
            IndexType dt_min       = 0,
            SizeType dt_step       = 1,
            bool use_box_smearing  = true,
            std::string_view mode  = "valid",
            Exec exec                          = {},
            SizeType nbeams                    = 1,
            bool fractional_delays             = true,
            std::span<const uint8_t> kill_mask = {});

    /// @brief Constructs an FDMTFFT engine with a custom delay trial grid
    /// (see the first constructor for the other parameters).
    FDMTFFT(float f_min,
            float f_max,
            SizeType nchans,
            SizeType nsamps,
            float tsamp,
            const std::vector<IndexType>& dt_grid,
            bool use_box_smearing  = true,
            std::string_view mode  = "valid",
            Exec exec                          = {},
            SizeType nbeams                    = 1,
            bool fractional_delays             = true,
            std::span<const uint8_t> kill_mask = {});

    /// @brief Constructs an FDMTFFT engine with a custom DM trial grid in
    /// pc/cm^3 (see the first constructor for the other parameters).
    FDMTFFT(float f_min,
            float f_max,
            SizeType nchans,
            SizeType nsamps,
            float tsamp,
            const std::vector<float>& dm_grid,
            bool use_box_smearing  = true,
            std::string_view mode  = "valid",
            Exec exec                          = {},
            SizeType nbeams                    = 1,
            bool fractional_delays             = true,
            std::span<const uint8_t> kill_mask = {});

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
    /// @brief True if merges use fractional delays (the default; false is
    /// the FDMT-equivalence test mode).
    [[nodiscard]] bool fractional_delays() const noexcept;
    /// @brief Samples the output lags the input: fdmt_fft::kFractionalGuard
    /// in valid mode with fractional delays (the interpolation look-ahead),
    /// else 0. Output sample o of a block is stream time
    /// block_start - latency + o.
    [[nodiscard]] SizeType get_output_latency() const noexcept;
    /// @brief Valid mode: a block length (samples per call) for which at
    /// least 80% of every transform is output (each transform also carries
    /// the overlap history); the plan's nsamps in the other modes.
    [[nodiscard]] SizeType get_suggested_nsamps() const noexcept;

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

    /**
     * @brief Executes FDMT-FFT on packed low-bit unsigned input, as
     * FDMT::execute() does: each channel row holds nsamps samples of
     * `nbits` bits (LSB first within a byte for nbits < 8, little-endian
     * for 16), padded to a whole byte: layout (nbeams, nchans,
     * ceil(nsamps * nbits / 8)) bytes. The samples are read as floats
     * straight into the transform rows; the output equals execute() on the
     * same values converted to float. On a GPU backend only the packed
     * bytes are copied to the device.
     * @param nbits Sample width: 1, 2, 4, 8 or 16.
     * @throws std::invalid_argument for another nbits or a size mismatch.
     */
    void execute(std::span<const uint8_t> waterfall_packed,
                 SizeType nbits,
                 std::span<float> dmt);
    /// @brief Device-memory analogue of the packed execute() (GPU backends).
    void execute(DeviceSpan<const uint8_t> d_waterfall_packed,
                 SizeType nbits,
                 DeviceSpan<float> d_dmt,
                 Stream stream = {});
    /// @brief Time-major packed filterbank (nbeams, nsamps, ceil(nchans *
    /// nbits / 8) bytes), as DDMT::execute_time_major(); shares the stream
    /// history with the other overloads.
    void execute_time_major(std::span<const uint8_t> filterbank_packed,
                            SizeType nbits,
                            std::span<float> dmt);

    /// @brief Initializes stepper engine with input waterfall and output
    /// buffer (host memory; a GPU backend stages them on the device)
    void reset(std::span<const float> waterfall, std::span<float> dmt);
    /// @brief Initializes stepper from device memory (GPU backends)
    void reset(DeviceSpan<const float> d_waterfall,
               DeviceSpan<float> d_dmt,
               Stream stream = {});
    /// @brief Packed analogue of reset() (layout as the packed execute()).
    /// The packed block must stay valid until finalize().
    void reset(std::span<const uint8_t> waterfall_packed,
               SizeType nbits,
               std::span<float> dmt);
    /// @brief Packed analogue of the device-memory reset() (GPU backends).
    void reset(DeviceSpan<const uint8_t> d_waterfall_packed,
               SizeType nbits,
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

    /// @brief Floats of the valid-mode streaming history (nbeams * nchans *
    /// overlap; 0 in full and roll mode), as used by save_history() and
    /// load_history(). The layout is [beam][channel][sample] on every
    /// backend, so a history can move between backends.
    [[nodiscard]] SizeType history_state_size() const noexcept;
    /// @brief Copies the streaming history out (e.g. to multiplex several
    /// streams through one instance). @throws std::invalid_argument if
    /// out.size() != history_state_size().
    void save_history(std::span<float> out) const;
    /// @brief Restores a history saved by save_history(), resuming that
    /// stream. @throws std::invalid_argument on a size mismatch.
    void load_history(std::span<const float> in);
    /// @brief Device-memory analogues (GPU backends), enqueued on @p stream.
    void save_history(DeviceSpan<float> d_out, Stream stream = {}) const;
    void load_history(DeviceSpan<const float> d_in, Stream stream = {});

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
                 IndexType dt_min       = 0,
                 SizeType dt_step       = 1,
                 bool use_box_smearing  = true,
                 std::string_view mode  = "valid",
                 Exec exec              = {},
                 SizeType nbeams        = 1,
                 bool fractional_delays = true);

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
                 bool use_box_smearing  = true,
                 std::string_view mode  = "valid",
                 Exec exec              = {},
                 SizeType nbeams        = 1,
                 bool fractional_delays = true);

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
                 bool use_box_smearing  = true,
                 std::string_view mode  = "valid",
                 Exec exec              = {},
                 SizeType nbeams        = 1,
                 bool fractional_delays = true);

} // namespace dmt::algorithms
