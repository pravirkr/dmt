#pragma once

/**
 * @file ddmt_fft.hpp
 * @brief Fourier-domain direct dedispersion (DDMT-FFT) with exact fractional
 * delays.
 */

#include <cstdint>
#include <memory>
#include <span>
#include <string_view>

#include "dmt/common/backend.hpp"
#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"

namespace dmt::algorithms {

/// @brief Algorithm used by DDMTFFT for the per-bin channel sum.
enum class DDMTFFTMethod : std::uint8_t {
    /// NUFFT on every uniformly spaced run of at least 32 DM trials (a linear
    /// grid is one run, a DDplan-style piecewise-uniform grid a few), brute
    /// force for the other trials; brute force if there is no such run.
    kAuto = 0,
    /// Exact phase rotation of every channel for every DM trial and bin
    /// (O(ndm * nchans) per bin); any DM grid.
    kBrute = 1,
    /// Type-1 non-uniform FFT over the DM axis (O(nchans * w + ndm log ndm)
    /// per bin and uniform run, accurate to `tolerance`), as kAuto but
    /// required: throws if the grid has no uniform run to use it on.
    kNUFFT = 2,
};

/// @brief Tuning options of DDMTFFT (all have safe defaults).
struct DDMTFFTOptions {
    /// Channel-sum algorithm.
    DDMTFFTMethod method{DDMTFFTMethod::kAuto};
    /// Target relative accuracy of the NUFFT path (1e-7 .. 1e-2).
    double tolerance{1.0E-6};
    /// Samples of look-behind and look-ahead kept around every transformed
    /// segment. A fractional delay applied as a phase ramp is band-limited
    /// (periodic sinc) interpolation, whose tails reach past the segment;
    /// the guard keeps the output away from the segment edges, bounding the
    /// truncation error to roughly 1 / (pi * sqrt(guard)) of the per-sample
    /// noise at the edges and much less inside. 0 is allowed.
    SizeType guard{64};
};

/// @brief Parses "auto", "brute" or "nufft".
[[nodiscard]] DDMTFFTMethod parse_ddmt_fft_method(std::string_view name);
/// @brief Name of a method ("auto", "brute", "nufft").
[[nodiscard]] std::string_view to_string(DDMTFFTMethod method) noexcept;

/**
 * @brief Direct dedispersion in the Fourier domain (FDD) with exact
 * fractional delays.
 *
 * Every channel is shifted by its exact (non-integer) dispersion delay
 * @f$ \tau_{d,c} = \mathrm{DM}_d \, r_c @f$ as a phase ramp,
 * @f[
 * \widetilde{y}_d[k] = \sum_c \widetilde{x}_c[k]\,
 * e^{+2\pi i k \tau_{d,c} / N},
 * @f]
 * i.e. band-limited interpolation instead of DDMT's rounding of each delay
 * to a whole sample. This removes DDMT's intra-trial delay-rounding smearing
 * (up to half a sample per channel) at the cost of Fourier transforms.
 *
 * Built on the same plan and streaming model as DDMT: every execute()
 * dedisperses the retained history followed by the new block, output sample
 * t of trial d being @f$ y_d(t) = \sum_c x_c(t + \tau_{d,c}) @f$. The
 * effective maximum delay is ceil(max tau) + guard (get_max_delay()):
 * get_output_nsamps() is input_nsamps - get_max_delay() on the first block
 * and input_nsamps once warm. The stream is taken as zero before its first
 * sample. Internally blocks are transformed in overlap-save segments.
 *
 * Packed low-bit input is unpacked to float; the output is always float.
 * Host memory works on every backend (a GPU backend stages it); device
 * memory (`DeviceSpan`) on GPU backends only, asynchronous on the given
 * Stream.
 */
class DDMTFFT {
public:
    /// @brief Linear DM grid (see DDMT for the parameters).
    DDMTFFT(float f_min,
            float f_max,
            SizeType nchans,
            float tsamp,
            float dm_max,
            float dm_step,
            float dm_min                       = 0.0F,
            Exec exec                          = {},
            SizeType nbits                     = 32,
            std::span<const uint8_t> kill_mask = {},
            SizeType nbeams                    = 1,
            DDMTFFTOptions options             = {});

    /// @brief Explicit DM trials in pc/cm^3.
    DDMTFFT(float f_min,
            float f_max,
            SizeType nchans,
            float tsamp,
            std::span<const float> dm_arr,
            Exec exec                          = {},
            SizeType nbits                     = 32,
            std::span<const uint8_t> kill_mask = {},
            SizeType nbeams                    = 1,
            DDMTFFTOptions options             = {});

    /// @brief Lina Levin DM grid.
    DDMTFFT(float f_min,
            float f_max,
            SizeType nchans,
            float tsamp,
            const plans::LevinConfig& levin,
            Exec exec                          = {},
            SizeType nbits                     = 32,
            std::span<const uint8_t> kill_mask = {},
            SizeType nbeams                    = 1,
            DDMTFFTOptions options             = {});

    /// @brief From a pre-configured DDMTPlan.
    explicit DDMTFFT(const plans::DDMTPlan& plan,
                     Exec exec              = {},
                     SizeType nbeams        = 1,
                     DDMTFFTOptions options = {});

    ~DDMTFFT();
    DDMTFFT(DDMTFFT&&) noexcept;
    DDMTFFT& operator=(DDMTFFT&&) noexcept;
    DDMTFFT(const DDMTFFT&)            = delete;
    DDMTFFT& operator=(const DDMTFFT&) = delete;

    /// @brief Read-only reference to the underlying plan
    [[nodiscard]] const plans::DDMTPlan& get_plan() const noexcept;
    /// @brief Number of beams batched per execute() call
    [[nodiscard]] SizeType get_nbeams() const noexcept;
    /// @brief Backend this instance runs on.
    [[nodiscard]] Backend backend() const noexcept;
    /// @brief OpenMP threads used (CPU backend); 1 on a GPU backend.
    [[nodiscard]] int nthreads() const noexcept;
    /// @brief Device ordinal used (GPU backend); -1 on the CPU backend.
    [[nodiscard]] int device() const noexcept;
    /// @brief Options in effect (method resolved: never kAuto).
    [[nodiscard]] DDMTFFTOptions get_options() const noexcept;
    /// @brief How the channel sum runs: "nufft" (one uniform grid),
    /// "piecewise_nufft" (one NUFFT per uniform run of the grid, trials
    /// outside such runs by brute force) or "brute".
    [[nodiscard]] std::string_view method_used() const noexcept;
    /// @brief Effective maximum delay in samples: ceil(max tau) + guard.
    [[nodiscard]] SizeType get_max_delay() const noexcept;
    /// @brief A block length (input samples per call) for which at least 80%
    /// of every transform is output: each call transforms the block plus
    /// get_max_delay() + guard samples of context, so blocks much shorter
    /// than the context waste most of the FFT work.
    [[nodiscard]] SizeType get_suggested_nsamps() const noexcept;

    /**
     * @brief Dedisperses a beam-major float32 waterfall (nbeams, nchans,
     * nsamps). Requires get_plan().get_nbits() == 32.
     * @param waterfall Input waterfall buffer.
     * @param dmt Output: shape (nbeams, ndm, get_output_nsamps(nsamps)).
     */
    void execute(std::span<const float> waterfall, std::span<float> dmt);
    /// @brief Device-memory analogue (GPU backends).
    void execute(DeviceSpan<const float> d_waterfall,
                 DeviceSpan<float> d_dmt,
                 Stream stream = {});

    /**
     * @brief Dedisperses a beam-major packed-integer waterfall (nbeams,
     * nchans, nsamps) at get_plan().get_nbits() bits per sample, LSB-first.
     * Samples are unpacked to float; the output is float.
     */
    void execute(std::span<const uint8_t> waterfall_packed,
                 SizeType nsamps,
                 std::span<float> dmt);
    /// @brief Device-memory analogue (GPU backends).
    void execute(DeviceSpan<const uint8_t> d_waterfall_packed,
                 SizeType nsamps,
                 DeviceSpan<float> d_dmt,
                 Stream stream = {});

    /// @brief Time-major packed filterbank (nbeams, nsamps, nchans), as in
    /// DDMT::execute_time_major(); shares the stream history.
    void execute_time_major(std::span<const uint8_t> filterbank_packed,
                            SizeType nsamps,
                            std::span<float> dmt);

    /// @brief Output samples for a block of @p input_nsamps new samples.
    [[nodiscard]] SizeType
    get_output_nsamps(SizeType input_nsamps) const noexcept;
    /// @brief Discards the retained history (cold start).
    void reset_history() noexcept;

    /// @brief Host-path chunk length, stored for API parity with DDMT. The
    /// Fourier engines transform every call whole on every backend:
    /// chunking a call would move the transform boundaries, and with them
    /// the (guard-level) result. 0: default.
    void set_gulp_size(SizeType gulp_size);
    /// @brief Current host-path chunk length in input samples.
    [[nodiscard]] SizeType get_gulp_size() const noexcept;

    /// @brief Floats in a fully warmed-up history: nbeams * nchans *
    /// (get_max_delay() + guard).
    [[nodiscard]] SizeType history_state_size() const noexcept;
    /// @brief Saves the history (throws std::logic_error before warm-up).
    void save_history(std::span<float> out) const;
    /// @brief Restores a history saved by save_history().
    void load_history(std::span<const float> in);
    /// @brief Device-memory analogues (GPU backends).
    void save_history(DeviceSpan<float> d_out, Stream stream = {}) const;
    void load_history(DeviceSpan<const float> d_in, Stream stream = {});

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace dmt::algorithms
