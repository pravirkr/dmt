#pragma once

#include <bit>
#include <cassert>
#include <cstdint>
#include <span>
#include <vector>

#include "dmt/common/types.hpp"

namespace dmt::utils {
/**
 * @brief Compute the ratio dispersion delay of frequency range.
 *
 * @param f_start Start frequency (MHz).
 * @param f_end End frequency (MHz).
 * @param f_min Minimum frequency (MHz).
 * @param f_max Maximum frequency (MHz).
 * @return The ratio of the dispersion delay of the frequency range to the
 * dispersion delay of the entire frequency range.
 */
float cff(float f_start, float f_end, float f_min, float f_max);

/**
 * @brief Calculate the subband delay (in bins) for a given frequency range.
 *
 * @param f_start Start frequency (MHz).
 * @param f_end End frequency (MHz).
 * @param f_min Minimum frequency (MHz).
 * @param f_max Maximum frequency (MHz).
 * @param dt Delay time (in bins).
 * @return The subband delay (in bins) for the given frequency range.
 */
SizeType calculate_dt_sub(
    float f_start, float f_end, float f_min, float f_max, SizeType dt);

/**
 * @brief Compute the DM conversion factor (in bins/DM unit).
 *
 * @param f_min Minimum frequency (MHz).
 * @param f_max Maximum frequency (MHz).
 * @param tsamp Sampling time (s).
 * @return A factor by which the DM is converted to a delay in bins.
 */
float get_dmconv(float f_min, float f_max, float tsamp);

/**
 * @brief  Locate the index of the element **closest** to @p val in a sorted
 * array.
 *
 * Breaks ties toward the *lower* index, i.e. when `val` lies exactly in the
 * middle of two equal‑distant neighbours the one with the smaller index is
 * returned.
 *
 * @param arr_sorted Monotonically non‑decreasing array.
 * @param val Search value.
 * @return Index of the element **closest** to @p val in the sorted array.
 *
 * @throws  std::invalid_argument if the array is empty.
 *
 * **Time Complexity**: **O(log n)** comparisons via `std::ranges::lower_bound`.
 */
SizeType find_nearest_sorted_idx(std::span<const SizeType> arr_sorted,
                                 SizeType val);
SizeType find_nearest_sorted_idx(std::span<const IndexType> arr_sorted,
                                 IndexType val);

/**
 * @brief Time offset of the brute-force delay tables referenced to @p f_ref:
 * the most negative dispersion delay of any channel centre behind f_ref over
 * the DM trials (channels above f_ref arrive early), negated and rounded
 * up; 0 when no channel lies above f_ref.
 *
 * Channel c is centred at f_min + (c + 1/2) df (ascending band). Output
 * sample k of a delay-table engine is the arrival time at f_ref of input
 * sample k + offset.
 *
 * @param dm_arr DM values (pc cm^-3), finite and >= 0.
 * @param nchans Number of channels.
 * @param f_min Bottom band edge (MHz).
 * @param df Channel width (MHz), > 0.
 * @param tsamp Sampling time (s).
 * @param f_ref Reference frequency (MHz), finite and > 0 (DDMTPlan allows
 *        only f_min or f_max).
 */
SizeType generate_delay_offset(std::span<const float> dm_arr,
                               SizeType nchans,
                               float f_min,
                               float df,
                               float tsamp,
                               double f_ref);

/**
 * @brief Delay table for brute-force dedispersion, referenced to @p f_ref.
 *
 * Row-major (ndm x nchans): entry (d, c) = offset + round(dm[d] * K
 * (f_c^-2 - f_ref^-2) / tsamp) with offset = generate_delay_offset(), so
 * out[k] = sum_c in_c[k + delay(d, c)] is the sum for f_ref arrival at input
 * sample k + offset. Entries are >= 0.
 *
 * @param dm_arr DM values (pc cm^-3).
 * @param nchans Number of channels.
 * @param f_min Bottom band edge (MHz).
 * @param df Channel width (MHz), > 0.
 * @param tsamp Sampling time (s).
 * @param f_ref Reference frequency (MHz), finite and > 0 (DDMTPlan allows
 *        only f_min or f_max).
 */
std::vector<SizeType> generate_delay_table(std::span<const float> dm_arr,
                                           SizeType nchans,
                                           float f_min,
                                           float df,
                                           float tsamp,
                                           double f_ref);

/**
 * @brief Per-channel derivative of the delay table with respect to DM, in
 * bins / (pc cm^-3): K (f_c^-2 - f_ref^-2) / tsamp, with channel c centred
 * at f_min + (c + 1/2) df. Negative for channels above f_ref.
 *
 * delay(d, c) = offset + nearbyint(dm[d] * fractional_delay_table[c]) (up to
 * the float rounding of the rate).
 *
 * @param nchans Number of channels.
 * @param f_min Bottom band edge (MHz).
 * @param df Channel width (MHz), > 0.
 * @param tsamp Sampling time (s).
 * @param f_ref Reference frequency (MHz), finite and > 0 (DDMTPlan allows
 *        only f_min or f_max).
 */
std::vector<float> generate_fractional_delay_table(
    SizeType nchans, float f_min, float df, float tsamp, double f_ref);

/**
 * @brief Generate an optimal DM trial grid using Lina Levin's pulse-broadening
 * tolerance rule (Levin 2012).
 *
 * Note: This function generates continuous physical DM trials (pc cm^-3)
 * parameterised by pulse-broadening tolerance (tol > 1.0) and evaluated at band
 * center. For FDMT discrete integer delay trials (dt_arr) bounded by fractional
 * S/N loss (max_snr_loss < 1.0) evaluated at band edge f_min, see the Python
 * module `dmtlib.grid` (`generate_optimal_dt_grid`).
 *
 * @param dm_start Starting DM (pc cm^-3).
 * @param dm_end Ending DM (pc cm^-3).
 * @param tsamp Sampling interval (s).
 * @param pulse_width Intrinsic pulse width (s).
 * @param f_min Lowest frequency (MHz).
 * @param f_max Highest frequency (MHz).
 * @param nchans Number of frequency channels.
 * @param tol Smearing tolerance factor (must be > 1.0, typically 1.15 to 1.25).
 * @return Monotonically increasing vector of DM trials starting at dm_start.
 * Normally ends at a value >= dm_end; if the step size collapses to zero
 * before reaching dm_end (an extreme tolerance/pulse_width/tsamp
 * combination with no valid further step), the grid stops short and a
 * warning is logged -- callers that need a hard guarantee should check
 * `grid.back() >= dm_end` themselves.
 */
std::vector<float> generate_levin_dm_grid(float dm_start,
                                          float dm_end,
                                          float tsamp,
                                          float pulse_width,
                                          float f_min,
                                          float f_max,
                                          SizeType nchans,
                                          float tol);

/**
 * @brief DDplan-style piecewise-uniform version of generate_levin_dm_grid().
 *
 * The range is split where the Levin step doubles; each segment is uniformly
 * spaced at the smallest Levin step inside it (never coarser than the Levin
 * grid, so at least its sensitivity), refined to at least @p min_run trials.
 * DDMTFFT runs one NUFFT per segment. Ends at the Levin grid's last trial or
 * just past it.
 */
std::vector<float> generate_levin_dm_grid_piecewise(float dm_start,
                                                    float dm_end,
                                                    float tsamp,
                                                    float pulse_width,
                                                    float f_min,
                                                    float f_max,
                                                    SizeType nchans,
                                                    float tol,
                                                    SizeType min_run = 32);

/**
 * @brief True if @p v is finite, tested on the bit pattern through a volatile
 * read. Release builds use -ffast-math, under which std::isfinite, NaN/Inf
 * comparisons, and non-volatile bit_cast checks may be folded away.
 */
[[nodiscard]] inline bool is_finite_bits(double v) noexcept {
    const volatile std::uint64_t bits = std::bit_cast<std::uint64_t>(v);
    constexpr std::uint64_t kExp      = 0x7FF0000000000000ULL;
    return (bits & kExp) != kExp;
}

[[nodiscard]] inline bool is_finite_bits(float v) noexcept {
    const volatile std::uint32_t bits = std::bit_cast<std::uint32_t>(v);
    constexpr std::uint32_t kExp      = 0x7F800000U;
    return (bits & kExp) != kExp;
}

/**
 * @brief Smallest FFT-friendly length >= n: even, and a product of the
 * primes 2, 3, 5 and 7 only (FFTW and cuFFT have fast codelets for these).
 */
SizeType next_fft_size(SizeType n);

} // namespace dmt::utils
