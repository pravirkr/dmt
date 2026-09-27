#pragma once

#include <cassert>
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
 * @brief Generate a delay table for brute-force dedispersion.
 *
 * @param dm_arr DM values (in pc cm^-3).
 * @param nchans Number of channels.
 * @param fch1 First channel (center) frequency (MHz).
 * @param foff Channel (center) frequency offset (MHz).
 * @param tsamp Sampling time (s).
 * @return Delay table (in bins) for brute-force dedispersion.
 */
std::vector<SizeType> generate_delay_table(std::span<const float> dm_arr,
                                           SizeType nchans,
                                           float fch1,
                                           float foff,
                                           float tsamp);

/**
 * @brief Generate a fractional delay table (one float per channel) in bins/(pc
 * cm^-3).
 *
 * For channel ichan, delay = std::nearbyint(dm *
 * fractional_delay_table[ichan]).
 *
 * @param nchans Number of channels.
 * @param fch1 First channel frequency (MHz).
 * @param foff Channel frequency offset (MHz).
 * @param tsamp Sampling time (s).
 * @return Per-channel fractional delay array (size == nchans).
 */
std::vector<float> generate_fractional_delay_table(SizeType nchans,
                                                   float fch1,
                                                   float foff,
                                                   float tsamp);

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

// Compute the optimal minimum overlap based on the dispersion delay
// and the number of channels.
SizeType minimum_overlap(float dm_max,
                         float fcenter,
                         float bw,
                         float tbin,
                         SizeType nsub,
                         SizeType nchan);

// Generate a vector of coherent DMs.
std::vector<float> generate_coherent_dms(
    float dm_min, float dm_max, float fcenter, float bw, float tbin, float tp);

// Precompute the flat (ndm * nchans) per-channel integer sample delay table
// used for inter-channel delay alignment, for every DM in dm_grid.
std::vector<int> generate_dedisperse_shift_table(std::span<const float> dm_grid,
                                                 float f_min,
                                                 float f_max,
                                                 SizeType nchans,
                                                 float tsamp);

// Precompute the flat (ndm * nchans) per-channel history offsets for the
// streaming delay line.
std::vector<SizeType>
generate_dedisperse_offset_table(std::span<const float> dm_grid,
                                 float f_min,
                                 float f_max,
                                 SizeType nchans,
                                 float tsamp);

/**
 * @brief Causal multi-block streaming delay line for inter-channel alignment
 * across coarse-DM trials.
 *
 * For each coarse-DM trial and each channel c, delays incoming samples by
 * S_c = round(kDispConst * dm * (f_min^-2 - f_c^-2) / tsamp) samples relative
 * to f_min. Maintains an isolated FIFO history per trial so consecutive
 * streamed blocks produce continuous output without circular wrapping or buffer
 * overrun.
 */
class ChannelDelayLineCPU {
public:
    ChannelDelayLineCPU() = default;

    void initialise(std::span<const float> dm_grid_coh,
                    float f_min,
                    float f_max,
                    SizeType nchans,
                    float tsamp);

    void process(std::span<const float> in,
                 std::span<float> out,
                 SizeType idm,
                 SizeType nchans,
                 SizeType nsamps);

    void reset_history() noexcept;

    [[nodiscard]] SizeType history_size(SizeType idm) const noexcept;
    [[nodiscard]] const std::vector<int>& get_shifts(SizeType idm) const;
    [[nodiscard]] const std::vector<SizeType>& get_offsets(SizeType idm) const;

private:
    std::vector<std::vector<int>> m_shifts;       // [idm][ichan]
    std::vector<std::vector<SizeType>> m_offsets; // [idm][ichan]
    std::vector<std::vector<float>> m_histories;  // [idm][total_hist_size]
};

// Legacy single-block dedisperse function (kept for backward compatibility).
void dedisperse(float* __restrict__ waterfall,
                SizeType waterfall_size,
                float dm,
                float f_min,
                float f_max,
                SizeType nchans,
                SizeType nsamps,
                float tsamp);

} // namespace dmt::utils
