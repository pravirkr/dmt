#pragma once

#include <cassert>
#include <span>
#include <vector>

#ifdef DMT_ENABLE_OPENMP
#include <omp.h>
#endif

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

// Dedisperse the input waterfall.
void dedisperse(float* __restrict__ waterfall,
                SizeType waterfall_size,
                float dm,
                float f_min,
                float f_max,
                SizeType nchans,
                SizeType nsamps,
                float tsamp);

} // namespace dmt::utils
