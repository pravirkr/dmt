#pragma once

#include <cassert>
#include <vector>

#ifdef DMT_ENABLE_OPENMP
#include <omp.h>
#endif

#include "dmt/common/types.hpp"

namespace dmt::utils {
// Compute the frequency-dependent dispersion delay compared to total delay.
float cff(float f1_start, float f1_end, float f2_start, float f2_end);

// Calculate the subband delay based on the frequency range.
SizeType calculate_dt_sub(
    float f_start, float f_end, float f_min, float f_max, SizeType dt);

// Compute the DM conversion factor.
float get_dmconv(float f_min, float f_max, float tsamp);

// Find the closest index in a sorted array.
SizeType find_closest_index(const std::vector<SizeType>& arr_sorted,
                            SizeType val);

// Generate a delay table for dedispersion.
std::vector<SizeType> generate_delay_table(const float* dm_arr,
                                           SizeType dm_count,
                                           float f0,
                                           float df,
                                           SizeType nchans,
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

/**
 * @brief Computes out[k] = arr1[k] for k < offset,
 * out[k] = arr1[k] + arr2[k - offset] for offset <= k < size_in1,
 * out[k] = arr2[k-offset] for size_in1 <= k < size_in1 + min(offset,
 * size_out-size_in1) out[k] = 0.0 otherwise up to size_out
 * @param arr1 Input array 1 (size_in1 elements)
 * @param size_in1 Size of arr1
 * @param arr2 Input array 2 (size_in2 elements)
 * @param size_in2 Size of arr2
 * @param arr_out Output array (size_out elements)
 * @param size_out Size of arr_out
 * @param offset Offset for summing arr2 into arr_out
 */

inline void add_offset_kernel(const float* __restrict__ arr1,
                              SizeType size_in1,
                              const float* __restrict__ arr2,
                              SizeType size_in2,
                              float* __restrict__ arr_out,
                              SizeType size_out,
                              SizeType offset) noexcept {
    // Debug checks using assert (only active when NDEBUG is not defined)
    assert(size_in1 == size_in2 && "Input sizes must be equal");
    assert(size_out >= size_in1 && "Output size must be >= input size");
    assert(offset < size_in1 && "Offset must be < input size");

    const SizeType nsum = size_in1 - offset;
    // Part 1: Direct copy (first 'offset' elements from arr1)
    std::copy_n(arr1, offset, arr_out);
    // Part 2: Vectorized addition (overlap region)
#pragma omp simd
    for (SizeType i = 0; i < nsum; ++i) {
        arr_out[offset + i] = arr1[offset + i] + arr2[i];
    }
    // Part 3: Copy remaining from arr2 if needed
    const SizeType nrest = std::min(offset, size_out - size_in1);
    if (nrest > 0) {
        std::copy_n(arr2 + nsum, nrest, arr_out + size_in1);
    }
    // Part 4: Zero-fill remaining output if needed
    const SizeType filled_so_far = size_in1 + nrest;
    if (filled_so_far < size_out) {
        std::fill(arr_out + filled_so_far, arr_out + size_out, 0.0F);
    }
}

} // namespace dmt::utils
