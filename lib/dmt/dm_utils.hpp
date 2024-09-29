#pragma once

#include <algorithm>
#include <complex>
#include <cstddef>
#include <vector>

#include "dmt/common/types.hpp"

constexpr float kDispCoeff   = -2.0F;
constexpr float kDispConstLK = 4.1488080e3; // L&K Handbook of Pulsar Astronomy
constexpr float kDispConstMT = 1 / 2.41e-4; // TEMPO2, Manchester&Taylor (1972)
constexpr float kDispConstSI = 4.1488064e3; // SI value, Kulkarni (2020)
constexpr float kDispConst   = kDispConstMT;

namespace dm_utils {
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

// Compute the chirp table for coherent dedispersion.
void compute_chirp(std::complex<float>* chirp_table,
                   SizeType chirp_table_size,
                   const float* dm_grid,
                   SizeType ndm,
                   float fcenter,
                   float bw,
                   SizeType nbin,
                   SizeType nsub,
                   SizeType nchan);

template <bool Debug = false>
inline void add_offset_kernel(const float* __restrict__ arr1,
                              SizeType size_in1,
                              const float* __restrict__ arr2,
                              SizeType size_in2,
                              float* __restrict__ arr_out,
                              SizeType size_out,
                              SizeType offset) {
    if constexpr (Debug) {
        if (size_in1 != size_in2) {
            throw std::runtime_error("Input sizes are not equal");
        }
        if (size_out < size_in1) {
            throw std::runtime_error("Output size is less than input size");
        }
        if (offset >= size_in1) {
            throw std::runtime_error("Offset is greater than input size");
        }
    }
    SizeType t          = 0;
    const SizeType nsum = size_in1 - offset;
    std::copy_n(arr1, offset, arr_out);
    t += offset;
#pragma omp simd
    for (SizeType i = 0; i < nsum; ++i) {
        arr_out[offset + i] = arr1[offset + i] + arr2[i];
    }
    t += nsum;
    const SizeType nrest = std::min(offset, size_out - size_in1);
    if (nrest > 0) {
        std::copy_n(arr2 + nsum, nrest, arr_out + size_in1);
        t += nrest;
    }
    if (t < size_out) {
        std::fill(arr_out + t, arr_out + size_out, 0.0F);
    }
}

} // namespace dm_utils
