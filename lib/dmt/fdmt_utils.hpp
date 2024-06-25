#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

constexpr float kDispCoeff   = -2.0;
constexpr float kDispConstLK = 4.1488080e3; // L&K Handbook of Pulsar Astronomy
constexpr float kDispConstMT = 1 / 2.41e-4; // TEMPO2, Manchester&Taylor (1972)
constexpr float kDispConstSI = 4.1488064e3; // SI value, Kulkarni (2020)
constexpr float kDispConst   = kDispConstMT;

using SizeType = std::size_t;

namespace ddmt {
std::vector<SizeType> generate_delay_table(const float* dm_arr,
                                           SizeType dm_count,
                                           float f0,
                                           float df,
                                           SizeType nchans,
                                           float tsamp);
}

namespace fdmt {

float cff(float f1_start, float f1_end, float f2_start, float f2_end);

SizeType calculate_dt_sub(
    float f_start, float f_end, float f_min, float f_max, SizeType dt);

inline void add_offset_kernel(const float* __restrict arr1,
                              SizeType size_in1,
                              const float* __restrict arr2,
                              SizeType size_in2,
                              float* __restrict arr_out,
                              SizeType size_out,
                              SizeType offset) {
    if (size_in1 != size_in2) {
        throw std::runtime_error("Input sizes are not equal");
    }
    if (size_out < size_in1) {
        throw std::runtime_error("Output size is less than input size");
    }
    if (offset >= size_in1) {
        throw std::runtime_error("Offset is greater than input size");
    }
    const SizeType nsum = size_in1 - offset;
    // SizeType t_ind      = 0;

    std::copy_n(arr1, offset, arr_out);
    // t_ind += offset;

#pragma omp simd
    for (SizeType i = 0; i < nsum; ++i) {
        arr_out[offset + i] = arr1[offset + i] + arr2[i];
    }
    // t_ind += nsum;

    const SizeType nrest = std::min(offset, size_out - size_in1);
    if (nrest > 0) {
        std::copy_n(arr2 + nsum, nrest, arr_out + size_in1);
        // t_ind += nrest;
    }
}

inline void copy_kernel(const float* __restrict arr1,
                        SizeType size_in,
                        float* __restrict arr_out,
                        SizeType size_out) {
    if (size_out < size_in) {
        throw std::runtime_error("Output size is less than input size");
    }
    std::copy_n(arr1, size_in, arr_out);
}

SizeType find_closest_index(const std::vector<SizeType>& arr_sorted,
                            SizeType val);

} // namespace fdmt
