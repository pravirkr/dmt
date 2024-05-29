#pragma once

#include <cstddef>
#include <vector>

constexpr float kDispCoeff   = -2.0;
constexpr float kDispConstLK = 4.1488080e3; // L&K Handbook of Pulsar Astronomy
constexpr float kDispConstMT = 1 / 2.41e-4; // TEMPO2, Manchester&Taylor (1972)
constexpr float kDispConstSI = 4.1488064e3; // SI value, Kulkarni (2020)
constexpr float kDispConst   = kDispConstMT;

using SizeType = std::size_t;

namespace fdmt {

float cff(float f1_start, float f1_end, float f2_start, float f2_end);

SizeType calculate_dt_sub(
    float f_start, float f_end, float f_min, float f_max, SizeType dt);

void add_offset_kernel(const float* arr1,
                       SizeType size_in1,
                       const float* arr2,
                       SizeType size_in2,
                       float* arr_out,
                       SizeType size_out,
                       SizeType offset);

void copy_kernel(const float* arr1,
                 SizeType size_in,
                 float* arr_out,
                 SizeType size_out);

SizeType find_closest_index(const std::vector<SizeType>& arr_sorted,
                            SizeType val);

} // namespace fdmt
