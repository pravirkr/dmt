#pragma once

#include <cstddef>
#include <vector>

constexpr float kDispCoeff = -2.0;
constexpr float kDispConst = 4.148808e3;

namespace fdmt {

float cff(float f1_start, float f1_end, float f2_start, float f2_end);

size_t calculate_dt_sub(float f_start, float f_end, float f_min, float f_max,
                        size_t dt);

void add_offset_kernel(const float* arr1, size_t size_in1, const float* arr2,
                       size_t size_in2, float* arr_out, size_t size_out,
                       size_t offset);

void copy_kernel(const float* arr1, size_t size_in, float* arr_out,
                 size_t size_out);

size_t find_closest_index(const std::vector<size_t>& arr_sorted, size_t val);

}  // namespace fdmt
