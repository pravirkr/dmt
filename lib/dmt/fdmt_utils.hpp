#pragma once

#include <cstddef>
#include <vector>

constexpr float disp_coeff = -2.0;
constexpr float disp_const = 4.148808e3;

namespace fdmt {

float cff(float f1_start, float f1_end, float f2_start, float f2_end);

size_t calculate_delta_t_max(float f_min, float f_max, float dm_max,
                             float tsamp);

std::vector<float> calculate_dm_arr(float f_min, float f_max, float tsamp,
                                    size_t dt_max, size_t dt_step = 1,
                                    size_t dt_min = 0);

size_t calculate_dt_sub(float f_start, float f_end, float f_min, float f_max,
                        size_t dt);

std::vector<size_t> calculate_dt_grid_sub(float f_start, float f_end,
                                          float f_min, float f_max,
                                          size_t dt_max, size_t dt_step,
                                          size_t dt_min);

void add_offset_kernel(const float* arr1, size_t size_in1, const float* arr2,
                       size_t size_in2, float* arr_out, size_t size_out,
                       size_t offset);

void copy_kernel(const float* arr1, size_t size_in, float* arr_out,
                 size_t size_out);

size_t find_closest_index(const std::vector<size_t>& arr_sorted, size_t val);

}  // namespace fdmt
