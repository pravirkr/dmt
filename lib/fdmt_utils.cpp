#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "dmt/fdmt_utils.hpp"

float fdmt::cff(float f1_start, float f1_end, float f2_start, float f2_end) {
    return (std::pow(f1_start, disp_coeff) - std::pow(f1_end, disp_coeff))
           / (std::pow(f2_start, disp_coeff) - std::pow(f2_end, disp_coeff));
}

size_t fdmt::calculate_delta_t_max(float f_min, float f_max, float dm_max,
                                   float tsamp) {
    if (dm_max < 0.0F) {
        throw std::runtime_error("DM max must be greater than 0");
    }
    if (f_min >= f_max) {
        throw std::runtime_error("f_min must be less than f_max");
    }
    float delta_t
        = disp_const * dm_max
          * (std::pow(f_min, disp_coeff) - std::pow(f_max, disp_coeff));
    return static_cast<size_t>(std::ceil(delta_t / tsamp));
}

std::vector<float> fdmt::calculate_dm_arr(float f_min, float f_max, float tsamp,
                                          size_t dt_max, size_t dt_step,
                                          size_t dt_min) {
    const size_t ndm = (dt_max - dt_min) / dt_step;
    std::vector<float> dm_arr(ndm);
    const float dm_conv
        = disp_const
          * (std::pow(f_min, disp_coeff) - std::pow(f_max, disp_coeff));
    const float dm_step = tsamp / dm_conv;
    for (size_t i = 0; i < ndm; ++i) {
        dm_arr[i] = static_cast<float>(dt_min + i * dt_step) * dm_step;
    }
    return dm_arr;
}

size_t fdmt::calculate_dt_sub(float f_start, float f_end, float f_min,
                              float f_max, size_t dt) {
    float ratio = cff(f_start, f_end, f_min, f_max);
    return static_cast<size_t>(std::round(static_cast<float>(dt) * ratio));
}

std::vector<size_t> fdmt::calculate_dt_grid_sub(float f_start, float f_end,
                                                float f_min, float f_max,
                                                size_t dt_max, size_t dt_step,
                                                size_t dt_min) {
    const size_t dt_max_sub
        = calculate_dt_sub(f_start, f_end, f_min, f_max, dt_max);
    const size_t dt_min_sub
        = calculate_dt_sub(f_start, f_end, f_min, f_max, dt_min);
    std::vector<size_t> dt_grid;
    for (size_t dt = dt_min_sub; dt <= dt_max_sub; dt += dt_step) {
        dt_grid.push_back(dt);
    }
    return dt_grid;
}

void fdmt::add_offset_kernel(const float* arr1, size_t size_in1,
                             const float* arr2, size_t size_in2, float* arr_out,
                             size_t size_out, size_t offset) {
    if (size_in1 != size_in2) {
        throw std::runtime_error("Input sizes are not equal");
    }
    if (size_out < size_in1) {
        throw std::runtime_error("Output size is less than input size");
    }
    if (offset >= size_in1) {
        throw std::runtime_error("Offset is greater than input size");
    }
    size_t nsum  = size_in1 - offset;
    size_t t_ind = 0;

    std::copy_n(arr1, offset, arr_out);
    t_ind += offset;

    for (size_t i = 0; i < nsum; ++i) {
        arr_out[t_ind + i] = arr1[t_ind + i] + arr2[i];
    }
    t_ind += nsum;

    size_t nrest = std::min(offset, size_out - t_ind);
    if (nrest > 0) {
        std::copy_n(arr2 + nsum, nrest, arr_out + t_ind);
        t_ind += nrest;
    }
}

void fdmt::copy_kernel(const float* arr1, size_t size_in, float* arr_out,
                       size_t size_out) {
    if (size_out < size_in) {
        throw std::runtime_error("Output size is less than input size");
    }
    std::copy(arr1, arr1 + size_in, arr_out);
}

size_t fdmt::find_closest_index(const std::vector<size_t>& arr_sorted,
                                size_t val) {
    if (arr_sorted.empty()) {
        throw std::runtime_error("Array is empty");
    }
    auto it    = std::lower_bound(arr_sorted.begin(), arr_sorted.end(), val);
    size_t idx = std::distance(arr_sorted.begin(), it);

    if (it != arr_sorted.end()) {
        if (it != arr_sorted.begin() && val - *(it - 1) < *it - val) {
            idx--;
        }
    } else {
        idx = arr_sorted.size() - 1;
    }
    return idx;
}