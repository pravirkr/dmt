#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "dmt/fdmt_utils.hpp"

float fdmt::cff(float f1_start, float f1_end, float f2_start, float f2_end) {
    return (std::pow(f1_start, kDispCoeff) - std::pow(f1_end, kDispCoeff)) /
           (std::pow(f2_start, kDispCoeff) - std::pow(f2_end, kDispCoeff));
}

size_t fdmt::calculate_dt_sub(float f_start, float f_end, float f_min,
                              float f_max, size_t dt) {
    float ratio = cff(f_start, f_end, f_min, f_max);
    return static_cast<size_t>(std::round(static_cast<float>(dt) * ratio));
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
    std::copy_n(arr1, size_in, arr_out);
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