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

void fdmt::add_offset_kernel(std::span<const float> arr1,
                             std::span<const float> arr2,
                             std::span<float> arr_out, size_t offset) {
    if (arr1.size() != arr2.size()) {
        throw std::runtime_error("Input sizes are not equal");
    }
    if (arr_out.size() < arr1.size()) {
        throw std::runtime_error("Output size is less than input size");
    }
    if (offset >= arr1.size()) {
        throw std::runtime_error("Offset is greater than input size");
    }
    size_t nsum  = arr1.size() - offset;
    size_t t_ind = 0;

    std::copy_n(arr1.data(), offset, arr_out.data());
    t_ind += offset;

    for (size_t i = 0; i < nsum; ++i) {
        arr_out[t_ind + i] = arr1[t_ind + i] + arr2[i];
    }
    t_ind += nsum;

    size_t nrest = std::min(offset, arr_out.size() - t_ind);
    if (nrest > 0) {
        std::copy_n(arr2.data() + nsum, nrest, arr_out.data() + t_ind);
        t_ind += nrest;
    }
}

void fdmt::copy_kernel(std::span<const float> arr1, std::span<float> arr_out) {
    if (arr_out.size() < arr1.size()) {
        throw std::runtime_error("Output size is less than input size");
    }
    std::copy_n(arr1.data(), arr1.size(), arr_out.data());
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