#include <algorithm>
#include <cmath>
#include <iterator>
#include <stdexcept>
#include <vector>

#include "dmt/fdmt_utils.hpp"

std::vector<SizeType> ddmt::generate_delay_table(const float* dm_arr,
                                                 SizeType dm_count,
                                                 float f0,
                                                 float df,
                                                 SizeType nchans,
                                                 float tsamp) {
    std::vector<SizeType> delay_table(nchans * dm_count);
    for (SizeType idm = 0; idm < dm_count; ++idm) {
        for (SizeType ichan = 0; ichan < nchans; ++ichan) {
            const auto a = 1.F / (f0 + static_cast<float>(ichan) * df);
            const auto b = 1.F / f0;
            const auto delay =
                kDispConst / tsamp * (a * a - b * b) * dm_arr[idm];
            delay_table[idm * nchans + ichan] =
                static_cast<SizeType>(std::round(delay));
        }
    }
    return delay_table;
}

float fdmt::cff(float f1_start, float f1_end, float f2_start, float f2_end) {
    return (std::pow(f1_start, kDispCoeff) - std::pow(f1_end, kDispCoeff)) /
           (std::pow(f2_start, kDispCoeff) - std::pow(f2_end, kDispCoeff));
}

SizeType fdmt::calculate_dt_sub(
    float f_start, float f_end, float f_min, float f_max, SizeType dt) {
    const float ratio = cff(f_start, f_end, f_min, f_max);
    return static_cast<SizeType>(std::round(static_cast<float>(dt) * ratio));
}

SizeType fdmt::find_closest_index(const std::vector<SizeType>& arr_sorted,
                                  SizeType val) {
    if (arr_sorted.empty()) {
        throw std::runtime_error("Array is empty");
    }
    auto it      = std::lower_bound(arr_sorted.begin(), arr_sorted.end(), val);
    SizeType idx = std::distance(arr_sorted.begin(), it);

    if (it != arr_sorted.end()) {
        if (it != arr_sorted.begin() && val - *(it - 1) < *it - val) {
            idx--;
        }
    } else {
        idx = arr_sorted.size() - 1;
    }
    return idx;
}