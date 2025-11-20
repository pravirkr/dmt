#include "dmt/dm_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <vector>

namespace dmt::utils {

float cff(float f_start, float f_end, float f_min, float f_max) {
    return (std::pow(f_start, kDispCoeff) - std::pow(f_end, kDispCoeff)) /
           (std::pow(f_min, kDispCoeff) - std::pow(f_max, kDispCoeff));
}

SizeType calculate_dt_sub(
    float f_start, float f_end, float f_min, float f_max, SizeType dt) {
    const float ratio = cff(f_start, f_end, f_min, f_max);
    return static_cast<SizeType>(std::ceil(static_cast<float>(dt) * ratio));
}

float get_dmconv(float f_min, float f_max, float tsamp) {
    const float dm_conv = kDispConst * (std::pow(f_min, kDispCoeff) -
                                        std::pow(f_max, kDispCoeff));
    return tsamp / dm_conv;
}

SizeType find_nearest_sorted_idx(std::span<const SizeType> arr_sorted,
                                 SizeType val) {
    if (arr_sorted.empty()) {
        throw std::invalid_argument("find_nearest_sorted_idx: array is empty");
    }
    const auto it = std::ranges::lower_bound(arr_sorted, val);
    auto idx = static_cast<SizeType>(std::distance(arr_sorted.begin(), it));

    // Handle case where val is larger than all elements
    if (it == arr_sorted.end()) {
        return arr_sorted.size() - 1;
    }
    // Check if previous element is closer
    if (it != arr_sorted.begin()) {
        const auto val_prev    = *(it - 1);
        const auto val_curr    = *it;
        const bool prev_closer = (val >= val_prev) && (val <= val_curr) &&
                                 (val - val_prev) <= (val_curr - val);
        if (prev_closer) {
            --idx;
        }
    }
    return idx;
}

std::vector<SizeType> generate_delay_table(std::span<const float> dm_arr,
                                           SizeType nchans,
                                           float fch1,
                                           float foff,
                                           float tsamp) {
    const auto ndm = dm_arr.size();
    std::vector<SizeType> delay_table(nchans * ndm);
    for (SizeType idm = 0; idm < ndm; ++idm) {
        for (SizeType ichan = 0; ichan < nchans; ++ichan) {
            const auto a = 1.F / (fch1 + static_cast<float>(ichan) * foff);
            const auto b = 1.F / fch1;
            const auto delay =
                kDispConst / tsamp * (a * a - b * b) * dm_arr[idm];
            delay_table[(idm * nchans) + ichan] =
                static_cast<SizeType>(std::nearbyint(delay));
        }
    }
    return delay_table;
}

SizeType minimum_overlap(float dm_max,
                         float fcenter,
                         float bw,
                         float tbin,
                         SizeType nsub,
                         SizeType nchan) {
    float bw_chan         = bw / static_cast<float>(nsub * nchan);
    float fmin_bottom_sub = fcenter - (bw / 2);
    float fmax_bottom_sub = fmin_bottom_sub + bw_chan;
    float delay           = kDispConst * dm_max *
                  (std::pow(fmin_bottom_sub, kDispCoeff) -
                   std::pow(fmax_bottom_sub, kDispCoeff));
    if (delay < 0) {
        throw std::runtime_error("Negative dispersion delay is not allowed");
    }
    float delay_samples = delay / tbin;
    return static_cast<SizeType>(std::nearbyint(delay_samples));
}

std::vector<float> generate_coherent_dms(
    float dm_min, float dm_max, float fcenter, float bw, float tbin, float tp) {
    float f_min = fcenter - (bw / 2);
    float f_max = fcenter + (bw / 2);
    float delay = kDispConst * dm_max *
                  (std::pow(f_min, kDispCoeff) - std::pow(f_max, kDispCoeff));
    auto ncoherent = static_cast<size_t>(std::ceil(delay * tbin / (tp * tp)));
    float coh_dm_step = dm_max / static_cast<float>(ncoherent);
    std::vector<float> dm_grid;
    dm_grid.reserve(ncoherent);
    for (size_t i = 0; i < ncoherent; ++i) {
        dm_grid.push_back(dm_min + (static_cast<float>(i) * coh_dm_step));
    }
    return dm_grid;
}

void dedisperse(float* __restrict__ waterfall,
                SizeType waterfall_size,
                float dm,
                float f_min,
                float f_max,
                SizeType nchans,
                SizeType nsamps,
                float tsamp) {
    if (waterfall_size != nchans * nsamps) {
        throw std::runtime_error("Waterfall size mismatch");
    }
    const float foff = (f_max - f_min) / static_cast<float>(nchans);
    std::vector<int> shifts(nchans);
    for (SizeType ichan = 0; ichan < nchans; ++ichan) {
        const float fchan = f_min + (foff * static_cast<float>(ichan));
        const float delay =
            kDispConst * dm *
            (std::pow(f_min, kDispCoeff) - std::pow(fchan, kDispCoeff));
        shifts[ichan] = static_cast<int>(std::nearbyint(delay / tsamp));
    }
    for (SizeType ichan = 0; ichan < nchans; ++ichan) {
        const SizeType start = ichan * nsamps;
        const SizeType end   = start + nsamps;
        const int shift      = shifts[ichan];
        if (shift == 0) {
            continue;
        }
        if (shift > 0) {
            std::rotate(waterfall + start, waterfall + end - shift,
                        waterfall + end);
        } else {
            std::rotate(waterfall + start, waterfall + start - shift,
                        waterfall + end);
        }
    }
}
} // namespace dmt::utils
