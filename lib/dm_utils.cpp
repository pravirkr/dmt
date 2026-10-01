#include "dmt/dm_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <format>
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

template <typename T>
static SizeType find_nearest_sorted_idx_impl(std::span<const T> arr_sorted,
                                             T val) {
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
                                 ((val - val_prev) <= (val_curr - val));
        if (prev_closer) {
            --idx;
        }
    }
    return idx;
}

SizeType find_nearest_sorted_idx(std::span<const SizeType> arr_sorted,
                                 SizeType val) {
    return find_nearest_sorted_idx_impl(arr_sorted, val);
}

SizeType find_nearest_sorted_idx(std::span<const IndexType> arr_sorted,
                                 IndexType val) {
    return find_nearest_sorted_idx_impl(arr_sorted, val);
}

std::vector<SizeType> generate_delay_table(std::span<const float> dm_arr,
                                           SizeType nchans,
                                           float fch1,
                                           float foff,
                                           float tsamp) {
    const auto ndm = dm_arr.size();
    std::vector<SizeType> delay_table(nchans * ndm);
    // Reference the highest-frequency channel, the first one to arrive, so
    // every other (lower-frequency, later-arriving) channel gets a
    // non-negative additive delay.
    const auto f_last = fch1 + (static_cast<float>(nchans - 1) * foff);
    const auto f_ref  = std::max(fch1, f_last);
    const auto b      = 1.F / f_ref;
    for (SizeType idm = 0; idm < ndm; ++idm) {
        for (SizeType ichan = 0; ichan < nchans; ++ichan) {
            const auto a = 1.F / (fch1 + (static_cast<float>(ichan) * foff));
            const auto delay =
                kDispConst / tsamp * ((a * a) - (b * b)) * dm_arr[idm];
            delay_table[(idm * nchans) + ichan] =
                static_cast<SizeType>(std::nearbyint(delay));
        }
    }
    return delay_table;
}

std::vector<float> generate_fractional_delay_table(SizeType nchans,
                                                   float fch1,
                                                   float foff,
                                                   float tsamp) {
    if (nchans == 0) {
        throw std::invalid_argument("nchans must be greater than 0");
    }
    if (tsamp <= 0.0F) {
        throw std::invalid_argument("tsamp must be greater than 0");
    }
    std::vector<float> frac_delays(nchans);
    const auto f_last = fch1 + (static_cast<float>(nchans - 1) * foff);
    const auto f_ref  = std::max(fch1, f_last);
    if (f_ref <= 0.0F || std::min(fch1, f_last) <= 0.0F) {
        throw std::invalid_argument("Frequencies must be positive");
    }
    const auto b = 1.F / f_ref;
    for (SizeType ichan = 0; ichan < nchans; ++ichan) {
        const auto a       = 1.F / (fch1 + (static_cast<float>(ichan) * foff));
        frac_delays[ichan] = (kDispConst / tsamp) * ((a * a) - (b * b));
    }
    return frac_delays;
}

std::vector<float> generate_levin_dm_grid(float dm_start,
                                          float dm_end,
                                          float tsamp,
                                          float pulse_width,
                                          float f_min,
                                          float f_max,
                                          SizeType nchans,
                                          float tol) {
    if (dm_start < 0.0F || dm_end < dm_start) {
        throw std::invalid_argument(std::format(
            "Invalid DM range[{}, {}]: require 0 <= dm_start <= dm_end",
            dm_start, dm_end));
    }
    if (tsamp <= 0.0F || pulse_width < 0.0F) {
        throw std::invalid_argument(
            std::format("Invalid tsamp[{}] or pulse_width[{}]: require tsamp > "
                        "0 and pulse_width >= 0",
                        tsamp, pulse_width));
    }
    if (f_min <= 0.0F || f_max <= f_min) {
        throw std::invalid_argument(std::format(
            "Invalid f_min[{}], f_max[{}]: require 0 < f_min < f_max", f_min,
            f_max));
    }
    if (nchans == 0) {
        throw std::invalid_argument(
            std::format("Invalid nchans[{}]: require nchans > 0", nchans));
    }
    if (tol <= 1.0F) {
        throw std::invalid_argument(
            std::format("Invalid tol[{}]: require tol > 1.0", tol));
    }

    if (dm_start == dm_end) {
        return {dm_start};
    }

    const double dt_us = static_cast<double>(tsamp) * 1.0e6;
    const double ti_us = static_cast<double>(pulse_width) * 1.0e6;
    const double df_mhz =
        static_cast<double>(f_max - f_min) / static_cast<double>(nchans);
    const double f_center_ghz =
        (static_cast<double>(f_min + f_max) * 0.5) * 1.0e-3;
    const auto tol_d  = static_cast<double>(tol);
    const double tol2 = tol_d * tol_d;

    const double a =
        8.3 * df_mhz / (f_center_ghz * f_center_ghz * f_center_ghz);
    const double a2        = a * a;
    const double b2        = a2 * (static_cast<double>(nchans * nchans) / 16.0);
    const double c         = ((dt_us * dt_us) + (ti_us * ti_us)) * (tol2 - 1.0);
    const double denom_inv = 1.0 / (a2 + b2);

    std::vector<float> dm_table;
    dm_table.push_back(dm_start);
    while (dm_table.back() < dm_end) {
        const auto prev    = static_cast<double>(dm_table.back());
        const double prev2 = prev * prev;
        const double k     = c + (tol2 * a2 * prev2);
        const double disc  = (-a2 * b2 * prev2) + ((a2 + b2) * k);
        if (disc <= 0.0) {
            throw std::runtime_error(std::format(
                "Numerical collapse in generate_levin_dm_grid at DM={}", prev));
        }
        const double next_dm = ((b2 * prev) + std::sqrt(disc)) * denom_inv;
        if (next_dm <= prev) {
            throw std::runtime_error(std::format(
                "Zero step size encountered in generate_levin_dm_grid at DM={}",
                prev));
        }
        dm_table.push_back(static_cast<float>(next_dm));
    }
    return dm_table;
}

std::vector<float> generate_levin_dm_grid_piecewise(float dm_start,
                                                    float dm_end,
                                                    float tsamp,
                                                    float pulse_width,
                                                    float f_min,
                                                    float f_max,
                                                    SizeType nchans,
                                                    float tol,
                                                    SizeType min_run) {
    const auto levin = generate_levin_dm_grid(
        dm_start, dm_end, tsamp, pulse_width, f_min, f_max, nchans, tol);
    if (levin.size() < 2) {
        return levin;
    }
    min_run = std::max<SizeType>(min_run, 1);
    // Local Levin step at trial i (non-decreasing in practice; the minimum
    // over a segment is used, so no segment is ever coarser than Levin).
    const auto n  = levin.size();
    const auto st = [&](SizeType i) {
        return static_cast<double>(levin[i + 1]) -
               static_cast<double>(levin[i]);
    };
    const double end = static_cast<double>(levin.back());
    std::vector<float> out;
    double a   = static_cast<double>(levin.front());
    SizeType i = 0; // Levin index with levin[i] <= a
    while (a < end) {
        while (i + 2 < n && static_cast<double>(levin[i + 1]) <= a) {
            ++i;
        }
        // Segment: from a while the Levin step stays below twice its value
        // at a (DDplan-style doublings), at that smallest step.
        double step = st(i);
        SizeType j  = i;
        while (j + 2 < n && st(j + 1) < 2.0 * st(i)) {
            ++j;
            step = std::min(step, st(j));
        }
        const double b = std::min(end, static_cast<double>(levin[j + 1]));
        auto count     = static_cast<SizeType>(std::ceil((b - a) / step));
        if (count < min_run) {
            // Too short for the NUFFT: refine it (never coarser).
            step  = (b - a) / static_cast<double>(min_run);
            count = min_run;
        }
        count = std::max<SizeType>(count, 1);
        for (SizeType q = 0; q < count; ++q) {
            out.push_back(
                static_cast<float>(a + (step * static_cast<double>(q))));
        }
        a += step * static_cast<double>(count);
    }
    // The last point closes the final run (exactly one more step).
    out.push_back(static_cast<float>(a));
    return out;
}

SizeType next_fft_size(SizeType n) {
    // Even lengths only: R2C/C2R of an odd length loses the Nyquist bin
    // symmetry FFTW exploits. Search m = 2 * (7-smooth) >= n.
    const SizeType half = std::max<SizeType>(1, (n + 1) / 2);
    for (SizeType m = half;; ++m) {
        SizeType r = m;
        for (const SizeType p : {2U, 3U, 5U, 7U}) {
            while (r % p == 0) {
                r /= p;
            }
        }
        if (r == 1) {
            return 2 * m;
        }
    }
}

} // namespace dmt::utils
