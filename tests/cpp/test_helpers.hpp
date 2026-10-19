#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include "dmt/common/types.hpp"

#include "dmt/common/backend.hpp"
#include "dmt/common/plans.hpp"
#include "dmt/fdmt_box_weights.hpp"

#ifdef DMT_ENABLE_GPU
#include "dmt/gpu_compat.cuh"
#endif

namespace dmt::test {

// Default observation parameters reused across FDMT-family tests.
inline constexpr float kFMin          = 1000.0F;
inline constexpr float kFMax          = 1500.0F;
inline constexpr SizeType kNchans     = 32;
inline constexpr SizeType kNsamps     = 256;
inline constexpr float kTsamp         = 0.001F;
inline constexpr IndexType kDtMax     = 32;
inline constexpr double kParityMargin = 1.0E-4;

inline std::vector<float> sequential_waterfall(SizeType nchans,
                                               SizeType nsamps,
                                               SizeType modulus = 17,
                                               SizeType offset  = 1) {
    std::vector<float> waterfall(nchans * nsamps);
    for (SizeType i = 0; i < waterfall.size(); ++i) {
        waterfall[i] = static_cast<float>((i % modulus) + offset);
    }
    return waterfall;
}

/// Effective level-0 weight of a box of `box_len` samples with box smearing:
/// the box-mean weight round(2^b / L) * 2^-b (fdmt_box_weights.hpp).
inline double box_mean_weight(const plans::FDMTPlan& plan, SizeType box_len) {
    const auto weights = plan.get_box_weights();
    REQUIRE(box_len >= 1);
    REQUIRE(box_len <= weights.size());
    return std::ldexp(static_cast<double>(weights[box_len - 1]),
                      -static_cast<int>(plan.get_box_fraction_bits()));
}

/// FDMT output of a unit impulse placed inside every channel's level-0 box of
/// trial `dm_idx` (add_frb_track): sum_c weight(L_c) with box smearing, or
/// nchans without.
inline double frb_track_gain(const plans::FDMTPlan& plan,
                             SizeType dm_idx,
                             bool use_box_smearing) {
    const auto nchans = plan.get_nchans();
    if (!use_box_smearing) {
        return static_cast<double>(nchans);
    }
    const auto smearing = plan.get_smearing_grid_final();
    double gain         = 0.0;
    for (SizeType c = 0; c < nchans; ++c) {
        const auto len = static_cast<SizeType>(smearing[(dm_idx * nchans) + c]);
        gain += box_mean_weight(plan, len + 1);
    }
    return gain;
}

/// Largest |packed - float| of every trial with box smearing: packed input
/// rounds each channel's box mean to f(L_c) fractional bits (at most
/// 2^-(f + 1) each, see fdmt_box_weights.hpp); float input does not.
inline std::vector<double> packed_rounding_bound(const plans::FDMTPlan& plan) {
    const auto nchans   = plan.get_nchans();
    const auto smearing = plan.get_smearing_grid_final();
    std::vector<double> bound(plan.get_dmt_ndms(), 0.0);
    for (SizeType i = 0; i < bound.size(); ++i) {
        for (SizeType c = 0; c < nchans; ++c) {
            const auto len =
                static_cast<SizeType>(smearing[(i * nchans) + c]) + 1;
            bound[i] += std::ldexp(
                1.0,
                -static_cast<int>(algorithms::detail::box_mean_bits(len)) - 1);
        }
    }
    return bound;
}

/// Packed-input output `actual` vs float-input output `expected` of the same
/// values with box smearing: the leading dmt_size values of each of `nbeams`
/// beams (dmt_size apart) agree within packed_rounding_bound() (plus float
/// round-off).
inline void require_packed_near(const std::vector<float>& actual,
                                const std::vector<float>& expected,
                                const plans::FDMTPlan& plan,
                                SizeType nbeams) {
    const auto stride = plan.get_dmt_size();
    const auto ndms   = plan.get_dmt_ndms();
    const auto nsamps = plan.get_dmt_nsamps();
    REQUIRE(actual.size() >= nbeams * stride);
    REQUIRE(expected.size() >= nbeams * stride);
    const auto bound = packed_rounding_bound(plan);
    double worst     = 0.0; // fraction of the bound used
    for (SizeType b = 0; b < nbeams; ++b) {
        for (SizeType i = 0; i < ndms; ++i) {
            for (SizeType t = 0; t < nsamps; ++t) {
                const auto k     = (b * stride) + (i * nsamps) + t;
                const double e   = expected[k];
                const double d   = std::abs(static_cast<double>(actual[k]) - e);
                const double tol = bound[i] + (1e-6 * std::abs(e)) + 1e-6;
                worst            = std::max(worst, d / tol);
            }
        }
    }
    INFO("worst |packed - float| / bound = " << worst);
    REQUIRE(worst <= 1.0);
}

inline std::vector<float> prefix(const std::vector<float>& values, SizeType n) {
    REQUIRE(values.size() >= n);
    return {values.begin(), values.begin() + static_cast<std::ptrdiff_t>(n)};
}

inline void require_exact(const std::vector<float>& actual,
                          const std::vector<float>& expected) {
    REQUIRE_THAT(actual, Catch::Matchers::Equals(expected));
}

inline void require_exact(const std::vector<float>& actual,
                          const std::vector<float>& expected,
                          SizeType n) {
    REQUIRE_THAT(prefix(actual, n),
                 Catch::Matchers::Equals(prefix(expected, n)));
}

inline void require_approx(const std::vector<float>& actual,
                           const std::vector<float>& expected,
                           double margin = kParityMargin) {
    REQUIRE_THAT(actual, Catch::Matchers::Approx(expected).margin(margin));
}

inline void require_approx(const std::vector<float>& actual,
                           const std::vector<float>& expected,
                           SizeType n,
                           double margin = kParityMargin) {
    REQUIRE_THAT(prefix(actual, n),
                 Catch::Matchers::Approx(prefix(expected, n)).margin(margin));
}

#ifdef DMT_ENABLE_GPU
/// The GPU backend of this build (CUDA or HIP), on device `device`.
inline Exec gpu_exec(int device = 0) {
#ifdef DMT_ENABLE_HIP
    return Exec::hip(device);
#else
    return Exec::cuda(device);
#endif
}

inline bool gpu_device_available() {
    int count      = 0;
    const auto err = cudaGetDeviceCount(&count);
    return err == cudaSuccess && count > 0;
}
#endif

} // namespace dmt::test
