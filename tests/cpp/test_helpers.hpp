#pragma once

#include <cstddef>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include "dmt/common/types.hpp"

#ifdef DMT_ENABLE_CUDA
#include <cuda_runtime_api.h>
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

inline std::vector<float> sequential_waterfall(SizeType nchans, SizeType nsamps,
                                               SizeType modulus = 17,
                                               SizeType offset  = 1) {
    std::vector<float> waterfall(nchans * nsamps);
    for (SizeType i = 0; i < waterfall.size(); ++i) {
        waterfall[i] = static_cast<float>((i % modulus) + offset);
    }
    return waterfall;
}

inline std::vector<float> prefix(const std::vector<float>& values, SizeType n) {
    REQUIRE(values.size() >= n);
    return {values.begin(),
            values.begin() + static_cast<std::ptrdiff_t>(n)};
}

inline void require_exact(const std::vector<float>& actual,
                          const std::vector<float>& expected) {
    REQUIRE_THAT(actual, Catch::Matchers::Equals(expected));
}

inline void require_exact(const std::vector<float>& actual,
                          const std::vector<float>& expected, SizeType n) {
    REQUIRE_THAT(prefix(actual, n), Catch::Matchers::Equals(prefix(expected, n)));
}

inline void require_approx(const std::vector<float>& actual,
                           const std::vector<float>& expected,
                           double margin = kParityMargin) {
    REQUIRE_THAT(actual, Catch::Matchers::Approx(expected).margin(margin));
}

inline void require_approx(const std::vector<float>& actual,
                           const std::vector<float>& expected, SizeType n,
                           double margin = kParityMargin) {
    REQUIRE_THAT(prefix(actual, n),
                 Catch::Matchers::Approx(prefix(expected, n)).margin(margin));
}

#ifdef DMT_ENABLE_CUDA
inline bool cuda_device_available() {
    int count      = 0;
    const auto err = cudaGetDeviceCount(&count);
    return err == cudaSuccess && count > 0;
}
#endif

} // namespace dmt::test
