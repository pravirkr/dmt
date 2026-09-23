#include <algorithm>
#include <numeric>
#include <ranges>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include "dmt/dmt.hpp"
#include "dmt/utils/simulate.hpp"

namespace dmt {

TEST_CASE("umbrella header dmt/dmt.hpp is self-contained", "[umbrella][cpu]") {
    CHECK(kDispConst > 0.0F);
}

TEST_CASE("generate_pure_frb injects energy on the dispersed track",
          "[simulate][cpu]") {
    const SizeType nchans = 16;
    const SizeType nsamps = 64;
    const float f_min     = 1000.0F;
    const float f_max     = 1500.0F;
    const SizeType dt     = 0;
    const float toa       = 20.0F;
    const float amp       = 3.0F;

    const auto [waterfall, n_disp] =
        utils::generate_pure_frb(nchans, nsamps, f_min, f_max, dt, toa, amp);
    REQUIRE(waterfall.size() == nchans * nsamps);
    CHECK(n_disp > 0);
    const auto energy =
        std::accumulate(waterfall.begin(), waterfall.end(), 0.0F);
    CHECK(energy == Catch::Approx(amp * static_cast<float>(nchans)));

    const auto t = static_cast<SizeType>(toa);
    for (SizeType c = 0; c < nchans; ++c) {
        CHECK(waterfall[(c * nsamps) + t] == Catch::Approx(amp));
    }
}

TEST_CASE("generate_pure_frb with positive dt smears across channels",
          "[simulate][cpu]") {
    const SizeType nchans          = 8;
    const SizeType nsamps          = 64;
    const auto [waterfall, n_disp] = utils::generate_pure_frb(
        nchans, nsamps, 1000.0F, 1500.0F, 12, 40.0F, 1.0F);
    CHECK(n_disp > nchans);
    CHECK(std::ranges::any_of(waterfall, [](float v) { return v > 0.0F; }));
}

} // namespace dmt
