#include <algorithm>
#include <cmath>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include "dmt/algorithms/ddmt.hpp"
#include "dmt/common/plans.hpp"
#include "dmt/utils/simulate.hpp"

namespace dmt {

using algorithms::DDMTCPU;
using plans::DDMTPlan;

TEST_CASE("DDMTPlan construction and getters", "[ddmt][cpu]") {
    const float f_min     = 1000.0F;
    const float f_max     = 1500.0F;
    const SizeType nchans = 16;
    const float tsamp     = 0.001F;

    SECTION("regular DM grid") {
        DDMTPlan plan(f_min, f_max, nchans, tsamp, 20.0F, 5.0F, 0.0F);
        CHECK(plan.get_f_min() == f_min);
        CHECK(plan.get_f_max() == f_max);
        CHECK(plan.get_nchans() == nchans);
        CHECK(plan.get_tsamp() == tsamp);
        CHECK(plan.get_dm_arr().size() > 1);
        CHECK(plan.get_container().nchans == nchans);
        CHECK(plan.get_container().delay_table.size() ==
              plan.get_dm_arr().size() * nchans);
        REQUIRE_THAT(plan.get_dm_grid(),
                     Catch::Matchers::Equals(plan.get_dm_arr()));
    }

    SECTION("custom dm_arr") {
        const std::vector<float> dms = {0.0F, 5.0F, 12.5F};
        DDMTPlan plan(f_min, f_max, nchans, tsamp, dms);
        REQUIRE_THAT(plan.get_dm_arr(), Catch::Matchers::Equals(dms));
    }
}

TEST_CASE("DDMTCPU execute recovers a zero-DM ones waterfall", "[ddmt][cpu]") {
    const float f_min     = 1000.0F;
    const float f_max     = 1500.0F;
    const SizeType nchans = 8;
    const SizeType nsamps = 64;
    const float tsamp     = 0.001F;

    DDMTCPU ddmt(f_min, f_max, nchans, tsamp, 10.0F, 5.0F, 0.0F);
    const auto& plan_c   = ddmt.get_plan().get_container();
    const auto max_delay = plan_c.delay_table.back();
    REQUIRE(nsamps > max_delay);
    const auto nsamps_reduced = nsamps - max_delay;
    const auto dm_count       = plan_c.dm_arr.size();

    std::vector<float> waterfall(nchans * nsamps, 1.0F);
    std::vector<float> dmt(dm_count * nsamps_reduced, 0.0F);
    ddmt.execute(waterfall, dmt);

    // DM 0 has zero delay on every channel, so each output sample is nchans.
    std::vector<float> dm0(
        dmt.begin(), dmt.begin() + static_cast<std::ptrdiff_t>(nsamps_reduced));
    REQUIRE_THAT(dm0, Catch::Matchers::Equals(std::vector<float>(
                          nsamps_reduced, static_cast<float>(nchans))));
}

TEST_CASE("DDMTCPU custom dm_arr and mismatched output is a no-op",
          "[ddmt][cpu]") {
    const float f_min            = 1000.0F;
    const float f_max            = 1500.0F;
    const SizeType nchans        = 8;
    const SizeType nsamps        = 32;
    const float tsamp            = 0.001F;
    const std::vector<float> dms = {0.0F, 2.0F};

    DDMTCPU ddmt(f_min, f_max, nchans, tsamp, dms);
    std::vector<float> waterfall(nchans * nsamps, 1.0F);
    std::vector<float> bad(4, 42.0F);
    ddmt.execute(waterfall, bad);
    REQUIRE_THAT(bad, Catch::Matchers::Equals(std::vector<float>(4, 42.0F)));
}

TEST_CASE("DDMTCPU add_frb_track-style peak on the injected DM",
          "[ddmt][cpu]") {
    const float f_min     = 1000.0F;
    const float f_max     = 1500.0F;
    const SizeType nchans = 16;
    const SizeType nsamps = 128;
    const float tsamp     = 0.001F;
    const SizeType dt     = 8;
    const float toa       = 40.0F;

    const auto [waterfall, n_disp] =
        utils::generate_pure_frb(nchans, nsamps, f_min, f_max, dt, toa, 1.0F);
    REQUIRE(n_disp > 0);

    const float dm = static_cast<float>(dt) * tsamp /
                     (kDispConst * (std::pow(f_min, kDispCoeff) -
                                    std::pow(f_max, kDispCoeff)));
    DDMTCPU ddmt(f_min, f_max, nchans, tsamp, std::vector<float>{0.0F, dm});
    const auto& plan_c        = ddmt.get_plan().get_container();
    const auto max_delay      = plan_c.delay_table.back();
    const auto nsamps_reduced = nsamps - max_delay;
    std::vector<float> dmt(plan_c.dm_arr.size() * nsamps_reduced, 0.0F);
    ddmt.execute(waterfall, dmt);

    const auto* row0 = dmt.data();
    const auto* row1 = dmt.data() + nsamps_reduced;
    const auto peak0 = *std::max_element(row0, row0 + nsamps_reduced);
    const auto peak1 = *std::max_element(row1, row1 + nsamps_reduced);
    CHECK(peak1 >= peak0);
    CHECK(peak1 > 0.0F);
}

} // namespace dmt
