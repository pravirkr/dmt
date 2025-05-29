#include <catch2/catch_test_macros.hpp>

#include <spdlog/spdlog.h>

#include <cstddef>
#include <span>

#include "dmt/algorithms/fdmt.hpp"

namespace dmt {

using algorithms::FDMTCPU;

TEST_CASE("FDMTCPU", "[fdmt_cpu]") {
    const float f_min    = 1000.0F;
    const float f_max    = 1500.0F;
    const size_t nchans  = 500;
    const size_t nsamps  = 1024;
    const float tsamp    = 0.001F;
    const size_t dt_max  = 512;
    const size_t dt_step = 1;
    const size_t dt_min  = 0;

    dmt::FDMTCPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_step,
                      dt_min);

    SECTION("Constructor and getter methods") {
        const auto& plan         = fdmt.get_plan();
        const auto ndms_expected = static_cast<SizeType>(
            std::floor((dt_max - dt_min) / static_cast<float>(dt_step)) + 1);
        CHECK(plan.get_dt_grid_final().size() == ndms_expected);
        CHECK(plan.get_dm_grid_final().size() == ndms_expected);
        CHECK(plan.get_dmt_size() ==
              static_cast<SizeType>(ndms_expected * (nsamps + dt_max)));
    }
    /*
    SECTION("initialise method") {
        std::vector<float> waterfall(nchans * nsamps, 1.0F);
        std::vector<float> state(fdmt.get_plan().get_buffer_size(), 0.0F);
        REQUIRE_NOTHROW(fdmt.initialise(waterfall.data(), waterfall.size(),
                                        state.data(), state.size()));
        // Check if the state is correctly initialized
        REQUIRE(
            std::ranges::all_of(state, [](float val) { return val > 0.0F; }));
    }
    */
    SECTION("execute method") {
        std::vector<float> waterfall(nchans * nsamps, 1.0F);
        std::vector<float> dmt(fdmt.get_plan().get_dmt_size(), 0.0F);
        REQUIRE_NOTHROW(fdmt.execute(
            std::span<const float>(waterfall.data(), waterfall.size()),
            std::span<float>(dmt.data(), dmt.size())));
    }
}

} // namespace dmt
