#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <dmt/fdmt_cpu.hpp>
#include <span>

TEST_CASE("FDMT class tests", "[fdmt]") {
    SECTION("Test case 1: Constructor and getter methods") {
        FDMTCPU fdmt(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 1, 0);
        REQUIRE(fdmt.get_df() == 1.0F);
        REQUIRE(fdmt.get_correction() == 0.5F);
        REQUIRE(fdmt.get_niters() == 9);
        REQUIRE(fdmt.get_plan().df_top.size() == 10);
        REQUIRE(fdmt.get_plan().df_bot.size() == 10);
        REQUIRE(fdmt.get_plan().dt_grid_sub_top.size() == 10);
        REQUIRE(fdmt.get_plan().state_shape.size() == 10);
        REQUIRE(fdmt.get_plan().sub_plan.size() == 10);
        REQUIRE(fdmt.get_dt_grid_final().size() == 513);
        REQUIRE(fdmt.get_dm_grid_final().size() == 513);
    }
    SECTION("Test case 2: initialise method") {
        FDMTCPU fdmt(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 1, 0);
        std::vector<float> waterfall(static_cast<size_t>(500 * 1024), 1.0F);
        const auto& plan      = fdmt.get_plan();
        const auto state_size = plan.state_shape[0][3] * plan.state_shape[0][4];
        std::vector<float> state(state_size, 0.0F);
        REQUIRE_NOTHROW(
            fdmt.initialise(std::span(waterfall), std::span(state)));
    }

    SECTION("Test case 3: execute method") {
        FDMTCPU fdmt(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 1, 0);
        std::vector<float> waterfall(static_cast<size_t>(500 * 1024), 1.0F);
        const size_t dt_final_size = fdmt.get_dt_grid_final().size();
        std::vector<float> dmt(dt_final_size * 1024, 0.0F);
        REQUIRE_NOTHROW(fdmt.execute(std::span(waterfall), std::span(dmt)));
    }
}
