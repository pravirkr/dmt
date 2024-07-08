#include <catch2/catch_test_macros.hpp>

#include <spdlog/spdlog.h>

#include <cstddef>
#include <dmt/fdmt_cpu.hpp>

TEST_CASE("FDMT class tests [CPU]", "[fdmt_cpu]") {
    SECTION("Test case 1: Constructor and getter methods") {
        FDMTCPU::set_log_level(spdlog::level::debug);
        FDMTCPU fdmt(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 1, 0);
        REQUIRE(fdmt.get_df() == 1.0F);
        REQUIRE(fdmt.get_correction() == 0.5F);
        REQUIRE(fdmt.get_niters() == 9);
        REQUIRE(fdmt.get_plan().df_top.size() == 10);
        REQUIRE(fdmt.get_plan().df_bot.size() == 10);
        REQUIRE(fdmt.get_plan().dt_grid_sub_top.size() == 10);
        REQUIRE(fdmt.get_plan().state_shape.size() == 10);
        REQUIRE(fdmt.get_dt_grid_final().size() == 513);
        REQUIRE(fdmt.get_dm_grid_final().size() == 513);
        REQUIRE(fdmt.get_dmt_size() == 513 * (1024 + 512));
    }
    SECTION("Test case 2: initialise method") {
        FDMTCPU::set_log_level(1);
        FDMTCPU fdmt(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 1, 0);
        std::vector<float> waterfall(static_cast<size_t>(500 * 1024), 1.0F);
        const auto& plan      = fdmt.get_plan();
        const auto state_size = plan.state_shape[0][3] * plan.state_shape[0][4];
        std::vector<float> state(state_size, 0.0F);
        REQUIRE_NOTHROW(fdmt.initialise(waterfall.data(), waterfall.size(),
                                        state.data(), state.size()));
    }

    SECTION("Test case 3: execute method") {
        FDMTCPU::set_num_threads(1);
        FDMTCPU::set_log_level(1);
        FDMTCPU fdmt(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 1, 0);
        std::vector<float> waterfall(static_cast<size_t>(500 * 1024), 1.0F);
        std::vector<float> dmt(fdmt.get_dmt_size(), 0.0F);
        REQUIRE_NOTHROW(fdmt.execute(waterfall.data(), waterfall.size(),
                                     dmt.data(), dmt.size()));
    }
}
