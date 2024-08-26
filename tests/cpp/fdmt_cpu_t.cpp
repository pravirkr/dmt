#include <catch2/catch_test_macros.hpp>

#include <spdlog/spdlog.h>

#include <cstddef>
#include <dmt/fdmt_cpu.hpp>

TEST_CASE("FDMTCPU", "[fdmt_cpu]") {
    FDMTCPU::set_log_level(spdlog::level::debug);
    SECTION("Constructor and getter methods") {
        FDMTCPU fdmt(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 1, 0);
        REQUIRE(fdmt.get_plan().get_dt_grid_final().size() == 513);
        REQUIRE(fdmt.get_plan().get_dm_grid_final().size() == 513);
        REQUIRE(fdmt.get_plan().get_dmt_size() == 513 * (1024 + 512));
    }
    SECTION("initialise method") {
        FDMTCPU fdmt(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 1, 0);
        std::vector<float> waterfall(static_cast<size_t>(500 * 1024), 1.0F);
        std::vector<float> state(fdmt.get_plan().get_buffer_size(), 0.0F);
        REQUIRE_NOTHROW(fdmt.initialise(waterfall.data(), waterfall.size(),
                                        state.data(), state.size()));
    }

    SECTION("execute method") {
        FDMTCPU::set_num_threads(1);
        FDMTCPU fdmt(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 1, 0);
        std::vector<float> waterfall(static_cast<size_t>(500 * 1024), 1.0F);
        std::vector<float> dmt(fdmt.get_plan().get_dmt_size(), 0.0F);
        REQUIRE_NOTHROW(fdmt.execute(waterfall.data(), waterfall.size(),
                                     dmt.data(), dmt.size()));
    }
}
