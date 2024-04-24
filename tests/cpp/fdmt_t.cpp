#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <dmt/fdmt.hpp>

TEST_CASE("FDMT class tests", "[fdmt]") {
    SECTION("Test case 1: Constructor and getter methods") {
        FDMT fdmt(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 1, 0);
        REQUIRE(fdmt.get_df() == 1.0F);
        REQUIRE(fdmt.get_correction() == 0.5F);
        REQUIRE(fdmt.get_dt_grid_final().size() == 513);
        REQUIRE(fdmt.get_niters() == 9);
        REQUIRE(fdmt.get_plan().df_top.size() == 10);
        REQUIRE(fdmt.get_plan().df_bot.size() == 10);
        REQUIRE(fdmt.get_plan().state_shape.size() == 10);
        REQUIRE(fdmt.get_plan().sub_plan.size() == 10);
    }
    SECTION("Test case 2: initialise method") {
        FDMT fdmt(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 1, 0);
        std::vector<float> waterfall(500 * 1024, 1.0f);
        const size_t dt_init_size = fdmt.get_dt_grid_init().size();
        std::vector<float> state(500 * 1024 * dt_init_size, 0.0f);
        REQUIRE_NOTHROW(fdmt.initialise(waterfall.data(), state.data()));
    }

    SECTION("Test case 3: execute method") {
        FDMT fdmt(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 1, 0);
        std::vector<float> waterfall(500 * 1024, 1.0f);
        const size_t dt_final_size = fdmt.get_dt_grid_final().size();
        std::vector<float> dmt(dt_final_size * 1024, 0.0f);
        REQUIRE_NOTHROW(fdmt.execute(waterfall.data(), waterfall.size(),
                                     dmt.data(), dmt.size()));
    }
}
