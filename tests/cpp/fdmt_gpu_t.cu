#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <thrust/device_vector.h>

#include <dmt/fdmt_gpu.hpp>

TEST_CASE("FDMT class tests [GPU]", "[fdmt_gpu]") {
    SECTION("Test case 1: Constructor and getter methods") {
        FDMTGPU fdmt(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 1, 0);
        REQUIRE(fdmt.get_df() == 1.0F);
        REQUIRE(fdmt.get_correction() == 0.5F);
        REQUIRE(fdmt.get_niters() == 9);
        REQUIRE(fdmt.get_plan().df_top.size() == 10);
        REQUIRE(fdmt.get_plan().df_bot.size() == 10);
        REQUIRE(fdmt.get_plan().dt_grid_sub_top.size() == 10);
        REQUIRE(fdmt.get_plan().state_shape.size() == 10);
        REQUIRE(fdmt.get_dt_grid_final().size() == 513);
        REQUIRE(fdmt.get_dm_grid_final().size() == 513);
    }
    SECTION("Test case 2: initialise method") {
        FDMTGPU fdmt(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 1, 0);
        thrust::device_vector<float> waterfall_d(
            static_cast<size_t>(500 * 1024), 1.0F);
        float* waterfall_d_ptr = thrust::raw_pointer_cast(waterfall_d.data());
        const auto& plan       = fdmt.get_plan();
        const auto state_size = plan.state_shape[0][3] * plan.state_shape[0][4];
        thrust::device_vector<float> state_d(state_size, 0.0F);
        float* state_d_ptr = thrust::raw_pointer_cast(state_d.data());
        REQUIRE_NOTHROW(fdmt.initialise(waterfall_d_ptr, state_d_ptr));
    }

    SECTION("Test case 3: execute method") {
        FDMTGPU fdmt(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 1, 0);
        thrust::device_vector<float> waterfall_d(
            static_cast<size_t>(500 * 1024), 1.0F);
        float* waterfall_d_ptr = thrust::raw_pointer_cast(waterfall_d.data());
        const size_t dt_final_size = fdmt.get_dt_grid_final().size();
        thrust::device_vector<float> dmt_d(dt_final_size * 1024, 0.0F);
        float* dmt_d_ptr = thrust::raw_pointer_cast(dmt_d.data());
        REQUIRE_NOTHROW(fdmt.execute(waterfall_d_ptr, waterfall_d.size(),
                                     dmt_d_ptr, dmt_d.size()));
    }
}