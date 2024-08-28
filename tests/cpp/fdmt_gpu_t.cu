#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include <cstddef>
#include <spdlog/spdlog.h>
#include <thrust/device_vector.h>

#include <dmt/fdmt_cpu.hpp>
#include <dmt/fdmt_gpu.hpp>

TEST_CASE("FDMTGPU", "[fdmt_gpu]") {
    FDMTGPU::set_log_level(spdlog::level::debug);
    SECTION("Constructor and getter methods") {
        FDMTGPU fdmt_gpu(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 1, 0);
        FDMTCPU fdmt_cpu(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 1, 0);
        REQUIRE(fdmt_gpu.get_plan().get_df() == fdmt_cpu.get_plan().get_df());
        REQUIRE(fdmt_gpu.get_plan().get_niters() ==
                fdmt_cpu.get_plan().get_niters());
        REQUIRE_THAT(
            fdmt_gpu.get_plan().get_dt_grid_final(),
            Catch::Matchers::Equals(fdmt_cpu.get_plan().get_dt_grid_final()));
        REQUIRE_THAT(
            fdmt_gpu.get_plan().get_dm_grid_final(),
            Catch::Matchers::Equals(fdmt_cpu.get_plan().get_dm_grid_final()));
    }
    SECTION("initialise method (on device)") {
        FDMTGPU fdmt_gpu(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 1, 0);
        FDMTCPU fdmt_cpu(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 1, 0);
        std::vector<float> waterfall(static_cast<size_t>(500 * 1024), 1.0F);
        thrust::device_vector<float> waterfall_d = waterfall;

        const auto state_size = fdmt_cpu.get_plan().get_buffer_size();
        std::vector<float> state(state_size, 0.0F);
        thrust::device_vector<float> state_d = state;
        REQUIRE_NOTHROW(fdmt_cpu.initialise(waterfall.data(), waterfall.size(),
                                            state.data(), state.size()));

        const float* waterfall_d_ptr =
            thrust::raw_pointer_cast(waterfall_d.data());
        float* state_d_ptr = thrust::raw_pointer_cast(state_d.data());
        REQUIRE_NOTHROW(fdmt_gpu.initialise(waterfall_d_ptr, waterfall_d.size(),
                                            state_d_ptr, state_d.size(), true));

        std::vector<float> state_h(state_size, 0.0F);
        thrust::copy(state_d.begin(), state_d.end(), state_h.begin());
        REQUIRE_THAT(state_h, Catch::Matchers::Approx(state).margin(0.0001));
    }

    SECTION("initialise method (on host)") {
        FDMTGPU fdmt_gpu(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 1, 0);
        FDMTCPU fdmt_cpu(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 1, 0);
        std::vector<float> waterfall(static_cast<size_t>(500 * 1024), 1.0F);
        const auto state_size = fdmt_cpu.get_plan().get_buffer_size();
        std::vector<float> state(state_size, 0.0F);
        std::vector<float> state_h(state_size, 0.0F);
        REQUIRE_NOTHROW(fdmt_cpu.initialise(waterfall.data(), waterfall.size(),
                                            state.data(), state.size()));
        REQUIRE_NOTHROW(fdmt_gpu.initialise(waterfall.data(), waterfall.size(),
                                            state_h.data(), state_h.size()));
        REQUIRE_THAT(state_h, Catch::Matchers::Approx(state).margin(0.0001));
    }

    SECTION("execute method (on device)") {
        FDMTCPU fdmt_cpu(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 1, 0);
        fdmt_cpu.get_plan().print_summary();
        FDMTGPU fdmt_gpu(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 1, 0);
        std::vector<float> waterfall(static_cast<size_t>(500 * 1024), 1.0F);
        thrust::device_vector<float> waterfall_d = waterfall;
        const size_t dmt_size = fdmt_cpu.get_plan().get_dmt_size();
        std::vector<float> dmt(dmt_size, 0.0F);
        thrust::device_vector<float> dmt_d = dmt;
        REQUIRE_NOTHROW(fdmt_cpu.execute(waterfall.data(), waterfall.size(),
                                         dmt.data(), dmt.size()));

        const float* waterfall_d_ptr =
            thrust::raw_pointer_cast(waterfall_d.data());
        float* dmt_d_ptr = thrust::raw_pointer_cast(dmt_d.data());
        REQUIRE_NOTHROW(fdmt_gpu.execute(waterfall_d_ptr, waterfall_d.size(),
                                         dmt_d_ptr, dmt_d.size(), true));

        std::vector<float> dmt_h(dmt_size, 0.0F);
        thrust::copy(dmt_d.begin(), dmt_d.end(), dmt_h.begin());
        REQUIRE_THAT(dmt_h, Catch::Matchers::Approx(dmt).margin(0.0001));
    }

    SECTION("execute method (on host)") {
        FDMTCPU fdmt_cpu(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 1, 0);
        fdmt_cpu.get_plan().print_summary();
        FDMTGPU fdmt_gpu(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 1, 0);
        std::vector<float> waterfall(static_cast<size_t>(500 * 1024), 1.0F);
        const size_t dmt_size = fdmt_cpu.get_plan().get_dmt_size();
        std::vector<float> dmt(dmt_size, 0.0F);
        std::vector<float> dmt_h(dmt_size, 0.0F);
        REQUIRE_NOTHROW(fdmt_cpu.execute(waterfall.data(), waterfall.size(),
                                         dmt.data(), dmt.size()));
        REQUIRE_NOTHROW(fdmt_gpu.execute(waterfall.data(), waterfall.size(),
                                         dmt_h.data(), dmt_h.size()));
        REQUIRE_THAT(dmt_h, Catch::Matchers::Approx(dmt).margin(0.0001));
    }
}