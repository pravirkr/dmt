#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include <cstddef>
#include <cuda/std/span>
#include <span>
#include <spdlog/spdlog.h>
#include <thrust/device_vector.h>

#include "dmt/algorithms/fdmt.hpp"

namespace dmt {

using algorithms::FDMTCPU;
using algorithms::FDMTCUDA;

TEST_CASE("FDMTGPU", "[fdmt_gpu]") {
    SECTION("Constructor and getter methods") {
        FDMTCUDA fdmt_cuda(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 1, 0);
        FDMTCPU fdmt_cpu(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 1, 0);
        REQUIRE(fdmt_cuda.get_plan().get_df() == fdmt_cpu.get_plan().get_df());
        REQUIRE(fdmt_cuda.get_plan().get_niters() ==
                fdmt_cpu.get_plan().get_niters());
        REQUIRE_THAT(
            fdmt_cuda.get_plan().get_dt_grid_final(),
            Catch::Matchers::Equals(fdmt_cpu.get_plan().get_dt_grid_final()));
        REQUIRE_THAT(
            fdmt_cuda.get_plan().get_dm_grid_final(),
            Catch::Matchers::Equals(fdmt_cpu.get_plan().get_dm_grid_final()));
    }
    /*
    SECTION("initialise method (on device)") {
        FDMTCUDA fdmt_cuda(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 1, 0);
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
        REQUIRE_NOTHROW(fdmt_cuda.initialise(waterfall_d_ptr,
                                             waterfall_d.size(), state_d_ptr,
                                             state_d.size(), true));

        std::vector<float> state_h(state_size, 0.0F);
        thrust::copy(state_d.begin(), state_d.end(), state_h.begin());
        REQUIRE_THAT(state_h, Catch::Matchers::Approx(state).margin(0.0001));
    }


    SECTION("initialise method (on host)") {
        FDMTCUDA fdmt_cuda(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 1, 0);
        FDMTCPU fdmt_cpu(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 1, 0);
        std::vector<float> waterfall(static_cast<size_t>(500 * 1024), 1.0F);
        const auto state_size = fdmt_cpu.get_plan().get_buffer_size();
        std::vector<float> state(state_size, 0.0F);
        std::vector<float> state_h(state_size, 0.0F);
        REQUIRE_NOTHROW(fdmt_cpu.initialise(waterfall.data(), waterfall.size(),
                                            state.data(), state.size()));
        REQUIRE_NOTHROW(fdmt_cuda.initialise(waterfall.data(), waterfall.size(),
                                             state_h.data(), state_h.size()));
        REQUIRE_THAT(state_h, Catch::Matchers::Approx(state).margin(0.0001));
    }
    */

    SECTION("execute method (on device)") {
        FDMTCPU fdmt_cpu(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 1, 0);
        fdmt_cpu.get_plan().print_summary();
        FDMTCUDA fdmt_cuda(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 1, 0);
        std::vector<float> waterfall(static_cast<size_t>(500 * 1024), 1.0F);
        thrust::device_vector<float> waterfall_d = waterfall;
        const size_t dmt_size = fdmt_cpu.get_plan().get_dmt_size();
        std::vector<float> dmt(dmt_size, 0.0F);
        thrust::device_vector<float> dmt_d = dmt;
        REQUIRE_NOTHROW(fdmt_cpu.execute(waterfall, dmt));
        REQUIRE_NOTHROW(fdmt_cuda.execute(
            cuda::std::span<const float>(
                thrust::raw_pointer_cast(waterfall_d.data()),
                waterfall_d.size()),
            cuda::std::span<float>(thrust::raw_pointer_cast(dmt_d.data()),
                                   dmt_d.size())));

        std::vector<float> dmt_h(dmt_size, 0.0F);
        thrust::copy(dmt_d.begin(), dmt_d.end(), dmt_h.begin());
        REQUIRE_THAT(dmt_h, Catch::Matchers::Approx(dmt).margin(0.0001));
    }

    SECTION("execute method (on host)") {
        FDMTCPU fdmt_cpu(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 1, 0);
        fdmt_cpu.get_plan().print_summary();
        FDMTCUDA fdmt_cuda(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 1, 0);
        std::vector<float> waterfall(static_cast<size_t>(500 * 1024), 1.0F);
        const size_t dmt_size = fdmt_cpu.get_plan().get_dmt_size();
        std::vector<float> dmt(dmt_size, 0.0F);
        std::vector<float> dmt_h(dmt_size, 0.0F);
        REQUIRE_NOTHROW(fdmt_cpu.execute(waterfall, dmt));
        REQUIRE_NOTHROW(fdmt_cuda.execute(std::span<const float>(waterfall),
                                          std::span<float>(dmt_h)));
        REQUIRE_THAT(dmt_h, Catch::Matchers::Approx(dmt).margin(0.0001));
    }

    SECTION(
        "stepper: bit-exact equivalence with single-shot execute on device") {
        FDMTCUDA fdmt_cuda(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 1, 0);
        FDMTCPU fdmt_cpu(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 1, 0);
        std::vector<float> waterfall(static_cast<size_t>(500 * 1024), 1.0F);
        thrust::device_vector<float> waterfall_d = waterfall;
        const size_t dmt_size = fdmt_cpu.get_plan().get_dmt_size();
        std::vector<float> dmt_ref(dmt_size, 0.0F);
        fdmt_cpu.execute(waterfall, dmt_ref);

        thrust::device_vector<float> dmt_d(dmt_size, 0.0F);
        auto d_wf_span = cuda::std::span<const float>(
            thrust::raw_pointer_cast(waterfall_d.data()), waterfall_d.size());
        auto d_dmt_span = cuda::std::span<float>(
            thrust::raw_pointer_cast(dmt_d.data()), dmt_d.size());

        fdmt_cuda.reset(d_wf_span, d_dmt_span);
        CHECK(fdmt_cuda.current_level() == 0);
        CHECK(fdmt_cuda.remaining_levels() == fdmt_cuda.total_levels() - 1);
        CHECK(fdmt_cuda.num_subbands() == 500);

        fdmt_cuda.advance_until_remaining(0);
        CHECK(fdmt_cuda.is_finished());
        CHECK(fdmt_cuda.remaining_levels() == 0);
        fdmt_cuda.finalize();

        std::vector<float> dmt_h(dmt_size, 0.0F);
        thrust::copy(dmt_d.begin(), dmt_d.end(), dmt_h.begin());
        REQUIRE_THAT(dmt_h, Catch::Matchers::Approx(dmt_ref).margin(0.0001));
    }

    SECTION("stepper: 1 level remaining (2 subbands) on device") {
        FDMTCUDA fdmt_cuda(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 1, 0);
        FDMTCPU fdmt_cpu(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 1, 0);
        std::vector<float> waterfall(static_cast<size_t>(500 * 1024), 1.0F);
        thrust::device_vector<float> waterfall_d = waterfall;
        const size_t dmt_size = fdmt_cpu.get_plan().get_dmt_size();
        std::vector<float> dmt_ref(dmt_size, 0.0F);
        fdmt_cpu.execute(waterfall, dmt_ref);

        thrust::device_vector<float> dmt_d(dmt_size, 0.0F);
        auto d_wf_span = cuda::std::span<const float>(
            thrust::raw_pointer_cast(waterfall_d.data()), waterfall_d.size());
        auto d_dmt_span = cuda::std::span<float>(
            thrust::raw_pointer_cast(dmt_d.data()), dmt_d.size());

        fdmt_cuda.reset(d_wf_span, d_dmt_span);
        fdmt_cuda.advance_until_remaining(1);
        CHECK(fdmt_cuda.remaining_levels() == 1);
        CHECK(fdmt_cuda.num_subbands() == 2);
        CHECK(!fdmt_cuda.is_finished());

        auto sub0 = fdmt_cuda.view_subband(0);
        auto sub1 = fdmt_cuda.view_subband(1);
        CHECK(sub0.subband_idx == 0);
        CHECK(sub1.subband_idx == 1);
        CHECK(sub0.f_start == Catch::Approx(1000.0F));
        CHECK(sub0.f_end == Catch::Approx(sub1.f_start));
        CHECK(sub1.f_end == Catch::Approx(1500.0F));
        CHECK(sub0.data.size() == sub0.ndt * sub0.nsamps);
        CHECK(sub1.data.size() == sub1.ndt * sub1.nsamps);

        fdmt_cuda.finalize();
        CHECK(fdmt_cuda.is_finished());

        std::vector<float> dmt_h(dmt_size, 0.0F);
        thrust::copy(dmt_d.begin(), dmt_d.end(), dmt_h.begin());
        REQUIRE_THAT(dmt_h, Catch::Matchers::Approx(dmt_ref).margin(0.0001));
    }

    SECTION("stepper: 2 levels remaining (4 subbands) on device") {
        FDMTCUDA fdmt_cuda(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 1, 0);
        std::vector<float> waterfall(static_cast<size_t>(500 * 1024), 1.0F);
        thrust::device_vector<float> waterfall_d = waterfall;
        const size_t dmt_size = fdmt_cuda.get_plan().get_dmt_size();
        thrust::device_vector<float> dmt_d(dmt_size, 0.0F);

        auto d_wf_span = cuda::std::span<const float>(
            thrust::raw_pointer_cast(waterfall_d.data()), waterfall_d.size());
        auto d_dmt_span = cuda::std::span<float>(
            thrust::raw_pointer_cast(dmt_d.data()), dmt_d.size());

        fdmt_cuda.reset(d_wf_span, d_dmt_span);
        fdmt_cuda.advance_until_remaining(2);
        CHECK(fdmt_cuda.remaining_levels() == 2);
        CHECK(fdmt_cuda.num_subbands() == 4);

        for (size_t s = 0; s < 4; ++s) {
            auto sub       = fdmt_cuda.view_subband(s);
            auto data_span = fdmt_cuda.view_subband_data(s);
            CHECK(sub.data.data() == data_span.data());
            CHECK(sub.data.size() == data_span.size());
        }
    }

    SECTION("stepper: lifecycle and error handling") {
        FDMTCUDA fdmt_cuda(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 1, 0);
        CHECK_THROWS_AS(fdmt_cuda.advance(), std::logic_error);
        CHECK_THROWS_AS(fdmt_cuda.view_level_data(), std::logic_error);
        CHECK_THROWS_AS(fdmt_cuda.view_subband(0), std::logic_error);
    }
}

} // namespace dmt