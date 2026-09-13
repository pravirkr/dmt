#include <algorithm>
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
        FDMTCUDA fdmt_cuda(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 0);
        FDMTCPU fdmt_cpu(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 0);
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

    SECTION("execute method (on device)") {
        FDMTCPU fdmt_cpu(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 0);
        fdmt_cpu.get_plan().print_summary();
        FDMTCUDA fdmt_cuda(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 0);
        std::vector<float> waterfall(static_cast<size_t>(500 * 1024), 1.0F);
        thrust::device_vector<float> waterfall_d = waterfall;
        const size_t dmt_size = fdmt_cpu.get_plan().get_buffer_size();
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
        FDMTCPU fdmt_cpu(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 0);
        fdmt_cpu.get_plan().print_summary();
        FDMTCUDA fdmt_cuda(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 0);
        std::vector<float> waterfall(static_cast<size_t>(500 * 1024), 1.0F);
        const size_t dmt_size = fdmt_cpu.get_plan().get_buffer_size();
        std::vector<float> dmt(dmt_size, 0.0F);
        std::vector<float> dmt_h(dmt_size, 0.0F);
        REQUIRE_NOTHROW(fdmt_cpu.execute(waterfall, dmt));
        REQUIRE_NOTHROW(fdmt_cuda.execute(std::span<const float>(waterfall),
                                          std::span<float>(dmt_h)));
        REQUIRE_THAT(dmt_h, Catch::Matchers::Approx(dmt).margin(0.0001));
    }

    SECTION(
        "stepper: bit-exact equivalence with single-shot execute on device") {
        FDMTCUDA fdmt_cuda(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 0);
        FDMTCPU fdmt_cpu(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 0);
        std::vector<float> waterfall(static_cast<size_t>(500 * 1024), 1.0F);
        thrust::device_vector<float> waterfall_d = waterfall;
        const size_t dmt_size = fdmt_cpu.get_plan().get_buffer_size();
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
        FDMTCUDA fdmt_cuda(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 0);
        FDMTCPU fdmt_cpu(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 0);
        std::vector<float> waterfall(static_cast<size_t>(500 * 1024), 1.0F);
        thrust::device_vector<float> waterfall_d = waterfall;
        const size_t dmt_size = fdmt_cpu.get_plan().get_buffer_size();
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
        FDMTCUDA fdmt_cuda(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 0);
        std::vector<float> waterfall(static_cast<size_t>(500 * 1024), 1.0F);
        thrust::device_vector<float> waterfall_d = waterfall;
        const size_t dmt_size = fdmt_cuda.get_plan().get_buffer_size();
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
        FDMTCUDA fdmt_cuda(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 0);
        CHECK_THROWS_AS(fdmt_cuda.advance(), std::logic_error);
        CHECK_THROWS_AS(fdmt_cuda.view_level_data(), std::logic_error);
        CHECK_THROWS_AS(fdmt_cuda.view_subband(0), std::logic_error);
    }

    SECTION("valid mode parity between CPU and CUDA") {
        FDMTCUDA fdmt_cuda(1000.0F, 1500.0F, 16, 1024, 0.001F, 64, 0, true, "valid");
        FDMTCPU fdmt_cpu(1000.0F, 1500.0F, 16, 1024, 0.001F, 64, 0, true, "valid");

        std::vector<float> wf1(16 * 1024, 1.0F);
        std::vector<float> wf2(16 * 1024, 2.0F);
        const size_t dmt_size = fdmt_cpu.get_plan().get_buffer_size();

        std::vector<float> dmt_cpu1(dmt_size, 0.0F);
        std::vector<float> dmt_cuda1(dmt_size, 0.0F);
        fdmt_cpu.execute(wf1, dmt_cpu1);
        fdmt_cuda.execute(wf1, dmt_cuda1);
        REQUIRE_THAT(dmt_cuda1, Catch::Matchers::Approx(dmt_cpu1).margin(0.0001));

        std::vector<float> dmt_cpu2(dmt_size, 0.0F);
        std::vector<float> dmt_cuda2(dmt_size, 0.0F);
        fdmt_cpu.execute(wf2, dmt_cpu2);
        fdmt_cuda.execute(wf2, dmt_cuda2);
        REQUIRE_THAT(dmt_cuda2, Catch::Matchers::Approx(dmt_cpu2).margin(0.0001));
    }

    SECTION("roll mode parity between CPU and CUDA") {
        FDMTCUDA fdmt_cuda(1000.0F, 1500.0F, 16, 1024, 0.001F, 64, 0, true, "roll");
        FDMTCPU fdmt_cpu(1000.0F, 1500.0F, 16, 1024, 0.001F, 64, 0, true, "roll");

        std::vector<float> wf(16 * 1024);
        for (size_t i = 0; i < wf.size(); ++i) {
            wf[i] = static_cast<float>((i % 13) + 1);
        }
        const size_t dmt_size = fdmt_cpu.get_plan().get_buffer_size();

        std::vector<float> dmt_cpu(dmt_size, 0.0F);
        std::vector<float> dmt_cuda(dmt_size, 0.0F);
        fdmt_cpu.execute(wf, dmt_cpu);
        fdmt_cuda.execute(wf, dmt_cuda);
        REQUIRE_THAT(dmt_cuda, Catch::Matchers::Approx(dmt_cpu).margin(0.0001));
    }

    SECTION("odd channels and padding safety on device") {
        const float f_min   = 1000.0F;
        const float f_max   = 1500.0F;
        const size_t nsamps = 256;
        const float tsamp   = 0.001F;
        const size_t dt_max = 32;
        const size_t dt_min = 0;

        const std::vector<size_t> odd_chans = {13, 63};

        for (const auto nchans : odd_chans) {
            DYNAMIC_SECTION("nchans = " << nchans) {
                FDMTCUDA fdmt_cuda(f_min, f_max, nchans, nsamps, tsamp, dt_max,
                                   dt_min);
                FDMTCPU fdmt_cpu(f_min, f_max, nchans, nsamps, tsamp, dt_max,
                                 dt_min);
                const auto& plan = fdmt_cpu.get_plan();

                std::vector<float> waterfall(nchans * nsamps);
                for (size_t i = 0; i < waterfall.size(); ++i) {
                    waterfall[i] = static_cast<float>((i % 7) + 1);
                }

                std::vector<float> dmt_ref(plan.get_buffer_size(), 0.0F);
                fdmt_cpu.execute(waterfall, dmt_ref);

                thrust::device_vector<float> waterfall_d = waterfall;
                thrust::device_vector<float> dmt_d(plan.get_buffer_size(),
                                                   -42.0F);
                auto d_wf_span = cuda::std::span<const float>(
                    thrust::raw_pointer_cast(waterfall_d.data()),
                    waterfall_d.size());
                auto d_dmt_span = cuda::std::span<float>(
                    thrust::raw_pointer_cast(dmt_d.data()), dmt_d.size());

                fdmt_cuda.reset(d_wf_span, d_dmt_span);
                while (!fdmt_cuda.is_finished()) {
                    fdmt_cuda.advance(1);
                }
                fdmt_cuda.finalize();

                std::vector<float> dmt_h(plan.get_buffer_size(), 0.0F);
                thrust::copy(dmt_d.begin(), dmt_d.end(), dmt_h.begin());

                const auto dmt_size = plan.get_dmt_size();
                for (size_t i = 0; i < dmt_size; ++i) {
                    REQUIRE_THAT(dmt_h[i],
                                 Catch::Matchers::WithinAbs(dmt_ref[i], 0.0001F));
                }
            }
        }
    }

    SECTION("no box smearing parity between CPU and CUDA") {
        FDMTCUDA fdmt_cuda(1000.0F, 1500.0F, 16, 1024, 0.001F, 64, 0, false, "full");
        FDMTCPU fdmt_cpu(1000.0F, 1500.0F, 16, 1024, 0.001F, 64, 0, false, "full");

        std::vector<float> wf(16 * 1024);
        for (size_t i = 0; i < wf.size(); ++i) {
            wf[i] = static_cast<float>((i % 19) + 1);
        }
        const size_t dmt_size = fdmt_cpu.get_plan().get_buffer_size();

        std::vector<float> dmt_cpu(dmt_size, 0.0F);
        std::vector<float> dmt_cuda(dmt_size, 0.0F);
        fdmt_cpu.execute(wf, dmt_cpu);
        fdmt_cuda.execute(wf, dmt_cuda);
        REQUIRE_THAT(dmt_cuda, Catch::Matchers::Approx(dmt_cpu).margin(0.0001));
    }

    SECTION("negative and symmetric dispersion parity between CPU and CUDA") {
        // Exercises the sign-aware level-0 init path in kernel_init_fdmt
        // (mirroring the CPU fdmt_init_subband fix) and the IndexType
        // dt_max/dt_min plumbing through FDMTCUDA/compute_fdmt_cuda.
        const float f_min      = 1000.0F;
        const float f_max      = 1500.0F;
        const size_t nchans    = 32;
        const size_t nsamps    = 512;
        const float tsamp      = 0.001F;
        const IndexType dt_max = 32;
        const IndexType dt_min = -32;

        for (const bool use_box_smearing : {false, true}) {
            DYNAMIC_SECTION("use_box_smearing=" << use_box_smearing) {
                FDMTCUDA fdmt_cuda(f_min, f_max, nchans, nsamps, tsamp, dt_max,
                                   dt_min, 1, use_box_smearing);
                FDMTCPU fdmt_cpu(f_min, f_max, nchans, nsamps, tsamp, dt_max,
                                 dt_min, 1, use_box_smearing);

                CHECK(fdmt_cuda.get_plan().get_dt_min() == -32);
                CHECK(fdmt_cuda.get_plan().get_dt_max() == 32);

                std::vector<float> wf(nchans * nsamps);
                for (size_t i = 0; i < wf.size(); ++i) {
                    wf[i] = static_cast<float>((i % 23) + 1);
                }
                const size_t dmt_size = fdmt_cpu.get_plan().get_buffer_size();
                std::vector<float> dmt_cpu(dmt_size, 0.0F);
                std::vector<float> dmt_cuda(dmt_size, 0.0F);
                fdmt_cpu.execute(wf, dmt_cpu);
                fdmt_cuda.execute(wf, dmt_cuda);
                REQUIRE_THAT(dmt_cuda,
                             Catch::Matchers::Approx(dmt_cpu).margin(0.0001));
            }
        }
    }

    SECTION("add_frb_track pulse recovery parity between CPU and CUDA") {
        const float f_min      = 1000.0F;
        const float f_max      = 1500.0F;
        const size_t nchans    = 64;
        const size_t nsamps    = 512;
        const float tsamp      = 0.001F;
        const IndexType dt_max = 32;
        const IndexType dt_min = -32;
        const SizeType toffset = 150;

        FDMTCPU fdmt_cpu(f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_min);
        FDMTCUDA fdmt_cuda(f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_min);
        const auto& plan = fdmt_cpu.get_plan();

        std::vector<float> waterfall(nchans * nsamps, 0.0F);
        algorithms::add_frb_track(waterfall, plan, /*dm_idx=*/16, 1.0F,
                                  static_cast<IndexType>(toffset), 1);

        const size_t dmt_size = plan.get_buffer_size();
        std::vector<float> dmt_cpu(dmt_size, 0.0F);
        std::vector<float> dmt_cuda(dmt_size, 0.0F);
        fdmt_cpu.execute(waterfall, dmt_cpu);
        fdmt_cuda.execute(waterfall, dmt_cuda);

        const auto dmt_nsamps = plan.get_dmt_nsamps();
        const auto value_cpu  = dmt_cpu[(16 * dmt_nsamps) + toffset];
        const auto value_cuda = dmt_cuda[(16 * dmt_nsamps) + toffset];
        CHECK(value_cpu == Catch::Approx(static_cast<float>(nchans)));
        CHECK(value_cuda == Catch::Approx(static_cast<float>(nchans)));
    }

    SECTION("get_effective_variance parity between CPU and CUDA") {
        FDMTCPU fdmt_cpu(1000.0F, 1500.0F, 32, 512, 0.001F, 32, -32);
        FDMTCUDA fdmt_cuda(1000.0F, 1500.0F, 32, 512, 0.001F, 32, -32);
        for (size_t dm_idx = 0; dm_idx < fdmt_cpu.get_plan().get_dmt_ndms();
             dm_idx += 8) {
            CHECK(fdmt_cuda.get_effective_variance(dm_idx, 4) ==
                  Catch::Approx(fdmt_cpu.get_effective_variance(dm_idx, 4)));
        }
        REQUIRE_THAT(fdmt_cuda.get_effective_variance_grid(2),
                    Catch::Matchers::Approx(
                        fdmt_cpu.get_effective_variance_grid(2)));
    }

    // --- Tree-history streaming (mode="valid") on CUDA -----------------
    // NOT compiled/run in this environment (no nvcc/GPU here) -- ported
    // from the CPU-side tests of the same name in fdmt_cpu_t.cpp/
    // test_libdmt.py, which are built and verified. Please build with
    // DMT_ENABLE_CUDA and run these before trusting the CUDA tree-history
    // port (kernel_execute_iter's hist_in/hist_out ping-pong in fdmt_cuda.cu).

    SECTION("valid-mode tree-history streaming parity between CPU and CUDA") {
        const float f_min      = 1000.0F;
        const float f_max      = 1500.0F;
        const size_t nchans    = 32;
        const float tsamp      = 0.001F;
        const IndexType dt_max = 32;
        const IndexType dt_min = -32;
        const size_t block_size = 128;
        const size_t n_blocks   = 5;
        const size_t total      = block_size * n_blocks;

        FDMTCPU fdmt_full(f_min, f_max, nchans, total, tsamp, dt_max, dt_min,
                          1, true, "full");
        std::vector<float> waterfall(nchans * total);
        for (size_t i = 0; i < waterfall.size(); ++i) {
            waterfall[i] = static_cast<float>((i % 23) + 1);
        }
        std::vector<float> dmt_full(fdmt_full.get_plan().get_buffer_size(),
                                    0.0F);
        fdmt_full.execute(waterfall, dmt_full);
        const auto full_nsamps = fdmt_full.get_plan().get_dmt_nsamps();

        FDMTCPU fdmt_cpu_valid(f_min, f_max, nchans, block_size, tsamp,
                              dt_max, dt_min, 1, true, "valid");
        FDMTCUDA fdmt_cuda_valid(f_min, f_max, nchans, block_size, tsamp,
                                dt_max, dt_min, 1, true, "valid");
        const auto ndms = fdmt_cpu_valid.get_plan().get_dmt_ndms();
        REQUIRE(ndms == fdmt_cuda_valid.get_plan().get_dmt_ndms());

        std::vector<float> streamed_cpu(ndms * total, 0.0F);
        std::vector<float> streamed_cuda(ndms * total, 0.0F);
        for (size_t b = 0; b < n_blocks; ++b) {
            std::vector<float> block(nchans * block_size);
            for (size_t c = 0; c < nchans; ++c) {
                std::copy_n(waterfall.data() + (c * total) + (b * block_size),
                           block_size, block.data() + (c * block_size));
            }
            std::vector<float> dmt_block_cpu(
                fdmt_cpu_valid.get_plan().get_buffer_size(), 0.0F);
            std::vector<float> dmt_block_cuda(
                fdmt_cuda_valid.get_plan().get_buffer_size(), 0.0F);
            fdmt_cpu_valid.execute(block, dmt_block_cpu);
            fdmt_cuda_valid.execute(block, dmt_block_cuda);
            for (size_t d = 0; d < ndms; ++d) {
                std::copy_n(dmt_block_cpu.data() + (d * block_size),
                           block_size,
                           streamed_cpu.data() + (d * total) + (b * block_size));
                std::copy_n(dmt_block_cuda.data() + (d * block_size),
                           block_size,
                           streamed_cuda.data() + (d * total) + (b * block_size));
            }
        }

        // Both backends must agree with each other AND with the monolithic
        // full-mode reference (this is the actual regression this test
        // guards: before the CUDA port, CUDA had no cross-block tree
        // history at all, so it would only match dmt_full at dt=0).
        const auto abs_dt_max = static_cast<size_t>(
            std::max(std::abs(dt_min), std::abs(dt_max)));
        for (size_t d = 0; d < ndms; ++d) {
            for (size_t t = abs_dt_max; t < total; ++t) {
                const auto ref = dmt_full[(d * full_nsamps) + t];
                CHECK(streamed_cpu[(d * total) + t] ==
                     Catch::Approx(ref).margin(1e-3));
                CHECK(streamed_cuda[(d * total) + t] ==
                     Catch::Approx(ref).margin(1e-3));
            }
        }
    }

    SECTION("reset_history() on CUDA matches a fresh instance") {
        const float f_min      = 1000.0F;
        const float f_max      = 1500.0F;
        const size_t nchans    = 32;
        const size_t nsamps    = 128;
        const float tsamp      = 0.001F;
        const IndexType dt_max = 32;
        const IndexType dt_min = -32;

        std::vector<float> block1(nchans * nsamps);
        std::vector<float> block2(nchans * nsamps);
        for (size_t i = 0; i < block1.size(); ++i) {
            block1[i] = static_cast<float>((i % 17) + 1);
            block2[i] = static_cast<float>((i % 13) + 1);
        }

        FDMTCUDA fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_min, 1,
                     true, "valid");
        std::vector<float> dmt1(fdmt.get_plan().get_buffer_size(), 0.0F);
        std::vector<float> dmt2_with_history(
            fdmt.get_plan().get_buffer_size(), 0.0F);
        fdmt.execute(block1, dmt1);
        fdmt.execute(block2, dmt2_with_history);

        fdmt.reset_history();
        std::vector<float> dmt2_after_reset(
            fdmt.get_plan().get_buffer_size(), 0.0F);
        fdmt.execute(block2, dmt2_after_reset);

        FDMTCUDA fdmt_fresh(f_min, f_max, nchans, nsamps, tsamp, dt_max,
                           dt_min, 1, true, "valid");
        std::vector<float> dmt2_fresh(fdmt_fresh.get_plan().get_buffer_size(),
                                      0.0F);
        fdmt_fresh.execute(block2, dmt2_fresh);

        REQUIRE_THAT(dmt2_after_reset,
                    Catch::Matchers::Approx(dmt2_fresh).margin(0.0001));
        CHECK_FALSE(std::equal(dmt2_with_history.begin(),
                               dmt2_with_history.end(),
                               dmt2_after_reset.begin()));
    }

    SECTION("add_frb_track recovery across a valid-mode block boundary on "
           "CUDA") {
        const float f_min         = 1000.0F;
        const float f_max         = 1500.0F;
        const size_t nchans       = 64;
        const size_t block_size   = 128;
        const size_t n_blocks     = 4;
        const size_t total        = block_size * n_blocks;
        const float tsamp         = 0.001F;
        const IndexType dt_max    = 32;
        const IndexType dt_min    = -32;
        const size_t toffset      = block_size + 5; // straddles the boundary

        plans::FDMTPlan plan(f_min, f_max, nchans, total, tsamp, dt_max,
                            dt_min);
        const auto& dt_grid = plan.get_dt_grid_final();
        const auto it = std::find(dt_grid.begin(), dt_grid.end(),
                                  IndexType{-16});
        REQUIRE(it != dt_grid.end());
        const auto dm_idx =
            static_cast<SizeType>(std::distance(dt_grid.begin(), it));

        std::vector<float> waterfall(nchans * total, 0.0F);
        algorithms::add_frb_track(waterfall, plan, dm_idx, 1.0F,
                                  static_cast<IndexType>(toffset), 1);

        FDMTCUDA fdmt_valid(f_min, f_max, nchans, block_size, tsamp, dt_max,
                           dt_min, 1, false, "valid");
        const auto ndms = plan.get_dmt_ndms();
        std::vector<float> streamed(ndms * total, 0.0F);
        for (size_t b = 0; b < n_blocks; ++b) {
            std::vector<float> block(nchans * block_size);
            for (size_t c = 0; c < nchans; ++c) {
                std::copy_n(waterfall.data() + (c * total) + (b * block_size),
                           block_size, block.data() + (c * block_size));
            }
            std::vector<float> dmt_block(fdmt_valid.get_plan().get_buffer_size(),
                                        0.0F);
            fdmt_valid.execute(block, dmt_block);
            for (size_t d = 0; d < ndms; ++d) {
                std::copy_n(dmt_block.data() + (d * block_size), block_size,
                           streamed.data() + (d * total) + (b * block_size));
            }
        }

        CHECK(streamed[(dm_idx * total) + toffset] ==
             Catch::Approx(static_cast<float>(nchans)));
    }
}

} // namespace dmt