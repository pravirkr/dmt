#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

#include "dmt/algorithms/fdmt.hpp"
#include "dmt/utils/simulate.hpp"

namespace dmt {

using algorithms::FDMTCPU;

TEST_CASE("FDMTCPU", "[fdmt_cpu]") {
    const float f_min   = 1000.0F;
    const float f_max   = 1500.0F;
    const size_t nchans = 500;
    const size_t nsamps = 1024;
    const float tsamp   = 0.001F;
    const size_t dt_max = 512;
    const size_t dt_min = 0;

    FDMTCPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_min);

    SECTION("Constructor and getter methods") {
        const auto& plan         = fdmt.get_plan();
        const auto ndms_expected = static_cast<SizeType>(dt_max - dt_min + 1);
        CHECK(plan.get_dt_grid_final().size() == ndms_expected);
        CHECK(plan.get_dm_grid_final().size() == ndms_expected);
        CHECK(plan.get_dmt_size() ==
              static_cast<SizeType>(ndms_expected * (nsamps + dt_max)));
    }

    SECTION("execute method") {
        std::vector<float> waterfall(nchans * nsamps, 1.0F);
        std::vector<float> dmt(fdmt.get_plan().get_buffer_size(), 0.0F);
        REQUIRE_NOTHROW(fdmt.execute(waterfall, dmt));
    }

    SECTION("stepper execution bit-exact match with execute") {
        std::vector<float> waterfall(nchans * nsamps);
        for (size_t i = 0; i < waterfall.size(); ++i) {
            waterfall[i] = static_cast<float>((i % 17) + 1);
        }
        std::vector<float> dmt_oneshot(fdmt.get_plan().get_buffer_size(), 0.0F);
        fdmt.execute(waterfall, dmt_oneshot);

        std::vector<float> dmt_stepped(fdmt.get_plan().get_buffer_size(), 0.0F);
        fdmt.reset(waterfall, dmt_stepped);
        CHECK(fdmt.current_level() == 0);
        CHECK(fdmt.total_levels() == fdmt.get_plan().get_niters() + 1);
        CHECK(fdmt.remaining_levels() == fdmt.total_levels() - 1);
        CHECK_FALSE(fdmt.is_finished());

        // Step by step to root
        while (!fdmt.is_finished()) {
            fdmt.advance(1);
        }
        CHECK(fdmt.is_finished());
        CHECK(fdmt.remaining_levels() == 0);
        fdmt.finalize();

        // Verify exact bit-for-bit equivalence
        const auto dmt_size = fdmt.get_plan().get_dmt_size();
        for (size_t i = 0; i < dmt_size; ++i) {
            REQUIRE(dmt_stepped[i] == dmt_oneshot[i]);
        }
    }

    SECTION("subband inspection at remaining_levels = 1 (2 children)") {
        std::vector<float> waterfall(nchans * nsamps, 1.0F);
        std::vector<float> dmt(fdmt.get_plan().get_buffer_size(), 0.0F);
        fdmt.reset(waterfall, dmt);

        fdmt.advance_until_remaining(1);
        CHECK(fdmt.remaining_levels() == 1);
        CHECK(fdmt.num_subbands() == 2);

        auto sub0 = fdmt.view_subband(0);
        auto sub1 = fdmt.view_subband(1);

        CHECK(sub0.subband_idx == 0);
        CHECK(sub1.subband_idx == 1);
        CHECK(sub0.ndt > 0);
        CHECK(sub1.ndt > 0);
        CHECK(sub0.nsamps > 0);
        CHECK(sub0.f_start == Catch::Approx(f_min));
        CHECK(sub1.f_end == Catch::Approx(f_max));
        CHECK(sub0.f_end == Catch::Approx(sub1.f_start));
        CHECK(sub0.data.size() == sub0.ndt * sub0.nsamps);
        CHECK(sub1.data.size() == sub1.ndt * sub1.nsamps);
        CHECK(!sub0.dt_grid.empty());
        CHECK(!sub1.dt_grid.empty());

        // view_subband_data gives identical slice
        auto sub0_data = fdmt.view_subband_data(0);
        CHECK(sub0_data.data() == sub0.data.data());
        CHECK(sub0_data.size() == sub0.data.size());

        // view_level_data covers all subbands at this level
        auto level_data = fdmt.view_level_data();
        CHECK(level_data.size() == sub0.data.size() + sub1.data.size());

        // Finalize to root
        fdmt.finalize();
        CHECK_FALSE(fdmt.is_finished()); // reset after finalize
    }

    SECTION("subband inspection at remaining_levels = 2 (4 children)") {
        std::vector<float> waterfall(nchans * nsamps, 1.0F);
        std::vector<float> dmt(fdmt.get_plan().get_buffer_size(), 0.0F);
        fdmt.reset(waterfall, dmt);

        fdmt.advance_until_remaining(2);
        CHECK(fdmt.remaining_levels() == 2);
        CHECK(fdmt.num_subbands() == 4);

        for (size_t s = 0; s < 4; ++s) {
            auto sub = fdmt.view_subband(s);
            CHECK(sub.subband_idx == s);
            CHECK(sub.data.size() == sub.ndt * sub.nsamps);
        }

        // Out of range check
        CHECK_THROWS_AS(fdmt.view_subband(4), std::out_of_range);
        CHECK_THROWS_AS(fdmt.view_subband_data(4), std::out_of_range);

        fdmt.finalize();
    }

    SECTION("stepper lifecycle and error handling") {
        std::vector<float> waterfall(nchans * nsamps, 1.0F);
        std::vector<float> dmt(fdmt.get_plan().get_buffer_size(), 0.0F);

        // Calling advance or view before reset should throw
        CHECK_THROWS_AS(fdmt.advance(), std::logic_error);
        CHECK_THROWS_AS(fdmt.advance_until_remaining(1), std::logic_error);
        CHECK_THROWS_AS(fdmt.view_level_data(), std::logic_error);
        CHECK_THROWS_AS(fdmt.view_subband(0), std::logic_error);
        CHECK_THROWS_AS(fdmt.view_subband_data(0), std::logic_error);
        CHECK_THROWS_AS(fdmt.num_subbands(), std::logic_error);
        CHECK_THROWS_AS(fdmt.finalize(), std::logic_error);

        // Invalid buffer sizes should throw invalid_argument
        std::vector<float> bad_wf(10, 1.0F);
        std::vector<float> bad_dmt(10, 0.0F);
        CHECK_THROWS_AS(fdmt.reset(bad_wf, dmt), std::invalid_argument);
        CHECK_THROWS_AS(fdmt.reset(waterfall, bad_dmt), std::invalid_argument);

        fdmt.reset(waterfall, dmt);
        // Advancing more than remaining levels stops cleanly at root
        fdmt.advance(100);
        CHECK(fdmt.is_finished());
        CHECK(fdmt.remaining_levels() == 0);

        REQUIRE_NOTHROW(fdmt.finalize());
    }
}

TEST_CASE("FDMTCPU modes and smearing matrix", "[fdmt_cpu]") {
    const float f_min   = 1000.0F;
    const float f_max   = 1500.0F;
    const size_t nchans = 64;
    const size_t nsamps = 256;
    const float tsamp   = 0.001F;
    const size_t dt_max = 32;
    const size_t dt_min = 0;

    const std::vector<std::string> modes = {"full", "roll", "valid"};
    const std::vector<bool> smearings    = {true, false};

    for (const auto& mode : modes) {
        for (const auto smearing : smearings) {
            DYNAMIC_SECTION("Mode: " << mode << ", Smearing: " << smearing) {
                FDMTCPU fdmt_sync(f_min, f_max, nchans, nsamps, tsamp, dt_max,
                                  dt_min, 1, smearing, mode);
                const auto& plan = fdmt_sync.get_plan();

                if (mode == "full") {
                    CHECK(plan.get_dmt_nsamps() == nsamps + dt_max);
                } else {
                    CHECK(plan.get_dmt_nsamps() == nsamps);
                }

                std::vector<float> waterfall(nchans * nsamps);
                for (size_t i = 0; i < waterfall.size(); ++i) {
                    waterfall[i] = static_cast<float>((i % 13) + 1);
                }

                // Synchronous execution
                std::vector<float> dmt_sync(plan.get_buffer_size(), 0.0F);
                fdmt_sync.execute(waterfall, dmt_sync);

                // Verify output has non-zero values
                float sum_sync = 0.0F;
                for (size_t i = 0; i < plan.get_dmt_size(); ++i) {
                    sum_sync += dmt_sync[i];
                }
                CHECK(sum_sync > 0.0F);

                // Stepper execution with fresh instance (matching cold-start
                // state) and dirty buffer reuse
                FDMTCPU fdmt_stepped(f_min, f_max, nchans, nsamps, tsamp,
                                     dt_max, dt_min, 1, smearing, mode);
                std::vector<float> dmt_stepped(plan.get_buffer_size(), 999.0F);
                fdmt_stepped.reset(waterfall, dmt_stepped);
                fdmt_stepped.advance_until_remaining(0);
                CHECK(fdmt_stepped.is_finished());
                fdmt_stepped.finalize();

                // Verify exact bit-for-bit equivalence between sync and stepper
                const auto dmt_size = plan.get_dmt_size();
                for (size_t i = 0; i < dmt_size; ++i) {
                    REQUIRE(dmt_stepped[i] == dmt_sync[i]);
                }

                // Also test second block on both: verify second block is
                // bit-exact
                std::vector<float> dmt_sync2(plan.get_buffer_size(), 0.0F);
                fdmt_sync.execute(waterfall, dmt_sync2);

                std::vector<float> dmt_stepped2(plan.get_buffer_size(),
                                                -888.0F);
                fdmt_stepped.reset(waterfall, dmt_stepped2);
                fdmt_stepped.advance_until_remaining(0);
                fdmt_stepped.finalize();

                for (size_t i = 0; i < dmt_size; ++i) {
                    REQUIRE(dmt_stepped2[i] == dmt_sync2[i]);
                }
            }
        }
    }
}

TEST_CASE("FDMTCPU odd channels and padding safety", "[fdmt_cpu]") {
    const float f_min   = 1000.0F;
    const float f_max   = 1500.0F;
    const size_t nsamps = 256;
    const float tsamp   = 0.001F;
    const size_t dt_max = 32;
    const size_t dt_min = 0;

    // Test channel counts that produce odd divisions (triggering do_copy)
    const std::vector<size_t> odd_chans = {13, 63};

    for (const auto nchans : odd_chans) {
        DYNAMIC_SECTION("nchans = " << nchans) {
            FDMTCPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_min, 1,
                         true, "full");
            const auto& plan = fdmt.get_plan();

            std::vector<float> waterfall(nchans * nsamps);
            for (size_t i = 0; i < waterfall.size(); ++i) {
                waterfall[i] = static_cast<float>((i % 7) + 1);
            }

            std::vector<float> dmt_sync(plan.get_buffer_size(), 0.0F);
            fdmt.execute(waterfall, dmt_sync);

            // Pre-fill stepped buffer with dirty non-zero floats to catch
            // padding leaks
            std::vector<float> dmt_stepped(plan.get_buffer_size(), -42.0F);
            fdmt.reset(waterfall, dmt_stepped);
            while (!fdmt.is_finished()) {
                fdmt.advance(1);
            }
            fdmt.finalize();

            const auto dmt_size = plan.get_dmt_size();
            for (size_t i = 0; i < dmt_size; ++i) {
                REQUIRE(dmt_stepped[i] == dmt_sync[i]);
            }
        }
    }
}

TEST_CASE("FDMTCPU valid mode streaming history", "[fdmt_cpu]") {
    const float f_min   = 1000.0F;
    const float f_max   = 1500.0F;
    const size_t nchans = 32;
    const size_t nsamps = 256;
    const float tsamp   = 0.001F;
    const size_t dt_max = 32;
    const size_t dt_min = 0;

    FDMTCPU fdmt_valid(f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_min, 1,
                       true, "valid");
    const auto& plan = fdmt_valid.get_plan();

    std::vector<float> block1(nchans * nsamps, 1.0F);
    std::vector<float> block2(nchans * nsamps, 2.0F);

    std::vector<float> dmt1(plan.get_buffer_size(), 0.0F);
    std::vector<float> dmt2(plan.get_buffer_size(), 0.0F);

    // Block 1 (cold start, zero history)
    REQUIRE_NOTHROW(fdmt_valid.execute(block1, dmt1));

    // Block 2 (should incorporate history from block 1 seamlessly)
    REQUIRE_NOTHROW(fdmt_valid.execute(block2, dmt2));

    // Values in block 2 should be strictly greater than cold start
    // because block 1's history contributes to the edge samples
    float sum1 = 0.0F;
    float sum2 = 0.0F;
    for (size_t i = 0; i < plan.get_dmt_size(); ++i) {
        sum1 += dmt1[i];
        sum2 += dmt2[i];
    }
    CHECK(sum1 > 0.0F);
    CHECK(sum2 > sum1);
}

TEST_CASE("FDMTCPU sparse and dt_min execution equivalence", "[fdmt_cpu]") {
    const float f_min    = 1000.0F;
    const float f_max    = 1500.0F;
    const size_t nchans  = 64;
    const size_t nsamps  = 256;
    const float tsamp    = 0.001F;
    const size_t dt_max  = 64;
    const size_t dt_min  = 16;
    const size_t dt_step = 4;

    // Dense full FDMT: dt_min = 0, dt_step = 1
    FDMTCPU fdmt_dense(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, false,
                       "full");
    // Sparse FDMT: dt_min = 16, dt_step = 4
    FDMTCPU fdmt_sparse(f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_min,
                        dt_step, false, "full");

    std::vector<float> waterfall(nchans * nsamps);
    for (size_t i = 0; i < waterfall.size(); ++i) {
        waterfall[i] = static_cast<float>((i % 29) + 1);
    }

    std::vector<float> dmt_dense(fdmt_dense.get_plan().get_buffer_size(), 0.0F);
    fdmt_dense.execute(waterfall, dmt_dense);

    std::vector<float> dmt_sparse(fdmt_sparse.get_plan().get_buffer_size(),
                                  0.0F);
    fdmt_sparse.execute(waterfall, dmt_sparse);

    const auto& dense_plan  = fdmt_dense.get_plan();
    const auto& sparse_plan = fdmt_sparse.get_plan();

    const auto dense_nsamps  = dense_plan.get_dmt_nsamps();
    const auto sparse_nsamps = sparse_plan.get_dmt_nsamps();
    REQUIRE(dense_nsamps == sparse_nsamps);

    const auto& dense_dt_grid  = dense_plan.get_dt_grid_final();
    const auto& sparse_dt_grid = sparse_plan.get_dt_grid_final();

    for (size_t s_idx = 0; s_idx < sparse_dt_grid.size(); ++s_idx) {
        const auto target_dt = sparse_dt_grid[s_idx];
        auto it =
            std::find(dense_dt_grid.begin(), dense_dt_grid.end(), target_dt);
        REQUIRE(it != dense_dt_grid.end());
        const auto d_idx = std::distance(dense_dt_grid.begin(), it);

        const float* sparse_row = dmt_sparse.data() + s_idx * sparse_nsamps;
        const float* dense_row  = dmt_dense.data() + d_idx * dense_nsamps;

        for (size_t t = 0; t < sparse_nsamps; ++t) {
            REQUIRE(sparse_row[t] == dense_row[t]);
        }
    }
}

TEST_CASE("FDMTCPU arbitrary dt_grid execution equivalence", "[fdmt_cpu]") {
    const float f_min   = 1000.0F;
    const float f_max   = 1500.0F;
    const size_t nchans = 64;
    const size_t nsamps = 256;
    const float tsamp   = 0.001F;
    const size_t dt_max = 64;

    const std::vector<SizeType> custom_dts = {12, 23, 37, 49, 64};

    // Dense full FDMT: dt_min = 0, dt_step = 1, dt_max = 64
    FDMTCPU fdmt_dense(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, false,
                       "full");
    // Custom grid FDMT
    FDMTCPU fdmt_custom(f_min, f_max, nchans, nsamps, tsamp, custom_dts, false,
                        "full");

    std::vector<float> waterfall(nchans * nsamps);
    for (size_t i = 0; i < waterfall.size(); ++i) {
        waterfall[i] = static_cast<float>((i % 23) + 1);
    }

    std::vector<float> dmt_dense(fdmt_dense.get_plan().get_buffer_size(), 0.0F);
    fdmt_dense.execute(waterfall, dmt_dense);

    std::vector<float> dmt_custom(fdmt_custom.get_plan().get_buffer_size(),
                                  0.0F);
    fdmt_custom.execute(waterfall, dmt_custom);

    const auto& dense_plan  = fdmt_dense.get_plan();
    const auto& custom_plan = fdmt_custom.get_plan();

    const auto dense_nsamps  = dense_plan.get_dmt_nsamps();
    const auto custom_nsamps = custom_plan.get_dmt_nsamps();
    REQUIRE(dense_nsamps == custom_nsamps);

    const auto& dense_dt_grid = dense_plan.get_dt_grid_final();
    const auto& custom_grid   = custom_plan.get_dt_grid_final();

    for (size_t c_idx = 0; c_idx < custom_grid.size(); ++c_idx) {
        const auto target_dt = custom_grid[c_idx];
        auto it =
            std::find(dense_dt_grid.begin(), dense_dt_grid.end(), target_dt);
        REQUIRE(it != dense_dt_grid.end());
        const auto d_idx = std::distance(dense_dt_grid.begin(), it);

        const float* custom_row = dmt_custom.data() + c_idx * custom_nsamps;
        const float* dense_row  = dmt_dense.data() + d_idx * dense_nsamps;

        for (size_t t = 0; t < custom_nsamps; ++t) {
            REQUIRE(custom_row[t] == dense_row[t]);
        }
    }
}

TEST_CASE("FDMTCPU physical FRB detection test", "[fdmt_cpu]") {
    const float f_min          = 1000.0F;
    const float f_max          = 1500.0F;
    const SizeType nchans      = 64;
    const SizeType nsamps      = 512;
    const float tsamp          = 0.001F;
    const SizeType injected_dt = 40;
    const float pulse_toa      = 200.0F;

    // Generate pure dispersed FRB
    const auto [waterfall, nsamps_dispersed] = utils::generate_pure_frb(
        nchans, nsamps, f_min, f_max, injected_dt, pulse_toa, 10.0F);
    REQUIRE(nsamps_dispersed > 0);

    // Dedisperse using custom grid containing injected_dt and other off-target
    // trials
    const std::vector<SizeType> target_dts = {10, 25, 40, 55, 70};
    FDMTCPU fdmt(f_min, f_max, nchans, nsamps, tsamp, target_dts, false,
                 "full");

    std::vector<float> dmt(fdmt.get_plan().get_buffer_size(), 0.0F);
    fdmt.execute(waterfall, dmt);

    const auto dmt_nsamps = fdmt.get_plan().get_dmt_nsamps();

    // Find row index for injected_dt = 40
    size_t row_idx_40 = 0;
    for (size_t i = 0; i < target_dts.size(); ++i) {
        if (target_dts[i] == injected_dt) {
            row_idx_40 = i;
            break;
        }
    }

    const float* row_40 = dmt.data() + (row_idx_40 * dmt_nsamps);

    // Find peak sample in row 40
    float max_val    = 0.0F;
    size_t peak_samp = 0;
    for (size_t t = 0; t < dmt_nsamps; ++t) {
        if (row_40[t] > max_val) {
            max_val   = row_40[t];
            peak_samp = t;
        }
    }

    // The peak should align closely with pulse_toa
    const auto expected_toa_samp = static_cast<size_t>(pulse_toa);
    CHECK(peak_samp >= expected_toa_samp - 1);
    CHECK(peak_samp <= expected_toa_samp + 1);

    // Also verify that peak value at correct dt is significantly higher than
    // off-pulse delays
    for (size_t i = 0; i < target_dts.size(); ++i) {
        if (i == row_idx_40) {
            continue;
        }
        const float* off_row = dmt.data() + (i * dmt_nsamps);
        float off_max        = 0.0F;
        for (size_t t = 0; t < dmt_nsamps; ++t) {
            off_max = std::max(off_max, off_row[t]);
        }
        CHECK(max_val > off_max);
    }
}

} // namespace dmt
