#include <algorithm>
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

    const std::vector<IndexType> custom_dts = {12, 23, 37, 49, 64};

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
    const std::vector<IndexType> target_dts = {10, 25, 40, 55, 70};
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

TEST_CASE("FDMTCPU negative and symmetric dispersion", "[fdmt_cpu]") {
    const float f_min      = 1000.0F;
    const float f_max      = 1500.0F;
    const SizeType nchans  = 64;
    const SizeType nsamps  = 512;
    const float tsamp      = 0.001F;
    const IndexType dt_max = 32;
    const IndexType dt_min = -32;

    SECTION("Symmetric range execution and variance queries") {
        FDMTCPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_min, 1,
                     false, "full", false);

        CHECK(fdmt.get_plan().get_dt_min() == -32);
        CHECK(fdmt.get_plan().get_dt_max() == 32);
        CHECK(fdmt.get_effective_variance(0, 1) ==
              Catch::Approx(static_cast<float>(nchans)));

        const auto var_grid = fdmt.get_effective_variance_grid(2);
        REQUIRE(var_grid.size() == 65);
        for (const auto v : var_grid) {
            CHECK(v == Catch::Approx(static_cast<float>(nchans * 2)));
        }

        std::vector<float> waterfall(nchans * nsamps, 1.0F);
        std::vector<float> dmt(fdmt.get_plan().get_buffer_size(), 0.0F);
        REQUIRE_NOTHROW(fdmt.execute(waterfall, dmt));

        const auto dmt_nsamps = fdmt.get_plan().get_dmt_nsamps();
        const auto& dt_grid   = fdmt.get_plan().get_dt_grid_final();

        // Check symmetry on constant input:
        // For interior sample t = 200 (well within valid region [|dt_max|, nsamps - 1]),
        // both +dt and -dt should sum exactly nchans * 1.0F = 64.0F.
        for (size_t i = 0; i < dt_grid.size() / 2; ++i) {
            const size_t opp_i = dt_grid.size() - 1 - i;
            const float* row_neg = dmt.data() + i * dmt_nsamps;
            const float* row_pos = dmt.data() + opp_i * dmt_nsamps;
            CHECK(row_neg[200] == Catch::Approx(static_cast<float>(nchans)));
            CHECK(row_pos[200] == Catch::Approx(static_cast<float>(nchans)));
        }
    }

    SECTION("Negative DM pulse recovery") {
        FDMTCPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_min, 1,
                     false, "full", false);

        std::vector<float> waterfall(nchans * nsamps, 0.0F);
        const IndexType target_dt = -16;
        const size_t t0           = 100;

        const float df          = (f_max - f_min) / static_cast<float>(nchans);
        const double f_min_inv2 = 1.0 / (static_cast<double>(f_min) * f_min);
        const double f_max_inv2 = 1.0 / (static_cast<double>(f_max) * f_max);

        for (size_t c = 0; c < nchans; ++c) {
            const double fc = static_cast<double>(f_min) +
                              (static_cast<double>(c) + 0.5) * static_cast<double>(df);
            const double fc_inv2 = 1.0 / (fc * fc);
            const auto tau = static_cast<IndexType>(std::round(
                static_cast<double>(target_dt) * (fc_inv2 - f_max_inv2) /
                (f_min_inv2 - f_max_inv2)));
            const auto t = static_cast<IndexType>(t0) + tau;
            if (t >= 0 && static_cast<size_t>(t) < nsamps) {
                waterfall[c * nsamps + static_cast<size_t>(t)] += 10.0F;
            }
        }

        std::vector<float> dmt(fdmt.get_plan().get_buffer_size(), 0.0F);
        fdmt.execute(waterfall, dmt);

        const auto dmt_nsamps = fdmt.get_plan().get_dmt_nsamps();
        const auto& dt_grid   = fdmt.get_plan().get_dt_grid_final();

        // Find global peak across entire DMT
        float max_val          = 0.0F;
        IndexType peak_dt      = 0;
        size_t peak_samp       = 0;

        for (size_t d = 0; d < dt_grid.size(); ++d) {
            const float* row = dmt.data() + d * dmt_nsamps;
            for (size_t t = 0; t < dmt_nsamps; ++t) {
                if (row[t] > max_val) {
                    max_val   = row[t];
                    peak_dt   = dt_grid[d];
                    peak_samp = t;
                }
            }
        }

        CHECK(std::abs(peak_dt - target_dt) <= 1);
        CHECK(peak_samp >= t0 - 1);
        CHECK(peak_samp <= t0 + 1);
    }
}

TEST_CASE("FDMTCPU purely negative dt range", "[fdmt_cpu]") {
    // Regression test for a range with no positive/zero trials at all --
    // promised in the original implementation plan but never delivered.
    const float f_min      = 1000.0F;
    const float f_max      = 1500.0F;
    const SizeType nchans  = 64;
    const SizeType nsamps  = 256;
    const float tsamp      = 0.001F;
    const IndexType dt_max = -10;
    const IndexType dt_min = -50;

    // use_box_smearing=false so a constant-1 waterfall sums to exactly
    // nchans regardless of trial dt (with smearing on, each row's value
    // depends on that trial's per-channel smearing width, which varies
    // continuously with dt -- not the point of this test).
    FDMTCPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_min, 1, false);
    const auto& plan = fdmt.get_plan();
    CHECK(plan.get_dt_min() == -50);
    CHECK(plan.get_dt_max() == -10);
    const auto& dt_grid = plan.get_dt_grid_final();
    REQUIRE(dt_grid.size() == 41);
    CHECK(dt_grid.front() == -50);
    CHECK(dt_grid.back() == -10);
    for (const auto dt : dt_grid) {
        CHECK(dt <= 0);
    }

    std::vector<float> waterfall(nchans * nsamps, 1.0F);
    std::vector<float> dmt(plan.get_buffer_size(), 0.0F);
    REQUIRE_NOTHROW(fdmt.execute(waterfall, dmt));
    const auto dmt_nsamps = plan.get_dmt_nsamps();
    for (size_t d = 0; d < dt_grid.size(); ++d) {
        CHECK(dmt[(d * dmt_nsamps) + 200] == Catch::Approx(static_cast<float>(nchans)));
    }
}

TEST_CASE("FDMTCPU use_box_smearing=false with coarse channelization",
         "[fdmt_cpu]") {
    // Regression test: with few channels relative to dt_max, a single
    // channel's own bandwidth induces a level-0 dt range spanning more than
    // one row. An earlier refactor of the level-0 init dispatcher
    // accidentally copied the *unshifted* raw waterfall into every such row
    // when use_box_smearing=false, instead of shifting each row by its own
    // |dt| samples as fdmt_init_impl_row0/row do -- this test exercises
    // exactly that path (few channels, moderate dt_max) and would have
    // caught the regression via add_frb_track's exact-recovery check below.
    const float f_min      = 1000.0F;
    const float f_max      = 1500.0F;
    const SizeType nsamps  = 256;
    const float tsamp      = 0.001F;
    const IndexType dt_max = 16;

    for (const SizeType nchans : {2U, 3U, 4U, 5U, 7U}) {
        DYNAMIC_SECTION("nchans=" << nchans) {
            FDMTCPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, false);
            const auto& plan = fdmt.get_plan();
            const auto ndms  = plan.get_dmt_ndms();
            REQUIRE(ndms > 1);
            const SizeType target_idx = ndms / 2;
            const SizeType toffset    = 150;

            std::vector<float> waterfall(nchans * nsamps, 0.0F);
            algorithms::add_frb_track(waterfall, plan, target_idx, 1.0F,
                                      static_cast<IndexType>(toffset), 1);
            std::vector<float> dmt(plan.get_buffer_size(), 0.0F);
            fdmt.execute(waterfall, dmt);

            const auto dmt_nsamps = plan.get_dmt_nsamps();
            const auto value =
                dmt[(target_idx * dmt_nsamps) + toffset];
            CHECK(value == Catch::Approx(static_cast<float>(nchans)));
        }
    }
}

TEST_CASE("FDMTCPU add_frb_track / trace_dm exact recovery", "[fdmt_cpu]") {
    // For every injected channel, add_frb_track places a unit impulse at
    // exactly the sample trace_dm computes; running FDMT on that waterfall
    // must reproduce a value of exactly nchans (all channels combining
    // constructively, no partial/rounding loss) at (dm_idx, toffset).
    const float f_min     = 1000.0F;
    const float f_max     = 1500.0F;
    const SizeType nsamps = 512;
    const float tsamp     = 0.001F;
    const SizeType toffset = 150;

    struct Config {
        SizeType nchans;
        IndexType dt_max;
        IndexType dt_min;
        IndexType target_dt;
    };
    const std::vector<Config> configs = {
        {2, 16, 0, 1},      {8, 16, 0, 4},       {64, 32, 0, 10},
        {65, 32, 0, 10},    {64, 32, -32, -16},  {64, 32, -32, 16},
        {64, 32, -32, 0},   {64, 32, -32, -32},  {64, 32, -32, 32},
    };

    for (const auto& cfg : configs) {
        for (const bool use_box_smearing : {false, true}) {
            DYNAMIC_SECTION("nchans=" << cfg.nchans << " dt=" << cfg.target_dt
                                      << " smear=" << use_box_smearing) {
                FDMTCPU fdmt(f_min, f_max, cfg.nchans, nsamps, tsamp,
                            cfg.dt_max, cfg.dt_min, 1, use_box_smearing);
                const auto& plan    = fdmt.get_plan();
                const auto& dt_grid = plan.get_dt_grid_final();
                const auto it =
                    std::find(dt_grid.begin(), dt_grid.end(), cfg.target_dt);
                REQUIRE(it != dt_grid.end());
                const auto dm_idx =
                    static_cast<SizeType>(std::distance(dt_grid.begin(), it));

                std::vector<float> waterfall(cfg.nchans * nsamps, 0.0F);
                algorithms::add_frb_track(waterfall, plan, dm_idx, 1.0F,
                                          static_cast<IndexType>(toffset), 1);
                std::vector<float> dmt(plan.get_buffer_size(), 0.0F);
                fdmt.execute(waterfall, dmt);

                const auto dmt_nsamps = plan.get_dmt_nsamps();
                const auto value = dmt[(dm_idx * dmt_nsamps) + toffset];
                CHECK(value == Catch::Approx(static_cast<float>(cfg.nchans)));
            }
        }
    }
}

TEST_CASE("FDMTCPU valid-mode cross-block streaming",
          "[fdmt_cpu]") {
    // Seamless cross-block streaming in mode="valid": m_tree_history caches
    // trailing samples at each tree merge node, enabling bit-exact match
    // with monolithic continuous execution across consecutive blocks.
    const float f_min     = 1000.0F;
    const float f_max     = 1500.0F;
    const SizeType nchans = 32;
    const float tsamp     = 0.001F;

    struct Config {
        IndexType dt_max;
        IndexType dt_min;
    };
    const std::vector<Config> configs = {{32, 0}, {32, -32}, {16, -32}};

    for (const auto& cfg : configs) {
        DYNAMIC_SECTION("dt_max=" << cfg.dt_max << " dt_min=" << cfg.dt_min) {
            const SizeType block_size  = 128;
            const SizeType n_blocks    = 5;
            const SizeType total_nsamp = block_size * n_blocks;

            std::vector<float> waterfall(nchans * total_nsamp);
            for (size_t i = 0; i < waterfall.size(); ++i) {
                waterfall[i] = static_cast<float>((i % 23) + 1);
            }

            FDMTCPU fdmt_full(f_min, f_max, nchans, total_nsamp, tsamp,
                              cfg.dt_max, cfg.dt_min, 1, false, "full");
            std::vector<float> dmt_full(fdmt_full.get_plan().get_buffer_size(),
                                        0.0F);
            fdmt_full.execute(waterfall, dmt_full);
            const auto full_nsamps = fdmt_full.get_plan().get_dmt_nsamps();

            FDMTCPU fdmt_valid(f_min, f_max, nchans, block_size, tsamp,
                               cfg.dt_max, cfg.dt_min, 1, false, "valid");
            const auto& plan_valid = fdmt_valid.get_plan();
            const auto& dt_grid    = plan_valid.get_dt_grid_final();
            const auto ndms        = plan_valid.get_dmt_ndms();
            REQUIRE(ndms == fdmt_full.get_plan().get_dmt_ndms());
            std::vector<float> dmt_streamed(ndms * total_nsamp, 0.0F);

            for (SizeType b = 0; b < n_blocks; ++b) {
                std::vector<float> block(nchans * block_size);
                for (SizeType c = 0; c < nchans; ++c) {
                    std::copy_n(waterfall.data() + (c * total_nsamp) +
                                    (b * block_size),
                                block_size, block.data() + (c * block_size));
                }
                std::vector<float> dmt_block(plan_valid.get_buffer_size(),
                                             0.0F);
                fdmt_valid.execute(block, dmt_block);
                for (SizeType d = 0; d < ndms; ++d) {
                    std::copy_n(dmt_block.data() + (d * block_size),
                                block_size,
                                dmt_streamed.data() + (d * total_nsamp) +
                                    (b * block_size));
                }
            }

            const auto abs_dt_max = static_cast<SizeType>(
                std::max(std::abs(cfg.dt_min), std::abs(cfg.dt_max)));
            for (SizeType d = 0; d < ndms; ++d) {
                for (SizeType t = abs_dt_max; t < total_nsamp; ++t) {
                    const auto streamed_val = dmt_streamed[(d * total_nsamp) + t];
                    const auto full_val     = dmt_full[(d * full_nsamps) + t];
                    CHECK(streamed_val == Catch::Approx(full_val));
                }
            }

            // Test reset_history()
            fdmt_valid.reset_history();
        }
    }
}

namespace {
// Streams `waterfall` (nchans x block_size*n_blocks) through `fdmt` in
// consecutive non-overlapping blocks, writing results into `streamed`
// (ndms x block_size*n_blocks).
void stream_blocks(FDMTCPU& fdmt, std::span<const float> waterfall,
                   SizeType nchans, SizeType block_size, SizeType n_blocks,
                   std::vector<float>& streamed) {
    const auto& plan       = fdmt.get_plan();
    const auto ndms        = plan.get_dmt_ndms();
    const auto total_nsamp = block_size * n_blocks;
    streamed.assign(ndms * total_nsamp, 0.0F);
    std::vector<float> block(nchans * block_size);
    std::vector<float> dmt_block(plan.get_buffer_size(), 0.0F);
    for (SizeType b = 0; b < n_blocks; ++b) {
        for (SizeType c = 0; c < nchans; ++c) {
            std::copy_n(waterfall.data() + (c * total_nsamp) + (b * block_size),
                       block_size, block.data() + (c * block_size));
        }
        fdmt.execute(block, dmt_block);
        for (SizeType d = 0; d < ndms; ++d) {
            std::copy_n(dmt_block.data() + (d * block_size), block_size,
                       streamed.data() + (d * total_nsamp) + (b * block_size));
        }
    }
}
} // namespace

TEST_CASE("FDMTCPU valid-mode streaming stress: smearing, odd channels, "
         "long chains",
         "[fdmt_cpu]") {
    // Beyond the base streaming test: box smearing combined with tree
    // history, odd nchans (exercises the do_copy passthrough path with no
    // history needed), and a chain long enough (20 blocks) to rule out any
    // slow drift/accumulation bug.
    const float f_min      = 1000.0F;
    const float f_max      = 1500.0F;
    const float tsamp      = 0.001F;
    const IndexType dt_max = 32;
    const IndexType dt_min = -32;

    struct Config {
        SizeType nchans;
        SizeType block_size;
        SizeType n_blocks;
        bool use_box_smearing;
    };
    const std::vector<Config> configs = {
        {32, 128, 5, true},   // box smearing + tree history combined
        {13, 128, 8, true},   // odd nchans
        {63, 100, 10, true},  // odd nchans, non-round block size
        {32, 128, 20, true},  // long chain
        {32, 40, 15, true},   // many small blocks
    };

    for (const auto& cfg : configs) {
        DYNAMIC_SECTION("nchans=" << cfg.nchans << " block_size="
                                  << cfg.block_size
                                  << " n_blocks=" << cfg.n_blocks) {
            const auto total_nsamp = cfg.block_size * cfg.n_blocks;
            std::vector<float> waterfall(cfg.nchans * total_nsamp);
            for (size_t i = 0; i < waterfall.size(); ++i) {
                waterfall[i] = static_cast<float>((i % 23) + 1);
            }

            FDMTCPU fdmt_full(f_min, f_max, cfg.nchans, total_nsamp, tsamp,
                              dt_max, dt_min, 1, cfg.use_box_smearing, "full");
            std::vector<float> dmt_full(fdmt_full.get_plan().get_buffer_size(),
                                        0.0F);
            fdmt_full.execute(waterfall, dmt_full);
            const auto full_nsamps = fdmt_full.get_plan().get_dmt_nsamps();

            FDMTCPU fdmt_valid(f_min, f_max, cfg.nchans, cfg.block_size, tsamp,
                               dt_max, dt_min, 1, cfg.use_box_smearing, "valid");
            std::vector<float> streamed;
            stream_blocks(fdmt_valid, waterfall, cfg.nchans, cfg.block_size,
                         cfg.n_blocks, streamed);

            const auto ndms       = fdmt_valid.get_plan().get_dmt_ndms();
            const auto abs_dt_max = static_cast<SizeType>(
                std::max(std::abs(dt_min), std::abs(dt_max)));
            for (SizeType d = 0; d < ndms; ++d) {
                for (SizeType t = abs_dt_max; t < total_nsamp; ++t) {
                    CHECK(streamed[(d * total_nsamp) + t] ==
                         Catch::Approx(dmt_full[(d * full_nsamps) + t])
                             .margin(1e-3));
                }
            }
        }
    }
}

TEST_CASE("FDMTCPU reset_history() actually resets streaming state",
         "[fdmt_cpu]") {
    // reset_history() must make the *next* block behave exactly like a
    // brand-new instance's first block, not merely be callable.
    const float f_min      = 1000.0F;
    const float f_max      = 1500.0F;
    const SizeType nchans  = 32;
    const SizeType nsamps  = 128;
    const float tsamp      = 0.001F;
    const IndexType dt_max = 32;
    const IndexType dt_min = -32;

    std::vector<float> block1(nchans * nsamps);
    std::vector<float> block2(nchans * nsamps);
    for (size_t i = 0; i < block1.size(); ++i) {
        block1[i] = static_cast<float>((i % 17) + 1);
        block2[i] = static_cast<float>((i % 13) + 1);
    }

    FDMTCPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_min, 1, true,
                "valid");
    std::vector<float> dmt1(fdmt.get_plan().get_buffer_size(), 0.0F);
    std::vector<float> dmt2_with_history(fdmt.get_plan().get_buffer_size(),
                                         0.0F);
    fdmt.execute(block1, dmt1);
    fdmt.execute(block2, dmt2_with_history);

    fdmt.reset_history();
    std::vector<float> dmt2_after_reset(fdmt.get_plan().get_buffer_size(),
                                        0.0F);
    fdmt.execute(block2, dmt2_after_reset);

    FDMTCPU fdmt_fresh(f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_min, 1,
                       true, "valid");
    std::vector<float> dmt2_fresh(fdmt_fresh.get_plan().get_buffer_size(),
                                  0.0F);
    fdmt_fresh.execute(block2, dmt2_fresh);

    const auto dmt_size = fdmt.get_plan().get_dmt_size();
    bool differs_before_reset = false;
    for (size_t i = 0; i < dmt_size; ++i) {
        CHECK(dmt2_after_reset[i] == Catch::Approx(dmt2_fresh[i]));
        if (dmt2_with_history[i] != dmt2_after_reset[i]) {
            differs_before_reset = true;
        }
    }
    // Sanity check that history was actually doing something (otherwise
    // this test wouldn't distinguish a working reset from a no-op history).
    CHECK(differs_before_reset);
}

TEST_CASE("FDMTCPU add_frb_track recovery across a valid-mode block boundary",
         "[fdmt_cpu]") {
    // The end-to-end scenario the streaming fix exists for: a real dispersed
    // pulse straddling a block boundary must still recover at full
    // amplitude, for both signs of dt and at the |dt| extremes.
    const float f_min      = 1000.0F;
    const float f_max      = 1500.0F;
    const SizeType nchans  = 64;
    const SizeType block_size = 128;
    const SizeType n_blocks   = 4;
    const auto total_nsamp    = block_size * n_blocks;
    const float tsamp       = 0.001F;
    const IndexType dt_max  = 32;
    const IndexType dt_min  = -32;

    plans::FDMTPlan plan(f_min, f_max, nchans, total_nsamp, tsamp, dt_max,
                        dt_min);
    const auto& dt_grid = plan.get_dt_grid_final();

    const std::vector<IndexType> target_dts = {16, -16, 0, 32, -32};
    const std::vector<SizeType> toffsets    = {
        block_size - 5, block_size, block_size + 5, (2 * block_size) - 3};

    for (const auto target_dt : target_dts) {
        for (const auto toffset : toffsets) {
            DYNAMIC_SECTION("dt=" << target_dt << " toffset=" << toffset) {
                const auto it =
                    std::find(dt_grid.begin(), dt_grid.end(), target_dt);
                REQUIRE(it != dt_grid.end());
                const auto dm_idx = static_cast<SizeType>(
                    std::distance(dt_grid.begin(), it));

                std::vector<float> waterfall(nchans * total_nsamp, 0.0F);
                algorithms::add_frb_track(waterfall, plan, dm_idx, 1.0F,
                                          static_cast<IndexType>(toffset), 1);

                FDMTCPU fdmt_valid(f_min, f_max, nchans, block_size, tsamp,
                                  dt_max, dt_min, 1, false, "valid");
                std::vector<float> streamed;
                stream_blocks(fdmt_valid, waterfall, nchans, block_size,
                             n_blocks, streamed);

                const auto value = streamed[(dm_idx * total_nsamp) + toffset];
                CHECK(value == Catch::Approx(static_cast<float>(nchans)));
            }
        }
    }
}

TEST_CASE("FDMTCPU stepper partial-advance across streamed blocks",
         "[fdmt_cpu]") {
    // The "stop 1-2 levels before root to inspect sub-bands" usage pattern
    // must not break cross-block tree history, as long as finalize() always
    // completes the block before the next reset().
    const float f_min      = 1000.0F;
    const float f_max      = 1500.0F;
    const SizeType nchans  = 32;
    const SizeType block_size = 128;
    const SizeType n_blocks   = 6;
    const auto total_nsamp    = block_size * n_blocks;
    const float tsamp       = 0.001F;
    const IndexType dt_max  = 32;
    const IndexType dt_min  = -32;

    std::vector<float> waterfall(nchans * total_nsamp);
    for (size_t i = 0; i < waterfall.size(); ++i) {
        waterfall[i] = static_cast<float>((i % 23) + 1);
    }

    FDMTCPU fdmt_full(f_min, f_max, nchans, total_nsamp, tsamp, dt_max, dt_min,
                      1, true, "full");
    std::vector<float> dmt_full(fdmt_full.get_plan().get_buffer_size(), 0.0F);
    fdmt_full.execute(waterfall, dmt_full);
    const auto full_nsamps = fdmt_full.get_plan().get_dmt_nsamps();

    FDMTCPU fdmt_valid(f_min, f_max, nchans, block_size, tsamp, dt_max, dt_min,
                      1, true, "valid");
    const auto& plan_valid = fdmt_valid.get_plan();
    const auto ndms        = plan_valid.get_dmt_ndms();
    std::vector<float> streamed(ndms * total_nsamp, 0.0F);

    for (SizeType b = 0; b < n_blocks; ++b) {
        std::vector<float> block(nchans * block_size);
        for (SizeType c = 0; c < nchans; ++c) {
            std::copy_n(waterfall.data() + (c * total_nsamp) +
                            (b * block_size),
                       block_size, block.data() + (c * block_size));
        }
        std::vector<float> dmt_buf(plan_valid.get_buffer_size(), 0.0F);
        fdmt_valid.reset(block, dmt_buf);
        // Stop 2 levels before root (4 sub-bands) to simulate per-subband
        // inspection, then complete the block via finalize().
        fdmt_valid.advance_until_remaining(2);
        REQUIRE(fdmt_valid.num_subbands() == 4);
        for (SizeType s = 0; s < 4; ++s) {
            const auto sub = fdmt_valid.view_subband(s);
            CHECK(sub.data.size() == sub.ndt * sub.nsamps);
        }
        fdmt_valid.finalize();

        for (SizeType d = 0; d < ndms; ++d) {
            std::copy_n(dmt_buf.data() + (d * block_size), block_size,
                       streamed.data() + (d * total_nsamp) + (b * block_size));
        }
    }

    const auto abs_dt_max = static_cast<SizeType>(
        std::max(std::abs(dt_min), std::abs(dt_max)));
    for (SizeType d = 0; d < ndms; ++d) {
        for (SizeType t = abs_dt_max; t < total_nsamp; ++t) {
            CHECK(streamed[(d * total_nsamp) + t] ==
                 Catch::Approx(dmt_full[(d * full_nsamps) + t]).margin(1e-3));
        }
    }
}

} // namespace dmt
