#include <catch2/catch_test_macros.hpp>

#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"

namespace dmt {

using plans::CohFDMTPlan;
using plans::FDMTPlan;

TEST_CASE("FDMTPlan basic", "[dmt_plans]") {
    const float f_min      = 1000.0F;
    const float f_max      = 1500.0F;
    const SizeType nchans  = 500;
    const SizeType nsamps  = 1024;
    const float tsamp      = 0.001F;
    const SizeType dt_max  = 512;
    const SizeType dt_step = 1;
    const SizeType dt_min  = 0;

    FDMTPlan plan(f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_min);

    SECTION("Constructor and getter methods") {
        CHECK(plan.get_f_min() == f_min);
        CHECK(plan.get_f_max() == f_max);
        CHECK(plan.get_nchans() == nchans);
        CHECK(plan.get_nsamps() == nsamps);
        CHECK(plan.get_tsamp() == tsamp);
        CHECK(plan.get_dt_max() == dt_max);
        CHECK(plan.get_dt_min() == dt_min);
    }

    SECTION("Container properties") {
        const auto& container = plan.get_container();
        CHECK(container.df_top.size() == 10);
        CHECK(container.df_bot.size() == 10);
        CHECK(container.dt_grid_sub_top.size() == 10);
        CHECK(container.state_shape.size() == 10);
    }
    SECTION("Grid and size calculations") {
        const auto ndms_expected = static_cast<SizeType>(dt_max - dt_min + 1);
        CHECK(plan.get_df() == 1.0F);
        CHECK(plan.get_niters() == 9);
        CHECK(plan.get_dt_grid_final().size() == ndms_expected);
        CHECK(plan.get_dm_grid_final().size() == ndms_expected);
        CHECK(plan.get_dmt_size() ==
              static_cast<SizeType>(ndms_expected * (nsamps + dt_max)));
    }

    SECTION("Smearing grid") {
        const auto smearing_grid = plan.get_smearing_grid_final();
        const auto ndms          = plan.get_dmt_ndms();
        CHECK(smearing_grid.size() == ndms * nchans);
        for (const auto& val : smearing_grid) {
            CHECK(val >= 0.0F);
        }
    }

    SECTION("Memory usage and buffer size") {
        CHECK(plan.get_buffer_size() > 0);
        CHECK(plan.get_container().get_memory_usage() > 0);
        CHECK(plan.get_container().get_buffer_size() == plan.get_buffer_size());
    }

    SECTION("Internal calculations") {
        const auto& container = plan.get_container();

        // Check the values of df_top and df_bot
        for (SizeType i = 0; i < container.df_top.size() - 1; ++i) {
            CHECK(container.df_top[i] <= container.df_top[i + 1]);
            CHECK(container.df_bot[i] < container.df_bot[i + 1]);
            CHECK(container.df_top[i] <= container.df_bot[i]);
        }

        // Check the values of state_shape
        for (SizeType i = 0; i < container.state_shape.size() - 1; ++i) {
            CHECK(container.state_shape[i].nchans >
                  container.state_shape[i + 1].nchans);
            CHECK(container.state_shape[i].nsamps <=
                  container.state_shape[i + 1].nsamps);
            CHECK(container.state_shape[i].nelements > 0);
            CHECK(container.state_shape[i].dt_max <=
                  container.state_shape[i + 1].dt_max);
        }

        // Check dt_grid_final
        const auto& dt_grid_final = plan.get_dt_grid_final();
        CHECK(dt_grid_final.front() == plan.get_dt_min());
        CHECK(dt_grid_final.back() == plan.get_dt_max());
        for (SizeType i = 0; i < dt_grid_final.size() - 1; ++i) {
            CHECK(dt_grid_final[i] < dt_grid_final[i + 1]);
        }
    }
}

TEST_CASE("FDMTPlan edge cases", "[dmt_plans]") {
    SECTION("Valid construction") {
        CHECK_NOTHROW(
            FDMTPlan(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 0, 1, "full"));
    }

    SECTION("Invalid parameters") {
        // f_min > f_max
        CHECK_THROWS_AS(
            FDMTPlan(1500.0F, 1000.0F, 500, 1024, 0.001F, 512, 0, 1, "full"),
            std::invalid_argument);
        // nchans < 2
        CHECK_THROWS_AS(
            FDMTPlan(1000.0F, 1500.0F, 0, 1024, 0.001F, 512, 0, 1, "full"),
            std::invalid_argument);
        CHECK_THROWS_AS(
            FDMTPlan(1000.0F, 1500.0F, 1, 1024, 0.001F, 512, 0, 1, "full"),
            std::invalid_argument);
        // nsamps == 0
        CHECK_THROWS_AS(
            FDMTPlan(1000.0F, 1500.0F, 500, 0, 0.001F, 512, 0, 1, "full"),
            std::invalid_argument);
        // tsamp <= 0
        CHECK_THROWS_AS(
            FDMTPlan(1000.0F, 1500.0F, 500, 1024, 0.0F, 512, 0, 1, "full"),
            std::invalid_argument);
        // dt_max == 0
        CHECK_THROWS_AS(
            FDMTPlan(1000.0F, 1500.0F, 500, 1024, 0.001F, 0, 0, 1, "full"),
            std::invalid_argument);
        // dt_min >= dt_max
        CHECK_THROWS_AS(
            FDMTPlan(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 512, 1, "full"),
            std::invalid_argument);
        // dt_step == 0
        CHECK_THROWS_AS(
            FDMTPlan(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 0, 0, "full"),
            std::invalid_argument);
        // dt_step > dt_max - dt_min
        CHECK_THROWS_AS(
            FDMTPlan(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 0, 600, "full"),
            std::invalid_argument);
        // invalid mode
        CHECK_THROWS_AS(
            FDMTPlan(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 0, 1, "invalid"),
            std::invalid_argument);
    }
}

TEST_CASE("CohFDMTPlan basic", "[dmt_plans]") {
    const SizeType chan_per_sub = 16;

    const float f_center = 895.21484375F;
    const float bw_sub   = 2.9296875F;
    const SizeType nsub  = 64;
    const float tbin     = 3.41333333333333E-07;
    const SizeType nbin  = 1 << 16;
    const SizeType nfft  = 3;
    const float t_p      = tbin * chan_per_sub;
    const float dm_max   = 30.0F;
    const float dm_min   = 0.0F;

    CohFDMTPlan plan(f_center, bw_sub, nsub, tbin, nbin, nfft, t_p, dm_max,
                     dm_min);
    SECTION("Constructor and getter methods") {
        CHECK(plan.get_f_center() == f_center);
        CHECK(plan.get_bw_sub() == bw_sub);
        CHECK(plan.get_nsub() == nsub);
        CHECK(plan.get_tbin() == tbin);
        CHECK(plan.get_nbin() == nbin);
        CHECK(plan.get_nfft() == nfft);
        CHECK(plan.get_t_p() == t_p);
        CHECK(plan.get_dm_max() == dm_max);
        CHECK(plan.get_dm_min() == dm_min);
        CHECK(plan.get_noverlap() == 8192);
        CHECK(plan.get_data_order() == "PRITF");

        CHECK(plan.get_bw() == nsub * bw_sub);
        CHECK(plan.get_f_min() == f_center - (0.5F * nsub * bw_sub));
        CHECK(plan.get_f_max() == f_center + (0.5F * nsub * bw_sub));
        CHECK(plan.get_n_p() == chan_per_sub);
        CHECK(plan.get_nchan() == chan_per_sub);
        CHECK(plan.get_nsamp() ==
              nfft * (nbin - static_cast<SizeType>(2 * 8192)));
        CHECK(plan.get_mbin() == nbin / chan_per_sub);
        CHECK(plan.get_mchan() == nsub * chan_per_sub);
        CHECK(plan.get_msamp() == plan.get_nsamp() / chan_per_sub);
        CHECK(plan.get_tsamp() == tbin * static_cast<float>(chan_per_sub));
        CHECK(plan.get_dt_max() == chan_per_sub - 1);
    }
}

TEST_CASE("FDMTPlan dt_min and dt_step top-down pruning", "[dmt_plans]") {
    const float f_min     = 1000.0F;
    const float f_max     = 1500.0F;
    const SizeType nchans = 64;
    const SizeType nsamps = 1024;
    const float tsamp     = 0.001F;

    SECTION("Sparse final grid with dt_min and dt_step") {
        const SizeType dt_min  = 20;
        const SizeType dt_max  = 100;
        const SizeType dt_step = 5;

        FDMTPlan plan(f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_min,
                      dt_step);
        CHECK(plan.get_dt_min() == dt_min);
        CHECK(plan.get_dt_max() == dt_max);
        CHECK(plan.get_dt_step() == dt_step);

        const auto& dt_grid          = plan.get_dt_grid_final();
        const SizeType expected_ndms = (dt_max - dt_min) / dt_step + 1;
        CHECK(dt_grid.size() == expected_ndms);
        CHECK(dt_grid.front() == dt_min);
        for (SizeType i = 0; i < dt_grid.size(); ++i) {
            CHECK(dt_grid[i] == dt_min + i * dt_step);
        }
    }

    SECTION("Equivalence of pruned tree coordinates with dense tree") {
        const SizeType dt_max  = 100;
        const SizeType dt_min  = 20;
        const SizeType dt_step = 4;

        FDMTPlan dense_plan(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1);
        FDMTPlan sparse_plan(f_min, f_max, nchans, nsamps, tsamp, dt_max,
                             dt_min, dt_step);

        const auto niters = dense_plan.get_niters();
        REQUIRE(sparse_plan.get_niters() == niters);

        const auto& dense_coords =
            dense_plan.get_container().coordinates_sum[niters];
        const auto& sparse_coords =
            sparse_plan.get_container().coordinates_sum[niters];

        const auto& dense_final_grid  = dense_plan.get_dt_grid_final();
        const auto& sparse_final_grid = sparse_plan.get_dt_grid_final();

        CHECK(sparse_coords.size() == sparse_final_grid.size());

        // For each dt in the sparse plan root level, find the corresponding
        // coord in dense plan
        for (SizeType s_idx = 0; s_idx < sparse_final_grid.size(); ++s_idx) {
            const auto target_dt = sparse_final_grid[s_idx];
            const auto& s_coord  = sparse_coords[s_idx];

            // Find in dense grid
            auto it = std::find(dense_final_grid.begin(),
                                dense_final_grid.end(), target_dt);
            REQUIRE(it != dense_final_grid.end());
            const auto d_idx    = std::distance(dense_final_grid.begin(), it);
            const auto& d_coord = dense_coords[d_idx];

            // Verify the delay shift matches exactly
            CHECK(s_coord.delay == d_coord.delay);

            // Verify child coordinate indices map to the exact same delay
            // values in level niters - 1
            const auto& dense_coords_prev =
                dense_plan.get_container().coordinates[niters - 1];
            const auto& sparse_coords_prev =
                sparse_plan.get_container().coordinates[niters - 1];

            const auto& dense_prev_grid_tail =
                dense_plan.get_container().grids[niters - 1][0].dt_grid;
            const auto& dense_prev_grid_head =
                dense_plan.get_container().grids[niters - 1][1].dt_grid;

            const auto& sparse_prev_grid_tail =
                sparse_plan.get_container().grids[niters - 1][0].dt_grid;
            const auto& sparse_prev_grid_head =
                sparse_plan.get_container().grids[niters - 1][1].dt_grid;

            const auto s_dt_tail =
                sparse_prev_grid_tail[sparse_coords_prev[s_coord.i_coord_tail]
                                          .i_dt];
            const auto d_dt_tail =
                dense_prev_grid_tail[dense_coords_prev[d_coord.i_coord_tail]
                                         .i_dt];
            CHECK(s_dt_tail == d_dt_tail);

            const auto s_dt_head =
                sparse_prev_grid_head[sparse_coords_prev[s_coord.i_coord_head]
                                          .i_dt];
            const auto d_dt_head =
                dense_prev_grid_head[dense_coords_prev[d_coord.i_coord_head]
                                         .i_dt];
            CHECK(s_dt_head == d_dt_head);
        }
    }
}

TEST_CASE("FDMTPlan complexity calculations", "[dmt_plans]") {
    const float f_min      = 1000.0F;
    const float f_max      = 1500.0F;
    const SizeType nchans  = 128;
    const SizeType nsamps  = 1024;
    const float tsamp      = 0.001F;
    const SizeType dt_max  = 256;
    const SizeType dt_min  = 64;
    const SizeType dt_step = 2;

    FDMTPlan plan(f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_min, dt_step);
    const auto comp = plan.get_complexity();

    const SizeType expected_ndt = (dt_max - dt_min) / dt_step + 1;
    CHECK(comp.n_dt == expected_ndt);
    CHECK(comp.n_chans == nchans);
    CHECK(comp.brute_force_ops == expected_ndt * nchans);
    CHECK(comp.total_tree_nodes > 0);
    CHECK(comp.sum_additions > 0);
    CHECK(comp.total_tree_nodes >= comp.sum_additions);
    CHECK(comp.ops_ratio >
          1.0); // Speedup: FDMT requires fewer additions than brute-force

    const auto summary = comp.to_string();
    CHECK(summary.find("FDMT Complexity:") != std::string::npos);
    CHECK(summary.find("Brute-force ops") != std::string::npos);
    CHECK(summary.find("Theoretical Speedup Factor") != std::string::npos);
}

TEST_CASE("FDMTPlan arbitrary dt_grid and dm_grid", "[dmt_plans]") {
    const float f_min     = 1000.0F;
    const float f_max     = 1500.0F;
    const SizeType nchans = 64;
    const SizeType nsamps = 1024;
    const float tsamp     = 0.001F;

    SECTION("Arbitrary sorted dt_grid") {
        const std::vector<SizeType> custom_dts = {25, 50, 75, 100, 130, 180};
        FDMTPlan plan(f_min, f_max, nchans, nsamps, tsamp, custom_dts);

        CHECK(plan.is_custom_grid());
        CHECK(plan.get_dt_min() == 25);
        CHECK(plan.get_dt_max() == 180);
        CHECK(plan.get_dt_grid_final() == custom_dts);
        CHECK(plan.get_dmt_ndms() == custom_dts.size());

        // Target grid at root iteration
        const auto niters     = plan.get_niters();
        const auto& root_grid = plan.get_container().grids[niters][0].dt_grid;
        CHECK(root_grid == custom_dts);
    }

    SECTION(
        "Level 0 starts from min_dt_chan > 0 when child delays require it") {
        // High delay range where child delays do not include 0
        const std::vector<SizeType> high_dts = {200, 250, 300};
        FDMTPlan plan(f_min, f_max, nchans, nsamps, tsamp, high_dts);

        const auto& l0_grids        = plan.get_container().grids[0];
        bool found_nonzero_l0_start = false;
        for (SizeType i = 0; i < nchans; ++i) {
            if (!l0_grids[i].dt_grid.empty() &&
                l0_grids[i].dt_grid.front() > 0) {
                found_nonzero_l0_start = true;
                break;
            }
        }
        CHECK(found_nonzero_l0_start);
    }

    SECTION("Arbitrary dm_grid conversion and deduplication") {
        const std::vector<float> dms = {10.0F, 25.5F, 50.0F};
        FDMTPlan plan(f_min, f_max, nchans, nsamps, tsamp, dms);

        CHECK(plan.is_custom_grid());
        CHECK(!plan.get_dt_grid_final().empty());
        CHECK(!plan.get_dm_grid_final().empty());

        // Check monotonicity of generated dt_grid
        const auto dt_final = plan.get_dt_grid_final();
        for (size_t i = 1; i < dt_final.size(); ++i) {
            CHECK(dt_final[i] > dt_final[i - 1]);
        }
    }

    SECTION("Validation errors for invalid custom grids") {
        // Empty dt_grid
        CHECK_THROWS_AS(FDMTPlan(f_min, f_max, nchans, nsamps, tsamp,
                                 std::vector<SizeType>{}),
                        std::invalid_argument);

        // All-zero dt_grid
        CHECK_THROWS_AS(FDMTPlan(f_min, f_max, nchans, nsamps, tsamp,
                                 std::vector<SizeType>{0}),
                        std::invalid_argument);

        // Empty dm_grid
        CHECK_THROWS_AS(
            FDMTPlan(f_min, f_max, nchans, nsamps, tsamp, std::vector<float>{}),
            std::invalid_argument);

        // Negative DM
        CHECK_THROWS_AS(FDMTPlan(f_min, f_max, nchans, nsamps, tsamp,
                                 std::vector<float>{-5.0F, 10.0F}),
                        std::invalid_argument);
    }

    SECTION("Unsorted and duplicate dt_grid is auto-sorted and deduplicated") {
        FDMTPlan plan(f_min, f_max, nchans, nsamps, tsamp,
                      std::vector<SizeType>{50, 20, 100, 20});
        CHECK(plan.get_dt_grid_final() == std::vector<SizeType>{20, 50, 100});
        CHECK(plan.get_dt_min() == 20);
        CHECK(plan.get_dt_max() == 100);
    }

    SECTION("Unsorted and duplicate dm_grid is auto-sorted and deduplicated") {
        FDMTPlan plan(f_min, f_max, nchans, nsamps, tsamp,
                      std::vector<float>{50.0F, 20.0F, 100.0F, 20.0F});
        const auto dt_final = plan.get_dt_grid_final();
        for (size_t i = 1; i < dt_final.size(); ++i) {
            CHECK(dt_final[i] > dt_final[i - 1]);
        }
    }

    SECTION("High sparsity target grid note in complexity summary") {
        // Very sparse grid (e.g., 2 delays) where brute-force ops < 2 * FDMT
        // tree ops
        const std::vector<SizeType> sparse_dts = {50, 100};
        FDMTPlan plan(f_min, f_max, nchans, nsamps, tsamp, sparse_dts);
        const auto comp = plan.get_complexity();
        if (comp.ops_ratio < 2.0F) {
            const auto summary = comp.to_string();
            CHECK(summary.find("Note: Speedup factor is < 2.0x") !=
                  std::string::npos);
        }
    }
}

} // namespace dmt