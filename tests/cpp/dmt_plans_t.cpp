#include <catch2/catch_test_macros.hpp>

#include <spdlog/spdlog.h>

#include <cstddef>
#include <dmt/common/plans.hpp>

TEST_CASE("FDMTPlan basic", "[dmt_plans]") {
    const float f_min    = 1000.0F;
    const float f_max    = 1500.0F;
    const size_t nchans  = 500;
    const size_t nsamps  = 1024;
    const float tsamp    = 0.001F;
    const size_t dt_max  = 512;
    const size_t dt_step = 1;
    const size_t dt_min  = 0;

    FDMTPlan plan(f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_step, dt_min);

    SECTION("Constructor and getter methods") {
        CHECK(plan.get_f_min() == f_min);
        CHECK(plan.get_f_max() == f_max);
        CHECK(plan.get_nchans() == nchans);
        CHECK(plan.get_nsamps() == nsamps);
        CHECK(plan.get_tsamp() == tsamp);
        CHECK(plan.get_dt_max() == dt_max);
        CHECK(plan.get_dt_step() == dt_step);
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
        const auto ndms_expected = static_cast<SizeType>(
            std::floor((dt_max - dt_min) / static_cast<float>(dt_step)) + 1);
        CHECK(plan.get_df() == 1.0F);
        CHECK(plan.get_correction() == 0.5F);
        CHECK(plan.get_niters() == 9);
        CHECK(plan.get_dt_grid_final().size() == ndms_expected);
        CHECK(plan.get_dm_grid_final().size() == ndms_expected);
        CHECK(plan.get_dmt_size() ==
              static_cast<SizeType>(ndms_expected * (nsamps + dt_max)));
    }

    SECTION("Memory usage and buffer size") {
        CHECK(plan.get_buffer_size() > 0);
        CHECK(plan.get_container().get_memory_usage() > 0);
        CHECK(plan.get_container().get_buffer_size() == plan.get_buffer_size());
    }

    SECTION("Internal calculations") {
        const auto& container = plan.get_container();

        // Check the values of df_top and df_bot
        for (size_t i = 0; i < container.df_top.size() - 1; ++i) {
            CHECK(container.df_bot[i] < container.df_bot[i + 1]);
            CHECK(container.df_top[i] <= container.df_bot[i]);
        }

        // Check the values of state_shape
        for (size_t i = 0; i < container.state_shape.size() - 1; ++i) {
            CHECK(container.state_shape[i].nchans >=
                  container.state_shape[i + 1].nchans);
            CHECK(container.state_shape[i].ndt_max <=
                  container.state_shape[i + 1].ndt_max);
            CHECK(container.state_shape[i].ndt_min <=
                  container.state_shape[i + 1].ndt_min);
            CHECK(container.state_shape[i].nsamps <=
                  container.state_shape[i + 1].nsamps);
        }

        // Check the values of dt_grid_final
        const auto& dt_grid_final = plan.get_dt_grid_final();
        CHECK(dt_grid_final.front() == 0);
        CHECK(dt_grid_final.back() == plan.get_dt_max());
        for (size_t i = 0; i < dt_grid_final.size() - 1; ++i) {
            CHECK(dt_grid_final[i] < dt_grid_final[i + 1]);
        }
    }
}

TEST_CASE("FDMTPlan edge cases", "[dmt_plans]") {
    SECTION("Valid construction") {
        CHECK_NOTHROW(FDMTPlan(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 1, 0));
    }

    SECTION("Invalid parameters") {
        // f_min > f_max
        CHECK_THROWS_AS(
            FDMTPlan(1500.0F, 1000.0F, 500, 1024, 0.001F, 512, 1, 0),
            std::invalid_argument);
        // nchans == 0
        CHECK_THROWS_AS(FDMTPlan(1000.0F, 1500.0F, 0, 1024, 0.001F, 512, 1, 0),
                        std::invalid_argument);
        // nsamps == 0
        CHECK_THROWS_AS(FDMTPlan(1000.0F, 1500.0F, 500, 0, 0.001F, 512, 1, 0),
                        std::invalid_argument);
        // tsamp <= 0
        CHECK_THROWS_AS(FDMTPlan(1000.0F, 1500.0F, 500, 1024, 0.0F, 512, 1, 0),
                        std::invalid_argument);
        // dt_max == 0
        CHECK_THROWS_AS(FDMTPlan(1000.0F, 1500.0F, 500, 1024, 0.001F, 0, 1, 0),
                        std::invalid_argument);
        // dt_min >= dt_max
        CHECK_THROWS_AS(
            FDMTPlan(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 1, 512),
            std::invalid_argument);
        // dt_step == 0
        CHECK_THROWS_AS(
            FDMTPlan(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 0, 0),
            std::invalid_argument);
    }
}