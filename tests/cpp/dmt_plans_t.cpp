#include <catch2/catch_test_macros.hpp>

#include <spdlog/spdlog.h>

#include <cstddef>
#include <dmt/dmt_plans.hpp>

TEST_CASE("FDMTPlan", "[dmt_plans]") {
    FDMTPlan::set_log_level(spdlog::level::debug);

    SECTION("Constructor and getter methods") {
        FDMTPlan plan(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 1, 0);

        REQUIRE(plan.get_f_min() == 1000.0F);
        REQUIRE(plan.get_f_max() == 1500.0F);
        REQUIRE(plan.get_nchans() == 500);
        REQUIRE(plan.get_nsamps() == 1024);
        REQUIRE(plan.get_tsamp() == 0.001F);
        REQUIRE(plan.get_dt_max() == 512);
        REQUIRE(plan.get_dt_step() == 1);
        REQUIRE(plan.get_dt_min() == 0);

        REQUIRE(plan.get_df() == 1.0F);
        REQUIRE(plan.get_correction() == 0.5F);
        REQUIRE(plan.get_niters() == 9);

        const auto& container = plan.get_container();
        REQUIRE(container.df_top.size() == 10);
        REQUIRE(container.df_bot.size() == 10);
        REQUIRE(container.dt_grid_sub_top.size() == 10);
        REQUIRE(container.state_shape.size() == 10);

        REQUIRE(plan.get_dt_grid_final().size() == 513);
        REQUIRE(plan.get_dm_grid_final().size() == 513);
        REQUIRE(plan.get_dmt_size() == 513 * (1024 + 512));
    }
    /*
    SECTION("Edge cases and input validation") {
        REQUIRE_NOTHROW(
            FDMTPlan(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 1, 0));
        // f_min > f_max
        spdlog::info("f_min > f_max");
        REQUIRE_THROWS_AS(
            FDMTPlan(1500.0F, 1000.0F, 500, 1024, 0.001F, 512, 1, 0),
            std::invalid_argument);
        spdlog::info("nchans == 0");
        // nchans == 0
        REQUIRE_THROWS_AS(
            FDMTPlan(1000.0F, 1500.0F, 0, 1024, 0.001F, 512, 1, 0),
            std::invalid_argument);
        // nsamps == 0
        REQUIRE_THROWS_AS(FDMTPlan(1000.0F, 1500.0F, 500, 0, 0.001F, 512, 1, 0),
                          std::invalid_argument);
        // tsamp <= 0
        REQUIRE_THROWS_AS(
            FDMTPlan(1000.0F, 1500.0F, 500, 1024, 0.0F, 512, 1, 0),
            std::invalid_argument);
        // dt_max == 0
        REQUIRE_THROWS_AS(
            FDMTPlan(1000.0F, 1500.0F, 500, 1024, 0.001F, 0, 1, 0),
            std::invalid_argument);
        // dt_min >= dt_max
        REQUIRE_THROWS_AS(
            FDMTPlan(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 1, 512),
            std::invalid_argument);
        // dt_step == 0
        REQUIRE_THROWS_AS(
            FDMTPlan(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 0, 0),
            std::invalid_argument);
    }
    */

    SECTION("Consistency of internal calculations") {
        FDMTPlan plan(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 1, 0);
        const auto& container = plan.get_container();

        // Check the values of df_top and df_bot
        for (size_t i = 0; i < container.df_top.size() - 1; ++i) {
            REQUIRE(container.df_bot[i] < container.df_bot[i + 1]);
            REQUIRE(container.df_top[i] <= container.df_bot[i]);
        }

        // Check the values of state_shape
        for (size_t i = 0; i < container.state_shape.size() - 1; ++i) {
            REQUIRE(container.state_shape[i].nchans >=
                    container.state_shape[i + 1].nchans);
            REQUIRE(container.state_shape[i].ndt_max <=
                    container.state_shape[i + 1].ndt_max);
            REQUIRE(container.state_shape[i].ndt_min <=
                    container.state_shape[i + 1].ndt_min);
            REQUIRE(container.state_shape[i].nsamps <=
                    container.state_shape[i + 1].nsamps);
        }

        // Check the values of dt_grid_final
        const auto& dt_grid_final = plan.get_dt_grid_final();
        REQUIRE(dt_grid_final.front() == 0);
        REQUIRE(dt_grid_final.back() == plan.get_dt_max());
        for (size_t i = 0; i < dt_grid_final.size() - 1; ++i) {
            REQUIRE(dt_grid_final[i] < dt_grid_final[i + 1]);
        }
    }

    SECTION("Memory usage and buffer size") {
        FDMTPlan plan(1000.0F, 1500.0F, 500, 1024, 0.001F, 512, 1, 0);
        REQUIRE(plan.get_buffer_size() > 0);
        REQUIRE(plan.get_container().get_memory_usage() > 0);
        REQUIRE(plan.get_container().get_buffer_size() ==
                plan.get_buffer_size());
    }
}