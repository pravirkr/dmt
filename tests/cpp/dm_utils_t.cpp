#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "dmt/common/types.hpp"
#include "dmt/dm_utils.hpp"

namespace dmt {

TEST_CASE("cff", "[fdmt_utils][cpu][internal]") {
    REQUIRE(utils::cff(1000.0F, 1500.0F, 1000.0F, 1500.0F) == 1.0F);
    REQUIRE(utils::cff(1500.0F, 1000.0F, 1500.0F, 1000.0F) == 1.0F);
    REQUIRE(utils::cff(1000.0F, 1000.0F, 1000.0F, 1500.0F) == 0.0F);
}

TEST_CASE("calculate_dt_sub", "[fdmt_utils][cpu][internal]") {
    REQUIRE(utils::calculate_dt_sub(1000.0F, 1500.0F, 1000.0F, 1500.0F, 100) ==
            100);
    REQUIRE(utils::calculate_dt_sub(1000.0F, 1500.0F, 1000.0F, 1500.0F, 0) ==
            0);
}

TEST_CASE("find_nearest_sorted_idx", "[fdmt_utils][cpu][internal]") {
    SECTION("Test case 1: Empty array") {
        std::vector<size_t> arr_sorted;
        REQUIRE_THROWS_AS(utils::find_nearest_sorted_idx(arr_sorted, 10),
                          std::invalid_argument);
    }

    SECTION("Test case 2: Array with one element - exact match") {
        std::vector<size_t> arr_sorted{10};
        size_t val      = 10;
        size_t expected = 0;
        size_t result   = utils::find_nearest_sorted_idx(arr_sorted, val);
        REQUIRE(result == expected);
    }

    SECTION("Test case 3: Array with one element - closest match") {
        std::vector<size_t> arr_sorted{10};
        size_t val      = 15;
        size_t expected = 0;
        size_t result   = utils::find_nearest_sorted_idx(arr_sorted, val);
        REQUIRE(result == expected);
    }

    SECTION("Test case 4: Array with multiple elements - exact match") {
        std::vector<size_t> arr_sorted{10, 20, 30, 40, 50};
        size_t val      = 30;
        size_t expected = 2;
        size_t result   = utils::find_nearest_sorted_idx(arr_sorted, val);
        REQUIRE(result == expected);
    }

    SECTION(
        "Test case 5: Array with multiple elements - closest match (lower)") {
        std::vector<size_t> arr_sorted{10, 20, 30, 40, 50};
        size_t val      = 24;
        size_t expected = 1;
        size_t result   = utils::find_nearest_sorted_idx(arr_sorted, val);
        REQUIRE(result == expected);
    }

    SECTION(
        "Test case 6: Array with multiple elements - closest match (upper)") {
        std::vector<size_t> arr_sorted{10, 20, 30, 40, 50};
        size_t val      = 26;
        size_t expected = 2;
        size_t result   = utils::find_nearest_sorted_idx(arr_sorted, val);
        REQUIRE(result == expected);
    }

    SECTION("Test case 7: Array with multiple elements - value smaller than "
            "all elements") {
        std::vector<size_t> arr_sorted{10, 20, 30, 40, 50};
        size_t val      = 5;
        size_t expected = 0;
        size_t result   = utils::find_nearest_sorted_idx(arr_sorted, val);
        REQUIRE(result == expected);
    }

    SECTION("Test case 8: Array with multiple elements - value larger than all "
            "elements") {
        std::vector<size_t> arr_sorted{10, 20, 30, 40, 50};
        size_t val      = 60;
        size_t expected = 4;
        size_t result   = utils::find_nearest_sorted_idx(arr_sorted, val);
        REQUIRE(result == expected);
    }

    SECTION("Test case 9: Array with multiple elements - duplicate values") {
        std::vector<size_t> arr_sorted{10, 20, 20, 30, 40, 50};
        size_t val      = 20;
        size_t expected = 1;
        size_t result   = utils::find_nearest_sorted_idx(arr_sorted, val);
        REQUIRE(result == expected);
    }
}

TEST_CASE("get_dmconv and delay tables", "[fdmt_utils][cpu][internal]") {
    const float f_min     = 1000.0F;
    const float f_max     = 1500.0F;
    const float tsamp     = 0.001F;
    const SizeType nchans = 8;

    SECTION("get_dmconv is positive for f_max > f_min") {
        const float conv = utils::get_dmconv(f_min, f_max, tsamp);
        CHECK(conv > 0.0F);
        CHECK(utils::get_dmconv(f_min, f_max, 2.0F * tsamp) ==
              Catch::Approx(2.0F * conv));
    }

    SECTION("generate_delay_table is ndm * nchans, offset at DM 0") {
        const std::vector<float> dm_arr = {0.0F, 10.0F};
        const float df    = (f_max - f_min) / static_cast<float>(nchans);
        const auto offset = utils::generate_delay_offset(dm_arr, nchans, f_min,
                                                         df, tsamp, f_min);
        const auto table  = utils::generate_delay_table(dm_arr, nchans, f_min,
                                                        df, tsamp, f_min);
        REQUIRE(table.size() == dm_arr.size() * nchans);
        CHECK(offset > 0);
        // DM 0: no lag, every channel at the offset.
        REQUIRE_THAT(
            std::vector<SizeType>(table.begin(),
                                  table.begin() +
                                      static_cast<std::ptrdiff_t>(nchans)),
            Catch::Matchers::Equals(std::vector<SizeType>(nchans, offset)));
        // The top channel of the largest DM lags f_min the most (offset is
        // its lag rounded up); the bottom channel the least.
        CHECK(table.back() <= 1);
        CHECK(table[nchans] > table.back());
        CHECK(table[nchans] <= offset);
        // Descending or non-positive bands are rejected.
        CHECK_THROWS_AS(utils::generate_delay_table(dm_arr, nchans, f_max, -df,
                                                    tsamp, f_min),
                        std::invalid_argument);
        CHECK_THROWS_AS(
            utils::generate_delay_table(dm_arr, nchans, f_min, df, tsamp, -1.0),
            std::invalid_argument);
        const auto nan_d = std::bit_cast<double>(0x7FF8000000000000ULL);
        const auto inf_d = std::bit_cast<double>(0x7FF0000000000000ULL);
        CHECK_THROWS_AS(utils::generate_delay_table(dm_arr, nchans, f_min, df,
                                                    tsamp, nan_d),
                        std::invalid_argument);
        CHECK_THROWS_AS(utils::generate_delay_table(dm_arr, nchans, f_min, df,
                                                    tsamp, inf_d),
                        std::invalid_argument);
    }

    SECTION("f_ref at f_max needs no offset") {
        const std::vector<float> dm_arr = {0.0F, 10.0F};
        const float df    = (f_max - f_min) / static_cast<float>(nchans);
        const double fref = static_cast<double>(f_max);
        CHECK(utils::generate_delay_offset(dm_arr, nchans, f_min, df, tsamp,
                                           fref) == 0);
        const auto table =
            utils::generate_delay_table(dm_arr, nchans, f_min, df, tsamp, fref);
        // Delays grow toward the bottom channel.
        CHECK(table[nchans] > table.back());
    }

    SECTION("generate_levin_dm_grid rejects non-finite parameters") {
        const auto nan_f = std::bit_cast<float>(0x7FC00000U);
        const auto inf_f = std::bit_cast<float>(0x7F800000U);
        CHECK_THROWS_AS(utils::generate_levin_dm_grid(0.0F, 100.0F, tsamp,
                                                      inf_f, f_min, f_max,
                                                      nchans, 1.25F),
                        std::invalid_argument);
        CHECK_THROWS_AS(utils::generate_levin_dm_grid(0.0F, inf_f, tsamp,
                                                      0.001F, f_min, f_max,
                                                      nchans, 1.25F),
                        std::invalid_argument);
        CHECK_THROWS_AS(utils::generate_levin_dm_grid(nan_f, 100.0F, tsamp,
                                                      0.001F, f_min, f_max,
                                                      nchans, 1.25F),
                        std::invalid_argument);
    }
}

} // namespace dmt
