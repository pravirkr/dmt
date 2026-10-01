#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <cmath>
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

    SECTION("generate_delay_table is ndm * nchans and zero at DM 0") {
        const std::vector<float> dm_arr = {0.0F, 10.0F};
        const float foff =
            -(f_max - f_min) / static_cast<float>(nchans); // descending freq
        const auto table =
            utils::generate_delay_table(dm_arr, nchans, f_max, foff, tsamp);
        REQUIRE(table.size() == dm_arr.size() * nchans);
        REQUIRE_THAT(std::vector<SizeType>(
                         table.begin(),
                         table.begin() + static_cast<std::ptrdiff_t>(nchans)),
                     Catch::Matchers::Equals(std::vector<SizeType>(nchans, 0)));
        // fch1 = f_max here (descending freq), so it is already the
        // highest-frequency (reference, zero-delay) channel; delay grows
        // toward the last (lowest-frequency) channel of the DM=10 row.
        CHECK(table.back() > 0);
        CHECK(table[nchans] == 0);
    }
}

} // namespace dmt
