#include <catch2/catch_test_macros.hpp>
#include <cstddef>

#include "dmt/dm_utils.hpp"

namespace dmt {

TEST_CASE("cff", "[fdmt_utils]") {
    REQUIRE(utils::cff(1000.0F, 1500.0F, 1000.0F, 1500.0F) == 1.0F);
    REQUIRE(utils::cff(1500.0F, 1000.0F, 1500.0F, 1000.0F) == 1.0F);
    REQUIRE(utils::cff(1000.0F, 1000.0F, 1000.0F, 1500.0F) == 0.0F);
}

TEST_CASE("calculate_dt_sub", "[fdmt_utils]") {
    REQUIRE(utils::calculate_dt_sub(1000.0F, 1500.0F, 1000.0F, 1500.0F, 100) ==
            100);
    REQUIRE(utils::calculate_dt_sub(1000.0F, 1500.0F, 1000.0F, 1500.0F, 0) ==
            0);
}

TEST_CASE("find_nearest_sorted_idx", "[fdmt_utils]") {
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

} // namespace dmt