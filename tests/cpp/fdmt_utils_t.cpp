#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <dmt/fdmt_utils.hpp>

TEST_CASE("cff", "[fdmt_utils]") {
    REQUIRE(fdmt::cff(1000.0F, 1500.0F, 1000.0F, 1500.0F) == 1.0F);
    REQUIRE(fdmt::cff(1500.0F, 1000.0F, 1500.0F, 1000.0F) == 1.0F);
    REQUIRE(fdmt::cff(1000.0F, 1000.0F, 1000.0F, 1500.0F) == 0.0F);
}

TEST_CASE("calculate_dt_sub", "[fdmt_utils]") {
    REQUIRE(fdmt::calculate_dt_sub(1000.0F, 1500.0F, 1000.0F, 1500.0F, 100) ==
            100);
    REQUIRE(fdmt::calculate_dt_sub(1000.0F, 1500.0F, 1000.0F, 1500.0F, 0) == 0);
}

TEST_CASE("add_offset_kernel", "[fdmt_utils]") {
    SECTION("Test case 1: Valid input and output vectors") {
        std::vector<float> arr1 = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
        std::vector<float> arr2 = {6.0F, 7.0F, 8.0F, 9.0F, 10.0F};
        std::vector<float> arr_out(8, 0.0F);
        size_t offset = 2;
        REQUIRE_NOTHROW(fdmt::add_offset_kernel(
            arr1.data(), arr1.size(), arr2.data(), arr2.size(), arr_out.data(),
            arr_out.size(), offset));
        std::vector<float> expected_output = {1.0F,  2.0F, 9.0F,  11.0F,
                                              13.0F, 9.0F, 10.0F, 0.0F};
        REQUIRE(arr_out == expected_output);
    }
    SECTION("Test case 2: Output size less than input size") {
        std::vector<float> arr1 = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
        std::vector<float> arr2 = {6.0F, 7.0F, 8.0F, 9.0F, 10.0F};
        std::vector<float> arr_out(4, 0.0F);
        size_t offset = 2;
        REQUIRE_THROWS_AS(fdmt::add_offset_kernel(arr1.data(), arr1.size(),
                                                  arr2.data(), arr2.size(),
                                                  arr_out.data(),
                                                  arr_out.size(), offset),
                          std::runtime_error);
    }

    SECTION("Test case 3: Offset greater than input size") {
        std::vector<float> arr1 = {1.0F, 2.0F, 3.0F};
        std::vector<float> arr2 = {4.0F, 5.0F};
        std::vector<float> arr_out(5, 0.0F);
        size_t offset = 4;
        REQUIRE_THROWS_AS(fdmt::add_offset_kernel(arr1.data(), arr1.size(),
                                                  arr2.data(), arr2.size(),
                                                  arr_out.data(),
                                                  arr_out.size(), offset),
                          std::runtime_error);
    }
    SECTION("Test case 4: Empty input vectors") {
        std::vector<float> arr1;
        std::vector<float> arr2;
        std::vector<float> arr_out(3, 0.0F);
        size_t offset = 0;
        REQUIRE_THROWS_AS(fdmt::add_offset_kernel(arr1.data(), arr1.size(),
                                                  arr2.data(), arr2.size(),
                                                  arr_out.data(),
                                                  arr_out.size(), offset),
                          std::runtime_error);
    }
    SECTION("Test case 5: Varying offsets") {
        std::vector<float> arr1 = {0.0F, 1.0F, 2.0F, 3.0F};
        std::vector<float> arr2 = {1.0F, 2.0F, 3.0F, 4.0F};
        std::vector<float> arr_out(6, 0.0F);
        size_t offset1 = 0;
        REQUIRE_NOTHROW(fdmt::add_offset_kernel(
            arr1.data(), arr1.size(), arr2.data(), arr2.size(), arr_out.data(),
            arr_out.size(), offset1));
        std::vector<float> expected_output = {1.0F, 3.0F, 5.0F,
                                              7.0F, 0.0F, 0.0F};
        REQUIRE(arr_out == expected_output);
        size_t offset2 = 1;
        REQUIRE_NOTHROW(fdmt::add_offset_kernel(
            arr1.data(), arr1.size(), arr2.data(), arr2.size(), arr_out.data(),
            arr_out.size(), offset2));
        expected_output = {0.0F, 2.0F, 4.0F, 6.0F, 4.0F, 0.0F};
        REQUIRE(arr_out == expected_output);
        size_t offset3 = 2;
        REQUIRE_NOTHROW(fdmt::add_offset_kernel(
            arr1.data(), arr1.size(), arr2.data(), arr2.size(), arr_out.data(),
            arr_out.size(), offset3));
        expected_output = {0.0F, 1.0F, 3.0F, 5.0F, 3.0F, 4.0F};
        REQUIRE(arr_out == expected_output);
        size_t offset4 = 3;
        REQUIRE_NOTHROW(fdmt::add_offset_kernel(
            arr1.data(), arr1.size(), arr2.data(), arr2.size(), arr_out.data(),
            arr_out.size(), offset4));
        expected_output = {0.0F, 1.0F, 2.0F, 4.0F, 2.0F, 3.0F};
        REQUIRE(arr_out == expected_output);
    }
}

TEST_CASE("copy_kernel", "[fdmt_utils]") {
    SECTION("Test case 1: Valid input and output vectors") {
        std::vector<float> arr1 = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
        std::vector<float> arr_out(10, 0.0F);
        ;
        REQUIRE_NOTHROW(fdmt::copy_kernel(arr1.data(), arr1.size(),
                                          arr_out.data(), arr_out.size()));
        for (size_t i = 0; i < arr1.size(); ++i) {
            REQUIRE(arr_out[i] == arr1[i]);
        }
        for (size_t i = arr1.size(); i < arr_out.size(); ++i) {
            REQUIRE(arr_out[i] == 0.0F);
        }
    }
    SECTION("Test case 2: Output size less than input size") {
        std::vector<float> arr1 = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
        std::vector<float> arr_out(3, 0.0F);
        REQUIRE_THROWS_AS(fdmt::copy_kernel(arr1.data(), arr1.size(),
                                            arr_out.data(), arr_out.size()),
                          std::runtime_error);
    }
    SECTION("Test case 4: Empty input vector") {
        std::vector<float> arr1;
        std::vector<float> arr_out(5, 0.0F);
        REQUIRE_NOTHROW(fdmt::copy_kernel(arr1.data(), arr1.size(),
                                          arr_out.data(), arr_out.size()));
        for (float i : arr_out) {
            REQUIRE(i == 0.0F);
        }
    }
}

TEST_CASE("find_closest_index", "[fdmt_utils]") {
    SECTION("Test case 1: Empty array") {
        std::vector<size_t> arr_sorted;
        REQUIRE_THROWS_AS(fdmt::find_closest_index(arr_sorted, 10),
                          std::runtime_error);
    }

    SECTION("Test case 2: Array with one element - exact match") {
        std::vector<size_t> arr_sorted{10};
        size_t val      = 10;
        size_t expected = 0;
        size_t result   = fdmt::find_closest_index(arr_sorted, val);
        REQUIRE(result == expected);
    }

    SECTION("Test case 3: Array with one element - closest match") {
        std::vector<size_t> arr_sorted{10};
        size_t val      = 15;
        size_t expected = 0;
        size_t result   = fdmt::find_closest_index(arr_sorted, val);
        REQUIRE(result == expected);
    }

    SECTION("Test case 4: Array with multiple elements - exact match") {
        std::vector<size_t> arr_sorted{10, 20, 30, 40, 50};
        size_t val      = 30;
        size_t expected = 2;
        size_t result   = fdmt::find_closest_index(arr_sorted, val);
        REQUIRE(result == expected);
    }

    SECTION(
        "Test case 5: Array with multiple elements - closest match (lower)") {
        std::vector<size_t> arr_sorted{10, 20, 30, 40, 50};
        size_t val      = 24;
        size_t expected = 1;
        size_t result   = fdmt::find_closest_index(arr_sorted, val);
        REQUIRE(result == expected);
    }

    SECTION(
        "Test case 6: Array with multiple elements - closest match (upper)") {
        std::vector<size_t> arr_sorted{10, 20, 30, 40, 50};
        size_t val      = 26;
        size_t expected = 2;
        size_t result   = fdmt::find_closest_index(arr_sorted, val);
        REQUIRE(result == expected);
    }

    SECTION("Test case 7: Array with multiple elements - value smaller than "
            "all elements") {
        std::vector<size_t> arr_sorted{10, 20, 30, 40, 50};
        size_t val      = 5;
        size_t expected = 0;
        size_t result   = fdmt::find_closest_index(arr_sorted, val);
        REQUIRE(result == expected);
    }

    SECTION("Test case 8: Array with multiple elements - value larger than all "
            "elements") {
        std::vector<size_t> arr_sorted{10, 20, 30, 40, 50};
        size_t val      = 60;
        size_t expected = 4;
        size_t result   = fdmt::find_closest_index(arr_sorted, val);
        REQUIRE(result == expected);
    }

    SECTION("Test case 9: Array with multiple elements - duplicate values") {
        std::vector<size_t> arr_sorted{10, 20, 20, 30, 40, 50};
        size_t val      = 20;
        size_t expected = 1;
        size_t result   = fdmt::find_closest_index(arr_sorted, val);
        REQUIRE(result == expected);
    }
}