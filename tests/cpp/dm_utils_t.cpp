#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>

#include "dmt/common/types.hpp"
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

TEST_CASE("generate_coherent_dms", "[cfdmt]") {
    const float fcenter = 1250.0F;
    const float bw      = 400.0F;
    const float tbin    = 1.0E-6F;
    const float tp      = 32.0F * tbin;
    const float f_min   = fcenter - (bw / 2);
    const float f_max   = fcenter + (bw / 2);

    // Same formula the "one-sided" (pre-symmetric-search) grid used, for
    // comparison against the new symmetric grid's coverage.
    auto one_sided_ncoherent = [&](float dm_span) {
        const float delay = kDispConst * dm_span *
                            (std::pow(f_min, kDispCoeff) -
                             std::pow(f_max, kDispCoeff));
        return static_cast<size_t>(std::ceil(delay * tbin / (tp * tp)));
    };

    SECTION("dm_min = 0: grid centers span (0, dm_max), no negative values") {
        const float dm_max = 500.0F;
        const auto dm_grid =
            utils::generate_coherent_dms(0.0F, dm_max, fcenter, bw, tbin, tp);
        REQUIRE(!dm_grid.empty());
        for (const auto dm : dm_grid) {
            CHECK(dm >= 0.0F);
            CHECK(dm <= dm_max);
        }
        // Symmetric search covers 2x the DM width per coarse trial, so
        // roughly half as many coherent trials as the one-sided search.
        const auto old_count = one_sided_ncoherent(dm_max);
        CHECK(dm_grid.size() <= old_count);
        CHECK(dm_grid.size() >=
              static_cast<size_t>(std::floor(
                  static_cast<float>(old_count) / 2.0F)));
    }

    SECTION("dm_min > 0: grid spans [dm_min, dm_max] using (dm_max - dm_min)") {
        const float dm_min = 100.0F;
        const float dm_max = 500.0F;
        const auto dm_grid = utils::generate_coherent_dms(dm_min, dm_max,
                                                           fcenter, bw, tbin,
                                                           tp);
        REQUIRE(!dm_grid.empty());
        for (const auto dm : dm_grid) {
            CHECK(dm >= dm_min);
            CHECK(dm <= dm_max);
        }
        // Trial count/step should depend on the requested span (dm_max -
        // dm_min), not dm_max alone.
        const auto span_count = one_sided_ncoherent(dm_max - dm_min);
        const auto full_count = one_sided_ncoherent(dm_max);
        CHECK(dm_grid.size() != full_count);
        CHECK(dm_grid.size() <= span_count);

        // First/last grid points are the *centers* of the first/last
        // symmetric windows, so they sit half a step inside [dm_min, dm_max].
        const float step = (dm_max - dm_min) /
                           static_cast<float>(dm_grid.size());
        CHECK(dm_grid.front() ==
              Catch::Approx(dm_min + (0.5F * step)).margin(1e-3F));
        CHECK(dm_grid.back() ==
              Catch::Approx(dm_max - (0.5F * step)).margin(1e-3F));
    }

    SECTION("degenerate dm_max == dm_min does not divide by zero") {
        const auto dm_grid =
            utils::generate_coherent_dms(10.0F, 10.0F, fcenter, bw, tbin, tp);
        REQUIRE(dm_grid.size() == 1);
        CHECK(dm_grid.front() == Catch::Approx(10.0F));
    }
}

TEST_CASE("ChannelDelayLineCPU causal multi-block streaming", "[fdmt_utils]") {
    const std::vector<float> dm_grid = {0.0F, 50.0F};
    const float f_min                = 1000.0F;
    const float f_max                = 2000.0F;
    const SizeType nchans            = 4;
    const float tsamp                = 1.0E-4F;

    utils::ChannelDelayLineCPU delay_line;
    delay_line.initialise(dm_grid, f_min, f_max, nchans, tsamp);

    SECTION("DM 0 has zero shifts across all channels") {
        const auto& shifts = delay_line.get_shifts(0);
        REQUIRE(shifts.size() == nchans);
        for (const auto s : shifts) {
            CHECK(s == 0);
        }
    }

    SECTION("DM > 0 has non-negative shifts and channel with f_min has zero shift") {
        const auto& shifts = delay_line.get_shifts(1);
        REQUIRE(shifts.size() == nchans);
        // Ascending frequency: chan 0 has lowest freq (reference f_min, zero shift)
        // Highest freq arrives earliest and requires maximum delay to align with f_min.
        CHECK(shifts.front() == 0);
        CHECK(shifts.back() > 0);
        for (SizeType c = 0; c < nchans; ++c) {
            CHECK(shifts[c] >= 0);
            if (c > 0) {
                CHECK(shifts[c] >= shifts[c - 1]);
            }
        }
    }

    SECTION("Streaming an impulse across 2 blocks produces exact causal delay") {
        const std::vector<float> small_dm_grid = {0.0F, 5.0F};
        utils::ChannelDelayLineCPU dl;
        dl.initialise(small_dm_grid, f_min, f_max, nchans, tsamp);

        const SizeType nsamps = 200;
        std::vector<float> block1(nchans * nsamps, 0.0F);
        std::vector<float> block2(nchans * nsamps, 0.0F);
        std::vector<float> out1(nchans * nsamps, 0.0F);
        std::vector<float> out2(nchans * nsamps, 0.0F);

        const auto& shifts = dl.get_shifts(1);
        const int shift_c1 = shifts[1];
        REQUIRE(shift_c1 > 0);
        REQUIRE(shift_c1 < static_cast<int>(nsamps));

        // Inject impulse in channel 1 near end of block 1
        const SizeType inj_t = nsamps - 20; // sample 180
        block1[1 * nsamps + inj_t] = 1.0F;

        // Process block 1
        dl.process(block1, out1, 1, nchans, nsamps);

        const SizeType expected_delayed_t =
            inj_t + static_cast<SizeType>(shift_c1);
        REQUIRE(expected_delayed_t >= nsamps); // Spills over into block 2
        CHECK(out1[1 * nsamps + inj_t] == 0.0F);

        // Process block 2
        dl.process(block2, out2, 1, nchans, nsamps);
        const SizeType spilled_t = expected_delayed_t - nsamps;
        CHECK(out2[1 * nsamps + spilled_t] == 1.0F);

        // Test reset_history clears history
        dl.reset_history();
        std::fill(out2.begin(), out2.end(), 0.0F);
        dl.process(block2, out2, 1, nchans, nsamps);
        for (const auto v : out2) {
            CHECK(v == 0.0F);
        }
    }
}

} // namespace dmt