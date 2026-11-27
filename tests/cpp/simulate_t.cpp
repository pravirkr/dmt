#include "dmt/utils/simulate.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <numeric>
#include <ranges>
#include <stdexcept>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include "dmt/dmt.hpp"

namespace dmt {

static_assert(ByteDataType<char> && ByteDataType<int8_t> &&
              ByteDataType<uint8_t>);
static_assert(!ByteDataType<const char> && !ByteDataType<const int8_t> &&
              !ByteDataType<const uint8_t> && !ByteDataType<volatile char> &&
              !ByteDataType<const volatile uint8_t>);

TEST_CASE("umbrella header dmt/dmt.hpp is self-contained", "[umbrella][cpu]") {
    CHECK(kDispConst > 0.0);
}

TEST_CASE("generate_pure_frb injects energy on the dispersed track",
          "[simulate][cpu]") {
    const SizeType nchans = 16;
    const SizeType nsamps = 64;
    const float f_min     = 1000.0F;
    const float f_max     = 1500.0F;
    const SizeType dt     = 0;
    const float toa       = 20.0F;
    const float amp       = 3.0F;

    const auto [waterfall, n_disp] =
        utils::generate_pure_frb(nchans, nsamps, f_min, f_max, dt, toa, amp);
    REQUIRE(waterfall.size() == nchans * nsamps);
    CHECK(n_disp > 0);
    const auto energy =
        std::accumulate(waterfall.begin(), waterfall.end(), 0.0F);
    CHECK(energy == Catch::Approx(amp * static_cast<float>(nchans)));

    const auto t = static_cast<SizeType>(toa);
    for (SizeType c = 0; c < nchans; ++c) {
        CHECK(waterfall[(c * nsamps) + t] == Catch::Approx(amp));
    }
}

TEST_CASE("generate_pure_frb with positive dt smears across channels",
          "[simulate][cpu]") {
    const SizeType nchans          = 8;
    const SizeType nsamps          = 64;
    const auto [waterfall, n_disp] = utils::generate_pure_frb(
        nchans, nsamps, 1000.0F, 1500.0F, 12, 40.0F, 1.0F);
    CHECK(n_disp > nchans);
    CHECK(std::ranges::any_of(waterfall, [](float v) { return v > 0.0F; }));
}

TEST_CASE("generate_pure_frb keeps sweeps before sample 0 out of the block",
          "[simulate][cpu]") {
    // The top channels sweep wholly before sample 0 (tstart in (-1, 0) for
    // some): truncation toward zero used to write negative or spurious
    // values into sample 0.
    const SizeType nchans = 16;
    const SizeType nsamps = 64;
    for (const float toa : {-0.25F, 0.5F, 3.7F}) {
        const auto [waterfall, n_disp] = utils::generate_pure_frb(
            nchans, nsamps, 1000.0F, 1500.0F, 30.0F, toa, 1.0F);
        CHECK(
            std::ranges::all_of(waterfall, [](float v) { return v >= 0.0F; }));
        SizeType nonzero = 0;
        for (SizeType c = 0; c < nchans; ++c) {
            double sum = 0.0;
            for (SizeType i = 0; i < nsamps; ++i) {
                sum += waterfall[(c * nsamps) + i];
                nonzero += waterfall[(c * nsamps) + i] != 0.0F ? 1 : 0;
            }
            // A channel holds at most its fluence (less when cut by sample 0).
            CHECK(sum <= 1.0 + 1e-5);
        }
        CHECK(nonzero <= n_disp);
    }
    // toa = -0.25: every channel sweeps before sample 0.
    const auto [none, n_none] = utils::generate_pure_frb(
        nchans, nsamps, 1000.0F, 1500.0F, 30.0F, -0.25F, 1.0F);
    CHECK(n_none == 0);
    CHECK(std::ranges::all_of(none, [](float v) { return v == 0.0F; }));
}

TEST_CASE("generate_pure_frb conserves each channel's fluence",
          "[simulate][cpu]") {
    const SizeType nchans          = 8;
    const SizeType nsamps          = 256;
    const auto [waterfall, n_disp] = utils::generate_pure_frb(
        nchans, nsamps, 1000.0F, 1500.0F, 90.25F, 140.5F, 2.0F);
    CHECK(n_disp > nchans);
    for (SizeType c = 0; c < nchans; ++c) {
        double sum = 0.0;
        for (SizeType i = 0; i < nsamps; ++i) {
            sum += waterfall[(c * nsamps) + i];
        }
        CHECK(sum == Catch::Approx(2.0).epsilon(1e-6));
    }
}

TEST_CASE("generate_pure_frb rejects invalid band, negative dt, and non-finite "
          "inputs",
          "[simulate][cpu]") {
    const float nan  = std::bit_cast<float>(uint32_t{0x7FC00000U});
    const float pinf = std::bit_cast<float>(uint32_t{0x7F800000U});
    const float ninf = std::bit_cast<float>(uint32_t{0xFF800000U});

    // Invalid frequency bounds.
    CHECK_THROWS_AS(utils::generate_pure_frb(8, 32, 0.0F, 1500.0F, 5.0F, 10.0F),
                    std::invalid_argument);
    CHECK_THROWS_AS(
        utils::generate_pure_frb(8, 32, -100.0F, 1500.0F, 5.0F, 10.0F),
        std::invalid_argument);
    CHECK_THROWS_AS(
        utils::generate_pure_frb(8, 32, 1500.0F, 1500.0F, 5.0F, 10.0F),
        std::invalid_argument);
    CHECK_THROWS_AS(
        utils::generate_pure_frb(8, 32, 1500.0F, 1000.0F, 5.0F, 10.0F),
        std::invalid_argument);

    // Negative dispersive delay.
    CHECK_THROWS_AS(
        utils::generate_pure_frb(8, 32, 1000.0F, 1500.0F, -1.0F, 10.0F),
        std::invalid_argument);

    // Non-finite arguments (checked via bit patterns under -ffast-math).
    for (const float bad : {nan, pinf, ninf}) {
        CHECK_THROWS_AS(
            utils::generate_pure_frb(8, 32, bad, 1500.0F, 5.0F, 10.0F, 1.0F),
            std::invalid_argument);
        CHECK_THROWS_AS(
            utils::generate_pure_frb(8, 32, 1000.0F, bad, 5.0F, 10.0F, 1.0F),
            std::invalid_argument);
        CHECK_THROWS_AS(
            utils::generate_pure_frb(8, 32, 1000.0F, 1500.0F, bad, 10.0F, 1.0F),
            std::invalid_argument);
        CHECK_THROWS_AS(
            utils::generate_pure_frb(8, 32, 1000.0F, 1500.0F, 5.0F, bad, 1.0F),
            std::invalid_argument);
        CHECK_THROWS_AS(
            utils::generate_pure_frb(8, 32, 1000.0F, 1500.0F, 5.0F, 10.0F, bad),
            std::invalid_argument);
    }

    // Extreme finite arrival times outside IndexType range stay well-defined.
    const auto [far_future, n_future] =
        utils::generate_pure_frb(8, 32, 1000.0F, 1500.0F, 10.0F, 1.0e30F, 1.0F);
    CHECK(n_future == 0);
    CHECK(std::ranges::all_of(far_future, [](float v) { return v == 0.0F; }));
}

} // namespace dmt
