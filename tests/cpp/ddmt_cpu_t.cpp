#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include "dmt/algorithms/ddmt.hpp"
#include "dmt/bit_pack_utils.hpp"
#include "dmt/common/plans.hpp"
#include "dmt/utils/simulate.hpp"

namespace dmt {

using algorithms::DDMTCPU;
using plans::DDMTPlan;

TEST_CASE("DDMTPlan construction and getters", "[ddmt][cpu]") {
    const float f_min     = 1000.0F;
    const float f_max     = 1500.0F;
    const SizeType nchans = 16;
    const float tsamp     = 0.001F;

    SECTION("regular DM grid") {
        DDMTPlan plan(f_min, f_max, nchans, tsamp, 20.0F, 5.0F, 0.0F);
        CHECK(plan.get_f_min() == f_min);
        CHECK(plan.get_f_max() == f_max);
        CHECK(plan.get_nchans() == nchans);
        CHECK(plan.get_tsamp() == tsamp);
        CHECK(plan.get_dm_arr().size() > 1);
        CHECK(plan.get_container().nchans == nchans);
        CHECK(plan.get_container().delay_table.size() ==
              plan.get_dm_arr().size() * nchans);
        REQUIRE_THAT(plan.get_dm_grid(),
                     Catch::Matchers::Equals(plan.get_dm_arr()));
    }

    SECTION("custom dm_arr") {
        const std::vector<float> dms = {0.0F, 5.0F, 12.5F};
        DDMTPlan plan(f_min, f_max, nchans, tsamp, dms);
        REQUIRE_THAT(plan.get_dm_arr(), Catch::Matchers::Equals(dms));
    }
}

TEST_CASE("DDMTPlan delay table is non-negative and non-increasing per DM",
          "[ddmt][cpu]") {
    const float f_min     = 1000.0F;
    const float f_max     = 1500.0F;
    const SizeType nchans = 16;
    const float tsamp     = 0.001F;

    DDMTPlan plan(f_min, f_max, nchans, tsamp, 20.0F, 5.0F, 0.0F);
    const auto& delay_table = plan.get_container().delay_table;
    const auto dm_arr       = plan.get_dm_arr();

    // Delay is referenced to the highest-frequency (last, first-arriving)
    // channel: execute_dedisp() reads d_in[i_samp + delay], which only looks
    // forward, so every channel must have delay >= 0, decreasing to exactly
    // 0 at the last (highest-frequency) channel.
    for (SizeType idm = 0; idm < dm_arr.size(); ++idm) {
        SizeType prev = std::numeric_limits<SizeType>::max();
        for (SizeType ichan = 0; ichan < nchans; ++ichan) {
            const auto delay = delay_table[(idm * nchans) + ichan];
            CHECK(delay <= prev);
            prev = delay;
        }
        CHECK(delay_table[(idm * nchans) + nchans - 1] == 0);
    }

    // Independently computed expected delay (samples) for the highest DM at
    // the first (lowest-frequency) channel, referenced to the *last
    // channel's* frequency (not the band edge f_max): delay =
    // round(kDispConst/tsamp * dm * (1/f_min^2 - 1/f_last_chan^2)).
    const auto df        = (f_max - f_min) / static_cast<float>(nchans);
    const auto f_last     = f_min + (static_cast<float>(nchans - 1) * df);
    const auto dm_max    = dm_arr.back();
    const auto expected  = static_cast<SizeType>(std::nearbyint(
        kDispConst / tsamp *
        ((1.0F / (f_min * f_min)) - (1.0F / (f_last * f_last))) * dm_max));
    const auto front_of_last_row = delay_table[(dm_arr.size() - 1) * nchans];
    CHECK(front_of_last_row == expected);
    CHECK(front_of_last_row > 0);
}

TEST_CASE("DDMTCPU execute recovers a zero-DM ones waterfall", "[ddmt][cpu]") {
    const float f_min     = 1000.0F;
    const float f_max     = 1500.0F;
    const SizeType nchans = 8;
    const SizeType nsamps = 64;
    const float tsamp     = 0.001F;

    DDMTCPU ddmt(f_min, f_max, nchans, tsamp, 10.0F, 5.0F, 0.0F);
    const auto& plan_c = ddmt.get_plan().get_container();
    const auto max_delay =
        *std::ranges::max_element(plan_c.delay_table);
    REQUIRE(nsamps > max_delay);
    const auto nsamps_reduced = nsamps - max_delay;
    const auto dm_count       = plan_c.dm_arr.size();

    std::vector<float> waterfall(nchans * nsamps, 1.0F);
    std::vector<float> dmt(dm_count * nsamps_reduced, 0.0F);
    ddmt.execute(waterfall, dmt);

    // DM 0 has zero delay on every channel, so each output sample is nchans.
    std::vector<float> dm0(
        dmt.begin(), dmt.begin() + static_cast<std::ptrdiff_t>(nsamps_reduced));
    REQUIRE_THAT(dm0, Catch::Matchers::Equals(std::vector<float>(
                          nsamps_reduced, static_cast<float>(nchans))));
}

TEST_CASE("DDMTCPU custom dm_arr and mismatched output is a no-op",
          "[ddmt][cpu]") {
    const float f_min            = 1000.0F;
    const float f_max            = 1500.0F;
    const SizeType nchans        = 8;
    const SizeType nsamps        = 32;
    const float tsamp            = 0.001F;
    const std::vector<float> dms = {0.0F, 2.0F};

    DDMTCPU ddmt(f_min, f_max, nchans, tsamp, dms);
    std::vector<float> waterfall(nchans * nsamps, 1.0F);
    std::vector<float> bad(4, 42.0F);
    ddmt.execute(waterfall, bad);
    REQUIRE_THAT(bad, Catch::Matchers::Equals(std::vector<float>(4, 42.0F)));
}

TEST_CASE("DDMTCPU packed-integer execute matches the float path",
          "[ddmt][cpu]") {
    const float f_min     = 1000.0F;
    const float f_max     = 1500.0F;
    const SizeType nchans = 16;
    const SizeType nsamps = 96;
    const float tsamp     = 0.001F;
    const std::vector<float> dms = {0.0F, 5.0F, 12.0F};

    // Deterministic 0-255 pattern, exactly representable as float.
    std::vector<float> waterfall_f(nchans * nsamps);
    std::vector<uint8_t> waterfall_u8(nchans * nsamps);
    for (SizeType ichan = 0; ichan < nchans; ++ichan) {
        for (SizeType isamp = 0; isamp < nsamps; ++isamp) {
            const auto val =
                static_cast<uint8_t>(((ichan * 7) + (isamp * 3)) % 256);
            waterfall_f[(ichan * nsamps) + isamp]  = static_cast<float>(val);
            waterfall_u8[(ichan * nsamps) + isamp] = val;
        }
    }

    DDMTCPU ddmt_f(f_min, f_max, nchans, tsamp, dms);
    DDMTCPU ddmt_i(f_min, f_max, nchans, tsamp, dms, /*nthreads=*/1,
                  /*nbits=*/8);
    CHECK(ddmt_i.get_plan().get_nbits() == 8);

    const auto max_delay =
        *std::ranges::max_element(ddmt_f.get_plan().get_container().delay_table);
    const auto nsamps_reduced = nsamps - max_delay;
    const auto dm_count       = dms.size();

    std::vector<float> dmt_f(dm_count * nsamps_reduced, 0.0F);
    std::vector<int32_t> dmt_i(dm_count * nsamps_reduced, 0);
    ddmt_f.execute(waterfall_f, dmt_f);
    ddmt_i.execute(waterfall_u8, nsamps, dmt_i);

    for (SizeType i = 0; i < dmt_f.size(); ++i) {
        CHECK(static_cast<float>(dmt_i[i]) == dmt_f[i]);
    }
}

TEST_CASE("DDMTCPU wrong execute overload for the configured nbits is a no-op",
          "[ddmt][cpu]") {
    const float f_min     = 1000.0F;
    const float f_max     = 1500.0F;
    const SizeType nchans = 8;
    const SizeType nsamps = 32;
    const float tsamp     = 0.001F;

    DDMTCPU ddmt_float(f_min, f_max, nchans, tsamp, 10.0F, 5.0F, 0.0F);
    std::vector<uint8_t> packed(nchans * nsamps, 1);
    std::vector<int32_t> bad_out(4, 42);
    ddmt_float.execute(packed, nsamps, bad_out);
    REQUIRE_THAT(bad_out, Catch::Matchers::Equals(std::vector<int32_t>(4, 42)));

    DDMTCPU ddmt_packed(f_min, f_max, nchans, tsamp, 10.0F, 5.0F, 0.0F,
                       /*nthreads=*/1, /*nbits=*/8);
    std::vector<float> float_waterfall(nchans * nsamps, 1.0F);
    std::vector<float> bad_out_f(4, 42.0F);
    ddmt_packed.execute(float_waterfall, bad_out_f);
    REQUIRE_THAT(bad_out_f,
                Catch::Matchers::Equals(std::vector<float>(4, 42.0F)));
}

TEST_CASE("DDMTCPU kill_mask excludes masked channels from the sum",
          "[ddmt][cpu]") {
    const float f_min     = 1000.0F;
    const float f_max     = 1500.0F;
    const SizeType nchans = 8;
    const SizeType nsamps = 64;
    const float tsamp     = 0.001F;

    std::vector<uint8_t> kill_mask(nchans, 1);
    kill_mask[0] = 0; // mask out channel 0

    DDMTCPU ddmt(f_min, f_max, nchans, tsamp, 10.0F, 5.0F, 0.0F,
                /*nthreads=*/1, /*nbits=*/32, kill_mask);
    REQUIRE_THAT(ddmt.get_plan().get_kill_mask(),
                Catch::Matchers::Equals(kill_mask));

    const auto& plan_c = ddmt.get_plan().get_container();
    const auto max_delay =
        *std::ranges::max_element(plan_c.delay_table);
    const auto nsamps_reduced = nsamps - max_delay;
    const auto dm_count       = plan_c.dm_arr.size();

    std::vector<float> waterfall(nchans * nsamps, 1.0F);
    std::vector<float> dmt(dm_count * nsamps_reduced, 0.0F);
    ddmt.execute(waterfall, dmt);

    // DM 0: masked channel excluded, so each output sample is nchans - 1.
    std::vector<float> dm0(
        dmt.begin(), dmt.begin() + static_cast<std::ptrdiff_t>(nsamps_reduced));
    REQUIRE_THAT(dm0, Catch::Matchers::Equals(std::vector<float>(
                          nsamps_reduced, static_cast<float>(nchans - 1))));
}

TEST_CASE("DDMTPlan effective variance tracks active channel count",
          "[ddmt][cpu]") {
    const float f_min     = 1000.0F;
    const float f_max     = 1500.0F;
    const SizeType nchans = 8;
    const float tsamp     = 0.001F;

    DDMTPlan plan(f_min, f_max, nchans, tsamp, 10.0F, 5.0F, 0.0F);
    CHECK(plan.get_effective_variance() == Catch::Approx(nchans));
    CHECK(plan.get_effective_sigma() ==
         Catch::Approx(std::sqrt(static_cast<float>(nchans))));
    const auto grid = plan.get_effective_variance_grid();
    REQUIRE(grid.size() == plan.get_dm_arr().size());
    for (const auto& v : grid) {
        CHECK(v == Catch::Approx(nchans));
    }

    std::vector<uint8_t> kill_mask(nchans, 1);
    kill_mask[0] = 0;
    kill_mask[1] = 0;
    plan.set_kill_mask(kill_mask);
    CHECK(plan.get_effective_variance() == Catch::Approx(nchans - 2));
}

TEST_CASE("DDMTCPU streaming reproduces a monolithic call", "[ddmt][cpu]") {
    const float f_min     = 1000.0F;
    const float f_max     = 1500.0F;
    const SizeType nchans = 8;
    const SizeType nsamps = 200;
    const float tsamp     = 0.001F;
    const std::vector<float> dms = {0.0F, 3.0F, 7.0F};

    std::vector<float> waterfall(nchans * nsamps);
    for (SizeType ichan = 0; ichan < nchans; ++ichan) {
        for (SizeType isamp = 0; isamp < nsamps; ++isamp) {
            waterfall[(ichan * nsamps) + isamp] = static_cast<float>(
                std::sin(0.1 * static_cast<double>(isamp)) +
                (0.01 * static_cast<double>(ichan)));
        }
    }

    DDMTCPU monolithic(f_min, f_max, nchans, tsamp, dms);
    const auto max_delay = *std::ranges::max_element(
        monolithic.get_plan().get_container().delay_table);
    const auto nsamps_reduced = nsamps - max_delay;
    std::vector<float> dmt_mono(dms.size() * nsamps_reduced, 0.0F);
    monolithic.execute(waterfall, dmt_mono);

    // Same data delivered as two unequal chunks to a fresh, history-aware
    // instance; concatenated output must match the monolithic call exactly.
    DDMTCPU streamed(f_min, f_max, nchans, tsamp, dms);
    const SizeType split = 70;
    std::vector<float> chunk1(nchans * split);
    std::vector<float> chunk2(nchans * (nsamps - split));
    for (SizeType ichan = 0; ichan < nchans; ++ichan) {
        std::copy_n(&waterfall[ichan * nsamps], split, &chunk1[ichan * split]);
        std::copy_n(&waterfall[(ichan * nsamps) + split], nsamps - split,
                   &chunk2[ichan * (nsamps - split)]);
    }

    const auto n_out1 = streamed.get_output_nsamps(split);
    std::vector<float> dmt1(dms.size() * n_out1, 0.0F);
    streamed.execute(chunk1, dmt1);

    const auto n_out2 = streamed.get_output_nsamps(nsamps - split);
    std::vector<float> dmt2(dms.size() * n_out2, 0.0F);
    streamed.execute(chunk2, dmt2);

    REQUIRE(n_out1 + n_out2 == nsamps_reduced);

    // Reassemble per-DM rows (streamed output is chunked in time, not
    // concatenated flat, since each chunk's output is itself DM-major).
    std::vector<float> dmt_streamed(dms.size() * nsamps_reduced, 0.0F);
    for (SizeType idm = 0; idm < dms.size(); ++idm) {
        std::copy_n(&dmt1[idm * n_out1], n_out1,
                   &dmt_streamed[idm * nsamps_reduced]);
        std::copy_n(&dmt2[idm * n_out2], n_out2,
                   &dmt_streamed[(idm * nsamps_reduced) + n_out1]);
    }

    for (SizeType i = 0; i < dmt_mono.size(); ++i) {
        CHECK(dmt_streamed[i] == Catch::Approx(dmt_mono[i]).margin(1e-5));
    }

    // reset_history() returns execute() to cold, one-shot behavior.
    streamed.reset_history();
    CHECK(streamed.get_output_nsamps(split) == split - max_delay);
}

TEST_CASE("DDMTCPU save_history/load_history multiplex two streams",
          "[ddmt][cpu]") {
    const float f_min     = 1000.0F;
    const float f_max     = 1500.0F;
    const SizeType nchans = 8;
    const float tsamp     = 0.001F;
    const std::vector<float> dms = {0.0F, 4.0F};

    DDMTCPU ddmt(f_min, f_max, nchans, tsamp, dms);
    const auto max_delay =
        *std::ranges::max_element(ddmt.get_plan().get_container().delay_table);
    REQUIRE(max_delay > 0);

    // Warm up stream A.
    const SizeType n0 = max_delay + 10;
    std::vector<float> block_a0(nchans * n0, 1.0F);
    std::vector<float> out_a0(dms.size() * ddmt.get_output_nsamps(n0), 0.0F);
    ddmt.execute(block_a0, out_a0);
    REQUIRE(ddmt.history_state_size() == nchans * max_delay);

    std::vector<float> state_a(ddmt.history_state_size());
    ddmt.save_history(state_a);

    // Reuse the same instance for a differently-warmed stream B.
    ddmt.reset_history();
    const SizeType n1 = max_delay + 5;
    std::vector<float> block_b0(nchans * n1, 2.0F);
    std::vector<float> out_b0(dms.size() * ddmt.get_output_nsamps(n1), 0.0F);
    ddmt.execute(block_b0, out_b0);
    std::vector<float> state_b(ddmt.history_state_size());
    ddmt.save_history(state_b);
    CHECK_FALSE(std::ranges::equal(state_a, state_b));

    // Resume stream A and confirm its history round-tripped correctly: DM=0
    // continuing on all-ones input should keep producing nchans exactly.
    ddmt.load_history(state_a);
    std::vector<float> block_a1(nchans * 5, 1.0F);
    std::vector<float> out_a1(dms.size() * ddmt.get_output_nsamps(5), 0.0F);
    ddmt.execute(block_a1, out_a1);
    for (SizeType i = 0; i < ddmt.get_output_nsamps(5); ++i) {
        CHECK(out_a1[i] == Catch::Approx(static_cast<float>(nchans)));
    }
}

TEST_CASE("DDMTCPU add_frb_track-style peak on the injected DM",
          "[ddmt][cpu]") {
    const float f_min     = 1000.0F;
    const float f_max     = 1500.0F;
    const SizeType nchans = 16;
    const SizeType nsamps = 128;
    const float tsamp     = 0.001F;
    const SizeType dt     = 8;
    const float toa       = 40.0F;

    const auto [waterfall, n_disp] =
        utils::generate_pure_frb(nchans, nsamps, f_min, f_max, dt, toa, 1.0F);
    REQUIRE(n_disp > 0);

    const float dm = static_cast<float>(dt) * tsamp /
                     (kDispConst * (std::pow(f_min, kDispCoeff) -
                                    std::pow(f_max, kDispCoeff)));
    DDMTCPU ddmt(f_min, f_max, nchans, tsamp, std::vector<float>{0.0F, dm});
    const auto& plan_c = ddmt.get_plan().get_container();
    const auto max_delay =
        *std::ranges::max_element(plan_c.delay_table);
    const auto nsamps_reduced = nsamps - max_delay;
    std::vector<float> dmt(plan_c.dm_arr.size() * nsamps_reduced, 0.0F);
    ddmt.execute(waterfall, dmt);

    const auto* row0 = dmt.data();
    const auto* row1 = dmt.data() + nsamps_reduced;
    const auto peak0 = *std::max_element(row0, row0 + nsamps_reduced);
    const auto peak1 = *std::max_element(row1, row1 + nsamps_reduced);
    CHECK(peak1 >= peak0);
    CHECK(peak1 > 0.0F);
}

TEST_CASE("DDMTPlan and DDMTCPU with LevinConfig", "[ddmt][cpu]") {
    const float f_min     = 1000.0F;
    const float f_max     = 1500.0F;
    const SizeType nchans = 16;
    const float tsamp     = 0.001F;

    plans::LevinConfig levin{
        .dm_start    = 0.0F,
        .dm_end      = 50.0F,
        .pulse_width = 0.001F,
        .tol         = 1.25F,
    };

    const auto grid = DDMTPlan::generate_levin_dm_grid(
        levin.dm_start, levin.dm_end, tsamp, levin.pulse_width, f_min, f_max,
        nchans, levin.tol);
    REQUIRE(grid.size() > 1);
    CHECK(grid.front() == 0.0F);
    CHECK(grid.back() >= levin.dm_end);

    // Verify grid is strictly monotonically increasing
    for (size_t i = 1; i < grid.size(); ++i) {
        CHECK(grid[i] > grid[i - 1]);
    }

    DDMTPlan plan(f_min, f_max, nchans, tsamp, levin);
    REQUIRE_THAT(plan.get_dm_arr(), Catch::Matchers::Equals(grid));

    DDMTCPU ddmt(f_min, f_max, nchans, tsamp, levin);
    REQUIRE_THAT(ddmt.get_plan().get_dm_arr(), Catch::Matchers::Equals(grid));

    DDMTCPU ddmt_from_plan(plan, /*nthreads=*/2);
    REQUIRE_THAT(ddmt_from_plan.get_plan().get_dm_arr(), Catch::Matchers::Equals(grid));
}

TEST_CASE("DDMTCPU execute_time_major matches channel-major packed", "[ddmt][cpu]") {
    const float f_min     = 1000.0F;
    const float f_max     = 1500.0F;
    const SizeType nchans = 16;
    const SizeType nsamps = 64;
    const float tsamp     = 0.001F;
    const std::vector<float> dms = {0.0F, 5.0F, 15.0F};

    const auto test_bits = [&](unsigned nbits) {
        DYNAMIC_SECTION("nbits = " << nbits) {
            DDMTPlan plan(f_min, f_max, nchans, tsamp, dms, /*verbose=*/false, nbits);
            DDMTCPU ddmt(plan);

            const auto max_delay = *std::ranges::max_element(plan.get_container().delay_table);
            const auto nsamps_reduced = nsamps - max_delay;
            const auto dm_count       = dms.size();

            // Prepare pseudo-random samples in range [0, (1<<nbits)-1]
            const auto mask = (1ULL << nbits) - 1;
            std::vector<uint64_t> raw_data(nchans * nsamps);
            for (size_t i = 0; i < raw_data.size(); ++i) {
                raw_data[i] = (i * 37 + 11) & mask;
            }

            // 1. Channel-major packing: row per channel
            const auto chan_row_bytes = (nsamps * nbits + 7) / 8;
            std::vector<uint8_t> chan_major(nchans * chan_row_bytes, 0);
            for (SizeType c = 0; c < nchans; ++c) {
                for (SizeType s = 0; s < nsamps; ++s) {
                    const auto val = raw_data[(c * nsamps) + s];
                    const auto bit_offset = s * nbits;
                    const auto byte_idx   = bit_offset / 8;
                    const auto bit_sub    = bit_offset % 8;
                    if (nbits == 8) {
                        chan_major[(c * chan_row_bytes) + s] = static_cast<uint8_t>(val);
                    } else if (nbits == 16) {
                        auto* ptr = reinterpret_cast<uint16_t*>(&chan_major[c * chan_row_bytes]);
                        ptr[s] = static_cast<uint16_t>(val);
                    } else {
                        chan_major[(c * chan_row_bytes) + byte_idx] |=
                            static_cast<uint8_t>(val << bit_sub);
                    }
                }
            }

            // 2. Time-major packing: channels packed per time sample
            const auto time_samp_bytes = (nchans * nbits + 7) / 8;
            std::vector<uint8_t> time_major(nsamps * time_samp_bytes, 0);
            for (SizeType s = 0; s < nsamps; ++s) {
                for (SizeType c = 0; c < nchans; ++c) {
                    const auto val = raw_data[(c * nsamps) + s];
                    const auto bit_offset = c * nbits;
                    const auto byte_idx   = bit_offset / 8;
                    const auto bit_sub    = bit_offset % 8;
                    if (nbits == 8) {
                        time_major[(s * time_samp_bytes) + c] = static_cast<uint8_t>(val);
                    } else if (nbits == 16) {
                        auto* ptr = reinterpret_cast<uint16_t*>(&time_major[s * time_samp_bytes]);
                        ptr[c] = static_cast<uint16_t>(val);
                    } else {
                        time_major[(s * time_samp_bytes) + byte_idx] |=
                            static_cast<uint8_t>(val << bit_sub);
                    }
                }
            }

            std::vector<int32_t> out_chan(dm_count * nsamps_reduced, 0);
            std::vector<int32_t> out_time(dm_count * nsamps_reduced, 0);

            ddmt.execute(chan_major, nsamps, out_chan);
            ddmt.execute_time_major(time_major, nsamps, out_time);

            REQUIRE_THAT(out_time, Catch::Matchers::Equals(out_chan));
        }
    };

    test_bits(1);
    test_bits(2);
    test_bits(4);
    test_bits(8);
    test_bits(16);
}

TEST_CASE("DDMTCPU multi-beam float execute matches per-beam single-beam calls",
          "[ddmt][cpu]") {
    const float f_min      = 1000.0F;
    const float f_max      = 1500.0F;
    const SizeType nchans  = 8;
    const SizeType nsamps  = 64;
    const SizeType nbeams  = 3;
    const float tsamp      = 0.001F;
    const std::vector<float> dms = {0.0F, 5.0F, 10.0F};

    // Distinct, deterministic data per beam so beams can't trivially agree.
    std::vector<float> waterfall(nbeams * nchans * nsamps);
    for (SizeType ibeam = 0; ibeam < nbeams; ++ibeam) {
        for (SizeType ichan = 0; ichan < nchans; ++ichan) {
            for (SizeType isamp = 0; isamp < nsamps; ++isamp) {
                const auto idx = (((ibeam * nchans) + ichan) * nsamps) + isamp;
                waterfall[idx] = static_cast<float>(
                    ((ibeam + 1) * 100) + (ichan * 7) + isamp);
            }
        }
    }

    DDMTCPU ddmt_multi(f_min, f_max, nchans, tsamp, dms, /*nthreads=*/1,
                      /*nbits=*/32, /*kill_mask=*/{}, nbeams);
    CHECK(ddmt_multi.get_nbeams() == nbeams);
    const auto max_delay = *std::ranges::max_element(
        ddmt_multi.get_plan().get_container().delay_table);
    const auto nsamps_reduced = nsamps - max_delay;
    const auto dm_count       = dms.size();

    std::vector<float> dmt_multi(nbeams * dm_count * nsamps_reduced, 0.0F);
    ddmt_multi.execute(waterfall, dmt_multi);

    // execute(float) retains streaming history across calls by design (see
    // reset_history()'s doc comment) -- a fresh instance per beam avoids
    // beam N's reference computation being contaminated by beam N-1's tail.
    for (SizeType ibeam = 0; ibeam < nbeams; ++ibeam) {
        DDMTCPU ddmt_single(f_min, f_max, nchans, tsamp, dms);
        std::vector<float> beam_waterfall(
            waterfall.begin() + static_cast<std::ptrdiff_t>(ibeam * nchans * nsamps),
            waterfall.begin() +
                static_cast<std::ptrdiff_t>((ibeam + 1) * nchans * nsamps));
        std::vector<float> dmt_single(dm_count * nsamps_reduced, 0.0F);
        ddmt_single.execute(beam_waterfall, dmt_single);

        const auto* beam_out = dmt_multi.data() + (ibeam * dm_count * nsamps_reduced);
        for (SizeType i = 0; i < dmt_single.size(); ++i) {
            CHECK(beam_out[i] == dmt_single[i]);
        }
    }
}

TEST_CASE("DDMTCPU multi-beam packed execute matches per-beam single-beam calls",
          "[ddmt][cpu]") {
    const float f_min      = 1000.0F;
    const float f_max      = 1500.0F;
    const SizeType nchans  = 8;
    const SizeType nsamps  = 64;
    const SizeType nbeams  = 2;
    const float tsamp      = 0.001F;
    const std::vector<float> dms = {0.0F, 6.0F};

    std::vector<uint8_t> waterfall(nbeams * nchans * nsamps);
    for (SizeType i = 0; i < waterfall.size(); ++i) {
        waterfall[i] = static_cast<uint8_t>((i * 13 + 5) % 256);
    }

    DDMTCPU ddmt_multi(f_min, f_max, nchans, tsamp, dms, /*nthreads=*/1,
                      /*nbits=*/8, /*kill_mask=*/{}, nbeams);
    const auto max_delay = *std::ranges::max_element(
        ddmt_multi.get_plan().get_container().delay_table);
    const auto nsamps_reduced = nsamps - max_delay;
    const auto dm_count       = dms.size();
    const auto row_bytes      = nsamps; // nbits == 8

    std::vector<int32_t> dmt_multi(nbeams * dm_count * nsamps_reduced, 0);
    ddmt_multi.execute(waterfall, nsamps, dmt_multi);

    for (SizeType ibeam = 0; ibeam < nbeams; ++ibeam) {
        DDMTCPU ddmt_single(f_min, f_max, nchans, tsamp, dms, /*nthreads=*/1,
                           /*nbits=*/8);
        std::vector<uint8_t> beam_waterfall(
            waterfall.begin() +
                static_cast<std::ptrdiff_t>(ibeam * nchans * row_bytes),
            waterfall.begin() +
                static_cast<std::ptrdiff_t>((ibeam + 1) * nchans * row_bytes));
        std::vector<int32_t> dmt_single(dm_count * nsamps_reduced, 0);
        ddmt_single.execute(beam_waterfall, nsamps, dmt_single);

        const auto* beam_out =
            dmt_multi.data() + (ibeam * dm_count * nsamps_reduced);
        for (SizeType i = 0; i < dmt_single.size(); ++i) {
            CHECK(beam_out[i] == dmt_single[i]);
        }
    }
}

TEST_CASE("DDMTCPU multi-beam time-major matches per-beam single-beam calls",
          "[ddmt][cpu]") {
    const float f_min      = 1000.0F;
    const float f_max      = 1500.0F;
    const SizeType nchans  = 8;
    const SizeType nsamps  = 48;
    const SizeType nbeams  = 2;
    const float tsamp      = 0.001F;
    const std::vector<float> dms = {0.0F, 4.0F};
    const auto samp_bytes  = nchans; // nbits == 8

    std::vector<uint8_t> filterbank(nbeams * nsamps * samp_bytes);
    for (SizeType i = 0; i < filterbank.size(); ++i) {
        filterbank[i] = static_cast<uint8_t>((i * 17 + 3) % 256);
    }

    DDMTCPU ddmt_multi(f_min, f_max, nchans, tsamp, dms, /*nthreads=*/1,
                      /*nbits=*/8, /*kill_mask=*/{}, nbeams);
    const auto max_delay = *std::ranges::max_element(
        ddmt_multi.get_plan().get_container().delay_table);
    const auto nsamps_reduced = nsamps - max_delay;
    const auto dm_count       = dms.size();

    std::vector<int32_t> dmt_multi(nbeams * dm_count * nsamps_reduced, 0);
    ddmt_multi.execute_time_major(filterbank, nsamps, dmt_multi);

    DDMTCPU ddmt_single(f_min, f_max, nchans, tsamp, dms, /*nthreads=*/1,
                       /*nbits=*/8);
    for (SizeType ibeam = 0; ibeam < nbeams; ++ibeam) {
        std::vector<uint8_t> beam_filterbank(
            filterbank.begin() +
                static_cast<std::ptrdiff_t>(ibeam * nsamps * samp_bytes),
            filterbank.begin() +
                static_cast<std::ptrdiff_t>((ibeam + 1) * nsamps * samp_bytes));
        std::vector<int32_t> dmt_single(dm_count * nsamps_reduced, 0);
        ddmt_single.execute_time_major(beam_filterbank, nsamps, dmt_single);

        const auto* beam_out =
            dmt_multi.data() + (ibeam * dm_count * nsamps_reduced);
        for (SizeType i = 0; i < dmt_single.size(); ++i) {
            CHECK(beam_out[i] == dmt_single[i]);
        }
    }
}

TEST_CASE("DDMTCPU multi-beam streaming reproduces a monolithic call",
          "[ddmt][cpu]") {
    const float f_min      = 1000.0F;
    const float f_max      = 1500.0F;
    const SizeType nchans  = 8;
    const SizeType nsamps  = 200;
    const SizeType nbeams  = 2;
    const float tsamp      = 0.001F;
    const std::vector<float> dms = {0.0F, 3.0F, 7.0F};

    std::vector<float> waterfall(nbeams * nchans * nsamps);
    for (SizeType i = 0; i < waterfall.size(); ++i) {
        waterfall[i] = static_cast<float>(
            std::sin(0.1 * static_cast<double>(i % nsamps)) +
            (0.01 * static_cast<double>(i)));
    }

    DDMTCPU monolithic(f_min, f_max, nchans, tsamp, dms, /*nthreads=*/1,
                      /*nbits=*/32, /*kill_mask=*/{}, nbeams);
    const auto max_delay = *std::ranges::max_element(
        monolithic.get_plan().get_container().delay_table);
    const auto nsamps_reduced = nsamps - max_delay;
    std::vector<float> dmt_mono(nbeams * dms.size() * nsamps_reduced, 0.0F);
    monolithic.execute(waterfall, dmt_mono);

    DDMTCPU streamed(f_min, f_max, nchans, tsamp, dms, /*nthreads=*/1,
                    /*nbits=*/32, /*kill_mask=*/{}, nbeams);
    const SizeType split = 70;
    std::vector<float> chunk1(nbeams * nchans * split);
    std::vector<float> chunk2(nbeams * nchans * (nsamps - split));
    for (SizeType ibeam = 0; ibeam < nbeams; ++ibeam) {
        for (SizeType ichan = 0; ichan < nchans; ++ichan) {
            const auto row = (ibeam * nchans) + ichan;
            std::copy_n(&waterfall[(row * nsamps)], split,
                      &chunk1[row * split]);
            std::copy_n(&waterfall[(row * nsamps) + split], nsamps - split,
                      &chunk2[row * (nsamps - split)]);
        }
    }

    const auto n_out1 = streamed.get_output_nsamps(split);
    std::vector<float> dmt1(nbeams * dms.size() * n_out1, 0.0F);
    streamed.execute(chunk1, dmt1);

    const auto n_out2 = streamed.get_output_nsamps(nsamps - split);
    std::vector<float> dmt2(nbeams * dms.size() * n_out2, 0.0F);
    streamed.execute(chunk2, dmt2);

    REQUIRE(n_out1 + n_out2 == nsamps_reduced);

    std::vector<float> dmt_streamed(nbeams * dms.size() * nsamps_reduced, 0.0F);
    for (SizeType ibeam = 0; ibeam < nbeams; ++ibeam) {
        for (SizeType idm = 0; idm < dms.size(); ++idm) {
            const auto row = (ibeam * dms.size()) + idm;
            std::copy_n(&dmt1[row * n_out1], n_out1,
                      &dmt_streamed[(row * nsamps_reduced)]);
            std::copy_n(&dmt2[row * n_out2], n_out2,
                      &dmt_streamed[(row * nsamps_reduced) + n_out1]);
        }
    }

    for (SizeType i = 0; i < dmt_mono.size(); ++i) {
        CHECK(dmt_streamed[i] == Catch::Approx(dmt_mono[i]).margin(1e-5));
    }
}

TEST_CASE("DDMTCPU packed streaming matches monolithic execution across all bit widths",
          "[ddmt][cpu][packed][streaming]") {
    const float f_min     = 1000.0F;
    const float f_max     = 1500.0F;
    const SizeType nchans = 8;
    const float tsamp     = 0.001F;
    const std::vector<float> dms = {0.0F, 5.0F, 10.0F};

    const auto test_nbits = [&](SizeType nbits) {
        INFO("Testing nbits = " << nbits);
        const SizeType nsamps = 120;
        // 67 is not a multiple of 8, 4, or 2, guaranteeing unaligned sub-byte boundaries.
        const SizeType split  = 67;

        DDMTCPU ddmt_mono(f_min, f_max, nchans, tsamp, dms, /*nthreads=*/1, nbits);
        const auto max_delay      = *std::ranges::max_element(
            ddmt_mono.get_plan().get_container().delay_table);
        const auto nsamps_reduced = nsamps - max_delay;
        const auto dm_count       = dms.size();

        const auto mono_row_bytes   = utils::packed_row_bytes(nsamps, nbits);
        const auto chunk1_row_bytes = utils::packed_row_bytes(split, nbits);
        const auto chunk2_row_bytes = utils::packed_row_bytes(nsamps - split, nbits);

        std::vector<uint8_t> mono_waterfall(nchans * mono_row_bytes, 0);
        std::vector<uint8_t> chunk1(nchans * chunk1_row_bytes, 0);
        std::vector<uint8_t> chunk2(nchans * chunk2_row_bytes, 0);

        const auto max_val = utils::max_sample_value(nbits);
        for (SizeType ichan = 0; ichan < nchans; ++ichan) {
            auto* mono_row = mono_waterfall.data() + (ichan * mono_row_bytes);
            auto* c1_row   = chunk1.data() + (ichan * chunk1_row_bytes);
            auto* c2_row   = chunk2.data() + (ichan * chunk2_row_bytes);

            for (SizeType isamp = 0; isamp < nsamps; ++isamp) {
                const auto val = static_cast<uint32_t>(
                    ((ichan * 31) + (isamp * 17) + 7) & max_val);
                auto write_sample = [&]<unsigned NB>() {
                    utils::write_packed_sample<NB>(mono_row, isamp, val);
                    if (isamp < split) {
                        utils::write_packed_sample<NB>(c1_row, isamp, val);
                    } else {
                        utils::write_packed_sample<NB>(c2_row, isamp - split, val);
                    }
                };
                switch (nbits) {
                case 1: write_sample.template operator()<1>(); break;
                case 2: write_sample.template operator()<2>(); break;
                case 4: write_sample.template operator()<4>(); break;
                case 8: write_sample.template operator()<8>(); break;
                case 16: write_sample.template operator()<16>(); break;
                }
            }
        }

        std::vector<int32_t> dmt_mono(dm_count * nsamps_reduced, 0);
        ddmt_mono.execute(mono_waterfall, nsamps, dmt_mono);

        DDMTCPU ddmt_stream(f_min, f_max, nchans, tsamp, dms, /*nthreads=*/1, nbits);
        const auto n_out1 = ddmt_stream.get_output_nsamps(split);
        REQUIRE(n_out1 == (split > max_delay ? split - max_delay : 0));
        std::vector<int32_t> dmt1(dm_count * n_out1, 0);
        ddmt_stream.execute(chunk1, split, dmt1);

        const auto n_out2 = ddmt_stream.get_output_nsamps(nsamps - split);
        REQUIRE(n_out1 + n_out2 == nsamps_reduced);
        std::vector<int32_t> dmt2(dm_count * n_out2, 0);
        ddmt_stream.execute(chunk2, nsamps - split, dmt2);

        std::vector<int32_t> dmt_streamed(dm_count * nsamps_reduced, 0);
        for (SizeType idm = 0; idm < dm_count; ++idm) {
            std::copy_n(&dmt1[idm * n_out1], n_out1,
                      &dmt_streamed[idm * nsamps_reduced]);
            std::copy_n(&dmt2[idm * n_out2], n_out2,
                      &dmt_streamed[(idm * nsamps_reduced) + n_out1]);
        }

        for (SizeType i = 0; i < dmt_mono.size(); ++i) {
            CHECK(dmt_streamed[i] == dmt_mono[i]);
        }
    };

    for (SizeType nb : {1, 2, 4, 8, 16}) {
        test_nbits(nb);
    }
}

TEST_CASE("DDMTCPU packed sub-byte alignment stress test with multi-chunk streaming",
          "[ddmt][cpu][packed][streaming]") {
    const float f_min     = 800.0F;
    const float f_max     = 1200.0F;
    const SizeType nchans = 4;
    const float tsamp     = 0.001F;
    const std::vector<float> dms = {0.0F, 7.0F};

    // Use irregular, odd chunk sizes specifically designed to test unaligned bit offsets.
    const std::vector<SizeType> chunk_sizes = {37, 29, 43};
    SizeType total_samps = 0;
    for (auto sz : chunk_sizes) total_samps += sz;

    for (SizeType nbits : {1, 2, 4}) {
        INFO("Stress test nbits = " << nbits);
        DDMTCPU ddmt_mono(f_min, f_max, nchans, tsamp, dms, /*nthreads=*/1, nbits);
        const auto max_delay      = *std::ranges::max_element(
            ddmt_mono.get_plan().get_container().delay_table);
        const auto nsamps_reduced = total_samps - max_delay;
        const auto dm_count       = dms.size();

        const auto mono_row_bytes = utils::packed_row_bytes(total_samps, nbits);
        std::vector<uint8_t> mono_waterfall(nchans * mono_row_bytes, 0);

        const auto max_val = utils::max_sample_value(nbits);
        for (SizeType ichan = 0; ichan < nchans; ++ichan) {
            auto* row = mono_waterfall.data() + (ichan * mono_row_bytes);
            for (SizeType s = 0; s < total_samps; ++s) {
                const auto val = static_cast<uint32_t>(((ichan * 19) + (s * 11) + 3) & max_val);
                auto write_s = [&]<unsigned NB>() {
                    utils::write_packed_sample<NB>(row, s, val);
                };
                switch (nbits) {
                case 1: write_s.template operator()<1>(); break;
                case 2: write_s.template operator()<2>(); break;
                case 4: write_s.template operator()<4>(); break;
                }
            }
        }

        std::vector<int32_t> dmt_mono(dm_count * nsamps_reduced, 0);
        ddmt_mono.execute(mono_waterfall, total_samps, dmt_mono);

        // Stream chunk by chunk
        DDMTCPU ddmt_stream(f_min, f_max, nchans, tsamp, dms, /*nthreads=*/1, nbits);
        std::vector<int32_t> dmt_streamed(dm_count * nsamps_reduced, 0);
        SizeType curr_offset = 0;
        SizeType out_accum   = 0;

        for (auto chunk_len : chunk_sizes) {
            const auto chunk_row_bytes = utils::packed_row_bytes(chunk_len, nbits);
            std::vector<uint8_t> chunk(nchans * chunk_row_bytes, 0);
            for (SizeType ichan = 0; ichan < nchans; ++ichan) {
                const auto* src = mono_waterfall.data() + (ichan * mono_row_bytes);
                auto* dst       = chunk.data() + (ichan * chunk_row_bytes);
                auto copy_s = [&]<unsigned NB>() {
                    utils::copy_packed_samples<NB>(src, curr_offset, dst, 0, chunk_len);
                };
                switch (nbits) {
                case 1: copy_s.template operator()<1>(); break;
                case 2: copy_s.template operator()<2>(); break;
                case 4: copy_s.template operator()<4>(); break;
                }
            }

            const auto out_len = ddmt_stream.get_output_nsamps(chunk_len);
            std::vector<int32_t> chunk_dmt(dm_count * out_len, 0);
            ddmt_stream.execute(chunk, chunk_len, chunk_dmt);

            for (SizeType idm = 0; idm < dm_count; ++idm) {
                std::copy_n(&chunk_dmt[idm * out_len], out_len,
                          &dmt_streamed[(idm * nsamps_reduced) + out_accum]);
            }
            curr_offset += chunk_len;
            out_accum   += out_len;
        }

        REQUIRE(out_accum == nsamps_reduced);
        for (SizeType i = 0; i < dmt_mono.size(); ++i) {
            CHECK(dmt_streamed[i] == dmt_mono[i]);
        }
    }
}

TEST_CASE("DDMTCPU packed multi-beam streaming matches monolithic execution",
          "[ddmt][cpu][packed][streaming]") {
    const float f_min     = 1000.0F;
    const float f_max     = 1500.0F;
    const SizeType nchans = 4;
    const SizeType nbeams = 3;
    const SizeType nbits  = 2;
    const float tsamp     = 0.001F;
    const std::vector<float> dms = {0.0F, 4.0F, 8.0F};

    const SizeType nsamps = 95;
    const SizeType split  = 53;

    DDMTCPU ddmt_mono(f_min, f_max, nchans, tsamp, dms, /*nthreads=*/1, nbits, {}, nbeams);
    const auto max_delay      = *std::ranges::max_element(
        ddmt_mono.get_plan().get_container().delay_table);
    const auto nsamps_reduced = nsamps - max_delay;
    const auto dm_count       = dms.size();

    const auto mono_row_bytes   = utils::packed_row_bytes(nsamps, nbits);
    const auto chunk1_row_bytes = utils::packed_row_bytes(split, nbits);
    const auto chunk2_row_bytes = utils::packed_row_bytes(nsamps - split, nbits);

    const auto in_rows = nbeams * nchans;
    std::vector<uint8_t> mono_waterfall(in_rows * mono_row_bytes, 0);
    std::vector<uint8_t> chunk1(in_rows * chunk1_row_bytes, 0);
    std::vector<uint8_t> chunk2(in_rows * chunk2_row_bytes, 0);

    const auto max_val = utils::max_sample_value(nbits);
    for (SizeType row = 0; row < in_rows; ++row) {
        auto* mono_row = mono_waterfall.data() + (row * mono_row_bytes);
        auto* c1_row   = chunk1.data() + (row * chunk1_row_bytes);
        auto* c2_row   = chunk2.data() + (row * chunk2_row_bytes);
        for (SizeType isamp = 0; isamp < nsamps; ++isamp) {
            const auto val = static_cast<uint32_t>(((row * 23) + (isamp * 13) + 5) & max_val);
            utils::write_packed_sample<2>(mono_row, isamp, val);
            if (isamp < split) {
                utils::write_packed_sample<2>(c1_row, isamp, val);
            } else {
                utils::write_packed_sample<2>(c2_row, isamp - split, val);
            }
        }
    }

    std::vector<int32_t> dmt_mono(nbeams * dm_count * nsamps_reduced, 0);
    ddmt_mono.execute(mono_waterfall, nsamps, dmt_mono);

    DDMTCPU ddmt_stream(f_min, f_max, nchans, tsamp, dms, /*nthreads=*/1, nbits, {}, nbeams);
    const auto n_out1 = ddmt_stream.get_output_nsamps(split);
    std::vector<int32_t> dmt1(nbeams * dm_count * n_out1, 0);
    ddmt_stream.execute(chunk1, split, dmt1);

    const auto n_out2 = ddmt_stream.get_output_nsamps(nsamps - split);
    std::vector<int32_t> dmt2(nbeams * dm_count * n_out2, 0);
    ddmt_stream.execute(chunk2, nsamps - split, dmt2);

    REQUIRE(n_out1 + n_out2 == nsamps_reduced);

    std::vector<int32_t> dmt_streamed(nbeams * dm_count * nsamps_reduced, 0);
    for (SizeType ibeam = 0; ibeam < nbeams; ++ibeam) {
        for (SizeType idm = 0; idm < dm_count; ++idm) {
            const auto row = (ibeam * dm_count) + idm;
            std::copy_n(&dmt1[row * n_out1], n_out1,
                      &dmt_streamed[row * nsamps_reduced]);
            std::copy_n(&dmt2[row * n_out2], n_out2,
                      &dmt_streamed[(row * nsamps_reduced) + n_out1]);
        }
    }

    for (SizeType i = 0; i < dmt_mono.size(); ++i) {
        CHECK(dmt_streamed[i] == dmt_mono[i]);
    }
}

TEST_CASE("DDMTCPU packed history snapshot save, load, and validation",
          "[ddmt][cpu][packed][history]") {
    const float f_min     = 1000.0F;
    const float f_max     = 1500.0F;
    const SizeType nchans = 4;
    const SizeType nbits  = 2;
    const float tsamp     = 0.001F;
    const std::vector<float> dms = {0.0F, 5.0F};

    DDMTCPU ddmt1(f_min, f_max, nchans, tsamp, dms, /*nthreads=*/1, nbits);
    const auto max_delay = *std::ranges::max_element(
        ddmt1.get_plan().get_container().delay_table);
    const auto state_bytes = ddmt1.history_state_size();
    CHECK(state_bytes == nchans * utils::packed_row_bytes(max_delay, nbits));

    std::vector<uint8_t> hist_buf(state_bytes, 0);

    // 1. save before stream is warmed up returns false
    CHECK_FALSE(ddmt1.save_history(hist_buf));

    // 2. float overload on packed plan returns false
    std::vector<float> float_buf(state_bytes);
    CHECK_FALSE(ddmt1.save_history(float_buf));
    CHECK_FALSE(ddmt1.load_history(float_buf));

    // Warm up stream with chunk1
    const SizeType chunk1_len = 64;
    const auto c1_row_bytes   = utils::packed_row_bytes(chunk1_len, nbits);
    std::vector<uint8_t> chunk1(nchans * c1_row_bytes, 0);
    for (SizeType ichan = 0; ichan < nchans; ++ichan) {
        auto* row = chunk1.data() + (ichan * c1_row_bytes);
        for (SizeType s = 0; s < chunk1_len; ++s) {
            utils::write_packed_sample<2>(row, s, (s + ichan) % 4);
        }
    }
    const auto out1 = ddmt1.get_output_nsamps(chunk1_len);
    std::vector<int32_t> dmt1(dms.size() * out1, 0);
    ddmt1.execute(chunk1, chunk1_len, dmt1);

    // 3. Save history succeeds now that stream is warm
    CHECK(ddmt1.save_history(hist_buf));

    // Wrong size buffer fails
    std::vector<uint8_t> wrong_size_buf(state_bytes + 1, 0);
    CHECK_FALSE(ddmt1.save_history(wrong_size_buf));
    CHECK_FALSE(ddmt1.load_history(wrong_size_buf));

    // 4. Load history into fresh instance ddmt2
    DDMTCPU ddmt2(f_min, f_max, nchans, tsamp, dms, /*nthreads=*/1, nbits);
    CHECK(ddmt2.get_output_nsamps(32) == 32 - max_delay); // Cold state
    CHECK(ddmt2.load_history(hist_buf));
    CHECK(ddmt2.get_output_nsamps(32) == 32);              // Warm state!

    // 5. Feed chunk2 to both instances, verify exact match
    const SizeType chunk2_len = 48;
    const auto c2_row_bytes   = utils::packed_row_bytes(chunk2_len, nbits);
    std::vector<uint8_t> chunk2(nchans * c2_row_bytes, 0);
    for (SizeType ichan = 0; ichan < nchans; ++ichan) {
        auto* row = chunk2.data() + (ichan * c2_row_bytes);
        for (SizeType s = 0; s < chunk2_len; ++s) {
            utils::write_packed_sample<2>(row, s, (s * 3 + ichan) % 4);
        }
    }
    std::vector<int32_t> out_dmt1(dms.size() * chunk2_len, 0);
    std::vector<int32_t> out_dmt2(dms.size() * chunk2_len, 0);
    ddmt1.execute(chunk2, chunk2_len, out_dmt1);
    ddmt2.execute(chunk2, chunk2_len, out_dmt2);

    for (SizeType i = 0; i < out_dmt1.size(); ++i) {
        CHECK(out_dmt1[i] == out_dmt2[i]);
    }

    // 6. reset_history clears state
    ddmt2.reset_history();
    CHECK(ddmt2.get_output_nsamps(32) == 32 - max_delay);
    CHECK_FALSE(ddmt2.save_history(hist_buf));
}

} // namespace dmt

