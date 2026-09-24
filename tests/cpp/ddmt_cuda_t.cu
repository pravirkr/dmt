#include <algorithm>
#include <cstdint>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <cuda/std/span>
#include <thrust/device_vector.h>

#include "dmt/algorithms/ddmt.hpp"
#include "dmt/common/plans.hpp"
#include "test_helpers.hpp"

namespace dmt {

using algorithms::DDMTCUDA;
using plans::DDMTPlan;

TEST_CASE("DDMTCUDA construction and getters", "[ddmt_gpu][gpu]") {
    DDMTCUDA ddmt(test::kFMin, test::kFMax, 16, test::kTsamp, 20.0F, 5.0F,
                 0.0F);
    CHECK(ddmt.get_plan().get_f_min() == test::kFMin);
    CHECK(ddmt.get_plan().get_f_max() == test::kFMax);
    CHECK(ddmt.get_plan().get_nchans() == 16);
    CHECK(ddmt.get_plan().get_nbits() == 32);
    CHECK(ddmt.get_plan().get_dm_arr().size() > 1);
}

TEST_CASE("DDMTCUDA execute (host float) recovers a zero-DM ones waterfall",
          "[ddmt_gpu][gpu]") {
    const SizeType nchans = 8;
    const SizeType nsamps = 64;

    DDMTCUDA ddmt(test::kFMin, test::kFMax, nchans, test::kTsamp, 10.0F, 5.0F,
                 0.0F);
    const auto max_delay =
        *std::ranges::max_element(ddmt.get_plan().get_container().delay_table);
    REQUIRE(nsamps > max_delay);
    const auto nsamps_reduced = nsamps - max_delay;
    const auto dm_count       = ddmt.get_plan().get_dm_arr().size();

    std::vector<float> waterfall(nchans * nsamps, 1.0F);
    std::vector<float> dmt(dm_count * nsamps_reduced, 0.0F);
    ddmt.execute(waterfall, dmt);

    std::vector<float> dm0(
        dmt.begin(), dmt.begin() + static_cast<std::ptrdiff_t>(nsamps_reduced));
    REQUIRE_THAT(dm0, Catch::Matchers::Equals(std::vector<float>(
                          nsamps_reduced, static_cast<float>(nchans))));
}

TEST_CASE("DDMTCUDA execute (device float span) matches the host overload",
          "[ddmt_gpu][gpu]") {
    const SizeType nchans = 16;
    const SizeType nsamps = 128;

    DDMTCUDA ddmt(test::kFMin, test::kFMax, nchans, test::kTsamp,
                 std::vector<float>{0.0F, 5.0F, 10.0F});
    const auto waterfall = test::sequential_waterfall(nchans, nsamps);
    const auto max_delay =
        *std::ranges::max_element(ddmt.get_plan().get_container().delay_table);
    const auto nsamps_reduced = nsamps - max_delay;
    const auto dm_count       = ddmt.get_plan().get_dm_arr().size();

    std::vector<float> dmt_host(dm_count * nsamps_reduced, 0.0F);
    ddmt.execute(waterfall, dmt_host);
    ddmt.reset_history();

    thrust::device_vector<float> d_waterfall(waterfall.begin(), waterfall.end());
    thrust::device_vector<float> d_dmt(dm_count * nsamps_reduced, 0.0F);
    ddmt.execute(cuda::std::span<const float>(
                    thrust::raw_pointer_cast(d_waterfall.data()),
                    d_waterfall.size()),
                cuda::std::span<float>(thrust::raw_pointer_cast(d_dmt.data()),
                                      d_dmt.size()));
    std::vector<float> dmt_device(d_dmt.size());
    thrust::copy(d_dmt.begin(), d_dmt.end(), dmt_device.begin());
    cudaDeviceSynchronize();

    test::require_approx(dmt_device, dmt_host);
}

TEST_CASE("DDMTCUDA packed-integer execute matches the float path",
          "[ddmt_gpu][gpu]") {
    const SizeType nchans = 16;
    const SizeType nsamps = 96;
    const std::vector<float> dms = {0.0F, 5.0F, 12.0F};

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

    DDMTCUDA ddmt_f(test::kFMin, test::kFMax, nchans, test::kTsamp, dms);
    DDMTCUDA ddmt_i(test::kFMin, test::kFMax, nchans, test::kTsamp, dms,
                   /*device_id=*/0, /*nbits=*/8);
    CHECK(ddmt_i.get_plan().get_nbits() == 8);

    const auto max_delay = *std::ranges::max_element(
        ddmt_f.get_plan().get_container().delay_table);
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

TEST_CASE("DDMTCUDA kill_mask excludes masked channels from the sum",
          "[ddmt_gpu][gpu]") {
    const SizeType nchans = 8;
    const SizeType nsamps = 64;

    std::vector<uint8_t> kill_mask(nchans, 1);
    kill_mask[0] = 0;

    DDMTCUDA ddmt(test::kFMin, test::kFMax, nchans, test::kTsamp, 10.0F, 5.0F,
                 0.0F, /*device_id=*/0, /*nbits=*/32, kill_mask);
    const auto max_delay =
        *std::ranges::max_element(ddmt.get_plan().get_container().delay_table);
    const auto nsamps_reduced = nsamps - max_delay;
    const auto dm_count       = ddmt.get_plan().get_dm_arr().size();

    std::vector<float> waterfall(nchans * nsamps, 1.0F);
    std::vector<float> dmt(dm_count * nsamps_reduced, 0.0F);
    ddmt.execute(waterfall, dmt);

    std::vector<float> dm0(
        dmt.begin(), dmt.begin() + static_cast<std::ptrdiff_t>(nsamps_reduced));
    REQUIRE_THAT(dm0, Catch::Matchers::Equals(std::vector<float>(
                          nsamps_reduced, static_cast<float>(nchans - 1))));
}

TEST_CASE("DDMTCUDA with LevinConfig", "[ddmt_gpu][gpu]") {
    const SizeType nchans = 16;
    plans::LevinConfig levin{
        .dm_start    = 0.0F,
        .dm_end      = 30.0F,
        .pulse_width = 0.001F,
        .tol         = 1.25F,
    };

    DDMTCUDA ddmt(test::kFMin, test::kFMax, nchans, test::kTsamp, levin);
    CHECK(ddmt.get_plan().get_dm_arr().size() > 1);
    CHECK(ddmt.get_plan().get_dm_arr().front() == 0.0F);
    CHECK(ddmt.get_plan().get_dm_arr().back() >= levin.dm_end);

    plans::DDMTPlan plan(test::kFMin, test::kFMax, nchans, test::kTsamp, levin);
    DDMTCUDA ddmt_from_plan(plan);
    CHECK(ddmt_from_plan.get_plan().get_dm_arr() == plan.get_dm_arr());
}

TEST_CASE("DDMTCUDA execute_time_major matches channel-major packed", "[ddmt_gpu][gpu]") {
    const SizeType nchans = 16;
    const SizeType nsamps = 64;
    const std::vector<float> dms = {0.0F, 5.0F, 12.0F};

    const auto test_bits = [&](unsigned nbits) {
        DYNAMIC_SECTION("nbits = " << nbits) {
            plans::DDMTPlan plan(test::kFMin, test::kFMax, nchans, test::kTsamp, dms, false, nbits);
            DDMTCUDA ddmt(plan);

            const auto max_delay = *std::ranges::max_element(plan.get_container().delay_table);
            const auto nsamps_reduced = nsamps - max_delay;
            const auto dm_count       = dms.size();

            const auto mask = (1ULL << nbits) - 1;
            std::vector<uint64_t> raw_data(nchans * nsamps);
            for (size_t i = 0; i < raw_data.size(); ++i) {
                raw_data[i] = (i * 37 + 11) & mask;
            }

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

TEST_CASE("DDMTCUDA streaming history across chunks matches monolithic execute",
          "[ddmt_gpu][gpu]") {
    const SizeType nchans = 8;
    const std::vector<float> dms = {0.0F, 4.0F};

    DDMTCUDA ddmt_mono(test::kFMin, test::kFMax, nchans, test::kTsamp, dms);
    DDMTCUDA ddmt_stream(test::kFMin, test::kFMax, nchans, test::kTsamp, dms);

    const auto max_delay =
        *std::ranges::max_element(ddmt_mono.get_plan().get_container().delay_table);
    const SizeType total_nsamps = 128;
    const auto waterfall = test::sequential_waterfall(nchans, total_nsamps);

    const auto nsamps_reduced = total_nsamps - max_delay;
    std::vector<float> dmt_expected(dms.size() * nsamps_reduced, 0.0F);
    ddmt_mono.execute(waterfall, dmt_expected);

    const SizeType split = 70;
    std::vector<float> block1(nchans * split);
    std::vector<float> block2(nchans * (total_nsamps - split));
    for (SizeType ichan = 0; ichan < nchans; ++ichan) {
        std::copy_n(&waterfall[ichan * total_nsamps], split,
                    &block1[ichan * split]);
        std::copy_n(&waterfall[(ichan * total_nsamps) + split],
                    total_nsamps - split,
                    &block2[ichan * (total_nsamps - split)]);
    }

    const auto n_out1 = ddmt_stream.get_output_nsamps(split);
    std::vector<float> dmt1(dms.size() * n_out1);
    ddmt_stream.execute(block1, dmt1);

    const auto n_out2 = ddmt_stream.get_output_nsamps(total_nsamps - split);
    std::vector<float> dmt2(dms.size() * n_out2);
    ddmt_stream.execute(block2, dmt2);

    CHECK(n_out1 + n_out2 == nsamps_reduced);

    std::vector<float> dmt_streamed(dms.size() * nsamps_reduced);
    for (SizeType idm = 0; idm < dms.size(); ++idm) {
        std::copy_n(&dmt1[idm * n_out1], n_out1,
                    &dmt_streamed[idm * nsamps_reduced]);
        std::copy_n(&dmt2[idm * n_out2], n_out2,
                    &dmt_streamed[(idm * nsamps_reduced) + n_out1]);
    }

    test::require_approx(dmt_streamed, dmt_expected);

    // Test save and load history
    const auto hist_size = ddmt_stream.history_state_size();
    std::vector<float> saved_history(hist_size);
    ddmt_stream.save_history(saved_history);

    ddmt_stream.reset_history();
    CHECK(ddmt_stream.get_output_nsamps(split) == split - max_delay);

    ddmt_stream.load_history(saved_history);
    CHECK(ddmt_stream.get_output_nsamps(split) == split);
}

TEST_CASE("DDMTCUDA multi-beam float execute matches per-beam single-beam calls",
          "[ddmt_gpu][gpu]") {
    const SizeType nchans = 8;
    const SizeType nsamps = 64;
    const SizeType nbeams = 3;
    const std::vector<float> dms = {0.0F, 5.0F, 10.0F};

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

    DDMTCUDA ddmt_multi(test::kFMin, test::kFMax, nchans, test::kTsamp, dms,
                       /*device_id=*/0, /*nbits=*/32, /*kill_mask=*/{}, nbeams);
    CHECK(ddmt_multi.get_nbeams() == nbeams);
    const auto max_delay = *std::ranges::max_element(
        ddmt_multi.get_plan().get_container().delay_table);
    const auto nsamps_reduced = nsamps - max_delay;
    const auto dm_count       = dms.size();

    std::vector<float> dmt_multi(nbeams * dm_count * nsamps_reduced, 0.0F);
    ddmt_multi.execute(waterfall, dmt_multi);

    // Fresh single-beam instance per beam: execute(float) retains streaming
    // history across calls by design, so reusing one instance across beams
    // would contaminate beam N's reference with beam N-1's tail.
    for (SizeType ibeam = 0; ibeam < nbeams; ++ibeam) {
        DDMTCUDA ddmt_single(test::kFMin, test::kFMax, nchans, test::kTsamp, dms);
        std::vector<float> beam_waterfall(
            waterfall.begin() + static_cast<std::ptrdiff_t>(ibeam * nchans * nsamps),
            waterfall.begin() +
                static_cast<std::ptrdiff_t>((ibeam + 1) * nchans * nsamps));
        std::vector<float> dmt_single(dm_count * nsamps_reduced, 0.0F);
        ddmt_single.execute(beam_waterfall, dmt_single);

        const auto* beam_out = dmt_multi.data() + (ibeam * dm_count * nsamps_reduced);
        for (SizeType i = 0; i < dmt_single.size(); ++i) {
            CHECK(beam_out[i] == Catch::Approx(dmt_single[i]));
        }
    }
}

} // namespace dmt

