#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <thrust/device_vector.h>

#include "dmt/algorithms/ddmt.hpp"
#include "dmt/common/plans.hpp"

#include "dmt/gpu_compat.cuh"
#include "dmt/gpu_utils.cuh"

#include "ddmt_estimate_cases.hpp"
#include "test_helpers.hpp"

namespace dmt {

using algorithms::DDMT;
using plans::DDMTPlan;

TEST_CASE("DDMT construction and getters", "[ddmt_gpu][gpu]") {
    DDMT ddmt(test::kFMin, test::kFMax, 16, test::kTsamp, 20.0F, 5.0F, 0.0F,
              test::gpu_exec());
    CHECK(ddmt.get_plan().get_f_min() == test::kFMin);
    CHECK(ddmt.get_plan().get_f_max() == test::kFMax);
    CHECK(ddmt.get_plan().get_nchans() == 16);
    CHECK(ddmt.get_plan().get_nbits() == 32);
    CHECK(ddmt.get_plan().get_dm_arr().size() > 1);
}

TEST_CASE("DDMT execute (host float) recovers a zero-DM ones waterfall",
          "[ddmt_gpu][gpu]") {
    const SizeType nchans = 8;
    const SizeType nsamps = 64;

    DDMT ddmt(test::kFMin, test::kFMax, nchans, test::kTsamp, 10.0F, 5.0F, 0.0F,
              test::gpu_exec());
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

TEST_CASE("DDMT (gpu) packed-integer execute matches the float path",
          "[ddmt_gpu][gpu]") {
    const SizeType nchans        = 16;
    const SizeType nsamps        = 96;
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

    DDMT ddmt_f(test::kFMin, test::kFMax, nchans, test::kTsamp, dms,
                test::gpu_exec());
    DDMT ddmt_i(test::kFMin, test::kFMax, nchans, test::kTsamp, dms,
                test::gpu_exec(), /*nbits=*/8);
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

TEST_CASE("DDMT with LevinConfig", "[ddmt_gpu][gpu]") {
    const SizeType nchans = 16;
    plans::LevinConfig levin{
        .dm_start    = 0.0F,
        .dm_end      = 30.0F,
        .pulse_width = 0.001F,
        .tol         = 1.25F,
    };

    DDMT ddmt(test::kFMin, test::kFMax, nchans, test::kTsamp, levin,
              test::gpu_exec());
    CHECK(ddmt.get_plan().get_dm_arr().size() > 1);
    CHECK(ddmt.get_plan().get_dm_arr().front() == 0.0F);
    CHECK(ddmt.get_plan().get_dm_arr().back() >= levin.dm_end);

    plans::DDMTPlan plan(test::kFMin, test::kFMax, nchans, test::kTsamp, levin);
    DDMT ddmt_from_plan(plan, test::gpu_exec());
    CHECK(ddmt_from_plan.get_plan().get_dm_arr() == plan.get_dm_arr());
}

namespace {

// A device-memory call of `nsamps` zero samples: channel-major or (packed)
// time-major.
template <typename Engine>
void device_call(Engine& eng,
                 const plans::DDMTPlan& plan,
                 SizeType nbeams,
                 SizeType nsamps,
                 SizeType out,
                 bool time_major) {
    const auto nbits  = plan.get_nbits();
    const auto nchans = plan.get_nchans();
    const auto rows   = nbeams * nchans;
    const auto nout   = nbeams * plan.get_dm_arr().size() * out;
    if (nbits == 32) {
        const thrust::device_vector<float> wf(rows * nsamps, 0.0F);
        thrust::device_vector<float> dmt(nout);
        eng.execute(DeviceSpan<const float>(thrust::raw_pointer_cast(wf.data()),
                                            wf.size()),
                    DeviceSpan<float>(thrust::raw_pointer_cast(dmt.data()),
                                      dmt.size()),
                    Stream{});
    } else {
        const auto bytes =
            time_major
                ? nbeams * nsamps *
                      bit_pack_utils::packed_row_bytes(nchans, nbits)
                : rows * bit_pack_utils::packed_row_bytes(nsamps, nbits);
        const thrust::device_vector<uint8_t> wf(bytes, 0);
        thrust::device_vector<int32_t> dmt(nout);
        const DeviceSpan<const uint8_t> in(thrust::raw_pointer_cast(wf.data()),
                                           wf.size());
        const DeviceSpan<int32_t> o(thrust::raw_pointer_cast(dmt.data()),
                                    dmt.size());
        if (time_major) {
            eng.execute_time_major(in, nsamps, o, Stream{});
        } else {
            eng.execute(in, nsamps, o, Stream{});
        }
    }
    cudaDeviceSynchronize();
}

} // namespace

TEST_CASE("DDMT and SDMT (gpu) estimate_memory equal the instance",
          "[ddmt_gpu][gpu][budget]") {
    if (!test::gpu_device_available()) {
        SKIP("no GPU device");
    }
    const auto exec = test::gpu_exec();
    ddmt_estimate::check_host(exec);
    ddmt_estimate::check_sdmt_differs(exec);
    for (const bool sdmt : {false, true}) {
        for (const auto& c : ddmt_estimate::cases()) {
            INFO((sdmt ? "SDMT " : "DDMT ") << c.name << " (device memory)");
            const auto run_dev = [&](const algorithms::DDMTMemoryEstimate& e,
                                     auto make_eng) {
                for (const bool tm : {false, true}) {
                    if (tm && c.plan.get_nbits() == 32) {
                        continue;
                    }
                    auto eng = make_eng();
                    ddmt_estimate::check_stream(
                        *eng, c.nsamps, e, tm ? e.device_time_major_staging : 0,
                        0, [&](SizeType out) {
                            device_call(*eng, c.plan, c.nbeams, c.nsamps, out,
                                        tm);
                        });
                }
            };
            if (sdmt) {
                const auto e = ddmt_estimate::make_forced_sdmt_gpu(c, exec, true)
                                   ->estimate(c.nsamps);
                run_dev(e, [&] {
                    return ddmt_estimate::make_forced_sdmt_gpu(c, exec, false);
                });
            } else {
                const auto e = ddmt_estimate::estimate(false, c, exec);
                run_dev(e, [&] { return ddmt_estimate::make(false, c, exec); });
            }
        }
    }
    // The estimate allocates nothing: one history copy alone (2^25 beams,
    // about 50 GiB) is beyond the device; the estimate-only engine also
    // checks it holds no device or pinned memory.
    const auto& props = gpu_utils::device_properties(exec.device);
    const auto big    = ddmt_estimate::many_beams(SizeType{1} << 25);
    for (const bool sdmt : {false, true}) {
        const auto e = ddmt_estimate::estimate(sdmt, big, exec);
        CHECK(e.usage.history / 2 > props.totalGlobalMem);
    }
}

TEST_CASE("DDMT (gpu) device-memory calls reject 32-bit sample overflow",
          "[ddmt_gpu][gpu]") {
    if (!test::gpu_device_available()) {
        SKIP("no GPU device");
    }
    constexpr auto kMaxInt =
        static_cast<SizeType>(std::numeric_limits<int>::max());
    const auto exec = test::gpu_exec();
    const Device dev{exec.backend, exec.device};
    constexpr SizeType nchans = 8;

    SECTION("float device execute: nsamps > INT_MAX and history + nsamps > INT_MAX") {
        DDMT ddmt(test::kFMin, test::kFMax, nchans, test::kTsamp, 100.0F,
                  50.0F, 0.0F, exec);
        thrust::device_vector<float> dummy(nchans * 16, 0.0F);
        auto* ptr = thrust::raw_pointer_cast(dummy.data());

        // Cold stream: nsamps alone exceeds INT_MAX.
        const SizeType over = kMaxInt + 1;
        CHECK_THROWS_AS(
            ddmt.execute(DeviceSpan<const float>(ptr, nchans * over, dev),
                         DeviceSpan<float>(ptr, ddmt.get_output_nsamps(over) *
                                                    ddmt.get_plan().get_dm_arr().size(),
                                           dev)),
            std::invalid_argument);

        // Warm the stream with a real small block so m_history_len > 0; then
        // nsamps == INT_MAX fits in int32 alone, but history + nsamps overflows.
        const SizeType warm_n = 16;
        const thrust::device_vector<float> wf_warm(nchans * warm_n, 0.0F);
        thrust::device_vector<float> out_warm(
            std::max<SizeType>(1, ddmt.get_output_nsamps(warm_n) *
                                      ddmt.get_plan().get_dm_arr().size()),
            0.0F);
        ddmt.execute(
            DeviceSpan<const float>(thrust::raw_pointer_cast(wf_warm.data()),
                                    wf_warm.size(), dev),
            DeviceSpan<float>(thrust::raw_pointer_cast(out_warm.data()),
                              ddmt.get_output_nsamps(warm_n) *
                                  ddmt.get_plan().get_dm_arr().size(),
                              dev));
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

        const SizeType at_max = kMaxInt;
        CHECK_THROWS_AS(
            ddmt.execute(
                DeviceSpan<const float>(ptr, nchans * at_max, dev),
                DeviceSpan<float>(ptr,
                                  ddmt.get_output_nsamps(at_max) *
                                      ddmt.get_plan().get_dm_arr().size(),
                                  dev)),
            std::invalid_argument);
    }

    SECTION("packed channel-major and time-major device execute") {
        DDMT ddmt(test::kFMin, test::kFMax, nchans, test::kTsamp, 100.0F,
                  50.0F, 0.0F, exec, /*nbits=*/8);
        thrust::device_vector<uint8_t> d_in(1, 0);
        thrust::device_vector<int32_t> d_out(1, 0);
        auto* in_ptr  = thrust::raw_pointer_cast(d_in.data());
        auto* out_ptr = thrust::raw_pointer_cast(d_out.data());
        const SizeType over = kMaxInt + 1;
        const SizeType out_elems =
            ddmt.get_output_nsamps(over) * ddmt.get_plan().get_dm_arr().size();
        const auto row_bytes = bit_pack_utils::packed_row_bytes(over, 8);
        const auto samp_bytes = bit_pack_utils::packed_row_bytes(nchans, 8);

        CHECK_THROWS_AS(
            ddmt.execute(DeviceSpan<const uint8_t>(in_ptr, nchans * row_bytes, dev),
                         over, DeviceSpan<int32_t>(out_ptr, out_elems, dev)),
            std::invalid_argument);
        CHECK_THROWS_AS(
            ddmt.execute_time_major(
                DeviceSpan<const uint8_t>(in_ptr, over * samp_bytes, dev), over,
                DeviceSpan<int32_t>(out_ptr, out_elems, dev)),
            std::invalid_argument);
    }
}

} // namespace dmt
