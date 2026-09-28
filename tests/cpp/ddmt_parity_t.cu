#include <cmath>
#include <cstdint>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include "dmt/algorithms/ddmt.hpp"
#include "dmt/utils/simulate.hpp"
#include "test_helpers.hpp"

namespace dmt {

using algorithms::DDMT;

TEST_CASE("parity: DDMT (gpu) execute matches CPU on an injected FRB",
          "[ddmt][gpu][parity]") {
    const SizeType nchans = 32;
    const SizeType nsamps = 256;
    const SizeType dt     = 12;
    const float toa       = 80.0F;

    const auto [waterfall, n_disp] = utils::generate_pure_frb(
        nchans, nsamps, test::kFMin, test::kFMax, dt, toa, 1.0F);
    REQUIRE(n_disp > 0);

    const float dm = static_cast<float>(dt) * test::kTsamp /
                     (kDispConst * (std::pow(test::kFMin, kDispCoeff) -
                                    std::pow(test::kFMax, kDispCoeff)));
    const std::vector<float> dms = {0.0F, dm};

    DDMT cpu(test::kFMin, test::kFMax, nchans, test::kTsamp, dms);
    DDMT gpu(test::kFMin, test::kFMax, nchans, test::kTsamp, dms,
             test::gpu_exec());
    const auto max_delay =
        *std::ranges::max_element(cpu.get_plan().get_container().delay_table);
    const auto nsamps_reduced = nsamps - max_delay;
    const auto dm_count       = dms.size();

    std::vector<float> dmt_cpu(dm_count * nsamps_reduced, 0.0F);
    std::vector<float> dmt_gpu(dm_count * nsamps_reduced, 0.0F);
    cpu.execute(waterfall, dmt_cpu);
    gpu.execute(waterfall, dmt_gpu);

    test::require_approx(dmt_gpu, dmt_cpu);
}

TEST_CASE("parity: DDMT (gpu) packed-integer execute matches CPU",
          "[ddmt][gpu][parity]") {
    const SizeType nchans        = 16;
    const SizeType nsamps        = 100;
    const std::vector<float> dms = {0.0F, 3.0F, 9.0F};

    std::vector<uint8_t> waterfall_u8(nchans * nsamps);
    for (SizeType ichan = 0; ichan < nchans; ++ichan) {
        for (SizeType isamp = 0; isamp < nsamps; ++isamp) {
            waterfall_u8[(ichan * nsamps) + isamp] =
                static_cast<uint8_t>(((ichan * 11) + (isamp * 5)) % 256);
        }
    }

    DDMT cpu(test::kFMin, test::kFMax, nchans, test::kTsamp, dms, Exec::cpu(1),
             /*nbits=*/8);
    DDMT gpu(test::kFMin, test::kFMax, nchans, test::kTsamp, dms,
             test::gpu_exec(),
             /*nbits=*/8);
    const auto max_delay =
        *std::ranges::max_element(cpu.get_plan().get_container().delay_table);
    const auto nsamps_reduced = nsamps - max_delay;
    const auto dm_count       = dms.size();

    std::vector<int32_t> dmt_cpu(dm_count * nsamps_reduced, 0);
    std::vector<int32_t> dmt_gpu(dm_count * nsamps_reduced, 0);
    cpu.execute(waterfall_u8, nsamps, dmt_cpu);
    gpu.execute(waterfall_u8, nsamps, dmt_gpu);

    REQUIRE_THAT(dmt_gpu, Catch::Matchers::Equals(dmt_cpu));
}

TEST_CASE("parity: DDMT (gpu) spans multiple internal gulps and still matches "
          "CPU",
          "[ddmt][gpu][parity]") {
    // Deliberately larger than the CUDA implementation's internal gulp size
    // (65536 output samples/gulp), to exercise the double-buffered
    // multi-gulp pipeline (H2D/kernel/D2H overlap, the strided
    // cudaMemcpy2DAsync host<->device staging, and the output copy-out) --
    // not just the common single-gulp path.
    const SizeType nchans        = 4;
    const SizeType dm_max        = 5.0F;
    const std::vector<float> dms = {0.0F, dm_max};

    DDMT probe(test::kFMin, test::kFMax, nchans, test::kTsamp, dms);
    const auto max_delay =
        *std::ranges::max_element(probe.get_plan().get_container().delay_table);
    const SizeType nsamps_reduced_target = 150000; // > 2 gulps
    const SizeType nsamps                = nsamps_reduced_target + max_delay;

    const auto waterfall = test::sequential_waterfall(nchans, nsamps);

    DDMT cpu(test::kFMin, test::kFMax, nchans, test::kTsamp, dms);
    DDMT gpu(test::kFMin, test::kFMax, nchans, test::kTsamp, dms,
             test::gpu_exec());
    const auto nsamps_reduced = nsamps - max_delay;
    const auto dm_count       = dms.size();

    std::vector<float> dmt_cpu(dm_count * nsamps_reduced, 0.0F);
    std::vector<float> dmt_gpu(dm_count * nsamps_reduced, 0.0F);
    cpu.execute(waterfall, dmt_cpu);
    gpu.execute(waterfall, dmt_gpu);

    test::require_approx(dmt_gpu, dmt_cpu);
}

TEST_CASE("parity: DDMT (gpu) matches CPU with LevinConfig",
          "[ddmt][gpu][parity]") {
    const SizeType nchans = 16;
    const SizeType nsamps = 128;
    plans::LevinConfig levin{
        .dm_start    = 0.0F,
        .dm_end      = 25.0F,
        .pulse_width = 0.001F,
        .tol         = 1.25F,
    };

    DDMT cpu(test::kFMin, test::kFMax, nchans, test::kTsamp, levin);
    DDMT gpu(test::kFMin, test::kFMax, nchans, test::kTsamp, levin,
             test::gpu_exec());

    const auto waterfall = test::sequential_waterfall(nchans, nsamps);
    const auto max_delay =
        *std::ranges::max_element(cpu.get_plan().get_container().delay_table);
    const auto nsamps_reduced = nsamps - max_delay;
    const auto dm_count       = cpu.get_plan().get_dm_arr().size();

    std::vector<float> dmt_cpu(dm_count * nsamps_reduced, 0.0F);
    std::vector<float> dmt_gpu(dm_count * nsamps_reduced, 0.0F);
    cpu.execute(waterfall, dmt_cpu);
    gpu.execute(waterfall, dmt_gpu);

    test::require_approx(dmt_gpu, dmt_cpu);
}

TEST_CASE("parity: DDMT (gpu) execute_time_major matches CPU across bits",
          "[ddmt][gpu][parity]") {
    const SizeType nchans        = 16;
    const SizeType nsamps        = 64;
    const std::vector<float> dms = {0.0F, 5.0F, 12.0F};

    const auto test_bits = [&](unsigned nbits) {
        DYNAMIC_SECTION("nbits = " << nbits) {
            plans::DDMTPlan plan(test::kFMin, test::kFMax, nchans, test::kTsamp,
                                 dms, nbits);
            DDMT cpu(plan);
            DDMT gpu(plan, test::gpu_exec());

            const auto max_delay =
                *std::ranges::max_element(plan.get_container().delay_table);
            const auto nsamps_reduced = nsamps - max_delay;
            const auto dm_count       = dms.size();

            const auto mask = (1ULL << nbits) - 1;
            std::vector<uint64_t> raw_data(nchans * nsamps);
            for (size_t i = 0; i < raw_data.size(); ++i) {
                raw_data[i] = (i * 37 + 11) & mask;
            }

            const auto time_samp_bytes = (nchans * nbits + 7) / 8;
            std::vector<uint8_t> time_major(nsamps * time_samp_bytes, 0);
            for (SizeType s = 0; s < nsamps; ++s) {
                for (SizeType c = 0; c < nchans; ++c) {
                    const auto val        = raw_data[(c * nsamps) + s];
                    const auto bit_offset = c * nbits;
                    const auto byte_idx   = bit_offset / 8;
                    const auto bit_sub    = bit_offset % 8;
                    if (nbits == 8) {
                        time_major[(s * time_samp_bytes) + c] =
                            static_cast<uint8_t>(val);
                    } else if (nbits == 16) {
                        auto* ptr = reinterpret_cast<uint16_t*>(
                            &time_major[s * time_samp_bytes]);
                        ptr[c] = static_cast<uint16_t>(val);
                    } else {
                        time_major[(s * time_samp_bytes) + byte_idx] |=
                            static_cast<uint8_t>(val << bit_sub);
                    }
                }
            }

            std::vector<int32_t> out_cpu(dm_count * nsamps_reduced, 0);
            std::vector<int32_t> out_gpu(dm_count * nsamps_reduced, 0);

            cpu.execute_time_major(time_major, nsamps, out_cpu);
            gpu.execute_time_major(time_major, nsamps, out_gpu);

            REQUIRE_THAT(out_gpu, Catch::Matchers::Equals(out_cpu));
        }
    };

    test_bits(1);
    test_bits(2);
    test_bits(4);
    test_bits(8);
    test_bits(16);
}

TEST_CASE("parity: DDMT (gpu) multi-beam float execute matches CPU",
          "[ddmt][gpu][parity]") {
    const SizeType nchans        = 16;
    const SizeType nsamps        = 96;
    const SizeType nbeams        = 3;
    const std::vector<float> dms = {0.0F, 4.0F, 9.0F};

    std::vector<float> waterfall(nbeams * nchans * nsamps);
    for (SizeType i = 0; i < waterfall.size(); ++i) {
        waterfall[i] = static_cast<float>((i % 37) + 1);
    }

    DDMT cpu(test::kFMin, test::kFMax, nchans, test::kTsamp, dms, Exec::cpu(1),
             /*nbits=*/32, /*kill_mask=*/{}, nbeams);
    DDMT gpu(test::kFMin, test::kFMax, nchans, test::kTsamp, dms,
             test::gpu_exec(),
             /*nbits=*/32, /*kill_mask=*/{}, nbeams);
    const auto max_delay =
        *std::ranges::max_element(cpu.get_plan().get_container().delay_table);
    const auto nsamps_reduced = nsamps - max_delay;
    const auto dm_count       = dms.size();

    std::vector<float> dmt_cpu(nbeams * dm_count * nsamps_reduced, 0.0F);
    std::vector<float> dmt_gpu(nbeams * dm_count * nsamps_reduced, 0.0F);
    cpu.execute(waterfall, dmt_cpu);
    gpu.execute(waterfall, dmt_gpu);

    test::require_approx(dmt_gpu, dmt_cpu);
}

TEST_CASE(
    "parity: DDMT (gpu) multi-beam spans multiple internal gulps and still "
    "matches CPU",
    "[ddmt][gpu][parity]") {
    const SizeType nchans        = 4;
    const SizeType nbeams        = 2;
    const std::vector<float> dms = {0.0F, 5.0F};

    DDMT probe(test::kFMin, test::kFMax, nchans, test::kTsamp, dms,
               Exec::cpu(1), 32, {}, nbeams);
    const auto max_delay =
        *std::ranges::max_element(probe.get_plan().get_container().delay_table);
    const SizeType nsamps_reduced_target = 150000; // > 2 gulps
    const SizeType nsamps                = nsamps_reduced_target + max_delay;

    std::vector<float> waterfall(nbeams * nchans * nsamps);
    for (SizeType i = 0; i < waterfall.size(); ++i) {
        waterfall[i] = static_cast<float>((i % 43) + 1);
    }

    DDMT cpu(test::kFMin, test::kFMax, nchans, test::kTsamp, dms, Exec::cpu(1),
             32, {}, nbeams);
    DDMT gpu(test::kFMin, test::kFMax, nchans, test::kTsamp, dms,
             test::gpu_exec(), 32, {}, nbeams);
    const auto nsamps_reduced = nsamps - max_delay;
    const auto dm_count       = dms.size();

    std::vector<float> dmt_cpu(nbeams * dm_count * nsamps_reduced, 0.0F);
    std::vector<float> dmt_gpu(nbeams * dm_count * nsamps_reduced, 0.0F);
    cpu.execute(waterfall, dmt_cpu);
    gpu.execute(waterfall, dmt_gpu);

    test::require_approx(dmt_gpu, dmt_cpu);
}

TEST_CASE("parity: DDMT (gpu) multi-beam packed-integer execute matches CPU",
          "[ddmt][gpu][parity]") {
    const SizeType nchans        = 8;
    const SizeType nsamps        = 64;
    const SizeType nbeams        = 2;
    const std::vector<float> dms = {0.0F, 6.0F};

    std::vector<uint8_t> waterfall(nbeams * nchans * nsamps);
    for (SizeType i = 0; i < waterfall.size(); ++i) {
        waterfall[i] = static_cast<uint8_t>((i * 13 + 5) % 256);
    }

    DDMT cpu(test::kFMin, test::kFMax, nchans, test::kTsamp, dms, Exec::cpu(1),
             8, {}, nbeams);
    DDMT gpu(test::kFMin, test::kFMax, nchans, test::kTsamp, dms,
             test::gpu_exec(), 8, {}, nbeams);
    const auto max_delay =
        *std::ranges::max_element(cpu.get_plan().get_container().delay_table);
    const auto nsamps_reduced = nsamps - max_delay;
    const auto dm_count       = dms.size();

    std::vector<int32_t> dmt_cpu(nbeams * dm_count * nsamps_reduced, 0);
    std::vector<int32_t> dmt_gpu(nbeams * dm_count * nsamps_reduced, 0);
    cpu.execute(waterfall, nsamps, dmt_cpu);
    gpu.execute(waterfall, nsamps, dmt_gpu);

    REQUIRE_THAT(dmt_gpu, Catch::Matchers::Equals(dmt_cpu));
}

} // namespace dmt
