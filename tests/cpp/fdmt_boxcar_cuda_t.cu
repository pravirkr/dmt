#include <thrust/device_vector.h>

#include "dmt/engines.hpp"
#include "dmt/gpu_compat.cuh"
#include "dmt/gpu_utils.cuh"

#include "fdmt_boxcar_reference.hpp"

using namespace boxcar_test;

TEST_CASE("FDMTBoxcar GPU host and device APIs match direct reference",
          "[gpu][parity][boxcar]") {
    const plans::FDMTPlan plan(700, 1200, 7, 733, 0.001F,
                               std::vector<IndexType>{-73, -12, 0, 4, 97},
                               "valid", 2);
    const auto input = waterfalls(plan, 2, 3);
    for (const bool smear : {true, false}) {
        for (const auto mode : {PeakFindingMode::kContiguousPeakPerWidth,
                                PeakFindingMode::kContiguousPeakAcrossWidths,
                                PeakFindingMode::kAllAboveThreshold}) {
            BoxcarConfig cfg;
            cfg.max_boxcar       = 37;
            cfg.wtsp             = 1;
            cfg.detect_thresh    = 2.7F;
            cfg.peak_mode        = mode;
            cfg.max_candidates   = 500000;
            cfg.max_cands_per_dm = 0;
            const auto expected = batched_reference(plan, cfg, smear, 2, input);
            for (const SizeType chunk : {SizeType{64}, plan.get_nsamps()}) {
                const Exec exec{algorithms::detail::kGPUBackend, 1, 0};
                FDMTBoxcar search(plan, cfg, smear, exec, 2,
                                  algorithms::kFDMTAutoFuse, chunk);
                std::vector<RawCandidate> actual;
                for (const auto& block : input) {
                    append(actual, search.execute(block));
                }
                append(actual, search.flush());
                compare(actual, expected);

                cudaStream_t stream;
                gpu_utils::check_gpu_call(cudaStreamCreate(&stream));
                search.reset_history(Stream{stream});
                thrust::device_vector<float> device_input(input.front().size());
                auto* ptr = thrust::raw_pointer_cast(device_input.data());
                actual.clear();
                for (const auto& block : input) {
                    gpu_utils::check_gpu_call(cudaMemcpyAsync(
                        ptr, block.data(), block.size() * sizeof(float),
                        cudaMemcpyHostToDevice, stream));
                    const auto result = search.execute(
                        DeviceSpan<const float>(ptr, block.size(),
                                                {exec.backend, 0}),
                        Stream{stream});
                    CHECK(result.candidates.size() == cfg.max_candidates);
                    CHECK(result.counts.size() == 1);
                    append(actual, search.copy_result(Stream{stream}));
                }
                (void)search.flush_device(Stream{stream});
                append(actual, search.copy_result(Stream{stream}));
                compare(actual, expected);
                (void)search.flush_device(Stream{stream});
                CHECK(search.copy_result(Stream{stream}).candidates.empty());
                CHECK_THROWS_AS(
                    search.execute(DeviceSpan<const float>(
                        ptr, device_input.size(), {Backend::kCPU, 0})),
                    std::invalid_argument);
                CHECK_THROWS_AS(
                    search.execute(DeviceSpan<const float>(
                        ptr, device_input.size(), {exec.backend, 100})),
                    std::invalid_argument);
                gpu_utils::check_gpu_call(cudaStreamDestroy(stream));
            }
        }
    }
}

TEST_CASE("FDMTBoxcar GPU supports width 4096 and bounded overflow",
          "[gpu][parity][boxcar]") {
    plans::FDMTPlan plan(700, 1200, 3, 1047, 0.001F,
                         std::vector<IndexType>{-5, 0, 7}, "valid", 1);
    const auto input = waterfalls(plan, 1, 5);
    BoxcarConfig cfg;
    cfg.max_boxcar       = 4096;
    cfg.wtsp             = 2;
    cfg.detect_thresh    = 4;
    cfg.max_cands_per_dm = 0;
    const auto expected  = reference(plan, cfg, true, 1, input);
    FDMTBoxcar search(plan, cfg, true, {algorithms::detail::kGPUBackend, 1, 0},
                      1, algorithms::kFDMTAutoFuse, 128);
    std::vector<RawCandidate> actual;
    for (const auto& block : input) {
        append(actual, search.execute(block));
    }
    append(actual, search.flush());
    compare(actual, expected);

    cfg.max_boxcar       = 9;
    cfg.wtsp             = 1;
    cfg.detect_thresh    = 1;
    cfg.max_candidates   = 2;
    cfg.max_cands_per_dm = 3;
    cfg.skip_cold_prefix = false;
    cfg.peak_mode        = PeakFindingMode::kAllAboveThreshold;
    FDMTBoxcar limited(plan, cfg, true,
                       {algorithms::detail::kGPUBackend, 1, 0});
    std::vector<float> loud(3 * plan.get_nsamps(), 20);
    const auto result = limited.execute(loud);
    CHECK(result.emitted_count == 2);
    CHECK(result.overflowed());
    CHECK(result.dropped_per_dm > 0);
    CHECK(result.dropped_global > 0);
    CHECK(result.total_detected ==
          result.emitted_count + result.dropped_global + result.dropped_per_dm);
    limited.reset_history();
    std::ranges::fill(loud, 0.0F);
    CHECK(limited.execute(loud).candidates.empty());
}

TEST_CASE("FDMTBoxcar GPU accepts all 4096 integer width trials",
          "[gpu][boxcar]") {
    plans::FDMTPlan plan(700, 1200, 2, 17, 0.001F, 1);
    BoxcarConfig cfg;
    cfg.max_boxcar = 4096;
    cfg.wtsp       = 1;
    cfg.peak_mode  = PeakFindingMode::kAllAboveThreshold;
    FDMTBoxcar search(plan, cfg, true, {algorithms::detail::kGPUBackend, 1, 0});
    std::vector<float> zero(34);
    CHECK(search.get_search_plans().front().widths.size() == 4096);
    CHECK(search.execute(zero).candidates.empty());
    CHECK(search.flush().candidates.empty());
}

TEST_CASE("Packed FDMTBoxcar GPU matches CPU on host and nondefault stream",
          "[gpu][parity][boxcar][packed]") {
    const plans::FDMTPlan plan(700, 1200, 7, 733, 0.001F,
                               std::vector<IndexType>{-73, -12, 0, 4, 97},
                               "valid", 2);
    constexpr SizeType beams = 2;
    for (const SizeType bits : {1, 2, 4, 8, 16}) {
        const SizeType bytes = (plan.get_nsamps() * bits + 7) / 8;
        std::mt19937 random(121);
        std::vector<std::vector<uint8_t>> input(
            7, std::vector<uint8_t>(beams * 7 * bytes));
        for (auto& block : input) {
            for (auto& x : block) {
                x = static_cast<uint8_t>(random());
            }
        }
        for (const bool smear : {true, false}) {
            for (const auto mode :
                 {PeakFindingMode::kContiguousPeakPerWidth,
                  PeakFindingMode::kContiguousPeakAcrossWidths,
                  PeakFindingMode::kAllAboveThreshold}) {
                DYNAMIC_SECTION("bits=" << bits << " smear=" << smear
                                        << " mode=" << static_cast<int>(mode)) {
                    BoxcarConfig cfg;
                    cfg.input_nbits       = bits;
                    const uint32_t maxval = (1U << bits) - 1;
                    cfg.input_mean        = maxval * 0.5F;
                    cfg.noise_sigma    = std::sqrt(static_cast<float>(maxval) *
                                                   (maxval + 2.0F) / 12.0F);
                    cfg.max_boxcar     = 4096;
                    cfg.detect_thresh  = 2.713F;
                    cfg.peak_mode      = mode;
                    cfg.max_candidates = 500000;
                    cfg.max_cands_per_dm = 0;
                    FDMTBoxcar cpu(plan, cfg, smear, Exec::cpu(2), beams, 0,
                                   64);
                    std::vector<RawCandidate> expected;
                    for (const auto& block : input) {
                        append(expected, cpu.execute(block));
                    }
                    append(expected, cpu.flush());
                    const Exec exec{algorithms::detail::kGPUBackend, 1, 0};
                    FDMTBoxcar gpu(plan, cfg, smear, exec, beams,
                                   algorithms::kFDMTAutoFuse, 128);
                    std::vector<RawCandidate> actual;
                    for (const auto& block : input) {
                        append(actual, gpu.execute(block));
                    }
                    append(actual, gpu.flush());
                    compare(actual, expected);
                    cudaStream_t stream;
                    gpu_utils::check_gpu_call(cudaStreamCreate(&stream));
                    gpu.reset_history(Stream{stream});
                    thrust::device_vector<uint8_t> device_input(
                        input.front().size());
                    auto* ptr = thrust::raw_pointer_cast(device_input.data());
                    actual.clear();
                    for (const auto& block : input) {
                        gpu_utils::check_gpu_call(
                            cudaMemcpyAsync(ptr, block.data(), block.size(),
                                            cudaMemcpyHostToDevice, stream));
                        (void)gpu.execute(
                            DeviceSpan<const uint8_t>(ptr, block.size(),
                                                      {exec.backend, 0}),
                            Stream{stream});
                        append(actual, gpu.copy_result(Stream{stream}));
                    }
                    (void)gpu.flush_device(Stream{stream});
                    append(actual, gpu.copy_result(Stream{stream}));
                    compare(actual, expected);
                    gpu_utils::check_gpu_call(cudaStreamDestroy(stream));
                }
            }
        }
    }
}
