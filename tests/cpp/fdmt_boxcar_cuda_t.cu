#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

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

TEST_CASE("GPU FDMTBoxcar kill mask agrees with the CPU",
          "[gpu][parity][boxcar][kill]") {
    // 512 channels at depth 1: packed input takes the direct levels and top,
    // float32 the tree. The mask kills channel 0, a run inside the lower
    // half and the whole upper half (rows without noise never detect; the
    // screen thresholds see their sentinels). Killed rows hold the largest
    // code or a strong float value. Packed candidates are bit-identical,
    // S/N included; float32 agrees within rounding near the threshold.
    // Host input and device input on a nondefault stream.
    const plans::FDMTPlan plan(1100.0F, 1714.0F, 512, 256, 455.6e-6F, 300, 0, 1,
                               "valid", 1);
    constexpr SizeType kBeams  = 2;
    constexpr SizeType kBlocks = 3;
    const SizeType nchans      = plan.get_nchans();
    std::vector<uint8_t> kill(nchans, 1);
    kill[0] = 0;
    std::fill(kill.begin() + 40, kill.begin() + 57, 0);
    std::fill(kill.begin() + 256, kill.end(), 0);
    const auto killed_row = [&](SizeType row) {
        return kill[row % nchans] == 0;
    };
    for (const SizeType bits : {SizeType{8}, SizeType{16}, SizeType{32}}) {
        DYNAMIC_SECTION("bits=" << bits) {
            const uint32_t maximum = bits == 32 ? 0U : (1U << bits) - 1U;
            const SizeType nsamps  = plan.get_nsamps();
            std::mt19937 random(static_cast<uint32_t>(5 + bits));
            std::normal_distribution<float> normal(0.0F, 1.0F);
            std::vector<std::vector<uint8_t>> packed(kBlocks);
            std::vector<std::vector<float>> floats(kBlocks);
            for (SizeType k = 0; k < kBlocks; ++k) {
                if (bits == 32) {
                    floats[k].resize(kBeams * nchans * nsamps);
                    for (SizeType i = 0; i < floats[k].size(); ++i) {
                        floats[k][i] =
                            killed_row(i / nsamps) ? 40.0F : normal(random);
                    }
                    continue;
                }
                const SizeType bytes = bits / 8;
                packed[k].resize(kBeams * nchans * nsamps * bytes);
                for (SizeType i = 0; i < kBeams * nchans * nsamps; ++i) {
                    const uint32_t v =
                        killed_row(i / nsamps) ? maximum : random() & maximum;
                    packed[k][i * bytes] = static_cast<uint8_t>(v);
                    if (bits == 16) {
                        packed[k][(i * 2) + 1] = static_cast<uint8_t>(v >> 8U);
                    }
                }
            }
            BoxcarConfig cfg(24, 1.5F, 3.2F, 1.0F,
                             PeakFindingMode::kContiguousPeakPerWidth);
            cfg.max_candidates   = 4000000;
            cfg.max_cands_per_dm = 0;
            if (bits != 32) {
                cfg.input_nbits = bits;
                cfg.input_mean  = static_cast<float>(maximum) * 0.5F;
                cfg.noise_sigma = std::sqrt(static_cast<float>(maximum) *
                                            (maximum + 2.0F) / 12.0F);
            }
            FDMTBoxcar cpu_s(plan, cfg, true, Exec::cpu(4), kBeams,
                             algorithms::kFDMTAutoFuse, 64,
                             algorithms::kFDMTDefaultMaxMemory, kill);
            const Exec exec{algorithms::detail::kGPUBackend, 1, 0};
            FDMTBoxcar gpu_s(plan, cfg, true, exec, kBeams,
                             algorithms::kFDMTAutoFuse, 0,
                             algorithms::kFDMTDefaultMaxMemory, kill);
            std::vector<RawCandidate> c;
            std::vector<RawCandidate> g;
            for (SizeType k = 0; k < kBlocks; ++k) {
                if (bits == 32) {
                    append(c, cpu_s.execute(floats[k]));
                    append(g, gpu_s.execute(floats[k]));
                } else {
                    append(c, cpu_s.execute(packed[k]));
                    append(g, gpu_s.execute(packed[k]));
                }
            }
            append(c, cpu_s.flush());
            append(g, gpu_s.flush());
            REQUIRE_FALSE(c.empty());
            for (const auto& x : c) {
                CHECK(x.chan_begin < 256);
            }
            const auto check = [&](std::vector<RawCandidate> actual) {
                if (bits == 32) {
                    compare_near_threshold(actual, c, cfg.detect_thresh);
                    return;
                }
                compare(actual, c);
                auto expected = c;
                sort(actual);
                sort(expected);
                for (SizeType i = 0; i < actual.size(); ++i) {
                    CHECK(actual[i].snr == expected[i].snr);
                }
            };
            check(g);
            CHECK(gpu_s.get_memory_usage().staging ==
                  (bits == 32 ? floats.front().size() * sizeof(float)
                              : packed.front().size()));

            cudaStream_t stream;
            gpu_utils::check_gpu_call(cudaStreamCreate(&stream));
            gpu_s.reset_history(Stream{stream});
            const SizeType size = bits == 32
                                      ? floats.front().size() * sizeof(float)
                                      : packed.front().size();
            thrust::device_vector<uint8_t> device_input(size);
            auto* ptr = thrust::raw_pointer_cast(device_input.data());
            std::vector<RawCandidate> d;
            for (SizeType k = 0; k < kBlocks; ++k) {
                const void* src =
                    bits == 32 ? static_cast<const void*>(floats[k].data())
                               : packed[k].data();
                gpu_utils::check_gpu_call(cudaMemcpyAsync(
                    ptr, src, size, cudaMemcpyHostToDevice, stream));
                if (bits == 32) {
                    (void)gpu_s.execute(DeviceSpan<const float>(
                                            reinterpret_cast<const float*>(ptr),
                                            floats[k].size(),
                                            {exec.backend, 0}),
                                        Stream{stream});
                } else {
                    (void)gpu_s.execute(
                        DeviceSpan<const uint8_t>(ptr, size, {exec.backend, 0}),
                        Stream{stream});
                }
                append(d, gpu_s.copy_result(Stream{stream}));
                // The caller's buffer is not modified.
                std::vector<uint8_t> back(size);
                gpu_utils::check_gpu_call(cudaMemcpyAsync(
                    back.data(), ptr, size, cudaMemcpyDeviceToHost, stream));
                gpu_utils::check_gpu_call(cudaStreamSynchronize(stream));
                CHECK(std::memcmp(back.data(), src, size) == 0);
            }
            (void)gpu_s.flush_device(Stream{stream});
            append(d, gpu_s.copy_result(Stream{stream}));
            check(d);
            gpu_utils::check_gpu_call(cudaStreamDestroy(stream));
        }
    }
}
