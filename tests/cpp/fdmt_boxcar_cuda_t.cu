#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <span>
#include <string>
#include <tuple>
#include <vector>

#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <thrust/device_vector.h>

#include "dmt/bit_pack_utils.hpp"
#include "dmt/boxcar.hpp"
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

TEST_CASE("FDMTBoxcar GPU searches plans whose root plane exceeds 32 bits",
          "[gpu][parity][boxcar]") {
    // 1024 delay trials x 2,101,248 samples: the root plane has more than
    // INT32_MAX elements. Standalone FDMT on the GPU rejects it (it writes
    // that plane); FDMTBoxcar never materialises it and must match the CPU.
    constexpr SizeType kNsamps = (SizeType{1} << 21) + 4096;
    constexpr IndexType kDtMax = 1023;
    static_assert(SizeType{kDtMax + 1} * kNsamps >
                  SizeType{std::numeric_limits<int32_t>::max()});
    CHECK_THROWS_AS(FDMT(1200.0F, 1500.0F, 4, kNsamps, 0.001F, kDtMax, 0, 1,
                         true, "valid",
                         {algorithms::detail::kGPUBackend, 1, 0}),
                    std::invalid_argument);

    const plans::FDMTPlan plan(1200.0F, 1500.0F, 4, kNsamps, 0.001F, kDtMax);
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> dist(-3, 3);
    std::vector<float> block(plan.get_nchans() * kNsamps);
    for (auto& v : block) {
        v = static_cast<float>(dist(rng));
    }
    // A bright pulse at zero delay, late in the block (row offsets past the
    // 32-bit range of the full-block root layout).
    const SizeType t_pulse = kNsamps - 50000;
    for (SizeType c = 0; c < plan.get_nchans(); ++c) {
        for (SizeType t = t_pulse; t < t_pulse + 3; ++t) {
            block[(c * kNsamps) + t] += 40.0F;
        }
    }
    BoxcarConfig cfg;
    cfg.max_boxcar       = 4;
    cfg.detect_thresh    = 7.0F;
    cfg.noise_sigma      = 2.0F;
    cfg.peak_mode        = PeakFindingMode::kContiguousPeakAcrossWidths;
    cfg.max_candidates   = SizeType{1} << 16;
    cfg.max_cands_per_dm = 0;
    FDMTBoxcar cpu(plan, cfg, true, {Backend::kCPU, 4, 0});
    FDMTBoxcar gpu(plan, cfg, true, {algorithms::detail::kGPUBackend, 1, 0});
    std::vector<RawCandidate> expected;
    std::vector<RawCandidate> actual;
    append(expected, cpu.execute(block));
    append(expected, cpu.flush());
    append(actual, gpu.execute(block));
    append(actual, gpu.flush());
    REQUIRE_FALSE(expected.empty());
    CHECK(std::ranges::any_of(expected, [&](const RawCandidate& c) {
        return c.peak_sample >= t_pulse && c.peak_sample < t_pulse + 8;
    }));
    compare(actual, expected);
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

TEST_CASE("FDMTBoxcar GPU fits wide boxes in small shared memory",
          "[gpu][parity][boxcar]") {
    // A scoring block holds a prefix ring of bit_ceil(max_boxcar + tile)
    // entries, a second one with the width screen (dense widths), and
    // per-thread run state. When that exceeds the shared memory per block,
    // the screen is dropped, then the block shrinks (256 -> 128 -> 64
    // threads); candidates stay the same. FDMTEngineConfig::
    // boxcar_shared_limit stands in for a small-shared-memory GPU. For
    // max_boxcar 4096, float rows need about 47 KB at 256 threads (80 KB
    // with the screen), 40 KB at 128 and 36 KB at 64.
    using namespace algorithms::detail; // NOLINT
    const plans::FDMTPlan plan(700, 1200, 7, 2048, 0.001F,
                               std::vector<IndexType>{-5, 0, 7, 40}, "valid",
                               1);
    const auto input = waterfalls(plan, 1, 4);
    const auto run   = [&](const BoxcarConfig& box, SizeType limit,
                           std::string& path) {
        const SizeType root = plan.get_niters();
        FDMTEngineConfig cfg{};
        cfg.use_box_smearing    = true;
        cfg.int_tree            = false;
        cfg.exec                = {kGPUBackend, 1, 0};
        cfg.max_fuse_levels     = root - 1;
        cfg.search_upper_levels = true;
        cfg.search_only         = true;
        cfg.boxcar_shared_limit = limit;
        auto engine             = make_fdmt_boxcar_gpu(
            plan, box, make_boxcar_plans(plan, box, true), cfg,
            make_boxcar_driver(make_fdmt_gpu(plan, cfg)));
        path = engine->search_path();
        std::vector<RawCandidate> out;
        for (const auto& block : input) {
            append(out, engine->execute(std::span<const float>(block)));
        }
        append(out, engine->flush());
        return out;
    };
    BoxcarConfig box;
    box.max_boxcar       = 4096;
    box.detect_thresh    = 3;
    box.max_cands_per_dm = 0;
    box.max_candidates   = 1U << 20;
    struct Case {
        float wtsp;
        SizeType limit;
        const char* note; // nullptr: no fallback taken
    };
    for (const auto& c :
         {Case{1.5F, 0, nullptr}, Case{1.5F, 44000, "128-thread blocks)"},
          Case{1.5F, 38000, "64-thread blocks)"}, Case{1.02F, 0, nullptr},
          Case{1.02F, 65536, "256-thread blocks, no width screen)"},
          Case{1.02F, 38000, "64-thread blocks, no width screen)"}}) {
        DYNAMIC_SECTION("wtsp " << c.wtsp << " limit " << c.limit) {
            box.wtsp            = c.wtsp;
            const auto expected = reference(plan, box, true, 1, input);
            REQUIRE(expected.size() > 100);
            std::string path;
            const auto actual = run(box, c.limit, path);
            INFO(path);
            if (c.note == nullptr) {
                CHECK(path.find("(boxcar:") == std::string::npos);
            } else {
                CHECK(path.ends_with(c.note));
            }
            compare(actual, expected);
        }
    }
    // Below the 64-thread block's ring nothing fits in shared memory: the
    // prefix rings move to device memory (no speed promise), and so do
    // boxes wider than 4096 at any limit.
    for (const auto& [width, limit] :
         {std::pair<SizeType, SizeType>{4096, 30000}, {6000, 0}}) {
        DYNAMIC_SECTION("device-memory ring, max_boxcar " << width << " limit "
                                                          << limit) {
            box.wtsp            = 1.5F;
            box.max_boxcar      = width;
            const auto expected = reference(plan, box, true, 1, input);
            REQUIRE(expected.size() > 100);
            std::string path;
            const auto actual = run(box, limit, path);
            INFO(path);
            CHECK(path.ends_with("(boxcar: device-memory ring)"));
            compare(actual, expected);
        }
    }
}

TEST_CASE("Packed FDMTBoxcar GPU searches boxes far wider than 4096",
          "[gpu][parity][boxcar][packed]") {
    // max_boxcar 65536 on 301 DM rows: each block's 1 MiB ring lives in
    // device memory, and the rows run in batches of 256 blocks that reuse
    // the rings. Exact against the CPU, across blocks shorter than the box.
    const plans::FDMTPlan plan(700, 1200, 16, 1024, 0.001F, 300);
    std::mt19937 random(11);
    std::vector<std::vector<uint8_t>> input(3, std::vector<uint8_t>(16 * 1024));
    for (auto& block : input) {
        for (auto& x : block) {
            x = static_cast<uint8_t>(random());
        }
    }
    BoxcarConfig cfg;
    cfg.input_nbits      = 8;
    cfg.input_mean       = 127.5F;
    cfg.noise_sigma      = std::sqrt(255.0F * 257.0F / 12.0F);
    cfg.max_boxcar       = 65536;
    cfg.wtsp             = 2;
    cfg.detect_thresh    = 2.5F;
    cfg.max_cands_per_dm = 0;
    cfg.max_candidates   = 1U << 20;
    const auto run       = [&](const Exec& exec) {
        FDMTBoxcar search(plan, cfg, true, exec);
        std::vector<RawCandidate> out;
        for (const auto& block : input) {
            append(out, search.execute(std::span<const uint8_t>(block)));
        }
        append(out, search.flush());
        return out;
    };
    const auto expected = run(Exec::cpu());
    REQUIRE(expected.size() > 100);
    compare(run({algorithms::detail::kGPUBackend, 1, 0}), expected);
}

TEST_CASE("Packed FDMTBoxcar GPU fits max_boxcar 4096 with dense widths",
          "[gpu][parity][boxcar][packed]") {
    // Packed rows keep 64-bit prefixes: with the width screen, max_boxcar
    // 4096 needs about 145 KB per block, over the L40's 99 KB. The search
    // drops the screen and matches the CPU exactly.
    const plans::FDMTPlan plan(700, 1200, 7, 2048, 0.001F,
                               std::vector<IndexType>{-5, 0, 7, 40}, "valid",
                               1);
    std::mt19937 random(7);
    std::vector<std::vector<uint8_t>> input(4, std::vector<uint8_t>(7 * 2048));
    for (auto& block : input) {
        for (auto& x : block) {
            x = static_cast<uint8_t>(random());
        }
    }
    BoxcarConfig cfg;
    cfg.input_nbits      = 8;
    cfg.input_mean       = 127.5F;
    cfg.noise_sigma      = std::sqrt(255.0F * 257.0F / 12.0F);
    cfg.max_boxcar       = 4096;
    cfg.wtsp             = 1.02F;
    cfg.detect_thresh    = 2.5F;
    cfg.max_cands_per_dm = 0;
    cfg.max_candidates   = 1U << 20;
    const auto run       = [&](const Exec& exec) {
        FDMTBoxcar search(plan, cfg, true, exec);
        std::vector<RawCandidate> out;
        for (const auto& block : input) {
            append(out, search.execute(std::span<const uint8_t>(block)));
        }
        append(out, search.flush());
        return out;
    };
    const auto expected = run(Exec::cpu());
    REQUIRE(expected.size() > 100);
    compare(run({algorithms::detail::kGPUBackend, 1, 0}), expected);

    // Float rows at the same width: the GPU's float prefix rings against
    // the CPU, within float rounding.
    const auto floats    = waterfalls(plan, 1, 3);
    BoxcarConfig fcfg    = cfg;
    fcfg.input_nbits     = 32;
    fcfg.input_mean      = 0.0F;
    fcfg.noise_sigma     = 1.0F;
    const auto run_float = [&](const Exec& exec) {
        FDMTBoxcar search(plan, fcfg, true, exec);
        std::vector<RawCandidate> out;
        for (const auto& block : floats) {
            append(out, search.execute(block));
        }
        append(out, search.flush());
        return out;
    };
    const auto expected_f = run_float(Exec::cpu());
    REQUIRE(expected_f.size() > 100);
    compare(run_float({algorithms::detail::kGPUBackend, 1, 0}), expected_f);
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
                    FDMTBoxcar cpu(plan, cfg, smear, Exec::cpu(), beams, 0, 64);
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
            FDMTBoxcar cpu_s(plan, cfg, true, Exec::cpu(), kBeams,
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
            // The transform's keep bytes (one per beam and channel): the
            // mask needs no block copy of its own, and host input is staged
            // chunk by chunk inside the transform.
            CHECK(gpu_s.get_memory_usage().staging ==
                  plan.get_nchans() * gpu_s.get_nbeams());

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

TEST_CASE("FDMTBoxcar GPU kKadane and kNone match kPrefix bit-for-bit",
          "[gpu][parity][boxcar][kadane]") {
    const plans::FDMTPlan plan(700, 1200, 7, 733, 0.001F,
                               std::vector<IndexType>{-73, -12, 0, 4, 97},
                               "valid", 2);
    const auto input = waterfalls(plan, 2, 3);
    const Exec gpu_exec{algorithms::detail::kGPUBackend, 1, 0};

    for (const bool trap : {false, true}) {
        for (const auto mode : {PeakFindingMode::kContiguousPeakPerWidth,
                                PeakFindingMode::kContiguousPeakAcrossWidths,
                                PeakFindingMode::kAllAboveThreshold}) {
            BoxcarConfig base_cfg;
            base_cfg.max_boxcar       = 64;
            base_cfg.wtsp             = 1.5F;
            base_cfg.detect_thresh    = 2.7F;
            base_cfg.peak_mode        = mode;
            base_cfg.use_trapezoid    = trap;
            base_cfg.max_candidates   = 500000;
            base_cfg.max_cands_per_dm = 0;

            FDMTBoxcar ref_gpu(plan, base_cfg, true, gpu_exec, 2,
                               algorithms::kFDMTAutoFuse, 64);
            std::vector<RawCandidate> expected;
            std::vector<uint64_t> expected_frontiers;
            for (const auto& block : input) {
                append(expected, ref_gpu.execute(block));
                expected_frontiers.push_back(ref_gpu.get_candidate_frontier());
            }
            append(expected, ref_gpu.drain());

            for (const auto screen :
                 {BoxcarScreenMode::kKadane, BoxcarScreenMode::kNone}) {
                for (const SizeType chunk : {SizeType{64}, plan.get_nsamps()}) {
                    BoxcarConfig cfg = base_cfg;
                    cfg.screen_mode  = screen;
                    FDMTBoxcar search(plan, cfg, true, gpu_exec, 2,
                                      algorithms::kFDMTAutoFuse, chunk);
                    std::vector<RawCandidate> actual;
                    for (SizeType b = 0; b < input.size(); ++b) {
                        append(actual, search.execute(input[b]));
                        if (chunk == 64) {
                            CHECK(search.get_candidate_frontier() ==
                                  expected_frontiers[b]);
                        }
                    }
                    append(actual, search.drain());
                    compare(actual, expected);

                    // Verify reset_history + async DeviceSpan stream execution
                    cudaStream_t stream;
                    gpu_utils::check_gpu_call(cudaStreamCreate(&stream));
                    search.reset_history(Stream{stream});
                    thrust::device_vector<float> dev_in(input.front().size());
                    auto* ptr = thrust::raw_pointer_cast(dev_in.data());
                    actual.clear();
                    for (const auto& block : input) {
                        gpu_utils::check_gpu_call(cudaMemcpyAsync(
                            ptr, block.data(), block.size() * sizeof(float),
                            cudaMemcpyHostToDevice, stream));
                        (void)search.execute(
                            DeviceSpan<const float>(ptr, block.size(),
                                                    {gpu_exec.backend, 0}),
                            Stream{stream});
                        append(actual, search.copy_result(Stream{stream}));
                    }
                    gpu_utils::check_gpu_call(cudaStreamSynchronize(stream));
                    append(actual, search.drain());
                    compare(actual, expected);
                    gpu_utils::check_gpu_call(cudaStreamDestroy(stream));
                }
            }
        }
    }

    // Width 4096 SMEM-lean Design C check
    const plans::FDMTPlan wide_plan(700, 1200, 3, 1047, 0.001F,
                                    std::vector<IndexType>{-5, 0, 7}, "valid",
                                    1);
    const auto wide_in = waterfalls(wide_plan, 1, 3);
    BoxcarConfig pfx_cfg;
    pfx_cfg.max_boxcar       = 4096;
    pfx_cfg.wtsp             = 2.0F;
    pfx_cfg.detect_thresh    = 3.8F;
    pfx_cfg.max_cands_per_dm = 0;
    BoxcarConfig kad_cfg     = pfx_cfg;
    kad_cfg.screen_mode      = BoxcarScreenMode::kKadane;
    FDMTBoxcar pfx_wide(wide_plan, pfx_cfg, true, gpu_exec, 1,
                        algorithms::kFDMTAutoFuse, 128);
    FDMTBoxcar kad_wide(wide_plan, kad_cfg, true, gpu_exec, 1,
                        algorithms::kFDMTAutoFuse, 128);
    std::vector<RawCandidate> pfx_cands;
    std::vector<RawCandidate> kad_cands;
    for (const auto& block : wide_in) {
        append(pfx_cands, pfx_wide.execute(block));
        append(kad_cands, kad_wide.execute(block));
    }
    append(pfx_cands, pfx_wide.flush());
    append(kad_cands, kad_wide.flush());
    compare(kad_cands, pfx_cands);
}

TEST_CASE("FDMTBoxcar GPU kKadane and kNone support wide boxcars and small "
          "shared memory",
          "[gpu][parity][boxcar][kadane]") {
    using namespace algorithms::detail; // NOLINT
    const Exec gpu_exec{kGPUBackend, 1, 0};
    const plans::FDMTPlan plan(700, 1200, 7, 2048, 0.001F,
                               std::vector<IndexType>{-5, 0, 7, 40}, "valid",
                               1);
    const auto input = waterfalls(plan, 1, 4);

    for (const SizeType max_w : {SizeType{8192}, SizeType{16384}}) {
        for (const auto screen :
             {BoxcarScreenMode::kKadane, BoxcarScreenMode::kNone}) {
            BoxcarConfig cfg;
            cfg.max_boxcar       = max_w;
            cfg.wtsp             = 2.0F;
            cfg.detect_thresh    = 3.0F;
            cfg.screen_mode      = screen;
            cfg.max_candidates   = 1U << 20;
            cfg.max_cands_per_dm = 0;

            const auto expected = reference(plan, cfg, true, 1, input);
            REQUIRE_FALSE(expected.empty());
            FDMTEngineConfig ecfg{};
            ecfg.use_box_smearing    = true;
            ecfg.int_tree            = false;
            ecfg.exec                = gpu_exec;
            ecfg.max_fuse_levels     = plan.get_niters() - 1;
            ecfg.search_upper_levels = true;
            ecfg.search_only         = true;
            auto engine              = make_fdmt_boxcar_gpu(
                plan, cfg, make_boxcar_plans(plan, cfg, true), ecfg,
                make_boxcar_driver(make_fdmt_gpu(plan, ecfg)));
            CHECK(engine->search_path().ends_with(
                "(boxcar: device-memory ring)"));

            FDMTBoxcar gpu(plan, cfg, true, gpu_exec, 1,
                           algorithms::kFDMTAutoFuse, 128);
            std::vector<RawCandidate> actual;
            for (const auto& block : input) {
                append(actual, gpu.execute(block));
            }
            append(actual, gpu.flush());
            compare(actual, expected);
        }
    }

    // Small-SMEM block shrinking (38000 B -> 64-thread Kadane) and fallback
    // (30000 B -> device-memory ring) on kKadane at max_boxcar = 4096.
    BoxcarConfig box;
    box.max_boxcar       = 4096;
    box.wtsp             = 1.5F;
    box.detect_thresh    = 3.0F;
    box.screen_mode      = BoxcarScreenMode::kKadane;
    box.max_candidates   = 1U << 20;
    box.max_cands_per_dm = 0;
    const auto expected  = reference(plan, box, true, 1, input);
    REQUIRE(expected.size() > 100);

    for (const auto& [limit, suffix] :
         {std::pair<SizeType, const char*>{38000, "64-thread blocks)"},
          {30000, "(boxcar: device-memory ring)"}}) {
        FDMTEngineConfig cfg{};
        cfg.use_box_smearing    = true;
        cfg.int_tree            = false;
        cfg.exec                = gpu_exec;
        cfg.max_fuse_levels     = plan.get_niters() - 1;
        cfg.search_upper_levels = true;
        cfg.search_only         = true;
        cfg.boxcar_shared_limit = limit;
        auto engine             = make_fdmt_boxcar_gpu(
            plan, box, make_boxcar_plans(plan, box, true), cfg,
            make_boxcar_driver(make_fdmt_gpu(plan, cfg)));
        CHECK(engine->search_path().ends_with(suffix));
        std::vector<RawCandidate> actual;
        for (const auto& block : input) {
            append(actual, engine->execute(std::span<const float>(block)));
        }
        append(actual, engine->flush());
        compare(actual, expected);
    }
}

TEST_CASE("FDMTBoxcar GPU float kKadane is bit-exact with kPrefix across "
          "widths and shared-memory limits",
          "[gpu][parity][boxcar][kadane]") {
    using namespace algorithms::detail; // NOLINT
    const Exec gpu_exec{kGPUBackend, 1, 0};
    const plans::FDMTPlan plan(700, 1200, 7, 2048, 0.001F,
                               std::vector<IndexType>{-5, 0, 7, 40}, "valid",
                               1);
    // waterfalls() rounds to 1/16 (exact float sums in any order); add
    // unrounded noise so the summation order shows in the S/N bits.
    auto input = waterfalls(plan, 1, 4);
    for (SizeType k = 0; k < input.size(); ++k) {
        const auto extra = test_random::normal<float>(input[k].size(), 7 + k);
        for (SizeType i = 0; i < input[k].size(); ++i) {
            input[k][i] += 0.37F * extra[i];
        }
    }

    const auto run = [&](const BoxcarConfig& box, SizeType limit) {
        FDMTEngineConfig cfg{};
        cfg.use_box_smearing    = true;
        cfg.int_tree            = false;
        cfg.exec                = gpu_exec;
        cfg.max_fuse_levels     = plan.get_niters() - 1;
        cfg.search_upper_levels = true;
        cfg.search_only         = true;
        cfg.boxcar_shared_limit = limit;
        auto engine             = make_fdmt_boxcar_gpu(
            plan, box, make_boxcar_plans(plan, box, true), cfg,
            make_boxcar_driver(make_fdmt_gpu(plan, cfg)));
        std::vector<RawCandidate> out;
        for (const auto& block : input) {
            append(out, engine->execute(std::span<const float>(block)));
        }
        append(out, engine->flush());
        const auto key = [](const RawCandidate& c) {
            return std::tuple(c.beam_idx, c.dm_idx, c.subband_level,
                              c.subband_idx, c.begin_sample, c.end_sample,
                              c.peak_sample, c.filter_idx,
                              std::bit_cast<uint32_t>(c.snr));
        };
        std::ranges::sort(
            out, [&](const auto& a, const auto& b) { return key(a) < key(b); });
        return out;
    };

    // The Stage 2 float prefix sum is associated per block tile, so kKadane
    // must pick the kPrefix block size (or fall back to kPrefix) everywhere.
    for (const SizeType max_w :
         {SizeType{150}, SizeType{1024}, SizeType{2048}, SizeType{4096}}) {
        for (const SizeType limit :
             {SizeType{0}, SizeType{60000}, SizeType{45000}, SizeType{38000},
              SizeType{30000}}) {
            CAPTURE(max_w, limit);
            BoxcarConfig box;
            box.max_boxcar       = max_w;
            box.wtsp             = 1.5F;
            box.detect_thresh    = 3.0F;
            box.max_candidates   = 1U << 20;
            box.max_cands_per_dm = 0;
            box.screen_mode      = BoxcarScreenMode::kPrefix;
            const auto pfx       = run(box, limit);
            box.screen_mode      = BoxcarScreenMode::kKadane;
            const auto kad       = run(box, limit);
            REQUIRE(pfx.size() > 10);
            REQUIRE(kad.size() == pfx.size());
            SizeType mismatched = 0;
            for (SizeType i = 0; i < pfx.size(); ++i) {
                const auto& a = pfx[i];
                const auto& b = kad[i];
                mismatched += static_cast<SizeType>(
                    a.dm_idx != b.dm_idx || a.begin_sample != b.begin_sample ||
                    a.end_sample != b.end_sample ||
                    a.peak_sample != b.peak_sample ||
                    a.filter_idx != b.filter_idx ||
                    std::bit_cast<uint32_t>(a.snr) !=
                        std::bit_cast<uint32_t>(b.snr));
            }
            CHECK(mismatched == 0);
        }
    }
}

TEST_CASE("Packed FDMTBoxcar GPU kKadane and kNone match CPU across bit depths",
          "[gpu][parity][boxcar][packed][kadane]") {
    const plans::FDMTPlan plan(700, 1200, 7, 733, 0.001F,
                               std::vector<IndexType>{-73, -12, 0, 4, 97},
                               "valid", 2);
    constexpr SizeType kBeams = 2;
    const Exec gpu_exec{algorithms::detail::kGPUBackend, 1, 0};

    for (const SizeType bits : {SizeType{4}, SizeType{8}, SizeType{16}}) {
        const SizeType bytes = (plan.get_nsamps() * bits + 7) / 8;
        std::mt19937 random(static_cast<uint32_t>(331 + bits));
        std::vector<std::vector<uint8_t>> input(
            4, std::vector<uint8_t>(kBeams * 7 * bytes));
        for (auto& block : input) {
            for (auto& x : block) {
                x = static_cast<uint8_t>(random());
            }
        }
        for (const bool trap : {false, true}) {
            for (const auto screen :
                 {BoxcarScreenMode::kKadane, BoxcarScreenMode::kNone}) {
                BoxcarConfig cfg;
                cfg.input_nbits     = bits;
                const uint32_t maxv = (1U << bits) - 1U;
                cfg.input_mean      = static_cast<float>(maxv) * 0.5F;
                cfg.noise_sigma =
                    std::sqrt(static_cast<float>(maxv) * (maxv + 2.0F) / 12.0F);
                cfg.max_boxcar       = 128;
                cfg.detect_thresh    = 2.75F;
                cfg.use_trapezoid    = trap;
                cfg.screen_mode      = screen;
                cfg.max_candidates   = 500000;
                cfg.max_cands_per_dm = 0;

                FDMTBoxcar cpu(plan, cfg, true, Exec::cpu(), kBeams, 0, 64);
                FDMTBoxcar gpu(plan, cfg, true, gpu_exec, kBeams,
                               algorithms::kFDMTAutoFuse, 128);
                std::vector<RawCandidate> expected;
                std::vector<RawCandidate> actual;
                for (const auto& block : input) {
                    append(expected, cpu.execute(block));
                    append(actual, gpu.execute(block));
                }
                append(expected, cpu.flush());
                append(actual, gpu.flush());
                compare(actual, expected);
            }
        }
    }
}

TEST_CASE("GPU host-memory calls stage inside the budget",
          "[gpu][boxcar][budget]") {
    // FDMTBoxcar copies a host block chunk by chunk into a buffer reserved
    // at construction, FDMTFFTBoxcar the whole block (both part of the
    // budgeted usage): host calls allocate nothing. Plain FDMT still copies
    // the block and output plane on the first host call, outside its
    // budget.
    const plans::FDMTPlan plan(1200.0F, 1500.0F, 64, 2048, 0.000256F, 200);
    const SizeType beams = 2;
    const SizeType block =
        beams * plan.get_nchans() * plan.get_nsamps() * sizeof(float);
    const auto input = waterfalls(plan, beams, 2);
    const Exec exec{algorithms::detail::kGPUBackend, 1, 0};
    BoxcarConfig cfg;
    cfg.max_boxcar = 16;

    SECTION("FDMTBoxcar") {
        FDMTBoxcar search(plan, cfg, true, exec, beams);
        const auto e =
            FDMTBoxcar::estimate_memory(plan, cfg, true, exec, beams);
        const auto before = search.get_memory_usage();
        CHECK(before.total() == e.usage.total());
        CHECK(before.staging == 0);
        (void)search.execute(std::span<const float>(input[0]));
        (void)search.execute(std::span<const float>(input[1]));
        CHECK(search.get_memory_usage().total() == before.total());
        CHECK(search.get_memory_usage().staging == 0);
    }
    SECTION("FDMTBoxcar, packed 8-bit input") {
        cfg.input_nbits = 8;
        cfg.input_mean  = 2.0F;
        FDMTBoxcar search(plan, cfg, true, exec, beams);
        const auto before = search.get_memory_usage();
        const std::vector<uint8_t> packed(block / sizeof(float), 2);
        (void)search.execute(std::span<const uint8_t>(packed));
        CHECK(search.get_memory_usage().total() == before.total());
        CHECK(search.get_memory_usage().staging == 0);
    }
    SECTION("FDMTFFTBoxcar") {
        algorithms::FDMTFFTBoxcar search(plan, cfg, true, exec, beams);
        const auto e = algorithms::FDMTFFTBoxcar::estimate_memory(
            plan, cfg, true, exec, beams);
        const auto before = search.get_memory_usage();
        CHECK(before.total() == e.usage.total());
        CHECK(before.staging == block);
        (void)search.execute(std::span<const float>(input[0]));
        (void)search.execute(std::span<const float>(input[1]));
        CHECK(search.get_memory_usage().total() == before.total());
    }
    SECTION("FDMT") {
        FDMT fdmt(plan, true, exec, beams);
        const auto before = fdmt.get_memory_usage();
        std::vector<float> out(beams * plan.get_dmt_size());
        fdmt.execute(std::span<const float>(input[0]), std::span<float>(out));
        const auto after = fdmt.get_memory_usage();
        CHECK(after.workspace ==
              before.workspace + block + (out.size() * sizeof(float)));
        fdmt.execute(std::span<const float>(input[1]), std::span<float>(out));
        CHECK(fdmt.get_memory_usage().total() == after.total());
    }
}

TEST_CASE("FDMTBoxcar GPU host input staged per chunk equals device input",
          "[gpu][boxcar][budget][parity]") {
    // Host blocks are copied chunk by chunk (rows `chunk` apart) into the
    // transform's reserved buffer. Every search path reads them as it reads
    // a device block: identical candidates, bit for bit, over several
    // blocks and drain(), and no memory beyond the estimate.
    const Exec exec{algorithms::detail::kGPUBackend, 1, 0};
    const float tsamp = 455.6e-6F;
    struct Case {
        std::string name;
        plans::FDMTPlan plan;
        SizeType nbits;
        SizeType beams;
        SizeType chunk;  // 0: automatic
        SizeType budget; // 0: default; 1: the minimum
        bool ctor_mask;
        bool call_masks;
        SizeType fuse;    // kFDMTAutoFuse: direct levels where eligible
        std::string path; // in search_path ("" : the tree)
    };
    const SizeType kAuto = algorithms::kFDMTAutoFuse;
    const plans::FDMTPlan small(1200.0F, 1500.0F, 64, 2500, 0.000256F, 200);
    const plans::FDMTPlan sub(1100.0F, 1714.0F, 1152, 4096, tsamp, 2700, 0, 1,
                              "valid", 2);
    const plans::FDMTPlan chan_dm(1100.0F, 1714.0F, 256, 2048, tsamp, 600, 0, 1,
                                  "valid", 1, 150.0);
    // Blocks that are not a multiple of the chunk (2500 samples), two
    // beams, constructor and per-call masks, float and 1-8 bits. The
    // skewed path on packed input is compared host vs device by the
    // drain() test below.
    const std::vector<Case> cases{
        {"tree f32", small, 32, 2, 1024, 0, true, false, 0, ""},
        {"tree 4-bit", small, 4, 2, 1024, 0, false, true, 0, ""},
        {"fused f32", small, 32, 1, 512, 0, false, true, 2, ""},
        {"direct f32", small, 32, 2, 1024, 0, true, false, kAuto,
         "direct levels"},
        {"direct 1-bit", small, 1, 1, 512, 0, true, false, kAuto,
         "direct levels"},
        {"subband direct f32", sub, 32, 1, 0, 0, false, true, kAuto,
         "direct top"},
        {"subband direct 8-bit", sub, 8, 1, 0, 0, true, false, kAuto,
         "direct top"},
        {"subband minimum f32", sub, 32, 1, 0, 1, true, false, kAuto,
         "skewed direct levels"},
        {"subband minimum 8-bit", sub, 8, 2, 0, 1, false, true, kAuto,
         "direct levels"},
        {"chan_dm 8-bit", chan_dm, 8, 1, 0, 0, true, true, kAuto,
         "direct levels"},
    };
    for (const auto& c : cases) {
        DYNAMIC_SECTION(c.name) {
            const auto& plan = c.plan;
            BoxcarConfig cfg;
            cfg.max_boxcar = 24;
            cfg.wtsp       = 1.5F;
            // Enough candidates on the 64-channel plan, few enough elsewhere.
            cfg.detect_thresh    = plan.get_nchans() < 256 ? 3.5F : 4.5F;
            cfg.max_candidates   = 1U << 20;
            cfg.max_cands_per_dm = 0;
            cfg.input_nbits      = c.nbits;
            const int top        = c.nbits == 32 ? 255 : (1 << c.nbits) - 1;
            cfg.input_mean = c.nbits == 32 ? 0.0F : static_cast<float>(top / 2);
            // Uniform codes 0..top (float: shifted by -127.5).
            cfg.noise_sigma = static_cast<float>(
                std::sqrt((((top + 1.0) * (top + 1.0)) - 1.0) / 12.0));
            cfg.per_call_masks    = c.call_masks;
            const SizeType nchans = plan.get_nchans();
            std::vector<uint8_t> ctor_mask;
            if (c.ctor_mask) {
                ctor_mask.assign(nchans, 1);
                for (SizeType ch = 3; ch < nchans; ch += 7) {
                    ctor_mask[ch] = 0;
                }
            }
            SizeType budget = algorithms::kFDMTDefaultMaxMemory;
            if (c.budget == 1) {
                budget = FDMTBoxcar::estimate_minimum_memory(
                             plan, cfg, true, exec, c.beams, c.fuse, c.chunk,
                             ctor_mask)
                             .max_memory_bytes;
            }
            const auto est =
                FDMTBoxcar::estimate_memory(plan, cfg, true, exec, c.beams,
                                            c.fuse, c.chunk, budget, ctor_mask);
            FDMTBoxcar host(plan, cfg, true, exec, c.beams, c.fuse, c.chunk,
                            budget, ctor_mask);
            FDMTBoxcar device(plan, cfg, true, exec, c.beams, c.fuse, c.chunk,
                              budget, ctor_mask);
            INFO(host.summary());
            INFO("path '" << est.search_path << "'");
            if (c.path.empty()) {
                CHECK(est.search_path.empty());
            } else {
                CHECK(est.search_path.find(c.path) != std::string::npos);
            }
            const SizeType per_block = c.beams * nchans * plan.get_nsamps();
            const SizeType bytes = c.nbits == 32
                                       ? per_block
                                       : c.beams * nchans *
                                             bit_pack_utils::packed_row_bytes(
                                                 plan.get_nsamps(), c.nbits);
            std::mt19937 gen(97);
            std::uniform_int_distribution<int> code(0, top);
            thrust::device_vector<float> fbuf(per_block);
            thrust::device_vector<uint8_t> pbuf(bytes);
            std::vector<RawCandidate> got;
            std::vector<RawCandidate> want;
            constexpr SizeType kBlocks = 3;
            for (SizeType b = 0; b < kBlocks; ++b) {
                std::vector<uint8_t> mask(nchans, 1);
                for (SizeType ch = b; ch < nchans; ch += 5) {
                    mask[ch] = 0;
                }
                const std::span<const uint8_t> call_mask =
                    c.call_masks ? std::span<const uint8_t>(mask)
                                 : std::span<const uint8_t>{};
                if (c.nbits == 32) {
                    std::vector<float> v(per_block);
                    for (auto& x : v) {
                        x = static_cast<float>(code(gen)) - 127.5F;
                    }
                    thrust::copy(v.begin(), v.end(), fbuf.begin());
                    const DeviceSpan<const float> d(
                        thrust::raw_pointer_cast(fbuf.data()), per_block,
                        {exec.backend, 0});
                    (void)device.execute(d, call_mask);
                    append(want, device.copy_result());
                    append(got,
                           host.execute(std::span<const float>(v), call_mask));
                } else {
                    std::vector<uint8_t> v(bytes);
                    for (auto& x : v) {
                        x = static_cast<uint8_t>(code(gen) & 0xFF);
                        if (c.nbits < 8) {
                            x = static_cast<uint8_t>(
                                std::uniform_int_distribution<int>(0,
                                                                   255)(gen));
                        }
                    }
                    thrust::copy(v.begin(), v.end(), pbuf.begin());
                    const DeviceSpan<const uint8_t> d(
                        thrust::raw_pointer_cast(pbuf.data()), bytes,
                        {exec.backend, 0});
                    (void)device.execute(d, call_mask);
                    append(want, device.copy_result());
                    append(got, host.execute(std::span<const uint8_t>(v),
                                             call_mask));
                }
                CHECK(host.get_memory_usage().total() == est.usage.total());
                CHECK(host.get_memory_usage().staging == est.usage.staging);
            }
            append(got, host.drain());
            append(want, device.drain());
            CHECK(host.get_memory_usage().total() == est.usage.total());
            REQUIRE(want.size() > 50);
            compare(got, want);
        }
    }
}

TEST_CASE("FDMTBoxcar estimate_memory equals the constructed instance (gpu)",
          "[boxcar][gpu][budget]") {
    const Exec gpu{algorithms::detail::kGPUBackend, 1, 0};
    const SizeType big = algorithms::kFDMTDefaultMaxMemory;
    for (const auto& c : estimate_cases()) {
        INFO(c.name << " beams " << c.beams);
        const auto e = FDMTBoxcar::estimate_memory(
            c.plan, c.config, true, gpu, c.beams, algorithms::kFDMTAutoFuse,
            c.chunk, c.budget, c.kill);
        FDMTBoxcar f(c.plan, c.config, true, gpu, c.beams,
                     algorithms::kFDMTAutoFuse, c.chunk, c.budget, c.kill);
        check_estimate(e, f);
        CHECK(e.usage.budgeted() <= c.budget);
        if (c.chunk != 0) {
            CHECK(e.chunk_nsamps == c.chunk);
        }
        if (c.budget < big) {
            CHECK(e.chunk_nsamps < c.plan.get_nsamps());
        }
        // Host calls stage inside the estimate: they add nothing.
        const SizeType rows = c.beams * c.plan.get_nchans();
        const auto total    = f.get_memory_usage().total();
        if (c.config.input_nbits == 32) {
            std::vector<float> in(rows * c.plan.get_nsamps());
            (void)f.execute(std::span<const float>(in));
        } else {
            std::vector<uint8_t> in(rows * c.plan.get_nsamps(), 127);
            (void)f.execute(std::span<const uint8_t>(in));
        }
        CHECK(f.get_memory_usage().total() == total);
    }
    SECTION("nothing is allocated") {
        // A 2^40-candidate store (tens of TB) on the device and host: sized,
        // never allocated (an allocation would throw).
        const auto cases   = estimate_cases();
        BoxcarConfig cfg   = cases.front().config;
        cfg.max_candidates = SizeType{1} << 40;
        const auto e       = FDMTBoxcar::estimate_memory(
            cases.front().plan, cfg, true, gpu, 1, algorithms::kFDMTAutoFuse, 0,
            SIZE_MAX / 2);
        CHECK(e.usage.candidates >= cfg.max_candidates * sizeof(RawCandidate));
    }
    SECTION("a transform the device cannot hold fails with the boxcar's need") {
        // The FDMT allocates first; its failure reports FDMTBoxcar's total,
        // not the FDMT's own message.
        const auto c         = estimate_cases().front();
        const SizeType beams = SizeType{1} << 14;
        const auto e         = FDMTBoxcar::estimate_memory(
            c.plan, c.config, true, gpu, beams, algorithms::kFDMTAutoFuse, 0,
            SIZE_MAX / 2);
        const auto& props = gpu_utils::device_properties(gpu.device);
        if (e.usage.state + e.usage.history <= props.totalGlobalMem) {
            SKIP("device larger than the probe");
        }
        // Checked against the free memory before anything is allocated
        // (the check's own wording), with the budget advice.
        CHECK_THROWS_MATCHES(
            FDMTBoxcar(c.plan, c.config, true, gpu, beams,
                       algorithms::kFDMTAutoFuse, 0, SIZE_MAX / 2),
            std::bad_alloc,
            Catch::Matchers::MessageMatches(
                Catch::Matchers::StartsWith(
                    "FDMTBoxcar (cuda): this configuration needs") &&
                Catch::Matchers::ContainsSubstring(
                    "is free on the device (nothing allocated)") &&
                Catch::Matchers::ContainsSubstring("max_memory_bytes") &&
                Catch::Matchers::ContainsSubstring(
                    "estimate_minimum_memory()") &&
                !Catch::Matchers::ContainsSubstring("nsamps") &&
                !Catch::Matchers::ContainsSubstring("FDMT (cuda): this")));
    }
}

TEST_CASE("The free-memory check fails early and only when over",
          "[gpu][budget]") {
    // Synthetic device figures: the check reads nothing itself.
    constexpr SizeType kGiB = SizeType{1} << 30;
    const gpu_utils::DeviceMemoryInfo dev{
        .free = 3 * kGiB, .total = 8 * kGiB, .known = true};
    CHECK_NOTHROW(gpu_utils::check_free_memory("X", 3 * kGiB, dev,
                                               gpu_utils::kAdviceBudget));
    CHECK_NOTHROW(
        gpu_utils::check_free_memory("X", kGiB, dev, gpu_utils::kAdviceBudget));
    // Unknown free memory: nothing to check against (the allocation
    // reports a failure).
    CHECK_NOTHROW(gpu_utils::check_free_memory("X", 100 * kGiB,
                                               gpu_utils::DeviceMemoryInfo{},
                                               gpu_utils::kAdviceBudget));
    try {
        gpu_utils::check_free_memory("FDMT (cuda)", (3 * kGiB) + 1, dev,
                                     gpu_utils::kAdviceBudget);
        FAIL("one byte over the free memory did not throw");
    } catch (const gpu_utils::DeviceAllocationError& e) {
        const std::string what = e.what();
        CHECK(what.starts_with("FDMT (cuda): this configuration needs 3.00 "
                               "GiB of device memory and only 3.00 GiB of "
                               "8.00 GiB is free on the device (nothing "
                               "allocated); "));
        CHECK(what.ends_with(gpu_utils::kAdviceBudget));
        CHECK(e.cause() == "3.00 GiB needed, 3.00 GiB free");
    }
}

TEST_CASE("Device allocation errors nest by cause", "[gpu][budget]") {
    const gpu_utils::DeviceMemoryInfo before{
        .free = SizeType{3} << 30, .total = SizeType{8} << 30, .known = true};
    const auto inner = gpu_utils::allocation_failure(
        "FDMT (cuda)", SizeType{5} << 30, before,
        std::runtime_error("device allocation: out of memory"),
        gpu_utils::kAdviceBudget);
    CHECK(inner.cause() == "device allocation: out of memory");
    const auto outer =
        gpu_utils::allocation_failure("CohFDMT (cuda)", SizeType{6} << 30,
                                      before, inner, gpu_utils::kAdviceBlocks);
    const std::string what = outer.what();
    CHECK(what.starts_with("CohFDMT (cuda): this configuration needs 6.00 "
                           "GiB of device memory and 3.00 GiB of 8.00 GiB "
                           "was free; use shorter blocks"));
    CHECK(what.ends_with(": device allocation: out of memory"));
    // One message, not the FDMT's nested inside.
    const std::string needs = "this configuration needs";
    CHECK(what.find(needs) == what.rfind(needs));
}

TEST_CASE("FDMTBoxcar GPU drain() after device input allocates nothing",
          "[gpu][boxcar][drain][memory]") {
    // A subband search on the windowed (skewed) top has a search latency, so
    // drain() pads the stream. Driven by device spans only, the instance
    // never exceeds its estimate, drain() included, and finds what a
    // host-driven instance finds.
    const float tsamp = 455.6e-6F;
    const auto grid   = plans::DDMTPlan::generate_levin_dm_grid_piecewise(
        0.0F, 2000.0F, tsamp, tsamp, 856.0F, 1712.0F, 4096, 1.1F);
    const plans::FDMTPlan plan(856.0F, 1712.0F, 4096, 4096, tsamp, grid,
                               "valid", 2);
    const Exec exec{algorithms::detail::kGPUBackend, 1, 0};
    constexpr SizeType kBlocks = 3;
    const SizeType n           = plan.get_nchans() * plan.get_nsamps();
    std::vector<std::vector<uint8_t>> bytes(kBlocks);
    std::vector<std::vector<float>> values(kBlocks);
    std::mt19937 gen(4242);
    std::uniform_int_distribution<int> dist(0, 255);
    for (SizeType b = 0; b < kBlocks; ++b) {
        bytes[b].resize(n);
        values[b].resize(n);
        for (SizeType i = 0; i < n; ++i) {
            bytes[b][i]  = static_cast<uint8_t>(dist(gen));
            values[b][i] = static_cast<float>(bytes[b][i]) - 128.0F;
        }
    }
    for (const bool packed : {false, true}) {
        DYNAMIC_SECTION("packed " << packed) {
            const BoxcarConfig cfg(
                24, 1.5F, 4.2F, 73.9F, PeakFindingMode::kContiguousPeakPerWidth,
                1U << 20, 0, true, packed ? 8 : 32, packed ? 127.5F : 0.0F);
            // The windowed top replaces the tree when the tree histories
            // exceed the budget: try budgets around their size.
            const SizeType hist =
                (plan.get_history_size() + plan.get_tree_history_size()) *
                sizeof(float);
            SizeType budget  = 0;
            uint64_t latency = 0;
            for (const SizeType tenths : {10, 9, 8}) {
                try {
                    FDMTBoxcar probe(plan, cfg, true, exec, 1,
                                     algorithms::kFDMTAutoFuse, 0,
                                     hist * tenths / 10);
                    if (probe.get_search_latency() > 0) {
                        budget  = hist * tenths / 10;
                        latency = probe.get_search_latency();
                        break;
                    }
                } catch (const std::invalid_argument&) {
                }
            }
            INFO("history " << hist << " B");
            // Several blocks of padding.
            REQUIRE(latency > plan.get_nsamps());
            const auto est = FDMTBoxcar::estimate_memory(
                plan, cfg, true, exec, 1, algorithms::kFDMTAutoFuse, 0, budget);
            FDMTBoxcar device(plan, cfg, true, exec, 1,
                              algorithms::kFDMTAutoFuse, 0, budget);
            FDMTBoxcar host(plan, cfg, true, exec, 1, algorithms::kFDMTAutoFuse,
                            0, budget);
            INFO(device.summary());
            REQUIRE(device.get_search_latency() == latency);
            REQUIRE(device.get_memory_usage().total() == est.usage.total());
            thrust::device_vector<float> fbuf(n);
            thrust::device_vector<uint8_t> pbuf(n);
            std::vector<RawCandidate> got;
            std::vector<RawCandidate> want;
            for (SizeType b = 0; b < kBlocks; ++b) {
                if (packed) {
                    thrust::copy(bytes[b].begin(), bytes[b].end(),
                                 pbuf.begin());
                    (void)device.execute(DeviceSpan<const uint8_t>(
                        thrust::raw_pointer_cast(pbuf.data()), n,
                        {exec.backend, 0}));
                    append(want,
                           host.execute(std::span<const uint8_t>(bytes[b])));
                } else {
                    thrust::copy(values[b].begin(), values[b].end(),
                                 fbuf.begin());
                    (void)device.execute(DeviceSpan<const float>(
                        thrust::raw_pointer_cast(fbuf.data()), n,
                        {exec.backend, 0}));
                    append(want,
                           host.execute(std::span<const float>(values[b])));
                }
                gpu_utils::check_gpu_call(cudaDeviceSynchronize());
                append(got, device.copy_result());
                CHECK(device.get_memory_usage().total() == est.usage.total());
            }
            append(got, device.drain());
            append(want, host.drain());
            CHECK(device.get_memory_usage().total() == est.usage.total());
            CHECK(device.get_memory_usage().staging == est.usage.staging);
            CHECK(device.get_searched_nsamps() == kBlocks * plan.get_nsamps());
            REQUIRE(want.size() > 100);
            compare(got, want);
            // reset_history() starts a new stream: an abandoned stream's
            // carried windowed-top spreads must not reach it. They would sit
            // in the cold prefix, so search it.
            auto cold             = cfg;
            cold.skip_cold_prefix = false;
            FDMTBoxcar fresh(plan, cold, true, exec, 1,
                             algorithms::kFDMTAutoFuse, 0, budget);
            FDMTBoxcar reused(plan, cold, true, exec, 1,
                              algorithms::kFDMTAutoFuse, 0, budget);
            REQUIRE(reused.get_search_latency() == latency);
            const auto run = [&](FDMTBoxcar& s, SizeType b) {
                return packed ? s.execute(std::span<const uint8_t>(bytes[b]))
                              : s.execute(std::span<const float>(values[b]));
            };
            // An abandoned loud stream (bright enough to show any carry).
            std::vector<float> loud(values[2]);
            for (auto& v : loud) {
                v *= 1000.0F;
            }
            const std::vector<uint8_t> loud_codes(n, 255);
            for (int k = 0; k < 3; ++k) {
                (void)(packed ? reused.execute(
                                    std::span<const uint8_t>(loud_codes))
                              : reused.execute(std::span<const float>(loud)));
            }
            reused.reset_history();
            std::vector<RawCandidate> first;
            std::vector<RawCandidate> again;
            for (SizeType b = 0; b < kBlocks; ++b) {
                append(first, run(fresh, b));
                append(again, run(reused, b));
            }
            append(first, fresh.drain());
            append(again, reused.drain());
            REQUIRE(first.size() > 100);
            compare(again, first);
        }
    }
}

TEST_CASE("GPU engines with different shared-memory needs coexist",
          "[gpu][boxcar]") {
    // A kernel's dynamic shared-memory limit is shared by every engine in
    // the process: an engine built later with a smaller need must not lower
    // it under one built earlier (whose launches then failed with "invalid
    // argument"). Each engine built first runs as when built alone.
    const auto blocks = [](SizeType n, SizeType count) {
        std::mt19937 gen(5);
        std::normal_distribution<float> nd;
        std::vector<std::vector<uint8_t>> out(count, std::vector<uint8_t>(n));
        for (auto& b : out) {
            for (auto& v : b) {
                v = static_cast<uint8_t>(std::clamp(
                    std::round(128.0F + (10.0F * nd(gen))), 0.0F, 255.0F));
            }
        }
        return out;
    };
    const Exec gpu{algorithms::detail::kGPUBackend, 1, 0};
    const auto run = [](FDMTBoxcar& s,
                        const std::vector<std::vector<uint8_t>>& in) {
        std::vector<RawCandidate> out;
        for (const auto& b : in) {
            append(out, s.execute(std::span<const uint8_t>(b)));
        }
        append(out, s.drain());
        return out;
    };
    SECTION("FDMT direct levels: box smearing, then none") {
        // The skewed direct levels of the smeared transform need more
        // shared memory than the unsmeared one's.
        const float tsamp = 455.6e-6F;
        const auto grid   = plans::DDMTPlan::generate_levin_dm_grid_piecewise(
            0.0F, 600.0F, tsamp, tsamp, 1100.0F, 1714.0F, 1152, 1.1F);
        const plans::FDMTPlan plan(1100.0F, 1714.0F, 1152, 128, tsamp, grid,
                                   "valid", 3);
        BoxcarConfig cfg(24);
        cfg.detect_thresh     = 3.75F;
        cfg.noise_sigma       = 10.0F;
        cfg.input_nbits       = 8;
        cfg.input_mean        = 128.0F;
        cfg.skip_cold_prefix  = false;
        cfg.per_call_masks    = true;
        cfg.max_candidates    = 1U << 20;
        const SizeType budget = 113372721; // the skewed path
        const auto in         = blocks(1152 * 128, 6);
        std::vector<RawCandidate> first;
        {
            FDMTBoxcar a(plan, cfg, true, gpu, 1, algorithms::kFDMTAutoFuse, 0,
                         budget);
            FDMTBoxcar b(plan, cfg, false, gpu, 1, algorithms::kFDMTAutoFuse, 0,
                         budget);
            first = run(a, in);
            (void)run(b, in);
        }
        FDMTBoxcar alone(plan, cfg, true, gpu, 1, algorithms::kFDMTAutoFuse, 0,
                         budget);
        REQUIRE(first.size() > 20);
        compare(first, run(alone, in));
    }
    SECTION("FDMTBoxcar kernels: a wide boxcar, then a narrow one") {
        // 8-bit rows keep a uint64 prefix ring of the boxcar's width in
        // shared memory: 4000 needs over the 48 KiB default, 37 far less.
        const plans::FDMTPlan plan(700, 1200, 32, 400, 0.001F,
                                   std::vector<IndexType>{0, 3, 9, 20, 41},
                                   "valid", 0);
        BoxcarConfig wide(4000);
        wide.detect_thresh  = 3.0F;
        wide.noise_sigma    = 10.0F;
        wide.input_nbits    = 8;
        wide.input_mean     = 128.0F;
        wide.max_candidates = 1U << 20;
        auto narrow         = wide;
        narrow.max_boxcar   = 37;
        const auto in       = blocks(32 * 400, 8);
        std::vector<RawCandidate> first;
        {
            FDMTBoxcar a(plan, wide, true, gpu);
            FDMTBoxcar b(plan, narrow, true, gpu);
            first = run(a, in);
            (void)run(b, in);
        }
        FDMTBoxcar alone(plan, wide, true, gpu);
        REQUIRE(first.size() > 20);
        compare(first, run(alone, in));
    }
}

TEST_CASE("GPU subband searches skip paths beyond the 32-bit index range",
          "[gpu][boxcar][budget]") {
    // UWL to DM 2000 at subband depth 2: the tree history exceeds the CUDA
    // backend's 32-bit index, so the direct entries cannot run and the
    // windowed (skewed) path is taken at any budget.
    const float tsamp = 64e-6F;
    const auto grid   = plans::DDMTPlan::generate_levin_dm_grid_piecewise(
        0.0F, 2000.0F, tsamp, tsamp, 704.0F, 4032.0F, 6656, 1.1F);
    const plans::FDMTPlan plan(704.0F, 4032.0F, 6656, 4096, tsamp, grid,
                               "valid", 2);
    REQUIRE(plan.get_tree_history_size() >
            std::numeric_limits<uint32_t>::max());
    const BoxcarConfig cfg(1562, 1.5F, 6.0F, 12.0F,
                           PeakFindingMode::kContiguousPeakPerWidth, 1U << 20,
                           0, true, 8, 127.0F);
    const Exec exec{algorithms::detail::kGPUBackend, 1, 0};
    for (const SizeType gib : {10, 40}) {
        INFO(gib << " GiB");
        const auto est = FDMTBoxcar::estimate_memory(plan, cfg, true, exec, 1,
                                                     algorithms::kFDMTAutoFuse,
                                                     0, SizeType{gib} << 30);
        CHECK(est.search_path.find("windowed") != std::string::npos);
        CHECK(est.search_latency > 0);
    }
}
