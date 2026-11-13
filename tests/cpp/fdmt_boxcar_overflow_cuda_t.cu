// FDMTBoxcar overflow policy on the GPU: OverflowPolicy::kKeepBrightest per
// call against the reference built from the GPU's own unlimited run (float
// S/N may differ from the CPU's by rounding at the q-th / K-th boundary;
// exact-sum packed input is also compared with the CPU). Covers both
// emission paths of the scoring kernel (all mode, run ends), the flush
// kernel, the three block sizes, calls of several chunks, the width
// screen, per-call masks, both packed carries, wide boxcars, ties cut by
// the global cap, and FDMTFFTBoxcar.

#include <algorithm>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

#include "dmt/engines.hpp"

#include "fdmt_boxcar_overflow_reference.hpp"

using namespace boxcar_test;

namespace {

const Exec kGpu{algorithms::detail::kGPUBackend, 1, 0};

using Pairs = std::vector<std::pair<SizeType, SizeType>>; // (q, K)

// Every (q, K) of `pairs`: run(cfg, counts) under kKeepBrightest against the
// reference of run() without caps. Returns the candidates of the open run.
template <typename Run>
SizeType check_keep(Run run, const BoxcarConfig& base, const Pairs& pairs) {
    auto open             = base;
    open.max_candidates   = SizeType{1} << 20;
    open.max_cands_per_dm = 0;
    const Calls all       = run(open, nullptr);
    SizeType total        = 0;
    for (const auto& c : all) {
        total += c.size();
    }
    for (const auto& [q, k] : pairs) {
        auto cfg             = base;
        cfg.overflow_policy  = OverflowPolicy::kKeepBrightest;
        cfg.max_cands_per_dm = q;
        cfg.max_candidates   = k;
        std::vector<CandidateCounts> counts;
        const Calls kept = run(cfg, &counts);
        REQUIRE(kept.size() == all.size());
        for (SizeType i = 0; i < all.size(); ++i) {
            INFO("q " << q << " K " << k << " call " << i);
            CandidateCounts want;
            require_same(kept[i], keep_brightest(all[i], q, k, want));
            CHECK(counts[i].total_detected == want.total_detected);
            CHECK(counts[i].emitted_count == want.emitted_count);
            CHECK(counts[i].dropped_per_dm == want.dropped_per_dm);
            CHECK(counts[i].dropped_global == want.dropped_global);
        }
    }
    return total;
}

std::vector<std::vector<uint8_t>> to_codes(
    const std::vector<std::vector<float>>& input, float scale, float mean) {
    std::vector<std::vector<uint8_t>> out;
    for (const auto& b : input) {
        std::vector<uint8_t> codes(b.size());
        for (SizeType i = 0; i < b.size(); ++i) {
            codes[i] = static_cast<uint8_t>(
                std::clamp(std::lround((b[i] * scale) + mean), 0L, 255L));
        }
        out.push_back(std::move(codes));
    }
    return out;
}

} // namespace

TEST_CASE("GPU FDMTBoxcar kKeepBrightest keeps the brightest of each call",
          "[gpu][boxcar][overflow]") {
    // Chunks of 64, 512 and the whole block select 64-, 128- and
    // 256-thread scoring blocks; the stores persist across a call's chunks.
    const Pairs pairs = {{3, 1U << 20}, {3, 150}, {1, 40}, {50, 300}};
    for (const SizeType depth : {SizeType{0}, SizeType{1}}) {
        const plans::FDMTPlan plan(700, 1200, 16, 2048, 0.001F,
                                   std::vector<IndexType>{0, 3, 9, 20, 41},
                                   "valid", depth);
        const auto input = waterfalls(plan, 2, 3);
        const auto codes = to_codes(input, 16.0F, 64.0F);
        for (const bool packed : {false, true}) {
            for (const auto mode :
                 {PeakFindingMode::kContiguousPeakPerWidth,
                  PeakFindingMode::kContiguousPeakAcrossWidths,
                  PeakFindingMode::kAllAboveThreshold}) {
                for (const SizeType chunk :
                     {SizeType{64}, SizeType{512}, plan.get_nsamps()}) {
                    DYNAMIC_SECTION("depth "
                                    << depth << " packed " << packed << " mode "
                                    << static_cast<int>(mode) << " chunk "
                                    << chunk) {
                        BoxcarConfig cfg;
                        cfg.max_boxcar    = 9;
                        cfg.wtsp          = 1.5F;
                        cfg.detect_thresh = 2.0F;
                        cfg.peak_mode     = mode;
                        if (packed) {
                            cfg.input_nbits = 8;
                            cfg.input_mean  = 64.0F;
                            cfg.noise_sigma = 16.0F;
                        }
                        const auto run = [&](const BoxcarConfig& c,
                                             std::vector<CandidateCounts>* n) {
                            FDMTBoxcar s(plan, c, true, kGpu, 2,
                                         algorithms::kFDMTAutoFuse, chunk);
                            return packed ? run_calls(s, codes, n)
                                          : run_calls(s, input, n);
                        };
                        CHECK(check_keep(run, cfg, pairs) > 400);
                        if (packed) {
                            // Exact sums: the GPU's kept set is the CPU's.
                            auto keep = cfg;
                            keep.overflow_policy =
                                OverflowPolicy::kKeepBrightest;
                            keep.max_cands_per_dm = 3;
                            keep.max_candidates   = 150;
                            FDMTBoxcar cpu(plan, keep, true, Exec::cpu(3), 2);
                            const auto expected = run_calls(cpu, codes);
                            const auto actual   = run(keep, nullptr);
                            for (SizeType i = 0; i < actual.size(); ++i) {
                                require_same(actual[i], expected[i]);
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("GPU FDMTBoxcar kKeepBrightest: screen, masks, wide and 16-bit "
          "boxcars",
          "[gpu][boxcar][overflow]") {
    SECTION("width screen (every width to 24, blocks of several tiles)") {
        const plans::FDMTPlan plan(700, 1200, 16, 4096, 0.001F,
                                   std::vector<IndexType>{0, 3, 9, 20, 41});
        const auto input = waterfalls(plan, 1, 3);
        BoxcarConfig cfg;
        cfg.max_boxcar    = 24;
        cfg.wtsp          = 1.0F;
        cfg.detect_thresh = 2.5F;
        const auto run    = [&](const BoxcarConfig& c,
                                std::vector<CandidateCounts>* n) {
            FDMTBoxcar s(plan, c, true, kGpu);
            return run_calls(s, input, n);
        };
        CHECK(check_keep(run, cfg, {{4, 200}, {2, 60}}) > 400);
    }
    SECTION("per-call kill masks") {
        const plans::FDMTPlan plan(700, 1200, 32, 1024, 0.001F,
                                   std::vector<IndexType>{0, 3, 9, 20, 41});
        const auto input = waterfalls(plan, 1, 4);
        std::vector<std::vector<uint8_t>> masks;
        std::mt19937 gen(9);
        std::bernoulli_distribution kill(0.25);
        for (SizeType k = 0; k < input.size(); ++k) {
            std::vector<uint8_t> m(plan.get_nchans());
            for (auto& v : m) {
                v = kill(gen) ? 0 : 1;
            }
            masks.push_back(std::move(m));
        }
        BoxcarConfig cfg;
        cfg.max_boxcar    = 16;
        cfg.detect_thresh = 2.0F;
        const auto run    = [&](const BoxcarConfig& c,
                                std::vector<CandidateCounts>* n) {
            FDMTBoxcar s(plan, c, true, kGpu);
            return run_calls_masked(s, input, masks, n);
        };
        CHECK(check_keep(run, cfg, {{3, 100}, {1, 30}}) > 400);
    }
    SECTION("max_boxcar 4096") {
        const plans::FDMTPlan plan(700, 1200, 3, 4500, 0.001F,
                                   std::vector<IndexType>{-5, 0, 7}, "valid",
                                   1);
        const auto input = waterfalls(plan, 1, 3);
        BoxcarConfig cfg;
        cfg.max_boxcar    = 4096;
        cfg.wtsp          = 2.0F;
        cfg.detect_thresh = 2.0F;
        const auto run    = [&](const BoxcarConfig& c,
                                std::vector<CandidateCounts>* n) {
            FDMTBoxcar s(plan, c, true, kGpu, 1, algorithms::kFDMTAutoFuse,
                         128);
            return run_calls(s, input, n);
        };
        CHECK(check_keep(run, cfg, {{3, 40}, {1, 10}}) > 100);
    }
    SECTION("16-bit packed input (uint64 carry)") {
        // 64 channels of 16-bit codes with boxcars to 1100: tree sums
        // exceed 32 bits, so the search carries uint64.
        const plans::FDMTPlan plan(700, 1200, 64, 2048, 0.001F,
                                   std::vector<IndexType>{0, 5, 17});
        std::mt19937 gen(4);
        std::uniform_int_distribution<int> dist(0, 65535);
        std::vector<std::vector<uint8_t>> blocks;
        for (int k = 0; k < 3; ++k) {
            std::vector<uint8_t> b(plan.get_nchans() * plan.get_nsamps() * 2);
            for (SizeType i = 0; i < b.size(); i += 2) {
                const auto v = static_cast<uint16_t>(dist(gen));
                b[i]         = static_cast<uint8_t>(v & 0xFF);
                b[i + 1]     = static_cast<uint8_t>(v >> 8);
            }
            blocks.push_back(std::move(b));
        }
        BoxcarConfig cfg;
        cfg.max_boxcar    = 1100;
        cfg.wtsp          = 2.0F;
        cfg.detect_thresh = 2.0F;
        cfg.input_nbits   = 16;
        cfg.input_mean    = 32767.5F;
        cfg.noise_sigma   = 18918.0F;
        const auto run    = [&](const BoxcarConfig& c,
                                std::vector<CandidateCounts>* n) {
            FDMTBoxcar s(plan, c, true, kGpu, 1, algorithms::kFDMTAutoFuse,
                         512);
            return run_calls(s, blocks, n);
        };
        CHECK(check_keep(run, cfg, {{3, 60}, {1, 20}}) > 100);
    }
}

TEST_CASE("GPU FDMTBoxcar kKeepBrightest cuts ties across beams and rows "
          "exactly",
          "[gpu][boxcar][overflow]") {
    // Two beams with identical packed input and a constant elevated span:
    // within a row every sample of the span has the same S/N, and the
    // beams tie row for row. A global cap through the tie group exercises
    // every byte of the selection key; packed sums are exact, so the GPU's
    // set must equal the CPU's.
    const plans::FDMTPlan plan(700, 1200, 16, 1024, 0.001F,
                               std::vector<IndexType>{0, 3, 9}, "valid", 1);
    std::vector<uint8_t> block(2 * plan.get_nchans() * plan.get_nsamps(), 60);
    for (SizeType b = 0; b < 2; ++b) {
        for (SizeType c = 0; c < plan.get_nchans(); ++c) {
            for (SizeType t = 300; t < 500; ++t) {
                block[((b * plan.get_nchans() + c) * plan.get_nsamps()) + t] =
                    90;
            }
        }
    }
    const std::vector<std::vector<uint8_t>> blocks(2, block);
    BoxcarConfig cfg;
    cfg.max_boxcar    = 8;
    cfg.wtsp          = 2.0F;
    cfg.detect_thresh = 3.0F;
    cfg.input_nbits   = 8;
    cfg.input_mean    = 60.0F;
    cfg.noise_sigma   = 4.0F;
    cfg.peak_mode     = PeakFindingMode::kAllAboveThreshold;
    const auto run    = [&](const BoxcarConfig& c,
                            std::vector<CandidateCounts>* n) {
        FDMTBoxcar s(plan, c, true, kGpu, 2);
        return run_calls(s, blocks, n);
    };
    const Pairs pairs = {{40, 77}, {40, 501}, {7, 13}, {300, 1999}};
    CHECK(check_keep(run, cfg, pairs) > 2000);
    for (const auto& [q, k] : pairs) {
        auto keep             = cfg;
        keep.overflow_policy  = OverflowPolicy::kKeepBrightest;
        keep.max_cands_per_dm = q;
        keep.max_candidates   = k;
        FDMTBoxcar cpu(plan, keep, true, Exec::cpu(3), 2);
        const auto expected = run_calls(cpu, blocks);
        const auto actual   = run(keep, nullptr);
        for (SizeType i = 0; i < actual.size(); ++i) {
            require_same(actual[i], expected[i]);
        }
    }
}

TEST_CASE("GPU FDMTFFTBoxcar kKeepBrightest keeps the brightest of each call",
          "[gpu][boxcar][overflow]") {
    const plans::FDMTPlan plan(700, 1200, 16, 512, 0.001F,
                               std::vector<IndexType>{0, 3, 9, 20, 41});
    const auto input = waterfalls(plan, 1, 3);
    BoxcarConfig cfg;
    cfg.max_boxcar    = 9;
    cfg.wtsp          = 1.5F;
    cfg.detect_thresh = 2.0F;
    cfg.peak_mode     = PeakFindingMode::kAllAboveThreshold;
    const auto run    = [&](const BoxcarConfig& c,
                            std::vector<CandidateCounts>* n) {
        algorithms::FDMTFFTBoxcar s(plan, c, true, kGpu);
        return run_calls(s, input, n);
    };
    CHECK(check_keep(run, cfg, {{4, 60}, {2, 10}}) > 400);
}
