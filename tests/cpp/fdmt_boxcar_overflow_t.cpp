// FDMTBoxcar and FDMTFFTBoxcar overflow policy
// (OverflowPolicy::kKeepBrightest): keeps each row's max_cands_per_dm brightest
// candidates of a call, then the call's max_candidates brightest of those.
// Checked against the full candidate set of an unlimited run, call by call, on
// CPU and (when built) GPU.

#include <algorithm>
#include <cstdint>
#include <map>
#include <random>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "fdmt_boxcar_overflow_reference.hpp"
#include "test_helpers.hpp"

using namespace boxcar_test;

namespace {

using Pairs = std::vector<std::pair<SizeType, SizeType>>; // (q, K)

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

void check_overflow_core(Exec exec, bool is_gpu) {
    const Pairs pairs     = {{3, 1U << 20}, {3, 150}, {1, 40}, {50, 300}};
    const SizeType nsamps = is_gpu ? 2048 : 512;
    for (const SizeType depth : {SizeType{0}, SizeType{1}}) {
        const plans::FDMTPlan plan(700, 1200, 16, nsamps, 0.001F,
                                   std::vector<IndexType>{0, 3, 9, 20, 41},
                                   "valid", depth);
        const auto input = waterfalls(plan, 2, 3);
        const auto codes = to_codes(input, 16.0F, 64.0F);
        const std::vector<SizeType> chunks =
            is_gpu ? std::vector<SizeType>{64, 512, plan.get_nsamps()}
                   : std::vector<SizeType>{0};
        for (const bool packed : {false, true}) {
            for (const auto mode :
                 {PeakFindingMode::kContiguousPeakPerWidth,
                  PeakFindingMode::kContiguousPeakAcrossWidths,
                  PeakFindingMode::kAllAboveThreshold}) {
                for (const SizeType chunk : chunks) {
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
                            FDMTBoxcar s(plan, c, true, exec, 2,
                                         algorithms::kFDMTAutoFuse, chunk);
                            return packed ? run_calls(s, codes, n)
                                          : run_calls(s, input, n);
                        };
                        CHECK(check_keep(run, cfg, pairs) > 400);
                        if (is_gpu && packed) {
                            auto keep = cfg;
                            keep.overflow_policy =
                                OverflowPolicy::kKeepBrightest;
                            keep.max_cands_per_dm = 3;
                            keep.max_candidates   = 150;
                            FDMTBoxcar cpu(plan, keep, true, Exec::cpu(), 2);
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

void check_fft_overflow(Exec exec) {
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
        algorithms::FDMTFFTBoxcar s(plan, c, true, exec);
        return run_calls(s, input, n);
    };
    CHECK(check_keep(run, cfg, {{4, 60}, {2, 10}}) > 400);
}

} // namespace

TEST_CASE("FDMTBoxcar kKeepBrightest keeps the brightest of each call (CPU)",
          "[boxcar][overflow][cpu]") {
    check_overflow_core(Exec::cpu(), false);
}

TEST_CASE("FDMTBoxcar kKeepBrightest finds a late pulse after an early storm",
          "[boxcar][overflow][cpu]") {
    const plans::FDMTPlan plan(1200, 1500, 64, 4096, 0.001F, 63, 0, 1, "valid",
                               1);
    std::mt19937 gen(5);
    std::normal_distribution<float> noise(0.0F, 1.0F);
    std::vector<float> block(plan.get_nchans() * plan.get_nsamps());
    for (SizeType c = 0; c < plan.get_nchans(); ++c) {
        for (SizeType t = 0; t < plan.get_nsamps(); ++t) {
            float v = noise(gen);
            if (t >= 100 && t < 1100) {
                v += 1.5F;
            }
            if (t >= 3500 && t < 3503) {
                v += 12.0F;
            }
            block[(c * plan.get_nsamps()) + t] = v;
        }
    }
    BoxcarConfig cfg;
    cfg.max_boxcar       = 16;
    cfg.detect_thresh    = 6.0F;
    cfg.peak_mode        = PeakFindingMode::kAllAboveThreshold;
    cfg.max_cands_per_dm = 8;
    cfg.max_candidates   = 400;
    const auto pulse     = [](const std::vector<RawCandidate>& v) {
        return std::ranges::any_of(v, [](const RawCandidate& c) {
            return c.peak_sample >= 3500 && c.peak_sample < 3520;
        });
    };
    FDMTBoxcar first(plan, cfg, true, Exec::cpu());
    const auto dropped = first.execute(std::span<const float>(block));
    REQUIRE(dropped.overflowed());
    CHECK_FALSE(pulse({dropped.candidates.begin(), dropped.candidates.end()}));

    cfg.overflow_policy = OverflowPolicy::kKeepBrightest;
    FDMTBoxcar bright(plan, cfg, true, Exec::cpu());
    const auto kept = bright.execute(std::span<const float>(block));
    REQUIRE(kept.overflowed());
    const std::vector<RawCandidate> v(kept.candidates.begin(),
                                      kept.candidates.end());
    CHECK(v.size() == cfg.max_candidates);
    REQUIRE(pulse(v));
    const auto best = std::ranges::min_element(v, brighter);
    CHECK(best->peak_sample >= 3500);
    CHECK(best->peak_sample < 3520);
    CHECK(kept.total_detected == dropped.total_detected);
    CHECK(kept.total_detected ==
          kept.emitted_count + kept.dropped_per_dm + kept.dropped_global);
}

TEST_CASE("FDMTBoxcar kKeepBrightest: validation and memory",
          "[boxcar][overflow][cpu]") {
    const plans::FDMTPlan plan(700, 1200, 16, 512, 0.001F, 41);
    BoxcarConfig cfg;
    cfg.max_boxcar       = 9;
    cfg.overflow_policy  = OverflowPolicy::kKeepBrightest;
    cfg.max_cands_per_dm = 0;
    CHECK_THROWS_AS(FDMTBoxcar(plan, cfg), std::invalid_argument);
    cfg.max_cands_per_dm = 16;
    const FDMTBoxcar kept(plan, cfg);
    cfg.overflow_policy = OverflowPolicy::kDropNew;
    const FDMTBoxcar plain(plan, cfg);
    const SizeType rows = plain.get_search_plans().front().dm_grid.size();
    CHECK(kept.get_memory_usage().candidates ==
          plain.get_memory_usage().candidates +
              (rows * 16 * sizeof(RawCandidate)));
    cfg.overflow_policy  = OverflowPolicy::kKeepBrightest;
    cfg.max_cands_per_dm = 4096;
    CHECK_THROWS_AS(FDMTBoxcar(plan, cfg), std::invalid_argument);
    cfg.overflow_policy   = OverflowPolicy::kDropNew;
    cfg.max_cands_per_dm  = 1024;
    const SizeType store  = rows * 1024 * sizeof(RawCandidate);
    const SizeType budget = plain.get_memory_usage().budgeted() + (store / 2);
    CHECK_NOTHROW(FDMTBoxcar(plan, cfg, true, Exec{}, 1,
                             algorithms::kFDMTAutoFuse, 0, budget));
    cfg.overflow_policy = OverflowPolicy::kKeepBrightest;
    CHECK_THROWS_AS(FDMTBoxcar(plan, cfg, true, Exec{}, 1,
                               algorithms::kFDMTAutoFuse, 0, budget),
                    std::invalid_argument);
}

TEST_CASE("drain() keeps the call's quotas over its whole tail (CPU)",
          "[boxcar][overflow][cpu]") {
    // FDMTFFTBoxcar with fractional delays drains its 64-sample latency in
    // four 16-sample padding steps and flush(): one call, whose quotas hold
    // over the union of the steps.
    const plans::FDMTPlan plan(700, 1200, 8, 16, 0.001F,
                               std::vector<IndexType>{0, 3, 9, 20});
    const auto input = waterfalls(plan, 1, 6);
    BoxcarConfig base;
    base.max_boxcar    = 8;
    base.wtsp          = 1.0F;
    base.detect_thresh = 1.0F;
    base.peak_mode     = PeakFindingMode::kAllAboveThreshold;
    const auto drain   = [&](const BoxcarConfig& c, CandidateCounts& counts) {
        algorithms::FDMTFFTBoxcar s(plan, c, true, Exec::cpu(), 1, true);
        REQUIRE(s.get_output_latency() > 3 * plan.get_nsamps());
        for (const auto& block : input) {
            static_cast<void>(s.execute(block));
        }
        const auto tail  = s.drain();
        counts           = tail;
        SizeType emitted = 0;
        for (const auto& p : tail.plane_counts) {
            CHECK(p.emitted_count + p.dropped_global + p.dropped_per_dm ==
                  p.total_detected);
            emitted += p.emitted_count;
        }
        CHECK(emitted == tail.emitted_count);
        CHECK(tail.candidates.size() == tail.emitted_count);
        CHECK(tail.emitted_count + tail.dropped_global + tail.dropped_per_dm ==
              tail.total_detected);
        return std::vector<RawCandidate>(tail.candidates.begin(),
                                         tail.candidates.end());
    };
    auto open             = base;
    open.max_candidates   = SizeType{1} << 20;
    open.max_cands_per_dm = 0;
    CandidateCounts all_counts;
    const auto all = drain(open, all_counts);
    REQUIRE(all.size() > 60);
    REQUIRE_FALSE(all_counts.overflowed());

    SECTION("kDropNew max_candidates") {
        auto cfg           = open;
        cfg.max_candidates = 25;
        CandidateCounts counts;
        const auto got = drain(cfg, counts);
        CHECK(got.size() == 25);
        CHECK(counts.total_detected == all.size());
        CHECK(counts.dropped_global == all.size() - 25);
    }
    SECTION("kDropNew max_candidates_per_plane") {
        auto cfg                     = open;
        cfg.max_candidates           = 40;
        cfg.max_candidates_per_plane = 15;
        CandidateCounts counts;
        CHECK(drain(cfg, counts).size() == 15);
    }
    SECTION("kKeepBrightest: the brightest of the whole tail") {
        auto cfg             = base;
        cfg.overflow_policy  = OverflowPolicy::kKeepBrightest;
        cfg.max_cands_per_dm = 1024; // above any row's count here
        cfg.max_candidates   = 20;
        CandidateCounts counts;
        const auto got = drain(cfg, counts);
        CandidateCounts want;
        require_same(got, keep_brightest(all, 1024, 20, want));
        CHECK(counts.total_detected == want.total_detected);
        CHECK(counts.emitted_count == want.emitted_count);
        CHECK(counts.dropped_global == want.dropped_global);
    }
    SECTION("kKeepBrightest max_cands_per_dm") {
        auto cfg             = base;
        cfg.overflow_policy  = OverflowPolicy::kKeepBrightest;
        cfg.max_cands_per_dm = 2;
        cfg.max_candidates   = 1000;
        CandidateCounts counts;
        const auto got = drain(cfg, counts);
        std::map<uint32_t, SizeType> per_row;
        for (const auto& c : got) {
            CHECK(++per_row[c.dm_idx] <= 2);
        }
    }
}

TEST_CASE("FDMTFFTBoxcar kKeepBrightest keeps the brightest of each call (CPU)",
          "[boxcar][overflow][cpu]") {
    check_fft_overflow(Exec::cpu());
}

#ifdef DMT_ENABLE_GPU
TEST_CASE("FDMTBoxcar kKeepBrightest keeps the brightest of each call (GPU)",
          "[gpu][boxcar][overflow]") {
    const Exec gpu = test::gpu_exec();
    check_overflow_core(gpu, true);
}

TEST_CASE("GPU FDMTBoxcar kKeepBrightest: screen, masks, wide and 16-bit",
          "[gpu][boxcar][overflow]") {
    const Exec gpu = test::gpu_exec();
    SECTION("width screen") {
        const plans::FDMTPlan plan(700, 1200, 16, 4096, 0.001F,
                                   std::vector<IndexType>{0, 3, 9, 20, 41});
        const auto input = waterfalls(plan, 1, 3);
        BoxcarConfig cfg;
        cfg.max_boxcar    = 24;
        cfg.wtsp          = 1.0F;
        cfg.detect_thresh = 2.5F;
        const auto run    = [&](const BoxcarConfig& c,
                                std::vector<CandidateCounts>* n) {
            FDMTBoxcar s(plan, c, true, gpu);
            return run_calls(s, input, n);
        };
        CHECK(check_keep(run, cfg, {{4, 200}, {2, 60}}) > 400);
    }
    SECTION("per-call kill masks and wide boxcar (max_boxcar 6000)") {
        const plans::FDMTPlan plan(700, 1200, 32, 8192, 0.001F,
                                   std::vector<IndexType>{0, 3, 9, 20, 41});
        const auto input = waterfalls(plan, 1, 3);
        std::vector<std::vector<uint8_t>> masks;
        std::mt19937 gen(19);
        std::bernoulli_distribution kill(0.25);
        for (SizeType k = 0; k < input.size(); ++k) {
            std::vector<uint8_t> m(plan.get_nchans());
            for (auto& v : m) {
                v = kill(gen) ? 0 : 1;
            }
            masks.push_back(std::move(m));
        }
        BoxcarConfig cfg;
        cfg.max_boxcar    = 6000;
        cfg.wtsp          = 2.0F;
        cfg.detect_thresh = 2.0F;
        const auto run    = [&](const BoxcarConfig& c,
                                std::vector<CandidateCounts>* n) {
            FDMTBoxcar s(plan, c, true, gpu);
            return run_calls_masked(s, input, masks, n);
        };
        CHECK(check_keep(run, cfg, {{3, 100}, {1, 30}}) > 400);
    }
    SECTION("16-bit packed input (uint64 carry)") {
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
            FDMTBoxcar s(plan, c, true, gpu, 1, algorithms::kFDMTAutoFuse, 512);
            return run_calls(s, blocks, n);
        };
        CHECK(check_keep(run, cfg, {{3, 60}, {1, 20}}) > 100);
    }
    SECTION("16-bit packed input beyond 2^32 (uint64 carry)") {
        // The plan above fits uint32_t (32-bit carry); this one's 16-bit
        // sums exceed 2^32 (smeared delays to 16384; dmt_internal_tests
        // checks int_search_fits_u32 is false for it), so the search runs
        // its uint64_t carry.
        const plans::FDMTPlan plan(700, 1200, 8, 1024, 0.001F,
                                   std::vector<IndexType>{0, 32, 16384});
        std::mt19937 gen(5);
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
        cfg.max_boxcar    = 64;
        cfg.wtsp          = 2.0F;
        cfg.detect_thresh = 2.0F;
        cfg.input_nbits   = 16;
        cfg.input_mean    = 32767.5F;
        cfg.noise_sigma   = 18918.0F;
        const auto run    = [&](const BoxcarConfig& c,
                                std::vector<CandidateCounts>* n) {
            FDMTBoxcar s(plan, c, true, gpu);
            return run_calls(s, blocks, n);
        };
        CHECK(check_keep(run, cfg, {{3, 60}, {1, 20}}) > 100);
    }
}

TEST_CASE("GPU FDMTBoxcar kKeepBrightest cuts ties across beams and rows",
          "[gpu][boxcar][overflow]") {
    const Exec gpu = test::gpu_exec();
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
        FDMTBoxcar s(plan, c, true, gpu, 2);
        return run_calls(s, blocks, n);
    };
    const Pairs pairs = {{40, 77}, {40, 501}, {7, 13}, {300, 1999}};
    CHECK(check_keep(run, cfg, pairs) > 2000);
    for (const auto& [q, k] : pairs) {
        auto keep             = cfg;
        keep.overflow_policy  = OverflowPolicy::kKeepBrightest;
        keep.max_cands_per_dm = q;
        keep.max_candidates   = k;
        FDMTBoxcar cpu(plan, keep, true, Exec::cpu(), 2);
        const auto expected = run_calls(cpu, blocks);
        const auto actual   = run(keep, nullptr);
        for (SizeType i = 0; i < actual.size(); ++i) {
            require_same(actual[i], expected[i]);
        }
    }
}

TEST_CASE("FDMTFFTBoxcar kKeepBrightest keeps the brightest of each call (GPU)",
          "[gpu][boxcar][overflow]") {
    check_fft_overflow(test::gpu_exec());
}
#endif
