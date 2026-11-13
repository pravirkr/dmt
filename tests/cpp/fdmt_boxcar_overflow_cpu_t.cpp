// FDMTBoxcar overflow policy on the CPU: OverflowPolicy::kKeepBrightest keeps
// exactly each row's max_cands_per_dm brightest candidates of a call, then
// the call's max_candidates brightest of those. Checked against the full
// candidate set of an unlimited run (kDropNew without caps), call by call.

#include <algorithm>
#include <cstdint>
#include <map>
#include <random>
#include <span>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "fdmt_boxcar_overflow_reference.hpp"

using namespace boxcar_test;

TEST_CASE("FDMTBoxcar kKeepBrightest keeps the brightest of each call, "
          "exactly",
          "[boxcar][overflow][cpu]") {
    for (const SizeType depth : {SizeType{0}, SizeType{1}}) {
        const plans::FDMTPlan plan(700, 1200, 16, 512, 0.001F,
                                   std::vector<IndexType>{0, 3, 9, 20, 41},
                                   "valid", depth);
        for (const bool packed : {false, true}) {
            for (const auto mode :
                 {PeakFindingMode::kContiguousPeakPerWidth,
                  PeakFindingMode::kContiguousPeakAcrossWidths,
                  PeakFindingMode::kAllAboveThreshold}) {
                DYNAMIC_SECTION("depth " << depth << " packed " << packed
                                         << " mode "
                                         << static_cast<int>(mode)) {
                    constexpr SizeType kBeams = 2;
                    BoxcarConfig cfg;
                    cfg.max_boxcar    = 9;
                    cfg.wtsp          = 1.5F;
                    cfg.detect_thresh = 2.0F;
                    cfg.peak_mode     = mode;
                    std::vector<std::vector<float>> floats;
                    std::vector<std::vector<uint8_t>> bytes;
                    const auto input = waterfalls(plan, kBeams, 3);
                    if (packed) {
                        cfg.input_nbits = 8;
                        cfg.input_mean  = 64.0F;
                        cfg.noise_sigma = 16.0F;
                        for (const auto& b : input) {
                            std::vector<uint8_t> codes(b.size());
                            for (SizeType i = 0; i < b.size(); ++i) {
                                codes[i] = static_cast<uint8_t>(std::clamp(
                                    std::lround((b[i] * 16.0F) + 64.0F), 0L,
                                    255L));
                            }
                            bytes.push_back(std::move(codes));
                        }
                    }
                    const auto calls_of =
                        [&](const BoxcarConfig& c,
                            std::vector<CandidateCounts>* counts) {
                            FDMTBoxcar search(plan, c, true, Exec::cpu(3),
                                              kBeams);
                            return packed ? run_calls(search, bytes, counts)
                                          : run_calls(search, input, counts);
                        };
                    auto open             = cfg;
                    open.max_candidates   = SizeType{1} << 20;
                    open.max_cands_per_dm = 0;
                    const auto all        = calls_of(open, nullptr);
                    SizeType total        = 0;
                    for (const auto& c : all) {
                        total += c.size();
                    }
                    REQUIRE(total > 400);
                    for (const auto& [q, k] :
                         {std::pair<SizeType, SizeType>{3, 1U << 20},
                          {3, 150},
                          {1, 40},
                          {50, 300}}) {
                        auto cfg_k             = cfg;
                        cfg_k.overflow_policy  = OverflowPolicy::kKeepBrightest;
                        cfg_k.max_cands_per_dm = q;
                        cfg_k.max_candidates   = k;
                        std::vector<CandidateCounts> counts;
                        const auto kept = calls_of(cfg_k, &counts);
                        REQUIRE(kept.size() == all.size());
                        for (SizeType i = 0; i < all.size(); ++i) {
                            CandidateCounts want;
                            require_same(kept[i],
                                         keep_brightest(all[i], q, k, want));
                            CHECK(counts[i].total_detected ==
                                  want.total_detected);
                            CHECK(counts[i].emitted_count ==
                                  want.emitted_count);
                            CHECK(counts[i].dropped_per_dm ==
                                  want.dropped_per_dm);
                            CHECK(counts[i].dropped_global ==
                                  want.dropped_global);
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("FDMTBoxcar kKeepBrightest finds a late pulse after an early storm",
          "[boxcar][overflow][cpu]") {
    // The first quarter of the block is a broadband storm (every row far
    // over the threshold for 1000 samples); a bright pulse comes at 3500.
    // First come fills every row's quota and the global buffer with the
    // storm and loses the pulse; the brightest-first policy keeps it.
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
    FDMTBoxcar first(plan, cfg, true, Exec::cpu(3));
    const auto dropped = first.execute(std::span<const float>(block));
    REQUIRE(dropped.overflowed());
    CHECK_FALSE(pulse({dropped.candidates.begin(), dropped.candidates.end()}));

    cfg.overflow_policy = OverflowPolicy::kKeepBrightest;
    FDMTBoxcar bright(plan, cfg, true, Exec::cpu(3));
    const auto kept = bright.execute(std::span<const float>(block));
    REQUIRE(kept.overflowed());
    const std::vector<RawCandidate> v(kept.candidates.begin(),
                                      kept.candidates.end());
    CHECK(v.size() == cfg.max_candidates);
    REQUIRE(pulse(v));
    // The brightest candidate of the call is the pulse.
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
    // Above 1024 per row (the kDropNew default 4096) is rejected.
    cfg.overflow_policy  = OverflowPolicy::kKeepBrightest;
    cfg.max_cands_per_dm = 4096;
    CHECK_THROWS_AS(FDMTBoxcar(plan, cfg), std::invalid_argument);
    cfg.overflow_policy = OverflowPolicy::kDropNew;
    // The store counts in the budget: a budget half a store (rows x 1024
    // candidates) above the kDropNew search cannot hold it.
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

TEST_CASE("FDMTFFTBoxcar kKeepBrightest keeps the brightest of each call",
          "[boxcar][overflow][cpu]") {
    using algorithms::FDMTFFTBoxcar;
    const plans::FDMTPlan plan(700, 1200, 16, 512, 0.001F,
                               std::vector<IndexType>{0, 3, 9, 20, 41});
    const auto input = waterfalls(plan, 1, 3);
    BoxcarConfig cfg;
    cfg.max_boxcar        = 9;
    cfg.wtsp              = 1.5F;
    cfg.detect_thresh     = 2.0F;
    cfg.peak_mode         = PeakFindingMode::kAllAboveThreshold;
    auto open             = cfg;
    open.max_candidates   = SizeType{1} << 20;
    open.max_cands_per_dm = 0;
    FDMTFFTBoxcar all_search(plan, open, true, Exec::cpu(3));
    const auto all       = run_calls(all_search, input);
    cfg.overflow_policy  = OverflowPolicy::kKeepBrightest;
    cfg.max_cands_per_dm = 4;
    cfg.max_candidates   = 60;
    FDMTFFTBoxcar kept_search(plan, cfg, true, Exec::cpu(3));
    const auto kept = run_calls(kept_search, input);
    REQUIRE(kept.size() == all.size());
    SizeType dropped = 0;
    for (SizeType i = 0; i < all.size(); ++i) {
        CandidateCounts want;
        require_same(kept[i], keep_brightest(all[i], 4, 60, want));
        dropped += want.dropped_per_dm + want.dropped_global;
    }
    CHECK(dropped > 0);
}
