// get_candidate_frontier() of FDMTBoxcar and FDMTFFTBoxcar: every candidate
// a later call emits begins at or after it, it never decreases, a long
// excursion holds it at its begin, and it equals the searched position
// after flush() and drain(). CPU and GPU (the GPU value is the CPU's).

#include <algorithm>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

#include "dmt/engines.hpp"

#include "fdmt_boxcar_reference.hpp"

using namespace boxcar_test;

namespace {

using Masks = std::vector<std::vector<uint8_t>>;

plans::FDMTPlan frontier_plan() {
    return {700,     1200,   32,
            400,     0.001F, std::vector<IndexType>{0, 3, 9, 20, 41, 77, 120},
            "valid", 2};
}

BoxcarConfig frontier_config(PeakFindingMode mode) {
    BoxcarConfig cfg;
    cfg.max_boxcar       = 37;
    cfg.wtsp             = 1.5F;
    cfg.detect_thresh    = 2.5F; // many noise excursions
    cfg.peak_mode        = mode;
    cfg.max_candidates   = 1U << 20;
    cfg.max_cands_per_dm = 0;
    return cfg;
}

Masks frontier_masks(SizeType blocks, bool masked) {
    Masks masks(blocks);
    if (!masked) {
        return masks;
    }
    for (SizeType k = 0; k < blocks; ++k) {
        if (k % 3 == 2) {
            continue; // the constructor mask alone
        }
        std::mt19937 gen(40 + static_cast<unsigned>(k));
        masks[k].resize(32);
        for (auto& v : masks[k]) {
            v = (gen() % 4) != 0 ? 1 : 0;
        }
    }
    return masks;
}

struct FrontierRun {
    std::vector<uint64_t> frontier; ///< before each call, then at the end
    std::vector<RawCandidate> candidates;
};

// Streams `input` (one beam), checking after every call that the frontier
// read before it bounds the call's candidates and never decreases.
template <typename Search>
FrontierRun stream(Search& search,
                   const std::vector<std::vector<float>>& input,
                   const Masks& masks,
                   bool drain) {
    FrontierRun run;
    uint64_t before = search.get_candidate_frontier();
    REQUIRE(before == 0);
    const auto check = [&](const CandidateResultView& r) {
        for (const auto& c : r.candidates) {
            INFO("candidate begin " << c.begin_sample << " frontier "
                                    << before);
            REQUIRE(c.begin_sample >= before);
        }
        append(run.candidates, r);
        const uint64_t after = search.get_candidate_frontier();
        REQUIRE(after >= before);
        REQUIRE(after <= search.get_searched_nsamps());
        run.frontier.push_back(before);
        before = after;
    };
    for (SizeType k = 0; k < input.size(); ++k) {
        if constexpr (requires {
                          search.execute(std::span<const float>{},
                                         std::span<const uint8_t>{});
                      }) {
            check(search.execute(std::span<const float>(input[k]),
                                 std::span<const uint8_t>(masks[k])));
        } else {
            check(search.execute(std::span<const float>(input[k])));
        }
    }
    check(drain ? search.drain() : search.flush());
    REQUIRE(search.get_candidate_frontier() == search.get_searched_nsamps());
    run.frontier.push_back(search.get_candidate_frontier());
    return run;
}

std::vector<Exec> frontier_backends() {
    std::vector<Exec> out{Exec::cpu(3)};
#ifdef DMT_ENABLE_GPU
    out.push_back({algorithms::detail::kGPUBackend, 1, 0});
#endif
    return out;
}

} // namespace

TEST_CASE("FDMTBoxcar candidate frontier bounds every later candidate",
          "[boxcar][frontier][cpu]") {
    const auto plan  = frontier_plan();
    const auto input = waterfalls(plan, 1, 6);
    for (const auto mode : {PeakFindingMode::kContiguousPeakPerWidth,
                            PeakFindingMode::kContiguousPeakAcrossWidths,
                            PeakFindingMode::kAllAboveThreshold}) {
        for (const bool masked : {false, true}) {
            for (const bool drain : {false, true}) {
                DYNAMIC_SECTION("mode " << static_cast<int>(mode) << " masked "
                                        << masked << " drain " << drain) {
                    const auto cfg   = frontier_config(mode);
                    const auto masks = frontier_masks(input.size(), masked);
                    FDMTBoxcar search(plan, cfg, true, Exec::cpu(3), 1,
                                      algorithms::kFDMTAutoFuse, 64);
                    const auto run = stream(search, input, masks, drain);
                    REQUIRE(run.candidates.size() > 100);
                    if (mode == PeakFindingMode::kAllAboveThreshold) {
                        // Nothing stays open: the searched position.
                        for (SizeType k = 1; k + 1 < run.frontier.size(); ++k) {
                            CHECK(run.frontier[k] == k * plan.get_nsamps());
                        }
                    }
                    search.reset_history();
                    CHECK(search.get_candidate_frontier() == 0);
                }
            }
        }
    }
}

TEST_CASE("FDMTBoxcar candidate frontier holds a long excursion's begin",
          "[boxcar][frontier][cpu]") {
    // A constant offset keeps the rows above threshold from sample 1000 to
    // 2600 (blocks 2 to 6): the frontier stays at the excursion's begin,
    // and the candidate that finally closes it begins exactly there.
    const auto plan   = frontier_plan();
    const SizeType ns = plan.get_nsamps();
    auto input        = waterfalls(plan, 1, 9);
    for (SizeType k = 0; k < input.size(); ++k) {
        for (SizeType c = 0; c < 32; ++c) {
            for (SizeType t = 0; t < ns; ++t) {
                const SizeType s = (k * ns) + t;
                if (s >= 1000 && s < 2600) {
                    input[k][(c * ns) + t] += 3.0F;
                }
            }
        }
    }
    for (const auto& exec : frontier_backends()) {
        for (const auto mode : {PeakFindingMode::kContiguousPeakPerWidth,
                                PeakFindingMode::kContiguousPeakAcrossWidths}) {
            DYNAMIC_SECTION("backend " << static_cast<int>(exec.backend)
                                       << " mode " << static_cast<int>(mode)) {
                FDMTBoxcar search(plan, frontier_config(mode), true, exec);
                std::vector<uint64_t> held;
                std::vector<RawCandidate> all;
                for (SizeType k = 0; k < input.size(); ++k) {
                    append(all, search.execute(input[k]));
                    if (k >= 3 && k <= 5) {
                        held.push_back(search.get_candidate_frontier());
                    }
                }
                append(all, search.flush());
                REQUIRE(held.size() == 3);
                CHECK(held[0] == held[1]);
                CHECK(held[1] == held[2]);
                CHECK(held[0] <= 1000 + 40);
                CHECK(held[0] < 3 * ns);
                // The closing candidate begins at the held frontier, and no
                // later candidate begins before it.
                const auto oldest = std::ranges::min_element(
                    all, {}, [&](const RawCandidate& c) {
                        return c.end_sample >= 2500 ? c.begin_sample
                                                    : UINT64_MAX;
                    });
                REQUIRE(oldest != all.end());
                CHECK(oldest->begin_sample == held[0]);
                CHECK(oldest->end_sample >= 2500);
            }
        }
    }
}

TEST_CASE("FDMTFFTBoxcar candidate frontier bounds every later candidate",
          "[boxcar][frontier][cpu]") {
    // FDMTFFTBoxcar searches with an output latency: the frontier follows
    // its searched position.
    const plans::FDMTPlan plan(700, 1200, 32, 512, 0.001F,
                               std::vector<IndexType>{0, 3, 9, 20, 41, 77},
                               "valid");
    const auto input = waterfalls(plan, 1, 5);
    for (const auto& exec : frontier_backends()) {
        for (const auto mode : {PeakFindingMode::kContiguousPeakPerWidth,
                                PeakFindingMode::kAllAboveThreshold}) {
            for (const bool drain : {false, true}) {
                DYNAMIC_SECTION("backend " << static_cast<int>(exec.backend)
                                           << " mode " << static_cast<int>(mode)
                                           << " drain " << drain) {
                    algorithms::FDMTFFTBoxcar search(
                        plan, frontier_config(mode), true, exec);
                    const auto run =
                        stream(search, input, Masks(input.size()), drain);
                    REQUIRE_FALSE(run.candidates.empty());
                    search.reset_history();
                    CHECK(search.get_candidate_frontier() == 0);
                }
            }
        }
    }
}

#ifdef DMT_ENABLE_GPU
TEST_CASE("GPU FDMTBoxcar candidate frontier equals the CPU's",
          "[gpu][boxcar][frontier]") {
    const auto plan  = frontier_plan();
    const auto input = waterfalls(plan, 1, 6);
    for (const auto mode : {PeakFindingMode::kContiguousPeakPerWidth,
                            PeakFindingMode::kContiguousPeakAcrossWidths,
                            PeakFindingMode::kAllAboveThreshold}) {
        for (const bool masked : {false, true}) {
            DYNAMIC_SECTION("mode " << static_cast<int>(mode) << " masked "
                                    << masked) {
                const auto cfg   = frontier_config(mode);
                const auto masks = frontier_masks(input.size(), masked);
                FDMTBoxcar cpu(plan, cfg, true, Exec::cpu(3));
                FDMTBoxcar gpu(plan, cfg, true,
                               {algorithms::detail::kGPUBackend, 1, 0});
                const auto a = stream(cpu, input, masks, true);
                const auto b = stream(gpu, input, masks, true);
                CHECK(a.frontier == b.frontier);
                compare(b.candidates, a.candidates);
            }
        }
    }
}
#endif
