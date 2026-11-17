// FDMTBoxcar per-plane candidate counts (CandidateResultView::plane_counts)
// and BoxcarConfig::max_candidates_per_plane, on the CPU and the GPU,
// against a run whose buffers never overflow.

#include <algorithm>
#include <bit>
#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "dmt/boxcar.hpp"

#include "fdmt_boxcar_reference.hpp"

using namespace boxcar_test;

namespace {

constexpr SizeType kBeams  = 2;
constexpr SizeType kBlocks = 3;

plans::FDMTPlan planes_plan(SizeType depth) {
    return {700,     1200,   16,
            512,     0.001F, std::vector<IndexType>{0, 3, 9, 20, 41},
            "valid", depth};
}

BoxcarConfig planes_config(OverflowPolicy policy) {
    BoxcarConfig cfg;
    cfg.max_boxcar       = 9;
    cfg.wtsp             = 1.5F;
    cfg.detect_thresh    = 2.0F;
    cfg.max_candidates   = 1U << 20;
    cfg.max_cands_per_dm = policy == OverflowPolicy::kKeepBrightest ? 1024 : 0;
    cfg.overflow_policy  = policy;
    return cfg;
}

/// Each call's candidates (execute() per block, then flush()), counts and
/// per-plane counts.
struct Run {
    std::vector<std::vector<RawCandidate>> calls;
    std::vector<CandidateCounts> counts;
    std::vector<std::vector<CandidateCounts>> planes;
    std::vector<plans::BoxcarPlanePlan> plans;
};

Run run(const plans::FDMTPlan& plan, const BoxcarConfig& cfg, Exec exec) {
    const auto input = waterfalls(plan, kBeams, kBlocks);
    FDMTBoxcar search(plan, cfg, true, exec, kBeams);
    Run r;
    r.plans.assign(search.get_search_plans().begin(),
                   search.get_search_plans().end());
    const auto record = [&](const CandidateResultView& v) {
        r.calls.emplace_back(v.candidates.begin(), v.candidates.end());
        r.counts.push_back(v);
        r.planes.emplace_back(v.plane_counts.begin(), v.plane_counts.end());
    };
    for (const auto& block : input) {
        record(search.execute(block));
    }
    record(search.flush());
    return r;
}

SizeType plane_of(const Run& r, const RawCandidate& c) {
    for (SizeType p = 0; p < r.plans.size(); ++p) {
        if (r.plans[p].subband_level == c.subband_level &&
            r.plans[p].subband_idx == c.subband_idx) {
            return p;
        }
    }
    FAIL("candidate of no plane");
    return 0;
}

/// A call's candidates by plane.
std::vector<std::vector<RawCandidate>>
by_plane(const Run& r, const std::vector<RawCandidate>& call) {
    std::vector<std::vector<RawCandidate>> out(r.plans.size());
    for (const auto& c : call) {
        out[plane_of(r, c)].push_back(c);
    }
    return out;
}

bool same_candidate(const RawCandidate& a, const RawCandidate& b) {
    return key(a) == key(b) &&
           std::bit_cast<uint32_t>(a.snr) == std::bit_cast<uint32_t>(b.snr);
}

/// The per-plane counters add up to the call's, and each plane's emitted
/// count is its number of candidates in the call.
void check_planes_add_up(const Run& r) {
    REQUIRE(r.planes.size() == r.calls.size());
    for (SizeType k = 0; k < r.calls.size(); ++k) {
        INFO("call " << k);
        REQUIRE(r.planes[k].size() == r.plans.size());
        CandidateCounts sum{};
        const auto mine = by_plane(r, r.calls[k]);
        for (SizeType p = 0; p < r.plans.size(); ++p) {
            const auto& c = r.planes[k][p];
            sum.emitted_count += c.emitted_count;
            sum.total_detected += c.total_detected;
            sum.dropped_global += c.dropped_global;
            sum.dropped_per_dm += c.dropped_per_dm;
            CHECK(c.emitted_count == mine[p].size());
            CHECK(c.total_detected ==
                  c.emitted_count + c.dropped_global + c.dropped_per_dm);
        }
        CHECK(sum.emitted_count == r.counts[k].emitted_count);
        CHECK(sum.total_detected == r.counts[k].total_detected);
        CHECK(sum.dropped_global == r.counts[k].dropped_global);
        CHECK(sum.dropped_per_dm == r.counts[k].dropped_per_dm);
    }
}

/// Detections per plane do not depend on what overflow dropped.
void check_same_detections(const Run& a, const Run& b) {
    REQUIRE(a.planes.size() == b.planes.size());
    for (SizeType k = 0; k < a.planes.size(); ++k) {
        for (SizeType p = 0; p < a.plans.size(); ++p) {
            INFO("call " << k << " plane " << p);
            CHECK(a.planes[k][p].total_detected ==
                  b.planes[k][p].total_detected);
        }
    }
}

void check_counts(Exec exec) {
    for (const SizeType depth : {SizeType{0}, SizeType{1}, SizeType{2}}) {
        const auto plan = planes_plan(depth);
        for (const auto policy :
             {OverflowPolicy::kDropNew, OverflowPolicy::kKeepBrightest}) {
            DYNAMIC_SECTION("depth "
                            << depth << " keep "
                            << (policy == OverflowPolicy::kKeepBrightest)) {
                const auto cfg = planes_config(policy);
                const auto ref = run(plan, cfg, exec);
                check_planes_add_up(ref);
                CHECK(ref.counts.front().total_detected > 100);
                for (const auto& call : ref.counts) {
                    CHECK_FALSE(call.overflowed());
                }
                // Per-DM drops (kDropNew: past the cap; kKeepBrightest:
                // past the store) and call-wide drops land in their planes.
                auto tight             = cfg;
                tight.max_cands_per_dm = 2;
                tight.max_candidates   = 10;
                const auto small       = run(plan, tight, exec);
                check_planes_add_up(small);
                check_same_detections(small, ref);
                CHECK(small.counts.front().dropped_per_dm > 0);
                CHECK(small.counts.front().dropped_global > 0);
            }
        }
    }
}

void check_share(Exec exec) {
    for (const SizeType depth : {SizeType{1}, SizeType{2}}) {
        const auto plan = planes_plan(depth);
        for (const auto policy :
             {OverflowPolicy::kDropNew, OverflowPolicy::kKeepBrightest}) {
            for (const SizeType share : {SizeType{3}, SizeType{1U << 14}}) {
                DYNAMIC_SECTION("depth "
                                << depth << " keep "
                                << (policy == OverflowPolicy::kKeepBrightest)
                                << " share " << share) {
                    const auto cfg                  = planes_config(policy);
                    const auto ref                  = run(plan, cfg, exec);
                    auto capped                     = cfg;
                    capped.max_candidates_per_plane = share;
                    capped.max_candidates           = std::min<SizeType>(
                        cfg.max_candidates, ref.plans.size() * share);
                    const auto got = run(plan, capped, exec);
                    check_planes_add_up(got);
                    check_same_detections(got, ref);
                    bool capped_any = false;
                    for (SizeType k = 0; k < ref.calls.size(); ++k) {
                        auto want       = by_plane(ref, ref.calls[k]);
                        const auto have = by_plane(got, got.calls[k]);
                        for (SizeType p = 0; p < want.size(); ++p) {
                            INFO("call " << k << " plane " << p);
                            const SizeType n =
                                std::min<SizeType>(share, want[p].size());
                            capped_any |= n < want[p].size();
                            REQUIRE(have[p].size() == n);
                            CHECK(got.planes[k][p].dropped_global ==
                                  want[p].size() - n);
                            if (policy == OverflowPolicy::kKeepBrightest) {
                                // Exactly the plane's brightest.
                                std::ranges::sort(want[p],
                                                  algorithms::detail::brighter);
                                want[p].resize(n);
                                auto h = have[p];
                                std::ranges::sort(h,
                                                  algorithms::detail::brighter);
                                for (SizeType i = 0; i < n; ++i) {
                                    REQUIRE(same_candidate(h[i], want[p][i]));
                                }
                            } else {
                                // Some of the plane's own.
                                for (const auto& c : have[p]) {
                                    REQUIRE(std::ranges::any_of(
                                        want[p], [&](const RawCandidate& w) {
                                            return same_candidate(c, w);
                                        }));
                                }
                            }
                        }
                    }
                    CHECK(capped_any == (share == 3));
                }
            }
        }
    }
    // The planes' shares must fit max_candidates.
    auto cfg                     = planes_config(OverflowPolicy::kDropNew);
    const auto plan              = planes_plan(1); // 4 planes
    cfg.max_candidates_per_plane = 10;
    cfg.max_candidates           = 39;
    CHECK_THROWS_AS(FDMTBoxcar(plan, cfg, true, exec), std::invalid_argument);
    CHECK_THROWS_AS(FDMTBoxcar::estimate_memory(plan, cfg, true, exec),
                    std::invalid_argument);
    cfg.max_candidates = 40;
    CHECK_NOTHROW(FDMTBoxcar(plan, cfg, true, exec));
}

} // namespace

TEST_CASE("FDMTBoxcar per-plane counts add up and follow the drops",
          "[boxcar][planes][cpu]") {
    check_counts(Exec::cpu(3));
}

TEST_CASE("FDMTBoxcar max_candidates_per_plane keeps each plane's share",
          "[boxcar][planes][cpu]") {
    check_share(Exec::cpu(3));
}

#ifdef DMT_ENABLE_GPU
TEST_CASE("GPU FDMTBoxcar per-plane counts add up and follow the drops",
          "[boxcar][planes][gpu]") {
    check_counts(Exec::cuda(0));
}

TEST_CASE("GPU FDMTBoxcar max_candidates_per_plane keeps each plane's share",
          "[boxcar][planes][gpu]") {
    check_share(Exec::cuda(0));
}

TEST_CASE("FDMTBoxcar per-plane counts and shares: GPU equals CPU",
          "[boxcar][planes][gpu][parity]") {
    for (const SizeType share : {SizeType{0}, SizeType{3}}) {
        DYNAMIC_SECTION("share " << share) {
            const auto plan = planes_plan(2);
            auto cfg        = planes_config(OverflowPolicy::kKeepBrightest);
            cfg.max_candidates_per_plane = share;
            cfg.max_candidates           = share == 0 ? 50 : 11 * share;
            const auto cpu               = run(plan, cfg, Exec::cpu(3));
            const auto gpu               = run(plan, cfg, Exec::cuda(0));
            for (SizeType k = 0; k < cpu.calls.size(); ++k) {
                for (SizeType p = 0; p < cpu.plans.size(); ++p) {
                    INFO("call " << k << " plane " << p);
                    const auto& a = cpu.planes[k][p];
                    const auto& b = gpu.planes[k][p];
                    CHECK(a.total_detected == b.total_detected);
                    CHECK(a.emitted_count == b.emitted_count);
                    CHECK(a.dropped_global == b.dropped_global);
                    CHECK(a.dropped_per_dm == b.dropped_per_dm);
                }
                compare(gpu.calls[k], cpu.calls[k]);
            }
        }
    }
}
#endif
