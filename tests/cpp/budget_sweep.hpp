#pragma once

// Rules 3/4 sweep (memory ground rules): every budget from
// estimate_minimum_memory() up builds, and the chosen configuration's rank
// in the backend's ordered list never gets worse as the budget grows (the
// list is a preference order, close to the fastest, not an exact one).

#include <cmath>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include "dmt/algorithms/fdmt_boxcar.hpp"

namespace dmt::test {

/// Rank of an estimate's configuration in the backend's list (0 =
/// fastest); nullopt when it is not in the list.
using SweepRank = std::function<std::optional<SizeType>(
    const algorithms::FDMTBoxcarMemoryEstimate&)>;

inline void check_budget_sweep(const plans::FDMTPlan& plan,
                               const BoxcarConfig& cfg,
                               Exec exec,
                               const SweepRank& rank,
                               int steps = 24) {
    using algorithms::FDMTBoxcar;
    const auto min = FDMTBoxcar::estimate_minimum_memory(plan, cfg, true, exec);
    const auto top = FDMTBoxcar::estimate_memory(
        plan, cfg, true, exec, 1, algorithms::kFDMTAutoFuse, 0, SIZE_MAX / 2);
    // The smallest budget that builds (CPU path choices keep headroom over
    // the usage).
    const auto lo = static_cast<double>(min.max_memory_bytes);
    const auto hi = static_cast<double>(top.usage.budgeted()) * 1.05;
    REQUIRE(lo <= hi);
    std::optional<SizeType> prev;
    for (int i = 0; i <= steps; ++i) {
        const auto budget = static_cast<SizeType>(
            lo * std::pow(hi / lo, static_cast<double>(i) / steps));
        INFO("budget " << budget << " (minimum " << min.max_memory_bytes
                       << ")");
        algorithms::FDMTBoxcarMemoryEstimate e;
        REQUIRE_NOTHROW(e = FDMTBoxcar::estimate_memory(
                            plan, cfg, true, exec, 1, algorithms::kFDMTAutoFuse,
                            0, budget));
        INFO("path '" << e.search_path << "' chunk " << e.chunk_nsamps
                      << " latency " << e.search_latency);
        CHECK(e.usage.budgeted() <= budget);
        const auto r = rank(e);
        REQUIRE(r.has_value());
        if (prev) {
            CHECK(*r <= *prev); // the rank never gets worse
        }
        prev = r;
    }
    // The unlimited budget's configuration is the head of what this plan
    // reaches, and the sweep's top budget gets it.
    CHECK(prev == rank(top));
}

} // namespace dmt::test
