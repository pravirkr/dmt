// FDMTBoxcar::estimate_minimum_memory() and the budget's monotonicity: every
// max_memory_bytes from the minimum up builds (no budget a larger path
// takes and then misses while a smaller path fits).

#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "dmt/algorithms/fdmt_boxcar.hpp"
#include "dmt/common/dm_grids.hpp"

using namespace dmt;
using algorithms::FDMTBoxcar;
using algorithms::FDMTBoxcarMemoryEstimate;

namespace {

plans::FDMTPlan
grid_plan(SizeType nchans, SizeType nsamps, double dm_max, SizeType depth) {
    plans::DMGridConfig g;
    g.f_min        = 1200.0;
    g.f_max        = 1600.0;
    g.nchans       = nchans;
    g.tsamp        = 1e-4;
    g.pulse_width  = 1e-4;
    g.tol          = 1.25;
    const auto dms = plans::generate_levin_dm_grid(g, 0.0, dm_max);
    return {1200.0F, 1600.0F, nchans,
            nsamps,  1e-4F,   std::vector<float>(dms.begin(), dms.end()),
            "valid", depth};
}

FDMTBoxcarMemoryEstimate estimate(const plans::FDMTPlan& plan,
                                  const BoxcarConfig& cfg,
                                  Exec exec,
                                  SizeType budget,
                                  SizeType chunk = 0) {
    return FDMTBoxcar::estimate_memory(
        plan, cfg, true, exec, 1, algorithms::kFDMTAutoFuse, chunk, budget);
}

/// The minimum builds, with the same configuration; one byte less throws.
FDMTBoxcarMemoryEstimate check_minimum(const plans::FDMTPlan& plan,
                                       const BoxcarConfig& cfg,
                                       Exec exec,
                                       SizeType chunk = 0) {
    const auto m = FDMTBoxcar::estimate_minimum_memory(
        plan, cfg, true, exec, 1, algorithms::kFDMTAutoFuse, chunk);
    REQUIRE(m.max_memory_bytes >= m.usage.budgeted());
    const auto e = estimate(plan, cfg, exec, m.max_memory_bytes, chunk);
    CHECK(e.usage.budgeted() == m.usage.budgeted());
    CHECK(e.usage.total() == m.usage.total());
    CHECK(e.chunk_nsamps == m.chunk_nsamps);
    CHECK(e.search_path == m.search_path);
    CHECK(e.max_memory_bytes == m.max_memory_bytes);
    if (chunk != 0) {
        CHECK(m.chunk_nsamps == chunk);
    }
    CHECK_THROWS_AS(estimate(plan, cfg, exec, m.max_memory_bytes - 1, chunk),
                    std::invalid_argument);
    return m;
}

/// Every budget from the minimum to past the unconstrained engine builds,
/// including one byte under each configuration's own usage (where a larger
/// path starts to be chosen).
void check_monotone(const plans::FDMTPlan& plan,
                    const BoxcarConfig& cfg,
                    Exec exec) {
    const SizeType lo = check_minimum(plan, cfg, exec).max_memory_bytes;
    const SizeType hi =
        estimate(plan, cfg, exec, SizeType{1} << 40).usage.budgeted() * 5 / 4;
    REQUIRE(hi > lo);
    constexpr SizeType kSteps = 128;
    std::set<SizeType> usages;
    for (SizeType k = 0; k <= kSteps; ++k) {
        const SizeType b = lo + ((hi - lo) * k / kSteps);
        INFO("budget " << b);
        FDMTBoxcarMemoryEstimate e;
        REQUIRE_NOTHROW(e = estimate(plan, cfg, exec, b));
        CHECK(e.usage.budgeted() <= b);
        usages.insert(e.usage.budgeted());
    }
    REQUIRE(usages.size() > 1);
    for (const SizeType u : usages) {
        if (u - 1 < lo) {
            continue;
        }
        INFO("budget " << u - 1 << " (one under a configuration's usage)");
        FDMTBoxcarMemoryEstimate e;
        REQUIRE_NOTHROW(e = estimate(plan, cfg, exec, u - 1));
        CHECK(e.usage.budgeted() <= u - 1);
    }
}

BoxcarConfig small_config() {
    BoxcarConfig cfg;
    cfg.max_boxcar = 64;
    return cfg;
}

void check_backend(Exec exec) {
    SECTION("tree-only plan (no smaller path)") {
        check_monotone(grid_plan(16, 4096, 300, 0), small_config(), exec);
    }
    SECTION("full band, 512 channels") {
        check_monotone(grid_plan(512, 4096, 400, 0), small_config(), exec);
    }
    SECTION("subband search, depth 1") {
        check_monotone(grid_plan(256, 4096, 600, 1), small_config(), exec);
    }
    SECTION("subband search, depth 2") {
        check_monotone(grid_plan(1024, 8192, 1500, 2), small_config(), exec);
    }
    SECTION("a plan whose budget search overshoots and narrows (GPU)") {
        check_minimum(grid_plan(2048, 4096, 600, 1), small_config(), exec);
    }
    SECTION("declared per-call masks") {
        auto cfg           = small_config();
        cfg.per_call_masks = true;
        // Blocks of at least the row support + max_boxcar - 1 samples.
        check_minimum(grid_plan(256, 8192, 600, 1), cfg, exec);
    }
    SECTION("explicit chunk") {
        check_minimum(grid_plan(256, 4096, 600, 1), small_config(), exec, 512);
    }
    SECTION("the budget error gives the engine's total") {
        const auto plan = grid_plan(256, 4096, 600, 1);
        const auto m    = check_minimum(plan, small_config(), exec);
        CHECK_THROWS_MATCHES(
            estimate(plan, small_config(), exec, m.max_memory_bytes - 1),
            std::invalid_argument,
            Catch::Matchers::MessageMatches(Catch::Matchers::StartsWith(
                "FDMTBoxcar: this configuration needs")));
    }
}

} // namespace

TEST_CASE("FDMTBoxcar minimum memory and monotone budgets (CPU)",
          "[boxcar][memory][cpu]") {
    check_backend(Exec::cpu(2));
    SECTION("an instance built at the minimum stays within it") {
        const auto plan = grid_plan(256, 4096, 600, 1);
        const auto m    = check_minimum(plan, small_config(), Exec::cpu(2));
        const FDMTBoxcar search(plan, small_config(), true, Exec::cpu(2), 1,
                                algorithms::kFDMTAutoFuse, 0,
                                m.max_memory_bytes);
        CHECK(search.get_memory_usage().budgeted() == m.usage.budgeted());
        CHECK(search.get_chunk_nsamps() == m.chunk_nsamps);
    }
}

#ifdef DMT_ENABLE_GPU
TEST_CASE("FDMTBoxcar minimum memory and monotone budgets (GPU)",
          "[boxcar][memory][gpu]") {
    check_backend(Exec::cuda(0));
}
#endif
