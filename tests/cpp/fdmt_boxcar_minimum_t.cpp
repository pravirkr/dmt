// FDMTBoxcar::estimate_minimum_memory() and the budget's monotonicity: every
// max_memory_bytes from the minimum up builds (no budget a larger path
// takes and then misses while a smaller path fits).

#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "dmt/algorithms/fdmt_boxcar.hpp"
#include "dmt/common/dm_grids.hpp"

#include "fdmt_boxcar_reference.hpp"
#include "test_helpers.hpp"
#include "test_random.hpp"
#include "test_sized.hpp"

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
                    Exec exec,
                    bool full) {
    const SizeType lo = check_minimum(plan, cfg, exec).max_memory_bytes;
    const SizeType hi =
        estimate(plan, cfg, exec, SizeType{1} << 40).usage.budgeted() * 5 / 4;
    REQUIRE(hi > lo);
    const SizeType kSteps = full ? 128 : 24;
    std::set<SizeType> usages;
    for (SizeType k = 0; k <= kSteps; ++k) {
        const SizeType b = lo + ((hi - lo) * k / kSteps);
        INFO("budget " << b);
        FDMTBoxcarMemoryEstimate e;
        REQUIRE_NOTHROW(e = estimate(plan, cfg, exec, b));
        // max_memory_bytes bounds budgeted(), not total(): on a GPU, total()
        // also counts host memory (plan copies, result storage) that the
        // budget leaves out, so total() may exceed the budget there.
        CHECK(e.usage.budgeted() <= b);
        CHECK(e.usage.total() == e.usage.budgeted() + e.usage.host);
        if (exec.backend == Backend::kCPU) {
            CHECK(e.usage.host == 0);
        }
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

void check_backend(Exec exec, bool full) {
    SECTION("tree-only plan (no smaller path)") {
        check_monotone(grid_plan(16, 4096, 300, 0), small_config(), exec, full);
    }
    SECTION("full band, 512 channels") {
        check_monotone(grid_plan(512, 4096, 400, 0), small_config(), exec,
                       full);
    }
    SECTION("subband search, depth 1") {
        check_monotone(grid_plan(256, 4096, 600, 1), small_config(), exec,
                       full);
    }
    SECTION("subband search, depth 2") {
        const auto plan = full ? grid_plan(1024, 8192, 1500, 2)
                               : grid_plan(512, 4096, 800, 2);
        check_monotone(plan, small_config(), exec, full);
    }
    SECTION("a plan whose budget search overshoots and narrows (GPU)") {
        check_minimum(grid_plan(2048, 4096, 600, 1), small_config(), exec);
    }
    SECTION("declared per-call masks") {
        auto cfg           = small_config();
        cfg.per_call_masks = true;
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

void check_minimum_equals_default(Exec exec) {
    const auto grid = plans::DDMTPlan::generate_levin_dm_grid_piecewise(
        0.0F, 1000.0F, 455.6e-6F, 455.6e-6F, 1100.0F, 1714.0F, 1152, 1.1F);
    const plans::FDMTPlan plan(1100.0F, 1714.0F, 1152, 4096, 455.6e-6F, grid,
                               "valid", 1);
    constexpr SizeType kBlocks = 3;
    const SizeType block_elems = plan.get_nchans() * plan.get_nsamps();
    std::vector<std::vector<uint8_t>> packed_blocks;
    std::vector<std::vector<float>> float_blocks;
    for (SizeType b = 0; b < kBlocks; ++b) {
        auto bytes = test_random::uniform_int<uint8_t>(
            block_elems, 310 + static_cast<unsigned>(b), 0, 255);
        std::vector<float> vals(block_elems);
        for (SizeType i = 0; i < block_elems; ++i) {
            vals[i] = static_cast<float>(bytes[i]) - 128.0F;
        }
        packed_blocks.push_back(std::move(bytes));
        float_blocks.push_back(std::move(vals));
    }
    for (const bool packed : {false, true}) {
        DYNAMIC_SECTION((packed ? "packed 8-bit" : "float")) {
            const BoxcarConfig cfg(
                24, 1.5F, 4.2F, 73.9F, PeakFindingMode::kContiguousPeakPerWidth,
                1U << 20, 0, true, packed ? 8 : 32, packed ? 127.5F : 0.0F);
            const auto min =
                FDMTBoxcar::estimate_minimum_memory(plan, cfg, true, exec);
            FDMTBoxcar at_min(plan, cfg, true, exec, 1,
                              algorithms::kFDMTAutoFuse, 0,
                              min.max_memory_bytes);
            FDMTBoxcar at_def(plan, cfg, true, exec, 1,
                              algorithms::kFDMTAutoFuse, 0,
                              algorithms::kFDMTDefaultMaxMemory);
            REQUIRE(at_min.get_search_latency() > 0);
            const auto run_all = [&](FDMTBoxcar& s) {
                std::vector<RawCandidate> out;
                for (SizeType b = 0; b < kBlocks; ++b) {
                    if (packed) {
                        boxcar_test::append(out,
                                            s.execute(std::span<const uint8_t>(
                                                packed_blocks[b])));
                    } else {
                        boxcar_test::append(
                            out,
                            s.execute(std::span<const float>(float_blocks[b])));
                    }
                }
                boxcar_test::append(out, s.drain());
                return out;
            };
            const auto got  = run_all(at_min);
            const auto want = run_all(at_def);
            REQUIRE_FALSE(got.empty());
            if (exec.backend == Backend::kCPU || packed) {
                boxcar_test::compare_exact(got, want);
            } else {
                boxcar_test::compare(got, want);
            }
        }
    }
}

} // namespace

DMT_SIZED_TEST_CASE("FDMTBoxcar minimum memory and monotone budgets (CPU)",
                    "[boxcar][memory][cpu]") {
    check_backend(Exec::cpu(), full);
    SECTION("an instance built at the minimum stays within it") {
        const auto plan = grid_plan(256, 4096, 600, 1);
        const auto m    = check_minimum(plan, small_config(), Exec::cpu());
        const FDMTBoxcar search(plan, small_config(), true, Exec::cpu(), 1,
                                algorithms::kFDMTAutoFuse, 0,
                                m.max_memory_bytes);
        CHECK(search.get_memory_usage().budgeted() == m.usage.budgeted());
        CHECK(search.get_chunk_nsamps() == m.chunk_nsamps);
    }
}

TEST_CASE("FDMTBoxcar candidates at estimate_minimum_memory match default "
          "budget (CPU)",
          "[boxcar][memory][cpu]") {
    check_minimum_equals_default(Exec::cpu());
}

TEST_CASE("FDMTBoxcar minimum memory on short blocks (CPU)",
          "[boxcar][memory][cpu]") {
    // Blocks of 256 samples or fewer: no skewed chunk is shorter than the
    // block, so the smaller path is unavailable and the tree sets the
    // minimum (plans the skewed path could otherwise take).
    for (const SizeType nsamps : {128, 256}) {
        for (const auto& [nchans, depth] :
             {std::pair<SizeType, SizeType>{32, 0},
              {64, 1},
              {128, 2},
              {128, 3}}) {
            DYNAMIC_SECTION(nchans << " channels, depth " << depth << ", "
                                   << nsamps << " samples") {
                check_minimum(grid_plan(nchans, nsamps, 10, depth),
                              small_config(), Exec::cpu());
            }
        }
    }
}

#ifdef DMT_ENABLE_GPU
TEST_CASE("FDMTBoxcar minimum memory and monotone budgets (GPU)",
          "[boxcar][memory][gpu]") {
    check_backend(test::gpu_exec(), true);
}

TEST_CASE("FDMTBoxcar candidates at estimate_minimum_memory match default "
          "budget (GPU)",
          "[boxcar][memory][gpu]") {
    check_minimum_equals_default(test::gpu_exec());
}
#endif
