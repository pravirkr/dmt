// Memory ground rules 3 and 4 on the CPU: every budget from the minimum up
// builds, and the configuration's rank in the CPU's speed order never gets
// worse as it grows. Float: the tree path, then
// FDMTCpuEngine::skew_candidates; packed: its chunks of at least
// kSkewFirstMinChunk, the tree, then the shorter chunks.

#include <cstdlib>
#include <optional>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include "budget_sweep.hpp"
#include "cpu/fdmt_cpu_engine.hpp"
#include "test_sized.hpp"

using namespace dmt;
using algorithms::FDMTBoxcarMemoryEstimate;
using dmt::BoxcarConfig;

namespace {
/// Leaf level K of a CPU skewed path ("skewed direct 0-K, ...").
std::optional<SizeType> skew_leaf(const std::string& path) {
    const std::string tag = "skewed direct 0-";
    const auto at         = path.find(tag);
    if (at == std::string::npos) {
        return std::nullopt;
    }
    return std::strtoull(path.c_str() + at + tag.size(), nullptr, 10);
}

using algorithms::detail::fdmt_cpu::FDMTCpuEngine;

test::SweepRank cpu_rank(SizeType nbits) {
    return [nbits](
               const FDMTBoxcarMemoryEstimate& e) -> std::optional<SizeType> {
        const SizeType ring = nbits == 32 ? 4 : nbits == 16 ? 2 : 1;
        const auto list     = FDMTCpuEngine::skew_candidates(ring);
        // Where the tree sits in the order.
        SizeType tree = 0;
        if (nbits != 32) {
            while (tree < list.size() &&
                   list[tree].first >= FDMTCpuEngine::kSkewFirstMinChunk) {
                ++tree;
            }
        }
        if (e.search_path.empty()) {
            return tree;
        }
        const auto leaf = skew_leaf(e.search_path);
        if (!leaf) {
            return std::nullopt;
        }
        // An exact (chunk, K), else the chunk's automatic-leaf entry.
        for (const SizeType want : {*leaf, SizeType{0}}) {
            for (SizeType i = 0; i < list.size(); ++i) {
                if (list[i].first == e.chunk_nsamps && list[i].second == want) {
                    return i < tree ? i : i + 1;
                }
            }
        }
        return std::nullopt;
    };
}

BoxcarConfig sweep_config(SizeType nbits) {
    BoxcarConfig cfg;
    cfg.max_boxcar = 128;
    if (nbits != 32) {
        cfg.input_nbits = nbits;
        cfg.input_mean  = 127;
        cfg.noise_sigma = 8;
    }
    return cfg;
}
} // namespace

DMT_SIZED_TEST_CASE(
    "FDMTBoxcar budget sweep: the list's rank never worsens (cpu)",
    "[boxcar][budget][sweep]") {
    // A 1152-channel subband search (cryopaf_lband-like) and its full band
    // (full size only).
    for (const SizeType depth : {SizeType{1}, SizeType{0}}) {
        if (!full && depth == 0) {
            continue;
        }
        const plans::FDMTPlan plan(1100.0F, 1714.0F, 1152, 16384, 455.6e-6F,
                                   2700, 0, 1, "valid", depth);
        for (const SizeType nbits : {SizeType{32}, SizeType{8}}) {
            INFO("depth " << depth << " nbits " << nbits);
            test::check_budget_sweep(plan, sweep_config(nbits), Exec::cpu(),
                                     cpu_rank(nbits));
        }
    }
    // Short blocks: no skewed chunk of at least kSkewFirstMinChunk.
    const plans::FDMTPlan small(1100.0F, 1714.0F, 1152, 2048, 455.6e-6F, 2700,
                                0, 1, "valid", 1);
    test::check_budget_sweep(small, sweep_config(8), Exec::cpu(), cpu_rank(8));
}

TEST_CASE("FDMTBoxcar (cpu): packed input takes long skewed chunks first",
          "[boxcar][budget]") {
    using algorithms::FDMTBoxcar;
    const auto path = [](SizeType nsamps, SizeType nbits) {
        const plans::FDMTPlan plan(1100.0F, 1714.0F, 1152, nsamps, 455.6e-6F,
                                   2700, 0, 1, "valid", 1);
        return FDMTBoxcar::estimate_memory(plan, sweep_config(nbits), true,
                                           Exec::cpu());
    };
    // Default budget: everything fits, so the order alone decides.
    const auto packed = path(16384, 8);
    CHECK(packed.chunk_nsamps == 4096);
    CHECK(skew_leaf(packed.search_path).has_value());
    CHECK(packed.search_latency > 0);
    const auto flt = path(16384, 32);
    CHECK(flt.search_path.empty()); // the tree
    CHECK(flt.search_latency == 0);
    const auto short_block = path(2048, 8);
    CHECK(short_block.search_path.empty());
    CHECK(short_block.search_latency == 0);
}
