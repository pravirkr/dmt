// Memory ground rules 3 and 4 on the CPU: every budget from the minimum up
// builds, and the configuration's rank in the CPU's speed order (tree path,
// then FDMTCpuEngine::skew_candidates) never gets worse as it grows.

#include <cstdlib>
#include <optional>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include "budget_sweep.hpp"
#include "cpu/fdmt_cpu_engine.hpp"

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

test::SweepRank cpu_rank(SizeType nbits) {
    return [nbits](
               const FDMTBoxcarMemoryEstimate& e) -> std::optional<SizeType> {
        if (e.search_path.empty()) {
            return 0; // the tree path
        }
        const auto leaf = skew_leaf(e.search_path);
        if (!leaf) {
            return std::nullopt;
        }
        const SizeType ring = nbits == 32 ? 4 : nbits == 16 ? 2 : 1;
        const auto list =
            algorithms::detail::fdmt_cpu::FDMTCpuEngine::skew_candidates(ring);
        // An exact (chunk, K), else the chunk's automatic-leaf entry.
        for (const SizeType want : {*leaf, SizeType{0}}) {
            for (SizeType i = 0; i < list.size(); ++i) {
                if (list[i].first == e.chunk_nsamps && list[i].second == want) {
                    return i + 1;
                }
            }
        }
        return std::nullopt;
    };
}
} // namespace

TEST_CASE("FDMTBoxcar budget sweep: the list's rank never worsens (cpu)",
          "[boxcar][budget][sweep]") {
    // A 1152-channel subband search (cryopaf_lband-like) and its full band.
    for (const SizeType depth : {SizeType{1}, SizeType{0}}) {
        const plans::FDMTPlan plan(1100.0F, 1714.0F, 1152, 16384, 455.6e-6F,
                                   2700, 0, 1, "valid", depth);
        for (const SizeType nbits : {SizeType{32}, SizeType{8}}) {
            INFO("depth " << depth << " nbits " << nbits);
            BoxcarConfig cfg;
            cfg.max_boxcar = 128;
            if (nbits != 32) {
                cfg.input_nbits = nbits;
                cfg.input_mean  = 127;
                cfg.noise_sigma = 8;
            }
            test::check_budget_sweep(plan, cfg, Exec::cpu(2), cpu_rank(nbits));
        }
    }
}
