#pragma once

// Subband search (plan subband_depth > 0) on the skewed-window paths against
// the tree path, shared by the CPU and CUDA skew tests. The skewed paths
// emit each subband row with a fixed lag (compute_skew_windows), so a row's
// last `lag` samples are searched one or more blocks later than on the tree
// path; candidates are compared up to a cutoff both paths have searched.

#include <algorithm>
#include <cstdint>
#include <random>
#include <span>
#include <tuple>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "dmt/algorithms/fdmt.hpp"
#include "dmt/common/candidate.hpp"
#include "dmt/common/plans.hpp"

#include "dmt/boxcar.hpp"
#include "dmt/engines.hpp"

namespace dmt::skew_search_test {

struct SearchRun {
    std::vector<RawCandidate> candidates;
    std::string path;
};

/// Integer-valued input (exact on every path): float samples v - 128 for
/// bytes v, or the bytes themselves packed.
struct SearchBlock {
    std::vector<float> values;
    std::vector<uint8_t> packed;
};

inline SearchBlock search_block(SizeType n, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<uint32_t> dis(0, 255);
    SearchBlock b{.values = std::vector<float>(n),
                  .packed = std::vector<uint8_t>(n)};
    for (SizeType i = 0; i < n; ++i) {
        const auto v = dis(gen);
        b.packed[i]  = static_cast<uint8_t>(v);
        b.values[i]  = static_cast<float>(v) - 128.0F;
    }
    return b;
}

inline BoxcarConfig search_config(bool packed) {
    // Uniform bytes: sigma 73.9 per sample. A low threshold gives
    // candidates from noise in every plane.
    return BoxcarConfig(24, 1.5F, 4.2F, 73.9F,
                        PeakFindingMode::kContiguousPeakPerWidth, 1U << 20, 0,
                        true, packed ? 8 : 32, packed ? 127.5F : 0.0F);
}

/// Streams `nblocks` blocks through FDMTBoxcar's engine on `exec`, on the
/// skewed path (`skew`) or the tree path, then flushes.
inline SearchRun run_subband_search(const plans::FDMTPlan& plan,
                                    Exec exec,
                                    bool skew,
                                    bool packed,
                                    SizeType nblocks,
                                    SizeType chunk = 1024) {
    using namespace algorithms::detail; // NOLINT
    const auto box      = search_config(packed);
    const auto planes   = make_boxcar_plans(plan, box, true);
    const SizeType root = plan.get_niters();
    FDMTEngineConfig cfg{};
    cfg.use_box_smearing = true;
    cfg.int_tree         = false;
    cfg.exec             = exec;
    cfg.chunk_nsamps     = chunk;
    cfg.max_fuse_levels  = std::min(root - 1, root - plan.get_subband_depth());
    cfg.search_upper_levels = plan.get_subband_depth() > 0;
    cfg.search_nbits        = packed ? 8 : 32;
    cfg.skew_history_bytes  = skew ? 0 : SIZE_MAX;
    cfg.force_skewed_direct = skew;
    const bool cpu          = exec.backend == Backend::kCPU;
    auto engine =
        cpu ? make_fdmt_boxcar_cpu(plan, box, planes, cfg,
                                   make_boxcar_driver(make_fdmt_cpu(plan, cfg)))
#ifdef DMT_ENABLE_GPU
            : make_fdmt_boxcar_gpu(
                  plan, box, planes, cfg,
                  make_boxcar_driver(make_fdmt_gpu(plan, cfg)));
#else
            : nullptr;
#endif
    SearchRun run;
    run.path         = engine->search_path();
    const SizeType n = plan.get_nchans() * plan.get_nsamps();
    const auto add   = [&](const CandidateResultView& r) {
        REQUIRE_FALSE(r.overflowed());
        run.candidates.insert(run.candidates.end(), r.candidates.begin(),
                              r.candidates.end());
    };
    for (SizeType b = 0; b < nblocks; ++b) {
        const auto blk = search_block(n, 900 + static_cast<unsigned>(b));
        add(packed ? engine->execute(std::span<const uint8_t>(blk.packed))
                   : engine->execute(std::span<const float>(blk.values)));
    }
    add(engine->flush());
    return run;
}

inline auto candidate_key(const RawCandidate& c) {
    return std::tuple(c.subband_level, c.subband_idx, c.dm_idx, c.filter_idx,
                      c.begin_sample, c.end_sample, c.peak_sample);
}

/// Candidates of both runs that end before `cutoff` are identical.
inline void require_same_before(std::vector<RawCandidate> actual,
                                std::vector<RawCandidate> expected,
                                uint64_t cutoff,
                                float snr_rtol = 0.0F) {
    const auto keep = [&](std::vector<RawCandidate>& v) {
        std::erase_if(v, [&](const RawCandidate& c) {
            return c.end_sample + 1 >= cutoff;
        });
        std::ranges::sort(v, {}, candidate_key);
    };
    keep(actual);
    keep(expected);
    REQUIRE(expected.size() > 100);
    REQUIRE(actual.size() == expected.size());
    SizeType sub  = 0;
    SizeType stag = 0;
    for (SizeType i = 0; i < actual.size(); ++i) {
        const auto& a = actual[i];
        const auto& e = expected[i];
        INFO("candidate " << i << " level " << e.subband_level << " band "
                          << e.subband_idx << " dm " << e.dm_idx << " filter "
                          << e.filter_idx << " [" << e.begin_sample << ", "
                          << e.end_sample << "] snr " << e.snr);
        REQUIRE(candidate_key(a) == candidate_key(e));
        REQUIRE(a.staggered == e.staggered);
        if (snr_rtol == 0.0F) {
            REQUIRE(a.snr == e.snr);
        } else {
            REQUIRE_THAT(a.snr, Catch::Matchers::WithinRel(e.snr, snr_rtol));
        }
        sub += e.subband_level > 0 ? 1 : 0;
        stag += e.staggered;
    }
    // Every kind of row took part.
    REQUIRE(sub > 0);
    REQUIRE(stag > 0);
    REQUIRE(sub < actual.size());
}

} // namespace dmt::skew_search_test
