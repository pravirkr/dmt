#pragma once

// OverflowPolicy::kKeepBrightest reference: the result of one call from the
// full candidate set of an unlimited run (kDropNew without caps), shared by
// the CPU and GPU overflow tests.

#include <algorithm>
#include <cstdint>
#include <map>
#include <span>
#include <tuple>
#include <vector>

#include "dmt/boxcar.hpp"

#include "fdmt_boxcar_reference.hpp"

namespace boxcar_test {

using algorithms::detail::brighter;

using Calls = std::vector<std::vector<RawCandidate>>;

// One row of the search: (beam, subband, DM).
inline auto row_key(const RawCandidate& c) {
    return std::make_tuple(c.beam_idx, c.subband_level, c.subband_idx,
                           c.staggered, c.dm_idx);
}

// The kKeepBrightest result of one call from its full candidate set.
inline std::vector<RawCandidate>
keep_brightest(const std::vector<RawCandidate>& all,
               SizeType per_row,
               SizeType global,
               CandidateCounts& counts) {
    std::map<decltype(row_key(all.front())), std::vector<RawCandidate>> rows;
    for (const auto& c : all) {
        rows[row_key(c)].push_back(c);
    }
    std::vector<RawCandidate> kept;
    counts                = {};
    counts.total_detected = all.size();
    for (auto& [key, v] : rows) {
        std::ranges::sort(v, brighter);
        const SizeType n = std::min<SizeType>(v.size(), per_row);
        counts.dropped_per_dm += v.size() - n;
        kept.insert(kept.end(), v.begin(), v.begin() + static_cast<long>(n));
    }
    std::ranges::sort(kept, brighter);
    if (kept.size() > global) {
        counts.dropped_global = kept.size() - global;
        kept.resize(global);
    }
    counts.emitted_count = kept.size();
    return kept;
}

// Every call's candidates (execute per block, then flush).
template <typename Search, typename Block>
Calls run_calls(Search& search,
                const std::vector<Block>& blocks,
                std::vector<CandidateCounts>* counts = nullptr) {
    Calls calls;
    const auto take = [&](const CandidateResultView& r) {
        calls.emplace_back(r.candidates.begin(), r.candidates.end());
        if (counts != nullptr) {
            counts->push_back(static_cast<const CandidateCounts&>(r));
        }
    };
    for (const auto& block : blocks) {
        take(search.execute(std::span(block)));
    }
    take(search.flush());
    return calls;
}

// As run_calls() with a per-call kill mask for each block.
inline Calls run_calls_masked(algorithms::FDMTBoxcar& search,
                              const std::vector<std::vector<float>>& blocks,
                              const std::vector<std::vector<uint8_t>>& masks,
                              std::vector<CandidateCounts>* counts = nullptr) {
    Calls calls;
    const auto take = [&](const CandidateResultView& r) {
        calls.emplace_back(r.candidates.begin(), r.candidates.end());
        if (counts != nullptr) {
            counts->push_back(static_cast<const CandidateCounts&>(r));
        }
    };
    for (SizeType k = 0; k < blocks.size(); ++k) {
        take(search.execute(std::span(blocks[k]),
                            std::span<const uint8_t>(masks[k])));
    }
    take(search.flush());
    return calls;
}

inline void require_same(std::vector<RawCandidate> a,
                         std::vector<RawCandidate> b) {
    std::ranges::sort(a, brighter);
    std::ranges::sort(b, brighter);
    REQUIRE(a.size() == b.size());
    for (SizeType i = 0; i < a.size(); ++i) {
        CHECK(row_key(a[i]) == row_key(b[i]));
        CHECK(a[i].peak_sample == b[i].peak_sample);
        CHECK(a[i].begin_sample == b[i].begin_sample);
        CHECK(a[i].end_sample == b[i].end_sample);
        CHECK(a[i].filter_idx == b[i].filter_idx);
        CHECK(a[i].snr == b[i].snr);
    }
}

} // namespace boxcar_test
