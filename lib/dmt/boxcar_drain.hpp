#pragma once

// End of stream for the boxcar searches (FDMTBoxcar::drain(),
// FDMTFFTBoxcar::drain()): search the samples the stream has not finished
// searching, then emit the open excursions.

#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "dmt/common/boxcar_plans.hpp"
#include "dmt/common/candidate.hpp"
#include "dmt/common/plans.hpp"

#include "dmt/bit_pack_utils.hpp"
#include "dmt/boxcar.hpp"

namespace dmt::algorithms::detail {

/// Facade state of drain(): the stream end and the candidates it returned.
struct DrainState {
    bool drained{false};
    uint64_t end{0}; ///< Stream samples searched, once drained
    std::vector<RawCandidate> candidates;
    std::vector<CandidateCounts> plane_counts; ///< summed over drain()'s calls

    void reset() noexcept {
        drained = false;
        end     = 0;
        candidates.clear();
        plane_counts.clear();
    }
    /// execute() and drain() after drain() are rejected until reset_history().
    void check_open(const char* what) const {
        if (drained) {
            throw std::logic_error(std::string(what) +
                                   ": the stream was drained; call "
                                   "reset_history() to start a new one");
        }
    }
};

/// Stream samples whose trailing-window endpoints every row has searched.
inline uint64_t searched_nsamps(const FDMTBoxcarEngine& engine,
                                const DrainState& state,
                                uint64_t latency) noexcept {
    if (state.drained) {
        return state.end;
    }
    const uint64_t pos = engine.stream_position();
    return pos > latency ? pos - latency : 0;
}

/**
 * drain() is one call: its internal steps (padding blocks, then flush())
 * each kept their own quotas, so their union can exceed them. Applies the
 * config's quotas once more over the union, as one call would: per DM row
 * (max_cands_per_dm), per plane (max_candidates_per_plane), then the call
 * (max_candidates); kDropNew keeps the first candidates (the steps' order),
 * kKeepBrightest the brightest (brighter()). The kept candidates stay in
 * their order; every one dropped here moves from the emitted to the dropped
 * counters (dropped_per_dm or dropped_global) of the call and of its plane.
 */
inline void reduce_drained(std::vector<RawCandidate>& candidates,
                           CandidateCounts& counts,
                           std::vector<CandidateCounts>& plane_counts,
                           const BoxcarConfig& config,
                           std::span<const plans::BoxcarPlanePlan> planes) {
    const bool keep  = config.overflow_policy == OverflowPolicy::kKeepBrightest;
    const SizeType n = candidates.size();
    std::map<std::tuple<uint32_t, uint32_t, bool>, SizeType> plane_of;
    for (SizeType p = 0; p < planes.size(); ++p) {
        plane_of[{planes[p].subband_level, planes[p].subband_idx,
                  planes[p].staggered}] = p;
    }
    std::vector<SizeType> plane(n, 0);
    for (SizeType i = 0; i < n; ++i) {
        const auto& c = candidates[i];
        const auto it =
            plane_of.find({c.subband_level, c.subband_idx, c.staggered != 0});
        plane[i] = it != plane_of.end() ? it->second : 0;
    }
    std::vector<bool> kept(n, true);
    // Keeps at most `cap` of the kept members of each group (the first ones,
    // or the brightest under kKeepBrightest); returns the indices dropped.
    const auto limit = [&](const auto& key_of, SizeType cap) {
        std::vector<SizeType> dropped;
        if (cap == 0) {
            return dropped;
        }
        std::map<decltype(key_of(SizeType{0})), std::vector<SizeType>> groups;
        for (SizeType i = 0; i < n; ++i) {
            if (kept[i]) {
                groups[key_of(i)].push_back(i);
            }
        }
        for (auto& [key, members] : groups) {
            if (members.size() <= cap) {
                continue;
            }
            if (keep) {
                std::ranges::stable_sort(members, [&](SizeType a, SizeType b) {
                    return brighter(candidates[a], candidates[b]);
                });
            }
            for (SizeType j = cap; j < members.size(); ++j) {
                kept[members[j]] = false;
                dropped.push_back(members[j]);
            }
        }
        return dropped;
    };
    const auto move_counts = [&](const std::vector<SizeType>& dropped,
                                 bool per_dm) {
        const auto move = [&](CandidateCounts& c) {
            --c.emitted_count;
            ++(per_dm ? c.dropped_per_dm : c.dropped_global);
        };
        for (const SizeType i : dropped) {
            move(counts);
            if (plane[i] < plane_counts.size()) {
                move(plane_counts[plane[i]]);
            }
        }
    };
    move_counts(limit(
                    [&](SizeType i) {
                        const auto& c = candidates[i];
                        return std::make_tuple(c.beam_idx, plane[i], c.dm_idx);
                    },
                    config.max_cands_per_dm),
                true);
    move_counts(limit([&](SizeType i) { return plane[i]; },
                      config.max_candidates_per_plane),
                false);
    move_counts(limit([](SizeType /*i*/) { return 0; }, config.max_candidates),
                false);
    SizeType out = 0;
    for (SizeType i = 0; i < n; ++i) {
        if (kept[i]) {
            candidates[out++] = candidates[i];
        }
    }
    candidates.resize(out);
}

/**
 * Ends the stream at its current position `end`: endpoints at or after `end`
 * are never searched (FDMTSearchInput::search_stop), ceil(latency / nsamps)
 * blocks of padding move every row's search up to `end`, and the open
 * excursions are emitted. FDMTBoxcar's engines search the padding without
 * an input block (FDMTBoxcarEngine::zero_blocks()); others push a zero block
 * through the host execute(). The FDMT is causal in its f_min reference, so its
 * rows before `end` do not depend on the padding (zero codes are used for
 * packed input); with fractional FFT delays the padding is zero signal, the
 * band-limited interpolation of a stream that ends.
 */
inline CandidateResultView
drain_stream(FDMTBoxcarEngine& engine,
             const plans::FDMTPlan& plan,
             SizeType nbeams,
             const BoxcarConfig& config,
             std::span<const plans::BoxcarPlanePlan> planes,
             uint64_t latency,
             DrainState& state,
             uint64_t n_tail                                  = 0,
             const std::function<CandidateResultView()>& tail = {}) {
    const SizeType nbits = config.input_nbits;
    state.check_open("drain");
    // A short last block (`tail`, a whole block of which the first n_tail
    // samples are data) moves the end n_tail samples on.
    const uint64_t end = engine.stream_position() + (tail ? n_tail : 0);
    state.candidates.clear();
    state.plane_counts.clear();
    CandidateCounts counts{};
    const auto add = [](CandidateCounts& to, const CandidateCounts& v) {
        to.emitted_count += v.emitted_count;
        to.total_detected += v.total_detected;
        to.dropped_global += v.dropped_global;
        to.dropped_per_dm += v.dropped_per_dm;
    };
    const auto append = [&](const CandidateResultView& v) {
        state.candidates.insert(state.candidates.end(), v.candidates.begin(),
                                v.candidates.end());
        add(counts, v);
        state.plane_counts.resize(v.plane_counts.size());
        for (SizeType p = 0; p < v.plane_counts.size(); ++p) {
            add(state.plane_counts[p], v.plane_counts[p]);
        }
    };
    engine.set_search_stop(end);
    if (tail) {
        append(tail());
    }
    if (end > 0 && latency > 0) {
        // Rows have searched up to the stream position less the latency.
        const SizeType nsamps = plan.get_nsamps();
        const uint64_t need   = end + latency;
        const uint64_t pos    = engine.stream_position();
        const uint64_t blocks =
            need > pos ? (need - pos + nsamps - 1) / nsamps : 0;
        if (engine.zero_blocks()) {
            // No input block is read, staged or allocated.
            for (uint64_t b = 0; b < blocks; ++b) {
                append(engine.execute_zero());
            }
        } else if (nbits == 32) {
            const std::vector<float> zeros(nbeams * plan.get_nchans() * nsamps,
                                           0.0F);
            for (uint64_t b = 0; b < blocks; ++b) {
                append(engine.execute(std::span<const float>(zeros)));
            }
        } else {
            const std::vector<uint8_t> zeros(
                nbeams * plan.get_nchans() *
                    bit_pack_utils::packed_row_bytes(nsamps, nbits),
                0);
            for (uint64_t b = 0; b < blocks; ++b) {
                append(engine.execute(std::span<const uint8_t>(zeros)));
            }
        }
    }
    append(engine.flush());
    reduce_drained(state.candidates, counts, state.plane_counts, config,
                   planes);
    state.drained = true;
    state.end     = end;
    CandidateResultView view;
    static_cast<CandidateCounts&>(view) = counts;
    view.candidates                     = state.candidates;
    view.plane_counts                   = state.plane_counts;
    view.input_sample_begin             = end;
    view.input_nsamps                   = 0;
    return view;
}

} // namespace dmt::algorithms::detail
