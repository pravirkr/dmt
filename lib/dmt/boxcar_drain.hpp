#pragma once

// End of stream for the boxcar searches (FDMTBoxcar::drain(),
// FDMTFFTBoxcar::drain()): search the samples the stream has not finished
// searching, then emit the open excursions.

#include <algorithm>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

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

    void reset() noexcept {
        drained = false;
        end     = 0;
        candidates.clear();
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
 * Ends the stream at its current position `end`: endpoints at or after `end`
 * are never searched (FDMTSearchInput::search_stop), ceil(latency / nsamps)
 * blocks of padding move every row's search up to `end`, and the open
 * excursions are emitted. The FDMT is causal in its f_min reference, so its
 * rows before `end` do not depend on the padding (zero codes are used for
 * packed input); with fractional FFT delays the padding is zero signal, the
 * band-limited interpolation of a stream that ends.
 */
inline CandidateResultView drain_stream(FDMTBoxcarEngine& engine,
                                        const plans::FDMTPlan& plan,
                                        SizeType nbeams,
                                        SizeType nbits,
                                        uint64_t latency,
                                        DrainState& state) {
    state.check_open("drain");
    const uint64_t end = engine.stream_position();
    state.candidates.clear();
    CandidateCounts counts{};
    const auto append = [&](const CandidateResultView& v) {
        state.candidates.insert(state.candidates.end(), v.candidates.begin(),
                                v.candidates.end());
        counts.emitted_count += v.emitted_count;
        counts.total_detected += v.total_detected;
        counts.dropped_global += v.dropped_global;
        counts.dropped_per_dm += v.dropped_per_dm;
    };
    engine.set_search_stop(end);
    if (end > 0 && latency > 0) {
        const SizeType nsamps = plan.get_nsamps();
        const uint64_t blocks = (latency + nsamps - 1) / nsamps;
        if (nbits == 32) {
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
    state.drained = true;
    state.end     = end;
    CandidateResultView view;
    static_cast<CandidateCounts&>(view) = counts;
    view.candidates                     = state.candidates;
    view.input_sample_begin             = end;
    view.input_nsamps                   = 0;
    return view;
}

} // namespace dmt::algorithms::detail
