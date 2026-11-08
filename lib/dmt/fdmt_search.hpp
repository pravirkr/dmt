#pragma once

#include <cstdint>

#include "dmt/common/backend.hpp"
#include "dmt/common/types.hpp"

namespace dmt::algorithms::detail {

/**
 * @brief One chunk available to a search sink, before state/history reuse.
 *
 * Below the root, state is the produced level; at the root it is the
 * preceding level, and the sink reads merge operands from the plan. Root
 * histories are rings indexed by stream sample (history_ring) or
 * oldest-first FIFOs. FDMT owns and advances all merge histories after the
 * root sink returns.
 */
struct FDMTSearchInput {
    const float* state      = nullptr;
    const float* prev_state = nullptr;
    const float* history    = nullptr;
    SizeType row_stride     = 0;
    SizeType beam_stride    = 0;
    SizeType history_stride = 0;
    /// Subtracted from a row's plan hist_offset (GPU: the ring holds only
    /// the levels above the fused FIFO prefix).
    SizeType history_base = 0;
    SizeType nsamps       = 0;
    uint64_t sample_begin = 0;
    /// Root rows are stored rows (separated-path verification, FDMT-FFT).
    bool materialized_root = false;
    bool history_ring      = false;
    /// Beams [beam_begin, beam_begin + beam_count) are present; beam b is at
    /// state + b * beam_stride (beam_stride 0 for a single-beam buffer).
    SizeType beam_begin = 0;
    SizeType beam_count = SIZE_MAX;
    Stream stream{};
    const uint64_t* integer_state        = nullptr;
    const uint64_t* integer_prev_state   = nullptr;
    const uint64_t* integer_history      = nullptr;
    const uint32_t* integer32_state      = nullptr;
    const uint32_t* integer32_prev_state = nullptr;
    const uint32_t* integer32_history    = nullptr;
    /// Lagged rows (skewed-window subband search, fdmt_skew.hpp): row r of
    /// the level starts at element row_offset[r] of the state (row_stride is
    /// not used) and holds stream samples [sample_begin - row_lag[r],
    /// sample_begin - row_lag[r] + nsamps). Samples before stream sample 0
    /// are not searched. Host pointers on the CPU, device on the GPU; null
    /// for ordinary rows. Rows are materialized (materialized_root).
    const SizeType* row_offset = nullptr;
    const uint64_t* row_lag    = nullptr;
};

/// The single-row view of `in` for row `row` of a lagged input (row_lag
/// set): the state pointers advanced to the row's first in-stream sample,
/// row_stride 0, sample_begin and nsamps that row's own. False when none
/// of the call's samples of the row are in the stream yet. Single beam.
DMT_HD inline bool lagged_row_input(const FDMTSearchInput& in,
                                    SizeType row,
                                    FDMTSearchInput& out) noexcept {
    const uint64_t lag  = in.row_lag[row];
    const uint64_t back = in.sample_begin >= lag ? 0 : lag - in.sample_begin;
    const uint64_t skip = back < in.nsamps ? back : in.nsamps;
    if (skip >= in.nsamps) {
        return false;
    }
    out              = in;
    const auto off   = in.row_offset[row] + static_cast<SizeType>(skip);
    out.row_stride   = 0;
    out.beam_stride  = 0;
    out.row_offset   = nullptr;
    out.row_lag      = nullptr;
    out.nsamps       = in.nsamps - static_cast<SizeType>(skip);
    out.sample_begin = in.sample_begin + skip - lag;
    if (in.state != nullptr) {
        out.state = in.state + off;
    }
    if (in.integer_state != nullptr) {
        out.integer_state = in.integer_state + off;
    }
    if (in.integer32_state != nullptr) {
        out.integer32_state = in.integer32_state + off;
    }
    return true;
}

class FDMTSearchSink {
public:
    virtual ~FDMTSearchSink() = default;
    virtual void consume(SizeType level, const FDMTSearchInput& input) = 0;
};

} // namespace dmt::algorithms::detail
