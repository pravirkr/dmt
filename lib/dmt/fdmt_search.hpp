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
};

class FDMTSearchSink {
public:
    virtual ~FDMTSearchSink() = default;
    virtual void consume(SizeType level, const FDMTSearchInput& input) = 0;
};

} // namespace dmt::algorithms::detail
