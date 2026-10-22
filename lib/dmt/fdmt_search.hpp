#pragma once

#include <cstdint>

#include "dmt/common/backend.hpp"
#include "dmt/common/types.hpp"

namespace dmt::algorithms::detail {

/**
 * @brief One chunk available to a search sink, before state/history reuse.
 *
 * Below the root, state is the produced level; at the root it is the
 * preceding level, and the sink reads merge operands from the plan. CPU
 * histories are stream-indexed rings; GPU histories are oldest-first FIFOs.
 * FDMT owns and advances all merge histories after the root sink returns.
 */
struct FDMTSearchInput {
    const float* state      = nullptr;
    const float* history    = nullptr;
    SizeType row_stride     = 0;
    SizeType beam_stride    = 0;
    SizeType history_stride = 0;
    SizeType nsamps         = 0;
    uint64_t sample_begin   = 0;
    bool materialized_root  = false; ///< Private separated-path verification.
    bool history_ring       = false;
    Stream stream{};
    const uint64_t* integer_state   = nullptr;
    const uint64_t* integer_history = nullptr;
};

class FDMTSearchSink {
public:
    virtual ~FDMTSearchSink() = default;
    virtual void consume(SizeType level, const FDMTSearchInput& input) = 0;
};

} // namespace dmt::algorithms::detail
