#pragma once

#include "dmt/boxcar.hpp"

namespace dmt::algorithms::detail {

/// Associative summary of an ordered segment: its trailing above-threshold
/// run, and whether it contains any below-threshold point. An empty segment
/// is the identity. A prefix scan supplies each worker's incoming run.
struct BoxcarRunSummary {
    BoxcarPeak trailing{};
    bool all = true;
};

DMT_HD inline BoxcarRunSummary boxcar_join(BoxcarRunSummary left,
                                           BoxcarRunSummary right,
                                           bool integer = false) noexcept {
    if (!right.all) {
        return right;
    }
    right.all = left.all;
    if (!right.trailing.open) {
        right.trailing = left.trailing;
    } else if (left.trailing.open) {
        const uint64_t end = right.trailing.end;
        const bool better =
            integer
                ? boxcar_integer_better(right.trailing.raw_sum,
                                        right.trailing.peak,
                                        right.trailing.filter, left.trailing)
                : boxcar_better(right.trailing.snr, right.trailing.peak,
                                right.trailing.filter, left.trailing);
        if (!better) {
            right.trailing = left.trailing;
        }
        right.trailing.begin = left.trailing.begin;
        right.trailing.end   = end;
    }
    return right;
}

} // namespace dmt::algorithms::detail
