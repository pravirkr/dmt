#pragma once

#include <algorithm>

#include "dmt/common/types.hpp"

namespace dmt::algorithms::detail {

/// Energy of a width-W boxcar convolved with a weighted length-L leaf box.
inline double
fdmt_box_variance(SizeType len, double width, double weight) noexcept {
    const double h = std::max(static_cast<double>(len), width);
    const double l = std::min(static_cast<double>(len), width);
    return weight * weight *
           (((h - l + 1.0) * (l * l)) +
            ((((2.0 * l) - 1.0) * l * (l - 1.0)) / 3.0));
}

} // namespace dmt::algorithms::detail
