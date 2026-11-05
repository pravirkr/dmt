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

/// Edge taper width of the symmetric {1, 2, 1} trapezoidal filter of total
/// support `width`. Widths 1 and 2 are pure boxcars; widths >= 3 use edge
/// width floor((width + 1) / 4) and core width `width - 2 * w_edge`.
DMT_HD constexpr uint32_t trapezoid_edge_width(uint32_t width) noexcept {
    return width >= 3U ? (width + 1U) / 4U : 0U;
}

/// Energy of a symmetric {1, 2, 1}-weighted trapezoidal filter of total
/// support `width` and edge taper `w_edge` (`w_core = width - 2 * w_edge`)
/// convolved with a weighted length-L leaf box.
inline double fdmt_trapezoid_variance(SizeType len,
                                      SizeType width,
                                      SizeType w_edge,
                                      double weight) noexcept {
    if (w_edge == 0) {
        return fdmt_box_variance(len, static_cast<double>(width), weight);
    }
    const double w_outer = static_cast<double>(width);
    const double w_core  = static_cast<double>(width - 2 * w_edge);
    const double w_eff   = static_cast<double>(width - w_edge);
    const double edge    = static_cast<double>(w_edge);
    return fdmt_box_variance(len, w_outer, weight) +
           fdmt_box_variance(len, w_core, weight) +
           2.0 * (fdmt_box_variance(len, w_eff, weight) -
                  fdmt_box_variance(len, edge, weight));
}

} // namespace dmt::algorithms::detail
