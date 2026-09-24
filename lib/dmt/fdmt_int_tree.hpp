#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "dmt/bit_pack_utils.hpp"
#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"

// Narrow-integer FDMT tree state (FDMTExecConfig::int_tree), shared by the
// CPU and CUDA backends so both pick identical per-level storage types.

namespace dmt::algorithms::detail {

/// Storage type of one FDMT tree level. Ordered by width: a level is never
/// narrower than the level before it.
enum class FDMTLevelType : uint8_t { kU8 = 0, kU16 = 1, kF32 = 2 };

/**
 * @brief Per-level storage types for packed `nbits` input with int_tree.
 *
 * The exact max value of every coordinate is propagated through the plan's
 * DAG (level 0: maxval * box width; then bound(tail) + bound(head)), and a
 * level is stored as uint8/uint16 only if its largest bound fits. All bounds
 * stay far below 2^24, so the integer sums equal the float path bit-for-bit.
 * The root level is always float (it is the output), and the first float
 * level is moved one level earlier when needed so that, with the float
 * levels ping-ponging between the output buffer (levels with niters - l
 * even) and the internal buffer, the first float level lands in the output
 * buffer -- never on top of the integer ping-pong halves, which live inside
 * the internal buffer.
 */
inline std::vector<FDMTLevelType> int_tree_level_types(
    const plans::FDMTPlan& plan, bool use_box_smearing, SizeType nbits) {
    const auto& pc        = plan.get_container();
    const SizeType niters = plan.get_niters();
    const auto maxval     = static_cast<double>(utils::max_sample_value(nbits));
    const auto type_for   = [](double bound) {
        if (bound <= 255.0) {
            return FDMTLevelType::kU8;
        }
        if (bound <= 65535.0) {
            return FDMTLevelType::kU16;
        }
        return FDMTLevelType::kF32;
    };

    std::vector<FDMTLevelType> types(niters + 1, FDMTLevelType::kF32);
    std::vector<double> prev(pc.state_shape[0].ncoords, 0.0);
    for (const auto& grid : pc.grids[0]) {
        for (SizeType i_dt = 0; i_dt < grid.ndt; ++i_dt) {
            const auto width =
                use_box_smearing
                    ? static_cast<double>(std::abs(grid.dt_grid[i_dt]) + 1)
                    : 1.0;
            prev[grid.coord_offset + i_dt] = maxval * width;
        }
    }
    types[0] = type_for(*std::max_element(prev.begin(), prev.end()));
    for (SizeType l = 1; l <= niters; ++l) {
        const auto& coords = pc.coordinates[l];
        std::vector<double> cur(coords.size(), 0.0);
        for (SizeType k = 0; k < coords.size(); ++k) {
            cur[k] = prev[coords[k].i_coord_tail];
            if (coords[k].i_coord_head != SIZE_MAX) {
                cur[k] += prev[coords[k].i_coord_head];
            }
        }
        types[l] = std::max(
            types[l - 1], type_for(*std::max_element(cur.begin(), cur.end())));
        prev.swap(cur);
    }
    types[niters]        = FDMTLevelType::kF32;
    const auto first_f32 = static_cast<SizeType>(
        std::find(types.begin(), types.end(), FDMTLevelType::kF32) -
        types.begin());
    if (first_f32 > 0 && (niters - first_f32) % 2 == 1) {
        types[first_f32 - 1] = FDMTLevelType::kF32;
    }
    return types;
}

} // namespace dmt::algorithms::detail
