#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"

#include "dmt/bit_pack_utils.hpp"
#include "dmt/fdmt_box_weights.hpp"

// Narrow-integer FDMT tree state (the `int_tree` constructor parameter),
// shared by the CPU and CUDA backends so both pick identical per-level
// storage types.

namespace dmt::algorithms::detail {

/// Storage type of one FDMT tree level. Ordered by width: a level is never
/// narrower than the level before it.
enum class FDMTLevelType : uint8_t { kU8 = 0, kU16 = 1, kF32 = 2 };

/**
 * @brief Per-level storage types for packed `nbits` input with int_tree.
 *
 * The exact max value of every coordinate is propagated through the plan's
 * DAG (level 0: with box smearing the rounded box mean of maxval samples in
 * units of 2^-F, see fdmt_box_weights.hpp, else maxval; then bound(tail) +
 * bound(head)), and a level is stored as uint8/uint16 only if its largest
 * bound fits. Integer levels hold exact integers; the int -> float
 * conversion scales by 2^-F, which commutes with float rounding, so the
 * integer path equals the float path (int_tree off, same packed input)
 * bit-for-bit.
 * The root level is always float (it is the output). The engines keep
 * every level below the root in one of two state halves, alternating, so
 * any mix of storage types is laid out safely.
 */
inline std::vector<FDMTLevelType> int_tree_level_types(
    const plans::FDMTPlan& plan, bool use_box_smearing, SizeType nbits) {
    const auto& pc        = plan.get_container();
    const SizeType niters = plan.get_niters();
    const auto maxval =
        static_cast<double>(bit_pack_utils::max_sample_value(nbits));
    const auto type_for = [](double bound) {
        if (bound <= 255.0) {
            return FDMTLevelType::kU8;
        }
        if (bound <= 65535.0) {
            return FDMTLevelType::kU16;
        }
        return FDMTLevelType::kF32;
    };

    std::vector<FDMTLevelType> types(niters + 1, FDMTLevelType::kF32);
    const SizeType scale_f = box_mean_bits(pc.state_shape[0].dt_max + 1);
    std::vector<double> prev(pc.state_shape[0].ncoords, 0.0);
    for (const auto& grid : pc.grids[0]) {
        for (SizeType i_dt = 0; i_dt < grid.ndt; ++i_dt) {
            const auto len =
                static_cast<SizeType>(std::abs(grid.dt_grid[i_dt])) + 1;
            const auto box_max = static_cast<uint64_t>(maxval) * len;
            prev[grid.coord_offset + i_dt] =
                use_box_smearing
                    ? static_cast<double>(box_round_apply<uint64_t>(
                          box_max, box_round(len, scale_f)))
                    : maxval;
        }
    }
    types[0] = type_for(*std::ranges::max_element(prev));
    for (SizeType l = 1; l <= niters; ++l) {
        const auto& coords = pc.coordinates[l];
        std::vector<double> cur(coords.size(), 0.0);
        for (SizeType k = 0; k < coords.size(); ++k) {
            cur[k] = prev[coords[k].i_coord_tail];
            if (coords[k].i_coord_head != SIZE_MAX) {
                cur[k] += prev[coords[k].i_coord_head];
            }
        }
        types[l] =
            std::max(types[l - 1], type_for(*std::ranges::max_element(cur)));
        prev.swap(cur);
    }
    types[niters] = FDMTLevelType::kF32;
    return types;
}

} // namespace dmt::algorithms::detail
