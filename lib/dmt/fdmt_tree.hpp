#pragma once

#include <algorithm>
#include <utility>

#include "dmt/common/plans.hpp"

// Channel groups of the FDMT merge tree for level fusion: level-F tree
// subband g and everything below it. The tree halves the band top-down (see
// plans::FDMTCoordGrid), so a group's descendants at every level l <= F are
// a contiguous run of that level's tree subbands, covering the group's
// channels; staggered subbands never lie inside a fused group
// (FDMTPlan::get_max_fuse_levels()).

namespace dmt::algorithms::detail {

/// @brief Tree subbands [first, last) of level @p l that descend from
/// level-@p fuse tree subband @p g (l <= fuse).
inline std::pair<SizeType, SizeType> fdmt_group_subbands(
    const plans::FDMTPlanContainer& pc, SizeType fuse, SizeType g, SizeType l) {
    const auto& top   = pc.grids[fuse][g];
    const auto& grids = pc.grids[l];
    const auto end =
        grids.begin() + static_cast<std::ptrdiff_t>(pc.state_shape[l].nchans);
    const auto by_chan = [](const plans::FDMTCoordGrid& s, SizeType chan) {
        return s.chan_begin < chan;
    };
    const auto first =
        std::lower_bound(grids.begin(), end, top.chan_begin, by_chan);
    const auto last = std::lower_bound(first, end, top.chan_end, by_chan);
    return {static_cast<SizeType>(first - grids.begin()),
            static_cast<SizeType>(last - grids.begin())};
}

/// @brief Coordinates [begin, end) of level @p l owned by fused group @p g.
inline std::pair<SizeType, SizeType> fdmt_group_coords(
    const plans::FDMTPlanContainer& pc, SizeType fuse, SizeType g, SizeType l) {
    const auto [first, last] = fdmt_group_subbands(pc, fuse, g, l);
    const auto& grids        = pc.grids[l];
    return {grids[first].coord_offset,
            grids[last - 1].coord_offset + grids[last - 1].ndt};
}

/// @brief Input channels [begin, end) of fused group @p g.
inline std::pair<SizeType, SizeType> fdmt_group_channels(
    const plans::FDMTPlanContainer& pc, SizeType fuse, SizeType g) {
    const auto& top = pc.grids[fuse][g];
    return {top.chan_begin, top.chan_end};
}

} // namespace dmt::algorithms::detail
