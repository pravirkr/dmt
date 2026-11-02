#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"

// Direct low levels of the FDMT (GPU, valid mode): level K computed straight
// from the input instead of materialising levels 0..K-1.
//
// A level-K row is the sum of its leaves: the level-0 rows (channel c, box
// of s + 1 samples with box-mean weight w, or the single sample t - s
// without box smearing) that the tree adds into it, each delayed by the sum
// of the head delays on its path. With P_c the running sum of channel c over
// a chunk's window, a leaf is w * (P_c[t - shift + 1] - P_c[t - shift - s]):
// two reads per leaf from shared memory, where the tree would stream two
// whole levels per merge through L2/DRAM. Integer-valued input gives the
// tree's values exactly; float input agrees within rounding.
//
// The tree histories of levels 1..K (the CPU engine's save_history format)
// are not maintained while levels are computed directly; they are rebuilt
// from the input window when needed (save_history, a stepper block) with the
// per-row leaf tables below.

namespace dmt::algorithms::detail {

struct FDMTDirectLeaf {
    int32_t c;     ///< channel (relative to the subband for the level-K table)
    int32_t s;     ///< box length - 1 (0 for a single sample)
    int32_t shift; ///< accumulated head delay (+ s without box smearing)
    float w;       ///< box-mean weight (1 without box smearing)
};

/// One level-K subband: its channel range and rows.
struct FDMTDirectSub {
    int32_t c0;    ///< first channel
    int32_t nc;    ///< channels
    int32_t row0;  ///< first level-K row
    int32_t nrows; ///< level-K rows
    int32_t span;  ///< samples before t any of its rows reads
};

/// A level 1..K merge coordinate's head row, for rebuilding its history.
struct FDMTDirectHistCoord {
    int32_t leaf_lo, leaf_hi; ///< head row's leaves in hist_leaves
    int32_t delay;
    uint32_t hist_offset;
};

struct FDMTDirectPlan {
    SizeType levels{0};
    std::vector<FDMTDirectSub> subs;
    std::vector<int32_t> leaf_off;      ///< per level-K row (size rows + 1)
    std::vector<FDMTDirectLeaf> leaves; ///< sorted by channel within a row
    int32_t window{0}; ///< input samples kept before a chunk (max span)
    int32_t max_nc{0};
    int32_t max_rows{0};
    std::vector<FDMTDirectLeaf> hist_leaves; ///< absolute channels
    std::vector<std::vector<FDMTDirectHistCoord>> hist_coords; ///< levels 1..K
};

namespace fdmt_direct_impl {

inline void collect_leaves(const plans::FDMTPlanContainer& pc,
                           SizeType level,
                           SizeType row,
                           int32_t shift,
                           bool smear,
                           const std::vector<float>& box_w,
                           std::vector<FDMTDirectLeaf>& out) {
    const auto& c = pc.coordinates[level][row];
    if (level == 0) {
        const auto dt = pc.grids[0][c.i_sub].dt_grid[c.i_dt];
        const auto s  = static_cast<int32_t>(dt < 0 ? -dt : dt);
        if (smear) {
            out.push_back({static_cast<int32_t>(c.i_sub), s, shift,
                           box_w[static_cast<SizeType>(s)]});
        } else {
            out.push_back({static_cast<int32_t>(c.i_sub), 0, shift + s, 1.0F});
        }
        return;
    }
    // The operands as the merge kernels read them: the row at
    // tail_buf_offset undelayed, the one at head_buf_offset delayed (for
    // negative dispersion that is the i_coord_tail subband).
    const auto stride = pc.state_shape[level - 1].nsamps;
    collect_leaves(pc, level - 1, c.tail_buf_offset / stride, shift, smear,
                   box_w, out);
    if (c.head_buf_offset != SIZE_MAX) {
        collect_leaves(pc, level - 1, c.head_buf_offset / stride,
                       shift + static_cast<int32_t>(c.delay), smear, box_w,
                       out);
    }
}

inline int32_t extent(const std::vector<FDMTDirectLeaf>& leaves) {
    int32_t e = 0;
    for (const auto& lf : leaves) {
        e = std::max(e, lf.shift + lf.s);
    }
    return e;
}

} // namespace fdmt_direct_impl

/**
 * @brief Tables of the direct evaluation of level `levels` (>= 1, below the
 * root) of a valid-mode plan. `box_w[s]` is the float box-mean weight of a
 * box of s + 1 samples (unused without box smearing). Level-K rows must be
 * laid out row-major (buf_offset = row * nsamps), as the plan does.
 */
inline FDMTDirectPlan build_fdmt_direct_plan(const plans::FDMTPlan& plan,
                                             SizeType levels,
                                             bool smear,
                                             const std::vector<float>& box_w) {
    using fdmt_direct_impl::collect_leaves;
    using fdmt_direct_impl::extent;
    const auto& pc = plan.get_container();
    if (levels == 0 || levels >= plan.get_niters()) {
        throw std::invalid_argument("FDMT direct levels: depth out of range");
    }
    FDMTDirectPlan d;
    d.levels = levels;
    d.leaf_off.push_back(0);
    for (const auto& g : pc.grids[levels]) {
        FDMTDirectSub sub{INT32_MAX, 0, static_cast<int32_t>(g.coord_offset),
                          static_cast<int32_t>(g.ndt), 0};
        std::vector<std::vector<FDMTDirectLeaf>> rows(g.ndt);
        int32_t cmax = 0;
        for (SizeType i = 0; i < g.ndt; ++i) {
            const auto row = g.coord_offset + i;
            if (pc.coordinates[levels][row].buf_offset !=
                row * pc.state_shape[levels].nsamps) {
                throw std::logic_error(
                    "FDMT direct levels: level rows are not row-major");
            }
            collect_leaves(pc, levels, row, 0, smear, box_w, rows[i]);
            for (const auto& lf : rows[i]) {
                sub.c0 = std::min(sub.c0, lf.c);
                cmax   = std::max(cmax, lf.c);
            }
            sub.span = std::max(sub.span, extent(rows[i]));
        }
        sub.nc = cmax - sub.c0 + 1;
        for (auto& row : rows) {
            for (auto& lf : row) {
                lf.c -= sub.c0;
            }
            // Channel order: a shared-memory pass covers a run of channels.
            std::ranges::stable_sort(
                row, {}, [](const FDMTDirectLeaf& lf) { return lf.c; });
            d.leaves.insert(d.leaves.end(), row.begin(), row.end());
            d.leaf_off.push_back(static_cast<int32_t>(d.leaves.size()));
        }
        d.window   = std::max(d.window, sub.span);
        d.max_nc   = std::max(d.max_nc, sub.nc);
        d.max_rows = std::max(d.max_rows, sub.nrows);
        d.subs.push_back(sub);
    }
    // Levels 1..K tree histories: each merge's head row (level l - 1) over
    // the last `delay` samples.
    d.hist_coords.resize(levels + 1);
    for (SizeType l = 1; l <= levels; ++l) {
        for (const auto& c : pc.coordinates_sum[l]) {
            if (c.delay == 0) {
                continue;
            }
            std::vector<FDMTDirectLeaf> head;
            collect_leaves(pc, l - 1,
                           c.head_buf_offset / pc.state_shape[l - 1].nsamps, 0,
                           smear, box_w, head);
            const auto lo = static_cast<int32_t>(d.hist_leaves.size());
            d.hist_leaves.insert(d.hist_leaves.end(), head.begin(), head.end());
            d.hist_coords[l].push_back(
                {lo, static_cast<int32_t>(d.hist_leaves.size()),
                 static_cast<int32_t>(c.delay),
                 static_cast<uint32_t>(c.hist_offset)});
            d.window = std::max(d.window,
                                static_cast<int32_t>(c.delay) + extent(head));
        }
    }
    return d;
}

/// A level-k row (index) delayed by `shift`, in the direct top.
struct FDMTDirectTopLeaf {
    int32_t row;
    int32_t shift;
};

/**
 * @brief The direct top: every final trial as the sum of its level-k rows at
 * their accumulated head delays, instead of merging levels k+1..root. A
 * level-k row keeps a ring of its last `ring_len` samples (the largest delay
 * any trial applies to it), so a chunk's trial reads the row's state for
 * t >= shift and its ring before. The replaced levels' tree histories are
 * rebuilt from the rings when needed (as for the direct low levels).
 */
struct FDMTDirectTopPlan {
    SizeType level{0};                     ///< k
    std::vector<int32_t> out_row;          ///< dmt row of each final trial
    std::vector<int32_t> leaf_off;         ///< per final trial (size + 1)
    std::vector<FDMTDirectTopLeaf> leaves; ///< tree order
    std::vector<int32_t> ring_len;         ///< per level-k row (0: unused)
    std::vector<int64_t> ring_off;         ///< per level-k row
    int64_t ring_size{0};                  ///< per beam
    int32_t max_ring{0};
    std::vector<FDMTDirectTopLeaf> hist_leaves;
    std::vector<std::vector<FDMTDirectHistCoord>> hist_coords; ///< by level
};

namespace fdmt_direct_impl {

inline void collect_top(const plans::FDMTPlanContainer& pc,
                        SizeType k,
                        SizeType level,
                        SizeType row,
                        int32_t shift,
                        std::vector<FDMTDirectTopLeaf>& out) {
    if (level == k) {
        out.push_back({static_cast<int32_t>(row), shift});
        return;
    }
    const auto& c     = pc.coordinates[level][row];
    const auto stride = pc.state_shape[level - 1].nsamps;
    collect_top(pc, k, level - 1, c.tail_buf_offset / stride, shift, out);
    if (c.head_buf_offset != SIZE_MAX) {
        collect_top(pc, k, level - 1, c.head_buf_offset / stride,
                    shift + static_cast<int32_t>(c.delay), out);
    }
}

} // namespace fdmt_direct_impl

/// Tables of the direct top from level `k` (1 <= k < root - 1) of a
/// valid-mode plan.
inline FDMTDirectTopPlan build_fdmt_direct_top(const plans::FDMTPlan& plan,
                                               SizeType k) {
    using fdmt_direct_impl::collect_top;
    const auto& pc  = plan.get_container();
    const auto root = plan.get_niters();
    if (k == 0 || k + 1 >= root) {
        throw std::invalid_argument("FDMT direct top: level out of range");
    }
    FDMTDirectTopPlan t;
    t.level = k;
    t.leaf_off.push_back(0);
    const auto nrows = pc.state_shape[k].ncoords;
    t.ring_len.assign(nrows, 0);
    for (SizeType f = 0; f < pc.coordinates[root].size(); ++f) {
        const auto& c = pc.coordinates[root][f];
        t.out_row.push_back(
            static_cast<int32_t>(c.buf_offset / pc.state_shape[root].nsamps));
        collect_top(pc, k, root, f, 0, t.leaves);
        t.leaf_off.push_back(static_cast<int32_t>(t.leaves.size()));
    }
    for (const auto& lf : t.leaves) {
        auto& len = t.ring_len[static_cast<SizeType>(lf.row)];
        len       = std::max(len, lf.shift);
    }
    // Levels k+1..root tree histories: each merge's head row over the last
    // `delay` samples, as level-k rows (rings extended where a merge reaches
    // further than any trial through it).
    t.hist_coords.resize(root + 1);
    for (SizeType l = k + 1; l <= root; ++l) {
        for (const auto& c : pc.coordinates_sum[l]) {
            if (c.delay == 0) {
                continue;
            }
            const auto lo = static_cast<int32_t>(t.hist_leaves.size());
            collect_top(pc, k, l - 1,
                        c.head_buf_offset / pc.state_shape[l - 1].nsamps, 0,
                        t.hist_leaves);
            for (auto m = static_cast<SizeType>(lo); m < t.hist_leaves.size();
                 ++m) {
                auto& len =
                    t.ring_len[static_cast<SizeType>(t.hist_leaves[m].row)];
                len = std::max(len, static_cast<int32_t>(c.delay) +
                                        t.hist_leaves[m].shift);
            }
            t.hist_coords[l].push_back(
                {lo, static_cast<int32_t>(t.hist_leaves.size()),
                 static_cast<int32_t>(c.delay),
                 static_cast<uint32_t>(c.hist_offset)});
        }
    }
    t.ring_off.resize(nrows);
    for (SizeType r = 0; r < nrows; ++r) {
        t.ring_off[r] = t.ring_size;
        t.ring_size += t.ring_len[r];
        t.max_ring = std::max(t.max_ring, t.ring_len[r]);
    }
    return t;
}

} // namespace dmt::algorithms::detail
