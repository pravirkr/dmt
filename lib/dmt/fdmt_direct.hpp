#pragma once

#include <algorithm>
#include <array>
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

/// One level-K subband (or row slice in skewed mode): its channel range and
/// rows.
struct FDMTDirectSub {
    int32_t c0;         ///< first channel
    int32_t nc;         ///< channels
    int32_t row0;       ///< first level-K row
    int32_t nrows;      ///< level-K rows
    int32_t span;       ///< samples before t any of its rows reads
    int32_t c_min[8]{}; ///< per-channel shift subtracted from leaves (skewed)
    int32_t extra{0};   ///< max local halo samples beyond active (skewed mode)
    int32_t out_off{0}; ///< element offset of row0 in level-K state (skewed)
};

/// A level 1..K merge coordinate's head row, for rebuilding its history.
struct FDMTDirectHistCoord {
    int32_t leaf_lo, leaf_hi; ///< head row's leaves in hist_leaves
    int32_t delay;
    uint32_t hist_offset;
};

struct FDMTDirectPlan {
    SizeType levels{0};
    bool skewed{false};
    std::vector<FDMTDirectSub> subs;
    std::vector<int32_t> leaf_off;      ///< per level-K row (size rows + 1)
    std::vector<FDMTDirectLeaf> leaves; ///< sorted by channel within a row
    int32_t window{0}; ///< input samples kept before a chunk (max span)
    int32_t max_span{0};
    int32_t max_nc{0};
    int32_t max_rows{0};
    int32_t max_extra{0};
    int32_t total_elems{0};             ///< level-K elements per beam (skewed)
    std::vector<int32_t> batch_sub_off; ///< slice offsets per subband batch
    std::vector<int32_t> row_off;       ///< per-row offset in level-K state
    std::vector<int32_t> chan_ring_len; ///< per-channel ring length (skewed)
    std::vector<int64_t> chan_ring_off; ///< per-channel ring offset (skewed)
    int64_t win_ring_size{0}; ///< total ring elements per beam (skewed)
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
    d.max_span = d.window;
    return d;
}

/**
 * @brief Skewed-window direct evaluation of level `levels`: each row `r` is
 * evaluated at stream shift `s_hi_k[r]` over `active + sp_k[r]` samples,
 * subbands are sliced so every slice's relative shift span fits shared memory,
 * and raw channel history is kept in a ragged per-channel circular ring of
 * length `chan_reach[c] + chunk`.
 */
inline FDMTDirectPlan build_fdmt_direct_skewed_plan(
    const plans::FDMTPlan& plan,
    SizeType levels,
    bool smear,
    const std::vector<float>& box_w,
    SizeType chunk,
    const std::vector<int32_t>& s_hi_k,
    const std::vector<int32_t>& sp_k,
    const std::vector<SizeType>& batch_grid_off = {}) {
    using fdmt_direct_impl::collect_leaves;
    const auto& pc = plan.get_container();
    if (levels == 0 || levels >= plan.get_niters()) {
        throw std::invalid_argument("FDMT direct levels: depth out of range");
    }
    constexpr int32_t kMaxSliceRows        = 16;
    constexpr int32_t kMaxSliceShiftSpread = 256;
    constexpr int32_t kMaxSliceSpan        = 8192;
    FDMTDirectPlan d;
    d.levels = levels;
    d.skewed = true;
    d.leaf_off.push_back(0);
    d.row_off.assign(pc.state_shape[levels].ncoords, 0);
    const auto nchans     = plan.get_nchans();
    const SizeType ngrids = pc.grids[levels].size();
    const SizeType ebatches =
        batch_grid_off.size() >= 2 ? (batch_grid_off.size() - 1) : 1;
    std::vector<int32_t> chan_reach(nchans, 0);
    for (SizeType b = 0; b < ebatches; ++b) {
        const SizeType g_lo =
            batch_grid_off.size() >= 2 ? batch_grid_off[b] : 0;
        const SizeType g_hi =
            batch_grid_off.size() >= 2 ? batch_grid_off[b + 1] : ngrids;
        d.batch_sub_off.push_back(static_cast<int32_t>(d.subs.size()));
        int32_t out_off = 0;
        for (SizeType g_idx = g_lo; g_idx < g_hi; ++g_idx) {
            const auto& g = pc.grids[levels][g_idx];
            int32_t c0    = INT32_MAX;
            int32_t cmax  = 0;
            std::vector<std::vector<FDMTDirectLeaf>> rows(g.ndt);
            std::vector<SizeType> used_i;
            used_i.reserve(g.ndt);
            for (SizeType i = 0; i < g.ndt; ++i) {
                const auto row = g.coord_offset + i;
                if (pc.coordinates[levels][row].buf_offset !=
                    row * pc.state_shape[levels].nsamps) {
                    throw std::logic_error(
                        "FDMT direct levels: level rows are not row-major");
                }
                if (s_hi_k[row] < 0) {
                    continue;
                }
                collect_leaves(pc, levels, row, s_hi_k[row], smear, box_w,
                               rows[i]);
                if (rows[i].empty()) {
                    continue;
                }
                used_i.push_back(i);
                for (const auto& lf : rows[i]) {
                    c0   = std::min(c0, lf.c);
                    cmax = std::max(cmax, lf.c);
                    chan_reach[static_cast<SizeType>(lf.c)] =
                        std::max(chan_reach[static_cast<SizeType>(lf.c)],
                                 lf.shift + lf.s);
                    d.window = std::max(d.window, lf.shift + lf.s);
                }
            }
            if (used_i.empty() || c0 == INT32_MAX) {
                continue;
            }
            const int32_t nc = cmax - c0 + 1;
            if (nc > 8) {
                throw std::logic_error("FDMT direct skewed: nc > 8");
            }
            d.max_nc = std::max(d.max_nc, nc);
            std::vector<std::array<int32_t, 8>> row_c_min(g.ndt);
            std::vector<std::array<int32_t, 8>> row_c_max_shift(g.ndt);
            std::vector<std::array<int32_t, 8>> row_c_ext(g.ndt);
            for (const SizeType i : used_i) {
                row_c_min[i].fill(INT32_MAX);
                row_c_max_shift[i].fill(0);
                row_c_ext[i].fill(0);
                for (const auto& lf : rows[i]) {
                    const auto rc    = static_cast<SizeType>(lf.c - c0);
                    row_c_min[i][rc] = std::min(row_c_min[i][rc], lf.shift);
                    row_c_max_shift[i][rc] =
                        std::max(row_c_max_shift[i][rc], lf.shift);
                    row_c_ext[i][rc] =
                        std::max(row_c_ext[i][rc], lf.shift + lf.s);
                }
            }
            SizeType u0 = 0;
            while (u0 < used_i.size()) {
                const SizeType i0 = used_i[u0];
                auto c_min        = row_c_min[i0];
                auto c_max_shift  = row_c_max_shift[i0];
                auto c_ext        = row_c_ext[i0];
                int32_t s_extra   = sp_k[g.coord_offset + i0];
                SizeType u1       = u0 + 1;
                while (u1 < used_i.size() &&
                       static_cast<int32_t>(u1 - u0) < kMaxSliceRows) {
                    const SizeType i1 = used_i[u1];
                    bool fits         = true;
                    for (int32_t c = 0; c < nc; ++c) {
                        const auto rc = static_cast<SizeType>(c);
                        const int32_t n_min =
                            std::min(c_min[rc], row_c_min[i1][rc]);
                        const int32_t n_max_s =
                            std::max(c_max_shift[rc], row_c_max_shift[i1][rc]);
                        const int32_t n_ext =
                            std::max(c_ext[rc], row_c_ext[i1][rc]);
                        if (n_min <= n_ext &&
                            (n_max_s - n_min > kMaxSliceShiftSpread ||
                             n_ext - n_min > kMaxSliceSpan)) {
                            fits = false;
                            break;
                        }
                    }
                    if (!fits) {
                        break;
                    }
                    for (int32_t c = 0; c < nc; ++c) {
                        const auto rc = static_cast<SizeType>(c);
                        c_min[rc]     = std::min(c_min[rc], row_c_min[i1][rc]);
                        c_max_shift[rc] =
                            std::max(c_max_shift[rc], row_c_max_shift[i1][rc]);
                        c_ext[rc] = std::max(c_ext[rc], row_c_ext[i1][rc]);
                    }
                    s_extra = std::max(s_extra, sp_k[g.coord_offset + i1]);
                    ++u1;
                }
                int32_t span = 0;
                for (int32_t c = 0; c < nc; ++c) {
                    const auto rc = static_cast<SizeType>(c);
                    if (c_min[rc] > c_ext[rc]) {
                        c_min[rc] = 0;
                        c_ext[rc] = 0;
                    }
                    span = std::max(span, c_ext[rc] - c_min[rc]);
                }
                const int32_t nrows = static_cast<int32_t>(u1 - u0);
                const auto slice_row0 =
                    static_cast<int32_t>(d.leaf_off.size() - 1);
                for (SizeType u = u0; u < u1; ++u) {
                    const SizeType i = used_i[u];
                    for (auto& lf : rows[i]) {
                        lf.c -= c0;
                        lf.shift -= c_min[static_cast<SizeType>(lf.c)];
                    }
                    std::ranges::stable_sort(
                        rows[i], {},
                        [](const FDMTDirectLeaf& lf) { return lf.c; });
                    d.leaves.insert(d.leaves.end(), rows[i].begin(),
                                    rows[i].end());
                    d.leaf_off.push_back(static_cast<int32_t>(d.leaves.size()));
                }
                const int32_t slice_stride =
                    static_cast<int32_t>(chunk) + s_extra;
                for (int32_t r = 0; r < nrows; ++r) {
                    const SizeType i = used_i[u0 + static_cast<SizeType>(r)];
                    d.row_off[g.coord_offset + i] =
                        out_off + (r * slice_stride);
                }
                FDMTDirectSub sub{.c0      = c0,
                                  .nc      = nc,
                                  .row0    = slice_row0,
                                  .nrows   = nrows,
                                  .span    = span,
                                  .c_min   = {},
                                  .extra   = s_extra,
                                  .out_off = out_off};
                for (int32_t c = 0; c < nc; ++c) {
                    sub.c_min[c] = c_min[static_cast<SizeType>(c)];
                }
                d.subs.push_back(sub);
                out_off += nrows * slice_stride;
                d.max_span  = std::max(d.max_span, span);
                d.max_rows  = std::max(d.max_rows, nrows);
                d.max_extra = std::max(d.max_extra, s_extra);
                u0          = u1;
            }
        }
        d.total_elems = std::max(d.total_elems, out_off);
    }
    d.batch_sub_off.push_back(static_cast<int32_t>(d.subs.size()));
    d.chan_ring_len.resize(nchans);
    d.chan_ring_off.resize(nchans);
    for (SizeType c = 0; c < nchans; ++c) {
        d.chan_ring_off[c] = d.win_ring_size;
        d.chan_ring_len[c] = chan_reach[c] + static_cast<int32_t>(chunk);
        d.win_ring_size += d.chan_ring_len[c];
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
 * any trial applies to it, or `s_hi - s_lo` when skewed), so a chunk's trial
 * reads the row's state for t >= shift and its ring before. The replaced
 * levels' tree histories are rebuilt from the rings when needed (as for the
 * direct low levels).
 */
struct FDMTDirectTopPlan {
    SizeType level{0};       ///< k
    bool skewed{false};      ///< row r skewed by row_s_lo[r]
    bool target_ring{false}; ///< ring per final trial instead of level-k row
    std::vector<int32_t> out_row;          ///< dmt row of each final trial
    std::vector<int32_t> leaf_off;         ///< per final trial (size + 1)
    std::vector<FDMTDirectTopLeaf> leaves; ///< tree order
    std::vector<int32_t> ring_len; ///< per level-k row (0: unused) or per trial
    std::vector<int64_t> ring_off; ///< per level-k row or per trial
    std::vector<int32_t> row_s_lo; ///< per level-k row min shift
    std::vector<int32_t> row_s_hi; ///< per level-k row max shift
    int64_t ring_size{0};          ///< per beam
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

/// Tables of the direct top from level `k` (1 <= k < target) to `target`
/// (defaults to root) of a valid-mode plan.
inline FDMTDirectTopPlan build_fdmt_direct_top(const plans::FDMTPlan& plan,
                                               SizeType k,
                                               SizeType target  = 0,
                                               bool skewed      = false,
                                               SizeType chunk   = 0,
                                               bool target_ring = false) {
    using fdmt_direct_impl::collect_top;
    const auto& pc  = plan.get_container();
    const auto root = target == 0 ? plan.get_niters() : target;
    if (k == 0 || k + 1 >= root || root > plan.get_niters()) {
        throw std::invalid_argument("FDMT direct top: level out of range");
    }
    FDMTDirectTopPlan t;
    t.level       = k;
    t.skewed      = skewed || target_ring;
    t.target_ring = target_ring;
    t.leaf_off.push_back(0);
    const auto nrows = pc.state_shape[k].ncoords;
    for (SizeType f = 0; f < pc.coordinates[root].size(); ++f) {
        const auto& c = pc.coordinates[root][f];
        t.out_row.push_back(
            static_cast<int32_t>(c.buf_offset / pc.state_shape[root].nsamps));
        collect_top(pc, k, root, f, 0, t.leaves);
        t.leaf_off.push_back(static_cast<int32_t>(t.leaves.size()));
    }
    if (target_ring) {
        t.row_s_lo.assign(nrows, 0);
        t.row_s_hi.assign(nrows, -1);
        for (const auto& lf : t.leaves) {
            t.row_s_hi[static_cast<SizeType>(lf.row)] = 0;
        }
        const SizeType ntrials = t.out_row.size();
        t.ring_len.assign(ntrials, 0);
        t.ring_off.resize(ntrials, 0);
        const auto pad = static_cast<int32_t>(chunk + 256);
        for (SizeType f = 0; f < ntrials; ++f) {
            int32_t max_shift = 0;
            for (int32_t m = t.leaf_off[f]; m < t.leaf_off[f + 1]; ++m) {
                max_shift = std::max(max_shift,
                                     t.leaves[static_cast<SizeType>(m)].shift);
            }
            t.ring_len[f] = max_shift + pad;
            t.ring_off[f] = t.ring_size;
            t.ring_size += t.ring_len[f];
            t.max_ring = std::max(t.max_ring, t.ring_len[f]);
        }
        return t;
    }
    t.ring_len.assign(nrows, 0);
    if (skewed) {
        t.row_s_lo.assign(nrows, INT32_MAX);
        t.row_s_hi.assign(nrows, -1);
        for (const auto& lf : t.leaves) {
            const auto r  = static_cast<SizeType>(lf.row);
            t.row_s_lo[r] = std::min(t.row_s_lo[r], lf.shift);
            t.row_s_hi[r] = std::max(t.row_s_hi[r], lf.shift);
        }
        for (auto& lf : t.leaves) {
            lf.shift -= t.row_s_lo[static_cast<SizeType>(lf.row)];
        }
        for (SizeType r = 0; r < nrows; ++r) {
            if (t.row_s_lo[r] <= t.row_s_hi[r]) {
                t.ring_len[r] = t.row_s_hi[r] - t.row_s_lo[r];
            } else {
                t.row_s_lo[r] = 0;
                t.row_s_hi[r] = -1;
            }
        }
    } else {
        for (const auto& lf : t.leaves) {
            auto& len = t.ring_len[static_cast<SizeType>(lf.row)];
            len       = std::max(len, lf.shift);
        }
        // Levels k+1..root tree histories: each merge's head row over the last
        // `delay` samples, as level-k rows (rings extended where a merge
        // reaches further than any trial through it).
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
                for (auto m = static_cast<SizeType>(lo);
                     m < t.hist_leaves.size(); ++m) {
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
