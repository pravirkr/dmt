#pragma once

// Per-row skewed windows of a valid-mode FDMT plan (see fdmt_skew.hpp), and
// the per-row lags of a subband search on them. Shared by the CPU skewed
// engine (fdmt_skew.hpp) and the CUDA windowed top (fdmt_cuda.cu).

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include <fmt/format.h>

#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"

namespace dmt::algorithms::detail {

/// Unused row marker in FDMTSkewWindows (never read on the way to the root:
/// staggered subbands, or pruned rows).
inline constexpr SizeType kSkewUnused = std::numeric_limits<SizeType>::max();

/// Per-row skewed windows of levels `level` .. root of a valid-mode plan.
struct FDMTSkewWindows {
    SizeType level{0}; ///< K
    SizeType root{0};
    /// s_lo[l][row], s_hi[l][row] for l in [level, root] (kSkewUnused rows
    /// are never consumed).
    std::vector<std::vector<SizeType>> s_lo;
    std::vector<std::vector<SizeType>> s_hi;
    /// First searched level (root: no subband search). lag[l][row] for l in
    /// [search_from, root): the row is emitted for stream samples
    /// [T - lag, T - lag + n) at the chunk of root samples [T, T + n).
    SizeType search_from{0};
    std::vector<std::vector<SizeType>> lag;

    [[nodiscard]] bool used(SizeType l, SizeType row) const noexcept {
        return s_lo[l][row] != kSkewUnused;
    }
    [[nodiscard]] SizeType spread(SizeType l, SizeType row) const noexcept {
        return used(l, row) ? s_hi[l][row] - s_lo[l][row] : 0;
    }
    /// Rows used, largest spread and summed spread of level l.
    struct LevelStats {
        SizeType rows{0};
        SizeType used{0};
        SizeType max_spread{0};
        SizeType sum_spread{0};
        SizeType max_s_hi{0};
    };
    [[nodiscard]] LevelStats stats(SizeType l) const noexcept {
        LevelStats st;
        st.rows = s_lo[l].size();
        for (SizeType r = 0; r < st.rows; ++r) {
            if (!used(l, r)) {
                continue;
            }
            ++st.used;
            st.max_spread = std::max(st.max_spread, spread(l, r));
            st.sum_spread += spread(l, r);
            st.max_s_hi = std::max(st.max_s_hi, s_hi[l][r]);
        }
        return st;
    }
};

/**
 * @brief Top-down accumulation of the head delays from the root: every root
 * row has s_lo = s_hi = 0; a merge at level l with head delay d passes
 * [s_lo, s_hi] to its tail child and [s_lo + d, s_hi + d] to its head
 * child; a copy passes its own. The operands are the rows the merge
 * kernels read (tail_buf_offset / head_buf_offset), as fdmt_direct.hpp.
 *
 * Subband search (`search_depth` d > 0): every row of levels root - d ..
 * root - 1, tree and staggered, is also a target, emitted in its own time
 * frame with a fixed per-row lag (FDMTSkewWindows::lag): at the chunk whose
 * root samples are [T, T + n), the row's samples [T - lag, T - lag + n).
 * A row the root already reads takes lag = s_lo (its window is unchanged).
 * Any other row (staggered, or a delay the root grid skips) takes the lag
 * that grows its children's windows least, so the subband rows ride on the
 * root's windows instead of widening them: [lag, lag] for the tail and
 * [lag + d, lag + d] for the head should fall inside their existing
 * windows, and ties go to the previous row of the same subband (adjacent
 * delays have nearly equal lags). Rows the root reads at a delay of D lag
 * by about D: a subband row is complete when the band's bottom edge has
 * caught up with it, the latency of the full-band search itself.
 */
inline FDMTSkewWindows compute_skew_windows(const plans::FDMTPlan& plan,
                                            SizeType level,
                                            SizeType search_depth = 0) {
    const auto& pc  = plan.get_container();
    const auto root = plan.get_niters();
    if (level == 0 || level >= root) {
        throw std::invalid_argument(fmt::format(
            "FDMT skew: level {} out of range [1, {})", level, root));
    }
    if (search_depth > 0 && root - search_depth <= level) {
        throw std::invalid_argument(fmt::format(
            "FDMT skew: subband depth {} reaches level {} (leaf level {})",
            search_depth, root - search_depth, level));
    }
    FDMTSkewWindows w;
    w.level       = level;
    w.root        = root;
    w.search_from = root - search_depth;
    w.s_lo.resize(root + 1);
    w.s_hi.resize(root + 1);
    w.lag.resize(root + 1);
    for (SizeType l = level; l <= root; ++l) {
        w.s_lo[l].assign(pc.state_shape[l].ncoords, kSkewUnused);
        w.s_hi[l].assign(pc.state_shape[l].ncoords, 0);
    }
    // Root rows: every final trial is read at delay 0.
    for (const auto& c : pc.coordinates[root]) {
        const auto row = c.buf_offset / pc.state_shape[root].nsamps; // dmt row
        w.s_lo[root][row] = 0;
        w.s_hi[root][row] = 0;
    }
    for (SizeType l = root; l > level; --l) {
        const auto stride   = pc.state_shape[l - 1].nsamps;
        const auto out_len  = pc.state_shape[l].nsamps;
        const auto push_row = [&](const plans::FDMTCoord& c, SizeType row) {
            const auto lo   = w.s_lo[l][row];
            const auto hi   = w.s_hi[l][row];
            const auto push = [&](SizeType child, SizeType d) {
                auto& clo = w.s_lo[l - 1][child];
                auto& chi = w.s_hi[l - 1][child];
                if (clo == kSkewUnused) {
                    clo = lo + d;
                    chi = hi + d;
                } else {
                    clo = std::min(clo, lo + d);
                    chi = std::max(chi, hi + d);
                }
            };
            push(c.tail_buf_offset / stride, 0);
            if (c.head_buf_offset != SIZE_MAX) {
                push(c.head_buf_offset / stride, c.delay);
            }
        };
        for (const auto& c : pc.coordinates[l]) {
            const auto row = c.buf_offset / out_len;
            if (w.used(l, row)) {
                push_row(c, row);
            }
        }
        if (l == root || l < w.search_from) {
            continue;
        }
        // Searched level: lags of the rows read from above, then the
        // remaining rows (coordinate order: ascending delay per subband,
        // tree subbands before staggered ones), each pushed as it is placed.
        const auto nrows = pc.state_shape[l].ncoords;
        w.lag[l].assign(nrows, 0);
        std::vector<SizeType> grid_of(nrows, 0);
        for (SizeType g = 0; g < pc.grids[l].size(); ++g) {
            const auto& grid = pc.grids[l][g];
            for (SizeType r = grid.coord_offset;
                 r < grid.coord_offset + grid.ndt && r < nrows; ++r) {
                grid_of[r] = g;
            }
        }
        // Reference lag per subband: its first row read from above.
        std::vector<SizeType> ref(pc.grids[l].size(), kSkewUnused);
        for (SizeType r = 0; r < nrows; ++r) {
            if (w.used(l, r)) {
                w.lag[l][r] = w.s_lo[l][r];
                if (ref[grid_of[r]] == kSkewUnused) {
                    ref[grid_of[r]] = w.s_lo[l][r];
                }
            }
        }
        for (const auto& c : pc.coordinates[l]) {
            const auto row = c.buf_offset / out_len;
            auto& lag_ref  = ref[grid_of[row]];
            if (w.used(l, row)) {
                lag_ref = w.lag[l][row];
                continue;
            }
            // Feasible lag interval of each placed child.
            int64_t lo     = 0;
            int64_t hi     = INT64_MAX;
            int64_t gap_lo = INT64_MAX;
            int64_t gap_hi = 0;
            int nint       = 0;
            const auto add = [&](SizeType child, SizeType d) {
                if (!w.used(l - 1, child)) {
                    return;
                }
                const auto a = static_cast<int64_t>(w.s_lo[l - 1][child]) -
                               static_cast<int64_t>(d);
                const auto b = static_cast<int64_t>(w.s_hi[l - 1][child]) -
                               static_cast<int64_t>(d);
                lo           = std::max(lo, a);
                hi           = std::min(hi, b);
                gap_lo       = std::min(gap_lo, b);
                gap_hi       = std::max(gap_hi, a);
                ++nint;
            };
            add(c.tail_buf_offset / stride, 0);
            if (c.head_buf_offset != SIZE_MAX) {
                add(c.head_buf_offset / stride, c.delay);
            }
            const auto want =
                static_cast<int64_t>(lag_ref == kSkewUnused ? 0 : lag_ref);
            int64_t lag = want;
            if (nint > 0) {
                if (lo <= hi) {
                    lag = std::clamp(want, lo, hi);
                } else {
                    // Disjoint: anywhere between the two costs the same.
                    lag = std::clamp(want, std::max<int64_t>(0, gap_lo),
                                     std::max<int64_t>(0, gap_hi));
                }
            }
            lag             = std::max<int64_t>(lag, 0);
            const auto ulag = static_cast<SizeType>(lag);
            w.s_lo[l][row]  = ulag;
            w.s_hi[l][row]  = ulag;
            w.lag[l][row]   = ulag;
            lag_ref         = ulag;
            push_row(c, row);
        }
    }
    return w;
}

/**
 * @brief Local skewed windows for levels `level .. fuse` when only the `n`
 * new samples [T - s_lo(fuse, r), T + n - s_lo(fuse, r)) are computed at
 * level `fuse` (its overlap is carried in `m_ring_f`).
 */
inline FDMTSkewWindows
compute_local_skew_windows(const plans::FDMTPlan& plan,
                           const FDMTSkewWindows& global_win,
                           SizeType level,
                           SizeType fuse) {
    const auto& pc  = plan.get_container();
    const auto root = plan.get_niters();
    FDMTSkewWindows w;
    w.level = level;
    w.root  = root;
    w.s_lo.resize(root + 1);
    w.s_hi.resize(root + 1);
    for (SizeType l = level; l <= fuse; ++l) {
        w.s_lo[l].assign(pc.state_shape[l].ncoords, kSkewUnused);
        w.s_hi[l].assign(pc.state_shape[l].ncoords, 0);
    }
    for (SizeType r = 0; r < pc.state_shape[fuse].ncoords; ++r) {
        if (global_win.used(fuse, r)) {
            w.s_lo[fuse][r] = global_win.s_lo[fuse][r];
            w.s_hi[fuse][r] = global_win.s_lo[fuse][r];
        }
    }
    for (SizeType l = fuse; l > level; --l) {
        const auto stride = pc.state_shape[l - 1].nsamps;
        for (const auto& c : pc.coordinates[l]) {
            const auto row = c.buf_offset / pc.state_shape[l].nsamps;
            if (!w.used(l, row)) {
                continue;
            }
            const auto lo   = w.s_lo[l][row];
            const auto hi   = w.s_hi[l][row];
            const auto push = [&](SizeType child, SizeType d) {
                auto& clo = w.s_lo[l - 1][child];
                auto& chi = w.s_hi[l - 1][child];
                if (clo == kSkewUnused) {
                    clo = lo + d;
                    chi = hi + d;
                } else {
                    clo = std::min(clo, lo + d);
                    chi = std::max(chi, hi + d);
                }
            };
            push(c.tail_buf_offset / stride, 0);
            if (c.head_buf_offset != SIZE_MAX) {
                push(c.head_buf_offset / stride, c.delay);
            }
        }
    }
    return w;
}

} // namespace dmt::algorithms::detail
