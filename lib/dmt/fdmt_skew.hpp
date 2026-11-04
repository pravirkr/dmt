#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <fmt/format.h>
#include <omp.h>

#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"

#include "dmt/bit_pack_utils.hpp"
#include "dmt/fdmt_box_weights.hpp"
#include "dmt/fdmt_direct.hpp"

// Skewed-window streaming FDMT (CPU prototype, mode "valid").
//
// The tree engine streams by keeping, at every merge, the last `delay`
// samples of its head input (plan tree_history_size = sum of delays). Here
// no level above K keeps a per-merge history. Instead every row r of a level
// l >= K is computed, for an output chunk [T, T + n), over its own skewed
// time window
//
//     [T - s_hi(r), T + n - s_lo(r))
//
// where s_lo / s_hi are the smallest / largest accumulated head delay from r
// to the root over the trials that read it (compute_skew_windows). A parent
// at its own window then finds its children inside theirs, so a chunk needs
// nothing of the past except the window overlap, spread(r) = s_hi - s_lo,
// which the row keeps as a short ring: each chunk computes exactly n new
// samples per row and carries the last spread(r) over.
//
// Rows below K cannot be mixed with this: a level-K row evaluated in the
// past needs its children in the past, and the spread of a level-0 row (one
// channel at one box length, read by every trial) is about dt_max. So
// levels 0..K-1 are never materialised. A level-K row is the sum of its
// leaves (channel, box length, accumulated delay; fdmt_direct.hpp) read from
// the running sum of each channel over a raw-input ring of the last
// max_r(s_hi(r) + leaf extent) + chunk samples, which can be evaluated at
// any time inside the ring (the CUDA direct path does this for a chunk's
// own time range; here the window is the row's skewed one).
//
// Exactness. Levels 0..K are summed in integer arithmetic (uint32 running
// sums, exact box sums, the plan's box-mean recipes for packed input), and
// converted once to float at level K, so for packed input and for float
// input holding integers (a filterbank) level K equals the tree's bit for
// bit, and the merges above are the tree's own additions in the tree's own
// operand order, so the output is bit-identical to the streaming tree
// engine for any chunk length. Non-integer float input is summed in double
// running sums and agrees within float rounding (as the GPU direct path).

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
 */
inline FDMTSkewWindows compute_skew_windows(const plans::FDMTPlan& plan,
                                            SizeType level) {
    const auto& pc  = plan.get_container();
    const auto root = plan.get_niters();
    if (level == 0 || level >= root) {
        throw std::invalid_argument(fmt::format(
            "FDMT skew: level {} out of range [1, {})", level, root));
    }
    FDMTSkewWindows w;
    w.level = level;
    w.root  = root;
    w.s_lo.resize(root + 1);
    w.s_hi.resize(root + 1);
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

/// Memory of the skewed-window engine for `plan` at level K and chunk
/// length `chunk`, from the plan alone (bytes): the raw ring (`ring_elem`
/// bytes per sample), the two state halves, the per-row rings and the
/// per-thread running-sum stripes. Also the tree engine's streaming state
/// for comparison: its two chunk halves and its level-0 + tree history.
struct FDMTSkewEstimate {
    SizeType ring_nsamps{0};
    SizeType ring{0};
    SizeType state{0};
    SizeType rings{0};
    SizeType stripes{0};
    SizeType tree_state{0};
    SizeType tree_history{0};
    /// Samples recomputed per chunk per row at levels >= K if the overlap
    /// were recomputed instead of kept: sum of spreads / (rows * chunk).
    double recompute_fraction{0.0};
    [[nodiscard]] SizeType total() const noexcept {
        return ring + state + rings + stripes;
    }
    [[nodiscard]] SizeType tree_total() const noexcept {
        return tree_state + tree_history;
    }
};

inline FDMTSkewEstimate estimate_skew_memory(const plans::FDMTPlan& plan,
                                             SizeType level,
                                             SizeType chunk,
                                             SizeType ring_elem,
                                             int nthreads,
                                             bool smear = true) {
    const auto& pc  = plan.get_container();
    const auto root = plan.get_niters();
    const auto w    = compute_skew_windows(plan, level);
    std::vector<float> box_w;
    if (smear) {
        const auto weights = plan.get_box_weights();
        const float unit =
            std::ldexp(1.0F, -static_cast<int>(plan.get_box_fraction_bits()));
        for (const auto v : weights) {
            box_w.push_back(static_cast<float>(v) * unit);
        }
    }
    const auto direct = build_fdmt_direct_plan(plan, level, smear, box_w);
    FDMTSkewEstimate e;
    SizeType need_back = 0;
    SizeType max_nc    = 0;
    for (const auto& sub : direct.subs) {
        for (int32_t i = 0; i < sub.nrows; ++i) {
            const auto row = static_cast<SizeType>(sub.row0 + i);
            if (!w.used(level, row)) {
                continue;
            }
            SizeType ext = 0;
            for (auto m = direct.leaf_off[row]; m < direct.leaf_off[row + 1];
                 ++m) {
                const auto& lf = direct.leaves[static_cast<SizeType>(m)];
                ext = std::max(ext, static_cast<SizeType>(lf.shift + lf.s));
            }
            need_back = std::max(need_back, w.s_hi[level][row] + ext);
        }
        max_nc = std::max(max_nc, static_cast<SizeType>(sub.nc));
    }
    e.ring_nsamps              = need_back + chunk;
    e.ring                     = plan.get_nchans() * e.ring_nsamps * ring_elem;
    const SizeType stripe_elem = ring_elem == sizeof(float) ? 8 : 4;
    e.stripes =
        static_cast<SizeType>(std::max(1, nthreads)) *
        ((max_nc * (need_back + chunk + 1) * stripe_elem) + (chunk * 8));
    SizeType half       = 0;
    SizeType rows       = 0;
    SizeType spread_sum = 0;
    SizeType tree_rows  = 0;
    for (SizeType l = level; l < root; ++l) {
        const auto st = w.stats(l);
        half          = std::max(half, st.sum_spread + (st.used * chunk));
        e.rings += st.sum_spread * sizeof(float);
        rows += st.used;
        spread_sum += st.sum_spread;
    }
    for (SizeType l = 0; l < root; ++l) {
        tree_rows = std::max(tree_rows, pc.state_shape[l].ncoords);
    }
    e.state = 2 * half * sizeof(float);
    e.recompute_fraction =
        rows == 0
            ? 0.0
            : static_cast<double>(spread_sum) /
                  (static_cast<double>(rows) * static_cast<double>(chunk));
    e.tree_state   = 2 * tree_rows * chunk * sizeof(float);
    e.tree_history = (plan.get_history_size() + plan.get_tree_history_size()) *
                     sizeof(float);
    return e;
}

/**
 * @brief Streaming FDMT on skewed windows (see the file comment). Valid
 * mode only; float or packed (1, 2, 4, 8, 16 bit) input, one beam.
 */
class FDMTSkewCpu {
public:
    /// `nbits` 32: float input (a 4-byte ring); 1, 2, 4, 8 or 16: packed
    /// input of that width (a 1- or 2-byte ring, integer running sums).
    FDMTSkewCpu(const plans::FDMTPlan& plan,
                bool use_box_smearing,
                int nthreads,
                SizeType chunk_nsamps,
                SizeType level = 4,
                SizeType nbits = 32)
        : m_plan(plan),
          m_smear(use_box_smearing),
          m_nthreads(std::max(1, nthreads)),
          m_level(level),
          m_root(plan.get_niters()),
          m_nchans(plan.get_nchans()),
          m_nsamps(plan.get_nsamps()),
          m_chunk(std::min(chunk_nsamps == 0 ? plan.get_nsamps() : chunk_nsamps,
                           plan.get_nsamps())) {
        if (plan.get_mode() != "valid") {
            throw std::invalid_argument("FDMT skew: mode must be 'valid'");
        }
        if (m_chunk == 0) {
            throw std::invalid_argument("FDMT skew: chunk must be > 0");
        }
        if (nbits != 32) {
            if (!bit_pack_utils::detail::is_packed_nbits(
                    static_cast<unsigned>(nbits))) {
                throw std::invalid_argument(
                    "FDMT skew: nbits must be 1, 2, 4, 8, 16 or 32");
            }
            m_nbits     = nbits;
            m_ring_elem = nbits == 16 ? 2 : 1;
        }
        m_win = compute_skew_windows(plan, level);
        build_tables();
        allocate();
    }

    void execute(std::span<const float> waterfall, std::span<float> dmt) {
        check_sizes(waterfall.size(), m_nchans * m_nsamps, dmt.size());
        if (m_ring_elem != sizeof(float)) {
            throw std::invalid_argument(
                "FDMT skew: this engine was built for packed input");
        }
        run(Input{.f32 = waterfall.data()}, dmt.data());
    }

    void execute(std::span<const uint8_t> packed,
                 SizeType nbits,
                 std::span<float> dmt) {
        if (nbits != m_nbits) {
            throw std::invalid_argument(fmt::format(
                "FDMT skew: built for {}-bit input, got {}", m_nbits, nbits));
        }
        const auto row_bytes =
            bit_pack_utils::packed_row_bytes(m_nsamps, nbits);
        check_sizes(packed.size(), m_nchans * row_bytes, dmt.size());
        run(Input{.packed = packed.data(), .row_bytes = row_bytes}, dmt.data());
    }

    void reset_history() noexcept {
        std::fill(m_ring.begin(), m_ring.end(), std::byte{0});
        for (auto& r : m_rings) {
            std::fill(r.begin(), r.end(), 0.0F);
        }
        m_stream_pos = 0;
    }

    /// Bytes the engine allocates (the output is the caller's).
    struct Memory {
        SizeType ring{0};    ///< raw input ring
        SizeType stripes{0}; ///< per-thread running-sum scratch
        SizeType state{0};   ///< the two level halves
        SizeType rings{0};   ///< per-row window overlap rings (levels >= K)
        [[nodiscard]] SizeType total() const noexcept {
            return ring + stripes + state + rings;
        }
    };
    [[nodiscard]] Memory memory_usage() const noexcept { return m_mem; }
    [[nodiscard]] const FDMTSkewWindows& windows() const noexcept {
        return m_win;
    }
    [[nodiscard]] SizeType ring_nsamps() const noexcept { return m_ring_len; }
    [[nodiscard]] SizeType get_chunk_nsamps() const noexcept { return m_chunk; }

    /// Wall-clock seconds spent per phase since construction (or the last
    /// reset_profile()): writing the ring, the level-K running-sum stripes
    /// and rows, and the merges above.
    struct Profile {
        double ring{0.0};
        double stripes{0.0};
        double rows{0.0};
        double merges{0.0};
    };
    [[nodiscard]] Profile profile() const noexcept { return m_prof; }
    void reset_profile() noexcept { m_prof = {}; }

    std::string summary() const {
        std::string s = fmt::format(
            "FDMTSkewCpu: level {} of {}, chunk {}, ring {} samples x {} "
            "channels ({} B/sample), {} threads\n",
            m_level, m_root, m_chunk, m_ring_len, m_nchans, m_ring_elem,
            m_nthreads);
        for (SizeType l = m_level; l <= m_root; ++l) {
            const auto st = m_win.stats(l);
            s += fmt::format("  level {:2d}: rows {:8d} used {:8d} spread "
                             "max {:7d} mean {:9.1f}\n",
                             l, st.rows, st.used, st.max_spread,
                             st.used ? static_cast<double>(st.sum_spread) /
                                           static_cast<double>(st.used)
                                     : 0.0);
        }
        s += fmt::format("  memory: ring {:.1f} MiB, stripes {:.1f} MiB, "
                         "state {:.1f} MiB, rings {:.1f} MiB, total {:.1f} "
                         "MiB\n",
                         mib(m_mem.ring), mib(m_mem.stripes), mib(m_mem.state),
                         mib(m_mem.rings), mib(m_mem.total()));
        return s;
    }

private:
    struct Input {
        const float* f32{nullptr};
        const uint8_t* packed{nullptr};
        SizeType row_bytes{0};
    };

    /// One level-K leaf, resolved to stripe offsets relative to the row's
    /// first new output sample: box sum = P[i + a] - P[i + b].
    struct Leaf {
        int32_t c; ///< stripe channel
        int32_t s; ///< box length - 1 (recipe index)
        int64_t a;
        int64_t b;
        float w;
    };
    struct KRow {
        SizeType row;
        SizeType out;      ///< offset of the new samples in the half
        SizeType ring_off; ///< offset in m_rings[K]
        SizeType spread;
        int64_t s_lo;
        SizeType leaf_lo, leaf_hi;
    };
    struct KSub {
        int32_t c0;
        int32_t nc;
        int64_t need_back; ///< stripe origin is T - need_back
        std::vector<KRow> rows;
    };
    /// A merge (or copy) at a level above K.
    struct Op {
        SizeType out;      ///< new samples of the row in the output half, or
                           ///< the dmt row offset at the root
        SizeType tail;     ///< child offsets already shifted to the row's
        SizeType head;     ///< window (SIZE_MAX: copy)
        SizeType ring_off; ///< m_rings[l]
        SizeType spread;
    };

    static double mib(SizeType b) noexcept {
        return static_cast<double>(b) / (1024.0 * 1024.0);
    }

    void check_sizes(SizeType in, SizeType in_expected, SizeType out) const {
        if (in != in_expected) {
            throw std::invalid_argument(
                fmt::format("FDMT skew: input size {} != {}", in, in_expected));
        }
        if (out < m_plan.get_dmt_size()) {
            throw std::invalid_argument(fmt::format(
                "FDMT skew: output size {} < {}", out, m_plan.get_dmt_size()));
        }
    }

    void build_tables() {
        const auto& pc = m_plan.get_container();
        // Box-mean weights (float) and recipes (packed).
        if (m_smear) {
            const auto weights = m_plan.get_box_weights();
            const auto scale_f = m_plan.get_box_mean_bits();
            const float w_unit = std::ldexp(
                1.0F, -static_cast<int>(m_plan.get_box_fraction_bits()));
            m_int_unit = std::ldexp(1.0F, -static_cast<int>(scale_f));
            m_box_w.resize(weights.size());
            m_box_round.resize(weights.size());
            for (SizeType s = 0; s < weights.size(); ++s) {
                m_box_w[s]     = static_cast<float>(weights[s]) * w_unit;
                m_box_round[s] = box_round(s + 1, scale_f);
            }
        } else {
            m_int_unit = 1.0F;
        }
        const auto direct =
            build_fdmt_direct_plan(m_plan, m_level, m_smear, m_box_w);

        // Level K: per subband, rows with their leaves at window offsets.
        const SizeType k = m_level;
        m_row_off.assign(m_root + 1, {});
        m_level_len.assign(m_root + 1, 0);
        for (SizeType l = k; l < m_root; ++l) {
            const auto n = pc.state_shape[l].ncoords;
            m_row_off[l].assign(n, SIZE_MAX);
            SizeType off = 0;
            for (SizeType r = 0; r < n; ++r) {
                if (!m_win.used(l, r)) {
                    continue;
                }
                m_row_off[l][r] = off;
                off += m_win.spread(l, r) + m_chunk;
            }
            m_level_len[l] = off;
        }
        m_rings.assign(m_root + 1, {});
        m_ring_off.assign(m_root + 1, {});
        for (SizeType l = k; l < m_root; ++l) {
            const auto n = pc.state_shape[l].ncoords;
            m_ring_off[l].assign(n, SIZE_MAX);
            SizeType off = 0;
            for (SizeType r = 0; r < n; ++r) {
                if (!m_win.used(l, r)) {
                    continue;
                }
                m_ring_off[l][r] = off;
                off += m_win.spread(l, r);
            }
            m_rings[l].assign(off, 0.0F);
        }

        m_need_back_max = 0;
        m_max_nc        = 0;
        for (const auto& sub : direct.subs) {
            KSub ks{.c0 = sub.c0, .nc = sub.nc, .need_back = 0, .rows = {}};
            for (int32_t i = 0; i < sub.nrows; ++i) {
                const auto row = static_cast<SizeType>(sub.row0 + i);
                if (!m_win.used(k, row)) {
                    continue;
                }
                const auto s_hi = static_cast<int64_t>(m_win.s_hi[k][row]);
                const auto lo   = direct.leaf_off[row];
                const auto hi   = direct.leaf_off[row + 1];
                int64_t ext     = 0;
                for (auto m = lo; m < hi; ++m) {
                    const auto& lf = direct.leaves[static_cast<SizeType>(m)];
                    ext = std::max(ext, static_cast<int64_t>(lf.shift + lf.s));
                }
                ks.need_back = std::max(ks.need_back, s_hi + ext);
                ks.rows.push_back(
                    {.row      = row,
                     .out      = m_row_off[k][row] + m_win.spread(k, row),
                     .ring_off = m_ring_off[k][row],
                     .spread   = m_win.spread(k, row),
                     .s_lo     = static_cast<int64_t>(m_win.s_lo[k][row]),
                     .leaf_lo  = 0,
                     .leaf_hi  = 0});
            }
            // Resolve the leaves now that the subband's origin is known:
            // output sample i of row r is stream time T - s_lo + i, the
            // stripe index of time t is t - (T - need_back), and the leaf
            // box over [t - shift - s, t - shift] is P[t - shift + 1] -
            // P[t - shift - s].
            for (auto& kr : ks.rows) {
                kr.leaf_lo    = m_leaves.size();
                const auto lo = direct.leaf_off[kr.row];
                const auto hi = direct.leaf_off[kr.row + 1];
                for (auto m = lo; m < hi; ++m) {
                    const auto& lf  = direct.leaves[static_cast<SizeType>(m)];
                    const int64_t a = ks.need_back - kr.s_lo - lf.shift + 1;
                    const int64_t b = a - lf.s - 1;
                    if (b < 0) {
                        throw std::logic_error("FDMT skew: leaf before ring");
                    }
                    m_leaves.push_back(
                        {.c = lf.c, .s = lf.s, .a = a, .b = b, .w = lf.w});
                }
                kr.leaf_hi = m_leaves.size();
            }
            m_need_back_max = std::max(m_need_back_max, ks.need_back);
            m_max_nc        = std::max(m_max_nc, static_cast<SizeType>(ks.nc));
            m_subs.push_back(std::move(ks));
        }
        m_ring_len = static_cast<SizeType>(m_need_back_max) + m_chunk;

        // Levels K+1 .. root: ops in the rows' windows.
        m_ops.assign(m_root + 1, {});
        for (SizeType l = k + 1; l <= m_root; ++l) {
            const auto stride_in = pc.state_shape[l - 1].nsamps;
            for (const auto& c : pc.coordinates[l]) {
                const auto row = c.buf_offset / pc.state_shape[l].nsamps;
                if (!m_win.used(l, row)) {
                    continue;
                }
                const auto s_lo = m_win.s_lo[l][row];
                const auto t    = c.tail_buf_offset / stride_in;
                Op op{};
                op.out = l == m_root ? c.buf_offset
                                     : m_row_off[l][row] + m_win.spread(l, row);
                op.ring_off = l == m_root ? SIZE_MAX : m_ring_off[l][row];
                op.spread   = l == m_root ? 0 : m_win.spread(l, row);
                // Child windows start at T - s_hi(child); the row's new
                // samples at T - s_lo(row): offset s_hi(child) - s_lo(row)
                // (- delay for the head), non-negative by construction.
                if (m_win.s_hi[l - 1][t] < s_lo) {
                    throw std::logic_error("FDMT skew: tail window");
                }
                op.tail = m_row_off[l - 1][t] + (m_win.s_hi[l - 1][t] - s_lo);
                if (c.head_buf_offset == SIZE_MAX) {
                    op.head = SIZE_MAX;
                } else {
                    const auto h = c.head_buf_offset / stride_in;
                    if (m_win.s_hi[l - 1][h] < s_lo + c.delay) {
                        throw std::logic_error("FDMT skew: head window");
                    }
                    op.head = m_row_off[l - 1][h] +
                              (m_win.s_hi[l - 1][h] - s_lo - c.delay);
                }
                m_ops[l].push_back(op);
            }
        }
    }

    void allocate() {
        const SizeType half =
            *std::max_element(m_level_len.begin(), m_level_len.end());
        m_state.assign(2 * half, 0.0F);
        m_ring.assign(m_nchans * m_ring_len * m_ring_elem, std::byte{0});
        // Stripe: P[0 .. need_back + chunk] per channel (+1 entry), the
        // widest subband, 4 bytes (integer) or 8 (double) per entry, plus
        // one accumulator row of 8 bytes per sample.
        const SizeType stripe_len =
            static_cast<SizeType>(m_need_back_max) + m_chunk + 1;
        const SizeType stripe_elem = m_ring_elem == sizeof(float) ? 8 : 4;
        m_stripe_stride            = stripe_len;
        m_stripe_bytes = ((m_max_nc * stripe_len * stripe_elem) + 63) / 64 * 64;
        m_acc_bytes    = m_chunk * 8;
        m_scratch.assign(static_cast<SizeType>(m_nthreads) *
                             (m_stripe_bytes + m_acc_bytes),
                         std::byte{0});
        m_mem.ring    = m_ring.size();
        m_mem.stripes = m_scratch.size();
        m_mem.state   = m_state.size() * sizeof(float);
        m_mem.rings   = 0;
        for (const auto& r : m_rings) {
            m_mem.rings += r.size() * sizeof(float);
        }
        m_stream_pos = 0;
    }

    // ---- execution -------------------------------------------------------

    void run(const Input& in, float* dmt) {
        // Float input holding integers (a filterbank) takes the exact
        // integer running sums of the packed path, at half the stripe
        // bytes; anything else takes double sums.
        const bool ints = in.f32 != nullptr && integer_valued(in.f32);
        for (SizeType t0 = 0; t0 < m_nsamps; t0 += m_chunk) {
            const SizeType n = std::min(m_chunk, m_nsamps - t0);
            double t         = omp_get_wtime();
            write_ring(in, t0, n);
            m_prof.ring += omp_get_wtime() - t;
            if (m_ring_elem == sizeof(float)) {
                if (ints) {
                    level_k<float, uint32_t, false>(n);
                } else {
                    level_k<float, double, false>(n);
                }
            } else if (m_ring_elem == 2) {
                level_k<uint16_t, uint32_t, true>(n);
            } else {
                level_k<uint8_t, uint32_t, true>(n);
            }
            t = omp_get_wtime();
            for (SizeType l = m_level + 1; l <= m_root; ++l) {
                merge_level(l, n, dmt, t0);
            }
            m_prof.merges += omp_get_wtime() - t;
            m_stream_pos += n;
        }
    }

    /// True when every sample of the block is an integer small enough for
    /// every level-0 box sum to stay below 2^31 (the uint32 running-sum
    /// difference is then the exact box sum): |x| < 2^31 / L_max, L_max
    /// the widest level-0 box.
    [[nodiscard]] bool integer_valued(const float* wf) const noexcept {
        const SizeType n = m_nchans * m_nsamps;
        const auto l_max = static_cast<float>(
            m_plan.get_container().state_shape[0].dt_max + 1);
        const float bound = 2147483648.0F / l_max;
        bool ok           = true;
#pragma omp parallel for num_threads(m_nthreads) schedule(static)              \
    reduction(&& : ok)
        for (SizeType i = 0; i < n; ++i) {
            const float v = wf[i];
            ok = ok && std::fabs(v) < bound && v == std::nearbyint(v);
        }
        return ok;
    }

    [[nodiscard]] float* half(SizeType l) noexcept {
        return m_state.data() + ((l % 2) * (m_state.size() / 2));
    }

    /// Writes samples [t0, t0 + n) of every channel at stream times
    /// [m_stream_pos, m_stream_pos + n) of the ring.
    void write_ring(const Input& in, SizeType t0, SizeType n) {
        const SizeType W   = m_ring_len;
        const SizeType pos = m_stream_pos % W;
#pragma omp parallel for num_threads(m_nthreads) schedule(static)
        for (SizeType c = 0; c < m_nchans; ++c) {
            std::byte* row = m_ring.data() + (c * W * m_ring_elem);
            const auto put = [&](auto* dst) {
                using T           = std::remove_pointer_t<decltype(dst)>;
                auto* tid_scratch = reinterpret_cast<T*>(
                    m_scratch.data() +
                    (static_cast<SizeType>(omp_get_thread_num()) *
                     (m_stripe_bytes + m_acc_bytes)));
                const T* src = nullptr;
                if constexpr (std::is_same_v<T, float>) {
                    src = in.f32 + (c * m_nsamps) + t0;
                } else {
                    bit_pack_utils::unpack_row(in.packed + (c * in.row_bytes) +
                                                   ((t0 * m_nbits) / 8),
                                               m_nbits, n, tid_scratch);
                    src = tid_scratch;
                }
                const SizeType first = std::min(n, W - pos);
                std::memcpy(dst + pos, src, first * sizeof(T));
                std::memcpy(dst, src + first, (n - first) * sizeof(T));
            };
            if (m_ring_elem == sizeof(float)) {
                put(reinterpret_cast<float*>(row));
            } else if (m_ring_elem == 2) {
                put(reinterpret_cast<uint16_t*>(row));
            } else {
                put(reinterpret_cast<uint8_t*>(row));
            }
        }
    }

    /// Level K: per subband, the running sums of its channels over the
    /// stripe window, then every row over its own window.
    template <typename RingT, typename AccT, bool Packed>
    void level_k(SizeType n) {
        const SizeType W = m_ring_len;
        double t_stripes = 0.0;
        double t_rows    = 0.0;
        float* out_half  = half(m_level);
        auto* rings      = m_rings[m_level].data();
        const int64_t T  = static_cast<int64_t>(m_stream_pos);
#pragma omp parallel for num_threads(m_nthreads) schedule(dynamic, 1)          \
    reduction(+ : t_stripes, t_rows)
        for (SizeType g = 0; g < m_subs.size(); ++g) {
            const auto& sub = m_subs[g];
            if (sub.rows.empty()) {
                continue;
            }
            const double t0    = omp_get_wtime();
            std::byte* scratch = m_scratch.data() +
                                 (static_cast<SizeType>(omp_get_thread_num()) *
                                  (m_stripe_bytes + m_acc_bytes));
            auto* P            = reinterpret_cast<AccT*>(scratch);
            // Running sums over stream times [T - need_back, T + n): P[0] =
            // 0, P[j + 1] = P[j] + x[T - need_back + j]. Times before the
            // stream are zero in the ring (never written, or long gone).
            const int64_t len = sub.need_back + static_cast<int64_t>(n);
            for (int32_t ci = 0; ci < sub.nc; ++ci) {
                const auto c = static_cast<SizeType>(sub.c0 + ci);
                const auto* x =
                    reinterpret_cast<const RingT*>(m_ring.data()) + (c * W);
                AccT* p   = P + (static_cast<SizeType>(ci) * m_stripe_stride);
                p[0]      = AccT{0};
                int64_t t = T - sub.need_back;
                int64_t j = 0;
                AccT acc{0};
                while (j < len) {
                    // Contiguous ring run from slot (t mod W).
                    const SizeType slot =
                        static_cast<SizeType>(((t % static_cast<int64_t>(W)) +
                                               static_cast<int64_t>(W)) %
                                              static_cast<int64_t>(W));
                    const int64_t run = std::min<int64_t>(
                        len - j, static_cast<int64_t>(W - slot));
                    const RingT* xs = x + slot;
                    for (int64_t m = 0; m < run; ++m) {
                        acc += to_acc<AccT>(xs[m]);
                        p[j + m + 1] = acc;
                    }
                    j += run;
                    t += run;
                }
            }
            const double t1 = omp_get_wtime();
            for (const auto& kr : sub.rows) {
                float* out = out_half + kr.out;
                float* ext = out - kr.spread;
                float* rg  = rings + kr.ring_off;
                // Carry the window overlap in.
                std::copy_n(rg, kr.spread, ext);
                row_k<AccT, Packed>(P, kr, n, out, scratch + m_stripe_bytes);
                // And the new overlap out.
                std::copy_n(ext + n, kr.spread, rg);
            }
            t_stripes += t1 - t0;
            t_rows += omp_get_wtime() - t1;
        }
        m_prof.stripes += t_stripes / m_nthreads;
        m_prof.rows += t_rows / m_nthreads;
    }

    /// A ring sample as a running-sum term: integer-valued float to the
    /// wrapping unsigned sum through int32 (negative values included).
    template <typename AccT, typename RingT>
    static AccT to_acc(RingT x) noexcept {
        if constexpr (std::is_same_v<RingT, float> &&
                      std::is_same_v<AccT, uint32_t>) {
            return static_cast<uint32_t>(static_cast<int32_t>(x));
        } else {
            return static_cast<AccT>(x);
        }
    }

    /// The box sum of a leaf from its running-sum difference, as float.
    template <typename AccT> static float box_sum(AccT a, AccT b) noexcept {
        if constexpr (std::is_same_v<AccT, uint32_t>) {
            return static_cast<float>(static_cast<int32_t>(a - b));
        } else {
            return static_cast<float>(a - b);
        }
    }

    template <typename AccT, bool Packed>
    void row_k(const AccT* P,
               const KRow& kr,
               SizeType n,
               float* __restrict__ out,
               std::byte* acc_scratch) const noexcept {
        const Leaf* leaves = m_leaves.data() + kr.leaf_lo;
        const SizeType nl  = kr.leaf_hi - kr.leaf_lo;
        if constexpr (!Packed) {
            // Float input: leaf box sum (exact for integer data) times its
            // dyadic weight, summed in float (exact while representable).
            std::fill_n(out, n, 0.0F);
            for (SizeType m = 0; m < nl; ++m) {
                const Leaf& lf = leaves[m];
                const AccT* pa =
                    P + (static_cast<SizeType>(lf.c) * m_stripe_stride) +
                    static_cast<SizeType>(lf.a);
                const AccT* pb =
                    P + (static_cast<SizeType>(lf.c) * m_stripe_stride) +
                    static_cast<SizeType>(lf.b);
                const float w = lf.w;
#pragma omp simd
                for (SizeType i = 0; i < n; ++i) {
                    out[i] += box_sum<AccT>(pa[i], pb[i]) * w;
                }
            }
        } else {
            // Packed input: the plan's integer box recipes, summed exactly,
            // converted once (the tree's int -> float conversion).
            auto* acc = reinterpret_cast<uint64_t*>(acc_scratch);
            std::fill_n(acc, n, uint64_t{0});
            for (SizeType m = 0; m < nl; ++m) {
                const Leaf& lf = leaves[m];
                const AccT* pa =
                    P + (static_cast<SizeType>(lf.c) * m_stripe_stride) +
                    static_cast<SizeType>(lf.a);
                const AccT* pb =
                    P + (static_cast<SizeType>(lf.c) * m_stripe_stride) +
                    static_cast<SizeType>(lf.b);
                if (m_smear) {
                    const FDMTBoxRound r =
                        m_box_round[static_cast<SizeType>(lf.s)];
#pragma omp simd
                    for (SizeType i = 0; i < n; ++i) {
                        const auto sum = static_cast<uint64_t>(
                            static_cast<AccT>(pa[i] - pb[i]));
                        acc[i] += box_round_apply<uint64_t>(sum, r);
                    }
                } else {
#pragma omp simd
                    for (SizeType i = 0; i < n; ++i) {
                        acc[i] += static_cast<uint64_t>(
                            static_cast<AccT>(pa[i] - pb[i]));
                    }
                }
            }
            const float unit = m_int_unit;
#pragma omp simd
            for (SizeType i = 0; i < n; ++i) {
                out[i] = static_cast<float>(acc[i]) * unit;
            }
        }
    }

    void merge_level(SizeType l, SizeType n, float* dmt, SizeType t0) {
        const float* in = half(l - 1);
        float* out_half = l == m_root ? dmt + t0 : half(l);
        auto* rings     = l == m_root ? nullptr : m_rings[l].data();
        const auto& ops = m_ops[l];
#pragma omp parallel for num_threads(m_nthreads) schedule(static)
        for (SizeType k = 0; k < ops.size(); ++k) {
            const Op& op = ops[k];
            float* out   = out_half + op.out;
            if (rings != nullptr && op.spread > 0) {
                std::copy_n(rings + op.ring_off, op.spread, out - op.spread);
            }
            const float* __restrict__ tail = in + op.tail;
            if (op.head == SIZE_MAX) {
                std::copy_n(tail, n, out);
            } else {
                const float* __restrict__ head = in + op.head;
#pragma omp simd
                for (SizeType i = 0; i < n; ++i) {
                    out[i] = tail[i] + head[i];
                }
            }
            if (rings != nullptr && op.spread > 0) {
                std::copy_n(out - op.spread + n, op.spread,
                            rings + op.ring_off);
            }
        }
    }

    const plans::FDMTPlan& m_plan;
    bool m_smear;
    int m_nthreads;
    SizeType m_level;
    SizeType m_root;
    SizeType m_nchans;
    SizeType m_nsamps;
    SizeType m_chunk;
    SizeType m_nbits{32};
    SizeType m_ring_elem{sizeof(float)};
    FDMTSkewWindows m_win;

    float m_int_unit{1.0F};
    std::vector<float> m_box_w;
    std::vector<FDMTBoxRound> m_box_round;

    std::vector<KSub> m_subs;
    std::vector<Leaf> m_leaves;
    int64_t m_need_back_max{0};
    SizeType m_max_nc{0};
    SizeType m_ring_len{0};

    std::vector<std::vector<SizeType>> m_row_off; ///< per level, per row
    std::vector<SizeType> m_level_len;
    std::vector<std::vector<SizeType>> m_ring_off;
    std::vector<std::vector<float>> m_rings;
    std::vector<std::vector<Op>> m_ops;

    std::vector<float> m_state;
    std::vector<std::byte> m_ring;
    std::vector<std::byte> m_scratch;
    SizeType m_stripe_stride{0};
    SizeType m_stripe_bytes{0};
    SizeType m_acc_bytes{0};
    SizeType m_stream_pos{0};
    Memory m_mem;
    Profile m_prof;
};

} // namespace dmt::algorithms::detail
