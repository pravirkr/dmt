#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <fmt/format.h>
#include <omp.h>

#if defined(__linux__)
#include <sys/mman.h>
#endif
#if defined(__AVX__)
#include <immintrin.h>
#endif

#include <array>

#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"

#include "dmt/bit_pack_utils.hpp"
#include "dmt/fdmt_box_weights.hpp"
#include "dmt/fdmt_direct.hpp"
#include "dmt/fdmt_history.hpp"
#include "dmt/fdmt_search.hpp"
#include "dmt/fdmt_skew_windows.hpp"
#include "dmt/fdmt_tree.hpp"

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

/// Automatic group-fusion materialization level F >= level.
inline SizeType auto_skew_fuse_level(const plans::FDMTPlan& plan,
                                     const FDMTSkewWindows& w,
                                     SizeType level,
                                     SizeType chunk,
                                     int nthreads,
                                     SizeType fuse = 0) noexcept {
    const auto& pc      = plan.get_container();
    const SizeType root = plan.get_niters();
    const SizeType cap =
        std::min(root - 1, std::max(level, plan.get_max_fuse_levels()));
    if (fuse >= level && fuse <= cap) {
        return fuse;
    }
    const SizeType nt         = static_cast<SizeType>(std::max(1, nthreads));
    const SizeType min_groups = std::max<SizeType>(2 * nt, 8);
    const auto st0            = w.stats(level);
    const bool large_state    = ((st0.sum_spread + (st0.used * chunk)) *
                                 sizeof(float)) >= (SizeType{512} << 20);
    const SizeType max_f      = large_state ? SizeType{8} : SizeType{6};
    const SizeType spread_mul = large_state ? SizeType{2} : SizeType{8};
    SizeType best             = level;
    for (SizeType f = level + 1; f <= std::min(cap, max_f); ++f) {
        if (pc.state_shape[f].nchans < min_groups) {
            break;
        }
        const auto st = w.stats(f - 1);
        if (st.sum_spread * spread_mul > st.used * chunk) {
            break;
        }
        best = f;
    }
    return best;
}

/// Automatic leaf level k_leaf in [1, level] for local windows up to
/// `fuse_ring`. Steps down to k_leaf = 1 only when the local group spread fits
/// the L2 time-tile threshold (max_spread <= 128), else keeps at least level 3.
inline SizeType auto_skew_leaf_level(const plans::FDMTPlan& plan,
                                     const FDMTSkewWindows& w,
                                     SizeType level,
                                     SizeType fuse_ring,
                                     SizeType chunk) noexcept {
    if (fuse_ring <= level) {
        return level;
    }
    for (SizeType k = 1; k < level; ++k) {
        const auto lw = compute_local_skew_windows(plan, w, k, fuse_ring);
        const auto st = lw.stats(k);
        if (st.max_spread <= 128 && st.sum_spread * 2 <= st.used * chunk) {
            return k;
        }
    }
    return level;
}

/// The leaf level of an engine: `leaf` when given (capped at the fusion
/// ring level), else auto_skew_leaf_level().
inline SizeType skew_leaf_level(const plans::FDMTPlan& plan,
                                const FDMTSkewWindows& w,
                                SizeType level,
                                SizeType fuse_ring,
                                SizeType chunk,
                                SizeType leaf) noexcept {
    if (leaf != 0) {
        return std::min(leaf, fuse_ring);
    }
    return auto_skew_leaf_level(plan, w, level, fuse_ring, chunk);
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
                                             bool smear            = true,
                                             SizeType fuse         = 0,
                                             SizeType search_depth = 0,
                                             SizeType leaf         = 0) {
    const auto& pc  = plan.get_container();
    const auto root = plan.get_niters();
    const auto w    = compute_skew_windows(plan, 1, search_depth);
    const auto f_lvl =
        auto_skew_fuse_level(plan, w, level, chunk, nthreads, fuse);
    const auto f_ring = f_lvl;
    const auto k_leaf = skew_leaf_level(plan, w, level, f_ring, chunk, leaf);
    const auto lw     = compute_local_skew_windows(plan, w, k_leaf, f_ring);
    std::vector<float> box_w;
    if (smear) {
        const auto weights = plan.get_box_weights();
        const float unit =
            std::ldexp(1.0F, -static_cast<int>(plan.get_box_fraction_bits()));
        for (const auto v : weights) {
            box_w.push_back(static_cast<float>(v) * unit);
        }
    }
    const auto direct = build_fdmt_direct_plan(plan, k_leaf, smear, box_w);
    FDMTSkewEstimate e;
    SizeType need_back        = 0;
    SizeType sub_stripe_elems = 0;
    std::vector<SizeType> chan_reach(plan.get_nchans(), 0);
    constexpr int64_t kMaxStripeSpan = 16384;
    constexpr SizeType kMaxSubRows   = 1024;
    for (const auto& sub : direct.subs) {
        for (int32_t i = 0; i < sub.nrows; ++i) {
            const auto row = static_cast<SizeType>(sub.row0 + i);
            if (!lw.used(k_leaf, row)) {
                continue;
            }
            const auto s_hi = lw.s_hi[k_leaf][row];
            for (auto m = direct.leaf_off[row]; m < direct.leaf_off[row + 1];
                 ++m) {
                const auto& lf  = direct.leaves[static_cast<SizeType>(m)];
                const auto c    = static_cast<SizeType>(sub.c0 + lf.c);
                const auto need = s_hi + static_cast<SizeType>(lf.shift + lf.s);
                chan_reach[c]   = std::max(chan_reach[c], need);
                need_back       = std::max(need_back, need);
            }
        }
        const bool at_f = (f_ring == k_leaf);
        std::vector<int64_t> min_j(static_cast<SizeType>(sub.nc), INT64_MAX);
        std::vector<int64_t> max_j(static_cast<SizeType>(sub.nc), 0);
        SizeType sub_rows    = 0;
        const auto flush_est = [&]() {
            if (sub_rows == 0) {
                return;
            }
            SizeType p_off = 0;
            for (int32_t ci = 0; ci < sub.nc; ++ci) {
                const auto uci = static_cast<SizeType>(ci);
                if (min_j[uci] <= max_j[uci]) {
                    p_off += static_cast<SizeType>(max_j[uci] - min_j[uci]) + 1;
                }
            }
            sub_stripe_elems = std::max(sub_stripe_elems, p_off);
            std::fill(min_j.begin(), min_j.end(), INT64_MAX);
            std::fill(max_j.begin(), max_j.end(), int64_t{0});
            sub_rows = 0;
        };
        for (int32_t i = 0; i < sub.nrows; ++i) {
            const auto row = static_cast<SizeType>(sub.row0 + i);
            if (!lw.used(k_leaf, row)) {
                continue;
            }
            const auto s_hi   = static_cast<int64_t>(lw.s_hi[k_leaf][row]);
            const auto loc_sp = at_f ? SizeType{0} : lw.spread(k_leaf, row);
            const auto rlen   = static_cast<int64_t>(chunk + loc_sp);
            const int64_t limit =
                std::max<int64_t>(kMaxStripeSpan, rlen + 4096);
            bool split = (sub_rows >= kMaxSubRows);
            if (sub_rows > 0 && !split) {
                for (auto m = direct.leaf_off[row];
                     m < direct.leaf_off[row + 1]; ++m) {
                    const auto& lf  = direct.leaves[static_cast<SizeType>(m)];
                    const auto c    = static_cast<SizeType>(sub.c0 + lf.c);
                    const int64_t a = static_cast<int64_t>(chan_reach[c]) -
                                      s_hi - lf.shift + 1;
                    const int64_t b = a - lf.s - 1;
                    const auto ci   = static_cast<SizeType>(lf.c);
                    const int64_t nmin = std::min(min_j[ci], b);
                    const int64_t nmax = std::max(max_j[ci], a - 1 + rlen);
                    if (nmax - nmin > limit) {
                        split = true;
                        break;
                    }
                }
            }
            if (split) {
                flush_est();
            }
            for (auto m = direct.leaf_off[row]; m < direct.leaf_off[row + 1];
                 ++m) {
                const auto& lf = direct.leaves[static_cast<SizeType>(m)];
                const auto c   = static_cast<SizeType>(sub.c0 + lf.c);
                const int64_t a =
                    static_cast<int64_t>(chan_reach[c]) - s_hi - lf.shift + 1;
                const int64_t b = a - lf.s - 1;
                const auto ci   = static_cast<SizeType>(lf.c);
                min_j[ci]       = std::min(min_j[ci], b);
                max_j[ci]       = std::max(max_j[ci], a - 1 + rlen);
            }
            ++sub_rows;
        }
        flush_est();
    }
    e.ring_nsamps = need_back + chunk;
    e.ring        = 0;
    for (SizeType c = 0; c < plan.get_nchans(); ++c) {
        e.ring += (chan_reach[c] + chunk) * ring_elem;
    }
    // Tiling decision as build_tables(); stripes allocated for double sums.
    const SizeType one_sub_b =
        sub_stripe_elems * (ring_elem == sizeof(float) ? 8 : 4);
    SizeType max_loc_sp = 0;
    for (SizeType l = k_leaf; l < f_ring; ++l) {
        max_loc_sp = std::max(max_loc_sp, lw.stats(l).max_spread);
    }
    const SizeType tile =
        (f_lvl > k_leaf && f_ring == f_lvl && chunk > 512 &&
         max_loc_sp <= 128 && one_sub_b <= (SizeType{512} << 10))
            ? SizeType{512}
            : chunk;
    const SizeType stripe_subs =
        (tile < chunk) ? (SizeType{1} << (f_lvl - k_leaf)) : SizeType{1};
    SizeType grp_a_elems   = 0;
    SizeType grp_b_elems   = 0;
    const SizeType ngroups = pc.state_shape[f_lvl].nchans;
    std::vector<SizeType> need_a(ngroups, 0);
    std::vector<SizeType> need_b(ngroups, 0);
    std::vector<SizeType> out_end(ngroups, 0);
    if (f_lvl > k_leaf) {
        SizeType f_off = 0;
        for (SizeType g = 0; g < ngroups; ++g) {
            const auto [fcb, fce] = fdmt_group_coords(pc, f_lvl, g, f_lvl);
            for (SizeType r = fcb; r < fce; ++r) {
                if (w.used(f_lvl, r)) {
                    f_off += w.spread(f_lvl, r) + chunk;
                }
            }
            out_end[g] = f_off;
            for (SizeType l = k_leaf; l < f_lvl; ++l) {
                const auto [cb, ce] = fdmt_group_coords(pc, f_lvl, g, l);
                SizeType off        = 0;
                for (SizeType r = cb; r < ce; ++r) {
                    const bool used =
                        (l < f_ring) ? lw.used(l, r) : w.used(l, r);
                    if (used) {
                        const SizeType sp =
                            (l < f_ring) ? lw.spread(l, r) : w.spread(l, r);
                        off += sp + tile;
                    }
                }
                if ((l - k_leaf) % 2 == 0) {
                    need_a[g]   = std::max(need_a[g], off);
                    grp_a_elems = std::max(grp_a_elems, off);
                } else {
                    need_b[g]   = std::max(need_b[g], off);
                    grp_b_elems = std::max(grp_b_elems, off);
                }
            }
        }
    }
    const SizeType stripe_b =
        ((stripe_subs * sub_stripe_elems * sizeof(double)) + 63) / 64 * 64;
    const SizeType nt      = static_cast<SizeType>(std::max(1, nthreads));
    const SizeType grp_a_b = ((grp_a_elems * sizeof(float)) + 63) / 64 * 64;
    const SizeType grp_b_b = ((grp_b_elems * sizeof(float)) + 63) / 64 * 64;
    const SizeType group_b = grp_a_b + grp_b_b;
    SizeType half0         = 0;
    SizeType half1         = 0;
    SizeType rows          = 0;
    SizeType spread_sum    = 0;
    SizeType tree_rows     = 0;
    for (SizeType l = k_leaf; l < root; ++l) {
        const auto st = w.stats(l);
        if (l >= f_lvl) {
            const SizeType len = st.sum_spread + (st.used * chunk);
            if (l % 2 == 0) {
                half0 = std::max(half0, len);
            } else {
                half1 = std::max(half1, len);
            }
        }
        rows += st.used;
        spread_sum += (l < f_ring) ? lw.stats(l).sum_spread
                                   : (l == f_ring ? 0 : st.sum_spread);
    }
    half0                  = ((half0 + 15) / 16) * 16;
    half1                  = ((half1 + 15) / 16) * 16;
    const SizeType out_cap = (f_lvl % 2 == 0) ? half0 : half1;
    SizeType idle_cap      = ((f_lvl + 1) % 2 == 0) ? half0 : half1;
    constexpr SizeType kIdleGroupThreshold = SizeType{64} << 20;
    constexpr SizeType kIdleGroupMax       = SizeType{768} << 20;
    const bool group_in_idle =
        (f_lvl > k_leaf) && (nt * group_b > kIdleGroupThreshold) &&
        (nt * group_b <= std::max(idle_cap * sizeof(float), kIdleGroupMax));
    constexpr SizeType kSharedGroupThreshold = SizeType{256} << 20;
    const bool shared_group = (f_lvl > k_leaf) && !group_in_idle &&
                              (tile == chunk) &&
                              ((nt * group_b > kSharedGroupThreshold) ||
                               (group_b > (SizeType{64} << 20)));
    e.stripes =
        nt * (stripe_b + ((shared_group || group_in_idle) ? 0 : group_b));
    if (group_in_idle) {
        const SizeType need_idle = (nt * group_b) / sizeof(float);
        if ((f_lvl + 1) % 2 == 0) {
            half0 = std::max(half0, need_idle);
        } else {
            half1 = std::max(half1, need_idle);
        }
    } else if (shared_group) {
        for (SizeType g = 0; g < ngroups; ++g) {
            const bool fit_idle = (need_a[g] + need_b[g] <= idle_cap);
            const bool fit_b_tail =
                (need_a[g] <= idle_cap) && (out_end[g] + need_b[g] <= out_cap);
            const bool fit_a_tail =
                (need_b[g] <= idle_cap) && (out_end[g] + need_a[g] <= out_cap);
            if (!fit_idle && !fit_b_tail && !fit_a_tail) {
                idle_cap = std::max(idle_cap, need_a[g] + need_b[g]);
            }
        }
        if ((f_lvl + 1) % 2 == 0) {
            half0 = std::max(half0, idle_cap);
        } else {
            half1 = std::max(half1, idle_cap);
        }
    }
    e.rings = w.stats(f_ring).sum_spread * sizeof(float);
    for (SizeType l = 0; l < root; ++l) {
        tree_rows = std::max(tree_rows, pc.state_shape[l].ncoords);
    }
    e.state = (half0 + half1) * sizeof(float);
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

/// 2 MiB-aligned hugepage-backed array zero-filled in parallel across threads.
template <typename T> class SkewLargeArray {
public:
    SkewLargeArray() = default;
    ~SkewLargeArray() { std::free(m_data); } // NOLINT
    SkewLargeArray(const SkewLargeArray&)            = delete;
    SkewLargeArray& operator=(const SkewLargeArray&) = delete;
    SkewLargeArray(SkewLargeArray&&)                 = delete;
    SkewLargeArray& operator=(SkewLargeArray&&)      = delete;

    void assign(SizeType n, int nthreads) {
        std::free(m_data); // NOLINT
        m_data = nullptr;
        m_size = n;
        if (n == 0) {
            return;
        }
        constexpr SizeType kAlign = SizeType{2} << 20;
        const SizeType bytes =
            (((n * sizeof(T)) + kAlign - 1) / kAlign) * kAlign;
        m_data = static_cast<T*>(std::aligned_alloc(kAlign, bytes)); // NOLINT
        if (m_data == nullptr) {
            throw std::bad_alloc();
        }
#if defined(__linux__) && defined(MADV_HUGEPAGE)
        ::madvise(m_data, bytes, MADV_HUGEPAGE);
#endif
        auto* raw            = reinterpret_cast<std::byte*>(m_data);
        const SizeType pages = bytes / kAlign;
#pragma omp parallel for num_threads(nthreads) schedule(static)
        for (SizeType i = 0; i < pages; ++i) {
            std::memset(raw + (i * kAlign), 0, kAlign);
        }
    }
    void swap(SkewLargeArray& other) noexcept {
        std::swap(m_data, other.m_data);
        std::swap(m_size, other.m_size);
    }
    [[nodiscard]] T* data() noexcept { return m_data; }
    [[nodiscard]] const T* data() const noexcept { return m_data; }
    [[nodiscard]] SizeType size() const noexcept { return m_size; }
    [[nodiscard]] T* begin() noexcept { return m_data; }
    [[nodiscard]] T* end() noexcept { return m_data + m_size; }

private:
    T* m_data{nullptr};
    SizeType m_size{0};
};

template <typename StateT>
inline void skew_stream_add(const StateT* __restrict__ a,
                            const StateT* __restrict__ b,
                            StateT* __restrict__ out,
                            SizeType m) noexcept {
    if constexpr (std::is_same_v<StateT, float>) {
#if defined(__AVX512F__) || defined(__AVX__)
#if defined(__AVX512F__)
        constexpr SizeType kLanes = 16;
#else
        constexpr SizeType kLanes = 8;
#endif
        constexpr std::uintptr_t kAlign = kLanes * sizeof(float);
        SizeType i                      = 0;
        for (;
             i < m && (reinterpret_cast<std::uintptr_t>(out + i) % kAlign) != 0;
             ++i) {
            out[i] = a[i] + b[i];
        }
        for (; i + kLanes <= m; i += kLanes) {
#if defined(__AVX512F__)
            _mm512_stream_ps(out + i, _mm512_add_ps(_mm512_loadu_ps(a + i),
                                                    _mm512_loadu_ps(b + i)));
#else
            _mm256_stream_ps(out + i, _mm256_add_ps(_mm256_loadu_ps(a + i),
                                                    _mm256_loadu_ps(b + i)));
#endif
        }
        for (; i < m; ++i) {
            out[i] = a[i] + b[i];
        }
#else
#pragma omp simd
        for (SizeType i = 0; i < m; ++i) {
            out[i] = a[i] + b[i];
        }
#endif
    } else {
#if defined(__AVX512F__)
        if constexpr (std::is_same_v<StateT, uint32_t>) {
            constexpr SizeType kLanes       = 16;
            constexpr std::uintptr_t kAlign = kLanes * sizeof(uint32_t);
            SizeType i                      = 0;
            for (; i < m &&
                   (reinterpret_cast<std::uintptr_t>(out + i) % kAlign) != 0;
                 ++i) {
                out[i] = a[i] + b[i];
            }
            for (; i + kLanes <= m; i += kLanes) {
                _mm512_stream_si512(
                    reinterpret_cast<__m512i*>(out + i),
                    _mm512_add_epi32(
                        _mm512_loadu_si512(
                            reinterpret_cast<const __m512i*>(a + i)),
                        _mm512_loadu_si512(
                            reinterpret_cast<const __m512i*>(b + i))));
            }
            for (; i < m; ++i) {
                out[i] = a[i] + b[i];
            }
            return;
        }
#endif
#pragma omp simd
        for (SizeType i = 0; i < m; ++i) {
            out[i] = a[i] + b[i];
        }
    }
}

inline void skew_stream_fence() noexcept {
#if defined(__AVX__)
    _mm_sfence();
#endif
}

/**
 * @brief Streaming FDMT on skewed windows (see the file comment). Valid
 * mode only; float or packed (1, 2, 4, 8, 16 bit) input, one beam.
 */
class FDMTSkewCpu {
public:
    /// `nbits` 32: float input (a 4-byte ring); 1, 2, 4, 8 or 16: packed
    /// input of that width (a 1- or 2-byte ring, integer running sums).
    /// `level` places the fusion level F (automatic from it unless `fuse`
    /// is given); `leaf` (1..F) is the leaf level K evaluated from the raw
    /// ring, 0 for the automatic choice (auto_skew_leaf_level, <= level).
    FDMTSkewCpu(const plans::FDMTPlan& plan,
                bool use_box_smearing,
                int nthreads,
                SizeType chunk_nsamps,
                SizeType level        = 4,
                SizeType nbits        = 32,
                SizeType fuse         = 0,
                bool search_int       = false,
                bool search_u32       = true,
                SizeType search_depth = 0,
                SizeType leaf         = 0)
        : m_plan(plan),
          m_smear(use_box_smearing),
          m_nthreads(std::max(1, nthreads)),
          m_fuse(level),
          m_level(level),
          m_root(plan.get_niters()),
          m_nchans(plan.get_nchans()),
          m_nsamps(plan.get_nsamps()),
          m_chunk(std::min(chunk_nsamps == 0 ? plan.get_nsamps() : chunk_nsamps,
                           plan.get_nsamps())),
          m_search_int(nbits != 32 && search_int),
          m_search_u32(search_u32),
          m_state_elem((!m_search_int || m_search_u32) ? sizeof(float)
                                                       : sizeof(uint64_t)) {
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
        if (search_depth > plan.get_subband_depth()) {
            throw std::invalid_argument(
                fmt::format("FDMT skew: search depth {} exceeds the plan's "
                            "subband depth {}",
                            search_depth, plan.get_subband_depth()));
        }
        m_win         = compute_skew_windows(plan, 1, search_depth);
        m_search_from = m_win.search_from;
        m_fuse =
            auto_skew_fuse_level(plan, m_win, level, m_chunk, m_nthreads, fuse);
        m_fuse_ring = m_fuse;
        m_level =
            skew_leaf_level(plan, m_win, level, m_fuse_ring, m_chunk, leaf);
        m_loc_win =
            compute_local_skew_windows(plan, m_win, m_level, m_fuse_ring);
        build_tables();
        allocate();
    }

    void execute(std::span<const float> waterfall, std::span<float> dmt) {
        check_sizes(waterfall.size(), m_nchans * m_nsamps, dmt.size());
        if (m_nbits != 32) {
            throw std::invalid_argument(
                "FDMT skew: this engine was built for packed input");
        }
        run(Input{.f32 = waterfall.data(), .ring_elem = sizeof(float)},
            dmt.data());
    }

    void execute(std::span<const uint8_t> packed,
                 SizeType nbits,
                 std::span<float> dmt) {
        if (m_nbits != 32 && nbits != m_nbits) {
            throw std::invalid_argument(fmt::format(
                "FDMT skew: built for {}-bit input, got {}", m_nbits, nbits));
        }
        if (!bit_pack_utils::detail::is_packed_nbits(
                static_cast<unsigned>(nbits))) {
            throw std::invalid_argument(
                fmt::format("FDMT skew: unsupported nbits={}", nbits));
        }
        const auto row_bytes =
            bit_pack_utils::packed_row_bytes(m_nsamps, nbits);
        check_sizes(packed.size(), m_nchans * row_bytes, dmt.size());
        run(Input{.packed    = packed.data(),
                  .row_bytes = row_bytes,
                  .nbits     = nbits,
                  .ring_elem = nbits == 16 ? SizeType{2} : SizeType{1}},
            dmt.data());
    }

    void execute_search(std::span<const float> waterfall,
                        FDMTSearchSink& sink) {
        if (waterfall.size() != m_nchans * m_nsamps) {
            throw std::invalid_argument(
                fmt::format("FDMT skew: input size {} != {}", waterfall.size(),
                            m_nchans * m_nsamps));
        }
        if (m_nbits != 32) {
            throw std::invalid_argument(
                "FDMT skew: this engine was built for packed input");
        }
        run_search(Input{.f32 = waterfall.data(), .ring_elem = sizeof(float)},
                   sink);
    }

    void execute_search(std::span<const uint8_t> packed,
                        SizeType nbits,
                        FDMTSearchSink& sink) {
        if (nbits != m_nbits || nbits == 32) {
            throw std::invalid_argument(fmt::format(
                "FDMT skew: built for {}-bit input, got {}", m_nbits, nbits));
        }
        const auto row_bytes =
            bit_pack_utils::packed_row_bytes(m_nsamps, nbits);
        if (packed.size() != m_nchans * row_bytes) {
            throw std::invalid_argument(
                fmt::format("FDMT skew: input size {} != {}", packed.size(),
                            m_nchans * row_bytes));
        }
        run_search(Input{.packed    = packed.data(),
                         .row_bytes = row_bytes,
                         .nbits     = nbits,
                         .ring_elem = m_ring_elem},
                   sink);
    }

    void reset_history() noexcept {
        std::fill(m_ring.begin(), m_ring.end(), std::byte{0});
        std::fill(m_ring_f.begin(), m_ring_f.end(), 0.0F);
        m_stream_pos   = 0;
        m_nonint_until = 0;
    }

    // ---- streaming history save/load ------------------------------------
    //
    // Layout (floats): a header of kHistHeader words holding small integers
    // exactly (16-bit pieces: magic, version, layout hash, stream position,
    // samples until non-integer input has left every channel's reach, ring
    // format), then the channels' causal rings in slot order, then the
    // level-F carry ring as stored. The ring is saved in its own sample
    // width: a float ring (float input seen) as one float per sample; an
    // 8- or 16-bit ring (only packed input so far) as those bits packed
    // into exact 24-bit integers, one per float word (8-bit: three samples
    // per word, a third of the float layout). The size therefore follows
    // the input: history_state_size() is the compact size while the ring is
    // narrow and the float size once it holds float input (or before any
    // input on an engine built for float). load_history() accepts either
    // layout from an engine with the same skewed-window layout (plan,
    // chunk, leaf and fusion levels); anything else throws.

    static constexpr SizeType kHistHeader = 16;

    [[nodiscard]] SizeType history_state_size() const noexcept {
        return history_size_for(m_ring_elem);
    }

    void save_history(std::span<float> out) const {
        if (out.size() != history_state_size()) {
            throw std::invalid_argument(fmt::format(
                "FDMT::save_history: Invalid output size. Expected {}, got {}",
                history_state_size(), out.size()));
        }
        std::fill_n(out.begin(), kHistHeader, 0.0F);
        const auto put16 = [&](SizeType at, uint64_t v, SizeType words) {
            for (SizeType i = 0; i < words; ++i) {
                out[at + i] = static_cast<float>((v >> (16 * i)) & 0xFFFFU);
            }
        };
        put16(0, kHistMagic, 1);
        put16(1, kHistVersion, 1);
        put16(2, layout_hash(), 4);
        put16(6, m_stream_pos, 4);
        put16(10,
              m_nonint_until > m_stream_pos ? m_nonint_until - m_stream_pos : 0,
              4);
        put16(14, m_ring_elem, 1);
        float* ring          = out.data() + kHistHeader;
        const SizeType tot   = m_chan_ring_off[m_nchans];
        const SizeType words = hist_ring_words(m_ring_elem);
        if (m_ring.size() == 0) {
            std::fill_n(ring, words, 0.0F);
        } else if (m_ring_elem != sizeof(float)) {
            ring_dispatch(m_ring_elem, [&](auto tag) {
                using T = decltype(tag);
                pack_history(reinterpret_cast<const T*>(m_ring.data()), ring);
            });
        } else {
            const auto* src = reinterpret_cast<const float*>(m_ring.data());
#pragma omp parallel for num_threads(m_nthreads) schedule(static)
            for (SizeType i = 0; i < tot; ++i) {
                ring[i] = src[i];
            }
        }
        std::copy(m_ring_f.begin(), m_ring_f.end(), ring + words);
    }

    void load_history(std::span<const float> in) {
        const auto get16 = [&](SizeType at, SizeType words) {
            uint64_t v = 0;
            for (SizeType i = 0; i < words; ++i) {
                const float x = in[at + i];
                if (!(x >= 0.0F && x <= 65535.0F) || x != std::floor(x)) {
                    throw std::invalid_argument(
                        "FDMT::load_history: not a skewed-window history");
                }
                v |= static_cast<uint64_t>(x) << (16 * i);
            }
            return v;
        };
        if (in.size() < kHistHeader || get16(0, 1) != kHistMagic ||
            get16(1, 1) != kHistVersion) {
            throw std::invalid_argument(
                "FDMT::load_history: not a skewed-window history (it was "
                "saved by a different engine path)");
        }
        uint64_t elem = get16(14, 1);
        if (elem == 0) { // written before the format word: a float ring
            elem = sizeof(float);
        }
        if (elem != 1 && elem != 2 && elem != sizeof(float)) {
            throw std::invalid_argument(
                "FDMT::load_history: unknown skewed-window ring format");
        }
        if (in.size() != history_size_for(elem)) {
            throw std::invalid_argument(fmt::format(
                "FDMT::load_history: Invalid input size. Expected {}, got {}",
                history_size_for(elem), in.size()));
        }
        if (get16(2, 4) != layout_hash()) {
            throw std::invalid_argument(
                "FDMT::load_history: the history was saved by an engine with "
                "a different skewed-window layout (plan, chunk_nsamps or "
                "fusion depth)");
        }
        const uint64_t pos   = get16(6, 4);
        const uint64_t rem   = get16(10, 4);
        const float* ring    = in.data() + kHistHeader;
        const SizeType tot   = m_chan_ring_off[m_nchans];
        const SizeType words = hist_ring_words(elem);
        // The ring type that holds every saved sample: the saved width for
        // a packed ring; for a float ring, the narrowest type holding its
        // samples exactly (a packed stream saved by a float ring stays
        // packed). Never narrower than the current ring.
        SizeType need = elem;
        if (elem != sizeof(float)) {
            bool exact = true;
#pragma omp parallel for num_threads(m_nthreads) schedule(static)              \
    reduction(&& : exact)
            for (SizeType i = 0; i < words; ++i) {
                const float v = ring[i];
                exact         = exact && v >= 0.0F && v < kHistWordRange &&
                                v == std::floor(v);
            }
            if (!exact) {
                throw std::invalid_argument(
                    "FDMT::load_history: a packed ring word is not a 24-bit "
                    "integer (the buffer was altered)");
            }
        } else {
            float lo  = 0.0F;
            float hi  = 0.0F;
            bool ints = true;
#pragma omp parallel for num_threads(m_nthreads) schedule(static)              \
    reduction(min : lo) reduction(max : hi) reduction(&& : ints)
            for (SizeType i = 0; i < tot; ++i) {
                const float v = ring[i];
                lo            = std::min(lo, v);
                hi            = std::max(hi, v);
                ints          = ints && v == std::nearbyint(v);
            }
            if (ints && lo >= 0.0F && hi <= 255.0F) {
                need = 1;
            } else if (ints && lo >= 0.0F && hi <= 65535.0F) {
                need = 2;
            }
        }
        if (m_ring.size() == 0) { // float-built, before any input
            alloc_ring(need);
        } else if (need > m_ring_elem) {
            if (m_nbits != 32) {
                throw std::invalid_argument(
                    "FDMT::load_history: the history holds samples this "
                    "packed-input engine cannot represent");
            }
            widen_ring(need);
        }
        ring_dispatch(m_ring_elem, [&](auto tag) {
            using T   = decltype(tag);
            auto* dst = reinterpret_cast<T*>(m_ring.data());
            if (elem == 1) {
                unpack_history<uint8_t>(ring, dst);
            } else if (elem == 2) {
                unpack_history<uint16_t>(ring, dst);
            } else {
#pragma omp parallel for num_threads(m_nthreads) schedule(static)
                for (SizeType i = 0; i < tot; ++i) {
                    dst[i] = static_cast<T>(ring[i]);
                }
            }
        });
        std::copy_n(ring + words, m_ring_f.size(), m_ring_f.begin());
        m_stream_pos   = pos;
        m_nonint_until = pos + rem;
    }

    /// Bytes the engine allocates (the output is the caller's).
    struct Memory {
        SizeType ring{0};    ///< raw input ring
        SizeType stripes{0}; ///< per-thread running-sum + fused group scratch
        SizeType state{0};   ///< the two global level halves (levels >= F)
        SizeType rings{0};   ///< per-row window overlap ring at level F
        [[nodiscard]] SizeType total() const noexcept {
            return ring + stripes + state + rings;
        }
    };
    [[nodiscard]] Memory memory_usage() const noexcept { return m_mem; }
    [[nodiscard]] SizeType plan_bytes() const noexcept {
        SizeType ops = 0;
        for (const auto& v : m_ops) {
            ops += v.size() * sizeof(Op);
        }
        return ops + (m_leaves.size() * sizeof(Leaf)) +
               (m_subs.size() * sizeof(KSub));
    }
    [[nodiscard]] const FDMTSkewWindows& windows() const noexcept {
        return m_win;
    }
    [[nodiscard]] SizeType ring_nsamps() const noexcept { return m_ring_len; }
    [[nodiscard]] SizeType get_chunk_nsamps() const noexcept { return m_chunk; }
    [[nodiscard]] SizeType get_fuse_level() const noexcept { return m_fuse; }
    [[nodiscard]] SizeType get_leaf_level() const noexcept { return m_level; }
    /// Largest lag of a searched row (subband search; 0 without one).
    [[nodiscard]] uint64_t max_emit_lag() const noexcept {
        uint64_t lag = 0;
        for (const auto& level : m_emit_lag) {
            for (const auto l : level) {
                lag = std::max(lag, l);
            }
        }
        return lag;
    }

    /// Wall-clock seconds spent per phase since construction (or the last
    /// reset_profile()): writing the ring, the level-K running-sum stripes
    /// and rows (+ fused group merges), and the global merges above F.
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
            "FDMTSkewCpu: level {} (fuse {}) of {}, chunk {}, ring {} samples "
            "x {} channels ({} B/sample), {} threads\n",
            m_level, m_fuse, m_root, m_chunk, m_ring_len, m_nchans, m_ring_elem,
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
        SizeType nbits{32};
        SizeType ring_elem{sizeof(float)};
    };

    static constexpr uint32_t kOpNone = UINT32_MAX;

    /// One level-K leaf, resolved to stripe offsets relative to the row's
    /// first computed sample: box sum = P[i + a] - P[i + b].
    struct Leaf {
        int32_t a;
        int32_t b;
        int32_t s; ///< box length - 1 (recipe index)
        float w;
    };
    struct KRow {
        uint32_t row;
        uint32_t out;          ///< offset in local scratch (or half(K) if F==K)
        uint32_t ring_off;     ///< offset in m_ring_f (kOpNone if F > K)
        uint32_t spread;       ///< carried in m_ring_f (0 if F > K)
        uint32_t local_spread; ///< extra samples computed by row_k
        int32_t s_start;       ///< stream time origin T - s_start of row_k
        uint32_t leaf_lo, leaf_hi;
    };
    struct StripeSpan {
        int64_t j0;     ///< start index relative to T - m_chan_reach[c]
        SizeType len;   ///< number of ring samples to sum
        SizeType p_off; ///< destination offset in channel stripe p
    };
    struct KSub {
        int32_t c0;
        int32_t nc;
        int64_t need_back;
        std::vector<KRow> rows;
        std::vector<StripeSpan> spans;
    };
    /// A merge (or copy) at a level above K.
    struct Op {
        uint32_t out;
        uint32_t tail;
        uint32_t head;         ///< kOpNone: copy
        uint32_t ring_off;     ///< offset in m_ring_f (only at l == F)
        uint32_t spread;       ///< carried in m_ring_f (only at l == F)
        uint32_t local_spread; ///< extra samples computed by the merge
        uint32_t row{0};       ///< coordinate index (used at l == m_root)
    };
    struct FusedGroup {
        SizeType sub_lo{0};
        SizeType sub_hi{0};
        std::vector<SizeType> op_lo; ///< indexed by l in [0, m_fuse]
        std::vector<SizeType> op_hi;
    };
    struct GroupBatch {
        SizeType g_lo{0};
        SizeType g_hi{0};
        SizeType a_off{0};
        SizeType b_off{0};
        bool a_in_out_tail{false};
        bool b_in_out_tail{false};
        bool in_l3{false};
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
            m_int_unit = m_search_int
                             ? 1.0F
                             : std::ldexp(1.0F, -static_cast<int>(scale_f));
            m_box_w.resize(weights.size());
            m_box_round.resize(weights.size());
            for (SizeType s = 0; s < weights.size(); ++s) {
                m_box_w[s]     = static_cast<float>(weights[s]) * w_unit;
                m_box_round[s] = m_search_int ? FDMTBoxRound{.mul = weights[s],
                                                             .add = 0U,
                                                             .rshift = 0U,
                                                             .lshift = 0U}
                                              : box_round(s + 1, scale_f);
            }
            const SizeType box_max = pc.state_shape[0].dt_max + 1;
            for (const SizeType b : {1, 2, 4, 8, 16}) {
                const SizeType maxval = (SizeType{1} << b) - 1;
                m_prod_u32_by_nbits[b] =
                    m_search_int ? m_search_u32
                                 : box_round_fits_u32(maxval, box_max);
            }
            if (m_nbits < 32) {
                m_prod_u32 = m_prod_u32_by_nbits[m_nbits];
            }
        } else {
            m_int_unit = 1.0F;
            m_prod_u32 = !m_search_int || m_search_u32;
            m_prod_u32_by_nbits.fill(m_prod_u32);
        }
        const auto direct =
            build_fdmt_direct_plan(m_plan, m_level, m_smear, m_box_w);

        const SizeType k      = m_level;
        const SizeType f      = m_fuse;
        const SizeType f_ring = m_fuse_ring;
        m_row_off.assign(m_root + 1, {});
        m_level_len.assign(m_root + 1, 0);
        for (SizeType l = k; l < m_root; ++l) {
            m_row_off[l].assign(pc.state_shape[l].ncoords, SIZE_MAX);
        }

        // Global state offsets for levels f .. root - 1.
        for (SizeType l = f; l < m_root; ++l) {
            const auto n = pc.state_shape[l].ncoords;
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

        // Subband search: where each searched row's emitted samples start in
        // its level's window (index 0 is stream sample T - s_hi), and its lag.
        m_emit_off.assign(m_root, {});
        m_emit_lag.assign(m_root, {});
        for (SizeType l = m_search_from; l < m_root; ++l) {
            if (l <= f) {
                throw std::logic_error(
                    "FDMT skew: searched level inside the fused levels");
            }
            const auto n = pc.state_shape[l].ncoords;
            m_emit_off[l].assign(n, 0);
            m_emit_lag[l].assign(n, 0);
            for (SizeType r = 0; r < n; ++r) {
                if (!m_win.used(l, r)) {
                    throw std::logic_error("FDMT skew: unplaced search row");
                }
                const auto lag   = m_win.lag[l][r];
                m_emit_off[l][r] = m_row_off[l][r] + (m_win.s_hi[l][r] - lag);
                m_emit_lag[l][r] = lag;
            }
        }

        // Overlap ring at level f_ring only (zero rings at all other levels).
        {
            const auto n = pc.state_shape[f_ring].ncoords;
            m_ring_f_off.assign(n, SIZE_MAX);
            SizeType off = 0;
            for (SizeType r = 0; r < n; ++r) {
                if (!m_win.used(f_ring, r)) {
                    continue;
                }
                m_ring_f_off[r] = off;
                off += m_win.spread(f_ring, r);
            }
            m_ring_f.assign(off * (m_state_elem / sizeof(float)), 0.0F);
        }

        // Local scratch row offsets for levels k .. f - 1 per fused group.
        m_need_back_max    = 0;
        m_sub_stripe_elems = 0;
        m_chan_reach.assign(m_nchans, 0);
        constexpr int64_t kMaxStripeSpan = 16384;
        constexpr SizeType kMaxSubRows   = 1024;
        std::vector<SizeType> sub_ksub_off(direct.subs.size() + 1, 0);
        for (SizeType s_idx = 0; s_idx < direct.subs.size(); ++s_idx) {
            sub_ksub_off[s_idx] = m_subs.size();
            const auto& sub     = direct.subs[s_idx];
            std::vector<KRow> all_rows;
            int64_t sub_need_back = 0;
            for (int32_t i = 0; i < sub.nrows; ++i) {
                const auto row = static_cast<SizeType>(sub.row0 + i);
                if (!m_loc_win.used(k, row)) {
                    continue;
                }
                const auto s_hi = static_cast<int64_t>(m_loc_win.s_hi[k][row]);
                const auto lo   = direct.leaf_off[row];
                const auto hi   = direct.leaf_off[row + 1];
                int64_t ext     = 0;
                for (auto m = lo; m < hi; ++m) {
                    const auto& lf   = direct.leaves[static_cast<SizeType>(m)];
                    const int64_t le = static_cast<int64_t>(lf.shift + lf.s);
                    ext              = std::max(ext, le);
                    const auto c     = static_cast<SizeType>(sub.c0 + lf.c);
                    m_chan_reach[c]  = std::max(m_chan_reach[c], s_hi + le);
                }
                sub_need_back           = std::max(sub_need_back, s_hi + ext);
                const bool at_f         = (f_ring == k);
                const SizeType carry_sp = at_f ? m_win.spread(k, row) : 0;
                const SizeType loc_sp   = at_f ? 0 : m_loc_win.spread(k, row);
                all_rows.push_back(
                    {.row      = static_cast<uint32_t>(row),
                     .out      = 0,
                     .ring_off = at_f ? static_cast<uint32_t>(m_ring_f_off[row])
                                      : kOpNone,
                     .spread   = static_cast<uint32_t>(carry_sp),
                     .local_spread = static_cast<uint32_t>(loc_sp),
                     .s_start      = static_cast<int32_t>(s_hi),
                     .leaf_lo      = 0,
                     .leaf_hi      = 0});
            }
            m_need_back_max = std::max(m_need_back_max, sub_need_back);
            std::vector<int64_t> min_j(static_cast<SizeType>(sub.nc),
                                       INT64_MAX);
            std::vector<int64_t> max_j(static_cast<SizeType>(sub.nc), 0);
            SizeType r_begin      = 0;
            const auto flush_ksub = [&](SizeType r_end) {
                KSub ks{
                    .c0        = sub.c0,
                    .nc        = sub.nc,
                    .need_back = sub_need_back,
                    .rows      = std::vector<KRow>(
                        all_rows.begin() + static_cast<std::ptrdiff_t>(r_begin),
                        all_rows.begin() + static_cast<std::ptrdiff_t>(r_end)),
                    .spans =
                        std::vector<StripeSpan>(static_cast<SizeType>(sub.nc)),
                };
                SizeType p_off = 0;
                for (int32_t ci = 0; ci < sub.nc; ++ci) {
                    const auto uci = static_cast<SizeType>(ci);
                    if (min_j[uci] > max_j[uci]) {
                        ks.spans[uci] = {.j0 = 0, .len = 0, .p_off = p_off};
                        continue;
                    }
                    const auto len =
                        static_cast<SizeType>(max_j[uci] - min_j[uci]);
                    ks.spans[uci] = {
                        .j0 = min_j[uci], .len = len, .p_off = p_off};
                    p_off += len + 1;
                }
                m_sub_stripe_elems = std::max(m_sub_stripe_elems, p_off);
                for (auto& kr : ks.rows) {
                    kr.leaf_lo    = static_cast<uint32_t>(m_leaves.size());
                    const auto lo = direct.leaf_off[kr.row];
                    const auto hi = direct.leaf_off[kr.row + 1];
                    for (auto m = lo; m < hi; ++m) {
                        const auto& lf =
                            direct.leaves[static_cast<SizeType>(m)];
                        const auto c = static_cast<SizeType>(sub.c0 + lf.c);
                        const int64_t a =
                            m_chan_reach[c] - kr.s_start - lf.shift + 1;
                        const int64_t b = a - lf.s - 1;
                        const auto& sp  = ks.spans[static_cast<SizeType>(lf.c)];
                        const int64_t comp_b =
                            static_cast<int64_t>(sp.p_off) + (b - sp.j0);
                        const int64_t comp_a = comp_b + lf.s + 1;
                        m_leaves.push_back({.a = static_cast<int32_t>(comp_a),
                                            .b = static_cast<int32_t>(comp_b),
                                            .s = lf.s,
                                            .w = lf.w});
                    }
                    kr.leaf_hi = static_cast<uint32_t>(m_leaves.size());
                }
                m_subs.push_back(std::move(ks));
                std::fill(min_j.begin(), min_j.end(), INT64_MAX);
                std::fill(max_j.begin(), max_j.end(), int64_t{0});
                r_begin = r_end;
            };
            for (SizeType ri = 0; ri < all_rows.size(); ++ri) {
                const auto& kr = all_rows[ri];
                const auto lo  = direct.leaf_off[kr.row];
                const auto hi  = direct.leaf_off[kr.row + 1];
                const auto rlen =
                    static_cast<int64_t>(m_chunk + kr.local_spread);
                const int64_t limit =
                    std::max<int64_t>(kMaxStripeSpan, rlen + 4096);
                bool split = (ri - r_begin >= kMaxSubRows);
                if (ri > r_begin && !split) {
                    for (auto m = lo; m < hi; ++m) {
                        const auto& lf =
                            direct.leaves[static_cast<SizeType>(m)];
                        const auto c = static_cast<SizeType>(sub.c0 + lf.c);
                        const int64_t a =
                            m_chan_reach[c] - kr.s_start - lf.shift + 1;
                        const int64_t b = a - lf.s - 1;
                        const auto uci  = static_cast<SizeType>(lf.c);
                        if (std::max(max_j[uci], a - 1 + rlen) -
                                std::min(min_j[uci], b) >
                            limit) {
                            split = true;
                            break;
                        }
                    }
                }
                if (split) {
                    flush_ksub(ri);
                }
                for (auto m = lo; m < hi; ++m) {
                    const auto& lf = direct.leaves[static_cast<SizeType>(m)];
                    const auto c   = static_cast<SizeType>(sub.c0 + lf.c);
                    const int64_t a =
                        m_chan_reach[c] - kr.s_start - lf.shift + 1;
                    const int64_t b = a - lf.s - 1;
                    if (b < 0) {
                        throw std::logic_error("FDMT skew: leaf before ring");
                    }
                    const auto uci = static_cast<SizeType>(lf.c);
                    min_j[uci]     = std::min(min_j[uci], b);
                    max_j[uci]     = std::max(max_j[uci], a - 1 + rlen);
                }
            }
            if (r_begin < all_rows.size() || all_rows.empty()) {
                flush_ksub(all_rows.size());
            }
        }
        sub_ksub_off[direct.subs.size()] = m_subs.size();

        const SizeType stripe_elem = m_ring_elem == sizeof(float) ? 8 : 4;
        const SizeType one_sub_stripe_bytes = m_sub_stripe_elems * stripe_elem;
        SizeType max_loc_sp                 = 0;
        for (SizeType l = k; l < f_ring; ++l) {
            max_loc_sp = std::max(max_loc_sp, m_loc_win.stats(l).max_spread);
        }
        if (f > k && f_ring == f && m_chunk > 512 && max_loc_sp <= 128 &&
            one_sub_stripe_bytes <= (SizeType{512} << 10)) {
            m_tile        = 512;
            m_stripe_subs = SizeType{1} << (f - k);
        } else {
            m_tile        = m_chunk;
            m_stripe_subs = 1;
        }

        m_half0_len = 0;
        m_half1_len = 0;
        for (SizeType l = f; l < m_root; ++l) {
            if (l % 2 == 0) {
                m_half0_len = std::max(m_half0_len, m_level_len[l]);
            } else {
                m_half1_len = std::max(m_half1_len, m_level_len[l]);
            }
        }
        const SizeType root_chunk_len =
            pc.state_shape[m_root].ncoords * m_chunk;
        if (m_root % 2 == 0) {
            m_half0_len = std::max(m_half0_len, root_chunk_len);
        } else {
            m_half1_len = std::max(m_half1_len, root_chunk_len);
        }

        m_half0_len = ((m_half0_len + 15) / 16) * 16;
        m_half1_len = ((m_half1_len + 15) / 16) * 16;

        m_group_a_elems = 0;
        m_group_b_elems = 0;
        m_group_in_idle = false;
        m_group_batches.clear();
        if (f > k) {
            const SizeType ngroups = pc.state_shape[f].nchans;
            std::vector<SizeType> need_a(ngroups, 0);
            std::vector<SizeType> need_b(ngroups, 0);
            std::vector<SizeType> out_end(ngroups, 0);
            for (SizeType g = 0; g < ngroups; ++g) {
                const auto [fcb, fce] = fdmt_group_coords(pc, f, g, f);
                SizeType f_end        = (g > 0) ? out_end[g - 1] : SizeType{0};
                for (SizeType r = fcb; r < fce; ++r) {
                    if (m_win.used(f, r)) {
                        f_end = m_row_off[f][r] + m_win.spread(f, r) + m_chunk;
                    }
                }
                out_end[g] = f_end;
                for (SizeType l = k; l < f; ++l) {
                    const auto [cb, ce] = fdmt_group_coords(pc, f, g, l);
                    SizeType off        = 0;
                    for (SizeType r = cb; r < ce; ++r) {
                        const bool used = (l < f_ring) ? m_loc_win.used(l, r)
                                                       : m_win.used(l, r);
                        if (used) {
                            const SizeType sp = (l < f_ring)
                                                    ? m_loc_win.spread(l, r)
                                                    : m_win.spread(l, r);
                            off += sp + m_tile;
                        }
                    }
                    if ((l - k) % 2 == 0) {
                        need_a[g] = std::max(need_a[g], off);
                    } else {
                        need_b[g] = std::max(need_b[g], off);
                    }
                }
                m_group_a_elems = std::max(m_group_a_elems, need_a[g]);
                m_group_b_elems = std::max(m_group_b_elems, need_b[g]);
            }
            m_group_a_bytes = ((m_group_a_elems * m_state_elem) + 63) / 64 * 64;
            m_group_b_bytes = ((m_group_b_elems * m_state_elem) + 63) / 64 * 64;
            const SizeType group_bytes = m_group_a_bytes + m_group_b_bytes;
            const SizeType nt          = static_cast<SizeType>(m_nthreads);
            const SizeType out_cap = (f % 2 == 0) ? m_half0_len : m_half1_len;
            SizeType idle_cap = ((f + 1) % 2 == 0) ? m_half0_len : m_half1_len;
            constexpr SizeType kIdleGroupThreshold = SizeType{64} << 20;
            constexpr SizeType kIdleGroupMax       = SizeType{768} << 20;
            m_group_in_idle =
                (nt * group_bytes > kIdleGroupThreshold) &&
                (nt * group_bytes <=
                 std::max(idle_cap * m_state_elem, kIdleGroupMax));
            if (m_group_in_idle) {
                const SizeType need_idle = (nt * group_bytes) / m_state_elem;
                if ((f + 1) % 2 == 0) {
                    m_half0_len = std::max(m_half0_len, need_idle);
                } else {
                    m_half1_len = std::max(m_half1_len, need_idle);
                }
            }
            constexpr SizeType kSharedGroupThreshold = SizeType{256} << 20;
            m_shared_group = !m_group_in_idle && (m_tile == m_chunk) &&
                             ((nt * group_bytes > kSharedGroupThreshold) ||
                              (group_bytes > (SizeType{64} << 20)));
            std::vector<SizeType> base_a(ngroups, 0);
            std::vector<SizeType> base_b(ngroups, 0);
            if (m_shared_group) {
                for (SizeType g = 0; g < ngroups; ++g) {
                    const bool fit_idle   = (need_a[g] + need_b[g] <= idle_cap);
                    const bool fit_b_tail = (need_a[g] <= idle_cap) &&
                                            (out_end[g] + need_b[g] <= out_cap);
                    const bool fit_a_tail = (need_b[g] <= idle_cap) &&
                                            (out_end[g] + need_a[g] <= out_cap);
                    if (!fit_idle && !fit_b_tail && !fit_a_tail) {
                        idle_cap = std::max(idle_cap, need_a[g] + need_b[g]);
                    }
                }
                if ((f + 1) % 2 == 0) {
                    m_half0_len = std::max(m_half0_len, idle_cap);
                } else {
                    m_half1_len = std::max(m_half1_len, idle_cap);
                }
                constexpr SizeType kL3BatchBytes = SizeType{24} << 20;
                const SizeType l3_cap            = kL3BatchBytes / m_state_elem;
                const auto can_fit = [&](SizeType end_f, SizeType sa,
                                         SizeType sb) {
                    return (sa + sb <= idle_cap) ||
                           (sa <= idle_cap && end_f + sb <= out_cap) ||
                           (sb <= idle_cap && end_f + sa <= out_cap);
                };
                const auto push_batch = [&](SizeType lo, SizeType hi,
                                            SizeType sa, SizeType sb) {
                    GroupBatch gb{};
                    gb.g_lo              = lo;
                    gb.g_hi              = hi;
                    gb.in_l3             = (sa + sb <= l3_cap);
                    const SizeType end_f = out_end[hi - 1];
                    if (sa + sb <= idle_cap) {
                        gb.a_off = 0;
                        gb.b_off = sa;
                    } else if (sa <= idle_cap && end_f + sb <= out_cap) {
                        gb.a_off         = 0;
                        gb.b_off         = out_cap - sb;
                        gb.b_in_out_tail = true;
                    } else {
                        gb.a_off         = out_cap - sa;
                        gb.b_off         = 0;
                        gb.a_in_out_tail = true;
                    }
                    m_group_batches.push_back(gb);
                };
                SizeType g_lo  = 0;
                SizeType sum_a = 0;
                SizeType sum_b = 0;
                for (SizeType g = 0; g < ngroups; ++g) {
                    if (g > g_lo && !can_fit(out_end[g], sum_a + need_a[g],
                                             sum_b + need_b[g])) {
                        push_batch(g_lo, g, sum_a, sum_b);
                        g_lo  = g;
                        sum_a = 0;
                        sum_b = 0;
                    }
                    base_a[g] = sum_a;
                    base_b[g] = sum_b;
                    sum_a += need_a[g];
                    sum_b += need_b[g];
                }
                if (g_lo < ngroups) {
                    push_batch(g_lo, ngroups, sum_a, sum_b);
                }
            }
            for (SizeType g = 0; g < ngroups; ++g) {
                for (SizeType l = k; l < f; ++l) {
                    const auto [cb, ce] = fdmt_group_coords(pc, f, g, l);
                    SizeType off = ((l - k) % 2 == 0) ? base_a[g] : base_b[g];
                    for (SizeType r = cb; r < ce; ++r) {
                        const bool used = (l < f_ring) ? m_loc_win.used(l, r)
                                                       : m_win.used(l, r);
                        if (!used) {
                            continue;
                        }
                        m_row_off[l][r]   = off;
                        const SizeType sp = (l < f_ring)
                                                ? m_loc_win.spread(l, r)
                                                : m_win.spread(l, r);
                        off += sp + m_tile;
                    }
                }
            }
        } else {
            m_group_a_bytes = 0;
            m_group_b_bytes = 0;
            m_shared_group  = false;
        }
        for (auto& ks : m_subs) {
            for (auto& kr : ks.rows) {
                kr.out =
                    static_cast<uint32_t>(m_row_off[k][kr.row] + kr.spread);
            }
        }
        m_ring_len = static_cast<SizeType>(m_need_back_max) + m_chunk;
        m_chan_ring_len.resize(m_nchans);
        m_chan_ring_off.resize(m_nchans + 1);
        m_chan_ring_off[0] = 0;
        for (SizeType c = 0; c < m_nchans; ++c) {
            m_chan_ring_len[c] =
                static_cast<SizeType>(m_chan_reach[c]) + m_chunk;
            m_chan_ring_off[c + 1] = m_chan_ring_off[c] + m_chan_ring_len[c];
        }

        // Build fused groups for levels K .. F, and ops for K+1 .. root.
        m_ops.assign(m_root + 1, {});
        if (f == k) {
            m_groups.resize(m_subs.size());
            for (SizeType g = 0; g < m_subs.size(); ++g) {
                m_groups[g].sub_lo = g;
                m_groups[g].sub_hi = g + 1;
            }
        } else {
            const SizeType ngroups = pc.state_shape[f].nchans;
            m_groups.resize(ngroups);
            for (SizeType g = 0; g < ngroups; ++g) {
                auto& fg                = m_groups[g];
                const auto [s_lo, s_hi] = fdmt_group_subbands(pc, f, g, k);
                fg.sub_lo               = sub_ksub_off[s_lo];
                fg.sub_hi               = sub_ksub_off[s_hi];
                fg.op_lo.assign(f + 1, 0);
                fg.op_hi.assign(f + 1, 0);
                for (SizeType l = k + 1; l <= f; ++l) {
                    fg.op_lo[l]          = m_ops[l].size();
                    const auto stride_in = pc.state_shape[l - 1].nsamps;
                    const auto [cb, ce]  = fdmt_group_coords(pc, f, g, l);
                    const bool at_ring   = (l == f_ring);
                    const bool post_ring = (l > f_ring);
                    for (SizeType r = cb; r < ce; ++r) {
                        const bool used =
                            post_ring ? m_win.used(l, r) : m_loc_win.used(l, r);
                        if (!used) {
                            continue;
                        }
                        const auto& c = pc.coordinates[l][r];
                        const auto s_ref =
                            post_ring ? m_win.s_hi[l][r] : m_loc_win.s_hi[l][r];
                        const auto& prev_shi = post_ring
                                                   ? m_win.s_hi[l - 1]
                                                   : m_loc_win.s_hi[l - 1];
                        const auto t         = c.tail_buf_offset / stride_in;
                        Op op{};
                        op.spread = static_cast<uint32_t>(
                            at_ring ? m_win.spread(l, r) : 0);
                        op.local_spread = static_cast<uint32_t>(
                            post_ring ? m_win.spread(l, r)
                                      : (at_ring ? 0 : m_loc_win.spread(l, r)));
                        op.out =
                            static_cast<uint32_t>(m_row_off[l][r] + op.spread);
                        op.ring_off =
                            at_ring ? static_cast<uint32_t>(m_ring_f_off[r])
                                    : kOpNone;
                        if (prev_shi[t] < s_ref) {
                            throw std::logic_error("FDMT skew: local tail");
                        }
                        op.tail = static_cast<uint32_t>(m_row_off[l - 1][t] +
                                                        (prev_shi[t] - s_ref));
                        if (c.head_buf_offset == SIZE_MAX) {
                            op.head = kOpNone;
                        } else {
                            const auto h = c.head_buf_offset / stride_in;
                            if (prev_shi[h] < s_ref + c.delay) {
                                throw std::logic_error("FDMT skew: local head");
                            }
                            op.head = static_cast<uint32_t>(
                                m_row_off[l - 1][h] +
                                (prev_shi[h] - s_ref - c.delay));
                        }
                        m_ops[l].push_back(op);
                    }
                    fg.op_hi[l] = m_ops[l].size();
                }
            }
        }

        // Tile mode keeps one running-sum stripe per level-K subband of a
        // group; a subband split (row or span limits) adds subbands, so size
        // the stripes for the largest group rather than 2^(F-K).
        if (m_stripe_subs > 1) {
            for (const auto& fg : m_groups) {
                m_stripe_subs = std::max(m_stripe_subs, fg.sub_hi - fg.sub_lo);
            }
        }

        // Global levels F+1 .. root: compute full window n + spread(l, row),
        // no overlap rings needed.
        for (SizeType l = f + 1; l <= m_root; ++l) {
            const auto stride_in = pc.state_shape[l - 1].nsamps;
            for (const auto& c : pc.coordinates[l]) {
                const auto row = c.buf_offset / pc.state_shape[l].nsamps;
                if (!m_win.used(l, row)) {
                    continue;
                }
                const auto s_hi = m_win.s_hi[l][row];
                const auto t    = c.tail_buf_offset / stride_in;
                Op op{};
                op.out = static_cast<uint32_t>(l == m_root ? c.buf_offset
                                                           : m_row_off[l][row]);
                op.ring_off     = kOpNone;
                op.spread       = 0;
                op.local_spread = static_cast<uint32_t>(
                    l == m_root ? 0 : m_win.spread(l, row));
                op.row = static_cast<uint32_t>(row);
                if (m_win.s_hi[l - 1][t] < s_hi) {
                    throw std::logic_error("FDMT skew: tail window");
                }
                op.tail = static_cast<uint32_t>(m_row_off[l - 1][t] +
                                                (m_win.s_hi[l - 1][t] - s_hi));
                if (c.head_buf_offset == SIZE_MAX) {
                    op.head = kOpNone;
                } else {
                    const auto h = c.head_buf_offset / stride_in;
                    if (m_win.s_hi[l - 1][h] < s_hi + c.delay) {
                        throw std::logic_error("FDMT skew: head window");
                    }
                    op.head = static_cast<uint32_t>(
                        m_row_off[l - 1][h] +
                        (m_win.s_hi[l - 1][h] - s_hi - c.delay));
                }
                m_ops[l].push_back(op);
            }
        }
    }

    void allocate() {
        // The compact tables (Op, KRow, Leaf) hold 32-bit offsets.
        constexpr SizeType kU32 = UINT32_MAX;
        if (m_half0_len > kU32 || m_half1_len > kU32 ||
            m_ring_f.size() > kU32 || m_group_a_elems > kU32 ||
            m_group_b_elems > kU32 ||
            m_stripe_subs * m_sub_stripe_elems >
                static_cast<SizeType>(INT32_MAX)) {
            throw std::length_error(
                "FDMT skew: plan too large for 32-bit table offsets");
        }
        // Sized for 8-byte (double) running sums whatever the ring's type:
        // a packed ring widened to float by non-integer input switches
        // level K to double sums (level_k_for) without reallocating.
        constexpr SizeType kStripeElem = sizeof(double);
        m_stripe_bytes =
            ((m_stripe_subs * m_sub_stripe_elems * kStripeElem) + 63) / 64 * 64;
        const SizeType group_bytes = m_group_a_bytes + m_group_b_bytes;
        const SizeType nt          = static_cast<SizeType>(m_nthreads);
        if (m_shared_group || m_group_in_idle) {
            m_thread_scratch_bytes = m_stripe_bytes;
        } else {
            m_thread_scratch_bytes = m_stripe_bytes + group_bytes;
        }
        const SizeType elem_scale = m_state_elem / sizeof(float);
        m_state.assign((m_half0_len + m_half1_len) * elem_scale, m_nthreads);
        // m_ring_f is sized in build_tables() (already in float words).
        if (m_nbits != 32) {
            m_ring.assign(m_chan_ring_off[m_nchans] * m_ring_elem, m_nthreads);
        }
        m_scratch.assign(nt * m_thread_scratch_bytes, m_nthreads);
        m_mem.ring    = m_chan_ring_off[m_nchans] * m_ring_elem;
        m_mem.stripes = m_scratch.size();
        m_mem.state   = m_state.size() * sizeof(float);
        m_mem.rings   = m_ring_f.size() * sizeof(float);
        m_stream_pos  = 0;
    }

    /// Calls f(T{}) with the ring's sample type for element size `elem`.
    template <typename F> static void ring_dispatch(SizeType elem, F&& f) {
        if (elem == sizeof(float)) {
            f(float{});
        } else if (elem == 2) {
            f(uint16_t{});
        } else {
            f(uint8_t{});
        }
    }

    void alloc_ring(SizeType elem) {
        m_ring_elem = elem;
        m_ring.assign(m_chan_ring_off[m_nchans] * elem, m_nthreads);
        m_mem.ring = m_ring.size();
    }

    /// Re-types the ring to a wider sample type, keeping every sample (and
    /// so the stream history) in place.
    void widen_ring(SizeType elem) {
        const SizeType tot = m_chan_ring_off[m_nchans];
        SkewLargeArray<std::byte> wide;
        wide.assign(tot * elem, m_nthreads);
        ring_dispatch(m_ring_elem, [&](auto from_tag) {
            using From      = decltype(from_tag);
            const auto* src = reinterpret_cast<const From*>(m_ring.data());
            ring_dispatch(elem, [&](auto to_tag) {
                using To  = decltype(to_tag);
                auto* dst = reinterpret_cast<To*>(wide.data());
#pragma omp parallel for num_threads(m_nthreads) schedule(static)
                for (SizeType i = 0; i < tot; ++i) {
                    dst[i] = static_cast<To>(src[i]);
                }
            });
        });
        m_ring.swap(wide);
        m_ring_elem = elem;
        m_mem.ring  = m_ring.size();
    }

    /// The ring holds the input's samples exactly: allocated on first use
    /// with the input's sample type, widened (never reset) when a wider
    /// input type arrives; narrower input is converted on write. The stream
    /// history survives any change of input type.
    void ensure_ring(SizeType ring_elem) {
        if (m_ring.size() == 0) {
            alloc_ring(ring_elem);
        } else if (ring_elem > m_ring_elem) {
            widen_ring(ring_elem);
        }
    }

    static constexpr uint64_t kHistMagic   = kFDMTSkewHistMagic; // "SK"
    static constexpr uint64_t kHistVersion = 1;
    /// A packed history word holds an exact integer below 2^24.
    static constexpr float kHistWordRange = 16777216.0F;

    /// Saved ring words for a ring of `elem`-byte samples: one float per
    /// sample for a float ring; for 8- and 16-bit rings, groups of three
    /// samples packed into one (8-bit) or two (16-bit) 24-bit words.
    [[nodiscard]] SizeType hist_ring_words(SizeType elem) const noexcept {
        const SizeType tot = m_chan_ring_off[m_nchans];
        if (elem == sizeof(float)) {
            return tot;
        }
        return ((tot + 2) / 3) * elem;
    }
    [[nodiscard]] SizeType history_size_for(SizeType elem) const noexcept {
        return kHistHeader + hist_ring_words(elem) + m_ring_f.size();
    }

    template <typename T>
    void pack_history(const T* src, float* words) const noexcept {
        constexpr SizeType kBits = 8 * sizeof(T);
        constexpr SizeType kW    = sizeof(T); // words per group of three
        const SizeType tot       = m_chan_ring_off[m_nchans];
        const SizeType ng        = (tot + 2) / 3;
#pragma omp parallel for num_threads(m_nthreads) schedule(static)
        for (SizeType q = 0; q < ng; ++q) {
            const SizeType i0 = q * 3;
            const SizeType m  = std::min<SizeType>(3, tot - i0);
            uint64_t v        = 0;
            for (SizeType j = 0; j < m; ++j) {
                v |= static_cast<uint64_t>(src[i0 + j]) << (kBits * j);
            }
            for (SizeType k = 0; k < kW; ++k) {
                words[(q * kW) + k] =
                    static_cast<float>((v >> (24 * k)) & 0xFFFFFFU);
            }
        }
    }

    /// Unpacks `S`-wide saved samples into a ring of type T (S or wider).
    template <typename S, typename T>
    void unpack_history(const float* words, T* dst) const noexcept {
        constexpr SizeType kBits = 8 * sizeof(S);
        constexpr SizeType kW    = sizeof(S);
        constexpr uint64_t kMask = (uint64_t{1} << kBits) - 1;
        const SizeType tot       = m_chan_ring_off[m_nchans];
        const SizeType ng        = (tot + 2) / 3;
#pragma omp parallel for num_threads(m_nthreads) schedule(static)
        for (SizeType q = 0; q < ng; ++q) {
            uint64_t v = 0;
            for (SizeType k = 0; k < kW; ++k) {
                v |= static_cast<uint64_t>(words[(q * kW) + k]) << (24 * k);
            }
            const SizeType i0 = q * 3;
            const SizeType m  = std::min<SizeType>(3, tot - i0);
            for (SizeType j = 0; j < m; ++j) {
                dst[i0 + j] = static_cast<T>((v >> (kBits * j)) & kMask);
            }
        }
    }

    /// Identifies the saved-history layout: everything that sizes or
    /// places the causal rings and the level-F carry ring.
    [[nodiscard]] uint64_t layout_hash() const noexcept {
        uint64_t h     = 1469598103934665603ULL;
        const auto mix = [&](uint64_t v) {
            for (int i = 0; i < 8; ++i) {
                h = (h ^ ((v >> (8 * i)) & 0xFFU)) * 1099511628211ULL;
            }
        };
        mix(m_nchans);
        mix(m_level);
        mix(m_fuse);
        mix(m_fuse_ring);
        mix(m_chunk);
        mix(m_search_from);
        mix(m_smear ? 1 : 0);
        mix(m_state_elem);
        mix(m_ring_f.size());
        for (SizeType c = 0; c < m_nchans; ++c) {
            mix(m_chan_ring_len[c]);
        }
        for (const auto off : m_ring_f_off) {
            mix(off);
        }
        return h;
    }

    /// Samples back from the current position that any channel reads.
    [[nodiscard]] SizeType max_reach() const noexcept {
        int64_t r = 0;
        for (const auto v : m_chan_reach) {
            r = std::max(r, v);
        }
        return static_cast<SizeType>(r);
    }

    /// Float input takes the exact integer running sums only when the block
    /// and every ring sample a row can still read are integers: non-integer
    /// input stays within reach for max_reach() samples after it ends.
    struct BlockKind {
        bool int_block{false}; ///< float block of exact small integers
        bool int_sums{false};  ///< ... and no non-integer history in reach
    };
    [[nodiscard]] BlockKind block_kind(const Input& in) const noexcept {
        BlockKind k;
        if (in.f32 != nullptr) {
            k.int_block = integer_valued(in.f32);
            k.int_sums  = k.int_block && m_stream_pos >= m_nonint_until;
        }
        return k;
    }
    void end_block(const Input& in, const BlockKind& k) noexcept {
        if (in.f32 != nullptr && !k.int_block) {
            m_nonint_until = m_stream_pos + max_reach();
        }
    }

    // ---- execution -------------------------------------------------------

    void run(const Input& in, float* dmt) {
        ensure_ring(in.ring_elem);
        if (in.packed != nullptr && m_smear) {
            m_prod_u32 = m_prod_u32_by_nbits[in.nbits];
        }
        // Float input holding integers (a filterbank) takes the exact
        // integer running sums of the packed path, at half the stripe
        // bytes; anything else takes double sums.
        const BlockKind kind = block_kind(in);
        for (SizeType t0 = 0; t0 < m_nsamps; t0 += m_chunk) {
            const SizeType n = std::min(m_chunk, m_nsamps - t0);
            double t         = omp_get_wtime();
            write_ring(in, t0, n);
            m_prof.ring += omp_get_wtime() - t;
            level_k_for<float>(in, kind, n);
            t = omp_get_wtime();
            for (SizeType l = m_fuse + 1; l <= m_root; ++l) {
                merge_level<float>(l, n, dmt, t0);
            }
            m_prof.merges += omp_get_wtime() - t;
            m_stream_pos += n;
        }
        end_block(in, kind);
    }

    /// Level-K evaluation for the ring's sample type and this block's input:
    /// packed input takes the integer running sums and the rounded packed
    /// box means whatever the ring type; float input (always a float ring)
    /// takes integer sums when exact (BlockKind), else double sums.
    template <typename StateT>
    void level_k_for(const Input& in, const BlockKind& kind, SizeType n) {
        if (m_ring_elem == sizeof(float)) {
            if constexpr (std::is_same_v<StateT, float>) {
                if (in.packed != nullptr) {
                    level_k<float, uint32_t, true, float>(n);
                } else if (kind.int_sums) {
                    level_k<float, uint32_t, false, float>(n);
                } else {
                    level_k<float, double, false, float>(n);
                }
            } else {
                throw std::logic_error(
                    "FDMT skew: integer search needs a packed ring");
            }
        } else if (m_ring_elem == 2) {
            level_k<uint16_t, uint32_t, true, StateT>(n);
        } else {
            level_k<uint8_t, uint32_t, true, StateT>(n);
        }
    }

    void run_search(const Input& in, FDMTSearchSink& sink) {
        ensure_ring(in.ring_elem);
        if (in.packed != nullptr && m_smear) {
            m_prod_u32 = m_prod_u32_by_nbits[in.nbits];
        }
        const BlockKind kind   = block_kind(in);
        const SizeType ntrials = m_plan.get_dmt_ndms();
        for (SizeType t0 = 0; t0 < m_nsamps; t0 += m_chunk) {
            const SizeType n = std::min(m_chunk, m_nsamps - t0);
            double t         = omp_get_wtime();
            write_ring(in, t0, n);
            m_prof.ring += omp_get_wtime() - t;
            if (!m_search_int) {
                level_k_for<float>(in, kind, n);
                t = omp_get_wtime();
                for (SizeType l = m_fuse + 1; l <= m_root; ++l) {
                    merge_level<float>(l, n, nullptr, 0);
                    if (l >= m_search_from && l < m_root) {
                        consume_level(sink, l, n, {.state = half<float>(l)});
                    }
                }
                m_prof.merges += omp_get_wtime() - t;
                sink.consume(m_root, {.state             = half<float>(m_root),
                                      .row_stride        = m_chunk,
                                      .beam_stride       = ntrials * m_chunk,
                                      .nsamps            = n,
                                      .sample_begin      = m_stream_pos,
                                      .materialized_root = true});
            } else if (m_search_u32) {
                level_k_for<uint32_t>(in, kind, n);
                t = omp_get_wtime();
                for (SizeType l = m_fuse + 1; l <= m_root; ++l) {
                    merge_level<uint32_t>(l, n, nullptr, 0);
                    if (l >= m_search_from && l < m_root) {
                        consume_level(sink, l, n,
                                      {.integer32_state = half<uint32_t>(l)});
                    }
                }
                m_prof.merges += omp_get_wtime() - t;
                sink.consume(m_root,
                             {.row_stride        = m_chunk,
                              .beam_stride       = ntrials * m_chunk,
                              .nsamps            = n,
                              .sample_begin      = m_stream_pos,
                              .materialized_root = true,
                              .integer32_state   = half<uint32_t>(m_root)});
            } else {
                level_k_for<uint64_t>(in, kind, n);
                t = omp_get_wtime();
                for (SizeType l = m_fuse + 1; l <= m_root; ++l) {
                    merge_level<uint64_t>(l, n, nullptr, 0);
                    if (l >= m_search_from && l < m_root) {
                        consume_level(sink, l, n,
                                      {.integer_state = half<uint64_t>(l)});
                    }
                }
                m_prof.merges += omp_get_wtime() - t;
                sink.consume(m_root, {.row_stride        = m_chunk,
                                      .beam_stride       = ntrials * m_chunk,
                                      .nsamps            = n,
                                      .sample_begin      = m_stream_pos,
                                      .materialized_root = true,
                                      .integer_state = half<uint64_t>(m_root)});
            }
            m_stream_pos += n;
        }
        end_block(in, kind);
    }

    /// Hands searched level l (lagged rows, see compute_skew_windows) to the
    /// sink. The merge timing of the levels above it counts as merges.
    void consume_level(FDMTSearchSink& sink,
                       SizeType l,
                       SizeType n,
                       FDMTSearchInput in) {
        in.row_stride        = m_chunk;
        in.nsamps            = n;
        in.sample_begin      = m_stream_pos;
        in.materialized_root = true;
        in.row_offset        = m_emit_off[l].data();
        in.row_lag           = m_emit_lag[l].data();
        sink.consume(l, in);
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

    template <typename StateT = float>
    [[nodiscard]] StateT* half(SizeType l) noexcept {
        return reinterpret_cast<StateT*>(m_state.data()) +
               ((l % 2 == 0) ? 0 : m_half0_len);
    }

    /// Writes samples [t0, t0 + n) of every channel at stream times
    /// [m_stream_pos, m_stream_pos + n) of its causal ring.
    /// The input is converted to the ring's sample type, which is never
    /// narrower than the input's (ensure_ring()).
    void write_ring(const Input& in, SizeType t0, SizeType n) {
        ring_dispatch(m_ring_elem, [&](auto tag) {
            using T = decltype(tag);
            if (in.f32 == nullptr && n * sizeof(T) > m_thread_scratch_bytes) {
                throw std::logic_error("FDMT skew: unpack scratch too small");
            }
            auto* ring = reinterpret_cast<T*>(m_ring.data());
#pragma omp parallel for num_threads(m_nthreads) schedule(static)
            for (SizeType c = 0; c < m_nchans; ++c) {
                const SizeType W   = m_chan_ring_len[c];
                const SizeType pos = m_stream_pos % W;
                T* dst             = ring + m_chan_ring_off[c];
                const T* src       = nullptr;
                if (in.f32 != nullptr) {
                    if constexpr (std::is_same_v<T, float>) {
                        src = in.f32 + (c * m_nsamps) + t0;
                    }
                } else {
                    auto* tid_scratch = reinterpret_cast<T*>(
                        m_scratch.data() +
                        (static_cast<SizeType>(omp_get_thread_num()) *
                         m_thread_scratch_bytes));
                    bit_pack_utils::unpack_row(in.packed + (c * in.row_bytes) +
                                                   ((t0 * in.nbits) / 8),
                                               in.nbits, n, tid_scratch);
                    src = tid_scratch;
                }
                const SizeType first = std::min(n, W - pos);
                std::memcpy(dst + pos, src, first * sizeof(T));
                std::memcpy(dst, src + first, (n - first) * sizeof(T));
            }
        });
    }

    /// Fused group evaluation (levels K .. F): per group, evaluate each
    /// level-K subband from its channel running sums, then merge K+1 .. F
    /// inside group scratch, writing only level F to global state.
    template <typename RingT,
              typename AccT,
              bool Packed,
              typename StateT = float>
    void level_k(SizeType n) {
        double t_stripes = 0.0;
        double t_rows    = 0.0;
        StateT* out_f    = half<StateT>(m_fuse);
        StateT* rings_f  = reinterpret_cast<StateT*>(m_ring_f.data());
        const int64_t T  = static_cast<int64_t>(m_stream_pos);
        const bool fused = (m_fuse > m_level);

        const auto build_stripes = [&](SizeType s, AccT* P, double& ts_acc) {
            const auto& sub = m_subs[s];
            if (sub.rows.empty()) {
                return;
            }
            const double t0 = omp_get_wtime();
            for (int32_t ci = 0; ci < sub.nc; ++ci) {
                const auto c          = static_cast<SizeType>(sub.c0 + ci);
                const int64_t rch     = m_chan_reach[c];
                const int64_t j_limit = rch + static_cast<int64_t>(n);
                const SizeType W      = m_chan_ring_len[c];
                const auto* x  = reinterpret_cast<const RingT*>(m_ring.data()) +
                                 m_chan_ring_off[c];
                const auto& sp = sub.spans[static_cast<SizeType>(ci)];
                if (sp.len == 0 || sp.j0 >= j_limit) {
                    continue;
                }
                const int64_t len = std::min<int64_t>(
                    static_cast<int64_t>(sp.len), j_limit - sp.j0);
                AccT* ps  = P + sp.p_off;
                ps[0]     = AccT{0};
                int64_t t = T - rch + sp.j0;
                int64_t j = 0;
                AccT acc{0};
                while (j < len) {
                    const SizeType slot =
                        static_cast<SizeType>(((t % static_cast<int64_t>(W)) +
                                               static_cast<int64_t>(W)) %
                                              static_cast<int64_t>(W));
                    const int64_t run = std::min<int64_t>(
                        len - j, static_cast<int64_t>(W - slot));
                    const RingT* xs = x + slot;
                    AccT* dst       = ps + j + 1;
                    int64_t m       = 0;
#if defined(__AVX512F__)
                    if constexpr (std::is_same_v<AccT, uint32_t>) {
                        const __m512i zero  = _mm512_setzero_si512();
                        const __m512i bcast = _mm512_set1_epi32(15);
                        __m512i carry =
                            _mm512_set1_epi32(static_cast<int32_t>(acc));
                        for (; m + 16 <= run; m += 16) {
                            __m512i v;
                            if constexpr (std::is_same_v<RingT, uint8_t>) {
                                v = _mm512_cvtepu8_epi32(_mm_loadu_si128(
                                    reinterpret_cast<const __m128i*>(xs + m)));
                            } else if constexpr (std::is_same_v<RingT,
                                                                uint16_t>) {
                                v = _mm512_cvtepu16_epi32(_mm256_loadu_si256(
                                    reinterpret_cast<const __m256i*>(xs + m)));
                            } else {
                                v = _mm512_cvttps_epi32(
                                    _mm512_loadu_ps(xs + m));
                            }
                            v = _mm512_add_epi32(
                                v, _mm512_alignr_epi32(v, zero, 15));
                            v = _mm512_add_epi32(
                                v, _mm512_alignr_epi32(v, zero, 14));
                            v = _mm512_add_epi32(
                                v, _mm512_alignr_epi32(v, zero, 12));
                            v = _mm512_add_epi32(
                                v, _mm512_alignr_epi32(v, zero, 8));
                            v = _mm512_add_epi32(v, carry);
                            _mm512_storeu_si512(
                                reinterpret_cast<__m512i*>(dst + m), v);
                            carry = _mm512_permutexvar_epi32(bcast, v);
                        }
                        acc = static_cast<uint32_t>(
                            _mm_cvtsi128_si32(_mm512_castsi512_si128(carry)));
                    } else if constexpr (std::is_same_v<AccT, double> &&
                                         std::is_same_v<RingT, float>) {
                        const __m512i zero  = _mm512_setzero_si512();
                        const __m512i bcast = _mm512_set1_epi64(7);
                        __m512d carry       = _mm512_set1_pd(acc);
                        for (; m + 8 <= run; m += 8) {
                            __m512d v =
                                _mm512_cvtps_pd(_mm256_loadu_ps(xs + m));
                            v = _mm512_add_pd(
                                v, _mm512_castsi512_pd(_mm512_alignr_epi64(
                                       _mm512_castpd_si512(v), zero, 7)));
                            v = _mm512_add_pd(
                                v, _mm512_castsi512_pd(_mm512_alignr_epi64(
                                       _mm512_castpd_si512(v), zero, 6)));
                            v = _mm512_add_pd(
                                v, _mm512_castsi512_pd(_mm512_alignr_epi64(
                                       _mm512_castpd_si512(v), zero, 4)));
                            v = _mm512_add_pd(v, carry);
                            _mm512_storeu_pd(dst + m, v);
                            carry = _mm512_permutexvar_pd(bcast, v);
                        }
                        acc = _mm512_cvtsd_f64(carry);
                    }
#endif
                    for (; m < run; ++m) {
                        acc += to_acc<AccT>(xs[m]);
                        dst[m] = acc;
                    }
                    j += run;
                    t += run;
                }
            }
            ts_acc += omp_get_wtime() - t0;
        };

        const auto eval_op_tile = [&](const Op& op, const StateT* in,
                                      StateT* out, SizeType m) {
            const StateT* __restrict__ tail = in + op.tail;
            if (op.head == kOpNone) {
                std::copy_n(tail, m, out);
            } else {
                const StateT* __restrict__ head = in + op.head;
#pragma omp simd
                for (SizeType i = 0; i < m; ++i) {
                    out[i] = tail[i] + head[i];
                }
            }
        };

        if (m_shared_group) {
            StateT* idle_half = half<StateT>(m_fuse + 1);
#pragma omp parallel num_threads(m_nthreads) reduction(+ : t_stripes, t_rows)
            {
                auto* P = reinterpret_cast<AccT*>(
                    m_scratch.data() +
                    (static_cast<SizeType>(omp_get_thread_num()) *
                     m_thread_scratch_bytes));
                for (const auto& batch : m_group_batches) {
                    StateT* grp_a         = batch.a_in_out_tail
                                                ? (out_f + batch.a_off)
                                                : (idle_half + batch.a_off);
                    StateT* grp_b         = batch.b_in_out_tail
                                                ? (out_f + batch.b_off)
                                                : (idle_half + batch.b_off);
                    const SizeType sub_lo = m_groups[batch.g_lo].sub_lo;
                    const SizeType sub_hi = m_groups[batch.g_hi - 1].sub_hi;
#pragma omp for schedule(dynamic, 1)
                    for (SizeType s = sub_lo; s < sub_hi; ++s) {
                        build_stripes(s, P, t_stripes);
                        const double t1 = omp_get_wtime();
                        for (const auto& kr : m_subs[s].rows) {
                            row_k<AccT, Packed, StateT>(
                                P, kr, n + kr.local_spread, grp_a + kr.out);
                        }
                        t_rows += omp_get_wtime() - t1;
                    }
                    for (SizeType l = m_level + 1; l <= m_fuse; ++l) {
                        const double t1 = omp_get_wtime();
                        const StateT* in =
                            ((l - 1 - m_level) % 2 == 0) ? grp_a : grp_b;
                        StateT* dst_half = (l == m_fuse)              ? out_f
                                           : ((l - m_level) % 2 == 0) ? grp_a
                                                                      : grp_b;
                        const bool at_f  = (l == m_fuse);
                        const bool stream =
                            !at_f && !batch.in_l3 && (l != m_fuse_ring);
                        const auto* ops   = m_ops[l].data();
                        const SizeType lo = m_groups[batch.g_lo].op_lo[l];
                        const SizeType hi = m_groups[batch.g_hi - 1].op_hi[l];
#pragma omp for schedule(static) nowait
                        for (SizeType k = lo; k < hi; ++k) {
                            const Op& op = ops[k];
                            StateT* out  = dst_half + op.out;
                            if (op.spread > 0) {
                                std::copy_n(rings_f + op.ring_off, op.spread,
                                            out - op.spread);
                            }
                            const SizeType m = n + op.local_spread;
                            const StateT* __restrict__ tail = in + op.tail;
                            if (op.head == kOpNone) {
                                std::copy_n(tail, m, out);
                            } else if (stream) {
                                skew_stream_add(tail, in + op.head, out, m);
                            } else {
                                eval_op_tile(op, in, out, m);
                            }
                            if (op.spread > 0) {
                                std::copy_n(out - op.spread + n, op.spread,
                                            rings_f + op.ring_off);
                            }
                        }
                        // Each thread fences its own non-temporal stores
                        // before the barrier that publishes them.
                        if (stream) {
                            skew_stream_fence();
                        }
#pragma omp barrier
                        t_rows += omp_get_wtime() - t1;
                    }
                }
            }
        } else {
            const bool ring_at_f = (m_fuse_ring == m_fuse);
#pragma omp parallel for num_threads(m_nthreads) schedule(dynamic, 1)          \
    reduction(+ : t_stripes, t_rows)
            for (SizeType g = 0; g < m_groups.size(); ++g) {
                const auto& fg = m_groups[g];
                const auto tid = static_cast<SizeType>(omp_get_thread_num());
                std::byte* scratch =
                    m_scratch.data() + (tid * m_thread_scratch_bytes);
                auto* P = reinterpret_cast<AccT*>(scratch);
                std::byte* grp_base =
                    m_group_in_idle
                        ? (reinterpret_cast<std::byte*>(
                               half<StateT>(m_fuse + 1)) +
                           (tid * (m_group_a_bytes + m_group_b_bytes)))
                        : (scratch + m_stripe_bytes);
                auto* grp_a = reinterpret_cast<StateT*>(grp_base);
                auto* grp_b =
                    reinterpret_cast<StateT*>(grp_base + m_group_a_bytes);

                // Pre-copy level-F carry-in before the tile loop.
                if (fused && ring_at_f) {
                    const auto* ops_f = m_ops[m_fuse].data();
                    for (SizeType k = fg.op_lo[m_fuse]; k < fg.op_hi[m_fuse];
                         ++k) {
                        const Op& op = ops_f[k];
                        if (op.spread > 0) {
                            std::copy_n(rings_f + op.ring_off, op.spread,
                                        out_f + op.out - op.spread);
                        }
                    }
                } else if (!fused) {
                    for (SizeType s = fg.sub_lo; s < fg.sub_hi; ++s) {
                        for (const auto& kr : m_subs[s].rows) {
                            if (kr.spread > 0) {
                                std::copy_n(rings_f + kr.ring_off, kr.spread,
                                            out_f + kr.out - kr.spread);
                            }
                        }
                    }
                }

                for (SizeType ts = 0; ts < n; ts += m_tile) {
                    const SizeType tn = std::min(m_tile, n - ts);
                    for (SizeType s = fg.sub_lo; s < fg.sub_hi; ++s) {
                        AccT* P_sub =
                            (m_stripe_subs > 1)
                                ? (P + ((s - fg.sub_lo) * m_sub_stripe_elems))
                                : P;
                        if (ts == 0) {
                            build_stripes(s, P_sub, t_stripes);
                        }
                        const double t1 = omp_get_wtime();
                        for (const auto& kr : m_subs[s].rows) {
                            StateT* dst = fused ? (grp_a + kr.out)
                                                : (out_f + kr.out + ts);
                            row_k<AccT, Packed, StateT>(
                                P_sub + ts, kr, tn + kr.local_spread, dst);
                        }
                        t_rows += omp_get_wtime() - t1;
                    }
                    if (fused) {
                        const double t1 = omp_get_wtime();
                        for (SizeType l = m_level + 1; l <= m_fuse; ++l) {
                            const StateT* in =
                                ((l - 1 - m_level) % 2 == 0) ? grp_a : grp_b;
                            const bool at_f = (l == m_fuse);
                            StateT* dst_half =
                                at_f ? out_f
                                     : (((l - m_level) % 2 == 0) ? grp_a
                                                                 : grp_b);
                            const auto* ops   = m_ops[l].data();
                            const SizeType hi = fg.op_hi[l];
                            for (SizeType k = fg.op_lo[l]; k < hi; ++k) {
                                const Op& op = ops[k];
                                StateT* out  = at_f ? (dst_half + op.out + ts)
                                                    : (dst_half + op.out);
                                if (!ring_at_f && op.spread > 0) {
                                    std::copy_n(rings_f + op.ring_off,
                                                op.spread, out - op.spread);
                                }
                                eval_op_tile(op, in, out, tn + op.local_spread);
                                if (!ring_at_f && op.spread > 0) {
                                    std::copy_n(out - op.spread + tn, op.spread,
                                                rings_f + op.ring_off);
                                }
                            }
                        }
                        t_rows += omp_get_wtime() - t1;
                    }
                }

                // Post-copy level-F carry-out after the tile loop.
                if (fused && ring_at_f) {
                    const auto* ops_f = m_ops[m_fuse].data();
                    for (SizeType k = fg.op_lo[m_fuse]; k < fg.op_hi[m_fuse];
                         ++k) {
                        const Op& op = ops_f[k];
                        if (op.spread > 0) {
                            std::copy_n(out_f + op.out - op.spread + n,
                                        op.spread, rings_f + op.ring_off);
                        }
                    }
                } else if (!fused) {
                    for (SizeType s = fg.sub_lo; s < fg.sub_hi; ++s) {
                        for (const auto& kr : m_subs[s].rows) {
                            if (kr.spread > 0) {
                                std::copy_n(out_f + kr.out - kr.spread + n,
                                            kr.spread, rings_f + kr.ring_off);
                            }
                        }
                    }
                }
            }
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

    template <typename AccT, bool Packed, typename StateT = float>
    void row_k(const AccT* P,
               const KRow& kr,
               SizeType len,
               StateT* __restrict__ out) const noexcept {
        const Leaf* leaves = m_leaves.data() + kr.leaf_lo;
        const SizeType nl  = kr.leaf_hi - kr.leaf_lo;
        if constexpr (!Packed) {
            if constexpr (std::is_same_v<StateT, float>) {
                if (nl == 2) {
                    const Leaf& lf0 = leaves[0];
                    const Leaf& lf1 = leaves[1];
                    const AccT* pa0 = P + static_cast<SizeType>(lf0.a);
                    const AccT* pb0 = P + static_cast<SizeType>(lf0.b);
                    const AccT* pa1 = P + static_cast<SizeType>(lf1.a);
                    const AccT* pb1 = P + static_cast<SizeType>(lf1.b);
                    const float w0  = lf0.w;
                    const float w1  = lf1.w;
#pragma omp simd
                    for (SizeType i = 0; i < len; ++i) {
                        out[i] = (box_sum<AccT>(pa0[i], pb0[i]) * w0) +
                                 (box_sum<AccT>(pa1[i], pb1[i]) * w1);
                    }
                    return;
                }
                if (nl == 1) {
                    const Leaf& lf0 = leaves[0];
                    const AccT* pa0 = P + static_cast<SizeType>(lf0.a);
                    const AccT* pb0 = P + static_cast<SizeType>(lf0.b);
                    const float w0  = lf0.w;
#pragma omp simd
                    for (SizeType i = 0; i < len; ++i) {
                        out[i] = box_sum<AccT>(pa0[i], pb0[i]) * w0;
                    }
                    return;
                }
            }
        } else {
            const float unit = m_int_unit;
            const auto store = [&](uint64_t v) -> StateT {
                if constexpr (std::is_integral_v<StateT>) {
                    return static_cast<StateT>(v);
                } else {
                    return static_cast<float>(v) * unit;
                }
            };
            if (nl == 2) {
                const Leaf& lf0 = leaves[0];
                const Leaf& lf1 = leaves[1];
                const AccT* pa0 = P + static_cast<SizeType>(lf0.a);
                const AccT* pb0 = P + static_cast<SizeType>(lf0.b);
                const AccT* pa1 = P + static_cast<SizeType>(lf1.a);
                const AccT* pb1 = P + static_cast<SizeType>(lf1.b);
                if (m_smear) {
                    const FDMTBoxRound r0 =
                        m_box_round[static_cast<SizeType>(lf0.s)];
                    const FDMTBoxRound r1 =
                        m_box_round[static_cast<SizeType>(lf1.s)];
#pragma omp simd
                    for (SizeType i = 0; i < len; ++i) {
                        const auto s0 = static_cast<uint64_t>(
                            static_cast<AccT>(pa0[i] - pb0[i]));
                        const auto s1 = static_cast<uint64_t>(
                            static_cast<AccT>(pa1[i] - pb1[i]));
                        const uint64_t v = box_round_apply<uint64_t>(s0, r0) +
                                           box_round_apply<uint64_t>(s1, r1);
                        out[i]           = store(v);
                    }
                } else {
#pragma omp simd
                    for (SizeType i = 0; i < len; ++i) {
                        const uint64_t v =
                            static_cast<uint64_t>(
                                static_cast<AccT>(pa0[i] - pb0[i])) +
                            static_cast<uint64_t>(
                                static_cast<AccT>(pa1[i] - pb1[i]));
                        out[i] = store(v);
                    }
                }
                return;
            }
            if (nl == 1) {
                const Leaf& lf0 = leaves[0];
                const AccT* pa0 = P + static_cast<SizeType>(lf0.a);
                const AccT* pb0 = P + static_cast<SizeType>(lf0.b);
                if (m_smear) {
                    const FDMTBoxRound r0 =
                        m_box_round[static_cast<SizeType>(lf0.s)];
#pragma omp simd
                    for (SizeType i = 0; i < len; ++i) {
                        const auto s0 = static_cast<uint64_t>(
                            static_cast<AccT>(pa0[i] - pb0[i]));
                        const uint64_t v = box_round_apply<uint64_t>(s0, r0);
                        out[i]           = store(v);
                    }
                } else {
#pragma omp simd
                    for (SizeType i = 0; i < len; ++i) {
                        const uint64_t v = static_cast<uint64_t>(
                            static_cast<AccT>(pa0[i] - pb0[i]));
                        out[i] = store(v);
                    }
                }
                return;
            }
        }
        constexpr SizeType kTile = 32;
        const auto eval_tile     = [&](SizeType i0, SizeType b) {
            if constexpr (!Packed) {
                if constexpr (std::is_same_v<StateT, float>) {
                    alignas(64) float acc[kTile]{};
                    for (SizeType m = 0; m < nl; ++m) {
                        const Leaf& lf = leaves[m];
                        const AccT* pa = P + static_cast<SizeType>(lf.a) + i0;
                        const AccT* pb = P + static_cast<SizeType>(lf.b) + i0;
                        const float w  = lf.w;
#pragma omp simd
                        for (SizeType k = 0; k < b; ++k) {
                            acc[k] += box_sum<AccT>(pa[k], pb[k]) * w;
                        }
                    }
                    std::copy_n(acc, b, out + i0);
                }
            } else {
                const float unit = m_int_unit;
                if (m_prod_u32) {
                    alignas(64) uint32_t acc[kTile]{};
                    for (SizeType m = 0; m < nl; ++m) {
                        const Leaf& lf = leaves[m];
                        const AccT* pa = P + static_cast<SizeType>(lf.a) + i0;
                        const AccT* pb = P + static_cast<SizeType>(lf.b) + i0;
                        if (m_smear) {
                            const FDMTBoxRound r =
                                m_box_round[static_cast<SizeType>(lf.s)];
                            if constexpr (std::is_integral_v<StateT>) {
                                const uint32_t mul = r.mul;
#pragma omp simd
                                for (SizeType k = 0; k < b; ++k) {
                                    const auto sum = static_cast<uint32_t>(
                                        static_cast<AccT>(pa[k] - pb[k]));
                                    acc[k] += sum * mul;
                                }
                            } else {
#pragma omp simd
                                for (SizeType k = 0; k < b; ++k) {
                                    const auto sum = static_cast<uint32_t>(
                                        static_cast<AccT>(pa[k] - pb[k]));
                                    acc[k] += box_round_apply<uint32_t>(sum, r);
                                }
                            }
                        } else {
#pragma omp simd
                            for (SizeType k = 0; k < b; ++k) {
                                acc[k] += static_cast<uint32_t>(
                                    static_cast<AccT>(pa[k] - pb[k]));
                            }
                        }
                    }
#pragma omp simd
                    for (SizeType k = 0; k < b; ++k) {
                        if constexpr (std::is_integral_v<StateT>) {
                            out[i0 + k] = static_cast<StateT>(acc[k]);
                        } else {
                            out[i0 + k] = static_cast<float>(acc[k]) * unit;
                        }
                    }
                } else {
                    alignas(64) uint64_t acc[kTile]{};
                    for (SizeType m = 0; m < nl; ++m) {
                        const Leaf& lf = leaves[m];
                        const AccT* pa = P + static_cast<SizeType>(lf.a) + i0;
                        const AccT* pb = P + static_cast<SizeType>(lf.b) + i0;
                        if (m_smear) {
                            const FDMTBoxRound r =
                                m_box_round[static_cast<SizeType>(lf.s)];
                            if constexpr (std::is_integral_v<StateT>) {
                                const auto mul = static_cast<uint64_t>(r.mul);
#pragma omp simd
                                for (SizeType k = 0; k < b; ++k) {
                                    const auto sum = static_cast<uint64_t>(
                                        static_cast<AccT>(pa[k] - pb[k]));
                                    acc[k] += sum * mul;
                                }
                            } else {
#pragma omp simd
                                for (SizeType k = 0; k < b; ++k) {
                                    const auto sum = static_cast<uint64_t>(
                                        static_cast<AccT>(pa[k] - pb[k]));
                                    acc[k] += box_round_apply<uint64_t>(sum, r);
                                }
                            }
                        } else {
#pragma omp simd
                            for (SizeType k = 0; k < b; ++k) {
                                acc[k] += static_cast<uint64_t>(
                                    static_cast<AccT>(pa[k] - pb[k]));
                            }
                        }
                    }
#pragma omp simd
                    for (SizeType k = 0; k < b; ++k) {
                        if constexpr (std::is_integral_v<StateT>) {
                            out[i0 + k] = static_cast<StateT>(acc[k]);
                        } else {
                            out[i0 + k] = static_cast<float>(acc[k]) * unit;
                        }
                    }
                }
            }
        };
        SizeType i0 = 0;
        for (; i0 + kTile <= len; i0 += kTile) {
            eval_tile(i0, kTile);
        }
        if (i0 < len) {
            eval_tile(i0, len - i0);
        }
    }

    template <typename StateT = float>
    void merge_level(SizeType l, SizeType n, StateT* dmt, SizeType t0) {
        const StateT* in      = half<StateT>(l - 1);
        const bool search_out = (l == m_root && dmt == nullptr);
        StateT* out_half =
            (l == m_root && dmt != nullptr) ? dmt + t0 : half<StateT>(l);
        const auto& ops = m_ops[l];
#pragma omp parallel num_threads(m_nthreads)
        {
#pragma omp for schedule(static) nowait
            for (SizeType k = 0; k < ops.size(); ++k) {
                const Op& op = ops[k];
                StateT* __restrict__ out =
                    search_out
                        ? (out_half + (static_cast<SizeType>(op.row) * m_chunk))
                        : (out_half + op.out);
                const SizeType m                = n + op.local_spread;
                const StateT* __restrict__ tail = in + op.tail;
                if (op.head == kOpNone) {
                    std::copy_n(tail, m, out);
                } else if (search_out) {
                    const StateT* __restrict__ head = in + op.head;
#pragma omp simd
                    for (SizeType i = 0; i < m; ++i) {
                        out[i] = tail[i] + head[i];
                    }
                } else {
                    const StateT* __restrict__ head = in + op.head;
                    skew_stream_add(tail, head, out, m);
                }
            }
            if (!search_out) {
                skew_stream_fence();
            }
        }
    }

    const plans::FDMTPlan& m_plan;
    bool m_smear;
    int m_nthreads;
    SizeType m_fuse;
    SizeType m_fuse_ring{0};
    SizeType m_level;
    SizeType m_root;
    SizeType m_nchans;
    SizeType m_nsamps;
    SizeType m_chunk;
    SizeType m_search_from{0}; ///< first searched level (m_root: none)
    std::vector<std::vector<SizeType>> m_emit_off; ///< [l][row], searched l
    std::vector<std::vector<uint64_t>> m_emit_lag; ///< [l][row], searched l
    bool m_search_int{false};
    bool m_search_u32{true};
    SizeType m_state_elem{sizeof(float)};
    SizeType m_nbits{32};
    SizeType m_ring_elem{sizeof(float)};
    SizeType m_nonint_until{0}; ///< stream position (see BlockKind)
    FDMTSkewWindows m_win;
    FDMTSkewWindows m_loc_win;

    float m_int_unit{1.0F};
    bool m_prod_u32{false};
    std::array<bool, 17> m_prod_u32_by_nbits{};
    std::vector<float> m_box_w;
    std::vector<FDMTBoxRound> m_box_round;

    std::vector<KSub> m_subs;
    std::vector<FusedGroup> m_groups;
    std::vector<GroupBatch> m_group_batches;
    std::vector<Leaf> m_leaves;
    int64_t m_need_back_max{0};
    SizeType m_ring_len{0};
    std::vector<int64_t> m_chan_reach;
    std::vector<SizeType> m_chan_ring_len;
    std::vector<SizeType> m_chan_ring_off;

    std::vector<std::vector<SizeType>> m_row_off; ///< per level, per row
    std::vector<SizeType> m_level_len;
    std::vector<SizeType> m_ring_f_off;
    std::vector<float> m_ring_f;
    std::vector<std::vector<Op>> m_ops;

    SizeType m_tile{0};
    SizeType m_stripe_subs{1};
    SizeType m_sub_stripe_elems{0};
    SizeType m_half0_len{0};
    SizeType m_half1_len{0};
    SizeType m_group_a_elems{0};
    SizeType m_group_b_elems{0};
    bool m_shared_group{false};
    bool m_group_in_idle{false};
    SkewLargeArray<float> m_state;
    SkewLargeArray<std::byte> m_ring;
    SkewLargeArray<std::byte> m_scratch;
    SizeType m_stripe_bytes{0};
    SizeType m_group_a_bytes{0};
    SizeType m_group_b_bytes{0};
    SizeType m_thread_scratch_bytes{0};
    SizeType m_stream_pos{0};
    Memory m_mem;
    Profile m_prof;
};

} // namespace dmt::algorithms::detail
