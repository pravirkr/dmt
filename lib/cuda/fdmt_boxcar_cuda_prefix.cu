#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <fmt/format.h>

#include "dmt/boxcar.hpp"
#include "dmt/boxcar_runs.hpp"
#include "dmt/fdmt_variance.hpp"
#include "dmt/gpu_compat.cuh"
#include "dmt/gpu_utils.cuh"
#include "dmt/logging.hpp"

#include "fdmt_boxcar_cuda_engine.cuh"

// GPU FDMTBoxcar engine: prefix-sum & sparse-table screen (`kPrefix`),
// unscreened (`kNone`), and wide-boxcar / small-SMEM `kGlobalRing` scorers.

namespace dmt::algorithms::detail::fdmt_boxcar_cuda {
namespace {

// Whether any sample of the tile crosses any width's threshold. Threads
// take samples in a striped order, so ring reads are conflict-free. The
// run bookkeeping runs only for tiles where this (or an open run) holds;
// otherwise every peak mode leaves its state unchanged.
template <typename T,
          int kBlockThreads,
          bool kFull,
          bool kCold,
          bool kTrapezoid>
DMT_D bool detect_tile(const T* ring,
                       uint32_t mask,
                       uint32_t position,
                       int n,
                       const uint32_t* widths,
                       const T* thresholds,
                       SizeType nfilters,
                       uint64_t start,
                       uint64_t warmup) {
    const int tid       = static_cast<int>(threadIdx.x);
    const uint32_t base = position + 1 + static_cast<uint32_t>(tid);
    T current[kItems];
#pragma unroll
    for (int k = 0; k < kItems; ++k) {
        current[k] = ring[(base + k * kBlockThreads) & mask];
    }
    bool hit = false;
    for (SizeType f = 0; f < nfilters; ++f) {
        const uint32_t width  = widths[f];
        const uint32_t w_edge = kTrapezoid ? trapezoid_edge_width(width) : 0U;
        const T threshold     = thresholds[f];
#pragma unroll
        for (int k = 0; k < kItems; ++k) {
            const int s = k * kBlockThreads + tid;
            if (!kFull && s >= n) {
                break;
            }
            const uint32_t idx = base + k * kBlockThreads;
            T sum              = current[k] - ring[(idx - width) & mask];
            if constexpr (kTrapezoid) {
                if (w_edge > 0) {
                    sum += ring[(idx - w_edge) & mask] -
                           ring[(idx - width + w_edge) & mask];
                }
            }
            bool h = sum >= threshold;
            if constexpr (kCold) {
                h = h && start + s >= warmup + width - 1;
            }
            hit = hit || h;
        }
    }
    return hit;
}

// detect_tile through the width screen (BoxcarScreenGroup, float rows, no
// trapezoid): `mn` (lookback + n + 1 entries of shared memory) receives the
// tile's prefixes in linear order and is raised in place to range-minimum
// level k as the groups need it (entry j then holds min(P[j .. j + 2^k - 1])).
// Each update reads only entries at or above its own index, so updating in
// ascending batches (read, sync, write, sync) is safe. Groups whose bound
// passes are checked width by width, exactly as detect_tile does; a group
// below the table's current level (sparse widths) is checked directly.
template <typename T, int kBlockThreads, bool kFull, bool kCold>
DMT_D bool detect_tile_grouped(const T* ring,
                               uint32_t mask,
                               uint32_t position,
                               uint32_t lookback,
                               int n,
                               const uint32_t* widths,
                               const T* thresholds,
                               const BoxcarScreenGroup* groups,
                               int ngroups,
                               const T* group_thresholds,
                               T* mn,
                               uint64_t start,
                               uint64_t warmup) {
    const int tid       = static_cast<int>(threadIdx.x);
    const int len       = static_cast<int>(lookback) + n + 1;
    const uint32_t lin0 = position - lookback;
    for (int j = tid; j < len; j += kBlockThreads) {
        mn[j] = ring[(lin0 + static_cast<uint32_t>(j)) & mask];
    }
    __syncthreads();
    T current[kItems];
#pragma unroll
    for (int k = 0; k < kItems; ++k) {
        const int j = static_cast<int>(lookback) + 1 + k * kBlockThreads + tid;
        current[k]  = j < len ? mn[j] : T{0};
    }
    const auto exact = [&](uint32_t f) {
        const uint32_t width = widths[f];
        const T th           = thresholds[f];
        bool hit             = false;
#pragma unroll
        for (int k = 0; k < kItems; ++k) {
            const int s = k * kBlockThreads + tid;
            if (!kFull && s >= n) {
                break;
            }
            const uint32_t idx = position + 1U + static_cast<uint32_t>(s);
            bool h             = current[k] - ring[(idx - width) & mask] >= th;
            if constexpr (kCold) {
                h = h && start + s >= warmup + width - 1;
            }
            hit = hit || h;
        }
        return hit;
    };
    bool hit  = false;
    int level = 0;
    for (int g = 0; g < ngroups; ++g) {
        const BoxcarScreenGroup grp = groups[g];
        if (static_cast<int>(grp.level) < level) {
            // Sparse gap: one width, checked directly from the ring.
            for (uint32_t f = grp.f_lo; f < grp.f_hi; ++f) {
                hit = exact(f) || hit;
            }
            continue;
        }
        while (level < static_cast<int>(grp.level)) {
            const int half  = 1 << level;
            const int valid = len - (2 * half) + 1;
            for (int b0 = 0; b0 < valid; b0 += kItems * kBlockThreads) {
                T v[kItems];
#pragma unroll
                for (int r = 0; r < kItems; ++r) {
                    const int j = b0 + (r * kBlockThreads) + tid;
                    if (j < valid) {
                        v[r] = fminf(mn[j], mn[j + half]);
                    }
                }
                __syncthreads();
#pragma unroll
                for (int r = 0; r < kItems; ++r) {
                    const int j = b0 + (r * kBlockThreads) + tid;
                    if (j < valid) {
                        mn[j] = v[r];
                    }
                }
                __syncthreads();
            }
            ++level;
        }
        // Bound of the whole group; a pass is checked width by width from
        // the ring (mn holds range minima now).
        const int span    = 1 << grp.level;
        const T bound_thr = group_thresholds[g];
        bool pass         = false;
#pragma unroll
        for (int k = 0; k < kItems; ++k) {
            const int s = k * kBlockThreads + tid;
            if (!kFull && s >= n) {
                break;
            }
            const int j = static_cast<int>(lookback) + 1 + s;
            const T m   = fminf(mn[j - static_cast<int>(grp.w_hi)],
                                mn[j - static_cast<int>(grp.w_lo) - span + 1]);
            pass        = pass || current[k] - m >= bound_thr;
        }
        if (pass) {
            for (uint32_t f = grp.f_lo; f < grp.f_hi; ++f) {
                hit = exact(f) || hit;
            }
        }
    }
    __syncthreads(); // mn is reused by the next tile
    return hit;
}

// detect_tile_grouped for packed rows (uint64 running sums, which only
// grow): with m1 the row's mean per sample, the centred sums
// c(j) = (P(j) - P(0)) - m1 j satisfy S_w - m1 w = c(t) - c(t - w), so the
// range-minimum bound of c is compared with the group's smallest
// cutoff - mean (rounded down on the host). c is float: a group passes
// within a margin of 2^-20 of the tile's magnitude, several times the
// float rounding of c, m1 and the bound, so the screen stays conservative.
// Passing groups are checked exactly in integers from the ring.
template <int kBlockThreads, bool kFull, bool kCold>
DMT_D bool detect_tile_grouped_int(const uint64_t* ring,
                                   uint32_t mask,
                                   uint32_t position,
                                   uint32_t lookback,
                                   int n,
                                   const uint32_t* widths,
                                   const uint64_t* thresholds,
                                   const BoxcarScreenGroup* groups,
                                   int ngroups,
                                   const float* group_thresholds,
                                   float m1,
                                   float* mn,
                                   uint64_t start,
                                   uint64_t warmup) {
    const int tid       = static_cast<int>(threadIdx.x);
    const int len       = static_cast<int>(lookback) + n + 1;
    const uint32_t lin0 = position - lookback;
    const uint64_t p0   = ring[lin0 & mask];
    for (int j = tid; j < len; j += kBlockThreads) {
        const uint64_t e = ring[(lin0 + static_cast<uint32_t>(j)) & mask] - p0;
        mn[j]            = __ull2float_rn(e) - (m1 * static_cast<float>(j));
    }
    const float extent =
        __ull2float_ru(ring[(lin0 + static_cast<uint32_t>(len - 1)) & mask] -
                       p0) +
        (fabsf(m1) * static_cast<float>(len));
    const float margin = (extent * 0x1p-20F) + 1.0F;
    __syncthreads();
    float current[kItems];
    uint64_t current_raw[kItems];
#pragma unroll
    for (int k = 0; k < kItems; ++k) {
        const int j = static_cast<int>(lookback) + 1 + k * kBlockThreads + tid;
        current[k]  = j < len ? mn[j] : 0.0F;
        current_raw[k] = ring[(position + 1U +
                               static_cast<uint32_t>(k * kBlockThreads + tid)) &
                              mask];
    }
    const auto exact = [&](uint32_t f) {
        const uint32_t width = widths[f];
        const uint64_t th    = thresholds[f];
        bool hit             = false;
#pragma unroll
        for (int k = 0; k < kItems; ++k) {
            const int s = k * kBlockThreads + tid;
            if (!kFull && s >= n) {
                break;
            }
            const uint32_t idx = position + 1U + static_cast<uint32_t>(s);
            bool h = current_raw[k] - ring[(idx - width) & mask] >= th;
            if constexpr (kCold) {
                h = h && start + s >= warmup + width - 1;
            }
            hit = hit || h;
        }
        return hit;
    };
    bool hit  = false;
    int level = 0;
    for (int g = 0; g < ngroups; ++g) {
        const BoxcarScreenGroup grp = groups[g];
        if (static_cast<int>(grp.level) < level) {
            for (uint32_t f = grp.f_lo; f < grp.f_hi; ++f) {
                hit = exact(f) || hit;
            }
            continue;
        }
        while (level < static_cast<int>(grp.level)) {
            const int half  = 1 << level;
            const int valid = len - (2 * half) + 1;
            for (int b0 = 0; b0 < valid; b0 += kItems * kBlockThreads) {
                float v[kItems];
#pragma unroll
                for (int r = 0; r < kItems; ++r) {
                    const int j = b0 + (r * kBlockThreads) + tid;
                    if (j < valid) {
                        v[r] = fminf(mn[j], mn[j + half]);
                    }
                }
                __syncthreads();
#pragma unroll
                for (int r = 0; r < kItems; ++r) {
                    const int j = b0 + (r * kBlockThreads) + tid;
                    if (j < valid) {
                        mn[j] = v[r];
                    }
                }
                __syncthreads();
            }
            ++level;
        }
        const int span        = 1 << grp.level;
        const float bound_thr = group_thresholds[g] - margin;
        bool pass             = false;
#pragma unroll
        for (int k = 0; k < kItems; ++k) {
            const int s = k * kBlockThreads + tid;
            if (!kFull && s >= n) {
                break;
            }
            const int j = static_cast<int>(lookback) + 1 + s;
            const float m =
                fminf(mn[j - static_cast<int>(grp.w_hi)],
                      mn[j - static_cast<int>(grp.w_lo) - span + 1]);
            pass = pass || current[k] - m >= bound_thr;
        }
        if (pass) {
            for (uint32_t f = grp.f_lo; f < grp.f_hi; ++f) {
                hit = exact(f) || hit;
            }
        }
    }
    __syncthreads(); // mn is reused by the next tile
    return hit;
}

// kScreen: tiles are screened by width group (detect_tile_grouped*); a
// separate instantiation, so the plain kernel keeps its register budget.
// kBlockNoise: per-call mask tables are in use (float rows only). A
// separate instance, so the common unmasked launch carries none of their
// state (it cost ~160 B of stack per thread and 15-50% of the run time).
// kKeep: OverflowPolicy::kKeepBrightest (KeepWriter, per-row stores); a
// separate instance, so kDropNew keeps its code and output.
// kGlobalRing: the prefix ring lives in device memory (`ring_global`, one
// ring of mask + 1 entries per block, blocks launched in batches from task
// `task_base`) instead of shared memory: boxcars wider than the fast path
// (kFastMaxWidth) or than the device's shared memory. No width screen and
// no shared tables; same candidates, no speed promise.
template <typename T,
          int kBlockThreads,
          bool kScreen,
          bool kBlockNoise = false,
          typename C       = T,
          bool kKeep       = false,
          bool kGlobalRing = false>
__global__ void __launch_bounds__(kBlockThreads, 4)
    kernel_fdmt_boxcar(FDMTSearchInput input,
                       const BoxcarRow* rows,
                       SizeType row_base,
                       SizeType ndm,
                       SizeType total_rows,
                       const uint32_t* widths,
                       SizeType nfilters,
                       SizeType max_width,
                       const float* thresholds,
                       const float* inv_sigma,
                       C* carry_all,
                       BoxcarPeak* peaks_all,
                       uint8_t* open_all,
                       SizeType peak_count,
                       RawCandidate* candidates,
                       CandidateCounts* counts,
                       uint64_t* slots,
                       uint64_t* per_dm_all,
                       SizeType capacity,
                       SizeType dm_cap,
                       PeakFindingMode mode,
                       bool skip_cold,
                       bool root,
                       const uint64_t* integer_thresholds,
                       const double* means,
                       double unit,
                       uint32_t mask,
                       bool shared_tables,
                       bool use_trapezoid,
                       const BoxcarScreenGroup* screen_groups,
                       int screen_ngroups,
                       const float* screen_thresholds,
                       const float* screen_m1,
                       SizeType screen_offset,
                       BlockTables block_tables,
                       SizeType group_offset,
                       BoxcarKeepEntry* keep_all,
                       SizeType keep_q,
                       T* ring_global,
                       SizeType task_base,
                       bool bypass_screen) {
    constexpr int kBlockTile = kBlockThreads * kItems;
    const SizeType task =
        kGlobalRing ? task_base + blockIdx.x : SizeType{blockIdx.x};
    // Lagged rows (skewed-window subband search): this row's own samples.
    if (input.row_lag != nullptr) {
        FDMTSearchInput lagged;
        if (!lagged_row_input(input, rows[row_base + task % ndm].row, lagged)) {
            return;
        }
        input = lagged;
    }
    if (!stop_row_input(input)) {
        return; // drain(): past the end of the stream
    }
    extern __shared__ __align__(16) unsigned char storage[];
    auto& shared = *reinterpret_cast<BoxcarShared<T, kBlockThreads>*>(storage);
    T* ring = kGlobalRing
                  ? ring_global + (SizeType{blockIdx.x} * (SizeType{mask} + 1))
                  : reinterpret_cast<T*>(storage + sizeof(shared));
    const SizeType beam      = input.beam_begin + task / ndm;
    const SizeType d         = task % ndm;
    const SizeType row_index = row_base + d;
    const SizeType index     = beam * total_rows + row_index;
    const BoxcarRow row      = rows[row_index];
    C* carry                 = carry_all + index * max_width;
    BoxcarPeak* peaks        = peaks_all + index * peak_count;
    RawCandidate prototype   = row.candidate;
    prototype.beam_idx       = static_cast<uint32_t>(beam);
    CandidateWriter writer{candidates, counts,    slots,    per_dm_all + index,
                           widths,     prototype, capacity, dm_cap};
    writer.means   = means ? means + row_index * nfilters : nullptr;
    writer.scales  = inv_sigma + row_index * nfilters;
    writer.unit    = unit;
    writer.integer = std::is_integral_v<T> &&
                     mode != PeakFindingMode::kContiguousPeakAcrossWidths;
    // kKeepBrightest: the row's store and this thread's pending candidates.
    struct NoKeep {};
    std::conditional_t<kKeep, KeepPending, NoKeep> pending{};
    std::conditional_t<kKeep, KeepWriter, NoKeep> keeper{};
    if constexpr (kKeep) {
        keeper = KeepWriter{counts,
                            keep_all + (index * keep_q),
                            per_dm_all + index,
                            keep_q,
                            &pending,
                            writer.means,
                            writer.scales,
                            unit,
                            writer.integer};
    }
    const auto& out = [&]() -> const auto& {
        if constexpr (kKeep) {
            return keeper;
        } else {
            return writer;
        }
    }();
    const int tid       = static_cast<int>(threadIdx.x);
    const auto lookback = static_cast<uint32_t>(max_width - 1);
    // Per-call kill masks (float rows): the tables this call's endpoints
    // use. One table: the row reads it as the constructor's. Several
    // ("mixed"): the screen takes their per-width minimum (a lower bound
    // for every endpoint) and the bookkeeping picks the table per sample.
    const float* plane_thr = thresholds + row_index * nfilters;
    const float* plane_inv = inv_sigma + row_index * nfilters;
    const float* inv_row   = plane_inv;
    const float* thr_float = plane_thr;
    const float* in_play[4][2];
    int n_play           = 0;
    bool mixed           = false;
    uint64_t row_support = 0;
    if constexpr (kBlockNoise && !std::is_integral_v<T>) {
        if (block_tables.count > 0) {
            row_support       = block_tables.support[row_index];
            const uint64_t s0 = input.sample_begin;
            const uint64_t s1 = s0 + input.nsamps - 1;
            const int e0      = block_entry(block_tables, s0);
            const int e1      = block_entry(block_tables, s1);
            const auto add    = [&](int e, int k) {
                const float* t = block_tables.thr[e][k];
                const float* v = block_tables.inv[e][k];
                t = t != nullptr ? t + row_index * nfilters : plane_thr;
                v = v != nullptr ? v + row_index * nfilters : plane_inv;
                for (int j = 0; j < n_play; ++j) {
                    if (in_play[j][0] == t) {
                        return;
                    }
                }
                in_play[n_play][0] = t;
                in_play[n_play][1] = v;
                ++n_play;
            };
            for (int e = e0; e <= e1; ++e) {
                const uint64_t first = block_tables.first[e];
                const uint64_t lo    = e == e0 ? s0 : first;
                const uint64_t hi = e < e1 ? block_tables.first[e + 1] - 1 : s1;
                if (!block_tables.transition[e]) {
                    add(e, 0);
                    continue;
                }
                // Transition [1] before first + support + w - 1 (as high as
                // max_width - 1), steady [0] from widths[0] - 1 onward.
                if (lo < first + row_support + max_width - 1) {
                    add(e, 1);
                }
                if (hi + 1 >= first + row_support + widths[0]) {
                    add(e, 0);
                }
            }
            if (n_play == 1) {
                thr_float = in_play[0][0];
                inv_row   = in_play[0][1];
            } else {
                mixed = true;
            }
        }
    }
    const T* thr_row = [&] {
        if constexpr (std::is_integral_v<T>) {
            return integer_thresholds + row_index * nfilters;
        } else {
            return thr_float;
        }
    }();
    const uint32_t* wtab = widths;
    const T* ttab        = thr_row;
    if (shared_tables) {
        T* s_thr       = ring + mask + 1;
        auto* s_widths = reinterpret_cast<uint32_t*>(s_thr + nfilters);
        for (SizeType f = tid; f < nfilters; f += kBlockThreads) {
            T th = thr_row[f];
            if constexpr (!std::is_integral_v<T>) {
                if (mixed) {
                    th = in_play[0][0][f];
                    for (int j = 1; j < n_play; ++j) {
                        th = th < in_play[j][0][f] ? th : in_play[j][0][f];
                    }
                }
            }
            s_thr[f]    = th;
            s_widths[f] = widths[f];
        }
        wtab = s_widths;
        ttab = s_thr;
    }
    // Mixed tables without the shared row, or kNone: every tile takes the
    // exact bookkeeping (no screen bound is stored / evaluated).
    bool force_bookkeeping = bypass_screen || (mixed && !shared_tables);
    // Group screen bounds of tables other than the constructor's: the
    // per-group minimum of the row's (staged) thresholds.
    const float* group_thr =
        screen_ngroups > 0 ? screen_thresholds + row_index * screen_ngroups
                           : nullptr;
    if constexpr (kScreen && !std::is_integral_v<T>) {
        if ((thr_float != plane_thr || mixed) && group_offset == 0) {
            force_bookkeeping = true; // no room for the group bounds
        } else if (thr_float != plane_thr || mixed) {
            auto* s_grp = reinterpret_cast<float*>(storage + group_offset);
            __syncthreads();
            for (int g = tid; g < screen_ngroups; g += kBlockThreads) {
                float th = 3.402823466e38F;
                for (uint32_t f = screen_groups[g].f_lo;
                     f < screen_groups[g].f_hi; ++f) {
                    th = th < ttab[f] ? th : ttab[f];
                }
                s_grp[g] = th;
            }
            group_thr = s_grp;
        }
    }
    // Open runs force the bookkeeping path until they close.
    const auto any_open = [&] {
        bool open = false;
        if (mode != PeakFindingMode::kAllAboveThreshold) {
            for (SizeType f = tid; f < peak_count; f += kBlockThreads) {
                open = open || peaks[f].open;
            }
        }
        return __syncthreads_or(open) != 0;
    };
    // Whether a run is open, as any_open() would find: kept per row by the
    // last call, so the common call never reads the row's peaks.
    bool open = input.sample_begin > 0 && open_all[index] != 0;
    if (input.sample_begin == 0) {
        for (uint32_t j = tid; j <= lookback; j += kBlockThreads) {
            ring[j & mask] = T{0};
        }
        __syncthreads();
    } else {
        if (tid == 0) {
            ring[0] = 0;
        }
        __syncthreads();
        // Bootstrap only the short raw carry, never an entire DM-time row.
        const auto carry0 =
            static_cast<uint32_t>(input.sample_begin % max_width);
        const auto width32 = static_cast<uint32_t>(max_width);
        for (uint32_t p = 0; p < lookback; p += kBlockTile) {
            const T base = ring[p & mask];
            T x[kItems];
            for (int j = 0; j < kItems; ++j) {
                const uint32_t t    = p + tid * kItems + j;
                const uint32_t back = t < lookback ? lookback - t : 0;
                x[j] = t < lookback && input.sample_begin >= back
                           ? carry[carry0 >= back ? carry0 - back
                                                  : carry0 + width32 - back]
                           : T{0};
            }
            prefix_tile<T, kBlockThreads>(shared, ring, mask, x, p, base);
        }
    }
    for (SizeType t0 = 0; t0 < input.nsamps; t0 += kBlockTile) {
        const int n = static_cast<int>(
            input.nsamps - t0 < kBlockTile ? input.nsamps - t0 : kBlockTile);
        const auto position = static_cast<uint32_t>(lookback + t0);
        const T base        = ring[position & mask];
        T x[kItems];
#pragma unroll
        for (int j = 0; j < kItems; ++j) {
            const SizeType local = tid * kItems + j;
            x[j]                 = 0;
            if (local < static_cast<SizeType>(n)) {
                if constexpr (std::is_integral_v<T>) {
                    x[j] = boxcar_integer_sample(input, row, beam, t0 + local,
                                                 root);
                } else {
                    x[j] = boxcar_sample(input, row, beam, t0 + local, root);
                }
            }
        }
        prefix_tile<T, kBlockThreads>(shared, ring, mask, x, position, base);
        const uint64_t start = input.sample_begin + t0;
        // Tiles past the warm-up of every width skip the per-width check.
        const bool cold       = skip_cold && start + 1 < row.warmup + max_width;
        const auto run_detect = [&](auto full_tag, auto cold_tag) {
            constexpr bool kFull = decltype(full_tag)::value;
            constexpr bool kCold = decltype(cold_tag)::value;
            if constexpr (kScreen) {
                if constexpr (std::is_integral_v<T>) {
                    return detect_tile_grouped_int<kBlockThreads, kFull, kCold>(
                        ring, mask, position, lookback, n, wtab, ttab,
                        screen_groups, screen_ngroups,
                        screen_thresholds + row_index * screen_ngroups,
                        screen_m1[row_index],
                        reinterpret_cast<float*>(storage + screen_offset),
                        start, row.warmup);
                } else {
                    return detect_tile_grouped<T, kBlockThreads, kFull, kCold>(
                        ring, mask, position, lookback, n, wtab, ttab,
                        screen_groups, screen_ngroups, group_thr,
                        reinterpret_cast<T*>(storage + screen_offset), start,
                        row.warmup);
                }
            }
            return use_trapezoid
                       ? detect_tile<T, kBlockThreads, kFull, kCold, true>(
                             ring, mask, position, n, wtab, ttab, nfilters,
                             start, row.warmup)
                       : detect_tile<T, kBlockThreads, kFull, kCold, false>(
                             ring, mask, position, n, wtab, ttab, nfilters,
                             start, row.warmup);
        };
        bool hit;
        if (force_bookkeeping) {
            hit = true;
        } else if (n == kBlockTile) {
            hit = cold ? run_detect(cuda::std::true_type{},
                                    cuda::std::true_type{})
                       : run_detect(cuda::std::true_type{},
                                    cuda::std::false_type{});
        } else {
            hit = cold ? run_detect(cuda::std::false_type{},
                                    cuda::std::true_type{})
                       : run_detect(cuda::std::false_type{},
                                    cuda::std::false_type{});
        }
        if (__syncthreads_or(hit || open) != 0) {
            // Rare path: exact run bookkeeping in each thread's four
            // consecutive samples.
            float best[kItems];
            uint32_t best_filter[kItems];
            T current[kItems];
            uint64_t raw_sums[kItems]{};
            for (int j = 0; j < kItems; ++j) {
                current[j]     = ring[(position + tid * kItems + j + 1) & mask];
                best[j]        = -1.0F;
                best_filter[j] = 0;
            }
            for (SizeType f = 0; f < nfilters; ++f) {
                bool above[kItems];
                float snr[kItems];
                uint32_t filters[kItems];
                const uint32_t width = wtab[f];
                const uint32_t w_edge =
                    use_trapezoid ? trapezoid_edge_width(width) : 0U;
                T threshold = ttab[f];
                float scale = inv_row[f];
                bool f_hit  = false;
                for (int j = 0; j < kItems; ++j) {
                    const uint32_t local  = tid * kItems + j;
                    const uint64_t sample = start + local;
                    if constexpr (!std::is_integral_v<T>) {
                        if (mixed) {
                            const int e = block_entry(block_tables, sample);
                            const int k = block_table(block_tables, e, sample,
                                                      width, row_support);
                            const float* t = block_tables.thr[e][k];
                            const float* v = block_tables.inv[e][k];
                            threshold      = t != nullptr
                                                 ? t[row_index * nfilters + f]
                                                 : plane_thr[f];
                            scale = v != nullptr ? v[row_index * nfilters + f]
                                                 : plane_inv[f];
                        }
                    }
                    const uint32_t idx = position + local + 1;
                    T sum = current[j] - ring[(idx - width) & mask];
                    if (w_edge > 0) {
                        sum += ring[(idx - w_edge) & mask] -
                               ring[(idx - width + w_edge) & mask];
                    }
                    above[j] =
                        local < static_cast<uint32_t>(n) &&
                        (!skip_cold || sample >= row.warmup + width - 1) &&
                        sum >= threshold;
                    f_hit  = f_hit || above[j];
                    snr[j] = 0;
                    if constexpr (std::is_integral_v<T>) {
                        // In per-width and all modes, integer threshold
                        // crossings stay integer until a candidate is emitted.
                        // Across-width comparison needs real S/N at each
                        // sample.
                        raw_sums[j] = sum;
                        if (above[j] &&
                            mode ==
                                PeakFindingMode::kContiguousPeakAcrossWidths) {
                            snr[j] = static_cast<float>(
                                (static_cast<double>(sum) -
                                 means[row_index * nfilters + f]) *
                                (scale * unit));
                        }
                    } else {
                        snr[j] = above[j] ? sum * scale : 0.0F;
                    }
                    filters[j] = static_cast<uint32_t>(f);
                    if (mode == PeakFindingMode::kContiguousPeakAcrossWidths) {
                        if (above[j] && snr[j] > best[j]) {
                            best[j]        = snr[j];
                            best_filter[j] = filters[j];
                        }
                    } else if (mode == PeakFindingMode::kAllAboveThreshold &&
                               above[j]) {
                        BoxcarPeak peak{sample, sample,     sample,
                                        snr[j], filters[j], true};
                        if constexpr (std::is_integral_v<T>) {
                            peak.raw_sum = sum;
                        }
                        out(peak);
                    }
                }
                if (mode == PeakFindingMode::kContiguousPeakPerWidth &&
                    __syncthreads_or(f_hit || peaks[f].open) != 0) {
                    reduce_runs<T, kBlockThreads>(
                        shared, above, snr, filters, start, n, peaks + f, out,
                        raw_sums, std::is_integral_v<T>);
                    // Only reduce_runs() emits in this mode: drain behind
                    // the same block vote, not once per width.
                    if constexpr (kKeep) {
                        keep_drain(shared, pending, keeper);
                    }
                }
                if constexpr (kKeep) {
                    // Direct emission above: drain each width (mode is
                    // block-uniform).
                    if (mode == PeakFindingMode::kAllAboveThreshold) {
                        keep_drain(shared, pending, keeper);
                    }
                }
            }
            if (mode == PeakFindingMode::kContiguousPeakAcrossWidths) {
                bool above[kItems];
                for (int j = 0; j < kItems; ++j) {
                    above[j] = best[j] >= 0;
                }
                reduce_runs<T, kBlockThreads>(shared, above, best, best_filter,
                                              start, n, peaks, out, raw_sums,
                                              false);
                if constexpr (kKeep) {
                    keep_drain(shared, pending, keeper);
                }
            }
            open = any_open();
        }
        // Recenter the retained prefix boundaries only when another tile
        // follows.
        if (t0 + kBlockTile < input.nsamps) {
            const uint32_t end = position + static_cast<uint32_t>(n);
            const T origin     = ring[(end - lookback) & mask];
            __syncthreads();
            for (uint32_t j = tid; j <= lookback; j += kBlockThreads) {
                ring[(end - lookback + j) & mask] -= origin;
            }
            __syncthreads();
        }
    }
    if (tid == 0) {
        open_all[index] = open ? 1 : 0;
    }
    // The root's old FDMT history remains available until this kernel ends.
    const SizeType keep = (input.nsamps < max_width ? input.nsamps : max_width);
    // Carry slot of the first kept sample; the kept samples follow it.
    const auto keep0 = static_cast<uint32_t>(
        (input.sample_begin + input.nsamps - keep) % max_width);
    const auto width32 = static_cast<uint32_t>(max_width);
    for (SizeType t = tid; t < keep; t += kBlockThreads) {
        const SizeType sample = input.nsamps - keep + t;
        uint32_t slot         = keep0 + static_cast<uint32_t>(t);
        slot -= slot >= width32 ? width32 : 0U;
        if constexpr (std::is_integral_v<T>) {
            carry[slot] = static_cast<C>(
                boxcar_integer_sample(input, row, beam, sample, root));
        } else {
            carry[slot] = boxcar_sample(input, row, beam, sample, root);
        }
    }
}

} // namespace

void FDMTBoxcarCudaEngine::init_prefix_screen() {
    if (m_config.use_trapezoid) {
        return;
    }
    const auto groups = boxcar_screen_groups(m_planes.front().widths, false);
    if (groups.empty()) {
        return;
    }
    // Per row, each group's smallest threshold (float rows) or smallest
    // cutoff - mean, rounded down (packed rows), and the packed rows' mean
    // per sample (from the width-1 mean).
    const bool packed = m_config.input_nbits != 32;
    std::vector<float> group_thr;
    std::vector<float> m1;
    group_thr.reserve(m_rows.size() * groups.size());
    for (const auto& p : m_planes) {
        for (SizeType d = 0; d < p.dm_grid.size(); ++d) {
            const SizeType base = d * m_filters;
            for (const auto& g : groups) {
                double th = std::numeric_limits<double>::infinity();
                for (uint32_t f = g.f_lo; f < g.f_hi; ++f) {
                    th = std::min(th,
                                  packed ? static_cast<double>(
                                               p.integer_thresh_lut[base + f]) -
                                               p.integer_mean_lut[base + f]
                                         : static_cast<double>(
                                               p.raw_thresh_lut[base + f]));
                }
                auto thf = static_cast<float>(th);
                if (static_cast<double>(thf) > th) {
                    thf = std::nextafter(
                        thf, -std::numeric_limits<float>::infinity());
                }
                group_thr.push_back(thf);
            }
            if (packed) {
                // The mean per sample comes from widths[0], a plain boxcar
                // (no screen with use_trapezoid).
                m1.push_back(
                    static_cast<float>(p.integer_mean_lut[base] /
                                       static_cast<double>(p.widths[0])));
            }
        }
    }
    m_screen_groups_d     = groups;
    m_screen_thresholds_d = group_thr;
    m_screen_m1_d         = m1;
}

template <typename T,
          int kBlockThreads,
          bool kScreen,
          bool kBlockNoise,
          typename C,
          bool kGlobalRing>
void FDMTBoxcarCudaEngine::configure_variants() {
    configure_kernel(kernel_fdmt_boxcar<T, kBlockThreads, kScreen, kBlockNoise,
                                        C, false, kGlobalRing>);
    if (m_bright_q != 0) {
        configure_kernel(kernel_fdmt_boxcar<T, kBlockThreads, kScreen,
                                            kBlockNoise, C, true, kGlobalRing>);
    }
}

template <typename T, int kBlockThreads>
void FDMTBoxcarCudaEngine::configure_shared_block(SizeType limit) {
    constexpr SizeType kBlockTile = kBlockThreads * kItems;
    const SizeType ring           = std::bit_ceil(m_width + kBlockTile);
    const SizeType fixed =
        sizeof(BoxcarShared<T, kBlockThreads>) + (ring * sizeof(T));
    const SizeType tables = m_filters * (sizeof(T) + sizeof(uint32_t));
    m_block_threads       = kBlockThreads;
    m_mask                = static_cast<uint32_t>(ring - 1);
    m_shared_tables = m_filters <= kSharedFilters && fixed + tables <= limit;
    m_smem          = fixed + (m_shared_tables ? tables : 0);
    // The width screen's range-minimum buffer (float rows): one entry
    // per prefix of a tile and its lookback.
    m_screen_offset = 0;
    if (!m_screen_groups_d.empty()) {
        m_smem          = (m_smem + 15) / 16 * 16;
        m_screen_offset = m_smem;
        m_smem += ring * sizeof(T);
    }
    // Group screen bounds of per-call mask tables, when they fit (else
    // those rows take the exact bookkeeping on every tile).
    m_group_offset = 0;
    if (!m_screen_groups_d.empty() && std::is_floating_point_v<T> &&
        m_smem + (m_screen_groups_d.size() * sizeof(float)) <= limit) {
        m_group_offset = m_smem;
        m_smem += m_screen_groups_d.size() * sizeof(float);
    }
    if (m_smem > limit) {
        throw std::invalid_argument(fmt::format(
            "FDMTBoxcar: max_boxcar {} needs {} B of shared memory per "
            "block; the device allows {} B",
            m_width, m_smem, limit));
    }
    if (m_screen_groups_d.empty()) {
        configure_variants<T, kBlockThreads, false, false, T>();
    } else {
        configure_variants<T, kBlockThreads, true, false, T>();
    }
    if constexpr (std::is_integral_v<T>) {
        if (m_screen_groups_d.empty()) {
            configure_variants<T, kBlockThreads, false, false, uint32_t>();
        } else {
            configure_variants<T, kBlockThreads, true, false, uint32_t>();
        }
    } else {
        if (m_screen_groups_d.empty()) {
            configure_variants<T, kBlockThreads, false, true, T>();
        } else {
            configure_variants<T, kBlockThreads, true, true, T>();
        }
    }
}

template <typename T>
void FDMTBoxcarCudaEngine::configure_global(SizeType limit) {
    if (!m_screen_groups_d.empty()) {
        m_screen_groups_d.clear();
        m_screen_thresholds_d.clear();
        m_screen_m1_d.clear();
        m_screen_groups_d.shrink_to_fit();
        m_screen_thresholds_d.shrink_to_fit();
        m_screen_m1_d.shrink_to_fit();
        m_screen_dropped = true;
    }
    const SizeType ring =
        std::bit_ceil(m_width + SizeType{kGlobalBlock * kItems});
    m_global_ring   = true;
    m_block_threads = kGlobalBlock;
    m_mask          = static_cast<uint32_t>(ring - 1);
    m_shared_tables = false;
    m_screen_offset = 0;
    m_group_offset  = 0;
    m_smem          = sizeof(BoxcarShared<T, kGlobalBlock>);
    if (m_smem > limit) {
        throw std::invalid_argument(fmt::format(
            "FDMTBoxcar: the search needs {} B of shared memory per "
            "block; the device allows {} B",
            m_smem, limit));
    }
    SizeType tasks = 1;
    for (const auto& g : m_level_groups) {
        tasks = std::max(tasks, g.row_count * m_beams);
    }
    const SizeType ring_words = ring * sizeof(T) / sizeof(uint64_t);
    m_ring_batch              = std::clamp<SizeType>(
        kGlobalRingBytes / (ring_words * sizeof(uint64_t)), 1, tasks);
    m_ring_d.resize(m_ring_batch * ring_words);
    logging::debug("FDMTBoxcar: max_boxcar {}: prefix rings in device "
                   "memory ({} blocks per batch, {} B)",
                   m_width, m_ring_batch, m_ring_d.size() * sizeof(uint64_t));
    configure_variants<T, kGlobalBlock, false, false, T, true>();
    if constexpr (std::is_integral_v<T>) {
        configure_variants<T, kGlobalBlock, false, false, uint32_t, true>();
    } else {
        configure_variants<T, kGlobalBlock, false, true, T, true>();
    }
}

template <typename T>
SizeType FDMTBoxcarCudaEngine::shared_need(int threads, bool screen) const {
    const auto need = [&]<int kBlockThreads>() {
        const SizeType ring =
            std::bit_ceil(m_width + SizeType{kBlockThreads * kItems});
        SizeType bytes =
            sizeof(BoxcarShared<T, kBlockThreads>) + (ring * sizeof(T));
        if (screen) {
            bytes = ((bytes + 15) / 16 * 16) + (ring * sizeof(T));
        }
        return bytes;
    };
    return threads == 64    ? need.template operator()<64>()
           : threads == 128 ? need.template operator()<128>()
                            : need.template operator()<256>();
}

template <typename T> void FDMTBoxcarCudaEngine::configure_shared() {
    int optin = 0;
    gpu_utils::check_gpu_call(
        cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                               m_device),
        "FDMTBoxcar: shared memory limit");
    const auto limit =
        m_shared_limit != 0
            ? std::min(m_shared_limit, static_cast<SizeType>(optin))
            : static_cast<SizeType>(optin);
    const SizeType eff_chunk =
        std::min(m_driver->chunk_nsamps(), m_plan.get_nsamps());
    int threads = eff_chunk <= 64 * kItems    ? 64
                  : eff_chunk <= 128 * kItems ? 128
                                              : 256;
    if (m_width > kFastMaxWidth) {
        configure_global<T>(limit);
        return;
    }
    // Wide boxes on small-shared-memory GPUs: when the block's prefix
    // ring does not fit, drop the width screen (its second ring; the
    // screen only skips work, candidates are the same), then take
    // smaller blocks (shorter tile and ring, smaller run state).
    // Configurations that fit are unchanged.
    if (!m_screen_groups_d.empty() && shared_need<T>(threads, true) > limit) {
        m_screen_groups_d.clear();
        m_screen_thresholds_d.clear();
        m_screen_m1_d.clear();
        m_screen_groups_d.shrink_to_fit();
        m_screen_thresholds_d.shrink_to_fit();
        m_screen_m1_d.shrink_to_fit();
        m_screen_dropped = true;
    }
    const bool screen = !m_screen_groups_d.empty();
    while (threads > 64 && shared_need<T>(threads, screen) > limit) {
        threads /= 2;
        m_block_shrunk = true;
    }
    if (shared_need<T>(threads, screen) > limit) {
        m_block_shrunk = false;
        configure_global<T>(limit);
        return;
    }
    if (m_screen_dropped || m_block_shrunk) {
        logging::debug("FDMTBoxcar: max_boxcar {} in {} B of shared "
                       "memory: {}-thread blocks{}",
                       m_width, limit, threads,
                       m_screen_dropped ? ", no width screen" : "");
    }
    if (threads == 64) {
        configure_shared_block<T, 64>(limit);
    } else if (threads == 128) {
        configure_shared_block<T, 128>(limit);
    } else {
        configure_shared_block<T, 256>(limit);
    }
}

template <typename T,
          int kBlockThreads,
          bool kScreen,
          bool kBlockNoise,
          typename C,
          bool kKeep,
          bool kGlobalRing>
void FDMTBoxcarCudaEngine::launch_score_args(const FDMTSearchInput& stopped,
                                             const LevelGroup& g,
                                             dim3 grid,
                                             dim3 block,
                                             C* carry,
                                             cudaStream_t stream,
                                             SizeType task_base) {
    kernel_fdmt_boxcar<T, kBlockThreads, kScreen, kBlockNoise, C, kKeep,
                       kGlobalRing><<<grid, block, m_smem, stream>>>(
        stopped, raw(m_rows_d), g.row_begin, g.row_count, m_rows.size(),
        raw(m_widths_d), m_filters, m_width, raw(m_thresholds_d),
        raw(m_scales_d), carry, raw(m_peaks_d), raw(m_open_d), m_peak_count,
        raw(m_candidates_d), raw(m_counts_d), raw(m_slots_d), raw(m_per_dm_d),
        m_config.max_candidates, m_config.max_cands_per_dm, m_config.peak_mode,
        m_config.skip_cold_prefix, g.level == m_plan.get_niters(),
        raw(m_integer_thresholds), raw(m_integer_means), m_integer_unit, m_mask,
        m_shared_tables, m_config.use_trapezoid, raw(m_screen_groups_d),
        static_cast<int>(m_screen_groups_d.size()), raw(m_screen_thresholds_d),
        raw(m_screen_m1_d), m_screen_offset,
        kBlockNoise ? m_block_tables : BlockTables{}, m_group_offset,
        raw(m_bright_d), m_bright_q,
        kGlobalRing ? reinterpret_cast<T*>(raw(m_ring_d)) : nullptr, task_base,
        m_config.screen_mode == BoxcarScreenMode::kNone);
}

template <typename T,
          int kBlockThreads,
          bool kScreen,
          bool kBlockNoise,
          typename C>
void FDMTBoxcarCudaEngine::launch_score_kernel(const FDMTSearchInput& input,
                                               const LevelGroup& g,
                                               dim3 grid,
                                               dim3 block,
                                               C* carry) {
    const auto stream       = static_cast<cudaStream_t>(input.stream.native);
    FDMTSearchInput stopped = input;
    stopped.search_stop     = m_search_stop;
    if (m_bright_q != 0) {
        launch_score_args<T, kBlockThreads, kScreen, kBlockNoise, C, true>(
            stopped, g, grid, block, carry, stream);
    } else {
        launch_score_args<T, kBlockThreads, kScreen, kBlockNoise, C, false>(
            stopped, g, grid, block, carry, stream);
    }
}

template <typename T, int kBlockThreads, typename C>
void FDMTBoxcarCudaEngine::launch_score_block(const FDMTSearchInput& input,
                                              const LevelGroup& g,
                                              dim3 grid,
                                              dim3 block,
                                              C* carry) {
    // Per-call mask tables take their own kernel instance (float rows only).
    if constexpr (!std::is_integral_v<T>) {
        if (m_block_tables.count > 0) {
            if (m_screen_groups_d.empty()) {
                launch_score_kernel<T, kBlockThreads, false, true, C>(
                    input, g, grid, block, carry);
            } else {
                launch_score_kernel<T, kBlockThreads, true, true, C>(
                    input, g, grid, block, carry);
            }
            return;
        }
    }
    if (m_screen_groups_d.empty()) {
        launch_score_kernel<T, kBlockThreads, false, false, C>(input, g, grid,
                                                               block, carry);
    } else {
        launch_score_kernel<T, kBlockThreads, true, false, C>(input, g, grid,
                                                              block, carry);
    }
}

template <typename T, bool kBlockNoise, typename C>
void FDMTBoxcarCudaEngine::launch_score_global(const FDMTSearchInput& input,
                                               const LevelGroup& g,
                                               dim3 grid,
                                               dim3 block,
                                               C* carry) {
    const auto stream       = static_cast<cudaStream_t>(input.stream.native);
    FDMTSearchInput stopped = input;
    stopped.search_stop     = m_search_stop;
    const SizeType tasks    = grid.x;
    for (SizeType base = 0; base < tasks; base += m_ring_batch) {
        const dim3 batch(
            static_cast<unsigned>(std::min(m_ring_batch, tasks - base)));
        if (m_bright_q != 0) {
            launch_score_args<T, kGlobalBlock, false, kBlockNoise, C, true,
                              true>(stopped, g, batch, block, carry, stream,
                                    base);
        } else {
            launch_score_args<T, kGlobalBlock, false, kBlockNoise, C, false,
                              true>(stopped, g, batch, block, carry, stream,
                                    base);
        }
    }
}

template <typename T, typename C>
void FDMTBoxcarCudaEngine::launch_score(const FDMTSearchInput& input,
                                        const LevelGroup& g,
                                        dim3 grid,
                                        dim3 block,
                                        C* carry) {
    if (m_global_ring) {
        if constexpr (!std::is_integral_v<T>) {
            if (m_block_tables.count > 0) {
                launch_score_global<T, true, C>(input, g, grid, block, carry);
                return;
            }
        }
        launch_score_global<T, false, C>(input, g, grid, block, carry);
        return;
    }
    if (m_block_threads == 64) {
        launch_score_block<T, 64, C>(input, g, grid, block, carry);
    } else if (m_block_threads == 128) {
        launch_score_block<T, 128, C>(input, g, grid, block, carry);
    } else {
        launch_score_block<T, 256, C>(input, g, grid, block, carry);
    }
}

template void FDMTBoxcarCudaEngine::configure_shared<float>();
template void FDMTBoxcarCudaEngine::configure_shared<uint64_t>();
template void FDMTBoxcarCudaEngine::launch_score<float, float>(
    const FDMTSearchInput&, const LevelGroup&, dim3, dim3, float*);
template void FDMTBoxcarCudaEngine::launch_score<uint64_t, uint64_t>(
    const FDMTSearchInput&, const LevelGroup&, dim3, dim3, uint64_t*);
template void FDMTBoxcarCudaEngine::launch_score<uint64_t, uint32_t>(
    const FDMTSearchInput&, const LevelGroup&, dim3, dim3, uint32_t*);

} // namespace dmt::algorithms::detail::fdmt_boxcar_cuda
