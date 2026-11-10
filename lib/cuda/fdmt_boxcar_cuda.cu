#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <thrust/device_vector.h>

#include "dmt/bit_pack_utils.hpp"
#include "dmt/boxcar.hpp"
#include "dmt/boxcar_runs.hpp"
#include "dmt/fdmt_int_tree.hpp"
#include "dmt/fdmt_variance.hpp"
#include "dmt/gpu_compat.cuh"
#include "dmt/gpu_utils.cuh"
#include "dmt/host_staging.cuh"

namespace dmt::algorithms::detail {
namespace {

constexpr int kItems = 4;
constexpr int kWarp = gpu_compat::kLogicalWarp; // a 64-lane wavefront holds two
// Width and threshold tables are staged in shared memory up to this count;
// longer tables are read through the cache.
constexpr SizeType kSharedFilters = 512;

// Fixed part of the block's shared memory. The dynamic part follows it:
// the prefix ring (a power of two of at least max_width + kBlockTile entries),
// then optionally the thresholds and widths of the row.
template <typename T, int kBlockThreads> struct BoxcarShared {
    BoxcarRunSummary runs[kBlockThreads];
    T warp_totals[kBlockThreads / kWarp];
};

template <typename T> DMT_D T shfl_up(T value, int delta) {
    return gpu_compat::shfl_up(value, static_cast<unsigned>(delta));
}

DMT_D uint64_t increment(uint64_t* count) {
    return atomicAdd(reinterpret_cast<unsigned long long*>(count), 1ULL);
}

struct CandidateWriter {
    RawCandidate* candidates;
    CandidateCounts* counts;
    uint64_t* slots;
    uint64_t* per_dm;
    const uint32_t* widths;
    RawCandidate prototype;
    SizeType capacity;
    SizeType dm_cap;
    const double* means = nullptr;
    const float* scales = nullptr;
    double unit         = 1;
    bool integer        = false;

    DMT_D void operator()(const BoxcarPeak& peak) const {
        increment(&counts->total_detected);
        if (dm_cap != 0 && increment(per_dm) >= dm_cap) {
            increment(&counts->dropped_per_dm);
            return;
        }
        const uint64_t slot = increment(slots);
        if (slot >= capacity) {
            increment(&counts->dropped_global);
            return;
        }
        RawCandidate candidate = prototype;
        candidate.peak_sample  = peak.peak;
        candidate.begin_sample = peak.begin;
        candidate.end_sample   = peak.end;
        candidate.snr =
            integer ? static_cast<float>((static_cast<double>(peak.raw_sum) -
                                          means[peak.filter]) *
                                         (scales[peak.filter] * unit))
                    : peak.snr;
        candidate.filter_idx = peak.filter;
        candidate.width      = widths[peak.filter];
        candidates[slot]     = candidate;
        increment(&counts->emitted_count);
    }
};

// Ordered block prefix of each thread's four consecutive input samples,
// written to ring[position + 1 .. position + kBlockTile]: a warp shuffle scan
// and one scan of the warp totals. The ring is rebased after every tile, so
// float32 prefixes stay near the magnitude of one tile and the lookback.
template <typename T, int kBlockThreads>
DMT_D void prefix_tile(BoxcarShared<T, kBlockThreads>& shared,
                       T* ring,
                       uint32_t mask,
                       const T (&x)[kItems],
                       uint32_t position,
                       T base) {
    constexpr int kBlockWarps = kBlockThreads / kWarp;
    const int tid             = static_cast<int>(threadIdx.x);
    const int lane            = tid % kWarp;
    const int warp            = tid / kWarp;
    T total                   = 0;
#pragma unroll
    for (int j = 0; j < kItems; ++j) {
        total += x[j];
    }
    T inclusive = total;
#pragma unroll
    for (int offset = 1; offset < kWarp; offset *= 2) {
        const T left = shfl_up(inclusive, offset);
        if (lane >= offset) {
            inclusive += left;
        }
    }
    if (lane == kWarp - 1) {
        shared.warp_totals[warp] = inclusive;
    }
    __syncthreads();
    if (warp == 0) {
        T w = lane < kBlockWarps ? shared.warp_totals[lane] : T{0};
#pragma unroll
        for (int offset = 1; offset < kBlockWarps; offset *= 2) {
            const T left = shfl_up(w, offset);
            if (lane >= offset) {
                w += left;
            }
        }
        if (lane < kBlockWarps) {
            shared.warp_totals[lane] = w;
        }
    }
    __syncthreads();
    T running = base + (inclusive - total) +
                (warp > 0 ? shared.warp_totals[warp - 1] : T{0});
#pragma unroll
    for (int j = 0; j < kItems; ++j) {
        running += x[j];
        ring[(position + tid * kItems + j + 1) & mask] = running;
    }
    __syncthreads();
}

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
                               const float* group_thresholds,
                               T* mn,
                               uint64_t start,
                               uint64_t warmup) {
    const int tid       = static_cast<int>(threadIdx.x);
    const int len       = static_cast<int>(lookback) + n + 1;
    const uint32_t lin0 = position - lookback; // ring index of mn[0]
    for (int j = tid; j < len; j += kBlockThreads) {
        mn[j] = ring[(lin0 + static_cast<uint32_t>(j)) & mask];
    }
    T current[kItems];
#pragma unroll
    for (int k = 0; k < kItems; ++k) {
        current[k] = ring[(position + 1U +
                           static_cast<uint32_t>(k * kBlockThreads + tid)) &
                          mask];
    }
    __syncthreads();
    // One width checked exactly (as detect_tile), from the ring: mn holds
    // the prefixes only until the first level is raised.
    const auto exact = [&](uint32_t f) {
        const uint32_t width = widths[f];
        const T threshold    = thresholds[f];
        bool hit             = false;
#pragma unroll
        for (int k = 0; k < kItems; ++k) {
            const int s = k * kBlockThreads + tid;
            if (!kFull && s >= n) {
                break;
            }
            const uint32_t idx = position + 1U + static_cast<uint32_t>(s);
            bool h = current[k] - ring[(idx - width) & mask] >= threshold;
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
        if (static_cast<int>(grp.level) < level ||
            (level == 0 && grp.level == 0)) {
            // A single-run group (no bound to gain), or the table is past
            // the group's level (sparse widths; never with sorted groups).
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

// Scan trailing-run summaries to give each worker the state of every
// preceding sample. Each worker then emits only excursions ending within
// its own four samples. The last worker saves the open trailing excursion.
template <typename T, int kBlockThreads>
DMT_D void reduce_runs(BoxcarShared<T, kBlockThreads>& shared,
                       const bool (&above)[kItems],
                       const float (&snr)[kItems],
                       const uint32_t (&filters)[kItems],
                       uint64_t start,
                       int n,
                       BoxcarPeak* persistent,
                       const CandidateWriter& writer,
                       const uint64_t (&raw_sum)[kItems],
                       bool integer) {
    const int tid   = static_cast<int>(threadIdx.x);
    const int first = tid * kItems;
    BoxcarRunSummary summary;
    for (int j = 0; j < kItems && first + j < n; ++j) {
        if (!above[j]) {
            summary.all           = false;
            summary.trailing.open = false;
        } else {
            const auto ignore = [] DMT_D(const BoxcarPeak&) {};
            if (integer) {
                boxcar_integer_peak_step(summary.trailing, true, raw_sum[j],
                                         start + first + j, filters[j], ignore);
            } else {
                boxcar_peak_step(summary.trailing, true, snr[j],
                                 start + first + j, filters[j], ignore);
            }
        }
    }
    const BoxcarRunSummary carried{*persistent, true};
    shared.runs[tid] = summary;
    __syncthreads();
    for (int offset = 1; offset < kBlockThreads; offset *= 2) {
        const auto left =
            tid >= offset ? shared.runs[tid - offset] : BoxcarRunSummary{};
        __syncthreads();
        if (tid >= offset) {
            summary = boxcar_join(left, summary, integer);
        }
        shared.runs[tid] = summary;
        __syncthreads();
    }
    BoxcarPeak incoming =
        tid == 0 ? carried.trailing
                 : boxcar_join(carried, shared.runs[tid - 1], integer).trailing;
    for (int j = 0; j < kItems && first + j < n; ++j) {
        if (integer) {
            boxcar_integer_peak_step(incoming, above[j], raw_sum[j],
                                     start + first + j, filters[j], writer);
        } else {
            boxcar_peak_step(incoming, above[j], snr[j], start + first + j,
                             filters[j], writer);
        }
    }
    // All threads have read persistent and shared summaries before writing
    // the state for the next width/tile.
    __syncthreads();
    if (tid == kBlockThreads - 1) {
        *persistent = boxcar_join(carried, summary, integer).trailing;
    }
    __syncthreads();
}

// Per-call kill masks (float rows): the tables of the (at most two) ring
// entries a call's endpoints can reach, older first. Per entry: its first
// stream sample, steady [0] and transition [1] thresholds and inverse
// sigmas (rows x filters; null: the constructor tables) and whether it has
// a transition (its first support + W - 1 endpoints per row use [1]).
// count 0: the constructor tables only.
struct BlockTables {
    uint64_t first[2]{0, 0};
    const float* thr[2][2]{{nullptr, nullptr}, {nullptr, nullptr}};
    const float* inv[2][2]{{nullptr, nullptr}, {nullptr, nullptr}};
    bool transition[2]{false, false};
    int count               = 0;
    const uint64_t* support = nullptr;
};

// The entry of stream sample `s` (the older one for samples before the
// newer's first).
DMT_D inline int block_entry(const BlockTables& bt, uint64_t s) {
    return bt.count == 2 && s >= bt.first[1] ? 1 : 0;
}

// Table (0 steady, 1 transition) of entry `e` for endpoint `s` and width w.
DMT_D inline int block_table(const BlockTables& bt,
                             int e,
                             uint64_t s,
                             uint32_t width,
                             uint64_t support) {
    return bt.transition[e] && s < bt.first[e] + support + width - 1 ? 1 : 0;
}

// kScreen: tiles are screened by width group (detect_tile_grouped*); a
// separate instantiation, so the plain kernel keeps its register budget.
// kBlockNoise: per-call mask tables are in use (float rows only). A
// separate instance, so the common unmasked launch carries none of their
// state (it cost ~160 B of stack per thread and 15-50% of the run time).
template <typename T,
          int kBlockThreads,
          bool kScreen,
          bool kBlockNoise = false,
          typename C       = T>
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
                       SizeType group_offset) {
    constexpr int kBlockTile = kBlockThreads * kItems;
    // Lagged rows (skewed-window subband search): this row's own samples.
    if (input.row_lag != nullptr) {
        FDMTSearchInput lagged;
        if (!lagged_row_input(input, rows[row_base + blockIdx.x % ndm].row,
                              lagged)) {
            return;
        }
        input = lagged;
    }
    if (!stop_row_input(input)) {
        return; // drain(): past the end of the stream
    }
    extern __shared__ __align__(16) unsigned char storage[];
    auto& shared = *reinterpret_cast<BoxcarShared<T, kBlockThreads>*>(storage);
    T* ring      = reinterpret_cast<T*>(storage + sizeof(shared));
    const SizeType task      = blockIdx.x;
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
    writer.means        = means ? means + row_index * nfilters : nullptr;
    writer.scales       = inv_sigma + row_index * nfilters;
    writer.unit         = unit;
    writer.integer      = std::is_integral_v<T> &&
                          mode != PeakFindingMode::kContiguousPeakAcrossWidths;
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
    // Row tables: raw thresholds in the units of the prefix ring.
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
    // Mixed tables without the shared row: every tile takes the exact
    // bookkeeping (no screen bound is stored).
    bool force_bookkeeping = mixed && !shared_tables;
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
                        writer(peak);
                    }
                }
                if (mode == PeakFindingMode::kContiguousPeakPerWidth &&
                    __syncthreads_or(f_hit || peaks[f].open) != 0) {
                    reduce_runs<T, kBlockThreads>(
                        shared, above, snr, filters, start, n, peaks + f,
                        writer, raw_sums, std::is_integral_v<T>);
                }
            }
            if (mode == PeakFindingMode::kContiguousPeakAcrossWidths) {
                bool above[kItems];
                for (int j = 0; j < kItems; ++j) {
                    above[j] = best[j] >= 0;
                }
                reduce_runs<T, kBlockThreads>(shared, above, best, best_filter,
                                              start, n, peaks, writer, raw_sums,
                                              false);
            }
            open = any_open();
        }
        // Recenter the retained prefix boundaries only when another tile
        // follows.
        if (t0 + kBlockTile < input.nsamps) {
            const uint32_t end = position + n;
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

__global__ void kernel_flush_boxcar(const BoxcarRow* rows,
                                    SizeType nrows,
                                    SizeType peak_count,
                                    BoxcarPeak* peaks,
                                    uint8_t* open,
                                    const uint32_t* widths,
                                    RawCandidate* candidates,
                                    CandidateCounts* counts,
                                    uint64_t* slots,
                                    uint64_t* per_dm,
                                    SizeType capacity,
                                    SizeType dm_cap,
                                    SizeType tasks,
                                    const double* means,
                                    const float* scales,
                                    SizeType nfilters,
                                    double unit,
                                    bool integer) {
    const SizeType index =
        static_cast<SizeType>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= tasks) {
        return;
    }
    const SizeType row_index = index % nrows;
    auto prototype           = rows[row_index].candidate;
    prototype.beam_idx       = static_cast<uint32_t>(index / nrows);
    CandidateWriter writer{candidates, counts,    slots,    per_dm + index,
                           widths,     prototype, capacity, dm_cap};
    writer.means   = means ? means + row_index * nfilters : nullptr;
    writer.scales  = scales + row_index * nfilters;
    writer.unit    = unit;
    writer.integer = integer;
    for (SizeType f = 0; f < peak_count; ++f) {
        auto& peak = peaks[index * peak_count + f];
        if (peak.open) {
            writer(peak);
            peak.open = false;
        }
    }
    open[index] = 0;
}

// Candidate frontier: the smallest begin sample of the open excursions,
// min-reduced into *oldest (preset to UINT64_MAX). Only rows flagged open
// are read.
__global__ void kernel_open_begin(const BoxcarPeak* peaks,
                                  const uint8_t* open,
                                  SizeType tasks,
                                  SizeType peak_count,
                                  uint64_t* oldest) {
    uint64_t local = UINT64_MAX;
    for (SizeType i = blockIdx.x * blockDim.x + threadIdx.x; i < tasks;
         i += static_cast<SizeType>(gridDim.x) * blockDim.x) {
        if (open[i] == 0) {
            continue;
        }
        for (SizeType f = 0; f < peak_count; ++f) {
            const BoxcarPeak& p = peaks[i * peak_count + f];
            if (p.open && p.begin < local) {
                local = p.begin;
            }
        }
    }
    if (local != UINT64_MAX) {
        atomicMin(reinterpret_cast<unsigned long long*>(oldest),
                  static_cast<unsigned long long>(local));
    }
}

template <typename T> T* raw(thrust::device_vector<T>& v) {
    return thrust::raw_pointer_cast(v.data());
}

class FDMTBoxcarCudaEngine final : public FDMTBoxcarEngine {
public:
    FDMTBoxcarCudaEngine(const plans::FDMTPlan& plan,
                         const BoxcarConfig& config,
                         const std::vector<plans::BoxcarPlanePlan>& planes,
                         const FDMTEngineConfig& cfg,
                         std::unique_ptr<BoxcarDriver> driver,
                         std::vector<std::pair<SizeType, SizeType>> kill_runs)
        : m_plan(plan),
          m_config(config),
          m_planes(planes),
          m_beams(cfg.nbeams),
          m_device(cfg.exec.device),
          m_rows(make_boxcar_rows(plan, planes)),
          m_width(planes.front().widths.back()),
          m_filters(planes.front().widths.size()),
          m_peak_count(config.peak_mode ==
                               PeakFindingMode::kContiguousPeakPerWidth
                           ? m_filters
                           : 1) {
        gpu_utils::set_device(m_device);
        if (m_rows.size() > SIZE_MAX / m_beams / m_width ||
            m_rows.size() > SIZE_MAX / m_beams / m_peak_count ||
            m_rows.size() * m_beams > static_cast<SizeType>(INT32_MAX)) {
            throw std::invalid_argument(
                "FDMTBoxcar: GPU search extent exceeds limits");
        }
        m_driver    = std::move(driver);
        m_kill_runs = std::move(kill_runs);
        if (!m_kill_runs.empty()) {
            // The staging block the killed rows are zeroed in.
            if (config.input_nbits == 32) {
                m_input_d.resize(m_beams * plan.get_nchans() *
                                 plan.get_nsamps());
            } else {
                m_packed_input.resize(
                    m_beams * plan.get_nchans() *
                    bit_pack_utils::packed_row_bytes(plan.get_nsamps(),
                                                     config.input_nbits));
            }
        }
        std::vector<float> thresholds, scales;
        SizeType row_offset = 0;
        for (const auto& p : planes) {
            thresholds.insert(thresholds.end(), p.raw_thresh_lut.begin(),
                              p.raw_thresh_lut.end());
            scales.insert(scales.end(), p.inv_sigma_lut.begin(),
                          p.inv_sigma_lut.end());
            const SizeType count = p.dm_grid.size();
            if (count > 0) {
                if (m_level_groups.empty() ||
                    m_level_groups.back().level != p.level) {
                    m_level_groups.push_back({.level     = p.level,
                                              .row_begin = row_offset,
                                              .row_count = count});
                } else {
                    m_level_groups.back().row_count += count;
                }
            }
            row_offset += count;
        }
        m_rows_d             = m_rows;
        m_widths_d           = planes.front().widths;
        m_thresholds_d       = thresholds;
        m_scales_d           = scales;
        const SizeType tasks = m_rows.size() * m_beams;
        m_integer_unit =
            cfg.use_box_smearing
                ? std::ldexp(1.0,
                             -static_cast<int>(plan.get_box_fraction_bits()))
                : 1.0;
        if (!config.use_trapezoid) {
            const auto groups =
                boxcar_screen_groups(planes.front().widths, false);
            if (!groups.empty()) {
                // Per row, each group's smallest threshold (float rows) or
                // smallest cutoff - mean, rounded down (packed rows), and the
                // packed rows' mean per sample (from the width-1 mean).
                const bool packed = config.input_nbits != 32;
                std::vector<float> group_thr;
                std::vector<float> m1;
                group_thr.reserve(m_rows.size() * groups.size());
                for (const auto& p : planes) {
                    for (SizeType d = 0; d < p.dm_grid.size(); ++d) {
                        const SizeType base = d * m_filters;
                        for (const auto& g : groups) {
                            double th = std::numeric_limits<double>::infinity();
                            for (uint32_t f = g.f_lo; f < g.f_hi; ++f) {
                                th = std::min(
                                    th,
                                    packed
                                        ? static_cast<double>(
                                              p.integer_thresh_lut[base + f]) -
                                              p.integer_mean_lut[base + f]
                                        : static_cast<double>(
                                              p.raw_thresh_lut[base + f]));
                            }
                            auto thf = static_cast<float>(th);
                            if (static_cast<double>(thf) > th) {
                                thf = std::nextafter(
                                    thf,
                                    -std::numeric_limits<float>::infinity());
                            }
                            group_thr.push_back(thf);
                        }
                        if (packed) {
                            // The mean per sample comes from widths[0], a
                            // plain boxcar (no screen with use_trapezoid).
                            m1.push_back(static_cast<float>(
                                p.integer_mean_lut[base] /
                                static_cast<double>(p.widths[0])));
                        }
                    }
                }
                m_screen_groups_d     = groups;
                m_screen_thresholds_d = group_thr;
                m_screen_m1_d         = m1;
            }
        }
        if (config.input_nbits == 32) {
            m_carry_d.resize(tasks * m_width, 0);
            configure_shared<float>();
        } else {
            // uint32_t whenever every tree value fits (int_search_fits_u32),
            // as the FDMT engines' own state: half the carry's bytes.
            if (int_search_fits_u32(plan, cfg.use_box_smearing,
                                    config.input_nbits)) {
                m_integer_carry32.resize(tasks * m_width, 0);
            } else {
                m_integer_carry.resize(tasks * m_width, 0);
            }
            configure_shared<uint64_t>();
        }
        std::vector<double> means;
        std::vector<uint64_t> cutoffs;
        for (const auto& p : planes) {
            means.insert(means.end(), p.integer_mean_lut.begin(),
                         p.integer_mean_lut.end());
            cutoffs.insert(cutoffs.end(), p.integer_thresh_lut.begin(),
                           p.integer_thresh_lut.end());
        }
        m_integer_means      = means;
        m_integer_thresholds = cutoffs;
        m_peaks_d.resize(tasks * m_peak_count);
        m_open_d.resize(tasks, 0);
        m_per_dm_d.resize(tasks, 0);
        m_candidates_d.resize(config.max_candidates);
        m_counts_d.resize(1);
        m_slots_d.resize(1, 0);
        m_candidates_h.resize(config.max_candidates);
        m_oldest_d.resize(1, UINT64_MAX);
        m_oldest_h.reserve(1);
        *m_oldest_h.data() = UINT64_MAX;
    }
    ~FDMTBoxcarCudaEngine() override {
        // Destruction must not release search buffers while queued work
        // still references them, including on a nondefault stream.
        cudaSetDevice(m_device);
        m_work.wait();
        for (auto& slot : m_slots) {
            if (slot->copied != nullptr) {
                cudaEventDestroy(slot->copied);
            }
            if (slot->released != nullptr) {
                cudaEventDestroy(slot->released);
            }
        }
    }
    CandidateResultView execute(std::span<const float> waterfall) override {
        if (m_config.input_nbits != 32) {
            throw std::invalid_argument(
                "FDMTBoxcar: expected packed uint8 input");
        }
        validate_input(waterfall.size());
        gpu_utils::set_device(m_device);
        m_work.wait();
        if (m_input_d.size() != waterfall.size()) {
            m_input_d.resize(waterfall.size());
        }
        gpu_utils::check_gpu_call(cudaMemcpyAsync(
            raw(m_input_d), waterfall.data(), waterfall.size_bytes(),
            cudaMemcpyHostToDevice, nullptr));
        (void)execute(DeviceSpan<const float>(raw(m_input_d), waterfall.size(),
                                              {kGPUBackend, m_device}),
                      {});
        return copy_result({});
    }
    CandidateDeviceResultView execute(DeviceSpan<const float> waterfall,
                                      Stream queue) override {
        if (m_config.input_nbits != 32) {
            throw std::invalid_argument(
                "FDMTBoxcar: expected packed uint8 input");
        }
        validate_input(waterfall.size());
        const auto stream = static_cast<cudaStream_t>(queue.native);
        commit_block_noise(stream);
        begin_result(m_plan.get_nsamps(), queue);
        m_driver->search(masked(waterfall, m_input_d, queue), *this, queue);
        m_position += m_plan.get_nsamps();
        release_slots(stream);
        update_open_begin(stream);
        m_work.mark(stream);
        return device_view();
    }
    CandidateResultView execute(std::span<const uint8_t> waterfall) override {
        validate_packed(waterfall.size());
        gpu_utils::set_device(m_device);
        m_work.wait();
        if (m_packed_input.size() != waterfall.size()) {
            m_packed_input.resize(waterfall.size());
        }
        gpu_utils::check_gpu_call(cudaMemcpyAsync(
            raw(m_packed_input), waterfall.data(), waterfall.size_bytes(),
            cudaMemcpyHostToDevice, nullptr));
        (void)execute(DeviceSpan<const uint8_t>(raw(m_packed_input),
                                                waterfall.size(),
                                                {kGPUBackend, m_device}),
                      {});
        return copy_result({});
    }
    CandidateDeviceResultView execute(DeviceSpan<const uint8_t> waterfall,
                                      Stream queue) override {
        validate_packed(waterfall.size());
        begin_result(m_plan.get_nsamps(), queue);
        m_driver->search(masked(waterfall, m_packed_input, queue),
                         m_config.input_nbits, *this, queue);
        m_position += m_plan.get_nsamps();
        update_open_begin(static_cast<cudaStream_t>(queue.native));
        m_work.mark(static_cast<cudaStream_t>(queue.native));
        return device_view();
    }
    CandidateResultView flush() override {
        gpu_utils::set_device(m_device);
        m_work.wait();
        (void)flush_device({});
        return copy_result({});
    }
    CandidateDeviceResultView flush_device(Stream queue) override {
        begin_result(0, queue);
        const auto tasks  = m_rows.size() * m_beams;
        const auto stream = static_cast<cudaStream_t>(queue.native);
        kernel_flush_boxcar<<<(tasks + 127) / 128, 128, 0, stream>>>(
            raw(m_rows_d), m_rows.size(), m_peak_count, raw(m_peaks_d),
            raw(m_open_d), raw(m_widths_d), raw(m_candidates_d),
            raw(m_counts_d), raw(m_slots_d), raw(m_per_dm_d),
            m_config.max_candidates, m_config.max_cands_per_dm, tasks,
            raw(m_integer_means), raw(m_scales_d), m_filters, m_integer_unit,
            m_config.input_nbits != 32 &&
                m_config.peak_mode !=
                    PeakFindingMode::kContiguousPeakAcrossWidths);
        gpu_utils::check_last_gpu_error("FDMTBoxcar flush failed");
        update_open_begin(stream);
        m_work.mark(stream);
        return device_view();
    }
    CandidateResultView copy_result(Stream queue) override {
        gpu_utils::set_device(m_device);
        const auto stream = static_cast<cudaStream_t>(queue.native);
        m_work.order(stream);
        gpu_utils::check_gpu_call(cudaMemcpyAsync(
            &m_counts_h, raw(m_counts_d), sizeof(CandidateCounts),
            cudaMemcpyDeviceToHost, stream));
        gpu_utils::check_gpu_call(cudaStreamSynchronize(stream));
        if (m_counts_h.emitted_count > 0) {
            gpu_utils::check_gpu_call(
                cudaMemcpyAsync(m_candidates_h.data(), raw(m_candidates_d),
                                m_counts_h.emitted_count * sizeof(RawCandidate),
                                cudaMemcpyDeviceToHost, stream));
            gpu_utils::check_gpu_call(cudaStreamSynchronize(stream));
        }
        return {m_counts_h,
                {m_candidates_h.data(),
                 static_cast<SizeType>(m_counts_h.emitted_count)},
                m_result_begin,
                m_result_samples};
    }
    void set_search_stop(uint64_t stop) noexcept override {
        m_search_stop = stop;
    }
    [[nodiscard]] uint64_t stream_position() const noexcept override {
        return m_position;
    }
    [[nodiscard]] uint64_t search_latency() const noexcept override {
        return m_driver->search_latency();
    }
    [[nodiscard]] uint64_t open_begin_min() const noexcept override {
        return *m_oldest_h.data();
    }
    [[nodiscard]] SizeType block_noise_bytes() const noexcept override {
        // Staging (unless allocated), the row supports, and the table slots
        // on the device and in pinned host memory.
        const SizeType full =
            m_beams * m_plan.get_nchans() * m_plan.get_nsamps();
        const SizeType staging = m_input_d.size() < full
                                     ? (full - m_input_d.size()) * sizeof(float)
                                     : 0;
        return staging + (m_rows.size() * sizeof(uint64_t)) +
               (slot_count() * 4 * table_floats() * sizeof(float));
    }
    void enable_block_noise(
        std::shared_ptr<const std::vector<uint64_t>> support) override {
        if (m_config.input_nbits != 32) {
            throw std::logic_error(
                "FDMTBoxcar (gpu): per-call kill masks need float input");
        }
        if (ring_size() > 2) {
            // A call's endpoints reach at most two blocks only when the
            // search latency is at most one block.
            throw std::invalid_argument(fmt::format(
                "FDMTBoxcar (gpu): per-call kill masks need a search latency "
                "of at most one block ({} > {} samples)",
                search_latency(), m_plan.get_nsamps()));
        }
        gpu_utils::set_device(m_device);
        const SizeType full =
            m_beams * m_plan.get_nchans() * m_plan.get_nsamps();
        if (m_input_d.size() < full) {
            m_work.wait(); // the staging buffer may still be read
            m_input_d.resize(full);
        }
        m_support_d = *support;
        m_slots.clear();
        for (SizeType i = 0; i < slot_count(); ++i) {
            auto slot = std::make_unique<TableSlot>();
            slot->device.resize(2 * table_floats());
            slot->host.reserve(2 * table_floats());
            gpu_utils::check_gpu_call(
                cudaEventCreateWithFlags(&slot->copied, cudaEventDisableTiming),
                "FDMTBoxcar: event creation failed");
            gpu_utils::check_gpu_call(
                cudaEventCreateWithFlags(&slot->released,
                                         cudaEventDisableTiming),
                "FDMTBoxcar: event creation failed");
            m_slots.push_back(std::move(slot));
        }
        m_ring.reserve(3);
        m_noise_on = true;
        reset_block_noise();
    }
    void push_block_noise(BlockNoise entry) override {
        m_pending = std::move(entry);
    }
    void drop_block_noise() noexcept override { m_pending.reset(); }
    void reset_history(Stream queue) override {
        gpu_utils::set_device(m_device);
        const auto stream = static_cast<cudaStream_t>(queue.native);
        if (m_noise_on) {
            reset_block_noise();
        }
        m_search_stop = UINT64_MAX;
        m_driver->reset(queue);
        gpu_utils::check_gpu_call(cudaMemsetAsync(
            raw(m_carry_d), 0, m_carry_d.size() * sizeof(float), stream));
        gpu_utils::check_gpu_call(cudaMemsetAsync(
            raw(m_peaks_d), 0, m_peaks_d.size() * sizeof(BoxcarPeak), stream));
        gpu_utils::check_gpu_call(
            cudaMemsetAsync(raw(m_open_d), 0, m_open_d.size(), stream));
        if (!m_integer_carry.empty()) {
            gpu_utils::check_gpu_call(cudaMemsetAsync(
                raw(m_integer_carry), 0,
                m_integer_carry.size() * sizeof(uint64_t), stream));
        }
        if (!m_integer_carry32.empty()) {
            gpu_utils::check_gpu_call(cudaMemsetAsync(
                raw(m_integer_carry32), 0,
                m_integer_carry32.size() * sizeof(uint32_t), stream));
        }
        m_position = 0;
        begin_result(0, queue);
        update_open_begin(stream);
        m_work.mark(stream);
    }
    void consume(SizeType level, const FDMTSearchInput& input) override {
        for (const auto& g : m_level_groups) {
            if (g.level != level) {
                continue;
            }
            const SizeType beams =
                std::min(input.beam_count, m_beams - input.beam_begin);
            const dim3 grid(g.row_count * beams);
            const dim3 block(m_block_threads);
            gpu_utils::check_kernel_launch_params(grid, block);
            if (m_config.input_nbits == 32) {
                launch_score<float>(input, g, grid, block, raw(m_carry_d));
            } else {
                if (!m_integer_carry32.empty()) {
                    launch_score<uint64_t, uint32_t>(input, g, grid, block,
                                                     raw(m_integer_carry32));
                } else {
                    launch_score<uint64_t>(input, g, grid, block,
                                           raw(m_integer_carry));
                }
            }
            gpu_utils::check_last_gpu_error("FDMTBoxcar scoring failed");
        }
    }
    FDMTBoxcarMemoryUsage memory_usage() const noexcept override {
        const auto fdmt    = m_driver->memory();
        SizeType host_plan = m_rows.size() * sizeof(BoxcarRow) +
                             m_level_groups.size() * sizeof(LevelGroup);
        for (const auto& p : m_planes) {
            host_plan +=
                sizeof(p) + p.widths.size() * sizeof(uint32_t) +
                p.dm_grid.size() * sizeof(float) +
                (p.warmup_samples.size() + p.integer_thresh_lut.size()) *
                    sizeof(uint64_t) +
                p.integer_mean_lut.size() * sizeof(double) +
                (p.inv_sigma_lut.size() + p.raw_thresh_lut.size()) *
                    sizeof(float);
        }
        return {
            .plan =
                fdmt.plan + host_plan + m_rows_d.size() * sizeof(BoxcarRow) +
                m_widths_d.size() * sizeof(uint32_t) +
                m_screen_groups_d.size() * sizeof(BoxcarScreenGroup) +
                (m_screen_thresholds_d.size() + m_screen_m1_d.size()) *
                    sizeof(float) +
                m_integer_means.size() * sizeof(double) +
                m_integer_thresholds.size() * sizeof(uint64_t) +
                (m_thresholds_d.size() + m_scales_d.size()) * sizeof(float) +
                m_support_d.size() * sizeof(uint64_t) +
                (m_slots.size() * 4 * table_floats() * sizeof(float)),
            .state   = fdmt.state,
            .history = fdmt.history + m_carry_d.size() * sizeof(float) +
                       m_integer_carry.size() * sizeof(uint64_t) +
                       m_integer_carry32.size() * sizeof(uint32_t),
            .peaks   = m_peaks_d.size() * sizeof(BoxcarPeak) + m_open_d.size(),
            .workspace = fdmt.workspace + m_per_dm_d.size() * sizeof(uint64_t) +
                         sizeof(CandidateCounts) + sizeof(uint64_t) +
                         (2 * sizeof(uint64_t)), // candidate frontier word
            .candidates = (m_candidates_d.size() + m_candidates_h.size()) *
                          sizeof(RawCandidate),
            .staging =
                m_input_d.size() * sizeof(float) + m_packed_input.size()};
    }
    SizeType chunk_nsamps() const noexcept override {
        return m_driver->chunk_nsamps();
    }
    std::string search_path() const override { return m_driver->search_path(); }
    SizeType fuse_levels() const noexcept override {
        return m_driver->fuse_levels();
    }

private:
    struct LevelGroup {
        SizeType level;
        SizeType row_begin;
        SizeType row_count;
    };

    template <typename T, int kBlockThreads, typename C>
    void launch_score_block(const FDMTSearchInput& input,
                            const LevelGroup& g,
                            dim3 grid,
                            dim3 block,
                            C* carry) {
        // Per-call mask tables take their own kernel instance (float
        // rows only).
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
            launch_score_kernel<T, kBlockThreads, false, false, C>(
                input, g, grid, block, carry);
        } else {
            launch_score_kernel<T, kBlockThreads, true, false, C>(
                input, g, grid, block, carry);
        }
    }

    template <typename T,
              int kBlockThreads,
              bool kScreen,
              bool kBlockNoise,
              typename C>
    void launch_score_kernel(const FDMTSearchInput& input,
                             const LevelGroup& g,
                             dim3 grid,
                             dim3 block,
                             C* carry) {
        const auto stream = static_cast<cudaStream_t>(input.stream.native);
        FDMTSearchInput stopped = input;
        stopped.search_stop     = m_search_stop;
        kernel_fdmt_boxcar<T, kBlockThreads, kScreen, kBlockNoise, C>
            <<<grid, block, m_smem, stream>>>(
                stopped, raw(m_rows_d), g.row_begin, g.row_count, m_rows.size(),
                raw(m_widths_d), m_filters, m_width, raw(m_thresholds_d),
                raw(m_scales_d), carry, raw(m_peaks_d), raw(m_open_d),
                m_peak_count, raw(m_candidates_d), raw(m_counts_d),
                raw(m_slots_d), raw(m_per_dm_d), m_config.max_candidates,
                m_config.max_cands_per_dm, m_config.peak_mode,
                m_config.skip_cold_prefix, g.level == m_plan.get_niters(),
                raw(m_integer_thresholds), raw(m_integer_means), m_integer_unit,
                m_mask, m_shared_tables, m_config.use_trapezoid,
                raw(m_screen_groups_d),
                static_cast<int>(m_screen_groups_d.size()),
                raw(m_screen_thresholds_d), raw(m_screen_m1_d), m_screen_offset,
                kBlockNoise ? m_block_tables : BlockTables{}, m_group_offset);
    }

    template <typename T, typename C = T>
    void launch_score(const FDMTSearchInput& input,
                      const LevelGroup& g,
                      dim3 grid,
                      dim3 block,
                      C* carry) {
        if (m_block_threads == 64) {
            launch_score_block<T, 64, C>(input, g, grid, block, carry);
        } else if (m_block_threads == 128) {
            launch_score_block<T, 128, C>(input, g, grid, block, carry);
        } else {
            launch_score_block<T, 256, C>(input, g, grid, block, carry);
        }
    }

    template <typename T, int kBlockThreads>
    void configure_shared_block(SizeType limit) {
        constexpr SizeType kBlockTile = kBlockThreads * kItems;
        const SizeType ring           = std::bit_ceil(m_width + kBlockTile);
        const SizeType fixed =
            sizeof(BoxcarShared<T, kBlockThreads>) + ring * sizeof(T);
        const SizeType tables = m_filters * (sizeof(T) + sizeof(uint32_t));
        m_block_threads       = kBlockThreads;
        m_mask                = static_cast<uint32_t>(ring - 1);
        m_shared_tables =
            m_filters <= kSharedFilters && fixed + tables <= limit;
        m_smem = fixed + (m_shared_tables ? tables : 0);
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
            configure_kernel(kernel_fdmt_boxcar<T, kBlockThreads, false>);
        } else {
            configure_kernel(kernel_fdmt_boxcar<T, kBlockThreads, true>);
        }
        if constexpr (std::is_integral_v<T>) {
            if (m_screen_groups_d.empty()) {
                configure_kernel(kernel_fdmt_boxcar<T, kBlockThreads, false,
                                                    false, uint32_t>);
            } else {
                configure_kernel(kernel_fdmt_boxcar<T, kBlockThreads, true,
                                                    false, uint32_t>);
            }
        } else {
            if (m_screen_groups_d.empty()) {
                configure_kernel(
                    kernel_fdmt_boxcar<T, kBlockThreads, false, true>);
            } else {
                configure_kernel(
                    kernel_fdmt_boxcar<T, kBlockThreads, true, true>);
            }
        }
    }

    template <typename Kernel> void configure_kernel(Kernel kernel) const {
        gpu_utils::check_gpu_call(
            cudaFuncSetAttribute(kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(m_smem)),
            "FDMTBoxcar: shared-memory opt-in");
#if !defined(DMT_ENABLE_HIP)
        // Several resident rows per SM need the shared-memory carveout.
        gpu_utils::check_gpu_call(
            cudaFuncSetAttribute(kernel,
                                 cudaFuncAttributePreferredSharedMemoryCarveout,
                                 cudaSharedmemCarveoutMaxShared),
            "FDMTBoxcar: shared-memory carveout");
#endif
    }

    // Ring of at least max_width + kBlockTile prefixes (a power of two), plus
    // the row's thresholds and widths when they fit.
    template <typename T> void configure_shared() {
        int optin = 0;
        gpu_utils::check_gpu_call(
            cudaDeviceGetAttribute(
                &optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, m_device),
            "FDMTBoxcar: shared memory limit");
        const auto limit = static_cast<SizeType>(optin);
        const SizeType eff_chunk =
            std::min(m_driver->chunk_nsamps(), m_plan.get_nsamps());
        if (eff_chunk <= 64 * kItems) {
            configure_shared_block<T, 64>(limit);
        } else if (eff_chunk <= 128 * kItems) {
            configure_shared_block<T, 128>(limit);
        } else {
            configure_shared_block<T, 256>(limit);
        }
    }
    /// The block with the killed channels' rows zeroed in `buffer` (copied
    /// there first unless the host path staged it there), or the block
    /// itself without a kill mask.
    template <typename T>
    DeviceSpan<const T> masked(DeviceSpan<const T> block,
                               thrust::device_vector<T>& buffer,
                               Stream queue) const {
        const auto& runs = m_noise_on ? m_ring.back().kill_runs : m_kill_runs;
        if (runs.empty()) {
            return block;
        }
        const auto stream = static_cast<cudaStream_t>(queue.native);
        T* dst            = raw(buffer);
        if (block.data() != dst) {
            // The previous block's search may still read the buffer.
            m_work.order(stream);
            gpu_utils::check_gpu_call(
                cudaMemcpyAsync(dst, block.data(), block.size_bytes(),
                                cudaMemcpyDeviceToDevice, stream));
        }
        const SizeType nchans = m_plan.get_nchans();
        const SizeType row    = block.size_bytes() / (m_beams * nchans);
        for (const auto& [c0, c1] : runs) {
            // Channels [c0, c1) of every beam: rows nchans rows apart.
            gpu_utils::check_gpu_call(cudaMemset2DAsync(
                reinterpret_cast<std::byte*>(dst) + (c0 * row), nchans * row, 0,
                (c1 - c0) * row, m_beams, stream));
        }
        return {dst, block.size(), block.device};
    }
    /// Candidate frontier: the oldest open begin sample, reduced on the
    /// device and copied to a pinned word on `stream` (no host wait).
    void update_open_begin(cudaStream_t stream) {
        gpu_utils::check_gpu_call(
            cudaMemsetAsync(raw(m_oldest_d), 0xFF, sizeof(uint64_t), stream));
        const SizeType tasks = m_rows.size() * m_beams;
        if (m_config.peak_mode != PeakFindingMode::kAllAboveThreshold) {
            const auto blocks = static_cast<unsigned>(
                std::min<SizeType>((tasks + 255) / 256, 1024));
            kernel_open_begin<<<blocks, 256, 0, stream>>>(
                raw(m_peaks_d), raw(m_open_d), tasks, m_peak_count,
                raw(m_oldest_d));
            gpu_utils::check_last_gpu_error("FDMTBoxcar frontier failed");
        }
        gpu_utils::check_gpu_call(
            cudaMemcpyAsync(m_oldest_h.data(), raw(m_oldest_d),
                            sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
    }
    /// Table slots: steady and transition sets of every ring entry, plus
    /// the entry being committed.
    [[nodiscard]] SizeType ring_size() const noexcept {
        const uint64_t ns = m_plan.get_nsamps();
        return 1 + static_cast<SizeType>((search_latency() + ns - 1) / ns);
    }
    [[nodiscard]] SizeType slot_count() const noexcept {
        return (2 * ring_size()) + 2;
    }
    [[nodiscard]] SizeType table_floats() const noexcept {
        return m_rows.size() * m_filters;
    }
    void reset_block_noise() {
        m_pending.reset();
        m_ring.clear();
        BlockNoise first; // the constructor tables from stream sample 0
        first.kill_runs = m_kill_runs;
        m_ring.push_back(std::move(first));
        m_block_tables = {};
    }
    /// The device copy of `tables` (uploaded into a slot no ring entry uses
    /// if needed); null for the constructor tables.
    const float* slot_of(const NoiseTables* tables, cudaStream_t stream) {
        if (tables == nullptr) {
            return nullptr;
        }
        for (auto& slot : m_slots) {
            if (slot->id == tables->id) {
                return raw(slot->device);
            }
        }
        const auto in_ring = [&](uint64_t id) {
            for (const auto& e : m_ring) {
                if ((e.steady && e.steady->id == id) ||
                    (e.transition_tables && e.transition_tables->id == id)) {
                    return true;
                }
            }
            return false;
        };
        for (auto& slot : m_slots) {
            if (slot->id != 0 && in_ring(slot->id)) {
                continue;
            }
            // The previous copy out of the pinned buffer has finished
            // before it is overwritten; the device buffer is rewritten only
            // after the kernels that read it (stream order, no host wait).
            if (slot->copy_pending) {
                gpu_utils::check_gpu_call(cudaEventSynchronize(slot->copied),
                                          "FDMTBoxcar: table copy failed");
            }
            const SizeType n = table_floats();
            std::copy_n(tables->thresh.data(), n, slot->host.data());
            std::copy_n(tables->inv_sigma.data(), n, slot->host.data() + n);
            if (slot->release_pending) {
                gpu_utils::check_gpu_call(
                    cudaStreamWaitEvent(stream, slot->released, 0),
                    "FDMTBoxcar: stream wait failed");
            }
            gpu_utils::check_gpu_call(
                cudaMemcpyAsync(raw(slot->device), slot->host.data(),
                                2 * n * sizeof(float), cudaMemcpyHostToDevice,
                                stream),
                "FDMTBoxcar: table upload failed");
            gpu_utils::check_gpu_call(cudaEventRecord(slot->copied, stream),
                                      "FDMTBoxcar: event record failed");
            slot->copy_pending = true;
            slot->id           = tables->id;
            return raw(slot->device);
        }
        throw std::logic_error("FDMTBoxcar (gpu): no free table slot");
    }
    /// Commits the pushed entry (or repeats the last steady tables), then
    /// sets the kernels' tables to the ring's last two entries.
    void commit_block_noise(cudaStream_t stream) {
        if (!m_noise_on) {
            return;
        }
        BlockNoise entry;
        if (m_pending) {
            entry = std::move(*m_pending);
            m_pending.reset();
        } else {
            entry.steady    = m_ring.back().steady;
            entry.kill_runs = m_kill_runs;
        }
        entry.first_sample = m_position;
        if (m_ring.size() == ring_size()) {
            m_ring.erase(m_ring.begin());
        }
        m_ring.push_back(std::move(entry));
        BlockTables bt;
        const SizeType n   = table_floats();
        const SizeType off = m_ring.size() > 2 ? m_ring.size() - 2 : 0;
        bt.count           = static_cast<int>(m_ring.size() - off);
        bt.support         = raw(m_support_d);
        for (int e = 0; e < bt.count; ++e) {
            const auto& r    = m_ring[off + e];
            bt.first[e]      = r.first_sample;
            bt.transition[e] = r.transition;
            const float* st  = slot_of(r.steady.get(), stream);
            bt.thr[e][0]     = st;
            bt.inv[e][0]     = st != nullptr ? st + n : nullptr;
            if (r.transition) {
                const float* tr = slot_of(r.transition_tables.get(), stream);
                bt.thr[e][1]    = tr;
                bt.inv[e][1]    = tr != nullptr ? tr + n : nullptr;
            }
        }
        m_block_tables = bt;
    }
    /// After a call's kernels: the slots its tables used are free for
    /// reuse only after them.
    void release_slots(cudaStream_t stream) {
        if (!m_noise_on) {
            return;
        }
        for (auto& slot : m_slots) {
            if (slot->id != 0) {
                gpu_utils::check_gpu_call(
                    cudaEventRecord(slot->released, stream),
                    "FDMTBoxcar: event record failed");
                slot->release_pending = true;
            }
        }
    }
    void validate_packed(SizeType size) const {
        if (m_config.input_nbits == 32 ||
            size != m_beams * m_plan.get_nchans() *
                        bit_pack_utils::packed_row_bytes(
                            m_plan.get_nsamps(), m_config.input_nbits)) {
            throw std::invalid_argument(
                "FDMTBoxcar: packed input size or bit depth mismatch");
        }
        if (m_position > UINT64_MAX - m_plan.get_nsamps()) {
            throw std::overflow_error(
                "FDMTBoxcar: stream sample index overflow");
        }
    }
    void validate_input(SizeType size) const {
        if (size != m_beams * m_plan.get_nchans() * m_plan.get_nsamps()) {
            throw std::invalid_argument("FDMTBoxcar: wrong waterfall size");
        }
        if (m_position > UINT64_MAX - m_plan.get_nsamps()) {
            throw std::overflow_error(
                "FDMTBoxcar: stream sample index overflow");
        }
    }
    void begin_result(SizeType samples, Stream queue) {
        gpu_utils::set_device(m_device);
        const auto stream = static_cast<cudaStream_t>(queue.native);
        gpu_utils::check_gpu_call(cudaMemsetAsync(
            raw(m_counts_d), 0, sizeof(CandidateCounts), stream));
        gpu_utils::check_gpu_call(
            cudaMemsetAsync(raw(m_slots_d), 0, sizeof(uint64_t), stream));
        gpu_utils::check_gpu_call(cudaMemsetAsync(
            raw(m_per_dm_d), 0, m_per_dm_d.size() * sizeof(uint64_t), stream));
        m_result_begin   = m_position;
        m_result_samples = samples;
    }
    CandidateDeviceResultView device_view() {
        const Device device{kGPUBackend, m_device};
        return {DeviceSpan<const RawCandidate>(raw(m_candidates_d),
                                               m_candidates_d.size(), device),
                DeviceSpan<const CandidateCounts>(raw(m_counts_d), 1, device),
                m_result_begin, m_result_samples};
    }

    const plans::FDMTPlan& m_plan;
    BoxcarConfig m_config;
    const std::vector<plans::BoxcarPlanePlan>& m_planes;
    SizeType m_beams;
    int m_device;
    std::vector<BoxcarRow> m_rows;
    SizeType m_width;
    SizeType m_filters;
    SizeType m_peak_count;
    std::unique_ptr<BoxcarDriver> m_driver;
    /// Killed channel runs, zeroed in m_input_d / m_packed_input.
    std::vector<std::pair<SizeType, SizeType>> m_kill_runs;
    std::vector<LevelGroup> m_level_groups;
    thrust::device_vector<BoxcarRow> m_rows_d;
    thrust::device_vector<uint32_t> m_widths_d;
    thrust::device_vector<float> m_thresholds_d, m_scales_d, m_carry_d,
        m_input_d;
    thrust::device_vector<uint64_t> m_integer_carry, m_integer_thresholds;
    thrust::device_vector<uint32_t> m_integer_carry32;
    thrust::device_vector<double> m_integer_means;
    thrust::device_vector<uint8_t> m_packed_input;
    double m_integer_unit = 1;
    int m_block_threads   = 256;
    thrust::device_vector<BoxcarScreenGroup> m_screen_groups_d;
    thrust::device_vector<float> m_screen_thresholds_d;
    thrust::device_vector<float> m_screen_m1_d; ///< packed: mean per sample
    SizeType m_screen_offset = 0;
    uint32_t m_mask          = 0;
    bool m_shared_tables     = false;
    SizeType m_smem          = 0;
    thrust::device_vector<BoxcarPeak> m_peaks_d;
    thrust::device_vector<uint8_t> m_open_d; ///< any peak open, per task
    thrust::device_vector<uint64_t> m_per_dm_d, m_slots_d;
    thrust::device_vector<CandidateCounts> m_counts_d;
    thrust::device_vector<RawCandidate> m_candidates_d;
    std::vector<RawCandidate> m_candidates_h;
    CandidateCounts m_counts_h{};
    gpu_utils::DeviceWorkFence m_work;
    uint64_t m_position       = 0;
    uint64_t m_search_stop    = UINT64_MAX; // drain(): end of stream
    uint64_t m_result_begin   = 0;
    SizeType m_result_samples = 0;
    SizeType m_group_offset   = 0;
    thrust::device_vector<uint64_t> m_oldest_d; ///< candidate frontier
    gpu_host::PinnedBuffer<uint64_t> m_oldest_h;
    // Per-call kill masks: off until the first one.
    struct TableSlot {
        thrust::device_vector<float> device; ///< thresholds, then inv sigma
        gpu_host::PinnedBuffer<float> host;
        uint64_t id          = 0; ///< NoiseTables::id held (0: none)
        cudaEvent_t copied   = nullptr;
        cudaEvent_t released = nullptr;
        bool copy_pending    = false;
        bool release_pending = false;
    };
    bool m_noise_on = false;
    std::vector<BlockNoise> m_ring; ///< oldest first, at most ring_size()
    std::optional<BlockNoise> m_pending;
    std::vector<std::unique_ptr<TableSlot>> m_slots;
    thrust::device_vector<uint64_t> m_support_d;
    BlockTables m_block_tables{};
};

} // namespace
std::unique_ptr<FDMTBoxcarEngine>
make_fdmt_boxcar_gpu(const plans::FDMTPlan& plan,
                     const BoxcarConfig& config,
                     const std::vector<plans::BoxcarPlanePlan>& planes,
                     const FDMTEngineConfig& cfg,
                     std::unique_ptr<BoxcarDriver> driver,
                     std::vector<std::pair<SizeType, SizeType>> kill_runs) {
    return std::make_unique<FDMTBoxcarCudaEngine>(
        plan, config, planes, cfg, std::move(driver), std::move(kill_runs));
}
} // namespace dmt::algorithms::detail
