#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <thrust/device_vector.h>

#include "dmt/bit_pack_utils.hpp"
#include "dmt/boxcar.hpp"
#include "dmt/boxcar_runs.hpp"
#include "dmt/fdmt_variance.hpp"
#include "dmt/gpu_compat.cuh"
#include "dmt/gpu_utils.cuh"

namespace dmt::algorithms::detail {
namespace {

constexpr int kItems = 4;
constexpr int kWarp  = 32; // Logical warp; a 64-wide wavefront holds two.
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
#if defined(DMT_ENABLE_HIP)
    return __shfl_up(value, delta, kWarp);
#else
    return __shfl_up_sync(0xFFFFFFFFU, value, delta);
#endif
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

// kScreen: tiles are screened by width group (detect_tile_grouped*); a
// separate instantiation, so the plain kernel keeps its register budget.
template <typename T, int kBlockThreads, bool kScreen>
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
                       T* carry_all,
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
                       SizeType screen_offset) {
    constexpr int kBlockTile = kBlockThreads * kItems;
    extern __shared__ __align__(16) unsigned char storage[];
    auto& shared = *reinterpret_cast<BoxcarShared<T, kBlockThreads>*>(storage);
    T* ring      = reinterpret_cast<T*>(storage + sizeof(shared));
    const SizeType task      = blockIdx.x;
    const SizeType beam      = input.beam_begin + task / ndm;
    const SizeType d         = task % ndm;
    const SizeType row_index = row_base + d;
    const SizeType index     = beam * total_rows + row_index;
    const BoxcarRow row      = rows[row_index];
    T* carry                 = carry_all + index * max_width;
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
    // Row tables: raw thresholds in the units of the prefix ring.
    const T* thr_row = [&] {
        if constexpr (std::is_integral_v<T>) {
            return integer_thresholds + row_index * nfilters;
        } else {
            return thresholds + row_index * nfilters;
        }
    }();
    const uint32_t* wtab = widths;
    const T* ttab        = thr_row;
    if (shared_tables) {
        T* s_thr       = ring + mask + 1;
        auto* s_widths = reinterpret_cast<uint32_t*>(s_thr + nfilters);
        for (SizeType f = tid; f < nfilters; f += kBlockThreads) {
            s_thr[f]    = thr_row[f];
            s_widths[f] = widths[f];
        }
        wtab = s_widths;
        ttab = s_thr;
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
                        screen_groups, screen_ngroups,
                        screen_thresholds + row_index * screen_ngroups,
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
        if (n == kBlockTile) {
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
                const T threshold = ttab[f];
                const float scale = inv_sigma[row_index * nfilters + f];
                bool f_hit        = false;
                for (int j = 0; j < kItems; ++j) {
                    const uint32_t local  = tid * kItems + j;
                    const uint64_t sample = start + local;
                    const uint32_t idx    = position + local + 1;
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
            carry[slot] = boxcar_integer_sample(input, row, beam, sample, root);
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

template <typename T> T* raw(thrust::device_vector<T>& v) {
    return thrust::raw_pointer_cast(v.data());
}

class FDMTBoxcarCudaEngine final : public FDMTBoxcarEngine {
public:
    FDMTBoxcarCudaEngine(const plans::FDMTPlan& plan,
                         const BoxcarConfig& config,
                         const std::vector<plans::BoxcarPlanePlan>& planes,
                         const FDMTEngineConfig& cfg,
                         std::unique_ptr<BoxcarDriver> driver)
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
        m_driver = std::move(driver);
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
            m_integer_carry.resize(tasks * m_width, 0);
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
    }
    ~FDMTBoxcarCudaEngine() override {
        // Destruction must not release search buffers while queued work
        // still references them, including on a nondefault stream.
        cudaSetDevice(m_device);
        m_work.wait();
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
        begin_result(m_plan.get_nsamps(), queue);
        m_driver->search(waterfall, *this, queue);
        m_position += m_plan.get_nsamps();
        m_work.mark(static_cast<cudaStream_t>(queue.native));
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
        m_driver->search(waterfall, m_config.input_nbits, *this, queue);
        m_position += m_plan.get_nsamps();
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
    void reset_history(Stream queue) override {
        gpu_utils::set_device(m_device);
        const auto stream = static_cast<cudaStream_t>(queue.native);
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
        m_position = 0;
        begin_result(0, queue);
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
                launch_score<uint64_t>(input, g, grid, block,
                                       raw(m_integer_carry));
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
            .plan = fdmt.plan + host_plan +
                    m_rows_d.size() * sizeof(BoxcarRow) +
                    m_widths_d.size() * sizeof(uint32_t) +
                    m_screen_groups_d.size() * sizeof(BoxcarScreenGroup) +
                    (m_screen_thresholds_d.size() + m_screen_m1_d.size()) *
                        sizeof(float) +
                    m_integer_means.size() * sizeof(double) +
                    m_integer_thresholds.size() * sizeof(uint64_t) +
                    (m_thresholds_d.size() + m_scales_d.size()) * sizeof(float),
            .state   = fdmt.state,
            .history = fdmt.history + m_carry_d.size() * sizeof(float) +
                       m_integer_carry.size() * sizeof(uint64_t),
            .peaks   = m_peaks_d.size() * sizeof(BoxcarPeak) + m_open_d.size(),
            .workspace = fdmt.workspace + m_per_dm_d.size() * sizeof(uint64_t) +
                         sizeof(CandidateCounts) + sizeof(uint64_t),
            .candidates = (m_candidates_d.size() + m_candidates_h.size()) *
                          sizeof(RawCandidate),
            .staging =
                m_input_d.size() * sizeof(float) + m_packed_input.size()};
    }
    SizeType chunk_nsamps() const noexcept override {
        return m_driver->chunk_nsamps();
    }
    SizeType fuse_levels() const noexcept override {
        return m_driver->fuse_levels();
    }

private:
    struct LevelGroup {
        SizeType level;
        SizeType row_begin;
        SizeType row_count;
    };

    template <typename T, int kBlockThreads>
    void launch_score_block(const FDMTSearchInput& input,
                            const LevelGroup& g,
                            dim3 grid,
                            dim3 block,
                            T* carry) {
        if (m_screen_groups_d.empty()) {
            launch_score_kernel<T, kBlockThreads, false>(input, g, grid, block,
                                                         carry);
        } else {
            launch_score_kernel<T, kBlockThreads, true>(input, g, grid, block,
                                                        carry);
        }
    }

    template <typename T, int kBlockThreads, bool kScreen>
    void launch_score_kernel(const FDMTSearchInput& input,
                             const LevelGroup& g,
                             dim3 grid,
                             dim3 block,
                             T* carry) {
        const auto stream = static_cast<cudaStream_t>(input.stream.native);
        kernel_fdmt_boxcar<T, kBlockThreads, kScreen>
            <<<grid, block, m_smem, stream>>>(
                input, raw(m_rows_d), g.row_begin, g.row_count, m_rows.size(),
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
                raw(m_screen_thresholds_d), raw(m_screen_m1_d),
                m_screen_offset);
    }

    template <typename T>
    void launch_score(const FDMTSearchInput& input,
                      const LevelGroup& g,
                      dim3 grid,
                      dim3 block,
                      T* carry) {
        if (m_block_threads == 64) {
            launch_score_block<T, 64>(input, g, grid, block, carry);
        } else if (m_block_threads == 128) {
            launch_score_block<T, 128>(input, g, grid, block, carry);
        } else {
            launch_score_block<T, 256>(input, g, grid, block, carry);
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
    std::vector<LevelGroup> m_level_groups;
    thrust::device_vector<BoxcarRow> m_rows_d;
    thrust::device_vector<uint32_t> m_widths_d;
    thrust::device_vector<float> m_thresholds_d, m_scales_d, m_carry_d,
        m_input_d;
    thrust::device_vector<uint64_t> m_integer_carry, m_integer_thresholds;
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
    uint64_t m_result_begin   = 0;
    SizeType m_result_samples = 0;
};

} // namespace
std::unique_ptr<FDMTBoxcarEngine>
make_fdmt_boxcar_gpu(const plans::FDMTPlan& plan,
                     const BoxcarConfig& config,
                     const std::vector<plans::BoxcarPlanePlan>& planes,
                     const FDMTEngineConfig& cfg,
                     std::unique_ptr<BoxcarDriver> driver) {
    return std::make_unique<FDMTBoxcarCudaEngine>(plan, config, planes, cfg,
                                                  std::move(driver));
}
} // namespace dmt::algorithms::detail
