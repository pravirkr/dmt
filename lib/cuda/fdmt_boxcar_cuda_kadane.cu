#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <fmt/format.h>

#include "dmt/boxcar.hpp"
#include "dmt/fdmt_int_tree.hpp"
#include "dmt/gpu_compat.cuh"
#include "dmt/gpu_utils.cuh"
#include "dmt/kadane_cuda.cuh"
#include "dmt/kadane_monoid.hpp"
#include "dmt/kadane_plan.hpp"
#include "dmt/logging.hpp"

#include "fdmt_boxcar_cuda_engine.cuh"

// GPU FDMTBoxcar engine: Design C shared-memory-lean tropical Kadane monoid
// gate (`BoxcarScreenMode::kKadane`) with lazy carry & prefix-ring bootstrap,
// kKeepBrightest support, and wide-boxcar / small-SMEM fallback.

namespace dmt::algorithms::detail::fdmt_boxcar_cuda {
namespace {

template <typename T, typename SegT, int kBlockThreads>
struct alignas(16) KadaneBlockShared {
    SegT warp_segs[kadane::kMaxGateBuckets][kBlockThreads / kWarp];
    float row_cuts[kadane::kMaxGateBuckets];
    BoxcarShared<T, kBlockThreads> boxcar;
};

template <typename T,
          typename SegT,
          int kBlockThreads,
          bool kUseIntGate,
          bool kBlockNoise = false,
          typename C       = T,
          bool kKeep       = false>
__global__ void __launch_bounds__(kBlockThreads, 4)
    kernel_fdmt_boxcar_kadane(FDMTSearchInput input,
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
                              kadane::KadaneGateDeviceView gate,
                              BlockTables block_tables,
                              BoxcarKeepEntry* keep_all,
                              SizeType keep_q) {
    constexpr int kBlockTile  = kBlockThreads * kItems;
    constexpr int kBlockWarps = kBlockThreads / kWarp;
    constexpr int kLeaderTid  = kBlockWarps - 1;
    const SizeType task       = SizeType{blockIdx.x};

    if (input.row_lag != nullptr) {
        FDMTSearchInput lagged;
        if (!lagged_row_input(input, rows[row_base + (task % ndm)].row,
                              lagged)) {
            return;
        }
        input = lagged;
    }
    if (!stop_row_input(input) || input.nsamps == 0) {
        return;
    }

    const SizeType beam      = input.beam_begin + (task / ndm);
    const SizeType d         = task % ndm;
    const SizeType row_index = row_base + d;
    const SizeType index     = (beam * total_rows) + row_index;
    const int tid            = static_cast<int>(threadIdx.x);
    const int lane           = tid % kWarp;
    const int warp           = tid / kWarp;

    bool open = input.sample_begin > 0 && open_all[index] != 0;
    if (gate.disabled_row[row_index] != 0U && !open) {
        if (tid == 0) {
            open_all[index] = 0;
        }
        return;
    }

    extern __shared__ __align__(16) unsigned char storage[];
    auto& kshared =
        *reinterpret_cast<KadaneBlockShared<T, SegT, kBlockThreads>*>(storage);
    auto& shared = kshared.boxcar;
    T* ring      = reinterpret_cast<T*>(storage + sizeof(kshared));

    const BoxcarRow row    = rows[row_index];
    C* carry               = carry_all + (index * max_width);
    BoxcarPeak* peaks      = peaks_all + (index * peak_count);
    RawCandidate prototype = row.candidate;
    prototype.beam_idx     = static_cast<uint32_t>(beam);
    CandidateWriter writer{candidates, counts,    slots,    per_dm_all + index,
                           widths,     prototype, capacity, dm_cap};
    writer.means  = means != nullptr ? means + (row_index * nfilters) : nullptr;
    writer.scales = inv_sigma + (row_index * nfilters);
    writer.unit   = unit;
    writer.integer = std::is_integral_v<T> &&
                     mode != PeakFindingMode::kContiguousPeakAcrossWidths;

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

    const auto lookback    = static_cast<uint32_t>(max_width - 1);
    const float* plane_thr = thresholds + (row_index * nfilters);
    const float* plane_inv = inv_sigma + (row_index * nfilters);
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
                t = t != nullptr ? t + (row_index * nfilters) : plane_thr;
                v = v != nullptr ? v + (row_index * nfilters) : plane_inv;
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

    const T* thr_row = [&] {
        if constexpr (std::is_integral_v<T>) {
            return integer_thresholds + (row_index * nfilters);
        } else {
            return thr_float;
        }
    }();
    T* s_thr             = ring + mask + 1;
    auto* s_widths       = reinterpret_cast<uint32_t*>(s_thr + nfilters);
    const uint32_t* wtab = shared_tables ? s_widths : widths;
    const T* ttab        = shared_tables ? s_thr : thr_row;
    bool tables_staged   = false;

    const auto stage_shared_tables = [&] {
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
        tables_staged = true;
    };

    const SizeType nb = gate.num_buckets;
    float row_biases_f32[kadane::kMaxGateBuckets]{};
    float row_cuts_f32[kadane::kMaxGateBuckets]{};
    int32_t row_ibias[kadane::kMaxGateBuckets]{};
    int32_t row_icut[kadane::kMaxGateBuckets]{};

    if constexpr (kUseIntGate) {
        for (SizeType k = 0; k < nb; ++k) {
            row_ibias[k] = gate.ibias[(k * gate.row_stride) + row_index];
            row_icut[k]  = gate.icut[(k * gate.row_stride) + row_index];
        }
    } else {
        for (SizeType k = 0; k < nb; ++k) {
            row_biases_f32[k] = gate.bias[(k * gate.row_stride) + row_index];
            row_cuts_f32[k]   = gate.cut[(k * gate.row_stride) + row_index];
        }
        if constexpr (kBlockNoise && !std::is_integral_v<T>) {
            if (thr_float != plane_thr || mixed) {
                if (shared_tables) {
                    stage_shared_tables();
                    __syncthreads();
                }
                if (static_cast<SizeType>(tid) < nb) {
                    const auto k = static_cast<SizeType>(tid);
                    if (gate.disabled_row[row_index] != 0U) {
                        kshared.row_cuts[k] = std::numeric_limits<float>::max();
                    } else {
                        const auto min_thr_at = [&](SizeType f) -> float {
                            if (shared_tables) {
                                return s_thr[f];
                            }
                            if (!mixed) {
                                return thr_float[f];
                            }
                            float th = in_play[0][0][f];
                            for (int j = 1; j < n_play; ++j) {
                                th = th < in_play[j][0][f] ? th
                                                           : in_play[j][0][f];
                            }
                            return th;
                        };
                        kshared.row_cuts[k] = kadane::compute_bucket_cut_hd_f32(
                            gate.buckets[k], shared_tables ? s_widths : widths,
                            nfilters, use_trapezoid, row_biases_f32[k],
                            min_thr_at);
                    }
                }
                __syncthreads();
                for (SizeType k = 0; k < nb; ++k) {
                    row_cuts_f32[k] = kshared.row_cuts[k];
                }
            }
        }
    }

    const double row_m1 =
        (std::is_integral_v<T> && !kUseIntGate && gate.row_int_m1 != nullptr)
            ? gate.row_int_m1[row_index]
            : 0.0;

    float c_state_f32[kadane::kMaxGateBuckets]{};
    int32_t c_state_i32[kadane::kMaxGateBuckets]{};
    kadane::LagWindow<float> lag_state_f32[kadane::kMaxGateBuckets]{};
    kadane::LagWindow<int32_t> lag_state_i32[kadane::kMaxGateBuckets]{};

    if (tid == kLeaderTid && input.sample_begin > 0) {
        const SizeType state_off = index * nb;
        for (SizeType k = 0; k < nb; ++k) {
            if constexpr (kUseIntGate) {
                c_state_i32[k] = gate.state_ic[state_off + k];
                if (gate.any_edge_lag) {
                    lag_state_i32[k] = gate.lag_i32[state_off + k];
                }
            } else {
                c_state_f32[k] = gate.state_fc[state_off + k];
                if (gate.any_edge_lag) {
                    lag_state_f32[k] = gate.lag_f32[state_off + k];
                }
            }
        }
    }

    const auto any_open = [&] {
        bool loc_open = false;
        if (mode != PeakFindingMode::kAllAboveThreshold) {
            for (SizeType f = tid; f < peak_count; f += kBlockThreads) {
                loc_open = loc_open || peaks[f].open;
            }
        }
        return __syncthreads_or(loc_open) != 0;
    };

    bool ring_ready        = false;
    SizeType ring_synced_t = 0;

    for (SizeType t0 = 0; t0 < input.nsamps; t0 += kBlockTile) {
        const int n = static_cast<int>(
            input.nsamps - t0 < kBlockTile ? input.nsamps - t0 : kBlockTile);
        const uint64_t start = input.sample_begin + t0;

        T x[kItems];
#pragma unroll
        for (int j = 0; j < kItems; ++j) {
            const SizeType local = static_cast<SizeType>(tid * kItems + j);
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

        float x_f32[kItems]{};
        if constexpr (!kUseIntGate) {
#pragma unroll
            for (int j = 0; j < kItems; ++j) {
                const SizeType local = static_cast<SizeType>(tid * kItems + j);
                const bool valid =
                    local < static_cast<SizeType>(n) &&
                    (!skip_cold || (start + local) >= row.warmup);
                if (valid) {
                    if constexpr (std::is_integral_v<T>) {
                        x_f32[j] = static_cast<float>(
                            (static_cast<double>(x[j]) - row_m1) * unit);
                    } else {
                        x_f32[j] = x[j];
                    }
                }
            }
        }

        for (SizeType k = 0; k < nb; ++k) {
            SegT local_seg{};
#pragma unroll
            for (int j = 0; j < kItems; ++j) {
                const SizeType local = static_cast<SizeType>(tid * kItems + j);
                const bool valid =
                    local < static_cast<SizeType>(n) &&
                    (!skip_cold || (start + local) >= row.warmup);
                SegT item_seg{};
                if constexpr (kUseIntGate) {
                    item_seg = kadane::make_kadane_seg_i64(
                        static_cast<uint64_t>(x[j]), gate.int_shift,
                        row_ibias[k], valid);
                } else {
                    item_seg = kadane::make_kadane_seg_f32(
                        x_f32[j], row_biases_f32[k], valid);
                }
                local_seg = j == 0
                                ? item_seg
                                : kadane::kadane_combine(local_seg, item_seg);
            }
            const SegT warp_tot = kadane::warp_kadane_reduce(local_seg, lane);
            if (lane == kWarp - 1) {
                kshared.warp_segs[k][warp] = warp_tot;
            }
        }
        __syncthreads();

        bool my_hit = false;
        if (warp == 0) {
            const bool has_cold_start = skip_cold && start < row.warmup;
            const uint64_t tile_end   = start + static_cast<uint64_t>(n);
            for (SizeType k = 0; k < nb; ++k) {
                const SegT total_seg =
                    kadane::warp_totals_kadane_reduce<SegT, kBlockWarps>(
                        kshared.warp_segs[k], lane);
                if (lane == kLeaderTid) {
                    const uint32_t lag_len = gate.max_edge_lag[k];
                    if constexpr (kUseIntGate) {
                        const int32_t c_in =
                            has_cold_start ? int32_t{0} : c_state_i32[k];
                        const auto app = kadane::kadane_apply(c_in, total_seg);
                        c_state_i32[k] = app.c_out;
                        int64_t eff_m  = app.tile_max;
                        if (lag_len > 0U) {
                            const uint64_t w_beg =
                                start > lag_len ? (start - lag_len) : 0ULL;
                            const auto lag_v = static_cast<int64_t>(
                                lag_state_i32[k].query_and_prune(w_beg));
                            eff_m = eff_m > lag_v ? eff_m : lag_v;
                            lag_state_i32[k].push(
                                tile_end,
                                kadane::clamp_kadane_i32(app.tile_max));
                        }
                        if (gate.disabled_row[row_index] == 0U &&
                            eff_m >= static_cast<int64_t>(row_icut[k])) {
                            my_hit = true;
                        }
                    } else {
                        const float c_in =
                            has_cold_start ? 0.0F : c_state_f32[k];
                        const auto app = kadane::kadane_apply(c_in, total_seg);
                        c_state_f32[k] = app.c_out;
                        float eff_m    = app.tile_max;
                        if (lag_len > 0U) {
                            const uint64_t w_beg =
                                start > lag_len ? (start - lag_len) : 0ULL;
                            const float lag_v =
                                lag_state_f32[k].query_and_prune(w_beg);
                            eff_m = kadane::monoid_max_f32(eff_m, lag_v);
                            lag_state_f32[k].push(tile_end, app.tile_max);
                        }
                        if (gate.disabled_row[row_index] == 0U &&
                            eff_m >= row_cuts_f32[k]) {
                            my_hit = true;
                        }
                    }
                }
            }
        }

        if (__syncthreads_or(my_hit || open) != 0) {
            if (!ring_ready) {
                if (shared_tables && !tables_staged) {
                    stage_shared_tables();
                }
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
                    const auto carry0 =
                        static_cast<uint32_t>(input.sample_begin % max_width);
                    const auto width32 = static_cast<uint32_t>(max_width);
                    for (uint32_t p = 0; p < lookback; p += kBlockTile) {
                        const T base_c = ring[p & mask];
                        T cx[kItems];
                        for (int j = 0; j < kItems; ++j) {
                            const uint32_t t = p + (tid * kItems) + j;
                            const uint32_t back =
                                t < lookback ? lookback - t : 0U;
                            cx[j] = t < lookback && input.sample_begin >= back
                                        ? carry[carry0 >= back
                                                    ? carry0 - back
                                                    : carry0 + width32 - back]
                                        : T{0};
                        }
                        prefix_tile<T, kBlockThreads>(shared, ring, mask, cx, p,
                                                      base_c);
                    }
                }
                ring_ready    = true;
                ring_synced_t = 0;
            }

            while (ring_synced_t < t0) {
                const auto pos_tau =
                    static_cast<uint32_t>(lookback + ring_synced_t);
                const T base_tau = ring[pos_tau & mask];
                T prev_x[kItems];
#pragma unroll
                for (int j = 0; j < kItems; ++j) {
                    const SizeType local =
                        static_cast<SizeType>(tid * kItems + j);
                    if constexpr (std::is_integral_v<T>) {
                        prev_x[j] = boxcar_integer_sample(
                            input, row, beam, ring_synced_t + local, root);
                    } else {
                        prev_x[j] = boxcar_sample(input, row, beam,
                                                  ring_synced_t + local, root);
                    }
                }
                prefix_tile<T, kBlockThreads>(shared, ring, mask, prev_x,
                                              pos_tau, base_tau);
                const uint32_t end_tau =
                    pos_tau + static_cast<uint32_t>(kBlockTile);
                const T origin_tau = ring[(end_tau - lookback) & mask];
                __syncthreads();
                for (uint32_t j = tid; j <= lookback; j += kBlockThreads) {
                    ring[(end_tau - lookback + j) & mask] -= origin_tau;
                }
                __syncthreads();
                ring_synced_t += kBlockTile;
            }

            const auto position = static_cast<uint32_t>(lookback + t0);
            const T base        = ring[position & mask];
            prefix_tile<T, kBlockThreads>(shared, ring, mask, x, position,
                                          base);

            float best[kItems];
            uint32_t best_filter[kItems];
            T current[kItems];
            uint64_t raw_sums[kItems]{};
            for (int j = 0; j < kItems; ++j) {
                current[j] = ring[(position + (tid * kItems) + j + 1) & mask];
                best[j]    = -1.0F;
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
                    const uint32_t local  = (tid * kItems) + j;
                    const uint64_t sample = start + local;
                    if constexpr (!std::is_integral_v<T>) {
                        if (mixed) {
                            const int e = block_entry(block_tables, sample);
                            const int k = block_table(block_tables, e, sample,
                                                      width, row_support);
                            const float* t = block_tables.thr[e][k];
                            const float* v = block_tables.inv[e][k];
                            threshold      = t != nullptr
                                                 ? t[(row_index * nfilters) + f]
                                                 : plane_thr[f];
                            scale = v != nullptr ? v[(row_index * nfilters) + f]
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
                                 means[(row_index * nfilters) + f]) *
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
                    if constexpr (kKeep) {
                        keep_drain(shared, pending, keeper);
                    }
                }
                if constexpr (kKeep) {
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

            if (t0 + kBlockTile < input.nsamps) {
                const uint32_t end = position + static_cast<uint32_t>(n);
                const T origin     = ring[(end - lookback) & mask];
                __syncthreads();
                for (uint32_t j = tid; j <= lookback; j += kBlockThreads) {
                    ring[(end - lookback + j) & mask] -= origin;
                }
                __syncthreads();
                ring_synced_t = t0 + kBlockTile;
            }
        }
    }

    if (tid == kLeaderTid) {
        const SizeType state_off = index * nb;
        for (SizeType k = 0; k < nb; ++k) {
            if constexpr (kUseIntGate) {
                gate.state_ic[state_off + k] = c_state_i32[k];
                if (gate.any_edge_lag) {
                    gate.lag_i32[state_off + k] = lag_state_i32[k];
                }
            } else {
                gate.state_fc[state_off + k] = c_state_f32[k];
                if (gate.any_edge_lag) {
                    gate.lag_f32[state_off + k] = lag_state_f32[k];
                }
            }
        }
    }
    if (tid == 0) {
        open_all[index] = open ? 1 : 0;
    }

    // Exact raw carry update directly from input.
    const SizeType keep = input.nsamps < max_width ? input.nsamps : max_width;
    const auto keep0    = static_cast<uint32_t>(
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

void FDMTBoxcarCudaEngine::populate_row_int_m1(std::span<const double> means) {
    if (!m_row_int_m1_d.empty()) {
        return;
    }
    const SizeType n_rows = m_rows.size();
    const uint32_t w0     = m_planes.front().widths[0];
    const uint32_t we0 = m_config.use_trapezoid ? trapezoid_edge_width(w0) : 0U;
    const double g0    = we0 == 0U ? static_cast<double>(w0)
                                   : static_cast<double>(2U * (w0 - we0));
    std::vector<double> row_m1(n_rows, 0.0);
    for (SizeType r = 0; r < n_rows; ++r) {
        row_m1[r] = means[r * m_filters] / g0;
    }
    m_row_int_m1_d = row_m1;
}

template <typename T, typename SegT>
SizeType FDMTBoxcarCudaEngine::kadane_shared_need(int threads) const {
    const auto need = [&]<int kBlockThreads>() {
        const SizeType ring =
            std::bit_ceil(m_width + SizeType{kBlockThreads * kItems});
        return sizeof(KadaneBlockShared<T, SegT, kBlockThreads>) +
               (ring * sizeof(T));
    };
    return threads == 64    ? need.template operator()<64>()
           : threads == 128 ? need.template operator()<128>()
                            : need.template operator()<256>();
}

template <typename T,
          typename SegT,
          int kBlockThreads,
          bool kUseIntGate,
          bool kBlockNoise,
          typename C>
void FDMTBoxcarCudaEngine::configure_kadane_variants() {
    configure_kernel(
        kernel_fdmt_boxcar_kadane<T, SegT, kBlockThreads, kUseIntGate,
                                  kBlockNoise, C, false>);
    if (m_bright_q != 0) {
        configure_kernel(
            kernel_fdmt_boxcar_kadane<T, SegT, kBlockThreads, kUseIntGate,
                                      kBlockNoise, C, true>);
    }
}

template <typename T, typename SegT, int kBlockThreads>
void FDMTBoxcarCudaEngine::configure_shared_kadane_block(SizeType limit) {
    constexpr SizeType kBlockTile = kBlockThreads * kItems;
    constexpr bool kUseIntGate    = std::is_same_v<SegT, kadane::KadaneSegI64>;
    const SizeType ring           = std::bit_ceil(m_width + kBlockTile);
    const SizeType fixed =
        sizeof(KadaneBlockShared<T, SegT, kBlockThreads>) + (ring * sizeof(T));
    const SizeType tables = m_filters * (sizeof(T) + sizeof(uint32_t));
    m_block_threads       = kBlockThreads;
    m_mask                = static_cast<uint32_t>(ring - 1);
    m_shared_tables = m_filters <= kSharedFilters && fixed + tables <= limit;
    m_smem          = fixed + (m_shared_tables ? tables : 0);
    // Design C SMEM halving: no s_screen range-minimum buffer allocated!
    m_screen_offset = 0;
    m_group_offset  = 0;
    if (m_smem > limit) {
        throw std::invalid_argument(fmt::format(
            "FDMTBoxcar (kKadane): max_boxcar {} needs {} B of shared memory "
            "per block; the device allows {} B",
            m_width, m_smem, limit));
    }
    configure_kadane_variants<T, SegT, kBlockThreads, kUseIntGate, false, T>();
    if constexpr (std::is_integral_v<T>) {
        configure_kadane_variants<T, SegT, kBlockThreads, kUseIntGate, false,
                                  uint32_t>();
    } else {
        configure_kadane_variants<T, SegT, kBlockThreads, kUseIntGate, true,
                                  T>();
    }
}

template <typename T, typename SegT>
bool FDMTBoxcarCudaEngine::try_configure_shared_kadane() {
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
    while (threads > 64 && kadane_shared_need<T, SegT>(threads) > limit) {
        threads /= 2;
        m_block_shrunk = true;
    }
    if (kadane_shared_need<T, SegT>(threads) > limit) {
        return false;
    }
    if (m_block_shrunk) {
        logging::debug("FDMTBoxcar (kKadane): max_boxcar {} in {} B of shared "
                       "memory: {}-thread blocks",
                       m_width, limit, threads);
    }
    if (threads == 64) {
        configure_shared_kadane_block<T, SegT, 64>(limit);
    } else if (threads == 128) {
        configure_shared_kadane_block<T, SegT, 128>(limit);
    } else {
        configure_shared_kadane_block<T, SegT, 256>(limit);
    }
    return true;
}

void FDMTBoxcarCudaEngine::init_kadane(bool use_box_smearing,
                                       std::span<const double> means,
                                       std::span<const uint64_t> cutoffs) {
    // Wide boxcars (> 4096) use the device-memory prefix-ring path directly.
    if (m_width > kFastMaxWidth) {
        m_kadane_fallback = true;
        if (m_config.input_nbits == 32) {
            configure_shared<float>();
        } else {
            configure_shared<uint64_t>();
        }
        return;
    }

    const SizeType n_rows  = m_rows.size();
    const SizeType total_f = n_rows * m_filters;
    std::vector<float> all_raw_thresh;
    all_raw_thresh.reserve(total_f);
    for (const auto& p : m_planes) {
        all_raw_thresh.insert(all_raw_thresh.end(), p.raw_thresh_lut.begin(),
                              p.raw_thresh_lut.end());
    }

    if (m_config.input_nbits != 32) {
        std::vector<float> eff_raw_thresh = all_raw_thresh;
        for (SizeType i = 0; i < total_f; ++i) {
            if (cutoffs[i] != UINT64_MAX) {
                const double centered =
                    (static_cast<double>(cutoffs[i]) - means[i]) *
                    m_integer_unit;
                if (centered > 0.0) {
                    eff_raw_thresh[i] = std::min(eff_raw_thresh[i],
                                                 static_cast<float>(centered));
                }
            }
        }
        const uint64_t max_int_sample = max_int_search_sample(
            m_plan, use_box_smearing, m_config.input_nbits);
        m_kadane_plan = kadane::make_kadane_gate_plan(
            m_planes.front().widths, m_config.use_trapezoid, n_rows,
            eff_raw_thresh, means, cutoffs, m_carry32, max_int_sample);
        m_kadane_int32 = m_carry32 && m_kadane_plan.has_int32;
    } else {
        m_kadane_plan  = kadane::make_kadane_gate_plan(m_planes.front().widths,
                                                       m_config.use_trapezoid,
                                                       n_rows, all_raw_thresh);
        m_kadane_int32 = false;
    }

    const bool configured =
        m_kadane_int32
            ? try_configure_shared_kadane<uint64_t, kadane::KadaneSegI64>()
        : m_config.input_nbits == 32
            ? try_configure_shared_kadane<float, kadane::KadaneSegF32>()
            : try_configure_shared_kadane<uint64_t, kadane::KadaneSegF32>();
    if (!configured) {
        // Tiny GPU SMEM cannot hold even a 64-thread KadaneBlockShared:
        // fall back cleanly to the unscreened / device-ring prefix scorer.
        m_kadane_fallback = true;
        m_block_shrunk    = false;
        m_kadane_plan     = {};
        m_kadane_int32    = false;
        if (m_config.input_nbits == 32) {
            configure_shared<float>();
        } else {
            configure_shared<uint64_t>();
        }
        return;
    }

    if (m_config.input_nbits != 32 && !m_kadane_int32) {
        populate_row_int_m1(means);
    }

    const SizeType nb    = m_kadane_plan.num_buckets;
    const SizeType tasks = m_rows.size() * m_beams;
    m_any_edge_lag       = false;
    for (SizeType k = 0; k < nb; ++k) {
        if (m_kadane_plan.max_edge_lag[k] > 0U) {
            m_any_edge_lag = true;
        }
    }

    m_kadane_disabled_d = m_kadane_plan.disabled_row;
    if (m_kadane_int32) {
        m_kadane_ibias_d = m_kadane_plan.ibias;
        m_kadane_icut_d  = m_kadane_plan.icut;
        m_kadane_ic_d.resize(tasks * nb, 0);
        if (m_any_edge_lag) {
            m_lag_i32_d.resize(tasks * nb);
            gpu_utils::check_gpu_call(cudaMemset(
                raw(m_lag_i32_d), 0,
                m_lag_i32_d.size() * sizeof(kadane::LagWindow<int32_t>)));
        }
    } else {
        m_kadane_bias_d = m_kadane_plan.bias;
        m_kadane_cut_d  = m_kadane_plan.cut;
        m_kadane_fc_d.resize(tasks * nb, 0.0F);
        if (m_any_edge_lag) {
            m_lag_f32_d.resize(tasks * nb);
            gpu_utils::check_gpu_call(cudaMemset(
                raw(m_lag_f32_d), 0,
                m_lag_f32_d.size() * sizeof(kadane::LagWindow<float>)));
        }
    }
}

void FDMTBoxcarCudaEngine::reset_kadane(cudaStream_t stream) {
    if (!m_kadane_fc_d.empty()) {
        gpu_utils::check_gpu_call(
            cudaMemsetAsync(raw(m_kadane_fc_d), 0,
                            m_kadane_fc_d.size() * sizeof(float), stream));
    }
    if (!m_kadane_ic_d.empty()) {
        gpu_utils::check_gpu_call(
            cudaMemsetAsync(raw(m_kadane_ic_d), 0,
                            m_kadane_ic_d.size() * sizeof(int32_t), stream));
    }
    if (!m_lag_f32_d.empty()) {
        gpu_utils::check_gpu_call(cudaMemsetAsync(
            raw(m_lag_f32_d), 0,
            m_lag_f32_d.size() * sizeof(kadane::LagWindow<float>), stream));
    }
    if (!m_lag_i32_d.empty()) {
        gpu_utils::check_gpu_call(cudaMemsetAsync(
            raw(m_lag_i32_d), 0,
            m_lag_i32_d.size() * sizeof(kadane::LagWindow<int32_t>), stream));
    }
}

kadane::KadaneGateDeviceView FDMTBoxcarCudaEngine::make_kadane_view() noexcept {
    kadane::KadaneGateDeviceView view;
    view.num_buckets  = m_kadane_plan.num_buckets;
    view.row_stride   = m_kadane_plan.row_stride;
    view.int_shift    = m_kadane_plan.int_shift;
    view.has_int32    = m_kadane_int32;
    view.any_edge_lag = m_any_edge_lag;
    for (SizeType k = 0; k < m_kadane_plan.num_buckets; ++k) {
        view.buckets[k]      = m_kadane_plan.buckets[k];
        view.max_edge_lag[k] = m_kadane_plan.max_edge_lag[k];
    }
    view.bias         = raw(m_kadane_bias_d);
    view.cut          = raw(m_kadane_cut_d);
    view.ibias        = raw(m_kadane_ibias_d);
    view.icut         = raw(m_kadane_icut_d);
    view.disabled_row = raw(m_kadane_disabled_d);
    view.row_int_m1   = raw(m_row_int_m1_d);
    view.state_fc     = raw(m_kadane_fc_d);
    view.state_ic     = raw(m_kadane_ic_d);
    view.lag_f32      = raw(m_lag_f32_d);
    view.lag_i32      = raw(m_lag_i32_d);
    return view;
}

template <typename T, typename C>
void FDMTBoxcarCudaEngine::launch_score_kadane(const FDMTSearchInput& input,
                                               const LevelGroup& g,
                                               dim3 grid,
                                               dim3 block,
                                               C* carry) {
    const auto stream       = static_cast<cudaStream_t>(input.stream.native);
    FDMTSearchInput stopped = input;
    stopped.search_stop     = m_search_stop;
    const auto gate         = make_kadane_view();
    const bool root         = g.level == m_plan.get_niters();

    const auto dispatch_keep = [&](auto seg_tag, auto int_gate_tag,
                                   auto noise_tag, auto keep_tag) {
        using SegT                 = decltype(seg_tag);
        constexpr bool kUseIntGate = decltype(int_gate_tag)::value;
        constexpr bool kNoise      = decltype(noise_tag)::value;
        constexpr bool kKeep       = decltype(keep_tag)::value;
        const BlockTables bt       = kNoise ? m_block_tables : BlockTables{};
        if (m_block_threads == 64) {
            kernel_fdmt_boxcar_kadane<T, SegT, 64, kUseIntGate, kNoise, C,
                                      kKeep><<<grid, block, m_smem, stream>>>(
                stopped, raw(m_rows_d), g.row_begin, g.row_count, m_rows.size(),
                raw(m_widths_d), m_filters, m_width, raw(m_thresholds_d),
                raw(m_scales_d), carry, raw(m_peaks_d), raw(m_open_d),
                m_peak_count, raw(m_candidates_d), raw(m_counts_d),
                raw(m_slots_d), raw(m_per_dm_d), m_config.max_candidates,
                m_config.max_cands_per_dm, m_config.peak_mode,
                m_config.skip_cold_prefix, root, raw(m_integer_thresholds),
                raw(m_integer_means), m_integer_unit, m_mask, m_shared_tables,
                m_config.use_trapezoid, gate, bt, raw(m_bright_d), m_bright_q);
        } else if (m_block_threads == 128) {
            kernel_fdmt_boxcar_kadane<T, SegT, 128, kUseIntGate, kNoise, C,
                                      kKeep><<<grid, block, m_smem, stream>>>(
                stopped, raw(m_rows_d), g.row_begin, g.row_count, m_rows.size(),
                raw(m_widths_d), m_filters, m_width, raw(m_thresholds_d),
                raw(m_scales_d), carry, raw(m_peaks_d), raw(m_open_d),
                m_peak_count, raw(m_candidates_d), raw(m_counts_d),
                raw(m_slots_d), raw(m_per_dm_d), m_config.max_candidates,
                m_config.max_cands_per_dm, m_config.peak_mode,
                m_config.skip_cold_prefix, root, raw(m_integer_thresholds),
                raw(m_integer_means), m_integer_unit, m_mask, m_shared_tables,
                m_config.use_trapezoid, gate, bt, raw(m_bright_d), m_bright_q);
        } else {
            kernel_fdmt_boxcar_kadane<T, SegT, 256, kUseIntGate, kNoise, C,
                                      kKeep><<<grid, block, m_smem, stream>>>(
                stopped, raw(m_rows_d), g.row_begin, g.row_count, m_rows.size(),
                raw(m_widths_d), m_filters, m_width, raw(m_thresholds_d),
                raw(m_scales_d), carry, raw(m_peaks_d), raw(m_open_d),
                m_peak_count, raw(m_candidates_d), raw(m_counts_d),
                raw(m_slots_d), raw(m_per_dm_d), m_config.max_candidates,
                m_config.max_cands_per_dm, m_config.peak_mode,
                m_config.skip_cold_prefix, root, raw(m_integer_thresholds),
                raw(m_integer_means), m_integer_unit, m_mask, m_shared_tables,
                m_config.use_trapezoid, gate, bt, raw(m_bright_d), m_bright_q);
        }
    };

    const auto dispatch_threads = [&](auto seg_tag, auto int_gate_tag,
                                      auto noise_tag) {
        if (m_bright_q != 0) {
            dispatch_keep(seg_tag, int_gate_tag, noise_tag,
                          cuda::std::true_type{});
        } else {
            dispatch_keep(seg_tag, int_gate_tag, noise_tag,
                          cuda::std::false_type{});
        }
    };

    if constexpr (std::is_integral_v<T>) {
        if (m_kadane_int32) {
            dispatch_threads(kadane::KadaneSegI64{}, cuda::std::true_type{},
                             cuda::std::false_type{});
        } else {
            dispatch_threads(kadane::KadaneSegF32{}, cuda::std::false_type{},
                             cuda::std::false_type{});
        }
    } else if (m_block_tables.count > 0) {
        dispatch_threads(kadane::KadaneSegF32{}, cuda::std::false_type{},
                         cuda::std::true_type{});
    } else {
        dispatch_threads(kadane::KadaneSegF32{}, cuda::std::false_type{},
                         cuda::std::false_type{});
    }
}

template void FDMTBoxcarCudaEngine::launch_score_kadane<float, float>(
    const FDMTSearchInput&, const LevelGroup&, dim3, dim3, float*);
template void FDMTBoxcarCudaEngine::launch_score_kadane<uint64_t, uint64_t>(
    const FDMTSearchInput&, const LevelGroup&, dim3, dim3, uint64_t*);
template void FDMTBoxcarCudaEngine::launch_score_kadane<uint64_t, uint32_t>(
    const FDMTSearchInput&, const LevelGroup&, dim3, dim3, uint32_t*);

} // namespace dmt::algorithms::detail::fdmt_boxcar_cuda
