#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include <thrust/device_vector.h>

#include "dmt/boxcar.hpp"
#include "dmt/boxcar_runs.hpp"
#include "dmt/fdmt_int_tree.hpp"
#include "dmt/gpu_compat.cuh"
#include "dmt/gpu_utils.cuh"
#include "dmt/kadane_cuda.cuh"
#include "dmt/kadane_monoid.hpp"
#include "dmt/kadane_plan.hpp"
#include "dmt/kadane_surrogate.hpp"

#include "fdmt_boxcar_cuda_engine.cuh"

// GPU FDMTBoxcar engine: standalone multi-strip Kadane surrogate scorer
// (kKadaneSurrogate), warp-per-row streaming kernel, kKeepBrightest per-row
// heap emission, frontier, and flush.

namespace dmt::algorithms::detail::fdmt_boxcar_cuda {
namespace {

inline constexpr int kSurrWarpsPerBlock = 4;
inline constexpr int kSurrBlockThreads  = kSurrWarpsPerBlock * kWarp;

struct SurrogateSubrangeTables {
    const float* thr = nullptr;
    const float* inv = nullptr;
    bool mixed       = false;
};

struct SurrogateEmitCtx {
    RawCandidate* candidates;
    CandidateCounts* counts;
    uint64_t* slots;
    uint64_t* per_dm;
    SizeType capacity;
    SizeType dm_cap;
    RawCandidate prototype;
    const uint32_t* widths;
    uint32_t max_width;
    BoxcarKeepEntry* keep_store;
    SizeType keep_q;
};

DMT_D inline void emit_surrogate_raw_gpu(const SurrogateEmitCtx& out,
                                         uint64_t begin_sample,
                                         uint64_t end_sample,
                                         uint64_t peak_sample,
                                         float snr,
                                         uint32_t filter_idx,
                                         uint32_t width) {
    const uint32_t w_lo      = width < 1U ? 1U : width;
    const uint32_t clamped_w = w_lo > out.max_width ? out.max_width : w_lo;
    increment(&out.counts->total_detected);
    if (out.keep_q != 0) {
        keep_insert(out.keep_store, out.per_dm, out.keep_q,
                    {peak_sample, begin_sample, end_sample, snr,
                     pack_keep_filter(filter_idx, clamped_w)},
                    out.counts);
        return;
    }
    if (out.dm_cap != 0 && increment(out.per_dm) >= out.dm_cap) {
        increment(&out.counts->dropped_per_dm);
        return;
    }
    const uint64_t slot = increment(out.slots);
    if (slot >= out.capacity) {
        increment(&out.counts->dropped_global);
        return;
    }
    RawCandidate candidate = out.prototype;
    candidate.peak_sample  = peak_sample;
    candidate.begin_sample = begin_sample;
    candidate.end_sample   = end_sample;
    candidate.snr          = snr;
    candidate.filter_idx   = filter_idx;
    candidate.width        = clamped_w;
    out.candidates[slot]   = candidate;
    increment(&out.counts->emitted_count);
}

DMT_D inline void emit_surrogate_peak_gpu(const SurrogateEmitCtx& out,
                                          const BoxcarPeak& peak) {
    const uint32_t width = peak.raw_sum > 0U
                               ? static_cast<uint32_t>(peak.raw_sum)
                               : out.widths[peak.filter];
    emit_surrogate_raw_gpu(out, peak.begin, peak.end, peak.peak, peak.snr,
                           peak.filter, width);
}

DMT_D inline const float* active_surrogate_table_tag(const BlockTables& bt,
                                                     SizeType row_index,
                                                     const uint32_t* widths,
                                                     uint64_t sample,
                                                     uint32_t filter_idx) {
    if (bt.count == 0) {
        return nullptr;
    }
    if (sample < bt.first[0]) {
        return reinterpret_cast<const float*>(uintptr_t{1});
    }
    const int e             = block_entry(bt, sample);
    const uint64_t row_supp = bt.support[row_index];
    const int tbl = block_table(bt, e, sample, widths[filter_idx], row_supp);
    return bt.inv[e][tbl];
}

template <bool kBlockNoise>
DMT_D inline float
eval_surrogate_excursion_snr_gpu(const kadane::KadaneSurrogateDeviceView& surr,
                                 SizeType row_index,
                                 const uint32_t* widths,
                                 SizeType nfilters,
                                 const float* plane_thr,
                                 const float* plane_inv,
                                 const BlockTables& bt,
                                 const SurrogateSubrangeTables* sub_tbl,
                                 uint64_t sub_start,
                                 uint64_t sub_end,
                                 const kadane::SurrogateExcursion& ex,
                                 uint32_t& out_filter_idx) {
    if (ex.best_len == 0U) {
        out_filter_idx = 0U;
        return 0.0F;
    }
    if constexpr (kBlockNoise) {
        if (ex.has_snr) {
            out_filter_idx = ex.best_filter;
            return ex.best_snr;
        }
    }
    const float snr = surr.eval_surrogate_snr(
        row_index, ex.best_raw_sum, ex.best_len, widths[0], out_filter_idx);
    if constexpr (!kBlockNoise) {
        return snr;
    } else {
        if (bt.count == 0) {
            return snr;
        }
        const float* active_thr = plane_thr;
        const float* active_inv = plane_inv;
        if (sub_tbl != nullptr && !sub_tbl->mixed && ex.best_end >= sub_start &&
            ex.best_end < sub_end) {
            active_thr = sub_tbl->thr;
            active_inv = sub_tbl->inv;
        } else {
            const int e        = block_entry(bt, ex.best_end);
            const uint64_t sup = bt.support[row_index];
            const int tbl =
                block_table(bt, e, ex.best_end, widths[out_filter_idx], sup);
            const float* t_ptr = bt.thr[e][tbl];
            const float* v_ptr = bt.inv[e][tbl];
            active_thr =
                t_ptr != nullptr ? t_ptr + (row_index * nfilters) : plane_thr;
            active_inv =
                v_ptr != nullptr ? v_ptr + (row_index * nfilters) : plane_inv;
        }
        if (active_inv == plane_inv) {
            return snr;
        }
        const float thr_f = active_thr[out_filter_idx];
        const float inv_f = active_inv[out_filter_idx];
        if (!kadane::is_finite_monoid_f32(thr_f) || thr_f >= 1.0e35F ||
            inv_f <= 0.0F || plane_inv[out_filter_idx] <= 0.0F) {
            return 0.0F;
        }
        return snr * (inv_f / plane_inv[out_filter_idx]);
    }
}

template <bool kBlockNoise>
DMT_D inline void
cache_surrogate_pass_snr_gpu(const kadane::KadaneSurrogateDeviceView& surr,
                             SizeType row_index,
                             const uint32_t* widths,
                             SizeType nfilters,
                             const float* plane_thr,
                             const float* plane_inv,
                             const BlockTables& bt,
                             const SurrogateSubrangeTables* sub_tbl,
                             uint64_t sub_start,
                             uint64_t sub_end,
                             kadane::KadaneSurrogatePassState& st,
                             uint32_t pass_idx,
                             const kadane::KadaneSurrogatePassState* prev_st) {
    if (!st.open || st.best_len == 0U) {
        return;
    }
    if (!st.has_snr) {
        uint32_t filt   = 0U;
        const float snr = eval_surrogate_excursion_snr_gpu<kBlockNoise>(
            surr, row_index, widths, nfilters, plane_thr, plane_inv, bt,
            sub_tbl, sub_start, sub_end,
            kadane::detail::make_excursion(st, pass_idx), filt);
        st.best_snr    = snr;
        st.best_filter = filt;
        st.has_snr     = true;
    }
    if constexpr (kBlockNoise) {
        if (prev_st != nullptr && prev_st->open && prev_st->has_snr &&
            prev_st->seg_begin == st.seg_begin &&
            prev_st->best_snr > st.best_snr &&
            active_surrogate_table_tag(bt, row_index, widths, prev_st->best_end,
                                       prev_st->best_filter) !=
                active_surrogate_table_tag(bt, row_index, widths, st.best_end,
                                           st.best_filter)) {
            st.best_snr     = prev_st->best_snr;
            st.best_filter  = prev_st->best_filter;
            st.best_end     = prev_st->best_end;
            st.best_len     = prev_st->best_len;
            st.best_raw_sum = prev_st->best_raw_sum;
        }
    }
}

template <bool kBlockNoise>
DMT_D inline void
handle_surrogate_excursion_gpu(const kadane::KadaneSurrogateDeviceView& surr,
                               SizeType row_index,
                               SizeType nfilters,
                               float detect_thresh,
                               PeakFindingMode mode,
                               const float* plane_thr,
                               const float* plane_inv,
                               const BlockTables& bt,
                               const SurrogateSubrangeTables* sub_tbl,
                               uint64_t sub_start,
                               uint64_t sub_end,
                               const SurrogateEmitCtx& out,
                               BoxcarPeak& pending,
                               const kadane::KadaneSurrogateRowState& st,
                               const kadane::SurrogateExcursion& ex) {
    uint32_t filter_idx = 0U;
    const float snr     = eval_surrogate_excursion_snr_gpu<kBlockNoise>(
        surr, row_index, out.widths, nfilters, plane_thr, plane_inv, bt,
        sub_tbl, sub_start, sub_end, ex, filter_idx);
    if (snr < detect_thresh) {
        return;
    }
    const uint32_t w_lo      = ex.best_len < 1U ? 1U : ex.best_len;
    const uint32_t clamped_w = w_lo > out.max_width ? out.max_width : w_lo;
    if (mode != PeakFindingMode::kContiguousPeakAcrossWidths) {
        emit_surrogate_raw_gpu(out, ex.seg_begin, ex.close_end, ex.best_end,
                               snr, filter_idx, clamped_w);
        return;
    }

    if (pending.open) {
        const bool overlaps = (ex.seg_begin <= pending.end + 1ULL) &&
                              (pending.begin <= ex.close_end + 1ULL);
        if (overlaps) {
            if (ex.seg_begin < pending.begin) {
                pending.begin = ex.seg_begin;
            }
            if (ex.close_end > pending.end) {
                pending.end = ex.close_end;
            }
            if (boxcar_better(snr, ex.best_end, filter_idx, pending)) {
                pending.snr     = snr;
                pending.peak    = ex.best_end;
                pending.filter  = filter_idx;
                pending.raw_sum = clamped_w;
            }
            return;
        }
        bool other_overlaps = false;
        for (SizeType k = 0; k < surr.num_passes; ++k) {
            if (st.passes[k].open &&
                st.passes[k].seg_begin <= pending.end + 1ULL) {
                other_overlaps = true;
                break;
            }
        }
        if (!other_overlaps) {
            emit_surrogate_peak_gpu(out, pending);
            pending.open = false;
        } else {
            emit_surrogate_raw_gpu(out, ex.seg_begin, ex.close_end, ex.best_end,
                                   snr, filter_idx, clamped_w);
            return;
        }
    }
    pending.begin   = ex.seg_begin;
    pending.end     = ex.close_end;
    pending.peak    = ex.best_end;
    pending.snr     = snr;
    pending.filter  = filter_idx;
    pending.raw_sum = clamped_w;
    pending.open    = true;
}

DMT_D inline void
maybe_finalize_surrogate_pending_gpu(SizeType num_passes,
                                     PeakFindingMode mode,
                                     uint64_t tile_end,
                                     const kadane::KadaneSurrogateRowState& st,
                                     BoxcarPeak& pending,
                                     const SurrogateEmitCtx& out) {
    if (mode != PeakFindingMode::kContiguousPeakAcrossWidths || !pending.open) {
        return;
    }
    if (pending.end + 4ULL >= tile_end) {
        return;
    }
    for (SizeType k = 0; k < num_passes; ++k) {
        if (st.passes[k].open && st.passes[k].seg_begin <= pending.end + 1ULL) {
            return;
        }
    }
    emit_surrogate_peak_gpu(out, pending);
    pending.open = false;
}

DMT_D inline SurrogateSubrangeTables
compute_surrogate_subrange_warp(const kadane::KadaneSurrogateDeviceView& surr,
                                SizeType row_index,
                                const uint32_t* widths,
                                SizeType nfilters,
                                uint32_t max_width,
                                const float* plane_thr,
                                const float* plane_inv,
                                const BlockTables& bt,
                                uint64_t sub_s0,
                                uint64_t sub_s1,
                                int lane,
                                float (&r_trigs)[kadane::kMaxSurrogatePasses]) {
    SurrogateSubrangeTables out{plane_thr, plane_inv, false};
    if (bt.count == 0) {
        for (SizeType k = 0; k < surr.num_passes; ++k) {
            r_trigs[k] = surr.row_trigger(k, row_index);
        }
        return out;
    }
    const uint64_t row_supp = bt.support[row_index];
    const int e0            = block_entry(bt, sub_s0);
    const int e1            = block_entry(bt, sub_s1);
    const float* in_play[4][2];
    int n_play     = 0;
    const auto add = [&](int e, int tbl) {
        const float* t = bt.thr[e][tbl];
        const float* v = bt.inv[e][tbl];
        t              = t != nullptr ? t + (row_index * nfilters) : plane_thr;
        v              = v != nullptr ? v + (row_index * nfilters) : plane_inv;
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
        const uint64_t first = bt.first[e];
        const uint64_t lo    = e == e0 ? sub_s0 : first;
        const uint64_t hi    = e < e1 ? bt.first[e + 1] - 1ULL : sub_s1;
        if (!bt.transition[e]) {
            add(e, 0);
            continue;
        }
        if (lo < first + row_supp + max_width - 1ULL) {
            add(e, 1);
        }
        if (hi + 1ULL >= first + row_supp + widths[0]) {
            add(e, 0);
        }
    }
    if (n_play == 1) {
        out.thr   = in_play[0][0];
        out.inv   = in_play[0][1];
        out.mixed = false;
        if (out.thr == plane_thr) {
            for (SizeType k = 0; k < surr.num_passes; ++k) {
                r_trigs[k] = surr.row_trigger(k, row_index);
            }
            return out;
        }
    } else {
        out.thr   = plane_thr;
        out.inv   = plane_inv;
        out.mixed = true;
    }

    float local_ratio = 1.0F;
    for (SizeType f = static_cast<SizeType>(lane); f < nfilters;
         f += static_cast<SizeType>(kWarp)) {
        float th = in_play[0][0][f];
        for (int j = 1; j < n_play; ++j) {
            const float cand = in_play[j][0][f];
            th               = cand < th ? cand : th;
        }
        if (!kadane::is_finite_monoid_f32(th) || th <= 0.0F || th >= 1.0e35F) {
            local_ratio = -1.0F;
            break;
        }
        const float ctor_th = plane_thr[f];
        if (ctor_th > 0.0F) {
            const float r = th / ctor_th;
            if (r < local_ratio) {
                local_ratio = r;
            }
        }
    }
    const float min_r = kadane::warp_min_f32(local_ratio, lane);
    if (min_r < 0.0F) {
        for (SizeType k = 0; k < surr.num_passes; ++k) {
            r_trigs[k] = FLT_MAX;
        }
    } else {
        const float clamped = min_r > 1.0F ? 1.0F : min_r;
        for (SizeType k = 0; k < surr.num_passes; ++k) {
            r_trigs[k] = surr.row_trigger(k, row_index) * clamped;
        }
    }
    return out;
}

template <bool kIsPacked, bool kBlockNoise>
__global__ void __launch_bounds__(kSurrBlockThreads, 4)
    kernel_fdmt_boxcar_surrogate(FDMTSearchInput input,
                                 const BoxcarRow* rows,
                                 SizeType row_base,
                                 SizeType ndm,
                                 SizeType total_rows,
                                 SizeType total_tasks,
                                 const uint32_t* widths,
                                 SizeType nfilters,
                                 uint32_t max_width,
                                 float detect_thresh,
                                 const float* thresholds,
                                 const float* inv_sigma,
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
                                 kadane::KadaneSurrogateDeviceView surr,
                                 BlockTables block_tables,
                                 BoxcarKeepEntry* keep_all,
                                 SizeType keep_q) {
    const int tid_in_block = static_cast<int>(threadIdx.x);
    const int lane         = tid_in_block % kWarp;
    const SizeType task =
        (static_cast<SizeType>(blockIdx.x) * kSurrWarpsPerBlock) +
        static_cast<SizeType>(tid_in_block / kWarp);
    if (task >= total_tasks) {
        return;
    }

    const SizeType beam      = input.beam_begin + (task / ndm);
    const SizeType d         = task % ndm;
    const SizeType row_index = row_base + d;
    const SizeType index     = (beam * total_rows) + row_index;
    const BoxcarRow row      = rows[row_index];

    if (input.row_lag != nullptr) {
        FDMTSearchInput lagged;
        if (!lagged_row_input(input, row.row, lagged)) {
            return;
        }
        input = lagged;
    }
    if (!stop_row_input(input) || input.nsamps == 0) {
        return;
    }

    if (surr.disabled_row[row_index] != 0U) {
        if (lane == 0) {
            surr.row_pos[index] = input.sample_begin + input.nsamps;
        }
        return;
    }

    const SizeType np = surr.num_passes;
    uint32_t dec_factors[kadane::kMaxSurrogatePasses]{};
    float r_biases[kadane::kMaxSurrogatePasses]{};
    float r_trigs[kadane::kMaxSurrogatePasses]{};
    for (SizeType k = 0; k < np; ++k) {
        dec_factors[k] = surr.decimate_factors[k];
        r_biases[k]    = surr.row_bias(k, row_index);
        r_trigs[k]     = surr.row_trigger(k, row_index);
    }

    const float* plane_thr = thresholds + (row_index * nfilters);
    const float* plane_inv = inv_sigma + (row_index * nfilters);
    const double row_m1 =
        (kIsPacked && !surr.int_u32_direct) ? surr.row_int_m1[row_index] : 0.0;

    RawCandidate prototype = row.candidate;
    prototype.beam_idx     = static_cast<uint32_t>(beam);
    const SurrogateEmitCtx out_ctx{
        candidates, counts,
        slots,      per_dm_all + index,
        capacity,   dm_cap,
        prototype,  widths,
        max_width,  keep_q != 0 ? keep_all + (index * keep_q) : nullptr,
        keep_q};
    BoxcarPeak pending{};
    kadane::KadaneSurrogateRowState row_st{};
    if (lane == 0) {
        pending = peaks_all[index * peak_count];
        if (input.sample_begin > 0) {
            row_st = surr.row_states[index];
            if (!row_st.initialized) {
                row_st.reset();
            }
        }
    }

    // Check if the entire chunk uses a single non-split noise table regime so
    // we do not recompute triggers every 32-sample warp tile unless
    // mixed/split.
    SurrogateSubrangeTables chunk_tbl{plane_thr, plane_inv, false};
    bool need_per_tile_triggers = false;
    if constexpr (kBlockNoise && !kIsPacked) {
        if (block_tables.count > 0) {
            const uint64_t chunk_s0 = input.sample_begin;
            const uint64_t chunk_s1 = chunk_s0 + input.nsamps - 1ULL;
            chunk_tbl               = compute_surrogate_subrange_warp(
                surr, row_index, widths, nfilters, max_width, plane_thr,
                plane_inv, block_tables, chunk_s0, chunk_s1, lane, r_trigs);
            need_per_tile_triggers =
                chunk_tbl.mixed ||
                (block_tables.count == 2 && chunk_s0 < block_tables.first[1] &&
                 block_tables.first[1] <= chunk_s1);
        }
    }

    for (SizeType t0 = 0; t0 < input.nsamps;
         t0 += static_cast<SizeType>(kWarp)) {
        const SizeType rem    = input.nsamps - t0;
        const SizeType tile_n = rem < static_cast<SizeType>(kWarp)
                                    ? rem
                                    : static_cast<SizeType>(kWarp);

        float x_lane = 0.0F;
        if (static_cast<SizeType>(lane) < tile_n) {
            const SizeType sample_t = t0 + static_cast<SizeType>(lane);
            if constexpr (kIsPacked) {
                const uint64_t raw_u =
                    boxcar_integer_sample(input, row, beam, sample_t, root);
                if (surr.int_u32_direct) {
                    x_lane = static_cast<float>(static_cast<uint32_t>(raw_u));
                } else {
                    x_lane = static_cast<float>(
                        (static_cast<double>(raw_u) - row_m1) *
                        surr.integer_unit);
                }
            } else {
                x_lane = boxcar_sample(input, row, beam, sample_t, root);
            }
        }

        const uint64_t s0    = input.sample_begin + t0;
        const uint64_t s_end = s0 + tile_n;
        uint64_t cur_s       = s0;
        while (cur_s < s_end) {
            uint64_t cur_end                = s_end;
            SurrogateSubrangeTables sub_tbl = chunk_tbl;
            if constexpr (kBlockNoise && !kIsPacked) {
                if (block_tables.count == 2 && cur_s < block_tables.first[1] &&
                    block_tables.first[1] < s_end) {
                    cur_end = block_tables.first[1];
                }
                if (need_per_tile_triggers) {
                    sub_tbl = compute_surrogate_subrange_warp(
                        surr, row_index, widths, nfilters, max_width, plane_thr,
                        plane_inv, block_tables, cur_s, cur_end - 1ULL, lane,
                        r_trigs);
                }
            }

            const auto prev_passes = row_st.passes;
            const auto sub_off     = static_cast<SizeType>(cur_s - s0);
            const auto sub_end_off = static_cast<SizeType>(cur_end - s0);
            for (SizeType s = sub_off; s < sub_end_off; ++s) {
                const float x_s = gpu_compat::shfl(x_lane, static_cast<int>(s));
                if (lane == 0) {
                    const uint64_t gt   = s0 + s;
                    const bool is_valid = !skip_cold || gt >= row.warmup;
                    const auto emit_fn =
                        [&] DMT_D(kadane::SurrogateExcursion ex) {
                            if constexpr (kBlockNoise && !kIsPacked) {
                                if (!ex.has_snr) {
                                    uint32_t filt = 0U;
                                    const float snr =
                                        eval_surrogate_excursion_snr_gpu<true>(
                                            surr, row_index, widths, nfilters,
                                            plane_thr, plane_inv, block_tables,
                                            &sub_tbl, cur_s, cur_end, ex, filt);
                                    ex.best_snr      = snr;
                                    ex.best_filter   = filt;
                                    ex.has_snr       = true;
                                    const auto& prev = prev_passes[ex.pass_idx];
                                    if (prev.open && prev.has_snr &&
                                        prev.seg_begin == ex.seg_begin &&
                                        prev.best_snr > ex.best_snr &&
                                        active_surrogate_table_tag(
                                            block_tables, row_index, widths,
                                            prev.best_end, prev.best_filter) !=
                                            active_surrogate_table_tag(
                                                block_tables, row_index, widths,
                                                ex.best_end, ex.best_filter)) {
                                        ex.best_snr     = prev.best_snr;
                                        ex.best_filter  = prev.best_filter;
                                        ex.best_end     = prev.best_end;
                                        ex.best_len     = prev.best_len;
                                        ex.best_raw_sum = prev.best_raw_sum;
                                    }
                                }
                            }
                            handle_surrogate_excursion_gpu<kBlockNoise &&
                                                           !kIsPacked>(
                                surr, row_index, nfilters, detect_thresh, mode,
                                plane_thr, plane_inv, block_tables, &sub_tbl,
                                cur_s, cur_end, out_ctx, pending, row_st, ex);
                        };
                    kadane::detail::step_surrogate_row_sample(
                        gt, x_s, is_valid, np, dec_factors, surr.has_decimated,
                        r_biases, r_trigs, row_st, emit_fn);
                }
            }

            if constexpr (kBlockNoise && !kIsPacked) {
                if (lane == 0 && block_tables.count > 0) {
                    for (SizeType k = 0; k < np; ++k) {
                        cache_surrogate_pass_snr_gpu<true>(
                            surr, row_index, widths, nfilters, plane_thr,
                            plane_inv, block_tables, &sub_tbl, cur_s, cur_end,
                            row_st.passes[k], static_cast<uint32_t>(k),
                            &prev_passes[k]);
                    }
                }
            }
            cur_s = cur_end;
        }

        if (lane == 0 && ((((t0 + tile_n) & 1023U) == 0U) ||
                          (t0 + tile_n == input.nsamps))) {
            maybe_finalize_surrogate_pending_gpu(np, mode, s_end, row_st,
                                                 pending, out_ctx);
        }
    }

    if (lane == 0) {
        const uint64_t chunk_end = input.sample_begin + input.nsamps;
        // Cache constructor-table S/N on open excursions even before
        // `enable_block_noise` so block-0 open excursions retain calibrated S/N
        // if per-call masks are enabled lazily on block 1.
        if constexpr (!kBlockNoise && !kIsPacked) {
            for (SizeType k = 0; k < np; ++k) {
                cache_surrogate_pass_snr_gpu<false>(
                    surr, row_index, widths, nfilters, plane_thr, plane_inv,
                    block_tables, nullptr, 0, 0, row_st.passes[k],
                    static_cast<uint32_t>(k), nullptr);
            }
        }
        row_st.initialized            = true;
        row_st.synced_sample          = chunk_end;
        surr.row_states[index]        = row_st;
        surr.row_pos[index]           = chunk_end;
        peaks_all[index * peak_count] = pending;
        const bool trig_open          = row_st.has_triggered_open(np, r_trigs);
        const bool pend_open =
            (mode == PeakFindingMode::kContiguousPeakAcrossWidths) &&
            pending.open;
        open_all[index] = (trig_open || pend_open) ? uint8_t{1} : uint8_t{0};
    }
}

template <bool kBlockNoise>
__global__ void __launch_bounds__(kSurrBlockThreads, 4)
    kernel_flush_surrogate(const BoxcarRow* rows,
                           SizeType nrows,
                           SizeType total_tasks,
                           SizeType peak_count,
                           BoxcarPeak* peaks_all,
                           uint8_t* open_all,
                           const uint32_t* widths,
                           SizeType nfilters,
                           uint32_t max_width,
                           float detect_thresh,
                           PeakFindingMode mode,
                           const float* thresholds,
                           const float* inv_sigma,
                           RawCandidate* candidates,
                           CandidateCounts* counts,
                           uint64_t* slots,
                           uint64_t* per_dm_all,
                           SizeType capacity,
                           SizeType dm_cap,
                           kadane::KadaneSurrogateDeviceView surr,
                           BlockTables block_tables,
                           BoxcarKeepEntry* keep_all,
                           SizeType keep_q) {
    const int tid_in_block = static_cast<int>(threadIdx.x);
    const int lane         = tid_in_block % kWarp;
    const SizeType index =
        (static_cast<SizeType>(blockIdx.x) * kSurrWarpsPerBlock) +
        static_cast<SizeType>(tid_in_block / kWarp);
    if (index >= total_tasks) {
        return;
    }

    const SizeType row_index = index % nrows;
    const SizeType beam      = index / nrows;
    if (surr.disabled_row[row_index] != 0U) {
        if (lane == 0) {
            peaks_all[index * peak_count].open = false;
            open_all[index]                    = 0U;
            surr.row_pos[index]                = 0ULL;
        }
        return;
    }

    const uint64_t pos     = surr.row_pos[index];
    const float* plane_thr = thresholds + (row_index * nfilters);
    const float* plane_inv = inv_sigma + (row_index * nfilters);
    float r_trigs[kadane::kMaxSurrogatePasses]{};
    SurrogateSubrangeTables sub_tbl{plane_thr, plane_inv, false};
    const SurrogateSubrangeTables* tbl_ptr = nullptr;
    if constexpr (kBlockNoise) {
        if (block_tables.count > 0 && pos > 0ULL) {
            sub_tbl = compute_surrogate_subrange_warp(
                surr, row_index, widths, nfilters, max_width, plane_thr,
                plane_inv, block_tables, pos - 1ULL, pos - 1ULL, lane, r_trigs);
            tbl_ptr = &sub_tbl;
        } else {
            for (SizeType k = 0; k < surr.num_passes; ++k) {
                r_trigs[k] = surr.row_trigger(k, row_index);
            }
        }
    } else {
        for (SizeType k = 0; k < surr.num_passes; ++k) {
            r_trigs[k] = surr.row_trigger(k, row_index);
        }
    }

    if (lane == 0) {
        RawCandidate prototype = rows[row_index].candidate;
        prototype.beam_idx     = static_cast<uint32_t>(beam);
        const SurrogateEmitCtx out_ctx{
            candidates, counts,
            slots,      per_dm_all + index,
            capacity,   dm_cap,
            prototype,  widths,
            max_width,  keep_q != 0 ? keep_all + (index * keep_q) : nullptr,
            keep_q};
        BoxcarPeak pending = peaks_all[index * peak_count];
        auto row_st        = surr.row_states[index];
        if (!row_st.initialized) {
            row_st.reset();
        }
        const uint64_t fl_s = pos > 0ULL ? (pos - 1ULL) : 0ULL;
        const auto emit_fn  = [&] DMT_D(const kadane::SurrogateExcursion& ex) {
            handle_surrogate_excursion_gpu<kBlockNoise>(
                surr, row_index, nfilters, detect_thresh, mode, plane_thr,
                plane_inv, block_tables, tbl_ptr, fl_s, pos, out_ctx, pending,
                row_st, ex);
        };
        kadane::flush_surrogate_row(surr.num_passes, r_trigs, row_st, emit_fn);
        if (pending.open) {
            emit_surrogate_peak_gpu(out_ctx, pending);
            pending.open = false;
        }
        row_st.reset();
        surr.row_states[index]        = row_st;
        surr.row_pos[index]           = 0ULL;
        peaks_all[index * peak_count] = pending;
        open_all[index]               = 0U;
    }
}

__global__ void
kernel_open_begin_surrogate(const BoxcarRow* rows,
                            SizeType nrows,
                            SizeType tasks,
                            SizeType peak_count,
                            PeakFindingMode mode,
                            bool skip_cold,
                            const BoxcarPeak* peaks_all,
                            kadane::KadaneSurrogateDeviceView surr,
                            uint64_t* oldest) {
    uint64_t local = UINT64_MAX;
    const SizeType stride =
        static_cast<SizeType>(gridDim.x) * static_cast<SizeType>(blockDim.x);
    for (SizeType idx =
             (static_cast<SizeType>(blockIdx.x) * blockDim.x) + threadIdx.x;
         idx < tasks; idx += stride) {
        const SizeType r = idx % nrows;
        if (surr.disabled_row[r] != 0U) {
            continue;
        }
        const uint64_t pos = surr.row_pos[idx];
        if (pos > 0ULL) {
            const auto& st = surr.row_states[idx];
            if (st.initialized) {
                for (SizeType k = 0; k < surr.num_passes; ++k) {
                    if (st.passes[k].open && st.passes[k].cur_c > 0.0F &&
                        st.passes[k].seg_begin < local) {
                        local = st.passes[k].seg_begin;
                    }
                }
            }
            const uint64_t eff_wu = skip_cold ? rows[r].warmup : 0ULL;
            if (surr.has_decimated && (pos & 3ULL) != 0ULL && pos > eff_wu) {
                const uint64_t blk_beg = pos & ~uint64_t{3};
                const uint64_t cand    = eff_wu > blk_beg ? eff_wu : blk_beg;
                if (cand < local) {
                    local = cand;
                }
            }
        }
        if (mode == PeakFindingMode::kContiguousPeakAcrossWidths) {
            const BoxcarPeak& pend = peaks_all[idx * peak_count];
            if (pend.open && pend.begin < local) {
                local = pend.begin;
            }
        }
    }
    if (local != UINT64_MAX) {
        atomicMin(reinterpret_cast<unsigned long long*>(oldest),
                  static_cast<unsigned long long>(local));
    }
}

} // namespace

void FDMTBoxcarCudaEngine::init_surrogate(bool use_box_smearing,
                                          std::span<const double> means) {
    const SizeType n_rows  = m_rows.size();
    const SizeType total_f = n_rows * m_filters;
    std::vector<float> all_raw_thresh;
    std::vector<float> all_inv_sigma;
    all_raw_thresh.reserve(total_f);
    all_inv_sigma.reserve(total_f);
    for (const auto& p : m_planes) {
        all_raw_thresh.insert(all_raw_thresh.end(), p.raw_thresh_lut.begin(),
                              p.raw_thresh_lut.end());
        all_inv_sigma.insert(all_inv_sigma.end(), p.inv_sigma_lut.begin(),
                             p.inv_sigma_lut.end());
    }

    if (m_config.input_nbits == 32) {
        m_surrogate_int_u32 = false;
        m_surrogate_plan    = kadane::make_kadane_surrogate_plan(
            m_planes.front().widths, m_config.use_trapezoid, n_rows,
            all_raw_thresh, all_inv_sigma);
    } else {
        const uint64_t max_int_sample = max_int_search_sample(
            m_plan, use_box_smearing, m_config.input_nbits);
        if (m_carry32 && max_int_sample <= (uint64_t{1} << 20)) {
            m_surrogate_int_u32 = true;
            m_surrogate_plan    = kadane::make_kadane_surrogate_plan(
                m_planes.front().widths, m_config.use_trapezoid, n_rows,
                all_raw_thresh, all_inv_sigma, means, m_integer_unit);
        } else {
            m_surrogate_int_u32 = false;
            populate_row_int_m1(means);
            m_surrogate_plan = kadane::make_kadane_surrogate_plan(
                m_planes.front().widths, m_config.use_trapezoid, n_rows,
                all_raw_thresh, all_inv_sigma);
        }
    }

    m_surr_bias_d     = m_surrogate_plan.bias;
    m_surr_trig_d     = m_surrogate_plan.trigger;
    m_surr_box_inv_d  = m_surrogate_plan.box_inv_sigma_lut;
    m_surr_mean1_d    = m_surrogate_plan.row_mean_1;
    m_surr_disabled_d = m_surrogate_plan.disabled_row;
    m_surr_nearest_d  = m_surrogate_plan.width_nearest_filter;
    m_surr_blo_d      = m_surrogate_plan.width_bracket_lo;
    m_surr_bhi_d      = m_surrogate_plan.width_bracket_hi;
    m_surr_lerp_d     = m_surrogate_plan.width_log_lerp;

    const SizeType tasks = m_rows.size() * m_beams;
    m_surrogate_rows_d.resize(tasks);
    m_surrogate_pos_d.assign(tasks, uint64_t{0});
    m_smem          = 0;
    m_block_threads = kSurrBlockThreads;
}

void FDMTBoxcarCudaEngine::reset_surrogate(cudaStream_t stream) {
    if (!m_surrogate_rows_d.empty()) {
        gpu_utils::check_gpu_call(cudaMemsetAsync(
            raw(m_surrogate_rows_d), 0,
            m_surrogate_rows_d.size() * sizeof(kadane::KadaneSurrogateRowState),
            stream));
    }
    if (!m_surrogate_pos_d.empty()) {
        gpu_utils::check_gpu_call(cudaMemsetAsync(
            raw(m_surrogate_pos_d), 0,
            m_surrogate_pos_d.size() * sizeof(uint64_t), stream));
    }
}

kadane::KadaneSurrogateDeviceView
FDMTBoxcarCudaEngine::make_surrogate_view() noexcept {
    kadane::KadaneSurrogateDeviceView view{};
    view.num_passes     = m_surrogate_plan.num_passes;
    view.n_rows         = m_surrogate_plan.n_rows;
    view.row_stride     = m_surrogate_plan.row_stride;
    view.n_filters      = m_filters;
    view.max_width      = m_surrogate_plan.max_width;
    view.has_decimated  = m_surrogate_plan.has_decimated;
    view.is_integer     = m_surrogate_plan.is_integer;
    view.int_u32_direct = m_surrogate_int_u32;
    view.integer_unit   = m_integer_unit;
    for (SizeType k = 0; k < m_surrogate_plan.num_passes; ++k) {
        view.decimate_factors[k] = m_surrogate_plan.passes[k].decimate_factor;
    }
    view.bias          = raw(m_surr_bias_d);
    view.trigger       = raw(m_surr_trig_d);
    view.box_inv_sigma = raw(m_surr_box_inv_d);
    view.row_mean_1    = m_surr_mean1_d.empty() ? nullptr : raw(m_surr_mean1_d);
    view.row_int_m1    = m_row_int_m1_d.empty() ? nullptr : raw(m_row_int_m1_d);
    view.disabled_row  = raw(m_surr_disabled_d);
    view.w_nearest     = raw(m_surr_nearest_d);
    view.w_bracket_lo  = raw(m_surr_blo_d);
    view.w_bracket_hi  = raw(m_surr_bhi_d);
    view.w_log_lerp    = raw(m_surr_lerp_d);
    view.row_states    = raw(m_surrogate_rows_d);
    view.row_pos       = raw(m_surrogate_pos_d);
    return view;
}

void FDMTBoxcarCudaEngine::launch_score_surrogate(const FDMTSearchInput& input,
                                                  const LevelGroup& g,
                                                  SizeType beams) {
    const SizeType total_tasks = g.row_count * beams;
    if (total_tasks == 0 || input.nsamps == 0) {
        return;
    }
    const dim3 grid(static_cast<unsigned>(
        (total_tasks + kSurrWarpsPerBlock - 1) / kSurrWarpsPerBlock));
    const dim3 block(static_cast<unsigned>(kSurrBlockThreads));
    gpu_utils::check_kernel_launch_params(grid, block);
    const auto stream       = static_cast<cudaStream_t>(input.stream.native);
    FDMTSearchInput stopped = input;
    stopped.search_stop     = m_search_stop;
    const auto surr_view    = make_surrogate_view();
    const auto max_w        = static_cast<uint32_t>(m_width);
    const bool root         = g.level == m_plan.get_niters();

    const auto launch = [&]<bool kPacked, bool kNoise>(const BlockTables& bt) {
        kernel_fdmt_boxcar_surrogate<kPacked, kNoise>
            <<<grid, block, 0, stream>>>(
                stopped, raw(m_rows_d), g.row_begin, g.row_count, m_rows.size(),
                total_tasks, raw(m_widths_d), m_filters, max_w,
                m_config.detect_thresh, raw(m_thresholds_d), raw(m_scales_d),
                raw(m_peaks_d), raw(m_open_d), m_peak_count,
                raw(m_candidates_d), raw(m_counts_d), raw(m_slots_d),
                raw(m_per_dm_d), m_config.max_candidates,
                m_config.max_cands_per_dm, m_config.peak_mode,
                m_config.skip_cold_prefix, root, surr_view, bt, raw(m_bright_d),
                m_bright_q);
    };
    if (m_config.input_nbits != 32) {
        launch.template operator()<true, false>(BlockTables{});
    } else if (m_block_tables.count > 0) {
        launch.template operator()<false, true>(m_block_tables);
    } else {
        launch.template operator()<false, false>(BlockTables{});
    }
}

void FDMTBoxcarCudaEngine::launch_flush_surrogate(cudaStream_t stream) {
    const SizeType total_tasks = m_rows.size() * m_beams;
    if (total_tasks == 0) {
        return;
    }
    const dim3 grid(static_cast<unsigned>(
        (total_tasks + kSurrWarpsPerBlock - 1) / kSurrWarpsPerBlock));
    const dim3 block(static_cast<unsigned>(kSurrBlockThreads));
    gpu_utils::check_kernel_launch_params(grid, block);
    const auto surr_view = make_surrogate_view();
    const auto max_w     = static_cast<uint32_t>(m_width);

    const auto launch_fl = [&]<bool kNoise>(const BlockTables& bt) {
        kernel_flush_surrogate<kNoise><<<grid, block, 0, stream>>>(
            raw(m_rows_d), m_rows.size(), total_tasks, m_peak_count,
            raw(m_peaks_d), raw(m_open_d), raw(m_widths_d), m_filters, max_w,
            m_config.detect_thresh, m_config.peak_mode, raw(m_thresholds_d),
            raw(m_scales_d), raw(m_candidates_d), raw(m_counts_d),
            raw(m_slots_d), raw(m_per_dm_d), m_config.max_candidates,
            m_config.max_cands_per_dm, surr_view, bt, raw(m_bright_d),
            m_bright_q);
    };
    if (m_block_tables.count > 0 && m_config.input_nbits == 32) {
        launch_fl.template operator()<true>(m_block_tables);
    } else {
        launch_fl.template operator()<false>(BlockTables{});
    }
    gpu_utils::check_last_gpu_error("FDMTBoxcar surrogate flush failed");
}

void FDMTBoxcarCudaEngine::launch_open_begin_surrogate(cudaStream_t stream) {
    const SizeType tasks = m_rows.size() * m_beams;
    if (tasks == 0) {
        return;
    }
    const auto blocks =
        static_cast<unsigned>(std::min<SizeType>((tasks + 255) / 256, 1024));
    const auto surr_view = make_surrogate_view();
    kernel_open_begin_surrogate<<<blocks, 256, 0, stream>>>(
        raw(m_rows_d), m_rows.size(), tasks, m_peak_count, m_config.peak_mode,
        m_config.skip_cold_prefix, raw(m_peaks_d), surr_view, raw(m_oldest_d));
    gpu_utils::check_last_gpu_error("FDMTBoxcar surrogate frontier failed");
}

} // namespace dmt::algorithms::detail::fdmt_boxcar_cuda
