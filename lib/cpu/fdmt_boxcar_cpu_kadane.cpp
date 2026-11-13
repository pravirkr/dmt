#include <algorithm>
#include <array>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <omp.h>

#include "dmt/fdmt_int_tree.hpp"
#include "dmt/fdmt_variance.hpp"

#include "fdmt_boxcar_cpu_engine.hpp"

// CPU FDMTBoxcar engine: Design A batched multi-row SIMD Kadane bound gate
// (kKadane), branchless row staging, and lazy exact prefix + score_tile().

namespace dmt::algorithms::detail::fdmt_boxcar_cpu {

namespace {

/// Branchless merge/pointer resolution for `n` samples starting at `t0`.
/// Returns a direct pointer into `in` for materialized / copy rows, or writes
/// merged `tail + head/hist` into `dst` and returns `dst`.
template <typename Elem>
const Elem* stage_row_typed(const Elem* state_base,
                            const Elem* prev_base,
                            const Elem* hist_base,
                            const FDMTSearchInput& in,
                            const BoxcarRow& row,
                            SizeType beam,
                            SizeType t0,
                            SizeType n,
                            Elem* dst) noexcept {
    if (!row.unmaterialized || in.materialized_root) {
        return state_base + (beam * in.beam_stride) +
               (row.row * in.row_stride) + t0;
    }
    const Elem* prev = (prev_base != nullptr ? prev_base : state_base) +
                       (beam * in.beam_stride);
    const Elem* tail = prev + (row.tail * in.row_stride) + t0;
    if (row.head == SIZE_MAX) {
        return tail;
    }
    const SizeType split =
        row.delay > t0 ? std::min(n, row.delay - t0) : SizeType{0};
    if (split > 0) {
        const Elem* hist = hist_base + (beam * in.history_stride) +
                           (row.hist - in.history_base);
        if (!in.history_ring) {
            const Elem* h0 = hist + t0;
#pragma omp simd
            for (SizeType j = 0; j < split; ++j) {
                dst[j] = tail[j] + h0[j];
            }
        } else {
            const auto slot0 =
                static_cast<SizeType>((in.sample_begin + t0) % row.delay);
            const SizeType part1 = std::min(split, row.delay - slot0);
            const Elem* h1       = hist + slot0;
#pragma omp simd
            for (SizeType j = 0; j < part1; ++j) {
                dst[j] = tail[j] + h1[j];
            }
            if (part1 < split) {
                const Elem* t2     = tail + part1;
                Elem* d2           = dst + part1;
                const SizeType rem = split - part1;
#pragma omp simd
                for (SizeType j = 0; j < rem; ++j) {
                    d2[j] = t2[j] + hist[j];
                }
            }
        }
    }
    if (split < n) {
        const Elem* head_p =
            prev + (row.head * in.row_stride) + ((t0 + split) - row.delay);
        const Elem* tail_p = tail + split;
        Elem* dst_p        = dst + split;
        const SizeType rem = n - split;
#pragma omp simd
        for (SizeType j = 0; j < rem; ++j) {
            dst_p[j] = tail_p[j] + head_p[j];
        }
    }
    return dst;
}

} // namespace

void FDMTBoxcarCpuEngine::init_kadane(bool use_box_smearing) {
    const SizeType n_rows  = m_rows.size();
    const SizeType total_f = n_rows * m_filters;
    std::vector<float> all_raw_thresh;
    all_raw_thresh.reserve(total_f);
    for (const auto& p : m_planes) {
        all_raw_thresh.insert(all_raw_thresh.end(), p.raw_thresh_lut.begin(),
                              p.raw_thresh_lut.end());
    }

    if (m_config.input_nbits != 32) {
        std::vector<uint64_t> all_int_thresh;
        all_int_thresh.reserve(total_f);
        for (const auto& p : m_planes) {
            all_int_thresh.insert(all_int_thresh.end(),
                                  p.integer_thresh_lut.begin(),
                                  p.integer_thresh_lut.end());
        }
        std::vector<float> eff_raw_thresh = all_raw_thresh;
        for (SizeType i = 0; i < total_f; ++i) {
            if (all_int_thresh[i] != UINT64_MAX) {
                const double centered =
                    (static_cast<double>(all_int_thresh[i]) -
                     m_integer_means[i]) *
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
            eff_raw_thresh, m_integer_means, all_int_thresh, m_carry32,
            max_int_sample);
        m_kadane_int32 = m_carry32 && m_kadane_plan.has_int32;
        if (!m_kadane_int32) {
            populate_row_int_m1();
        }
    } else {
        m_kadane_plan  = kadane::make_kadane_gate_plan(m_planes.front().widths,
                                                       m_config.use_trapezoid,
                                                       n_rows, all_raw_thresh);
        m_kadane_int32 = false;
    }

    const SizeType nb   = m_kadane_plan.num_buckets;
    const SizeType rs   = m_kadane_plan.row_stride;
    const SizeType rows = m_beams * n_rows;
    m_open.assign(rows, uint8_t{0});
    m_any_edge_lag = false;
    for (SizeType k = 0; k < nb; ++k) {
        if (m_kadane_plan.max_edge_lag[k] > 0U) {
            m_any_edge_lag = true;
        }
    }
    if (m_kadane_int32) {
        m_kadane_ic.assign(m_beams * nb * rs, int32_t{0});
        m_tile_u32.resize(static_cast<SizeType>(m_threads) *
                          kadane::kBatchRows * kGateTile);
        if (m_any_edge_lag) {
            m_lag_i32.resize(rows * nb);
        }
    } else {
        m_kadane_fc.assign(m_beams * nb * rs, 0.0F);
        m_tile_f32.resize(static_cast<SizeType>(m_threads) *
                          kadane::kBatchRows * kGateTile);
        if (m_any_edge_lag) {
            m_lag_f32.resize(rows * nb);
        }
    }
}

void FDMTBoxcarCpuEngine::populate_row_int_m1() {
    const SizeType n_rows = m_rows.size();
    const uint32_t w0     = m_planes.front().widths[0];
    const uint32_t we0 = m_config.use_trapezoid ? trapezoid_edge_width(w0) : 0U;
    const double g0    = we0 == 0U ? static_cast<double>(w0)
                                   : static_cast<double>(2U * (w0 - we0));
    m_row_int_m1.assign(n_rows, 0.0);
    for (SizeType r = 0; r < n_rows; ++r) {
        m_row_int_m1[r] = m_integer_means[r * m_filters] / g0;
    }
}

const float* FDMTBoxcarCpuEngine::stage_row_f32(const FDMTSearchInput& in,
                                                const BoxcarRow& row,
                                                SizeType beam,
                                                SizeType t0,
                                                SizeType n,
                                                float* dst) noexcept {
    return stage_row_typed<float>(in.state, in.prev_state, in.history, in, row,
                                  beam, t0, n, dst);
}

const uint32_t* FDMTBoxcarCpuEngine::stage_row_u32(const FDMTSearchInput& in,
                                                   const BoxcarRow& row,
                                                   SizeType beam,
                                                   SizeType t0,
                                                   SizeType n,
                                                   uint32_t* dst) noexcept {
    return stage_row_typed<uint32_t>(
        in.integer32_state, in.integer32_prev_state, in.integer32_history, in,
        row, beam, t0, n, dst);
}

const float*
FDMTBoxcarCpuEngine::stage_row_int_centered_f32(const FDMTSearchInput& in,
                                                const BoxcarRow& row,
                                                SizeType row_idx,
                                                SizeType beam,
                                                SizeType t0,
                                                SizeType n,
                                                float* dst) const noexcept {
    const double m1   = m_row_int_m1[row_idx];
    const double unit = m_integer_unit;
    const auto run    = [&](const auto* state_base, const auto* prev_base,
                            const auto* hist_base) {
        if (!row.unmaterialized || in.materialized_root) {
            const auto* s = state_base + (beam * in.beam_stride) +
                            (row.row * in.row_stride) + t0;
#pragma omp simd
            for (SizeType j = 0; j < n; ++j) {
                dst[j] =
                    static_cast<float>((static_cast<double>(s[j]) - m1) * unit);
            }
            return;
        }
        const auto* prev = (prev_base != nullptr ? prev_base : state_base) +
                           (beam * in.beam_stride);
        const auto* tail = prev + (row.tail * in.row_stride) + t0;
        if (row.head == SIZE_MAX) {
#pragma omp simd
            for (SizeType j = 0; j < n; ++j) {
                dst[j] = static_cast<float>(
                    (static_cast<double>(tail[j]) - m1) * unit);
            }
            return;
        }
        const SizeType split =
            row.delay > t0 ? std::min(n, row.delay - t0) : SizeType{0};
        if (split > 0) {
            const auto* hist = hist_base + (beam * in.history_stride) +
                               (row.hist - in.history_base);
            if (!in.history_ring) {
                const auto* h0 = hist + t0;
#pragma omp simd
                for (SizeType j = 0; j < split; ++j) {
                    const uint64_t sum = static_cast<uint64_t>(tail[j]) +
                                         static_cast<uint64_t>(h0[j]);
                    dst[j]             = static_cast<float>(
                        (static_cast<double>(sum) - m1) * unit);
                }
            } else {
                const auto slot0 =
                    static_cast<SizeType>((in.sample_begin + t0) % row.delay);
                const SizeType part1 = std::min(split, row.delay - slot0);
                const auto* h1       = hist + slot0;
#pragma omp simd
                for (SizeType j = 0; j < part1; ++j) {
                    const uint64_t sum = static_cast<uint64_t>(tail[j]) +
                                         static_cast<uint64_t>(h1[j]);
                    dst[j]             = static_cast<float>(
                        (static_cast<double>(sum) - m1) * unit);
                }
                if (part1 < split) {
                    const auto* t2     = tail + part1;
                    float* d2          = dst + part1;
                    const SizeType rem = split - part1;
#pragma omp simd
                    for (SizeType j = 0; j < rem; ++j) {
                        const uint64_t sum = static_cast<uint64_t>(t2[j]) +
                                             static_cast<uint64_t>(hist[j]);
                        d2[j]              = static_cast<float>(
                            (static_cast<double>(sum) - m1) * unit);
                    }
                }
            }
        }
        if (split < n) {
            const auto* head_p =
                prev + (row.head * in.row_stride) + ((t0 + split) - row.delay);
            const auto* tail_p = tail + split;
            float* dst_p       = dst + split;
            const SizeType rem = n - split;
#pragma omp simd
            for (SizeType j = 0; j < rem; ++j) {
                const uint64_t sum = static_cast<uint64_t>(tail_p[j]) +
                                     static_cast<uint64_t>(head_p[j]);
                dst_p[j] =
                    static_cast<float>((static_cast<double>(sum) - m1) * unit);
            }
        }
    };
    if (in.integer32_state != nullptr) {
        run(in.integer32_state, in.integer32_prev_state, in.integer32_history);
    } else {
        run(in.integer_state, in.integer_prev_state, in.integer_history);
    }
    return dst;
}

template <typename T, typename C>
void FDMTBoxcarCpuEngine::save_row_carry(SizeType row_idx,
                                         SizeType beam,
                                         const FDMTSearchInput& in) noexcept {
    const SizeType lookback = m_width - 1;
    const SizeType keep     = std::min(in.nsamps, lookback);
    if (keep == 0) {
        return;
    }
    const SizeType index = (beam * m_rows.size()) + row_idx;
    C* carry             = nullptr;
    if constexpr (std::is_same_v<C, uint32_t>) {
        carry = m_integer_carry32.data() + (index * m_width);
    } else if constexpr (std::is_integral_v<T>) {
        carry = m_integer_carry.data() + (index * m_width);
    } else {
        carry = m_carry.data() + (index * m_width);
    }
    const auto& row        = m_rows[row_idx];
    const SizeType p       = m_row_plane[row_idx];
    const bool root        = m_planes[p].level == m_plan.get_niters();
    const SizeType ring0   = boxcar_ring0(in, row);
    const SizeType t_first = in.nsamps - keep;
    auto slot = static_cast<SizeType>((in.sample_begin + t_first) % m_width);
    for (SizeType t = t_first; t < in.nsamps; ++t) {
        if constexpr (std::is_integral_v<T>) {
            carry[slot] = static_cast<C>(
                boxcar_integer_sample(in, row, beam, t, root, ring0));
        } else {
            carry[slot] =
                static_cast<C>(boxcar_sample(in, row, beam, t, root, ring0));
        }
        slot = (slot + 1 == m_width) ? 0 : (slot + 1);
    }
}

template <typename T, typename C>
void FDMTBoxcarCpuEngine::lazy_score_tile(
    SizeType row_idx,
    SizeType beam,
    const FDMTSearchInput& in,
    SizeType t0,
    SizeType n,
    const TileTables* precomputed_tables) {
    const auto tid = static_cast<SizeType>(omp_get_thread_num());
    using Prefix = std::conditional_t<std::is_integral_v<T>, uint64_t, double>;
    Prefix* prefix       = nullptr;
    const C* carry       = nullptr;
    const SizeType index = (beam * m_rows.size()) + row_idx;
    if constexpr (std::is_same_v<C, uint32_t>) {
        prefix = m_integer_prefix.data() + (tid * m_prefix_stride);
        carry  = m_integer_carry32.data() + (index * m_width);
    } else if constexpr (std::is_integral_v<T>) {
        prefix = m_integer_prefix.data() + (tid * m_prefix_stride);
        carry  = m_integer_carry.data() + (index * m_width);
    } else {
        prefix = m_prefix.data() + (tid * m_prefix_stride);
        carry  = m_carry.data() + (index * m_width);
    }

    const SizeType p        = m_row_plane[row_idx];
    const SizeType dm       = row_idx - m_offsets[p];
    const auto& plane       = m_planes[p];
    const auto& row         = m_rows[row_idx];
    const bool root         = plane.level == m_plan.get_niters();
    const SizeType ring0    = boxcar_ring0(in, row);
    const SizeType lookback = m_width - 1;
    const uint64_t start    = in.sample_begin + t0;

    const Prefix* tile_prefix = prefix;
    if constexpr (!std::is_integral_v<T>) {
        if (t0 + n <= m_kadane_exact_prefix_cap) {
            const uint64_t first = in.sample_begin;
            prefix[0]            = 0.0;
            const SizeType cold =
                first >= lookback ? SizeType{0}
                                  : (lookback - static_cast<SizeType>(first));
            for (SizeType j = 0; j < cold; ++j) {
                prefix[j + 1] = 0.0;
            }
            if (cold < lookback) {
                auto slot = static_cast<SizeType>((first - (lookback - cold)) %
                                                  m_width);
                for (SizeType j = cold; j < lookback; ++j) {
                    prefix[j + 1] = prefix[j] + carry[slot];
                    slot          = (slot + 1 == m_width) ? 0 : (slot + 1);
                }
            }
            prefix_tile<T>(in, row, beam, root, ring0, 0, t0 + n,
                           prefix + lookback);
            tile_prefix = prefix + t0;
        }
    }
    if (tile_prefix == prefix &&
        (std::is_integral_v<T> || t0 + n > m_kadane_exact_prefix_cap)) {
        prefix[0] = Prefix{0};
        if (t0 >= lookback) {
            prefix_tile<T>(in, row, beam, root, ring0, t0 - lookback,
                           lookback + n, prefix);
        } else {
            const SizeType pre_in = lookback - t0;
            const SizeType cold =
                in.sample_begin >= pre_in
                    ? SizeType{0}
                    : (pre_in - static_cast<SizeType>(in.sample_begin));
            for (SizeType j = 0; j < cold; ++j) {
                prefix[j + 1] = Prefix{0};
            }
            if (cold < pre_in) {
                const SizeType from_carry = pre_in - cold;
                auto slot                 = static_cast<SizeType>(
                    (in.sample_begin - from_carry) % m_width);
                for (SizeType j = cold; j < pre_in; ++j) {
                    prefix[j + 1] = prefix[j] + carry[slot];
                    slot          = (slot + 1 == m_width) ? 0 : (slot + 1);
                }
            }
            prefix_tile<T>(in, row, beam, root, ring0, 0, t0 + n,
                           prefix + pre_in);
        }
    }

    TileTables local_tables{
        .thr = plane.raw_thresh_lut.data() + (dm * m_filters),
        .inv = plane.inv_sigma_lut.data() + (dm * m_filters),
    };
    const TileTables* tables_ptr = &local_tables;
    if constexpr (!std::is_integral_v<T>) {
        if (precomputed_tables != nullptr) {
            tables_ptr = precomputed_tables;
        } else if (m_noise_on) {
            (void)tile_tables(plane, dm, row_idx, start, n, tid, local_tables);
        }
    }

    auto* peaks = m_peaks.data() + (index * m_peak_count);
    score_tile<T>(plane, dm, row, beam, tile_prefix, lookback, start, n, peaks,
                  row_idx, *tables_ptr);
    bool any_open = false;
    if (m_config.peak_mode != PeakFindingMode::kAllAboveThreshold) {
        any_open = std::any_of(peaks, peaks + m_peak_count,
                               [](const BoxcarPeak& pk) { return pk.open; });
    }
    m_open[index] = any_open ? uint8_t{1} : uint8_t{0};
}

void FDMTBoxcarCpuEngine::consume_kadane(SizeType row_lo,
                                         SizeType row_hi,
                                         SizeType first_beam,
                                         SizeType beams,
                                         const FDMTSearchInput& input) {
    constexpr SizeType kBatch = kadane::kBatchRows;
    const SizeType nrows      = row_hi - row_lo;
    const SizeType nbatches   = (nrows + kBatch - 1) / kBatch;
    const SizeType ntasks     = beams * nbatches;
    const SizeType nb         = m_kadane_plan.num_buckets;
    const SizeType rs         = m_kadane_plan.row_stride;

    const auto batch_task = [&](SizeType task_id) {
        const auto tid           = static_cast<SizeType>(omp_get_thread_num());
        const SizeType beam      = first_beam + (task_id / nbatches);
        const SizeType batch_idx = task_id % nbatches;
        const SizeType b_lo      = row_lo + (batch_idx * kBatch);
        const SizeType b_hi      = std::min(row_hi, b_lo + kBatch);
        const SizeType b_cap     = b_hi - b_lo;

        std::array<FDMTSearchInput, kBatch> lane_in{};
        std::array<bool, kBatch> lane_active{};
        for (SizeType q = 0; q < b_cap; ++q) {
            FDMTSearchInput in = input;
            if (input.row_lag != nullptr &&
                !lagged_row_input(input, m_rows[b_lo + q].row, in)) {
                continue;
            }
            in.search_stop = m_search_stop;
            if (!stop_row_input(in) || in.nsamps == 0) {
                continue;
            }
            lane_in[q]     = in;
            lane_active[q] = true;
        }

        for (SizeType q0 = 0; q0 < b_cap;) {
            if (!lane_active[q0]) {
                ++q0;
                continue;
            }
            const SizeType total_nsamps = lane_in[q0].nsamps;
            SizeType q1                 = q0 + 1;
            while (q1 < b_cap && lane_active[q1] &&
                   lane_in[q1].nsamps == total_nsamps) {
                ++q1;
            }
            const SizeType sub_n = q1 - q0;
            const SizeType r_off = b_lo + q0;

            for (SizeType t0 = 0; t0 < total_nsamps; t0 += kGateTile) {
                const SizeType n = std::min(kGateTile, total_nsamps - t0);
                std::array<SizeType, kBatch> fv{};
                bool any_fv = false;
                if (m_config.skip_cold_prefix) {
                    for (SizeType l = 0; l < sub_n; ++l) {
                        const uint64_t s0 = lane_in[q0 + l].sample_begin + t0;
                        const uint64_t wu = m_rows[r_off + l].warmup;
                        if (wu > s0) {
                            fv[l] = static_cast<SizeType>(
                                std::min<uint64_t>(n, wu - s0));
                            any_fv = true;
                        }
                    }
                }
                const SizeType* fv_ptr = any_fv ? fv.data() : nullptr;

                if (m_kadane_int32) {
                    std::array<const uint32_t*, kBatch> ptrs{};
                    uint32_t* scratch =
                        m_tile_u32.data() + (tid * kBatch * kGateTile);
                    for (SizeType l = 0; l < sub_n; ++l) {
                        ptrs[l] = stage_row_u32(lane_in[q0 + l],
                                                m_rows[r_off + l], beam, t0, n,
                                                scratch + (l * kGateTile));
                    }
                    std::array<int32_t, kadane::kMaxGateBuckets * kBatch>
                        lag_in{};
                    std::array<int32_t, kadane::kMaxGateBuckets * kBatch>
                        max_out{};
                    if (m_any_edge_lag) {
                        for (SizeType k = 0; k < nb; ++k) {
                            const uint32_t lag_len =
                                m_kadane_plan.max_edge_lag[k];
                            if (lag_len > 0U) {
                                for (SizeType l = 0; l < sub_n; ++l) {
                                    const uint64_t s0 =
                                        lane_in[q0 + l].sample_begin + t0;
                                    const uint64_t w_beg =
                                        s0 > lag_len ? (s0 - lag_len) : 0ULL;
                                    const SizeType idx =
                                        (beam * m_rows.size()) + r_off + l;
                                    lag_in[(k * kBatch) + l] =
                                        m_lag_i32[(idx * nb) + k]
                                            .query_and_prune(w_beg);
                                }
                            }
                        }
                    }

                    int32_t* state_base =
                        m_kadane_ic.data() + (beam * nb * rs) + r_off;
                    const uint32_t hit_mask = kadane::scan_gate_tile_i32(
                        m_kadane_plan, r_off, sub_n, n, ptrs.data(), state_base,
                        rs, fv_ptr, m_any_edge_lag ? lag_in.data() : nullptr,
                        kBatch, m_any_edge_lag ? max_out.data() : nullptr,
                        kBatch);

                    if (m_any_edge_lag) {
                        for (SizeType k = 0; k < nb; ++k) {
                            if (m_kadane_plan.max_edge_lag[k] > 0U) {
                                for (SizeType l = 0; l < sub_n; ++l) {
                                    const uint64_t s_end =
                                        lane_in[q0 + l].sample_begin + t0 + n;
                                    const SizeType idx =
                                        (beam * m_rows.size()) + r_off + l;
                                    m_lag_i32[(idx * nb) + k].push(
                                        s_end, max_out[(k * kBatch) + l]);
                                }
                            }
                        }
                    }

                    for (SizeType l = 0; l < sub_n; ++l) {
                        const SizeType r   = r_off + l;
                        const SizeType idx = (beam * m_rows.size()) + r;
                        const bool hit =
                            ((hit_mask >> static_cast<uint32_t>(l)) & 1U) != 0U;
                        if (hit || m_open[idx] != 0U) {
                            lazy_score_tile<uint64_t, uint32_t>(
                                r, beam, lane_in[q0 + l], t0, n, nullptr);
                        }
                    }
                } else {
                    std::array<const float*, kBatch> ptrs{};
                    float* scratch =
                        m_tile_f32.data() + (tid * kBatch * kGateTile);
                    for (SizeType l = 0; l < sub_n; ++l) {
                        if (m_config.input_nbits == 32) {
                            ptrs[l] = stage_row_f32(
                                lane_in[q0 + l], m_rows[r_off + l], beam, t0, n,
                                scratch + (l * kGateTile));
                        } else {
                            ptrs[l] = stage_row_int_centered_f32(
                                lane_in[q0 + l], m_rows[r_off + l], r_off + l,
                                beam, t0, n, scratch + (l * kGateTile));
                        }
                    }

                    std::array<TileTables, kBatch> lane_tables{};
                    const float* cut_ptr = m_kadane_plan.cut.data() + r_off;
                    SizeType cut_stride  = rs;
                    if (m_noise_on) {
                        float* thr_cuts = m_thread_cuts.data() +
                                          (tid * kadane::kMaxGateBuckets *
                                           kadane::kMaxBatchRows);
                        bool any_custom = false;
                        std::array<float, kadane::kMaxGateBuckets> r_cuts{};
                        for (SizeType l = 0; l < sub_n; ++l) {
                            const SizeType r  = r_off + l;
                            const SizeType p  = m_row_plane[r];
                            const SizeType dm = r - m_offsets[p];
                            const uint64_t s0 =
                                lane_in[q0 + l].sample_begin + t0;
                            const float* screen_thr = tile_tables(
                                m_planes[p], dm, r, s0, n, tid, lane_tables[l]);
                            const float* ctor_thr =
                                m_planes[p].raw_thresh_lut.data() +
                                (dm * m_filters);
                            if (screen_thr != ctor_thr) {
                                any_custom = true;
                                m_kadane_plan.recompute_row_cuts(
                                    r, {screen_thr, m_filters}, r_cuts);
                                for (SizeType k = 0; k < nb; ++k) {
                                    thr_cuts[(k * kBatch) + l] = r_cuts[k];
                                }
                            } else {
                                for (SizeType k = 0; k < nb; ++k) {
                                    thr_cuts[(k * kBatch) + l] =
                                        m_kadane_plan.row_cut(k, r);
                                }
                            }
                        }
                        if (any_custom) {
                            cut_ptr    = thr_cuts;
                            cut_stride = kBatch;
                        }
                    }

                    std::array<float, kadane::kMaxGateBuckets * kBatch>
                        lag_in{};
                    std::array<float, kadane::kMaxGateBuckets * kBatch>
                        max_out{};
                    if (m_any_edge_lag) {
                        for (SizeType k = 0; k < nb; ++k) {
                            const uint32_t lag_len =
                                m_kadane_plan.max_edge_lag[k];
                            if (lag_len > 0U) {
                                for (SizeType l = 0; l < sub_n; ++l) {
                                    const uint64_t s0 =
                                        lane_in[q0 + l].sample_begin + t0;
                                    const uint64_t w_beg =
                                        s0 > lag_len ? (s0 - lag_len) : 0ULL;
                                    const SizeType idx =
                                        (beam * m_rows.size()) + r_off + l;
                                    lag_in[(k * kBatch) + l] =
                                        m_lag_f32[(idx * nb) + k]
                                            .query_and_prune(w_beg);
                                }
                            }
                        }
                    }

                    float* state_base =
                        m_kadane_fc.data() + (beam * nb * rs) + r_off;
                    const uint32_t hit_mask = kadane::scan_gate_tile_f32(
                        nb, sub_n, n, ptrs.data(),
                        m_kadane_plan.bias.data() + r_off, rs, cut_ptr,
                        cut_stride, state_base, rs, fv_ptr,
                        m_any_edge_lag ? lag_in.data() : nullptr, kBatch,
                        m_any_edge_lag ? max_out.data() : nullptr, kBatch);

                    if (m_any_edge_lag) {
                        for (SizeType k = 0; k < nb; ++k) {
                            if (m_kadane_plan.max_edge_lag[k] > 0U) {
                                for (SizeType l = 0; l < sub_n; ++l) {
                                    const uint64_t s_end =
                                        lane_in[q0 + l].sample_begin + t0 + n;
                                    const SizeType idx =
                                        (beam * m_rows.size()) + r_off + l;
                                    m_lag_f32[(idx * nb) + k].push(
                                        s_end, max_out[(k * kBatch) + l]);
                                }
                            }
                        }
                    }

                    for (SizeType l = 0; l < sub_n; ++l) {
                        const SizeType r   = r_off + l;
                        const SizeType idx = (beam * m_rows.size()) + r;
                        const bool hit =
                            ((hit_mask >> static_cast<uint32_t>(l)) & 1U) != 0U;
                        if (hit || m_open[idx] != 0U) {
                            if (m_config.input_nbits == 32) {
                                lazy_score_tile<float, float>(
                                    r, beam, lane_in[q0 + l], t0, n,
                                    m_noise_on ? &lane_tables[l] : nullptr);
                            } else if (m_carry32) {
                                lazy_score_tile<uint64_t, uint32_t>(
                                    r, beam, lane_in[q0 + l], t0, n, nullptr);
                            } else {
                                lazy_score_tile<uint64_t, uint64_t>(
                                    r, beam, lane_in[q0 + l], t0, n, nullptr);
                            }
                        }
                    }
                }
            }

            for (SizeType l = 0; l < sub_n; ++l) {
                const SizeType r          = r_off + l;
                const FDMTSearchInput& in = lane_in[q0 + l];
                if (m_config.input_nbits == 32) {
                    save_row_carry<float, float>(r, beam, in);
                } else if (m_carry32) {
                    save_row_carry<uint64_t, uint32_t>(r, beam, in);
                } else {
                    save_row_carry<uint64_t, uint64_t>(r, beam, in);
                }
            }
            q0 = q1;
        }
    };

    const bool one_plane = m_row_plane[row_lo] == m_row_plane[row_hi - 1];
    if (one_plane) {
#pragma omp parallel for num_threads(m_threads) schedule(static)
        for (SizeType t = 0; t < ntasks; ++t) {
            batch_task(t);
        }
    } else {
#pragma omp parallel for num_threads(m_threads) schedule(dynamic, 1)
        for (SizeType t = 0; t < ntasks; ++t) {
            batch_task(t);
        }
    }
}

} // namespace dmt::algorithms::detail::fdmt_boxcar_cpu
