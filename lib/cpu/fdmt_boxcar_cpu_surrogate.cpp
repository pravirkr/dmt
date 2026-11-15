#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include <omp.h>

#include "dmt/fdmt_int_tree.hpp"

#include "fdmt_boxcar_cpu_engine.hpp"

// CPU FDMTBoxcar engine: standalone multi-strip Kadane surrogate scorer
// (kKadaneSurrogate), exact streaming replay, candidate frontier, and flush.

namespace dmt::algorithms::detail::fdmt_boxcar_cpu {

void FDMTBoxcarCpuEngine::init_surrogate(bool use_box_smearing) {
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
                all_raw_thresh, all_inv_sigma, m_integer_means, m_integer_unit);
            m_tile_u32.resize(static_cast<SizeType>(m_threads) * kGateTile);
        } else {
            m_surrogate_int_u32 = false;
            populate_row_int_m1();
            m_surrogate_plan = kadane::make_kadane_surrogate_plan(
                m_planes.front().widths, m_config.use_trapezoid, n_rows,
                all_raw_thresh, all_inv_sigma);
        }
    }

    const SizeType np   = m_surrogate_plan.num_passes;
    const SizeType rs   = m_surrogate_plan.row_stride;
    const SizeType rows = m_beams * n_rows;
    m_surrogate_c.assign(m_beams * np * rs, 0.0F);
    if (m_surrogate_plan.has_decimated) {
        m_surrogate_dec.assign(m_beams * rs, 0.0F);
    }
    m_surrogate_rows.resize(rows);
    m_surrogate_open.assign(rows, uint8_t{0});
    m_surrogate_pos.assign(rows, uint64_t{0});
    m_tile_f32.resize(static_cast<SizeType>(m_threads) * kadane::kBatchRows *
                      kGateTile);
    m_surrogate_chk_c.resize(static_cast<SizeType>(m_threads) * np *
                             kadane::kBatchRows);
    m_surrogate_chk_dec.resize(static_cast<SizeType>(m_threads) *
                               kadane::kBatchRows);
    m_surrogate_lb.resize(static_cast<SizeType>(m_threads) *
                          (m_surrogate_carry_ring - 1));
}

void FDMTBoxcarCpuEngine::emit_surrogate_raw(SizeType row,
                                             SizeType beam,
                                             uint64_t begin_sample,
                                             uint64_t end_sample,
                                             uint64_t peak_sample,
                                             float snr,
                                             uint32_t filter_idx,
                                             uint32_t width) {
    const auto make_candidate = [&] {
        auto candidate         = m_rows[row].candidate;
        candidate.beam_idx     = static_cast<uint32_t>(beam);
        candidate.peak_sample  = peak_sample;
        candidate.begin_sample = begin_sample;
        candidate.end_sample   = end_sample;
        candidate.snr          = snr;
        candidate.filter_idx   = filter_idx;
        candidate.width =
            std::clamp<uint32_t>(width, 1U, static_cast<uint32_t>(m_width));
        return candidate;
    };
#pragma omp atomic
    ++m_counts.total_detected;
    auto& count = m_per_dm[(beam * m_rows.size()) + row];
    if (m_keep_q != 0) {
        keep((beam * m_rows.size()) + row, count, make_candidate());
        return;
    }
    if (m_config.max_cands_per_dm != 0 &&
        count++ >= m_config.max_cands_per_dm) {
#pragma omp atomic
        ++m_counts.dropped_per_dm;
        return;
    }
    const auto slot = m_next_slot.fetch_add(1, std::memory_order_relaxed);
    if (slot >= m_candidates.size()) {
#pragma omp atomic
        ++m_counts.dropped_global;
        return;
    }
    m_candidates[slot] = make_candidate();
#pragma omp atomic
    ++m_counts.emitted_count;
}

void FDMTBoxcarCpuEngine::emit_surrogate_peak(SizeType row,
                                              SizeType beam,
                                              const BoxcarPeak& peak) {
    const auto width = peak.raw_sum > 0U ? static_cast<uint32_t>(peak.raw_sum)
                                         : m_planes.front().widths[peak.filter];
    emit_surrogate_raw(row, beam, peak.begin, peak.end, peak.peak, peak.snr,
                       peak.filter, width);
}

const NoiseTables* FDMTBoxcarCpuEngine::active_surrogate_table(
    SizeType row_idx, uint64_t sample, uint32_t filter_idx) const noexcept {
    if (!m_noise_on || m_ring.empty()) {
        return nullptr;
    }
    const SizeType p      = m_row_plane[row_idx];
    const uint32_t w_near = m_planes[p].widths[filter_idx];
    return sample_table(sample, w_near, (*m_support)[row_idx]).table;
}

float FDMTBoxcarCpuEngine::eval_surrogate_excursion_snr(
    SizeType row_idx,
    const kadane::SurrogateExcursion& ex,
    const TileTables* tables,
    uint64_t tile_start,
    uint64_t tile_end,
    uint32_t& out_filter_idx) const noexcept {
    if (ex.best_len == 0U) {
        out_filter_idx = 0U;
        return 0.0F;
    }
    if (m_noise_on && ex.has_snr) {
        out_filter_idx = ex.best_filter;
        return ex.best_snr;
    }
    const float snr = m_surrogate_plan.eval_surrogate_snr(
        row_idx, ex.best_raw_sum, ex.best_len, &out_filter_idx);
    if (!m_noise_on) {
        return snr;
    }
    const SizeType p        = m_row_plane[row_idx];
    const SizeType dm       = row_idx - m_offsets[p];
    const auto& plane       = m_planes[p];
    const float* plane_thr  = plane.raw_thresh_lut.data() + (dm * m_filters);
    const float* plane_inv  = plane.inv_sigma_lut.data() + (dm * m_filters);
    const uint32_t w_near   = plane.widths[out_filter_idx];
    const float* active_thr = plane_thr;
    const float* active_inv = plane_inv;
    if (tables != nullptr && !tables->mixed && ex.best_end >= tile_start &&
        ex.best_end < tile_end) {
        active_thr = tables->thr;
        active_inv = tables->inv;
    } else if (!m_ring.empty()) {
        const uint64_t sup  = (*m_support)[row_idx];
        const auto [tbl, _] = sample_table(ex.best_end, w_near, sup);
        active_thr          = table_thr(tbl, plane_thr, row_idx, m_filters);
        active_inv          = table_inv(tbl, plane_inv, row_idx, m_filters);
    }
    if (active_inv == plane_inv) {
        return snr;
    }
    const float thr_f = active_thr[out_filter_idx];
    const float inv_f = active_inv[out_filter_idx];
    if (!is_finite_f32(thr_f) || thr_f >= 1.0e35F || inv_f <= 0.0F ||
        plane_inv[out_filter_idx] <= 0.0F) {
        return 0.0F;
    }
    return snr * (inv_f / plane_inv[out_filter_idx]);
}

void FDMTBoxcarCpuEngine::cache_surrogate_pass_snr(
    SizeType row_idx,
    kadane::KadaneSurrogatePassState& st,
    uint32_t pass_idx,
    const TileTables* tables,
    uint64_t tile_start,
    uint64_t tile_end,
    const kadane::KadaneSurrogatePassState* prev_st) const noexcept {
    if (!st.open || st.best_len == 0U) {
        return;
    }
    if (!st.has_snr) {
        uint32_t filt   = 0U;
        const float snr = eval_surrogate_excursion_snr(
            row_idx, kadane::detail::make_excursion(st, pass_idx), tables,
            tile_start, tile_end, filt);
        st.best_snr    = snr;
        st.best_filter = filt;
        st.has_snr     = true;
    }
    if (prev_st != nullptr && prev_st->open && prev_st->has_snr &&
        prev_st->seg_begin == st.seg_begin && prev_st->best_snr > st.best_snr &&
        active_surrogate_table(row_idx, prev_st->best_end,
                               prev_st->best_filter) !=
            active_surrogate_table(row_idx, st.best_end, st.best_filter)) {
        st.best_snr     = prev_st->best_snr;
        st.best_filter  = prev_st->best_filter;
        st.best_end     = prev_st->best_end;
        st.best_len     = prev_st->best_len;
        st.best_raw_sum = prev_st->best_raw_sum;
    }
}

void FDMTBoxcarCpuEngine::handle_surrogate_excursion(
    SizeType row_idx,
    SizeType beam,
    const kadane::SurrogateExcursion& ex,
    const kadane::KadaneSurrogateRowState& st,
    const TileTables* tables,
    uint64_t tile_start,
    uint64_t tile_end) {
    uint32_t filter_idx = 0U;
    const float snr     = eval_surrogate_excursion_snr(
        row_idx, ex, tables, tile_start, tile_end, filter_idx);
    if (snr < m_config.detect_thresh) {
        return;
    }
    const uint32_t clamped_w =
        std::clamp<uint32_t>(ex.best_len, 1U, static_cast<uint32_t>(m_width));
    if (m_config.peak_mode != PeakFindingMode::kContiguousPeakAcrossWidths) {
        emit_surrogate_raw(row_idx, beam, ex.seg_begin, ex.close_end,
                           ex.best_end, snr, filter_idx, clamped_w);
        return;
    }

    const SizeType idx = (beam * m_rows.size()) + row_idx;
    auto& pending      = m_peaks[idx * m_peak_count];
    if (pending.open) {
        const bool overlaps = (ex.seg_begin <= pending.end + 1ULL) &&
                              (pending.begin <= ex.close_end + 1ULL);
        if (overlaps) {
            pending.begin = std::min(pending.begin, ex.seg_begin);
            pending.end   = std::max(pending.end, ex.close_end);
            if (boxcar_better(snr, ex.best_end, filter_idx, pending)) {
                pending.snr     = snr;
                pending.peak    = ex.best_end;
                pending.filter  = filter_idx;
                pending.raw_sum = clamped_w;
            }
            return;
        }
        bool other_overlaps = false;
        for (SizeType k = 0; k < m_surrogate_plan.num_passes; ++k) {
            if (st.passes[k].open &&
                st.passes[k].seg_begin <= pending.end + 1ULL) {
                other_overlaps = true;
                break;
            }
        }
        if (!other_overlaps) {
            emit_surrogate_peak(row_idx, beam, pending);
            pending.open = false;
        } else {
            // Disjoint excursion arrived while another pass still bridges
            // `pending`: emit the non-overlapping one without losing either.
            emit_surrogate_raw(row_idx, beam, ex.seg_begin, ex.close_end,
                               ex.best_end, snr, filter_idx, clamped_w);
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

void FDMTBoxcarCpuEngine::maybe_finalize_surrogate_pending(
    SizeType row_idx,
    SizeType beam,
    uint64_t tile_end,
    const kadane::KadaneSurrogateRowState& st) {
    if (m_config.peak_mode != PeakFindingMode::kContiguousPeakAcrossWidths) {
        return;
    }
    const SizeType idx = (beam * m_rows.size()) + row_idx;
    auto& pending      = m_peaks[idx * m_peak_count];
    if (!pending.open) {
        return;
    }
    if (pending.end + 4ULL >= tile_end) {
        return;
    }
    for (SizeType k = 0; k < m_surrogate_plan.num_passes; ++k) {
        if (st.passes[k].open && st.passes[k].seg_begin <= pending.end + 1ULL) {
            return;
        }
    }
    emit_surrogate_peak(row_idx, beam, pending);
    pending.open = false;
}

const float*
FDMTBoxcarCpuEngine::stage_surrogate_row(const FDMTSearchInput& in,
                                         SizeType row_idx,
                                         SizeType beam,
                                         SizeType t0,
                                         SizeType n,
                                         float* dst_f32,
                                         uint32_t* tmp_u32) const noexcept {
    const auto& row = m_rows[row_idx];
    if (m_config.input_nbits == 32) {
        return stage_row_f32(in, row, beam, t0, n, dst_f32);
    }
    if (m_surrogate_int_u32) {
        const uint32_t* u32_ptr = stage_row_u32(in, row, beam, t0, n, tmp_u32);
        kadane::convert_u32_to_f32(u32_ptr, dst_f32, n);
        return dst_f32;
    }
    return stage_row_int_centered_f32(in, row, row_idx, beam, t0, n, dst_f32);
}

std::span<const float>
FDMTBoxcarCpuEngine::build_surrogate_lookback(SizeType tid,
                                              SizeType row_idx,
                                              SizeType beam,
                                              const FDMTSearchInput& in,
                                              SizeType t0) noexcept {
    const SizeType max_lb = m_surrogate_carry_ring - 1;
    const uint64_t start  = in.sample_begin + t0;
    const auto avail = static_cast<SizeType>(std::min<uint64_t>(start, max_lb));
    if (avail == 0) {
        return {};
    }
    float* lb_buf = m_surrogate_lb.data() + (tid * max_lb);
    if (t0 >= avail) {
        const SizeType t_beg = t0 - avail;
        uint32_t* tmp_u32    = m_surrogate_int_u32
                                   ? (m_tile_u32.data() + (tid * kGateTile))
                                   : nullptr;
        SizeType done        = 0;
        while (done < avail) {
            const SizeType step = std::min(kGateTile, avail - done);
            const float* p      = stage_surrogate_row(
                in, row_idx, beam, t_beg + done, step, lb_buf + done, tmp_u32);
            if (p != lb_buf + done) {
                std::copy_n(p, step, lb_buf + done);
            }
            done += step;
        }
        return {lb_buf, avail};
    }

    const SizeType from_carry = avail - t0;
    const SizeType index      = (beam * m_rows.size()) + row_idx;
    const float* carry = m_carry.data() + (index * m_surrogate_carry_ring);
    auto slot          = static_cast<SizeType>((in.sample_begin - from_carry) %
                                               m_surrogate_carry_ring);
    for (SizeType j = 0; j < from_carry; ++j) {
        lb_buf[j] = carry[slot];
        slot      = (slot + 1 == m_surrogate_carry_ring) ? 0 : (slot + 1);
    }
    if (t0 > 0) {
        uint32_t* tmp_u32 = m_surrogate_int_u32
                                ? (m_tile_u32.data() + (tid * kGateTile))
                                : nullptr;
        SizeType done     = 0;
        while (done < t0) {
            const SizeType step = std::min(kGateTile, t0 - done);
            const float* p =
                stage_surrogate_row(in, row_idx, beam, done, step,
                                    lb_buf + from_carry + done, tmp_u32);
            if (p != lb_buf + from_carry + done) {
                std::copy_n(p, step, lb_buf + from_carry + done);
            }
            done += step;
        }
    }
    return {lb_buf, avail};
}

void FDMTBoxcarCpuEngine::save_surrogate_carry(
    SizeType tid,
    SizeType row_idx,
    SizeType beam,
    const FDMTSearchInput& in) noexcept {
    const SizeType max_lb = m_surrogate_carry_ring - 1;
    const SizeType keep   = std::min(in.nsamps, max_lb);
    if (keep == 0) {
        return;
    }
    const SizeType index = (beam * m_rows.size()) + row_idx;
    float* carry         = m_carry.data() + (index * m_surrogate_carry_ring);
    const SizeType t_beg = in.nsamps - keep;
    auto slot            = static_cast<SizeType>((in.sample_begin + t_beg) %
                                                 m_surrogate_carry_ring);
    float* tmp_f32 = m_tile_f32.data() + (tid * kadane::kBatchRows * kGateTile);
    uint32_t* tmp_u32 =
        m_surrogate_int_u32 ? (m_tile_u32.data() + (tid * kGateTile)) : nullptr;
    SizeType done = 0;
    while (done < keep) {
        const SizeType step = std::min(kGateTile, keep - done);
        const float* src = stage_surrogate_row(in, row_idx, beam, t_beg + done,
                                               step, tmp_f32, tmp_u32);
        for (SizeType j = 0; j < step; ++j) {
            carry[slot] = src[j];
            slot        = (slot + 1 == m_surrogate_carry_ring) ? 0 : (slot + 1);
        }
        done += step;
    }
}

void FDMTBoxcarCpuEngine::compute_surrogate_row_triggers(
    SizeType row_idx,
    const float* screen_thr,
    std::array<float, kadane::kMaxSurrogatePasses>& out_triggers)
    const noexcept {
    const SizeType np = m_surrogate_plan.num_passes;
    if (m_surrogate_plan.disabled_row[row_idx] != 0U) {
        out_triggers.fill(std::numeric_limits<float>::max());
        return;
    }
    if (screen_thr == nullptr) {
        for (SizeType k = 0; k < np; ++k) {
            out_triggers[k] = m_surrogate_plan.row_trigger(k, row_idx);
        }
        return;
    }
    const SizeType p  = m_row_plane[row_idx];
    const SizeType dm = row_idx - m_offsets[p];
    const float* ctor_thr =
        m_planes[p].raw_thresh_lut.data() + (dm * m_filters);
    if (screen_thr == ctor_thr) {
        for (SizeType k = 0; k < np; ++k) {
            out_triggers[k] = m_surrogate_plan.row_trigger(k, row_idx);
        }
        return;
    }
    float min_ratio = 1.0F;
    for (SizeType f = 0; f < m_filters; ++f) {
        const float th = screen_thr[f];
        if (!is_finite_f32(th) || th <= 0.0F || th >= 1.0e35F) {
            out_triggers.fill(std::numeric_limits<float>::max());
            return;
        }
        if (ctor_thr[f] > 0.0F) {
            min_ratio = std::min(min_ratio, th / ctor_thr[f]);
        }
    }
    min_ratio = std::clamp(min_ratio, 0.0F, 1.0F);
    for (SizeType k = 0; k < np; ++k) {
        out_triggers[k] = m_surrogate_plan.row_trigger(k, row_idx) * min_ratio;
    }
}

void FDMTBoxcarCpuEngine::consume_surrogate(SizeType row_lo,
                                            SizeType row_hi,
                                            SizeType first_beam,
                                            SizeType beams,
                                            const FDMTSearchInput& input) {
    constexpr SizeType kBatch = kadane::kBatchRows;
    const SizeType nrows      = row_hi - row_lo;
    const SizeType nbatches   = (nrows + kBatch - 1) / kBatch;
    const SizeType ntasks     = beams * nbatches;
    const SizeType np         = m_surrogate_plan.num_passes;
    const SizeType rs         = m_surrogate_plan.row_stride;
    const bool has_dec        = m_surrogate_plan.has_decimated;

    std::array<uint32_t, kadane::kMaxSurrogatePasses> dec_factors{};
    for (SizeType k = 0; k < np; ++k) {
        dec_factors[k] = m_surrogate_plan.passes[k].decimate_factor;
    }

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

        float* chk_c =
            m_surrogate_chk_c.data() + (tid * np * kadane::kBatchRows);
        float* chk_dec =
            m_surrogate_chk_dec.data() + (tid * kadane::kBatchRows);
        uint32_t* tmp_u32 = m_surrogate_int_u32
                                ? (m_tile_u32.data() + (tid * kGateTile))
                                : nullptr;

        for (SizeType q0 = 0; q0 < b_cap;) {
            if (!lane_active[q0]) {
                ++q0;
                continue;
            }
            const SizeType total_nsamps = lane_in[q0].nsamps;
            const uint64_t phase_ref    = lane_in[q0].sample_begin & 3ULL;
            SizeType q1                 = q0 + 1;
            while (
                q1 < b_cap && lane_active[q1] &&
                lane_in[q1].nsamps == total_nsamps &&
                (!has_dec || (lane_in[q1].sample_begin & 3ULL) == phase_ref)) {
                ++q1;
            }
            const SizeType sub_n = q1 - q0;
            const SizeType r_off = b_lo + q0;

            for (SizeType t0 = 0; t0 < total_nsamps; t0 += kGateTile) {
                const SizeType n = std::min(kGateTile, total_nsamps - t0);
                std::array<const float*, kBatch> ptrs{};
                float* scratch = m_tile_f32.data() + (tid * kBatch * kGateTile);
                for (SizeType l = 0; l < sub_n; ++l) {
                    ptrs[l] = stage_surrogate_row(
                        lane_in[q0 + l], r_off + l, beam, t0, n,
                        scratch + (l * kGateTile), tmp_u32);
                }

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

                std::array<TileTables, kBatch> lane_tables{};
                const float* trig_ptr = m_surrogate_plan.trigger.data() + r_off;
                SizeType trig_stride  = rs;
                float* thr_cuts       = nullptr;
                bool any_custom       = false;
                if (m_noise_on) {
                    thr_cuts =
                        m_thread_cuts.data() +
                        (tid * kadane::kMaxGateBuckets * kadane::kMaxBatchRows);
                    std::array<float, kadane::kMaxSurrogatePasses> r_trig{};
                    for (SizeType l = 0; l < sub_n; ++l) {
                        const SizeType r  = r_off + l;
                        const SizeType p  = m_row_plane[r];
                        const SizeType dm = r - m_offsets[p];
                        const uint64_t s0 = lane_in[q0 + l].sample_begin + t0;
                        const float* screen_thr = tile_tables(
                            m_planes[p], dm, r, s0, n, tid, lane_tables[l]);
                        const float* ctor_thr =
                            m_planes[p].raw_thresh_lut.data() +
                            (dm * m_filters);
                        if (screen_thr != ctor_thr) {
                            any_custom = true;
                        }
                        compute_surrogate_row_triggers(r, screen_thr, r_trig);
                        for (SizeType k = 0; k < np; ++k) {
                            thr_cuts[(k * kBatch) + l] = r_trig[k];
                        }
                    }
                    if (any_custom) {
                        trig_ptr    = thr_cuts;
                        trig_stride = kBatch;
                    }
                }

                float* state_base =
                    m_surrogate_c.data() + (beam * np * rs) + r_off;
                float* dec_base =
                    has_dec ? (m_surrogate_dec.data() + (beam * rs) + r_off)
                            : nullptr;
                const auto dec_phase = static_cast<uint32_t>(
                    (lane_in[q0].sample_begin + t0) & 3ULL);

                const uint32_t hit_mask = kadane::scan_surrogate_tile_f32(
                    np, sub_n, n, ptrs.data(), dec_factors.data(),
                    m_surrogate_plan.bias.data() + r_off, rs, trig_ptr,
                    trig_stride, state_base, rs, dec_phase, dec_base, fv_ptr,
                    chk_c, kBatch, has_dec ? chk_dec : nullptr);

                const SizeType max_lb = m_surrogate_carry_ring - 1;
                for (SizeType l = 0; l < sub_n; ++l) {
                    const SizeType r = r_off + l;
                    if (m_surrogate_plan.disabled_row[r] != 0U) {
                        continue;
                    }
                    const SizeType idx   = (beam * m_rows.size()) + r;
                    const uint64_t s0    = lane_in[q0 + l].sample_begin + t0;
                    const uint64_t s_end = s0 + n;
                    auto& row_st         = m_surrogate_rows[idx];
                    bool any_pos_c       = false;
                    for (SizeType k = 0; k < np; ++k) {
                        if (state_base[(k * rs) + l] > 0.0F) {
                            any_pos_c = true;
                            break;
                        }
                    }
                    const bool hit =
                        ((hit_mask >> static_cast<uint32_t>(l)) & 1U) != 0U;
                    const uint64_t sync_base =
                        row_st.initialized ? row_st.synced_sample : 0ULL;
                    const bool need_cap_sync =
                        any_pos_c &&
                        ((s_end - sync_base) + kGateTile + 4ULL > max_lb);

                    if (hit || m_surrogate_open[idx] != 0U || need_cap_sync) {
                        std::span<const float> lb{};
                        if (!row_st.initialized || row_st.synced_sample != s0) {
                            lb = build_surrogate_lookback(tid, r, beam,
                                                          lane_in[q0 + l], t0);
                        }
                        std::array<float, kadane::kMaxSurrogatePasses>
                            r_biases{};
                        std::array<float, kadane::kMaxSurrogatePasses>
                            r_trigs{};
                        std::array<float, kadane::kMaxSurrogatePasses> r_chk{};
                        for (SizeType k = 0; k < np; ++k) {
                            r_biases[k] = m_surrogate_plan.row_bias(k, r);
                            r_trigs[k] =
                                any_custom ? thr_cuts[(k * kBatch) + l]
                                           : m_surrogate_plan.row_trigger(k, r);
                            r_chk[k] = chk_c[(k * kBatch) + l];
                        }
                        const uint64_t eff_wu =
                            m_config.skip_cold_prefix ? m_rows[r].warmup : 0ULL;
                        bool trig_open = false;
                        for (uint64_t cur_s = s0; cur_s < s_end;) {
                            uint64_t cur_end = s_end;
                            if (m_noise_on) {
                                for (const auto& e : m_ring) {
                                    if (e.first_sample > cur_s &&
                                        e.first_sample < cur_end) {
                                        cur_end = e.first_sample;
                                    }
                                }
                            }
                            const auto off = static_cast<SizeType>(cur_s - s0);
                            const auto sub_n_s =
                                static_cast<SizeType>(cur_end - cur_s);
                            const SizeType sub_fv =
                                fv[l] > off ? std::min(sub_n_s, fv[l] - off)
                                            : SizeType{0};
                            TileTables sub_tbl{};
                            const TileTables* tbl_ptr = nullptr;
                            if (m_noise_on) {
                                if (cur_s == s0 && cur_end == s_end) {
                                    tbl_ptr = &lane_tables[l];
                                } else {
                                    const SizeType p  = m_row_plane[r];
                                    const SizeType dm = r - m_offsets[p];
                                    const float* sub_thr =
                                        tile_tables(m_planes[p], dm, r, cur_s,
                                                    sub_n_s, tid, sub_tbl);
                                    compute_surrogate_row_triggers(r, sub_thr,
                                                                   r_trigs);
                                    tbl_ptr = &sub_tbl;
                                }
                            }
                            const auto prev_passes = row_st.passes;
                            trig_open = kadane::replay_surrogate_row(
                                cur_s, sub_n_s, ptrs[l] + off, np,
                                dec_factors.data(), r_biases.data(),
                                r_trigs.data(), row_st,
                                [&](kadane::SurrogateExcursion ex) {
                                    if (m_noise_on && !ex.has_snr) {
                                        uint32_t filt = 0U;
                                        const float snr =
                                            eval_surrogate_excursion_snr(
                                                r, ex, tbl_ptr, cur_s, cur_end,
                                                filt);
                                        ex.best_snr    = snr;
                                        ex.best_filter = filt;
                                        ex.has_snr     = true;
                                        const auto& prev =
                                            prev_passes[ex.pass_idx];
                                        if (prev.open && prev.has_snr &&
                                            prev.seg_begin == ex.seg_begin &&
                                            prev.best_snr > ex.best_snr &&
                                            active_surrogate_table(
                                                r, prev.best_end,
                                                prev.best_filter) !=
                                                active_surrogate_table(
                                                    r, ex.best_end,
                                                    ex.best_filter)) {
                                            ex.best_snr     = prev.best_snr;
                                            ex.best_filter  = prev.best_filter;
                                            ex.best_end     = prev.best_end;
                                            ex.best_len     = prev.best_len;
                                            ex.best_raw_sum = prev.best_raw_sum;
                                        }
                                    }
                                    handle_surrogate_excursion(r, beam, ex,
                                                               row_st, tbl_ptr,
                                                               cur_s, cur_end);
                                },
                                sub_fv, cur_s == s0 ? r_chk.data() : nullptr,
                                (cur_s == s0 && has_dec) ? chk_dec[l] : 0.0F,
                                cur_s == s0 ? lb : std::span<const float>{},
                                eff_wu);
                            if (m_noise_on) {
                                for (SizeType k = 0; k < np; ++k) {
                                    cache_surrogate_pass_snr(
                                        r, row_st.passes[k],
                                        static_cast<uint32_t>(k), tbl_ptr,
                                        cur_s, cur_end, &prev_passes[k]);
                                }
                            }
                            cur_s = cur_end;
                        }
                        maybe_finalize_surrogate_pending(r, beam, s_end,
                                                         row_st);
                        const bool pend_open =
                            (m_config.peak_mode ==
                             PeakFindingMode::kContiguousPeakAcrossWidths) &&
                            m_peaks[idx * m_peak_count].open;
                        m_surrogate_open[idx] =
                            (trig_open || pend_open) ? uint8_t{1} : uint8_t{0};
                    } else if (!any_pos_c) {
                        const auto ph_end = static_cast<SizeType>(s_end & 3ULL);
                        if (!has_dec || ph_end == 0) {
                            row_st.passes              = {};
                            row_st.dec_partial_sum     = 0.0F;
                            row_st.dec_partial_raw_sum = 0.0;
                            row_st.dec_block_begin     = s_end;
                            row_st.dec_valid_len       = 0U;
                            row_st.initialized         = true;
                            row_st.synced_sample       = s_end;
                        } else if (n >= ph_end) {
                            row_st.passes              = {};
                            row_st.dec_partial_sum     = 0.0F;
                            row_st.dec_partial_raw_sum = 0.0;
                            row_st.dec_block_begin     = s_end - ph_end;
                            row_st.dec_valid_len       = 0U;
                            const uint64_t eff_wu = m_config.skip_cold_prefix
                                                        ? m_rows[r].warmup
                                                        : 0ULL;
                            const float* tail_p   = ptrs[l] + (n - ph_end);
                            for (SizeType q = 0; q < ph_end; ++q) {
                                const uint64_t gt = (s_end - ph_end) + q;
                                const bool valid  = gt >= eff_wu;
                                const float x_eff = valid ? tail_p[q] : 0.0F;
                                if (q == 0) {
                                    row_st.dec_partial_sum = x_eff;
                                    row_st.dec_partial_raw_sum =
                                        valid ? static_cast<double>(x_eff)
                                              : 0.0;
                                    row_st.dec_block_begin =
                                        valid ? gt : (gt + 1ULL);
                                    row_st.dec_valid_len = valid ? 1U : 0U;
                                } else {
                                    row_st.dec_partial_sum =
                                        kadane::ScalarBackend::add_f32(
                                            row_st.dec_partial_sum, x_eff);
                                    if (valid) {
                                        if (row_st.dec_valid_len == 0U) {
                                            row_st.dec_block_begin = gt;
                                        }
                                        row_st.dec_partial_raw_sum +=
                                            static_cast<double>(x_eff);
                                        ++row_st.dec_valid_len;
                                    } else {
                                        row_st.dec_partial_raw_sum = 0.0;
                                        row_st.dec_valid_len       = 0U;
                                    }
                                }
                            }
                            row_st.initialized   = true;
                            row_st.synced_sample = s_end;
                        }
                    }
                }
            }

            for (SizeType l = 0; l < sub_n; ++l) {
                const SizeType r   = r_off + l;
                const SizeType idx = (beam * m_rows.size()) + r;
                save_surrogate_carry(tid, r, beam, lane_in[q0 + l]);
                m_surrogate_pos[idx] =
                    lane_in[q0 + l].sample_begin + total_nsamps;
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

void FDMTBoxcarCpuEngine::sync_surrogate_row_at_pos(SizeType tid,
                                                    SizeType row_idx,
                                                    SizeType beam,
                                                    uint64_t pos) {
    const SizeType np  = m_surrogate_plan.num_passes;
    const SizeType rs  = m_surrogate_plan.row_stride;
    const SizeType idx = (beam * m_rows.size()) + row_idx;
    auto& row_st       = m_surrogate_rows[idx];
    if (row_st.initialized && row_st.synced_sample == pos) {
        return;
    }
    bool any_pos_c = false;
    std::array<float, kadane::kMaxSurrogatePasses> cur_c{};
    for (SizeType k = 0; k < np; ++k) {
        cur_c[k] = m_surrogate_c[(((beam * np) + k) * rs) + row_idx];
        if (cur_c[k] > 0.0F) {
            any_pos_c = true;
        }
    }
    if (!any_pos_c) {
        return;
    }

    const SizeType max_lb = m_surrogate_carry_ring - 1;
    const auto avail   = static_cast<SizeType>(std::min<uint64_t>(pos, max_lb));
    float* lb_buf      = m_surrogate_lb.data() + (tid * max_lb);
    const float* carry = m_carry.data() + (idx * m_surrogate_carry_ring);
    if (avail > 0) {
        auto slot =
            static_cast<SizeType>((pos - avail) % m_surrogate_carry_ring);
        for (SizeType j = 0; j < avail; ++j) {
            lb_buf[j] = carry[slot];
            slot      = (slot + 1 == m_surrogate_carry_ring) ? 0 : (slot + 1);
        }
    }
    const float cur_dec = m_surrogate_plan.has_decimated
                              ? m_surrogate_dec[(beam * rs) + row_idx]
                              : 0.0F;
    const uint64_t eff_wu =
        m_config.skip_cold_prefix ? m_rows[row_idx].warmup : 0ULL;
    std::array<uint32_t, kadane::kMaxSurrogatePasses> dec_factors{};
    std::array<float, kadane::kMaxSurrogatePasses> r_biases{};
    std::array<float, kadane::kMaxSurrogatePasses> r_trigs{};
    for (SizeType k = 0; k < np; ++k) {
        dec_factors[k] = m_surrogate_plan.passes[k].decimate_factor;
        r_biases[k]    = m_surrogate_plan.row_bias(k, row_idx);
        r_trigs[k]     = m_surrogate_plan.row_trigger(k, row_idx);
    }
    if (m_noise_on && !m_ring.empty() && pos > 0) {
        TileTables tbls{};
        const SizeType p  = m_row_plane[row_idx];
        const SizeType dm = row_idx - m_offsets[p];
        const float* screen_thr =
            tile_tables(m_planes[p], dm, row_idx, pos - 1ULL, 1, tid, tbls);
        compute_surrogate_row_triggers(row_idx, screen_thr, r_trigs);
    }
    const auto prev_passes = row_st.passes;
    const auto noop        = [](const kadane::SurrogateExcursion&) noexcept {};
    (void)kadane::replay_surrogate_row(pos, 0, nullptr, np, dec_factors.data(),
                                       r_biases.data(), r_trigs.data(), row_st,
                                       noop, 0, cur_c.data(), cur_dec,
                                       {lb_buf, avail}, eff_wu);
    if (m_noise_on && !m_ring.empty() && pos > 0) {
        for (SizeType k = 0; k < np; ++k) {
            cache_surrogate_pass_snr(row_idx, row_st.passes[k],
                                     static_cast<uint32_t>(k), nullptr, 0, 0,
                                     &prev_passes[k]);
        }
    }
}

void FDMTBoxcarCpuEngine::update_open_begin_surrogate() {
    uint64_t oldest     = UINT64_MAX;
    const SizeType np   = m_surrogate_plan.num_passes;
    const SizeType rs   = m_surrogate_plan.row_stride;
    const SizeType ntot = m_beams * m_rows.size();
    const bool has_dec  = m_surrogate_plan.has_decimated;
    const auto n        = static_cast<std::ptrdiff_t>(ntot);
#pragma omp parallel for num_threads(m_threads) schedule(static)               \
    reduction(min : oldest)
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        const auto idx      = static_cast<SizeType>(i);
        const SizeType beam = idx / m_rows.size();
        const SizeType r    = idx % m_rows.size();
        if (m_surrogate_plan.disabled_row[r] != 0U) {
            continue;
        }
        const uint64_t pos = m_surrogate_pos[idx];
        if (pos > 0) {
            const auto tid = static_cast<SizeType>(omp_get_thread_num());
            sync_surrogate_row_at_pos(tid, r, beam, pos);
            const auto& st = m_surrogate_rows[idx];
            for (SizeType k = 0; k < np; ++k) {
                if (m_surrogate_c[(((beam * np) + k) * rs) + r] > 0.0F &&
                    st.passes[k].open) {
                    oldest = std::min(oldest, st.passes[k].seg_begin);
                }
            }
            const uint64_t eff_wu =
                m_config.skip_cold_prefix ? m_rows[r].warmup : 0ULL;
            if (has_dec && (pos & 3ULL) != 0ULL && pos > eff_wu) {
                oldest = std::min(oldest, std::max(eff_wu, pos & ~uint64_t{3}));
            }
        }
        if (m_config.peak_mode ==
                PeakFindingMode::kContiguousPeakAcrossWidths &&
            m_peaks[idx * m_peak_count].open) {
            oldest = std::min(oldest, m_peaks[idx * m_peak_count].begin);
        }
    }
    m_open_begin = oldest;
}

void FDMTBoxcarCpuEngine::flush_surrogate() {
    const SizeType np = m_surrogate_plan.num_passes;
    const SizeType rs = m_surrogate_plan.row_stride;
    for (SizeType b = 0; b < m_beams; ++b) {
        for (SizeType r = 0; r < m_rows.size(); ++r) {
            if (m_surrogate_plan.disabled_row[r] != 0U) {
                continue;
            }
            const SizeType idx = (b * m_rows.size()) + r;
            const uint64_t pos = m_surrogate_pos[idx];
            if (pos > 0) {
                sync_surrogate_row_at_pos(0, r, b, pos);
            }
            auto& row_st = m_surrogate_rows[idx];
            std::array<float, kadane::kMaxSurrogatePasses> r_trigs{};
            TileTables tbls{};
            const TileTables* tbl_ptr = nullptr;
            if (m_noise_on && !m_ring.empty() && pos > 0) {
                const SizeType p  = m_row_plane[r];
                const SizeType dm = r - m_offsets[p];
                const float* screen_thr =
                    tile_tables(m_planes[p], dm, r, pos - 1ULL, 1, 0, tbls);
                compute_surrogate_row_triggers(r, screen_thr, r_trigs);
                tbl_ptr = &tbls;
            } else {
                compute_surrogate_row_triggers(r, nullptr, r_trigs);
            }
            const uint64_t fl_s = pos > 0 ? (pos - 1ULL) : 0ULL;
            kadane::flush_surrogate_row(
                np, r_trigs.data(), row_st,
                [&](const kadane::SurrogateExcursion& ex) {
                    handle_surrogate_excursion(r, b, ex, row_st, tbl_ptr, fl_s,
                                               pos);
                });
            for (SizeType k = 0; k < np; ++k) {
                m_surrogate_c[(((b * np) + k) * rs) + r] = 0.0F;
            }
            m_surrogate_open[idx] = 0U;
        }
    }
}

} // namespace dmt::algorithms::detail::fdmt_boxcar_cpu
