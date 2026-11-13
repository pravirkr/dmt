#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <omp.h>

#include "dmt/fdmt_variance.hpp"

#include "fdmt_boxcar_cpu_engine.hpp"

// CPU FDMTBoxcar engine: prefix-sum sparse width screen (kPrefix, kNone) and
// the exact per-tile multi-width scorer (score_tile, shared with kKadane).

namespace dmt::algorithms::detail::fdmt_boxcar_cpu {

void FDMTBoxcarCpuEngine::make_screen_groups(
    const std::vector<uint32_t>& widths) {
    m_screen_levels = 0;
    m_groups        = boxcar_screen_groups(widths, m_config.use_trapezoid);
    // The packed screen takes the mean per sample from widths[0]
    // (screen_tile_int()), which must be a plain boxcar.
    if (!m_groups.empty() &&
        (widths.empty() ||
         (m_config.use_trapezoid && trapezoid_edge_width(widths[0]) > 0))) {
        throw std::logic_error(
            "FDMTBoxcar (cpu): the width screen needs a plain first width");
    }
    for (const auto& g : m_groups) {
        m_screen_levels = std::max<SizeType>(m_screen_levels, g.level);
    }
    // Packed rows keep their mean-centred running sums as level 0.
    m_range_min.resize(static_cast<SizeType>(m_threads) * screen_slots() *
                       (m_width + m_tile + 1));
    m_group_thresh.resize(static_cast<SizeType>(m_threads) * m_groups.size());
}

template <typename T, typename Prefix>
void FDMTBoxcarCpuEngine::prefix_tile(const FDMTSearchInput& input,
                                      const BoxcarRow& row,
                                      SizeType beam,
                                      bool root,
                                      SizeType ring0,
                                      SizeType t0,
                                      SizeType n,
                                      Prefix* out) noexcept {
    const auto run = [&](const auto* state_base, const auto* prev_base,
                         const auto* hist_base) {
        // Float rows add in float, then widen (boxcar_sample); integer
        // rows add in uint64_t (boxcar_integer_sample_typed).
        const auto value = [](auto x) {
            if constexpr (std::is_integral_v<T>) {
                return static_cast<uint64_t>(x);
            } else {
                return x;
            }
        };
        if (!(row.unmaterialized || root) || input.materialized_root) {
            const auto* s = state_base + (beam * input.beam_stride) +
                            (row.row * input.row_stride) + t0;
            for (SizeType i = 0; i < n; ++i) {
                out[i + 1] = out[i] + static_cast<Prefix>(value(s[i]));
            }
            return;
        }
        const auto* prev = (prev_base != nullptr ? prev_base : state_base) +
                           (beam * input.beam_stride);
        const auto* tail = prev + (row.tail * input.row_stride) + t0;
        if (row.head == SIZE_MAX) {
            for (SizeType i = 0; i < n; ++i) {
                out[i + 1] = out[i] + static_cast<Prefix>(value(tail[i]));
            }
            return;
        }
        const SizeType split = row.delay > t0 ? std::min(n, row.delay - t0) : 0;
        const auto* hist     = split > 0
                                   ? hist_base + (beam * input.history_stride) +
                                         (row.hist - input.history_base)
                                   : hist_base;
        for (SizeType i = 0; i < split; ++i) {
            const SizeType slot =
                boxcar_history_slot(input, row, t0 + i, ring0);
            out[i + 1] = out[i] + static_cast<Prefix>(value(tail[i]) +
                                                      value(hist[slot]));
        }
        // Sample t reads head-row sample t - delay (t >= delay here).
        const auto* head = prev + (row.head * input.row_stride);
        for (SizeType i = split; i < n; ++i) {
            out[i + 1] =
                out[i] + static_cast<Prefix>(value(tail[i]) +
                                             value(head[t0 + i - row.delay]));
        }
    };
    if constexpr (std::is_integral_v<T>) {
        if (input.integer32_state != nullptr) {
            run(input.integer32_state, input.integer32_prev_state,
                input.integer32_history);
        } else {
            run(input.integer_state, input.integer_prev_state,
                input.integer_history);
        }
    } else {
        run(input.state, input.prev_state, input.history);
    }
}

template <typename T, typename C>
void FDMTBoxcarCpuEngine::score(const plans::BoxcarPlanePlan& plane,
                                SizeType row_idx,
                                SizeType dm,
                                SizeType beam,
                                const FDMTSearchInput& input) {
    const auto tid = static_cast<SizeType>(omp_get_thread_num());
    using Prefix = std::conditional_t<std::is_integral_v<T>, uint64_t, double>;
    Prefix* prefix       = nullptr;
    C* carry             = nullptr;
    const SizeType index = (beam * m_rows.size()) + row_idx;
    if constexpr (std::is_same_v<C, uint32_t>) {
        prefix = m_integer_prefix.data() + (tid * (m_width + m_tile + 1));
        carry  = m_integer_carry32.data() + (index * m_width);
    } else if constexpr (std::is_integral_v<T>) {
        prefix = m_integer_prefix.data() + (tid * (m_width + m_tile + 1));
        carry  = m_integer_carry.data() + (index * m_width);
    } else {
        prefix = m_prefix.data() + (tid * (m_width + m_tile + 1));
        carry  = m_carry.data() + (index * m_width);
    }
    auto* peaks          = m_peaks.data() + (index * m_peak_count);
    const auto& row      = m_rows[row_idx];
    const bool root      = plane.level == m_plan.get_niters();
    const SizeType ring0 = boxcar_ring0(input, row);
    const auto sample_at = [&](SizeType t) {
        if constexpr (std::is_integral_v<T>) {
            return boxcar_integer_sample(input, row, beam, t, root, ring0);
        } else {
            return boxcar_sample(input, row, beam, t, root, ring0);
        }
    };
    // prefix[j] for j <= lookback holds the max_width - 1 samples before
    // the tile (from the carry once per call, then slid along), and
    // prefix[lookback + t + 1] the tile's running sums.
    const SizeType lookback = m_width - 1;
    {
        // Ring slot of sample first - lookback, then advanced by one
        // per entry (one 64-bit remainder per call, not per sample).
        const uint64_t first = input.sample_begin;
        prefix[0]            = 0;
        const SizeType cold =
            first >= lookback ? 0 : lookback - static_cast<SizeType>(first);
        for (SizeType j = 0; j < cold; ++j) {
            prefix[j + 1] = prefix[j];
        }
        if (cold < lookback) {
            auto slot =
                static_cast<SizeType>((first - (lookback - cold)) % m_width);
            for (SizeType j = cold; j < lookback; ++j) {
                prefix[j + 1] = prefix[j] + carry[slot];
                slot          = slot + 1 == m_width ? 0 : slot + 1;
            }
        }
    }
    // Open runs force the bookkeeping until they close.
    const auto any_open = [&] {
        if (m_config.peak_mode == PeakFindingMode::kAllAboveThreshold) {
            return false;
        }
        return std::any_of(peaks, peaks + m_peak_count,
                           [](const BoxcarPeak& p) { return p.open; });
    };
    bool open            = any_open();
    double* group_thresh = nullptr;
    double* range_min    = nullptr;
    // Float rows: the thresholds of the row (TileTables), per tile under
    // per-call masks; the screen's per-width thresholds and the group
    // minima follow them.
    TileTables tables{
        .thr = plane.raw_thresh_lut.data() + (dm * m_filters),
        .inv = plane.inv_sigma_lut.data() + (dm * m_filters),
    };
    const float* screen_thr = tables.thr;
    const float* groups_for = nullptr; // thresholds group_thresh holds
    const auto set_groups   = [&](const float* thr) {
        if (thr == groups_for) {
            return;
        }
        for (SizeType g = 0; g < m_groups.size(); ++g) {
            float th = std::numeric_limits<float>::max();
            for (SizeType f = m_groups[g].f_lo; f < m_groups[g].f_hi; ++f) {
                th = std::min(th, thr[f]);
            }
            group_thresh[g] = th;
        }
        groups_for = thr;
    };
    if (!m_groups.empty()) {
        group_thresh = m_group_thresh.data() + (tid * m_groups.size());
        range_min    = m_range_min.data() +
                       (tid * screen_slots() * (m_width + m_tile + 1));
        for (SizeType g = 0; g < m_groups.size(); ++g) {
            if constexpr (std::is_integral_v<T>) {
                // Mean-centred cutoffs: integer cutoff minus the
                // width's mean (screen_tile_int()). Set above any centered
                // cutoff without relying on Inf under -ffast-math.
                double th = std::numeric_limits<double>::max();
                for (SizeType f = m_groups[g].f_lo; f < m_groups[g].f_hi; ++f) {
                    const SizeType i = (dm * m_filters) + f;
                    th               = std::min(
                        th, static_cast<double>(plane.integer_thresh_lut[i]) -
                                plane.integer_mean_lut[i]);
                }
                group_thresh[g] = th;
            }
        }
    }
    for (SizeType t0 = 0; t0 < input.nsamps; t0 += m_tile) {
        const SizeType n     = std::min(m_tile, input.nsamps - t0);
        const uint64_t start = input.sample_begin + t0;
        prefix_tile<T>(input, row, beam, root, ring0, t0, n, prefix + lookback);
        // Rare-event check: any width over its threshold in this tile.
        // Without it (and no open run) every peak mode is unchanged.
        const bool cold =
            m_config.skip_cold_prefix && start + 1 < row.warmup + m_width;
        if constexpr (!std::is_integral_v<T>) {
            if (m_noise_on) {
                screen_thr =
                    tile_tables(plane, dm, row_idx, start, n, tid, tables);
                if (tables.mixed) {
                    // Mixed tiles share one buffer: new minima each time.
                    groups_for = nullptr;
                }
            }
        }
        bool hit      = m_config.screen_mode == BoxcarScreenMode::kNone;
        bool screened = hit;
        if (!m_groups.empty() && !open) {
            if constexpr (std::is_integral_v<T>) {
                hit = screen_tile_int(plane, dm, row, prefix, lookback, start,
                                      n, cold, group_thresh, range_min);
            } else {
                set_groups(screen_thr);
                hit = screen_tile(plane, screen_thr, row, prefix, lookback,
                                  start, n, cold, group_thresh, range_min);
            }
            screened = true;
        }
        for (SizeType f = 0; f < m_filters && !hit && !screened; ++f) {
            const SizeType width  = plane.widths[f];
            const SizeType w_edge = m_config.use_trapezoid
                                        ? trapezoid_edge_width(plane.widths[f])
                                        : 0;
            const Prefix* lo      = prefix + lookback + 1 - width;
            const Prefix* hi      = prefix + lookback + 1;
            const auto threshold  = [&] {
                if constexpr (std::is_integral_v<T>) {
                    return plane.integer_thresh_lut[(dm * m_filters) + f];
                } else {
                    return static_cast<Prefix>(screen_thr[f]);
                }
            }();
            const uint64_t first_valid = row.warmup + width - 1;
            if (w_edge == 0) {
#pragma omp simd reduction(|| : hit)
                for (SizeType t = 0; t < n; ++t) {
                    hit = hit || (hi[t] - lo[t] >= threshold &&
                                  (!cold || start + t >= first_valid));
                }
            } else {
                const Prefix* lo_in = prefix + lookback + 1 - width + w_edge;
                const Prefix* hi_in = prefix + lookback + 1 - w_edge;
#pragma omp simd reduction(|| : hit)
                for (SizeType t = 0; t < n; ++t) {
                    const Prefix sum = (hi[t] - lo[t]) + (hi_in[t] - lo_in[t]);
                    hit = hit || (sum >= threshold &&
                                  (!cold || start + t >= first_valid));
                }
            }
        }
        if (hit || open) {
            score_tile<T>(plane, dm, row, beam, prefix, lookback, start, n,
                          peaks, row_idx, tables);
            open = any_open();
        }
        // Slide the last lookback prefixes in front of the next tile.
        std::copy(prefix + n, prefix + n + lookback + 1, prefix);
    }
    // Carry the last max_width - 1 samples (at most) to the next call.
    const SizeType keep = std::min(input.nsamps, lookback);
    if (keep > 0) {
        const SizeType t_first = input.nsamps - keep;
        auto slot =
            static_cast<SizeType>((input.sample_begin + t_first) % m_width);
        for (SizeType t = t_first; t < input.nsamps; ++t) {
            carry[slot] = static_cast<C>(sample_at(t));
            slot        = slot + 1 == m_width ? 0 : slot + 1;
        }
    }
}

bool FDMTBoxcarCpuEngine::screen_tile(const plans::BoxcarPlanePlan& plane,
                                      const float* thr_row,
                                      const BoxcarRow& row,
                                      const double* prefix,
                                      SizeType lookback,
                                      uint64_t start,
                                      SizeType n,
                                      bool cold,
                                      const double* group_thresh,
                                      double* range_min) const {
    const SizeType len    = lookback + n + 1; // prefix entries
    const SizeType stride = m_width + m_tile + 1;
    for (SizeType k = 1; k <= m_screen_levels; ++k) {
        const SizeType half = SizeType{1} << (k - 1);
        if (half >= len) {
            break;
        }
        const double* src = k == 1 ? prefix : range_min + ((k - 2) * stride);
        double* dst       = range_min + ((k - 1) * stride);
        const SizeType m  = len - (2 * half) + 1;
#pragma omp simd
        for (SizeType j = 0; j < m; ++j) {
            dst[j] = std::min(src[j], src[j + half]);
        }
    }
    const double* hi = prefix + lookback + 1;
    for (SizeType g = 0; g < m_groups.size(); ++g) {
        const auto& grp     = m_groups[g];
        const double thresh = group_thresh[g];
        const SizeType span = SizeType{1} << grp.level;
        const double* mins =
            grp.level == 0 ? prefix : range_min + ((grp.level - 1) * stride);
        // Window [idx - w_hi, idx - w_lo] as two overlapping 2^k runs.
        const double* a = mins + lookback + 1 - grp.w_hi;
        const double* b = mins + lookback + 1 - grp.w_lo - span + 1;
        // A trapezoid (always a group of one) is not bounded by the
        // range minimum: check it exactly.
        bool any = m_config.use_trapezoid && trapezoid_edge_width(grp.w_lo) > 0;
        if (!any) {
#pragma omp simd reduction(|| : any)
            for (SizeType t = 0; t < n; ++t) {
                any = any || (hi[t] - std::min(a[t], b[t]) >= thresh);
            }
        }
        if (!any) {
            continue;
        }
        for (SizeType f = grp.f_lo; f < grp.f_hi; ++f) {
            const SizeType width = plane.widths[f];
            const SizeType w_edge =
                m_config.use_trapezoid ? trapezoid_edge_width(width) : 0;
            const double* lo           = prefix + lookback + 1 - width;
            const auto th              = static_cast<double>(thr_row[f]);
            const uint64_t first_valid = row.warmup + width - 1;
            bool hit                   = false;
            if (w_edge == 0) {
#pragma omp simd reduction(|| : hit)
                for (SizeType t = 0; t < n; ++t) {
                    hit = hit || (hi[t] - lo[t] >= th &&
                                  (!cold || start + t >= first_valid));
                }
            } else {
                const double* lo_in = prefix + lookback + 1 - width + w_edge;
                const double* hi_in = prefix + lookback + 1 - w_edge;
#pragma omp simd reduction(|| : hit)
                for (SizeType t = 0; t < n; ++t) {
                    const double sum = (hi[t] - lo[t]) + (hi_in[t] - lo_in[t]);
                    hit = hit ||
                          (sum >= th && (!cold || start + t >= first_valid));
                }
            }
            if (hit) {
                return true;
            }
        }
    }
    return false;
}

bool FDMTBoxcarCpuEngine::screen_tile_int(const plans::BoxcarPlanePlan& plane,
                                          SizeType dm,
                                          const BoxcarRow& row,
                                          const uint64_t* prefix,
                                          SizeType lookback,
                                          uint64_t start,
                                          SizeType n,
                                          bool cold,
                                          const double* group_thresh,
                                          double* range_min) const {
    const SizeType len    = lookback + n + 1;
    const SizeType stride = m_width + m_tile + 1;
    // m1 from the first boxcar width (widths 1 and 2 are never
    // trapezoids); a killed row has mean 0 and infinite cutoffs.
    const double m1 = plane.integer_mean_lut[dm * m_filters] /
                      static_cast<double>(plane.widths[0]);
    double* centred = range_min;
#pragma omp simd
    for (SizeType j = 0; j < len; ++j) {
        centred[j] =
            static_cast<double>(prefix[j]) - (m1 * static_cast<double>(j));
    }
    const double scale = static_cast<double>(prefix[len - 1]) +
                         (std::abs(m1) * static_cast<double>(len)) + 1.0;
    for (SizeType k = 1; k <= m_screen_levels; ++k) {
        const SizeType half = SizeType{1} << (k - 1);
        if (half >= len) {
            break;
        }
        const double* src = range_min + ((k - 1) * stride);
        double* dst       = range_min + (k * stride);
        const SizeType m  = len - (2 * half) + 1;
#pragma omp simd
        for (SizeType j = 0; j < m; ++j) {
            dst[j] = std::min(src[j], src[j + half]);
        }
    }
    const double* hi = centred + lookback + 1;
    for (SizeType g = 0; g < m_groups.size(); ++g) {
        const auto& grp = m_groups[g];
        const double margin =
            (1e-12 * (scale + std::abs(group_thresh[g]))) + 1.0;
        const double thresh = group_thresh[g] - margin;
        const SizeType span = SizeType{1} << grp.level;
        const double* mins  = range_min + (grp.level * stride);
        const double* a     = mins + lookback + 1 - grp.w_hi;
        const double* b     = mins + lookback + 1 - grp.w_lo - span + 1;
        bool any = m_config.use_trapezoid && trapezoid_edge_width(grp.w_lo) > 0;
        if (!any) {
#pragma omp simd reduction(|| : any)
            for (SizeType t = 0; t < n; ++t) {
                any = any || (hi[t] - std::min(a[t], b[t]) >= thresh);
            }
        }
        if (!any) {
            continue;
        }
        const uint64_t* phi = prefix + lookback + 1;
        for (SizeType f = grp.f_lo; f < grp.f_hi; ++f) {
            const SizeType width = plane.widths[f];
            const SizeType w_edge =
                m_config.use_trapezoid ? trapezoid_edge_width(width) : 0;
            const uint64_t* lo = prefix + lookback + 1 - width;
            const uint64_t th  = plane.integer_thresh_lut[(dm * m_filters) + f];
            const uint64_t first_valid = row.warmup + width - 1;
            bool hit                   = false;
            if (w_edge == 0) {
#pragma omp simd reduction(|| : hit)
                for (SizeType t = 0; t < n; ++t) {
                    hit = hit || (phi[t] - lo[t] >= th &&
                                  (!cold || start + t >= first_valid));
                }
            } else {
                const uint64_t* lo_in = prefix + lookback + 1 - width + w_edge;
                const uint64_t* hi_in = prefix + lookback + 1 - w_edge;
#pragma omp simd reduction(|| : hit)
                for (SizeType t = 0; t < n; ++t) {
                    const uint64_t sum =
                        (phi[t] - lo[t]) + (hi_in[t] - lo_in[t]);
                    hit = hit ||
                          (sum >= th && (!cold || start + t >= first_valid));
                }
            }
            if (hit) {
                return true;
            }
        }
    }
    return false;
}

template <typename T, typename Prefix>
void FDMTBoxcarCpuEngine::score_tile(const plans::BoxcarPlanePlan& plane,
                                     SizeType dm,
                                     const BoxcarRow& row,
                                     SizeType beam,
                                     const Prefix* prefix,
                                     SizeType lookback,
                                     uint64_t start,
                                     SizeType n,
                                     BoxcarPeak* peaks,
                                     SizeType row_idx,
                                     const TileTables& tables) {
    const auto tid        = static_cast<SizeType>(omp_get_thread_num());
    float* best           = m_best.data() + (tid * m_tile);
    uint32_t* best_filter = m_best_filter.data() + (tid * m_tile);
    const auto emit_peak  = [&](const BoxcarPeak& peak) {
        emit(row_idx, beam, peak);
    };
    for (SizeType t = 0; t < n; ++t) {
        best[t]        = -1.0F;
        best_filter[t] = 0;
    }
    for (SizeType f = 0; f < m_filters; ++f) {
        const SizeType width = plane.widths[f];
        const SizeType w_edge =
            m_config.use_trapezoid ? trapezoid_edge_width(plane.widths[f]) : 0;
        // Float rows take the tile's tables; packed rows the planes' (no
        // per-call masks). Mixed tiles go in runs of samples with one
        // table each; otherwise the tile is one run. The per-sample body
        // stays inline here: moved into a lambda the scorer ran 35%
        // slower, even forced inline.
        const float* thr_row =
            std::is_integral_v<T>
                ? plane.raw_thresh_lut.data() + (dm * m_filters)
                : tables.thr;
        const float* inv_row =
            std::is_integral_v<T>
                ? plane.inv_sigma_lut.data() + (dm * m_filters)
                : tables.inv;
        const bool mixed = !std::is_integral_v<T> && tables.mixed;
        for (SizeType t_run = 0; t_run < n;) {
            SizeType t_end  = n;
            float threshold = thr_row[f];
            float inv_sigma = inv_row[f];
            if (mixed) {
                const auto [table, until] =
                    sample_table(start + t_run, width, tables.support);
                t_end = until - start < n ? static_cast<SizeType>(until - start)
                                          : n;
                threshold = table_thr(table, tables.thr, row_idx, m_filters)[f];
                inv_sigma = table_inv(table, tables.inv, row_idx, m_filters)[f];
            }
            for (SizeType t = t_run; t < t_end; ++t) {
                const uint64_t sample = start + t;
                const SizeType idx    = lookback + t + 1;
                Prefix sum            = prefix[idx] - prefix[idx - width];
                if (w_edge > 0) {
                    sum += prefix[idx - w_edge] - prefix[idx - width + w_edge];
                }
                const bool valid = !m_config.skip_cold_prefix ||
                                   sample >= row.warmup + width - 1;
                bool above       = false;
                float snr        = 0;
                if constexpr (std::is_integral_v<T>) {
                    above =
                        valid &&
                        sum >= plane.integer_thresh_lut[(dm * m_filters) + f];
                    if (above &&
                        m_config.peak_mode ==
                            PeakFindingMode::kContiguousPeakAcrossWidths) {
                        snr = static_cast<float>(
                            (static_cast<double>(sum) -
                             plane.integer_mean_lut[(dm * m_filters) + f]) *
                            (inv_sigma * m_integer_unit));
                    }
                } else {
                    above = valid && sum >= threshold;
                    snr   = above ? static_cast<float>(sum * inv_sigma) : 0.0F;
                }
                if (m_config.peak_mode ==
                    PeakFindingMode::kContiguousPeakAcrossWidths) {
                    if (above && snr > best[t]) {
                        best[t]        = snr;
                        best_filter[t] = static_cast<uint32_t>(f);
                    }
                } else if (m_config.peak_mode ==
                           PeakFindingMode::kAllAboveThreshold) {
                    if (above) {
                        BoxcarPeak peak{sample,
                                        sample,
                                        sample,
                                        snr,
                                        static_cast<uint32_t>(f),
                                        true};
                        if constexpr (std::is_integral_v<T>) {
                            peak.raw_sum = sum;
                        }
                        emit_peak(peak);
                    }
                } else {
                    if constexpr (std::is_integral_v<T>) {
                        boxcar_integer_peak_step(peaks[f], above, sum, sample,
                                                 static_cast<uint32_t>(f),
                                                 emit_peak);
                    } else {
                        boxcar_peak_step(peaks[f], above, snr, sample,
                                         static_cast<uint32_t>(f), emit_peak);
                    }
                }
            }
            t_run = t_end;
        }
    }
    if (m_config.peak_mode == PeakFindingMode::kContiguousPeakAcrossWidths) {
        for (SizeType t = 0; t < n; ++t) {
            boxcar_peak_step(peaks[0], best[t] >= 0, best[t], start + t,
                             best_filter[t], emit_peak);
        }
    }
}

template void
FDMTBoxcarCpuEngine::prefix_tile<float, double>(const FDMTSearchInput&,
                                                const BoxcarRow&,
                                                SizeType,
                                                bool,
                                                SizeType,
                                                SizeType,
                                                SizeType,
                                                double*) noexcept;
template void
FDMTBoxcarCpuEngine::prefix_tile<uint64_t, uint64_t>(const FDMTSearchInput&,
                                                     const BoxcarRow&,
                                                     SizeType,
                                                     bool,
                                                     SizeType,
                                                     SizeType,
                                                     SizeType,
                                                     uint64_t*) noexcept;

template void
FDMTBoxcarCpuEngine::score<float, float>(const plans::BoxcarPlanePlan&,
                                         SizeType,
                                         SizeType,
                                         SizeType,
                                         const FDMTSearchInput&);
template void
FDMTBoxcarCpuEngine::score<uint64_t, uint32_t>(const plans::BoxcarPlanePlan&,
                                               SizeType,
                                               SizeType,
                                               SizeType,
                                               const FDMTSearchInput&);
template void
FDMTBoxcarCpuEngine::score<uint64_t, uint64_t>(const plans::BoxcarPlanePlan&,
                                               SizeType,
                                               SizeType,
                                               SizeType,
                                               const FDMTSearchInput&);

template void
FDMTBoxcarCpuEngine::score_tile<float, double>(const plans::BoxcarPlanePlan&,
                                               SizeType,
                                               const BoxcarRow&,
                                               SizeType,
                                               const double*,
                                               SizeType,
                                               uint64_t,
                                               SizeType,
                                               BoxcarPeak*,
                                               SizeType,
                                               const TileTables&);
template void FDMTBoxcarCpuEngine::score_tile<uint64_t, uint64_t>(
    const plans::BoxcarPlanePlan&,
    SizeType,
    const BoxcarRow&,
    SizeType,
    const uint64_t*,
    SizeType,
    uint64_t,
    SizeType,
    BoxcarPeak*,
    SizeType,
    const TileTables&);

} // namespace dmt::algorithms::detail::fdmt_boxcar_cpu
