#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <omp.h>

#include "dmt/bit_pack_utils.hpp"
#include "dmt/boxcar.hpp"
#include "dmt/fdmt_variance.hpp"

namespace dmt::algorithms::detail {
namespace {

constexpr SizeType kTile = 1024;

class FDMTBoxcarCpuEngine final : public FDMTBoxcarEngine {
public:
    FDMTBoxcarCpuEngine(const plans::FDMTPlan& plan,
                        const BoxcarConfig& config,
                        const std::vector<plans::BoxcarPlanePlan>& planes,
                        const FDMTEngineConfig& cfg,
                        std::unique_ptr<BoxcarDriver> driver)
        : m_plan(plan),
          m_config(config),
          m_planes(planes),
          m_rows(make_boxcar_rows(plan, planes)),
          m_beams(cfg.nbeams),
          m_threads(std::max(1, cfg.exec.nthreads)),
          m_width(planes.front().widths.back()),
          m_filters(planes.front().widths.size()),
          m_peak_count(config.peak_mode ==
                               PeakFindingMode::kContiguousPeakPerWidth
                           ? m_filters
                           : 1),
          m_driver(std::move(driver)) {
        if (m_rows.size() > SIZE_MAX / m_beams / m_width ||
            m_rows.size() > SIZE_MAX / m_beams / m_peak_count) {
            throw std::invalid_argument("FDMTBoxcar: history size overflow");
        }
        const SizeType rows = m_rows.size() * m_beams;
        m_integer_unit =
            cfg.use_box_smearing
                ? std::ldexp(1.0,
                             -static_cast<int>(plan.get_box_fraction_bits()))
                : 1.0;
        if (config.input_nbits == 32) {
            m_carry.resize(rows * m_width);
        } else {
            m_integer_carry.resize(rows * m_width);
        }
        m_peaks.resize(rows * m_peak_count);
        m_per_dm.resize(rows);
        m_candidates.resize(config.max_candidates);
        if (config.input_nbits == 32) {
            m_prefix.resize(m_threads * (m_width + kTile + 1));
        } else {
            m_integer_prefix.resize(m_threads * (m_width + kTile + 1));
        }
        m_best.resize(m_threads * kTile);
        m_best_filter.resize(m_threads * kTile);
        make_screen_groups(planes.front().widths);
        m_offsets.push_back(0);
        for (const auto& p : planes) {
            m_offsets.push_back(m_offsets.back() + p.dm_grid.size());
            m_integer_means.insert(m_integer_means.end(),
                                   p.integer_mean_lut.begin(),
                                   p.integer_mean_lut.end());
            if (config.input_nbits != 32) {
                for (const float inv : p.inv_sigma_lut) {
                    m_integer_scales.push_back(inv * m_integer_unit);
                }
            }
        }
    }

    CandidateResultView execute(std::span<const float> waterfall) override {
        if (m_config.input_nbits != 32 ||
            waterfall.size() !=
                m_beams * m_plan.get_nchans() * m_plan.get_nsamps()) {
            throw std::invalid_argument("FDMTBoxcar: wrong waterfall size");
        }
        if (m_position > UINT64_MAX - m_plan.get_nsamps()) {
            throw std::overflow_error(
                "FDMTBoxcar: stream sample index overflow");
        }
        begin_result(m_plan.get_nsamps());
        m_driver->search(waterfall, *this);
        m_position += m_plan.get_nsamps();
        return view();
    }
    CandidateDeviceResultView execute(DeviceSpan<const float>,
                                      Stream) override {
        throw_no_device_memory("FDMTBoxcar::execute", Backend::kCPU);
    }
    CandidateResultView execute(std::span<const uint8_t> waterfall) override {
        if (m_config.input_nbits == 32 ||
            waterfall.size() !=
                m_beams * m_plan.get_nchans() *
                    bit_pack_utils::packed_row_bytes(m_plan.get_nsamps(),
                                                     m_config.input_nbits)) {
            throw std::invalid_argument(
                "FDMTBoxcar: packed input size or bit depth mismatch");
        }
        if (m_position > UINT64_MAX - m_plan.get_nsamps()) {
            throw std::overflow_error(
                "FDMTBoxcar: stream sample index overflow");
        }
        begin_result(m_plan.get_nsamps());
        m_driver->search(waterfall, m_config.input_nbits, *this);
        m_position += m_plan.get_nsamps();
        return view();
    }
    CandidateDeviceResultView execute(DeviceSpan<const uint8_t>,
                                      Stream) override {
        throw_no_device_memory("FDMTBoxcar::execute", Backend::kCPU);
    }
    CandidateDeviceResultView flush_device(Stream) override {
        throw_no_device_memory("FDMTBoxcar::flush_device", Backend::kCPU);
    }
    CandidateResultView copy_result(Stream) override { return view(); }
    CandidateResultView flush() override {
        begin_result(0);
        for (SizeType b = 0; b < m_beams; ++b) {
            for (SizeType row = 0; row < m_rows.size(); ++row) {
                const SizeType index = b * m_rows.size() + row;
                for (SizeType f = 0; f < m_peak_count; ++f) {
                    auto& peak = m_peaks[index * m_peak_count + f];
                    if (peak.open) {
                        emit(row, b, peak);
                        peak.open = false;
                    }
                }
            }
        }
        return view();
    }
    void reset_history(Stream) override {
        m_driver->reset({});
        std::ranges::fill(m_carry, 0.0F);
        std::ranges::fill(m_integer_carry, uint64_t{0});
        std::ranges::fill(m_peaks, BoxcarPeak{});
        m_position = 0;
        begin_result(0);
    }

    void consume(SizeType level, const FDMTSearchInput& input) override {
        for (SizeType p = 0; p < m_planes.size(); ++p) {
            if (m_planes[p].level != level) {
                continue;
            }
            const auto& plane    = m_planes[p];
            const SizeType ndm   = plane.dm_grid.size();
            const SizeType base  = m_offsets[p];
            const SizeType first = input.beam_begin;
            const SizeType beams =
                std::min(input.beam_count, m_beams - input.beam_begin);
#pragma omp parallel for num_threads(m_threads) schedule(static)
            for (SizeType task = 0; task < beams * ndm; ++task) {
                const SizeType beam    = first + task / ndm;
                const SizeType d       = task % ndm;
                const SizeType row_idx = base + d;
                if (m_config.input_nbits == 32) {
                    score<float>(plane, row_idx, d, beam, input);
                } else {
                    score<uint64_t>(plane, row_idx, d, beam, input);
                }
            }
        }
    }

    FDMTBoxcarMemoryUsage memory_usage() const noexcept override {
        const auto fdmt     = m_driver->memory();
        SizeType plan_bytes = m_rows.size() * sizeof(BoxcarRow) +
                              m_offsets.size() * sizeof(SizeType);
        for (const auto& p : m_planes) {
            plan_bytes +=
                sizeof(p) + p.widths.size() * sizeof(uint32_t) +
                p.dm_grid.size() * sizeof(float) +
                (p.warmup_samples.size() + p.integer_thresh_lut.size()) *
                    sizeof(uint64_t) +
                p.integer_mean_lut.size() * sizeof(double) +
                (p.inv_sigma_lut.size() + p.raw_thresh_lut.size()) *
                    sizeof(float);
        }
        return {.plan       = fdmt.plan + plan_bytes,
                .state      = fdmt.state,
                .history    = fdmt.history + m_carry.size() * sizeof(float) +
                              m_integer_carry.size() * sizeof(uint64_t),
                .peaks      = m_peaks.size() * sizeof(BoxcarPeak),
                .workspace  = fdmt.workspace +
                              (m_prefix.size() + m_integer_means.size() +
                               m_integer_scales.size() + m_range_min.size() +
                               m_group_thresh.size()) *
                                  sizeof(double) +
                              m_integer_prefix.size() * sizeof(uint64_t) +
                              m_best.size() * sizeof(float) +
                              m_best_filter.size() * sizeof(uint32_t) +
                              m_per_dm.size() * sizeof(uint64_t),
                .candidates = m_candidates.size() * sizeof(RawCandidate)};
    }
    SizeType chunk_nsamps() const noexcept override {
        return m_driver->chunk_nsamps();
    }
    SizeType fuse_levels() const noexcept override {
        return m_driver->fuse_levels();
    }

private:
    void make_screen_groups(const std::vector<uint32_t>& widths) {
        m_screen_levels = 0;
        m_groups        = boxcar_screen_groups(widths, m_config.use_trapezoid);
        for (const auto& g : m_groups) {
            m_screen_levels = std::max<SizeType>(m_screen_levels, g.level);
        }
        // Packed rows keep their mean-centred running sums as level 0.
        m_range_min.resize(m_threads * screen_slots() * (m_width + kTile + 1));
        m_group_thresh.resize(m_threads * m_groups.size());
    }

    [[nodiscard]] SizeType screen_slots() const noexcept {
        return m_screen_levels + (m_config.input_nbits != 32 ? 1 : 0);
    }

    void begin_result(SizeType samples) {
        m_counts = {};
        m_next_slot.store(0, std::memory_order_relaxed);
        std::ranges::fill(m_per_dm, 0);
        m_result_begin   = m_position;
        m_result_samples = samples;
    }
    CandidateResultView view() const {
        return {m_counts,
                {m_candidates.data(),
                 static_cast<SizeType>(m_counts.emitted_count)},
                m_result_begin,
                m_result_samples};
    }
    void emit(SizeType row, SizeType beam, const BoxcarPeak& peak) {
#pragma omp atomic
        ++m_counts.total_detected;
        auto& count = m_per_dm[beam * m_rows.size() + row];
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
        auto candidate         = m_rows[row].candidate;
        candidate.beam_idx     = static_cast<uint32_t>(beam);
        candidate.peak_sample  = peak.peak;
        candidate.begin_sample = peak.begin;
        candidate.end_sample   = peak.end;
        candidate.snr          = peak.snr;
        if (m_config.input_nbits != 32 &&
            m_config.peak_mode !=
                PeakFindingMode::kContiguousPeakAcrossWidths) {
            const SizeType i = row * m_filters + peak.filter;
            candidate.snr    = static_cast<float>(
                (static_cast<double>(peak.raw_sum) - m_integer_means[i]) *
                m_integer_scales[i]);
        }
        candidate.filter_idx = peak.filter;
        candidate.width      = m_planes.front().widths[peak.filter];
        m_candidates[slot]   = candidate;
#pragma omp atomic
        ++m_counts.emitted_count;
    }
    template <typename T>
    void score(const plans::BoxcarPlanePlan& plane,
               SizeType row_idx,
               SizeType dm,
               SizeType beam,
               const FDMTSearchInput& input) {
        const SizeType tid = static_cast<SizeType>(omp_get_thread_num());
        using Prefix =
            std::conditional_t<std::is_integral_v<T>, uint64_t, double>;
        Prefix* prefix;
        T* carry;
        const SizeType index = beam * m_rows.size() + row_idx;
        if constexpr (std::is_integral_v<T>) {
            prefix = m_integer_prefix.data() + tid * (m_width + kTile + 1);
            carry  = m_integer_carry.data() + index * m_width;
        } else {
            prefix = m_prefix.data() + tid * (m_width + kTile + 1);
            carry  = m_carry.data() + index * m_width;
        }
        auto* peaks          = m_peaks.data() + index * m_peak_count;
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
            const uint64_t first = input.sample_begin;
            prefix[0]            = 0;
            for (SizeType j = 0; j < lookback; ++j) {
                const SizeType back = lookback - j;
                const T x =
                    first >= back ? carry[(first - back) % m_width] : T{0};
                prefix[j + 1] = prefix[j] + x;
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
        if (!m_groups.empty()) {
            group_thresh = m_group_thresh.data() + tid * m_groups.size();
            range_min    = m_range_min.data() +
                           tid * screen_slots() * (m_width + kTile + 1);
            for (SizeType g = 0; g < m_groups.size(); ++g) {
                if constexpr (std::is_integral_v<T>) {
                    // Mean-centred cutoffs: integer cutoff minus the
                    // width's mean (screen_tile_int()).
                    double th = std::numeric_limits<double>::infinity();
                    for (SizeType f = m_groups[g].f_lo; f < m_groups[g].f_hi;
                         ++f) {
                        const SizeType i = dm * m_filters + f;
                        th = std::min(th, static_cast<double>(
                                              plane.integer_thresh_lut[i]) -
                                              plane.integer_mean_lut[i]);
                    }
                    group_thresh[g] = th;
                } else {
                    float th = std::numeric_limits<float>::max();
                    for (SizeType f = m_groups[g].f_lo; f < m_groups[g].f_hi;
                         ++f) {
                        th = std::min(th,
                                      plane.raw_thresh_lut[dm * m_filters + f]);
                    }
                    group_thresh[g] = th;
                }
            }
        }
        for (SizeType t0 = 0; t0 < input.nsamps; t0 += kTile) {
            const SizeType n     = std::min(kTile, input.nsamps - t0);
            const uint64_t start = input.sample_begin + t0;
            for (SizeType t = 0; t < n; ++t) {
                prefix[lookback + t + 1] =
                    prefix[lookback + t] + sample_at(t0 + t);
            }
            // Rare-event check: any width over its threshold in this tile.
            // Without it (and no open run) every peak mode is unchanged.
            const bool cold =
                m_config.skip_cold_prefix && start + 1 < row.warmup + m_width;
            bool hit      = false;
            bool screened = false;
            if (!m_groups.empty() && !open) {
                if constexpr (std::is_integral_v<T>) {
                    hit =
                        screen_tile_int(plane, dm, row, prefix, lookback, start,
                                        n, cold, group_thresh, range_min);
                } else {
                    hit = screen_tile(plane, dm, row, prefix, lookback, start,
                                      n, cold, group_thresh, range_min);
                }
                screened = true;
            }
            for (SizeType f = 0; f < m_filters && !hit && !screened; ++f) {
                const SizeType width = plane.widths[f];
                const SizeType w_edge =
                    m_config.use_trapezoid
                        ? trapezoid_edge_width(plane.widths[f])
                        : 0;
                const Prefix* lo     = prefix + lookback + 1 - width;
                const Prefix* hi     = prefix + lookback + 1;
                const auto threshold = [&] {
                    if constexpr (std::is_integral_v<T>) {
                        return plane.integer_thresh_lut[dm * m_filters + f];
                    } else {
                        return static_cast<Prefix>(
                            plane.raw_thresh_lut[dm * m_filters + f]);
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
                    const Prefix* lo_in =
                        prefix + lookback + 1 - width + w_edge;
                    const Prefix* hi_in = prefix + lookback + 1 - w_edge;
#pragma omp simd reduction(|| : hit)
                    for (SizeType t = 0; t < n; ++t) {
                        const Prefix sum =
                            (hi[t] - lo[t]) + (hi_in[t] - lo_in[t]);
                        hit = hit || (sum >= threshold &&
                                      (!cold || start + t >= first_valid));
                    }
                }
            }
            if (hit || open) {
                score_tile<T>(plane, dm, row, beam, prefix, lookback, start, n,
                              peaks, row_idx);
                open = any_open();
            }
            // Slide the last lookback prefixes in front of the next tile.
            std::copy(prefix + n, prefix + n + lookback + 1, prefix);
        }
        // Carry the last max_width - 1 samples (at most) to the next call.
        const SizeType keep = std::min(input.nsamps, lookback);
        for (SizeType t = input.nsamps - keep; t < input.nsamps; ++t) {
            carry[(input.sample_begin + t) % m_width] = sample_at(t);
        }
    }

    /**
     * Whether any width exceeds its threshold in this tile (float rows, no
     * open run), as the per-width check in score() decides. A group's
     * bound P(t) - min(P[t - w_hi .. t - w_lo]) >= P(t) - P(t - w) for
     * each of its widths w (rounding is monotonic), so a group whose bound
     * stays under its smallest threshold has no hit; the others are checked
     * width by width. range_min level k holds min(P[j .. j + 2^k - 1]).
     */
    bool screen_tile(const plans::BoxcarPlanePlan& plane,
                     SizeType dm,
                     const BoxcarRow& row,
                     const double* prefix,
                     SizeType lookback,
                     uint64_t start,
                     SizeType n,
                     bool cold,
                     const double* group_thresh,
                     double* range_min) const {
        const SizeType len    = lookback + n + 1; // prefix entries
        const SizeType stride = m_width + kTile + 1;
        for (SizeType k = 1; k <= m_screen_levels; ++k) {
            const SizeType half = SizeType{1} << (k - 1);
            if (half >= len) {
                break;
            }
            const double* src = k == 1 ? prefix : range_min + (k - 2) * stride;
            double* dst       = range_min + (k - 1) * stride;
            const SizeType m  = len - 2 * half + 1;
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
                grp.level == 0 ? prefix : range_min + (grp.level - 1) * stride;
            // Window [idx - w_hi, idx - w_lo] as two overlapping 2^k runs.
            const double* a = mins + lookback + 1 - grp.w_hi;
            const double* b = mins + lookback + 1 - grp.w_lo - span + 1;
            // A trapezoid (always a group of one) is not bounded by the
            // range minimum: check it exactly.
            bool any =
                m_config.use_trapezoid && trapezoid_edge_width(grp.w_lo) > 0;
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
                const double* lo = prefix + lookback + 1 - width;
                const auto th    = static_cast<double>(
                    plane.raw_thresh_lut[dm * m_filters + f]);
                const uint64_t first_valid = row.warmup + width - 1;
                bool hit                   = false;
                if (w_edge == 0) {
#pragma omp simd reduction(|| : hit)
                    for (SizeType t = 0; t < n; ++t) {
                        hit = hit || (hi[t] - lo[t] >= th &&
                                      (!cold || start + t >= first_valid));
                    }
                } else {
                    const double* lo_in =
                        prefix + lookback + 1 - width + w_edge;
                    const double* hi_in = prefix + lookback + 1 - w_edge;
#pragma omp simd reduction(|| : hit)
                    for (SizeType t = 0; t < n; ++t) {
                        const double sum =
                            (hi[t] - lo[t]) + (hi_in[t] - lo_in[t]);
                        hit = hit || (sum >= th &&
                                      (!cold || start + t >= first_valid));
                    }
                }
                if (hit) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * screen_tile() for packed rows (exact uint64 running sums P, which only
     * grow, so the raw bound is useless). With m1 the row's mean per sample
     * (integer_mean_lut[f] = m1 * w for a boxcar), D(j) = P(j) - m1 j gives
     * S_w - m1 w = D(t) - D(t - w), so D(t) - min(D[t - w_hi .. t - w_lo])
     * bounds every width's centred sum and is compared with the group's
     * smallest cutoff - mean. D is in double: a group passes within a
     * margin far above the rounding of D and of the cutoffs, so the screen
     * stays conservative; passing groups are checked exactly in integers.
     */
    bool screen_tile_int(const plans::BoxcarPlanePlan& plane,
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
        const SizeType stride = m_width + kTile + 1;
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
            const double* src = range_min + (k - 1) * stride;
            double* dst       = range_min + k * stride;
            const SizeType m  = len - 2 * half + 1;
#pragma omp simd
            for (SizeType j = 0; j < m; ++j) {
                dst[j] = std::min(src[j], src[j + half]);
            }
        }
        const double* hi = centred + lookback + 1;
        for (SizeType g = 0; g < m_groups.size(); ++g) {
            const auto& grp = m_groups[g];
            const double margin =
                1e-12 * (scale + std::abs(group_thresh[g])) + 1.0;
            const double thresh = group_thresh[g] - margin;
            const SizeType span = SizeType{1} << grp.level;
            const double* mins  = range_min + grp.level * stride;
            const double* a     = mins + lookback + 1 - grp.w_hi;
            const double* b     = mins + lookback + 1 - grp.w_lo - span + 1;
            bool any =
                m_config.use_trapezoid && trapezoid_edge_width(grp.w_lo) > 0;
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
                const uint64_t th =
                    plane.integer_thresh_lut[dm * m_filters + f];
                const uint64_t first_valid = row.warmup + width - 1;
                bool hit                   = false;
                if (w_edge == 0) {
#pragma omp simd reduction(|| : hit)
                    for (SizeType t = 0; t < n; ++t) {
                        hit = hit || (phi[t] - lo[t] >= th &&
                                      (!cold || start + t >= first_valid));
                    }
                } else {
                    const uint64_t* lo_in =
                        prefix + lookback + 1 - width + w_edge;
                    const uint64_t* hi_in = prefix + lookback + 1 - w_edge;
#pragma omp simd reduction(|| : hit)
                    for (SizeType t = 0; t < n; ++t) {
                        const uint64_t sum =
                            (phi[t] - lo[t]) + (hi_in[t] - lo_in[t]);
                        hit = hit || (sum >= th &&
                                      (!cold || start + t >= first_valid));
                    }
                }
                if (hit) {
                    return true;
                }
            }
        }
        return false;
    }

    // Exact run bookkeeping of one tile whose running sums are in
    // prefix[lookback + 1 .. lookback + n].
    template <typename T, typename Prefix>
    void score_tile(const plans::BoxcarPlanePlan& plane,
                    SizeType dm,
                    const BoxcarRow& row,
                    SizeType beam,
                    const Prefix* prefix,
                    SizeType lookback,
                    uint64_t start,
                    SizeType n,
                    BoxcarPeak* peaks,
                    SizeType row_idx) {
        const SizeType tid    = static_cast<SizeType>(omp_get_thread_num());
        float* best           = m_best.data() + tid * kTile;
        uint32_t* best_filter = m_best_filter.data() + tid * kTile;
        const auto emit_peak  = [&](const BoxcarPeak& peak) {
            emit(row_idx, beam, peak);
        };
        for (SizeType t = 0; t < n; ++t) {
            best[t]        = -1.0F;
            best_filter[t] = 0;
        }
        for (SizeType f = 0; f < m_filters; ++f) {
            const SizeType width  = plane.widths[f];
            const SizeType w_edge = m_config.use_trapezoid
                                        ? trapezoid_edge_width(plane.widths[f])
                                        : 0;
            const float threshold = plane.raw_thresh_lut[dm * m_filters + f];
            const float inv_sigma = plane.inv_sigma_lut[dm * m_filters + f];
            for (SizeType t = 0; t < n; ++t) {
                const uint64_t sample = start + t;
                const SizeType idx    = lookback + t + 1;
                Prefix sum            = prefix[idx] - prefix[idx - width];
                if (w_edge > 0) {
                    sum += prefix[idx - w_edge] - prefix[idx - width + w_edge];
                }
                const bool valid = !m_config.skip_cold_prefix ||
                                   sample >= row.warmup + width - 1;
                bool above;
                float snr = 0;
                if constexpr (std::is_integral_v<T>) {
                    above = valid &&
                            sum >= plane.integer_thresh_lut[dm * m_filters + f];
                    if (above &&
                        m_config.peak_mode ==
                            PeakFindingMode::kContiguousPeakAcrossWidths) {
                        snr = static_cast<float>(
                            (static_cast<double>(sum) -
                             plane.integer_mean_lut[dm * m_filters + f]) *
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
        }
        if (m_config.peak_mode ==
            PeakFindingMode::kContiguousPeakAcrossWidths) {
            for (SizeType t = 0; t < n; ++t) {
                boxcar_peak_step(peaks[0], best[t] >= 0, best[t], start + t,
                                 best_filter[t], emit_peak);
            }
        }
    }

    const plans::FDMTPlan& m_plan;
    BoxcarConfig m_config;
    const std::vector<plans::BoxcarPlanePlan>& m_planes;
    std::vector<BoxcarRow> m_rows;
    SizeType m_beams;
    int m_threads;
    SizeType m_width;
    SizeType m_filters;
    SizeType m_peak_count;
    std::unique_ptr<BoxcarDriver> m_driver;
    std::vector<SizeType> m_offsets;
    std::vector<float> m_carry;
    std::vector<BoxcarPeak> m_peaks;
    std::vector<uint64_t> m_per_dm;
    std::vector<RawCandidate> m_candidates;
    std::vector<double> m_prefix;
    std::vector<uint64_t> m_integer_prefix, m_integer_carry;
    std::vector<double> m_integer_means, m_integer_scales;
    double m_integer_unit = 1;
    std::vector<float> m_best;
    std::vector<uint32_t> m_best_filter;
    std::vector<BoxcarScreenGroup> m_groups;
    SizeType m_screen_levels = 0;
    std::vector<double> m_range_min;
    std::vector<double> m_group_thresh;
    CandidateCounts m_counts{};
    std::atomic<uint64_t> m_next_slot{0};
    uint64_t m_position       = 0;
    uint64_t m_result_begin   = 0;
    SizeType m_result_samples = 0;
};

} // namespace

std::unique_ptr<FDMTBoxcarEngine>
make_fdmt_boxcar_cpu(const plans::FDMTPlan& plan,
                     const BoxcarConfig& config,
                     const std::vector<plans::BoxcarPlanePlan>& planes,
                     const FDMTEngineConfig& cfg,
                     std::unique_ptr<BoxcarDriver> driver) {
    return std::make_unique<FDMTBoxcarCpuEngine>(plan, config, planes, cfg,
                                                 std::move(driver));
}

} // namespace dmt::algorithms::detail
