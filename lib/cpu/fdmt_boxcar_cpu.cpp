#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <omp.h>

#include "dmt/bit_pack_utils.hpp"
#include "dmt/fdmt_int_tree.hpp"

#include "fdmt_boxcar_cpu_engine.hpp"

// CPU FDMTBoxcar engine: construction, memory accounting, per-block kill-mask
// ring, streaming entrypoints, and consume() dispatch.

namespace dmt::algorithms::detail::fdmt_boxcar_cpu {

FDMTBoxcarCpuEngine::FDMTBoxcarCpuEngine(
    const plans::FDMTPlan& plan,
    const BoxcarConfig& config,
    const std::vector<plans::BoxcarPlanePlan>& planes,
    const FDMTEngineConfig& cfg,
    std::unique_ptr<BoxcarDriver> driver,
    std::vector<std::pair<SizeType, SizeType>> kill_runs)
    : m_plan(plan),
      m_config(config),
      m_planes(planes),
      m_rows(make_boxcar_rows(plan, planes)),
      m_beams(cfg.nbeams),
      m_threads(std::max(1, cfg.exec.nthreads)),
      m_width(planes.front().widths.back()),
      m_tile(config.screen_mode == BoxcarScreenMode::kKadane
                 ? kGateTile
                 : scoring_tile(m_width)),
      m_filters(planes.front().widths.size()),
      m_peak_count(config.peak_mode == PeakFindingMode::kContiguousPeakPerWidth
                       ? m_filters
                       : 1),
      m_driver(std::move(driver)),
      m_kill_runs(std::move(kill_runs)) {
    // Deferred: the buffers below are sized, not allocated, until
    // allocate().
    m_deferred.defer(cfg.defer_allocation);
    if (!m_kill_runs.empty()) {
        m_killed.assign(plan.get_nchans(), 0);
        for (const auto& [c0, c1] : m_kill_runs) {
            std::fill(m_killed.begin() + static_cast<std::ptrdiff_t>(c0),
                      m_killed.begin() + static_cast<std::ptrdiff_t>(c1), 1);
        }
        if (config.input_nbits == 32) {
            m_deferred.resize(m_masked,
                              m_beams * plan.get_nchans() * plan.get_nsamps());
        } else {
            m_deferred.resize(m_masked_packed,
                              m_beams * plan.get_nchans() *
                                  bit_pack_utils::packed_row_bytes(
                                      plan.get_nsamps(), config.input_nbits));
        }
    }
    if (m_rows.size() > SIZE_MAX / m_beams / m_width ||
        m_rows.size() > SIZE_MAX / m_beams / m_peak_count) {
        throw std::invalid_argument("FDMTBoxcar: history size overflow");
    }
    const SizeType rows = m_rows.size() * m_beams;
    m_integer_unit =
        cfg.use_box_smearing
            ? std::ldexp(1.0, -static_cast<int>(plan.get_box_fraction_bits()))
            : 1.0;
    // Packed rows fit uint32_t whenever every tree value does (as the
    // FDMT engines' own uint32_t state): half the carry's bytes.
    m_carry32 =
        config.input_nbits != 32 &&
        int_search_fits_u32(plan, cfg.use_box_smearing, config.input_nbits);
    if (config.input_nbits == 32) {
        m_deferred.resize(m_carry, rows * m_width);
    } else if (m_carry32) {
        m_deferred.resize(m_integer_carry32, rows * m_width);
    } else {
        m_deferred.resize(m_integer_carry, rows * m_width);
    }
    m_deferred.resize(m_peaks, rows * m_peak_count);
    m_deferred.resize(m_per_dm, rows);
    m_deferred.resize(m_candidates, config.max_candidates);
    m_level_offsets = boxcar_level_offsets(planes);
    m_plane_counts.resize(planes.size());
    m_dm_cap =
        config.max_cands_per_dm == 0 ? UINT64_MAX : config.max_cands_per_dm;
    m_plane_cap = config.max_candidates_per_plane;
    if (m_plane_cap != 0) {
        check_plane_share(config, planes.size());
        m_plane_next.resize(planes.size());
        if (config.overflow_policy != OverflowPolicy::kKeepBrightest) {
            m_deferred.resize(m_plane_scratch, planes.size() * m_plane_cap);
        }
    }
    if (config.overflow_policy == OverflowPolicy::kKeepBrightest) {
        // Each row's brightest max_cands_per_dm of the call (a heap).
        m_keep_q = config.max_cands_per_dm;
        if (m_keep_q > 0 && rows > SIZE_MAX / m_keep_q) {
            throw std::invalid_argument("FDMTBoxcar: keep-store size overflow");
        }
        m_deferred.resize(m_keep, rows * m_keep_q);
        m_deferred.resize(m_row_detected, rows);
    }
    m_kadane_exact_prefix_cap = std::min<SizeType>(
        SizeType{1} << 18,
        std::max(kKadaneExactPrefixCap, m_driver->chunk_nsamps()));
    const SizeType pref_tile =
        (config.screen_mode == BoxcarScreenMode::kKadane &&
         config.input_nbits == 32)
            ? m_kadane_exact_prefix_cap
            : m_tile;
    m_prefix_stride = m_width + pref_tile + 1;
    if (config.input_nbits == 32) {
        m_deferred.resize(m_prefix,
                          static_cast<SizeType>(m_threads) * m_prefix_stride);
    } else {
        m_deferred.resize(m_integer_prefix,
                          static_cast<SizeType>(m_threads) * m_prefix_stride);
    }
    m_deferred.resize(m_best, static_cast<SizeType>(m_threads) * m_tile);
    m_deferred.resize(m_best_filter, static_cast<SizeType>(m_threads) * m_tile);
    if (config.screen_mode == BoxcarScreenMode::kPrefix) {
        make_screen_groups(planes.front().widths);
    }
    m_offsets.push_back(0);
    m_row_plane.reserve(m_rows.size());
    for (SizeType ip = 0; ip < planes.size(); ++ip) {
        const auto& p = planes[ip];
        // Planes are ordered by level, so a level's rows are contiguous.
        if (m_level_rows.size() <= p.level) {
            m_level_rows.resize(p.level + 1, {0, 0});
        }
        auto& lr = m_level_rows[p.level];
        if (lr.first == lr.second) {
            lr.first = m_offsets.back();
        } else if (lr.second != m_offsets.back()) {
            throw std::logic_error("FDMTBoxcar (cpu): planes out of order");
        }
        lr.second = m_offsets.back() + p.dm_grid.size();
        m_row_plane.insert(m_row_plane.end(), p.dm_grid.size(),
                           static_cast<uint32_t>(ip));
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
    if (config.screen_mode == BoxcarScreenMode::kKadane) {
        init_kadane(cfg.use_box_smearing);
    }
    m_allocated = !cfg.defer_allocation;
}

void FDMTBoxcarCpuEngine::allocate() {
    if (m_allocated) {
        return;
    }
    const auto planned = memory_usage();
    m_driver->allocate();
    m_deferred.materialize();
    const auto got = memory_usage();
    if (got.total() != planned.total() || got.history != planned.history ||
        got.peaks != planned.peaks || got.workspace != planned.workspace ||
        got.candidates != planned.candidates ||
        got.staging != planned.staging) {
        throw std::logic_error(
            "FDMTBoxcar (cpu): allocation differs from the planned sizes");
    }
    m_allocated = true;
}

CandidateResultView
FDMTBoxcarCpuEngine::execute(std::span<const float> waterfall) {
    if (m_config.input_nbits != 32 ||
        waterfall.size() !=
            m_beams * m_plan.get_nchans() * m_plan.get_nsamps()) {
        throw std::invalid_argument("FDMTBoxcar: wrong waterfall size");
    }
    if (m_position > UINT64_MAX - m_plan.get_nsamps()) {
        throw std::overflow_error("FDMTBoxcar: stream sample index overflow");
    }
    m_noise_ring.commit(m_position, m_kill_runs);
    begin_result(m_plan.get_nsamps());
    m_driver->search(masked(waterfall, m_masked), *this);
    m_noise_ring.finish_commit();
    m_position += m_plan.get_nsamps();
    update_open_begin();
    finish_result();
    return view();
}

CandidateDeviceResultView
FDMTBoxcarCpuEngine::execute(DeviceSpan<const float> /*unused*/,
                             Stream /*unused*/) {
    throw_no_device_memory("FDMTBoxcar::execute", Backend::kCPU);
}

CandidateResultView
FDMTBoxcarCpuEngine::execute(std::span<const uint8_t> waterfall) {
    if (m_config.input_nbits == 32 ||
        waterfall.size() !=
            m_beams * m_plan.get_nchans() *
                bit_pack_utils::packed_row_bytes(m_plan.get_nsamps(),
                                                 m_config.input_nbits)) {
        throw std::invalid_argument(
            "FDMTBoxcar: packed input size or bit depth mismatch");
    }
    if (m_position > UINT64_MAX - m_plan.get_nsamps()) {
        throw std::overflow_error("FDMTBoxcar: stream sample index overflow");
    }
    m_noise_ring.commit(m_position, m_kill_runs);
    begin_result(m_plan.get_nsamps());
    m_driver->search(masked(waterfall, m_masked_packed), m_config.input_nbits,
                     *this);
    m_noise_ring.finish_commit();
    m_position += m_plan.get_nsamps();
    update_open_begin();
    finish_result();
    return view();
}

CandidateDeviceResultView
FDMTBoxcarCpuEngine::execute(DeviceSpan<const uint8_t> /*unused*/,
                             Stream /*unused*/) {
    throw_no_device_memory("FDMTBoxcar::execute", Backend::kCPU);
}

bool FDMTBoxcarCpuEngine::zero_blocks() const noexcept {
    return m_driver->search_zero_supported();
}

CandidateResultView FDMTBoxcarCpuEngine::execute_zero() {
    if (m_position > UINT64_MAX - m_plan.get_nsamps()) {
        throw std::overflow_error("FDMTBoxcar: stream sample index overflow");
    }
    // As execute() on a zero block: killed rows of zeros are zeros, so the
    // staging copy is skipped; the noise ring repeats the last tables.
    m_noise_ring.commit(m_position, m_kill_runs);
    begin_result(m_plan.get_nsamps());
    m_driver->search_zero(*this);
    m_noise_ring.finish_commit();
    m_position += m_plan.get_nsamps();
    update_open_begin();
    finish_result();
    return view();
}

CandidateDeviceResultView FDMTBoxcarCpuEngine::flush_device(Stream /*unused*/) {
    throw_no_device_memory("FDMTBoxcar::flush_device", Backend::kCPU);
}

CandidateResultView FDMTBoxcarCpuEngine::copy_result(Stream /*unused*/) {
    return view();
}

CandidateResultView FDMTBoxcarCpuEngine::flush() {
    begin_result(0);
    for (SizeType b = 0; b < m_beams; ++b) {
        for (SizeType row = 0; row < m_rows.size(); ++row) {
            const SizeType index = (b * m_rows.size()) + row;
            for (SizeType f = 0; f < m_peak_count; ++f) {
                auto& peak = m_peaks[(index * m_peak_count) + f];
                if (peak.open) {
                    emit(row, b, peak);
                    peak.open = false;
                }
            }
        }
    }
    std::ranges::fill(m_open, uint8_t{0});
    m_open_begin = UINT64_MAX;
    finish_result();
    return view();
}

void FDMTBoxcarCpuEngine::set_search_stop(uint64_t stop) noexcept {
    m_search_stop = stop;
}

void FDMTBoxcarCpuEngine::set_valid_nsamps(SizeType n_valid) noexcept {
    m_driver->set_valid_nsamps(n_valid);
}

uint64_t FDMTBoxcarCpuEngine::stream_position() const noexcept {
    return m_position;
}

uint64_t FDMTBoxcarCpuEngine::search_latency() const noexcept {
    return m_driver->search_latency();
}

uint64_t FDMTBoxcarCpuEngine::open_begin_min() const noexcept {
    return m_open_begin;
}

SizeType FDMTBoxcarCpuEngine::block_noise_bytes() const noexcept {
    // The staging block (when the constructor mask has none) and the
    // per-thread screen thresholds of tiles that mix tables.
    const bool packed      = m_config.input_nbits != 32;
    const SizeType row     = packed ? bit_pack_utils::packed_row_bytes(
                                          m_plan.get_nsamps(), m_config.input_nbits)
                                    : m_plan.get_nsamps() * sizeof(float);
    const SizeType staging = (packed ? m_deferred.size(m_masked_packed)
                                     : m_deferred.size(m_masked)) == 0
                                 ? m_beams * m_plan.get_nchans() * row
                                 : 0;
    // Packed: the fill row and the per-thread integer screen cutoffs.
    const SizeType fill = packed ? row + (static_cast<SizeType>(m_threads) *
                                          m_filters * sizeof(uint64_t))
                                 : 0;
    // Packed Kadane: the int32 gate's cuts, or the float gate's cutoffs.
    SizeType gate = 0;
    if (packed && m_config.screen_mode == BoxcarScreenMode::kKadane) {
        gate =
            m_kadane_int32
                ? static_cast<SizeType>(m_threads) * kadane::kMaxGateBuckets *
                      kadane::kMaxBatchRows * sizeof(int32_t)
                : static_cast<SizeType>(m_threads) * m_filters * sizeof(float);
    }
    const SizeType cuts =
        m_config.screen_mode == BoxcarScreenMode::kKadane
            ? (static_cast<SizeType>(m_threads) * kadane::kMaxGateBuckets *
               kadane::kMaxBatchRows * sizeof(float))
            : 0;
    return staging + fill + gate + cuts +
           (static_cast<SizeType>(m_threads) * m_filters * sizeof(float));
}

void FDMTBoxcarCpuEngine::enable_block_noise(
    std::shared_ptr<const std::vector<uint64_t>> support) {
    const bool packed = m_config.input_nbits != 32;
    // Sized, not allocated, while the construction is deferred
    // (BoxcarConfig::per_call_masks): allocate() makes them.
    if (packed) {
        const SizeType row = bit_pack_utils::packed_row_bytes(
            m_plan.get_nsamps(), m_config.input_nbits);
        const SizeType full = m_beams * m_plan.get_nchans() * row;
        if (m_deferred.size(m_masked_packed) < full) {
            m_deferred.resize(m_masked_packed, full);
        }
        // Channels a per-call mask kills read the code input_mean (an
        // integer code, checked by the noise model).
        m_fill_row.assign(row, uint8_t{0});
        const auto code = static_cast<uint32_t>(m_config.input_mean);
        const auto fill = [&]<unsigned NB>() {
            for (SizeType t = 0; t < m_plan.get_nsamps(); ++t) {
                bit_pack_utils::write_packed_sample<NB>(m_fill_row.data(), t,
                                                        code);
            }
        };
        switch (m_config.input_nbits) {
        case 1:
            fill.template operator()<1>();
            break;
        case 2:
            fill.template operator()<2>();
            break;
        case 4:
            fill.template operator()<4>();
            break;
        case 8:
            fill.template operator()<8>();
            break;
        default:
            fill.template operator()<16>();
            break;
        }
        m_deferred.resize(m_screen_ithr,
                          static_cast<SizeType>(m_threads) * m_filters);
        if (m_config.screen_mode == BoxcarScreenMode::kKadane) {
            if (m_kadane_int32) {
                m_deferred.resize(m_thread_icuts,
                                  static_cast<SizeType>(m_threads) *
                                      kadane::kMaxGateBuckets *
                                      kadane::kMaxBatchRows);
            } else {
                m_deferred.resize(m_screen_eff,
                                  static_cast<SizeType>(m_threads) * m_filters);
            }
        }
    } else {
        const SizeType full =
            m_beams * m_plan.get_nchans() * m_plan.get_nsamps();
        if (m_deferred.size(m_masked) < full) {
            m_deferred.resize(m_masked, full);
        }
    }
    m_killed.resize(m_plan.get_nchans());
    m_deferred.resize(m_screen_thr,
                      static_cast<SizeType>(m_threads) * m_filters);
    if (m_config.screen_mode == BoxcarScreenMode::kKadane) {
        m_deferred.resize(m_thread_cuts, static_cast<SizeType>(m_threads) *
                                             kadane::kMaxGateBuckets *
                                             kadane::kMaxBatchRows);
    }
    m_support         = std::move(support);
    const uint64_t ns = m_plan.get_nsamps();
    // Blocks shorter than the largest reach: at most the entries blocks of
    // that reach would need; older blocks share the catch-all.
    uint64_t reach = 0;
    for (const uint64_t s : *m_support) {
        reach = std::max(reach, s);
    }
    reach += m_width - 1;
    const auto blocks = [&](uint64_t n) {
        return 1 + static_cast<SizeType>((search_latency() + n - 1) / n);
    };
    const SizeType ring_needed = blocks(ns);
    const SizeType ring_size =
        std::min(ring_needed, blocks(std::max<uint64_t>(reach, 1)));
    m_noise_ring.enable(ring_needed, ring_size, m_kill_runs);
    // Packed per-width peaks rank by S/N from now on: give the open ones
    // theirs (the constructor's tables so far).
    if (packed &&
        m_config.peak_mode == PeakFindingMode::kContiguousPeakPerWidth) {
        const SizeType nrows = m_rows.size();
        for (SizeType idx = 0; idx < m_peaks.size(); ++idx) {
            auto& peak = m_peaks[idx];
            if (peak.open) {
                const SizeType i =
                    (((idx / m_peak_count) % nrows) * m_filters) + peak.filter;
                peak.snr = static_cast<float>(
                    (static_cast<double>(peak.raw_sum) - m_integer_means[i]) *
                    m_integer_scales[i]);
            }
        }
    }
}

void FDMTBoxcarCpuEngine::push_block_noise(BlockNoise entry) {
    m_noise_ring.push(std::move(entry));
}

void FDMTBoxcarCpuEngine::drop_block_noise() noexcept {
    (void)m_noise_ring.drop();
}

uint64_t FDMTBoxcarCpuEngine::noise_catch_all_begin() const noexcept {
    return m_noise_ring.catch_all_begin(m_position, m_plan.get_nsamps());
}

void FDMTBoxcarCpuEngine::reset_history(Stream /*unused*/) {
    if (m_noise_ring.enabled()) {
        m_noise_ring.reset(m_kill_runs);
    }
    m_search_stop = UINT64_MAX;
    m_driver->set_valid_nsamps(std::numeric_limits<SizeType>::max());
    m_driver->reset({});
    std::ranges::fill(m_carry, 0.0F);
    std::ranges::fill(m_integer_carry, uint64_t{0});
    std::ranges::fill(m_integer_carry32, uint32_t{0});
    std::ranges::fill(m_peaks, BoxcarPeak{});
    std::ranges::fill(m_kadane_fc, 0.0F);
    std::ranges::fill(m_kadane_ic, int32_t{0});
    for (auto& w : m_lag_f32) {
        w.reset();
    }
    for (auto& w : m_lag_i32) {
        w.reset();
    }
    std::ranges::fill(m_open, uint8_t{0});
    m_position   = 0;
    m_open_begin = UINT64_MAX;
    begin_result(0);
}

void FDMTBoxcarCpuEngine::consume(SizeType level,
                                  const FDMTSearchInput& input) {
    // Every row of every plane of the level in one loop. A level of many
    // small planes (subband search) is scheduled dynamically: rows differ
    // in cost (lags, cold prefixes, threshold hits) and static blocks
    // leave threads idle (parkes_uwl depth 4: 11 s against 15 s per
    // block). A single plane keeps static blocks, a few per cent faster
    // on its uniform rows (meerkat_lband: 516 against 539 ms).
    if (input.nsamps == 0 || level >= m_level_rows.size() ||
        m_level_rows[level].first == m_level_rows[level].second) {
        return;
    }
    const SizeType row_lo = m_level_rows[level].first;
    const SizeType row_hi = m_level_rows[level].second;
    const SizeType first  = input.beam_begin;
    const SizeType beams =
        std::min(input.beam_count, m_beams - input.beam_begin);
    if (beams == 0) {
        return;
    }
    if (m_config.screen_mode == BoxcarScreenMode::kKadane) {
        consume_kadane(row_lo, row_hi, first, beams, input);
        return;
    }
    const SizeType nrows = row_hi - row_lo;
    const auto task      = [&](SizeType t) {
        const SizeType beam    = first + (t / nrows);
        const SizeType row_idx = row_lo + (t % nrows);
        const SizeType p       = m_row_plane[row_idx];
        const SizeType d       = row_idx - m_offsets[p];
        FDMTSearchInput row_in = input;
        if (input.row_lag != nullptr &&
            !lagged_row_input(input, m_rows[row_idx].row, row_in)) {
            return;
        }
        row_in.search_stop = m_search_stop;
        if (!stop_row_input(row_in)) {
            return; // drain(): past the end of the stream
        }
        const FDMTSearchInput* in = &row_in;
        if (m_config.input_nbits == 32) {
            score<float, float>(m_planes[p], row_idx, d, beam, *in);
        } else if (m_noise_ring.enabled()) {
            if (m_carry32) {
                score<uint64_t, uint32_t, true>(m_planes[p], row_idx, d, beam,
                                                *in);
            } else {
                score<uint64_t, uint64_t, true>(m_planes[p], row_idx, d, beam,
                                                *in);
            }
        } else if (m_carry32) {
            score<uint64_t, uint32_t>(m_planes[p], row_idx, d, beam, *in);
        } else {
            score<uint64_t, uint64_t>(m_planes[p], row_idx, d, beam, *in);
        }
    };
    const SizeType ntasks = beams * nrows;
    const bool one_plane  = m_row_plane[row_lo] == m_row_plane[row_hi - 1];
    if (one_plane) {
#pragma omp parallel for num_threads(m_threads) schedule(static)
        for (SizeType t = 0; t < ntasks; ++t) {
            task(t);
        }
    } else {
#pragma omp parallel for num_threads(m_threads) schedule(dynamic, 4)
        for (SizeType t = 0; t < ntasks; ++t) {
            task(t);
        }
    }
}

FDMTBoxcarMemoryUsage FDMTBoxcarCpuEngine::memory_usage() const noexcept {
    const auto fdmt     = m_driver->memory();
    SizeType plan_bytes = (m_rows.size() * sizeof(BoxcarRow)) +
                          (m_offsets.size() * sizeof(SizeType)) +
                          (m_row_plane.size() * sizeof(uint32_t)) +
                          m_kadane_plan.memory_bytes() +
                          (m_row_int_m1.size() * sizeof(double));
    for (const auto& p : m_planes) {
        plan_bytes += sizeof(p) + (p.widths.size() * sizeof(uint32_t)) +
                      (p.dm_grid.size() * sizeof(float)) +
                      ((p.warmup_samples.size() + p.integer_thresh_lut.size()) *
                       sizeof(uint64_t)) +
                      (p.integer_mean_lut.size() * sizeof(double)) +
                      ((p.inv_sigma_lut.size() + p.raw_thresh_lut.size()) *
                       sizeof(float));
    }
    const SizeType kadane_hist =
        (m_deferred.size(m_kadane_fc) * sizeof(float)) +
        (m_deferred.size(m_kadane_ic) * sizeof(int32_t)) +
        (m_deferred.size(m_lag_f32) * sizeof(LagWindow<float>)) +
        (m_deferred.size(m_lag_i32) * sizeof(LagWindow<int32_t>)) +
        (m_deferred.size(m_open) * sizeof(uint8_t));
    const SizeType kadane_work =
        ((m_deferred.size(m_tile_f32) + m_deferred.size(m_thread_cuts)) *
         sizeof(float)) +
        (m_deferred.size(m_tile_u32) * sizeof(uint32_t));
    return {
        .plan      = fdmt.plan + plan_bytes,
        .state     = fdmt.state,
        .history   = fdmt.history + (m_deferred.size(m_carry) * sizeof(float)) +
                     (m_deferred.size(m_integer_carry) * sizeof(uint64_t)) +
                     (m_deferred.size(m_integer_carry32) * sizeof(uint32_t)) +
                     kadane_hist,
        .peaks     = m_deferred.size(m_peaks) * sizeof(BoxcarPeak),
        .workspace = fdmt.workspace +
                     ((m_deferred.size(m_prefix) + m_integer_means.size() +
                       m_integer_scales.size() + m_range_min.size() +
                       m_group_thresh.size()) *
                      sizeof(double)) +
                     (m_deferred.size(m_integer_prefix) * sizeof(uint64_t)) +
                     (m_deferred.size(m_best) * sizeof(float)) +
                     (m_deferred.size(m_best_filter) * sizeof(uint32_t)) +
                     (m_deferred.size(m_per_dm) * sizeof(uint64_t)) +
                     (m_deferred.size(m_row_detected) * sizeof(uint64_t)) +
                     (m_plane_counts.size() * sizeof(CandidateCounts)) +
                     (m_plane_next.size() * sizeof(uint64_t)) +
                     (m_level_offsets.size() * sizeof(uint32_t)) +
                     (m_deferred.size(m_screen_thr) * sizeof(float)) +
                     (m_deferred.size(m_screen_ithr) * sizeof(uint64_t)) +
                     (m_deferred.size(m_thread_icuts) * sizeof(int32_t)) +
                     (m_deferred.size(m_screen_eff) * sizeof(float)) +
                     kadane_work,
        .candidates = (m_deferred.size(m_candidates) + m_deferred.size(m_keep) +
                       m_deferred.size(m_plane_scratch)) *
                      sizeof(RawCandidate),
        .staging    = (m_deferred.size(m_masked) * sizeof(float)) +
                      m_deferred.size(m_masked_packed) + m_fill_row.size()};
}

SizeType FDMTBoxcarCpuEngine::chunk_nsamps() const noexcept {
    return m_driver->chunk_nsamps();
}

FDMTPath FDMTBoxcarCpuEngine::path() const noexcept { return m_driver->path(); }

std::string FDMTBoxcarCpuEngine::search_path() const {
    return m_driver->search_path();
}

SizeType FDMTBoxcarCpuEngine::fuse_levels() const noexcept {
    return m_driver->fuse_levels();
}

void FDMTBoxcarCpuEngine::begin_result(SizeType samples) {
    m_counts    = {};
    m_keep_open = m_keep_q != 0;
    m_next_slot.store(0, std::memory_order_relaxed);
    std::ranges::fill(m_per_dm, 0);
    std::ranges::fill(m_row_detected, 0);
    std::ranges::fill(m_plane_counts, CandidateCounts{});
    std::ranges::fill(m_plane_next, 0);
    m_result_begin   = m_position;
    m_result_samples = samples;
}

CandidateResultView FDMTBoxcarCpuEngine::view() const {
    return {
        m_counts,
        {m_candidates.data(), static_cast<SizeType>(m_counts.emitted_count)},
        m_result_begin,
        m_result_samples,
        m_plane_counts};
}

void FDMTBoxcarCpuEngine::emit(SizeType row,
                               SizeType beam,
                               const BoxcarPeak& peak) {
#pragma omp atomic
    ++m_counts.total_detected;
    auto& count = m_per_dm[(beam * m_rows.size()) + row];
    if (m_keep_q != 0) {
        ++m_row_detected[(beam * m_rows.size()) + row];
        keep((beam * m_rows.size()) + row, count,
             candidate_of(row, beam, peak));
        return;
    }
    // Every detection counts in its row (per-plane counts).
    if (count++ >= m_dm_cap) {
#pragma omp atomic
        ++m_counts.dropped_per_dm;
        return;
    }
    if (m_plane_cap != 0) {
        emit_to_plane(row, candidate_of(row, beam, peak));
        return;
    }
    const auto slot = m_next_slot.fetch_add(1, std::memory_order_relaxed);
    if (slot >= m_candidates.size()) {
#pragma omp atomic
        ++m_counts.dropped_global;
        return;
    }
    m_candidates[slot] = candidate_of(row, beam, peak);
#pragma omp atomic
    ++m_counts.emitted_count;
}

void FDMTBoxcarCpuEngine::emit_to_plane(SizeType row,
                                        const RawCandidate& candidate) {
    const SizeType p = m_rows[row].plane;
    uint64_t slot    = 0;
#pragma omp atomic capture
    slot = m_plane_next[p]++;
    if (slot >= m_plane_cap) {
#pragma omp atomic
        ++m_counts.dropped_global;
        return;
    }
    m_plane_scratch[(p * m_plane_cap) + slot] = candidate;
#pragma omp atomic
    ++m_counts.emitted_count;
}

RawCandidate FDMTBoxcarCpuEngine::candidate_of(SizeType row,
                                               SizeType beam,
                                               const BoxcarPeak& peak) const {
    auto candidate         = m_rows[row].candidate;
    candidate.beam_idx     = static_cast<uint32_t>(beam);
    candidate.peak_sample  = peak.peak;
    candidate.begin_sample = peak.begin;
    candidate.end_sample   = peak.end;
    candidate.snr          = peak.snr;
    // Packed peaks under per-call masks carry the S/N of their table.
    if (m_config.input_nbits != 32 && !m_noise_ring.enabled() &&
        m_config.peak_mode != PeakFindingMode::kContiguousPeakAcrossWidths) {
        const SizeType i = (row * m_filters) + peak.filter;
        candidate.snr    = static_cast<float>(
            (static_cast<double>(peak.raw_sum) - m_integer_means[i]) *
            m_integer_scales[i]);
    }
    candidate.filter_idx = peak.filter;
    candidate.width      = m_planes.front().widths[peak.filter];
    return candidate;
}

void FDMTBoxcarCpuEngine::keep(SizeType index,
                               uint64_t& count,
                               const RawCandidate& candidate) {
    auto* store       = m_keep.data() + (index * m_keep_q);
    const auto dimmer = [](const RawCandidate& a, const RawCandidate& b) {
        return brighter(a, b);
    };
    if (count < m_keep_q) {
        store[count++] = candidate;
        std::push_heap(store, store + count, dimmer);
        return;
    }
#pragma omp atomic
    ++m_counts.dropped_per_dm;
    if (brighter(candidate, store[0])) {
        std::pop_heap(store, store + m_keep_q, dimmer);
        store[m_keep_q - 1] = candidate;
        std::push_heap(store, store + m_keep_q, dimmer);
    }
}

void FDMTBoxcarCpuEngine::finish_result() {
    finish_keep_result();
    if (m_plane_cap != 0 && m_keep_q == 0) {
        // The planes' shares, in plane order, to the front.
        SizeType off = 0;
        for (SizeType p = 0; p < m_plane_next.size(); ++p) {
            const auto n = static_cast<SizeType>(
                std::min<uint64_t>(m_plane_next[p], m_plane_cap));
            const auto* from = m_plane_scratch.data() + (p * m_plane_cap);
            std::copy_n(from, n,
                        m_candidates.begin() +
                            static_cast<std::ptrdiff_t>(off));
            off += n;
        }
    }
    finish_plane_counts();
    // chan_dm plans: rows lag the bottom-edge arrival by the plan's level-0
    // causal delay; candidates report the arrival (saturating at 0).
    const uint64_t lag = m_plan.get_level0_causal_delay();
    if (lag == 0) {
        return;
    }
    const auto n = static_cast<SizeType>(
        std::min<uint64_t>(m_counts.emitted_count, m_candidates.size()));
    for (SizeType i = 0; i < n; ++i) {
        auto& c        = m_candidates[i];
        c.peak_sample  = c.peak_sample > lag ? c.peak_sample - lag : 0;
        c.begin_sample = c.begin_sample > lag ? c.begin_sample - lag : 0;
        c.end_sample   = c.end_sample > lag ? c.end_sample - lag : 0;
    }
}

void FDMTBoxcarCpuEngine::finish_plane_counts() {
    // Detections and per-row drops from the rows' counters (kDropNew:
    // m_per_dm counts every detection, drops past the cap; kKeepBrightest:
    // m_row_detected, with m_per_dm the kept ones), emitted from the call's
    // candidates.
    const auto plane_of = [&](const RawCandidate& c) -> CandidateCounts& {
        return m_plane_counts[m_level_offsets[c.subband_level] + c.subband_idx];
    };
    for (SizeType i = 0; i < m_per_dm.size(); ++i) {
        auto& out        = plane_of(m_rows[i % m_rows.size()].candidate);
        const uint64_t n = m_per_dm[i];
        if (m_keep_q != 0) {
            out.total_detected += m_row_detected[i];
            out.dropped_per_dm += m_row_detected[i] - n;
        } else {
            out.total_detected += n;
            out.dropped_per_dm += n > m_dm_cap ? n - m_dm_cap : 0;
        }
    }
    const auto emitted = static_cast<SizeType>(
        std::min<uint64_t>(m_counts.emitted_count, m_candidates.size()));
    for (SizeType j = 0; j < emitted; ++j) {
        ++plane_of(m_candidates[j]).emitted_count;
    }
    for (auto& p : m_plane_counts) {
        p.dropped_global =
            p.total_detected - p.dropped_per_dm - p.emitted_count;
    }
}

void FDMTBoxcarCpuEngine::finish_keep_result() {
    if (!m_keep_open) {
        return;
    }
    m_keep_open = false;
    // Compact the rows' entries to the front of the store (in place).
    SizeType n = 0;
    for (SizeType index = 0; index < m_per_dm.size(); ++index) {
        const auto* store = m_keep.data() + (index * m_keep_q);
        const auto count  = static_cast<SizeType>(m_per_dm[index]);
        if (n != index * m_keep_q) { // not already in place
            std::move(store, store + count,
                      m_keep.begin() + static_cast<std::ptrdiff_t>(n));
        }
        n += count;
    }
    const auto first = m_keep.begin();
    if (m_plane_cap == 0) {
        const SizeType kept = std::min(n, m_candidates.size());
        if (kept < n) {
            std::nth_element(first, first + static_cast<std::ptrdiff_t>(kept),
                             first + static_cast<std::ptrdiff_t>(n), brighter);
        }
        std::copy_n(first, kept, m_candidates.begin());
        m_counts.emitted_count  = kept;
        m_counts.dropped_global = n - kept;
        return;
    }
    // Each plane's brightest share: group the entries by plane, then
    // select within each group.
    const auto plane_of = [&](const RawCandidate& c) {
        return m_level_offsets[c.subband_level] + c.subband_idx;
    };
    std::sort(first, first + static_cast<std::ptrdiff_t>(n),
              [&](const RawCandidate& a, const RawCandidate& b) {
                  return plane_of(a) < plane_of(b);
              });
    SizeType kept = 0;
    for (SizeType b = 0; b < n;) {
        SizeType e = b + 1;
        while (e < n && plane_of(m_keep[e]) == plane_of(m_keep[b])) {
            ++e;
        }
        const SizeType take = std::min(e - b, m_plane_cap);
        const auto lo       = first + static_cast<std::ptrdiff_t>(b);
        if (take < e - b) {
            std::nth_element(lo, lo + static_cast<std::ptrdiff_t>(take),
                             first + static_cast<std::ptrdiff_t>(e), brighter);
        }
        std::copy_n(lo, take,
                    m_candidates.begin() + static_cast<std::ptrdiff_t>(kept));
        kept += take;
        b = e;
    }
    m_counts.emitted_count  = kept;
    m_counts.dropped_global = n - kept;
}

const float*
FDMTBoxcarCpuEngine::tile_tables(const plans::BoxcarPlanePlan& plane,
                                 SizeType dm,
                                 SizeType row_idx,
                                 uint64_t start,
                                 SizeType n,
                                 SizeType tid,
                                 TileTables& tables) {
    const uint64_t support = (*m_support)[row_idx];
    const uint64_t last    = start + n - 1;
    const NoiseTables* in_play[4];
    int count      = 0;
    const auto add = [&](const NoiseTables* t) {
        for (int j = 0; j < count; ++j) {
            if (in_play[j] == t) {
                return;
            }
        }
        in_play[count++] = t;
    };
    const auto ring   = m_noise_ring.entries();
    const SizeType i0 = m_noise_ring.index_for(start);
    const SizeType i1 = m_noise_ring.index_for(last);
    for (SizeType i = i0; i <= i1; ++i) {
        const auto& e     = ring[i];
        const uint64_t lo = i == i0 ? start : e.first_sample;
        const uint64_t hi = i + 1 <= i1 ? ring[i + 1].first_sample - 1 : last;
        if (!e.transition) {
            add(e.steady.get());
            continue;
        }
        if (lo < e.first_sample + support + m_width - 1) {
            add(e.transition_tables.get());
        }
        if (hi + 1 >= e.first_sample + support + m_planes.front().widths[0]) {
            add(e.steady.get());
        }
    }
    const float* plane_thr = plane.raw_thresh_lut.data() + (dm * m_filters);
    const float* plane_inv = plane.inv_sigma_lut.data() + (dm * m_filters);
    const bool packed      = m_config.input_nbits != 32;
    const uint64_t* plane_ithr =
        packed ? plane.integer_thresh_lut.data() + (dm * m_filters) : nullptr;
    if (count == 1) {
        tables.thr   = table_thr(in_play[0], plane_thr, row_idx, m_filters);
        tables.inv   = table_inv(in_play[0], plane_inv, row_idx, m_filters);
        tables.mixed = false;
        if (packed) {
            tables.ithr =
                table_ithr(in_play[0], plane_ithr, row_idx, m_filters);
            tables.screen_ithr = tables.ithr;
        }
        return tables.thr;
    }
    tables.thr     = plane_thr;
    tables.inv     = plane_inv;
    tables.mixed   = true;
    tables.support = support;
    if (packed) {
        tables.ithr       = plane_ithr;
        uint64_t* iscreen = m_screen_ithr.data() + (tid * m_filters);
        for (SizeType f = 0; f < m_filters; ++f) {
            uint64_t th = UINT64_MAX;
            for (int j = 0; j < count; ++j) {
                th = std::min(th, table_ithr(in_play[j], plane_ithr, row_idx,
                                             m_filters)[f]);
            }
            iscreen[f] = th;
        }
        tables.screen_ithr = iscreen;
    }
    float* screen = m_screen_thr.data() + (tid * m_filters);
    for (SizeType f = 0; f < m_filters; ++f) {
        float th = std::numeric_limits<float>::max();
        for (int j = 0; j < count; ++j) {
            th = std::min(
                th, table_thr(in_play[j], plane_thr, row_idx, m_filters)[f]);
        }
        screen[f] = th;
    }
    return screen;
}

FDMTBoxcarCpuEngine::SampleTable FDMTBoxcarCpuEngine::sample_table(
    uint64_t sample, SizeType width, uint64_t support) const noexcept {
    const auto ring  = m_noise_ring.entries();
    const SizeType i = m_noise_ring.index_for(sample);
    const auto& e    = ring[i];
    const uint64_t next =
        i + 1 < ring.size() ? ring[i + 1].first_sample : UINT64_MAX;
    if (e.transition) {
        const uint64_t end = e.first_sample + support + width - 1;
        if (sample < end) {
            return {e.transition_tables.get(), std::min(end, next)};
        }
    }
    return {e.steady.get(), next};
}

template <typename T>
std::span<const T> FDMTBoxcarCpuEngine::masked(std::span<const T> block,
                                               std::vector<T>& buffer) {
    // The block's own kill runs under per-call masks.
    const auto& runs =
        m_noise_ring.enabled() ? m_noise_ring.current_kill_runs() : m_kill_runs;
    if (runs.empty()) {
        return block;
    }
    if (m_noise_ring.enabled()) {
        // Packed: channels only the per-call mask kills read the mean code
        // (2); the constructor's read zero (1).
        const uint8_t call_killed = std::is_same_v<T, float> ? 1 : 2;
        std::ranges::fill(m_killed, uint8_t{0});
        for (const auto& [c0, c1] : runs) {
            std::fill(m_killed.begin() + static_cast<std::ptrdiff_t>(c0),
                      m_killed.begin() + static_cast<std::ptrdiff_t>(c1),
                      call_killed);
        }
        if (call_killed == 2) {
            for (const auto& [c0, c1] : m_kill_runs) {
                std::fill(m_killed.begin() + static_cast<std::ptrdiff_t>(c0),
                          m_killed.begin() + static_cast<std::ptrdiff_t>(c1),
                          uint8_t{1});
            }
        }
    }
    const SizeType nchans = m_plan.get_nchans();
    const SizeType rows   = m_beams * nchans;
    const SizeType row    = block.size() / rows;
    const T* src          = block.data();
    T* dst                = buffer.data();
#pragma omp parallel for num_threads(m_threads) schedule(static)
    for (SizeType r = 0; r < rows; ++r) {
        const uint8_t k = m_killed[r % nchans];
        if (k == 2) {
            if constexpr (std::is_same_v<T, uint8_t>) {
                std::copy_n(m_fill_row.data(), row, dst + (r * row));
            }
        } else if (k != 0) {
            std::fill_n(dst + (r * row), row, T{0});
        } else {
            std::copy_n(src + (r * row), row, dst + (r * row));
        }
    }
    return buffer;
}

void FDMTBoxcarCpuEngine::update_open_begin() {
    uint64_t oldest = UINT64_MAX;
    if (m_config.peak_mode != PeakFindingMode::kAllAboveThreshold) {
        const auto n = static_cast<std::ptrdiff_t>(m_peaks.size());
#pragma omp parallel for num_threads(m_threads) schedule(static)               \
    reduction(min : oldest)
        for (std::ptrdiff_t i = 0; i < n; ++i) {
            if (m_peaks[i].open) {
                oldest = std::min(oldest, m_peaks[i].begin);
            }
        }
    }
    m_open_begin = oldest;
}

} // namespace dmt::algorithms::detail::fdmt_boxcar_cpu

namespace dmt::algorithms::detail {

std::unique_ptr<FDMTBoxcarEngine>
make_fdmt_boxcar_cpu(const plans::FDMTPlan& plan,
                     const BoxcarConfig& config,
                     const std::vector<plans::BoxcarPlanePlan>& planes,
                     const FDMTEngineConfig& cfg,
                     std::unique_ptr<BoxcarDriver> driver,
                     std::vector<std::pair<SizeType, SizeType>> kill_runs) {
    return std::make_unique<fdmt_boxcar_cpu::FDMTBoxcarCpuEngine>(
        plan, resolve_screen_mode(config, false), planes, cfg,
        std::move(driver), std::move(kill_runs));
}

} // namespace dmt::algorithms::detail
