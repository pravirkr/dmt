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
      m_tile((config.screen_mode == BoxcarScreenMode::kKadane ||
              config.screen_mode == BoxcarScreenMode::kKadaneSurrogate)
                 ? kGateTile
                 : scoring_tile(m_width)),
      m_filters(planes.front().widths.size()),
      m_peak_count(config.peak_mode == PeakFindingMode::kContiguousPeakPerWidth
                       ? m_filters
                       : 1),
      m_driver(std::move(driver)),
      m_kill_runs(std::move(kill_runs)) {
    if (!m_kill_runs.empty()) {
        m_killed.assign(plan.get_nchans(), 0);
        for (const auto& [c0, c1] : m_kill_runs) {
            std::fill(m_killed.begin() + static_cast<std::ptrdiff_t>(c0),
                      m_killed.begin() + static_cast<std::ptrdiff_t>(c1), 1);
        }
        if (config.input_nbits == 32) {
            m_masked.resize(m_beams * plan.get_nchans() * plan.get_nsamps());
        } else {
            m_masked_packed.resize(m_beams * plan.get_nchans() *
                                   bit_pack_utils::packed_row_bytes(
                                       plan.get_nsamps(), config.input_nbits));
        }
    }
    const SizeType carry_ring =
        config.screen_mode == BoxcarScreenMode::kKadaneSurrogate
            ? (std::max<SizeType>(m_width, kGateTile + 4) + 1)
            : m_width;
    m_surrogate_carry_ring = carry_ring;
    if (m_rows.size() > SIZE_MAX / m_beams / carry_ring ||
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
    if (config.screen_mode == BoxcarScreenMode::kKadaneSurrogate ||
        config.input_nbits == 32) {
        m_carry.resize(rows * carry_ring);
    } else if (m_carry32) {
        m_integer_carry32.resize(rows * m_width);
    } else {
        m_integer_carry.resize(rows * m_width);
    }
    m_peaks.resize(rows * m_peak_count);
    m_per_dm.resize(rows);
    m_candidates.resize(config.max_candidates);
    if (config.overflow_policy == OverflowPolicy::kKeepBrightest) {
        // Each row's brightest max_cands_per_dm of the call (a heap).
        m_keep_q = config.max_cands_per_dm;
        if (m_keep_q > 0 && rows > SIZE_MAX / m_keep_q) {
            throw std::invalid_argument("FDMTBoxcar: keep-store size overflow");
        }
        m_keep.resize(rows * m_keep_q);
    }
    if (config.screen_mode != BoxcarScreenMode::kKadaneSurrogate) {
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
            m_prefix.resize(static_cast<SizeType>(m_threads) * m_prefix_stride);
        } else {
            m_integer_prefix.resize(static_cast<SizeType>(m_threads) *
                                    m_prefix_stride);
        }
        m_best.resize(static_cast<SizeType>(m_threads) * m_tile);
        m_best_filter.resize(static_cast<SizeType>(m_threads) * m_tile);
    }
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
    } else if (config.screen_mode == BoxcarScreenMode::kKadaneSurrogate) {
        init_surrogate(cfg.use_box_smearing);
    }
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
    commit_block_noise();
    begin_result(m_plan.get_nsamps());
    m_driver->search(masked(waterfall, m_masked), *this);
    m_commit_open = false;
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
    begin_result(m_plan.get_nsamps());
    m_driver->search(masked(waterfall, m_masked_packed), m_config.input_nbits,
                     *this);
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

CandidateDeviceResultView FDMTBoxcarCpuEngine::flush_device(Stream /*unused*/) {
    throw_no_device_memory("FDMTBoxcar::flush_device", Backend::kCPU);
}

CandidateResultView FDMTBoxcarCpuEngine::copy_result(Stream /*unused*/) {
    return view();
}

CandidateResultView FDMTBoxcarCpuEngine::flush() {
    begin_result(0);
    if (m_config.screen_mode == BoxcarScreenMode::kKadaneSurrogate) {
        flush_surrogate();
    }
    for (SizeType b = 0; b < m_beams; ++b) {
        for (SizeType row = 0; row < m_rows.size(); ++row) {
            const SizeType index = (b * m_rows.size()) + row;
            for (SizeType f = 0; f < m_peak_count; ++f) {
                auto& peak = m_peaks[(index * m_peak_count) + f];
                if (peak.open) {
                    if (m_config.screen_mode ==
                        BoxcarScreenMode::kKadaneSurrogate) {
                        emit_surrogate_peak(row, b, peak);
                    } else {
                        emit(row, b, peak);
                    }
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
    const SizeType staging = m_masked.empty()
                                 ? m_beams * m_plan.get_nchans() *
                                       m_plan.get_nsamps() * sizeof(float)
                                 : 0;
    const SizeType cuts =
        (m_config.screen_mode == BoxcarScreenMode::kKadane ||
         m_config.screen_mode == BoxcarScreenMode::kKadaneSurrogate)
            ? (static_cast<SizeType>(m_threads) * kadane::kMaxGateBuckets *
               kadane::kMaxBatchRows * sizeof(float))
            : 0;
    return staging + cuts +
           (static_cast<SizeType>(m_threads) * m_filters * sizeof(float));
}

void FDMTBoxcarCpuEngine::enable_block_noise(
    std::shared_ptr<const std::vector<uint64_t>> support) {
    if (m_config.input_nbits != 32) {
        throw std::logic_error(
            "FDMTBoxcar (cpu): per-call kill masks need float input");
    }
    m_masked.resize(m_beams * m_plan.get_nchans() * m_plan.get_nsamps());
    m_killed.resize(m_plan.get_nchans());
    m_screen_thr.resize(static_cast<SizeType>(m_threads) * m_filters);
    if (m_config.screen_mode == BoxcarScreenMode::kKadane ||
        m_config.screen_mode == BoxcarScreenMode::kKadaneSurrogate) {
        m_thread_cuts.resize(static_cast<SizeType>(m_threads) *
                             kadane::kMaxGateBuckets * kadane::kMaxBatchRows);
    }
    m_support         = std::move(support);
    const uint64_t ns = m_plan.get_nsamps();
    const SizeType extra_ring =
        m_config.screen_mode == BoxcarScreenMode::kKadaneSurrogate ? 1 : 0;
    m_ring_size = 1 + extra_ring +
                  static_cast<SizeType>((search_latency() + ns - 1) / ns);
    m_ring.reserve(m_ring_size + 1);
    m_noise_on = true;
    reset_block_noise();
    if (m_config.screen_mode == BoxcarScreenMode::kKadaneSurrogate) {
        const SizeType np = m_surrogate_plan.num_passes;
        for (SizeType idx = 0; idx < m_surrogate_rows.size(); ++idx) {
            const SizeType b   = idx / m_rows.size();
            const SizeType r   = idx % m_rows.size();
            const uint64_t pos = m_surrogate_pos[idx];
            if (pos > 0 && m_surrogate_plan.disabled_row[r] == 0U) {
                sync_surrogate_row_at_pos(0, r, b, pos);
                for (SizeType k = 0; k < np; ++k) {
                    cache_surrogate_pass_snr(r, m_surrogate_rows[idx].passes[k],
                                             static_cast<uint32_t>(k), nullptr,
                                             0, 0, nullptr);
                }
            }
        }
    }
}

void FDMTBoxcarCpuEngine::push_block_noise(BlockNoise entry) {
    m_pending = std::move(entry);
}

// m_ring is pre-reserved to m_ring_size + 1 in enable_block_noise().
// NOLINTNEXTLINE(bugprone-exception-escape)
void FDMTBoxcarCpuEngine::drop_block_noise() noexcept {
    m_pending.reset();
    // A call that threw after its commit: restore the ring.
    if (m_commit_open) {
        m_ring.pop_back();
        if (m_evicted) {
            m_ring.insert(m_ring.begin(), std::move(*m_evicted));
            m_evicted.reset();
        }
        m_commit_open = false;
    }
}

void FDMTBoxcarCpuEngine::reset_history(Stream /*unused*/) {
    if (m_noise_on) {
        reset_block_noise();
    }
    m_search_stop = UINT64_MAX;
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
    std::ranges::fill(m_surrogate_c, 0.0F);
    std::ranges::fill(m_surrogate_dec, 0.0F);
    for (auto& st : m_surrogate_rows) {
        st.reset();
    }
    std::ranges::fill(m_surrogate_open, uint8_t{0});
    std::ranges::fill(m_surrogate_pos, uint64_t{0});
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
    if (m_config.screen_mode == BoxcarScreenMode::kKadaneSurrogate) {
        consume_surrogate(row_lo, row_hi, first, beams, input);
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
                          m_surrogate_plan.memory_bytes() +
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
        ((m_kadane_fc.size() + m_surrogate_c.size() + m_surrogate_dec.size()) *
         sizeof(float)) +
        (m_kadane_ic.size() * sizeof(int32_t)) +
        (m_lag_f32.size() * sizeof(LagWindow<float>)) +
        (m_lag_i32.size() * sizeof(LagWindow<int32_t>)) +
        ((m_open.size() + m_surrogate_open.size()) * sizeof(uint8_t)) +
        (m_surrogate_rows.size() * sizeof(kadane::KadaneSurrogateRowState)) +
        (m_surrogate_pos.size() * sizeof(uint64_t));
    const SizeType kadane_work =
        ((m_tile_f32.size() + m_thread_cuts.size() + m_surrogate_chk_c.size() +
          m_surrogate_chk_dec.size() + m_surrogate_lb.size()) *
         sizeof(float)) +
        (m_tile_u32.size() * sizeof(uint32_t));
    return {
        .plan    = fdmt.plan + plan_bytes,
        .state   = fdmt.state,
        .history = fdmt.history + (m_carry.size() * sizeof(float)) +
                   (m_integer_carry.size() * sizeof(uint64_t)) +
                   (m_integer_carry32.size() * sizeof(uint32_t)) + kadane_hist,
        .peaks   = m_peaks.size() * sizeof(BoxcarPeak),
        .workspace = fdmt.workspace +
                     ((m_prefix.size() + m_integer_means.size() +
                       m_integer_scales.size() + m_range_min.size() +
                       m_group_thresh.size()) *
                      sizeof(double)) +
                     (m_integer_prefix.size() * sizeof(uint64_t)) +
                     (m_best.size() * sizeof(float)) +
                     (m_best_filter.size() * sizeof(uint32_t)) +
                     (m_per_dm.size() * sizeof(uint64_t)) +
                     (m_screen_thr.size() * sizeof(float)) + kadane_work,
        .candidates =
            (m_candidates.size() + m_keep.size()) * sizeof(RawCandidate),
        .staging = (m_masked.size() * sizeof(float)) + m_masked_packed.size()};
}

SizeType FDMTBoxcarCpuEngine::chunk_nsamps() const noexcept {
    return m_driver->chunk_nsamps();
}

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
    m_result_begin   = m_position;
    m_result_samples = samples;
}

CandidateResultView FDMTBoxcarCpuEngine::view() const {
    return {
        m_counts,
        {m_candidates.data(), static_cast<SizeType>(m_counts.emitted_count)},
        m_result_begin,
        m_result_samples};
}

void FDMTBoxcarCpuEngine::emit(SizeType row,
                               SizeType beam,
                               const BoxcarPeak& peak) {
#pragma omp atomic
    ++m_counts.total_detected;
    auto& count = m_per_dm[(beam * m_rows.size()) + row];
    if (m_keep_q != 0) {
        keep((beam * m_rows.size()) + row, count,
             candidate_of(row, beam, peak));
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
    m_candidates[slot] = candidate_of(row, beam, peak);
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
    if (m_config.input_nbits != 32 &&
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
    const SizeType kept = std::min(n, m_candidates.size());
    if (kept < n) {
        std::nth_element(
            m_keep.begin(), m_keep.begin() + static_cast<std::ptrdiff_t>(kept),
            m_keep.begin() + static_cast<std::ptrdiff_t>(n), brighter);
    }
    std::copy_n(m_keep.begin(), kept, m_candidates.begin());
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
    const SizeType i0 = ring_index(start);
    const SizeType i1 = ring_index(last);
    for (SizeType i = i0; i <= i1; ++i) {
        const auto& e     = m_ring[i];
        const uint64_t lo = i == i0 ? start : e.first_sample;
        const uint64_t hi = i + 1 <= i1 ? m_ring[i + 1].first_sample - 1 : last;
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
    if (count == 1) {
        tables.thr   = table_thr(in_play[0], plane_thr, row_idx, m_filters);
        tables.inv   = table_inv(in_play[0], plane_inv, row_idx, m_filters);
        tables.mixed = false;
        return tables.thr;
    }
    tables.thr     = plane_thr;
    tables.inv     = plane_inv;
    tables.mixed   = true;
    tables.support = support;
    float* screen  = m_screen_thr.data() + (tid * m_filters);
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
    const SizeType i = ring_index(sample);
    const auto& e    = m_ring[i];
    const uint64_t next =
        i + 1 < m_ring.size() ? m_ring[i + 1].first_sample : UINT64_MAX;
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
    const auto& runs = m_noise_on ? m_ring.back().kill_runs : m_kill_runs;
    if (runs.empty()) {
        return block;
    }
    if (m_noise_on) {
        std::ranges::fill(m_killed, uint8_t{0});
        for (const auto& [c0, c1] : runs) {
            std::fill(m_killed.begin() + static_cast<std::ptrdiff_t>(c0),
                      m_killed.begin() + static_cast<std::ptrdiff_t>(c1),
                      uint8_t{1});
        }
    }
    const SizeType nchans = m_plan.get_nchans();
    const SizeType rows   = m_beams * nchans;
    const SizeType row    = block.size() / rows;
    const T* src          = block.data();
    T* dst                = buffer.data();
#pragma omp parallel for num_threads(m_threads) schedule(static)
    for (SizeType r = 0; r < rows; ++r) {
        if (m_killed[r % nchans] != 0) {
            std::fill_n(dst + (r * row), row, T{0});
        } else {
            std::copy_n(src + (r * row), row, dst + (r * row));
        }
    }
    return buffer;
}

void FDMTBoxcarCpuEngine::update_open_begin() {
    if (m_config.screen_mode == BoxcarScreenMode::kKadaneSurrogate) {
        update_open_begin_surrogate();
        return;
    }
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

void FDMTBoxcarCpuEngine::reset_block_noise() {
    m_pending.reset();
    m_evicted.reset();
    m_commit_open = false;
    m_ring.clear();
    BlockNoise first; // the constructor tables from stream sample 0
    first.kill_runs = m_kill_runs;
    m_ring.push_back(std::move(first));
}

void FDMTBoxcarCpuEngine::commit_block_noise() {
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
    m_evicted.reset();
    if (m_ring.size() == m_ring_size) {
        m_evicted = std::move(m_ring.front());
        m_ring.erase(m_ring.begin());
    }
    m_ring.push_back(std::move(entry));
    m_commit_open = true;
}

SizeType FDMTBoxcarCpuEngine::ring_index(uint64_t sample) const noexcept {
    SizeType i = m_ring.size() - 1;
    while (i > 0 && m_ring[i].first_sample > sample) {
        --i;
    }
    return i;
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
        plan, config, planes, cfg, std::move(driver), std::move(kill_runs));
}

} // namespace dmt::algorithms::detail
