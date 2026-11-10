#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <omp.h>

#include "dmt/bit_pack_utils.hpp"
#include "dmt/boxcar.hpp"
#include "dmt/fdmt_int_tree.hpp"
#include "dmt/fdmt_variance.hpp"

namespace dmt::algorithms::detail {
namespace {

/// Scoring tile: each tile rebuilds its running sums and screen minima over
/// the max_width - 1 lookback samples before it, so wide boxcars take longer
/// tiles (about 4x the lookback, 1024 to 4096 samples) to amortize that.
SizeType scoring_tile(SizeType max_width) noexcept {
    SizeType tile = 1024;
    while (tile < 4096 && tile < 4 * max_width) {
        tile *= 2;
    }
    return tile;
}

class FDMTBoxcarCpuEngine final : public FDMTBoxcarEngine {
public:
    FDMTBoxcarCpuEngine(const plans::FDMTPlan& plan,
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
          m_tile(scoring_tile(m_width)),
          m_filters(planes.front().widths.size()),
          m_peak_count(config.peak_mode ==
                               PeakFindingMode::kContiguousPeakPerWidth
                           ? m_filters
                           : 1),
          m_driver(std::move(driver)),
          m_kill_runs(std::move(kill_runs)) {
        if (!m_kill_runs.empty()) {
            m_killed.assign(plan.get_nchans(), 0);
            for (const auto& [c0, c1] : m_kill_runs) {
                std::fill(m_killed.begin() + static_cast<std::ptrdiff_t>(c0),
                          m_killed.begin() + static_cast<std::ptrdiff_t>(c1),
                          1);
            }
            if (config.input_nbits == 32) {
                m_masked.resize(m_beams * plan.get_nchans() *
                                plan.get_nsamps());
            } else {
                m_masked_packed.resize(
                    m_beams * plan.get_nchans() *
                    bit_pack_utils::packed_row_bytes(plan.get_nsamps(),
                                                     config.input_nbits));
            }
        }
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
        // Packed rows fit uint32_t whenever every tree value does (as the
        // FDMT engines' own uint32_t state): half the carry's bytes.
        m_carry32 =
            config.input_nbits != 32 &&
            int_search_fits_u32(plan, cfg.use_box_smearing, config.input_nbits);
        if (config.input_nbits == 32) {
            m_carry.resize(rows * m_width);
        } else if (m_carry32) {
            m_integer_carry32.resize(rows * m_width);
        } else {
            m_integer_carry.resize(rows * m_width);
        }
        m_peaks.resize(rows * m_peak_count);
        m_per_dm.resize(rows);
        m_candidates.resize(config.max_candidates);
        if (config.input_nbits == 32) {
            m_prefix.resize(m_threads * (m_width + m_tile + 1));
        } else {
            m_integer_prefix.resize(m_threads * (m_width + m_tile + 1));
        }
        m_best.resize(m_threads * m_tile);
        m_best_filter.resize(m_threads * m_tile);
        make_screen_groups(planes.front().widths);
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
        commit_block_noise();
        begin_result(m_plan.get_nsamps());
        m_driver->search(masked(waterfall, m_masked), *this);
        m_position += m_plan.get_nsamps();
        update_open_begin();
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
        m_driver->search(masked(waterfall, m_masked_packed),
                         m_config.input_nbits, *this);
        m_position += m_plan.get_nsamps();
        update_open_begin();
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
        m_open_begin = UINT64_MAX;
        return view();
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
        return m_open_begin;
    }
    [[nodiscard]] SizeType block_noise_bytes() const noexcept override {
        // The staging block (when the constructor mask has none) and the
        // per-thread screen thresholds of tiles that mix tables.
        const SizeType staging = m_masked.empty()
                                     ? m_beams * m_plan.get_nchans() *
                                           m_plan.get_nsamps() * sizeof(float)
                                     : 0;
        return staging +
               (static_cast<SizeType>(m_threads) * m_filters * sizeof(float));
    }
    void enable_block_noise(
        std::shared_ptr<const std::vector<uint64_t>> support) override {
        if (m_config.input_nbits != 32) {
            throw std::logic_error(
                "FDMTBoxcar (cpu): per-call kill masks need float input");
        }
        m_masked.resize(m_beams * m_plan.get_nchans() * m_plan.get_nsamps());
        m_killed.resize(m_plan.get_nchans());
        m_screen_thr.resize(static_cast<SizeType>(m_threads) * m_filters);
        m_support         = std::move(support);
        const uint64_t ns = m_plan.get_nsamps();
        m_ring_size =
            1 + static_cast<SizeType>((search_latency() + ns - 1) / ns);
        m_ring.reserve(m_ring_size + 1);
        m_noise_on = true;
        reset_block_noise();
    }
    void push_block_noise(BlockNoise entry) override {
        m_pending = std::move(entry);
    }
    void drop_block_noise() noexcept override { m_pending.reset(); }
    void reset_history(Stream) override {
        if (m_noise_on) {
            reset_block_noise();
        }
        m_search_stop = UINT64_MAX;
        m_driver->reset({});
        std::ranges::fill(m_carry, 0.0F);
        std::ranges::fill(m_integer_carry, uint64_t{0});
        std::ranges::fill(m_integer_carry32, uint32_t{0});
        std::ranges::fill(m_peaks, BoxcarPeak{});
        m_position   = 0;
        m_open_begin = UINT64_MAX;
        begin_result(0);
    }

    void consume(SizeType level, const FDMTSearchInput& input) override {
        // Every row of every plane of the level in one loop. A level of many
        // small planes (subband search) is scheduled dynamically: rows differ
        // in cost (lags, cold prefixes, threshold hits) and static blocks
        // leave threads idle (parkes_uwl depth 4: 11 s against 15 s per
        // block). A single plane keeps static blocks, a few per cent faster
        // on its uniform rows (meerkat_lband: 516 against 539 ms).
        if (level >= m_level_rows.size() ||
            m_level_rows[level].first == m_level_rows[level].second) {
            return;
        }
        const SizeType row_lo = m_level_rows[level].first;
        const SizeType row_hi = m_level_rows[level].second;
        const SizeType nrows  = row_hi - row_lo;
        const SizeType first  = input.beam_begin;
        const SizeType beams =
            std::min(input.beam_count, m_beams - input.beam_begin);
        const auto task = [&](SizeType t) {
            const SizeType beam    = first + t / nrows;
            const SizeType row_idx = row_lo + t % nrows;
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

    FDMTBoxcarMemoryUsage memory_usage() const noexcept override {
        const auto fdmt     = m_driver->memory();
        SizeType plan_bytes = m_rows.size() * sizeof(BoxcarRow) +
                              m_offsets.size() * sizeof(SizeType) +
                              m_row_plane.size() * sizeof(uint32_t);
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
                              m_integer_carry.size() * sizeof(uint64_t) +
                              m_integer_carry32.size() * sizeof(uint32_t),
                .peaks      = m_peaks.size() * sizeof(BoxcarPeak),
                .workspace  = fdmt.workspace +
                              (m_prefix.size() + m_integer_means.size() +
                               m_integer_scales.size() + m_range_min.size() +
                               m_group_thresh.size()) *
                                  sizeof(double) +
                              m_integer_prefix.size() * sizeof(uint64_t) +
                              m_best.size() * sizeof(float) +
                              m_best_filter.size() * sizeof(uint32_t) +
                              m_per_dm.size() * sizeof(uint64_t) +
                              m_screen_thr.size() * sizeof(float),
                .candidates = m_candidates.size() * sizeof(RawCandidate),
                .staging =
                    (m_masked.size() * sizeof(float)) + m_masked_packed.size()};
    }
    SizeType chunk_nsamps() const noexcept override {
        return m_driver->chunk_nsamps();
    }
    std::string search_path() const override { return m_driver->search_path(); }
    SizeType fuse_levels() const noexcept override {
        return m_driver->fuse_levels();
    }

private:
    void make_screen_groups(const std::vector<uint32_t>& widths) {
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
        m_range_min.resize(m_threads * screen_slots() * (m_width + m_tile + 1));
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
    /// Thresholds and inverse sigmas of a float row's tile: one table
    /// (`thr`, `inv`: the row's entries), or, when `mixed`, a table per
    /// sample and width chosen from the block ring (support: the row's
    /// support with every channel active).
    struct TileTables {
        const float* thr = nullptr;
        const float* inv = nullptr;
        bool mixed       = false;
        uint64_t support = 0;
    };
    [[nodiscard]] static const float* table_thr(const NoiseTables* t,
                                                const float* plane_row,
                                                SizeType row,
                                                SizeType filters) noexcept {
        return t == nullptr ? plane_row : t->thresh.data() + (row * filters);
    }
    [[nodiscard]] static const float* table_inv(const NoiseTables* t,
                                                const float* plane_row,
                                                SizeType row,
                                                SizeType filters) noexcept {
        return t == nullptr ? plane_row : t->inv_sigma.data() + (row * filters);
    }
    /**
     * Tables of the tile [start, start + n) of a float row under per-call
     * masks: the steady tables of every block it touches, and a block's
     * transition tables where it touches the first S + max_width - 1
     * endpoints of a block with a transition. One table: `tables` points at
     * it. Several: `tables` is mixed, and the screen thresholds returned
     * are the per-width minimum over them (a lower bound for every sample,
     * so the screen stays exact).
     */
    const float* tile_tables(const plans::BoxcarPlanePlan& plane,
                             SizeType dm,
                             SizeType row_idx,
                             uint64_t start,
                             SizeType n,
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
            const uint64_t hi =
                i + 1 <= i1 ? m_ring[i + 1].first_sample - 1 : last;
            if (!e.transition) {
                add(e.steady.get());
                continue;
            }
            if (lo < e.first_sample + support + m_width - 1) {
                add(e.transition_tables.get());
            }
            if (hi + 1 >=
                e.first_sample + support + m_planes.front().widths[0]) {
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
        const auto tid = static_cast<SizeType>(omp_get_thread_num());
        float* screen  = m_screen_thr.data() + (tid * m_filters);
        for (SizeType f = 0; f < m_filters; ++f) {
            float th = std::numeric_limits<float>::max();
            for (int j = 0; j < count; ++j) {
                th = std::min(th, table_thr(in_play[j], plane_thr, row_idx,
                                            m_filters)[f]);
            }
            screen[f] = th;
        }
        return screen;
    }
    /// Under mixed tables: the table of endpoint `sample` for width
    /// `width` and the first sample after it that may use another.
    struct SampleTable {
        const NoiseTables* table;
        uint64_t until;
    };
    [[nodiscard]] SampleTable sample_table(uint64_t sample,
                                           SizeType width,
                                           uint64_t support) const noexcept {
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
    /// Running sums out[i + 1] = out[i] + sample(t0 + i), i < n, of the
    /// same samples as boxcar_sample() / boxcar_integer_sample(), with the
    /// row's pointers resolved once and the loop split at row.delay (ring
    /// history before, plain head-row walk after) instead of branching per
    /// sample. Same additions in the same order: exact.
    template <typename T, typename Prefix>
    static void prefix_tile(const FDMTSearchInput& input,
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
            const SizeType split =
                row.delay > t0 ? std::min(n, row.delay - t0) : 0;
            const auto* hist = split > 0
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
                out[i + 1] = out[i] + static_cast<Prefix>(
                                          value(tail[i]) +
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
    void score(const plans::BoxcarPlanePlan& plane,
               SizeType row_idx,
               SizeType dm,
               SizeType beam,
               const FDMTSearchInput& input) {
        const SizeType tid = static_cast<SizeType>(omp_get_thread_num());
        using Prefix =
            std::conditional_t<std::is_integral_v<T>, uint64_t, double>;
        Prefix* prefix;
        C* carry;
        const SizeType index = beam * m_rows.size() + row_idx;
        if constexpr (std::is_same_v<C, uint32_t>) {
            prefix = m_integer_prefix.data() + tid * (m_width + m_tile + 1);
            carry  = m_integer_carry32.data() + index * m_width;
        } else if constexpr (std::is_integral_v<T>) {
            prefix = m_integer_prefix.data() + tid * (m_width + m_tile + 1);
            carry  = m_integer_carry.data() + index * m_width;
        } else {
            prefix = m_prefix.data() + tid * (m_width + m_tile + 1);
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
                SizeType slot = static_cast<SizeType>(
                    (first - (lookback - cold)) % m_width);
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
            group_thresh = m_group_thresh.data() + tid * m_groups.size();
            range_min    = m_range_min.data() +
                           tid * screen_slots() * (m_width + m_tile + 1);
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
                }
            }
        }
        for (SizeType t0 = 0; t0 < input.nsamps; t0 += m_tile) {
            const SizeType n     = std::min(m_tile, input.nsamps - t0);
            const uint64_t start = input.sample_begin + t0;
            prefix_tile<T>(input, row, beam, root, ring0, t0, n,
                           prefix + lookback);
            // Rare-event check: any width over its threshold in this tile.
            // Without it (and no open run) every peak mode is unchanged.
            const bool cold =
                m_config.skip_cold_prefix && start + 1 < row.warmup + m_width;
            if constexpr (!std::is_integral_v<T>) {
                if (m_noise_on) {
                    screen_thr =
                        tile_tables(plane, dm, row_idx, start, n, tables);
                }
            }
            bool hit      = false;
            bool screened = false;
            if (!m_groups.empty() && !open) {
                if constexpr (std::is_integral_v<T>) {
                    hit =
                        screen_tile_int(plane, dm, row, prefix, lookback, start,
                                        n, cold, group_thresh, range_min);
                } else {
                    set_groups(screen_thr);
                    hit = screen_tile(plane, screen_thr, row, prefix, lookback,
                                      start, n, cold, group_thresh, range_min);
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
            SizeType slot =
                static_cast<SizeType>((input.sample_begin + t_first) % m_width);
            for (SizeType t = t_first; t < input.nsamps; ++t) {
                carry[slot] = static_cast<C>(sample_at(t));
                slot        = slot + 1 == m_width ? 0 : slot + 1;
            }
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
                    SizeType row_idx,
                    const TileTables& tables) {
        const SizeType tid    = static_cast<SizeType>(omp_get_thread_num());
        float* best           = m_best.data() + tid * m_tile;
        uint32_t* best_filter = m_best_filter.data() + tid * m_tile;
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
                    t_end = until - start < n
                                ? static_cast<SizeType>(until - start)
                                : n;
                    threshold =
                        table_thr(table, tables.thr, row_idx, m_filters)[f];
                    inv_sigma =
                        table_inv(table, tables.inv, row_idx, m_filters)[f];
                }
                for (SizeType t = t_run; t < t_end; ++t) {
                    const uint64_t sample = start + t;
                    const SizeType idx    = lookback + t + 1;
                    Prefix sum            = prefix[idx] - prefix[idx - width];
                    if (w_edge > 0) {
                        sum +=
                            prefix[idx - w_edge] - prefix[idx - width + w_edge];
                    }
                    const bool valid = !m_config.skip_cold_prefix ||
                                       sample >= row.warmup + width - 1;
                    bool above;
                    float snr = 0;
                    if constexpr (std::is_integral_v<T>) {
                        above =
                            valid &&
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
                        snr =
                            above ? static_cast<float>(sum * inv_sigma) : 0.0F;
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
                            boxcar_integer_peak_step(
                                peaks[f], above, sum, sample,
                                static_cast<uint32_t>(f), emit_peak);
                        } else {
                            boxcar_peak_step(peaks[f], above, snr, sample,
                                             static_cast<uint32_t>(f),
                                             emit_peak);
                        }
                    }
                }
                t_run = t_end;
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

    /// The block with the killed channels' rows zeroed (in `buffer`), or
    /// the block itself without a kill mask. Rows are copied in parallel:
    /// the copy is bound by memory bandwidth.
    template <typename T>
    std::span<const T> masked(std::span<const T> block,
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

    /// The oldest begin sample of the open excursions (candidate
    /// frontier); no excursion stays open in kAllAboveThreshold.
    void update_open_begin() {
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
    /// The ring back to one constructor entry (stream sample 0).
    void reset_block_noise() {
        m_pending.reset();
        m_ring.clear();
        BlockNoise first; // the constructor tables from stream sample 0
        first.kill_runs = m_kill_runs;
        m_ring.push_back(std::move(first));
    }
    /// Commits the pushed entry of the block about to run (or, without one,
    /// a repeat of the last block's steady tables) and drops entries no
    /// endpoint can reach any more.
    void commit_block_noise() {
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
        if (m_ring.size() == m_ring_size) {
            m_ring.erase(m_ring.begin());
        }
        m_ring.push_back(std::move(entry));
    }
    /// Index of the ring entry of stream sample `sample` (the oldest entry
    /// for samples before it).
    [[nodiscard]] SizeType ring_index(uint64_t sample) const noexcept {
        SizeType i = m_ring.size() - 1;
        while (i > 0 && m_ring[i].first_sample > sample) {
            --i;
        }
        return i;
    }

    const plans::FDMTPlan& m_plan;
    BoxcarConfig m_config;
    const std::vector<plans::BoxcarPlanePlan>& m_planes;
    std::vector<BoxcarRow> m_rows;
    SizeType m_beams;
    int m_threads;
    SizeType m_width;
    SizeType m_tile;
    SizeType m_filters;
    SizeType m_peak_count;
    std::unique_ptr<BoxcarDriver> m_driver;
    /// Killed channel runs and the staging block they are zeroed in.
    std::vector<std::pair<SizeType, SizeType>> m_kill_runs;
    std::vector<uint8_t> m_killed; ///< per channel, 1 = killed
    std::vector<float> m_masked;
    std::vector<uint8_t> m_masked_packed;
    std::vector<SizeType> m_offsets;
    /// Row range [first, second) of each level's planes, and each row's plane.
    std::vector<std::pair<SizeType, SizeType>> m_level_rows;
    std::vector<uint32_t> m_row_plane;
    std::vector<float> m_carry;
    std::vector<BoxcarPeak> m_peaks;
    std::vector<uint64_t> m_per_dm;
    std::vector<RawCandidate> m_candidates;
    std::vector<double> m_prefix;
    std::vector<uint64_t> m_integer_prefix, m_integer_carry;
    std::vector<uint32_t> m_integer_carry32;
    bool m_carry32{false};
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
    uint64_t m_search_stop    = UINT64_MAX; // drain(): end of stream
    uint64_t m_result_begin   = 0;
    SizeType m_result_samples = 0;
    // Per-call kill masks: off until the first one (then the per-block
    // tables below replace the planes' tables).
    uint64_t m_open_begin = UINT64_MAX; ///< see open_begin_min()
    bool m_noise_on       = false;
    std::shared_ptr<const std::vector<uint64_t>> m_support;
    std::vector<BlockNoise> m_ring; ///< oldest first, at most m_ring_size
    std::optional<BlockNoise> m_pending;
    SizeType m_ring_size = 1;
    std::vector<float> m_screen_thr; ///< per thread: min thresholds of a tile
};

} // namespace

std::unique_ptr<FDMTBoxcarEngine>
make_fdmt_boxcar_cpu(const plans::FDMTPlan& plan,
                     const BoxcarConfig& config,
                     const std::vector<plans::BoxcarPlanePlan>& planes,
                     const FDMTEngineConfig& cfg,
                     std::unique_ptr<BoxcarDriver> driver,
                     std::vector<std::pair<SizeType, SizeType>> kill_runs) {
    return std::make_unique<FDMTBoxcarCpuEngine>(
        plan, config, planes, cfg, std::move(driver), std::move(kill_runs));
}

} // namespace dmt::algorithms::detail
