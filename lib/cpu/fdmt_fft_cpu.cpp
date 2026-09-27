#include "dmt/algorithms/fdmt_fft.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <format>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"
#include "dmt/engines.hpp"
#include "dmt/fft.hpp"
#include "dmt/modes.hpp"

namespace dmt::algorithms {

namespace {

void advance_overlap_window(const float* __restrict__ new_data,
                            float* __restrict__ hist,
                            SizeType nsamps,
                            SizeType capacity) noexcept {
    if (capacity == 0) {
        return;
    }
    if (nsamps >= capacity) {
        std::copy_n(new_data + nsamps - capacity, capacity, hist);
    } else {
        std::copy(hist + nsamps, hist + capacity, hist);
        std::copy_n(new_data, nsamps, hist + capacity - nsamps);
    }
}

class FDMTFFTCpuEngine final : public detail::FDMTFFTEngine {
public:
    FDMTFFTCpuEngine(const plans::FDMTPlan& plan,
                     const detail::FDMTFFTEngineConfig& cfg)
        : m_nchans(plan.get_nchans()),
          m_nsamps(plan.get_nsamps()),
          m_nbeams(cfg.nbeams),
          m_nthreads(std::max(1, cfg.exec.nthreads)),
          m_use_box_smearing(cfg.use_box_smearing),
          m_mode(cfg.mode),
          m_plan(&plan) {
        initialize();
    }

    void execute(std::span<const float> waterfall,
                 std::span<float> dmt) override {
        const auto total_in  = m_nbeams * m_nchans * m_nsamps;
        const auto total_out = m_nbeams * m_ndms * m_nsamps_out;
        if (waterfall.size() != total_in) {
            throw std::invalid_argument(std::format(
                "FDMTFFT::execute: expected waterfall size {}, got {}",
                total_in, waterfall.size()));
        }
        if (dmt.size() < total_out) {
            throw std::invalid_argument(std::format(
                "FDMTFFT::execute: dmt buffer size {} must be >= {}",
                dmt.size(), total_out));
        }
        for (SizeType b = 0; b < m_nbeams; ++b) {
            process_beam(waterfall.data() + (b * m_nchans * m_nsamps),
                         dmt.data() + (b * m_ndms * m_nsamps_out), b);
        }
    }

    void reset(std::span<const float> waterfall,
               std::span<float> dmt) override {
        const auto total_in  = m_nbeams * m_nchans * m_nsamps;
        const auto total_out = m_nbeams * m_ndms * m_nsamps_out;
        if (waterfall.size() != total_in) {
            throw std::invalid_argument(std::format(
                "FDMTFFT::reset: expected waterfall size {}, got {}", total_in,
                waterfall.size()));
        }
        if (dmt.size() < total_out) {
            throw std::invalid_argument(
                std::format("FDMTFFT::reset: dmt buffer size {} must be >= {}",
                            dmt.size(), total_out));
        }

        m_waterfall_ptr  = waterfall.data();
        m_dmt_target_ptr = dmt.data();
        m_current_level  = 0;
        m_state_in       = m_state_a.data();
        m_state_out      = m_state_b.data();
        m_view_valid     = false;

        // Every beam is transformed (same as the CPU FDMT). Views still expose
        // beam 0 only.
        for (SizeType b = 0; b < m_nbeams; ++b) {
            fill_window(waterfall.data() + (b * m_nchans * m_nsamps), b);
            m_fft_forward->execute(m_time_window, m_spectra);
            init_level0(m_state_in + (b * m_fft_buf_size));
        }
        m_is_initialized = true;
    }

    void advance(SizeType levels, Stream /*stream*/) override {
        require_stepper();
        const auto total_lvl = total_levels();
        while (levels > 0 && m_current_level < total_lvl - 1) {
            const SizeType next_level = m_current_level + 1;
            merge_iter(m_state_in, m_state_out, next_level, m_nbeams);
            std::swap(m_state_in, m_state_out);
            m_current_level = next_level;
            m_view_valid    = false;
            --levels;
        }
    }

    void advance_until_remaining(SizeType remaining_levels,
                                 Stream stream) override {
        require_stepper();
        const auto total_lvl = total_levels();
        if (remaining_levels >= total_lvl) {
            return;
        }
        const SizeType target_level = total_lvl - 1 - remaining_levels;
        if (target_level > m_current_level) {
            advance(target_level - m_current_level, stream);
        }
    }

    [[nodiscard]] std::span<const float> view_level_data() const override {
        require_stepper();
        materialize_view();
        const auto& shape =
            m_plan->get_container().state_shape[m_current_level];
        return {m_view_time.data(), shape.ncoords * view_nsamps()};
    }

    [[nodiscard]] FDMTSubbandView
    view_subband(SizeType subband_idx) const override {
        require_stepper();
        materialize_view();
        const auto& plan_c = m_plan->get_container();
        const auto& shape  = plan_c.state_shape[m_current_level];
        if (subband_idx >= shape.nchans) {
            throw std::out_of_range(std::format(
                "FDMTFFT: Subband index {} out of range ({} subbands)",
                subband_idx, shape.nchans));
        }
        const auto& grid    = plan_c.grids[m_current_level][subband_idx];
        const auto nsamps_v = view_nsamps();
        const auto offset   = grid.coord_offset * nsamps_v;
        const auto count    = grid.ndt * nsamps_v;
        return FDMTSubbandView{
            .data = std::span<const float>(m_view_time.data() + offset, count),
            .subband_idx = subband_idx,
            .ndt         = grid.ndt,
            .nsamps      = nsamps_v,
            .f_start     = grid.f_start,
            .f_end       = grid.f_end,
            .dt_grid     = std::span<const IndexType>(grid.dt_grid.data(),
                                                      grid.dt_grid.size()),
        };
    }

    [[nodiscard]] SizeType current_level() const noexcept override {
        return m_current_level;
    }
    [[nodiscard]] SizeType total_levels() const noexcept {
        return m_plan->get_niters() + 1;
    }
    [[nodiscard]] SizeType num_subbands() const override {
        require_stepper();
        return m_plan->get_container().state_shape[m_current_level].nchans;
    }
    [[nodiscard]] bool is_finished() const noexcept override {
        return m_is_initialized && (m_current_level >= total_levels() - 1);
    }

    void finalize(Stream stream) override {
        require_stepper();
        if (!is_finished()) {
            advance_until_remaining(0, stream);
        }
        for (SizeType b = 0; b < m_nbeams; ++b) {
            inverse_and_store(m_state_in + (b * m_fft_buf_size),
                              m_dmt_target_ptr + (b * m_ndms * m_nsamps_out));
            if (m_mode == FDMTMode::kValid && m_waterfall_ptr != nullptr) {
                update_overlap(m_waterfall_ptr + (b * m_nchans * m_nsamps), b);
            }
        }
        m_is_initialized = false;
        m_view_valid     = false;
    }

    void reset_history() noexcept override {
        std::ranges::fill(m_overlap, 0.0F);
    }

protected:
    [[nodiscard]] Backend backend() const noexcept override {
        return Backend::kCPU;
    }

private:
    SizeType m_nchans;
    SizeType m_nsamps;
    SizeType m_nbeams;
    int m_nthreads;
    bool m_use_box_smearing;
    FDMTMode m_mode;

    const plans::FDMTPlan* m_plan; // owned by the FDMTFFT facade
    SizeType m_ndms{};
    SizeType m_n_bins{};
    SizeType m_n_fft{};
    SizeType m_fft_buf_size{};
    SizeType m_max_s{};
    SizeType m_max_coords{};
    SizeType m_overlap_len{};
    SizeType m_nsamps_out{};
    SizeType m_out_skip{};

    std::vector<ComplexType> m_phasors;
    std::vector<ComplexType> m_boxcar_window;
    std::vector<ComplexType> m_spectra;
    std::vector<ComplexType> m_state_a;
    std::vector<ComplexType> m_state_b;
    std::vector<float> m_time_window;
    mutable std::vector<float> m_time_out;
    mutable std::vector<float> m_view_time;
    std::vector<float> m_overlap;
    mutable std::vector<ComplexType> m_ifft_in;

    std::unique_ptr<utils::FFTWManager> m_fft_forward;
    std::unique_ptr<utils::FFTWManager> m_fft_backward;

    const float* m_waterfall_ptr{nullptr};
    float* m_dmt_target_ptr{nullptr};
    ComplexType* m_state_in{nullptr};
    ComplexType* m_state_out{nullptr};
    SizeType m_current_level{0};
    bool m_is_initialized{false};
    mutable bool m_view_valid{false};

    void initialize() {
        m_ndms         = m_plan->get_dmt_ndms();
        m_n_fft        = m_plan->get_fft_size();
        m_n_bins       = m_plan->get_fft_n_bins();
        m_fft_buf_size = m_plan->get_fft_buffer_size();
        m_max_s        = m_plan->get_max_shift();
        m_overlap_len  = m_plan->get_fft_overlap();
        m_nsamps_out   = m_plan->get_dmt_nsamps();
        m_out_skip     = (m_mode == FDMTMode::kValid) ? m_overlap_len : 0;

        m_max_coords = 0;
        for (const auto& shape : m_plan->get_container().state_shape) {
            m_max_coords = std::max(m_max_coords, shape.ncoords);
        }

        m_phasors = m_plan->get_fft_phasor_table();
        m_boxcar_window.resize((m_max_s + 1) * m_n_bins);
        for (SizeType k = 0; k < m_n_bins; ++k) {
            ComplexType accum{0.0F, 0.0F};
            for (SizeType s = 0; s <= m_max_s; ++s) {
                accum += m_phasors[(s * m_n_bins) + k];
                m_boxcar_window[(s * m_n_bins) + k] = accum;
            }
        }

        m_spectra.resize(m_nchans * m_n_bins);
        m_state_a.resize(m_nbeams * m_fft_buf_size);
        m_state_b.resize(m_nbeams * m_fft_buf_size);
        m_ifft_in.resize(m_fft_buf_size);
        m_time_window.resize(m_nchans * m_n_fft, 0.0F);
        m_time_out.resize(m_max_coords * m_n_fft, 0.0F);
        m_view_time.resize(m_max_coords * m_nsamps_out, 0.0F);
        if (m_mode == FDMTMode::kValid) {
            m_overlap.assign(m_nbeams * m_nchans * m_overlap_len, 0.0F);
        }

        m_fft_forward = std::make_unique<utils::FFTWManager>(
            utils::FFTKind::kR2C, m_n_fft, m_nchans, m_nthreads);
        m_fft_backward = std::make_unique<utils::FFTWManager>(
            utils::FFTKind::kC2R, m_n_fft, m_max_coords, m_nthreads);
    }

    void process_beam(const float* wf, float* dmt_ptr, SizeType beam) {
        fill_window(wf, beam);
        m_fft_forward->execute(m_time_window, m_spectra);
        ComplexType* state_in  = m_state_a.data();
        ComplexType* state_out = m_state_b.data();
        init_level0(state_in);
        const auto niters = m_plan->get_niters();
        for (SizeType i_iter = 1; i_iter <= niters; ++i_iter) {
            merge_iter(state_in, state_out, i_iter, 1);
            std::swap(state_in, state_out);
        }
        inverse_and_store(state_in, dmt_ptr);
        if (m_mode == FDMTMode::kValid) {
            update_overlap(wf, beam);
        }
    }

    void fill_window(const float* wf, SizeType beam) {
        std::ranges::fill(m_time_window, 0.0F);
        if (m_mode == FDMTMode::kRoll) {
            std::copy_n(wf, m_nchans * m_nsamps, m_time_window.data());
            return;
        }
        if (m_mode == FDMTMode::kFull) {
            for (SizeType c = 0; c < m_nchans; ++c) {
                std::copy_n(wf + (c * m_nsamps), m_nsamps,
                            m_time_window.data() + (c * m_n_fft));
            }
            return;
        }
        for (SizeType c = 0; c < m_nchans; ++c) {
            float* dst = m_time_window.data() + (c * m_n_fft);
            const float* ov =
                m_overlap.data() + (((beam * m_nchans) + c) * m_overlap_len);
            if (m_overlap_len > 0) {
                std::copy_n(ov, m_overlap_len, dst);
            }
            std::copy_n(wf + (c * m_nsamps), m_nsamps, dst + m_overlap_len);
        }
    }

    void update_overlap(const float* wf, SizeType beam) {
        if (m_overlap_len == 0) {
            return;
        }
        for (SizeType c = 0; c < m_nchans; ++c) {
            float* ov =
                m_overlap.data() + (((beam * m_nchans) + c) * m_overlap_len);
            advance_overlap_window(wf + (c * m_nsamps), ov, m_nsamps,
                                   m_overlap_len);
        }
    }

    void init_level0(ComplexType* state_0) const {
        const auto& grid0 = m_plan->get_container().grids[0];
#pragma omp parallel for default(none) num_threads(m_nthreads)                 \
    shared(grid0, state_0)
        for (SizeType i_sub = 0; i_sub < m_nchans; ++i_sub) {
            const auto& g           = grid0[i_sub];
            const auto coord_base   = g.coord_offset;
            const auto ndt          = g.ndt;
            const ComplexType* spec = &m_spectra[i_sub * m_n_bins];
            for (SizeType i_dt = 0; i_dt < ndt; ++i_dt) {
                const auto coord_idx   = coord_base + i_dt;
                const auto dt          = g.dt_grid[i_dt];
                const auto s           = static_cast<SizeType>(std::abs(dt));
                const ComplexType* win = m_use_box_smearing
                                             ? &m_boxcar_window[s * m_n_bins]
                                             : &m_phasors[s * m_n_bins];
                ComplexType* out       = &state_0[coord_idx * m_n_bins];
#pragma omp simd
                for (SizeType k = 0; k < m_n_bins; ++k) {
                    out[k] = spec[k] * win[k];
                }
            }
        }
    }

    void merge_iter(const ComplexType* state_in,
                    ComplexType* state_out,
                    SizeType i_iter,
                    SizeType nbeams) const {
        const auto& plan_c      = m_plan->get_container();
        const auto& coords_sum  = plan_c.coordinates_sum[i_iter];
        const auto& coords_copy = plan_c.coordinates_copy[i_iter];
        const auto n_sum        = coords_sum.size();
        const auto n_copy       = coords_copy.size();

        for (SizeType b = 0; b < nbeams; ++b) {
            const ComplexType* in_b = state_in + (b * m_fft_buf_size);
            ComplexType* out_b      = state_out + (b * m_fft_buf_size);
#pragma omp parallel default(none) num_threads(m_nthreads)                     \
    shared(in_b, out_b, coords_sum, coords_copy, n_sum, n_copy)
            {
#pragma omp for nowait
                for (SizeType i = 0; i < n_sum; ++i) {
                    const auto& coord   = coords_sum[i];
                    const auto tail_idx = coord_index_from_offset(
                        coord.tail_buf_offset, coord.tail_nsamps);
                    const auto head_idx = coord_index_from_offset(
                        coord.head_buf_offset, coord.head_nsamps);
                    const auto cur_idx =
                        coord_index_from_offset(coord.buf_offset, coord.nsamps);
                    const auto s            = coord.delay;
                    const ComplexType* p    = &m_phasors[s * m_n_bins];
                    const ComplexType* tail = &in_b[tail_idx * m_n_bins];
                    const ComplexType* head = &in_b[head_idx * m_n_bins];
                    ComplexType* out        = &out_b[cur_idx * m_n_bins];
#pragma omp simd
                    for (SizeType k = 0; k < m_n_bins; ++k) {
                        out[k] = tail[k] + (head[k] * p[k]);
                    }
                }
#pragma omp for
                for (SizeType i = 0; i < n_copy; ++i) {
                    const auto& coord   = coords_copy[i];
                    const auto tail_idx = coord_index_from_offset(
                        coord.tail_buf_offset, coord.tail_nsamps);
                    const auto cur_idx =
                        coord_index_from_offset(coord.buf_offset, coord.nsamps);
                    std::copy_n(&in_b[tail_idx * m_n_bins], m_n_bins,
                                &out_b[cur_idx * m_n_bins]);
                }
            }
        }
    }

    static SizeType coord_index_from_offset(SizeType buf_offset,
                                            SizeType nsamps) {
        assert(nsamps != 0 && "FDMTFFT: nsamps must be non-zero");
        return buf_offset / nsamps;
    }

    // Copies the IFFT of `state_root` into `dmt_ptr` of length
    // ndms * nsamps_out. For mode=full, samples t >= nsamps (the delay
    // tail) are the Fourier linear-convolution continuation and are not
    // guaranteed to match the CPU FDMT's per-level growing-buffer tail; the
    // input-aligned region t < nsamps does match. See docs/fdmt-fft.md.
    void inverse_and_store(const ComplexType* state_root, float* dmt_ptr) {
        std::copy_n(state_root, m_fft_buf_size, m_ifft_in.data());
        m_fft_backward->execute(m_time_out, m_ifft_in);

        const float norm = 1.0F / static_cast<float>(m_n_fft);
#pragma omp parallel for default(none) num_threads(m_nthreads)                 \
    shared(dmt_ptr, norm)
        for (SizeType dm = 0; dm < m_ndms; ++dm) {
            const float* src = m_time_out.data() + (dm * m_n_fft) + m_out_skip;
            float* dst       = dmt_ptr + (dm * m_nsamps_out);
            for (SizeType t = 0; t < m_nsamps_out; ++t) {
                dst[t] = src[t] * norm;
            }
        }
    }

    [[nodiscard]] SizeType view_nsamps() const {
        if (m_mode == FDMTMode::kFull) {
            return m_plan->get_container().state_shape[m_current_level].nsamps;
        }
        return m_nsamps;
    }

    [[nodiscard]] SizeType view_skip() const {
        return (m_mode == FDMTMode::kValid) ? m_overlap_len : 0;
    }

    void materialize_view() const {
        if (m_view_valid) {
            return;
        }
        std::copy_n(m_state_in, m_fft_buf_size, m_ifft_in.data());
        m_fft_backward->execute(m_time_out, m_ifft_in);

        const auto& shape =
            m_plan->get_container().state_shape[m_current_level];
        const auto nsamps_v = view_nsamps();
        const auto skip     = view_skip();
        const float norm    = 1.0F / static_cast<float>(m_n_fft);
        m_view_time.assign(shape.ncoords * nsamps_v, 0.0F);
        for (SizeType c = 0; c < shape.ncoords; ++c) {
            const float* src = m_time_out.data() + (c * m_n_fft) + skip;
            float* dst       = m_view_time.data() + (c * nsamps_v);
            const auto ncopy =
                (m_n_fft > skip) ? std::min(nsamps_v, m_n_fft - skip) : 0;
            for (SizeType t = 0; t < ncopy; ++t) {
                dst[t] = src[t] * norm;
            }
        }
        m_view_valid = true;
    }

    void require_stepper() const {
        if (!m_is_initialized) {
            throw std::logic_error(
                "FDMTFFT: Stepper is not initialized. Call reset() first.");
        }
    }
};

} // namespace

std::unique_ptr<detail::FDMTFFTEngine>
detail::make_fdmt_fft_cpu(const plans::FDMTPlan& plan,
                          const detail::FDMTFFTEngineConfig& cfg) {
    return std::make_unique<FDMTFFTCpuEngine>(plan, cfg);
}

} // namespace dmt::algorithms
