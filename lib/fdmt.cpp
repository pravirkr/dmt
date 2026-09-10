#include "dmt/algorithms/fdmt.hpp"

#include <cstddef>
#include <format>
#include <stdexcept>
#include <utility>

#ifdef DMT_ENABLE_OPENMP
#include <omp.h>
#endif

#include <spdlog/spdlog.h>

#include "dmt/common/types.hpp"

namespace dmt::algorithms {

namespace {

enum class FDMTMode : uint8_t { kFull = 0, kValid = 1, kRoll = 2 };

/**
 * @brief Performs offset addition of two input arrays with mode-specific
 * boundary handling. This is the innermost hot loop of the FDMT algorithm.
 *
 * Core operation: out[k] = tail[k] + head[k - delay_shift]
 *
 * The function always performs addition in the overlap region
 * [delay_shift, size_tail), then handles edges according to the mode:
 *
 * - Full: Extends output beyond input size, zero-filling if needed
 * - Valid: Only valid overlap region (same size in/out)
 * - Roll: Cyclic boundary - head wraps around using modulo arithmetic
 *
 * @tparam Mode FDMTMode enum specifying boundary handling behavior
 *
 * @param data_tail Input array (tail component) - lower frequency band
 * @param size_tail Size of tail array (must equal size_head)
 * @param data_head Input array (head component) - higher frequency band
 * @param size_head Size of head array (must equal size_tail)
 * @param out Output array receiving the sum
 * @param size_out Size of output array (mode-dependent constraints)
 * @param delay_shift Delay offset in samples (dispersion-induced shift)
 *
 */
template <FDMTMode Mode>
void offset_add(const float* __restrict__ data_tail,
                SizeType size_tail,
                const float* __restrict__ data_head,
                SizeType size_head,
                float* __restrict__ out,
                SizeType size_out,
                SizeType delay_shift) noexcept {
    assert(size_tail == size_head && "Input tail and head sizes must be equal");
    assert(size_out >= size_tail && "Output size must be >= Input tail size");
    assert(delay_shift < size_tail && "Offset must be < input tail size");

    if constexpr (Mode == FDMTMode::kFull) {
        // Full mode: Output can extend beyond input size
        // Part 1: Copy tail-only region [0, delay_shift)
        std::copy_n(data_tail, delay_shift, out);

        // Part 2: Overlap region [delay_shift, size_tail)
        const SizeType nsum = size_tail - delay_shift;
#pragma omp simd
        for (SizeType i = 0; i < nsum; ++i) {
            out[delay_shift + i] = data_tail[delay_shift + i] + data_head[i];
        }

        // Part 3: Head-only region [size_tail, size_tail + nrest)
        const SizeType nrest = std::min(delay_shift, size_out - size_tail);
        if (nrest > 0) {
            std::copy_n(data_head + nsum, nrest, out + size_tail);
        }
        // Part 4: Zero-fill any remaining [size_tail + nrest, size_out)
        const SizeType filled = size_tail + nrest;
        if (filled < size_out) {
            std::fill(out + filled, out + size_out, 0.0F);
        }
    } else if constexpr (Mode == FDMTMode::kValid) {
        // Valid mode: only overlap region
        assert(size_out == size_tail &&
               "All sizes must be equal in valid mode");
        // Part 1: Copy tail-only region [0, delay_shift)
        std::copy_n(data_tail, delay_shift, out);

        // Part 2: Overlap region [delay_shift, size_tail)
        const SizeType nsum = size_tail - delay_shift;
#pragma omp simd
        for (SizeType i = 0; i < nsum; ++i) {
            out[delay_shift + i] = data_tail[delay_shift + i] + data_head[i];
        }
    } else if constexpr (Mode == FDMTMode::kRoll) {
        // Roll mode: cyclic addition
        assert(size_out == size_tail && "All sizes must be equal in roll mode");
        // Part 1: Wrapped region [0, delay_shift)
        for (SizeType k = 0; k < delay_shift; ++k) {
            out[k] = data_tail[k] + data_head[size_tail - delay_shift + k];
        }

        // Part 2: Overlap region [delay_shift, size_tail)
        const SizeType nsum = size_tail - delay_shift;
#pragma omp simd
        for (SizeType i = 0; i < nsum; ++i) {
            out[delay_shift + i] = data_tail[delay_shift + i] + data_head[i];
        }
    }
}

template <FDMTMode Mode>
void fdmt_iter(const float* __restrict__ state_in,
               float* __restrict__ state_out,
               const plans::FDMTCoord* __restrict__ coords_sum_cur,
               const plans::FDMTCoord* __restrict__ coords_copy_cur,
               SizeType ncoords_sum_cur,
               SizeType ncoords_copy_cur) noexcept {
#ifdef DMT_ENABLE_OPENMP
#pragma omp parallel default(none)                                             \
    shared(state_in, state_out, coords_sum_cur, coords_copy_cur,               \
               ncoords_sum_cur, ncoords_copy_cur)
#endif
    {
#ifdef DMT_ENABLE_OPENMP
#pragma omp for nowait
#endif
        for (SizeType i_coord = 0; i_coord < ncoords_sum_cur; ++i_coord) {
            const auto* __restrict__ coord = &coords_sum_cur[i_coord];
            const float* __restrict__ tail = &state_in[coord->tail_buf_offset];
            const float* __restrict__ head = &state_in[coord->head_buf_offset];
            float* __restrict__ out        = &state_out[coord->buf_offset];
            offset_add<Mode>(tail, coord->tail_nsamps, head, coord->head_nsamps,
                             out, coord->nsamps, coord->delay);
        }
#ifdef DMT_ENABLE_OPENMP
#pragma omp for
#endif
        for (SizeType i_coord = 0; i_coord < ncoords_copy_cur; ++i_coord) {
            const auto* __restrict__ coord = &coords_copy_cur[i_coord];
            const float* __restrict__ tail = &state_in[coord->tail_buf_offset];
            float* __restrict__ out        = &state_out[coord->buf_offset];
            std::copy_n(tail, coord->tail_nsamps, out);
            if (coord->nsamps > coord->tail_nsamps) {
                std::fill(out + coord->tail_nsamps, out + coord->nsamps, 0.0F);
            }
        }
    }
}

template <bool UseRoll, bool UseBoxSmearing>
void fdmt_init_impl_row0(const float* __restrict__ wf_sub,
                         float* __restrict__ buf_row0,
                         SizeType dt_min_sub,
                         SizeType nsamps) noexcept {
    // ===== First DT row (dt_min_sub) =====
    if constexpr (UseBoxSmearing) {
        float running_sum = 0.0F;

        // Initialize sum for isamp=0
        if constexpr (UseRoll) {
            // Sum samples at wrapped indices: [nsamps - dt_min_sub, nsamps)
            // and [0]
            for (SizeType i = 0; i < dt_min_sub; ++i) {
                running_sum += wf_sub[nsamps - dt_min_sub + i];
            }
            running_sum += wf_sub[0];
        } else {
            running_sum = wf_sub[0];
        }
        buf_row0[0] = running_sum;

        // Triangle region [1, dt_min_sub]: window slides into valid data
        for (SizeType isamp = 1; isamp <= dt_min_sub; ++isamp) {
            running_sum += wf_sub[isamp];
            if constexpr (UseRoll) {
                running_sum -= wf_sub[nsamps - dt_min_sub - 1 + isamp];
            }
            buf_row0[isamp] = running_sum;
        }

        // Main region [dt_min_sub + 1, nsamps): fully within bounds
        for (SizeType isamp = dt_min_sub + 1; isamp < nsamps; ++isamp) {
            running_sum += wf_sub[isamp];
            running_sum -= wf_sub[isamp - dt_min_sub - 1];
            buf_row0[isamp] = running_sum;
        }
    } else {
        // No smearing: just shift by dt_min_sub
        if constexpr (UseRoll) {
            // Triangle: wrap indices
            for (SizeType isamp = 0; isamp < dt_min_sub; ++isamp) {
                buf_row0[isamp] = wf_sub[nsamps - dt_min_sub + isamp];
            }
        } else {
            // Triangle: partial (no history)
            std::fill(buf_row0, buf_row0 + dt_min_sub, 0.0F);
        }
        // Main region
        for (SizeType isamp = dt_min_sub; isamp < nsamps; ++isamp) {
            buf_row0[isamp] = wf_sub[isamp - dt_min_sub];
        }
    }
}

template <bool UseRoll, bool UseBoxSmearing>
void fdmt_init_impl_row(const float* __restrict__ wf_sub,
                        const float* __restrict__ buf_prev,
                        float* __restrict__ buf_cur,
                        SizeType dt_cur,
                        SizeType nsamps) noexcept {
    if constexpr (UseBoxSmearing) {
        // Extend box sum: new_sum = prev_sum + waterfall[isamp - dt_cur]
        if constexpr (UseRoll) {
            for (SizeType isamp = 0; isamp < dt_cur; ++isamp) {
                buf_cur[isamp] =
                    buf_prev[isamp] + wf_sub[nsamps - dt_cur + isamp];
            }
        } else {
            // No new sample available, just copy previous partial sum
            for (SizeType isamp = 0; isamp < dt_cur; ++isamp) {
                buf_cur[isamp] = buf_prev[isamp];
            }
        }
        for (SizeType isamp = dt_cur; isamp < nsamps; ++isamp) {
            buf_cur[isamp] = buf_prev[isamp] + wf_sub[isamp - dt_cur];
        }
    } else {
        // No smearing: just shift by dt_cur
        if constexpr (UseRoll) {
            for (SizeType isamp = 0; isamp < dt_cur; ++isamp) {
                buf_cur[isamp] = wf_sub[nsamps - dt_cur + isamp];
            }
        } else {
            std::fill(buf_cur, buf_cur + dt_cur, 0.0F);
        }
        for (SizeType isamp = dt_cur; isamp < nsamps; ++isamp) {
            buf_cur[isamp] = wf_sub[isamp - dt_cur];
        }
    }
}

template <bool UseRoll, bool UseBoxSmearing>
void fdmt_init_impl(const float* __restrict__ waterfall,
                    float* __restrict__ init_buffer,
                    const plans::FDMTCoordGrid* __restrict__ grids_init,
                    SizeType nsubs,
                    SizeType nsamps) noexcept {
    // Preconditions (enforced by FDMTPlan::validate_inputs and make_plan):
    // - nsamps > 0, non-empty dt_grid per sub-band, dt_min_sub < nsamps

#ifdef DMT_ENABLE_OPENMP
#pragma omp parallel for default(none)                                         \
    shared(waterfall, init_buffer, grids_init, nsubs, nsamps)
#endif
    for (SizeType i_sub = 0; i_sub < nsubs; ++i_sub) {
        const auto& dt_grid_sub = grids_init[i_sub].dt_grid;
        const auto dt_min_sub   = dt_grid_sub[0];
        const auto ndt_sub      = dt_grid_sub.size();

        // Hoist pointer arithmetic
        const float* __restrict__ wf_sub = waterfall + (i_sub * nsamps);
        float* __restrict__ buf_base =
            init_buffer + (grids_init[i_sub].coord_offset * nsamps);

        fdmt_init_impl_row0<UseRoll, UseBoxSmearing>(wf_sub, buf_base,
                                                     dt_min_sub, nsamps);

        // ===== Subsequent DT rows (dt = dt_min_sub + 1, ...) =====
        for (SizeType i_dt = 1; i_dt < ndt_sub; ++i_dt) {
            const auto dt_cur           = dt_grid_sub[i_dt];
            float* __restrict__ buf_cur = buf_base + (i_dt * nsamps);
            const float* __restrict__ buf_prev =
                buf_base + ((i_dt - 1) * nsamps);
            fdmt_init_impl_row<UseRoll, UseBoxSmearing>(
                wf_sub, buf_prev, buf_cur, dt_cur, nsamps);
        }
    }
}

void fdmt_init_valid_row0_box(const float* __restrict__ wf_sub,
                              const float* __restrict__ hist_sub,
                              float* __restrict__ buf_row0,
                              SizeType dt_min_sub,
                              SizeType dt_max_final,
                              SizeType nsamps) noexcept {
    // ===== First DT row (dt_min_sub) =====
    // Box sum of (dt_min_sub + 1) samples ending at current position
    float running_sum = 0.0F;

    // --- isamp = 0: sum of logical indices [-dt_min_sub, 0] ---
    // Samples [-dt_min_sub, -1] from hist_sub, [0] from wf_sub
    for (SizeType i = 0; i < dt_min_sub; ++i) {
        running_sum += hist_sub[dt_max_final - dt_min_sub + i];
    }
    running_sum += wf_sub[0];
    buf_row0[0] = running_sum;

    // --- isamp in [1, dt_min_sub]: window slides into valid data ---
    // Add from wf_sub, remove from hist_sub
    for (SizeType isamp = 1; isamp <= dt_min_sub; ++isamp) {
        running_sum += wf_sub[isamp];
        running_sum -= hist_sub[dt_max_final + isamp - dt_min_sub - 1];
        buf_row0[isamp] = running_sum;
    }

    // --- isamp in [dt_min_sub + 1, nsamps): fully within waterfall ---
    // Add from wf_sub, remove from wf_sub
    for (SizeType isamp = dt_min_sub + 1; isamp < nsamps; ++isamp) {
        running_sum += wf_sub[isamp];
        running_sum -= wf_sub[isamp - dt_min_sub - 1];
        buf_row0[isamp] = running_sum;
    }
}

void fdmt_init_valid_row0_shift(const float* __restrict__ wf_sub,
                                const float* __restrict__ hist_sub,
                                float* __restrict__ buf_row0,
                                SizeType dt_min_sub,
                                SizeType dt_max_final,
                                SizeType nsamps) noexcept {
    // ===== First DT row (dt_min_sub) without smearing =====
    // For isamp in [0, dt_min_sub): read from history
    for (SizeType isamp = 0; isamp < dt_min_sub; ++isamp) {
        buf_row0[isamp] = hist_sub[dt_max_final - dt_min_sub + isamp];
    }
    // For isamp in [dt_min_sub, nsamps): read from waterfall
    for (SizeType isamp = dt_min_sub; isamp < nsamps; ++isamp) {
        buf_row0[isamp] = wf_sub[isamp - dt_min_sub];
    }
}

template <bool UseBoxSmearing>
void fdmt_init_valid_row(const float* __restrict__ wf_sub,
                         const float* __restrict__ hist_sub,
                         const float* __restrict__ buf_prev,
                         float* __restrict__ buf_cur,
                         SizeType dt_cur,
                         SizeType dt_max_final,
                         SizeType nsamps) noexcept {
    if constexpr (UseBoxSmearing) {
        // Extend box: new_sum[isamp] = prev_sum[isamp] + input[isamp - dt_cur]
        for (SizeType isamp = 0; isamp < dt_cur; ++isamp) {
            buf_cur[isamp] =
                buf_prev[isamp] + hist_sub[dt_max_final + isamp - dt_cur];
        }
        for (SizeType isamp = dt_cur; isamp < nsamps; ++isamp) {
            buf_cur[isamp] = buf_prev[isamp] + wf_sub[isamp - dt_cur];
        }
    } else {
        // No smearing: just shift by dt_cur
        for (SizeType isamp = 0; isamp < dt_cur; ++isamp) {
            buf_cur[isamp] = hist_sub[dt_max_final - dt_cur + isamp];
        }
        for (SizeType isamp = dt_cur; isamp < nsamps; ++isamp) {
            buf_cur[isamp] = wf_sub[isamp - dt_cur];
        }
    }
}

void fdmt_init_valid_update_history(const float* __restrict__ waterfall,
                                    float* __restrict__ hist_buffer,
                                    float* __restrict__ hist_init_buffer,
                                    SizeType nsubs,
                                    SizeType nsamps,
                                    SizeType dt_max_init,
                                    SizeType dt_max_final) noexcept {
    for (SizeType i_sub = 0; i_sub < nsubs; ++i_sub) {
        const float* __restrict__ wf_sub = waterfall + (i_sub * nsamps);
        float* __restrict__ hist_sub     = hist_buffer + (i_sub * dt_max_final);
        std::copy_n(wf_sub + nsamps - dt_max_final, dt_max_final, hist_sub);

        if (hist_init_buffer != nullptr && dt_max_init > 0) {
            float* __restrict__ hist_init_sub =
                hist_init_buffer + (i_sub * dt_max_init);
            std::copy_n(wf_sub + nsamps - dt_max_init, dt_max_init,
                        hist_init_sub);
        }
    }
}

// Specialized implementation for VALID mode with history
template <bool UseBoxSmearing>
void fdmt_init_valid_impl(const float* __restrict__ waterfall,
                          float* __restrict__ init_buffer,
                          float* __restrict__ hist_buffer,
                          float* __restrict__ hist_init_buffer,
                          const plans::FDMTCoordGrid* __restrict__ grids_init,
                          SizeType nsubs,
                          SizeType nsamps,
                          SizeType dt_max_init,
                          SizeType dt_max_final) noexcept {
    // Preconditions (enforced by FDMTPlan::validate_inputs and make_plan):
    // - nsamps > 0, dt_max_final >= 1, non-empty dt_grid per sub-band
    // - dt_min_sub <= dt_max_final for all sub-bands
#ifdef DMT_ENABLE_OPENMP
#pragma omp parallel for default(none)                                         \
    shared(waterfall, init_buffer, hist_buffer, grids_init, nsubs, nsamps,     \
               dt_max_final)
#endif
    for (SizeType i_sub = 0; i_sub < nsubs; ++i_sub) {
        const auto& dt_grid_sub = grids_init[i_sub].dt_grid;
        const auto dt_min_sub   = dt_grid_sub[0];
        const auto ndt_sub      = dt_grid_sub.size();

        const float* __restrict__ wf_sub = waterfall + (i_sub * nsamps);
        const float* __restrict__ hist_sub =
            hist_buffer + (i_sub * dt_max_final);
        float* __restrict__ buf_base =
            init_buffer + (grids_init[i_sub].coord_offset * nsamps);

        if constexpr (UseBoxSmearing) {
            fdmt_init_valid_row0_box(wf_sub, hist_sub, buf_base, dt_min_sub,
                                     dt_max_final, nsamps);
        } else {
            fdmt_init_valid_row0_shift(wf_sub, hist_sub, buf_base, dt_min_sub,
                                       dt_max_final, nsamps);
        }

        // Subsequent DT rows
        for (SizeType i_dt = 1; i_dt < ndt_sub; ++i_dt) {
            const auto dt_cur           = dt_grid_sub[i_dt];
            float* __restrict__ buf_cur = buf_base + (i_dt * nsamps);
            const float* __restrict__ buf_prev =
                buf_base + ((i_dt - 1) * nsamps);
            fdmt_init_valid_row<UseBoxSmearing>(wf_sub, hist_sub, buf_prev,
                                                buf_cur, dt_cur, dt_max_final,
                                                nsamps);
        }
    }

    fdmt_init_valid_update_history(waterfall, hist_buffer, hist_init_buffer,
                                   nsubs, nsamps, dt_max_init, dt_max_final);
}
} // namespace

class FDMTCPU::Impl {
public:
    Impl(float f_min,
         float f_max,
         SizeType nchans,
         SizeType nsamps,
         float tsamp,
         SizeType dt_max,
         SizeType dt_min,
         SizeType dt_step,
         bool use_box_smearing,
         std::string_view mode,
         bool verbose,
         int nthreads)
        : m_use_box_smearing(use_box_smearing),
          m_mode(parse_mode(mode)),
          m_plan(f_min,
                 f_max,
                 nchans,
                 nsamps,
                 tsamp,
                 dt_max,
                 dt_min,
                 dt_step,
                 mode,
                 verbose),
          m_state_internal(m_plan.get_buffer_size(), 0.0F),
          m_history(m_mode == FDMTMode::kValid ? m_plan.get_history_size() : 0,
                    0.0F),
          m_history_init(
              m_mode == FDMTMode::kValid ? m_plan.get_history_init_size() : 0,
              0.0F) {
        set_dmt_openmp_threads(nthreads);
    }

    Impl(float f_min,
         float f_max,
         SizeType nchans,
         SizeType nsamps,
         float tsamp,
         const std::vector<SizeType>& dt_grid,
         bool use_box_smearing,
         std::string_view mode,
         bool verbose,
         int nthreads)
        : m_use_box_smearing(use_box_smearing),
          m_mode(parse_mode(mode)),
          m_plan(f_min, f_max, nchans, nsamps, tsamp, dt_grid, mode, verbose),
          m_state_internal(m_plan.get_buffer_size(), 0.0F),
          m_history(m_mode == FDMTMode::kValid ? m_plan.get_history_size() : 0,
                    0.0F),
          m_history_init(
              m_mode == FDMTMode::kValid ? m_plan.get_history_init_size() : 0,
              0.0F) {
        set_dmt_openmp_threads(nthreads);
    }

    Impl(float f_min,
         float f_max,
         SizeType nchans,
         SizeType nsamps,
         float tsamp,
         const std::vector<float>& dm_grid,
         bool use_box_smearing,
         std::string_view mode,
         bool verbose,
         int nthreads)
        : m_use_box_smearing(use_box_smearing),
          m_mode(parse_mode(mode)),
          m_plan(f_min, f_max, nchans, nsamps, tsamp, dm_grid, mode, verbose),
          m_state_internal(m_plan.get_buffer_size(), 0.0F),
          m_history(m_mode == FDMTMode::kValid ? m_plan.get_history_size() : 0,
                    0.0F),
          m_history_init(
              m_mode == FDMTMode::kValid ? m_plan.get_history_init_size() : 0,
              0.0F) {
        set_dmt_openmp_threads(nthreads);
    }

    ~Impl()                      = default;
    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&)                 = delete;
    Impl& operator=(Impl&&)      = delete;

    const plans::FDMTPlan& get_plan() const { return m_plan; }

    void execute(std::span<const float> waterfall, std::span<float> dmt) {
        reset(waterfall, dmt);
        advance_until_remaining(0);
        finalize();
        spdlog::debug("FDMTCPU::Impl: Execution complete.");
    }

    void reset(std::span<const float> waterfall, std::span<float> dmt) {
        const auto nchans = m_plan.get_nchans();
        const auto nsamps = m_plan.get_nsamps();
        if (waterfall.size() != nchans * nsamps) {
            throw std::invalid_argument(std::format(
                "FDMTCPU: Invalid size of waterfall. Expected {}, got {}",
                nchans * nsamps, waterfall.size()));
        }
        if (dmt.size() < m_plan.get_buffer_size()) {
            throw std::invalid_argument(std::format(
                "FDMTCPU: Invalid size of dmt. Expected at least {}, got {}",
                m_plan.get_buffer_size(), dmt.size()));
        }

        m_dmt_target_ptr = dmt.data();
        m_current_level  = 0;

        // Number of internal ping-pong iterations (excluding the final write)
        const SizeType internal_iters = total_levels() - 2;
        const bool odd_swaps          = (internal_iters % 2) == 1;
        if (odd_swaps) {
            m_current_in_ptr  = dmt.data();
            m_current_out_ptr = m_state_internal.data();
        } else {
            m_current_in_ptr  = m_state_internal.data();
            m_current_out_ptr = dmt.data();
        }

        initialise(waterfall, m_current_in_ptr);
        m_is_initialized = true;
        spdlog::debug("FDMTCPU: Stepper initialized at level 0.");
    }

    void advance(SizeType levels = 1) {
        if (!m_is_initialized) {
            throw std::logic_error(
                "FDMTCPU: Stepper is not initialized. Call reset() first.");
        }
        const auto total_lvl = total_levels();
        while (levels > 0 && m_current_level < total_lvl - 1) {
            const SizeType next_level = m_current_level + 1;
            execute_iter(m_current_in_ptr, m_current_out_ptr, next_level);
            std::swap(m_current_in_ptr, m_current_out_ptr);
            m_current_level = next_level;
            --levels;
        }
    }

    void advance_until_remaining(SizeType remaining_levels) {
        if (!m_is_initialized) {
            throw std::logic_error(
                "FDMTCPU: Stepper is not initialized. Call reset() first.");
        }
        const auto total_lvl = total_levels();
        if (remaining_levels >= total_lvl) {
            return;
        }
        const SizeType target_level = total_lvl - 1 - remaining_levels;
        if (target_level > m_current_level) {
            advance(target_level - m_current_level);
        }
    }

    [[nodiscard]] std::span<const float> view_level_data() const {
        if (!m_is_initialized) {
            throw std::logic_error(
                "FDMTCPU: Stepper is not initialized. Call reset() first.");
        }
        const auto& state_shape =
            m_plan.get_container().state_shape[m_current_level];
        return {m_current_in_ptr, state_shape.nelements};
    }

    [[nodiscard]] std::span<const float>
    view_subband_data(SizeType subband_idx) const {
        if (!m_is_initialized) {
            throw std::logic_error(
                "FDMTCPU: Stepper is not initialized. Call reset() first.");
        }
        const auto& plan_c      = m_plan.get_container();
        const auto& state_shape = plan_c.state_shape[m_current_level];
        if (subband_idx >= state_shape.nchans) {
            throw std::out_of_range(std::format(
                "FDMTCPU: Subband index {} out of range (current level has {} "
                "subbands)",
                subband_idx, state_shape.nchans));
        }
        const auto& grid  = plan_c.grids[m_current_level][subband_idx];
        const auto offset = grid.coord_offset * state_shape.nsamps;
        const auto count  = grid.ndt * state_shape.nsamps;
        return {m_current_in_ptr + offset, count};
    }

    [[nodiscard]] FDMTSubbandView view_subband(SizeType subband_idx) const {
        if (!m_is_initialized) {
            throw std::logic_error(
                "FDMTCPU: Stepper is not initialized. Call reset() first.");
        }
        const auto& plan_c      = m_plan.get_container();
        const auto& state_shape = plan_c.state_shape[m_current_level];
        if (subband_idx >= state_shape.nchans) {
            throw std::out_of_range(std::format(
                "FDMTCPU: Subband index {} out of range (current level has {} "
                "subbands)",
                subband_idx, state_shape.nchans));
        }
        const auto& grid  = plan_c.grids[m_current_level][subband_idx];
        const auto offset = grid.coord_offset * state_shape.nsamps;
        const auto count  = grid.ndt * state_shape.nsamps;
        return FDMTSubbandView{
            .data = std::span<const float>(m_current_in_ptr + offset, count),
            .subband_idx = subband_idx,
            .ndt         = grid.ndt,
            .nsamps      = state_shape.nsamps,
            .f_start     = grid.f_start,
            .f_end       = grid.f_end,
            .dt_grid     = std::span<const SizeType>(grid.dt_grid.data(),
                                                     grid.dt_grid.size()),
        };
    }

    [[nodiscard]] SizeType current_level() const noexcept {
        return m_current_level;
    }

    [[nodiscard]] SizeType total_levels() const noexcept {
        return m_plan.get_niters() + 1;
    }

    [[nodiscard]] SizeType remaining_levels() const noexcept {
        if (total_levels() <= 1 || m_current_level >= total_levels() - 1) {
            return 0;
        }
        return (total_levels() - 1) - m_current_level;
    }

    [[nodiscard]] SizeType num_subbands() const {
        if (!m_is_initialized) {
            throw std::logic_error(
                "FDMTCPU: Stepper is not initialized. Call reset() first.");
        }
        return m_plan.get_container().state_shape[m_current_level].nchans;
    }

    [[nodiscard]] bool is_finished() const noexcept {
        return m_is_initialized && (m_current_level >= total_levels() - 1);
    }

    void finalize() {
        if (!m_is_initialized) {
            throw std::logic_error(
                "FDMTCPU: Stepper is not initialized. Call reset() first.");
        }
        if (!is_finished()) {
            advance_until_remaining(0);
        }
        m_is_initialized = false;
    }

private:
    bool m_use_box_smearing;
    FDMTMode m_mode;
    plans::FDMTPlan m_plan;
    // Internal state buffer (for ping-pong buffering)
    std::vector<float> m_state_internal;
    // History buffers for valid-mode streaming across FDMT blocks
    std::vector<float> m_history;
    std::vector<float> m_history_init; // only when use_box_smearing is true

    // Stepper state
    float* m_current_in_ptr{nullptr};
    float* m_current_out_ptr{nullptr};
    float* m_dmt_target_ptr{nullptr};
    SizeType m_current_level{0};
    bool m_is_initialized{false};

    static FDMTMode parse_mode(std::string_view mode) {
        if (mode == "full") {
            return FDMTMode::kFull;
        }
        if (mode == "roll") {
            return FDMTMode::kRoll;
        }
        if (mode == "valid") {
            return FDMTMode::kValid;
        }
        throw std::invalid_argument(std::format(
            "Invalid mode '{}'. Expected 'full', 'roll', or 'valid'", mode));
    }

    void initialise(std::span<const float> waterfall,
                    float* __restrict__ init_buffer) {
        const auto& plan_c     = m_plan.get_container();
        const auto& grids_init = plan_c.grids[0];
        const auto nsamps      = plan_c.state_shape[0].nsamps;
        const auto dt_max_init = plan_c.state_shape[0].dt_max;
        const auto dt_max_final =
            plan_c.state_shape[m_plan.get_niters()].dt_max;
        const auto nsubs = plan_c.state_shape[0].nchans;

        if (m_mode == FDMTMode::kFull) {
            if (m_use_box_smearing) {
                fdmt_init_impl<false, true>(waterfall.data(), init_buffer,
                                            grids_init.data(), nsubs, nsamps);
            } else {
                fdmt_init_impl<false, false>(waterfall.data(), init_buffer,
                                             grids_init.data(), nsubs, nsamps);
            }
        } else if (m_mode == FDMTMode::kRoll) {
            if (m_use_box_smearing) {
                fdmt_init_impl<true, true>(waterfall.data(), init_buffer,
                                           grids_init.data(), nsubs, nsamps);
            } else {
                fdmt_init_impl<true, false>(waterfall.data(), init_buffer,
                                            grids_init.data(), nsubs, nsamps);
            }
        } else {
            if (m_use_box_smearing) {
                fdmt_init_valid_impl<true>(
                    waterfall.data(), init_buffer, m_history.data(),
                    m_history_init.data(), grids_init.data(), nsubs, nsamps,
                    dt_max_init, dt_max_final);
            } else {
                fdmt_init_valid_impl<false>(
                    waterfall.data(), init_buffer, m_history.data(),
                    m_history_init.data(), grids_init.data(), nsubs, nsamps,
                    dt_max_init, dt_max_final);
            }
        }
    }

    void execute_iter(const float* __restrict__ state_in,
                      float* __restrict__ state_out,
                      SizeType i_iter) {
        const auto& plan_c          = m_plan.get_container();
        const auto& coords_sum_cur  = plan_c.coordinates_sum[i_iter];
        const auto& coords_copy_cur = plan_c.coordinates_copy[i_iter];
        const auto ncoords_sum_cur  = coords_sum_cur.size();
        const auto ncoords_copy_cur = coords_copy_cur.size();

        if (m_mode == FDMTMode::kFull) {
            fdmt_iter<FDMTMode::kFull>(
                state_in, state_out, coords_sum_cur.data(),
                coords_copy_cur.data(), ncoords_sum_cur, ncoords_copy_cur);
        } else if (m_mode == FDMTMode::kRoll) {
            fdmt_iter<FDMTMode::kRoll>(
                state_in, state_out, coords_sum_cur.data(),
                coords_copy_cur.data(), ncoords_sum_cur, ncoords_copy_cur);
        } else {
            fdmt_iter<FDMTMode::kValid>(
                state_in, state_out, coords_sum_cur.data(),
                coords_copy_cur.data(), ncoords_sum_cur, ncoords_copy_cur);
        }
    }
}; // End FDMTCPU::Impl definition

FDMTCPU::FDMTCPU(float f_min,
                 float f_max,
                 SizeType nchans,
                 SizeType nsamps,
                 float tsamp,
                 SizeType dt_max,
                 SizeType dt_min,
                 SizeType dt_step,
                 bool use_box_smearing,
                 std::string_view mode,
                 bool verbose,
                 int nthreads)
    : m_impl(std::make_unique<Impl>(f_min,
                                    f_max,
                                    nchans,
                                    nsamps,
                                    tsamp,
                                    dt_max,
                                    dt_min,
                                    dt_step,
                                    use_box_smearing,
                                    mode,
                                    verbose,
                                    nthreads)) {}
FDMTCPU::FDMTCPU(float f_min,
                 float f_max,
                 SizeType nchans,
                 SizeType nsamps,
                 float tsamp,
                 const std::vector<SizeType>& dt_grid,
                 bool use_box_smearing,
                 std::string_view mode,
                 bool verbose,
                 int nthreads)
    : m_impl(std::make_unique<Impl>(f_min,
                                    f_max,
                                    nchans,
                                    nsamps,
                                    tsamp,
                                    dt_grid,
                                    use_box_smearing,
                                    mode,
                                    verbose,
                                    nthreads)) {}

FDMTCPU::FDMTCPU(float f_min,
                 float f_max,
                 SizeType nchans,
                 SizeType nsamps,
                 float tsamp,
                 const std::vector<float>& dm_grid,
                 bool use_box_smearing,
                 std::string_view mode,
                 bool verbose,
                 int nthreads)
    : m_impl(std::make_unique<Impl>(f_min,
                                    f_max,
                                    nchans,
                                    nsamps,
                                    tsamp,
                                    dm_grid,
                                    use_box_smearing,
                                    mode,
                                    verbose,
                                    nthreads)) {}

FDMTCPU::~FDMTCPU()                                   = default;
FDMTCPU::FDMTCPU(FDMTCPU&& other) noexcept            = default;
FDMTCPU& FDMTCPU::operator=(FDMTCPU&& other) noexcept = default;
const plans::FDMTPlan& FDMTCPU::get_plan() const noexcept {
    return m_impl->get_plan();
}
void FDMTCPU::execute(std::span<const float> waterfall, std::span<float> dmt) {
    m_impl->execute(waterfall, dmt);
}
void FDMTCPU::reset(std::span<const float> waterfall, std::span<float> dmt) {
    m_impl->reset(waterfall, dmt);
}
void FDMTCPU::advance(SizeType levels) { m_impl->advance(levels); }
void FDMTCPU::advance_until_remaining(SizeType remaining_levels) {
    m_impl->advance_until_remaining(remaining_levels);
}
std::span<const float> FDMTCPU::view_level_data() const {
    return m_impl->view_level_data();
}
std::span<const float> FDMTCPU::view_subband_data(SizeType subband_idx) const {
    return m_impl->view_subband_data(subband_idx);
}
FDMTSubbandView FDMTCPU::view_subband(SizeType subband_idx) const {
    return m_impl->view_subband(subband_idx);
}
SizeType FDMTCPU::current_level() const noexcept {
    return m_impl->current_level();
}
SizeType FDMTCPU::total_levels() const noexcept {
    return m_impl->total_levels();
}
SizeType FDMTCPU::remaining_levels() const noexcept {
    return m_impl->remaining_levels();
}
SizeType FDMTCPU::num_subbands() const { return m_impl->num_subbands(); }
bool FDMTCPU::is_finished() const noexcept { return m_impl->is_finished(); }
void FDMTCPU::finalize() { m_impl->finalize(); }

[[nodiscard]] std::tuple<std::vector<float>, plans::FDMTPlan>
compute_fdmt(std::span<const float> waterfall,
             float f_min,
             float f_max,
             SizeType nchans,
             SizeType nsamps,
             float tsamp,
             SizeType dt_max,
             SizeType dt_min,
             SizeType dt_step,
             bool use_box_smearing,
             std::string_view mode,
             bool verbose,
             int nthreads) {
    FDMTCPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_min, dt_step,
                 use_box_smearing, mode, verbose, nthreads);
    const plans::FDMTPlan& fdmt_plan = fdmt.get_plan();
    const auto buffer_size           = fdmt_plan.get_buffer_size();
    std::vector<float> dmt(buffer_size, 0.0F);
    fdmt.execute(waterfall, dmt);
    // RESIZE to actual result size
    dmt.resize(fdmt_plan.get_dmt_size());
    return std::make_tuple(std::move(dmt), fdmt_plan);
}

[[nodiscard]] std::tuple<std::vector<float>, plans::FDMTPlan>
compute_fdmt(std::span<const float> waterfall,
             float f_min,
             float f_max,
             SizeType nchans,
             SizeType nsamps,
             float tsamp,
             const std::vector<SizeType>& dt_grid,
             bool use_box_smearing,
             std::string_view mode,
             bool verbose,
             int nthreads) {
    FDMTCPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_grid, use_box_smearing,
                 mode, verbose, nthreads);
    const plans::FDMTPlan& fdmt_plan = fdmt.get_plan();
    const auto buffer_size           = fdmt_plan.get_buffer_size();
    std::vector<float> dmt(buffer_size, 0.0F);
    fdmt.execute(waterfall, dmt);
    dmt.resize(fdmt_plan.get_dmt_size());
    return std::make_tuple(std::move(dmt), fdmt_plan);
}

[[nodiscard]] std::tuple<std::vector<float>, plans::FDMTPlan>
compute_fdmt(std::span<const float> waterfall,
             float f_min,
             float f_max,
             SizeType nchans,
             SizeType nsamps,
             float tsamp,
             const std::vector<float>& dm_grid,
             bool use_box_smearing,
             std::string_view mode,
             bool verbose,
             int nthreads) {
    FDMTCPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dm_grid, use_box_smearing,
                 mode, verbose, nthreads);
    const plans::FDMTPlan& fdmt_plan = fdmt.get_plan();
    const auto buffer_size           = fdmt_plan.get_buffer_size();
    std::vector<float> dmt(buffer_size, 0.0F);
    fdmt.execute(waterfall, dmt);
    dmt.resize(fdmt_plan.get_dmt_size());
    return std::make_tuple(std::move(dmt), fdmt_plan);
}

} // namespace dmt::algorithms