#include "dmt/algorithms/fdmt.hpp"

#include <cstddef>
#include <format>
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

    const SizeType nsum = size_tail - delay_shift;
    // Part 1: Overlap region [delay_shift, size_tail)
#pragma omp simd
    for (SizeType i = 0; i < nsum; ++i) {
        out[delay_shift + i] = data_tail[delay_shift + i] + data_head[i];
    }

    if constexpr (Mode == FDMTMode::kFull) {
        // Full mode: Output can extend beyond input size
        // Part 2: Copy tail-only region [0, delay_shift)
        std::copy_n(data_tail, delay_shift, out);
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
        // Part 2: Copy tail-only region [0, delay_shift)
        std::copy_n(data_tail, delay_shift, out);
    } else if constexpr (Mode == FDMTMode::kRoll) {
        // Roll mode: cyclic addition
        assert(size_out == size_tail && "All sizes must be equal in roll mode");
        // Part 2: Wrapped region [0, delay_shift)
        for (SizeType k = 0; k < delay_shift; ++k) {
            out[k] = data_tail[k] + data_head[size_tail - delay_shift + k];
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
        }
    }
}

/*
// Before calling fdmt_init_impl:
assert(nsamps > 0);
for (SizeType i_sub = 0; i_sub < nsubs; ++i_sub) {
    assert(!grids_init[i_sub].dt_grid.empty());
    assert(grids_init[i_sub].dt_grid[0] < nsamps);
    assert(grids_init[i_sub].dt_grid.back() < nsamps);  // if applicable
}
*/
template <bool UseRoll, bool UseBoxSmearing>
void fdmt_init_impl(const float* __restrict__ waterfall,
                    float* __restrict__ init_buffer,
                    const plans::FDMTCoordGrid* __restrict__ grids_init,
                    SizeType nsubs,
                    SizeType nsamps) noexcept {
    // Preconditions (enforced at higher level):
    // - For all i_sub: grids_init[i_sub].dt_grid[0] < nsamps
    // - For all i_sub: grids_init[i_sub].dt_grid.size() > 0

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

        // ===== First DT row (dt_min_sub) =====
        float* __restrict__ buf_row0 = buf_base;

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

        // ===== Subsequent DT rows (dt = dt_min_sub + 1, ...) =====
        for (SizeType i_dt = 1; i_dt < ndt_sub; ++i_dt) {
            const auto dt_cur           = dt_grid_sub[i_dt];
            float* __restrict__ buf_cur = buf_base + (i_dt * nsamps);
            const float* __restrict__ buf_prev =
                buf_base + ((i_dt - 1) * nsamps);

            if constexpr (UseBoxSmearing) {
                // Extend box sum: new_sum = prev_sum + waterfall[isamp -
                // dt_cur]
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
    }
}

/*
assert(nsamps > dt_max_final + dt_max_init);  // nsamps >>> these values
assert(dt_max_init >= 1);
assert(dt_max_final >= 1);
for (SizeType i_sub = 0; i_sub < nsubs; ++i_sub) {
    assert(!grids_init[i_sub].dt_grid.empty());
    assert(grids_init[i_sub].dt_grid[0] <= dt_max_init);  // dt_min <=
dt_max_init assert(grids_init[i_sub].dt_grid.back() <= dt_max_init);  // all dt
values in init stage
}
*/
// Specialized implementation for VALID mode with two history buffers
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
    // Preconditions (enforced at higher level):
    // - dt_min_sub <= dt_max_init for all sub-bands
    // - nsamps >> dt_max_final + dt_max_init
    // - dt_max_init >= 1, dt_max_final >= 1
    const SizeType nsamps_ext = dt_max_final + nsamps;

#ifdef DMT_ENABLE_OPENMP
#pragma omp parallel for default(none)                                         \
    shared(waterfall, init_buffer, hist_buffer, hist_init_buffer, grids_init,  \
               nsubs, nsamps, nsamps_ext, dt_max_init, dt_max_final)
#endif
    for (SizeType i_sub = 0; i_sub < nsubs; ++i_sub) {
        const auto& dt_grid_sub = grids_init[i_sub].dt_grid;
        const auto dt_min_sub   = dt_grid_sub[0];
        const auto ndt_sub      = dt_grid_sub.size();

        // Hoist pointer arithmetic
        const float* __restrict__ wf_sub = waterfall + (i_sub * nsamps);
        const float* __restrict__ hist_sub =
            hist_buffer + (i_sub * dt_max_final);
        const float* __restrict__ hist_init_sub =
            hist_init_buffer + (i_sub * dt_max_init);
        float* __restrict__ buf_base =
            init_buffer + (grids_init[i_sub].coord_offset * nsamps_ext);

        // ===== First DT row (dt_min_sub) =====
        float* __restrict__ buf_row0 = buf_base;

        if constexpr (UseBoxSmearing) {
            // Box sum of (dt_min_sub + 1) samples ending at current position
            float running_sum = 0.0F;

            // --- isamp = 0: sum of logical indices [-dt_min_sub, 0] ---
            // [-dt_min_sub, -1] from hist_init, [0] from hist_buffer
            for (SizeType i = 0; i < dt_min_sub; ++i) {
                // Logical index: -dt_min_sub + i (ranges from -dt_min_sub to
                // -1) hist_init index: dt_max_init - dt_min_sub + i
                running_sum += hist_init_sub[dt_max_init - dt_min_sub + i];
            }
            running_sum += hist_sub[0]; // Logical index 0
            buf_row0[0] = running_sum;

            // --- Zone A: isamp in [1, dt_min_sub] ---
            // Add from hist_buffer, remove from hist_init
            for (SizeType isamp = 1; isamp <= dt_min_sub; ++isamp) {
                // Add: logical index isamp -> hist_buffer[isamp]
                running_sum += hist_sub[isamp];
                // Remove: logical index (isamp - dt_min_sub - 1) -> hist_init
                // isamp - dt_min_sub - 1 ranges from -dt_min_sub to -1
                running_sum -=
                    hist_init_sub[dt_max_init + (isamp - dt_min_sub - 1)];
                buf_row0[isamp] = running_sum;
            }

            // --- Zone B: isamp in [dt_min_sub + 1, dt_max_final) ---
            // Add from hist_buffer, remove from hist_buffer
            for (SizeType isamp = dt_min_sub + 1; isamp < dt_max_final;
                 ++isamp) {
                running_sum += hist_sub[isamp];
                running_sum -= hist_sub[isamp - dt_min_sub - 1];
                buf_row0[isamp] = running_sum;
            }

            // --- Zone C: isamp in [dt_max_final, dt_max_final + dt_min + 1)
            // --- Add from waterfall, remove from hist_buffer
            const SizeType zone_c_end = dt_max_final + dt_min_sub + 1;
            for (SizeType isamp = dt_max_final; isamp < zone_c_end; ++isamp) {
                // Add: logical index isamp -> waterfall[isamp - dt_max_final]
                running_sum += wf_sub[isamp - dt_max_final];
                // Remove: logical index (isamp - dt_min_sub - 1) -> hist_buffer
                running_sum -= hist_sub[isamp - dt_min_sub - 1];
                buf_row0[isamp] = running_sum;
            }

            // --- Zone D: isamp in [zone_c_end, nsamps_ext) ---
            // Add from waterfall, remove from waterfall
            for (SizeType isamp = zone_c_end; isamp < nsamps_ext; ++isamp) {
                const SizeType wf_idx = isamp - dt_max_final;
                running_sum += wf_sub[wf_idx];
                running_sum -= wf_sub[wf_idx - dt_min_sub - 1];
                buf_row0[isamp] = running_sum;
            }
        } else {
            // No smearing: just shift by dt_min_sub
            // --- Zone A: isamp in [0, dt_min_sub) ---
            // Source: hist_init (logical index isamp - dt_min_sub is in
            // [-dt_min_sub, -1])
            for (SizeType isamp = 0; isamp < dt_min_sub; ++isamp) {
                // Logical source: isamp - dt_min_sub (negative)
                buf_row0[isamp] =
                    hist_init_sub[dt_max_init - dt_min_sub + isamp];
            }
            // --- Zone B: isamp in [dt_min_sub, dt_max_final + dt_min_sub) ---
            // Source: hist_buffer (logical index isamp - dt_min_sub is in [0,
            // dt_max_final))
            for (SizeType isamp = dt_min_sub; isamp < dt_max_final + dt_min_sub;
                 ++isamp) {
                buf_row0[isamp] = hist_sub[isamp - dt_min_sub];
            }

            // --- Zone C: isamp in [dt_max_final + dt_min_sub, nsamps_ext) ---
            // Source: waterfall (logical index isamp - dt_min_sub is in
            // [dt_max_final, ...))
            for (SizeType isamp = dt_max_final + dt_min_sub; isamp < nsamps_ext;
                 ++isamp) {
                buf_row0[isamp] = wf_sub[(isamp - dt_max_final) - dt_min_sub];
            }
        }

        // ===== Subsequent DT rows (dt = dt_min_sub + 1, ...) =====
        for (SizeType i_dt = 1; i_dt < ndt_sub; ++i_dt) {
            const auto dt_cur           = dt_grid_sub[i_dt];
            float* __restrict__ buf_cur = buf_base + (i_dt * nsamps_ext);
            const float* __restrict__ buf_prev =
                buf_base + ((i_dt - 1) * nsamps_ext);

            if constexpr (UseBoxSmearing) {
                // Extend box: new_sum[isamp] = prev_sum[isamp] + input[isamp -
                // dt_cur]

                // --- Zone A: isamp in [0, dt_cur) ---
                for (SizeType isamp = 0; isamp < dt_cur; ++isamp) {
                    buf_cur[isamp] =
                        buf_prev[isamp] +
                        hist_init_sub[dt_max_init + (isamp - dt_cur)];
                }

                // --- Zone B: isamp in [dt_cur, dt_max_final + dt_cur) ---
                const SizeType zone_b_end = dt_max_final + dt_cur;
                for (SizeType isamp = dt_cur; isamp < zone_b_end; ++isamp) {
                    buf_cur[isamp] = buf_prev[isamp] + hist_sub[isamp - dt_cur];
                }

                // --- Zone C: isamp in [zone_b_end, nsamps_ext) ---
                for (SizeType isamp = zone_b_end; isamp < nsamps_ext; ++isamp) {
                    buf_cur[isamp] =
                        buf_prev[isamp] + wf_sub[isamp - dt_cur - dt_max_final];
                }
            } else {
                // No smearing: just shift by dt_cur
                // --- Zone A: isamp in [0, dt_cur) ---
                for (SizeType isamp = 0; isamp < dt_cur; ++isamp) {
                    buf_cur[isamp] =
                        hist_init_sub[dt_max_init - dt_cur + isamp];
                }
                // --- Zone B: isamp in [dt_cur, dt_max_final + dt_cur) ---
                const SizeType zone_b_end = dt_max_final + dt_cur;
                for (SizeType isamp = dt_cur; isamp < zone_b_end; ++isamp) {
                    buf_cur[isamp] = hist_sub[isamp - dt_cur];
                }

                // --- Zone C: isamp in [zone_b_end, nsamps_ext) ---
                for (SizeType isamp = zone_b_end; isamp < nsamps_ext; ++isamp) {
                    buf_cur[isamp] = wf_sub[isamp - dt_cur - dt_max_final];
                }
            }
        }
    }

    // Update history buffer
    for (SizeType i_sub = 0; i_sub < nsubs; ++i_sub) {
        const float* __restrict__ wf_sub = waterfall + (i_sub * nsamps);
        float* __restrict__ hist_sub     = hist_buffer + (i_sub * dt_max_final);
        float* __restrict__ hist_init_sub =
            hist_init_buffer + (i_sub * dt_max_init);

        // hist_init <- waterfall[nsamps - dt_max_final - dt_max_init : nsamps -
        // dt_max_final]
        std::copy_n(wf_sub + nsamps - dt_max_final - dt_max_init, dt_max_init,
                    hist_init_sub);

        // hist_buffer <- waterfall[nsamps - dt_max_final : nsamps]
        std::copy_n(wf_sub + nsamps - dt_max_final, dt_max_final, hist_sub);
    }
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
                 mode,
                 verbose),
          m_state_internal(m_plan.get_buffer_size(), 0.0F),
          m_history(m_mode == FDMTMode::kValid ? m_plan.get_history_size() : 0,
                    0.0F),
          m_history_init(m_mode == FDMTMode::kValid && m_use_box_smearing
                             ? m_plan.get_history_init_size()
                             : 0,
                         0.0F),
          m_nthreads(set_dmt_openmp_threads(nthreads)) {}

    ~Impl()                      = default;
    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&)                 = delete;
    Impl& operator=(Impl&&)      = delete;

    const plans::FDMTPlan& get_plan() const { return m_plan; }

    void execute(std::span<const float> waterfall, std::span<float> dmt) {
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
        spdlog::debug("FDMTCPU: Input dimensions check passed: {}x{}", nchans,
                      nsamps);
        execute_unified(waterfall, dmt);
    }

private:
    bool m_use_box_smearing;
    FDMTMode m_mode;
    plans::FDMTPlan m_plan;
    // Internal state buffer (for ping-pong buffering)
    std::vector<float> m_state_internal;
    // History buffers - only allocated for "valid" mode
    std::vector<float> m_history;
    std::vector<float> m_history_init; // only when use_box_smearing is true
    int m_nthreads;

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

    void execute_unified(std::span<const float> waterfall,
                         std::span<float> dmt) {
        const auto levels = m_plan.get_niters() + 1;
        if (levels < 2) {
            throw std::invalid_argument(
                std::format("FDMTCPU: Invalid number of levels. Expected at "
                            "least 2, got {}",
                            levels));
        }
        float* current_in_ptr  = nullptr;
        float* current_out_ptr = nullptr;
        // Number of internal ping-pong iterations (excluding the final write)
        const SizeType internal_iters = levels - 2;
        // Determine starting configuration to ensure final result lands in the
        // correct side of the ping-pong table
        const bool odd_swaps = (internal_iters % 2) == 1;
        if (odd_swaps) {
            // init -> result, odd swaps -> dmt ends in result
            current_in_ptr  = dmt.data();
            current_out_ptr = m_state_internal.data();
        } else {
            // init -> internal, even swaps -> dmt ends in result
            current_in_ptr  = m_state_internal.data();
            current_out_ptr = dmt.data();
        }

        // Initialize in the current buffer
        initialise(waterfall, current_in_ptr);
        for (SizeType i_level = 1; i_level < levels; ++i_level) {
            const bool is_last = i_level == levels - 1;
            execute_iter(current_in_ptr, current_out_ptr, i_level);
            // Ping-pong buffers (unless it's the final iteration)
            if (!is_last) {
                std::swap(current_in_ptr, current_out_ptr);
            }
        }
        spdlog::debug("FDMTCPU::Impl: Execution complete.");
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

[[nodiscard]] std::tuple<std::vector<float>, plans::FDMTPlan>
compute_fdmt(std::span<const float> waterfall,
             float f_min,
             float f_max,
             SizeType nchans,
             SizeType nsamps,
             float tsamp,
             SizeType dt_max,
             SizeType dt_min,
             bool use_box_smearing,
             std::string_view mode,
             bool verbose,
             int nthreads) {
    FDMTCPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_min,
                 use_box_smearing, mode, verbose, nthreads);
    const plans::FDMTPlan& fdmt_plan = fdmt.get_plan();
    const auto buffer_size           = fdmt_plan.get_buffer_size();
    std::vector<float> dmt(buffer_size, 0.0F);
    fdmt.execute(waterfall, dmt);
    // RESIZE to actual result size
    dmt.resize(fdmt_plan.get_dmt_size());
    return std::make_tuple(std::move(dmt), fdmt_plan);
}

} // namespace dmt::algorithms