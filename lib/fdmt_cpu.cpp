#include <cstddef>
#include <utility>

#ifdef USE_OPENMP
#include <omp.h>
#endif

#include <spdlog/spdlog.h>

#include "dmt/dm_utils.hpp"
#include "dmt/dmt_types.hpp"
#include <dmt/fdmt_cpu.hpp>

FDMTCPU::FDMTCPU(float f_min,
                 float f_max,
                 SizeType nchans,
                 SizeType nsamps,
                 float tsamp,
                 SizeType dt_max,
                 SizeType dt_step,
                 SizeType dt_min,
                 bool use_history)
    : m_use_history(use_history),
      m_plan(f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_step, dt_min) {
    // Allocate memory for the state buffers
    const auto state_size = m_plan.get_buffer_size();
    const auto hist_size  = m_plan.get_history_size();
    m_state_in.resize(state_size, 0.0F);
    m_state_out.resize(state_size, 0.0F);
    m_history.resize(hist_size, 0.0F);
}

void FDMTCPU::set_num_threads(int nthreads) {
#ifdef USE_OPENMP
    omp_set_num_threads(nthreads);
#endif
}

void FDMTCPU::set_log_level(int level) { FDMTPlan::set_log_level(level); }

const FDMTPlan& FDMTCPU::get_plan() const { return m_plan; }

void FDMTCPU::execute(const float* __restrict waterfall,
                      SizeType waterfall_size,
                      float* __restrict dmt,
                      SizeType dmt_size,
                      bool normalize) {
    check_inputs(waterfall_size, dmt_size);
    float* state_in_ptr  = m_state_in.data();
    float* state_out_ptr = m_state_out.data();

    initialise(waterfall, waterfall_size, state_in_ptr, m_state_in.size(),
               normalize);
    const auto niters = m_plan.get_niters();
    for (SizeType i_iter = 1; i_iter < niters; ++i_iter) {
        execute_iter(state_in_ptr, state_out_ptr, i_iter);
        std::swap(state_in_ptr, state_out_ptr);
    }
    // Last iteration directly writes to the output buffer
    execute_iter(state_in_ptr, dmt, niters);
}

template <bool Normalize>
void initialize_impl(const float* __restrict waterfall,
                     SizeType /*waterfall_size*/,
                     float* __restrict state,
                     SizeType /*state_size*/,
                     const std::vector<FDMTCoordGrid>& grids_init,
                     SizeType nsamps,
                     SizeType dt_max,
                     const float* __restrict hist) {

#ifdef USE_OPENMP
#pragma omp parallel for default(none)                                         \
    shared(waterfall, state, grids_init, nsamps, hist, dt_max)
#endif
    for (SizeType i_sub = 0; i_sub < grids_init.size(); ++i_sub) {
        const auto& dt_grid_sub     = grids_init[i_sub].dt_grid;
        const auto buffer_offset    = grids_init[i_sub].coord_offset * nsamps;
        const auto waterfall_offset = i_sub * nsamps;
        const auto hist_offset      = i_sub * dt_max;

        // Initialise state for [:, dt_init_min, dt_init_min:]
        const auto dt_min = dt_grid_sub[0];
        for (SizeType isamp = dt_min; isamp < nsamps; ++isamp) {
            float sum = 0.0F;
            for (SizeType i = isamp - dt_min; i <= isamp; ++i) {
                sum += waterfall[waterfall_offset + i];
            }
            if constexpr (Normalize) {
                state[buffer_offset + isamp] =
                    sum / static_cast<float>(dt_min + 1);
            } else {
                state[buffer_offset + isamp] = sum;
            }
        }
        for (SizeType i_dt = 1; i_dt < dt_grid_sub.size(); ++i_dt) {
            const auto dt_cur            = dt_grid_sub[i_dt];
            const auto dt_prev           = dt_grid_sub[i_dt - 1];
            const auto state_offset_cur  = buffer_offset + i_dt * nsamps;
            const auto state_offset_prev = buffer_offset + (i_dt - 1) * nsamps;

            // Initialise state for [i_sub, i_dt, dt_cur:]
            for (SizeType isamp = dt_cur; isamp < nsamps; ++isamp) {
                float sum = 0.0F;
                for (SizeType i = isamp - dt_cur; i < isamp - dt_prev; ++i) {
                    sum += waterfall[waterfall_offset + i];
                }
                if constexpr (Normalize) {
                    state[state_offset_cur + isamp] =
                        (state[state_offset_prev + isamp] *
                             static_cast<float>(dt_prev + 1) +
                         sum) /
                        static_cast<float>(dt_cur + 1);
                } else {
                    state[state_offset_cur + isamp] =
                        state[state_offset_prev + isamp] + sum;
                }
            }
            // Initialise state for [i_sub, i_dt, 0:dt_cur]
            for (SizeType isamp = 0; isamp < dt_cur; ++isamp) {
                float sum  = 0.0F;
                auto i     = static_cast<std::ptrdiff_t>(isamp - dt_cur);
                auto i_end = static_cast<std::ptrdiff_t>(isamp - dt_prev);
                // Sum from history
                for (; i < 0 && i < i_end; ++i) {
                    sum += hist[hist_offset + (dt_max + i)];
                }
                // Sum from waterfall
                for (; i < i_end; ++i) {
                    sum += waterfall[waterfall_offset + i];
                }
                if constexpr (Normalize) {
                    state[state_offset_cur + isamp] =
                        (state[state_offset_prev + isamp] *
                             static_cast<float>(dt_prev + 1) +
                         sum) /
                        static_cast<float>(dt_cur + 1);
                } else {
                    state[state_offset_cur + isamp] =
                        state[state_offset_prev + isamp] + sum;
                }
            }
        }
    }
}

void FDMTCPU::initialise(const float* __restrict waterfall,
                         SizeType waterfall_size,
                         float* __restrict state,
                         SizeType state_size,
                         bool normalize) {
    const auto& plan_c     = m_plan.get_container();
    const auto& grids_init = plan_c.grids[0];
    const auto nsamps      = plan_c.state_shape[0].nsamps;
    const auto dt_max      = plan_c.state_shape[0].dt_max;
    auto* hist             = m_history.data();

    if (normalize) {
        initialize_impl<true>(waterfall, waterfall_size, state, state_size,
                              grids_init, nsamps, dt_max, hist);
    } else {
        initialize_impl<false>(waterfall, waterfall_size, state, state_size,
                               grids_init, nsamps, dt_max, hist);
    }
    if (m_use_history) {
        // Copy the last nchans x dt_max elements from waterfall to hist
        for (SizeType i_sub = 0; i_sub < grids_init.size(); ++i_sub) {
            std::copy_n(&waterfall[i_sub * nsamps + nsamps - dt_max], dt_max,
                        &hist[i_sub * dt_max]);
        }
    }
}

void FDMTCPU::execute_iter(const float* __restrict state_in,
                           float* __restrict state_out,
                           SizeType i_iter) {
    const auto& plan_c = m_plan.get_container();
    // const auto& coords_prev     = plan_c.coordinates[i_iter - 1];
    const auto& coords_sum_cur  = plan_c.coordinates_sum[i_iter];
    const auto& coords_copy_cur = plan_c.coordinates_copy[i_iter];

#pragma omp parallel default(none)                                             \
    shared(state_in, state_out, coords_sum_cur, coords_copy_cur)
    {
#pragma omp for nowait
        for (SizeType i_coord = 0; i_coord < coords_sum_cur.size(); ++i_coord) {
            const auto& coord = coords_sum_cur[i_coord];
            const float* tail = &state_in[coord.tail_buf_offset];
            const float* head = &state_in[coord.head_buf_offset];
            float* out        = &state_out[coord.buf_offset];
            dm_utils::add_offset_kernel(tail, coord.tail_nsamps, head,
                                        coord.head_nsamps, out, coord.nsamps,
                                        coord.offset);
        }
#pragma omp for
        for (SizeType i_coord = 0; i_coord < coords_copy_cur.size();
             ++i_coord) {
            const auto& coord = coords_copy_cur[i_coord];
            const float* tail = &state_in[coord.tail_buf_offset];
            float* out        = &state_out[coord.buf_offset];
            std::copy_n(tail, coord.tail_nsamps, out);
        }
    }
}

void FDMTCPU::check_inputs(SizeType waterfall_size, SizeType dmt_size) const {
    const auto nchans = m_plan.get_nchans();
    const auto nsamps = m_plan.get_nsamps();
    if (waterfall_size != nchans * nsamps) {
        throw std::invalid_argument("Invalid size of waterfall");
    }
    if (dmt_size != m_plan.get_dmt_size()) {
        throw std::invalid_argument("Invalid size of dmt");
    }
    spdlog::debug("FDMT: Input dimensions: {}x{}", nchans, nsamps);
}