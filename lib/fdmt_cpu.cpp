#include "dmt/dmt_types.hpp"
#include <utility>

#ifdef USE_OPENMP
#include <omp.h>
#endif

#include "dmt/dm_utils.hpp"
#include <dmt/fdmt_base.hpp>
#include <dmt/fdmt_cpu.hpp>

FDMTCPU::FDMTCPU(float f_min,
                 float f_max,
                 SizeType nchans,
                 SizeType nsamps,
                 float tsamp,
                 SizeType dt_max,
                 SizeType dt_step,
                 SizeType dt_min)
    : FDMT(f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_step, dt_min) {
    // Allocate memory for the state buffers
    const auto& plan      = get_plan();
    const auto state_size = plan.get_buffer_size();
    m_state_in.resize(state_size, 0.0F);
    m_state_out.resize(state_size, 0.0F);
}

void FDMTCPU::set_num_threads(int nthreads) {
#ifdef USE_OPENMP
    omp_set_num_threads(nthreads);
#endif
}

void FDMTCPU::execute(const float* __restrict waterfall,
                      SizeType waterfall_size,
                      float* __restrict dmt,
                      SizeType dmt_size) {
    check_inputs(waterfall_size, dmt_size);
    float* state_in_ptr  = m_state_in.data();
    float* state_out_ptr = m_state_out.data();

    initialise(waterfall, waterfall_size, state_in_ptr, m_state_in.size());
    const auto niters = get_niters();
    for (SizeType i_iter = 1; i_iter < niters; ++i_iter) {
        execute_iter(state_in_ptr, state_out_ptr, i_iter);
        std::swap(state_in_ptr, state_out_ptr);
    }
    // Last iteration directly writes to the output buffer
    execute_iter(state_in_ptr, dmt, niters);
}

void FDMTCPU::initialise(const float* __restrict waterfall,
                         SizeType /*waterfall_size*/,
                         float* __restrict state,
                         SizeType /*state_size*/) {
    const auto& plan          = get_plan();
    const auto& dt_grids_init = plan.dt_grids[0];
    const auto& nsamps        = plan.state_shape[0][4];
#ifdef USE_OPENMP
#pragma omp parallel for default(none)                                         \
    shared(waterfall, state, dt_grids_init, nsamps)
#endif
    for (SizeType i_sub = 0; i_sub < dt_grids_init.size(); ++i_sub) {
        const auto& dt_grid_sub  = dt_grids_init[i_sub].dt_grid;
        const auto buffer_offset = dt_grids_init[i_sub].sub_offset * nsamps;
        // Initialise state for [:, dt_init_min, dt_init_min:]
        const auto& dt_grid_sub_min = dt_grid_sub[0];
        for (SizeType isamp = dt_grid_sub_min; isamp < nsamps; ++isamp) {
            float sum = 0.0F;
            for (SizeType i = isamp - dt_grid_sub_min; i <= isamp; ++i) {
                sum += waterfall[i_sub * nsamps + i];
            }
            state[buffer_offset + isamp] =
                sum / static_cast<float>(dt_grid_sub_min + 1);
        }
        // Initialise state for [:, dt_grid_init[i_dt], dt_grid_init[i_dt]:]
        for (SizeType i_dt = 1; i_dt < dt_grid_sub.size(); ++i_dt) {
            const auto dt_cur  = dt_grid_sub[i_dt];
            const auto dt_prev = dt_grid_sub[i_dt - 1];
            for (SizeType isamp = dt_cur; isamp < nsamps; ++isamp) {
                float sum = 0.0F;
                for (SizeType i = isamp - dt_cur; i < isamp - dt_prev; ++i) {
                    sum += waterfall[i_sub * nsamps + i];
                }
                state[buffer_offset + i_dt * nsamps + isamp] =
                    (state[buffer_offset + (i_dt - 1) * nsamps + isamp] *
                         (static_cast<float>(dt_prev) + 1.0F) +
                     sum) /
                    (static_cast<float>(dt_cur) + 1.0F);
            }
        }
    }
}

void FDMTCPU::initialise2(const float* __restrict waterfall,
                          SizeType /*waterfall_size*/,
                          float* __restrict state,
                          SizeType /*state_size*/) {
    const auto& plan          = get_plan();
    const auto& dt_grids_init = plan.dt_grids[0];
    const auto& nsamps        = plan.state_shape[0][4];
#ifdef USE_OPENMP
#pragma omp parallel for default(none)                                         \
    shared(waterfall, state, dt_grids_init, nsamps)
#endif
    for (SizeType i_sub = 0; i_sub < dt_grids_init.size(); ++i_sub) {
        const auto& dt_grid_sub  = dt_grids_init[i_sub].dt_grid;
        const auto buffer_offset = dt_grids_init[i_sub].sub_offset * nsamps;
        // Initialise state for [:, dt_init_min, dt_init_min:]
        const auto& dt_grid_sub_min = dt_grid_sub[0];
        for (SizeType isamp = dt_grid_sub_min; isamp < nsamps; ++isamp) {
            float sum = 0.0F;
            for (SizeType i = isamp - dt_grid_sub_min; i <= isamp; ++i) {
                sum += waterfall[i_sub * nsamps + i];
            }
            state[buffer_offset + isamp] = sum;
        }
        // Initialise state for [:, dt_grid_init[i_dt], dt_grid_init[i_dt]:]
        for (SizeType i_dt = 1; i_dt < dt_grid_sub.size(); ++i_dt) {
            const auto dt_cur  = dt_grid_sub[i_dt];
            const auto dt_prev = dt_grid_sub[i_dt - 1];
            for (SizeType isamp = dt_cur; isamp < nsamps; ++isamp) {
                float sum = 0.0F;
                for (SizeType i = isamp - dt_cur; i < isamp - dt_prev; ++i) {
                    sum += waterfall[i_sub * nsamps + i];
                }
                state[buffer_offset + i_dt * nsamps + isamp] =
                    state[buffer_offset + (i_dt - 1) * nsamps + isamp] + sum;
            }
        }
    }
}

void FDMTCPU::execute_iter(const float* __restrict state_in,
                           float* __restrict state_out,
                           SizeType i_iter) {
    const auto& plan            = get_plan();
    const auto& coords_prev     = plan.coordinates[i_iter - 1];
    const auto& coords_sum_cur  = plan.coordinates_to_sum[i_iter];
    const auto& coords_copy_cur = plan.coordinates_to_copy[i_iter];

#pragma omp parallel default(none)                                             \
    shared(state_in, state_out, coords_prev, coords_sum_cur, coords_copy_cur)
    {
#pragma omp for nowait
        for (SizeType i_coord = 0; i_coord < coords_sum_cur.size(); ++i_coord) {
            const auto& coord      = coords_sum_cur[i_coord];
            const auto& coord_tail = coords_prev[coord.i_coord_tail];
            const auto& coord_head = coords_prev[coord.i_coord_head];
            const float* tail      = &state_in[coord_tail.buffer_offset];
            const float* head      = &state_in[coord_head.buffer_offset];
            float* out             = &state_out[coord.buffer_offset];
            dm_utils::add_offset_kernel(tail, coord_tail.nsamps, head,
                                        coord_head.nsamps, out, coord.nsamps,
                                        coord.offset);
        }
#pragma omp for
        for (SizeType i_coord = 0; i_coord < coords_copy_cur.size();
             ++i_coord) {
            const auto& coord      = coords_copy_cur[i_coord];
            const auto& coord_tail = coords_prev[coord.i_coord_tail];
            const float* tail      = &state_in[coord_tail.buffer_offset];
            float* out             = &state_out[coord.buffer_offset];
            std::copy_n(tail, coord_tail.nsamps, out);
        }
    }
}
