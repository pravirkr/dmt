#include <spdlog/spdlog.h>
#ifdef USE_OPENMP
#include <omp.h>
#endif

#include <dmt/fdmt_cpu.hpp>
#include <dmt/fdmt_utils.hpp>

FDMTCPU::FDMTCPU(float f_min, float f_max, size_t nchans, size_t nsamps,
                 float tsamp, size_t dt_max, size_t dt_step, size_t dt_min)
    : FDMT(f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_step, dt_min) {
    // Allocate memory for the state buffers
    const auto& plan      = get_plan();
    const auto state_size = plan.state_shape[0][3] * plan.state_shape[0][4];
    m_state_in.resize(state_size, 0.0F);
    m_state_out.resize(state_size, 0.0F);
}

void FDMTCPU::set_num_threads(int nthreads) {
#ifdef USE_OPENMP
    omp_set_num_threads(nthreads);
#endif
}

void FDMTCPU::execute(const float* waterfall, size_t waterfall_size, float* dmt,
                      size_t dmt_size) {
    check_inputs(waterfall_size, dmt_size);
    float* state_in_ptr  = m_state_in.data();
    float* state_out_ptr = m_state_out.data();

    initialise(waterfall, state_in_ptr);
    const auto niters = get_niters();
    for (size_t i_iter = 1; i_iter < niters + 1; ++i_iter) {
        execute_iter(state_in_ptr, state_out_ptr, i_iter);
        if (i_iter < niters) {
            std::swap(state_in_ptr, state_out_ptr);
        }
    }
    std::copy_n(state_out_ptr, dmt_size, dmt);
}

void FDMTCPU::initialise(const float* waterfall, float* state) {
    const auto& plan               = get_plan();
    const auto& dt_grid_init       = plan.dt_grid[0];
    const auto& state_sub_idx_init = plan.state_sub_idx[0];
    const auto& nsamps             = plan.state_shape[0][4];
#ifdef USE_OPENMP
#pragma omp parallel for default(none)                                         \
    shared(waterfall, state, dt_grid_init, state_sub_idx_init, nsamps)
#endif
    for (size_t i_sub = 0; i_sub < dt_grid_init.size(); ++i_sub) {
        const auto& dt_grid_sub   = dt_grid_init[i_sub];
        const auto& state_sub_idx = state_sub_idx_init[i_sub];
        // Initialise state for [:, dt_init_min, dt_init_min:]
        const auto& dt_grid_sub_min = dt_grid_sub[0];
        for (size_t isamp = dt_grid_sub_min; isamp < nsamps; ++isamp) {
            float sum = 0.0F;
            for (size_t i = isamp - dt_grid_sub_min; i <= isamp; ++i) {
                sum += waterfall[i_sub * nsamps + i];
            }
            state[state_sub_idx + isamp] =
                sum / static_cast<float>(dt_grid_sub_min + 1);
        }
        // Initialise state for [:, dt_grid_init[i_dt], dt_grid_init[i_dt]:]
        for (size_t i_dt = 1; i_dt < dt_grid_sub.size(); ++i_dt) {
            const auto dt_cur  = dt_grid_sub[i_dt];
            const auto dt_prev = dt_grid_sub[i_dt - 1];
            for (size_t isamp = dt_cur; isamp < nsamps; ++isamp) {
                float sum = 0.0F;
                for (size_t i = isamp - dt_cur; i < isamp - dt_prev; ++i) {
                    sum += waterfall[i_sub * nsamps + i];
                }
                state[state_sub_idx + i_dt * nsamps + isamp] =
                    (state[state_sub_idx + (i_dt - 1) * nsamps + isamp] *
                         (static_cast<float>(dt_prev) + 1.0F) +
                     sum) /
                    (static_cast<float>(dt_cur) + 1.0F);
            }
        }
    }

    const auto& [nchans_l, ndt_min, ndt_max, nchans_ndt, nsamps_l] =
        plan.state_shape[0];
    spdlog::debug("FDMT: Iteration {}, dimensions: {} ({}x[{}..{}]) x {}", 0,
                  nchans_ndt, nchans_l, ndt_min, ndt_max, nsamps_l);
}

void FDMTCPU::execute_iter(const float* state_in, float* state_out,
                           size_t i_iter) {
    const auto& plan               = get_plan();
    const auto& nsamps             = plan.state_shape[i_iter][4];
    const auto& coords_cur         = plan.coordinates[i_iter];
    const auto& mappings_cur       = plan.mappings[i_iter];
    const auto& coords_copy_cur    = plan.coordinates_to_copy[i_iter];
    const auto& mappings_copy_cur  = plan.mappings_to_copy[i_iter];
    const auto& state_sub_idx_cur  = plan.state_sub_idx[i_iter];
    const auto& state_sub_idx_prev = plan.state_sub_idx[i_iter - 1];

#ifdef USE_OPENMP
#pragma omp parallel for default(none)                                         \
    shared(state_in, state_out, coords_cur, mappings_cur, state_sub_idx_cur,   \
               state_sub_idx_prev, nsamps)
#endif
    for (size_t i_coord = 0; i_coord < coords_cur.size(); ++i_coord) {
        const auto& i_sub              = coords_cur[i_coord].first;
        const auto& i_dt               = coords_cur[i_coord].second;
        const auto& i_sub_tail         = mappings_cur[i_coord].tail.first;
        const auto& i_dt_tail          = mappings_cur[i_coord].tail.second;
        const auto& i_sub_head         = mappings_cur[i_coord].head.first;
        const auto& i_dt_head          = mappings_cur[i_coord].head.second;
        const auto& offset             = mappings_cur[i_coord].offset;
        const auto& state_sub_idx      = state_sub_idx_cur[i_sub];
        const auto& state_sub_idx_tail = state_sub_idx_prev[i_sub_tail];
        const auto& state_sub_idx_head = state_sub_idx_prev[i_sub_head];

        const float* tail = &state_in[state_sub_idx_tail + i_dt_tail * nsamps];
        const float* head = &state_in[state_sub_idx_head + i_dt_head * nsamps];
        float* out        = &state_out[state_sub_idx + i_dt * nsamps];
        fdmt::add_offset_kernel(tail, nsamps, head, nsamps, out, nsamps,
                                offset);
    }
#ifdef USE_OPENMP
#pragma omp parallel for default(none)                                         \
    shared(state_in, state_out, coords_copy_cur, mappings_copy_cur,            \
               state_sub_idx_cur, state_sub_idx_prev, nsamps)
#endif
    for (size_t i_coord = 0; i_coord < coords_copy_cur.size(); ++i_coord) {
        const auto& i_sub              = coords_copy_cur[i_coord].first;
        const auto& i_dt               = coords_copy_cur[i_coord].second;
        const auto& i_sub_tail         = mappings_copy_cur[i_coord].tail.first;
        const auto& i_dt_tail          = mappings_copy_cur[i_coord].tail.second;
        const auto& state_sub_idx      = state_sub_idx_cur[i_sub];
        const auto& state_sub_idx_tail = state_sub_idx_prev[i_sub_tail];

        const float* tail = &state_in[state_sub_idx_tail + i_dt_tail * nsamps];
        float* out        = &state_out[state_sub_idx + i_dt * nsamps];
        fdmt::copy_kernel(tail, nsamps, out, nsamps);
    }

    const auto& [nchans_l, ndt_min, ndt_max, nchans_ndt, nsamps_l] =
        plan.state_shape[i_iter];
    spdlog::debug("FDMT: Iteration {}, dimensions: {} ({}x[{}..{}]) x {}",
                  i_iter, nchans_ndt, nchans_l, ndt_min, ndt_max, nsamps_l);
}
