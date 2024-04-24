#include <cmath>
#include <cstdint>
#include <spdlog/spdlog.h>

#include <dmt/fdmt.hpp>
#include "dmt/fdmt_utils.hpp"

FDMT::FDMT(float f_min, float f_max, size_t nchans, size_t nsamps, float tsamp,
           size_t dt_max, size_t dt_step, size_t dt_min)
    : f_min(f_min),
      f_max(f_max),
      nchans(nchans),
      nsamps(nsamps),
      tsamp(tsamp),
      dt_max(dt_max),
      dt_step(dt_step),
      dt_min(dt_min),
      df(calculate_df(f_min, f_max, nchans)),
      correction(df / 2),
      niters(calculate_niters(nchans)) {
    configure_fdmt_plan();
    spdlog::info("FDMT: df={}, dt_max={}, ndt_init={}, niters={}", df, dt_max,
                 get_dt_grid_init().size(), niters);
    dm_arr
        = fdmt::calculate_dm_arr(f_min, f_max, tsamp, dt_max, dt_step, dt_min);
    // Allocate memory for the state buffers
    state_in.resize(nchans * get_dt_grid_init().size() * nsamps, 0.0F);
    state_out.resize(nchans * get_dt_grid_init().size() * nsamps, 0.0F);
}

// Getters
float FDMT::get_df() const { return df; }
float FDMT::get_correction() const { return correction; }
DtGrid FDMT::get_dt_grid_init() const {
    return fdmt_plan.sub_plan[0][0].dt_grid;
}
DtGrid FDMT::get_dt_grid_final() const {
    return calculate_dt_grid_sub(f_min, f_max);
}
size_t FDMT::get_niters() const { return niters; }
FDMTPlan FDMT::get_plan() const { return fdmt_plan; }
std::vector<float> FDMT::get_dm_arr() const { return dm_arr; }

size_t FDMT::calculate_niters(size_t nchans) {
    return static_cast<size_t>(std::ceil(std::log2(nchans)));
}
float FDMT::calculate_df(float f_min, float f_max, size_t nchans) {
    return (f_max - f_min) / static_cast<float>(nchans);
}
size_t FDMT::calculate_dt_sub(float f_start, float f_end, size_t dt) const {
    return fdmt::calculate_dt_sub(f_start, f_end, f_min, f_max, dt);
}

std::vector<size_t> FDMT::calculate_dt_grid_sub(float f_start,
                                                float f_end) const {
    return fdmt::calculate_dt_grid_sub(f_start, f_end, f_min, f_max, dt_max,
                                       dt_step, dt_min);
}

void FDMT::execute(const float* waterfall, size_t waterfall_size, float* dmt,
                   size_t dmt_size) {
    check_inputs(waterfall_size, dmt_size);
    float* state_in_ptr  = state_in.data();
    float* state_out_ptr = state_out.data();

    initialise(waterfall, state_in_ptr);
    for (size_t i_iter = 1; i_iter < niters + 1; ++i_iter) {
        execute_iter(state_in_ptr, state_out_ptr, i_iter);
        if (i_iter < niters) {
            std::swap(state_in_ptr, state_out_ptr);
        }
    }
    std::copy_n(state_out_ptr, dmt_size, dmt);
}

void FDMT::initialise(const float* waterfall, float* state) {
    const auto& dt_grid_init = get_dt_grid_init();
    const size_t ndt         = dt_grid_init.size();
    const auto dt_init_min   = dt_grid_init[0];
    // Initialise state for [:, dt_init_min, dt_init_min:]
#pragma omp parallel for
    for (size_t ichan = 0; ichan < nchans; ++ichan) {
        for (size_t isamp = dt_init_min; isamp < nsamps; ++isamp) {
            float sum = 0.0F;
            for (size_t i = isamp - dt_init_min; i <= isamp; ++i) {
                sum += waterfall[ichan * nsamps + i];
            }
            state[ichan * ndt * nsamps + isamp]
                = sum / static_cast<float>(dt_init_min + 1);
        }
    }

    // Initialise state for [:, dt_grid_init[i_dt], dt_grid_init[i_dt]:]
    for (size_t i_dt = 1; i_dt < ndt; ++i_dt) {
        const auto dt_init_cur  = dt_grid_init[i_dt];
        const auto dt_init_prev = dt_grid_init[i_dt - 1];
#pragma omp parallel for
        for (size_t ichan = 0; ichan < nchans; ++ichan) {
            for (size_t isamp = dt_init_cur; isamp < nsamps; ++isamp) {
                float sum = 0.0F;
                for (size_t i = isamp - dt_init_cur; i < isamp - dt_init_prev;
                     ++i) {
                    sum += waterfall[ichan * nsamps + i];
                }
                state[ichan * ndt * nsamps + i_dt * nsamps + isamp]
                    = (state[ichan * ndt * nsamps + (i_dt - 1) * nsamps + isamp]
                           * (static_cast<float>(dt_init_prev) + 1.0F)
                       + sum)
                      / (static_cast<float>(dt_init_cur) + 1.0F);
            }
        }
    }

    spdlog::info("FDMT: initialised dimensions: {}x{}x{}", nchans, ndt, nsamps);
}

// Private methods
void FDMT::check_inputs(size_t waterfall_size, size_t dmt_size) const {
    if (waterfall_size != nchans * nsamps) {
        throw std::invalid_argument("Invalid size of waterfall");
    }
    const auto& [nchans_final, dt_final, nsamps_final]
        = fdmt_plan.state_shape[niters];
    if (dmt_size != nchans_final * dt_final * nsamps_final) {
        throw std::invalid_argument("Invalid size of dmt");
    }
    spdlog::info("FDMT: Input dimensions: {}x{}", nchans, nsamps);
}

void FDMT::execute_iter(const float* state_in, float* state_out,
                        size_t i_iter) {
    const auto& [nchans_cur, ndt_cur, nsamps_cur]
        = fdmt_plan.state_shape[i_iter];
    const auto& [nchans_prev, ndt_prev, nsamps_prev]
        = fdmt_plan.state_shape[i_iter - 1];
    spdlog::info("FDMT: Iteration {}, dimensions: {}x{}x{}", i_iter, nchans_cur,
                 ndt_cur, nsamps_cur);
    for (size_t i_sub = 0; i_sub < nchans_cur; ++i_sub) {
        const auto& dt_grid_sub = fdmt_plan.sub_plan[i_iter][i_sub].dt_grid;
        for (size_t i_dt = 0; i_dt < dt_grid_sub.size(); ++i_dt) {
            const auto& [i_dt_out, offset, i_dt_tail, i_dt_head]
                = fdmt_plan.sub_plan[i_iter][i_sub].dt_plan[i_dt];
            const float* tail = &state_in[(2 * i_sub) * ndt_prev * nsamps_prev
                                          + i_dt_tail * nsamps_prev];
            float* out        = &state_out[i_sub * ndt_cur * nsamps_cur
                                    + i_dt_out * nsamps_cur];
            if (i_dt_head == SIZE_MAX) {
                fdmt::copy_kernel(tail, nsamps_prev, out, nsamps_cur);
            } else {
                const float* head
                    = &state_in[(2 * i_sub + 1) * ndt_prev * nsamps_prev
                                + i_dt_head * nsamps_prev];
                fdmt::add_offset_kernel(tail, nsamps_prev, head, nsamps_prev, out,
                                        nsamps_cur, offset);
            }
        }
    }
}

void FDMT::configure_fdmt_plan() {
    // Allocate memory/size for plan members
    fdmt_plan.df_top.resize(niters + 1);
    fdmt_plan.df_bot.resize(niters + 1);
    fdmt_plan.dt_grid_sub_top.resize(niters + 1);
    fdmt_plan.state_shape.resize(niters + 1);
    fdmt_plan.sub_plan.resize(niters + 1);
    // For iteration 0
    make_fdmt_plan_iter0();
    // For iterations 1 to niters
    for (size_t i_iter = 1; i_iter < niters + 1; ++i_iter) {
        make_fdmt_plan(i_iter);
    }
    spdlog::info("FDMT: configured fdmt plan");
}

void FDMT::make_fdmt_plan_iter0() {
    // For iteration 0
    fdmt_plan.df_top[0] = df;
    fdmt_plan.df_bot[0] = df;
    fdmt_plan.sub_plan[0].resize(nchans);
    for (size_t i_sub = 0; i_sub < nchans; ++i_sub) {
        float f_start, f_end, f_mid, f_mid1, f_mid2;
        DtGrid dt_grid_sub;
        f_start     = df * static_cast<float>(i_sub) + f_min;
        f_end       = f_start + df;
        f_mid       = f_start + df / 2;
        f_mid1      = f_mid - correction;
        f_mid2      = f_mid + correction;
        dt_grid_sub = calculate_dt_grid_sub(f_start, f_end);
        fdmt_plan.sub_plan[0][i_sub]
            = {f_start, f_end, f_mid1, f_mid2, dt_grid_sub, {}};
    }
    fdmt_plan.dt_grid_sub_top[0] = fdmt_plan.sub_plan[0][nchans - 1].dt_grid;
    fdmt_plan.state_shape[0]     = std::make_tuple(
        nchans, fdmt_plan.sub_plan[0][0].dt_grid.size(), nsamps);
}

void FDMT::make_fdmt_plan(size_t i_iter) {
    if (i_iter < 1 || i_iter > niters) {
        throw std::invalid_argument("Invalid iteration number");
    }
    const float& df_bot_prev  = fdmt_plan.df_bot[i_iter - 1];
    const float& df_top_prev  = fdmt_plan.df_top[i_iter - 1];
    const size_t& nchans_prev = std::get<0>(fdmt_plan.state_shape[i_iter - 1]);
    const DtGrid& dt_grid_sub_top_prev = fdmt_plan.dt_grid_sub_top[i_iter - 1];

    const size_t nchans_cur = nchans_prev / 2 + nchans_prev % 2;
    const bool do_copy = nchans_prev % 2 == 1;  // true if nchans_prev is odd
    // Update df_top, df_bot, ndt_top
    float df_top = (do_copy) ? df_top_prev : df_top_prev + df_bot_prev;
    float df_bot = df_bot_prev * 2;
    // Take care of max_dt for the last iteration
    DtGrid dt_grid_sub_top = dt_grid_sub_top_prev;
    float f_max_sub_tail = (nchans_cur == 1) ? f_min + df_top : f_min + df_bot;
    // Calculate the dt grid for the tail sub-band of the current iteration
    // which will be used to decide the uniform shape of the output array
    const DtGrid& dt_grid_sub_tail
        = calculate_dt_grid_sub(f_min, f_max_sub_tail);

    fdmt_plan.sub_plan[i_iter].resize(nchans_cur);
    for (size_t i_sub = 0; i_sub < nchans_cur; ++i_sub) {
        float f_start, f_end, f_mid, f_mid1, f_mid2;
        DtGrid dt_grid_sub;
        const auto& dt_grid_sub_tail_prev
            = fdmt_plan.sub_plan[i_iter - 1][2 * i_sub].dt_grid;
        const auto& dt_grid_sub_head_prev
            = fdmt_plan.sub_plan[i_iter - 1][2 * i_sub + 1].dt_grid;
        f_start = df_bot * static_cast<float>(i_sub) + f_min;
        if (i_sub == nchans_cur - 1) {
            // For the top sub-band
            if (do_copy) {
                f_end       = f_start + df_top * 2;
                f_mid       = f_start + df_top;
                dt_grid_sub = dt_grid_sub_top;
            } else {
                f_end           = f_start + df_top;
                f_mid           = f_start + df_bot / 2;
                dt_grid_sub     = calculate_dt_grid_sub(f_start, f_end);
                dt_grid_sub_top = dt_grid_sub;
            }
        } else {
            // For the bottom sub-bands
            f_end       = f_start + df_bot;
            f_mid       = f_start + df_bot / 2;
            dt_grid_sub = calculate_dt_grid_sub(f_start, f_end);
        }
        f_mid1 = f_mid - correction;
        f_mid2 = f_mid + correction;

        fdmt_plan.df_top[i_iter]          = df_top;
        fdmt_plan.df_bot[i_iter]          = df_bot;
        fdmt_plan.dt_grid_sub_top[i_iter] = dt_grid_sub_top;
        fdmt_plan.state_shape[i_iter]
            = std::make_tuple(nchans_cur, dt_grid_sub_tail.size(), nsamps);

        // Populate the dt_plan mapping current dt grid to the previous dt grid
        std::vector<DtPlan> dt_plan_sub(dt_grid_sub.size());
        for (size_t i_dt = 0; i_dt < dt_grid_sub.size(); ++i_dt) {
            const auto& dt = dt_grid_sub[i_dt];
            // dt ~= dt_tail (dt_mid) + dt_head
            size_t dt_mid1, dt_mid2, dt_head;
            dt_mid1 = static_cast<size_t>(
                std::round(static_cast<float>(dt)
                           * fdmt::cff(f_start, f_mid1, f_start, f_end)));
            dt_mid2 = static_cast<size_t>(
                std::round(static_cast<float>(dt)
                           * fdmt::cff(f_start, f_mid2, f_start, f_end)));
            // check dt_head is always >= 0, otherwise throw error
            if (dt_mid1 > dt || dt_mid2 > dt) {
                throw std::runtime_error("Invalid dt_mid values");
            }
            dt_head = dt - dt_mid2;

            size_t i_dt_head, i_dt_tail, offset;
            if (i_sub == nchans_cur - 1 && do_copy) {
                i_dt_head = SIZE_MAX;
                i_dt_tail = fdmt::find_closest_index(dt_grid_sub_tail_prev, dt);
                offset    = 0;
            } else {
                i_dt_head
                    = fdmt::find_closest_index(dt_grid_sub_head_prev, dt_head);
                i_dt_tail
                    = fdmt::find_closest_index(dt_grid_sub_tail_prev, dt_mid1);
                offset = dt_mid2;
            }
            dt_plan_sub[i_dt]
                = std::make_tuple(i_dt, offset, i_dt_tail, i_dt_head);
        }
        fdmt_plan.sub_plan[i_iter][i_sub]
            = {f_start, f_end, f_mid1, f_mid2, dt_grid_sub, dt_plan_sub};
    }
}
